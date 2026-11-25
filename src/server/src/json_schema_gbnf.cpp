#include "json_schema_gbnf.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference::server::detail {

namespace {

constexpr std::size_t kMaxSchemaDepth = 32;
constexpr std::size_t kMaxGrammarBytes = 256u * 1024u;
constexpr std::size_t kMaxRules = 1024;
constexpr std::size_t kMaxUnrolledBounds = 32;
constexpr std::size_t kMaxSchemaNodes = 2048;
constexpr std::size_t kMaxPatternBytes = 512;
constexpr std::size_t kMaxPatternAtoms = 256;
constexpr std::size_t kMaxPatternGroupDepth = 8;
constexpr std::int64_t kMaxBoundedIntegerAbs = 999999;

bool is_supported_schema_keyword(std::string_view key) noexcept {
    // Functional keywords enforced by the converter:
    if (key == "$ref" || key == "$defs" || key == "definitions" || key == "type" || key == "const" || key == "enum" ||
        key == "anyOf" || key == "oneOf" || key == "properties" || key == "required" ||
        key == "additionalProperties" || key == "items" || key == "minLength" || key == "maxLength" ||
        key == "minItems" || key == "maxItems" || key == "pattern" || key == "minimum" || key == "maximum") {
        return true;
    }
    // Non-validating JSON Schema annotation/metadata keywords:
    if (key == "$schema" || key == "$id" || key == "$anchor" || key == "$comment" || key == "title" ||
        key == "description" || key == "default" || key == "examples" || key == "readOnly" || key == "writeOnly" ||
        key == "deprecated") {
        return true;
    }
    return false;
}

const char* const kCommonRules =
    "space ::= | \" \" | \"\\n\" [ \\t]{0,20}\n"
    "string ::= \"\\\"\" ([^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}))* \"\\\"\" space\n"
    "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n"
    "integer ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) space\n"
    "number ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)? space\n"
    "boolean ::= (\"true\" | \"false\") space\n"
    "null ::= \"null\" space\n"
    "value ::= object | array | string | number | boolean | null\n"
    "object ::= \"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space\n"
    "array ::= \"[\" space ( value (\",\" space value)* )? \"]\" space\n";

// Escapes raw bytes as a GBNF double-quoted string literal (including quotes).
std::string gbnf_literal(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 4);
    out.push_back('"');
    for (const char ch : text) {
        const auto u = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (u < 0x20 || u == 0x7F) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\x";
                    out.push_back(kHex[(u >> 4) & 0x0F]);
                    out.push_back(kHex[u & 0x0F]);
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

std::string sanitize_rule_name(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (const char ch : raw) {
        const auto u = static_cast<unsigned char>(ch);
        if (std::isalnum(u) != 0) {
            out.push_back(static_cast<char>(std::tolower(u)));
        } else if (ch == '-' || ch == '_') {
            out.push_back('-');
        } else {
            out.push_back('-');
        }
        if (out.size() >= 40) {
            break;
        }
    }
    while (!out.empty() && out.front() == '-') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    if (out.empty() || !std::isalpha(static_cast<unsigned char>(out.front()))) {
        out = "r-" + out;
    }
    return out;
}

// ================================================================= Canonical JSON value equality & type masks

enum JsonTypeBit : std::uint32_t {
    kTypeNull = 1u << 0,
    kTypeBool = 1u << 1,
    kTypeInt = 1u << 2,
    kTypeNum = 1u << 3,
    kTypeStr = 1u << 4,
    kTypeArr = 1u << 5,
    kTypeObj = 1u << 6,
    kTypeAll = 0x7Fu,
};

bool type_masks_overlap(std::uint32_t a, std::uint32_t b) noexcept {
    if ((a & b) != 0u) {
        return true;
    }
    // JSON Schema "integer" is a strict subset of "number", so they overlap.
    if (((a & kTypeInt) != 0u && (b & kTypeNum) != 0u) || ((a & kTypeNum) != 0u && (b & kTypeInt) != 0u)) {
        return true;
    }
    return false;
}

std::uint32_t type_bit_for_name(std::string_view name) noexcept {
    if (name == "null") return kTypeNull;
    if (name == "boolean") return kTypeBool;
    if (name == "integer") return kTypeInt;
    if (name == "number") return kTypeNum;
    if (name == "string") return kTypeStr;
    if (name == "array") return kTypeArr;
    if (name == "object") return kTypeObj;
    return 0u;
}

std::uint32_t type_bit_for_value(const json::Value& v) noexcept {
    if (v.is_null()) return kTypeNull;
    if (v.is_bool()) return kTypeBool;
    if (v.is_integer()) return kTypeInt;
    if (v.is_number()) return kTypeNum;
    if (v.is_string()) return kTypeStr;
    if (v.is_array()) return kTypeArr;
    if (v.is_object()) return kTypeObj;
    return kTypeAll;
}

bool json_values_equal(const json::Value& a, const json::Value& b) {
    if (a.is_null() && b.is_null()) return true;
    if (a.is_bool() && b.is_bool()) return a.as_bool() == b.as_bool();
    if (a.is_number() && b.is_number()) {
        if (a.is_integer() && b.is_integer()) {
            // Exact integer comparison where representable in int64/uint64.
            const double da = a.as_double();
            const double db = b.as_double();
            if (da < 0.0 && db < 0.0) return a.as_int() == b.as_int();
            if (da >= 0.0 && db >= 0.0) return a.as_uint() == b.as_uint();
            return false;
        }
        return a.as_double() == b.as_double();
    }
    if (a.is_string() && b.is_string()) return a.as_string() == b.as_string();
    if (a.is_array() && b.is_array()) {
        const auto& aa = a.as_array();
        const auto& ba = b.as_array();
        if (aa.size() != ba.size()) return false;
        for (std::size_t i = 0; i < aa.size(); ++i) {
            if (!json_values_equal(aa[i], ba[i])) return false;
        }
        return true;
    }
    if (a.is_object() && b.is_object()) {
        const auto& ao = a.as_object();
        const auto& bo = b.as_object();
        if (ao.size() != bo.size()) return false;
        for (const auto& [k, av] : ao) {
            const json::Value* bv = bo.find(k);
            if (bv == nullptr || !json_values_equal(av, *bv)) return false;
        }
        return true;
    }
    return false;
}

bool literal_sets_overlap(const std::vector<json::Value>& a, const std::vector<json::Value>& b) {
    for (const auto& va : a) {
        for (const auto& vb : b) {
            if (json_values_equal(va, vb)) {
                return true;
            }
        }
    }
    return false;
}

// ================================================================= Bounded integer range -> GBNF

std::string gbnf_digit_range(char lo, char hi) {
    if (lo == hi) {
        return gbnf_literal(std::string_view(&lo, 1));
    }
    std::string out = "[";
    out.push_back(lo);
    out.push_back('-');
    out.push_back(hi);
    out.push_back(']');
    return out;
}

std::string gbnf_any_digits(std::size_t count) {
    if (count == 0) return {};
    if (count == 1) return "[0-9]";
    return "[0-9]{" + std::to_string(count) + "}";
}

// Emits exact GBNF alternatives for all decimal integers in [lo, hi] where
// lo and hi are non-negative and have the exact same decimal digit length.
void emit_same_length_int_range(std::int64_t lo, std::int64_t hi, std::vector<std::string>& out) {
    const std::string s_lo = std::to_string(lo);
    const std::string s_hi = std::to_string(hi);
    const std::size_t len = s_lo.size();

    std::size_t prefix_len = 0;
    while (prefix_len < len && s_lo[prefix_len] == s_hi[prefix_len]) {
        ++prefix_len;
    }
    if (prefix_len == len) {
        out.push_back(gbnf_literal(s_lo));
        return;
    }

    auto format_prefix = [&]() -> std::string {
        return prefix_len > 0 ? gbnf_literal(std::string_view(s_lo).substr(0, prefix_len)) : std::string();
    };

    const bool tail_lo_zero =
        std::all_of(s_lo.begin() + static_cast<std::ptrdiff_t>(prefix_len), s_lo.end(), [](char c) { return c == '0'; });
    const bool tail_hi_nine =
        std::all_of(s_hi.begin() + static_cast<std::ptrdiff_t>(prefix_len), s_hi.end(), [](char c) { return c == '9'; });
    if (tail_lo_zero && tail_hi_nine) {
        std::string item = format_prefix();
        if (!item.empty()) item += " ";
        item += gbnf_any_digits(len - prefix_len);
        out.push_back(std::move(item));
        return;
    }
    if (prefix_len + 1 == len) {
        std::string item = format_prefix();
        if (!item.empty()) item += " ";
        item += gbnf_digit_range(s_lo[prefix_len], s_hi[prefix_len]);
        out.push_back(std::move(item));
        return;
    }

    std::int64_t step = 1;
    for (std::size_t i = prefix_len + 1; i < len; ++i) {
        step *= 10;
    }

    const bool lo_sub_zero = std::all_of(s_lo.begin() + static_cast<std::ptrdiff_t>(prefix_len + 1), s_lo.end(),
                                         [](char c) { return c == '0'; });
    const bool hi_sub_nine = std::all_of(s_hi.begin() + static_cast<std::ptrdiff_t>(prefix_len + 1), s_hi.end(),
                                         [](char c) { return c == '9'; });

    char mid_lo = s_lo[prefix_len];
    char mid_hi = s_hi[prefix_len];
    if (!lo_sub_zero) {
        const std::int64_t lo_block_end = ((lo / step) + 1) * step - 1;
        emit_same_length_int_range(lo, lo_block_end, out);
        ++mid_lo;
    }
    std::optional<std::pair<std::int64_t, std::int64_t>> top_piece;
    if (!hi_sub_nine) {
        const std::int64_t hi_block_start = (hi / step) * step;
        top_piece = std::make_pair(hi_block_start, hi);
        --mid_hi;
    }
    if (mid_lo <= mid_hi) {
        std::string item = format_prefix();
        if (!item.empty()) item += " ";
        item += gbnf_digit_range(mid_lo, mid_hi);
        const std::size_t rem = len - prefix_len - 1;
        if (rem > 0) {
            item += " " + gbnf_any_digits(rem);
        }
        out.push_back(std::move(item));
    }
    if (top_piece) {
        emit_same_length_int_range(top_piece->first, top_piece->second, out);
    }
}

void emit_nonneg_int_range(std::int64_t lo, std::int64_t hi, std::vector<std::string>& out) {
    std::int64_t cur = lo;
    std::int64_t pow10 = 1;
    while (pow10 * 10 <= cur) {
        pow10 *= 10;
    }
    while (cur <= hi) {
        const std::int64_t bucket_hi = std::min(hi, pow10 * 10 - 1);
        emit_same_length_int_range(cur, bucket_hi, out);
        if (bucket_hi == hi) {
            break;
        }
        cur = bucket_hi + 1;
        pow10 *= 10;
    }
}

std::string bounded_integer_gbnf(std::int64_t lo, std::int64_t hi) {
    std::vector<std::string> branches;
    if (lo < 0) {
        const std::int64_t neg_lo = hi < 0 ? -hi : 1;
        const std::int64_t neg_hi = -lo;
        std::vector<std::string> neg_parts;
        emit_nonneg_int_range(neg_lo, neg_hi, neg_parts);
        for (const auto& p : neg_parts) {
            branches.push_back("\"-\" " + p);
        }
    }
    if (hi >= 0) {
        const std::int64_t pos_lo = lo > 0 ? lo : 0;
        emit_nonneg_int_range(pos_lo, hi, branches);
    }
    std::string expr = "(";
    for (std::size_t i = 0; i < branches.size(); ++i) {
        if (i > 0) expr += " | ";
        expr += "(" + branches[i] + ")";
    }
    expr += ") space";
    return expr;
}

// ================================================================= Safe regex pattern subset -> GBNF

class RegexToGbnfParser {
public:
    explicit RegexToGbnfParser(std::string_view pat) : pat_(pat) {}

    Result<std::string> convert() {
        if (pat_.size() > kMaxPatternBytes) {
            return Status(ErrorCode::invalid_argument,
                          "pattern exceeds maximum supported length of " + std::to_string(kMaxPatternBytes) + " bytes");
        }
        for (const char ch : pat_) {
            const auto u = static_cast<unsigned char>(ch);
            if (u < 0x20u || u == 0x7Fu) {
                return Status(ErrorCode::invalid_argument, "pattern must not contain control characters");
            }
        }
        if (pat_.empty()) {
            return std::string("string");
        }

        bool anchored_start = false;
        bool anchored_end = false;
        std::string_view body = pat_;
        if (!body.empty() && body.front() == '^') {
            anchored_start = true;
            body.remove_prefix(1);
        }
        if (!body.empty() && body.back() == '$') {
            // Check whether trailing '$' is escaped by an odd run of backslashes.
            std::size_t slashes = 0;
            for (std::size_t i = body.size() - 1; i > 0 && body[i - 1] == '\\'; --i) {
                ++slashes;
            }
            if ((slashes % 2u) == 0u) {
                anchored_end = true;
                body.remove_suffix(1);
            }
        }
        pat_ = body;
        pos_ = 0;

        auto expr = parse_alternation(0);
        if (!expr.ok()) {
            return expr.status();
        }
        if (pos_ != pat_.size()) {
            return Status(ErrorCode::invalid_argument, "unexpected trailing character in regex pattern");
        }

        std::string full;
        if (!anchored_start) full += "char* ";
        if (!expr.value().empty()) {
            full += "(" + expr.value() + ")";
        }
        if (!anchored_end) {
            if (!full.empty()) full += " ";
            full += "char*";
        }
        if (full.empty()) {
            return std::string("\"\\\"\\\"\" space");
        }
        return "\"\\\"\" " + full + " \"\\\"\" space";
    }

private:
    static std::string escape_gbnf_class_char(char ch) {
        if (ch == ']' || ch == '\\' || ch == '^' || ch == '-') {
            std::string s = "\\";
            s.push_back(ch);
            return s;
        }
        return std::string(1, ch);
    }

    Result<std::string> parse_alternation(std::size_t depth) {
        auto first = parse_sequence(depth);
        if (!first.ok()) return first.status();
        std::string out = std::move(first).value();
        while (pos_ < pat_.size() && pat_[pos_] == '|') {
            ++pos_;
            auto next = parse_sequence(depth);
            if (!next.ok()) return next.status();
            out += " | " + next.value();
        }
        return out;
    }

    Result<std::string> parse_sequence(std::size_t depth) {
        std::string seq;
        while (pos_ < pat_.size() && pat_[pos_] != ')' && pat_[pos_] != '|') {
            auto atom = parse_atom(depth);
            if (!atom.ok()) return atom.status();
            auto quant = parse_quantifier();
            if (!quant.ok()) return quant.status();
            if (!seq.empty()) seq.push_back(' ');
            seq += atom.value();
            seq += quant.value();
        }
        return seq;
    }

    Result<std::string> parse_atom(std::size_t depth) {
        if (++atoms_ > kMaxPatternAtoms) {
            return Status(ErrorCode::invalid_argument, "regex pattern exceeds maximum complexity");
        }
        const char ch = pat_[pos_];
        if (ch == '^' || ch == '$') {
            return Status(ErrorCode::invalid_argument,
                          "mid-pattern anchors (^/$) are not supported in GBNF pattern conversion");
        }
        if (ch == '*' || ch == '+' || ch == '?' || ch == '{') {
            return Status(ErrorCode::invalid_argument, "dangling quantifier in regex pattern");
        }
        if (ch == '(') {
            if (depth >= kMaxPatternGroupDepth) {
                return Status(ErrorCode::invalid_argument, "regex pattern group nesting too deep");
            }
            ++pos_;
            if (pos_ < pat_.size() && pat_[pos_] == '?') {
                // Support plain non-capturing (?:...) only; reject lookarounds/flags.
                if (pos_ + 1 < pat_.size() && pat_[pos_ + 1] == ':') {
                    pos_ += 2;
                } else {
                    return Status(ErrorCode::invalid_argument,
                                  "unsupported regex construct '(?...)' (lookarounds/flags are not supported in GBNF)");
                }
            }
            auto inner = parse_alternation(depth + 1);
            if (!inner.ok()) return inner.status();
            if (pos_ >= pat_.size() || pat_[pos_] != ')') {
                return Status(ErrorCode::invalid_argument, "unclosed '(' in regex pattern");
            }
            ++pos_;
            return "(" + inner.value() + ")";
        }
        if (ch == '[') {
            return parse_char_class();
        }
        if (ch == '.') {
            ++pos_;
            return std::string("[^\"\\\\\\x7F\\x00-\\x1F]");
        }
        if (ch == '\\') {
            return parse_escape();
        }
        ++pos_;
        if (ch == '"') {
            return std::string("\"\\\\\\\"\"");
        }
        return gbnf_literal(std::string_view(&ch, 1));
    }

    Result<std::string> parse_escape() {
        ++pos_;
        if (pos_ >= pat_.size()) {
            return Status(ErrorCode::invalid_argument, "trailing backslash in regex pattern");
        }
        const char e = pat_[pos_++];
        if (e >= '1' && e <= '9') {
            return Status(ErrorCode::invalid_argument,
                          "regex backreferences (\\1-\\9) are not supported in GBNF");
        }
        if (e == 'd') return std::string("[0-9]");
        if (e == 'w') return std::string("[a-zA-Z0-9_]");
        if (e == 's') return std::string("[ \\t]");
        if (e == '\\') return std::string("\"\\\\\\\\\"");
        if (e == '"') return std::string("\"\\\\\\\"\"");
        switch (e) {
            case '.': case '-': case '_': case '/': case '+': case '*': case '?':
            case '(': case ')': case '[': case ']': case '{': case '}': case '|':
            case '^': case '$': case ':': case '@': case ',': case ';': case '!':
            case '#': case '%': case '&': case '=': case '<': case '>': case '~':
                return gbnf_literal(std::string_view(&e, 1));
            default:
                return Status(ErrorCode::invalid_argument,
                              std::string("unsupported regex escape '\\") + e + "' in GBNF pattern conversion");
        }
    }

    Result<std::string> parse_char_class() {
        ++pos_;  // consume '['
        if (pos_ >= pat_.size()) {
            return Status(ErrorCode::invalid_argument, "unclosed '[' in regex pattern");
        }
        bool negated = false;
        if (pat_[pos_] == '^') {
            negated = true;
            ++pos_;
        }
        std::string content;
        if (negated) {
            // Always exclude unescaped JSON quotes, backslashes, and control chars.
            content += "\"\\\\\\x7F\\x00-\\x1F";
        }
        bool any = false;
        auto read_class_atom = [&](char& out_ch, std::string& out_shorthand) -> Status {
            out_shorthand.clear();
            if (pos_ >= pat_.size()) {
                return Status(ErrorCode::invalid_argument, "unclosed '[' in regex pattern");
            }
            const char c = pat_[pos_++];
            if (c == '\\') {
                if (pos_ >= pat_.size()) {
                    return Status(ErrorCode::invalid_argument, "trailing backslash in character class");
                }
                const char e = pat_[pos_++];
                if (e == 'd') {
                    out_shorthand = "0-9";
                    return Status::success();
                }
                if (e == 'w') {
                    out_shorthand = "a-zA-Z0-9_";
                    return Status::success();
                }
                if (e == 's') {
                    out_shorthand = " \\t";
                    return Status::success();
                }
                if (e == '-' || e == ']' || e == '[' || e == '^' || e == '.' || e == '/' || e == '_' || e == '+' ||
                    e == '*' || e == '?' || e == ':' || e == '@') {
                    out_ch = e;
                    return Status::success();
                }
                return Status(ErrorCode::invalid_argument,
                              std::string("unsupported escape '\\") + e + "' inside regex character class");
            }
            if (c == '"' || c == '\\') {
                return Status(ErrorCode::invalid_argument,
                              "unescaped quote or backslash inside regex character class is not supported");
            }
            out_ch = c;
            return Status::success();
        };

        while (pos_ < pat_.size() && pat_[pos_] != ']') {
            char first = 0;
            std::string sh;
            if (Status st = read_class_atom(first, sh); !st.ok()) return st;
            any = true;
            if (!sh.empty()) {
                content += sh;
                continue;
            }
            if (pos_ + 1 < pat_.size() && pat_[pos_] == '-' && pat_[pos_ + 1] != ']') {
                ++pos_;  // consume '-'
                char second = 0;
                std::string sh2;
                if (Status st = read_class_atom(second, sh2); !st.ok()) return st;
                if (!sh2.empty()) {
                    return Status(ErrorCode::invalid_argument, "invalid character class range with shorthand");
                }
                if (static_cast<unsigned char>(first) > static_cast<unsigned char>(second)) {
                    return Status(ErrorCode::invalid_argument, "inverted range in regex character class");
                }
                if (!negated) {
                    const auto u1 = static_cast<unsigned char>(first);
                    const auto u2 = static_cast<unsigned char>(second);
                    if ((u1 <= static_cast<unsigned char>('"') && static_cast<unsigned char>('"') <= u2) ||
                        (u1 <= static_cast<unsigned char>('\\') && static_cast<unsigned char>('\\') <= u2)) {
                        return Status(ErrorCode::invalid_argument,
                                      "character class range crossing '\"' or '\\\\' is not supported in GBNF");
                    }
                }
                content += escape_gbnf_class_char(first);
                content.push_back('-');
                content += escape_gbnf_class_char(second);
            } else {
                content += escape_gbnf_class_char(first);
            }
        }
        if (pos_ >= pat_.size() || pat_[pos_] != ']') {
            return Status(ErrorCode::invalid_argument, "unclosed '[' in regex pattern");
        }
        ++pos_;
        if (!any) {
            return Status(ErrorCode::invalid_argument, "empty character class '[]' in regex pattern");
        }
        return std::string("[") + (negated ? "^" : "") + content + "]";
    }

    Result<std::string> parse_quantifier() {
        if (pos_ >= pat_.size()) return std::string();
        const char q = pat_[pos_];
        if (q == '*' || q == '+' || q == '?') {
            ++pos_;
            if (pos_ < pat_.size() && (pat_[pos_] == '?' || pat_[pos_] == '+')) {
                return Status(ErrorCode::invalid_argument,
                              "lazy/possessive regex quantifiers are not supported in GBNF");
            }
            return std::string(1, q);
        }
        if (q == '{') {
            ++pos_;
            auto read_num = [&](std::size_t& val) -> bool {
                const std::size_t start = pos_;
                val = 0;
                while (pos_ < pat_.size() && pat_[pos_] >= '0' && pat_[pos_] <= '9') {
                    val = val * 10 + static_cast<std::size_t>(pat_[pos_] - '0');
                    if (val > kMaxUnrolledBounds) return false;
                    ++pos_;
                }
                return pos_ > start;
            };
            std::size_t lo = 0;
            if (!read_num(lo)) {
                return Status(ErrorCode::invalid_argument,
                              "invalid or out-of-range regex repetition quantifier (max 32)");
            }
            std::optional<std::size_t> hi;
            bool has_comma = false;
            if (pos_ < pat_.size() && pat_[pos_] == ',') {
                has_comma = true;
                ++pos_;
                if (pos_ < pat_.size() && pat_[pos_] != '}') {
                    std::size_t h = 0;
                    if (!read_num(h)) {
                        return Status(ErrorCode::invalid_argument,
                                      "invalid or out-of-range regex repetition upper bound (max 32)");
                    }
                    if (h < lo) {
                        return Status(ErrorCode::invalid_argument, "regex quantifier upper bound < lower bound");
                    }
                    hi = h;
                }
            }
            if (pos_ >= pat_.size() || pat_[pos_] != '}') {
                return Status(ErrorCode::invalid_argument, "unclosed '{...}' quantifier in regex pattern");
            }
            ++pos_;
            if (pos_ < pat_.size() && (pat_[pos_] == '?' || pat_[pos_] == '+')) {
                return Status(ErrorCode::invalid_argument,
                              "lazy/possessive regex quantifiers are not supported in GBNF");
            }
            if (!has_comma) {
                return "{" + std::to_string(lo) + "}";
            }
            if (!hi) {
                return "{" + std::to_string(lo) + ",}";
            }
            return "{" + std::to_string(lo) + "," + std::to_string(*hi) + "}";
        }
        return std::string();
    }

    std::string_view pat_;
    std::size_t pos_ = 0;
    std::size_t atoms_ = 0;
};

class SchemaConverter {
public:
    explicit SchemaConverter(const json::Value& root_schema) : root_schema_(root_schema) {}

    Result<std::string> convert() {
        if (root_schema_.is_bool()) {
            if (!root_schema_.as_bool()) {
                return Status(ErrorCode::invalid_argument, "JSON schema 'false' allows no valid values");
            }
            return std::string("root ::= value\n") + kCommonRules;
        }
        if (!root_schema_.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema must be an object");
        }
        auto root_expr = build_node(root_schema_, "root", 0);
        if (!root_expr.ok()) {
            return root_expr.status();
        }
        std::string out;
        out.reserve(4096);
        out += "root ::= " + root_expr.value() + "\n";
        for (const auto& [name, expr] : rules_) {
            out += name;
            out += " ::= ";
            out += expr;
            out += "\n";
            if (out.size() > kMaxGrammarBytes) {
                return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
            }
        }
        out += kCommonRules;
        if (out.size() > kMaxGrammarBytes) {
            return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
        }
        return out;
    }

private:
    struct BranchDomain {
        bool is_literal_only = false;
        std::vector<json::Value> literals;
        std::uint32_t type_mask = kTypeAll;
        std::map<std::string, std::vector<json::Value>> required_const_props;
    };

    Result<BranchDomain> classify_branch(const json::Value& schema, std::size_t depth,
                                         std::set<std::string>& ref_guard) const {
        if (depth > kMaxSchemaDepth) {
            return Status(ErrorCode::invalid_argument, "JSON schema nesting exceeds maximum depth");
        }
        if (schema.is_bool()) {
            BranchDomain d;
            d.type_mask = schema.as_bool() ? kTypeAll : 0u;
            return d;
        }
        if (!schema.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema node must be an object or boolean");
        }
        if (const json::Value* ref = schema.find("$ref"); ref != nullptr) {
            if (!ref->is_string()) {
                return Status(ErrorCode::invalid_argument, "$ref must be a string");
            }
            const std::string& r = ref->as_string();
            if (!ref_guard.insert(r).second) {
                return Status(ErrorCode::invalid_argument,
                              "direct or left-recursive $ref cycle is not supported: '" + r + "'");
            }
            auto target = resolve_ref(r);
            if (!target.ok()) {
                ref_guard.erase(r);
                return target.status();
            }
            auto res = classify_branch(*target.value(), depth + 1, ref_guard);
            ref_guard.erase(r);
            return res;
        }
        if (const json::Value* c = schema.find("const"); c != nullptr) {
            BranchDomain d;
            d.is_literal_only = true;
            d.literals.push_back(*c);
            d.type_mask = type_bit_for_value(*c);
            return d;
        }
        if (const json::Value* en = schema.find("enum"); en != nullptr) {
            if (!en->is_array() || en->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "enum must be a non-empty array");
            }
            BranchDomain d;
            d.is_literal_only = true;
            d.type_mask = 0u;
            for (const auto& item : en->as_array()) {
                d.literals.push_back(item);
                d.type_mask |= type_bit_for_value(item);
            }
            return d;
        }
        if (const json::Value* alts = schema.find("anyOf") != nullptr ? schema.find("anyOf") : schema.find("oneOf");
            alts != nullptr) {
            if (!alts->is_array() || alts->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "anyOf/oneOf must be a non-empty array");
            }
            BranchDomain combined;
            combined.is_literal_only = true;
            combined.type_mask = 0u;
            for (const auto& sub : alts->as_array()) {
                auto sd = classify_branch(sub, depth + 1, ref_guard);
                if (!sd.ok()) return sd.status();
                combined.type_mask |= sd.value().type_mask;
                if (!sd.value().is_literal_only) {
                    combined.is_literal_only = false;
                } else if (combined.is_literal_only) {
                    for (const auto& lit : sd.value().literals) {
                        combined.literals.push_back(lit);
                    }
                }
            }
            return combined;
        }

        BranchDomain d;
        const json::Value* type_v = schema.find("type");
        if (type_v != nullptr) {
            if (type_v->is_string()) {
                d.type_mask = type_bit_for_name(type_v->as_string());
                if (d.type_mask == 0u) {
                    return Status(ErrorCode::invalid_argument,
                                  "unsupported JSON schema type '" + type_v->as_string() + "'");
                }
            } else if (type_v->is_array() && !type_v->as_array().empty()) {
                d.type_mask = 0u;
                for (const auto& item : type_v->as_array()) {
                    if (!item.is_string()) {
                        return Status(ErrorCode::invalid_argument, "type array entries must be strings");
                    }
                    const std::uint32_t bit = type_bit_for_name(item.as_string());
                    if (bit == 0u) {
                        return Status(ErrorCode::invalid_argument,
                                      "unsupported JSON schema type '" + item.as_string() + "'");
                    }
                    d.type_mask |= bit;
                }
            } else {
                return Status(ErrorCode::invalid_argument, "type must be a string or non-empty array of strings");
            }
        } else if (schema.find("properties") != nullptr || schema.find("additionalProperties") != nullptr ||
                   schema.find("required") != nullptr) {
            d.type_mask = kTypeObj;
        } else if (schema.find("items") != nullptr || schema.find("minItems") != nullptr ||
                   schema.find("maxItems") != nullptr) {
            d.type_mask = kTypeArr;
        } else if (schema.find("minLength") != nullptr || schema.find("maxLength") != nullptr ||
                   schema.find("pattern") != nullptr) {
            d.type_mask = kTypeStr;
        } else {
            d.type_mask = kTypeAll;
        }

        // Capture required const/enum discriminator properties for object branches.
        if (d.type_mask == kTypeObj) {
            const json::Value* props = schema.find("properties");
            const json::Value* req = schema.find("required");
            if (props != nullptr && props->is_object() && req != nullptr && req->is_array()) {
                for (const auto& rname : req->as_array()) {
                    if (!rname.is_string()) continue;
                    const json::Value* pschema = props->find(rname.as_string());
                    if (pschema == nullptr) continue;
                    auto pdom = classify_branch(*pschema, depth + 1, ref_guard);
                    if (pdom.ok() && pdom.value().is_literal_only && !pdom.value().literals.empty()) {
                        d.required_const_props[rname.as_string()] = std::move(pdom.value().literals);
                    }
                }
            }
        }
        return d;
    }

    static bool domains_provably_disjoint(const BranchDomain& a, const BranchDomain& b) {
        if (a.type_mask == 0u || b.type_mask == 0u) {
            return true;
        }
        if (a.is_literal_only && b.is_literal_only) {
            return !literal_sets_overlap(a.literals, b.literals);
        }
        if (!type_masks_overlap(a.type_mask, b.type_mask)) {
            return true;
        }
        if (!a.is_literal_only && !b.is_literal_only && a.type_mask == kTypeObj && b.type_mask == kTypeObj) {
            for (const auto& [prop, vals_a] : a.required_const_props) {
                auto it = b.required_const_props.find(prop);
                if (it != b.required_const_props.end() && !literal_sets_overlap(vals_a, it->second)) {
                    return true;
                }
            }
        }
        return false;
    }

    std::string alloc_rule_name(std::string_view hint) {
        const std::string_view short_hint = hint.size() > 24 ? hint.substr(0, 24) : hint;
        std::string base = sanitize_rule_name(short_hint);
        if (base == "root" || base == "space" || base == "string" || base == "char" || base == "integer" ||
            base == "number" || base == "boolean" || base == "null" || base == "value" || base == "object" ||
            base == "array") {
            base = "s-" + base;
        }
        std::string candidate = base;
        while (used_names_.count(candidate) != 0) {
            candidate = base + "-" + std::to_string(++counter_);
        }
        used_names_.insert(candidate);
        return candidate;
    }

    std::string next_hint(std::string_view tag) {
        return std::string(tag) + "-" + std::to_string(++hint_counter_);
    }

    Result<std::string> add_rule(std::string_view hint, std::string expr) {
        if (rules_.size() + reserved_rules_ >= kMaxRules) {
            return Status(ErrorCode::invalid_argument, "JSON schema produces too many grammar rules");
        }
        std::string name = alloc_rule_name(hint);
        grammar_bytes_ += name.size() + expr.size() + 6;
        if (grammar_bytes_ > kMaxGrammarBytes) {
            return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
        }
        rules_.emplace_back(name, std::move(expr));
        return name;
    }

    static std::string unescape_json_pointer_token(std::string_view token) {
        std::string out;
        out.reserve(token.size());
        for (std::size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '~' && i + 1 < token.size()) {
                if (token[i + 1] == '1') {
                    out.push_back('/');
                    ++i;
                    continue;
                }
                if (token[i + 1] == '0') {
                    out.push_back('~');
                    ++i;
                    continue;
                }
            }
            out.push_back(token[i]);
        }
        return out;
    }

    Result<const json::Value*> resolve_ref(std::string_view ref) const {
        constexpr std::string_view kDefsPrefix = "#/$defs/";
        constexpr std::string_view kLegacyPrefix = "#/definitions/";
        const json::Value* container = nullptr;
        std::string_view raw_key;
        if (ref.rfind(kDefsPrefix, 0) == 0) {
            container = root_schema_.find("$defs");
            raw_key = ref.substr(kDefsPrefix.size());
        } else if (ref.rfind(kLegacyPrefix, 0) == 0) {
            container = root_schema_.find("definitions");
            raw_key = ref.substr(kLegacyPrefix.size());
        } else {
            return Status(ErrorCode::invalid_argument,
                          "unsupported $ref '" + std::string(ref) + "' (only local #/$defs/... or #/definitions/... supported)");
        }
        if (container == nullptr || !container->is_object()) {
            return Status(ErrorCode::invalid_argument, "unresolved $ref '" + std::string(ref) + "'");
        }
        const std::string key = unescape_json_pointer_token(raw_key);
        const json::Value* target = container->find(key);
        if (target == nullptr) {
            return Status(ErrorCode::invalid_argument, "unresolved $ref '" + std::string(ref) + "'");
        }
        return target;
    }

    Result<std::string> build_node(const json::Value& schema, std::string_view hint, std::size_t depth) {
        if (++node_visits_ > kMaxSchemaNodes) {
            return Status(ErrorCode::invalid_argument, "JSON schema traversal exceeds maximum complexity budget");
        }
        if (depth > kMaxSchemaDepth) {
            return Status(ErrorCode::invalid_argument, "JSON schema nesting exceeds maximum depth");
        }
        if (schema.is_bool()) {
            if (!schema.as_bool()) {
                return Status(ErrorCode::invalid_argument, "JSON schema 'false' allows no valid values");
            }
            return std::string("value");
        }
        if (!schema.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema node must be an object or boolean");
        }

        // Fail closed on any validation keyword this GBNF converter cannot faithfully enforce.
        for (const auto& [k, v] : schema.as_object()) {
            (void)v;
            if (!is_supported_schema_keyword(k)) {
                return Status(ErrorCode::invalid_argument,
                              "unsupported JSON schema validation keyword '" + k + "'");
            }
        }

        if (const json::Value* ref = schema.find("$ref"); ref != nullptr) {
            if (!ref->is_string()) {
                return Status(ErrorCode::invalid_argument, "$ref must be a string");
            }
            const std::string& ref_str = ref->as_string();
            if (active_refs_.count(ref_str) != 0) {
                return Status(ErrorCode::invalid_argument,
                              "direct or left-recursive $ref cycle is not supported: '" + ref_str + "'");
            }
            if (auto it = ref_rules_.find(ref_str); it != ref_rules_.end()) {
                return it->second;
            }
            auto target = resolve_ref(ref_str);
            if (!target.ok()) {
                return target.status();
            }
            if (rules_.size() + reserved_rules_ >= kMaxRules) {
                return Status(ErrorCode::invalid_argument, "JSON schema produces too many grammar rules");
            }
            // Reserve rule slot up front to prevent recursion overshoot past kMaxRules.
            const std::string rule_name = alloc_rule_name(ref_str);
            ref_rules_[ref_str] = rule_name;
            active_refs_.insert(ref_str);
            ++reserved_rules_;
            auto body = build_node(*target.value(), rule_name, depth + 1);
            --reserved_rules_;
            active_refs_.erase(ref_str);
            if (!body.ok()) {
                return body.status();
            }
            grammar_bytes_ += rule_name.size() + body.value().size() + 6;
            if (grammar_bytes_ > kMaxGrammarBytes) {
                return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
            }
            rules_.emplace_back(rule_name, std::move(body).value());
            return rule_name;
        }

        if (const json::Value* c = schema.find("const"); c != nullptr) {
            return gbnf_literal(c->dump()) + " space";
        }

        if (const json::Value* en = schema.find("enum"); en != nullptr) {
            if (!en->is_array() || en->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "enum must be a non-empty array");
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& item : en->as_array()) {
                if (!first) expr += " | ";
                first = false;
                expr += gbnf_literal(item.dump());
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ") space";
            return expr;
        }

        if (const json::Value* alts = schema.find("anyOf"); alts != nullptr) {
            if (!alts->is_array() || alts->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "anyOf must be a non-empty array");
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& branch : alts->as_array()) {
                auto built = build_node(branch, next_hint("alt"), depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                if (!first) expr += " | ";
                first = false;
                expr += "(" + built.value() + ")";
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ")";
            return expr;
        }

        if (const json::Value* one = schema.find("oneOf"); one != nullptr) {
            if (!one->is_array() || one->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "oneOf must be a non-empty array");
            }
            const auto& arr = one->as_array();
            std::vector<BranchDomain> domains;
            domains.reserve(arr.size());
            for (const auto& branch : arr) {
                std::set<std::string> ref_guard;
                auto dom = classify_branch(branch, depth + 1, ref_guard);
                if (!dom.ok()) {
                    return dom.status();
                }
                domains.push_back(std::move(dom).value());
            }
            for (std::size_t i = 0; i < domains.size(); ++i) {
                for (std::size_t j = i + 1; j < domains.size(); ++j) {
                    if (!domains_provably_disjoint(domains[i], domains[j])) {
                        return Status(ErrorCode::invalid_argument,
                                      "oneOf branches " + std::to_string(i) + " and " + std::to_string(j) +
                                          " are not provably disjoint and cannot be safely enforced by GBNF");
                    }
                }
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& branch : arr) {
                auto built = build_node(branch, next_hint("oneof"), depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                if (!first) expr += " | ";
                first = false;
                expr += "(" + built.value() + ")";
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ")";
            return expr;
        }

        const json::Value* type_v = schema.find("type");
        if (type_v != nullptr && type_v->is_array()) {
            if (type_v->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "type array must not be empty");
            }
            std::string expr = "(";
            bool first = true;
            std::set<std::string> seen_types;
            for (const auto& t : type_v->as_array()) {
                if (!t.is_string()) {
                    return Status(ErrorCode::invalid_argument, "type array entries must be strings");
                }
                if (!seen_types.insert(t.as_string()).second) {
                    continue;
                }
                auto built = build_for_type(schema, t.as_string(), hint, depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                if (!first) expr += " | ";
                first = false;
                expr += "(" + built.value() + ")";
                if (expr.size() + grammar_bytes_ > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
            }
            expr += ")";
            return expr;
        }

        std::string type_name;
        if (type_v != nullptr) {
            if (!type_v->is_string()) {
                return Status(ErrorCode::invalid_argument, "type must be a string or array of strings");
            }
            type_name = type_v->as_string();
        } else if (schema.find("properties") != nullptr || schema.find("additionalProperties") != nullptr ||
                   schema.find("required") != nullptr) {
            type_name = "object";
        } else if (schema.find("items") != nullptr || schema.find("minItems") != nullptr ||
                   schema.find("maxItems") != nullptr) {
            type_name = "array";
        } else if (schema.find("minLength") != nullptr || schema.find("maxLength") != nullptr ||
                   schema.find("pattern") != nullptr) {
            type_name = "string";
        } else if (schema.find("minimum") != nullptr || schema.find("maximum") != nullptr) {
            return Status(ErrorCode::invalid_argument,
                          "minimum/maximum requires explicit type 'integer' in GBNF conversion");
        } else {
            return std::string("value");
        }
        return build_for_type(schema, type_name, hint, depth);
    }

    Result<std::string> build_for_type(const json::Value& schema, std::string_view type_name, std::string_view hint,
                                       std::size_t depth) {
        if (type_name == "string") {
            return build_string(schema);
        }
        if (type_name == "integer") {
            return build_integer(schema);
        }
        if (type_name == "number") {
            if (schema.find("minimum") != nullptr || schema.find("maximum") != nullptr) {
                return Status(ErrorCode::invalid_argument,
                              "minimum/maximum is only supported for bounded integer schemas, not floating-point numbers");
            }
            return std::string("number");
        }
        if (type_name == "boolean") {
            return std::string("boolean");
        }
        if (type_name == "null") {
            return std::string("null");
        }
        if (type_name == "array") {
            return build_array(schema, hint, depth);
        }
        if (type_name == "object") {
            return build_object(schema, hint, depth);
        }
        return Status(ErrorCode::invalid_argument, "unsupported JSON schema type '" + std::string(type_name) + "'");
    }

    Result<std::string> build_integer(const json::Value& schema) {
        const json::Value* mn = schema.find("minimum");
        const json::Value* mx = schema.find("maximum");
        if (mn == nullptr && mx == nullptr) {
            return std::string("integer");
        }
        std::optional<std::int64_t> min_v;
        std::optional<std::int64_t> max_v;
        auto read_bound = [](const json::Value* v, const char* name, std::optional<std::int64_t>& out) -> Status {
            if (v == nullptr) return Status::success();
            if (v->is_integer()) {
                out = v->as_int();
                return Status::success();
            }
            if (v->is_number() && std::isfinite(v->as_double()) && std::floor(v->as_double()) == v->as_double() &&
                std::fabs(v->as_double()) <= static_cast<double>(kMaxBoundedIntegerAbs)) {
                out = static_cast<std::int64_t>(v->as_double());
                return Status::success();
            }
            return Status(ErrorCode::invalid_argument,
                          std::string(name) + " must be an integer within [-" +
                              std::to_string(kMaxBoundedIntegerAbs) + ", " + std::to_string(kMaxBoundedIntegerAbs) + "]");
        };
        if (Status st = read_bound(mn, "minimum", min_v); !st.ok()) return st;
        if (Status st = read_bound(mx, "maximum", max_v); !st.ok()) return st;

        if (min_v && max_v) {
            if (*min_v > *max_v) {
                return Status(ErrorCode::invalid_argument, "maximum must be >= minimum");
            }
            if (*min_v < -kMaxBoundedIntegerAbs || *max_v > kMaxBoundedIntegerAbs) {
                return Status(ErrorCode::invalid_argument,
                              "integer minimum/maximum outside [-" + std::to_string(kMaxBoundedIntegerAbs) + ", " +
                                  std::to_string(kMaxBoundedIntegerAbs) + "] is not supported by GBNF conversion");
            }
            return bounded_integer_gbnf(*min_v, *max_v);
        }
        if (min_v && *min_v == 0) {
            return std::string("([0-9] | [1-9] [0-9]{0,15}) space");
        }
        if (max_v && *max_v == 0) {
            return std::string("(\"0\" | \"-\" [1-9] [0-9]{0,15}) space");
        }
        return Status(ErrorCode::invalid_argument,
                      "half-open integer minimum/maximum other than 0 is not supported by GBNF conversion; "
                      "specify both minimum and maximum within [-999999, 999999]");
    }

    Result<std::string> build_string(const json::Value& schema) {
        const json::Value* pat = schema.find("pattern");
        const json::Value* mn = schema.find("minLength");
        const json::Value* mx = schema.find("maxLength");
        if (pat != nullptr) {
            if (!pat->is_string()) {
                return Status(ErrorCode::invalid_argument, "pattern must be a string");
            }
            if (mn != nullptr || mx != nullptr) {
                return Status(ErrorCode::invalid_argument,
                              "combining pattern with minLength/maxLength is not supported by GBNF conversion");
            }
            RegexToGbnfParser parser(pat->as_string());
            return parser.convert();
        }
        std::size_t min_len = 0;
        std::optional<std::size_t> max_len;
        if (mn != nullptr) {
            if (!mn->is_integer() || mn->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "minLength must be a non-negative integer");
            }
            min_len = static_cast<std::size_t>(mn->as_uint());
        }
        if (mx != nullptr) {
            if (!mx->is_integer() || mx->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "maxLength must be a non-negative integer");
            }
            max_len = static_cast<std::size_t>(mx->as_uint());
        }
        if (max_len && *max_len < min_len) {
            return Status(ErrorCode::invalid_argument, "maxLength must be >= minLength");
        }
        if (min_len == 0 && !max_len) {
            return std::string("string");
        }
        if (min_len <= kMaxUnrolledBounds && (!max_len || *max_len <= kMaxUnrolledBounds)) {
            std::string rep = "char{" + std::to_string(min_len) + ",";
            if (max_len) {
                rep += std::to_string(*max_len);
            }
            rep += "}";
            return "\"\\\"\" " + rep + " \"\\\"\" space";
        }
        return Status(ErrorCode::invalid_argument,
                      "minLength/maxLength above " + std::to_string(kMaxUnrolledBounds) +
                          " is not supported by GBNF conversion");
    }

    Result<std::string> build_array(const json::Value& schema, std::string_view hint, std::size_t depth) {
        (void)hint;
        std::string item_rule = "value";
        if (const json::Value* items = schema.find("items"); items != nullptr) {
            const std::string item_hint = next_hint("item");
            auto built = build_node(*items, item_hint, depth + 1);
            if (!built.ok()) {
                return built.status();
            }
            auto named = add_rule(item_hint, std::move(built).value());
            if (!named.ok()) {
                return named.status();
            }
            item_rule = std::move(named).value();
        }
        std::size_t min_items = 0;
        std::optional<std::size_t> max_items;
        if (const json::Value* mn = schema.find("minItems"); mn != nullptr) {
            if (!mn->is_integer() || mn->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "minItems must be a non-negative integer");
            }
            min_items = static_cast<std::size_t>(mn->as_uint());
        }
        if (const json::Value* mx = schema.find("maxItems"); mx != nullptr) {
            if (!mx->is_integer() || mx->as_int(-1) < 0) {
                return Status(ErrorCode::invalid_argument, "maxItems must be a non-negative integer");
            }
            max_items = static_cast<std::size_t>(mx->as_uint());
        }
        if (max_items && *max_items < min_items) {
            return Status(ErrorCode::invalid_argument, "maxItems must be >= minItems");
        }
        if (min_items == 0 && !max_items) {
            return "\"[\" space ( " + item_rule + " (\",\" space " + item_rule + ")* )? \"]\" space";
        }
        if (max_items && *max_items == 0) {
            return std::string("\"[\" space \"]\" space");
        }
        if (min_items <= kMaxUnrolledBounds && (!max_items || *max_items <= kMaxUnrolledBounds)) {
            std::string body;
            if (min_items > 0) {
                body += item_rule;
                for (std::size_t i = 1; i < min_items; ++i) {
                    body += " \",\" space " + item_rule;
                }
                if (!max_items) {
                    body += " (\",\" space " + item_rule + ")*";
                } else {
                    for (std::size_t i = min_items; i < *max_items; ++i) {
                        body += " (\",\" space " + item_rule + ")?";
                    }
                }
            } else {
                // min_items == 0, max_items >= 1
                body = "( " + item_rule;
                for (std::size_t i = 1; i < *max_items; ++i) {
                    body += " (\",\" space " + item_rule + ")?";
                }
                body += " )?";
            }
            return "\"[\" space " + body + " \"]\" space";
        }
        return Status(ErrorCode::invalid_argument,
                      "minItems/maxItems above " + std::to_string(kMaxUnrolledBounds) +
                          " is not supported by GBNF conversion");
    }

    Result<std::string> build_object(const json::Value& schema, std::string_view hint, std::size_t depth) {
        (void)hint;
        const json::Value* props_v = schema.find("properties");
        if (props_v != nullptr && !props_v->is_object()) {
            return Status(ErrorCode::invalid_argument, "properties must be an object");
        }
        const bool has_declared_props = (props_v != nullptr && !props_v->as_object().empty());
        std::set<std::string> required_set;
        if (const json::Value* req_v = schema.find("required"); req_v != nullptr) {
            if (!req_v->is_array()) {
                return Status(ErrorCode::invalid_argument, "required must be an array of strings");
            }
            for (const auto& item : req_v->as_array()) {
                if (!item.is_string()) {
                    return Status(ErrorCode::invalid_argument, "required entries must be strings");
                }
                required_set.insert(item.as_string());
            }
        }

        // additionalProperties: default true only when no properties are declared.
        bool allow_additional = !has_declared_props;
        std::string additional_val_rule = "value";
        if (const json::Value* ap = schema.find("additionalProperties"); ap != nullptr) {
            if (ap->is_bool()) {
                allow_additional = ap->as_bool();
            } else if (ap->is_object()) {
                allow_additional = true;
                const std::string extra_hint = next_hint("extra");
                auto built = build_node(*ap, extra_hint, depth + 1);
                if (!built.ok()) {
                    return built.status();
                }
                auto named = add_rule(extra_hint, std::move(built).value());
                if (!named.ok()) {
                    return named.status();
                }
                additional_val_rule = std::move(named).value();
            } else {
                return Status(ErrorCode::invalid_argument, "additionalProperties must be a boolean or schema object");
            }
        }

        // GBNF cannot express "any string EXCEPT declared property names" cleanly without
        // exponential negation; fail closed instead of letting generic additionalProperties
        // bypass a declared property's schema.
        if (has_declared_props && allow_additional) {
            return Status(ErrorCode::invalid_argument,
                          "combining declared properties with permissive additionalProperties is not supported by "
                          "GBNF conversion; set additionalProperties to false");
        }

        if (!has_declared_props && required_set.empty()) {
            if (!allow_additional) {
                return std::string("\"{\" space \"}\" space");
            }
            if (additional_val_rule == "value") {
                return std::string("object");
            }
            const std::string kv = "string \":\" space " + additional_val_rule;
            return "\"{\" space ( " + kv + " (\",\" space " + kv + ")* )? \"}\" space";
        }

        std::vector<std::string> req_rules;
        std::vector<std::string> opt_rules;
        std::set<std::string> seen_props;
        if (has_declared_props) {
            for (const auto& [prop_name, prop_schema] : props_v->as_object()) {
                seen_props.insert(prop_name);
                const std::string prop_hint = next_hint("prop");
                auto val_expr = build_node(prop_schema, prop_hint, depth + 1);
                if (!val_expr.ok()) {
                    return val_expr.status();
                }
                const std::string key_json = json::Value(prop_name).dump();
                const std::string kv_expr = gbnf_literal(key_json) + " space \":\" space (" + val_expr.value() + ")";
                auto kv_rule = add_rule(next_hint("kv"), kv_expr);
                if (!kv_rule.ok()) {
                    return kv_rule.status();
                }
                if (required_set.count(prop_name) != 0) {
                    req_rules.push_back(std::move(kv_rule).value());
                } else {
                    opt_rules.push_back(std::move(kv_rule).value());
                }
            }
        }
        // Honor any `required` keys not explicitly listed under `properties`.
        for (const std::string& req_name : required_set) {
            if (seen_props.count(req_name) != 0) continue;
            if (!allow_additional) {
                return Status(ErrorCode::invalid_argument,
                              "required property '" + req_name + "' is absent from properties while additionalProperties is false");
            }
            const std::string key_json = json::Value(req_name).dump();
            const std::string kv_expr = gbnf_literal(key_json) + " space \":\" space (" + additional_val_rule + ")";
            auto kv_rule = add_rule(next_hint("kv"), kv_expr);
            if (!kv_rule.ok()) {
                return kv_rule.status();
            }
            req_rules.push_back(std::move(kv_rule).value());
        }

        std::string extra_kv_rule;
        if (allow_additional) {
            auto kv = add_rule(next_hint("extra-kv"), "string \":\" space " + additional_val_rule);
            if (!kv.ok()) {
                return kv.status();
            }
            extra_kv_rule = std::move(kv).value();
        }

        std::string obj = "\"{\" space ";
        if (!req_rules.empty()) {
            for (std::size_t i = 0; i < req_rules.size(); ++i) {
                if (i > 0) obj += " \",\" space ";
                obj += req_rules[i];
            }
            for (const auto& opt : opt_rules) {
                obj += " (\",\" space " + opt + ")?";
            }
            if (!extra_kv_rule.empty()) {
                obj += " (\",\" space " + extra_kv_rule + ")*";
            }
        } else {
            // All properties are optional: build exact comma-safe alternatives so any
            // subset (including empty {}) is valid JSON with no leading/trailing commas.
            std::vector<std::string> branches;
            std::size_t total_branch_bytes = 0;
            for (std::size_t i = 0; i < opt_rules.size(); ++i) {
                std::string branch = opt_rules[i];
                for (std::size_t j = i + 1; j < opt_rules.size(); ++j) {
                    branch += " (\",\" space " + opt_rules[j] + ")?";
                }
                if (!extra_kv_rule.empty()) {
                    branch += " (\",\" space " + extra_kv_rule + ")*";
                }
                total_branch_bytes += branch.size() + 5;
                if (grammar_bytes_ + total_branch_bytes > kMaxGrammarBytes) {
                    return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
                }
                branches.push_back(std::move(branch));
            }
            if (!extra_kv_rule.empty()) {
                branches.push_back(extra_kv_rule + " (\",\" space " + extra_kv_rule + ")*");
            }
            if (!branches.empty()) {
                obj += "( ";
                for (std::size_t i = 0; i < branches.size(); ++i) {
                    if (i > 0) obj += " | ";
                    obj += "(" + branches[i] + ")";
                }
                obj += " )?";
            }
        }
        obj += " \"}\" space";
        auto named = add_rule(next_hint("obj"), std::move(obj));
        if (!named.ok()) {
            return named.status();
        }
        return named.value();
    }

    const json::Value& root_schema_;
    std::vector<std::pair<std::string, std::string>> rules_;
    std::set<std::string> used_names_;
    std::map<std::string, std::string> ref_rules_;
    std::set<std::string> active_refs_;
    std::size_t reserved_rules_ = 0;
    std::size_t grammar_bytes_ = 0;
    std::size_t counter_ = 0;
    std::size_t hint_counter_ = 0;
    std::size_t node_visits_ = 0;
};

class GbnfStructuredGrammarEngine final : public StructuredGrammarEngine {
public:
    GrammarEngineKind kind() const noexcept override { return GrammarEngineKind::gbnf; }
    std::string_view name() const noexcept override { return "gbnf"; }

    Result<CompiledGrammar> compile_json_object() const override {
        CompiledGrammar out;
        out.engine = GrammarEngineKind::gbnf;
        out.gbnf = json_object_gbnf();
        out.raw_schema = R"({"type":"object"})";
        return out;
    }

    Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const override {
        auto gbnf = json_schema_to_gbnf(schema);
        if (!gbnf.ok()) {
            return gbnf.status();
        }
        CompiledGrammar out;
        out.engine = GrammarEngineKind::gbnf;
        out.gbnf = std::move(gbnf).value();
        out.raw_schema = schema.dump();
        return out;
    }
};

#if defined(SONDER_WITH_LLGUIDANCE)
// Compile-time plug-in seam for Rust `llguidance` structured decoding.
// When SONDER_WITH_LLGUIDANCE is enabled, callers continue using
// default_structured_grammar_engine() with zero API changes.
class LlguidanceStructuredGrammarEngine final : public StructuredGrammarEngine {
public:
    GrammarEngineKind kind() const noexcept override { return GrammarEngineKind::llguidance; }
    std::string_view name() const noexcept override { return "llguidance"; }

    Result<CompiledGrammar> compile_json_object() const override {
        CompiledGrammar out;
        out.engine = GrammarEngineKind::llguidance;
        out.raw_schema = R"({"type":"object"})";
        out.gbnf = json_object_gbnf();
        return out;
    }

    Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const override {
        if (!schema.is_object() && !schema.is_bool()) {
            return Status(ErrorCode::invalid_argument, "JSON schema must be an object or boolean");
        }
        if (schema.is_bool() && !schema.as_bool()) {
            return Status(ErrorCode::invalid_argument, "JSON schema 'false' allows no valid values");
        }
        CompiledGrammar out;
        out.engine = GrammarEngineKind::llguidance;
        out.raw_schema = schema.dump();
        // Best-effort GBNF fallback for backends that still consume GBNF directly.
        if (auto fallback = json_schema_to_gbnf(schema); fallback.ok()) {
            out.gbnf = std::move(fallback).value();
        }
        return out;
    }
};
#endif

}  // namespace

std::string json_object_gbnf() {
    return std::string("root ::= object\n") + kCommonRules;
}

Result<std::string> json_schema_to_gbnf(const json::Value& schema) {
    SchemaConverter converter(schema);
    return converter.convert();
}

const StructuredGrammarEngine& gbnf_structured_grammar_engine() noexcept {
    static const GbnfStructuredGrammarEngine kEngine;
    return kEngine;
}

const StructuredGrammarEngine& default_structured_grammar_engine() noexcept {
#if defined(SONDER_WITH_LLGUIDANCE)
    static const LlguidanceStructuredGrammarEngine kEngine;
    return kEngine;
#else
    return gbnf_structured_grammar_engine();
#endif
}

}  // namespace sonder::inference::server::detail
