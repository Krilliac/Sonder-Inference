#include "json_schema_gbnf.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sha256.hpp"

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

bool is_schema_annotation_keyword(std::string_view key) noexcept {
    return key == "$schema" || key == "$id" || key == "$anchor" || key == "$comment" || key == "title" ||
           key == "description" || key == "default" || key == "examples" || key == "readOnly" ||
           key == "writeOnly" || key == "deprecated";
}

bool is_supported_schema_keyword(std::string_view key) noexcept {
    // Functional keywords enforced by the converter:
    if (key == "$ref" || key == "$defs" || key == "definitions" || key == "type" || key == "const" || key == "enum" ||
        key == "anyOf" || key == "oneOf" || key == "properties" || key == "required" ||
        key == "additionalProperties" || key == "items" || key == "minLength" || key == "maxLength" ||
        key == "minItems" || key == "maxItems" || key == "pattern" || key == "minimum" || key == "maximum") {
        return true;
    }
    return is_schema_annotation_keyword(key);
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

// Strict RFC 3629 UTF-8 code-point decoder (rejects overlongs, UTF-16
// surrogates U+D800..U+DFFF, code points > U+10FFFF, and truncated sequences).
bool decode_utf8_cp(std::string_view s, std::size_t pos, std::uint32_t& cp, std::size_t& len) noexcept {
    if (pos >= s.size()) return false;
    const auto b0 = static_cast<unsigned char>(s[pos]);
    if (b0 < 0x80u) {
        cp = b0;
        len = 1;
        return true;
    }
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        if (pos + 2 > s.size()) return false;
        const auto b1 = static_cast<unsigned char>(s[pos + 1]);
        if (b1 < 0x80u || b1 > 0xBFu) return false;
        cp = (static_cast<std::uint32_t>(b0 & 0x1Fu) << 6) | static_cast<std::uint32_t>(b1 & 0x3Fu);
        len = 2;
        return true;
    }
    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        if (pos + 3 > s.size()) return false;
        const auto b1 = static_cast<unsigned char>(s[pos + 1]);
        const auto b2 = static_cast<unsigned char>(s[pos + 2]);
        const unsigned char lo = (b0 == 0xE0u) ? 0xA0u : 0x80u;
        const unsigned char hi = (b0 == 0xEDu) ? 0x9Fu : 0xBFu;
        if (b1 < lo || b1 > hi || b2 < 0x80u || b2 > 0xBFu) return false;
        cp = (static_cast<std::uint32_t>(b0 & 0x0Fu) << 12) | (static_cast<std::uint32_t>(b1 & 0x3Fu) << 6) |
             static_cast<std::uint32_t>(b2 & 0x3Fu);
        len = 3;
        return true;
    }
    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        if (pos + 4 > s.size()) return false;
        const auto b1 = static_cast<unsigned char>(s[pos + 1]);
        const auto b2 = static_cast<unsigned char>(s[pos + 2]);
        const auto b3 = static_cast<unsigned char>(s[pos + 3]);
        const unsigned char lo = (b0 == 0xF0u) ? 0x90u : 0x80u;
        const unsigned char hi = (b0 == 0xF4u) ? 0x8Fu : 0xBFu;
        if (b1 < lo || b1 > hi || b2 < 0x80u || b2 > 0xBFu || b3 < 0x80u || b3 > 0xBFu) return false;
        cp = (static_cast<std::uint32_t>(b0 & 0x07u) << 18) | (static_cast<std::uint32_t>(b1 & 0x3Fu) << 12) |
             (static_cast<std::uint32_t>(b2 & 0x3Fu) << 6) | static_cast<std::uint32_t>(b3 & 0x3Fu);
        len = 4;
        return true;
    }
    return false;
}

// Escapes raw bytes as a GBNF double-quoted string literal (including quotes).
std::string gbnf_literal(std::string_view text) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() + 4);
    out.push_back('"');
    for (std::size_t i = 0; i < text.size();) {
        const char ch = text[i];
        const auto u = static_cast<unsigned char>(ch);
        if (u >= 0x80u) {
            std::uint32_t cp = 0;
            std::size_t len = 0;
            if (decode_utf8_cp(text, i, cp, len)) {
                out.append(text.data() + i, len);
                i += len;
            } else {
                out += "\\x";
                out.push_back(kHex[(u >> 4) & 0x0Fu]);
                out.push_back(kHex[u & 0x0Fu]);
                ++i;
            }
            continue;
        }
        ++i;
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (u < 0x20u || u == 0x7Fu) {
                    out += "\\x";
                    out.push_back(kHex[(u >> 4) & 0x0Fu]);
                    out.push_back(kHex[u & 0x0Fu]);
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

bool value_matches_type_mask(const json::Value& v, std::uint32_t mask) noexcept {
    const std::uint32_t bit = type_bit_for_value(v);
    if ((bit & mask) != 0u) {
        return true;
    }
    // JSON Schema "number" accepts JSON integers (and whole-valued floats satisfy "integer").
    if (bit == kTypeInt && (mask & kTypeNum) != 0u) {
        return true;
    }
    if (bit == kTypeNum && (mask & kTypeInt) != 0u && std::isfinite(v.as_double()) &&
        std::floor(v.as_double()) == v.as_double()) {
        return true;
    }
    return false;
}

Result<std::uint32_t> parse_type_mask(const json::Value& type_v) {
    if (type_v.is_string()) {
        const std::uint32_t bit = type_bit_for_name(type_v.as_string());
        if (bit == 0u) {
            return Status(ErrorCode::invalid_argument, "unsupported JSON schema type '" + type_v.as_string() + "'");
        }
        return bit;
    }
    if (type_v.is_array() && !type_v.as_array().empty()) {
        std::uint32_t mask = 0u;
        for (const auto& item : type_v.as_array()) {
            if (!item.is_string()) {
                return Status(ErrorCode::invalid_argument, "type array entries must be strings");
            }
            const std::uint32_t bit = type_bit_for_name(item.as_string());
            if (bit == 0u) {
                return Status(ErrorCode::invalid_argument, "unsupported JSON schema type '" + item.as_string() + "'");
            }
            mask |= bit;
        }
        return mask;
    }
    return Status(ErrorCode::invalid_argument, "type must be a string or non-empty array of strings");
}

bool int_equals_double(const json::Value& int_val, double d) noexcept {
    if (!std::isfinite(d) || std::floor(d) != d) {
        return false;
    }
    if (const auto exact = int_val.try_as_int()) {
        // IEEE-754 double represents all integers in [-2^53, 2^53] exactly.
        // Beyond that, only exact powers/multiples of 2 are representable; check
        // by converting `d` back when in int64 range, otherwise compare via uint64.
        constexpr double kMinInt64 = -9223372036854775808.0;  // -2^63 (exact in double)
        constexpr double kMaxInt64Exclusive = 9223372036854775808.0;  // +2^63 (exact in double)
        if (d < kMinInt64 || d >= kMaxInt64Exclusive) {
            return false;
        }
        return *exact == static_cast<std::int64_t>(d);
    }
    constexpr double kMaxUint64Exclusive = 18446744073709551616.0;  // 2^64 (exact in double)
    if (d < 0.0 || d >= kMaxUint64Exclusive) {
        return false;
    }
    return int_val.as_uint() == static_cast<std::uint64_t>(d);
}

bool json_values_equal(const json::Value& a, const json::Value& b, std::size_t depth = 0) {
    if (depth > 128u) {
        // Fail closed for disjointness checks (treat overly deep values as potentially equal).
        return true;
    }
    if (a.is_null() && b.is_null()) return true;
    if (a.is_bool() && b.is_bool()) return a.as_bool() == b.as_bool();
    if (a.is_number() && b.is_number()) {
        if (a.is_integer() && b.is_integer()) {
            const auto ia = a.try_as_int();
            const auto ib = b.try_as_int();
            if (ia && ib) return *ia == *ib;
            if (ia && *ia < 0) return false;
            if (ib && *ib < 0) return false;
            return a.as_uint() == b.as_uint();
        }
        if (a.is_integer()) return int_equals_double(a, b.as_double());
        if (b.is_integer()) return int_equals_double(b, a.as_double());
        return a.as_double() == b.as_double();
    }
    if (a.is_string() && b.is_string()) return a.as_string() == b.as_string();
    if (a.is_array() && b.is_array()) {
        const auto& aa = a.as_array();
        const auto& ba = b.as_array();
        if (aa.size() != ba.size()) return false;
        for (std::size_t i = 0; i < aa.size(); ++i) {
            if (!json_values_equal(aa[i], ba[i], depth + 1)) return false;
        }
        return true;
    }
    if (a.is_object() && b.is_object()) {
        const auto& ao = a.as_object();
        const auto& bo = b.as_object();
        if (ao.size() != bo.size()) return false;
        for (const auto& [k, av] : ao) {
            const json::Value* bv = bo.find(k);
            if (bv == nullptr || !json_values_equal(av, *bv, depth + 1)) return false;
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
        for (std::size_t i = 0; i < pat_.size();) {
            std::uint32_t cp = 0;
            std::size_t len = 0;
            if (!decode_utf8_cp(pat_, i, cp, len)) {
                return Status(ErrorCode::invalid_argument,
                              "pattern must be valid UTF-8 without surrogate or truncated sequences");
            }
            if (cp < 0x20u || cp == 0x7Fu) {
                return Status(ErrorCode::invalid_argument, "pattern must not contain control characters");
            }
            i += len;
        }
        if (pat_.empty()) {
            return std::string("string");
        }

        // Split top-level alternations BEFORE stripping ^/$ anchors so patterns
        // like `^foo|bar$` keep per-branch anchoring instead of turning into
        // `^(foo|bar)$`.
        pos_ = 0;
        auto expr = parse_top_alternation();
        if (!expr.ok()) {
            return expr.status();
        }
        if (pos_ != pat_.size()) {
            return Status(ErrorCode::invalid_argument, "unexpected trailing character in regex pattern");
        }
        if (expr.value().empty()) {
            return std::string("\"\\\"\\\"\" space");
        }
        return "\"\\\"\" (" + expr.value() + ") \"\\\"\" space";
    }

private:
    static constexpr std::size_t kMaxQuantifierExpansion = 64;

    struct ParsedAtom {
        std::string gbnf;
        bool has_unbounded_quantifier = false;
        bool can_match_empty = false;
        std::size_t max_repeat_factor = 1;
    };

    Result<std::string> parse_top_alternation() {
        auto parse_branch = [&]() -> Result<std::string> {
            bool anchored_start = false;
            if (pos_ < pat_.size() && pat_[pos_] == '^') {
                anchored_start = true;
                ++pos_;
            }
            auto seq = parse_sequence(0, /*allow_trailing_dollar=*/true);
            if (!seq.ok()) return seq.status();
            bool anchored_end = false;
            if (pos_ < pat_.size() && pat_[pos_] == '$') {
                anchored_end = true;
                ++pos_;
            }
            std::string branch;
            if (!anchored_start) branch += "char* ";
            if (!seq.value().gbnf.empty()) {
                branch += "(" + seq.value().gbnf + ")";
            }
            if (!anchored_end) {
                if (!branch.empty()) branch += " ";
                branch += "char*";
            }
            return branch;
        };
        auto first = parse_branch();
        if (!first.ok()) return first.status();
        std::string out = std::move(first).value();
        while (pos_ < pat_.size() && pat_[pos_] == '|') {
            ++pos_;
            auto next = parse_branch();
            if (!next.ok()) return next.status();
            out += " | " + next.value();
        }
        return out;
    }

    static std::string escape_gbnf_class_atom(std::string_view utf8_cp) {
        if (utf8_cp.size() == 1) {
            const char ch = utf8_cp[0];
            if (ch == ']' || ch == '\\' || ch == '^' || ch == '-') {
                std::string s = "\\";
                s.push_back(ch);
                return s;
            }
        }
        return std::string(utf8_cp);
    }

    Result<ParsedAtom> parse_alternation(std::size_t depth) {
        auto first = parse_sequence(depth, false);
        if (!first.ok()) return first.status();
        ParsedAtom out = std::move(first).value();
        while (pos_ < pat_.size() && pat_[pos_] == '|') {
            ++pos_;
            auto next = parse_sequence(depth, false);
            if (!next.ok()) return next.status();
            out.gbnf += " | " + next.value().gbnf;
            out.has_unbounded_quantifier = out.has_unbounded_quantifier || next.value().has_unbounded_quantifier;
            out.can_match_empty = out.can_match_empty || next.value().can_match_empty;
            out.max_repeat_factor = std::max(out.max_repeat_factor, next.value().max_repeat_factor);
        }
        return out;
    }

    Result<ParsedAtom> parse_sequence(std::size_t depth, bool allow_trailing_dollar) {
        ParsedAtom out;
        out.can_match_empty = true;
        while (pos_ < pat_.size() && pat_[pos_] != ')' && pat_[pos_] != '|') {
            if (allow_trailing_dollar && pat_[pos_] == '$' &&
                (pos_ + 1 == pat_.size() || pat_[pos_ + 1] == '|')) {
                break;
            }
            auto atom = parse_atom(depth);
            if (!atom.ok()) return atom.status();
            bool q_unbounded = false;
            bool q_can_be_zero = false;
            std::size_t q_max = 1;
            auto quant = parse_quantifier(q_unbounded, q_can_be_zero, q_max);
            if (!quant.ok()) return quant.status();
            if (!quant.value().empty()) {
                // Reject nested unbounded quantifiers (`(a+)+`, `(a*)+`, `(a+)*`,
                // `(a?)+`, `(|a)+`, `(a{0,5})+`) and multiplicative blowup
                // (`((a?){32}){32}`) to prevent Earley/backtracking DoS.
                if (q_unbounded &&
                    (atom.value().has_unbounded_quantifier || atom.value().max_repeat_factor > 1 ||
                     atom.value().can_match_empty)) {
                    return Status(ErrorCode::invalid_argument,
                                  "nested or nullable unbounded regex quantifier is not supported in GBNF");
                }
                if (atom.value().has_unbounded_quantifier && q_max > 1) {
                    return Status(ErrorCode::invalid_argument,
                                  "nested unbounded regex quantifiers are not supported in GBNF");
                }
                if (q_max > 1 && atom.value().max_repeat_factor > kMaxQuantifierExpansion / q_max) {
                    return Status(ErrorCode::invalid_argument,
                                  "nested regex quantifier expansion exceeds maximum complexity");
                }
                atom.value().has_unbounded_quantifier = atom.value().has_unbounded_quantifier || q_unbounded;
                atom.value().can_match_empty = atom.value().can_match_empty || q_can_be_zero;
                atom.value().max_repeat_factor *= q_max;
            }
            if (!out.gbnf.empty()) out.gbnf.push_back(' ');
            out.gbnf += atom.value().gbnf;
            out.gbnf += quant.value();
            out.has_unbounded_quantifier = out.has_unbounded_quantifier || atom.value().has_unbounded_quantifier;
            out.can_match_empty = out.can_match_empty && atom.value().can_match_empty;
            out.max_repeat_factor = std::max(out.max_repeat_factor, atom.value().max_repeat_factor);
        }
        return out;
    }

    Result<ParsedAtom> parse_atom(std::size_t depth) {
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
            ParsedAtom grouped = std::move(inner).value();
            grouped.gbnf = grouped.gbnf.empty() ? std::string("\"\"") : ("(" + grouped.gbnf + ")");
            return grouped;
        }
        if (ch == '[') {
            auto cc = parse_char_class();
            if (!cc.ok()) return cc.status();
            return ParsedAtom{std::move(cc).value(), false, false, 1};
        }
        if (ch == '.') {
            ++pos_;
            return ParsedAtom{std::string("[^\"\\\\\\x7F\\x00-\\x1F]"), false, false, 1};
        }
        if (ch == '\\') {
            auto esc = parse_escape();
            if (!esc.ok()) return esc.status();
            return ParsedAtom{std::move(esc).value(), false, false, 1};
        }
        std::uint32_t cp = 0;
        std::size_t len = 0;
        if (!decode_utf8_cp(pat_, pos_, cp, len)) {
            return Status(ErrorCode::invalid_argument, "invalid UTF-8 sequence in regex pattern");
        }
        const std::string_view atom_sv = pat_.substr(pos_, len);
        pos_ += len;
        if (cp == '"') {
            return ParsedAtom{std::string("\"\\\\\\\"\""), false, false, 1};
        }
        return ParsedAtom{gbnf_literal(atom_sv), false, false, 1};
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
        // RFC 8259 forbids raw unescaped 0x09/0x0A/0x0D inside JSON strings;
        // match space or JSON-escaped whitespace sequences instead.
        if (e == 's') return std::string("(\" \" | \"\\\\t\" | \"\\\\n\" | \"\\\\r\")");
        if (e == 't') return std::string("\"\\\\t\"");
        if (e == 'n') return std::string("\"\\\\n\"");
        if (e == 'r') return std::string("\"\\\\r\"");
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
        bool saw_ws_shorthand = false;
        auto read_class_atom = [&](std::uint32_t& out_cp, std::string& out_utf8, std::string& out_shorthand) -> Status {
            out_cp = 0;
            out_utf8.clear();
            out_shorthand.clear();
            if (pos_ >= pat_.size()) {
                return Status(ErrorCode::invalid_argument, "unclosed '[' in regex pattern");
            }
            const char c = pat_[pos_];
            if (c == '\\') {
                ++pos_;
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
                    // Never emit raw 0x09 tab into a JSON string character class:
                    // negated classes already exclude \x00-\x1F; positive classes
                    // add space here and lift JSON "\\t" into an alternation below.
                    out_shorthand = " ";
                    if (!negated) saw_ws_shorthand = true;
                    return Status::success();
                }
                if (e == '-' || e == ']' || e == '[' || e == '^' || e == '.' || e == '/' || e == '_' || e == '+' ||
                    e == '*' || e == '?' || e == ':' || e == '@') {
                    out_cp = static_cast<unsigned char>(e);
                    out_utf8.assign(1, e);
                    return Status::success();
                }
                return Status(ErrorCode::invalid_argument,
                              std::string("unsupported escape '\\") + e + "' inside regex character class");
            }
            if (c == '"' || c == '\\') {
                return Status(ErrorCode::invalid_argument,
                              "unescaped quote or backslash inside regex character class is not supported");
            }
            std::size_t len = 0;
            if (!decode_utf8_cp(pat_, pos_, out_cp, len)) {
                return Status(ErrorCode::invalid_argument, "invalid UTF-8 sequence inside regex character class");
            }
            out_utf8.assign(pat_.data() + pos_, len);
            pos_ += len;
            return Status::success();
        };

        while (pos_ < pat_.size() && pat_[pos_] != ']') {
            std::uint32_t first_cp = 0;
            std::string first_utf8;
            std::string sh;
            if (Status st = read_class_atom(first_cp, first_utf8, sh); !st.ok()) return st;
            any = true;
            if (!sh.empty()) {
                content += sh;
                continue;
            }
            if (pos_ + 1 < pat_.size() && pat_[pos_] == '-' && pat_[pos_ + 1] != ']') {
                ++pos_;  // consume '-'
                std::uint32_t second_cp = 0;
                std::string second_utf8;
                std::string sh2;
                if (Status st = read_class_atom(second_cp, second_utf8, sh2); !st.ok()) return st;
                if (!sh2.empty()) {
                    return Status(ErrorCode::invalid_argument, "invalid character class range with shorthand");
                }
                if (first_cp > second_cp) {
                    return Status(ErrorCode::invalid_argument, "inverted range in regex character class");
                }
                if (!negated) {
                    if ((first_cp <= static_cast<std::uint32_t>('"') && static_cast<std::uint32_t>('"') <= second_cp) ||
                        (first_cp <= static_cast<std::uint32_t>('\\') && static_cast<std::uint32_t>('\\') <= second_cp) ||
                        (first_cp <= 0x7Fu && 0x7Fu <= second_cp)) {
                        return Status(ErrorCode::invalid_argument,
                                      "character class range crossing '\"', '\\\\', or DEL is not supported in GBNF");
                    }
                    if (first_cp <= 0xDFFFu && second_cp >= 0xD800u) {
                        return Status(ErrorCode::invalid_argument,
                                      "character class range crossing UTF-16 surrogates is not supported in GBNF");
                    }
                }
                content += escape_gbnf_class_atom(first_utf8);
                content.push_back('-');
                content += escape_gbnf_class_atom(second_utf8);
            } else {
                content += escape_gbnf_class_atom(first_utf8);
            }
        }
        if (pos_ >= pat_.size() || pat_[pos_] != ']') {
            return Status(ErrorCode::invalid_argument, "unclosed '[' in regex pattern");
        }
        ++pos_;
        if (!any) {
            return Status(ErrorCode::invalid_argument, "empty character class '[]' in regex pattern");
        }
        const std::string cls = std::string("[") + (negated ? "^" : "") + content + "]";
        if (saw_ws_shorthand && !negated) {
            return "(" + cls + " | \"\\\\t\")";
        }
        return cls;
    }

    Result<std::string> parse_quantifier(bool& out_unbounded, bool& out_can_be_zero, std::size_t& out_max_repeat) {
        out_unbounded = false;
        out_can_be_zero = false;
        out_max_repeat = 1;
        if (pos_ >= pat_.size()) return std::string();
        const char q = pat_[pos_];
        if (q == '*' || q == '+' || q == '?') {
            ++pos_;
            if (pos_ < pat_.size() && (pat_[pos_] == '?' || pat_[pos_] == '+')) {
                return Status(ErrorCode::invalid_argument,
                              "lazy/possessive regex quantifiers are not supported in GBNF");
            }
            out_unbounded = (q == '*' || q == '+');
            out_can_be_zero = (q == '*' || q == '?');
            out_max_repeat = (q == '?') ? 2u : 1u;
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
            out_can_be_zero = (lo == 0u);
            if (!has_comma) {
                out_max_repeat = std::max<std::size_t>(1u, lo);
                return "{" + std::to_string(lo) + "}";
            }
            if (!hi) {
                out_unbounded = true;
                out_max_repeat = std::max<std::size_t>(1u, lo);
                return "{" + std::to_string(lo) + ",}";
            }
            out_max_repeat = std::max<std::size_t>(1u, *hi);
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

    static constexpr std::size_t kMaxClassifyLiterals = 256;

    Result<BranchDomain> classify_branch(const json::Value& schema, std::size_t depth,
                                         std::set<std::string>& ref_guard) const {
        if (depth > kMaxSchemaDepth) {
            return Status(ErrorCode::invalid_argument, "JSON schema nesting exceeds maximum depth");
        }
        if (++classify_visits_ > kMaxSchemaNodes) {
            return Status(ErrorCode::invalid_argument, "JSON schema oneOf analysis exceeds maximum node budget");
        }
        if (schema.is_bool()) {
            BranchDomain d;
            d.type_mask = schema.as_bool() ? kTypeAll : 0u;
            return d;
        }
        if (!schema.is_object()) {
            return Status(ErrorCode::invalid_argument, "JSON schema node must be an object or boolean");
        }
        auto reject_siblings = [&](const char* primary, bool allow_type) -> Status {
            for (const auto& [k, _] : schema.as_object()) {
                if (k == primary || k == "$defs" || k == "definitions" || is_schema_annotation_keyword(k)) {
                    continue;
                }
                if (allow_type && k == "type") {
                    continue;
                }
                return Status(ErrorCode::invalid_argument,
                              std::string("sibling JSON schema constraint '") + k + "' next to '" + primary +
                                  "' cannot be safely enforced by GBNF");
            }
            return Status::success();
        };

        if (const json::Value* ref = schema.find("$ref"); ref != nullptr) {
            if (Status st = reject_siblings("$ref", /*allow_type=*/false); !st.ok()) return st;
            if (!ref->is_string()) {
                return Status(ErrorCode::invalid_argument, "$ref must be a string");
            }
            const std::string& r = ref->as_string();
            if (auto cached = classify_ref_cache_.find(r); cached != classify_ref_cache_.end()) {
                return cached->second;
            }
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
            if (res.ok()) {
                classify_ref_cache_.emplace(r, res.value());
            }
            return res;
        }
        if (const json::Value* c = schema.find("const"); c != nullptr) {
            if (Status st = reject_siblings("const", /*allow_type=*/true); !st.ok()) return st;
            if (const json::Value* type_v = schema.find("type"); type_v != nullptr) {
                auto mask = parse_type_mask(*type_v);
                if (!mask.ok()) return mask.status();
                if (!value_matches_type_mask(*c, mask.value())) {
                    return Status(ErrorCode::invalid_argument, "const value does not match declared schema type");
                }
            }
            BranchDomain d;
            d.is_literal_only = true;
            d.literals.push_back(*c);
            d.type_mask = type_bit_for_value(*c);
            return d;
        }
        if (const json::Value* en = schema.find("enum"); en != nullptr) {
            if (Status st = reject_siblings("enum", /*allow_type=*/true); !st.ok()) return st;
            if (!en->is_array() || en->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "enum must be a non-empty array");
            }
            std::optional<std::uint32_t> required_type_mask;
            if (const json::Value* type_v = schema.find("type"); type_v != nullptr) {
                auto mask = parse_type_mask(*type_v);
                if (!mask.ok()) return mask.status();
                required_type_mask = mask.value();
            }
            BranchDomain d;
            d.is_literal_only = true;
            d.type_mask = 0u;
            for (const auto& item : en->as_array()) {
                if (required_type_mask && !value_matches_type_mask(item, *required_type_mask)) {
                    return Status(ErrorCode::invalid_argument, "enum value does not match declared schema type");
                }
                if (d.literals.size() >= kMaxClassifyLiterals) {
                    d.is_literal_only = false;
                    d.literals.clear();
                } else if (d.is_literal_only) {
                    d.literals.push_back(item);
                }
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
                    combined.literals.clear();
                } else if (combined.is_literal_only) {
                    for (const auto& lit : sd.value().literals) {
                        if (combined.literals.size() >= kMaxClassifyLiterals) {
                            combined.is_literal_only = false;
                            combined.literals.clear();
                            break;
                        }
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
        if (container == nullptr || !container->is_object() || raw_key.empty()) {
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

        auto reject_siblings = [&](const char* primary, bool allow_type) -> Status {
            for (const auto& [k, _] : schema.as_object()) {
                if (k == primary || k == "$defs" || k == "definitions" || is_schema_annotation_keyword(k)) {
                    continue;
                }
                if (allow_type && k == "type") {
                    continue;
                }
                return Status(ErrorCode::invalid_argument,
                              std::string("sibling JSON schema constraint '") + k + "' next to '" + primary +
                                  "' cannot be safely enforced by GBNF");
            }
            return Status::success();
        };

        if (const json::Value* ref = schema.find("$ref"); ref != nullptr) {
            if (Status st = reject_siblings("$ref", /*allow_type=*/false); !st.ok()) return st;
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
                ref_rules_.erase(ref_str);
                return body.status();
            }
            grammar_bytes_ += rule_name.size() + body.value().size() + 6;
            if (grammar_bytes_ > kMaxGrammarBytes) {
                ref_rules_.erase(ref_str);
                return Status(ErrorCode::invalid_argument, "generated GBNF grammar exceeds maximum size");
            }
            rules_.emplace_back(rule_name, std::move(body).value());
            return rule_name;
        }

        if (const json::Value* c = schema.find("const"); c != nullptr) {
            if (Status st = reject_siblings("const", /*allow_type=*/true); !st.ok()) return st;
            if (const json::Value* type_v = schema.find("type"); type_v != nullptr) {
                auto mask = parse_type_mask(*type_v);
                if (!mask.ok()) return mask.status();
                if (!value_matches_type_mask(*c, mask.value())) {
                    return Status(ErrorCode::invalid_argument, "const value does not match declared schema type");
                }
            }
            return gbnf_literal(c->dump()) + " space";
        }

        if (const json::Value* en = schema.find("enum"); en != nullptr) {
            if (Status st = reject_siblings("enum", /*allow_type=*/true); !st.ok()) return st;
            if (!en->is_array() || en->as_array().empty()) {
                return Status(ErrorCode::invalid_argument, "enum must be a non-empty array");
            }
            std::optional<std::uint32_t> required_type_mask;
            if (const json::Value* type_v = schema.find("type"); type_v != nullptr) {
                auto mask = parse_type_mask(*type_v);
                if (!mask.ok()) return mask.status();
                required_type_mask = mask.value();
            }
            std::string expr = "(";
            bool first = true;
            for (const auto& item : en->as_array()) {
                if (required_type_mask && !value_matches_type_mask(item, *required_type_mask)) {
                    return Status(ErrorCode::invalid_argument, "enum value does not match declared schema type");
                }
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
            if (Status st = reject_siblings("anyOf", /*allow_type=*/false); !st.ok()) return st;
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
            if (Status st = reject_siblings("oneOf", /*allow_type=*/false); !st.ok()) return st;
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
            if (const auto exact = v->try_as_int()) {
                if (*exact < -kMaxBoundedIntegerAbs || *exact > kMaxBoundedIntegerAbs) {
                    return Status(ErrorCode::invalid_argument,
                                  std::string(name) + " must be an integer within [-" +
                                      std::to_string(kMaxBoundedIntegerAbs) + ", " +
                                      std::to_string(kMaxBoundedIntegerAbs) + "]");
                }
                out = *exact;
                return Status::success();
            }
            if (!v->is_integer() && v->is_number() && std::isfinite(v->as_double()) &&
                std::floor(v->as_double()) == v->as_double() &&
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
        if (max_len && *max_len == 0) {
            return std::string("\"\\\"\\\"\" space");
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
    mutable std::size_t classify_visits_ = 0;
    mutable std::map<std::string, BranchDomain> classify_ref_cache_;
};

json::Value canonicalize_value(const json::Value& v) {
    if (v.is_array()) {
        json::Array arr;
        const auto& src = v.as_array();
        arr.reserve(src.size());
        for (const auto& item : src) {
            arr.push_back(canonicalize_value(item));
        }
        return json::Value(std::move(arr));
    }
    if (v.is_object()) {
        std::vector<json::Member> members;
        const auto& obj = v.as_object();
        members.reserve(obj.size());
        for (const auto& [k, val] : obj) {
            members.emplace_back(k, canonicalize_value(val));
        }
        std::sort(members.begin(), members.end(),
                  [](const json::Member& a, const json::Member& b) { return a.first < b.first; });
        return json::Value(json::Object::from_members(std::move(members)));
    }
    return v;
}

void hash_combine(std::size_t& seed, std::size_t h) noexcept {
    seed ^= h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

class GbnfStructuredGrammarEngine final : public StructuredGrammarEngine {
public:
    GrammarEngineKind kind() const noexcept override { return GrammarEngineKind::gbnf; }
    std::string_view name() const noexcept override { return "gbnf"; }

    GrammarEngineCapabilities capabilities() const noexcept override {
        GrammarEngineCapabilities caps;
        caps.engine = GrammarEngineKind::gbnf;
        caps.engine_name = "gbnf";
        caps.compiled_in = true;
        caps.sampler_linked = true;
        caps.active = true;
        caps.supports_json_object = true;
        caps.supports_json_schema = true;
        caps.supports_regex = false;
        caps.supports_regex_subset = true;
        caps.supports_structural_tags = false;
        caps.max_schema_depth = kMaxSchemaDepth;
        caps.max_unrolled_bounds = kMaxUnrolledBounds;
        caps.max_bounded_integer_abs = kMaxBoundedIntegerAbs;
        caps.max_grammar_bytes = kMaxGrammarBytes;
        caps.detail =
            "built-in GBNF engine backed by backend GBNF sampler (bounded anchored JSON Schema regex subset only; "
            "no standalone regex or structural tags)";
        return caps;
    }

    Result<CompiledGrammar> compile_json_object() const override {
        const auto t0 = std::chrono::steady_clock::now();
        CompiledGrammar out;
        out.engine = GrammarEngineKind::gbnf;
        out.constraint_kind = GrammarConstraintKind::json_object;
        out.normalized_schema_key = R"({"type":"object"})";
        out.normalized_schema_hash =
            normalized_schema_hash(GrammarConstraintKind::json_object, out.normalized_schema_key);
        out.gbnf = json_object_gbnf();
        out.payload = out.gbnf;
        out.raw_schema = out.normalized_schema_key;
        out.capabilities.engine = GrammarEngineKind::gbnf;
        out.capabilities.constraint_kind = GrammarConstraintKind::json_object;
        out.capabilities.enforces_strict_schema = true;
        out.capabilities.requires_gbnf_sampler = true;
        out.capabilities.sampler_linked = true;
        out.compile_duration = std::chrono::steady_clock::now() - t0;
        return out;
    }

    Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const override {
        const auto t0 = std::chrono::steady_clock::now();
        auto gbnf = json_schema_to_gbnf(schema);
        if (!gbnf.ok()) {
            return gbnf.status();
        }
        CompiledGrammar out;
        out.engine = GrammarEngineKind::gbnf;
        out.constraint_kind = GrammarConstraintKind::json_schema;
        out.normalized_schema_key = canonical_json_dump(schema);
        out.normalized_schema_hash =
            normalized_schema_hash(GrammarConstraintKind::json_schema, out.normalized_schema_key);
        out.gbnf = std::move(gbnf).value();
        out.payload = out.gbnf;
        out.raw_schema = schema.dump();
        out.capabilities.engine = GrammarEngineKind::gbnf;
        out.capabilities.constraint_kind = GrammarConstraintKind::json_schema;
        out.capabilities.enforces_strict_schema = true;
        out.capabilities.requires_gbnf_sampler = true;
        out.capabilities.sampler_linked = true;
        out.compile_duration = std::chrono::steady_clock::now() - t0;
        return out;
    }

    Result<CompiledGrammar> compile_regex(std::string_view pattern) const override {
        (void)pattern;
        return Status(ErrorCode::unsupported,
                      "gbnf grammar engine does not support standalone regex compilation (only bounded anchored "
                      "JSON Schema `pattern` strings via compile_json_schema)");
    }

    Result<CompiledGrammar> compile_structural_tag(std::string_view spec) const override {
        (void)spec;
        return Status(ErrorCode::unsupported, "gbnf grammar engine does not support structural_tag constraints");
    }

    Result<std::unique_ptr<GrammarMatcher>> create_matcher(
        std::shared_ptr<const CompiledGrammar> compiled) const override {
        if (!compiled) {
            return Status(ErrorCode::invalid_argument, "compiled grammar is null");
        }
        if (compiled->engine != GrammarEngineKind::gbnf) {
            return Status(ErrorCode::unsupported, "compiled grammar does not target gbnf engine");
        }
        if (compiled->gbnf.empty() || !compiled->capabilities.sampler_linked) {
            return Status(ErrorCode::invalid_argument, "compiled GBNF grammar has no linked payload");
        }
        return std::make_unique<GrammarMatcher>(std::move(compiled), kMaxGrammarBytes);
    }
};

// Compile-time plug-in wrapper for an external `llguidance` structured decoding engine.
// Honest capability contract: unless an actual native llguidance runtime library is
// compiled AND linked (`SONDER_HAS_LINKED_LLGUIDANCE_SAMPLER`), `sampler_linked`
// and `active` remain false and compilation/matcher requests fail closed with
// ErrorCode::unsupported rather than claiming a guidance sampler is active.
class LlguidanceStructuredGrammarEngine final : public StructuredGrammarEngine {
public:
    GrammarEngineKind kind() const noexcept override { return GrammarEngineKind::llguidance; }
    std::string_view name() const noexcept override { return "llguidance"; }

    GrammarEngineCapabilities capabilities() const noexcept override {
        GrammarEngineCapabilities caps;
        caps.engine = GrammarEngineKind::llguidance;
        caps.engine_name = "llguidance";
#if defined(SONDER_WITH_LLGUIDANCE)
        caps.compiled_in = true;
#else
        caps.compiled_in = false;
#endif
#if defined(SONDER_HAS_LINKED_LLGUIDANCE_SAMPLER) && SONDER_HAS_LINKED_LLGUIDANCE_SAMPLER
        caps.sampler_linked = true;
#else
        caps.sampler_linked = false;
#endif
        caps.active = caps.compiled_in && caps.sampler_linked;
        caps.supports_json_object = false;
        caps.supports_json_schema = false;
        caps.supports_regex = false;
        caps.supports_regex_subset = false;
        caps.supports_structural_tags = false;
        caps.detail = caps.compiled_in
            ? "SONDER_WITH_LLGUIDANCE compile-time hook is enabled, but no llguidance engine/sampler library is "
              "linked in this build"
            : "llguidance grammar engine is not compiled or linked in this build";
        return caps;
    }

    Result<CompiledGrammar> compile_json_object() const override {
        return Status(ErrorCode::unsupported, capabilities().detail);
    }

    Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const override {
        (void)schema;
        return Status(ErrorCode::unsupported, capabilities().detail);
    }

    Result<CompiledGrammar> compile_regex(std::string_view pattern) const override {
        (void)pattern;
        return Status(ErrorCode::unsupported, capabilities().detail);
    }

    Result<CompiledGrammar> compile_structural_tag(std::string_view spec) const override {
        (void)spec;
        return Status(ErrorCode::unsupported, capabilities().detail);
    }

    Result<std::unique_ptr<GrammarMatcher>> create_matcher(
        std::shared_ptr<const CompiledGrammar> compiled) const override {
        (void)compiled;
        return Status(ErrorCode::unsupported, capabilities().detail);
    }
};

}  // namespace

const char* to_string(GrammarEngineKind kind) noexcept {
    switch (kind) {
        case GrammarEngineKind::gbnf: return "gbnf";
        case GrammarEngineKind::llguidance: return "llguidance";
    }
    return "unknown";
}

const char* to_string(GrammarConstraintKind kind) noexcept {
    switch (kind) {
        case GrammarConstraintKind::json_object: return "json_object";
        case GrammarConstraintKind::json_schema: return "json_schema";
        case GrammarConstraintKind::regex: return "regex";
        case GrammarConstraintKind::structural_tag: return "structural_tag";
        case GrammarConstraintKind::tool_call: return "tool_call";
    }
    return "unknown";
}

std::string canonical_json_dump(const json::Value& value) {
    return canonicalize_value(value).dump();
}

std::string build_tokenizer_identity_key(std::string_view backend_name, std::string_view model_id,
                                         const ModelDescriptor* descriptor,
                                         std::optional<std::uint64_t> vocab_size) {
    std::string out = "backend=";
    out.append(backend_name);
    out += "|model=";
    out.append(model_id);
    if (descriptor != nullptr) {
        out += "|fmt=" + descriptor->format;
        out += "|fam=" + descriptor->family;
        out += "|param=" + descriptor->parameter_size;
        out += "|quant=" + descriptor->quantization;
        out += "|bytes=" + std::to_string(descriptor->size_bytes);
        out += "|ctx=" + std::to_string(descriptor->context_length);
        out += "|arch=" + std::to_string(static_cast<int>(descriptor->architecture));
    }
    if (vocab_size) {
        out += "|vocab=" + std::to_string(*vocab_size);
    }
    return out;
}

std::string GrammarCompileOptions::cache_key_fragment() const {
    return std::string("strict=") + (strict ? "1" : "0") +
           "|coerce=" + (coerce_unsupported_tool_args ? "1" : "0") +
           "|fmt=" + grammar_format;
}

std::size_t CompiledGrammar::approx_bytes() const noexcept {
    return sizeof(CompiledGrammar) + normalized_schema_key.size() + normalized_schema_hash.size() +
           tokenizer_identity.size() + gbnf.size() + payload.size() + raw_schema.size();
}

std::string normalized_schema_hash(GrammarConstraintKind kind, std::string_view normalized_key,
                                   const GrammarCompileOptions& options) {
    std::string canonical = std::string(to_string(kind));
    canonical.push_back('\n');
    canonical.append(options.cache_key_fragment());
    canonical.push_back('\n');
    canonical.append(normalized_key);
    return sha256_hex(canonical);
}

GrammarMatcher::GrammarMatcher(std::shared_ptr<const CompiledGrammar> compiled, std::size_t max_bytes)
    : compiled_(std::move(compiled)), max_bytes_(max_bytes) {}

Result<std::string> GrammarMatcher::bind_for_request() {
    if (!compiled_) {
        rejected_ = true;
        last_error_ = "compiled grammar artifact is null";
        return Status(ErrorCode::invalid_argument, last_error_);
    }
    if (!compiled_->capabilities.sampler_linked) {
        rejected_ = true;
        last_error_ = std::string("grammar engine '") + to_string(compiled_->engine) + "' has no linked sampler";
        return Status(ErrorCode::unsupported, last_error_);
    }
    if (compiled_->capabilities.requires_gbnf_sampler && compiled_->gbnf.empty()) {
        rejected_ = true;
        last_error_ = "compiled grammar artifact is missing required GBNF payload";
        return Status(ErrorCode::invalid_argument, last_error_);
    }
    active_ = true;
    finished_ = false;
    rejected_ = false;
    last_error_.clear();
    return compiled_->gbnf;
}

Status GrammarMatcher::accept_chunk(std::string_view piece) {
    if (!active_) {
        last_error_ = "grammar matcher is not bound to an active request";
        return Status(ErrorCode::invalid_argument, last_error_);
    }
    if (finished_ || rejected_) {
        return Status(ErrorCode::invalid_argument, "grammar matcher has already finished or rejected");
    }
    if (bytes_fed_ > max_bytes_ || piece.size() > max_bytes_ - bytes_fed_) {
        rejected_ = true;
        last_error_ = "grammar matcher byte limit exceeded";
        return Status(ErrorCode::invalid_argument, last_error_);
    }
    bytes_fed_ += piece.size();
    ++token_steps_;
    return Status::success();
}

void GrammarMatcher::mark_finished() noexcept {
    if (active_) {
        finished_ = true;
    }
}

void GrammarMatcher::reset() noexcept {
    active_ = false;
    finished_ = false;
    rejected_ = false;
    bytes_fed_ = 0;
    token_steps_ = 0;
    last_error_.clear();
}

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

const StructuredGrammarEngine& llguidance_structured_grammar_engine() noexcept {
    static const LlguidanceStructuredGrammarEngine kEngine;
    return kEngine;
}

const StructuredGrammarEngine& structured_grammar_engine(GrammarEngineKind kind) noexcept {
    switch (kind) {
        case GrammarEngineKind::gbnf: return gbnf_structured_grammar_engine();
        case GrammarEngineKind::llguidance: return llguidance_structured_grammar_engine();
    }
    return gbnf_structured_grammar_engine();
}

const StructuredGrammarEngine& default_structured_grammar_engine() noexcept {
    const auto& ll = llguidance_structured_grammar_engine();
    if (ll.capabilities().active) {
        return ll;
    }
    return gbnf_structured_grammar_engine();
}

Result<std::unique_ptr<GrammarMatcher>> bind_request_grammar_matcher(
    const std::shared_ptr<const CompiledGrammar>& compiled,
    std::string& out_grammar,
    std::string_view explicit_fallback_grammar) {
    if (!compiled) {
        out_grammar.assign(explicit_fallback_grammar.data(), explicit_fallback_grammar.size());
        return std::unique_ptr<GrammarMatcher>{nullptr};
    }
    auto matcher_res = structured_grammar_engine(compiled->engine).create_matcher(compiled);
    if (!matcher_res.ok()) {
        return matcher_res.status();
    }
    std::unique_ptr<GrammarMatcher> matcher = std::move(matcher_res).value();
    auto bound = matcher->bind_for_request();
    if (!bound.ok()) {
        return bound.status();
    }
    out_grammar = std::move(bound).value();
    return matcher;
}

std::size_t CompiledGrammarCacheKeyHash::operator()(const CompiledGrammarCacheKey& k) const noexcept {
    std::size_t seed = std::hash<int>{}(static_cast<int>(k.engine));
    hash_combine(seed, std::hash<int>{}(static_cast<int>(k.constraint_kind)));
    hash_combine(seed, std::hash<std::string>{}(k.tokenizer_identity));
    hash_combine(seed, std::hash<std::string>{}(k.normalized_schema_key));
    hash_combine(seed, std::hash<std::string>{}(k.options_key));
    return seed;
}

CompiledGrammarCache::CompiledGrammarCache(std::size_t max_entries, std::size_t max_bytes, std::size_t max_entry_bytes)
    : max_entries_(max_entries), max_bytes_(max_bytes), max_entry_bytes_(max_entry_bytes) {}

void CompiledGrammarCache::evict_until_fit_locked(std::size_t needed_bytes) noexcept {
    while (!lru_.empty() &&
           (lru_.size() >= max_entries_ || used_bytes_ + needed_bytes > max_bytes_)) {
        const Slot& back = lru_.back();
        used_bytes_ -= back.bytes;
        index_.erase(back.key);
        lru_.pop_back();
        ++evictions_;
    }
}

std::shared_ptr<const CompiledGrammar> CompiledGrammarCache::find(const CompiledGrammarCacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(key);
    if (it == index_.end()) {
        ++misses_;
        return nullptr;
    }
    lru_.splice(lru_.begin(), lru_, it->second);
    ++hits_;
    return it->second->artifact;
}

void CompiledGrammarCache::insert(const CompiledGrammarCacheKey& key,
                                  const std::shared_ptr<const CompiledGrammar>& artifact) {
    if (!artifact) {
        return;
    }
    const std::size_t bytes = artifact->approx_bytes();
    std::lock_guard<std::mutex> lock(mu_);
    if (max_entries_ == 0 || max_bytes_ == 0 || bytes > max_entry_bytes_ || bytes > max_bytes_) {
        ++bypassed_oversize_;
        return;
    }
    if (auto it = index_.find(key); it != index_.end()) {
        used_bytes_ -= it->second->bytes;
        it->second->artifact = artifact;
        it->second->bytes = bytes;
        lru_.splice(lru_.begin(), lru_, it->second);
        evict_until_fit_locked(bytes);
        used_bytes_ += bytes;
        return;
    }
    evict_until_fit_locked(bytes);
    lru_.push_front(Slot{key, artifact, bytes});
    index_.emplace(key, lru_.begin());
    used_bytes_ += bytes;
    ++inserts_;
}

Result<std::shared_ptr<const CompiledGrammar>> CompiledGrammarCache::get_or_compile(
    const CompiledGrammarCacheKey& key,
    const std::function<Result<CompiledGrammar>()>& compiler) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (auto it = index_.find(key); it != index_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            ++hits_;
            return it->second->artifact;
        }
        ++misses_;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto compiled = compiler();
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    if (!compiled.ok()) {
        std::lock_guard<std::mutex> lock(mu_);
        ++compile_failures_;
        return compiled.status();
    }

    CompiledGrammar artifact = std::move(compiled).value();
    if (artifact.normalized_schema_key.empty()) {
        artifact.normalized_schema_key = key.normalized_schema_key;
    }
    if (artifact.tokenizer_identity.empty()) {
        artifact.tokenizer_identity = key.tokenizer_identity;
    }
    if (artifact.normalized_schema_hash.empty()) {
        GrammarCompileOptions default_opts;
        artifact.normalized_schema_hash =
            normalized_schema_hash(key.constraint_kind, artifact.normalized_schema_key, default_opts);
    }
    if (artifact.payload.empty() && !artifact.gbnf.empty()) {
        artifact.payload = artifact.gbnf;
    }
    if (artifact.compile_duration.count() == 0) {
        artifact.compile_duration = elapsed;
    }

    auto immutable = std::make_shared<const CompiledGrammar>(std::move(artifact));
    const std::size_t bytes = immutable->approx_bytes();

    std::lock_guard<std::mutex> lock(mu_);
    if (auto it = index_.find(key); it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second);
        return it->second->artifact;
    }
    if (max_entries_ == 0 || max_bytes_ == 0 || bytes > max_entry_bytes_ || bytes > max_bytes_) {
        ++bypassed_oversize_;
        return immutable;
    }
    evict_until_fit_locked(bytes);
    lru_.push_front(Slot{key, immutable, bytes});
    index_.emplace(key, lru_.begin());
    used_bytes_ += bytes;
    ++inserts_;
    return immutable;
}

Result<std::shared_ptr<const CompiledGrammar>> CompiledGrammarCache::get_or_compile_json_object(
    const StructuredGrammarEngine& engine, std::string_view tokenizer_identity,
    const GrammarCompileOptions& options) {
    CompiledGrammarCacheKey key;
    key.engine = engine.kind();
    key.constraint_kind = GrammarConstraintKind::json_object;
    key.tokenizer_identity = tokenizer_identity.empty() ? std::string("default") : std::string(tokenizer_identity);
    key.normalized_schema_key = R"({"type":"object"})";
    key.options_key = options.cache_key_fragment();
    return get_or_compile(key, [&]() -> Result<CompiledGrammar> {
        auto res = engine.compile_json_object();
        if (!res.ok()) {
            return res.status();
        }
        res.value().tokenizer_identity = key.tokenizer_identity;
        res.value().normalized_schema_hash =
            normalized_schema_hash(GrammarConstraintKind::json_object, key.normalized_schema_key, options);
        return res;
    });
}

Result<std::shared_ptr<const CompiledGrammar>> CompiledGrammarCache::get_or_compile_schema(
    const StructuredGrammarEngine& engine, const json::Value& schema, std::string_view tokenizer_identity,
    const GrammarCompileOptions& options) {
    const std::string canonical = canonical_json_dump(schema);
    CompiledGrammarCacheKey key;
    key.engine = engine.kind();
    key.constraint_kind = GrammarConstraintKind::json_schema;
    key.tokenizer_identity = tokenizer_identity.empty() ? std::string("default") : std::string(tokenizer_identity);
    key.normalized_schema_key = canonical;
    key.options_key = options.cache_key_fragment();
    return get_or_compile(key, [&]() -> Result<CompiledGrammar> {
        auto res = engine.compile_json_schema(schema);
        if (!res.ok()) {
            return res.status();
        }
        res.value().normalized_schema_key = canonical;
        res.value().normalized_schema_hash =
            normalized_schema_hash(GrammarConstraintKind::json_schema, canonical, options);
        res.value().tokenizer_identity = key.tokenizer_identity;
        return res;
    });
}

CompiledGrammarCacheStats CompiledGrammarCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return CompiledGrammarCacheStats{
        lru_.size(), used_bytes_, max_entries_, max_bytes_, max_entry_bytes_,
        hits_, misses_, inserts_, evictions_, bypassed_oversize_, compile_failures_,
    };
}

void CompiledGrammarCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    lru_.clear();
    index_.clear();
    used_bytes_ = 0;
}

}  // namespace sonder::inference::server::detail
