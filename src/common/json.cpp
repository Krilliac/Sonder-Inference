#include "sonder/inference/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace sonder::inference::json {

// ----------------------------------------------------------------- Object

Object::Object(std::initializer_list<Member> members) {
    members_.reserve(members.size());
    for (const auto& m : members) {
        set(m.first, m.second);
    }
}

std::size_t Object::size() const noexcept { return members_.size(); }
bool Object::empty() const noexcept { return members_.empty(); }
std::vector<Member>::const_iterator Object::begin() const { return members_.begin(); }
std::vector<Member>::const_iterator Object::end() const { return members_.end(); }

Value& Object::operator[](std::string_view key) {
    if (Value* v = find(key)) {
        return *v;
    }
    members_.emplace_back(std::string(key), Value());
    return members_.back().second;
}

const Value* Object::find(std::string_view key) const {
    for (const auto& m : members_) {
        if (m.first == key) {
            return &m.second;
        }
    }
    return nullptr;
}

Value* Object::find(std::string_view key) {
    for (auto& m : members_) {
        if (m.first == key) {
            return &m.second;
        }
    }
    return nullptr;
}

Object Object::from_members(std::vector<Member> members) {
    Object out;
    constexpr std::size_t kLinearThreshold = 16;
    if (members.size() <= kLinearThreshold) {
        out.members_.reserve(members.size());
        for (auto& m : members) {
            out.set(std::move(m.first), std::move(m.second));
        }
        return out;
    }
    // Reserve up front so the keys (and the string_views into them) never move.
    out.members_.reserve(members.size());
    std::unordered_map<std::string_view, std::size_t> index;
    index.reserve(members.size());
    for (auto& m : members) {
        if (auto it = index.find(m.first); it != index.end()) {
            out.members_[it->second].second = std::move(m.second);
            continue;
        }
        out.members_.emplace_back(std::move(m.first), std::move(m.second));
        index.emplace(out.members_.back().first, out.members_.size() - 1);
    }
    return out;
}

void Object::set(std::string key, Value value) {
    if (Value* v = find(key)) {
        *v = std::move(value);
        return;
    }
    members_.emplace_back(std::move(key), std::move(value));
}

// ------------------------------------------------------------------ Value

namespace {
const std::string kEmptyString;
const Array kEmptyArray;
const Object kEmptyObject;
}  // namespace

bool Value::as_bool(bool fallback) const noexcept {
    if (const bool* b = std::get_if<bool>(&data_)) {
        return *b;
    }
    return fallback;
}

std::int64_t Value::as_int(std::int64_t fallback) const noexcept {
    if (const auto* i = std::get_if<std::int64_t>(&data_)) {
        return *i;
    }
    if (const auto* d = std::get_if<double>(&data_)) {
        if (std::isfinite(*d) && *d >= -9.2e18 && *d <= 9.2e18) {
            return static_cast<std::int64_t>(*d);
        }
    }
    return fallback;
}

double Value::as_double(double fallback) const noexcept {
    if (const auto* d = std::get_if<double>(&data_)) {
        return *d;
    }
    if (const auto* i = std::get_if<std::int64_t>(&data_)) {
        return static_cast<double>(*i);
    }
    return fallback;
}

const std::string& Value::as_string() const {
    if (const auto* s = std::get_if<std::string>(&data_)) {
        return *s;
    }
    return kEmptyString;
}

const Array& Value::as_array() const {
    if (const auto* a = std::get_if<Array>(&data_)) {
        return *a;
    }
    return kEmptyArray;
}

const Object& Value::as_object() const {
    if (const auto* o = std::get_if<Object>(&data_)) {
        return *o;
    }
    return kEmptyObject;
}

Array& Value::array() {
    if (!is_array()) {
        data_ = Array{};
    }
    return std::get<Array>(data_);
}

Object& Value::object() {
    if (!is_object()) {
        data_ = Object{};
    }
    return std::get<Object>(data_);
}

const Value* Value::find(std::string_view key) const {
    if (const auto* o = std::get_if<Object>(&data_)) {
        return o->find(key);
    }
    return nullptr;
}

namespace {
// Length of the well-formed UTF-8 sequence starting at s[i] (lead byte >= 0x80),
// or 0 if it is ill-formed (Unicode Table 3-7: no overlongs, no surrogates,
// nothing above U+10FFFF, no truncation).
std::size_t utf8_sequence_length(std::string_view s, std::size_t i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t n = 0;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        n = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        n = 3;
        if (b0 == 0xE0) lo = 0xA0;
        if (b0 == 0xED) hi = 0x9F;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        n = 4;
        if (b0 == 0xF0) lo = 0x90;
        if (b0 == 0xF4) hi = 0x8F;
    } else {
        return 0;
    }
    if (i + n > s.size()) return 0;
    const auto b1 = static_cast<unsigned char>(s[i + 1]);
    if (b1 < lo || b1 > hi) return 0;
    for (std::size_t k = 2; k < n; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if (b < 0x80 || b > 0xBF) return 0;
    }
    return n;
}
}  // namespace

void append_escaped(std::string& out, std::string_view s) {
    out.push_back('"');
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else if (c < 0x80) {
                    out.push_back(ch);
                } else {
                    const std::size_t n = utf8_sequence_length(s, i);
                    if (n == 0) {
                        out += "\xEF\xBF\xBD";  // U+FFFD for one ill-formed byte
                    } else {
                        out.append(s.substr(i, n));
                        i += n - 1;
                    }
                }
        }
    }
    out.push_back('"');
}

namespace {
void append_double(std::string& out, double d) {
    if (!std::isfinite(d)) {
        out += "null";  // JSON has no NaN/Inf
        return;
    }
    char buf[32];
    auto res = std::to_chars(buf, buf + sizeof(buf), d);
    std::string_view sv(buf, static_cast<std::size_t>(res.ptr - buf));
    out.append(sv);
    // Keep the value recognisably floating point.
    if (sv.find_first_of(".eE") == std::string_view::npos) {
        out += ".0";
    }
}
}  // namespace

void Value::dump_to(std::string& out) const {
    switch (type()) {
        case Type::null: out += "null"; break;
        case Type::boolean: out += std::get<bool>(data_) ? "true" : "false"; break;
        case Type::integer: {
            char buf[24];
            auto res = std::to_chars(buf, buf + sizeof(buf), std::get<std::int64_t>(data_));
            out.append(buf, static_cast<std::size_t>(res.ptr - buf));
            break;
        }
        case Type::number: append_double(out, std::get<double>(data_)); break;
        case Type::string: append_escaped(out, std::get<std::string>(data_)); break;
        case Type::array: {
            out.push_back('[');
            bool first = true;
            for (const auto& v : std::get<Array>(data_)) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                v.dump_to(out);
            }
            out.push_back(']');
            break;
        }
        case Type::object: {
            out.push_back('{');
            bool first = true;
            for (const auto& [k, v] : std::get<Object>(data_)) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                append_escaped(out, k);
                out.push_back(':');
                v.dump_to(out);
            }
            out.push_back('}');
            break;
        }
    }
}

std::string Value::dump() const {
    std::string out;
    dump_to(out);
    return out;
}

// ----------------------------------------------------------------- Parser

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Result<Value> parse_document() {
        skip_ws();
        Value v;
        if (!parse_value(v, 0)) {
            return fail();
        }
        skip_ws();
        if (pos_ != text_.size()) {
            error_ = "trailing characters";
            return fail();
        }
        return v;
    }

private:
    static constexpr int kMaxDepth = 128;

    Status fail() const {
        return Status(ErrorCode::protocol_error,
                      "json parse error at offset " + std::to_string(pos_) + ": " + error_);
    }

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool consume_literal(std::string_view lit) {
        if (text_.substr(pos_, lit.size()) == lit) {
            pos_ += lit.size();
            return true;
        }
        error_ = "invalid literal";
        return false;
    }

    bool parse_value(Value& out, int depth) {
        if (depth > kMaxDepth) {
            error_ = "nesting too deep";
            return false;
        }
        if (pos_ >= text_.size()) {
            error_ = "unexpected end of input";
            return false;
        }
        const char c = text_[pos_];
        switch (c) {
            case 'n': if (!consume_literal("null")) return false; out = Value(); return true;
            case 't': if (!consume_literal("true")) return false; out = Value(true); return true;
            case 'f': if (!consume_literal("false")) return false; out = Value(false); return true;
            case '"': {
                std::string s;
                if (!parse_string(s)) return false;
                out = Value(std::move(s));
                return true;
            }
            case '[': return parse_array(out, depth);
            case '{': return parse_object(out, depth);
            default:
                if (c == '-' || (c >= '0' && c <= '9')) {
                    return parse_number(out);
                }
                error_ = "unexpected character";
                return false;
        }
    }

    bool parse_array(Value& out, int depth) {
        ++pos_;  // [
        Array arr;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            out = Value(std::move(arr));
            return true;
        }
        while (true) {
            skip_ws();
            Value v;
            if (!parse_value(v, depth + 1)) return false;
            arr.push_back(std::move(v));
            skip_ws();
            if (pos_ >= text_.size()) { error_ = "unterminated array"; return false; }
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == ']') { ++pos_; break; }
            error_ = "expected ',' or ']'";
            return false;
        }
        out = Value(std::move(arr));
        return true;
    }

    bool parse_object(Value& out, int depth) {
        ++pos_;  // {
        // Collect members first and deduplicate once at the end:
        // Object::set() per member would make parsing O(n^2) in the key count.
        std::vector<Member> members;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            out = Value(Object{});
            return true;
        }
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') { error_ = "expected object key"; return false; }
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ':') { error_ = "expected ':'"; return false; }
            ++pos_;
            skip_ws();
            Value v;
            if (!parse_value(v, depth + 1)) return false;
            members.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (pos_ >= text_.size()) { error_ = "unterminated object"; return false; }
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == '}') { ++pos_; break; }
            error_ = "expected ',' or '}'";
            return false;
        }
        out = Value(Object::from_members(std::move(members)));
        return true;
    }

    static void append_utf8(std::string& s, std::uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool parse_hex4(std::uint32_t& out) {
        if (pos_ + 4 > text_.size()) { error_ = "truncated \\u escape"; return false; }
        out = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') out |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= static_cast<std::uint32_t>(c - 'A' + 10);
            else { error_ = "invalid \\u escape"; return false; }
        }
        return true;
    }

    bool parse_string(std::string& out) {
        ++pos_;  // opening quote
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) { error_ = "control character in string"; return false; }
            if (static_cast<unsigned char>(c) >= 0x80) {
                // Raw non-ASCII: keep well-formed UTF-8, replace each ill-formed
                // byte with U+FFFD so parsed strings are always valid UTF-8
                // (and parse(dump(v)) is a fixed point).
                const std::size_t n = utf8_sequence_length(text_, pos_ - 1);
                if (n == 0) {
                    out += "\xEF\xBF\xBD";
                } else {
                    out.append(text_.substr(pos_ - 1, n));
                    pos_ += n - 1;
                }
                continue;
            }
            if (c != '\\') { out.push_back(c); continue; }
            if (pos_ >= text_.size()) break;
            const char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t cp = 0;
                    if (!parse_hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 2 <= text_.size() && text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            std::uint32_t lo = 0;
                            if (!parse_hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                // Unpaired high surrogate: replace it alone and
                                // re-read the following escape on its own.
                                cp = 0xFFFD;
                                pos_ -= 6;
                            }
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: error_ = "invalid escape"; return false;
            }
        }
        error_ = "unterminated string";
        return false;
    }

    bool parse_number(Value& out) {
        const std::size_t start = pos_;
        bool is_float = false;
        if (text_[pos_] == '-') ++pos_;
        if (pos_ >= text_.size()) { error_ = "invalid number"; return false; }
        if (text_[pos_] == '0') {
            ++pos_;
        } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        } else {
            error_ = "invalid number";
            return false;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            is_float = true;
            ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') { error_ = "invalid fraction"; return false; }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') { error_ = "invalid exponent"; return false; }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
        }
        const std::string token(text_.substr(start, pos_ - start));
        if (!is_float) {
            std::int64_t i = 0;
            auto res = std::from_chars(token.data(), token.data() + token.size(), i);
            if (res.ec == std::errc() && res.ptr == token.data() + token.size()) {
                out = Value(i);
                return true;
            }
            // Out of int64 range: fall through to double.
        }
        char* end = nullptr;
        const double d = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) { error_ = "invalid number"; return false; }
        if (!std::isfinite(d)) { error_ = "number out of range"; return false; }
        out = Value(d);
        return true;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    std::string error_;
};

}  // namespace

Result<Value> parse(std::string_view text) { return Parser(text).parse_document(); }

}  // namespace sonder::inference::json
