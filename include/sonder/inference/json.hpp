// Sonder Inference: small JSON value type used for telemetry, benchmark
// results, and backend wire formats. Objects preserve insertion order so that
// emitted envelopes are stable and diffable.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference::json {

class Value;
using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;

class Object {
public:
    Object() = default;
    Object(std::initializer_list<Member> members);

    Value& operator[](std::string_view key);  // insert-or-get
    [[nodiscard]] const Value* find(std::string_view key) const;
    [[nodiscard]] Value* find(std::string_view key);
    [[nodiscard]] bool contains(std::string_view key) const { return find(key) != nullptr; }
    void set(std::string key, Value value);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::vector<Member>::const_iterator begin() const;
    [[nodiscard]] std::vector<Member>::const_iterator end() const;

private:
    std::vector<Member> members_;
};

enum class Type { null, boolean, integer, number, string, array, object };

class Value {
public:
    Value() = default;
    Value(std::nullptr_t) {}                                   // NOLINT
    Value(bool b) : data_(b) {}                                // NOLINT
    Value(int i) : data_(static_cast<std::int64_t>(i)) {}      // NOLINT
    Value(long i) : data_(static_cast<std::int64_t>(i)) {}     // NOLINT
    Value(long long i) : data_(static_cast<std::int64_t>(i)) {}  // NOLINT
    Value(unsigned i) : data_(static_cast<std::int64_t>(i)) {}   // NOLINT
    Value(unsigned long i) : data_(static_cast<std::int64_t>(i)) {}  // NOLINT
    Value(unsigned long long i) : data_(static_cast<std::int64_t>(i)) {}  // NOLINT
    Value(double d) : data_(d) {}                              // NOLINT
    Value(float d) : data_(static_cast<double>(d)) {}          // NOLINT
    Value(const char* s) : data_(std::string(s)) {}            // NOLINT
    Value(std::string s) : data_(std::move(s)) {}              // NOLINT
    Value(std::string_view s) : data_(std::string(s)) {}       // NOLINT
    Value(Array a) : data_(std::move(a)) {}                    // NOLINT
    Value(Object o) : data_(std::move(o)) {}                   // NOLINT
    template <class T>
    Value(const std::optional<T>& opt) {                       // NOLINT
        if (opt) {
            *this = Value(*opt);
        }
    }

    [[nodiscard]] Type type() const noexcept { return static_cast<Type>(data_.index()); }
    [[nodiscard]] bool is_null() const noexcept { return type() == Type::null; }
    [[nodiscard]] bool is_bool() const noexcept { return type() == Type::boolean; }
    [[nodiscard]] bool is_integer() const noexcept { return type() == Type::integer; }
    [[nodiscard]] bool is_number() const noexcept { return type() == Type::integer || type() == Type::number; }
    [[nodiscard]] bool is_string() const noexcept { return type() == Type::string; }
    [[nodiscard]] bool is_array() const noexcept { return type() == Type::array; }
    [[nodiscard]] bool is_object() const noexcept { return type() == Type::object; }

    [[nodiscard]] bool as_bool(bool fallback = false) const noexcept;
    [[nodiscard]] std::int64_t as_int(std::int64_t fallback = 0) const noexcept;
    [[nodiscard]] double as_double(double fallback = 0.0) const noexcept;
    [[nodiscard]] const std::string& as_string() const;  // empty string if not a string
    [[nodiscard]] const Array& as_array() const;         // empty array if not an array
    [[nodiscard]] const Object& as_object() const;       // empty object if not an object
    Array& array();    // converts null to array
    Object& object();  // converts null to object

    // Object convenience: returns nullptr when absent or not an object.
    [[nodiscard]] const Value* find(std::string_view key) const;

    [[nodiscard]] std::string dump() const;  // compact, single line
    void dump_to(std::string& out) const;

private:
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object> data_{nullptr};
};

// Parses exactly one JSON document (surrounding whitespace allowed).
Result<Value> parse(std::string_view text);

// Appends a JSON string literal (with quotes) for `s` to `out`.
void append_escaped(std::string& out, std::string_view s);

}  // namespace sonder::inference::json
