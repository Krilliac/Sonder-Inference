#include "json_schema_normalize.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace sonder::inference::llamacpp_detail {

namespace {

using json::Array;
using json::Member;
using json::Object;
using json::Value;

// Keywords whose value is a single subschema.
constexpr std::string_view kSingleSchemaKeys[] = {
    "additionalProperties", "additionalItems", "contains",      "not",
    "if",                   "then",            "else",          "propertyNames",
    "unevaluatedProperties", "unevaluatedItems", "contentSchema",
};
// Keywords whose value is an array of subschemas. (`items` is handled separately: it is either.)
constexpr std::string_view kSchemaArrayKeys[] = {"anyOf", "oneOf", "allOf", "prefixItems"};
// Keywords whose value is an object mapping names to subschemas.
constexpr std::string_view kSchemaMapKeys[] = {"properties", "patternProperties", "$defs", "definitions",
                                               "dependentSchemas"};

template <std::size_t N>
bool In(const std::string_view (&set)[N], std::string_view key) {
    for (const std::string_view k : set) {
        if (k == key) return true;
    }
    return false;
}

Status Invalid(std::string message) { return Status(ErrorCode::invalid_argument, std::move(message)); }

class Normalizer {
public:
    explicit Normalizer(const JsonSchemaNormalizeLimits& limits) : limits_(limits) {}

    // Normalises `in` (a subschema at `depth`) into `out`. Non-object values (boolean schemas and
    // anything malformed) are copied unchanged.
    Status Schema(const Value& in, int depth, bool allof_branch, Value& out) {
        if (!in.is_object()) {
            out = in;
            return Status::success();
        }
        if (depth > limits_.max_depth) {
            return Invalid("schema nesting exceeds the depth limit of " + std::to_string(limits_.max_depth));
        }
        if (++nodes_ > limits_.max_schema_nodes) {
            return Invalid("schema has more than " + std::to_string(limits_.max_schema_nodes) + " subschemas");
        }
        const Object& src = in.as_object();
        std::vector<Member> members;
        members.reserve(src.size() + 1);
        for (const Member& m : src) {
            Value v;
            if (Status s = Child(m.first, m.second, depth, v); !s.ok()) return s;
            members.emplace_back(m.first, std::move(v));
        }
        if (ShouldClose(src, allof_branch)) {
            members.emplace_back("additionalProperties", Value(false));
        }
        out = Value(Object::from_members(std::move(members)));
        return Status::success();
    }

private:
    // Normalises the member `key` of a schema object at `depth`.
    Status Child(const std::string& key, const Value& v, int depth, Value& out) {
        if (In(kSingleSchemaKeys, key)) return Schema(v, depth + 1, false, out);
        if (key == "items") {
            if (v.is_array()) return SchemaArray(v, depth, false, out);
            return Schema(v, depth + 1, false, out);
        }
        if (In(kSchemaArrayKeys, key)) {
            return SchemaArray(v, depth, /*allof_branch=*/key == "allOf", out);
        }
        if (In(kSchemaMapKeys, key)) return SchemaMap(v, depth, out);
        if (key == "dependencies") return Dependencies(v, depth, out);
        out = v;  // data or annotation keyword: copied verbatim
        return Status::success();
    }

    Status SchemaArray(const Value& v, int depth, bool allof_branch, Value& out) {
        if (!v.is_array()) {
            out = v;
            return Status::success();
        }
        Array arr;
        arr.reserve(v.as_array().size());
        for (const Value& e : v.as_array()) {
            Value n;
            if (Status s = Schema(e, depth + 1, allof_branch, n); !s.ok()) return s;
            arr.push_back(std::move(n));
        }
        out = Value(std::move(arr));
        return Status::success();
    }

    Status SchemaMap(const Value& v, int depth, Value& out) {
        if (!v.is_object()) {
            out = v;
            return Status::success();
        }
        std::vector<Member> members;
        members.reserve(v.as_object().size());
        for (const Member& m : v.as_object()) {
            Value n;
            if (Status s = Schema(m.second, depth + 1, false, n); !s.ok()) return s;
            members.emplace_back(m.first, std::move(n));
        }
        out = Value(Object::from_members(std::move(members)));
        return Status::success();
    }

    // draft-07 `dependencies`: each value is either a subschema or an array of property names.
    Status Dependencies(const Value& v, int depth, Value& out) {
        if (!v.is_object()) {
            out = v;
            return Status::success();
        }
        std::vector<Member> members;
        members.reserve(v.as_object().size());
        for (const Member& m : v.as_object()) {
            Value n;
            if (m.second.is_object()) {
                if (Status s = Schema(m.second, depth + 1, false, n); !s.ok()) return s;
            } else {
                n = m.second;
            }
            members.emplace_back(m.first, std::move(n));
        }
        out = Value(Object::from_members(std::move(members)));
        return Status::success();
    }

    static bool IsObjectSchema(const Object& o) {
        if (const Value* t = o.find("type")) {
            if (t->is_string()) return t->as_string() == "object";
            if (t->is_array()) {
                for (const Value& e : t->as_array()) {
                    if (e.is_string() && e.as_string() == "object") return true;
                }
            }
            return false;
        }
        return o.contains("properties") || o.contains("patternProperties");
    }

    static bool ShouldClose(const Object& o, bool allof_branch) {
        if (allof_branch || !IsObjectSchema(o)) return false;
        if (o.contains("additionalProperties") || o.contains("unevaluatedProperties")) return false;
        if (o.contains("$ref") || o.contains("allOf")) return false;
        return true;
    }

    const JsonSchemaNormalizeLimits& limits_;
    std::size_t nodes_ = 0;
};

}  // namespace

Result<json::Value> NormalizeJsonSchema(const json::Value& schema, const JsonSchemaNormalizeLimits& limits) {
    if (limits.max_depth <= 0) {
        return Invalid("schema depth limit must be positive");
    }
    if (!schema.is_object() && !schema.is_bool()) {
        return Invalid("a JSON schema must be an object or a boolean");
    }
    Normalizer n(limits);
    json::Value out;
    if (Status s = n.Schema(schema, /*depth=*/1, /*allof_branch=*/false, out); !s.ok()) return s;
    return out;
}

Result<std::string> NormalizeJsonSchemaText(std::string_view schema_json, const JsonSchemaNormalizeLimits& limits) {
    if (limits.max_depth <= 0) {
        return Invalid("schema depth limit must be positive");
    }
    if (schema_json.size() > limits.max_input_bytes) {
        return Invalid("schema is larger than " + std::to_string(limits.max_input_bytes) + " bytes");
    }
    json::ParseLimits pl;
    // Two JSON levels per schema level (e.g. properties -> name -> schema) plus slack for data.
    // Clamp before multiplying so oversized limits.max_depth cannot overflow int.
    constexpr int kMaxSafeDepth = (std::numeric_limits<int>::max() - 64) / 2;
    const int safe_depth = limits.max_depth > kMaxSafeDepth ? kMaxSafeDepth : limits.max_depth;
    pl.max_depth = 2 * safe_depth + 64;
    pl.max_values = limits.max_json_values;
    auto parsed = json::parse(schema_json, pl);
    if (!parsed.ok()) return Invalid("schema is not valid JSON: " + parsed.status().message());
    auto normalized = NormalizeJsonSchema(parsed.value(), limits);
    if (!normalized.ok()) return normalized.status();
    return normalized.value().dump();
}

Result<std::string> LlguidanceSchemaForResponseFormat(const std::optional<std::string>& type,
                                                      const std::optional<std::string>& schema_text,
                                                      const JsonSchemaNormalizeLimits& limits) {
    if (!type.has_value() || *type != "json_schema") return std::string();
    if (!schema_text.has_value() || schema_text->empty()) return std::string();
    return NormalizeJsonSchemaText(*schema_text, limits);
}

}  // namespace sonder::inference::llamacpp_detail
