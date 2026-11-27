// Pure tests for the llguidance JSON-Schema normaliser (no model, no llama.cpp calls).
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "json_schema_normalize.hpp"
#include <doctest/doctest.h>

using namespace sonder::inference;
using namespace sonder::inference::llamacpp_detail;

namespace {

// Normalises `in` and returns the compact result, or "ERR:<message>" on failure.
std::string Norm(std::string_view in, const JsonSchemaNormalizeLimits& limits = {}) {
    auto r = NormalizeJsonSchemaText(in, limits);
    if (!r.ok()) return "ERR:" + r.status().message();
    return r.value();
}

bool IsInvalidArgument(std::string_view in, const JsonSchemaNormalizeLimits& limits = {}) {
    auto r = NormalizeJsonSchemaText(in, limits);
    return !r.ok() && r.status().code() == ErrorCode::invalid_argument;
}

// A chain of `n` nested object schemas: depth n.
std::string NestedObjects(int n) {
    if (n <= 1) return R"({"type":"object"})";
    return R"({"type":"object","properties":{"a":)" + NestedObjects(n - 1) + "}}";
}

constexpr const char* kClosed = R"("additionalProperties":false)";

}  // namespace

TEST_CASE("schema normalize: a top-level object is closed") {
    CHECK(Norm(R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})") ==
          R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false})");
}

TEST_CASE("schema normalize: nested objects are closed at every level") {
    CHECK(Norm(R"({"type":"object","properties":{"o":{"type":"object","properties":{"i":{"type":"object","properties":{}}}}}})") ==
          R"({"type":"object","properties":{"o":{"type":"object","properties":{"i":{"type":"object","properties":{},"additionalProperties":false}},"additionalProperties":false}},"additionalProperties":false})");
}

TEST_CASE("schema normalize: arrays of objects (items, prefixItems) are closed") {
    CHECK(Norm(R"({"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"}}}})") ==
          R"({"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false}})");
    CHECK(Norm(R"({"type":"array","prefixItems":[{"type":"object"},{"type":"string"}]})") ==
          R"({"type":"array","prefixItems":[{"type":"object","additionalProperties":false},{"type":"string"}]})");
    // Tuple-form items.
    CHECK(Norm(R"({"type":"array","items":[{"type":"object"}]})") ==
          R"({"type":"array","items":[{"type":"object","additionalProperties":false}]})");
}

TEST_CASE("schema normalize: $defs are closed where defined and $ref nodes are left alone") {
    CHECK(Norm(R"({"$defs":{"pt":{"type":"object","properties":{"x":{"type":"number"}}}},"type":"object","properties":{"p":{"$ref":"#/$defs/pt"}}})") ==
          R"({"$defs":{"pt":{"type":"object","properties":{"x":{"type":"number"}},"additionalProperties":false}},"type":"object","properties":{"p":{"$ref":"#/$defs/pt"}},"additionalProperties":false})");
    // A $ref node that also says type: object is not closed (it would forbid the referenced
    // schema's properties).
    CHECK(Norm(R"({"type":"object","$ref":"#/$defs/a"})") == R"({"type":"object","$ref":"#/$defs/a"})");
    // Legacy `definitions` behaves like $defs.
    CHECK(Norm(R"({"definitions":{"d":{"type":"object"}},"$ref":"#/definitions/d"})") ==
          R"({"definitions":{"d":{"type":"object","additionalProperties":false}},"$ref":"#/definitions/d"})");
    // A recursive $ref does not loop: refs are never expanded.
    CHECK(Norm(R"({"$defs":{"n":{"type":"object","properties":{"next":{"$ref":"#/$defs/n"}}}},"$ref":"#/$defs/n"})") ==
          R"({"$defs":{"n":{"type":"object","properties":{"next":{"$ref":"#/$defs/n"}},"additionalProperties":false}},"$ref":"#/$defs/n"})");
}

TEST_CASE("schema normalize: anyOf / oneOf branches are each closed") {
    CHECK(Norm(R"({"anyOf":[{"type":"object","properties":{"a":{}}},{"type":"null"}]})") ==
          R"({"anyOf":[{"type":"object","properties":{"a":{}},"additionalProperties":false},{"type":"null"}]})");
}

TEST_CASE("schema normalize: oneOf keeps both branches closed") {
    CHECK(Norm(R"({"oneOf":[{"type":"object"},{"type":"object","properties":{"b":{}}}]})") ==
          R"({"oneOf":[{"type":"object","additionalProperties":false},{"type":"object","properties":{"b":{}},"additionalProperties":false}]})");
}

TEST_CASE("schema normalize: an explicit additionalProperties is kept as is") {
    CHECK(Norm(R"({"type":"object","additionalProperties":true})") == R"({"type":"object","additionalProperties":true})");
    CHECK(Norm(R"({"type":"object","additionalProperties":false})") == R"({"type":"object","additionalProperties":false})");
    // A schema-valued additionalProperties is kept, and the schema inside is itself normalised.
    CHECK(Norm(R"({"type":"object","additionalProperties":{"type":"object","properties":{"v":{}}}})") ==
          R"({"type":"object","additionalProperties":{"type":"object","properties":{"v":{}},"additionalProperties":false}})");
    // Explicit on an inner object only: the outer one is still closed.
    CHECK(Norm(R"({"type":"object","properties":{"free":{"type":"object","additionalProperties":true}}})") ==
          R"({"type":"object","properties":{"free":{"type":"object","additionalProperties":true}},"additionalProperties":false})");
    // unevaluatedProperties counts as an explicit statement too.
    CHECK(Norm(R"({"type":"object","unevaluatedProperties":false})") == R"({"type":"object","unevaluatedProperties":false})");
}

TEST_CASE("schema normalize: non-object schemas and boolean schemas are unchanged") {
    CHECK(Norm(R"({"type":"string","minLength":1})") == R"({"type":"string","minLength":1})");
    CHECK(Norm(R"({"type":["string","null"]})") == R"({"type":["string","null"]})");
    CHECK(Norm("{}") == "{}");
    CHECK(Norm("true") == "true");
    CHECK(Norm("false") == "false");
    // No type and no object keywords: not an object schema.
    CHECK(Norm(R"({"description":"anything"})") == R"({"description":"anything"})");
}

TEST_CASE("schema normalize: object detection covers type arrays and type-less properties") {
    CHECK(Norm(R"({"type":["object","null"],"properties":{"a":{}}})") ==
          R"({"type":["object","null"],"properties":{"a":{}},"additionalProperties":false})");
    CHECK(Norm(R"({"properties":{"a":{}}})") == R"({"properties":{"a":{}},"additionalProperties":false})");
    CHECK(Norm(R"({"patternProperties":{"^x":{"type":"object"}}})") ==
          R"({"patternProperties":{"^x":{"type":"object","additionalProperties":false}},"additionalProperties":false})");
}

TEST_CASE("schema normalize: enum / const / default / examples are data and are never rewritten") {
    const std::string in =
        R"({"type":"object","properties":{"a":{"type":"string","enum":[{"type":"object"}],"default":{"type":"object","properties":{}},"const":{"type":"object"},"examples":[{"type":"object"}]}}})";
    CHECK(Norm(in) ==
          R"({"type":"object","properties":{"a":{"type":"string","enum":[{"type":"object"}],"default":{"type":"object","properties":{}},"const":{"type":"object"},"examples":[{"type":"object"}]}},"additionalProperties":false})");
}

TEST_CASE("schema normalize: property names that collide with keywords are property names") {
    CHECK(Norm(R"({"type":"object","properties":{"properties":{"type":"object"},"type":{"type":"string"},"additionalProperties":{"type":"object"}}})") ==
          R"({"type":"object","properties":{"properties":{"type":"object","additionalProperties":false},"type":{"type":"string"},"additionalProperties":{"type":"object","additionalProperties":false}},"additionalProperties":false})");
}

TEST_CASE("schema normalize: allOf schemas and their branches are not closed") {
    // Closing a branch would forbid the sibling branch's properties.
    CHECK(Norm(R"({"type":"object","allOf":[{"type":"object","properties":{"a":{}}},{"type":"object","properties":{"b":{}}}]})") ==
          R"({"type":"object","allOf":[{"type":"object","properties":{"a":{}}},{"type":"object","properties":{"b":{}}}]})");
    // Objects nested inside the branches are still closed.
    CHECK(Norm(R"({"allOf":[{"type":"object","properties":{"n":{"type":"object"}}}]})") ==
          R"({"allOf":[{"type":"object","properties":{"n":{"type":"object","additionalProperties":false}}}]})");
}

TEST_CASE("schema normalize: not / if / then / else / contains / propertyNames / dependentSchemas") {
    CHECK(Norm(R"({"not":{"type":"object"}})") == R"({"not":{"type":"object","additionalProperties":false}})");
    CHECK(Norm(R"({"if":{"type":"object"},"then":{"type":"object"},"else":{"type":"object"}})") ==
          R"({"if":{"type":"object","additionalProperties":false},"then":{"type":"object","additionalProperties":false},"else":{"type":"object","additionalProperties":false}})");
    CHECK(Norm(R"({"type":"array","contains":{"type":"object"}})") ==
          R"({"type":"array","contains":{"type":"object","additionalProperties":false}})");
    CHECK(Norm(R"({"dependentSchemas":{"a":{"type":"object"}},"dependencies":{"b":["c"],"d":{"type":"object"}}})") ==
          R"({"dependentSchemas":{"a":{"type":"object","additionalProperties":false}},"dependencies":{"b":["c"],"d":{"type":"object","additionalProperties":false}}})");
}

TEST_CASE("schema normalize: idempotent, and the input value is not modified") {
    const std::string in =
        R"({"type":"object","properties":{"a":{"type":"array","items":{"type":"object"}},"b":{"anyOf":[{"type":"object"},{"type":"null"}]}},"$defs":{"d":{"type":"object"}}})";
    const std::string once = Norm(in);
    REQUIRE(once.rfind("ERR:", 0) == std::string::npos);
    CHECK(Norm(once) == once);

    auto parsed = json::parse(in);
    REQUIRE(parsed.ok());
    const std::string before = parsed.value().dump();
    auto out = NormalizeJsonSchema(parsed.value());
    REQUIRE(out.ok());
    CHECK(parsed.value().dump() == before);
    CHECK(out.value().dump() == once);
    CHECK(out.value().dump().find(kClosed) != std::string::npos);
}

TEST_CASE("schema normalize: invalid input is invalid_argument") {
    CHECK(IsInvalidArgument(""));
    CHECK(IsInvalidArgument("not json"));
    CHECK(IsInvalidArgument(R"({"type":"object")"));  // truncated
    CHECK(IsInvalidArgument("[]"));
    CHECK(IsInvalidArgument("[{\"type\":\"object\"}]"));
    CHECK(IsInvalidArgument("\"object\""));
    CHECK(IsInvalidArgument("42"));
    CHECK(IsInvalidArgument("null"));
}

TEST_CASE("schema normalize: depth bound is exact") {
    JsonSchemaNormalizeLimits lim;
    lim.max_depth = 5;
    CHECK(Norm(NestedObjects(5), lim).rfind("ERR:", 0) == std::string::npos);
    CHECK(IsInvalidArgument(NestedObjects(6), lim));
    // Nesting through other applicators counts the same way.
    CHECK(IsInvalidArgument(R"({"anyOf":[{"anyOf":[{"anyOf":[{"anyOf":[{"anyOf":[{"type":"object"}]}]}]}]}]})", lim));
    // Default limit: 64 passes, 65 fails.
    CHECK(Norm(NestedObjects(64)).rfind("ERR:", 0) == std::string::npos);
    CHECK(IsInvalidArgument(NestedObjects(65)));
    // Non-positive depth limits are rejected; extreme positive limits clamp safely without overflow.
    for (int bad_depth : {0, -1, std::numeric_limits<int>::min()}) {
        JsonSchemaNormalizeLimits nonpos;
        nonpos.max_depth = bad_depth;
        CHECK(IsInvalidArgument(R"({"type":"object"})", nonpos));
        auto parsed = json::parse(R"({"type":"object"})");
        REQUIRE(parsed.ok());
        auto r = NormalizeJsonSchema(parsed.value(), nonpos);
        CHECK_FALSE(r.ok());
        CHECK(r.status().code() == ErrorCode::invalid_argument);
    }
    JsonSchemaNormalizeLimits huge;
    huge.max_depth = std::numeric_limits<int>::max();
    CHECK(Norm(NestedObjects(40), huge).rfind("ERR:", 0) == std::string::npos);
}

TEST_CASE("schema normalize: hostile inputs are rejected without crashing") {
    // Extremely deep nesting is refused by the bounded parser or by the depth limit.
    std::string deep;
    for (int i = 0; i < 20000; ++i) deep += R"({"properties":{"a":)";
    deep += "{}";
    for (int i = 0; i < 20000; ++i) deep += "}}";
    CHECK(IsInvalidArgument(deep));

    // Deep nesting inside data keywords (never walked) is still bounded by the parser.
    std::string deep_enum = R"({"enum":)";
    for (int i = 0; i < 20000; ++i) deep_enum += "[";
    for (int i = 0; i < 20000; ++i) deep_enum += "]";
    deep_enum += "}";
    CHECK(IsInvalidArgument(deep_enum));

    // Node cap.
    JsonSchemaNormalizeLimits few;
    few.max_schema_nodes = 10;
    std::string wide = R"({"type":"object","properties":{)";
    for (int i = 0; i < 50; ++i) wide += (i ? "," : "") + std::string("\"p") + std::to_string(i) + R"(":{"type":"string"})";
    wide += "}}";
    CHECK(IsInvalidArgument(wide, few));
    CHECK(Norm(wide).rfind("ERR:", 0) == std::string::npos);

    // Input size cap, checked before parsing.
    JsonSchemaNormalizeLimits small;
    small.max_input_bytes = 16;
    CHECK(IsInvalidArgument(R"({"type":"object","properties":{"a":{}}})", small));

    // Value cap of the parser.
    JsonSchemaNormalizeLimits values;
    values.max_json_values = 5;
    CHECK(IsInvalidArgument(R"({"type":"object","properties":{"a":{},"b":{},"c":{}}})", values));
}

TEST_CASE("schema normalize: wide schemas stay linear and exact") {
    std::string wide = R"({"type":"object","properties":{)";
    constexpr int kProps = 5000;
    for (int i = 0; i < kProps; ++i) {
        wide += (i ? "," : "") + std::string("\"p") + std::to_string(i) + R"(":{"type":"object"})";
    }
    wide += "}}";
    const std::string out = Norm(wide);
    REQUIRE(out.rfind("ERR:", 0) == std::string::npos);
    std::size_t closed = 0;
    for (std::size_t pos = out.find(kClosed); pos != std::string::npos; pos = out.find(kClosed, pos + 1)) ++closed;
    CHECK(closed == static_cast<std::size_t>(kProps) + 1);
}

TEST_CASE("schema normalize: response_format selects the llguidance path only for json_schema") {
    const std::optional<std::string> schema = R"({"type":"object","properties":{"a":{}}})";
    const std::string closed = R"({"type":"object","properties":{"a":{}},"additionalProperties":false})";

    auto r = LlguidanceSchemaForResponseFormat(std::string("json_schema"), schema);
    REQUIRE(r.ok());
    CHECK(r.value() == closed);

    // Not json_schema, or no schema text: empty = "do not use llguidance" (GBNF path unchanged).
    for (const char* t : {"text", "json_object", ""}) {
        auto x = LlguidanceSchemaForResponseFormat(std::string(t), schema);
        REQUIRE(x.ok());
        CHECK(x.value().empty());
    }
    auto none = LlguidanceSchemaForResponseFormat(std::nullopt, schema);
    REQUIRE(none.ok());
    CHECK(none.value().empty());
    auto no_schema = LlguidanceSchemaForResponseFormat(std::string("json_schema"), std::nullopt);
    REQUIRE(no_schema.ok());
    CHECK(no_schema.value().empty());
    auto empty_schema = LlguidanceSchemaForResponseFormat(std::string("json_schema"), std::string());
    REQUIRE(empty_schema.ok());
    CHECK(empty_schema.value().empty());

    // json_schema with a schema that cannot be normalised fails (never silently unconstrained).
    auto bad = LlguidanceSchemaForResponseFormat(std::string("json_schema"), std::string("[1,2]"));
    CHECK_FALSE(bad.ok());
    CHECK(bad.status().code() == ErrorCode::invalid_argument);
    auto garbage = LlguidanceSchemaForResponseFormat(std::string("json_schema"), std::string("{nope"));
    CHECK_FALSE(garbage.ok());
}
