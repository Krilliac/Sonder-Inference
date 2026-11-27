// JSON-Schema normalisation for the optional llguidance structured-output path
// (SONDER_WITH_LLGUIDANCE). Pure and model-free: it only rewrites a schema document.
//
// llguidance (like OpenAI strict structured outputs) treats an object schema that does not say
// anything about `additionalProperties` as open, so a constrained model may append arbitrary extra
// keys. NormalizeJsonSchema closes every object schema that leaves the field unset and leaves
// everything else untouched. See the rules on NormalizeJsonSchema.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamacpp_detail {

// Bounds against hostile schemas. Every bound is enforced before the work it guards, and a
// breach returns ErrorCode::invalid_argument (never a truncated or partially normalised schema).
struct JsonSchemaNormalizeLimits {
    // Deepest schema nesting accepted. The root schema is depth 1; a subschema reached through
    // `properties`, `items`, `anyOf`, `$defs`, ... is one deeper than its parent.
    int max_depth = 64;
    // Most schema nodes (objects visited as schemas) accepted.
    std::size_t max_schema_nodes = 100000;
    // Largest schema text accepted by NormalizeJsonSchemaText.
    std::size_t max_input_bytes = std::size_t{4} * 1024 * 1024;
    // Most JSON values (of any kind) the text parser accepts.
    std::size_t max_json_values = 1000000;
};

// Returns a copy of `schema` in which every object schema without an explicit
// `additionalProperties` (or `unevaluatedProperties`) gets `"additionalProperties": false`.
//
//  * An object schema is one with `type` "object" (or an array of types containing "object"),
//    or with no `type` but a `properties` / `patternProperties` keyword.
//  * Subschemas are found through the standard applicator keywords: properties,
//    patternProperties, additionalProperties, items, prefixItems, contains, anyOf, oneOf, allOf,
//    not, if/then/else, $defs/definitions, dependentSchemas, dependencies, propertyNames and the
//    unevaluated* keywords. Data keywords (enum, const, default, examples) are copied verbatim.
//  * An explicit `additionalProperties` (true, false, or a schema) is kept as is; a schema value
//    is itself normalised.
//  * A node with `$ref` is not closed (the referenced definition is closed where it is defined,
//    and sibling keywords would otherwise also constrain the referenced schema).
//  * A schema with `allOf`, and the objects that are direct `allOf` branches, are not closed:
//    closing one branch would forbid the properties of its siblings.
//  * Boolean schemas (true / false) are returned unchanged.
//
// The result is idempotent: normalising an already-normalised schema returns it unchanged.
// `schema` must be a JSON object or a boolean, else invalid_argument.
[[nodiscard]] Result<json::Value> NormalizeJsonSchema(const json::Value& schema,
                                                      const JsonSchemaNormalizeLimits& limits = {});

// Text form: parses `schema_json` (bounded by `limits`), normalises it, and returns the compact
// JSON text. invalid_argument on oversized, unparsable or too deeply nested input.
[[nodiscard]] Result<std::string> NormalizeJsonSchemaText(std::string_view schema_json,
                                                          const JsonSchemaNormalizeLimits& limits = {});

// Decides what the llguidance path should enforce for a request's `response_format`.
//  * `type` is not "json_schema", or no schema text is given: returns an empty string (the request
//    does not use llguidance; the existing GBNF path applies unchanged).
//  * otherwise: the normalised schema text, or invalid_argument when it cannot be normalised.
[[nodiscard]] Result<std::string> LlguidanceSchemaForResponseFormat(const std::optional<std::string>& type,
                                                                    const std::optional<std::string>& schema_text,
                                                                    const JsonSchemaNormalizeLimits& limits = {});

}  // namespace sonder::inference::llamacpp_detail
