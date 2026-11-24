// Internal: converts JSON Schema (draft-07 / 2020-12 subset) and OpenAI
// response_format specs into llama.cpp-compatible GBNF grammars. Pure functions.
#pragma once

#include <string>

#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// Strict RFC 8259 JSON object GBNF grammar (root ::= object).
[[nodiscard]] std::string json_object_gbnf();

// Converts a JSON Schema object into a self-contained GBNF grammar whose root
// rule matches JSON values conforming to `schema`.
// Supports: type (object, array, string, integer, number, boolean, null, or
// array of types), properties, required, additionalProperties (bool or schema),
// items, minItems, maxItems, enum, const, anyOf, oneOf, minLength, maxLength,
// and local $ref into $defs / definitions.
// Errors: invalid_argument for unsupported/malformed schemas or exceeded limits.
[[nodiscard]] Result<std::string> json_schema_to_gbnf(const json::Value& schema);

}  // namespace sonder::inference::server::detail
