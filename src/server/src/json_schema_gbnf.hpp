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
// array of types), properties, required, additionalProperties (false when
// properties are declared; bool or schema when properties are absent/empty),
// items, minItems/maxItems (<= 32), enum, const, anyOf, minLength/maxLength
// (<= 32), and local $ref into $defs / definitions, plus JSON Schema
// annotation metadata (title, description, default, examples, $schema, $id,
// $anchor, $comment, readOnly, writeOnly, deprecated).
// Errors: invalid_argument for unsupported validation keywords (e.g. oneOf,
// allOf, pattern, numeric bounds, bounds > 32, properties + permissive
// additionalProperties), malformed schemas, or exceeded traversal/size limits.
[[nodiscard]] Result<std::string> json_schema_to_gbnf(const json::Value& schema);

}  // namespace sonder::inference::server::detail
