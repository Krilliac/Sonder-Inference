// Internal: converts JSON Schema (draft-07 / 2020-12 subset) and OpenAI
// response_format specs into llama.cpp-compatible GBNF grammars, and provides
// the StructuredGrammarEngine abstraction (with optional SONDER_WITH_LLGUIDANCE
// compile-time hook). Pure functions.
#pragma once

#include <string>
#include <string_view>

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
// items, minItems/maxItems (<= 32), enum, const, anyOf, provably disjoint
// oneOf (disjoint JSON types, disjoint const/enum literals, or tagged-union
// objects with disjoint required discriminator const/enum), basic regex
// pattern subset (^...$, literals, character classes, *, +, ?, {m,n} <= 32),
// exact bounded integer minimum/maximum ([-999999, 999999], plus half-open 0),
// minLength/maxLength (<= 32), and local $ref into $defs / definitions, plus
// JSON Schema annotation metadata.
// Errors: invalid_argument for unsupported/unenforceable validation keywords
// (e.g. overlapping oneOf, allOf, lookaround/backref patterns, floating-point
// or out-of-range numeric bounds, bounds > 32, properties + permissive
// additionalProperties), malformed schemas, or exceeded traversal/size limits.
[[nodiscard]] Result<std::string> json_schema_to_gbnf(const json::Value& schema);

// Pluggable structured grammar compilation backend.
enum class GrammarEngineKind { gbnf, llguidance };

struct CompiledGrammar {
    GrammarEngineKind engine = GrammarEngineKind::gbnf;
    // Llama.cpp GBNF text (populated when engine == gbnf).
    std::string gbnf;
    // Canonical JSON schema representation (for llguidance or diagnostics).
    std::string raw_schema;
};

class StructuredGrammarEngine {
public:
    virtual ~StructuredGrammarEngine() = default;
    [[nodiscard]] virtual GrammarEngineKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_json_object() const = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const = 0;
};

// Returns the built-in GBNF structured grammar engine.
[[nodiscard]] const StructuredGrammarEngine& gbnf_structured_grammar_engine() noexcept;

// Returns the process-default structured grammar engine (llguidance when
// compiled with SONDER_WITH_LLGUIDANCE, otherwise the built-in GBNF engine).
[[nodiscard]] const StructuredGrammarEngine& default_structured_grammar_engine() noexcept;

}  // namespace sonder::inference::server::detail
