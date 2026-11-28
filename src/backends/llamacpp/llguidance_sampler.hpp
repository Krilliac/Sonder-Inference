// llguidance-backed constrained-decoding sampler (optional, SONDER_WITH_LLGUIDANCE).
//
// Private to the wrapper library: it names llama.cpp types, so only llamacpp_backend.cpp and
// llguidance_sampler.cpp include it. The core adapter reaches it through
// SamplingParams::llguidance_json_schema.
#pragma once

#include <string>
#include <string_view>

struct llama_sampler;
struct llama_vocab;

namespace sonder::backends::llamacpp {

// True when this build links llguidance (SONDER_LLAMACPP_HAS_LLGUIDANCE).
[[nodiscard]] bool LlguidanceAvailable() noexcept;

// Builds a sampler that constrains output to `json_schema` (JSON Schema text; callers normalise it
// first, see json_schema_normalize.hpp). Returns nullptr and sets `error` when llguidance is not
// compiled in or rejects the schema or the vocabulary. The sampler follows the llama.cpp grammar
// sampler contract (accept / apply / reset / clone), with one difference: when llguidance reports a
// matcher error it throws std::runtime_error from accept/apply instead of silently dropping the
// constraint, because the backend's TryAcceptGrammar / TryApplyGrammar turn that into a failed
// request. Never a sampler that has quietly stopped constraining.
[[nodiscard]] llama_sampler* MakeLlguidanceJsonSchemaSampler(const llama_vocab* vocab, std::string_view json_schema,
                                                             std::string& error);

}  // namespace sonder::backends::llamacpp
