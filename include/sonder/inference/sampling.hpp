// Sonder Inference: sampling policy configuration.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference {

// Additive bias for one vocabulary token. -infinity bans the token.
struct LogitBias {
    std::int32_t token = -1;  // vocabulary index, >= 0
    float bias = 0.0f;        // finite, or -infinity
};

struct SamplingConfig {
    // 0 means greedy (argmax). Valid range [0, 10].
    float temperature = 0.8f;
    // Nucleus sampling mass, (0, 1]. 1 disables.
    float top_p = 0.95f;
    // 0 disables top-k. Valid range [0, 100000].
    std::int32_t top_k = 40;
    // Minimum relative probability, [0, 1]. 0 disables.
    float min_p = 0.0f;
    // Multiplicative repetition penalty, (0, 10]. 1 disables.
    float repeat_penalty = 1.1f;
    // Locally typical sampling mass, (0, 1]. 1 disables.
    float typical_p = 1.0f;
    // Additive OpenAI-style penalties, [-2, 2]. 0 disables.
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    // Window of recent tokens the penalties look at: -1 = whole context,
    // 0 disables the penalties, otherwise [1, kMaxPenaltyWindow].
    std::int32_t penalty_last_n = 64;
    // Per-token logit bias (at most kMaxLogitBias entries). Backends that
    // cannot apply it report ErrorCode::unsupported rather than ignoring it.
    std::vector<LogitBias> logit_bias;
    // Context window requested for this request, in tokens. 0 = backend or
    // model default; otherwise [16, kMaxContextTokens]. Passed to Ollama as
    // options.num_ctx; caps the llama.cpp generation budget; bounds the
    // engine's KV accounting for the request.
    std::int32_t num_ctx = 0;
    // Seed for reproducible sampling. Unset lets the backend choose.
    std::optional<std::uint64_t> seed;
    // Upper bound on generated tokens, [1, kMaxTokensLimit].
    std::int32_t max_tokens = 256;
    // Stop sequences (each non-empty, at most kMaxStopSequences).
    std::vector<std::string> stop;

    static constexpr std::int32_t kMaxTokensLimit = 1 << 20;
    static constexpr std::size_t kMaxStopSequences = 16;
    static constexpr std::size_t kMaxStopSequenceBytes = 256;
    static constexpr std::size_t kMaxLogitBias = 1024;
    static constexpr std::int32_t kMaxPenaltyWindow = 1 << 20;
    static constexpr std::int32_t kMaxContextTokens = 1 << 22;

    // Deterministic preset used by tests and benchmarks.
    static SamplingConfig greedy(std::int32_t max_tokens = 128, std::uint64_t seed = 42);
};

// Returns invalid_argument with a field-specific message on the first violation.
Status validate(const SamplingConfig& config);

}  // namespace sonder::inference
