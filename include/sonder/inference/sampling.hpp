// Sonder Inference: sampling policy configuration.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference {

// Additive logit bias for one vocabulary token.
struct TokenLogitBias {
    // Vocabulary index, >= 0.
    std::int32_t token = -1;
    // Added to the token's logit. Finite within [-100, 100], or -infinity to ban the token.
    float bias = 0.0f;
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
    // Window of recent tokens the penalties look at: -1 = whole context, 0 disables
    // the window, otherwise [1, kMaxContextLimit]. 64 matches llama.cpp and Ollama.
    std::int32_t repeat_last_n = 64;
    // Additive presence penalty, [-2, 2]. 0 disables.
    float presence_penalty = 0.0f;
    // Additive per-occurrence frequency penalty, [-2, 2]. 0 disables.
    float frequency_penalty = 0.0f;
    // Per-token logit bias (at most kMaxLogitBiasEntries, token ids unique).
    // Not every backend supports it: the Ollama adapter does not forward it.
    std::vector<TokenLogitBias> logit_bias;
    // Requested context window in tokens. 0 = backend/model default, otherwise
    // [1, kMaxContextLimit]. A backend request option, not a sampler stage.
    std::int32_t num_ctx = 0;
    // Seed for reproducible sampling. Unset lets the backend choose.
    std::optional<std::uint64_t> seed;
    // Upper bound on generated tokens, [1, kMaxTokensLimit].
    std::int32_t max_tokens = 256;
    // Stop sequences (each non-empty, at most kMaxStopSequences).
    std::vector<std::string> stop;

    // Which sampler fields the caller set explicitly. C++ only (not in the C ABI).
    // When `explicit_only` is true, a backend that has its own per-model defaults
    // (Ollama applies the GGUF's) sends only the fields flagged here, so unset
    // fields keep the model's defaults instead of this struct's. Backends that
    // sample natively ignore it and use the values above.
    enum Field : std::uint32_t {
        kTemperature = 1u << 0,
        kTopP = 1u << 1,
        kTopK = 1u << 2,
        kMinP = 1u << 3,
        kRepeatPenalty = 1u << 4,
        kRepeatLastN = 1u << 5,
        kPresencePenalty = 1u << 6,
        kFrequencyPenalty = 1u << 7,
    };
    bool explicit_only = false;
    std::uint32_t explicit_fields = 0;
    bool is_explicit(Field f) const noexcept { return (explicit_fields & f) != 0; }

    static constexpr std::int32_t kMaxTokensLimit = 1 << 20;
    static constexpr std::size_t kMaxStopSequences = 16;
    static constexpr std::size_t kMaxStopSequenceBytes = 256;
    static constexpr std::int32_t kMaxContextLimit = 1 << 24;
    static constexpr std::size_t kMaxLogitBiasEntries = 1024;
    static constexpr float kMaxPenaltyMagnitude = 2.0f;
    static constexpr float kMaxLogitBiasMagnitude = 100.0f;

    // Deterministic preset used by tests and benchmarks.
    static SamplingConfig greedy(std::int32_t max_tokens = 128, std::uint64_t seed = 42);
};

// Returns invalid_argument with a field-specific message on the first violation.
Status validate(const SamplingConfig& config);

}  // namespace sonder::inference
