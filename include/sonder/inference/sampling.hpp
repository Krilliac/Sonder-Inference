// Sonder Inference: sampling policy configuration.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference {

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
    // Seed for reproducible sampling. Unset lets the backend choose.
    std::optional<std::uint64_t> seed;
    // Upper bound on generated tokens, [1, kMaxTokensLimit].
    std::int32_t max_tokens = 256;
    // Stop sequences (each non-empty, at most kMaxStopSequences).
    std::vector<std::string> stop;

    static constexpr std::int32_t kMaxTokensLimit = 1 << 20;
    static constexpr std::size_t kMaxStopSequences = 16;
    static constexpr std::size_t kMaxStopSequenceBytes = 256;

    // Deterministic preset used by tests and benchmarks.
    static SamplingConfig greedy(std::int32_t max_tokens = 128, std::uint64_t seed = 42);
};

// Returns invalid_argument with a field-specific message on the first violation.
Status validate(const SamplingConfig& config);

}  // namespace sonder::inference
