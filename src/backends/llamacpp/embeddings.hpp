// Pure helpers behind the native embeddings path. No llama.cpp header
// dependency, so everything here is unit-testable without model weights
// (tests/test_embeddings.cpp):
//   - L2 normalisation of an output vector;
//   - request limits on the tokenized inputs (count, per-input, total);
//   - choosing the pooling type: the caller's explicit choice, else the one
//     the model's GGUF metadata declares, else a safe fallback.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

// Scales `v` to unit L2 length in place and returns the norm it had. The sum
// of squares is accumulated in double, so large or tiny components neither
// overflow nor underflow. An empty vector, a zero vector, or one whose norm
// is not finite (NaN/inf component) is left untouched and the returned norm
// is 0.
double L2Normalize(std::span<float> v) noexcept;

struct EmbedLimits {
    std::size_t max_inputs = 256;          // 0 = unlimited
    std::size_t max_input_tokens = 0;      // per input; 0 = bounded only by the context window
    std::size_t max_total_tokens = 262144; // across the request; 0 = unlimited
};

enum class EmbedLimitViolation { none, no_inputs, empty_input, too_many_inputs, input_too_long, total_too_long };

struct EmbedLimitCheck {
    bool ok = true;
    EmbedLimitViolation violation = EmbedLimitViolation::none;
    std::size_t index = 0;  // offending input for empty_input / input_too_long
    std::string message;
};

// Checks the per-input token counts of one request against `limits`. Totals
// are summed with saturation, so hostile lengths cannot wrap around.
EmbedLimitCheck CheckEmbedLimits(std::span<const std::size_t> lengths, const EmbedLimits& limits);

// GGUF stores the model's pooling as an integer under "<arch>.pooling_type".
// Returns that key ("" for an empty architecture name).
std::string PoolingMetadataKey(std::string_view architecture);

// Parses the metadata value ("0".."4"; none, mean, cls, last, rank). Anything
// else (empty, negative, out of range, trailing junk) is nullopt.
std::optional<PoolingType> ParsePoolingMetadata(std::string_view value) noexcept;

// Pooling for a dense-embedding pass:
//   1. an explicit request (anything but kUnspecified) wins;
//   2. otherwise the model's own declared pooling when it is mean, cls, last,
//      or rank (rank is then rejected by the caller as not a dense model);
//   3. otherwise (declared none/absent): encoder models stay kUnspecified
//      (llama.cpp resolves it itself), decoder-only models use mean.
PoolingType ResolveEmbedPooling(PoolingType requested, std::optional<PoolingType> model_default,
                                bool has_encoder) noexcept;

}  // namespace sonder::backends::llamacpp
