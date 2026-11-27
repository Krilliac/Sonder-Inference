#include "embeddings.hpp"

#include <cmath>
#include <limits>

namespace sonder::backends::llamacpp {

namespace {

std::size_t saturating_add(std::size_t a, std::size_t b) noexcept {
    return a > std::numeric_limits<std::size_t>::max() - b ? std::numeric_limits<std::size_t>::max() : a + b;
}

EmbedLimitCheck fail(EmbedLimitViolation v, std::size_t index, std::string message) {
    EmbedLimitCheck c;
    c.ok = false;
    c.violation = v;
    c.index = index;
    c.message = std::move(message);
    return c;
}

}  // namespace

double L2Normalize(std::span<float> v) noexcept {
    double sq = 0.0;
    for (const float x : v) sq += static_cast<double>(x) * static_cast<double>(x);
    const double norm = std::sqrt(sq);
    if (!(norm > 0.0) || !std::isfinite(norm)) return 0.0;
    const double inv = 1.0 / norm;
    for (float& x : v) x = static_cast<float>(static_cast<double>(x) * inv);
    return norm;
}

EmbedLimitCheck CheckEmbedLimits(std::span<const std::size_t> lengths, const EmbedLimits& limits) {
    if (lengths.empty()) return fail(EmbedLimitViolation::no_inputs, 0, "embedding request has no inputs");
    if (limits.max_inputs != 0 && lengths.size() > limits.max_inputs) {
        return fail(EmbedLimitViolation::too_many_inputs, lengths.size() - 1,
                    "embedding request exceeds maximum of " + std::to_string(limits.max_inputs) + " inputs");
    }
    std::size_t total = 0;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] == 0) {
            return fail(EmbedLimitViolation::empty_input, i, "embedding input " + std::to_string(i) + " is empty");
        }
        if (limits.max_input_tokens != 0 && lengths[i] > limits.max_input_tokens) {
            return fail(EmbedLimitViolation::input_too_long, i,
                        "embedding input " + std::to_string(i) + " has " + std::to_string(lengths[i]) +
                            " tokens, over the per-input maximum of " + std::to_string(limits.max_input_tokens));
        }
        total = saturating_add(total, lengths[i]);
    }
    if (limits.max_total_tokens != 0 && total > limits.max_total_tokens) {
        return fail(EmbedLimitViolation::total_too_long, 0,
                    "embedding request exceeds the total maximum of " + std::to_string(limits.max_total_tokens) +
                        " tokens");
    }
    return {};
}

std::string PoolingMetadataKey(std::string_view architecture) {
    if (architecture.empty()) return {};
    std::string key(architecture);
    key += ".pooling_type";
    return key;
}

std::optional<PoolingType> ParsePoolingMetadata(std::string_view value) noexcept {
    if (value.size() != 1) return std::nullopt;
    switch (value.front()) {
        case '0': return PoolingType::kNone;
        case '1': return PoolingType::kMean;
        case '2': return PoolingType::kCls;
        case '3': return PoolingType::kLast;
        case '4': return PoolingType::kRank;
        default: return std::nullopt;
    }
}

PoolingType ResolveEmbedPooling(PoolingType requested, std::optional<PoolingType> model_default,
                                bool has_encoder) noexcept {
    if (requested != PoolingType::kUnspecified) return requested;
    if (model_default) {
        switch (*model_default) {
            case PoolingType::kMean:
            case PoolingType::kCls:
            case PoolingType::kLast:
            case PoolingType::kRank: return *model_default;
            default: break;
        }
    }
    return has_encoder ? PoolingType::kUnspecified : PoolingType::kMean;
}

}  // namespace sonder::backends::llamacpp
