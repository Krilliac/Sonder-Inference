#include "sonder/inference/sampling.hpp"

#include <cmath>
#include <string>
#include <unordered_set>

namespace sonder::inference {

SamplingConfig SamplingConfig::greedy(std::int32_t max_tokens, std::uint64_t seed) {
    SamplingConfig c;
    c.temperature = 0.0f;
    c.top_p = 1.0f;
    c.top_k = 0;
    c.min_p = 0.0f;
    c.repeat_penalty = 1.0f;
    c.seed = seed;
    c.max_tokens = max_tokens;
    return c;
}

namespace {
Status invalid(const std::string& message) { return Status(ErrorCode::invalid_argument, message); }
}  // namespace

Status validate(const SamplingConfig& c) {
    if (!std::isfinite(c.temperature) || c.temperature < 0.0f || c.temperature > 10.0f) {
        return invalid("temperature must be finite and within [0, 10]");
    }
    if (!std::isfinite(c.top_p) || c.top_p <= 0.0f || c.top_p > 1.0f) {
        return invalid("top_p must be within (0, 1]");
    }
    if (c.top_k < 0 || c.top_k > 100000) {
        return invalid("top_k must be within [0, 100000] (0 disables)");
    }
    if (!std::isfinite(c.min_p) || c.min_p < 0.0f || c.min_p > 1.0f) {
        return invalid("min_p must be within [0, 1]");
    }
    if (!std::isfinite(c.repeat_penalty) || c.repeat_penalty <= 0.0f || c.repeat_penalty > 10.0f) {
        return invalid("repeat_penalty must be within (0, 10]");
    }
    if (!std::isfinite(c.typical_p) || c.typical_p <= 0.0f || c.typical_p > 1.0f) {
        return invalid("typical_p must be within (0, 1]");
    }
    if (c.repeat_last_n < -1 || c.repeat_last_n > SamplingConfig::kMaxContextLimit) {
        return invalid("repeat_last_n must be -1 (whole context), 0 (disabled) or within [1, " +
                       std::to_string(SamplingConfig::kMaxContextLimit) + "]");
    }
    constexpr float kPen = SamplingConfig::kMaxPenaltyMagnitude;
    if (!std::isfinite(c.presence_penalty) || c.presence_penalty < -kPen || c.presence_penalty > kPen) {
        return invalid("presence_penalty must be finite and within [-2, 2]");
    }
    if (!std::isfinite(c.frequency_penalty) || c.frequency_penalty < -kPen || c.frequency_penalty > kPen) {
        return invalid("frequency_penalty must be finite and within [-2, 2]");
    }
    if (c.logit_bias.size() > SamplingConfig::kMaxLogitBiasEntries) {
        return invalid("at most " + std::to_string(SamplingConfig::kMaxLogitBiasEntries) +
                       " logit_bias entries allowed");
    }
    std::unordered_set<std::int32_t> biased_tokens;
    for (const auto& b : c.logit_bias) {
        if (b.token < 0) {
            return invalid("logit_bias token ids must be >= 0");
        }
        const bool ban = std::isinf(b.bias) && b.bias < 0.0f;
        if (!ban && (!std::isfinite(b.bias) || b.bias < -SamplingConfig::kMaxLogitBiasMagnitude ||
                     b.bias > SamplingConfig::kMaxLogitBiasMagnitude)) {
            return invalid("logit_bias values must be within [-100, 100] or -infinity (ban)");
        }
        if (!biased_tokens.insert(b.token).second) {
            return invalid("logit_bias token " + std::to_string(b.token) + " appears more than once");
        }
    }
    if (c.num_ctx < 0 || c.num_ctx > SamplingConfig::kMaxContextLimit) {
        return invalid("num_ctx must be 0 (backend default) or within [1, " +
                       std::to_string(SamplingConfig::kMaxContextLimit) + "]");
    }
    if (c.max_tokens < 1 || c.max_tokens > SamplingConfig::kMaxTokensLimit) {
        return invalid("max_tokens must be within [1, " + std::to_string(SamplingConfig::kMaxTokensLimit) + "]");
    }
    if (c.stop.size() > SamplingConfig::kMaxStopSequences) {
        return invalid("at most " + std::to_string(SamplingConfig::kMaxStopSequences) + " stop sequences allowed");
    }
    for (const auto& s : c.stop) {
        if (s.empty()) {
            return invalid("stop sequences must be non-empty");
        }
        if (s.size() > SamplingConfig::kMaxStopSequenceBytes) {
            return invalid("stop sequence exceeds " + std::to_string(SamplingConfig::kMaxStopSequenceBytes) + " bytes");
        }
    }
    return Status::success();
}

}  // namespace sonder::inference
