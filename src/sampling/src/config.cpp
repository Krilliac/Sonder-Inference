#include "sonder/sampling/config.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace sonder::inference::sampling {

std::string_view to_string(StageKind kind) noexcept {
    switch (kind) {
    case StageKind::Penalties: return "penalties";
    case StageKind::TopK: return "top_k";
    case StageKind::TypicalP: return "typ_p";
    case StageKind::TopP: return "top_p";
    case StageKind::MinP: return "min_p";
    case StageKind::Temperature: return "temperature";
    }
    return "unknown";
}

std::optional<StageKind> parse_stage(std::string_view name) noexcept {
    if (name == "penalties") return StageKind::Penalties;
    if (name == "top_k") return StageKind::TopK;
    if (name == "typ_p" || name == "typical_p" || name == "typical") return StageKind::TypicalP;
    if (name == "top_p") return StageKind::TopP;
    if (name == "min_p") return StageKind::MinP;
    if (name == "temperature" || name == "temp") return StageKind::Temperature;
    return std::nullopt;
}

std::vector<StageKind> default_stage_order() {
    return {StageKind::Penalties, StageKind::TopK,  StageKind::TypicalP,
            StageKind::TopP,      StageKind::MinP,  StageKind::Temperature};
}

std::string ValidationResult::to_string() const {
    if (errors.empty()) {
        return "sampling config valid";
    }
    std::string out = "sampling config invalid: ";
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (i != 0) {
            out += "; ";
        }
        out += errors[i].field;
        out += ": ";
        out += errors[i].message;
    }
    return out;
}

namespace {

template <typename T>
std::string got(T value) {
    std::ostringstream os;
    os << " (got " << value << ")";
    return os.str();
}

void check_unit_interval(ValidationResult& r, const char* field, float v) {
    if (!std::isfinite(v) || v < 0.0f || v > 1.0f) {
        r.errors.push_back({field, "must be a finite value in [0, 1]" + got(v)});
    }
}

void check_token(ValidationResult& r, const std::string& field, TokenId id, std::optional<std::size_t> vocab) {
    if (id < 0) {
        r.errors.push_back({field, "token id must be >= 0" + got(id)});
    } else if (vocab && static_cast<std::size_t>(id) >= *vocab) {
        r.errors.push_back({field, "token id must be < vocab size " + std::to_string(*vocab) + got(id)});
    }
}

}  // namespace

ValidationResult validate(const SamplerConfig& c, std::optional<std::size_t> vocab_size) {
    ValidationResult r;

    if (vocab_size && *vocab_size == 0) {
        r.errors.push_back({"vocab_size", "must be > 0"});
    }
    if (!std::isfinite(c.temperature) || c.temperature < 0.0f) {
        r.errors.push_back({"temperature", "must be a finite value >= 0 (0 = greedy)" + got(c.temperature)});
    }
    if (c.top_k < 0) {
        r.errors.push_back({"top_k", "must be >= 0 (0 = disabled)" + got(c.top_k)});
    }
    check_unit_interval(r, "top_p", c.top_p);
    check_unit_interval(r, "min_p", c.min_p);
    check_unit_interval(r, "typical_p", c.typical_p);
    if (c.min_keep < 0) {
        r.errors.push_back({"min_keep", "must be >= 0" + got(c.min_keep)});
    }
    if (c.penalty_last_n < -1) {
        r.errors.push_back({"penalty_last_n", "must be >= -1 (-1 = unbounded, 0 = disabled)" + got(c.penalty_last_n)});
    }
    if (!std::isfinite(c.repeat_penalty) || c.repeat_penalty <= 0.0f) {
        r.errors.push_back({"repeat_penalty", "must be a finite value > 0 (1 = disabled)" + got(c.repeat_penalty)});
    }
    if (!std::isfinite(c.frequency_penalty)) {
        r.errors.push_back({"frequency_penalty", "must be finite" + got(c.frequency_penalty)});
    }
    if (!std::isfinite(c.presence_penalty)) {
        r.errors.push_back({"presence_penalty", "must be finite" + got(c.presence_penalty)});
    }
    for (std::size_t i = 0; i < c.logit_bias.size(); ++i) {
        const auto& b = c.logit_bias[i];
        const std::string field = "logit_bias[" + std::to_string(i) + "]";
        check_token(r, field, b.token, vocab_size);
        if (std::isnan(b.bias) || b.bias == std::numeric_limits<float>::infinity()) {
            r.errors.push_back({field, "bias must be finite or -inf (to ban a token)" + got(b.bias)});
        }
    }
    for (std::size_t i = 0; i < c.stop_sequences.size(); ++i) {
        if (c.stop_sequences[i].empty()) {
            r.errors.push_back({"stop_sequences[" + std::to_string(i) + "]", "must not be empty"});
        }
    }
    for (std::size_t i = 0; i < c.stop_tokens.size(); ++i) {
        check_token(r, "stop_tokens[" + std::to_string(i) + "]", c.stop_tokens[i], vocab_size);
    }
    std::unordered_set<int> seen;
    for (std::size_t i = 0; i < c.stage_order.size(); ++i) {
        const auto kind = c.stage_order[i];
        if (to_string(kind) == "unknown") {
            r.errors.push_back({"stage_order[" + std::to_string(i) + "]", "unknown stage kind"});
        } else if (!seen.insert(static_cast<int>(kind)).second) {
            r.errors.push_back({"stage_order[" + std::to_string(i) + "]",
                                "duplicate stage '" + std::string(to_string(kind)) + "'"});
        }
    }
    return r;
}

}  // namespace sonder::inference::sampling
