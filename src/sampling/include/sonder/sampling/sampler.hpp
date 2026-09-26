#pragma once

#include <memory>
#include <string_view>

#include "sonder/sampling/types.hpp"

namespace sonder::inference::sampling {

/// One composable stage of a sampler chain. Stages transform or truncate the
/// candidate set in place; they must be deterministic given their inputs and
/// internal state.
class Sampler {
public:
    virtual ~Sampler() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Transform `candidates` in place.
    virtual void apply(Candidates& candidates) = 0;

    /// Observe a committed token (penalty history etc). Default: stateless.
    virtual void accept(TokenId /*token*/) {}

    /// Clear per-sequence state. Default: stateless.
    virtual void reset() {}

    /// Deep copy including state (used to fork sessions).
    [[nodiscard]] virtual std::unique_ptr<Sampler> clone() const = 0;
};

}  // namespace sonder::inference::sampling
