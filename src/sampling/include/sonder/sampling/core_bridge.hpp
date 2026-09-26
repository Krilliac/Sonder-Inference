#pragma once

// Bridge between the core API's placeholder policy
// (sonder::inference::SamplingConfig, include/sonder/inference/sampling.hpp)
// and the sampler chain in this module.

#include <memory>
#include <optional>

#include "sonder/inference/error.hpp"
#include "sonder/inference/sampling.hpp"
#include "sonder/sampling/chain.hpp"
#include "sonder/sampling/config.hpp"

namespace sonder::inference::sampling {

/// Map the core config onto a SamplerConfig. Shared fields are copied
/// (typical_p, presence/frequency/repeat penalties, repeat_last_n ->
/// penalty_last_n, logit_bias); fields the core does not carry keep the
/// chain's llama.cpp-compatible defaults (min_keep, stop_tokens, llama.cpp
/// stage order). temperature == 0 selects the greedy selector. `max_tokens`
/// is a generation-loop concern and `num_ctx` a backend request option;
/// neither is part of the chain.
[[nodiscard]] SamplerConfig from_core(const sonder::inference::SamplingConfig& core);

/// Convert a ValidationResult into the core Status type
/// (ErrorCode::invalid_argument listing every failing field, or success).
[[nodiscard]] sonder::inference::Status to_status(const ValidationResult& result);

/// Validate the core config with the core rules, map it, validate the mapped
/// config, and build a chain. Never throws for invalid input.
[[nodiscard]] sonder::inference::Result<SamplerChain> make_chain(
    const sonder::inference::SamplingConfig& core, std::shared_ptr<Constraint> constraint = nullptr,
    std::optional<std::size_t> vocab_size = std::nullopt);

}  // namespace sonder::inference::sampling
