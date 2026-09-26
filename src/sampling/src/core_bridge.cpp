#include "sonder/sampling/core_bridge.hpp"

#include <utility>

namespace sonder::inference::sampling {

SamplerConfig from_core(const sonder::inference::SamplingConfig& core) {
    SamplerConfig c;
    c.temperature = core.temperature;
    c.greedy = core.temperature == 0.0f;
    c.top_p = core.top_p;
    c.top_k = core.top_k;
    c.min_p = core.min_p;
    c.repeat_penalty = core.repeat_penalty;
    c.seed = core.seed;
    c.stop_sequences = core.stop;
    return c;
}

sonder::inference::Status to_status(const ValidationResult& result) {
    if (result.ok()) {
        return sonder::inference::Status::success();
    }
    return {sonder::inference::ErrorCode::invalid_argument, result.to_string()};
}

sonder::inference::Result<SamplerChain> make_chain(const sonder::inference::SamplingConfig& core,
                                                   std::shared_ptr<Constraint> constraint,
                                                   std::optional<std::size_t> vocab_size) {
    if (auto st = sonder::inference::validate(core); !st.ok()) {
        return st;
    }
    const SamplerConfig mapped = from_core(core);
    if (auto st = to_status(validate(mapped, vocab_size)); !st.ok()) {
        return st;
    }
    return build_chain(mapped, std::move(constraint), vocab_size);
}

}  // namespace sonder::inference::sampling
