#include "sonder/sampling/chain.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "sonder/sampling/samplers.hpp"

namespace sonder::inference::sampling {

std::string_view to_string(SampleStatus status) noexcept {
    switch (status) {
    case SampleStatus::Ok: return "ok";
    case SampleStatus::EmptyLogits: return "empty logits";
    case SampleStatus::NoViableCandidates: return "no viable candidates (all logits -inf)";
    }
    return "unknown";
}

SamplerChain::SamplerChain(std::uint64_t seed, Selector selector) : selector_(selector), rng_(seed) {}

SamplerChain::SamplerChain(const SamplerChain& other)
    : constraint_(other.constraint_),
      stop_tokens_(other.stop_tokens_),
      selector_(other.selector_),
      rng_(other.rng_),
      cur_(other.cur_) {
    stages_.reserve(other.stages_.size());
    for (const auto& s : other.stages_) {
        stages_.push_back(s->clone());
    }
}

SamplerChain& SamplerChain::operator=(const SamplerChain& other) {
    if (this != &other) {
        SamplerChain tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

SamplerChain& SamplerChain::add(std::unique_ptr<Sampler> stage) {
    if (!stage) {
        throw std::invalid_argument("SamplerChain::add: null stage");
    }
    stages_.push_back(std::move(stage));
    return *this;
}

void SamplerChain::set_constraint(std::shared_ptr<Constraint> constraint) { constraint_ = std::move(constraint); }

void SamplerChain::set_stop_tokens(std::vector<TokenId> tokens) {
    stop_tokens_ = std::unordered_set<TokenId>(tokens.begin(), tokens.end());
}

SampleResult SamplerChain::sample(std::span<const float> logits) {
    SampleResult result;
    if (logits.empty()) {
        cur_.data.clear();
        result.status = SampleStatus::EmptyLogits;
        return result;
    }
    cur_.data.resize(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float l = logits[i];
        cur_.data[i] = TokenData{static_cast<TokenId>(i), std::isnan(l) ? kNegInf : l, 0.0f};
    }
    cur_.sorted = false;

    if (constraint_) {
        constraint_->apply(cur_);
    }
    for (auto& stage : stages_) {
        stage->apply(cur_);
    }

    if (cur_.empty() || viable_count(cur_) == 0) {
        result.status = SampleStatus::NoViableCandidates;
        return result;
    }

    sort_canonical(cur_);
    softmax(cur_);

    std::size_t pick = 0;
    if (selector_ == Selector::Distribution) {
        const double u = rng_.uniform();
        double cum = 0.0;
        std::size_t last_viable = 0;
        pick = cur_.size();
        for (std::size_t i = 0; i < cur_.size(); ++i) {
            if (cur_.data[i].p <= 0.0f) {
                continue;
            }
            last_viable = i;
            cum += cur_.data[i].p;
            if (u < cum) {
                pick = i;
                break;
            }
        }
        if (pick == cur_.size()) {  // rounding: cum ended slightly below 1
            pick = last_viable;
        }
    }
    result.status = SampleStatus::Ok;
    result.token = cur_.data[pick].id;
    result.probability = cur_.data[pick].p;
    return result;
}

void SamplerChain::accept(TokenId token) {
    for (auto& stage : stages_) {
        stage->accept(token);
    }
    if (constraint_) {
        constraint_->accept(token);
    }
}

void SamplerChain::accept_prompt(std::span<const TokenId> tokens) {
    for (const TokenId t : tokens) {
        for (auto& stage : stages_) {
            stage->accept(t);
        }
    }
}

void SamplerChain::reset() {
    for (auto& stage : stages_) {
        stage->reset();
    }
    if (constraint_) {
        constraint_->reset();
    }
    cur_.data.clear();
    cur_.sorted = false;
}

bool SamplerChain::is_stop_token(TokenId token) const { return stop_tokens_.count(token) != 0; }

std::vector<std::string_view> SamplerChain::stage_names() const {
    std::vector<std::string_view> names;
    names.reserve(stages_.size());
    for (const auto& s : stages_) {
        names.push_back(s->name());
    }
    return names;
}

SamplerChain build_chain(const SamplerConfig& config, std::shared_ptr<Constraint> constraint,
                         std::optional<std::size_t> vocab_size) {
    const ValidationResult v = validate(config, vocab_size);
    if (!v.ok()) {
        throw std::invalid_argument(v.to_string());
    }
    const std::uint64_t seed = config.seed.value_or(Rng::entropy_seed());
    SamplerChain chain(seed, config.greedy ? Selector::Greedy : Selector::Distribution);
    chain.set_constraint(std::move(constraint));
    chain.set_stop_tokens(config.stop_tokens);

    if (!config.logit_bias.empty()) {
        chain.add(std::make_unique<LogitBiasSampler>(config.logit_bias));
    }
    for (const StageKind kind : config.stage_order) {
        // Greedy keeps only stages that can change the argmax.
        if (config.greedy && kind != StageKind::Penalties) {
            continue;
        }
        chain.add(make_stage(kind, config));
    }
    return chain;
}

}  // namespace sonder::inference::sampling
