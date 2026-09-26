// Sampling stages. Semantics follow llama.cpp's documented sampler behaviour
// (tools/completion/README.md, MIT); the code is an independent
// implementation, see src/sampling/README.md.
#include "sonder/sampling/samplers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sonder::inference::sampling {

namespace {
std::size_t effective_min_keep(std::size_t min_keep) noexcept { return std::max<std::size_t>(min_keep, 1); }
}  // namespace

// ---------------------------------------------------------------- logit bias
LogitBiasSampler::LogitBiasSampler(std::vector<LogitBias> biases) {
    for (const auto& b : biases) {
        bias_[b.token] += b.bias;  // repeated entries accumulate
    }
}

void LogitBiasSampler::apply(Candidates& c) {
    if (bias_.empty()) {
        return;
    }
    for (auto& t : c.data) {
        if (auto it = bias_.find(t.id); it != bias_.end()) {
            t.logit += it->second;
        }
    }
    c.sorted = false;
}

std::unique_ptr<Sampler> LogitBiasSampler::clone() const { return std::make_unique<LogitBiasSampler>(*this); }

// ----------------------------------------------------------------- penalties
PenaltiesSampler::PenaltiesSampler(std::int32_t last_n, float repeat, float frequency, float presence)
    : last_n_(last_n), repeat_(repeat), frequency_(frequency), presence_(presence) {}

void PenaltiesSampler::apply(Candidates& c) {
    if (last_n_ == 0 || counts_.empty() ||
        (repeat_ == 1.0f && frequency_ == 0.0f && presence_ == 0.0f)) {
        return;
    }
    for (auto& t : c.data) {
        const auto it = counts_.find(t.id);
        if (it == counts_.end() || it->second <= 0) {
            continue;
        }
        if (t.logit <= 0.0f) {
            t.logit *= repeat_;
        } else {
            t.logit /= repeat_;
        }
        t.logit -= static_cast<float>(it->second) * frequency_ + presence_;
    }
    c.sorted = false;
}

void PenaltiesSampler::accept(TokenId token) {
    if (last_n_ == 0) {
        return;
    }
    history_.push_back(token);
    ++counts_[token];
    if (last_n_ > 0 && history_.size() > static_cast<std::size_t>(last_n_)) {
        const TokenId old = history_.front();
        history_.pop_front();
        if (auto it = counts_.find(old); it != counts_.end() && --it->second <= 0) {
            counts_.erase(it);
        }
    }
}

void PenaltiesSampler::reset() {
    history_.clear();
    counts_.clear();
}

std::int32_t PenaltiesSampler::count(TokenId token) const {
    const auto it = counts_.find(token);
    return it == counts_.end() ? 0 : it->second;
}

std::unique_ptr<Sampler> PenaltiesSampler::clone() const { return std::make_unique<PenaltiesSampler>(*this); }

// --------------------------------------------------------------------- top-k
void TopKSampler::apply(Candidates& c) {
    if (k_ <= 0 || static_cast<std::size_t>(k_) >= c.size()) {
        return;
    }
    const auto k = static_cast<std::size_t>(k_);
    if (!c.sorted) {
        std::partial_sort(c.data.begin(), c.data.begin() + static_cast<std::ptrdiff_t>(k), c.data.end(),
                          [](const TokenData& a, const TokenData& b) {
                              return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
                          });
    }
    c.data.resize(k);
    c.sorted = true;
}

std::unique_ptr<Sampler> TopKSampler::clone() const { return std::make_unique<TopKSampler>(*this); }

// --------------------------------------------------------------------- top-p
void TopPSampler::apply(Candidates& c) {
    if (p_ >= 1.0f || c.empty()) {
        return;
    }
    sort_canonical(c);
    if (!softmax(c)) {
        return;
    }
    const std::size_t min_keep = effective_min_keep(min_keep_);
    double cum = 0.0;
    std::size_t keep = c.size();
    for (std::size_t i = 0; i < c.size(); ++i) {
        cum += c.data[i].p;
        if (cum >= static_cast<double>(p_) && i + 1 >= min_keep) {
            keep = i + 1;
            break;
        }
    }
    c.data.resize(keep);
}

std::unique_ptr<Sampler> TopPSampler::clone() const { return std::make_unique<TopPSampler>(*this); }

// --------------------------------------------------------------------- min-p
void MinPSampler::apply(Candidates& c) {
    if (p_ <= 0.0f || c.empty()) {
        return;
    }
    sort_canonical(c);
    if (!softmax(c)) {
        return;
    }
    const float threshold = p_ * c.data.front().p;  // front is the max in canonical order
    std::size_t keep = 0;
    while (keep < c.size() && c.data[keep].p >= threshold) {
        ++keep;
    }
    keep = std::min(c.size(), std::max(keep, effective_min_keep(min_keep_)));
    c.data.resize(keep);
}

std::unique_ptr<Sampler> MinPSampler::clone() const { return std::make_unique<MinPSampler>(*this); }

// ----------------------------------------------------------------- typical-p
void TypicalPSampler::apply(Candidates& c) {
    if (p_ >= 1.0f || c.empty()) {
        return;
    }
    sort_canonical(c);
    if (!softmax(c)) {
        return;
    }
    double entropy = 0.0;
    for (const auto& t : c.data) {
        if (t.p > 0.0f) {
            entropy -= static_cast<double>(t.p) * std::log(static_cast<double>(t.p));
        }
    }
    std::vector<double> shifted(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double p = c.data[i].p;
        shifted[i] = p > 0.0 ? std::fabs(-std::log(p) - entropy) : std::numeric_limits<double>::infinity();
    }
    std::vector<std::size_t> order(c.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Stable: equal scores keep canonical order.
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return shifted[a] < shifted[b]; });

    const std::size_t min_keep = effective_min_keep(min_keep_);
    double cum = 0.0;
    std::size_t keep = c.size();
    for (std::size_t i = 0; i < order.size(); ++i) {
        cum += c.data[order[i]].p;
        if (cum > static_cast<double>(p_) && i + 1 >= min_keep) {
            keep = i + 1;
            break;
        }
    }
    std::vector<TokenData> kept;
    kept.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) {
        kept.push_back(c.data[order[i]]);
    }
    c.data = std::move(kept);
    c.sorted = false;
    sort_canonical(c);
}

std::unique_ptr<Sampler> TypicalPSampler::clone() const { return std::make_unique<TypicalPSampler>(*this); }

// --------------------------------------------------------------- temperature
void TemperatureSampler::apply(Candidates& c) {
    if (c.empty()) {
        return;
    }
    if (t_ <= 0.0f) {
        sort_canonical(c);
        if (c.data.front().logit != kNegInf) {
            c.data.resize(1);
        }
        return;
    }
    if (t_ == 1.0f) {
        return;
    }
    for (auto& t : c.data) {
        t.logit /= t_;
    }
    c.sorted = false;
}

std::unique_ptr<Sampler> TemperatureSampler::clone() const { return std::make_unique<TemperatureSampler>(*this); }

// ------------------------------------------------------------------- factory
std::unique_ptr<Sampler> make_stage(StageKind kind, const SamplerConfig& cfg) {
    const auto min_keep = static_cast<std::size_t>(std::max(cfg.min_keep, 0));
    switch (kind) {
    case StageKind::Penalties:
        return std::make_unique<PenaltiesSampler>(cfg.penalty_last_n, cfg.repeat_penalty, cfg.frequency_penalty,
                                                  cfg.presence_penalty);
    case StageKind::TopK:
        return std::make_unique<TopKSampler>(cfg.top_k);
    case StageKind::TypicalP:
        return std::make_unique<TypicalPSampler>(cfg.typical_p, min_keep);
    case StageKind::TopP:
        return std::make_unique<TopPSampler>(cfg.top_p, min_keep);
    case StageKind::MinP:
        return std::make_unique<MinPSampler>(cfg.min_p, min_keep);
    case StageKind::Temperature:
        return std::make_unique<TemperatureSampler>(cfg.temperature);
    }
    throw std::invalid_argument("unknown sampling stage kind");
}

}  // namespace sonder::inference::sampling
