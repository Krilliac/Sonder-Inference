#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "sonder/sampling/config.hpp"
#include "sonder/sampling/sampler.hpp"

namespace sonder::inference::sampling {

/// Adds a fixed bias to specific token logits. Tokens not present in the
/// candidate set are ignored.
class LogitBiasSampler final : public Sampler {
public:
    explicit LogitBiasSampler(std::vector<LogitBias> biases);
    [[nodiscard]] std::string_view name() const noexcept override { return "logit_bias"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    std::unordered_map<TokenId, float> bias_;
};

/// Repetition, frequency and presence penalties over the last `last_n`
/// accepted tokens (llama.cpp semantics):
///   if count(t) > 0:
///     logit = logit <= 0 ? logit * repeat : logit / repeat
///     logit -= count(t) * frequency + presence
class PenaltiesSampler final : public Sampler {
public:
    PenaltiesSampler(std::int32_t last_n, float repeat, float frequency, float presence);
    [[nodiscard]] std::string_view name() const noexcept override { return "penalties"; }
    void apply(Candidates& c) override;
    void accept(TokenId token) override;
    void reset() override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

    [[nodiscard]] std::int32_t count(TokenId token) const;

private:
    std::int32_t last_n_;
    float repeat_;
    float frequency_;
    float presence_;
    std::deque<TokenId> history_;
    std::unordered_map<TokenId, std::int32_t> counts_;
};

/// Keep the k highest-logit candidates. k <= 0 disables; k >= size keeps all.
class TopKSampler final : public Sampler {
public:
    explicit TopKSampler(std::int32_t k) : k_(k) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "top_k"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    std::int32_t k_;
};

/// Nucleus sampling: keep the smallest prefix (canonical order) whose
/// cumulative probability is >= p, but at least min_keep. p >= 1 disables.
class TopPSampler final : public Sampler {
public:
    TopPSampler(float p, std::size_t min_keep) : p_(p), min_keep_(min_keep) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "top_p"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    float p_;
    std::size_t min_keep_;
};

/// Keep candidates whose probability is >= p * max probability, but at least
/// min_keep. p <= 0 disables.
class MinPSampler final : public Sampler {
public:
    MinPSampler(float p, std::size_t min_keep) : p_(p), min_keep_(min_keep) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "min_p"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    float p_;
    std::size_t min_keep_;
};

/// Locally typical sampling (Meister et al. 2022): rank candidates by
/// |-log p - H| ascending and keep the smallest set whose cumulative
/// probability exceeds p, but at least min_keep. p >= 1 disables.
class TypicalPSampler final : public Sampler {
public:
    TypicalPSampler(float p, std::size_t min_keep) : p_(p), min_keep_(min_keep) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "typical_p"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    float p_;
    std::size_t min_keep_;
};

/// Divide logits by t. t <= 0 keeps only the argmax candidate (llama.cpp
/// "temp_ext" behaviour), which makes the following distribution step
/// deterministic.
class TemperatureSampler final : public Sampler {
public:
    explicit TemperatureSampler(float t) : t_(t) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "temperature"; }
    void apply(Candidates& c) override;
    [[nodiscard]] std::unique_ptr<Sampler> clone() const override;

private:
    float t_;
};

/// Build the stage for `kind` from `config` (used by build_chain and tests).
[[nodiscard]] std::unique_ptr<Sampler> make_stage(StageKind kind, const SamplerConfig& config);

}  // namespace sonder::inference::sampling
