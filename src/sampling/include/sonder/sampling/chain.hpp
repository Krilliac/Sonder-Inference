#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "sonder/sampling/config.hpp"
#include "sonder/sampling/constraint.hpp"
#include "sonder/sampling/rng.hpp"
#include "sonder/sampling/sampler.hpp"

namespace sonder::inference::sampling {

enum class SampleStatus : std::uint8_t {
    Ok,
    EmptyLogits,          ///< Input logits span was empty.
    NoViableCandidates,   ///< Every candidate ended at -inf (e.g. over-constrained).
};

[[nodiscard]] std::string_view to_string(SampleStatus status) noexcept;

struct SampleResult {
    SampleStatus status = SampleStatus::EmptyLogits;
    TokenId token = kInvalidToken;
    float probability = 0.0f;  ///< Probability of `token` in the final distribution.

    [[nodiscard]] bool ok() const noexcept { return status == SampleStatus::Ok; }
};

enum class Selector : std::uint8_t {
    Greedy,        ///< argmax (ties -> lowest token id)
    Distribution,  ///< seeded draw from softmax of the surviving candidates
};

/// Ordered pipeline: [constraint] -> stages (build_chain puts logit bias first)
/// -> selector.
///
/// Not thread-safe; one chain per sequence. The candidate buffer is reused, so
/// after warm-up sample() does not allocate unless the vocab grows.
class SamplerChain {
public:
    explicit SamplerChain(std::uint64_t seed = 0, Selector selector = Selector::Distribution);

    SamplerChain(const SamplerChain& other);
    SamplerChain& operator=(const SamplerChain& other);
    SamplerChain(SamplerChain&&) = default;
    SamplerChain& operator=(SamplerChain&&) = default;
    ~SamplerChain() = default;

    SamplerChain& add(std::unique_ptr<Sampler> stage);

    /// Install (or clear with nullptr) the grammar/constraint hook. It runs
    /// before every stage added with add().
    void set_constraint(std::shared_ptr<Constraint> constraint);
    void set_selector(Selector selector) noexcept { selector_ = selector; }
    void set_stop_tokens(std::vector<TokenId> tokens);

    /// Sample one token from raw logits (index = token id). NaN logits are
    /// treated as -inf. Does NOT call accept(); the caller commits the token.
    [[nodiscard]] SampleResult sample(std::span<const float> logits);

    /// Commit a generated token: updates penalty history and the constraint.
    void accept(TokenId token);

    /// Feed prompt tokens into penalty history without advancing the
    /// constraint (mirrors llama.cpp accepting the prompt with grammar off).
    void accept_prompt(std::span<const TokenId> tokens);

    void reset();
    void reseed(std::uint64_t seed) noexcept { rng_.reseed(seed); }

    [[nodiscard]] bool is_stop_token(TokenId token) const;
    [[nodiscard]] const Candidates& last_candidates() const noexcept { return cur_; }
    [[nodiscard]] std::vector<std::string_view> stage_names() const;
    [[nodiscard]] std::size_t stage_count() const noexcept { return stages_.size(); }

private:
    std::vector<std::unique_ptr<Sampler>> stages_;
    std::shared_ptr<Constraint> constraint_;
    std::unordered_set<TokenId> stop_tokens_;
    Selector selector_;
    Rng rng_;
    Candidates cur_;
};

/// Build a chain from a config. Throws std::invalid_argument (message from
/// ValidationResult::to_string()) if the config is invalid.
[[nodiscard]] SamplerChain build_chain(const SamplerConfig& config,
                                       std::shared_ptr<Constraint> constraint = nullptr,
                                       std::optional<std::size_t> vocab_size = std::nullopt);

}  // namespace sonder::inference::sampling
