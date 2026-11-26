// Sonder Inference: exact speculative decoding verifier.
//
// Provides two lossless verification couplings over post-processor target
// distribution `p` and optional proposal distribution `q` (over the same
// grammar-masked / filtered vocabulary support):
//
//   1. `SpeculativeVerificationMode::TargetMatch` (default reference path):
//      Samples `y_j ~ p_j` independently and accepts draft token `x_j` iff
//      `y_j == x_j`. Expected single-token acceptance is `sum_v p(v) * q(v)`.
//      For deterministic proposers (suffix, n-gram, greedy MTP) where `q` is a
//      point mass at `x*`, this equals `p(x*)` identically with zero extra
//      probability-vector overhead.
//
//   2. `SpeculativeVerificationMode::StochasticRejection` (opt-in):
//      Leviathan et al. / Chen et al. exact speculative rejection sampling.
//      Accepts `x_j ~ q_j` with probability `min(1, p_j(x_j) / q_j(x_j))`;
//      on rejection, samples the correction token from the exact normalized
//      residual `r(v) = max(0, p_j(v) - q_j(v)) / Z` (falling back safely to
//      sampling directly from `p_j` when `Z <= residual_min_mass` due to
//      floating-point rounding). Expected single-token acceptance is
//      `sum_v min(p(v), q(v)) = 1 - TV(p, q)`. Only activates when a
//      stochastic proposer supplies real post-processor proposal probabilities;
//      deterministic/greedy proposers stay on `TargetMatch`.
//
// Pure C++20 header with no backend dependencies so it can be unit-tested
// standalone under both default builds and SONDER_WITH_LLAMA_CPP=ON.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace sonder::inference {

// Verification coupling policy for speculative decoding.
enum class SpeculativeVerificationMode : std::uint8_t {
    TargetMatch = 0,          // Default exact target-sample matching reference path.
    StochasticRejection = 1,  // Opt-in Leviathan et al. exact rejection + residual resampling.
};

inline constexpr double kDefaultResidualMinMass = 1e-12;
inline constexpr std::int32_t kInvalidSpeculativeToken = -1;

[[nodiscard]] constexpr bool is_valid_speculative_verification_mode(SpeculativeVerificationMode mode) noexcept {
    return mode == SpeculativeVerificationMode::TargetMatch ||
           mode == SpeculativeVerificationMode::StochasticRejection;
}

[[nodiscard]] constexpr std::string_view to_string(SpeculativeVerificationMode mode) noexcept {
    switch (mode) {
        case SpeculativeVerificationMode::TargetMatch: return "target_match";
        case SpeculativeVerificationMode::StochasticRejection: return "stochastic_rejection";
    }
    return "unknown";
}

namespace detail {

[[nodiscard]] inline bool ascii_iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        const auto la = static_cast<unsigned char>(ca >= 'A' && ca <= 'Z' ? ca + ('a' - 'A') : ca);
        const auto lb = static_cast<unsigned char>(cb >= 'A' && cb <= 'Z' ? cb + ('a' - 'A') : cb);
        if (la != lb) return false;
    }
    return true;
}

[[nodiscard]] inline double elapsed_microseconds(std::chrono::steady_clock::time_point start) noexcept {
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    return (std::isfinite(us) && us > 0.0) ? us : 0.0;
}

}  // namespace detail

// Case-insensitive parser for SpeculativeVerificationMode.
// Accepts "target_match", "target-match", "exact_target_match",
// "exact-target-match", "stochastic_rejection", and "stochastic-rejection".
// Leaves `out` untouched and returns false on unknown input.
[[nodiscard]] inline bool parse_speculative_verification_mode(std::string_view name,
                                                              SpeculativeVerificationMode& out) noexcept {
    if (detail::ascii_iequals(name, "target_match") || detail::ascii_iequals(name, "target-match") ||
        detail::ascii_iequals(name, "exact_target_match") ||
        detail::ascii_iequals(name, "exact-target-match")) {
        out = SpeculativeVerificationMode::TargetMatch;
        return true;
    }
    if (detail::ascii_iequals(name, "stochastic_rejection") ||
        detail::ascii_iequals(name, "stochastic-rejection")) {
        out = SpeculativeVerificationMode::StochasticRejection;
        return true;
    }
    return false;
}

// Resolves the effective verification mode given a requested policy and whether
// the active proposer actually supplies post-processor stochastic proposal
// probabilities. Deterministic/greedy proposers always remain on `TargetMatch`.
[[nodiscard]] constexpr SpeculativeVerificationMode resolve_speculative_verification_mode(
    SpeculativeVerificationMode requested, bool proposer_supplies_probabilities) noexcept {
    if (requested == SpeculativeVerificationMode::StochasticRejection && proposer_supplies_probabilities) {
        return SpeculativeVerificationMode::StochasticRejection;
    }
    return SpeculativeVerificationMode::TargetMatch;
}

// Normalizes a non-negative weight/probability vector into `out` (in double
// precision, summing to 1.0) using max-rescaling so both tiny subnormals and
// very large unnormalized weights normalize cleanly without underflow/overflow.
// Non-positive or NaN entries are clamped to 0.0; +inf or zero total mass fails
// closed and returns false.
template <typename Real>
[[nodiscard]] inline bool normalize_probability_distribution(std::span<const Real> raw,
                                                             std::vector<double>& out) {
    static_assert(std::is_floating_point_v<Real>, "Real must be a floating-point type");
    if (raw.empty()) {
        out.clear();
        return false;
    }

    double max_val = 0.0;
    for (const Real entry : raw) {
        const double v = static_cast<double>(entry);
        if (std::isinf(v) && v > 0.0) {
            out.clear();
            return false;
        }
        if (std::isfinite(v) && v > max_val) {
            max_val = v;
        }
    }
    if (!(max_val > 0.0) || !std::isfinite(max_val)) {
        out.clear();
        return false;
    }

    out.resize(raw.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const double v = static_cast<double>(raw[i]);
        const double scaled = (std::isfinite(v) && v > 0.0) ? (v / max_val) : 0.0;
        out[i] = scaled;
        sum += scaled;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        out.clear();
        return false;
    }

    const double inv_sum = 1.0 / sum;
    for (double& p : out) {
        p *= inv_sum;
    }
    return true;
}

// Numerically stable softmax from logits into `out` (in double precision).
// `-inf` logits map cleanly to 0.0 (grammar-masked / banned tokens).
// Any NaN, `+inf`, or all-`-inf` (grammar deadlock) fails closed with false.
[[nodiscard]] inline bool softmax_logits_to_probabilities(std::span<const float> logits,
                                                          std::vector<double>& out) {
    if (logits.empty()) {
        out.clear();
        return false;
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    bool any_finite = false;
    for (const float raw : logits) {
        if (std::isnan(raw) || (std::isinf(raw) && raw > 0.0F)) {
            out.clear();
            return false;
        }
        if (std::isfinite(raw)) {
            const double v = static_cast<double>(raw);
            if (!any_finite || v > max_logit) {
                max_logit = v;
                any_finite = true;
            }
        }
    }
    if (!any_finite) {
        out.clear();
        return false;
    }

    out.resize(logits.size());
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float raw = logits[i];
        if (!std::isfinite(raw)) {
            out[i] = 0.0;
            continue;
        }
        const double w = std::exp(static_cast<double>(raw) - max_logit);
        out[i] = w;
        sum_exp += w;
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        out.clear();
        return false;
    }

    const double inv_sum = 1.0 / sum_exp;
    for (double& p : out) {
        p *= inv_sum;
    }
    return true;
}

// Exact Leviathan et al. acceptance probability `min(1, p_x / q_x)` hardened
// against `q_x == 0`, underflow, NaN, and floating-point division rounding:
//   - `p_x <= 0` or non-finite or `q_x` NaN => 0.0 (never accepts zero-prob or NaN)
//   - `q_x <= 0` (finite, with `p_x > 0`) => 1.0 (exact `q_x -> 0+` limit, no div-by-zero)
//   - `q_x == +inf` => 0.0
//   - `p_x >= q_x` => 1.0 (exact, avoids rounding below 1.0)
//   - `0 < p_x < q_x` => `clamp(p_x / q_x, 0.0, 1.0)`
[[nodiscard]] inline double speculative_acceptance_probability(double p_x, double q_x) noexcept {
    if (!std::isfinite(p_x) || p_x <= 0.0 || std::isnan(q_x)) {
        return 0.0;
    }
    if (q_x <= 0.0) {
        return 1.0;
    }
    if (!std::isfinite(q_x)) {
        return 0.0;
    }
    if (p_x >= q_x) {
        return 1.0;
    }
    const double ratio = p_x / q_x;
    if (!std::isfinite(ratio)) {
        return 0.0;
    }
    return std::clamp(ratio, 0.0, 1.0);
}

// Draws a uniform double strictly in `[0.0, 1.0)` from any standard URBG.
template <typename Rng>
[[nodiscard]] inline double sample_unit_interval(Rng& rng) {
    using ResultT = typename Rng::result_type;
    if constexpr (std::is_unsigned_v<ResultT> && sizeof(ResultT) == sizeof(std::uint64_t)) {
        if constexpr (Rng::min() == 0 && Rng::max() == std::numeric_limits<std::uint64_t>::max()) {
            constexpr double kInv2Pow53 = 1.0 / 9007199254740992.0;  // 2^-53
            return static_cast<double>(static_cast<std::uint64_t>(rng()) >> 11) * kInv2Pow53;
        }
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    if (!(u < 1.0)) {
        u = std::nextafter(1.0, 0.0);
    }
    if (u < 0.0) {
        u = 0.0;
    }
    return u;
}

// Categorical draw from non-negative `weights` with known positive `total_weight`.
// Never selects a zero-weight (e.g. grammar-masked) token even under tail
// floating-point rounding. Returns `kInvalidSpeculativeToken` on invalid input.
template <typename Rng>
[[nodiscard]] inline std::int32_t sample_categorical_weights(std::span<const double> weights,
                                                             double total_weight, Rng& rng) {
    if (weights.empty() || !(total_weight > 0.0) || !std::isfinite(total_weight)) {
        return kInvalidSpeculativeToken;
    }
    const double threshold = sample_unit_interval(rng) * total_weight;
    double cdf = 0.0;
    std::int32_t last_positive = kInvalidSpeculativeToken;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double w = weights[i];
        if (!std::isfinite(w) || w <= 0.0) {
            continue;
        }
        last_positive = static_cast<std::int32_t>(i);
        cdf += w;
        if (cdf > threshold) {
            return last_positive;
        }
    }
    return last_positive;
}

// Convenience categorical sampler that computes total positive weight.
template <typename Rng>
[[nodiscard]] inline std::int32_t sample_categorical(std::span<const double> probs, Rng& rng) {
    double total = 0.0;
    for (const double p : probs) {
        if (std::isfinite(p) && p > 0.0) {
            total += p;
        }
    }
    return sample_categorical_weights(probs, total, rng);
}

// Computes the unnormalized positive residual `out_residual[v] = max(0, p[v] - q[v])`
// and returns its total mass `Z = sum_v max(0, p[v] - q[v])`.
[[nodiscard]] inline double compute_residual_distribution(std::span<const double> p,
                                                          std::span<const double> q,
                                                          std::vector<double>& out_residual) {
    if (p.size() != q.size() || p.empty()) {
        out_residual.clear();
        return 0.0;
    }
    out_residual.resize(p.size());
    double z = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double diff = p[i] - q[i];
        const double pos = (std::isfinite(diff) && diff > 0.0) ? diff : 0.0;
        out_residual[i] = pos;
        z += pos;
    }
    return std::isfinite(z) ? z : 0.0;
}

// Analytic expected acceptance rate for TargetMatch: `A_match = sum_v p(v) q(v)`.
template <typename Real>
[[nodiscard]] inline double expected_target_match_acceptance(std::span<const Real> raw_p,
                                                             std::span<const Real> raw_q) {
    if (raw_p.size() != raw_q.size() || raw_p.empty()) return 0.0;
    std::vector<double> p;
    std::vector<double> q;
    if (!normalize_probability_distribution(raw_p, p) ||
        !normalize_probability_distribution(raw_q, q)) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        sum += p[i] * q[i];
    }
    return std::clamp(sum, 0.0, 1.0);
}

// Analytic expected acceptance rate for StochasticRejection:
// `A_reject = sum_v min(p(v), q(v)) = 1 - TV(p, q)`.
template <typename Real>
[[nodiscard]] inline double expected_stochastic_rejection_acceptance(std::span<const Real> raw_p,
                                                                     std::span<const Real> raw_q) {
    if (raw_p.size() != raw_q.size() || raw_p.empty()) return 0.0;
    std::vector<double> p;
    std::vector<double> q;
    if (!normalize_probability_distribution(raw_p, p) ||
        !normalize_probability_distribution(raw_q, q)) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        sum += (p[i] < q[i]) ? p[i] : q[i];
    }
    return std::clamp(sum, 0.0, 1.0);
}

// Total variation distance `TV(p, q) = 0.5 * sum_v |p(v) - q(v)|`.
template <typename Real>
[[nodiscard]] inline double total_variation_distance(std::span<const Real> raw_p,
                                                     std::span<const Real> raw_q) {
    if (raw_p.size() != raw_q.size() || raw_p.empty()) return 0.0;
    std::vector<double> p;
    std::vector<double> q;
    if (!normalize_probability_distribution(raw_p, p) ||
        !normalize_probability_distribution(raw_q, q)) {
        return 0.0;
    }
    double l1 = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        l1 += std::abs(p[i] - q[i]);
    }
    return std::clamp(0.5 * l1, 0.0, 1.0);
}

// Cumulative verification overhead and acceptance metrics.
struct SpeculativeVerifierMetrics {
    SpeculativeVerificationMode verifier_mode = SpeculativeVerificationMode::TargetMatch;
    std::uint64_t steps = 0;
    std::uint64_t draft_tokens = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t rejected_steps = 0;
    std::uint64_t residual_resamples = 0;
    std::uint64_t residual_fallbacks = 0;
    std::uint64_t bonus_samples = 0;
    double accept_rate = 0.0;
    double verify_overhead_us = 0.0;

    void reset(SpeculativeVerificationMode mode = SpeculativeVerificationMode::TargetMatch) noexcept {
        *this = SpeculativeVerifierMetrics{};
        verifier_mode = mode;
    }

    void record_step(SpeculativeVerificationMode effective_mode, std::size_t k_drafted,
                     std::size_t n_accepted, bool rejected, bool used_residual_resample,
                     bool used_residual_fallback, bool sampled_bonus, double elapsed_us) noexcept {
        verifier_mode = effective_mode;
        ++steps;
        draft_tokens += static_cast<std::uint64_t>(k_drafted);
        const std::size_t clamped_accepted = (n_accepted < k_drafted) ? n_accepted : k_drafted;
        accepted_tokens += static_cast<std::uint64_t>(clamped_accepted);
        if (rejected) ++rejected_steps;
        if (used_residual_resample) ++residual_resamples;
        if (used_residual_fallback) ++residual_fallbacks;
        if (sampled_bonus) ++bonus_samples;
        if (std::isfinite(elapsed_us) && elapsed_us > 0.0) {
            verify_overhead_us += elapsed_us;
        }
        accept_rate = draft_tokens > 0
                          ? static_cast<double>(accepted_tokens) / static_cast<double>(draft_tokens)
                          : 0.0;
    }
};

// Single-position verification outcome.
struct SpeculativeTokenDecision {
    bool ok = false;
    bool accepted = false;
    bool used_residual_resample = false;
    bool used_residual_fallback = false;
    std::int32_t token = kInvalidSpeculativeToken;
    double acceptance_prob = 0.0;
    double residual_mass = 0.0;
    SpeculativeVerificationMode effective_mode = SpeculativeVerificationMode::TargetMatch;
};

// Multi-token verification outcome for a draft sequence of length K >= 0.
struct SpeculativeSequenceResult {
    bool ok = false;
    SpeculativeVerificationMode effective_mode = SpeculativeVerificationMode::TargetMatch;
    std::size_t draft_tokens = 0;
    std::size_t accepted_draft_tokens = 0;
    bool rejected = false;
    bool used_residual_resample = false;
    bool used_residual_fallback = false;
    bool sampled_bonus = false;
    double verify_overhead_us = 0.0;
    // Emits `accepted_draft_tokens` accepted draft tokens followed by 1 terminal
    // token (either the rejection correction token or the K+1 bonus token).
    std::vector<std::int32_t> emitted_tokens;
};

struct SpeculativeVerifierConfig {
    SpeculativeVerificationMode mode = SpeculativeVerificationMode::TargetMatch;
    double residual_min_mass = kDefaultResidualMinMass;
};

// Reusable speculative decoding verifier holding scratch buffers and metrics.
class SpeculativeVerifier {
public:
    explicit SpeculativeVerifier(SpeculativeVerifierConfig config = {}) noexcept
        : config_(sanitize_config(config)) {
        metrics_.verifier_mode = config_.mode;
    }

    [[nodiscard]] SpeculativeVerificationMode requested_mode() const noexcept { return config_.mode; }

    void set_mode(SpeculativeVerificationMode mode) noexcept {
        config_.mode = is_valid_speculative_verification_mode(mode)
                           ? mode
                           : SpeculativeVerificationMode::TargetMatch;
    }

    [[nodiscard]] double residual_min_mass() const noexcept { return config_.residual_min_mass; }

    [[nodiscard]] const SpeculativeVerifierMetrics& metrics() const noexcept { return metrics_; }

    void reset_metrics() noexcept { metrics_.reset(config_.mode); }

    // Records an external verification step (e.g., direct llama.cpp fast-path
    // TargetMatch verification) into the same metrics summary.
    void record_external_step(SpeculativeVerificationMode effective_mode, std::size_t k_drafted,
                              std::size_t n_accepted, bool rejected, bool sampled_bonus,
                              double elapsed_us) noexcept {
        metrics_.record_step(effective_mode, k_drafted, n_accepted, rejected,
                             /*used_residual_resample=*/false, /*used_residual_fallback=*/false,
                             sampled_bonus, elapsed_us);
    }

    // Verifies a single proposed token `draft_token` against `target_probs` and
    // optional `proposal_probs`. If `proposal_probs` is empty, effective mode
    // resolves to `TargetMatch`. Updates cumulative metrics.
    template <typename Real, typename Rng>
    [[nodiscard]] SpeculativeTokenDecision verify_token(std::int32_t draft_token,
                                                        std::span<const Real> target_probs,
                                                        std::span<const Real> proposal_probs,
                                                        Rng& rng) {
        const auto start = std::chrono::steady_clock::now();
        const SpeculativeTokenDecision decision =
            verify_token_impl(draft_token, target_probs, proposal_probs, rng);
        if (decision.ok) {
            const double us = detail::elapsed_microseconds(start);
            metrics_.record_step(decision.effective_mode, /*k_drafted=*/1u,
                                 decision.accepted ? 1u : 0u, !decision.accepted,
                                 decision.used_residual_resample, decision.used_residual_fallback,
                                 /*sampled_bonus=*/false, us);
        }
        return decision;
    }

    // Verifies `K >= 0` draft tokens against `K + 1` target distributions and
    // either 0 (deterministic/no-prob proposer -> TargetMatch) or `K` proposal
    // distributions.
    template <typename Real, typename Rng>
    [[nodiscard]] SpeculativeSequenceResult verify_sequence(
        std::span<const std::int32_t> draft_tokens,
        std::span<const std::span<const Real>> target_dists,
        std::span<const std::span<const Real>> proposal_dists, Rng& rng) {
        const std::size_t k = draft_tokens.size();
        if (target_dists.size() != k + 1u) {
            return {};
        }
        if (!proposal_dists.empty() && proposal_dists.size() != k) {
            return {};
        }
        const bool has_proposals = !proposal_dists.empty() && k > 0u;
        return verify_sequence_lazy(
            draft_tokens, has_proposals,
            [&](std::size_t pos, std::vector<double>& out_p) -> bool {
                return normalize_probability_distribution(target_dists[pos], out_p);
            },
            [&](std::size_t pos, std::vector<double>& out_q) -> bool {
                return normalize_probability_distribution(proposal_dists[pos], out_q);
            },
            [](std::size_t, std::int32_t) noexcept {}, rng);
    }

    // Convenience overload accepting `std::vector<std::vector<Real>>`.
    template <typename Real, typename Rng>
    [[nodiscard]] SpeculativeSequenceResult verify_sequence(
        std::span<const std::int32_t> draft_tokens,
        const std::vector<std::vector<Real>>& target_dists,
        const std::vector<std::vector<Real>>& proposal_dists, Rng& rng) {
        const std::size_t k = draft_tokens.size();
        if (target_dists.size() != k + 1u) {
            return {};
        }
        if (!proposal_dists.empty() && proposal_dists.size() != k) {
            return {};
        }
        const bool has_proposals = !proposal_dists.empty() && k > 0u;
        return verify_sequence_lazy(
            draft_tokens, has_proposals,
            [&](std::size_t pos, std::vector<double>& out_p) -> bool {
                return normalize_probability_distribution(std::span<const Real>(target_dists[pos]),
                                                          out_p);
            },
            [&](std::size_t pos, std::vector<double>& out_q) -> bool {
                return normalize_probability_distribution(
                    std::span<const Real>(proposal_dists[pos]), out_q);
            },
            [](std::size_t, std::int32_t) noexcept {}, rng);
    }

    // Lazy callback-driven sequence verifier. Requests target distribution `p_j`
    // one position at a time, invokes `on_commit(j, tok)` immediately after each
    // committed token (so stateful grammar/penalty processors stay causally
    // synchronized before `p_{j+1}` is built), and stops immediately on rejection.
    template <typename TargetDistFn, typename ProposalDistFn, typename CommitFn, typename Rng>
    [[nodiscard]] SpeculativeSequenceResult verify_sequence_lazy(
        std::span<const std::int32_t> draft_tokens, bool has_proposal_dists,
        TargetDistFn&& get_target_dist, ProposalDistFn&& get_proposal_dist, CommitFn&& on_commit,
        Rng& rng) {
        const auto start = std::chrono::steady_clock::now();
        SpeculativeSequenceResult res;
        const std::size_t k = draft_tokens.size();
        res.draft_tokens = k;
        res.effective_mode =
            resolve_speculative_verification_mode(config_.mode, has_proposal_dists && k > 0u);
        res.emitted_tokens.reserve(k + 1u);

        for (std::size_t j = 0; j < k; ++j) {
            if (!get_target_dist(j, p_scratch_)) {
                return {};
            }
            if (!normalize_probability_distribution(std::span<const double>(p_scratch_), p_norm_)) {
                return {};
            }
            const std::int32_t draft_tok = draft_tokens[j];
            if (draft_tok < 0 || static_cast<std::size_t>(draft_tok) >= p_norm_.size()) {
                return {};
            }

            SpeculativeTokenDecision step;
            if (res.effective_mode == SpeculativeVerificationMode::StochasticRejection) {
                if (!get_proposal_dist(j, q_scratch_)) {
                    return {};
                }
                if (!normalize_probability_distribution(std::span<const double>(q_scratch_),
                                                        q_norm_) ||
                    q_norm_.size() != p_norm_.size()) {
                    return {};
                }
                step = verify_normalized_stochastic(draft_tok, p_norm_, q_norm_, rng);
            } else {
                step = verify_normalized_target_match(draft_tok, p_norm_, rng);
            }
            if (!step.ok) {
                return {};
            }

            res.emitted_tokens.push_back(step.token);
            on_commit(j, step.token);
            if (step.accepted) {
                ++res.accepted_draft_tokens;
            } else {
                res.rejected = true;
                res.used_residual_resample = step.used_residual_resample;
                res.used_residual_fallback = step.used_residual_fallback;
                break;
            }
        }

        if (!res.rejected) {
            if (!get_target_dist(k, p_scratch_)) {
                return {};
            }
            if (!normalize_probability_distribution(std::span<const double>(p_scratch_), p_norm_)) {
                return {};
            }
            const std::int32_t bonus = sample_categorical(p_norm_, rng);
            if (bonus == kInvalidSpeculativeToken) {
                return {};
            }
            res.emitted_tokens.push_back(bonus);
            res.sampled_bonus = true;
            on_commit(k, bonus);
        }

        res.ok = true;
        res.verify_overhead_us = detail::elapsed_microseconds(start);
        metrics_.record_step(res.effective_mode, k, res.accepted_draft_tokens, res.rejected,
                             res.used_residual_resample, res.used_residual_fallback,
                             res.sampled_bonus, res.verify_overhead_us);
        return res;
    }

private:
    static SpeculativeVerifierConfig sanitize_config(SpeculativeVerifierConfig cfg) noexcept {
        if (!is_valid_speculative_verification_mode(cfg.mode)) {
            cfg.mode = SpeculativeVerificationMode::TargetMatch;
        }
        if (!std::isfinite(cfg.residual_min_mass) || cfg.residual_min_mass < 0.0) {
            cfg.residual_min_mass = kDefaultResidualMinMass;
        }
        return cfg;
    }

    template <typename Real, typename Rng>
    SpeculativeTokenDecision verify_token_impl(std::int32_t draft_token,
                                               std::span<const Real> target_probs,
                                               std::span<const Real> proposal_probs, Rng& rng) {
        if (!normalize_probability_distribution(target_probs, p_norm_)) {
            return {};
        }
        if (draft_token < 0 || static_cast<std::size_t>(draft_token) >= p_norm_.size()) {
            return {};
        }
        const SpeculativeVerificationMode effective =
            resolve_speculative_verification_mode(config_.mode, !proposal_probs.empty());
        if (effective == SpeculativeVerificationMode::StochasticRejection) {
            if (proposal_probs.size() != p_norm_.size() ||
                !normalize_probability_distribution(proposal_probs, q_norm_)) {
                return {};
            }
            return verify_normalized_stochastic(draft_token, p_norm_, q_norm_, rng);
        }
        return verify_normalized_target_match(draft_token, p_norm_, rng);
    }

    template <typename Rng>
    SpeculativeTokenDecision verify_normalized_target_match(std::int32_t draft_token,
                                                            std::span<const double> p, Rng& rng) {
        SpeculativeTokenDecision d;
        d.effective_mode = SpeculativeVerificationMode::TargetMatch;
        const std::int32_t sampled = sample_categorical(p, rng);
        if (sampled == kInvalidSpeculativeToken) {
            return {};
        }
        d.ok = true;
        d.token = sampled;
        d.accepted = (sampled == draft_token);
        d.acceptance_prob = p[static_cast<std::size_t>(draft_token)];
        return d;
    }

    template <typename Rng>
    SpeculativeTokenDecision verify_normalized_stochastic(std::int32_t draft_token,
                                                          std::span<const double> p,
                                                          std::span<const double> q, Rng& rng) {
        SpeculativeTokenDecision d;
        d.effective_mode = SpeculativeVerificationMode::StochasticRejection;
        const auto idx = static_cast<std::size_t>(draft_token);
        const double alpha = speculative_acceptance_probability(p[idx], q[idx]);
        d.acceptance_prob = alpha;

        if (alpha >= 1.0) {
            d.ok = true;
            d.accepted = true;
            d.token = draft_token;
            return d;
        }
        if (alpha > 0.0 && sample_unit_interval(rng) < alpha) {
            d.ok = true;
            d.accepted = true;
            d.token = draft_token;
            return d;
        }

        // Rejected: sample correction token from normalized residual
        // r(v) = max(0, p(v) - q(v)) / Z, falling back cleanly to `p` when
        // Z <= residual_min_mass due to floating-point rounding.
        const double z = compute_residual_distribution(p, q, residual_scratch_);
        d.residual_mass = z;
        std::int32_t correction = kInvalidSpeculativeToken;
        if (z > config_.residual_min_mass && std::isfinite(z)) {
            correction = sample_categorical_weights(residual_scratch_, z, rng);
            if (correction != kInvalidSpeculativeToken) {
                d.used_residual_resample = true;
            }
        }
        if (correction == kInvalidSpeculativeToken) {
            correction = sample_categorical(p, rng);
            if (correction == kInvalidSpeculativeToken) {
                return {};
            }
            d.used_residual_fallback = true;
        }
        d.ok = true;
        d.accepted = false;
        d.token = correction;
        return d;
    }

    SpeculativeVerifierConfig config_{};
    SpeculativeVerifierMetrics metrics_{};
    std::vector<double> p_scratch_;
    std::vector<double> q_scratch_;
    std::vector<double> p_norm_;
    std::vector<double> q_norm_;
    std::vector<double> residual_scratch_;
};

}  // namespace sonder::inference
