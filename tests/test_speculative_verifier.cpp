// Standalone mathematical and distributional verification tests for exact
// speculative decoding (`sonder/inference/speculative_verifier.hpp`).
// Compiled under BOTH default builds (`ci-windows`) and `SONDER_WITH_LLAMA_CPP=ON`.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sonder/inference/speculative_verifier.hpp"

using namespace sonder::inference;

namespace {

double PearsonChiSquared(const std::vector<std::uint64_t>& counts, std::span<const double> expected_probs,
                         std::uint64_t total_trials, std::size_t& out_positive_bins) {
    out_positive_bins = 0;
    double chi2 = 0.0;
    for (std::size_t i = 0; i < expected_probs.size(); ++i) {
        const double p = expected_probs[i];
        if (p <= 0.0) continue;
        ++out_positive_bins;
        const double expected = p * static_cast<double>(total_trials);
        const double diff = static_cast<double>(counts[i]) - expected;
        chi2 += (diff * diff) / expected;
    }
    return chi2;
}

double EmpiricalTvDistance(const std::vector<std::uint64_t>& counts, std::span<const double> expected_probs,
                           std::uint64_t total_trials) {
    double l1 = 0.0;
    for (std::size_t i = 0; i < expected_probs.size(); ++i) {
        const double emp = static_cast<double>(counts[i]) / static_cast<double>(total_trials);
        l1 += std::abs(emp - expected_probs[i]);
    }
    return 0.5 * l1;
}

// Conservative upper critical value for chi-squared with `df` degrees of freedom
// at alpha ~ 1e-5 using Wilson-Hilferty approximation (z = 4.5 -> ~3.4e-6 tail).
double ChiSquaredUpperBound(std::size_t df) {
    if (df == 0) return 0.0;
    const double k = static_cast<double>(df);
    const double z = 4.5;
    const double h = 2.0 / (9.0 * k);
    const double cube = 1.0 - h + z * std::sqrt(h);
    return k * cube * cube * cube;
}

}  // namespace

TEST_CASE("speculative verifier: mode parsing, string conversion, and proposer-aware resolution") {
    CHECK(is_valid_speculative_verification_mode(SpeculativeVerificationMode::TargetMatch));
    CHECK(is_valid_speculative_verification_mode(SpeculativeVerificationMode::StochasticRejection));
    CHECK_FALSE(is_valid_speculative_verification_mode(static_cast<SpeculativeVerificationMode>(42)));

    CHECK(to_string(SpeculativeVerificationMode::TargetMatch) == "target_match");
    CHECK(to_string(SpeculativeVerificationMode::StochasticRejection) == "stochastic_rejection");
    CHECK(to_string(static_cast<SpeculativeVerificationMode>(99)) == "unknown");

    SpeculativeVerificationMode parsed = SpeculativeVerificationMode::StochasticRejection;
    CHECK(parse_speculative_verification_mode("target_match", parsed));
    CHECK(parsed == SpeculativeVerificationMode::TargetMatch);
    CHECK(parse_speculative_verification_mode("STOCHASTIC-REJECTION", parsed));
    CHECK(parsed == SpeculativeVerificationMode::StochasticRejection);
    CHECK(parse_speculative_verification_mode("Exact-Target-Match", parsed));
    CHECK(parsed == SpeculativeVerificationMode::TargetMatch);
    CHECK(parse_speculative_verification_mode("stochastic_rejection", parsed));
    CHECK(parsed == SpeculativeVerificationMode::StochasticRejection);

    // Unknown strings fail without modifying output.
    for (const char* bad : {"", "greedy", "rejection", "target", "stochastic_rejection "}) {
        SpeculativeVerificationMode untouched = SpeculativeVerificationMode::StochasticRejection;
        CHECK_FALSE(parse_speculative_verification_mode(bad, untouched));
        CHECK(untouched == SpeculativeVerificationMode::StochasticRejection);
    }

    // Proposer-aware resolution: StochasticRejection activates ONLY when opted in
    // AND the proposer supplies real stochastic proposal probabilities.
    CHECK(resolve_speculative_verification_mode(SpeculativeVerificationMode::TargetMatch, false) ==
          SpeculativeVerificationMode::TargetMatch);
    CHECK(resolve_speculative_verification_mode(SpeculativeVerificationMode::TargetMatch, true) ==
          SpeculativeVerificationMode::TargetMatch);
    CHECK(resolve_speculative_verification_mode(SpeculativeVerificationMode::StochasticRejection, false) ==
          SpeculativeVerificationMode::TargetMatch);
    CHECK(resolve_speculative_verification_mode(SpeculativeVerificationMode::StochasticRejection, true) ==
          SpeculativeVerificationMode::StochasticRejection);
}

TEST_CASE("speculative verifier: analytic acceptance on p = q = (0.5, 0.5) gives 100% vs 50%") {
    const std::array<double, 2> p{0.5, 0.5};
    const std::array<double, 2> q{0.5, 0.5};

    CHECK(expected_stochastic_rejection_acceptance<double>(p, q) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(expected_target_match_acceptance<double>(p, q) == doctest::Approx(0.5).epsilon(1e-12));
    CHECK(total_variation_distance<double>(p, q) == doctest::Approx(0.0).epsilon(1e-12));

    constexpr std::uint64_t kTrials = 60000;
    std::mt19937_64 draft_rng_sr(0x1111222233334444ULL);
    std::mt19937_64 verify_rng_sr(0x5555666677778888ULL);
    SpeculativeVerifier sr_verifier({SpeculativeVerificationMode::StochasticRejection});

    std::vector<std::uint64_t> sr_counts(2, 0);
    for (std::uint64_t t = 0; t < kTrials; ++t) {
        const std::int32_t draft = sample_categorical(q, draft_rng_sr);
        const SpeculativeTokenDecision d = sr_verifier.verify_token<double>(draft, p, q, verify_rng_sr);
        REQUIRE(d.ok);
        CHECK(d.accepted);
        CHECK(d.token == draft);
        CHECK(d.effective_mode == SpeculativeVerificationMode::StochasticRejection);
        ++sr_counts[static_cast<std::size_t>(d.token)];
    }

    const SpeculativeVerifierMetrics& sr_m = sr_verifier.metrics();
    CHECK(sr_m.verifier_mode == SpeculativeVerificationMode::StochasticRejection);
    CHECK(sr_m.steps == kTrials);
    CHECK(sr_m.draft_tokens == kTrials);
    CHECK(sr_m.accepted_tokens == kTrials);
    CHECK(sr_m.accept_rate == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(sr_m.residual_resamples == 0u);
    CHECK(sr_m.residual_fallbacks == 0u);
    CHECK(sr_m.verify_overhead_us >= 0.0);
    CHECK(std::isfinite(sr_m.verify_overhead_us));
    CHECK(EmpiricalTvDistance(sr_counts, p, kTrials) < 0.008);

    std::mt19937_64 draft_rng_tm(0x1111222233334444ULL);
    std::mt19937_64 verify_rng_tm(0x9999AAAABBBBCCCCULL);
    SpeculativeVerifier tm_verifier({SpeculativeVerificationMode::TargetMatch});

    std::vector<std::uint64_t> tm_counts(2, 0);
    for (std::uint64_t t = 0; t < kTrials; ++t) {
        const std::int32_t draft = sample_categorical(q, draft_rng_tm);
        const SpeculativeTokenDecision d = tm_verifier.verify_token<double>(draft, p, q, verify_rng_tm);
        REQUIRE(d.ok);
        CHECK(d.effective_mode == SpeculativeVerificationMode::TargetMatch);
        ++tm_counts[static_cast<std::size_t>(d.token)];
    }

    const SpeculativeVerifierMetrics& tm_m = tm_verifier.metrics();
    CHECK(tm_m.verifier_mode == SpeculativeVerificationMode::TargetMatch);
    CHECK(tm_m.draft_tokens == kTrials);
    CHECK(tm_m.accept_rate == doctest::Approx(0.5).epsilon(0.01));
    CHECK(tm_m.residual_resamples == 0u);
    CHECK(tm_m.verify_overhead_us >= 0.0);
    CHECK(EmpiricalTvDistance(tm_counts, p, kTrials) < 0.008);
}

TEST_CASE("speculative verifier: exact target distribution preservation across skewed, multimodal, mismatched, and grammar-masked distributions") {
    struct DistributionCase {
        const char* name;
        std::vector<double> p;
        std::vector<double> q;
    };

    const std::vector<DistributionCase> cases{
        {
            "skewed",
            {0.72, 0.18, 0.06, 0.025, 0.01, 0.005},
            {0.35, 0.30, 0.18, 0.10, 0.05, 0.02},
        },
        {
            "multimodal",
            {0.02, 0.38, 0.05, 0.04, 0.03, 0.41, 0.04, 0.03},
            {0.22, 0.16, 0.12, 0.05, 0.18, 0.12, 0.10, 0.05},
        },
        {
            "mismatched_zero_support",
            // Token 1 and 4: q(v) == 0 where p(v) > 0 (must appear via residual!)
            // Token 3 and 5: p(v) == 0 where q(v) > 0 (must NEVER be emitted!)
            {0.30, 0.25, 0.20, 0.0, 0.25, 0.0},
            {0.40, 0.0, 0.15, 0.25, 0.0, 0.20},
        },
        {
            "grammar_masked_sparse",
            // 16-token vocab with 10 grammar-masked zero entries and 1 q==0/p>0 entry.
            {0.0, 0.32, 0.0, 0.0, 0.18, 0.0, 0.24, 0.0, 0.0, 0.14, 0.0, 0.0, 0.08, 0.0, 0.04, 0.0},
            {0.0, 0.20, 0.0, 0.0, 0.35, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0, 0.10, 0.0},
        },
    };

    constexpr std::uint64_t kTrials = 80000;

    for (std::size_t c_idx = 0; c_idx < cases.size(); ++c_idx) {
        const auto& tc = cases[c_idx];
        CAPTURE(tc.name);

        const double analytic_sr = expected_stochastic_rejection_acceptance<double>(tc.p, tc.q);
        const double analytic_tm = expected_target_match_acceptance<double>(tc.p, tc.q);
        const double tv_pq = total_variation_distance<double>(tc.p, tc.q);
        CHECK(analytic_sr == doctest::Approx(1.0 - tv_pq).epsilon(1e-12));
        CHECK(analytic_sr >= analytic_tm - 1e-12);

        // 1. StochasticRejection mode.
        {
            SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
            std::mt19937_64 draft_rng(0x10000000ULL + c_idx * 97ULL);
            std::mt19937_64 verify_rng(0x20000000ULL + c_idx * 131ULL);
            std::vector<std::uint64_t> counts(tc.p.size(), 0);

            for (std::uint64_t t = 0; t < kTrials; ++t) {
                const std::int32_t draft = sample_categorical(tc.q, draft_rng);
                const SpeculativeTokenDecision d = verifier.verify_token<double>(draft, tc.p, tc.q, verify_rng);
                REQUIRE(d.ok);
                REQUIRE(d.token >= 0);
                REQUIRE(static_cast<std::size_t>(d.token) < tc.p.size());
                ++counts[static_cast<std::size_t>(d.token)];
            }

            // Hard zero-support invariant: forbidden / grammar-masked tokens NEVER emitted.
            for (std::size_t v = 0; v < tc.p.size(); ++v) {
                if (tc.p[v] == 0.0) {
                    CHECK(counts[v] == 0u);
                }
            }

            const double emp_tv = EmpiricalTvDistance(counts, tc.p, kTrials);
            std::size_t pos_bins = 0;
            const double chi2 = PearsonChiSquared(counts, tc.p, kTrials, pos_bins);
            REQUIRE(pos_bins >= 2u);
            CHECK(emp_tv < 0.008);
            CHECK(chi2 < ChiSquaredUpperBound(pos_bins - 1u));

            const SpeculativeVerifierMetrics& m = verifier.metrics();
            CHECK(m.accept_rate == doctest::Approx(analytic_sr).epsilon(0.01));
            CHECK(m.residual_resamples > 0u);
            CHECK(m.residual_fallbacks == 0u);
        }

        // 2. TargetMatch reference mode on the same (p, q).
        {
            SpeculativeVerifier verifier({SpeculativeVerificationMode::TargetMatch});
            std::mt19937_64 draft_rng(0x30000000ULL + c_idx * 97ULL);
            std::mt19937_64 verify_rng(0x40000000ULL + c_idx * 131ULL);
            std::vector<std::uint64_t> counts(tc.p.size(), 0);

            for (std::uint64_t t = 0; t < kTrials; ++t) {
                const std::int32_t draft = sample_categorical(tc.q, draft_rng);
                const SpeculativeTokenDecision d = verifier.verify_token<double>(draft, tc.p, tc.q, verify_rng);
                REQUIRE(d.ok);
                ++counts[static_cast<std::size_t>(d.token)];
            }

            for (std::size_t v = 0; v < tc.p.size(); ++v) {
                if (tc.p[v] == 0.0) {
                    CHECK(counts[v] == 0u);
                }
            }

            const double emp_tv = EmpiricalTvDistance(counts, tc.p, kTrials);
            std::size_t pos_bins = 0;
            const double chi2 = PearsonChiSquared(counts, tc.p, kTrials, pos_bins);
            CHECK(emp_tv < 0.008);
            CHECK(chi2 < ChiSquaredUpperBound(pos_bins - 1u));
            CHECK(verifier.metrics().accept_rate == doctest::Approx(analytic_tm).epsilon(0.01));
        }
    }

    // Float32 input overload preserves exact target distribution as well.
    {
        const std::vector<float> pf{0.15F, 0.55F, 0.0F, 0.30F};
        const std::vector<float> qf{0.40F, 0.25F, 0.20F, 0.15F};
        const std::vector<double> pd{0.15, 0.55, 0.0, 0.30};
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 draft_rng(0xA1B2C3D4ULL);
        std::mt19937_64 verify_rng(0xE5F60718ULL);
        std::vector<std::uint64_t> counts(4, 0);
        const std::vector<double> qd{0.40, 0.25, 0.20, 0.15};
        for (std::uint64_t t = 0; t < 50000; ++t) {
            const std::int32_t draft = sample_categorical(qd, draft_rng);
            const SpeculativeTokenDecision d =
                verifier.verify_token<float>(draft, std::span<const float>(pf), std::span<const float>(qf), verify_rng);
            REQUIRE(d.ok);
            ++counts[static_cast<std::size_t>(d.token)];
        }
        CHECK(counts[2] == 0u);
        std::size_t pos_bins = 0;
        CHECK(EmpiricalTvDistance(counts, pd, 50000) < 0.009);
        CHECK(PearsonChiSquared(counts, pd, 50000, pos_bins) < ChiSquaredUpperBound(pos_bins - 1u));
    }
}

TEST_CASE("speculative verifier: multi-token sequence verification (K=4) preserves exact marginal and joint distribution") {
    // 2nd-order autoregressive distributions over V=3 tokens (3^4 = 81 4-token sequences).
    // Includes a zero-probability transition (prev == 2 -> next == 2 is forbidden in P)
    // to verify structural/grammar masking is preserved across multi-token speculative blocks.
    constexpr std::size_t kVocab = 3;
    constexpr std::size_t kSeqLen = 4;
    constexpr std::size_t kNumJoint = 81;  // 3^4

    auto target_dist_for = [](const std::vector<std::int32_t>& prefix) -> std::vector<double> {
        const std::size_t len = prefix.size();
        if (len == 0) return {0.50, 0.35, 0.15};
        const std::int32_t last = prefix.back();
        const std::int32_t prev = len >= 2 ? prefix[len - 2] : 0;
        if (last == 2) {
            // Forbidden self-transition 2 -> 2 (exact zero mass).
            return (prev == 0) ? std::vector<double>{0.65, 0.35, 0.0} : std::vector<double>{0.30, 0.70, 0.0};
        }
        if (last == 0) {
            return (prev == 1) ? std::vector<double>{0.20, 0.55, 0.25} : std::vector<double>{0.45, 0.35, 0.20};
        }
        return (prev == 2) ? std::vector<double>{0.50, 0.20, 0.30} : std::vector<double>{0.25, 0.45, 0.30};
    };

    auto proposal_dist_for = [](const std::vector<std::int32_t>& prefix) -> std::vector<double> {
        const std::size_t len = prefix.size();
        if (len == 0) return {0.35, 0.40, 0.25};
        const std::int32_t last = prefix.back();
        if (last == 2) return {0.45, 0.45, 0.10};  // Proposes forbidden token 2 with 10% prob!
        if (last == 0) return {0.30, 0.45, 0.25};
        return {0.35, 0.35, 0.30};
    };

    // Compute exact analytic joint probability for all 81 4-token sequences + marginals.
    std::vector<double> exact_joint(kNumJoint, 0.0);
    std::array<std::vector<double>, kSeqLen> exact_marginals;
    for (auto& m : exact_marginals) m.assign(kVocab, 0.0);

    for (std::int32_t t0 = 0; t0 < 3; ++t0) {
        const auto p0 = target_dist_for({});
        for (std::int32_t t1 = 0; t1 < 3; ++t1) {
            const auto p1 = target_dist_for({t0});
            for (std::int32_t t2 = 0; t2 < 3; ++t2) {
                const auto p2 = target_dist_for({t0, t1});
                for (std::int32_t t3 = 0; t3 < 3; ++t3) {
                    const auto p3 = target_dist_for({t0, t1, t2});
                    const double prob = p0[static_cast<std::size_t>(t0)] * p1[static_cast<std::size_t>(t1)] *
                                        p2[static_cast<std::size_t>(t2)] * p3[static_cast<std::size_t>(t3)];
                    const std::size_t code = static_cast<std::size_t>(t0 * 27 + t1 * 9 + t2 * 3 + t3);
                    exact_joint[code] = prob;
                    exact_marginals[0][static_cast<std::size_t>(t0)] += prob;
                    exact_marginals[1][static_cast<std::size_t>(t1)] += prob;
                    exact_marginals[2][static_cast<std::size_t>(t2)] += prob;
                    exact_marginals[3][static_cast<std::size_t>(t3)] += prob;
                }
            }
        }
    }
    const double joint_sum = std::accumulate(exact_joint.begin(), exact_joint.end(), 0.0);
    REQUIRE(joint_sum == doctest::Approx(1.0).epsilon(1e-12));

    auto run_monte_carlo = [&](SpeculativeVerificationMode mode, std::uint64_t seed) -> double {
        constexpr std::uint64_t kSequences = 120000;
        SpeculativeVerifier verifier({mode});
        std::mt19937_64 draft_rng(seed);
        std::mt19937_64 verify_rng(seed ^ 0xF00DBABECAFEBABEULL);

        std::vector<std::uint64_t> joint_counts(kNumJoint, 0);
        std::array<std::vector<std::uint64_t>, kSeqLen> marginal_counts;
        for (auto& mc : marginal_counts) mc.assign(kVocab, 0);
        std::uint64_t full_k4_bonus_steps = 0;

        std::vector<std::int32_t> generated;
        std::vector<std::int32_t> draft_ctx;
        std::vector<std::int32_t> draft_tokens;
        std::vector<std::vector<double>> target_dists;
        std::vector<std::vector<double>> proposal_dists;

        for (std::uint64_t seq = 0; seq < kSequences; ++seq) {
            generated.clear();
            while (generated.size() < kSeqLen) {
                // Always draft K=4 tokens on the first step to exercise full K=4 verification,
                // then draft remaining tokens on subsequent steps if any rejection occurred.
                const std::size_t k = (generated.empty()) ? kSeqLen : (kSeqLen - generated.size());
                draft_ctx = generated;
                draft_tokens.clear();
                proposal_dists.clear();
                target_dists.clear();

                for (std::size_t j = 0; j < k; ++j) {
                    target_dists.push_back(target_dist_for(draft_ctx));
                    auto q_j = proposal_dist_for(draft_ctx);
                    const std::int32_t x_j = sample_categorical(q_j, draft_rng);
                    proposal_dists.push_back(std::move(q_j));
                    draft_tokens.push_back(x_j);
                    draft_ctx.push_back(x_j);
                }
                // Bonus target distribution p_K conditioned on full accepted draft prefix.
                target_dists.push_back(target_dist_for(draft_ctx));

                const SpeculativeSequenceResult res =
                    verifier.verify_sequence<double>(draft_tokens, target_dists, proposal_dists, verify_rng);
                REQUIRE(res.ok);
                REQUIRE(res.emitted_tokens.size() == res.accepted_draft_tokens + 1u);
                CHECK(res.sampled_bonus == (res.accepted_draft_tokens == k));
                CHECK(res.rejected == !res.sampled_bonus);
                if (k == kSeqLen && res.sampled_bonus) {
                    ++full_k4_bonus_steps;
                }

                for (const std::int32_t tok : res.emitted_tokens) {
                    if (generated.size() < kSeqLen) {
                        generated.push_back(tok);
                    }
                }
            }

            const std::size_t code = static_cast<std::size_t>(
                generated[0] * 27 + generated[1] * 9 + generated[2] * 3 + generated[3]);
            ++joint_counts[code];
            for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
                ++marginal_counts[pos][static_cast<std::size_t>(generated[pos])];
            }
        }

        // Verify K=4 full-acceptance bonus path was actually exercised.
        CHECK(full_k4_bonus_steps > 1000u);

        // Forbidden sequences (containing 2 -> 2) must have literally zero occurrences!
        for (std::size_t c = 0; c < kNumJoint; ++c) {
            if (exact_joint[c] == 0.0) {
                CHECK(joint_counts[c] == 0u);
            }
        }

        // Check every position marginal matches exact autoregressive target marginals.
        for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
            CHECK(EmpiricalTvDistance(marginal_counts[pos], exact_marginals[pos], kSequences) < 0.006);
            std::size_t m_bins = 0;
            const double m_chi2 = PearsonChiSquared(marginal_counts[pos], exact_marginals[pos], kSequences, m_bins);
            CHECK(m_chi2 < ChiSquaredUpperBound(m_bins - 1u));
        }

        // Check full 4-token joint distribution across all positive outcomes.
        CHECK(EmpiricalTvDistance(joint_counts, exact_joint, kSequences) < 0.012);
        std::size_t j_bins = 0;
        const double j_chi2 = PearsonChiSquared(joint_counts, exact_joint, kSequences, j_bins);
        REQUIRE(j_bins > 20u);
        CHECK(j_chi2 < ChiSquaredUpperBound(j_bins - 1u));

        return verifier.metrics().accept_rate;
    };

    const double sr_accept = run_monte_carlo(SpeculativeVerificationMode::StochasticRejection, 0xCAFE1234ULL);
    const double tm_accept = run_monte_carlo(SpeculativeVerificationMode::TargetMatch, 0xCAFE5678ULL);
    // StochasticRejection must strictly outperform independent TargetMatch on a stochastic drafter.
    CHECK(sr_accept > tm_accept + 0.15);
}

TEST_CASE("speculative verifier: deterministic q equivalence matches TargetMatch identically") {
    const std::vector<double> p{0.10, 0.25, 0.45, 0.15, 0.05};
    const std::int32_t x_star = 2;
    const std::vector<double> q_det{0.0, 0.0, 1.0, 0.0, 0.0};

    // Analytic identity: for deterministic q = delta_{x*}, sum min(p, q) == sum p*q == p(x*).
    CHECK(expected_stochastic_rejection_acceptance<double>(p, q_det) == doctest::Approx(p[2]).epsilon(1e-12));
    CHECK(expected_target_match_acceptance<double>(p, q_det) == doctest::Approx(p[2]).epsilon(1e-12));

    constexpr std::uint64_t kTrials = 80000;
    auto run_det = [&](SpeculativeVerificationMode mode, bool supply_q, std::uint64_t seed) {
        SpeculativeVerifier verifier({mode});
        std::mt19937_64 rng(seed);
        std::vector<std::uint64_t> counts(p.size(), 0);
        const std::span<const double> q_span = supply_q ? std::span<const double>(q_det) : std::span<const double>{};

        for (std::uint64_t t = 0; t < kTrials; ++t) {
            const SpeculativeTokenDecision d = verifier.verify_token<double>(x_star, p, q_span, rng);
            REQUIRE(d.ok);
            const SpeculativeVerificationMode expected_eff =
                resolve_speculative_verification_mode(mode, supply_q);
            CHECK(d.effective_mode == expected_eff);
            ++counts[static_cast<std::size_t>(d.token)];
        }

        CHECK(verifier.metrics().accept_rate == doctest::Approx(p[2]).epsilon(0.008));
        CHECK(EmpiricalTvDistance(counts, p, kTrials) < 0.008);
        std::size_t bins = 0;
        CHECK(PearsonChiSquared(counts, p, kTrials, bins) < ChiSquaredUpperBound(bins - 1u));
    };

    // 1. StochasticRejection with explicit one-hot deterministic q.
    run_det(SpeculativeVerificationMode::StochasticRejection, /*supply_q=*/true, 0xD1D1D1ULL);
    // 2. TargetMatch reference path.
    run_det(SpeculativeVerificationMode::TargetMatch, /*supply_q=*/true, 0xD2D2D2ULL);
    // 3. StochasticRejection with empty q (deterministic proposer without q -> stays on TargetMatch).
    run_det(SpeculativeVerificationMode::StochasticRejection, /*supply_q=*/false, 0xD3D3D3ULL);

    // Deterministic q on a zero-probability target token: 0% acceptance, exact target distribution p.
    {
        const std::vector<double> p_zero{0.4, 0.6, 0.0};
        const std::vector<double> q_zero{0.0, 0.0, 1.0};
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xD4D4D4ULL);
        std::vector<std::uint64_t> counts(3, 0);
        for (std::uint64_t t = 0; t < 50000; ++t) {
            const SpeculativeTokenDecision d = verifier.verify_token<double>(2, p_zero, q_zero, rng);
            REQUIRE(d.ok);
            CHECK_FALSE(d.accepted);
            CHECK(d.used_residual_resample);
            ++counts[static_cast<std::size_t>(d.token)];
        }
        CHECK(verifier.metrics().accept_rate == 0.0);
        CHECK(counts[2] == 0u);
        CHECK(EmpiricalTvDistance(counts, p_zero, 50000) < 0.008);
    }

    // Deterministic q matching deterministic target: 100% acceptance, zero rejections.
    {
        const std::vector<double> one_hot{0.0, 1.0, 0.0};
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xD5D5D5ULL);
        for (int t = 0; t < 100; ++t) {
            const SpeculativeTokenDecision d = verifier.verify_token<double>(1, one_hot, one_hot, rng);
            REQUIRE(d.ok);
            CHECK(d.accepted);
            CHECK(d.token == 1);
        }
        CHECK(verifier.metrics().accept_rate == 1.0);
        CHECK(verifier.metrics().residual_resamples == 0u);
    }
}

TEST_CASE("speculative verifier: numerical edge cases, unnormalized inputs, tiny probs, Z ~ 0, q(x) == 0, and K == 0") {
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    const double inf_v = std::numeric_limits<double>::infinity();

    // 1. Direct acceptance probability hardening (no NaN, no div-by-zero).
    CHECK(speculative_acceptance_probability(0.0, 0.0) == 0.0);
    CHECK(speculative_acceptance_probability(0.0, 0.5) == 0.0);
    CHECK(speculative_acceptance_probability(0.5, 0.0) == 1.0);
    CHECK(speculative_acceptance_probability(0.5, 0.5) == 1.0);
    CHECK(speculative_acceptance_probability(1e-300, 1e-300) == 1.0);
    CHECK(speculative_acceptance_probability(1e-300, 2e-300) == doctest::Approx(0.5).epsilon(1e-12));
    CHECK(speculative_acceptance_probability(nan_v, 0.5) == 0.0);
    CHECK(speculative_acceptance_probability(0.5, nan_v) == 0.0);
    CHECK(speculative_acceptance_probability(0.5, inf_v) == 0.0);
    CHECK(speculative_acceptance_probability(inf_v, 0.5) == 0.0);
    CHECK(speculative_acceptance_probability(-0.1, 0.5) == 0.0);

    // 2. Unnormalized huge and subnormal tiny probability vectors normalize cleanly.
    {
        std::vector<double> norm;
        const std::vector<double> huge{2e200, 6e200};
        REQUIRE(normalize_probability_distribution<double>(huge, norm));
        CHECK(norm[0] == doctest::Approx(0.25).epsilon(1e-12));
        CHECK(norm[1] == doctest::Approx(0.75).epsilon(1e-12));

        const std::vector<double> tiny{1e-320, 3e-320};
        REQUIRE(normalize_probability_distribution<double>(tiny, norm));
        CHECK(norm[0] == doctest::Approx(0.25).epsilon(1e-12));
        CHECK(norm[1] == doctest::Approx(0.75).epsilon(1e-12));

        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xE1E1E1ULL);
        const SpeculativeTokenDecision d_huge = verifier.verify_token<double>(0, huge, huge, rng);
        REQUIRE(d_huge.ok);
        CHECK(d_huge.accepted);
        CHECK(d_huge.acceptance_prob == 1.0);

        const SpeculativeTokenDecision d_tiny = verifier.verify_token<double>(1, tiny, tiny, rng);
        REQUIRE(d_tiny.ok);
        CHECK(d_tiny.accepted);
        CHECK(d_tiny.acceptance_prob == 1.0);

        // Softmax stability on huge shifted logits with grammar-masked -inf entries.
        const std::vector<float> logits{10000.0F, 10000.0F, -std::numeric_limits<float>::infinity()};
        REQUIRE(softmax_logits_to_probabilities(logits, norm));
        CHECK(norm[0] == doctest::Approx(0.5).epsilon(1e-12));
        CHECK(norm[1] == doctest::Approx(0.5).epsilon(1e-12));
        CHECK(norm[2] == 0.0);
        CHECK_FALSE(softmax_logits_to_probabilities(std::vector<float>{-std::numeric_limits<float>::infinity()}, norm));
        CHECK_FALSE(softmax_logits_to_probabilities(std::vector<float>{std::numeric_limits<float>::quiet_NaN()}, norm));
        CHECK_FALSE(softmax_logits_to_probabilities(std::vector<float>{std::numeric_limits<float>::infinity()}, norm));
    }

    // 3. Direct q(x) == 0 handling in verify_token (underflow / zero-support safety).
    {
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xE2E2E2ULL);
        // p(0) > 0, q(0) == 0 -> accepts without div-by-zero or NaN.
        const std::vector<double> p{0.4, 0.6};
        const std::vector<double> q{0.0, 1.0};
        const SpeculativeTokenDecision d1 = verifier.verify_token<double>(0, p, q, rng);
        REQUIRE(d1.ok);
        CHECK(d1.accepted);
        CHECK(d1.token == 0);
        CHECK(std::isfinite(d1.acceptance_prob));

        // p(0) == 0, q(0) == 0 -> rejects safely without 0/0 NaN and falls back to p (emits 1).
        const std::vector<double> p2{0.0, 1.0};
        const std::vector<double> q2{0.0, 1.0};
        const SpeculativeTokenDecision d2 = verifier.verify_token<double>(0, p2, q2, rng);
        REQUIRE(d2.ok);
        CHECK_FALSE(d2.accepted);
        CHECK(d2.token == 1);
        CHECK(d2.used_residual_fallback);
        CHECK_FALSE(d2.used_residual_resample);
        CHECK(verifier.metrics().residual_fallbacks == 1u);
    }

    // 4. Tiny residual mass Z <= eps triggers safe direct-from-p fallback.
    {
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection, /*residual_min_mass=*/1e-3});
        std::mt19937_64 rng(0xE3E3E3ULL);
        const std::vector<double> p{0.4995, 0.5005};
        const std::vector<double> q{0.5000, 0.5000};
        std::uint64_t rejections = 0;
        for (int t = 0; t < 20000; ++t) {
            const SpeculativeTokenDecision d = verifier.verify_token<double>(0, p, q, rng);
            REQUIRE(d.ok);
            if (!d.accepted) {
                ++rejections;
                CHECK(d.used_residual_fallback);
                CHECK_FALSE(d.used_residual_resample);
                CHECK((d.token == 0 || d.token == 1));
            }
        }
        CHECK(rejections > 0u);
        CHECK(verifier.metrics().residual_fallbacks == rejections);
        CHECK(verifier.metrics().residual_resamples == 0u);
    }

    // 5. Zero-draft sequence K == 0 samples 1 bonus token cleanly from p_0.
    {
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xE4E4E4ULL);
        const std::vector<std::int32_t> empty_draft;
        const std::vector<std::vector<double>> target_dists{{0.0, 0.25, 0.75}};
        const std::vector<std::vector<double>> empty_props;

        std::vector<std::uint64_t> counts(3, 0);
        for (int t = 0; t < 10000; ++t) {
            const SpeculativeSequenceResult r = verifier.verify_sequence<double>(empty_draft, target_dists, empty_props, rng);
            REQUIRE(r.ok);
            CHECK(r.draft_tokens == 0u);
            CHECK(r.accepted_draft_tokens == 0u);
            CHECK_FALSE(r.rejected);
            CHECK(r.sampled_bonus);
            CHECK(r.effective_mode == SpeculativeVerificationMode::TargetMatch);
            REQUIRE(r.emitted_tokens.size() == 1u);
            ++counts[static_cast<std::size_t>(r.emitted_tokens[0])];
        }
        CHECK(counts[0] == 0u);
        CHECK(verifier.metrics().accept_rate == 0.0);
        CHECK(verifier.metrics().bonus_samples == 10000u);
        CHECK(EmpiricalTvDistance(counts, target_dists[0], 10000) < 0.015);
    }

    // 6. Lazy verification causal on_commit order and early stop on rejection.
    {
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xE5E5E5ULL);
        const std::vector<std::int32_t> drafts{0, 1, 0, 1};
        std::vector<std::size_t> queried_target_positions;
        std::vector<std::pair<std::size_t, std::int32_t>> commits;

        // Position 0: p = q = {1, 0} -> deterministic accept token 0.
        // Position 1: p = {1, 0}, q = {0, 1} -> deterministic reject token 1, emit 0.
        // Positions 2, 3, 4 MUST NOT be queried!
        const SpeculativeSequenceResult res = verifier.verify_sequence_lazy(
            drafts, /*has_proposal_dists=*/true,
            [&](std::size_t pos, std::vector<double>& out_p) -> bool {
                // Commit from previous position must have happened BEFORE querying next target dist!
                CHECK(commits.size() == pos);
                queried_target_positions.push_back(pos);
                out_p = {1.0, 0.0};
                return true;
            },
            [&](std::size_t pos, std::vector<double>& out_q) -> bool {
                out_q = (pos == 0) ? std::vector<double>{1.0, 0.0} : std::vector<double>{0.0, 1.0};
                return true;
            },
            [&](std::size_t pos, std::int32_t tok) { commits.emplace_back(pos, tok); }, rng);

        REQUIRE(res.ok);
        CHECK(res.accepted_draft_tokens == 1u);
        CHECK(res.rejected);
        CHECK(res.used_residual_resample);
        CHECK_FALSE(res.sampled_bonus);
        CHECK(res.emitted_tokens == std::vector<std::int32_t>{0, 0});
        CHECK(queried_target_positions == std::vector<std::size_t>{0u, 1u});
        REQUIRE(commits.size() == 2u);
        CHECK(commits[0] == std::pair<std::size_t, std::int32_t>{0u, 0});
        CHECK(commits[1] == std::pair<std::size_t, std::int32_t>{1u, 0});
    }

    // 7. Fail-closed invalid inputs.
    {
        SpeculativeVerifier verifier({SpeculativeVerificationMode::StochasticRejection});
        std::mt19937_64 rng(0xE6E6E6ULL);
        const std::vector<double> valid{0.5, 0.5};
        const std::vector<double> zeros{0.0, 0.0};
        const std::vector<double> has_inf{inf_v, 1.0};
        const std::vector<double> diff_len{0.25, 0.25, 0.5};

        CHECK_FALSE(verifier.verify_token<double>(-1, valid, valid, rng).ok);
        CHECK_FALSE(verifier.verify_token<double>(2, valid, valid, rng).ok);
        CHECK_FALSE(verifier.verify_token<double>(0, zeros, valid, rng).ok);
        CHECK_FALSE(verifier.verify_token<double>(0, valid, zeros, rng).ok);
        CHECK_FALSE(verifier.verify_token<double>(0, has_inf, valid, rng).ok);
        CHECK_FALSE(verifier.verify_token<double>(0, valid, diff_len, rng).ok);

        const std::vector<std::int32_t> drafts{0, 1};
        const std::vector<std::vector<double>> two_targets{valid, valid};  // needs K+1 = 3!
        const std::vector<std::vector<double>> three_targets{valid, valid, valid};
        const std::vector<std::vector<double>> one_prop{valid};  // non-empty but != K!
        CHECK_FALSE(verifier.verify_sequence<double>(drafts, two_targets, {}, rng).ok);
        CHECK_FALSE(verifier.verify_sequence<double>(drafts, three_targets, one_prop, rng).ok);
    }
}
