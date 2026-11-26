// Direct llama.cpp backend tests for opt-in probability-aware speculative
// verification policy (`SpeculativeVerificationMode`) and verifier integration.
// Compiled under `SONDER_WITH_LLAMA_CPP=ON` (`sonder_llamacpp_tests`).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/speculative_verifier.hpp"

using namespace sonder::backends::llamacpp;
namespace si = sonder::inference;

TEST_CASE("llamacpp spec verifier: options default to TargetMatch, accept StochasticRejection, and reject invalid enum") {
    // 1. Public adapter options (LlamaCppBackendOptions).
    si::LlamaCppBackendOptions bo;
    CHECK(bo.spec_verification_mode == si::SpeculativeVerificationMode::TargetMatch);
    CHECK(si::validate_llamacpp_options(bo).ok());

    bo.spec_type = "suffix";
    bo.spec_verification_mode = si::SpeculativeVerificationMode::StochasticRejection;
    CHECK(si::validate_llamacpp_options(bo).ok());

    bo.spec_verification_mode = static_cast<si::SpeculativeVerificationMode>(77);
    const si::Status bad_bo = si::validate_llamacpp_options(bo);
    CHECK(bad_bo.code() == si::ErrorCode::invalid_argument);
    CHECK(bad_bo.message().find("spec_verification_mode") != std::string::npos);

    // 2. Wrapper LoadOptions validation & string round-trip.
    LoadOptions lo;
    CHECK(lo.spec_verification_mode == SpeculativeVerificationMode::TargetMatch);
    CHECK(ValidateContextOptions(lo).ok());

    lo.spec_verification_mode = SpeculativeVerificationMode::StochasticRejection;
    CHECK(ValidateContextOptions(lo).ok());

    lo.spec_verification_mode = static_cast<SpeculativeVerificationMode>(123);
    const Status bad_lo = ValidateContextOptions(lo);
    CHECK(bad_lo.code == ErrorCode::kInvalidArgument);
    CHECK(bad_lo.message.find("spec_verification_mode") != std::string::npos);

    CHECK(ToString(SpeculativeVerificationMode::TargetMatch) == "target_match");
    CHECK(ToString(SpeculativeVerificationMode::StochasticRejection) == "stochastic_rejection");

    SpeculativeVerificationMode parsed = SpeculativeVerificationMode::TargetMatch;
    CHECK(ParseSpeculativeVerificationMode("stochastic-rejection", parsed));
    CHECK(parsed == SpeculativeVerificationMode::StochasticRejection);
    CHECK(ParseSpeculativeVerificationMode("target_match", parsed));
    CHECK(parsed == SpeculativeVerificationMode::TargetMatch);
    CHECK_FALSE(ParseSpeculativeVerificationMode("bogus", parsed));
    CHECK(parsed == SpeculativeVerificationMode::TargetMatch);
}

TEST_CASE("llamacpp spec verifier: greedy MTP and deterministic suffix/ngram stay on TargetMatch") {
    const SpeculativeType all_types[] = {
        SpeculativeType::kNone,    SpeculativeType::kAuto,   SpeculativeType::kNgram,
        SpeculativeType::kNgramMod, SpeculativeType::kSuffix, SpeculativeType::kDraftMtp,
    };

    for (const SpeculativeType type : all_types) {
        CHECK_FALSE(ProposerSuppliesStochasticProbabilities(type));

        // Even when caller opts into StochasticRejection, deterministic suffix/ngram
        // and greedy MTP stay cleanly on TargetMatch until a stochastic proposer supplies q.
        CHECK(ResolveSpeculativeVerificationMode(SpeculativeVerificationMode::StochasticRejection, type,
                                                 /*proposer_has_probabilities=*/false) ==
              SpeculativeVerificationMode::TargetMatch);

        // When a stochastic proposer DOES supply real proposal probabilities, opt-in activates!
        CHECK(ResolveSpeculativeVerificationMode(SpeculativeVerificationMode::StochasticRejection, type,
                                                 /*proposer_has_probabilities=*/true) ==
              SpeculativeVerificationMode::StochasticRejection);

        // Default TargetMatch stays on TargetMatch regardless of proposal probabilities.
        CHECK(ResolveSpeculativeVerificationMode(SpeculativeVerificationMode::TargetMatch, type,
                                                 /*proposer_has_probabilities=*/true) ==
              SpeculativeVerificationMode::TargetMatch);
    }

    // Verify unloaded backend defaults to TargetMatch.
    LlamaCppBackend backend;
    CHECK(backend.RequestedSpecVerificationMode() == SpeculativeVerificationMode::TargetMatch);
    CHECK(backend.EffectiveSpecVerificationMode() == SpeculativeVerificationMode::TargetMatch);

    // Vocab-only load with opt-in StochasticRejection records requested mode while
    // keeping effective mode on TargetMatch (no stochastic proposer loaded).
    LoadOptions vocab_opts;
    vocab_opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    vocab_opts.vocab_only = true;
    vocab_opts.spec_type = "suffix";
    vocab_opts.spec_verification_mode = SpeculativeVerificationMode::StochasticRejection;
    REQUIRE(backend.Load(vocab_opts).ok());
    CHECK(backend.RequestedSpecVerificationMode() == SpeculativeVerificationMode::StochasticRejection);
    CHECK(backend.EffectiveSpecVerificationMode() == SpeculativeVerificationMode::TargetMatch);
    backend.Unload();
    CHECK(backend.RequestedSpecVerificationMode() == SpeculativeVerificationMode::TargetMatch);
    CHECK(backend.EffectiveSpecVerificationMode() == SpeculativeVerificationMode::TargetMatch);
}

TEST_CASE("llamacpp spec verifier: GenerateResult metrics and softmax parity with ComputeTokenLogprobs") {
    GenerateResult r;
    CHECK(r.spec_verifier_mode == SpeculativeVerificationMode::TargetMatch);
    CHECK(r.spec_residual_resamples == 0u);
    CHECK(r.spec_verify_overhead_us == 0.0);
    CHECK(r.spec_accept_rate() == 0.0);

    r.draft_tokens = 20;
    r.draft_accepted_tokens = 15;
    CHECK(r.spec_accept_rate() == doctest::Approx(0.75).epsilon(1e-12));

    // Clamps malformed accepted > drafted or negative counts safely.
    r.draft_accepted_tokens = 99;
    CHECK(r.spec_accept_rate() == doctest::Approx(1.0).epsilon(1e-12));
    r.draft_accepted_tokens = -4;
    CHECK(r.spec_accept_rate() == 0.0);
    r.draft_tokens = -1;
    CHECK(r.spec_accept_rate() == 0.0);

    // Parity between llamacpp's ComputeTokenLogprobs and SpeculativeVerifier's
    // softmax_logits_to_probabilities, plus end-to-end exact rejection verification.
    const std::vector<float> target_logits{1.5F, 2.5F, -std::numeric_limits<float>::infinity(), 0.5F};
    const std::vector<float> draft_logits{2.0F, 1.0F, -std::numeric_limits<float>::infinity(), 1.5F};
    std::vector<double> p;
    std::vector<double> q;
    REQUIRE(si::softmax_logits_to_probabilities(target_logits, p));
    REQUIRE(si::softmax_logits_to_probabilities(draft_logits, q));
    CHECK(p[2] == 0.0);
    CHECK(q[2] == 0.0);

    for (Token tok : {0, 1, 3}) {
        const TokenLogprobInfo lp = ComputeTokenLogprobs(target_logits, tok, /*top_k=*/0);
        REQUIRE(lp.valid);
        CHECK(std::exp(static_cast<double>(lp.logprob)) == doctest::Approx(p[static_cast<std::size_t>(tok)]).epsilon(1e-5));
    }

    si::SpeculativeVerifier verifier({si::SpeculativeVerificationMode::StochasticRejection});
    std::mt19937_64 draft_rng(0x77770001ULL);
    std::mt19937_64 verify_rng(0x77770002ULL);
    std::vector<std::uint64_t> counts(4, 0);
    constexpr std::uint64_t kTrials = 50000;
    for (std::uint64_t i = 0; i < kTrials; ++i) {
        const Token draft_tok = si::sample_categorical(q, draft_rng);
        const si::SpeculativeTokenDecision dec = verifier.verify_token<double>(draft_tok, p, q, verify_rng);
        REQUIRE(dec.ok);
        ++counts[static_cast<std::size_t>(dec.token)];
    }

    CHECK(counts[2] == 0u);  // grammar-masked -inf token NEVER emitted
    const double expected_accept = si::expected_stochastic_rejection_acceptance<double>(p, q);
    CHECK(verifier.metrics().accept_rate == doctest::Approx(expected_accept).epsilon(0.01));
    CHECK(verifier.metrics().residual_resamples > 0u);
    for (std::size_t v = 0; v < p.size(); ++v) {
        const double emp = static_cast<double>(counts[v]) / static_cast<double>(kTrials);
        CHECK(emp == doctest::Approx(p[v]).epsilon(0.01));
    }
}
