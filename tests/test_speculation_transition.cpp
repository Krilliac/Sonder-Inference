// Header-level unit tests for `speculation_transition.hpp` (compiled in every
// CI build regardless of SONDER_WITH_LLAMA_CPP):
//   - Post-batch single-request speculation reactivation state machine
//   - Sustained low-concurrency cooldown and pending/exclusive reset semantics
//   - Explicit MTP draft-context & pending-feature readiness checks + hybrid fallback
//   - Remaining max-token & context-window cost guard
//   - Cancellation & hybrid/recurrent model rollback guards
//   - Bounded small-batch concurrent linear draft coordinator, cost-regression gate,
//     and isolated per-sequence KV rollback ranges
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backends/llamacpp/speculation_transition.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;
using sonder::inference::ModelArchitecture;

TEST_CASE("speculation transition: single to multi to drain cooldown and safe history reactivation") {
    SpeculationTransitionState st;
    SlotSpecCapability cap;
    cap.configured_mode = SlotSpecMode::kSuffix;
    cap.configured_draft_max = 16;
    cap.architecture = ModelArchitecture::attention_only;
    st.Configure(cap);

    CHECK(st.active_mode() == SlotSpecMode::kSuffix);
    CHECK(st.configured_mode() == SlotSpecMode::kSuffix);
    CHECK(st.active_draft_max() == 16);
    CHECK_FALSE(st.suspended_for_batch());

    // Enter multi-request batch -> temporarily suspended, configured capability preserved.
    st.SuspendForBatch(SpecSuspendReason::kMultiRequestBatch);
    CHECK(st.suspended_for_batch());
    CHECK(st.active_mode() == SlotSpecMode::kNone);
    CHECK(st.active_draft_max() == 0);
    CHECK(st.configured_mode() == SlotSpecMode::kSuffix);
    CHECK(st.suspend_count() == 1u);
    CHECK(st.low_concurrency_steps() == 0u);

    SpeculationReactivationPolicy policy;
    policy.cooldown_steps = 4;
    policy.min_remaining_tokens_history = 16;

    SpeculationReactivationInput in;
    in.active_count = 1;
    in.pending_count = 0;
    in.pending_exclusive = false;
    in.cancelled = false;
    in.in_decode_phase = true;
    in.slot_valid = true;
    in.remaining_max_tokens = 128;
    in.remaining_ctx_tokens = 1024;
    in.target_pos = 80;
    in.committed_history_len = 80;
    in.history_matches_target_pos = true;

    // Steps 1..3 remain in cooldown.
    for (std::uint32_t step = 1; step < policy.cooldown_steps; ++step) {
        const SpeculationTransitionDecision d = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(d.should_reactivate);
        CHECK(d.block_reason == SpecSuspendReason::kCooldownActive);
        CHECK(st.low_concurrency_steps() == step);
        CHECK(st.suspended_for_batch());
        ++in.target_pos;
        ++in.committed_history_len;
        --in.remaining_max_tokens;
    }

    // Stale/out-of-sync committed history blocks reactivation even once cooldown expires.
    {
        SpeculationReactivationInput stale = in;
        stale.history_matches_target_pos = false;
        const SpeculationTransitionDecision d = st.EvaluateSingleRequestStep(stale, policy);
        CHECK_FALSE(d.should_reactivate);
        CHECK(d.block_reason == SpecSuspendReason::kHistoryOutOfSync);
        CHECK(st.suspended_for_batch());
    }

    // Exact committed history at step >= cooldown_steps allows safe reactivation with rebuild flag.
    const SpeculationTransitionDecision ready = st.EvaluateSingleRequestStep(in, policy);
    REQUIRE(ready.should_reactivate);
    CHECK(ready.target_mode == SlotSpecMode::kSuffix);
    CHECK(ready.effective_draft_max == 16);
    CHECK(ready.requires_history_rebuild);
    CHECK_FALSE(ready.fell_back_from_mtp);
    CHECK(ready.block_reason == SpecSuspendReason::kNone);
    // Two-phase contract: state stays suspended until caller actually rebuilds and commits!
    CHECK(st.suspended_for_batch());

    st.CommitReactivation(ready);
    CHECK_FALSE(st.suspended_for_batch());
    CHECK(st.active_mode() == SlotSpecMode::kSuffix);
    CHECK(st.active_draft_max() == 16);
    CHECK(st.history_synced());
    CHECK(st.reactivation_count() == 1u);
}

TEST_CASE("speculation transition: pending or exclusive request during cooldown resets reactivation timer") {
    SpeculationTransitionState st;
    SlotSpecCapability cap;
    cap.configured_mode = SlotSpecMode::kNgramMod;
    cap.configured_draft_max = 32;
    cap.architecture = ModelArchitecture::attention_only;
    st.Configure(cap);
    st.SuspendForBatch();

    SpeculationReactivationPolicy policy;
    policy.cooldown_steps = 4;
    policy.min_remaining_tokens_history = 16;

    SpeculationReactivationInput in;
    in.active_count = 1;
    in.pending_count = 0;
    in.pending_exclusive = false;
    in.in_decode_phase = true;
    in.slot_valid = true;
    in.remaining_max_tokens = 200;
    in.remaining_ctx_tokens = 1000;
    in.target_pos = 64;
    in.committed_history_len = 64;
    in.history_matches_target_pos = true;

    // Accumulate 3 of 4 cooldown steps.
    for (int i = 0; i < 3; ++i) {
        CHECK_FALSE(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
    }
    CHECK(st.low_concurrency_steps() == 3u);

    // A pending normal request arrives -> resets timer to 0 immediately.
    in.pending_count = 1;
    const SpeculationTransitionDecision d_pend = st.EvaluateSingleRequestStep(in, policy);
    CHECK_FALSE(d_pend.should_reactivate);
    CHECK(d_pend.block_reason == SpecSuspendReason::kPendingRequest);
    CHECK(st.low_concurrency_steps() == 0u);

    // Clear pending and accumulate 3 steps again.
    in.pending_count = 0;
    for (int i = 0; i < 3; ++i) {
        CHECK_FALSE(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
    }
    CHECK(st.low_concurrency_steps() == 3u);

    // An exclusive request arrives -> resets timer to 0 immediately.
    in.pending_exclusive = true;
    const SpeculationTransitionDecision d_excl = st.EvaluateSingleRequestStep(in, policy);
    CHECK_FALSE(d_excl.should_reactivate);
    CHECK(d_excl.block_reason == SpecSuspendReason::kExclusivePending);
    CHECK(st.low_concurrency_steps() == 0u);

    // Must now complete a full fresh 4-step sustained window before reactivating.
    in.pending_exclusive = false;
    for (int i = 0; i < 3; ++i) {
        const SpeculationTransitionDecision d = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(d.should_reactivate);
        CHECK(d.block_reason == SpecSuspendReason::kCooldownActive);
    }
    const SpeculationTransitionDecision d_ok = st.EvaluateSingleRequestStep(in, policy);
    CHECK(d_ok.should_reactivate);
    CHECK(d_ok.target_mode == SlotSpecMode::kNgramMod);
    CHECK(d_ok.requires_history_rebuild);
}

TEST_CASE("speculation transition: MTP readiness blocks drifted state and allows safe hybrid suffix fallback") {
    // 1. Pure MTP (no suffix fallback): strictly requires draft_pos and pending_features_pos == target_pos.
    {
        SpeculationTransitionState st;
        SlotSpecCapability cap;
        cap.configured_mode = SlotSpecMode::kDraftMtp;
        cap.hybrid_suffix_configured = false;
        cap.suffix_fallback_available = false;
        cap.configured_draft_max = 3;
        cap.architecture = ModelArchitecture::attention_only;
        st.Configure(cap);
        st.SuspendForBatch();

        SpeculationReactivationPolicy policy;
        policy.cooldown_steps = 1;
        policy.min_remaining_tokens_mtp = 8;

        SpeculationReactivationInput in;
        in.active_count = 1;
        in.in_decode_phase = true;
        in.slot_valid = true;
        in.remaining_max_tokens = 64;
        in.remaining_ctx_tokens = 512;
        in.target_pos = 120;
        in.committed_history_len = 120;
        in.history_matches_target_pos = true;

        // Flipping slot_mtp_synced=true alone is NOT sufficient when draft_pos drifted!
        in.mtp_readiness.draft_ctx_available = true;
        in.mtp_readiness.slot_valid = true;
        in.mtp_readiness.slot_mtp_synced = true;
        in.mtp_readiness.mem_shared = false;
        in.mtp_readiness.target_pos = 120;
        in.mtp_readiness.draft_pos = 115;  // drifted behind target_pos!
        in.mtp_readiness.pending_features_valid = true;
        in.mtp_readiness.pending_features_pos = 120;
        in.mtp_readiness.pending_features_size_ok = true;
        CHECK_FALSE(IsMtpDraftStateReady(in.mtp_readiness));
        const SpeculationTransitionDecision d_drift_pos = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(d_drift_pos.should_reactivate);
        CHECK(d_drift_pos.block_reason == SpecSuspendReason::kMtpStateDrifted);

        // Matching draft_pos but stale pending_features_pos is STILL blocked!
        in.mtp_readiness.draft_pos = 120;
        in.mtp_readiness.pending_features_pos = 119;
        CHECK_FALSE(IsMtpDraftStateReady(in.mtp_readiness));
        const SpeculationTransitionDecision d_drift_h = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(d_drift_h.should_reactivate);
        CHECK(d_drift_h.block_reason == SpecSuspendReason::kMtpStateDrifted);

        // Non-contiguous deferred catch-up rows are blocked.
        in.mtp_readiness.draft_pos = 116;
        in.mtp_readiness.has_deferred = true;
        in.mtp_readiness.deferred_first_pos = 117;  // gap at 116!
        in.mtp_readiness.deferred_last_pos = 119;
        in.mtp_readiness.deferred_count = 3;
        in.mtp_readiness.pending_features_pos = 120;
        CHECK_FALSE(IsMtpDraftStateReady(in.mtp_readiness));

        // Contiguous deferred catch-up bridging draft_pos..target_pos + fresh pending features IS ready!
        in.mtp_readiness.deferred_first_pos = 116;
        in.mtp_readiness.deferred_last_pos = 119;
        in.mtp_readiness.deferred_count = 4;
        CHECK(IsMtpDraftStateReady(in.mtp_readiness));
        const SpeculationTransitionDecision d_def_ok = st.EvaluateSingleRequestStep(in, policy);
        CHECK(d_def_ok.should_reactivate);
        CHECK(d_def_ok.target_mode == SlotSpecMode::kDraftMtp);
        CHECK(d_def_ok.effective_draft_max == 3);
        CHECK_FALSE(d_def_ok.fell_back_from_mtp);

        // Shared-memory MTP still checks pending_features_pos == target_pos.
        MtpReadinessState shared_st = in.mtp_readiness;
        shared_st.mem_shared = true;
        shared_st.has_deferred = false;
        shared_st.pending_features_pos = 118;
        CHECK_FALSE(IsMtpDraftStateReady(shared_st));
        shared_st.pending_features_pos = 120;
        CHECK(IsMtpDraftStateReady(shared_st));
    }

    // 2. Hybrid draft-mtp + suffix: when MTP drifted during batching, safely falls back to suffix!
    {
        SpeculationTransitionState st;
        SlotSpecCapability cap;
        cap.configured_mode = SlotSpecMode::kDraftMtp;
        cap.hybrid_suffix_configured = true;
        cap.suffix_fallback_available = true;
        cap.configured_draft_max = 3;
        cap.fallback_suffix_draft_max = 12;
        cap.architecture = ModelArchitecture::attention_only;
        st.Configure(cap);
        st.SuspendForBatch();

        SpeculationReactivationPolicy policy;
        policy.cooldown_steps = 2;
        policy.min_remaining_tokens_history = 16;
        policy.allow_mtp_to_suffix_fallback = true;

        SpeculationReactivationInput in;
        in.active_count = 1;
        in.in_decode_phase = true;
        in.slot_valid = true;
        in.remaining_max_tokens = 100;
        in.remaining_ctx_tokens = 500;
        in.target_pos = 150;
        in.committed_history_len = 150;
        in.history_matches_target_pos = true;
        in.mtp_readiness.draft_ctx_available = true;
        in.mtp_readiness.slot_valid = true;
        in.mtp_readiness.slot_mtp_synced = false;  // drifted in Case B
        in.mtp_readiness.target_pos = 150;
        in.mtp_readiness.draft_pos = 100;

        CHECK_FALSE(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
        const SpeculationTransitionDecision fb = st.EvaluateSingleRequestStep(in, policy);
        REQUIRE(fb.should_reactivate);
        CHECK(fb.target_mode == SlotSpecMode::kSuffix);
        CHECK(fb.effective_draft_max == 12);
        CHECK(fb.requires_history_rebuild);
        CHECK(fb.fell_back_from_mtp);
        CHECK_FALSE(fb.enable_hybrid_suffix);

        // If fallback policy is disabled, keeps plain decode cleanly.
        policy.allow_mtp_to_suffix_fallback = false;
        const SpeculationTransitionDecision no_fb = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(no_fb.should_reactivate);
        CHECK(no_fb.block_reason == SpecSuspendReason::kMtpStateDrifted);
    }
}

TEST_CASE("speculation transition: remaining-budget cost guard keeps plain decode near completion") {
    SpeculationTransitionState st;
    SlotSpecCapability cap;
    cap.configured_mode = SlotSpecMode::kSuffix;
    cap.configured_draft_max = 16;
    cap.architecture = ModelArchitecture::attention_only;
    st.Configure(cap);
    st.SuspendForBatch();

    SpeculationReactivationPolicy policy;
    policy.cooldown_steps = 1;
    policy.min_remaining_tokens_history = 16;
    policy.extra_remaining_per_4k_history = 2;

    SpeculationReactivationInput in;
    in.active_count = 1;
    in.in_decode_phase = true;
    in.slot_valid = true;
    in.target_pos = 512;
    in.committed_history_len = 512;
    in.history_matches_target_pos = true;

    // Few max_new_tokens remaining -> stay on plain decode.
    in.remaining_max_tokens = 6;
    in.remaining_ctx_tokens = 2048;
    const SpeculationTransitionDecision d_low_max = st.EvaluateSingleRequestStep(in, policy);
    CHECK_FALSE(d_low_max.should_reactivate);
    CHECK(d_low_max.block_reason == SpecSuspendReason::kBudgetTooSmall);

    // Ample max_new_tokens, but near context limit -> stay on plain decode.
    in.remaining_max_tokens = 500;
    in.remaining_ctx_tokens = 10;
    const SpeculationTransitionDecision d_low_ctx = st.EvaluateSingleRequestStep(in, policy);
    CHECK_FALSE(d_low_ctx.should_reactivate);
    CHECK(d_low_ctx.block_reason == SpecSuspendReason::kBudgetTooSmall);

    // Large committed history (32K tokens) increases required remaining tokens (16 + 8*2 = 32).
    in.target_pos = 32768;
    in.committed_history_len = 32768;
    in.remaining_ctx_tokens = 4096;
    in.remaining_max_tokens = 24;
    CHECK(RequiredRemainingTokensForRecovery(SlotSpecMode::kSuffix, 32768, policy) == 32);
    CHECK(st.EvaluateSingleRequestStep(in, policy).block_reason == SpecSuspendReason::kBudgetTooSmall);

    in.remaining_max_tokens = 32;
    CHECK(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
}

TEST_CASE("speculation transition: cancellation, invalidation, and hybrid or recurrent rollback guards") {
    SpeculationReactivationPolicy policy;
    policy.cooldown_steps = 2;

    // 1. Pure recurrent model: NEVER activates or reactivates.
    {
        SpeculationTransitionState st;
        SlotSpecCapability cap;
        cap.configured_mode = SlotSpecMode::kSuffix;
        cap.configured_draft_max = 16;
        cap.architecture = ModelArchitecture::recurrent;
        st.Configure(cap);
        CHECK(st.active_mode() == SlotSpecMode::kNone);
        CHECK(st.permanently_disabled());

        SpeculationReactivationInput in;
        in.active_count = 1;
        in.in_decode_phase = true;
        in.slot_valid = true;
        in.remaining_max_tokens = 200;
        in.remaining_ctx_tokens = 2000;
        in.target_pos = 100;
        in.committed_history_len = 100;
        in.history_matches_target_pos = true;
        CHECK_FALSE(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
    }

    // 2. Hybrid model without rs_cap and without partial checkpoints: blocked.
    {
        const ArchitectureSpecGuardResult g_mtp =
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kDraftMtp, 3, 0, false);
        CHECK_FALSE(g_mtp.allowed);
        const ArchitectureSpecGuardResult g_suf =
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kSuffix, 16, 0, false);
        CHECK_FALSE(g_suf.allowed);

        // With partial checkpoints: suffix allowed, MTP still blocked without rs_cap.
        const ArchitectureSpecGuardResult g_suf_ckpt =
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kSuffix, 16, 0, true, true);
        CHECK(g_suf_ckpt.allowed);
        CHECK(g_suf_ckpt.requires_partial_checkpoint);
        CHECK_FALSE(
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kDraftMtp, 3, 0, true, true).allowed);

        // With rs_cap = 2: MTP capped to 2; suffix without partial checkpoints capped to 2.
        const ArchitectureSpecGuardResult g_mtp_rs =
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kDraftMtp, 6, 2, true);
        CHECK(g_mtp_rs.allowed);
        CHECK(g_mtp_rs.effective_draft_max == 2);
        const ArchitectureSpecGuardResult g_suf_no_ckpt =
            CheckArchitectureSpecGuard(ModelArchitecture::hybrid, SlotSpecMode::kSuffix, 16, 2, true, false);
        CHECK(g_suf_no_ckpt.allowed);
        CHECK(g_suf_no_ckpt.effective_draft_max == 2);
    }

    // 3. Cancellation resets cooldown and blocks reactivation; slot invalidation disables permanently.
    {
        SpeculationTransitionState st;
        SlotSpecCapability cap;
        cap.configured_mode = SlotSpecMode::kNgramShort;
        cap.configured_draft_max = 8;
        cap.architecture = ModelArchitecture::attention_only;
        st.Configure(cap);
        st.SuspendForBatch();

        SpeculationReactivationInput in;
        in.active_count = 1;
        in.in_decode_phase = true;
        in.slot_valid = true;
        in.remaining_max_tokens = 128;
        in.remaining_ctx_tokens = 512;
        in.target_pos = 50;
        in.committed_history_len = 50;
        in.history_matches_target_pos = true;

        CHECK_FALSE(st.EvaluateSingleRequestStep(in, policy).should_reactivate);
        CHECK(st.low_concurrency_steps() == 1u);

        in.cancelled = true;
        const SpeculationTransitionDecision d_cancel = st.EvaluateSingleRequestStep(in, policy);
        CHECK_FALSE(d_cancel.should_reactivate);
        CHECK(d_cancel.block_reason == SpecSuspendReason::kCancelled);
        CHECK(st.low_concurrency_steps() == 0u);

        in.cancelled = false;
        st.MarkSlotInvalidated();
        CHECK(st.permanently_disabled());
        CHECK(st.EvaluateSingleRequestStep(in, policy).block_reason == SpecSuspendReason::kSlotInvalidated);
    }
}

TEST_CASE("speculation transition: small-batch concurrent coordinator budget allocation and fallbacks") {
    ConcurrentSpecPolicy policy;
    policy.enabled = true;
    policy.max_concurrent_slots = 4;
    policy.max_aggregate_draft_tokens = 7;
    policy.max_per_slot_draft_tokens = 6;
    std::size_t rr = 0;

    // 1. Disabled by opt-in flag -> immediate fallback.
    {
        ConcurrentSpecPolicy off = policy;
        off.enabled = false;
        const std::vector<ConcurrentSpecSlotInput> slots{
            {0, 0, 50, false, false, true, 6, 4},
            {1, 1, 60, false, false, true, 6, 4},
        };
        const ConcurrentSpecBatchPlan p = PlanConcurrentLinearSpeculation(
            slots, off, 256, 64, 16, ModelArchitecture::attention_only, 0, false, rr);
        CHECK_FALSE(p.use_concurrent_spec);
        CHECK(p.fallback_reason == ConcurrentSpecFallbackReason::kDisabled);
    }

    // 2. Active prefill slot triggers immediate fallback to ordinary continuous batching.
    {
        const std::vector<ConcurrentSpecSlotInput> mixed{
            {0, 0, 50, false, false, true, 6, 4},
            {1, 1, 0, true, false, false, 0, 0},  // active prefill!
            {2, 2, 60, false, false, true, 6, 4},
        };
        const ConcurrentSpecBatchPlan p = PlanConcurrentLinearSpeculation(
            mixed, policy, 256, 64, 16, ModelArchitecture::attention_only, 0, false, rr);
        CHECK_FALSE(p.use_concurrent_spec);
        CHECK(p.fallback_reason == ConcurrentSpecFallbackReason::kPrefillActive);
    }

    // 3. Exceeding small-batch threshold (5 > 4) falls back immediately.
    {
        std::vector<ConcurrentSpecSlotInput> five;
        for (std::size_t i = 0; i < 5; ++i) {
            five.push_back({i, static_cast<std::int32_t>(i), 40, false, false, true, 6, 4});
        }
        const ConcurrentSpecBatchPlan p = PlanConcurrentLinearSpeculation(
            five, policy, 256, 64, 16, ModelArchitecture::attention_only, 0, false, rr);
        CHECK_FALSE(p.use_concurrent_spec);
        CHECK(p.fallback_reason == ConcurrentSpecFallbackReason::kBatchExceedsThreshold);
    }

    // 4. Hybrid without rs_cap falls back; hybrid with rs_cap=2 caps each slot to <= 2.
    {
        const std::vector<ConcurrentSpecSlotInput> two{
            {0, 0, 40, false, false, true, 6, 5},
            {1, 1, 50, false, false, true, 6, 5},
        };
        const ConcurrentSpecBatchPlan p_no_rs = PlanConcurrentLinearSpeculation(
            two, policy, 256, 64, 16, ModelArchitecture::hybrid, 0, false, rr);
        CHECK_FALSE(p_no_rs.use_concurrent_spec);
        CHECK(p_no_rs.fallback_reason == ConcurrentSpecFallbackReason::kArchitectureUnsupported);

        const ConcurrentSpecBatchPlan p_rs2 = PlanConcurrentLinearSpeculation(
            two, policy, 256, 64, 16, ModelArchitecture::hybrid, 2, false, rr);
        REQUIRE(p_rs2.use_concurrent_spec);
        REQUIRE(p_rs2.slots.size() == 2u);
        CHECK(p_rs2.slots[0].allocated_draft_tokens == 2);
        CHECK(p_rs2.slots[1].allocated_draft_tokens == 2);
        CHECK(p_rs2.total_draft_tokens == 4);
    }

    // 5. Fair water-filling aggregate budget distribution with round-robin remainder rotation
    // and strictly disjoint contiguous batch index spans.
    {
        const std::vector<ConcurrentSpecSlotInput> three{
            {0, 0, 100, false, false, true, 6, 6},
            {1, 1, 200, false, false, true, 6, 6},
            {2, 2, 300, false, false, true, 6, 2},  // only proposes 2
        };
        rr = 0;
        const ConcurrentSpecBatchPlan p1 = PlanConcurrentLinearSpeculation(
            three, policy, 256, 64, 16, ModelArchitecture::attention_only, 0, false, rr);
        REQUIRE(p1.use_concurrent_spec);
        REQUIRE(p1.slots.size() == 3u);
        CHECK(p1.total_base_tokens == 3);
        CHECK(p1.total_draft_tokens == 7);  // exact aggregate cap
        CHECK(p1.total_batch_tokens == 10);
        CHECK(p1.slots[0].allocated_draft_tokens == 3);
        CHECK(p1.slots[1].allocated_draft_tokens == 2);
        CHECK(p1.slots[2].allocated_draft_tokens == 2);
        // Contiguous disjoint batch spans: [0..3], [4..6], [7..9].
        CHECK(p1.slots[0].batch_start_idx == 0);
        CHECK(p1.slots[0].total_slot_tokens == 4);
        CHECK(p1.slots[1].batch_start_idx == 4);
        CHECK(p1.slots[1].total_slot_tokens == 3);
        CHECK(p1.slots[2].batch_start_idx == 7);
        CHECK(p1.slots[2].total_slot_tokens == 3);

        // Next step rotates remainder fairly to slot 1!
        const ConcurrentSpecBatchPlan p2 = PlanConcurrentLinearSpeculation(
            three, policy, 256, 64, 16, ModelArchitecture::attention_only, 0, false, rr);
        REQUIRE(p2.use_concurrent_spec);
        CHECK(p2.slots[0].allocated_draft_tokens == 2);
        CHECK(p2.slots[1].allocated_draft_tokens == 3);
        CHECK(p2.slots[2].allocated_draft_tokens == 2);
    }

    // 6. Measured verify cost regression gate trips pause and recovers after cooldown.
    {
        ConcurrentSpecCostGate gate;
        ConcurrentSpecPolicy cpol = policy;
        cpol.min_cost_samples = 3;
        cpol.verify_regression_ratio = 1.05;
        cpol.regression_cooldown_steps = 4;

        // Plain baseline: 20 ms for 2 tokens = 10 ms/tok.
        for (int i = 0; i < 3; ++i) {
            gate.ObservePlainStep(20.0, 2);
        }
        CHECK_FALSE(gate.IsRegressed());

        // Spec verify is slower per committed token (30 ms for 2 tokens = 15 ms/tok > 10.5 ms/tok).
        CHECK_FALSE(gate.ObserveSpecStep(30.0, 2, 4, 0, cpol));
        CHECK_FALSE(gate.ObserveSpecStep(30.0, 2, 4, 0, cpol));
        CHECK(gate.ObserveSpecStep(30.0, 2, 4, 0, cpol));
        CHECK(gate.IsRegressed());
        CHECK(gate.regression_trips() == 1u);

        const std::vector<ConcurrentSpecSlotInput> two{
            {0, 0, 40, false, false, true, 6, 4},
            {1, 1, 50, false, false, true, 6, 4},
        };
        const ConcurrentSpecBatchPlan blocked = PlanConcurrentLinearSpeculation(
            two, cpol, 256, 64, 16, ModelArchitecture::attention_only, 0, gate.IsRegressed(), rr);
        CHECK_FALSE(blocked.use_concurrent_spec);
        CHECK(blocked.fallback_reason == ConcurrentSpecFallbackReason::kVerifyCostRegressed);

        // Cooldown ticks down on plain steps, allowing a bounded re-probe.
        for (int i = 0; i < 4; ++i) {
            gate.NotePlainStepInCooldown();
        }
        CHECK_FALSE(gate.IsRegressed());
    }
}

TEST_CASE("speculation transition: per-slot linear verification and isolated KV rollback ranges") {
    // Three concurrent slots verified in one step with completely independent outcomes:
    //   Slot 0 (seq 0, base_pos 100): proposes 4 drafts, rejects immediately at j=0
    //   Slot 1 (seq 1, base_pos 200): proposes 4 drafts, accepts 2, rejects at j=2
    //   Slot 2 (seq 2, base_pos 300): proposes 3 drafts, accepts all 3 + bonus token
    const std::vector<SpecToken> prop0{10, 11, 12, 13};
    const std::vector<SpecToken> samp0{99, 11, 12, 13, 14};
    const LinearDraftVerifyOutcome out0 = VerifyLinearDraftSlot(0, 100, prop0, samp0);
    CHECK(out0.accepted_drafts == 0u);
    CHECK(out0.emitted_tokens == std::vector<SpecToken>{99});
    CHECK(out0.rollback.seq_id == 0);
    CHECK(out0.rollback.needs_rollback);
    CHECK(out0.rollback.keep_pos == 101);
    CHECK(out0.rollback.batch_end_pos == 105);
    CHECK(out0.rollback.rm_from_pos == 101);
    CHECK(out0.rollback.rejected_tokens == 4);

    const std::vector<SpecToken> prop1{20, 21, 22, 23};
    const std::vector<SpecToken> samp1{20, 21, 88, 23, 24};
    const LinearDraftVerifyOutcome out1 = VerifyLinearDraftSlot(1, 200, prop1, samp1);
    CHECK(out1.accepted_drafts == 2u);
    CHECK(out1.emitted_tokens == std::vector<SpecToken>{20, 21, 88});
    CHECK(out1.rollback.seq_id == 1);
    CHECK(out1.rollback.needs_rollback);
    CHECK(out1.rollback.keep_pos == 203);
    CHECK(out1.rollback.batch_end_pos == 205);
    CHECK(out1.rollback.rm_from_pos == 203);
    CHECK(out1.rollback.rejected_tokens == 2);

    const std::vector<SpecToken> prop2{30, 31, 32};
    const std::vector<SpecToken> samp2{30, 31, 32, 33};
    const LinearDraftVerifyOutcome out2 = VerifyLinearDraftSlot(2, 300, prop2, samp2);
    CHECK(out2.accepted_drafts == 3u);
    CHECK(out2.emitted_tokens == std::vector<SpecToken>{30, 31, 32, 33});
    CHECK(out2.rollback.seq_id == 2);
    CHECK_FALSE(out2.rollback.needs_rollback);
    CHECK(out2.rollback.keep_pos == 304);
    CHECK(out2.rollback.batch_end_pos == 304);
    CHECK(out2.rollback.rejected_tokens == 0);
}
