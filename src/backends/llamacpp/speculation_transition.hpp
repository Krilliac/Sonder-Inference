// Safe post-batch speculation recovery state machine AND bounded concurrent
// small-batch linear speculation coordinator for the direct llama.cpp backend.
//
// Pure stdlib-only C++20 (no llama.cpp headers) so every state transition,
// MTP readiness guard, architecture rollback guard, cost-regression gate,
// aggregate verification budget allocation, and per-sequence KV rollback range
// can be exhaustively unit-tested without model weights.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "sonder/inference/model_architecture.hpp"

namespace sonder::backends::llamacpp {

using SpecToken = std::int32_t;

// Concrete per-slot speculative decoding mode (after resolving "auto").
enum class SlotSpecMode : std::uint8_t {
    kNone = 0,
    kNgramShort,
    kNgramMod,
    kSuffix,
    kDraftMtp,
};

[[nodiscard]] constexpr std::string_view ToString(SlotSpecMode mode) noexcept {
    switch (mode) {
        case SlotSpecMode::kNone: return "none";
        case SlotSpecMode::kNgramShort: return "ngram";
        case SlotSpecMode::kNgramMod: return "ngram-mod";
        case SlotSpecMode::kSuffix: return "suffix";
        case SlotSpecMode::kDraftMtp: return "draft-mtp";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool IsHistoryBasedSpecMode(SlotSpecMode mode) noexcept {
    return mode == SlotSpecMode::kSuffix || mode == SlotSpecMode::kNgramMod || mode == SlotSpecMode::kNgramShort;
}

// Diagnostic reason why speculation is suspended or blocked from reactivating.
enum class SpecSuspendReason : std::uint8_t {
    kNone = 0,
    kMultiRequestBatch,
    kPendingRequest,
    kExclusivePending,
    kCancelled,
    kArchitectureUnsupported,
    kMtpStateDrifted,
    kBudgetTooSmall,
    kCooldownActive,
    kVerifyCostRegression,
    kSlotInvalidated,
    kHistoryOutOfSync,
};

[[nodiscard]] constexpr std::string_view ToString(SpecSuspendReason reason) noexcept {
    switch (reason) {
        case SpecSuspendReason::kNone: return "none";
        case SpecSuspendReason::kMultiRequestBatch: return "multi_request_batch";
        case SpecSuspendReason::kPendingRequest: return "pending_request";
        case SpecSuspendReason::kExclusivePending: return "exclusive_pending";
        case SpecSuspendReason::kCancelled: return "cancelled";
        case SpecSuspendReason::kArchitectureUnsupported: return "architecture_unsupported";
        case SpecSuspendReason::kMtpStateDrifted: return "mtp_state_drifted";
        case SpecSuspendReason::kBudgetTooSmall: return "budget_too_small";
        case SpecSuspendReason::kCooldownActive: return "cooldown_active";
        case SpecSuspendReason::kVerifyCostRegression: return "verify_cost_regression";
        case SpecSuspendReason::kSlotInvalidated: return "slot_invalidated";
        case SpecSuspendReason::kHistoryOutOfSync: return "history_out_of_sync";
    }
    return "unknown";
}

// Rollback safety check for a given model architecture and speculation mode.
struct ArchitectureSpecGuardResult {
    bool allowed = false;
    std::int32_t effective_draft_max = 0;
    bool requires_partial_checkpoint = false;
};

[[nodiscard]] inline ArchitectureSpecGuardResult CheckArchitectureSpecGuard(
    sonder::inference::ModelArchitecture architecture, SlotSpecMode mode, std::int32_t requested_draft_max,
    std::int32_t hybrid_rs_cap, bool can_hybrid_spec_ckpt, bool allow_partial_checkpoint = true) noexcept {
    ArchitectureSpecGuardResult out;
    if (mode == SlotSpecMode::kNone || requested_draft_max <= 0) {
        return out;
    }
    if (architecture == sonder::inference::ModelArchitecture::recurrent) {
        // Pure recurrent models do not support speculative tail rollback.
        return out;
    }
    if (architecture == sonder::inference::ModelArchitecture::attention_only) {
        out.allowed = true;
        out.effective_draft_max = requested_draft_max;
        out.requires_partial_checkpoint = false;
        return out;
    }

    // Hybrid (attention + recurrent/SSM layers).
    const std::int32_t safe_rs = std::max<std::int32_t>(0, hybrid_rs_cap);
    if (mode == SlotSpecMode::kDraftMtp) {
        if (safe_rs <= 0) {
            return out;
        }
        out.allowed = true;
        out.effective_draft_max = std::min<std::int32_t>(requested_draft_max, safe_rs);
        out.requires_partial_checkpoint = false;
        return out;
    }

    // History-based modes (suffix / ngram / ngram-mod) on hybrid architectures.
    if (safe_rs > 0) {
        out.allowed = true;
        if (can_hybrid_spec_ckpt && allow_partial_checkpoint) {
            out.effective_draft_max = requested_draft_max;
            out.requires_partial_checkpoint = (requested_draft_max > safe_rs);
        } else {
            out.effective_draft_max = std::min<std::int32_t>(requested_draft_max, safe_rs);
            out.requires_partial_checkpoint = false;
        }
        return out;
    }
    if (can_hybrid_spec_ckpt && allow_partial_checkpoint) {
        out.allowed = true;
        out.effective_draft_max = requested_draft_max;
        out.requires_partial_checkpoint = true;
        return out;
    }
    return out;
}

// Explicit draft-context & pending target-feature readiness snapshot required
// before reactivating native `draft-mtp` speculation.
struct MtpReadinessState {
    bool draft_ctx_available = false;       // ctx_dft != nullptr
    bool slot_valid = false;                // target KV sequence is intact
    bool slot_mtp_synced = false;           // slot has not drifted or failed MTP sync
    bool mem_shared = false;                // draft context shares KV memory with target
    std::int64_t target_pos = 0;            // current target KV token count (s.pos)
    std::int64_t draft_pos = -1;            // llama_memory_seq_pos_max(mem_dft, seq) + 1
    bool has_deferred = false;              // deferred catch-up rows queued for this slot
    std::int64_t deferred_first_pos = -1;   // first deferred token position
    std::int64_t deferred_last_pos = -1;    // last deferred token position
    std::size_t deferred_count = 0;         // number of deferred rows
    bool pending_features_valid = false;    // mtp_pending_h captured for this exact slot
    std::int64_t pending_features_pos = -1; // target_pos matching captured h_{target_pos - 1}
    bool pending_features_size_ok = false;  // mtp_pending_h.size() == mtp_n_embd && mtp_n_embd > 0
};

// Verifies that MTP draft context position (including any contiguous deferred
// catch-up rows) AND pending target hidden features are strictly synchronized
// with `target_pos`. Never accepts a bare boolean flip without matching positions.
[[nodiscard]] inline bool IsMtpDraftStateReady(const MtpReadinessState& st) noexcept {
    if (!st.draft_ctx_available || !st.slot_valid || !st.slot_mtp_synced) {
        return false;
    }
    if (st.target_pos <= 0) {
        return false;
    }
    if (!st.pending_features_valid || !st.pending_features_size_ok) {
        return false;
    }
    if (st.pending_features_pos != st.target_pos) {
        return false;
    }
    if (st.mem_shared) {
        return true;
    }
    if (st.has_deferred) {
        if (st.deferred_count == 0 || st.draft_pos < 0) {
            return false;
        }
        if (st.deferred_first_pos != st.draft_pos) {
            return false;
        }
        if (st.deferred_last_pos < st.deferred_first_pos) {
            return false;
        }
        if (st.deferred_last_pos + 1 != st.target_pos) {
            return false;
        }
        const std::int64_t span = st.deferred_last_pos - st.deferred_first_pos + 1;
        if (span != static_cast<std::int64_t>(st.deferred_count)) {
            return false;
        }
        return true;
    }
    return st.draft_pos == st.target_pos;
}

// Per-request originally configured speculation capability, tracked separately
// from temporary batch-suspension state so draining a batch does not lose it.
struct SlotSpecCapability {
    SlotSpecMode configured_mode = SlotSpecMode::kNone;
    bool hybrid_suffix_configured = false;   // draft-mtp + suffix burst escalation
    bool suffix_fallback_available = false;  // suffix matcher available if MTP drifts
    std::int32_t configured_draft_max = 0;
    std::int32_t fallback_suffix_draft_max = 16;
    std::int32_t hybrid_rs_cap = 0;
    bool can_hybrid_spec_ckpt = false;
    sonder::inference::ModelArchitecture architecture = sonder::inference::ModelArchitecture::attention_only;
};

struct SpeculationReactivationPolicy {
    // Consecutive sustained low-concurrency decode steps (1 active, 0 pending,
    // 0 exclusive pending) required after suspension before reactivating.
    std::uint32_t cooldown_steps = 4;

    // Minimum remaining token budget (min(max_new - generated, n_ctx - pos))
    // required to justify recovery overhead.
    std::int64_t min_remaining_tokens_history = 16;
    std::int64_t min_remaining_tokens_mtp = 8;

    // Additional remaining-token budget required per 4096 tokens of committed
    // history to amortize O(N) suffix/ngram rebuild cost.
    std::int64_t extra_remaining_per_4k_history = 2;

    // Allow hybrid draft-mtp + suffix requests whose MTP draft context drifted
    // during batching to safely fall back to history-based suffix drafting.
    bool allow_mtp_to_suffix_fallback = true;
};

[[nodiscard]] inline std::int64_t RequiredRemainingTokensForRecovery(
    SlotSpecMode target_mode, std::size_t committed_history_len,
    const SpeculationReactivationPolicy& policy) noexcept {
    if (target_mode == SlotSpecMode::kNone) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (target_mode == SlotSpecMode::kDraftMtp) {
        return std::max<std::int64_t>(1, policy.min_remaining_tokens_mtp);
    }
    const std::int64_t base = std::max<std::int64_t>(1, policy.min_remaining_tokens_history);
    constexpr std::size_t kMaxRebuildWindow = 131072;
    const std::size_t capped_len = std::min<std::size_t>(committed_history_len, kMaxRebuildWindow);
    const std::int64_t blocks = static_cast<std::int64_t>(capped_len / 4096u);
    const std::int64_t extra = std::max<std::int64_t>(0, policy.extra_remaining_per_4k_history) * blocks;
    return base + extra;
}

struct SpeculationReactivationInput {
    std::size_t active_count = 0;
    std::size_t pending_count = 0;
    bool pending_exclusive = false;
    bool cancelled = false;
    bool in_decode_phase = false;
    bool slot_valid = true;
    std::int64_t remaining_max_tokens = 0;
    std::int64_t remaining_ctx_tokens = 0;
    std::int64_t target_pos = 0;
    std::size_t committed_history_len = 0;
    bool history_matches_target_pos = false;
    MtpReadinessState mtp_readiness{};
};

struct SpeculationTransitionDecision {
    bool should_reactivate = false;
    SlotSpecMode target_mode = SlotSpecMode::kNone;
    bool enable_hybrid_suffix = false;
    std::int32_t effective_draft_max = 0;
    bool requires_history_rebuild = false;
    bool fell_back_from_mtp = false;
    SpecSuspendReason block_reason = SpecSuspendReason::kNone;
};

// Per-request state machine managing safe suspend -> cooldown -> validate ->
// rebuild -> reactivate transitions across single-request and batched phases.
class SpeculationTransitionState {
public:
    void Configure(const SlotSpecCapability& cap) noexcept {
        capability_ = cap;
        suspended_for_batch_ = false;
        low_concurrency_steps_ = 0;
        history_synced_ = false;
        suspend_count_ = 0;
        reactivation_count_ = 0;
        last_suspend_reason_ = SpecSuspendReason::kNone;

        const ArchitectureSpecGuardResult guard = CheckArchitectureSpecGuard(
            cap.architecture, cap.configured_mode, cap.configured_draft_max, cap.hybrid_rs_cap,
            cap.can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/true);
        if (!guard.allowed) {
            active_mode_ = SlotSpecMode::kNone;
            active_hybrid_suffix_ = false;
            active_draft_max_ = 0;
            permanently_disabled_ = true;
            if (cap.configured_mode != SlotSpecMode::kNone) {
                last_suspend_reason_ = SpecSuspendReason::kArchitectureUnsupported;
            }
            return;
        }

        permanently_disabled_ = false;
        active_mode_ = cap.configured_mode;
        active_hybrid_suffix_ = (cap.configured_mode == SlotSpecMode::kDraftMtp) && cap.hybrid_suffix_configured;
        active_draft_max_ = guard.effective_draft_max;
    }

    // Temporarily suspends active speculation while preserving originally
    // configured capability so the slot can safely reactivate after batch drain.
    void SuspendForBatch(SpecSuspendReason reason = SpecSuspendReason::kMultiRequestBatch) noexcept {
        if (!suspended_for_batch_) {
            ++suspend_count_;
        }
        suspended_for_batch_ = true;
        active_mode_ = SlotSpecMode::kNone;
        active_hybrid_suffix_ = false;
        active_draft_max_ = 0;
        history_synced_ = false;
        low_concurrency_steps_ = 0;
        last_suspend_reason_ = reason;
    }

    // Resets the sustained low-concurrency cooldown timer whenever concurrency,
    // pending work, exclusive work, or cancellation interrupts recovery.
    void NoteConcurrencyDisruption(SpecSuspendReason reason) noexcept {
        low_concurrency_steps_ = 0;
        if (suspended_for_batch_) {
            last_suspend_reason_ = reason;
        }
    }

    // Permanently disables MTP on this request after a runtime draft-context failure.
    // Keeps fallback suffix capability only if configured and slot stays valid.
    void MarkMtpRuntimeFailure() noexcept {
        if (capability_.configured_mode == SlotSpecMode::kDraftMtp) {
            if (capability_.suffix_fallback_available) {
                capability_.configured_mode = SlotSpecMode::kSuffix;
                capability_.configured_draft_max = capability_.fallback_suffix_draft_max;
                capability_.hybrid_suffix_configured = false;
            } else {
                capability_.configured_mode = SlotSpecMode::kNone;
                capability_.configured_draft_max = 0;
                permanently_disabled_ = true;
            }
        }
        if (active_mode_ == SlotSpecMode::kDraftMtp) {
            active_mode_ = SlotSpecMode::kNone;
            active_hybrid_suffix_ = false;
            active_draft_max_ = 0;
        }
        low_concurrency_steps_ = 0;
        last_suspend_reason_ = SpecSuspendReason::kMtpStateDrifted;
    }

    // Permanently disables all speculation on this request when its KV slot is invalidated.
    void MarkSlotInvalidated() noexcept {
        permanently_disabled_ = true;
        suspended_for_batch_ = false;
        active_mode_ = SlotSpecMode::kNone;
        active_hybrid_suffix_ = false;
        active_draft_max_ = 0;
        history_synced_ = false;
        low_concurrency_steps_ = 0;
        last_suspend_reason_ = SpecSuspendReason::kSlotInvalidated;
    }

    void MarkHistorySynced(bool synced) noexcept { history_synced_ = synced; }

    // Evaluates whether a currently suspended request may safely reactivate on
    // this single-request step. Call `CommitReactivation(decision)` ONLY after
    // completing any required history rebuild / readiness verification.
    [[nodiscard]] SpeculationTransitionDecision EvaluateSingleRequestStep(
        const SpeculationReactivationInput& in, const SpeculationReactivationPolicy& policy) noexcept {
        SpeculationTransitionDecision out;
        if (in.cancelled) {
            low_concurrency_steps_ = 0;
            out.block_reason = SpecSuspendReason::kCancelled;
            return out;
        }
        if (!in.slot_valid || permanently_disabled_) {
            low_concurrency_steps_ = 0;
            out.block_reason = !in.slot_valid ? SpecSuspendReason::kSlotInvalidated
                                              : (last_suspend_reason_ != SpecSuspendReason::kNone
                                                     ? last_suspend_reason_
                                                     : SpecSuspendReason::kArchitectureUnsupported);
            return out;
        }
        if (capability_.configured_mode == SlotSpecMode::kNone) {
            low_concurrency_steps_ = 0;
            out.block_reason = SpecSuspendReason::kNone;
            return out;
        }
        if (in.pending_exclusive) {
            low_concurrency_steps_ = 0;
            out.block_reason = SpecSuspendReason::kExclusivePending;
            return out;
        }
        if (in.pending_count > 0u) {
            low_concurrency_steps_ = 0;
            out.block_reason = SpecSuspendReason::kPendingRequest;
            return out;
        }
        if (in.active_count != 1u) {
            low_concurrency_steps_ = 0;
            out.block_reason = SpecSuspendReason::kMultiRequestBatch;
            return out;
        }
        if (!suspended_for_batch_) {
            out.target_mode = active_mode_;
            out.enable_hybrid_suffix = active_hybrid_suffix_;
            out.effective_draft_max = active_draft_max_;
            return out;
        }
        if (!in.in_decode_phase) {
            out.block_reason = SpecSuspendReason::kCooldownActive;
            return out;
        }

        if (low_concurrency_steps_ < std::numeric_limits<std::uint32_t>::max()) {
            ++low_concurrency_steps_;
        }
        if (low_concurrency_steps_ < policy.cooldown_steps) {
            out.block_reason = SpecSuspendReason::kCooldownActive;
            return out;
        }

        const std::int64_t effective_remaining = std::min(in.remaining_max_tokens, in.remaining_ctx_tokens);
        if (effective_remaining <= 1) {
            out.block_reason = SpecSuspendReason::kBudgetTooSmall;
            return out;
        }

        if (capability_.configured_mode == SlotSpecMode::kDraftMtp) {
            const ArchitectureSpecGuardResult mtp_guard = CheckArchitectureSpecGuard(
                capability_.architecture, SlotSpecMode::kDraftMtp, capability_.configured_draft_max,
                capability_.hybrid_rs_cap, capability_.can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/true);
            const bool mtp_ready = mtp_guard.allowed && IsMtpDraftStateReady(in.mtp_readiness);
            if (mtp_ready) {
                const std::int64_t min_rem =
                    RequiredRemainingTokensForRecovery(SlotSpecMode::kDraftMtp, in.committed_history_len, policy);
                if (effective_remaining < min_rem) {
                    out.block_reason = SpecSuspendReason::kBudgetTooSmall;
                    return out;
                }
                const bool can_hybrid_burst = capability_.hybrid_suffix_configured && in.history_matches_target_pos &&
                                              in.committed_history_len > 0 && in.target_pos > 0;
                out.should_reactivate = true;
                out.target_mode = SlotSpecMode::kDraftMtp;
                out.enable_hybrid_suffix = can_hybrid_burst;
                out.effective_draft_max = mtp_guard.effective_draft_max;
                out.requires_history_rebuild = can_hybrid_burst;
                out.fell_back_from_mtp = false;
                out.block_reason = SpecSuspendReason::kNone;
                return out;
            }

            // MTP draft state drifted or cannot safely reactivate; fall back to
            // suffix ONLY if hybrid suffix fallback is configured, architecture
            // rollback permits it, committed history is exact, and budget justifies it.
            if (policy.allow_mtp_to_suffix_fallback && capability_.suffix_fallback_available) {
                const std::int32_t fb_draft =
                    capability_.fallback_suffix_draft_max > 0 ? capability_.fallback_suffix_draft_max : 16;
                const ArchitectureSpecGuardResult suffix_guard = CheckArchitectureSpecGuard(
                    capability_.architecture, SlotSpecMode::kSuffix, fb_draft, capability_.hybrid_rs_cap,
                    capability_.can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/true);
                if (!suffix_guard.allowed) {
                    out.block_reason = SpecSuspendReason::kArchitectureUnsupported;
                    return out;
                }
                if (!in.history_matches_target_pos || in.committed_history_len == 0 || in.target_pos <= 0) {
                    out.block_reason = SpecSuspendReason::kHistoryOutOfSync;
                    return out;
                }
                const std::int64_t min_rem =
                    RequiredRemainingTokensForRecovery(SlotSpecMode::kSuffix, in.committed_history_len, policy);
                if (effective_remaining < min_rem) {
                    out.block_reason = SpecSuspendReason::kBudgetTooSmall;
                    return out;
                }
                out.should_reactivate = true;
                out.target_mode = SlotSpecMode::kSuffix;
                out.enable_hybrid_suffix = false;
                out.effective_draft_max = suffix_guard.effective_draft_max;
                out.requires_history_rebuild = true;
                out.fell_back_from_mtp = true;
                out.block_reason = SpecSuspendReason::kNone;
                return out;
            }

            out.block_reason =
                mtp_guard.allowed ? SpecSuspendReason::kMtpStateDrifted : SpecSuspendReason::kArchitectureUnsupported;
            return out;
        }

        // History-based modes: suffix, ngram-mod, ngram.
        const ArchitectureSpecGuardResult guard = CheckArchitectureSpecGuard(
            capability_.architecture, capability_.configured_mode, capability_.configured_draft_max,
            capability_.hybrid_rs_cap, capability_.can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/true);
        if (!guard.allowed) {
            out.block_reason = SpecSuspendReason::kArchitectureUnsupported;
            return out;
        }
        if (!in.history_matches_target_pos || in.committed_history_len == 0 || in.target_pos <= 0) {
            out.block_reason = SpecSuspendReason::kHistoryOutOfSync;
            return out;
        }
        const std::int64_t min_rem =
            RequiredRemainingTokensForRecovery(capability_.configured_mode, in.committed_history_len, policy);
        if (effective_remaining < min_rem) {
            out.block_reason = SpecSuspendReason::kBudgetTooSmall;
            return out;
        }

        out.should_reactivate = true;
        out.target_mode = capability_.configured_mode;
        out.enable_hybrid_suffix = false;
        out.effective_draft_max = guard.effective_draft_max;
        out.requires_history_rebuild = true;
        out.fell_back_from_mtp = false;
        out.block_reason = SpecSuspendReason::kNone;
        return out;
    }

    // Commits a previously evaluated reactivation ONLY after the caller has
    // rebuilt/validated all required draft state.
    void CommitReactivation(const SpeculationTransitionDecision& d) noexcept {
        if (!d.should_reactivate || d.target_mode == SlotSpecMode::kNone || d.effective_draft_max <= 0) {
            return;
        }
        suspended_for_batch_ = false;
        active_mode_ = d.target_mode;
        active_hybrid_suffix_ = d.enable_hybrid_suffix;
        active_draft_max_ = d.effective_draft_max;
        history_synced_ = true;
        low_concurrency_steps_ = 0;
        last_suspend_reason_ = SpecSuspendReason::kNone;
        ++reactivation_count_;
    }

    [[nodiscard]] const SlotSpecCapability& capability() const noexcept { return capability_; }
    [[nodiscard]] SlotSpecMode configured_mode() const noexcept { return capability_.configured_mode; }
    [[nodiscard]] SlotSpecMode active_mode() const noexcept { return active_mode_; }
    [[nodiscard]] bool active_hybrid_suffix() const noexcept { return active_hybrid_suffix_; }
    [[nodiscard]] std::int32_t active_draft_max() const noexcept { return active_draft_max_; }
    [[nodiscard]] bool suspended_for_batch() const noexcept { return suspended_for_batch_; }
    [[nodiscard]] bool permanently_disabled() const noexcept { return permanently_disabled_; }
    [[nodiscard]] bool history_synced() const noexcept { return history_synced_; }
    [[nodiscard]] std::uint32_t low_concurrency_steps() const noexcept { return low_concurrency_steps_; }
    [[nodiscard]] std::uint32_t suspend_count() const noexcept { return suspend_count_; }
    [[nodiscard]] std::uint32_t reactivation_count() const noexcept { return reactivation_count_; }
    [[nodiscard]] SpecSuspendReason last_suspend_reason() const noexcept { return last_suspend_reason_; }

private:
    SlotSpecCapability capability_{};
    SlotSpecMode active_mode_ = SlotSpecMode::kNone;
    bool active_hybrid_suffix_ = false;
    std::int32_t active_draft_max_ = 0;
    bool suspended_for_batch_ = false;
    bool permanently_disabled_ = false;
    bool history_synced_ = false;
    std::uint32_t low_concurrency_steps_ = 0;
    std::uint32_t suspend_count_ = 0;
    std::uint32_t reactivation_count_ = 0;
    SpecSuspendReason last_suspend_reason_ = SpecSuspendReason::kNone;
};

// ============================================================================
// Bounded Concurrent Small-Batch Linear Speculation Coordinator (Doc 2 #20)
// ============================================================================

struct ConcurrentSpecPolicy {
    bool enabled = false;                             // opt-in by default
    std::uint32_t max_concurrent_slots = 4;           // small-batch threshold (2..max_concurrent_slots)
    std::int32_t max_aggregate_draft_tokens = 16;     // total draft tokens across all slots per verify step
    std::int32_t max_per_slot_draft_tokens = 8;       // per-slot draft token cap during concurrent batching
    double verify_regression_ratio = 1.05;            // pause if spec ms/tok exceeds plain ms/tok * ratio
    double min_acceptance_floor = 0.15;               // pause after low-acceptance streak even before plain baseline
    std::uint32_t min_cost_samples = 3;               // minimum observations before regression comparison fires
    std::uint32_t low_acceptance_streak_limit = 4;    // consecutive poor-acceptance steps triggering cooldown
    std::uint32_t regression_cooldown_steps = 8;      // initial plain-batch cooldown on regression (backs off to 64)
};

enum class ConcurrentSpecFallbackReason : std::uint8_t {
    kNone = 0,
    kDisabled,
    kSingleOrEmptyBatch,
    kBatchExceedsThreshold,
    kPrefillActive,
    kArchitectureUnsupported,
    kVerifyCostRegressed,
    kBudgetExhausted,
    kNoProposals,
};

[[nodiscard]] constexpr std::string_view ToString(ConcurrentSpecFallbackReason reason) noexcept {
    switch (reason) {
        case ConcurrentSpecFallbackReason::kNone: return "none";
        case ConcurrentSpecFallbackReason::kDisabled: return "disabled";
        case ConcurrentSpecFallbackReason::kSingleOrEmptyBatch: return "single_or_empty_batch";
        case ConcurrentSpecFallbackReason::kBatchExceedsThreshold: return "batch_exceeds_threshold";
        case ConcurrentSpecFallbackReason::kPrefillActive: return "prefill_active";
        case ConcurrentSpecFallbackReason::kArchitectureUnsupported: return "architecture_unsupported";
        case ConcurrentSpecFallbackReason::kVerifyCostRegressed: return "verify_cost_regressed";
        case ConcurrentSpecFallbackReason::kBudgetExhausted: return "budget_exhausted";
        case ConcurrentSpecFallbackReason::kNoProposals: return "no_proposals";
    }
    return "unknown";
}

// Measured verification cost-regression gate comparing concurrent speculative
// ms/committed-token against ordinary 1-token batched decode ms/committed-token.
class ConcurrentSpecCostGate {
public:
    void Reset() noexcept {
        plain_ms_per_tok_ema_ = 0.0;
        spec_ms_per_tok_ema_ = 0.0;
        plain_samples_ = 0;
        spec_samples_ = 0;
        low_accept_streak_ = 0;
        cooldown_remaining_ = 0;
        cooldown_period_ = 0;
        regression_trips_ = 0;
    }

    void ObservePlainStep(double step_ms, std::size_t committed_tokens) noexcept {
        if (!std::isfinite(step_ms) || step_ms <= 0.0 || committed_tokens == 0) {
            return;
        }
        const double ms_per_tok = step_ms / static_cast<double>(committed_tokens);
        constexpr double kAlpha = 0.25;
        plain_ms_per_tok_ema_ =
            (plain_samples_ == 0) ? ms_per_tok : (1.0 - kAlpha) * plain_ms_per_tok_ema_ + kAlpha * ms_per_tok;
        if (plain_samples_ < std::numeric_limits<std::uint32_t>::max()) {
            ++plain_samples_;
        }
    }

    void NotePlainStepInCooldown() noexcept {
        if (cooldown_remaining_ > 0) {
            --cooldown_remaining_;
        }
    }

    // Returns true if this step tripped a cost-regression or low-acceptance pause.
    bool ObserveSpecStep(double step_ms, std::size_t committed_tokens, std::size_t drafted_tokens,
                         std::size_t accepted_tokens, const ConcurrentSpecPolicy& policy) noexcept {
        if (committed_tokens == 0) {
            return false;
        }
        const std::size_t clamped_acc = std::min(accepted_tokens, drafted_tokens);
        if (std::isfinite(step_ms) && step_ms > 0.0) {
            const double ms_per_tok = step_ms / static_cast<double>(committed_tokens);
            constexpr double kAlpha = 0.25;
            spec_ms_per_tok_ema_ =
                (spec_samples_ == 0) ? ms_per_tok : (1.0 - kAlpha) * spec_ms_per_tok_ema_ + kAlpha * ms_per_tok;
            if (spec_samples_ < std::numeric_limits<std::uint32_t>::max()) {
                ++spec_samples_;
            }
        }

        const std::uint32_t min_samples = std::max<std::uint32_t>(1u, policy.min_cost_samples);
        const double ratio = (std::isfinite(policy.verify_regression_ratio) && policy.verify_regression_ratio > 0.0)
                                 ? policy.verify_regression_ratio
                                 : 1.05;
        if (plain_samples_ >= min_samples && spec_samples_ >= min_samples && plain_ms_per_tok_ema_ > 0.0 &&
            spec_ms_per_tok_ema_ > plain_ms_per_tok_ema_ * ratio) {
            TripCooldown(policy);
            return true;
        }

        if (drafted_tokens > 0) {
            const double accept_rate = static_cast<double>(clamped_acc) / static_cast<double>(drafted_tokens);
            const double floor = std::clamp(policy.min_acceptance_floor, 0.0, 1.0);
            if (accept_rate < floor) {
                ++low_accept_streak_;
                const std::uint32_t streak_lim = std::max<std::uint32_t>(1u, policy.low_acceptance_streak_limit);
                if (low_accept_streak_ >= streak_lim) {
                    low_accept_streak_ = 0;
                    TripCooldown(policy);
                    return true;
                }
            } else {
                low_accept_streak_ = 0;
                // Healthy acceptance resets backoff period so recovery stays responsive.
                cooldown_period_ = std::max<std::uint32_t>(1u, policy.regression_cooldown_steps);
            }
        }
        return false;
    }

    [[nodiscard]] bool IsRegressed() const noexcept { return cooldown_remaining_ > 0; }
    [[nodiscard]] std::uint32_t cooldown_remaining() const noexcept { return cooldown_remaining_; }
    [[nodiscard]] std::uint32_t regression_trips() const noexcept { return regression_trips_; }
    [[nodiscard]] double plain_ms_per_tok_ema() const noexcept { return plain_ms_per_tok_ema_; }
    [[nodiscard]] double spec_ms_per_tok_ema() const noexcept { return spec_ms_per_tok_ema_; }
    [[nodiscard]] std::uint32_t plain_samples() const noexcept { return plain_samples_; }
    [[nodiscard]] std::uint32_t spec_samples() const noexcept { return spec_samples_; }

private:
    void TripCooldown(const ConcurrentSpecPolicy& policy) noexcept {
        const std::uint32_t base = std::max<std::uint32_t>(1u, policy.regression_cooldown_steps);
        if (cooldown_period_ < base) {
            cooldown_period_ = base;
        }
        cooldown_remaining_ = std::min<std::uint32_t>(cooldown_period_, 64u);
        cooldown_period_ = std::min<std::uint32_t>(cooldown_period_ * 2u, 64u);
        ++regression_trips_;
    }

    double plain_ms_per_tok_ema_ = 0.0;
    double spec_ms_per_tok_ema_ = 0.0;
    std::uint32_t plain_samples_ = 0;
    std::uint32_t spec_samples_ = 0;
    std::uint32_t low_accept_streak_ = 0;
    std::uint32_t cooldown_remaining_ = 0;
    std::uint32_t cooldown_period_ = 0;
    std::uint32_t regression_trips_ = 0;
};

struct ConcurrentSpecSlotInput {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    std::int64_t base_pos = 0;
    bool is_prefill = false;
    bool cancelled = false;
    bool eligible_for_spec = false;
    std::int32_t max_draft_for_slot = 0;
    std::size_t proposed_draft_count = 0;
};

struct ConcurrentSpecSlotPlan {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    std::int64_t base_pos = 0;
    std::int32_t allocated_draft_tokens = 0;  // 0..proposed_draft_count
    std::int32_t batch_start_idx = 0;         // index of base token in packed batch; drafts follow contiguously
    std::int32_t total_slot_tokens = 1;       // 1 + allocated_draft_tokens
};

struct ConcurrentSpecBatchPlan {
    bool use_concurrent_spec = false;
    ConcurrentSpecFallbackReason fallback_reason = ConcurrentSpecFallbackReason::kNone;
    std::vector<ConcurrentSpecSlotPlan> slots;
    std::int32_t total_base_tokens = 0;
    std::int32_t total_draft_tokens = 0;
    std::int32_t total_batch_tokens = 0;
};

// Plans ONE bounded small-batch concurrent linear speculation step across
// active decoding slots, enforcing:
//   - opt-in enable flag
//   - zero active prefill chunks (falls back immediately if any prefill active)
//   - small-batch cap (2 <= decode_slots <= max_concurrent_slots)
//   - architecture rollback guard (attention_only, or hybrid with rs_cap > 0)
//   - measured verify cost regression gate
//   - aggregate draft budget <= min(max_aggregate_draft_tokens, token_budget - B, output_budget - B)
//   - per-slot cap <= min(max_per_slot_draft_tokens, per_seq_output_budget - 1, slot.max_draft_for_slot, hybrid_rs_cap)
//   - fair water-filling distribution with round-robin remainder rotation
//   - disjoint contiguous [batch_start_idx, batch_start_idx + allocated_draft_tokens] spans per slot
[[nodiscard]] inline ConcurrentSpecBatchPlan PlanConcurrentLinearSpeculation(
    std::span<const ConcurrentSpecSlotInput> active_slots, const ConcurrentSpecPolicy& policy,
    std::int32_t step_token_budget, std::int32_t step_output_budget, std::int32_t per_seq_output_budget,
    sonder::inference::ModelArchitecture architecture, std::int32_t hybrid_rs_cap, bool cost_regressed,
    std::size_t& rr_cursor) {
    ConcurrentSpecBatchPlan plan;
    if (!policy.enabled) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kDisabled;
        return plan;
    }
    if (cost_regressed) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kVerifyCostRegressed;
        return plan;
    }
    if (architecture == sonder::inference::ModelArchitecture::recurrent) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kArchitectureUnsupported;
        return plan;
    }
    if (architecture == sonder::inference::ModelArchitecture::hybrid && hybrid_rs_cap <= 0) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kArchitectureUnsupported;
        return plan;
    }

    std::vector<const ConcurrentSpecSlotInput*> decode_slots;
    decode_slots.reserve(active_slots.size());
    for (const ConcurrentSpecSlotInput& s : active_slots) {
        if (s.cancelled) continue;
        if (s.is_prefill) {
            // Immediate fallback to ordinary chunked-prefill continuous batching.
            plan.fallback_reason = ConcurrentSpecFallbackReason::kPrefillActive;
            return plan;
        }
        decode_slots.push_back(&s);
    }

    const std::size_t n_decode = decode_slots.size();
    if (n_decode < 2u) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kSingleOrEmptyBatch;
        return plan;
    }
    if (policy.max_concurrent_slots < 2u || n_decode > static_cast<std::size_t>(policy.max_concurrent_slots)) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kBatchExceedsThreshold;
        return plan;
    }

    const auto n_base = static_cast<std::int32_t>(n_decode);
    const std::int32_t max_tokens_avail = std::min(step_token_budget, step_output_budget);
    if (max_tokens_avail <= n_base || per_seq_output_budget <= 1 || policy.max_aggregate_draft_tokens <= 0 ||
        policy.max_per_slot_draft_tokens <= 0) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kBudgetExhausted;
        return plan;
    }

    const std::int32_t agg_budget = std::min<std::int32_t>(policy.max_aggregate_draft_tokens, max_tokens_avail - n_base);
    if (agg_budget <= 0) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kBudgetExhausted;
        return plan;
    }

    const std::int32_t global_per_slot_cap = [&]() noexcept {
        std::int32_t cap = std::min<std::int32_t>(policy.max_per_slot_draft_tokens, per_seq_output_budget - 1);
        if (architecture == sonder::inference::ModelArchitecture::hybrid) {
            cap = std::min<std::int32_t>(cap, hybrid_rs_cap);
        }
        return std::max<std::int32_t>(0, cap);
    }();
    if (global_per_slot_cap <= 0) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kBudgetExhausted;
        return plan;
    }

    // Compute per-slot demand clamped to all safety bounds.
    std::vector<std::int32_t> demand(n_decode, 0);
    std::int32_t total_demand = 0;
    for (std::size_t i = 0; i < n_decode; ++i) {
        const ConcurrentSpecSlotInput& s = *decode_slots[i];
        if (!s.eligible_for_spec || s.max_draft_for_slot <= 0 || s.proposed_draft_count == 0) {
            continue;
        }
        const std::int32_t prop =
            static_cast<std::int32_t>(std::min<std::size_t>(s.proposed_draft_count, static_cast<std::size_t>(global_per_slot_cap)));
        const std::int32_t d = std::min({prop, s.max_draft_for_slot, global_per_slot_cap});
        if (d > 0) {
            demand[i] = d;
            total_demand += d;
        }
    }
    if (total_demand <= 0) {
        plan.fallback_reason = ConcurrentSpecFallbackReason::kNoProposals;
        return plan;
    }

    // Fair round-robin water-filling allocation within `agg_budget`.
    std::vector<std::int32_t> alloc(n_decode, 0);
    std::int32_t rem = std::min(agg_budget, total_demand);
    const std::size_t start = rr_cursor % n_decode;
    while (rem > 0) {
        bool progressed = false;
        for (std::size_t off = 0; off < n_decode && rem > 0; ++off) {
            const std::size_t idx = (start + off) % n_decode;
            if (alloc[idx] < demand[idx]) {
                ++alloc[idx];
                --rem;
                progressed = true;
            }
        }
        if (!progressed) break;
    }

    plan.slots.reserve(n_decode);
    std::int32_t batch_cursor = 0;
    for (std::size_t i = 0; i < n_decode; ++i) {
        const ConcurrentSpecSlotInput& s = *decode_slots[i];
        ConcurrentSpecSlotPlan item;
        item.slot_index = s.slot_index;
        item.seq_id = s.seq_id;
        item.base_pos = s.base_pos;
        item.allocated_draft_tokens = alloc[i];
        item.batch_start_idx = batch_cursor;
        item.total_slot_tokens = 1 + alloc[i];
        batch_cursor += item.total_slot_tokens;
        plan.total_base_tokens += 1;
        plan.total_draft_tokens += alloc[i];
        plan.slots.push_back(item);
    }
    plan.total_batch_tokens = batch_cursor;
    plan.use_concurrent_spec = (plan.total_draft_tokens > 0);
    if (!plan.use_concurrent_spec) {
        plan.slots.clear();
        plan.total_base_tokens = 0;
        plan.total_batch_tokens = 0;
        plan.fallback_reason = ConcurrentSpecFallbackReason::kNoProposals;
        return plan;
    }

    rr_cursor = (start + 1u) % n_decode;
    return plan;
}

// Exact per-sequence KV rollback descriptor after partial or full verification
// of one slot inside a single or concurrent speculative batch.
struct SlotKvRollbackRange {
    std::int32_t seq_id = 0;
    bool needs_rollback = false;
    std::int64_t keep_pos = 0;        // new KV length for this sequence after commit
    std::int64_t batch_end_pos = 0;   // KV length written by the verification batch
    std::int64_t rm_from_pos = 0;     // llama_memory_seq_rm(mem, seq_id, rm_from_pos, -1)
    std::int64_t rejected_tokens = 0; // batch_end_pos - keep_pos
};

// Computes the exact per-sequence KV rollback range for one slot whose batch
// evaluated `base_token` at `base_pos` plus `drafted_count` draft tokens at
// `[base_pos + 1, base_pos + drafted_count]`, and committed `committed_drafts`
// of those draft tokens (0 <= committed_drafts <= drafted_count).
[[nodiscard]] inline SlotKvRollbackRange ComputeSlotRollbackRange(std::int32_t seq_id, std::int64_t base_pos,
                                                                  std::size_t drafted_count,
                                                                  std::size_t committed_drafts) noexcept {
    const std::size_t clamped_committed = std::min(committed_drafts, drafted_count);
    SlotKvRollbackRange r;
    r.seq_id = seq_id;
    r.keep_pos = base_pos + 1 + static_cast<std::int64_t>(clamped_committed);
    r.batch_end_pos = base_pos + 1 + static_cast<std::int64_t>(drafted_count);
    r.rm_from_pos = r.keep_pos;
    r.rejected_tokens = r.batch_end_pos - r.keep_pos;
    r.needs_rollback = (r.keep_pos < r.batch_end_pos);
    return r;
}

// Pure linear-verification outcome for one slot's target-sampled tokens
// against its proposed draft vector (used by unit tests and verification logic).
struct LinearDraftVerifyOutcome {
    std::size_t accepted_drafts = 0;
    std::vector<SpecToken> emitted_tokens;  // accepted prefix + 1 correction or bonus token
    SlotKvRollbackRange rollback{};
};

// Verifies one slot's proposed linear draft tokens against target samples at
// positions `[0 .. proposed.size()]` (where sample[j] is sampled from the logits
// after `base_token` + `proposed[0..j-1]`), returning the exact accepted prefix
// + correction/bonus token and the isolated per-sequence KV rollback range.
[[nodiscard]] inline LinearDraftVerifyOutcome VerifyLinearDraftSlot(std::int32_t seq_id, std::int64_t base_pos,
                                                                    std::span<const SpecToken> proposed,
                                                                    std::span<const SpecToken> target_samples) {
    LinearDraftVerifyOutcome out;
    const std::size_t k = proposed.size();
    out.emitted_tokens.reserve(k + 1u);
    std::size_t n_accepted = 0;
    for (std::size_t j = 0; j < k && j < target_samples.size(); ++j) {
        const SpecToken sampled = target_samples[j];
        out.emitted_tokens.push_back(sampled);
        if (sampled == proposed[j]) {
            ++n_accepted;
        } else {
            break;
        }
    }
    if (n_accepted == k && k < target_samples.size()) {
        out.emitted_tokens.push_back(target_samples[k]);
    }
    out.accepted_drafts = n_accepted;
    out.rollback = ComputeSlotRollbackRange(seq_id, base_pos, k, n_accepted);
    return out;
}

}  // namespace sonder::backends::llamacpp
