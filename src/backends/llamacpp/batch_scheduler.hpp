// Continuous Batching + Chunked Prefill Scheduler State Machine.
//
// Pure stdlib-only C++20 planner shared by LlamaCppBackend's single-owner batch
// worker loop and unit tests. Handles:
//   - Free-slot prefix selection (session_key -> longest prefix -> LRU)
//   - Zero-overhead single-request fast-path eligibility check
//   - Compact online EMA StepCostEstimator conditioned on decode tokens,
//     prefill tokens, and total KV context length
//   - Latency-target-aware adaptive mixed-batch prefill budgeting protecting
//     foreground decode ITL while guaranteeing anti-starvation aging for
//     queued prefills and background work
//   - Lockstep co-packing of 1 decode token per decoding slot TOGETHER with
//     priority-ordered chunked prefill without head-of-line blocking.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

enum class BatchSlotPhase {
    kIdle = 0,
    kPrefill,
    kDecode,
};

struct BatchSlotPlanView {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    BatchSlotPhase phase = BatchSlotPhase::kIdle;
    std::size_t prompt_len = 0;
    std::size_t prefill_offset = 0;
    std::int64_t pos = 0;
    bool cancelled = false;
    // Scheduling priority, optional latency targets (ms), monotonic aging
    // timestamps (microseconds), current KV context length, and selective
    // safe-point suspension flag.
    sonder::inference::RequestPriority priority = sonder::inference::RequestPriority::subagent;
    std::optional<double> target_ttft_ms = std::nullopt;
    std::optional<double> target_itl_ms = std::nullopt;
    std::int64_t enqueue_time_us = 0;
    std::int64_t last_service_time_us = 0;
    std::int64_t kv_context_tokens = 0;  // <= 0 falls back to max(0, pos)
    bool suspended = false;
};

// Cost-aware physical preemption policy for direct llama.cpp slots (Item #9).
enum class PhysicalPreemptionPolicy : std::uint8_t {
    kRecompute = 0,
    kSpillToCheckpoint = 1,
};

struct PreemptionVictimView {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    int priority_rank = 0;          // higher = less urgent = preferred victim
    std::size_t kv_tokens = 0;
    std::uint64_t admit_order = 0;  // newer = preferred victim on equal priority
    std::uint32_t requeue_count = 0;
    bool cancelled = false;
    bool suspended = false;
    bool allow_spill = false;
};

struct PhysicalPreemptionCostParams {
    std::uint64_t bytes_per_token = 4096;
    double prefill_tokens_per_sec = 2000.0;
    double spill_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    double restore_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    double fixed_spill_overhead_us = 250.0;
    double fixed_recompute_overhead_us = 50.0;
    std::size_t min_spill_tokens = 256;
    std::uint64_t available_spill_bytes = 0;  // 0 = unbounded / caller cap
    double spill_margin = 1.05;
    std::uint32_t max_requeue_count = 8;
    std::uint32_t anti_starvation_cap = 3;
};

struct PhysicalPreemptionChoice {
    std::size_t victim_slot_index = 0;
    PhysicalPreemptionPolicy policy = PhysicalPreemptionPolicy::kRecompute;
    double recompute_cost_us = 0.0;
    double spill_restore_cost_us = 0.0;
    std::uint64_t estimated_spill_bytes = 0;
    bool budget_exceeded = false;
    bool hit_requeue_limit = false;
    bool bypassed_protected_peer = false;
};

// Evaluates Recompute (default) vs SpillToCheckpoint from measured/estimated
// spill+restore cost vs recompute cost and available spill byte budget.
[[nodiscard]] inline PhysicalPreemptionChoice EvaluatePhysicalPreemptionCost(
    std::size_t kv_tokens, bool allow_spill, std::uint32_t requeue_count,
    const PhysicalPreemptionCostParams& params) noexcept {
    PhysicalPreemptionChoice out;
    out.policy = PhysicalPreemptionPolicy::kRecompute;
    out.hit_requeue_limit = requeue_count >= params.max_requeue_count;

    const std::uint64_t tokens = static_cast<std::uint64_t>(kv_tokens);
    const std::uint64_t max_u64 = std::numeric_limits<std::uint64_t>::max();
    if (params.bytes_per_token > 0 && tokens > max_u64 / params.bytes_per_token) {
        out.estimated_spill_bytes = max_u64;
    } else {
        out.estimated_spill_bytes = tokens * params.bytes_per_token;
    }

    const bool valid_prefill = std::isfinite(params.prefill_tokens_per_sec) && params.prefill_tokens_per_sec > 0.0;
    const bool valid_spill = std::isfinite(params.spill_bytes_per_sec) && params.spill_bytes_per_sec > 0.0;
    const bool valid_restore = std::isfinite(params.restore_bytes_per_sec) && params.restore_bytes_per_sec > 0.0;
    const double fixed_recompute =
        (std::isfinite(params.fixed_recompute_overhead_us) && params.fixed_recompute_overhead_us >= 0.0)
            ? params.fixed_recompute_overhead_us
            : 0.0;
    const double fixed_spill =
        (std::isfinite(params.fixed_spill_overhead_us) && params.fixed_spill_overhead_us >= 0.0)
            ? params.fixed_spill_overhead_us
            : 0.0;

    if (valid_prefill) {
        out.recompute_cost_us =
            fixed_recompute + (static_cast<double>(tokens) / params.prefill_tokens_per_sec) * 1'000'000.0;
    }
    if (valid_spill && valid_restore && out.estimated_spill_bytes > 0) {
        const double b = static_cast<double>(out.estimated_spill_bytes);
        out.spill_restore_cost_us = fixed_spill + ((b / params.spill_bytes_per_sec) + (b / params.restore_bytes_per_sec)) * 1'000'000.0;
    } else {
        out.spill_restore_cost_us = fixed_spill;
    }

    if (!allow_spill || kv_tokens < params.min_spill_tokens || !valid_prefill || !valid_spill || !valid_restore ||
        out.estimated_spill_bytes == 0) {
        return out;
    }
    if (params.available_spill_bytes > 0 && out.estimated_spill_bytes > params.available_spill_bytes) {
        out.budget_exceeded = true;
        return out;
    }
    const double margin = (std::isfinite(params.spill_margin) && params.spill_margin >= 1.0) ? params.spill_margin : 1.0;
    if (out.spill_restore_cost_us * margin < out.recompute_cost_us) {
        out.policy = PhysicalPreemptionPolicy::kSpillToCheckpoint;
    }
    return out;
}

// Picks one active slot to selectively preempt under physical KV pressure:
//   - Prefers unprotected slots (requeue_count < anti_starvation_cap) over
//     anti-starvation-protected slots when an unprotected peer exists.
//   - Within the eligible set, picks the least urgent (highest priority_rank),
//     breaking ties by newest admit_order, then largest kv_tokens.
//   - Never evicts the sole live slot unless `allow_last_slot` is true.
[[nodiscard]] inline std::optional<PhysicalPreemptionChoice> SelectSelectivePreemptionVictim(
    std::span<const PreemptionVictimView> active_slots, const PhysicalPreemptionCostParams& params,
    bool allow_last_slot = false) noexcept {
    std::size_t live_count = 0;
    for (const auto& s : active_slots) {
        if (!s.cancelled && !s.suspended) ++live_count;
    }
    if (live_count == 0 || (!allow_last_slot && live_count <= 1u)) {
        return std::nullopt;
    }

    auto pick_from = [&](bool skip_protected, bool& saw_protected) -> std::optional<std::size_t> {
        std::optional<std::size_t> best;
        for (std::size_t i = 0; i < active_slots.size(); ++i) {
            const auto& s = active_slots[i];
            if (s.cancelled || s.suspended) continue;
            const bool is_protected = params.anti_starvation_cap > 0 && s.requeue_count >= params.anti_starvation_cap;
            if (skip_protected && is_protected) {
                saw_protected = true;
                continue;
            }
            if (!best.has_value()) {
                best = i;
                continue;
            }
            const auto& cur = active_slots[*best];
            if (s.priority_rank != cur.priority_rank) {
                if (s.priority_rank > cur.priority_rank) best = i;
            } else if (s.admit_order != cur.admit_order) {
                if (s.admit_order > cur.admit_order) best = i;
            } else if (s.kv_tokens > cur.kv_tokens) {
                best = i;
            }
        }
        return best;
    };

    bool saw_protected = false;
    auto chosen_idx = pick_from(/*skip_protected=*/true, saw_protected);
    bool bypassed = saw_protected && chosen_idx.has_value();
    if (!chosen_idx.has_value()) {
        chosen_idx = pick_from(/*skip_protected=*/false, saw_protected);
        bypassed = false;
    }
    if (!chosen_idx.has_value()) {
        return std::nullopt;
    }
    const auto& v = active_slots[*chosen_idx];
    PhysicalPreemptionChoice choice =
        EvaluatePhysicalPreemptionCost(v.kv_tokens, v.allow_spill, v.requeue_count, params);
    choice.victim_slot_index = v.slot_index;
    choice.bypassed_protected_peer = bypassed;
    return choice;
}

struct PlannedBatchItem {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    bool is_prefill = false;
    std::size_t token_offset = 0;  // index into prompt for prefill; ignored for decode
    std::int32_t token_count = 0;  // 1 for decode, 1..budget for prefill chunk
    std::int64_t pos_start = 0;
    bool request_logits = false;   // true on decode or final prompt token of prefill
};

struct PlannedBatchStep {
    std::vector<PlannedBatchItem> items;
    std::int32_t total_tokens = 0;
    std::int32_t total_outputs = 0;
    std::size_t decode_slots = 0;
    std::size_t prefill_slots = 0;
};

// One measured `llama_decode` step observation fed into StepCostEstimator.
struct StepCostObservation {
    std::int32_t decode_tokens = 0;
    std::int32_t prefill_tokens = 0;
    std::int64_t total_kv_tokens = 0;
    double duration_ms = 0.0;
};

struct StepCostEstimatorConfig {
    double ema_alpha = 0.25;               // steady-state EMA smoothing factor in (0, 1]
    std::uint64_t min_samples = 3;         // minimum observations before calibrated
    double max_observation_ms = 600000.0;  // sanity clamp (10 minutes)
};

// Compact online EMA cost model over observed `llama_decode` step durations:
//   T(d, p, K) = base_ms
//              + ms_per_decode_token  * d
//              + ms_per_prefill_token * p
//              + ms_per_kv_ktoken     * (K / 1024)
// All coefficients are projected to >= 0 so predictions are strictly monotonic
// in decode tokens, prefill tokens, and KV context length, and invertible in
// O(1) for prefill budgeting. Zero heap allocations.
class StepCostEstimator {
public:
    explicit StepCostEstimator(StepCostEstimatorConfig cfg = {}) noexcept {
        ema_alpha_ = (std::isfinite(cfg.ema_alpha) && cfg.ema_alpha > 0.0 && cfg.ema_alpha <= 1.0)
                         ? cfg.ema_alpha
                         : 0.25;
        min_samples_ = cfg.min_samples > 0u ? cfg.min_samples : 1u;
        max_observation_ms_ = (std::isfinite(cfg.max_observation_ms) && cfg.max_observation_ms > 0.0)
                                  ? cfg.max_observation_ms
                                  : 600000.0;
    }

    void Reset() noexcept {
        base_ms_ = 0.0;
        ms_per_decode_token_ = 0.0;
        ms_per_prefill_token_ = 0.0;
        ms_per_kv_ktoken_ = 0.0;
        total_samples_ = 0;
        decode_samples_ = 0;
        prefill_samples_ = 0;
        mixed_samples_ = 0;
        last_observed_ms_ = 0.0;
    }

    void Observe(const StepCostObservation& obs) noexcept {
        if (!std::isfinite(obs.duration_ms) || obs.duration_ms < 0.0) return;
        const std::int32_t d = obs.decode_tokens > 0 ? obs.decode_tokens : 0;
        const std::int32_t p = obs.prefill_tokens > 0 ? obs.prefill_tokens : 0;
        if (d == 0 && p == 0) return;

        const std::int64_t kv = obs.total_kv_tokens > 0 ? obs.total_kv_tokens : std::int64_t{0};
        const double kv_k = static_cast<double>(kv) / 1024.0;
        const double y = std::min(obs.duration_ms, max_observation_ms_);

        const bool had_decode = decode_samples_ > 0u;
        const bool had_prefill = prefill_samples_ > 0u;

        // First mixed observation after an established decode baseline:
        // attribute positive excess directly to marginal prefill token cost so
        // prefill chunk sizing adapts immediately on the very first mixed step.
        if (d > 0 && p > 0 && !had_prefill && had_decode) {
            const double decode_base = PredictStepMs(d, 0, kv);
            const double rem = y - decode_base;
            if (rem > 0.0) {
                ms_per_prefill_token_ = rem / static_cast<double>(p);
                ++total_samples_;
                ++decode_samples_;
                ++prefill_samples_;
                ++mixed_samples_;
                last_observed_ms_ = y;
                return;
            }
        }

        // Characteristic units equalize feature magnitudes for well-conditioned
        // projected normalized LMS / EMA updates across pure & mixed regimes.
        const double u0 = 1.0 / kScaleBase;
        const double u1 = static_cast<double>(d) / kScaleDecode;
        const double u2 = static_cast<double>(p) / kScalePrefill;
        const double u3 = kv_k / kScaleKvKtok;
        const double norm2 = u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3;

        double w0 = base_ms_ * kScaleBase;
        double w1 = ms_per_decode_token_ * kScaleDecode;
        double w2 = ms_per_prefill_token_ * kScalePrefill;
        double w3 = ms_per_kv_ktoken_ * kScaleKvKtok;

        const double y_hat = w0 * u0 + w1 * u1 + w2 * u2 + w3 * u3;
        const double err = y - y_hat;

        double eta = 1.0;
        if (total_samples_ > 0u) {
            if (d > 0 && p == 0 && !had_decode && had_prefill) {
                // First pure-decode observation after prefill-only warmup:
                // snap the decode/base/KV subspace without disturbing w2.
                eta = 1.0;
            } else {
                const double inv_n = 1.0 / static_cast<double>(total_samples_ + 1u);
                eta = std::min(1.0, std::max(ema_alpha_, inv_n));
            }
        }

        const double step = (eta * err) / norm2;
        w0 = std::max(0.0, w0 + step * u0);
        w1 = std::max(0.0, w1 + step * u1);
        w2 = std::max(0.0, w2 + step * u2);
        w3 = std::max(0.0, w3 + step * u3);

        base_ms_ = w0 / kScaleBase;
        ms_per_decode_token_ = w1 / kScaleDecode;
        ms_per_prefill_token_ = w2 / kScalePrefill;
        ms_per_kv_ktoken_ = w3 / kScaleKvKtok;

        ++total_samples_;
        if (d > 0) ++decode_samples_;
        if (p > 0) ++prefill_samples_;
        if (d > 0 && p > 0) ++mixed_samples_;
        last_observed_ms_ = y;
    }

    [[nodiscard]] double PredictStepMs(std::int32_t decode_tokens, std::int32_t prefill_tokens,
                                       std::int64_t total_kv_tokens) const noexcept {
        const std::int32_t d = decode_tokens > 0 ? decode_tokens : 0;
        const std::int32_t p = prefill_tokens > 0 ? prefill_tokens : 0;
        if (d == 0 && p == 0) return 0.0;
        const double kv_k =
            static_cast<double>(total_kv_tokens > 0 ? total_kv_tokens : std::int64_t{0}) / 1024.0;
        return base_ms_ + ms_per_decode_token_ * static_cast<double>(d) +
               ms_per_prefill_token_ * static_cast<double>(p) + ms_per_kv_ktoken_ * kv_k;
    }

    [[nodiscard]] bool IsCalibrated() const noexcept { return total_samples_ >= min_samples_; }
    [[nodiscard]] bool HasPrefillSamples() const noexcept { return prefill_samples_ > 0u; }
    [[nodiscard]] std::uint64_t total_samples() const noexcept { return total_samples_; }
    [[nodiscard]] std::uint64_t decode_samples() const noexcept { return decode_samples_; }
    [[nodiscard]] std::uint64_t prefill_samples() const noexcept { return prefill_samples_; }
    [[nodiscard]] std::uint64_t mixed_samples() const noexcept { return mixed_samples_; }
    [[nodiscard]] std::uint64_t min_samples() const noexcept { return min_samples_; }
    [[nodiscard]] double base_ms() const noexcept { return base_ms_; }
    [[nodiscard]] double ms_per_decode_token() const noexcept { return ms_per_decode_token_; }
    [[nodiscard]] double ms_per_prefill_token() const noexcept { return ms_per_prefill_token_; }
    [[nodiscard]] double ms_per_kv_ktoken() const noexcept { return ms_per_kv_ktoken_; }
    [[nodiscard]] double last_observed_ms() const noexcept { return last_observed_ms_; }

private:
    static constexpr double kScaleBase = 1.0;
    static constexpr double kScaleDecode = 4.0;
    static constexpr double kScalePrefill = 128.0;
    static constexpr double kScaleKvKtok = 8.0;

    double ema_alpha_ = 0.25;
    std::uint64_t min_samples_ = 3;
    double max_observation_ms_ = 600000.0;

    double base_ms_ = 0.0;
    double ms_per_decode_token_ = 0.0;
    double ms_per_prefill_token_ = 0.0;
    double ms_per_kv_ktoken_ = 0.0;

    std::uint64_t total_samples_ = 0;
    std::uint64_t decode_samples_ = 0;
    std::uint64_t prefill_samples_ = 0;
    std::uint64_t mixed_samples_ = 0;
    double last_observed_ms_ = 0.0;
};

struct BatchPlannerConfig {
    // Safe bounded mixed-batch prefill cap used before StepCostEstimator is
    // calibrated or when no active decoding slot carries an ITL objective.
    std::int32_t default_mixed_prefill_tokens = 256;
    // Hard upper cap on prefill tokens packed into a mixed decode+prefill step
    // (0 = bounded only by mixed_token_budget - decode_slots).
    std::int32_t max_mixed_prefill_tokens = 0;
    // Implicit ITL target (ms) applied to interactive/foreground decoding slots
    // without an explicit `target_itl_ms` (<= 0 disables).
    double default_foreground_itl_ms = 50.0;
    // Fraction of the tightest ITL target aimed for when sizing mixed prefill
    // chunks so step-time jitter stays below target (clamped to (0, 1]).
    double itl_safety_factor = 0.90;
    // Wait duration (ms) per priority-tier aging promotion for waiting slots.
    double aging_interval_ms = 250.0;
    // Wait threshold (ms) after which any waiting prefill (including background)
    // is promoted as starved and guaranteed forward progress.
    double starvation_threshold_ms = 500.0;
    // Tighter anti-starvation wait threshold (ms) for interactive/foreground prefills.
    double foreground_starvation_threshold_ms = 150.0;
    // Minimum prefill tokens guaranteed per starvation trigger so background and
    // queued prefills never stall permanently under tight ITL constraints.
    std::int32_t starvation_prefill_tokens = 32;
};

struct AdaptiveBudgetDecision {
    std::int32_t total_token_budget = 0;
    std::int32_t prefill_token_budget = 0;
    std::size_t decode_slots = 0;
    std::size_t prefill_slots = 0;
    std::size_t starved_prefill_slots = 0;
    std::int64_t total_kv_tokens = 0;
    std::optional<double> tightest_itl_ms = std::nullopt;
    double predicted_decode_ms = 0.0;
    double predicted_step_ms = 0.0;
    bool used_cost_estimator = false;
    bool used_fallback_default = false;
    bool starvation_floor_applied = false;
};

struct PendingQueueSlotView {
    sonder::inference::RequestPriority priority = sonder::inference::RequestPriority::subagent;
    std::int64_t enqueue_time_us = 0;
    std::optional<double> target_ttft_ms = std::nullopt;
    bool exclusive = false;
};

namespace detail {

[[nodiscard]] inline int BasePriorityTier(sonder::inference::RequestPriority p) noexcept {
    switch (p) {
        case sonder::inference::RequestPriority::interactive: return 0;
        case sonder::inference::RequestPriority::subagent: return 1;
        case sonder::inference::RequestPriority::background: return 2;
    }
    return 1;
}

[[nodiscard]] inline std::int64_t EffectiveKvContextTokens(const BatchSlotPlanView& s) noexcept {
    if (s.kv_context_tokens > 0) return s.kv_context_tokens;
    return s.pos > 0 ? s.pos : std::int64_t{0};
}

[[nodiscard]] inline std::int64_t SaturatingAddI64(std::int64_t a, std::int64_t b) noexcept {
    if (b <= 0) return a;
    if (a > std::numeric_limits<std::int64_t>::max() - b) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return a + b;
}

[[nodiscard]] inline double ElapsedMsSince(std::int64_t ref_us, std::int64_t now_us) noexcept {
    if (now_us <= 0) return 0.0;
    const std::int64_t safe_ref = ref_us > 0 ? ref_us : std::int64_t{0};
    if (now_us <= safe_ref) return 0.0;
    return static_cast<double>(now_us - safe_ref) / 1000.0;
}

[[nodiscard]] inline double SlotWaitSinceServiceMs(const BatchSlotPlanView& s, std::int64_t now_us) noexcept {
    if (now_us <= 0) return 0.0;
    const std::int64_t ref_us = s.last_service_time_us > 0 ? s.last_service_time_us : s.enqueue_time_us;
    return ElapsedMsSince(ref_us, now_us);
}

[[nodiscard]] inline std::optional<double> ValidPositiveMs(const std::optional<double>& v) noexcept {
    if (v.has_value() && std::isfinite(*v) && *v > 0.0) {
        return v;
    }
    return std::nullopt;
}

[[nodiscard]] inline double EffectiveStarvationThresholdMs(sonder::inference::RequestPriority priority,
                                                           const std::optional<double>& target_ttft_ms,
                                                           const BatchPlannerConfig& config) noexcept {
    double thresh = (std::isfinite(config.starvation_threshold_ms) && config.starvation_threshold_ms > 0.0)
                        ? config.starvation_threshold_ms
                        : 500.0;
    if (priority == sonder::inference::RequestPriority::interactive &&
        std::isfinite(config.foreground_starvation_threshold_ms) &&
        config.foreground_starvation_threshold_ms > 0.0) {
        thresh = std::min(thresh, config.foreground_starvation_threshold_ms);
    }
    if (const auto ttft = ValidPositiveMs(target_ttft_ms); ttft.has_value()) {
        thresh = std::min(thresh, std::max(1.0, (*ttft) * 0.5));
    }
    return thresh;
}

[[nodiscard]] inline bool IsSlotStarved(sonder::inference::RequestPriority priority,
                                        const std::optional<double>& target_ttft_ms, double wait_ms,
                                        std::int64_t now_us, const BatchPlannerConfig& config) noexcept {
    if (now_us <= 0) return false;
    return wait_ms >= EffectiveStarvationThresholdMs(priority, target_ttft_ms, config);
}

[[nodiscard]] inline int EffectiveSlotTier(sonder::inference::RequestPriority priority,
                                           const std::optional<double>& target_ttft_ms, std::int64_t enqueue_time_us,
                                           double wait_ms, std::int64_t now_us,
                                           const BatchPlannerConfig& config) noexcept {
    if (IsSlotStarved(priority, target_ttft_ms, wait_ms, now_us, config)) {
        return -1;
    }
    int tier = BasePriorityTier(priority);
    if (now_us > 0 && std::isfinite(config.aging_interval_ms) && config.aging_interval_ms > 0.0) {
        const double promotions = std::floor(wait_ms / config.aging_interval_ms);
        if (promotions >= 1.0) {
            const int promo_int = static_cast<int>(std::min(2.0, promotions));
            tier = std::max(0, tier - promo_int);
        }
    }
    if (now_us > 0) {
        if (const auto ttft = ValidPositiveMs(target_ttft_ms); ttft.has_value()) {
            const double total_wait_ms = ElapsedMsSince(enqueue_time_us, now_us);
            if (total_wait_ms >= *ttft) {
                tier = std::max(0, tier - 1);
            }
        }
    }
    return tier;
}

[[nodiscard]] inline std::optional<double> ComputeTtftSlackMs(const std::optional<double>& target_ttft_ms,
                                                              std::int64_t enqueue_time_us,
                                                              std::int64_t now_us) noexcept {
    const auto ttft = ValidPositiveMs(target_ttft_ms);
    if (!ttft.has_value()) return std::nullopt;
    const double elapsed = (now_us > 0) ? ElapsedMsSince(enqueue_time_us, now_us) : 0.0;
    return *ttft - elapsed;
}

struct CandidateOrder {
    std::size_t idx = 0;
    int tier = 1;
    bool has_ttft_slack = false;
    double ttft_slack_ms = 0.0;
    std::size_t rr_order = 0;
};

[[nodiscard]] inline bool CompareCandidateOrder(const CandidateOrder& a, const CandidateOrder& b) noexcept {
    if (a.tier != b.tier) return a.tier < b.tier;
    if (a.has_ttft_slack != b.has_ttft_slack) return a.has_ttft_slack;
    if (a.has_ttft_slack && a.ttft_slack_ms != b.ttft_slack_ms) {
        return a.ttft_slack_ms < b.ttft_slack_ms;
    }
    return a.rr_order < b.rr_order;
}

[[nodiscard]] inline PlannedBatchStep PlanContinuousBatchStepImpl(std::span<const BatchSlotPlanView> active_slots,
                                                                  std::int32_t token_budget, std::int32_t output_budget,
                                                                  std::size_t& rr_cursor, std::int64_t now_us,
                                                                  const BatchPlannerConfig& config,
                                                                  bool only_starved_prefill) {
    PlannedBatchStep step;
    if (active_slots.empty() || token_budget <= 0 || output_budget <= 0) {
        return step;
    }

    const std::size_t n = active_slots.size();
    const std::size_t start_idx = rr_cursor % n;
    step.items.reserve(n);

    // Inline stack scratch avoids hot-path heap allocations for the normal
    // n <= kMaxSlots (16) continuous-batching case.
    std::array<std::int32_t, kMaxSlots> inline_assigned{};
    std::array<std::uint8_t, kMaxSlots> inline_flags{};
    std::array<CandidateOrder, kMaxSlots> inline_candidates{};
    std::vector<std::int32_t> heap_assigned;
    std::vector<std::uint8_t> heap_flags;
    std::vector<CandidateOrder> heap_candidates;
    std::int32_t* assigned = inline_assigned.data();
    std::uint8_t* flags = inline_flags.data();
    CandidateOrder* candidates = inline_candidates.data();
    if (n > kMaxSlots) {
        heap_assigned.assign(n, 0);
        heap_flags.assign(n, 0u);
        heap_candidates.resize(n);
        assigned = heap_assigned.data();
        flags = heap_flags.data();
        candidates = heap_candidates.data();
    }

    // Pass 1: Pack 1 decode token per decoding slot (low inter-token latency).
    std::size_t decode_waiting = 0;
    for (const BatchSlotPlanView& s : active_slots) {
        if (!s.cancelled && !s.suspended && s.phase == BatchSlotPhase::kDecode) {
            ++decode_waiting;
        }
    }
    const auto decode_cap = static_cast<std::size_t>(std::min(token_budget, output_budget));
    if (decode_waiting > 0u) {
        if (decode_waiting <= decode_cap) {
            // All decoding slots fit: mark in canonical ascending order.
            for (std::size_t idx = 0; idx < n; ++idx) {
                const BatchSlotPlanView& s = active_slots[idx];
                if (!s.cancelled && !s.suspended && s.phase == BatchSlotPhase::kDecode) {
                    flags[idx] = 1u;
                }
            }
        } else {
            // Tight decode budget: rank by effective priority tier (with aging)
            // and tie-break by rr_cursor fairness, then emit in canonical order.
            std::size_t d_count = 0;
            for (std::size_t off = 0; off < n; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (s.cancelled || s.suspended || s.phase != BatchSlotPhase::kDecode) continue;
                const double wait_ms = SlotWaitSinceServiceMs(s, now_us);
                CandidateOrder c;
                c.idx = idx;
                c.tier = EffectiveSlotTier(s.priority, s.target_itl_ms, s.enqueue_time_us, wait_ms, now_us, config);
                c.has_ttft_slack = false;
                c.ttft_slack_ms = 0.0;
                c.rr_order = off;
                candidates[d_count++] = c;
            }
            std::sort(candidates, candidates + static_cast<std::ptrdiff_t>(d_count), CompareCandidateOrder);
            const std::size_t pick = std::min(d_count, decode_cap);
            for (std::size_t k = 0; k < pick; ++k) {
                flags[candidates[k].idx] = 1u;
            }
        }
        for (std::size_t idx = 0; idx < n; ++idx) {
            if (flags[idx] == 0u) continue;
            flags[idx] = 0u;
            const BatchSlotPlanView& s = active_slots[idx];
            PlannedBatchItem item;
            item.slot_index = s.slot_index;
            item.seq_id = s.seq_id;
            item.is_prefill = false;
            item.token_offset = 0;
            item.token_count = 1;
            item.pos_start = s.pos;
            item.request_logits = true;
            step.items.push_back(item);
            step.total_tokens += 1;
            step.total_outputs += 1;
            step.decode_slots += 1;
        }
    }

    // Pass 2: Pack priority-ordered & anti-starvation-aged chunked prefill
    // tokens into the remaining token budget.
    std::int32_t rem_budget = token_budget - step.total_tokens;
    if (rem_budget > 0) {
        std::size_t cand_count = 0;
        for (std::size_t off = 0; off < n; ++off) {
            const std::size_t idx = (start_idx + off) % n;
            const BatchSlotPlanView& s = active_slots[idx];
            if (s.cancelled || s.suspended || s.phase != BatchSlotPhase::kPrefill || s.prefill_offset >= s.prompt_len) continue;
            const double wait_ms = SlotWaitSinceServiceMs(s, now_us);
            const std::optional<double> slack = ComputeTtftSlackMs(s.target_ttft_ms, s.enqueue_time_us, now_us);
            CandidateOrder c;
            c.idx = idx;
            c.tier = EffectiveSlotTier(s.priority, s.target_ttft_ms, s.enqueue_time_us, wait_ms, now_us, config);
            c.has_ttft_slack = slack.has_value();
            c.ttft_slack_ms = slack.value_or(0.0);
            c.rr_order = off;
            candidates[cand_count++] = c;
        }

        if (cand_count > 0u) {
            std::sort(candidates, candidates + static_cast<std::ptrdiff_t>(cand_count), CompareCandidateOrder);

            auto allocate_for_slot = [&](std::size_t idx, std::size_t cap_sz, std::int32_t& budget_pool) {
                if (budget_pool <= 0 || cap_sz == 0u) return;
                const BatchSlotPlanView& s = active_slots[idx];
                const std::size_t total_rem = s.prompt_len - s.prefill_offset;
                const auto already = static_cast<std::size_t>(assigned[idx]);
                if (total_rem <= already) return;
                const std::size_t rem_prompt = total_rem - already;
                std::size_t take_sz = std::min({rem_prompt, cap_sz, static_cast<std::size_t>(budget_pool)});
                const bool finishes = (take_sz == rem_prompt);
                if (finishes && step.total_outputs >= output_budget) {
                    if (take_sz > 1u) {
                        --take_sz;
                    } else {
                        return;
                    }
                }
                const auto take = static_cast<std::int32_t>(take_sz);
                assigned[idx] += take;
                budget_pool -= take;
                rem_budget -= take;
                if (take_sz == rem_prompt) {
                    flags[idx] = 1u;
                    step.total_outputs += 1;
                }
            };

            auto allocate_group_fair = [&](std::size_t g_begin, std::size_t g_end, std::int32_t group_cap) {
                if (g_begin >= g_end || group_cap <= 0 || rem_budget <= 0) return;
                std::int32_t local_budget = std::min(group_cap, rem_budget);
                const std::size_t group_count = g_end - g_begin;
                const std::size_t fair_slice =
                    std::max<std::size_t>(1u, static_cast<std::size_t>(local_budget) / group_count);
                for (std::size_t k = g_begin; k < g_end && local_budget > 0; ++k) {
                    allocate_for_slot(candidates[k].idx, fair_slice, local_budget);
                }
                for (std::size_t k = g_begin; k < g_end && local_budget > 0; ++k) {
                    allocate_for_slot(candidates[k].idx, static_cast<std::size_t>(local_budget), local_budget);
                }
            };

            // Partition starved candidates (tier == -1) at the front.
            std::size_t starved_end = 0;
            while (starved_end < cand_count && candidates[starved_end].tier < 0) {
                ++starved_end;
            }

            if (starved_end > 0u) {
                bool has_non_starved_higher = false;
                for (std::size_t k = starved_end; k < cand_count; ++k) {
                    if (candidates[k].tier <= 1) {
                        has_non_starved_higher = true;
                        break;
                    }
                }
                if (only_starved_prefill || !has_non_starved_higher) {
                    allocate_group_fair(0u, starved_end, rem_budget);
                } else {
                    const std::int32_t per_starved = std::max<std::int32_t>(1, config.starvation_prefill_tokens);
                    const std::size_t cap_sz =
                        std::min<std::size_t>(static_cast<std::size_t>(rem_budget),
                                              starved_end * static_cast<std::size_t>(per_starved));
                    allocate_group_fair(0u, starved_end, static_cast<std::int32_t>(cap_sz));
                }
            }

            // Allocate across non-starved priority tiers (0 -> 1 -> 2) unless the
            // step budget was raised strictly as an anti-starvation floor.
            if (!only_starved_prefill && rem_budget > 0) {
                std::size_t g_begin = starved_end;
                while (g_begin < cand_count && rem_budget > 0) {
                    const int tier = candidates[g_begin].tier;
                    std::size_t g_end = g_begin + 1u;
                    while (g_end < cand_count && candidates[g_end].tier == tier) {
                        ++g_end;
                    }
                    allocate_group_fair(g_begin, g_end, rem_budget);
                    g_begin = g_end;
                }
            }

            // Leftover pass: if starved slots were capped to preserve room for
            // higher-priority prefills that finished early, return unused tokens
            // to remaining candidates in priority order so microbatch is utilized.
            if (!only_starved_prefill && rem_budget > 0) {
                std::int32_t leftover = rem_budget;
                for (std::size_t k = 0; k < cand_count && leftover > 0; ++k) {
                    allocate_for_slot(candidates[k].idx, static_cast<std::size_t>(leftover), leftover);
                }
            }

            // Emit prefill items in priority-tier order (starved -> foreground ->
            // normal -> background; intra-tier order preserved).
            for (std::size_t k = 0; k < cand_count; ++k) {
                const std::size_t idx = candidates[k].idx;
                if (assigned[idx] <= 0) continue;
                const BatchSlotPlanView& s = active_slots[idx];
                PlannedBatchItem item;
                item.slot_index = s.slot_index;
                item.seq_id = s.seq_id;
                item.is_prefill = true;
                item.token_offset = s.prefill_offset;
                item.token_count = assigned[idx];
                item.pos_start = s.pos;
                item.request_logits = (flags[idx] != 0u);
                step.items.push_back(item);
                step.total_tokens += assigned[idx];
                step.prefill_slots += 1;
            }
        }
    }

    rr_cursor = (start_idx + 1u) % n;
    return step;
}

}  // namespace detail

// Selects the best FREE (`busy[i] == 0`) KV slot for `prompt` and `session_key`:
//   1. Exact non-empty `session_key` match among free slots (longest LCP, then MRU)
//   2. Longest common token prefix (> 0) among free slots, tie-broken by MRU
//   3. Least recently used free slot (lowest `last_used`)
// Returns std::nullopt when every slot is currently busy.
[[nodiscard]] inline std::optional<SlotSelection> SelectAvailablePrefixSlot(std::span<const PrefixSlotMeta> slots,
                                                                            std::span<const std::uint8_t> busy,
                                                                            std::span<const Token> prompt,
                                                                            std::string_view session_key) noexcept {
    if (slots.empty() || slots.size() != busy.size()) return std::nullopt;
    const std::size_t max_reuse = prompt.size() > 1 ? prompt.size() - 1 : 0;

    // 1. Exact non-empty session_key match among free slots (best LCP, then MRU).
    if (!session_key.empty()) {
        bool found_sess = false;
        std::size_t best_sess_idx = 0;
        std::size_t best_sess_lcp = 0;
        std::uint64_t best_sess_used = 0;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (busy[i] != 0u) continue;
            if (slots[i].session_key == session_key) {
                const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
                if (!found_sess || lcp > best_sess_lcp ||
                    (lcp == best_sess_lcp && slots[i].last_used > best_sess_used)) {
                    found_sess = true;
                    best_sess_idx = i;
                    best_sess_lcp = lcp;
                    best_sess_used = slots[i].last_used;
                }
            }
        }
        if (found_sess) return SlotSelection{best_sess_idx, best_sess_lcp};
    }

    // 2. Longest common prefix (> 0) among free slots, tie-broken by MRU.
    bool found_lcp = false;
    std::size_t best_idx = 0;
    std::size_t best_lcp = 0;
    std::uint64_t best_used = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (busy[i] != 0u) continue;
        const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
        if (lcp > best_lcp || (lcp > 0 && lcp == best_lcp && slots[i].last_used > best_used)) {
            found_lcp = true;
            best_idx = i;
            best_lcp = lcp;
            best_used = slots[i].last_used;
        }
    }
    if (found_lcp) return SlotSelection{best_idx, best_lcp};

    // 3. Fallback to least recently used free slot.
    bool found_free = false;
    std::size_t lru_idx = 0;
    std::uint64_t min_used = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (busy[i] != 0u) continue;
        if (!found_free || slots[i].last_used < min_used) {
            found_free = true;
            lru_idx = i;
            min_used = slots[i].last_used;
        }
    }
    if (!found_free) return std::nullopt;
    return SlotSelection{lru_idx, 0};
}

// Picks the next pending queue index to admit when free KV slots exist.
// Respects exclusive-request barrier semantics at index 0 (caller drains active
// slots before admitting an exclusive request); otherwise selects the highest
// effective-priority non-exclusive request ahead of any exclusive barrier,
// with anti-starvation aging, TTFT urgency, and FIFO tie-breaking.
[[nodiscard]] inline std::optional<std::size_t> SelectNextPendingRequest(
    std::span<const PendingQueueSlotView> queue, bool has_active_slots, std::int64_t now_us,
    const BatchPlannerConfig& config = {}) noexcept {
    if (queue.empty()) return std::nullopt;
    if (queue[0].exclusive) {
        if (has_active_slots) return std::nullopt;
        return std::size_t{0};
    }

    std::size_t best_idx = 0;
    detail::CandidateOrder best{};
    bool have_best = false;

    for (std::size_t i = 0; i < queue.size(); ++i) {
        if (queue[i].exclusive) break;  // do not starve a queued exclusive barrier
        const double wait_ms = (now_us > 0) ? detail::ElapsedMsSince(queue[i].enqueue_time_us, now_us) : 0.0;
        const std::optional<double> slack =
            detail::ComputeTtftSlackMs(queue[i].target_ttft_ms, queue[i].enqueue_time_us, now_us);
        detail::CandidateOrder c;
        c.idx = i;
        c.tier = detail::EffectiveSlotTier(queue[i].priority, queue[i].target_ttft_ms, queue[i].enqueue_time_us,
                                           wait_ms, now_us, config);
        c.has_ttft_slack = slack.has_value();
        c.ttft_slack_ms = slack.value_or(0.0);
        c.rr_order = i;  // FIFO tie-breaker
        if (!have_best || detail::CompareCandidateOrder(c, best)) {
            have_best = true;
            best = c;
            best_idx = i;
        }
    }
    if (!have_best) return std::nullopt;
    return best_idx;
}

// Preserves the zero-overhead single-request fast path (full speculative
// decoding + GPU backend sampling) whenever exactly one non-exclusive request
// is active and no other request is queued.
[[nodiscard]] inline bool CanUseSingleRequestFastPath(std::size_t active_count, std::size_t pending_count,
                                                      bool exclusive_request) noexcept {
    return active_count == 1u && pending_count == 0u && !exclusive_request;
}

// Computes the adaptive total and prefill token budgets for one batch step
// given active slots, measured StepCostEstimator, and latency/aging policy:
//   - Zero-decode (prefill-only) steps use the full `prefill_only_token_budget`
//     (e.g. `n_batch`) without artificial mixed-batch throttling.
//   - Mixed decode+prefill steps size the prefill chunk from the tightest
//     active decoding ITL target and measured step cost once calibrated.
//   - Falls back to safe bounded `config.default_mixed_prefill_tokens` before
//     enough step-cost samples exist (or when no decoding slot has an ITL target).
//   - Applies an anti-starvation floor whenever any queued prefill ages past
//     its starvation threshold so background work always makes steady progress.
[[nodiscard]] inline AdaptiveBudgetDecision ComputeAdaptivePrefillBudget(
    std::span<const BatchSlotPlanView> active_slots, std::int32_t mixed_token_budget,
    std::int32_t prefill_only_token_budget, std::int32_t output_budget, std::int64_t now_us,
    const StepCostEstimator& estimator, const BatchPlannerConfig& config = {}) noexcept {
    AdaptiveBudgetDecision decision;
    if (active_slots.empty() || output_budget <= 0) {
        return decision;
    }

    std::size_t decode_waiting = 0;
    std::size_t prefill_waiting = 0;
    std::size_t starved_prefills = 0;
    std::size_t total_remaining_prefill = 0;
    std::size_t starved_remaining_prefill = 0;
    std::int64_t decode_kv_tokens = 0;
    std::int64_t total_active_kv_tokens = 0;
    std::optional<double> tightest_itl;

    const bool have_default_fg_itl =
        std::isfinite(config.default_foreground_itl_ms) && config.default_foreground_itl_ms > 0.0;

    for (const BatchSlotPlanView& s : active_slots) {
        if (s.cancelled || s.suspended) continue;
        if (s.phase == BatchSlotPhase::kDecode) {
            ++decode_waiting;
            const std::int64_t kv = detail::EffectiveKvContextTokens(s);
            decode_kv_tokens = detail::SaturatingAddI64(decode_kv_tokens, kv);
            total_active_kv_tokens = detail::SaturatingAddI64(total_active_kv_tokens, kv);

            std::optional<double> slot_itl = detail::ValidPositiveMs(s.target_itl_ms);
            if (!slot_itl.has_value() && s.priority == sonder::inference::RequestPriority::interactive &&
                have_default_fg_itl) {
                slot_itl = config.default_foreground_itl_ms;
            }
            if (slot_itl.has_value()) {
                if (!tightest_itl.has_value() || *slot_itl < *tightest_itl) {
                    tightest_itl = *slot_itl;
                }
            }
        } else if (s.phase == BatchSlotPhase::kPrefill && s.prefill_offset < s.prompt_len) {
            ++prefill_waiting;
            const std::size_t rem = s.prompt_len - s.prefill_offset;
            if (total_remaining_prefill <= std::numeric_limits<std::size_t>::max() - rem) {
                total_remaining_prefill += rem;
            } else {
                total_remaining_prefill = std::numeric_limits<std::size_t>::max();
            }
            const std::int64_t kv = detail::EffectiveKvContextTokens(s);
            total_active_kv_tokens = detail::SaturatingAddI64(total_active_kv_tokens, kv);

            const double wait_ms = detail::SlotWaitSinceServiceMs(s, now_us);
            if (detail::IsSlotStarved(s.priority, s.target_ttft_ms, wait_ms, now_us, config)) {
                ++starved_prefills;
                if (starved_remaining_prefill <= std::numeric_limits<std::size_t>::max() - rem) {
                    starved_remaining_prefill += rem;
                } else {
                    starved_remaining_prefill = std::numeric_limits<std::size_t>::max();
                }
            }
        }
    }

    decision.decode_slots = decode_waiting;
    decision.prefill_slots = prefill_waiting;
    decision.starved_prefill_slots = starved_prefills;
    decision.total_kv_tokens = total_active_kv_tokens;
    decision.tightest_itl_ms = tightest_itl;

    // Pure-prefill wave (zero active decode slots): use full configured batch /
    // microbatch throughput without artificial mixed-step throttling.
    if (decode_waiting == 0u) {
        if (prefill_only_token_budget <= 0 || prefill_waiting == 0u) {
            return decision;
        }
        decision.prefill_token_budget = prefill_only_token_budget;
        decision.total_token_budget = prefill_only_token_budget;
        decision.used_cost_estimator = estimator.IsCalibrated();
        if (estimator.IsCalibrated()) {
            const auto planned_prefill = static_cast<std::int32_t>(
                std::min<std::size_t>(total_remaining_prefill, static_cast<std::size_t>(prefill_only_token_budget)));
            decision.predicted_step_ms = estimator.PredictStepMs(0, planned_prefill, total_active_kv_tokens);
        }
        return decision;
    }

    if (mixed_token_budget <= 0) {
        return decision;
    }

    const auto decode_cap = static_cast<std::size_t>(std::min(mixed_token_budget, output_budget));
    const std::size_t planned_decode = std::min(decode_waiting, decode_cap);
    const auto planned_decode_i32 = static_cast<std::int32_t>(planned_decode);
    std::int32_t available_prefill_room = std::max<std::int32_t>(0, mixed_token_budget - planned_decode_i32);
    if (config.max_mixed_prefill_tokens > 0) {
        available_prefill_room = std::min(available_prefill_room, config.max_mixed_prefill_tokens);
    }

    const std::int32_t fallback_cap =
        config.default_mixed_prefill_tokens > 0 ? config.default_mixed_prefill_tokens : 256;

    std::int32_t prefill_budget = 0;
    if (prefill_waiting > 0u && available_prefill_room > 0) {
        if (!tightest_itl.has_value() || !estimator.IsCalibrated()) {
            prefill_budget = std::min(available_prefill_room, fallback_cap);
            decision.used_fallback_default = true;
        } else {
            decision.used_cost_estimator = true;
            const double safety = (std::isfinite(config.itl_safety_factor) && config.itl_safety_factor > 0.0)
                                      ? std::min(1.0, config.itl_safety_factor)
                                      : 1.0;
            const double effective_itl = (*tightest_itl) * safety;
            const double decode_only_ms = estimator.PredictStepMs(planned_decode_i32, 0, decode_kv_tokens);
            const double mixed_fixed_ms = estimator.PredictStepMs(planned_decode_i32, 0, total_active_kv_tokens);
            decision.predicted_decode_ms = decode_only_ms;

            if (mixed_fixed_ms >= effective_itl) {
                prefill_budget = 0;
            } else if (!estimator.HasPrefillSamples()) {
                // Calibrated on pure-decode steps only; use safe bounded fallback
                // until at least one prefill step calibrates marginal prefill cost.
                prefill_budget = std::min(available_prefill_room, fallback_cap);
                decision.used_fallback_default = true;
            } else {
                const double headroom_ms = effective_itl - mixed_fixed_ms;
                const double c_pref = estimator.ms_per_prefill_token();
                if (c_pref <= 1e-9) {
                    prefill_budget = available_prefill_room;
                } else {
                    const double raw_toks = std::floor(headroom_ms / c_pref);
                    if (!std::isfinite(raw_toks) || raw_toks <= 0.0) {
                        prefill_budget = 0;
                    } else if (raw_toks >= static_cast<double>(available_prefill_room)) {
                        prefill_budget = available_prefill_room;
                    } else {
                        prefill_budget = static_cast<std::int32_t>(raw_toks);
                    }
                }
            }
        }

        // Anti-starvation guarantee: if any queued prefill has waited past its
        // starvation threshold, grant a bounded forward-progress floor even
        // when tight foreground ITL would otherwise shrink prefill below it.
        if (starved_prefills > 0u) {
            const std::int32_t floor_cap = std::max<std::int32_t>(1, config.starvation_prefill_tokens);
            const std::size_t desired_sz =
                std::min<std::size_t>(starved_remaining_prefill, static_cast<std::size_t>(floor_cap));
            const std::int32_t target_floor = std::min(available_prefill_room, static_cast<std::int32_t>(desired_sz));
            if (prefill_budget < target_floor) {
                prefill_budget = target_floor;
                decision.starvation_floor_applied = true;
            }
        }
    }

    decision.prefill_token_budget = prefill_budget;
    decision.total_token_budget = planned_decode_i32 + prefill_budget;
    if (estimator.IsCalibrated()) {
        const auto planned_prefill = static_cast<std::int32_t>(
            std::min<std::size_t>(total_remaining_prefill, static_cast<std::size_t>(prefill_budget)));
        const std::int64_t step_kv = planned_prefill > 0 ? total_active_kv_tokens : decode_kv_tokens;
        decision.predicted_step_ms = estimator.PredictStepMs(planned_decode_i32, planned_prefill, step_kv);
    }
    return decision;
}

// Plans ONE lockstep `llama_decode` step packing:
//   1. Up to 1 decode token per active `kDecode` slot (priority for low ITL;
//      priority + aging + round-robin if decode slots exceed min(token_budget,
//      output_budget), emitted in canonical ascending index order for CUDA
//      graph stability)
//   2. Chunked prefill tokens from active `kPrefill` slots into the remaining
//      `token_budget`, ordered by effective priority tier (starved -> foreground
//      -> normal -> background) with fair intra-tier sharing via `rr_cursor`.
// Enforces `total_tokens <= token_budget` and `total_outputs <= output_budget`.
[[nodiscard]] inline PlannedBatchStep PlanContinuousBatchStep(std::span<const BatchSlotPlanView> active_slots,
                                                              std::int32_t token_budget, std::int32_t output_budget,
                                                              std::size_t& rr_cursor, std::int64_t now_us,
                                                              const BatchPlannerConfig& config) {
    return detail::PlanContinuousBatchStepImpl(active_slots, token_budget, output_budget, rr_cursor, now_us, config,
                                               /*only_starved_prefill=*/false);
}

// Backward-compatible 4-argument overload (no timestamps; default planner config).
[[nodiscard]] inline PlannedBatchStep PlanContinuousBatchStep(std::span<const BatchSlotPlanView> active_slots,
                                                              std::int32_t token_budget, std::int32_t output_budget,
                                                              std::size_t& rr_cursor) {
    return detail::PlanContinuousBatchStepImpl(active_slots, token_budget, output_budget, rr_cursor, /*now_us=*/0,
                                               BatchPlannerConfig{}, /*only_starved_prefill=*/false);
}

// Combined latency-target-aware physical batch planner: computes the adaptive
// step budget from `StepCostEstimator` + active ITL objectives + anti-starvation
// aging, then packs the resulting lockstep `PlannedBatchStep`.
[[nodiscard]] inline PlannedBatchStep PlanLatencyAwareBatchStep(
    std::span<const BatchSlotPlanView> active_slots, std::int32_t mixed_token_budget,
    std::int32_t prefill_only_token_budget, std::int32_t output_budget, std::size_t& rr_cursor, std::int64_t now_us,
    const StepCostEstimator& estimator, const BatchPlannerConfig& config = {},
    AdaptiveBudgetDecision* out_decision = nullptr) {
    const AdaptiveBudgetDecision decision = ComputeAdaptivePrefillBudget(
        active_slots, mixed_token_budget, prefill_only_token_budget, output_budget, now_us, estimator, config);
    if (out_decision != nullptr) {
        *out_decision = decision;
    }
    if (decision.total_token_budget <= 0) {
        return PlannedBatchStep{};
    }
    return detail::PlanContinuousBatchStepImpl(active_slots, decision.total_token_budget, output_budget, rr_cursor,
                                               now_us, config, decision.starvation_floor_applied);
}

}  // namespace sonder::backends::llamacpp
