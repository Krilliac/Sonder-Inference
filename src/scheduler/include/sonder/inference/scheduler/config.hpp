// Tunable scheduling policy. Defaults are conservative starting points, not
// benchmarked values (see docs/BENCHMARK_PLAN.md).
#pragma once

#include "sonder/inference/scheduler/types.hpp"

namespace sonder::inference::scheduler {

struct SchedulerConfig {
    // ---- Per-step budgets (continuous batching) ----
    /// Maximum tokens (prefill + decode) processed in one engine step.
    TokenCount max_step_tokens = 2048;
    /// Maximum sequences participating in one engine step.
    std::uint32_t max_step_sequences = 64;
    /// Maximum sequences resident (admitted, holding KV) at once.
    std::uint32_t max_running_sequences = 128;

    // ---- Chunked prefill (Sarathi-style) ----
    /// Largest prefill chunk for a single sequence in one step.
    TokenCount prefill_chunk_tokens = 512;
    /// Upper bound on total prefill tokens in one step. Together with the
    /// decode-first ordering this reserves capacity for decode (TTFT vs TBT).
    TokenCount max_step_prefill_tokens = 1024;

    // ---- Admission control ----
    /// Blocks kept free at admission time as headroom for decode growth, so
    /// running sequences are not immediately preempted (no OOM-as-control-flow).
    BlockCount admission_watermark_blocks = 4;
    /// How many waiting requests past a blocked head may be considered for
    /// admission in one step (bounded head-of-line bypass). Bypass is disabled
    /// while the blocked head is starving.
    std::uint32_t admission_lookahead = 8;
    /// Allow urgent waiting requests to preempt lower-class running ones.
    bool enable_priority_preemption = true;
    /// Minimum rank gap (victim rank - candidate rank) for priority preemption.
    int priority_preemption_min_rank_gap = 2;

    // ---- Fairness / starvation ----
    /// Every `aging_interval_us` of queue wait improves the effective rank by 1.
    TimeUs aging_interval_us = 2'000'000;
    /// A waiting request older than this is treated as top priority and
    /// cannot be bypassed by lookahead admission.
    TimeUs starvation_threshold_us = 10'000'000;

    // ---- Preemption / no-progress guards ----
    /// A request preempted more than this many times fails (RequeueLimit).
    std::uint32_t max_requeue_count = 8;
    /// Swap (instead of recompute) when enabled and the victim holds at least
    /// this many tokens of KV. Swap is a decision stub: the engine performs
    /// the actual movement.
    bool enable_swap = false;
    TokenCount swap_min_tokens = 1024;

    // ---- Statistics ----
    /// Inter-token gaps retained per workload class for SchedulerStats (the
    /// most recent ones; a ring). Bounds memory in a long-running engine.
    std::uint32_t latency_sample_window = 4096;
};

}  // namespace sonder::inference::scheduler
