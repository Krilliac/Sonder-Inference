// Scheduler telemetry (docs/SCHEDULER.md "Metrics").
#pragma once

#include "sonder/inference/scheduler/types.hpp"

#include <cstdint>
#include <vector>

namespace sonder::inference::scheduler {

/// Latency summary over a set of samples (microseconds).
struct LatencySummary {
    std::uint64_t count = 0;
    TimeUs min = 0;
    TimeUs max = 0;
    double mean = 0.0;
    TimeUs p50 = 0;
    TimeUs p95 = 0;
    TimeUs p99 = 0;

    [[nodiscard]] static LatencySummary from(std::vector<TimeUs> samples);
};

/// Lifecycle timestamps for a single request (kept after completion).
struct RequestTimeline {
    RequestId id = 0;
    WorkloadClass workload = WorkloadClass::ImplementationWorker;
    RequestState final_state = RequestState::Created;
    FailureReason failure = FailureReason::None;
    TimeUs arrival = 0;
    TimeUs first_admitted = -1;
    TimeUs first_token = -1;
    TimeUs finished = -1;
    TokenCount generated_tokens = 0;
    std::uint32_t preemptions = 0;
    TimeUs max_inter_token_gap = 0;
    TimeUs total_queue_time = 0;  // includes requeue waits after preemption

    [[nodiscard]] TimeUs ttft() const noexcept {
        return first_token < 0 ? -1 : first_token - arrival;
    }
};

struct ClassStats {
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t failed = 0;
    std::uint64_t rejected = 0;
    std::uint64_t preemptions = 0;
    std::uint64_t generated_tokens = 0;
    LatencySummary queue_time;
    LatencySummary ttft;
    LatencySummary inter_token;
    LatencySummary end_to_end;
};

/// Aggregated snapshot. Cheap counters are maintained live; latency
/// summaries are computed on demand by Scheduler::stats().
struct SchedulerStats {
    TimeUs now = 0;
    std::uint64_t steps = 0;
    std::uint64_t empty_steps = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t decode_tokens = 0;
    std::uint64_t sequences_scheduled = 0;  // sum over steps
    std::uint64_t budget_tokens = 0;        // sum of max_step_tokens over non-empty steps
    std::uint64_t preemptions_kv_pressure = 0;
    std::uint64_t preemptions_priority = 0;
    std::uint64_t preemptions_recompute = 0;
    std::uint64_t preemptions_swap = 0;
    std::uint64_t recomputed_tokens = 0;    // KV discarded by recompute preemption
    std::uint64_t swapped_tokens = 0;       // KV moved out by swap preemption
    std::uint64_t starvation_promotions = 0;
    std::uint64_t lookahead_admissions = 0; // admitted past a blocked head
    std::uint64_t admission_kv_blocked = 0; // steps where head was KV-blocked
    std::size_t active_sequences = 0;       // admitted and holding KV
    std::size_t queued_sequences = 0;       // waiting (incl. preempted)
    TimeUs max_starvation_age = 0;          // oldest queue age ever observed
    TimeUs current_oldest_queue_age = 0;

    PerClass<ClassStats> per_class{};

    [[nodiscard]] double batch_token_occupancy() const noexcept {
        return budget_tokens == 0
                   ? 0.0
                   : static_cast<double>(prefill_tokens + decode_tokens) /
                         static_cast<double>(budget_tokens);
    }
    [[nodiscard]] double prefill_share() const noexcept {
        const auto total = prefill_tokens + decode_tokens;
        return total == 0 ? 0.0
                          : static_cast<double>(prefill_tokens) / static_cast<double>(total);
    }
    [[nodiscard]] double mean_batch_sequences() const noexcept {
        const auto busy = steps - empty_steps;
        return busy == 0 ? 0.0
                         : static_cast<double>(sequences_scheduled) / static_cast<double>(busy);
    }
};

}  // namespace sonder::inference::scheduler
