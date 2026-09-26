// Deterministic discrete-step simulator: drives a Scheduler against a fixed
// KV pool and a synthetic step-cost model. No backend is executed.
#pragma once

#include "sonder/inference/scheduler/scheduler.hpp"

#include <functional>
#include <vector>

namespace sonder::inference::scheduler {

/// Synthetic engine cost: step_us = overhead + prefill tokens * cost + decode seqs * cost.
struct StepCostModel {
    TimeUs step_overhead_us = 2'000;
    TimeUs prefill_us_per_token = 20;
    TimeUs decode_us_per_sequence = 150;

    [[nodiscard]] TimeUs step_time(const StepPlan& plan) const noexcept;
};

struct Arrival {
    TimeUs at = 0;
    RequestSpec spec;
};

struct CancellationEvent {
    TimeUs at = 0;
    RequestId id = 0;
};

struct SimulationResult {
    SchedulerStats stats;
    std::vector<RequestTimeline> timelines;
    TimeUs makespan = 0;
    std::uint64_t steps = 0;
    bool hit_step_limit = false;
    // Invariant observations (checked every step).
    TokenCount max_step_tokens_seen = 0;
    std::uint32_t max_step_sequences_seen = 0;
    TokenCount max_prefill_chunk_seen = 0;
    std::size_t max_running_seen = 0;
    BlockCount peak_kv_blocks = 0;
    std::uint64_t budget_violations = 0;
    std::uint64_t kv_overcommits = 0;
    std::uint64_t rejected_submissions = 0;

    [[nodiscard]] double generated_tokens_per_second() const noexcept;
};

class Simulator {
public:
    Simulator(SchedulerConfig config, BlockCount kv_blocks, TokenCount block_size,
              StepCostModel cost = {});

    /// Optional per-step observer (called after planning, before completion).
    std::function<void(const StepPlan&, const Scheduler&)> on_step;
    /// Optional EOS injection: return true to finish `id` early this step.
    std::function<bool(RequestId, const Scheduler&)> early_stop;

    SimulationResult run(std::vector<Arrival> arrivals,
                         std::vector<CancellationEvent> cancellations = {},
                         std::uint64_t max_steps = 1'000'000);

private:
    SchedulerConfig config_;
    BlockCount kv_blocks_;
    TokenCount block_size_;
    StepCostModel cost_;
};

}  // namespace sonder::inference::scheduler
