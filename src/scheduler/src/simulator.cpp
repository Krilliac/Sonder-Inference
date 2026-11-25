#include "sonder/inference/scheduler/simulator.hpp"

#include <algorithm>
#include <limits>
#include <optional>

namespace sonder::inference::scheduler {

TimeUs StepCostModel::step_time(const StepPlan& plan) const noexcept {
    if (plan.empty()) return 0;
    std::size_t decode_seqs = 0;
    for (const auto& w : plan.work) {
        if (w.phase == Phase::Decode) ++decode_seqs;
    }
    constexpr TimeUs kMax = std::numeric_limits<TimeUs>::max();
    auto sat_add = [](TimeUs a, TimeUs b) noexcept -> TimeUs {
        if (b <= 0) return a;
        return a > kMax - b ? kMax : a + b;
    };
    auto sat_mul = [](TimeUs a, std::uint64_t count) noexcept -> TimeUs {
        if (a <= 0 || count == 0) return 0;
        const auto ua = static_cast<std::uint64_t>(a);
        const auto umax = static_cast<std::uint64_t>(kMax);
        if (ua > umax / count) return kMax;
        return static_cast<TimeUs>(ua * count);
    };
    TimeUs total = std::max<TimeUs>(0, step_overhead_us);
    total = sat_add(total, sat_mul(prefill_us_per_token, plan.prefill_tokens));
    total = sat_add(total, sat_mul(decode_us_per_sequence, static_cast<std::uint64_t>(decode_seqs)));
    return total;
}

double SimulationResult::generated_tokens_per_second() const noexcept {
    if (makespan <= 0) return 0.0;
    std::uint64_t tokens = 0;
    for (const auto& t : timelines) tokens += t.generated_tokens;
    return static_cast<double>(tokens) * 1e6 / static_cast<double>(makespan);
}

Simulator::Simulator(SchedulerConfig config, BlockCount kv_blocks, TokenCount block_size,
                     StepCostModel cost)
    : config_(config), kv_blocks_(kv_blocks), block_size_(block_size), cost_(cost) {}

SimulationResult Simulator::run(std::vector<Arrival> arrivals,
                                std::vector<CancellationEvent> cancellations,
                                std::uint64_t max_steps) {
    std::stable_sort(arrivals.begin(), arrivals.end(),
                     [](const Arrival& a, const Arrival& b) { return a.at < b.at; });
    std::stable_sort(cancellations.begin(), cancellations.end(),
                     [](const CancellationEvent& a, const CancellationEvent& b) {
                         return a.at < b.at;
                     });

    const TimeUs start = arrivals.empty() ? 0 : arrivals.front().at;
    SimClock clock(start);
    FixedKvCapacity kv(kv_blocks_, block_size_);
    Scheduler sched(config_, clock, kv);
    SimulationResult result;

    std::size_t next_arrival = 0;
    std::size_t next_cancel = 0;

    while (result.steps < max_steps) {
        while (next_arrival < arrivals.size() && arrivals[next_arrival].at <= clock.now()) {
            if (!sched.submit(arrivals[next_arrival].spec).accepted) result.rejected_submissions++;
            ++next_arrival;
        }
        while (next_cancel < cancellations.size() &&
               cancellations[next_cancel].at <= clock.now()) {
            (void)sched.cancel(cancellations[next_cancel].id);
            ++next_cancel;
        }
        if (sched.idle() && next_arrival >= arrivals.size()) break;

        StepPlan plan = sched.plan_step();
        result.steps++;

        // Invariants.
        const auto seqs = static_cast<std::uint32_t>(plan.work.size());
        result.max_step_tokens_seen = std::max(result.max_step_tokens_seen, plan.total_tokens());
        result.max_step_sequences_seen = std::max(result.max_step_sequences_seen, seqs);
        result.max_running_seen = std::max(result.max_running_seen, sched.running().size());
        bool violated = plan.total_tokens() > config_.max_step_tokens ||
                        plan.prefill_tokens > config_.max_step_prefill_tokens ||
                        seqs > config_.max_step_sequences ||
                        sched.running().size() > config_.max_running_sequences;
        for (const auto& w : plan.work) {
            if (w.phase == Phase::Prefill) {
                result.max_prefill_chunk_seen = std::max(result.max_prefill_chunk_seen, w.num_tokens);
                if (w.num_tokens > config_.prefill_chunk_tokens) violated = true;
            }
        }
        if (violated) result.budget_violations++;
        if (kv.used_blocks() > kv.total_blocks()) result.kv_overcommits++;
        if (on_step) on_step(plan, sched);

        StepOutcome outcome;
        if (early_stop) {
            for (const auto& w : plan.work) {
                if (early_stop(w.id, sched)) outcome.finished_early.push_back(w.id);
            }
        }

        if (plan.empty()) {
            sched.complete_step(plan, outcome);
            if (!plan.admitted.empty() || !plan.preempted.empty() || !plan.failed.empty()) {
                continue;  // state advanced (e.g. swap-in resume); plan again
            }
            // Nothing runnable: jump to the next external event.
            std::optional<TimeUs> next;
            if (next_arrival < arrivals.size()) next = arrivals[next_arrival].at;
            if (next_cancel < cancellations.size() &&
                (!next || cancellations[next_cancel].at < *next)) {
                next = cancellations[next_cancel].at;
            }
            if (!next) break;  // stuck with no future events (should not happen)
            clock.advance_to(*next);
            continue;
        }
        clock.advance(cost_.step_time(plan));
        sched.complete_step(plan, outcome);
    }

    result.hit_step_limit = result.steps >= max_steps && !sched.idle();
    result.stats = sched.stats();
    result.timelines = sched.timelines();
    result.makespan = clock.now() - start;
    result.peak_kv_blocks = kv.peak_used_blocks();
    return result;
}

}  // namespace sonder::inference::scheduler
