// Iteration-level scheduler: priority request queue, continuous batching step
// planner, chunked prefill, KV admission control, preemption and fairness.
//
// Usage per engine step:
//     StepPlan plan = scheduler.plan_step();
//     ... backend executes plan.work (outside this module) ...
//     clock advances by the step duration
//     scheduler.complete_step(plan, outcome);
//
// Exactly one plan may be in flight at a time.
#pragma once

#include "sonder/inference/scheduler/clock.hpp"
#include "sonder/inference/scheduler/config.hpp"
#include "sonder/inference/scheduler/kv_capacity.hpp"
#include "sonder/inference/scheduler/stats.hpp"
#include "sonder/inference/scheduler/types.hpp"

#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

namespace sonder::inference::scheduler {

/// One sequence's work in a step.
struct ScheduledWork {
    RequestId id = 0;
    Phase phase = Phase::Decode;
    /// Tokens to process: a prefill chunk, or 1 for decode.
    TokenCount num_tokens = 0;
    /// Tokens already holding KV before this work (position offset).
    TokenCount context_offset = 0;
    /// True when this chunk finishes prefill (and so yields the first token).
    bool completes_prefill = false;
};

struct PreemptionEvent {
    RequestId id = 0;
    PreemptionReason reason = PreemptionReason::KvPressure;
    PreemptionMode mode = PreemptionMode::Recompute;
    /// KV tokens discarded (recompute) or moved (swap).
    TokenCount kv_tokens = 0;
    /// Request whose progress required the preemption (0 if none).
    RequestId beneficiary = 0;
    /// True if the request hit the requeue limit and failed instead.
    bool failed = false;
};

struct StepPlan {
    std::uint64_t step_index = 0;
    TimeUs planned_at = 0;
    std::vector<ScheduledWork> work;
    std::vector<RequestId> admitted;
    std::vector<PreemptionEvent> preempted;
    std::vector<RequestId> failed;
    TokenCount prefill_tokens = 0;
    TokenCount decode_tokens = 0;

    [[nodiscard]] TokenCount total_tokens() const noexcept { return prefill_tokens + decode_tokens; }
    [[nodiscard]] bool empty() const noexcept { return work.empty(); }
};

/// Backend feedback for a completed step.
struct StepOutcome {
    /// Sequences that emitted an end-of-sequence token this step.
    std::vector<RequestId> finished_early;
    /// Sequences that did not execute their planned work this step (the
    /// backend had not produced the token yet when the engine closed the
    /// step). Their work is not accounted: no prefill progress, no KV growth,
    /// no token. The next plan schedules them again.
    std::vector<RequestId> stalled;
};

struct SubmitResult {
    bool accepted = false;
    FailureReason reason = FailureReason::None;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, const Clock& clock, KvCapacity& kv);

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Enqueue a request. Rejected requests are recorded but never scheduled.
    SubmitResult submit(RequestSpec spec);

    /// Cancel a queued or running request, releasing its KV immediately.
    /// Returns false for unknown, terminal, or non-cancellable requests.
    bool cancel(RequestId id);

    /// Drop a terminal request's bookkeeping (sequence, timeline). Engines
    /// that run indefinitely call this once they no longer need the request,
    /// so memory stays bounded by live requests. Returns false for unknown or
    /// non-terminal requests. Counters in stats() are unaffected; per-request
    /// latency summaries then cover only requests not yet forgotten.
    bool forget(RequestId id);

    /// Requests currently tracked (live plus terminal, not yet forgotten).
    [[nodiscard]] std::size_t tracked() const noexcept { return seqs_.size(); }

    /// Plan the next engine step. Throws std::logic_error if a plan is in flight.
    [[nodiscard]] StepPlan plan_step();

    /// Apply the results of `plan`. Throws std::logic_error on mismatch.
    void complete_step(const StepPlan& plan, const StepOutcome& outcome = {});

    /// True when nothing is waiting or running.
    [[nodiscard]] bool idle() const noexcept { return waiting_.empty() && running_.empty(); }
    [[nodiscard]] bool plan_in_flight() const noexcept { return in_flight_.has_value(); }

    [[nodiscard]] std::optional<RequestState> state(RequestId id) const;
    [[nodiscard]] std::optional<RequestTimeline> timeline(RequestId id) const;
    [[nodiscard]] std::vector<RequestTimeline> timelines() const;

    /// Effective queue rank of a waiting request (lower = sooner), including
    /// aging. Returns nullopt if the request is not waiting.
    [[nodiscard]] std::optional<int> effective_rank(RequestId id) const;

    /// Waiting requests in the order admission would consider them.
    [[nodiscard]] std::vector<RequestId> queue_order() const;
    [[nodiscard]] const std::vector<RequestId>& running() const noexcept { return running_; }

    /// Pure decision stub: how a victim would be preempted right now.
    [[nodiscard]] PreemptionMode decide_preemption_mode(TokenCount kv_tokens) const noexcept;

    [[nodiscard]] SchedulerStats stats() const;
    [[nodiscard]] const SchedulerConfig& config() const noexcept { return config_; }

private:
    struct Sequence {
        RequestSpec spec;
        RequestState state = RequestState::Created;
        int base_rank = 0;
        std::uint64_t submit_order = 0;
        TimeUs enqueued_at = 0;
        TokenCount prefill_target = 0;  // tokens to (re)compute before decode
        TokenCount prefill_done = 0;
        TokenCount kv_tokens = 0;       // tokens whose KV is materialised
        TokenCount generated = 0;
        bool swapped = false;
        std::uint32_t requeues = 0;
        TimeUs last_token_time = -1;
        bool starving_counted = false;
        RequestTimeline tl;
    };

    struct Budget {
        TokenCount tokens = 0;
        TokenCount prefill_tokens = 0;
        std::uint32_t sequences = 0;
    };

    [[nodiscard]] int running_rank(const Sequence& s) const noexcept;
    [[nodiscard]] bool running_better(const Sequence& a, const Sequence& b) const noexcept;
    [[nodiscard]] int waiting_rank(const Sequence& s, TimeUs now) const noexcept;
    [[nodiscard]] bool is_starving(const Sequence& s, TimeUs now) const noexcept;
    [[nodiscard]] std::vector<RequestId> sorted_waiting(TimeUs now) const;
    [[nodiscard]] std::vector<RequestId> sorted_running_best_first() const;

    void add_work(StepPlan& plan, Budget& budget, Sequence& s, Phase phase, TokenCount n);
    void remove_work(StepPlan& plan, Budget& budget, RequestId id);
    void preempt(StepPlan& plan, Budget& budget, RequestId victim, PreemptionReason reason,
                 RequestId beneficiary, TimeUs now);
    void finish(Sequence& s, RequestState terminal, FailureReason why, TimeUs now);
    void erase_from(std::vector<RequestId>& v, RequestId id);
    void emit_token(Sequence& s, TimeUs now);
    bool try_priority_preemption(StepPlan& plan, Budget& budget, const Sequence& cand,
                                 BlockCount need, TimeUs now);
    void schedule_decode(StepPlan& plan, Budget& budget, TimeUs now);
    void schedule_prefill(StepPlan& plan, Budget& budget);
    void admit(StepPlan& plan, Budget& budget, TimeUs now);

    SchedulerConfig config_;
    const Clock& clock_;
    KvCapacity& kv_;

    std::map<RequestId, Sequence> seqs_;
    std::vector<RequestId> waiting_;
    std::vector<RequestId> running_;
    std::unordered_set<std::string> active_fingerprints_;
    std::optional<std::uint64_t> in_flight_;
    std::uint64_t next_submit_order_ = 0;

    // Live counters.
    SchedulerStats counters_;
    // Most recent inter-token gaps per class (ring of latency_sample_window).
    PerClass<std::vector<TimeUs>> inter_token_samples_{};
    PerClass<std::size_t> inter_token_next_{};
};

}  // namespace sonder::inference::scheduler
