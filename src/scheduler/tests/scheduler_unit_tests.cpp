// Unit tests for the scheduler policy (queue, batching, admission,
// preemption, fairness, telemetry).
#include <doctest/doctest.h>

#include "sonder/inference/scheduler/scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sonder::inference::scheduler;

namespace {

RequestSpec spec(RequestId id, WorkloadClass cls, TokenCount prompt, TokenCount max_new) {
    RequestSpec s;
    s.id = id;
    s.workload = cls;
    s.task_id = "task-" + std::to_string(id);
    s.prompt_tokens = prompt;
    s.max_new_tokens = max_new;
    return s;
}

StepPlan step(Scheduler& sched, SimClock& clock, TimeUs dt = 1000, StepOutcome outcome = {}) {
    StepPlan plan = sched.plan_step();
    clock.advance(dt);
    sched.complete_step(plan, outcome);
    return plan;
}

const ScheduledWork* find_work(const StepPlan& plan, RequestId id) {
    auto it = std::find_if(plan.work.begin(), plan.work.end(),
                           [&](const ScheduledWork& w) { return w.id == id; });
    return it == plan.work.end() ? nullptr : &*it;
}

bool was_preempted(const StepPlan& plan, RequestId id) {
    return std::any_of(plan.preempted.begin(), plan.preempted.end(),
                       [&](const PreemptionEvent& e) { return e.id == id; });
}

}  // namespace

TEST_CASE("queue_orders_by_workload_class") {
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched({}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 32, 4)).accepted);
    CHECK(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 32, 4)).accepted);
    CHECK(sched.submit(spec(3, WorkloadClass::InteractiveUser, 32, 4)).accepted);
    CHECK(sched.submit(spec(4, WorkloadClass::ImplementationWorker, 32, 4)).accepted);
    const auto order = sched.queue_order();
    CHECK(order == (std::vector<RequestId>{3, 2, 4, 1}));
}

TEST_CASE("priority_override_applies") {
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched({}, clock, kv);
    auto bg = spec(1, WorkloadClass::Maintenance, 32, 4);
    bg.priority_override = -1;
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 32, 4)).accepted);
    CHECK(sched.submit(bg).accepted);
    CHECK(sched.queue_order().front() == 1u);
    CHECK_EQ(*sched.effective_rank(1), -1);
}

TEST_CASE("aging_improves_effective_rank") {
    SchedulerConfig cfg;
    cfg.aging_interval_us = 1000;
    cfg.starvation_threshold_us = 1'000'000;
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 32, 4)).accepted);
    CHECK_EQ(*sched.effective_rank(1), 5);
    clock.advance(2500);
    CHECK_EQ(*sched.effective_rank(1), 3);
    // A freshly arrived research worker (rank 4) now sorts behind the aged request.
    CHECK(sched.submit(spec(2, WorkloadClass::ResearchWorker, 32, 4)).accepted);
    CHECK(sched.queue_order() == (std::vector<RequestId>{1, 2}));
}

TEST_CASE("starving_request_jumps_to_front") {
    SchedulerConfig cfg;
    cfg.aging_interval_us = 1'000'000'000;  // isolate the starvation guard
    cfg.starvation_threshold_us = 5000;
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::Maintenance, 32, 4)).accepted);
    clock.advance(6000);
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 32, 4)).accepted);
    CHECK(sched.queue_order().front() == 1u);
    auto plan = step(sched, clock);
    CHECK_EQ(sched.stats().starvation_promotions, 1u);
    CHECK_EQ(sched.stats().max_starvation_age, 6000);
    CHECK(plan.admitted.front() == 1u);
}

TEST_CASE("chunked_prefill_splits_long_prompt") {
    SchedulerConfig cfg;
    cfg.prefill_chunk_tokens = 512;
    cfg.max_step_prefill_tokens = 1024;
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ImplementationWorker, 1300, 3)).accepted);

    auto p1 = step(sched, clock);
    CHECK_EQ(p1.work.size(), 1u);
    CHECK_EQ(p1.work[0].num_tokens, 512u);
    CHECK_EQ(p1.work[0].context_offset, 0u);
    CHECK(!p1.work[0].completes_prefill);
    auto p2 = step(sched, clock);
    CHECK_EQ(p2.work[0].num_tokens, 512u);
    CHECK_EQ(p2.work[0].context_offset, 512u);
    auto p3 = step(sched, clock);
    CHECK_EQ(p3.work[0].num_tokens, 276u);
    CHECK(p3.work[0].completes_prefill);
    CHECK(sched.state(1) == RequestState::Decode);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 1u);
    CHECK_EQ(sched.timeline(1)->first_token, 3000);

    auto p4 = step(sched, clock);
    CHECK(p4.work[0].phase == Phase::Decode);
    CHECK_EQ(p4.work[0].num_tokens, 1u);
    step(sched, clock);
    CHECK(sched.state(1) == RequestState::Completed);
    CHECK(sched.idle());
    CHECK_EQ(kv.used_blocks(), 0u);
}

TEST_CASE("decode_first_and_prefill_capped") {
    SchedulerConfig cfg;
    cfg.max_step_tokens = 600;
    cfg.max_step_prefill_tokens = 256;
    cfg.prefill_chunk_tokens = 512;
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 16, 50)).accepted);
    step(sched, clock);  // prefill completes -> decoding
    CHECK(sched.state(1) == RequestState::Decode);
    // A huge interactive prefill must not displace the running decode.
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 4000, 5)).accepted);
    auto plan = step(sched, clock);
    const auto* d = find_work(plan, 1);
    const auto* p = find_work(plan, 2);
    CHECK((d != nullptr && d->phase == Phase::Decode));
    CHECK((p != nullptr && p->phase == Phase::Prefill));
    CHECK((p != nullptr && p->num_tokens == 256u));
    CHECK(plan.total_tokens() <= cfg.max_step_tokens);
}

TEST_CASE("step_budgets_respected_under_load") {
    SchedulerConfig cfg;
    cfg.max_step_tokens = 300;
    cfg.max_step_prefill_tokens = 200;
    cfg.max_step_sequences = 5;
    cfg.prefill_chunk_tokens = 64;
    SimClock clock;
    FixedKvCapacity kv(10'000, 16);
    Scheduler sched(cfg, clock, kv);
    for (RequestId i = 1; i <= 40; ++i) {
        CHECK(sched.submit(spec(i, static_cast<WorkloadClass>(i % 7), 100 + 10 * static_cast<TokenCount>(i), 8)).accepted);
    }
    int guard = 0;
    while (!sched.idle() && guard++ < 10'000) {
        auto plan = step(sched, clock);
        CHECK(plan.total_tokens() <= cfg.max_step_tokens);
        CHECK(plan.prefill_tokens <= cfg.max_step_prefill_tokens);
        CHECK(plan.work.size() <= cfg.max_step_sequences);
        for (const auto& w : plan.work) CHECK(w.num_tokens <= cfg.prefill_chunk_tokens);
    }
    CHECK(sched.idle());
    const auto st = sched.stats();
    std::uint64_t completed = 0;
    for (const auto& c : st.per_class) completed += c.completed;
    CHECK_EQ(completed, 40u);
}

TEST_CASE("admission_blocked_by_kv_and_watermark") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 2;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ImplementationWorker, 64, 30)).accepted);  // 4 blocks
    step(sched, clock);
    CHECK_EQ(kv.used_blocks(), 4u);
    // Needs 5 blocks; 6 free but watermark requires 5 + 2.
    CHECK(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 80, 4)).accepted);
    auto plan = step(sched, clock);
    CHECK(plan.admitted.empty());
    CHECK(sched.state(2) == RequestState::WaitingAdmission);
    CHECK_EQ(sched.stats().admission_kv_blocked, 1u);
}

TEST_CASE("watermark_waived_when_nothing_running") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 8;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ImplementationWorker, 150, 10)).accepted);
    auto plan = step(sched, clock);
    CHECK(plan.admitted.size() == 1u);
}

TEST_CASE("lookahead_admits_past_blocked_head") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ResearchWorker, 96, 40)).accepted);  // 6 blocks
    step(sched, clock);
    CHECK(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 96, 4)).accepted);  // needs 6, blocked
    CHECK(sched.submit(spec(3, WorkloadClass::ImplementationWorker, 32, 4)).accepted);  // needs 2, fits
    auto plan = step(sched, clock);
    CHECK(plan.admitted == (std::vector<RequestId>{3}));
    CHECK_EQ(sched.stats().lookahead_admissions, 1u);
}

TEST_CASE("interactive_head_is_never_bypassed") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ResearchWorker, 96, 40)).accepted);
    step(sched, clock);
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 96, 4)).accepted);
    CHECK(sched.submit(spec(3, WorkloadClass::ImplementationWorker, 32, 4)).accepted);
    auto plan = step(sched, clock);
    CHECK(plan.admitted.empty());
}

TEST_CASE("kv_pressure_preempts_lowest_priority") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 32, 16)).accepted);      // 2 blocks
    CHECK(sched.submit(spec(2, WorkloadClass::BackgroundIndexing, 32, 16)).accepted);   // 2 blocks
    step(sched, clock);  // both prefill, each emits first token
    CHECK_EQ(kv.free_blocks(), 0u);
    // Next decode for either needs a 3rd block per sequence; background yields.
    auto plan = step(sched, clock);
    CHECK(was_preempted(plan, 2));
    CHECK(!was_preempted(plan, 1));
    CHECK(find_work(plan, 1) != nullptr);
    CHECK(sched.state(2) == RequestState::Preempted);
    CHECK(plan.preempted[0].reason == PreemptionReason::KvPressure);
    CHECK(plan.preempted[0].mode == PreemptionMode::Recompute);
    CHECK_EQ(plan.preempted[0].kv_tokens, 32u);
    CHECK_EQ(plan.preempted[0].beneficiary, 1u);
    const auto st = sched.stats();
    CHECK_EQ(st.preemptions_kv_pressure, 1u);
    CHECK_EQ(st.recomputed_tokens, 32u);
    // Everything still finishes: the preempted request recomputes later.
    int guard = 0;
    while (!sched.idle() && guard++ < 1000) step(sched, clock);
    CHECK(sched.state(1) == RequestState::Completed);
    CHECK(sched.state(2) == RequestState::Completed);
    CHECK_EQ(sched.timeline(2)->generated_tokens, 16u);
}

TEST_CASE("swap_decision_stub_and_resume") {
    SchedulerConfig cfg;
    cfg.enable_swap = true;
    cfg.swap_min_tokens = 32;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.decide_preemption_mode(31) == PreemptionMode::Recompute);
    CHECK(sched.decide_preemption_mode(32) == PreemptionMode::Swap);

    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 32, 3)).accepted);
    CHECK(sched.submit(spec(2, WorkloadClass::BackgroundIndexing, 32, 3)).accepted);
    step(sched, clock);
    auto plan = step(sched, clock);
    CHECK(was_preempted(plan, 2));
    CHECK(plan.preempted[0].mode == PreemptionMode::Swap);
    CHECK_EQ(sched.stats().swapped_tokens, 32u);
    // After 1 completes, 2 swaps back in straight to decode (no re-prefill).
    int guard = 0;
    bool resumed_in_decode = false;
    while (!sched.idle() && guard++ < 100) {
        auto p = step(sched, clock);
        if (const auto* w = find_work(p, 2)) {
            CHECK(w->phase == Phase::Decode);
            resumed_in_decode = true;
        }
    }
    CHECK(resumed_in_decode);
    CHECK(sched.state(2) == RequestState::Completed);
    CHECK_EQ(sched.stats().prefill_tokens, 64u);
}

TEST_CASE("priority_preemption_admits_interactive") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    SimClock clock;
    FixedKvCapacity kv(8, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 60, 4)).accepted);  // 4 blocks
    CHECK(sched.submit(spec(2, WorkloadClass::Maintenance, 60, 4)).accepted);         // 4 blocks
    step(sched, clock);
    CHECK_EQ(kv.free_blocks(), 0u);
    CHECK(sched.submit(spec(3, WorkloadClass::InteractiveUser, 48, 4)).accepted);
    auto plan = step(sched, clock);
    CHECK(was_preempted(plan, 2));   // maintenance (worst) yields first
    CHECK(!was_preempted(plan, 1));  // one victim was enough
    CHECK(std::find(plan.admitted.begin(), plan.admitted.end(), RequestId{3}) != plan.admitted.end());
    CHECK(plan.preempted[0].reason == PreemptionReason::PriorityAdmission);
    CHECK_EQ(plan.preempted[0].beneficiary, 3u);
    CHECK(find_work(plan, 2) == nullptr);  // victim's decode was withdrawn
    CHECK_EQ(sched.stats().preemptions_priority, 1u);
}

TEST_CASE("priority_preemption_never_evicts_a_request_admitted_in_the_same_pass") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.aging_interval_us = 1'000'000'000;  // isolate the starvation guard
    cfg.starvation_threshold_us = 5000;
    SimClock clock;
    FixedKvCapacity kv(8, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 60, 4)).accepted);  // 4 blocks, will starve
    clock.advance(6000);
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 80, 4)).accepted);  // 5 blocks, does not fit after 1
    auto plan = step(sched, clock);
    CHECK(std::find(plan.admitted.begin(), plan.admitted.end(), RequestId{1}) != plan.admitted.end());
    CHECK(!was_preempted(plan, 1));
    CHECK(plan.preempted.empty());
    CHECK(sched.state(1) != RequestState::Preempted);
}

TEST_CASE("priority_preemption_respects_rank_gap") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.priority_preemption_min_rank_gap = 2;
    SimClock clock;
    FixedKvCapacity kv(8, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::CriticVerification, 120, 8)).accepted);  // rank 2, 8 blocks
    step(sched, clock);
    CHECK(sched.submit(spec(2, WorkloadClass::OwnerOrchestrator, 48, 4)).accepted);  // rank 1
    auto plan = step(sched, clock);
    CHECK(plan.preempted.empty());
    CHECK(plan.admitted.empty());
}

TEST_CASE("requeue_limit_fails_request") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    cfg.max_requeue_count = 0;
    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 32, 16)).accepted);
    CHECK(sched.submit(spec(2, WorkloadClass::BackgroundIndexing, 32, 16)).accepted);
    step(sched, clock);
    auto plan = step(sched, clock);
    CHECK(plan.failed == (std::vector<RequestId>{2}));
    CHECK(plan.preempted[0].failed);
    CHECK(sched.state(2) == RequestState::Failed);
    CHECK(sched.timeline(2)->failure == FailureReason::RequeueLimit);
    CHECK_EQ(sched.stats().per_class[index_of(WorkloadClass::BackgroundIndexing)].failed, 1u);
}

TEST_CASE("cancel_releases_kv_and_ignores_inflight_work") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::ImplementationWorker, 64, 10)).accepted);
    CHECK(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 64, 10)).accepted);
    auto plan = sched.plan_step();
    CHECK(kv.blocks_held(1) > 0u);
    CHECK(sched.cancel(1));
    CHECK_EQ(kv.blocks_held(1), 0u);
    clock.advance(1000);
    sched.complete_step(plan);
    CHECK(sched.state(1) == RequestState::Cancelled);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 0u);
    CHECK_EQ(sched.timeline(2)->generated_tokens, 1u);
    CHECK(!sched.cancel(1));   // already terminal
    CHECK(!sched.cancel(99));  // unknown
    CHECK(sched.submit(spec(3, WorkloadClass::ImplementationWorker, 64, 10)).accepted);
    CHECK(sched.cancel(3));  // queued request
    CHECK(sched.state(3) == RequestState::Cancelled);
}

TEST_CASE("non_cancellable_request_refused") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, clock, kv);
    auto s = spec(1, WorkloadClass::Maintenance, 16, 2);
    s.cancellable = false;
    CHECK(sched.submit(s).accepted);
    CHECK(!sched.cancel(1));
    CHECK(sched.state(1) == RequestState::WaitingAdmission);
}

TEST_CASE("duplicate_sequence_guard") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, clock, kv);
    auto a = spec(1, WorkloadClass::ImplementationWorker, 16, 1);
    a.sequence_fingerprint = "model@rev:prompt#abc";
    auto b = a;
    b.id = 2;
    auto c = a;
    c.id = 3;
    c.allow_duplicate = true;
    CHECK(sched.submit(a).accepted);
    const auto r = sched.submit(b);
    CHECK(!r.accepted);
    CHECK(r.reason == FailureReason::DuplicateSequence);
    CHECK(sched.submit(c).accepted);  // explicitly requested duplicate
    while (!sched.idle()) step(sched, clock);
    auto d = a;
    d.id = 4;
    CHECK(sched.submit(d).accepted);  // fingerprint freed after completion
}

TEST_CASE("invalid_and_oversized_requests_rejected") {
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched({}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 1)).accepted);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 1)).reason == FailureReason::DuplicateId);
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 0, 1)).reason == FailureReason::InvalidRequest);
    CHECK(sched.submit(spec(3, WorkloadClass::InteractiveUser, 16, 0)).reason == FailureReason::InvalidRequest);
    CHECK(sched.submit(spec(4, WorkloadClass::InteractiveUser, 150, 20)).reason == FailureReason::NeverFits);
    CHECK_EQ(sched.stats().per_class[0].rejected, 4u);
    CHECK_EQ(sched.stats().per_class[0].submitted, 5u);
}

TEST_CASE("plan_complete_protocol_enforced") {
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched({}, clock, kv);
    auto plan = sched.plan_step();
    bool threw = false;
    try {
        (void)sched.plan_step();
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    sched.complete_step(plan);
    threw = false;
    try {
        sched.complete_step(plan);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(sched.stats().empty_steps, 1u);
}

TEST_CASE("invalid_config_rejected") {
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    SchedulerConfig cfg;
    cfg.max_step_tokens = 0;
    bool threw = false;
    try {
        Scheduler s(cfg, clock, kv);
        (void)s;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("early_stop_completes_sequence") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 100)).accepted);
    step(sched, clock);
    StepOutcome eos;
    eos.finished_early.push_back(1);
    step(sched, clock, 1000, eos);
    CHECK(sched.state(1) == RequestState::Completed);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 2u);
    CHECK_EQ(kv.used_blocks(), 0u);
}

TEST_CASE("stats_track_latency_and_occupancy") {
    SchedulerConfig cfg;
    cfg.max_step_tokens = 100;
    SimClock clock(10'000);
    FixedKvCapacity kv(100, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 20, 4)).accepted);
    clock.advance(500);  // queue wait before first plan
    step(sched, clock, 1000);  // prefill + first token at 11'500
    step(sched, clock, 200);
    step(sched, clock, 300);
    step(sched, clock, 400);
    const auto st = sched.stats();
    const auto& c = st.per_class[0];
    CHECK_EQ(c.completed, 1u);
    CHECK_EQ(c.generated_tokens, 4u);
    CHECK_EQ(c.queue_time.max, 500);
    CHECK_EQ(c.ttft.max, 1500);
    CHECK_EQ(c.inter_token.count, 3u);
    CHECK_EQ(c.inter_token.min, 200);
    CHECK_EQ(c.inter_token.max, 400);
    CHECK_EQ(c.end_to_end.max, 2400);
    CHECK_EQ(st.prefill_tokens, 20u);
    CHECK_EQ(st.decode_tokens, 3u);
    CHECK_EQ(st.steps, 4u);
    CHECK((st.batch_token_occupancy() > 0.05 && st.batch_token_occupancy() < 0.07));
    CHECK_EQ(sched.timeline(1)->max_inter_token_gap, 400);
}

TEST_CASE("latency_summary_percentiles") {
    std::vector<TimeUs> v;
    for (TimeUs i = 100; i >= 1; --i) v.push_back(i);
    const auto s = LatencySummary::from(v);
    CHECK_EQ(s.count, 100u);
    CHECK_EQ(s.min, 1);
    CHECK_EQ(s.max, 100);
    CHECK_EQ(s.p50, 50);
    CHECK_EQ(s.p95, 95);
    CHECK_EQ(s.p99, 99);
    CHECK((s.mean > 50.49 && s.mean < 50.51));
    CHECK_EQ(LatencySummary::from({}).count, 0u);
}

TEST_CASE("sim_clock_is_monotonic_and_manual") {
    SimClock c(5);
    CHECK_EQ(c.now(), 5);
    c.advance(10);
    CHECK_EQ(c.now(), 15);
    c.advance(-3);
    CHECK_EQ(c.now(), 15);
    c.advance_to(12);
    CHECK_EQ(c.now(), 15);
    c.advance_to(40);
    CHECK_EQ(c.now(), 40);
}

TEST_CASE("fixed_kv_capacity_all_or_nothing") {
    FixedKvCapacity kv(10, 16);
    CHECK(kv.try_reserve(1, 6));
    CHECK(!kv.try_reserve(2, 5));
    CHECK_EQ(kv.blocks_held(2), 0u);
    CHECK(kv.try_reserve(2, 4));
    CHECK_EQ(kv.free_blocks(), 0u);
    kv.release(1);
    CHECK_EQ(kv.free_blocks(), 6u);
    CHECK_EQ(kv.peak_used_blocks(), 10u);
    CHECK_EQ(blocks_for_tokens(33, 16), 3u);
    CHECK_EQ(blocks_for_tokens(32, 16), 2u);
}

TEST_CASE("forget_drops_terminal_requests_only") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched(SchedulerConfig{}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 2)).accepted);
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 16, 50)).accepted);
    CHECK_EQ(sched.tracked(), 2u);
    CHECK_FALSE(sched.forget(1));  // still running
    step(sched, clock);
    step(sched, clock);
    REQUIRE(sched.state(1) == RequestState::Completed);
    CHECK(sched.forget(1));
    CHECK_FALSE(sched.forget(1));  // already gone
    CHECK_FALSE(sched.state(1).has_value());
    CHECK_EQ(sched.tracked(), 1u);
    CHECK(sched.cancel(2));
    CHECK(sched.forget(2));
    CHECK_EQ(sched.tracked(), 0u);
    CHECK_FALSE(sched.forget(99));
    // Counters survive forgetting.
    CHECK_EQ(sched.stats().per_class[index_of(WorkloadClass::InteractiveUser)].completed, 1u);
    CHECK(sched.idle());
}

TEST_CASE("inter_token_samples_are_bounded_by_the_window") {
    SchedulerConfig cfg;
    cfg.latency_sample_window = 8;
    SimClock clock;
    FixedKvCapacity kv(1000, 16);
    Scheduler sched(cfg, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 100)).accepted);
    step(sched, clock, 1000);  // prefill + first token
    for (TimeUs i = 1; i < 100; ++i) {
        step(sched, clock, i);  // gaps 1..99
    }
    REQUIRE(sched.state(1) == RequestState::Completed);
    // stats() returns a snapshot by value: keep it alive, then refer into it.
    const SchedulerStats stats = sched.stats();
    const auto& c = stats.per_class[index_of(WorkloadClass::InteractiveUser)];
    CHECK_EQ(c.inter_token.count, 8u);
    CHECK_EQ(c.inter_token.min, 92);  // only the most recent gaps are kept
    CHECK_EQ(c.inter_token.max, 99);
}

TEST_CASE("stalled_work_is_not_accounted_and_is_planned_again") {
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched(SchedulerConfig{}, clock, kv);
    CHECK(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 4)).accepted);
    StepOutcome stalled;
    stalled.stalled.push_back(1);
    const StepPlan first = step(sched, clock, 1000, stalled);
    const ScheduledWork* w = find_work(first, 1);
    REQUIRE(w != nullptr);
    CHECK(w->completes_prefill);
    CHECK(sched.state(1) == RequestState::Prefill);  // prefill not accounted
    CHECK_EQ(sched.timeline(1)->generated_tokens, 0u);
    const StepPlan again = step(sched, clock);
    w = find_work(again, 1);
    REQUIRE(w != nullptr);
    CHECK(w->phase == Phase::Prefill);
    CHECK_EQ(w->context_offset, 0u);
    CHECK(w->completes_prefill);
    CHECK(sched.state(1) == RequestState::Decode);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 1u);
    const StepPlan decode = step(sched, clock, 1000, stalled);
    w = find_work(decode, 1);
    REQUIRE(w != nullptr);
    CHECK(w->phase == Phase::Decode);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 1u);  // stalled decode: no token
    const StepPlan retry = step(sched, clock);
    CHECK_EQ(find_work(retry, 1)->context_offset, w->context_offset);
    CHECK_EQ(sched.timeline(1)->generated_tokens, 2u);
}

TEST_CASE("admission_watermark_overflow_does_not_bypass_kv_guard") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = std::numeric_limits<BlockCount>::max();
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);

    REQUIRE(sched.submit(spec(1, WorkloadClass::ImplementationWorker, 64, 10)).accepted);
    step(sched, clock);  // admitted when running_ was empty (watermark waived), holds 4 blocks

    REQUIRE(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 32, 4)).accepted);  // needs 2 blocks
    const StepPlan plan = step(sched, clock);
    CHECK(plan.admitted.empty());
    CHECK(sched.state(2) == RequestState::WaitingAdmission);
}

TEST_CASE("priority_preemption_waives_watermark_when_emptying_running") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 4;
    cfg.enable_priority_preemption = true;
    SimClock clock;
    FixedKvCapacity kv(8, 16);
    Scheduler sched(cfg, clock, kv);

    REQUIRE(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 118, 10)).accepted);  // 8 blocks
    step(sched, clock);
    REQUIRE_EQ(kv.free_blocks(), 0u);

    REQUIRE(sched.submit(spec(2, WorkloadClass::InteractiveUser, 96, 4)).accepted);  // 6 blocks
    const StepPlan plan = step(sched, clock);
    CHECK(was_preempted(plan, 1));
    CHECK(plan.admitted == (std::vector<RequestId>{2}));
    CHECK_EQ(sched.stats().preemptions_priority, 1u);
}

TEST_CASE("priority_preemption_displaces_lower_class_at_max_running_sequences") {
    SchedulerConfig cfg;
    cfg.max_running_sequences = 1;
    cfg.max_step_sequences = 1;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = true;
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler sched(cfg, clock, kv);

    REQUIRE(sched.submit(spec(1, WorkloadClass::Maintenance, 32, 10)).accepted);
    step(sched, clock);
    REQUIRE(sched.state(1) == RequestState::Decode);

    REQUIRE(sched.submit(spec(2, WorkloadClass::InteractiveUser, 32, 4)).accepted);
    const StepPlan plan = step(sched, clock);
    CHECK(was_preempted(plan, 1));
    CHECK(plan.admitted == (std::vector<RequestId>{2}));
    CHECK(find_work(plan, 2) != nullptr);
    CHECK_EQ(sched.stats().preemptions_priority, 1u);
}

TEST_CASE("extreme_priority_overrides_do_not_overflow_aging_or_preemption") {
    SchedulerConfig cfg;
    cfg.aging_interval_us = 1000;
    cfg.starvation_threshold_us = 5000;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = true;
    cfg.priority_preemption_min_rank_gap = 2;
    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);

    RequestSpec ultra = spec(1, WorkloadClass::InteractiveUser, 32, 4);
    ultra.priority_override = std::numeric_limits<int>::min();
    RequestSpec normal = spec(2, WorkloadClass::Maintenance, 32, 4);
    REQUIRE(sched.submit(ultra).accepted);
    REQUIRE(sched.submit(normal).accepted);

    clock.advance(2000);  // aging boost applied: must not underflow INT_MIN
    REQUIRE(sched.effective_rank(1).has_value());
    CHECK_EQ(*sched.effective_rank(1), std::numeric_limits<int>::min());
    CHECK(sched.queue_order() == (std::vector<RequestId>{1, 2}));

    // Even when starving, INT_MIN must not be demoted behind a less urgent override.
    clock.advance(10'000);
    RequestSpec less_urgent = spec(3, WorkloadClass::InteractiveUser, 32, 4);
    less_urgent.priority_override = std::numeric_limits<int>::min() + 1;
    REQUIRE(sched.submit(less_urgent).accepted);
    CHECK_EQ(*sched.effective_rank(1), std::numeric_limits<int>::min());
    CHECK(sched.queue_order().front() == 1u);

    // Admit req 1 (2 blocks) and req 3 (2 blocks), filling KV.
    // Wait: submit a separate scheduler to isolate rank-gap preemption overflow cleanly.
    SimClock clock2;
    FixedKvCapacity kv2(4, 16);
    Scheduler sched2(cfg, clock2, kv2);
    RequestSpec running_ultra = spec(10, WorkloadClass::InteractiveUser, 60, 4);
    running_ultra.priority_override = std::numeric_limits<int>::min();
    REQUIRE(sched2.submit(running_ultra).accepted);
    step(sched2, clock2);  // req 10 holds all 4 blocks

    // A low-priority request must NEVER preempt INT_MIN via signed subtraction wrap.
    REQUIRE(sched2.submit(spec(11, WorkloadClass::Maintenance, 32, 4)).accepted);
    const StepPlan no_preempt = step(sched2, clock2);
    CHECK(no_preempt.preempted.empty());
    CHECK(no_preempt.admitted.empty());

    // Conversely, INT_MIN MUST preempt a running Maintenance request without overflow.
    SimClock clock3;
    FixedKvCapacity kv3(4, 16);
    Scheduler sched3(cfg, clock3, kv3);
    REQUIRE(sched3.submit(spec(20, WorkloadClass::Maintenance, 60, 4)).accepted);
    step(sched3, clock3);  // req 20 holds all 4 blocks
    RequestSpec waiting_ultra = spec(21, WorkloadClass::InteractiveUser, 32, 4);
    waiting_ultra.priority_override = std::numeric_limits<int>::min();
    REQUIRE(sched3.submit(waiting_ultra).accepted);
    const StepPlan did_preempt = step(sched3, clock3);
    CHECK(was_preempted(did_preempt, 20));
    CHECK(did_preempt.admitted == (std::vector<RequestId>{21}));
}

TEST_CASE("admission_lookahead_bounds_admitted_bypasses_past_blocked_head") {
    SchedulerConfig cfg;
    cfg.admission_lookahead = 2;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    SimClock clock;
    FixedKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);

    REQUIRE(sched.submit(spec(1, WorkloadClass::ResearchWorker, 80, 20)).accepted);  // 5 blocks
    step(sched, clock);
    REQUIRE_EQ(kv.free_blocks(), 5u);

    // Blocked head needs 6 blocks; five 1-block requests sit behind it.
    REQUIRE(sched.submit(spec(2, WorkloadClass::ImplementationWorker, 96, 4)).accepted);
    for (RequestId id = 3; id <= 7; ++id) {
        REQUIRE(sched.submit(spec(id, WorkloadClass::ImplementationWorker, 16, 4)).accepted);
    }

    const StepPlan plan = step(sched, clock);
    CHECK(plan.admitted == (std::vector<RequestId>{3, 4}));
    CHECK_EQ(sched.stats().lookahead_admissions, 2u);
    CHECK(sched.state(5) == RequestState::WaitingAdmission);
}

TEST_CASE("invalid_workload_and_zero_prefill_or_negative_starvation_rejected") {
    SimClock clock;
    FixedKvCapacity kv(10, 16);

    {
        SchedulerConfig bad;
        bad.max_step_prefill_tokens = 0;
        CHECK_THROWS_AS(Scheduler(bad, clock, kv), std::invalid_argument);
    }
    {
        SchedulerConfig bad;
        bad.starvation_threshold_us = -1;
        CHECK_THROWS_AS(Scheduler(bad, clock, kv), std::invalid_argument);
    }
    {
        SchedulerConfig bad;
        bad.enable_priority_preemption = true;
        bad.priority_preemption_min_rank_gap = 0;
        CHECK_THROWS_AS(Scheduler(bad, clock, kv), std::invalid_argument);
    }

    Scheduler sched({}, clock, kv);
    RequestSpec bad_cls = spec(1, static_cast<WorkloadClass>(99), 16, 2);
    const SubmitResult res = sched.submit(bad_cls);
    CHECK_FALSE(res.accepted);
    CHECK(res.reason == FailureReason::InvalidRequest);
    CHECK(sched.idle());
}

TEST_CASE("stats_current_oldest_queue_age_reflects_live_queue_and_clock") {
    SimClock clock(1000);
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, clock, kv);

    REQUIRE(sched.submit(spec(1, WorkloadClass::BackgroundIndexing, 32, 2)).accepted);
    clock.advance(6000);
    CHECK_EQ(sched.stats().current_oldest_queue_age, 6000);
    CHECK_EQ(sched.stats().max_starvation_age, 6000);

    step(sched, clock, 1000);  // req 1 admitted; waiting_ is now empty
    CHECK_EQ(sched.stats().queued_sequences, 0u);
    CHECK_EQ(sched.stats().current_oldest_queue_age, 0);
    CHECK_EQ(sched.stats().max_starvation_age, 6000);
}

TEST_CASE("sim_clock_saturates_and_backward_clock_never_produces_negative_latencies") {
    SimClock max_clock(std::numeric_limits<TimeUs>::max() - 5);
    max_clock.advance(100);
    CHECK_EQ(max_clock.now(), std::numeric_limits<TimeUs>::max());

    class SkewClock final : public Clock {
    public:
        TimeUs t = 10'000;
        [[nodiscard]] TimeUs now() const noexcept override { return t; }
    };
    SkewClock skew;
    FixedKvCapacity kv(100, 16);
    Scheduler sched({}, skew, kv);
    REQUIRE(sched.submit(spec(1, WorkloadClass::InteractiveUser, 16, 3)).accepted);
    skew.t = 5'000;  // clock stepped backward before admission
    auto p1 = sched.plan_step();
    skew.t = 4'000;
    sched.complete_step(p1);
    auto p2 = sched.plan_step();
    skew.t = 3'000;
    sched.complete_step(p2);
    auto p3 = sched.plan_step();
    skew.t = 2'000;
    sched.complete_step(p3);

    const auto tl = sched.timeline(1);
    REQUIRE(tl.has_value());
    CHECK(tl->total_queue_time >= 0);
    CHECK(tl->ttft() >= 0);
    CHECK(tl->max_inter_token_gap >= 0);
    const auto st = sched.stats();
    CHECK(st.per_class[0].end_to_end.min >= 0);
    CHECK(st.per_class[0].inter_token.min >= 0);
}

TEST_CASE("cost_aware_preemption_chooses_recompute_vs_spill_and_respects_budget") {
    SchedulerConfig cfg;
    cfg.enable_swap = true;
    cfg.swap_min_tokens = 256;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;
    cfg.preemption_cost.kv_bytes_per_token = 4096;
    cfg.preemption_cost.prefill_tokens_per_sec = 2000.0;
    cfg.preemption_cost.spill_bandwidth_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    cfg.preemption_cost.restore_bandwidth_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    cfg.preemption_cost.fixed_spill_overhead_us = 250.0;
    cfg.preemption_cost.spill_cost_margin = 1.05;

    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);

    // Below min_tokens -> Recompute.
    const auto below = sched.evaluate_preemption(128);
    CHECK(below.mode == PreemptionMode::Recompute);
    CHECK_FALSE(below.budget_exceeded);

    // Above min_tokens with fast spill/restore -> SpillToCheckpoint wins over Recompute.
    const auto fast_spill = sched.evaluate_preemption(1024);
    CHECK(fast_spill.mode == PreemptionMode::SpillToCheckpoint);
    CHECK(fast_spill.spill_restore_cost_us < fast_spill.recompute_cost_us);
    CHECK_EQ(fast_spill.estimated_spill_bytes, 1024ULL * 4096ULL);

    // Exceeding spill budget falls back to Recompute with budget_exceeded set.
    const auto over_budget = sched.evaluate_preemption(1024, /*spill_budget_free_bytes=*/512ULL * 1024ULL);
    CHECK(over_budget.mode == PreemptionMode::Recompute);
    CHECK(over_budget.budget_exceeded);

    // Slow spill bandwidth makes spill+restore slower than recompute -> chooses Recompute.
    PreemptionCostModel slow_model = cfg.preemption_cost;
    slow_model.spill_bandwidth_bytes_per_sec = 1.0 * 1024.0 * 1024.0;
    slow_model.restore_bandwidth_bytes_per_sec = 1.0 * 1024.0 * 1024.0;
    sched.update_preemption_cost_model(slow_model);
    const auto slow_dec = sched.evaluate_preemption(1024);
    CHECK(slow_dec.mode == PreemptionMode::Recompute);
    CHECK(slow_dec.spill_restore_cost_us > slow_dec.recompute_cost_us);

    // Live preemption under tight spill budget records spill_budget_fallbacks and falls back to Recompute.
    SchedulerConfig live_cfg = cfg;
    live_cfg.swap_min_tokens = 32;
    live_cfg.preemption_cost.spill_budget_free_bytes = 1024;  // < 32 * 4096
    FixedKvCapacity kv2(4, 16);
    Scheduler sched2(live_cfg, clock, kv2);
    CHECK(sched2.submit(spec(1, WorkloadClass::InteractiveUser, 32, 3)).accepted);
    CHECK(sched2.submit(spec(2, WorkloadClass::BackgroundIndexing, 32, 3)).accepted);
    step(sched2, clock);
    auto plan = step(sched2, clock);
    REQUIRE(was_preempted(plan, 2));
    CHECK(plan.preempted[0].mode == PreemptionMode::Recompute);
    CHECK_EQ(sched2.stats().spill_budget_fallbacks, 1u);
    CHECK_EQ(sched2.requeue_count(2), 1u);
}

TEST_CASE("anti_starvation_preemption_cap_shields_requeued_victim_from_starvation") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = true;
    cfg.anti_starvation_preemption_cap = 1;
    cfg.max_requeue_count = 4;
    SimClock clock;
    FixedKvCapacity kv(4, 16);
    Scheduler sched(cfg, clock, kv);

    // Req 1 (Maintenance) occupies all 4 blocks.
    CHECK(sched.submit(spec(1, WorkloadClass::Maintenance, 60, 4)).accepted);
    step(sched, clock);
    CHECK_EQ(kv.free_blocks(), 0u);

    // Interactive req 2 arrives -> preempts req 1 once (requeue_count becomes 1).
    CHECK(sched.submit(spec(2, WorkloadClass::InteractiveUser, 60, 2)).accepted);
    auto p_preempt = step(sched, clock);
    CHECK(was_preempted(p_preempt, 1));
    CHECK_EQ(sched.requeue_count(1), 1u);

    // Wait for req 2 to finish and req 1 to be re-admitted and running.
    auto is_running = [&](RequestId id) {
        const auto& r = sched.running();
        return std::find(r.begin(), r.end(), id) != r.end();
    };
    int guard = 0;
    while (!is_running(1) && guard++ < 50) {
        step(sched, clock);
    }
    REQUIRE(is_running(1));

    // Interactive req 3 arrives while req 1 is protected by anti_starvation_preemption_cap == 1.
    CHECK(sched.submit(spec(3, WorkloadClass::InteractiveUser, 60, 2)).accepted);
    auto p_shielded = step(sched, clock);
    CHECK_FALSE(was_preempted(p_shielded, 1));
    CHECK(sched.stats().anti_starvation_protections >= 1u);

    // Both req 1 and req 3 complete cleanly without starvation.
    while (!sched.idle() && guard++ < 200) {
        step(sched, clock);
    }
    CHECK(sched.state(1) == RequestState::Completed);
    CHECK(sched.state(2) == RequestState::Completed);
    CHECK(sched.state(3) == RequestState::Completed);
}
