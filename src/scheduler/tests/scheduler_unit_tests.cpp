// Unit tests for the scheduler policy (queue, batching, admission,
// preemption, fairness, telemetry).
#include <doctest/doctest.h>

#include "sonder/inference/scheduler/scheduler.hpp"

#include <algorithm>
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
