// Deterministic simulation tests: mixed interactive / agent fan-out /
// background workloads with latency, throughput, and safety invariants.
#include <doctest/doctest.h>

#include "sonder/inference/scheduler/simulator.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace sonder::inference::scheduler;

namespace {

RequestSpec spec(RequestId id, WorkloadClass cls, TokenCount prompt, TokenCount max_new) {
    RequestSpec s;
    s.id = id;
    s.workload = cls;
    s.task_id = "sim-" + std::to_string(id);
    s.prompt_tokens = prompt;
    s.max_new_tokens = max_new;
    return s;
}

constexpr TimeUs kMs = 1000;

SchedulerConfig mixed_config() {
    SchedulerConfig cfg;
    cfg.max_step_tokens = 2048;
    cfg.max_step_prefill_tokens = 1024;
    cfg.prefill_chunk_tokens = 512;
    cfg.max_step_sequences = 64;
    cfg.max_running_sequences = 48;
    cfg.admission_watermark_blocks = 8;
    cfg.aging_interval_us = 500 * kMs;
    cfg.starvation_threshold_us = 3000 * kMs;
    return cfg;
}

/// 4 long background jobs, a 28-request agent fan-out burst, and a steady
/// trickle of interactive turns.
std::vector<Arrival> mixed_workload() {
    std::vector<Arrival> a;
    RequestId id = 1;
    for (int i = 0; i < 4; ++i) {
        a.push_back({0, spec(id++, WorkloadClass::BackgroundIndexing, 3000, 200)});
    }
    for (int i = 0; i < 24; ++i) {
        a.push_back({50 * kMs, spec(id++, WorkloadClass::ImplementationWorker, 800, 96)});
    }
    for (int i = 0; i < 4; ++i) {
        a.push_back({60 * kMs, spec(id++, WorkloadClass::CriticVerification, 600, 64)});
    }
    for (int i = 0; i < 10; ++i) {
        a.push_back({100 * kMs + i * 200 * kMs, spec(id++, WorkloadClass::InteractiveUser, 200, 48)});
    }
    return a;
}

void print_summary(const char* label, const SimulationResult& r) {
    std::printf("%s: steps=%llu makespan=%.1fms tok/s=%.0f occ=%.2f seqs/step=%.1f preempt=%llu peak_kv=%llu\n",
                label, static_cast<unsigned long long>(r.steps), static_cast<double>(r.makespan) / 1000.0,
                r.generated_tokens_per_second(), r.stats.batch_token_occupancy(),
                r.stats.mean_batch_sequences(),
                static_cast<unsigned long long>(r.stats.preemptions_kv_pressure + r.stats.preemptions_priority),
                static_cast<unsigned long long>(r.peak_kv_blocks));
    for (std::size_t c = 0; c < kWorkloadClassCount; ++c) {
        const auto& s = r.stats.per_class[c];
        if (s.submitted == 0) continue;
        std::printf("  %-22s done=%llu ttft p50=%.1fms p95=%.1fms itl max=%.1fms queue max=%.1fms\n",
                    std::string(to_string(static_cast<WorkloadClass>(c))).c_str(),
                    static_cast<unsigned long long>(s.completed), static_cast<double>(s.ttft.p50) / 1000.0,
                    static_cast<double>(s.ttft.p95) / 1000.0, static_cast<double>(s.inter_token.max) / 1000.0,
                    static_cast<double>(s.queue_time.max) / 1000.0);
    }
}

const ClassStats& cls(const SimulationResult& r, WorkloadClass c) {
    return r.stats.per_class[index_of(c)];
}

}  // namespace

TEST_CASE("mixed_workload_latency_and_throughput_invariants") {
    const auto cfg = mixed_config();
    Simulator sim(cfg, 600, 16);
    const auto workload = mixed_workload();
    const auto r = sim.run(workload);
    print_summary("mixed", r);

    // Safety: budgets and KV capacity are never exceeded.
    CHECK(!r.hit_step_limit);
    CHECK_EQ(r.budget_violations, 0u);
    CHECK_EQ(r.kv_overcommits, 0u);
    CHECK(r.peak_kv_blocks <= 600u);
    CHECK(r.max_step_tokens_seen <= cfg.max_step_tokens);
    CHECK(r.max_prefill_chunk_seen <= cfg.prefill_chunk_tokens);
    CHECK(r.max_running_seen <= cfg.max_running_sequences);

    // Liveness: every request completes, including background work.
    std::uint64_t completed = 0;
    for (const auto& t : r.timelines) {
        CHECK(t.final_state == RequestState::Completed);
        if (t.final_state == RequestState::Completed) ++completed;
    }
    CHECK_EQ(completed, static_cast<std::uint64_t>(workload.size()));
    CHECK_EQ(cls(r, WorkloadClass::BackgroundIndexing).completed, 4u);

    // Latency: interactive turns get their first token quickly and ahead of
    // the agent fan-out and background classes.
    const auto& inter = cls(r, WorkloadClass::InteractiveUser);
    const auto& worker = cls(r, WorkloadClass::ImplementationWorker);
    const auto& bg = cls(r, WorkloadClass::BackgroundIndexing);
    CHECK(inter.ttft.p95 < worker.ttft.p50);
    CHECK(inter.ttft.p95 < bg.ttft.p50);
    CHECK(inter.ttft.max < 100 * kMs);
    // Decode-first + chunked prefill bound inter-token latency for
    // interactive streams to roughly one maximal step.
    CHECK(inter.inter_token.max < 60 * kMs);
    CHECK_EQ(inter.preemptions, 0u);

    // Throughput: continuous batching actually batches.
    CHECK(r.stats.mean_batch_sequences() > 4.0);
    CHECK(r.generated_tokens_per_second() > 1000.0);
    CHECK((r.stats.decode_tokens > 0u && r.stats.prefill_tokens > 0u));
}

TEST_CASE("simulation_is_deterministic") {
    Simulator sim(mixed_config(), 600, 16);
    const auto a = sim.run(mixed_workload());
    const auto b = sim.run(mixed_workload());
    CHECK_EQ(a.steps, b.steps);
    CHECK_EQ(a.makespan, b.makespan);
    CHECK_EQ(a.timelines.size(), b.timelines.size());
    for (std::size_t i = 0; i < a.timelines.size() && i < b.timelines.size(); ++i) {
        CHECK_EQ(a.timelines[i].first_token, b.timelines[i].first_token);
        CHECK_EQ(a.timelines[i].finished, b.timelines[i].finished);
        CHECK_EQ(a.timelines[i].preemptions, b.timelines[i].preemptions);
    }
}

TEST_CASE("kv_pressure_workload_stays_safe_and_live") {
    auto cfg = mixed_config();
    cfg.admission_watermark_blocks = 1;
    Simulator sim(cfg, 240, 16);  // 3840 tokens: heavy contention
    const auto r = sim.run(mixed_workload());
    print_summary("kv-pressure", r);
    CHECK(!r.hit_step_limit);
    CHECK_EQ(r.kv_overcommits, 0u);
    CHECK(r.peak_kv_blocks <= 240u);
    CHECK(r.stats.preemptions_kv_pressure + r.stats.preemptions_priority > 0u);
    std::uint64_t done = 0;
    std::uint64_t failed = 0;
    for (const auto& c : r.stats.per_class) {
        done += c.completed;
        failed += c.failed;
    }
    CHECK_EQ(done + failed, 42u);
    CHECK_EQ(cls(r, WorkloadClass::InteractiveUser).completed, 10u);
    // Urgent work is preempted no more than background work.
    CHECK(cls(r, WorkloadClass::InteractiveUser).preemptions <=
          cls(r, WorkloadClass::BackgroundIndexing).preemptions +
              cls(r, WorkloadClass::ImplementationWorker).preemptions);
}

TEST_CASE("cancellation_mid_run_frees_capacity") {
    Simulator sim(mixed_config(), 600, 16);
    std::vector<CancellationEvent> cancels;
    for (RequestId id = 5; id <= 12; ++id) cancels.push_back({80 * kMs, id});  // 8 workers
    const auto r = sim.run(mixed_workload(), cancels);
    CHECK(!r.hit_step_limit);
    CHECK_EQ(cls(r, WorkloadClass::ImplementationWorker).cancelled, 8u);
    CHECK_EQ(cls(r, WorkloadClass::ImplementationWorker).completed, 16u);
    CHECK_EQ(r.stats.active_sequences, 0u);
    CHECK_EQ(r.stats.queued_sequences, 0u);
}

TEST_CASE("starvation_guard_bounds_background_wait") {
    // Two slots, a saturating stream of interactive turns and one background job.
    auto make = [](TimeUs starvation) {
        SchedulerConfig cfg;
        cfg.max_running_sequences = 2;
        cfg.aging_interval_us = 1'000'000 * kMs;  // disable aging, isolate the guard
        cfg.starvation_threshold_us = starvation;
        cfg.enable_priority_preemption = false;
        return cfg;
    };
    std::vector<Arrival> w;
    w.push_back({0, spec(1, WorkloadClass::BackgroundIndexing, 256, 16)});
    RequestId id = 2;
    for (TimeUs t = 0; t < 4000 * kMs; t += 20 * kMs) {
        w.push_back({t, spec(id++, WorkloadClass::InteractiveUser, 128, 16)});
        w.push_back({t, spec(id++, WorkloadClass::InteractiveUser, 128, 16)});
    }

    const auto guarded = Simulator(make(500 * kMs), 1000, 16).run(w);
    const auto unguarded = Simulator(make(1'000'000 * kMs), 1000, 16).run(w);
    const auto bg_guarded = guarded.timelines.front();
    const auto bg_unguarded = unguarded.timelines.front();
    std::printf("background queue wait: guarded=%.1fms unguarded=%.1fms promotions=%llu\n",
                static_cast<double>(bg_guarded.total_queue_time) / 1000.0,
                static_cast<double>(bg_unguarded.total_queue_time) / 1000.0,
                static_cast<unsigned long long>(guarded.stats.starvation_promotions));
    CHECK(bg_guarded.final_state == RequestState::Completed);
    CHECK(bg_guarded.total_queue_time < 700 * kMs);
    CHECK(bg_unguarded.total_queue_time > 2000 * kMs);
    CHECK(guarded.stats.starvation_promotions >= 1u);
}

TEST_CASE("early_stop_shortens_generation") {
    Simulator sim(mixed_config(), 600, 16);
    sim.early_stop = [](RequestId id, const Scheduler& s) {
        const auto t = s.timeline(id);
        return t && t->generated_tokens >= 10;
    };
    const auto r = sim.run(mixed_workload());
    CHECK(!r.hit_step_limit);
    for (const auto& t : r.timelines) {
        CHECK(t.final_state == RequestState::Completed);
        CHECK(t.generated_tokens <= 11u);
    }
}
