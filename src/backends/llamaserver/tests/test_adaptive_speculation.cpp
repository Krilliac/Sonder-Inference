#include <doctest/doctest.h>

#include "../adaptive_speculation.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;

namespace {

SpeculationObservation make_obs(SpeculationWorkload workload, std::uint64_t draft_n, std::uint64_t accepted,
                                std::uint64_t predicted_n = 32, double decode_tps = 64.0) {
    SpeculationObservation obs;
    obs.workload = workload;
    obs.draft_n = draft_n;
    obs.draft_n_accepted = accepted;
    obs.predicted_n = predicted_n;
    obs.decode_tps = decode_tps;
    return obs;
}

}  // namespace

TEST_SUITE("llamaserver_adaptive_speculation") {

TEST_CASE("workload string round trip and unknown rejection") {
    constexpr std::array<SpeculationWorkload, kSpeculationWorkloadCount> kWorkloads = {
        SpeculationWorkload::general,
        SpeculationWorkload::code_edit,
        SpeculationWorkload::tool_json,
        SpeculationWorkload::prose,
        SpeculationWorkload::reasoning,
    };
    for (const auto workload : kWorkloads) {
        const char *name = to_string(workload);
        REQUIRE(name != nullptr);
        const auto parsed = parse_speculation_workload(name);
        REQUIRE(parsed.has_value());
        CHECK(*parsed == workload);
    }
    CHECK_FALSE(parse_speculation_workload("").has_value());
    CHECK_FALSE(parse_speculation_workload("unknown").has_value());
    CHECK_FALSE(parse_speculation_workload("CODE_EDIT").has_value());
    CHECK_FALSE(parse_speculation_workload("code-edit").has_value());
    CHECK(std::string(to_string(static_cast<SpeculationWorkload>(99))) == "unknown");
}

TEST_CASE("config validation rejects nonfinite reversed and out of range settings") {
    const AdaptiveSpeculationConfig valid{};
    CHECK(validate_adaptive_speculation_config(valid).ok());

    const auto expect_invalid = [](const AdaptiveSpeculationConfig &cfg) {
        const Status status = validate_adaptive_speculation_config(cfg);
        CHECK_FALSE(status.ok());
        CHECK(status.code() == ErrorCode::invalid_argument);
        CHECK_FALSE(status.message().empty());
    };

    // Non-finite and out-of-range thresholds.
    {
        auto cfg = valid;
        cfg.pause_acceptance_threshold = std::numeric_limits<double>::quiet_NaN();
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.step_down_acceptance_threshold = std::numeric_limits<double>::infinity();
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.step_up_acceptance_threshold = -std::numeric_limits<double>::infinity();
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.pause_acceptance_threshold = -0.01;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.step_up_acceptance_threshold = 1.01;
        expect_invalid(cfg);
    }

    // Reversed thresholds.
    {
        auto cfg = valid;
        cfg.pause_acceptance_threshold = 0.50;
        cfg.step_down_acceptance_threshold = 0.40;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.step_down_acceptance_threshold = 0.80;
        cfg.step_up_acceptance_threshold = 0.70;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.pause_acceptance_threshold = 0.80;
        cfg.step_down_acceptance_threshold = 0.80;
        cfg.step_up_acceptance_threshold = 0.70;
        expect_invalid(cfg);
    }

    // Equal ordered thresholds remain valid.
    {
        auto cfg = valid;
        cfg.pause_acceptance_threshold = 0.30;
        cfg.step_down_acceptance_threshold = 0.30;
        cfg.step_up_acceptance_threshold = 0.60;
        CHECK(validate_adaptive_speculation_config(cfg).ok());
    }

    // Draft length bounds.
    {
        auto cfg = valid;
        cfg.min_draft_n = 5;
        cfg.max_draft_n = 3;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.min_draft_n = 2;
        cfg.initial_draft_n = 1;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.initial_draft_n = 7;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.max_draft_n = 0;
        cfg.initial_draft_n = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.max_draft_n = AdaptiveSpeculationConfig::kMaxDraftN + 1;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.probe_draft_n = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.probe_draft_n = cfg.max_draft_n + 1;
        expect_invalid(cfg);
    }

    // Window size and min_samples bounds.
    {
        auto cfg = valid;
        cfg.window_size = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.window_size = AdaptiveSpeculationConfig::kMaxWindowSize + 1;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.min_samples = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.min_samples = cfg.window_size + 1;
        expect_invalid(cfg);
    }
}

TEST_CASE("high acceptance on code_edit steps up cleanly to max_draft_n") {
    AdaptiveSpeculationConfig cfg;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    const auto initial = policy.recommend(SpeculationWorkload::code_edit);
    CHECK(initial.draft_n_max == 3);
    CHECK_FALSE(initial.paused);
    CHECK_FALSE(initial.is_probe);
    CHECK(initial.sample_count == 0);

    // First 3 samples stay in warmup (< min_samples = 4).
    for (std::size_t i = 1; i <= 3; ++i) {
        const auto d = policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 17));  // 85%
        CHECK(d.draft_n_max == 3);
        CHECK_FALSE(d.paused);
        CHECK(d.sample_count == i);
        CHECK(d.rolling_acceptance == doctest::Approx(0.85));
        CHECK(d.reason == "warming_up");
    }

    // Samples 4, 5, 6 step up 3 -> 4 -> 5 -> 6.
    const auto d4 = policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 17));
    CHECK(d4.draft_n_max == 4);
    CHECK(d4.reason == "step_up");

    const auto d5 = policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 17));
    CHECK(d5.draft_n_max == 5);
    CHECK(d5.reason == "step_up");

    const auto d6 = policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 17));
    CHECK(d6.draft_n_max == 6);
    CHECK(d6.reason == "step_up");

    // Further high-acceptance samples cap cleanly at max_draft_n.
    const auto d7 = policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 18));
    CHECK(d7.draft_n_max == cfg.max_draft_n);
    CHECK(d7.reason == "at_max_draft");

    // Hysteresis deadband [0.45, 0.72) holds steady without oscillation.
    AdaptiveSpeculationConfig hold_cfg;
    hold_cfg.window_size = 4;
    hold_cfg.min_samples = 4;
    AdaptiveSpeculationPolicy hold_policy(hold_cfg);
    for (int i = 0; i < 4; ++i) {
        hold_policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 12));  // 60%
    }
    const auto held = hold_policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 12));
    CHECK(held.draft_n_max == hold_cfg.initial_draft_n);
    CHECK_FALSE(held.paused);
    CHECK(held.reason == "hold_hysteresis");
}

TEST_CASE("low acceptance on prose steps down and pauses without affecting code_edit") {
    AdaptiveSpeculationConfig cfg;
    cfg.window_size = 4;
    cfg.min_samples = 4;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    // Ramp code_edit to max_draft_n first.
    for (int i = 0; i < 6; ++i) {
        policy.observe(make_obs(SpeculationWorkload::code_edit, 20, 17));  // 85%
    }
    CHECK(policy.recommend(SpeculationWorkload::code_edit).draft_n_max == 6);

    // Moderate low acceptance (35%, between pause 0.25 and step_down 0.45) steps
    // prose down 3 -> 2 -> 1 without pausing.
    for (int i = 0; i < 3; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 7));  // 35%
    }
    const auto step1 = policy.observe(make_obs(SpeculationWorkload::prose, 20, 7));
    CHECK(step1.draft_n_max == 2);
    CHECK_FALSE(step1.paused);
    CHECK(step1.reason == "step_down");

    const auto step2 = policy.observe(make_obs(SpeculationWorkload::prose, 20, 7));
    CHECK(step2.draft_n_max == 1);
    CHECK_FALSE(step2.paused);
    CHECK(step2.reason == "step_down");

    const auto hold_floor = policy.observe(make_obs(SpeculationWorkload::prose, 20, 7));
    CHECK(hold_floor.draft_n_max == 1);
    CHECK_FALSE(hold_floor.paused);
    CHECK(hold_floor.reason == "at_min_draft");

    // Very low acceptance (15%, below pause 0.25) pauses prose cleanly.
    SpeculationDecision paused;
    for (int i = 0; i < 4; ++i) {
        paused = policy.observe(make_obs(SpeculationWorkload::prose, 20, 3));  // 15%
    }
    CHECK(paused.paused);
    CHECK(paused.draft_n_max == 0);
    CHECK(paused.rolling_acceptance == doctest::Approx(0.15));
    CHECK(paused.reason == "paused_low_acceptance");

    // Workload isolation: code_edit stays at 6 and untouched classes stay initial.
    const auto code_decision = policy.recommend(SpeculationWorkload::code_edit);
    CHECK_FALSE(code_decision.paused);
    CHECK(code_decision.draft_n_max == 6);

    const auto tool_decision = policy.recommend(SpeculationWorkload::tool_json);
    CHECK_FALSE(tool_decision.paused);
    CHECK(tool_decision.draft_n_max == cfg.initial_draft_n);
    CHECK(tool_decision.sample_count == 0);

    // Non-zero min_draft_n prevents pausing and floors at min_draft_n.
    AdaptiveSpeculationConfig floor_cfg;
    floor_cfg.min_draft_n = 2;
    floor_cfg.initial_draft_n = 3;
    floor_cfg.max_draft_n = 6;
    floor_cfg.min_samples = 4;
    AdaptiveSpeculationPolicy floor_policy(floor_cfg);
    SpeculationDecision floored;
    for (int i = 0; i < 6; ++i) {
        floored = floor_policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));  // 10%
    }
    CHECK_FALSE(floored.paused);
    CHECK(floored.draft_n_max == 2);
    CHECK(floored.reason == "at_min_draft");
}

TEST_CASE("cooldown reprobe recovers from paused state when acceptance improves") {
    AdaptiveSpeculationConfig cfg;
    cfg.window_size = 8;
    cfg.min_samples = 4;
    cfg.cooldown_requests = 4;
    cfg.probe_draft_n = 1;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    // Push prose into paused state with 4 bad samples (15%).
    for (int i = 0; i < 4; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 3));
    }

    // Cooldown skips exactly cooldown_requests calls before issuing one probe.
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        const auto d = policy.recommend(SpeculationWorkload::prose);
        CHECK(d.paused);
        CHECK_FALSE(d.is_probe);
        CHECK(d.draft_n_max == 0);
        CHECK(d.reason == "paused_cooldown");
    }

    const auto probe1 = policy.recommend(SpeculationWorkload::prose);
    CHECK_FALSE(probe1.paused);
    CHECK(probe1.is_probe);
    CHECK(probe1.probe_id > 0u);
    CHECK(probe1.draft_n_max == cfg.probe_draft_n);
    CHECK(probe1.reason == "cooldown_probe");

    // While a probe is already in flight, concurrent recommend() stays paused.
    const auto concurrent = policy.recommend(SpeculationWorkload::prose);
    CHECK(concurrent.paused);
    CHECK_FALSE(concurrent.is_probe);
    CHECK(concurrent.probe_id == 0u);
    CHECK(concurrent.draft_n_max == 0);
    CHECK(concurrent.reason == "paused_probe_in_flight");

    // If the probe fails (10% < 25%), bucket stays paused and restarts cooldown.
    auto fail_obs = make_obs(SpeculationWorkload::prose, 20, 2);
    fail_obs.is_probe = true;
    fail_obs.probe_id = probe1.probe_id;
    const auto failed_probe = policy.observe(fail_obs);
    CHECK(failed_probe.paused);
    CHECK(failed_probe.draft_n_max == 0);
    CHECK(failed_probe.reason == "probe_failed_repaused");

    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        const auto d = policy.recommend(SpeculationWorkload::prose);
        CHECK(d.paused);
        CHECK_FALSE(d.is_probe);
        CHECK(d.draft_n_max == 0);
    }

    const auto probe2 = policy.recommend(SpeculationWorkload::prose);
    CHECK_FALSE(probe2.paused);
    CHECK(probe2.is_probe);
    CHECK(probe2.probe_id > probe1.probe_id);
    CHECK(probe2.draft_n_max == cfg.probe_draft_n);

    // If the probe succeeds (85% >= 25%), stale bad history is cleared and
    // speculation unpauses immediately at probe_draft_n.
    auto recov_obs = make_obs(SpeculationWorkload::prose, 20, 17);
    recov_obs.is_probe = true;
    recov_obs.probe_id = probe2.probe_id;
    const auto recovered = policy.observe(recov_obs);
    CHECK_FALSE(recovered.paused);
    CHECK_FALSE(recovered.is_probe);
    CHECK(recovered.draft_n_max == cfg.probe_draft_n);
    CHECK(recovered.sample_count == 1);
    CHECK(recovered.rolling_acceptance == doctest::Approx(0.85));
    CHECK(recovered.reason == "probe_recovered");

    // Continued high acceptance ramps prose back up above probe_draft_n.
    SpeculationDecision ramped;
    for (int i = 0; i < 4; ++i) {
        ramped = policy.observe(make_obs(SpeculationWorkload::prose, 20, 17));
    }
    CHECK_FALSE(ramped.paused);
    CHECK(ramped.draft_n_max > cfg.probe_draft_n);

    // cancel_probe releases an in-flight probe and restarts cooldown cleanly.
    for (int i = 0; i < 8; ++i) {
        policy.observe(make_obs(SpeculationWorkload::reasoning, 20, 2));
    }
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        CHECK(policy.recommend(SpeculationWorkload::reasoning).paused);
    }
    const auto rp1 = policy.recommend(SpeculationWorkload::reasoning);
    CHECK(rp1.is_probe);
    CHECK(rp1.probe_id > probe2.probe_id);
    policy.cancel_probe(SpeculationWorkload::reasoning, rp1.probe_id);
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        const auto d = policy.recommend(SpeculationWorkload::reasoning);
        CHECK(d.paused);
        CHECK_FALSE(d.is_probe);
    }
    const auto rp2 = policy.recommend(SpeculationWorkload::reasoning);
    CHECK(rp2.is_probe);
    CHECK(rp2.probe_id > rp1.probe_id);
}

TEST_CASE("ignores tiny noisy samples clamps malformed counts and converts timings") {
    AdaptiveSpeculationConfig cfg;
    cfg.min_draft_tokens_per_sample = 4;
    cfg.min_samples = 4;
    AdaptiveSpeculationPolicy policy(cfg);

    // Noisy samples below min_draft_tokens_per_sample are ignored.
    const auto no_draft = policy.observe(make_obs(SpeculationWorkload::general, 0, 0));
    CHECK(no_draft.sample_count == 0);
    CHECK(no_draft.draft_n_max == cfg.initial_draft_n);
    CHECK(no_draft.reason == "ignored_no_draft");

    const auto tiny1 = policy.observe(make_obs(SpeculationWorkload::general, 1, 0));
    CHECK(tiny1.sample_count == 0);
    CHECK(tiny1.draft_n_max == cfg.initial_draft_n);
    CHECK(tiny1.reason == "ignored_below_min_draft_tokens");

    const auto tiny3 = policy.observe(make_obs(SpeculationWorkload::general, 3, 0));
    CHECK(tiny3.sample_count == 0);
    CHECK(tiny3.draft_n_max == cfg.initial_draft_n);

    // Malformed observation with draft_n_accepted > draft_n is clamped safely.
    const auto clamped = policy.observe(make_obs(SpeculationWorkload::general, 10, 25));
    CHECK(clamped.sample_count == 1);
    CHECK(clamped.rolling_acceptance == doctest::Approx(1.0));

    // Overflow-safe accumulation with values near UINT64_MAX.
    AdaptiveSpeculationPolicy big_policy(cfg);
    const std::uint64_t huge_draft = std::numeric_limits<std::uint64_t>::max() - 10;
    const std::uint64_t huge_accepted = huge_draft / 2;
    SpeculationDecision big_decision;
    for (int i = 0; i < 4; ++i) {
        big_decision = big_policy.observe(make_obs(SpeculationWorkload::tool_json, huge_draft, huge_accepted));
    }
    CHECK(big_decision.sample_count == 4);
    CHECK(big_decision.rolling_acceptance == doctest::Approx(0.5).epsilon(1e-6));
    CHECK_FALSE(big_decision.paused);

    // observation_from_timings(Timings) computes decode_tps from completion_tokens and eval_ns.
    {
        const auto parsed = json::parse(
            R"({"timings":{"draft_n":20,"draft_n_accepted":15,"predicted_n":50,"predicted_ms":500.0}})");
        REQUIRE(parsed.ok());
        Timings t;
        REQUIRE(parse_timings(parsed.value(), t).ok());
        const auto obs = observation_from_timings(t, SpeculationWorkload::code_edit);
        CHECK(obs.workload == SpeculationWorkload::code_edit);
        CHECK(obs.draft_n == 20);
        CHECK(obs.draft_n_accepted == 15);
        CHECK(obs.predicted_n == 50);
        CHECK(obs.decode_tps == doctest::Approx(100.0));
    }

    // observation_from_timings(Timings) preserves explicit predicted_tokens_per_second
    // and clamps malformed draft_accepted_tokens > draft_tokens.
    {
        Timings t;
        t.draft_present = true;
        t.draft_accepted_present = true;
        t.completion_present = true;
        t.speed_present = true;
        t.draft_tokens = 12;
        t.draft_accepted_tokens = 30;
        t.completion_tokens = 40;
        t.predicted_tokens_per_second = 88.5;
        const auto obs = observation_from_timings(t, SpeculationWorkload::reasoning);
        CHECK(obs.workload == SpeculationWorkload::reasoning);
        CHECK(obs.draft_n == 12);
        CHECK(obs.draft_n_accepted == 12);
        CHECK(obs.predicted_n == 40);
        CHECK(obs.decode_tps == doctest::Approx(88.5));
    }

    // observation_from_timings(BackendTimings) extracts raw optional counters cleanly.
    {
        BackendTimings raw;
        raw.draft_n = 16;
        raw.draft_n_accepted = 24;  // malformed, must clamp to 16
        raw.predicted_n = 32;
        raw.predicted_ms = 250.0;
        const auto obs = observation_from_timings(raw, SpeculationWorkload::tool_json);
        CHECK(obs.workload == SpeculationWorkload::tool_json);
        CHECK(obs.draft_n == 16);
        CHECK(obs.draft_n_accepted == 16);
        CHECK(obs.predicted_n == 32);
        CHECK(obs.decode_tps == doctest::Approx(128.0));
    }
}

TEST_CASE("status_json exposes schema and reset restores initial state") {
    AdaptiveSpeculationConfig cfg;
    cfg.cooldown_requests = 2;
    cfg.min_samples = 4;
    AdaptiveSpeculationPolicy policy(cfg);

    policy.observe(make_obs(SpeculationWorkload::code_edit, 2, 1));   // ignored
    policy.observe(make_obs(SpeculationWorkload::code_edit, 10, 15)); // clamped to 10
    for (int i = 0; i < 4; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 3, 40, 50.0));
    }
    CHECK(policy.recommend(SpeculationWorkload::prose).paused);
    CHECK(policy.recommend(SpeculationWorkload::prose).paused);
    const auto prose_probe = policy.recommend(SpeculationWorkload::prose);
    REQUIRE(prose_probe.is_probe);
    CHECK(prose_probe.probe_id > 0u);
    auto prose_probe_obs = make_obs(SpeculationWorkload::prose, 20, 16, 40, 75.0);
    prose_probe_obs.is_probe = true;
    prose_probe_obs.probe_id = prose_probe.probe_id;
    policy.observe(prose_probe_obs);

    const json::Object status = policy.status_json();
    REQUIRE(status.find("enabled") != nullptr);
    CHECK(status.find("enabled")->as_bool() == true);

    const auto *config_val = status.find("config");
    REQUIRE(config_val != nullptr);
    REQUIRE(config_val->is_object());
    CHECK(config_val->find("min_draft_n")->as_uint() == cfg.min_draft_n);
    CHECK(config_val->find("initial_draft_n")->as_uint() == cfg.initial_draft_n);
    CHECK(config_val->find("max_draft_n")->as_uint() == cfg.max_draft_n);
    CHECK(config_val->find("window_size")->as_uint() == cfg.window_size);
    CHECK(config_val->find("min_samples")->as_uint() == cfg.min_samples);
    CHECK(config_val->find("cooldown_requests")->as_uint() == cfg.cooldown_requests);
    CHECK(config_val->find("probe_draft_n")->as_uint() == cfg.probe_draft_n);
    CHECK(config_val->find("pause_acceptance_threshold")->as_double() ==
          doctest::Approx(cfg.pause_acceptance_threshold));
    CHECK(config_val->find("step_down_acceptance_threshold")->as_double() ==
          doctest::Approx(cfg.step_down_acceptance_threshold));
    CHECK(config_val->find("step_up_acceptance_threshold")->as_double() ==
          doctest::Approx(cfg.step_up_acceptance_threshold));
    CHECK(config_val->find("min_draft_tokens_per_sample")->as_uint() == cfg.min_draft_tokens_per_sample);

    const auto *workloads_val = status.find("workloads");
    REQUIRE(workloads_val != nullptr);
    REQUIRE(workloads_val->is_object());
    CHECK(workloads_val->as_object().size() == kSpeculationWorkloadCount);

    for (const char *name : {"general", "code_edit", "tool_json", "prose", "reasoning"}) {
        const auto *bucket = workloads_val->find(name);
        REQUIRE(bucket != nullptr);
        REQUIRE(bucket->is_object());
        CHECK(bucket->find("draft_n_max") != nullptr);
        CHECK(bucket->find("paused") != nullptr);
        CHECK(bucket->find("probe_in_flight") != nullptr);
        CHECK(bucket->find("cooldown_remaining") != nullptr);
        CHECK(bucket->find("rolling_acceptance") != nullptr);
        CHECK(bucket->find("rolling_decode_tps") != nullptr);
        CHECK(bucket->find("sample_count") != nullptr);
        CHECK(bucket->find("window_draft_tokens") != nullptr);
        CHECK(bucket->find("window_accepted_tokens") != nullptr);
        CHECK(bucket->find("total_observations") != nullptr);
        CHECK(bucket->find("ignored_observations") != nullptr);
        CHECK(bucket->find("clamped_observations") != nullptr);
        CHECK(bucket->find("probes_issued") != nullptr);
        CHECK(bucket->find("probe_recoveries") != nullptr);
        CHECK(bucket->find("reason") != nullptr);
    }

    const auto *code_bucket = workloads_val->find("code_edit");
    REQUIRE(code_bucket != nullptr);
    CHECK(code_bucket->find("ignored_observations")->as_uint() == 1);
    CHECK(code_bucket->find("clamped_observations")->as_uint() == 1);
    CHECK(code_bucket->find("sample_count")->as_uint() == 1);

    const auto *prose_bucket = workloads_val->find("prose");
    REQUIRE(prose_bucket != nullptr);
    CHECK(prose_bucket->find("paused")->as_bool() == false);
    CHECK(prose_bucket->find("probes_issued")->as_uint() == 1);
    CHECK(prose_bucket->find("probe_recoveries")->as_uint() == 1);
    CHECK(prose_bucket->find("rolling_decode_tps")->as_double() == doctest::Approx(75.0));

    // Serialized JSON round-trips cleanly.
    const std::string serialized = json::Value(status).dump();
    const auto reparsed = json::parse(serialized);
    CHECK(reparsed.ok());

    // reset() clears every bucket back to initial state.
    policy.reset();
    const json::Object after_reset = policy.status_json();
    const auto *reset_workloads = after_reset.find("workloads");
    REQUIRE(reset_workloads != nullptr);
    for (const char *name : {"general", "code_edit", "tool_json", "prose", "reasoning"}) {
        const auto *bucket = reset_workloads->find(name);
        REQUIRE(bucket != nullptr);
        CHECK(bucket->find("draft_n_max")->as_uint() == cfg.initial_draft_n);
        CHECK(bucket->find("paused")->as_bool() == false);
        CHECK(bucket->find("sample_count")->as_uint() == 0);
        CHECK(bucket->find("total_observations")->as_uint() == 0);
        CHECK(bucket->find("ignored_observations")->as_uint() == 0);
        CHECK(bucket->find("clamped_observations")->as_uint() == 0);
        CHECK(bucket->find("probes_issued")->as_uint() == 0);
        CHECK(bucket->find("probe_recoveries")->as_uint() == 0);
        CHECK(bucket->find("rolling_acceptance")->as_double() == doctest::Approx(0.0));
    }

    // Disabled policy keeps static initial_draft_n without adapting.
    AdaptiveSpeculationConfig disabled_cfg;
    disabled_cfg.enabled = false;
    AdaptiveSpeculationPolicy disabled(disabled_cfg);
    const auto rec = disabled.recommend(SpeculationWorkload::prose);
    CHECK(rec.draft_n_max == disabled_cfg.initial_draft_n);
    CHECK(rec.reason == "disabled");
    const auto obs_dec = disabled.observe(make_obs(SpeculationWorkload::prose, 20, 0));
    CHECK(obs_dec.draft_n_max == disabled_cfg.initial_draft_n);
    CHECK(obs_dec.reason == "disabled");
    CHECK(disabled.status_json().find("enabled")->as_bool() == false);
}

TEST_CASE("concurrent recommend observe and status_json stay thread safe and bounded") {
    AdaptiveSpeculationConfig cfg;
    cfg.window_size = 16;
    cfg.min_samples = 4;
    cfg.cooldown_requests = 3;
    AdaptiveSpeculationPolicy policy(cfg);

    constexpr std::array<SpeculationWorkload, kSpeculationWorkloadCount> kWorkloads = {
        SpeculationWorkload::general,
        SpeculationWorkload::code_edit,
        SpeculationWorkload::tool_json,
        SpeculationWorkload::prose,
        SpeculationWorkload::reasoning,
    };

    std::atomic<int> violations{0};
    std::vector<std::thread> workers;
    workers.reserve(4);
    for (int tid = 0; tid < 4; ++tid) {
        workers.emplace_back([&, tid] {
            for (int iter = 0; iter < 150; ++iter) {
                const auto workload = kWorkloads[static_cast<std::size_t>((tid + iter) % 5)];
                const auto rec = policy.recommend(workload);
                if (rec.draft_n_max > cfg.max_draft_n || rec.sample_count > cfg.window_size ||
                    !std::isfinite(rec.rolling_acceptance) || rec.rolling_acceptance < 0.0 ||
                    rec.rolling_acceptance > 1.0 || (rec.draft_n_max == 0) != rec.paused) {
                    violations.fetch_add(1);
                }
                const std::uint64_t accepted = (iter % 3 == 0) ? 18 : ((iter % 3 == 1) ? 3 : 12);
                auto obs = make_obs(workload, 20, accepted);
                obs.is_probe = rec.is_probe;
                obs.probe_id = rec.probe_id;
                const auto upd = policy.observe(obs);
                if (upd.draft_n_max > cfg.max_draft_n || upd.sample_count > cfg.window_size ||
                    !std::isfinite(upd.rolling_acceptance) || upd.rolling_acceptance < 0.0 ||
                    upd.rolling_acceptance > 1.0 || (upd.draft_n_max == 0) != upd.paused) {
                    violations.fetch_add(1);
                }
                if (iter % 25 == 0) {
                    const auto snap = policy.status_json();
                    if (!snap.contains("workloads"))
                        violations.fetch_add(1);
                }
                if (tid == 0 && iter == 75) {
                    policy.reset();
                }
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    CHECK(violations.load() == 0);
}

TEST_CASE("probe completing with zero draft tokens releases probe slot and restarts cooldown") {
    AdaptiveSpeculationConfig cfg;
    cfg.window_size = 8;
    cfg.min_samples = 4;
    cfg.cooldown_requests = 3;
    cfg.probe_draft_n = 1;
    cfg.min_draft_tokens_per_sample = 4;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    for (int i = 0; i < 4; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));
    }
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        CHECK(policy.recommend(SpeculationWorkload::prose).paused);
    }
    const auto probe = policy.recommend(SpeculationWorkload::prose);
    REQUIRE(probe.is_probe);
    CHECK(probe.probe_id > 0u);

    // Concurrent non-probe request with 0 draft tokens must leave probe_in_flight intact.
    SpeculationObservation concurrent_non_probe = make_obs(SpeculationWorkload::prose, 0, 0);
    concurrent_non_probe.is_probe = false;
    policy.observe(concurrent_non_probe);
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // When the probe itself completes with 0 draft tokens (e.g. 1-token EOS),
    // the probe slot must release and restart cooldown rather than wedging permanently.
    SpeculationObservation empty_probe = make_obs(SpeculationWorkload::prose, 0, 0);
    empty_probe.is_probe = true;
    empty_probe.probe_id = probe.probe_id;
    const auto after_empty = policy.observe(empty_probe);
    CHECK(after_empty.paused);
    CHECK(after_empty.reason == "ignored_no_draft");

    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        const auto d = policy.recommend(SpeculationWorkload::prose);
        CHECK(d.paused);
        CHECK_FALSE(d.is_probe);
        CHECK(d.reason == "paused_cooldown");
    }
    const auto next_probe = policy.recommend(SpeculationWorkload::prose);
    CHECK(next_probe.is_probe);
    CHECK(next_probe.probe_id > probe.probe_id);
    CHECK(next_probe.reason == "cooldown_probe");
}

TEST_CASE("concurrent non-probe and stale probe observations cannot hijack active probe") {
    AdaptiveSpeculationConfig cfg;
    cfg.window_size = 4;
    cfg.min_samples = 4;
    cfg.cooldown_requests = 2;
    cfg.probe_draft_n = 1;
    cfg.min_draft_tokens_per_sample = 4;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    for (int i = 0; i < 4; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));
    }
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        CHECK(policy.recommend(SpeculationWorkload::prose).paused);
    }
    const auto probe1 = policy.recommend(SpeculationWorkload::prose);
    REQUIRE(probe1.is_probe);
    REQUIRE(probe1.probe_id > 0u);

    // Concurrent non-probe with below-floor positive draft_n must not clear probe_in_flight.
    policy.observe(make_obs(SpeculationWorkload::prose, 2, 1));
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // Concurrent non-probe with above-floor high or low acceptance must not hijack probe1.
    const auto non_probe_high = policy.observe(make_obs(SpeculationWorkload::prose, 20, 18));
    CHECK(non_probe_high.paused);
    CHECK(non_probe_high.reason == "paused_probe_in_flight");
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    const auto non_probe_low = policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));
    CHECK(non_probe_low.paused);
    CHECK(non_probe_low.reason == "paused_probe_in_flight");
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // Mismatched probe_id on cancel_probe and observe (below or above floor) must be ignored.
    const std::uint64_t bogus_id = probe1.probe_id + 999u;
    policy.cancel_probe(SpeculationWorkload::prose, bogus_id);
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    auto bogus_empty = make_obs(SpeculationWorkload::prose, 0, 0);
    bogus_empty.is_probe = true;
    bogus_empty.probe_id = bogus_id;
    policy.observe(bogus_empty);
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    auto bogus_high = make_obs(SpeculationWorkload::prose, 20, 18);
    bogus_high.is_probe = true;
    bogus_high.probe_id = bogus_id;
    const auto bogus_dec = policy.observe(bogus_high);
    CHECK(bogus_dec.paused);
    CHECK(bogus_dec.reason == "paused_probe_in_flight");
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // Matching cancel_probe releases probe1 and restarts cooldown.
    policy.cancel_probe(SpeculationWorkload::prose, probe1.probe_id);
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_cooldown");
    }
    const auto probe2 = policy.recommend(SpeculationWorkload::prose);
    REQUIRE(probe2.is_probe);
    CHECK(probe2.probe_id > probe1.probe_id);

    // A stale ProbeGuard or late observation from probe1 must not cancel or resolve probe2.
    policy.cancel_probe(SpeculationWorkload::prose, probe1.probe_id);
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    auto stale_obs = make_obs(SpeculationWorkload::prose, 20, 18);
    stale_obs.is_probe = true;
    stale_obs.probe_id = probe1.probe_id;
    CHECK(policy.observe(stale_obs).reason == "paused_probe_in_flight");
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // Matching observation for probe2 resolves cleanly and recovers.
    auto probe2_obs = make_obs(SpeculationWorkload::prose, 20, 17);
    probe2_obs.is_probe = true;
    probe2_obs.probe_id = probe2.probe_id;
    const auto recovered = policy.observe(probe2_obs);
    CHECK_FALSE(recovered.paused);
    CHECK(recovered.draft_n_max == cfg.probe_draft_n);
    CHECK(recovered.reason == "probe_recovered");

    // probe_id remains strictly monotonic across reset().
    policy.reset();
    for (int i = 0; i < 4; ++i) {
        policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));
    }
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        CHECK(policy.recommend(SpeculationWorkload::prose).paused);
    }
    const auto probe3 = policy.recommend(SpeculationWorkload::prose);
    CHECK(probe3.is_probe);
    CHECK(probe3.probe_id > probe2.probe_id);
}

TEST_CASE("measured cost config validation rejects invalid ratios and bounds") {
    const AdaptiveSpeculationConfig valid{};
    CHECK(validate_adaptive_speculation_config(valid).ok());

    const auto expect_invalid = [](const AdaptiveSpeculationConfig &cfg) {
        const Status st = validate_adaptive_speculation_config(cfg);
        CHECK_FALSE(st.ok());
        CHECK(st.code() == ErrorCode::invalid_argument);
    };

    {
        auto cfg = valid;
        cfg.max_entries = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.max_entries = AdaptiveSpeculationConfig::kMaxEntries + 1;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.min_baseline_samples = 0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_step_down_ratio = 0.99;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_step_down_ratio = std::numeric_limits<double>::quiet_NaN();
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_step_down_ratio = 1.20;
        cfg.cost_pause_ratio = 1.10;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_probe_recover_ratio = 0.0;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_probe_recover_ratio = cfg.cost_step_down_ratio + 0.05;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.cost_step_cooldown = 2000;
        expect_invalid(cfg);
    }
    {
        auto cfg = valid;
        cfg.baseline_probe_interval = 200000;
        expect_invalid(cfg);
    }
}

TEST_CASE("high acceptance with worse ms per committed token steps down and chooses K=0") {
    AdaptiveSpeculationConfig cfg;
    cfg.min_draft_n = 0;
    cfg.initial_draft_n = 3;
    cfg.max_draft_n = 6;
    cfg.window_size = 8;
    cfg.min_samples = 4;
    cfg.min_baseline_samples = 2;
    cfg.cost_step_down_ratio = 1.05;
    cfg.cost_pause_ratio = 1.12;
    cfg.cost_probe_recover_ratio = 0.98;
    cfg.cooldown_requests = 2;
    cfg.probe_draft_n = 1;
    cfg.cost_step_cooldown = 4;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    SpeculationContextKey key;
    key.target_identity = "spawn|exe=llama-server|model_path=/models/qwen.gguf|ctx=32768|slots=4";
    key.draft_identity = "type=draft-mtp";
    key.context_band = ContextBand::short_ctx;
    key.concurrency_band = ConcurrencyBand::solo;
    key.workload = SpeculationWorkload::code_edit;

    auto make_cost_obs = [&](std::uint32_t k_used, std::uint64_t draft_n, std::uint64_t accepted,
                             std::uint64_t committed, double decode_ms, double prompt_ms = 0.0) {
        SpeculationObservation obs;
        obs.workload = key.workload;
        obs.context_key = key;
        obs.has_context_key = true;
        obs.draft_n_used = k_used;
        obs.draft_n = draft_n;
        obs.draft_n_accepted = accepted;
        obs.predicted_n = committed;
        obs.decode_elapsed_ms = decode_ms;
        obs.prompt_or_restore_ms = prompt_ms;
        return obs;
    };

    // Seed K=0 baseline: 20 committed tokens in 200 ms = 10.0 ms/committed-token.
    const auto b1 = policy.observe(make_cost_obs(0, 0, 0, 20, 200.0, /*prompt_ms=*/4000.0));
    CHECK(b1.reason == "baseline_observed");
    CHECK(b1.baseline_ms_per_token == doctest::Approx(10.0));
    const auto b2 = policy.observe(make_cost_obs(0, 0, 0, 20, 200.0, /*prompt_ms=*/4000.0));
    CHECK(b2.baseline_ms_per_token == doctest::Approx(10.0));

    // Mild cost regression (10.8 ms/tok = 1.08x baseline) at 85% acceptance steps DOWN 3 -> 2!
    SpeculationDecision mild;
    for (int i = 0; i < 4; ++i) {
        mild = policy.observe(make_cost_obs(3, 20, 17, 20, 216.0));  // 85% acceptance, 10.8 ms/tok
    }
    CHECK_FALSE(mild.paused);
    CHECK(mild.draft_n_max == 2u);
    CHECK(mild.rolling_acceptance == doctest::Approx(0.85));
    CHECK(mild.rolling_ms_per_token == doctest::Approx(10.8));
    CHECK(mild.baseline_ms_per_token == doctest::Approx(10.0));
    CHECK(mild.cost_ratio == doctest::Approx(1.08));
    CHECK(mild.reason == "step_down_cost");

    // Strong cost regression (13.0 ms/tok = 1.30x baseline) at 85% acceptance chooses K=0!
    const auto k2_s1 = policy.observe(make_cost_obs(2, 20, 17, 20, 260.0));
    CHECK_FALSE(k2_s1.paused);  // waits for 2nd sample at K=2 before reversing
    const auto k2_s2 = policy.observe(make_cost_obs(2, 20, 17, 20, 260.0));
    CHECK(k2_s2.paused);
    CHECK(k2_s2.draft_n_max == 0u);
    CHECK(k2_s2.rolling_acceptance == doctest::Approx(0.85));
    CHECK(k2_s2.reason == "paused_cost_regression");

    // Cooldown skips 2 requests, then issues a single probe at K=1.
    CHECK(policy.recommend(key).reason == "paused_cooldown");
    CHECK(policy.recommend(key).reason == "paused_cooldown");
    const auto probe1 = policy.recommend(key);
    REQUIRE(probe1.is_probe);
    CHECK(probe1.draft_n_max == 1u);

    // Probe has 85% acceptance, yet still slower than K=0 baseline (12.0 ms/tok vs 10.0): stays K=0!
    auto bad_probe = make_cost_obs(1, 20, 17, 20, 240.0);
    bad_probe.is_probe = true;
    bad_probe.probe_id = probe1.probe_id;
    const auto repaused = policy.observe(bad_probe);
    CHECK(repaused.paused);
    CHECK(repaused.draft_n_max == 0u);
    CHECK(repaused.reason == "probe_failed_cost_repaused");

    // Next cooldown probe beats baseline (8.0 ms/tok vs 10.0) even with huge prompt_or_restore_ms!
    CHECK(policy.recommend(key).reason == "paused_cooldown");
    CHECK(policy.recommend(key).reason == "paused_cooldown");
    const auto probe2 = policy.recommend(key);
    REQUIRE(probe2.is_probe);
    auto good_probe = make_cost_obs(1, 20, 16, 20, 160.0, /*prompt_ms=*/9500.0);
    good_probe.is_probe = true;
    good_probe.probe_id = probe2.probe_id;
    const auto recovered = policy.observe(good_probe);
    CHECK_FALSE(recovered.paused);
    CHECK(recovered.draft_n_max == 1u);
    CHECK(recovered.rolling_ms_per_token == doctest::Approx(8.0));  // prompt_ms excluded!
    CHECK(recovered.cost_ratio == doctest::Approx(0.80));
    CHECK(recovered.reason == "probe_recovered");
}

TEST_CASE("phase critical path avoids double counting overlapped draft and verify") {
    AdaptiveSpeculationConfig cfg;
    cfg.min_samples = 2;
    cfg.min_baseline_samples = 2;
    cfg.cost_step_down_ratio = 1.05;
    cfg.cost_pause_ratio = 1.12;
    AdaptiveSpeculationPolicy policy(cfg);

    SpeculationContextKey overlapped_key;
    overlapped_key.target_identity = "target-phase";
    overlapped_key.draft_identity = "draft-overlap";
    overlapped_key.workload = SpeculationWorkload::code_edit;

    SpeculationContextKey sequential_key = overlapped_key;
    sequential_key.draft_identity = "draft-sequential";

    // Seed identical 10.0 ms/tok K=0 baseline for both keys.
    for (const auto &k : {overlapped_key, sequential_key}) {
        for (int i = 0; i < 2; ++i) {
            SpeculationObservation b;
            b.context_key = k;
            b.has_context_key = true;
            b.draft_n_used = 0;
            b.predicted_n = 20;
            b.decode_elapsed_ms = 200.0;
            policy.observe(b);
        }
    }

    // Overlapped draft=120ms, verify=140ms, rollback=10ms, coord=10ms -> critical path 160ms / 20 = 8.0ms/tok.
    SpeculationDecision ov_dec;
    for (int i = 0; i < 2; ++i) {
        SpeculationObservation obs;
        obs.context_key = overlapped_key;
        obs.has_context_key = true;
        obs.draft_n_used = 3;
        obs.draft_n = 20;
        obs.draft_n_accepted = 17;
        obs.predicted_n = 20;
        obs.draft_ms = 120.0;
        obs.verify_ms = 140.0;
        obs.rollback_ms = 10.0;
        obs.coordination_ms = 10.0;
        obs.draft_verify_overlapped = true;
        obs.prompt_or_restore_ms = 5000.0;  // excluded from decode speed
        ov_dec = policy.observe(obs);
    }
    CHECK_FALSE(ov_dec.paused);
    CHECK(ov_dec.draft_n_max == 4u);  // stepped up because 8.0ms/tok beats 10.0ms/tok
    CHECK(ov_dec.rolling_ms_per_token == doctest::Approx(8.0));

    // Sequential draft+verify = 280ms / 20 = 14.0ms/tok -> worse than 10.0ms/tok -> chooses K=0!
    SpeculationDecision seq_dec;
    for (int i = 0; i < 2; ++i) {
        SpeculationObservation obs;
        obs.context_key = sequential_key;
        obs.has_context_key = true;
        obs.draft_n_used = 3;
        obs.draft_n = 20;
        obs.draft_n_accepted = 17;
        obs.predicted_n = 20;
        obs.draft_ms = 120.0;
        obs.verify_ms = 140.0;
        obs.rollback_ms = 10.0;
        obs.coordination_ms = 10.0;
        obs.draft_verify_overlapped = false;
        seq_dec = policy.observe(obs);
    }
    CHECK(seq_dec.paused);
    CHECK(seq_dec.draft_n_max == 0u);
    CHECK(seq_dec.rolling_ms_per_token == doctest::Approx(14.0));
    CHECK(seq_dec.reason == "paused_cost_regression");
}

TEST_CASE("concurrency band context band and target draft identities isolate policy state") {
    AdaptiveSpeculationConfig cfg;
    cfg.initial_draft_n = 3;
    cfg.max_draft_n = 6;
    cfg.min_samples = 2;
    cfg.min_baseline_samples = 2;
    AdaptiveSpeculationPolicy policy(cfg);

    CHECK(classify_context_band(0) == ContextBand::short_ctx);
    CHECK(classify_context_band(4096) == ContextBand::short_ctx);
    CHECK(classify_context_band(4097) == ContextBand::medium_ctx);
    CHECK(classify_context_band(16384) == ContextBand::medium_ctx);
    CHECK(classify_context_band(16385) == ContextBand::long_ctx);
    CHECK(classify_context_band(65536) == ContextBand::long_ctx);
    CHECK(classify_context_band(65537) == ContextBand::very_long_ctx);

    CHECK(classify_concurrency_band(0) == ConcurrencyBand::solo);
    CHECK(classify_concurrency_band(1) == ConcurrencyBand::solo);
    CHECK(classify_concurrency_band(2) == ConcurrencyBand::low);
    CHECK(classify_concurrency_band(3) == ConcurrencyBand::low);
    CHECK(classify_concurrency_band(4) == ConcurrencyBand::medium);
    CHECK(classify_concurrency_band(7) == ConcurrencyBand::medium);
    CHECK(classify_concurrency_band(8) == ConcurrencyBand::high);

    const SpeculationContextKey solo_short{"sha256:model-a|ctx=65536", "type=draft-mtp", ContextBand::short_ctx,
                                           ConcurrencyBand::solo, SpeculationWorkload::code_edit};
    const SpeculationContextKey high_conc{"sha256:model-a|ctx=65536", "type=draft-mtp", ContextBand::short_ctx,
                                          ConcurrencyBand::high, SpeculationWorkload::code_edit};
    const SpeculationContextKey long_ctx{"sha256:model-a|ctx=65536", "type=draft-mtp", ContextBand::long_ctx,
                                         ConcurrencyBand::solo, SpeculationWorkload::code_edit};
    const SpeculationContextKey other_draft{"sha256:model-a|ctx=65536", "type=ngram-mod", ContextBand::short_ctx,
                                            ConcurrencyBand::solo, SpeculationWorkload::code_edit};

    auto seed_baseline = [&](const SpeculationContextKey &k, double ms_per_tok) {
        for (int i = 0; i < 2; ++i) {
            SpeculationObservation b;
            b.context_key = k;
            b.has_context_key = true;
            b.draft_n_used = 0;
            b.predicted_n = 20;
            b.decode_elapsed_ms = ms_per_tok * 20.0;
            policy.observe(b);
        }
    };
    auto observe_spec = [&](const SpeculationContextKey &k, std::uint32_t k_used, double ms_per_tok) {
        SpeculationObservation o;
        o.context_key = k;
        o.has_context_key = true;
        o.draft_n_used = k_used;
        o.draft_n = 20;
        o.draft_n_accepted = 17;  // 85% acceptance in all bands
        o.predicted_n = 20;
        o.decode_elapsed_ms = ms_per_tok * 20.0;
        return policy.observe(o);
    };

    seed_baseline(solo_short, 10.0);
    seed_baseline(high_conc, 10.0);
    seed_baseline(long_ctx, 10.0);
    seed_baseline(other_draft, 10.0);

    // Solo + short context wins strongly (6 ms/tok): ramps up to K=6.
    for (int i = 0; i < 5; ++i) {
        const auto rec = policy.recommend(solo_short);
        observe_spec(solo_short, rec.draft_n_max, 6.0);
    }
    CHECK(policy.recommend(solo_short).draft_n_max == 6u);

    // High concurrency (B >= 8): slow at 14 ms/tok even with 85% acceptance -> chooses K=0!
    for (int i = 0; i < 2; ++i) {
        observe_spec(high_conc, 3, 14.0);
    }
    CHECK(policy.recommend(high_conc).paused);
    CHECK(policy.recommend(high_conc).draft_n_max == 0u);

    // Long context (C > 16K): mildly slow at K=3 (10.8 ms/tok) -> steps down to K=2 while solo+short stays K=6.
    for (int i = 0; i < 2; ++i) {
        observe_spec(long_ctx, 3, 10.8);
    }
    CHECK_FALSE(policy.recommend(long_ctx).paused);
    CHECK(policy.recommend(long_ctx).draft_n_max == 2u);

    // Same target & bands with different draft_identity remains independent at initial_draft_n.
    CHECK(policy.recommend(other_draft).draft_n_max == cfg.initial_draft_n);
    CHECK(policy.recommend(solo_short).draft_n_max == 6u);
}

TEST_CASE("per-K cost hysteresis prevents oscillation and foreground requests defer probes") {
    AdaptiveSpeculationConfig cfg;
    cfg.initial_draft_n = 3;
    cfg.max_draft_n = 6;
    cfg.min_samples = 2;
    cfg.cooldown_requests = 2;
    cfg.cost_step_down_ratio = 1.05;
    cfg.cost_step_cooldown = 4;
    cfg.baseline_probe_interval = 3;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    const SpeculationContextKey key{"target-hyst", "draft-hyst", ContextBand::short_ctx, ConcurrencyBand::solo,
                                    SpeculationWorkload::code_edit};

    auto obs_k = [&](std::uint32_t k_used, double ms_per_tok) {
        SpeculationObservation o;
        o.context_key = key;
        o.has_context_key = true;
        o.draft_n_used = k_used;
        o.draft_n = 20;
        o.draft_n_accepted = 18;  // 90% acceptance
        o.predicted_n = 20;
        o.decode_elapsed_ms = ms_per_tok * 20.0;
        return policy.observe(o);
    };

    // 1. K=3 runs at 10.0 ms/tok -> steps up to K=4.
    CHECK(obs_k(3, 10.0).draft_n_max == 3u);
    const auto up = obs_k(3, 10.0);
    CHECK(up.draft_n_max == 4u);
    CHECK(up.reason == "step_up");

    // 2. K=4 regresses to 12.5 ms/tok (even with 90% acceptance!):
    //    Sample 1 holds (does not overshoot to 5); Sample 2 steps down 4 -> 3!
    const auto k4_s1 = obs_k(4, 12.5);
    CHECK(k4_s1.draft_n_max == 4u);
    CHECK(k4_s1.reason == "hold_cost_hysteresis");
    const auto k4_s2 = obs_k(4, 12.5);
    CHECK(k4_s2.draft_n_max == 3u);
    CHECK(k4_s2.reason == "step_down_cost_vs_shorter");

    // 3. Hysteresis holdoff keeps K steady at 3 for cost_step_cooldown observations (no 3<->4 oscillation).
    for (std::uint32_t i = 0; i < cfg.cost_step_cooldown - 1u; ++i) {
        const auto held = obs_k(3, 10.0);
        CHECK(held.draft_n_max == 3u);
        CHECK(held.reason == "hold_cost_cooldown");
    }

    // 4. Baseline calibration probe respects foreground allow_probe=false and single-probe ownership.
    //    After 2 calls, the 3rd request hits baseline_probe_interval=3.
    CHECK_FALSE(policy.recommend(key, true).is_probe);
    CHECK_FALSE(policy.recommend(key, true).is_probe);
    // Foreground request (allow_probe=false) must NOT be diverted into a K=0 calibration probe.
    const auto fg = policy.recommend(key, /*allow_probe=*/false);
    CHECK_FALSE(fg.is_probe);
    CHECK_FALSE(fg.is_baseline_probe);
    CHECK(fg.draft_n_max == 3u);
    // Probe-eligible request claims the single baseline calibration probe.
    const auto bp = policy.recommend(key, /*allow_probe=*/true);
    CHECK(bp.is_probe);
    CHECK(bp.is_baseline_probe);
    CHECK(bp.draft_n_max == 0u);
    CHECK(bp.paused);
    CHECK(bp.probe_id > 0u);
    CHECK(bp.reason == "baseline_calibration_probe");
    // Concurrent request while baseline probe is in flight gets normal K=3 (no probe storm).
    const auto conc = policy.recommend(key, /*allow_probe=*/true);
    CHECK_FALSE(conc.is_probe);
    CHECK(conc.draft_n_max == 3u);
    // Stale cancel_probe is ignored; matching cancel_probe releases cleanly.
    policy.cancel_probe(key, bp.probe_id + 100u);
    CHECK_FALSE(policy.recommend(key, true).is_probe);
    policy.cancel_probe(key, bp.probe_id);

    // 5. Paused cooldown probe also defers when allow_probe=false and issues once when allow_probe=true.
    AdaptiveSpeculationConfig pause_cfg;
    pause_cfg.min_samples = 2;
    pause_cfg.cooldown_requests = 1;
    AdaptiveSpeculationPolicy pause_policy(pause_cfg);
    for (int i = 0; i < 2; ++i) {
        SpeculationObservation low;
        low.context_key = key;
        low.has_context_key = true;
        low.draft_n = 20;
        low.draft_n_accepted = 2;
        low.predicted_n = 20;
        low.decode_elapsed_ms = 200.0;
        pause_policy.observe(low);
    }
    CHECK(pause_policy.recommend(key, true).reason == "paused_cooldown");
    const auto deferred = pause_policy.recommend(key, /*allow_probe=*/false);
    CHECK(deferred.paused);
    CHECK_FALSE(deferred.is_probe);
    CHECK(deferred.probe_id == 0u);
    CHECK(deferred.reason == "paused_probe_deferred");
    const auto granted = pause_policy.recommend(key, /*allow_probe=*/true);
    CHECK(granted.is_probe);
    CHECK(granted.probe_id > 0u);
    CHECK(granted.reason == "cooldown_probe");
    // Concurrent probe storm prevented.
    for (int i = 0; i < 5; ++i) {
        const auto storm = pause_policy.recommend(key, /*allow_probe=*/true);
        CHECK(storm.paused);
        CHECK_FALSE(storm.is_probe);
        CHECK(storm.reason == "paused_probe_in_flight");
    }
}

TEST_CASE("bounded LRU state table enforces strict capacity under thousands of synthetic keys") {
    AdaptiveSpeculationConfig cfg;
    cfg.max_entries = 64;
    cfg.window_size = 8;
    cfg.min_samples = 2;
    REQUIRE(validate_adaptive_speculation_config(cfg).ok());
    AdaptiveSpeculationPolicy policy(cfg);

    // Verify over-length identities are bounded to kMaxIdentityLength while staying distinct.
    const std::string long_prefix(400, 'a');
    const std::string norm1 = normalize_speculation_identity(long_prefix + "-model-one");
    const std::string norm2 = normalize_speculation_identity(long_prefix + "-model-two");
    CHECK(norm1.size() <= SpeculationContextKey::kMaxIdentityLength);
    CHECK(norm2.size() <= SpeculationContextKey::kMaxIdentityLength);
    CHECK(norm1 != norm2);

    // Insert 4,000 synthetic keys and assert hard cap <= 64 on every step.
    constexpr std::size_t kTotalKeys = 4000;
    for (std::size_t i = 0; i < kTotalKeys; ++i) {
        SpeculationContextKey k;
        k.target_identity =
            (i % 100 == 0) ? (long_prefix + "-target-" + std::to_string(i)) : ("target-" + std::to_string(i));
        k.draft_identity = "draft-" + std::to_string(i % 7);
        k.context_band = static_cast<ContextBand>(i % kContextBandCount);
        k.concurrency_band = static_cast<ConcurrencyBand>((i / 4) % kConcurrencyBandCount);
        k.workload = static_cast<SpeculationWorkload>((i / 16) % kSpeculationWorkloadCount);

        (void)policy.recommend(k);
        SpeculationObservation obs;
        obs.context_key = k;
        obs.has_context_key = true;
        obs.draft_n = 20;
        obs.draft_n_accepted = 16;
        obs.predicted_n = 20;
        obs.decode_elapsed_ms = 180.0;
        (void)policy.observe(obs);
        REQUIRE(policy.entry_count() <= cfg.max_entries);
    }

    CHECK(policy.entry_count() == cfg.max_entries);
    CHECK(policy.table_evictions() == kTotalKeys - cfg.max_entries);

    const json::Object snap = policy.status_json();
    const auto *table = snap.find("table");
    REQUIRE(table != nullptr);
    CHECK(table->find("size")->as_uint() == cfg.max_entries);
    CHECK(table->find("capacity")->as_uint() == cfg.max_entries);
    CHECK(table->find("evictions")->as_uint() == kTotalKeys - cfg.max_entries);
    const auto *entries = snap.find("entries");
    REQUIRE(entries != nullptr);
    REQUIRE(entries->is_array());
    CHECK(entries->as_array().size() == cfg.max_entries);
    for (const auto &entry : entries->as_array()) {
        const auto *tid = entry.find("target_identity");
        REQUIRE(tid != nullptr);
        CHECK(tid->as_string().size() <= SpeculationContextKey::kMaxIdentityLength);
    }

    // Verify LRU recency: touching key_hot keeps it resident while an older key is evicted.
    policy.reset();
    CHECK(policy.entry_count() == 0u);
    CHECK(policy.table_evictions() == 0u);

    AdaptiveSpeculationConfig small_cfg;
    small_cfg.max_entries = 2;
    small_cfg.min_samples = 2;
    AdaptiveSpeculationPolicy small(small_cfg);
    const SpeculationContextKey k1{"t1", "d1", ContextBand::short_ctx, ConcurrencyBand::solo,
                                   SpeculationWorkload::code_edit};
    const SpeculationContextKey k2{"t2", "d1", ContextBand::short_ctx, ConcurrencyBand::solo,
                                   SpeculationWorkload::code_edit};
    const SpeculationContextKey k3{"t3", "d1", ContextBand::short_ctx, ConcurrencyBand::solo,
                                   SpeculationWorkload::code_edit};

    for (int i = 0; i < 3; ++i) {
        SpeculationObservation o;
        o.context_key = k1;
        o.has_context_key = true;
        o.draft_n = 20;
        o.draft_n_accepted = 18;
        o.predicted_n = 20;
        o.decode_elapsed_ms = 160.0;
        small.observe(o);
    }
    CHECK(small.recommend(k1).draft_n_max == 5u);
    (void)small.recommend(k2);
    // Refresh k1 so k2 is now LRU.
    CHECK(small.recommend(k1).draft_n_max == 5u);
    // Inserting k3 evicts k2, preserving k1's learned draft_n_max == 5.
    (void)small.recommend(k3);
    CHECK(small.entry_count() == 2u);
    CHECK(small.table_evictions() == 1u);
    CHECK(small.recommend(k1).draft_n_max == 5u);
}

TEST_CASE("live workload classifiers genuinely reach all five buckets including prose") {
    // 1. tool_json via grammar, tools, or non-text response_format.
    {
        GenerateRequest req{"", "Write an essay about oceans.", {}};
        CHECK(classify_generate_workload(req, "root ::= \"ok\"") == SpeculationWorkload::tool_json);
        req.grammar = "root ::= \"ok\"";
        CHECK(classify_generate_workload(req, {}) == SpeculationWorkload::tool_json);
    }
    {
        ChatRequest req;
        req.messages = {{"user", "Write a short story about dawn."}};
        req.response_format_type = "json_object";
        CHECK(classify_chat_workload(req, {}) == SpeculationWorkload::tool_json);
    }

    // 2. reasoning via thinking or separate_reasoning.
    {
        ChatRequest req;
        req.messages = {{"user", "Write an essay on thermodynamics."}};
        req.separate_reasoning = true;
        CHECK(classify_chat_workload(req, {}) == SpeculationWorkload::reasoning);
    }

    // 3. code_edit beats prose when code/diff fences are present.
    {
        GenerateRequest req{"", "Write an article showing this patch:\n```cpp\nint main() { return 0; }\n```", {}};
        CHECK(classify_generate_workload(req, {}) == SpeculationWorkload::code_edit);
    }

    // 4. prose is genuinely reachable via explicit prose cues AND multi-sentence natural text.
    {
        GenerateRequest story{"", "Write a short story about a lighthouse keeper during a winter storm.", {}};
        CHECK(classify_generate_workload(story, {}) == SpeculationWorkload::prose);

        GenerateRequest essay{"", "Draft an essay explaining coastal navigation history.", {}};
        CHECK(classify_generate_workload(essay, {}) == SpeculationWorkload::prose);

        GenerateRequest passage{
            "",
            "The afternoon light drifted slowly across the harbor while fishing boats returned along the breakwater. "
            "Merchants packed crates of salt and cedar near the stone steps before dusk settled over the quay. "
            "Farther inland, lanterns flickered in the bakery windows as neighbors gathered along the cobbled street.",
            {}};
        CHECK(classify_generate_workload(passage, {}) == SpeculationWorkload::prose);

        ChatRequest chat_prose;
        chat_prose.messages = {
            {"system", "You are a helpful assistant."},
            {"user", "Proofread this blog post draft and tighten the narrative flow."},
        };
        CHECK(classify_chat_workload(chat_prose, {}) == SpeculationWorkload::prose);
    }

    // 5. general remains default for short plain prompts and code-like text without fences.
    {
        GenerateRequest plain{"", "Write a brief summary of cloud formations.", {}};
        CHECK(classify_generate_workload(plain, {}) == SpeculationWorkload::general);

        GenerateRequest code_no_fence{
            "",
            "Explain why std::vector<int> fn(int x) { return x + 1; } avoids heap allocation here. "
            "Also compare against struct Foo { int a; }; in C++20. "
            "Finally list the assembly emitted for this function.",
            {}};
        CHECK(classify_generate_workload(code_no_fence, {}) == SpeculationWorkload::general);
    }
}

}  // TEST_SUITE("llamaserver_adaptive_speculation")
