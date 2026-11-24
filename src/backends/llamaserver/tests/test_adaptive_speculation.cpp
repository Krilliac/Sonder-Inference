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
    CHECK(probe1.draft_n_max == cfg.probe_draft_n);
    CHECK(probe1.reason == "cooldown_probe");

    // While a probe is already in flight, concurrent recommend() stays paused.
    const auto concurrent = policy.recommend(SpeculationWorkload::prose);
    CHECK(concurrent.paused);
    CHECK_FALSE(concurrent.is_probe);
    CHECK(concurrent.draft_n_max == 0);
    CHECK(concurrent.reason == "paused_probe_in_flight");

    // If the probe fails (10% < 25%), bucket stays paused and restarts cooldown.
    const auto failed_probe = policy.observe(make_obs(SpeculationWorkload::prose, 20, 2));
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
    CHECK(probe2.draft_n_max == cfg.probe_draft_n);

    // If the probe succeeds (85% >= 25%), stale bad history is cleared and
    // speculation unpauses immediately at probe_draft_n.
    const auto recovered = policy.observe(make_obs(SpeculationWorkload::prose, 20, 17));
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
    CHECK(policy.recommend(SpeculationWorkload::reasoning).is_probe);
    policy.cancel_probe(SpeculationWorkload::reasoning);
    for (std::uint32_t i = 0; i < cfg.cooldown_requests; ++i) {
        const auto d = policy.recommend(SpeculationWorkload::reasoning);
        CHECK(d.paused);
        CHECK_FALSE(d.is_probe);
    }
    CHECK(policy.recommend(SpeculationWorkload::reasoning).is_probe);
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
    CHECK(policy.recommend(SpeculationWorkload::prose).is_probe);
    policy.observe(make_obs(SpeculationWorkload::prose, 20, 16, 40, 75.0));

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
                const auto upd = policy.observe(make_obs(workload, 20, accepted));
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

    // Concurrent non-probe request with 0 draft tokens must leave probe_in_flight intact.
    SpeculationObservation concurrent_non_probe = make_obs(SpeculationWorkload::prose, 0, 0);
    concurrent_non_probe.is_probe = false;
    policy.observe(concurrent_non_probe);
    CHECK(policy.recommend(SpeculationWorkload::prose).reason == "paused_probe_in_flight");

    // When the probe itself completes with 0 draft tokens (e.g. 1-token EOS),
    // the probe slot must release and restart cooldown rather than wedging permanently.
    SpeculationObservation empty_probe = make_obs(SpeculationWorkload::prose, 0, 0);
    empty_probe.is_probe = true;
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
    CHECK(next_probe.reason == "cooldown_probe");
}

}  // TEST_SUITE("llamaserver_adaptive_speculation")
