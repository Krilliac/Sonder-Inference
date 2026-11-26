// Standalone deterministic unit tests for latency-target-aware physical batch
// planning, StepCostEstimator, anti-starvation aging, priority ordering, and
// Session -> BackendModel priority/latency metadata forwarding.
// Compiled in default ci-windows / ci-linux builds without llama.cpp linkage.
#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "batch_scheduler.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/session.hpp"

namespace {

namespace lc = sonder::backends::llamacpp;
namespace si = sonder::inference;

lc::BatchSlotPlanView MakeSlot(std::size_t idx, lc::BatchSlotPhase phase, std::size_t prompt_len,
                               std::size_t prefill_offset, std::int64_t pos, si::RequestPriority priority,
                               std::optional<double> target_itl_ms = std::nullopt,
                               std::optional<double> target_ttft_ms = std::nullopt, std::int64_t enqueue_us = 0,
                               std::int64_t last_service_us = 0, std::int64_t kv_ctx = 0) {
    lc::BatchSlotPlanView s;
    s.slot_index = idx;
    s.seq_id = static_cast<std::int32_t>(idx);
    s.phase = phase;
    s.prompt_len = prompt_len;
    s.prefill_offset = prefill_offset;
    s.pos = pos;
    s.cancelled = false;
    s.priority = priority;
    s.target_ttft_ms = target_ttft_ms;
    s.target_itl_ms = target_itl_ms;
    s.enqueue_time_us = enqueue_us;
    s.last_service_time_us = last_service_us;
    s.kv_context_tokens = kv_ctx;
    return s;
}

class RecordingModel final : public si::BackendModel {
public:
    explicit RecordingModel(bool native_chat) : native_chat_(native_chat) {
        desc_.name = "rec:tiny";
        desc_.backend = "recording";
        desc_.format = "mock";
        desc_.context_length = 4096;
    }

    [[nodiscard]] const si::ModelDescriptor& descriptor() const override { return desc_; }
    [[nodiscard]] bool has_native_chat() const override { return native_chat_; }

    si::Result<si::GenerateStats> generate(const si::GenerateRequest& request, const si::CancellationToken&,
                                           const si::TokenCallback& on_chunk) override {
        last_generate = request;
        if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
        si::GenerateStats st;
        st.prompt_tokens = 4;
        st.completion_tokens = 1;
        st.chunks = 1;
        st.stop_reason = si::StopReason::end_of_sequence;
        return st;
    }

    si::Result<si::GenerateStats> chat(const si::ChatRequest& request, const si::CancellationToken& cancel,
                                       const si::TokenCallback& on_chunk) override {
        if (!native_chat_) {
            return si::BackendModel::chat(request, cancel, on_chunk);
        }
        last_chat = request;
        if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
        si::GenerateStats st;
        st.prompt_tokens = 4;
        st.completion_tokens = 1;
        st.chunks = 1;
        st.stop_reason = si::StopReason::end_of_sequence;
        return st;
    }

    std::optional<si::GenerateRequest> last_generate;
    std::optional<si::ChatRequest> last_chat;

private:
    bool native_chat_ = false;
    si::ModelDescriptor desc_;
};

class RecordingBackend final : public si::Backend {
public:
    explicit RecordingBackend(std::shared_ptr<RecordingModel> model) : model_(std::move(model)) {}
    [[nodiscard]] std::string name() const override { return "recording"; }
    [[nodiscard]] std::string description() const override { return "recording test backend"; }
    [[nodiscard]] si::BackendCapabilities capabilities() const override {
        si::BackendCapabilities caps;
        caps.add(si::Capability::streaming);
        return caps;
    }
    si::Result<std::string> probe() override { return std::string("recording"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model_->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::shared_ptr<si::BackendModel>(model_);
    }

private:
    std::shared_ptr<RecordingModel> model_;
};

}  // namespace

TEST_CASE("batch scheduler: fallback to safe bounded defaults before enough step-cost samples are measured") {
    lc::StepCostEstimator est(lc::StepCostEstimatorConfig{/*ema_alpha=*/0.25, /*min_samples=*/3});
    CHECK_FALSE(est.IsCalibrated());
    CHECK(est.total_samples() == 0u);

    // Invalid observations (negative, NaN, Inf, zero tokens) are ignored safely.
    est.Observe({1, 0, 1024, -5.0});
    est.Observe({1, 0, 1024, std::numeric_limits<double>::quiet_NaN()});
    est.Observe({1, 0, 1024, std::numeric_limits<double>::infinity()});
    est.Observe({0, 0, 1024, 10.0});
    CHECK(est.total_samples() == 0u);
    CHECK_FALSE(est.IsCalibrated());

    // Foreground decode with tight 25ms ITL target + background prefill of 1000 tokens.
    const std::vector<lc::BatchSlotPlanView> slots{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 100, 100, 200, si::RequestPriority::interactive, /*itl=*/25.0),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 1000, 0, 0, si::RequestPriority::background),
    };
    lc::BatchPlannerConfig cfg;
    cfg.default_mixed_prefill_tokens = 256;

    // 0, 1, and 2 samples (< min_samples=3) all use the safe bounded 256-token default.
    for (int sample = 0; sample < 3; ++sample) {
        if (sample > 0) {
            est.Observe({1, 128, 200, 80.0});  // even high latency waits until calibrated
        }
        CHECK_FALSE(est.IsCalibrated());
        std::size_t rr = 0;
        lc::AdaptiveBudgetDecision dec;
        const lc::PlannedBatchStep step = lc::PlanLatencyAwareBatchStep(
            slots, /*mixed_token_budget=*/512, /*prefill_only_token_budget=*/1024, /*output_budget=*/4, rr,
            /*now_us=*/1000, est, cfg, &dec);
        CHECK(dec.used_fallback_default);
        CHECK_FALSE(dec.used_cost_estimator);
        CHECK(dec.prefill_token_budget == 256);
        CHECK(dec.total_token_budget == 257);
        CHECK(step.decode_slots == 1u);
        CHECK(step.prefill_slots == 1u);
        REQUIRE(step.items.size() == 2u);
        CHECK(step.items[1].token_count == 256);
    }

    // 3rd sample calibrates the estimator -> immediately switches off fallback and shrinks prefill.
    est.Observe({1, 128, 200, 80.0});
    CHECK(est.IsCalibrated());
    std::size_t rr = 0;
    lc::AdaptiveBudgetDecision dec;
    const lc::PlannedBatchStep calibrated = lc::PlanLatencyAwareBatchStep(
        slots, /*mixed_token_budget=*/512, /*prefill_only_token_budget=*/1024, /*output_budget=*/4, rr,
        /*now_us=*/1000, est, cfg, &dec);
    CHECK(dec.used_cost_estimator);
    CHECK_FALSE(dec.used_fallback_default);
    CHECK(dec.prefill_token_budget < 256);
    CHECK(calibrated.decode_slots == 1u);
}

TEST_CASE("batch scheduler: tight foreground ITL target shrinks mixed prefill chunk when measured step cost is high") {
    // Ground-truth latency model: decode_base(1, 2048) = 12ms, prefill = 0.50 ms/tok.
    lc::StepCostEstimator est(lc::StepCostEstimatorConfig{/*ema_alpha=*/0.25, /*min_samples=*/3});
    est.Observe({1, 0, 2048, 12.0});
    est.Observe({1, 0, 2048, 12.0});
    est.Observe({1, 0, 2048, 12.0});
    est.Observe({1, 100, 2048, 62.0});  // +50ms for 100 prefill tokens => 0.50 ms/tok
    REQUIRE(est.IsCalibrated());
    REQUIRE(est.HasPrefillSamples());
    CHECK(est.PredictStepMs(1, 0, 2048) == doctest::Approx(12.0).epsilon(0.05));
    CHECK(est.ms_per_prefill_token() == doctest::Approx(0.50).epsilon(0.02));
    CHECK(est.PredictStepMs(1, 40, 2048) == doctest::Approx(32.0).epsilon(0.1));

    lc::BatchPlannerConfig cfg;
    cfg.itl_safety_factor = 1.0;  // exact analytic check: headroom = target - 12ms

    // Tight 30ms foreground ITL target => 18ms headroom / 0.50 ms/tok = 36 tokens!
    const std::vector<lc::BatchSlotPlanView> tight_slots{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 2048, 2048, 2048, si::RequestPriority::interactive, /*itl=*/30.0,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000, /*kv=*/2048),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 2000, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000, /*kv=*/0),
    };
    std::size_t rr = 0;
    lc::AdaptiveBudgetDecision tight_dec;
    const lc::PlannedBatchStep tight_step = lc::PlanLatencyAwareBatchStep(
        tight_slots, /*mixed_token_budget=*/512, /*prefill_only_token_budget=*/1024, /*output_budget=*/4, rr,
        /*now_us=*/20000, est, cfg, &tight_dec);
    CHECK(tight_dec.used_cost_estimator);
    CHECK(tight_dec.tightest_itl_ms == std::optional<double>(30.0));
    CHECK(tight_dec.prefill_token_budget == 36);
    CHECK(tight_dec.predicted_step_ms <= 30.01);
    REQUIRE(tight_step.items.size() == 2u);
    CHECK_FALSE(tight_step.items[0].is_prefill);
    CHECK(tight_step.items[1].is_prefill);
    CHECK(tight_step.items[1].token_count == 36);

    // Multiple foreground decoders: tightest ITL (22ms vs 80ms) governs the chunk size!
    // At 22ms: headroom <= 10ms => <= 20 prefill tokens (strictly smaller than 36).
    const std::vector<lc::BatchSlotPlanView> multi_fg{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 1024, 1024, 1024, si::RequestPriority::interactive, /*itl=*/80.0,
                 std::nullopt, 1000, 1000, 1024),
        MakeSlot(1, lc::BatchSlotPhase::kDecode, 1024, 1024, 1024, si::RequestPriority::interactive, /*itl=*/22.0,
                 std::nullopt, 1000, 1000, 1024),
        MakeSlot(2, lc::BatchSlotPhase::kPrefill, 2000, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, 1000, 1000, 0),
    };
    lc::AdaptiveBudgetDecision multi_dec;
    (void)lc::PlanLatencyAwareBatchStep(multi_fg, 512, 1024, 4, rr, 20000, est, cfg, &multi_dec);
    CHECK(multi_dec.tightest_itl_ms == std::optional<double>(22.0));
    CHECK(multi_dec.prefill_token_budget < tight_dec.prefill_token_budget);

    // Low step-cost GPU: decode=4ms, prefill=0.05 ms/tok => 30ms target allows 511 tokens (capped by ubatch)!
    lc::StepCostEstimator fast_est(lc::StepCostEstimatorConfig{0.25, 3});
    fast_est.Observe({1, 0, 2048, 4.0});
    fast_est.Observe({1, 0, 2048, 4.0});
    fast_est.Observe({1, 0, 2048, 4.0});
    fast_est.Observe({1, 200, 2048, 14.0});  // 0.05 ms/tok
    lc::AdaptiveBudgetDecision fast_dec;
    const lc::PlannedBatchStep fast_step =
        lc::PlanLatencyAwareBatchStep(tight_slots, 512, 1024, 4, rr, 20000, fast_est, cfg, &fast_dec);
    CHECK(fast_dec.prefill_token_budget == 511);  // 512 - 1 decode token
    CHECK(fast_step.total_tokens == 512);

    // KV-context sensitivity: observing higher step cost at 32K KV shrinks prefill budget vs 2K KV.
    lc::StepCostEstimator kv_est(lc::StepCostEstimatorConfig{0.35, 3});
    for (int i = 0; i < 6; ++i) {
        kv_est.Observe({1, 0, 2048, 8.0});
        kv_est.Observe({1, 0, 32768, 38.0});
        kv_est.Observe({1, 64, 2048, 40.0});
    }
    CHECK(kv_est.ms_per_kv_ktoken() > 0.2);
    CHECK(kv_est.PredictStepMs(1, 0, 32768) > kv_est.PredictStepMs(1, 0, 2048) + 10.0);

    const std::vector<lc::BatchSlotPlanView> short_kv_slots{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 2048, 2048, 2048, si::RequestPriority::interactive, 50.0,
                 std::nullopt, 1000, 1000, 2048),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 1000, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, 1000, 1000, 0),
    };
    const std::vector<lc::BatchSlotPlanView> long_kv_slots{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 32768, 32768, 32768, si::RequestPriority::interactive, 50.0,
                 std::nullopt, 1000, 1000, 32768),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 1000, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, 1000, 1000, 0),
    };
    lc::AdaptiveBudgetDecision short_dec;
    lc::AdaptiveBudgetDecision long_dec;
    (void)lc::PlanLatencyAwareBatchStep(short_kv_slots, 512, 1024, 4, rr, 20000, kv_est, cfg, &short_dec);
    (void)lc::PlanLatencyAwareBatchStep(long_kv_slots, 512, 1024, 4, rr, 20000, kv_est, cfg, &long_dec);
    CHECK(long_dec.prefill_token_budget < short_dec.prefill_token_budget);
}

TEST_CASE("batch scheduler: higher-priority prefill is ordered ahead of background prefill") {
    // Background slot sits at index 0, normal subagent at index 1, foreground interactive at index 2.
    const std::vector<lc::BatchSlotPlanView> three_tiers{
        MakeSlot(0, lc::BatchSlotPhase::kPrefill, 400, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 400, 0, 0, si::RequestPriority::subagent, std::nullopt,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000),
        MakeSlot(2, lc::BatchSlotPhase::kPrefill, 400, 0, 0, si::RequestPriority::interactive, std::nullopt,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000),
    };
    std::size_t rr = 0;
    const lc::BatchPlannerConfig cfg{};

    // Tight budget (100 tokens): foreground interactive prefill gets all 100 tokens ahead of normal & background.
    {
        const lc::PlannedBatchStep step =
            lc::PlanContinuousBatchStep(three_tiers, /*token_budget=*/100, /*output_budget=*/4, rr, /*now_us=*/5000,
                                        cfg);
        REQUIRE(step.items.size() == 1u);
        CHECK(step.items[0].slot_index == 2u);
        CHECK(step.items[0].token_count == 100);
    }

    // Generous budget (550 tokens): interactive finishes all 400 first, leftover 150 goes to subagent,
    // and emitted item order is strictly interactive -> subagent!
    {
        rr = 0;
        const lc::PlannedBatchStep step =
            lc::PlanContinuousBatchStep(three_tiers, /*token_budget=*/550, /*output_budget=*/4, rr, /*now_us=*/5000,
                                        cfg);
        REQUIRE(step.items.size() == 2u);
        CHECK(step.items[0].slot_index == 2u);
        CHECK(step.items[0].token_count == 400);
        CHECK(step.items[0].request_logits);
        CHECK(step.items[1].slot_index == 1u);
        CHECK(step.items[1].token_count == 150);
        CHECK_FALSE(step.items[1].request_logits);
    }

    // Pending-queue admission prefers interactive over queued background, while preserving
    // exclusive barrier at head of queue.
    {
        const std::vector<lc::PendingQueueSlotView> q{
            {si::RequestPriority::background, /*enqueue_time_us=*/1000, std::nullopt, false},
            {si::RequestPriority::interactive, /*enqueue_time_us=*/2000, std::nullopt, false},
        };
        const auto pick = lc::SelectNextPendingRequest(q, /*has_active_slots=*/true, /*now_us=*/5000, cfg);
        REQUIRE(pick.has_value());
        CHECK(*pick == 1u);

        // Once background waits past starvation_threshold_ms (500ms), aging elevates it ahead of fresh interactive!
        const auto aged_pick =
            lc::SelectNextPendingRequest(q, /*has_active_slots=*/true, /*now_us=*/600000, cfg);
        REQUIRE(aged_pick.has_value());
        CHECK(*aged_pick == 0u);

        // Exclusive barrier at index 0 blocks admission while active slots exist, admits when drained.
        const std::vector<lc::PendingQueueSlotView> barrier_q{
            {si::RequestPriority::background, 1000, std::nullopt, true},
            {si::RequestPriority::interactive, 2000, std::nullopt, false},
        };
        CHECK_FALSE(lc::SelectNextPendingRequest(barrier_q, /*has_active_slots=*/true, 5000, cfg).has_value());
        const auto drained_pick = lc::SelectNextPendingRequest(barrier_q, /*has_active_slots=*/false, 5000, cfg);
        REQUIRE(drained_pick.has_value());
        CHECK(*drained_pick == 0u);
    }
}

TEST_CASE("batch scheduler: anti-starvation aging elevates long-waiting background prefills for steady progress") {
    // 1. Under tight foreground decode ITL where decode alone saturates the target (adaptive prefill == 0),
    //    anti-starvation periodically grants a bounded floor so background completes instead of stalling forever!
    lc::StepCostEstimator est(lc::StepCostEstimatorConfig{0.25, 3});
    est.Observe({1, 0, 4096, 25.0});
    est.Observe({1, 0, 4096, 25.0});
    est.Observe({1, 0, 4096, 25.0});
    est.Observe({1, 64, 4096, 57.0});  // 0.50 ms/tok prefill

    lc::BatchPlannerConfig cfg;
    cfg.itl_safety_factor = 1.0;
    cfg.aging_interval_ms = 200.0;
    cfg.starvation_threshold_ms = 400.0;
    cfg.starvation_prefill_tokens = 32;

    std::vector<lc::BatchSlotPlanView> slots{
        MakeSlot(0, lc::BatchSlotPhase::kDecode, 4096, 4096, 4096, si::RequestPriority::interactive, /*itl=*/20.0,
                 std::nullopt, /*enq=*/1000, /*svc=*/1000, /*kv=*/4096),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 128, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, /*enq=*/1000, /*svc=*/0, /*kv=*/0),
        MakeSlot(2, lc::BatchSlotPhase::kPrefill, 500, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, /*enq=*/450000, /*svc=*/0, /*kv=*/0),  // brand-new, unstarved
    };

    std::size_t rr = 0;
    // Before starvation threshold (t = 100ms): prefill budget is 0 (pure decode protects foreground ITL).
    {
        lc::AdaptiveBudgetDecision dec;
        const lc::PlannedBatchStep step =
            lc::PlanLatencyAwareBatchStep(slots, 512, 1024, 4, rr, /*now_us=*/100000, est, cfg, &dec);
        CHECK(dec.prefill_token_budget == 0);
        CHECK_FALSE(dec.starvation_floor_applied);
        CHECK(step.decode_slots == 1u);
        CHECK(step.prefill_slots == 0u);
    }

    // Past starvation threshold for slot 1 (t = 450ms, wait = 449ms >= 400ms), while slot 2 just arrived:
    // ONLY starved slot 1 gets the 32-token anti-starvation floor; unstarved slot 2 gets 0!
    {
        lc::AdaptiveBudgetDecision dec;
        const lc::PlannedBatchStep step =
            lc::PlanLatencyAwareBatchStep(slots, 512, 1024, 4, rr, /*now_us=*/450000, est, cfg, &dec);
        CHECK(dec.starvation_floor_applied);
        CHECK(dec.starved_prefill_slots == 1u);
        CHECK(dec.prefill_token_budget == 32);
        REQUIRE(step.items.size() == 2u);
        CHECK(step.items[0].slot_index == 0u);  // foreground decode preserved
        CHECK(step.items[1].slot_index == 1u);  // starved background advances!
        CHECK(step.items[1].token_count == 32);
    }

    // 2. Multi-step simulation: continuous foreground prefill load + waiting background prefill.
    //    Background prefill MUST make steady progress and finish in bounded steps!
    {
        lc::BatchSlotPlanView fg =
            MakeSlot(0, lc::BatchSlotPhase::kPrefill, 2000, 0, 0, si::RequestPriority::interactive, std::nullopt,
                     std::nullopt, /*enq=*/1000, /*svc=*/1000);
        lc::BatchSlotPlanView bg =
            MakeSlot(1, lc::BatchSlotPhase::kPrefill, 160, 0, 0, si::RequestPriority::background, std::nullopt,
                     std::nullopt, /*enq=*/1000, /*svc=*/1000);

        std::int64_t clock_us = 1000;
        std::size_t bg_services = 0;
        std::size_t fg_services = 0;
        rr = 0;
        for (int step_i = 0; step_i < 80 && bg.prefill_offset < bg.prompt_len; ++step_i) {
            clock_us += 50000;  // 50ms per step
            std::vector<lc::BatchSlotPlanView> active{fg, bg};
            const lc::PlannedBatchStep p =
                lc::PlanContinuousBatchStep(active, /*token_budget=*/96, /*output_budget=*/4, rr, clock_us, cfg);
            REQUIRE(p.total_tokens > 0);
            for (const auto& item : p.items) {
                if (item.slot_index == 0u) {
                    fg.prefill_offset += static_cast<std::size_t>(item.token_count);
                    fg.pos += item.token_count;
                    fg.last_service_time_us = clock_us;
                    ++fg_services;
                } else if (item.slot_index == 1u) {
                    bg.prefill_offset += static_cast<std::size_t>(item.token_count);
                    bg.pos += item.token_count;
                    bg.last_service_time_us = clock_us;
                    ++bg_services;
                }
            }
        }
        // Background completed its entire 160-token prompt even though foreground still had prompt left!
        CHECK(bg.prefill_offset == bg.prompt_len);
        CHECK(bg_services >= 1u);
        CHECK(fg_services > bg_services);  // foreground still received the majority of bandwidth
    }
}

TEST_CASE("batch scheduler: zero-decode prefill-only steps use full configured microbatch throughput") {
    // Calibrate estimator with high per-token cost that would tightly cap a mixed step.
    lc::StepCostEstimator est(lc::StepCostEstimatorConfig{0.25, 3});
    est.Observe({1, 0, 4096, 30.0});
    est.Observe({1, 0, 4096, 30.0});
    est.Observe({1, 0, 4096, 30.0});
    est.Observe({1, 64, 4096, 94.0});  // 1.0 ms/tok prefill

    lc::BatchPlannerConfig cfg;
    cfg.default_foreground_itl_ms = 25.0;
    cfg.default_mixed_prefill_tokens = 256;

    const std::vector<lc::BatchSlotPlanView> prefill_only{
        MakeSlot(0, lc::BatchSlotPhase::kPrefill, 4096, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, 1000, 1000),
        MakeSlot(1, lc::BatchSlotPhase::kPrefill, 4096, 0, 0, si::RequestPriority::background, std::nullopt,
                 std::nullopt, 1000, 1000),
    };

    std::size_t rr = 0;
    lc::AdaptiveBudgetDecision dec;
    const lc::PlannedBatchStep step = lc::PlanLatencyAwareBatchStep(
        prefill_only, /*mixed_token_budget=*/512, /*prefill_only_token_budget=*/2048, /*output_budget=*/4, rr,
        /*now_us=*/10000, est, cfg, &dec);
    CHECK(dec.decode_slots == 0u);
    CHECK(dec.prefill_slots == 2u);
    CHECK(dec.total_token_budget == 2048);
    CHECK(dec.prefill_token_budget == 2048);
    CHECK(step.total_tokens == 2048);
    REQUIRE(step.items.size() == 2u);
    CHECK(step.items[0].token_count == 1024);
    CHECK(step.items[1].token_count == 1024);
}

TEST_CASE("batch scheduler: Session forwards priority_class and latency targets to BackendModel and validates bounds") {
    auto model = std::make_shared<RecordingModel>(/*native_chat=*/true);
    auto backend = std::make_shared<RecordingBackend>(model);

    si::EngineOptions eo;
    eo.sample_devices_on_start = false;
    eo.sample_gpu_devices = false;
    si::Engine engine(eo);
    REQUIRE(engine.register_backend(backend).ok());
    auto loaded = engine.load_model("recording", {"rec:tiny", "cpu:0"});
    REQUIRE(loaded.ok());

    // 1. Explicit priority_class + target_ttft_ms + target_itl_ms forward through generate() and chat().
    {
        si::SessionOptions so;
        so.workload = si::WorkloadClass::background_indexing;
        auto sess = engine.create_session(loaded.value(), so);
        REQUIRE(sess.ok());

        si::RequestOptions ro;
        ro.priority_class = si::RequestPriority::interactive;
        ro.target_ttft_ms = 120.0;
        ro.target_itl_ms = 35.0;

        auto gen = sess.value()->generate("hello", {}, std::nullopt, ro);
        REQUIRE(gen.ok());
        REQUIRE(model->last_generate.has_value());
        CHECK(model->last_generate->priority_class == std::optional<si::RequestPriority>(si::RequestPriority::interactive));
        CHECK(model->last_generate->target_ttft_ms == std::optional<double>(120.0));
        CHECK(model->last_generate->target_itl_ms == std::optional<double>(35.0));

        auto ch = sess.value()->chat({si::ChatMessage{"user", "hi"}}, {}, std::nullopt, ro);
        REQUIRE(ch.ok());
        REQUIRE(model->last_chat.has_value());
        CHECK(model->last_chat->priority_class == std::optional<si::RequestPriority>(si::RequestPriority::interactive));
        CHECK(model->last_chat->target_ttft_ms == std::optional<double>(120.0));
        CHECK(model->last_chat->target_itl_ms == std::optional<double>(35.0));

        // Invalid latency targets fail fast with invalid_argument.
        ro.target_itl_ms = 0.0;
        CHECK(sess.value()->generate("x", {}, std::nullopt, ro).status().code() == si::ErrorCode::invalid_argument);
        ro.target_itl_ms = -10.0;
        CHECK(sess.value()->generate("x", {}, std::nullopt, ro).status().code() == si::ErrorCode::invalid_argument);
        ro.target_itl_ms = std::numeric_limits<double>::quiet_NaN();
        CHECK(sess.value()->generate("x", {}, std::nullopt, ro).status().code() == si::ErrorCode::invalid_argument);
        ro.target_itl_ms = std::nullopt;
        ro.target_ttft_ms = std::numeric_limits<double>::infinity();
        CHECK(sess.value()->generate("x", {}, std::nullopt, ro).status().code() == si::ErrorCode::invalid_argument);
    }

    // 2. Omitting RequestOptions::priority_class derives priority from SessionOptions::workload.
    {
        si::SessionOptions bg_so;
        bg_so.workload = si::WorkloadClass::background_indexing;
        auto bg_sess = engine.create_session(loaded.value(), bg_so);
        REQUIRE(bg_sess.ok());
        REQUIRE(bg_sess.value()->generate("bg").ok());
        REQUIRE(model->last_generate.has_value());
        CHECK(model->last_generate->priority_class == std::optional<si::RequestPriority>(si::RequestPriority::background));

        si::SessionOptions fg_so;
        fg_so.workload = si::WorkloadClass::interactive_user;
        auto fg_sess = engine.create_session(loaded.value(), fg_so);
        REQUIRE(fg_sess.ok());
        REQUIRE(fg_sess.value()->generate("fg").ok());
        REQUIRE(model->last_generate.has_value());
        CHECK(model->last_generate->priority_class == std::optional<si::RequestPriority>(si::RequestPriority::interactive));
    }

    // 3. Default BackendModel::chat() preserves priority_class & latency targets when delegating to generate().
    {
        auto non_native = std::make_shared<RecordingModel>(/*native_chat=*/false);
        si::ChatRequest cr;
        cr.messages = {si::ChatMessage{"user", "ping"}};
        cr.priority_class = si::RequestPriority::background;
        cr.target_ttft_ms = 250.0;
        cr.target_itl_ms = 80.0;
        REQUIRE(non_native->chat(cr, {}, {}).ok());
        REQUIRE(non_native->last_generate.has_value());
        CHECK(non_native->last_generate->priority_class ==
              std::optional<si::RequestPriority>(si::RequestPriority::background));
        CHECK(non_native->last_generate->target_ttft_ms == std::optional<double>(250.0));
        CHECK(non_native->last_generate->target_itl_ms == std::optional<double>(80.0));
    }
}
