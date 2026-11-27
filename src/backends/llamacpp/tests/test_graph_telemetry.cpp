// Unit tests for deep llama.cpp/GGML graph telemetry (layer & operator aggregation,
// level gating, boundedness, Observatory event shapes, and backend capability/hooks).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "graph_telemetry.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/telemetry.hpp"

namespace {

namespace lc = sonder::backends::llamacpp;
namespace si = sonder::inference;

struct FakeClock {
    std::uint64_t now_ns = 1'000'000;
    static std::uint64_t Read(void* ctx) noexcept {
        return static_cast<const FakeClock*>(ctx)->now_ns;
    }
};

struct SyntheticNode {
    const char* name;
    int op_id;
    const char* op_name;
    const char* device;
    std::uint64_t duration_ns;
};

void FeedNodes(lc::GraphTelemetryAggregator& agg, FakeClock& clock, const std::vector<SyntheticNode>& nodes) {
    for (const auto& n : nodes) {
        const bool need_after = agg.OnNodeAsk(n.name, n.op_id, n.op_name, n.device);
        clock.now_ns += n.duration_ns;
        if (need_after) {
            CHECK(agg.OnNodeAfter(n.name, n.op_id));
        }
    }
}

const si::CapabilityStatus* FindCap(const std::vector<si::CapabilityStatusEntry>& entries, std::string_view key) {
    for (const auto& e : entries) {
        if (e.first == key) return &e.second;
    }
    return nullptr;
}

}  // namespace

TEST_CASE("graph_telemetry: ParseLayerIndex handles llama.cpp v1.0.0 tensor naming") {
    // Standard llama.cpp v1.0.0 `ggml_format_name(cur, "%s-%d", name, il)` names.
    CHECK(lc::ParseLayerIndex("norm-0") == 0);
    CHECK(lc::ParseLayerIndex("Qcur-0") == 0);
    CHECK(lc::ParseLayerIndex("Kcur-12") == 12);
    CHECK(lc::ParseLayerIndex("ffn_down-31") == 31);
    CHECK(lc::ParseLayerIndex("ffn_moe_logits-7") == 7);
    CHECK(lc::ParseLayerIndex("l_out-127") == 127);

    // Trailing ggml view/reshape/cont tags from ggml.c (`%s (view)`, `%s (reshaped)`, `%s (cont)`).
    CHECK(lc::ParseLayerIndex("Qcur-0 (view)") == 0);
    CHECK(lc::ParseLayerIndex("Kcur-12 (reshaped)") == 12);
    CHECK(lc::ParseLayerIndex("Vcur-5 (cont)") == 5);
    CHECK(lc::ParseLayerIndex("Qcur-3 (view) (reshaped)") == 3);

    // Non-layer tensors and malformed names return -1.
    CHECK(lc::ParseLayerIndex("") == -1);
    CHECK(lc::ParseLayerIndex("inp_embd") == -1);
    CHECK(lc::ParseLayerIndex("result_norm") == -1);
    CHECK(lc::ParseLayerIndex("result_output") == -1);
    CHECK(lc::ParseLayerIndex("logits_seq_0_0") == -1);
    CHECK(lc::ParseLayerIndex("-0") == -1);
    CHECK(lc::ParseLayerIndex("foo-") == -1);
    CHECK(lc::ParseLayerIndex("foo-bar") == -1);
    CHECK(lc::ParseLayerIndex("foo-12a") == -1);
    CHECK(lc::ParseLayerIndex("foo--3") == -1);
    CHECK(lc::ParseLayerIndex("foo-999999999999999") == -1);
}

TEST_CASE("graph_telemetry: level gating disables callback and emits nothing below deep") {
    CHECK_FALSE(lc::ShouldInstallGraphCallback(si::TelemetryLevel::off));
    CHECK_FALSE(lc::ShouldInstallGraphCallback(si::TelemetryLevel::metrics));
    CHECK_FALSE(lc::ShouldInstallGraphCallback(si::TelemetryLevel::standard));
    CHECK(lc::ShouldInstallGraphCallback(si::TelemetryLevel::deep));

    // Hot-path callback methods must be noexcept.
    static_assert(noexcept(std::declval<lc::GraphTelemetryAggregator&>().OnNodeAsk("Qcur-0", 1, "MUL_MAT", "CUDA0")));
    static_assert(noexcept(std::declval<lc::GraphTelemetryAggregator&>().OnNodeAfter("Qcur-0", 1)));

    for (const auto lvl : {si::TelemetryLevel::off, si::TelemetryLevel::metrics, si::TelemetryLevel::standard}) {
        lc::GraphTelemetryAggregator agg;
        lc::GraphTelemetryConfig cfg;
        cfg.level = lvl;
        cfg.sample_every_n = 1;
        cfg.collect_timings = true;
        agg.Configure(cfg);

        CHECK_FALSE(agg.Enabled());
        CHECK_FALSE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
        CHECK_FALSE(agg.ActiveForCurrentEval());
        CHECK_FALSE(agg.OnNodeAsk("Qcur-0", 28, "MUL_MAT", "CUDA0"));
        CHECK(agg.OnNodeAfter("Qcur-0", 28));  // never cancels graph compute
        lc::GraphEvalSummary summary;
        CHECK_FALSE(agg.EndEval(summary));
    }

    // Even given a populated summary, EmitGraphTelemetry emits nothing below deep.
    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    si::TelemetryOptions topts;
    topts.level = si::TelemetryLevel::deep;
    si::TelemetryBus bus(topts);
    bus.add_sink(sink);

    lc::GraphEvalSummary populated;
    populated.eval_index = 1;
    populated.total_nodes = 2;
    populated.layers.push_back(lc::GraphLayerSummary{0, 2, 2, 100'000, 1, "CPU", {"CPU"}});
    populated.operators.push_back(lc::GraphOpSummary{28, "MUL_MAT", 2, 2, 100'000, "CPU", {{"CPU", 2}}});

    si::TelemetryContext ctx;
    ctx.session_id = "sess-test";
    CHECK(lc::EmitGraphTelemetry(bus, ctx, populated, si::TelemetryLevel::off) == 0u);
    CHECK(lc::EmitGraphTelemetry(bus, ctx, populated, si::TelemetryLevel::metrics) == 0u);
    CHECK(lc::EmitGraphTelemetry(bus, ctx, populated, si::TelemetryLevel::standard) == 0u);
    bus.flush();
    CHECK(sink->lines().empty());
}

TEST_CASE("graph_telemetry: aggregates synthetic nodes per layer and per op with deterministic timings") {
    FakeClock clock;
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.sample_every_n = 1;
    cfg.collect_timings = true;
    cfg.clock_fn = &FakeClock::Read;
    cfg.clock_ctx = &clock;
    agg.Configure(cfg);

    REQUIRE(agg.Enabled());
    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kPrefill, 128));
    CHECK(agg.ActiveForCurrentEval());

    const std::vector<SyntheticNode> seq = {
        {"inp_embd", 39, "GET_ROWS", "CPU", 50'000},
        {"norm-0", 24, "RMS_NORM", "CUDA0", 100'000},
        {"Qcur-0", 28, "MUL_MAT", "CUDA0", 400'000},
        {"Qcur-0 (view)", 36, "VIEW", "CUDA0", 10'000},
        {"ffn_down-0", 28, "MUL_MAT", "CUDA0", 500'000},
        {"norm-2", 24, "RMS_NORM", "CUDA0", 120'000},
        {"ffn_moe_up-2", 29, "MUL_MAT_ID", "CPU", 800'000},
        {"result_norm", 24, "RMS_NORM", "CUDA0", 80'000},
        {"result_output", 28, "MUL_MAT", "CUDA0", 300'000},
    };
    FeedNodes(agg, clock, seq);

    lc::GraphEvalSummary summary;
    REQUIRE(agg.EndEval(summary));

    CHECK(summary.eval_index == 1u);
    CHECK(summary.sample_every_n == 1u);
    CHECK(summary.eval_kind == lc::GraphEvalKind::kPrefill);
    CHECK(summary.batch_tokens == 128);
    CHECK(summary.total_nodes == 9u);
    CHECK(summary.total_timed_nodes == 9u);
    CHECK(summary.total_duration_ns == 2'360'000u);
    CHECK(summary.unlayered_nodes == 3u);
    CHECK(summary.unlayered_duration_ns == 430'000u);
    CHECK(summary.has_timings);

    // Layers sorted ascending: 0, 2.
    REQUIRE(summary.layers.size() == 2u);
    CHECK(summary.layers[0].layer_index == 0);
    CHECK(summary.layers[0].node_count == 4u);
    CHECK(summary.layers[0].timed_nodes == 4u);
    CHECK(summary.layers[0].duration_ns == 1'010'000u);
    CHECK(summary.layers[0].op_count == 3u);  // RMS_NORM, MUL_MAT, VIEW
    CHECK(summary.layers[0].primary_device == "CUDA0");
    REQUIRE(summary.layers[0].devices.size() == 1u);
    CHECK(summary.layers[0].devices[0] == "CUDA0");

    CHECK(summary.layers[1].layer_index == 2);
    CHECK(summary.layers[1].node_count == 2u);
    CHECK(summary.layers[1].timed_nodes == 2u);
    CHECK(summary.layers[1].duration_ns == 920'000u);
    CHECK(summary.layers[1].op_count == 2u);  // RMS_NORM, MUL_MAT_ID
    CHECK(summary.layers[1].devices.size() == 2u);

    // Operators sorted by duration_ns desc: MUL_MAT (1.2M), MUL_MAT_ID (0.8M), RMS_NORM (0.3M), GET_ROWS (0.05M), VIEW (0.01M).
    REQUIRE(summary.operators.size() == 5u);
    CHECK(summary.operators[0].op_type == "MUL_MAT");
    CHECK(summary.operators[0].op_id == 28);
    CHECK(summary.operators[0].node_count == 3u);
    CHECK(summary.operators[0].duration_ns == 1'200'000u);
    CHECK(summary.operators[0].primary_device == "CUDA0");

    CHECK(summary.operators[1].op_type == "MUL_MAT_ID");
    CHECK(summary.operators[1].node_count == 1u);
    CHECK(summary.operators[1].duration_ns == 800'000u);
    CHECK(summary.operators[1].primary_device == "CPU");

    CHECK(summary.operators[2].op_type == "RMS_NORM");
    CHECK(summary.operators[2].node_count == 3u);
    CHECK(summary.operators[2].duration_ns == 300'000u);

    // Second eval resets cleanly without bleeding state from eval 1.
    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    FeedNodes(agg, clock, {{"Qcur-5", 28, "MUL_MAT", "CUDA0", 250'000}});
    lc::GraphEvalSummary summary2;
    REQUIRE(agg.EndEval(summary2));
    CHECK(summary2.eval_index == 2u);
    CHECK(summary2.total_nodes == 1u);
    REQUIRE(summary2.layers.size() == 1u);
    CHECK(summary2.layers[0].layer_index == 5);
    CHECK(summary2.layers[0].duration_ns == 250'000u);
    REQUIRE(summary2.operators.size() == 1u);
    CHECK(summary2.operators[0].node_count == 1u);
}

TEST_CASE("graph_telemetry: counting-only mode avoids post-node sync and omits duration_ms") {
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.collect_timings = false;
    agg.Configure(cfg);

    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    // ask=true returns false so ggml batches the whole split without syncing.
    CHECK_FALSE(agg.OnNodeAsk("Qcur-0", 28, "MUL_MAT", "Vulkan0"));
    CHECK_FALSE(agg.OnNodeAsk("ffn_down-0", 28, "MUL_MAT", "Vulkan0"));

    lc::GraphEvalSummary summary;
    REQUIRE(agg.EndEval(summary));
    CHECK_FALSE(summary.has_timings);
    CHECK(summary.total_nodes == 2u);
    CHECK(summary.total_timed_nodes == 0u);
    REQUIRE(summary.layers.size() == 1u);
    CHECK(summary.layers[0].timed_nodes == 0u);

    const si::json::Object layer_attrs = lc::BuildLayerTimingAttributes(summary, summary.layers[0]);
    CHECK(layer_attrs.find("duration_ms") == nullptr);
    CHECK(layer_attrs.find("timed_nodes") == nullptr);
    CHECK(layer_attrs.find("node_count")->as_uint() == 2u);
    CHECK(layer_attrs.find("backend_device")->as_string() == "Vulkan0");

    const si::json::Object op_attrs = lc::BuildOperatorActivityAttributes(summary, summary.operators[0]);
    CHECK(op_attrs.find("duration_ms") == nullptr);
    CHECK(op_attrs.find("timed_nodes") == nullptr);
    CHECK(op_attrs.find("op_type")->as_string() == "MUL_MAT");
}

TEST_CASE("graph_telemetry: sampling every N evals skips intermediate evals") {
    FakeClock clock;
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.sample_every_n = 3;
    cfg.collect_timings = true;
    cfg.clock_fn = &FakeClock::Read;
    cfg.clock_ctx = &clock;
    agg.Configure(cfg);

    lc::GraphEvalSummary summary;
    // Eval 1: sampled.
    CHECK(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    FeedNodes(agg, clock, {{"Qcur-0", 28, "MUL_MAT", "CPU", 100'000}});
    CHECK(agg.EndEval(summary));
    CHECK(summary.eval_index == 1u);
    CHECK(summary.sample_every_n == 3u);

    // Eval 2: skipped.
    CHECK_FALSE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    CHECK_FALSE(agg.OnNodeAsk("Qcur-0", 28, "MUL_MAT", "CPU"));
    CHECK_FALSE(agg.EndEval(summary));

    // Eval 3: skipped.
    CHECK_FALSE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    CHECK_FALSE(agg.EndEval(summary));

    // Eval 4: sampled.
    CHECK(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    FeedNodes(agg, clock, {{"Qcur-1", 28, "MUL_MAT", "CPU", 200'000}});
    CHECK(agg.EndEval(summary));
    CHECK(summary.eval_index == 4u);
    REQUIRE(summary.layers.size() == 1u);
    CHECK(summary.layers[0].layer_index == 1);
}

TEST_CASE("graph_telemetry: bounded aggregation caps layers, ops, and devices safely") {
    FakeClock clock;
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.collect_timings = true;
    cfg.max_layers_per_eval = 2;
    cfg.max_ops_per_eval = 3;
    cfg.clock_fn = &FakeClock::Read;
    cfg.clock_ctx = &clock;
    agg.Configure(cfg);

    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kBatch, 64));

    // Out-of-range layer indices (>= kMaxTrackedLayers) and out-of-range op IDs.
    FeedNodes(agg, clock, {
        {"Qcur-0", 1, "DUP", "CUDA0", 10'000},
        {"Qcur-1", 2, "ADD", "CUDA0", 20'000},
        {"Qcur-2", 3, "SUB", "CUDA0", 30'000},
        {"Qcur-3", 4, "MUL", "CUDA0", 40'000},
        {"Qcur-4", 5, "DIV", "CUDA0", 50'000},
        {"Qcur-300", 6, "SQR", "CUDA0", 60'000},     // overflow layer (>= 256)
        {"Qcur-0", 500, "CUSTOM_X", "CUDA0", 70'000}, // overflow op (>= 128)
        {"Qcur-0", -5, "NEG_OP", "CUDA0", 15'000},    // negative op (< 0)
    });

    // Feed more distinct devices than kMaxTrackedDevices (16).
    std::vector<std::string> dev_names;
    for (int i = 0; i < 22; ++i) {
        dev_names.push_back("DEV" + std::to_string(i));
    }
    for (const auto& d : dev_names) {
        const bool need = agg.OnNodeAsk("Qcur-0", 1, "DUP", d.c_str());
        clock.now_ns += 1'000;
        if (need) CHECK(agg.OnNodeAfter("Qcur-0", 1));
    }

    lc::GraphEvalSummary summary;
    REQUIRE(agg.EndEval(summary));

    // Bounded to max_layers_per_eval == 2 and max_ops_per_eval == 3.
    CHECK(summary.layers.size() == 2u);
    CHECK(summary.truncated_layers == 3u);  // layers 0..4 => 5 layers, 2 kept, 3 truncated
    CHECK(summary.overflow_layer_nodes == 1u);
    CHECK(summary.overflow_layer_duration_ns == 60'000u);

    CHECK(summary.operators.size() == 3u);
    CHECK(summary.truncated_ops >= 3u);
    // Top-3 ops sorted by duration desc: unknown/overflow (85k), SQR (60k), DIV (50k).
    CHECK(summary.operators[0].duration_ns >= summary.operators[1].duration_ns);
    CHECK(summary.operators[1].duration_ns >= summary.operators[2].duration_ns);

    // Tracked devices bounded to kMaxTrackedDevices (16); excess counted as overflow.
    CHECK(summary.overflow_device_nodes == 7u);  // CUDA0 + 15 of DEV0..14 = 16; DEV15..21 = 7 overflow
}

TEST_CASE("graph_telemetry: Observatory envelope shapes for backend.layer.timing and backend.operator.activity") {
    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    si::TelemetryOptions topts;
    topts.level = si::TelemetryLevel::deep;
    si::TelemetryBus bus(topts);
    bus.add_sink(sink);

    FakeClock clock;
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.clock_fn = &FakeClock::Read;
    cfg.clock_ctx = &clock;
    agg.Configure(cfg);

    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    FeedNodes(agg, clock, {
        {"Qcur-0", 28, "MUL_MAT", "CUDA0", 1'500'000},
        {"attn_out-0", 72, "FLASH_ATTN_EXT", "CUDA0", 500'000},
        {"ffn_down-1", 28, "MUL_MAT", "CPU", 2'000'000},
    });
    lc::GraphEvalSummary summary;
    REQUIRE(agg.EndEval(summary));

    si::TelemetryContext ctx;
    ctx.session_id = "sess-45";
    ctx.request_id = "req-45";
    const std::size_t emitted = lc::EmitGraphTelemetry(bus, ctx, summary, si::TelemetryLevel::deep, "llamacpp");
    CHECK(emitted == 4u);  // 2 layers + 2 ops
    bus.flush();

    const auto lines = sink->lines();
    REQUIRE(lines.size() == 4u);

    int layer_events = 0;
    int op_events = 0;
    for (const auto& line : lines) {
        const auto parsed = si::json::parse(line);
        REQUIRE(parsed.ok());
        const auto& env = parsed.value();
        CHECK(env.find("schema")->as_string() == "sonder.observatory.event/1");
        CHECK(env.find("sampling")->find("level")->as_string() == "deep");
        const std::string etype = env.find("event_type")->as_string();
        const auto* attrs = env.find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("backend")->as_string() == "llamacpp");
        CHECK(attrs->find("eval_index")->as_uint() == 1u);
        CHECK(attrs->find("eval_kind")->as_string() == "decode");
        CHECK(attrs->find("batch_tokens")->as_int() == 1);

        if (etype == "backend.layer.timing") {
            ++layer_events;
            CHECK(attrs->find("layer_index") != nullptr);
            CHECK(attrs->find("node_count")->as_uint() >= 1u);
            CHECK(attrs->find("op_count")->as_uint() >= 1u);
            CHECK(attrs->find("duration_ms")->as_double() > 0.0);
            CHECK(attrs->find("timed_nodes")->as_uint() >= 1u);
            CHECK(attrs->find("backend_device") != nullptr);
            CHECK(attrs->find("devices")->is_array());
        } else if (etype == "backend.operator.activity") {
            ++op_events;
            CHECK(attrs->find("op_type") != nullptr);
            CHECK(attrs->find("op_id") != nullptr);
            CHECK(attrs->find("node_count")->as_uint() >= 1u);
            CHECK(attrs->find("total_eval_nodes")->as_uint() == 3u);
            CHECK(attrs->find("duration_ms")->as_double() > 0.0);
            CHECK(attrs->find("timed_nodes")->as_uint() >= 1u);
            CHECK(attrs->find("backend_device") != nullptr);
            CHECK(attrs->find("devices")->is_array());
        }
    }
    CHECK(layer_events == 2);
    CHECK(op_events == 2);

    // A bus configured at standard must drop deep events even if called with deep level.
    auto std_sink = std::make_shared<si::MemoryTelemetrySink>();
    si::TelemetryOptions std_opts;
    std_opts.level = si::TelemetryLevel::standard;
    si::TelemetryBus std_bus(std_opts);
    std_bus.add_sink(std_sink);
    CHECK(lc::EmitGraphTelemetry(std_bus, ctx, summary, si::TelemetryLevel::deep, "llamacpp") == 0u);
    std_bus.flush();
    CHECK(std_sink->lines().empty());
}

TEST_CASE("graph_telemetry: backend capability and context callback installation gate strictly on deep") {
    // 1. Default options (standard): layer_telemetry NOT advertised; cb_eval NOT installed.
    {
        auto be = si::make_llamacpp_backend();
        REQUIRE(be != nullptr);
        CHECK_FALSE(be->capabilities().has(si::Capability::layer_telemetry));
        const auto caps = be->capability_statuses();
        const auto* lt = FindCap(caps, "layer_telemetry");
        REQUIRE(lt != nullptr);
        CHECK(lt->adapter_supports == si::CapabilityState::yes);
        CHECK(lt->build_supports == si::CapabilityState::yes);
        CHECK(lt->configured == si::CapabilityState::no);
        CHECK(lt->observed_active == si::CapabilityState::no);

        lc::LoadOptions lo;
        CHECK_FALSE(lo.graph_telemetry_enabled);
        CHECK_FALSE(lc::LlamaCppBackend::ContextParamsFor(lo).has_eval_callback);
    }

    // 2. off / metrics / standard: layer_telemetry NOT advertised.
    for (const auto lvl : {si::TelemetryLevel::off, si::TelemetryLevel::metrics, si::TelemetryLevel::standard}) {
        si::LlamaCppBackendOptions opts;
        opts.telemetry_level = lvl;
        CHECK(si::validate_llamacpp_options(opts).ok());
        auto be = si::make_llamacpp_backend(opts);
        CHECK_FALSE(be->capabilities().has(si::Capability::layer_telemetry));
    }

    // 3. deep: layer_telemetry IS advertised, and LoadOptions with graph_telemetry_enabled installs cb_eval.
    {
        si::LlamaCppBackendOptions opts;
        opts.telemetry_level = si::TelemetryLevel::deep;
        opts.graph_telemetry_sample_every = 4;
        CHECK(si::validate_llamacpp_options(opts).ok());
        auto be = si::make_llamacpp_backend(opts);
        CHECK(be->capabilities().has(si::Capability::layer_telemetry));
        const auto cap_names = be->capabilities().names();
        CHECK(std::find(cap_names.begin(), cap_names.end(), "layer_telemetry") != cap_names.end());

        const auto caps = be->capability_statuses();
        const auto* lt = FindCap(caps, "layer_telemetry");
        REQUIRE(lt != nullptr);
        CHECK(lt->configured == si::CapabilityState::yes);
        CHECK(lt->observed_active == si::CapabilityState::unknown);

        lc::LoadOptions lo;
        lo.graph_telemetry_enabled = true;
        CHECK(lc::LlamaCppBackend::ContextParamsFor(lo).has_eval_callback);
    }

    // 4. Invalid sample_every == 0 is rejected by validate_llamacpp_options.
    {
        si::LlamaCppBackendOptions bad;
        bad.graph_telemetry_sample_every = 0;
        CHECK(si::validate_llamacpp_options(bad).code() == si::ErrorCode::invalid_argument);
    }
}

TEST_CASE("graph_telemetry: empty evals, CancelEval, Disable, and null/empty names behave safely") {
    FakeClock clock;
    lc::GraphTelemetryAggregator agg;
    lc::GraphTelemetryConfig cfg;
    cfg.level = si::TelemetryLevel::deep;
    cfg.clock_fn = &FakeClock::Read;
    cfg.clock_ctx = &clock;
    agg.Configure(cfg);

    // Empty eval (0 nodes observed) returns false and produces nothing.
    lc::GraphEvalSummary summary;
    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kEval, 0));
    CHECK_FALSE(agg.EndEval(summary));

    // CancelEval aborts an in-progress eval cleanly without emitting.
    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    CHECK(agg.OnNodeAsk("Qcur-0", 28, "MUL_MAT", "CUDA0"));
    agg.CancelEval();
    CHECK_FALSE(agg.ActiveForCurrentEval());
    CHECK_FALSE(agg.EndEval(summary));

    // Null/empty tensor_name, op_name, and device_name do not crash and fall back safely.
    REQUIRE(agg.BeginEval(lc::GraphEvalKind::kEmbed, 4));
    CHECK(agg.OnNodeAsk(nullptr, 28, nullptr, nullptr));
    clock.now_ns += 50'000;
    CHECK(agg.OnNodeAfter(nullptr, 28));
    CHECK(agg.OnNodeAsk("", 28, "", ""));
    clock.now_ns += 25'000;
    CHECK(agg.OnNodeAfter("", 28));
    REQUIRE(agg.EndEval(summary));
    CHECK(summary.total_nodes == 2u);
    CHECK(summary.unlayered_nodes == 2u);
    CHECK(summary.layers.empty());
    REQUIRE(summary.operators.size() == 1u);
    CHECK(summary.operators[0].op_type == "UNKNOWN");
    CHECK(summary.operators[0].primary_device == "unknown");
    CHECK(summary.operators[0].duration_ns == 75'000u);

    // Disable() turns off further evals cleanly.
    agg.Disable();
    CHECK_FALSE(agg.Enabled());
    CHECK_FALSE(agg.BeginEval(lc::GraphEvalKind::kDecode, 1));
    CHECK_FALSE(agg.OnNodeAsk("Qcur-0", 28, "MUL_MAT", "CPU"));
    CHECK(agg.OnNodeAfter("Qcur-0", 28));
}
