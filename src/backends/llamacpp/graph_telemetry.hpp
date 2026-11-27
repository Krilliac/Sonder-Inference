// Sonder Inference - llama.cpp/GGML deep graph telemetry aggregator.
//
// Collects per-graph-eval operator activity, backend/device attribution, node
// counts, and (when enabled) per-layer and per-op timings via ggml's
// `ggml_backend_sched_eval_callback` (llama.cpp v1.0.0:
//   - include/llama.h: llama_context_params::cb_eval, cb_eval_user_data
//   - ggml/include/ggml-backend.h: ggml_backend_sched_eval_callback
//   - ggml/src/ggml-backend.cpp: ggml_backend_sched_compute_splits
//   - src/llama-context.cpp: llama_context::graph_get_cb ("%s-%d", name, il))
//
// Performance contract:
// - Strictly active at TelemetryLevel::deep only. Below deep, callers MUST NOT
//   install `cb_eval` on `llama_context_params` at all.
// - Inside `OnNodeAsk` and `OnNodeAfter` (invoked on the hot eval path):
//   zero heap allocations, zero string formatting, bounded fixed-size arrays.
// - Emits bounded aggregated `backend.layer.timing` and
//   `backend.operator.activity` events after the graph evaluation completes.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/json.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::backends::llamacpp {

// Fixed upper bounds enforced on the hot path (no per-node heap allocations).
inline constexpr std::size_t kMaxTrackedLayers = 256;
inline constexpr std::size_t kMaxTrackedOps = 128;
inline constexpr std::size_t kMaxTrackedDevices = 16;
inline constexpr std::size_t kMaxGraphDeviceNameBytes = 32;
inline constexpr std::size_t kDefaultMaxOpsPerEval = 64;
inline constexpr std::size_t kDefaultMaxLayersPerEval = kMaxTrackedLayers;

enum class GraphEvalKind : std::uint8_t {
    kEval = 0,
    kPrefill = 1,
    kDecode = 2,
    kBatch = 3,
    kEmbed = 4,
};

[[nodiscard]] const char* ToString(GraphEvalKind kind) noexcept;

// True iff deep graph telemetry callback and `Capability::layer_telemetry`
// should be enabled for `level`. Strictly true only for TelemetryLevel::deep.
[[nodiscard]] bool ShouldInstallGraphCallback(sonder::inference::TelemetryLevel level) noexcept;

// Parses the non-negative layer index from a llama.cpp v1.0.0 compute tensor
// name (`<name>-<layer>` with optional trailing ggml tags such as ` (view)`,
// ` (reshaped)`, ` (cont)`). Returns -1 for unlayered/malformed names.
// Pure, noexcept, zero allocations.
[[nodiscard]] std::int32_t ParseLayerIndex(std::string_view tensor_name) noexcept;

// Zero-allocation monotonic nanosecond clock hook (for deterministic unit tests).
using GraphTelemetryClockFn = std::uint64_t (*)(void* ctx) noexcept;

struct GraphTelemetryConfig {
    sonder::inference::TelemetryLevel level = sonder::inference::TelemetryLevel::standard;
    std::uint32_t sample_every_n = 1;           // emit every Nth eval (>= 1)
    bool collect_timings = true;                // true => request post-node callback for wall timings
    std::size_t max_layers_per_eval = kDefaultMaxLayersPerEval;
    std::size_t max_ops_per_eval = kDefaultMaxOpsPerEval;
    GraphTelemetryClockFn clock_fn = nullptr;   // nullptr => steady_clock ns
    void* clock_ctx = nullptr;
};

struct GraphDeviceCount {
    std::string device;
    std::uint64_t node_count = 0;
};

struct GraphLayerSummary {
    std::int32_t layer_index = 0;
    std::uint64_t node_count = 0;
    std::uint64_t timed_nodes = 0;
    std::uint64_t duration_ns = 0;
    std::uint32_t op_count = 0;
    std::string primary_device;
    std::vector<std::string> devices;
};

struct GraphOpSummary {
    std::int32_t op_id = -1;
    std::string op_type;
    std::uint64_t node_count = 0;
    std::uint64_t timed_nodes = 0;
    std::uint64_t duration_ns = 0;
    std::string primary_device;
    std::vector<GraphDeviceCount> devices;
};

struct GraphEvalSummary {
    std::uint64_t eval_index = 0;
    std::uint32_t sample_every_n = 1;
    GraphEvalKind eval_kind = GraphEvalKind::kEval;
    std::int32_t batch_tokens = 0;
    std::uint64_t total_nodes = 0;
    std::uint64_t total_timed_nodes = 0;
    std::uint64_t total_duration_ns = 0;
    std::uint64_t unlayered_nodes = 0;
    std::uint64_t unlayered_duration_ns = 0;
    std::uint64_t overflow_layer_nodes = 0;
    std::uint64_t overflow_layer_duration_ns = 0;
    std::uint64_t overflow_device_nodes = 0;
    std::size_t truncated_layers = 0;
    std::size_t truncated_ops = 0;
    bool has_timings = false;
    std::vector<GraphLayerSummary> layers;
    std::vector<GraphOpSummary> operators;
};

// Fixed-storage, single-eval-at-a-time aggregator invoked by ggml's `cb_eval`.
// All per-node methods (`OnNodeAsk`, `OnNodeAfter`) are noexcept and perform
// zero heap allocations and zero string formatting.
class GraphTelemetryAggregator {
public:
    GraphTelemetryAggregator() noexcept;

    void Configure(const GraphTelemetryConfig& config) noexcept;
    void Disable() noexcept;

    [[nodiscard]] bool Enabled() const noexcept { return enabled_; }
    [[nodiscard]] bool ActiveForCurrentEval() const noexcept { return active_for_eval_; }
    [[nodiscard]] sonder::inference::TelemetryLevel Level() const noexcept { return config_.level; }
    [[nodiscard]] std::uint64_t EvalCount() const noexcept { return eval_seq_; }

    // Starts a graph evaluation. Increments eval sequence and activates
    // collection when enabled and `(eval_seq - 1) % sample_every_n == 0`.
    // Returns true iff this eval will be sampled.
    bool BeginEval(GraphEvalKind kind = GraphEvalKind::kEval, std::int32_t batch_tokens = 0) noexcept;

    // Called from `ggml_backend_sched_eval_callback(t, /*ask=*/true, user_data)`.
    // Returns true iff ggml should invoke `OnNodeAfter` after computing+syncing
    // this node (only when active and `collect_timings` is enabled).
    bool OnNodeAsk(const char* tensor_name, int op_id, const char* op_name,
                   const char* device_name) noexcept;

    // Called from `ggml_backend_sched_eval_callback(t, /*ask=*/false, user_data)`.
    // Always returns true so graph execution is never cancelled.
    bool OnNodeAfter(const char* tensor_name, int op_id) noexcept;

    // Finalizes the current eval and populates `out` (outside the hot callback).
    // Returns true iff this eval was sampled and observed at least one node.
    bool EndEval(GraphEvalSummary& out);

    // Aborts the current eval without emitting a summary (e.g. on decode error).
    void CancelEval() noexcept;

private:
    struct DeviceSlot {
        const char* name_ptr = nullptr;
        char name_buf[kMaxGraphDeviceNameBytes]{};
        std::uint64_t node_count = 0;
        std::uint64_t duration_ns = 0;
    };

    struct OpBucket {
        const char* op_name = nullptr;
        std::uint64_t node_count = 0;
        std::uint64_t timed_nodes = 0;
        std::uint64_t duration_ns = 0;
        std::array<std::uint32_t, kMaxTrackedDevices> device_counts{};
        bool active = false;
    };

    struct LayerBucket {
        std::uint64_t node_count = 0;
        std::uint64_t timed_nodes = 0;
        std::uint64_t duration_ns = 0;
        std::array<std::uint32_t, kMaxTrackedDevices> device_counts{};
        std::array<std::uint64_t, 2> op_mask{};  // 128-bit mask for distinct op_count
        std::uint32_t overflow_op_seen = 0;
        bool active = false;
    };

    void ResetEvalState() noexcept;
    [[nodiscard]] std::uint64_t ReadClockNs() const noexcept;
    [[nodiscard]] std::int8_t FindOrAddDevice(const char* device_name) noexcept;

    GraphTelemetryConfig config_{};
    bool enabled_ = false;
    bool active_for_eval_ = false;
    std::uint64_t eval_seq_ = 0;
    GraphEvalKind current_kind_ = GraphEvalKind::kEval;
    std::int32_t current_batch_tokens_ = 0;

    std::uint64_t total_nodes_ = 0;
    std::uint64_t total_timed_nodes_ = 0;
    std::uint64_t total_duration_ns_ = 0;
    std::uint64_t unlayered_nodes_ = 0;
    std::uint64_t unlayered_duration_ns_ = 0;
    std::uint64_t overflow_layer_nodes_ = 0;
    std::uint64_t overflow_layer_duration_ns_ = 0;
    std::uint64_t overflow_device_nodes_ = 0;

    std::array<DeviceSlot, kMaxTrackedDevices> devices_{};
    std::size_t device_count_ = 0;

    std::array<OpBucket, kMaxTrackedOps> op_buckets_{};
    std::array<std::uint16_t, kMaxTrackedOps> active_ops_{};
    std::size_t active_op_count_ = 0;
    OpBucket unknown_op_bucket_{};

    std::array<LayerBucket, kMaxTrackedLayers> layer_buckets_{};
    std::array<std::uint16_t, kMaxTrackedLayers> active_layers_{};
    std::size_t active_layer_count_ = 0;

    // Pending node state between ask=true and ask=false (single-threaded scheduler).
    bool has_pending_node_ = false;
    std::uint64_t pending_start_ns_ = 0;
    int pending_op_id_ = -1;
    std::int32_t pending_layer_ = -1;
    std::int8_t pending_device_idx_ = -1;
};

// ---------------------------------------------------------------------------
// Observatory event serialization & emission helpers (deep level only).
// Inline so `sonder_backend_llamacpp` does not depend on out-of-line
// `sonder_inference` symbols on static-library link orders.
// ---------------------------------------------------------------------------

inline sonder::inference::json::Object BuildLayerTimingAttributes(const GraphEvalSummary& summary,
                                                                  const GraphLayerSummary& layer,
                                                                  std::string_view backend_name = "llamacpp") {
    namespace json = sonder::inference::json;
    json::Array devices_arr;
    devices_arr.reserve(layer.devices.size());
    for (const auto& d : layer.devices) {
        devices_arr.emplace_back(d);
    }
    json::Object attrs{
        {"backend", std::string(backend_name)},
        {"eval_index", summary.eval_index},
        {"sample_every_n", summary.sample_every_n},
        {"layer_index", layer.layer_index},
        {"node_count", layer.node_count},
        {"backend_device", layer.primary_device.empty() ? std::string("unknown") : layer.primary_device},
        {"devices", std::move(devices_arr)},
        {"op_count", layer.op_count},
        {"eval_kind", ToString(summary.eval_kind)},
    };
    if (summary.batch_tokens > 0) {
        attrs.set("batch_tokens", summary.batch_tokens);
    }
    if (layer.timed_nodes > 0) {
        attrs.set("duration_ms", static_cast<double>(layer.duration_ns) / 1e6);
        attrs.set("timed_nodes", layer.timed_nodes);
    }
    return attrs;
}

inline sonder::inference::json::Object BuildOperatorActivityAttributes(const GraphEvalSummary& summary,
                                                                       const GraphOpSummary& op,
                                                                       std::string_view backend_name = "llamacpp") {
    namespace json = sonder::inference::json;
    json::Array devices_arr;
    devices_arr.reserve(op.devices.size());
    for (const auto& dc : op.devices) {
        devices_arr.emplace_back(json::Object{{"device", dc.device}, {"node_count", dc.node_count}});
    }
    json::Object attrs{
        {"backend", std::string(backend_name)},
        {"eval_index", summary.eval_index},
        {"sample_every_n", summary.sample_every_n},
        {"op_type", op.op_type.empty() ? std::string("UNKNOWN") : op.op_type},
        {"op_id", op.op_id},
        {"node_count", op.node_count},
        {"total_eval_nodes", summary.total_nodes},
        {"backend_device", op.primary_device.empty() ? std::string("unknown") : op.primary_device},
        {"devices", std::move(devices_arr)},
        {"eval_kind", ToString(summary.eval_kind)},
    };
    if (summary.batch_tokens > 0) {
        attrs.set("batch_tokens", summary.batch_tokens);
    }
    if (op.timed_nodes > 0) {
        attrs.set("duration_ms", static_cast<double>(op.duration_ns) / 1e6);
        attrs.set("timed_nodes", op.timed_nodes);
    }
    return attrs;
}

// Emits bounded `backend.layer.timing` and `backend.operator.activity` events
// onto `bus` at `TelemetryLevel::deep`. Returns the number of events accepted
// by `bus`. Emits nothing (returns 0) whenever `active_level != deep` or
// `!bus.enabled(TelemetryLevel::deep)`.
inline std::size_t EmitGraphTelemetry(sonder::inference::TelemetryBus& bus,
                                      const sonder::inference::TelemetryContext& context,
                                      const GraphEvalSummary& summary,
                                      sonder::inference::TelemetryLevel active_level,
                                      std::string_view backend_name = "llamacpp") {
    using sonder::inference::TelemetryLevel;
    if (!ShouldInstallGraphCallback(active_level) || !bus.enabled(TelemetryLevel::deep) || summary.total_nodes == 0) {
        return 0;
    }
    std::size_t emitted = 0;
    for (const auto& layer : summary.layers) {
        if (bus.emit("backend.layer.timing", context, BuildLayerTimingAttributes(summary, layer, backend_name),
                     TelemetryLevel::deep)) {
            ++emitted;
        }
    }
    for (const auto& op : summary.operators) {
        if (bus.emit("backend.operator.activity", context, BuildOperatorActivityAttributes(summary, op, backend_name),
                     TelemetryLevel::deep)) {
            ++emitted;
        }
    }
    return emitted;
}

}  // namespace sonder::backends::llamacpp
