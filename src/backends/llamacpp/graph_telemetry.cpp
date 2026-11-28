// Sonder Inference - llama.cpp/GGML deep graph telemetry aggregator.
#include "graph_telemetry.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>

namespace sonder::backends::llamacpp {

namespace {

constexpr std::int32_t kMaxParsedLayerIndex = 1'000'000;

std::uint64_t DefaultClockNs(void* /*ctx*/) noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void CopyCStringBounded(char* dst, std::size_t dst_size, const char* src) noexcept {
    if (dst_size == 0) return;
    if (src == nullptr || src[0] == '\0') {
        src = "unknown";
    }
    std::size_t i = 0;
    while (i + 1 < dst_size && src[i] != '\0') {
        dst[i] = src[i];
        ++i;
    }
    dst[i] = '\0';
}

bool EqualsCStringPrefix(const char* a, const char* b, std::size_t max_chars) noexcept {
    if (a == nullptr || b == nullptr) return false;
    for (std::size_t i = 0; i < max_chars; ++i) {
        if (a[i] != b[i]) return false;
        if (a[i] == '\0') return true;
    }
    return true;
}

}  // namespace

const char* ToString(GraphEvalKind kind) noexcept {
    switch (kind) {
        case GraphEvalKind::kEval: return "eval";
        case GraphEvalKind::kPrefill: return "prefill";
        case GraphEvalKind::kDecode: return "decode";
        case GraphEvalKind::kBatch: return "batch";
        case GraphEvalKind::kEmbed: return "embed";
    }
    return "eval";
}

bool ShouldInstallGraphCallback(sonder::inference::TelemetryLevel level) noexcept {
    return level == sonder::inference::TelemetryLevel::deep;
}

std::int32_t ParseLayerIndex(std::string_view tensor_name) noexcept {
    if (tensor_name.empty()) return -1;

    // Strip repeated trailing ggml annotations appended by ggml.c, e.g.
    // "Qcur-12 (view)", "Kcur-5 (reshaped)", "Vcur-0 (cont)", "Qcur-3 (view) (reshaped)".
    std::string_view s = tensor_name;
    while (s.size() >= 4 && s.back() == ')') {
        const std::size_t open = s.rfind(" (");
        if (open == std::string_view::npos) break;
        s = s.substr(0, open);
    }

    const std::size_t dash = s.rfind('-');
    if (dash == std::string_view::npos || dash == 0 || dash + 1 >= s.size()) {
        return -1;
    }
    // Reject double-dash negative patterns such as "foo--3".
    if (s[dash - 1] == '-') {
        return -1;
    }

    std::int64_t val = 0;
    for (std::size_t i = dash + 1; i < s.size(); ++i) {
        const char ch = s[i];
        if (ch < '0' || ch > '9') return -1;
        val = val * 10 + static_cast<std::int64_t>(ch - '0');
        if (val > kMaxParsedLayerIndex) return -1;
    }
    return static_cast<std::int32_t>(val);
}

GraphTelemetryAggregator::GraphTelemetryAggregator() noexcept = default;

void GraphTelemetryAggregator::Configure(const GraphTelemetryConfig& config) noexcept {
    ResetEvalState();
    config_ = config;
    if (config_.sample_every_n == 0) {
        config_.sample_every_n = 1;
    }
    config_.max_layers_per_eval = std::clamp<std::size_t>(config_.max_layers_per_eval, 1, kMaxTrackedLayers);
    config_.max_ops_per_eval = std::clamp<std::size_t>(config_.max_ops_per_eval, 1, kMaxTrackedOps);
    enabled_ = ShouldInstallGraphCallback(config_.level);
    active_for_eval_ = false;
    eval_seq_ = 0;
}

void GraphTelemetryAggregator::Disable() noexcept {
    ResetEvalState();
    enabled_ = false;
    active_for_eval_ = false;
    config_.level = sonder::inference::TelemetryLevel::off;
}

void GraphTelemetryAggregator::ResetEvalState() noexcept {
    total_nodes_ = 0;
    total_timed_nodes_ = 0;
    total_duration_ns_ = 0;
    unlayered_nodes_ = 0;
    unlayered_duration_ns_ = 0;
    overflow_layer_nodes_ = 0;
    overflow_layer_duration_ns_ = 0;
    overflow_device_nodes_ = 0;

    for (std::size_t i = 0; i < device_count_; ++i) {
        devices_[i] = DeviceSlot{};
    }
    device_count_ = 0;

    for (std::size_t i = 0; i < active_op_count_; ++i) {
        op_buckets_[active_ops_[i]] = OpBucket{};
    }
    active_op_count_ = 0;
    unknown_op_bucket_ = OpBucket{};

    for (std::size_t i = 0; i < active_layer_count_; ++i) {
        layer_buckets_[active_layers_[i]] = LayerBucket{};
    }
    active_layer_count_ = 0;

    has_pending_node_ = false;
    pending_start_ns_ = 0;
    pending_op_id_ = -1;
    pending_layer_ = -1;
    pending_device_idx_ = -1;
}

std::uint64_t GraphTelemetryAggregator::ReadClockNs() const noexcept {
    if (config_.clock_fn != nullptr) {
        return config_.clock_fn(config_.clock_ctx);
    }
    return DefaultClockNs(nullptr);
}

std::int8_t GraphTelemetryAggregator::FindOrAddDevice(const char* device_name) noexcept {
    const char* effective = (device_name != nullptr && device_name[0] != '\0') ? device_name : "unknown";
    for (std::size_t i = 0; i < device_count_; ++i) {
        if (devices_[i].name_ptr == effective ||
            EqualsCStringPrefix(devices_[i].name_buf, effective, kMaxGraphDeviceNameBytes - 1)) {
            return static_cast<std::int8_t>(i);
        }
    }
    if (device_count_ >= kMaxTrackedDevices) {
        ++overflow_device_nodes_;
        return -1;
    }
    const std::size_t idx = device_count_++;
    devices_[idx].name_ptr = effective;
    CopyCStringBounded(devices_[idx].name_buf, kMaxGraphDeviceNameBytes, effective);
    devices_[idx].node_count = 0;
    devices_[idx].duration_ns = 0;
    return static_cast<std::int8_t>(idx);
}

bool GraphTelemetryAggregator::BeginEval(GraphEvalKind kind, std::int32_t batch_tokens) noexcept {
    if (!enabled_) {
        active_for_eval_ = false;
        return false;
    }
    ++eval_seq_;
    const std::uint32_t stride = config_.sample_every_n == 0 ? 1u : config_.sample_every_n;
    if (((eval_seq_ - 1) % stride) != 0) {
        active_for_eval_ = false;
        return false;
    }
    ResetEvalState();
    current_kind_ = kind;
    current_batch_tokens_ = batch_tokens;
    active_for_eval_ = true;
    return true;
}

bool GraphTelemetryAggregator::OnNodeAsk(const char* tensor_name, int op_id, const char* op_name,
                                         const char* device_name) noexcept {
    if (!active_for_eval_) {
        return false;
    }
    ++total_nodes_;

    const std::int8_t dev_idx = FindOrAddDevice(device_name);
    if (dev_idx >= 0) {
        ++devices_[static_cast<std::size_t>(dev_idx)].node_count;
    }

    if (op_id >= 0 && op_id < static_cast<int>(kMaxTrackedOps)) {
        const auto uid = static_cast<std::size_t>(op_id);
        OpBucket& ob = op_buckets_[uid];
        if (!ob.active) {
            ob.active = true;
            active_ops_[active_op_count_++] = static_cast<std::uint16_t>(uid);
        }
        if (ob.op_name == nullptr && op_name != nullptr && op_name[0] != '\0') {
            ob.op_name = op_name;
        }
        ++ob.node_count;
        if (dev_idx >= 0) {
            ++ob.device_counts[static_cast<std::size_t>(dev_idx)];
        }
    } else {
        unknown_op_bucket_.active = true;
        if (unknown_op_bucket_.op_name == nullptr) {
            unknown_op_bucket_.op_name = "UNKNOWN";
        }
        ++unknown_op_bucket_.node_count;
        if (dev_idx >= 0) {
            ++unknown_op_bucket_.device_counts[static_cast<std::size_t>(dev_idx)];
        }
    }

    const std::string_view name_sv = tensor_name != nullptr ? std::string_view(tensor_name) : std::string_view{};
    const std::int32_t layer = ParseLayerIndex(name_sv);
    if (layer >= 0 && layer < static_cast<std::int32_t>(kMaxTrackedLayers)) {
        const auto lid = static_cast<std::size_t>(layer);
        LayerBucket& lb = layer_buckets_[lid];
        if (!lb.active) {
            lb.active = true;
            active_layers_[active_layer_count_++] = static_cast<std::uint16_t>(lid);
        }
        ++lb.node_count;
        if (dev_idx >= 0) {
            ++lb.device_counts[static_cast<std::size_t>(dev_idx)];
        }
        if (op_id >= 0 && op_id < static_cast<int>(kMaxTrackedOps)) {
            const auto uid = static_cast<std::size_t>(op_id);
            lb.op_mask[uid >> 6] |= (std::uint64_t{1} << (uid & 63u));
        } else {
            lb.overflow_op_seen = 1u;
        }
    } else if (layer >= static_cast<std::int32_t>(kMaxTrackedLayers)) {
        ++overflow_layer_nodes_;
    } else {
        ++unlayered_nodes_;
    }

    if (!config_.collect_timings) {
        has_pending_node_ = false;
        return false;
    }

    pending_start_ns_ = ReadClockNs();
    pending_op_id_ = op_id;
    pending_layer_ = layer;
    pending_device_idx_ = dev_idx;
    has_pending_node_ = true;
    return true;
}

bool GraphTelemetryAggregator::OnNodeAfter(const char* /*tensor_name*/, int /*op_id*/) noexcept {
    if (!active_for_eval_ || !has_pending_node_) {
        return true;
    }
    has_pending_node_ = false;
    const std::uint64_t end_ns = ReadClockNs();
    const std::uint64_t dt = end_ns >= pending_start_ns_ ? (end_ns - pending_start_ns_) : 0u;

    ++total_timed_nodes_;
    total_duration_ns_ += dt;

    if (pending_device_idx_ >= 0) {
        devices_[static_cast<std::size_t>(pending_device_idx_)].duration_ns += dt;
    }

    if (pending_op_id_ >= 0 && pending_op_id_ < static_cast<int>(kMaxTrackedOps)) {
        OpBucket& ob = op_buckets_[static_cast<std::size_t>(pending_op_id_)];
        ++ob.timed_nodes;
        ob.duration_ns += dt;
    } else {
        ++unknown_op_bucket_.timed_nodes;
        unknown_op_bucket_.duration_ns += dt;
    }

    if (pending_layer_ >= 0 && pending_layer_ < static_cast<std::int32_t>(kMaxTrackedLayers)) {
        LayerBucket& lb = layer_buckets_[static_cast<std::size_t>(pending_layer_)];
        ++lb.timed_nodes;
        lb.duration_ns += dt;
    } else if (pending_layer_ >= static_cast<std::int32_t>(kMaxTrackedLayers)) {
        overflow_layer_duration_ns_ += dt;
    } else {
        unlayered_duration_ns_ += dt;
    }

    return true;
}

bool GraphTelemetryAggregator::EndEval(GraphEvalSummary& out) {
    if (!active_for_eval_) {
        return false;
    }
    active_for_eval_ = false;
    has_pending_node_ = false;
    if (total_nodes_ == 0) {
        return false;
    }

    out = GraphEvalSummary{};
    out.eval_index = eval_seq_;
    out.sample_every_n = config_.sample_every_n;
    out.eval_kind = current_kind_;
    out.batch_tokens = current_batch_tokens_;
    out.total_nodes = total_nodes_;
    out.total_timed_nodes = total_timed_nodes_;
    out.total_duration_ns = total_duration_ns_;
    out.unlayered_nodes = unlayered_nodes_;
    out.unlayered_duration_ns = unlayered_duration_ns_;
    out.overflow_layer_nodes = overflow_layer_nodes_;
    out.overflow_layer_duration_ns = overflow_layer_duration_ns_;
    out.overflow_device_nodes = overflow_device_nodes_;
    out.has_timings = total_timed_nodes_ > 0;

    // Layers: ascending layer_index, bounded to config_.max_layers_per_eval.
    std::sort(active_layers_.begin(), active_layers_.begin() + static_cast<std::ptrdiff_t>(active_layer_count_));
    const std::size_t layer_limit = std::min(active_layer_count_, config_.max_layers_per_eval);
    out.truncated_layers = active_layer_count_ - layer_limit;
    out.layers.reserve(layer_limit);
    for (std::size_t i = 0; i < layer_limit; ++i) {
        const std::uint16_t lid = active_layers_[i];
        const LayerBucket& lb = layer_buckets_[lid];
        GraphLayerSummary ls;
        ls.layer_index = static_cast<std::int32_t>(lid);
        ls.node_count = lb.node_count;
        ls.timed_nodes = lb.timed_nodes;
        ls.duration_ns = lb.duration_ns;
        ls.op_count = static_cast<std::uint32_t>(std::popcount(lb.op_mask[0]) + std::popcount(lb.op_mask[1])) +
                      lb.overflow_op_seen;

        std::uint32_t best_count = 0;
        std::size_t best_dev = kMaxTrackedDevices;
        for (std::size_t d = 0; d < device_count_; ++d) {
            if (lb.device_counts[d] > 0) {
                ls.devices.emplace_back(devices_[d].name_buf);
                if (lb.device_counts[d] > best_count) {
                    best_count = lb.device_counts[d];
                    best_dev = d;
                }
            }
        }
        ls.primary_device = best_dev < device_count_ ? std::string(devices_[best_dev].name_buf) : std::string("unknown");
        out.layers.push_back(std::move(ls));
    }

    // Operators: build full list, sort hottest first (duration_ns desc, node_count desc, op_id asc),
    // then bound to config_.max_ops_per_eval.
    std::vector<GraphOpSummary> ops;
    ops.reserve(active_op_count_ + (unknown_op_bucket_.active ? 1u : 0u));
    auto append_op = [&](std::int32_t op_id, const OpBucket& ob) {
        GraphOpSummary os;
        os.op_id = op_id;
        os.op_type = (ob.op_name != nullptr && ob.op_name[0] != '\0') ? std::string(ob.op_name) : std::string("UNKNOWN");
        os.node_count = ob.node_count;
        os.timed_nodes = ob.timed_nodes;
        os.duration_ns = ob.duration_ns;
        for (std::size_t d = 0; d < device_count_; ++d) {
            if (ob.device_counts[d] > 0) {
                os.devices.push_back(GraphDeviceCount{std::string(devices_[d].name_buf), ob.device_counts[d]});
            }
        }
        std::sort(os.devices.begin(), os.devices.end(), [](const GraphDeviceCount& a, const GraphDeviceCount& b) {
            if (a.node_count != b.node_count) return a.node_count > b.node_count;
            return a.device < b.device;
        });
        os.primary_device = os.devices.empty() ? std::string("unknown") : os.devices.front().device;
        ops.push_back(std::move(os));
    };
    for (std::size_t i = 0; i < active_op_count_; ++i) {
        const std::uint16_t uid = active_ops_[i];
        append_op(static_cast<std::int32_t>(uid), op_buckets_[uid]);
    }
    if (unknown_op_bucket_.active) {
        append_op(-1, unknown_op_bucket_);
    }

    std::sort(ops.begin(), ops.end(), [](const GraphOpSummary& a, const GraphOpSummary& b) {
        if (a.duration_ns != b.duration_ns) return a.duration_ns > b.duration_ns;
        if (a.node_count != b.node_count) return a.node_count > b.node_count;
        return a.op_id < b.op_id;
    });

    if (ops.size() > config_.max_ops_per_eval) {
        out.truncated_ops = ops.size() - config_.max_ops_per_eval;
        ops.resize(config_.max_ops_per_eval);
    }
    out.operators = std::move(ops);
    return true;
}

void GraphTelemetryAggregator::CancelEval() noexcept {
    active_for_eval_ = false;
    has_pending_node_ = false;
}

}  // namespace sonder::backends::llamacpp
