// Bounded online MoE expert-offload routing diagnostics (Item #27).
//
// Active ONLY when MoE experts are CPU-offloaded (`offload_active == true`,
// e.g. `VramEstimate::host_expert_bytes > 0`). When all experts reside in VRAM,
// recording is a lock-free no-op with zero per-layer storage allocations.
//
// Tracks:
//   - Aggregate and per-layer expert cache hits / misses
//   - Transfer bytes and transfer wait time (total and peak ms)
//   - Bounded per-layer top-K observed hot expert frequencies using
//     deterministic Space-Saving heavy-hitter counters capped at
//     `max_layers` x `max_experts_per_layer`
//
// Thread-safe: safe to call concurrently from execution and telemetry threads.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"

namespace sonder::inference {

struct MoeRoutingTelemetryConfig {
    // Must be enabled only when expert weights move across memory tiers
    // (host_expert_bytes > 0). False turns recording into a fast no-op.
    bool offload_active = false;
    // Upper bound on distinct layer IDs tracked (clamped to kMaxTrackedLayers).
    std::size_t max_layers = 256;
    // Upper bound on tracked expert slots per layer (Space-Saving heavy-hitter
    // table; exact when distinct active experts <= max_experts_per_layer,
    // clamped to kMaxExpertsPerLayer).
    std::size_t max_experts_per_layer = 64;
    // Default K for top-K hot expert reporting per layer.
    std::size_t top_k = 8;

    static constexpr std::size_t kMaxTrackedLayers = 1024;
    static constexpr std::size_t kMaxExpertsPerLayer = 512;
};

struct MoeExpertObservation {
    std::uint32_t expert_id = 0;
    bool cache_hit = false;
    std::uint64_t transfer_bytes = 0;
    double wait_ms = 0.0;
};

struct MoeExpertFrequency {
    std::uint32_t expert_id = 0;
    std::uint64_t activations = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    // Space-Saving error bound (0 when no expert slot was ever evicted in this layer).
    std::uint64_t overestimate_error = 0;
    // Observed share of this layer's total routed expert accesses in [0, 1].
    double share = 0.0;
};

struct MoeLayerTelemetry {
    std::uint32_t layer_id = 0;
    std::uint64_t total_routes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t transfer_bytes = 0;
    double transfer_wait_ms = 0.0;
    std::size_t tracked_experts = 0;
    std::uint64_t table_evictions = 0;
    // Sorted descending by `activations` (tie-broken by ascending `expert_id`),
    // bounded to at most `top_k` entries.
    std::vector<MoeExpertFrequency> top_experts;
};

struct MoeRoutingSnapshot {
    bool offload_active = false;
    std::uint64_t total_routes = 0;
    std::uint64_t expert_hits = 0;
    std::uint64_t expert_misses = 0;
    std::uint64_t transfer_count = 0;
    std::uint64_t transfer_bytes = 0;
    double transfer_wait_ms = 0.0;
    double max_transfer_wait_ms = 0.0;
    std::uint64_t dropped_layer_observations = 0;
    std::size_t tracked_layers = 0;
    std::size_t max_layers = 0;
    std::size_t max_experts_per_layer = 0;
    // Sorted by ascending `layer_id`.
    std::vector<MoeLayerTelemetry> layers;

    [[nodiscard]] double hit_rate() const noexcept {
        const std::uint64_t total = expert_hits + expert_misses;
        return total > 0 ? static_cast<double>(expert_hits) / static_cast<double>(total) : 0.0;
    }
};

class MoeRoutingTelemetry {
public:
    explicit MoeRoutingTelemetry(MoeRoutingTelemetryConfig config = {});
    MoeRoutingTelemetry(const MoeRoutingTelemetry&) = delete;
    MoeRoutingTelemetry& operator=(const MoeRoutingTelemetry&) = delete;

    [[nodiscard]] bool offload_active() const noexcept {
        return offload_active_.load(std::memory_order_acquire);
    }

    // Enables or disables recording dynamically. Disabling stops future writes
    // immediately without throwing away already-collected diagnostics until reset().
    void set_offload_active(bool active) noexcept;

    // Records a single routed expert access. Fast no-op when !offload_active().
    // Non-finite or negative `wait_ms` is clamped to 0.0.
    void record_expert_access(std::uint32_t layer_id, std::uint32_t expert_id, bool cache_hit,
                              std::uint64_t transfer_bytes = 0, double wait_ms = 0.0);

    // Records a batch of routed expert accesses for one layer under a single lock.
    void record_routing_batch(std::uint32_t layer_id, const std::vector<MoeExpertObservation>& observations);

    // Returns a consistent point-in-time snapshot with up to `top_k_override`
    // (or config.top_k when 0) hottest experts per tracked layer.
    [[nodiscard]] MoeRoutingSnapshot snapshot(std::size_t top_k_override = 0) const;

    // Returns the top-K observed hot experts for `layer_id` (empty if untracked).
    [[nodiscard]] std::vector<MoeExpertFrequency> hot_experts_for_layer(std::uint32_t layer_id,
                                                                        std::size_t top_k_override = 0) const;

    // Clears all accumulated counters and per-layer tables while keeping config.
    void reset();

private:
    struct ExpertSlot {
        std::uint32_t expert_id = 0;
        std::uint64_t activations = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t overestimate_error = 0;
    };

    struct LayerTable {
        std::uint32_t layer_id = 0;
        std::uint64_t total_routes = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t transfer_bytes = 0;
        double transfer_wait_ms = 0.0;
        std::uint64_t table_evictions = 0;
        std::vector<ExpertSlot> slots;
    };

    void record_locked(LayerTable* layer, std::uint32_t expert_id, bool cache_hit, std::uint64_t transfer_bytes,
                       double wait_ms);
    LayerTable* find_or_create_layer_locked(std::uint32_t layer_id);
    [[nodiscard]] std::vector<MoeExpertFrequency> build_top_k_locked(const LayerTable& layer,
                                                                     std::size_t k) const;

    std::atomic<bool> offload_active_{false};
    std::size_t max_layers_ = 256;
    std::size_t max_experts_per_layer_ = 64;
    std::size_t default_top_k_ = 8;

    mutable std::mutex mu_;
    std::uint64_t total_routes_ = 0;
    std::uint64_t expert_hits_ = 0;
    std::uint64_t expert_misses_ = 0;
    std::uint64_t transfer_count_ = 0;
    std::uint64_t transfer_bytes_ = 0;
    double transfer_wait_ms_ = 0.0;
    double max_transfer_wait_ms_ = 0.0;
    std::uint64_t dropped_layer_observations_ = 0;
    std::vector<LayerTable> layers_;
};

// Builds a bounded MoeRoutingTelemetryConfig directly from a VramEstimate:
// sets `offload_active = (estimate.host_expert_bytes > 0)` so diagnostic
// overhead is zero whenever all experts fit in VRAM.
[[nodiscard]] MoeRoutingTelemetryConfig make_moe_routing_telemetry_config(
    const VramEstimate& estimate, std::size_t top_k = 8, std::size_t max_layers = 256,
    std::size_t max_experts_per_layer = 64) noexcept;

// Serializes a routing telemetry snapshot into a stable JSON envelope
// (`schema: "sonder.inference.moe_routing_telemetry/1"`).
[[nodiscard]] json::Object moe_routing_telemetry_to_json(const MoeRoutingSnapshot& snapshot);

}  // namespace sonder::inference
