// Workload-aware adaptive speculation controller for the external llama-server
// backend (docs/ROADMAP.md Phase 3 "adaptive speculation policy").
//
// Tracks per-workload rolling draft acceptance in a bounded sliding ring buffer
// and adjusts the recommended speculative draft length (`draft_n_max`) with
// hysteresis. When a workload drops below the pause threshold (and
// `min_draft_n == 0`), speculation temporarily pauses for that workload class
// only, then schedules a single bounded re-probe after `cooldown_requests` so a
// transient low-acceptance stretch does not disable MTP/n-gram speculation
// permanently.
//
// Pure policy: no threads, no I/O, and bounded O(5 * window_size) memory.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "llamaserver_protocol.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamaserver {

enum class SpeculationWorkload {
    general,
    code_edit,
    tool_json,
    prose,
    reasoning,
};

inline constexpr std::size_t kSpeculationWorkloadCount = 5;

[[nodiscard]] const char *to_string(SpeculationWorkload workload) noexcept;
[[nodiscard]] std::optional<SpeculationWorkload> parse_speculation_workload(std::string_view text) noexcept;

struct AdaptiveSpeculationConfig {
    static constexpr std::size_t kMaxWindowSize = 256;
    static constexpr std::uint32_t kMaxDraftN = 128;

    bool enabled = true;
    // 0 allows temporarily pausing speculation for a workload class.
    std::uint32_t min_draft_n = 0;
    std::uint32_t initial_draft_n = 3;
    std::uint32_t max_draft_n = 6;
    // Bounded sliding ring buffer length per workload class (max 256).
    std::size_t window_size = 16;
    // Minimum qualifying observations before stepping up/down or pausing.
    std::size_t min_samples = 4;
    // Pause when rolling acceptance drops strictly below this threshold
    // (requires min_draft_n == 0).
    double pause_acceptance_threshold = 0.25;
    // Step down by 1 when rolling acceptance drops strictly below this threshold.
    double step_down_acceptance_threshold = 0.45;
    // Step up by 1 when rolling acceptance reaches or exceeds this threshold.
    double step_up_acceptance_threshold = 0.72;
    // After pausing, skip this many recommend() requests before issuing a
    // single probe_draft_n re-probe. 0 disables automatic re-probing.
    std::uint32_t cooldown_requests = 8;
    std::uint32_t probe_draft_n = 1;
    // Ignore observations with fewer drafted tokens than this floor as noisy.
    std::uint64_t min_draft_tokens_per_sample = 4;
};

[[nodiscard]] Status validate_adaptive_speculation_config(const AdaptiveSpeculationConfig &config);

struct SpeculationObservation {
    SpeculationWorkload workload = SpeculationWorkload::general;
    std::uint64_t draft_n = 0;
    std::uint64_t draft_n_accepted = 0;
    std::uint64_t predicted_n = 0;
    // Optional measured decode throughput in tokens/sec; 0 when unknown.
    double decode_tps = 0.0;
    // True when this observation belongs to a cooldown re-probe issued by recommend().
    bool is_probe = false;
};

[[nodiscard]] SpeculationObservation observation_from_timings(
    const Timings &timings, SpeculationWorkload workload = SpeculationWorkload::general);
[[nodiscard]] SpeculationObservation observation_from_timings(
    const BackendTimings &timings, SpeculationWorkload workload = SpeculationWorkload::general);

struct SpeculationDecision {
    std::uint32_t draft_n_max = 0;
    bool paused = false;
    bool is_probe = false;
    double rolling_acceptance = 0.0;
    std::size_t sample_count = 0;
    std::string reason;
};

class AdaptiveSpeculationPolicy {
  public:
    // If `config` fails validation, defensively clamps out-of-range fields so
    // internal invariants always hold; callers should validate upfront with
    // validate_adaptive_speculation_config().
    explicit AdaptiveSpeculationPolicy(AdaptiveSpeculationConfig config = {});

    AdaptiveSpeculationPolicy(const AdaptiveSpeculationPolicy &) = delete;
    AdaptiveSpeculationPolicy &operator=(const AdaptiveSpeculationPolicy &) = delete;

    [[nodiscard]] const AdaptiveSpeculationConfig &config() const noexcept { return config_; }

    // Recommends `draft_n_max` for the next request of `workload`. When paused,
    // advances cooldown and schedules at most one `probe_draft_n` re-probe once
    // cooldown elapses.
    [[nodiscard]] SpeculationDecision recommend(SpeculationWorkload workload);

    // Records a completed request's speculative draft counters, updates the
    // sliding window with overflow-safe accumulation, clamps malformed
    // `draft_n_accepted > draft_n`, applies hysteresis step-up/step-down/pause,
    // and returns the updated decision.
    SpeculationDecision observe(const SpeculationObservation &obs);

    // Releases an in-flight probe (e.g. if the probe request was cancelled or
    // failed before producing timings) and restarts cooldown.
    void cancel_probe(SpeculationWorkload workload);

    // Observatory telemetry and /health inspection snapshot.
    [[nodiscard]] json::Object status_json() const;

    // Resets all workload buckets to their initial state without reallocating.
    void reset();

  private:
    struct Sample {
        std::uint64_t draft_n = 0;
        std::uint64_t draft_n_accepted = 0;
        double decode_tps = 0.0;
    };

    struct WindowStats {
        double rolling_acceptance = 0.0;
        double rolling_decode_tps = 0.0;
        std::uint64_t draft_tokens = 0;
        std::uint64_t accepted_tokens = 0;
    };

    struct BucketState {
        std::vector<Sample> window;
        std::size_t head = 0;
        std::size_t count = 0;
        std::uint32_t current_draft_n = 0;
        bool paused = false;
        bool probe_in_flight = false;
        std::uint32_t cooldown_remaining = 0;
        std::uint64_t total_observations = 0;
        std::uint64_t ignored_observations = 0;
        std::uint64_t clamped_observations = 0;
        std::uint64_t probes_issued = 0;
        std::uint64_t probe_recoveries = 0;
        std::string last_reason;
    };

    static void clear_window(BucketState &bucket) noexcept;
    static void push_sample(BucketState &bucket, const Sample &sample) noexcept;
    [[nodiscard]] static WindowStats compute_window_stats(const BucketState &bucket) noexcept;
    void reset_bucket_locked(BucketState &bucket) const;

    AdaptiveSpeculationConfig config_;
    mutable std::mutex mutex_;
    std::array<BucketState, kSpeculationWorkloadCount> buckets_{};
};

}  // namespace sonder::inference::llamaserver
