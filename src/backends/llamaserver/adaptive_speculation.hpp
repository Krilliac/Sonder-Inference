// Time- and load-aware measured-cost adaptive speculation controller for the
// external llama-server backend (Doc 1 Package C / Doc 2 #21).
//
// Estimates cost per committed token L_hat(m, K, B, C) = elapsed_ms / committed_token
// from non-overlapping decode elapsed intervals (keeping prompt/prefix restore time
// strictly outside decode-only speed), tracks a measured K=0 (no-speculation)
// baseline per band alongside per-K costs, and stores policy state in a strictly
// bounded LRU table keyed by:
//   (target_identity, draft_identity, context_band, concurrency_band, workload)
//
// High acceptance alone never overrides a measured latency regression: the
// controller steps down K or chooses K=0 (paused) even at high acceptance when
// measured ms/committed-token is worse than the K=0 baseline or a shorter K.
// Cooldown, hysteresis, direction-reversal holdoff, and single-probe ownership
// (with interactive foreground probe deferral) prevent oscillation and probe storms.
//
// Pure policy: no threads, no I/O, strictly bounded O(max_entries * window_size) memory.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
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

enum class ContextBand : std::uint8_t {
    short_ctx = 0,      // <= 4096 tokens
    medium_ctx = 1,     // 4097 .. 16384 tokens
    long_ctx = 2,       // 16385 .. 65536 tokens
    very_long_ctx = 3,  // > 65536 tokens
};

inline constexpr std::size_t kContextBandCount = 4;

[[nodiscard]] const char *to_string(ContextBand band) noexcept;
[[nodiscard]] ContextBand classify_context_band(std::uint64_t context_tokens) noexcept;

enum class ConcurrencyBand : std::uint8_t {
    solo = 0,    // <= 1 active request
    low = 1,     // 2 .. 3 active requests
    medium = 2,  // 4 .. 7 active requests
    high = 3,    // >= 8 active requests
};

inline constexpr std::size_t kConcurrencyBandCount = 4;

[[nodiscard]] const char *to_string(ConcurrencyBand band) noexcept;
[[nodiscard]] ConcurrencyBand classify_concurrency_band(std::uint32_t active_requests) noexcept;

// Bounded composite key isolating policy state across immutable target/draft
// identities, context-length band, concurrency band, and workload class.
struct SpeculationContextKey {
    static constexpr std::size_t kMaxIdentityLength = 256;

    std::string target_identity;
    std::string draft_identity;
    ContextBand context_band = ContextBand::short_ctx;
    ConcurrencyBand concurrency_band = ConcurrencyBand::solo;
    SpeculationWorkload workload = SpeculationWorkload::general;

    [[nodiscard]] bool operator==(const SpeculationContextKey &other) const noexcept {
        return context_band == other.context_band && concurrency_band == other.concurrency_band &&
               workload == other.workload && target_identity == other.target_identity &&
               draft_identity == other.draft_identity;
    }
};

struct SpeculationContextKeyHash {
    [[nodiscard]] std::size_t operator()(const SpeculationContextKey &key) const noexcept;
};

// Bounds an arbitrary identity string to kMaxIdentityLength while preserving
// collision resistance for over-length inputs via an FNV-1a 64-bit hex suffix.
[[nodiscard]] std::string normalize_speculation_identity(std::string_view identity);

// Live request workload classifiers shared by LlamaServerModel and unit tests.
// Content-based only (never conflates scheduling priority with content class).
[[nodiscard]] SpeculationWorkload classify_generate_workload(const GenerateRequest &request,
                                                             std::string_view grammar = {}) noexcept;
[[nodiscard]] SpeculationWorkload classify_chat_workload(const ChatRequest &request,
                                                         std::string_view grammar = {}) noexcept;

// Conservative zero-I/O prompt token count estimators for context banding at admission.
[[nodiscard]] std::uint64_t estimate_generate_context_tokens(const GenerateRequest &request) noexcept;
[[nodiscard]] std::uint64_t estimate_chat_context_tokens(const ChatRequest &request) noexcept;

struct AdaptiveSpeculationConfig {
    static constexpr std::size_t kMaxWindowSize = 256;
    static constexpr std::uint32_t kMaxDraftN = 128;
    static constexpr std::size_t kMaxEntries = 4096;
    static constexpr std::size_t kDefaultMaxEntries = 128;
    static constexpr std::size_t kMaxTrackedDraftLevels = 16;

    bool enabled = true;
    // 0 allows temporarily pausing speculation (choosing K=0) for a band.
    std::uint32_t min_draft_n = 0;
    std::uint32_t initial_draft_n = 3;
    std::uint32_t max_draft_n = 6;
    // Bounded sliding ring buffer length per policy entry (max 256).
    std::size_t window_size = 16;
    // Minimum qualifying speculative observations before stepping up/down or pausing.
    std::size_t min_samples = 4;
    // Pause when rolling acceptance drops strictly below this threshold
    // (requires min_draft_n == 0).
    double pause_acceptance_threshold = 0.25;
    // Step down by 1 when rolling acceptance drops strictly below this threshold.
    double step_down_acceptance_threshold = 0.45;
    // Step up by 1 when rolling acceptance reaches or exceeds this threshold
    // (subject to measured-cost and hysteresis gates).
    double step_up_acceptance_threshold = 0.72;
    // After pausing, skip this many recommend() requests before issuing a
    // single probe_draft_n re-probe. 0 disables automatic re-probing.
    std::uint32_t cooldown_requests = 8;
    std::uint32_t probe_draft_n = 1;
    // Ignore speculative observations with fewer drafted tokens than this floor as noisy.
    std::uint64_t min_draft_tokens_per_sample = 4;

    // ---- Measured-cost & bounded-LRU controls ----
    // Maximum number of (target, draft, context, concurrency, workload) entries
    // retained in the LRU state table (in [1, kMaxEntries]).
    std::size_t max_entries = kDefaultMaxEntries;
    // Minimum K=0 baseline observations required before comparing L_hat(K) vs L_hat(0).
    std::size_t min_baseline_samples = 2;
    // Step down K when measured L_hat(K) / L_hat(ref) >= cost_step_down_ratio (>= 1.0).
    double cost_step_down_ratio = 1.05;
    // Pause (choose K=0 when min_draft_n == 0) when L_hat(K) / L_hat(0) >= cost_pause_ratio
    // even if draft acceptance is high.
    double cost_pause_ratio = 1.12;
    // Maximum allowed L_hat(probe) / L_hat(0) ratio for a cooldown probe to unpause
    // when a measured K=0 baseline exists (<= cost_step_down_ratio).
    double cost_probe_recover_ratio = 0.98;
    // Observations to hold off stepping back UP after a cost-driven step-down
    // so K <-> K+1 cannot oscillate.
    std::uint32_t cost_step_cooldown = 4;
    // Optional periodic K=0 baseline calibration interval (requests per band while
    // active; 0 disables automatic active-band K=0 probes).
    std::uint32_t baseline_probe_interval = 0;
};

[[nodiscard]] Status validate_adaptive_speculation_config(const AdaptiveSpeculationConfig &config);

struct SpeculationObservation {
    SpeculationWorkload workload = SpeculationWorkload::general;
    // Optional full band key. When `has_context_key` is true, `context_key` is
    // authoritative (including its workload); otherwise `workload` is used with
    // default identity and solo/short bands.
    SpeculationContextKey context_key{};
    bool has_context_key = false;
    // Effective draft_n_max used to execute this request. Set to 0 to record a
    // measured K=0 (no-speculation) baseline observation.
    std::optional<std::uint32_t> draft_n_used = std::nullopt;
    std::uint64_t draft_n = 0;
    std::uint64_t draft_n_accepted = 0;
    std::uint64_t predicted_n = 0;
    // Optional measured decode throughput in tokens/sec; 0 when unknown.
    double decode_tps = 0.0;
    // Preferred non-overlapping elapsed decode interval in milliseconds
    // (MUST exclude prefix/prompt prefill & state-restore time). 0 when unknown.
    double decode_elapsed_ms = 0.0;
    // Optional phase durations for diagnosis or critical-path fallback when
    // decode_elapsed_ms is unavailable (never double-counted if decode_elapsed_ms > 0).
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double rollback_ms = 0.0;
    double coordination_ms = 0.0;
    bool draft_verify_overlapped = false;
    // Prompt prefill / prefix-restore duration in ms, recorded strictly OUTSIDE
    // decode-only ms/committed-token for end-to-end diagnostics.
    double prompt_or_restore_ms = 0.0;
    // True when this observation belongs to a probe issued by recommend().
    bool is_probe = false;
    // Monotonic probe identifier returned by recommend() when is_probe == true (0 = unspecified).
    std::uint64_t probe_id = 0;
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
    // Monotonic non-zero probe identifier when is_probe == true; 0 otherwise.
    std::uint64_t probe_id = 0;
    // Measured cost per committed token L_hat (ms/token, 0 when unknown).
    double rolling_ms_per_token = 0.0;
    // Measured K=0 baseline cost per committed token (ms/token, 0 when unknown).
    double baseline_ms_per_token = 0.0;
    // Cost ratio L_hat(K) / L_hat(0) when both are measured; 0 otherwise.
    double cost_ratio = 0.0;
    // True when this probe is an exploratory K=0 baseline calibration probe.
    bool is_baseline_probe = false;
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

    // Recommends `draft_n_max` for the next request of `workload` (default
    // identity, short context, solo concurrency).
    [[nodiscard]] SpeculationDecision recommend(SpeculationWorkload workload);

    // Recommends `draft_n_max` for a full composite key. When `allow_probe` is
    // false (e.g. latency-sensitive interactive foreground request), no
    // exploratory cooldown or baseline calibration probe is assigned to this
    // request.
    [[nodiscard]] SpeculationDecision recommend(const SpeculationContextKey &key, bool allow_probe = true);

    // Records a completed request's timing and draft counters, updates the
    // bounded sliding window / K=0 baseline / per-K cost estimates with
    // overflow-safe accumulation, clamps malformed `draft_n_accepted > draft_n`,
    // applies measured-cost and acceptance hysteresis decisions, and returns
    // the updated decision.
    SpeculationDecision observe(const SpeculationObservation &obs);

    // Releases an in-flight probe (cooldown re-probe or baseline calibration)
    // and restarts cooldown/interval. When `probe_id` is non-zero, only cancels
    // if it matches the currently active probe for that entry.
    void cancel_probe(SpeculationWorkload workload, std::uint64_t probe_id = 0);
    void cancel_probe(const SpeculationContextKey &key, std::uint64_t probe_id = 0);

    // Bounded LRU table inspection helpers.
    [[nodiscard]] std::size_t entry_count() const;
    [[nodiscard]] std::uint64_t table_evictions() const;

    // Observatory telemetry and /health inspection snapshot.
    [[nodiscard]] json::Object status_json() const;

    // Clears all entries from the bounded table while keeping probe_id strictly monotonic.
    void reset();

  private:
    enum class ProbeKind : std::uint8_t {
        none = 0,
        cooldown_reprobe = 1,
        baseline_calibration = 2,
    };

    struct Sample {
        std::uint32_t draft_n_used = 0;
        std::uint64_t draft_n = 0;
        std::uint64_t draft_n_accepted = 0;
        std::uint64_t committed_tokens = 0;
        double decode_elapsed_ms = 0.0;
        double ms_per_token = 0.0;
        double decode_tps = 0.0;
        double prompt_or_restore_ms = 0.0;
    };

    struct BaselineSample {
        std::uint64_t committed_tokens = 0;
        double decode_elapsed_ms = 0.0;
        double ms_per_token = 0.0;
    };

    struct DraftCostStats {
        std::uint32_t draft_n = 0;
        std::size_t sample_count = 0;
        double ema_ms_per_token = 0.0;
        double ema_acceptance = 0.0;
        std::uint64_t last_seen_seq = 0;
    };

    struct WindowStats {
        double rolling_acceptance = 0.0;
        double rolling_decode_tps = 0.0;
        double rolling_ms_per_token = 0.0;
        double rolling_prompt_or_restore_ms = 0.0;
        std::uint64_t draft_tokens = 0;
        std::uint64_t accepted_tokens = 0;
    };

    struct BaselineStats {
        double ms_per_token = 0.0;
        std::size_t sample_count = 0;
    };

    struct BucketState {
        std::vector<Sample> window;
        std::size_t head = 0;
        std::size_t count = 0;

        std::vector<BaselineSample> baseline_window;
        std::size_t baseline_head = 0;
        std::size_t baseline_count = 0;

        std::vector<DraftCostStats> per_k_costs;
        std::uint64_t cost_seq = 0;

        std::uint32_t current_draft_n = 0;
        bool paused = false;
        bool probe_in_flight = false;
        ProbeKind active_probe_kind = ProbeKind::none;
        std::uint64_t active_probe_id = 0;
        std::uint32_t cooldown_remaining = 0;
        std::uint32_t step_up_hold_remaining = 0;
        std::uint32_t requests_since_baseline = 0;

        std::uint64_t total_observations = 0;
        std::uint64_t baseline_observations = 0;
        std::uint64_t ignored_observations = 0;
        std::uint64_t clamped_observations = 0;
        std::uint64_t probes_issued = 0;
        std::uint64_t probe_recoveries = 0;
        std::uint64_t cost_step_downs = 0;
        std::uint64_t cost_pauses = 0;
        std::string last_reason;
    };

    struct TableEntry {
        SpeculationContextKey key;
        BucketState state;
    };

    static SpeculationContextKey normalize_key(const SpeculationContextKey &key);
    static SpeculationContextKey make_default_key(SpeculationWorkload workload);
    static void clear_window(BucketState &bucket) noexcept;
    static void push_sample(BucketState &bucket, const Sample &sample) noexcept;
    static void push_baseline_sample(BucketState &bucket, const BaselineSample &sample) noexcept;
    static void update_per_k_cost(BucketState &bucket, std::uint32_t draft_n, double ms_per_token,
                                  double acceptance) noexcept;
    [[nodiscard]] static const DraftCostStats *find_per_k_cost(const BucketState &bucket,
                                                               std::uint32_t draft_n) noexcept;
    [[nodiscard]] static WindowStats compute_window_stats(const BucketState &bucket) noexcept;
    [[nodiscard]] static BaselineStats compute_baseline_stats(const BucketState &bucket) noexcept;
    void reset_bucket_locked(BucketState &bucket) const;
    BucketState &find_or_create_locked(const SpeculationContextKey &key);
    BucketState *find_existing_locked(const SpeculationContextKey &key);
    [[nodiscard]] json::Object serialize_bucket_locked(const BucketState &bucket,
                                                       const SpeculationContextKey *key) const;
    [[nodiscard]] SpeculationDecision evaluate_active_locked(BucketState &bucket, std::uint32_t effective_k,
                                                             const WindowStats &stats,
                                                             const BaselineStats &baseline_stats,
                                                             bool decrement_hold,
                                                             bool cost_regression_only = false);

    AdaptiveSpeculationConfig config_;
    mutable std::mutex mutex_;
    std::uint64_t next_probe_id_ = 0;
    std::list<TableEntry> lru_;
    std::unordered_map<SpeculationContextKey, std::list<TableEntry>::iterator, SpeculationContextKeyHash> index_;
    std::uint64_t table_evictions_ = 0;
};

}  // namespace sonder::inference::llamaserver
