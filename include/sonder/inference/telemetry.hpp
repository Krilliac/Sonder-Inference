// Sonder Inference: Observatory telemetry producer.
//
// Events use the Sonder Observatory envelope v1
// (schema "sonder.observatory.event/1", see
// https://github.com/Krilliac/Sonder-Observatory protocol/observatory-events.schema.json).
// Emission never blocks the caller: events go through a bounded queue drained
// by a background writer thread, and are dropped (and counted) under pressure.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "sonder/inference/json.hpp"

namespace sonder::inference {

inline constexpr std::string_view kObservatorySchema = "sonder.observatory.event/1";

// Matches the envelope "sampling.level" enum.
enum class TelemetryLevel { off = 0, metrics = 1, standard = 2, deep = 3 };

const char* to_string(TelemetryLevel level) noexcept;
std::optional<TelemetryLevel> parse_telemetry_level(std::string_view text) noexcept;

// ---------------------------------------------------------------------------
// Structured 5-state capability evidence, 3-tier cache accounting, effective
// execution reporting, and opt-in stage profiling (Doc 1 Package A + Doc 2 #25).
// ---------------------------------------------------------------------------

// Maximum byte length retained for bounded diagnostic strings in evidence
// records (model/draft identities, reasons, capability notes, overlap tags).
inline constexpr std::size_t kMaxEvidenceStringBytes = 256;
inline constexpr std::size_t kMaxCapabilityStatusEntries = 64;
inline constexpr std::size_t kMaxStageOverlapNotes = 16;

// Tri-state evidence value for one capability dimension. Unmeasured dimensions
// stay explicitly `unknown` rather than being fabricated as false or true.
enum class CapabilityState { unknown = 0, no = 1, yes = 2 };

[[nodiscard]] const char* to_string(CapabilityState state) noexcept;
[[nodiscard]] std::optional<CapabilityState> parse_capability_state(std::string_view text) noexcept;
[[nodiscard]] constexpr CapabilityState capability_state_from_bool(bool value) noexcept {
    return value ? CapabilityState::yes : CapabilityState::no;
}

// 5-state capability evidence model: never conflates adapter plumbing with
// upstream build support, active configuration, runtime observation, or
// measured workload speedup.
struct CapabilityStatus {
    CapabilityState adapter_supports = CapabilityState::unknown;
    CapabilityState build_supports = CapabilityState::unknown;
    CapabilityState configured = CapabilityState::unknown;
    CapabilityState observed_active = CapabilityState::unknown;
    CapabilityState qualified_beneficial = CapabilityState::unknown;
    // Optional short diagnostic note (bounded and control-char sanitized).
    std::string note;
};

using CapabilityStatusEntry = std::pair<std::string, CapabilityStatus>;

// Canonical disable/fallback reason labels for EffectiveExecutionReport.
inline constexpr std::string_view kSpecReasonNone = "none";
inline constexpr std::string_view kSpecReasonConcurrencyBatch = "concurrency_batch";
inline constexpr std::string_view kSpecReasonHybridUnsupported = "hybrid_unsupported";
inline constexpr std::string_view kSpecReasonRecurrentUnsupported = "recurrent_unsupported";
inline constexpr std::string_view kSpecReasonDisabledByPolicy = "disabled_by_policy";
inline constexpr std::string_view kSpecReasonNotSupported = "not_supported";
inline constexpr std::string_view kSpecReasonNotConfigured = "not_configured";
inline constexpr std::string_view kSpecReasonAutoFallbackNgramMod = "auto_fallback_ngram_mod";
inline constexpr std::string_view kSpecReasonMediaExclusive = "media_exclusive";
inline constexpr std::string_view kSpecReasonStateRestored = "state_restored";
inline constexpr std::string_view kSpecReasonDraftSyncFailed = "draft_sync_failed";
inline constexpr std::string_view kSpecReasonUnknown = "unknown";

// Separate 3-tier cache reuse counters. Unmeasured tiers stay std::nullopt.
//
// Invariant: `logical_cache_hit_tokens` (engine KvCacheManager accounting)
// MUST NEVER be copied into `device_prefix_reused_tokens` or
// `host_l2_restored_tokens`, and is strictly excluded from
// `physical_reused_tokens()`.
struct CacheTierCounters {
    // Tier 0 (accounting only): engine KvCacheManager logical prefix reuse.
    std::optional<std::uint64_t> logical_cache_hit_tokens = std::nullopt;
    // Tier 1 (physical device): in-VRAM same-slot or cross-slot seq_cp reuse.
    std::optional<std::uint64_t> device_prefix_reused_tokens = std::nullopt;
    // Optional sub-breakdown of device_prefix_reused_tokens via cross-slot seq_cp.
    std::optional<std::uint64_t> device_cross_slot_reused_tokens = std::nullopt;
    // Tier 2 (physical host RAM): serialized state restored from host L2 cache.
    std::optional<std::uint64_t> host_l2_restored_tokens = std::nullopt;
    std::optional<double> host_l2_restore_ms = std::nullopt;

    // Physical prefill tokens reused on the backend (device + host L2 only).
    // Returns std::nullopt when neither physical tier was measured; never
    // counts logical accounting hits as physical prefill savings.
    [[nodiscard]] std::optional<std::uint64_t> physical_reused_tokens() const noexcept;
};

// Bounded record of configured vs effective speculative execution, fallback
// reasons, target/draft identity, committed vs proposed token counts, and
// 3-tier cache reuse accounting.
struct EffectiveExecutionReport {
    std::string configured_spec_method;  // empty => "unknown"
    std::string effective_spec_method;   // empty => "unknown"
    std::optional<std::uint32_t> configured_draft_limit = std::nullopt;
    std::optional<std::uint32_t> effective_draft_limit = std::nullopt;
    // Explicit disable/fallback reason when effective differs from configured
    // or speculation was disabled/paused (empty when none).
    std::string fallback_reason;

    std::string target_identity;  // model name / GGUF basename; empty => unknown
    std::string draft_identity;   // draft artifact / mechanism id; empty => none/unknown
    std::string backend_version;  // upstream version/build tag; empty => unknown

    // Target-verified committed output tokens MUST stay separate from proposed
    // speculative draft tokens.
    std::optional<std::uint64_t> committed_tokens = std::nullopt;
    std::optional<std::uint64_t> proposed_draft_tokens = std::nullopt;
    std::optional<std::uint64_t> accepted_draft_tokens = std::nullopt;

    CacheTierCounters cache;

    CapabilityState speculation_observed_active = CapabilityState::unknown;
    // True when speculation was eligible at admission and transitioned to
    // disabled mid-request (e.g. concurrency batch transition).
    bool transitioned_during_request = false;
};

// Opt-in per-request execution stage timings (milliseconds).
// Unmeasured stages stay std::nullopt (never fabricated as 0.0).
//
// `total_elapsed_ms` is the wall-clock end-to-end span. Individual stage
// accumulators may overlap or nest (e.g. callback_gate_wait inside decode, or
// async GPU overlap); `stages_may_overlap` and `overlap_notes` keep overlapping
// phases clearly distinguished so consumers never double-count them.
struct StageTimings {
    bool enabled = false;
    std::optional<double> queue_wait_ms = std::nullopt;
    std::optional<double> cache_lookup_ms = std::nullopt;
    std::optional<double> l2_restore_ms = std::nullopt;
    std::optional<double> prefill_ms = std::nullopt;
    std::optional<double> draft_ms = std::nullopt;
    std::optional<double> verify_ms = std::nullopt;
    std::optional<double> rollback_ms = std::nullopt;
    std::optional<double> sampling_ms = std::nullopt;
    std::optional<double> callback_gate_wait_ms = std::nullopt;
    std::optional<double> total_elapsed_ms = std::nullopt;

    bool stages_may_overlap = false;
    std::vector<std::string> overlap_notes;

    // Adds a non-negative finite millisecond delta into `slot`.
    static void add_ms(std::optional<double>& slot, double delta_ms) noexcept;
    // Records a bounded, deduplicated overlap tag (capped at kMaxStageOverlapNotes).
    void note_overlap(std::string_view label);
    // True when at least one stage measurement is populated.
    [[nodiscard]] bool has_any_measurement() const noexcept;
};

// Zero-overhead opt-in RAII timer: touches steady_clock only when `slot != nullptr`.
class ScopedStageTimer {
public:
    ScopedStageTimer() noexcept = default;
    explicit ScopedStageTimer(std::optional<double>* slot) noexcept
        : slot_(slot),
          start_(slot != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~ScopedStageTimer() noexcept { stop(); }
    ScopedStageTimer(const ScopedStageTimer&) = delete;
    ScopedStageTimer& operator=(const ScopedStageTimer&) = delete;

    void stop() noexcept {
        if (slot_ != nullptr) {
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            StageTimings::add_ms(*slot_, ms);
            slot_ = nullptr;
        }
    }

private:
    std::optional<double>* slot_ = nullptr;
    std::chrono::steady_clock::time_point start_{};
};

// Strips control characters and caps length to `max_bytes` so diagnostic
// records stay bounded and safe for JSON/log emission.
[[nodiscard]] std::string sanitize_bounded_evidence_string(std::string_view text,
                                                           std::size_t max_bytes = kMaxEvidenceStringBytes);
[[nodiscard]] EffectiveExecutionReport bound_effective_execution_report(EffectiveExecutionReport report);

[[nodiscard]] json::Object to_json(const CapabilityStatus& status);
[[nodiscard]] json::Object to_json_capability_statuses(const std::vector<CapabilityStatusEntry>& entries);
[[nodiscard]] json::Object to_json(const CacheTierCounters& cache);
[[nodiscard]] json::Object to_json(const EffectiveExecutionReport& report);
[[nodiscard]] json::Object to_json(const StageTimings& timings);

// Correlation identifiers carried on every event (OBSERVATORY_CONTRACT.md).
struct TelemetryContext {
    std::string session_id;  // required by the envelope; must be non-empty
    std::optional<std::string> run_id;
    std::optional<std::string> request_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    std::optional<std::string> model_instance_id;
    std::optional<std::string> device_id;
    // Positive evidence that this event describes synthetic work. False
    // means no event-specific classification, so the host setting applies.
    // A known mock event cannot be labelled real by a host-wide false value.
    bool synthetic_work = false;
};

// Sequence stream an event belongs to. Every producer instance numbers each
// stream contiguously from 0 with its own counter (docs/TELEMETRY.md "Stream
// identity contract"). Periodic samples get their own stream so that
// retaining recent samples never leaves a hole between retained lifecycle
// events.
enum class TelemetryStream : std::uint8_t { main = 0, samples = 1 };
inline constexpr std::size_t kTelemetryStreamCount = 2;

// Event types ending in ".sample" belong to the samples stream; every other
// event is on the main (default) stream.
[[nodiscard]] TelemetryStream telemetry_stream_of(std::string_view event_type) noexcept;

// Receives fully serialized single-line JSON envelopes (no trailing newline).
// Called only from the telemetry writer thread. An exception from write_stream_event()
// or flush() retires this sink without retrying it or logging exception contents;
// healthy sinks continue. Implementations must not block indefinitely.
class TelemetrySink {
public:
    virtual ~TelemetrySink() = default;
    virtual void write(std::string_view json_line) = 0;
    // Same as write(), plus the envelope's stream sequence number (the
    // "<instance_id>-<sequence>" suffix of event_id). Sinks that index events
    // by sequence (the live telemetry hub of `sonder-infer serve`) override
    // it; the default forwards to write(). Called on the writer thread.
    virtual void write_event(std::uint64_t sequence, std::string_view json_line) {
        (void)sequence;
        write(json_line);
    }
    // What the bus calls for every event. `sequence` is the number within
    // `stream` (two streams share numbers, so it is not unique across them).
    // Main-stream events reach write_event(); events of any other stream reach
    // write(), so a sink that indexes by sequence never sees two number spaces
    // mixed. Override to receive the stream. Called on the writer thread.
    virtual void write_stream_event(TelemetryStream stream, std::uint64_t sequence, std::string_view json_line) {
        if (stream == TelemetryStream::main) {
            write_event(sequence, json_line);
        } else {
            write(json_line);
        }
    }
    virtual void flush() {}
};

// Appends JSONL to a file (created/truncated or appended).
std::unique_ptr<TelemetrySink> make_jsonl_file_sink(const std::string& path, bool append, Status* status = nullptr);
// Writes JSONL to an existing stream (not owned), e.g. std::cerr.
std::unique_ptr<TelemetrySink> make_ostream_sink(std::ostream& stream);

// Thread-safe in-memory sink for tests and embedding hosts.
class MemoryTelemetrySink final : public TelemetrySink {
public:
    void write(std::string_view json_line) override;
    [[nodiscard]] std::vector<std::string> lines() const;

private:
    mutable std::mutex mutex_;
    std::vector<std::string> lines_;
};

struct TelemetryOptions {
    TelemetryLevel level = TelemetryLevel::standard;
    std::size_t queue_capacity = 4096;
    // Include generated text in token events. Off by default: raw text capture
    // is independent from structural events per the Observatory contract.
    bool capture_text = false;
    // Opt-in per-request stage timing hooks (Doc 2 #25). Off by default so
    // unprofiled runs incur zero extra stage clock reads or timing events.
    bool enable_stage_profiling = false;
    std::string producer_name = "sonder-inference";
    std::string node_id;  // defaults to host name
    // Envelope producer.role (additive field, contract sonder.observatory.event/1).
    // Empty omits the field.
    std::string role = "inference";
    // Envelope producer.synthetic: true when the events describe synthetic
    // work (the MOCK backend), false when the host knows they do not.
    // Unset (the default) omits the field, meaning "unknown", except when
    // the event context positively identifies mock work. Hosts that do not
    // decide it (the CLI, the C ABI, bench) never claim real output.
    std::optional<bool> synthetic;
    // Minimum spacing between live telemetry.dropped reports while running.
    // The first drop is reported as soon as the writer catches up; a final
    // report is emitted at shutdown if anything is still unreported.
    std::chrono::milliseconds drop_report_interval{1000};
    // Longest shutdown() (and so the destructor) waits for the writer thread
    // to drain and exit. A sink stuck in write()/flush() (a stalled disk or
    // pipe) must not hang the process: on expiry the writer is detached and
    // shutdown() returns false. Zero or negative means do not wait at all.
    std::chrono::milliseconds shutdown_timeout{5000};
    // Backend-sampled token probabilities (llamaserver against llama.cpp),
    // used only when `standard` events are recorded. top_k is how many
    // alternatives the backend asks its upstream for per sampled token,
    // clamped to [0, kMaxTelemetryTopK = 10]; 0 turns the probe off (no
    // probability on backend chunks, no inference.sampling.candidates).
    int token_probability_top_k = 5;
    // inference.sampling.candidates volume bounds, per request: emit for
    // every `stride`-th probability-bearing token (0 is read as 1), and at
    // most `max_events` events (0 = no cap). The upstream still computes the
    // probabilities for every token of the request; these bound the stream.
    std::uint32_t token_candidates_stride = 1;
    std::uint32_t token_candidates_max_events = 256;
};

class TelemetryBus {
public:
    explicit TelemetryBus(TelemetryOptions options = {});
    ~TelemetryBus();
    TelemetryBus(const TelemetryBus&) = delete;
    TelemetryBus& operator=(const TelemetryBus&) = delete;

    // Sinks must be added before events are emitted.
    void add_sink(std::shared_ptr<TelemetrySink> sink);

    [[nodiscard]] const TelemetryOptions& options() const noexcept { return options_; }
    // True when an event of `level` would be recorded (level != off, sinks present).
    [[nodiscard]] bool enabled(TelemetryLevel level) const noexcept;
    [[nodiscard]] bool stage_profiling_enabled() const noexcept { return options_.enable_stage_profiling; }

    // Non-blocking. Returns false if the event was filtered or dropped.
    bool emit(std::string_view event_type, const TelemetryContext& context, json::Object attributes,
              TelemetryLevel level = TelemetryLevel::metrics);

    // Bounded, non-blocking helpers for structured effective execution evidence
    // and opt-in per-request stage timings.
    bool emit_effective_execution(const TelemetryContext& context, const EffectiveExecutionReport& report,
                                  TelemetryLevel level = TelemetryLevel::metrics);
    bool emit_stage_timings(const TelemetryContext& context, const StageTimings& timings,
                            std::string_view backend = {}, std::string_view backend_version = {},
                            TelemetryLevel level = TelemetryLevel::metrics);

    // Blocks until the writer has processed everything queued so far. Failed
    // sinks are retired; draining completes even when no healthy sink remains.
    void flush();
    // Emits a final telemetry.dropped event if anything was dropped, drains, and
    // stops the writer. Idempotent; called by the destructor.
    // Waits at most TelemetryOptions::shutdown_timeout for the writer. Returns
    // true when the writer drained and was joined; false when a sink did not
    // return in time: the writer is then detached and the events not yet
    // written to sinks are abandoned (a repeated call returns the first result).
    bool shutdown();

    [[nodiscard]] std::uint64_t emitted_events() const noexcept;
    [[nodiscard]] std::uint64_t dropped_events() const noexcept;
    // Distinct sink objects retired after a write/flush exception, counted once
    // even if registered more than once. Separate from queue-pressure drops.
    [[nodiscard]] std::uint64_t failed_sinks() const noexcept;

    // Stream identity: producer.instance_id names the producer instance; each
    // stream (main, samples) is numbered contiguously from 0. event_id is
    // "<instance_id>-<sequence>" on the main stream and
    // "<instance_id>-s<sequence>" on the samples stream, whose events also
    // carry producer.stream = "samples" (docs/TELEMETRY.md).
    [[nodiscard]] const std::string& instance_id() const noexcept { return instance_id_; }

    // Builds an envelope without queueing it (exposed for tests/tools).
    // `sequence` is the number within the event's stream (see
    // telemetry_stream_of). sampling.level is the level the event was emitted at.
    json::Object make_envelope(std::string_view event_type, const TelemetryContext& context,
                               json::Object attributes, std::uint64_t sequence,
                               TelemetryLevel level = TelemetryLevel::metrics) const;

private:
    // Queue, sinks, counters and flags, shared with the writer thread so a
    // writer abandoned by shutdown() never touches this object (defined in
    // telemetry.cpp).
    struct Shared;

    TelemetryOptions options_;
    std::string instance_id_;
    std::shared_ptr<Shared> shared_;
    // Touched by the constructor and shutdown() only.
    std::thread writer_;
};

// RFC 3339 UTC timestamp with millisecond precision, e.g. 2026-09-26T08:01:02.345Z.
std::string utc_timestamp_now();
// Monotonic nanoseconds (steady clock; epoch is platform-defined, usually
// boot). Envelope mono_ns uses this value; see docs/TELEMETRY.md for the
// 2^53 precision note.
std::uint64_t monotonic_ns() noexcept;
// Random-ish unique identifier with a prefix, e.g. "req-4f1c...".
std::string make_id(std::string_view prefix);

}  // namespace sonder::inference
