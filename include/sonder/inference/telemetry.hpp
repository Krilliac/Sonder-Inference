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

// Correlation identifiers carried on every event (OBSERVATORY_CONTRACT.md).
struct TelemetryContext {
    std::string session_id;  // required by the envelope; must be non-empty
    std::optional<std::string> run_id;
    std::optional<std::string> request_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    std::optional<std::string> model_instance_id;
    std::optional<std::string> device_id;
};

// Receives fully serialized single-line JSON envelopes (no trailing newline).
// Called only from the telemetry writer thread. An exception from write_event()
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
    std::string producer_name = "sonder-inference";
    std::string node_id;  // defaults to host name
    // Envelope producer.role (additive field, contract sonder.observatory.event/1).
    // Empty omits the field.
    std::string role = "inference";
    // Envelope producer.synthetic: true when the events describe synthetic
    // work (the MOCK backend), false when the host knows they do not.
    // Unset (the default) omits the field, meaning "unknown": hosts that do
    // not decide it (the CLI, the C ABI, bench) never claim real output.
    std::optional<bool> synthetic;
    // Minimum spacing between live telemetry.dropped reports while running.
    // The first drop is reported as soon as the writer catches up; a final
    // report is emitted at shutdown if anything is still unreported.
    std::chrono::milliseconds drop_report_interval{1000};
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

    // Non-blocking. Returns false if the event was filtered or dropped.
    bool emit(std::string_view event_type, const TelemetryContext& context, json::Object attributes,
              TelemetryLevel level = TelemetryLevel::metrics);

    // Blocks until the writer has processed everything queued so far. Failed
    // sinks are retired; draining completes even when no healthy sink remains.
    void flush();
    // Emits a final telemetry.dropped event if anything was dropped, drains, and
    // stops the writer. Idempotent; called by the destructor.
    void shutdown();

    [[nodiscard]] std::uint64_t emitted_events() const noexcept { return emitted_.load(); }
    [[nodiscard]] std::uint64_t dropped_events() const noexcept { return dropped_.load(); }
    // Distinct sink objects retired after a write/flush exception, counted once
    // even if registered more than once. Separate from queue-pressure drops.
    [[nodiscard]] std::uint64_t failed_sinks() const noexcept { return failed_sinks_.load(); }

    // Stream identity: event_id is "<instance_id>-<sequence>" and
    // producer.instance_id carries the same value (docs/TELEMETRY.md).
    [[nodiscard]] const std::string& instance_id() const noexcept { return instance_id_; }

    // Builds an envelope without queueing it (exposed for tests/tools).
    // sampling.level is the level the event was emitted at.
    json::Object make_envelope(std::string_view event_type, const TelemetryContext& context,
                               json::Object attributes, std::uint64_t sequence,
                               TelemetryLevel level = TelemetryLevel::metrics) const;

private:
    void writer_loop();
    // Queues a telemetry.dropped report (bypasses the capacity limit). Lock held.
    void enqueue_drop_report_locked(bool final_report);

    TelemetryOptions options_;
    std::string instance_id_;
    std::vector<std::shared_ptr<TelemetrySink>> sinks_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drained_cv_;
    // (sequence, serialized envelope) in emission order.
    std::deque<std::pair<std::uint64_t, std::string>> queue_;
    std::uint64_t next_sequence_ = 0;
    std::uint64_t written_sequence_ = 0;  // count of lines handed to sinks
    std::uint64_t enqueued_count_ = 0;
    bool stopping_ = false;
    bool stopped_ = false;
    std::atomic<std::uint64_t> emitted_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> failed_sinks_{0};
    std::uint64_t reported_dropped_ = 0;
    std::chrono::steady_clock::time_point last_drop_report_{};
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
