// Internal: live telemetry hub for `sonder-infer serve` (contract sections
// 5.2-5.4, docs/SERVER.md "Live telemetry").
//
// A TelemetrySink on the TelemetryBus writer thread. It keeps one bounded ring
// per sequence stream (the main stream and the periodic-sample stream, see
// TelemetryStream) of the most recent envelopes (for resume) and fans each
// event out to at most `max_subscribers` bounded per-subscriber queues. Each
// ring holds a contiguous suffix of its own stream, so a replay is contiguous
// per stream and samples never evict lifecycle events. write_stream_event()
// does no I/O and never waits on a subscriber: a subscriber that falls behind
// loses its oldest undelivered events (counted per subscriber and in the hub
// total). Stream threads drain their queue and do the socket writes.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server::detail {

struct HubEvent {
    TelemetryStream stream = TelemetryStream::main;
    std::uint64_t sequence = 0;  // within `stream`
    std::uint64_t order = 0;     // hub arrival ordinal, across streams
    std::string line;            // one serialized envelope, no newline
};
using HubEventPtr = std::shared_ptr<const HubEvent>;

// A resume cursor older than a stream's retained window: sequences
// [from, to] of `stream` are no longer available.
struct ReplayGap {
    TelemetryStream stream = TelemetryStream::main;
    std::uint64_t from = 0;
    std::uint64_t to = 0;
};

class Subscription {
public:
    explicit Subscription(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    struct Batch {
        // A replay gap announced directly before events[before] (SSE only).
        struct Gap {
            std::size_t before = 0;
            ReplayGap gap;
        };
        std::vector<HubEventPtr> events;
        std::vector<Gap> gaps;      // at most one per stream, ascending `before`
        std::uint64_t dropped = 0;  // events lost since the previous batch
        bool closed = false;        // hub closed and nothing left to deliver
    };

    // Waits up to `timeout` for events (or closure, or `*stop`), then takes
    // everything queued.
    Batch next(std::chrono::milliseconds timeout, const std::atomic<bool>* stop);

    [[nodiscard]] std::uint64_t dropped_total() const;

private:
    friend class LiveTelemetryHub;
    // Hub side (hub mutex held). `gap_before`, when set, is announced directly
    // before `event`; it is forgotten if overflow evicts `event` first (the
    // `dropped` count covers it).
    void push(const HubEventPtr& event, const ReplayGap* gap_before = nullptr);
    void close();

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<HubEventPtr> queue_;
    // Gaps for events still in queue_ (at most one per stream).
    std::vector<std::pair<const HubEvent*, ReplayGap>> gaps_;
    std::uint64_t dropped_pending_ = 0;
    std::uint64_t dropped_total_ = 0;
    bool closed_ = false;
};

struct ResumeRequest {
    // Last-Event-ID or ?last_event_id=: one event id, or several separated by
    // ',' (at most one per stream).
    std::optional<std::string> last_event_id;
    bool since_now = false;  // ?since=now
};

struct HubStats {
    std::size_t subscribers = 0;
    // Main stream ring (the default stream of the resume discovery fields).
    std::size_t retained = 0;
    std::size_t capacity = 0;
    std::optional<std::uint64_t> oldest_sequence;
    std::uint64_t next_sequence = 0;
    // Samples stream ring.
    std::size_t sample_retained = 0;
    std::size_t sample_capacity = 0;
    std::optional<std::uint64_t> sample_oldest_sequence;
    std::uint64_t sample_next_sequence = 0;
    std::uint64_t subscriber_dropped_events = 0;
};

class LiveTelemetryHub final : public TelemetrySink {
public:
    // `capacity` is the retained-event budget of the main stream and
    // `sample_capacity` that of the samples stream (0 = `capacity`).
    // `subscriber_queue` 0 = capacity + sample capacity, so a fresh
    // subscriber can hold a full replay.
    LiveTelemetryHub(std::size_t capacity, std::size_t max_subscribers, std::size_t subscriber_queue,
                     std::size_t sample_capacity = 0);

    // The bus instance id (event_id prefix); set once the engine exists.
    void set_instance_id(std::string instance_id);
    [[nodiscard]] std::string instance_id() const;

    // TelemetrySink. The bus calls write_stream_event(); write() serves
    // producers other than the TelemetryBus and recovers the sequence and the
    // stream (`producer.stream`) from the envelope.
    void write(std::string_view json_line) override;
    void write_event(std::uint64_t sequence, std::string_view json_line) override;
    void write_stream_event(TelemetryStream stream, std::uint64_t sequence, std::string_view json_line) override;

    enum class SubscribeError { none, over_capacity, closed };
    struct Subscribed {
        std::shared_ptr<Subscription> subscription;
        SubscribeError error = SubscribeError::none;
    };
    // Registers a subscriber and preloads the replay chosen by `resume`
    // (contract 5.3).
    Subscribed subscribe(const ResumeRequest& resume);
    void unsubscribe(const std::shared_ptr<Subscription>& subscription);

    // Ends every subscription once it has delivered what it holds; later
    // subscribe() calls fail with closed.
    void close();

    [[nodiscard]] HubStats stats() const;

private:
    struct Ring {
        std::deque<HubEventPtr> events;  // contiguous suffix of the stream
        std::size_t capacity = 1;
        std::uint64_t next_sequence = 0;
    };

    const std::size_t max_subscribers_;
    const std::size_t subscriber_queue_;

    mutable std::mutex mutex_;
    std::string instance_id_;
    std::array<Ring, kTelemetryStreamCount> rings_;
    std::uint64_t next_order_ = 0;
    std::vector<std::shared_ptr<Subscription>> subscribers_;
    std::uint64_t departed_dropped_ = 0;  // drops of subscribers already gone
    bool closed_ = false;
};

// Wire framing of one Subscription::Batch (used by the telemetry route, and by
// tests so they exercise the real framing). SSE: a `: dropped <n>` comment
// block when batch.dropped > 0, then per event an optional
// `: resume-gap <from>-<to>` (main) or `: resume-gap s<from>-s<to>` (samples)
// comment block directly before it, then `id: <instance>-<seq>` /
// `id: <instance>-s<seq>` and `data: <envelope>`. Every block ends with a
// blank line. NDJSON: one envelope line per event and nothing else.
[[nodiscard]] std::string format_sse_batch(std::string_view instance_id, const Subscription::Batch& batch);
[[nodiscard]] std::string format_ndjson_batch(const Subscription::Batch& batch);

}  // namespace sonder::inference::server::detail
