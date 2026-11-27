// Internal: live telemetry hub for `sonder-infer serve` (contract sections
// 5.2-5.4, docs/SERVER.md "Live telemetry").
//
// A TelemetrySink on the TelemetryBus writer thread. It keeps a bounded ring
// of the most recent envelopes (for resume) and fans each event out to at
// most `max_subscribers` bounded per-subscriber queues. write_event() does no
// I/O and never waits on a subscriber: a subscriber that falls behind loses
// its oldest undelivered events (counted per subscriber and in the hub
// total). Stream threads drain their queue and do the socket writes.
#pragma once

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
#include <unordered_map>
#include <vector>

#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server::detail {

struct HubEvent {
    std::uint64_t sequence = 0;
    std::string line;  // one serialized envelope, no newline
};
using HubEventPtr = std::shared_ptr<const HubEvent>;

class Subscription {
public:
    explicit Subscription(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    struct Batch {
        std::vector<HubEventPtr> events;
        std::uint64_t dropped = 0;  // events lost since the previous batch
        bool closed = false;        // hub closed and nothing left to deliver
    };

    // Waits up to `timeout` for events (or closure, or `*stop`), then takes
    // everything queued.
    Batch next(std::chrono::milliseconds timeout, const std::atomic<bool>* stop);

    [[nodiscard]] std::uint64_t dropped_total() const;

private:
    friend class LiveTelemetryHub;
    // Hub side (hub mutex held).
    void push(const HubEventPtr& event);
    void close();

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<HubEventPtr> queue_;
    std::uint64_t dropped_pending_ = 0;
    std::uint64_t dropped_total_ = 0;
    bool closed_ = false;
};

struct ResumeRequest {
    std::optional<std::string> last_event_id;  // Last-Event-ID or ?last_event_id=
    bool since_now = false;                     // ?since=now
};

struct HubStats {
    std::size_t subscribers = 0;
    std::size_t retained = 0;         // lifecycle ring size (the resume window)
    std::size_t capacity = 0;         // lifecycle ring budget
    std::size_t sample_retained = 0;  // distinct sample events kept (history plus latest-per-key)
    std::size_t sample_capacity = 0;  // sample history budget
    std::optional<std::uint64_t> oldest_sequence;  // oldest lifecycle event in the ring
    std::uint64_t next_sequence = 0;
    std::uint64_t subscriber_dropped_events = 0;
};

class LiveTelemetryHub final : public TelemetrySink {
public:
    // `capacity` is the lifecycle budget (everything except `*.sample`
    // events). Periodic `*.sample` events are retained separately: the latest
    // per (event type, device/backend) plus a short history of
    // `sample_history` events (0 = max(1, capacity / 16)), so a sample flood
    // can never evict lifecycle events.
    LiveTelemetryHub(std::size_t capacity, std::size_t max_subscribers, std::size_t subscriber_queue,
                     std::size_t sample_history = 0);

    // The bus instance id (event_id prefix); set once the engine exists.
    void set_instance_id(std::string instance_id);
    [[nodiscard]] std::string instance_id() const;

    // TelemetrySink. write() recovers the sequence from the envelope; the bus
    // calls write_event().
    void write(std::string_view json_line) override;
    void write_event(std::uint64_t sequence, std::string_view json_line) override;

    enum class SubscribeError { none, over_capacity, closed };
    struct Subscribed {
        std::shared_ptr<Subscription> subscription;
        SubscribeError error = SubscribeError::none;
        // ": resume-gap <from>-<to>" when the requested id is older than the
        // retained window (SSE only; empty otherwise).
        std::string gap_comment;
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
    const std::size_t capacity_;
    const std::size_t max_subscribers_;
    const std::size_t subscriber_queue_;

    // Every retained event in sequence order (mutex_ held).
    [[nodiscard]] std::vector<HubEventPtr> collect_locked() const;
    [[nodiscard]] std::size_t sample_retained_locked() const;

    const std::size_t sample_capacity_;

    mutable std::mutex mutex_;
    std::string instance_id_;
    std::deque<HubEventPtr> ring_;  // lifecycle events
    std::deque<HubEventPtr> sample_history_;
    std::unordered_map<std::string, HubEventPtr> sample_latest_;
    std::uint64_t lifecycle_floor_ = 0;  // sequence after the newest evicted lifecycle event
    std::uint64_t next_sequence_ = 0;
    std::vector<std::shared_ptr<Subscription>> subscribers_;
    std::uint64_t departed_dropped_ = 0;  // drops of subscribers already gone
    bool closed_ = false;
};

}  // namespace sonder::inference::server::detail
