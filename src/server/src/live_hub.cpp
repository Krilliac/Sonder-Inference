#include "live_hub.hpp"

#include <algorithm>

#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// ----------------------------------------------------------- Subscription

void Subscription::push(const HubEventPtr& event) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        queue_.push_back(event);
        while (queue_.size() > capacity_) {
            queue_.pop_front();
            ++dropped_pending_;
            ++dropped_total_;
        }
    }
    cv_.notify_one();
}

void Subscription::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

Subscription::Batch Subscription::next(std::chrono::milliseconds timeout, const std::atomic<bool>* stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    // Wake in short slices so an external stop flag is observed.
    while (queue_.empty() && !closed_ && dropped_pending_ == 0) {
        if (stop != nullptr && stop->load(std::memory_order_acquire)) {
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            break;
        }
        cv_.wait_for(lock, std::min<std::chrono::steady_clock::duration>(deadline - now, std::chrono::milliseconds(50)));
    }
    Batch batch;
    batch.events.assign(queue_.begin(), queue_.end());
    queue_.clear();
    batch.dropped = dropped_pending_;
    dropped_pending_ = 0;
    batch.closed = closed_ && batch.events.empty();
    return batch;
}

std::uint64_t Subscription::dropped_total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_total_;
}

// -------------------------------------------------------- LiveTelemetryHub

LiveTelemetryHub::LiveTelemetryHub(std::size_t capacity, std::size_t max_subscribers, std::size_t subscriber_queue)
    : capacity_(capacity == 0 ? 1 : capacity),
      max_subscribers_(max_subscribers),
      subscriber_queue_(subscriber_queue == 0 ? (capacity == 0 ? 1 : capacity) : subscriber_queue) {}

void LiveTelemetryHub::set_instance_id(std::string instance_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    instance_id_ = std::move(instance_id);
}

std::string LiveTelemetryHub::instance_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return instance_id_;
}

void LiveTelemetryHub::write(std::string_view json_line) {
    // Only reached when a producer other than TelemetryBus feeds the hub.
    auto parsed = json::parse(json_line);
    if (!parsed.ok()) {
        return;
    }
    const json::Value* seq = parsed.value().find("sequence");
    if (seq == nullptr || !seq->is_integer() || seq->as_int() < 0) {
        return;
    }
    write_event(static_cast<std::uint64_t>(seq->as_int()), json_line);
}

void LiveTelemetryHub::write_event(std::uint64_t sequence, std::string_view json_line) {
    auto event = std::make_shared<const HubEvent>(HubEvent{sequence, std::string(json_line)});
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    ring_.push_back(event);
    while (ring_.size() > capacity_) {
        ring_.pop_front();
    }
    next_sequence_ = std::max(next_sequence_, sequence + 1);
    for (const auto& sub : subscribers_) {
        sub->push(event);
    }
}

namespace {
// "<instance>-<sequence>", split at the last '-'.
bool parse_event_id(std::string_view id, std::string& instance, std::uint64_t& sequence) {
    const std::size_t dash = id.rfind('-');
    if (dash == std::string_view::npos || dash == 0 || dash + 1 >= id.size() || id.size() - dash - 1 > 19) {
        return false;
    }
    std::uint64_t n = 0;
    for (const char c : id.substr(dash + 1)) {
        if (c < '0' || c > '9') {
            return false;
        }
        n = n * 10 + static_cast<std::uint64_t>(c - '0');
    }
    instance = std::string(id.substr(0, dash));
    sequence = n;
    return true;
}
}  // namespace

LiveTelemetryHub::Subscribed LiveTelemetryHub::subscribe(const ResumeRequest& resume) {
    Subscribed out;
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        out.error = SubscribeError::closed;
        return out;
    }
    if (subscribers_.size() >= max_subscribers_) {
        out.error = SubscribeError::over_capacity;
        return out;
    }
    auto sub = std::make_shared<Subscription>(subscriber_queue_);

    // Choose where the replay starts (index into ring_).
    std::size_t start = 0;
    if (resume.since_now) {
        start = ring_.size();
    } else if (resume.last_event_id) {
        std::string instance;
        std::uint64_t seq = 0;
        if (parse_event_id(*resume.last_event_id, instance, seq) && instance == instance_id_ && !ring_.empty()) {
            const std::uint64_t oldest = ring_.front()->sequence;
            if (seq + 1 >= oldest) {
                // Resume right after the last seen event (nothing when caught up).
                start = static_cast<std::size_t>(std::min<std::uint64_t>(seq + 1 - oldest, ring_.size()));
            } else {
                // Older than the window: announce the gap, then the whole window.
                out.gap_comment = ": resume-gap " + std::to_string(seq + 1) + "-" + std::to_string(oldest - 1);
            }
        }
        // Unknown or different instance (producer restarted), malformed id,
        // or an empty ring: replay the whole retained window.
    }
    for (std::size_t i = start; i < ring_.size(); ++i) {
        sub->push(ring_[i]);
    }
    subscribers_.push_back(sub);
    out.subscription = std::move(sub);
    return out;
}

void LiveTelemetryHub::unsubscribe(const std::shared_ptr<Subscription>& subscription) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find(subscribers_.begin(), subscribers_.end(), subscription);
    if (it != subscribers_.end()) {
        departed_dropped_ += (*it)->dropped_total();
        subscribers_.erase(it);
    }
}

void LiveTelemetryHub::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    for (const auto& sub : subscribers_) {
        sub->close();
    }
}

HubStats LiveTelemetryHub::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    HubStats s;
    s.subscribers = subscribers_.size();
    s.retained = ring_.size();
    s.capacity = capacity_;
    if (!ring_.empty()) {
        s.oldest_sequence = ring_.front()->sequence;
    }
    s.next_sequence = next_sequence_;
    s.subscriber_dropped_events = departed_dropped_;
    for (const auto& sub : subscribers_) {
        s.subscriber_dropped_events += sub->dropped_total();
    }
    return s;
}

}  // namespace sonder::inference::server::detail
