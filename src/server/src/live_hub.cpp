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

namespace {
constexpr std::size_t kMaxSampleKeys = 64;

// Raw (still escaped) string value of the first `"<key>":"` in a serialized
// envelope; empty when absent. A raw quote cannot occur inside a JSON string,
// so the first match is a key.
std::string_view raw_string_field(std::string_view line, std::string_view needle) {
    const std::size_t at = line.find(needle);
    if (at == std::string_view::npos) {
        return {};
    }
    const std::size_t begin = at + needle.size();
    std::size_t end = begin;
    while (end < line.size() && line[end] != '"') {
        end += (line[end] == '\\' && end + 1 < line.size()) ? 2 : 1;
    }
    return line.substr(begin, std::min(end, line.size()) - begin);
}

// Retention key for a periodic sample event, or nullopt for lifecycle events.
std::optional<std::string> sample_key(std::string_view line) {
    const std::string_view type = raw_string_field(line, "\"event_type\":\"");
    constexpr std::string_view suffix = ".sample";
    if (type.size() <= suffix.size() || type.substr(type.size() - suffix.size()) != suffix) {
        return std::nullopt;
    }
    std::string key(type);
    key += '|';
    key += raw_string_field(line, "\"device_id\":\"");
    key += '|';
    key += raw_string_field(line, "\"backend\":\"");
    return key;
}
}  // namespace

LiveTelemetryHub::LiveTelemetryHub(std::size_t capacity, std::size_t max_subscribers, std::size_t subscriber_queue,
                                   std::size_t sample_history)
    : capacity_(capacity == 0 ? 1 : capacity),
      max_subscribers_(max_subscribers),
      // A full replay is the lifecycle ring plus the sample history plus at
      // most kMaxSampleKeys latest-per-key samples; the default queue holds
      // all of it so a fresh subscriber never loses retained events.
      subscriber_queue_(subscriber_queue != 0
                            ? subscriber_queue
                            : (capacity == 0 ? 1 : capacity) +
                                  (sample_history != 0 ? sample_history
                                                       : std::max<std::size_t>(1, (capacity == 0 ? 1 : capacity) / 16)) +
                                  kMaxSampleKeys),
      sample_capacity_(sample_history != 0 ? sample_history
                                           : std::max<std::size_t>(1, (capacity == 0 ? 1 : capacity) / 16)) {}

std::vector<HubEventPtr> LiveTelemetryHub::collect_locked() const {
    std::vector<HubEventPtr> all;
    all.reserve(ring_.size() + sample_history_.size() + sample_latest_.size());
    all.insert(all.end(), ring_.begin(), ring_.end());
    all.insert(all.end(), sample_history_.begin(), sample_history_.end());
    for (const auto& kv : sample_latest_) {
        all.push_back(kv.second);
    }
    std::sort(all.begin(), all.end(),
              [](const HubEventPtr& a, const HubEventPtr& b) { return a->sequence < b->sequence; });
    all.erase(std::unique(all.begin(), all.end(),
                          [](const HubEventPtr& a, const HubEventPtr& b) {
                              return a == b || (a->sequence == b->sequence && a->line == b->line);
                          }),
              all.end());
    return all;
}

std::size_t LiveTelemetryHub::sample_retained_locked() const {
    std::size_t n = sample_history_.size();
    for (const auto& kv : sample_latest_) {
        if (sample_history_.empty() || kv.second->sequence < sample_history_.front()->sequence) {
            ++n;
        }
    }
    return n;
}

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
    auto key = sample_key(event->line);  // classified outside the lock
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    if (key) {
        sample_history_.push_back(event);
        while (sample_history_.size() > sample_capacity_) {
            sample_history_.pop_front();
        }
        const auto it = sample_latest_.find(*key);
        if (it != sample_latest_.end()) {
            it->second = event;
        } else {
            if (sample_latest_.size() >= kMaxSampleKeys) {
                auto oldest = sample_latest_.begin();
                for (auto cur = sample_latest_.begin(); cur != sample_latest_.end(); ++cur) {
                    if (cur->second->sequence < oldest->second->sequence) {
                        oldest = cur;
                    }
                }
                sample_latest_.erase(oldest);
            }
            sample_latest_.emplace(std::move(*key), event);
        }
    } else {
        ring_.push_back(event);
        while (ring_.size() > capacity_) {
            lifecycle_floor_ = std::max(lifecycle_floor_, ring_.front()->sequence + 1);
            ring_.pop_front();
        }
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

    // Replay = lifecycle ring plus retained samples, in sequence order.
    const std::vector<HubEventPtr> replay = collect_locked();
    std::size_t start = 0;
    bool resumed = false;
    std::uint64_t resumed_seq = 0;
    if (resume.since_now) {
        start = replay.size();
    } else if (resume.last_event_id) {
        std::string instance;
        std::uint64_t seq = 0;
        if (parse_event_id(*resume.last_event_id, instance, seq) && instance == instance_id_ && !replay.empty()) {
            if (seq + 1 < lifecycle_floor_) {
                // Lifecycle events after the id were evicted: announce the gap
                // (lifecycle loss only; retained samples are a declared sparse class).
                out.gap_comment =
                    ": resume-gap " + std::to_string(seq + 1) + "-" + std::to_string(lifecycle_floor_ - 1);
            }
            resumed = true;
            resumed_seq = seq;
            // Either way replay only what is newer than the id; never go backwards.
            start = static_cast<std::size_t>(
                std::partition_point(replay.begin(), replay.end(),
                                     [seq](const HubEventPtr& e) { return e->sequence <= seq; }) -
                replay.begin());
        }
        // Unknown or different instance (producer restarted), malformed id,
        // or nothing retained: replay the whole retained window.
    }
    // Announce every sequence hole inside the replay (evicted samples or
    // evicted lifecycle events), so no discontinuity is unannounced (SSE only;
    // NDJSON has no comment channel).
    std::uint64_t expected = 0;
    bool have_expected = false;
    if (resumed) {
        expected = resumed_seq + 1;
        have_expected = true;
    }
    if (!out.gap_comment.empty()) {
        expected = std::max(expected, lifecycle_floor_);
        have_expected = true;
    }
    for (std::size_t i = start; i < replay.size(); ++i) {
        const std::uint64_t seq_i = replay[i]->sequence;
        if (have_expected && seq_i > expected) {
            if (!out.gap_comment.empty()) {
                out.gap_comment += "\n\n";
            }
            out.gap_comment += ": resume-gap " + std::to_string(expected) + "-" + std::to_string(seq_i - 1);
        }
        expected = std::max(expected, seq_i + 1);
        have_expected = true;
        sub->push(replay[i]);
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
    s.sample_capacity = sample_capacity_;
    s.sample_retained = sample_retained_locked();
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
