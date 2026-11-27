#include "live_hub.hpp"

#include <algorithm>

#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// ----------------------------------------------------------- Subscription

void Subscription::push(const HubEventPtr& event, const ReplayGap* gap_before) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        queue_.push_back(event);
        if (gap_before != nullptr) {
            gaps_.emplace_back(event.get(), *gap_before);
        }
        while (queue_.size() > capacity_) {
            const HubEvent* evicted = queue_.front().get();
            std::erase_if(gaps_, [evicted](const auto& entry) { return entry.first == evicted; });
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
    for (const auto& [event, gap] : gaps_) {
        // push() erased the entries of evicted events, so each remaining one
        // points at an event of this batch.
        const auto it = std::find_if(batch.events.begin(), batch.events.end(),
                                     [event](const HubEventPtr& e) { return e.get() == event; });
        if (it != batch.events.end()) {
            batch.gaps.push_back({static_cast<std::size_t>(it - batch.events.begin()), gap});
        }
    }
    std::sort(batch.gaps.begin(), batch.gaps.end(),
              [](const Batch::Gap& x, const Batch::Gap& y) { return x.before < y.before; });
    gaps_.clear();
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

LiveTelemetryHub::LiveTelemetryHub(std::size_t capacity, std::size_t max_subscribers, std::size_t subscriber_queue,
                                   std::size_t sample_capacity)
    : max_subscribers_(max_subscribers),
      subscriber_queue_(subscriber_queue != 0
                            ? subscriber_queue
                            : (capacity == 0 ? 1 : capacity) +
                                  (sample_capacity == 0 ? (capacity == 0 ? 1 : capacity) : sample_capacity)) {
    const std::size_t main_capacity = capacity == 0 ? 1 : capacity;
    rings_[static_cast<std::size_t>(TelemetryStream::main)].capacity = main_capacity;
    rings_[static_cast<std::size_t>(TelemetryStream::samples)].capacity =
        sample_capacity == 0 ? main_capacity : sample_capacity;
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
    TelemetryStream stream = TelemetryStream::main;
    if (const json::Value* producer = parsed.value().find("producer"); producer != nullptr && producer->is_object()) {
        const json::Value* named = producer->find("stream");
        if (named != nullptr && named->is_string() && named->as_string() == "samples") {
            stream = TelemetryStream::samples;
        }
    }
    write_stream_event(stream, static_cast<std::uint64_t>(seq->as_int()), json_line);
}

void LiveTelemetryHub::write_event(std::uint64_t sequence, std::string_view json_line) {
    write_stream_event(TelemetryStream::main, sequence, json_line);
}

void LiveTelemetryHub::write_stream_event(TelemetryStream stream, std::uint64_t sequence,
                                          std::string_view json_line) {
    // Built outside the lock; `order` is assigned under it, before the event
    // is shared with any other thread.
    auto built = std::make_shared<HubEvent>(HubEvent{stream, sequence, 0, std::string(json_line)});
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    built->order = next_order_++;
    const HubEventPtr event = std::move(built);
    Ring& ring = rings_[static_cast<std::size_t>(stream)];
    ring.events.push_back(event);
    while (ring.events.size() > ring.capacity) {
        ring.events.pop_front();
    }
    ring.next_sequence = std::max(ring.next_sequence, sequence + 1);
    for (const auto& sub : subscribers_) {
        sub->push(event);
    }
}

namespace {
// "<instance>-<sequence>" (main) or "<instance>-s<sequence>" (samples), split
// at the last '-'.
bool parse_event_id(std::string_view id, std::string& instance, TelemetryStream& stream, std::uint64_t& sequence) {
    const std::size_t dash = id.rfind('-');
    if (dash == std::string_view::npos || dash == 0 || dash + 1 >= id.size()) {
        return false;
    }
    std::string_view tail = id.substr(dash + 1);
    TelemetryStream parsed_stream = TelemetryStream::main;
    if (tail.front() == 's') {
        parsed_stream = TelemetryStream::samples;
        tail.remove_prefix(1);
    }
    if (tail.empty() || tail.size() > 19) {
        return false;
    }
    std::uint64_t n = 0;
    for (const char c : tail) {
        if (c < '0' || c > '9') {
            return false;
        }
        n = n * 10 + static_cast<std::uint64_t>(c - '0');
    }
    instance = std::string(id.substr(0, dash));
    stream = parsed_stream;
    sequence = n;
    return true;
}

constexpr std::size_t kMaxCursorMembers = 4;

std::string_view trim_blanks(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
        text.remove_suffix(1);
    }
    return text;
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

    // The resume cursor: at most one member per stream. Members of another
    // instance (producer restarted) and malformed members are ignored, which
    // replays that stream's whole window.
    std::array<std::optional<std::uint64_t>, kTelemetryStreamCount> cursor;
    if (resume.last_event_id && !resume.since_now) {
        std::string_view rest = *resume.last_event_id;
        for (std::size_t member = 0; member < kMaxCursorMembers; ++member) {
            const std::size_t comma = rest.find(',');
            const std::string_view piece = trim_blanks(rest.substr(0, comma));
            std::string instance;
            TelemetryStream stream = TelemetryStream::main;
            std::uint64_t seq = 0;
            if (parse_event_id(piece, instance, stream, seq) && instance == instance_id_) {
                auto& slot = cursor[static_cast<std::size_t>(stream)];
                slot = slot ? std::max(*slot, seq) : seq;
            }
            if (comma == std::string_view::npos) {
                break;
            }
            rest.remove_prefix(comma + 1);
        }
    }

    // Per stream: where the replay starts, and the gap to announce before its
    // first event when the cursor is older than the window.
    std::array<std::size_t, kTelemetryStreamCount> pos{};
    std::array<std::optional<ReplayGap>, kTelemetryStreamCount> gap;
    for (std::size_t k = 0; k < kTelemetryStreamCount; ++k) {
        const auto& events = rings_[k].events;
        if (resume.since_now) {
            pos[k] = events.size();
        } else if (cursor[k] && !events.empty()) {
            const std::uint64_t seq = *cursor[k];
            // Replay only what is newer than the id; never go backwards.
            pos[k] = static_cast<std::size_t>(
                std::partition_point(events.begin(), events.end(),
                                     [seq](const HubEventPtr& e) { return e->sequence <= seq; }) -
                events.begin());
            if (seq + 1 < events.front()->sequence) {
                gap[k] = ReplayGap{static_cast<TelemetryStream>(k), seq + 1, events.front()->sequence - 1};
            }
        }
    }

    // Merge the two tails in arrival order.
    while (true) {
        std::size_t pick = kTelemetryStreamCount;
        for (std::size_t k = 0; k < kTelemetryStreamCount; ++k) {
            if (pos[k] < rings_[k].events.size() &&
                (pick == kTelemetryStreamCount ||
                 rings_[k].events[pos[k]]->order < rings_[pick].events[pos[pick]]->order)) {
                pick = k;
            }
        }
        if (pick == kTelemetryStreamCount) {
            break;
        }
        const ReplayGap* announce = gap[pick] ? &*gap[pick] : nullptr;
        sub->push(rings_[pick].events[pos[pick]], announce);
        gap[pick].reset();  // announced before the first replayed event of the stream only
        ++pos[pick];
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
    const Ring& main_ring = rings_[static_cast<std::size_t>(TelemetryStream::main)];
    const Ring& sample_ring = rings_[static_cast<std::size_t>(TelemetryStream::samples)];
    s.retained = main_ring.events.size();
    s.capacity = main_ring.capacity;
    if (!main_ring.events.empty()) {
        s.oldest_sequence = main_ring.events.front()->sequence;
    }
    s.next_sequence = main_ring.next_sequence;
    s.sample_retained = sample_ring.events.size();
    s.sample_capacity = sample_ring.capacity;
    if (!sample_ring.events.empty()) {
        s.sample_oldest_sequence = sample_ring.events.front()->sequence;
    }
    s.sample_next_sequence = sample_ring.next_sequence;
    s.subscriber_dropped_events = departed_dropped_;
    for (const auto& sub : subscribers_) {
        s.subscriber_dropped_events += sub->dropped_total();
    }
    return s;
}

// ----------------------------------------------------------------- framing

std::string format_sse_batch(std::string_view instance_id, const Subscription::Batch& batch) {
    std::string frame;
    if (batch.dropped > 0) {
        frame += ": dropped " + std::to_string(batch.dropped) + "\n\n";
    }
    std::size_t next_gap = 0;
    for (std::size_t i = 0; i < batch.events.size(); ++i) {
        while (next_gap < batch.gaps.size() && batch.gaps[next_gap].before == i) {
            const ReplayGap& g = batch.gaps[next_gap].gap;
            const char* prefix = g.stream == TelemetryStream::samples ? "s" : "";
            frame += ": resume-gap ";
            frame += prefix;
            frame += std::to_string(g.from);
            frame += '-';
            frame += prefix;
            frame += std::to_string(g.to);
            frame += "\n\n";
            ++next_gap;
        }
        const HubEvent& ev = *batch.events[i];
        frame += "id: ";
        frame += instance_id;
        frame += ev.stream == TelemetryStream::samples ? "-s" : "-";
        frame += std::to_string(ev.sequence);
        frame += "\ndata: ";
        frame += ev.line;
        frame += "\n\n";
    }
    return frame;
}

std::string format_ndjson_batch(const Subscription::Batch& batch) {
    std::string frame;
    for (const auto& ev : batch.events) {
        frame += ev->line;
        frame += '\n';
    }
    return frame;
}

}  // namespace sonder::inference::server::detail
