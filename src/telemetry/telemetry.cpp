#include "sonder/inference/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>

#include "sonder/inference/device.hpp"
#include "sonder/inference/engine.hpp"

namespace sonder::inference {

const char* to_string(TelemetryLevel level) noexcept {
    switch (level) {
        case TelemetryLevel::off: return "off";
        case TelemetryLevel::metrics: return "metrics";
        case TelemetryLevel::standard: return "standard";
        case TelemetryLevel::deep: return "deep";
    }
    return "off";
}

std::optional<TelemetryLevel> parse_telemetry_level(std::string_view text) noexcept {
    if (text == "off") return TelemetryLevel::off;
    if (text == "metrics") return TelemetryLevel::metrics;
    if (text == "standard") return TelemetryLevel::standard;
    if (text == "deep") return TelemetryLevel::deep;
    return std::nullopt;
}

std::string utc_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - secs).count();
    const std::time_t t = std::chrono::system_clock::to_time_t(secs);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms));
    return buf;
}

std::uint64_t monotonic_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

std::string make_id(std::string_view prefix) {
    static std::mutex mutex;
    static std::mt19937_64 rng([] {
        std::random_device rd;
        return (static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^
               static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }());
    std::uint64_t r = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        r = rng();
    }
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(r));
    std::string id(prefix);
    id.push_back('-');
    id += buf;
    return id;
}

// ------------------------------------------------------------------ sinks

namespace {

class FileSink final : public TelemetrySink {
public:
    explicit FileSink(std::ofstream stream) : stream_(std::move(stream)) {}
    void write(std::string_view line) override {
        stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
        stream_.put('\n');
        if (!stream_) throw std::ios_base::failure("telemetry file write failed");
    }
    void flush() override {
        stream_.flush();
        if (!stream_) throw std::ios_base::failure("telemetry file flush failed");
    }

private:
    std::ofstream stream_;
};

class OstreamSink final : public TelemetrySink {
public:
    explicit OstreamSink(std::ostream& stream) : stream_(stream) {}
    void write(std::string_view line) override {
        stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
        stream_.put('\n');
        if (!stream_) throw std::ios_base::failure("telemetry stream write failed");
    }
    void flush() override {
        stream_.flush();
        if (!stream_) throw std::ios_base::failure("telemetry stream flush failed");
    }

private:
    std::ostream& stream_;
};

}  // namespace

std::unique_ptr<TelemetrySink> make_jsonl_file_sink(const std::string& path, bool append, Status* status) {
    std::ofstream stream(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!stream) {
        if (status) {
            *status = Status(ErrorCode::io_error, "cannot open telemetry file: " + path);
        }
        return nullptr;
    }
    if (status) {
        *status = Status::success();
    }
    return std::make_unique<FileSink>(std::move(stream));
}

std::unique_ptr<TelemetrySink> make_ostream_sink(std::ostream& stream) { return std::make_unique<OstreamSink>(stream); }

void MemoryTelemetrySink::write(std::string_view json_line) {
    std::lock_guard<std::mutex> lock(mutex_);
    lines_.emplace_back(json_line);
}

std::vector<std::string> MemoryTelemetrySink::lines() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lines_;
}

// -------------------------------------------------------------------- bus

// State shared between the bus and its writer thread. The thread co-owns it,
// so shutdown() can give up on a writer stuck inside a sink and detach it: the
// detached thread only touches this object and its own sink references, never
// the TelemetryBus.
struct TelemetryBus::Shared {
    Shared(TelemetryOptions opts, std::string id) : options(std::move(opts)), instance_id(std::move(id)) {}

    // Immutable after construction.
    const TelemetryOptions options;
    const std::string instance_id;

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable drained_cv;
    std::condition_variable exit_cv;  // shutdown(): the writer has left its loop
    std::vector<std::shared_ptr<TelemetrySink>> sinks;
    // (sequence, serialized envelope) in emission order.
    std::deque<std::pair<std::uint64_t, std::string>> queue;
    std::uint64_t next_sequence = 0;
    std::uint64_t written_sequence = 0;  // count of lines handed to sinks
    std::uint64_t enqueued_count = 0;
    bool stopping = false;
    bool stopped = false;
    bool exited = false;
    bool clean = true;  // false once shutdown() abandoned the writer
    // Set by shutdown() when the writer did not exit in time; the writer then
    // starts no further sink call. Atomic because the writer reads it with
    // the mutex released.
    std::atomic<bool> abandoned{false};
    std::atomic<std::uint64_t> emitted{0};
    std::atomic<std::uint64_t> dropped{0};
    std::atomic<std::uint64_t> failed_sinks{0};
    std::uint64_t reported_dropped = 0;
    std::chrono::steady_clock::time_point last_drop_report{};

    // Queues a telemetry.dropped report (bypasses the capacity limit). Lock held.
    void enqueue_drop_report_locked(bool final_report);
    static void writer_loop(const std::shared_ptr<Shared>& shared);
};

namespace {

json::Object build_envelope(const TelemetryOptions& options, const std::string& instance_id,
                            std::string_view event_type, const TelemetryContext& context, json::Object attributes,
                            std::uint64_t sequence, TelemetryLevel level) {
    json::Object env;
    env.set("schema", std::string(kObservatorySchema));
    env.set("event_id", instance_id + "-" + std::to_string(sequence));
    env.set("sequence", sequence);
    env.set("event_type", std::string(event_type));
    env.set("wall_time", utc_timestamp_now());
    env.set("mono_ns", monotonic_ns());
    env.set("session_id", context.session_id.empty() ? std::string("unscoped") : context.session_id);
    env.set("run_id", context.run_id);
    env.set("request_id", context.request_id);
    env.set("agent_id", context.agent_id);
    env.set("task_id", context.task_id);
    env.set("model_instance_id", context.model_instance_id);
    env.set("device_id", context.device_id);
    json::Object producer{{"name", options.producer_name},
                          {"version", version_string()},
                          {"node_id", options.node_id},
                          {"instance_id", instance_id}};
    if (!options.role.empty()) {
        producer.set("role", options.role);
    }
    const auto synthetic = context.synthetic_work ? std::optional<bool>(true) : options.synthetic;
    if (synthetic) {
        producer.set("synthetic", *synthetic);
    }
    env.set("producer", std::move(producer));
    env.set("sampling", json::Object{{"level", to_string(level)}, {"sampled", true}});
    env.set("attributes", std::move(attributes));
    return env;
}

}  // namespace

TelemetryBus::TelemetryBus(TelemetryOptions options) : options_(std::move(options)), instance_id_(make_id("tel")) {
    if (options_.node_id.empty()) {
        options_.node_id = host_name();
    }
    if (options_.queue_capacity == 0) {
        options_.queue_capacity = 1;
    }
    shared_ = std::make_shared<Shared>(options_, instance_id_);
    writer_ = std::thread([shared = shared_] { Shared::writer_loop(shared); });
}

TelemetryBus::~TelemetryBus() { shutdown(); }

void TelemetryBus::add_sink(std::shared_ptr<TelemetrySink> sink) {
    if (!sink) {
        return;
    }
    std::lock_guard<std::mutex> lock(shared_->mutex);
    shared_->sinks.push_back(std::move(sink));
}

bool TelemetryBus::enabled(TelemetryLevel level) const noexcept {
    if (options_.level == TelemetryLevel::off || level == TelemetryLevel::off) {
        return false;
    }
    if (static_cast<int>(level) > static_cast<int>(options_.level)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(shared_->mutex);
    return !shared_->sinks.empty() && !shared_->stopping;
}

std::uint64_t TelemetryBus::emitted_events() const noexcept { return shared_->emitted.load(); }
std::uint64_t TelemetryBus::dropped_events() const noexcept { return shared_->dropped.load(); }
std::uint64_t TelemetryBus::failed_sinks() const noexcept { return shared_->failed_sinks.load(); }

json::Object TelemetryBus::make_envelope(std::string_view event_type, const TelemetryContext& context,
                                         json::Object attributes, std::uint64_t sequence,
                                         TelemetryLevel level) const {
    return build_envelope(options_, instance_id_, event_type, context, std::move(attributes), sequence, level);
}

bool TelemetryBus::emit(std::string_view event_type, const TelemetryContext& context, json::Object attributes,
                        TelemetryLevel level) {
    if (!enabled(level)) {
        return false;
    }
    Shared& s = *shared_;
    std::unique_lock<std::mutex> lock(s.mutex);
    if (s.stopping || s.sinks.empty()) {
        return false;
    }
    if (s.queue.size() >= options_.queue_capacity) {
        s.dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint64_t seq = s.next_sequence++;
    s.queue.emplace_back(
        seq, json::Value(make_envelope(event_type, context, std::move(attributes), seq, level)).dump());
    ++s.enqueued_count;
    s.emitted.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    s.cv.notify_one();
    return true;
}

void TelemetryBus::Shared::writer_loop(const std::shared_ptr<Shared>& shared) {
    Shared& s = *shared;
    std::unique_lock<std::mutex> lock(s.mutex);
    while (true) {
        s.cv.wait(lock, [&s] { return s.stopping || s.abandoned.load() || !s.queue.empty(); });
        if (s.abandoned.load() || (s.queue.empty() && s.stopping)) {
            break;
        }
        std::deque<std::pair<std::uint64_t, std::string>> batch;
        batch.swap(s.queue);
        auto sinks = s.sinks;
        lock.unlock();
        auto retire_sink = [&](std::shared_ptr<TelemetrySink> failed) {
            lock.lock();
            std::erase(s.sinks, failed);
            s.failed_sinks.fetch_add(1, std::memory_order_relaxed);
            lock.unlock();
            // Retain shared ownership through registry removal, then discard
            // every snapshot alias outside the mutex. User destruction must
            // never run while the bus mutex is held or on the emitting thread.
            for (auto& alias : sinks) {
                if (alias == failed) alias.reset();
            }
        };
        for (const auto& [sequence, line] : batch) {
            for (const auto& sink : sinks) {
                if (!sink) continue;
                // shutdown() gave up on this writer: start no further sink call.
                if (s.abandoned.load()) break;
                try {
                    sink->write_event(sequence, line);
                } catch (...) {
                    retire_sink(sink);
                }
            }
        }
        for (const auto& sink : sinks) {
            if (!sink) continue;
            if (s.abandoned.load()) break;
            try {
                sink->flush();
            } catch (...) {
                retire_sink(sink);
            }
        }
        lock.lock();
        if (s.abandoned.load()) {
            break;
        }
        s.written_sequence += batch.size();
        // Live drop reporting: first drop immediately, then rate limited.
        if (!s.stopping && !s.sinks.empty() && s.dropped.load() > s.reported_dropped &&
            s.options.level != TelemetryLevel::off) {
            const auto now = std::chrono::steady_clock::now();
            if (s.reported_dropped == 0 || now - s.last_drop_report >= s.options.drop_report_interval) {
                s.enqueue_drop_report_locked(false);
            }
        }
        s.drained_cv.notify_all();
    }
    s.exited = true;
    s.exit_cv.notify_all();
    s.drained_cv.notify_all();
}

void TelemetryBus::Shared::enqueue_drop_report_locked(bool final_report) {
    const std::uint64_t dropped_now = dropped.load();
    TelemetryContext ctx;
    ctx.session_id = instance_id;
    const std::uint64_t seq = next_sequence++;
    queue.emplace_back(seq, json::Value(build_envelope(options, instance_id, "telemetry.dropped", ctx,
                                               json::Object{{"dropped_events", dropped_now},
                                                            {"emitted_events", emitted.load()},
                                                            {"queue_capacity", options.queue_capacity},
                                                            {"final", final_report}},
                                               seq, TelemetryLevel::metrics))
                         .dump());
    ++enqueued_count;
    reported_dropped = dropped_now;
    last_drop_report = std::chrono::steady_clock::now();
}

void TelemetryBus::flush() {
    Shared& s = *shared_;
    std::unique_lock<std::mutex> lock(s.mutex);
    const std::uint64_t target = s.enqueued_count;
    s.drained_cv.wait(lock, [&s, target] { return s.written_sequence >= target || s.stopped; });
}

bool TelemetryBus::shutdown() {
    Shared& s = *shared_;
    std::unique_lock<std::mutex> lock(s.mutex);
    if (s.stopped || s.stopping) {
        return s.clean;
    }
    if (s.dropped.load() > s.reported_dropped && !s.sinks.empty() && s.options.level != TelemetryLevel::off) {
        // Final accounting event for drops not reported live yet;
        // bypasses the capacity limit on purpose.
        s.enqueue_drop_report_locked(true);
    }
    s.stopping = true;
    s.cv.notify_all();
    const auto timeout = std::max(options_.shutdown_timeout, std::chrono::milliseconds(0));
    const bool exited = s.exit_cv.wait_for(lock, timeout, [&s] { return s.exited; });
    if (exited) {
        lock.unlock();
        if (writer_.joinable()) {
            writer_.join();
        }
        lock.lock();
    } else {
        // A sink did not return within the timeout. Abandon what is still
        // queued and leave the writer to finish its current sink call on its
        // own: it holds the shared state and the sinks it snapshotted, and
        // starts no further sink call once `abandoned` is set.
        s.abandoned.store(true);
        s.clean = false;
        std::deque<std::pair<std::uint64_t, std::string>> discarded;
        discarded.swap(s.queue);
        s.cv.notify_all();
        lock.unlock();
        discarded.clear();  // freed with the mutex released
        writer_.detach();
        lock.lock();
    }
    s.stopped = true;
    s.drained_cv.notify_all();
    return s.clean;
}

}  // namespace sonder::inference
