#include "sonder/inference/telemetry.hpp"

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
    }
    void flush() override { stream_.flush(); }

private:
    std::ofstream stream_;
};

class OstreamSink final : public TelemetrySink {
public:
    explicit OstreamSink(std::ostream& stream) : stream_(stream) {}
    void write(std::string_view line) override {
        stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
        stream_.put('\n');
    }
    void flush() override { stream_.flush(); }

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

TelemetryBus::TelemetryBus(TelemetryOptions options) : options_(std::move(options)), instance_id_(make_id("tel")) {
    if (options_.node_id.empty()) {
        options_.node_id = host_name();
    }
    if (options_.queue_capacity == 0) {
        options_.queue_capacity = 1;
    }
    writer_ = std::thread([this] { writer_loop(); });
}

TelemetryBus::~TelemetryBus() { shutdown(); }

void TelemetryBus::add_sink(std::shared_ptr<TelemetrySink> sink) {
    if (!sink) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_.push_back(std::move(sink));
}

bool TelemetryBus::enabled(TelemetryLevel level) const noexcept {
    if (options_.level == TelemetryLevel::off || level == TelemetryLevel::off) {
        return false;
    }
    if (static_cast<int>(level) > static_cast<int>(options_.level)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return !sinks_.empty() && !stopping_;
}

json::Object TelemetryBus::make_envelope(std::string_view event_type, const TelemetryContext& context,
                                         json::Object attributes, std::uint64_t sequence) const {
    json::Object env;
    env.set("schema", std::string(kObservatorySchema));
    env.set("event_id", instance_id_ + "-" + std::to_string(sequence));
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
    env.set("producer", json::Object{{"name", options_.producer_name},
                                     {"version", version_string()},
                                     {"node_id", options_.node_id}});
    env.set("sampling", json::Object{{"level", to_string(options_.level)}, {"sampled", true}});
    env.set("attributes", std::move(attributes));
    return env;
}

bool TelemetryBus::emit(std::string_view event_type, const TelemetryContext& context, json::Object attributes,
                        TelemetryLevel level) {
    if (!enabled(level)) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) {
        return false;
    }
    if (queue_.size() >= options_.queue_capacity) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const std::uint64_t seq = next_sequence_++;
    queue_.push_back(json::Value(make_envelope(event_type, context, std::move(attributes), seq)).dump());
    ++enqueued_count_;
    emitted_.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    cv_.notify_one();
    return true;
}

void TelemetryBus::writer_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) {
            break;
        }
        std::deque<std::string> batch;
        batch.swap(queue_);
        auto sinks = sinks_;
        lock.unlock();
        for (const auto& line : batch) {
            for (const auto& sink : sinks) {
                sink->write(line);
            }
        }
        for (const auto& sink : sinks) {
            sink->flush();
        }
        lock.lock();
        written_sequence_ += batch.size();
        drained_cv_.notify_all();
    }
    drained_cv_.notify_all();
}

void TelemetryBus::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::uint64_t target = enqueued_count_;
    drained_cv_.wait(lock, [this, target] { return written_sequence_ >= target || stopped_; });
}

void TelemetryBus::shutdown() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_ || stopping_) {
            return;
        }
        const std::uint64_t dropped = dropped_.load();
        if (dropped > 0 && !sinks_.empty() && options_.level != TelemetryLevel::off) {
            // Final accounting event; bypasses the capacity limit on purpose.
            TelemetryContext ctx;
            ctx.session_id = instance_id_;
            const std::uint64_t seq = next_sequence_++;
            queue_.push_back(
                json::Value(make_envelope("telemetry.dropped", ctx,
                                          json::Object{{"dropped_events", dropped},
                                                       {"emitted_events", emitted_.load()},
                                                       {"queue_capacity", options_.queue_capacity}},
                                          seq))
                    .dump());
            ++enqueued_count_;
        }
        stopping_ = true;
    }
    cv_.notify_all();
    if (writer_.joinable()) {
        writer_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    drained_cv_.notify_all();
}

}  // namespace sonder::inference
