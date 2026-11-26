#include "sonder/inference/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <limits>
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

const char* to_string(CapabilityState state) noexcept {
    switch (state) {
        case CapabilityState::unknown: return "unknown";
        case CapabilityState::no: return "no";
        case CapabilityState::yes: return "yes";
    }
    return "unknown";
}

std::optional<CapabilityState> parse_capability_state(std::string_view text) noexcept {
    if (text == "unknown") return CapabilityState::unknown;
    if (text == "no" || text == "false") return CapabilityState::no;
    if (text == "yes" || text == "true") return CapabilityState::yes;
    return std::nullopt;
}

std::string sanitize_bounded_evidence_string(std::string_view text, std::size_t max_bytes) {
    if (text.empty() || max_bytes == 0) {
        return {};
    }
    const std::size_t limit = std::min(text.size(), max_bytes);
    std::string out;
    out.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto ch = static_cast<unsigned char>(text[i]);
        if (ch < 0x20u || ch == 0x7Fu) {
            out.push_back(' ');
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

namespace {

std::optional<double> sanitize_nonneg_ms(std::optional<double> value) noexcept {
    if (!value.has_value() || !std::isfinite(*value) || *value < 0.0) {
        return std::nullopt;
    }
    return value;
}

std::uint64_t saturating_add_u64(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

json::Value optional_string_or_null(std::string_view text, std::size_t max_bytes = kMaxEvidenceStringBytes) {
    std::string clean = sanitize_bounded_evidence_string(text, max_bytes);
    if (clean.empty()) {
        return json::Value(nullptr);
    }
    return json::Value(std::move(clean));
}

json::Value method_or_unknown(std::string_view text) {
    std::string clean = sanitize_bounded_evidence_string(text, kMaxEvidenceStringBytes);
    if (clean.empty()) {
        return json::Value(std::string(kSpecReasonUnknown));
    }
    return json::Value(std::move(clean));
}

}  // namespace

std::optional<std::uint64_t> CacheTierCounters::physical_reused_tokens() const noexcept {
    if (!device_prefix_reused_tokens.has_value() && !host_l2_restored_tokens.has_value()) {
        return std::nullopt;
    }
    return saturating_add_u64(device_prefix_reused_tokens.value_or(0u), host_l2_restored_tokens.value_or(0u));
}

void StageTimings::add_ms(std::optional<double>& slot, double delta_ms) noexcept {
    if (!std::isfinite(delta_ms) || delta_ms < 0.0) {
        return;
    }
    if (slot.has_value()) {
        *slot += delta_ms;
    } else {
        slot = delta_ms;
    }
}

void StageTimings::note_overlap(std::string_view label) {
    stages_may_overlap = true;
    std::string clean = sanitize_bounded_evidence_string(label, 96);
    if (clean.empty() || overlap_notes.size() >= kMaxStageOverlapNotes) {
        return;
    }
    if (std::find(overlap_notes.begin(), overlap_notes.end(), clean) == overlap_notes.end()) {
        overlap_notes.push_back(std::move(clean));
    }
}

bool StageTimings::has_any_measurement() const noexcept {
    return queue_wait_ms.has_value() || cache_lookup_ms.has_value() || l2_restore_ms.has_value() ||
           prefill_ms.has_value() || draft_ms.has_value() || verify_ms.has_value() || rollback_ms.has_value() ||
           sampling_ms.has_value() || callback_gate_wait_ms.has_value() || total_elapsed_ms.has_value();
}

EffectiveExecutionReport bound_effective_execution_report(EffectiveExecutionReport report) {
    report.configured_spec_method = sanitize_bounded_evidence_string(report.configured_spec_method);
    report.effective_spec_method = sanitize_bounded_evidence_string(report.effective_spec_method);
    report.fallback_reason = sanitize_bounded_evidence_string(report.fallback_reason);
    report.target_identity = sanitize_bounded_evidence_string(report.target_identity);
    report.draft_identity = sanitize_bounded_evidence_string(report.draft_identity);
    report.backend_version = sanitize_bounded_evidence_string(report.backend_version);
    if (report.proposed_draft_tokens.has_value() && report.accepted_draft_tokens.has_value() &&
        *report.accepted_draft_tokens > *report.proposed_draft_tokens) {
        report.accepted_draft_tokens = report.proposed_draft_tokens;
    }
    report.cache.host_l2_restore_ms = sanitize_nonneg_ms(report.cache.host_l2_restore_ms);
    if (report.cache.device_prefix_reused_tokens.has_value() &&
        report.cache.device_cross_slot_reused_tokens.has_value() &&
        *report.cache.device_cross_slot_reused_tokens > *report.cache.device_prefix_reused_tokens) {
        report.cache.device_cross_slot_reused_tokens = report.cache.device_prefix_reused_tokens;
    }
    return report;
}

json::Object to_json(const CapabilityStatus& status) {
    json::Object out;
    out.set("adapter_supports", to_string(status.adapter_supports));
    out.set("build_supports", to_string(status.build_supports));
    out.set("configured", to_string(status.configured));
    out.set("observed_active", to_string(status.observed_active));
    out.set("qualified_beneficial", to_string(status.qualified_beneficial));
    std::string note = sanitize_bounded_evidence_string(status.note);
    if (!note.empty()) {
        out.set("note", std::move(note));
    }
    return out;
}

json::Object to_json_capability_statuses(const std::vector<CapabilityStatusEntry>& entries) {
    json::Object out;
    const std::size_t limit = std::min(entries.size(), kMaxCapabilityStatusEntries);
    for (std::size_t i = 0; i < limit; ++i) {
        std::string key = sanitize_bounded_evidence_string(entries[i].first, 96);
        if (key.empty()) continue;
        out.set(std::move(key), to_json(entries[i].second));
    }
    return out;
}

json::Object to_json(const CacheTierCounters& cache) {
    const auto restore_ms = sanitize_nonneg_ms(cache.host_l2_restore_ms);
    const auto physical = cache.physical_reused_tokens();
    json::Object out;
    out.set("logical_cache_hit_tokens", json::Value(cache.logical_cache_hit_tokens));
    out.set("device_prefix_reused_tokens", json::Value(cache.device_prefix_reused_tokens));
    if (cache.device_cross_slot_reused_tokens.has_value()) {
        out.set("device_cross_slot_reused_tokens", json::Value(cache.device_cross_slot_reused_tokens));
    }
    out.set("host_l2_restored_tokens", json::Value(cache.host_l2_restored_tokens));
    out.set("host_l2_restore_ms", json::Value(restore_ms));
    out.set("physical_reused_tokens", json::Value(physical));
    return out;
}

json::Object to_json(const EffectiveExecutionReport& raw_report) {
    const EffectiveExecutionReport r = bound_effective_execution_report(raw_report);
    json::Object out;
    out.set("configured_spec_method", method_or_unknown(r.configured_spec_method));
    out.set("effective_spec_method", method_or_unknown(r.effective_spec_method));
    out.set("configured_draft_limit", json::Value(r.configured_draft_limit));
    out.set("effective_draft_limit", json::Value(r.effective_draft_limit));
    out.set("fallback_reason", optional_string_or_null(r.fallback_reason));
    out.set("target_identity", optional_string_or_null(r.target_identity));
    out.set("draft_identity", optional_string_or_null(r.draft_identity));
    out.set("backend_version", optional_string_or_null(r.backend_version));
    out.set("committed_tokens", json::Value(r.committed_tokens));
    out.set("proposed_draft_tokens", json::Value(r.proposed_draft_tokens));
    out.set("accepted_draft_tokens", json::Value(r.accepted_draft_tokens));
    out.set("speculation_observed_active", to_string(r.speculation_observed_active));
    out.set("transitioned_during_request", r.transitioned_during_request);
    out.set("cache", to_json(r.cache));
    return out;
}

json::Object to_json(const StageTimings& timings) {
    json::Object out;
    out.set("enabled", timings.enabled);
    out.set("queue_wait_ms", json::Value(sanitize_nonneg_ms(timings.queue_wait_ms)));
    out.set("cache_lookup_ms", json::Value(sanitize_nonneg_ms(timings.cache_lookup_ms)));
    out.set("l2_restore_ms", json::Value(sanitize_nonneg_ms(timings.l2_restore_ms)));
    out.set("prefill_ms", json::Value(sanitize_nonneg_ms(timings.prefill_ms)));
    out.set("draft_ms", json::Value(sanitize_nonneg_ms(timings.draft_ms)));
    out.set("verify_ms", json::Value(sanitize_nonneg_ms(timings.verify_ms)));
    out.set("rollback_ms", json::Value(sanitize_nonneg_ms(timings.rollback_ms)));
    out.set("sampling_ms", json::Value(sanitize_nonneg_ms(timings.sampling_ms)));
    out.set("callback_gate_wait_ms", json::Value(sanitize_nonneg_ms(timings.callback_gate_wait_ms)));
    out.set("total_elapsed_ms", json::Value(sanitize_nonneg_ms(timings.total_elapsed_ms)));
    out.set("stages_may_overlap", timings.stages_may_overlap);
    json::Array notes;
    const std::size_t limit = std::min(timings.overlap_notes.size(), kMaxStageOverlapNotes);
    notes.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        std::string clean = sanitize_bounded_evidence_string(timings.overlap_notes[i], 96);
        if (!clean.empty()) {
            notes.emplace_back(std::move(clean));
        }
    }
    out.set("overlap_notes", std::move(notes));
    return out;
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

bool TelemetryBus::emit_effective_execution(const TelemetryContext& context, const EffectiveExecutionReport& report,
                                            TelemetryLevel level) {
    if (!enabled(level)) {
        return false;
    }
    return emit("inference.effective_execution", context, to_json(report), level);
}

bool TelemetryBus::emit_stage_timings(const TelemetryContext& context, const StageTimings& timings,
                                      std::string_view backend, std::string_view backend_version,
                                      TelemetryLevel level) {
    if (!timings.enabled || !timings.has_any_measurement() || !enabled(level)) {
        return false;
    }
    json::Object attrs = to_json(timings);
    std::string clean_backend = sanitize_bounded_evidence_string(backend, 96);
    if (!clean_backend.empty()) {
        attrs.set("backend", std::move(clean_backend));
    }
    std::string clean_version = sanitize_bounded_evidence_string(backend_version, kMaxEvidenceStringBytes);
    if (!clean_version.empty()) {
        attrs.set("backend_version", std::move(clean_version));
    }
    return emit("inference.stage.timings", context, std::move(attrs), level);
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
