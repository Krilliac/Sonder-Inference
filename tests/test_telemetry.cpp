#include <doctest/doctest.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {

bool is_rfc3339_utc(const std::string& s) {
    // YYYY-MM-DDTHH:MM:SS.mmmZ
    if (s.size() != 24 || s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':' ||
        s[19] != '.' || s[23] != 'Z') {
        return false;
    }
    for (std::size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u, 11u, 12u, 14u, 15u, 17u, 18u, 20u, 21u, 22u}) {
        if (s[i] < '0' || s[i] > '9') return false;
    }
    return true;
}

bool nullable_string(const json::Value* v) { return v != nullptr && (v->is_null() || v->is_string()); }

// Checks one event against the Sonder Observatory envelope v1
// (protocol/observatory-events.schema.json in Krilliac/Sonder-Observatory).
void check_envelope(const json::Value& e) {
    REQUIRE(e.is_object());
    CHECK(e.find("schema")->as_string() == "sonder.observatory.event/1");
    REQUIRE(e.find("event_id"));
    CHECK(e.find("event_id")->is_string());
    CHECK_FALSE(e.find("event_id")->as_string().empty());
    REQUIRE(e.find("sequence"));
    CHECK(e.find("sequence")->is_integer());
    CHECK(e.find("sequence")->as_int() >= 0);
    CHECK_FALSE(e.find("event_type")->as_string().empty());
    CHECK(is_rfc3339_utc(e.find("wall_time")->as_string()));
    REQUIRE(e.find("mono_ns"));
    CHECK(e.find("mono_ns")->is_integer());
    CHECK(e.find("mono_ns")->as_int() >= 0);
    CHECK_FALSE(e.find("session_id")->as_string().empty());
    for (const char* key : {"run_id", "request_id", "agent_id", "task_id", "model_instance_id", "device_id"}) {
        CHECK_MESSAGE(nullable_string(e.find(key)), key);
    }
    const json::Value* producer = e.find("producer");
    REQUIRE(producer);
    REQUIRE(producer->is_object());
    for (const char* key : {"name", "version", "node_id"}) {
        REQUIRE(producer->find(key));
        CHECK(producer->find(key)->is_string());
    }
    const json::Value* sampling = e.find("sampling");
    REQUIRE(sampling);
    const std::set<std::string> levels{"off", "metrics", "standard", "deep"};
    CHECK(levels.count(sampling->find("level")->as_string()) == 1);
    CHECK(sampling->find("sampled")->is_bool());
    REQUIRE(e.find("attributes"));
    CHECK(e.find("attributes")->is_object());
}

// Blocks inside write() until released, so the bus queue can fill up.
class GateSink final : public TelemetrySink {
public:
    void write(std::string_view line) override {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return open_; });
        lines_.emplace_back(line);
    }
    void open() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_ = true;
        }
        cv_.notify_all();
    }
    std::vector<std::string> lines() {
        std::lock_guard<std::mutex> lock(mutex_);
        return lines_;
    }
    bool wait_until_entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = false;
    bool entered_ = false;
    std::vector<std::string> lines_;
};

}  // namespace

TEST_SUITE("telemetry") {
TEST_CASE("every emitted event matches the Observatory envelope") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(6));
    REQUIRE(session);
    REQUIRE(session->generate("envelope check").ok());
    session->close();
    const auto events = h.events();
    REQUIRE(events.size() > 10);
    std::int64_t expected_seq = 0;
    std::set<std::string> ids;
    for (const auto& e : events) {
        check_envelope(e);
        CHECK(e.find("sequence")->as_int() == expected_seq++);
        ids.insert(e.find("event_id")->as_string());
    }
    CHECK(ids.size() == events.size());
    std::set<std::string> types;
    for (const auto& e : events) types.insert(e.find("event_type")->as_string());
    for (const char* t : {"engine.started", "device.memory.sample", "model.load.started", "model.load.completed",
                          "session.created", "request.queued", "request.started", "inference.decode.started",
                          "inference.token.generated", "inference.decode.completed", "request.completed",
                          "session.closed"}) {
        CHECK_MESSAGE(types.count(t) == 1, t);
    }
}

TEST_CASE("metrics level omits per-token events") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    auto session = h.session(SamplingConfig::greedy(6));
    REQUIRE(session->generate("metrics only").ok());
    CHECK(h.events_of("inference.token.generated").empty());
    CHECK(h.events_of("request.completed").size() == 1);
    CHECK(h.events()[0].find("sampling")->find("level")->as_string() == "metrics");
}

TEST_CASE("off level emits nothing") {
    sonder_test::Harness h({}, TelemetryLevel::off);
    auto session = h.session();
    REQUIRE(session->generate("silent").ok());
    CHECK(h.events().empty());
}

TEST_CASE("token text is captured only when enabled") {
    {
        sonder_test::Harness h;
        auto session = h.session(SamplingConfig::greedy(3));
        REQUIRE(session->generate("no text").ok());
        for (const auto& e : h.events_of("inference.token.generated")) {
            CHECK(e.find("attributes")->find("text") == nullptr);
            CHECK(e.find("attributes")->find("bytes")->as_int() > 0);
        }
    }
    {
        sonder_test::Harness h({}, TelemetryLevel::standard, true);
        auto session = h.session(SamplingConfig::greedy(3));
        auto r = session->generate("with text");
        REQUIRE(r.ok());
        std::string joined;
        for (const auto& e : h.events_of("inference.token.generated")) {
            joined += e.find("attributes")->find("text")->as_string();
        }
        CHECK(joined == r.value().text);
    }
}

TEST_CASE("bounded queue drops under pressure and reports it") {
    auto gate = std::make_shared<GateSink>();
    TelemetryOptions opts;
    opts.queue_capacity = 4;
    auto bus = std::make_unique<TelemetryBus>(opts);
    bus->add_sink(gate);
    TelemetryContext ctx;
    ctx.session_id = "sess-test";
    int accepted = 0;
    for (int i = 0; i < 100; ++i) {
        if (bus->emit("test.event", ctx, json::Object{{"i", i}})) ++accepted;
    }
    CHECK(bus->dropped_events() > 0);
    CHECK(static_cast<std::uint64_t>(accepted) + bus->dropped_events() == 100);
    gate->open();
    bus->shutdown();
    const auto lines = gate->lines();
    REQUIRE_FALSE(lines.empty());
    auto last = json::parse(lines.back());
    REQUIRE(last.ok());
    check_envelope(last.value());
    CHECK(last.value().find("event_type")->as_string() == "telemetry.dropped");
    CHECK(static_cast<std::uint64_t>(last.value().find("attributes")->find("dropped_events")->as_int()) ==
          bus->dropped_events());
    CHECK(lines.size() == static_cast<std::size_t>(accepted) + 1);
}

TEST_CASE("jsonl file sink writes one event per line") {
    const std::string path = "sonder-telemetry-test.jsonl";
    {
        Status st;
        auto sink = make_jsonl_file_sink(path, false, &st);
        REQUIRE(st.ok());
        EngineOptions eo;
        eo.telemetry_sinks.push_back(std::shared_ptr<TelemetrySink>(std::move(sink)));
        Engine engine(std::move(eo));
    }
    std::ifstream in(path);
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        auto v = json::parse(line);
        REQUIRE(v.ok());
        check_envelope(v.value());
        ++n;
    }
    CHECK(n >= 2);  // engine.started, device sample(s), engine.stopped
    in.close();
    std::remove(path.c_str());
}

TEST_CASE("level parsing") {
    CHECK(parse_telemetry_level("deep") == TelemetryLevel::deep);
    CHECK_FALSE(parse_telemetry_level("verbose").has_value());
    CHECK(std::string(to_string(TelemetryLevel::metrics)) == "metrics");
}
}

TEST_CASE("Observatory requests: stream identity, run grouping, capture policy, per-event level") {
    sonder_test::Harness h({}, TelemetryLevel::standard, true);
    auto session = h.session(SamplingConfig::greedy(4));
    REQUIRE(session->generate("observatory").ok());
    session->close();
    const std::string engine_id = h.engine->engine_id();
    const std::string instance = h.engine->telemetry().instance_id();
    for (const auto& e : h.events()) {
        const auto* producer = e.find("producer");
        REQUIRE(producer->find("instance_id"));
        CHECK(producer->find("instance_id")->as_string() == instance);
        CHECK(e.find("event_id")->as_string() == instance + "-" + std::to_string(e.find("sequence")->as_int()));
        const std::string type = e.find("event_type")->as_string();
        // Every engine, session and request event carries the engine id as run_id.
        REQUIRE(e.find("run_id")->is_string());
        CHECK(e.find("run_id")->as_string() == engine_id);
        const std::string level = e.find("sampling")->find("level")->as_string();
        if (type == "inference.token.generated" || type == "scheduler.batch.formed") {
            CHECK(level == "standard");
        } else if (type.rfind("request.", 0) == 0 || type.rfind("session.", 0) == 0 ||
                   type.rfind("engine.", 0) == 0 || type.rfind("model.", 0) == 0) {
            CHECK(level == "metrics");
        }
    }
    auto started = h.events_of("engine.started");
    REQUIRE(started.size() == 1);
    CHECK(started[0].find("attributes")->find("text_capture")->as_string() == "on");
    auto created = h.events_of("session.created");
    REQUIRE(created.size() == 1);
    CHECK(created[0].find("attributes")->find("text_capture")->as_string() == "on");
    auto loaded = h.events_of("model.load.completed");
    REQUIRE(loaded.size() == 1);
    CHECK(loaded[0].find("attributes")->find("resident")->as_bool());
    auto prefill = h.events_of("inference.prefill.completed");
    REQUIRE(prefill.size() == 1);
    CHECK(prefill[0].find("attributes")->find("prompt_tokens")->as_int() >= 1);
    for (const auto& e : h.events_of("inference.token.generated")) {
        CHECK(e.find("attributes")->find("unit")->as_string() == "chunk");
    }
}

TEST_CASE("an explicit run_id is kept") {
    sonder_test::Harness h;
    SessionOptions so;
    so.run_id = "run-explicit";
    auto s = h.engine->create_session(h.model, so);
    REQUIRE(s.ok());
    REQUIRE(s.value()->generate("hi").ok());
    for (const auto& e : h.events_of("request.completed")) {
        CHECK(e.find("run_id")->as_string() == "run-explicit");
    }
}

TEST_CASE("devices are sampled periodically") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds(10);
    std::size_t samples = 0;
    {
        Engine engine(std::move(eo));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            engine.telemetry().flush();
            samples = 0;
            for (const auto& l : sink->lines()) {
                if (l.find("\"device.memory.sample\"") != std::string::npos) ++samples;
            }
            if (samples >= 2 * engine.devices().size() && samples > 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    CHECK(samples >= 2);
}

TEST_CASE("drops are reported while running, not only at shutdown") {
    auto gate = std::make_shared<GateSink>();
    TelemetryOptions opts;
    opts.queue_capacity = 2;
    opts.drop_report_interval = std::chrono::milliseconds(0);
    TelemetryBus bus(opts);
    bus.add_sink(gate);
    TelemetryContext ctx;
    ctx.session_id = "sess-test";
    for (int i = 0; i < 20; ++i) (void)bus.emit("test.event", ctx, json::Object{{"i", i}});
    REQUIRE(bus.dropped_events() > 0);
    gate->open();
    bus.flush();  // writer drains, sees the drops and queues a live report
    bus.flush();
    bool live = false;
    for (const auto& l : gate->lines()) {
        auto v = json::parse(l);
        REQUIRE(v.ok());
        if (v.value().find("event_type")->as_string() == "telemetry.dropped") {
            CHECK_FALSE(v.value().find("attributes")->find("final")->as_bool());
            live = true;
        }
    }
    CHECK(live);
    bus.shutdown();
}

namespace {
TelemetryContext ctx_s() {
    TelemetryContext c;
    c.session_id = "s";
    return c;
}
}  // namespace

TEST_CASE("producer role and synthetic flag (additive envelope fields)") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    TelemetryOptions opts;
    TelemetryBus plain(opts);
    plain.add_sink(sink);
    const json::Object env = plain.make_envelope("x.y", ctx_s(), json::Object{}, 0);
    CHECK(env.find("producer")->find("role")->as_string() == "inference");
    // Unknown by default: the field is absent, never a false claim.
    CHECK(env.find("producer")->find("synthetic") == nullptr);

    opts.synthetic = false;
    TelemetryBus real(opts);
    CHECK_FALSE(real.make_envelope("x.y", ctx_s(), json::Object{}, 0).find("producer")->find("synthetic")->as_bool());

    opts.synthetic = true;
    opts.role = "";
    TelemetryBus synthetic(opts);
    const json::Object env2 = synthetic.make_envelope("x.y", ctx_s(), json::Object{}, 0);
    CHECK(env2.find("producer")->find("role") == nullptr);
    CHECK(env2.find("producer")->find("synthetic")->as_bool());
}

namespace {
class SequenceSink final : public TelemetrySink {
public:
    void write(std::string_view) override { ++plain_writes; }
    void write_event(std::uint64_t sequence, std::string_view line) override {
        std::lock_guard<std::mutex> lock(mutex);
        auto v = json::parse(line);
        if (v.ok()) {
            pairs.emplace_back(sequence, static_cast<std::uint64_t>(v.value().find("sequence")->as_int()));
        }
    }
    std::mutex mutex;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
    int plain_writes = 0;
};
}  // namespace

TEST_CASE("sinks receive each envelope's sequence through write_event") {
    auto sink = std::make_shared<SequenceSink>();
    {
        TelemetryBus bus;
        bus.add_sink(sink);
        for (int i = 0; i < 20; ++i) {
            bus.emit("test.event", ctx_s(), json::Object{{"i", i}});
        }
        bus.flush();
    }
    std::lock_guard<std::mutex> lock(sink->mutex);
    REQUIRE(sink->pairs.size() == 20);
    for (std::size_t i = 0; i < sink->pairs.size(); ++i) {
        CHECK(sink->pairs[i].first == i);
        CHECK(sink->pairs[i].second == i);
    }
    CHECK(sink->plain_writes == 0);
    // The default write_event forwards to write().
    MemoryTelemetrySink memory;
    memory.write_event(7, "{\"a\":1}");
    REQUIRE(memory.lines().size() == 1);
    CHECK(memory.lines()[0] == "{\"a\":1}");
}

TEST_CASE("an engine running the mock backend never labels its events synthetic:false") {
    // Hosts that do not set TelemetryOptions::synthetic (CLI, C ABI, bench)
    // must not claim mock output is real.
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(4));
    REQUIRE(session);
    REQUIRE(session->generate("synthetic label").ok());
    session->close();
    const auto events = h.events();
    REQUIRE_FALSE(events.empty());
    for (const auto& e : events) {
        const json::Value* synthetic = e.find("producer")->find("synthetic");
        CHECK_MESSAGE((synthetic == nullptr || synthetic->as_bool()), e.find("event_type")->as_string());
    }
}

TEST_CASE("engine.started carries the server listener when one is set") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.server = EngineServerInfo{"127.0.0.1", 18437, 1};
    {
        Engine engine(std::move(eo));
        engine.telemetry().flush();
    }
    bool found = false;
    for (const auto& line : sink->lines()) {
        auto v = json::parse(line);
        REQUIRE(v.ok());
        if (v.value().find("event_type")->as_string() != "engine.started") continue;
        found = true;
        const json::Value* server = v.value().find("attributes")->find("server");
        REQUIRE(server != nullptr);
        CHECK(server->find("host")->as_string() == "127.0.0.1");
        CHECK(server->find("port")->as_int() == 18437);
        CHECK(server->find("api_version")->as_int() == 1);
    }
    CHECK(found);
}

namespace {
class ThrowingWriteSink final : public TelemetrySink {
public:
    void write(std::string_view) override {
        ++calls;
        throw std::ios_base::failure("synthetic telemetry write failure");
    }
    std::atomic<int> calls{0};
};

class DeferredFailureSink final : public TelemetrySink {
public:
    void write(std::string_view line) override {
        ++calls;
        gate.write(line);
        throw std::ios_base::failure("synthetic delayed write failure");
    }
    GateSink gate;
    std::atomic<int> calls{0};
};

enum class SinkFailure { event_override, flush, nonstandard };

class FaultSink final : public TelemetrySink {
public:
    explicit FaultSink(SinkFailure failure) : failure_(failure) {}
    void write(std::string_view) override { ++plain_calls; }
    void write_event(std::uint64_t sequence, std::string_view line) override {
        ++event_calls;
        if (failure_ == SinkFailure::event_override) throw std::ios_base::failure("synthetic event failure");
        if (failure_ == SinkFailure::nonstandard) throw 7;
        TelemetrySink::write_event(sequence, line);
    }
    void flush() override {
        ++flush_calls;
        if (failure_ == SinkFailure::flush) throw std::ios_base::failure("synthetic flush failure");
    }
    std::atomic<int> event_calls{0};
    std::atomic<int> plain_calls{0};
    std::atomic<int> flush_calls{0};

private:
    SinkFailure failure_;
};

struct RetirementObservation {
    std::atomic<bool> destroyed{false};
    std::atomic<bool> healthy_bus_enabled{false};
    std::atomic<bool> on_emitting_thread{true};
};

class DestructorQuerySink final : public TelemetrySink {
public:
    DestructorQuerySink(TelemetryBus& bus, RetirementObservation& observation)
        : bus_(bus), observation_(observation), emitting_thread_(std::this_thread::get_id()) {}
    ~DestructorQuerySink() override {
        // Querying the bus would deadlock if retirement held its mutex.
        observation_.healthy_bus_enabled = bus_.enabled(TelemetryLevel::metrics);
        observation_.on_emitting_thread = std::this_thread::get_id() == emitting_thread_;
        observation_.destroyed = true;
    }
    void write(std::string_view) override { throw std::ios_base::failure("synthetic destruction control"); }

private:
    TelemetryBus& bus_;
    RetirementObservation& observation_;
    std::thread::id emitting_thread_;
};

class FailingStreamBuffer final : public std::stringbuf {
public:
    explicit FailingStreamBuffer(bool fail_flush) : fail_flush_(fail_flush) {}
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        ++write_calls;
        return fail_flush_ ? std::stringbuf::xsputn(data, size) : 0;
    }
    int sync() override {
        ++flush_calls;
        return fail_flush_ ? -1 : std::stringbuf::sync();
    }
    int write_calls = 0;
    int flush_calls = 0;

private:
    bool fail_flush_;
};

TelemetryOptions synthetic_sink_options() {
    TelemetryOptions options;
    options.synthetic = true;
    return options;
}

void check_sink_sequences(const std::vector<std::string>& lines, const std::string& instance) {
    for (std::size_t i = 0; i < lines.size(); ++i) {
        auto value = json::parse(lines[i]);
        REQUIRE(value.ok());
        CHECK(value.value().find("sequence")->as_int() == static_cast<std::int64_t>(i));
        CHECK(value.value().find("event_id")->as_string() == instance + "-" + std::to_string(i));
        CHECK(value.value().find("producer")->find("instance_id")->as_string() == instance);
        CHECK(value.value().find("producer")->find("synthetic")->as_bool());
    }
}
}  // namespace

TEST_CASE("telemetry write failure retires the sink and preserves healthy delivery") {
    auto failed = std::make_shared<ThrowingWriteSink>();
    auto healthy = std::make_shared<MemoryTelemetrySink>();
    auto sibling = std::make_shared<MemoryTelemetrySink>();
    TelemetryBus bus(synthetic_sink_options());
    bus.add_sink(failed);
    bus.add_sink(healthy);
    bus.add_sink(sibling);
    TelemetryContext ctx;
    ctx.session_id = "synthetic-sink-failure";
    for (int i = 0; i < 32; ++i) {
        REQUIRE(bus.emit("test.event", ctx, json::Object{{"i", i}}));
    }
    bus.flush();
    CHECK(failed->calls == 1);
    CHECK(bus.failed_sinks() == 1);
    CHECK(bus.dropped_events() == 0);
    CHECK(bus.emitted_events() == 32);
    CHECK(bus.enabled(TelemetryLevel::metrics));
    REQUIRE(healthy->lines().size() == 32);
    CHECK(healthy->lines() == sibling->lines());
    check_sink_sequences(healthy->lines(), bus.instance_id());
}

TEST_CASE("telemetry override and flush exceptions stop retries while siblings continue") {
    for (const auto failure : {SinkFailure::event_override, SinkFailure::flush, SinkFailure::nonstandard}) {
        CAPTURE(static_cast<int>(failure));
        auto failed = std::make_shared<FaultSink>(failure);
        auto healthy = std::make_shared<MemoryTelemetrySink>();
        TelemetryBus bus(synthetic_sink_options());
        bus.add_sink(failed);
        bus.add_sink(healthy);
        TelemetryContext ctx;
        ctx.session_id = "synthetic-sink-failure";
        for (int i = 0; i < 8; ++i) CHECK(bus.emit("test.event", ctx, json::Object{{"i", i}}));
        bus.flush();
        const auto event_calls = failed->event_calls.load();
        const auto flush_calls = failed->flush_calls.load();
        CHECK(bus.failed_sinks() == 1);
        if (failure == SinkFailure::flush) {
            CHECK(flush_calls == 1);
            CHECK(event_calls >= 1);
        } else {
            CHECK(event_calls == 1);
            CHECK(flush_calls == 0);
            CHECK(failed->plain_calls == 0);
        }
        for (int i = 8; i < 16; ++i) CHECK(bus.emit("test.event", ctx, json::Object{{"i", i}}));
        bus.flush();
        CHECK(failed->event_calls == event_calls);
        CHECK(failed->flush_calls == flush_calls);
        CHECK(bus.failed_sinks() == 1);
        REQUIRE(healthy->lines().size() == 16);
        check_sink_sequences(healthy->lines(), bus.instance_id());
    }
}

TEST_CASE("telemetry retires duplicate failed aliases and drains with no surviving sink") {
    auto failed = std::make_shared<ThrowingWriteSink>();
    TelemetryBus bus(synthetic_sink_options());
    bus.add_sink(failed);
    bus.add_sink(failed);
    TelemetryContext ctx;
    ctx.session_id = "synthetic-all-sinks-failed";
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    bus.flush();
    bus.flush();
    CHECK(failed->calls == 1);
    CHECK(bus.failed_sinks() == 1);
    CHECK_FALSE(bus.enabled(TelemetryLevel::metrics));
    CHECK_FALSE(bus.emit("test.event", ctx, json::Object{}));
    CHECK(bus.emitted_events() == 1);
    CHECK(bus.dropped_events() == 0);
    bus.shutdown();
    bus.shutdown();
    bus.flush();
    CHECK(failed->calls == 1);
}

TEST_CASE("failed sink destruction runs outside the bus mutex and emitting thread") {
    RetirementObservation observation;
    TelemetryBus bus(synthetic_sink_options());
    auto failed = std::make_shared<DestructorQuerySink>(bus, observation);
    std::weak_ptr<TelemetrySink> weak = failed;
    auto healthy = std::make_shared<MemoryTelemetrySink>();
    bus.add_sink(failed);
    bus.add_sink(healthy);
    failed.reset();
    TelemetryContext ctx;
    ctx.session_id = "synthetic-destructor-control";
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    bus.flush();
    CHECK(weak.expired());
    CHECK(observation.destroyed);
    CHECK(observation.healthy_bus_enabled);
    CHECK_FALSE(observation.on_emitting_thread);
    CHECK(bus.failed_sinks() == 1);
    CHECK(healthy->lines().size() == 1);
}

TEST_CASE("telemetry drains accepted backlog after its last sink fails") {
    auto failed = std::make_shared<DeferredFailureSink>();
    auto options = synthetic_sink_options();
    options.queue_capacity = 4;
    TelemetryBus bus(options);
    bus.add_sink(failed);
    TelemetryContext ctx;
    ctx.session_id = "synthetic-all-failed-backlog";
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    CHECK(failed->gate.wait_until_entered());
    for (int i = 0; i < 4; ++i) CHECK(bus.emit("test.event", ctx, json::Object{}));
    failed->gate.open();
    bus.flush();
    CHECK(bus.emitted_events() == 5);
    CHECK(bus.failed_sinks() == 1);
    CHECK(failed->calls == 1);
    CHECK_FALSE(bus.enabled(TelemetryLevel::metrics));
    CHECK_FALSE(bus.emit("test.event", ctx, json::Object{}));
    bus.shutdown();
    bus.flush();
}

TEST_CASE("ostream exception masks cannot terminate the telemetry writer") {
    for (const bool fail_flush : {false, true}) {
        CAPTURE(fail_flush);
        FailingStreamBuffer buffer(fail_flush);
        std::ostream stream(&buffer);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        auto healthy = std::make_shared<MemoryTelemetrySink>();
        TelemetryBus bus(synthetic_sink_options());
        bus.add_sink(std::shared_ptr<TelemetrySink>(make_ostream_sink(stream)));
        bus.add_sink(healthy);
        TelemetryContext ctx;
        ctx.session_id = "synthetic-ostream-failure";
        CHECK(bus.emit("test.event", ctx, json::Object{}));
        bus.flush();
        CHECK(bus.failed_sinks() == 1);
        const int writes = buffer.write_calls;
        const int flushes = buffer.flush_calls;
        CHECK(bus.emit("test.event", ctx, json::Object{}));
        bus.flush();
        CHECK(buffer.write_calls == writes);
        CHECK(buffer.flush_calls == flushes);
        CHECK(healthy->lines().size() == 2);
        check_sink_sequences(healthy->lines(), bus.instance_id());
    }
}

TEST_CASE("builtin ostream fail bits retire the sink without changing its exception mask") {
    for (const bool fail_flush : {false, true}) {
        CAPTURE(fail_flush);
        FailingStreamBuffer buffer(fail_flush);
        std::ostream stream(&buffer);
        REQUIRE(stream.exceptions() == std::ios::goodbit);
        auto healthy = std::make_shared<MemoryTelemetrySink>();
        auto sibling = std::make_shared<MemoryTelemetrySink>();
        auto failed = std::shared_ptr<TelemetrySink>(make_ostream_sink(stream));
        TelemetryBus bus(synthetic_sink_options());
        bus.add_sink(failed);
        bus.add_sink(failed);
        bus.add_sink(healthy);
        bus.add_sink(sibling);
        TelemetryContext ctx;
        ctx.session_id = "synthetic-default-mask-failure";
        for (int i = 0; i < 8; ++i) CHECK(bus.emit("test.event", ctx, json::Object{{"i", i}}));
        bus.flush();
        CHECK(stream.bad());
        CHECK(stream.exceptions() == std::ios::goodbit);
        CHECK(bus.failed_sinks() == 1);
        const int writes = buffer.write_calls;
        const int flushes = buffer.flush_calls;
        for (int i = 8; i < 16; ++i) CHECK(bus.emit("test.event", ctx, json::Object{{"i", i}}));
        bus.flush();
        CHECK(buffer.write_calls == writes);
        CHECK(buffer.flush_calls == flushes);
        CHECK(bus.failed_sinks() == 1);
        CHECK(bus.dropped_events() == 0);
        REQUIRE(healthy->lines().size() == 16);
        CHECK(healthy->lines() == sibling->lines());
        check_sink_sequences(healthy->lines(), bus.instance_id());
    }
}

TEST_CASE("an already failed builtin stream drains without retry or mask mutation") {
    for (const auto state : {std::ios::failbit, std::ios::badbit}) {
        CAPTURE(static_cast<int>(state));
        FailingStreamBuffer buffer(false);
        std::ostream stream(&buffer);
        stream.setstate(state);
        TelemetryBus bus(synthetic_sink_options());
        bus.add_sink(std::shared_ptr<TelemetrySink>(make_ostream_sink(stream)));
        TelemetryContext ctx;
        ctx.session_id = "synthetic-already-failed-stream";
        CHECK(bus.emit("test.event", ctx, json::Object{}));
        bus.flush();
        CHECK(bus.failed_sinks() == 1);
        CHECK(stream.exceptions() == std::ios::goodbit);
        CHECK(buffer.write_calls == 0);
        CHECK(buffer.flush_calls == 0);
        CHECK_FALSE(bus.enabled(TelemetryLevel::metrics));
        CHECK_FALSE(bus.emit("test.event", ctx, json::Object{}));
        CHECK(bus.emitted_events() == 1);
        CHECK(bus.dropped_events() == 0);
        bus.shutdown();
        bus.shutdown();
        bus.flush();
    }
}

TEST_CASE("healthy builtin sinks preserve JSONL bytes and file append and open errors") {
    struct TemporaryDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() / make_id("sonder-telemetry");
        TemporaryDirectory() { std::filesystem::create_directory(path); }
        ~TemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } temporary;
    const auto path = temporary.path / "events.jsonl";
    Status status;
    REQUIRE(make_jsonl_file_sink((temporary.path / "absent" / "events.jsonl").string(), false, &status) == nullptr);
    CHECK(status.code() == ErrorCode::io_error);
    auto read_file = [&] {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    };
    std::string expected;
    for (const bool append : {false, true}) {
        std::ostringstream stream;
        auto file = make_jsonl_file_sink(path.string(), append, &status);
        REQUIRE(file);
        CHECK(status.ok());
        auto healthy = std::make_shared<MemoryTelemetrySink>();
        TelemetryBus bus(synthetic_sink_options());
        bus.add_sink(std::shared_ptr<TelemetrySink>(std::move(file)));
        bus.add_sink(std::shared_ptr<TelemetrySink>(make_ostream_sink(stream)));
        bus.add_sink(healthy);
        TelemetryContext ctx;
        ctx.session_id = "synthetic-healthy-builtin-sinks";
        for (int i = 0; i < 8; ++i) CHECK(bus.emit("test.event", ctx, json::Object{{"i", i}}));
        bus.flush();
        CHECK(bus.failed_sinks() == 0);
        CHECK(bus.dropped_events() == 0);
        CHECK(stream.good());
        CHECK(stream.exceptions() == std::ios::goodbit);
        std::string lines;
        for (const auto& line : healthy->lines()) lines += line + '\n';
        REQUIRE(healthy->lines().size() == 8);
        CHECK(stream.str() == lines);
        expected += lines;
        CHECK(read_file() == expected);
        check_sink_sequences(healthy->lines(), bus.instance_id());
    }
}

#if defined(__linux__)
TEST_CASE("builtin file sink retires on ordinary device I/O failure") {
    // /dev/full is a standard Linux ENOSPC control; append avoids truncation.
    Status status;
    auto file = make_jsonl_file_sink("/dev/full", true, &status);
    REQUIRE(file);
    REQUIRE(status.ok());
    auto healthy = std::make_shared<MemoryTelemetrySink>();
    TelemetryBus bus(synthetic_sink_options());
    bus.add_sink(std::shared_ptr<TelemetrySink>(std::move(file)));
    bus.add_sink(healthy);
    TelemetryContext ctx;
    ctx.session_id = "synthetic-file-io-error";
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    bus.flush();
    CHECK(bus.failed_sinks() == 1);
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    bus.flush();
    CHECK(bus.failed_sinks() == 1);
    CHECK(bus.dropped_events() == 0);
    REQUIRE(healthy->lines().size() == 2);
    check_sink_sequences(healthy->lines(), bus.instance_id());
}
#endif

TEST_CASE("mock generation survives builtin default-mask failures with text capture off") {
    for (const bool fail_flush : {false, true}) {
        FailingStreamBuffer buffer(fail_flush);
        std::ostream stream(&buffer);
        auto healthy = std::make_shared<MemoryTelemetrySink>();
        EngineOptions options;
        options.telemetry = synthetic_sink_options();
        options.device_sample_interval = std::chrono::milliseconds(0);
        options.telemetry_sinks = {std::shared_ptr<TelemetrySink>(make_ostream_sink(stream)), healthy};
        Engine engine(options);
        REQUIRE(engine.register_backend(make_mock_backend()).ok());
        ModelLoadOptions load;
        load.model = "mock:tiny";
        const auto model = engine.load_model(kMockBackendName, load);
        REQUIRE(model.ok());
        SessionOptions session_options;
        session_options.sampling = SamplingConfig::greedy(8);
        const auto session = engine.create_session(model.value(), session_options);
        REQUIRE(session.ok());
        const std::string prompt = "synthetic builtin failure private prompt canary";
        const auto result = session.value()->generate(prompt);
        REQUIRE(result.ok());
        CHECK(result.value().stats.completion_tokens == 8);
        session.value()->close();
        engine.telemetry().flush();
        CHECK(engine.telemetry().failed_sinks() == 1);
        CHECK(stream.bad());
        CHECK(stream.exceptions() == std::ios::goodbit);
        const auto lines = healthy->lines();
        REQUIRE_FALSE(lines.empty());
        bool completed = false;
        for (const auto& line : lines) {
            CHECK(line.find(prompt) == std::string::npos);
            CHECK(line.find("telemetry stream write failed") == std::string::npos);
            CHECK(line.find("telemetry stream flush failed") == std::string::npos);
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            const auto type = parsed.value().find("event_type")->as_string();
            if (type == "request.completed") completed = true;
            if (type == "inference.token.generated") CHECK(parsed.value().find("attributes")->find("text") == nullptr);
        }
        CHECK(completed);
        check_sink_sequences(lines, engine.telemetry().instance_id());
    }
}

TEST_CASE("sink failures remain separate from bounded queue drops") {
    auto gate = std::make_shared<GateSink>();
    auto failed = std::make_shared<ThrowingWriteSink>();
    auto options = synthetic_sink_options();
    options.queue_capacity = 4;
    options.drop_report_interval = std::chrono::milliseconds(0);
    TelemetryBus bus(options);
    bus.add_sink(gate);
    bus.add_sink(failed);
    TelemetryContext ctx;
    ctx.session_id = "synthetic-queue-pressure";
    CHECK(bus.emit("test.event", ctx, json::Object{}));
    CHECK(gate->wait_until_entered());
    int accepted = 1;
    for (int i = 0; i < 64; ++i) {
        if (bus.emit("test.event", ctx, json::Object{{"i", i}})) ++accepted;
    }
    CHECK(accepted == 5);
    CHECK(bus.dropped_events() == 60);
    gate->open();
    bus.shutdown();
    CHECK(bus.failed_sinks() == 1);
    CHECK(failed->calls == 1);
    CHECK(bus.emitted_events() == static_cast<std::uint64_t>(accepted));
    const auto lines = gate->lines();
    REQUIRE(lines.size() == static_cast<std::size_t>(accepted + 1));
    check_sink_sequences(lines, bus.instance_id());
    const auto report = json::parse(lines.back());
    REQUIRE(report.ok());
    CHECK(report.value().find("event_type")->as_string() == "telemetry.dropped");
    CHECK(report.value().find("attributes")->find("dropped_events")->as_int() == 60);
}

TEST_CASE("mock generation survives a failed telemetry sink without capturing text") {
    auto failed = std::make_shared<ThrowingWriteSink>();
    auto healthy = std::make_shared<MemoryTelemetrySink>();
    EngineOptions options;
    options.telemetry = synthetic_sink_options();
    options.device_sample_interval = std::chrono::milliseconds(0);
    options.telemetry_sinks = {failed, healthy};
    Engine engine(options);
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    ModelLoadOptions load;
    load.model = "mock:tiny";
    auto model = engine.load_model(kMockBackendName, load);
    REQUIRE(model.ok());
    SessionOptions session_options;
    session_options.sampling = SamplingConfig::greedy(8);
    auto session = engine.create_session(model.value(), session_options);
    REQUIRE(session.ok());
    const std::string prompt = "synthetic private prompt canary";
    const auto result = session.value()->generate(prompt);
    REQUIRE(result.ok());
    CHECK(result.value().stats.completion_tokens > 0);
    session.value()->close();
    engine.telemetry().flush();
    CHECK(failed->calls == 1);
    CHECK(engine.telemetry().failed_sinks() == 1);
    const auto lines = healthy->lines();
    REQUIRE_FALSE(lines.empty());
    bool completed = false;
    for (const auto& line : lines) {
        CHECK(line.find(prompt) == std::string::npos);
        CHECK(line.find("synthetic telemetry write failure") == std::string::npos);
        auto value = json::parse(line);
        REQUIRE(value.ok());
        const auto type = value.value().find("event_type")->as_string();
        if (type == "request.completed") completed = true;
        if (type == "inference.token.generated") CHECK(value.value().find("attributes")->find("text") == nullptr);
    }
    CHECK(completed);
    check_sink_sequences(lines, engine.telemetry().instance_id());
}
