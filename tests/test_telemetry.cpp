#include <doctest/doctest.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
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

// The stream identity contract: `sequence` is contiguous from 0 within each
// sequence stream of a producer instance (producer.stream absent = the main
// stream, "samples" = the periodic-sample stream), and event_id is
// "<instance_id>-<sequence>" (main) or "<instance_id>-s<sequence>" (samples).
// Feed events in emission order.
struct StreamSequences {
    std::int64_t next[2] = {0, 0};
    void check(const json::Value& e) {
        const json::Value* producer = e.find("producer");
        REQUIRE(producer);
        const json::Value* named = producer->find("stream");
        const bool samples = named != nullptr && named->is_string() && named->as_string() == "samples";
        CHECK_MESSAGE((named == nullptr || samples), "only the samples stream is named");
        const std::string type = e.find("event_type")->as_string();
        const bool sample_type = type.size() > 7 && type.compare(type.size() - 7, 7, ".sample") == 0;
        CHECK_MESSAGE(samples == sample_type, type);
        const std::int64_t seq = e.find("sequence")->as_int();
        CHECK_MESSAGE(seq == next[samples ? 1 : 0]++, type);
        CHECK(e.find("event_id")->as_string() ==
              producer->find("instance_id")->as_string() + (samples ? "-s" : "-") + std::to_string(seq));
    }
};

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
    StreamSequences streams;
    std::set<std::string> ids;
    for (const auto& e : events) {
        check_envelope(e);
        streams.check(e);
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

TEST_CASE("shutdown is bounded when a sink never returns") {
    auto gate = std::make_shared<GateSink>();
    TelemetryOptions opts;
    opts.shutdown_timeout = std::chrono::milliseconds(200);
    auto bus = std::make_unique<TelemetryBus>(opts);
    bus->add_sink(gate);
    TelemetryContext ctx;
    ctx.session_id = "sess-bounded-shutdown";
    REQUIRE(bus->emit("test.event", ctx, json::Object{}));
    REQUIRE(gate->wait_until_entered());  // the writer is now stuck inside the sink

    // Run on a helper thread so a regression fails instead of hanging the suite.
    auto result = std::async(std::launch::async, [&bus] { return bus->shutdown(); });
    const auto status = result.wait_for(std::chrono::seconds(10));
    CHECK(status == std::future_status::ready);
    gate->open();  // releases the abandoned writer (and the helper on a regression)
    REQUIRE(result.get() == false);
    CHECK(bus->shutdown() == false);  // idempotent: repeats the first result
    bus.reset();  // destroying the bus while the abandoned writer winds down must be safe
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_CASE("shutdown_timeout does not cut a healthy drain short") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    TelemetryOptions opts;
    opts.shutdown_timeout = std::chrono::milliseconds(5000);
    TelemetryBus bus(opts);
    bus.add_sink(sink);
    TelemetryContext ctx;
    ctx.session_id = "sess-healthy-shutdown";
    for (int i = 0; i < 20; ++i) REQUIRE(bus.emit("test.event", ctx, json::Object{{"i", i}}));
    CHECK(bus.shutdown() == true);
    CHECK(sink->lines().size() == 20);
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
    StreamSequences streams;
    for (const auto& e : h.events()) {
        const auto* producer = e.find("producer");
        REQUIRE(producer->find("instance_id"));
        CHECK(producer->find("instance_id")->as_string() == instance);
        streams.check(e);
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
    eo.sample_gpu_devices = false;
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
        // A samples-stream event reaches write(), never write_event(), so a
        // sink indexed by sequence never sees two number spaces mixed.
        bus.emit("x.sample", ctx_s(), json::Object{});
        bus.emit("test.event", ctx_s(), json::Object{{"i", 20}});
        bus.flush();
    }
    std::lock_guard<std::mutex> lock(sink->mutex);
    REQUIRE(sink->pairs.size() == 21);
    for (std::size_t i = 0; i < sink->pairs.size(); ++i) {
        CHECK(sink->pairs[i].first == i);
        CHECK(sink->pairs[i].second == i);
    }
    CHECK(sink->plain_writes == 1);
    // The default write_event forwards to write().
    MemoryTelemetrySink memory;
    memory.write_event(7, "{\"a\":1}");
    REQUIRE(memory.lines().size() == 1);
    CHECK(memory.lines()[0] == "{\"a\":1}");
}

namespace {
class StreamSink final : public TelemetrySink {
public:
    void write(std::string_view) override { ++plain_writes; }
    void write_stream_event(TelemetryStream stream, std::uint64_t sequence, std::string_view line) override {
        seen.push_back({stream, sequence, std::string(line)});
    }
    struct Seen {
        TelemetryStream stream;
        std::uint64_t sequence;
        std::string line;
    };
    std::vector<Seen> seen;  // written by the writer thread; read after flush()
    int plain_writes = 0;
};
}  // namespace

TEST_CASE("the two sequence streams count independently and sinks can see the stream") {
    CHECK(telemetry_stream_of("device.memory.sample") == TelemetryStream::samples);
    CHECK(telemetry_stream_of("backend.metrics.sample") == TelemetryStream::samples);
    CHECK(telemetry_stream_of("x.sample") == TelemetryStream::samples);
    CHECK(telemetry_stream_of(".sample") == TelemetryStream::main);
    CHECK(telemetry_stream_of("sample") == TelemetryStream::main);
    CHECK(telemetry_stream_of("request.sampled") == TelemetryStream::main);
    CHECK(telemetry_stream_of("telemetry.dropped") == TelemetryStream::main);
    CHECK(telemetry_stream_of("") == TelemetryStream::main);

    auto memory = std::make_shared<MemoryTelemetrySink>();
    auto stream_sink = std::make_shared<StreamSink>();
    std::string instance;
    {
        TelemetryBus bus;
        instance = bus.instance_id();
        bus.add_sink(memory);
        bus.add_sink(stream_sink);
        for (const char* type : {"test.event", "x.sample", "test.event", "x.sample", "x.sample", "test.event"}) {
            REQUIRE(bus.emit(type, ctx_s(), json::Object{}));
        }
        bus.flush();
    }
    const auto lines = memory->lines();
    // The memory sink is not stream-aware: it gets every event through write().
    REQUIRE(lines.size() == 6);
    StreamSequences streams;
    std::vector<std::string> ids;
    for (const auto& line : lines) {
        auto parsed = json::parse(line);
        REQUIRE(parsed.ok());
        check_envelope(parsed.value());
        streams.check(parsed.value());
        ids.push_back(parsed.value().find("event_id")->as_string());
    }
    CHECK(ids == std::vector<std::string>{instance + "-0", instance + "-s0", instance + "-1", instance + "-s1",
                                          instance + "-s2", instance + "-2"});
    CHECK(streams.next[0] == 3);
    CHECK(streams.next[1] == 3);
    // A sample envelope names its stream; a main envelope has no such field.
    CHECK(json::parse(lines[1]).value().find("producer")->find("stream")->as_string() == "samples");
    CHECK(json::parse(lines[0]).value().find("producer")->find("stream") == nullptr);
    // A stream-aware sink gets the stream and the per-stream number, and write() is not used.
    REQUIRE(stream_sink->seen.size() == 6);
    CHECK(stream_sink->plain_writes == 0);
    const std::vector<std::pair<TelemetryStream, std::uint64_t>> expected{
        {TelemetryStream::main, 0},    {TelemetryStream::samples, 0}, {TelemetryStream::main, 1},
        {TelemetryStream::samples, 1}, {TelemetryStream::samples, 2}, {TelemetryStream::main, 2}};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(stream_sink->seen[i].stream == expected[i].first);
        CHECK(stream_sink->seen[i].sequence == expected[i].second);
        CHECK(stream_sink->seen[i].line == lines[i]);
    }
    // make_envelope follows the event type.
    TelemetryBus plain;
    const json::Object sample_env = plain.make_envelope("x.sample", ctx_s(), json::Object{}, 4);
    CHECK(sample_env.find("event_id")->as_string() == plain.instance_id() + "-s4");
    CHECK(sample_env.find("producer")->find("stream")->as_string() == "samples");
    const json::Object main_env = plain.make_envelope("x.y", ctx_s(), json::Object{}, 4);
    CHECK(main_env.find("event_id")->as_string() == plain.instance_id() + "-4");
    CHECK(main_env.find("producer")->find("stream") == nullptr);
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
        if (e.find("session_id")->as_string() == session->options().session_id ||
            (e.find("attributes")->find("backend") &&
             e.find("attributes")->find("backend")->as_string() == kMockBackendName)) {
            REQUIRE(synthetic);
            CHECK(synthetic->as_bool());
        }
    }
}

TEST_CASE("known synthetic work overrides a host false without classifying other events") {
    for (const auto host : {std::optional<bool>{}, std::optional<bool>{false}, std::optional<bool>{true}}) {
        TelemetryOptions options;
        options.synthetic = host;
        TelemetryBus bus(options);
        auto context = ctx_s();
        const auto unclassified = bus.make_envelope("x.y", context, json::Object{}, 0);
        const auto* inherited = unclassified.find("producer")->find("synthetic");
        if (host) {
            REQUIRE(inherited);
            CHECK(inherited->as_bool() == *host);
        } else {
            CHECK(inherited == nullptr);
        }
        context.synthetic_work = true;
        const auto known = bus.make_envelope("x.y", context, json::Object{}, 1);
        REQUIRE(known.find("producer")->find("synthetic"));
        CHECK(known.find("producer")->find("synthetic")->as_bool());
    }
}

namespace {
// A deterministic test adapter with an unclassified backend identity. The
// descriptor's mock model name must not decide envelope provenance.
class UnclassifiedTestBackend final : public Backend {
public:
    std::string name() const override { return "unclassified-test"; }
    std::string description() const override { return "synthetic test adapter with unknown identity"; }
    BackendCapabilities capabilities() const override { return mock_->capabilities(); }
    Result<std::string> probe() override { return mock_->probe(); }
    Result<std::vector<ModelDescriptor>> list_models() override { return mock_->list_models(); }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        return mock_->load_model(options);
    }
private:
    std::shared_ptr<Backend> mock_ = make_mock_backend();
};
}  // namespace

TEST_CASE("a mixed engine labels only known mock backend model and session work") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions options;
    options.telemetry.level = TelemetryLevel::deep;
    options.telemetry_sinks.push_back(sink);
    options.device_sample_interval = std::chrono::milliseconds(0);
    Engine engine(std::move(options));
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    REQUIRE(engine.register_backend(std::make_shared<UnclassifiedTestBackend>()).ok());
    std::set<std::string> mock_events;
    std::set<std::string> unknown_events;
    for (const char* backend : {kMockBackendName, "unclassified-test"}) {
        ModelLoadOptions load;
        load.model = "mock:tiny";
        auto model = engine.load_model(backend, load);
        REQUIRE(model.ok());
        SessionOptions config;
        config.session_id = backend;
        config.sampling = SamplingConfig::greedy(4);
        auto session = engine.create_session(model.value(), config);
        REQUIRE(session.ok());
        REQUIRE(session.value()->generate("private provenance canary").ok());
        REQUIRE(session.value()->chat({{"user", "private provenance canary"}}).ok());
        session.value()->close();
        REQUIRE(engine.unload_model(model.value()->instance_id()).ok());
    }
    engine.telemetry().flush();
    StreamSequences streams;
    std::set<std::string> event_ids;
    for (const auto& line : sink->lines()) {
        CHECK(line.find("private provenance canary") == std::string::npos);
        auto parsed = json::parse(line);
        REQUIRE(parsed.ok());
        const auto& event = parsed.value();
        streams.check(event);
        CHECK(event_ids.insert(event.find("event_id")->as_string()).second);
        const auto type = event.find("event_type")->as_string();
        const auto sid = event.find("session_id")->as_string();
        const auto* backend = event.find("attributes")->find("backend");
        const bool known = sid == kMockBackendName || (backend && backend->as_string() == kMockBackendName);
        const auto* synthetic = event.find("producer")->find("synthetic");
        if (known) {
            REQUIRE(synthetic);
            CHECK(synthetic->as_bool());
            mock_events.insert(type);
        } else {
            CHECK(synthetic == nullptr);
            if (sid == "unclassified-test" || (backend && backend->as_string() == "unclassified-test"))
                unknown_events.insert(type);
        }
    }
    for (const char* type : {"backend.registered", "model.load.started", "model.load.completed", "model.unload",
                             "session.created", "session.closed", "request.queued", "request.completed",
                             "inference.token.generated"}) {
        CHECK_MESSAGE(mock_events.count(type) == 1, type);
        CHECK_MESSAGE(unknown_events.count(type) == 1, type);
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
    StreamSequences streams;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        auto value = json::parse(lines[i]);
        REQUIRE(value.ok());
        streams.check(value.value());
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

TEST_CASE("gpu_memory_sample_attributes covers discrete RTX, AMD RX XT, Ryzen APU UMA, and unknown readings") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // 1. NVIDIA RTX discrete GPU.
    GpuDeviceInfo rtx;
    rtx.id = "gpu:0";
    rtx.name = "NVIDIA GeForce RTX 5070 Ti";
    rtx.vendor = GpuVendor::nvidia;
    rtx.vendor_id = 0x10DEu;
    rtx.device_id = 0x2C05u;
    rtx.dedicated_total = MemoryReading::known(16303 * kMiB);
    rtx.dedicated_free = MemoryReading::known(14800 * kMiB);
    rtx.shared_total = MemoryReading::known(16 * kGiB);
    rtx.is_integrated_uma = false;
    rtx.backend_hint = "cuda";
    rtx.source = "dxgi";

    const json::Object rtx_attrs = gpu_memory_sample_attributes(rtx);
    REQUIRE(rtx_attrs.find("kind") != nullptr);
    CHECK(rtx_attrs.find("kind")->as_string() == "gpu");
    REQUIRE(rtx_attrs.find("name") != nullptr);
    CHECK(rtx_attrs.find("name")->as_string() == "NVIDIA GeForce RTX 5070 Ti");
    REQUIRE(rtx_attrs.find("vendor") != nullptr);
    CHECK(rtx_attrs.find("vendor")->as_string() == "nvidia");
    REQUIRE(rtx_attrs.find("vendor_id") != nullptr);
    CHECK(rtx_attrs.find("vendor_id")->as_uint() == std::uint64_t{0x10DEu});
    REQUIRE(rtx_attrs.find("device_id") != nullptr);
    CHECK(rtx_attrs.find("device_id")->as_uint() == std::uint64_t{0x2C05u});
    REQUIRE(rtx_attrs.find("is_integrated_uma") != nullptr);
    CHECK_FALSE(rtx_attrs.find("is_integrated_uma")->as_bool());
    REQUIRE(rtx_attrs.find("backend_hint") != nullptr);
    CHECK(rtx_attrs.find("backend_hint")->as_string() == "cuda");
    REQUIRE(rtx_attrs.find("source") != nullptr);
    CHECK(rtx_attrs.find("source")->as_string() == "dxgi");
    REQUIRE(rtx_attrs.find("dedicated_total_bytes") != nullptr);
    CHECK(rtx_attrs.find("dedicated_total_bytes")->as_uint() == 16303 * kMiB);
    REQUIRE(rtx_attrs.find("total_bytes") != nullptr);
    CHECK(rtx_attrs.find("total_bytes")->as_uint() == 16303 * kMiB);
    REQUIRE(rtx_attrs.find("dedicated_free_bytes") != nullptr);
    CHECK(rtx_attrs.find("dedicated_free_bytes")->as_uint() == 14800 * kMiB);
    REQUIRE(rtx_attrs.find("available_bytes") != nullptr);
    CHECK(rtx_attrs.find("available_bytes")->as_uint() == 14800 * kMiB);
    REQUIRE(rtx_attrs.find("used_bytes") != nullptr);
    CHECK(rtx_attrs.find("used_bytes")->as_uint() == 1503 * kMiB);
    REQUIRE(rtx_attrs.find("shared_total_bytes") != nullptr);
    CHECK(rtx_attrs.find("shared_total_bytes")->as_uint() == 16 * kGiB);
    REQUIRE(rtx_attrs.find("effective_vram_mib") != nullptr);
    CHECK(rtx_attrs.find("effective_vram_mib")->as_uint() == std::uint64_t{16303});
    REQUIRE(rtx_attrs.find("effective_free_vram_mib") != nullptr);
    CHECK(rtx_attrs.find("effective_free_vram_mib")->as_uint() == std::uint64_t{14800});

    // Known-zero free VRAM on a discrete GPU reports 0 free and full used_bytes.
    GpuDeviceInfo rtx_full = rtx;
    rtx_full.dedicated_free = MemoryReading::known(0);
    const json::Object rtx_full_attrs = gpu_memory_sample_attributes(rtx_full);
    REQUIRE(rtx_full_attrs.find("dedicated_free_bytes") != nullptr);
    CHECK(rtx_full_attrs.find("dedicated_free_bytes")->as_uint() == std::uint64_t{0});
    REQUIRE(rtx_full_attrs.find("available_bytes") != nullptr);
    CHECK(rtx_full_attrs.find("available_bytes")->as_uint() == std::uint64_t{0});
    REQUIRE(rtx_full_attrs.find("used_bytes") != nullptr);
    CHECK(rtx_full_attrs.find("used_bytes")->as_uint() == 16303 * kMiB);
    REQUIRE(rtx_full_attrs.find("effective_free_vram_mib") != nullptr);
    CHECK(rtx_full_attrs.find("effective_free_vram_mib")->as_uint() == std::uint64_t{0});

    // Anomalous free > total omits used_bytes rather than underflowing.
    GpuDeviceInfo rtx_anomalous = rtx;
    rtx_anomalous.dedicated_free = MemoryReading::known(17000 * kMiB);
    const json::Object rtx_anomalous_attrs = gpu_memory_sample_attributes(rtx_anomalous);
    CHECK(rtx_anomalous_attrs.find("used_bytes") == nullptr);

    // Envelope round-trip obeys sonder.observatory.event/1.
    TelemetryBus bus;
    auto gpu_ctx = ctx_s();
    gpu_ctx.device_id = rtx.id;
    const auto parsed_env = json::parse(json::Value(bus.make_envelope("device.memory.sample", gpu_ctx, rtx_attrs, 0)).dump());
    REQUIRE(parsed_env.ok());
    check_envelope(parsed_env.value());
    CHECK(parsed_env.value().find("device_id")->as_string() == "gpu:0");

    // 2. AMD Radeon RX XT discrete GPU.
    GpuDeviceInfo rx7900xt;
    rx7900xt.id = "gpu:0";
    rx7900xt.name = "AMD Radeon RX 7900 XT";
    rx7900xt.vendor = GpuVendor::amd;
    rx7900xt.vendor_id = 0x1002u;
    rx7900xt.device_id = 0x744Cu;
    rx7900xt.dedicated_total = MemoryReading::known(20 * kGiB);
    rx7900xt.dedicated_free = MemoryReading::known(18 * kGiB);
    rx7900xt.shared_total = MemoryReading::known(16 * kGiB);
    rx7900xt.is_integrated_uma = false;
    rx7900xt.backend_hint = "rocm";
    rx7900xt.source = "amdgpu-sysfs";

    const json::Object amd_attrs = gpu_memory_sample_attributes(rx7900xt);
    CHECK(amd_attrs.find("kind")->as_string() == "gpu");
    CHECK(amd_attrs.find("name")->as_string() == "AMD Radeon RX 7900 XT");
    CHECK(amd_attrs.find("vendor")->as_string() == "amd");
    CHECK(amd_attrs.find("vendor_id")->as_uint() == std::uint64_t{0x1002u});
    CHECK(amd_attrs.find("device_id")->as_uint() == std::uint64_t{0x744Cu});
    CHECK_FALSE(amd_attrs.find("is_integrated_uma")->as_bool());
    CHECK(amd_attrs.find("backend_hint")->as_string() == "rocm");
    CHECK(amd_attrs.find("source")->as_string() == "amdgpu-sysfs");
    REQUIRE(amd_attrs.find("dedicated_total_bytes") != nullptr);
    CHECK(amd_attrs.find("dedicated_total_bytes")->as_uint() == 20 * kGiB);
    REQUIRE(amd_attrs.find("total_bytes") != nullptr);
    CHECK(amd_attrs.find("total_bytes")->as_uint() == 20 * kGiB);
    REQUIRE(amd_attrs.find("dedicated_free_bytes") != nullptr);
    CHECK(amd_attrs.find("dedicated_free_bytes")->as_uint() == 18 * kGiB);
    REQUIRE(amd_attrs.find("available_bytes") != nullptr);
    CHECK(amd_attrs.find("available_bytes")->as_uint() == 18 * kGiB);
    REQUIRE(amd_attrs.find("used_bytes") != nullptr);
    CHECK(amd_attrs.find("used_bytes")->as_uint() == 2 * kGiB);
    REQUIRE(amd_attrs.find("shared_total_bytes") != nullptr);
    CHECK(amd_attrs.find("shared_total_bytes")->as_uint() == 16 * kGiB);
    REQUIRE(amd_attrs.find("effective_vram_mib") != nullptr);
    CHECK(amd_attrs.find("effective_vram_mib")->as_uint() == std::uint64_t{20480});
    REQUIRE(amd_attrs.find("effective_free_vram_mib") != nullptr);
    CHECK(amd_attrs.find("effective_free_vram_mib")->as_uint() == std::uint64_t{18432});

    // 3. AMD Ryzen APU integrated UMA GPU.
    GpuDeviceInfo apu;
    apu.id = "gpu:1";
    apu.name = "AMD Radeon 780M Graphics";
    apu.vendor = GpuVendor::amd;
    apu.vendor_id = 0x1002u;
    apu.device_id = 0x15BFu;
    apu.dedicated_total = MemoryReading::known(512 * kMiB);
    apu.dedicated_free = MemoryReading::known(384 * kMiB + 14 * kGiB);
    apu.shared_total = MemoryReading::known(16 * kGiB);
    apu.is_integrated_uma = true;
    apu.backend_hint = "rocm";
    apu.source = "amdgpu-sysfs";

    const json::Object apu_attrs = gpu_memory_sample_attributes(apu);
    CHECK(apu_attrs.find("kind")->as_string() == "gpu");
    REQUIRE(apu_attrs.find("gpu_id") != nullptr);
    CHECK(apu_attrs.find("gpu_id")->as_string() == "gpu:1");
    CHECK(apu_attrs.find("name")->as_string() == "AMD Radeon 780M Graphics");
    CHECK(apu_attrs.find("vendor")->as_string() == "amd");
    CHECK(apu_attrs.find("device_id")->as_uint() == std::uint64_t{0x15BFu});
    CHECK(apu_attrs.find("is_integrated_uma")->as_bool());
    REQUIRE(apu_attrs.find("dedicated_total_bytes") != nullptr);
    CHECK(apu_attrs.find("dedicated_total_bytes")->as_uint() == 512 * kMiB);
    REQUIRE(apu_attrs.find("dedicated_free_bytes") != nullptr);
    CHECK(apu_attrs.find("dedicated_free_bytes")->as_uint() == 384 * kMiB + 14 * kGiB);
    // UMA total/available/used bytes are paired consistently against the
    // unified pool (dedicated BIOS carveout + shared/GTT), matching
    // effective_vram_mib and effective_free_vram_mib so Observatory tracks
    // Radeon 780M iGPU samples without dropping them.
    const std::uint64_t expected_uma_total = 512 * kMiB + 16 * kGiB;
    const std::uint64_t expected_uma_free = 384 * kMiB + 14 * kGiB;
    REQUIRE(apu_attrs.find("total_bytes") != nullptr);
    CHECK(apu_attrs.find("total_bytes")->as_uint() == expected_uma_total);
    REQUIRE(apu_attrs.find("available_bytes") != nullptr);
    CHECK(apu_attrs.find("available_bytes")->as_uint() == expected_uma_free);
    REQUIRE(apu_attrs.find("used_bytes") != nullptr);
    CHECK(apu_attrs.find("used_bytes")->as_uint() == expected_uma_total - expected_uma_free);
    REQUIRE(apu_attrs.find("shared_total_bytes") != nullptr);
    CHECK(apu_attrs.find("shared_total_bytes")->as_uint() == 16 * kGiB);
    REQUIRE(apu_attrs.find("effective_vram_mib") != nullptr);
    CHECK(apu_attrs.find("effective_vram_mib")->as_uint() == std::uint64_t{512 + 16384});
    REQUIRE(apu_attrs.find("effective_free_vram_mib") != nullptr);
    CHECK(apu_attrs.find("effective_free_vram_mib")->as_uint() == std::uint64_t{384 + 14336});
    CHECK(apu_attrs.find("total_bytes")->as_uint() / kMiB == apu_attrs.find("effective_vram_mib")->as_uint());
    CHECK(apu_attrs.find("available_bytes")->as_uint() / kMiB == apu_attrs.find("effective_free_vram_mib")->as_uint());

    // UMA envelope round-trip carries distinct device_id ("gpu:1") and kind ("gpu").
    auto uma_ctx = ctx_s();
    uma_ctx.device_id = apu.id;
    const auto parsed_uma_env =
        json::parse(json::Value(bus.make_envelope("device.memory.sample", uma_ctx, apu_attrs, 1)).dump());
    REQUIRE(parsed_uma_env.ok());
    check_envelope(parsed_uma_env.value());
    CHECK(parsed_uma_env.value().find("device_id")->as_string() == "gpu:1");
    CHECK(parsed_uma_env.value().find("attributes")->find("kind")->as_string() == "gpu");
    CHECK(parsed_uma_env.value().find("attributes")->find("total_bytes")->as_uint() == expected_uma_total);
    CHECK(parsed_uma_env.value().find("attributes")->find("used_bytes")->as_uint() ==
          expected_uma_total - expected_uma_free);

    // UMA with known-zero free reports available_bytes=0 and full used_bytes.
    GpuDeviceInfo apu_full = apu;
    apu_full.dedicated_free = MemoryReading::known(0);
    const json::Object apu_full_attrs = gpu_memory_sample_attributes(apu_full);
    REQUIRE(apu_full_attrs.find("total_bytes") != nullptr);
    CHECK(apu_full_attrs.find("total_bytes")->as_uint() == expected_uma_total);
    REQUIRE(apu_full_attrs.find("available_bytes") != nullptr);
    CHECK(apu_full_attrs.find("available_bytes")->as_uint() == std::uint64_t{0});
    REQUIRE(apu_full_attrs.find("used_bytes") != nullptr);
    CHECK(apu_full_attrs.find("used_bytes")->as_uint() == expected_uma_total);

    // Anomalous UMA free > total caps available_bytes to total_pool (matching
    // effective_free_vram_mib) and omits used_bytes rather than underflowing,
    // while preserving raw dedicated_free_bytes.
    GpuDeviceInfo apu_anomalous = apu;
    apu_anomalous.dedicated_free = MemoryReading::known(expected_uma_total + 2 * kGiB);
    const json::Object apu_anomalous_attrs = gpu_memory_sample_attributes(apu_anomalous);
    REQUIRE(apu_anomalous_attrs.find("dedicated_free_bytes") != nullptr);
    CHECK(apu_anomalous_attrs.find("dedicated_free_bytes")->as_uint() == expected_uma_total + 2 * kGiB);
    REQUIRE(apu_anomalous_attrs.find("total_bytes") != nullptr);
    CHECK(apu_anomalous_attrs.find("total_bytes")->as_uint() == expected_uma_total);
    REQUIRE(apu_anomalous_attrs.find("available_bytes") != nullptr);
    CHECK(apu_anomalous_attrs.find("available_bytes")->as_uint() == expected_uma_total);
    CHECK(apu_anomalous_attrs.find("used_bytes") == nullptr);
    REQUIRE(apu_anomalous_attrs.find("effective_free_vram_mib") != nullptr);
    CHECK(apu_anomalous_attrs.find("effective_free_vram_mib")->as_uint() == std::uint64_t{512 + 16384});

    // UMA with shared_total only (no dedicated carveout) still emits total_bytes.
    GpuDeviceInfo apu_shared_only;
    apu_shared_only.id = "gpu:1";
    apu_shared_only.name = "AMD Radeon 780M Graphics";
    apu_shared_only.vendor = GpuVendor::amd;
    apu_shared_only.shared_total = MemoryReading::known(32 * kGiB);
    apu_shared_only.dedicated_free = MemoryReading::known(28 * kGiB);
    apu_shared_only.is_integrated_uma = true;
    apu_shared_only.backend_hint = "vulkan";
    apu_shared_only.source = "dxgi";
    const json::Object apu_shared_attrs = gpu_memory_sample_attributes(apu_shared_only);
    REQUIRE(apu_shared_attrs.find("total_bytes") != nullptr);
    CHECK(apu_shared_attrs.find("total_bytes")->as_uint() == 32 * kGiB);
    REQUIRE(apu_shared_attrs.find("available_bytes") != nullptr);
    CHECK(apu_shared_attrs.find("available_bytes")->as_uint() == 28 * kGiB);
    REQUIRE(apu_shared_attrs.find("used_bytes") != nullptr);
    CHECK(apu_shared_attrs.find("used_bytes")->as_uint() == 4 * kGiB);

    // UMA with unknown free reading emits total_bytes and effective_vram_mib,
    // while omitting available_bytes, used_bytes, and effective_free_vram_mib.
    GpuDeviceInfo apu_no_free = apu;
    apu_no_free.dedicated_free = MemoryReading::unknown();
    const json::Object apu_no_free_attrs = gpu_memory_sample_attributes(apu_no_free);
    REQUIRE(apu_no_free_attrs.find("total_bytes") != nullptr);
    CHECK(apu_no_free_attrs.find("total_bytes")->as_uint() == expected_uma_total);
    CHECK(apu_no_free_attrs.find("dedicated_free_bytes") == nullptr);
    CHECK(apu_no_free_attrs.find("available_bytes") == nullptr);
    CHECK(apu_no_free_attrs.find("used_bytes") == nullptr);
    REQUIRE(apu_no_free_attrs.find("effective_vram_mib") != nullptr);
    CHECK(apu_no_free_attrs.find("effective_vram_mib")->as_uint() == std::uint64_t{512 + 16384});
    CHECK(apu_no_free_attrs.find("effective_free_vram_mib") == nullptr);

    // UMA with unknown dedicated & shared totals omits total_bytes and used_bytes.
    GpuDeviceInfo apu_unknown_pool;
    apu_unknown_pool.id = "gpu:1";
    apu_unknown_pool.name = "Unknown UMA";
    apu_unknown_pool.is_integrated_uma = true;
    const json::Object apu_unknown_attrs = gpu_memory_sample_attributes(apu_unknown_pool);
    CHECK(apu_unknown_attrs.find("total_bytes") == nullptr);
    CHECK(apu_unknown_attrs.find("available_bytes") == nullptr);
    CHECK(apu_unknown_attrs.find("used_bytes") == nullptr);
    CHECK(apu_unknown_attrs.find("effective_vram_mib") == nullptr);
    CHECK(apu_unknown_attrs.find("effective_free_vram_mib") == nullptr);

    // 4. Unknown memory readings omit all optional byte and MiB metrics.
    GpuDeviceInfo unknown_gpu;
    unknown_gpu.id = "gpu:2";
    unknown_gpu.name = "Unknown GPU";
    unknown_gpu.vendor = GpuVendor::unknown;
    unknown_gpu.vendor_id = 0;
    unknown_gpu.device_id = 0;
    unknown_gpu.is_integrated_uma = false;
    unknown_gpu.backend_hint = "";
    unknown_gpu.source = "probe";

    const json::Object unknown_attrs = gpu_memory_sample_attributes(unknown_gpu);
    CHECK(unknown_attrs.find("kind")->as_string() == "gpu");
    CHECK(unknown_attrs.find("name")->as_string() == "Unknown GPU");
    CHECK(unknown_attrs.find("vendor")->as_string() == "unknown");
    CHECK(unknown_attrs.find("vendor_id")->as_uint() == std::uint64_t{0});
    CHECK(unknown_attrs.find("device_id")->as_uint() == std::uint64_t{0});
    CHECK_FALSE(unknown_attrs.find("is_integrated_uma")->as_bool());
    CHECK(unknown_attrs.find("backend_hint")->as_string().empty());
    CHECK(unknown_attrs.find("source")->as_string() == "probe");
    CHECK(unknown_attrs.find("dedicated_total_bytes") == nullptr);
    CHECK(unknown_attrs.find("total_bytes") == nullptr);
    CHECK(unknown_attrs.find("dedicated_free_bytes") == nullptr);
    CHECK(unknown_attrs.find("available_bytes") == nullptr);
    CHECK(unknown_attrs.find("used_bytes") == nullptr);
    CHECK(unknown_attrs.find("shared_total_bytes") == nullptr);
    CHECK(unknown_attrs.find("effective_vram_mib") == nullptr);
    CHECK(unknown_attrs.find("effective_free_vram_mib") == nullptr);
}

TEST_CASE("engine startup honors sample_gpu_devices false vs true") {
    // Disabled: no GPU enumeration, gpu_device_count == 0, zero GPU memory samples.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = true;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        std::size_t cpu_count = 0;
        {
            Engine engine(std::move(eo));
            CHECK(engine.gpu_devices().empty());
            cpu_count = engine.devices().size();
            engine.telemetry().flush();
        }
        std::size_t started_events = 0;
        std::size_t cpu_samples = 0;
        std::size_t gpu_samples = 0;
        for (const auto& line : sink->lines()) {
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            const std::string type = parsed.value().find("event_type")->as_string();
            const json::Value* attrs = parsed.value().find("attributes");
            REQUIRE(attrs != nullptr);
            if (type == "engine.started") {
                ++started_events;
                REQUIRE(attrs->find("gpu_device_count") != nullptr);
                CHECK(attrs->find("gpu_device_count")->as_uint() == std::uint64_t{0});
                REQUIRE(attrs->find("device_count") != nullptr);
                CHECK(attrs->find("device_count")->as_uint() == static_cast<std::uint64_t>(cpu_count));
            } else if (type == "device.memory.sample") {
                REQUIRE(attrs->find("kind") != nullptr);
                const std::string kind = attrs->find("kind")->as_string();
                if (kind == "cpu") {
                    ++cpu_samples;
                } else if (kind == "gpu") {
                    ++gpu_samples;
                }
            }
        }
        CHECK(started_events == 1);
        CHECK(cpu_samples == cpu_count);
        CHECK(gpu_samples == 0);
    }

    // Enabled (default): gpu_device_count and startup GPU samples match engine.gpu_devices().
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        CHECK(eo.sample_gpu_devices);
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = true;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        std::vector<GpuDeviceInfo> gpus;
        std::size_t cpu_count = 0;
        {
            Engine engine(std::move(eo));
            gpus = engine.gpu_devices();
            cpu_count = engine.devices().size();
            engine.telemetry().flush();
        }
        std::size_t started_events = 0;
        std::size_t cpu_samples = 0;
        std::vector<json::Value> gpu_sample_events;
        for (const auto& line : sink->lines()) {
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            const std::string type = parsed.value().find("event_type")->as_string();
            const json::Value* attrs = parsed.value().find("attributes");
            REQUIRE(attrs != nullptr);
            if (type == "engine.started") {
                ++started_events;
                REQUIRE(attrs->find("gpu_device_count") != nullptr);
                CHECK(attrs->find("gpu_device_count")->as_uint() == static_cast<std::uint64_t>(gpus.size()));
                REQUIRE(attrs->find("device_count") != nullptr);
                CHECK(attrs->find("device_count")->as_uint() == static_cast<std::uint64_t>(cpu_count));
            } else if (type == "device.memory.sample") {
                REQUIRE(attrs->find("kind") != nullptr);
                const std::string kind = attrs->find("kind")->as_string();
                if (kind == "cpu") {
                    ++cpu_samples;
                } else if (kind == "gpu") {
                    gpu_sample_events.push_back(parsed.value());
                }
            }
        }
        CHECK(started_events == 1);
        CHECK(cpu_samples == cpu_count);
        REQUIRE(gpu_sample_events.size() == gpus.size());
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            const auto& event = gpu_sample_events[i];
            const auto& gpu = gpus[i];
            CHECK(event.find("producer")->find("synthetic") == nullptr);
            if (!gpu.id.empty()) {
                REQUIRE(event.find("device_id")->is_string());
                CHECK(event.find("device_id")->as_string() == gpu.id);
            }
            const json::Value* attrs = event.find("attributes");
            REQUIRE(attrs != nullptr);
            CHECK(attrs->find("kind")->as_string() == "gpu");
            CHECK(attrs->find("name")->as_string() == gpu.name);
            CHECK(attrs->find("vendor")->as_string() == std::string(to_string(gpu.vendor)));
            CHECK(attrs->find("is_integrated_uma")->as_bool() == gpu.is_integrated_uma);
            CHECK(attrs->find("source")->as_string() == gpu.source);
        }
    }

    // sample_devices_on_start = false suppresses startup GPU samples while keeping startup GPU inventory.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = true;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        std::size_t gpu_count = 0;
        {
            Engine engine(std::move(eo));
            gpu_count = engine.gpu_devices().size();
            engine.telemetry().flush();
        }
        std::size_t sample_events = 0;
        std::size_t started_events = 0;
        for (const auto& line : sink->lines()) {
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            const std::string type = parsed.value().find("event_type")->as_string();
            if (type == "engine.started") {
                ++started_events;
                REQUIRE(parsed.value().find("attributes")->find("gpu_device_count") != nullptr);
                CHECK(parsed.value().find("attributes")->find("gpu_device_count")->as_uint() ==
                      static_cast<std::uint64_t>(gpu_count));
            } else if (type == "device.memory.sample") {
                ++sample_events;
            }
        }
        CHECK(started_events == 1);
        CHECK(sample_events == 0);
    }
}

TEST_CASE("periodic device sampler updates gpu_devices snapshot and respects sample_gpu_devices") {
    // With sample_gpu_devices = false, periodic ticks emit only CPU samples.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(10);
        {
            Engine engine(std::move(eo));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                engine.telemetry().flush();
                CHECK(engine.gpu_devices().empty());
                bool saw_cpu = false;
                for (const auto& line : sink->lines()) {
                    const auto parsed = json::parse(line);
                    REQUIRE(parsed.ok());
                    if (parsed.value().find("event_type")->as_string() != "device.memory.sample") continue;
                    if (parsed.value().find("attributes")->find("kind")->as_string() == "cpu") {
                        saw_cpu = true;
                        break;
                    }
                }
                if (saw_cpu) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        std::size_t cpu_samples = 0;
        std::size_t gpu_samples = 0;
        for (const auto& line : sink->lines()) {
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            if (parsed.value().find("event_type")->as_string() != "device.memory.sample") continue;
            const std::string kind = parsed.value().find("attributes")->find("kind")->as_string();
            if (kind == "cpu") ++cpu_samples;
            if (kind == "gpu") ++gpu_samples;
        }
        CHECK(cpu_samples >= 1);
        CHECK(gpu_samples == 0);
    }

    // With sample_gpu_devices = true, concurrent gpu_devices() snapshots stay synchronized with periodic ticks.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = true;
        eo.device_sample_interval = std::chrono::milliseconds(10);
        std::size_t expected_gpus = 0;
        {
            Engine engine(std::move(eo));
            expected_gpus = engine.gpu_devices().size();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                for (const auto& gpu : engine.gpu_devices()) {
                    CHECK_FALSE(gpu.id.empty());
                }
                engine.telemetry().flush();
                std::size_t saw_cpu = 0;
                std::size_t saw_gpu = 0;
                for (const auto& line : sink->lines()) {
                    const auto parsed = json::parse(line);
                    REQUIRE(parsed.ok());
                    if (parsed.value().find("event_type")->as_string() != "device.memory.sample") continue;
                    const std::string kind = parsed.value().find("attributes")->find("kind")->as_string();
                    if (kind == "cpu") ++saw_cpu;
                    if (kind == "gpu") ++saw_gpu;
                }
                if (saw_cpu >= 1 && (expected_gpus == 0 || saw_gpu >= expected_gpus)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        std::size_t cpu_samples = 0;
        std::size_t gpu_samples = 0;
        for (const auto& line : sink->lines()) {
            const auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            if (parsed.value().find("event_type")->as_string() != "device.memory.sample") continue;
            const std::string kind = parsed.value().find("attributes")->find("kind")->as_string();
            if (kind == "cpu") ++cpu_samples;
            if (kind == "gpu") ++gpu_samples;
        }
        CHECK(cpu_samples >= 1);
        if (expected_gpus > 0) {
            CHECK(gpu_samples >= expected_gpus);
        }
    }
}

TEST_CASE("5-state capability status model serializes all dimensions without conflating speedups") {
    CHECK(std::string(to_string(CapabilityState::unknown)) == "unknown");
    CHECK(std::string(to_string(CapabilityState::no)) == "no");
    CHECK(std::string(to_string(CapabilityState::yes)) == "yes");
    CHECK(parse_capability_state("unknown") == CapabilityState::unknown);
    CHECK(parse_capability_state("yes") == CapabilityState::yes);
    CHECK(parse_capability_state("true") == CapabilityState::yes);
    CHECK(parse_capability_state("no") == CapabilityState::no);
    CHECK(parse_capability_state("false") == CapabilityState::no);
    CHECK_FALSE(parse_capability_state("maybe").has_value());

    CapabilityStatus st;
    st.adapter_supports = CapabilityState::yes;
    st.build_supports = CapabilityState::yes;
    st.configured = CapabilityState::yes;
    st.observed_active = CapabilityState::unknown;
    st.qualified_beneficial = CapabilityState::unknown;
    st.note = std::string("line1\nline2\t") + std::string(400, 'x');

    const json::Object obj = to_json(st);
    CHECK(obj.find("adapter_supports")->as_string() == "yes");
    CHECK(obj.find("build_supports")->as_string() == "yes");
    CHECK(obj.find("configured")->as_string() == "yes");
    CHECK(obj.find("observed_active")->as_string() == "unknown");
    CHECK(obj.find("qualified_beneficial")->as_string() == "unknown");
    REQUIRE(obj.find("note") != nullptr);
    CHECK(obj.find("note")->as_string().size() == kMaxEvidenceStringBytes);
    CHECK(obj.find("note")->as_string().find('\n') == std::string::npos);

    // Entry map is bounded at kMaxCapabilityStatusEntries.
    std::vector<CapabilityStatusEntry> many;
    for (std::size_t i = 0; i < kMaxCapabilityStatusEntries + 10; ++i) {
        many.emplace_back("cap_" + std::to_string(i), CapabilityStatus{});
    }
    const json::Object map = to_json_capability_statuses(many);
    CHECK(map.size() == kMaxCapabilityStatusEntries);
}

TEST_CASE("EffectiveExecutionReport keeps configured vs effective modes and 3-tier cache counters distinct") {
    // 1. Pure logical accounting hit MUST NEVER count as physical prefill savings.
    {
        EffectiveExecutionReport rep;
        rep.configured_spec_method = "auto";
        rep.effective_spec_method = "ngram-mod";
        rep.configured_draft_limit = 64u;
        rep.effective_draft_limit = 64u;
        rep.fallback_reason = std::string(kSpecReasonAutoFallbackNgramMod);
        rep.target_identity = "qwen3-coder.gguf";
        rep.draft_identity = "ngram-mod";
        rep.committed_tokens = 40u;
        rep.proposed_draft_tokens = 30u;
        rep.accepted_draft_tokens = 99u;  // malformed > proposed must clamp
        rep.speculation_observed_active = CapabilityState::yes;
        rep.cache.logical_cache_hit_tokens = 512u;
        rep.cache.device_prefix_reused_tokens = std::nullopt;
        rep.cache.host_l2_restored_tokens = std::nullopt;

        CHECK_FALSE(rep.cache.physical_reused_tokens().has_value());
        const json::Object j = to_json(rep);
        CHECK(j.find("configured_spec_method")->as_string() == "auto");
        CHECK(j.find("effective_spec_method")->as_string() == "ngram-mod");
        CHECK(j.find("fallback_reason")->as_string() == "auto_fallback_ngram_mod");
        CHECK(j.find("committed_tokens")->as_uint() == 40u);
        CHECK(j.find("proposed_draft_tokens")->as_uint() == 30u);
        CHECK(j.find("accepted_draft_tokens")->as_uint() == 30u);
        CHECK(j.find("speculation_observed_active")->as_string() == "yes");

        const auto* cache = j.find("cache");
        REQUIRE(cache != nullptr);
        CHECK(cache->find("logical_cache_hit_tokens")->as_uint() == 512u);
        CHECK(cache->find("device_prefix_reused_tokens")->is_null());
        CHECK(cache->find("host_l2_restored_tokens")->is_null());
        CHECK(cache->find("host_l2_restore_ms")->is_null());
        CHECK(cache->find("physical_reused_tokens")->is_null());
    }

    // 2. Physical device + L2 restore stay separate from logical accounting and sum cleanly.
    {
        EffectiveExecutionReport rep;
        rep.configured_spec_method = "draft-mtp";
        rep.effective_spec_method = "none";
        rep.configured_draft_limit = 3u;
        rep.effective_draft_limit = 0u;
        rep.fallback_reason = std::string(kSpecReasonConcurrencyBatch);
        rep.transitioned_during_request = true;
        rep.speculation_observed_active = CapabilityState::no;
        rep.cache.logical_cache_hit_tokens = 1000u;
        rep.cache.device_prefix_reused_tokens = 256u;
        rep.cache.device_cross_slot_reused_tokens = 256u;
        rep.cache.host_l2_restored_tokens = 128u;
        rep.cache.host_l2_restore_ms = 3.5;

        REQUIRE(rep.cache.physical_reused_tokens().has_value());
        CHECK(*rep.cache.physical_reused_tokens() == 384u);  // strictly excludes logical 1000
        const json::Object j = to_json(rep);
        CHECK(j.find("fallback_reason")->as_string() == "concurrency_batch");
        CHECK(j.find("transitioned_during_request")->as_bool());
        const auto* cache = j.find("cache");
        REQUIRE(cache != nullptr);
        CHECK(cache->find("logical_cache_hit_tokens")->as_uint() == 1000u);
        CHECK(cache->find("device_prefix_reused_tokens")->as_uint() == 256u);
        CHECK(cache->find("device_cross_slot_reused_tokens")->as_uint() == 256u);
        CHECK(cache->find("host_l2_restored_tokens")->as_uint() == 128u);
        CHECK(cache->find("host_l2_restore_ms")->as_double() == doctest::Approx(3.5));
        CHECK(cache->find("physical_reused_tokens")->as_uint() == 384u);
    }

    // 3. Unpopulated report uses explicit "unknown" / null rather than fabricating zeroes.
    {
        const json::Object j = to_json(EffectiveExecutionReport{});
        CHECK(j.find("configured_spec_method")->as_string() == "unknown");
        CHECK(j.find("effective_spec_method")->as_string() == "unknown");
        CHECK(j.find("configured_draft_limit")->is_null());
        CHECK(j.find("effective_draft_limit")->is_null());
        CHECK(j.find("fallback_reason")->is_null());
        CHECK(j.find("committed_tokens")->is_null());
        CHECK(j.find("proposed_draft_tokens")->is_null());
        CHECK(j.find("speculation_observed_active")->as_string() == "unknown");
    }
}

TEST_CASE("session separates logical KvCacheManager reuse from physical tiers and supports opt-in stage profiling") {
    // 1. Default run (stage profiling OFF): zero stage.timings events, logical cache hits never claim physical reuse.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        eo.scheduling.enabled = true;
        eo.scheduling.kv_block_size_tokens = 4;
        eo.scheduling.prefix_caching = true;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(make_mock_backend()).ok());
        auto model = engine.load_model(kMockBackendName, ModelLoadOptions{"mock:tiny", "cpu:0"});
        REQUIRE(model.ok());

        const std::string shared_prompt =
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi";
        SessionOptions so1;
        so1.sampling = SamplingConfig::greedy(3);
        auto s1 = engine.create_session(model.value(), so1);
        REQUIRE(s1.ok());
        auto r1 = s1.value()->generate(shared_prompt);
        REQUIRE(r1.ok());
        CHECK_FALSE(r1->stage_timings.has_value());

        SessionOptions so2;
        so2.sampling = SamplingConfig::greedy(3);
        auto s2 = engine.create_session(model.value(), so2);
        REQUIRE(s2.ok());
        auto r2 = s2.value()->generate(shared_prompt + " rho sigma");
        REQUIRE(r2.ok());
        CHECK_FALSE(r2->stage_timings.has_value());
        REQUIRE(r2->effective_execution.has_value());
        REQUIRE(r2->effective_execution->cache.logical_cache_hit_tokens.has_value());
        CHECK(*r2->effective_execution->cache.logical_cache_hit_tokens > 0u);
        CHECK_FALSE(r2->effective_execution->cache.device_prefix_reused_tokens.has_value());
        CHECK_FALSE(r2->effective_execution->cache.host_l2_restored_tokens.has_value());
        CHECK_FALSE(r2->effective_execution->cache.physical_reused_tokens().has_value());

        engine.telemetry().flush();
        for (const auto& line : sink->lines()) {
            const auto ev = json::parse(line);
            REQUIRE(ev.ok());
            CHECK(ev->find("event_type")->as_string() != "inference.stage.timings");
        }
    }

    // 2. Opt-in stage profiling ON (per-request): measures active stages, marks overlaps, leaves unmeasured null.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        eo.scheduling.enabled = true;
        Engine engine(std::move(eo));
        MockBackendOptions mopt;
        mopt.token_logits = true;
        REQUIRE(engine.register_backend(make_mock_backend(mopt)).ok());
        auto model = engine.load_model(kMockBackendName, ModelLoadOptions{"mock:tiny", "cpu:0"});
        REQUIRE(model.ok());
        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4);
        auto sess = engine.create_session(model.value(), so);
        REQUIRE(sess.ok());

        RequestOptions ro;
        ro.enable_stage_profiling = true;
        auto r = sess.value()->generate("stage timing profiling probe", {}, std::nullopt, ro);
        REQUIRE(r.ok());
        REQUIRE(r->stage_timings.has_value());
        CHECK(r->stage_timings->enabled);
        CHECK(r->stage_timings->queue_wait_ms.has_value());
        CHECK(r->stage_timings->prefill_ms.has_value());
        CHECK(r->stage_timings->sampling_ms.has_value());
        CHECK(r->stage_timings->callback_gate_wait_ms.has_value());
        CHECK(r->stage_timings->total_elapsed_ms.has_value());
        CHECK(r->stage_timings->stages_may_overlap);
        CHECK_FALSE(r->stage_timings->l2_restore_ms.has_value());
        CHECK_FALSE(r->stage_timings->draft_ms.has_value());
        CHECK_FALSE(r->stage_timings->verify_ms.has_value());
        CHECK_FALSE(r->stage_timings->rollback_ms.has_value());

        engine.telemetry().flush();
        bool saw_stage_event = false;
        bool saw_completed_after_stage = false;
        for (const auto& line : sink->lines()) {
            const auto ev = json::parse(line);
            REQUIRE(ev.ok());
            check_envelope(ev.value());
            const std::string type = ev->find("event_type")->as_string();
            if (type == "inference.stage.timings") {
                saw_stage_event = true;
                const auto* attrs = ev->find("attributes");
                REQUIRE(attrs != nullptr);
                CHECK(attrs->find("enabled")->as_bool());
                CHECK(attrs->find("queue_wait_ms")->is_number());
                CHECK(attrs->find("sampling_ms")->is_number());
                CHECK(attrs->find("callback_gate_wait_ms")->is_number());
                CHECK(attrs->find("total_elapsed_ms")->is_number());
                CHECK(attrs->find("draft_ms")->is_null());
                CHECK(attrs->find("l2_restore_ms")->is_null());
                CHECK(attrs->find("stages_may_overlap")->as_bool());
            } else if (type == "request.completed" && saw_stage_event) {
                saw_completed_after_stage = true;
                CHECK(ev->find("attributes")->find("stage_timings") != nullptr);
                CHECK(ev->find("attributes")->find("effective_execution") != nullptr);
            }
        }
        CHECK(saw_stage_event);
        CHECK(saw_completed_after_stage);
    }

    // 3. Stage timing emission uses the bounded non-blocking queue and drops under pressure without stalling.
    {
        auto gate = std::make_shared<GateSink>();
        TelemetryOptions opts;
        opts.queue_capacity = 2;
        TelemetryBus bus(opts);
        bus.add_sink(gate);

        StageTimings st;
        st.enabled = true;
        st.prefill_ms = 2.0;
        st.total_elapsed_ms = 5.0;
        TelemetryContext ctx;
        ctx.session_id = "sess-stage-pressure";
        int accepted = 0;
        for (int i = 0; i < 64; ++i) {
            if (bus.emit_stage_timings(ctx, st, "mock")) ++accepted;
        }
        CHECK(bus.dropped_events() > 0u);
        CHECK(static_cast<std::uint64_t>(accepted) + bus.dropped_events() == 64u);
        gate->open();
        CHECK(bus.shutdown());
    }

    // 4. ScopedStageTimer accuracy when active and zero measurement when disabled (nullptr).
    {
        StageTimings disabled;
        for (int i = 0; i < 10000; ++i) {
            ScopedStageTimer noop(nullptr);
        }
        CHECK_FALSE(disabled.has_any_measurement());

        StageTimings active;
        active.enabled = true;
        {
            ScopedStageTimer timer(&active.prefill_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        REQUIRE(active.prefill_ms.has_value());
        CHECK(*active.prefill_ms >= 5.0);
        CHECK(*active.prefill_ms < 2000.0);
    }
}

TEST_CASE("NAT-08: gpu_memory_sample_attributes emits additive device telemetry fields cleanly") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    GpuDeviceInfo rx9060;
    rx9060.id = "gpu:0";
    rx9060.name = "AMD Radeon RX 9060 XT";
    rx9060.vendor = GpuVendor::amd;
    rx9060.vendor_id = 0x1002u;
    rx9060.device_id = 0x7590u;
    rx9060.dedicated_total = MemoryReading::known(16ull * kGiB);
    rx9060.dedicated_free = MemoryReading::known(15ull * kGiB);
    rx9060.shared_total = MemoryReading::known(24ull * kGiB);
    rx9060.is_integrated_uma = false;
    rx9060.backend_hint = "vulkan";
    rx9060.source = "dxgi";
    rx9060.pci_bus_id = "0000:03:00.0";
    rx9060.driver_name = "AMD proprietary driver";
    rx9060.driver_build = "25.10.2 (32.0.21025.1024)";
    rx9060.rebar = true;
    rx9060.shared_memory_bytes = 32768ull;
    rx9060.warp_size = 64u;
    rx9060.matrix_cores = "KHR_coopmat";
    rx9060.pcie.current_gen = 5u;
    rx9060.pcie.current_width = 16u;
    rx9060.pcie.max_gen = 5u;
    rx9060.pcie.max_width = 16u;
    rx9060.pcie.aspm_state = "off";
    rx9060.pcie.aspm_source = "windows_power_plan";

    VulkanMemoryHeapInfo heap0;
    heap0.heap_index = 0u;
    heap0.size_bytes = 16ull * kGiB;
    heap0.budget_bytes = MemoryReading::known(15ull * kGiB);
    heap0.usage_bytes = MemoryReading::known(512ull * kMiB);
    heap0.device_local = true;
    heap0.host_visible = true;
    heap0.host_coherent = true;
    rx9060.vulkan_heaps.push_back(heap0);

    const auto ggml_line = parse_ggml_device_line(
        "ggml_vulkan: 0 = AMD Radeon RX 9060 XT (AMD proprietary driver) | uma: 0 | rebar: 1 | "
        "fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | "
        "matrix cores: KHR_coopmat");
    REQUIRE(ggml_line.has_value());
    rx9060.ggml_device = *ggml_line;

    const json::Object attrs = gpu_memory_sample_attributes(rx9060);
    REQUIRE(attrs.find("driver_name") != nullptr);
    CHECK(attrs.find("driver_name")->as_string() == "AMD proprietary driver");
    REQUIRE(attrs.find("driver_build") != nullptr);
    CHECK(attrs.find("driver_build")->as_string() == "25.10.2 (32.0.21025.1024)");
    REQUIRE(attrs.find("pci_bus_id") != nullptr);
    CHECK(attrs.find("pci_bus_id")->as_string() == "0000:03:00.0");
    REQUIRE(attrs.find("rebar") != nullptr);
    CHECK(attrs.find("rebar")->as_bool());
    REQUIRE(attrs.find("shared_memory_bytes") != nullptr);
    CHECK(attrs.find("shared_memory_bytes")->as_uint() == 32768ull);
    REQUIRE(attrs.find("warp_size") != nullptr);
    CHECK(attrs.find("warp_size")->as_uint() == 64ull);
    REQUIRE(attrs.find("matrix_cores") != nullptr);
    CHECK(attrs.find("matrix_cores")->as_string() == "KHR_coopmat");

    const json::Value* pcie = attrs.find("pcie");
    REQUIRE(pcie != nullptr);
    REQUIRE(pcie->is_object());
    CHECK(pcie->find("current_gen")->as_uint() == 5ull);
    CHECK(pcie->find("current_width")->as_uint() == 16ull);
    CHECK(pcie->find("max_gen")->as_uint() == 5ull);
    CHECK(pcie->find("max_width")->as_uint() == 16ull);
    CHECK(pcie->find("aspm_state")->as_string() == "off");
    CHECK(pcie->find("aspm_source")->as_string() == "windows_power_plan");

    const json::Value* heaps = attrs.find("vulkan_heaps");
    REQUIRE(heaps != nullptr);
    REQUIRE(heaps->is_array());
    REQUIRE(heaps->as_array().size() == 1u);
    const json::Value& h0 = heaps->as_array()[0];
    CHECK(h0.find("heap_index")->as_uint() == 0ull);
    CHECK(h0.find("size_bytes")->as_uint() == 16ull * kGiB);
    CHECK(h0.find("budget_bytes")->as_uint() == 15ull * kGiB);
    CHECK(h0.find("usage_bytes")->as_uint() == 512ull * kMiB);
    CHECK(h0.find("device_local")->as_bool());
    CHECK(h0.find("host_visible")->as_bool());
    CHECK(h0.find("host_coherent")->as_bool());

    const json::Value* ggml = attrs.find("ggml_device");
    REQUIRE(ggml != nullptr);
    REQUIRE(ggml->is_object());
    CHECK(ggml->find("backend")->as_string() == "vulkan");
    CHECK(ggml->find("rebar")->as_bool());
    CHECK(ggml->find("matrix_cores")->as_string() == "KHR_coopmat");
    CHECK(ggml->find("shared_memory_bytes")->as_uint() == 32768ull);

    // Unset NAT-08 fields are omitted so existing minimal payloads stay compact.
    GpuDeviceInfo minimal;
    minimal.id = "gpu:1";
    minimal.name = "Minimal";
    const json::Object min_attrs = gpu_memory_sample_attributes(minimal);
    CHECK(min_attrs.find("driver_build") == nullptr);
    CHECK(min_attrs.find("rebar") == nullptr);
    CHECK(min_attrs.find("pcie") == nullptr);
    CHECK(min_attrs.find("vulkan_heaps") == nullptr);
    CHECK(min_attrs.find("ggml_device") == nullptr);
}
#if defined(SONDER_HAS_SAMPLER_CHAIN)
TEST_CASE("VIS-03: Sonder sampler emits inference.sampling.candidates with bounds and opt-in text") {
    auto run_sonder = [](TelemetryOptions telemetry, const SamplingConfig& sampling, bool use_chat = false,
                         int default_tokens = 4, std::function<void(const TokenChunk&)> on_chunk = {}) {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry = telemetry;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        eo.scheduling.enabled = false;
        Engine engine(std::move(eo));
        MockBackendOptions mo;
        mo.token_logits = true;
        mo.default_completion_tokens = default_tokens;
        REQUIRE(engine.register_backend(make_mock_backend(mo)).ok());
        auto model = engine.load_model(kMockBackendName, ModelLoadOptions{"mock:tiny", "cpu:0"});
        REQUIRE(model.ok());
        auto sess = engine.create_session(model.value(), {});
        REQUIRE(sess.ok());
        auto cb = [&](const TokenChunk& c) {
            if (on_chunk) on_chunk(c);
            return true;
        };
        // Named, not a braced temporary inside the ternary: GCC's -Wdangling-pointer flags the latter.
        const std::vector<ChatMessage> messages{{"user", "candidates probe"}};
        Result<GenerationResult> res = use_chat ? sess.value()->chat(messages, cb, sampling)
                                                : sess.value()->generate("candidates probe", cb, sampling);
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        engine.telemetry().flush();
        std::vector<json::Value> events;
        for (const auto& line : sink->lines()) {
            auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            check_envelope(parsed.value());
            events.push_back(std::move(parsed).value());
        }
        return std::pair{std::move(res).value(), std::move(events)};
    };

    auto filter_events = [](const std::vector<json::Value>& all, std::string_view type) {
        std::vector<json::Value> out;
        for (const auto& e : all) {
            if (e.find("event_type")->as_string() == type) out.push_back(e);
        }
        return out;
    };

    SUBCASE("default standard telemetry emits top-5 post-sampling candidates without text or logprob leak") {
        TelemetryOptions t;
        SamplingConfig sc = SamplingConfig::greedy(4);
        unsigned caller_logprob_chunks = 0;
        auto [res, events] = run_sonder(t, sc, false, 8, [&](const TokenChunk& c) {
            if (c.logprob.has_value() || !c.top_logprobs.empty()) ++caller_logprob_chunks;
        });
        CHECK(res.logprobs.empty());
        CHECK(caller_logprob_chunks == 0u);

        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 4u);
        REQUIRE(cands.size() == 4u);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto& ta = *tokens[i].find("attributes");
            const auto& ca = *cands[i].find("attributes");
            CHECK(cands[i].find("sampling")->find("level")->as_string() == "standard");
            CHECK(ta.find("unit")->as_string() == "token");
            CHECK(ta.find("probability_basis")->as_string() == "post_sampling");
            CHECK(ca.find("index")->as_int() == ta.find("index")->as_int());
            CHECK(ca.find("sampled_token_id")->as_int() == ta.find("token_id")->as_int());
            CHECK(ca.find("probability_basis")->as_string() == "post_sampling");
            const auto& list = ca.find("candidates")->as_array();
            REQUIRE(list.size() == 5u);
            // Greedy selects argmax: candidate[0] matches the sampled token and probability.
            CHECK(list[0].find("token_id")->as_int() == ta.find("token_id")->as_int());
            CHECK(list[0].find("probability")->as_double() ==
                  doctest::Approx(ta.find("probability")->as_double()).epsilon(1e-6));
            for (std::size_t j = 0; j < list.size(); ++j) {
                const double p = list[j].find("probability")->as_double();
                CHECK(p > 0.0);
                CHECK(p <= 1.0);
                CHECK(list[j].find("text") == nullptr);
                if (j > 0) {
                    CHECK(list[j - 1].find("probability")->as_double() >= p);
                }
            }
        }
    }

    SUBCASE("stochastic chain respects top_k truncation and logit_bias bans in candidates") {
        TelemetryOptions t;
        t.token_probability_top_k = 10;
        SamplingConfig sc;
        sc.max_tokens = 3;
        sc.temperature = 0.8f;
        sc.top_k = 3;
        sc.top_p = 1.0f;
        sc.min_p = 0.0f;
        sc.seed = 42;
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}};
        auto [res, events] = run_sonder(t, sc, false, 8);
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(cands.size() == 3u);
        for (const auto& ev : cands) {
            const auto& list = ev.find("attributes")->find("candidates")->as_array();
            REQUIRE(list.size() >= 1u);
            CHECK(list.size() <= 3u);
            double sum = 0.0;
            for (const auto& entry : list) {
                CHECK(entry.find("token_id")->as_int() != 0);
                sum += entry.find("probability")->as_double();
            }
            CHECK(sum == doctest::Approx(1.0).epsilon(1e-4));
        }
    }

    SUBCASE("stride, cap, top_k clamping, and capture_text opt-in") {
        TelemetryOptions t;
        t.token_probability_top_k = 50;  // clamped to kMaxTelemetryTopK (10)
        t.token_candidates_stride = 2;
        t.token_candidates_max_events = 0;
        t.capture_text = true;
        SamplingConfig sc = SamplingConfig::greedy(5);
        auto [res, events] = run_sonder(t, sc, false, 8);
        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 5u);
        REQUIRE(cands.size() == 3u);  // indices 0, 2, 4
        CHECK(cands[0].find("attributes")->find("index")->as_int() == 0);
        CHECK(cands[1].find("attributes")->find("index")->as_int() == 2);
        CHECK(cands[2].find("attributes")->find("index")->as_int() == 4);
        for (const auto& ev : cands) {
            const auto& list = ev.find("attributes")->find("candidates")->as_array();
            REQUIRE(list.size() == static_cast<std::size_t>(kMaxTelemetryTopK));
            for (const auto& entry : list) {
                REQUIRE(entry.find("text") != nullptr);
                CHECK_FALSE(entry.find("text")->as_string().empty());
            }
        }

        // Cap 1 and stride 0 (treated as 1) keeps only the first token's candidates.
        TelemetryOptions capped;
        capped.token_candidates_stride = 0;
        capped.token_candidates_max_events = 1;
        auto [res2, events2] = run_sonder(capped, SamplingConfig::greedy(4), false, 8);
        CHECK(filter_events(events2, "inference.token.generated").size() == 4u);
        const auto cands2 = filter_events(events2, "inference.sampling.candidates");
        REQUIRE(cands2.size() == 1u);
        CHECK(cands2[0].find("attributes")->find("index")->as_int() == 0);
    }

    SUBCASE("top_k 0 or negative disables candidates while keeping unit token probabilities") {
        for (const int k : {0, -5}) {
            CAPTURE(k);
            TelemetryOptions t;
            t.token_probability_top_k = k;
            auto [res, events] = run_sonder(t, SamplingConfig::greedy(3), false, 8);
            const auto tokens = filter_events(events, "inference.token.generated");
            REQUIRE(tokens.size() == 3u);
            for (const auto& ev : tokens) {
                CHECK(ev.find("attributes")->find("unit")->as_string() == "token");
                CHECK(ev.find("attributes")->find("probability") != nullptr);
            }
            CHECK(filter_events(events, "inference.sampling.candidates").empty());
        }
    }

    SUBCASE("metrics level omits both token and candidates events") {
        TelemetryOptions t;
        t.level = TelemetryLevel::metrics;
        auto [res, events] = run_sonder(t, SamplingConfig::greedy(3), false, 8);
        CHECK(filter_events(events, "inference.token.generated").empty());
        CHECK(filter_events(events, "inference.sampling.candidates").empty());
    }

    SUBCASE("Session::chat on token_logits backend emits inference.sampling.candidates") {
        TelemetryOptions t;
        t.token_probability_top_k = 3;
        auto [res, events] = run_sonder(t, SamplingConfig::greedy(3), /*use_chat=*/true, 8);
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(cands.size() == 3u);
        CHECK(cands[0].find("attributes")->find("candidates")->as_array().size() == 3u);
    }
}
#endif
