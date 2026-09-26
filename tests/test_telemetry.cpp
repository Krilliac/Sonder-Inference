#include <doctest/doctest.h>

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
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

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = false;
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
