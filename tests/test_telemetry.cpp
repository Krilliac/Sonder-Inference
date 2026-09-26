#include <doctest/doctest.h>

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>

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
