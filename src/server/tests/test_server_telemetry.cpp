// Live telemetry over HTTP: discovery, SSE and NDJSON framing, heartbeats,
// resume (same instance, unknown instance, older than the window), since=now,
// the subscriber cap, and a stalled subscriber that loses events while chat
// requests keep completing.
#include <doctest/doctest.h>

#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

struct SseEvent {
    std::string id;
    std::string data;
    bool has_event_field = false;
};

// Parses complete SSE events (blocks ending in a blank line) from `text`;
// comment lines are collected separately.
std::vector<SseEvent> parse_sse(const std::string& text, std::vector<std::string>* comments = nullptr) {
    std::vector<SseEvent> out;
    std::size_t pos = 0;
    for (;;) {
        const auto end = text.find("\n\n", pos);
        if (end == std::string::npos) break;
        const std::string block = text.substr(pos, end - pos);
        pos = end + 2;
        SseEvent ev;
        bool any_data = false;
        std::size_t lp = 0;
        while (lp <= block.size()) {
            const auto le = block.find('\n', lp);
            const std::string line = block.substr(lp, le == std::string::npos ? std::string::npos : le - lp);
            if (line.rfind(":", 0) == 0) {
                if (comments) comments->push_back(line);
            } else if (line.rfind("id: ", 0) == 0) {
                ev.id = line.substr(4);
            } else if (line.rfind("data: ", 0) == 0) {
                ev.data = line.substr(6);
                any_data = true;
            } else if (line.rfind("event:", 0) == 0) {
                ev.has_event_field = true;
            }
            if (le == std::string::npos) break;
            lp = le + 1;
        }
        if (any_data) out.push_back(ev);
    }
    return out;
}

// Body of a streaming response (after the head).
std::string body_of(const std::string& raw) {
    const auto p = raw.find("\r\n\r\n");
    return p == std::string::npos ? std::string() : raw.substr(p + 4);
}

}  // namespace

TEST_CASE("discovery: producer document per contract 5.1") {
    Fixture f;
    f.envelopes();  // startup events have reached the hub
    const Reply r = get(f.port, "/.well-known/sonder-telemetry");
    REQUIRE(r.status == 200);
    CHECK(r.header("content-type") == "application/json");
    const json::Value d = r.json();
    CHECK(d.find("schema")->as_string() == "sonder.telemetry.producer/1");
    const json::Value* p = d.find("producer");
    REQUIRE(p != nullptr);
    CHECK(p->find("name")->as_string() == "sonder-inference");
    CHECK_FALSE(p->find("version")->as_string().empty());
    CHECK_FALSE(p->find("node_id")->as_string().empty());
    CHECK(p->find("instance_id")->as_string() == f.server->instance_id());
    CHECK(p->find("role")->as_string() == "inference");
    CHECK(p->find("synthetic")->as_bool());
    CHECK(d.find("event_schema")->as_string() == "sonder.observatory.event/1");
    const auto& streams = d.find("streams")->as_array();
    REQUIRE(streams.size() == 2);
    CHECK(streams[0].find("transport")->as_string() == "sse");
    CHECK(streams[0].find("url")->as_string() == "/v1/telemetry/sse");
    CHECK(streams[1].find("transport")->as_string() == "ndjson");
    CHECK(streams[1].find("url")->as_string() == "/v1/telemetry/ndjson");
    const json::Value* resume = d.find("resume");
    CHECK(resume->find("header")->as_string() == "Last-Event-ID");
    CHECK(resume->find("query")->as_string() == "last_event_id");
    CHECK(resume->find("retained_events")->as_int() > 0);
    CHECK(resume->find("oldest_sequence")->as_int() == 0);
    CHECK(resume->find("next_sequence")->as_int() > 0);
    CHECK_FALSE(d.find("auth")->find("required")->as_bool());
    CHECK(d.find("auth")->find("schemes")->as_array()[0].as_string() == "bearer");
    CHECK(d.find("clock")->find("mono_ns")->as_string() == "host-monotonic");
    CHECK(d.find("sampling_level")->as_string() == "standard");
    CHECK(d.find("text_capture")->as_string() == "off");
    CHECK(d.find("links")->find("health")->as_string() == "/v1/sonder/health");
    CHECK(d.find("links")->find("identity")->as_string() == "/v1/sonder/identity");
    CHECK(d.find("links")->find("models")->as_string() == "/v1/models");
    CHECK(d.find("vocabularies")->find("sonder.inference.events")->as_int() == 1);

    // Envelopes carry the same instance id, the role and the synthetic flag.
    const auto started = f.of_type("engine.started");
    REQUIRE(started.size() == 1);
    CHECK(started[0].find("producer")->find("instance_id")->as_string() == f.server->instance_id());
    const json::Value* server_attr = started[0].find("attributes")->find("server");
    REQUIRE(server_attr != nullptr);
    CHECK(server_attr->find("host")->as_string() == "127.0.0.1");
    CHECK(server_attr->find("port")->as_int() == f.port);
    CHECK(server_attr->find("api_version")->as_int() == 1);
}

TEST_CASE("sse: retry line, id == event_id, one envelope per data line, no event field") {
    Fixture f;
    Conn c(f.port);
    c.send(build_request("GET", "/v1/telemetry/sse", f.port, {{"Accept", "text/event-stream"}}));
    std::string acc;
    REQUIRE(c.read_until(acc, "engine.started"));
    const Reply head = parse_reply(acc);
    CHECK(head.status == 200);
    CHECK(head.header("content-type") == "text/event-stream; charset=utf-8");
    CHECK(head.header("x-sonder-inference-api") == "1");
    CHECK(head.header("cache-control") == "no-store");
    CHECK_FALSE(head.has_header("content-length"));
    const std::string body = body_of(acc);
    CHECK(body.rfind("retry: 2000\n\n", 0) == 0);

    // A chat request appears live on the stream.
    const Reply chat = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":3)"));
    REQUIRE(chat.status == 200);
    const std::string request_id = chat.json().find("sonder")->find("request_id")->as_string();
    REQUIRE(c.read_until(acc, "\"request.completed\""));
    c.read_until(acc, "\n\n", std::chrono::milliseconds(200));
    const auto events = parse_sse(body_of(acc));
    REQUIRE(events.size() > 3);
    std::uint64_t expected_seq = 0;
    bool saw_chat = false;
    for (const auto& ev : events) {
        CHECK_FALSE(ev.has_event_field);
        auto env = json::parse(ev.data);
        REQUIRE(env.ok());
        CHECK(env.value().find("event_id")->as_string() == ev.id);
        CHECK(env.value().find("sequence")->as_int() == static_cast<std::int64_t>(expected_seq));
        ++expected_seq;
        const json::Value* rid = env.value().find("request_id");
        if (rid != nullptr && rid->is_string() && rid->as_string() == request_id) saw_chat = true;
    }
    CHECK(saw_chat);
}

TEST_CASE("ndjson: one envelope per line, format selection by path, query and Accept") {
    Fixture f;
    for (const auto& [target, accept] : std::vector<std::pair<std::string, std::string>>{
             {"/v1/telemetry/ndjson", ""},
             {"/v1/telemetry?format=ndjson", ""},
             {"/v1/telemetry", "application/x-ndjson"}}) {
        CAPTURE(target);
        Conn c(f.port);
        std::vector<std::pair<std::string, std::string>> h;
        if (!accept.empty()) h.emplace_back("Accept", accept);
        c.send(build_request("GET", target, f.port, h));
        std::string acc;
        REQUIRE(c.read_until(acc, "scheduler.configured"));
        c.read_until(acc, "\n", std::chrono::milliseconds(100));
        const Reply head = parse_reply(acc);
        CHECK(head.header("content-type") == "application/x-ndjson");
        std::string body = body_of(acc);
        body = body.substr(0, body.rfind('\n'));
        std::size_t pos = 0;
        int lines = 0;
        while (pos < body.size()) {
            const auto nl = body.find('\n', pos);
            const std::string line = body.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
            if (!line.empty()) {
                auto env = json::parse(line);
                REQUIRE_MESSAGE(env.ok(), line);
                CHECK(env.value().find("schema")->as_string() == "sonder.observatory.event/1");
                ++lines;
            }
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        CHECK(lines >= 2);
        CHECK(body.find("retry:") == std::string::npos);
    }
    // ?format=sse on the generic path, and a bad format.
    Conn sse(f.port);
    sse.send(build_request("GET", "/v1/telemetry?format=sse", f.port, {{"Accept", "application/x-ndjson"}}));
    std::string acc;
    REQUIRE(sse.read_until(acc, "retry: 2000"));
    CHECK(get(f.port, "/v1/telemetry?format=xml").status == 400);
}

TEST_CASE("heartbeats: SSE comment and NDJSON blank line while idle") {
    auto o = Fixture::defaults();
    o.heartbeat_interval = std::chrono::milliseconds(100);
    Fixture f(o);
    Conn sse(f.port);
    sse.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(sse.read_until(acc, ": keepalive\n\n", std::chrono::milliseconds(3000)));
    Conn nd(f.port);
    nd.send(build_request("GET", "/v1/telemetry/ndjson?since=now", f.port));
    std::string nacc;
    REQUIRE(nd.read_until(nacc, "\r\n\r\n"));
    const std::size_t head_end = nacc.find("\r\n\r\n") + 4;
    REQUIRE(nd.read_until(nacc, "\n", std::chrono::milliseconds(3000)));
    CHECK(eventually([&] {
        nd.read_until(nacc, "\n\n", std::chrono::milliseconds(300));
        return nacc.size() > head_end && nacc.substr(head_end).find_first_not_of('\n') == std::string::npos;
    }));
}

TEST_CASE("resume: Last-Event-ID from the same instance, unknown instance, too old, and since=now") {
    auto o = Fixture::defaults();
    o.telemetry_buffer = 16;
    Fixture f(o);
    const std::string instance = f.server->instance_id();
    // Generate enough events to overflow the 16-event window.
    for (int i = 0; i < 3; ++i) {
        REQUIRE(post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)")).status == 200);
    }
    f.envelopes();
    const json::Value disc = get(f.port, "/.well-known/sonder-telemetry").json();
    const std::int64_t oldest = disc.find("resume")->find("oldest_sequence")->as_int();
    const std::int64_t next = disc.find("resume")->find("next_sequence")->as_int();
    REQUIRE(oldest > 1);
    CHECK(disc.find("resume")->find("retained_events")->as_int() == 16);
    CHECK(next - oldest == 16);

    const auto first_ids = [&](const std::vector<std::pair<std::string, std::string>>& headers,
                               const std::string& target, std::string* comments_out) {
        Conn c(f.port);
        c.send(build_request("GET", target, f.port, headers));
        std::string acc;
        c.read_until(acc, "id: " + instance + "-" + std::to_string(next - 1) + "\n", std::chrono::milliseconds(3000));
        c.read_until(acc, "\n\n", std::chrono::milliseconds(200));
        std::vector<std::string> comments;
        const auto events = parse_sse(body_of(acc), &comments);
        if (comments_out) {
            for (const auto& cm : comments) *comments_out += cm + "\n";
        }
        std::vector<std::string> ids;
        for (const auto& e : events) ids.push_back(e.id);
        return ids;
    };
    // Same instance inside the window: resumes right after the id (header).
    const std::int64_t mid = oldest + 5;
    auto ids = first_ids({{"Last-Event-ID", instance + "-" + std::to_string(mid)}}, "/v1/telemetry/sse", nullptr);
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == instance + "-" + std::to_string(mid + 1));
    // The query parameter works too, and the header wins over it.
    ids = first_ids({}, "/v1/telemetry/sse?last_event_id=" + instance + "-" + std::to_string(mid + 2), nullptr);
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == instance + "-" + std::to_string(mid + 3));
    ids = first_ids({{"Last-Event-ID", instance + "-" + std::to_string(mid)}},
                    "/v1/telemetry/sse?last_event_id=" + instance + "-" + std::to_string(mid + 2), nullptr);
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == instance + "-" + std::to_string(mid + 1));
    // Unknown instance (producer restarted): the whole retained window.
    ids = first_ids({{"Last-Event-ID", "tel-0000000000000000-3"}}, "/v1/telemetry/sse", nullptr);
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == instance + "-" + std::to_string(oldest));
    // Older than the window: resume-gap comment, then the whole window.
    std::string comments;
    ids = first_ids({{"Last-Event-ID", instance + "-0"}}, "/v1/telemetry/sse", &comments);
    REQUIRE_FALSE(ids.empty());
    CHECK(ids.front() == instance + "-" + std::to_string(oldest));
    CHECK(comments.find(": resume-gap 1-" + std::to_string(oldest - 1)) != std::string::npos);
    // since=now: nothing replayed, only live events.
    Conn live(f.port);
    live.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(live.read_until(acc, "retry: 2000\n\n"));
    REQUIRE(post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)")).status == 200);
    REQUIRE(live.read_until(acc, "request.completed"));
    const auto live_events = parse_sse(body_of(acc));
    REQUIRE_FALSE(live_events.empty());
    CHECK(json::parse(live_events.front().data).value().find("sequence")->as_int() >= next);
}

TEST_CASE("subscriber cap: the ninth stream gets 429 overloaded") {
    Fixture f;
    std::vector<std::unique_ptr<Conn>> streams;
    for (int i = 0; i < 8; ++i) {
        streams.push_back(std::make_unique<Conn>(f.port));
        streams.back()->send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
        std::string acc;
        REQUIRE(streams.back()->read_until(acc, "retry: 2000"));
    }
    CHECK(get(f.port, "/v1/sonder/health").json().find("telemetry")->find("subscribers")->as_int() == 8);
    const Reply over = get(f.port, "/v1/telemetry/ndjson");
    CHECK(over.status == 429);
    CHECK(over.header("retry-after") == "1");
    CHECK(over.json().find("error")->find("code")->as_string() == "overloaded");
    CHECK(over.json().find("error")->find("type")->as_string() == "rate_limit_error");
    streams.front()->close();
    CHECK(eventually([&] {
        return get(f.port, "/v1/sonder/health").json().find("telemetry")->find("subscribers")->as_int() == 7;
    }));
}

TEST_CASE("backpressure: a stalled subscriber loses events (counted) while chats keep completing") {
    auto o = Fixture::defaults();
    o.subscriber_queue = 8;
    o.write_stall_timeout = std::chrono::milliseconds(20000);
    Fixture f(o);
    // Subscribe and never read: the socket buffers fill, the stream thread
    // blocks in send, and the hub drops from this subscriber's queue.
    Conn stalled(f.port);
    stalled.shrink_receive_buffer();
    stalled.send(build_request("GET", "/v1/telemetry/ndjson", f.port));
    const auto t0 = std::chrono::steady_clock::now();
    int completed = 0;
    for (int i = 0; i < 3000; ++i) {
        const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":16)"));
        if (r.status == 200) ++completed;
        const auto health = get(f.port, "/v1/sonder/health").json();
        if (health.find("telemetry")->find("subscriber_dropped_events")->as_int() > 0 && i > 5) break;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(completed > 5);
    CHECK(seconds < 60.0);
    const auto health = get(f.port, "/v1/sonder/health").json();
    CHECK(health.find("telemetry")->find("subscriber_dropped_events")->as_int() > 0);
    // Bus-level drops are a different counter and stay at zero.
    CHECK(health.find("telemetry")->find("dropped")->as_int() == 0);
    const auto disc = get(f.port, "/.well-known/sonder-telemetry").json();
    CHECK(disc.find("stats")->find("subscriber_dropped_events")->as_int() > 0);
    stalled.close();
}
