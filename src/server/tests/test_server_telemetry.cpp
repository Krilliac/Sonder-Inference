// Live telemetry over HTTP: discovery, SSE and NDJSON framing, heartbeats,
// resume (same instance, unknown instance, older than the window), since=now,
// the subscriber cap, and a stalled subscriber that loses events while chat
// requests keep completing.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "src/openai.hpp"

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

TEST_CASE("backpressure: a stalled SSE subscriber sees ': dropped N' and a matching sequence gap") {
    auto o = Fixture::defaults();
    o.subscriber_queue = 8;
    o.write_stall_timeout = std::chrono::milliseconds(20000);
    Fixture f(o);
    Conn stalled(f.port, 4096);
    stalled.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(stalled.read_until(acc, "retry: 2000"));
    // Something is delivered before the reader stalls.
    REQUIRE(post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":1)")).status == 200);
    REQUIRE(stalled.read_until(acc, "\ndata: "));
    const auto dropped_so_far = [&] {
        return get(f.port, "/v1/sonder/health").json().find("telemetry")->find("subscriber_dropped_events")->as_int();
    };
    // Losses that happen from here on fall between delivered events (flush
    // first: the bus reaches the hub asynchronously).
    f.server->engine()->telemetry().flush();
    const std::int64_t baseline = dropped_so_far();
    for (int i = 0; i < 3000; ++i) {
        (void)post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":16)"));
        if (dropped_so_far() > baseline && i > 5) break;
    }
    REQUIRE(dropped_so_far() > baseline);
    // Now read: the per-subscriber loss is reported in-band as a comment,
    // never as a telemetry.dropped event on the shared stream.
    const std::size_t mark = acc.size();
    REQUIRE_MESSAGE(eventually([&] {
        std::string more;
        (void)stalled.read_until(more, ": dropped ", std::chrono::milliseconds(200));
        acc += more;
        return acc.find(": dropped ", mark) != std::string::npos;
    }, std::chrono::milliseconds(20000)),
                    "stream closed: " << stalled.closed() << ", bytes read: " << acc.size());
    const std::size_t at = acc.rfind(": dropped ");
    // Read on until an event after the (last) dropped comment is complete.
    REQUIRE(eventually([&] {
        std::string more;
        (void)stalled.read_until(more, "\n\n", std::chrono::milliseconds(200));
        acc += more;
        const std::size_t next = acc.find("id: ", at);
        return next != std::string::npos && acc.find("\n\n", next) != std::string::npos;
    }, std::chrono::milliseconds(10000)));
    // Walk the stream: every sequence gap is announced by ': dropped N'
    // comments that add up to exactly the gap, and there are no other gaps.
    std::uint64_t pending = 0;
    std::optional<std::uint64_t> last;
    int announced_gaps = 0;
    std::size_t pos = 0;
    while (pos < acc.size()) {
        const std::size_t eol = acc.find('\n', pos);
        if (eol == std::string::npos) break;
        const std::string line = acc.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.rfind(": dropped ", 0) == 0) {
            pending += std::stoull(line.substr(10));
        } else if (line.rfind("id: ", 0) == 0) {
            const std::uint64_t seq = std::stoull(line.substr(line.rfind('-') + 1));
            if (last) {
                CHECK(seq - *last - 1 == pending);
                if (pending > 0) ++announced_gaps;
            }
            last = seq;
            pending = 0;
        }
    }
    CHECK(announced_gaps > 0);
    CHECK(acc.find("\"telemetry.dropped\"") == std::string::npos);
    stalled.close();
}

TEST_CASE("drain: a chat that arrives after shutdown began gets 503 not_ready (nothing executed)") {
    Fixture f;
    const std::string body = chat_body(R"(,"max_tokens":2)");
    // The head is read before shutdown; the body arrives during the drain.
    Conn c(f.port);
    c.send("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
           "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::thread stopper([&] { f.server->stop(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    c.send(body);
    const Reply r = parse_reply(c.read_all(std::chrono::milliseconds(5000)));
    stopper.join();
    CHECK(r.status == 503);
    const json::Value doc = r.json();
    const json::Value* err = doc.find("error");
    REQUIRE(err != nullptr);
    CHECK(err->find("code")->as_string() == "not_ready");
    CHECK(err->find("message")->as_string().find("safe to send elsewhere") != std::string::npos);
    CHECK(f.of_type("request.queued").empty());
}

TEST_CASE("timings_json computes prompt_per_second from uncached tokens and omits rate on full cache hit") {
    // 1. 99% prefix cache hit (llamacpp / llamaserver with backend_timings):
    //    1000 total prompt tokens, 990 cached, 10 uncached evaluated in 5 ms -> 2000 tok/s (not 200,000!).
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 1000;
        r.stats.cached_tokens = 990;
        r.stats.completion_tokens = 20;
        r.stats.prompt_eval_ns = 5'000'000;  // 5.0 ms
        r.stats.eval_ns = 200'000'000;       // 200.0 ms
        si::BackendTimings bt;
        bt.prompt_n = 10;
        bt.cache_n = 990;
        bt.prompt_ms = 5.0;
        r.stats.backend_timings = bt;

        const auto usage = json::Value(det::usage_json(r));
        const auto timings = json::Value(det::timings_json(r));
        CHECK(usage.find("prompt_tokens")->as_uint() == 1000u);
        CHECK(usage.find("prompt_tokens_details")->find("cached_tokens")->as_uint() == 990u);
        CHECK(timings.find("prompt_n")->as_uint() == 10u);
        CHECK(timings.find("cache_n")->as_uint() == 990u);
        CHECK(timings.find("prompt_ms")->as_double() == doctest::Approx(5.0));
        REQUIRE(timings.find("prompt_per_second") != nullptr);
        CHECK(timings.find("prompt_per_second")->as_double() == doctest::Approx(2000.0));
        REQUIRE(timings.find("predicted_per_second") != nullptr);
        CHECK(timings.find("predicted_per_second")->as_double() == doctest::Approx(100.0));
    }

    // 2. 100% prefix cache hit (0 uncached tokens): prompt_n == 0 and prompt_per_second is omitted.
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 1000;
        r.stats.cached_tokens = 1000;
        r.stats.prompt_eval_ns = 500'000;  // 0.5 ms overhead
        si::BackendTimings bt;
        bt.prompt_n = 0;
        bt.cache_n = 1000;
        r.stats.backend_timings = bt;

        const auto timings = json::Value(det::timings_json(r));
        CHECK(timings.find("prompt_n")->as_uint() == 0u);
        CHECK(timings.find("cache_n")->as_uint() == 1000u);
        CHECK(timings.find("prompt_ms")->as_double() == doctest::Approx(0.5));
        CHECK(timings.find("prompt_per_second") == nullptr);
    }

    // 3. Partial and full cache hits without backend_timings (e.g. Ollama / checkpoint restore).
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 500;
        r.stats.cached_tokens = 450;
        r.stats.prompt_eval_ns = 25'000'000;  // 25.0 ms

        const auto timings = json::Value(det::timings_json(r));
        CHECK(timings.find("prompt_n")->as_uint() == 50u);
        CHECK(timings.find("cache_n")->as_uint() == 450u);
        REQUIRE(timings.find("prompt_per_second") != nullptr);
        CHECK(timings.find("prompt_per_second")->as_double() == doctest::Approx(2000.0));

        // 100% cached and cached > prompt_tokens clamp cleanly to 0 without underflow.
        r.stats.cached_tokens = 500;
        const auto full_hit = json::Value(det::timings_json(r));
        CHECK(full_hit.find("prompt_n")->as_uint() == 0u);
        CHECK(full_hit.find("prompt_per_second") == nullptr);

        r.stats.cached_tokens = 750;
        const auto clamped = json::Value(det::timings_json(r));
        CHECK(clamped.find("prompt_n")->as_uint() == 0u);
        CHECK(clamped.find("prompt_per_second") == nullptr);
    }

    // 4. Cache miss (cached_tokens == 0 or unset) preserves full prompt_tokens and rate.
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 200;
        r.stats.cached_tokens = 0;
        r.stats.prompt_eval_ns = 100'000'000;  // 100.0 ms
        const auto miss = json::Value(det::timings_json(r));
        CHECK(miss.find("prompt_n")->as_uint() == 200u);
        CHECK(miss.find("cache_n")->as_uint() == 0u);
        REQUIRE(miss.find("prompt_per_second") != nullptr);
        CHECK(miss.find("prompt_per_second")->as_double() == doctest::Approx(2000.0));

        r.stats.cached_tokens.reset();
        const auto unreported = json::Value(det::timings_json(r));
        CHECK(unreported.find("prompt_n")->as_uint() == 200u);
        CHECK(unreported.find("cache_n") == nullptr);
        REQUIRE(unreported.find("prompt_per_second") != nullptr);
        CHECK(unreported.find("prompt_per_second")->as_double() == doctest::Approx(2000.0));
    }

    // 5. backend_timings.cache_n acts as fallback when cached_tokens is unset, while existing cached_tokens wins.
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 40;
        r.stats.prompt_eval_ns = 5'000'000;  // 5.0 ms
        si::BackendTimings bt;
        bt.cache_n = 30;
        r.stats.backend_timings = bt;

        const auto fallback = json::Value(det::timings_json(r));
        CHECK(fallback.find("cache_n")->as_uint() == 30u);
        CHECK(fallback.find("prompt_n")->as_uint() == 10u);
        REQUIRE(fallback.find("prompt_per_second") != nullptr);
        CHECK(fallback.find("prompt_per_second")->as_double() == doctest::Approx(2000.0));

        r.stats.cached_tokens = 0;
        bt.prompt_n = 20;
        bt.cache_n = 19;
        r.stats.backend_timings = bt;
        r.stats.prompt_eval_ns = 4'000'000;  // 4.0 ms
        const auto precedence = json::Value(det::timings_json(r));
        CHECK(precedence.find("cache_n")->as_uint() == 0u);
        CHECK(precedence.find("prompt_n")->as_uint() == 20u);
        REQUIRE(precedence.find("prompt_per_second") != nullptr);
        CHECK(precedence.find("prompt_per_second")->as_double() == doctest::Approx(5000.0));
    }

    // 6. Unmeasured prompt_eval_ns omits prompt_ms and prompt_per_second.
    {
        si::GenerationResult r;
        r.stats.prompt_tokens = 128;
        r.stats.cached_tokens = 64;
        r.stats.prompt_eval_ns = 0;
        const auto unmeasured = json::Value(det::timings_json(r));
        CHECK(unmeasured.find("prompt_n")->as_uint() == 64u);
        CHECK(unmeasured.find("cache_n")->as_uint() == 64u);
        CHECK(unmeasured.find("prompt_ms") == nullptr);
        CHECK(unmeasured.find("prompt_per_second") == nullptr);
    }
}

TEST_CASE("embeddings: HTTP correlation headers emit request.queued, request.started and terminal telemetry") {
    Fixture f;
    const Reply r = post(f.port, "/v1/embeddings", R"({"input":["alpha","beta"]})",
                         {{"X-Sonder-Run-Id", "embed-run-42"},
                          {"X-Sonder-Parent-Request-Id", "turn-E42"},
                          {"X-Sonder-Agent-Id", "agent-embed"},
                          {"X-Sonder-Task-Id", "task-embed"},
                          {"X-Sonder-Workload", "owner_orchestrator"},
                          {"X-Sonder-Priority", "subagent"}});
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const std::string request_id = r.json().find("sonder")->find("request_id")->as_string();
    REQUIRE_FALSE(request_id.empty());

    std::vector<std::string> lifecycle;
    for (const auto& e : f.envelopes()) {
        const json::Value* rid = e.find("request_id");
        if (rid == nullptr || !rid->is_string() || rid->as_string() != request_id) continue;
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        lifecycle.push_back(type);
        CHECK(e.find("run_id")->as_string() == "embed-run-42");
        CHECK(e.find("agent_id")->as_string() == "agent-embed");
        CHECK(e.find("task_id")->as_string() == "task-embed");
        const json::Value* attrs = e.find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("kind")->as_string() == "embed");
        REQUIRE(attrs->find("parent_request_id") != nullptr);
        CHECK(attrs->find("parent_request_id")->as_string() == "turn-E42");
        REQUIRE(attrs->find("priority_class") != nullptr);
        CHECK(attrs->find("priority_class")->as_string() == "subagent");
        REQUIRE(attrs->find("admission") != nullptr);
        CHECK(attrs->find("admission")->find("priority")->as_string() == "subagent");
        if (type == "request.started") {
            CHECK(attrs->find("inputs")->as_uint() == 2u);
            CHECK_FALSE(attrs->find("scheduled")->as_bool());
        } else if (type == "request.completed") {
            CHECK(attrs->find("outcome")->as_string() == "completed");
            CHECK(attrs->find("inputs")->as_uint() == 2u);
            CHECK(attrs->find("prompt_tokens")->as_uint() > 0u);
            CHECK(attrs->find("total_ms")->as_double() >= 0.0);
        }
    }
    CHECK(lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});

    // Plain /v1/embeddings without headers still emits request.started and omits parent_request_id.
    const Reply plain = post(f.port, "/v1/embeddings", R"({"input":"plain"})");
    REQUIRE(plain.status == 200);
    const std::string plain_id = plain.json().find("sonder")->find("request_id")->as_string();
    std::vector<std::string> plain_lifecycle;
    for (const auto& e : f.envelopes()) {
        const json::Value* rid = e.find("request_id");
        if (rid == nullptr || !rid->is_string() || rid->as_string() != plain_id) continue;
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        plain_lifecycle.push_back(type);
        CHECK(e.find("attributes")->find("parent_request_id") == nullptr);
    }
    CHECK(plain_lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});

    // Unsupported backend (501) still emits request.queued -> request.started -> request.failed with parent_request_id.
    struct NoEmbedModel final : si::BackendModel {
        si::ModelDescriptor desc;
        NoEmbedModel() {
            desc.name = "noembed:tiny";
            desc.backend = "noembed";
            desc.context_length = 4096;
        }
        const si::ModelDescriptor& descriptor() const override { return desc; }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                               const si::TokenCallback&) override {
            return si::Status(si::ErrorCode::unsupported, "unused");
        }
    };
    struct NoEmbedBackend final : si::Backend {
        std::shared_ptr<NoEmbedModel> model = std::make_shared<NoEmbedModel>();
        std::string name() const override { return "noembed"; }
        std::string description() const override { return "no-embed test backend"; }
        si::BackendCapabilities capabilities() const override { return {}; }
        si::Result<std::string> probe() override { return std::string("1"); }
        si::Result<std::vector<si::ModelDescriptor>> list_models() override {
            return std::vector<si::ModelDescriptor>{model->desc};
        }
        si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
            return std::shared_ptr<si::BackendModel>(model);
        }
    };
    auto opts = Fixture::defaults();
    opts.models = {"noembed:tiny"};
    opts.backend.backend.clear();
    opts.backend_instance = std::make_shared<NoEmbedBackend>();
    Fixture noembed(opts);
    const Reply fail = post(noembed.port, "/v1/embeddings", R"({"input":"x"})",
                            {{"X-Sonder-Run-Id", "embed-run-501"},
                             {"X-Sonder-Parent-Request-Id", "turn-E501"}});
    CHECK(fail.status == 501);
    std::vector<std::string> fail_lifecycle;
    for (const auto& e : noembed.envelopes()) {
        const json::Value* run = e.find("run_id");
        if (run == nullptr || !run->is_string() || run->as_string() != "embed-run-501") continue;
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        fail_lifecycle.push_back(type);
        const json::Value* attrs = e.find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("kind")->as_string() == "embed");
        REQUIRE(attrs->find("parent_request_id") != nullptr);
        CHECK(attrs->find("parent_request_id")->as_string() == "turn-E501");
        if (type == "request.failed") {
            CHECK(attrs->find("error_code")->as_string() == "unsupported");
        }
    }
    CHECK(fail_lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.failed"});
}
