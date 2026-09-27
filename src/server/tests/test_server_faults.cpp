// Resource faults in request handling must fail one request, never the
// process: a thread the OS refuses to start is answered with 503.
#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>

#include "server_test_support.hpp"
#include "src/test_hooks.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

TEST_CASE("chat: a refused disconnect-watcher thread is 503 overloaded, and the server keeps serving") {
    Fixture f;
    det::fail_watcher_spawns_for_test().store(1);
    const Reply refused = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)"));
    det::fail_watcher_spawns_for_test().store(0);
    REQUIRE_MESSAGE(refused.status == 503, refused.body);
    CHECK(refused.header("retry-after") == "1");
    const json::Value doc = refused.json();
    CHECK(doc.find("error")->find("code")->as_string() == "overloaded");
    // Nothing ran for the refused request.
    CHECK(f.of_type("request.started").empty());

    // The process survived and the refused request left nothing in flight.
    const Reply ok = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)"));
    CHECK(ok.status == 200);
    const Reply health = get(f.port, "/v1/sonder/health");
    CHECK(health.status == 200);
}

TEST_CASE("capacity: connections dribbling their request head cannot starve new clients (slowloris)") {
    auto o = Fixture::defaults();
    o.max_connections = 2;
    Fixture f(o);
    // Fill every slot with a connection that never finishes its head.
    Conn slow1(f.port);
    Conn slow2(f.port);
    slow1.send("GET /v1/sonder/health HTTP/1.1\r\nHost: 127.0.0.1\r\n");
    slow2.send("GET /v1/sonder/health HTTP/1.1\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    // A fresh, well-behaved client is served, not refused with 503.
    const Reply r = get(f.port, "/v1/sonder/health");
    CHECK_MESSAGE(r.status == 200, r.body);
    // The oldest slow connection was evicted with 408 well before the 10 s
    // read timeout.
    const Reply evicted = parse_reply(slow1.read_all(std::chrono::milliseconds(3000)));
    CHECK(evicted.status == 408);
    CHECK(slow1.closed());
    // A second fresh client evicts the other slow one.
    CHECK(get(f.port, "/v1/sonder/health").status == 200);
}

TEST_CASE("capacity: a connection past its request head is never evicted") {
    auto o = Fixture::defaults();
    o.max_connections = 1;
    Fixture f(o);
    Conn stream(f.port);
    stream.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(stream.read_until(acc, "retry: 2000"));
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    CHECK(get(f.port, "/v1/sonder/health").status == 503);
    // The stream is still open and delivering.
    CHECK_FALSE(stream.closed());
}
