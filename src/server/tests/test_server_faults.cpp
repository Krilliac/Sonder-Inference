// Resource faults in request handling must fail one request, never the
// process: a thread the OS refuses to start is answered with 503.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
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

namespace {

// A log sink that counts lines and, once armed, blocks every call on a gate.
// Each wait is bounded (60 s backstop) so a test bug cannot hang the suite.
struct GatedLog {
    std::mutex mu;
    std::condition_variable cv;
    bool armed = false;
    bool open = true;
    std::atomic<int> lines{0};
    std::atomic<int> blocked{0};

    void write() {
        ++lines;
        std::unique_lock<std::mutex> lock(mu);
        if (!armed) return;
        ++blocked;
        cv.wait_for(lock, std::chrono::seconds(60), [this] { return open; });
    }
    void arm() {
        std::lock_guard<std::mutex> lock(mu);
        armed = true;
        open = false;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mu);
            open = true;
        }
        cv.notify_all();
    }
};

// Opens the gate on every exit path (declared after the Fixture, so it runs
// first on unwind and lets a blocked server drain instead of hanging).
struct GateGuard {
    GatedLog& gate;
    ~GateGuard() { gate.release(); }
};

// Like get(), but every read is bounded by `timeout` (get() waits up to 15 s).
Reply bounded_get(std::uint16_t port, const std::string& target, std::chrono::milliseconds timeout) {
    Conn c(port);
    c.send(build_request("GET", target, port));
    Reply r = parse_reply(c.read_all(timeout));
    r.complete = c.closed();
    return r;
}

}  // namespace

TEST_CASE("log sink: a blocked log sink never blocks health, the accept loop or shutdown") {
    GatedLog gate;
    auto o = Fixture::defaults();
    o.max_connections = 4;
    o.shutdown_grace = std::chrono::milliseconds(500);
    o.log = [&gate](const std::string&) { gate.write(); };
    Fixture f(o);
    GateGuard guard{gate};

    gate.arm();

    const int requests = 3 * static_cast<int>(o.max_connections);
    for (int i = 0; i < requests; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const Reply r = bounded_get(f.port, "/v1/sonder/health", std::chrono::milliseconds(2000));
        const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
        INFO("request " << i << " of " << requests << " took " << took.count() << " ms");
        // Stop at the first failure: each failing request costs the full 2 s.
        REQUIRE(r.status == 200);
        REQUIRE(r.complete);
        REQUIRE(took < std::chrono::milliseconds(2000));
    }

    // The gate was really exercised: the sink was reached and is still blocked.
    CHECK(eventually([&] { return gate.blocked.load() >= 1; }, std::chrono::milliseconds(2000)));

    // Shutdown with the sink still blocked: run stop() on a helper so a hang is
    // a failed bounded wait, not a stuck test.
    const auto budget = o.shutdown_grace + std::chrono::seconds(2);
    std::promise<void> stopped;
    auto stopped_future = stopped.get_future();
    std::thread stopper([&] {
        f.server->stop();
        stopped.set_value();
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool in_time = stopped_future.wait_for(budget) == std::future_status::ready;
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
    gate.release();  // let a blocked stop() finish so the helper can be joined
    stopper.join();
    CHECK_MESSAGE(in_time, "stop() took " << stop_ms.count() << " ms with a blocked log sink (budget "
                                          << std::chrono::duration_cast<std::chrono::milliseconds>(budget).count()
                                          << " ms)");
}
