// Resource faults in request handling must fail one request, never the
// process: a thread the OS refuses to start is answered with 503.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
    std::atomic<int> drop_lines{0};  // "lines dropped" reports among `lines`
    std::atomic<int> waiting{0};     // calls waiting at the closed gate right now
    std::atomic<int> inside{0};      // calls in progress

    void write(const std::string& text) {
        ++inside;
        ++lines;
        if (text.find("lines dropped") != std::string::npos) ++drop_lines;
        {
            std::unique_lock<std::mutex> lock(mu);
            if (armed) {
                ++blocked;
                if (!open) {
                    ++waiting;
                    cv.wait_for(lock, std::chrono::seconds(60), [this] { return open; });
                    --waiting;
                }
            }
        }
        --inside;  // last access: the gate may be destroyed once this reads 0
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

// The additive `log` object of a health reply.
struct LogCounters {
    bool present = false;
    std::uint64_t queued = 0;
    std::uint64_t capacity = 0;
    std::uint64_t written = 0;
    std::uint64_t dropped = 0;
};

LogCounters log_counters(const Reply& health) {
    LogCounters c;
    const json::Value doc = health.json();
    const json::Value* counters = doc.find("log");
    if (counters == nullptr) return c;
    const json::Value* queued = counters->find("queued");
    const json::Value* capacity = counters->find("capacity");
    const json::Value* written = counters->find("written");
    const json::Value* dropped = counters->find("dropped");
    if (queued == nullptr || capacity == nullptr || written == nullptr || dropped == nullptr) return c;
    c.present = true;
    c.queued = queued->as_uint();
    c.capacity = capacity->as_uint();
    c.written = written->as_uint();
    c.dropped = dropped->as_uint();
    return c;
}

}  // namespace

TEST_CASE("log sink: a blocked log sink never blocks health, the accept loop or shutdown") {
    constexpr std::uint64_t kLogCapacity = 8;
    const auto request_timeout = std::chrono::milliseconds(2000);
    GatedLog gate;
    auto o = Fixture::defaults();
    o.max_connections = 4;
    o.shutdown_grace = std::chrono::milliseconds(500);
    o.log_queue_capacity = static_cast<std::size_t>(kLogCapacity);
    o.log = [&gate](const std::string& text) { gate.write(text); };
    Fixture f(o);
    GateGuard guard{gate};

    gate.arm();

    // Every answered request produces exactly one access-log line, queued
    // before its connection closes; start() logs nothing with these options.
    std::uint64_t produced = 0;
    const int requests = 3 * static_cast<int>(o.max_connections);
    for (int i = 0; i < requests; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const Reply r = bounded_get(f.port, "/v1/sonder/health", request_timeout);
        const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
        INFO("request " << i << " of " << requests << " took " << took.count() << " ms");
        // Stop at the first failure: each failing request costs the full 2 s.
        REQUIRE(r.status == 200);
        REQUIRE(r.complete);
        REQUIRE(took < request_timeout);
        ++produced;
    }

    // The gate was really exercised: the sink was reached and is still blocked.
    CHECK(eventually([&] { return gate.blocked.load() >= 1; }, std::chrono::milliseconds(2000)));

    // The lines the blocked sink cannot take are dropped and counted, and
    // health says so. The logger thread holds at most one batch (at most
    // `capacity` lines) and the queue `capacity` more, so the line after
    // 2 * capacity is dropped at the latest; a reply reports the counters as
    // they were before its own line.
    LogCounters blocked_counters;
    for (std::uint64_t i = 0; i < 3 * kLogCapacity && blocked_counters.dropped == 0; ++i) {
        const Reply r = bounded_get(f.port, "/v1/sonder/health", request_timeout);
        REQUIRE(r.status == 200);
        REQUIRE(r.complete);
        ++produced;
        blocked_counters = log_counters(r);
        REQUIRE(blocked_counters.present);
        CHECK(blocked_counters.capacity == kLogCapacity);
        CHECK(blocked_counters.queued <= blocked_counters.capacity);
    }
    CHECK(blocked_counters.dropped > 0);

    // Once the sink takes lines again the queue drains, and every line the
    // server produced was either written or counted as dropped.
    gate.release();
    LogCounters drained;
    const bool settled = eventually(
        [&] {
            const Reply r = bounded_get(f.port, "/v1/sonder/health", request_timeout);
            if (r.status != 200 || !r.complete) return false;
            const std::uint64_t before = produced;  // lines produced before this request's own
            ++produced;
            drained = log_counters(r);
            return drained.present && drained.queued == 0 && drained.written + drained.dropped == before;
        },
        std::chrono::milliseconds(5000));
    CHECK_MESSAGE(settled, "written " << drained.written << " + dropped " << drained.dropped << " (queued "
                                      << drained.queued << ") never matched the lines produced (" << produced
                                      << " after the last poll)");
    CHECK(drained.dropped >= blocked_counters.dropped);
    // The sink got each written line (the last poll's own line included) plus
    // the "lines dropped" report, which is not counted as written.
    CHECK(eventually(
        [&] {
            const std::uint64_t reports = static_cast<std::uint64_t>(gate.drop_lines.load());
            return reports >= 1 && static_cast<std::uint64_t>(gate.lines.load()) == drained.written + 1 + reports;
        },
        std::chrono::milliseconds(5000)));

    // Block the sink again, with the logger thread inside it.
    gate.arm();
    {
        const Reply r = bounded_get(f.port, "/v1/sonder/health", request_timeout);
        CHECK(r.status == 200);
        CHECK(r.complete);
    }
    CHECK(eventually([&] { return gate.waiting.load() == 1; }, std::chrono::milliseconds(2000)));

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
    // stop() gave up on the sink call that was blocked; that call ends now and
    // the sink is not entered again, so the gate may go out of scope.
    CHECK(eventually([&] { return gate.inside.load() == 0; }, std::chrono::milliseconds(5000)));
}
