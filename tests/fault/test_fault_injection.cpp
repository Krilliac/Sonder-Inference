// Fault-injection tests for the llamaserver Supervisor against a REAL child
// process (sonder_fault_child, tests/fault/fault_child.cpp). No model, no
// weights and no network beyond 127.0.0.1.
//
// Evidence rule for every scenario: assert the supervisor's observable outcome
// (error code, restart count, no hung request, resources released) AND assert
// that the fault actually fired, using the child's own event log. A scenario
// that injects nothing must fail: require_fired() demands N > 0 matching fault
// events, and the "no fault" control test proves the gate reads zero when
// nothing was injected.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "fault_sock.hpp"  // first: pulls winsock2.h before any windows.h
#include "backends/llamaserver/supervisor.hpp"
#include "net/http_client.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/types.h>
#endif

#ifndef SONDER_FAULT_CHILD_PATH
#error "SONDER_FAULT_CHILD_PATH must name the sonder_fault_child executable"
#endif

using namespace sonder::inference;
namespace ls = sonder::inference::llamaserver;
namespace fs = std::filesystem;
using std::chrono::milliseconds;
using std::chrono::seconds;
using Clock = std::chrono::steady_clock;

namespace {

// ---- child event log -------------------------------------------------------

struct Event {
    std::string name;
    std::map<std::string, std::string> fields;
    [[nodiscard]] std::string get(const std::string &key) const {
        const auto it = fields.find(key);
        return it == fields.end() ? std::string{} : it->second;
    }
};

std::vector<Event> read_events(const fs::path &path) {
    std::vector<Event> events;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        std::istringstream words(line);
        Event event;
        if (!(words >> event.name))
            continue;
        std::string word;
        while (words >> word) {
            const auto eq = word.find('=');
            if (eq != std::string::npos)
                event.fields[word.substr(0, eq)] = word.substr(eq + 1);
        }
        events.push_back(std::move(event));
    }
    return events;
}

std::vector<Event> events_named(const std::vector<Event> &events, std::string_view name) {
    std::vector<Event> out;
    for (const auto &event : events)
        if (event.name == name)
            out.push_back(event);
    return out;
}

std::size_t count_events(const std::vector<Event> &events, std::string_view name) {
    return events_named(events, name).size();
}

// The N > 0 gate: a fault test that did not see its fault fire is a failed
// test, never a pass.
std::size_t fired_count(const std::vector<Event> &events, std::string_view fault) {
    return count_events(events, fault);
}

#define REQUIRE_FIRED(events, fault) \
    REQUIRE_MESSAGE(fired_count((events), (fault)) > 0, "fault '" << (fault) << "' never fired: the test injected nothing")

bool wait_until(const std::function<bool()> &predicate, milliseconds timeout) {
    const auto until = Clock::now() + timeout;
    while (Clock::now() < until) {
        if (predicate())
            return true;
        std::this_thread::sleep_for(milliseconds(10));
    }
    return predicate();
}

// ---- process and port liveness ----------------------------------------------

bool process_gone(long pid) {
#if defined(_WIN32)
    HANDLE handle = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (!handle)
        return true;
    const bool exited = WaitForSingleObject(handle, 0) == WAIT_OBJECT_0;
    CloseHandle(handle);
    return exited;
#else
    // A zombie has exited but is not reaped yet: treat it as gone.
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    if (stat) {
        std::string text;
        std::getline(stat, text);
        const auto close_paren = text.rfind(')');
        if (close_paren != std::string::npos && close_paren + 2 < text.size())
            return text[close_paren + 2] == 'Z' || text[close_paren + 2] == 'X';
    }
    return kill(static_cast<pid_t>(pid), 0) != 0;
#endif
}

bool port_refused(std::uint16_t port) {
    const auto s = fault_net::connect_loopback(port);
    if (s == fault_net::kBad)
        return true;
    fault_net::close_sock(s);
    return false;
}

// ---- rig --------------------------------------------------------------------

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path() /
               ("sonder_fault_" + std::to_string(Clock::now().time_since_epoch().count()) + "_" +
                std::to_string(counter.fetch_add(1)));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;
};

// One Supervisor driving the real fault child. The child options come first;
// the supervisor appends --host/--port itself.
struct Rig {
    TempDir dir;
    fs::path events_path = dir.path / "events.log";
    fs::path marker_path = dir.path / "once.marker";
    std::unique_ptr<ls::Supervisor> supervisor;

    explicit Rig(std::vector<std::string> child_args, std::size_t max_restarts = 3,
                 milliseconds readiness = milliseconds(8000)) {
        ls::SupervisorOptions options;
        options.executable = SONDER_FAULT_CHILD_PATH;
        options.arguments = {"--events", events_path.string(), "--once-marker", marker_path.string()};
        options.arguments.insert(options.arguments.end(), child_args.begin(), child_args.end());
        options.readiness_timeout = readiness;
        options.health_poll_interval = milliseconds(20);
        options.restart_initial_backoff = milliseconds(20);
        options.restart_max_backoff = milliseconds(100);
        options.shutdown_timeout = milliseconds(2000);
        options.max_restarts = max_restarts;
        options.native_completion = false;     // no /metrics sampling of the fake
        options.spill_guard.enabled = false;   // no GPU counters
        options.kv_pairing_check = false;
        supervisor = std::make_unique<ls::Supervisor>(std::move(options));
    }
    ~Rig() {
        if (supervisor)
            supervisor->stop();
    }
    Rig(const Rig &) = delete;
    Rig &operator=(const Rig &) = delete;

    [[nodiscard]] std::vector<Event> events() const { return read_events(events_path); }
    [[nodiscard]] std::size_t starts() const { return count_events(events(), "start"); }

    // Port of the newest child that reported "listening".
    [[nodiscard]] std::uint16_t latest_listening_port() const {
        const auto listening = events_named(events(), "listening");
        if (listening.empty())
            return 0;
        return static_cast<std::uint16_t>(std::stoi(listening.back().get("port")));
    }

    // Waits for the supervisor to report the newest child as ready.
    bool wait_ready_on_latest(milliseconds timeout = milliseconds(8000)) {
        return wait_until(
            [&] {
                const auto port = latest_listening_port();
                return port != 0 && supervisor->running() && supervisor->port() == port;
            },
            timeout);
    }

    // After supervisor->stop(): every child is gone and its port refuses.
    void check_resources_released() {
        supervisor->stop();
        const auto all = events();
        const auto started = events_named(all, "start");
        REQUIRE(!started.empty());
        for (const auto &event : started) {
            const long pid = std::stol(event.get("pid"));
            CHECK_MESSAGE(wait_until([&] { return process_gone(pid); }, seconds(5)), "child pid " << pid << " still alive");
        }
        for (const auto &event : events_named(all, "listening")) {
            const auto port = static_cast<std::uint16_t>(std::stoi(event.get("port")));
            CHECK_MESSAGE(wait_until([&] { return port_refused(port); }, seconds(5)),
                          "port " << port << " still accepts connections");
        }
        CHECK_FALSE(supervisor->running());
    }
};

// ---- streaming client ---------------------------------------------------------

struct StreamResult {
    Status status;
    int http_status = 0;
    std::size_t data_events = 0;  // SSE "data: {" events seen
    bool done_marker = false;
    std::size_t bytes = 0;
    milliseconds elapsed{0};
};

StreamResult stream_request(std::uint16_t port, const CancellationToken &cancel, milliseconds total_timeout,
                            const std::string &target = "/v1/chat/completions") {
    net::HttpRequest request;
    request.method = "POST";
    request.host = "127.0.0.1";
    request.port = port;
    request.target = target;
    request.body = R"({"model":"fault","stream":true,"temperature":0,"messages":[{"role":"user","content":"hi"}]})";
    request.connect_timeout = milliseconds(3000);
    request.total_timeout = total_timeout;
    StreamResult result;
    std::string text;
    request.on_status = [&](int status) { result.http_status = status; };
    const auto begin = Clock::now();
    auto response = net::http_request(
        request,
        [&](std::string_view body) {
            result.bytes += body.size();
            text.append(body);
            return true;
        },
        cancel);
    result.elapsed = std::chrono::duration_cast<milliseconds>(Clock::now() - begin);
    result.status = response.ok() ? Status::success() : response.status();
    for (std::size_t at = text.find("data: {"); at != std::string::npos; at = text.find("data: {", at + 1))
        ++result.data_events;
    result.done_marker = text.find("[DONE]") != std::string::npos;
    return result;
}

bool is_transport_break(ErrorCode code) { return code == ErrorCode::protocol_error || code == ErrorCode::io_error; }

} // namespace

// ---- scenarios ----------------------------------------------------------------

TEST_CASE("fault: kill child mid-stream truncates the stream and the supervisor restarts it") {
    Rig rig({"--chunks", "6", "--chunk-interval-ms", "20", "--die-after-chunks", "3"});
    auto first_port = rig.supervisor->start();
    REQUIRE_MESSAGE(first_port.ok(), first_port.status().to_string());

    auto lease = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(lease.ok(), lease.status().to_string());
    const auto broken = stream_request(lease.value().port(), {}, seconds(10));

    // The fault fired, after exactly the configured number of events.
    const auto after_kill = rig.events();
    REQUIRE_FIRED(after_kill, "die_midstream");
    CHECK(events_named(after_kill, "die_midstream").front().get("after_chunks") == "3");
    CHECK(count_events(after_kill, "request_received") == 1);

    // Observable client outcome: a started stream that ends in a transport error.
    CHECK(broken.http_status == 200);
    CHECK_FALSE(broken.status.ok());
    CHECK_MESSAGE(is_transport_break(broken.status.code()), broken.status.to_string());
    CHECK(broken.data_events >= 1);
    CHECK(broken.data_events <= 3);
    CHECK_FALSE(broken.done_marker);
    CHECK(broken.elapsed < seconds(5));  // not hung
    lease.value() = ls::Supervisor::RequestLease{};

    // Supervisor outcome: exactly one restart, service restored.
    REQUIRE(wait_until([&] { return rig.starts() >= 2; }, seconds(8)));
    REQUIRE(rig.wait_ready_on_latest());
    CHECK(rig.starts() == 2);
    CHECK(rig.supervisor->failure().ok());

    auto second = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(second.ok(), second.status().to_string());
    const auto healthy = stream_request(second.value().port(), {}, seconds(10));
    CHECK(healthy.status.ok());
    CHECK(healthy.data_events == 6);
    CHECK(healthy.done_marker);
    CHECK(rig.starts() == 2);  // the healthy request cost no further restart
    second.value() = ls::Supervisor::RequestLease{};

    rig.check_resources_released();
}

TEST_CASE("fault: kill child during warm-up restarts it and readiness still completes") {
    Rig rig({"--warmup-ms", "1200", "--die-warmup-ms", "300"});
    const auto begin = Clock::now();
    auto port = rig.supervisor->start();
    const auto waited = std::chrono::duration_cast<milliseconds>(Clock::now() - begin);
    REQUIRE_MESSAGE(port.ok(), port.status().to_string());

    const auto all = rig.events();
    REQUIRE_FIRED(all, "die_warmup");
    const auto starts = events_named(all, "start");
    REQUIRE(starts.size() == 2);  // one restart
    const auto died = events_named(all, "die_warmup");
    CHECK(died.front().get("pid") == starts[0].get("pid"));
    // The first child was still loading (503) and never became ready; only the
    // replacement did, and the supervisor reports the replacement's port.
    CHECK(count_events(all, "health_503") >= 1);
    const auto ready = events_named(all, "ready");
    REQUIRE(ready.size() == 1);
    CHECK(ready.front().get("pid") == starts[1].get("pid"));
    CHECK(port.value() == rig.latest_listening_port());
    CHECK(rig.supervisor->failure().ok());
    CHECK(waited < seconds(8));

    rig.check_resources_released();
}

TEST_CASE("fault: cancel during a long prefill returns promptly and leaves the child serving") {
    Rig rig({"--first-byte-delay-ms", "6000", "--delay-requests", "1", "--chunks", "3"});
    REQUIRE(rig.supervisor->start().ok());
    auto lease = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(lease.ok(), lease.status().to_string());

    CancellationSource cancel;
    StreamResult cancelled;
    std::thread client([&] { cancelled = stream_request(lease.value().port(), cancel.token(), seconds(30)); });

    // The request must reach the child, which then sits in "prefill".
    const bool arrived = wait_until([&] { return count_events(rig.events(), "request_received") >= 1; }, seconds(8));
    if (!arrived) {
        cancel.cancel();
        client.join();
    }
    REQUIRE_MESSAGE(arrived, "the long-prefill request never reached the child");
    std::this_thread::sleep_for(milliseconds(150));
    CHECK(count_events(rig.events(), "first_byte_sent") == 0);  // still in prefill when cancelled
    cancel.cancel();
    client.join();

    CHECK(cancelled.status.code() == ErrorCode::cancelled);
    CHECK(cancelled.http_status == 0);
    CHECK(cancelled.bytes == 0);
    CHECK(cancelled.elapsed < seconds(4));  // returned long before the 6 s prefill ended

    // Resources released on the child side: it noticed the client leave.
    REQUIRE(wait_until([&] { return count_events(rig.events(), "client_gone") >= 1; }, seconds(5)));
    const auto gone = events_named(rig.events(), "client_gone");
    CHECK(gone.front().get("phase") == "prefill");
    CHECK(count_events(rig.events(), "first_byte_sent") == 0);

    // The supervisor was not disturbed: no restart, and the next request works.
    lease.value() = ls::Supervisor::RequestLease{};
    CHECK(rig.starts() == 1);
    CHECK(rig.supervisor->running());
    auto next = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(next.ok(), next.status().to_string());
    const auto healthy = stream_request(next.value().port(), {}, seconds(10));
    CHECK(healthy.status.ok());
    CHECK(healthy.data_events == 3);
    CHECK(rig.starts() == 1);
    next.value() = ls::Supervisor::RequestLease{};

    rig.check_resources_released();
}

TEST_CASE("fault: a client that stops reading is cut off without affecting other requests") {
    Rig rig({"--chunks", "4", "--flood-timeout-ms", "300"});
    REQUIRE(rig.supervisor->start().ok());
    auto lease = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(lease.ok(), lease.status().to_string());
    const std::uint16_t port = lease.value().port();

    // A raw client with a tiny receive buffer sends a request and never reads.
    const auto stalled = fault_net::connect_loopback(port, 4096);
    REQUIRE(stalled != fault_net::kBad);
    const std::string head = "POST /flood HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
    REQUIRE(fault_net::send_all(stalled, head.data(), head.size()) == head.size());

    // While it is stalled a second client is served normally.
    const auto other = stream_request(port, {}, seconds(10));
    CHECK(other.status.ok());
    CHECK(other.data_events == 4);
    CHECK(other.done_marker);

    // The child gave up on the stalled writer after its send timeout.
    const bool fired = wait_until([&] { return count_events(rig.events(), "write_timeout") >= 1; }, seconds(10));
    fault_net::close_sock(stalled);
    REQUIRE_MESSAGE(fired, "the stalled client never back-pressured the child");
    const auto timeouts = events_named(rig.events(), "write_timeout");
    CHECK(std::stoull(timeouts.front().get("bytes_sent")) > 0);
    CHECK(count_events(rig.events(), "flood_cap_reached") == 0);

    // The supervisor kept the same healthy child throughout.
    CHECK(rig.starts() == 1);
    CHECK(rig.supervisor->running());
    CHECK(rig.supervisor->failure().ok());
    lease.value() = ls::Supervisor::RequestLease{};

    rig.check_resources_released();
}

TEST_CASE("fault: child that exits non-zero at start exhausts the restart budget and fails fast") {
    constexpr std::size_t kMaxRestarts = 2;
    Rig rig({"--exit-at-start", "3"}, kMaxRestarts, milliseconds(5000));
    const auto begin = Clock::now();
    const auto port = rig.supervisor->start();
    const auto waited = std::chrono::duration_cast<milliseconds>(Clock::now() - begin);

    const auto all = rig.events();
    REQUIRE_FIRED(all, "exit_at_start");
    // 1 initial launch + max_restarts relaunches, each exiting with code 3.
    CHECK(count_events(all, "exit_at_start") == kMaxRestarts + 1);
    for (const auto &event : events_named(all, "exit_at_start"))
        CHECK(event.get("code") == "3");
    CHECK(count_events(all, "start") == kMaxRestarts + 1);
    CHECK(count_events(all, "listening") == 0);

    CHECK_FALSE(port.ok());
    CHECK(port.status().code() == ErrorCode::unavailable);
    CHECK(rig.supervisor->failure().code() == ErrorCode::unavailable);
    CHECK_FALSE(rig.supervisor->running());
    CHECK(rig.supervisor->port() == 0);
    CHECK(waited < seconds(4));  // failed on the exits, not on the 5 s readiness timeout

    // A later caller gets the same terminal error immediately, not a hang.
    const auto again_begin = Clock::now();
    auto lease = rig.supervisor->acquire_request();
    CHECK_FALSE(lease.ok());
    CHECK(lease.status().code() == ErrorCode::unavailable);
    CHECK(Clock::now() - again_begin < seconds(2));
    CHECK(count_events(rig.events(), "start") == kMaxRestarts + 1);  // no relaunch after failure

    rig.check_resources_released();
}

// Pins CURRENT behaviour: only the request's total deadline ends an upstream
// that goes silent after the headers (no idle or inter-chunk timeout exists),
// and the supervisor does not restart a child that is merely quiet. If an idle
// timeout is added later, the elapsed lower bound below must be revisited.
TEST_CASE("fault: an upstream that stops sending mid-stream hits the request total timeout and is not restarted") {
    Rig rig({"--chunks", "6", "--chunk-interval-ms", "5", "--stall-after-chunks", "2"});
    auto started = rig.supervisor->start();
    REQUIRE_MESSAGE(started.ok(), started.status().to_string());
    auto lease = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(lease.ok(), lease.status().to_string());

    const auto r = stream_request(lease.value().port(), {}, milliseconds(1000));

    const auto after = rig.events();
    REQUIRE_FIRED(after, "stall_midstream");
    CHECK(events_named(after, "stall_midstream").front().get("after_chunks") == "2");
    CHECK(r.http_status == 200);
    CHECK_MESSAGE(r.status.code() == ErrorCode::timeout, r.status.to_string());
    CHECK(r.data_events == 2);
    CHECK_FALSE(r.done_marker);
    CHECK(r.elapsed >= milliseconds(900));
    CHECK(r.elapsed < seconds(4));

    // The child noticed the client leave while stalled.
    const bool gone = wait_until(
        [&] {
            for (const auto &event : events_named(rig.events(), "client_gone"))
                if (event.get("phase") == "stalled")
                    return true;
            return false;
        },
        seconds(5));
    CHECK_MESSAGE(gone, "the stalled child never saw the client leave");

    CHECK(rig.starts() == 1);
    CHECK(rig.supervisor->running());
    CHECK(rig.supervisor->failure().ok());
    lease.value() = ls::Supervisor::RequestLease{};

    // The stall fires once (marker), so the next request is healthy.
    auto second = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(second.ok(), second.status().to_string());
    const auto healthy = stream_request(second.value().port(), {}, seconds(10));
    CHECK(healthy.status.ok());
    CHECK(healthy.data_events == 6);
    CHECK(healthy.done_marker);
    CHECK(rig.starts() == 1);
    second.value() = ls::Supervisor::RequestLease{};

    rig.check_resources_released();
}

// RED proof of the gate: the mid-stream scenario WITHOUT --die-after-chunks
// injects nothing, so REQUIRE_FIRED must fail. expected_failures(1) makes the
// case pass only when exactly one assertion fails. If setup itself broke, two
// failures are recorded instead, so the case still fails.
TEST_CASE("fault gate: a scenario that injects nothing fails on its fired gate" * doctest::expected_failures(1)) {
    Rig rig({"--chunks", "6", "--chunk-interval-ms", "20"});
    bool setup_ok = false;
    std::vector<Event> all;
    {
        const auto started = rig.supervisor->start();
        if (started.ok()) {
            auto lease = rig.supervisor->acquire_request();
            if (lease.ok()) {
                const auto result = stream_request(lease.value().port(), {}, seconds(10));
                lease.value() = ls::Supervisor::RequestLease{};
                all = rig.events();
                setup_ok = result.status.ok() && result.data_events == 6 && count_events(all, "request_received") == 1;
            }
        }
    }
    if (!setup_ok) {
        FAIL_CHECK("setup failed 1/2");
        FAIL_CHECK("setup failed 2/2");
        return;
    }
    REQUIRE_FIRED(all, "die_midstream");
}

TEST_CASE("fault control: a run that injects nothing reads zero on every fault gate") {
    // Same rig and client as the fault tests, no fault options. Every fault
    // gate must read 0, so any scenario that asserted "the fault fired" (or
    // an error outcome) would fail here instead of passing vacuously.
    Rig rig({"--chunks", "5"});
    REQUIRE(rig.supervisor->start().ok());
    auto lease = rig.supervisor->acquire_request();
    REQUIRE_MESSAGE(lease.ok(), lease.status().to_string());
    const auto result = stream_request(lease.value().port(), {}, seconds(10));
    lease.value() = ls::Supervisor::RequestLease{};

    CHECK(result.status.ok());
    CHECK(result.data_events == 5);
    CHECK(result.done_marker);

    // The child logs stream_complete after it sends the terminating chunk, and
    // the client returns without waiting for the close, so the event can land
    // after the client returns. Wait for it, then take the snapshot.
    CHECK_MESSAGE(wait_until([&] { return count_events(rig.events(), "stream_complete") >= 1; }, seconds(10)),
                  "the child never logged stream_complete");
    const auto all = rig.events();
    CHECK(count_events(all, "request_received") == 1);  // the traffic was real
    CHECK(count_events(all, "stream_complete") == 1);
    for (const char *fault :
         {"die_midstream", "die_warmup", "exit_at_start", "client_gone", "write_timeout", "stall_midstream"})
        CHECK_MESSAGE(fired_count(all, fault) == 0, "control run recorded fault " << fault);
    CHECK(rig.starts() == 1);

    rig.check_resources_released();
}
