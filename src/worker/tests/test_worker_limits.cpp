#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include "server_test_support.hpp"
#include "server/src/identity.hpp"
#include "worker/ipc.hpp"
#include "worker/process.hpp"
#include "worker/protocol.hpp"
#include "worker/remote_backend.hpp"

namespace {
namespace si = sonder::inference;
namespace wk = si::worker;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

wk::RemoteBackendOptions options(const std::string &mode = {}) {
    wk::RemoteBackendOptions out;
    out.executable = mode.empty() ? SONDER_WORKER_EXECUTABLE : SONDER_WORKER_ADVERSARY;
    out.test_worker_mode = mode;
    out.max_restarts_per_hour = 0; // retirement is observable without another child
    out.inactivity_timeout = 1000ms;
    out.request_timeout = 5000ms;
    out.cancel_grace = 80ms;
    return out;
}

si::GenerateRequest request(int count = 2) {
    si::GenerateRequest out;
    out.prompt = "bounded request";
    out.sampling = si::SamplingConfig::greedy(count);
    return out;
}

void retired(wk::RemoteBackend &backend) {
    CHECK(server_test::eventually([&] { return backend.worker_pid() == 0 && backend.circuit_open(); }, 4000ms));
}

void credited_cancel_case(const std::string &mode, std::uint64_t expected_callbacks, bool healthy, int max_tokens = 4) {
    const auto marker = std::filesystem::temp_directory_path() /
        ("sonder-credit-" + mode + "-" + std::to_string(Clock::now().time_since_epoch().count()) + ".ready");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{marker};
    auto config = options(mode + "|" + marker.string());
    config.cancel_grace = 1000ms;
    config.max_restarts_per_hour = healthy ? 2 : 0;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    const auto pid = backend.worker_pid();
    REQUIRE(pid != 0);
    si::CancellationSource cancellation;
    std::atomic<bool> saw_credit{false};
    std::jthread canceller([&](std::stop_token stop) {
        const auto until = Clock::now() + 5s;
        while (!stop.stop_requested() && Clock::now() < until) {
            std::error_code error;
            if (std::filesystem::exists(marker, error)) {
                saw_credit.store(true);
                cancellation.cancel();
                return;
            }
            std::this_thread::sleep_for(1ms);
        }
    });
    std::uint64_t callbacks = 0;
    const auto result = model.value()->generate(request(max_tokens), cancellation.token(), [&](const si::TokenChunk &) {
        ++callbacks;
        return true;
    });
    canceller.request_stop();
    canceller.join();
    REQUIRE(saw_credit.load());
    REQUIRE_FALSE(result.ok());
    CHECK(callbacks == expected_callbacks);
    if (healthy) {
        CHECK(result.status().code() == si::ErrorCode::cancelled);
        REQUIRE(backend.probe().ok()); // also proves no stray ACK/terminal frame poisons reuse
        REQUIRE(backend.worker_pid() == pid);
        REQUIRE(backend.restart_count() == 0);
        std::uint64_t reused_callbacks = 0;
        const auto reused = model.value()->generate(request(), {}, [&](const si::TokenChunk &) {
            ++reused_callbacks;
            return true;
        });
        REQUIRE_MESSAGE(reused.ok(), reused.status().to_string());
        CHECK(reused_callbacks == 2);
        CHECK(backend.worker_pid() == pid);
        CHECK(backend.restart_count() == 0);
    } else {
        CHECK(result.status().code() == si::ErrorCode::protocol_error);
        retired(backend);
    }
}

struct Peer {
    std::unique_ptr<wk::Listener> listener;
    std::unique_ptr<wk::Process> process;
    std::shared_ptr<wk::Channel> channel;
    Peer() {
        auto listening = wk::make_listener();
        REQUIRE(listening.ok());
        listener = std::move(listening).value();
        auto child = wk::spawn_process(SONDER_WORKER_EXECUTABLE, {"worker", "--endpoint", listener->endpoint(),
                                                                  "--engine", "mock", "--token-delay-ms", "20"});
        REQUIRE(child.ok());
        process = std::move(child).value();
        auto accepted = listener->accept(5000ms);
        REQUIRE(accepted.ok());
        channel = accepted.value();
        REQUIRE(read().find("type")->as_string() == "hello");
        auto ready = wk::message("ready", 0);
        ready.set("engine", "mock");
        send(std::move(ready));
    }
    void send(si::json::Value value) { REQUIRE(channel->write_frame(value.dump(), 1000ms).ok()); }
    si::json::Value read() {
        auto frame = channel->read_frame(5000ms);
        REQUIRE_MESSAGE(frame.ok(), frame.status().to_string());
        auto parsed = wk::parse_message(frame.value());
        REQUIRE(parsed.ok());
        return std::move(parsed).value();
    }
    void generate(std::uint64_t id, int count) {
        auto encoded = wk::generate_to_json(request(count));
        REQUIRE(encoded.ok());
        auto msg = wk::message("generate", id);
        msg.set("model", "mock:tiny");
        msg.set("device", "cpu:0");
        msg.set("request", encoded.value());
        send(std::move(msg));
    }
};
} // namespace

TEST_CASE("worker: a frame after a sub-poll inactivity deadline is rejected") {
    auto config = options("trickle"); // first frame arrives after 10ms
    config.inactivity_timeout = 1ms;  // receive() polls for up to 50ms
    wk::RemoteBackend backend(config);
    auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    std::size_t callbacks = 0;
    const auto result = model.value()->generate(request(), {}, [&](const si::TokenChunk &) {
        ++callbacks;
        return true;
    });
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::timeout);
    CHECK(callbacks == 0);
    retired(backend);
}

TEST_CASE("worker: a one millisecond read consumes an already queued frame") {
    auto listener = wk::make_listener();
    REQUIRE(listener.ok());
    const auto client = wk::connect(listener.value()->endpoint(), 1000ms);
    REQUIRE(client.ok());
    const auto server = listener.value()->accept(1000ms);
    REQUIRE(server.ok());
    REQUIRE(client.value()->write_frame("queued", 1000ms).ok());
    auto received = server.value()->read_frame(1ms);
    for (int i = 0; !received && received.status().code() == si::ErrorCode::timeout && i < 10; ++i)
        received = server.value()->read_frame(1ms);
    REQUIRE_MESSAGE(received.ok(), received.status().to_string());
    CHECK(received.value() == "queued");
}

TEST_CASE("worker: short IPC deadlines preserve completed frames and accepts") {
    for (int i = 0; i < 30; ++i) {
        auto listener = wk::make_listener();
        REQUIRE(listener.ok());
        si::Result<std::shared_ptr<wk::Channel>> connected = si::Status(si::ErrorCode::internal, "pending");
        std::jthread connector([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(9 + i % 3));
            connected = wk::connect(listener.value()->endpoint(), 1000ms);
        });
        auto accepted = listener.value()->accept(10ms);
        if (!accepted && accepted.status().code() == si::ErrorCode::timeout)
            accepted = listener.value()->accept(1000ms);
        connector.join();
        REQUIRE_MESSAGE(accepted.ok(), accepted.status().to_string());
        REQUIRE(connected.ok());
        si::Status written;
        std::jthread writer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(9 + i % 3));
            written = connected.value()->write_frame("complete", 1000ms);
        });
        auto frame = accepted.value()->read_frame(10ms);
        for (int retry = 0; !frame && frame.status().code() == si::ErrorCode::timeout && retry < 10; ++retry)
            frame = accepted.value()->read_frame(10ms);
        writer.join();
        REQUIRE(written.ok());
        REQUIRE_MESSAGE(frame.ok(), frame.status().to_string());
        CHECK(frame.value() == "complete");
        REQUIRE(connected.value()->write_frame("still open", 1000ms).ok());
        auto next = accepted.value()->read_frame(1000ms);
        REQUIRE(next.ok());
        CHECK(next.value() == "still open");
    }
}

TEST_CASE("worker: metadata and diagnostic counters have semantic bounds") {
    si::ModelDescriptor d;
    d.name = "mock:tiny";
    d.backend = "mock";
    d.context_length = 4096;
    const auto valid = wk::descriptor_to_json(d);
    REQUIRE(wk::descriptor_from_json(valid).ok());
    for (const char *key : {"size_bytes", "context_length"}) {
        auto bad = valid;
        bad.object().set(key, std::numeric_limits<std::uint64_t>::max());
        CHECK_FALSE(wk::descriptor_from_json(bad).ok());
    }
    auto wrong_backend = valid;
    wrong_backend.object().set("backend", "llamacpp");
    CHECK_FALSE(wk::descriptor_from_json(wrong_backend).ok());
    si::GenerateStats stats;
    stats.prompt_tokens = 3;
    stats.chunks = stats.completion_tokens = 2;
    stats.stop_reason = si::StopReason::max_tokens;
    const auto wire = wk::stats_to_json(stats);
    REQUIRE(wk::stats_from_json(wire).ok());
    for (const char *key : {"prompt_tokens", "completion_tokens", "chunks", "load_ns", "prompt_eval_ns", "eval_ns"}) {
        auto bad = wire;
        bad.object().set(key, std::numeric_limits<std::uint64_t>::max());
        CHECK_FALSE(wk::stats_from_json(bad).ok());
    }
}

TEST_CASE("worker: loaded model identity must match the request") {
    for (const char *mode : {"bad-model", "bad-backend", "bad-descriptor"}) {
        CAPTURE(mode);
        wk::RemoteBackend backend(options(mode));
        const auto model = backend.load_model({"mock:tiny"});
        REQUIRE_FALSE(model.ok());
        CHECK(model.status().code() == si::ErrorCode::protocol_error);
        retired(backend);
    }
}

TEST_CASE("worker: overruns oversized tokens and inconsistent stats retire the channel") {
    for (const char *mode :
         {"overrun", "oversized", "reasoning", "candidates", "bad-stats", "inconsistent-stats", "bad-timing"}) {
        CAPTURE(mode);
        wk::RemoteBackend backend(options(mode));
        const auto model = backend.load_model({"mock:tiny"});
        REQUIRE(model.ok());
        std::uint64_t delivered = 0;
        auto input = request();
        if (std::string(mode) == "candidates") {
            // Keep this a byte-budget test; the candidates are explicitly requested.
            input.sampling.logprobs = true;
            input.sampling.top_logprobs = si::SamplingConfig::kMaxTopLogprobs;
        }
        const auto result = model.value()->generate(input, {}, [&](const si::TokenChunk &) {
            ++delivered;
            return true;
        });
        REQUIRE_FALSE(result.ok());
        CHECK(result.status().code() == si::ErrorCode::protocol_error);
        CHECK(delivered <= 2);
        if (std::string(mode) != "overrun")
            CHECK(delivered == 0);
        retired(backend);
    }
}

TEST_CASE("worker: cumulative output ceiling bounds many valid token frames") {
    auto config = options("cumulative");
    config.request_timeout = 60s;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    std::uint64_t bytes = 0;
    const auto result = model.value()->generate(request(1100), {}, [&](const si::TokenChunk &chunk) {
        bytes += chunk.text.size();
        return true;
    });
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::protocol_error);
    CHECK(bytes == wk::kMaxRequestOutputBytes);
    retired(backend);
}

TEST_CASE("worker: absolute deadline bounds a continuously active worker") {
    auto config = options("trickle");
    config.request_timeout = 150ms;
    config.inactivity_timeout = 100ms;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    std::uint64_t count = 0;
    const auto result = model.value()->generate(request(1000), {}, [&](const si::TokenChunk &) {
        ++count;
        return true;
    });
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::timeout);
    CHECK(count > 0);
    CHECK(count < 1000);
    retired(backend);
}

TEST_CASE("worker: configurable inactivity timeout retires a silent worker") {
    auto config = options("silence");
    config.inactivity_timeout = 80ms;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    const auto before = Clock::now();
    const auto result = model.value()->generate(request(), {}, {});
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::timeout);
    CHECK(Clock::now() - before < 2s);
    retired(backend);
}

TEST_CASE("worker: cancellation grace starts at dispatch and retires an unresponsive worker") {
    wk::RemoteBackend backend(options("ignore-cancel"));
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    si::CancellationSource cancellation;
    Clock::time_point cancelled_at{};
    const auto result = model.value()->generate(request(), cancellation.token(), [&](const si::TokenChunk &) {
        std::this_thread::sleep_for(150ms); // greater than grace, but consumer time
        cancelled_at = Clock::now();
        cancellation.cancel();
        return true;
    });
    const auto elapsed = Clock::now() - cancelled_at;
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::cancelled);
    CHECK(elapsed >= 70ms);
    CHECK(elapsed < 2s);
    retired(backend);
}

TEST_CASE("worker: tokens after cancellation or completion are never delivered") {
    for (const char *mode : {"post-cancel", "post-terminal"}) {
        CAPTURE(mode);
        auto config = options(mode);
        config.cancel_grace = 1000ms;
        wk::RemoteBackend backend(config);
        const auto model = backend.load_model({"mock:tiny"});
        REQUIRE(model.ok());
        si::CancellationSource cancellation;
        std::uint64_t count = 0;
        const auto result = model.value()->generate(request(), cancellation.token(), [&](const si::TokenChunk &) {
            ++count;
            cancellation.cancel();
            return true;
        });
        if (std::string(mode) == "post-cancel") {
            CHECK_FALSE(result.ok());
            CHECK(count == 1);
        } else {
            CHECK(count == 0);
            if (result)
                CHECK(result->completion_tokens == 0);
        }
        retired(backend);
    }
}

TEST_CASE("worker: asynchronous cancellation drains one credited token without restarting") {
    credited_cancel_case("credited-cancel", 1, true);
    credited_cancel_case("credited-initial", 0, true);
}

TEST_CASE("worker: a second token after credited cancellation retires the worker") {
    credited_cancel_case("credited-overrun", 1, false);
}

TEST_CASE("worker: credited cancellation never bypasses the request token cap") {
    credited_cancel_case("credited-cap", 1, false, 1);
}

TEST_CASE("worker: unrequested or excess logprobs are rejected before callbacks") {
    for (const char *mode : {"unrequested-logprob", "unrequested-candidates", "excess-candidates"}) {
        CAPTURE(mode);
        wk::RemoteBackend backend(options(mode));
        const auto model = backend.load_model({"mock:tiny"});
        REQUIRE(model.ok());
        auto input = request();
        if (std::string(mode) == "excess-candidates") {
            input.sampling.logprobs = true;
            input.sampling.top_logprobs = 1;
        }
        std::uint64_t callbacks = 0;
        const auto result = model.value()->generate(input, {}, [&](const si::TokenChunk &) {
            ++callbacks;
            return true;
        });
        REQUIRE_FALSE(result.ok());
        CHECK(result.status().code() == si::ErrorCode::protocol_error);
        CHECK(callbacks == 0);
        retired(backend);
    }
}

TEST_CASE("worker: consumer backpressure does not count as worker inactivity") {
    auto config = options();
    config.token_delay = 20ms;
    config.inactivity_timeout = 60ms;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    const auto pid = backend.worker_pid();
    const auto result = model.value()->generate(request(), {}, [](const si::TokenChunk &) {
        std::this_thread::sleep_for(150ms);
        return true;
    });
    REQUIRE_MESSAGE(result.ok(), result.status().to_string());
    CHECK(result->completion_tokens == 2);
    CHECK(backend.worker_pid() == pid);
    CHECK(backend.restart_count() == 0);
}

TEST_CASE("worker: destroying the backend cancels an in-flight retained model request") {
    auto config = options();
    config.token_delay = 100ms;
    auto backend = std::make_unique<wk::RemoteBackend>(config);
    const auto model = backend->load_model({"mock:tiny"});
    REQUIRE(model.ok());
    std::promise<void> first;
    auto started = first.get_future();
    std::promise<si::Result<si::GenerateStats>> completion;
    auto finished = completion.get_future();
    std::jthread generation([&] {
        bool announced = false;
        completion.set_value(model.value()->generate(request(16), {}, [&](const si::TokenChunk &) {
            if (!announced) {
                announced = true;
                first.set_value();
            }
            return true;
        }));
    });
    REQUIRE(started.wait_for(3s) == std::future_status::ready);
    const auto before = Clock::now();
    backend.reset();
    CHECK(Clock::now() - before < 4s);
    REQUIRE(finished.wait_for(3s) == std::future_status::ready);
    const auto result = finished.get();
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::cancelled);
}

TEST_CASE("worker: command admission is bounded while another request owns IPC") {
    auto config = options();
    config.request_timeout = 200ms;
    wk::RemoteBackend backend(config);
    const auto model = backend.load_model({"mock:tiny"});
    REQUIRE(model.ok());
    std::promise<void> first;
    auto entered = first.get_future();
    std::atomic<bool> release{false};
    std::jthread generation([&] {
        (void)model.value()->generate(request(1), {}, [&](const si::TokenChunk &) {
            first.set_value();
            while (!release.load())
                std::this_thread::sleep_for(1ms);
            return true;
        });
    });
    const bool started = entered.wait_for(3s) == std::future_status::ready;
    if (!started)
        release.store(true);
    REQUIRE(started);
    const auto result = backend.probe();
    release.store(true);
    REQUIRE_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::timeout);
}

TEST_CASE("worker: late cancel and stop cannot poison the next request") {
    Peer peer;
    peer.generate(1, 1);
    auto initial = peer.read();
    REQUIRE(initial.find("type")->as_string() == "token");
    auto initial_ack = wk::message("ack", 1);
    initial_ack.set("index", 0);
    peer.send(std::move(initial_ack));
    auto done = peer.read();
    REQUIRE(done.find("type")->as_string() == "done");
    peer.send(wk::message("cancel", 1));
    auto stale = wk::message("stop", 1);
    stale.set("extra", true);
    peer.send(std::move(stale));
    peer.generate(2, 2);
    auto first = peer.read();
    REQUIRE(first.find("type")->as_string() == "token");
    CHECK(first.find("id")->as_uint() == 2);
    peer.send(wk::message("stop", 1));
    for (std::uint64_t i = 0; i < 2; ++i) {
        auto ack = wk::message("ack", 2);
        ack.set("index", i);
        peer.send(std::move(ack));
        const auto next = peer.read();
        CHECK(next.find("id")->as_uint() == 2);
        CHECK(next.find("type")->as_string() == (i == 0 ? "token" : "done"));
    }
    CHECK(peer.process->running());
    peer.send(wk::message("shutdown", 0));
}

TEST_CASE("worker: remote mock identity uses the mock digests and its own backend name") {
    si::ModelDescriptor d;
    d.name = "mock:tiny";
    d.backend = "remote-mock";
    d.context_length = 4096;
    const auto remote = si::server::detail::backend_identity("remote-mock", d.name, &d, std::string("mock-1"));
    d.backend = "mock";
    const auto local = si::server::detail::backend_identity("mock", d.name, &d, std::string("mock-1"));
    REQUIRE(remote.backend_identity.is_object());
    REQUIRE(local.backend_identity.is_object());
    CHECK(remote.backend_identity.find("backend")->as_string() == "remote-mock");
    for (const char *key : {"model_digest", "tokenizer_digest", "template_digest"})
        CHECK(remote.backend_identity.find(key)->as_string() == local.backend_identity.find(key)->as_string());
}
