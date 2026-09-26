#include <doctest/doctest.h>

#include <chrono>
#include <thread>

#include "test_helpers.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

TEST_SUITE("cancellation") {
TEST_CASE("tokens are cheap, sticky, and shared") {
    CancellationToken never;
    CHECK_FALSE(never.cancelled());
    CHECK_FALSE(never.can_be_cancelled());

    CancellationSource src;
    auto a = src.token();
    auto b = a;
    CHECK(a.can_be_cancelled());
    CHECK_FALSE(b.cancelled());
    src.cancel();
    src.cancel();  // idempotent
    CHECK(a.cancelled());
    CHECK(b.cancelled());
    CHECK(src.cancelled());
}

TEST_CASE("cancel from inside the stream callback") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(64));
    REQUIRE(session);
    int seen = 0;
    auto r = session->generate("cancel me", [&](const TokenChunk&) {
        if (++seen == 3) {
            session->cancel();
        }
        return true;
    });
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::cancelled);
    CHECK(r.value().stats.stop_reason == StopReason::cancelled);
    CHECK(r.value().stats.chunks == 3);
    CHECK(session->last_outcome() == RequestOutcome::cancelled);
    CHECK(session->state() == SessionState::idle);

    auto cancelled = h.events_of("request.cancelled");
    REQUIRE(cancelled.size() == 1);
    const auto& attrs = *cancelled[0].find("attributes");
    CHECK(attrs.find("outcome")->as_string() == "cancelled");
    CHECK(attrs.find("chunks")->as_int() == 3);
    REQUIRE(attrs.find("cancel_latency_ms") != nullptr);
    CHECK(attrs.find("cancel_latency_ms")->as_double() >= 0.0);
    CHECK(h.events_of("request.completed").empty());
}

TEST_CASE("cancel from another thread returns promptly") {
    MockBackendOptions slow;
    slow.token_delay = 50ms;
    slow.default_completion_tokens = 200;  // ~10 s if never cancelled
    sonder_test::Harness h(slow);
    auto session = h.session(SamplingConfig::greedy(200));
    REQUIRE(session);

    std::thread canceller([&] {
        std::this_thread::sleep_for(150ms);
        session->cancel();
    });
    const auto t0 = std::chrono::steady_clock::now();
    auto r = session->generate("slow request");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    canceller.join();
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::cancelled);
    CHECK(elapsed < 3s);
    CHECK(r.value().stats.chunks < 200);
}

TEST_CASE("cancel while idle does not poison the next request") {
    sonder_test::Harness h;
    auto session = h.session();
    REQUIRE(session);
    session->cancel();
    auto r = session->generate("after idle cancel");
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::completed);
}

TEST_CASE("close cancels an in-flight request and is terminal") {
    MockBackendOptions slow;
    slow.token_delay = 20ms;
    slow.default_completion_tokens = 500;
    sonder_test::Harness h(slow);
    auto session = h.session(SamplingConfig::greedy(500));
    REQUIRE(session);
    std::thread closer([&] {
        std::this_thread::sleep_for(100ms);
        session->close();
    });
    auto r = session->generate("closing");
    closer.join();
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::cancelled);
    CHECK(session->state() == SessionState::closed);
    CHECK(session->generate("again").status().code() == ErrorCode::invalid_state);
}
}
