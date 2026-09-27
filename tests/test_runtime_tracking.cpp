// The request runtime must not keep per-request state after a request ends:
// a long-running server serves an unbounded number of requests.
#include <doctest/doctest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "engine/request_runtime.hpp"
#include "sonder/inference.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

TEST_SUITE("runtime_tracking") {
#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

TEST_CASE("finished requests leave no scheduler or runtime state behind") {
    MockBackendOptions mock;
    mock.default_completion_tokens = 6;
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::off;
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(make_mock_backend(mock)).ok());
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto model = engine.load_model(kMockBackendName, lo);
    REQUIRE(model.ok());
    detail::RequestRuntime* runtime = engine.request_runtime();
    REQUIRE(runtime != nullptr);

    SessionOptions so;
    so.sampling = SamplingConfig::greedy(8);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());
    for (int i = 0; i < 50; ++i) {
        auto r = session.value()->generate("request " + std::to_string(i));
        REQUIRE(r.ok());
    }
    // A request stopped early by its callback.
    int chunks = 0;
    auto stopped = session.value()->generate("stop me", [&](const TokenChunk&) { return ++chunks < 2; });
    REQUIRE(stopped.ok());

    // The runtime settles a completed request on its next step.
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (runtime->tracked_requests() != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK(runtime->tracked_requests() == 0);
}

#endif
}
