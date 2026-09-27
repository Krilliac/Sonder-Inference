#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>
#include <thread>

#include "test_helpers.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

TEST_SUITE("session") {
TEST_CASE("telemetry records a 64-bit seed above INT64_MAX exactly") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    REQUIRE(h.model);
    auto sampling = SamplingConfig::greedy(4);
    sampling.seed = 18446744073709551615ull;
    auto session = h.session(sampling);
    REQUIRE(session);
    REQUIRE(session->generate("seeded").ok());
    h.engine->telemetry().flush();
    int seen = 0;
    for (const auto& line : h.sink->lines()) {
        if (line.find("\"event_type\":\"session.created\"") != std::string::npos ||
            line.find("\"event_type\":\"request.started\"") != std::string::npos) {
            CHECK(line.find("\"seed\":18446744073709551615") != std::string::npos);
            CHECK(line.find("\"seed\":-") == std::string::npos);
            ++seen;
        }
    }
    CHECK(seen == 2);
}

TEST_CASE("lifecycle: idle -> running -> idle -> closed") {
    sonder_test::Harness h;
    REQUIRE(h.model);
    auto session = h.session();
    REQUIRE(session);
    CHECK(session->state() == SessionState::idle);
    CHECK(session->last_outcome() == RequestOutcome::none);
    CHECK_FALSE(session->id().empty());

    SessionState during = SessionState::idle;
    std::string streamed;
    auto r = session->generate("hello there", [&](const TokenChunk& c) {
        during = session->state();
        streamed.append(c.text);
        return true;
    });
    REQUIRE(r.ok());
    CHECK(during == SessionState::running);
    CHECK(session->state() == SessionState::idle);
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK(r.value().text == streamed);
    CHECK(r.value().ttft_ms >= 0.0);
    CHECK(r.value().total_ms >= r.value().ttft_ms);
    CHECK(session->last_outcome() == RequestOutcome::completed);

    auto r2 = session->generate("second request");
    REQUIRE(r2.ok());
    CHECK(r2.value().request_id != r.value().request_id);
    CHECK(session->requests_started() == 2);

    session->close();
    CHECK(session->state() == SessionState::closed);
    session->close();  // idempotent
    CHECK(session->generate("closed").status().code() == ErrorCode::invalid_state);
}

TEST_CASE("rejects a concurrent request on the same session") {
    MockBackendOptions slow;
    slow.token_delay = 20ms;
    slow.default_completion_tokens = 20;
    sonder_test::Harness h(slow);
    auto session = h.session(SamplingConfig::greedy(20));
    REQUIRE(session);
    std::atomic<bool> started{false};
    std::thread worker([&] {
        (void)session->generate("long", [&](const TokenChunk&) {
            started = true;
            return true;
        });
    });
    while (!started) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK(session->generate("overlap").status().code() == ErrorCode::invalid_state);
    session->cancel();
    worker.join();
    CHECK(session->state() == SessionState::idle);
}

TEST_CASE("backend failure marks the request failed but keeps the session usable") {
    MockBackendOptions failing;
    failing.fail_after_tokens = 3;
    sonder_test::Harness h(failing);
    auto session = h.session();
    REQUIRE(session);
    auto r = session->generate("will fail");
    CHECK(r.status().code() == ErrorCode::backend_error);
    CHECK(session->last_outcome() == RequestOutcome::failed);
    CHECK(session->state() == SessionState::idle);
    auto failed = h.events_of("request.failed");
    REQUIRE(failed.size() == 1);
    CHECK(failed[0].find("attributes")->find("error_code")->as_string() == "backend_error");
}

TEST_CASE("emits correlated lifecycle events in order") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(4));
    REQUIRE(session);
    auto r = session->generate("trace me");
    REQUIRE(r.ok());
    const std::string rid = r.value().request_id;

    std::vector<std::string> types;
    std::vector<std::string> all_types;
    for (const auto& e : h.events()) {
        const json::Value* req = e.find("request_id");
        if (req && req->as_string() == rid) {
            const std::string type = e.find("event_type")->as_string();
            all_types.push_back(type);
            // Scheduler/KV/prefill events are covered by test_engine_runtime.cpp.
            if (type.rfind("request.", 0) == 0 || type.rfind("inference.decode.", 0) == 0 ||
                type == "inference.token.generated") {
                types.push_back(type);
            }
            CHECK(e.find("session_id")->as_string() == session->id());
            CHECK(e.find("model_instance_id")->as_string() == h.model->instance_id());
            CHECK(e.find("device_id")->as_string() == "cpu:0");
        }
    }
    const std::vector<std::string> expected = {
        "request.queued",           "request.started",          "inference.decode.started",
        "inference.token.generated", "inference.token.generated", "inference.token.generated",
        "inference.token.generated", "inference.decode.completed", "request.completed"};
    CHECK(types == expected);
    if (h.engine->scheduling_active()) {
        auto pos = [&](const std::string& t) {
            return std::find(all_types.begin(), all_types.end(), t) - all_types.begin();
        };
        const auto n = static_cast<std::ptrdiff_t>(all_types.size());
        REQUIRE(pos("scheduler.enqueued") < n);
        CHECK(pos("scheduler.enqueued") < pos("scheduler.admitted"));
        CHECK(pos("scheduler.admitted") < pos("scheduler.prefill.completed"));
        CHECK(pos("scheduler.prefill.completed") < pos("inference.decode.started"));
        CHECK(pos("kv.freed") < pos("request.completed"));
    }

    auto done = h.events_of("request.completed");
    REQUIRE(done.size() == 1);
    const auto& a = *done[0].find("attributes");
    CHECK(a.find("completion_tokens")->as_int() == 4);
    CHECK(a.find("prompt_tokens")->as_int() == 2);
    CHECK(a.find("stop_reason")->as_string() == "max_tokens");
    CHECK(a.find("total_ms")->is_number());
    CHECK(a.find("ttft_ms")->is_number());
}

TEST_CASE("model handles outlive unload while sessions hold them") {
    sonder_test::Harness h;
    auto session = h.session();
    REQUIRE(session);
    const std::string id = h.model->instance_id();
    CHECK(h.engine->unload_model(id).ok());
    CHECK(h.engine->unload_model(id).code() == ErrorCode::not_found);
    CHECK(h.engine->loaded_models().empty());
    auto r = session->generate("still works");
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK(h.events_of("model.unload").size() == 1);
}

TEST_CASE("engine rejects unknown backends and duplicate registration") {
    sonder_test::Harness h;
    ModelLoadOptions lo;
    lo.model = "x";
    CHECK(h.engine->load_model("nope", lo).status().code() == ErrorCode::not_found);
    CHECK(h.engine->register_backend(make_mock_backend()).code() == ErrorCode::invalid_state);
    CHECK(h.engine->register_backend(nullptr).code() == ErrorCode::invalid_argument);
    CHECK(h.engine->create_session(nullptr).status().code() == ErrorCode::invalid_argument);
    CHECK(h.events_of("model.load.completed").size() == 1);
}
}
