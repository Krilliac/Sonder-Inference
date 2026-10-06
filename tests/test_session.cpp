#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <set>
#include <thread>

#include "test_helpers.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

namespace {
template <class Marker>
[[noreturn]] void throw_callback_marker(Marker marker, const void*& address) {
    try {
        throw marker;
    } catch (const Marker& caught) {
        address = &caught;
        throw;  // Rethrow the actual live exception object, without capture/copy.
    }
}
}  // namespace

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
TEST_CASE("throwing token callbacks preserve exceptions and allow immediate reuse") {
    for (const bool chat : {false, true}) {
        for (const int throw_at : {1, 3}) {
            for (const bool nonstandard : {false, true}) {
                CAPTURE(chat);
                CAPTURE(throw_at);
                CAPTURE(nonstandard);
                sonder_test::Harness h({}, TelemetryLevel::standard, false);
                auto session = h.session(SamplingConfig::greedy(5));
                REQUIRE(session);
                const std::string secret = "callback-exception-private-canary";
                const void* thrown = nullptr;
                RequestOptions options;
                options.request_id = "callback-failed-request";
                options.parent_request_id = "callback-parent";
                int calls = 0;
                bool overlap_rejected = false;
                std::string delivered;
                const TokenCallback callback = [&](const TokenChunk& chunk) {
                    if (calls == 0) {
                        overlap_rejected = session->generate("nested request").status().code() == ErrorCode::invalid_state
                                           && session->requests_started() == 1;
                    }
                    delivered.append(chunk.text);
                    if (++calls == throw_at) {
                        if (nonstandard) throw_callback_marker(73, thrown);
                        throw_callback_marker(std::runtime_error(secret), thrown);
                    }
                    return true;
                };
                bool caught = false;
                try {
                    if (chat) (void)session->chat({{"user", "first request"}}, callback, std::nullopt, options);
                    else (void)session->generate("first request", callback, std::nullopt, options);
                } catch (const std::runtime_error& error) {
                    caught = true;
                    CHECK_FALSE(nonstandard);
                    CHECK(static_cast<const void*>(&error) == thrown);
                    CHECK(std::string(error.what()) == secret);
                } catch (const int& marker) {
                    caught = true;
                    CHECK(nonstandard);
                    CHECK(static_cast<const void*>(&marker) == thrown);
                    CHECK(marker == 73);
                }
                CHECK(caught);
                CHECK(overlap_rejected);
                CHECK(calls == throw_at);
                CHECK_FALSE(delivered.empty());  // Already delivered effects survive.
                CHECK(session->state() == SessionState::idle);
                CHECK(session->last_outcome() == RequestOutcome::failed);
                CHECK_FALSE(session->last_scheduler_rejected());
                CHECK(session->requests_started() == 1);
                const auto failed = h.events_of("request.failed");
                REQUIRE(failed.size() == 1);
                CHECK(failed[0].find("request_id")->as_string() == "callback-failed-request");
                const auto* attrs = failed[0].find("attributes");
                REQUIRE(attrs);
                CHECK(attrs->find("parent_request_id")->as_string() == "callback-parent");
                CHECK(attrs->find("error_code")->as_string() == "internal");
                CHECK(attrs->find("chunks")->as_int() == throw_at);
                CHECK(h.events_of("request.completed").empty());
                CHECK(h.events_of("request.cancelled").empty());
                for (const auto& line : h.sink->lines()) CHECK(line.find(secret) == std::string::npos);
                const auto retained = delivered;
                auto reused = session->generate("fresh request");
                REQUIRE(reused.ok());
                CHECK(reused->outcome == RequestOutcome::completed);
                CHECK(session->requests_started() == 2);
                CHECK(calls == throw_at);
                CHECK(delivered == retained);
            }
        }
    }
}

TEST_CASE("cancel or close before a callback exception preserves terminal state") {
    for (const bool close : {false, true}) {
        CAPTURE(close);
        sonder_test::Harness h;
        auto session = h.session(SamplingConfig::greedy(4));
        REQUIRE(session);
        const void* thrown = nullptr;
        int calls = 0;
        bool caught = false;
        try {
            (void)session->generate("close or cancel", [&](const TokenChunk&) -> bool {
                ++calls;
                if (close) session->close();
                else session->cancel();
                throw_callback_marker(std::runtime_error("private-close-canary"), thrown);
            });
        } catch (const std::runtime_error& error) {
            caught = true;
            CHECK(static_cast<const void*>(&error) == thrown);
            CHECK(std::string(error.what()) == "private-close-canary");
        }
        CHECK(caught);
        CHECK(calls == 1);
        CHECK(session->last_outcome() == RequestOutcome::failed);
        CHECK_FALSE(session->last_scheduler_rejected());
        if (close) {
            CHECK(session->state() == SessionState::closed);
            CHECK(session->generate("must stay closed").status().code() == ErrorCode::invalid_state);
            CHECK(session->requests_started() == 1);
        } else {
            CHECK(session->state() == SessionState::idle);
            session->cancel();  // No stale cancellation source affects reuse.
            const auto reused = session->generate("new cancellation epoch");
            REQUIRE(reused.ok());
            CHECK(reused->outcome == RequestOutcome::completed);
            CHECK(session->requests_started() == 2);
        }
    }
}

TEST_CASE("callback exception cleanup also works without scheduling") {
    EngineOptions options;
    options.scheduling.enabled = false;
    options.telemetry.level = TelemetryLevel::off;
    Engine engine(options);
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    ModelLoadOptions load;
    load.model = "mock:tiny";
    auto model = engine.load_model(kMockBackendName, load);
    REQUIRE(model.ok());
    SessionOptions session_options;
    session_options.sampling = SamplingConfig::greedy(5);
    auto created = engine.create_session(model.value(), session_options);
    REQUIRE(created.ok());
    auto session = created.value();
    REQUIRE_FALSE(engine.scheduling_active());
    for (const int throw_at : {1, 3}) {
        CAPTURE(throw_at);
        const void* thrown = nullptr;
        int calls = 0;
        bool caught = false;
        try {
            (void)session->generate("unscheduled", [&](const TokenChunk&) {
                if (++calls == throw_at) throw_callback_marker(91, thrown);
                return true;
            });
        } catch (const int& marker) {
            caught = true;
            CHECK(static_cast<const void*>(&marker) == thrown);
            CHECK(marker == 91);
        }
        CHECK(caught);
        CHECK(calls == throw_at);
        CHECK(session->state() == SessionState::idle);
        CHECK(session->last_outcome() == RequestOutcome::failed);
        auto reused = session->generate("new unscheduled request");
        REQUIRE(reused.ok());
        CHECK(reused->outcome == RequestOutcome::completed);
        CHECK_FALSE(reused->scheduling.scheduled);
        CHECK(calls == throw_at);
    }
}

}
