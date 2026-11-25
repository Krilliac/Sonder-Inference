#include <doctest/doctest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../llamaserver_options.hpp"
#include "../llamaserver_protocol.hpp"
#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/engine.hpp"
#if defined(SONDER_HAS_SERVER)
#include <filesystem>
#include <fstream>
#include <system_error>

#include "server/src/openai.hpp"
#include "sonder/inference/backend_setup.hpp"
#endif

using namespace sonder::inference;

namespace {
struct PrefillCancellationObservation {
    bool loaded = false;
    bool entered = false;
    ErrorCode status = ErrorCode::internal;
    unsigned callbacks = 0;
    unsigned disconnects = 0;
};

PrefillCancellationObservation observe_prefill_cancellation(bool yield_before_cancel) {
    PrefillCancellationObservation observation;
    sonder_test::FakeLlamaServer server;
    server.set_body("data: {\"choices\":[{\"text\":\"x\",\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n");
    server.hold_prefill();
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.native_completion = false;
    auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
    observation.loaded = model.ok();
    if (!observation.loaded)
        return observation;
    CancellationSource source;
    std::exception_ptr canceller_error;
    std::jthread canceller([&] {
        try {
            observation.entered = server.wait_for_prefill(1);
            if (yield_before_cancel)
                std::this_thread::yield();
        } catch (...) {
            canceller_error = std::current_exception();
        }
        source.cancel();
    });
    auto result = model.value()->generate(GenerateRequest{"", "x", {}}, source.token(),
                                         [&](const TokenChunk &) {
                                             ++observation.callbacks;
                                             return true;
                                         });
    canceller.join();
    observation.status = result.status().code();
    // generate() has returned and destroyed its transport socket before the
    // fake provider resumes. No token payload was available before cancellation.
    server.hold_prefill(false);
    for (int i = 0; i < 100 && server.disconnects() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    observation.disconnects = server.disconnects();
    if (canceller_error)
        std::rethrow_exception(canceller_error);
    return observation;
}

void check_prefill_cancellation(const PrefillCancellationObservation &observation) {
    CHECK(observation.loaded);
    CHECK(observation.entered);
    CHECK(observation.status == ErrorCode::cancelled);
    CHECK(observation.callbacks == 0);
    CHECK(observation.disconnects == 1);
}
} // namespace

TEST_SUITE("llamaserver_backend") {

    TEST_CASE("timings preserve cache and optional speculative counters") {
        auto j = json::parse(
            R"({"timings":{"prompt_n":7,"cache_n":5,"predicted_n":9,"predicted_per_second":12.5,"prompt_ms":1.5,"predicted_ms":2.25}})");
        REQUIRE(j.ok());
        llamaserver::Timings t;
        REQUIRE(llamaserver::parse_timings(j.value(), t).ok());
        CHECK(t.prompt_tokens == 7);
        CHECK(t.cached_tokens == 5);
        CHECK(t.completion_tokens == 9);
        CHECK(t.prompt_present);
        CHECK(t.cache_present);
        CHECK(t.completion_present);
        CHECK(t.speed_present);
        CHECK(t.prompt_ns == 1500000);
        CHECK(t.eval_ns == 2250000);
        auto empty = json::parse(R"({"timings":{"prompt_n":0}})");
        REQUIRE(empty.ok());
        llamaserver::Timings z;
        REQUIRE(llamaserver::parse_timings(empty.value(), z).ok());
        CHECK(z.prompt_present);
        CHECK_FALSE(z.cache_present);
        CHECK_FALSE(z.draft_present);
    }

    TEST_CASE("cached /props slot count supplies admission capacity without rereading") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":1024},"total_slots":2})");
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        auto backend = make_llamaserver_backend(options);
        CHECK(backend->max_concurrent_requests() == 0);
        auto model = backend->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        const auto props_reads = server.props_requests();
        CHECK(backend->max_concurrent_requests() == 2);
        CHECK(server.props_requests() == props_reads);
    }

    TEST_CASE("transient /props failure is not cached and is refetched later") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":1024},"total_slots":2})");
        server.set_props_status(503);
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        auto backend = make_llamaserver_backend(options);
        REQUIRE(backend->load_model({"fake-model", "cpu:0"}).ok());
        CHECK(server.props_requests() >= 1);
        CHECK(backend->max_concurrent_requests() == 0);
        server.set_props_status(0);
        REQUIRE(backend->load_model({"fake-model", "cpu:0"}).ok());
        CHECK(backend->max_concurrent_requests() == 2);
    }

    TEST_CASE("single-slot or disabled affinity omits id_slot during overlapping keyed chats") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
                        "data: [DONE]\n\n");
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        SUBCASE("one slot with default affinity") {
            server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":1})");
        }
        SUBCASE("two slots with affinity disabled") {
            server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":2})");
            options.slot_affinity = false;
        }
        auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        ChatRequest request;
        request.messages = {{"user", "hi"}};
        request.session_key = "a";
        int callbacks = 0;
        auto result = model.value()->chat(request, {}, [&](const TokenChunk &) {
            ++callbacks;
            CHECK(model.value()->chat(request, {}, {}).ok());
            auto other = request;
            other.session_key = "b";
            CHECK(model.value()->chat(other, {}, {}).ok());
            return true;
        });
        REQUIRE(result.ok());
        CHECK(callbacks == 1);
        const auto bodies = server.bodies();
        REQUIRE(bodies.size() == 3);
        for (const auto &body : bodies) {
            const auto parsed = json::parse(body);
            REQUIRE(parsed.ok());
            CHECK(parsed->find("id_slot") == nullptr);
            REQUIRE(parsed->find("cache_prompt"));
            CHECK(parsed->find("cache_prompt")->as_bool());
        }
    }

    TEST_CASE("sampling emits only explicitly selected values") {
        GenerateRequest req;
        req.prompt = "hello";
        auto omitted = llamaserver::build_completion_body("m", req, true);
        CHECK(omitted.find("temperature") == nullptr);
        CHECK(omitted.find("n_predict") == nullptr);
        req.sampling.temperature = 0.8f;
        req.sampling.explicit_fields |= SamplingConfig::kTemperature;
        req.sampling.max_tokens = 12;
        auto explicit_default = llamaserver::build_completion_body("m", req, true);
        REQUIRE(explicit_default.find("temperature"));
        CHECK(explicit_default.find("temperature")->as_double() == doctest::Approx(0.8));
        REQUIRE(explicit_default.find("n_predict"));
        CHECK(explicit_default.find("n_predict")->as_int() == 12);
        req.sampling = SamplingConfig::greedy();
        req.sampling.explicit_only = true;
        auto parsed_omission = llamaserver::build_completion_body("m", req, true);
        CHECK(parsed_omission.find("seed") == nullptr);
        CHECK(parsed_omission.find("temperature") == nullptr);
        CHECK(parsed_omission.find("n_predict") == nullptr);
    }

#if defined(SONDER_HAS_SERVER)
    TEST_CASE("incoming HTTP sampling defaults are not forwarded as explicit settings") {
        namespace api = sonder::inference::server::detail;
        auto parsed = api::parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE(std::holds_alternative<api::ChatJob>(parsed));
        const auto &job = std::get<api::ChatJob>(parsed);
        CHECK(job.sampling.max_tokens == api::kDefaultMaxTokens);
        ChatRequest request;
        request.messages = job.messages;
        request.sampling = job.sampling;
        auto body = llamaserver::build_chat_body("fake-model", request);
        CHECK(body.find("max_tokens") == nullptr);
        CHECK(body.find("temperature") == nullptr);
        parsed = api::parse_chat_request(
            R"({"messages":[{"role":"user","content":"hi"}],"temperature":0.8,"max_completion_tokens":256,"stop":[]})");
        REQUIRE(std::holds_alternative<api::ChatJob>(parsed));
        request.sampling = std::get<api::ChatJob>(parsed).sampling;
        body = llamaserver::build_chat_body("fake-model", request);
        REQUIRE(body.find("temperature"));
        REQUIRE(body.find("max_tokens"));
        REQUIRE(body.find("stop"));
        CHECK(body.find("max_tokens")->as_uint() == 256);
        CHECK(body.find("top_p") == nullptr);
    }
#endif

    TEST_CASE("logit bias uses the native array and OpenAI object forms") {
        GenerateRequest request;
        request.sampling.logit_bias = {{7, 2.5f}, {9, -std::numeric_limits<float>::infinity()}};
        const auto native = llamaserver::build_completion_body("m", request, true);
        REQUIRE(native.find("logit_bias"));
        const auto &values = native.find("logit_bias")->as_array();
        REQUIRE(values.size() == 2);
        CHECK(values[0].as_array()[1].as_double() == 2.5);
        CHECK(values[1].as_array()[1].is_bool());
        CHECK_FALSE(values[1].as_array()[1].as_bool());
        const auto openai = llamaserver::build_completion_body("m", request, false);
        CHECK(openai.find("logit_bias")->find("9")->as_double() == -100.0);
    }

    TEST_CASE("HTTP observations reach Session telemetry without claiming logical cache reuse") {
        sonder_test::FakeLlamaServer server;
        server.set_body(
            "event: message\r\nid: 1\r\n: heartbeat\r\n"
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"length\"}],\n"
            "data: "
            "\"timings\":{\"prompt_n\":7,\"cache_n\":5,\"predicted_n\":2,\"prompt_ms\":1.5,\"predicted_ms\":"
            "2.25,\"predicted_per_second\":888.5,\"draft_n\":4,\"draft_n_accepted\":2}}\n\n"
            "data: "
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2,\"prompt_tokens_"
            "details\":{\"cached_tokens\":5}}}\n\n"
            "data: [DONE]\n\n");
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions options;
        options.telemetry_sinks.push_back(sink);
        options.scheduling.enabled = false;
        Engine engine(options);
        LlamaServerBackendOptions upstream;
        upstream.base_url = server.url();
        REQUIRE(engine.register_backend(make_llamaserver_backend(upstream)).ok());
        auto model = engine.load_model("llamaserver", ModelLoadOptions{"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        auto session = engine.create_session(model.value(), {});
        REQUIRE(session.ok());
        auto result = session.value()->chat({{"user", "hello"}});
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(result->text == "ok");
        CHECK(result->stats.prompt_tokens == 12);
        CHECK(result->stats.cached_tokens == 5);
        CHECK(result->stats.draft_tokens == 4);
        CHECK(result->stats.draft_accepted_tokens == 2);
        CHECK(result->stats.prompt_eval_ns == 1500000);
        CHECK(result->stats.eval_ns == 2250000);
        CHECK(result->stats.stop_reason == StopReason::max_tokens);
        CHECK_FALSE(result->scheduling.scheduled);
        engine.telemetry().flush();
        bool summary_seen = false, prefill_seen = false, decode_seen = false;
        for (const auto &line : sink->lines()) {
            auto event = json::parse(line);
            REQUIRE(event.ok());
            const auto type = event->find("event_type")->as_string();
            const auto *attrs = event->find("attributes");
            if (type == "request.completed" || type == "inference.decode.completed") {
                REQUIRE(attrs->find("backend_draft_acceptance_ratio"));
                CHECK(attrs->find("backend_draft_acceptance_ratio")->as_double() == 0.5);
                CHECK(attrs->find("backend_cached_tokens")->as_uint() == 5);
                CHECK(attrs->find("backend_predicted_tokens_per_second")->as_double() == 888.5);
                if (type == "request.completed")
                    summary_seen = true;
                else
                    decode_seen = true;
            }
            if (type == "inference.prefill.completed") {
                REQUIRE(attrs->find("backend_cached_tokens"));
                CHECK(attrs->find("backend_cached_tokens")->as_uint() == 5);
                prefill_seen = true;
            }
        }
        CHECK(summary_seen);
        CHECK(prefill_seen);
        CHECK(decode_seen);
    }

    TEST_CASE("zero observations remain present while missing counters stay absent") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: "
                        "{\"content\":\"ok\",\"stop\":true,\"timings\":{\"prompt_n\":7,\"cache_n\":5,"
                        "\"predicted_n\":2,\"draft_n\":0,\"draft_n_accepted\":0}}\n\n");
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        auto result = model.value()->generate(GenerateRequest{"", "hello", {}}, {}, {});
        REQUIRE(result.ok());
        CHECK(result->prompt_tokens == 12);
        CHECK(result->draft_tokens == 0);
        CHECK(result->draft_accepted_tokens == 0);
        CHECK_FALSE(result->predicted_tokens_per_second.has_value());
        server.set_body("data: {\"content\":\"ok\",\"stop\":true,\"timings\":{\"prompt_n\":0}}\n\n");
        result = model.value()->generate(GenerateRequest{"", "hello", {}}, {}, {});
        REQUIRE(result.ok());
        CHECK_FALSE(result->cached_tokens.has_value());
        CHECK_FALSE(result->draft_tokens.has_value());
    }

    TEST_CASE("tool call completion fails explicitly and cannot turn length into a tool call") {
        sonder_test::FakeLlamaServer server;
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        ChatRequest request;
        request.messages = {{"user", "hi"}};
        server.set_body("data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n");
        CHECK(model.value()->chat(request, {}, {}).status().code() == ErrorCode::unsupported);
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"length\"}]}\n\n");
        auto result = model.value()->chat(request, {}, {});
        REQUIRE(result.ok());
        CHECK(result->stop_reason == StopReason::max_tokens);
    }

    TEST_CASE("OpenAI chat streaming maps text and finish reasons") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\r\n\r\n"
                        ":keepalive\r\n"
                        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
                        "data: {\"choices\":[{\"finish_reason\":\"length\"}]}\n\n"
                        "data: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n"
                        "data: [DONE]\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto backend = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto model = backend->load_model(lo);
        REQUIRE(model.ok());
        ChatRequest req;
        req.messages = {{"user", "hi"}};
        req.sampling = SamplingConfig::greedy(8);
        std::string text;
        auto result = model.value()->chat(req, {}, [&](const TokenChunk &c) {
            text += c.text;
            return true;
        });
        REQUIRE(result.ok());
        CHECK(text == "Hello");
        CHECK(result.value().stop_reason == StopReason::max_tokens);
        CHECK(result.value().prompt_tokens == 3);
        CHECK(result.value().completion_tokens == 2);
    }

    TEST_CASE("native completion maps stop bool and generic completion text") {
        sonder_test::FakeLlamaServer server;
        server.set_body(
            R"({"content":"ok","stop":true,"stop_type":"limit","tokens_evaluated":4,"tokens_predicted":2})");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto backend = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto model = backend->load_model(lo);
        REQUIRE(model.ok());
        GenerateRequest req;
        req.prompt = "x";
        std::string out;
        auto r = model.value()->generate(req, {}, [&](const TokenChunk &c) {
            out += c.text;
            return true;
        });
        REQUIRE(r.ok());
        CHECK(out == "ok");
        CHECK(r.value().stop_reason == StopReason::max_tokens);
        CHECK(r.value().prompt_tokens == 4);
    }

    TEST_CASE("generic completions use choices text and usage") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"text\":\"gen\"}]}\n\n"
                        "data: "
                        "{\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,"
                        "\"completion_tokens\":1}}\n\n"
                        "data: [DONE]\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        std::string out;
        auto r = m.value()->generate(q, {}, [&](const TokenChunk &c) {
            out += c.text;
            return true;
        });
        REQUIRE(r.ok());
        CHECK(out == "gen");
        CHECK(r.value().prompt_tokens == 4);
        CHECK(r.value().stop_reason == StopReason::end_of_sequence);
    }

    TEST_CASE("callback cancellation and reentrant probe are safe") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        ChatRequest q;
        q.messages = {{"user", "x"}};
        auto r = m.value()->chat(q, {}, [&](const TokenChunk &) {
            CHECK(b->probe().ok());
            return false;
        });
        REQUIRE(r.ok());
        CHECK(r.value().stop_reason == StopReason::callback);
        CancellationSource source;
        source.cancel();
        CHECK(m.value()->generate(GenerateRequest{"", "x", {}}, source.token(), {}).status().code() ==
              ErrorCode::cancelled);
    }

    TEST_CASE("cancellation during upstream prefill closes the response") {
        check_prefill_cancellation(observe_prefill_cancellation(false));
    }

    TEST_CASE("prefill cancellation closes responses under serial and parallel repetition") {
        for (int i = 0; i < 20; ++i)
            check_prefill_cancellation(observe_prefill_cancellation(i % 2 != 0));
        std::array<std::array<PrefillCancellationObservation, 8>, 2> observations{};
        std::array<std::exception_ptr, 2> worker_errors{};
        std::array<std::jthread, 2> workers;
        for (std::size_t worker = 0; worker < workers.size(); ++worker)
            workers[worker] = std::jthread([&, worker] {
                try {
                    for (std::size_t i = 0; i < observations[worker].size(); ++i)
                        observations[worker][i] = observe_prefill_cancellation(i % 2 != 0);
                } catch (...) {
                    worker_errors[worker] = std::current_exception();
                }
            });
        for (auto &worker : workers)
            worker.join();
        for (const auto &error : worker_errors)
            if (error)
                std::rethrow_exception(error);
        for (const auto &worker : observations)
            for (const auto &observation : worker)
                check_prefill_cancellation(observation);
    }

    TEST_CASE("held prefill observes a live peer without reporting cancellation") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"text\":\"x\",\"finish_reason\":\"stop\"}]}\n\n"
                        "data: [DONE]\n\n");
        server.hold_prefill();
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        auto m = b->load_model({"fake-model", "cpu:0"});
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        bool succeeded = false;
        std::string text;
        std::exception_ptr client_error;
        std::jthread client([&] {
            try {
                succeeded = m.value()->generate(q, {}, [&](const TokenChunk &chunk) {
                    text += chunk.text;
                    return true;
                }).ok();
            } catch (...) {
                client_error = std::current_exception();
            }
        });
        const bool entered = server.wait_for_prefill(1);
        const auto disconnects_before_release = server.disconnects();
        server.hold_prefill(false);
        client.join();
        if (client_error)
            std::rethrow_exception(client_error);
        CHECK(entered);
        CHECK(disconnects_before_release == 0);
        CHECK(server.disconnects() == 0);
        CHECK(succeeded);
        CHECK(text == "x");
    }

    TEST_CASE("pre-cancelled generation never enters held prefill") {
        sonder_test::FakeLlamaServer server;
        server.hold_prefill();
        LlamaServerBackendOptions options;
        options.base_url = server.url();
        options.native_completion = false;
        auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        CancellationSource source;
        source.cancel();
        unsigned callbacks = 0;
        auto result = model.value()->generate(GenerateRequest{"", "x", {}}, source.token(),
                                             [&](const TokenChunk &) {
                                                 ++callbacks;
                                                 return true;
                                             });
        CHECK(result.status().code() == ErrorCode::cancelled);
        CHECK(callbacks == 0);
        CHECK(server.bodies().empty());
        CHECK(server.prefill_entries() == 0);
        CHECK(server.disconnects() == 0);
    }

    TEST_CASE("malformed trailing and oversized SSE frames fail closed") {
        sonder_test::FakeLlamaServer server;
        server.set_body(
            "data: {\"choices\":[{\"text\":\"ok\",\"finish_reason\":\"stop\"}]}\n\ndata: {bad}\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        CHECK(m.value()->generate(q, {}, {}).status().code() == ErrorCode::protocol_error);
        server.set_body("data: " + std::string(4u * 1024u * 1024u + 1u, 'x') + "\n\n");
        CHECK(m.value()->generate(q, {}, {}).status().code() == ErrorCode::protocol_error);
    }

    TEST_CASE("HTTP errors never deliver token-looking bodies") {
        sonder_test::FakeLlamaServer server;
        server.set_status(500);
        server.set_body("data: {\"content\":\"must not deliver\",\"stop\":true}\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        int chunks = 0;
        GenerateRequest q;
        q.prompt = "x";
        auto r = m.value()->generate(q, {}, [&](const TokenChunk &) {
            ++chunks;
            return true;
        });
        CHECK_FALSE(r.ok());
        CHECK(chunks == 0);
        CHECK(r.status().code() == ErrorCode::backend_error);
    }

    TEST_CASE("num_ctx beyond the served context fails before the completion request") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":1024},"total_slots":1})");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        q.sampling.num_ctx = 2048;
        const auto r = m.value()->generate(q, {}, {});
        CHECK(r.status().code() == ErrorCode::invalid_argument);
        CHECK(server.last_body().empty());
    }

    TEST_CASE("remote plain HTTP and slot path traversal are rejected") {
        LlamaServerBackendOptions remote;
        remote.base_url = "http://192.0.2.10:8080";
        remote.allow_remote = true;
        auto b = make_llamaserver_backend(remote);
        CHECK(b->probe().status().code() == ErrorCode::invalid_argument);
        sonder_test::FakeLlamaServer server;
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto local = make_llamaserver_backend(o);
        CHECK(local->save_slot(1, "../bad").code() == ErrorCode::invalid_argument);
        CHECK(local->save_slot(1, "ok.bin").ok());
    }

    TEST_CASE("missing SSE termination is a protocol error") {
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        ModelLoadOptions lo;
        lo.model = "fake-model";
        auto m = b->load_model(lo);
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        CHECK(m.value()->generate(q, {}, {}).status().code() == ErrorCode::protocol_error);
    }

    TEST_CASE("llamaserver uma_mode JSON parser accepts booleans and strings and rejects invalid values") {
        std::string mode;
        auto parsed = json::parse("true");
        REQUIRE(parsed.ok());
        REQUIRE(llamaserver::parse_uma_mode_json(parsed.value(), mode).ok());
        CHECK(mode == "on");

        parsed = json::parse("false");
        REQUIRE(parsed.ok());
        REQUIRE(llamaserver::parse_uma_mode_json(parsed.value(), mode).ok());
        CHECK(mode == "off");

        const std::pair<const char *, const char *> valid_strings[] = {
            {"\"auto\"", "auto"},
            {"\"on\"", "on"},
            {"\"off\"", "off"},
            {"\"true\"", "on"},
            {"\"false\"", "off"},
        };
        for (const auto &[raw, expected] : valid_strings) {
            parsed = json::parse(raw);
            REQUIRE(parsed.ok());
            REQUIRE(llamaserver::parse_uma_mode_json(parsed.value(), mode).ok());
            CHECK(mode == expected);
        }

        for (const auto *bad : {"\"\"", "\"yes\"", "\"no\"", "\"AUTO\"", "0", "1", "null", "[]", "{}"}) {
            parsed = json::parse(bad);
            REQUIRE(parsed.ok());
            const auto st = llamaserver::parse_uma_mode_json(parsed.value(), mode);
            CHECK_MESSAGE(st.code() == ErrorCode::invalid_argument, bad);
        }
    }

    TEST_CASE("llamaserver resolve_uma_mode handles discrete, mixed, APU-only, and headless inventories") {
        constexpr std::uint64_t kMiB = 1024ull * 1024ull;
        constexpr std::uint64_t kGiB = 1024ull * kMiB;

        GpuDeviceInfo rtx;
        rtx.id = "gpu:0";
        rtx.name = "NVIDIA GeForce RTX 5070 Ti";
        rtx.vendor = GpuVendor::nvidia;
        rtx.vendor_id = 0x10DEu;
        rtx.dedicated_total = MemoryReading::known(16303 * kMiB);
        rtx.shared_total = MemoryReading::known(16 * kGiB);
        rtx.is_integrated_uma = false;
        rtx.backend_hint = "cuda";

        GpuDeviceInfo ryzen_780m;
        ryzen_780m.id = "gpu:1";
        ryzen_780m.name = "AMD Radeon 780M Graphics";
        ryzen_780m.vendor = GpuVendor::amd;
        ryzen_780m.vendor_id = 0x1002u;
        ryzen_780m.dedicated_total = MemoryReading::known(512 * kMiB);
        ryzen_780m.shared_total = MemoryReading::known(16 * kGiB);
        ryzen_780m.is_integrated_uma = true;
        ryzen_780m.backend_hint = "vulkan";

        GpuDeviceInfo rx_9060_xt;
        rx_9060_xt.id = "gpu:0";
        rx_9060_xt.name = "AMD Radeon RX 9060 XT";
        rx_9060_xt.vendor = GpuVendor::amd;
        rx_9060_xt.vendor_id = 0x1002u;
        rx_9060_xt.dedicated_total = MemoryReading::known(16384 * kMiB);
        rx_9060_xt.shared_total = MemoryReading::known(32 * kGiB);
        rx_9060_xt.is_integrated_uma = false;
        rx_9060_xt.backend_hint = "vulkan";

        GpuDeviceInfo rx_9060_rocm = rx_9060_xt;
        rx_9060_rocm.id = "gpu:1";
        rx_9060_rocm.backend_hint = "rocm";
        rx_9060_rocm.rocm_runtime_ordinal = 0u;

        GpuDeviceInfo rx_second = rx_9060_xt;
        rx_second.id = "gpu:1";

        GpuDeviceInfo apu_zero = ryzen_780m;
        apu_zero.id = "gpu:0";
        GpuDeviceInfo rtx_one = rtx;
        rtx_one.id = "gpu:1";

        const std::vector<GpuDeviceInfo> rtx_only{rtx};
        const std::vector<GpuDeviceInfo> dual_discrete{rtx, rx_second};
        const std::vector<GpuDeviceInfo> mixed_rtx_first{rtx, ryzen_780m};
        const std::vector<GpuDeviceInfo> mixed_igpu_first{apu_zero, rtx_one};
        const std::vector<GpuDeviceInfo> node1_hybrid{rx_9060_xt, ryzen_780m};
        const std::vector<GpuDeviceInfo> node1_igpu_first{apu_zero, rx_9060_rocm};
        const std::vector<GpuDeviceInfo> apu_only{apu_zero};
        const std::vector<GpuDeviceInfo> empty_gpus;

        // Explicit modes
        CHECK(llamaserver::resolve_uma_mode("on", rtx_only));
        CHECK(llamaserver::resolve_uma_selection("on", rtx_only) == llamaserver::UmaSelectionMode::uma_only);
        CHECK(llamaserver::resolve_uma_mode("true", empty_gpus));
        CHECK_FALSE(llamaserver::resolve_uma_mode("off", apu_only));
        CHECK(llamaserver::resolve_uma_selection("off", apu_only) == llamaserver::UmaSelectionMode::discrete_only);
        CHECK_FALSE(llamaserver::resolve_uma_mode("false", apu_only));

        // Discrete RTX only and dual discrete -> discrete_only / false
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", rtx_only));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", rtx_only, "CUDA0"));
        CHECK(llamaserver::resolve_uma_selection("auto", dual_discrete, "Vulkan0,Vulkan1") ==
              llamaserver::UmaSelectionMode::discrete_only);
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", dual_discrete, "Vulkan0,Vulkan1"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", dual_discrete, "CUDA0,Vulkan1"));

        // Mixed RTX + Ryzen iGPU: default -> discrete_only, single iGPU -> uma_only, hybrid -> hybrid (true)
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_rtx_first));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_igpu_first));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "Vulkan0"));
        CHECK(llamaserver::resolve_uma_selection("auto", mixed_rtx_first, "Vulkan1") ==
              llamaserver::UmaSelectionMode::uma_only);
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "Vulkan1"));
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_igpu_first, "Vulkan0"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_igpu_first, "Vulkan1"));
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "ROCm0"));
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "gpu:1"));
        CHECK(llamaserver::resolve_uma_selection("auto", mixed_rtx_first, "uma") ==
              llamaserver::UmaSelectionMode::uma_only);
        CHECK(llamaserver::resolve_uma_selection("auto", mixed_rtx_first, "CUDA0,Vulkan1") ==
              llamaserver::UmaSelectionMode::hybrid);
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "CUDA0,Vulkan1"));
        CHECK(llamaserver::resolve_uma_selection("auto", mixed_rtx_first, "Vulkan0,Vulkan1") ==
              llamaserver::UmaSelectionMode::hybrid);
        CHECK(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "Vulkan0,Vulkan1"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "none"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, "Vulkan1,bogus"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", mixed_rtx_first, ","));

        // Node1 hybrid discrete RX 9060 XT + integrated Radeon 780M
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid) == llamaserver::UmaSelectionMode::discrete_only);
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", node1_hybrid));
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid, "Vulkan0") ==
              llamaserver::UmaSelectionMode::discrete_only);
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", node1_hybrid, "Vulkan0"));
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid, "Vulkan1") ==
              llamaserver::UmaSelectionMode::uma_only);
        CHECK(llamaserver::resolve_uma_mode("auto", node1_hybrid, "Vulkan1"));
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid, "Vulkan0,Vulkan1") ==
              llamaserver::UmaSelectionMode::hybrid);
        CHECK(llamaserver::resolve_uma_mode("auto", node1_hybrid, "Vulkan0,Vulkan1"));
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid, "Vulkan1, Vulkan0") ==
              llamaserver::UmaSelectionMode::hybrid);
        CHECK(llamaserver::resolve_uma_selection("auto", node1_hybrid, "gpu:0,gpu:1") ==
              llamaserver::UmaSelectionMode::hybrid);
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", node1_hybrid, "Vulkan0,Vulkan2"));

        // Explicit ROCm runtime ordinal selects discrete AMD card over integrated iGPU
        CHECK(llamaserver::resolve_uma_selection("auto", node1_igpu_first, "ROCm0") ==
              llamaserver::UmaSelectionMode::discrete_only);
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", node1_igpu_first, "ROCm0"));
        CHECK(llamaserver::resolve_uma_selection("auto", node1_igpu_first, "Vulkan1,Vulkan0") ==
              llamaserver::UmaSelectionMode::hybrid);

        // AMD Ryzen 780M APU-only -> uma_only / true
        CHECK(llamaserver::resolve_uma_selection("auto", apu_only) == llamaserver::UmaSelectionMode::uma_only);
        CHECK(llamaserver::resolve_uma_mode("auto", apu_only));
        CHECK(llamaserver::resolve_uma_mode("auto", apu_only, "Vulkan0"));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", apu_only, "CUDA0"));

        // Empty GPU list (headless CI) -> discrete_only / false
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", empty_gpus));
        CHECK_FALSE(llamaserver::resolve_uma_mode("auto", empty_gpus, "Vulkan0"));
    }

    TEST_CASE("llamaserver spill_guard mapping propagates uma_mode and validates uma_mode_policy") {
        GpuDeviceInfo rtx;
        rtx.id = "gpu:0";
        rtx.vendor = GpuVendor::nvidia;
        rtx.is_integrated_uma = false;
        rtx.backend_hint = "cuda";

        GpuDeviceInfo apu;
        apu.id = "gpu:1";
        apu.vendor = GpuVendor::amd;
        apu.is_integrated_uma = true;
        apu.backend_hint = "vulkan";

        const std::vector<GpuDeviceInfo> mixed{rtx, apu};
        const std::vector<GpuDeviceInfo> apu_only{apu};

        // Omitted uma_mode keeps both flags false for backwards compatibility.
        {
            LlamaServerSpillGuardOptions o;
            const auto mapped = llamaserver::to_spill_guard_options(o, {}, apu_only);
            CHECK_FALSE(mapped.uma_mode);
            CHECK_FALSE(mapped.residency.uma_mode);
        }

        // Boolean spill_guard.uma_mode = true propagates to residency unless explicitly overridden.
        {
            LlamaServerSpillGuardOptions o;
            o.uma_mode = true;
            const auto propagated = llamaserver::to_spill_guard_options(o, {}, mixed);
            CHECK(propagated.uma_mode);
            CHECK(propagated.residency.uma_mode);

            o.residency.uma_mode = false;
            o.residency_uma_mode_explicit = true;
            const auto overridden = llamaserver::to_spill_guard_options(o, {}, mixed);
            CHECK(overridden.uma_mode);
            CHECK_FALSE(overridden.residency.uma_mode);
        }

        // Policy "auto" honors --device / -dev hint in args and propagates consistently.
        {
            LlamaServerSpillGuardOptions o;
            o.uma_mode_policy = "auto";
            const auto on_default = llamaserver::to_spill_guard_options(o, {}, mixed);
            CHECK_FALSE(on_default.uma_mode);
            CHECK_FALSE(on_default.hybrid_uma_mode);
            CHECK_FALSE(on_default.residency.uma_mode);

            const auto on_vk1 = llamaserver::to_spill_guard_options(o, {"--device", "Vulkan1"}, mixed);
            CHECK(on_vk1.uma_mode);
            CHECK_FALSE(on_vk1.hybrid_uma_mode);
            CHECK(on_vk1.residency.uma_mode);

            const auto on_hybrid = llamaserver::to_spill_guard_options(o, {"--device", "Vulkan0,Vulkan1"}, mixed);
            CHECK(on_hybrid.uma_mode);
            CHECK(on_hybrid.hybrid_uma_mode);
            CHECK(on_hybrid.residency.uma_mode);

            LlamaServerSpillGuardOptions hybrid_override = o;
            hybrid_override.residency.uma_mode = false;
            hybrid_override.residency_uma_mode_explicit = true;
            const auto on_hybrid_override =
                llamaserver::to_spill_guard_options(hybrid_override, {"--device", "Vulkan0,Vulkan1"}, mixed);
            CHECK(on_hybrid_override.uma_mode);
            CHECK(on_hybrid_override.hybrid_uma_mode);
            CHECK_FALSE(on_hybrid_override.residency.uma_mode);

            const auto on_dev_eq = llamaserver::to_spill_guard_options(o, {"-dev=Vulkan1"}, mixed);
            CHECK(on_dev_eq.uma_mode);
            CHECK_FALSE(on_dev_eq.hybrid_uma_mode);
            CHECK(on_dev_eq.residency.uma_mode);

            const auto on_apu = llamaserver::to_spill_guard_options(o, {}, apu_only);
            CHECK(on_apu.uma_mode);
            CHECK_FALSE(on_apu.hybrid_uma_mode);
            CHECK(on_apu.residency.uma_mode);

            // Last --device wins, and a dangling --device clears the hint (discrete RTX wins).
            const std::vector<std::string> multi_dev{"--device", "CUDA0", "--device", "Vulkan1"};
            const std::vector<std::string> dangling_dev{"--device", "Vulkan1", "--device"};
            CHECK(llamaserver::extract_device_hint(multi_dev) == "Vulkan1");
            CHECK(llamaserver::extract_device_hint(dangling_dev).empty());
        }

        // validate_backend_options accepts valid uma_mode_policy and rejects invalid ones.
        {
            LlamaServerBackendOptions spawn;
            spawn.mode = LlamaServerMode::spawn;
            spawn.executable = "llama-server";
            spawn.spill_guard.uma_mode_policy = "auto";
            CHECK(llamaserver::validate_backend_options(spawn).ok());

            spawn.spill_guard.uma_mode_policy = "invalid";
            CHECK(llamaserver::validate_backend_options(spawn).code() == ErrorCode::invalid_argument);
        }
    }

    TEST_CASE("protocol spec_draft_n_max is omitted by default and serialized only in native mode") {
        GenerateRequest gen_req;
        gen_req.prompt = "hello";
        ChatRequest chat_req;
        chat_req.messages = {{"user", "hello"}};

        // Omitted by default in every mode.
        CHECK(llamaserver::build_completion_body("m", gen_req, true).find("speculative.n_max") == nullptr);
        CHECK(llamaserver::build_completion_body("m", gen_req, false).find("speculative.n_max") == nullptr);
        CHECK(llamaserver::build_chat_body("m", chat_req).find("speculative.n_max") == nullptr);

        // Native completion emits speculative.n_max (including 0 when paused).
        llamaserver::RequestExtras native_extras;
        native_extras.cache_prompt = true;
        native_extras.spec_draft_n_max = 4u;
        const auto native_comp = llamaserver::build_completion_body("m", gen_req, true, {}, native_extras);
        REQUIRE(native_comp.find("speculative.n_max") != nullptr);
        CHECK(native_comp.find("speculative.n_max")->as_int() == 4);

        native_extras.spec_draft_n_max = 0u;
        const auto paused_comp = llamaserver::build_completion_body("m", gen_req, true, {}, native_extras);
        REQUIRE(paused_comp.find("speculative.n_max") != nullptr);
        CHECK(paused_comp.find("speculative.n_max")->as_int() == 0);

        // Non-native completion omits speculative.n_max even when populated on RequestExtras.
        native_extras.spec_draft_n_max = 4u;
        CHECK(llamaserver::build_completion_body("m", gen_req, false, {}, native_extras).find("speculative.n_max") ==
              nullptr);

        // Native chat emits speculative.n_max; generic chat omits it.
        native_extras.spec_draft_n_max = 5u;
        const auto native_chat = llamaserver::build_chat_body("m", chat_req, {}, native_extras);
        REQUIRE(native_chat.find("speculative.n_max") != nullptr);
        CHECK(native_chat.find("speculative.n_max")->as_int() == 5);

        native_extras.spec_draft_n_max = 0u;
        const auto paused_chat = llamaserver::build_chat_body("m", chat_req, {}, native_extras);
        REQUIRE(paused_chat.find("speculative.n_max") != nullptr);
        CHECK(paused_chat.find("speculative.n_max")->as_int() == 0);

        llamaserver::RequestExtras generic_extras;
        generic_extras.generic_thinking = true;
        generic_extras.spec_draft_n_max = 5u;
        CHECK(llamaserver::build_chat_body("m", chat_req, {}, generic_extras).find("speculative.n_max") == nullptr);
        CHECK(llamaserver::build_completion_body("m", gen_req, true, {}, generic_extras).find("speculative.n_max") ==
              nullptr);
    }

    TEST_CASE("adaptive speculation options map faithfully and validate only when enabled") {
        LlamaServerAdaptiveSpeculationOptions defaults;
        CHECK_FALSE(defaults.enabled);
        CHECK(defaults.min_draft_n == 0u);
        CHECK(defaults.initial_draft_n == 3u);
        CHECK(defaults.max_draft_n == 6u);
        CHECK(defaults.window_size == 16u);
        CHECK(defaults.min_samples == 4u);
        CHECK(defaults.pause_acceptance_threshold == doctest::Approx(0.25));
        CHECK(defaults.step_down_acceptance_threshold == doctest::Approx(0.45));
        CHECK(defaults.step_up_acceptance_threshold == doctest::Approx(0.72));
        CHECK(defaults.cooldown_requests == 8u);
        CHECK(defaults.probe_draft_n == 1u);
        CHECK(defaults.min_draft_tokens_per_sample == 4u);

        LlamaServerAdaptiveSpeculationOptions custom;
        custom.enabled = true;
        custom.min_draft_n = 1;
        custom.initial_draft_n = 2;
        custom.max_draft_n = 8;
        custom.window_size = 32;
        custom.min_samples = 6;
        custom.pause_acceptance_threshold = 0.20;
        custom.step_down_acceptance_threshold = 0.40;
        custom.step_up_acceptance_threshold = 0.80;
        custom.cooldown_requests = 12;
        custom.probe_draft_n = 2;
        custom.min_draft_tokens_per_sample = 8;
        const auto mapped = llamaserver::to_adaptive_speculation_config(custom);
        CHECK(mapped.enabled == custom.enabled);
        CHECK(mapped.min_draft_n == custom.min_draft_n);
        CHECK(mapped.initial_draft_n == custom.initial_draft_n);
        CHECK(mapped.max_draft_n == custom.max_draft_n);
        CHECK(mapped.window_size == custom.window_size);
        CHECK(mapped.min_samples == custom.min_samples);
        CHECK(mapped.pause_acceptance_threshold == doctest::Approx(custom.pause_acceptance_threshold));
        CHECK(mapped.step_down_acceptance_threshold == doctest::Approx(custom.step_down_acceptance_threshold));
        CHECK(mapped.step_up_acceptance_threshold == doctest::Approx(custom.step_up_acceptance_threshold));
        CHECK(mapped.cooldown_requests == custom.cooldown_requests);
        CHECK(mapped.probe_draft_n == custom.probe_draft_n);
        CHECK(mapped.min_draft_tokens_per_sample == custom.min_draft_tokens_per_sample);

        // Disabled adaptive speculation stays inert even if bounds are invalid or in generic mode.
        LlamaServerBackendOptions inert;
        inert.adaptive_speculation.enabled = false;
        inert.adaptive_speculation.max_draft_n = 0;
        CHECK(llamaserver::validate_backend_options(inert).ok());
        inert.native_completion = false;
        CHECK(llamaserver::validate_backend_options(inert).ok());

        // Enabled adaptive speculation validates ranges, thresholds, and native mode strictly.
        LlamaServerBackendOptions valid;
        valid.adaptive_speculation = custom;
        CHECK(llamaserver::validate_backend_options(valid).ok());

        {
            LlamaServerBackendOptions generic_enabled;
            generic_enabled.native_completion = false;
            generic_enabled.adaptive_speculation.enabled = true;
            CHECK(llamaserver::validate_backend_options(generic_enabled).code() == ErrorCode::invalid_argument);
        }

        const auto expect_invalid = [](LlamaServerBackendOptions bad) {
            bad.adaptive_speculation.enabled = true;
            CHECK(llamaserver::validate_backend_options(bad).code() == ErrorCode::invalid_argument);
        };
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.max_draft_n = 0;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.min_draft_n = 7;
            b.adaptive_speculation.max_draft_n = 6;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.initial_draft_n = 7;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.probe_draft_n = 0;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.window_size = 0;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.min_samples = b.adaptive_speculation.window_size + 1;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.pause_acceptance_threshold = 0.60;
            b.adaptive_speculation.step_down_acceptance_threshold = 0.40;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.step_down_acceptance_threshold = 0.85;
            b.adaptive_speculation.step_up_acceptance_threshold = 0.70;
            expect_invalid(b);
        }
        {
            LlamaServerBackendOptions b;
            b.adaptive_speculation.pause_acceptance_threshold = std::numeric_limits<double>::quiet_NaN();
            expect_invalid(b);
        }
    }

    TEST_CASE("adaptive speculation live loopback adjusts draft_n_max cancels failed probes and reports runtime diagnostics") {
        sonder_test::FakeLlamaServer server;
        const auto last_spec_n_max = [&server]() -> std::optional<std::int64_t> {
            const auto parsed = json::parse(server.last_body());
            REQUIRE(parsed.ok());
            const auto *v = parsed->find("speculative.n_max");
            if (!v)
                return std::nullopt;
            return v->as_int(-1);
        };

        const std::string high_chat =
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
            "\"timings\":{\"prompt_n\":4,\"predicted_n\":20,\"predicted_ms\":200.0,"
            "\"draft_n\":20,\"draft_n_accepted\":18}}\n\n"
            "data: [DONE]\n\n";
        const std::string low_gen =
            "data: {\"content\":\"ok\",\"stop\":true,"
            "\"timings\":{\"prompt_n\":4,\"predicted_n\":10,\"predicted_ms\":200.0,"
            "\"draft_n\":20,\"draft_n_accepted\":2}}\n\n";
        const std::string paused_gen =
            "data: {\"content\":\"ok\",\"stop\":true,"
            "\"timings\":{\"prompt_n\":4,\"predicted_n\":10,\"predicted_ms\":200.0,"
            "\"draft_n\":0,\"draft_n_accepted\":0}}\n\n";
        const std::string high_gen =
            "data: {\"content\":\"ok\",\"stop\":true,"
            "\"timings\":{\"prompt_n\":4,\"predicted_n\":20,\"predicted_ms\":250.0,"
            "\"draft_n\":20,\"draft_n_accepted\":16}}\n\n";

        // Default (disabled): omits speculative.n_max and runtime_status diagnostics entry.
        {
            server.set_body(high_gen);
            LlamaServerBackendOptions disabled_opts;
            disabled_opts.base_url = server.url();
            auto backend = make_llamaserver_backend(disabled_opts);
            auto model = backend->load_model({"fake-model", "cpu:0"});
            REQUIRE(model.ok());
            REQUIRE(model.value()->generate(GenerateRequest{"", "hello", {}}, {}, {}).ok());
            CHECK_FALSE(last_spec_n_max().has_value());
            if (const auto rt = backend->runtime_status(); rt && rt->diagnostics)
                CHECK(rt->diagnostics->find("adaptive_speculation") == nullptr);
        }

        // Generic mode (native_completion = false) with adaptive_speculation.enabled = true is rejected.
        {
            LlamaServerBackendOptions generic_opts;
            generic_opts.base_url = server.url();
            generic_opts.native_completion = false;
            generic_opts.adaptive_speculation.enabled = true;
            CHECK(llamaserver::validate_backend_options(generic_opts).code() == ErrorCode::invalid_argument);
            auto backend = make_llamaserver_backend(generic_opts);
            CHECK(backend->load_model({"fake-model", "cpu:0"}).status().code() == ErrorCode::invalid_argument);
            CHECK_FALSE(backend->runtime_status().has_value());
        }

        // Native mode with adaptive speculation enabled: live feedback loop.
        LlamaServerBackendOptions opts;
        opts.base_url = server.url();
        opts.adaptive_speculation.enabled = true;
        opts.adaptive_speculation.min_draft_n = 0;
        opts.adaptive_speculation.initial_draft_n = 3;
        opts.adaptive_speculation.max_draft_n = 5;
        opts.adaptive_speculation.window_size = 4;
        opts.adaptive_speculation.min_samples = 2;
        opts.adaptive_speculation.pause_acceptance_threshold = 0.25;
        opts.adaptive_speculation.step_down_acceptance_threshold = 0.45;
        opts.adaptive_speculation.step_up_acceptance_threshold = 0.72;
        opts.adaptive_speculation.cooldown_requests = 2;
        opts.adaptive_speculation.probe_draft_n = 1;
        opts.adaptive_speculation.min_draft_tokens_per_sample = 4;

        auto backend = make_llamaserver_backend(opts);
        auto model = backend->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());

        // 1. High-acceptance code_edit chat requests step up draft_n_max: 3 -> 4 -> 5.
        server.set_body(high_chat);
        ChatRequest code_req;
        code_req.messages = {{"user", "Update this block:\n```cpp\nint x = 1;\n```"}};
        REQUIRE(model.value()->chat(code_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(3));
        REQUIRE(model.value()->chat(code_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(3));
        REQUIRE(model.value()->chat(code_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(4));
        REQUIRE(model.value()->chat(code_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(5));

        // 2. Low-acceptance general completions pause general without disturbing code_edit.
        server.set_body(low_gen);
        GenerateRequest gen_req{"", "Write a brief summary of cloud formations.", {}};
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(3));
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(3));

        // Workload isolation: code_edit remains at 5 while general is paused.
        server.set_body(high_chat);
        REQUIRE(model.value()->chat(code_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(5));

        // 3. Cooldown skips 2 paused requests (emitting speculative.n_max = 0).
        server.set_body(paused_gen);
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(0));
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(0));

        // 4. First cooldown probe fails upstream with HTTP 500: RAII guard cancels probe
        // and restarts cooldown so the workload cannot wedge in probe_in_flight.
        server.set_status(500);
        const auto failed_probe = model.value()->generate(gen_req, {}, {});
        CHECK_FALSE(failed_probe.ok());
        CHECK(failed_probe.status().code() == ErrorCode::backend_error);
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(1));

        {
            const auto rt = backend->runtime_status();
            REQUIRE(rt.has_value());
            REQUIRE(rt->diagnostics.has_value());
            const auto *spec = rt->diagnostics->find("adaptive_speculation");
            REQUIRE(spec != nullptr);
            const auto *spec_workloads = spec->find("workloads");
            REQUIRE(spec_workloads != nullptr);
            const auto *general_bucket = spec_workloads->find("general");
            REQUIRE(general_bucket != nullptr);
            CHECK(general_bucket->find("paused")->as_bool());
            CHECK_FALSE(general_bucket->find("probe_in_flight")->as_bool());
            CHECK(general_bucket->find("cooldown_remaining")->as_uint() == 2u);
            CHECK(general_bucket->find("probes_issued")->as_uint() == 1u);
            CHECK(general_bucket->find("probe_recoveries")->as_uint() == 0u);
        }

        // 5. Restarted cooldown runs 2 paused requests, then probe #2 succeeds and recovers.
        server.set_status(200);
        server.set_body(paused_gen);
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(0));
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(0));

        server.set_body(high_gen);
        REQUIRE(model.value()->generate(gen_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(1));

        // 6. Reasoning chat classifies into the reasoning bucket.
        server.set_body(high_chat);
        ChatRequest reason_req;
        reason_req.messages = {{"user", "Derive the recurrence"}};
        reason_req.separate_reasoning = true;
        REQUIRE(model.value()->chat(reason_req, {}, {}).ok());
        CHECK(last_spec_n_max() == std::optional<std::int64_t>(3));

        // 7. Inspect full runtime_status()->diagnostics["adaptive_speculation"] snapshot.
        const auto rt = backend->runtime_status();
        REQUIRE(rt.has_value());
        REQUIRE(rt->diagnostics.has_value());
        const auto *spec = rt->diagnostics->find("adaptive_speculation");
        REQUIRE(spec != nullptr);
        CHECK(spec->find("enabled")->as_bool());
        const auto *workloads = spec->find("workloads");
        REQUIRE(workloads != nullptr);
        const auto *code_bucket = workloads->find("code_edit");
        const auto *general_bucket = workloads->find("general");
        const auto *reasoning_bucket = workloads->find("reasoning");
        REQUIRE(code_bucket != nullptr);
        REQUIRE(general_bucket != nullptr);
        REQUIRE(reasoning_bucket != nullptr);
        CHECK(code_bucket->find("draft_n_max")->as_uint() == 5u);
        CHECK_FALSE(code_bucket->find("paused")->as_bool());
        CHECK_FALSE(general_bucket->find("paused")->as_bool());
        CHECK(general_bucket->find("draft_n_max")->as_uint() == 1u);
        CHECK(general_bucket->find("probes_issued")->as_uint() == 2u);
        CHECK(general_bucket->find("probe_recoveries")->as_uint() == 1u);
        CHECK(general_bucket->find("reason")->as_string() == "probe_recovered");
        CHECK(general_bucket->find("rolling_decode_tps")->as_double() == doctest::Approx(80.0));
        CHECK(reasoning_bucket->find("total_observations")->as_uint() == 1u);

        // 8. Non-empty grammar classifies requests into tool_json.
        LlamaServerBackendOptions grammar_opts = opts;
        grammar_opts.grammar = "root ::= \"ok\"";
        auto grammar_backend = make_llamaserver_backend(grammar_opts);
        auto grammar_model = grammar_backend->load_model({"fake-model", "cpu:0"});
        REQUIRE(grammar_model.ok());
        server.set_body(high_gen);
        REQUIRE(grammar_model.value()->generate(gen_req, {}, {}).ok());
        const auto grt = grammar_backend->runtime_status();
        REQUIRE(grt.has_value());
        REQUIRE(grt->diagnostics.has_value());
        const auto *gspec = grt->diagnostics->find("adaptive_speculation");
        REQUIRE(gspec != nullptr);
        const auto *gworkloads = gspec->find("workloads");
        REQUIRE(gworkloads != nullptr);
        REQUIRE(gworkloads->find("tool_json") != nullptr);
        REQUIRE(gworkloads->find("general") != nullptr);
        CHECK(gworkloads->find("tool_json")->find("total_observations")->as_uint() == 1u);
        CHECK(gworkloads->find("general")->find("total_observations")->as_uint() == 0u);
    }

#if defined(SONDER_HAS_SERVER)
    TEST_CASE("llamaserver JSON config parses adaptive_speculation strictly and wires into backend") {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path = std::filesystem::temp_directory_path() /
                          ("sonder-adaptive-spec-cfg-test-" + std::to_string(unique) + ".json");
        struct Cleanup {
            std::filesystem::path file;
            ~Cleanup() {
                std::error_code ec;
                std::filesystem::remove(file, ec);
            }
        } cleanup{path};
        const auto write = [&](const std::string &body) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out << body;
        };

        // Omitted adaptive_speculation keeps enabled = false.
        write(R"({"mode":"attach","base_url":"http://127.0.0.1:8080"})");
        {
            BackendSetup setup;
            REQUIRE(load_llamaserver_config(path.string(), setup).ok());
            CHECK_FALSE(setup.llamaserver_adaptive_speculation.enabled);
        }

        // Valid full adaptive_speculation object parses and wires into make_backend().
        sonder_test::FakeLlamaServer server;
        write(std::string(R"({"mode":"attach","base_url":")") + server.url() +
              R"(","adaptive_speculation":{"enabled":true,"min_draft_n":1,"initial_draft_n":2,)"
              R"("max_draft_n":8,"window_size":24,"min_samples":5,)"
              R"("pause_acceptance_threshold":0.2,"step_down_acceptance_threshold":0.4,)"
              R"("step_up_acceptance_threshold":0.75,"cooldown_requests":10,)"
              R"("probe_draft_n":2,"min_draft_tokens_per_sample":6}})");
        {
            BackendSetup setup;
            setup.backend = "llamaserver";
            const auto loaded = load_llamaserver_config(path.string(), setup);
            REQUIRE_MESSAGE(loaded.ok(), loaded.to_string());
            const auto &cfg = setup.llamaserver_adaptive_speculation;
            CHECK(cfg.enabled);
            CHECK(cfg.min_draft_n == 1u);
            CHECK(cfg.initial_draft_n == 2u);
            CHECK(cfg.max_draft_n == 8u);
            CHECK(cfg.window_size == 24u);
            CHECK(cfg.min_samples == 5u);
            CHECK(cfg.pause_acceptance_threshold == doctest::Approx(0.2));
            CHECK(cfg.step_down_acceptance_threshold == doctest::Approx(0.4));
            CHECK(cfg.step_up_acceptance_threshold == doctest::Approx(0.75));
            CHECK(cfg.cooldown_requests == 10u);
            CHECK(cfg.probe_draft_n == 2u);
            CHECK(cfg.min_draft_tokens_per_sample == 6u);

            auto built = make_backend(setup);
            REQUIRE_MESSAGE(built.ok(), built.status().to_string());
            const auto rt = built.value()->runtime_status();
            REQUIRE(rt.has_value());
            REQUIRE(rt->diagnostics.has_value());
            const auto *spec = rt->diagnostics->find("adaptive_speculation");
            REQUIRE(spec != nullptr);
            CHECK(spec->find("enabled")->as_bool());
            const auto *rcfg = spec->find("config");
            REQUIRE(rcfg != nullptr);
            CHECK(rcfg->find("min_draft_n")->as_uint() == 1u);
            CHECK(rcfg->find("initial_draft_n")->as_uint() == 2u);
            CHECK(rcfg->find("max_draft_n")->as_uint() == 8u);
            CHECK(rcfg->find("window_size")->as_uint() == 24u);
            CHECK(rcfg->find("min_samples")->as_uint() == 5u);
            CHECK(rcfg->find("pause_acceptance_threshold")->as_double() == doctest::Approx(0.2));
            CHECK(rcfg->find("step_down_acceptance_threshold")->as_double() == doctest::Approx(0.4));
            CHECK(rcfg->find("step_up_acceptance_threshold")->as_double() == doctest::Approx(0.75));
            CHECK(rcfg->find("cooldown_requests")->as_uint() == 10u);
            CHECK(rcfg->find("probe_draft_n")->as_uint() == 2u);
            CHECK(rcfg->find("min_draft_tokens_per_sample")->as_uint() == 6u);
        }

        // Strict rejection of unknown keys, wrong types, out-of-range values, and reversed thresholds.
        const char *const invalid_docs[] = {
            R"({"adaptive_speculation":true})",
            R"({"adaptive_speculation":{"unknown_key":1}})",
            R"({"adaptive_speculation":{"enabled":1}})",
            R"({"adaptive_speculation":{"max_draft_n":true}})",
            R"({"adaptive_speculation":{"max_draft_n":3.5}})",
            R"({"adaptive_speculation":{"max_draft_n":0}})",
            R"({"adaptive_speculation":{"max_draft_n":129}})",
            R"({"adaptive_speculation":{"min_draft_n":-1}})",
            R"({"adaptive_speculation":{"min_draft_n":5,"max_draft_n":4}})",
            R"({"adaptive_speculation":{"initial_draft_n":7,"max_draft_n":6}})",
            R"({"adaptive_speculation":{"min_draft_n":3,"initial_draft_n":2,"max_draft_n":6}})",
            R"({"adaptive_speculation":{"probe_draft_n":0}})",
            R"({"adaptive_speculation":{"probe_draft_n":7,"max_draft_n":6}})",
            R"({"adaptive_speculation":{"window_size":0}})",
            R"({"adaptive_speculation":{"window_size":257}})",
            R"({"adaptive_speculation":{"window_size":4,"min_samples":5}})",
            R"({"adaptive_speculation":{"pause_acceptance_threshold":"0.25"}})",
            R"({"adaptive_speculation":{"pause_acceptance_threshold":-0.1}})",
            R"({"adaptive_speculation":{"step_up_acceptance_threshold":1.1}})",
            R"({"adaptive_speculation":{"pause_acceptance_threshold":0.5,"step_down_acceptance_threshold":0.4}})",
            R"({"adaptive_speculation":{"step_down_acceptance_threshold":0.8,"step_up_acceptance_threshold":0.7}})",
            R"({"adaptive_speculation":{"cooldown_requests":-1}})",
            R"({"adaptive_speculation":{"min_draft_tokens_per_sample":-1}})",
            R"({"adaptive_speculation":{"min_draft_tokens_per_sample":1.5}})",
        };
        for (const char *body : invalid_docs) {
            write(body);
            BackendSetup untouched;
            untouched.llamaserver_adaptive_speculation.enabled = true;
            untouched.llamaserver_adaptive_speculation.max_draft_n = 10;
            const auto st = load_llamaserver_config(path.string(), untouched);
            CHECK_MESSAGE(st.code() == ErrorCode::invalid_argument, body);
            CHECK(untouched.llamaserver_adaptive_speculation.enabled);
            CHECK(untouched.llamaserver_adaptive_speculation.max_draft_n == 10u);
        }

        // Oversized unknown key is bounded in the error message.
        write(std::string(R"({"adaptive_speculation":{")") + std::string(900, 'z') + R"(":1}})");
        {
            BackendSetup setup;
            const auto st = load_llamaserver_config(path.string(), setup);
            REQUIRE(st.code() == ErrorCode::invalid_argument);
            CHECK(st.message().size() < 1024u);
            CHECK(st.message().find(std::string(64, 'z') + "...") != std::string::npos);
        }
    }
#endif
}
