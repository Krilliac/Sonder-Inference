#include <doctest/doctest.h>

#include <chrono>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "../llamaserver_protocol.hpp"
#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/engine.hpp"
#if defined(SONDER_HAS_SERVER)
#include "server/src/openai.hpp"
#endif

using namespace sonder::inference;

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
        sonder_test::FakeLlamaServer server;
        server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
        server.set_initial_delay(std::chrono::milliseconds(300));
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.native_completion = false;
        auto b = make_llamaserver_backend(o);
        auto m = b->load_model({"fake-model", "cpu:0"});
        REQUIRE(m.ok());
        GenerateRequest q;
        q.prompt = "x";
        CancellationSource source;
        std::thread canceller([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (server.bodies().empty() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            source.cancel();
        });
        auto result = m.value()->generate(q, source.token(), {});
        canceller.join();
        REQUIRE(!server.bodies().empty());
        REQUIRE(result.status().code() == ErrorCode::cancelled);
        for (int i = 0; i < 100 && server.disconnects() == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(server.disconnects() > 0);
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
        remote.base_url = "http://10.77.0.2:8080";
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
}
