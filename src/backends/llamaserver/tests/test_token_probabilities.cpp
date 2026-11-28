// Telemetry token probabilities for backend-sampled tokens (llamaserver):
// the n_probs probe, inference.token.generated unit "token" events and
// inference.sampling.candidates, against fake llama-server streams.
#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/text_capture.hpp"

using namespace sonder::inference;

namespace {

// Native /completion stream with llama-server post_sampling_probs frames
// (tools/server/server-task.cpp probs_vector_to_json, "prob"/"top_probs").
// The second token's alternatives are deliberately unsorted.
const char *kNativePostSamplingBody =
    "data: {\"content\":\"Hi\",\"stop\":false,\"completion_probabilities\":[{\"id\":13,\"token\":\"Hi\","
    "\"bytes\":[72,105],\"prob\":0.75,\"top_probs\":[{\"id\":13,\"token\":\"Hi\",\"bytes\":[72,105],\"prob\":0.75},"
    "{\"id\":7,\"token\":\"Hey\",\"bytes\":[],\"prob\":0.2},{\"id\":9,\"token\":\"Yo\",\"bytes\":[],\"prob\":0.05}]}]}"
    "\n\n"
    "data: {\"content\":\" there\",\"stop\":false,\"completion_probabilities\":[{\"id\":21,\"token\":\" there\","
    "\"bytes\":[],\"prob\":0.5,\"top_probs\":[{\"id\":4,\"token\":\" you\",\"bytes\":[],\"prob\":0.3},"
    "{\"id\":21,\"token\":\" there\",\"bytes\":[],\"prob\":0.5},{\"id\":2,\"token\":\"!\",\"bytes\":[],\"prob\":0.2}]}]}"
    "\n\n"
    "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\",\"tokens_evaluated\":3,\"tokens_predicted\":2}\n\n";

// Four native tokens with three alternatives each, for the volume bounds.
std::string four_token_body() {
    std::string body;
    for (int i = 0; i < 4; ++i) {
        const std::string id = std::to_string(100 + i);
        body += "data: {\"content\":\"t" + std::to_string(i) +
                "\",\"stop\":false,\"completion_probabilities\":[{\"id\":" + id + ",\"token\":\"t" +
                std::to_string(i) + "\",\"prob\":0.6,\"top_probs\":[{\"id\":" + id + ",\"token\":\"t" +
                std::to_string(i) +
                "\",\"prob\":0.6},{\"id\":1,\"token\":\"a\",\"prob\":0.3},{\"id\":2,\"token\":\"b\",\"prob\":0.1}]}]}\n\n";
    }
    body += "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\",\"tokens_evaluated\":1,\"tokens_predicted\":4}\n\n";
    return body;
}

struct Event {
    std::string type;
    std::string level;
    json::Value attributes;
};

std::vector<Event> read_events(const MemoryTelemetrySink &sink, std::string_view type) {
    std::vector<Event> out;
    for (const auto &line : sink.lines()) {
        auto parsed = json::parse(line);
        REQUIRE(parsed.ok());
        const auto *event_type = parsed->find("event_type");
        REQUIRE(event_type != nullptr);
        if (event_type->as_string() != type)
            continue;
        const auto *sampling = parsed->find("sampling");
        const auto *attrs = parsed->find("attributes");
        REQUIRE(sampling != nullptr);
        REQUIRE(attrs != nullptr);
        out.push_back(Event{event_type->as_string(), sampling->find("level")->as_string(), *attrs});
    }
    return out;
}

// An engine with one memory sink and the llamaserver backend on `server`.
struct Harness {
    std::shared_ptr<MemoryTelemetrySink> sink = std::make_shared<MemoryTelemetrySink>();
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Session> session;

    Harness(const sonder_test::FakeLlamaServer &server, TelemetryOptions telemetry = {}) {
        EngineOptions options;
        options.telemetry = telemetry;
        options.telemetry_sinks.push_back(sink);
        options.scheduling.enabled = false;
        engine = std::make_unique<Engine>(options);
        LlamaServerBackendOptions upstream;
        upstream.base_url = server.url();
        REQUIRE(engine->register_backend(make_llamaserver_backend(upstream)).ok());
        auto model = engine->load_model("llamaserver", ModelLoadOptions{"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        auto created = engine->create_session(model.value(), {});
        REQUIRE(created.ok());
        session = created.value();
    }
    void flush() { engine->telemetry().flush(); }
};

// The body of the completion/chat request (the props probe is a GET).
json::Value request_body(const sonder_test::FakeLlamaServer &server) {
    auto parsed = json::parse(server.last_body());
    REQUIRE(parsed.ok());
    return parsed.value();
}

}  // namespace

TEST_SUITE("llamaserver_token_probabilities") {
    TEST_CASE("telemetry probe asks llama-server for post-sampling probabilities and keeps them from the caller") {
        sonder_test::FakeLlamaServer server;
        server.set_body(kNativePostSamplingBody);
        // Token ids are output text: capture must be on for them to be exported.
        TelemetryOptions t;
        t.text_capture = TextCapturePolicy::on;
        Harness h(server, t);
        unsigned chunks_with_logprobs = 0;
        auto result = h.session->generate("x", [&](const TokenChunk &chunk) {
            if (chunk.logprob.has_value() || !chunk.top_logprobs.empty())
                ++chunks_with_logprobs;
            return true;
        });
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(result->text == "Hi there");
        CHECK(result->logprobs.empty());
        CHECK(chunks_with_logprobs == 0);

        const auto body = request_body(server);
        REQUIRE(body.find("n_probs") != nullptr);
        CHECK(body.find("n_probs")->as_int() == 5);
        REQUIRE(body.find("post_sampling_probs") != nullptr);
        CHECK(body.find("post_sampling_probs")->as_bool());
        CHECK(body.find("logprobs") == nullptr);

        h.flush();
        const auto tokens = read_events(*h.sink, "inference.token.generated");
        REQUIRE(tokens.size() == 2);
        const TokenId expected_ids[] = {13, 21};
        const double expected_p[] = {0.75, 0.5};
        const char *expected_text[] = {"Hi", " there"};
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto &a = tokens[i].attributes;
            CHECK(tokens[i].level == "standard");
            CHECK(a.find("unit")->as_string() == "token");
            CHECK(a.find("count")->as_int() == 1);
            CHECK(a.find("index")->as_int() == static_cast<std::int64_t>(i));
            REQUIRE(a.find("token_id") != nullptr);
            CHECK(a.find("token_id")->as_int() == expected_ids[i]);
            CHECK(a.find("probability")->as_double() == doctest::Approx(expected_p[i]));
            CHECK(a.find("probability_basis")->as_string() == "post_sampling");
            REQUIRE(a.find("text") != nullptr);
            CHECK(a.find("text")->as_string() == expected_text[i]);
        }

        const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
        REQUIRE(candidates.size() == 2);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const auto &a = candidates[i].attributes;
            CHECK(candidates[i].level == "standard");
            CHECK(a.find("index")->as_int() == static_cast<std::int64_t>(i));
            REQUIRE(a.find("sampled_token_id") != nullptr);
            CHECK(a.find("sampled_token_id")->as_int() == expected_ids[i]);
            CHECK(a.find("probability_basis")->as_string() == "post_sampling");
            const auto &list = a.find("candidates")->as_array();
            REQUIRE(list.size() == 3);
            for (std::size_t j = 1; j < list.size(); ++j)
                CHECK(list[j - 1].find("probability")->as_double() >= list[j].find("probability")->as_double());
            for (const auto &c : list) {
                REQUIRE(c.find("token_id") != nullptr);
                CHECK(c.find("text") != nullptr);
            }
        }
        // The second token's unsorted alternatives arrive most probable first.
        const auto &second = candidates[1].attributes.find("candidates")->as_array();
        CHECK(second[0].find("token_id")->as_int() == 21);
        CHECK(second[1].find("token_id")->as_int() == 4);
        CHECK(second[2].find("token_id")->as_int() == 2);
        CHECK(second[1].find("probability")->as_double() == doctest::Approx(0.3));
    }

    TEST_CASE("candidates honour top_k, stride and the per-request cap, and carry text only with capture") {
        sonder_test::FakeLlamaServer server;
        server.set_body(four_token_body());
        SUBCASE("stride 2, no cap, top_k 2, text capture on") {
            TelemetryOptions t;
            t.token_probability_top_k = 2;
            t.token_candidates_stride = 2;
            t.token_candidates_max_events = 0;
            t.capture_text = true;
            Harness h(server, t);
            auto result = h.session->generate("x");
            REQUIRE_MESSAGE(result.ok(), result.status().to_string());
            CHECK(request_body(server).find("n_probs")->as_int() == 2);
            h.flush();
            CHECK(read_events(*h.sink, "inference.token.generated").size() == 4);
            const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
            REQUIRE(candidates.size() == 2);
            CHECK(candidates[0].attributes.find("index")->as_int() == 0);
            CHECK(candidates[1].attributes.find("index")->as_int() == 2);
            for (const auto &e : candidates) {
                const auto &list = e.attributes.find("candidates")->as_array();
                REQUIRE(list.size() == 2);
                REQUIRE(list[1].find("text") != nullptr);
                CHECK(list[1].find("text")->as_string() == "a");
            }
        }
        SUBCASE("cap 1 keeps the first event of the request") {
            TelemetryOptions t;
            t.token_candidates_max_events = 1;
            Harness h(server, t);
            REQUIRE(h.session->generate("x").ok());
            h.flush();
            const auto tokens = read_events(*h.sink, "inference.token.generated");
            REQUIRE(tokens.size() == 4);
            for (const auto &e : tokens)
                CHECK(e.attributes.find("unit")->as_string() == "token");
            const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
            REQUIRE(candidates.size() == 1);
            CHECK(candidates[0].attributes.find("index")->as_int() == 0);
        }
        SUBCASE("top_k above the bound is clamped to 10") {
            TelemetryOptions t;
            t.token_probability_top_k = 50;
            Harness h(server, t);
            REQUIRE(h.session->generate("x").ok());
            CHECK(request_body(server).find("n_probs")->as_int() == kMaxTelemetryTopK);
        }
    }

    TEST_CASE("no probe without standard token telemetry or with top_k 0") {
        sonder_test::FakeLlamaServer server;
        server.set_body(kNativePostSamplingBody);
        SUBCASE("metrics level") {
            TelemetryOptions t;
            t.level = TelemetryLevel::metrics;
            Harness h(server, t);
            REQUIRE(h.session->generate("x").ok());
            const auto body = request_body(server);
            CHECK(body.find("n_probs") == nullptr);
            CHECK(body.find("post_sampling_probs") == nullptr);
            h.flush();
            CHECK(read_events(*h.sink, "inference.token.generated").empty());
            CHECK(read_events(*h.sink, "inference.sampling.candidates").empty());
        }
        SUBCASE("top_k 0 keeps chunk events") {
            TelemetryOptions t;
            t.token_probability_top_k = 0;
            Harness h(server, t);
            REQUIRE(h.session->generate("x").ok());
            CHECK(request_body(server).find("n_probs") == nullptr);
            h.flush();
            const auto tokens = read_events(*h.sink, "inference.token.generated");
            REQUIRE(tokens.size() == 2);
            for (const auto &e : tokens) {
                CHECK(e.attributes.find("unit")->as_string() == "chunk");
                CHECK(e.attributes.find("probability") == nullptr);
            }
            CHECK(read_events(*h.sink, "inference.sampling.candidates").empty());
        }
    }

    TEST_CASE("caller-requested chat logprobs are reused unchanged and feed telemetry") {
        sonder_test::FakeLlamaServer server;
        // llama-server OpenAI chat stream with logprobs.content (pre-sampling).
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null}]}\n\n"
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null,\"logprobs\":{\"content\":"
            "[{\"id\":42,\"token\":\"ok\",\"bytes\":[111,107],\"logprob\":-0.105360515,\"top_logprobs\":[{\"id\":42,"
            "\"token\":\"ok\",\"bytes\":[],\"logprob\":-0.105360515},{\"id\":8,\"token\":\"no\",\"bytes\":[],"
            "\"logprob\":-2.302585093}]}]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n"
            "data: [DONE]\n\n");
        TelemetryOptions t;
        t.text_capture = TextCapturePolicy::on;  // token ids are output text
        Harness h(server, t);
        SamplingConfig sampling;
        sampling.logprobs = true;
        sampling.top_logprobs = 2;
        auto result = h.session->chat({{"user", "hi"}}, {}, sampling);
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(result->text == "ok");
        REQUIRE(result->logprobs.size() == 1);
        CHECK(result->logprobs[0].logprob == doctest::Approx(-0.105360515));
        CHECK(result->logprobs[0].top_logprobs.size() == 2);

        const auto body = request_body(server);
        CHECK(body.find("logprobs")->as_bool());
        CHECK(body.find("top_logprobs")->as_int() == 2);
        CHECK(body.find("n_probs") == nullptr);
        CHECK(body.find("post_sampling_probs") == nullptr);

        h.flush();
        const auto tokens = read_events(*h.sink, "inference.token.generated");
        REQUIRE(tokens.size() == 1);
        const auto &a = tokens[0].attributes;
        CHECK(a.find("unit")->as_string() == "token");
        REQUIRE(a.find("token_id") != nullptr);
        CHECK(a.find("token_id")->as_int() == 42);
        CHECK(a.find("probability")->as_double() == doctest::Approx(0.9).epsilon(1e-5));
        CHECK(a.find("probability_basis")->as_string() == "pre_sampling");
        const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
        REQUIRE(candidates.size() == 1);
        const auto &list = candidates[0].attributes.find("candidates")->as_array();
        REQUIRE(list.size() == 2);
        REQUIRE(list[1].find("token_id") != nullptr);
        CHECK(list[1].find("token_id")->as_int() == 8);
        CHECK(list[1].find("probability")->as_double() == doctest::Approx(0.1).epsilon(1e-5));
    }

    TEST_CASE("chat telemetry probe uses post-sampling probabilities from logprobs.content") {
        sonder_test::FakeLlamaServer server;
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"content\":\"yes\"},\"finish_reason\":null,\"logprobs\":{\"content\":"
            "[{\"id\":5,\"token\":\"yes\",\"bytes\":[],\"prob\":0.6,\"top_probs\":[{\"id\":5,\"token\":\"yes\","
            "\"bytes\":[],\"prob\":0.6},{\"id\":6,\"token\":\"no\",\"bytes\":[],\"prob\":0.4}]}]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n");
        TelemetryOptions t;
        t.text_capture = TextCapturePolicy::on;  // token ids are output text
        Harness h(server, t);
        unsigned chunks_with_logprobs = 0;
        auto result = h.session->chat({{"user", "hi"}}, [&](const TokenChunk &chunk) {
            if (chunk.logprob.has_value() || !chunk.top_logprobs.empty())
                ++chunks_with_logprobs;
            return true;
        });
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(result->logprobs.empty());
        CHECK(chunks_with_logprobs == 0);
        const auto body = request_body(server);
        CHECK(body.find("n_probs")->as_int() == 5);
        CHECK(body.find("post_sampling_probs")->as_bool());
        CHECK(body.find("logprobs") == nullptr);
        h.flush();
        const auto tokens = read_events(*h.sink, "inference.token.generated");
        REQUIRE(tokens.size() == 1);
        REQUIRE(tokens[0].attributes.find("token_id") != nullptr);
        CHECK(tokens[0].attributes.find("token_id")->as_int() == 5);
        CHECK(tokens[0].attributes.find("probability")->as_double() == doctest::Approx(0.6));
        CHECK(read_events(*h.sink, "inference.sampling.candidates").size() == 1);
    }

    TEST_CASE("no probe for chat with tools or for a generic OpenAI upstream") {
        sonder_test::FakeLlamaServer server;
        SUBCASE("native chat with tools") {
            server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n"
                            "data: [DONE]\n\n");
            LlamaServerBackendOptions o;
            o.base_url = server.url();
            auto model = make_llamaserver_backend(o)->load_model({"fake-model", "cpu:0"});
            REQUIRE(model.ok());
            ChatRequest req;
            req.messages = {{"user", "hi"}};
            req.tools = {ToolDefinition{"lookup", "find a thing", "", std::nullopt}};
            req.telemetry_top_k = 5;
            REQUIRE(model.value()->chat(req, {}, [](const TokenChunk &) { return true; }).ok());
            const auto body = request_body(server);
            CHECK(body.find("n_probs") == nullptr);
            CHECK(body.find("post_sampling_probs") == nullptr);
            CHECK(body.find("logprobs") == nullptr);
        }
        SUBCASE("generic /v1/completions") {
            server.set_body("data: {\"choices\":[{\"text\":\"g\",\"finish_reason\":\"stop\"}]}\n\n"
                            "data: [DONE]\n\n");
            LlamaServerBackendOptions o;
            o.base_url = server.url();
            o.native_completion = false;
            auto model = make_llamaserver_backend(o)->load_model({"fake-model", "cpu:0"});
            REQUIRE(model.ok());
            GenerateRequest req;
            req.prompt = "x";
            req.telemetry_top_k = 5;
            REQUIRE(model.value()->generate(req, {}, [](const TokenChunk &) { return true; }).ok());
            const auto body = request_body(server);
            CHECK(body.find("n_probs") == nullptr);
            CHECK(body.find("post_sampling_probs") == nullptr);
            CHECK(body.find("logprobs") == nullptr);
        }
    }

    TEST_CASE("malformed probability entries report no sampled probability") {
        sonder_test::FakeLlamaServer server;
        server.set_body(
            // No id on the sampled token.
            "data: {\"content\":\"a\",\"stop\":false,\"completion_probabilities\":[{\"token\":\"a\",\"prob\":0.5,"
            "\"top_probs\":[]}]}\n\n"
            // Probability above 1.
            "data: {\"content\":\"b\",\"stop\":false,\"completion_probabilities\":[{\"id\":3,\"token\":\"b\","
            "\"prob\":1.5,\"top_probs\":[]}]}\n\n"
            // Valid sampled token; one alternative lacks an id, one has a negative
            // probability, one has an id beyond int32: only the good one is kept.
            "data: {\"content\":\"c\",\"stop\":false,\"completion_probabilities\":[{\"id\":4,\"token\":\"c\","
            "\"prob\":0.9,\"top_probs\":[{\"token\":\"x\",\"prob\":0.5},{\"id\":5,\"token\":\"y\",\"prob\":-0.1},"
            "{\"id\":4294967296,\"token\":\"z\",\"prob\":0.2},{\"id\":4,\"token\":\"c\",\"prob\":0.9}]}]}\n\n"
            "data: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\"}\n\n");
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = make_llamaserver_backend(o)->load_model({"fake-model", "cpu:0"});
        REQUIRE(model.ok());
        GenerateRequest req;
        req.prompt = "x";
        req.telemetry_top_k = 3;
        std::vector<bool> has_sampled;
        std::vector<std::size_t> candidate_counts;
        TokenId last_id = -1;
        auto r = model.value()->generate(req, {}, [&](const TokenChunk &chunk) {
            has_sampled.push_back(chunk.sampled_probability != nullptr);
            candidate_counts.push_back(chunk.sampled_probability ? chunk.sampled_probability->candidates.size() : 0);
            if (chunk.sampled_probability)
                last_id = chunk.sampled_probability->candidates.empty()
                              ? -1
                              : chunk.sampled_probability->candidates.front().token_id;
            CHECK_FALSE(chunk.logprob.has_value());
            return true;
        });
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        REQUIRE(has_sampled.size() == 3);
        CHECK_FALSE(has_sampled[0]);
        CHECK_FALSE(has_sampled[1]);
        CHECK(has_sampled[2]);
        CHECK(candidate_counts[2] == 1);
        CHECK(last_id == 4);
    }

    // Token ids map 1:1 to text (POST /v1/detokenize), so they follow the
    // output-text capture policy like the "text" fields do.
    TEST_CASE("backend-probed token ids are output text: policy off and the request opt-out export none") {
        sonder_test::FakeLlamaServer server;
        server.set_body(kNativePostSamplingBody);
        const auto assert_no_ids = [](Harness &h, std::size_t expected_tokens) {
            h.flush();
            const auto tokens = read_events(*h.sink, "inference.token.generated");
            REQUIRE(tokens.size() == expected_tokens);
            for (const auto &e : tokens) {
                const auto &a = e.attributes;
                CHECK(a.find("token_id") == nullptr);
                CHECK(a.find("text") == nullptr);
                // The non-identifying measurements stay.
                CHECK(a.find("index") != nullptr);
                CHECK(a.find("probability") != nullptr);
                CHECK(a.find("probability_basis") != nullptr);
            }
            const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
            REQUIRE(candidates.size() == expected_tokens);
            for (const auto &e : candidates) {
                const auto &a = e.attributes;
                CHECK(a.find("sampled_token_id") == nullptr);
                CHECK(a.find("probability_basis") != nullptr);
                for (const auto &c : a.find("candidates")->as_array()) {
                    CHECK(c.find("token_id") == nullptr);
                    CHECK(c.find("text") == nullptr);
                    CHECK(c.find("probability") != nullptr);
                }
            }
        };
        SUBCASE("operator policy off") {
            TelemetryOptions t;
            t.text_capture = TextCapturePolicy::off;
            Harness h(server, t);
            REQUIRE(h.session->generate("x").ok());
            assert_no_ids(h, 2);
        }
        SUBCASE("operator policy full, request sets no_text_capture") {
            TelemetryOptions t;
            t.text_capture = TextCapturePolicy::full;
            Harness h(server, t);
            RequestOptions quiet;
            quiet.no_text_capture = true;
            REQUIRE(h.session->generate("x", {}, std::nullopt, quiet).ok());
            assert_no_ids(h, 2);
        }
    }

    TEST_CASE("backend-probed token ids are still exported when the policy captures output text") {
        sonder_test::FakeLlamaServer server;
        server.set_body(kNativePostSamplingBody);
        TelemetryOptions t;
        t.text_capture = TextCapturePolicy::on;
        Harness h(server, t);
        REQUIRE(h.session->generate("x").ok());
        h.flush();
        const auto tokens = read_events(*h.sink, "inference.token.generated");
        REQUIRE(tokens.size() == 2);
        CHECK(tokens[0].attributes.find("token_id")->as_int() == 13);
        CHECK(tokens[1].attributes.find("token_id")->as_int() == 21);
        const auto candidates = read_events(*h.sink, "inference.sampling.candidates");
        REQUIRE(candidates.size() == 2);
        CHECK(candidates[0].attributes.find("sampled_token_id")->as_int() == 13);
        CHECK(candidates[1].attributes.find("sampled_token_id")->as_int() == 21);
        for (const auto &e : candidates) {
            for (const auto &c : e.attributes.find("candidates")->as_array())
                CHECK(c.find("token_id") != nullptr);
        }
    }
}
