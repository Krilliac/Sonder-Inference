// llama-server request path: served-context num_ctx, cache_prompt, slot
// affinity, thinking controls and the OpenAI usage/timings they surface.
#include <doctest/doctest.h>

#include <optional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../llamaserver_protocol.hpp"
#include "../slot_affinity.hpp"
#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/engine.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/server.hpp"
#endif

using namespace sonder::inference;

namespace {

constexpr const char *kChatStream =
    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
    "\"timings\":{\"prompt_n\":3,\"cache_n\":9,\"predicted_n\":1,\"draft_n\":4,\"draft_n_accepted\":3}}\n\n"
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":1,"
    "\"prompt_tokens_details\":{\"cached_tokens\":9}}}\n\n"
    "data: [DONE]\n\n";

constexpr const char *kPlainChatStream =
    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":1}}\n\n"
    "data: [DONE]\n\n";

#if defined(SONDER_HAS_SERVER)
constexpr const char *kTimedChatStream =
    "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],"
    "\"timings\":{\"prompt_n\":12,\"cache_n\":9,\"prompt_ms\":1.5,"
    "\"predicted_n\":7,\"predicted_ms\":2.25,\"draft_n\":4,\"draft_n_accepted\":3}}\n\n"
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,"
    "\"prompt_tokens_details\":{\"cached_tokens\":5}}}\n\n"
    "data: [DONE]\n\n";

constexpr const char *kEmptyLengthStream =
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"
    "data: [DONE]\n\n";
#endif

json::Value parse(const std::string &text) {
    auto v = json::parse(text);
    REQUIRE_MESSAGE(v.ok(), "not JSON: " << text);
    return v.value();
}

#if defined(SONDER_HAS_SERVER)
json::Value last_sse_json(const std::string &body) {
    const std::size_t done = body.rfind("data: [DONE]");
    REQUIRE(done != std::string::npos);
    REQUIRE(done > 0);
    const std::size_t frame = body.rfind("data: ", done - 1);
    REQUIRE(frame != std::string::npos);
    const std::size_t start = frame + 6;
    const std::size_t end = body.find("\n\n", start);
    REQUIRE(end != std::string::npos);
    return parse(body.substr(start, end - start));
}
#endif

std::shared_ptr<BackendModel> load(const std::shared_ptr<LlamaServerBackend> &backend) {
    ModelLoadOptions lo;
    lo.model = "fake-model";
    auto m = backend->load_model(lo);
    REQUIRE_MESSAGE(m.ok(), m.status().to_string());
    return m.value();
}

ChatRequest chat(std::string key = {}) {
    ChatRequest r;
    r.messages = {{"user", "hi"}};
    r.session_key = std::move(key);
    return r;
}

// id_slot of the last streamed request body, or -1 when absent.
std::int64_t last_slot(const sonder_test::FakeLlamaServer &server) {
    const auto body = parse(server.last_body());
    const auto *slot = body.find("id_slot");
    return slot ? slot->as_int() : -1;
}

} // namespace

TEST_SUITE("llamaserver_request_path") {

    TEST_CASE("props parsing reads the per-slot context and slot count") {
        llamaserver::ServerProps p;
        REQUIRE(llamaserver::parse_props(parse(R"({"default_generation_settings":{"n_ctx":131072},"total_slots":2})"), p)
                    .ok());
        CHECK(p.n_ctx == 131072);
        CHECK(p.total_slots == 2);
        REQUIRE(llamaserver::parse_props(parse(R"({"n_ctx":4096})"), p).ok());
        CHECK(p.n_ctx == 4096);
        CHECK(p.total_slots == 0);
        CHECK_FALSE(llamaserver::parse_props(parse(R"({"total_slots":-1})"), p).ok());
        CHECK_FALSE(llamaserver::parse_props(parse(R"([1])"), p).ok());
    }

    TEST_CASE("num_ctx up to the served context is a no-op and larger values name both numbers") {
        SamplingConfig s;
        s.explicit_only = true;
        s.explicit_fields |= SamplingConfig::kNumCtx;
        s.num_ctx = 4096;
        CHECK(llamaserver::validate_sampling(s, 4096).ok());
        CHECK(llamaserver::validate_sampling(s, 131072).ok());
        s.num_ctx = 0;  // explicit 0 = backend default
        CHECK(llamaserver::validate_sampling(s, 4096).ok());
        s.num_ctx = 8192;
        const Status too_big = llamaserver::validate_sampling(s, 4096);
        CHECK(too_big.code() == ErrorCode::invalid_argument);
        CHECK(too_big.message().find("8192") != std::string::npos);
        CHECK(too_big.message().find("4096") != std::string::npos);
        // Unknown served context: accepted, the upstream keeps its own window.
        CHECK(llamaserver::validate_sampling(s, 0).ok());
    }

    TEST_CASE("a bridged chat with num_ctx within the served context streams and a larger one fails before the stream") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":32768},"total_slots":1})");
        server.set_body(kChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));
        CHECK(model->descriptor().context_length == 32768);
        auto req = chat();
        req.sampling.explicit_only = true;
        req.sampling.explicit_fields |= SamplingConfig::kNumCtx;
        req.sampling.num_ctx = 32768;
        auto ok = model->chat(req, {}, {});
        REQUIRE_MESSAGE(ok.ok(), ok.status().to_string());
        CHECK(parse(server.last_body()).find("num_ctx") == nullptr);  // never forwarded
        const auto streamed = server.bodies().size();
        req.sampling.num_ctx = 65536;
        auto refused = model->chat(req, {}, {});
        REQUIRE_FALSE(refused.ok());
        CHECK(refused.status().code() == ErrorCode::invalid_argument);
        CHECK(refused.status().message().find("65536") != std::string::npos);
        CHECK(refused.status().message().find("32768") != std::string::npos);
        CHECK(server.bodies().size() == streamed);
        CHECK(server.props_requests() == 1);  // fetched once, then cached
    }

    TEST_CASE("the configured context_length stands in when the upstream has no /props") {
        sonder_test::FakeLlamaServer server;
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        o.context_length = 2048;
        auto model = load(make_llamaserver_backend(o));
        auto req = chat();
        req.sampling.num_ctx = 2048;
        CHECK(model->chat(req, {}, {}).ok());
        req.sampling.num_ctx = 4096;
        CHECK(model->chat(req, {}, {}).status().code() == ErrorCode::invalid_argument);
    }

    TEST_CASE("native requests always send cache_prompt and keyed chats are pinned to slots") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":3})");
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));

        REQUIRE(model->chat(chat(), {}, {}).ok());
        auto body = parse(server.last_body());
        REQUIRE(body.find("cache_prompt"));
        CHECK(body.find("cache_prompt")->as_bool());
        CHECK(body.find("id_slot") == nullptr);  // no key: the upstream picks

        REQUIRE(model->chat(chat("run=r;agent=a"), {}, {}).ok());
        CHECK(last_slot(server) == 0);
        REQUIRE(model->chat(chat("run=r;agent=b"), {}, {}).ok());
        CHECK(last_slot(server) == 1);
        REQUIRE(model->chat(chat("run=r;agent=a"), {}, {}).ok());
        CHECK(last_slot(server) == 0);

        GenerateRequest g;
        g.prompt = "x";
        server.set_body("data: {\"content\":\"ok\",\"stop\":true}\n\n");
        REQUIRE(model->generate(g, {}, {}).ok());
        body = parse(server.last_body());
        CHECK(body.find("cache_prompt")->as_bool());
        CHECK(body.find("id_slot") == nullptr);
    }

    TEST_CASE("a single-slot server pins every keyed chat to slot 0") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":1})");
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));
        for (const char *key : {"a", "b", "c", "a"}) {
            REQUIRE(model->chat(chat(key), {}, {}).ok());
            CHECK(last_slot(server) == 0);
        }
    }

    TEST_CASE("generic OpenAI mode and disabled affinity send no llama.cpp-only fields") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":4})");
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions generic;
        generic.base_url = server.url();
        generic.native_completion = false;
        REQUIRE(load(make_llamaserver_backend(generic))->chat(chat("k"), {}, {}).ok());
        auto body = parse(server.last_body());
        CHECK(body.find("cache_prompt") == nullptr);
        CHECK(body.find("id_slot") == nullptr);

        LlamaServerBackendOptions no_affinity;
        no_affinity.base_url = server.url();
        no_affinity.slot_affinity = false;
        REQUIRE(load(make_llamaserver_backend(no_affinity))->chat(chat("k"), {}, {}).ok());
        body = parse(server.last_body());
        CHECK(body.find("cache_prompt")->as_bool());
        CHECK(body.find("id_slot") == nullptr);
    }

    TEST_CASE("thinking controls are forwarded as chat_template_kwargs only when set") {
        ChatRequest r = chat();
        auto body = llamaserver::build_chat_body("m", r);
        CHECK(body.find("chat_template_kwargs") == nullptr);
        r.thinking.enable_thinking = false;
        body = llamaserver::build_chat_body("m", r);
        REQUIRE(body.find("chat_template_kwargs"));
        CHECK_FALSE(body.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
        CHECK(body.find("chat_template_kwargs")->find("reasoning_effort") == nullptr);
        r.thinking.reasoning_effort = "high";
        body = llamaserver::build_chat_body("m", r);
        CHECK(body.find("chat_template_kwargs")->find("reasoning_effort")->as_string() == "high");
    }

    TEST_CASE("reasoning budget and assistant reasoning history are forwarded") {
        ChatRequest r = chat();
        r.messages.push_back(ChatMessage{"assistant", "answer", std::string("private thought")});
        auto body = llamaserver::build_chat_body("m", r);
        CHECK(body.find("reasoning_budget_tokens") == nullptr);
        CHECK(body.find("reasoning_budget_message") == nullptr);
        r.reasoning_budget_tokens = 512;
        r.reasoning_budget_message = "keep the reasoning concise";
        body = llamaserver::build_chat_body("m", r);
        REQUIRE(body.find("messages"));
        REQUIRE(body.find("messages")->as_array().size() == 2);
        CHECK(body.find("messages")->as_array()[1].find("reasoning_content")->as_string() == "private thought");
        CHECK(body.find("reasoning_budget_tokens")->as_int() == 512);
        CHECK(body.find("reasoning_budget_message")->as_string() == "keep the reasoning concise");
        r.messages.front().reasoning_content = "user-side text is not assistant reasoning";
        body = llamaserver::build_chat_body("m", r);
        CHECK(body.find("messages")->as_array()[0].find("reasoning_content") == nullptr);
    }

    TEST_CASE("llama-server timings parse into optional backend counters") {
        BackendTimings timings;
        REQUIRE(llamaserver::parse_timings(
                     parse(R"({"timings":{"prompt_n":12,"cache_n":9,"prompt_ms":1.5,
                         "predicted_n":7,"predicted_ms":2.25,"draft_n":4,"draft_n_accepted":3}})"),
                     timings)
                     .ok());
        REQUIRE(timings.prompt_n);
        REQUIRE(timings.cache_n);
        REQUIRE(timings.prompt_ms);
        REQUIRE(timings.predicted_n);
        REQUIRE(timings.predicted_ms);
        REQUIRE(timings.draft_n);
        REQUIRE(timings.draft_n_accepted);
        CHECK(*timings.prompt_n == 12);
        CHECK(*timings.cache_n == 9);
        CHECK(*timings.prompt_ms == doctest::Approx(1.5));
        CHECK(*timings.predicted_n == 7);
        CHECK(*timings.predicted_ms == doctest::Approx(2.25));
        CHECK(*timings.draft_n == 4);
        CHECK(*timings.draft_n_accepted == 3);
        CHECK_FALSE(llamaserver::parse_timings(parse(R"({"timings":{"draft_n":2,"draft_n_accepted":3}})"), timings).ok());
        REQUIRE(llamaserver::parse_timings(parse(R"({"choices":[]})"), timings).ok());
        CHECK_FALSE(timings.prompt_n.has_value());
    }

    TEST_CASE("partial timing frames merge without losing fields and preserve zero") {
        std::optional<BackendTimings> merged;
        BackendTimings first;
        first.prompt_n = 12;
        first.cache_n = 9;
        first.predicted_n = 7;
        llamaserver::merge_timings(merged, std::optional<BackendTimings>{first});
        BackendTimings second;
        second.cache_n = 0;
        second.draft_n = 4;
        llamaserver::merge_timings(merged, std::optional<BackendTimings>{second});
        REQUIRE(merged);
        CHECK(merged->prompt_n == std::optional<std::uint64_t>(12));
        CHECK(merged->cache_n == std::optional<std::uint64_t>(0));
        CHECK(merged->predicted_n == std::optional<std::uint64_t>(7));
        CHECK(merged->draft_n == std::optional<std::uint64_t>(4));
        llamaserver::merge_timings(merged, std::nullopt);
        CHECK(merged->prompt_n == std::optional<std::uint64_t>(12));
        CHECK(merged->draft_n == std::optional<std::uint64_t>(4));
    }

    TEST_CASE("slot affinity keeps busy slots, evicts the least recently used idle key and is bounded") {
        llamaserver::SlotAffinity a;
        CHECK_FALSE(a.acquire("", 4).slot());
        CHECK_FALSE(a.acquire("k", 0).slot());
        {
            auto x = a.acquire("x", 2);
            auto y = a.acquire("y", 2);
            CHECK(x.slot() == std::optional<std::uint32_t>(0));
            CHECK(y.slot() == std::optional<std::uint32_t>(1));
            // Both slots owned and busy: a new key is not pinned.
            CHECK_FALSE(a.acquire("z", 2).slot());
            // A known key keeps its slot even while busy.
            auto x2 = a.acquire("x", 2);
            CHECK(x2.slot() == std::optional<std::uint32_t>(0));
        }
        // All idle now; "y" is the least recently used, so "z" takes slot 1.
        CHECK(a.acquire("z", 2).slot() == std::optional<std::uint32_t>(1));
        CHECK_FALSE(a.owned_slot("y"));
        CHECK(a.owned_slot("x") == std::optional<std::uint32_t>(0));
        CHECK(a.keys() == 2);
        // A different slot count resets the layout; stale leases are harmless.
        auto stale = a.acquire("x", 2);
        CHECK(a.acquire("q", 3).slot() == std::optional<std::uint32_t>(0));
        stale.reset();
        CHECK(a.keys() == 1);
        for (int i = 0; i < 100; ++i) {
            (void)a.acquire("key" + std::to_string(i), 3);
        }
        CHECK(a.keys() <= 3);
    }

#if defined(SONDER_HAS_SERVER)
    TEST_CASE("HTTP default pin mode preserves explicit thinking and applies unset pins") {
        sonder_test::FakeLlamaServer upstream;
        upstream.set_body(kPlainChatStream);
        LlamaServerBackendOptions lo;
        lo.base_url = upstream.url();
        server::ServerOptions so;
        so.port = 0;
        so.backend_instance = make_llamaserver_backend(lo);
        so.models = {"fake-model"};
        so.pin_enable_thinking = true;
        so.pin_mode = server::PinMode::default_value;
        so.pin_reasoning_budget_tokens = 1024;
        so.pin_reasoning_budget_message = "pinned";
        server::Server srv(so);
        REQUIRE(srv.start().ok());
        httplib::Client client("127.0.0.1", srv.port());
        client.set_read_timeout(60, 0);

        const auto post = [&](const std::string &body, const httplib::Headers &headers = {}) {
            auto response = client.Post("/v1/chat/completions", headers, body, "application/json");
            REQUIRE(response);
            return std::pair<int, json::Value>{response->status, parse(response->body)};
        };

        auto [status, doc] = post(
            R"({"chat_template_kwargs":{"enable_thinking":false},"reasoning_budget_tokens":-1,
                "reasoning_budget_message":"" ,"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE_MESSAGE(status == 200, doc.dump());
        auto sent = parse(upstream.last_body());
        CHECK_FALSE(sent.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
        CHECK(sent.find("reasoning_budget_tokens")->as_int() == -1);
        CHECK(sent.find("reasoning_budget_message")->as_string().empty());
        CHECK(doc.find("sonder")->find("warnings") == nullptr);

        std::tie(status, doc) = post(R"({"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE_MESSAGE(status == 200, doc.dump());
        sent = parse(upstream.last_body());
        CHECK(sent.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
        CHECK(sent.find("reasoning_budget_tokens")->as_int() == 1024);
        CHECK(sent.find("reasoning_budget_message")->as_string() == "pinned");

        std::tie(status, doc) = post(R"({"reasoning_budget_tokens":512,"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE_MESSAGE(status == 200, doc.dump());
        CHECK(parse(upstream.last_body()).find("reasoning_budget_tokens")->as_int() == 512);

        std::tie(status, doc) = post(
            R"({"reasoning_budget_tokens":512,"messages":[{"role":"user","content":"hi"}]})",
            httplib::Headers{{"X-Sonder-Reasoning-Budget", "1024"}});
        REQUIRE_MESSAGE(status == 200, doc.dump());
        CHECK(parse(upstream.last_body()).find("reasoning_budget_tokens")->as_int() == 1024);

        std::tie(status, doc) = post(R"({"reasoning_budget_tokens":-2,"messages":[{"role":"user","content":"hi"}]})");
        CHECK(status == 400);
        CHECK(doc.find("error")->find("param")->as_string() == "reasoning_budget_tokens");
        std::tie(status, doc) = post("{\"reasoning_budget_message\":\"" + std::string(513, 'x') +
                                     "\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        CHECK(status == 400);
        CHECK(doc.find("error")->find("param")->as_string() == "reasoning_budget_message");
        srv.stop();
    }

    TEST_CASE("HTTP assistant reasoning history, features, timings and telemetry are additive") {
        sonder_test::FakeLlamaServer upstream;
        upstream.set_props(R"({"default_generation_settings":{"n_ctx":32768},"total_slots":1})");
        upstream.set_body(kTimedChatStream);
        LlamaServerBackendOptions lo;
        lo.base_url = upstream.url();
        server::ServerOptions so;
        so.port = 0;
        so.backend_instance = make_llamaserver_backend(lo);
        so.models = {"fake-model"};
        auto sink = std::make_shared<MemoryTelemetrySink>();
        so.extra_sinks.push_back(sink);
        server::Server srv(so);
        REQUIRE(srv.start().ok());
        httplib::Client client("127.0.0.1", srv.port());
        client.set_read_timeout(60, 0);

        auto response = client.Get("/v1/sonder/health");
        REQUIRE(response);
        const auto health = parse(response->body);
        const auto expected_features =
            R"(["thinking","chat_template_kwargs","enable_thinking","reasoning_effort","reasoning_budget","prompt_cache_key","priority_classes"])";
        const auto *health_sonder = health.find("sonder");
        REQUIRE(health_sonder);
        const auto *health_features = health_sonder->find("features");
        REQUIRE(health_features);
        CHECK(health_features->dump() == expected_features);
        const auto *health_pins = health_sonder->find("pins");
        REQUIRE(health_pins);
        REQUIRE(health_pins->find("mode"));
        CHECK(health_pins->find("mode")->as_string() == "override");
        response = client.Get("/v1/models");
        REQUIRE(response);
        const auto models = parse(response->body);
        const auto *model_sonder = models.find("sonder");
        REQUIRE(model_sonder);
        REQUIRE(model_sonder->find("features"));
        CHECK(model_sonder->find("features")->dump() == expected_features);

        const auto body = R"({"reasoning_budget_tokens":77,"messages":[
            {"role":"assistant","content":"answer","reasoning_content":"private thought"},
            {"role":"user","content":"next"}]})";
        response = client.Post("/v1/chat/completions", {}, body, "application/json");
        REQUIRE(response);
        REQUIRE_MESSAGE(response->status == 200, response->body);
        const auto doc = parse(response->body);
        const auto sent = parse(upstream.last_body());
        CHECK(sent.find("messages")->as_array()[0].find("reasoning_content")->as_string() == "private thought");
        const auto *usage = doc.find("usage");
        REQUIRE(usage);
        const auto *usage_sonder = usage->find("sonder");
        REQUIRE(usage_sonder);
        const auto *usage_timings = usage_sonder->find("timings");
        REQUIRE(usage_timings);
        CHECK(usage_timings->find("prompt_n")->as_int() == 12);
        CHECK(usage_timings->find("cache_n")->as_int() == 9);
        CHECK(usage_timings->find("prompt_ms")->as_double() == doctest::Approx(1.5));
        CHECK(usage_timings->find("predicted_ms")->as_double() == doctest::Approx(2.25));
        const auto *prompt_details = usage->find("prompt_tokens_details");
        REQUIRE(prompt_details);
        CHECK(prompt_details->find("cached_tokens")->as_int() == 5);

        response = client.Post("/v1/chat/completions", {},
            R"({"stream":true,"stream_options":{"include_usage":true},"reasoning_budget_tokens":77,
                "messages":[{"role":"user","content":"stream timings"}]})", "application/json");
        REQUIRE(response);
        REQUIRE_MESSAGE(response->status == 200, response->body);
        CHECK(response->get_header_value("Content-Type").find("text/event-stream") != std::string::npos);
        const auto stream_doc = last_sse_json(response->body);
        const auto* stream_usage = stream_doc.find("usage");
        REQUIRE(stream_usage);
        REQUIRE(stream_usage->find("sonder"));
        REQUIRE(stream_usage->find("sonder")->find("timings"));
        CHECK(stream_usage->find("sonder")->find("timings")->dump() == usage_timings->dump());
        REQUIRE(stream_usage->find("prompt_tokens_details"));
        CHECK(stream_usage->find("prompt_tokens_details")->find("cached_tokens")->as_int() == 5);

        srv.engine()->telemetry().flush();
        bool completed = false;
        for (const auto &line : sink->lines()) {
            const auto event = json::parse(line);
            if (!event.ok() || event->find("event_type")->as_string() != "request.completed")
                continue;
            const auto *attrs = event->find("attributes");
            REQUIRE(attrs);
            REQUIRE(attrs->find("backend_cache_n"));
            REQUIRE(attrs->find("backend_prompt_n"));
            REQUIRE(attrs->find("backend_predicted_n"));
            REQUIRE(attrs->find("backend_draft_n"));
            REQUIRE(attrs->find("backend_draft_accepted"));
            REQUIRE(attrs->find("reasoning_budget"));
            CHECK(attrs->find("backend_cache_n")->as_int() == 9);
            CHECK(attrs->find("backend_prompt_n")->as_int() == 12);
            CHECK(attrs->find("backend_predicted_n")->as_int() == 7);
            CHECK(attrs->find("backend_draft_n")->as_int() == 4);
            CHECK(attrs->find("backend_draft_accepted")->as_int() == 3);
            CHECK(attrs->find("reasoning_budget")->as_int() == 77);
            completed = true;
        }
        CHECK(completed);
        upstream.set_body(kPlainChatStream);
        response = client.Post("/v1/chat/completions", {},
                               R"({"messages":[{"role":"user","content":"no timings"}]})",
                               "application/json");
        REQUIRE(response);
        REQUIRE(response->status == 200);
        const auto no_timing_doc = parse(response->body);
        const auto *no_timing_usage = no_timing_doc.find("usage");
        REQUIRE(no_timing_usage);
        CHECK(no_timing_usage->find("sonder") == nullptr);
        srv.stop();
    }

    TEST_CASE("HTTP empty length responses warn for streaming and non-streaming clients") {
        sonder_test::FakeLlamaServer upstream;
        upstream.set_body(kEmptyLengthStream);
        LlamaServerBackendOptions lo;
        lo.base_url = upstream.url();
        server::ServerOptions so;
        so.port = 0;
        so.backend_instance = make_llamaserver_backend(lo);
        so.models = {"fake-model"};
        server::Server srv(so);
        REQUIRE(srv.start().ok());
        httplib::Client client("127.0.0.1", srv.port());
        client.set_read_timeout(60, 0);
        const auto post = [&](const std::string &body) {
            auto response = client.Post("/v1/chat/completions", {}, body, "application/json");
            REQUIRE(response);
            return std::pair<int, json::Value>{response->status, parse(response->body)};
        };
        int status = 0;
        json::Value doc;
        // The streaming response is SSE; inspect its final JSON frame rather
        // than attempting to parse the complete wire body as one object.
        auto stream_response = client.Post("/v1/chat/completions", {},
                                           R"({"stream":true,"messages":[{"role":"user","content":"hi"}]})",
                                           "application/json");
        REQUIRE(stream_response);
        REQUIRE_MESSAGE(stream_response->status == 200, stream_response->body);
        CHECK(stream_response->get_header_value("Content-Type").find("text/event-stream") != std::string::npos);
        CHECK(stream_response->body.find("data: [DONE]") != std::string::npos);
        const auto stream_doc = last_sse_json(stream_response->body);
        const auto *stream_sonder = stream_doc.find("sonder");
        REQUIRE(stream_sonder);
        const auto *stream_warnings = stream_sonder->find("warnings");
        REQUIRE(stream_warnings);
        REQUIRE(stream_warnings->as_array().size() == 1);
        CHECK(stream_warnings->as_array()[0].as_string() == "empty_content_at_length");
        std::tie(status, doc) = post(R"({"stream":false,"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE_MESSAGE(status == 200, doc.dump());
        const auto *nonstream_sonder = doc.find("sonder");
        REQUIRE(nonstream_sonder);
        const auto *nonstream_warnings = nonstream_sonder->find("warnings");
        REQUIRE(nonstream_warnings);
        REQUIRE(nonstream_warnings->as_array().size() == 1);
        CHECK(nonstream_warnings->as_array()[0].as_string() == "empty_content_at_length");
        srv.stop();
    }

    TEST_CASE("the HTTP API forwards bridged turns: num_ctx, slot keys, thinking pins and cache usage") {
        sonder_test::FakeLlamaServer upstream;
        upstream.set_props(R"({"default_generation_settings":{"n_ctx":32768},"total_slots":4})");
        upstream.set_body(kChatStream);
        LlamaServerBackendOptions lo;
        lo.base_url = upstream.url();
        server::ServerOptions so;
        so.port = 0;
        so.backend_instance = make_llamaserver_backend(lo);
        so.models = {"fake-model"};
        so.pin_enable_thinking = false;
        server::Server srv(so);
        REQUIRE(srv.start().ok());
        httplib::Client client("127.0.0.1", srv.port());
        client.set_read_timeout(60, 0);
        int status = 0;
        json::Value doc;
        const auto post = [&](const std::string &body, const httplib::Headers &headers = {}) {
            auto res = client.Post("/v1/chat/completions", headers, body, "application/json");
            REQUIRE(res);
            status = res->status;
            doc = parse(res->body);
        };
        const httplib::Headers agent_a{{"X-Sonder-Run-Id", "run-1"}, {"X-Sonder-Agent-Id", "agent-a"}};
        const httplib::Headers agent_b{{"X-Sonder-Run-Id", "run-1"}, {"X-Sonder-Agent-Id", "agent-b"}};

        // A Runtime-bridged turn: positive num_ctx within the served context.
        post(R"({"model":"fake-model","num_ctx":32768,
            "messages":[{"role":"user","content":"hi"}]})", agent_a);
        REQUIRE_MESSAGE(status == 200, json::Value(doc).dump());
        CHECK(last_slot(upstream) == 0);
        const auto *usage = doc.find("usage");
        REQUIRE(usage->find("prompt_tokens_details"));
        CHECK(usage->find("prompt_tokens_details")->find("cached_tokens")->as_int() == 9);
        CHECK(usage->find("prompt_tokens")->as_int() == 12);
        const auto *timings = doc.find("timings");
        CHECK(timings->find("cache_n")->as_int() == 9);
        CHECK(timings->find("draft_n")->as_int() == 4);
        CHECK(timings->find("draft_n_accepted")->as_int() == 3);
        CHECK(timings->find("queue_ms") != nullptr);  // admitted by the scheduler (account mode)
        // The pin is forwarded even when the request does not ask.
        auto sent = parse(upstream.last_body());
        CHECK_FALSE(sent.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
        CHECK(doc.find("sonder")->find("warnings") == nullptr);

        // Another agent of the same run gets its own slot; the first keeps slot 0.
        post(R"({"messages":[{"role":"user","content":"hi"}]})", agent_b);
        REQUIRE(status == 200);
        CHECK(last_slot(upstream) == 1);
        post(R"({"messages":[{"role":"user","content":"hi"}]})", agent_a);
        REQUIRE(status == 200);
        CHECK(last_slot(upstream) == 0);
        // prompt_cache_key wins over the correlation headers.
        post(R"({"prompt_cache_key":"conv-7","messages":[{"role":"user","content":"hi"}]})", agent_a);
        REQUIRE(status == 200);
        CHECK(last_slot(upstream) == 2);

        // A conflicting thinking request is overridden with a warning.
        post(R"({"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"},
            "messages":[{"role":"user","content":"hi"}]})");
        REQUIRE(status == 200);
        sent = parse(upstream.last_body());
        CHECK_FALSE(sent.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
        CHECK(sent.find("chat_template_kwargs")->find("reasoning_effort")->as_string() == "low");
        REQUIRE(doc.find("sonder")->find("warnings"));
        CHECK(doc.find("sonder")->find("warnings")->as_array().size() == 1);

        // num_ctx beyond the served context: 400 naming both numbers, no stream.
        const auto streamed = upstream.bodies().size();
        post(R"({"num_ctx":65536,"messages":[{"role":"user","content":"hi"}]})");
        CHECK(status == 400);
        const std::string message = doc.find("error")->find("message")->as_string();
        CHECK(message.find("65536") != std::string::npos);
        CHECK(message.find("32768") != std::string::npos);
        CHECK(upstream.bodies().size() == streamed);

        // Malformed extensions are refused before anything runs.
        post(R"({"think":"yes","messages":[{"role":"user","content":"hi"}]})");
        CHECK(status == 400);
        post(R"({"chat_template_kwargs":[],"messages":[{"role":"user","content":"hi"}]})");
        CHECK(status == 400);

        // An upstream without cache counters: no cached-token fields at all.
        upstream.set_body(kPlainChatStream);
        post(R"({"messages":[{"role":"user","content":"hi"}]})");
        REQUIRE(status == 200);
        CHECK(doc.find("usage")->find("prompt_tokens_details") == nullptr);
        CHECK(doc.find("timings")->find("cache_n") == nullptr);
        CHECK(doc.find("timings")->find("draft_n") == nullptr);
        srv.stop();
    }
#endif
}
