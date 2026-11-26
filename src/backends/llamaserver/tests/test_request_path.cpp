// llama-server request path: served-context num_ctx, cache_prompt, slot
// affinity, thinking controls and the OpenAI usage/timings they surface.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "../llamaserver_protocol.hpp"
#include "../llamaserver_stream.hpp"
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

    TEST_CASE("a single-slot server leaves every keyed chat unpinned") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":1})");
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));
        for (const char *key : {"a", "b", "c", "a"}) {
            REQUIRE(model->chat(chat(key), {}, {}).ok());
            CHECK(last_slot(server) == -1);
            const auto body = parse(server.last_body());
            REQUIRE(body.find("cache_prompt"));
            CHECK(body.find("cache_prompt")->as_bool());
        }
    }

    TEST_CASE("overlapping keyed chats never pin a busy slot and reuse ownership after release") {
        sonder_test::FakeLlamaServer server;
        server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":2})");
        server.set_body(kPlainChatStream);
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));
        int a_callbacks = 0, b_callbacks = 0;
        // Reentrant calls keep both outer request leases alive without sleeps
        // or timing races. The fake handles HTTP only, never slot selection.
        auto a = model->chat(chat("a"), {}, [&](const TokenChunk &) {
            ++a_callbacks;
            CHECK(last_slot(server) == 0);
            auto b = model->chat(chat("b"), {}, [&](const TokenChunk &) {
                ++b_callbacks;
                CHECK(last_slot(server) == 1);
                CHECK(model->chat(chat("a"), {}, {}).ok());
                CHECK(last_slot(server) == -1); // known busy key
                CHECK(model->chat(chat("c"), {}, {}).ok());
                CHECK(last_slot(server) == -1); // no new key can take a busy slot
                CHECK(model->chat(chat("a"), {}, {}).ok());
                CHECK(last_slot(server) == -1); // unpinned completion did not release A
                return true;
            });
            CHECK(b.ok());
            CHECK(model->chat(chat("c"), {}, {}).ok());
            CHECK(last_slot(server) == 1); // steal B's idle slot, never A's busy slot
            return true;
        });
        REQUIRE(a.ok());
        CHECK(a_callbacks == 1);
        CHECK(b_callbacks == 1);
        REQUIRE(model->chat(chat("a"), {}, {}).ok());
        CHECK(last_slot(server) == 0);
        CHECK(server.bodies().size() == 7);
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

    TEST_CASE("Timings overload resets its output on every error path") {
        const char *const invalid[] = {
            R"({"timings":5})",
            R"({"timings":{"prompt_n":3,"draft_n":2,"draft_n_accepted":3}})",
            R"({"timings":{"prompt_n":3,"predicted_per_second":-1}})",
            R"({"timings":{"prompt_n":3,"prompt_ms":-1}})",
            R"({"timings":{"prompt_n":3,"predicted_ms":"x"}})",
            R"({"timings":{"prompt_n":3,"predicted_n":-1}})"};
        for (const char *text : invalid) {
            CAPTURE(text);
            llamaserver::Timings timings;
            REQUIRE(llamaserver::parse_timings(parse(R"({"timings":{"prompt_n":9,"predicted_per_second":4}})"), timings)
                        .ok());
            REQUIRE(timings.present);
            CHECK_FALSE(llamaserver::parse_timings(parse(text), timings).ok());
            CHECK_FALSE(timings.present);
            CHECK_FALSE(timings.prompt_present);
            CHECK(timings.prompt_tokens == 0);
            CHECK(timings.predicted_tokens_per_second == 0.0);
            CHECK_FALSE(timings.backend_timings.has_value());
        }
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

    TEST_CASE("slot affinity keeps ownership, evicts the least recently used idle key and is bounded") {
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
            // A known busy key keeps ownership without pinning the request.
            auto x2 = a.acquire("x", 2);
            CHECK_FALSE(x2.slot());
            CHECK(a.owned_slot("x") == std::optional<std::uint32_t>(0));
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

TEST_SUITE("llamaserver_slot_affinity") {
    TEST_CASE("empty keys and unknown counts do not consume or release existing slots") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 2);
        CHECK_FALSE(affinity.acquire("", 3).slot());
        CHECK_FALSE(affinity.acquire("b", 0).slot());
        CHECK(affinity.keys() == 1);
        CHECK(affinity.owned_slot("a") == std::optional<std::uint32_t>(0));
        CHECK_FALSE(affinity.acquire("a", 2).slot());
        auto b = affinity.acquire("b", 2);
        CHECK(b.slot() == std::optional<std::uint32_t>(1));
    }

    TEST_CASE("slot counts above the cap share a bounded layout without releasing busy slots") {
        llamaserver::SlotAffinity affinity;
        constexpr auto cap = llamaserver::SlotAffinity::kMaxSlots;
        auto a = affinity.acquire("a", cap + 1u, {cap, cap - 1u});
        CHECK(a.slot() == std::optional<std::uint32_t>(cap - 1u));
        auto b = affinity.acquire("b", cap + 2u, {cap});
        CHECK(b.slot() == std::optional<std::uint32_t>(0));
        CHECK_FALSE(affinity.acquire("a", cap).slot());
        CHECK(affinity.keys() == 2);
        a.reset();
        auto resumed = affinity.acquire("a", cap);
        CHECK(resumed.slot() == std::optional<std::uint32_t>(cap - 1u));
    }

    TEST_CASE("lease outcomes distinguish skipped busy new hit and stolen requests") {
        llamaserver::SlotAffinity affinity;
        CHECK(affinity.acquire("", 2).outcome() == "none");
        CHECK(affinity.acquire("a", 0).outcome() == "none");
        auto single = affinity.acquire("a", 1);
        CHECK(single.outcome() == "single_slot");
        CHECK_FALSE(single.slot());
        auto a = affinity.acquire("a", 2);
        CHECK(a.outcome() == "new");
        CHECK(a.slot() == std::optional<std::uint32_t>(0));
        auto again = affinity.acquire("a", 2);
        CHECK(again.outcome() == "busy_unpinned");
        CHECK_FALSE(again.slot());
        auto b = affinity.acquire("b", 2);
        CHECK(b.outcome() == "new");
        auto full = affinity.acquire("c", 2);
        CHECK(full.outcome() == "busy_unpinned");
        CHECK_FALSE(full.slot());
        a.reset();
        auto hit = affinity.acquire("a", 2);
        CHECK(hit.outcome() == "hit");
        CHECK(hit.slot() == std::optional<std::uint32_t>(0));
        b.reset();
        auto stolen = affinity.acquire("c", 2);
        CHECK(stolen.outcome() == "stolen");
        CHECK(stolen.slot() == std::optional<std::uint32_t>(1));
        CHECK_FALSE(affinity.owned_slot("b"));
    }

    TEST_CASE("lease moves carry pinned and unpinned outcomes and reset clears them") {
        llamaserver::SlotAffinity affinity;
        llamaserver::SlotAffinity::Lease lease;
        CHECK(lease.outcome() == "none");
        auto a = affinity.acquire("a", 2);
        lease = std::move(a);
        CHECK(lease.outcome() == "new");
        CHECK(a.outcome() == "none");
        auto busy = affinity.acquire("a", 2);
        auto moved_busy = std::move(busy);
        CHECK(moved_busy.outcome() == "busy_unpinned");
        CHECK_FALSE(moved_busy.slot());
        CHECK(busy.outcome() == "none");
        lease = std::move(moved_busy); // release the destination's pinned slot
        CHECK(lease.outcome() == "busy_unpinned");
        CHECK_FALSE(lease.slot());
        CHECK(moved_busy.outcome() == "none");
        CHECK(affinity.acquire("a", 2).outcome() == "hit");
        lease.reset();
        CHECK(lease.outcome() == "none");
    }

    TEST_CASE("one slot never pins or assigns ownership even with a warmed preference") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 1);
        auto b = affinity.acquire("b", 1, {0});
        auto again = affinity.acquire("a", 1);
        CHECK_FALSE(a.slot());
        CHECK_FALSE(b.slot());
        CHECK_FALSE(again.slot());
        CHECK_FALSE(affinity.owned_slot("a"));
        CHECK(affinity.keys() == 0);
    }

    TEST_CASE("a busy key stays unpinned even when another slot is idle") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 2);
        REQUIRE(a.slot() == std::optional<std::uint32_t>(0));
        auto again = affinity.acquire("a", 2);
        CHECK_FALSE(again.slot());
        CHECK(affinity.owned_slot("a") == a.slot());
        again.reset();
        CHECK_FALSE(affinity.acquire("a", 2).slot());
        auto b = affinity.acquire("b", 2);
        REQUIRE(b.slot() == std::optional<std::uint32_t>(1));
        CHECK_FALSE(affinity.acquire("c", 2).slot());
        CHECK_FALSE(affinity.owned_slot("c"));
        a.reset();
        auto resumed = affinity.acquire("a", 2);
        CHECK(resumed.slot() == std::optional<std::uint32_t>(0));
        CHECK(affinity.keys() == 2);
    }

    TEST_CASE("stealing chooses the least recently used idle slot and skips busy owners") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 3);
        auto b = affinity.acquire("b", 3);
        auto c = affinity.acquire("c", 3);
        b.reset();
        c.reset();
        auto d = affinity.acquire("d", 3);
        CHECK(d.slot() == std::optional<std::uint32_t>(1));
        CHECK_FALSE(affinity.owned_slot("b"));
        CHECK(affinity.owned_slot("a") == std::optional<std::uint32_t>(0));
        auto e = affinity.acquire("e", 3);
        CHECK(e.slot() == std::optional<std::uint32_t>(2));
        CHECK_FALSE(affinity.owned_slot("c"));
        CHECK_FALSE(affinity.acquire("f", 3).slot());
        CHECK(affinity.keys() == 3);
    }

    TEST_CASE("warmed preferences never take owned busy slots or redirect known keys") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 3, {99, 2});
        CHECK(a.slot() == std::optional<std::uint32_t>(2));
        auto b = affinity.acquire("b", 3, {2, 99, 1});
        CHECK(b.slot() == std::optional<std::uint32_t>(1));
        auto c = affinity.acquire("c", 3, {2, 1, 99});
        CHECK(c.slot() == std::optional<std::uint32_t>(0));
        CHECK_FALSE(affinity.acquire("a", 3, {0}).slot());
        a.reset();
        CHECK(affinity.acquire("a", 3, {0}).slot() == std::optional<std::uint32_t>(2));
    }

    TEST_CASE("moves transfer the busy mark and overwriting releases the previous lease") {
        llamaserver::SlotAffinity affinity;
        auto a = affinity.acquire("a", 2);
        auto b = affinity.acquire("b", 2);
        auto moved = std::move(a);
        CHECK_FALSE(a.slot());
        a.reset();
        CHECK_FALSE(affinity.acquire("a", 2).slot());
        moved = std::move(b);
        CHECK_FALSE(b.slot());
        CHECK(moved.slot() == std::optional<std::uint32_t>(1));
        auto resumed = affinity.acquire("a", 2);
        CHECK(resumed.slot() == std::optional<std::uint32_t>(0));
        CHECK_FALSE(affinity.acquire("b", 2).slot());
        moved.reset();
        moved.reset();
        CHECK(affinity.acquire("b", 2).slot() == std::optional<std::uint32_t>(1));
    }

    TEST_CASE("one-slot transitions invalidate old leases before returning to multiple slots") {
        llamaserver::SlotAffinity affinity;
        auto stale = affinity.acquire("old", 2);
        auto single = affinity.acquire("single", 1);
        CHECK_FALSE(single.slot());
        CHECK(affinity.keys() == 0);
        auto current = affinity.acquire("current", 2);
        REQUIRE(current.slot() == std::optional<std::uint32_t>(0));
        stale.reset();
        CHECK_FALSE(affinity.acquire("current", 2).slot());
        CHECK_FALSE(affinity.owned_slot("old"));
        current.reset();
        CHECK(affinity.acquire("current", 2).slot() == std::optional<std::uint32_t>(0));
    }
}

TEST_SUITE("llamaserver_tool_calling_and_stream") {
    TEST_CASE("build_chat_body emits default tool parameter schema when empty or omitted") {
        ChatRequest r = chat();
        r.tools.push_back(ToolDefinition{"no_params", "No parameters", ""});
        r.tools.push_back(ToolDefinition{"empty_obj", "Empty object schema", "{}"});
        r.tools.push_back(ToolDefinition{"bad_json", "Unparseable schema", "{not-json"});
        r.tools.push_back(ToolDefinition{
            "with_schema", "Explicit schema",
            R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"});

        CHECK(llamaserver::chat_tools_requested(r));
        const auto body = llamaserver::build_chat_body("m", r);
        const auto *tools = body.find("tools");
        REQUIRE(tools != nullptr);
        REQUIRE(tools->is_array());
        REQUIRE(tools->as_array().size() == 4u);

        for (std::size_t i = 0; i < 3u; ++i) {
            const auto *fn = tools->as_array()[i].find("function");
            REQUIRE(fn != nullptr);
            const auto *params = fn->find("parameters");
            REQUIRE(params != nullptr);
            REQUIRE(params->is_object());
            REQUIRE(params->find("type") != nullptr);
            CHECK(params->find("type")->as_string() == "object");
            REQUIRE(params->find("properties") != nullptr);
            CHECK(params->find("properties")->is_object());
            CHECK(params->find("properties")->as_object().empty());
        }

        const auto *explicit_params = tools->as_array()[3].find("function")->find("parameters");
        REQUIRE(explicit_params != nullptr);
        REQUIRE(explicit_params->find("properties") != nullptr);
        REQUIRE(explicit_params->find("properties")->find("city") != nullptr);
        REQUIRE(explicit_params->find("required") != nullptr);
        CHECK(explicit_params->find("required")->as_array().size() == 1u);
    }

    TEST_CASE("build_chat_body omits tools and tool_choice when tool_choice is none or tools empty") {
        ChatRequest r = chat();
        r.tools.push_back(ToolDefinition{"get_weather", "Get weather", ""});
        r.tool_choice = ToolChoice{ToolChoiceMode::none, ""};
        r.grammar = "root ::= \"ok\"";

        CHECK_FALSE(llamaserver::chat_tools_requested(r));
        const auto none_body = llamaserver::build_chat_body("m", r);
        CHECK(none_body.find("tools") == nullptr);
        CHECK(none_body.find("tool_choice") == nullptr);
        REQUIRE(none_body.find("grammar") != nullptr);
        CHECK(none_body.find("grammar")->as_string() == "root ::= \"ok\"");

        // Orphan tool_choice without tools is also omitted.
        ChatRequest empty_tools = chat();
        empty_tools.tool_choice = ToolChoice{ToolChoiceMode::auto_choice, ""};
        CHECK_FALSE(llamaserver::chat_tools_requested(empty_tools));
        const auto empty_body = llamaserver::build_chat_body("m", empty_tools);
        CHECK(empty_body.find("tools") == nullptr);
        CHECK(empty_body.find("tool_choice") == nullptr);

        // Active modes transmit both tools and tool_choice and suppress grammar.
        r.tool_choice = ToolChoice{ToolChoiceMode::required, ""};
        CHECK(llamaserver::chat_tools_requested(r));
        const auto req_body = llamaserver::build_chat_body("m", r);
        REQUIRE(req_body.find("tools") != nullptr);
        REQUIRE(req_body.find("tool_choice") != nullptr);
        CHECK(req_body.find("tool_choice")->as_string() == "required");
        CHECK(req_body.find("grammar") == nullptr);

        r.tool_choice = ToolChoice{ToolChoiceMode::auto_choice, ""};
        const auto auto_body = llamaserver::build_chat_body("m", r);
        REQUIRE(auto_body.find("tools") != nullptr);
        REQUIRE(auto_body.find("tool_choice") != nullptr);
        CHECK(auto_body.find("tool_choice")->as_string() == "auto");
    }

    TEST_CASE("stream tool parsing relies on explicit tools_requested, never prompt substring matching") {
        sonder_test::FakeLlamaServer server;
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));

        // 1. User prompt containing literal `"tools":[` must NOT enable tool calling.
        ChatRequest spoofed;
        spoofed.messages = {{"user", R"(Explain {"tools":[{"type":"function"}]} and "tools": [] in JSON.)"}};
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\","
            "\"function\":{\"name\":\"evil\",\"arguments\":\"{}\"}}]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
            "data: [DONE]\n\n");
        std::vector<ToolCallDelta> seen_deltas;
        auto spoofed_res = model->chat(spoofed, {}, [&](const TokenChunk &c) {
            seen_deltas.insert(seen_deltas.end(), c.tool_call_deltas.begin(), c.tool_call_deltas.end());
            return true;
        });
        REQUIRE_FALSE(spoofed_res.ok());
        CHECK(spoofed_res.status().code() == ErrorCode::unsupported);
        CHECK(seen_deltas.empty());

        // 2. tools non-empty + tool_choice="none" must NOT transmit tools or accept tool calls.
        ChatRequest disabled = chat();
        disabled.tools.push_back(ToolDefinition{"lookup", "Lookup", ""});
        disabled.tool_choice = ToolChoice{ToolChoiceMode::none, ""};
        seen_deltas.clear();
        auto disabled_res = model->chat(disabled, {}, [&](const TokenChunk &c) {
            seen_deltas.insert(seen_deltas.end(), c.tool_call_deltas.begin(), c.tool_call_deltas.end());
            return true;
        });
        REQUIRE_FALSE(disabled_res.ok());
        CHECK(disabled_res.status().code() == ErrorCode::unsupported);
        CHECK(seen_deltas.empty());
        const auto disabled_wire = parse(server.last_body());
        CHECK(disabled_wire.find("tools") == nullptr);
        CHECK(disabled_wire.find("tool_choice") == nullptr);

        // 3. Explicit active tools succeed and deliver tool_call_deltas.
        ChatRequest active = chat();
        active.tools.push_back(ToolDefinition{"lookup", "Lookup", ""});
        active.tool_choice = ToolChoice{ToolChoiceMode::auto_choice, ""};
        seen_deltas.clear();
        auto active_res = model->chat(active, {}, [&](const TokenChunk &c) {
            seen_deltas.insert(seen_deltas.end(), c.tool_call_deltas.begin(), c.tool_call_deltas.end());
            return true;
        });
        REQUIRE_MESSAGE(active_res.ok(), active_res.status().to_string());
        CHECK(active_res->stop_reason == StopReason::tool_calls);
        REQUIRE(seen_deltas.size() == 1u);
        CHECK(seen_deltas[0].id == "call_1");
        CHECK(seen_deltas[0].name == "evil");
        CHECK(seen_deltas[0].arguments_delta == "{}");
    }

    TEST_CASE("stream serializes object-typed function.arguments and rejects invalid argument types") {
        sonder_test::FakeLlamaServer server;
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));

        ChatRequest req = chat();
        req.tools.push_back(ToolDefinition{"get_weather", "Weather", ""});
        req.tools.push_back(ToolDefinition{"ping", "Ping", ""});

        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":["
            "{\"index\":0,\"id\":\"call_obj\",\"function\":{\"name\":\"get_weather\","
            "\"arguments\":{\"city\":\"Austin\",\"units\":\"c\"}}},"
            "{\"index\":1,\"id\":\"call_empty\",\"function\":{\"name\":\"ping\",\"arguments\":{}}}"
            "]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
            "data: [DONE]\n\n");

        std::vector<ToolCallDelta> deltas;
        auto res = model->chat(req, {}, [&](const TokenChunk &c) {
            deltas.insert(deltas.end(), c.tool_call_deltas.begin(), c.tool_call_deltas.end());
            return true;
        });
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        CHECK(res->stop_reason == StopReason::tool_calls);
        REQUIRE(deltas.size() == 2u);
        CHECK(deltas[0].index == 0u);
        CHECK(deltas[0].id == "call_obj");
        CHECK(deltas[0].name == "get_weather");
        const auto parsed_args = parse(deltas[0].arguments_delta);
        REQUIRE(parsed_args.is_object());
        REQUIRE(parsed_args.find("city") != nullptr);
        CHECK(parsed_args.find("city")->as_string() == "Austin");
        REQUIRE(parsed_args.find("units") != nullptr);
        CHECK(parsed_args.find("units")->as_string() == "c");
        CHECK(deltas[1].index == 1u);
        CHECK(deltas[1].id == "call_empty");
        CHECK(deltas[1].name == "ping");
        CHECK(deltas[1].arguments_delta == "{}");

        // Non-string, non-object, non-null arguments fail closed with protocol_error.
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":["
            "{\"index\":0,\"id\":\"call_bad\",\"function\":{\"name\":\"ping\",\"arguments\":[1,2]}}"
            "]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
            "data: [DONE]\n\n");
        auto bad_res = model->chat(req, {}, {});
        REQUIRE_FALSE(bad_res.ok());
        CHECK(bad_res.status().code() == ErrorCode::protocol_error);
    }

    TEST_CASE("stream enforces M-16 cumulative byte cap on accumulated tool_call_deltas") {
        sonder_test::FakeLlamaServer server;
        const auto url = net::parse_url(server.url());
        REQUIRE(url.ok());

        const auto run_direct = [&](std::size_t cap, std::size_t &delivered_bytes) {
            net::HttpRequest http;
            http.method = "POST";
            http.host = url->host;
            http.port = url->port;
            http.target = "/v1/chat/completions";
            http.body = "{}";
            std::uint64_t progress = 0;
            delivered_bytes = 0;
            return llamaserver::run_completion_stream(
                http, /*chat=*/true, /*native=*/false, /*separate_reasoning=*/false, {},
                [&](const TokenChunk &c) {
                    for (const auto &d : c.tool_call_deltas)
                        delivered_bytes += d.id.size() + d.name.size() + d.arguments_delta.size();
                    return true;
                },
                progress, /*tools_requested=*/true, cap);
        };

        // Two deltas: ("c1" + "fn" + "{\"a\":1}") = 2 + 2 + 7 = 11 bytes, then ("{\"b\":2}") = 7 bytes => 18 bytes.
        server.set_body(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
            "\"function\":{\"name\":\"fn\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"
            "\"function\":{\"arguments\":{\"b\":2}}}]}}]}\n\n"
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
            "data: [DONE]\n\n");

        std::size_t delivered = 0;
        auto exact_ok = run_direct(18u, delivered);
        REQUIRE_MESSAGE(exact_ok.ok(), exact_ok.status().to_string());
        CHECK(delivered == 18u);

        auto over_cap = run_direct(17u, delivered);
        REQUIRE_FALSE(over_cap.ok());
        CHECK(over_cap.status().code() == ErrorCode::protocol_error);
        CHECK(delivered == 11u);  // overflowing second delta was never delivered

        // End-to-end BackendModel::chat() enforces the default kMaxToolCallDeltaBytes cap across frames.
        const std::string chunk_arg(32u * 1024u, 'a');
        std::string big_stream;
        big_stream.reserve(llamaserver::kMaxToolCallDeltaBytes + 128u * 1024u);
        const std::size_t frames = (llamaserver::kMaxToolCallDeltaBytes / chunk_arg.size()) + 2u;
        for (std::size_t i = 0; i < frames; ++i) {
            big_stream += "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"";
            big_stream += chunk_arg;
            big_stream += "\"}}]}}]}\n\n";
        }
        big_stream += "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
                      "data: [DONE]\n\n";
        server.set_body(std::move(big_stream));

        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));
        ChatRequest req = chat();
        req.tools.push_back(ToolDefinition{"big_tool", "Big", ""});
        std::size_t chat_bytes = 0;
        auto big_res = model->chat(req, {}, [&](const TokenChunk &c) {
            for (const auto &d : c.tool_call_deltas)
                chat_bytes += d.id.size() + d.name.size() + d.arguments_delta.size();
            return true;
        });
        REQUIRE_FALSE(big_res.ok());
        CHECK(big_res.status().code() == ErrorCode::protocol_error);
        CHECK(chat_bytes <= llamaserver::kMaxToolCallDeltaBytes);
    }

    TEST_CASE("native completion preserves total prompt_tokens and cached_tokens <= prompt_tokens invariant") {
        sonder_test::FakeLlamaServer server;
        LlamaServerBackendOptions o;
        o.base_url = server.url();
        auto model = load(make_llamaserver_backend(o));

        GenerateRequest req;
        req.prompt = "hello";

        // 1. Both timings (prompt_n + cache_n) and top-level tokens_evaluated/tokens_cached present.
        server.set_body(
            "data: {\"content\":\"ok\",\"stop\":true,\"tokens_evaluated\":3,\"tokens_cached\":14,"
            "\"tokens_predicted\":2,\"timings\":{\"prompt_n\":3,\"cache_n\":9,\"predicted_n\":2}}\n\n");
        auto r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 12u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 9u);
        CHECK(*r->cached_tokens <= r->prompt_tokens);
        CHECK(r->completion_tokens == 2u);
        REQUIRE(r->backend_timings.has_value());
        CHECK(r->backend_timings->prompt_n == std::optional<std::uint64_t>(3u));
        CHECK(r->backend_timings->cache_n == std::optional<std::uint64_t>(9u));

        // 2. 100% cache hit: timings.prompt_n == 0, cache_n == 16, tokens_evaluated == 0.
        server.set_body(
            "data: {\"content\":\"ok\",\"stop\":true,\"tokens_evaluated\":0,\"tokens_predicted\":1,"
            "\"timings\":{\"prompt_n\":0,\"cache_n\":16,\"predicted_n\":1}}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 16u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 16u);
        CHECK(*r->cached_tokens <= r->prompt_tokens);

        // 3. Top-level tokens_evaluated + tokens_cached without timings.
        server.set_body(
            "data: {\"content\":\"ok\",\"stop\":true,\"tokens_evaluated\":4,\"tokens_cached\":9,"
            "\"tokens_predicted\":2}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 13u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 9u);
        CHECK(*r->cached_tokens <= r->prompt_tokens);
        CHECK(r->completion_tokens == 2u);

        // 4. Timings prompt_n without cache_n + top-level tokens_cached.
        server.set_body(
            "data: {\"content\":\"ok\",\"stop\":true,\"tokens_evaluated\":4,\"tokens_cached\":9,"
            "\"timings\":{\"prompt_n\":4,\"predicted_n\":1}}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 13u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 9u);
        CHECK(*r->cached_tokens <= r->prompt_tokens);

        // 5. Split frames: tokens_evaluated repeated, tokens_cached arrives later (idempotent, no double-add).
        server.set_body(
            "data: {\"content\":\"a\",\"stop\":false,\"tokens_evaluated\":3,\"tokens_predicted\":1}\n\n"
            "data: {\"content\":\"b\",\"stop\":true,\"tokens_evaluated\":3,\"tokens_cached\":9,"
            "\"tokens_predicted\":2}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 12u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 9u);
        CHECK(r->completion_tokens == 2u);

        // 6. Timings on earlier frame, later stop frame carries only tokens_evaluated + tokens_predicted.
        server.set_body(
            "data: {\"content\":\"a\",\"stop\":false,\"timings\":{\"prompt_n\":3,\"cache_n\":9,\"predicted_n\":1}}\n\n"
            "data: {\"content\":\"b\",\"stop\":true,\"tokens_evaluated\":3,\"tokens_predicted\":2}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_MESSAGE(r.ok(), r.status().to_string());
        CHECK(r->prompt_tokens == 12u);
        REQUIRE(r->cached_tokens.has_value());
        CHECK(*r->cached_tokens == 9u);
        CHECK(r->completion_tokens == 2u);

        // 7. Overflowing tokens_evaluated + tokens_cached fails closed.
        const std::string max_u64 = std::to_string(std::numeric_limits<std::uint64_t>::max());
        server.set_body("data: {\"content\":\"x\",\"stop\":true,\"tokens_evaluated\":" + max_u64 +
                        ",\"tokens_cached\":1}\n\n");
        r = model->generate(req, {}, {});
        REQUIRE_FALSE(r.ok());
        CHECK(r.status().code() == ErrorCode::protocol_error);
    }

    TEST_CASE("build_chat_body preserves interleaved content part ordering and legacy compatibility") {
        // 1. Interleaved (text1, image1, text2, image2, text3) preserves exact order and MIME types.
        ChatRequest r;
        ChatMessage user;
        user.role = "user";
        user.content = "text1text2text3";
        MediaContent img1{"image/png", {1, 2, 3}};
        MediaContent img2{"image/jpeg", {4, 5, 6}};
        user.media = {img1, img2};
        user.parts = {
            TextContentPart{"text1"},
            ImageContentPart{img1},
            TextContentPart{""},  // skipped when images exist
            TextContentPart{"text2"},
            ImageContentPart{img2},
            TextContentPart{"text3"},
        };
        r.messages.push_back(user);

        const auto body = llamaserver::build_chat_body("m", r);
        const auto& msgs = body.find("messages")->as_array();
        REQUIRE(msgs.size() == 1u);
        const auto* content = msgs[0].find("content");
        REQUIRE(content != nullptr);
        REQUIRE(content->is_array());
        const auto& parts = content->as_array();
        REQUIRE(parts.size() == 5u);
        CHECK(parts[0].find("type")->as_string() == "text");
        CHECK(parts[0].find("text")->as_string() == "text1");
        CHECK(parts[1].find("type")->as_string() == "image_url");
        CHECK(parts[1].find("image_url")->find("url")->as_string() == "data:image/png;base64,AQID");
        CHECK(parts[2].find("type")->as_string() == "text");
        CHECK(parts[2].find("text")->as_string() == "text2");
        CHECK(parts[3].find("type")->as_string() == "image_url");
        CHECK(parts[3].find("image_url")->find("url")->as_string() == "data:image/jpeg;base64,BAUG");
        CHECK(parts[4].find("type")->as_string() == "text");
        CHECK(parts[4].find("text")->as_string() == "text3");

        // 2. Text-only parts collapse cleanly to a plain string.
        ChatRequest text_only;
        ChatMessage txt_msg;
        txt_msg.role = "user";
        txt_msg.parts = {TextContentPart{"hello "}, TextContentPart{"world"}};
        text_only.messages.push_back(txt_msg);
        const auto txt_body = llamaserver::build_chat_body("m", text_only);
        const auto* txt_content = txt_body.find("messages")->as_array()[0].find("content");
        REQUIRE(txt_content != nullptr);
        REQUIRE(txt_content->is_string());
        CHECK(txt_content->as_string() == "hello world");

        // 3. Image-only parts omit text items.
        ChatRequest img_only;
        ChatMessage io_msg;
        io_msg.role = "user";
        io_msg.parts = {ImageContentPart{img1}};
        img_only.messages.push_back(io_msg);
        const auto io_body = llamaserver::build_chat_body("m", img_only);
        const auto& io_parts = io_body.find("messages")->as_array()[0].find("content")->as_array();
        REQUIRE(io_parts.size() == 1u);
        CHECK(io_parts[0].find("type")->as_string() == "image_url");

        // 4. Legacy fallback (empty parts, content + media) preserves [text, media...] layout.
        ChatRequest legacy;
        ChatMessage leg_msg;
        leg_msg.role = "user";
        leg_msg.content = "legacy";
        leg_msg.media = {img1, img2};
        legacy.messages.push_back(leg_msg);
        const auto leg_body = llamaserver::build_chat_body("m", legacy);
        const auto& leg_parts = leg_body.find("messages")->as_array()[0].find("content")->as_array();
        REQUIRE(leg_parts.size() == 3u);
        CHECK(leg_parts[0].find("text")->as_string() == "legacy");
        CHECK(leg_parts[1].find("image_url")->find("url")->as_string() == "data:image/png;base64,AQID");
        CHECK(leg_parts[2].find("image_url")->find("url")->as_string() == "data:image/jpeg;base64,BAUG");
    }
}
