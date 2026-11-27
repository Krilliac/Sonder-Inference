// POST /v1/chat/completions end to end on the MOCK backend: non-streaming and
// streaming responses, the default alias, request errors, correlation headers
// in telemetry envelopes, and cancellation on client disconnect.
#include <doctest/doctest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "src/openai.hpp"
#include "src/request_path.hpp"
#if defined(SONDER_HAS_OLLAMA_CLOUD_BACKEND)
#include "sonder/inference/backends/ollama_cloud.hpp"
#endif

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

// Splits an SSE body into the payloads of its "data:" lines.
std::vector<std::string> sse_data(const std::string& body) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while ((pos = body.find("data: ", pos)) != std::string::npos) {
        const auto end = body.find("\n\n", pos);
        out.push_back(body.substr(pos + 6, end - pos - 6));
        pos = end == std::string::npos ? body.size() : end + 2;
    }
    return out;
}

namespace si = sonder::inference;

// Native-chat backend that replays scripted chunks (text, reasoning) and records
// whether the server asked for separated reasoning.
struct ScriptedBackend final : si::Backend {
    struct Model final : si::BackendModel {
        si::ModelDescriptor description;
        std::vector<std::pair<std::string, std::string>> chunks;
        std::atomic<bool> separate_reasoning{false};
        std::atomic<int> local_only{-1};  // last ChatRequest::local_only seen (-1 = none yet)
        Model() {
            description.name = "scripted:test";
            description.backend = "scripted-test";
            description.context_length = 2048;
        }
        const si::ModelDescriptor& descriptor() const override { return description; }
        bool has_native_chat() const override { return true; }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                              const si::TokenCallback&) override {
            return si::Status(si::ErrorCode::internal, "this test requires native chat");
        }
        si::Result<si::GenerateStats> chat(const si::ChatRequest& request, const si::CancellationToken&,
                                          const si::TokenCallback& callback) override {
            separate_reasoning.store(request.separate_reasoning);
            local_only.store(request.local_only ? 1 : 0);
            std::uint64_t index = 0;
            for (const auto& [text, reasoning] : chunks) {
                if (callback && !callback(si::TokenChunk{text, index++, reasoning})) break;
            }
            si::GenerateStats stats;
            stats.chunks = index;
            stats.prompt_tokens = 5;
            stats.completion_tokens = 7;
            stats.token_counts_from_backend = true;
            stats.stop_reason = si::StopReason::end_of_sequence;
            return stats;
        }
    };
    std::shared_ptr<Model> model = std::make_shared<Model>();
    std::string name() const override { return "scripted-test"; }
    std::string description() const override { return "scripted reasoning test backend"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming);
    }
    si::Result<std::string> probe() override { return std::string("test"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

srv::ServerOptions scripted_options(const std::shared_ptr<ScriptedBackend>& backend) {
    auto options = Fixture::defaults();
    options.models = {"scripted:test"};
    options.backend.backend.clear();
    options.backend_instance = backend;
    return options;
}

}  // namespace

// L1 pure request regressions (also runnable without a listening server).
TEST_CASE("reasoning budget: invalid body values name the field") {
    for (const auto* value : {"-2", "1.5", "true", "\"512\"", "null", "9223372036854775808"}) {
        const auto parsed = det::parse_chat_request(chat_body(std::string(",\"reasoning_budget_tokens\":") + value));
        REQUIRE(std::holds_alternative<det::ApiError>(parsed));
        const auto& error = std::get<det::ApiError>(parsed);
        CHECK(error.status == 400);
        CHECK(error.param == std::optional<std::string>("reasoning_budget_tokens"));
    }
}

TEST_CASE("reasoning budget: message byte limit and type are enforced") {
    std::string oversized_utf8;
    for (int i = 0; i < 257; ++i) oversized_utf8 += "\xC3\xA9";
    for (const auto& value : {json::Value(std::string(513, 'x')), json::Value(3), json::Value(nullptr),
                              json::Value(oversized_utf8)}) {
        auto body = json::parse(chat_body());
        REQUIRE(body.ok());
        body->object().set("reasoning_budget_message", value);
        const auto parsed = det::parse_chat_request(body->dump());
        REQUIRE(std::holds_alternative<det::ApiError>(parsed));
        const auto& error = std::get<det::ApiError>(parsed);
        CHECK(error.status == 400);
        CHECK(error.param == std::optional<std::string>("reasoning_budget_message"));
    }
}

TEST_CASE("reasoning history: assistant reasoning content must be a string") {
    const auto parsed = det::parse_chat_request(
        R"({"messages":[{"role":"assistant","content":"answer","reasoning_content":42},{"role":"user","content":"next"}]})");
    REQUIRE(std::holds_alternative<det::ApiError>(parsed));
    CHECK(std::get<det::ApiError>(parsed).status == 400);
    CHECK(std::get<det::ApiError>(parsed).param == std::optional<std::string>("messages[0].reasoning_content"));
}

TEST_CASE("pin mode: default flag preserves explicit false without warnings") {
    srv::ServerOptions options;
    options.pin_enable_thinking = true;
    REQUIRE(det::parse_request_path_flags([](const std::string& name) -> std::optional<std::string> {
        return name == "pin-mode" ? std::optional<std::string>("default") : std::nullopt;
    }, options).ok());
    si::ThinkingOptions thinking;
    thinking.enable_thinking = false;
    CHECK(det::apply_thinking_pins(options, thinking).empty());
    CHECK(thinking.enable_thinking == std::optional<bool>(false));
}
// End L1 pure request regressions.

TEST_CASE("chat: mock health and model features are explicitly empty") {
    auto options = Fixture::defaults();
    options.pin_mode = srv::PinMode::default_value;
    options.pin_enable_thinking = true;
    Fixture f(options);
    for (const auto* path : {"/v1/sonder/health", "/v1/models"}) {
        const auto reply = get(f.port, path);
        REQUIRE(reply.status == 200);
        const auto document = reply.json();
        const auto* meta = document.find("sonder");
        REQUIRE(meta != nullptr);
        REQUIRE(meta->find("features") != nullptr);
        CHECK(meta->find("features")->is_array());
        CHECK(meta->find("features")->as_array().empty());
        REQUIRE(meta->find("pins") != nullptr);
        CHECK(meta->find("pins")->find("mode")->as_string() == "default");
    }
}

TEST_CASE("chat: mock accepts budget controls with warnings and no timing fabrication") {
    Fixture f;
    const auto reply = post(f.port, "/v1/chat/completions",
        chat_body(R"(,"max_tokens":1,"reasoning_budget_tokens":512,"reasoning_budget_message":"short")"));
    REQUIRE(reply.status == 200);
    const auto document = reply.json();
    const auto* warnings = document.find("sonder")->find("warnings");
    REQUIRE(warnings != nullptr);
    REQUIRE(warnings->as_array().size() == 2);
    CHECK(warnings->as_array()[0].as_string().find("reasoning_budget_tokens") != std::string::npos);
    CHECK(warnings->as_array()[1].as_string().find("reasoning_budget_message") != std::string::npos);
    CHECK(document.find("usage")->find("sonder") == nullptr);
    CHECK(document.find("usage")->find("prompt_tokens_details") == nullptr);
    const auto completed = f.of_type("request.completed");
    REQUIRE(completed.size() == 1);
    const auto* attributes = completed[0].find("attributes");
    CHECK(attributes->find("reasoning_budget") == nullptr);
    CHECK(attributes->find("backend_prompt_n") == nullptr);
}

TEST_CASE("chat: malformed reasoning budget headers fail before execution") {
    Fixture f;
    for (const auto* value : {"-2", "1.5", "none", "9223372036854775808"}) {
        const auto reply = post(f.port, "/v1/chat/completions", chat_body(), {{"X-Sonder-Reasoning-Budget", value}});
        REQUIRE(reply.status == 400);
        CHECK(reply.json().find("error")->find("param")->as_string() == "X-Sonder-Reasoning-Budget");
    }
    CHECK(f.of_type("request.started").empty());
}

TEST_CASE("chat: non-streaming completion with usage, measured timings and Sonder metadata") {
    Fixture f;
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":8,"seed":3)"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(r.header("x-sonder-inference-api") == "1");
    const json::Value doc = r.json();
    CHECK(doc.find("id")->as_string().rfind("chatcmpl-req-", 0) == 0);
    CHECK(doc.find("object")->as_string() == "chat.completion");
    CHECK(doc.find("created")->as_int() > 1700000000);
    // The resolved model id, never the alias.
    CHECK(doc.find("model")->as_string() == "mock:tiny");
    const auto& choices = doc.find("choices")->as_array();
    REQUIRE(choices.size() == 1);
    CHECK(choices[0].find("index")->as_int() == 0);
    CHECK(choices[0].find("message")->find("role")->as_string() == "assistant");
    const std::string content = choices[0].find("message")->find("content")->as_string();
    CHECK_FALSE(content.empty());
    CHECK(choices[0].find("finish_reason")->as_string() == "length");
    const json::Value* usage = doc.find("usage");
    REQUIRE(usage != nullptr);
    CHECK(usage->find("completion_tokens")->as_int() == 8);
    CHECK(usage->find("prompt_tokens")->as_int() > 0);
    CHECK(usage->find("total_tokens")->as_int() ==
          usage->find("prompt_tokens")->as_int() + usage->find("completion_tokens")->as_int());
    const json::Value* t = doc.find("timings");
    REQUIRE(t != nullptr);
    CHECK(t->find("prompt_n")->as_int() == usage->find("prompt_tokens")->as_int());
    CHECK(t->find("predicted_n")->as_int() == 8);
    CHECK(t->find("total_ms")->as_double() >= 0.0);
    CHECK(t->find("ttft_ms")->as_double() >= 0.0);
    // The mock reports decode time but no prompt-eval time: only measured values appear.
    CHECK(t->find("prompt_ms") == nullptr);
    CHECK(t->find("prompt_per_second") == nullptr);
    CHECK(t->find("predicted_ms") != nullptr);
    const json::Value* meta = doc.find("sonder");
    REQUIRE(meta != nullptr);
    CHECK(meta->find("api_version")->as_int() == 1);
    CHECK(doc.find("id")->as_string() == "chatcmpl-" + meta->find("request_id")->as_string());
    CHECK(r.header("x-sonder-request-id") == meta->find("request_id")->as_string());
    CHECK(meta->find("session_id")->as_string().rfind("sess-", 0) == 0);
    CHECK(meta->find("backend")->as_string() == "mock");
    CHECK(meta->find("synthetic")->as_bool());
    CHECK(meta->find("token_counts_from_backend")->as_bool());

    // Deterministic mock: same request, same content.
    const Reply again = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":8,"seed":3)"));
    CHECK(again.json().find("choices")->as_array()[0].find("message")->find("content")->as_string() == content);
    // An explicit model id works like the alias.
    const Reply named = post(f.port, "/v1/chat/completions",
                             R"({"model":"mock:tiny","messages":[{"role":"user","content":"hello sonder"}],"max_tokens":8,"seed":3})");
    CHECK(named.status == 200);
}

TEST_CASE("chat: streaming sends chunks, a final chunk with usage, then [DONE]") {
    Fixture f;
    const Reply r = post(f.port, "/v1/chat/completions",
                         chat_body(R"(,"max_tokens":6,"stream":true,"stream_options":{"include_usage":true})"));
    REQUIRE(r.status == 200);
    CHECK(r.header("content-type") == "text/event-stream; charset=utf-8");
    CHECK_FALSE(r.has_header("content-length"));
    const auto events = sse_data(r.body);
    REQUIRE(events.size() >= 4);
    CHECK(events.back() == "[DONE]");
    std::string text;
    std::string id;
    for (std::size_t i = 0; i + 1 < events.size(); ++i) {
        auto chunk = json::parse(events[i]);
        REQUIRE(chunk.ok());
        const json::Value& c = chunk.value();
        CHECK(c.find("object")->as_string() == "chat.completion.chunk");
        CHECK(c.find("model")->as_string() == "mock:tiny");
        if (id.empty()) id = c.find("id")->as_string();
        CHECK(c.find("id")->as_string() == id);
        const json::Value& choice = c.find("choices")->as_array()[0];
        if (i == 0) {
            CHECK(choice.find("delta")->find("role")->as_string() == "assistant");
        }
        if (const json::Value* content = choice.find("delta")->find("content")) {
            text += content->as_string();
        }
        if (i + 2 == events.size()) {
            CHECK(choice.find("finish_reason")->as_string() == "length");
            CHECK(c.find("usage")->find("completion_tokens")->as_int() == 6);
            CHECK(c.find("timings")->find("predicted_n")->as_int() == 6);
            CHECK(c.find("sonder")->find("synthetic")->as_bool());
        } else {
            CHECK(choice.find("finish_reason")->is_null());
            CHECK(c.find("usage") == nullptr);
        }
    }
    // Same text as the non-streaming answer.
    const Reply plain = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":6)"));
    CHECK(plain.json().find("choices")->as_array()[0].find("message")->find("content")->as_string() == text);

    // Without include_usage the final chunk carries no usage.
    const Reply no_usage = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":3,"stream":true)"));
    const auto ev2 = sse_data(no_usage.body);
    REQUIRE(ev2.size() >= 2);
    auto last = json::parse(ev2[ev2.size() - 2]);
    REQUIRE(last.ok());
    CHECK(last.value().find("usage") == nullptr);
}

TEST_CASE("chat: request errors use the OpenAI error shape") {
    Fixture f;
    const auto code_of = [&](const std::string& body, const std::vector<std::pair<std::string, std::string>>& h = {}) {
        const Reply r = post(f.port, "/v1/chat/completions", body, h);
        const json::Value doc = r.json();
        REQUIRE_MESSAGE(doc.find("error") != nullptr, r.body);
        return std::make_pair(r.status, doc.find("error")->find("code")->as_string());
    };
    CHECK(code_of("{not json") == std::make_pair(400, std::string("invalid_json")));
    CHECK(code_of(R"({"messages":"hi"})") == std::make_pair(400, std::string("invalid_messages")));
    CHECK(code_of(chat_body(R"(,"functions":[{"name":"f"}])")) ==
          std::make_pair(400, std::string("unsupported_parameter")));
    CHECK(code_of(chat_body(R"(,"tools":[{"type":"function"}])")) ==
          std::make_pair(400, std::string("invalid_request")));
    CHECK(code_of(chat_body(R"(,"n":3)")) == std::make_pair(400, std::string("unsupported_parameter")));
    CHECK(code_of(chat_body(R"(,"top_p":0)")) == std::make_pair(400, std::string("invalid_sampling")));
    CHECK(code_of(R"({"model":"gpt-4","messages":[{"role":"user","content":"x"}]})") ==
          std::make_pair(404, std::string("model_not_found")));
    CHECK(code_of(chat_body(), {{"X-Sonder-Run-Id", "bad value!"}}) ==
          std::make_pair(400, std::string("invalid_correlation_header")));
    CHECK(code_of(chat_body(), {{"X-Sonder-Workload", "vip"}}) ==
          std::make_pair(400, std::string("invalid_correlation_header")));
    CHECK(code_of(chat_body(), {{"X-Sonder-Priority", "99"}}) ==
          std::make_pair(400, std::string("invalid_correlation_header")));
}

TEST_CASE("chat: a backend failure is 503 backend_unavailable (request.failed)") {
    auto o = Fixture::defaults();
    o.backend.mock_fail_after_tokens = 2;
    Fixture f(o);
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":8)"));
    CHECK(r.status == 503);
    const json::Value doc = r.json();
    CHECK(doc.find("error")->find("type")->as_string() == "service_unavailable");
    CHECK(doc.find("error")->find("code")->as_string() == "backend_unavailable");
    // Streaming: chunks were already sent, so the error arrives as an SSE
    // data event and the stream ends without [DONE].
    const Reply s = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":8,"stream":true)"));
    CHECK(s.status == 200);
    const auto events = sse_data(s.body);
    REQUIRE_FALSE(events.empty());
    CHECK(events.back().find("backend_unavailable") != std::string::npos);
    CHECK(s.body.find("[DONE]") == std::string::npos);
    CHECK(eventually([&] { return f.of_type("request.failed").size() == 2; }));
}

TEST_CASE("chat: correlation headers land in telemetry envelopes") {
    Fixture f;
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)"),
                         {{"X-Sonder-Run-Id", "turn-R1"},
                          {"X-Sonder-Parent-Request-Id", "turn-R1"},
                          {"X-Sonder-Agent-Id", "agent-7"},
                          {"X-Sonder-Task-Id", "task-9"},
                          {"X-Sonder-Workload", "owner_orchestrator"},
                          {"X-Sonder-Priority", "2"}});
    REQUIRE(r.status == 200);
    const std::string request_id = r.json().find("sonder")->find("request_id")->as_string();
    std::vector<std::string> lifecycle;
    bool token_event = false;
    for (const auto& e : f.envelopes()) {
        const json::Value* rid = e.find("request_id");
        if (rid == nullptr || !rid->is_string() || rid->as_string() != request_id) continue;
        const std::string type = e.find("event_type")->as_string();
        CHECK(e.find("run_id")->as_string() == "turn-R1");
        CHECK(e.find("agent_id")->as_string() == "agent-7");
        CHECK(e.find("task_id")->as_string() == "task-9");
        CHECK(e.find("producer")->find("role")->as_string() == "inference");
        CHECK(e.find("producer")->find("synthetic")->as_bool());
        if (type.rfind("request.", 0) == 0) {
            lifecycle.push_back(type);
            CHECK(e.find("attributes")->find("parent_request_id")->as_string() == "turn-R1");
        }
        if (type == "request.queued") {
            CHECK(e.find("attributes")->find("kind")->as_string() == "chat");
            CHECK(e.find("attributes")->find("workload")->as_string() == "owner_orchestrator");
            CHECK(e.find("attributes")->find("priority")->as_int() == 2);
        }
        if (type == "inference.token.generated") token_event = true;
    }
    CHECK(lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});
    CHECK(token_event);

    // Without headers: run_id stays the engine id and the workload is interactive_user.
    const Reply plain = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)"));
    const std::string plain_id = plain.json().find("sonder")->find("request_id")->as_string();
    bool seen = false;
    for (const auto& e : f.of_type("request.queued")) {
        if (e.find("request_id")->as_string() != plain_id) continue;
        seen = true;
        CHECK(e.find("run_id")->as_string().rfind("engine-", 0) == 0);
        CHECK(e.find("attributes")->find("workload")->as_string() == "interactive_user");
        CHECK(e.find("attributes")->find("parent_request_id") == nullptr);
    }
    CHECK(seen);
}

TEST_CASE("chat: priority header forms preserve numeric telemetry and override body class") {
    Fixture f;
    const auto queued_for = [&](const std::string& body,
                                const std::vector<std::pair<std::string, std::string>>& headers) {
        const Reply reply = post(f.port, "/v1/chat/completions", body, headers);
        if (reply.status != 200) return std::optional<json::Value>{};
        const std::string request_id = reply.json().find("sonder")->find("request_id")->as_string();
        for (const auto& event : f.of_type("request.queued")) {
            if (event.find("request_id")->as_string() != request_id) continue;
            return std::optional<json::Value>(*event.find("attributes"));
        }
        return std::optional<json::Value>{};
    };

    // Legacy numeric headers retain the integer scheduler/telemetry value,
    // while the header's interactive admission class wins over the body.
    for (const auto* value : {"-16", "0", "+5", "+16"}) {
        const auto numeric = queued_for(chat_body(R"(,"priority":"background","max_tokens":2)"),
                                        {{"X-Sonder-Priority", value}});
        REQUIRE(numeric.has_value());
        CHECK(numeric->find("priority")->as_int() == std::stoi(value));
        CHECK(numeric->find("priority_class")->as_string() == "interactive");
        CHECK(numeric->find("admission")->find("priority")->as_string() == "interactive");
    }

    // Named headers select admission directly and do not invent a numeric
    // scheduler priority; the body class is overridden by the header.
    for (const auto* value : {"interactive", "subagent", "background"}) {
        const auto named = queued_for(chat_body(R"(,"priority":"background","max_tokens":2)"),
                                      {{"X-Sonder-Priority", value}});
        REQUIRE(named.has_value());
        CHECK(named->find("priority")->as_int() == 0);
        CHECK(named->find("priority_class")->as_string() == value);
        CHECK(named->find("admission")->find("priority")->as_string() == value);
    }

    const auto body_only = queued_for(chat_body(R"(,"priority":"background","max_tokens":2)"), {});
    REQUIRE(body_only.has_value());
    CHECK(body_only->find("priority_class")->as_string() == "background");
    const auto no_hints = queued_for(chat_body(R"(,"max_tokens":2)"), {});
    REQUIRE(no_hints.has_value());
    CHECK(no_hints->find("priority_class")->as_string() == "interactive");

    CHECK(post(f.port, "/v1/chat/completions", chat_body(R"(,"priority":"unknown")"),
               {{"X-Sonder-Priority", "interactive"}})
              .status == 400);
    CHECK(post(f.port, "/v1/chat/completions", chat_body(R"(,"priority":"background")"),
               {{"X-Sonder-Priority", "unknown"}})
              .status == 400);
}

#if defined(SONDER_HAS_SCHEDULER) && defined(SONDER_HAS_KV_CACHE)
TEST_CASE("chat: numeric header retains its workload-based scheduler rank with admission enabled or disabled") {
    for (int policy = 0; policy < 4; ++policy) {
        auto options = Fixture::defaults();
        if (policy == 1) options.priority_admission = srv::PriorityAdmissionPolicy::on;
        if (policy == 2) options.max_concurrent_background = 1;  // auto enabled by a cap
        if (policy == 3) {
            options.priority_admission = srv::PriorityAdmissionPolicy::off;
            options.max_concurrent_background = 1;
        }
        Fixture fixture(std::move(options));
        for (const auto* value : {"-16", "0", "+16"}) {
            const Reply reply = post(fixture.port, "/v1/chat/completions",
                                     chat_body(R"(,"priority":"background","max_tokens":2)"),
                                     {{"X-Sonder-Priority", value}, {"X-Sonder-Workload", "maintenance"}});
            REQUIRE(reply.status == 200);
            const std::string id = reply.json().find("sonder")->find("request_id")->as_string();
            bool found = false;
            for (const auto& event : fixture.of_type("scheduler.enqueued")) {
                if (event.find("request_id")->as_string() != id) continue;
                found = true;
                CHECK(event.find("attributes")->find("priority_rank")->as_int() ==
                      static_cast<int>(si::WorkloadClass::maintenance) - std::stoi(value));
            }
            CHECK(found);
        }
    }
}
#endif

TEST_CASE("chat: a client disconnect cancels the request (request.cancelled)") {
    auto o = Fixture::defaults();
    o.backend.mock_token_delay = std::chrono::milliseconds(40);
    Fixture f(o);
    for (const bool stream : {false, true}) {
        CAPTURE(stream);
        const std::string body = chat_body(std::string(R"(,"max_tokens":200)") + (stream ? R"(,"stream":true)" : ""));
        Conn c(f.port);
        c.send(build_request("POST", "/v1/chat/completions", f.port, {{"X-Sonder-Run-Id", stream ? "disc-s" : "disc-n"}},
                             body));
        // Wait until generation is under way, then hang up.
        REQUIRE(eventually([&] {
            for (const auto& e : f.of_type("inference.token.generated")) {
                if (e.find("run_id")->as_string() == (stream ? "disc-s" : "disc-n")) return true;
            }
            return false;
        }));
        c.close();
        REQUIRE(eventually([&] {
            for (const auto& e : f.of_type("request.cancelled")) {
                if (e.find("run_id")->as_string() == (stream ? "disc-s" : "disc-n")) return true;
            }
            return false;
        }));
    }
    // The server keeps serving.
    CHECK(post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)")).status == 200);
}

TEST_CASE("chat: X-Sonder-Local-Only reaches the backend request") {
    auto backend = std::make_shared<ScriptedBackend>();
    backend->model->chunks = {{"ok", ""}};
    Fixture f(scripted_options(backend));
    Reply r = post(f.port, "/v1/chat/completions", chat_body(), {{"X-Sonder-Local-Only", "1"}});
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(backend->model->local_only.load() == 1);
    r = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(backend->model->local_only.load() == 0);
    r = post(f.port, "/v1/responses", R"({"model":"default","input":"hi"})", {{"X-Sonder-Local-Only", "true"}});
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(backend->model->local_only.load() == 1);
    r = post(f.port, "/v1/chat/completions", chat_body(), {{"X-Sonder-Local-Only", "maybe"}});
    CHECK(r.status == 400);
    CHECK(r.body.find("X-Sonder-Local-Only") != std::string::npos);
}

#if defined(SONDER_HAS_OLLAMA_CLOUD_BACKEND)
TEST_CASE("chat: ollama_cloud refuses a local-only request over HTTP without calling the cloud") {
    // Scripted stand-in for https://ollama.com; counts chat and generate calls.
    struct CloudStub final : si::OllamaCloudTransport {
        std::atomic<int> chats{0};
        std::atomic<int> generates{0};
        si::Status send(const si::OllamaCloudHttpRequest& request, const std::function<void(int)>& on_status,
                        const std::function<bool(std::string_view)>& on_body,
                        const si::CancellationToken&) override {
            on_status(200);
            if (request.target == "/api/tags") {
                on_body(R"({"models":[{"name":"gpt-oss:120b"}]})");
            } else if (request.target == "/api/generate") {
                ++generates;
                on_body(R"({"response":"from the cloud","done":false})"
                        "\n"
                        R"({"response":"","done":true,"done_reason":"stop"})"
                        "\n");
            } else {
                ++chats;
                on_body(R"({"message":{"content":"from the cloud"},"done":false})"
                        "\n"
                        R"({"message":{"content":""},"done":true,"done_reason":"stop"})"
                        "\n");
            }
            return si::Status::success();
        }
    };
    auto stub = std::make_shared<CloudStub>();
    si::OllamaCloudOptions co;
    co.api_key_env = "SONDER_TEST_CLOUD_KEY";
    co.env_lookup = [](const char*) -> std::optional<std::string> { return std::string("sk-FAKE-http-test"); };
    co.models = {"gpt-oss:120b"};
    co.transport = stub;
    auto backend = si::make_ollama_cloud_backend(co);
    REQUIRE(backend.ok());
    auto options = Fixture::defaults();
    options.models = {"gpt-oss:120b"};
    options.backend.backend.clear();
    options.backend_instance = backend.value();
    Fixture f(options);
    const Reply refused = post(f.port, "/v1/chat/completions", chat_body(), {{"X-Sonder-Local-Only", "1"}});
    CHECK(refused.status == 400);
    CHECK(refused.body.find("local-only") != std::string::npos);
    CHECK(refused.body.find("sk-FAKE") == std::string::npos);
    CHECK(stub->chats.load() == 0);
    const Reply ok = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(ok.status == 200, ok.body);
    CHECK(ok.body.find("from the cloud") != std::string::npos);
    CHECK(stub->chats.load() == 1);

    // /v1/completions forwards the flag to GenerateRequest::local_only too.
    const std::string completion = R"({"model":"gpt-oss:120b","prompt":"hi","max_tokens":4})";
    const Reply refused_completion = post(f.port, "/v1/completions", completion, {{"X-Sonder-Local-Only", "true"}});
    CHECK(refused_completion.status == 400);
    CHECK(refused_completion.body.find("local-only") != std::string::npos);
    CHECK(refused_completion.body.find("sk-FAKE") == std::string::npos);
    CHECK(stub->generates.load() == 0);
    const Reply ok_completion = post(f.port, "/v1/completions", completion);
    REQUIRE_MESSAGE(ok_completion.status == 200, ok_completion.body);
    CHECK(ok_completion.body.find("from the cloud") != std::string::npos);
    CHECK(stub->generates.load() == 1);
    CHECK(stub->chats.load() == 1);
}
#endif

TEST_CASE("chat: non-streaming message carries reasoning_content only when reasoning was produced") {
    auto backend = std::make_shared<ScriptedBackend>();
    backend->model->chunks = {{"", "step one, "}, {"", "step two"}, {"No, 91 ", ""}, {"is not prime.", ""}};
    Fixture f(scripted_options(backend));
    const Reply r = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(backend->model->separate_reasoning.load());
    const json::Value doc = r.json();
    const json::Value& message = *doc.find("choices")->as_array()[0].find("message");
    CHECK(message.find("content")->as_string() == "No, 91 is not prime.");
    REQUIRE(message.find("reasoning_content") != nullptr);
    CHECK(message.find("reasoning_content")->as_string() == "step one, step two");
    CHECK(doc.find("usage")->find("prompt_tokens")->as_int() == 5);
    CHECK(doc.find("usage")->find("completion_tokens")->as_int() == 7);
    CHECK(doc.find("timings")->find("predicted_n")->as_int() == 7);

    backend->model->chunks = {{"plain ", ""}, {"answer", ""}};
    const Reply none = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(none.status == 200, none.body);
    const json::Value none_doc = none.json();
    const json::Value& none_message = *none_doc.find("choices")->as_array()[0].find("message");
    CHECK(none_message.find("content")->as_string() == "plain answer");
    CHECK(none_message.find("reasoning_content") == nullptr);
}

TEST_CASE("chat: streaming deltas carry reasoning_content next to content") {
    auto backend = std::make_shared<ScriptedBackend>();
    backend->model->chunks = {{"", "think "}, {"", "more"}, {"No, ", ""}, {"91 is not prime.", ""}};
    Fixture f(scripted_options(backend));
    const Reply r = post(f.port, "/v1/chat/completions",
                         chat_body(R"(,"stream":true,"stream_options":{"include_usage":true})"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto events = sse_data(r.body);
    REQUIRE(events.size() >= 4);
    CHECK(events.back() == "[DONE]");
    std::string text;
    std::string reasoning;
    for (std::size_t i = 0; i + 1 < events.size(); ++i) {
        auto chunk = json::parse(events[i]);
        REQUIRE(chunk.ok());
        const json::Value* delta = chunk.value().find("choices")->as_array()[0].find("delta");
        if (const json::Value* c = delta->find("content")) text += c->as_string();
        if (const json::Value* rc = delta->find("reasoning_content")) {
            CHECK(delta->find("content") == nullptr);
            reasoning += rc->as_string();
        }
        if (i + 2 == events.size()) {
            CHECK(chunk.value().find("usage")->find("completion_tokens")->as_int() == 7);
            CHECK(chunk.value().find("timings")->find("predicted_n")->as_int() == 7);
        }
    }
    CHECK(text == "No, 91 is not prime.");
    CHECK(reasoning == "think more");

    backend->model->chunks = {{"plain ", ""}, {"answer", ""}};
    const Reply none = post(f.port, "/v1/chat/completions", chat_body(R"(,"stream":true)"));
    REQUIRE_MESSAGE(none.status == 200, none.body);
    CHECK(none.body.find("reasoning_content") == std::string::npos);
}

TEST_CASE("chat: a chunk carrying both content and reasoning keeps both fields") {
    auto backend = std::make_shared<ScriptedBackend>();
    backend->model->chunks = {{"answer", "because"}};
    Fixture f(scripted_options(backend));

    const Reply plain = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(plain.status == 200, plain.body);
    const json::Value doc = plain.json();
    const json::Value& message = *doc.find("choices")->as_array()[0].find("message");
    REQUIRE(message.find("content") != nullptr);
    REQUIRE(message.find("reasoning_content") != nullptr);
    CHECK(message.find("content")->as_string() == "answer");
    CHECK(message.find("reasoning_content")->as_string() == "because");

    const Reply streamed = post(f.port, "/v1/chat/completions", chat_body(R"(,"stream":true)"));
    REQUIRE_MESSAGE(streamed.status == 200, streamed.body);
    const auto events = sse_data(streamed.body);
    bool seen = false;
    for (std::size_t i = 0; i + 1 < events.size(); ++i) {
        auto chunk = json::parse(events[i]);
        REQUIRE(chunk.ok());
        const json::Value* delta = chunk.value().find("choices")->as_array()[0].find("delta");
        if (delta->find("reasoning_content") == nullptr) continue;
        seen = true;
        REQUIRE(delta->find("content") != nullptr);
        CHECK(delta->find("content")->as_string() == "answer");
        CHECK(delta->find("reasoning_content")->as_string() == "because");
    }
    CHECK(seen);
}

TEST_CASE("chat: streaming deltas keep backend order, reasoning chunks before content chunks") {
    auto backend = std::make_shared<ScriptedBackend>();
    backend->model->chunks = {{"", "r1"}, {"", "r2"}, {"c1", ""}, {"c2", ""}};
    Fixture f(scripted_options(backend));
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"stream":true)"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto events = sse_data(r.body);
    std::vector<std::string> order;
    for (std::size_t i = 0; i + 1 < events.size(); ++i) {
        auto chunk = json::parse(events[i]);
        REQUIRE(chunk.ok());
        const json::Value* delta = chunk.value().find("choices")->as_array()[0].find("delta");
        if (const json::Value* rc = delta->find("reasoning_content")) order.push_back("r:" + rc->as_string());
        if (const json::Value* c = delta->find("content"); c && !c->as_string().empty()) {
            order.push_back("c:" + c->as_string());
        }
    }
    const std::vector<std::string> expected{"r:r1", "r:r2", "c:c1", "c:c2"};
    CHECK(order == expected);
}

#if defined(SONDER_HAS_OLLAMA_CLOUD_BACKEND)
namespace {

// Scripted stand-in for https://ollama.com that counts every call it gets.
struct CountingCloudStub final : si::OllamaCloudTransport {
    std::atomic<int> calls{0};
    std::atomic<int> tags_status{200};
    std::atomic<int> chat_status{200};
    std::string error_body;  // set before the server starts; read-only afterwards
    si::Status send(const si::OllamaCloudHttpRequest& request, const std::function<void(int)>& on_status,
                    const std::function<bool(std::string_view)>& on_body, const si::CancellationToken&) override {
        ++calls;
        if (request.target == "/api/tags") {
            on_status(tags_status.load());
            if (tags_status.load() != 200) {
                on_body(error_body);
            } else {
                on_body(R"({"models":[{"name":"gpt-oss:120b"}]})");
            }
        } else if (chat_status.load() != 200) {
            on_status(chat_status.load());
            on_body(error_body);
        } else {
            on_status(200);
            on_body(R"({"message":{"content":"from the cloud"},"done":false})"
                    "\n"
                    R"({"message":{"content":""},"done":true,"done_reason":"stop"})"
                    "\n");
        }
        return si::Status::success();
    }
};

// Delegates to a real ollama_cloud backend but answers probe() locally, so
// the server's periodic reachability probe does not add transport calls and
// the stub's count is exactly the model loads and requests.
struct NoProbeBackend final : si::Backend {
    std::shared_ptr<si::Backend> inner;
    explicit NoProbeBackend(std::shared_ptr<si::Backend> b) : inner(std::move(b)) {}
    std::string name() const override { return inner->name(); }
    std::string description() const override { return inner->description(); }
    si::BackendCapabilities capabilities() const override { return inner->capabilities(); }
    si::Result<std::string> probe() override { return std::string("probe answered locally"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& o) override {
        return inner->load_model(o);
    }
    std::optional<si::BackendRuntimeStatus> runtime_status() const override { return inner->runtime_status(); }
};

std::shared_ptr<si::Backend> cloud_backend(const std::shared_ptr<CountingCloudStub>& stub, const std::string& key) {
    si::OllamaCloudOptions co;
    co.api_key_env = "SONDER_TEST_CLOUD_KEY";
    co.env_lookup = [key](const char*) -> std::optional<std::string> { return key; };
    co.models = {"gpt-oss:120b"};
    co.transport = stub;
    auto made = si::make_ollama_cloud_backend(co);
    REQUIRE_MESSAGE(made.ok(), made.status().to_string());
    return std::make_shared<NoProbeBackend>(made.value());
}

std::string b64(std::string_view in) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8) |
                           static_cast<unsigned char>(in[i + 2]);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += kAlphabet[v & 63];
    }
    if (i + 1 == in.size()) {
        const unsigned v = static_cast<unsigned char>(in[i]) << 16;
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == in.size()) {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

}  // namespace

TEST_CASE("chat: a local-only request against ollama_cloud makes no network call, even with lazy models") {
    auto stub = std::make_shared<CountingCloudStub>();
    auto options = Fixture::defaults();
    options.models = {"gpt-oss:120b"};
    options.backend.backend.clear();
    options.backend_instance = cloud_backend(stub, "sk-FAKE-lazy-test");
    options.lazy_models = true;  // nothing resident: a request would load the model (GET /api/tags)
    Fixture f(options);
    const int before = stub->calls.load();

    struct Probe {
        const char* target;
        std::string body;
    };
    const std::vector<Probe> probes = {
        {"/v1/chat/completions", chat_body()},
        {"/v1/completions", R"({"model":"gpt-oss:120b","prompt":"hi","max_tokens":4})"},
        {"/v1/responses", R"({"model":"gpt-oss:120b","input":"hi"})"},
        {"/v1/messages", R"({"model":"gpt-oss:120b","max_tokens":8,"messages":[{"role":"user","content":"hi"}]})"},
        {"/v1/embeddings", R"({"model":"gpt-oss:120b","input":"hi"})"},
    };
    for (const auto& p : probes) {
        CAPTURE(p.target);
        const Reply r = post(f.port, p.target, p.body, {{"X-Sonder-Local-Only", "true"}});
        CHECK(r.status == 400);
        CHECK(r.body.find("local-only") != std::string::npos);
        CHECK(stub->calls.load() == before);
    }

    // Without the header the same request loads the model and reaches the cloud.
    const Reply ok = post(f.port, "/v1/chat/completions", chat_body());
    REQUIRE_MESSAGE(ok.status == 200, ok.body);
    CHECK(ok.body.find("from the cloud") != std::string::npos);
    CHECK(stub->calls.load() > before);
}

TEST_CASE("chat: ollama_cloud error bodies reach neither the client, the log nor telemetry") {
    const std::string key = "sk-FAKE-http-error-body-0123456789";
    const std::string encoded = b64(key);
    const std::string partial = key.substr(0, 16);
    const auto leaks = [&](const std::string& text) {
        return text.find(key) != std::string::npos || text.find(encoded) != std::string::npos ||
               text.find(partial) != std::string::npos || text.find("UPSTREAM-DETAIL") != std::string::npos;
    };
    auto log_mu = std::make_shared<std::mutex>();
    auto log_lines = std::make_shared<std::vector<std::string>>();
    const auto capture_log = [log_mu, log_lines](const std::string& line) {
        std::lock_guard<std::mutex> lock(*log_mu);
        log_lines->push_back(line);
    };
    {
        auto stub = std::make_shared<CountingCloudStub>();
        stub->error_body = R"({"error":")" + encoded + " UPSTREAM-DETAIL " + partial + R"("})";
        auto options = Fixture::defaults();
        options.models = {"gpt-oss:120b"};
        options.backend.backend.clear();
        options.backend_instance = cloud_backend(stub, key);
        options.log = capture_log;
        Fixture f(options);
        for (const int status : {401, 403, 429, 500}) {
            CAPTURE(status);
            stub->chat_status.store(status);
            const Reply r = post(f.port, "/v1/chat/completions", chat_body());
            CHECK(r.status >= 400);
            CHECK_FALSE(leaks(r.body));
            CHECK(r.body.find("HTTP " + std::to_string(status)) != std::string::npos);
            const Reply streamed = post(f.port, "/v1/chat/completions", chat_body(R"(,"stream":true)"));
            CHECK_FALSE(leaks(streamed.body));
        }
        for (const auto& line : f.events->lines()) {
            CHECK_FALSE(leaks(line));
        }
    }
    {
        // Startup: the initial model load fails on GET /api/tags.
        auto stub = std::make_shared<CountingCloudStub>();
        stub->error_body = R"({"error":")" + encoded + " UPSTREAM-DETAIL " + partial + R"("})";
        stub->tags_status.store(401);
        auto options = Fixture::defaults();
        options.models = {"gpt-oss:120b"};
        options.backend.backend.clear();
        options.backend_instance = cloud_backend(stub, key);
        options.log = capture_log;
        srv::Server server(std::move(options));
        const si::Status st = server.start();
        REQUIRE_FALSE(st.ok());
        CHECK_FALSE(leaks(st.message()));
        CHECK(st.message().find("auth failed") != std::string::npos);
        server.stop();
    }
    std::lock_guard<std::mutex> lock(*log_mu);
    CHECK_FALSE(log_lines->empty());
    for (const auto& line : *log_lines) {
        CHECK_FALSE(leaks(line));
    }
}
#endif
