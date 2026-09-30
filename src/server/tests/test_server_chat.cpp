// POST /v1/chat/completions end to end on the MOCK backend: non-streaming and
// streaming responses, the default alias, request errors, correlation headers
// in telemetry envelopes, and cancellation on client disconnect.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"

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

}  // namespace

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
    CHECK(code_of(chat_body(R"(,"tools":[{"type":"function"}])")) ==
          std::make_pair(400, std::string("unsupported_parameter")));
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
