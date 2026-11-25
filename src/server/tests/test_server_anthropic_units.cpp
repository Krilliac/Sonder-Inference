#include <doctest/doctest.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "src/anthropic.hpp"
#include "src/json_schema_gbnf.hpp"
#include "src/reasoning_budget.hpp"
#include "src/request_path.hpp"
#include "src/tool_call_parser.hpp"

using namespace sonder::inference;
using namespace sonder::inference::server;
using namespace sonder::inference::server::detail;

namespace {
json::Object request() {
    return {{"model", "default"},
            {"max_tokens", 32},
            {"messages", json::Array{json::Object{{"role", "user"}, {"content", "hello"}}}}};
}

ChatJob parse_ok(const json::Object& body) {
    const auto parsed = parse_messages_request(json::Value(body).dump());
    REQUIRE(std::holds_alternative<ChatJob>(parsed));
    return std::get<ChatJob>(parsed);
}

void rejected(const json::Object& body, const std::string& field) {
    const auto parsed = parse_messages_request(json::Value(body).dump());
    REQUIRE(std::holds_alternative<ApiError>(parsed));
    const auto& error = std::get<ApiError>(parsed);
    CHECK(error.status == 400);
    CHECK(error.message.find(field) != std::string::npos);
}

std::vector<json::Value> events(const std::string& frames) {
    std::vector<json::Value> out;
    std::size_t pos = 0;
    while (pos < frames.size()) {
        const auto end = frames.find("\n\n", pos);
        REQUIRE(end != std::string::npos);
        const auto newline = frames.find('\n', pos);
        REQUIRE(newline < end);
        CHECK(frames.substr(pos, 7) == "event: ");
        const auto name = frames.substr(pos + 7, newline - pos - 7);
        CHECK(frames.substr(newline + 1, 6) == "data: ");
        auto parsed = json::parse(frames.substr(newline + 7, end - newline - 7));
        REQUIRE(parsed.ok());
        REQUIRE(parsed->find("type") != nullptr);
        CHECK(parsed->find("type")->as_string() == name);
        out.push_back(parsed.value());
        pos = end + 2;
    }
    return out;
}
} // namespace

TEST_CASE("anthropic request: scheduling hints share chat priority and deadline validation") {
    const auto defaults = parse_ok(request());
    CHECK_FALSE(defaults.priority_class);
    CHECK_FALSE(defaults.deadline_ms);
    for (const auto priority : {RequestPriority::interactive, RequestPriority::subagent,
                               RequestPriority::background}) {
        auto body = request();
        body.set("priority", to_string(priority));
        body.set("deadline_ms", 1234);
        const auto job = parse_ok(body);
        CHECK(job.priority_class == priority);
        CHECK(job.deadline_ms == 1234);
    }
    for (const json::Value& value : {json::Value(nullptr), json::Value(0), json::Value("urgent"),
                                    json::Value("Interactive")}) {
        auto body = request();
        body.set("priority", value);
        rejected(body, "priority");
    }
    for (const json::Value& value : {json::Value(nullptr), json::Value(0), json::Value(-1),
                                    json::Value(1.5), json::Value("25"), json::Value(2147483648ULL)}) {
        auto body = request();
        body.set("deadline_ms", value);
        rejected(body, "deadline_ms");
    }
    auto body = request();
    body.set("deadline_ms", 2147483647.0);
    CHECK(parse_ok(body).deadline_ms == 2147483647ULL);
}

TEST_CASE("anthropic request: strings, blocks, roles, defaults and sampler fields") {
    const auto plain = parse_ok(request());
    CHECK(plain.model == "default");
    CHECK(plain.sampling.max_tokens == 32);
    CHECK(plain.sampling.explicit_only);
    CHECK(plain.sampling.explicit_fields == SamplingConfig::kMaxTokens);
    CHECK_FALSE(plain.stream);
    CHECK(plain.thinking.empty());
    CHECK_FALSE(plain.session_key.has_value());

    auto body = request();
    body.set("system", json::Array{json::Object{{"type", "text"}, {"text", "Be "}},
                                   json::Object{{"type", "text"}, {"text", "concise."}}});
    body.set("messages",
             json::Array{
                 json::Object{{"role", "user"}, {"content", "first"}},
                 json::Object{{"role", "assistant"},
                              {"content", json::Array{json::Object{{"type", "text"}, {"text", "reply"}}}}},
                 json::Object{{"role", "user"},
                              {"content", json::Array{json::Object{{"type", "text"}, {"text", "Hello "}},
                                                      json::Object{{"type", "text"}, {"text", "world"}}}}}});
    body.set("temperature", 0.2);
    body.set("top_p", 0.8);
    body.set("top_k", 12);
    body.set("stop_sequences", json::Array{"END", "STOP"});
    body.set("stream", true);
    const auto job = parse_ok(body);
    REQUIRE(job.messages.size() == 4);
    CHECK(job.messages[0].role == "system");
    CHECK(job.messages[0].content == "Be concise.");
    CHECK(job.messages[2].role == "assistant");
    CHECK(job.messages[3].content == "Hello world");
    CHECK(job.sampling.temperature == doctest::Approx(0.2));
    CHECK(job.sampling.top_p == doctest::Approx(0.8));
    CHECK(job.sampling.top_k == 12);
    CHECK((job.sampling.stop == std::vector<std::string>{"END", "STOP"}));
    CHECK(job.sampling.is_explicit(SamplingConfig::kTemperature));
    CHECK(job.sampling.is_explicit(SamplingConfig::kTopP));
    CHECK(job.sampling.is_explicit(SamplingConfig::kTopK));
    CHECK(job.sampling.is_explicit(SamplingConfig::kStop));
    CHECK(job.stream);
    body.set("system", "plain system");
    CHECK(parse_ok(body).messages.front().content == "plain system");
}

TEST_CASE("anthropic request: effort aliases map each level and pins override") {
    for (const std::string level : {"off", "low", "medium", "high"}) {
        for (bool nested : {false, true}) {
            auto body = request();
            if (nested)
                body.set("output_config", json::Object{{"effort", level}});
            else
                body.set("reasoning_effort", level);
            auto job = parse_ok(body);
            REQUIRE(job.thinking.enable_thinking.has_value());
            CHECK(*job.thinking.enable_thinking == (level != "off"));
            if (level == "off")
                CHECK_FALSE(job.thinking.reasoning_effort.has_value());
            else
                CHECK(job.thinking.reasoning_effort == (level == "high" ? "xhigh" : level));
            ServerOptions pins;
            pins.pin_enable_thinking = level == "off";
            pins.pin_reasoning_effort = "pinned";
            const auto warnings = apply_thinking_pins(pins, job.thinking);
            CHECK(job.thinking.enable_thinking == pins.pin_enable_thinking);
            CHECK(job.thinking.reasoning_effort == pins.pin_reasoning_effort);
            CHECK(warnings.size() == (level == "off" ? 1u : 2u));
        }
    }
    auto body = request();
    body.set("thinking", json::Object{{"type", "enabled"}, {"budget_tokens", 8}});
    CHECK(parse_ok(body).thinking.enable_thinking == true);
    body.set("reasoning_effort", "high");
    body.set("output_config", json::Object{{"effort", "high"}});
    CHECK(parse_ok(body).thinking.reasoning_effort == "xhigh");
    body = request();
    body.set("thinking", json::Object{{"type", "disabled"}});
    CHECK(parse_ok(body).thinking.enable_thinking == false);
    auto job = parse_ok(body);
    ServerOptions same_pin;
    same_pin.pin_enable_thinking = false;
    CHECK(apply_thinking_pins(same_pin, job.thinking).empty());
    job = parse_ok(request());
    CHECK(apply_thinking_pins(same_pin, job.thinking).empty());
    CHECK(job.thinking.enable_thinking == false);
}

TEST_CASE("anthropic request: session key uses metadata ahead of correlation headers") {
    Correlation correlation;
    correlation.run_id = "run";
    correlation.agent_id = "agent";
    CHECK(chat_session_key(parse_ok(request()), correlation) == "run=run;agent=agent");
    auto body = request();
    body.set("metadata", json::Object{{"user_id", "user-7"}, {"ignored", true}});
    CHECK(chat_session_key(parse_ok(body), correlation) == "user-7");
    body.set("metadata", json::Object{{"user_id", std::string(256, 'x')}});
    CHECK(parse_ok(body).session_key->size() == 256);
    for (const json::Value& value : {json::Value(""), json::Value(3), json::Value(std::string(257, 'x'))}) {
        body.set("metadata", json::Object{{"user_id", value}});
        rejected(body, "metadata.user_id");
    }
}

TEST_CASE("anthropic request: malformed and unsupported content fails clearly") {
    for (const std::string body : {"{", "[]", "null", "{}"}) {
        CHECK(std::holds_alternative<ApiError>(parse_messages_request(body)));
    }
    auto body = request();
    // Invalid tools / tool_choice fail clearly.
    body = request();
    body.set("tools", json::Object{});
    rejected(body, "tools");
    body = request();
    body.set("tool_choice", json::Array{});
    rejected(body, "tool_choice");
    body = request();
    body.set("tool_choice", json::Object{{"type", "tool"}, {"name", "missing"}});
    rejected(body, "tool_choice");
    // System prompts remain text-only; unsupported user blocks (audio, tool_use) still fail clearly.
    for (const char* type : {"image", "tool_use", "tool_result", "audio"}) {
        const json::Array blocks{json::Object{{"type", type}, {"source", json::Object{}}}};
        body = request();
        body.set("system", blocks);
        rejected(body, type == std::string("image") ? "image" : "content block");
    }
    for (const char* type : {"tool_use", "audio"}) {
        const json::Array blocks{json::Object{{"type", type}, {"source", json::Object{}}}};
        body = request();
        body.set("messages", json::Array{json::Object{{"role", "user"}, {"content", blocks}}});
        rejected(body, "content block");
    }
    // Remote URL image sources are strictly rejected (SSRF protection).
    body = request();
    body.set("messages",
             json::Array{json::Object{
                 {"role", "user"},
                 {"content", json::Array{json::Object{
                                 {"type", "image"},
                                 {"source", json::Object{{"type", "url"}, {"url", "http://127.0.0.1/x.png"}}}}}}}});
    rejected(body, "base64");
    body = request();
    body.set("system", json::Array{json::Object{{"type", "text"}, {"text", 42}}});
    rejected(body, "text");
    body = request();
    body.set("messages", json::Array{json::Object{{"role", "system"}, {"content", "wrong role"}}});
    rejected(body, "role");
    body.set("messages", json::Array{});
    rejected(body, "messages");
}

TEST_CASE("anthropic request: numeric boundaries and conflicting thinking fail safely") {
    for (const auto& field : {"max_tokens", "top_k"}) {
        for (const json::Value& value :
             {json::Value(-1), json::Value(1.5), json::Value("4"), json::Value(std::uint64_t{1} << 63),
              json::Value(std::numeric_limits<std::uint64_t>::max())}) {
            auto body = request();
            body.set(field, value);
            rejected(body, field);
        }
    }
    for (const char* field : {"temperature", "top_p"}) {
        auto body = request();
        body.set(field, 1e300);
        rejected(body, field);
    }
    for (const json::Value& thinking : {json::Value(json::Object{{"type", "enabled"}}),
                                        json::Value(json::Object{{"type", "enabled"}, {"budget_tokens", 0}}),
                                        json::Value(json::Object{{"type", "adaptive"}})}) {
        auto body = request();
        body.set("thinking", thinking);
        rejected(body, thinking.find("type")->as_string() == "enabled" ? "budget_tokens" : "thinking.type");
    }
    auto body = request();
    body.set("thinking", json::Object{{"type", "disabled"}});
    body.set("reasoning_effort", "medium");
    rejected(body, "conflicts");
    body.set("thinking", json::Object{{"type", "enabled"}, {"budget_tokens", 8}});
    body.set("reasoning_effort", "off");
    rejected(body, "conflicts");
    body = request();
    body.set("reasoning_effort", "low");
    body.set("output_config", json::Object{{"effort", "high"}});
    rejected(body, "disagree");
    body.set("output_config", 1);
    rejected(body, "output_config");
    body = request();
    body.set("reasoning_effort", "xhigh");
    rejected(body, "off, low, medium, or high");
    body = request();
    body.set("max_tokens", 0);
    rejected(body, "max_tokens");
}

TEST_CASE("anthropic response: reasoning, stops and optional cache usage") {
    GenerationResult result;
    result.text = "answer";
    result.reasoning = "plan";
    result.stats.prompt_tokens = 10;
    result.stats.completion_tokens = 3;
    for (const auto stop : {StopReason::end_of_sequence, StopReason::max_tokens, StopReason::stop_sequence}) {
        result.stats.stop_reason = stop;
        result.stats.matched_stop = "END";
        const auto response = anthropic_message_response("msg_1", "mock:tiny", result, {"pin override"});
        CHECK(response.find("type")->as_string() == "message");
        CHECK(response.find("role")->as_string() == "assistant");
        CHECK(response.find("id")->as_string() == "msg_1");
        CHECK(response.find("model")->as_string() == "mock:tiny");
        const auto expected = stop == StopReason::max_tokens      ? "max_tokens"
                              : stop == StopReason::stop_sequence ? "stop_sequence"
                                                                  : "end_turn";
        CHECK(response.find("stop_reason")->as_string() == expected);
        if (stop == StopReason::stop_sequence)
            CHECK(response.find("stop_sequence")->as_string() == "END");
        else
            CHECK(response.find("stop_sequence")->is_null());
        const auto& content = response.find("content")->as_array();
        REQUIRE(content.size() == 2);
        CHECK(content[0].find("type")->as_string() == "thinking");
        CHECK(content[0].find("thinking")->as_string() == "plan");
        CHECK(content[1].find("text")->as_string() == "answer");
        CHECK(response.find("sonder")->find("warnings")->as_array().size() == 1);
        CHECK(response.find("usage")->find("input_tokens")->as_uint() == 10);
        CHECK(response.find("usage")->find("output_tokens")->as_uint() == 3);
        CHECK(response.find("usage")->find("cache_read_input_tokens") == nullptr);
    }
    for (std::uint64_t cached : {0u, 6u, 12u}) {
        result.stats.cached_tokens = cached;
        const auto response = anthropic_message_response("msg_1", "mock:tiny", result);
        CHECK(response.find("usage")->find("input_tokens")->as_uint() == (cached > 10 ? 0 : 10 - cached));
        CHECK(response.find("usage")->find("cache_read_input_tokens")->as_uint() == cached);
    }
    result.stats.matched_stop.reset();
    CHECK(anthropic_message_response("msg_1", "mock:tiny", result).find("stop_sequence")->is_null());
}

TEST_CASE("anthropic stream: exact thinking and text sequence, stops and final usage") {
    AnthropicStream stream("msg_1", "mock:tiny");
    GenerationResult result;
    result.stats.prompt_tokens = 10;
    result.stats.completion_tokens = 3;
    result.stats.cached_tokens = 6;
    result.stats.stop_reason = StopReason::stop_sequence;
    result.stats.matched_stop = "END";
    std::string wire = stream.start();
    wire += stream.chunk(TokenChunk{"", 0, "plan"});
    wire += stream.chunk(TokenChunk{"answer", 1, {}});
    wire += stream.finish(result, {"pin override"});
    const auto frames = events(wire);
    const std::vector<std::string> expected{
        "message_start",      "content_block_start", "content_block_delta",
        "content_block_stop", "content_block_start", "content_block_delta",
        "content_block_stop", "message_delta",       "message_stop"};
    REQUIRE(frames.size() == expected.size());
    for (std::size_t i = 0; i < frames.size(); ++i)
        CHECK(frames[i].find("type")->as_string() == expected[i]);
    CHECK(frames[0].find("message")->find("id")->as_string() == "msg_1");
    CHECK(frames[0].find("message")->find("content")->as_array().empty());
    CHECK(frames[1].find("content_block")->find("thinking")->as_string().empty());
    CHECK(frames[2].find("delta")->find("type")->as_string() == "thinking_delta");
    CHECK(frames[2].find("delta")->find("thinking")->as_string() == "plan");
    CHECK(frames[4].find("index")->as_int() == 1);
    CHECK(frames[5].find("delta")->find("type")->as_string() == "text_delta");
    CHECK(frames[5].find("delta")->find("text")->as_string() == "answer");
    CHECK(frames[7].find("delta")->find("stop_sequence")->as_string() == "END");
    CHECK(frames[7].find("usage")->find("input_tokens")->as_uint() == 4);
    CHECK(frames[7].find("usage")->find("output_tokens")->as_uint() == 3);
    CHECK(frames[7].find("sonder")->find("warnings")->as_array().size() == 1);
    CHECK(stream.start().empty());
    CHECK(stream.finish(result).empty());
    CHECK(stream.chunk(TokenChunk{"late", 2}).empty());
}

TEST_CASE("anthropic stream: empty and reasoning-only completions close a text block") {
    for (bool thinking : {false, true}) {
        AnthropicStream stream("msg_empty", "mock:tiny");
        std::string wire = stream.start();
        if (thinking)
            wire += stream.chunk(TokenChunk{"", 0, "plan"});
        wire += stream.finish(GenerationResult{});
        const auto frames = events(wire);
        REQUIRE(frames.size() == (thinking ? 8u : 5u));
        const auto start_index = frames.size() - 4;
        CHECK(frames[start_index].find("type")->as_string() == "content_block_start");
        CHECK(frames[start_index].find("content_block")->find("type")->as_string() == "text");
        CHECK(frames[start_index + 1].find("type")->as_string() == "content_block_stop");
        CHECK(frames.back().find("type")->as_string() == "message_stop");
    }
}

TEST_CASE("anthropic errors: protocol type and HTTP status stay distinct") {
    for (const auto& [status, type] :
         std::array<std::pair<int, const char*>, 7>{{{400, "invalid_request_error"},
                                                     {401, "authentication_error"},
                                                     {403, "permission_error"},
                                                     {404, "not_found_error"},
                                                     {413, "request_too_large"},
                                                     {429, "rate_limit_error"},
                                                     {503, "overloaded_error"}}}) {
        const auto document = anthropic_error_body(make_error(status, "test", "detail"));
        CHECK(document.find("type")->as_string() == "error");
        CHECK(document.find("error")->find("type")->as_string() == type);
        CHECK(document.find("error")->find("message")->as_string() == "detail");
    }
}

TEST_CASE("anthropic stream: deferred tool name and interleaved parallel tool deltas emit valid blocks") {
    AnthropicStream stream("msg_tools", "mock:tiny");
    std::string wire = stream.start();
    // Arguments arrive before name on index 0; index 1 interleaves before index 0 completes.
    wire += stream.tool_call_delta(ToolCallDelta{0, "call_0", "", "{\"a\":"});
    wire += stream.tool_call_delta(ToolCallDelta{1, "call_1", "second_fn", "{\"b\":2}"});
    wire += stream.tool_call_delta(ToolCallDelta{0, "", "first_fn", "1}"});
    wire += stream.finish(GenerationResult{});

    const auto frames = events(wire);
    std::vector<std::string> started_names;
    std::vector<std::string> partial_jsons;
    std::size_t stops = 0;
    bool stop_reason_tool_use = false;
    for (const auto& f : frames) {
        const std::string t = f.find("type")->as_string();
        if (t == "content_block_start") {
            const auto* blk = f.find("content_block");
            REQUIRE(blk != nullptr);
            CHECK(blk->find("type")->as_string() == "tool_use");
            started_names.push_back(blk->find("name")->as_string());
        } else if (t == "content_block_delta") {
            partial_jsons.push_back(f.find("delta")->find("partial_json")->as_string());
        } else if (t == "content_block_stop") {
            ++stops;
        } else if (t == "message_delta") {
            stop_reason_tool_use = f.find("delta")->find("stop_reason")->as_string() == "tool_use";
        }
    }
    REQUIRE(started_names.size() == 2u);
    CHECK(started_names[0] == "first_fn");
    CHECK(started_names[1] == "second_fn");
    REQUIRE(partial_jsons.size() == 2u);
    CHECK(partial_jsons[0] == "{\"a\":1}");
    CHECK(partial_jsons[1] == "{\"b\":2}");
    CHECK(stops == 2u);
    CHECK(stop_reason_tool_use);
}

TEST_CASE("anthropic hardening: thinking budget_tokens propagates and message block bounds enforce") {
    // 1. thinking.budget_tokens propagates onto ChatJob::reasoning_budget_tokens.
    {
        auto body = request();
        body.set("thinking", json::Object{{"type", "enabled"}, {"budget_tokens", 256}});
        const auto job = parse_ok(body);
        CHECK(job.thinking.enable_thinking == std::optional<bool>(true));
        CHECK(job.reasoning_budget_tokens == std::optional<std::int64_t>(256));
    }

    // 2. Empty content arrays are rejected on both user and assistant messages.
    {
        auto body = request();
        body.set("messages", json::Array{json::Object{{"role", "user"}, {"content", json::Array{}}}});
        rejected(body, "content");

        body = request();
        body.set("messages", json::Array{json::Object{{"role", "user"}, {"content", "hi"}},
                                         json::Object{{"role", "assistant"}, {"content", json::Array{}}},
                                         json::Object{{"role", "user"}, {"content", "next"}}});
        rejected(body, "content");
    }

    // 3. Invalid tool_use name, control-char id, non-boolean is_error, and >64 tool_use blocks fail closed.
    {
        auto body = request();
        body.set("messages",
                 json::Array{
                     json::Object{{"role", "user"}, {"content", "hi"}},
                     json::Object{{"role", "assistant"},
                                  {"content", json::Array{json::Object{{"type", "tool_use"},
                                                                       {"id", "toolu_1"},
                                                                       {"name", "bad name!"},
                                                                       {"input", json::Object{}}}}}},
                     json::Object{{"role", "user"},
                                  {"content", json::Array{json::Object{{"type", "tool_result"},
                                                                       {"tool_use_id", "toolu_1"},
                                                                       {"content", "ok"}}}}}});
        rejected(body, "name");

        body = request();
        body.set("messages",
                 json::Array{
                     json::Object{{"role", "user"}, {"content", "hi"}},
                     json::Object{{"role", "assistant"},
                                  {"content", json::Array{json::Object{{"type", "tool_use"},
                                                                       {"id", "toolu_\n1"},
                                                                       {"name", "valid_fn"},
                                                                       {"input", json::Object{}}}}}},
                     json::Object{{"role", "user"},
                                  {"content", json::Array{json::Object{{"type", "tool_result"},
                                                                       {"tool_use_id", "toolu_1"},
                                                                       {"content", "ok"}}}}}});
        rejected(body, ".id");

        body = request();
        body.set("messages",
                 json::Array{
                     json::Object{{"role", "user"}, {"content", "hi"}},
                     json::Object{{"role", "assistant"},
                                  {"content", json::Array{json::Object{{"type", "tool_use"},
                                                                       {"id", "toolu_1"},
                                                                       {"name", "valid_fn"},
                                                                       {"input", json::Object{}}}}}},
                     json::Object{{"role", "user"},
                                  {"content", json::Array{json::Object{{"type", "tool_result"},
                                                                       {"tool_use_id", "toolu_1"},
                                                                       {"is_error", "yes"},
                                                                       {"content", "err"}}}}}});
        rejected(body, "is_error");

        json::Array many_tools;
        for (std::size_t i = 0; i < kMaxToolCallsPerMessage + 1u; ++i) {
            many_tools.push_back(json::Object{{"type", "tool_use"},
                                              {"id", "toolu_" + std::to_string(i)},
                                              {"name", "fn"},
                                              {"input", json::Object{}}});
        }
        body = request();
        body.set("messages", json::Array{json::Object{{"role", "user"}, {"content", "hi"}},
                                         json::Object{{"role", "assistant"}, {"content", std::move(many_tools)}},
                                         json::Object{{"role", "user"}, {"content", "next"}}});
        rejected(body, "tool_use");
    }
}

TEST_CASE("anthropic stream: tool followed by text never duplicates tool_use and malformed empty-name deltas skip") {
    // 1. Live-streamed tool_call_delta followed by trailing text must NOT duplicate
    //    the tool_use block during finish() (regression for missing slot.stopped mark).
    {
        AnthropicStream stream("msg_no_dup", "mock:tiny");
        std::string wire = stream.start();
        wire += stream.tool_call_delta(ToolCallDelta{0, "call_0", "lookup", R"({"q":"hi"})"});
        wire += stream.chunk(TokenChunk{"Done.", 1, {}});
        // Late delta on already-closed slot 0 is ignored cleanly.
        wire += stream.tool_call_delta(ToolCallDelta{0, "", "", R"(,"extra":1)"});
        wire += stream.finish(GenerationResult{});

        const auto frames = events(wire);
        std::size_t tool_starts = 0;
        std::size_t text_starts = 0;
        std::size_t stops = 0;
        for (const auto& f : frames) {
            const std::string t = f.find("type")->as_string();
            if (t == "content_block_start") {
                const std::string bt = f.find("content_block")->find("type")->as_string();
                if (bt == "tool_use") ++tool_starts;
                if (bt == "text") ++text_starts;
            } else if (t == "content_block_stop") {
                ++stops;
            }
        }
        CHECK(tool_starts == 1u);
        CHECK(text_starts == 1u);
        CHECK(stops == 2u);
    }

    // 2. Empty-name tool delta alone must not emit an invalid tool_use block.
    {
        AnthropicStream stream("msg_bad_delta", "mock:tiny");
        std::string wire = stream.start();
        wire += stream.tool_call_delta(ToolCallDelta{0, "call_0", "", "{}"});
        wire += stream.finish(GenerationResult{});
        const auto frames = events(wire);
        bool saw_tool_start = false;
        std::string sr;
        for (const auto& f : frames) {
            const std::string t = f.find("type")->as_string();
            if (t == "content_block_start" && f.find("content_block")->find("type")->as_string() == "tool_use") {
                saw_tool_start = true;
            }
            if (t == "message_delta") {
                sr = f.find("delta")->find("stop_reason")->as_string();
            }
        }
        CHECK_FALSE(saw_tool_start);
        CHECK(sr == "end_turn");
    }

    // 3. Empty id synthesis and empty-name filtering in anthropic_message_response and tool_call().
    {
        GenerationResult r;
        const auto doc = anthropic_message_response("msg_synth", "mock:tiny", r, {},
                                                    {ToolCall{"", "", "{}"}, ToolCall{"", "valid_fn", R"({"k":1})"}});
        const auto& content = doc.find("content")->as_array();
        REQUIRE(content.size() == 1u);
        CHECK(content[0].find("type")->as_string() == "tool_use");
        CHECK(content[0].find("name")->as_string() == "valid_fn");
        CHECK(content[0].find("id")->as_string().rfind("toolu_", 0) == 0);
        CHECK(doc.find("stop_reason")->as_string() == "tool_use");

        AnthropicStream s("msg_tc", "mock:tiny");
        std::string wire = s.start();
        wire += s.tool_call(ToolCall{"", "valid_fn", ""});
        wire += s.finish(GenerationResult{});
        const auto frames = events(wire);
        CHECK(frames[1].find("content_block")->find("id")->as_string().rfind("toolu_", 0) == 0);
    }
}

TEST_CASE("tool_call_parser hardening: chunk-split fences, non-boundary tags, linear scan, and validation") {
    // 1. Fence identifiers like ```tool_call_example stay plain text even when split across chunks.
    {
        const std::string raw = "Before ```tool_call_example\n{\"name\":\"fn\",\"arguments\":{}}\n``` after";
        for (std::size_t step : {std::size_t{1}, std::size_t{3}, std::size_t{12}}) {
            CAPTURE(step);
            ToolCallStreamParser parser("c_");
            std::string vis;
            std::vector<ParsedToolEvent> evs;
            for (std::size_t i = 0; i < raw.size(); i += step) {
                const auto out = parser.feed(std::string_view(raw).substr(i, std::min(step, raw.size() - i)));
                vis += out.visible_text;
                for (const auto& e : out.tool_calls) evs.push_back(e);
            }
            const auto tail = parser.flush();
            vis += tail.visible_text;
            for (const auto& e : tail.tool_calls) evs.push_back(e);
            CHECK(evs.empty());
            CHECK(vis == raw);
        }
    }

    // 2. Bare trailing ```tool_call at EOF flushes cleanly as visible text.
    {
        ToolCallStreamParser parser("c_");
        std::string vis = parser.feed("hello ```tool_call").visible_text;
        vis += parser.flush().visible_text;
        CHECK_FALSE(parser.has_tool_calls());
        CHECK(vis == "hello ```tool_call");
    }

    // 3. Embedded escaped quotes and closing tags inside JSON strings survive 1-byte chunk splits.
    {
        const std::string raw =
            "```tool_call\n{\"name\":\"emit_code\",\"arguments\":{\"snippet\":\"a \\\"```\\\" b\"}}\n```"
            "<tool_call>{\"name\":\"emit_xml\",\"arguments\":{\"body\":\"x \\\"</tool_call>\\\" y\"}}</tool_call>";
        ToolCallStreamParser parser("c_");
        std::string vis;
        std::vector<ParsedToolEvent> evs;
        for (const char ch : raw) {
            const auto out = parser.feed(std::string_view(&ch, 1));
            vis += out.visible_text;
            for (const auto& e : out.tool_calls) evs.push_back(e);
        }
        const auto tail = parser.flush();
        vis += tail.visible_text;
        for (const auto& e : tail.tool_calls) evs.push_back(e);
        CHECK(vis.empty());
        REQUIRE(evs.size() == 2u);
        CHECK(evs[0].name == "emit_code");
        CHECK(evs[1].name == "emit_xml");
        auto p0 = json::parse(evs[0].arguments_json);
        auto p1 = json::parse(evs[1].arguments_json);
        REQUIRE((p0.ok() && p1.ok()));
        CHECK(p0->find("snippet")->as_string() == "a \"```\" b");
        CHECK(p1->find("body")->as_string() == "x \"</tool_call>\" y");
    }

    // 4. Malformed tool names, non-object arguments, and >64 batched calls fall back to visible text.
    for (const char* bad_block : {
             R"(<tool_call>{"name":"bad name!","arguments":{}}</tool_call>)",
             R"(<tool_call>{"name":"ok","arguments":42}</tool_call>)",
             R"(<tool_call>{"name":"ok","arguments":[1,2]}</tool_call>)",
             R"(<tool_call>{"name":"ok","arguments":"not json"}</tool_call>)",
             R"(<tool_call>{"name":"ok","arguments":"[1,2]"}</tool_call>)",
         }) {
        CAPTURE(bad_block);
        const auto ext = extract_tool_calls(bad_block, "c_");
        CHECK(ext.tool_calls.empty());
        CHECK(ext.visible_text == bad_block);
    }
    {
        json::Array too_many_batch;
        for (std::size_t i = 0; i < kMaxToolCallsPerMessage + 1u; ++i) {
            too_many_batch.push_back(json::Object{{"name", "fn"}, {"arguments", json::Object{}}});
        }
        const std::string batch_block =
            "<tool_call>" + json::Value(std::move(too_many_batch)).dump() + "</tool_call>";
        const auto ext = extract_tool_calls(batch_block, "c_");
        CHECK(ext.tool_calls.empty());
        CHECK(ext.visible_text == batch_block);
    }
    // Valid stringified JSON object arguments succeed.
    {
        const auto ext = extract_tool_calls(R"(<tool_call>{"name":"ok","arguments":"{\"n\":7}"}</tool_call>)", "c_");
        REQUIRE(ext.tool_calls.size() == 1u);
        CHECK(ext.tool_calls[0].arguments_json == R"({"n":7})");
    }

    // 5. Large 64 KiB payload fed in 32-byte chunks runs in linear time, and >256 KiB payload resets safely.
    {
        const std::string big_val(64u * 1024u, 'z');
        const std::string raw = "<tool_call>{\"name\":\"big\",\"arguments\":{\"d\":\"" + big_val + "\"}}</tool_call>";
        ToolCallStreamParser parser("c_");
        for (std::size_t i = 0; i < raw.size(); i += 32u) {
            (void)parser.feed(std::string_view(raw).substr(i, std::min<std::size_t>(32u, raw.size() - i)));
        }
        (void)parser.flush();
        REQUIRE(parser.completed_calls().size() == 1u);
        CHECK(parser.completed_calls()[0].name == "big");
    }
    {
        const std::string huge = "<tool_call>" + std::string(kMaxToolCallPayloadBytes + 16u, 'x') +
                                 "<tool_call>{\"name\":\"after\",\"arguments\":{}}</tool_call>";
        const auto ext = extract_tool_calls(huge, "c_");
        REQUIRE(ext.tool_calls.size() == 1u);
        CHECK(ext.tool_calls[0].name == "after");
        CHECK(ext.visible_text.size() > kMaxToolCallPayloadBytes);
    }

    // 6. parse_openai_message_tool_calls enforces id, name, count, and payload size bounds.
    {
        std::vector<ToolCall> calls;
        auto bad_name = json::parse(R"([{"id":"call_1","type":"function","function":{"name":"a b","arguments":"{}"}}])");
        REQUIRE(bad_name.ok());
        CHECK(parse_openai_message_tool_calls(bad_name.value(), calls, "m.tool_calls").has_value());

        auto bad_id = json::parse(R"([{"id":"call_\n1","type":"function","function":{"name":"ok","arguments":"{}"}}])");
        REQUIRE(bad_id.ok());
        CHECK(parse_openai_message_tool_calls(bad_id.value(), calls, "m.tool_calls").has_value());

        json::Array too_many;
        for (std::size_t i = 0; i < kMaxToolCallsPerMessage + 1u; ++i) {
            too_many.push_back(json::Object{{"id", "c" + std::to_string(i)},
                                            {"type", "function"},
                                            {"function", json::Object{{"name", "fn"}, {"arguments", "{}"}}}});
        }
        CHECK(parse_openai_message_tool_calls(json::Value(std::move(too_many)), calls, "m.tool_calls").has_value());
    }
}

TEST_CASE("reasoning_budget hardening: control chars, UTF-8 clamping, and pin vs thinking-off precedence") {
    // 1. Disallowed control characters in reasoning_budget_message fail with 400; \n and \t remain allowed.
    {
        json::Object bad_ctrl{{"reasoning_budget_message", std::string("stop\x1b[31m")}};
        ChatJob job;
        const auto err = reasoning_budget::parse_body(bad_ctrl, job);
        REQUIRE(err.has_value());
        CHECK(err->status == 400);
        CHECK(err->param == std::optional<std::string>("reasoning_budget_message"));

        json::Object nul_msg{{"reasoning_budget_message", std::string("a\0b", 3)}};
        CHECK(reasoning_budget::parse_body(nul_msg, job).has_value());

        json::Object ok_ws{{"reasoning_budget_message", "line1\n\tline2"}};
        CHECK_FALSE(reasoning_budget::parse_body(ok_ws, job).has_value());
        CHECK(job.reasoning_budget_message == std::optional<std::string>("line1\n\tline2"));
    }

    // 2. When thinking is explicitly disabled, positive server default pin is NOT injected,
    //    while explicit request budgets (body or header) still take effect.
    {
        ChatJob job;
        ServerOptions pins;
        pins.pin_reasoning_budget_tokens = 512;
        pins.pin_reasoning_budget_message = "wrap up";
        RequestOptions req;
        req.thinking.enable_thinking = false;
        std::vector<std::string> warnings;
        reasoning_budget::apply(job, {}, pins, "llamaserver", req, warnings);
        CHECK_FALSE(req.reasoning_budget_tokens.has_value());
        CHECK_FALSE(req.reasoning_budget_message.has_value());
        CHECK(warnings.empty());

        // Explicit body budget=-1 still passes through even with enable_thinking=false.
        job.reasoning_budget_tokens = -1;
        reasoning_budget::apply(job, {}, pins, "llamaserver", req, warnings);
        CHECK(req.reasoning_budget_tokens == std::optional<std::int64_t>(-1));
        CHECK(req.reasoning_budget_message == std::optional<std::string>("wrap up"));
    }

    // 3. Programmatic budget < -1 clamps to -1 and >512-byte UTF-8 pin message clamps on a code-point boundary.
    {
        ChatJob job;
        ServerOptions pins;
        pins.pin_reasoning_budget_tokens = -42;
        std::string over_utf8;
        // 171 * 3 bytes = 513 bytes of U+20AC ("€") -> must clamp cleanly to 510 bytes (170 * 3).
        for (int i = 0; i < 171; ++i) over_utf8 += "\xE2\x82\xAC";
        pins.pin_reasoning_budget_message = over_utf8;
        RequestOptions req;
        std::vector<std::string> warnings;
        reasoning_budget::apply(job, {}, pins, "llamaserver", req, warnings);
        CHECK(req.reasoning_budget_tokens == std::optional<std::int64_t>(-1));
        REQUIRE(req.reasoning_budget_message.has_value());
        CHECK(req.reasoning_budget_message->size() == 510u);
        CHECK(warnings.size() == 2u);
    }
}

TEST_CASE("json_schema_gbnf hardening: UTF-8/surrogates, JSON-safe whitespace, nullable ReDoS, type-const, cycles") {
    // 1. Multi-byte UTF-8 pattern atoms and Unicode character-class ranges stay atomic and valid UTF-8.
    {
        const auto s_cafe = json::parse(std::string(R"({"type":"string","pattern":"^caf)") + "\xC3\xA9" + R"(+$"})");
        REQUIRE(s_cafe.ok());
        const auto g_cafe = json_schema_to_gbnf(s_cafe.value());
        REQUIRE_MESSAGE(g_cafe.ok(), g_cafe.status().message());
        CHECK(g_cafe->find(std::string("\"") + "\xC3\xA9" + "\"+") != std::string::npos);

        const auto s_emoji = json::parse(std::string(R"({"type":"string","pattern":"^)") + "\xF0\x9F\x9A\x80" + R"(?$"})");
        REQUIRE(s_emoji.ok());
        const auto g_emoji = json_schema_to_gbnf(s_emoji.value());
        REQUIRE_MESSAGE(g_emoji.ok(), g_emoji.status().message());
        CHECK(g_emoji->find(std::string("\"") + "\xF0\x9F\x9A\x80" + "\"?") != std::string::npos);

        const auto s_greek = json::parse(std::string(R"({"type":"string","pattern":"^[)") + "\xCE\xB1-\xCF\x89" + R"(]{1,4}$"})");
        REQUIRE(s_greek.ok());
        CHECK(json_schema_to_gbnf(s_greek.value()).ok());

        const auto s_inv = json::parse(std::string(R"({"type":"string","pattern":"^[)") + "\xCF\x89-\xCE\xB1" + R"(]$"})");
        REQUIRE(s_inv.ok());
        CHECK_FALSE(json_schema_to_gbnf(s_inv.value()).ok());
    }

    // 2. Invalid UTF-8 and UTF-16 surrogate sequences in pattern fail closed.
    for (const std::string& bad_pat : {
             std::string("^\x80$"),              // stray continuation byte
             std::string("^\xC0\xAF$"),          // overlong '/'
             std::string("^\xED\xA0\x80$"),      // UTF-16 high surrogate U+D800
             std::string("^\xE2\x82$"),          // truncated 3-byte sequence
             std::string("^\xF4\x90\x80\x80$"),  // > U+10FFFF
         }) {
        json::Object schema{{"type", "string"}, {"pattern", bad_pat}};
        CHECK_FALSE(json_schema_to_gbnf(json::Value(std::move(schema))).ok());
    }

    // 3. \s never emits a raw 0x09 tab into a GBNF character class inside string pattern rules
    //    (RFC 8259 safe JSON strings; the trailing structural `space ::=` rule is outside strings).
    {
        for (const char* pat : {R"(^a\sb$)", R"(^[a-z\s]+$)", R"(^[^\s]+$)", R"(^a\t\nb$)"}) {
            CAPTURE(pat);
            json::Object schema{{"type", "string"}, {"pattern", pat}};
            const auto g = json_schema_to_gbnf(json::Value(std::move(schema)));
            REQUIRE_MESSAGE(g.ok(), g.status().message());
            const std::size_t common_pos = g->find("space ::=");
            REQUIRE(common_pos != std::string::npos);
            const std::string_view pattern_rules(g->data(), common_pos);
            CHECK(pattern_rules.find("[ \\t]") == std::string_view::npos);
            CHECK(pattern_rules.find("a-z \\t") == std::string_view::npos);
            if (std::string_view(pat) != R"(^[^\s]+$)") {
                CHECK(pattern_rules.find(R"("\\t")") != std::string_view::npos);
            }
        }
    }

    // 4. Nullable/optional nested unbounded quantifiers fail closed (Earley/ReDoS protection).
    for (const char* redos_pat : {R"(^(a?)+$)", R"(^(a?)*$)", R"(^(|a)+$)", R"(^(a{0,5})+$)", R"(^()+$)", R"(^(a+){2}$)"}) {
        CAPTURE(redos_pat);
        json::Object schema{{"type", "string"}, {"pattern", redos_pat}};
        CHECK_FALSE(json_schema_to_gbnf(json::Value(std::move(schema))).ok());
    }
    // Bounded optional repetition and non-nullable alternation stay valid.
    for (const char* safe_pat : {R"(^(a?){3}$)", R"(^(a|b)+$)", R"(^()$)"}) {
        CAPTURE(safe_pat);
        json::Object schema{{"type", "string"}, {"pattern", safe_pat}};
        CHECK(json_schema_to_gbnf(json::Value(std::move(schema))).ok());
    }

    // 5. Sibling type + const/enum consistency checks fail closed on mismatch, allow numeric promotion.
    {
        for (const char* mismatch : {
                 R"({"type":"string","const":42})",
                 R"({"type":"integer","enum":["a",1]})",
                 R"({"type":"boolean","const":"true"})",
             }) {
            CAPTURE(mismatch);
            auto v = json::parse(mismatch);
            REQUIRE(v.ok());
            CHECK_FALSE(json_schema_to_gbnf(v.value()).ok());
        }
        for (const char* ok_s : {
                 R"({"type":"number","const":1})",
                 R"({"type":"integer","const":42})",
                 R"({"type":["string","null"],"enum":["ok",null]})",
             }) {
            CAPTURE(ok_s);
            auto v = json::parse(ok_s);
            REQUIRE(v.ok());
            CHECK(json_schema_to_gbnf(v.value()).ok());
        }
    }

    // 6. Exact int64 vs double equality (2^53+1 != 2^53.0), cyclic $ref rejection, empty $ref key,
    //    maxLength: 0 emission, and deep nesting > 32 rejection.
    {
        auto distinct_big = json::parse(R"({"oneOf":[{"const":9007199254740993},{"const":9007199254740992.0}]})");
        REQUIRE(distinct_big.ok());
        CHECK(json_schema_to_gbnf(distinct_big.value()).ok());

        auto cycle = json::parse(R"({"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"$ref":"#/$defs/A"}},"$ref":"#/$defs/A"})");
        REQUIRE(cycle.ok());
        CHECK_FALSE(json_schema_to_gbnf(cycle.value()).ok());

        auto empty_ref = json::parse(R"({"$defs":{"":{"type":"string"}},"$ref":"#/$defs/"})");
        REQUIRE(empty_ref.ok());
        CHECK_FALSE(json_schema_to_gbnf(empty_ref.value()).ok());

        auto zero_str = json::parse(R"({"type":"string","minLength":0,"maxLength":0})");
        REQUIRE(zero_str.ok());
        const auto g_zero = json_schema_to_gbnf(zero_str.value());
        REQUIRE(g_zero.ok());
        CHECK(g_zero->find("char{0,0}") == std::string::npos);

        // Deep nesting > kMaxSchemaDepth (32) fails cleanly without stack overflow.
        json::Value deep(json::Object{{"type", "string"}});
        for (int i = 0; i < 40; ++i) {
            deep = json::Value(json::Object{{"type", "array"}, {"items", std::move(deep)}});
        }
        CHECK_FALSE(json_schema_to_gbnf(deep).ok());
    }
}
