#include <doctest/doctest.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "src/anthropic.hpp"
#include "src/request_path.hpp"

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
    for (const char* field : {"tools", "tool_choice"}) {
        body = request();
        body.set(field, json::Array{});
        rejected(body, field);
    }
    for (const char* type : {"image", "tool_use", "tool_result", "audio"}) {
        const json::Array blocks{json::Object{{"type", type}, {"source", json::Object{}}}};
        body = request();
        body.set("messages", json::Array{json::Object{{"role", "user"}, {"content", blocks}}});
        rejected(body, type == std::string("image") ? "image" : "content block");
        body = request();
        body.set("system", blocks);
        rejected(body, type == std::string("image") ? "image" : "content block");
    }
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
