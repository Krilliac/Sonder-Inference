// Pure-function tests for the server module: SHA-256, the HTTP head parser,
// query decoding, OpenAI request mapping, correlation headers, error mapping,
// the live telemetry hub (ring, fan-out, resume), backend setup and identity,
// and the log queue (bounded, never waits for its sink).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "server_test_support.hpp"
#include "src/http.hpp"
#include "src/identity.hpp"
#include "src/live_hub.hpp"
#include "src/log_queue.hpp"
#include "src/openai.hpp"
#include "src/responses_handler.hpp"
#include "src/server_endpoints.hpp"
#include "src/sha256.hpp"
#include "src/tool_call_parser.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/launch_profile.hpp"

using namespace sonder::inference;
using namespace sonder::inference::server::detail;

TEST_CASE("sha256: FIPS 180-4 test vectors") {
    CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 1,000,000 x 'a' fed in odd-sized pieces exercises block boundaries.
    Sha256 h;
    const std::string piece(997, 'a');
    std::size_t fed = 0;
    while (fed + piece.size() <= 1000000) {
        h.update(piece);
        fed += piece.size();
    }
    h.update(std::string(1000000 - fed, 'a'));
    const auto d = h.finish();
    CHECK(d[0] == 0xcd);
    CHECK(d[1] == 0xc7);
    CHECK(d[31] == 0xd0);
}

TEST_CASE("http: parses a request head") {
    RequestHead head;
    const std::string raw =
        "GET /v1/sonder/identity?model=mock%3Atiny&x=1 HTTP/1.1\r\nHost: 127.0.0.1:1\r\nX-Test:  padded  \r\n\r\nrest";
    const ParseResult r = parse_request_head(raw, head);
    REQUIRE(r.state == ParseState::complete);
    CHECK(r.head_bytes == raw.size() - 4);
    CHECK(head.method == "GET");
    CHECK(head.path == "/v1/sonder/identity");
    CHECK(head.query == "model=mock%3Atiny&x=1");
    REQUIRE(head.header("x-test") != nullptr);
    CHECK(*head.header("x-test") == "padded");
    CHECK(query_param(head.query, "model") == std::optional<std::string>("mock:tiny"));
    CHECK(query_param(head.query, "x") == std::optional<std::string>("1"));
    CHECK_FALSE(query_param(head.query, "y").has_value());
    CHECK_FALSE(url_decode("%zz").has_value());
    CHECK_FALSE(url_decode("%4").has_value());
    CHECK(url_decode("a+b%20c") == std::optional<std::string>("a b c"));
}

TEST_CASE("http: incomplete, malformed and oversized heads") {
    RequestHead head;
    CHECK(parse_request_head("GET / HTTP/1.1\r\nHost: x\r\n", head).state == ParseState::incomplete);
    const auto status_of = [&](const std::string& raw) {
        const ParseResult r = parse_request_head(raw, head);
        return r.state == ParseState::error ? r.status : 0;
    };
    CHECK(status_of("GET / HTTP/1.1 extra\r\n\r\n") == 400);
    CHECK(status_of("GET http://x/ HTTP/1.1\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/2.0\r\n\r\n") == 505);
    CHECK(status_of("GET / HTTX/1.1\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/1.1\r\nBad Header: x\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/1.1\r\nA: b\r\n folded\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/1.1\r\nNoColon\r\n\r\n") == 400);
    CHECK(status_of("POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n") == 400);
    CHECK(status_of("POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n") == 400);
    CHECK(status_of("POST / HTTP/1.1\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/1.1\r\nA: b\x01\r\n\r\n") == 400);
    // RFC 9112 section 3.2: more than one Host line is a 400, in any order.
    CHECK(status_of("GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.com\r\n\r\n") == 400);
    CHECK(status_of("GET / HTTP/1.1\r\nHost: evil.com\r\nX: y\r\nhost: 127.0.0.1\r\n\r\n") == 400);

    std::string many = "GET / HTTP/1.1\r\n";
    for (int i = 0; i < 65; ++i) many += "H" + std::to_string(i) + ": v\r\n";
    CHECK(status_of(many + "\r\n") == 431);
    const std::string big = "GET / HTTP/1.1\r\nX: " + std::string(kMaxHeadBytes, 'a') + "\r\n\r\n";
    CHECK(status_of(big) == 431);
    // No terminator yet but already over the limit.
    CHECK(status_of("GET / HTTP/1.1\r\nX: " + std::string(kMaxHeadBytes + 1, 'a')) == 431);

    REQUIRE(parse_request_head("POST /x HTTP/1.1\r\nContent-Length: 12\r\n\r\n", head).state == ParseState::complete);
    CHECK(head.content_length == std::optional<std::uint64_t>(12));
    REQUIRE(parse_request_head("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", head).state ==
            ParseState::complete);
    CHECK(head.has_transfer_encoding);
}

TEST_CASE("http: helpers") {
    CHECK(constant_time_equals("secret", "secret"));
    CHECK_FALSE(constant_time_equals("secret", "secreT"));
    CHECK_FALSE(constant_time_equals("secret", "secret2"));
    CHECK(is_correlation_value("rt-abc_1.2:3"));
    CHECK_FALSE(is_correlation_value(""));
    CHECK_FALSE(is_correlation_value("has space"));
    CHECK_FALSE(is_correlation_value(std::string(129, 'a')));
    CHECK(is_correlation_value(std::string(128, 'a')));
    CHECK(is_loopback_host_header("127.0.0.1"));
    CHECK(is_loopback_host_header("127.0.0.1:11437"));
    CHECK(is_loopback_host_header("LocalHost:80"));
    CHECK(is_loopback_host_header("[::1]:5"));
    CHECK(is_loopback_host_header("[::1]"));
    CHECK_FALSE(is_loopback_host_header("evil.example:11437"));
    CHECK_FALSE(is_loopback_host_header("127.0.0.1.evil.example"));
    CHECK_FALSE(is_loopback_host_header("0.0.0.0:11437"));
    CHECK_FALSE(is_loopback_host_header("127.0.0.1:"));
    CHECK_FALSE(is_loopback_host_header("127.0.0.1:abc"));
    const std::string head = format_head(200, {{"A", "b\r\nInjected: 1"}});
    CHECK(head == "HTTP/1.1 200 OK\r\nA: bInjected: 1\r\n\r\n");
}

TEST_CASE("openai: request mapping and Sonder extensions") {
    const auto parsed = parse_chat_request(
        R"({"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u","name":"x"}],
            "temperature":0.5,"top_p":0.9,"top_k":7,"min_p":0.05,"seed":9,"max_tokens":12,"stop":"END",
            "presence_penalty":0.1,"frequency_penalty":0.2,"repeat_penalty":1.3,"logit_bias":{"5":-1.5},
            "num_ctx":2048,"typical_p":0.8,"repeat_last_n":-1,"user":"ignored","chat_template_kwargs":{},
            "some_future_field":true,"n":1,"logprobs":false,"stream":true,"stream_options":{"include_usage":true}})");
    REQUIRE(std::holds_alternative<ChatJob>(parsed));
    const ChatJob& job = std::get<ChatJob>(parsed);
    CHECK(job.model == "m");
    REQUIRE(job.messages.size() == 2);
    CHECK(job.messages[1].content == "u");
    CHECK(job.stream);
    CHECK(job.include_usage);
    CHECK(job.sampling.temperature == doctest::Approx(0.5));
    CHECK(job.sampling.top_p == doctest::Approx(0.9));
    CHECK(job.sampling.top_k == 7);
    CHECK(job.sampling.min_p == doctest::Approx(0.05));
    CHECK(job.sampling.seed == std::optional<std::uint64_t>(9));
    CHECK(job.sampling.max_tokens == 12);
    REQUIRE(job.sampling.stop.size() == 1);
    CHECK(job.sampling.stop[0] == "END");
    CHECK(job.sampling.repeat_penalty == doctest::Approx(1.3));
    REQUIRE(job.sampling.logit_bias.size() == 1);
    CHECK(job.sampling.logit_bias[0].token == 5);
    CHECK(job.sampling.num_ctx == 2048);
    CHECK(job.sampling.typical_p == doctest::Approx(0.8));
    CHECK(job.sampling.repeat_last_n == -1);
    CHECK(job.sampling.explicit_only);
    CHECK(job.sampling.is_explicit(SamplingConfig::kTemperature));
    CHECK(job.sampling.is_explicit(SamplingConfig::kTopK));
    CHECK(job.sampling.is_explicit(SamplingConfig::kRepeatPenalty));
    CHECK(job.sampling.is_explicit(SamplingConfig::kRepeatLastN));

    const auto minimal = parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})");
    REQUIRE(std::holds_alternative<ChatJob>(minimal));
    CHECK(std::get<ChatJob>(minimal).model == "default");
    CHECK(std::get<ChatJob>(minimal).sampling.max_tokens == kDefaultMaxTokens);
    // No sampler field in the request: a model-default backend keeps its own.
    CHECK(std::get<ChatJob>(minimal).sampling.explicit_only);
    CHECK(std::get<ChatJob>(minimal).sampling.explicit_fields == 0u);

    const auto only_temp =
        parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"temperature":0.2,"top_p":null})");
    REQUIRE(std::holds_alternative<ChatJob>(only_temp));
    CHECK(std::get<ChatJob>(only_temp).sampling.explicit_fields == SamplingConfig::kTemperature);
    CHECK_FALSE(std::get<ChatJob>(minimal).stream);

    const auto both = parse_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":5,"max_completion_tokens":7})");
    REQUIRE(std::holds_alternative<ChatJob>(both));
    CHECK(std::get<ChatJob>(both).sampling.max_tokens == 7);
}

TEST_CASE("openai: rejected and invalid requests name the field") {
    const auto error_of = [](const std::string& body) {
        const auto parsed = parse_chat_request(body);
        REQUIRE(std::holds_alternative<ApiError>(parsed));
        return std::get<ApiError>(parsed);
    };
    const std::string msgs = R"("messages":[{"role":"user","content":"hi"}])";
    CHECK(error_of("{").code == "invalid_json");
    CHECK(error_of("[]").code == "invalid_json");
    for (const char* key : {"functions", "function_call"}) {
        const ApiError e = error_of("{" + msgs + R"(,")" + key + R"(":[1]})");
        CHECK(e.status == 400);
        CHECK(e.code == "unsupported_parameter");
        CHECK(e.param == std::optional<std::string>(key));
    }
    // Valid response_format is accepted; malformed/unknown types are rejected.
    {
        const auto rf_ok = parse_chat_request("{" + msgs + R"(,"response_format":{"type":"json_object"}})");
        REQUIRE(std::holds_alternative<ChatJob>(rf_ok));
        CHECK(std::get<ChatJob>(rf_ok).response_format_type == std::optional<std::string>("json_object"));
        CHECK_FALSE(std::get<ChatJob>(rf_ok).grammar.empty());
    }
    CHECK(error_of("{" + msgs + R"(,"response_format":{"type":"xml"}})").code == "unsupported_parameter");
    CHECK(error_of("{" + msgs + R"(,"response_format":[1]})").code == "invalid_request");
    // Valid logprobs + top_logprobs accepted; top_logprobs without logprobs:true or >20 rejected.
    {
        const auto lp_ok = parse_chat_request("{" + msgs + R"(,"logprobs":true,"top_logprobs":5})");
        REQUIRE(std::holds_alternative<ChatJob>(lp_ok));
        CHECK(std::get<ChatJob>(lp_ok).sampling.logprobs);
        CHECK(std::get<ChatJob>(lp_ok).sampling.top_logprobs == 5);
    }
    CHECK(error_of("{" + msgs + R"(,"top_logprobs":3})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"logprobs":true,"top_logprobs":21})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"logprobs":"yes"})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"n":2})").code == "unsupported_parameter");
    // Text/image content parts and assistant tool_calls are supported; unknown parts / SSRF URLs rejected.
    {
        const auto parts_ok =
            parse_chat_request(R"({"messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]})");
        REQUIRE(std::holds_alternative<ChatJob>(parts_ok));
        CHECK(std::get<ChatJob>(parts_ok).messages[0].content == "x");
    }
    CHECK(error_of(R"({"messages":[{"role":"user","content":[{"type":"input_audio","data":"x"}]}]})").code ==
          "unsupported_parameter");
    CHECK(error_of(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://127.0.0.1/a.png"}}]}]})")
              .code == "invalid_request");
    {
        const auto tc_ok = parse_chat_request(
            R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"ok"}]})");
        REQUIRE(std::holds_alternative<ChatJob>(tc_ok));
        REQUIRE(std::get<ChatJob>(tc_ok).messages.size() == 2u);
        REQUIRE(std::get<ChatJob>(tc_ok).messages[0].tool_calls.size() == 1u);
        CHECK(std::get<ChatJob>(tc_ok).messages[1].tool_call_id == std::optional<std::string>("c1"));
    }
    CHECK(error_of("{" + msgs + R"(,"tool_choice":"required"})").code == "invalid_request");
    CHECK(error_of(R"({"messages":[]})").code == "invalid_messages");
    CHECK(error_of(R"({"messages":[{"role":"narrator","content":"x"}]})").code == "invalid_messages");
    CHECK(error_of(R"({"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]})").code ==
          "invalid_messages");
    const ApiError temp = error_of("{" + msgs + R"(,"temperature":"hot"})");
    CHECK(temp.code == "invalid_sampling");
    CHECK(temp.param == std::optional<std::string>("temperature"));
    const ApiError range = error_of("{" + msgs + R"(,"temperature":50})");
    CHECK(range.code == "invalid_sampling");
    CHECK(range.message.find("temperature") != std::string::npos);
    CHECK(error_of("{" + msgs + R"(,"max_tokens":0})").param == std::optional<std::string>("max_tokens"));
    CHECK(error_of("{" + msgs + R"(,"seed":-1})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"stop":["a","b","c","d","e"]})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"logit_bias":{"x":1}})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"top_k":1.5})").code == "invalid_sampling");
    CHECK(error_of("{" + msgs + R"(,"model":7})").param == std::optional<std::string>("model"));
    CHECK(error_of("{" + msgs + R"(,"stream":"yes"})").param == std::optional<std::string>("stream"));

    const json::Object body = error_body(make_error(400, "invalid_json", "bad", std::nullopt));
    const json::Value* err = body.find("error");
    REQUIRE(err != nullptr);
    CHECK(err->find("type")->as_string() == "invalid_request_error");
    CHECK(err->find("param")->is_null());
    CHECK(body.find("sonder")->find("api_version")->as_int() == 1);
}

TEST_CASE("openai: correlation headers") {
    RequestHead head;
    head.headers = {{"x-sonder-run-id", "rt-1"},
                    {"x-sonder-parent-request-id", "turn:42"},
                    {"x-sonder-agent-id", "agent.a"},
                    {"x-sonder-task-id", "task_b"},
                    {"x-sonder-workload", "maintenance"},
                    {"x-sonder-priority", "-16"}};
    const auto ok = parse_correlation(head);
    REQUIRE(std::holds_alternative<Correlation>(ok));
    const Correlation& c = std::get<Correlation>(ok);
    CHECK(c.run_id == std::optional<std::string>("rt-1"));
    CHECK(c.parent_request_id == std::optional<std::string>("turn:42"));
    CHECK(c.workload == WorkloadClass::maintenance);
    CHECK(c.priority == -16);
    CHECK(c.priority_class == RequestPriority::interactive);

    for (const auto* value : {"interactive", "subagent", "background"}) {
        RequestHead class_head;
        class_head.headers = {{"x-sonder-priority", value}};
        const auto class_result = parse_correlation(class_head);
        REQUIRE(std::holds_alternative<Correlation>(class_result));
        CHECK(std::get<Correlation>(class_result).priority == 0);
        CHECK(std::get<Correlation>(class_result).priority_class == parse_request_priority(value));
    }

    RequestHead none;
    const auto defaults = parse_correlation(none);
    REQUIRE(std::holds_alternative<Correlation>(defaults));
    CHECK(std::get<Correlation>(defaults).workload == WorkloadClass::interactive_user);

    for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
             {"x-sonder-run-id", "has space"},
             {"x-sonder-agent-id", std::string(129, 'a')},
             {"x-sonder-workload", "urgent"},
             {"x-sonder-priority", "17"},
             {"x-sonder-priority", "backgroundx"},
             {"x-sonder-priority", "1.5"},
             {"x-sonder-priority", ""}}) {
        RequestHead bad;
        bad.headers = {{name, value}};
        const auto r = parse_correlation(bad);
        REQUIRE(std::holds_alternative<ApiError>(r));
        CHECK(std::get<ApiError>(r).code == "invalid_correlation_header");
    }
}

TEST_CASE("openai: session failures map to API errors") {
    const ApiError rejected = map_session_failure(Status(ErrorCode::internal, "scheduler rejected"), true);
    CHECK(rejected.status == 429);
    CHECK(rejected.code == "overloaded");
    CHECK(rejected.retry_after == std::optional<int>(1));
    const ApiError never_fits = map_session_failure(Status(ErrorCode::invalid_argument, "prompt needs 9 KV tokens"), true);
    CHECK(never_fits.status == 400);
    CHECK(never_fits.code == "invalid_messages");
    CHECK(never_fits.param == std::optional<std::string>("messages"));
    // Backend refusals of a request field are not blamed on the messages.
    const ApiError bias = map_session_failure(
        Status(ErrorCode::invalid_argument, "the ollama backend does not support logit_bias"), false);
    CHECK(bias.status == 400);
    CHECK(bias.code == "unsupported_parameter");
    CHECK(bias.param == std::optional<std::string>("logit_bias"));
    const ApiError other = map_session_failure(Status(ErrorCode::invalid_argument, "temperature out of range"), false);
    CHECK(other.status == 400);
    CHECK(other.code == "invalid_request");
    CHECK_FALSE(other.param.has_value());
    CHECK(map_session_failure(Status(ErrorCode::invalid_argument, "backend does not support 42!"), false).code ==
          "invalid_request");
    CHECK(map_session_failure(Status(ErrorCode::invalid_argument, "sampling failed: x"), false).code ==
          "invalid_sampling");
    CHECK(map_session_failure(Status(ErrorCode::not_found, "x"), false).code == "model_not_found");
    CHECK(map_session_failure(Status(ErrorCode::unsupported, "x"), false).code == "unsupported_parameter");
    CHECK(map_session_failure(Status(ErrorCode::unavailable, "x"), false).code == "backend_unavailable");
    CHECK(map_session_failure(Status(ErrorCode::backend_error, "x"), false).status == 503);
    const ApiError internal = map_session_failure(Status(ErrorCode::internal, "x"), false);
    CHECK(internal.status == 500);
    CHECK(internal.code == "internal_error");
    CHECK(std::string(error_type(500)) == "server_error");
    CHECK(std::string(error_type(503)) == "service_unavailable");
}

TEST_CASE("hub: ring, fan-out, per-subscriber drops and resume") {
    LiveTelemetryHub hub(4, 2, 3);
    hub.set_instance_id("tel-abc");
    for (std::uint64_t i = 0; i < 6; ++i) {
        hub.write_event(i, "{\"sequence\":" + std::to_string(i) + "}");
    }
    HubStats st = hub.stats();
    CHECK(st.retained == 4);
    CHECK(st.oldest_sequence == std::optional<std::uint64_t>(2));
    CHECK(st.next_sequence == 6);

    const auto sequences = [](const Subscription::Batch& b) {
        std::vector<std::uint64_t> out;
        for (const auto& e : b.events) out.push_back(e->sequence);
        return out;
    };

    // Same instance, inside the window: replay after the id.
    {
        ResumeRequest r;
        r.last_event_id = "tel-abc-3";
        auto s = hub.subscribe(r);
        REQUIRE(s.subscription);
        CHECK(s.gap_comment.empty());
        CHECK(sequences(s.subscription->next(std::chrono::milliseconds(0), nullptr)) ==
              std::vector<std::uint64_t>{4, 5});
        hub.unsubscribe(s.subscription);
    }
    // Same instance, older than the window: gap comment then the whole window
    // (queue of 3 keeps the newest three and counts one drop).
    {
        ResumeRequest r;
        r.last_event_id = "tel-abc-0";
        auto s = hub.subscribe(r);
        REQUIRE(s.subscription);
        CHECK(s.gap_comment == ": resume-gap 1-1");
        const auto batch = s.subscription->next(std::chrono::milliseconds(0), nullptr);
        CHECK(sequences(batch) == std::vector<std::uint64_t>{3, 4, 5});
        CHECK(batch.dropped == 1);
        hub.unsubscribe(s.subscription);
    }
    // Unknown instance (producer restarted) and malformed ids: whole window.
    for (const char* id : {"tel-other-5", "garbage", "tel-abc-x"}) {
        ResumeRequest r;
        r.last_event_id = id;
        auto s = hub.subscribe(r);
        REQUIRE(s.subscription);
        CHECK(s.gap_comment.empty());
        CHECK(s.subscription->next(std::chrono::milliseconds(0), nullptr).events.size() == 3);
        hub.unsubscribe(s.subscription);
    }
    // Caught up: nothing replayed. since=now: live only.
    {
        ResumeRequest caught;
        caught.last_event_id = "tel-abc-5";
        auto a = hub.subscribe(caught);
        ResumeRequest now;
        now.since_now = true;
        auto b = hub.subscribe(now);
        REQUIRE(a.subscription);
        REQUIRE(b.subscription);
        CHECK(a.subscription->next(std::chrono::milliseconds(0), nullptr).events.empty());
        CHECK(b.subscription->next(std::chrono::milliseconds(0), nullptr).events.empty());
        // Cap of two subscribers.
        CHECK(hub.subscribe(ResumeRequest{}).error == LiveTelemetryHub::SubscribeError::over_capacity);
        // Live fan-out; a subscriber that does not drain loses its oldest events.
        for (std::uint64_t i = 6; i < 11; ++i) {
            hub.write_event(i, "{}");
        }
        const auto live = a.subscription->next(std::chrono::milliseconds(0), nullptr);
        CHECK(sequences(live) == std::vector<std::uint64_t>{8, 9, 10});
        CHECK(live.dropped == 2);
        CHECK(hub.stats().subscriber_dropped_events >= 2);
        hub.close();
        CHECK(b.subscription->next(std::chrono::milliseconds(0), nullptr).events.size() == 3);
        CHECK(b.subscription->next(std::chrono::milliseconds(0), nullptr).closed);
        CHECK(hub.subscribe(ResumeRequest{}).error == LiveTelemetryHub::SubscribeError::closed);
    }
}

TEST_CASE("hub: recovers the sequence when fed through write()") {
    LiveTelemetryHub hub(8, 1, 8);
    hub.write(R"({"sequence":41,"event_type":"x"})");
    hub.write("not json");
    CHECK(hub.stats().retained == 1);
    CHECK(hub.stats().next_sequence == 42);
}

TEST_CASE("backend setup: env defaults and factory") {
    const auto env = backend_env_defaults([](const char* name) -> std::optional<std::string> {
        const std::string n = name;
        if (n == "SONDER_INFER_BACKEND") return std::string("ollama");
        if (n == "SONDER_INFER_MODEL") return std::string("");
        if (n == "OLLAMA_HOST") return std::string("0.0.0.0");
        return std::nullopt;
    });
    CHECK(env.backend == std::optional<std::string>("ollama"));
    CHECK_FALSE(env.model.has_value());
    CHECK(env.ollama_url == std::optional<std::string>("http://127.0.0.1:11434"));
    const auto explicit_url = backend_env_defaults([](const char* name) -> std::optional<std::string> {
        const std::string n = name;
        if (n == "SONDER_OLLAMA_URL") return std::string("http://127.0.0.1:9");
        if (n == "OLLAMA_HOST") return std::string("ignored:1");
        return std::nullopt;
    });
    CHECK(explicit_url.ollama_url == std::optional<std::string>("http://127.0.0.1:9"));
    CHECK(normalize_ollama_host("myhost") == "http://myhost:11434");
    CHECK(normalize_ollama_host("myhost:1234") == "http://myhost:1234");
    CHECK(normalize_ollama_host("https://h/") == "https://h:443");
    CHECK(normalize_ollama_host("[::1]:7") == "http://[::1]:7");

    const auto names = available_backend_names();
    REQUIRE_FALSE(names.empty());
    CHECK(names.front() == "mock");
    CHECK(is_synthetic_backend("mock"));
    CHECK_FALSE(is_synthetic_backend("ollama"));
    BackendSetup mock;
    mock.backend = "mock";
    auto made = make_backend(mock);
    REQUIRE(made.ok());
    CHECK(made.value()->name() == "mock");
    BackendSetup unknown;
    unknown.backend = "vllm";
    CHECK(make_backend(unknown).status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("identity: mock has the nine keys and 64-hex digests, others are honest nulls") {
    auto backend = make_mock_backend();
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto impl = backend->load_model(lo);
    REQUIRE(impl.ok());
    Model model("model-1", "mock", "cpu:0", impl.value());
    const IdentityResult id = backend_identity(model, std::string("mock-1"));
    REQUIRE(id.backend_identity.is_object());
    CHECK_FALSE(id.reason.has_value());
    const auto& o = id.backend_identity.as_object();
    CHECK(o.size() == 9);
    for (const char* key : {"backend", "model", "model_digest", "quantization", "backend_version", "tokenizer_digest",
                            "template_digest", "context_tokens", "hardware"}) {
        CHECK_MESSAGE(o.contains(key), key);
    }
    for (const char* key : {"model_digest", "tokenizer_digest", "template_digest"}) {
        const std::string& digest = o.find(key)->as_string();
        CHECK(digest.size() == 64);
        CHECK(digest.find_first_not_of("0123456789abcdef") == std::string::npos);
    }
    CHECK(o.find("context_tokens")->as_int() > 0);
    CHECK_FALSE(o.find("hardware")->as_string().empty());
    CHECK(o.find("backend_version")->as_string() == "mock-1");
    // Deterministic across calls.
    CHECK(backend_identity(model, std::string("mock-1")).backend_identity.dump() == id.backend_identity.dump());

    Model ollama_model("model-2", "ollama", "cpu:0", impl.value());
    const IdentityResult none = backend_identity(ollama_model, std::nullopt);
    CHECK(none.backend_identity.is_null());
    REQUIRE(none.reason.has_value());
    CHECK(none.reason->find("tokenizer") != std::string::npos);
    Model llama_model("model-3", "llamacpp", "cpu:0", impl.value());
    CHECK(backend_identity(llama_model, std::nullopt).backend_identity.is_null());
}

TEST_CASE("options: refused combinations") {
    namespace srv = sonder::inference::server;
    srv::ServerOptions o;
    o.backend.backend = "mock";
    CHECK(srv::validate_options(o).ok());
    o.host = "0.0.0.0";
    CHECK(srv::validate_options(o).code() == ErrorCode::invalid_argument);
    o.token = "t";
    CHECK(srv::validate_options(o).ok());
    // A DNS name that only looks like loopback may resolve to a LAN address:
    // it needs a token like any other non-loopback bind.
    o.token.clear();
    for (const char* host : {"127.0.0.1.attacker.example", "127.evil", "127.0.0.1.", "127.0.0.01", "127.1"}) {
        CAPTURE(host);
        o.host = host;
        CHECK(srv::validate_options(o).code() == ErrorCode::invalid_argument);
    }
    for (const char* host : {"127.0.0.1", "127.9.8.7", "localhost", "::1", "[::1]"}) {
        CAPTURE(host);
        o.host = host;
        CHECK(srv::validate_options(o).ok());
    }
    o.host = "127.0.0.1";
    o.token.clear();
    o.capture_text = true;
    CHECK(srv::validate_options(o).message().find("--capture-text") != std::string::npos);
    o.capture_text = false;
    o.backend.backend = "ollama";
    CHECK(srv::validate_options(o).message().find("sonder-infer models --backend ollama") != std::string::npos);
    o.models = {"a", "a"};
    CHECK_FALSE(srv::validate_options(o).ok());
    o.models = {"default"};
    CHECK_FALSE(srv::validate_options(o).ok());
    o.models = {"a"};
    o.cors_origins = {"*"};
    CHECK_FALSE(srv::validate_options(o).ok());
    // Only origins a browser can send match: scheme://host[:port], lowercase.
    for (const char* bad : {"http://127.0.0.1:4173/", "http://127.0.0.1:4173/app", "HTTP://127.0.0.1:4173",
                            "http://Localhost:5173", "null", "127.0.0.1:4173", "http://", "http://a b",
                            "http://127.0.0.1:", "http://127.0.0.1:12x", "http://u@127.0.0.1", "http://h?x=1",
                            "http://[::1", "http://[]:80", "http://a,http://b"}) {
        CAPTURE(bad);
        o.cors_origins = {bad};
        CHECK(srv::validate_options(o).code() == ErrorCode::invalid_argument);
    }
    for (const char* good : {"http://127.0.0.1:4173", "https://observatory.example", "tauri://localhost",
                             "http://tauri.localhost", "http://[::1]:5173"}) {
        CAPTURE(good);
        o.cors_origins = {good};
        CHECK(srv::validate_options(o).ok());
    }
    o.cors_origins.clear();
    // A backend this build does not have is a usage error, not a runtime one.
    o.backend.backend = "vllm";
    const Status unknown = srv::validate_options(o);
    CHECK(unknown.code() == ErrorCode::invalid_argument);
    CHECK(unknown.message().find("vllm") != std::string::npos);
    CHECK(unknown.message().find("mock") != std::string::npos);
    // A caller-built backend replaces the name check.
    o.backend.backend.clear();
    o.backend_instance = sonder::inference::make_mock_backend();
    o.models.clear();
    CHECK(srv::validate_options(o).ok());
    // The log queue needs room for at least one line.
    o.log_queue_capacity = 0;
    const Status no_queue = srv::validate_options(o);
    CHECK(no_queue.code() == ErrorCode::invalid_argument);
    CHECK(no_queue.message().find("log_queue_capacity") != std::string::npos);
    o.log_queue_capacity = 1;
    CHECK(srv::validate_options(o).ok());
}

TEST_CASE("backend setup: tensor overrides parse as PATTERN=DEVICE") {
    auto ok = parse_tensor_override("ffn_.*_exps=cpu");
    REQUIRE(ok);
    CHECK(ok.value().first == "ffn_.*_exps");
    CHECK(ok.value().second == "cpu");
    // Split at the last '=' so a pattern may itself contain '='.
    auto eq = parse_tensor_override("a=b=Vulkan0");
    REQUIRE(eq);
    CHECK(eq.value().first == "a=b");
    CHECK(eq.value().second == "Vulkan0");
    CHECK_FALSE(parse_tensor_override("no-device"));
    CHECK_FALSE(parse_tensor_override("=cpu"));
    CHECK_FALSE(parse_tensor_override("pattern="));
    CHECK(parse_tensor_override("=cpu").status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("backend setup: llama.cpp context options are validated up front") {
    BackendSetup unset;
    unset.backend = "llamacpp";
    CHECK(validate_backend_setup(unset).ok());  // defaults are always valid
    CHECK_FALSE(unset.llamacpp_batch_size.has_value());
    CHECK_FALSE(unset.llamacpp_ubatch_size.has_value());
    CHECK_FALSE(unset.llamacpp_kv_cache_type_k.has_value());
    CHECK_FALSE(unset.llamacpp_kv_cache_type_v.has_value());
    CHECK_FALSE(unset.llamacpp_flash_attention.has_value());

    BackendSetup ok = unset;
    ok.llamacpp_kv_cache_type_k = "q8_0";
    ok.llamacpp_kv_cache_type_v = "q8_0";
    ok.llamacpp_flash_attention = "on";
    ok.llamacpp_batch_size = 1024;
    ok.llamacpp_ubatch_size = 256;
    CHECK(validate_backend_setup(ok).ok());

    BackendSetup bad = unset;
    bad.llamacpp_kv_cache_type_v = "q4_0";
    bad.llamacpp_flash_attention = "off";
    // Options for other backends are never checked.
    BackendSetup mock = bad;
    mock.backend = "mock";
    CHECK(validate_backend_setup(mock).ok());
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    const Status st = validate_backend_setup(bad);
    CHECK(st.code() == ErrorCode::invalid_argument);
    CHECK(st.message().find("flash attention") != std::string::npos);
    CHECK(make_backend(bad).status().code() == ErrorCode::invalid_argument);
    BackendSetup name = unset;
    name.llamacpp_kv_cache_type_k = "q9_9";
    CHECK(validate_backend_setup(name).code() == ErrorCode::invalid_argument);
    BackendSetup ub = unset;
    ub.llamacpp_ubatch_size = 1024;  // > default batch 512
    CHECK(validate_backend_setup(ub).code() == ErrorCode::invalid_argument);
    BackendSetup tov = unset;
    tov.llamacpp_tensor_overrides = {"no-device"};
    CHECK(validate_backend_setup(tov).code() == ErrorCode::invalid_argument);
    auto made = make_backend(ok);
    REQUIRE(made.ok());
    CHECK(made.value()->name() == "llamacpp");
#else
    // Without llama.cpp there is nothing to check against.
    CHECK(validate_backend_setup(bad).ok());
#endif
}

namespace {

// A LogQueue sink that records every line and, while closed, blocks each
// call. A blocked call gives up after 10 s and leaves the gate open, so a
// queue that waits for its sink fails the test instead of hanging the suite.
// The destructor opens the gate and waits for the call in progress: a logger
// thread that stop() gave up on may still be inside write().
struct SinkGate {
    std::mutex mu;
    std::condition_variable cv;
    bool is_open = true;                // guarded by mu
    std::vector<std::string> received;  // guarded by mu
    std::atomic<int> waiting{0};        // calls blocked at the closed gate
    std::atomic<int> inside{0};         // calls in progress

    ~SinkGate() {
        release();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (inside.load() != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    void write(const std::string& text) {
        ++inside;
        {
            std::unique_lock<std::mutex> lock(mu);
            received.push_back(text);
            if (!is_open) {
                ++waiting;
                if (!cv.wait_for(lock, std::chrono::seconds(10), [this] { return is_open; })) {
                    is_open = true;
                }
                --waiting;
            }
        }
        --inside;  // last access to the gate
    }
    void shut() {
        std::lock_guard<std::mutex> lock(mu);
        is_open = false;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mu);
            is_open = true;
        }
        cv.notify_all();
    }
    std::vector<std::string> lines() {
        std::lock_guard<std::mutex> lock(mu);
        return received;
    }
    LogQueue::Sink sink() {
        return [this](const std::string& text) { write(text); };
    }
};

std::string test_drop_line(std::uint64_t dropped, bool final_report) {
    return "dropped " + std::to_string(dropped) + (final_report ? " final" : " live");
}

bool is_drop_line(const std::string& text) { return text.rfind("dropped ", 0) == 0; }

}  // namespace

TEST_CASE("log queue: push never waits for a blocked sink, and every line is written or counted as dropped") {
    constexpr std::size_t kCapacity = 16;
    constexpr std::size_t kLines = 10 * kCapacity;
    SinkGate gate;
    gate.shut();
    LogQueue q(gate.sink(), kCapacity, test_drop_line);

    std::vector<std::string> expected;  // the lines the queue accepted, in order
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kLines; ++i) {
        std::string text = "line " + std::to_string(i);
        if (q.push(text)) expected.push_back(std::move(text));
    }
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
    // A push that called the sink, or waited for it, would sit at the gate
    // for 10 s. The bound is generous so a loaded machine does not trip it.
    CHECK_MESSAGE(took < std::chrono::milliseconds(1000), "the pushes took " << took.count() << " ms");

    // The queue holds `capacity` lines; the logger thread may have taken one
    // batch (at most `capacity` more) out before it blocked in the sink.
    const std::size_t accepted = expected.size();
    CHECK(accepted >= kCapacity);
    CHECK(accepted <= 2 * kCapacity);
    const std::uint64_t refused = static_cast<std::uint64_t>(kLines - accepted);
    LogQueueStats s = q.stats();
    CHECK(s.capacity == kCapacity);
    CHECK(s.queued <= kCapacity);
    CHECK(s.dropped == refused);
    CHECK(s.written == 0);

    gate.release();
    CHECK(q.stop(std::chrono::seconds(10)));
    s = q.stats();
    CHECK(s.written == static_cast<std::uint64_t>(accepted));
    CHECK(s.dropped == refused);
    CHECK(s.queued == 0);

    // The sink got the accepted lines in push order, and one report with the
    // number of dropped lines (not counted as written).
    std::vector<std::string> written_lines;
    std::vector<std::string> reports;
    for (const auto& text : gate.lines()) {
        if (is_drop_line(text)) {
            reports.push_back(text);
        } else {
            written_lines.push_back(text);
        }
    }
    CHECK(written_lines == expected);
    REQUIRE(reports.size() == 1);
    CHECK(reports[0].rfind("dropped " + std::to_string(refused) + " ", 0) == 0);
}

TEST_CASE("log queue: the sink gets the lines in push order") {
    SinkGate gate;
    LogQueue q(gate.sink(), 256, test_drop_line);
    std::vector<std::string> expected;
    for (int i = 0; i < 200; ++i) {
        expected.push_back("line " + std::to_string(i));
        REQUIRE(q.push(expected.back()));
    }
    CHECK(q.stop(std::chrono::seconds(10)));
    CHECK(gate.lines() == expected);
    const LogQueueStats s = q.stats();
    CHECK(s.written == 200);
    CHECK(s.dropped == 0);
    CHECK(s.queued == 0);
    CHECK(s.capacity == 256);

    // Stopped: a later line is refused and counted, and stop() keeps its answer.
    CHECK_FALSE(q.push("late"));
    CHECK(q.stats().dropped == 1);
    CHECK(q.stop(std::chrono::milliseconds(0)));
    CHECK(gate.lines() == expected);
}

TEST_CASE("log queue: concurrent producers keep their own order and every line is accounted for") {
    constexpr int kThreads = 4;
    constexpr int kPerThread = 500;
    SinkGate gate;
    LogQueue q(gate.sink(), 64, test_drop_line);
    std::atomic<std::uint64_t> accepted{0};
    std::vector<std::thread> producers;
    for (int t = 0; t < kThreads; ++t) {
        producers.emplace_back([&q, &accepted, t] {
            for (int i = 0; i < kPerThread; ++i) {
                if (q.push(std::to_string(t) + ":" + std::to_string(i))) ++accepted;
            }
        });
    }
    for (auto& p : producers) p.join();
    CHECK(q.stop(std::chrono::seconds(10)));

    // Drops are allowed here (the queue is small on purpose); losing a line
    // without counting it is not.
    const LogQueueStats s = q.stats();
    CHECK(s.written == accepted.load());
    CHECK(s.written + s.dropped == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(s.queued == 0);

    std::vector<int> last(static_cast<std::size_t>(kThreads), -1);
    std::uint64_t seen = 0;
    bool ordered = true;
    for (const auto& text : gate.lines()) {
        if (is_drop_line(text)) continue;
        const std::size_t colon = text.find(':');
        REQUIRE(colon != std::string::npos);
        const std::size_t producer = static_cast<std::size_t>(std::stoi(text.substr(0, colon)));
        const int index = std::stoi(text.substr(colon + 1));
        REQUIRE(producer < last.size());
        ordered = ordered && index > last[producer];
        last[producer] = index;
        ++seen;
    }
    CHECK(ordered);
    CHECK(seen == s.written);
}

TEST_CASE("log queue: stop() gives up on a blocked sink after its patience") {
    SinkGate gate;
    gate.shut();
    LogQueue q(gate.sink(), 8, test_drop_line);
    REQUIRE(q.push("first"));
    REQUIRE(server_test::eventually([&] { return gate.waiting.load() == 1; }));
    // The logger thread is inside the sink with "first"; these wait behind it.
    REQUIRE(q.push("second"));
    REQUIRE(q.push("third"));
    CHECK(q.stats().queued == 2);

    const auto t0 = std::chrono::steady_clock::now();
    const bool drained = q.stop(std::chrono::milliseconds(100));
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
    CHECK_FALSE(drained);
    // It waited for its patience, and not for the sink (the gate's own
    // backstop is 10 s).
    CHECK_MESSAGE(took >= std::chrono::milliseconds(90), "stop() took " << took.count() << " ms");
    CHECK_MESSAGE(took < std::chrono::milliseconds(2000), "stop() took " << took.count() << " ms");

    // What never reached the sink is counted as dropped; nothing is queued.
    const LogQueueStats s = q.stats();
    CHECK(s.written == 0);
    CHECK(s.dropped == 2);
    CHECK(s.queued == 0);
    CHECK_FALSE(q.push("late"));
    CHECK(q.stats().dropped == 3);
    CHECK_FALSE(q.stop(std::chrono::milliseconds(0)));  // same answer, no second wait

    // The call that was in progress finishes on its own, and the sink is not
    // entered again.
    gate.release();
    CHECK(server_test::eventually([&] { return gate.inside.load() == 0 && q.stats().written == 1; }));
    CHECK(gate.lines() == std::vector<std::string>{"first"});
    CHECK(q.stats().dropped == 3);
}

TEST_CASE("log queue: stop() queues the final drop report even when the queue is full") {
    constexpr std::size_t kCapacity = 4;
    SinkGate gate;
    gate.shut();
    std::atomic<bool> final_built{false};
    LogQueue q(gate.sink(), kCapacity, [&final_built](std::uint64_t dropped, bool final_report) {
        if (final_report) final_built.store(true);
        return test_drop_line(dropped, final_report);
    });
    std::vector<std::string> expected{"first"};
    REQUIRE(q.push("first"));
    REQUIRE(server_test::eventually([&] { return gate.waiting.load() == 1; }));
    // The logger thread is blocked in the sink with "first": the next
    // `capacity` lines fill the queue and the rest are dropped. No live
    // report can be queued, because the logger thread is not coming back.
    std::uint64_t refused = 0;
    for (int i = 0; i < 20; ++i) {
        std::string text = "line " + std::to_string(i);
        if (q.push(text)) {
            expected.push_back(std::move(text));
        } else {
            ++refused;
        }
    }
    CHECK(refused == 16);
    CHECK(q.stats().queued == kCapacity);
    CHECK(q.stats().dropped == refused);

    // stop() adds the report to the full queue, then waits for the drain.
    std::future<bool> drained = std::async(std::launch::async, [&q] { return q.stop(std::chrono::seconds(10)); });
    const bool built = server_test::eventually([&] { return final_built.load(); });
    gate.release();  // before any assertion that can throw: stop() must be able to finish
    CHECK(built);
    REQUIRE(drained.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
    CHECK(drained.get());

    // Everything accepted was written, in order, and the report came last.
    expected.push_back(test_drop_line(refused, true));
    CHECK(gate.lines() == expected);
    const LogQueueStats s = q.stats();
    CHECK(s.written == 5);
    CHECK(s.dropped == refused);
    CHECK(s.queued == 0);
}

TEST_CASE("log queue: a sink that throws loses only that line") {
    std::mutex mu;
    std::vector<std::string> got;  // guarded by mu
    const auto snapshot = [&mu, &got] {
        std::lock_guard<std::mutex> lock(mu);
        return got;
    };
    LogQueue q(
        [&mu, &got](const std::string& text) {
            if (text == "bad") throw std::runtime_error("sink failed");
            std::lock_guard<std::mutex> lock(mu);
            got.push_back(text);
        },
        16, test_drop_line);
    REQUIRE(q.push("one"));
    REQUIRE(q.push("bad"));
    REQUIRE(q.push("two"));
    // The line the sink threw on is counted as dropped, and the line behind
    // it still arrives.
    REQUIRE(server_test::eventually([&] {
        const LogQueueStats now_stats = q.stats();
        return now_stats.written == 2 && now_stats.dropped == 1;
    }));
    // So does a later one, and the drop is reported like any other.
    REQUIRE(q.push("three"));
    CHECK(server_test::eventually([&] {
        const std::vector<std::string> seen = snapshot();
        return q.stats().written == 3 && std::count(seen.begin(), seen.end(), "dropped 1 live") == 1;
    }));

    // stop() is not held up by the failure.
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(q.stop(std::chrono::seconds(10)));
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
    CHECK_MESSAGE(took < std::chrono::milliseconds(5000), "stop() took " << took.count() << " ms");
    const LogQueueStats s = q.stats();
    CHECK(s.written == 3);
    CHECK(s.dropped == 1);
    CHECK(s.queued == 0);
    std::vector<std::string> delivered;
    int reports = 0;
    for (const auto& text : snapshot()) {
        if (is_drop_line(text)) {
            ++reports;
        } else {
            delivered.push_back(text);
        }
    }
    CHECK(delivered == std::vector<std::string>{"one", "two", "three"});
    CHECK(reports == 1);
}

TEST_CASE("log queue: without a sink every line is counted as dropped and stop() has nothing to wait for") {
    LogQueue q(LogQueue::Sink{}, 8, test_drop_line);
    CHECK_FALSE(q.push("lost"));
    CHECK_FALSE(q.push("lost too"));
    // True means that no logger thread is left and that stop() itself threw
    // nothing away; the lines push() refused are in the count.
    CHECK(q.stop(std::chrono::seconds(10)));
    const LogQueueStats s = q.stats();
    CHECK(s.written == 0);
    CHECK(s.dropped == 2);
    CHECK(s.queued == 0);
    CHECK(s.capacity == 8);
}

namespace {

// Puts one queue (capacity 2) through two rounds of drops with the sink
// gated, so that the logger thread's batches are known, then stops it and
// returns every line the sink received. Round 1 drops 2 lines, round 2 drops
// 3 more.
std::vector<std::string> two_rounds_of_drops(std::chrono::milliseconds report_interval) {
    SinkGate gate;
    gate.shut();
    LogQueue q(gate.sink(), 2, test_drop_line, report_interval);

    // Round 1: "a" is in the sink, "b" and "c" fill the queue, "d" and "e"
    // are dropped.
    REQUIRE(q.push("a"));
    REQUIRE(server_test::eventually([&] { return gate.waiting.load() == 1; }));
    REQUIRE(q.push("b"));
    REQUIRE(q.push("c"));
    CHECK_FALSE(q.push("d"));
    CHECK_FALSE(q.push("e"));
    gate.release();
    // The first drop is reported at once: the report is queued when the
    // batch in progress ("a") is done, behind "b" and "c". Wait until that
    // fourth sink call has returned.
    REQUIRE(server_test::eventually([&] { return gate.lines().size() == 4 && gate.inside.load() == 0; }));

    // Round 2: "f" is in the sink, "g" and "h" fill the queue, "i", "j" and
    // "k" are dropped.
    gate.shut();
    REQUIRE(q.push("f"));
    REQUIRE(server_test::eventually([&] { return gate.waiting.load() == 1; }));
    REQUIRE(q.push("g"));
    REQUIRE(q.push("h"));
    CHECK_FALSE(q.push("i"));
    CHECK_FALSE(q.push("j"));
    CHECK_FALSE(q.push("k"));
    gate.release();
    REQUIRE(server_test::eventually([&] { return q.stats().written == 6; }));
    // One more line. The logger thread decides about a report after each
    // batch and queues it before it takes the next one, so a report for
    // round 2 is ahead of "z" in the queue: once "z" is written, a second
    // live report is in the sink if there is ever going to be one.
    REQUIRE(q.push("z"));
    REQUIRE(server_test::eventually([&] { return q.stats().written == 7; }));

    CHECK(q.stop(std::chrono::seconds(10)));
    const LogQueueStats s = q.stats();
    CHECK(s.written == 7);
    CHECK(s.dropped == 5);
    return gate.lines();
}

}  // namespace

TEST_CASE("log queue: drop reports are rate limited and stop() reports what is left") {
    // An interval the test never reaches: after the first report no second
    // live one appears, however many more lines are dropped, and the final
    // report at stop() carries the total (5), of which 3 were unreported.
    const std::vector<std::string> got = two_rounds_of_drops(std::chrono::hours(1));
    const std::vector<std::string> expected{"a", "b", "c", "dropped 2 live", "f", "g", "h", "z", "dropped 5 final"};
    CHECK(got == expected);
}

TEST_CASE("log queue: with no report interval every round of drops is reported live") {
    // The same rounds with the limit off: the second round is reported live
    // (so the case above passes because of the interval, not by accident),
    // and stop() has nothing left to report.
    const std::vector<std::string> got = two_rounds_of_drops(std::chrono::milliseconds(0));
    const std::vector<std::string> expected{"a", "b", "c", "dropped 2 live", "f", "g", "h", "dropped 5 live", "z"};
    CHECK(got == expected);
}

TEST_CASE("auth: fixed-size SHA-256 constant-time token comparison (L-05)") {
    CHECK(constant_time_equals("", ""));
    CHECK(constant_time_equals("sonder-secret-token", "sonder-secret-token"));
    CHECK_FALSE(constant_time_equals("sonder-secret-token", "sonder-secret-tokeN"));
    CHECK_FALSE(constant_time_equals("short", "much-longer-secret-token"));
    CHECK_FALSE(constant_time_equals("prefix", "prefix-extended"));

    const auto digest = sha256_digest("bearer-token-value-123");
    CHECK(constant_time_equals_digest("bearer-token-value-123", digest));
    CHECK_FALSE(constant_time_equals_digest("bearer-token-value-124", digest));
    CHECK_FALSE(constant_time_equals_digest("", digest));
    CHECK_FALSE(constant_time_equals_digest("bearer-token-value-123-extra", digest));
}

TEST_CASE("openai: ingress JSON parser rejects high-node-count array bombs (M-17)") {
    std::string bomb = "{\"model\":\"m\",\"messages\":[";
    for (int i = 0; i < 70000; ++i) {
        if (i > 0) bomb.push_back(',');
        bomb.push_back('0');
    }
    bomb += "]}";
    const auto parsed = parse_chat_request(bomb);
    const auto* err = std::get_if<ApiError>(&parsed);
    REQUIRE(err != nullptr);
    CHECK(err->status == 400);
    CHECK(err->code == "invalid_json");
}

namespace {

class SyntheticGgufBuilder {
public:
    void kv_str(std::string_view key, std::string_view value) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 8);
        append_str(kv_bytes_, value);
        ++kv_count_;
    }
    void kv_u32(std::string_view key, std::uint32_t value) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 4);
        append_u32(kv_bytes_, value);
        ++kv_count_;
    }
    void kv_bool(std::string_view key, bool value) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 7);
        kv_bytes_.push_back(value ? static_cast<char>(1) : static_cast<char>(0));
        ++kv_count_;
    }
    void kv_str_array(std::string_view key, const std::vector<std::string>& values) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 9);
        append_u32(kv_bytes_, 8);
        append_u64(kv_bytes_, values.size());
        for (const std::string& item : values) {
            append_str(kv_bytes_, item);
        }
        ++kv_count_;
    }
    void kv_i32_array(std::string_view key, const std::vector<std::int32_t>& values) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 9);
        append_u32(kv_bytes_, 5);
        append_u64(kv_bytes_, values.size());
        for (const std::int32_t item : values) {
            append_u32(kv_bytes_, static_cast<std::uint32_t>(item));
        }
        ++kv_count_;
    }
    void kv_f32_array(std::string_view key, const std::vector<float>& values) {
        append_str(kv_bytes_, key);
        append_u32(kv_bytes_, 9);
        append_u32(kv_bytes_, 6);
        append_u64(kv_bytes_, values.size());
        for (const float item : values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &item, sizeof(bits));
            append_u32(kv_bytes_, bits);
        }
        ++kv_count_;
    }
    void tensor(std::string_view name, const std::vector<std::uint64_t>& dims, std::uint32_t ggml_type) {
        append_str(tensor_bytes_, name);
        append_u32(tensor_bytes_, static_cast<std::uint32_t>(dims.size()));
        for (const std::uint64_t dim : dims) {
            append_u64(tensor_bytes_, dim);
        }
        append_u32(tensor_bytes_, ggml_type);
        append_u64(tensor_bytes_, 0);
        ++tensor_count_;
    }
    std::string finish(std::string_view trailing_payload = {}) const {
        std::string out = "GGUF";
        append_u32(out, 3);
        append_u64(out, tensor_count_);
        append_u64(out, kv_count_);
        out.append(kv_bytes_);
        out.append(tensor_bytes_);
        out.append(trailing_payload);
        return out;
    }

private:
    static void append_u32(std::string& dst, std::uint32_t v) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            dst.push_back(static_cast<char>((v >> shift) & 0xFFu));
        }
    }
    static void append_u64(std::string& dst, std::uint64_t v) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            dst.push_back(static_cast<char>((v >> shift) & 0xFFu));
        }
    }
    static void append_str(std::string& dst, std::string_view s) {
        append_u64(dst, s.size());
        dst.append(s);
    }

    std::string kv_bytes_;
    std::string tensor_bytes_;
    std::uint64_t kv_count_ = 0;
    std::uint64_t tensor_count_ = 0;
};

struct TempGgufFile {
    std::filesystem::path path;

    explicit TempGgufFile(const std::string& content)
        : path(std::filesystem::temp_directory_path() / (make_id("sonder-gguf-id") + ".gguf")) {
        write(content);
    }
    ~TempGgufFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempGgufFile(const TempGgufFile&) = delete;
    TempGgufFile& operator=(const TempGgufFile&) = delete;

    void write(const std::string& content) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out.good());
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        REQUIRE(out.good());
    }
    std::string path_string() const { return path_to_utf8(path); }
};

struct TempGgufDir {
    std::filesystem::path dir;

    TempGgufDir() : dir(std::filesystem::temp_directory_path() / make_id("sonder-gguf-dir")) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        REQUIRE_FALSE(ec);
    }
    ~TempGgufDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    TempGgufDir(const TempGgufDir&) = delete;
    TempGgufDir& operator=(const TempGgufDir&) = delete;

    std::string write_file(std::string_view utf8_filename, const std::string& content) const {
        const std::filesystem::path file_path = dir / path_from_utf8(utf8_filename);
        std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
        REQUIRE(out.good());
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        REQUIRE(out.good());
        return path_to_utf8(file_path);
    }
    std::string dir_string() const { return path_to_utf8(dir); }
};

class StubLlamaCppBackendModel final : public BackendModel {
public:
    explicit StubLlamaCppBackendModel(ModelDescriptor descriptor) : descriptor_(std::move(descriptor)) {}
    [[nodiscard]] const ModelDescriptor& descriptor() const override { return descriptor_; }
    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        (void)request;
        (void)cancel;
        (void)on_chunk;
        return Status(ErrorCode::unsupported, "stub backend model");
    }

private:
    ModelDescriptor descriptor_;
};

std::string build_complete_synthetic_gguf(const std::vector<std::string>& tokens, std::string_view chat_template,
                                          std::string_view payload, bool reverse_tokenizer_order = false) {
    SyntheticGgufBuilder b;
    b.kv_str("general.architecture", "llama");
    b.kv_u32("llama.block_count", 2);
    b.kv_u32("llama.context_length", 8192);
    b.kv_u32("general.file_type", 15);  // Q4_K_M
    if (!reverse_tokenizer_order) {
        b.kv_str("tokenizer.ggml.model", "llama");
        b.kv_str("tokenizer.ggml.pre", "default");
        b.kv_str_array("tokenizer.ggml.tokens", tokens);
        b.kv_str_array("tokenizer.ggml.merges", {"h e", "he llo"});
        b.kv_f32_array("tokenizer.ggml.scores", {0.0f, -1.0f, -2.0f, 0.0f});
        b.kv_i32_array("tokenizer.ggml.token_type", {3, 1, 1, 3});
        b.kv_u32("tokenizer.ggml.bos_token_id", 0);
        b.kv_u32("tokenizer.ggml.eos_token_id", 3);
        b.kv_bool("tokenizer.ggml.add_bos_token", true);
    } else {
        b.kv_bool("tokenizer.ggml.add_bos_token", true);
        b.kv_u32("tokenizer.ggml.eos_token_id", 3);
        b.kv_u32("tokenizer.ggml.bos_token_id", 0);
        b.kv_i32_array("tokenizer.ggml.token_type", {3, 1, 1, 3});
        b.kv_f32_array("tokenizer.ggml.scores", {0.0f, -1.0f, -2.0f, 0.0f});
        b.kv_str_array("tokenizer.ggml.merges", {"h e", "he llo"});
        b.kv_str_array("tokenizer.ggml.tokens", tokens);
        b.kv_str("tokenizer.ggml.pre", "default");
        b.kv_str("tokenizer.ggml.model", "llama");
    }
    if (!chat_template.empty()) {
        b.kv_str("tokenizer.chat_template", chat_template);
    }
    b.tensor("blk.0.attn_q.weight", {64, 64}, 12);  // Q4_K
    return b.finish(payload);
}

}  // namespace

TEST_CASE("identity: llamacpp GGUF produces all nine keys with real SHA-256 digests and cache invalidation") {
    const std::string chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
    const std::vector<std::string> tokens_a = {"<s>", "hello", "world", "</s>"};
    const std::string bytes_a = build_complete_synthetic_gguf(tokens_a, chat_template, "synthetic-weights-v1");
    TempGgufFile file(bytes_a);

    const IdentityResult id =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string("llama.cpp b11195"));
    REQUIRE_MESSAGE(id.backend_identity.is_object(), id.reason.value_or(""));
    CHECK_FALSE(id.reason.has_value());
    const auto& o = id.backend_identity.as_object();
    CHECK(o.size() == 9);
    for (const char* key : {"backend", "model", "model_digest", "quantization", "backend_version", "tokenizer_digest",
                            "template_digest", "context_tokens", "hardware"}) {
        CHECK_MESSAGE(o.contains(key), key);
    }
    CHECK(o.find("backend")->as_string() == "llamacpp");
    CHECK(o.find("model")->as_string() == "tiny-llama");
    CHECK(o.find("model_digest")->as_string() == sha256_hex(bytes_a));
    CHECK(o.find("quantization")->as_string() == "Q4_K_M");
    CHECK(o.find("backend_version")->as_string() == "llama.cpp b11195");
    CHECK(o.find("template_digest")->as_string() == sha256_hex(chat_template));
    const std::string tok_digest_a = o.find("tokenizer_digest")->as_string();
    CHECK(tok_digest_a.size() == 64);
    CHECK(tok_digest_a.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(o.find("context_tokens")->as_int() == 4096);
    CHECK_FALSE(o.find("hardware")->as_string().empty());

    // Deterministic across repeated calls (cache hit).
    const IdentityResult id_again =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string("llama.cpp b11195"));
    REQUIRE(id_again.backend_identity.is_object());
    CHECK(id_again.backend_identity.dump() == id.backend_identity.dump());

    // Zero runtime context falls back to <arch>.context_length from the GGUF.
    const IdentityResult id_train_ctx =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 0, std::string("llama.cpp b11195"));
    REQUIRE(id_train_ctx.backend_identity.is_object());
    CHECK(id_train_ctx.backend_identity.find("context_tokens")->as_int() == 8192);

    // backend_identity resolves GGUF path from either model_id or descriptor->name.
    ModelDescriptor desc;
    desc.name = "tiny-llama.gguf";
    desc.backend = "llamacpp";
    desc.format = "gguf";
    desc.context_length = 4096;
    const IdentityResult via_model_id =
        backend_identity("llamacpp", file.path_string(), &desc, std::string("llama.cpp b11195"));
    REQUIRE_MESSAGE(via_model_id.backend_identity.is_object(), via_model_id.reason.value_or(""));
    CHECK(via_model_id.backend_identity.find("model_digest")->as_string() == sha256_hex(bytes_a));
    CHECK(via_model_id.backend_identity.find("tokenizer_digest")->as_string() == tok_digest_a);

    desc.name = file.path_string();
    const IdentityResult via_desc_name =
        backend_identity("llamacpp", "served-alias", &desc, std::string("llama.cpp b11195"));
    REQUIRE_MESSAGE(via_desc_name.backend_identity.is_object(), via_desc_name.reason.value_or(""));
    CHECK(via_desc_name.backend_identity.find("model")->as_string() == "served-alias");
    CHECK(via_desc_name.backend_identity.find("model_digest")->as_string() == sha256_hex(bytes_a));

    // Tokenizer digest is canonical across KV order and sensitive to vocab elements.
    const std::string bytes_reordered =
        build_complete_synthetic_gguf(tokens_a, chat_template, "synthetic-weights-v1", true);
    const auto parsed_reordered = parse_gguf_model_info(bytes_reordered);
    REQUIRE(parsed_reordered.ok());
    CHECK(parsed_reordered.value().tokenizer_digest == tok_digest_a);
    CHECK(parsed_reordered.value().template_digest == sha256_hex(chat_template));
    CHECK(parsed_reordered.value().training_context_length == 8192);
    for (const auto& [meta_k, meta_v] : parsed_reordered.value().metadata) {
        (void)meta_v;
        CHECK(meta_k != "tokenizer.ggml.token_type");
    }

    const std::string bytes_diff_vocab =
        build_complete_synthetic_gguf({"<s>", "hello", "sonder", "</s>"}, chat_template, "synthetic-weights-v1");
    const auto parsed_diff_vocab = parse_gguf_model_info(bytes_diff_vocab);
    REQUIRE(parsed_diff_vocab.ok());
    CHECK(parsed_diff_vocab.value().tokenizer_digest.size() == 64);
    CHECK(parsed_diff_vocab.value().tokenizer_digest != tok_digest_a);

    // Cache invalidation on size change.
    const std::string bytes_b =
        build_complete_synthetic_gguf(tokens_a, chat_template, "synthetic-weights-v2-extended");
    REQUIRE(bytes_b.size() != bytes_a.size());
    file.write(bytes_b);
    const IdentityResult id_b =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string("llama.cpp b11195"));
    REQUIRE(id_b.backend_identity.is_object());
    CHECK(id_b.backend_identity.find("model_digest")->as_string() == sha256_hex(bytes_b));
    CHECK(id_b.backend_identity.find("model_digest")->as_string() != sha256_hex(bytes_a));

    // Cache invalidation on same-size content rewrite with updated mtime.
    const std::string bytes_c =
        build_complete_synthetic_gguf(tokens_a, chat_template, "synthetic-weights-v3-extended");
    REQUIRE(bytes_c.size() == bytes_b.size());
    std::error_code ec;
    const auto prev_mtime = std::filesystem::last_write_time(file.path, ec);
    REQUIRE_FALSE(ec);
    file.write(bytes_c);
    std::filesystem::last_write_time(file.path, prev_mtime + std::chrono::seconds(5), ec);
    REQUIRE_FALSE(ec);
    const IdentityResult id_c =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string("llama.cpp b11195"));
    REQUIRE(id_c.backend_identity.is_object());
    CHECK(id_c.backend_identity.find("model_digest")->as_string() == sha256_hex(bytes_c));
    CHECK(id_c.backend_identity.find("model_digest")->as_string() != sha256_hex(bytes_b));
}

TEST_CASE("identity: llamacpp GGUF missing chat template or tokenizer honestly returns null") {
    // Missing tokenizer.chat_template must never fabricate template_digest.
    const std::string no_template = build_complete_synthetic_gguf({"<s>", "hi", "there", "</s>"}, "", "payload");
    TempGgufFile file_no_tpl(no_template);
    const auto info_no_tpl = read_gguf_model_info(file_no_tpl.path_string());
    REQUIRE(info_no_tpl.ok());
    CHECK(info_no_tpl.value().tokenizer_digest.size() == 64);
    CHECK(info_no_tpl.value().template_digest.empty());

    const IdentityResult res_no_tpl =
        gguf_backend_identity("llamacpp", "no-tpl", file_no_tpl.path_string(), 4096, std::string("llama.cpp"));
    CHECK(res_no_tpl.backend_identity.is_null());
    REQUIRE(res_no_tpl.reason.has_value());
    CHECK(res_no_tpl.reason->find("tokenizer.chat_template missing in GGUF") != std::string::npos);

    // Missing all tokenizer.ggml.* metadata must never fabricate tokenizer_digest.
    SyntheticGgufBuilder b;
    b.kv_str("general.architecture", "llama");
    b.kv_u32("llama.block_count", 1);
    b.kv_u32("llama.context_length", 4096);
    b.kv_u32("general.file_type", 7);  // Q8_0
    b.kv_str("tokenizer.chat_template", "{{ prompt }}");
    b.tensor("blk.0.attn_q.weight", {32, 32}, 8);
    TempGgufFile file_no_tok(b.finish("payload"));
    const auto info_no_tok = read_gguf_model_info(file_no_tok.path_string());
    REQUIRE(info_no_tok.ok());
    CHECK(info_no_tok.value().tokenizer_digest.empty());
    CHECK(info_no_tok.value().template_digest == sha256_hex("{{ prompt }}"));

    const IdentityResult res_no_tok =
        gguf_backend_identity("llamacpp", "no-tok", file_no_tok.path_string(), 4096, std::string("llama.cpp"));
    CHECK(res_no_tok.backend_identity.is_null());
    REQUIRE(res_no_tok.reason.has_value());
    CHECK(res_no_tok.reason->find("tokenizer") != std::string::npos);

    // Lazy residency (null descriptor) reports an honest null reason.
    const IdentityResult lazy = backend_identity("llamacpp", "unloaded.gguf", nullptr, std::nullopt);
    CHECK(lazy.backend_identity.is_null());
    REQUIRE(lazy.reason.has_value());
    CHECK(lazy.reason->find("lazy") != std::string::npos);
}

TEST_CASE("identity: llamacpp missing or non-regular GGUF path returns null with clear reason") {
    const std::string missing_path =
        path_to_utf8(std::filesystem::temp_directory_path() / (make_id("sonder-missing-gguf") + ".gguf"));
    const IdentityResult missing_res =
        gguf_backend_identity("llamacpp", "missing", missing_path, 4096, std::string("llama.cpp b11195"));
    CHECK(missing_res.backend_identity.is_null());
    REQUIRE(missing_res.reason.has_value());
    CHECK_FALSE(missing_res.reason->empty());

    std::string hex;
    std::string err;
    CHECK_FALSE(compute_gguf_file_sha256(missing_path, hex, err));
    CHECK(hex.empty());
    CHECK_FALSE(err.empty());

    const IdentityResult empty_res =
        gguf_backend_identity("llamacpp", "empty", "", 4096, std::string("llama.cpp b11195"));
    CHECK(empty_res.backend_identity.is_null());
    REQUIRE(empty_res.reason.has_value());
    CHECK_FALSE(compute_gguf_file_sha256("", hex, err));
    CHECK_FALSE(err.empty());

    const std::string dir_path = path_to_utf8(std::filesystem::temp_directory_path());
    const IdentityResult dir_res =
        gguf_backend_identity("llamacpp", "dir", dir_path, 4096, std::string("llama.cpp b11195"));
    CHECK(dir_res.backend_identity.is_null());
    REQUIRE(dir_res.reason.has_value());
    CHECK(dir_res.reason->find("regular") != std::string::npos);
    CHECK_FALSE(compute_gguf_file_sha256(dir_path, hex, err));
    CHECK(err.find("regular") != std::string::npos);
}

TEST_CASE("gguf header: quantization mapping, dominant tensor fallback and multi-chunk file hashing") {
    // Without general.file_type, quantization falls back to the largest-footprint tensor type.
    SyntheticGgufBuilder fallback_builder;
    fallback_builder.kv_str("general.architecture", "llama");
    fallback_builder.kv_u32("llama.block_count", 1);
    fallback_builder.tensor("blk.0.attn_norm.weight", {64}, 0);         // F32: 256 B
    fallback_builder.tensor("blk.0.ffn_gate.weight", {128, 32}, 142);   // PQ2_0: 32 * 34 = 1088 B
    const auto fallback_info = parse_gguf_model_info(fallback_builder.finish());
    REQUIRE(fallback_info.ok());
    CHECK(fallback_info.value().quantization == "PQ2_0");

    // Explicit general.file_type takes precedence, including 0 (F32) and ternary/1-bit values.
    const std::vector<std::pair<std::uint32_t, std::string>> ftypes = {
        {0, "F32"}, {1, "F16"}, {15, "Q4_K_M"}, {25, "IQ4_NL"}, {36, "TQ1_0"}, {37, "TQ2_0"}, {40, "Q1_0"},
    };
    for (const auto& [ftype, expected_name] : ftypes) {
        CAPTURE(ftype);
        SyntheticGgufBuilder b;
        b.kv_str("general.architecture", "llama");
        b.kv_u32("llama.block_count", 1);
        b.kv_u32("general.file_type", ftype);
        b.tensor("blk.0.attn_q.weight", {32, 32}, 8);  // Q8_0 tensor overridden by general.file_type
        const auto info = parse_gguf_model_info(b.finish());
        REQUIRE(info.ok());
        CHECK(info.value().quantization == expected_name);
    }

    // Streaming file SHA-256 across multiple 64 KiB chunks matches in-memory SHA-256 exactly.
    std::string big_payload(150000, '\0');
    for (std::size_t idx = 0; idx < big_payload.size(); ++idx) {
        big_payload[idx] = static_cast<char>((idx * 37u + 13u) & 0xFFu);
    }
    TempGgufFile big_file(big_payload);
    std::string streamed_hex;
    std::string hash_err;
    REQUIRE(compute_gguf_file_sha256(big_file.path_string(), streamed_hex, hash_err));
    CHECK(hash_err.empty());
    CHECK(streamed_hex == sha256_hex(big_payload));
}

TEST_CASE("identity: llamacpp GGUF rejects missing or empty backend version honestly") {
    const std::string bytes =
        build_complete_synthetic_gguf({"<s>", "hello", "world", "</s>"}, "{{ prompt }}", "ver-check");
    TempGgufFile file(bytes);

    const IdentityResult null_ver =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::nullopt);
    CHECK(null_ver.backend_identity.is_null());
    REQUIRE(null_ver.reason.has_value());
    CHECK(null_ver.reason->find("backend version") != std::string::npos);

    const IdentityResult empty_ver =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string(""));
    CHECK(empty_ver.backend_identity.is_null());
    REQUIRE(empty_ver.reason.has_value());
    CHECK(empty_ver.reason->find("backend version") != std::string::npos);

    // Caller-supplied cached GPU inventory is formatted into `hardware`.
    GpuDeviceInfo fake_gpu;
    fake_gpu.id = "gpu:0";
    fake_gpu.name = "Synthetic Accelerator";
    fake_gpu.dedicated_total = MemoryReading::known(16384ull << 20);
    const std::vector<GpuDeviceInfo> gpus{fake_gpu};
    const IdentityResult with_gpu =
        gguf_backend_identity("llamacpp", "tiny-llama", file.path_string(), 4096, std::string("llama.cpp b11195"),
                              &gpus);
    REQUIRE_MESSAGE(with_gpu.backend_identity.is_object(), with_gpu.reason.value_or(""));
    const std::string hw = with_gpu.backend_identity.find("hardware")->as_string();
    CHECK(hw.find("gpu:0 Synthetic Accelerator (16384 MiB)") != std::string::npos);
}

TEST_CASE("identity: build_identity_body resolves llamacpp GGUF via model_dirs and profile bindings") {
    TempGgufDir models_dir;
    const std::string chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}";
    const std::string dir_bytes =
        build_complete_synthetic_gguf({"<s>", "dir", "model", "</s>"}, chat_template, "dir-payload");
    const std::string profile_bytes =
        build_complete_synthetic_gguf({"<s>", "profile", "model", "</s>"}, chat_template, "profile-payload-v2");
    (void)models_dir.write_file("dir-model.gguf", dir_bytes);
    const std::string profile_full_path = models_dir.write_file("profile-model.gguf", profile_bytes);

    auto residency = std::make_shared<ModelResidency>(
        ResidencyConfig{},
        [](const std::string& id) -> Result<std::shared_ptr<Model>> {
            ModelDescriptor d;
            d.backend = "llamacpp";
            d.format = "gguf";
            d.context_length = 4096;
            // Match LlamaCppCoreBackend::load_model(), which stores only the filename in descriptor().name.
            if (id == "dir-model.gguf") {
                d.name = "dir-model.gguf";
            } else if (id == "profile-alias") {
                d.name = "profile-model.gguf";
            } else {
                d.name = "dir-model.gguf";
            }
            auto impl = std::make_shared<StubLlamaCppBackendModel>(std::move(d));
            return std::make_shared<Model>("inst-" + id, "llamacpp", "cpu:0", std::move(impl));
        },
        [](Eviction&) {});

    REQUIRE(residency->add("dir-model.gguf", "llamacpp", true));
    REQUIRE(residency->add("profile-alias", "llamacpp", false));
    REQUIRE(residency->add("stale-alias", "llamacpp", false));

    auto pin_dir = residency->acquire("dir-model.gguf");
    REQUIRE(pin_dir.ok());
    auto pin_profile = residency->acquire("profile-alias");
    REQUIRE(pin_profile.ok());
    auto pin_stale = residency->acquire("stale-alias");
    REQUIRE(pin_stale.ok());

    server::ServerOptions opts;
    opts.backend.backend = "llamacpp";
    opts.backend.model_dirs = {models_dir.dir_string()};

    server::ModelProfileBinding profile_binding;
    profile_binding.model = "profile-alias";
    profile_binding.load_name = profile_full_path;
    opts.profile_bindings.push_back(std::move(profile_binding));

    server::ModelProfileBinding stale_binding;
    stale_binding.model = "stale-alias";
    stale_binding.load_name =
        path_to_utf8(std::filesystem::temp_directory_path() / (make_id("sonder-absent") + ".gguf"));
    opts.profile_bindings.push_back(std::move(stale_binding));

    // Case 1: basename served_id resolved via opts.backend.model_dirs (NOT in CWD).
    const json::Object dir_body =
        build_identity_body(opts, *residency, "dir-model.gguf", false, std::string("llama.cpp b11195"));
    const json::Value* dir_id = dir_body.find("backend_identity");
    REQUIRE(dir_id != nullptr);
    REQUIRE(dir_id->is_object());
    CHECK(dir_body.find("reason")->is_null());
    CHECK(dir_id->find("model")->as_string() == "dir-model.gguf");
    CHECK(dir_id->find("model_digest")->as_string() == sha256_hex(dir_bytes));
    CHECK(dir_id->find("template_digest")->as_string() == sha256_hex(chat_template));

    // Case 2: profile alias resolved via opts.profile_bindings[i].load_name.
    const json::Object profile_body =
        build_identity_body(opts, *residency, "profile-alias", false, std::string("llama.cpp b11195"));
    const json::Value* profile_id = profile_body.find("backend_identity");
    REQUIRE(profile_id != nullptr);
    REQUIRE(profile_id->is_object());
    CHECK(profile_body.find("reason")->is_null());
    CHECK(profile_id->find("model")->as_string() == "profile-alias");
    CHECK(profile_id->find("model_digest")->as_string() == sha256_hex(profile_bytes));

    // Case 3: missing profile load_name must NOT fall back to descriptor->name in model_dirs or CWD.
    const json::Object stale_body =
        build_identity_body(opts, *residency, "stale-alias", false, std::string("llama.cpp b11195"));
    const json::Value* stale_id = stale_body.find("backend_identity");
    REQUIRE(stale_id != nullptr);
    CHECK(stale_id->is_null());
    REQUIRE(stale_body.find("reason") != nullptr);
    CHECK(stale_body.find("reason")->is_string());
    CHECK_FALSE(stale_body.find("reason")->as_string().empty());
}

TEST_CASE("identity: llamacpp GGUF handles non-ASCII UTF-8 filenames cleanly") {
    TempGgufDir utf8_dir;
    const std::string utf8_name = "sonder-caf\xC3\xA9-\xE6\xA8\xA1\xE5\x9E\x8B.gguf";
    const std::string chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}";
    std::vector<std::string> vocab;
    vocab.reserve(4);
    vocab.emplace_back("<s>");
    vocab.emplace_back("utf8");
    vocab.emplace_back("path");
    vocab.emplace_back("</s>");
    const std::string bytes = build_complete_synthetic_gguf(vocab, chat_template, "utf8-payload");
    const std::string utf8_path = utf8_dir.write_file(utf8_name, bytes);

    const auto parsed = read_gguf_model_info(utf8_path);
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    CHECK(parsed.value().template_digest == sha256_hex(chat_template));
    CHECK(parsed.value().tokenizer_digest.size() == 64);

    std::string hex;
    std::string err;
    REQUIRE_MESSAGE(compute_gguf_file_sha256(utf8_path, hex, err), err);
    CHECK(hex == sha256_hex(bytes));

    // Passing empty model_id also exercises UTF-8 filename extraction without ANSI narrowing.
    const IdentityResult id =
        gguf_backend_identity("llamacpp", "", utf8_path, 4096, std::string("llama.cpp b11195"));
    REQUIRE_MESSAGE(id.backend_identity.is_object(), id.reason.value_or(""));
    CHECK(id.backend_identity.find("model")->as_string() == utf8_name);
    CHECK(id.backend_identity.find("model_digest")->as_string() == sha256_hex(bytes));
    CHECK(id.backend_identity.find("template_digest")->as_string() == sha256_hex(chat_template));
    CHECK(id.backend_identity.find("tokenizer_digest")->as_string() == parsed.value().tokenizer_digest);
}

TEST_CASE("tool_call_parser: incremental byte-by-byte XML and fenced blocks, fallback, and GBNF") {
    // 1. Byte-by-byte feed across <tool_call> and </tool_call> never leaks markup into visible_text.
    {
        const std::string raw =
            "Before <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"NYC\"}}</tool_call> after";
        ToolCallStreamParser parser("call_t_");
        std::string visible;
        std::vector<ParsedToolEvent> events;
        for (const char ch : raw) {
            const auto out = parser.feed(std::string_view(&ch, 1));
            visible += out.visible_text;
            for (const auto& ev : out.tool_calls) events.push_back(ev);
        }
        const auto tail = parser.flush();
        visible += tail.visible_text;
        for (const auto& ev : tail.tool_calls) events.push_back(ev);

        CHECK(visible == "Before  after");
        REQUIRE(events.size() == 1u);
        CHECK(events[0].index == 0u);
        CHECK(events[0].id == "call_t_0");
        CHECK(events[0].name == "get_weather");
        CHECK(events[0].arguments_json == R"({"city":"NYC"})");
    }

    // 2. Fenced ```tool_call blocks + multiple sequential calls
    {
        const std::string raw =
            "```tool_call\n{\"name\":\"a\",\"arguments\":{\"k\":1}}\n```\n"
            "<tool_call>{\"name\":\"b\",\"arguments\":{\"k\":2}}</tool_call>";
        const auto ext = extract_tool_calls(raw, "c_");
        REQUIRE(ext.tool_calls.size() == 2u);
        CHECK(ext.tool_calls[0].name == "a");
        CHECK(ext.tool_calls[0].id == "c_0");
        CHECK(ext.tool_calls[1].name == "b");
        CHECK(ext.tool_calls[1].id == "c_1");
        CHECK(ext.visible_text.find("tool_call") == std::string::npos);
    }

    // 3. Invalid JSON inside <tool_call> falls back to visible text safely (never drops output).
    {
        const auto ext = extract_tool_calls("<tool_call>not json</tool_call>");
        CHECK(ext.tool_calls.empty());
        CHECK(ext.visible_text == "<tool_call>not json</tool_call>");
    }

    // 4. build_tool_call_gbnf for required and named function
    {
        const std::vector<ToolDefinition> tools = {
            {"alpha", "desc a", R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false})"},
            {"beta", "desc b", R"({"type":"object","properties":{"s":{"type":"string"}},"additionalProperties":false})"},
        };
        const auto g_req = build_tool_call_gbnf(tools, ToolChoice{ToolChoiceMode::required, ""});
        REQUIRE_MESSAGE(g_req.ok(), g_req.status().message());
        CHECK(g_req.value().find("<tool_call>") != std::string::npos);
        CHECK(g_req.value().find("alpha") != std::string::npos);
        CHECK(g_req.value().find("beta") != std::string::npos);

        const auto g_fn = build_tool_call_gbnf(tools, ToolChoice{ToolChoiceMode::function, "beta"});
        REQUIRE_MESSAGE(g_fn.ok(), g_fn.status().message());
        CHECK(g_fn.value().find("beta") != std::string::npos);
        CHECK(g_fn.value().find("alpha") == std::string::npos);
    }
}

TEST_CASE("multimodal base64: strict RFC 4648 decode, size limit, and SSRF URI rejection") {
    const std::vector<std::uint8_t> raw = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    const std::string b64 = encode_base64(raw);
    const auto uri_ok = parse_data_image_uri("data:image/png;base64," + b64, "url");
    REQUIRE(std::holds_alternative<MediaContent>(uri_ok));
    CHECK(std::get<MediaContent>(uri_ok).mime_type == "image/png");
    CHECK(std::get<MediaContent>(uri_ok).bytes == raw);

    // Enforces max decoded size cap
    const auto too_big = decode_strict_base64(b64, 4, "img");
    REQUIRE(std::holds_alternative<ApiError>(too_big));
    CHECK(std::get<ApiError>(too_big).code == "invalid_request");

    // Rejects non-canonical padding bits and whitespace
    CHECK(std::holds_alternative<ApiError>(decode_strict_base64("AB==", 16, "img")));
    CHECK(std::holds_alternative<ApiError>(decode_strict_base64("AA == ", 16, "img")));

    // Rejects remote & file URLs strictly
    for (const char* url : {"http://example.com/a.png", "https://example.com/a.png", "file:///etc/passwd",
                            "data:text/plain;base64,SGVsbG8="}) {
        CAPTURE(url);
        const auto res = parse_data_image_uri(url, "url");
        REQUIRE(std::holds_alternative<ApiError>(res));
        CHECK(std::get<ApiError>(res).status == 400);
        CHECK(std::get<ApiError>(res).code == "invalid_request");
    }
}

TEST_CASE("responses_handler: LRU history store eviction and request parsing") {
    ResponseHistoryStore store(3);
    store.put({"r1", "sk1", "m", {{"user", "1"}}});
    store.put({"r2", "sk2", "m", {{"user", "2"}}});
    store.put({"r3", "sk3", "m", {{"user", "3"}}});
    CHECK(store.size() == 3u);
    // Touch r1 so r2 becomes LRU victim.
    REQUIRE(store.get("r1").has_value());
    store.put({"r4", "sk4", "m", {{"user", "4"}}});
    CHECK(store.size() == 3u);
    CHECK(store.get("r1").has_value());
    CHECK_FALSE(store.get("r2").has_value());
    CHECK(store.get("r3").has_value());
    CHECK(store.get("r4").has_value());

    // Valid responses request parsing (flat tool + text.format + reasoning + input array)
    const auto parsed = parse_responses_request(
        R"({"model":"m","instructions":"sys","previous_response_id":"resp_1",
            "input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
            "tools":[{"type":"function","name":"fn1","parameters":{"type":"object"}}],
            "tool_choice":{"type":"function","name":"fn1"},
            "text":{"format":{"type":"json_object"}},
            "reasoning":{"effort":"medium"},
            "max_output_tokens":32,"temperature":0.2,"stream":true})");
    REQUIRE(std::holds_alternative<ResponsesJob>(parsed));
    const auto& job = std::get<ResponsesJob>(parsed);
    CHECK(job.model == "m");
    CHECK(job.instructions == std::optional<std::string>("sys"));
    CHECK(job.previous_response_id == std::optional<std::string>("resp_1"));
    REQUIRE(job.tools.size() == 1u);
    CHECK(job.tools[0].name == "fn1");
    REQUIRE(job.tool_choice.has_value());
    CHECK(job.tool_choice->mode == ToolChoiceMode::function);
    CHECK(job.tool_choice->function_name == "fn1");
    CHECK(job.response_format_type == std::optional<std::string>("json_object"));
    CHECK(job.sampling.max_tokens == 32);
    CHECK(job.thinking.reasoning_effort == std::optional<std::string>("medium"));
    CHECK(job.stream);

    // SamplingConfig validation for logprobs / top_logprobs
    SamplingConfig sc;
    sc.logprobs = false;
    sc.top_logprobs = 2;
    CHECK_FALSE(validate(sc).ok());
    sc.logprobs = true;
    sc.top_logprobs = 20;
    CHECK(validate(sc).ok());
    sc.top_logprobs = 21;
    CHECK_FALSE(validate(sc).ok());
}
