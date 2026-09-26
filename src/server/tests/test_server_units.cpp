// Pure-function tests for the server module: SHA-256, the HTTP head parser,
// query decoding, OpenAI request mapping, correlation headers, error mapping,
// the live telemetry hub (ring, fan-out, resume), backend setup and identity.
#include <doctest/doctest.h>

#include <string>
#include <variant>

#include "server_test_support.hpp"
#include "src/http.hpp"
#include "src/identity.hpp"
#include "src/live_hub.hpp"
#include "src/openai.hpp"
#include "src/sha256.hpp"
#include "sonder/inference/backend_setup.hpp"

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

    const auto minimal = parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})");
    REQUIRE(std::holds_alternative<ChatJob>(minimal));
    CHECK(std::get<ChatJob>(minimal).model == "default");
    CHECK(std::get<ChatJob>(minimal).sampling.max_tokens == kDefaultMaxTokens);
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
    for (const char* key : {"tools", "tool_choice", "functions", "function_call", "response_format", "top_logprobs"}) {
        const ApiError e = error_of("{" + msgs + R"(,")" + key + R"(":[1]})");
        CHECK(e.status == 400);
        CHECK(e.code == "unsupported_parameter");
        CHECK(e.param == std::optional<std::string>(key));
    }
    CHECK(error_of("{" + msgs + R"(,"logprobs":true})").code == "unsupported_parameter");
    CHECK(error_of("{" + msgs + R"(,"n":2})").code == "unsupported_parameter");
    CHECK(error_of(R"({"messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]})").code ==
          "unsupported_parameter");
    CHECK(error_of(R"({"messages":[{"role":"assistant","content":null,"tool_calls":[]},{"role":"user","content":"x"}]})")
              .code == "unsupported_parameter");
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

    RequestHead none;
    const auto defaults = parse_correlation(none);
    REQUIRE(std::holds_alternative<Correlation>(defaults));
    CHECK(std::get<Correlation>(defaults).workload == WorkloadClass::interactive_user);

    for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
             {"x-sonder-run-id", "has space"},
             {"x-sonder-agent-id", std::string(129, 'a')},
             {"x-sonder-workload", "urgent"},
             {"x-sonder-priority", "17"},
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
}
