// End-to-end and unit tests for Wave 1 Track 2 HTTP server features:
//   - POST /v1/embeddings (mock 200 + unsupported backend 501 + validation)
//   - POST /v1/completions (non-streaming, echo, SSE streaming, grammar)
//   - POST /v1/tokenize and POST /v1/detokenize
//   - response_format (text, json_object, json_schema -> GBNF) and grammar on chat
//   - Bounded HTTP/1.1 Connection: keep-alive (multi-request reuse, pipelining,
//     max_keep_alive_requests, idle timeout, and close on SSE)
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "server_test_support.hpp"
#include "src/json_schema_gbnf.hpp"
#include "src/openai.hpp"

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#  include "backends/llamaserver/llamaserver_protocol.hpp"
#endif

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

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

// Minimal backend that does NOT implement embed/tokenize/detokenize, and
// captures the grammar/response_format forwarded on chat() and generate().
struct CapturingBackend final : si::Backend {
    struct Model final : si::BackendModel {
        si::ModelDescriptor desc;
        std::mutex mu;
        std::string last_grammar;
        std::optional<std::string> last_rf_type;
        std::optional<std::string> last_rf_schema;
        std::string last_prompt;

        Model() {
            desc.name = "capture:test";
            desc.backend = "capture-test";
            desc.context_length = 4096;
        }
        const si::ModelDescriptor& descriptor() const override { return desc; }
        bool has_native_chat() const override { return true; }

        si::Result<si::GenerateStats> generate(const si::GenerateRequest& req, const si::CancellationToken&,
                                               const si::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                last_grammar = req.grammar;
                last_prompt = req.prompt;
            }
            if (on_chunk) {
                (void)on_chunk(si::TokenChunk{"hello", 0});
                (void)on_chunk(si::TokenChunk{" world", 1});
            }
            si::GenerateStats st;
            st.prompt_tokens = 3;
            st.completion_tokens = 2;
            st.chunks = 2;
            st.token_counts_from_backend = true;
            st.stop_reason = si::StopReason::end_of_sequence;
            return st;
        }

        si::Result<si::GenerateStats> chat(const si::ChatRequest& req, const si::CancellationToken&,
                                           const si::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                last_grammar = req.grammar;
                last_rf_type = req.response_format_type;
                last_rf_schema = req.response_format_schema;
            }
            if (on_chunk) {
                (void)on_chunk(si::TokenChunk{R"({"ok":true})", 0});
            }
            si::GenerateStats st;
            st.prompt_tokens = 4;
            st.completion_tokens = 3;
            st.chunks = 1;
            st.token_counts_from_backend = true;
            st.stop_reason = si::StopReason::end_of_sequence;
            return st;
        }
    };

    std::shared_ptr<Model> model = std::make_shared<Model>();
    std::string name() const override { return "capture-test"; }
    std::string description() const override { return "capturing test backend without embeddings/tokenization"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming);
    }
    si::Result<std::string> probe() override { return std::string("capture-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

srv::ServerOptions capturing_options(const std::shared_ptr<CapturingBackend>& backend) {
    auto o = Fixture::defaults();
    o.models = {"capture:test"};
    o.backend.backend.clear();
    o.backend_instance = backend;
    return o;
}

}  // namespace

// ================================================================= GBNF unit tests

TEST_CASE("json_schema_gbnf: json_object_gbnf and primitive/object/array/enum/ref schemas") {
    const std::string obj_gbnf = det::json_object_gbnf();
    CHECK(obj_gbnf.find("root ::= object") != std::string::npos);
    CHECK(obj_gbnf.find("string ::=") != std::string::npos);
    CHECK(obj_gbnf.find("number ::=") != std::string::npos);

    // Object with required + optional properties + nested array + enum + $ref
    const auto schema_v = json::parse(R"({
        "type": "object",
        "$defs": {
            "Tag": {"type": "string", "minLength": 1, "maxLength": 16}
        },
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "status": {"enum": ["ok", "fail"]},
            "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}, "minItems": 1, "maxItems": 3},
            "note": {"type": ["string", "null"]}
        },
        "required": ["name", "status"],
        "additionalProperties": false
    })");
    REQUIRE(schema_v.ok());
    const auto gbnf = det::json_schema_to_gbnf(schema_v.value());
    REQUIRE_MESSAGE(gbnf.ok(), gbnf.status().message());
    CHECK(gbnf.value().find("root ::= ") != std::string::npos);
    CHECK(gbnf.value().find("\\\"name\\\"") != std::string::npos);
    CHECK(gbnf.value().find("\\\"status\\\"") != std::string::npos);
    CHECK(gbnf.value().find("\\\"ok\\\"") != std::string::npos);
    CHECK(gbnf.value().find("\\\"fail\\\"") != std::string::npos);
    CHECK(gbnf.value().find("char{1,16}") != std::string::npos);

    // All-optional object produces comma-safe optional grammar.
    const auto opt_schema = json::parse(R"({
        "type": "object",
        "properties": {"a": {"type": "boolean"}, "b": {"type": "number"}},
        "additionalProperties": false
    })");
    REQUIRE(opt_schema.ok());
    const auto opt_gbnf = det::json_schema_to_gbnf(opt_schema.value());
    REQUIRE_MESSAGE(opt_gbnf.ok(), opt_gbnf.status().message());
    CHECK(opt_gbnf.value().find("root ::= ") != std::string::npos);

    // Const & anyOf
    const auto comb_schema = json::parse(R"({
        "anyOf": [{"const": "yes"}, {"type": "integer"}]
    })");
    REQUIRE(comb_schema.ok());
    CHECK(det::json_schema_to_gbnf(comb_schema.value()).ok());

    // Invalid schemas are rejected.
    CHECK_FALSE(det::json_schema_to_gbnf(json::Value(false)).ok());
    const auto bad_type = json::parse(R"({"type": "widget"})");
    REQUIRE(bad_type.ok());
    CHECK_FALSE(det::json_schema_to_gbnf(bad_type.value()).ok());
    const auto bad_ref = json::parse(R"({"$ref": "#/$defs/Missing"})");
    REQUIRE(bad_ref.ok());
    CHECK_FALSE(det::json_schema_to_gbnf(bad_ref.value()).ok());
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
TEST_CASE("llamaserver protocol: forwards native response_format and per-request grammar") {
    si::ChatRequest chat_req;
    chat_req.messages = {{"user", "hi"}};
    chat_req.response_format_type = "json_object";
    chat_req.grammar = "root ::= object";
    {
        const auto body = si::llamaserver::build_chat_body("m", chat_req);
        const auto* rf = body.find("response_format");
        REQUIRE(rf != nullptr);
        CHECK(rf->find("type")->as_string() == "json_object");
        CHECK(body.find("grammar") == nullptr);
    }
    chat_req.response_format_type = "json_schema";
    chat_req.response_format_schema = R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})";
    {
        const auto body = si::llamaserver::build_chat_body("m", chat_req);
        const auto* rf = body.find("response_format");
        REQUIRE(rf != nullptr);
        CHECK(rf->find("type")->as_string() == "json_schema");
        REQUIRE(rf->find("json_schema") != nullptr);
        CHECK(rf->find("json_schema")->find("schema")->find("type")->as_string() == "object");
        CHECK(body.find("grammar") == nullptr);
    }
    chat_req.response_format_type.reset();
    chat_req.response_format_schema.reset();
    chat_req.grammar = "root ::= \"yes\"";
    {
        const auto body = si::llamaserver::build_chat_body("m", chat_req);
        CHECK(body.find("response_format") == nullptr);
        REQUIRE(body.find("grammar") != nullptr);
        CHECK(body.find("grammar")->as_string() == "root ::= \"yes\"");
    }

    si::GenerateRequest gen_req;
    gen_req.prompt = "hello";
    gen_req.grammar = "root ::= \"ok\"";
    {
        const auto body = si::llamaserver::build_completion_body("m", gen_req, true);
        REQUIRE(body.find("grammar") != nullptr);
        CHECK(body.find("grammar")->as_string() == "root ::= \"ok\"");
    }
}
#endif

// ================================================================= /v1/embeddings

TEST_CASE("embeddings: single input, batch array, validation and 501 on unsupported backend") {
    Fixture f;
    // Single string input
    {
        const Reply r = post(f.port, "/v1/embeddings", R"({"input":"hello world","encoding_format":"float"})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("object")->as_string() == "list");
        CHECK(doc.find("model")->as_string() == "mock:tiny");
        const auto& data = doc.find("data")->as_array();
        REQUIRE(data.size() == 1u);
        CHECK(data[0].find("object")->as_string() == "embedding");
        CHECK(data[0].find("index")->as_int() == 0);
        CHECK(data[0].find("embedding")->as_array().size() == 16u);
        CHECK(doc.find("usage")->find("prompt_tokens")->as_uint() > 0u);
        CHECK(doc.find("usage")->find("total_tokens")->as_uint() ==
              doc.find("usage")->find("prompt_tokens")->as_uint());
        CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);
        CHECK(doc.find("sonder")->find("backend")->as_string() == "mock");
        CHECK(doc.find("sonder")->find("synthetic")->as_bool());
        CHECK_FALSE(doc.find("sonder")->find("request_id")->as_string().empty());
    }
    // Batch array of strings
    {
        const Reply r = post(f.port, "/v1/embeddings", R"({"model":"mock:tiny","input":["alpha","beta","gamma"]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& data = doc.find("data")->as_array();
        REQUIRE(data.size() == 3u);
        CHECK(data[0].find("index")->as_int() == 0);
        CHECK(data[1].find("index")->as_int() == 1);
        CHECK(data[2].find("index")->as_int() == 2);
        CHECK(data[0].find("embedding")->dump() != data[1].find("embedding")->dump());
    }
    // Validation errors
    CHECK(post(f.port, "/v1/embeddings", R"({"input":[]})").status == 400);
    CHECK(post(f.port, "/v1/embeddings", R"({"input":[1,2]})").status == 400);
    CHECK(post(f.port, "/v1/embeddings", R"({"input":42})").status == 400);
    {
        const Reply bad_fmt = post(f.port, "/v1/embeddings", R"({"input":"x","encoding_format":"base64"})");
        CHECK(bad_fmt.status == 400);
        CHECK(bad_fmt.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }
    {
        // > 256 items rejected
        std::string many = R"({"input":[)";
        for (int i = 0; i < 257; ++i) {
            if (i > 0) many += ',';
            many += "\"x\"";
        }
        many += "]}";
        const Reply r = post(f.port, "/v1/embeddings", many);
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "invalid_request");
    }
    // Backend returning ErrorCode::unsupported maps to 501 not_implemented
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const Reply r = post(cap.port, "/v1/embeddings", R"({"input":"hello"})");
        CHECK(r.status == 501);
        CHECK(r.json().find("error")->find("code")->as_string() == "not_implemented");
    }
}

// ================================================================= /v1/completions

TEST_CASE("completions: non-streaming, echo, SSE streaming and grammar forwarding") {
    Fixture f;
    // Non-streaming
    {
        const Reply r = post(f.port, "/v1/completions", R"({"model":"default","prompt":"once upon","max_tokens":5,"seed":7})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("id")->as_string().rfind("cmpl-req-", 0) == 0);
        CHECK(doc.find("object")->as_string() == "text_completion");
        CHECK(doc.find("model")->as_string() == "mock:tiny");
        const auto& choices = doc.find("choices")->as_array();
        REQUIRE(choices.size() == 1u);
        CHECK(choices[0].find("index")->as_int() == 0);
        CHECK(choices[0].find("logprobs")->is_null());
        CHECK_FALSE(choices[0].find("text")->as_string().empty());
        CHECK(choices[0].find("finish_reason")->as_string() == "length");
        CHECK(doc.find("usage")->find("completion_tokens")->as_int() == 5);
        CHECK(doc.find("timings")->find("predicted_n")->as_int() == 5);
        CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);
    }
    // Echo
    {
        const Reply r = post(f.port, "/v1/completions", R"({"prompt":"PREFIX:","max_tokens":3,"echo":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const std::string text = r.json().find("choices")->as_array()[0].find("text")->as_string();
        CHECK(text.rfind("PREFIX:", 0) == 0);
        CHECK(text.size() > 7u);
    }
    // SSE streaming with include_usage
    {
        const Reply r = post(
            f.port, "/v1/completions",
            R"({"prompt":"stream me","max_tokens":4,"stream":true,"stream_options":{"include_usage":true},"echo":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.header("content-type") == "text/event-stream; charset=utf-8");
        const auto events = sse_data(r.body);
        REQUIRE(events.size() >= 3u);
        CHECK(events.back() == "[DONE]");
        auto first = json::parse(events.front());
        REQUIRE(first.ok());
        CHECK(first.value().find("object")->as_string() == "text_completion");
        CHECK(first.value().find("id")->as_string().rfind("cmpl-req-", 0) == 0);
        CHECK(first.value().find("choices")->as_array()[0].find("text")->as_string() == "stream me");
        auto last = json::parse(events[events.size() - 2]);
        REQUIRE(last.ok());
        CHECK(last.value().find("choices")->as_array()[0].find("finish_reason")->as_string() == "length");
        CHECK(last.value().find("usage")->find("completion_tokens")->as_int() == 4);
    }
    // Validation errors
    CHECK(post(f.port, "/v1/completions", R"({"prompt":""})").status == 400);
    CHECK(post(f.port, "/v1/completions", R"({"prompt":"hi","n":2})").status == 400);
    // Grammar forwarding on capturing backend
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const Reply r = post(cap.port, "/v1/completions", R"({"prompt":"say hi","grammar":"root ::= \"hello\""})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.json().find("choices")->as_array()[0].find("text")->as_string() == "hello world");
        std::lock_guard<std::mutex> lock(backend->model->mu);
        CHECK(backend->model->last_grammar == "root ::= \"hello\"");
    }
}

// ================================================================= /v1/tokenize & /v1/detokenize

TEST_CASE("tokenize and detokenize: roundtrip on mock, text alias, validation and 501") {
    Fixture f;
    // content field
    const Reply tok = post(f.port, "/v1/tokenize", R"({"content":"hello sonder engine"})");
    REQUIRE_MESSAGE(tok.status == 200, tok.body);
    const auto tok_doc = tok.json();
    CHECK(tok_doc.find("model")->as_string() == "mock:tiny");
    const auto& tokens = tok_doc.find("tokens")->as_array();
    REQUIRE_FALSE(tokens.empty());
    CHECK(tok_doc.find("count")->as_uint() == tokens.size());
    CHECK(tok_doc.find("sonder")->find("api_version")->as_int() == 1);

    // text alias
    const Reply tok_alias = post(f.port, "/v1/tokenize", R"({"text":"hello sonder engine"})");
    REQUIRE(tok_alias.status == 200);
    CHECK(tok_alias.json().find("tokens")->dump() == tok_doc.find("tokens")->dump());

    // detokenize
    const Reply detok = post(f.port, "/v1/detokenize", R"({"tokens":)" + tok_doc.find("tokens")->dump() + "}");
    REQUIRE_MESSAGE(detok.status == 200, detok.body);
    CHECK(detok.json().find("model")->as_string() == "mock:tiny");
    CHECK_FALSE(detok.json().find("content")->as_string().empty());

    // Validation
    CHECK(post(f.port, "/v1/tokenize", R"({})").status == 400);
    CHECK(post(f.port, "/v1/detokenize", R"({"tokens":"bad"})").status == 400);
    CHECK(post(f.port, "/v1/detokenize", R"({"tokens":["not-int"]})").status == 400);

    // Unsupported backend returns 501
    auto backend = std::make_shared<CapturingBackend>();
    Fixture cap(capturing_options(backend));
    CHECK(post(cap.port, "/v1/tokenize", R"({"content":"hi"})").status == 501);
    CHECK(post(cap.port, "/v1/detokenize", R"({"tokens":[1,2]})").status == 501);
}

// ================================================================= response_format on chat

TEST_CASE("chat response_format: text, json_object, json_schema and grammar wiring") {
    auto backend = std::make_shared<CapturingBackend>();
    Fixture f(capturing_options(backend));

    // json_object
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"return json"}],"response_format":{"type":"json_object"}})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        std::lock_guard<std::mutex> lock(backend->model->mu);
        CHECK(backend->model->last_rf_type == std::optional<std::string>("json_object"));
        CHECK_FALSE(backend->model->last_rf_schema.has_value());
        CHECK(backend->model->last_grammar.find("root ::= object") != std::string::npos);
    }
    // json_schema
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"return person"}],"response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}}}})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        std::lock_guard<std::mutex> lock(backend->model->mu);
        CHECK(backend->model->last_rf_type == std::optional<std::string>("json_schema"));
        REQUIRE(backend->model->last_rf_schema.has_value());
        CHECK(backend->model->last_rf_schema->find("\"age\"") != std::string::npos);
        CHECK(backend->model->last_grammar.find("root ::= ") != std::string::npos);
    }
    // text + explicit grammar allowed
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"say ok"}],"response_format":{"type":"text"},"grammar":"root ::= \"ok\""})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        std::lock_guard<std::mutex> lock(backend->model->mu);
        CHECK_FALSE(backend->model->last_rf_type.has_value());
        CHECK(backend->model->last_grammar == "root ::= \"ok\"");
    }
    // grammar combined with json_object rejected
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"conflict"}],"response_format":{"type":"json_object"},"grammar":"root ::= \"x\""})");
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "invalid_request");
    }
}

// ================================================================= HTTP/1.1 Keep-Alive

TEST_CASE("http keep-alive: reuses one TCP socket across sequential and pipelined requests") {
    auto o = Fixture::defaults();
    o.keep_alive = true;
    o.keep_alive_timeout = std::chrono::milliseconds(5000);
    o.max_keep_alive_requests = 4;
    Fixture f(o);

    Conn c(f.port);
    std::string buf;

    // 1 & 2. Pipelined: send GET health + POST tokenize in one write!
    const std::string req1 = build_request("GET", "/v1/sonder/health", f.port, {{"Connection", "keep-alive"}});
    const std::string req2 =
        build_request("POST", "/v1/tokenize", f.port, {{"Connection", "keep-alive"}}, R"({"content":"hello"})");
    c.send(req1 + req2);

    const Reply r1 = c.read_response(buf);
    REQUIRE(r1.status == 200);
    CHECK(r1.complete);
    CHECK(r1.header("connection") == "keep-alive");
    CHECK(r1.header("keep-alive") == "timeout=5, max=3");

    const Reply r2 = c.read_response(buf);
    REQUIRE(r2.status == 200);
    CHECK(r2.header("connection") == "keep-alive");
    CHECK(r2.header("keep-alive") == "timeout=5, max=2");

    // 3. Sequential POST /v1/embeddings on the same socket.
    c.send(build_request("POST", "/v1/embeddings", f.port, {{"Connection", "keep-alive"}}, R"({"input":"one"})"));
    const Reply r3 = c.read_response(buf);
    REQUIRE(r3.status == 200);
    CHECK(r3.header("connection") == "keep-alive");
    CHECK(r3.header("keep-alive") == "timeout=5, max=1");

    // 4. Sequential POST /v1/chat/completions hits max_keep_alive_requests=4 -> Connection: close.
    c.send(build_request("POST", "/v1/chat/completions", f.port, {{"Connection", "keep-alive"}},
                         chat_body(R"(,"max_tokens":2)")));
    const Reply r4 = c.read_response(buf);
    REQUIRE(r4.status == 200);
    CHECK(r4.header("connection") == "close");
    CHECK_FALSE(r4.has_header("keep-alive"));
    // Server closes the connection after reaching max_keep_alive_requests.
    CHECK(c.read_all(std::chrono::milliseconds(2000)).empty());
    CHECK(c.closed());
}

TEST_CASE("http keep-alive: idle timeout closes silently with no error and no spurious access log") {
    auto o = Fixture::defaults();
    o.keep_alive = true;
    o.keep_alive_timeout = std::chrono::milliseconds(200);
    std::mutex log_mu;
    std::vector<std::string> logs;
    o.log = [&](const std::string& line) {
        std::lock_guard<std::mutex> lock(log_mu);
        logs.push_back(line);
    };
    Fixture f(o);

    Conn c(f.port);
    std::string buf;
    c.send(build_request("GET", "/v1/sonder/health", f.port, {{"Connection", "keep-alive"}}));
    const Reply r1 = c.read_response(buf);
    REQUIRE(r1.status == 200);
    CHECK(r1.header("connection") == "keep-alive");

    // Stay idle: server should close cleanly after keep_alive_timeout with 0 bytes.
    const std::string trailing = c.read_all(std::chrono::milliseconds(3000));
    CHECK(trailing.empty());
    CHECK(c.closed());

    f.server->stop();
    std::lock_guard<std::mutex> lock(log_mu);
    std::size_t health_logs = 0;
    std::size_t error_logs = 0;
    for (const auto& line : logs) {
        if (line.find("/v1/sonder/health") != std::string::npos) ++health_logs;
        if (line.find(" 408 ") != std::string::npos) ++error_logs;
    }
    CHECK(health_logs == 1u);
    CHECK(error_logs == 0u);
}

TEST_CASE("http keep-alive: streaming SSE forces Connection: close and keep_alive=false disables persistence") {
    {
        Fixture f;
        Conn c(f.port);
        c.send(build_request("POST", "/v1/chat/completions", f.port, {{"Connection", "keep-alive"}},
                             chat_body(R"(,"max_tokens":2,"stream":true)")));
        const Reply r = parse_reply(c.read_all(std::chrono::milliseconds(5000)));
        REQUIRE(r.status == 200);
        CHECK(r.header("connection") == "close");
        CHECK(c.closed());
    }
    {
        auto o = Fixture::defaults();
        o.keep_alive = false;
        Fixture f(o);
        Conn c(f.port);
        c.send(build_request("GET", "/v1/sonder/health", f.port, {{"Connection", "keep-alive"}}));
        const Reply r = parse_reply(c.read_all(std::chrono::milliseconds(5000)));
        REQUIRE(r.status == 200);
        CHECK(r.header("connection") == "close");
        CHECK(c.closed());
    }
}
