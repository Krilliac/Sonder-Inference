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
        return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::structured_output);
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

    // Bounds boundary: 32 accepted, 33 rejected (never silently dropped).
    {
        const auto s32 = json::parse(R"({"type":"string","minLength":32,"maxLength":32})");
        const auto s33 = json::parse(R"({"type":"string","minLength":33})");
        const auto a32 = json::parse(R"({"type":"array","items":{"type":"integer"},"minItems":32,"maxItems":32})");
        const auto a33 = json::parse(R"({"type":"array","items":{"type":"integer"},"maxItems":33})");
        REQUIRE((s32.ok() && s33.ok() && a32.ok() && a33.ok()));
        CHECK(det::json_schema_to_gbnf(s32.value()).ok());
        CHECK_FALSE(det::json_schema_to_gbnf(s33.value()).ok());
        CHECK(det::json_schema_to_gbnf(a32.value()).ok());
        CHECK_FALSE(det::json_schema_to_gbnf(a33.value()).ok());
    }

    // Upgraded structured grammar engine: disjoint oneOf, basic regex pattern,
    // bounded integer ranges, and StructuredGrammarEngine abstraction.
    {
        // Provably disjoint oneOf: disjoint types, disjoint literals, and tagged-union objects.
        for (const char* ok_json : {
                 R"({"oneOf":[{"type":"string"},{"type":"integer"}]})",
                 R"({"oneOf":[{"const":"a"},{"const":"b"},{"enum":[1,2]}]})",
                 R"({"oneOf":[{"type":"object","properties":{"kind":{"const":"cat"},"meows":{"type":"boolean"}},"required":["kind"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"dog"},"barks":{"type":"boolean"}},"required":["kind"],"additionalProperties":false}]})",
                 R"({"type":"string","pattern":"^[a-z0-9_-]{1,16}$"})",
                 R"({"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}$"})",
                 R"({"type":"integer","minimum":0,"maximum":10})",
                 R"({"type":"integer","minimum":0,"maximum":999})",
                 R"({"type":"integer","minimum":-5,"maximum":5})",
                 R"({"type":"integer","minimum":0})",
             }) {
            CAPTURE(ok_json);
            const auto v = json::parse(ok_json);
            REQUIRE(v.ok());
            const auto g = det::json_schema_to_gbnf(v.value());
            REQUIRE_MESSAGE(g.ok(), g.status().message());
            CHECK(g.value().find("root ::= ") != std::string::npos);
        }

        // Fail closed on overlapping oneOf, unsupported regex features, out-of-range/float bounds, and unsupported keywords.
        for (const char* bad_json : {
                 R"({"oneOf":[{"type":"string"},{"type":"string"}]})",
                 R"({"oneOf":[{"type":"integer"},{"type":"number"}]})",
                 R"({"oneOf":[{"const":1},{"const":1.0}]})",
                 R"({"oneOf":[{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false},{"type":"object","properties":{"b":{"type":"integer"}},"additionalProperties":false}]})",
                 R"({"type":"string","pattern":"^(?=foo)[a-z]+$"})",
                 R"({"type":"string","pattern":"^([a-z]+)\\1$"})",
                 R"({"type":"string","pattern":"^[a-z]+?$"})",
                 R"({"type":"string","pattern":"^[a-z]{0,33}$"})",
                 R"({"type":"string","pattern":"^[a-z]+$","minLength":2})",
                 R"({"type":"number","minimum":0,"maximum":10})",
                 R"({"type":"integer","minimum":0,"maximum":10000000})",
                 R"({"type":"integer","minimum":10,"maximum":5})",
                 R"({"allOf":[{"type":"object"}]})",
                 R"({"type":"array","uniqueItems":true})",
                 R"({"type":"object","properties":{"role":{"const":"admin"}},"additionalProperties":true})",
             }) {
            CAPTURE(bad_json);
            const auto v = json::parse(bad_json);
            REQUIRE(v.ok());
            CHECK_FALSE(det::json_schema_to_gbnf(v.value()).ok());
        }

        const auto& engine = det::default_structured_grammar_engine();
        CHECK_FALSE(engine.name().empty());
        const auto obj_compiled = engine.compile_json_object();
        REQUIRE(obj_compiled.ok());
        CHECK(obj_compiled.value().gbnf.find("root ::= object") != std::string::npos);

        const auto annotated = json::parse(R"({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Sample",
            "description": "Metadata annotations are permitted",
            "type": "object",
            "properties": {"ok": {"type": "boolean", "default": true}},
            "required": ["ok"],
            "additionalProperties": false
        })");
        REQUIRE(annotated.ok());
        CHECK(det::json_schema_to_gbnf(annotated.value()).ok());
    }

    // Required keys without `properties` must still enforce the required key (not shortcut to `{}` / `object`).
    {
        const auto req_only = json::parse(R"({"type":"object","required":["id"]})");
        REQUIRE(req_only.ok());
        const auto g = det::json_schema_to_gbnf(req_only.value());
        REQUIRE_MESSAGE(g.ok(), g.status().message());
        CHECK(g.value().find("\\\"id\\\"") != std::string::npos);
    }

    // Long property names + wide anyOf must stay O(1) in generated rule-name length.
    {
        std::string long_key(2000, 'a');
        std::string alts;
        for (int i = 0; i < 64; ++i) {
            if (i > 0) alts += ",";
            alts += R"({"const":)" + std::to_string(i) + "}";
        }
        const auto wide = json::parse(R"({"type":"object","properties":{")" + long_key + R"(":{"anyOf":[)" + alts +
                                      R"(]}},"required":[")" + long_key + R"("],"additionalProperties":false})");
        REQUIRE(wide.ok());
        const auto g = det::json_schema_to_gbnf(wide.value());
        REQUIRE_MESSAGE(g.ok(), g.status().message());
        CHECK(g.value().size() < 8192u);
    }
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
    // Backend without Capability::structured_output (e.g. mock) rejects constrained requests cleanly (400).
    {
        Fixture mock_fix;
        const Reply r_chat = post(
            mock_fix.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"}})");
        CHECK(r_chat.status == 400);
        CHECK(r_chat.json().find("error")->find("code")->as_string() == "unsupported_parameter");
        const Reply r_cmpl = post(
            mock_fix.port, "/v1/completions",
            R"({"prompt":"hi","grammar":"root ::= \"ok\""})");
        CHECK(r_cmpl.status == 400);
        CHECK(r_cmpl.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }
}

TEST_CASE("embeddings: respects priority admission capacity and recovers session on backend exception") {
    // 1. Capacity=1 admission test: an in-flight chat holds the sole permit;
    //    /v1/embeddings MUST wait until chat releases the permit (never bypass!).
    struct GatedEmbedBackend final : si::Backend {
        struct Model final : si::BackendModel {
            si::ModelDescriptor desc;
            std::atomic<bool> chat_entered{false};
            std::atomic<bool> release_chat{false};
            std::atomic<bool> embed_entered_while_chat_active{false};
            std::atomic<bool> throw_on_embed{false};
            std::atomic<bool> chat_active{false};

            Model() {
                desc.name = "gated:test";
                desc.backend = "gated-test";
                desc.context_length = 4096;
            }
            const si::ModelDescriptor& descriptor() const override { return desc; }
            bool has_native_chat() const override { return true; }

            si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                                   const si::TokenCallback&) override {
                return si::GenerateStats{};
            }
            si::Result<si::GenerateStats> chat(const si::ChatRequest&, const si::CancellationToken& cancel,
                                               const si::TokenCallback& on_chunk) override {
                chat_active.store(true);
                chat_entered.store(true);
                while (!release_chat.load() && !cancel.cancelled()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                chat_active.store(false);
                if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
                si::GenerateStats st;
                st.prompt_tokens = 2;
                st.completion_tokens = 1;
                st.chunks = 1;
                st.token_counts_from_backend = true;
                st.stop_reason = si::StopReason::end_of_sequence;
                return st;
            }
            si::Result<si::EmbedResult> embed(const si::EmbedRequest& req, const si::CancellationToken&) override {
                if (throw_on_embed.load()) {
                    throw std::runtime_error("simulated embed backend exception");
                }
                if (chat_active.load()) {
                    embed_entered_while_chat_active.store(true);
                }
                si::EmbedResult out;
                out.prompt_tokens = static_cast<std::uint64_t>(req.inputs.size());
                out.embeddings.assign(req.inputs.size(), std::vector<float>{0.5f, 0.5f});
                return out;
            }
        };

        std::shared_ptr<Model> model = std::make_shared<Model>();
        std::string name() const override { return "gated-test"; }
        std::string description() const override { return "gated admission test backend"; }
        si::BackendCapabilities capabilities() const override {
            return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::embeddings);
        }
        std::size_t max_concurrent_requests() const override { return 1; }
        si::Result<std::string> probe() override { return std::string("gated-1"); }
        si::Result<std::vector<si::ModelDescriptor>> list_models() override {
            return std::vector<si::ModelDescriptor>{model->descriptor()};
        }
        si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
            return std::static_pointer_cast<si::BackendModel>(model);
        }
    };

    auto be = std::make_shared<GatedEmbedBackend>();
    auto o = Fixture::defaults();
    o.models = {"gated:test"};
    o.backend.backend.clear();
    o.backend_instance = be;
    o.priority_admission = srv::PriorityAdmissionPolicy::on;
    o.backend_capacity = 1;
    Fixture f(o);

    Reply chat_reply;
    std::thread t_chat([&] {
        chat_reply = post(f.port, "/v1/chat/completions", R"({"messages":[{"role":"user","content":"hold"}]})");
    });
    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!be->model->chat_entered.load() && std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(be->model->chat_entered.load());

    Reply embed_reply;
    std::thread t_embed([&] { embed_reply = post(f.port, "/v1/embeddings", R"({"input":"queued"})"); });
    // Give /v1/embeddings time to queue behind the occupied permit.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_FALSE(be->model->embed_entered_while_chat_active.load());

    be->model->release_chat.store(true);
    t_chat.join();
    t_embed.join();
    REQUIRE_MESSAGE(chat_reply.status == 200, chat_reply.body);
    REQUIRE_MESSAGE(embed_reply.status == 200, embed_reply.body);
    CHECK_FALSE(be->model->embed_entered_while_chat_active.load());

    // 2. Session::embed() throwing backend must recover SessionState::idle (never wedge in running).
    si::EngineOptions eo;
    eo.scheduling.enabled = false;
    si::Engine engine(eo);
    REQUIRE(engine.register_backend(be).ok());
    auto loaded = engine.load_model("gated-test", si::ModelLoadOptions{"gated:test", "cpu:0"});
    REQUIRE(loaded.ok());
    auto sess = engine.create_session(loaded.value());
    REQUIRE(sess.ok());
    be->model->throw_on_embed.store(true);
    CHECK_THROWS_AS((void)sess.value()->embed({"boom"}), std::runtime_error);
    CHECK(sess.value()->state() == si::SessionState::idle);
    be->model->throw_on_embed.store(false);
    auto ok_after = sess.value()->embed({"recovered"});
    REQUIRE_MESSAGE(ok_after.ok(), ok_after.status().message());
    CHECK(ok_after.value().embeddings.size() == 1u);
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

TEST_CASE("socket peer_gone: preserves live pipelined byte and detects peer FIN") {
    bool in_use = false;
    auto listener = det::listen_tcp("127.0.0.1", 0, in_use);
    REQUIRE(listener.ok());
    const std::uint16_t port = listener.value().port;

    auto client = det::connect_tcp("127.0.0.1", port, std::chrono::milliseconds(2000));
    REQUIRE(client.ok());
    std::atomic<bool> stop{false};
    REQUIRE(det::wait_socket(listener.value().socket.get(), false, std::chrono::milliseconds(2000), &stop) ==
            det::WaitResult::ready);
    auto accepted = det::accept_client(listener.value().socket);
    REQUIRE(accepted.status == det::AcceptStatus::accepted);

    // Live connection with a buffered pipelined byte: peer_gone() must return false
    // and leave the byte intact in the kernel receive buffer.
    const std::string pipelined = "G";
    REQUIRE(det::send_all(client.value().get(), pipelined, std::chrono::milliseconds(1000), &stop).ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK_FALSE(det::peer_gone(accepted.socket.get()));

    // Peer sends FIN while that pipelined byte is still unread.
    client.value().shutdown_write();
    client.value().close();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
#if defined(__linux__)
    CHECK(det::peer_gone(accepted.socket.get()));
#endif
    // Drain the 1 pipelined byte; now peer_gone() MUST detect EOF on every OS.
    char ch = 0;
    CHECK(det::recv_some(accepted.socket.get(), &ch, 1) == 1);
    CHECK(ch == 'G');
    CHECK(det::peer_gone(accepted.socket.get()));
}

// ================================================================= Track 2 Frontier APIs

namespace {

// 1x1 transparent PNG encoded as base64 (valid binary PNG bytes).
constexpr const char* kTinyPngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

std::vector<std::pair<std::string, std::string>> parse_named_sse(const std::string& body) {
    std::vector<std::pair<std::string, std::string>> out;
    std::size_t pos = 0;
    while (pos < body.size()) {
        std::size_t end = body.find("\n\n", pos);
        const std::string block =
            end == std::string::npos ? body.substr(pos) : body.substr(pos, end - pos);
        pos = end == std::string::npos ? body.size() : end + 2;
        if (block.empty()) continue;
        std::string ev;
        std::string data;
        std::size_t line_pos = 0;
        while (line_pos < block.size()) {
            const std::size_t nl = block.find('\n', line_pos);
            std::string line =
                nl == std::string::npos ? block.substr(line_pos) : block.substr(line_pos, nl - line_pos);
            line_pos = nl == std::string::npos ? block.size() : nl + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("event: ", 0) == 0) {
                ev = line.substr(7);
            } else if (line.rfind("data: ", 0) == 0) {
                if (!data.empty()) data.push_back('\n');
                data += line.substr(6);
            }
        }
        if (!data.empty()) out.emplace_back(std::move(ev), std::move(data));
    }
    return out;
}

}  // namespace

TEST_CASE("tool calling: OpenAI non-streaming, streaming delta.tool_calls, tool_choice, and multi-turn") {
    Fixture f;
    const std::string tools_json = R"([
        {"type":"function","function":{"name":"get_weather","description":"Fetch weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"enum":["c","f"]}},"required":["city"],"additionalProperties":false}}},
        {"type":"function","function":{"name":"calc","description":"Calculate","parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}
    ])";

    // 1. Non-streaming tool_choice: "required"
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Check weather"}],"tools":)" + tools_json +
                R"(,"tool_choice":"required"})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        CHECK(choice.find("finish_reason")->as_string() == "tool_calls");
        const auto* msg = choice.find("message");
        REQUIRE(msg != nullptr);
        CHECK(msg->find("content")->is_null());
        const auto* tcs = msg->find("tool_calls");
        REQUIRE((tcs != nullptr && tcs->is_array() && tcs->as_array().size() == 1u));
        const auto& tc0 = tcs->as_array()[0];
        CHECK(tc0.find("type")->as_string() == "function");
        CHECK_FALSE(tc0.find("id")->as_string().empty());
        CHECK(tc0.find("function")->find("name")->as_string() == "get_weather");
        const auto parsed_args = json::parse(tc0.find("function")->find("arguments")->as_string());
        REQUIRE(parsed_args.ok());
        CHECK(parsed_args.value().find("city")->as_string() == "mock_city");
    }

    // 2. Named function tool_choice picks exact function ("calc")
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"anything"}],"tools":)" + tools_json +
                R"(,"tool_choice":{"type":"function","function":{"name":"calc"}}})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& tc0 = doc.find("choices")->as_array()[0].find("message")->find("tool_calls")->as_array()[0];
        CHECK(tc0.find("function")->find("name")->as_string() == "calc");
        const auto parsed_args = json::parse(tc0.find("function")->find("arguments")->as_string());
        REQUIRE(parsed_args.ok());
        CHECK(parsed_args.value().find("x")->as_int() == 42);
    }

    // 3. tool_choice: "none" suppresses tool calls
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Call weather tool"}],"tools":)" + tools_json +
                R"(,"tool_choice":"none","max_tokens":4})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        CHECK(choice.find("finish_reason")->as_string() != "tool_calls");
        CHECK(choice.find("message")->find("tool_calls") == nullptr);
        CHECK_FALSE(choice.find("message")->find("content")->as_string().empty());
    }

    // 4. Streaming delta.tool_calls suppresses raw <tool_call> tags from delta.content
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Check weather"}],"tools":)" + tools_json +
                R"(,"tool_choice":"required","stream":true,"stream_options":{"include_usage":true}})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto events = sse_data(r.body);
        REQUIRE(events.size() >= 3u);
        CHECK(events.back() == "[DONE]");
        std::string visible_content;
        std::string tool_name;
        std::string tool_args;
        std::string tool_id;
        std::string finish;
        for (std::size_t i = 0; i + 1 < events.size(); ++i) {
            auto parsed = json::parse(events[i]);
            REQUIRE(parsed.ok());
            const auto& choice = parsed.value().find("choices")->as_array()[0];
            if (const auto* fr = choice.find("finish_reason"); fr != nullptr && fr->is_string()) {
                finish = fr->as_string();
            }
            const auto* delta = choice.find("delta");
            REQUIRE(delta != nullptr);
            if (const auto* c = delta->find("content"); c != nullptr && c->is_string()) {
                visible_content += c->as_string();
            }
            if (const auto* tcs = delta->find("tool_calls"); tcs != nullptr && tcs->is_array()) {
                for (const auto& tc : tcs->as_array()) {
                    CHECK(tc.find("index")->as_int() == 0);
                    if (const auto* id = tc.find("id"); id != nullptr && id->is_string()) tool_id = id->as_string();
                    if (const auto* fn = tc.find("function"); fn != nullptr && fn->is_object()) {
                        if (const auto* nm = fn->find("name"); nm != nullptr && nm->is_string())
                            tool_name = nm->as_string();
                        if (const auto* ar = fn->find("arguments"); ar != nullptr && ar->is_string())
                            tool_args += ar->as_string();
                    }
                }
            }
        }
        CHECK(visible_content.find("<tool_call>") == std::string::npos);
        CHECK(finish == "tool_calls");
        CHECK_FALSE(tool_id.empty());
        CHECK(tool_name == "get_weather");
        CHECK(tool_args.find("mock_city") != std::string::npos);
    }

    // 5. Multi-turn with assistant tool_calls + role:"tool" returns final text
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[
                {"role":"user","content":"Weather in Paris?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"{\"temp_c\":18}"}
            ],"tools":)" + tools_json + R"(,"max_tokens":5})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        CHECK(choice.find("finish_reason")->as_string() != "tool_calls");
        CHECK_FALSE(choice.find("message")->find("content")->as_string().empty());
    }

    // 6. Grammar-capable backend synthesizes GBNF grammar for tool_choice: required
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const Reply r = post(
            cap.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"run calc"}],"tools":)" + tools_json +
                R"(,"tool_choice":{"type":"function","function":{"name":"calc"}}})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        std::lock_guard<std::mutex> lock(backend->model->mu);
        CHECK(backend->model->last_grammar.find("<tool_call>") != std::string::npos);
        CHECK(backend->model->last_grammar.find("calc") != std::string::npos);
    }
}

TEST_CASE("tool calling: Anthropic /v1/messages tool_use, streaming input_json_delta, tool_result, and vision") {
    Fixture f;
    const std::string tools_json =
        R"([{"name":"lookup","description":"Lookup record","input_schema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}])";

    // 1. Non-streaming tool_use
    {
        const Reply r = post(
            f.port, "/v1/messages",
            R"({"model":"default","max_tokens":64,"tools":)" + tools_json +
                R"(,"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"find record"}]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("stop_reason")->as_string() == "tool_use");
        const auto& content = doc.find("content")->as_array();
        bool saw_tool_use = false;
        for (const auto& blk : content) {
            if (blk.find("type")->as_string() == "tool_use") {
                saw_tool_use = true;
                CHECK(blk.find("name")->as_string() == "lookup");
                CHECK_FALSE(blk.find("id")->as_string().empty());
                CHECK(blk.find("input")->find("id")->as_string() == "mock_id");
            }
        }
        CHECK(saw_tool_use);
    }

    // 2. Streaming tool_use SSE events
    {
        const Reply r = post(
            f.port, "/v1/messages",
            R"({"model":"default","max_tokens":64,"stream":true,"tools":)" + tools_json +
                R"(,"tool_choice":{"type":"tool","name":"lookup"},"messages":[{"role":"user","content":"find"}]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto frames = parse_named_sse(r.body);
        bool saw_start = false;
        bool saw_delta = false;
        bool saw_stop_tool = false;
        for (const auto& [ev, data] : frames) {
            auto parsed = json::parse(data);
            REQUIRE(parsed.ok());
            if (ev == "content_block_start" &&
                parsed.value().find("content_block")->find("type")->as_string() == "tool_use") {
                saw_start = true;
                CHECK(parsed.value().find("content_block")->find("name")->as_string() == "lookup");
            }
            if (ev == "content_block_delta" &&
                parsed.value().find("delta")->find("type")->as_string() == "input_json_delta") {
                saw_delta = true;
                CHECK(parsed.value().find("delta")->find("partial_json")->as_string().find("mock_id") !=
                      std::string::npos);
            }
            if (ev == "message_delta" &&
                parsed.value().find("delta")->find("stop_reason")->as_string() == "tool_use") {
                saw_stop_tool = true;
            }
        }
        CHECK(saw_start);
        CHECK(saw_delta);
        CHECK(saw_stop_tool);
    }

    // 3. Multi-turn assistant tool_use + user tool_result
    {
        const Reply r = post(
            f.port, "/v1/messages",
            R"({"model":"default","max_tokens":8,"tools":)" + tools_json +
                R"(,"messages":[
                    {"role":"user","content":"lookup 123"},
                    {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"lookup","input":{"id":"123"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"found"}]}
                ]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.json().find("stop_reason")->as_string() != "tool_use");
    }

    // 4. Valid Anthropic base64 image block succeeds
    {
        const Reply r = post(
            f.port, "/v1/messages",
            std::string(R"({"model":"default","max_tokens":4,"messages":[{"role":"user","content":[)") +
                R"({"type":"text","text":"describe"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":")" +
                kTinyPngBase64 + R"("}}]}]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
    }
}

TEST_CASE("responses API: POST /v1/responses non-streaming, previous_response_id chaining, tools, and SSE") {
    Fixture f;

    // 1. Non-streaming Turn 1
    std::string resp1_id;
    std::string resp1_session_key;
    {
        const Reply r = post(
            f.port, "/v1/responses",
            R"({"model":"default","instructions":"Be terse.","input":"Hello responses API","max_output_tokens":5})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("object")->as_string() == "response");
        resp1_id = doc.find("id")->as_string();
        CHECK(resp1_id.rfind("resp_", 0) == 0);
        CHECK(doc.find("status")->as_string() == "completed");
        CHECK(doc.find("previous_response_id")->is_null());
        CHECK_FALSE(doc.find("output_text")->as_string().empty());
        const auto& out_arr = doc.find("output")->as_array();
        REQUIRE(out_arr.size() == 1u);
        CHECK(out_arr[0].find("type")->as_string() == "message");
        CHECK(doc.find("usage")->find("input_tokens")->as_uint() > 0u);
        CHECK(doc.find("usage")->find("output_tokens")->as_uint() == 5u);
        CHECK(doc.find("usage")->find("total_tokens")->as_uint() ==
              doc.find("usage")->find("input_tokens")->as_uint() + 5u);
        resp1_session_key = doc.find("sonder")->find("session_key")->as_string();
        CHECK_FALSE(resp1_session_key.empty());
    }

    // 2. Turn 2 chaining via previous_response_id inherits session_key for KV prefix reuse
    {
        const Reply r = post(
            f.port, "/v1/responses",
            R"({"model":"default","previous_response_id":")" + resp1_id +
                R"(","input":"Follow up question","max_output_tokens":4})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(doc.find("previous_response_id")->as_string() == resp1_id);
        CHECK(doc.find("sonder")->find("session_key")->as_string() == resp1_session_key);
        CHECK(doc.find("usage")->find("input_tokens")->as_uint() > 5u);
    }

    // 3. Unknown previous_response_id -> 404
    {
        const Reply r = post(f.port, "/v1/responses",
                             R"({"previous_response_id":"resp_nonexistent","input":"hi"})");
        CHECK(r.status == 404);
    }

    // 4. SSE streaming text events: response.created -> response.output_text.delta -> response.completed
    {
        const Reply r = post(f.port, "/v1/responses",
                             R"({"input":"Stream this","max_output_tokens":4,"stream":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.header("content-type") == "text/event-stream; charset=utf-8");
        const auto frames = parse_named_sse(r.body);
        REQUIRE(frames.size() >= 3u);
        CHECK(frames.front().first == "response.created");
        CHECK(frames.back().first == "response.completed");
        bool saw_text_delta = false;
        for (const auto& [ev, data] : frames) {
            if (ev == "response.output_text.delta") {
                saw_text_delta = true;
                auto p = json::parse(data);
                REQUIRE(p.ok());
                CHECK_FALSE(p.value().find("delta")->as_string().empty());
            }
        }
        CHECK(saw_text_delta);
    }

    // 5. Tool calling + function_call_output on /v1/responses (streaming & non-streaming)
    {
        const Reply r = post(
            f.port, "/v1/responses",
            R"({"input":"Run search","tools":[{"type":"function","name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}],"tool_choice":"required","stream":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto frames = parse_named_sse(r.body);
        bool saw_fn_delta = false;
        std::string resp_id;
        std::string call_id;
        for (const auto& [ev, data] : frames) {
            auto p = json::parse(data);
            REQUIRE(p.ok());
            if (ev == "response.function_call_arguments.delta") {
                saw_fn_delta = true;
                CHECK(p.value().find("name")->as_string() == "search");
                call_id = p.value().find("call_id")->as_string();
            }
            if (ev == "response.completed") {
                resp_id = p.value().find("response")->find("id")->as_string();
            }
        }
        CHECK(saw_fn_delta);
        REQUIRE_FALSE(resp_id.empty());
        REQUIRE_FALSE(call_id.empty());

        // Follow up with function_call_output chained to resp_id
        const Reply r2 = post(
            f.port, "/v1/responses",
            R"({"previous_response_id":")" + resp_id +
                R"(","input":[{"type":"function_call_output","call_id":")" + call_id +
                R"(","output":"result_42"}],"max_output_tokens":4})");
        REQUIRE_MESSAGE(r2.status == 200, r2.body);
        CHECK_FALSE(r2.json().find("output_text")->as_string().empty());
    }
}

TEST_CASE("logprobs: chat completions and legacy completions (streaming + non-streaming + validation)") {
    Fixture f;

    // 1. Chat non-streaming with logprobs + top_logprobs=3
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":4,"logprobs":true,"top_logprobs":3})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        const auto* lp = choice.find("logprobs");
        REQUIRE((lp != nullptr && lp->is_object()));
        const auto& content = lp->find("content")->as_array();
        REQUIRE(content.size() == 4u);
        for (const auto& tok : content) {
            CHECK_FALSE(tok.find("token")->as_string().empty());
            CHECK(tok.find("logprob")->as_double() < 0.0);
            CHECK_FALSE(tok.find("bytes")->as_array().empty());
            CHECK(tok.find("top_logprobs")->as_array().size() == 3u);
        }
    }

    // 2. Chat streaming with logprobs + top_logprobs=2
    {
        const Reply r = post(
            f.port, "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":3,"stream":true,"logprobs":true,"top_logprobs":2})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto events = sse_data(r.body);
        std::size_t lp_chunks = 0;
        for (std::size_t i = 0; i + 1 < events.size(); ++i) {
            auto p = json::parse(events[i]);
            REQUIRE(p.ok());
            const auto* lp = p.value().find("choices")->as_array()[0].find("logprobs");
            if (lp != nullptr && lp->is_object()) {
                ++lp_chunks;
                const auto& c = lp->find("content")->as_array();
                REQUIRE(c.size() == 1u);
                CHECK(c[0].find("top_logprobs")->as_array().size() == 2u);
            }
        }
        CHECK(lp_chunks == 3u);
    }

    // 3. Completions non-streaming & streaming with logprobs=3
    {
        const Reply r = post(f.port, "/v1/completions",
                             R"({"prompt":"hi","max_tokens":3,"logprobs":3})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto* lp = doc.find("choices")->as_array()[0].find("logprobs");
        REQUIRE((lp != nullptr && lp->is_object()));
        CHECK(lp->find("tokens")->as_array().size() == 3u);
        CHECK(lp->find("token_logprobs")->as_array().size() == 3u);
        CHECK(lp->find("top_logprobs")->as_array().size() == 3u);
        CHECK(lp->find("text_offset")->as_array().size() == 3u);

        const Reply rs = post(f.port, "/v1/completions",
                              R"({"prompt":"hi","max_tokens":3,"stream":true,"logprobs":2})");
        REQUIRE_MESSAGE(rs.status == 200, rs.body);
        const auto evs = sse_data(rs.body);
        std::size_t cmpl_lp_chunks = 0;
        for (std::size_t i = 0; i + 1 < evs.size(); ++i) {
            auto p = json::parse(evs[i]);
            REQUIRE(p.ok());
            const auto* slp = p.value().find("choices")->as_array()[0].find("logprobs");
            if (slp != nullptr && slp->is_object()) {
                ++cmpl_lp_chunks;
                CHECK(slp->find("tokens")->as_array().size() == 1u);
            }
        }
        CHECK(cmpl_lp_chunks == 3u);
    }

    // 4. Validation & capability check
    CHECK(post(f.port, "/v1/chat/completions",
               R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":2})")
              .status == 400);
    CHECK(post(f.port, "/v1/chat/completions",
               R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":21})")
              .status == 400);
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const Reply r = post(cap.port, "/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true})");
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }
}

TEST_CASE("multimodal vision: accepts inline base64 data URIs and strictly rejects SSRF URLs") {
    Fixture f;

    // 1. Valid inline base64 PNG succeeds
    {
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":[)") +
            R"({"type":"text","text":"what is this?"},)" +
            R"({"type":"image_url","image_url":{"url":"data:image/png;base64,)" + kTinyPngBase64 +
            R"("}}]}],"max_tokens":4})";
        const Reply r = post(f.port, "/v1/chat/completions", body);
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK_FALSE(r.json().find("choices")->as_array()[0].find("message")->find("content")->as_string().empty());
    }

    // 2. SSRF rejection: http://, https://, file://, non-allowlisted MIME, invalid base64
    for (const char* bad_url : {
             "http://127.0.0.1:8080/secret.png",
             "https://attacker.example/img.png",
             "file:///C:/Windows/win.ini",
             "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
             "data:image/png;base64,!!!not-base64!!!",
         }) {
        CAPTURE(bad_url);
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":")") +
            bad_url + R"("}}]}]})";
        const Reply r = post(f.port, "/v1/chat/completions", body);
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "invalid_request");
    }

    // 3. Exceeding max 8 images per request rejected with 400
    {
        std::string parts;
        for (int i = 0; i < 9; ++i) {
            if (i > 0) parts += ",";
            parts += std::string(R"({"type":"image_url","image_url":{"url":"data:image/png;base64,)") +
                     kTinyPngBase64 + R"("}})";
        }
        const Reply r = post(f.port, "/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":[)" + parts + R"(]}]})");
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "invalid_request");
    }

    // 4. Backend without Capability::multimodal rejects media with 400 unsupported_parameter
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,)") +
            kTinyPngBase64 + R"("}}]}]})";
        const Reply r = post(cap.port, "/v1/chat/completions", body);
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "unsupported_parameter");
    }
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
TEST_CASE("llamaserver protocol: forwards tools, tool_choice, tool_calls, multimodal image_url, and logprobs") {
    si::ChatRequest req;
    si::ChatMessage user_msg;
    user_msg.role = "user";
    user_msg.content = "inspect image";
    user_msg.media.push_back(si::MediaContent{"image/png", {0x89, 0x50, 0x4E, 0x47}});
    si::ChatMessage asst_msg;
    asst_msg.role = "assistant";
    asst_msg.tool_calls.push_back(si::ToolCall{"call_1", "lookup", R"({"id":"a"})"});
    si::ChatMessage tool_msg;
    tool_msg.role = "tool";
    tool_msg.tool_call_id = "call_1";
    tool_msg.content = "ok";
    req.messages = {user_msg, asst_msg, tool_msg};
    req.tools.push_back(si::ToolDefinition{"lookup", "Lookup item", R"({"type":"object"})"});
    req.tool_choice = si::ToolChoice{si::ToolChoiceMode::function, "lookup"};
    req.sampling.logprobs = true;
    req.sampling.top_logprobs = 4;

    const auto body = si::llamaserver::build_chat_body("m", req);
    REQUIRE(body.find("tools") != nullptr);
    CHECK(body.find("tools")->as_array().size() == 1u);
    REQUIRE(body.find("tool_choice") != nullptr);
    CHECK(body.find("tool_choice")->find("function")->find("name")->as_string() == "lookup");
    CHECK(body.find("logprobs")->as_bool());
    CHECK(body.find("top_logprobs")->as_int() == 4);
    const auto& msgs = body.find("messages")->as_array();
    REQUIRE(msgs.size() == 3u);
    CHECK(msgs[0].find("content")->is_array());
    CHECK(msgs[0].find("content")->as_array()[1].find("image_url")->find("url")->as_string().rfind(
              "data:image/png;base64,", 0) == 0);
    CHECK(msgs[1].find("tool_calls")->as_array().size() == 1u);
    CHECK(msgs[2].find("tool_call_id")->as_string() == "call_1");
}
#endif
