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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
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

namespace {

// Backend that blocks in prefill until `release()` (or cancellation), so tests
// can deterministically observe SSE `: keep-alive\n\n` heartbeats before the
// first token arrives.
struct SlowPrefillBackend final : si::Backend {
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        bool entered = false;
        bool released = false;
        bool fail_after_release = false;
        bool saw_cancel = false;
        std::string reply_text = "ok";
    };

    struct Model final : si::BackendModel {
        std::shared_ptr<State> st;
        si::ModelDescriptor desc;

        explicit Model(std::shared_ptr<State> s) : st(std::move(s)) {
            desc.name = "slow:test";
            desc.backend = "slow-test";
            desc.context_length = 4096;
        }

        const si::ModelDescriptor& descriptor() const override { return desc; }
        bool has_native_chat() const override { return true; }

        si::Result<si::GenerateStats> wait_and_emit(const si::CancellationToken& cancel,
                                                    const si::TokenCallback& on_chunk) {
            {
                std::unique_lock<std::mutex> lock(st->mu);
                st->entered = true;
                st->cv.notify_all();
                while (!st->released && !cancel.cancelled()) {
                    st->cv.wait_for(lock, std::chrono::milliseconds(10));
                }
                if (cancel.cancelled()) {
                    st->saw_cancel = true;
                    st->cv.notify_all();
                    return si::Status(si::ErrorCode::cancelled, "cancelled in prefill");
                }
                if (st->fail_after_release) {
                    return si::Status(si::ErrorCode::backend_error, "prefill boom");
                }
            }
            if (on_chunk) {
                (void)on_chunk(si::TokenChunk{st->reply_text, 0});
            }
            si::GenerateStats gs;
            gs.prompt_tokens = 12;
            gs.completion_tokens = 1;
            gs.chunks = 1;
            gs.token_counts_from_backend = true;
            gs.stop_reason = si::StopReason::end_of_sequence;
            return gs;
        }

        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken& cancel,
                                               const si::TokenCallback& on_chunk) override {
            return wait_and_emit(cancel, on_chunk);
        }

        si::Result<si::GenerateStats> chat(const si::ChatRequest&, const si::CancellationToken& cancel,
                                           const si::TokenCallback& on_chunk) override {
            return wait_and_emit(cancel, on_chunk);
        }
    };

    std::shared_ptr<State> state = std::make_shared<State>();
    std::shared_ptr<Model> model = std::make_shared<Model>(state);

    void reset(bool fail = false) {
        std::lock_guard<std::mutex> lock(state->mu);
        state->entered = false;
        state->released = false;
        state->fail_after_release = fail;
        state->saw_cancel = false;
    }

    bool wait_entered(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(state->mu);
        return state->cv.wait_for(lock, timeout, [&] { return state->entered; });
    }

    bool wait_cancelled(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(state->mu);
        return state->cv.wait_for(lock, timeout, [&] { return state->saw_cancel; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->released = true;
        }
        state->cv.notify_all();
    }

    std::string name() const override { return "slow-test"; }
    std::string description() const override { return "slow prefill test backend"; }
    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::tool_calling);
    }
    si::Result<std::string> probe() override { return std::string("slow-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

// Native-tool-capable backend that replays scripted TokenChunks and records
// incoming ChatRequest fields for contract assertions.
struct ScriptedToolBackend final : si::Backend {
    struct Model final : si::BackendModel {
        si::ModelDescriptor desc;
        std::mutex mu;
        std::vector<si::TokenChunk> chunks;
        std::vector<std::string> owned_text;
        si::StopReason stop_reason = si::StopReason::tool_calls;
        std::string last_grammar;
        std::vector<si::ChatMessage> last_messages;
        std::vector<si::ToolDefinition> last_tools;
        std::optional<si::ToolChoice> last_tool_choice;

        Model() {
            desc.name = "tool:test";
            desc.backend = "tool-test";
            desc.context_length = 4096;
        }

        const si::ModelDescriptor& descriptor() const override { return desc; }
        bool has_native_chat() const override { return true; }

        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                               const si::TokenCallback&) override {
            return si::Status(si::ErrorCode::unsupported, "unused");
        }

        si::Result<si::GenerateStats> chat(const si::ChatRequest& req, const si::CancellationToken&,
                                           const si::TokenCallback& on_chunk) override {
            std::vector<si::TokenChunk> script;
            si::StopReason sr = si::StopReason::end_of_sequence;
            {
                std::lock_guard<std::mutex> lock(mu);
                last_grammar = req.grammar;
                last_messages = req.messages;
                last_tools = req.tools;
                last_tool_choice = req.tool_choice;
                script = chunks;
                sr = stop_reason;
            }
            std::uint64_t idx = 0;
            for (auto c : script) {
                c.index = idx++;
                if (on_chunk && !on_chunk(c)) {
                    break;
                }
            }
            si::GenerateStats st;
            st.prompt_tokens = 10;
            st.completion_tokens = idx;
            st.chunks = idx;
            st.token_counts_from_backend = true;
            st.stop_reason = sr;
            return st;
        }
    };

    bool advertise_native_tools = true;
    std::shared_ptr<Model> model = std::make_shared<Model>();

    std::string name() const override { return "tool-test"; }
    std::string description() const override { return "scripted tool backend"; }
    si::BackendCapabilities capabilities() const override {
        si::BackendCapabilities caps;
        caps.add(si::Capability::streaming).add(si::Capability::structured_output);
        if (advertise_native_tools) {
            caps.add(si::Capability::tool_calling);
        }
        return caps;
    }
    si::Result<std::string> probe() override { return std::string("tool-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{model->descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::static_pointer_cast<si::BackendModel>(model);
    }
};

}  // namespace

TEST_CASE("capabilities: health and identity advertise native_tools, tools, and sse_keepalive") {
    // 1. Default mock backend (native tool_calling + default 15s keep-alive enabled).
    {
        Fixture f;
        const Reply hr = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(hr.status == 200, hr.body);
        const auto h = hr.json();
        const auto* hcaps = h.find("capabilities");
        REQUIRE((hcaps != nullptr && hcaps->is_object()));
        CHECK(hcaps->find("tools")->as_bool());
        CHECK(hcaps->find("native_tools")->as_bool());
        CHECK(hcaps->find("backend_native_tools")->as_bool());
        CHECK(hcaps->find("tool_choice")->as_bool());
        CHECK(hcaps->find("responses_api")->as_bool());
        CHECK(hcaps->find("sse_keepalive")->as_bool());
        CHECK_FALSE(hcaps->find("structured_output")->as_bool());
        CHECK(hcaps->find("multimodal")->as_bool());
        CHECK(hcaps->find("logprobs")->as_bool());
        CHECK(hcaps->find("embeddings")->as_bool());
        // Preserve existing sonder.features contract for non-llamaserver backends.
        CHECK(h.find("sonder")->find("features")->as_array().empty());

        const Reply ir = get(f.port, "/v1/sonder/identity");
        REQUIRE_MESSAGE(ir.status == 200, ir.body);
        const auto id = ir.json();
        const auto* icaps = id.find("capabilities");
        REQUIRE((icaps != nullptr && icaps->is_object()));
        CHECK(icaps->find("tools")->as_bool());
        CHECK(icaps->find("native_tools")->as_bool());
        CHECK(icaps->find("backend_native_tools")->as_bool());
        CHECK(icaps->find("tool_choice")->as_bool());
        CHECK(icaps->find("responses_api")->as_bool());
        CHECK(icaps->find("sse_keepalive")->as_bool());
        // Preserve exact 9-key backend_identity schema.
        CHECK(id.find("backend_identity")->as_object().size() == 9u);
    }

    // 2. Non-native backend (CapturingBackend) with SSE keep-alive disabled (0ms).
    {
        auto backend = std::make_shared<CapturingBackend>();
        auto opts = capturing_options(backend);
        opts.stream_keepalive_interval = std::chrono::milliseconds(0);
        Fixture cap(opts);

        for (const char* path : {"/v1/sonder/health", "/v1/sonder/identity"}) {
            CAPTURE(path);
            const Reply r = get(cap.port, path);
            REQUIRE_MESSAGE(r.status == 200, r.body);
            const auto doc = r.json();
            const auto* caps = doc.find("capabilities");
            REQUIRE((caps != nullptr && caps->is_object()));
            CHECK(caps->find("tools")->as_bool());
            CHECK(caps->find("native_tools")->as_bool());
            CHECK_FALSE(caps->find("backend_native_tools")->as_bool());
            CHECK_FALSE(caps->find("sse_keepalive")->as_bool());
            CHECK(caps->find("structured_output")->as_bool());
            CHECK_FALSE(caps->find("multimodal")->as_bool());
            CHECK_FALSE(caps->find("logprobs")->as_bool());
            CHECK_FALSE(caps->find("embeddings")->as_bool());
        }
    }
}

TEST_CASE("sse keepalive: emits keep-alive comments during long prefill across streaming endpoints") {
    auto backend = std::make_shared<SlowPrefillBackend>();
    auto opts = Fixture::defaults();
    opts.models = {"slow:test"};
    opts.backend.backend.clear();
    opts.backend_instance = backend;
    opts.stream_keepalive_interval = std::chrono::milliseconds(30);
    Fixture f(opts);

    // 1. /v1/chat/completions streaming receives `: keep-alive\n\n` during prefill,
    //    then receives valid OpenAI SSE chunks + [DONE] once released.
    {
        backend->reset();
        Conn c(f.port);
        c.send(build_request("POST", "/v1/chat/completions", f.port, {},
                             R"({"messages":[{"role":"user","content":"long prompt"}],"stream":true})"));
        REQUIRE(backend->wait_entered());
        std::string acc;
        REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
        CHECK(acc.find("HTTP/1.1 200 OK") == 0);
        CHECK(acc.find("text/event-stream") != std::string::npos);
        CHECK(acc.find("chat.completion.chunk") == std::string::npos);
        backend->release();
        acc += c.read_all();
        const Reply r = parse_reply(acc);
        CHECK(r.status == 200);
        CHECK(r.body.find(": keep-alive\n\n") != std::string::npos);
        const auto events = sse_data(r.body);
        REQUIRE(events.size() >= 4u);
        CHECK(events.back() == "[DONE]");
        std::string text;
        bool saw_role = false;
        for (std::size_t i = 0; i + 1 < events.size(); ++i) {
            auto p = json::parse(events[i]);
            REQUIRE_MESSAGE(p.ok(), events[i]);
            const auto& delta = p.value().find("choices")->as_array()[0].find("delta")->as_object();
            if (const auto* role = delta.find("role"); role && role->is_string() && role->as_string() == "assistant") {
                saw_role = true;
            }
            if (const auto* ct = delta.find("content"); ct && ct->is_string()) {
                text += ct->as_string();
            }
        }
        CHECK(saw_role);
        CHECK(text == "ok");
    }

    // 2. /v1/messages (Anthropic) streaming receives `: keep-alive\n\n` during prefill,
    //    then receives message_start -> content_block_delta -> message_stop cleanly.
    {
        backend->reset();
        Conn c(f.port);
        c.send(build_request("POST", "/v1/messages", f.port, {{"anthropic-version", "2023-06-01"}},
                             R"({"model":"slow:test","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]})"));
        REQUIRE(backend->wait_entered());
        std::string acc;
        REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
        CHECK(acc.find("event: message_start") == std::string::npos);
        backend->release();
        acc += c.read_all();
        const Reply r = parse_reply(acc);
        CHECK(r.status == 200);
        CHECK(r.body.find(": keep-alive\n\n") != std::string::npos);
        CHECK(r.body.find("event: message_start") != std::string::npos);
        CHECK(r.body.find("event: content_block_delta") != std::string::npos);
        CHECK(r.body.find("event: message_stop") != std::string::npos);
    }

    // 3. /v1/completions streaming receives `: keep-alive\n\n` during prefill,
    //    then receives text completion chunk + [DONE].
    {
        backend->reset();
        Conn c(f.port);
        c.send(build_request("POST", "/v1/completions", f.port, {},
                             R"({"prompt":"long prompt","stream":true})"));
        REQUIRE(backend->wait_entered());
        std::string acc;
        REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
        backend->release();
        acc += c.read_all();
        const Reply r = parse_reply(acc);
        CHECK(r.status == 200);
        CHECK(r.body.find(": keep-alive\n\n") != std::string::npos);
        const auto events = sse_data(r.body);
        REQUIRE(events.size() >= 2u);
        CHECK(events.back() == "[DONE]");
    }

    // 4. /v1/responses streaming receives `: keep-alive\n\n` during prefill,
    //    then receives response.created -> output_text.delta -> response.completed.
    {
        backend->reset();
        Conn c(f.port);
        c.send(build_request("POST", "/v1/responses", f.port, {},
                             R"({"input":"long prompt","stream":true})"));
        REQUIRE(backend->wait_entered());
        std::string acc;
        REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
        CHECK(acc.find("event: response.created") == std::string::npos);
        backend->release();
        acc += c.read_all();
        const Reply r = parse_reply(acc);
        CHECK(r.status == 200);
        CHECK(r.body.find(": keep-alive\n\n") != std::string::npos);
        CHECK(r.body.find("event: response.created") != std::string::npos);
        CHECK(r.body.find("event: response.output_text.delta") != std::string::npos);
        CHECK(r.body.find("event: response.completed") != std::string::npos);
    }

    // 5. Prefill error after keep-alive already committed headers emits a clean SSE
    //    error frame without fake assistant preamble or [DONE].
    {
        backend->reset(/*fail=*/true);
        Conn c(f.port);
        c.send(build_request("POST", "/v1/chat/completions", f.port, {},
                             R"({"messages":[{"role":"user","content":"will fail"}],"stream":true})"));
        REQUIRE(backend->wait_entered());
        std::string acc;
        REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
        backend->release();
        acc += c.read_all();
        const Reply r = parse_reply(acc);
        CHECK(r.status == 200);
        CHECK(r.body.find("chat.completion.chunk") == std::string::npos);
        CHECK(r.body.find("[DONE]") == std::string::npos);
        const auto events = sse_data(r.body);
        REQUIRE(events.size() == 1u);
        auto err_doc = json::parse(events[0]);
        REQUIRE(err_doc.ok());
        CHECK(err_doc.value().find("error") != nullptr);
    }

    // 6. Client disconnect after keep-alive cancels backend cleanly.
    {
        backend->reset();
        {
            Conn c(f.port);
            c.send(build_request("POST", "/v1/chat/completions", f.port, {},
                                 R"({"messages":[{"role":"user","content":"drop me"}],"stream":true})"));
            REQUIRE(backend->wait_entered());
            std::string acc;
            REQUIRE(c.read_until(acc, ": keep-alive\n\n", std::chrono::milliseconds(5000)));
            c.close();
        }
        CHECK(backend->wait_cancelled(std::chrono::milliseconds(5000)));
    }

    // 7. Disabled when stream_keepalive_interval is 0ms.
    {
        auto b2 = std::make_shared<SlowPrefillBackend>();
        auto opts2 = Fixture::defaults();
        opts2.models = {"slow:test"};
        opts2.backend.backend.clear();
        opts2.backend_instance = b2;
        opts2.stream_keepalive_interval = std::chrono::milliseconds(0);
        Fixture f2(opts2);

        b2->reset();
        Conn c(f2.port);
        c.send(build_request("POST", "/v1/chat/completions", f2.port, {},
                             R"({"messages":[{"role":"user","content":"quiet"}],"stream":true})"));
        REQUIRE(b2->wait_entered());
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        b2->release();
        const Reply r = parse_reply(c.read_all());
        CHECK(r.status == 200);
        CHECK(r.body.find(": keep-alive") == std::string::npos);
        CHECK(r.body.find("[DONE]") != std::string::npos);
    }
}

TEST_CASE("streaming tool_calls: incremental deltas, parallel calls, id synthesis, and tool_choice") {
    auto backend = std::make_shared<ScriptedToolBackend>();
    auto opts = Fixture::defaults();
    opts.models = {"tool:test"};
    opts.backend.backend.clear();
    opts.backend_instance = backend;
    Fixture f(opts);

    const std::string tools_json =
        R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},)"
        R"({"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}])";

    // 1. Multi-chunk streaming deltas across 2 parallel tool calls (one explicit id,
    //    one omitted id that must be auto-synthesized on its first delta only).
    {
        std::lock_guard<std::mutex> lock(backend->model->mu);
        backend->model->stop_reason = si::StopReason::tool_calls;
        backend->model->chunks.clear();
        si::TokenChunk c0;
        c0.tool_call_deltas.push_back(si::ToolCallDelta{0, "call_explicit_0", "get_weather", R"({"city":)"});
        si::TokenChunk c1;
        c1.tool_call_deltas.push_back(si::ToolCallDelta{0, "", "", R"("NYC"})"});
        si::TokenChunk c2;
        c2.tool_call_deltas.push_back(si::ToolCallDelta{1, "", "get_time", R"({"tz":)"});
        si::TokenChunk c3;
        c3.tool_call_deltas.push_back(si::ToolCallDelta{1, "", "", R"("UTC"})"});
        backend->model->chunks = {c0, c1, c2, c3};
    }
    {
        const Reply r = post(f.port, "/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"weather and time"}],"tools":)" + tools_json +
                                 R"(,"tool_choice":"required","stream":true})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        // Native tool-capable backend must NOT receive synthesized raw GBNF grammar.
        {
            std::lock_guard<std::mutex> lock(backend->model->mu);
            CHECK(backend->model->last_grammar.empty());
        }
        const auto events = sse_data(r.body);
        REQUIRE(events.size() >= 6u);
        CHECK(events.back() == "[DONE]");

        std::string id0, id1, name0, name1, args0, args1;
        int id1_emissions = 0;
        std::string finish;
        for (std::size_t i = 0; i + 1 < events.size(); ++i) {
            auto p = json::parse(events[i]);
            REQUIRE_MESSAGE(p.ok(), events[i]);
            const auto& choice = p.value().find("choices")->as_array()[0];
            if (const auto* fr = choice.find("finish_reason"); fr && fr->is_string()) {
                finish = fr->as_string();
            }
            const auto& delta = choice.find("delta")->as_object();
            if (const auto* tcs = delta.find("tool_calls"); tcs && tcs->is_array()) {
                for (const auto& tc : tcs->as_array()) {
                    const auto idx = tc.find("index")->as_int();
                    const auto* idv = tc.find("id");
                    const auto& fn = tc.find("function")->as_object();
                    const auto* nv = fn.find("name");
                    const auto* av = fn.find("arguments");
                    if (idx == 0) {
                        if (idv && idv->is_string()) id0 = idv->as_string();
                        if (nv && nv->is_string()) name0 = nv->as_string();
                        if (av && av->is_string()) args0 += av->as_string();
                    } else if (idx == 1) {
                        if (idv && idv->is_string()) {
                            id1 = idv->as_string();
                            ++id1_emissions;
                        }
                        if (nv && nv->is_string()) name1 = nv->as_string();
                        if (av && av->is_string()) args1 += av->as_string();
                    }
                }
            }
        }
        CHECK(id0 == "call_explicit_0");
        CHECK(name0 == "get_weather");
        CHECK(args0 == R"({"city":"NYC"})");
        CHECK(id1.rfind("call_", 0) == 0);
        CHECK(id1_emissions == 1);  // synthesized on first delta only, not repeated
        CHECK(name1 == "get_time");
        CHECK(args1 == R"({"tz":"UTC"})");
        CHECK(finish == "tool_calls");
    }

    // 2. Non-streaming chat synthesizes non-empty id when backend omits tool_call id.
    {
        std::lock_guard<std::mutex> lock(backend->model->mu);
        backend->model->chunks.clear();
        si::TokenChunk c;
        c.tool_call_deltas.push_back(si::ToolCallDelta{0, "", "get_weather", R"({"city":"Paris"})"});
        backend->model->chunks = {c};
    }
    {
        const Reply r = post(f.port, "/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"paris"}],"tools":)" + tools_json + R"(})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        CHECK(choice.find("finish_reason")->as_string() == "tool_calls");
        const auto& tcs = choice.find("message")->find("tool_calls")->as_array();
        REQUIRE(tcs.size() == 1u);
        CHECK(tcs[0].find("id")->as_string().rfind("call_", 0) == 0);
        CHECK(tcs[0].find("function")->find("name")->as_string() == "get_weather");
        CHECK(tcs[0].find("function")->find("arguments")->as_string() == R"({"city":"Paris"})");
    }

    // 3. /v1/responses streaming with sparse tool call index (index 2 only) filters
    //    out empty placeholder slots before response.completed.
    {
        std::lock_guard<std::mutex> lock(backend->model->mu);
        backend->model->chunks.clear();
        si::TokenChunk c;
        c.tool_call_deltas.push_back(si::ToolCallDelta{2, "", "get_time", R"({"tz":"CST"})"});
        backend->model->chunks = {c};
    }
    {
        const Reply r = post(
            f.port, "/v1/responses",
            R"({"input":"time?","stream":true,"tools":[{"type":"function","name":"get_time","parameters":{"type":"object"}}]})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.body.find("event: response.function_call_arguments.delta") != std::string::npos);
        const auto completed_pos = r.body.find("event: response.completed\ndata: ");
        REQUIRE(completed_pos != std::string::npos);
        const auto data_start = completed_pos + std::string("event: response.completed\ndata: ").size();
        const auto data_end = r.body.find("\n\n", data_start);
        auto parsed = json::parse(r.body.substr(data_start, data_end - data_start));
        REQUIRE(parsed.ok());
        const auto& output = parsed.value().find("response")->find("output")->as_array();
        REQUIRE(output.size() == 1u);
        CHECK(output[0].find("type")->as_string() == "function_call");
        CHECK(output[0].find("name")->as_string() == "get_time");
        CHECK(output[0].find("call_id")->as_string().rfind("call_", 0) == 0);
    }

    // 4. Non-native backend with tool_choice: "none" must NOT inject tool prompts or
    //    parse literal <tool_call> tags into structured tool_calls.
    {
        auto non_native = std::make_shared<ScriptedToolBackend>();
        non_native->advertise_native_tools = false;
        non_native->model->owned_text.push_back(R"(<tool_call>{"name":"get_weather","arguments":{"city":"LA"}}</tool_call>)");
        non_native->model->stop_reason = si::StopReason::end_of_sequence;
        non_native->model->chunks = {si::TokenChunk{non_native->model->owned_text.back(), 0}};
        auto nn_opts = Fixture::defaults();
        nn_opts.models = {"tool:test"};
        nn_opts.backend.backend.clear();
        nn_opts.backend_instance = non_native;
        Fixture f_nn(nn_opts);

        const Reply r = post(f_nn.port, "/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"echo tag"}],"tools":)" + tools_json +
                                 R"(,"tool_choice":"none"})");
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        const auto& choice = doc.find("choices")->as_array()[0];
        CHECK(choice.find("finish_reason")->as_string() == "stop");
        CHECK(choice.find("message")->find("tool_calls") == nullptr);
        CHECK(choice.find("message")->find("content")->as_string().find("<tool_call>") != std::string::npos);
        std::lock_guard<std::mutex> lock(non_native->model->mu);
        CHECK(non_native->model->last_messages.size() == 1u);  // no synthetic system tool prompt added
    }

    // 5. tool_choice and tools validation error paths on chat and responses.
    for (const char* bad_body : {
             R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"})",
             R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":"bogus"})",
             R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":123})",
             R"({"messages":[{"role":"user","content":"hi"}],"tools":"not-array"})",
             R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"bad name!"}}]})",
             R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"dup"}},{"type":"function","function":{"name":"dup"}}]})",
             R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":{"type":"function","function":{"name":"missing"}}})",
         }) {
        CAPTURE(bad_body);
        const Reply r = post(f.port, "/v1/chat/completions", bad_body);
        CHECK(r.status == 400);
        CHECK(r.json().find("error") != nullptr);
    }
    {
        const Reply r = post(
            f.port, "/v1/responses",
            R"({"input":"hi","tools":[{"type":"function","name":"a"}],"tool_choice":{"type":"function","name":"missing"}})");
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("param")->as_string() == "tool_choice");
    }
}

TEST_CASE("responses previous_response_id: error paths and malformed JSON resilience") {
    auto opts = Fixture::defaults();
    opts.models = {"mock:a", "mock:b"};
    Fixture f(opts);

    // 1. Unknown previous_response_id -> 404 not_found
    {
        const Reply r = post(f.port, "/v1/responses",
                             R"({"model":"mock:a","previous_response_id":"resp_nonexistent","input":"hi"})");
        CHECK(r.status == 404);
        CHECK(r.json().find("error")->find("code")->as_string() == "not_found");
        CHECK(r.json().find("error")->find("param")->as_string() == "previous_response_id");
    }

    // 2. Create stored response on mock:a, then chain on mock:b -> 400 cross-model error
    {
        const Reply r1 = post(f.port, "/v1/responses",
                              R"({"model":"mock:a","input":"turn 1","max_output_tokens":4})");
        REQUIRE_MESSAGE(r1.status == 200, r1.body);
        const std::string resp_id = r1.json().find("id")->as_string();

        const Reply r_cross = post(f.port, "/v1/responses",
                                   R"({"model":"mock:b","previous_response_id":")" + resp_id + R"(","input":"turn 2"})");
        CHECK(r_cross.status == 400);
        CHECK(r_cross.json().find("error")->find("code")->as_string() == "invalid_request");
        CHECK(r_cross.json().find("error")->find("param")->as_string() == "previous_response_id");
    }

    // 3. store: false turn is NOT retained -> chaining its ID returns 404
    {
        const Reply r_nostore = post(f.port, "/v1/responses",
                                     R"({"model":"mock:a","input":"ephemeral","store":false,"max_output_tokens":4})");
        REQUIRE_MESSAGE(r_nostore.status == 200, r_nostore.body);
        const std::string ephem_id = r_nostore.json().find("id")->as_string();

        const Reply r_chain = post(f.port, "/v1/responses",
                                   R"({"model":"mock:a","previous_response_id":")" + ephem_id + R"(","input":"follow"})");
        CHECK(r_chain.status == 404);
        CHECK(r_chain.json().find("error")->find("code")->as_string() == "not_found");
    }

    // 4. Malformed previous_response_id values (empty, >128 bytes, non-string, control char) -> 400
    const std::string long_id(129, 'x');
    for (const std::string& bad_prev : {
             std::string(R"({"input":"hi","previous_response_id":""})"),
             std::string(R"({"input":"hi","previous_response_id":")") + long_id + R"("})",
             std::string(R"({"input":"hi","previous_response_id":42})"),
             std::string(R"({"input":"hi","previous_response_id":"resp_1\n"})"),
         }) {
        CAPTURE(bad_prev);
        const Reply r = post(f.port, "/v1/responses", bad_prev);
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("param")->as_string() == "previous_response_id");
    }

    // 5. Non-string input[0].type and invalid role=tool tool_call_id -> 400, valid tool_call_id -> 200
    {
        const Reply r_type = post(f.port, "/v1/responses",
                                  R"({"input":[{"type":123,"role":"user","content":"hi"}]})");
        CHECK(r_type.status == 400);
        CHECK(r_type.json().find("error")->find("param")->as_string() == "input[0].type");

        const Reply r_bad_tcid = post(f.port, "/v1/responses",
                                      R"({"input":[{"role":"user","content":"hi"},{"role":"tool","tool_call_id":"","content":"out"}]})");
        CHECK(r_bad_tcid.status == 400);
        CHECK(r_bad_tcid.json().find("error")->find("param")->as_string() == "input[1].tool_call_id");

        const Reply r_ok_tcid = post(
            f.port, "/v1/responses",
            R"({"input":[{"role":"user","content":"hi"},{"type":"function_call","call_id":"c1","name":"fn","arguments":"{}"},{"role":"tool","tool_call_id":"c1","content":"ok"}],"max_output_tokens":4})");
        CHECK(r_ok_tcid.status == 200);
    }

    // 6. Malformed JSON resilience across all JSON POST endpoints (truncated, non-object,
    //    deeply nested >128, wrong field types).
    std::string deep_json;
    for (int i = 0; i < 160; ++i) deep_json += "{\"a\":";
    deep_json += "1";
    for (int i = 0; i < 160; ++i) deep_json += "}";

    for (const char* path :
         {"/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions", "/v1/embeddings"}) {
        for (const std::string& payload :
             {std::string(R"({"truncated":[)"), std::string("[1,2,3]"), std::string(R"("scalar")"),
              std::string("null"), deep_json}) {
            CAPTURE(path);
            const Reply r = post(f.port, path, payload);
            CHECK(r.status == 400);
            const auto doc = r.json();
            CHECK(doc.find("error") != nullptr);
        }
    }
    // Wrong field types return structured 400s without crashing.
    CHECK(post(f.port, "/v1/chat/completions", R"({"messages":"not-an-array"})").status == 400);
    CHECK(post(f.port, "/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],"stream":"yes"})").status == 400);
    CHECK(post(f.port, "/v1/responses", R"({"input":[]})").status == 400);
    CHECK(post(f.port, "/v1/completions", R"({"prompt":123})").status == 400);

    // Server remains healthy after malformed traffic burst.
    CHECK(get(f.port, "/v1/sonder/health").status == 200);
}

TEST_CASE("health_features HTTP: direct llamacpp advertises capability-derived sonder.features on health and models") {
    struct FakeDirectLlamaCppBackend final : si::Backend {
        struct Model final : si::BackendModel {
            si::ModelDescriptor desc;
            Model() {
                desc.name = "direct:q4";
                desc.backend = "llamacpp";
                desc.context_length = 4096;
            }
            const si::ModelDescriptor& descriptor() const override { return desc; }
            bool has_native_chat() const override { return true; }
            si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                                   const si::TokenCallback& on_chunk) override {
                if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
                si::GenerateStats st;
                st.prompt_tokens = 2;
                st.completion_tokens = 1;
                st.chunks = 1;
                st.token_counts_from_backend = true;
                st.stop_reason = si::StopReason::end_of_sequence;
                return st;
            }
        };
        bool enable_multimodal = false;
        bool enable_multi_slot = false;
        std::shared_ptr<Model> model = std::make_shared<Model>();
        std::string name() const override { return "llamacpp"; }
        std::string description() const override { return "fake direct llamacpp backend"; }
        si::BackendCapabilities capabilities() const override {
            si::BackendCapabilities caps;
            caps.add(si::Capability::tokenization)
                .add(si::Capability::streaming)
                .add(si::Capability::batched_prefill)
                .add(si::Capability::prefix_reuse)
                .add(si::Capability::speculative_decode)
                .add(si::Capability::structured_output)
                .add(si::Capability::embeddings)
                .add(si::Capability::deterministic)
                .add(si::Capability::recurrent_checkpoint)
                .add(si::Capability::logprobs);
            if (enable_multi_slot) caps.add(si::Capability::continuous_batch_decode);
            if (enable_multimodal) caps.add(si::Capability::multimodal);
            return caps;
        }
        si::Result<std::string> probe() override { return std::string("llama.cpp test"); }
        si::Result<std::vector<si::ModelDescriptor>> list_models() override {
            return std::vector<si::ModelDescriptor>{model->descriptor()};
        }
        si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
            return std::static_pointer_cast<si::BackendModel>(model);
        }
    };

    auto be = std::make_shared<FakeDirectLlamaCppBackend>();
    be->enable_multimodal = true;
    be->enable_multi_slot = true;
    auto opts = Fixture::defaults();
    opts.models = {"direct:q4"};
    opts.backend.backend.clear();
    opts.backend_instance = be;
    Fixture f(opts);

    auto has_feature = [](const json::Value& doc, std::string_view name) {
        const auto* arr = doc.find("sonder")->find("features");
        if (arr == nullptr || !arr->is_array()) return false;
        for (const auto& item : arr->as_array()) {
            if (item.is_string() && item.as_string() == name) return true;
        }
        return false;
    };

    for (const char* path : {"/v1/sonder/health", "/v1/models"}) {
        CAPTURE(path);
        const Reply r = get(f.port, path);
        REQUIRE_MESSAGE(r.status == 200, r.body);
        const auto doc = r.json();
        CHECK(has_feature(doc, "native_tools"));
        CHECK(has_feature(doc, "responses_api"));
        CHECK(has_feature(doc, "priority_classes"));
        CHECK(has_feature(doc, "structured_output"));
        CHECK(has_feature(doc, "embeddings"));
        CHECK(has_feature(doc, "logprobs"));
        CHECK(has_feature(doc, "multimodal"));
        CHECK(has_feature(doc, "continuous_batch_decode"));
        CHECK(has_feature(doc, "prompt_cache_key"));
        CHECK_FALSE(has_feature(doc, "backend_native_tools"));
    }
}

TEST_CASE("responses parity: timings, logprobs, reasoning_budget, admission telemetry, scope isolation, TTL, disable") {
    // 1. Timings, logprobs (non-streaming, include[], and streaming), and reasoning_budget warnings on mock.
    {
        Fixture f;
        // Non-streaming with timings + logprobs + top_logprobs=3 + reasoning_budget warning.
        {
            const Reply r = post(
                f.port, "/v1/responses",
                R"({"input":"hello","max_output_tokens":4,"logprobs":true,"top_logprobs":3,"reasoning_budget_tokens":256,"reasoning_budget_message":"be brief"})");
            REQUIRE_MESSAGE(r.status == 200, r.body);
            const auto doc = r.json();
            const auto* timings = doc.find("timings");
            REQUIRE((timings != nullptr && timings->is_object()));
            CHECK(timings->find("prompt_n")->as_uint() > 0u);
            CHECK(timings->find("predicted_n")->as_uint() == 4u);
            CHECK(timings->find("total_ms")->as_double() >= 0.0);

            const auto& content0 = doc.find("output")->as_array()[0].find("content")->as_array()[0];
            const auto* lp = content0.find("logprobs");
            REQUIRE((lp != nullptr && lp->is_array()));
            CHECK(lp->as_array().size() == 4u);
            for (const auto& tok : lp->as_array()) {
                CHECK_FALSE(tok.find("token")->as_string().empty());
                CHECK(tok.find("logprob")->as_double() < 0.0);
                CHECK_FALSE(tok.find("bytes")->as_array().empty());
                CHECK(tok.find("top_logprobs")->as_array().size() == 3u);
            }

            const auto* warnings = doc.find("sonder")->find("warnings");
            REQUIRE((warnings != nullptr && warnings->is_array()));
            REQUIRE(warnings->as_array().size() == 2u);
            CHECK(warnings->as_array()[0].as_string().find("reasoning_budget_tokens") != std::string::npos);
            CHECK(warnings->as_array()[1].as_string().find("reasoning_budget_message") != std::string::npos);
        }

        // Responses-native include: ["message.output_text.logprobs"] + top_logprobs=2.
        {
            const Reply r = post(
                f.port, "/v1/responses",
                R"({"input":"hello","max_output_tokens":3,"include":["message.output_text.logprobs"],"top_logprobs":2})");
            REQUIRE_MESSAGE(r.status == 200, r.body);
            const auto doc = r.json();
            const auto& content0 = doc.find("output")->as_array()[0].find("content")->as_array()[0];
            const auto* lp = content0.find("logprobs");
            REQUIRE((lp != nullptr && lp->is_array()));
            CHECK(lp->as_array().size() == 3u);
            CHECK(lp->as_array()[0].find("top_logprobs")->as_array().size() == 2u);
        }

        // Streaming with logprobs: delta events carry logprobs and response.completed carries timings + logprobs.
        {
            const Reply r = post(
                f.port, "/v1/responses",
                R"({"input":"hello","max_output_tokens":3,"stream":true,"logprobs":true,"top_logprobs":2})");
            REQUIRE_MESSAGE(r.status == 200, r.body);
            const auto frames = parse_named_sse(r.body);
            std::size_t delta_lp_count = 0;
            bool saw_completed_timings = false;
            bool saw_completed_logprobs = false;
            for (const auto& [ev, data] : frames) {
                auto p = json::parse(data);
                REQUIRE(p.ok());
                if (ev == "response.output_text.delta") {
                    const auto* lp = p.value().find("logprobs");
                    REQUIRE((lp != nullptr && lp->is_array()));
                    if (!lp->as_array().empty()) {
                        ++delta_lp_count;
                        CHECK(lp->as_array()[0].find("top_logprobs")->as_array().size() == 2u);
                    }
                }
                if (ev == "response.completed") {
                    const auto* resp = p.value().find("response");
                    REQUIRE(resp != nullptr);
                    const auto* timings = resp->find("timings");
                    if (timings != nullptr && timings->is_object() && timings->find("predicted_n")->as_uint() == 3u) {
                        saw_completed_timings = true;
                    }
                    const auto* lp = resp->find("output")->as_array()[0].find("content")->as_array()[0].find("logprobs");
                    if (lp != nullptr && lp->is_array() && lp->as_array().size() == 3u) {
                        saw_completed_logprobs = true;
                    }
                }
            }
            CHECK(delta_lp_count == 3u);
            CHECK(saw_completed_timings);
            CHECK(saw_completed_logprobs);
        }
    }

    // 2. Backend without Capability::logprobs rejects /v1/responses logprobs with 400 unsupported_parameter.
    {
        auto backend = std::make_shared<CapturingBackend>();
        Fixture cap(capturing_options(backend));
        const Reply r = post(cap.port, "/v1/responses", R"({"input":"hi","logprobs":true})");
        CHECK(r.status == 400);
        CHECK(r.json().find("error")->find("code")->as_string() == "unsupported_parameter");
        CHECK(r.json().find("error")->find("param")->as_string() == "logprobs");
    }

    // 3. reasoning_budget::apply() forwards budget fields to llamaserver backends (header > body > pin).
    {
        struct BudgetCaptureBackend final : si::Backend {
            struct Model final : si::BackendModel {
                si::ModelDescriptor desc;
                std::mutex mu;
                std::optional<std::int64_t> last_budget;
                std::optional<std::string> last_budget_msg;
                Model() {
                    desc.name = "ls:test";
                    desc.backend = "llamaserver";
                    desc.context_length = 4096;
                }
                const si::ModelDescriptor& descriptor() const override { return desc; }
                bool has_native_chat() const override { return true; }
                si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                                       const si::TokenCallback&) override {
                    return si::Status(si::ErrorCode::unsupported, "unused");
                }
                si::Result<si::GenerateStats> chat(const si::ChatRequest& req, const si::CancellationToken&,
                                                   const si::TokenCallback& on_chunk) override {
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        last_budget = req.reasoning_budget_tokens;
                        last_budget_msg = req.reasoning_budget_message;
                    }
                    if (on_chunk) (void)on_chunk(si::TokenChunk{"ok", 0});
                    si::GenerateStats st;
                    st.prompt_tokens = 3;
                    st.completion_tokens = 1;
                    st.chunks = 1;
                    st.token_counts_from_backend = true;
                    st.stop_reason = si::StopReason::end_of_sequence;
                    return st;
                }
            };
            std::shared_ptr<Model> model = std::make_shared<Model>();
            std::string name() const override { return "llamaserver"; }
            std::string description() const override { return "budget capturing llamaserver"; }
            si::BackendCapabilities capabilities() const override {
                return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::remote_process);
            }
            si::Result<std::string> probe() override { return std::string("ls-1"); }
            si::Result<std::vector<si::ModelDescriptor>> list_models() override {
                return std::vector<si::ModelDescriptor>{model->descriptor()};
            }
            si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
                return std::static_pointer_cast<si::BackendModel>(model);
            }
        };

        auto be = std::make_shared<BudgetCaptureBackend>();
        auto opts = Fixture::defaults();
        opts.models = {"ls:test"};
        opts.backend.backend.clear();
        opts.backend_instance = be;
        opts.pin_reasoning_budget_tokens = 500;
        opts.pin_reasoning_budget_message = "pin-msg";
        Fixture f(opts);

        // Pin default applied when omitted.
        const Reply r1 = post(f.port, "/v1/responses", R"({"input":"hi"})");
        REQUIRE_MESSAGE(r1.status == 200, r1.body);
        {
            std::lock_guard<std::mutex> lock(be->model->mu);
            CHECK(be->model->last_budget == std::optional<std::int64_t>(500));
            CHECK(be->model->last_budget_msg == std::optional<std::string>("pin-msg"));
        }

        // Header X-Sonder-Reasoning-Budget overrides body reasoning_budget_tokens.
        const Reply r2 = post(
            f.port, "/v1/responses",
            R"({"input":"hi","reasoning_budget_tokens":100,"reasoning_budget_message":"body-msg"})",
            {{"X-Sonder-Reasoning-Budget", "42"}});
        REQUIRE_MESSAGE(r2.status == 200, r2.body);
        CHECK(r2.json().find("sonder")->find("warnings") == nullptr);
        {
            std::lock_guard<std::mutex> lock(be->model->mu);
            CHECK(be->model->last_budget == std::optional<std::int64_t>(42));
            CHECK(be->model->last_budget_msg == std::optional<std::string>("body-msg"));
        }
    }

    // 4. Admission-rejection telemetry parity on /v1/responses (429 queue full & 504 deadline expired).
    {
        auto be = std::make_shared<SlowPrefillBackend>();
        auto opts = Fixture::defaults();
        opts.models = {"slow:test"};
        opts.backend.backend.clear();
        opts.backend_instance = be;
        opts.priority_admission = srv::PriorityAdmissionPolicy::on;
        opts.backend_capacity = 1;
        opts.max_queue_per_class = 1;
        Fixture f(opts);

        be->reset();
        Reply hold_reply;
        std::thread t_hold([&] {
            hold_reply = post(f.port, "/v1/responses", R"({"input":"hold slot"})");
        });
        REQUIRE(be->wait_entered());

        Reply queued_reply;
        std::thread t_queued([&] {
            queued_reply = post(f.port, "/v1/responses", R"({"input":"queued bg","priority":"background"})");
        });
        REQUIRE(eventually([&] {
            return get(f.port, "/v1/sonder/health").json().find("queued_by_class")->find("background")->as_uint() == 1;
        }));

        // Queue full -> 429 overloaded + request.failed telemetry.
        const Reply full_reply = post(
            f.port, "/v1/responses", R"({"input":"overflow","priority":"background"})",
            {{"X-Sonder-Parent-Request-Id", "parent-resp-full"}});
        CHECK(full_reply.status == 429);
        CHECK(full_reply.header("retry-after") == "1");
        const std::string full_req_id = full_reply.header("x-sonder-request-id");

        // Queued deadline expiry -> 504 deadline_exceeded + request.failed telemetry.
        const Reply expired_reply = post(
            f.port, "/v1/responses", R"({"input":"short deadline","deadline_ms":60})",
            {{"X-Sonder-Parent-Request-Id", "parent-resp-exp"}});
        CHECK(expired_reply.status == 504);
        const std::string exp_req_id = expired_reply.header("x-sonder-request-id");

        be->release();
        t_hold.join();
        t_queued.join();
        CHECK(hold_reply.status == 200);
        CHECK(queued_reply.status == 200);

        bool saw_full_telemetry = false;
        bool saw_exp_telemetry = false;
        for (const auto& ev : f.of_type("request.failed")) {
            const auto* rid = ev.find("request_id");
            const auto* attrs = ev.find("attributes");
            if (rid == nullptr || attrs == nullptr || !rid->is_string() || !attrs->is_object()) continue;
            if (rid->as_string() == full_req_id) {
                CHECK(attrs->find("kind")->as_string() == "chat");
                CHECK(attrs->find("error_code")->as_string() == "overloaded");
                CHECK_FALSE(attrs->find("backend_started")->as_bool());
                CHECK(attrs->find("priority_class")->as_string() == "background");
                CHECK(attrs->find("admission")->find("priority")->as_string() == "background");
                CHECK(attrs->find("parent_request_id")->as_string() == "parent-resp-full");
                saw_full_telemetry = true;
            }
            if (rid->as_string() == exp_req_id) {
                const std::string ec = attrs->find("error_code")->as_string();
                CHECK((ec == "deadline_exceeded" || ec == "timeout"));
                CHECK(attrs->find("parent_request_id")->as_string() == "parent-resp-exp");
                saw_exp_telemetry = true;
            }
        }
        CHECK(saw_full_telemetry);
        CHECK(saw_exp_telemetry);
        for (const auto& ev : f.of_type("request.cancelled")) {
            const auto* rid = ev.find("request_id");
            if (rid != nullptr && rid->is_string()) {
                CHECK(rid->as_string() != exp_req_id);
            }
        }

        // Running deadline expiry (after backend_started) still cooperatively cancels
        // through the watcher and returns status == "incomplete".
        be->reset();
        const Reply running_exp = post(f.port, "/v1/responses", R"({"input":"slow run","deadline_ms":200})");
        REQUIRE_MESSAGE(running_exp.status == 200, running_exp.body);
        const auto run_doc = running_exp.json();
        REQUIRE(run_doc.find("status") != nullptr);
        CHECK(run_doc.find("status")->as_string() == "incomplete");
        CHECK(be->wait_cancelled());
    }

    // 5. Caller-scope isolation over HTTP (agent/run headers and bearer scopes).
    {
        Fixture f;
        const std::vector<std::pair<std::string, std::string>> caller_a = {
            {"X-Sonder-Run-Id", "run-1"}, {"X-Sonder-Agent-Id", "agent-a"}};
        const std::vector<std::pair<std::string, std::string>> caller_b = {
            {"X-Sonder-Run-Id", "run-1"}, {"X-Sonder-Agent-Id", "agent-b"}};

        const Reply r1 = post(f.port, "/v1/responses", R"({"input":"turn 1","max_output_tokens":4})", caller_a);
        REQUIRE_MESSAGE(r1.status == 200, r1.body);
        const std::string id_a = r1.json().find("id")->as_string();

        // Same caller chains cleanly.
        const Reply r2_ok = post(
            f.port, "/v1/responses",
            R"({"previous_response_id":")" + id_a + R"(","input":"turn 2","max_output_tokens":4})", caller_a);
        CHECK(r2_ok.status == 200);

        // Different agent and unauthenticated/unscoped caller both get uniform 404.
        const Reply r2_other = post(
            f.port, "/v1/responses",
            R"({"previous_response_id":")" + id_a + R"(","input":"steal"})", caller_b);
        CHECK(r2_other.status == 404);
        CHECK(r2_other.json().find("error")->find("code")->as_string() == "not_found");

        const Reply r2_anon = post(
            f.port, "/v1/responses",
            R"({"previous_response_id":")" + id_a + R"(","input":"steal anon"})");
        CHECK(r2_anon.status == 404);

        // Distinct Authorization bearer scopes are also isolated.
        const Reply rt1 = post(f.port, "/v1/responses", R"({"input":"bearer 1","max_output_tokens":3})",
                               {{"Authorization", "Bearer caller-token-1"}});
        REQUIRE(rt1.status == 200);
        const std::string id_tok1 = rt1.json().find("id")->as_string();
        CHECK(post(f.port, "/v1/responses",
                   R"({"previous_response_id":")" + id_tok1 + R"(","input":"ok","max_output_tokens":3})",
                   {{"Authorization", "Bearer caller-token-1"}})
                  .status == 200);
        CHECK(post(f.port, "/v1/responses",
                   R"({"previous_response_id":")" + id_tok1 + R"(","input":"nope"})",
                   {{"Authorization", "Bearer caller-token-2"}})
                  .status == 404);
    }

    // 6. Short TTL expires entries over HTTP.
    {
        auto opts = Fixture::defaults();
        opts.response_history_ttl = std::chrono::milliseconds(80);
        Fixture f(opts);
        const Reply r1 = post(f.port, "/v1/responses", R"({"input":"ephemeral ttl","max_output_tokens":3})");
        REQUIRE(r1.status == 200);
        const std::string id = r1.json().find("id")->as_string();
        std::this_thread::sleep_for(std::chrono::milliseconds(130));
        const Reply r2 = post(f.port, "/v1/responses",
                              R"({"previous_response_id":")" + id + R"(","input":"after ttl"})");
        CHECK(r2.status == 404);
        CHECK(r2.json().find("error")->find("code")->as_string() == "not_found");
    }

    // 7. --disable-response-store opts out cleanly: stateless works, previous_response_id returns 400, health advertises false.
    {
        auto opts = Fixture::defaults();
        opts.disable_response_store = true;
        Fixture f(opts);
        const Reply hr = get(f.port, "/v1/sonder/health");
        REQUIRE(hr.status == 200);
        CHECK_FALSE(hr.json().find("capabilities")->find("response_store")->as_bool());

        const Reply r1 = post(f.port, "/v1/responses", R"({"input":"stateless","store":true,"max_output_tokens":3})");
        REQUIRE_MESSAGE(r1.status == 200, r1.body);
        const std::string id = r1.json().find("id")->as_string();

        const Reply r2 = post(f.port, "/v1/responses",
                              R"({"previous_response_id":")" + id + R"(","input":"chain attempt"})");
        CHECK(r2.status == 400);
        CHECK(r2.json().find("error")->find("code")->as_string() == "unsupported_parameter");
        CHECK(r2.json().find("error")->find("param")->as_string() == "previous_response_id");
    }
}
