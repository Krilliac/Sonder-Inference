// llguidance sampler wiring tests. They use llama.cpp's bundled vocab-only GGUF (no weights, no
// downloads). Cases that need llguidance itself run only in a SONDER_WITH_LLGUIDANCE=ON build; the
// OFF cases pin the "never silently unconstrained" behaviour of the default build.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "json_schema_normalize.hpp"
#include "llguidance_sampler.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#ifdef SONDER_HAS_SERVER
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

#include "server/src/chat_handler.hpp"
#include "server/src/tool_call_parser.hpp"
#include "sonder/inference/engine.hpp"
#endif
#include <doctest/doctest.h>

#ifdef SONDER_LLAMACPP_VOCAB_DIR

using namespace sonder::backends::llamacpp;

namespace {

constexpr const char* kIntObjectSchema =
    R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})";

bool LoadVocab(LlamaCppBackend& be) {
    LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-bpe.gguf";
    opts.vocab_only = true;
    return be.Load(opts).ok();
}

std::size_t CountAllowed(const std::vector<std::uint8_t>& allowed) {
    std::size_t n = 0;
    for (const std::uint8_t a : allowed) n += a != 0 ? 1U : 0U;
    return n;
}

}  // namespace

TEST_CASE("llguidance: availability matches the build option") {
#ifdef SONDER_LLAMACPP_HAS_LLGUIDANCE
    CHECK(LlguidanceAvailable());
#else
    CHECK_FALSE(LlguidanceAvailable());
#endif
}

TEST_CASE("llguidance: a lazy grammar combined with a schema is refused (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    SamplingParams p;
    p.llguidance_json_schema = kIntObjectSchema;
    p.grammar = "root ::= \"x\"";
    p.grammar_lazy = true;
    p.trigger_patterns = {"x"};
    std::vector<std::uint8_t> mask;
    CHECK(be.ProbeGrammarMask(p, {}, mask).code == ErrorCode::kInvalidArgument);
    be.Unload();
}

#ifndef SONDER_LLAMACPP_HAS_LLGUIDANCE
TEST_CASE("llguidance OFF: a schema request fails closed instead of running unconstrained (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    SamplingParams p;
    p.llguidance_json_schema = kIntObjectSchema;
    std::vector<std::uint8_t> mask;
    const Status st = be.ProbeGrammarMask(p, {}, mask);
    CHECK(st.code == ErrorCode::kInvalidArgument);
    CHECK(st.message.find("llguidance") != std::string::npos);

    // With no schema the request is unaffected: no grammar, nothing masked.
    SamplingParams none;
    REQUIRE(be.ProbeGrammarMask(none, {}, mask).ok());
    CHECK(mask.empty() == false);
    be.Unload();
}
#else
TEST_CASE("llguidance ON: the schema constrains the first token (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    SamplingParams p;
    p.llguidance_json_schema = kIntObjectSchema;
    std::vector<std::uint8_t> mask;
    REQUIRE(be.ProbeGrammarMask(p, {}, mask).ok());
    REQUIRE_FALSE(mask.empty());
    CHECK(CountAllowed(mask) > 0);
    CHECK(CountAllowed(mask) < mask.size() / 10);  // only "{" (and whitespace) may start the object

    // After `{"a":` only digits / '-' / whitespace may follow.
    std::vector<Token> prefix;
    REQUIRE(be.Tokenize("{\"a\":", false, prefix).ok());
    std::vector<std::uint8_t> after;
    REQUIRE(be.ProbeGrammarMask(p, prefix, after).ok());
    CHECK(CountAllowed(after) > 0);
    CHECK(CountAllowed(after) < after.size() / 10);
    be.Unload();
}

TEST_CASE("llguidance ON: additionalProperties:false is enforced after normalisation (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    SamplingParams p;
    p.llguidance_json_schema = kIntObjectSchema;
    std::vector<Token> ok_text;
    REQUIRE(be.Tokenize("{\"a\":1}", false, ok_text).ok());
    std::vector<std::uint8_t> mask;
    CHECK(be.ProbeGrammarMask(p, ok_text, mask).ok());

    // An extra key is not a valid continuation: llguidance rejects the token, and the sampler
    // reports it as a failure instead of dropping the constraint.
    std::vector<Token> extra;
    REQUIRE(be.Tokenize("{\"a\":1,\"b\":2}", false, extra).ok());
    Status st = Status::Ok();
    CHECK_NOTHROW(st = be.ProbeGrammarMask(p, extra, mask));
    CHECK(st.code == ErrorCode::kDecodeFailed);
    be.Unload();
}

TEST_CASE("llguidance ON: normalisation is what closes an open object (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    const std::string open_schema = R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})";
    std::vector<Token> extra;
    REQUIRE(be.Tokenize("{\"a\":1,\"b\":2}", false, extra).ok());
    std::vector<std::uint8_t> mask;

    // Raw schema: llguidance honours the open object, so the extra key is accepted.
    SamplingParams raw;
    raw.llguidance_json_schema = open_schema;
    CHECK(be.ProbeGrammarMask(raw, extra, mask).ok());

    // The same schema after NormalizeJsonSchemaText: the extra key is refused.
    auto normalized = sonder::inference::llamacpp_detail::NormalizeJsonSchemaText(open_schema);
    REQUIRE(normalized.ok());
    SamplingParams closed;
    closed.llguidance_json_schema = normalized.value();
    CHECK(be.ProbeGrammarMask(closed, extra, mask).code == ErrorCode::kDecodeFailed);
    be.Unload();
}

TEST_CASE("llguidance ON: an invalid schema is kInvalidArgument (vocab-only)") {
    LlamaCppBackend be;
    REQUIRE(LoadVocab(be));
    SamplingParams p;
    p.llguidance_json_schema = R"({"type":"definitely-not-a-type"})";
    std::vector<std::uint8_t> mask;
    CHECK(be.ProbeGrammarMask(p, {}, mask).code == ErrorCode::kInvalidArgument);
    p.llguidance_json_schema = "not json at all";
    CHECK(be.ProbeGrammarMask(p, {}, mask).code == ErrorCode::kInvalidArgument);
    be.Unload();
}
#endif  // SONDER_LLAMACPP_HAS_LLGUIDANCE

#endif  // SONDER_LLAMACPP_VOCAB_DIR

#ifdef SONDER_HAS_SERVER
namespace {

sonder::inference::server::detail::ChatJob MakeSchemaJob(std::string_view schema_text, bool set_pending = true) {
    namespace srv = sonder::inference::server::detail;
    srv::ChatJob job;
    job.response_format_type = "json_schema";
    job.response_format_schema = std::string(schema_text);
    if (set_pending) {
        auto parsed = sonder::inference::json::parse(schema_text);
        REQUIRE(parsed.ok());
        job.pending_response_schema = std::move(parsed).value();
    }
    return job;
}

}  // namespace

#ifndef SONDER_LLAMACPP_HAS_LLGUIDANCE
TEST_CASE("llguidance OFF: prepare_chat_grammar keeps GBNF compilation byte-for-byte unchanged") {
    namespace srv = sonder::inference::server::detail;
    const std::string raw_schema =
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})";
    srv::ChatJob job = MakeSchemaJob(raw_schema);
    srv::CompiledGrammarCache cache;

    const auto err = srv::prepare_chat_grammar(job, "llamacpp:test", /*backend_supports_structured_output=*/true,
                                               /*backend_native_tools=*/false, "llamacpp", &cache);
    CHECK_FALSE(err.has_value());
    REQUIRE(job.compiled_grammar != nullptr);
    CHECK_FALSE(job.grammar.empty());
    CHECK(job.grammar == job.compiled_grammar->gbnf);
    CHECK(job.response_format_schema == raw_schema);  // untouched when option is OFF
    CHECK(cache.stats().misses == 1u);

    // Compare byte-for-byte against direct resolve_structured_grammar.
    auto direct = srv::resolve_structured_grammar(
        "response_format.json_schema.schema", "", job.response_format_type, job.pending_response_schema,
        job.response_format_schema, {}, std::nullopt, true, false, "llamacpp", "llamacpp:test", nullptr);
    REQUIRE(direct.ok());
    CHECK(job.grammar == direct.value().grammar);

    // Schemas GBNF cannot compile (minLength > 32, allOf) are still rejected with 400 when OFF.
    srv::ChatJob long_min = MakeSchemaJob(
        R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"]})");
    const auto err_min = srv::prepare_chat_grammar(long_min, "llamacpp:test", true, false, "llamacpp", &cache);
    REQUIRE(err_min.has_value());
    CHECK(err_min->status == 400);
    CHECK(err_min->code == "invalid_request");

    srv::ChatJob allof_job = MakeSchemaJob(R"({"allOf":[{"type":"object","properties":{"a":{"type":"string"}}}]})");
    const auto err_allof = srv::prepare_chat_grammar(allof_job, "llamacpp:test", true, false, "llamacpp", &cache);
    REQUIRE(err_allof.has_value());
    CHECK(err_allof->status == 400);
    CHECK(err_allof->code == "invalid_request");
}
#else
TEST_CASE("llguidance ON: prepare_chat_grammar hands normalised schema to llamacpp without compiling GBNF") {
    namespace srv = sonder::inference::server::detail;
    srv::CompiledGrammarCache cache;

    // minLength: 40 exceeds GBNF's 32-char limit, so GBNF compilation would fail with 400.
    // On llamacpp with llguidance ON, prepare_chat_grammar skips GBNF and normalises the schema.
    srv::ChatJob job = MakeSchemaJob(
        R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"]})");
    const auto err = srv::prepare_chat_grammar(job, "llamacpp:test", /*backend_supports_structured_output=*/true,
                                               /*backend_native_tools=*/false, "llamacpp", &cache);
    CHECK_FALSE(err.has_value());
    CHECK(job.compiled_grammar == nullptr);
    CHECK(job.grammar.empty());
    CHECK(cache.stats().misses == 0u);
    CHECK(cache.stats().inserts == 0u);
    REQUIRE(job.response_format_schema.has_value());
    CHECK(*job.response_format_schema ==
          R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"],"additionalProperties":false})");

    // allOf (also rejected by GBNF) passes through; nested objects inside branches are closed.
    srv::ChatJob allof_job =
        MakeSchemaJob(R"({"allOf":[{"type":"object","properties":{"n":{"type":"object"}}}]})");
    const auto err_allof = srv::prepare_chat_grammar(allof_job, "llamacpp:test", true, false, "llamacpp", &cache);
    CHECK_FALSE(err_allof.has_value());
    CHECK(allof_job.grammar.empty());
    CHECK(allof_job.compiled_grammar == nullptr);
    REQUIRE(allof_job.response_format_schema.has_value());
    CHECK(*allof_job.response_format_schema ==
          R"({"allOf":[{"type":"object","properties":{"n":{"type":"object","additionalProperties":false}}}]})");

    // Text-only response_format_schema (no pending_response_schema) is normalised too.
    srv::ChatJob text_only = MakeSchemaJob(R"({"type":"object","properties":{"x":{"type":"integer"}}})", false);
    const auto err_text = srv::prepare_chat_grammar(text_only, "llamacpp:test", true, false, "llamacpp", &cache);
    CHECK_FALSE(err_text.has_value());
    CHECK(text_only.grammar.empty());
    REQUIRE(text_only.response_format_schema.has_value());
    CHECK(*text_only.response_format_schema ==
          R"({"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false})");

    // Non-llamacpp backend still uses GBNF: rejects minLength: 40 and compiles normal schemas.
    srv::ChatJob other_reject = MakeSchemaJob(
        R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"]})");
    const auto err_other = srv::prepare_chat_grammar(other_reject, "ollama:test", true, false, "ollama", &cache);
    REQUIRE(err_other.has_value());
    CHECK(err_other->status == 400);

    srv::ChatJob other_ok = MakeSchemaJob(
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})");
    const auto err_other_ok = srv::prepare_chat_grammar(other_ok, "ollama:test", true, false, "ollama", &cache);
    CHECK_FALSE(err_other_ok.has_value());
    CHECK_FALSE(other_ok.grammar.empty());
    CHECK(other_ok.compiled_grammar != nullptr);

    // json_object on llamacpp keeps the GBNF path.
    srv::ChatJob json_obj;
    json_obj.response_format_type = "json_object";
    const auto err_obj = srv::prepare_chat_grammar(json_obj, "llamacpp:test", true, false, "llamacpp", &cache);
    CHECK_FALSE(err_obj.has_value());
    CHECK_FALSE(json_obj.grammar.empty());
    CHECK(json_obj.compiled_grammar != nullptr);
}

TEST_CASE("llguidance ON: prepare_chat_grammar rejects invalid or hostile schemas with 400") {
    namespace srv = sonder::inference::server::detail;
    srv::CompiledGrammarCache cache;

    // Missing schema.
    srv::ChatJob missing;
    missing.response_format_type = "json_schema";
    const auto err_missing = srv::prepare_chat_grammar(missing, "llamacpp:test", true, false, "llamacpp", &cache);
    REQUIRE(err_missing.has_value());
    CHECK(err_missing->status == 400);
    CHECK(err_missing->code == "invalid_request");
    CHECK(err_missing->param == std::optional<std::string>("response_format.json_schema.schema"));

    // Non-object text schema (boolean).
    srv::ChatJob bool_schema = MakeSchemaJob("true", false);
    const auto err_bool = srv::prepare_chat_grammar(bool_schema, "llamacpp:test", true, false, "llamacpp", &cache);
    REQUIRE(err_bool.has_value());
    CHECK(err_bool->status == 400);
    CHECK(err_bool->code == "invalid_request");

    // Hostile depth > 64.
    std::string deep = R"({"type":"object"})";
    for (int i = 0; i < 65; ++i) deep = R"({"type":"object","properties":{"a":)" + deep + "}}";
    srv::ChatJob deep_job = MakeSchemaJob(deep, false);
    const auto err_deep = srv::prepare_chat_grammar(deep_job, "llamacpp:test", true, false, "llamacpp", &cache);
    REQUIRE(err_deep.has_value());
    CHECK(err_deep->status == 400);
    CHECK(err_deep->code == "invalid_request");
}

namespace {

class FakeLlamacppChatBackend final : public sonder::inference::Backend {
public:
    struct Model final : public sonder::inference::BackendModel {
        sonder::inference::ModelDescriptor desc;
        bool native_chat = false;
        std::mutex mu;
        std::vector<std::string> generate_grammars;
        std::vector<std::string> chat_grammars;
        std::vector<std::optional<std::string>> chat_rf_schemas;

        explicit Model(bool native) : native_chat(native) {
            desc.name = "fake:m";
            desc.backend = "llamacpp";
            desc.format = "gguf";
            desc.context_length = 4096;
        }

        const sonder::inference::ModelDescriptor& descriptor() const override { return desc; }
        bool has_native_chat() const override { return native_chat; }

        sonder::inference::Result<sonder::inference::GenerateStats> generate(
            const sonder::inference::GenerateRequest& req, const sonder::inference::CancellationToken&,
            const sonder::inference::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                generate_grammars.push_back(req.grammar);
            }
            if (on_chunk) (void)on_chunk(sonder::inference::TokenChunk{R"({"a":"ok"})", 0});
            sonder::inference::GenerateStats st;
            st.prompt_tokens = 2;
            st.completion_tokens = 2;
            st.chunks = 1;
            st.token_counts_from_backend = true;
            st.stop_reason = sonder::inference::StopReason::end_of_sequence;
            return st;
        }

        sonder::inference::Result<sonder::inference::GenerateStats> chat(
            const sonder::inference::ChatRequest& req, const sonder::inference::CancellationToken&,
            const sonder::inference::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                chat_grammars.push_back(req.grammar);
                chat_rf_schemas.push_back(req.response_format_schema);
            }
            if (on_chunk) (void)on_chunk(sonder::inference::TokenChunk{R"({"a":"ok"})", 0});
            sonder::inference::GenerateStats st;
            st.prompt_tokens = 2;
            st.completion_tokens = 2;
            st.chunks = 1;
            st.token_counts_from_backend = true;
            st.stop_reason = sonder::inference::StopReason::end_of_sequence;
            return st;
        }
    };

    explicit FakeLlamacppChatBackend(bool native_chat) : model(std::make_shared<Model>(native_chat)) {}

    std::string name() const override { return "llamacpp"; }
    std::string description() const override { return "fake llamacpp backend for llguidance fallback test"; }
    sonder::inference::BackendCapabilities capabilities() const override {
        return sonder::inference::BackendCapabilities{}
            .add(sonder::inference::Capability::streaming)
            .add(sonder::inference::Capability::structured_output);
    }
    sonder::inference::Result<std::string> probe() override { return std::string("fake"); }
    sonder::inference::Result<std::vector<sonder::inference::ModelDescriptor>> list_models() override {
        return std::vector<sonder::inference::ModelDescriptor>{model->descriptor()};
    }
    sonder::inference::Result<std::shared_ptr<sonder::inference::BackendModel>> load_model(
        const sonder::inference::ModelLoadOptions&) override {
        return std::static_pointer_cast<sonder::inference::BackendModel>(model);
    }

    std::shared_ptr<Model> model;
};

struct ChatRunOutcome {
    int http_status = 0;
    std::string error_code;
};

ChatRunOutcome RunExecuteChat(const std::shared_ptr<FakeLlamacppChatBackend>& backend, std::string_view schema_text,
                              sonder::inference::server::detail::CompiledGrammarCache* cache) {
    namespace si = sonder::inference;
    namespace srv = sonder::inference::server::detail;

    srv::ensure_network();
    bool in_use = false;
    auto listener = srv::listen_tcp("127.0.0.1", 0, in_use);
    REQUIRE(listener.ok());
    auto client = srv::connect_tcp("127.0.0.1", listener.value().port, std::chrono::milliseconds(1000));
    REQUIRE(client.ok());
    srv::Accepted accepted{};
    for (int i = 0; i < 20 && accepted.status != srv::AcceptStatus::accepted; ++i) {
        (void)srv::wait_socket(listener.value().socket.get(), false, std::chrono::milliseconds(50), nullptr);
        accepted = srv::accept_client(listener.value().socket);
    }
    REQUIRE(accepted.status == srv::AcceptStatus::accepted);

    si::Engine engine;
    REQUIRE(engine.register_backend(backend).ok());
    auto model_res = engine.load_model("llamacpp", si::ModelLoadOptions{"fake:m"});
    REQUIRE(model_res.ok());
    auto sess_res = engine.create_session(model_res.value());
    REQUIRE(sess_res.ok());

    srv::ChatJob job = MakeSchemaJob(schema_text);
    job.model = "fake:m";
    job.messages = {si::ChatMessage{"user", "hi"}};
    const auto prep_err = srv::prepare_chat_grammar(job, "llamacpp:fake:m", true, false, "llamacpp", cache);
    REQUIRE_FALSE(prep_err.has_value());

    si::server::ServerOptions opts;
    std::string req_id;
    int status = 0;
    std::atomic<bool> hard_stop{false};
    srv::PriorityAdmission admission;
    ChatRunOutcome outcome;
    srv::ChatExchange ex{
        opts,
        accepted.socket.get(),
        req_id,
        status,
        /*synthetic=*/false,
        std::chrono::steady_clock::now(),
        admission,
        hard_stop,
        engine.telemetry(),
        /*leave_inflight=*/[] {},
        /*write=*/[](std::string_view) { return true; },
        /*start_stream=*/[] { return true; },
        /*send_json=*/[&](int code, const si::json::Object&) { outcome.http_status = code; },
        /*send_error=*/[&](const srv::ApiError& e) {
            outcome.http_status = e.status;
            outcome.error_code = e.code;
        },
        /*watcher_failure=*/[] {},
    };
    srv::Correlation corr;
    srv::execute_chat(ex, job, corr, "fake:m", "llamacpp", sess_res.value(), srv::ChatProtocol::openai,
                      /*backend_native_tools=*/false, cache, "llamacpp:fake:m",
                      /*backend_supports_structured_output=*/true);
    return outcome;
}

}  // namespace

TEST_CASE("llguidance ON: execute_chat never runs unconstrained when has_native_chat() is false") {
    namespace srv = sonder::inference::server::detail;
    const std::string gbnf_ok_schema =
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})";
    const std::string gbnf_unsupported_schema =
        R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"]})";

    // Both with and without CompiledGrammarCache, a template-less llamacpp model (has_native_chat() == false)
    // must enforce GBNF on generate() or fail closed with 400 — never run unconstrained.
    for (bool with_cache : {true, false}) {
        srv::CompiledGrammarCache cache;
        srv::CompiledGrammarCache* cache_ptr = with_cache ? &cache : nullptr;

        auto be_ok = std::make_shared<FakeLlamacppChatBackend>(/*native_chat=*/false);
        const ChatRunOutcome ok_out = RunExecuteChat(be_ok, gbnf_ok_schema, cache_ptr);
        CHECK(ok_out.http_status == 200);
        REQUIRE(be_ok->model->generate_grammars.size() == 1u);
        CHECK_FALSE(be_ok->model->generate_grammars[0].empty());
        CHECK(be_ok->model->chat_grammars.empty());

        auto be_bad = std::make_shared<FakeLlamacppChatBackend>(/*native_chat=*/false);
        const ChatRunOutcome bad_out = RunExecuteChat(be_bad, gbnf_unsupported_schema, cache_ptr);
        CHECK(bad_out.http_status == 400);
        CHECK(bad_out.error_code == "invalid_request");
        CHECK(be_bad->model->generate_grammars.empty());
        CHECK(be_bad->model->chat_grammars.empty());
    }

    // Contrast: when has_native_chat() is true, execute_chat hands the normalised schema straight to chat()
    // without compiling GBNF, so minLength: 40 succeeds on llguidance.
    srv::CompiledGrammarCache cache;
    auto be_native = std::make_shared<FakeLlamacppChatBackend>(/*native_chat=*/true);
    const ChatRunOutcome native_out = RunExecuteChat(be_native, gbnf_unsupported_schema, &cache);
    CHECK(native_out.http_status == 200);
    CHECK(be_native->model->generate_grammars.empty());
    REQUIRE(be_native->model->chat_grammars.size() == 1u);
    CHECK(be_native->model->chat_grammars[0].empty());
    REQUIRE(be_native->model->chat_rf_schemas.size() == 1u);
    REQUIRE(be_native->model->chat_rf_schemas[0].has_value());
    CHECK(*be_native->model->chat_rf_schemas[0] ==
          R"({"type":"object","properties":{"a":{"type":"string","minLength":40}},"required":["a"],"additionalProperties":false})");
}
#endif  // SONDER_LLAMACPP_HAS_LLGUIDANCE
#endif  // SONDER_HAS_SERVER
