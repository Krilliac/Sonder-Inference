// Unit & HTTP server integration tests for grammar-engine abstraction, bounded
// compiled-grammar LRU cache, per-request GrammarMatcher isolation, Chat +
// Responses parity, tool-call grammar caching, and honest LLGUIDANCE reporting.
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "src/json_schema_gbnf.hpp"
#include "src/tool_call_parser.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

class RecordingGrammarBackend final : public si::Backend {
public:
    struct Model final : public si::BackendModel {
        si::ModelDescriptor desc;
        std::mutex mu;
        std::vector<std::string> recorded_grammars;
        std::vector<std::optional<std::string>> recorded_rf_types;
        std::vector<std::optional<std::string>> recorded_rf_schemas;

        Model(std::string name, std::string family, std::uint64_t size_bytes) {
            desc.name = std::move(name);
            desc.backend = "recording-grammar";
            desc.format = "gguf";
            desc.family = std::move(family);
            desc.size_bytes = size_bytes;
            desc.context_length = 4096;
        }

        const si::ModelDescriptor& descriptor() const override { return desc; }
        bool has_native_chat() const override { return true; }

        si::Result<si::GenerateStats> generate(const si::GenerateRequest& req, const si::CancellationToken&,
                                               const si::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                recorded_grammars.push_back(req.grammar);
            }
            if (on_chunk) {
                (void)on_chunk(si::TokenChunk{R"({"ok":true})", 0});
            }
            si::GenerateStats st;
            st.prompt_tokens = 4;
            st.completion_tokens = 2;
            st.chunks = 1;
            st.token_counts_from_backend = true;
            st.stop_reason = si::StopReason::end_of_sequence;
            return st;
        }

        si::Result<si::GenerateStats> chat(const si::ChatRequest& req, const si::CancellationToken&,
                                           const si::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(mu);
                recorded_grammars.push_back(req.grammar);
                recorded_rf_types.push_back(req.response_format_type);
                recorded_rf_schemas.push_back(req.response_format_schema);
            }
            const std::string reply = req.tools.empty()
                ? R"({"name":"ada","age":42})"
                : R"(<tool_call>{"name":"lookup","arguments":{"query":"weather"}}</tool_call>)";
            if (on_chunk) {
                (void)on_chunk(si::TokenChunk{reply, 0});
            }
            si::GenerateStats st;
            st.prompt_tokens = 6;
            st.completion_tokens = 4;
            st.chunks = 1;
            st.token_counts_from_backend = true;
            st.stop_reason = si::StopReason::end_of_sequence;
            return st;
        }
    };

    RecordingGrammarBackend() {
        models_.emplace("alpha:7b", std::make_shared<Model>("alpha:7b", "qwen2", 1024u));
        models_.emplace("beta:8b", std::make_shared<Model>("beta:8b", "llama", 2048u));
    }

    std::string name() const override { return "recording-grammar"; }
    std::string description() const override { return "structured-output test backend without native tools"; }

    si::BackendCapabilities capabilities() const override {
        return si::BackendCapabilities{}.add(si::Capability::streaming).add(si::Capability::structured_output);
    }

    si::Result<std::string> probe() override { return std::string("rec-1"); }

    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        std::vector<si::ModelDescriptor> out;
        for (const auto& kv : models_) {
            out.push_back(kv.second->descriptor());
        }
        return out;
    }

    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& opts) override {
        if (auto it = models_.find(opts.model); it != models_.end()) {
            return std::static_pointer_cast<si::BackendModel>(it->second);
        }
        return std::static_pointer_cast<si::BackendModel>(models_.begin()->second);
    }

    std::map<std::string, std::shared_ptr<Model>> models_;
};

srv::ServerOptions multi_model_grammar_options(const std::shared_ptr<RecordingGrammarBackend>& backend) {
    auto o = Fixture::defaults();
    o.models = {"alpha:7b", "beta:8b"};
    o.backend.backend.clear();
    o.backend_instance = backend;
    return o;
}

json::Value parse_obj(std::string_view text) {
    auto parsed = json::parse(text);
    REQUIRE(parsed.ok());
    return std::move(parsed).value();
}

}  // namespace

TEST_CASE("grammar cache: canonical normalization cache hit and tokenizer or schema miss isolation") {
    det::CompiledGrammarCache cache;
    const auto& engine = det::default_structured_grammar_engine();

    const json::Value schema_a = parse_obj(R"({
        "type": "object",
        "$defs": {"Item": {"type": "integer", "minimum": 1, "maximum": 100}},
        "properties": {"b": {"$ref": "#/$defs/Item"}, "a": {"type": "string"}},
        "required": ["a", "b"],
        "additionalProperties": false
    })");
    const json::Value schema_b = parse_obj(R"({
        "additionalProperties": false,
        "required": ["a", "b"],
        "properties": {"a": {"type": "string"}, "b": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"maximum": 100, "minimum": 1, "type": "integer"}},
        "type": "object"
    })");

    CHECK(det::canonical_json_dump(schema_a) == det::canonical_json_dump(schema_b));

    si::ModelDescriptor desc_alpha;
    desc_alpha.name = "alpha:7b";
    desc_alpha.backend = "llamacpp";
    desc_alpha.family = "qwen2";
    desc_alpha.size_bytes = 1000;
    const std::string tok_alpha = det::build_tokenizer_identity_key("llamacpp", "alpha:7b", &desc_alpha, 151936u);

    si::ModelDescriptor desc_beta = desc_alpha;
    desc_beta.name = "beta:8b";
    desc_beta.family = "llama";
    const std::string tok_beta = det::build_tokenizer_identity_key("llamacpp", "beta:8b", &desc_beta, 128256u);

    auto r1 = cache.get_or_compile_schema(engine, schema_a, tok_alpha);
    REQUIRE_MESSAGE(r1.ok(), r1.status().message());
    auto r2 = cache.get_or_compile_schema(engine, schema_b, tok_alpha);
    REQUIRE_MESSAGE(r2.ok(), r2.status().message());
    CHECK(r1.value().get() == r2.value().get());
    CHECK(r1.value()->normalized_schema_hash == r2.value()->normalized_schema_hash);
    CHECK(r1.value()->tokenizer_identity == tok_alpha);
    CHECK_FALSE(r1.value()->gbnf.empty());
    CHECK(r1.value()->payload == r1.value()->gbnf);
    CHECK(r1.value()->capabilities.sampler_linked);

    auto st = cache.stats();
    CHECK(st.misses == 1u);
    CHECK(st.hits == 1u);
    CHECK(st.inserts == 1u);

    // Distinct tokenizer identity misses cleanly even for the exact same schema.
    auto r3 = cache.get_or_compile_schema(engine, schema_a, tok_beta);
    REQUIRE(r3.ok());
    CHECK(r3.value().get() != r1.value().get());
    CHECK(r3.value()->tokenizer_identity == tok_beta);
    CHECK(cache.stats().misses == 2u);
    CHECK(cache.stats().inserts == 2u);

    // Distinct compile options miss cleanly.
    det::GrammarCompileOptions relaxed;
    relaxed.coerce_unsupported_tool_args = true;
    auto r4 = cache.get_or_compile_schema(engine, schema_a, tok_alpha, relaxed);
    REQUIRE(r4.ok());
    CHECK(r4.value().get() != r1.value().get());
    CHECK(cache.stats().misses == 3u);

    // json_object stays keyed under GrammarConstraintKind::json_object, distinct from {"type":"object"}.
    auto robj = cache.get_or_compile_json_object(engine, tok_alpha);
    REQUIRE(robj.ok());
    auto rplain = cache.get_or_compile_schema(engine, parse_obj(R"({"type":"object"})"), tok_alpha);
    REQUIRE(rplain.ok());
    CHECK(robj.value().get() != rplain.value().get());
    CHECK(robj.value()->constraint_kind == det::GrammarConstraintKind::json_object);
    CHECK(rplain.value()->constraint_kind == det::GrammarConstraintKind::json_schema);
}

TEST_CASE("grammar cache: bounded LRU eviction by entries and byte caps plus oversize bypass") {
    const auto& engine = det::default_structured_grammar_engine();
    const json::Value s1 = parse_obj(R"({"type":"object","properties":{"k1":{"type":"string"}},"additionalProperties":false})");
    const json::Value s2 = parse_obj(R"({"type":"object","properties":{"k2":{"type":"integer"}},"additionalProperties":false})");
    const json::Value s3 = parse_obj(R"({"type":"object","properties":{"k3":{"type":"boolean"}},"additionalProperties":false})");

    // 1. Entry cap = 2 with MRU promotion.
    {
        det::CompiledGrammarCache cache(2u, std::size_t{16} << 20, std::size_t{1} << 20);
        auto a1 = cache.get_or_compile_schema(engine, s1, "tok");
        auto a2 = cache.get_or_compile_schema(engine, s2, "tok");
        REQUIRE((a1.ok() && a2.ok()));
        // Touch s1 so s2 becomes the LRU victim.
        auto a1_hit = cache.get_or_compile_schema(engine, s1, "tok");
        REQUIRE(a1_hit.ok());
        CHECK(a1_hit.value().get() == a1.value().get());

        auto a3 = cache.get_or_compile_schema(engine, s3, "tok");
        REQUIRE(a3.ok());
        CHECK(cache.stats().entries == 2u);
        CHECK(cache.stats().evictions == 1u);

        // s1 survived; s2 was evicted and must miss.
        CHECK(cache.get_or_compile_schema(engine, s1, "tok").value().get() == a1.value().get());
        CHECK(cache.stats().hits == 2u);
        CHECK(cache.get_or_compile_schema(engine, s2, "tok").value().get() != a2.value().get());
        CHECK(cache.stats().evictions == 2u);
    }

    // 2. Total byte cap evicts older entries, while oversize single entry bypasses cleanly.
    {
        auto compiled_sample = engine.compile_json_schema(s1);
        REQUIRE(compiled_sample.ok());
        const std::size_t one_entry_bytes = compiled_sample.value().approx_bytes();
        // Fit 1 entry comfortably, reject 2 simultaneous entries, cap per-entry tightly for big schemas.
        det::CompiledGrammarCache cache(16u, one_entry_bytes + 512u, one_entry_bytes + 512u);
        REQUIRE(cache.get_or_compile_schema(engine, s1, "tok").ok());
        REQUIRE(cache.get_or_compile_schema(engine, s2, "tok").ok());
        const auto st = cache.stats();
        CHECK(st.entries == 1u);
        CHECK(st.used_bytes <= st.max_bytes);
        CHECK(st.evictions >= 1u);

        // Build a valid schema whose GBNF exceeds max_entry_bytes.
        json::Object big_props;
        json::Array big_req;
        for (int i = 0; i < 40; ++i) {
            const std::string key = "field_" + std::to_string(i);
            big_props.set(key, json::Object{{"type", "string"}, {"minLength", 1}, {"maxLength", 8}});
            big_req.emplace_back(key);
        }
        const json::Value big_schema(json::Object{{"type", "object"},
                                                  {"properties", std::move(big_props)},
                                                  {"required", std::move(big_req)},
                                                  {"additionalProperties", false}});
        auto big_res = cache.get_or_compile_schema(engine, big_schema, "tok");
        REQUIRE(big_res.ok());
        CHECK(cache.stats().bypassed_oversize == 1u);
        CHECK(cache.stats().entries == 1u);
    }
}

TEST_CASE("grammar cache: thread-safe concurrent compilation and hits") {
    det::CompiledGrammarCache cache(8u, std::size_t{8} << 20, std::size_t{1} << 20);
    const auto& engine = det::default_structured_grammar_engine();
    const std::vector<json::Value> schemas = {
        parse_obj(R"({"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":false})"),
        parse_obj(R"({"additionalProperties":false,"properties":{"x":{"type":"integer"}},"type":"object"})"),
        parse_obj(R"({"type":"object","properties":{"y":{"type":"string"}},"additionalProperties":false})"),
        parse_obj(R"({"type":"object","properties":{"z":{"type":"boolean"}},"additionalProperties":false})"),
    };

    constexpr int kThreads = 8;
    constexpr int kIters = 40;
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            const std::string tok = (t % 2 == 0) ? "tok-a" : "tok-b";
            for (int i = 0; i < kIters; ++i) {
                const auto& s = schemas[static_cast<std::size_t>(i + t) % schemas.size()];
                auto res = cache.get_or_compile_schema(engine, s, tok);
                if (!res.ok() || res.value()->gbnf.empty() || res.value()->tokenizer_identity != tok) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    CHECK(failures.load() == 0);
    const auto st = cache.stats();
    CHECK(st.hits + st.misses == static_cast<std::uint64_t>(kThreads * kIters));
    CHECK(st.hits > 0u);
    CHECK(st.entries <= 8u);
}

TEST_CASE("grammar matcher: per-request mutable state never leaks across shared compiled artifacts") {
    det::CompiledGrammarCache cache;
    const auto& engine = det::default_structured_grammar_engine();
    const json::Value schema = parse_obj(R"({"type":"object","properties":{"k":{"type":"string"}},"additionalProperties":false})");
    auto compiled = cache.get_or_compile_schema(engine, schema, "tok-shared");
    REQUIRE(compiled.ok());

    auto m1_res = engine.create_matcher(compiled.value());
    auto m2_res = engine.create_matcher(compiled.value());
    REQUIRE((m1_res.ok() && m2_res.ok()));
    auto m1 = std::move(m1_res).value();
    auto m2 = std::move(m2_res).value();

    // Unbound accept fails closed.
    CHECK_FALSE(m1->accept_chunk("{}").ok());

    auto g1 = m1->bind_for_request();
    REQUIRE(g1.ok());
    CHECK(m1->active());
    CHECK_FALSE(m2->active());

    REQUIRE(m1->accept_chunk(R"({"k":)").ok());
    REQUIRE(m1->accept_chunk(R"("v"})").ok());
    m1->mark_finished();
    CHECK(m1->finished());
    CHECK(m1->bytes_fed() == 9u);
    CHECK(m1->token_steps() == 2u);

    // Second matcher referencing the same CompiledGrammar stays pristine and independent.
    CHECK_FALSE(m2->finished());
    CHECK(m2->bytes_fed() == 0u);
    auto g2 = m2->bind_for_request();
    REQUIRE(g2.ok());
    CHECK(g2.value() == g1.value());
    REQUIRE(m2->accept_chunk("{}").ok());
    CHECK(m2->bytes_fed() == 2u);
    CHECK(m1->bytes_fed() == 9u);

    m2->reset();
    CHECK_FALSE(m2->active());
    CHECK(m2->bytes_fed() == 0u);
}

TEST_CASE("grammar engine capabilities: honest LLGUIDANCE reporting and explicit unsupported errors") {
    const auto& gbnf = det::gbnf_structured_grammar_engine();
    const auto gc = gbnf.capabilities();
    CHECK(gc.engine == det::GrammarEngineKind::gbnf);
    CHECK(gc.compiled_in);
    CHECK(gc.sampler_linked);
    CHECK(gc.active);
    CHECK(gc.supports_json_object);
    CHECK(gc.supports_json_schema);
    CHECK_FALSE(gc.supports_regex);
    CHECK(gc.supports_regex_subset);
    CHECK_FALSE(gc.supports_structural_tags);

    const auto regex_res = gbnf.compile_regex("^[a-z]+$");
    CHECK_FALSE(regex_res.ok());
    CHECK(regex_res.status().code() == si::ErrorCode::unsupported);

    const auto tag_res = gbnf.compile_structural_tag("<tool>");
    CHECK_FALSE(tag_res.ok());
    CHECK(tag_res.status().code() == si::ErrorCode::unsupported);

    const auto& llg = det::llguidance_structured_grammar_engine();
    const auto lc = llg.capabilities();
    CHECK(lc.engine == det::GrammarEngineKind::llguidance);
    CHECK_FALSE(lc.sampler_linked);
    CHECK_FALSE(lc.active);
    CHECK_FALSE(lc.supports_json_object);
    CHECK_FALSE(lc.supports_json_schema);
    CHECK_FALSE(lc.supports_regex);
    CHECK_FALSE(lc.supports_regex_subset);
    CHECK_FALSE(lc.supports_structural_tags);
    CHECK_FALSE(lc.detail.empty());

    // Unlinked llguidance refuses compilation and matcher creation rather than faking active execution.
    CHECK(llg.compile_json_object().status().code() == si::ErrorCode::unsupported);
    CHECK(llg.compile_json_schema(parse_obj(R"({"type":"object"})")).status().code() == si::ErrorCode::unsupported);
    CHECK(llg.compile_regex("^[a-z]+$").status().code() == si::ErrorCode::unsupported);
    CHECK(llg.compile_structural_tag("tag").status().code() == si::ErrorCode::unsupported);

    auto gbnf_compiled = gbnf.compile_json_object();
    REQUIRE(gbnf_compiled.ok());
    auto gbnf_ptr = std::make_shared<const det::CompiledGrammar>(std::move(gbnf_compiled).value());
    CHECK(llg.create_matcher(gbnf_ptr).status().code() == si::ErrorCode::unsupported);

    det::CompiledGrammar fake_unlinked;
    fake_unlinked.engine = det::GrammarEngineKind::llguidance;
    fake_unlinked.capabilities.engine = det::GrammarEngineKind::llguidance;
    fake_unlinked.capabilities.sampler_linked = false;
    auto fake_ptr = std::make_shared<const det::CompiledGrammar>(std::move(fake_unlinked));
    det::GrammarMatcher unlinked_matcher(fake_ptr);
    CHECK(unlinked_matcher.bind_for_request().status().code() == si::ErrorCode::unsupported);

    // Default active engine must always have sampler_linked == true and active == true.
    CHECK(det::default_structured_grammar_engine().capabilities().active);
    CHECK(det::default_structured_grammar_engine().capabilities().sampler_linked);
}

TEST_CASE("http grammar cache: Chat and Responses share compiled schemas and tool grammars across turns") {
    auto backend = std::make_shared<RecordingGrammarBackend>();
    Fixture f(multi_model_grammar_options(backend));

    // 1. Health & identity report honest grammar engine capability and initial empty cache stats.
    {
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(h.status == 200, h.body);
        const auto hdoc = h.json();
        const auto* ge = hdoc.find("capabilities")->find("grammar_engine");
        REQUIRE((ge != nullptr && ge->is_object()));
        CHECK(ge->find("active_engine")->as_string() == "gbnf");
        CHECK(ge->find("sampler_linked")->as_bool());
        CHECK(ge->find("supports_json_schema")->as_bool());
        CHECK_FALSE(ge->find("supports_regex")->as_bool());
        CHECK(ge->find("supports_regex_subset")->as_bool());
        CHECK_FALSE(ge->find("supports_structural_tags")->as_bool());
        const auto* llg = ge->find("engines")->find("llguidance");
        REQUIRE((llg != nullptr && llg->is_object()));
        CHECK_FALSE(llg->find("sampler_linked")->as_bool());
        CHECK_FALSE(llg->find("active")->as_bool());

        const auto* gc = hdoc.find("grammar_cache");
        REQUIRE((gc != nullptr && gc->is_object()));
        CHECK(gc->find("hits")->as_uint() == 0u);
        CHECK(gc->find("misses")->as_uint() == 0u);
    }

    // 2. Chat compiles schema with key order A -> miss 1; Responses with key order B -> cache hit 1.
    const Reply c1 = post(
        f.port, "/v1/chat/completions",
        R"({"model":"alpha:7b","messages":[{"role":"user","content":"go"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object","required":["name","age"],"properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":false}}}})");
    REQUIRE_MESSAGE(c1.status == 200, c1.body);

    const Reply r1 = post(
        f.port, "/v1/responses",
        R"({"model":"alpha:7b","input":"go",
            "text":{"format":{"type":"json_schema","schema":{"additionalProperties":false,"properties":{"age":{"type":"integer"},"name":{"type":"string"}},"required":["name","age"],"type":"object"}}}})");
    REQUIRE_MESSAGE(r1.status == 200, r1.body);

    {
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(h.status == 200, h.body);
        const auto hdoc = h.json();
        const auto* gc = hdoc.find("grammar_cache");
        CHECK(gc->find("misses")->as_uint() == 1u);
        CHECK(gc->find("hits")->as_uint() == 1u);
        CHECK(gc->find("inserts")->as_uint() == 1u);
    }

    {
        std::lock_guard<std::mutex> lock(backend->models_["alpha:7b"]->mu);
        REQUIRE(backend->models_["alpha:7b"]->recorded_grammars.size() == 2u);
        CHECK_FALSE(backend->models_["alpha:7b"]->recorded_grammars[0].empty());
        CHECK(backend->models_["alpha:7b"]->recorded_grammars[0] ==
              backend->models_["alpha:7b"]->recorded_grammars[1]);
    }

    // 3. Forced tool_choice schema hits CompiledGrammarCache across Chat and Responses.
    const Reply tc1 = post(
        f.port, "/v1/chat/completions",
        R"({"model":"alpha:7b","messages":[{"role":"user","content":"call lookup"}],
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","required":["query"],"properties":{"query":{"type":"string"}},"additionalProperties":false}}}],
            "tool_choice":"required"})");
    REQUIRE_MESSAGE(tc1.status == 200, tc1.body);

    const Reply tr1 = post(
        f.port, "/v1/responses",
        R"({"model":"alpha:7b","input":"call lookup",
            "tools":[{"type":"function","name":"lookup","parameters":{"additionalProperties":false,"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"}}],
            "tool_choice":"required"})");
    REQUIRE_MESSAGE(tr1.status == 200, tr1.body);

    {
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(h.status == 200, h.body);
        const auto hdoc = h.json();
        const auto* gc = hdoc.find("grammar_cache");
        CHECK(gc->find("misses")->as_uint() == 2u);
        CHECK(gc->find("hits")->as_uint() == 2u);
        CHECK(gc->find("inserts")->as_uint() == 2u);
    }

    // 4. Different served model / tokenizer misses cleanly on beta:8b.
    const Reply c_beta = post(
        f.port, "/v1/chat/completions",
        R"({"model":"beta:8b","messages":[{"role":"user","content":"go"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object","required":["name","age"],"properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":false}}}})");
    REQUIRE_MESSAGE(c_beta.status == 200, c_beta.body);
    {
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(h.status == 200, h.body);
        const auto hdoc = h.json();
        const auto* gc = hdoc.find("grammar_cache");
        CHECK(gc->find("misses")->as_uint() == 3u);
        CHECK(gc->find("hits")->as_uint() == 2u);
        CHECK(gc->find("inserts")->as_uint() == 3u);
    }

    // 5. Invalid schema returns 400 invalid_request and tracks compile failure cleanly on both endpoints.
    const Reply bad_chat = post(
        f.port, "/v1/chat/completions",
        R"({"model":"alpha:7b","messages":[{"role":"user","content":"bad"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"bad","schema":{"allOf":[{"type":"object"}]}}}})");
    CHECK(bad_chat.status == 400);
    CHECK(bad_chat.json().find("error")->find("code")->as_string() == "invalid_request");

    const Reply bad_resp = post(
        f.port, "/v1/responses",
        R"({"model":"alpha:7b","input":"bad","text":{"format":{"type":"json_schema","schema":{"allOf":[{"type":"object"}]}}}})");
    CHECK(bad_resp.status == 400);
    CHECK(bad_resp.json().find("error")->find("code")->as_string() == "invalid_request");

    {
        const Reply h = get(f.port, "/v1/sonder/health");
        REQUIRE_MESSAGE(h.status == 200, h.body);
        const auto hdoc = h.json();
        CHECK(hdoc.find("grammar_cache")->find("compile_failures")->as_uint() == 2u);
    }
}
