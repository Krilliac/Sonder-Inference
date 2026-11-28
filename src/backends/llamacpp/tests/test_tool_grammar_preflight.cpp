// Unit tests for tool-schema grammar preflight (bisection, bounded LRU cache,
// key-order-independent canonical hashing, and native llama.cpp vocab-only path).
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "tool_grammar_preflight.hpp"

namespace {

using namespace sonder::inference;
using namespace sonder::inference::llamacpp_detail;

ToolDefinition MakeTool(std::string name, std::string params_json = R"({"type":"object","properties":{"x":{"type":"string"}}})",
                        std::optional<bool> strict = true) {
    ToolDefinition t;
    t.name = std::move(name);
    t.description = "tool " + t.name;
    t.parameters_json = std::move(params_json);
    t.strict = strict;
    return t;
}

std::vector<ToolDefinition> MakeToolset(std::size_t count, std::string_view prefix = "tool_") {
    std::vector<ToolDefinition> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(MakeTool(std::string(prefix) + std::to_string(i)));
    }
    return out;
}

class FakeToolGrammarCompiler final : public ToolGrammarCompiler {
public:
    explicit FakeToolGrammarCompiler(std::unordered_set<std::string> bad_tools = {},
                                     std::size_t interaction_fail_min_size = 0)
        : bad_tools_(std::move(bad_tools)), interaction_fail_min_size_(interaction_fail_min_size) {}

    Status Compile(std::string_view chat_template_id, std::span<const ToolDefinition> tools) const override {
        (void)chat_template_id;
        calls.fetch_add(1, std::memory_order_relaxed);
        for (const auto& t : tools) {
            if (bad_tools_.contains(t.name)) {
                return Status(ErrorCode::invalid_argument, "grammar error in schema for " + t.name);
            }
        }
        if (interaction_fail_min_size_ > 0 && tools.size() >= interaction_fail_min_size_) {
            return Status(ErrorCode::invalid_argument, "combined toolset grammar exceeds rule limit");
        }
        return Status::success();
    }

    void ResetCalls() const noexcept { calls.store(0, std::memory_order_relaxed); }
    std::size_t CallCount() const noexcept { return calls.load(std::memory_order_relaxed); }

    mutable std::atomic<std::size_t> calls{0};

private:
    std::unordered_set<std::string> bad_tools_;
    std::size_t interaction_fail_min_size_ = 0;
};

}  // namespace

TEST_CASE("tool grammar preflight: bisection finds 1 bad tool in toolset of 1") {
    auto compiler = std::make_shared<FakeToolGrammarCompiler>(std::unordered_set<std::string>{"tool_0"});
    ToolGrammarPreflight preflight(compiler);

    const auto tools = MakeToolset(1);
    const auto res = preflight.Preflight("chatml", tools);

    CHECK_FALSE(res.ok);
    CHECK_FALSE(res.cache_hit);
    CHECK(res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
    CHECK(res.to_status().code() == ErrorCode::invalid_argument);
    REQUIRE(res.error.offending_tools.size() == 1u);
    CHECK(res.error.offending_tools[0].name == "tool_0");
    CHECK(res.error.offending_tools[0].index == 0u);
    CHECK(res.error.message.find("tool_0") != std::string::npos);
    CHECK(res.to_status().message().find("tool_0") != std::string::npos);
    // Single-tool failure requires only the initial whole-toolset compile (no redundant recompile).
    CHECK(compiler->CallCount() == 1u);
}

TEST_CASE("tool grammar preflight: bisection finds 1 or 2 bad tools in toolset of 7") {
    const auto tools = MakeToolset(7);

    SUBCASE("1 bad tool at first, middle, and last positions") {
        for (const std::size_t bad_idx : {std::size_t{0}, std::size_t{3}, std::size_t{6}}) {
            CAPTURE(bad_idx);
            const std::string bad_name = "tool_" + std::to_string(bad_idx);
            auto compiler = std::make_shared<FakeToolGrammarCompiler>(std::unordered_set<std::string>{bad_name});
            ToolGrammarPreflight preflight(compiler);

            const auto res = preflight.Preflight("chatml", tools);
            CHECK_FALSE(res.ok);
            CHECK(res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
            REQUIRE(res.error.offending_tools.size() == 1u);
            CHECK(res.error.offending_tools[0].name == bad_name);
            CHECK(res.error.offending_tools[0].index == bad_idx);
            CHECK(res.error.message.find(bad_name) != std::string::npos);
            CHECK(compiler->CallCount() <= 7u);
        }
    }

    SUBCASE("2 bad tools in opposite halves") {
        auto compiler = std::make_shared<FakeToolGrammarCompiler>(
            std::unordered_set<std::string>{"tool_1", "tool_5"});
        ToolGrammarPreflight preflight(compiler);

        const auto res = preflight.Preflight("chatml", tools);
        CHECK_FALSE(res.ok);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
        REQUIRE(res.error.offending_tools.size() == 2u);
        CHECK(res.error.offending_tools[0].name == "tool_1");
        CHECK(res.error.offending_tools[0].index == 1u);
        CHECK(res.error.offending_tools[1].name == "tool_5");
        CHECK(res.error.offending_tools[1].index == 5u);
        CHECK(res.error.message.find("tool_1") != std::string::npos);
        CHECK(res.error.message.find("tool_5") != std::string::npos);
    }

    SUBCASE("2 bad tools in the same half") {
        auto compiler = std::make_shared<FakeToolGrammarCompiler>(
            std::unordered_set<std::string>{"tool_0", "tool_2"});
        ToolGrammarPreflight preflight(compiler);

        const auto res = preflight.Preflight("chatml", tools);
        CHECK_FALSE(res.ok);
        REQUIRE(res.error.offending_tools.size() == 2u);
        CHECK(res.error.offending_tools[0].name == "tool_0");
        CHECK(res.error.offending_tools[0].index == 0u);
        CHECK(res.error.offending_tools[1].name == "tool_2");
        CHECK(res.error.offending_tools[1].index == 2u);
    }
}

TEST_CASE("tool grammar preflight: bisection finds 1 or 2 bad tools in toolset of 64 logarithmically") {
    const auto tools = MakeToolset(64);

    SUBCASE("1 bad tool in 64") {
        auto compiler = std::make_shared<FakeToolGrammarCompiler>(
            std::unordered_set<std::string>{"tool_41"});
        ToolGrammarPreflight preflight(compiler);

        const auto res = preflight.Preflight("qwen3", tools);
        CHECK_FALSE(res.ok);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
        REQUIRE(res.error.offending_tools.size() == 1u);
        CHECK(res.error.offending_tools[0].name == "tool_41");
        CHECK(res.error.offending_tools[0].index == 41u);
        CHECK(res.error.message.find("tool_41") != std::string::npos);
        // Binary bisection over 64 tools takes 1 + 2*log2(64) = 13 compiles, never linear 64.
        CHECK(compiler->CallCount() == 13u);
    }

    SUBCASE("2 bad tools in 64 (far apart)") {
        auto compiler = std::make_shared<FakeToolGrammarCompiler>(
            std::unordered_set<std::string>{"tool_5", "tool_53"});
        ToolGrammarPreflight preflight(compiler);

        const auto res = preflight.Preflight("qwen3", tools);
        CHECK_FALSE(res.ok);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
        REQUIRE(res.error.offending_tools.size() == 2u);
        CHECK(res.error.offending_tools[0].name == "tool_5");
        CHECK(res.error.offending_tools[0].index == 5u);
        CHECK(res.error.offending_tools[1].name == "tool_53");
        CHECK(res.error.offending_tools[1].index == 53u);
        CHECK(res.error.message.find("tool_5") != std::string::npos);
        CHECK(res.error.message.find("tool_53") != std::string::npos);
        CHECK(compiler->CallCount() <= 23u);
    }

    SUBCASE("2 bad tools in 64 (adjacent across midpoint)") {
        auto compiler = std::make_shared<FakeToolGrammarCompiler>(
            std::unordered_set<std::string>{"tool_31", "tool_32"});
        ToolGrammarPreflight preflight(compiler);

        const auto res = preflight.Preflight("qwen3", tools);
        CHECK_FALSE(res.ok);
        REQUIRE(res.error.offending_tools.size() == 2u);
        CHECK(res.error.offending_tools[0].name == "tool_31");
        CHECK(res.error.offending_tools[0].index == 31u);
        CHECK(res.error.offending_tools[1].name == "tool_32");
        CHECK(res.error.offending_tools[1].index == 32u);
        CHECK(compiler->CallCount() <= 23u);
    }
}

TEST_CASE("tool grammar preflight: interaction failure reports cleanly without false blame") {
    // Fails only when >= 5 tools are compiled together; every split half of 4 succeeds.
    auto compiler = std::make_shared<FakeToolGrammarCompiler>(std::unordered_set<std::string>{}, 5u);
    ToolGrammarPreflight preflight(compiler);

    const auto tools = MakeToolset(8);
    const auto res = preflight.Preflight("chatml", tools);

    CHECK_FALSE(res.ok);
    CHECK(res.error.kind == ToolGrammarPreflightErrorKind::interaction_failure);
    CHECK(res.error.offending_tools.empty());
    CHECK(res.to_status().code() == ErrorCode::invalid_argument);
    CHECK(res.error.compiler_error.find("rule limit") != std::string::npos);
    CHECK(compiler->CallCount() == 3u);  // whole (8) + left (4) + right (4)
}

TEST_CASE("tool grammar preflight: cache hits and bounded LRU eviction") {
    auto compiler = std::make_shared<FakeToolGrammarCompiler>(
        std::unordered_set<std::string>{"broken_tool"});
    ToolGrammarPreflightOptions opts;
    opts.max_cache_entries = 2;
    ToolGrammarPreflight preflight(compiler, opts);

    const auto set_a = MakeToolset(3, "alpha_");
    const auto set_b = MakeToolset(3, "beta_");
    const auto set_c = MakeToolset(3, "gamma_");
    const std::vector<ToolDefinition> set_bad = {MakeTool("good_0"), MakeTool("broken_tool")};

    // Registration compiles once and populates the cache.
    const auto r_a1 = preflight.RegisterToolset("tmpl-1", set_a);
    CHECK(r_a1.ok);
    CHECK_FALSE(r_a1.cache_hit);
    CHECK(compiler->CallCount() == 1u);

    // Subsequent request hits cache without recompiling.
    const auto r_a2 = preflight.Preflight("tmpl-1", set_a);
    CHECK(r_a2.ok);
    CHECK(r_a2.cache_hit);
    CHECK(r_a2.cache_key == r_a1.cache_key);
    CHECK(compiler->CallCount() == 1u);

    // Different chat template id misses cache even for identical toolset.
    const auto r_a_other_tmpl = preflight.Preflight("tmpl-2", set_a);
    CHECK(r_a_other_tmpl.ok);
    CHECK_FALSE(r_a_other_tmpl.cache_hit);
    CHECK(r_a_other_tmpl.cache_key != r_a1.cache_key);
    CHECK(compiler->CallCount() == 2u);

    // Cache now holds [tmpl-2/A, tmpl-1/A] (capacity = 2). Touch tmpl-1/A to make it MRU.
    const auto r_a3 = preflight.Preflight("tmpl-1", set_a);
    CHECK(r_a3.cache_hit);
    CHECK(compiler->CallCount() == 2u);

    // Inserting B evicts LRU (tmpl-2/A), keeping tmpl-1/A.
    const auto r_b1 = preflight.Preflight("tmpl-1", set_b);
    CHECK(r_b1.ok);
    CHECK_FALSE(r_b1.cache_hit);
    CHECK(compiler->CallCount() == 3u);

    auto stats = preflight.Stats();
    CHECK(stats.entries == 2u);
    CHECK(stats.max_entries == 2u);
    CHECK(stats.evictions == 1u);

    // tmpl-1/A is still cached; tmpl-2/A was evicted and must recompile.
    CHECK(preflight.Preflight("tmpl-1", set_a).cache_hit);
    CHECK(compiler->CallCount() == 3u);
    CHECK_FALSE(preflight.Preflight("tmpl-2", set_a).cache_hit);
    CHECK(compiler->CallCount() == 4u);

    // Inserting C keeps entries bounded at 2.
    const auto r_c1 = preflight.Preflight("tmpl-1", set_c);
    CHECK(r_c1.ok);
    CHECK_FALSE(r_c1.cache_hit);
    CHECK(preflight.Stats().entries == 2u);

    // Failing toolset results are also cached so repeated bad requests don't re-bisect.
    compiler->ResetCalls();
    const auto r_bad1 = preflight.Preflight("tmpl-1", set_bad);
    CHECK_FALSE(r_bad1.ok);
    CHECK_FALSE(r_bad1.cache_hit);
    REQUIRE(r_bad1.error.offending_tools.size() == 1u);
    CHECK(r_bad1.error.offending_tools[0].name == "broken_tool");
    const std::size_t first_bad_calls = compiler->CallCount();
    CHECK(first_bad_calls > 0u);

    const auto r_bad2 = preflight.Preflight("tmpl-1", set_bad);
    CHECK_FALSE(r_bad2.ok);
    CHECK(r_bad2.cache_hit);
    REQUIRE(r_bad2.error.offending_tools.size() == 1u);
    CHECK(r_bad2.error.offending_tools[0].name == "broken_tool");
    CHECK(compiler->CallCount() == first_bad_calls);
    CHECK(preflight.Stats().entries == 2u);
}

TEST_CASE("tool grammar preflight: canonical hash is stable regardless of JSON key order") {
    const std::string schema_a = R"({
        "type": "object",
        "required": ["city", "units"],
        "additionalProperties": false,
        "properties": {
            "units": {"enum": ["c", "f"], "type": "string"},
            "city": {"type": "string", "minLength": 1}
        },
        "$defs": {
            "meta": {"properties": {"z": {"type": "integer"}, "a": {"type": "boolean"}}, "type": "object"}
        }
    })";
    const std::string schema_b = R"({
        "$defs": {
            "meta": {"type": "object", "properties": {"a": {"type": "boolean"}, "z": {"type": "integer"}}}
        },
        "properties": {
            "city": {"minLength": 1, "type": "string"},
            "units": {"type": "string", "enum": ["c", "f"]}
        },
        "additionalProperties": false,
        "required": ["city", "units"],
        "type": "object"
    })";

    const std::vector<ToolDefinition> tools_a = {MakeTool("get_weather", schema_a, true)};
    const std::vector<ToolDefinition> tools_b = {MakeTool("get_weather", schema_b, true)};

    CHECK(CanonicalizeToolsetJson(tools_a) == CanonicalizeToolsetJson(tools_b));
    const std::string key_a = ComputeToolsetPreflightKey("chatml", tools_a);
    const std::string key_b = ComputeToolsetPreflightKey("chatml", tools_b);
    CHECK(key_a.size() == 64u);
    CHECK(key_a == key_b);

    // Different template id or different schema must yield a different hash.
    CHECK(ComputeToolsetPreflightKey("llama3", tools_a) != key_a);
    const std::vector<ToolDefinition> tools_c = {
        MakeTool("get_weather", R"({"type":"object","properties":{"city":{"type":"integer"}}})", true)};
    CHECK(ComputeToolsetPreflightKey("chatml", tools_c) != key_a);

    // Cache hit across reordered JSON keys.
    auto compiler = std::make_shared<FakeToolGrammarCompiler>();
    ToolGrammarPreflight preflight(compiler);
    const auto first = preflight.Preflight("chatml", tools_a);
    const auto second = preflight.Preflight("chatml", tools_b);
    CHECK(first.ok);
    CHECK_FALSE(first.cache_hit);
    CHECK(second.ok);
    CHECK(second.cache_hit);
    CHECK(second.cache_key == first.cache_key);
    CHECK(compiler->CallCount() == 1u);
}

TEST_CASE("tool grammar preflight: empty toolset succeeds and disabled cache never stores entries") {
    auto compiler = std::make_shared<FakeToolGrammarCompiler>();
    ToolGrammarPreflightOptions opts;
    opts.max_cache_entries = 0;
    ToolGrammarPreflight preflight(compiler, opts);

    const std::vector<ToolDefinition> empty_tools;
    const auto empty_res = preflight.Preflight("chatml", empty_tools);
    CHECK(empty_res.ok);
    CHECK(compiler->CallCount() == 0u);

    const auto one_tool = MakeToolset(1);
    CHECK_FALSE(preflight.Preflight("chatml", one_tool).cache_hit);
    CHECK_FALSE(preflight.Preflight("chatml", one_tool).cache_hit);
    CHECK(compiler->CallCount() == 2u);
    CHECK(preflight.Stats().entries == 0u);
}

TEST_CASE("tool grammar preflight: CanonicalizeJsonString, offender cap, and grammar non-interference") {
    // CanonicalizeJsonString normalizes nested object keys and rejects malformed JSON.
    const auto c1 = CanonicalizeJsonString(R"({"b":1,"a":{"d":true,"c":[2,1]}})");
    const auto c2 = CanonicalizeJsonString(R"({"a":{"c":[2,1],"d":true},"b":1})");
    REQUIRE(c1.ok());
    REQUIRE(c2.ok());
    CHECK(c1.value() == c2.value());
    CHECK(c1.value() == R"({"a":{"c":[2,1],"d":true},"b":1})");
    CHECK_FALSE(CanonicalizeJsonString("{not valid json").ok());

    // max_offending_tools bounds bisection output on pathological toolsets and only appends
    // ", ..." when additional offenders exist beyond the cap.
    std::unordered_set<std::string> all_bad;
    for (std::size_t i = 0; i < 8; ++i) all_bad.insert("tool_" + std::to_string(i));
    auto bad_compiler = std::make_shared<FakeToolGrammarCompiler>(all_bad);
    ToolGrammarPreflightOptions cap_opts;
    cap_opts.max_offending_tools = 3;
    ToolGrammarPreflight capped(bad_compiler, cap_opts);
    const auto cap_res = capped.Preflight("chatml", MakeToolset(8));
    CHECK_FALSE(cap_res.ok);
    CHECK(cap_res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
    CHECK(cap_res.error.offending_tools.size() == 3u);
    CHECK(cap_res.error.message.find(", ...") != std::string::npos);

    // Exactly max_offending_tools bad tools must NOT append ", ...".
    auto exact_cap_compiler = std::make_shared<FakeToolGrammarCompiler>(
        std::unordered_set<std::string>{"tool_0", "tool_3", "tool_7"});
    ToolGrammarPreflight exact_capped(exact_cap_compiler, cap_opts);
    const auto exact_res = exact_capped.Preflight("chatml", MakeToolset(8));
    CHECK_FALSE(exact_res.ok);
    CHECK(exact_res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
    CHECK(exact_res.error.offending_tools.size() == 3u);
    CHECK(exact_res.error.message.find(", ...") == std::string::npos);

    // Malformed parameters JSON cannot collide with a valid schema object containing "$invalid_json".
    const std::vector<ToolDefinition> malformed = {MakeTool("t", "{bad", true)};
    const std::vector<ToolDefinition> valid_literal = {MakeTool("t", R"({"$invalid_json":"{bad"})", true)};
    CHECK(CanonicalizeToolsetJson(malformed) != CanonicalizeToolsetJson(valid_literal));
    CHECK(ComputeToolsetPreflightKey("chatml", malformed) != ComputeToolsetPreflightKey("chatml", valid_literal));

    // Preflight never mutates a request's active generation grammar.
    ChatRequest req;
    req.grammar = "root ::= \"preserved\"";
    req.tools = MakeToolset(2);
    auto ok_compiler = std::make_shared<FakeToolGrammarCompiler>();
    ToolGrammarPreflight preflight(ok_compiler);
    const auto res = preflight.Preflight("chatml", req.tools);
    CHECK(res.ok);
    CHECK(req.grammar == "root ::= \"preserved\"");
}

TEST_CASE("tool grammar preflight: non-schema compiler errors are not bisected, blamed, or cached") {
    class FixedStatusCompiler final : public ToolGrammarCompiler {
    public:
        explicit FixedStatusCompiler(Status st) : st_(std::move(st)) {}
        Status Compile(std::string_view, std::span<const ToolDefinition>) const override {
            calls.fetch_add(1, std::memory_order_relaxed);
            return st_;
        }
        mutable std::atomic<std::size_t> calls{0};
    private:
        Status st_;
    };

    const auto tools = MakeToolset(7);

    SUBCASE("unsupported error") {
        auto comp = std::make_shared<FixedStatusCompiler>(
            Status(ErrorCode::unsupported, "tool-call GBNF builder unavailable"));
        ToolGrammarPreflight preflight(comp);

        const auto res = preflight.Preflight("chatml", tools);
        CHECK_FALSE(res.ok);
        CHECK_FALSE(res.cache_hit);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::unsupported);
        CHECK(res.error.offending_tools.empty());
        CHECK(res.to_status().code() == ErrorCode::unsupported);
        CHECK(comp->calls.load() == 1u);
        CHECK(preflight.Stats().entries == 0u);
    }

    SUBCASE("backend_error") {
        auto comp = std::make_shared<FixedStatusCompiler>(
            Status(ErrorCode::backend_error, "grammar sampler failed to apply"));
        ToolGrammarPreflight preflight(comp);

        const auto res = preflight.Preflight("chatml", tools);
        CHECK_FALSE(res.ok);
        CHECK_FALSE(res.cache_hit);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::backend_error);
        CHECK(res.error.offending_tools.empty());
        CHECK(res.to_status().code() == ErrorCode::backend_error);
        CHECK(comp->calls.load() == 1u);
        CHECK(preflight.Stats().entries == 0u);
    }

    SUBCASE("internal error") {
        auto comp = std::make_shared<FixedStatusCompiler>(
            Status(ErrorCode::internal, "unexpected GBNF root layout"));
        ToolGrammarPreflight preflight(comp);

        const auto res = preflight.Preflight("chatml", tools);
        CHECK_FALSE(res.ok);
        CHECK(res.error.kind == ToolGrammarPreflightErrorKind::backend_error);
        CHECK(res.error.offending_tools.empty());
        CHECK(res.to_status().code() == ErrorCode::internal);
        CHECK(comp->calls.load() == 1u);
        CHECK(preflight.Stats().entries == 0u);
    }
}

#if defined(SONDER_LLAMACPP_VOCAB_DIR) && defined(SONDER_HAS_SERVER)
TEST_CASE("tool grammar preflight: native llama.cpp compiler validates real toolsets (vocab-only)") {
    namespace lc = sonder::backends::llamacpp;
    lc::LlamaCppBackend be;
    lc::LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-bpe.gguf";
    opts.vocab_only = true;
    REQUIRE(be.Load(opts).ok());

    auto compiler = MakeLlamaCppToolGrammarCompiler(be);
    REQUIRE(compiler != nullptr);
    ToolGrammarPreflight preflight(compiler);

    const std::vector<ToolDefinition> valid_tools = {
        MakeTool("get_weather",
                 R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})",
                 true),
        MakeTool("get_time",
                 R"({"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"],"additionalProperties":false})",
                 true),
    };
    const auto ok_res = preflight.Preflight("chatml", valid_tools);
    CHECK_MESSAGE(ok_res.ok, ok_res.error.message);
    CHECK_FALSE(ok_res.cache_hit);

    // Same valid tools with reordered JSON keys hit the cache.
    const std::vector<ToolDefinition> reordered_tools = {
        MakeTool("get_weather",
                 R"({"additionalProperties":false,"required":["city"],"properties":{"city":{"type":"string"}},"type":"object"})",
                 true),
        MakeTool("get_time",
                 R"({"additionalProperties":false,"required":["tz"],"properties":{"tz":{"type":"string"}},"type":"object"})",
                 true),
    };
    const auto hit_res = preflight.Preflight("chatml", reordered_tools);
    CHECK(hit_res.ok);
    CHECK(hit_res.cache_hit);

    // Strict tool with unsupported JSON Schema keyword (`allOf`) fails and is bisected by name.
    const std::vector<ToolDefinition> mixed_tools = {
        valid_tools[0],
        MakeTool("bad_allof_tool", R"({"type":"object","allOf":[{"type":"object"}]})", true),
        valid_tools[1],
    };
    const auto fail_res = preflight.Preflight("chatml", mixed_tools);
    CHECK_FALSE(fail_res.ok);
    CHECK(fail_res.error.kind == ToolGrammarPreflightErrorKind::compile_failed);
    REQUIRE(fail_res.error.offending_tools.size() == 1u);
    CHECK(fail_res.error.offending_tools[0].name == "bad_allof_tool");
    CHECK(fail_res.error.offending_tools[0].index == 1u);
    CHECK(fail_res.error.message.find("bad_allof_tool") != std::string::npos);

    be.Unload();
}
#endif
