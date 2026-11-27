// Unit tests for the per-route speculative-decoding controller
// (src/backends/llamacpp/spec_controller.hpp). Pure policy: no model weights,
// no llama.cpp calls, time comes from an injected fake clock.
#include <doctest/doctest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "sonder/inference/backends/llamacpp.hpp"
#include "spec_controller.hpp"

using namespace sonder::backends::llamacpp;
namespace si = sonder::inference;

namespace {

struct FakeClock {
    std::int64_t now_ms = 1'000'000;
    SpecClock fn() {
        return [this]() { return now_ms; };
    }
};

SpecControllerConfig BaseConfig(SpecStrategy mode = SpecStrategy::kSuffix, std::int32_t n_max = 8) {
    SpecControllerConfig cfg;
    cfg.base.mode = mode;
    cfg.base.n_max = n_max;
    cfg.acceptance.window_steps = 8;
    cfg.acceptance.min_drafted_tokens = 16;
    cfg.acceptance.disable_below = 0.25;
    cfg.acceptance.reenable_at_least = 0.40;
    cfg.acceptance.probe_drafted_tokens = 16;
    cfg.acceptance.reprobe_initial_ms = 1000;
    cfg.acceptance.reprobe_max_ms = 8000;
    cfg.acceptance.backoff_factor = 2.0;
    return cfg;
}

SpecRequestContext Ctx(const char* route, SpecTaskClass task) {
    SpecRequestContext c;
    c.route = route;
    c.task = task;
    return c;
}

SpecPolicyOverlay Parse(const char* text) {
    SpecPolicyOverlay o;
    std::string err;
    INFO(text << " -> " << err);
    REQUIRE(ParseSpecPolicyOverlay(text, o, err));
    return o;
}

}  // namespace

TEST_CASE("spec controller: names, parsing and task classification") {
    CHECK(ToString(SpecTaskClass::kChat) == "chat");
    CHECK(ToString(SpecTaskClass::kCode) == "code");
    CHECK(ToString(SpecTaskClass::kToolCall) == "tool-call");
    CHECK(ToString(SpecTaskClass::kReasoning) == "reasoning");

    SpecTaskClass c = SpecTaskClass::kChat;
    CHECK(ParseSpecTaskClass("CODE", c));
    CHECK(c == SpecTaskClass::kCode);
    CHECK(ParseSpecTaskClass("tool_call", c));
    CHECK(c == SpecTaskClass::kToolCall);
    CHECK(ParseSpecTaskClass("tool_json", c));  // llamaserver's adaptive-speculation name
    CHECK(c == SpecTaskClass::kToolCall);
    CHECK(ParseSpecTaskClass("code_edit", c));
    CHECK(c == SpecTaskClass::kCode);
    CHECK(ParseSpecTaskClass("prose", c));
    CHECK(c == SpecTaskClass::kChat);
    CHECK(ParseSpecTaskClass("general", c));
    CHECK(c == SpecTaskClass::kChat);
    CHECK(ParseSpecTaskClass("Reasoning", c));
    CHECK(c == SpecTaskClass::kReasoning);
    CHECK_FALSE(ParseSpecTaskClass("", c));
    CHECK_FALSE(ParseSpecTaskClass("poetry", c));

    SpecStrategy m = SpecStrategy::kNone;
    CHECK(ParseSpecStrategy("suffix", m));
    CHECK(m == SpecStrategy::kSuffix);
    CHECK(ParseSpecStrategy("NGRAM", m));
    CHECK(m == SpecStrategy::kNgram);
    CHECK(ParseSpecStrategy("ngram-mod", m));
    CHECK(m == SpecStrategy::kNgramMod);
    CHECK(ParseSpecStrategy("draft-model", m));
    CHECK(m == SpecStrategy::kDraftModel);
    CHECK(ParseSpecStrategy("draft_simple", m));
    CHECK(m == SpecStrategy::kDraftModel);
    CHECK(ParseSpecStrategy("draft-mtp", m));
    CHECK(m == SpecStrategy::kDraftMtp);
    CHECK(ParseSpecStrategy("mtp", m));
    CHECK(m == SpecStrategy::kDraftMtp);
    CHECK(ParseSpecStrategy("none", m));
    CHECK(m == SpecStrategy::kNone);
    CHECK_FALSE(ParseSpecStrategy("auto", m));  // "auto" depends on the model; rules need a concrete type
    CHECK_FALSE(ParseSpecStrategy("medusa", m));

    CHECK(ToSlotSpecMode(SpecStrategy::kNgram) == SlotSpecMode::kNgramShort);
    CHECK(ToSlotSpecMode(SpecStrategy::kNgramMod) == SlotSpecMode::kNgramMod);
    CHECK(ToSlotSpecMode(SpecStrategy::kSuffix) == SlotSpecMode::kSuffix);
    CHECK(ToSlotSpecMode(SpecStrategy::kDraftMtp) == SlotSpecMode::kDraftMtp);
    CHECK(ToSlotSpecMode(SpecStrategy::kDraftModel) == SlotSpecMode::kNone);
    CHECK(ToSpeculativeType(SpecStrategy::kDraftModel) == std::nullopt);
    CHECK(ToSpeculativeType(SpecStrategy::kDraftMtp) == SpeculativeType::kDraftMtp);

    SpecTaskFeatures f;
    CHECK(ClassifySpecTask(f) == SpecTaskClass::kChat);
    f.has_code_markers = true;
    CHECK(ClassifySpecTask(f) == SpecTaskClass::kCode);
    f.thinking_enabled = true;
    CHECK(ClassifySpecTask(f) == SpecTaskClass::kReasoning);  // reasoning beats code
    f.tools_requested = true;
    CHECK(ClassifySpecTask(f) == SpecTaskClass::kToolCall);   // tool call beats everything
    SpecTaskFeatures g;
    g.has_grammar = true;
    CHECK(ClassifySpecTask(g) == SpecTaskClass::kToolCall);
    SpecTaskFeatures h;
    h.structured_output = true;
    CHECK(ClassifySpecTask(h) == SpecTaskClass::kToolCall);
}

TEST_CASE("spec controller: classifies SI GenerateRequest and ChatRequest directly") {
    si::GenerateRequest gr;
    gr.prompt = "Tell me about birds.";
    CHECK(ClassifyGenerateRequest(gr) == SpecTaskClass::kChat);
    gr.prompt = "Fix this:\n```cpp\nint x = 0;\n```";
    CHECK(ClassifyGenerateRequest(gr) == SpecTaskClass::kCode);
    gr.grammar = "root ::= \"ok\"";
    CHECK(ClassifyGenerateRequest(gr) == SpecTaskClass::kToolCall);

    si::ChatRequest cr;
    cr.messages.push_back({"user", "Hello there."});
    CHECK(ClassifyChatRequest(cr) == SpecTaskClass::kChat);
    cr.messages.push_back({"user", "Apply:\ndiff --git a/x b/x\n@@ -1 +1 @@"});
    CHECK(ClassifyChatRequest(cr) == SpecTaskClass::kCode);
    cr.thinking.enable_thinking = true;
    CHECK(ClassifyChatRequest(cr) == SpecTaskClass::kReasoning);
    cr.thinking.enable_thinking = false;
    cr.separate_reasoning = true;
    CHECK(ClassifyChatRequest(cr) == SpecTaskClass::kReasoning);
    cr.response_format_type = "json_schema";
    CHECK(ClassifyChatRequest(cr) == SpecTaskClass::kToolCall);

    si::ChatRequest tool_req;
    tool_req.messages.push_back({"user", "What is the weather?"});
    tool_req.tools.push_back({"get_weather", "weather", "{}"});
    CHECK(ClassifyChatRequest(tool_req) == SpecTaskClass::kToolCall);
    tool_req.tool_choice = si::ToolChoice{si::ToolChoiceMode::none, {}};
    CHECK(ClassifyChatRequest(tool_req) == SpecTaskClass::kChat);
}

TEST_CASE("spec controller: policy overlay parsing") {
    const SpecPolicyOverlay a = Parse("suffix,n_max=16,p_min=0.5");
    CHECK(a.mode == SpecStrategy::kSuffix);
    CHECK(a.n_max == 16);
    REQUIRE(a.p_min.has_value());
    CHECK(*a.p_min == doctest::Approx(0.5));

    const SpecPolicyOverlay b = Parse("n_max=4");
    CHECK_FALSE(b.mode.has_value());
    CHECK(b.n_max == 4);
    CHECK_FALSE(b.p_min.has_value());

    const SpecPolicyOverlay c = Parse(" mode = draft-model , n_max = 5 , p_min = 0.75 ");
    CHECK(c.mode == SpecStrategy::kDraftModel);
    CHECK(c.n_max == 5);
    REQUIRE(c.p_min.has_value());
    CHECK(*c.p_min == doctest::Approx(0.75));

    SpecPolicyOverlay o;
    std::string err;
    CHECK_FALSE(ParseSpecPolicyOverlay("", o, err));
    CHECK_FALSE(err.empty());
    for (const char* bad : {"warp", "mode=auto", "n_max=-1", "n_max=129", "n_max=abc", "n_max=", "p_min=1.5",
                            "p_min=-0.1", "p_min=nan", "p_min=x", "colour=red", "suffix,suffix", "n_max=1,n_max=2"}) {
        err.clear();
        INFO(bad);
        CHECK_FALSE(ParseSpecPolicyOverlay(bad, o, err));
        CHECK_FALSE(err.empty());
    }
}

TEST_CASE("spec controller: rule parsing and LlamaCppBackendOptions integration") {
    SpecControllerConfig cfg;
    std::string err;
    REQUIRE(ParseSpecRule("*=none", cfg, err));
    REQUIRE(ParseSpecRule("code=suffix,n_max=16", cfg, err));
    REQUIRE(ParseSpecRule("@org/big-model=ngram-mod", cfg, err));
    REQUIRE(ParseSpecRule("tool-call@org/big-model=draft-model,n_max=6,p_min=0.7", cfg, err));
    CHECK(cfg.base.mode == SpecStrategy::kNone);
    REQUIRE(cfg.class_rules[static_cast<std::size_t>(SpecTaskClass::kCode)].has_value());
    CHECK(cfg.class_rules[static_cast<std::size_t>(SpecTaskClass::kCode)]->n_max == 16);
    REQUIRE(cfg.routes.count("org/big-model") == 1);
    CHECK(cfg.routes["org/big-model"].any->mode == SpecStrategy::kNgramMod);
    CHECK(cfg.routes["org/big-model"].by_class[static_cast<std::size_t>(SpecTaskClass::kToolCall)]->mode ==
          SpecStrategy::kDraftModel);
    CHECK(cfg.routes["org/big-model"].by_class[static_cast<std::size_t>(SpecTaskClass::kToolCall)]->n_max == 6);

    for (const char* bad : {"", "code", "code=", "poetry=suffix", "code@=suffix", "code=warp", "=suffix",
                            "@a@b=suffix"}) {
        SpecControllerConfig scratch;
        err.clear();
        INFO(bad);
        CHECK_FALSE(ParseSpecRule(bad, scratch, err));
        CHECK_FALSE(err.empty());
    }

    // Backend options validation hook (validate_llamacpp_options).
    si::LlamaCppBackendOptions bo;
    bo.spec_type = "auto";
    bo.spec_p_min = 0.25;
    bo.spec_route_rules = {"code=suffix,n_max=16", "reasoning@model.gguf=draft-mtp,n_max=3,p_min=0.6"};
    bo.deterministic = true;
    CHECK(si::validate_llamacpp_options(bo).ok());

    SpecControllerConfig built;
    REQUIRE(BuildSpecControllerConfig(bo, built, err));
    CHECK(built.auto_base);
    CHECK(built.deterministic);
    CHECK(built.base.p_min == doctest::Approx(0.25));

    si::LlamaCppBackendOptions bad_pmin = bo;
    bad_pmin.spec_p_min = 1.5;
    CHECK_FALSE(si::validate_llamacpp_options(bad_pmin).ok());

    si::LlamaCppBackendOptions bad_rule = bo;
    bad_rule.spec_route_rules = {"bogus-class=suffix"};
    CHECK_FALSE(si::validate_llamacpp_options(bad_rule).ok());
}

TEST_CASE("spec controller: per-route config resolution, most specific field wins") {
    SpecControllerConfig cfg;
    cfg.base = {SpecStrategy::kNone, 0, 0.0};
    std::string err;
    REQUIRE(ParseSpecRule("code=suffix,n_max=16", cfg, err));
    REQUIRE(ParseSpecRule("chat=ngram-mod,n_max=4", cfg, err));
    REQUIRE(ParseSpecRule("@r1=ngram", cfg, err));  // route-wide: mode only
    REQUIRE(ParseSpecRule("code@r1=draft-model,n_max=5,p_min=0.8", cfg, err));
    REQUIRE(ParseSpecRule("reasoning@r2=none", cfg, err));
    REQUIRE(cfg.Validate().empty());

    // Unknown route: class defaults.
    SpecPolicy p = cfg.Resolve("other", SpecTaskClass::kCode);
    CHECK(p.mode == SpecStrategy::kSuffix);
    CHECK(p.n_max == 16);
    p = cfg.Resolve("other", SpecTaskClass::kChat);
    CHECK(p.mode == SpecStrategy::kNgramMod);
    CHECK(p.n_max == 4);
    // Class with no rule falls back to the base policy (none).
    p = cfg.Resolve("other", SpecTaskClass::kToolCall);
    CHECK(p.mode == SpecStrategy::kNone);

    // Route-wide overlay replaces only the mode; the class n_max survives.
    p = cfg.Resolve("r1", SpecTaskClass::kChat);
    CHECK(p.mode == SpecStrategy::kNgram);
    CHECK(p.n_max == 4);
    // Route+class beats route-wide and class defaults, field by field.
    p = cfg.Resolve("r1", SpecTaskClass::kCode);
    CHECK(p.mode == SpecStrategy::kDraftModel);
    CHECK(p.n_max == 5);
    CHECK(p.p_min == doctest::Approx(0.8));
    // Another route is untouched by r1's rules.
    p = cfg.Resolve("r2", SpecTaskClass::kCode);
    CHECK(p.mode == SpecStrategy::kSuffix);
    CHECK(p.n_max == 16);
    p = cfg.Resolve("r2", SpecTaskClass::kReasoning);
    CHECK(p.mode == SpecStrategy::kNone);
}

TEST_CASE("spec controller: auto base and n_max 0 defaults") {
    CHECK(DefaultSpecDraftMax(SpecStrategy::kNone) == 0);
    CHECK(DefaultSpecDraftMax(SpecStrategy::kDraftMtp) == kDefaultMtpDraftMax);
    CHECK(DefaultSpecDraftMax(SpecStrategy::kDraftModel) == kDefaultDraftModelDraftMax);
    CHECK(DefaultSpecDraftMax(SpecStrategy::kSuffix) == kDefaultSuffixDraftMax);
    CHECK(DefaultSpecDraftMax(SpecStrategy::kNgram) == kDefaultNgramDraftMax);
    CHECK(DefaultSpecDraftMax(SpecStrategy::kNgramMod) == kDefaultNgramModDraftMax);

    FakeClock clk;
    SpecController ctl(BaseConfig(SpecStrategy::kSuffix, 0), clk.fn());
    const SpecDecision d = ctl.Decide(Ctx("r", SpecTaskClass::kChat));
    CHECK(d.mode == SpecStrategy::kSuffix);
    CHECK(d.n_max == kDefaultSuffixDraftMax);
    CHECK(d.reason == SpecDecisionReason::kConfigured);
    CHECK_FALSE(d.is_probe);

    // auto_base selects MTP when available, else ngram-mod.
    SpecControllerConfig auto_cfg = BaseConfig(SpecStrategy::kNone, 0);
    auto_cfg.auto_base = true;
    SpecController auto_ctl(auto_cfg, clk.fn());
    SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
    c.caps.mtp_available = false;
    CHECK(auto_ctl.Decide(c).mode == SpecStrategy::kNgramMod);
    c.caps.mtp_available = true;
    CHECK(auto_ctl.Decide(c).mode == SpecStrategy::kDraftMtp);
    // In determinism mode, auto_base with MTP turns off model-based drafting.
    c.deterministic = true;
    CHECK(auto_ctl.Decide(c).reason == SpecDecisionReason::kDeterminismModelDraftOff);
}

TEST_CASE("spec controller: config validation") {
    CHECK(BaseConfig().Validate().empty());
    {
        auto c = BaseConfig();
        c.acceptance.window_steps = 0;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.window_steps = 100000;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.disable_below = 0.5;
        c.acceptance.reenable_at_least = 0.4;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.disable_below = 1.5;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.backoff_factor = 0.5;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.reprobe_initial_ms = 0;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.acceptance.reprobe_max_ms = c.acceptance.reprobe_initial_ms - 1;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.base.n_max = 129;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.base.p_min = 2.0;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.max_tracked_entries = 0;
        CHECK_FALSE(c.Validate().empty());
    }
    {
        auto c = BaseConfig();
        c.max_spec_slots = 17;
        CHECK_FALSE(c.Validate().empty());
    }
}

TEST_CASE("spec controller: rolling acceptance window maths") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());  // window of 8 steps
    const auto kRoute = "r";
    const auto kTask = SpecTaskClass::kChat;
    const auto kMode = SpecStrategy::kSuffix;

    CHECK(ctl.Stats(kRoute, kTask, kMode).window_steps == 0);
    CHECK(ctl.Stats(kRoute, kTask, kMode).rolling_acceptance == doctest::Approx(0.0));

    ctl.Observe(kRoute, kTask, kMode, 8, 6);
    ctl.Observe(kRoute, kTask, kMode, 8, 2);
    SpecRouteStats s = ctl.Stats(kRoute, kTask, kMode);
    CHECK(s.window_steps == 2);
    CHECK(s.window_drafted == 16);
    CHECK(s.window_accepted == 8);
    CHECK(s.rolling_acceptance == doctest::Approx(0.5));

    // accepted > drafted is clamped, drafted == 0 carries no information.
    ctl.Observe(kRoute, kTask, kMode, 4, 99);
    ctl.Observe(kRoute, kTask, kMode, 0, 0);
    s = ctl.Stats(kRoute, kTask, kMode);
    CHECK(s.window_steps == 3);
    CHECK(s.window_drafted == 20);
    CHECK(s.window_accepted == 12);
    CHECK(s.total_drafted == 20);
    CHECK(s.total_accepted == 12);

    // Fill past the 8-step window: the oldest steps (6/8, 2/8, 4/4) fall out.
    for (int i = 0; i < 8; ++i) ctl.Observe(kRoute, kTask, kMode, 10, 9);
    s = ctl.Stats(kRoute, kTask, kMode);
    CHECK(s.window_steps == 8);
    CHECK(s.window_drafted == 80);
    CHECK(s.window_accepted == 72);
    CHECK(s.rolling_acceptance == doctest::Approx(0.9));
    CHECK(s.total_drafted == 100);  // totals are never windowed
    CHECK(s.total_accepted == 84);
}

TEST_CASE("spec controller: no judgement before enough drafted tokens, healthy routes stay on") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());  // min_drafted_tokens = 16, disable_below = 0.25
    const auto ctx = Ctx("r", SpecTaskClass::kCode);
    CHECK(ctl.Decide(ctx).mode == SpecStrategy::kSuffix);
    ctl.Observe("r", SpecTaskClass::kCode, SpecStrategy::kSuffix, 8, 0);  // 0% but only 8 tokens of evidence
    CHECK(ctl.Decide(ctx).mode == SpecStrategy::kSuffix);
    CHECK(ctl.Stats("r", SpecTaskClass::kCode, SpecStrategy::kSuffix).state == SpecAcceptanceState::kEnabled);
    for (int i = 0; i < 20; ++i) ctl.Observe("r", SpecTaskClass::kCode, SpecStrategy::kSuffix, 8, 6);
    CHECK(ctl.Decide(ctx).mode == SpecStrategy::kSuffix);
    CHECK(ctl.Stats("r", SpecTaskClass::kCode, SpecStrategy::kSuffix).disable_count == 0);
    // Exactly at the threshold is not "below" it.
    SpecController edge(BaseConfig(), clk.fn());
    edge.Observe("e", SpecTaskClass::kChat, SpecStrategy::kSuffix, 16, 4);  // 0.25
    CHECK(edge.Decide(Ctx("e", SpecTaskClass::kChat)).mode == SpecStrategy::kSuffix);
}

TEST_CASE("spec controller: low acceptance auto-disables only that route and class") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());
    ctl.Observe("a", SpecTaskClass::kChat, SpecStrategy::kSuffix, 16, 1);  // 6%
    const SpecDecision off = ctl.Decide(Ctx("a", SpecTaskClass::kChat));
    CHECK(off.mode == SpecStrategy::kNone);
    CHECK(off.n_max == 0);
    CHECK(off.reason == SpecDecisionReason::kAcceptanceDisabled);
    CHECK(off.configured_mode == SpecStrategy::kSuffix);
    CHECK_FALSE(off.is_probe);

    // Same route, other task class and other route are unaffected.
    CHECK(ctl.Decide(Ctx("a", SpecTaskClass::kCode)).mode == SpecStrategy::kSuffix);
    CHECK(ctl.Decide(Ctx("b", SpecTaskClass::kChat)).mode == SpecStrategy::kSuffix);

    const SpecRouteStats s = ctl.Stats("a", SpecTaskClass::kChat, SpecStrategy::kSuffix);
    CHECK(s.state == SpecAcceptanceState::kDisabled);
    CHECK(s.disable_count == 1);
    CHECK(s.window_steps == 0);  // evidence from before the disable is dropped
    CHECK(s.retry_in_ms == 1000);

    // Observations from requests that were already in flight are ignored while disabled.
    ctl.Observe("a", SpecTaskClass::kChat, SpecStrategy::kSuffix, 16, 16);
    const SpecRouteStats s2 = ctl.Stats("a", SpecTaskClass::kChat, SpecStrategy::kSuffix);
    CHECK(s2.state == SpecAcceptanceState::kDisabled);
    CHECK(s2.ignored_observations == 1);
    CHECK(s2.total_drafted == 16);
}

TEST_CASE("spec controller: re-probe timing with exponential backoff, cancel and cap") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());  // initial 1000 ms, x2, max 8000 ms
    const auto ctx = Ctx("r", SpecTaskClass::kChat);
    const auto M = SpecStrategy::kSuffix;
    const auto T = SpecTaskClass::kChat;
    const std::int64_t t0 = clk.now_ms;

    ctl.Observe("r", T, M, 16, 0);  // disable #1
    clk.now_ms = t0 + 999;
    CHECK(ctl.Decide(ctx).mode == SpecStrategy::kNone);
    CHECK(ctl.Stats("r", T, M).retry_in_ms == 1);
    clk.now_ms = t0 + 1000;
    SpecDecision d = ctl.Decide(ctx);  // first probe
    CHECK(d.mode == SpecStrategy::kSuffix);
    CHECK(d.is_probe);
    CHECK(d.reason == SpecDecisionReason::kProbe);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kProbing);
    CHECK(ctl.Stats("r", T, M).probe_count == 1);

    // Cancelling an in-flight probe with a mismatched probe_id is ignored; matching probe_id returns to disabled.
    const std::uint64_t pid1 = d.probe_id;
    CHECK(pid1 > 0);
    ctl.CancelProbe("r", T, M, pid1 + 99);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kProbing);
    ctl.CancelProbe("r", T, M, pid1);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kDisabled);
    CHECK(ctl.Stats("r", T, M).failed_probes == 0);
    CHECK(ctl.Stats("r", T, M).backoff_ms == 1000);
    clk.now_ms += 1000;
    const SpecDecision d2 = ctl.Decide(ctx);
    CHECK(d2.is_probe);
    CHECK(d2.probe_id > pid1);
    CHECK(ctl.Stats("r", T, M).probe_count == 2);

    // Single-probe ownership: while one probe is in flight, a concurrent request gets disabled.
    const SpecDecision concurrent = ctl.Decide(ctx);
    CHECK(concurrent.mode == SpecStrategy::kNone);
    CHECK_FALSE(concurrent.is_probe);
    CHECK(concurrent.reason == SpecDecisionReason::kAcceptanceDisabled);
    CHECK(ctl.Stats("r", T, M).probe_count == 2);

    // Partial probe evidence (8 < 16 drafted) releases probe ownership so the next request continues it.
    ctl.Observe("r", T, M, 8, 0);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kProbing);
    const SpecDecision cont = ctl.Decide(ctx);
    CHECK(cont.is_probe);
    CHECK(cont.probe_id > d2.probe_id);
    CHECK(ctl.Stats("r", T, M).probe_count == 2);  // continuation of the same probe, not a new one
    // The probe fails once it has 16 drafted tokens at 0%: backoff doubles to 2000 ms.
    const std::int64_t t1 = clk.now_ms;
    ctl.Observe("r", T, M, 8, 0);
    SpecRouteStats s = ctl.Stats("r", T, M);
    CHECK(s.state == SpecAcceptanceState::kDisabled);
    CHECK(s.disable_count == 1);  // a failed probe is not a new disable
    CHECK(s.failed_probes == 1);
    CHECK(s.backoff_ms == 2000);
    CHECK(s.retry_in_ms == 2000);
    clk.now_ms = t1 + 1999;
    CHECK(ctl.Decide(ctx).mode == SpecStrategy::kNone);
    clk.now_ms = t1 + 2000;
    CHECK(ctl.Decide(ctx).is_probe);
    // Fail again: 4000, then 8000, then capped at 8000.
    std::int64_t expected[] = {4000, 8000, 8000, 8000};
    for (std::int64_t want : expected) {
        ctl.Observe("r", T, M, 16, 0);
        s = ctl.Stats("r", T, M);
        CHECK(s.state == SpecAcceptanceState::kDisabled);
        CHECK(s.backoff_ms == want);
        clk.now_ms += want;
        CHECK(ctl.Decide(ctx).is_probe);
    }

    // A probe at or above reenable_at_least (0.40) turns speculation back on and resets the backoff.
    ctl.Observe("r", T, M, 16, 7);  // 0.4375
    s = ctl.Stats("r", T, M);
    CHECK(s.state == SpecAcceptanceState::kEnabled);
    CHECK(s.backoff_ms == 0);
    CHECK(s.window_steps == 0);
    d = ctl.Decide(ctx);
    CHECK(d.mode == SpecStrategy::kSuffix);
    CHECK_FALSE(d.is_probe);

    // The next disable starts from the initial backoff again.
    ctl.Observe("r", T, M, 16, 0);
    CHECK(ctl.Stats("r", T, M).backoff_ms == 1000);
    CHECK(ctl.Stats("r", T, M).disable_count == 2);
}

TEST_CASE("spec controller: a probe between disable and re-enable thresholds fails (hysteresis)") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());  // disable < 0.25, re-enable >= 0.40
    const auto ctx = Ctx("r", SpecTaskClass::kChat);
    const auto M = SpecStrategy::kSuffix;
    const auto T = SpecTaskClass::kChat;
    ctl.Observe("r", T, M, 16, 0);
    clk.now_ms += 1000;
    REQUIRE(ctl.Decide(ctx).is_probe);
    ctl.Observe("r", T, M, 16, 5);  // 0.3125: above disable line, below re-enable line
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kDisabled);
    CHECK(ctl.Stats("r", T, M).backoff_ms == 2000);
}

TEST_CASE("spec controller: determinism mode turns all model-based drafting off") {
    FakeClock clk;
    // 1. MTP (per-request determinism).
    {
        SpecController ctl(BaseConfig(SpecStrategy::kDraftMtp, 3), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
        c.caps.mtp_available = true;
        CHECK(ctl.Decide(c).mode == SpecStrategy::kDraftMtp);
        c.deterministic = true;
        const SpecDecision d = ctl.Decide(c);
        CHECK(d.mode == SpecStrategy::kNone);
        CHECK(d.n_max == 0);
        CHECK(d.reason == SpecDecisionReason::kDeterminismModelDraftOff);
        CHECK(d.configured_mode == SpecStrategy::kDraftMtp);
        CHECK(ctl.TrackedEntries() == 0);
    }
    // 2. Standalone draft-model (process-wide and per-request determinism).
    {
        auto cfg = BaseConfig(SpecStrategy::kDraftModel, 5);
        cfg.base.p_min = 0.65;
        SpecController ctl(cfg, clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kCode);
        c.caps.draft_model_available = true;
        const SpecDecision on = ctl.Decide(c);
        CHECK(on.mode == SpecStrategy::kDraftModel);
        CHECK(on.n_max == 5);
        CHECK(on.p_min == doctest::Approx(0.65));
        c.deterministic = true;
        const SpecDecision off = ctl.Decide(c);
        CHECK(off.mode == SpecStrategy::kNone);
        CHECK(off.reason == SpecDecisionReason::kDeterminismModelDraftOff);
    }
    {
        auto cfg = BaseConfig(SpecStrategy::kDraftModel, 5);
        cfg.deterministic = true;
        SpecController ctl(cfg, clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kCode);
        c.caps.draft_model_available = true;
        CHECK(ctl.Decide(c).reason == SpecDecisionReason::kDeterminismModelDraftOff);
    }
    // 3. History-based drafting needs no second model and stays available.
    {
        auto cfg = BaseConfig(SpecStrategy::kSuffix, 8);
        cfg.deterministic = true;
        SpecController ctl(cfg, clk.fn());
        CHECK(ctl.Decide(Ctx("r", SpecTaskClass::kChat)).mode == SpecStrategy::kSuffix);
        auto cfg2 = BaseConfig(SpecStrategy::kNgramMod, 8);
        cfg2.deterministic = true;
        SpecController ctl2(cfg2, clk.fn());
        CHECK(ctl2.Decide(Ctx("r", SpecTaskClass::kChat)).mode == SpecStrategy::kNgramMod);
    }
    CHECK(IsModelBasedSpecStrategy(SpecStrategy::kDraftMtp));
    CHECK(IsModelBasedSpecStrategy(SpecStrategy::kDraftModel));
    CHECK_FALSE(IsModelBasedSpecStrategy(SpecStrategy::kSuffix));
    CHECK_FALSE(IsModelBasedSpecStrategy(SpecStrategy::kNgram));
    CHECK_FALSE(IsModelBasedSpecStrategy(SpecStrategy::kNgramMod));
    CHECK_FALSE(IsModelBasedSpecStrategy(SpecStrategy::kNone));
    CHECK(IsHistoryBasedSpecStrategy(SpecStrategy::kSuffix));
    CHECK(IsHistoryBasedSpecStrategy(SpecStrategy::kNgramMod));
    CHECK_FALSE(IsHistoryBasedSpecStrategy(SpecStrategy::kDraftModel));
}

TEST_CASE("spec controller: model capability guards and transition/suffix reuse") {
    FakeClock clk;
    // MTP configured but the model has no MTP layers / build lacks MTP.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kDraftMtp, 3), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
        c.caps.mtp_available = false;
        const SpecDecision d = ctl.Decide(c);
        CHECK(d.mode == SpecStrategy::kNone);
        CHECK(d.reason == SpecDecisionReason::kMtpUnavailable);
    }
    // Draft-model configured but no draft model loaded.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kDraftModel, 4), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
        c.caps.draft_model_available = false;
        const SpecDecision d = ctl.Decide(c);
        CHECK(d.mode == SpecStrategy::kNone);
        CHECK(d.reason == SpecDecisionReason::kDraftModelUnavailable);
    }
    // Pure recurrent models cannot roll back a speculative tail.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kSuffix, 8), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
        c.caps.architecture = si::ModelArchitecture::recurrent;
        const SpecDecision d = ctl.Decide(c);
        CHECK(d.mode == SpecStrategy::kNone);
        CHECK(d.reason == SpecDecisionReason::kArchitectureUnsupported);
    }
    // Hybrid with a small recurrent-state cap and no checkpoint support: n_max is clamped to the cap.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kSuffix, 8), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kChat);
        c.caps.architecture = si::ModelArchitecture::hybrid;
        c.caps.hybrid_rs_cap = 4;
        c.caps.can_hybrid_spec_ckpt = false;
        const SpecDecision d = ctl.Decide(c);
        CHECK(d.mode == SpecStrategy::kSuffix);
        CHECK(d.n_max == 4);
    }
    // Bridge to SpeculationTransitionState and SuffixMatcher pause state.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kDraftMtp, 3), clk.fn());
        SpecRequestContext c = Ctx("r", SpecTaskClass::kCode);
        c.caps.mtp_available = true;
        const SpecDecision d = ctl.Decide(c);
        SuffixMatcher matcher(64);
        SlotSpecCapability cap = BuildSlotSpecCapability(d, c.caps, /*suffix_available=*/true, &matcher);
        CHECK(cap.configured_mode == SlotSpecMode::kDraftMtp);
        CHECK(cap.configured_draft_max == 3);
        CHECK(cap.hybrid_suffix_configured);
        CHECK(cap.fallback_suffix_draft_max == static_cast<std::int32_t>(SuffixMatcher::kHybridMtpMaxBurstDraft));
        SpeculationTransitionState state;
        state.Configure(cap);
        CHECK(state.active_mode() == SlotSpecMode::kDraftMtp);
        CHECK(state.active_hybrid_suffix());

        // Drive SuffixMatcher into local cooldown: hybrid burst turns off while MTP stays active.
        for (int i = 0; i < 8; ++i) matcher.Observe(8, 0);
        REQUIRE(matcher.paused());
        const SlotSpecCapability cap2 = BuildSlotSpecCapability(d, c.caps, /*suffix_available=*/true, &matcher);
        CHECK_FALSE(cap2.hybrid_suffix_configured);
        CHECK(cap2.configured_mode == SlotSpecMode::kDraftMtp);
    }
    // A route configured with mode none reports so, without touching acceptance state.
    {
        SpecController ctl(BaseConfig(SpecStrategy::kNone, 0), clk.fn());
        const SpecDecision d = ctl.Decide(Ctx("r", SpecTaskClass::kChat));
        CHECK(d.mode == SpecStrategy::kNone);
        CHECK(d.reason == SpecDecisionReason::kConfiguredNone);
        CHECK(ctl.TrackedEntries() == 0);
    }
}

TEST_CASE("spec controller: acceptance is tracked per strategy, so a config change starts fresh") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());
    ctl.Observe("r", SpecTaskClass::kChat, SpecStrategy::kNgramMod, 16, 0);
    CHECK(ctl.Stats("r", SpecTaskClass::kChat, SpecStrategy::kNgramMod).state == SpecAcceptanceState::kDisabled);
    CHECK(ctl.Stats("r", SpecTaskClass::kChat, SpecStrategy::kSuffix).state == SpecAcceptanceState::kEnabled);
    CHECK(ctl.Decide(Ctx("r", SpecTaskClass::kChat)).mode == SpecStrategy::kSuffix);
}

TEST_CASE("spec controller: tracked entries are bounded (LRU)") {
    FakeClock clk;
    auto cfg = BaseConfig();
    cfg.max_tracked_entries = 3;
    SpecController ctl(cfg, clk.fn());
    const auto T = SpecTaskClass::kChat;
    const auto M = SpecStrategy::kSuffix;
    ctl.Observe("r1", T, M, 8, 4);
    ctl.Observe("r2", T, M, 8, 4);
    ctl.Observe("r3", T, M, 8, 4);
    CHECK(ctl.TrackedEntries() == 3);
    ctl.Observe("r1", T, M, 8, 4);  // refresh r1; r2 is now least recently used
    ctl.Observe("r4", T, M, 8, 4);
    CHECK(ctl.TrackedEntries() == 3);
    CHECK(ctl.Stats("r2", T, M).total_drafted == 0);   // evicted
    CHECK(ctl.Stats("r1", T, M).total_drafted == 16);  // kept
    CHECK(ctl.Stats("r4", T, M).total_drafted == 8);
    // Hostile route strings cannot grow the table.
    for (int i = 0; i < 100; ++i) ctl.Observe("flood-" + std::to_string(i), T, M, 8, 4);
    CHECK(ctl.TrackedEntries() == 3);
    // Over-long route names are truncated to a fixed key length, not rejected.
    const std::string huge(10000, 'x');
    ctl.Observe(huge, T, M, 8, 4);
    CHECK(ctl.Stats(huge, T, M).total_drafted == 8);
}

TEST_CASE("spec controller: slot (-np) and draft advice from v1.0.0 output-row limits") {
    // Speculation off: nothing changes.
    SpecSlotAdvice a = AdviseSpecSlots(8, 0, 512, 4);
    CHECK(a.n_parallel == 8);
    CHECK(a.n_max == 0);
    CHECK_FALSE(a.reduced_slots);
    CHECK_FALSE(a.reduced_draft);

    // Speculation on: default cap matches ConcurrentSpecPolicy::max_concurrent_slots (4).
    a = AdviseSpecSlots(8, 8, 4096);
    CHECK(a.n_parallel == 4);
    CHECK(a.n_max == 8);
    CHECK(a.reduced_slots);
    CHECK_FALSE(a.reduced_draft);

    // A cap of 0 means "do not cap".
    a = AdviseSpecSlots(8, 8, 4096, 0);
    CHECK(a.n_parallel == 8);
    CHECK_FALSE(a.reduced_slots);

    // llama.cpp v1.0.0 needs n_parallel * (1 + n_draft) output rows per decode, bounded by n_batch
    // (common_speculative_get_output_limits). 4 slots * (1 + 16) = 68 > 64 rows: n_max drops to 15.
    a = AdviseSpecSlots(4, 16, 64, 4);
    CHECK(a.n_parallel == 4);
    CHECK(a.n_max == 15);
    CHECK(a.reduced_draft);
    // 4 * (1 + 15) = 64 fits exactly.
    a = AdviseSpecSlots(4, 15, 64, 4);
    CHECK(a.n_max == 15);
    CHECK_FALSE(a.reduced_draft);
    // Not even one draft token fits per slot: speculation is switched off (n_max 0), slots untouched.
    a = AdviseSpecSlots(4, 8, 4, 4);
    CHECK(a.n_max == 0);
    CHECK(a.reduced_draft);
    CHECK(a.n_parallel == 4);
    // One slot, tiny batch.
    a = AdviseSpecSlots(1, 8, 4, 4);
    CHECK(a.n_max == 3);
    // Degenerate inputs never produce zero slots or negative drafts.
    a = AdviseSpecSlots(0, -5, 0, 4);
    CHECK(a.n_parallel == 1);
    CHECK(a.n_max == 0);
}

TEST_CASE("spec controller: route MTP hint, max planned draft, and ModelParamsFor load_mtp") {
    SpecControllerConfig cfg;
    std::string err;
    CHECK_FALSE(cfg.MayUseMtp());
    CHECK(cfg.MaxPlannedDraft() == 0);

    // Base spec_type == "none" + route rule selecting draft-mtp still signals MayUseMtp().
    si::LlamaCppBackendOptions bo;
    bo.spec_type = "none";
    bo.spec_route_rules = {"code=draft-mtp,n_max=3", "tool-call@m1=suffix,n_max=16"};
    REQUIRE(BuildSpecControllerConfig(bo, cfg, err));
    CHECK(cfg.MayUseMtp());
    CHECK(cfg.MaxPlannedDraft() == 16);

    // Determinism mode suppresses MayUseMtp() even when rules mention draft-mtp.
    bo.deterministic = true;
    REQUIRE(BuildSpecControllerConfig(bo, cfg, err));
    CHECK_FALSE(cfg.MayUseMtp());

    // ModelParamsFor propagates spec_may_use_mtp -> load_mtp when built with MTP support.
    LoadOptions lo;
    lo.spec_type = "none";
    lo.spec_may_use_mtp = false;
    CHECK_FALSE(LlamaCppBackend::ModelParamsFor(lo).load_mtp);
    lo.spec_may_use_mtp = true;
    CHECK(LlamaCppBackend::ModelParamsFor(lo).load_mtp == LlamaCppBackend::HasMtpBuildSupport());
}

TEST_CASE("spec controller: Decide enforces v1.0.0 batch output-row bound via AdviseSpecSlots") {
    FakeClock clk;
    SpecController ctl(BaseConfig(SpecStrategy::kSuffix, 16), clk.fn());

    SpecRequestContext c = Ctx("r", SpecTaskClass::kCode);
    c.n_slots = 4;
    c.n_batch = 64;  // 4 * (1 + 15) = 64 => clamps 16 down to 15
    SpecDecision d = ctl.Decide(c);
    CHECK(d.mode == SpecStrategy::kSuffix);
    CHECK(d.n_max == 15);
    CHECK(d.reason == SpecDecisionReason::kConfigured);

    c.n_batch = 4;  // 4 / 4 - 1 = 0 => cannot fit any draft token per slot
    d = ctl.Decide(c);
    CHECK(d.mode == SpecStrategy::kNone);
    CHECK(d.n_max == 0);
    CHECK(d.reason == SpecDecisionReason::kBatchOutputLimit);
}

TEST_CASE("spec controller: concurrent probe cancel does not race and Stats retry_in_ms saturates") {
    FakeClock clk;
    SpecController ctl(BaseConfig(), clk.fn());
    const auto ctx = Ctx("r", SpecTaskClass::kChat);
    const auto T = SpecTaskClass::kChat;
    const auto M = SpecStrategy::kSuffix;

    // Disable route, advance past retry_at_ms.
    ctl.Observe("r", T, M, 16, 0);
    clk.now_ms += 1000;

    // Request 1 acquires single-probe ownership; concurrent Request 2 is refused speculation.
    const SpecDecision r1 = ctl.Decide(ctx);
    REQUIRE(r1.is_probe);
    REQUIRE(r1.probe_id > 0);
    const SpecDecision r2 = ctl.Decide(ctx);
    CHECK_FALSE(r2.is_probe);
    CHECK(r2.mode == SpecStrategy::kNone);
    CHECK(r2.reason == SpecDecisionReason::kAcceptanceDisabled);

    // Request 1 cancels without drafting; no second probe existed to lose its observation.
    ctl.CancelProbe("r", T, M, r1.probe_id);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kDisabled);
    CHECK(ctl.Stats("r", T, M).ignored_observations == 0);

    // Next retry succeeds cleanly.
    clk.now_ms += 1000;
    const SpecDecision r3 = ctl.Decide(ctx);
    REQUIRE(r3.is_probe);
    ctl.Observe("r", T, M, 16, 16);
    CHECK(ctl.Stats("r", T, M).state == SpecAcceptanceState::kEnabled);

    // Signed overflow protection in Stats().retry_in_ms when retry_at_ms == INT64_MAX and clock < 0.
    FakeClock extreme_clk;
    extreme_clk.now_ms = std::numeric_limits<std::int64_t>::max() - 10;
    SpecController extreme_ctl(BaseConfig(), extreme_clk.fn());
    extreme_ctl.Observe("r", T, M, 16, 0);  // retry_at_ms saturates at INT64_MAX
    extreme_clk.now_ms = -1000;
    const SpecRouteStats st = extreme_ctl.Stats("r", T, M);
    CHECK(st.state == SpecAcceptanceState::kDisabled);
    CHECK(st.retry_in_ms == std::numeric_limits<std::int64_t>::max());
}

TEST_CASE("spec controller: NgramModTable Configure updates per-request draft bounds") {
    NgramModTable table(NgramModTable::kDefaultMatchLen, 1024, 4, 4);
    CHECK(table.n_min() == 4);
    CHECK(table.n_max() == 4);
    table.Configure(48, 64);
    CHECK(table.n_min() == 48);
    CHECK(table.n_max() == 64);
    table.Configure(8, 8);
    CHECK(table.n_min() == 8);
    CHECK(table.n_max() == 8);
}
