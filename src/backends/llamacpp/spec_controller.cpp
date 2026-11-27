// Per-route speculative-decoding controller. See spec_controller.hpp.
#include "spec_controller.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <system_error>

namespace sonder::backends::llamacpp {

namespace {

std::string LowerNormalized(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        c = (c == '_') ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string_view Trim(std::string_view s) noexcept {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

bool ParseInt32(std::string_view text, std::int32_t& out) noexcept {
    if (text.empty()) return false;
    std::int32_t v = 0;
    const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    if (r.ec != std::errc() || r.ptr != text.data() + text.size()) return false;
    out = v;
    return true;
}

bool ParseDouble(std::string_view text, double& out) noexcept {
    if (text.empty()) return false;
    double v = 0.0;
    const auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    if (r.ec != std::errc() || r.ptr != text.data() + text.size()) return false;
    out = v;
    return true;
}

void ApplyOverlay(SpecPolicy& p, const SpecPolicyOverlay& o, bool& mode_set) noexcept {
    if (o.mode) {
        p.mode = *o.mode;
        mode_set = true;
    }
    if (o.n_max) p.n_max = *o.n_max;
    if (o.p_min) p.p_min = *o.p_min;
}

void MergeOverlay(std::optional<SpecPolicyOverlay>& slot, const SpecPolicyOverlay& o) {
    if (!slot) {
        slot = o;
        return;
    }
    if (o.mode) slot->mode = o.mode;
    if (o.n_max) slot->n_max = o.n_max;
    if (o.p_min) slot->p_min = o.p_min;
}

bool IsUnit(double v) noexcept { return std::isfinite(v) && v >= 0.0 && v <= 1.0; }

std::string CheckOverlay(const SpecPolicyOverlay& o, const std::string& where) {
    if (o.n_max && (*o.n_max < 0 || *o.n_max > kMaxSpecDraftMax)) {
        return where + ": n_max must be in [0, " + std::to_string(kMaxSpecDraftMax) + "]";
    }
    if (o.p_min && !IsUnit(*o.p_min)) {
        return where + ": p_min must be in [0, 1]";
    }
    return {};
}

std::int64_t SaturatingAdd(std::int64_t a, std::int64_t b) noexcept {
    if (b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return a + b;
}

std::int64_t SaturatingSubNonNeg(std::int64_t a, std::int64_t b) noexcept {
    if (a <= b) return 0;
    if (b < 0 && a > std::numeric_limits<std::int64_t>::max() + b) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return a - b;
}

bool ChatToolsRequested(const sonder::inference::ChatRequest& request) noexcept {
    return !request.tools.empty() &&
           (!request.tool_choice.has_value() ||
            request.tool_choice->mode != sonder::inference::ToolChoiceMode::none);
}

bool ChatThinkingEnabled(const sonder::inference::ChatRequest& request) noexcept {
    if (request.separate_reasoning) return true;
    if (request.thinking.enable_thinking.has_value()) return *request.thinking.enable_thinking;
    if (request.thinking.reasoning_effort.has_value() && !request.thinking.reasoning_effort->empty() &&
        *request.thinking.reasoning_effort != "none") {
        return true;
    }
    return false;
}

bool MessageHasCodeMarkers(const sonder::inference::ChatMessage& msg) noexcept {
    if (!msg.parts.empty()) {
        for (const auto& part : msg.parts) {
            if (const auto* text = std::get_if<sonder::inference::TextContentPart>(&part)) {
                if (ContainsCodeOrDiffMarkers(text->text)) return true;
            }
        }
        return false;
    }
    return ContainsCodeOrDiffMarkers(msg.content);
}

}  // namespace

// --- strategies and task classes -------------------------------------------

bool ParseSpecStrategy(std::string_view text, SpecStrategy& out) noexcept {
    std::string s;
    try {
        s = LowerNormalized(text);
    } catch (...) {
        return false;
    }
    if (s == "none") {
        out = SpecStrategy::kNone;
    } else if (s == "ngram") {
        out = SpecStrategy::kNgram;
    } else if (s == "ngram-mod") {
        out = SpecStrategy::kNgramMod;
    } else if (s == "suffix") {
        out = SpecStrategy::kSuffix;
    } else if (s == "draft-model" || s == "draft-simple" || s == "draft") {
        out = SpecStrategy::kDraftModel;
    } else if (s == "draft-mtp" || s == "mtp") {
        out = SpecStrategy::kDraftMtp;
    } else {
        return false;
    }
    return true;
}

bool ParseSpecTaskClass(std::string_view text, SpecTaskClass& out) noexcept {
    std::string s;
    try {
        s = LowerNormalized(text);
    } catch (...) {
        return false;
    }
    if (s == "chat" || s == "prose" || s == "general") {
        out = SpecTaskClass::kChat;
    } else if (s == "code" || s == "code-edit") {
        out = SpecTaskClass::kCode;
    } else if (s == "tool-call" || s == "tool-json") {
        out = SpecTaskClass::kToolCall;
    } else if (s == "reasoning") {
        out = SpecTaskClass::kReasoning;
    } else {
        return false;
    }
    return true;
}

SpecTaskClass ClassifySpecTask(const SpecTaskFeatures& f) noexcept {
    if (f.has_grammar || f.tools_requested || f.structured_output) return SpecTaskClass::kToolCall;
    if (f.thinking_enabled) return SpecTaskClass::kReasoning;
    if (f.has_code_markers) return SpecTaskClass::kCode;
    return SpecTaskClass::kChat;
}

bool ContainsCodeOrDiffMarkers(std::string_view text) noexcept {
    return text.find("```") != std::string_view::npos || text.find("diff --git") != std::string_view::npos ||
           text.find("@@ ") != std::string_view::npos;
}

SpecTaskClass ClassifyGenerateRequest(const sonder::inference::GenerateRequest& request,
                                      std::string_view grammar) noexcept {
    SpecTaskFeatures f;
    f.has_grammar = !grammar.empty() || !request.grammar.empty();
    f.has_code_markers = ContainsCodeOrDiffMarkers(request.prompt);
    return ClassifySpecTask(f);
}

SpecTaskClass ClassifyChatRequest(const sonder::inference::ChatRequest& request, std::string_view grammar) noexcept {
    SpecTaskFeatures f;
    f.has_grammar = !grammar.empty() || !request.grammar.empty();
    f.tools_requested = ChatToolsRequested(request);
    f.structured_output = request.response_format_type.has_value() && *request.response_format_type != "text";
    f.thinking_enabled = ChatThinkingEnabled(request);
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "user") {
            f.has_code_markers = MessageHasCodeMarkers(*it);
            break;
        }
    }
    return ClassifySpecTask(f);
}

// --- config parsing --------------------------------------------------------

bool ParseSpecPolicyOverlay(std::string_view text, SpecPolicyOverlay& out, std::string& error) {
    SpecPolicyOverlay result;
    std::string_view rest = text;
    bool first = true;
    while (true) {
        const std::size_t comma = rest.find(',');
        const std::string_view token = Trim(rest.substr(0, comma));
        if (token.empty()) {
            error = first && comma == std::string_view::npos ? "empty speculation policy" : "empty field in policy";
            return false;
        }
        std::string_view key;
        std::string_view value;
        const std::size_t eq = token.find('=');
        if (eq == std::string_view::npos) {
            if (!first) {
                error = "only the first field may omit its key: '" + std::string(token) + "'";
                return false;
            }
            key = "mode";
            value = token;
        } else {
            key = Trim(token.substr(0, eq));
            value = Trim(token.substr(eq + 1));
        }
        const std::string k = LowerNormalized(key);
        if (k == "mode" || k == "type") {
            if (result.mode) {
                error = "mode given twice";
                return false;
            }
            SpecStrategy m = SpecStrategy::kNone;
            if (!ParseSpecStrategy(value, m)) {
                error = "unknown speculation type '" + std::string(value) +
                        "' (expected none, ngram, ngram-mod, suffix, draft-model or draft-mtp)";
                return false;
            }
            result.mode = m;
        } else if (k == "n-max") {
            if (result.n_max) {
                error = "n_max given twice";
                return false;
            }
            std::int32_t n = 0;
            if (!ParseInt32(value, n) || n < 0 || n > kMaxSpecDraftMax) {
                error = "n_max must be an integer in [0, " + std::to_string(kMaxSpecDraftMax) + "]";
                return false;
            }
            result.n_max = n;
        } else if (k == "p-min") {
            if (result.p_min) {
                error = "p_min given twice";
                return false;
            }
            double p = 0.0;
            if (!ParseDouble(value, p) || !IsUnit(p)) {
                error = "p_min must be a number in [0, 1]";
                return false;
            }
            result.p_min = p;
        } else {
            error = "unknown policy field '" + std::string(key) + "' (expected mode, n_max or p_min)";
            return false;
        }
        first = false;
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
    }
    if (result.mode && !IsModelBasedSpecStrategy(*result.mode) && result.p_min && *result.p_min > 0.0) {
        error = "p_min > 0 is only supported for model-based strategies (draft-model, draft-mtp)";
        return false;
    }
    out = result;
    return true;
}

void SpecControllerConfig::SetRule(std::string_view route, std::optional<SpecTaskClass> task,
                                   const SpecPolicyOverlay& overlay) {
    if (route.empty()) {
        if (task) {
            MergeOverlay(class_rules[static_cast<std::size_t>(*task)], overlay);
        } else {
            bool ignored = false;
            ApplyOverlay(base, overlay, ignored);
            if (overlay.mode) auto_base = false;
        }
        return;
    }
    SpecRouteRules& rules = routes[std::string(route)];
    if (task) {
        MergeOverlay(rules.by_class[static_cast<std::size_t>(*task)], overlay);
    } else {
        MergeOverlay(rules.any, overlay);
    }
}

bool ParseSpecRule(std::string_view text, SpecControllerConfig& config, std::string& error) {
    const std::size_t eq = text.find('=');
    if (eq == std::string_view::npos) {
        error = "rule must look like <selector>=<policy>";
        return false;
    }
    const std::string_view selector = Trim(text.substr(0, eq));
    if (selector.empty()) {
        error = "empty selector (use '*' for the base policy)";
        return false;
    }
    std::string_view class_part = selector;
    std::string_view route_part;
    bool has_route = false;
    if (const std::size_t at = selector.find('@'); at != std::string_view::npos) {
        class_part = Trim(selector.substr(0, at));
        route_part = Trim(selector.substr(at + 1));
        has_route = true;
        if (route_part.empty()) {
            error = "empty route after '@'";
            return false;
        }
        if (route_part.find('@') != std::string_view::npos) {
            error = "route names must not contain '@'";
            return false;
        }
        if (route_part.size() > kMaxSpecRouteKeyLength) {
            error = "route name longer than " + std::to_string(kMaxSpecRouteKeyLength) + " bytes";
            return false;
        }
    }
    std::optional<SpecTaskClass> task;
    if (!class_part.empty() && class_part != "*") {
        SpecTaskClass c = SpecTaskClass::kChat;
        if (!ParseSpecTaskClass(class_part, c)) {
            error = "unknown task class '" + std::string(class_part) +
                    "' (expected chat, code, tool-call, reasoning or *)";
            return false;
        }
        task = c;
    }
    if (has_route && route_part == "*") {
        route_part = {};
    }
    SpecPolicyOverlay overlay;
    if (!ParseSpecPolicyOverlay(text.substr(eq + 1), overlay, error)) {
        return false;
    }
    config.SetRule(route_part, task, overlay);
    return true;
}

bool BuildSpecControllerConfig(const sonder::inference::LlamaCppBackendOptions& options, SpecControllerConfig& out,
                               std::string& error) {
    SpecControllerConfig cfg;
    const std::string lower_spec = LowerNormalized(options.spec_type);
    if (lower_spec.empty() || lower_spec == "none") {
        cfg.base.mode = SpecStrategy::kNone;
    } else if (lower_spec == "auto") {
        cfg.auto_base = true;
        cfg.base.mode = SpecStrategy::kNone;
    } else if (!ParseSpecStrategy(options.spec_type, cfg.base.mode)) {
        error = "unknown spec_type '" + options.spec_type + "'";
        return false;
    }
    if (options.spec_draft_max < 0 || options.spec_draft_max > kMaxSpecDraftMax) {
        error = "spec_draft_max must be in [0, " + std::to_string(kMaxSpecDraftMax) + "]";
        return false;
    }
    cfg.base.n_max = options.spec_draft_max;
    cfg.base.p_min = options.spec_p_min;
    cfg.deterministic = options.deterministic;
    cfg.max_spec_slots = options.spec_batch_max_slots;

    for (const std::string& rule : options.spec_route_rules) {
        if (!ParseSpecRule(rule, cfg, error)) {
            error = "spec_route_rules: " + error;
            return false;
        }
    }
    if (std::string v = cfg.Validate(); !v.empty()) {
        error = std::move(v);
        return false;
    }
    out = std::move(cfg);
    return true;
}

std::string SpecControllerConfig::Validate() const {
    if (auto e = CheckOverlay({base.mode, base.n_max, base.p_min}, "base"); !e.empty()) return e;
    if (!auto_base && !IsModelBasedSpecStrategy(base.mode) && base.p_min > 0.0) {
        return "base: p_min > 0 is only supported for model-based strategies (draft-model, draft-mtp)";
    }
    const auto check_overlay_pmin = [](const SpecPolicyOverlay& o, SpecStrategy inherited_mode, bool inherited_auto,
                                       const std::string& where) -> std::string {
        const SpecStrategy eff_mode = o.mode.value_or(inherited_mode);
        const bool eff_auto = !o.mode.has_value() && inherited_auto;
        if (o.p_min && *o.p_min > 0.0 && !eff_auto && !IsModelBasedSpecStrategy(eff_mode)) {
            return where + ": p_min > 0 is only supported for model-based strategies (draft-model, draft-mtp)";
        }
        return {};
    };
    for (std::size_t i = 0; i < class_rules.size(); ++i) {
        if (class_rules[i]) {
            const std::string where = std::string("class ") + std::string(ToString(static_cast<SpecTaskClass>(i)));
            if (auto e = CheckOverlay(*class_rules[i], where); !e.empty()) return e;
            if (auto e = check_overlay_pmin(*class_rules[i], base.mode, auto_base, where); !e.empty()) return e;
        }
    }
    for (const auto& [route, rules] : routes) {
        if (route.empty() || route.size() > kMaxSpecRouteKeyLength) {
            return "route names must be 1.." + std::to_string(kMaxSpecRouteKeyLength) + " bytes";
        }
        if (rules.any) {
            if (auto e = CheckOverlay(*rules.any, "route " + route); !e.empty()) return e;
            if (auto e = check_overlay_pmin(*rules.any, base.mode, auto_base, "route " + route); !e.empty()) return e;
        }
        for (std::size_t i = 0; i < rules.by_class.size(); ++i) {
            if (rules.by_class[i]) {
                const std::string where = "route " + route;
                if (auto e = CheckOverlay(*rules.by_class[i], where); !e.empty()) return e;
                SpecStrategy inherited = base.mode;
                bool inherited_auto = auto_base;
                if (class_rules[i] && class_rules[i]->mode) {
                    inherited = *class_rules[i]->mode;
                    inherited_auto = false;
                }
                if (rules.any && rules.any->mode) {
                    inherited = *rules.any->mode;
                    inherited_auto = false;
                }
                if (auto e = check_overlay_pmin(*rules.by_class[i], inherited, inherited_auto, where); !e.empty()) {
                    return e;
                }
            }
        }
    }
    const SpecAcceptanceConfig& a = acceptance;
    if (a.window_steps < 1 || a.window_steps > 4096) return "acceptance.window_steps must be in [1, 4096]";
    if (a.min_drafted_tokens < 1) return "acceptance.min_drafted_tokens must be >= 1";
    if (!IsUnit(a.disable_below)) return "acceptance.disable_below must be in [0, 1]";
    if (!IsUnit(a.reenable_at_least) || a.reenable_at_least < a.disable_below) {
        return "acceptance.reenable_at_least must be in [disable_below, 1]";
    }
    if (a.probe_drafted_tokens < 1) return "acceptance.probe_drafted_tokens must be >= 1";
    if (a.reprobe_initial_ms < 1) return "acceptance.reprobe_initial_ms must be >= 1";
    if (a.reprobe_max_ms < a.reprobe_initial_ms) return "acceptance.reprobe_max_ms must be >= reprobe_initial_ms";
    if (!std::isfinite(a.backoff_factor) || a.backoff_factor < 1.0 || a.backoff_factor > 16.0) {
        return "acceptance.backoff_factor must be in [1, 16]";
    }
    if (a.probe_timeout_ms < 1) return "acceptance.probe_timeout_ms must be >= 1";
    if (max_tracked_entries < 1 || max_tracked_entries > 65536) return "max_tracked_entries must be in [1, 65536]";
    if (max_spec_slots > kMaxSlots) return "max_spec_slots must be in [0, " + std::to_string(kMaxSlots) + "]";
    return {};
}

SpecPolicy SpecControllerConfig::Resolve(std::string_view route, SpecTaskClass task, bool& mode_from_rule) const {
    const std::size_t ci = static_cast<std::size_t>(task);
    SpecPolicy p = base;
    mode_from_rule = false;
    if (class_rules[ci]) ApplyOverlay(p, *class_rules[ci], mode_from_rule);
    if (!route.empty()) {
        if (const auto it = routes.find(std::string(route)); it != routes.end()) {
            if (it->second.any) ApplyOverlay(p, *it->second.any, mode_from_rule);
            if (it->second.by_class[ci]) ApplyOverlay(p, *it->second.by_class[ci], mode_from_rule);
        }
    }
    if (p.mode == SpecStrategy::kNone) {
        if (!auto_base || mode_from_rule) {
            p.n_max = 0;
        }
    } else if (p.n_max <= 0) {
        p.n_max = DefaultSpecDraftMax(p.mode);
    }
    return p;
}

SpecPolicy SpecControllerConfig::Resolve(std::string_view route, SpecTaskClass task) const {
    bool ignored = false;
    return Resolve(route, task, ignored);
}

bool SpecControllerConfig::MayUseMtp() const noexcept {
    if (deterministic) return false;
    if (auto_base || base.mode == SpecStrategy::kDraftMtp) return true;
    for (const auto& cr : class_rules) {
        if (cr && cr->mode == SpecStrategy::kDraftMtp) return true;
    }
    for (const auto& [route, rules] : routes) {
        (void)route;
        if (rules.any && rules.any->mode == SpecStrategy::kDraftMtp) return true;
        for (const auto& r : rules.by_class) {
            if (r && r->mode == SpecStrategy::kDraftMtp) return true;
        }
    }
    return false;
}

std::int32_t SpecControllerConfig::MaxPlannedDraft() const noexcept {
    std::int32_t best = 0;
    const auto consider = [&](std::string_view route, SpecTaskClass task) {
        bool mode_from_rule = false;
        SpecPolicy p = Resolve(route, task, mode_from_rule);
        if (auto_base && !mode_from_rule) {
            const std::int32_t auto_max =
                p.n_max > 0 ? p.n_max
                            : std::max(DefaultSpecDraftMax(SpecStrategy::kDraftMtp),
                                       DefaultSpecDraftMax(SpecStrategy::kNgramMod));
            best = std::max(best, auto_max);
            return;
        }
        if (p.mode != SpecStrategy::kNone) {
            best = std::max(best, p.n_max);
        }
    };
    for (std::size_t i = 0; i < kSpecTaskClassCount; ++i) {
        const auto tc = static_cast<SpecTaskClass>(i);
        consider({}, tc);
        for (const auto& [route, rules] : routes) {
            (void)rules;
            consider(route, tc);
        }
    }
    return best;
}

std::int32_t SpecControllerConfig::MaxMtpRouteDraft() const noexcept {
    if (deterministic) return 0;
    std::int32_t best = 0;
    const auto consider = [&](std::string_view route, SpecTaskClass task) {
        bool mode_from_rule = false;
        const SpecPolicy p = Resolve(route, task, mode_from_rule);
        if (mode_from_rule && p.mode == SpecStrategy::kDraftMtp) {
            best = std::max(best, p.n_max);
        }
    };
    for (std::size_t i = 0; i < kSpecTaskClassCount; ++i) {
        const auto tc = static_cast<SpecTaskClass>(i);
        consider({}, tc);
        for (const auto& [route, rules] : routes) {
            (void)rules;
            consider(route, tc);
        }
    }
    return best;
}

// --- bridge to speculation_transition.hpp & suffix_matcher.hpp -------------

SlotSpecCapability BuildSlotSpecCapability(const SpecDecision& decision, const SpecModelCaps& caps,
                                           bool suffix_available, const SuffixMatcher* suffix_matcher) noexcept {
    SlotSpecCapability cap;
    cap.architecture = caps.architecture;
    cap.hybrid_rs_cap = caps.hybrid_rs_cap;
    cap.can_hybrid_spec_ckpt = caps.can_hybrid_spec_ckpt;
    cap.fallback_suffix_draft_max = static_cast<std::int32_t>(SuffixMatcher::kHybridMtpMaxBurstDraft);

    const SlotSpecMode slot_mode = ToSlotSpecMode(decision.mode);
    if (slot_mode == SlotSpecMode::kNone || decision.n_max <= 0) {
        return cap;
    }
    cap.configured_mode = slot_mode;
    cap.configured_draft_max = decision.n_max;
    if (slot_mode == SlotSpecMode::kDraftMtp && suffix_available) {
        const bool burst_ok = (suffix_matcher == nullptr) || (!suffix_matcher->paused() && !suffix_matcher->cost_paused());
        cap.hybrid_suffix_configured = burst_ok;
        cap.suffix_fallback_available = true;
    }
    return cap;
}

// --- controller ------------------------------------------------------------

SpecController::SpecController(SpecControllerConfig config, SpecClock clock)
    : config_(std::move(config)), clock_(std::move(clock)) {
    SpecAcceptanceConfig& a = config_.acceptance;
    a.window_steps = std::clamp<std::size_t>(a.window_steps, 1, 4096);
    a.min_drafted_tokens = std::max<std::uint32_t>(a.min_drafted_tokens, 1);
    a.probe_drafted_tokens = std::max<std::uint32_t>(a.probe_drafted_tokens, 1);
    a.reprobe_initial_ms = std::max<std::int64_t>(a.reprobe_initial_ms, 1);
    a.reprobe_max_ms = std::max(a.reprobe_max_ms, a.reprobe_initial_ms);
    if (!std::isfinite(a.backoff_factor) || a.backoff_factor < 1.0) a.backoff_factor = 1.0;
    a.backoff_factor = std::min(a.backoff_factor, 16.0);
    a.probe_timeout_ms = std::max<std::int64_t>(a.probe_timeout_ms, 1);
    config_.max_tracked_entries = std::clamp<std::size_t>(config_.max_tracked_entries, 1, 65536);
    if (!clock_) {
        clock_ = []() {
            return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 std::chrono::steady_clock::now().time_since_epoch())
                                                 .count());
        };
    }
}

std::string SpecController::MakeKey(std::string_view route, SpecTaskClass task, SpecStrategy mode) {
    std::string key(route.substr(0, kMaxSpecRouteKeyLength));
    key.push_back('\x1f');
    key.push_back(static_cast<char>('0' + static_cast<int>(task)));
    key.push_back('\x1f');
    key.push_back(static_cast<char>('0' + static_cast<int>(mode)));
    return key;
}

std::int64_t SpecController::Now() const { return clock_(); }

SpecController::Entry& SpecController::FindOrCreateLocked(const std::string& key) {
    if (auto it = entries_.find(key); it != entries_.end()) {
        it->second.last_used = ++tick_;
        return it->second;
    }
    while (entries_.size() >= config_.max_tracked_entries) {
        auto victim = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.last_used < victim->second.last_used) victim = it;
        }
        entries_.erase(victim);
    }
    Entry& e = entries_[key];
    e.last_used = ++tick_;
    return e;
}

void SpecController::EnterDisabledLocked(Entry& e, std::int64_t backoff_ms) {
    e.state = SpecAcceptanceState::kDisabled;
    e.backoff_ms = backoff_ms;
    e.retry_at_ms = SaturatingAdd(Now(), backoff_ms);
    e.steps.clear();
    e.window_drafted = 0;
    e.window_accepted = 0;
    e.probe_drafted = 0;
    e.probe_accepted = 0;
    e.probe_in_flight = false;
    e.active_probe_id = 0;
    e.probe_started_ms = 0;
}

SpecDecision SpecController::Decide(const SpecRequestContext& ctx) {
    bool mode_from_rule = false;
    SpecPolicy policy = config_.Resolve(ctx.route, ctx.task, mode_from_rule);
    if (config_.auto_base && !mode_from_rule) {
        policy.mode = ctx.caps.mtp_available ? SpecStrategy::kDraftMtp : SpecStrategy::kNgramMod;
        if (policy.n_max <= 0) {
            policy.n_max = DefaultSpecDraftMax(policy.mode);
        }
    }

    SpecDecision d;
    d.configured_mode = policy.mode;
    d.p_min = IsModelBasedSpecStrategy(policy.mode) ? policy.p_min : 0.0;
    if (policy.mode == SpecStrategy::kNone) {
        d.reason = SpecDecisionReason::kConfiguredNone;
        d.p_min = 0.0;
        return d;
    }
    const auto off = [&](SpecDecisionReason why) {
        SpecDecision o;
        o.configured_mode = policy.mode;
        o.reason = why;
        return o;
    };
    if ((ctx.deterministic || config_.deterministic) && IsModelBasedSpecStrategy(policy.mode)) {
        return off(SpecDecisionReason::kDeterminismModelDraftOff);
    }
    if (policy.mode == SpecStrategy::kDraftMtp && !ctx.caps.mtp_available) {
        return off(SpecDecisionReason::kMtpUnavailable);
    }
    if (policy.mode == SpecStrategy::kDraftModel && !ctx.caps.draft_model_available) {
        return off(SpecDecisionReason::kDraftModelUnavailable);
    }

    // CheckArchitectureSpecGuard treats model-based drafting (MTP / draft-model)
    // identically on hybrid/recurrent models: pure recurrent cannot roll back, and
    // hybrid requires recurrent-state room (hybrid_rs_cap > 0).
    const SlotSpecMode guard_mode =
        (policy.mode == SpecStrategy::kDraftModel) ? SlotSpecMode::kDraftMtp : ToSlotSpecMode(policy.mode);
    const ArchitectureSpecGuardResult guard =
        CheckArchitectureSpecGuard(ctx.caps.architecture, guard_mode, policy.n_max, ctx.caps.hybrid_rs_cap,
                                   ctx.caps.can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/true);
    if (!guard.allowed || guard.effective_draft_max <= 0) {
        return off(SpecDecisionReason::kArchitectureUnsupported);
    }

    std::int32_t effective_n_max = guard.effective_draft_max;
    if (ctx.caps.ctx_outputs_max_per_seq > 0) {
        const std::int32_t max_from_ctx = ctx.caps.ctx_outputs_max_per_seq - 1;
        if (max_from_ctx <= 0) {
            return off(SpecDecisionReason::kBatchOutputLimit);
        }
        effective_n_max = std::min(effective_n_max, max_from_ctx);
    }

    d.mode = policy.mode;
    d.n_max = effective_n_max;
    d.reason = SpecDecisionReason::kConfigured;

    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(MakeKey(ctx.route, ctx.task, policy.mode));
    if (it == entries_.end()) return d;
    Entry& e = it->second;
    e.last_used = ++tick_;
    switch (e.state) {
        case SpecAcceptanceState::kEnabled: break;
        case SpecAcceptanceState::kDisabled: {
            const std::int64_t now = Now();
            if (now < e.retry_at_ms) {
                return off(SpecDecisionReason::kAcceptanceDisabled);
            }
            e.state = SpecAcceptanceState::kProbing;
            e.probe_drafted = 0;
            e.probe_accepted = 0;
            e.probe_in_flight = true;
            e.active_probe_id = ++next_probe_id_;
            e.probe_started_ms = now;
            ++e.probe_count;
            d.is_probe = true;
            d.probe_id = e.active_probe_id;
            d.reason = SpecDecisionReason::kProbe;
            break;
        }
        case SpecAcceptanceState::kProbing: {
            const std::int64_t now = Now();
            if (e.probe_in_flight) {
                // Safety-net timeout: reclaim stale probe ownership if a caller never reported back.
                if (SaturatingSubNonNeg(now, e.probe_started_ms) < config_.acceptance.probe_timeout_ms) {
                    return off(SpecDecisionReason::kAcceptanceDisabled);
                }
                e.probe_in_flight = false;
                e.active_probe_id = 0;
                e.probe_started_ms = 0;
            }
            // Previous probe request completed with partial/zero evidence (< probe_drafted_tokens)
            // or timed out: grant ownership to this request without incrementing probe_count again.
            e.probe_in_flight = true;
            e.active_probe_id = ++next_probe_id_;
            e.probe_started_ms = now;
            d.is_probe = true;
            d.probe_id = e.active_probe_id;
            d.reason = SpecDecisionReason::kProbe;
            break;
        }
    }
    return d;
}

void SpecController::Observe(std::string_view route, SpecTaskClass task, SpecStrategy mode, std::uint32_t drafted,
                             std::uint32_t accepted) {
    if (mode == SpecStrategy::kNone || drafted == 0) return;
    accepted = std::min(accepted, drafted);
    const SpecAcceptanceConfig& a = config_.acceptance;

    std::lock_guard<std::mutex> lock(mutex_);
    Entry& e = FindOrCreateLocked(MakeKey(route, task, mode));
    switch (e.state) {
        case SpecAcceptanceState::kDisabled:
            ++e.ignored;
            return;
        case SpecAcceptanceState::kEnabled: {
            e.total_drafted += drafted;
            e.total_accepted += accepted;
            e.steps.emplace_back(drafted, accepted);
            e.window_drafted += drafted;
            e.window_accepted += accepted;
            while (e.steps.size() > a.window_steps) {
                e.window_drafted -= e.steps.front().first;
                e.window_accepted -= e.steps.front().second;
                e.steps.pop_front();
            }
            if (e.window_drafted >= a.min_drafted_tokens) {
                const double rate = static_cast<double>(e.window_accepted) / static_cast<double>(e.window_drafted);
                if (rate < a.disable_below) {
                    ++e.disable_count;
                    EnterDisabledLocked(e, a.reprobe_initial_ms);
                }
            }
            return;
        }
        case SpecAcceptanceState::kProbing: {
            e.total_drafted += drafted;
            e.total_accepted += accepted;
            e.probe_drafted += drafted;
            e.probe_accepted += accepted;
            e.probe_in_flight = false;
            e.active_probe_id = 0;
            e.probe_started_ms = 0;
            if (e.probe_drafted < a.probe_drafted_tokens) return;
            const double rate = static_cast<double>(e.probe_accepted) / static_cast<double>(e.probe_drafted);
            if (rate >= a.reenable_at_least) {
                e.state = SpecAcceptanceState::kEnabled;
                e.backoff_ms = 0;
                e.retry_at_ms = 0;
                e.steps.clear();
                e.window_drafted = 0;
                e.window_accepted = 0;
                e.probe_drafted = 0;
                e.probe_accepted = 0;
            } else {
                ++e.failed_probes;
                const double grown = static_cast<double>(e.backoff_ms) * a.backoff_factor;
                const std::int64_t next =
                    (!std::isfinite(grown) || grown >= static_cast<double>(a.reprobe_max_ms))
                        ? a.reprobe_max_ms
                        : std::max(static_cast<std::int64_t>(grown), a.reprobe_initial_ms);
                EnterDisabledLocked(e, std::min(next, a.reprobe_max_ms));
            }
            return;
        }
    }
}

void SpecController::CancelProbe(std::string_view route, SpecTaskClass task, SpecStrategy mode,
                                 std::uint64_t probe_id) {
    if (mode == SpecStrategy::kNone) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(MakeKey(route, task, mode));
    if (it == entries_.end()) return;
    Entry& e = it->second;
    if (e.state != SpecAcceptanceState::kProbing) return;
    if (probe_id != 0 && probe_id != e.active_probe_id) return;
    e.last_used = ++tick_;
    const std::int64_t keep_backoff =
        e.backoff_ms > 0 ? std::min(e.backoff_ms, config_.acceptance.reprobe_max_ms) : config_.acceptance.reprobe_initial_ms;
    EnterDisabledLocked(e, keep_backoff);
}

void SpecController::ReleaseProbe(std::string_view route, SpecTaskClass task, SpecStrategy mode,
                                  std::uint64_t probe_id) {
    if (mode == SpecStrategy::kNone) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(MakeKey(route, task, mode));
    if (it == entries_.end()) return;
    Entry& e = it->second;
    if (e.state != SpecAcceptanceState::kProbing || !e.probe_in_flight) return;
    if (probe_id != 0 && probe_id != e.active_probe_id) return;
    e.last_used = ++tick_;
    e.probe_in_flight = false;
    e.active_probe_id = 0;
    e.probe_started_ms = 0;
}

void FinishSpecRequest(SpecController& controller, const SpecDecision& decision, std::string_view route,
                       SpecTaskClass task, std::int32_t draft_tokens, std::int32_t draft_accepted_tokens,
                       bool failed_or_cancelled) {
    if (decision.mode == SpecStrategy::kNone) return;
    if (draft_tokens > 0) {
        controller.Observe(route, task, decision.mode, static_cast<std::uint32_t>(draft_tokens),
                           static_cast<std::uint32_t>(std::max<std::int32_t>(0, draft_accepted_tokens)));
    } else if (decision.is_probe) {
        if (failed_or_cancelled) {
            controller.CancelProbe(route, task, decision.mode, decision.probe_id);
        } else {
            controller.ReleaseProbe(route, task, decision.mode, decision.probe_id);
        }
    }
}

double ComputeTopTokenProbability(std::span<const float> logits, Token token) noexcept {
    if (logits.empty() || token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
        return 0.0;
    }
    const double target = static_cast<double>(logits[static_cast<std::size_t>(token)]);
    if (!std::isfinite(target)) return 0.0;

    double max_logit = -std::numeric_limits<double>::infinity();
    for (float v : logits) {
        const double dv = static_cast<double>(v);
        if (std::isfinite(dv) && dv > max_logit) {
            max_logit = dv;
        }
    }
    if (!std::isfinite(max_logit)) return 0.0;

    double sum_exp = 0.0;
    for (float v : logits) {
        const double dv = static_cast<double>(v);
        if (std::isfinite(dv)) {
            sum_exp += std::exp(dv - max_logit);
        }
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) return 0.0;
    const double p = std::exp(target - max_logit) / sum_exp;
    return std::isfinite(p) ? std::clamp(p, 0.0, 1.0) : 0.0;
}

bool DraftTokenPassesPMin(std::span<const float> logits, Token token, double p_min) noexcept {
    if (!std::isfinite(p_min) || p_min <= 0.0) return true;
    return ComputeTopTokenProbability(logits, token) >= p_min;
}

SpecRouteStats SpecController::Stats(std::string_view route, SpecTaskClass task, SpecStrategy mode) const {
    SpecRouteStats s;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(MakeKey(route, task, mode));
    if (it == entries_.end()) return s;
    const Entry& e = it->second;
    s.state = e.state;
    s.window_steps = e.steps.size();
    s.window_drafted = e.window_drafted;
    s.window_accepted = e.window_accepted;
    s.rolling_acceptance =
        e.window_drafted == 0 ? 0.0 : static_cast<double>(e.window_accepted) / static_cast<double>(e.window_drafted);
    s.total_drafted = e.total_drafted;
    s.total_accepted = e.total_accepted;
    s.disable_count = e.disable_count;
    s.probe_count = e.probe_count;
    s.failed_probes = e.failed_probes;
    s.ignored_observations = e.ignored;
    s.backoff_ms = e.backoff_ms;
    if (e.state == SpecAcceptanceState::kDisabled) {
        s.retry_in_ms = SaturatingSubNonNeg(e.retry_at_ms, clock_());
    }
    return s;
}

std::size_t SpecController::TrackedEntries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

// --- slots (-np) -----------------------------------------------------------

SpecSlotAdvice AdviseSpecSlots(std::uint32_t requested_slots, std::int32_t n_max, std::uint32_t n_batch,
                               std::uint32_t max_spec_slots) noexcept {
    SpecSlotAdvice out;
    out.n_parallel = std::max<std::uint32_t>(requested_slots, 1);
    if (n_max <= 0) {
        out.n_max = 0;
        return out;
    }
    if (max_spec_slots > 0 && out.n_parallel > max_spec_slots) {
        out.n_parallel = max_spec_slots;
        out.reduced_slots = true;
    }
    out.n_max = n_max;
    if (n_batch > 0) {
        // llama.cpp v1.0.0 common_speculative_get_output_limits:
        // n_parallel * (1 + n_draft) <= n_batch  =>  n_draft <= n_batch / n_parallel - 1
        const std::int64_t rows_per_slot = static_cast<std::int64_t>(n_batch / out.n_parallel);
        const std::int64_t allowed = std::max<std::int64_t>(0, rows_per_slot - 1);
        if (allowed < n_max) {
            out.n_max = static_cast<std::int32_t>(allowed);
            out.reduced_draft = true;
        }
    }
    return out;
}

}  // namespace sonder::backends::llamacpp
