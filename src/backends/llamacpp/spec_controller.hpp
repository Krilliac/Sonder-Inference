// Per-route speculative-decoding controller for the direct llama.cpp backend.
//
// Decides, for each request, which speculation strategy to use (none / ngram /
// ngram-mod / suffix / draft-model / draft-mtp where llama.cpp v1.0.0 supports
// it), along with `n_max` and `p_min`, from a layered config keyed by route
// (model / endpoint name) and task class (code, chat, tool-call, reasoning).
// Tracks rolling draft acceptance per (route, task class, strategy), auto-
// disables routes whose acceptance falls below a threshold, and re-probes them
// later with exponential backoff.
//
// Pure stdlib-only C++20 (no llama.cpp headers, no I/O, time injected) so every
// rule can be unit-tested without model weights. Reuses:
//   - SlotSpecMode, IsHistoryBasedSpecMode, CheckArchitectureSpecGuard,
//     SlotSpecCapability and ConcurrentSpecPolicy from speculation_transition.hpp;
//   - SuffixMatcher (hybrid MTP+suffix burst cap and local pause state) from
//     suffix_matcher.hpp;
//   - SpeculativeType and default draft-length constants from llamacpp_backend.h.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/backends/llamacpp.hpp"
#include "speculation_transition.hpp"
#include "suffix_matcher.hpp"

namespace sonder::backends::llamacpp {

// ---------------------------------------------------------------------------
// Speculation strategies
// ---------------------------------------------------------------------------

// Concrete per-route speculation strategy. Extends SlotSpecMode with
// standalone draft-model drafting (`COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE` in
// llama.cpp v1.0.0, common/common.h:178) so configs can select none / ngram /
// ngram-mod / suffix / draft-model / draft-mtp uniformly.
enum class SpecStrategy : std::uint8_t {
    kNone = 0,
    kNgram,
    kNgramMod,
    kSuffix,
    kDraftModel,
    kDraftMtp,
};

[[nodiscard]] constexpr std::string_view ToString(SpecStrategy s) noexcept {
    switch (s) {
        case SpecStrategy::kNone: return "none";
        case SpecStrategy::kNgram: return "ngram";
        case SpecStrategy::kNgramMod: return "ngram-mod";
        case SpecStrategy::kSuffix: return "suffix";
        case SpecStrategy::kDraftModel: return "draft-model";
        case SpecStrategy::kDraftMtp: return "draft-mtp";
    }
    return "unknown";
}

// Concrete speculation types only: none, ngram, ngram-mod, suffix, draft-model
// (aliases: draft-simple, draft), draft-mtp (alias: mtp). Case-insensitive;
// '_' and '-' interchangeable. "auto" is rejected because it depends on the
// loaded model.
[[nodiscard]] bool ParseSpecStrategy(std::string_view text, SpecStrategy& out) noexcept;

// Maps strategies implemented by the per-slot transition state machine to
// SlotSpecMode. kDraftModel has no SlotSpecMode yet and maps to kNone.
[[nodiscard]] constexpr SlotSpecMode ToSlotSpecMode(SpecStrategy s) noexcept {
    switch (s) {
        case SpecStrategy::kNone: return SlotSpecMode::kNone;
        case SpecStrategy::kNgram: return SlotSpecMode::kNgramShort;
        case SpecStrategy::kNgramMod: return SlotSpecMode::kNgramMod;
        case SpecStrategy::kSuffix: return SlotSpecMode::kSuffix;
        case SpecStrategy::kDraftMtp: return SlotSpecMode::kDraftMtp;
        case SpecStrategy::kDraftModel: return SlotSpecMode::kNone;
    }
    return SlotSpecMode::kNone;
}

// Maps strategies supported by LlamaCppBackend::Generate() to SpeculativeType.
// kDraftModel returns std::nullopt because in-process standalone draft-model
// loading is not yet wired into LlamaCppBackend.
[[nodiscard]] constexpr std::optional<SpeculativeType> ToSpeculativeType(SpecStrategy s) noexcept {
    switch (s) {
        case SpecStrategy::kNone: return SpeculativeType::kNone;
        case SpecStrategy::kNgram: return SpeculativeType::kNgram;
        case SpecStrategy::kNgramMod: return SpeculativeType::kNgramMod;
        case SpecStrategy::kSuffix: return SpeculativeType::kSuffix;
        case SpecStrategy::kDraftMtp: return SpeculativeType::kDraftMtp;
        case SpecStrategy::kDraftModel: return std::nullopt;
    }
    return std::nullopt;
}

// True when drafting runs a model forward pass (standalone draft model or MTP
// heads). History-based modes (ngram, ngram-mod, suffix) only inspect tokens.
[[nodiscard]] constexpr bool IsModelBasedSpecStrategy(SpecStrategy s) noexcept {
    return s == SpecStrategy::kDraftModel || s == SpecStrategy::kDraftMtp;
}

[[nodiscard]] constexpr bool IsHistoryBasedSpecStrategy(SpecStrategy s) noexcept {
    return IsHistoryBasedSpecMode(ToSlotSpecMode(s));
}

// Default draft length used when a policy sets n_max = 0 ("strategy default").
// Reuses llamacpp_backend.h defaults; kDraftModel matches llama.cpp v1.0.0
// common_params_speculative_draft::n_max (common/common.h:343).
inline constexpr std::int32_t kDefaultDraftModelDraftMax = 3;

[[nodiscard]] constexpr std::int32_t DefaultSpecDraftMax(SpecStrategy s) noexcept {
    switch (s) {
        case SpecStrategy::kNone: return 0;
        case SpecStrategy::kNgram: return kDefaultNgramDraftMax;
        case SpecStrategy::kNgramMod: return kDefaultNgramModDraftMax;
        case SpecStrategy::kSuffix: return kDefaultSuffixDraftMax;
        case SpecStrategy::kDraftModel: return kDefaultDraftModelDraftMax;
        case SpecStrategy::kDraftMtp: return kDefaultMtpDraftMax;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Task classes (request classification)
// ---------------------------------------------------------------------------

// Content class of a request. Mirrors SI's request classifier in
// src/backends/llamaserver/adaptive_speculation.cpp (tool_json -> kToolCall,
// reasoning -> kReasoning, code_edit -> kCode, prose/general -> kChat) without
// depending on the retiring llamaserver module. Distinct from the scheduler's
// WorkloadClass (priority rank).
enum class SpecTaskClass : std::uint8_t {
    kChat = 0,
    kCode,
    kToolCall,
    kReasoning,
};

inline constexpr std::size_t kSpecTaskClassCount = 4;

[[nodiscard]] constexpr std::string_view ToString(SpecTaskClass c) noexcept {
    switch (c) {
        case SpecTaskClass::kChat: return "chat";
        case SpecTaskClass::kCode: return "code";
        case SpecTaskClass::kToolCall: return "tool-call";
        case SpecTaskClass::kReasoning: return "reasoning";
    }
    return "unknown";
}

// Case-insensitive; '_' and '-' are interchangeable. Also accepts the
// llamaserver names code_edit, tool_json, prose and general.
[[nodiscard]] bool ParseSpecTaskClass(std::string_view text, SpecTaskClass& out) noexcept;

// Request features known at admission.
struct SpecTaskFeatures {
    bool has_grammar = false;        // explicit grammar / GBNF
    bool tools_requested = false;    // tool schemas supplied and not tool_choice=none
    bool structured_output = false;  // response_format json_object / json_schema
    bool thinking_enabled = false;   // separate_reasoning / enable_thinking / reasoning_effort
    bool has_code_markers = false;   // fenced code ("```") or diff markers ("diff --git", "@@ ")
};

// Precedence: tool-call > reasoning > code > chat (matches SI's adaptive_speculation).
[[nodiscard]] SpecTaskClass ClassifySpecTask(const SpecTaskFeatures& f) noexcept;

// Matches SI's code/diff marker check from adaptive_speculation.cpp.
[[nodiscard]] bool ContainsCodeOrDiffMarkers(std::string_view text) noexcept;

// Classifies core GenerateRequest / ChatRequest objects directly at backend admission.
[[nodiscard]] SpecTaskClass ClassifyGenerateRequest(const sonder::inference::GenerateRequest& request,
                                                    std::string_view grammar = {}) noexcept;
[[nodiscard]] SpecTaskClass ClassifyChatRequest(const sonder::inference::ChatRequest& request,
                                                std::string_view grammar = {}) noexcept;

// ---------------------------------------------------------------------------
// Policy and config
// ---------------------------------------------------------------------------

inline constexpr std::int32_t kMaxSpecDraftMax = 128;
inline constexpr std::size_t kMaxSpecRouteKeyLength = 128;

// A fully resolved policy.
struct SpecPolicy {
    SpecStrategy mode = SpecStrategy::kNone;
    std::int32_t n_max = 0;  // 0 = DefaultSpecDraftMax(mode) after Resolve()
    // Minimum draft-token probability threshold (llama.cpp v1.0.0
    // common_params_speculative_draft::p_min, common/common.h:347, enforced in
    // common/speculative.cpp:421, 871 and in LlamaCppBackend::Impl::MtpDraft).
    // Valid only for model-based strategies (draft-model, draft-mtp, or auto_base
    // when resolving to MTP); p_min > 0 is rejected for history-based strategies.
    double p_min = 0.0;
};

// A partial policy: only the fields that are set override the layer below.
struct SpecPolicyOverlay {
    std::optional<SpecStrategy> mode;
    std::optional<std::int32_t> n_max;
    std::optional<double> p_min;
};

// "suffix,n_max=16,p_min=0.5", "n_max=4", "mode=draft-model,n_max=5,p_min=0.7".
// A bare first token is the mode. Each field at most once. On failure returns
// false and sets `error`.
[[nodiscard]] bool ParseSpecPolicyOverlay(std::string_view text, SpecPolicyOverlay& out, std::string& error);

struct SpecRouteRules {
    std::optional<SpecPolicyOverlay> any;                                        // every task class on this route
    std::array<std::optional<SpecPolicyOverlay>, kSpecTaskClassCount> by_class;  // one task class on this route
};

struct SpecAcceptanceConfig {
    // Rolling window length, in verification steps (or completed requests).
    std::size_t window_steps = 32;
    // Do not judge a route until the window holds at least this many drafted tokens.
    std::uint32_t min_drafted_tokens = 64;
    // Disable speculation when the rolling acceptance falls strictly below this.
    double disable_below = 0.25;
    // A re-probe must reach this acceptance to switch speculation back on (hysteresis: >= disable_below).
    double reenable_at_least = 0.35;
    // Drafted tokens a re-probe collects before it is judged.
    std::uint32_t probe_drafted_tokens = 64;
    // First re-probe delay after a disable; each failed probe multiplies it by backoff_factor up to the max.
    std::int64_t reprobe_initial_ms = 30'000;
    std::int64_t reprobe_max_ms = 15 * 60 * 1000;
    double backoff_factor = 2.0;
    // Safety-net timeout on an in-flight probe: if a probe owner never reports
    // completion within this window, Decide() reclaims ownership so speculation
    // cannot stay permanently locked out.
    std::int64_t probe_timeout_ms = 60'000;
};

struct SpecControllerConfig {
    // Lowest layer: used when no rule matches (typically LlamaCppBackendOptions::spec_type / spec_draft_max).
    SpecPolicy base;
    // True when LlamaCppBackendOptions::spec_type == "auto": if no rule sets a
    // mode, Decide() resolves to kDraftMtp when caps.mtp_available, else kNgramMod.
    bool auto_base = false;
    // Layered above base, per task class, for every route.
    std::array<std::optional<SpecPolicyOverlay>, kSpecTaskClassCount> class_rules;
    // Layered above class_rules, per exact route name.
    std::unordered_map<std::string, SpecRouteRules> routes;
    // Determinism mode (LlamaCppBackendOptions::deterministic): model-based
    // drafting (draft-model, draft-mtp) is off for every request.
    bool deterministic = false;
    SpecAcceptanceConfig acceptance;
    // Upper bound on the (route, class, strategy) acceptance table; LRU entries are evicted.
    std::size_t max_tracked_entries = 256;
    // Small-batch slot cap used by AdviseSpecSlots when speculation is on (0 = do not cap).
    // Defaults to ConcurrentSpecPolicy::max_concurrent_slots (4) from speculation_transition.hpp.
    std::uint32_t max_spec_slots = ConcurrentSpecPolicy{}.max_concurrent_slots;

    // Empty string when valid, otherwise the first problem found.
    [[nodiscard]] std::string Validate() const;

    // Layers base < class rule < route-wide rule < route+class rule, field by field,
    // then fills n_max (0 -> strategy default; mode none -> 0).
    [[nodiscard]] SpecPolicy Resolve(std::string_view route, SpecTaskClass task) const;

    // Same as Resolve(), plus tells the caller whether `mode` was explicitly set by a
    // rule (so auto_base only applies when no rule selected a concrete mode).
    [[nodiscard]] SpecPolicy Resolve(std::string_view route, SpecTaskClass task, bool& mode_from_rule) const;

    // Merges `overlay` into the slot chosen by (route, task). Empty route and no task = base.
    void SetRule(std::string_view route, std::optional<SpecTaskClass> task, const SpecPolicyOverlay& overlay);

    // True when !deterministic and the base policy, auto_base, or any class/route rule can select kDraftMtp.
    [[nodiscard]] bool MayUseMtp() const noexcept;

    // Largest draft length (n_max) that any configured base, class, or route rule can request (0 if all none).
    [[nodiscard]] std::int32_t MaxPlannedDraft() const noexcept;

    // Largest draft length (n_max) requested specifically by class/route rules targeting kDraftMtp
    // (0 when deterministic or when no class/route rule selects kDraftMtp).
    [[nodiscard]] std::int32_t MaxMtpRouteDraft() const noexcept;
};

// One config rule as `<selector>=<policy>`:
//   selector = [<class>][@<route>]   ("*" or an empty side means "any")
//   "*=none"                         base policy
//   "code=suffix,n_max=16"           a task class on every route
//   "@org/model=ngram-mod"           every class on one route
//   "tool-call@org/model=draft-model,n_max=6,p_min=0.7"
// The selector ends at the first '='; route names must not contain '=' or '@'.
[[nodiscard]] bool ParseSpecRule(std::string_view text, SpecControllerConfig& config, std::string& error);

// Builds and validates a SpecControllerConfig from LlamaCppBackendOptions
// (spec_type, spec_draft_max, spec_p_min, spec_route_rules, deterministic,
// spec_batch_max_slots). Returns false and sets `error` on invalid config.
[[nodiscard]] bool BuildSpecControllerConfig(const sonder::inference::LlamaCppBackendOptions& options,
                                             SpecControllerConfig& out, std::string& error);

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

// Monotonic milliseconds. Injected so tests control time.
using SpecClock = std::function<std::int64_t()>;

// What the host knows about the loaded model and build.
struct SpecModelCaps {
    bool mtp_available = false;          // build has SONDER_LLAMACPP_HAS_MTP and model has MTP layers
    bool draft_model_available = false;  // standalone draft model loaded and vocab-compatible
    sonder::inference::ModelArchitecture architecture = sonder::inference::ModelArchitecture::attention_only;
    std::int32_t hybrid_rs_cap = 0;      // llama_n_rs_seq(ctx) on hybrid models; see CheckArchitectureSpecGuard
    bool can_hybrid_spec_ckpt = false;   // partial recurrent checkpoints available for hybrid rollback
    // Loaded context's per-sequence output row limit (cp.n_outputs_max_per_seq; 0 = unknown/unbounded).
    // Decide() clamps n_max to ctx_outputs_max_per_seq - 1 when > 0.
    std::int32_t ctx_outputs_max_per_seq = 0;
};

struct SpecRequestContext {
    std::string route;
    SpecTaskClass task = SpecTaskClass::kChat;
    // Per-request determinism flag (identical inputs + seed => identical output;
    // Capability::deterministic in include/sonder/inference/backend.hpp:45).
    // OR-ed with config.deterministic.
    bool deterministic = false;
    SpecModelCaps caps;
};

enum class SpecDecisionReason : std::uint8_t {
    kConfigured = 0,
    kConfiguredNone,
    kDeterminismModelDraftOff,
    kMtpUnavailable,
    kDraftModelUnavailable,
    kArchitectureUnsupported,
    kBatchOutputLimit,
    kAcceptanceDisabled,
    kProbe,
};

[[nodiscard]] constexpr std::string_view ToString(SpecDecisionReason r) noexcept {
    switch (r) {
        case SpecDecisionReason::kConfigured: return "configured";
        case SpecDecisionReason::kConfiguredNone: return "configured_none";
        case SpecDecisionReason::kDeterminismModelDraftOff: return "determinism_model_draft_off";
        case SpecDecisionReason::kMtpUnavailable: return "mtp_unavailable";
        case SpecDecisionReason::kDraftModelUnavailable: return "draft_model_unavailable";
        case SpecDecisionReason::kArchitectureUnsupported: return "architecture_unsupported";
        case SpecDecisionReason::kBatchOutputLimit: return "batch_output_limit";
        case SpecDecisionReason::kAcceptanceDisabled: return "acceptance_disabled";
        case SpecDecisionReason::kProbe: return "probe";
    }
    return "unknown";
}

struct SpecDecision {
    SpecStrategy mode = SpecStrategy::kNone;             // what to run (kNone = no speculation)
    std::int32_t n_max = 0;
    double p_min = 0.0;
    SpecDecisionReason reason = SpecDecisionReason::kConfiguredNone;
    bool is_probe = false;                               // a re-probe after an acceptance disable
    std::uint64_t probe_id = 0;                          // non-zero monotonic owner token when is_probe == true
    SpecStrategy configured_mode = SpecStrategy::kNone;  // what config asked for before any override
};

// Bridges SpecDecision into SlotSpecCapability for SpeculationTransitionState::Configure(),
// reusing CheckArchitectureSpecGuard metadata and SuffixMatcher's hybrid burst cap & pause state.
[[nodiscard]] SlotSpecCapability BuildSlotSpecCapability(const SpecDecision& decision, const SpecModelCaps& caps,
                                                         bool suffix_available,
                                                         const SuffixMatcher* suffix_matcher = nullptr) noexcept;

enum class SpecAcceptanceState : std::uint8_t { kEnabled = 0, kDisabled, kProbing };

[[nodiscard]] constexpr std::string_view ToString(SpecAcceptanceState s) noexcept {
    switch (s) {
        case SpecAcceptanceState::kEnabled: return "enabled";
        case SpecAcceptanceState::kDisabled: return "disabled";
        case SpecAcceptanceState::kProbing: return "probing";
    }
    return "unknown";
}

struct SpecRouteStats {
    SpecAcceptanceState state = SpecAcceptanceState::kEnabled;
    std::size_t window_steps = 0;
    std::uint64_t window_drafted = 0;
    std::uint64_t window_accepted = 0;
    double rolling_acceptance = 0.0;  // window_accepted / window_drafted, 0 when empty
    std::uint64_t total_drafted = 0;  // never windowed
    std::uint64_t total_accepted = 0;
    std::uint32_t disable_count = 0;  // enabled -> disabled transitions
    std::uint32_t probe_count = 0;    // probes started
    std::uint32_t failed_probes = 0;
    std::uint64_t ignored_observations = 0;  // observations that arrived while disabled
    std::int64_t backoff_ms = 0;      // current re-probe delay (0 when enabled)
    std::int64_t retry_in_ms = 0;     // time until the next probe (disabled only)
};

class SpecController {
public:
    // Callers should run config.Validate() first; the constructor defensively clamps
    // values that would otherwise break invariants.
    explicit SpecController(SpecControllerConfig config, SpecClock clock = {});

    SpecController(const SpecController&) = delete;
    SpecController& operator=(const SpecController&) = delete;

    // Picks the speculation to run for one request. Creates no table entries; a disabled
    // entry whose retry time has passed transitions to probing here.
    [[nodiscard]] SpecDecision Decide(const SpecRequestContext& ctx);

    // Feeds one verification step (or one request's totals) back. `mode` is the strategy
    // that actually ran. drafted == 0 and mode none carry no information and are ignored.
    // accepted is clamped to drafted.
    void Observe(std::string_view route, SpecTaskClass task, SpecStrategy mode, std::uint32_t drafted,
                 std::uint32_t accepted);

    // Aborts an in-flight probe (e.g. request cancelled or errored before drafting) and
    // returns the entry to disabled with its current backoff without counting a failed probe.
    // When `probe_id` is non-zero, only cancels if it matches the currently active probe.
    void CancelProbe(std::string_view route, SpecTaskClass task, SpecStrategy mode, std::uint64_t probe_id = 0);

    // Releases in-flight probe ownership when a probe request completes with zero drafted
    // tokens (e.g. immediate EOS, max_tokens=1, stop sequence, media, or state_restored)
    // while keeping the entry in kProbing so the next request can complete the probe.
    void ReleaseProbe(std::string_view route, SpecTaskClass task, SpecStrategy mode, std::uint64_t probe_id = 0);

    // Read-only snapshot; zeroed (state kEnabled) when the entry does not exist.
    [[nodiscard]] SpecRouteStats Stats(std::string_view route, SpecTaskClass task, SpecStrategy mode) const;

    [[nodiscard]] std::size_t TrackedEntries() const;
    [[nodiscard]] const SpecControllerConfig& config() const noexcept { return config_; }

private:
    struct Entry {
        SpecAcceptanceState state = SpecAcceptanceState::kEnabled;
        std::deque<std::pair<std::uint32_t, std::uint32_t>> steps;
        std::uint64_t window_drafted = 0;
        std::uint64_t window_accepted = 0;
        std::uint64_t probe_drafted = 0;
        std::uint64_t probe_accepted = 0;
        bool probe_in_flight = false;
        std::uint64_t active_probe_id = 0;
        std::int64_t probe_started_ms = 0;
        std::uint64_t total_drafted = 0;
        std::uint64_t total_accepted = 0;
        std::uint32_t disable_count = 0;
        std::uint32_t probe_count = 0;
        std::uint32_t failed_probes = 0;
        std::uint64_t ignored = 0;
        std::int64_t backoff_ms = 0;
        std::int64_t retry_at_ms = 0;
        std::uint64_t last_used = 0;
    };

    [[nodiscard]] static std::string MakeKey(std::string_view route, SpecTaskClass task, SpecStrategy mode);
    [[nodiscard]] std::int64_t Now() const;
    Entry& FindOrCreateLocked(const std::string& key);
    void EnterDisabledLocked(Entry& e, std::int64_t backoff_ms);

    SpecControllerConfig config_;
    SpecClock clock_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    std::uint64_t tick_ = 0;
    std::uint64_t next_probe_id_ = 0;
};

// Completes one request against SpecController: observes draft counts when
// draft_tokens > 0, cancels a failed/cancelled probe, or releases probe ownership
// on zero-draft normal completion so probe_in_flight never leaks.
void FinishSpecRequest(SpecController& controller, const SpecDecision& decision, std::string_view route,
                       SpecTaskClass task, std::int32_t draft_tokens, std::int32_t draft_accepted_tokens,
                       bool failed_or_cancelled);

// Computes the numerically-stable softmax probability of `token` in `logits` (0.0 on invalid input).
[[nodiscard]] double ComputeTopTokenProbability(std::span<const float> logits, Token token) noexcept;

// Returns true when `p_min <= 0` or the softmax probability of `token` in `logits` is >= `p_min`.
// Used by the MTP draft loop to stop drafting early when draft confidence falls below `p_min`
// (matching llama.cpp v1.0.0 common/speculative.cpp:421, 871).
[[nodiscard]] bool DraftTokenPassesPMin(std::span<const float> logits, Token token, double p_min) noexcept;

// ---------------------------------------------------------------------------
// Slots (-np) and draft length
// ---------------------------------------------------------------------------

struct SpecSlotAdvice {
    std::uint32_t n_parallel = 1;
    std::int32_t n_max = 0;
    bool reduced_slots = false;
    bool reduced_draft = false;
};

// Opt-in SI concurrent small-batch slot (-np) and per-slot draft-length sizing helper:
//  * llama.cpp v1.0.0 (common/speculative.cpp:2903-2911, common_speculative_get_output_limits;
//    tools/server/server-context.cpp:62-63, 1486-1496) computes
//    `per_seq = min(n_batch, 1 + n_draft)` and `total = min(n_batch, n_parallel * per_seq)`.
//    v1.0.0 clamps the aggregate total and does NOT require `n_draft <= n_batch / n_parallel - 1`
//    for single-slot decode.
//  * SI's opt-in concurrent small-batch policy (speculation_transition.hpp:570-580,
//    ConcurrentSpecPolicy::max_concurrent_slots = 4) keeps `-np` small (<= 4 by default) when
//    all slots speculate simultaneously in one verify batch, and bounds per-slot drafts to
//    `n_batch / n_parallel - 1` when sharing a single `n_batch` across all `n_parallel` slots.
//    `max_spec_slots == 0` disables the slot cap.
// With n_max <= 0 only the "at least one slot" floor applies.
// `n_batch == 0` means "unknown, do not clamp n_max".
[[nodiscard]] SpecSlotAdvice AdviseSpecSlots(
    std::uint32_t requested_slots, std::int32_t n_max, std::uint32_t n_batch,
    std::uint32_t max_spec_slots = ConcurrentSpecPolicy{}.max_concurrent_slots) noexcept;

}  // namespace sonder::backends::llamacpp
