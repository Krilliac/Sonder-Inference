#include "adaptive_speculation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace sonder::inference::llamaserver {
namespace {

constexpr std::array<SpeculationWorkload, kSpeculationWorkloadCount> kAllWorkloads = {
    SpeculationWorkload::general,
    SpeculationWorkload::code_edit,
    SpeculationWorkload::tool_json,
    SpeculationWorkload::prose,
    SpeculationWorkload::reasoning,
};

constexpr double kMaxCostRatio = 10.0;
constexpr std::uint32_t kMaxCostStepCooldown = 1024u;
constexpr std::uint32_t kMaxBaselineProbeInterval = 100000u;
constexpr std::size_t kClassifierScanLimit = 4096u;

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::uint32_t saturating_inc_u32(std::uint32_t a) noexcept {
    return a == std::numeric_limits<std::uint32_t>::max() ? a : a + 1u;
}

std::uint64_t fnv1a_64(std::string_view text) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char ch : text) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(ch));
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex_u64(std::uint64_t value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(16u, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kHex[value & 0xFU];
        value >>= 4;
    }
    return out;
}

bool valid_probability(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool valid_positive_finite(double value, double lo, double hi) noexcept {
    return std::isfinite(value) && value >= lo && value <= hi;
}

double clean_nonneg(double value) noexcept {
    return (std::isfinite(value) && value > 0.0) ? value : 0.0;
}

bool is_ascii_word_char(unsigned char ch) noexcept {
    return std::isalnum(ch) != 0 || ch == '_';
}

char to_ascii_lower(char ch) noexcept {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
}

bool contains_code_or_diff_markers(std::string_view text) noexcept {
    return text.find("```") != std::string_view::npos || text.find("diff --git") != std::string_view::npos ||
           text.find("@@ ") != std::string_view::npos;
}

bool contains_strong_code_syntax(std::string_view text) noexcept {
    if (text.find('{') != std::string_view::npos || text.find('}') != std::string_view::npos ||
        text.find("();") != std::string_view::npos || text.find("::") != std::string_view::npos ||
        text.find("->") != std::string_view::npos || text.find("=>") != std::string_view::npos ||
        text.find("#include") != std::string_view::npos || text.find("std::") != std::string_view::npos) {
        return true;
    }
    static constexpr std::array<std::string_view, 7> kCodeTokens = {
        "def ", "fn ", "struct ", "class ", "function ", "return ", "SELECT ",
    };
    for (const std::string_view tok : kCodeTokens) {
        if (text.find(tok) != std::string_view::npos)
            return true;
    }
    return false;
}

bool contains_word_ci(std::string_view haystack_lower, std::string_view needle_lower) noexcept {
    if (needle_lower.empty() || haystack_lower.size() < needle_lower.size())
        return false;
    std::size_t pos = 0;
    while ((pos = haystack_lower.find(needle_lower, pos)) != std::string_view::npos) {
        const bool left_ok = (pos == 0) || !is_ascii_word_char(static_cast<unsigned char>(haystack_lower[pos - 1]));
        const std::size_t end = pos + needle_lower.size();
        const bool right_ok =
            (end == haystack_lower.size()) || !is_ascii_word_char(static_cast<unsigned char>(haystack_lower[end]));
        if (left_ok && right_ok)
            return true;
        pos += 1u;
    }
    return false;
}

bool looks_like_prose_text(std::string_view raw) noexcept {
    if (raw.empty())
        return false;
    const std::string_view sample = raw.substr(0, std::min<std::size_t>(raw.size(), kClassifierScanLimit));
    if (contains_strong_code_syntax(sample))
        return false;

    std::string lower;
    lower.reserve(sample.size());
    std::size_t alpha_or_space = 0;
    std::size_t sentence_terminators = 0;
    for (const char ch : sample) {
        const auto uch = static_cast<unsigned char>(ch);
        lower.push_back(to_ascii_lower(ch));
        if (std::isalpha(uch) != 0 || std::isspace(uch) != 0)
            ++alpha_or_space;
        if (ch == '.' || ch == '!' || ch == '?')
            ++sentence_terminators;
    }

    static constexpr std::array<std::string_view, 18> kProseCues = {
        "essay",         "short story",      "novel",         "poem",          "poetry",        "prose",
        "blog post",     "article",          "narrative",     "press release", "cover letter",  "proofread",
        "copyedit",      "creative writing", "fiction",       "biography",     "editorial",     "speech",
    };
    for (const std::string_view cue : kProseCues) {
        if (contains_word_ci(lower, cue))
            return true;
    }

    // Structural natural-language prose heuristic: multi-sentence passage (>= 240
    // bytes, >= 3 sentences, >= 70% letters/whitespace, no code syntax).
    if (sample.size() >= 240u && sentence_terminators >= 3u &&
        alpha_or_space * 10u >= sample.size() * 7u) {
        return true;
    }
    return false;
}

bool chat_thinking_enabled(const ChatRequest &request) noexcept {
    if (request.separate_reasoning)
        return true;
    if (request.thinking.enable_thinking.has_value())
        return *request.thinking.enable_thinking;
    if (request.thinking.reasoning_effort.has_value() && !request.thinking.reasoning_effort->empty() &&
        *request.thinking.reasoning_effort != "none") {
        return true;
    }
    return false;
}

AdaptiveSpeculationConfig sanitize_config(AdaptiveSpeculationConfig config) {
    if (validate_adaptive_speculation_config(config).ok())
        return config;

    const AdaptiveSpeculationConfig defaults{};
    config.max_draft_n = std::clamp<std::uint32_t>(config.max_draft_n, 1, AdaptiveSpeculationConfig::kMaxDraftN);
    config.min_draft_n = std::min(config.min_draft_n, config.max_draft_n);
    config.initial_draft_n = std::clamp(config.initial_draft_n, config.min_draft_n, config.max_draft_n);
    config.probe_draft_n = std::clamp<std::uint32_t>(config.probe_draft_n, 1, config.max_draft_n);
    config.window_size = std::clamp<std::size_t>(config.window_size, 1, AdaptiveSpeculationConfig::kMaxWindowSize);
    config.min_samples = std::clamp<std::size_t>(config.min_samples, 1, config.window_size);

    if (!valid_probability(config.pause_acceptance_threshold))
        config.pause_acceptance_threshold = defaults.pause_acceptance_threshold;
    if (!valid_probability(config.step_down_acceptance_threshold))
        config.step_down_acceptance_threshold = defaults.step_down_acceptance_threshold;
    if (!valid_probability(config.step_up_acceptance_threshold))
        config.step_up_acceptance_threshold = defaults.step_up_acceptance_threshold;

    if (config.step_down_acceptance_threshold < config.pause_acceptance_threshold)
        config.step_down_acceptance_threshold = config.pause_acceptance_threshold;
    if (config.step_up_acceptance_threshold < config.step_down_acceptance_threshold)
        config.step_up_acceptance_threshold = config.step_down_acceptance_threshold;

    config.max_entries = std::clamp<std::size_t>(config.max_entries, 1, AdaptiveSpeculationConfig::kMaxEntries);
    config.min_baseline_samples =
        std::clamp<std::size_t>(config.min_baseline_samples, 1, AdaptiveSpeculationConfig::kMaxWindowSize);
    if (!valid_positive_finite(config.cost_step_down_ratio, 1.0, kMaxCostRatio))
        config.cost_step_down_ratio = defaults.cost_step_down_ratio;
    if (!valid_positive_finite(config.cost_pause_ratio, 1.0, kMaxCostRatio))
        config.cost_pause_ratio = defaults.cost_pause_ratio;
    if (config.cost_pause_ratio < config.cost_step_down_ratio)
        config.cost_pause_ratio = config.cost_step_down_ratio;
    if (!valid_positive_finite(config.cost_probe_recover_ratio, 0.01, kMaxCostRatio))
        config.cost_probe_recover_ratio = defaults.cost_probe_recover_ratio;
    if (config.cost_probe_recover_ratio > config.cost_step_down_ratio)
        config.cost_probe_recover_ratio = config.cost_step_down_ratio;
    config.cost_step_cooldown = std::min(config.cost_step_cooldown, kMaxCostStepCooldown);
    config.baseline_probe_interval = std::min(config.baseline_probe_interval, kMaxBaselineProbeInterval);
    return config;
}

struct ComputedCost {
    double decode_elapsed_ms = 0.0;
    double ms_per_token = 0.0;
    double decode_tps = 0.0;
    double prompt_or_restore_ms = 0.0;
};

ComputedCost compute_observation_cost(const SpeculationObservation &obs) noexcept {
    ComputedCost out;
    out.prompt_or_restore_ms = clean_nonneg(obs.prompt_or_restore_ms);

    double elapsed_ms = clean_nonneg(obs.decode_elapsed_ms);
    if (elapsed_ms <= 0.0) {
        const double d = clean_nonneg(obs.draft_ms);
        const double v = clean_nonneg(obs.verify_ms);
        const double r = clean_nonneg(obs.rollback_ms);
        const double c = clean_nonneg(obs.coordination_ms);
        const double phase_ms = (obs.draft_verify_overlapped ? std::max(d, v) : (d + v)) + r + c;
        if (std::isfinite(phase_ms) && phase_ms > 0.0)
            elapsed_ms = phase_ms;
    }
    out.decode_elapsed_ms = elapsed_ms;

    if (elapsed_ms > 0.0 && obs.predicted_n > 0u) {
        const double ms_per_tok = elapsed_ms / static_cast<double>(obs.predicted_n);
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0) {
            out.ms_per_token = ms_per_tok;
            out.decode_tps = 1000.0 / ms_per_tok;
        }
    } else if (const double tps = clean_nonneg(obs.decode_tps); tps > 0.0) {
        out.decode_tps = tps;
        const double ms_per_tok = 1000.0 / tps;
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0)
            out.ms_per_token = ms_per_tok;
    }
    return out;
}

}  // namespace

const char *to_string(SpeculationWorkload workload) noexcept {
    switch (workload) {
    case SpeculationWorkload::general:
        return "general";
    case SpeculationWorkload::code_edit:
        return "code_edit";
    case SpeculationWorkload::tool_json:
        return "tool_json";
    case SpeculationWorkload::prose:
        return "prose";
    case SpeculationWorkload::reasoning:
        return "reasoning";
    }
    return "unknown";
}

std::optional<SpeculationWorkload> parse_speculation_workload(std::string_view text) noexcept {
    if (text == "general")
        return SpeculationWorkload::general;
    if (text == "code_edit")
        return SpeculationWorkload::code_edit;
    if (text == "tool_json")
        return SpeculationWorkload::tool_json;
    if (text == "prose")
        return SpeculationWorkload::prose;
    if (text == "reasoning")
        return SpeculationWorkload::reasoning;
    return std::nullopt;
}

const char *to_string(ContextBand band) noexcept {
    switch (band) {
    case ContextBand::short_ctx:
        return "short";
    case ContextBand::medium_ctx:
        return "medium";
    case ContextBand::long_ctx:
        return "long";
    case ContextBand::very_long_ctx:
        return "very_long";
    }
    return "unknown";
}

ContextBand classify_context_band(std::uint64_t context_tokens) noexcept {
    if (context_tokens <= 4096u)
        return ContextBand::short_ctx;
    if (context_tokens <= 16384u)
        return ContextBand::medium_ctx;
    if (context_tokens <= 65536u)
        return ContextBand::long_ctx;
    return ContextBand::very_long_ctx;
}

const char *to_string(ConcurrencyBand band) noexcept {
    switch (band) {
    case ConcurrencyBand::solo:
        return "solo";
    case ConcurrencyBand::low:
        return "low";
    case ConcurrencyBand::medium:
        return "medium";
    case ConcurrencyBand::high:
        return "high";
    }
    return "unknown";
}

ConcurrencyBand classify_concurrency_band(std::uint32_t active_requests) noexcept {
    if (active_requests <= 1u)
        return ConcurrencyBand::solo;
    if (active_requests <= 3u)
        return ConcurrencyBand::low;
    if (active_requests <= 7u)
        return ConcurrencyBand::medium;
    return ConcurrencyBand::high;
}

std::size_t SpeculationContextKeyHash::operator()(const SpeculationContextKey &key) const noexcept {
    const auto mix = [](std::uint64_t seed, std::uint64_t val) noexcept -> std::uint64_t {
        return seed ^ (val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };
    std::uint64_t h = fnv1a_64(key.target_identity);
    h = mix(h, fnv1a_64(key.draft_identity));
    h = mix(h, static_cast<std::uint64_t>(key.context_band));
    h = mix(h, static_cast<std::uint64_t>(key.concurrency_band));
    h = mix(h, static_cast<std::uint64_t>(key.workload));
    return static_cast<std::size_t>(h ^ (h >> 32));
}

std::string normalize_speculation_identity(std::string_view identity) {
    if (identity.size() <= SpeculationContextKey::kMaxIdentityLength)
        return std::string(identity);
    constexpr std::size_t kSuffixLen = 17u;  // "#" + 16 hex digits
    constexpr std::size_t kPrefixLen = SpeculationContextKey::kMaxIdentityLength - kSuffixLen;
    std::string out;
    out.reserve(SpeculationContextKey::kMaxIdentityLength);
    out.append(identity.substr(0, kPrefixLen));
    out.push_back('#');
    out.append(hex_u64(fnv1a_64(identity)));
    return out;
}

SpeculationWorkload classify_generate_workload(const GenerateRequest &request, std::string_view grammar) noexcept {
    if (!grammar.empty() || !request.grammar.empty())
        return SpeculationWorkload::tool_json;
    if (contains_code_or_diff_markers(request.prompt))
        return SpeculationWorkload::code_edit;
    if (looks_like_prose_text(request.prompt))
        return SpeculationWorkload::prose;
    return SpeculationWorkload::general;
}

SpeculationWorkload classify_chat_workload(const ChatRequest &request, std::string_view grammar) noexcept {
    if (!grammar.empty() || !request.grammar.empty() || chat_tools_requested(request) ||
        (request.response_format_type && *request.response_format_type != "text")) {
        return SpeculationWorkload::tool_json;
    }
    if (chat_thinking_enabled(request))
        return SpeculationWorkload::reasoning;
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "user") {
            if (contains_code_or_diff_markers(it->content))
                return SpeculationWorkload::code_edit;
            if (looks_like_prose_text(it->content))
                return SpeculationWorkload::prose;
            break;
        }
    }
    return SpeculationWorkload::general;
}

std::uint64_t estimate_generate_context_tokens(const GenerateRequest &request) noexcept {
    const std::uint64_t bytes = static_cast<std::uint64_t>(request.prompt.size());
    return bytes == 0u ? 0u : ((bytes + 3u) / 4u);
}

std::uint64_t estimate_chat_context_tokens(const ChatRequest &request) noexcept {
    std::uint64_t total_bytes = 0u;
    for (const auto &msg : request.messages) {
        total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(msg.content.size()));
        if (msg.reasoning_content)
            total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(msg.reasoning_content->size()));
        for (const auto &tc : msg.tool_calls) {
            total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(tc.name.size()));
            total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(tc.arguments_json.size()));
        }
    }
    for (const auto &tool : request.tools) {
        total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(tool.name.size()));
        total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(tool.description.size()));
        total_bytes = saturating_add(total_bytes, static_cast<std::uint64_t>(tool.parameters_json.size()));
    }
    return total_bytes == 0u ? 0u : ((total_bytes + 3u) / 4u);
}

Status validate_adaptive_speculation_config(const AdaptiveSpeculationConfig &config) {
    const auto invalid = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: adaptive speculation: ") + what);
    };
    if (config.max_draft_n == 0 || config.max_draft_n > AdaptiveSpeculationConfig::kMaxDraftN)
        return invalid("max_draft_n must be in [1, 128]");
    if (config.min_draft_n > config.max_draft_n)
        return invalid("min_draft_n must not exceed max_draft_n");
    if (config.initial_draft_n < config.min_draft_n || config.initial_draft_n > config.max_draft_n)
        return invalid("initial_draft_n must be in [min_draft_n, max_draft_n]");
    if (config.probe_draft_n == 0 || config.probe_draft_n > config.max_draft_n)
        return invalid("probe_draft_n must be in [1, max_draft_n]");
    if (config.window_size == 0 || config.window_size > AdaptiveSpeculationConfig::kMaxWindowSize)
        return invalid("window_size must be in [1, 256]");
    if (config.min_samples == 0 || config.min_samples > config.window_size)
        return invalid("min_samples must be in [1, window_size]");
    if (!valid_probability(config.pause_acceptance_threshold))
        return invalid("pause_acceptance_threshold must be finite and in [0, 1]");
    if (!valid_probability(config.step_down_acceptance_threshold))
        return invalid("step_down_acceptance_threshold must be finite and in [0, 1]");
    if (!valid_probability(config.step_up_acceptance_threshold))
        return invalid("step_up_acceptance_threshold must be finite and in [0, 1]");
    if (config.pause_acceptance_threshold > config.step_down_acceptance_threshold)
        return invalid("pause_acceptance_threshold must not exceed step_down_acceptance_threshold");
    if (config.step_down_acceptance_threshold > config.step_up_acceptance_threshold)
        return invalid("step_down_acceptance_threshold must not exceed step_up_acceptance_threshold");
    if (config.max_entries == 0 || config.max_entries > AdaptiveSpeculationConfig::kMaxEntries)
        return invalid("max_entries must be in [1, 4096]");
    if (config.min_baseline_samples == 0 || config.min_baseline_samples > AdaptiveSpeculationConfig::kMaxWindowSize)
        return invalid("min_baseline_samples must be in [1, 256]");
    if (!valid_positive_finite(config.cost_step_down_ratio, 1.0, kMaxCostRatio))
        return invalid("cost_step_down_ratio must be finite and in [1.0, 10.0]");
    if (!valid_positive_finite(config.cost_pause_ratio, 1.0, kMaxCostRatio))
        return invalid("cost_pause_ratio must be finite and in [1.0, 10.0]");
    if (config.cost_pause_ratio < config.cost_step_down_ratio)
        return invalid("cost_pause_ratio must not be less than cost_step_down_ratio");
    if (!valid_positive_finite(config.cost_probe_recover_ratio, 0.01, kMaxCostRatio))
        return invalid("cost_probe_recover_ratio must be finite and in [0.01, 10.0]");
    if (config.cost_probe_recover_ratio > config.cost_step_down_ratio)
        return invalid("cost_probe_recover_ratio must not exceed cost_step_down_ratio");
    if (config.cost_step_cooldown > kMaxCostStepCooldown)
        return invalid("cost_step_cooldown must be at most 1024");
    if (config.baseline_probe_interval > kMaxBaselineProbeInterval)
        return invalid("baseline_probe_interval must be at most 100000");
    return Status::success();
}

SpeculationObservation observation_from_timings(const Timings &timings, SpeculationWorkload workload) {
    SpeculationObservation obs;
    obs.workload = workload;

    if (timings.draft_present || timings.draft_tokens != 0) {
        obs.draft_n = timings.draft_tokens;
    } else if (timings.backend_timings && timings.backend_timings->draft_n) {
        obs.draft_n = *timings.backend_timings->draft_n;
    }

    if (timings.draft_accepted_present || timings.draft_accepted_tokens != 0) {
        obs.draft_n_accepted = timings.draft_accepted_tokens;
    } else if (timings.backend_timings && timings.backend_timings->draft_n_accepted) {
        obs.draft_n_accepted = *timings.backend_timings->draft_n_accepted;
    }

    if (obs.draft_n_accepted > obs.draft_n)
        obs.draft_n_accepted = obs.draft_n;

    if (timings.completion_present || timings.completion_tokens != 0) {
        obs.predicted_n = timings.completion_tokens;
    } else if (timings.backend_timings && timings.backend_timings->predicted_n) {
        obs.predicted_n = *timings.backend_timings->predicted_n;
    }

    // Keep prompt/prefix restore duration strictly separate from decode elapsed time.
    if (timings.backend_timings && timings.backend_timings->prompt_ms &&
        std::isfinite(*timings.backend_timings->prompt_ms) && *timings.backend_timings->prompt_ms > 0.0) {
        obs.prompt_or_restore_ms = *timings.backend_timings->prompt_ms;
    } else if (timings.prompt_ns > 0u) {
        obs.prompt_or_restore_ms = static_cast<double>(timings.prompt_ns) / 1e6;
    }

    if (timings.backend_timings && timings.backend_timings->predicted_ms &&
        std::isfinite(*timings.backend_timings->predicted_ms) && *timings.backend_timings->predicted_ms > 0.0) {
        obs.decode_elapsed_ms = *timings.backend_timings->predicted_ms;
    } else if (timings.eval_ns > 0u) {
        obs.decode_elapsed_ms = static_cast<double>(timings.eval_ns) / 1e6;
    }

    if ((timings.speed_present || timings.predicted_tokens_per_second > 0.0) &&
        std::isfinite(timings.predicted_tokens_per_second) && timings.predicted_tokens_per_second > 0.0) {
        obs.decode_tps = timings.predicted_tokens_per_second;
    } else if (obs.predicted_n > 0 && timings.eval_ns > 0) {
        const double tps = static_cast<double>(obs.predicted_n) * 1e9 / static_cast<double>(timings.eval_ns);
        if (std::isfinite(tps) && tps > 0.0)
            obs.decode_tps = tps;
    } else if (obs.predicted_n > 0 && obs.decode_elapsed_ms > 0.0) {
        const double tps = static_cast<double>(obs.predicted_n) * 1000.0 / obs.decode_elapsed_ms;
        if (std::isfinite(tps) && tps > 0.0)
            obs.decode_tps = tps;
    }

    return obs;
}

SpeculationObservation observation_from_timings(const BackendTimings &timings, SpeculationWorkload workload) {
    SpeculationObservation obs;
    obs.workload = workload;
    obs.draft_n = timings.draft_n.value_or(0);
    obs.draft_n_accepted = std::min(timings.draft_n_accepted.value_or(0), obs.draft_n);
    obs.predicted_n = timings.predicted_n.value_or(0);
    if (timings.prompt_ms && std::isfinite(*timings.prompt_ms) && *timings.prompt_ms > 0.0)
        obs.prompt_or_restore_ms = *timings.prompt_ms;
    if (timings.predicted_ms && std::isfinite(*timings.predicted_ms) && *timings.predicted_ms > 0.0) {
        obs.decode_elapsed_ms = *timings.predicted_ms;
        if (obs.predicted_n > 0) {
            const double tps = static_cast<double>(obs.predicted_n) * 1000.0 / *timings.predicted_ms;
            if (std::isfinite(tps) && tps > 0.0)
                obs.decode_tps = tps;
        }
    }
    return obs;
}

AdaptiveSpeculationPolicy::AdaptiveSpeculationPolicy(AdaptiveSpeculationConfig config)
    : config_(sanitize_config(config)) {}

SpeculationContextKey AdaptiveSpeculationPolicy::normalize_key(const SpeculationContextKey &key) {
    SpeculationContextKey out = key;
    out.target_identity = normalize_speculation_identity(key.target_identity);
    out.draft_identity = normalize_speculation_identity(key.draft_identity);
    return out;
}

SpeculationContextKey AdaptiveSpeculationPolicy::make_default_key(SpeculationWorkload workload) {
    SpeculationContextKey key;
    key.workload = workload;
    return key;
}

void AdaptiveSpeculationPolicy::clear_window(BucketState &bucket) noexcept {
    std::fill(bucket.window.begin(), bucket.window.end(), Sample{});
    bucket.head = 0;
    bucket.count = 0;
}

void AdaptiveSpeculationPolicy::push_sample(BucketState &bucket, const Sample &sample) noexcept {
    if (bucket.window.empty())
        return;
    bucket.window[bucket.head] = sample;
    bucket.head = (bucket.head + 1u) % bucket.window.size();
    if (bucket.count < bucket.window.size())
        ++bucket.count;
}

void AdaptiveSpeculationPolicy::push_baseline_sample(BucketState &bucket, const BaselineSample &sample) noexcept {
    if (bucket.baseline_window.empty())
        return;
    bucket.baseline_window[bucket.baseline_head] = sample;
    bucket.baseline_head = (bucket.baseline_head + 1u) % bucket.baseline_window.size();
    if (bucket.baseline_count < bucket.baseline_window.size())
        ++bucket.baseline_count;
}

void AdaptiveSpeculationPolicy::update_per_k_cost(BucketState &bucket, std::uint32_t draft_n, double ms_per_token,
                                                  double acceptance) noexcept {
    if (!std::isfinite(ms_per_token) || ms_per_token <= 0.0)
        return;
    bucket.cost_seq = saturating_add(bucket.cost_seq, 1u);
    auto it = std::find_if(bucket.per_k_costs.begin(), bucket.per_k_costs.end(),
                           [draft_n](const DraftCostStats &s) { return s.draft_n == draft_n; });
    if (it == bucket.per_k_costs.end()) {
        if (bucket.per_k_costs.size() >= AdaptiveSpeculationConfig::kMaxTrackedDraftLevels) {
            // Evict least recently seen non-zero K entry so K=0 baseline stays pinned.
            auto victim = bucket.per_k_costs.end();
            for (auto cand = bucket.per_k_costs.begin(); cand != bucket.per_k_costs.end(); ++cand) {
                if (cand->draft_n == 0u)
                    continue;
                if (victim == bucket.per_k_costs.end() || cand->last_seen_seq < victim->last_seen_seq)
                    victim = cand;
            }
            if (victim == bucket.per_k_costs.end())
                victim = bucket.per_k_costs.begin();
            *victim = DraftCostStats{draft_n, 1u, ms_per_token, acceptance, bucket.cost_seq};
            return;
        }
        bucket.per_k_costs.push_back(DraftCostStats{draft_n, 1u, ms_per_token, acceptance, bucket.cost_seq});
        return;
    }

    const std::size_t prev_n = it->sample_count;
    if (it->sample_count < std::numeric_limits<std::size_t>::max())
        ++it->sample_count;
    if (prev_n < 4u) {
        const double denom = static_cast<double>(prev_n + 1u);
        it->ema_ms_per_token =
            (it->ema_ms_per_token * static_cast<double>(prev_n) + ms_per_token) / denom;
        it->ema_acceptance = (it->ema_acceptance * static_cast<double>(prev_n) + acceptance) / denom;
    } else {
        constexpr double kAlpha = 0.35;
        it->ema_ms_per_token = (1.0 - kAlpha) * it->ema_ms_per_token + kAlpha * ms_per_token;
        it->ema_acceptance = (1.0 - kAlpha) * it->ema_acceptance + kAlpha * acceptance;
    }
    it->last_seen_seq = bucket.cost_seq;
}

const AdaptiveSpeculationPolicy::DraftCostStats *AdaptiveSpeculationPolicy::find_per_k_cost(
    const BucketState &bucket, std::uint32_t draft_n) noexcept {
    for (const auto &entry : bucket.per_k_costs) {
        if (entry.draft_n == draft_n)
            return &entry;
    }
    return nullptr;
}

AdaptiveSpeculationPolicy::WindowStats AdaptiveSpeculationPolicy::compute_window_stats(
    const BucketState &bucket) noexcept {
    WindowStats stats;
    long double total_draft = 0.0L;
    long double total_accepted = 0.0L;
    long double total_tps = 0.0L;
    std::size_t tps_samples = 0u;
    long double total_elapsed_ms = 0.0L;
    long double total_committed = 0.0L;
    long double total_ms_per_tok = 0.0L;
    std::size_t ms_per_tok_samples = 0u;
    long double total_prompt_ms = 0.0L;
    std::size_t prompt_samples = 0u;

    for (std::size_t i = 0; i < bucket.count; ++i) {
        const Sample &sample = bucket.window[i];
        total_draft += static_cast<long double>(sample.draft_n);
        total_accepted += static_cast<long double>(sample.draft_n_accepted);
        stats.draft_tokens = saturating_add(stats.draft_tokens, sample.draft_n);
        stats.accepted_tokens = saturating_add(stats.accepted_tokens, sample.draft_n_accepted);
        if (std::isfinite(sample.decode_tps) && sample.decode_tps > 0.0) {
            total_tps += static_cast<long double>(sample.decode_tps);
            ++tps_samples;
        }
        if (std::isfinite(sample.decode_elapsed_ms) && sample.decode_elapsed_ms > 0.0 &&
            sample.committed_tokens > 0u) {
            total_elapsed_ms += static_cast<long double>(sample.decode_elapsed_ms);
            total_committed += static_cast<long double>(sample.committed_tokens);
        }
        if (std::isfinite(sample.ms_per_token) && sample.ms_per_token > 0.0) {
            total_ms_per_tok += static_cast<long double>(sample.ms_per_token);
            ++ms_per_tok_samples;
        }
        if (std::isfinite(sample.prompt_or_restore_ms) && sample.prompt_or_restore_ms > 0.0) {
            total_prompt_ms += static_cast<long double>(sample.prompt_or_restore_ms);
            ++prompt_samples;
        }
    }

    if (total_draft > 0.0L) {
        const double ratio = static_cast<double>(total_accepted / total_draft);
        stats.rolling_acceptance = std::isfinite(ratio) ? std::clamp(ratio, 0.0, 1.0) : 0.0;
    }
    if (total_committed > 0.0L && total_elapsed_ms > 0.0L) {
        const double ms_per_tok = static_cast<double>(total_elapsed_ms / total_committed);
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0)
            stats.rolling_ms_per_token = ms_per_tok;
    } else if (ms_per_tok_samples > 0u) {
        const double ms_per_tok = static_cast<double>(total_ms_per_tok / static_cast<long double>(ms_per_tok_samples));
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0)
            stats.rolling_ms_per_token = ms_per_tok;
    }
    if (tps_samples > 0u) {
        const double mean_tps = static_cast<double>(total_tps / static_cast<long double>(tps_samples));
        if (std::isfinite(mean_tps) && mean_tps > 0.0)
            stats.rolling_decode_tps = mean_tps;
    } else if (stats.rolling_ms_per_token > 0.0) {
        stats.rolling_decode_tps = 1000.0 / stats.rolling_ms_per_token;
    }
    if (prompt_samples > 0u) {
        const double mean_prompt = static_cast<double>(total_prompt_ms / static_cast<long double>(prompt_samples));
        if (std::isfinite(mean_prompt) && mean_prompt > 0.0)
            stats.rolling_prompt_or_restore_ms = mean_prompt;
    }
    return stats;
}

AdaptiveSpeculationPolicy::BaselineStats AdaptiveSpeculationPolicy::compute_baseline_stats(
    const BucketState &bucket) noexcept {
    BaselineStats stats;
    stats.sample_count = bucket.baseline_count;
    long double total_elapsed_ms = 0.0L;
    long double total_committed = 0.0L;
    long double total_ms_per_tok = 0.0L;
    std::size_t ms_samples = 0u;
    for (std::size_t i = 0; i < bucket.baseline_count; ++i) {
        const BaselineSample &s = bucket.baseline_window[i];
        if (std::isfinite(s.decode_elapsed_ms) && s.decode_elapsed_ms > 0.0 && s.committed_tokens > 0u) {
            total_elapsed_ms += static_cast<long double>(s.decode_elapsed_ms);
            total_committed += static_cast<long double>(s.committed_tokens);
        }
        if (std::isfinite(s.ms_per_token) && s.ms_per_token > 0.0) {
            total_ms_per_tok += static_cast<long double>(s.ms_per_token);
            ++ms_samples;
        }
    }
    if (total_committed > 0.0L && total_elapsed_ms > 0.0L) {
        const double ms_per_tok = static_cast<double>(total_elapsed_ms / total_committed);
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0)
            stats.ms_per_token = ms_per_tok;
    } else if (ms_samples > 0u) {
        const double ms_per_tok = static_cast<double>(total_ms_per_tok / static_cast<long double>(ms_samples));
        if (std::isfinite(ms_per_tok) && ms_per_tok > 0.0)
            stats.ms_per_token = ms_per_tok;
    }
    return stats;
}

void AdaptiveSpeculationPolicy::reset_bucket_locked(BucketState &bucket) const {
    if (bucket.window.size() != config_.window_size)
        bucket.window.resize(config_.window_size);
    clear_window(bucket);
    bucket.baseline_window.assign(config_.window_size, BaselineSample{});
    bucket.baseline_head = 0;
    bucket.baseline_count = 0;
    bucket.per_k_costs.clear();
    bucket.cost_seq = 0;
    bucket.current_draft_n = config_.initial_draft_n;
    bucket.paused = (config_.initial_draft_n == 0);
    bucket.probe_in_flight = false;
    bucket.active_probe_kind = ProbeKind::none;
    bucket.active_probe_id = 0;
    bucket.cooldown_remaining = bucket.paused ? config_.cooldown_requests : 0;
    bucket.step_up_hold_remaining = 0;
    bucket.requests_since_baseline = 0;
    bucket.total_observations = 0;
    bucket.baseline_observations = 0;
    bucket.ignored_observations = 0;
    bucket.clamped_observations = 0;
    bucket.probes_issued = 0;
    bucket.probe_recoveries = 0;
    bucket.cost_step_downs = 0;
    bucket.cost_pauses = 0;
    bucket.last_reason = !config_.enabled ? "disabled" : (bucket.paused ? "initial_paused" : "initial");
}

AdaptiveSpeculationPolicy::BucketState &AdaptiveSpeculationPolicy::find_or_create_locked(
    const SpeculationContextKey &raw_key) {
    const SpeculationContextKey key = normalize_key(raw_key);
    if (auto it = index_.find(key); it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second);
        return it->second->state;
    }

    while (index_.size() >= config_.max_entries && !lru_.empty()) {
        // Prefer evicting the oldest entry without an active in-flight probe;
        // fall back to true LRU tail so capacity is never exceeded.
        auto victim = std::prev(lru_.end());
        for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
            if (!it->state.probe_in_flight) {
                victim = std::next(it).base();
                break;
            }
        }
        index_.erase(victim->key);
        lru_.erase(victim);
        table_evictions_ = saturating_add(table_evictions_, 1u);
    }

    TableEntry entry;
    entry.key = key;
    reset_bucket_locked(entry.state);
    lru_.push_front(std::move(entry));
    index_.emplace(lru_.front().key, lru_.begin());
    return lru_.front().state;
}

AdaptiveSpeculationPolicy::BucketState *AdaptiveSpeculationPolicy::find_existing_locked(
    const SpeculationContextKey &raw_key) {
    const SpeculationContextKey key = normalize_key(raw_key);
    auto it = index_.find(key);
    if (it == index_.end())
        return nullptr;
    return &it->second->state;
}

SpeculationDecision AdaptiveSpeculationPolicy::recommend(SpeculationWorkload workload) {
    return recommend(make_default_key(workload), true);
}

SpeculationDecision AdaptiveSpeculationPolicy::recommend(const SpeculationContextKey &key, bool allow_probe) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.enabled) {
        return SpeculationDecision{
            config_.initial_draft_n,
            config_.initial_draft_n == 0,
            false,
            0.0,
            0,
            "disabled",
        };
    }

    BucketState &bucket = find_or_create_locked(key);
    const WindowStats stats = compute_window_stats(bucket);
    const BaselineStats baseline = compute_baseline_stats(bucket);
    const double cost_ratio =
        (stats.rolling_ms_per_token > 0.0 && baseline.ms_per_token > 0.0)
            ? (stats.rolling_ms_per_token / baseline.ms_per_token)
            : 0.0;

    if (!bucket.paused) {
        if (config_.baseline_probe_interval > 0u) {
            bucket.requests_since_baseline = saturating_inc_u32(bucket.requests_since_baseline);
            if (allow_probe && !bucket.probe_in_flight &&
                bucket.requests_since_baseline >= config_.baseline_probe_interval) {
                if (++next_probe_id_ == 0u)
                    ++next_probe_id_;
                bucket.probe_in_flight = true;
                bucket.active_probe_kind = ProbeKind::baseline_calibration;
                bucket.active_probe_id = next_probe_id_;
                bucket.probes_issued = saturating_add(bucket.probes_issued, 1u);
                return SpeculationDecision{
                    0u,
                    true,
                    true,
                    stats.rolling_acceptance,
                    bucket.count,
                    "baseline_calibration_probe",
                    bucket.active_probe_id,
                    stats.rolling_ms_per_token,
                    baseline.ms_per_token,
                    cost_ratio,
                    true,
                };
            }
        }
        return SpeculationDecision{
            bucket.current_draft_n,
            false,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
            0u,
            stats.rolling_ms_per_token,
            baseline.ms_per_token,
            cost_ratio,
            false,
        };
    }

    if (config_.cooldown_requests == 0u) {
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_no_probe",
            0u,
            stats.rolling_ms_per_token,
            baseline.ms_per_token,
            cost_ratio,
            false,
        };
    }

    if (bucket.probe_in_flight) {
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_probe_in_flight",
            0u,
            stats.rolling_ms_per_token,
            baseline.ms_per_token,
            cost_ratio,
            false,
        };
    }

    if (bucket.cooldown_remaining > 0u) {
        --bucket.cooldown_remaining;
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_cooldown",
            0u,
            stats.rolling_ms_per_token,
            baseline.ms_per_token,
            cost_ratio,
            false,
        };
    }

    if (!allow_probe) {
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_probe_deferred",
            0u,
            stats.rolling_ms_per_token,
            baseline.ms_per_token,
            cost_ratio,
            false,
        };
    }

    if (++next_probe_id_ == 0u)
        ++next_probe_id_;
    bucket.probe_in_flight = true;
    bucket.active_probe_kind = ProbeKind::cooldown_reprobe;
    bucket.active_probe_id = next_probe_id_;
    bucket.probes_issued = saturating_add(bucket.probes_issued, 1u);
    return SpeculationDecision{
        config_.probe_draft_n,
        false,
        true,
        stats.rolling_acceptance,
        bucket.count,
        "cooldown_probe",
        bucket.active_probe_id,
        stats.rolling_ms_per_token,
        baseline.ms_per_token,
        cost_ratio,
        false,
    };
}

SpeculationDecision AdaptiveSpeculationPolicy::evaluate_active_locked(
    BucketState &bucket, std::uint32_t effective_k, const WindowStats &stats,
    const BaselineStats &baseline_stats, bool decrement_hold, bool cost_regression_only) {
    const std::uint32_t active_floor = std::max<std::uint32_t>(1u, config_.min_draft_n);
    const std::size_t eff_min_baseline = std::min(config_.min_baseline_samples, config_.window_size);
    const std::size_t min_k_samples = std::min<std::size_t>(2u, config_.min_samples);

    if (decrement_hold && bucket.step_up_hold_remaining > 0u)
        --bucket.step_up_hold_remaining;

    const DraftCostStats *cur_k = find_per_k_cost(bucket, effective_k);
    const bool has_cur_k_cost = (cur_k != nullptr && cur_k->ema_ms_per_token > 0.0);
    const double measured_ms = has_cur_k_cost ? cur_k->ema_ms_per_token : stats.rolling_ms_per_token;
    const bool cur_k_mature =
        (cur_k != nullptr && cur_k->sample_count >= min_k_samples && cur_k->ema_ms_per_token > 0.0) ||
        (cost_regression_only && bucket.count >= min_k_samples && stats.rolling_ms_per_token > 0.0);

    const bool has_baseline =
        (baseline_stats.sample_count >= eff_min_baseline && baseline_stats.ms_per_token > 0.0);
    const double baseline_ratio =
        (has_baseline && measured_ms > 0.0) ? (measured_ms / baseline_stats.ms_per_token) : 0.0;

    // Locate best previously measured shorter speculative depth 0 < K_lower < current_draft_n.
    const DraftCostStats *best_lower_k = nullptr;
    for (const auto &entry : bucket.per_k_costs) {
        if (entry.draft_n > 0u && entry.draft_n < bucket.current_draft_n &&
            entry.sample_count >= min_k_samples && entry.ema_ms_per_token > 0.0) {
            if (best_lower_k == nullptr || entry.ema_ms_per_token < best_lower_k->ema_ms_per_token)
                best_lower_k = &entry;
        }
    }

    const auto make_decision = [&](std::uint32_t k, bool is_paused) {
        const double report_ms = measured_ms > 0.0 ? measured_ms : stats.rolling_ms_per_token;
        const double report_ratio =
            (report_ms > 0.0 && baseline_stats.ms_per_token > 0.0)
                ? (report_ms / baseline_stats.ms_per_token)
                : 0.0;
        return SpeculationDecision{
            k,
            is_paused,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
            0u,
            report_ms,
            baseline_stats.ms_per_token,
            report_ratio,
            false,
        };
    };

    // 1. Measured-cost regression against K=0 baseline overrides acceptance rate.
    if (has_baseline && cur_k_mature && baseline_ratio >= config_.cost_step_down_ratio) {
        if (config_.min_draft_n == 0u &&
            (baseline_ratio >= config_.cost_pause_ratio || bucket.current_draft_n <= active_floor)) {
            bucket.paused = true;
            bucket.probe_in_flight = false;
            bucket.active_probe_kind = ProbeKind::none;
            bucket.active_probe_id = 0u;
            bucket.current_draft_n = 0u;
            bucket.cooldown_remaining = config_.cooldown_requests;
            bucket.cost_pauses = saturating_add(bucket.cost_pauses, 1u);
            bucket.last_reason = "paused_cost_regression";
            return make_decision(0u, true);
        }
        if (bucket.current_draft_n > active_floor) {
            --bucket.current_draft_n;
            bucket.step_up_hold_remaining = config_.cost_step_cooldown;
            bucket.cost_step_downs = saturating_add(bucket.cost_step_downs, 1u);
            bucket.last_reason = "step_down_cost";
        } else {
            bucket.current_draft_n = active_floor;
            bucket.step_up_hold_remaining = config_.cost_step_cooldown;
            bucket.last_reason = "at_min_draft_cost";
        }
        return make_decision(bucket.current_draft_n, false);
    }

    // Baseline-only observations never advance speculative depth upward without
    // a new speculative observation; they only trigger cost step-down / pause.
    if (cost_regression_only) {
        const double report_ms = measured_ms > 0.0 ? measured_ms : stats.rolling_ms_per_token;
        const double report_ratio =
            (report_ms > 0.0 && baseline_stats.ms_per_token > 0.0)
                ? (report_ms / baseline_stats.ms_per_token)
                : 0.0;
        return SpeculationDecision{
            bucket.current_draft_n,
            false,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "baseline_observed",
            0u,
            report_ms,
            baseline_stats.ms_per_token,
            report_ratio,
            false,
        };
    }

    // 2. Measured-cost regression against a shorter K_lower < current_draft_n
    //    (works even without a K=0 baseline and stops K <-> K+1 oscillation).
    if (cur_k_mature && best_lower_k != nullptr &&
        cur_k->ema_ms_per_token >= best_lower_k->ema_ms_per_token * config_.cost_step_down_ratio) {
        if (bucket.current_draft_n > active_floor) {
            --bucket.current_draft_n;
            bucket.step_up_hold_remaining = config_.cost_step_cooldown;
            bucket.cost_step_downs = saturating_add(bucket.cost_step_downs, 1u);
            bucket.last_reason = "step_down_cost_vs_shorter";
        } else {
            bucket.current_draft_n = active_floor;
            bucket.step_up_hold_remaining = config_.cost_step_cooldown;
            bucket.last_reason = "at_min_draft_cost";
        }
        return make_decision(bucket.current_draft_n, false);
    }

    // 3. Acceptance safety floor (pause when acceptance collapses).
    if (config_.min_draft_n == 0u && stats.rolling_acceptance < config_.pause_acceptance_threshold) {
        bucket.paused = true;
        bucket.probe_in_flight = false;
        bucket.active_probe_kind = ProbeKind::none;
        bucket.active_probe_id = 0u;
        bucket.current_draft_n = 0u;
        bucket.cooldown_remaining = config_.cooldown_requests;
        bucket.last_reason = "paused_low_acceptance";
        return make_decision(0u, true);
    }

    // 4. Low acceptance step-down.
    if (stats.rolling_acceptance < config_.step_down_acceptance_threshold) {
        if (bucket.current_draft_n > active_floor) {
            --bucket.current_draft_n;
            bucket.last_reason = "step_down";
        } else {
            bucket.current_draft_n = active_floor;
            bucket.last_reason = "at_min_draft";
        }
        return make_decision(bucket.current_draft_n, false);
    }

    // 5. Step-up gated by acceptance, cost non-regression, and reversal holdoff.
    if (stats.rolling_acceptance >= config_.step_up_acceptance_threshold) {
        if (bucket.step_up_hold_remaining > 0u) {
            bucket.last_reason = "hold_cost_cooldown";
            return make_decision(bucket.current_draft_n, false);
        }
        if (has_baseline && measured_ms > 0.0 && baseline_ratio > 1.0 + 1e-6) {
            bucket.last_reason = "hold_cost_hysteresis";
            return make_decision(bucket.current_draft_n, false);
        }
        if (has_cur_k_cost && best_lower_k != nullptr &&
            cur_k->ema_ms_per_token > best_lower_k->ema_ms_per_token * 1.01) {
            bucket.last_reason = "hold_cost_hysteresis";
            return make_decision(bucket.current_draft_n, false);
        }
        if (bucket.current_draft_n < config_.max_draft_n) {
            ++bucket.current_draft_n;
            bucket.last_reason = "step_up";
        } else {
            bucket.current_draft_n = config_.max_draft_n;
            bucket.last_reason = "at_max_draft";
        }
        return make_decision(bucket.current_draft_n, false);
    }

    bucket.last_reason = "hold_hysteresis";
    return make_decision(bucket.current_draft_n, false);
}

SpeculationDecision AdaptiveSpeculationPolicy::observe(const SpeculationObservation &obs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.enabled) {
        return SpeculationDecision{
            config_.initial_draft_n,
            config_.initial_draft_n == 0,
            false,
            0.0,
            0,
            "disabled",
        };
    }

    const SpeculationContextKey key = obs.has_context_key ? obs.context_key : make_default_key(obs.workload);
    BucketState &bucket = find_or_create_locked(key);

    const ComputedCost cost = compute_observation_cost(obs);
    const bool matches_active_probe =
        bucket.probe_in_flight && (obs.is_probe || obs.probe_id != 0u) &&
        (obs.probe_id == 0u || obs.probe_id == bucket.active_probe_id);
    const bool matches_baseline_probe =
        matches_active_probe && bucket.active_probe_kind == ProbeKind::baseline_calibration;
    const bool matches_cooldown_probe =
        matches_active_probe && bucket.paused && bucket.active_probe_kind == ProbeKind::cooldown_reprobe;

    // Handle K=0 baseline observations (explicit draft_n_used == 0 or baseline calibration probe).
    const bool is_baseline_obs =
        matches_baseline_probe ||
        (obs.draft_n_used.has_value() && *obs.draft_n_used == 0u && !matches_cooldown_probe);
    if (is_baseline_obs) {
        if (matches_baseline_probe) {
            bucket.probe_in_flight = false;
            bucket.active_probe_kind = ProbeKind::none;
            bucket.active_probe_id = 0u;
            bucket.requests_since_baseline = 0u;
        }
        if (cost.ms_per_token > 0.0) {
            push_baseline_sample(bucket, BaselineSample{obs.predicted_n, cost.decode_elapsed_ms, cost.ms_per_token});
            update_per_k_cost(bucket, 0u, cost.ms_per_token, 0.0);
            bucket.baseline_observations = saturating_add(bucket.baseline_observations, 1u);
            bucket.requests_since_baseline = 0u;
        }
        const WindowStats stats = compute_window_stats(bucket);
        const BaselineStats baseline_stats = compute_baseline_stats(bucket);
        if (!bucket.paused && bucket.count >= config_.min_samples && cost.ms_per_token > 0.0) {
            return evaluate_active_locked(bucket, bucket.current_draft_n, stats, baseline_stats, false, true);
        }
        const double ratio = (stats.rolling_ms_per_token > 0.0 && baseline_stats.ms_per_token > 0.0)
                                 ? (stats.rolling_ms_per_token / baseline_stats.ms_per_token)
                                 : 0.0;
        const std::string reason =
            bucket.paused ? (bucket.probe_in_flight ? "paused_probe_in_flight" : "baseline_observed_paused")
                          : "baseline_observed";
        return SpeculationDecision{
            bucket.paused ? 0u : bucket.current_draft_n,
            bucket.paused,
            false,
            stats.rolling_acceptance,
            bucket.count,
            reason,
            0u,
            stats.rolling_ms_per_token,
            baseline_stats.ms_per_token,
            ratio,
            false,
        };
    }

    const std::uint64_t min_tokens = std::max<std::uint64_t>(1u, config_.min_draft_tokens_per_sample);
    if (obs.draft_n < min_tokens) {
        bucket.ignored_observations = saturating_add(bucket.ignored_observations, 1u);
        // Release an in-flight cooldown probe only when the matching probe itself
        // completes below the evaluation floor so the bucket cannot wedge permanently,
        // and restart cooldown so short responses do not spin-probe every request.
        if (matches_cooldown_probe) {
            bucket.probe_in_flight = false;
            bucket.active_probe_kind = ProbeKind::none;
            bucket.active_probe_id = 0u;
            bucket.cooldown_remaining = config_.cooldown_requests;
        }
        const WindowStats stats = compute_window_stats(bucket);
        const BaselineStats baseline_stats = compute_baseline_stats(bucket);
        const double ratio = (stats.rolling_ms_per_token > 0.0 && baseline_stats.ms_per_token > 0.0)
                                 ? (stats.rolling_ms_per_token / baseline_stats.ms_per_token)
                                 : 0.0;
        return SpeculationDecision{
            bucket.paused ? 0u : bucket.current_draft_n,
            bucket.paused,
            false,
            stats.rolling_acceptance,
            bucket.count,
            obs.draft_n == 0u ? "ignored_no_draft" : "ignored_below_min_draft_tokens",
            0u,
            stats.rolling_ms_per_token,
            baseline_stats.ms_per_token,
            ratio,
            false,
        };
    }

    std::uint64_t accepted = obs.draft_n_accepted;
    if (accepted > obs.draft_n) {
        accepted = obs.draft_n;
        bucket.clamped_observations = saturating_add(bucket.clamped_observations, 1u);
    }

    const std::uint32_t active_floor = std::max<std::uint32_t>(1u, config_.min_draft_n);
    const std::uint32_t effective_k =
        (obs.draft_n_used.has_value() && *obs.draft_n_used > 0u)
            ? *obs.draft_n_used
            : (matches_cooldown_probe ? std::clamp(config_.probe_draft_n, active_floor, config_.max_draft_n)
                                      : std::max<std::uint32_t>(1u, bucket.current_draft_n));

    const double sample_acceptance = std::clamp(
        static_cast<double>(static_cast<long double>(accepted) / static_cast<long double>(obs.draft_n)), 0.0, 1.0);
    const Sample sample{effective_k, obs.draft_n,      accepted,         obs.predicted_n,
                        cost.decode_elapsed_ms, cost.ms_per_token, cost.decode_tps, cost.prompt_or_restore_ms};
    bucket.total_observations = saturating_add(bucket.total_observations, 1u);
    update_per_k_cost(bucket, effective_k, cost.ms_per_token, sample_acceptance);

    const std::size_t eff_min_baseline = std::min(config_.min_baseline_samples, config_.window_size);
    const BaselineStats baseline_stats = compute_baseline_stats(bucket);
    const bool has_baseline =
        (baseline_stats.sample_count >= eff_min_baseline && baseline_stats.ms_per_token > 0.0);

    // Resolve an in-flight cooldown re-probe only from the matching probe observation.
    if (matches_cooldown_probe) {
        bucket.probe_in_flight = false;
        bucket.active_probe_kind = ProbeKind::none;
        bucket.active_probe_id = 0u;
        const bool acceptance_ok = (sample_acceptance >= config_.pause_acceptance_threshold);
        const bool cost_ok = (!has_baseline || cost.ms_per_token <= 0.0 ||
                              cost.ms_per_token <= baseline_stats.ms_per_token * config_.cost_probe_recover_ratio);
        if (acceptance_ok && cost_ok) {
            clear_window(bucket);
            push_sample(bucket, sample);
            bucket.paused = false;
            bucket.cooldown_remaining = 0u;
            bucket.current_draft_n = std::clamp(config_.probe_draft_n, active_floor, config_.max_draft_n);
            bucket.probe_recoveries = saturating_add(bucket.probe_recoveries, 1u);
            bucket.last_reason = "probe_recovered";
            const WindowStats stats = compute_window_stats(bucket);
            const double ratio = (stats.rolling_ms_per_token > 0.0 && baseline_stats.ms_per_token > 0.0)
                                     ? (stats.rolling_ms_per_token / baseline_stats.ms_per_token)
                                     : 0.0;
            return SpeculationDecision{
                bucket.current_draft_n,
                false,
                false,
                stats.rolling_acceptance,
                bucket.count,
                bucket.last_reason,
                0u,
                stats.rolling_ms_per_token,
                baseline_stats.ms_per_token,
                ratio,
                false,
            };
        }

        push_sample(bucket, sample);
        bucket.current_draft_n = 0u;
        bucket.cooldown_remaining = config_.cooldown_requests;
        bucket.last_reason = !acceptance_ok ? "probe_failed_repaused" : "probe_failed_cost_repaused";
        const WindowStats stats = compute_window_stats(bucket);
        const double ratio = (stats.rolling_ms_per_token > 0.0 && baseline_stats.ms_per_token > 0.0)
                                 ? (stats.rolling_ms_per_token / baseline_stats.ms_per_token)
                                 : 0.0;
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
            0u,
            stats.rolling_ms_per_token,
            baseline_stats.ms_per_token,
            ratio,
            false,
        };
    }

    push_sample(bucket, sample);
    const WindowStats stats = compute_window_stats(bucket);
    const double rolling_ratio = (stats.rolling_ms_per_token > 0.0 && baseline_stats.ms_per_token > 0.0)
                                     ? (stats.rolling_ms_per_token / baseline_stats.ms_per_token)
                                     : 0.0;

    if (bucket.paused) {
        const bool cost_ok =
            (!has_baseline || stats.rolling_ms_per_token <= 0.0 ||
             stats.rolling_ms_per_token <= baseline_stats.ms_per_token * config_.cost_probe_recover_ratio);
        if (!bucket.probe_in_flight && bucket.count >= config_.min_samples &&
            stats.rolling_acceptance >= config_.step_down_acceptance_threshold && cost_ok) {
            bucket.paused = false;
            bucket.cooldown_remaining = 0u;
            bucket.current_draft_n = std::clamp(config_.probe_draft_n, active_floor, config_.max_draft_n);
            bucket.last_reason = "recovered";
            return SpeculationDecision{
                bucket.current_draft_n,
                false,
                false,
                stats.rolling_acceptance,
                bucket.count,
                bucket.last_reason,
                0u,
                stats.rolling_ms_per_token,
                baseline_stats.ms_per_token,
                rolling_ratio,
                false,
            };
        }
        if (!bucket.probe_in_flight)
            bucket.last_reason = cost_ok ? "paused_low_acceptance" : "paused_cost_regression";
        return SpeculationDecision{
            0u,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.probe_in_flight ? "paused_probe_in_flight" : bucket.last_reason,
            0u,
            stats.rolling_ms_per_token,
            baseline_stats.ms_per_token,
            rolling_ratio,
            false,
        };
    }

    if (bucket.count < config_.min_samples) {
        bucket.last_reason = "warming_up";
        return SpeculationDecision{
            bucket.current_draft_n,
            false,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
            0u,
            stats.rolling_ms_per_token,
            baseline_stats.ms_per_token,
            rolling_ratio,
            false,
        };
    }

    return evaluate_active_locked(bucket, effective_k, stats, baseline_stats, true);
}

void AdaptiveSpeculationPolicy::cancel_probe(SpeculationWorkload workload, std::uint64_t probe_id) {
    cancel_probe(make_default_key(workload), probe_id);
}

void AdaptiveSpeculationPolicy::cancel_probe(const SpeculationContextKey &key, std::uint64_t probe_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    BucketState *bucket = find_existing_locked(key);
    if (bucket == nullptr || !bucket->probe_in_flight)
        return;
    if (probe_id != 0u && probe_id != bucket->active_probe_id)
        return;
    const ProbeKind kind = bucket->active_probe_kind;
    bucket->probe_in_flight = false;
    bucket->active_probe_kind = ProbeKind::none;
    bucket->active_probe_id = 0u;
    if (kind == ProbeKind::cooldown_reprobe || bucket->paused) {
        bucket->cooldown_remaining = config_.cooldown_requests;
    } else {
        bucket->requests_since_baseline = 0u;
    }
}

std::size_t AdaptiveSpeculationPolicy::entry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.size();
}

std::uint64_t AdaptiveSpeculationPolicy::table_evictions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_evictions_;
}

json::Object AdaptiveSpeculationPolicy::serialize_bucket_locked(const BucketState &bucket,
                                                                const SpeculationContextKey *key) const {
    const WindowStats stats = compute_window_stats(bucket);
    const BaselineStats baseline = compute_baseline_stats(bucket);
    const double cost_ratio =
        (stats.rolling_ms_per_token > 0.0 && baseline.ms_per_token > 0.0)
            ? (stats.rolling_ms_per_token / baseline.ms_per_token)
            : 0.0;

    json::Object entry;
    if (key != nullptr) {
        entry.set("target_identity", key->target_identity);
        entry.set("draft_identity", key->draft_identity);
        entry.set("context_band", to_string(key->context_band));
        entry.set("concurrency_band", to_string(key->concurrency_band));
        entry.set("workload", to_string(key->workload));
    }
    entry.set("draft_n_max", static_cast<unsigned long long>(bucket.paused ? 0u : bucket.current_draft_n));
    entry.set("paused", bucket.paused);
    entry.set("probe_in_flight", bucket.probe_in_flight);
    entry.set("cooldown_remaining", static_cast<unsigned long long>(bucket.cooldown_remaining));
    entry.set("step_up_hold_remaining", static_cast<unsigned long long>(bucket.step_up_hold_remaining));
    entry.set("rolling_acceptance", stats.rolling_acceptance);
    entry.set("rolling_decode_tps", stats.rolling_decode_tps);
    entry.set("rolling_ms_per_token", stats.rolling_ms_per_token);
    entry.set("baseline_ms_per_token", baseline.ms_per_token);
    entry.set("baseline_sample_count", static_cast<unsigned long long>(baseline.sample_count));
    entry.set("cost_ratio", cost_ratio);
    entry.set("rolling_prompt_or_restore_ms", stats.rolling_prompt_or_restore_ms);
    entry.set("sample_count", static_cast<unsigned long long>(bucket.count));
    entry.set("window_draft_tokens", static_cast<unsigned long long>(stats.draft_tokens));
    entry.set("window_accepted_tokens", static_cast<unsigned long long>(stats.accepted_tokens));
    entry.set("total_observations", static_cast<unsigned long long>(bucket.total_observations));
    entry.set("baseline_observations", static_cast<unsigned long long>(bucket.baseline_observations));
    entry.set("ignored_observations", static_cast<unsigned long long>(bucket.ignored_observations));
    entry.set("clamped_observations", static_cast<unsigned long long>(bucket.clamped_observations));
    entry.set("probes_issued", static_cast<unsigned long long>(bucket.probes_issued));
    entry.set("probe_recoveries", static_cast<unsigned long long>(bucket.probe_recoveries));
    entry.set("cost_step_downs", static_cast<unsigned long long>(bucket.cost_step_downs));
    entry.set("cost_pauses", static_cast<unsigned long long>(bucket.cost_pauses));
    entry.set("reason", bucket.last_reason);

    json::Array per_k_arr;
    per_k_arr.reserve(bucket.per_k_costs.size());
    for (const auto &kstat : bucket.per_k_costs) {
        json::Object ko;
        ko.set("draft_n", static_cast<unsigned long long>(kstat.draft_n));
        ko.set("sample_count", static_cast<unsigned long long>(kstat.sample_count));
        ko.set("ema_ms_per_token", kstat.ema_ms_per_token);
        ko.set("ema_acceptance", kstat.ema_acceptance);
        per_k_arr.emplace_back(std::move(ko));
    }
    entry.set("per_k_costs", std::move(per_k_arr));
    return entry;
}

json::Object AdaptiveSpeculationPolicy::status_json() const {
    std::lock_guard<std::mutex> lock(mutex_);

    json::Object config_obj;
    config_obj.set("min_draft_n", static_cast<unsigned long long>(config_.min_draft_n));
    config_obj.set("initial_draft_n", static_cast<unsigned long long>(config_.initial_draft_n));
    config_obj.set("max_draft_n", static_cast<unsigned long long>(config_.max_draft_n));
    config_obj.set("window_size", static_cast<unsigned long long>(config_.window_size));
    config_obj.set("min_samples", static_cast<unsigned long long>(config_.min_samples));
    config_obj.set("pause_acceptance_threshold", config_.pause_acceptance_threshold);
    config_obj.set("step_down_acceptance_threshold", config_.step_down_acceptance_threshold);
    config_obj.set("step_up_acceptance_threshold", config_.step_up_acceptance_threshold);
    config_obj.set("cooldown_requests", static_cast<unsigned long long>(config_.cooldown_requests));
    config_obj.set("probe_draft_n", static_cast<unsigned long long>(config_.probe_draft_n));
    config_obj.set("min_draft_tokens_per_sample",
                   static_cast<unsigned long long>(config_.min_draft_tokens_per_sample));
    config_obj.set("max_entries", static_cast<unsigned long long>(config_.max_entries));
    config_obj.set("min_baseline_samples", static_cast<unsigned long long>(config_.min_baseline_samples));
    config_obj.set("cost_step_down_ratio", config_.cost_step_down_ratio);
    config_obj.set("cost_pause_ratio", config_.cost_pause_ratio);
    config_obj.set("cost_probe_recover_ratio", config_.cost_probe_recover_ratio);
    config_obj.set("cost_step_cooldown", static_cast<unsigned long long>(config_.cost_step_cooldown));
    config_obj.set("baseline_probe_interval", static_cast<unsigned long long>(config_.baseline_probe_interval));

    json::Object table_obj;
    table_obj.set("size", static_cast<unsigned long long>(index_.size()));
    table_obj.set("capacity", static_cast<unsigned long long>(config_.max_entries));
    table_obj.set("evictions", static_cast<unsigned long long>(table_evictions_));

    BucketState empty_bucket;
    reset_bucket_locked(empty_bucket);

    json::Object workloads_obj;
    for (const auto workload : kAllWorkloads) {
        const TableEntry *latest = nullptr;
        for (const auto &item : lru_) {
            if (item.key.workload == workload) {
                latest = &item;
                break;
            }
        }
        if (latest != nullptr) {
            workloads_obj.set(to_string(workload), serialize_bucket_locked(latest->state, &latest->key));
        } else {
            workloads_obj.set(to_string(workload), serialize_bucket_locked(empty_bucket, nullptr));
        }
    }

    json::Array entries_arr;
    entries_arr.reserve(lru_.size());
    for (const auto &item : lru_) {
        entries_arr.emplace_back(serialize_bucket_locked(item.state, &item.key));
    }

    json::Object root;
    root.set("enabled", config_.enabled);
    root.set("config", std::move(config_obj));
    root.set("table", std::move(table_obj));
    root.set("workloads", std::move(workloads_obj));
    root.set("entries", std::move(entries_arr));
    return root;
}

void AdaptiveSpeculationPolicy::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.clear();
    lru_.clear();
    table_evictions_ = 0u;
}

}  // namespace sonder::inference::llamaserver
