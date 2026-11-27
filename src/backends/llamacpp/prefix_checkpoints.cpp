#include "prefix_checkpoints.hpp"

#include <algorithm>
#include <cctype>
#include <limits>

namespace sonder::backends::llamacpp {

namespace {

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();
constexpr std::uint64_t kU64Max = std::numeric_limits<std::uint64_t>::max();

std::size_t SatAdd(std::size_t a, std::size_t b) noexcept { return a > kSizeMax - b ? kSizeMax : a + b; }

std::size_t SatMul(std::size_t a, std::size_t b) noexcept {
    if (a == 0 || b == 0) return 0;
    return a > kSizeMax / b ? kSizeMax : a * b;
}

std::uint64_t SatAdd64(std::uint64_t a, std::uint64_t b) noexcept { return a > kU64Max - b ? kU64Max : a + b; }

std::string LowerCopy(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool LowerIntoStack(std::string_view name, char (&buf)[32], std::string_view& lowered) noexcept {
    if (name.empty() || name.size() > sizeof(buf)) return false;
    for (std::size_t i = 0; i < name.size(); ++i) {
        buf[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(name[i])));
    }
    lowered = std::string_view(buf, name.size());
    return true;
}

bool IsPrefix(const std::vector<Token>& tokens, std::span<const Token> of) noexcept {
    if (tokens.size() > of.size()) return false;
    return std::equal(tokens.begin(), tokens.end(), of.begin());
}

}  // namespace

std::string_view ToString(CheckpointMode mode) noexcept {
    switch (mode) {
        case CheckpointMode::kOff: return "off";
        case CheckpointMode::kSparse: return "sparse";
        case CheckpointMode::kDense: return "dense";
    }
    return "off";
}

PrefixCheckpointConfig SanitizeConfig(PrefixCheckpointConfig cfg) noexcept {
    const PrefixCheckpointConfig defaults;
    auto spacing = [&](std::uint32_t v, std::uint32_t dflt) {
        if (v == 0) return dflt;
        return std::clamp(v, PrefixCheckpointConfig::kMinSpacingTokens, PrefixCheckpointConfig::kMaxSpacingTokens);
    };
    cfg.spacing_tokens = spacing(cfg.spacing_tokens, defaults.spacing_tokens);
    cfg.dense_spacing_tokens = spacing(cfg.dense_spacing_tokens, defaults.dense_spacing_tokens);
    if (cfg.tail_offset_tokens == 0) cfg.tail_offset_tokens = defaults.tail_offset_tokens;
    if (cfg.max_per_request == 0) cfg.max_per_request = defaults.max_per_request;
    cfg.max_per_request = std::min(cfg.max_per_request, PrefixCheckpointConfig::kMaxPerRequestLimit);
    cfg.cache_ram_mb = std::min(cfg.cache_ram_mb, PrefixCheckpointConfig::kMaxHostCacheMb);
    cfg.host_max_entries = std::min(cfg.host_max_entries, PrefixCheckpointConfig::kMaxHostEntries);
    return cfg;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

bool IsHybridArchitectureName(std::string_view name) noexcept {
    // llama.cpp v1.0.0 src/llama-arch.cpp: llm_arch_is_hybrid GGUF strings.
    static constexpr std::string_view kHybridNames[] = {
        "jamba", "falcon-h1", "plamo2", "granitehybrid", "lfm2", "lfm2moe", "lfm3", "nemotron_h", "nemotron_h_moe",
        "metis16", "qwen3next", "kimi-linear", "bailingmoe3", "kimi-k3", "kimi-k31", "glm5-next", "qwen35",
        "qwen35moe", "clover_moe", "qwen4exp", "deepseek4", "deepseek41", "minimax-01"};
    char buf[32];
    std::string_view lowered;
    if (!LowerIntoStack(name, buf, lowered)) return false;
    return std::find(std::begin(kHybridNames), std::end(kHybridNames), lowered) != std::end(kHybridNames);
}

bool IsRecurrentArchitectureName(std::string_view name) noexcept {
    // llama.cpp v1.0.0 src/llama-arch.cpp: llm_arch_is_recurrent GGUF strings.
    static constexpr std::string_view kRecurrentNames[] = {
        "mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"};
    char buf[32];
    std::string_view lowered;
    if (!LowerIntoStack(name, buf, lowered)) return false;
    return std::find(std::begin(kRecurrentNames), std::end(kRecurrentNames), lowered) != std::end(kRecurrentNames);
}

bool IsStatefulArchitectureName(std::string_view name) noexcept {
    return IsHybridArchitectureName(name) || IsRecurrentArchitectureName(name);
}

sonder::inference::ModelArchitecture DetectCheckpointArchitecture(const sonder::inference::ModelMetadata& metadata) {
    using sonder::inference::ModelArchitecture;
    const ModelArchitecture shared = sonder::inference::classify_model_architecture(metadata);
    if (shared != ModelArchitecture::attention_only) return shared;
    for (const auto& [key, value] : metadata) {
        const std::string k = LowerCopy(key);
        if (k == "general.architecture" || k == "architecture") {
            if (IsHybridArchitectureName(value)) return ModelArchitecture::hybrid;
            if (IsRecurrentArchitectureName(value)) return ModelArchitecture::recurrent;
        }
    }
    return ModelArchitecture::attention_only;
}

CheckpointMode ClassifyCheckpointMode(const sonder::inference::ModelMetadata& metadata, bool sliding_window,
                                      std::optional<bool> runtime_stateful, const PrefixCheckpointConfig& cfg) {
    if (!cfg.enabled) return CheckpointMode::kOff;
    const bool stateful =
        runtime_stateful.has_value()
            ? *runtime_stateful
            : DetectCheckpointArchitecture(metadata) != sonder::inference::ModelArchitecture::attention_only;
    if (stateful) return CheckpointMode::kDense;
    if (sliding_window || cfg.force_attention_only) return CheckpointMode::kSparse;
    return CheckpointMode::kOff;
}

std::uint32_t SpacingFor(CheckpointMode mode, const PrefixCheckpointConfig& cfg) noexcept {
    const PrefixCheckpointConfig s = SanitizeConfig(cfg);
    switch (mode) {
        case CheckpointMode::kDense: return s.dense_spacing_tokens;
        case CheckpointMode::kSparse: return s.spacing_tokens;
        case CheckpointMode::kOff: return 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

std::vector<std::size_t> PlanCheckpointPositions(std::size_t prefill_start, std::size_t n_prompt, CheckpointMode mode,
                                                 const PrefixCheckpointConfig& raw_cfg) {
    std::vector<std::size_t> out;
    if (mode == CheckpointMode::kOff || !raw_cfg.enabled) return out;
    const PrefixCheckpointConfig cfg = SanitizeConfig(raw_cfg);
    if (n_prompt < cfg.min_prompt_tokens) return out;
    if (n_prompt <= cfg.tail_offset_tokens) return out;

    const std::size_t tail = n_prompt - cfg.tail_offset_tokens;
    if (tail <= prefill_start) return out;  // nothing past the cached prefix

    const std::size_t spacing = SpacingFor(mode, cfg);
    const std::size_t min_gap_to_tail = spacing / 2;

    // Newest first, so the work is bounded by max_per_request and not by the
    // prompt length.
    out.push_back(tail);
    if (tail > min_gap_to_tail) {
        std::size_t p = ((tail - min_gap_to_tail) / spacing) * spacing;  // largest multiple <= tail - gap
        while (p > prefill_start && out.size() < cfg.max_per_request) {
            if (p < tail) out.push_back(p);
            if (p < spacing) break;
            p -= spacing;
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::size_t ClampPrefillStepToCheckpoint(std::span<const std::size_t> planned, std::size_t& cursor, std::size_t off,
                                         std::size_t step_end) noexcept {
    while (cursor < planned.size() && planned[cursor] <= off) {
        ++cursor;
    }
    if (cursor < planned.size() && planned[cursor] < step_end) {
        return planned[cursor];
    }
    return step_end;
}

bool ShouldCaptureAtOffset(std::span<const std::size_t> planned, std::size_t& cursor,
                           std::size_t new_offset) noexcept {
    while (cursor < planned.size() && planned[cursor] < new_offset) {
        ++cursor;
    }
    if (cursor < planned.size() && planned[cursor] == new_offset) {
        ++cursor;
        return true;
    }
    return false;
}

HostCacheBudgetSplit SplitHostCacheBudget(std::uint32_t cache_ram_mb, CheckpointMode mode) noexcept {
    HostCacheBudgetSplit out;
    const std::uint32_t clamped_mb = std::min(cache_ram_mb, PrefixCheckpointConfig::kMaxHostCacheMb);
    const std::uint64_t bytes64 = static_cast<std::uint64_t>(clamped_mb) << 20;
    out.total_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes64, kSizeMax));
    if (out.total_bytes == 0) return out;
    switch (mode) {
        case CheckpointMode::kOff:
            out.checkpoint_bytes = 0;
            out.radix_bytes = out.total_bytes;
            break;
        case CheckpointMode::kSparse:
            out.checkpoint_bytes = out.total_bytes / 8;
            out.radix_bytes = out.total_bytes - out.checkpoint_bytes;
            break;
        case CheckpointMode::kDense:
            out.checkpoint_bytes = out.total_bytes / 4;
            out.radix_bytes = out.total_bytes - out.checkpoint_bytes;
            break;
    }
    return out;
}

L2CheckpointRestorePlan ChooseL2RestorePlan(bool direct_trim_allowed, std::size_t min_beat_lcp,
                                            std::size_t min_gain_tokens, std::size_t trimmed_matched_tokens,
                                            std::size_t trimmed_state_tokens, std::size_t host_ckpt_tokens,
                                            std::size_t exact_matched_tokens) noexcept {
    const auto meets_gain = [&](std::size_t tokens) noexcept -> bool {
        return tokens > min_beat_lcp && (tokens - min_beat_lcp) >= min_gain_tokens;
    };

    const bool exact_ok = meets_gain(exact_matched_tokens);

    std::size_t trimmed_target = 0;
    bool apply_ckpt = false;
    if (trimmed_matched_tokens > 0 && trimmed_state_tokens >= trimmed_matched_tokens) {
        if (trimmed_state_tokens == trimmed_matched_tokens) {
            trimmed_target = trimmed_matched_tokens;
            apply_ckpt = false;
        } else if (direct_trim_allowed) {
            trimmed_target = trimmed_matched_tokens;
            apply_ckpt = false;
        } else if (host_ckpt_tokens > 0 && host_ckpt_tokens <= trimmed_matched_tokens) {
            trimmed_target = host_ckpt_tokens;
            apply_ckpt = true;
        }
    }
    const bool trimmed_ok = meets_gain(trimmed_target);

    L2CheckpointRestorePlan plan;
    if (!exact_ok && !trimmed_ok) return plan;
    if (exact_ok && (!trimmed_ok || exact_matched_tokens >= trimmed_target)) {
        plan.use_exact_l2 = true;
        plan.target_tokens = exact_matched_tokens;
        return plan;
    }
    plan.use_trimmed_l2 = true;
    plan.apply_host_ckpt = apply_ckpt;
    plan.target_tokens = trimmed_target;
    return plan;
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

void PrefixCheckpointStats::OnPlanned(std::size_t n) noexcept {
    checkpoints_planned = SatAdd64(checkpoints_planned, static_cast<std::uint64_t>(n));
}

void PrefixCheckpointStats::OnCaptured(std::size_t bytes) noexcept {
    checkpoints_captured = SatAdd64(checkpoints_captured, 1);
    captured_bytes = SatAdd64(captured_bytes, static_cast<std::uint64_t>(bytes));
}

void PrefixCheckpointStats::OnHit(CheckpointTier tier, std::size_t restored) noexcept {
    if (tier == CheckpointTier::kSlot) {
        slot_hits = SatAdd64(slot_hits, 1);
    } else {
        host_hits = SatAdd64(host_hits, 1);
    }
    restored_tokens = SatAdd64(restored_tokens, static_cast<std::uint64_t>(restored));
}

void PrefixCheckpointStats::OnMiss() noexcept { misses = SatAdd64(misses, 1); }

std::uint64_t PrefixCheckpointStats::Lookups() const noexcept {
    return SatAdd64(SatAdd64(slot_hits, host_hits), misses);
}

double PrefixCheckpointStats::HitRate() const noexcept {
    const std::uint64_t hits = SatAdd64(slot_hits, host_hits);
    const std::uint64_t total = SatAdd64(hits, misses);
    if (total == 0) return 0.0;
    return static_cast<double>(hits) / static_cast<double>(total);
}

PrefixCheckpointCounters PrefixCheckpointStats::ToCounters() const noexcept {
    PrefixCheckpointCounters out;
    out.checkpoints_planned = checkpoints_planned;
    out.checkpoints_captured = checkpoints_captured;
    out.captured_bytes = captured_bytes;
    out.slot_hits = slot_hits;
    out.host_hits = host_hits;
    out.misses = misses;
    out.restored_tokens = restored_tokens;
    out.host_puts = host_puts;
    out.host_evictions = host_evictions;
    out.host_rejected = host_rejected;
    out.host_bytes = host_bytes;
    out.host_entries = host_entries;
    return out;
}

std::vector<std::pair<std::string, std::uint64_t>> StatsToFields(const PrefixCheckpointStats& s) {
    return {
        {"prefix_checkpoint.checkpoints_planned", s.checkpoints_planned},
        {"prefix_checkpoint.checkpoints_captured", s.checkpoints_captured},
        {"prefix_checkpoint.captured_bytes", s.captured_bytes},
        {"prefix_checkpoint.slot_hits", s.slot_hits},
        {"prefix_checkpoint.host_hits", s.host_hits},
        {"prefix_checkpoint.misses", s.misses},
        {"prefix_checkpoint.restored_tokens", s.restored_tokens},
        {"prefix_checkpoint.host_puts", s.host_puts},
        {"prefix_checkpoint.host_evictions", s.host_evictions},
        {"prefix_checkpoint.host_rejected", s.host_rejected},
        {"prefix_checkpoint.host_bytes", s.host_bytes},
        {"prefix_checkpoint.host_entries", s.host_entries},
    };
}

std::string FormatStatsDetail(std::string_view event, const PrefixCheckpointStats& s) {
    std::string out;
    out.reserve(160);
    out.append(event);
    out.append(" slot_hits=");
    out.append(std::to_string(s.slot_hits));
    out.append(" host_hits=");
    out.append(std::to_string(s.host_hits));
    out.append(" misses=");
    out.append(std::to_string(s.misses));
    out.append(" captured=");
    out.append(std::to_string(s.checkpoints_captured));
    out.append(" restored_tokens=");
    out.append(std::to_string(s.restored_tokens));
    out.append(" host_entries=");
    out.append(std::to_string(s.host_entries));
    out.append(" host_bytes=");
    out.append(std::to_string(s.host_bytes));
    out.append(" host_evictions=");
    out.append(std::to_string(s.host_evictions));
    return out;
}

// ---------------------------------------------------------------------------
// Host tier
// ---------------------------------------------------------------------------

std::size_t CheckpointHostTier::EntryBytes(std::size_t n_tokens, std::size_t state_bytes,
                                           std::size_t pending_h_floats) noexcept {
    std::size_t total = CheckpointHostTier::kEntryOverheadBytes;
    total = SatAdd(total, state_bytes);
    total = SatAdd(total, SatMul(n_tokens, sizeof(Token)));
    total = SatAdd(total, SatMul(pending_h_floats, sizeof(float)));
    return total;
}

void CheckpointHostTier::EvictBack() noexcept {
    if (entries_.empty()) return;
    const std::size_t charged = entries_.back().charged;
    bytes_ = bytes_ >= charged ? bytes_ - charged : 0;
    entries_.pop_back();
    ++evictions_;
}

void CheckpointHostTier::Configure(std::size_t max_bytes, std::size_t max_entries) {
    max_bytes_ = max_bytes;
    max_entries_ = max_entries;
    if (!enabled()) {
        entries_.clear();
        bytes_ = 0;
        return;
    }
    while (!entries_.empty() && (bytes_ > max_bytes_ || entries_.size() > max_entries_)) EvictBack();
}

void CheckpointHostTier::ConfigureMiB(std::uint32_t cache_ram_mb, std::size_t max_entries) {
    const std::uint32_t clamped_mb = std::min(cache_ram_mb, PrefixCheckpointConfig::kMaxHostCacheMb);
    const std::uint64_t bytes64 = static_cast<std::uint64_t>(clamped_mb) << 20;
    const std::size_t cap_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes64, kSizeMax));
    const std::size_t cap_entries = std::min<std::size_t>(max_entries, PrefixCheckpointConfig::kMaxHostEntries);
    Configure(cap_bytes, cap_entries);
}

void CheckpointHostTier::Clear() noexcept {
    entries_.clear();
    bytes_ = 0;
}

bool CheckpointHostTier::Put(std::span<const Token> prefix, std::vector<std::uint8_t> state,
                             std::vector<float> pending_h) {
    if (!enabled() || prefix.empty() || state.empty()) {
        ++rejected_;
        return false;
    }
    const std::size_t charged = EntryBytes(prefix.size(), state.size(), pending_h.size());
    if (charged > max_bytes_) {
        ++rejected_;  // alone it exceeds the budget: reject without evicting anything
        return false;
    }

    // Replacing an identical prefix frees its bytes first, only once we know
    // the new entry fits (`charged <= max_bytes_`).
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->tokens.size() == prefix.size() && std::equal(prefix.begin(), prefix.end(), it->tokens.begin())) {
            bytes_ = bytes_ >= it->charged ? bytes_ - it->charged : 0;
            entries_.erase(it);
            break;
        }
    }

    // Evict BEFORE storing so the cap is never exceeded, even transiently.
    while (!entries_.empty() && (entries_.size() >= max_entries_ || bytes_ > max_bytes_ - charged)) EvictBack();

    Entry e;
    e.tokens.assign(prefix.begin(), prefix.end());
    e.state = std::move(state);
    e.pending_h = std::move(pending_h);
    e.charged = charged;
    bytes_ += charged;
    entries_.push_front(std::move(e));
    ++puts_;
    return true;
}

std::optional<CheckpointHostTier::Hit> CheckpointHostTier::Lookup(std::span<const Token> prompt,
                                                                  std::size_t max_match_tokens) {
    if (!enabled() || prompt.empty() || max_match_tokens == 0) return std::nullopt;
    auto best = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const std::size_t n = it->tokens.size();
        if (n > max_match_tokens || n > prompt.size()) continue;
        if (best != entries_.end() && n <= best->tokens.size()) continue;
        if (IsPrefix(it->tokens, prompt)) best = it;
    }
    if (best == entries_.end()) return std::nullopt;
    entries_.splice(entries_.begin(), entries_, best);  // most recently used; iterators stay valid
    const Entry& front = entries_.front();
    return Hit{front.tokens.size(), &front.state, &front.pending_h};
}

std::size_t CheckpointHostTier::EraseNotPrefixOf(std::span<const Token> keep) noexcept {
    std::size_t dropped = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (IsPrefix(it->tokens, keep)) {
            ++it;
            continue;
        }
        bytes_ = bytes_ >= it->charged ? bytes_ - it->charged : 0;
        it = entries_.erase(it);
        ++dropped;
    }
    return dropped;
}

void CheckpointHostTier::FillStats(PrefixCheckpointStats& out) const noexcept {
    out.host_puts = puts_;
    out.host_evictions = evictions_;
    out.host_rejected = rejected_;
    out.host_bytes = bytes_;
    out.host_entries = entries_.size();
}

}  // namespace sonder::backends::llamacpp
