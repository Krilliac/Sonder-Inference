// Prefix-cache checkpoint policy for the native llama.cpp backend.
//
// Pure, model-free pieces (no llama.h) so they can be unit tested directly and
// wired cleanly into LlamaCppBackend:
//
//   1. ClassifyCheckpointMode  - does this model need context checkpoints at
//      all, and how dense? Hybrid / recurrent models (Qwen3.5 / Bonsai-2,
//      Jamba, Mamba, ...) cannot roll a sequence back with
//      `llama_memory_seq_rm`, so a partial prefix hit is only possible from a
//      saved state; they get the dense spacing. SWA models get the sparse
//      spacing. Plain attention-only models can trim KV, so they get none
//      unless the config forces it.
//   2. PlanCheckpointPositions - which prompt positions to capture while
//      prefilling `[prefill_start, n_prompt)`. Follows llama.cpp v1.0.0
//      `tools/server` (`--ctx-checkpoints`, `--checkpoint-min-step`: bounded
//      count, minimum spacing, newest kept) while planning the prompt up front.
//   3. CheckpointHostTier      - a strictly bounded host-RAM LRU tier for
//      evicted prefix checkpoints (`cache_ram_mb`, default 0 = off). Eviction
//      runs BEFORE the new entry is stored, so the byte cap is never exceeded,
//      not even transiently.
//   4. PrefixCheckpointStats   - hit / miss / eviction counters and stable
//      telemetry field/detail formatting.
//
// Not thread-safe: owned by the backend's single worker loop, like the other
// per-slot checkpoint state; LlamaCppBackend mirrors counters into atomics for
// lock-free cross-thread telemetry inspection.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/model_architecture.hpp"

namespace sonder::backends::llamacpp {

// ---------------------------------------------------------------------------
// Mode and configuration
// ---------------------------------------------------------------------------

enum class CheckpointMode {
    kOff = 0,     // KV can be trimmed in place; checkpoints add nothing.
    kSparse = 1,  // SWA attention: checkpoint at the normal spacing.
    kDense = 2,   // hybrid / recurrent: checkpoint at the dense spacing.
};

[[nodiscard]] std::string_view ToString(CheckpointMode mode) noexcept;

struct PrefixCheckpointConfig {
    bool enabled = true;
    // Spacing between checkpoints, in tokens (1-2K default).
    std::uint32_t spacing_tokens = 2048;        // kSparse
    std::uint32_t dense_spacing_tokens = 1024;  // kDense (hybrid / recurrent)
    // Prompts shorter than this get no checkpoints.
    std::uint32_t min_prompt_tokens = 512;
    // A "tail" checkpoint is always planned at `n_prompt - tail_offset_tokens`
    // (the existing hybrid behaviour is 16): the next turn restores it and
    // re-decodes only the few trailing tokens.
    std::uint32_t tail_offset_tokens = 16;
    // Upper bound on checkpoints planned per request / retained per slot.
    std::uint32_t max_per_request = 8;
    // Force checkpoints on plain attention-only models (testing / benchmarks).
    bool force_attention_only = false;
    // Bounded host-RAM tier budget in MiB for evicted prefixes (0 = off).
    std::uint32_t cache_ram_mb = 0;
    // Hard cap on host-tier entries, independent of bytes.
    std::uint32_t host_max_entries = 64;

    static constexpr std::uint32_t kMinSpacingTokens = 64;
    static constexpr std::uint32_t kMaxSpacingTokens = 1u << 20;
    static constexpr std::uint32_t kMaxPerRequestLimit = 64;
    static constexpr std::uint32_t kMaxHostCacheMb = 65536;
    static constexpr std::uint32_t kMaxHostEntries = 4096;
};

// Clamps every field into its valid range (never throws, never returns an
// unusable config: a 0 spacing becomes the default, not "every token").
[[nodiscard]] PrefixCheckpointConfig SanitizeConfig(PrefixCheckpointConfig cfg) noexcept;

// ---------------------------------------------------------------------------
// Hybrid detection and mode selection
// ---------------------------------------------------------------------------

// True when `general.architecture` names a hybrid or recurrent architecture in
// llama.cpp v1.0.0 (`llm_arch_is_hybrid` / `llm_arch_is_recurrent` in
// src/llama-arch.cpp). Case-insensitive.
[[nodiscard]] bool IsHybridArchitectureName(std::string_view name) noexcept;
[[nodiscard]] bool IsRecurrentArchitectureName(std::string_view name) noexcept;
[[nodiscard]] bool IsStatefulArchitectureName(std::string_view name) noexcept;

// Detects a hybrid / recurrent model from GGUF metadata: the shared
// `classify_model_architecture` result, plus the v1.0.0 architecture-name
// list for models whose metadata keys carry no recurrent hint.
[[nodiscard]] sonder::inference::ModelArchitecture DetectCheckpointArchitecture(
    const sonder::inference::ModelMetadata& metadata);

// `runtime_stateful` is the answer of `llama_model_is_hybrid(model) ||
// llama_model_is_recurrent(model)` when the caller has a loaded model; it wins
// over metadata when set. `sliding_window` is true for SWA models; `swa_full`
// is true when the context retains full-context SWA KV (`cp.swa_full = true`,
// e.g. `SwaKvMode::Full`), in which case `llama_memory_seq_rm` rolls SWA back
// directly without checkpoints (matching llama.cpp v1.0.0
// `tools/server/server-context.cpp:1439,4425-4428`).
[[nodiscard]] CheckpointMode ClassifyCheckpointMode(const sonder::inference::ModelMetadata& metadata,
                                                    bool sliding_window, std::optional<bool> runtime_stateful,
                                                    const PrefixCheckpointConfig& cfg, bool swa_full = false);

[[nodiscard]] std::uint32_t SpacingFor(CheckpointMode mode, const PrefixCheckpointConfig& cfg) noexcept;

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

// Token positions (a checkpoint at position p holds the state after tokens
// [0, p)) to capture while prefilling `[prefill_start, n_prompt)`.
//
// Rules:
//   * empty when disabled, mode is kOff, or n_prompt < min_prompt_tokens;
//   * the tail position n_prompt - tail_offset_tokens is always planned when
//     it lies strictly after prefill_start;
//   * every multiple of the spacing strictly between prefill_start and the
//     tail is planned, except those closer than spacing/2 to the tail (they
//     would be near-duplicates of it);
//   * at most max_per_request positions, tail first, then the newest spaced
//     ones;
//   * ascending, unique, every value in (prefill_start, n_prompt).
[[nodiscard]] std::vector<std::size_t> PlanCheckpointPositions(std::size_t prefill_start, std::size_t n_prompt,
                                                               CheckpointMode mode,
                                                               const PrefixCheckpointConfig& cfg);

// Prefill chunk cursor helpers shared by single-request and continuous-batch
// prefill loops. `cursor` automatically advances past any planned positions
// <= `off`, so switching between batching modes can never wedge the cursor.
[[nodiscard]] std::size_t ClampPrefillStepToCheckpoint(std::span<const std::size_t> planned, std::size_t& cursor,
                                                       std::size_t off, std::size_t step_end) noexcept;
[[nodiscard]] bool ShouldCaptureAtOffset(std::span<const std::size_t> planned, std::size_t& cursor,
                                         std::size_t new_offset) noexcept;

// ---------------------------------------------------------------------------
// Shared host-RAM budget & L2+checkpoint restore selection
// ---------------------------------------------------------------------------

// Partitions one `cache_ram_mb` budget across the L2 Radix full-state cache
// and the partial-checkpoint host tier so `radix_bytes + checkpoint_bytes ==
// total_bytes` always holds (never double-allocates `cache_ram_mb`).
struct HostCacheBudgetSplit {
    std::size_t total_bytes = 0;
    std::size_t radix_bytes = 0;
    std::size_t checkpoint_bytes = 0;
};

[[nodiscard]] HostCacheBudgetSplit SplitHostCacheBudget(std::uint32_t cache_ram_mb, CheckpointMode mode) noexcept;

struct L2CheckpointRestorePlan {
    bool use_exact_l2 = false;      // restore an exact-length L2 snapshot directly
    bool use_trimmed_l2 = false;    // restore a longer L2 snapshot and trim to target_tokens
    bool apply_host_ckpt = false;   // overlay PARTIAL_ONLY host checkpoint before seq_rm
    std::size_t target_tokens = 0;  // final restored token count (0 = do not restore)
};

// Pure decision table for restoring an evicted prefix from L2 when VRAM does
// not hold it: picks the longest valid restore >= min_beat_lcp + min_gain_tokens
// between an exact L2 snapshot and a trimmed L2 snapshot (direct trim on
// attention-only, or via a host-tier PARTIAL_ONLY checkpoint on hybrid /
// recurrent / SWA models). Ties prefer the exact L2 snapshot.
[[nodiscard]] L2CheckpointRestorePlan ChooseL2RestorePlan(bool direct_trim_allowed, std::size_t min_beat_lcp,
                                                          std::size_t min_gain_tokens,
                                                          std::size_t trimmed_matched_tokens,
                                                          std::size_t trimmed_state_tokens,
                                                          std::size_t host_ckpt_tokens,
                                                          std::size_t exact_matched_tokens) noexcept;

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

enum class CheckpointTier { kSlot = 0, kHost = 1 };

struct PrefixCheckpointStats {
    // Placement / capture.
    std::uint64_t checkpoints_planned = 0;
    std::uint64_t checkpoints_captured = 0;
    std::uint64_t captured_bytes = 0;
    // Lookups. A "hit" restored a checkpoint; a "miss" fell back to a cold
    // prefill. restored_tokens sums the prompt tokens skipped by hits.
    std::uint64_t slot_hits = 0;
    std::uint64_t host_hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t restored_tokens = 0;
    // Host tier.
    std::uint64_t host_puts = 0;
    std::uint64_t host_evictions = 0;
    std::uint64_t host_rejected = 0;  // too large for the budget, or tier off
    std::uint64_t host_bytes = 0;     // gauge
    std::uint64_t host_entries = 0;   // gauge

    void OnPlanned(std::size_t n) noexcept;
    void OnCaptured(std::size_t bytes) noexcept;
    void OnHit(CheckpointTier tier, std::size_t restored) noexcept;
    void OnMiss() noexcept;

    [[nodiscard]] std::uint64_t Lookups() const noexcept;
    // hits / lookups in [0, 1]; 0 when there were no lookups.
    [[nodiscard]] double HitRate() const noexcept;
    [[nodiscard]] PrefixCheckpointCounters ToCounters() const noexcept;
};

// Stable (name, value) list for telemetry. Names are `prefix_checkpoint.*`.
[[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> StatsToFields(const PrefixCheckpointStats& s);

// Compact single-line detail string for `TelemetryEvent::detail`.
[[nodiscard]] std::string FormatStatsDetail(std::string_view event, const PrefixCheckpointStats& s);

// ---------------------------------------------------------------------------
// Bounded host-RAM LRU tier
// ---------------------------------------------------------------------------

class CheckpointHostTier {
public:
    // Fixed per-entry accounting overhead (list node, bookkeeping).
    static constexpr std::size_t kEntryOverheadBytes = 128;

    struct Hit {
        std::size_t n_tokens = 0;  // checkpoint position
        const std::vector<std::uint8_t>* state = nullptr;
        const std::vector<float>* pending_h = nullptr;
        // Pointers stay valid until the next mutating call (Put / Clear /
        // Configure / EraseNotPrefixOf) on the tier.
    };

    CheckpointHostTier() = default;
    CheckpointHostTier(std::size_t max_bytes, std::size_t max_entries) { Configure(max_bytes, max_entries); }

    // max_bytes == 0 or max_entries == 0 turns the tier off and drops contents.
    void Configure(std::size_t max_bytes, std::size_t max_entries);
    // Clamps cache_ram_mb to kMaxHostCacheMb and converts MiB to bytes safely.
    void ConfigureMiB(std::uint32_t cache_ram_mb, std::size_t max_entries);
    void Clear() noexcept;

    [[nodiscard]] bool enabled() const noexcept { return max_bytes_ > 0 && max_entries_ > 0; }
    [[nodiscard]] std::size_t max_bytes() const noexcept { return max_bytes_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    // Stores the checkpoint covering `prefix` (the tokens [0, n)). An existing
    // entry for the identical prefix is replaced. LRU entries are evicted
    // first until the new entry fits; an entry that alone exceeds the budget,
    // an empty prefix/state, or a disabled tier is rejected (returns false)
    // WITHOUT evicting anything.
    bool Put(std::span<const Token> prefix, std::vector<std::uint8_t> state, std::vector<float> pending_h = {});

    // Longest stored checkpoint that is a token-for-token prefix of `prompt`
    // with n_tokens <= max_match_tokens. Marks it most recently used.
    [[nodiscard]] std::optional<Hit> Lookup(std::span<const Token> prompt, std::size_t max_match_tokens);

    // Drops every entry whose prefix is NOT a prefix of `keep` (used when the
    // slot's history diverges). Returns the number dropped.
    std::size_t EraseNotPrefixOf(std::span<const Token> keep) noexcept;

    // Copies the host counters into `out` (gauges and cumulative counts).
    void FillStats(PrefixCheckpointStats& out) const noexcept;

    // Bytes one entry is charged, saturating.
    [[nodiscard]] static std::size_t EntryBytes(std::size_t n_tokens, std::size_t state_bytes,
                                                std::size_t pending_h_floats) noexcept;

private:
    struct Entry {
        std::vector<Token> tokens;
        std::vector<std::uint8_t> state;
        std::vector<float> pending_h;
        std::size_t charged = 0;
    };

    void EvictBack() noexcept;

    std::size_t max_bytes_ = 0;
    std::size_t max_entries_ = 0;
    std::size_t bytes_ = 0;
    std::list<Entry> entries_;  // front = most recently used
    std::uint64_t puts_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t rejected_ = 0;
};

}  // namespace sonder::backends::llamacpp
