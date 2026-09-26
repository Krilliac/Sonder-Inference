// Sonder Inference: KV-cache module: logical block/page manager.
#pragma once

// Logical block/page KV-cache manager (docs/KV_CACHE.md, Phase 2).
//
// Pure policy/bookkeeping: the manager owns *logical* fixed-size blocks and
// per-sequence block tables. It never touches device memory. Backends map a
// BlockId onto their physical storage and execute the copy operations the
// manager reports (copy-on-write).

#include "sonder/inference/cache/eviction_policy.hpp"
#include "sonder/inference/cache/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonder::inference::cache {

struct KvCacheConfig {
    /// Tokens per logical block (vLLM-style page size).
    std::uint32_t block_size_tokens = 16;
    /// Size of the block pool.
    std::uint32_t num_blocks = 0;
    /// Backend-reported KV bytes per token (all layers/heads). Accounting only.
    std::uint64_t bytes_per_token = 0;
    /// Keep full blocks addressable by prefix hash after their sequences end.
    bool enable_prefix_caching = true;
    /// Pinned-utilisation thresholds for PressureLevel::high / ::critical.
    double high_watermark = 0.80;
    double critical_watermark = 0.95;
};

struct SequenceOptions {
    CacheFingerprint fingerprint{};
    Priority priority = 0;
};

/// Result of append_tokens().
struct AppendResult {
    std::size_t tokens_appended = 0;
    /// Tokens satisfied by prefix-cache hits in this call (prefill can skip them).
    std::size_t tokens_reused = 0;
    std::size_t blocks_allocated = 0;
    std::size_t blocks_reused = 0;
    std::size_t copy_on_write = 0;
};

/// A logical copy the backend must perform before writing to `dst`
/// (copy-on-write of a shared, partially filled block).
struct BlockCopy {
    BlockId src = kInvalidBlock;
    BlockId dst = kInvalidBlock;
    std::uint32_t num_tokens = 0;
    friend bool operator==(const BlockCopy&, const BlockCopy&) = default;
};

/// Telemetry event (maps onto kv.allocated / kv.reused / kv.evicted /
/// kv.pressure families in docs/KV_CACHE.md). Emitted synchronously.
struct CacheEvent {
    enum class Kind : std::uint8_t { allocated, reused, evicted, copy_on_write, pressure_changed };
    Kind kind = Kind::allocated;
    SequenceId sequence = 0;  // 0 when not attributable (eviction)
    BlockId block = kInvalidBlock;
    BlockId source_block = kInvalidBlock;  // copy_on_write only
    PressureLevel pressure = PressureLevel::normal;  // pressure_changed only
};

using CacheEventListener = std::function<void(const CacheEvent&)>;

struct KvCacheStats {
    // Gauges
    std::size_t total_blocks = 0;
    std::size_t free_blocks = 0;     // never used / fully released
    std::size_t cached_blocks = 0;   // refcount 0, reusable via prefix index, evictable
    std::size_t pinned_blocks = 0;   // refcount > 0
    std::size_t shared_blocks = 0;   // refcount > 1
    std::size_t sequences = 0;
    std::uint32_t block_size_tokens = 0;
    std::uint64_t bytes_per_block = 0;
    std::uint64_t pinned_bytes = 0;
    std::uint64_t cached_bytes = 0;
    double utilization = 0.0;        // pinned / total
    PressureLevel pressure = PressureLevel::normal;
    // Counters (monotonic)
    std::uint64_t blocks_allocated = 0;
    std::uint64_t prefix_hit_blocks = 0;
    std::uint64_t prefix_miss_lookups = 0;
    std::uint64_t avoided_prefill_tokens = 0;
    std::uint64_t evictions = 0;
    std::uint64_t copy_on_write = 0;
    std::uint64_t allocation_failures = 0;
};

class KvCacheManager {
public:
    /// A null policy selects PriorityAwareEvictionPolicy.
    explicit KvCacheManager(KvCacheConfig config, std::unique_ptr<EvictionPolicy> policy = nullptr);

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;
    KvCacheManager(KvCacheManager&&) = default;
    KvCacheManager& operator=(KvCacheManager&&) = default;
    ~KvCacheManager() = default;

    [[nodiscard]] const KvCacheConfig& config() const noexcept { return config_; }
    [[nodiscard]] const EvictionPolicy& eviction_policy() const noexcept { return *policy_; }

    // --- sequence lifecycle -------------------------------------------------
    Status add_sequence(SequenceId id, const SequenceOptions& options = {});
    /// Session fork: `child` references every block of `parent` (no copy).
    /// A shared partial tail block is copied lazily on first write.
    Status fork_sequence(SequenceId parent, SequenceId child,
                         std::optional<Priority> child_priority = std::nullopt);
    /// Append tokens, allocating blocks as needed. While a sequence's content
    /// so far has been fully served from cache, whole blocks are looked up in
    /// the prefix index first. All-or-nothing: on ErrorCode::unavailable nothing changes.
    Status append_tokens(SequenceId id, std::span<const TokenId> tokens,
                         AppendResult* result = nullptr);
    /// Roll back to `new_length` tokens (speculative rejection, edits).
    Status truncate(SequenceId id, std::size_t new_length);
    Status free_sequence(SequenceId id);
    Status set_priority(SequenceId id, Priority priority);

    // --- queries ---------------------------------------------------------------
    [[nodiscard]] bool has_sequence(SequenceId id) const noexcept;
    [[nodiscard]] std::size_t num_tokens(SequenceId id) const noexcept;
    /// Leading tokens of the sequence that were served from the prefix cache.
    [[nodiscard]] std::size_t cached_prefix_tokens(SequenceId id) const noexcept;
    [[nodiscard]] std::span<const BlockId> block_table(SequenceId id) const noexcept;
    [[nodiscard]] std::uint32_t ref_count(BlockId block) const noexcept;
    /// Logical token contents recorded for a block (verification/debugging;
    /// a shared partial block may hold more tokens than a given sequence uses).
    [[nodiscard]] std::span<const TokenId> block_tokens(BlockId block) const noexcept;

    /// Tokens of `tokens` (whole blocks) currently reusable from cache. Pure
    /// query for prefix-aware scheduling / admission; does not touch LRU state.
    [[nodiscard]] std::size_t match_prefix(const CacheFingerprint& fingerprint,
                                           std::span<const TokenId> tokens) const;
    /// Worst-case new blocks needed to append `n` tokens (ignores cache hits).
    [[nodiscard]] std::size_t blocks_needed(SequenceId id, std::size_t n) const noexcept;
    /// Blocks needed for a fresh sequence of `n` tokens.
    [[nodiscard]] std::size_t blocks_for_tokens(std::size_t n) const noexcept;
    /// Blocks obtainable right now (free + evictable).
    [[nodiscard]] std::size_t available_blocks() const noexcept;
    [[nodiscard]] bool can_append(SequenceId id, std::size_t n) const noexcept;

    [[nodiscard]] PressureLevel pressure() const noexcept;
    [[nodiscard]] KvCacheStats stats() const;

    /// Copy operations accumulated since the last call (backend must apply
    /// them before running the step that writes the destination blocks).
    [[nodiscard]] std::vector<BlockCopy> take_pending_copies();

    void set_event_listener(CacheEventListener listener) { listener_ = std::move(listener); }

    /// Drop every cached (refcount 0) block back to the free list.
    std::size_t clear_cached();

    /// Full invariant check (refcounts, pool partition, prefix index). Tests/debug.
    [[nodiscard]] bool validate() const;

private:
    struct Block {
        std::uint32_t ref_count = 0;
        bool has_hash = false;
        bool registered = false;
        bool evictable = false;
        std::uint64_t hash = 0;
        std::uint64_t parent_hash = 0;
        std::uint32_t depth = 0;
        Priority priority = 0;
        std::uint64_t last_use = 0;
        CacheFingerprint fingerprint{};
        std::vector<TokenId> tokens;
    };

    struct Sequence {
        SequenceOptions options;
        std::vector<BlockId> blocks;
        std::size_t num_tokens = 0;
        std::size_t cached_prefix = 0;
        bool prefix_streak = true;
    };

    [[nodiscard]] std::uint64_t chain_hash(std::uint64_t parent, const CacheFingerprint& fp,
                                           std::span<const TokenId> tokens) const noexcept;
    [[nodiscard]] std::uint64_t prev_hash(const Sequence& seq, std::size_t block_index) const noexcept;
    [[nodiscard]] std::optional<BlockId> lookup(std::uint64_t hash, std::uint64_t parent,
                                                const CacheFingerprint& fp,
                                                std::span<const TokenId> tokens) const;

    BlockId allocate_block(SequenceId owner);
    void acquire(BlockId b, const Sequence& seq);
    void release(BlockId b, Priority priority);
    void reset_block(BlockId b);
    void unregister(BlockId b);
    void finalize_block(Sequence& seq, std::size_t block_index);
    void ensure_tail_writable(SequenceId id, Sequence& seq, AppendResult& result);
    void emit(const CacheEvent& event) const;
    void update_pressure();

    KvCacheConfig config_;
    std::unique_ptr<EvictionPolicy> policy_;
    std::vector<Block> blocks_;
    std::vector<BlockId> free_list_;
    std::unordered_map<std::uint64_t, BlockId> prefix_index_;
    std::unordered_map<SequenceId, Sequence> sequences_;
    std::vector<BlockCopy> pending_copies_;
    CacheEventListener listener_;
    std::uint64_t clock_ = 0;
    PressureLevel last_pressure_ = PressureLevel::normal;
    KvCacheStats counters_{};
};

}  // namespace sonder::inference::cache
