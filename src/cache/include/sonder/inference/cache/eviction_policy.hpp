// Sonder Inference: KV-cache module: pluggable eviction policies.
#pragma once

#include "sonder/inference/cache/types.hpp"

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace sonder::inference::cache {

/// Metadata the manager hands to an eviction policy when a block becomes
/// evictable (reference count dropped to zero while it is still reusable
/// through the prefix index).
struct EvictionCandidate {
    BlockId block = kInvalidBlock;
    /// Monotonic logical clock value of the last use.
    std::uint64_t last_use = 0;
    /// Highest priority of any sequence that referenced the block.
    Priority priority = 0;
    /// Position of the block in its prefix chain (0 = first block). Deeper
    /// blocks are cheaper to lose: evicting a shallow block orphans the
    /// usefulness of every descendant.
    std::uint32_t depth = 0;
    /// Estimated cost (in tokens) to recompute this block and its missing
    /// ancestors if evicted.
    std::uint64_t recompute_tokens = 0;
};

/// Pluggable eviction policy. Only blocks with reference count zero are ever
/// offered; the manager never asks a policy to evict a pinned block.
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    /// Block became evictable.
    virtual void insert(const EvictionCandidate& candidate) = 0;
    /// Block is no longer evictable (reused, or removed by the manager).
    virtual void erase(BlockId block) = 0;
    /// Choose (and forget) the next victim, or nullopt when empty.
    [[nodiscard]] virtual std::optional<BlockId> pop_victim() = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
};

/// Classic least-recently-used eviction.
class LruEvictionPolicy final : public EvictionPolicy {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "lru"; }
    void insert(const EvictionCandidate& candidate) override;
    void erase(BlockId block) override;
    [[nodiscard]] std::optional<BlockId> pop_victim() override;
    [[nodiscard]] std::size_t size() const noexcept override { return index_.size(); }

private:
    // key: (last_use, insertion sequence) -> block
    std::map<std::pair<std::uint64_t, std::uint64_t>, BlockId> order_;
    std::unordered_map<BlockId, std::pair<std::uint64_t, std::uint64_t>> index_;
    std::uint64_t seq_ = 0;
};

/// Priority-aware eviction (docs/KV_CACHE.md "Avoid naive LRU as the only
/// policy once shared prefixes exist"). Victim order:
///   1. lowest priority first,
///   2. then deepest block in its prefix chain first (leaves before roots),
///   3. then cheapest recompute first,
///   4. then least recently used.
class PriorityAwareEvictionPolicy final : public EvictionPolicy {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "priority_lru"; }
    void insert(const EvictionCandidate& candidate) override;
    void erase(BlockId block) override;
    [[nodiscard]] std::optional<BlockId> pop_victim() override;
    [[nodiscard]] std::size_t size() const noexcept override { return index_.size(); }

private:
    // Sort key; smaller = evicted first. depth is negated via (max - depth).
    using Key = std::tuple<Priority, std::uint32_t, std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<Key, BlockId> order_;
    std::unordered_map<BlockId, Key> index_;
    std::uint64_t seq_ = 0;
};

enum class EvictionPolicyKind : std::uint8_t { lru, priority_aware };

[[nodiscard]] std::unique_ptr<EvictionPolicy> make_eviction_policy(EvictionPolicyKind kind);

}  // namespace sonder::inference::cache
