// Sonder Inference: KV-cache module: LRU and priority-aware eviction.
#include "sonder/inference/cache/eviction_policy.hpp"

#include <limits>

namespace sonder::inference::cache {

void LruEvictionPolicy::insert(const EvictionCandidate& candidate) {
    erase(candidate.block);
    const auto key = std::make_pair(candidate.last_use, seq_++);
    order_.emplace(key, candidate.block);
    index_.emplace(candidate.block, key);
}

void LruEvictionPolicy::erase(BlockId block) {
    const auto it = index_.find(block);
    if (it == index_.end()) return;
    order_.erase(it->second);
    index_.erase(it);
}

std::optional<BlockId> LruEvictionPolicy::pop_victim() {
    if (order_.empty()) return std::nullopt;
    const auto it = order_.begin();
    const BlockId victim = it->second;
    order_.erase(it);
    index_.erase(victim);
    return victim;
}

void PriorityAwareEvictionPolicy::insert(const EvictionCandidate& candidate) {
    erase(candidate.block);
    const Key key{candidate.priority,
                  std::numeric_limits<std::uint32_t>::max() - candidate.depth,
                  candidate.recompute_tokens, candidate.last_use, seq_++};
    order_.emplace(key, candidate.block);
    index_.emplace(candidate.block, key);
}

void PriorityAwareEvictionPolicy::erase(BlockId block) {
    const auto it = index_.find(block);
    if (it == index_.end()) return;
    order_.erase(it->second);
    index_.erase(it);
}

std::optional<BlockId> PriorityAwareEvictionPolicy::pop_victim() {
    if (order_.empty()) return std::nullopt;
    const auto it = order_.begin();
    const BlockId victim = it->second;
    order_.erase(it);
    index_.erase(victim);
    return victim;
}

std::unique_ptr<EvictionPolicy> make_eviction_policy(EvictionPolicyKind kind) {
    switch (kind) {
        case EvictionPolicyKind::lru: return std::make_unique<LruEvictionPolicy>();
        case EvictionPolicyKind::priority_aware: return std::make_unique<PriorityAwareEvictionPolicy>();
    }
    return std::make_unique<PriorityAwareEvictionPolicy>();
}

}  // namespace sonder::inference::cache
