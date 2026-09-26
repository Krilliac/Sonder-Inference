// Adapter: cache::KvCacheManager -> scheduler::KvCapacity.
//
// The scheduler admits work by reserving block *counts*; the cache allocates
// concrete blocks only when the engine appends tokens. This adapter keeps a
// reservation ledger per request and reports as free only the blocks the
// cache can still hand out (free + evictable) minus reservations that have
// not yet been materialised as blocks in the request's sequence.
//
// Conventions:
// - RequestId and cache SequenceId are the same value (the engine uses the
//   scheduler request id as the cache sequence id).
// - release() drops the reservation only. Sequence lifetime (add_sequence,
//   free_sequence, fork) stays with the engine, so KV can outlive a
//   reservation, e.g. for prefix reuse after completion.
// - An optional release hook lets the engine free the request's sequence
//   synchronously inside release(). The scheduler preempts victims one at a
//   time and re-reads free_blocks() after each release, so blocks must
//   become reusable immediately or it would preempt every running request.
// - Prefix-cache hits are not discounted; reservations are worst case.
//
// Only available when both modules are built.
#pragma once

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

#include <algorithm>
#include <functional>
#include <utility>
#include <unordered_map>

#include "sonder/inference/cache/kv_cache_manager.hpp"
#include "sonder/inference/scheduler/kv_capacity.hpp"

namespace sonder::inference {

class KvCacheCapacityAdapter final : public scheduler::KvCapacity {
public:
    explicit KvCacheCapacityAdapter(cache::KvCacheManager& manager) noexcept
        : manager_(manager) {}

    [[nodiscard]] scheduler::TokenCount block_size_tokens() const noexcept override {
        return static_cast<scheduler::TokenCount>(manager_.config().block_size_tokens);
    }
    [[nodiscard]] scheduler::BlockCount total_blocks() const noexcept override {
        return static_cast<scheduler::BlockCount>(manager_.config().num_blocks);
    }
    [[nodiscard]] scheduler::BlockCount free_blocks() const noexcept override {
        const auto available = static_cast<scheduler::BlockCount>(manager_.available_blocks());
        const scheduler::BlockCount pending = outstanding();
        return available > pending ? available - pending : 0;
    }
    [[nodiscard]] bool try_reserve(scheduler::RequestId id,
                                   scheduler::BlockCount blocks) override {
        if (blocks > free_blocks()) {
            return false;
        }
        held_[id] += blocks;
        return true;
    }
    void release(scheduler::RequestId id) override {
        held_.erase(id);
        if (release_hook_) {
            release_hook_(id);
        }
    }
    /// Called after a reservation is dropped (preemption, completion, failure,
    /// cancellation). Runs on the scheduler's caller thread.
    void set_release_hook(std::function<void(scheduler::RequestId)> hook) { release_hook_ = std::move(hook); }
    [[nodiscard]] scheduler::BlockCount blocks_held(
        scheduler::RequestId id) const noexcept override {
        const auto it = held_.find(id);
        return it == held_.end() ? 0 : it->second;
    }

    /// Reserved blocks not yet backed by blocks in the request's sequence.
    [[nodiscard]] scheduler::BlockCount outstanding() const noexcept {
        scheduler::BlockCount total = 0;
        for (const auto& [id, held] : held_) {
            const auto used = static_cast<scheduler::BlockCount>(
                manager_.block_table(static_cast<cache::SequenceId>(id)).size());
            total += held > used ? held - used : 0;
        }
        return total;
    }

private:
    cache::KvCacheManager& manager_;
    std::unordered_map<scheduler::RequestId, scheduler::BlockCount> held_;
    std::function<void(scheduler::RequestId)> release_hook_;
};

}  // namespace sonder::inference

#endif  // SONDER_HAS_KV_CACHE && SONDER_HAS_SCHEDULER
