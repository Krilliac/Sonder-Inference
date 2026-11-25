// Radix Prefix Tree + Bounded L2 Host-RAM KV/SSM State Cache.
//
// Indexes cached token prefixes in a compressed radix tree mapped to serialized
// llama.cpp sequence state blobs stored in a strictly bounded host-RAM budget
// (`cache_ram_mib`). Enforces global LRU eviction BEFORE allocation so host
// commit headroom never spikes past the configured cap, supports RAII in-flight
// node pinning (`LookupAndPin` / `PinnedRadixLookup`), multi-branch shared
// prefix invalidation (`ErasePrefix`), and thread-safe leaf-locked access.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

struct RadixCacheEntry {
    std::vector<std::uint8_t> state_blob;
    std::vector<std::uint8_t> dft_state_blob;
    std::vector<float> mtp_pending_h;
    std::string session_key;

    [[nodiscard]] std::size_t ByteSize() const noexcept {
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        auto sat_add = [](std::size_t a, std::size_t b) noexcept -> std::size_t {
            return a > kMax - b ? kMax : a + b;
        };
        const std::size_t h_bytes =
            mtp_pending_h.size() > kMax / sizeof(float) ? kMax : mtp_pending_h.size() * sizeof(float);
        return sat_add(sat_add(state_blob.size(), dft_state_blob.size()), h_bytes);
    }
};

struct RadixLookupResult {
    // Number of prompt tokens matched (clamped to caller's max_match_tokens).
    std::size_t matched_tokens = 0;
    // Exact token length encoded in `entry->state_blob` (>= matched_tokens).
    // When `state_tokens > matched_tokens`, the caller must trim KV suffix
    // `[matched_tokens, -1)` after restoring `state_blob`.
    std::size_t state_tokens = 0;
    const RadixCacheEntry* entry = nullptr;

    [[nodiscard]] bool found() const noexcept { return entry != nullptr && matched_tokens > 0; }
};

class RadixPrefixCache {
private:
    struct Node;

    struct EntryRecord {
        RadixCacheEntry payload;
        std::size_t bytes = 0;
        std::size_t pin_count = 0;
        std::uint64_t lru_seq = 0;
        bool retired = false;
    };

    struct Node {
        Node* parent = nullptr;
        std::vector<Token> edge;
        std::size_t depth = 0;
        std::unordered_map<Token, std::unique_ptr<Node>> children;
        std::shared_ptr<EntryRecord> record;
        std::list<Node*>::iterator lru_it;

        [[nodiscard]] bool has_entry() const noexcept { return record != nullptr; }
    };

    struct CacheState {
        mutable std::mutex mu;
        std::size_t max_bytes = 0;
        std::size_t used_bytes = 0;
        std::size_t entry_count = 0;
        std::size_t active_pinned_count = 0;
        std::size_t active_pinned_bytes = 0;
        std::size_t retired_pinned_count = 0;
        std::size_t retired_pinned_bytes = 0;
        std::uint64_t lru_clock = 0;
        std::uint64_t evictions = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::unique_ptr<Node> root = std::make_unique<Node>();
        std::list<Node*> lru_list;
        std::vector<Node*> search_scratch;
    };

public:
    // Move-only RAII pin handle returned by `LookupAndPin`. Keeps the matched
    // entry's payload alive and protected from LRU eviction (and accounted
    // against `used_bytes()`) until destroyed or released.
    class PinnedLookup {
    public:
        PinnedLookup() = default;
        ~PinnedLookup() { Release(); }

        PinnedLookup(const PinnedLookup&) = delete;
        PinnedLookup& operator=(const PinnedLookup&) = delete;

        PinnedLookup(PinnedLookup&& other) noexcept
            : matched_tokens_(std::exchange(other.matched_tokens_, 0)),
              state_tokens_(std::exchange(other.state_tokens_, 0)),
              state_(std::move(other.state_)),
              record_(std::move(other.record_)) {}

        PinnedLookup& operator=(PinnedLookup&& other) noexcept {
            if (this != &other) {
                Release();
                matched_tokens_ = std::exchange(other.matched_tokens_, 0);
                state_tokens_ = std::exchange(other.state_tokens_, 0);
                state_ = std::move(other.state_);
                record_ = std::move(other.record_);
            }
            return *this;
        }

        [[nodiscard]] bool found() const noexcept { return record_ != nullptr && matched_tokens_ > 0; }
        [[nodiscard]] std::size_t matched_tokens() const noexcept { return matched_tokens_; }
        [[nodiscard]] std::size_t state_tokens() const noexcept { return state_tokens_; }
        [[nodiscard]] const RadixCacheEntry* entry() const noexcept {
            return record_ != nullptr ? &record_->payload : nullptr;
        }

        void Release() noexcept {
            if (!record_) {
                matched_tokens_ = 0;
                state_tokens_ = 0;
                state_.reset();
                return;
            }
            std::shared_ptr<EntryRecord> rec = std::move(record_);
            std::shared_ptr<CacheState> st = state_.lock();
            state_.reset();
            matched_tokens_ = 0;
            state_tokens_ = 0;
            std::vector<std::shared_ptr<EntryRecord>> to_free;
            if (st) {
                std::lock_guard<std::mutex> lock(st->mu);
                if (rec->pin_count > 0) {
                    --rec->pin_count;
                    if (rec->pin_count == 0) {
                        if (rec->retired) {
                            st->used_bytes = st->used_bytes >= rec->bytes ? (st->used_bytes - rec->bytes) : 0;
                            st->retired_pinned_bytes = st->retired_pinned_bytes >= rec->bytes
                                                           ? (st->retired_pinned_bytes - rec->bytes)
                                                           : 0;
                            if (st->retired_pinned_count > 0) --st->retired_pinned_count;
                        } else {
                            st->active_pinned_bytes = st->active_pinned_bytes >= rec->bytes
                                                          ? (st->active_pinned_bytes - rec->bytes)
                                                          : 0;
                            if (st->active_pinned_count > 0) --st->active_pinned_count;
                            // If SetMaxBytes() shrank the budget while this entry was pinned,
                            // enforce the cap as soon as the last pin releases.
                            try {
                                while (st->used_bytes > st->max_bytes) {
                                    if (!EvictOldestFromState(*st, to_free)) break;
                                }
                            } catch (...) {
                                // Destructor / Release() must remain noexcept even under OOM.
                            }
                        }
                    }
                }
            }
            // Destroy `rec` and `to_free` outside `st->mu` so freeing large state blobs never stalls waiters.
        }

    private:
        friend class RadixPrefixCache;

        PinnedLookup(std::size_t matched_tokens, std::size_t state_tokens, std::weak_ptr<CacheState> state,
                     std::shared_ptr<EntryRecord> record) noexcept
            : matched_tokens_(matched_tokens),
              state_tokens_(state_tokens),
              state_(std::move(state)),
              record_(std::move(record)) {}

        std::size_t matched_tokens_ = 0;
        std::size_t state_tokens_ = 0;
        std::weak_ptr<CacheState> state_;
        std::shared_ptr<EntryRecord> record_;
    };

    explicit RadixPrefixCache(std::size_t max_bytes = 0) : state_(std::make_shared<CacheState>()) {
        state_->max_bytes = max_bytes;
    }

    RadixPrefixCache(const RadixPrefixCache&) = delete;
    RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;
    RadixPrefixCache(RadixPrefixCache&&) noexcept = default;
    RadixPrefixCache& operator=(RadixPrefixCache&&) noexcept = default;

    void SetMaxBytes(std::size_t max_bytes) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->max_bytes = max_bytes;
            if (max_bytes == 0) {
                ClearLocked(to_free);
                return;
            }
            while (state_->used_bytes > state_->max_bytes) {
                if (!EvictOldestLocked(to_free)) break;
            }
        }
    }

    void Clear() {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            ClearLocked(to_free);
        }
    }

    [[nodiscard]] bool enabled() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->max_bytes > 0;
    }
    [[nodiscard]] std::size_t max_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->max_bytes;
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->used_bytes;
    }
    [[nodiscard]] std::size_t entry_count() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->entry_count;
    }
    [[nodiscard]] std::size_t pinned_entries() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->active_pinned_count + state_->retired_pinned_count;
    }
    [[nodiscard]] std::size_t pinned_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->active_pinned_bytes + state_->retired_pinned_bytes;
    }
    [[nodiscard]] std::size_t retired_pinned_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->retired_pinned_bytes;
    }
    [[nodiscard]] std::uint64_t evictions() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->evictions;
    }
    [[nodiscard]] std::uint64_t hits() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->hits;
    }
    [[nodiscard]] std::uint64_t misses() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->misses;
    }

    // Strict global LRU eviction BEFORE caller allocates a state buffer.
    // Returns false with ZERO evictions if `incoming_bytes` cannot fit within
    // `max_bytes_` (including when pinned in-flight entries occupy too much budget).
    bool ReserveBeforeAlloc(std::size_t incoming_bytes, std::span<const Token> overwrite_prefix = {}) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            ok = ReserveBeforeAllocLocked(incoming_bytes, overwrite_prefix, to_free);
        }
        return ok;
    }

    // Inserts or overwrites the state blob for `prefix`. Enforces strict global
    // LRU eviction before mutating the tree, and never evicts or invalidates
    // pinned in-flight entries (retiring an overwritten pinned entry in RAM
    // until its last pin is released, strictly within `max_bytes_`).
    bool Insert(std::span<const Token> prefix, RadixCacheEntry entry) {
        const std::size_t new_bytes = entry.ByteSize();
        if (prefix.empty() || entry.state_blob.empty() || new_bytes == 0) {
            return false;
        }
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            if (state_->max_bytes == 0 || new_bytes > state_->max_bytes) {
                return false;
            }
            // Evict unpinned entries BEFORE EnsureNode so EnsureNode can never be
            // invalidated by PruneIfEmpty during eviction.
            if (!ReserveBeforeAllocLocked(new_bytes, prefix, to_free)) {
                return false;
            }

            Node* node = EnsureNodeLocked(prefix);
            if (node == nullptr) {
                return false;
            }

            if (node->has_entry()) {
                if (node->record->pin_count == 0) {
                    DetachEntryLocked(node, to_free);
                } else {
                    RetireEntryLocked(node);
                }
            }

            auto rec = std::make_shared<EntryRecord>();
            rec->payload = std::move(entry);
            rec->bytes = new_bytes;
            rec->lru_seq = ++state_->lru_clock;

            node->record = std::move(rec);
            state_->lru_list.push_back(node);
            node->lru_it = std::prev(state_->lru_list.end());
            state_->used_bytes += new_bytes;
            ++state_->entry_count;
            return true;
        }
    }

    // Finds the longest cached prefix matching `prompt`, capped at
    // `max_match_tokens` (typically `prompt.size() - 1`).
    // When `allow_prefix_trim` is true and a deeper descendant shares a longer
    // prefix than any exact ancestor node, returns that descendant so callers on
    // attention-only models can restore and trim the trailing KV positions.
    // Note: Prefer `LookupAndPin` whenever the entry will be read across concurrent
    // cache mutations or subsequent `ReserveBeforeAlloc`/`Insert` calls.
    [[nodiscard]] RadixLookupResult LookupLongestPrefix(
        std::span<const Token> prompt, std::size_t max_match_tokens = std::numeric_limits<std::size_t>::max(),
        bool allow_prefix_trim = true) {
        RadixLookupResult result;
        std::lock_guard<std::mutex> lock(state_->mu);
        std::size_t matched_len = 0;
        Node* chosen = LookupNodeLocked(prompt, max_match_tokens, allow_prefix_trim, matched_len);
        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            ++state_->misses;
            return result;
        }
        TouchLocked(chosen);
        ++state_->hits;
        result.matched_tokens = matched_len;
        result.state_tokens = chosen->depth;
        result.entry = &chosen->record->payload;
        return result;
    }

    // Thread-safe & eviction-safe lookup returning an RAII pin handle that
    // prevents the matched entry from being evicted or freed while in flight.
    [[nodiscard]] PinnedLookup LookupAndPin(
        std::span<const Token> prompt, std::size_t max_match_tokens = std::numeric_limits<std::size_t>::max(),
        bool allow_prefix_trim = true) {
        std::lock_guard<std::mutex> lock(state_->mu);
        std::size_t matched_len = 0;
        Node* chosen = LookupNodeLocked(prompt, max_match_tokens, allow_prefix_trim, matched_len);
        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            ++state_->misses;
            return {};
        }
        TouchLocked(chosen);
        ++state_->hits;
        std::shared_ptr<EntryRecord> rec = chosen->record;
        if (rec->pin_count == 0) {
            ++state_->active_pinned_count;
            state_->active_pinned_bytes += rec->bytes;
        }
        ++rec->pin_count;
        return PinnedLookup(matched_len, chosen->depth, state_, std::move(rec));
    }

    // Removes an exact cached prefix if present. If currently pinned in flight,
    // retires it immediately from lookup visibility while keeping its payload
    // alive and counted in `used_bytes()` until the last pin releases.
    bool Erase(std::span<const Token> prefix) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        bool erased = false;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            Node* node = FindExactNodeLocked(prefix);
            if (node == nullptr || !node->has_entry()) {
                return false;
            }
            if (node->record->pin_count == 0) {
                DetachEntryLocked(node, to_free);
            } else {
                RetireEntryLocked(node);
            }
            PruneIfEmptyLocked(node);
            erased = true;
        }
        return erased;
    }

    // Multi-branch shared prefix invalidation: removes every cached entry whose
    // token sequence begins with `prefix` (including when `prefix` falls inside
    // a compressed radix edge). Returns the number of cached entries invalidated.
    // Empty `prefix` is a safe no-op returning 0 (use `Clear()` to wipe all).
    std::size_t ErasePrefix(std::span<const Token> prefix) {
        if (prefix.empty()) return 0;
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        std::size_t removed = 0;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            Node* cur = state_->root.get();
            std::size_t pos = 0;
            Node* target = nullptr;
            while (pos < prefix.size()) {
                const auto it = cur->children.find(prefix[pos]);
                if (it == cur->children.end()) {
                    return 0;
                }
                Node* child = it->second.get();
                const std::size_t edge_len = child->edge.size();
                const std::size_t rem = prefix.size() - pos;
                const std::size_t cmp_len = std::min(edge_len, rem);
                for (std::size_t k = 0; k < cmp_len; ++k) {
                    if (child->edge[k] != prefix[pos + k]) {
                        return 0;
                    }
                }
                if (rem <= edge_len) {
                    target = child;
                    break;
                }
                pos += edge_len;
                cur = child;
            }
            if (target == nullptr) {
                return 0;
            }

            // Iteratively detach/retire every cached entry in `target`'s subtree.
            auto& stack = state_->search_scratch;
            stack.clear();
            stack.push_back(target);
            while (!stack.empty()) {
                Node* n = stack.back();
                stack.pop_back();
                if (n->has_entry()) {
                    if (n->record->pin_count == 0) {
                        DetachEntryLocked(n, to_free);
                    } else {
                        RetireEntryLocked(n);
                    }
                    ++removed;
                }
                for (const auto& [tok, ch] : n->children) {
                    (void)tok;
                    stack.push_back(ch.get());
                }
            }
            stack.clear();

            Node* parent = target->parent;
            if (parent != nullptr && !target->edge.empty()) {
                const Token key = target->edge.front();
                parent->children.erase(key);
                PruneIfEmptyLocked(parent);
            }
        }
        return removed;
    }

    // Verifies all structural, compression, depth, LRU, byte-budget, and pin
    // accounting invariants. Intended for stress/property tests and audits.
    [[nodiscard]] bool CheckInvariants() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        if (!state_->root || state_->root->parent != nullptr || !state_->root->edge.empty() ||
            state_->root->depth != 0 || state_->root->has_entry()) {
            return false;
        }
        std::size_t counted_entries = 0;
        std::size_t counted_bytes = 0;
        std::size_t counted_active_pinned = 0;
        std::size_t counted_active_pinned_bytes = 0;

        std::vector<const Node*> stack;
        stack.push_back(state_->root.get());
        while (!stack.empty()) {
            const Node* n = stack.back();
            stack.pop_back();
            if (n != state_->root.get()) {
                if (n->parent == nullptr || n->edge.empty()) return false;
                if (n->depth != n->parent->depth + n->edge.size()) return false;
                if (!n->has_entry() && n->children.size() <= 1) return false;
                if (n->has_entry()) {
                    if (n->record->retired || n->record->bytes == 0 ||
                        n->record->bytes != n->record->payload.ByteSize()) {
                        return false;
                    }
                    if (*n->lru_it != n) return false;
                    ++counted_entries;
                    counted_bytes += n->record->bytes;
                    if (n->record->pin_count > 0) {
                        ++counted_active_pinned;
                        counted_active_pinned_bytes += n->record->bytes;
                    }
                }
            }
            for (const auto& [tok, ch] : n->children) {
                if (!ch || ch->parent != n || ch->edge.empty() || ch->edge.front() != tok) {
                    return false;
                }
                stack.push_back(ch.get());
            }
        }

        if (counted_entries != state_->entry_count) return false;
        if (counted_entries != state_->lru_list.size()) return false;
        if (counted_active_pinned != state_->active_pinned_count) return false;
        if (counted_active_pinned_bytes != state_->active_pinned_bytes) return false;
        if (counted_bytes + state_->retired_pinned_bytes != state_->used_bytes) return false;
        const std::size_t pinned_total = state_->active_pinned_bytes + state_->retired_pinned_bytes;
        if (state_->used_bytes > state_->max_bytes && state_->used_bytes > pinned_total) {
            return false;
        }
        for (const Node* n : state_->lru_list) {
            if (n == nullptr || !n->has_entry()) return false;
        }
        return true;
    }

private:
    void ClearLocked(std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        for (Node* n : state_->lru_list) {
            if (n == nullptr || !n->has_entry()) continue;
            if (n->record->pin_count == 0) {
                to_free.push_back(std::move(n->record));
            } else {
                n->record->retired = true;
                state_->retired_pinned_bytes += n->record->bytes;
                ++state_->retired_pinned_count;
                n->record.reset();
            }
        }
        state_->lru_list.clear();
        state_->root = std::make_unique<Node>();
        state_->entry_count = 0;
        state_->active_pinned_count = 0;
        state_->active_pinned_bytes = 0;
        state_->used_bytes = state_->retired_pinned_bytes;
    }

    void TouchLocked(Node* node) noexcept {
        if (node == nullptr || !node->has_entry()) return;
        node->record->lru_seq = ++state_->lru_clock;
        state_->lru_list.splice(state_->lru_list.end(), state_->lru_list, node->lru_it);
    }

    static void DetachEntryFromState(CacheState& st, Node* node,
                                     std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        if (node == nullptr || !node->has_entry()) return;
        const std::size_t bytes = node->record->bytes;
        st.used_bytes = st.used_bytes >= bytes ? (st.used_bytes - bytes) : 0;
        st.lru_list.erase(node->lru_it);
        to_free.push_back(std::move(node->record));
        if (st.entry_count > 0) --st.entry_count;
    }

    static void PruneIfEmptyFromState(CacheState& st, Node* node) {
        while (node != nullptr && node != st.root.get()) {
            if (node->has_entry()) {
                return;
            }
            if (node->children.empty()) {
                Node* parent = node->parent;
                if (parent == nullptr || node->edge.empty()) return;
                const Token first_tok = node->edge.front();
                parent->children.erase(first_tok);
                node = parent;
                continue;
            }
            // Compress single-child non-cached internal node back into its child.
            if (node->children.size() == 1) {
                Node* parent = node->parent;
                if (parent == nullptr) return;
                auto child_it = node->children.begin();
                std::unique_ptr<Node> child = std::move(child_it->second);
                std::vector<Token> merged;
                merged.reserve(node->edge.size() + child->edge.size());
                merged.insert(merged.end(), node->edge.begin(), node->edge.end());
                merged.insert(merged.end(), child->edge.begin(), child->edge.end());
                child->edge = std::move(merged);
                child->parent = parent;
                const Token parent_key = child->edge.front();
                parent->children[parent_key] = std::move(child);
                return;
            }
            return;
        }
    }

    static bool EvictOldestFromState(CacheState& st, std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        for (auto it = st.lru_list.begin(); it != st.lru_list.end(); ++it) {
            Node* victim = *it;
            if (victim != nullptr && victim->has_entry() && victim->record->pin_count == 0) {
                DetachEntryFromState(st, victim, to_free);
                ++st.evictions;
                PruneIfEmptyFromState(st, victim);
                return true;
            }
        }
        return false;
    }

    void DetachEntryLocked(Node* node, std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        DetachEntryFromState(*state_, node, to_free);
    }

    void RetireEntryLocked(Node* node) noexcept {
        if (node == nullptr || !node->has_entry()) return;
        const std::size_t bytes = node->record->bytes;
        node->record->retired = true;
        state_->lru_list.erase(node->lru_it);
        if (state_->entry_count > 0) --state_->entry_count;
        state_->active_pinned_bytes =
            state_->active_pinned_bytes >= bytes ? (state_->active_pinned_bytes - bytes) : 0;
        if (state_->active_pinned_count > 0) --state_->active_pinned_count;
        state_->retired_pinned_bytes += bytes;
        ++state_->retired_pinned_count;
        node->record.reset();
    }

    bool ReserveBeforeAllocLocked(std::size_t incoming_bytes, std::span<const Token> overwrite_prefix,
                                  std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        if (state_->max_bytes == 0 || incoming_bytes == 0 || incoming_bytes > state_->max_bytes) {
            return false;
        }
        const std::size_t budget_limit = state_->max_bytes - incoming_bytes;
        if (state_->used_bytes <= budget_limit) {
            return true;
        }
        // Fail fast BEFORE mutating or evicting anything if pinned entries alone
        // exceed the remaining budget.
        const std::size_t pinned_total = state_->active_pinned_bytes + state_->retired_pinned_bytes;
        if (pinned_total > budget_limit) {
            return false;
        }
        // Release any unpinned existing payload at `overwrite_prefix` FIRST so peak
        // RAM stays strictly <= max_bytes while the caller allocates the replacement.
        if (!overwrite_prefix.empty()) {
            if (Node* existing = FindExactNodeLocked(overwrite_prefix);
                existing != nullptr && existing->has_entry() && existing->record->pin_count == 0) {
                DetachEntryLocked(existing, to_free);
                PruneIfEmptyLocked(existing);
                if (state_->used_bytes <= budget_limit) {
                    return true;
                }
            }
        }
        while (state_->used_bytes > budget_limit) {
            if (!EvictOldestLocked(to_free)) {
                return false;
            }
        }
        return true;
    }

    bool EvictOldestLocked(std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        return EvictOldestFromState(*state_, to_free);
    }

    void PruneIfEmptyLocked(Node* node) { PruneIfEmptyFromState(*state_, node); }

    [[nodiscard]] Node* FindExactNodeLocked(std::span<const Token> prefix) const noexcept {
        if (prefix.empty()) return nullptr;
        Node* cur = state_->root.get();
        std::size_t pos = 0;
        while (pos < prefix.size()) {
            const auto it = cur->children.find(prefix[pos]);
            if (it == cur->children.end()) return nullptr;
            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            if (prefix.size() - pos < edge_len) return nullptr;
            for (std::size_t k = 0; k < edge_len; ++k) {
                if (child->edge[k] != prefix[pos + k]) return nullptr;
            }
            pos += edge_len;
            cur = child;
        }
        return cur;
    }

    Node* EnsureNodeLocked(std::span<const Token> prefix) {
        Node* cur = state_->root.get();
        std::size_t pos = 0;
        while (pos < prefix.size()) {
            const Token key = prefix[pos];
            auto it = cur->children.find(key);
            if (it == cur->children.end()) {
                auto leaf = std::make_unique<Node>();
                leaf->parent = cur;
                leaf->edge.assign(prefix.begin() + static_cast<std::ptrdiff_t>(pos), prefix.end());
                leaf->depth = prefix.size();
                Node* raw = leaf.get();
                cur->children.emplace(key, std::move(leaf));
                return raw;
            }

            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            const std::size_t rem = prefix.size() - pos;
            const std::size_t cmp_len = std::min(edge_len, rem);
            std::size_t k = 0;
            while (k < cmp_len && child->edge[k] == prefix[pos + k]) {
                ++k;
            }

            if (k == edge_len) {
                pos += edge_len;
                cur = child;
                continue;
            }

            // Split `child` at offset `k`: create internal node for `edge[0..k)`
            // and move `child` beneath it with `edge[k..)`.
            auto split = std::make_unique<Node>();
            split->parent = cur;
            split->edge.assign(child->edge.begin(), child->edge.begin() + static_cast<std::ptrdiff_t>(k));
            split->depth = cur->depth + k;

            std::unique_ptr<Node> existing_child = std::move(it->second);
            existing_child->edge.erase(existing_child->edge.begin(),
                                       existing_child->edge.begin() + static_cast<std::ptrdiff_t>(k));
            existing_child->parent = split.get();
            const Token existing_key = existing_child->edge.front();
            split->children.emplace(existing_key, std::move(existing_child));

            Node* split_raw = split.get();
            it->second = std::move(split);
            pos += k;
            cur = split_raw;
        }
        return cur;
    }

    [[nodiscard]] Node* LookupNodeLocked(std::span<const Token> prompt, std::size_t max_match_tokens,
                                         bool allow_prefix_trim, std::size_t& out_matched_len) {
        out_matched_len = 0;
        if (state_->max_bytes == 0 || state_->entry_count == 0 || prompt.empty() || max_match_tokens == 0) {
            return nullptr;
        }
        const std::size_t limit = std::min(prompt.size(), max_match_tokens);
        Node* cur = state_->root.get();
        Node* best_exact = nullptr;
        std::size_t pos = 0;

        Node* partial_branch = nullptr;
        std::size_t partial_matched = 0;

        while (pos < limit) {
            const auto it = cur->children.find(prompt[pos]);
            if (it == cur->children.end()) {
                break;
            }
            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            const std::size_t cmp_limit = std::min(edge_len, limit - pos);
            std::size_t k = 0;
            while (k < cmp_limit && child->edge[k] == prompt[pos + k]) {
                ++k;
            }
            if (k < edge_len) {
                if (k > 0) {
                    partial_branch = child;
                    partial_matched = pos + k;
                }
                break;
            }
            pos += edge_len;
            cur = child;
            if (cur->has_entry()) {
                best_exact = cur;
            }
        }

        // If we stopped cleanly at an internal node `cur` with pos > best_exact->depth,
        // `cur` itself may have descendants that share all `pos` tokens with `prompt`.
        if (partial_branch == nullptr && pos > 0 && (best_exact == nullptr || pos > best_exact->depth)) {
            partial_branch = cur;
            partial_matched = pos;
        }

        Node* chosen = best_exact;
        std::size_t matched_len = best_exact != nullptr ? best_exact->depth : 0;

        if (allow_prefix_trim && partial_branch != nullptr && partial_matched > matched_len) {
            Node* desc = FindShallowestCachedDescendantLocked(partial_branch);
            if (desc != nullptr) {
                chosen = desc;
                matched_len = partial_matched;
            }
        }

        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            return nullptr;
        }
        out_matched_len = matched_len;
        return chosen;
    }

    // Deterministic, allocation-light search for the best cached descendant:
    // smallest depth first, tie-broken by smallest byte size (cheapest restore),
    // then most recently used (`lru_seq`).
    [[nodiscard]] Node* FindShallowestCachedDescendantLocked(Node* start) {
        if (start == nullptr) return nullptr;
        if (start->has_entry()) return start;

        constexpr std::size_t kStackInlineCap = 64;
        std::array<Node*, kStackInlineCap> inline_stack{};
        std::size_t inline_top = 0;
        auto& overflow = state_->search_scratch;
        overflow.clear();

        auto push_node = [&](Node* p) {
            if (inline_top < kStackInlineCap) {
                inline_stack[inline_top++] = p;
            } else {
                overflow.push_back(p);
            }
        };
        auto pop_node = [&]() -> Node* {
            if (!overflow.empty()) {
                Node* p = overflow.back();
                overflow.pop_back();
                return p;
            }
            return inline_stack[--inline_top];
        };

        auto is_better = [](const Node* cand, const Node* cur_best) noexcept -> bool {
            if (cur_best == nullptr) return true;
            if (cand->depth != cur_best->depth) return cand->depth < cur_best->depth;
            if (cand->record->bytes != cur_best->record->bytes) return cand->record->bytes < cur_best->record->bytes;
            return cand->record->lru_seq > cur_best->record->lru_seq;
        };

        push_node(start);
        Node* best = nullptr;
        while (inline_top > 0 || !overflow.empty()) {
            Node* n = pop_node();
            if (n->has_entry()) {
                if (is_better(n, best)) {
                    best = n;
                }
                continue;
            }
            if (best != nullptr && n->depth >= best->depth) {
                continue;
            }
            for (const auto& [tok, child] : n->children) {
                (void)tok;
                push_node(child.get());
            }
        }
        overflow.clear();
        return best;
    }

    std::shared_ptr<CacheState> state_;
};

using PinnedRadixLookup = RadixPrefixCache::PinnedLookup;

}  // namespace sonder::backends::llamacpp
