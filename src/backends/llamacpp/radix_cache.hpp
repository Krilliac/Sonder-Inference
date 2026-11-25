// Radix Prefix Tree + Bounded L2 Host-RAM KV/SSM State Cache.
//
// Indexes cached token prefixes in a compressed radix tree mapped to serialized
// llama.cpp sequence state blobs stored in a strictly bounded host-RAM budget
// (`cache_ram_mib`). Enforces global LRU eviction BEFORE allocation so host
// commit headroom never spikes past the configured cap.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
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
        return state_blob.size() + dft_state_blob.size() + mtp_pending_h.size() * sizeof(float);
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
public:
    explicit RadixPrefixCache(std::size_t max_bytes = 0) : max_bytes_(max_bytes), root_(std::make_unique<Node>()) {}

    void SetMaxBytes(std::size_t max_bytes) {
        max_bytes_ = max_bytes;
        if (max_bytes_ == 0) {
            Clear();
            return;
        }
        while (used_bytes_ > max_bytes_) {
            if (!EvictOldest()) break;
        }
    }

    void Clear() {
        lru_list_.clear();
        root_ = std::make_unique<Node>();
        used_bytes_ = 0;
        entry_count_ = 0;
    }

    [[nodiscard]] bool enabled() const noexcept { return max_bytes_ > 0; }
    [[nodiscard]] std::size_t max_bytes() const noexcept { return max_bytes_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t entry_count() const noexcept { return entry_count_; }
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }
    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::uint64_t misses() const noexcept { return misses_; }

    // Strict global LRU eviction BEFORE caller allocates a state buffer.
    // Returns false if `incoming_bytes` cannot fit within `max_bytes_`.
    bool ReserveBeforeAlloc(std::size_t incoming_bytes, std::span<const Token> overwrite_prefix = {}) {
        if (max_bytes_ == 0 || incoming_bytes == 0 || incoming_bytes > max_bytes_) {
            return false;
        }
        Node* existing = overwrite_prefix.empty() ? nullptr : FindExactNode(overwrite_prefix);
        const std::size_t reclaimable = (existing != nullptr && existing->has_entry) ? existing->entry.ByteSize() : 0;
        const std::size_t effective_used = used_bytes_ >= reclaimable ? (used_bytes_ - reclaimable) : 0;
        if (effective_used + incoming_bytes <= max_bytes_) {
            return true;
        }
        while (true) {
            const std::size_t cur_reclaim =
                (existing != nullptr && existing->has_entry) ? existing->entry.ByteSize() : 0;
            const std::size_t cur_eff = used_bytes_ >= cur_reclaim ? (used_bytes_ - cur_reclaim) : 0;
            if (cur_eff + incoming_bytes <= max_bytes_) {
                return true;
            }
            if (lru_list_.empty()) {
                return false;
            }
            // If the oldest entry is the target prefix itself, release it too.
            if (lru_list_.front() == existing) {
                existing = nullptr;
            }
            if (!EvictOldest()) {
                return false;
            }
        }
    }

    // Inserts or overwrites the state blob for `prefix`. Enforces strict global
    // LRU eviction before attaching the new entry.
    bool Insert(std::span<const Token> prefix, RadixCacheEntry entry) {
        const std::size_t new_bytes = entry.ByteSize();
        if (max_bytes_ == 0 || prefix.empty() || entry.state_blob.empty() || new_bytes > max_bytes_) {
            return false;
        }

        // First evict LRU entries BEFORE modifying/inserting if needed, accounting
        // for any existing entry at `prefix` that we are about to replace.
        if (!ReserveBeforeAlloc(new_bytes, prefix)) {
            return false;
        }

        Node* node = EnsureNode(prefix);
        if (node == nullptr) {
            return false;
        }

        if (node->has_entry) {
            const std::size_t old_bytes = node->entry.ByteSize();
            used_bytes_ = used_bytes_ >= old_bytes ? (used_bytes_ - old_bytes) : 0;
            lru_list_.erase(node->lru_it);
            node->entry = {};
            node->has_entry = false;
            --entry_count_;
        }

        while (used_bytes_ + new_bytes > max_bytes_) {
            if (!EvictOldest()) {
                PruneIfEmpty(node);
                return false;
            }
        }

        node->entry = std::move(entry);
        node->has_entry = true;
        lru_list_.push_back(node);
        node->lru_it = std::prev(lru_list_.end());
        used_bytes_ += new_bytes;
        ++entry_count_;
        return true;
    }

    // Finds the longest cached prefix matching `prompt`, capped at
    // `max_match_tokens` (typically `prompt.size() - 1`).
    // When `allow_prefix_trim` is true and a deeper descendant shares a longer
    // prefix than any exact ancestor node, returns that descendant so callers on
    // attention-only models can restore and trim the trailing KV positions.
    [[nodiscard]] RadixLookupResult LookupLongestPrefix(std::span<const Token> prompt,
                                                        std::size_t max_match_tokens = std::numeric_limits<std::size_t>::max(),
                                                        bool allow_prefix_trim = true) {
        RadixLookupResult result;
        if (max_bytes_ == 0 || entry_count_ == 0 || prompt.empty() || max_match_tokens == 0) {
            ++misses_;
            return result;
        }
        const std::size_t limit = std::min(prompt.size(), max_match_tokens);
        Node* cur = root_.get();
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
            if (cur->has_entry) {
                best_exact = cur;
            }
        }

        // If we stopped cleanly at an internal node `cur` with pos == limit and
        // `!cur->has_entry`, `cur` itself may have descendants that share all
        // `pos` tokens with `prompt`.
        if (partial_branch == nullptr && pos > 0 && (best_exact == nullptr || pos > best_exact->depth)) {
            partial_branch = cur;
            partial_matched = pos;
        }

        Node* chosen = best_exact;
        std::size_t matched_len = best_exact != nullptr ? best_exact->depth : 0;

        if (allow_prefix_trim && partial_branch != nullptr && partial_matched > matched_len) {
            Node* desc = FindShallowestCachedDescendant(partial_branch);
            if (desc != nullptr) {
                chosen = desc;
                matched_len = partial_matched;
            }
        }

        if (chosen == nullptr || !chosen->has_entry || matched_len == 0) {
            ++misses_;
            return result;
        }

        Touch(chosen);
        ++hits_;
        result.matched_tokens = matched_len;
        result.state_tokens = chosen->depth;
        result.entry = &chosen->entry;
        return result;
    }

    // Removes an exact cached prefix if present.
    bool Erase(std::span<const Token> prefix) {
        Node* node = FindExactNode(prefix);
        if (node == nullptr || !node->has_entry) {
            return false;
        }
        DetachEntry(node);
        PruneIfEmpty(node);
        return true;
    }

private:
    struct Node {
        Node* parent = nullptr;
        std::vector<Token> edge;
        std::size_t depth = 0;
        std::unordered_map<Token, std::unique_ptr<Node>> children;
        bool has_entry = false;
        RadixCacheEntry entry;
        std::list<Node*>::iterator lru_it;
    };

    void Touch(Node* node) noexcept {
        if (node == nullptr || !node->has_entry) return;
        lru_list_.splice(lru_list_.end(), lru_list_, node->lru_it);
    }

    void DetachEntry(Node* node) noexcept {
        if (node == nullptr || !node->has_entry) return;
        const std::size_t bytes = node->entry.ByteSize();
        used_bytes_ = used_bytes_ >= bytes ? (used_bytes_ - bytes) : 0;
        lru_list_.erase(node->lru_it);
        node->entry = {};
        node->has_entry = false;
        if (entry_count_ > 0) --entry_count_;
    }

    bool EvictOldest() {
        if (lru_list_.empty()) return false;
        Node* victim = lru_list_.front();
        DetachEntry(victim);
        ++evictions_;
        PruneIfEmpty(victim);
        return true;
    }

    void PruneIfEmpty(Node* node) {
        while (node != nullptr && node != root_.get()) {
            if (node->has_entry) {
                return;
            }
            if (node->children.empty()) {
                Node* parent = node->parent;
                if (parent == nullptr) return;
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

    [[nodiscard]] Node* FindExactNode(std::span<const Token> prefix) const noexcept {
        if (prefix.empty()) return nullptr;
        Node* cur = root_.get();
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

    Node* EnsureNode(std::span<const Token> prefix) {
        Node* cur = root_.get();
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

    [[nodiscard]] static Node* FindShallowestCachedDescendant(Node* start) noexcept {
        if (start == nullptr) return nullptr;
        if (start->has_entry) return start;
        // Bounded iterative search for the shallowest cached descendant.
        std::vector<Node*> stack;
        stack.push_back(start);
        Node* best = nullptr;
        while (!stack.empty()) {
            Node* n = stack.back();
            stack.pop_back();
            if (n->has_entry) {
                if (best == nullptr || n->depth < best->depth) {
                    best = n;
                }
                continue;
            }
            if (best != nullptr && n->depth >= best->depth) {
                continue;
            }
            for (const auto& [tok, child] : n->children) {
                (void)tok;
                stack.push_back(child.get());
            }
        }
        return best;
    }

    std::size_t max_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    std::size_t entry_count_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::unique_ptr<Node> root_;
    std::list<Node*> lru_list_;
};

}  // namespace sonder::backends::llamacpp
