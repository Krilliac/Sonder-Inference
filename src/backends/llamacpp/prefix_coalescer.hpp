// Cold Shared-Prefix Prefill Coalescer.
//
// Pure stdlib-only C++20 state machine shared by LlamaCppBackend's single-owner
// continuous-batching worker loop and unit tests.
//
// When multiple cold requests arrive simultaneously sharing a common token
// prefix >= `min_coalesce_prefix_tokens` (on attention-only models where
// `llama_memory_seq_cp` is valid; explicitly bypassed for hybrid/recurrent
// models without a valid checkpoint):
//   1. Elects one active prefilling request as the producer for `[0, shared_len)`.
//   2. Holds bounded sibling followers (`<= max_followers_per_producer`) from
//      redundantly prefilling `[0, shared_len)` up to `max_coalesce_wait_ms`.
//   3. Releases followers as soon as the producer publishes their shared span
//      (so they clone `[0, shared_len)` via `llama_memory_seq_cp` and prefill
//      only their divergent suffix `[shared_len, end)`).
//   4. Guarantees zero deadlock on producer cancellation/failure/eviction
//      (promoting the next follower or releasing all followers immediately),
//      follower timeout (harvesting any partial published prefix if available),
//      and follower cancellation (cleaning up group state with zero leaks).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/model_architecture.hpp"

namespace sonder::backends::llamacpp {

struct PrefixCoalescerConfig {
    bool enabled = true;
    sonder::inference::ModelArchitecture architecture =
        sonder::inference::ModelArchitecture::attention_only;
    // Hybrid/recurrent models cannot safely clone arbitrary KV prefixes via
    // `llama_memory_seq_cp` without a valid recurrent checkpoint; keep false
    // by default so hybrid/recurrent models cleanly bypass coalescing.
    bool has_valid_checkpoint = false;
    std::size_t min_coalesce_prefix_tokens = kDefaultMinCoalescePrefixTokens;
    std::size_t max_followers_per_producer = kDefaultMaxCoalesceFollowers;
    std::uint64_t max_coalesce_wait_ms = kDefaultMaxCoalesceWaitMs;
};

enum class CoalesceRole {
    kIndependent = 0,
    kProducer,
    kHeldFollower,
};

enum class CoalesceReleaseReason {
    kPrefixPublished = 0,
    kTimeout,
    kProducerFailed,
    kPromotedToProducer,
};

struct CoalesceAdmitResult {
    CoalesceRole role = CoalesceRole::kIndependent;
    std::optional<std::size_t> producer_slot;
    std::size_t target_shared_len = 0;
};

struct CoalesceReleaseAction {
    std::size_t follower_slot = 0;
    std::size_t producer_slot = 0;
    std::size_t clone_prefix_len = 0;  // > 0 => clone [0, clone_prefix_len) from producer_slot
    std::size_t tokens_saved = 0;      // newly avoided prefill tokens (clone_prefix_len - prior offset)
    CoalesceReleaseReason reason = CoalesceReleaseReason::kPrefixPublished;
};

class PrefixCoalescer {
public:
    explicit PrefixCoalescer(PrefixCoalescerConfig config = {}) noexcept : config_(config) {}

    void Configure(PrefixCoalescerConfig config) {
        config_ = config;
        slots_.clear();
        stats_ = {};
    }

    // Clears active slot/group tracking while preserving cumulative telemetry.
    void ResetActive() noexcept { slots_.clear(); }

    // Clears both active state and cumulative telemetry.
    void Reset() noexcept {
        slots_.clear();
        stats_ = {};
    }

    [[nodiscard]] const PrefixCoalescerConfig& config() const noexcept { return config_; }
    [[nodiscard]] PrefixCoalescerStats stats() const noexcept { return stats_; }

    [[nodiscard]] static bool IsArchitectureEligible(sonder::inference::ModelArchitecture architecture,
                                                     bool has_valid_checkpoint = false) noexcept {
        if (architecture == sonder::inference::ModelArchitecture::attention_only) {
            return true;
        }
        return has_valid_checkpoint;
    }

    [[nodiscard]] bool IsEnabled() const noexcept {
        return config_.enabled && IsArchitectureEligible(config_.architecture, config_.has_valid_checkpoint) &&
               config_.min_coalesce_prefix_tokens > 0 && config_.max_followers_per_producer > 0 &&
               config_.max_coalesce_wait_ms > 0;
    }

    // Registers a newly admitted prefilling slot. Either attaches it as a
    // bounded held follower of an existing active prefiller sharing
    // >= `min_coalesce_prefix_tokens` uncomputed tokens, or registers it as
    // an active independent prefiller that subsequent siblings may follow.
    [[nodiscard]] CoalesceAdmitResult OnAdmit(std::size_t slot_index, std::span<const Token> prompt,
                                              std::size_t prefill_offset, std::uint64_t now_ms) {
        if (!IsEnabled()) {
            return {};
        }
        if (slots_.find(slot_index) != slots_.end()) {
            (void)OnSlotRemoved(slot_index);
        }
        if (prompt.size() <= 1 || prefill_offset >= prompt.size()) {
            return {};
        }

        const std::size_t max_reuse = prompt.size() - 1;
        std::optional<std::size_t> best_producer;
        std::size_t best_shared_len = 0;
        std::size_t best_producer_offset = 0;
        std::size_t best_follower_count = std::numeric_limits<std::size_t>::max();

        for (const auto& [cand_idx, cand] : slots_) {
            if (cand_idx == slot_index) continue;
            if (cand.role == CoalesceRole::kHeldFollower) continue;
            if (cand.prefill_offset >= cand.prompt.size()) continue;
            if (cand.followers.size() >= config_.max_followers_per_producer) continue;

            const std::size_t shared_len = std::min(LongestCommonPrefix(cand.prompt, prompt), max_reuse);
            if (shared_len <= prefill_offset) continue;
            if (shared_len <= cand.prefill_offset) continue;
            const std::size_t gain = shared_len - prefill_offset;
            if (gain < config_.min_coalesce_prefix_tokens) continue;

            const bool better =
                !best_producer.has_value() || shared_len > best_shared_len ||
                (shared_len == best_shared_len && cand.prefill_offset > best_producer_offset) ||
                (shared_len == best_shared_len && cand.prefill_offset == best_producer_offset &&
                 cand.followers.size() < best_follower_count) ||
                (shared_len == best_shared_len && cand.prefill_offset == best_producer_offset &&
                 cand.followers.size() == best_follower_count && cand_idx < *best_producer);
            if (better) {
                best_producer = cand_idx;
                best_shared_len = shared_len;
                best_producer_offset = cand.prefill_offset;
                best_follower_count = cand.followers.size();
            }
        }

        SlotEntry entry;
        entry.slot_index = slot_index;
        entry.prompt.assign(prompt.begin(), prompt.end());
        entry.prefill_offset = prefill_offset;

        if (best_producer.has_value()) {
            SlotEntry& prod = slots_.at(*best_producer);
            if (prod.followers.empty()) {
                prod.role = CoalesceRole::kProducer;
                ++stats_.coalesced_groups;
            }
            prod.followers.push_back(slot_index);

            entry.role = CoalesceRole::kHeldFollower;
            entry.producer_slot = *best_producer;
            entry.target_shared_len = best_shared_len;
            entry.wait_start_ms = now_ms;
            slots_.emplace(slot_index, std::move(entry));

            ++stats_.coalesced_followers;
            return CoalesceAdmitResult{CoalesceRole::kHeldFollower, *best_producer, best_shared_len};
        }

        entry.role = CoalesceRole::kIndependent;
        slots_.emplace(slot_index, std::move(entry));
        return CoalesceAdmitResult{CoalesceRole::kIndependent, std::nullopt, 0};
    }

    // Advances a producer's published KV prefix length after a successful
    // prefill step. Releases any followers whose `target_shared_len` has now
    // been reached so they can immediately clone `[0, target_shared_len)` via
    // `llama_memory_seq_cp` and prefill their divergent suffixes.
    [[nodiscard]] std::vector<CoalesceReleaseAction> OnProducerProgress(std::size_t producer_slot,
                                                                        std::size_t new_prefill_offset) {
        std::vector<CoalesceReleaseAction> actions;
        auto pit = slots_.find(producer_slot);
        if (pit == slots_.end()) {
            return actions;
        }
        SlotEntry& prod = pit->second;
        if (prod.role == CoalesceRole::kHeldFollower) {
            return actions;
        }

        const std::size_t clamped = std::min(new_prefill_offset, prod.prompt.size());
        if (clamped > prod.prefill_offset) {
            prod.prefill_offset = clamped;
        }

        if (!prod.followers.empty()) {
            std::vector<std::size_t> remaining;
            remaining.reserve(prod.followers.size());
            for (const std::size_t follower_idx : prod.followers) {
                auto fit = slots_.find(follower_idx);
                if (fit == slots_.end()) continue;
                SlotEntry& fol = fit->second;
                if (prod.prefill_offset >= fol.target_shared_len) {
                    const std::size_t target = fol.target_shared_len;
                    const std::size_t saved = target > fol.prefill_offset ? (target - fol.prefill_offset) : 0u;
                    fol.prefill_offset = target;
                    fol.role = CoalesceRole::kIndependent;
                    fol.producer_slot.reset();
                    fol.target_shared_len = 0;
                    stats_.coalesced_tokens_saved += saved;

                    actions.push_back(CoalesceReleaseAction{
                        follower_idx,
                        producer_slot,
                        target,
                        saved,
                        CoalesceReleaseReason::kPrefixPublished,
                    });
                } else {
                    remaining.push_back(follower_idx);
                }
            }
            prod.followers = std::move(remaining);
            if (prod.followers.empty()) {
                prod.role = CoalesceRole::kIndependent;
            }
        }

        // Once the producer finishes prefilling its prompt, every follower
        // (whose target_shared_len <= prod.prompt.size()) has been released,
        // and the full prompt is now warm in VRAM for direct VRAM cloning.
        if (prod.prefill_offset >= prod.prompt.size()) {
            slots_.erase(pit);
        }

        return actions;
    }

    // Releases any held followers that have waited >= `max_coalesce_wait_ms`.
    // If their live producer has already published a partial prefix past the
    // follower's current offset, the release action includes that partial
    // `clone_prefix_len` so work already completed by the producer is not lost.
    [[nodiscard]] std::vector<CoalesceReleaseAction> PollTimeouts(std::uint64_t now_ms) {
        std::vector<CoalesceReleaseAction> actions;
        if (!IsEnabled()) {
            return actions;
        }

        std::vector<std::size_t> expired;
        for (const auto& [slot_idx, entry] : slots_) {
            if (entry.role != CoalesceRole::kHeldFollower) continue;
            if (now_ms >= entry.wait_start_ms && (now_ms - entry.wait_start_ms) >= config_.max_coalesce_wait_ms) {
                expired.push_back(slot_idx);
            }
        }
        std::sort(expired.begin(), expired.end());

        actions.reserve(expired.size());
        for (const std::size_t follower_idx : expired) {
            auto fit = slots_.find(follower_idx);
            if (fit == slots_.end() || fit->second.role != CoalesceRole::kHeldFollower) continue;
            SlotEntry& fol = fit->second;
            const std::size_t prod_idx = fol.producer_slot.value_or(0);
            std::size_t partial_clone = 0;
            if (fol.producer_slot.has_value()) {
                auto pit = slots_.find(*fol.producer_slot);
                if (pit != slots_.end()) {
                    SlotEntry& prod = pit->second;
                    RemoveFollowerId(prod.followers, follower_idx);
                    if (prod.followers.empty()) {
                        prod.role = CoalesceRole::kIndependent;
                    }
                    const std::size_t available = std::min(prod.prefill_offset, fol.target_shared_len);
                    if (available > fol.prefill_offset) {
                        partial_clone = available;
                    }
                }
            }

            std::size_t saved = 0;
            if (partial_clone > fol.prefill_offset) {
                saved = partial_clone - fol.prefill_offset;
                fol.prefill_offset = partial_clone;
                stats_.coalesced_tokens_saved += saved;
            }
            fol.role = CoalesceRole::kIndependent;
            fol.producer_slot.reset();
            fol.target_shared_len = 0;
            ++stats_.coalesce_fallbacks_timeout;

            actions.push_back(CoalesceReleaseAction{
                follower_idx,
                prod_idx,
                partial_clone,
                saved,
                CoalesceReleaseReason::kTimeout,
            });
        }
        return actions;
    }

    // Removes a slot due to cancellation, error, eviction, or completion.
    // - If a held follower is removed, detaches cleanly from its producer with
    //   zero leaked group state.
    // - If a producer with waiting followers is removed, immediately promotes
    //   the oldest follower to take over prefilling (re-attaching eligible
    //   siblings while preserving their original wait_start_ms) or releases
    //   followers to independent prefill with zero deadlock.
    [[nodiscard]] std::vector<CoalesceReleaseAction> OnSlotRemoved(std::size_t slot_index) {
        std::vector<CoalesceReleaseAction> actions;
        auto it = slots_.find(slot_index);
        if (it == slots_.end()) {
            return actions;
        }

        if (it->second.role == CoalesceRole::kHeldFollower) {
            if (it->second.producer_slot.has_value()) {
                auto pit = slots_.find(*it->second.producer_slot);
                if (pit != slots_.end()) {
                    RemoveFollowerId(pit->second.followers, slot_index);
                    if (pit->second.followers.empty()) {
                        pit->second.role = CoalesceRole::kIndependent;
                    }
                }
            }
            slots_.erase(it);
            return actions;
        }

        std::vector<std::size_t> orphan_followers = std::move(it->second.followers);
        slots_.erase(it);
        if (orphan_followers.empty()) {
            return actions;
        }

        // Filter to still-present waiting followers in original FIFO order.
        std::vector<std::size_t> live_followers;
        live_followers.reserve(orphan_followers.size());
        for (const std::size_t f_idx : orphan_followers) {
            auto fit = slots_.find(f_idx);
            if (fit != slots_.end() && fit->second.role == CoalesceRole::kHeldFollower) {
                live_followers.push_back(f_idx);
            }
        }
        if (live_followers.empty()) {
            return actions;
        }

        // Promote the oldest follower so it immediately begins prefilling.
        const std::size_t promoted_idx = live_followers.front();
        SlotEntry& promoted = slots_.at(promoted_idx);
        promoted.role = CoalesceRole::kIndependent;
        promoted.producer_slot.reset();
        promoted.target_shared_len = 0;
        ++stats_.coalesce_fallbacks_producer_failed;

        actions.push_back(CoalesceReleaseAction{
            promoted_idx,
            slot_index,
            0,
            0,
            CoalesceReleaseReason::kProducerFailed,
        });

        // Re-attach remaining followers to the promoted producer when eligible;
        // otherwise release them immediately to independent prefill.
        const bool promoted_can_produce =
            IsEnabled() && promoted.prefill_offset < promoted.prompt.size() && promoted.prompt.size() > 1;
        for (std::size_t i = 1; i < live_followers.size(); ++i) {
            const std::size_t sib_idx = live_followers[i];
            SlotEntry& sib = slots_.at(sib_idx);

            bool reattached = false;
            if (promoted_can_produce && promoted.followers.size() < config_.max_followers_per_producer &&
                sib.prompt.size() > 1 && sib.prefill_offset < sib.prompt.size()) {
                const std::size_t max_reuse = sib.prompt.size() - 1;
                const std::size_t shared_len = std::min(LongestCommonPrefix(promoted.prompt, sib.prompt), max_reuse);
                if (shared_len > sib.prefill_offset && shared_len > promoted.prefill_offset &&
                    (shared_len - sib.prefill_offset) >= config_.min_coalesce_prefix_tokens) {
                    sib.producer_slot = promoted_idx;
                    sib.target_shared_len = shared_len;
                    sib.role = CoalesceRole::kHeldFollower;
                    // Preserve sib.wait_start_ms so timeout ceilings never reset across promotions.
                    promoted.followers.push_back(sib_idx);
                    reattached = true;
                }
            }

            if (!reattached) {
                sib.role = CoalesceRole::kIndependent;
                sib.producer_slot.reset();
                sib.target_shared_len = 0;
                ++stats_.coalesce_fallbacks_producer_failed;
                actions.push_back(CoalesceReleaseAction{
                    sib_idx,
                    slot_index,
                    0,
                    0,
                    CoalesceReleaseReason::kProducerFailed,
                });
            }
        }

        if (!promoted.followers.empty()) {
            promoted.role = CoalesceRole::kProducer;
            actions[0].reason = CoalesceReleaseReason::kPromotedToProducer;
        }

        return actions;
    }

    // Rolls back saved-token telemetry and tracked prefill offset if a runtime
    // `llama_memory_seq_cp` failed or a follower was cancelled before cloning.
    void OnCloneFailed(std::size_t slot_index, std::size_t fallback_prefill_offset,
                       std::size_t reverted_tokens_saved) noexcept {
        if (stats_.coalesced_tokens_saved >= reverted_tokens_saved) {
            stats_.coalesced_tokens_saved -= reverted_tokens_saved;
        } else {
            stats_.coalesced_tokens_saved = 0;
        }
        auto it = slots_.find(slot_index);
        if (it != slots_.end()) {
            it->second.prefill_offset = std::min(fallback_prefill_offset, it->second.prompt.size());
        }
    }

    // Emergency anti-deadlock release of all currently held followers.
    [[nodiscard]] std::vector<CoalesceReleaseAction> ReleaseAllFollowers() {
        std::vector<CoalesceReleaseAction> actions;
        std::vector<std::size_t> held;
        for (const auto& [slot_idx, entry] : slots_) {
            if (entry.role == CoalesceRole::kHeldFollower) {
                held.push_back(slot_idx);
            }
        }
        std::sort(held.begin(), held.end());
        for (const std::size_t f_idx : held) {
            SlotEntry& fol = slots_.at(f_idx);
            const std::size_t prod_idx = fol.producer_slot.value_or(0);
            if (fol.producer_slot.has_value()) {
                auto pit = slots_.find(*fol.producer_slot);
                if (pit != slots_.end()) {
                    RemoveFollowerId(pit->second.followers, f_idx);
                    if (pit->second.followers.empty()) {
                        pit->second.role = CoalesceRole::kIndependent;
                    }
                }
            }
            fol.role = CoalesceRole::kIndependent;
            fol.producer_slot.reset();
            fol.target_shared_len = 0;
            ++stats_.coalesce_fallbacks_producer_failed;
            actions.push_back(CoalesceReleaseAction{
                f_idx,
                prod_idx,
                0,
                0,
                CoalesceReleaseReason::kProducerFailed,
            });
        }
        return actions;
    }

    [[nodiscard]] bool IsFollowerHeld(std::size_t slot_index) const noexcept {
        const auto it = slots_.find(slot_index);
        return it != slots_.end() && it->second.role == CoalesceRole::kHeldFollower;
    }

    [[nodiscard]] CoalesceRole RoleOf(std::size_t slot_index) const noexcept {
        const auto it = slots_.find(slot_index);
        return it != slots_.end() ? it->second.role : CoalesceRole::kIndependent;
    }

    [[nodiscard]] std::optional<std::size_t> ProducerOf(std::size_t follower_slot) const noexcept {
        const auto it = slots_.find(follower_slot);
        if (it == slots_.end() || it->second.role != CoalesceRole::kHeldFollower) {
            return std::nullopt;
        }
        return it->second.producer_slot;
    }

    [[nodiscard]] std::size_t TargetSharedLen(std::size_t follower_slot) const noexcept {
        const auto it = slots_.find(follower_slot);
        if (it == slots_.end() || it->second.role != CoalesceRole::kHeldFollower) {
            return 0;
        }
        return it->second.target_shared_len;
    }

    // Returns the earliest `target_shared_len` awaited by any follower of
    // `producer_slot`, allowing chunked prefill to publish cleanly at a shared
    // boundary before spending compute on a long producer-only suffix.
    [[nodiscard]] std::optional<std::size_t> SuggestedProducerPublishBoundary(std::size_t producer_slot) const noexcept {
        const auto it = slots_.find(producer_slot);
        if (it == slots_.end() || it->second.followers.empty()) {
            return std::nullopt;
        }
        std::optional<std::size_t> earliest;
        for (const std::size_t f_idx : it->second.followers) {
            const auto fit = slots_.find(f_idx);
            if (fit == slots_.end()) continue;
            const std::size_t t = fit->second.target_shared_len;
            if (!earliest.has_value() || t < *earliest) {
                earliest = t;
            }
        }
        return earliest;
    }

    [[nodiscard]] std::vector<std::size_t> FollowersOf(std::size_t producer_slot) const {
        const auto it = slots_.find(producer_slot);
        if (it == slots_.end()) {
            return {};
        }
        return it->second.followers;
    }

    [[nodiscard]] std::size_t ActiveSlotCount() const noexcept { return slots_.size(); }

    [[nodiscard]] std::size_t ActiveGroupCount() const noexcept {
        std::size_t count = 0;
        for (const auto& [idx, entry] : slots_) {
            (void)idx;
            if (!entry.followers.empty()) ++count;
        }
        return count;
    }

    [[nodiscard]] std::size_t HeldFollowerCount() const noexcept {
        std::size_t count = 0;
        for (const auto& [idx, entry] : slots_) {
            (void)idx;
            if (entry.role == CoalesceRole::kHeldFollower) ++count;
        }
        return count;
    }

    // Verifies all structural invariants (acyclic star topology of depth <= 1,
    // follower bounds, bi-directional links, token-for-token prefix equality,
    // and strict progress-ahead-of-producer guarantees).
    [[nodiscard]] bool CheckInvariants() const noexcept {
        for (const auto& [slot_idx, entry] : slots_) {
            if (entry.slot_index != slot_idx) return false;
            if (entry.prompt.size() <= 1) return false;
            if (entry.prefill_offset >= entry.prompt.size()) return false;

            if (entry.role == CoalesceRole::kHeldFollower) {
                if (!entry.followers.empty()) return false;
                if (!entry.producer_slot.has_value()) return false;
                if (*entry.producer_slot == slot_idx) return false;
                const auto pit = slots_.find(*entry.producer_slot);
                if (pit == slots_.end()) return false;
                const SlotEntry& prod = pit->second;
                if (prod.role != CoalesceRole::kProducer) return false;
                if (std::find(prod.followers.begin(), prod.followers.end(), slot_idx) == prod.followers.end()) {
                    return false;
                }
                if (entry.target_shared_len == 0 || entry.target_shared_len >= entry.prompt.size()) return false;
                if (entry.target_shared_len > prod.prompt.size()) return false;
                if (entry.target_shared_len <= prod.prefill_offset) return false;
                if (entry.target_shared_len <= entry.prefill_offset) return false;
                if (LongestCommonPrefix(prod.prompt, entry.prompt) < entry.target_shared_len) return false;
            } else {
                if (entry.producer_slot.has_value()) return false;
                if (entry.target_shared_len != 0) return false;
                if (entry.role == CoalesceRole::kProducer && entry.followers.empty()) return false;
                if (entry.role == CoalesceRole::kIndependent && !entry.followers.empty()) return false;
                if (entry.followers.size() > config_.max_followers_per_producer) return false;
                for (std::size_t i = 0; i < entry.followers.size(); ++i) {
                    const std::size_t f_idx = entry.followers[i];
                    if (f_idx == slot_idx) return false;
                    for (std::size_t j = i + 1; j < entry.followers.size(); ++j) {
                        if (entry.followers[j] == f_idx) return false;
                    }
                    const auto fit = slots_.find(f_idx);
                    if (fit == slots_.end()) return false;
                    if (fit->second.role != CoalesceRole::kHeldFollower) return false;
                    if (!fit->second.producer_slot.has_value() || *fit->second.producer_slot != slot_idx) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

private:
    struct SlotEntry {
        std::size_t slot_index = 0;
        std::vector<Token> prompt;
        std::size_t prefill_offset = 0;
        CoalesceRole role = CoalesceRole::kIndependent;
        std::optional<std::size_t> producer_slot;
        std::size_t target_shared_len = 0;
        std::uint64_t wait_start_ms = 0;
        std::vector<std::size_t> followers;
    };

    static void RemoveFollowerId(std::vector<std::size_t>& followers, std::size_t slot_index) noexcept {
        followers.erase(std::remove(followers.begin(), followers.end(), slot_index), followers.end());
    }

    PrefixCoalescerConfig config_{};
    PrefixCoalescerStats stats_{};
    std::unordered_map<std::size_t, SlotEntry> slots_;
};

}  // namespace sonder::backends::llamacpp
