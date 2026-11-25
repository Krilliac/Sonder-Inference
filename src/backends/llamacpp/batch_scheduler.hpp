// Continuous Batching + Chunked Prefill Scheduler State Machine.
//
// Pure stdlib-only C++20 planner shared by LlamaCppBackend's single-owner batch
// worker loop and unit tests. Handles:
//   - Free-slot prefix selection (session_key -> longest prefix -> LRU)
//   - Zero-overhead single-request fast-path eligibility check
//   - Lockstep co-packing of 1 decode token per decoding slot TOGETHER with
//     chunked prefill (<= n_ubatch tokens total) without head-of-line blocking.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

enum class BatchSlotPhase {
    kIdle = 0,
    kPrefill,
    kDecode,
};

struct BatchSlotPlanView {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    BatchSlotPhase phase = BatchSlotPhase::kIdle;
    std::size_t prompt_len = 0;
    std::size_t prefill_offset = 0;
    std::int64_t pos = 0;
    bool cancelled = false;
};

struct PlannedBatchItem {
    std::size_t slot_index = 0;
    std::int32_t seq_id = 0;
    bool is_prefill = false;
    std::size_t token_offset = 0;  // index into prompt for prefill; ignored for decode
    std::int32_t token_count = 0;  // 1 for decode, 1..budget for prefill chunk
    std::int64_t pos_start = 0;
    bool request_logits = false;   // true on decode or final prompt token of prefill
};

struct PlannedBatchStep {
    std::vector<PlannedBatchItem> items;
    std::int32_t total_tokens = 0;
    std::int32_t total_outputs = 0;
    std::size_t decode_slots = 0;
    std::size_t prefill_slots = 0;
};

// Selects the best FREE (`busy[i] == 0`) KV slot for `prompt` and `session_key`:
//   1. Exact non-empty `session_key` match among free slots
//   2. Longest common token prefix (> 0) among free slots, tie-broken by MRU
//   3. Least recently used free slot (lowest `last_used`)
// Returns std::nullopt when every slot is currently busy.
[[nodiscard]] inline std::optional<SlotSelection> SelectAvailablePrefixSlot(std::span<const PrefixSlotMeta> slots,
                                                                            std::span<const std::uint8_t> busy,
                                                                            std::span<const Token> prompt,
                                                                            std::string_view session_key) noexcept {
    if (slots.empty() || slots.size() != busy.size()) return std::nullopt;
    const std::size_t max_reuse = prompt.size() > 1 ? prompt.size() - 1 : 0;

    // 1. Exact non-empty session_key match among free slots.
    if (!session_key.empty()) {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (busy[i] != 0u) continue;
            if (slots[i].session_key == session_key) {
                const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
                return SlotSelection{i, lcp};
            }
        }
    }

    // 2. Longest common prefix (> 0) among free slots, tie-broken by MRU.
    bool found_lcp = false;
    std::size_t best_idx = 0;
    std::size_t best_lcp = 0;
    std::uint64_t best_used = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (busy[i] != 0u) continue;
        const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
        if (lcp > best_lcp || (lcp > 0 && lcp == best_lcp && slots[i].last_used > best_used)) {
            found_lcp = true;
            best_idx = i;
            best_lcp = lcp;
            best_used = slots[i].last_used;
        }
    }
    if (found_lcp) return SlotSelection{best_idx, best_lcp};

    // 3. Fallback to least recently used free slot.
    bool found_free = false;
    std::size_t lru_idx = 0;
    std::uint64_t min_used = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (busy[i] != 0u) continue;
        if (!found_free || slots[i].last_used < min_used) {
            found_free = true;
            lru_idx = i;
            min_used = slots[i].last_used;
        }
    }
    if (!found_free) return std::nullopt;
    return SlotSelection{lru_idx, 0};
}

// Preserves the zero-overhead single-request fast path (full speculative
// decoding + GPU backend sampling) whenever exactly one non-exclusive request
// is active and no other request is queued.
[[nodiscard]] inline bool CanUseSingleRequestFastPath(std::size_t active_count, std::size_t pending_count,
                                                      bool exclusive_request) noexcept {
    return active_count == 1u && pending_count == 0u && !exclusive_request;
}

// Plans ONE lockstep `llama_decode` step packing:
//   1. Up to 1 decode token per active `kDecode` slot (priority for low ITL)
//   2. Chunked prefill tokens from active `kPrefill` slots into the remaining
//      `token_budget` (<= n_ubatch), rotating fairly via `rr_cursor`.
// Enforces `total_tokens <= token_budget` and `total_outputs <= output_budget`.
[[nodiscard]] inline PlannedBatchStep PlanContinuousBatchStep(std::span<const BatchSlotPlanView> active_slots,
                                                              std::int32_t token_budget, std::int32_t output_budget,
                                                              std::size_t& rr_cursor) {
    PlannedBatchStep step;
    if (active_slots.empty() || token_budget <= 0 || output_budget <= 0) {
        return step;
    }

    const std::size_t n = active_slots.size();
    const std::size_t start_idx = rr_cursor % n;

    // Pass 1: Pack 1 decode token per decoding slot (low inter-token latency).
    // Keep decoding slots in stable canonical order so CUDA graphs see
    // consistent seq_id ordering across consecutive decode steps.
    for (std::size_t idx = 0; idx < n; ++idx) {
        if (step.total_tokens >= token_budget || step.total_outputs >= output_budget) {
            break;
        }
        const BatchSlotPlanView& s = active_slots[idx];
        if (s.cancelled || s.phase != BatchSlotPhase::kDecode) continue;
        PlannedBatchItem item;
        item.slot_index = s.slot_index;
        item.seq_id = s.seq_id;
        item.is_prefill = false;
        item.token_offset = 0;
        item.token_count = 1;
        item.pos_start = s.pos;
        item.request_logits = true;
        step.items.push_back(item);
        step.total_tokens += 1;
        step.total_outputs += 1;
        step.decode_slots += 1;
    }

    // Pass 2: Pack chunked prefill tokens into remaining token budget.
    std::int32_t rem_budget = token_budget - step.total_tokens;
    if (rem_budget > 0) {
        // Count eligible prefilling slots so multiple concurrent prefills share
        // the remaining micro-batch budget fairly instead of one giant prompt
        // starving another.
        std::size_t prefill_waiting = 0;
        for (const BatchSlotPlanView& s : active_slots) {
            if (!s.cancelled && s.phase == BatchSlotPhase::kPrefill && s.prefill_offset < s.prompt_len) {
                ++prefill_waiting;
            }
        }
        if (prefill_waiting > 0) {
            const std::int32_t fair_slice =
                std::max<std::int32_t>(1, rem_budget / static_cast<std::int32_t>(prefill_waiting));
            std::vector<std::int32_t> assigned(n, 0);
            std::vector<bool> need_logits(n, false);

            // First pass gives each prefilling slot up to `fair_slice`.
            for (std::size_t off = 0; off < n && rem_budget > 0; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (s.cancelled || s.phase != BatchSlotPhase::kPrefill || s.prefill_offset >= s.prompt_len) continue;
                const auto rem_prompt = static_cast<std::int64_t>(s.prompt_len - s.prefill_offset);
                std::int32_t take = static_cast<std::int32_t>(
                    std::min<std::int64_t>({rem_prompt, static_cast<std::int64_t>(fair_slice),
                                            static_cast<std::int64_t>(rem_budget)}));
                const bool finishes = (static_cast<std::int64_t>(take) == rem_prompt);
                if (finishes && step.total_outputs >= output_budget) {
                    if (take > 1) {
                        --take;
                    } else {
                        continue;
                    }
                }
                assigned[idx] = take;
                rem_budget -= take;
                if (static_cast<std::int64_t>(take) == rem_prompt) {
                    need_logits[idx] = true;
                    step.total_outputs += 1;
                }
            }

            // Second pass distributes any leftover budget so the micro-batch is
            // fully utilized (e.g. when one prefill was shorter than fair_slice).
            for (std::size_t off = 0; off < n && rem_budget > 0; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (s.cancelled || s.phase != BatchSlotPhase::kPrefill || s.prefill_offset >= s.prompt_len) continue;
                const auto rem_prompt = static_cast<std::int64_t>(s.prompt_len - s.prefill_offset) - assigned[idx];
                if (rem_prompt <= 0) continue;
                std::int32_t extra = static_cast<std::int32_t>(
                    std::min<std::int64_t>(rem_prompt, static_cast<std::int64_t>(rem_budget)));
                const bool finishes = (static_cast<std::int64_t>(extra) == rem_prompt);
                if (finishes && step.total_outputs >= output_budget) {
                    if (extra > 1) {
                        --extra;
                    } else {
                        continue;
                    }
                }
                assigned[idx] += extra;
                rem_budget -= extra;
                if (static_cast<std::int64_t>(extra) == rem_prompt) {
                    need_logits[idx] = true;
                    step.total_outputs += 1;
                }
            }

            for (std::size_t off = 0; off < n; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                if (assigned[idx] <= 0) continue;
                const BatchSlotPlanView& s = active_slots[idx];
                PlannedBatchItem item;
                item.slot_index = s.slot_index;
                item.seq_id = s.seq_id;
                item.is_prefill = true;
                item.token_offset = s.prefill_offset;
                item.token_count = assigned[idx];
                item.pos_start = s.pos;
                item.request_logits = need_logits[idx];
                step.items.push_back(item);
                step.total_tokens += assigned[idx];
                step.prefill_slots += 1;
            }
        }
    }

    rr_cursor = (start_idx + 1u) % n;
    return step;
}

}  // namespace sonder::backends::llamacpp
