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
#include <array>
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
//   1. Exact non-empty `session_key` match among free slots (longest LCP, then MRU)
//   2. Longest common token prefix (> 0) among free slots, tie-broken by MRU
//   3. Least recently used free slot (lowest `last_used`)
// Returns std::nullopt when every slot is currently busy.
[[nodiscard]] inline std::optional<SlotSelection> SelectAvailablePrefixSlot(std::span<const PrefixSlotMeta> slots,
                                                                            std::span<const std::uint8_t> busy,
                                                                            std::span<const Token> prompt,
                                                                            std::string_view session_key) noexcept {
    if (slots.empty() || slots.size() != busy.size()) return std::nullopt;
    const std::size_t max_reuse = prompt.size() > 1 ? prompt.size() - 1 : 0;

    // 1. Exact non-empty session_key match among free slots (best LCP, then MRU).
    if (!session_key.empty()) {
        bool found_sess = false;
        std::size_t best_sess_idx = 0;
        std::size_t best_sess_lcp = 0;
        std::uint64_t best_sess_used = 0;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (busy[i] != 0u) continue;
            if (slots[i].session_key == session_key) {
                const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
                if (!found_sess || lcp > best_sess_lcp ||
                    (lcp == best_sess_lcp && slots[i].last_used > best_sess_used)) {
                    found_sess = true;
                    best_sess_idx = i;
                    best_sess_lcp = lcp;
                    best_sess_used = slots[i].last_used;
                }
            }
        }
        if (found_sess) return SlotSelection{best_sess_idx, best_sess_lcp};
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
//   1. Up to 1 decode token per active `kDecode` slot (priority for low ITL;
//      fairly round-robined if decode slots exceed min(token_budget, output_budget),
//      emitted in canonical ascending index order for CUDA graph stability)
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
    step.items.reserve(n);

    // Inline stack scratch avoids hot-path heap allocations for the normal
    // n <= kMaxSlots (16) continuous-batching case.
    std::array<std::int32_t, kMaxSlots> inline_assigned{};
    std::array<std::uint8_t, kMaxSlots> inline_flags{};
    std::vector<std::int32_t> heap_assigned;
    std::vector<std::uint8_t> heap_flags;
    std::int32_t* assigned = inline_assigned.data();
    std::uint8_t* flags = inline_flags.data();
    if (n > kMaxSlots) {
        heap_assigned.assign(n, 0);
        heap_flags.assign(n, 0u);
        assigned = heap_assigned.data();
        flags = heap_flags.data();
    }

    // Pass 1: Pack 1 decode token per decoding slot (low inter-token latency).
    std::size_t decode_waiting = 0;
    for (const BatchSlotPlanView& s : active_slots) {
        if (!s.cancelled && s.phase == BatchSlotPhase::kDecode) {
            ++decode_waiting;
        }
    }
    const auto decode_cap = static_cast<std::size_t>(std::min(token_budget, output_budget));
    if (decode_waiting > 0) {
        if (decode_waiting <= decode_cap) {
            // All decoding slots fit: mark in canonical order.
            for (std::size_t idx = 0; idx < n; ++idx) {
                const BatchSlotPlanView& s = active_slots[idx];
                if (!s.cancelled && s.phase == BatchSlotPhase::kDecode) {
                    flags[idx] = 1u;
                }
            }
        } else {
            // Tight budget: round-robin selection from start_idx so tail decode
            // slots are never starved, then emit in canonical ascending order.
            std::size_t picked = 0;
            for (std::size_t off = 0; off < n && picked < decode_cap; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (!s.cancelled && s.phase == BatchSlotPhase::kDecode) {
                    flags[idx] = 1u;
                    ++picked;
                }
            }
        }
        for (std::size_t idx = 0; idx < n; ++idx) {
            if (flags[idx] == 0u) continue;
            flags[idx] = 0u;
            const BatchSlotPlanView& s = active_slots[idx];
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
    }

    // Pass 2: Pack chunked prefill tokens into remaining token budget.
    std::int32_t rem_budget = token_budget - step.total_tokens;
    if (rem_budget > 0) {
        std::size_t prefill_waiting = 0;
        for (const BatchSlotPlanView& s : active_slots) {
            if (!s.cancelled && s.phase == BatchSlotPhase::kPrefill && s.prefill_offset < s.prompt_len) {
                ++prefill_waiting;
            }
        }
        if (prefill_waiting > 0) {
            const std::size_t rem_budget_sz = static_cast<std::size_t>(rem_budget);
            const std::size_t fair_slice_sz = std::max<std::size_t>(1u, rem_budget_sz / prefill_waiting);

            // First pass gives each prefilling slot up to `fair_slice_sz`.
            for (std::size_t off = 0; off < n && rem_budget > 0; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (s.cancelled || s.phase != BatchSlotPhase::kPrefill || s.prefill_offset >= s.prompt_len) continue;
                const std::size_t rem_prompt = s.prompt_len - s.prefill_offset;
                std::size_t take_sz = std::min({rem_prompt, fair_slice_sz, static_cast<std::size_t>(rem_budget)});
                const bool finishes = (take_sz == rem_prompt);
                if (finishes && step.total_outputs >= output_budget) {
                    if (take_sz > 1u) {
                        --take_sz;
                    } else {
                        continue;
                    }
                }
                const auto take = static_cast<std::int32_t>(take_sz);
                assigned[idx] = take;
                rem_budget -= take;
                if (take_sz == rem_prompt) {
                    flags[idx] = 1u;
                    step.total_outputs += 1;
                }
            }

            // Second pass distributes any leftover budget so the micro-batch is
            // fully utilized (e.g. when one prefill was shorter than fair_slice).
            for (std::size_t off = 0; off < n && rem_budget > 0; ++off) {
                const std::size_t idx = (start_idx + off) % n;
                const BatchSlotPlanView& s = active_slots[idx];
                if (s.cancelled || s.phase != BatchSlotPhase::kPrefill || s.prefill_offset >= s.prompt_len) continue;
                const std::size_t total_rem = s.prompt_len - s.prefill_offset;
                const auto already = static_cast<std::size_t>(assigned[idx]);
                if (total_rem <= already) continue;
                const std::size_t rem_prompt = total_rem - already;
                std::size_t extra_sz = std::min(rem_prompt, static_cast<std::size_t>(rem_budget));
                const bool finishes = (extra_sz == rem_prompt);
                if (finishes && step.total_outputs >= output_budget) {
                    if (extra_sz > 1u) {
                        --extra_sz;
                    } else {
                        continue;
                    }
                }
                const auto extra = static_cast<std::int32_t>(extra_sz);
                assigned[idx] += extra;
                rem_budget -= extra;
                if (extra_sz == rem_prompt) {
                    flags[idx] = 1u;
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
                item.request_logits = (flags[idx] != 0u);
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
