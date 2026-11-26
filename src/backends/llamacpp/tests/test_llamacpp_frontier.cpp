// Frontier engine upgrade unit tests:
//   - Arctic-style SuffixMatcher (exact/variable-length/ring-eviction/hybrid MTP escalation)
//   - RadixPrefixCache (longest prefix match, LRU byte-budget eviction before alloc, split/overwrite)
//   - Continuous batching + chunked prefill scheduler state machine
//   - Option validation & plumbing (suffix, cache_ram_mib, moe_cache_mib, mmproj_path, media_blobs, logprobs)
//   - Exact numerically stable per-token logprob + top-K math
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "batch_scheduler.hpp"
#include "prefix_coalescer.hpp"
#include "radix_cache.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "suffix_matcher.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

TEST_CASE("llamacpp frontier: SuffixMatcher exact, variable-length, and confidence-scored drafting") {
    SuffixMatcher matcher(/*session_ring_capacity=*/4096, /*active_capacity=*/4096);

    // Prompt has both a short 3-token match [10, 11, 12] -> [99, 98] and a much
    // longer 7-token match [50, 51, 52, 53, 10, 11, 12] -> [20, 21, 22, 23, 24, 25].
    const std::vector<Token> prompt{
        1,  2,  10, 11, 12, 99, 98, 3,  4,  50, 51, 52, 53, 10, 11, 12,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 5,  6,
    };
    matcher.BeginRequest(prompt);
    CHECK(matcher.active_size() == prompt.size());

    // Query ending with the 7-token suffix picks the longer match over the 3-token match.
    const std::vector<Token> long_tail{50, 51, 52, 53, 10, 11, 12};
    const SuffixMatchResult long_res = matcher.Draft(long_tail, /*max_draft=*/16);
    CHECK(long_res.match_len == 7u);
    CHECK_FALSE(long_res.from_session_ring);
    CHECK(long_res.confidence > 0.80F);
    REQUIRE(long_res.draft.size() >= 6u);
    CHECK(long_res.draft[0] == 20);
    CHECK(long_res.draft[1] == 21);
    CHECK(long_res.draft[2] == 22);
    CHECK(long_res.draft[3] == 23);

    // Shorter 3-token suffix [99, 98, 3] has lower confidence than a 7-token match
    // and respects dynamic draft scaling + max_draft clamping.
    const std::vector<Token> short_tail{99, 98, 3};
    const SuffixMatchResult short_res = matcher.Draft(short_tail, /*max_draft=*/4);
    CHECK(short_res.match_len == 3u);
    CHECK(short_res.confidence > 0.0F);
    CHECK(short_res.confidence < long_res.confidence);
    REQUIRE(short_res.draft.size() == 4u);
    CHECK(short_res.draft == std::vector<Token>{4, 50, 51, 52});

    // Incremental CommitToken + DraftForCurrent works without matching the live tail.
    matcher.CommitToken(50);
    matcher.CommitToken(51);
    matcher.CommitToken(52);
    matcher.CommitToken(53);
    matcher.CommitToken(10);
    matcher.CommitToken(11);
    const SuffixMatchResult inc_res = matcher.DraftForCurrent(/*id_last=*/12, /*max_draft=*/10);
    CHECK(inc_res.match_len == 7u);
    REQUIRE(inc_res.draft.size() == 10u);
    CHECK(inc_res.draft[0] == 20);
    CHECK(inc_res.draft[9] == 29);

    // Repeated zero acceptance triggers brief adaptive cooldown, and NoteSkippedStep recovers.
    for (int i = 0; i < 6; ++i) {
        matcher.Observe(8, 0);
    }
    CHECK(matcher.paused());
    CHECK(matcher.Draft(long_tail, 8).draft.empty());
    for (int i = 0; i < 16 && matcher.paused(); ++i) {
        matcher.NoteSkippedStep();
    }
    CHECK_FALSE(matcher.paused());
    CHECK_FALSE(matcher.Draft(long_tail, 8).draft.empty());
}

TEST_CASE("llamacpp frontier: SuffixMatcher cross-request ring buffer, separators, and ring eviction") {
    // Small ring capacity (64 tokens) to deterministically exercise eviction/compaction.
    SuffixMatcher matcher(/*session_ring_capacity=*/64, /*active_capacity=*/256);

    // Request 1 records a distinct 20-token code pattern into the session ring.
    std::vector<Token> req1;
    for (Token t = 100; t < 120; ++t) req1.push_back(t);
    matcher.BeginRequest(req1);
    matcher.EndRequest();
    CHECK(matcher.session_ring_size() > 0u);

    // Request 2 starts with a different prompt, then hits a suffix from Request 1.
    matcher.BeginRequest(std::vector<Token>{500, 501, 502});
    const SuffixMatchResult cross = matcher.Draft(std::vector<Token>{100, 101, 102, 103}, /*max_draft=*/8);
    CHECK(cross.from_session_ring);
    CHECK(cross.match_len == 4u);
    REQUIRE(cross.draft.size() == 8u);
    CHECK(cross.draft == std::vector<Token>{104, 105, 106, 107, 108, 109, 110, 111});

    // Separator isolation: draft at end of req1 [116, 117, 118, 119] must NOT bleed
    // into the next request pushed into the ring.
    matcher.IngestSessionSequence(std::vector<Token>{200, 201, 202, 203, 204, 205, 206, 207});
    const SuffixMatchResult boundary = matcher.Draft(std::vector<Token>{116, 117, 118, 119}, /*max_draft=*/8);
    CHECK(boundary.draft.empty());

    // Push > 128 new tokens into the 64-token ring -> triggers compaction/eviction
    // of the old req1 tokens while keeping the newest sequence intact.
    for (int batch = 0; batch < 5; ++batch) {
        std::vector<Token> seq;
        for (int j = 0; j < 20; ++j) {
            seq.push_back(static_cast<Token>(1000 + batch * 100 + j));
        }
        matcher.IngestSessionSequence(seq);
    }
    CHECK(matcher.session_ring_compactions() >= 1u);
    CHECK(matcher.session_ring_size() <= matcher.session_ring_capacity());

    // Old evicted sequence from req1 is gone; newest sequence (batch 4: 1400..1419) matches.
    CHECK(matcher.Draft(std::vector<Token>{100, 101, 102, 103}, 8).draft.empty());
    const SuffixMatchResult recent = matcher.Draft(std::vector<Token>{1400, 1401, 1402, 1403}, /*max_draft=*/8);
    CHECK(recent.from_session_ring);
    REQUIRE(recent.draft.size() == 8u);
    CHECK(recent.draft[0] == 1404);
    CHECK(recent.draft[7] == 1411);
}

TEST_CASE("llamacpp frontier: SuffixMatcher hybrid draft-mtp escalation trigger") {
    SuffixMatcher matcher(/*session_ring_capacity=*/4096, /*active_capacity=*/4096);

    // Prompt containing a 5-token repeat (below threshold 6) and a 8-token verbatim
    // code block followed by 20 continuation tokens (triggers 6..16 burst).
    std::vector<Token> prompt{10, 11, 12, 13, 14, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1};
    for (Token t = 200; t < 228; ++t) prompt.push_back(t);  // 200..207 match + 208..227 continuation
    // Short tail match with only 4 continuation tokens (insufficient for 6-token burst).
    for (Token t = 300; t < 310; ++t) prompt.push_back(t);  // 300..305 (6) + 306..309 (4)
    matcher.BeginRequest(prompt);

    // 1. Match length 5 (< 6) must NOT escalate -> caller uses standard draft-mtp (k=3).
    const SuffixMatchResult too_short_match =
        matcher.TryHybridMtpBurst(std::vector<Token>{10, 11, 12, 13, 14}, /*max_k=*/16);
    CHECK(too_short_match.draft.empty());

    // 2. Match length 6 but only 4 continuation tokens (< 6) must NOT escalate.
    const SuffixMatchResult short_cont =
        matcher.TryHybridMtpBurst(std::vector<Token>{300, 301, 302, 303, 304, 305}, /*max_k=*/16);
    CHECK(short_cont.draft.empty());

    // 3. max_k < 6 must NOT escalate.
    const std::vector<Token> verbatim{200, 201, 202, 203, 204, 205, 206, 207};
    CHECK(matcher.TryHybridMtpBurst(verbatim, /*max_k=*/5).draft.empty());

    // 4. High-confidence exact match >= 6 with ample continuation escalates to 6..16 tokens!
    const SuffixMatchResult burst = matcher.TryHybridMtpBurst(verbatim, /*max_k=*/32);
    CHECK(burst.match_len == 8u);
    CHECK(burst.confidence >= SuffixMatcher::kHybridMtpMinConfidence);
    REQUIRE(burst.draft.size() == SuffixMatcher::kHybridMtpMaxBurstDraft);  // capped at 16
    for (std::size_t i = 0; i < burst.draft.size(); ++i) {
        CHECK(burst.draft[i] == static_cast<Token>(208 + i));
    }

    // 5. Respects tighter max_k in [6, 16].
    const SuffixMatchResult burst10 = matcher.TryHybridMtpBurst(verbatim, /*max_k=*/10);
    REQUIRE(burst10.draft.size() == 10u);
    CHECK(burst10.draft.back() == 217);
}

TEST_CASE("llamacpp frontier: RadixPrefixCache longest prefix, edge splitting, overwrite, and trim") {
    RadixPrefixCache cache(/*max_bytes=*/4096);
    CHECK(cache.enabled());
    CHECK(cache.used_bytes() == 0u);
    CHECK(cache.entry_count() == 0u);

    auto make_entry = [](std::size_t n_bytes, std::uint8_t fill, std::string key = {}) {
        RadixCacheEntry e;
        e.state_blob.assign(n_bytes, fill);
        e.session_key = std::move(key);
        return e;
    };

    const std::vector<Token> p_a{1, 2, 3, 4, 5, 6};
    const std::vector<Token> p_b{1, 2, 3, 7, 8, 9};
    const std::vector<Token> p_mid{1, 2, 3};

    REQUIRE(cache.Insert(p_a, make_entry(100, 0xA1, "a")));
    // Inserting p_b splits the compressed edge at [1, 2, 3].
    REQUIRE(cache.Insert(p_b, make_entry(120, 0xB2, "b")));
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.used_bytes() == 220u);

    // Exact ancestor lookup on both branches.
    {
        const std::vector<Token> q{1, 2, 3, 4, 5, 6, 99};
        const RadixLookupResult r = cache.LookupLongestPrefix(q, q.size() - 1, /*allow_prefix_trim=*/false);
        REQUIRE(r.found());
        CHECK(r.matched_tokens == 6u);
        CHECK(r.state_tokens == 6u);
        CHECK(r.entry->state_blob.front() == 0xA1);
    }
    {
        const std::vector<Token> q{1, 2, 3, 7, 8, 9, 42};
        const RadixLookupResult r = cache.LookupLongestPrefix(q, q.size() - 1, /*allow_prefix_trim=*/false);
        REQUIRE(r.found());
        CHECK(r.matched_tokens == 6u);
        CHECK(r.state_tokens == 6u);
        CHECK(r.entry->state_blob.front() == 0xB2);
    }

    // Inserting intermediate prefix [1, 2, 3] populates the split internal node cleanly.
    REQUIRE(cache.Insert(p_mid, make_entry(60, 0xC3, "mid")));
    CHECK(cache.entry_count() == 3u);
    CHECK(cache.used_bytes() == 280u);

    // Overwrite [1, 2, 3] updates bytes and payload without duplicating entries.
    REQUIRE(cache.Insert(p_mid, make_entry(80, 0xC4, "mid2")));
    CHECK(cache.entry_count() == 3u);
    CHECK(cache.used_bytes() == 300u);

    // Diverging mid-edge: [1, 2, 3, 4, 5, 77]:
    //   - allow_prefix_trim=false -> exact cached ancestor [1, 2, 3] (3 tokens)
    //   - allow_prefix_trim=true  -> descendant p_a with matched_tokens=5, state_tokens=6
    const std::vector<Token> q_mid{1, 2, 3, 4, 5, 77};
    {
        const RadixLookupResult exact_only = cache.LookupLongestPrefix(q_mid, q_mid.size() - 1, false);
        REQUIRE(exact_only.found());
        CHECK(exact_only.matched_tokens == 3u);
        CHECK(exact_only.state_tokens == 3u);
        CHECK(exact_only.entry->state_blob.front() == 0xC4);
    }
    {
        const RadixLookupResult trimmed = cache.LookupLongestPrefix(q_mid, q_mid.size() - 1, true);
        REQUIRE(trimmed.found());
        CHECK(trimmed.matched_tokens == 5u);
        CHECK(trimmed.state_tokens == 6u);
        CHECK(trimmed.entry->state_blob.front() == 0xA1);
    }

    // Respects max_match_tokens clamping (at least 1 token left for fresh logits).
    {
        const RadixLookupResult clamped = cache.LookupLongestPrefix(p_a, /*max_match_tokens=*/4, true);
        REQUIRE(clamped.found());
        CHECK(clamped.matched_tokens == 4u);
        CHECK(clamped.state_tokens == 6u);
    }

    // Erase removes entry and re-compresses without breaking sibling lookup.
    CHECK(cache.Erase(p_mid));
    CHECK_FALSE(cache.Erase(p_mid));
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.used_bytes() == 220u);
    CHECK(cache.LookupLongestPrefix(p_b, 6, false).found());
}

TEST_CASE("llamacpp frontier: RadixPrefixCache strict global LRU eviction BEFORE allocation") {
    RadixPrefixCache cache(/*max_bytes=*/100);

    auto make_entry = [](std::size_t n_bytes, std::uint8_t fill) {
        RadixCacheEntry e;
        e.state_blob.assign(n_bytes, fill);
        return e;
    };

    const std::vector<Token> p1{10, 11, 12};
    const std::vector<Token> p2{20, 21, 22};
    const std::vector<Token> p3{30, 31, 32};

    REQUIRE(cache.Insert(p1, make_entry(40, 0x01)));
    REQUIRE(cache.Insert(p2, make_entry(40, 0x02)));
    CHECK(cache.used_bytes() == 80u);
    CHECK(cache.evictions() == 0u);

    // Touch p1 so p2 becomes least-recently-used.
    REQUIRE(cache.LookupLongestPrefix(p1, 3, false).found());

    // Oversized reservation (> max_bytes) fails without evicting existing entries.
    CHECK_FALSE(cache.ReserveBeforeAlloc(120));
    CHECK(cache.used_bytes() == 80u);
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.evictions() == 0u);

    // ReserveBeforeAlloc(40) evicts LRU (p2) BEFORE the caller allocates a buffer!
    REQUIRE(cache.ReserveBeforeAlloc(40));
    CHECK(cache.used_bytes() == 40u);
    CHECK(cache.entry_count() == 1u);
    CHECK(cache.evictions() == 1u);
    CHECK_FALSE(cache.LookupLongestPrefix(p2, 3, false).found());
    CHECK(cache.LookupLongestPrefix(p1, 3, false).found());

    // Now inserting p3 (40B) fits directly within the 100B budget with zero further evictions.
    REQUIRE(cache.Insert(p3, make_entry(40, 0x03)));
    CHECK(cache.used_bytes() == 80u);
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.evictions() == 1u);

    // Disabled cache (0 bytes) rejects everything cleanly.
    RadixPrefixCache disabled(0);
    CHECK_FALSE(disabled.enabled());
    CHECK_FALSE(disabled.ReserveBeforeAlloc(16));
    CHECK_FALSE(disabled.Insert(p1, make_entry(16, 0x11)));
}

TEST_CASE("llamacpp frontier: continuous batching scheduler & chunked prefill state machine") {
    // 1. Free-slot prefix selection skips busy slots and returns nullopt when saturated.
    std::vector<PrefixSlotMeta> slots(3);
    slots[0] = {0, {1, 2, 3, 4, 5}, "sess-0", 10};
    slots[1] = {1, {1, 2, 3, 9}, "sess-1", 20};
    slots[2] = {2, {8, 9}, "", 5};
    std::vector<std::uint8_t> busy{1u, 0u, 0u};

    // Slot 0 matches session_key "sess-0", but it's busy -> picks free slot 1 by longest prefix!
    {
        const std::vector<Token> prompt{1, 2, 3, 4, 5, 6};
        const auto sel = SelectAvailablePrefixSlot(slots, busy, prompt, "sess-0");
        REQUIRE(sel.has_value());
        CHECK(sel->slot_index == 1u);
        CHECK(sel->matched_prefix == 3u);
    }
    // All slots busy -> returns nullopt cleanly.
    busy = {1u, 1u, 1u};
    CHECK_FALSE(SelectAvailablePrefixSlot(slots, busy, std::vector<Token>{1, 2, 3}, "").has_value());

    // 2. Single-request fast-path gate.
    CHECK(CanUseSingleRequestFastPath(1, 0, false));
    CHECK_FALSE(CanUseSingleRequestFastPath(0, 0, false));
    CHECK_FALSE(CanUseSingleRequestFastPath(2, 0, false));
    CHECK_FALSE(CanUseSingleRequestFastPath(1, 1, false));
    CHECK_FALSE(CanUseSingleRequestFastPath(1, 0, true));

    // 3. Mixed decode + chunked prefill co-packing within ubatch token budget.
    std::vector<BatchSlotPlanView> active{
        {0, 0, BatchSlotPhase::kDecode, 100, 100, 120, false},
        {1, 1, BatchSlotPhase::kPrefill, 1000, 0, 0, false},
        {2, 2, BatchSlotPhase::kDecode, 50, 50, 60, false},
    };
    std::size_t rr = 0;
    const PlannedBatchStep s1 = PlanContinuousBatchStep(active, /*token_budget=*/512, /*output_budget=*/4, rr);
    CHECK(s1.decode_slots == 2u);
    CHECK(s1.prefill_slots == 1u);
    CHECK(s1.total_tokens == 512);
    CHECK(s1.total_outputs == 2);  // intermediate prefill chunk needs 0 logits!
    REQUIRE(s1.items.size() == 3u);
    CHECK_FALSE(s1.items[0].is_prefill);
    CHECK(s1.items[0].request_logits);
    CHECK_FALSE(s1.items[1].is_prefill);
    CHECK(s1.items[1].request_logits);
    CHECK(s1.items[2].is_prefill);
    CHECK(s1.items[2].slot_index == 1u);
    CHECK(s1.items[2].token_offset == 0u);
    CHECK(s1.items[2].token_count == 510);
    CHECK_FALSE(s1.items[2].request_logits);

    // Advance slot 1 by 510 tokens -> second chunk (490 tokens) finishes prefill AND requests logits!
    active[1].prefill_offset = 510;
    active[1].pos = 510;
    const PlannedBatchStep s2 = PlanContinuousBatchStep(active, /*token_budget=*/512, /*output_budget=*/4, rr);
    CHECK(s2.decode_slots == 2u);
    CHECK(s2.prefill_slots == 1u);
    CHECK(s2.total_tokens == 492);
    CHECK(s2.total_outputs == 3);  // 2 decodes + final prefill chunk
    REQUIRE(s2.items.size() == 3u);
    CHECK(s2.items[2].is_prefill);
    CHECK(s2.items[2].token_offset == 510u);
    CHECK(s2.items[2].token_count == 490);
    CHECK(s2.items[2].request_logits);

    // 4. Two concurrent long prefills share ubatch budget fairly (no head-of-line starvation).
    std::vector<BatchSlotPlanView> two_prefills{
        {0, 0, BatchSlotPhase::kPrefill, 1000, 0, 0, false},
        {1, 1, BatchSlotPhase::kPrefill, 1000, 0, 0, false},
    };
    rr = 0;
    const PlannedBatchStep fair = PlanContinuousBatchStep(two_prefills, /*token_budget=*/256, /*output_budget=*/4, rr);
    CHECK(fair.prefill_slots == 2u);
    CHECK(fair.total_tokens == 256);
    CHECK(fair.total_outputs == 0);
    REQUIRE(fair.items.size() == 2u);
    CHECK(fair.items[0].token_count == 128);
    CHECK(fair.items[1].token_count == 128);

    // 5. Cancelled slot is skipped cleanly; tight output_budget trims final prefill token cleanly.
    std::vector<BatchSlotPlanView> tight{
        {0, 0, BatchSlotPhase::kDecode, 10, 10, 10, false},
        {1, 1, BatchSlotPhase::kDecode, 10, 10, 10, true},  // cancelled
        {2, 2, BatchSlotPhase::kPrefill, 20, 10, 10, false},
    };
    rr = 0;
    const PlannedBatchStep tight_step = PlanContinuousBatchStep(tight, /*token_budget=*/64, /*output_budget=*/1, rr);
    CHECK(tight_step.decode_slots == 1u);
    CHECK(tight_step.prefill_slots == 1u);
    CHECK(tight_step.total_outputs == 1);
    REQUIRE(tight_step.items.size() == 2u);
    CHECK(tight_step.items[1].slot_index == 2u);
    CHECK(tight_step.items[1].token_count == 9);  // leaves final token for next step when output_budget == 1
    CHECK_FALSE(tight_step.items[1].request_logits);
}

TEST_CASE("llamacpp frontier: ComputeTokenLogprobs exact log-softmax, numerical stability, and top-K") {
    // 1. Exact analytical values on [1.0, 2.0, 3.0, 0.0].
    const std::vector<float> logits{1.0F, 2.0F, 3.0F, 0.0F};
    const double z = std::exp(1.0) + std::exp(2.0) + std::exp(3.0) + std::exp(0.0);
    const double log_z = std::log(z);

    const TokenLogprobInfo info = ComputeTokenLogprobs(logits, /*chosen_token=*/2, /*top_k=*/3);
    REQUIRE(info.valid);
    CHECK(info.logprob == doctest::Approx(static_cast<float>(3.0 - log_z)).epsilon(1e-5));
    REQUIRE(info.top_logprobs.size() == 3u);
    CHECK(info.top_logprobs[0].first == 2);
    CHECK(info.top_logprobs[1].first == 1);
    CHECK(info.top_logprobs[2].first == 0);
    CHECK(info.top_logprobs[0].second > info.top_logprobs[1].second);
    CHECK(info.top_logprobs[1].second > info.top_logprobs[2].second);

    // Probabilities sum to 1.0 across full vocab.
    const TokenLogprobInfo all4 = ComputeTokenLogprobs(logits, /*chosen_token=*/0, /*top_k=*/4);
    REQUIRE(all4.top_logprobs.size() == 4u);
    double prob_sum = 0.0;
    for (const auto& [tok, lp] : all4.top_logprobs) {
        (void)tok;
        prob_sum += std::exp(static_cast<double>(lp));
    }
    CHECK(prob_sum == doctest::Approx(1.0).epsilon(1e-5));

    // 2. Shift invariance / numerical stability on huge logits (no exp() overflow).
    const std::vector<float> huge{10001.0F, 10002.0F, 10003.0F, 10000.0F};
    const TokenLogprobInfo huge_info = ComputeTokenLogprobs(huge, /*chosen_token=*/2, /*top_k=*/2);
    REQUIRE(huge_info.valid);
    CHECK(huge_info.logprob == doctest::Approx(info.logprob).epsilon(1e-5));
    REQUIRE(huge_info.top_logprobs.size() == 2u);
    CHECK(huge_info.top_logprobs[0].first == 2);
    CHECK(huge_info.top_logprobs[1].first == 1);

    // 3. Tie-breaking favors smaller token id; -inf logits are safely excluded from Z.
    const std::vector<float> tied{-std::numeric_limits<float>::infinity(), 2.0F, 2.0F, 1.0F};
    const TokenLogprobInfo tie_info = ComputeTokenLogprobs(tied, /*chosen_token=*/1, /*top_k=*/3);
    REQUIRE(tie_info.valid);
    REQUIRE(tie_info.top_logprobs.size() == 3u);
    CHECK(tie_info.top_logprobs[0].first == 1);
    CHECK(tie_info.top_logprobs[1].first == 2);
    CHECK(tie_info.top_logprobs[2].first == 3);
    CHECK(tie_info.top_logprobs[0].second == doctest::Approx(tie_info.top_logprobs[1].second).epsilon(1e-6));

    // 4. Invalid inputs return valid == false.
    CHECK_FALSE(ComputeTokenLogprobs({}, 0, 5).valid);
    CHECK_FALSE(ComputeTokenLogprobs(logits, -1, 5).valid);
    CHECK_FALSE(ComputeTokenLogprobs(logits, 4, 5).valid);
}

TEST_CASE("llamacpp frontier: suffix, cache_ram_mib, moe_cache_mib, mmproj_path, media_blobs, and logprobs options") {
    // SpeculativeType::kSuffix round-trips case-insensitively.
    SpeculativeType sp = SpeculativeType::kNone;
    CHECK(ToString(SpeculativeType::kSuffix) == "suffix");
    CHECK(ParseSpeculativeType("suffix", sp));
    CHECK(sp == SpeculativeType::kSuffix);
    CHECK(ParseSpeculativeType("SUFFIX", sp));
    CHECK(sp == SpeculativeType::kSuffix);

    // LoadOptions validation for new fields.
    LoadOptions lo;
    CHECK(lo.cache_ram_mib == 0u);
    CHECK(lo.moe_cache_mib == 0u);
    CHECK(lo.mmproj_path.empty());
    CHECK(ValidateContextOptions(lo).ok());

    lo.spec_type = "suffix";
    lo.cache_ram_mib = kMaxCacheRamMiB;
    lo.moe_cache_mib = kMaxMoeCacheMiB;
    lo.mmproj_path = "mmproj-model.gguf";
    CHECK(ValidateContextOptions(lo).ok());

    lo.cache_ram_mib = kMaxCacheRamMiB + 1;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.cache_ram_mib = 1024;
    lo.moe_cache_mib = kMaxMoeCacheMiB + 1;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.moe_cache_mib = 2048;
    lo.moe_cache_cpu_share = 0.0;
    CHECK(ValidateContextOptions(lo).ok());
    lo.moe_cache_cpu_share = 0.25;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.moe_cache_cpu_share = std::numeric_limits<double>::quiet_NaN();
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.moe_cache_cpu_share = 0.0;

    sonder::inference::LlamaCppBackendOptions backend_opts;
    backend_opts.moe_cache_cpu_share = 0.5;
    CHECK(sonder::inference::validate_llamacpp_options(backend_opts).code() ==
          sonder::inference::ErrorCode::invalid_argument);

    std::string bad_mm = "mmproj.gguf";
    bad_mm[6] = '\0';
    lo.mmproj_path = bad_mm;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);

    // ContextParamsFor maps moe_cache_mib -> moe_cache_size when probed.
    LoadOptions moe_lo;
    moe_lo.moe_cache_mib = 512;
    const ContextParamsSummary cs = LlamaCppBackend::ContextParamsFor(moe_lo);
#if defined(SONDER_LLAMACPP_HAS_MOE_CACHE_SIZE)
    CHECK(cs.moe_cache_size == (std::uint64_t{512} << 20));
#else
    CHECK(cs.moe_cache_size == 0u);
#endif

    // SamplingParams logprobs validation.
    SamplingParams sparams;
    CHECK(Validate(sparams).ok());
    sparams.compute_logprobs = true;
    sparams.top_logprobs = 0;
    CHECK(Validate(sparams).ok());
    sparams.top_logprobs = kMaxTopLogprobs;
    CHECK(Validate(sparams).ok());
    sparams.top_logprobs = kMaxTopLogprobs + 1;
    CHECK(Validate(sparams).code == ErrorCode::kInvalidArgument);
    sparams.top_logprobs = -1;
    CHECK(Validate(sparams).code == ErrorCode::kInvalidArgument);
    sparams.compute_logprobs = false;
    sparams.top_logprobs = 5;
    CHECK(Validate(sparams).code == ErrorCode::kInvalidArgument);

    // Media blobs validation (max 8 images, non-empty, max 16 MiB per blob).
    CHECK(ValidateMediaBlobs({}).ok());
    std::vector<std::vector<std::uint8_t>> ok_blobs(kMaxMediaImages, std::vector<std::uint8_t>{0x89, 0x50, 0x4E, 0x47});
    CHECK(ValidateMediaBlobs(ok_blobs).ok());
    ok_blobs.push_back({0x01});
    CHECK(ValidateMediaBlobs(ok_blobs).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateMediaBlobs({{}}).code == ErrorCode::kInvalidArgument);
    std::vector<std::vector<std::uint8_t>> huge_blob;
    huge_blob.emplace_back(kMaxMediaBlobBytes + 1, std::uint8_t{0xFF});
    CHECK(ValidateMediaBlobs(huge_blob).code == ErrorCode::kInvalidArgument);

    // LlamaCppBackendOptions adapter validation & continuous_batch_decode capability advertisement.
    namespace si = sonder::inference;
    si::LlamaCppBackendOptions bo;
    bo.spec_type = "suffix";
    bo.cache_ram_mib = 2048;
    bo.moe_cache_mib = 4096;
    bo.mmproj_path = "vision.gguf";
    bo.slots = 4;
    CHECK(si::validate_llamacpp_options(bo).ok());

    auto be_multi = si::make_llamacpp_backend(bo);
    CHECK(be_multi->capabilities().has(si::Capability::continuous_batch_decode));
    CHECK(be_multi->max_concurrent_requests() == 4u);

    si::LlamaCppBackendOptions bo_single;
    bo_single.slots = 1;
    auto be_single = si::make_llamacpp_backend(bo_single);
    CHECK_FALSE(be_single->capabilities().has(si::Capability::continuous_batch_decode));
    CHECK(be_single->max_concurrent_requests() == 1u);

    bo.cache_ram_mib = kMaxCacheRamMiB + 1;
    CHECK(si::validate_llamacpp_options(bo).code() == si::ErrorCode::invalid_argument);
    bo.cache_ram_mib = 0;
    bo.moe_cache_mib = kMaxMoeCacheMiB + 1;
    CHECK(si::validate_llamacpp_options(bo).code() == si::ErrorCode::invalid_argument);

    // Media sent without mmproj loaded fails cleanly on vocab-only/no-mmproj backend.
    LlamaCppBackend raw_be;
    LoadOptions mm_opts;
    mm_opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    mm_opts.mmproj_path = "nonexistent-mmproj.gguf";
    mm_opts.vocab_only = false;
    const Status mm_load = raw_be.Load(mm_opts);
#if defined(SONDER_LLAMACPP_HAS_MTMD)
    CHECK(mm_load.code == ErrorCode::kFileNotFound);
#else
    CHECK(mm_load.code == ErrorCode::kInvalidArgument);
#endif
    CHECK_FALSE(raw_be.HasMmproj());
}

TEST_CASE("llamacpp frontier: RadixPrefixCache pinned in-flight protection and atomic tight-budget eviction") {
    RadixPrefixCache cache(/*max_bytes=*/100);
    auto make_entry = [](std::size_t n_bytes, std::uint8_t fill) {
        RadixCacheEntry e;
        e.state_blob.assign(n_bytes, fill);
        return e;
    };

    const std::vector<Token> p1{1, 2, 3};
    const std::vector<Token> p2{4, 5, 6};
    const std::vector<Token> p3{7, 8, 9};
    const std::vector<Token> p4{10, 11, 12};

    REQUIRE(cache.Insert(p1, make_entry(60, 0x11)));
    REQUIRE(cache.Insert(p2, make_entry(40, 0x22)));
    CHECK(cache.used_bytes() == 100u);
    CHECK(cache.CheckInvariants());

    // Pin p1 (which is the oldest LRU entry!) twice to verify ref-counting.
    PinnedRadixLookup pin1a = cache.LookupAndPin(p1, 3, false);
    PinnedRadixLookup pin1b = cache.LookupAndPin(p1, 3, false);
    REQUIRE(pin1a.found());
    REQUIRE(pin1b.found());
    CHECK(pin1a.entry() == pin1b.entry());
    CHECK(cache.pinned_entries() == 1u);
    CHECK(cache.pinned_bytes() == 60u);
    CHECK(cache.CheckInvariants());

    // Touch p2 so p1 is strictly the oldest entry in LRU.
    REQUIRE(cache.LookupLongestPrefix(p2, 3, false).found());

    // Tight budget failure: 50B cannot fit alongside 60B pinned (60 + 50 > 100).
    // Must fail ATOMICALLY with ZERO evictions (p2 stays intact!).
    CHECK_FALSE(cache.ReserveBeforeAlloc(50));
    CHECK(cache.evictions() == 0u);
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.used_bytes() == 100u);
    CHECK(cache.LookupLongestPrefix(p2, 3, false).found());
    CHECK_FALSE(cache.Insert(p3, make_entry(50, 0x33)));
    CHECK(cache.evictions() == 0u);
    CHECK(cache.entry_count() == 2u);
    CHECK(cache.CheckInvariants());

    // Inserting 40B MUST skip pinned LRU p1 and evict unpinned p2 instead.
    REQUIRE(cache.Insert(p3, make_entry(40, 0x33)));
    CHECK(cache.evictions() == 1u);
    CHECK(cache.used_bytes() == 100u);
    CHECK(cache.entry_count() == 2u);
    CHECK_FALSE(cache.LookupLongestPrefix(p2, 3, false).found());
    CHECK(cache.LookupLongestPrefix(p1, 3, false).found());
    CHECK(pin1a.entry()->state_blob.size() == 60u);
    CHECK(pin1a.entry()->state_blob.front() == 0x11);
    CHECK(cache.CheckInvariants());

    // Move construction & assignment preserve pin ownership cleanly.
    PinnedRadixLookup moved_pin(std::move(pin1b));
    CHECK_FALSE(pin1b.found());
    CHECK(moved_pin.found());
    CHECK(cache.pinned_entries() == 1u);
    moved_pin.Release();
    CHECK(cache.pinned_entries() == 1u);  // pin1a still holds a pin
    CHECK(cache.pinned_bytes() == 60u);

    // Erasing a pinned entry retires it from lookup visibility immediately while
    // keeping its payload valid and its bytes counted until pin1a releases.
    CHECK(cache.Erase(p1));
    CHECK_FALSE(cache.LookupLongestPrefix(p1, 3, false).found());
    CHECK(cache.entry_count() == 1u);
    CHECK(cache.retired_pinned_bytes() == 60u);
    CHECK(cache.used_bytes() == 100u);  // 40B resident (p3) + 60B retired in-flight (p1)
    CHECK(pin1a.entry()->state_blob.front() == 0x11);
    CHECK(cache.CheckInvariants());

    // While retired p1 (60B) is still in flight, inserting 50B still fails atomically.
    CHECK_FALSE(cache.Insert(p4, make_entry(50, 0x44)));
    CHECK(cache.entry_count() == 1u);

    // Releasing the last pin reclaims retired RAM immediately.
    pin1a.Release();
    CHECK(cache.pinned_entries() == 0u);
    CHECK(cache.retired_pinned_bytes() == 0u);
    CHECK(cache.used_bytes() == 40u);
    CHECK(cache.CheckInvariants());

    // Overwriting a pinned entry while room exists retires the old payload cleanly.
    PinnedRadixLookup pin3 = cache.LookupAndPin(p3, 3, false);
    REQUIRE(pin3.found());
    REQUIRE(cache.Insert(p3, make_entry(50, 0x55)));
    CHECK(cache.entry_count() == 1u);
    CHECK(cache.retired_pinned_bytes() == 40u);
    CHECK(cache.used_bytes() == 90u);
    CHECK(pin3.entry()->state_blob.front() == 0x33);
    const RadixLookupResult fresh3 = cache.LookupLongestPrefix(p3, 3, false);
    REQUIRE(fresh3.found());
    CHECK(fresh3.entry->state_blob.front() == 0x55);
    pin3.Release();
    CHECK(cache.used_bytes() == 50u);
    CHECK(cache.CheckInvariants());

    // SetMaxBytes(0) while pinned retires without UAF and reclaims on release.
    PinnedRadixLookup pin_last = cache.LookupAndPin(p3, 3, false);
    REQUIRE(pin_last.found());
    cache.SetMaxBytes(0);
    CHECK_FALSE(cache.enabled());
    CHECK(cache.entry_count() == 0u);
    CHECK(cache.retired_pinned_bytes() == 50u);
    CHECK(pin_last.entry()->state_blob.front() == 0x55);
    pin_last.Release();
    CHECK(cache.used_bytes() == 0u);
    CHECK(cache.retired_pinned_bytes() == 0u);
    CHECK(cache.CheckInvariants());
}

TEST_CASE("llamacpp frontier: RadixPrefixCache multi-branch ErasePrefix, empty/single-token, and deterministic trim") {
    RadixPrefixCache cache(/*max_bytes=*/8192);
    auto make_entry = [](std::size_t n_bytes, std::uint8_t fill) {
        RadixCacheEntry e;
        e.state_blob.assign(n_bytes, fill);
        return e;
    };

    // Empty and single-token edge cases.
    CHECK_FALSE(cache.Insert({}, make_entry(16, 0x01)));
    CHECK_FALSE(cache.Erase({}));
    CHECK(cache.ErasePrefix({}) == 0u);
    CHECK_FALSE(cache.LookupLongestPrefix({}, 4, true).found());

    const std::vector<Token> single_a{42};
    const std::vector<Token> single_b{99};
    REQUIRE(cache.Insert(single_a, make_entry(32, 0xAA)));
    REQUIRE(cache.Insert(single_b, make_entry(32, 0xBB)));
    CHECK_FALSE(cache.LookupLongestPrefix(single_a, /*max_match_tokens=*/0, true).found());
    {
        const RadixLookupResult r = cache.LookupLongestPrefix(single_a, /*max_match_tokens=*/1, false);
        REQUIRE(r.found());
        CHECK(r.matched_tokens == 1u);
        CHECK(r.entry->state_blob.front() == 0xAA);
    }

    // Build a multi-branch tree sharing prefix [1, 2, 3] plus sibling [1, 9, 10].
    REQUIRE(cache.Insert(std::vector<Token>{1, 2, 3}, make_entry(50, 0x10)));
    REQUIRE(cache.Insert(std::vector<Token>{1, 2, 3, 4, 5}, make_entry(100, 0x11)));
    REQUIRE(cache.Insert(std::vector<Token>{1, 2, 3, 4, 6}, make_entry(100, 0x12)));
    REQUIRE(cache.Insert(std::vector<Token>{1, 2, 3, 7, 8}, make_entry(100, 0x13)));
    REQUIRE(cache.Insert(std::vector<Token>{1, 9, 10}, make_entry(70, 0x19)));
    CHECK(cache.entry_count() == 7u);
    CHECK(cache.CheckInvariants());

    // Pin one descendant while invalidating the shared mid-edge prefix [1, 2].
    PinnedRadixLookup pinned_branch = cache.LookupAndPin(std::vector<Token>{1, 2, 3, 4, 5}, 5, false);
    REQUIRE(pinned_branch.found());
    CHECK(cache.ErasePrefix(std::vector<Token>{1, 2}) == 4u);
    CHECK(cache.entry_count() == 3u);  // single_a, single_b, [1, 9, 10]
    CHECK_FALSE(cache.LookupLongestPrefix(std::vector<Token>{1, 2, 3, 4, 5}, 5, false).found());
    CHECK_FALSE(cache.LookupLongestPrefix(std::vector<Token>{1, 2, 3, 7, 8}, 5, false).found());
    // With trim enabled, only token [1] from untouched sibling [1, 9, 10] matches (not erased [1, 2, ...]).
    {
        const RadixLookupResult sib = cache.LookupLongestPrefix(std::vector<Token>{1, 2, 3, 4, 5}, 5, true);
        REQUIRE(sib.found());
        CHECK(sib.matched_tokens == 1u);
        CHECK(sib.state_tokens == 3u);
        CHECK(sib.entry->state_blob.front() == 0x19);
    }
    CHECK(cache.LookupLongestPrefix(std::vector<Token>{1, 9, 10}, 3, false).found());
    CHECK(pinned_branch.entry()->state_blob.front() == 0x11);
    CHECK(cache.retired_pinned_bytes() == 100u);
    CHECK(cache.CheckInvariants());
    pinned_branch.Release();
    CHECK(cache.retired_pinned_bytes() == 0u);
    CHECK(cache.CheckInvariants());

    // Prefix longer than stored leaf [1, 9, 10] must NOT invalidate [1, 9, 10].
    CHECK(cache.ErasePrefix(std::vector<Token>{1, 9, 10, 11}) == 0u);
    CHECK(cache.LookupLongestPrefix(std::vector<Token>{1, 9, 10}, 3, false).found());

    // Deterministic tie-breaking on equal-depth descendants: prefers smaller byte size.
    REQUIRE(cache.Insert(std::vector<Token>{10, 20, 30, 1}, make_entry(200, 0xD1)));
    REQUIRE(cache.Insert(std::vector<Token>{10, 20, 30, 2}, make_entry(80, 0xD2)));
    const RadixLookupResult det =
        cache.LookupLongestPrefix(std::vector<Token>{10, 20, 30, 99}, /*max_match_tokens=*/3, true);
    REQUIRE(det.found());
    CHECK(det.matched_tokens == 3u);
    CHECK(det.state_tokens == 4u);
    CHECK(det.entry->state_blob.size() == 80u);
    CHECK(det.entry->state_blob.front() == 0xD2);

    // ByteSize accounts state_blob + dft_state_blob + mtp_pending_h cleanly.
    RadixCacheEntry full;
    full.state_blob.assign(128, 0x01);
    full.dft_state_blob.assign(64, 0x02);
    full.mtp_pending_h.assign(16, 1.0F);
    CHECK(full.ByteSize() == 128u + 64u + 16u * sizeof(float));
    CHECK(cache.CheckInvariants());
}

TEST_CASE("llamacpp frontier: RadixPrefixCache randomized property stress and concurrent thread safety") {
    // 1. Deterministic single-threaded property stress under tight byte budget.
    RadixPrefixCache cache(/*max_bytes=*/256);
    std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
    auto next_u64 = [&]() -> std::uint64_t {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    auto gen_prefix = [&]() -> std::vector<Token> {
        const std::size_t len = static_cast<std::size_t>(next_u64() % 6u);  // 0..5 tokens
        std::vector<Token> p;
        p.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            p.push_back(static_cast<Token>(1 + (next_u64() % 4u)));  // dense branching alphabet
        }
        return p;
    };

    std::vector<PinnedRadixLookup> held_pins;
    for (int step = 0; step < 1200; ++step) {
        const std::uint64_t op = next_u64() % 100u;
        const std::vector<Token> p = gen_prefix();
        if (op < 35u) {
            const std::size_t sz = static_cast<std::size_t>(16u + (next_u64() % 96u));
            RadixCacheEntry e;
            e.state_blob.assign(sz, static_cast<std::uint8_t>(step & 0xFF));
            (void)cache.Insert(p, std::move(e));
        } else if (op < 50u) {
            const std::size_t sz = static_cast<std::size_t>(16u + (next_u64() % 128u));
            (void)cache.ReserveBeforeAlloc(sz, p);
        } else if (op < 70u) {
            const bool trim = (next_u64() & 1u) != 0u;
            const std::size_t cap = static_cast<std::size_t>(next_u64() % 6u);
            auto pin = cache.LookupAndPin(p, cap, trim);
            if (pin.found()) {
                CHECK(pin.matched_tokens() <= std::min(p.size(), cap));
                CHECK(pin.state_tokens() >= pin.matched_tokens());
                REQUIRE(pin.entry() != nullptr);
                CHECK_FALSE(pin.entry()->state_blob.empty());
                if (held_pins.size() < 3u) {
                    held_pins.push_back(std::move(pin));
                }
            }
        } else if (op < 80u) {
            if (!held_pins.empty()) {
                held_pins.pop_back();
            }
        } else if (op < 90u) {
            (void)cache.Erase(p);
        } else if (op < 97u) {
            (void)cache.ErasePrefix(p);
        } else {
            const std::size_t new_cap = static_cast<std::size_t>(64u + (next_u64() % 320u));
            cache.SetMaxBytes(new_cap);
        }
        REQUIRE(cache.CheckInvariants());
    }
    held_pins.clear();
    CHECK(cache.pinned_entries() == 0u);
    CHECK(cache.retired_pinned_bytes() == 0u);
    CHECK(cache.CheckInvariants());

    // 2. Concurrent multi-threaded stress: 4 threads inserting, pinning, verifying
    // payload integrity, erasing, and prefix-invalidating simultaneously.
    RadixPrefixCache shared_cache(/*max_bytes=*/2048);
    std::atomic<bool> integrity_ok{true};
    std::vector<std::thread> workers;
    for (int tid = 0; tid < 4; ++tid) {
        workers.emplace_back([&, tid]() {
            std::uint64_t s = 0xA5A5A5A500000001ULL ^ static_cast<std::uint64_t>(tid * 2654435761ULL);
            auto rnd = [&]() -> std::uint64_t {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                return s;
            };
            for (int iter = 0; iter < 400; ++iter) {
                std::vector<Token> pref;
                const std::size_t len = 1u + static_cast<std::size_t>(rnd() % 5u);
                for (std::size_t j = 0; j < len; ++j) {
                    pref.push_back(static_cast<Token>(1 + (rnd() % 3u)));
                }
                const std::uint64_t action = rnd() % 10u;
                if (action < 4u) {
                    const auto tag = static_cast<std::uint8_t>(1u + (rnd() % 250u));
                    RadixCacheEntry e;
                    e.state_blob.assign(64u + static_cast<std::size_t>(rnd() % 128u), tag);
                    (void)shared_cache.Insert(pref, std::move(e));
                } else if (action < 8u) {
                    PinnedRadixLookup pin = shared_cache.LookupAndPin(pref, pref.size(), true);
                    if (pin.found()) {
                        const auto& blob = pin.entry()->state_blob;
                        if (blob.empty()) {
                            integrity_ok.store(false, std::memory_order_relaxed);
                        } else {
                            const std::uint8_t expected = blob.front();
                            for (std::uint8_t b : blob) {
                                if (b != expected) {
                                    integrity_ok.store(false, std::memory_order_relaxed);
                                    break;
                                }
                            }
                        }
                    }
                } else if (action == 8u) {
                    (void)shared_cache.Erase(pref);
                } else {
                    (void)shared_cache.ErasePrefix(pref);
                }
            }
        });
    }
    for (auto& th : workers) th.join();
    CHECK(integrity_ok.load());
    CHECK(shared_cache.pinned_entries() == 0u);
    CHECK(shared_cache.retired_pinned_bytes() == 0u);
    CHECK(shared_cache.used_bytes() <= shared_cache.max_bytes());
    CHECK(shared_cache.CheckInvariants());
}

TEST_CASE("llamacpp frontier: continuous batching fairness, overflow resilience, duplicate session_key, and simulation") {
    // 1. Duplicate session_key across free slots picks best LCP (tie-broken by MRU), not slot 0.
    {
        std::vector<PrefixSlotMeta> slots(3);
        slots[0] = {0, {1, 2}, "dup", 10};
        slots[1] = {1, {1, 2, 3, 4, 5}, "dup", 20};
        slots[2] = {2, {1, 2, 3, 4, 5}, "dup", 30};
        const std::vector<std::uint8_t> busy{0u, 0u, 0u};
        const std::vector<Token> prompt{1, 2, 3, 4, 5, 6};

        const auto avail = SelectAvailablePrefixSlot(slots, busy, prompt, "dup");
        REQUIRE(avail.has_value());
        CHECK(avail->slot_index == 2u);
        CHECK(avail->matched_prefix == 5u);

        const SlotSelection single = SelectPrefixSlot(slots, prompt, "dup");
        CHECK(single.slot_index == 2u);
        CHECK(single.matched_prefix == 5u);

        // Empty and single-token prompts always clamp matched_prefix to 0.
        const auto empty_sel = SelectAvailablePrefixSlot(slots, busy, {}, "dup");
        REQUIRE(empty_sel.has_value());
        CHECK(empty_sel->matched_prefix == 0u);
        CHECK(empty_sel->slot_index == 2u);  // tie-broken by MRU among same session_key
        const auto single_tok_sel = SelectAvailablePrefixSlot(slots, busy, std::vector<Token>{1}, "dup");
        REQUIRE(single_tok_sel.has_value());
        CHECK(single_tok_sel->matched_prefix == 0u);
        // Mismatched busy vector fails cleanly.
        CHECK_FALSE(SelectAvailablePrefixSlot(slots, std::vector<std::uint8_t>{0u}, prompt, "dup").has_value());
    }

    // 2. Decode fairness under tight output_budget/token_budget prevents tail-slot starvation
    // while keeping emitted decode items in canonical ascending order.
    {
        std::vector<BatchSlotPlanView> four_decodes;
        for (std::size_t i = 0; i < 4; ++i) {
            four_decodes.push_back({i, static_cast<std::int32_t>(i), BatchSlotPhase::kDecode, 10, 10, 10, false});
        }
        std::size_t rr = 0;
        std::array<int, 4> counts{0, 0, 0, 0};
        for (int step = 0; step < 4; ++step) {
            const PlannedBatchStep s = PlanContinuousBatchStep(four_decodes, /*token_budget=*/2, /*output_budget=*/2, rr);
            REQUIRE(s.decode_slots == 2u);
            REQUIRE(s.items.size() == 2u);
            CHECK(s.items[0].slot_index < s.items[1].slot_index);  // canonical ascending order
            for (const auto& item : s.items) {
                ++counts[item.slot_index];
            }
        }
        for (int c : counts) CHECK(c == 2);
    }

    // 3. Huge prompt_len and boundary inputs never overflow or wrap negative.
    {
        std::vector<BatchSlotPlanView> huge{
            {0, 0, BatchSlotPhase::kPrefill, std::numeric_limits<std::size_t>::max(), 0, 0, false},
            {1, 1, BatchSlotPhase::kIdle, 0, 0, 0, false},
            {2, 2, BatchSlotPhase::kPrefill, 10, 10, 10, false},  // already finished prefill
        };
        std::size_t rr = std::numeric_limits<std::size_t>::max();
        const PlannedBatchStep s = PlanContinuousBatchStep(huge, /*token_budget=*/128, /*output_budget=*/4, rr);
        CHECK(s.total_tokens == 128);
        CHECK(s.total_outputs == 0);
        REQUIRE(s.items.size() == 1u);
        CHECK(s.items[0].slot_index == 0u);
        CHECK(s.items[0].token_count == 128);
        CHECK_FALSE(s.items[0].request_logits);

        // Zero/negative budgets and empty active_slots return empty cleanly.
        CHECK(PlanContinuousBatchStep({}, 64, 4, rr).items.empty());
        CHECK(PlanContinuousBatchStep(huge, 0, 4, rr).items.empty());
        CHECK(PlanContinuousBatchStep(huge, -10, 4, rr).items.empty());
        CHECK(PlanContinuousBatchStep(huge, 64, 0, rr).items.empty());
    }

    // 4. More than kMaxSlots (24 slots) exercises heap scratch fallback cleanly.
    {
        std::vector<BatchSlotPlanView> many;
        for (std::size_t i = 0; i < 24; ++i) {
            many.push_back({i, static_cast<std::int32_t>(i), BatchSlotPhase::kPrefill, 50, 0, 0, false});
        }
        std::size_t rr = 0;
        const PlannedBatchStep s = PlanContinuousBatchStep(many, /*token_budget=*/120, /*output_budget=*/24, rr);
        CHECK(s.total_tokens == 120);
        CHECK(s.prefill_slots == 24u);
    }

    // 5. End-to-end deterministic continuous-batching state machine simulation:
    // 8 concurrent prompts (lengths 1..257), tight ubatch=32 and output_budget=3.
    {
        const std::vector<std::size_t> prompt_lens{1, 2, 17, 64, 129, 257, 33, 5};
        const std::vector<int> target_gen{3, 1, 5, 4, 2, 3, 6, 2};
        std::vector<BatchSlotPlanView> state;
        std::vector<int> generated(prompt_lens.size(), 0);
        std::vector<bool> done(prompt_lens.size(), false);
        for (std::size_t i = 0; i < prompt_lens.size(); ++i) {
            state.push_back({i, static_cast<std::int32_t>(i), BatchSlotPhase::kPrefill, prompt_lens[i], 0, 0, false});
        }
        std::size_t rr = 0;
        int steps = 0;
        while (steps < 500) {
            std::vector<BatchSlotPlanView> active_view;
            for (std::size_t i = 0; i < state.size(); ++i) {
                if (!done[i]) active_view.push_back(state[i]);
            }
            if (active_view.empty()) break;
            const PlannedBatchStep p = PlanContinuousBatchStep(active_view, /*token_budget=*/32, /*output_budget=*/3, rr);
            REQUIRE(p.total_tokens > 0);
            REQUIRE(p.total_tokens <= 32);
            REQUIRE(p.total_outputs <= 3);
            std::int32_t sum_toks = 0;
            std::int32_t sum_outs = 0;
            for (const auto& item : p.items) {
                sum_toks += item.token_count;
                if (item.request_logits) ++sum_outs;
                BatchSlotPlanView& st = state[item.slot_index];
                if (item.is_prefill) {
                    REQUIRE(item.token_offset == st.prefill_offset);
                    REQUIRE(item.pos_start == st.pos);
                    st.prefill_offset += static_cast<std::size_t>(item.token_count);
                    st.pos += item.token_count;
                    if (st.prefill_offset == st.prompt_len) {
                        REQUIRE(item.request_logits);
                        ++generated[item.slot_index];
                        if (generated[item.slot_index] >= target_gen[item.slot_index]) {
                            done[item.slot_index] = true;
                        } else {
                            st.phase = BatchSlotPhase::kDecode;
                        }
                    } else {
                        REQUIRE_FALSE(item.request_logits);
                    }
                } else {
                    REQUIRE(item.token_count == 1);
                    REQUIRE(item.request_logits);
                    st.pos += 1;
                    ++generated[item.slot_index];
                    if (generated[item.slot_index] >= target_gen[item.slot_index]) {
                        done[item.slot_index] = true;
                    }
                }
            }
            CHECK(sum_toks == p.total_tokens);
            CHECK(sum_outs == p.total_outputs);
            ++steps;
        }
        for (std::size_t i = 0; i < state.size(); ++i) {
            CHECK(done[i]);
            CHECK(state[i].prefill_offset == prompt_lens[i]);
            CHECK(generated[i] == target_gen[i]);
        }
    }
}

TEST_CASE("llamacpp frontier: SuffixMatcher GSAM invariants, separator dead-end preservation, overflow, and burst recovery") {
    // 1. GSAM invariant: repeated identical/overlapping sequences across separators
    // never create equal-length suffix links or exceed 2 * capacity + 1 states.
    {
        BoundedSuffixAutomaton sam(/*max_tokens=*/128);
        const std::vector<Token> seq_a{10, 20, 30, 40, 50};
        const std::vector<Token> seq_b{20, 30, 40, 60, 70};
        for (int i = 0; i < 40; ++i) {
            sam.AppendSpan(i % 2 == 0 ? seq_a : seq_b);
            sam.AppendSeparator();
            REQUIRE(sam.CheckInvariants());
        }
        CHECK(sam.size() <= sam.capacity());
        CHECK(sam.state_count() <= 2u * sam.capacity() + 1u);
    }

    // 2. Dead-end separator-terminated occurrences must NOT evict valid earlier
    // continuations from `prev_pos`.
    {
        BoundedSuffixAutomaton sam(/*max_tokens=*/512);
        // Occurrence 1 has a rich continuation [70, 71, 72, 73].
        sam.AppendSpan(std::vector<Token>{10, 11, 12, 13, 70, 71, 72, 73});
        sam.AppendSeparator();
        // Occurrences 2, 3, 4 all end right at [10, 11, 12, 13] followed by separator.
        for (int k = 0; k < 3; ++k) {
            sam.AppendSpan(std::vector<Token>{10, 11, 12, 13});
            sam.AppendSeparator();
        }
        REQUIRE(sam.CheckInvariants());
        const SuffixMatchResult m =
            sam.MatchLongestSuffix(std::vector<Token>{10, 11, 12, 13}, /*min_match_len=*/3, /*max_draft=*/4);
        CHECK(m.match_len == 4u);
        REQUIRE(m.draft == std::vector<Token>{70, 71, 72, 73});
    }

    // 3. Unsigned overflow & NaN/Inf/huge max_draft safety in MatchLongestSuffix.
    {
        BoundedSuffixAutomaton sam(/*max_tokens=*/256);
        sam.AppendSpan(std::vector<Token>{1, 2, 3, 4, 5, 6, 7, 8});
        const std::vector<Token> q{1, 2, 3};
        CHECK(sam.MatchLongestSuffix(q, 3, 4, std::numeric_limits<std::size_t>::max()).draft.empty());
        CHECK(sam.MatchLongestSuffix(q, 3, 4, std::numeric_limits<std::size_t>::max() - 1).draft.empty());
        CHECK(sam.MatchLongestSuffix(q, 3, 4, sam.size()).draft.empty());
        CHECK(sam.MatchLongestSuffix(q, 3, 4, sam.size() - 1).draft.empty());
        CHECK(sam.MatchLongestSuffix({}, 1, 4).draft.empty());
        CHECK(sam.MatchLongestSuffix(std::vector<Token>{1}, 1, 0).draft.empty());
        CHECK(sam.MatchLongestSuffix(std::vector<Token>{-1, -2, -3}, 1, 4).draft.empty());

        const SuffixMatchResult nan_fac =
            sam.MatchLongestSuffix(q, 3, std::numeric_limits<std::size_t>::max(), 0,
                                   std::numeric_limits<double>::quiet_NaN());
        CHECK(nan_fac.draft == std::vector<Token>{4, 5, 6});
        const SuffixMatchResult inf_fac =
            sam.MatchLongestSuffix(q, 3, std::numeric_limits<std::size_t>::max(), 0,
                                   std::numeric_limits<double>::infinity());
        CHECK(inf_fac.draft == std::vector<Token>{4, 5, 6});  // non-finite factor safely defaults to 1.0
    }

    // 4. Hybrid MTP burst recovery: skips short-continuation last_pos to pick
    // qualifying >= 6-token continuation from prev_pos OR session_ring.
    {
        SuffixMatcher matcher(/*session_ring_capacity=*/512, /*active_capacity=*/512);
        std::vector<Token> prompt;
        for (Token t = 200; t < 218; ++t) prompt.push_back(t);  // 200..205 + 12 continuation tokens (206..217)
        prompt.push_back(999);
        for (Token t = 200; t < 206; ++t) prompt.push_back(t);  // repeats 200..205 near end
        prompt.push_back(888);
        prompt.push_back(889);  // only 2 continuation tokens!
        matcher.BeginRequest(prompt);
        const SuffixMatchResult burst =
            matcher.TryHybridMtpBurst(std::vector<Token>{200, 201, 202, 203, 204, 205}, /*max_k=*/16);
        REQUIRE(burst.draft.size() == 12u);
        CHECK(burst.draft.front() == 206);
        CHECK(burst.draft.back() == 217);

        // Session ring has a 10-token burst while active only has a 3-token continuation.
        std::vector<Token> ring_seq;
        for (Token t = 300; t < 316; ++t) ring_seq.push_back(t);  // 300..305 + 306..315 (10 tokens)
        matcher.IngestSessionSequence(ring_seq);
        matcher.BeginRequest(std::vector<Token>{300, 301, 302, 303, 304, 305, 701, 702, 703});
        const SuffixMatchResult ring_burst =
            matcher.TryHybridMtpBurst(std::vector<Token>{300, 301, 302, 303, 304, 305}, /*max_k=*/16);
        CHECK(ring_burst.from_session_ring);
        REQUIRE(ring_burst.draft.size() == 10u);
        CHECK(ring_burst.draft.front() == 306);
        CHECK(ring_burst.draft.back() == 315);
    }

    // 5. Huge AppendSpan compaction + current_request_ memory bound + ClearActive
    // + adaptive cooldown_period recovery & draft_factor preservation.
    {
        SuffixMatcher matcher(/*session_ring_capacity=*/64, /*active_capacity=*/64);
        std::vector<Token> giant;
        giant.reserve(500);
        for (Token t = 1; t <= 500; ++t) giant.push_back(t);
        matcher.BeginRequest(giant);
        CHECK(matcher.active_size() <= 64u);
        CHECK(matcher.CheckInvariants());
        for (Token t = 501; t <= 700; ++t) {
            matcher.CommitToken(t);
        }
        CHECK(matcher.CheckInvariants());
        matcher.ClearActive();
        CHECK(matcher.active_size() == 0u);
        CHECK(matcher.session_ring_size() == 0u);  // ClearActive does not pollute ring

        // Adaptive controller: cooldown_period resets on recovery, draft_factor
        // stays >= 3.0 on high acceptance after reaching 3.0.
        for (int i = 0; i < 4; ++i) matcher.Observe(8, 0);
        CHECK(matcher.paused());
        CHECK(matcher.cooldown_period() == 8);
        while (matcher.paused()) matcher.NoteSkippedStep();
        for (int i = 0; i < 6; ++i) matcher.Observe(8, 8);
        CHECK(matcher.cooldown_period() == 4);
        CHECK(matcher.draft_factor() == doctest::Approx(3.0));
        matcher.Observe(10, 9);  // 90% acceptance must NOT snap draft_factor down to 2.75
        CHECK(matcher.draft_factor() == doctest::Approx(3.0));
    }

    // 6. Overflow-safe NgramModTable::Observe & AdaptiveDraftController::Observe.
    {
        NgramModTable tbl(/*n_match=*/4, /*table_size=*/1024, /*n_min=*/4, /*n_max=*/8);
        CHECK_FALSE(tbl.Observe(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()));
        CHECK_FALSE(tbl.Observe(8, 999));  // accepted > drafted clamped cleanly
        AdaptiveDraftController ctrl(8);
        ctrl.Observe(std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max());
        ctrl.Observe(std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max());
        CHECK(ctrl.NextDraftLength() == 8);
    }

    // 7. Randomized GSAM property oracle: every drafted continuation MUST match
    // an actual occurrence of the queried suffix inside the retained token buffer
    // without crossing any separator.
    {
        BoundedSuffixAutomaton sam(/*max_tokens=*/96);
        std::vector<Token> oracle_buf;
        std::uint64_t rng = 0x123456789ABCDEF0ULL;
        auto rnd = [&]() -> std::uint64_t {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            return rng;
        };
        for (int step = 0; step < 300; ++step) {
            if (rnd() % 9u == 0u) {
                sam.AppendSeparator();
                if (!oracle_buf.empty() && oracle_buf.back() != BoundedSuffixAutomaton::kSeparator) {
                    oracle_buf.push_back(BoundedSuffixAutomaton::kSeparator);
                }
            } else {
                const Token t = static_cast<Token>(1 + (rnd() % 4u));
                sam.Append(t);
                oracle_buf.push_back(t);
            }
            REQUIRE(sam.CheckInvariants());
            if (step % 3 == 0 && oracle_buf.size() >= 6) {
                std::vector<Token> tail;
                const std::size_t qlen = 2u + static_cast<std::size_t>(rnd() % 6u);
                for (std::size_t j = 0; j < qlen; ++j) {
                    tail.push_back(static_cast<Token>(1 + (rnd() % 4u)));
                }
                const SuffixMatchResult r = sam.MatchLongestSuffix(tail, /*min_match_len=*/2, /*max_draft=*/6);
                if (!r.draft.empty()) {
                    CHECK(r.match_len >= 2u);
                    CHECK(r.match_len <= tail.size());
                    CHECK(r.draft.size() <= 6u);
                    // Verify no separator leaked into the draft.
                    for (Token dt : r.draft) {
                        CHECK(dt >= 0);
                    }
                    // Verify the exact (matched suffix + draft) exists contiguously in oracle_buf!
                    std::vector<Token> pattern(tail.end() - static_cast<std::ptrdiff_t>(r.match_len), tail.end());
                    pattern.insert(pattern.end(), r.draft.begin(), r.draft.end());
                    const auto it = std::search(oracle_buf.begin(), oracle_buf.end(), pattern.begin(), pattern.end());
                    CHECK(it != oracle_buf.end());
                }
            }
        }
    }
}

namespace {

std::vector<Token> MakePrefixedPrompt(std::size_t shared_len, Token shared_base, std::size_t suffix_len,
                                      Token suffix_base) {
    std::vector<Token> p;
    p.reserve(shared_len + suffix_len);
    for (std::size_t i = 0; i < shared_len; ++i) {
        p.push_back(shared_base + static_cast<Token>(i));
    }
    for (std::size_t i = 0; i < suffix_len; ++i) {
        p.push_back(suffix_base + static_cast<Token>(i));
    }
    return p;
}

}  // namespace

TEST_CASE("llamacpp frontier: PrefixCoalescer simultaneous cold siblings exact token-work savings") {
    PrefixCoalescerConfig cfg;
    cfg.enabled = true;
    cfg.architecture = sonder::inference::ModelArchitecture::attention_only;
    cfg.min_coalesce_prefix_tokens = 64;
    cfg.max_followers_per_producer = 8;
    cfg.max_coalesce_wait_ms = 5000;
    PrefixCoalescer coalescer(cfg);
    REQUIRE(coalescer.IsEnabled());

    // 4 simultaneous cold siblings sharing a 256-token prefix + distinct 64-token suffixes.
    constexpr std::size_t kSharedLen = 256;
    constexpr std::size_t kSuffixLen = 64;
    constexpr std::size_t kSiblings = 4;
    std::vector<std::vector<Token>> prompts;
    std::vector<BatchSlotPlanView> slots;
    std::vector<std::size_t> evaluated_tokens(kSiblings, 0);
    std::vector<std::size_t> cached_tokens(kSiblings, 0);
    for (std::size_t i = 0; i < kSiblings; ++i) {
        prompts.push_back(MakePrefixedPrompt(kSharedLen, 1000, kSuffixLen, static_cast<Token>(50000 + i * 1000)));
        slots.push_back(BatchSlotPlanView{i, static_cast<std::int32_t>(i), BatchSlotPhase::kPrefill,
                                          prompts[i].size(), 0, 0, false});
    }

    // Simultaneous cold admission at t = 0: slot 0 becomes producer, slots 1..3 become held followers.
    const CoalesceAdmitResult r0 = coalescer.OnAdmit(0, prompts[0], 0, /*now_ms=*/0);
    CHECK(r0.role == CoalesceRole::kIndependent);
    CHECK_FALSE(r0.producer_slot.has_value());

    for (std::size_t i = 1; i < kSiblings; ++i) {
        const CoalesceAdmitResult ri = coalescer.OnAdmit(i, prompts[i], 0, /*now_ms=*/0);
        CHECK(ri.role == CoalesceRole::kHeldFollower);
        REQUIRE(ri.producer_slot.has_value());
        CHECK(*ri.producer_slot == 0u);
        CHECK(ri.target_shared_len == kSharedLen);
        CHECK(coalescer.IsFollowerHeld(i));
    }
    CHECK(coalescer.RoleOf(0) == CoalesceRole::kProducer);
    CHECK(coalescer.ActiveGroupCount() == 1u);
    CHECK(coalescer.HeldFollowerCount() == 3u);
    CHECK(coalescer.SuggestedProducerPublishBoundary(0) == std::optional<std::size_t>(kSharedLen));
    CHECK(coalescer.CheckInvariants());

    // Drive continuous batching steps with chunk budget = 128 until all 4 prompts finish prefill.
    std::size_t rr = 0;
    int steps = 0;
    while (steps < 64) {
        std::vector<BatchSlotPlanView> views;
        for (const auto& s : slots) {
            if (s.phase == BatchSlotPhase::kPrefill && !coalescer.IsFollowerHeld(s.slot_index)) {
                views.push_back(s);
            }
        }
        if (views.empty()) break;

        // While producer hasn't reached kSharedLen yet, ONLY slot 0 may be scheduled!
        if (slots[0].prefill_offset < kSharedLen) {
            REQUIRE(views.size() == 1u);
            CHECK(views[0].slot_index == 0u);
        }

        const PlannedBatchStep plan = PlanContinuousBatchStep(views, /*token_budget=*/128, /*output_budget=*/4, rr);
        REQUIRE_FALSE(plan.items.empty());
        for (const PlannedBatchItem& item : plan.items) {
            REQUIRE(item.is_prefill);
            BatchSlotPlanView& st = slots[item.slot_index];
            st.prefill_offset += static_cast<std::size_t>(item.token_count);
            st.pos += item.token_count;
            evaluated_tokens[item.slot_index] += static_cast<std::size_t>(item.token_count);

            const auto released = coalescer.OnProducerProgress(item.slot_index, st.prefill_offset);
            for (const CoalesceReleaseAction& act : released) {
                CHECK(act.reason == CoalesceReleaseReason::kPrefixPublished);
                CHECK(act.producer_slot == 0u);
                CHECK(act.clone_prefix_len == kSharedLen);
                CHECK(act.tokens_saved == kSharedLen);
                BatchSlotPlanView& f = slots[act.follower_slot];
                f.prefill_offset = act.clone_prefix_len;
                f.pos = static_cast<std::int64_t>(act.clone_prefix_len);
                cached_tokens[act.follower_slot] += act.tokens_saved;
            }
            if (st.prefill_offset >= st.prompt_len) {
                st.phase = BatchSlotPhase::kDecode;
            }
            REQUIRE(coalescer.CheckInvariants());
        }
        ++steps;
    }

    // Verify exact work accounting: producer evaluated full 320, each follower evaluated ONLY its 64-token suffix!
    CHECK(evaluated_tokens[0] == kSharedLen + kSuffixLen);
    CHECK(cached_tokens[0] == 0u);
    std::size_t total_evaluated = evaluated_tokens[0];
    for (std::size_t i = 1; i < kSiblings; ++i) {
        CHECK(slots[i].phase == BatchSlotPhase::kDecode);
        CHECK(evaluated_tokens[i] == kSuffixLen);
        CHECK(cached_tokens[i] == kSharedLen);
        CHECK(evaluated_tokens[i] + cached_tokens[i] == prompts[i].size());
        total_evaluated += evaluated_tokens[i];
    }
    const std::size_t uncoalesced_total = kSiblings * (kSharedLen + kSuffixLen);
    const std::size_t expected_saved = (kSiblings - 1u) * kSharedLen;
    CHECK(total_evaluated == uncoalesced_total - expected_saved);

    const PrefixCoalescerStats st = coalescer.stats();
    CHECK(st.coalesced_groups == 1u);
    CHECK(st.coalesced_followers == 3u);
    CHECK(st.coalesced_tokens_saved == expected_saved);
    CHECK(st.coalesce_fallbacks_timeout == 0u);
    CHECK(st.coalesce_fallbacks_producer_failed == 0u);
    CHECK(coalescer.ActiveSlotCount() == 0u);
    CHECK(coalescer.ActiveGroupCount() == 0u);
    CHECK(coalescer.HeldFollowerCount() == 0u);
}

TEST_CASE("llamacpp frontier: PrefixCoalescer identical prompts, asymmetric boundaries, partial cache, and follower cap") {
    // 1. 100% identical prompts clamp target_shared_len to prompt.size() - 1 so follower still gets final logits.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 32;
        PrefixCoalescer c(cfg);
        const std::vector<Token> identical = MakePrefixedPrompt(100, 10, 0, 0);
        CHECK(c.OnAdmit(0, identical, 0, 0).role == CoalesceRole::kIndependent);
        const CoalesceAdmitResult f = c.OnAdmit(1, identical, 0, 0);
        CHECK(f.role == CoalesceRole::kHeldFollower);
        CHECK(f.target_shared_len == 99u);  // leaves token index 99 for logits!
        const auto rel = c.OnProducerProgress(0, 100);
        REQUIRE(rel.size() == 1u);
        CHECK(rel[0].clone_prefix_len == 99u);
        CHECK(rel[0].tokens_saved == 99u);
        CHECK(c.CheckInvariants());
    }

    // 2. Asymmetric shared prefixes: F1 shares 128 tokens, F2 shares 256 tokens -> chunked staggered release.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(256, 100, 100, 10000);
        const auto f1 = MakePrefixedPrompt(128, 100, 40, 20000);
        const auto f2 = MakePrefixedPrompt(256, 100, 40, 30000);
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, f1, 0, 0).target_shared_len == 128u);
        CHECK(c.OnAdmit(2, f2, 0, 0).target_shared_len == 256u);
        CHECK(c.SuggestedProducerPublishBoundary(0) == std::optional<std::size_t>(128u));

        // Chunk 1 reaches 128 -> releases F1 immediately while F2 stays held for 256!
        const auto rel1 = c.OnProducerProgress(0, 128);
        REQUIRE(rel1.size() == 1u);
        CHECK(rel1[0].follower_slot == 1u);
        CHECK(rel1[0].clone_prefix_len == 128u);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.IsFollowerHeld(2));
        CHECK(c.SuggestedProducerPublishBoundary(0) == std::optional<std::size_t>(256u));

        // Chunk 2 reaches 256 -> releases F2!
        const auto rel2 = c.OnProducerProgress(0, 256);
        REQUIRE(rel2.size() == 1u);
        CHECK(rel2[0].follower_slot == 2u);
        CHECK(rel2[0].clone_prefix_len == 256u);
        CHECK(c.stats().coalesced_tokens_saved == 384u);
        CHECK(c.CheckInvariants());
    }

    // 3. Partially warm arrival counts only incremental tokens saved (shared_len - initial_offset).
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(300, 500, 50, 1000);
        const auto f1 = MakePrefixedPrompt(300, 500, 50, 2000);
        CHECK(c.OnAdmit(0, p0, 100, 0).role == CoalesceRole::kIndependent);
        const CoalesceAdmitResult r = c.OnAdmit(1, f1, 100, 0);
        CHECK(r.role == CoalesceRole::kHeldFollower);
        CHECK(r.target_shared_len == 300u);
        const auto rel = c.OnProducerProgress(0, 300);
        REQUIRE(rel.size() == 1u);
        CHECK(rel[0].clone_prefix_len == 300u);
        CHECK(rel[0].tokens_saved == 200u);  // 300 - 100 already warm
        CHECK(c.stats().coalesced_tokens_saved == 200u);
    }

    // 4. Bounded `max_followers_per_producer`: excess siblings start independent (and form a second group).
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        cfg.max_followers_per_producer = 2;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(128, 1, 20, 1000);
        const auto p1 = MakePrefixedPrompt(128, 1, 20, 2000);
        const auto p2 = MakePrefixedPrompt(128, 1, 20, 3000);
        const auto p3 = MakePrefixedPrompt(128, 1, 20, 4000);
        const auto p4 = MakePrefixedPrompt(128, 1, 20, 5000);
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, p1, 0, 0).producer_slot == std::optional<std::size_t>(0u));
        CHECK(c.OnAdmit(2, p2, 0, 0).producer_slot == std::optional<std::size_t>(0u));
        // Slot 0 is full (2/2 followers) -> slot 3 proceeds independently, and slot 4 follows slot 3!
        CHECK(c.OnAdmit(3, p3, 0, 0).role == CoalesceRole::kIndependent);
        const CoalesceAdmitResult r4 = c.OnAdmit(4, p4, 0, 0);
        CHECK(r4.role == CoalesceRole::kHeldFollower);
        CHECK(r4.producer_slot == std::optional<std::size_t>(3u));
        CHECK(c.ActiveGroupCount() == 2u);
        CHECK(c.stats().coalesced_groups == 2u);
        CHECK(c.stats().coalesced_followers == 3u);
        CHECK(c.CheckInvariants());
    }
}

TEST_CASE("llamacpp frontier: PrefixCoalescer never delays unrelated concurrent requests") {
    PrefixCoalescerConfig cfg;
    cfg.min_coalesce_prefix_tokens = 64;
    PrefixCoalescer c(cfg);

    const auto sib_a = MakePrefixedPrompt(256, 100, 32, 1000);
    const auto sib_b = MakePrefixedPrompt(256, 100, 32, 2000);
    const auto unrelated = MakePrefixedPrompt(200, 9000, 40, 9900);

    CHECK(c.OnAdmit(0, sib_a, 0, 0).role == CoalesceRole::kIndependent);
    CHECK(c.OnAdmit(1, sib_b, 0, 0).role == CoalesceRole::kHeldFollower);
    const CoalesceAdmitResult u_res = c.OnAdmit(2, unrelated, 0, 0);
    CHECK(u_res.role == CoalesceRole::kIndependent);
    CHECK_FALSE(u_res.producer_slot.has_value());
    CHECK_FALSE(c.IsFollowerHeld(2));

    // Step 1 continuous batch view includes active decode slot 3, producer 0, and unrelated 2 — skips ONLY held follower 1!
    std::vector<BatchSlotPlanView> all{
        {0, 0, BatchSlotPhase::kPrefill, sib_a.size(), 0, 0, false},
        {1, 1, BatchSlotPhase::kPrefill, sib_b.size(), 0, 0, false},
        {2, 2, BatchSlotPhase::kPrefill, unrelated.size(), 0, 0, false},
        {3, 3, BatchSlotPhase::kDecode, 64, 64, 70, false},
    };
    std::vector<BatchSlotPlanView> runnable;
    for (const auto& s : all) {
        if (!c.IsFollowerHeld(s.slot_index)) runnable.push_back(s);
    }
    REQUIRE(runnable.size() == 3u);
    std::size_t rr = 0;
    const PlannedBatchStep step = PlanContinuousBatchStep(runnable, /*token_budget=*/256, /*output_budget=*/4, rr);
    bool saw_decode = false;
    bool saw_producer = false;
    bool saw_unrelated = false;
    bool saw_held_follower = false;
    for (const auto& item : step.items) {
        if (item.slot_index == 3u) saw_decode = true;
        if (item.slot_index == 0u) saw_producer = true;
        if (item.slot_index == 2u) saw_unrelated = true;
        if (item.slot_index == 1u) saw_held_follower = true;
    }
    CHECK(saw_decode);
    CHECK(saw_producer);
    CHECK(saw_unrelated);
    CHECK_FALSE(saw_held_follower);
    CHECK(c.CheckInvariants());
}

TEST_CASE("llamacpp frontier: PrefixCoalescer producer cancellation/failure promotes next follower without deadlock") {
    // 1. Producer P0 cancels with 2 followers F1, F2 -> F1 promoted to producer, F2 re-attached to F1!
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        cfg.max_coalesce_wait_ms = 500;
        PrefixCoalescer c(cfg);

        const auto p0 = MakePrefixedPrompt(200, 10, 40, 1000);
        const auto f1 = MakePrefixedPrompt(200, 10, 40, 2000);
        const auto f2 = MakePrefixedPrompt(200, 10, 40, 3000);
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, f1, 0, 10).role == CoalesceRole::kHeldFollower);
        CHECK(c.OnAdmit(2, f2, 0, 20).role == CoalesceRole::kHeldFollower);

        // Producer 0 fails/cancels at t = 100 before publishing anything.
        const auto fail_actions = c.OnSlotRemoved(0);
        REQUIRE(fail_actions.size() == 1u);
        CHECK(fail_actions[0].follower_slot == 1u);
        CHECK(fail_actions[0].reason == CoalesceReleaseReason::kPromotedToProducer);
        CHECK(fail_actions[0].clone_prefix_len == 0u);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.RoleOf(1) == CoalesceRole::kProducer);
        CHECK(c.IsFollowerHeld(2));
        CHECK(c.ProducerOf(2) == std::optional<std::size_t>(1u));
        CHECK(c.TargetSharedLen(2) == 200u);
        CHECK(c.stats().coalesce_fallbacks_producer_failed == 1u);
        CHECK(c.stats().coalesced_groups == 1u);  // promotion preserves the group, doesn't double-count
        CHECK(c.CheckInvariants());

        // Promoted producer F1 finishes [0..200) -> F2 clones from F1 and saves 200 tokens!
        const auto rel = c.OnProducerProgress(1, 200);
        REQUIRE(rel.size() == 1u);
        CHECK(rel[0].follower_slot == 2u);
        CHECK(rel[0].producer_slot == 1u);
        CHECK(rel[0].clone_prefix_len == 200u);
        CHECK(rel[0].tokens_saved == 200u);
        CHECK(c.stats().coalesced_tokens_saved == 200u);
        CHECK(c.CheckInvariants());
    }

    // 2. Single follower: producer fails -> follower immediately falls back to independent prefill.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(128, 10, 20, 1000);
        const auto f1 = MakePrefixedPrompt(128, 10, 20, 2000);
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, f1, 0, 0).role == CoalesceRole::kHeldFollower);

        const auto actions = c.OnSlotRemoved(0);
        REQUIRE(actions.size() == 1u);
        CHECK(actions[0].follower_slot == 1u);
        CHECK(actions[0].reason == CoalesceReleaseReason::kProducerFailed);
        CHECK(actions[0].clone_prefix_len == 0u);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.RoleOf(1) == CoalesceRole::kIndependent);
        CHECK(c.stats().coalesce_fallbacks_producer_failed == 1u);
        CHECK(c.CheckInvariants());
    }

    // 3. Chain promotion preserves original wait_start_ms so timeout ceilings never reset.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        cfg.max_coalesce_wait_ms = 500;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(128, 10, 20, 1000);
        const auto f1 = MakePrefixedPrompt(128, 10, 20, 2000);
        const auto f2 = MakePrefixedPrompt(128, 10, 20, 3000);
        (void)c.OnAdmit(0, p0, 0, 0);
        (void)c.OnAdmit(1, f1, 0, 0);
        (void)c.OnAdmit(2, f2, 0, 100);  // wait_start_ms = 100

        (void)c.OnSlotRemoved(0);  // promotes F1, F2 keeps wait_start_ms = 100
        CHECK(c.PollTimeouts(599).empty());
        const auto expired = c.PollTimeouts(600);  // 600 - 100 == 500 >= max_coalesce_wait_ms!
        REQUIRE(expired.size() == 1u);
        CHECK(expired[0].follower_slot == 2u);
        CHECK(expired[0].reason == CoalesceReleaseReason::kTimeout);
        CHECK_FALSE(c.IsFollowerHeld(2));
        CHECK(c.stats().coalesce_fallbacks_timeout == 1u);
        CHECK(c.CheckInvariants());
    }

    // 4. Runtime clone failure rollback restores honest saved-token accounting.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        const auto p0 = MakePrefixedPrompt(128, 10, 20, 1000);
        const auto f1 = MakePrefixedPrompt(128, 10, 20, 2000);
        (void)c.OnAdmit(0, p0, 0, 0);
        (void)c.OnAdmit(1, f1, 0, 0);
        const auto rel = c.OnProducerProgress(0, 128);
        REQUIRE(rel.size() == 1u);
        CHECK(c.stats().coalesced_tokens_saved == 128u);
        c.OnCloneFailed(1, /*fallback_prefill_offset=*/0, rel[0].tokens_saved);
        CHECK(c.stats().coalesced_tokens_saved == 0u);
        CHECK(c.CheckInvariants());
    }
}

TEST_CASE("llamacpp frontier: PrefixCoalescer follower timeout falls back cleanly to independent prefill") {
    // 1. Timeout before any producer progress -> clean independent prefill from offset 0.
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        cfg.max_coalesce_wait_ms = 250;
        PrefixCoalescer c(cfg);

        const auto p0 = MakePrefixedPrompt(256, 10, 32, 1000);
        const auto f1 = MakePrefixedPrompt(256, 10, 32, 2000);
        CHECK(c.OnAdmit(0, p0, 0, 1000).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, f1, 0, 1000).role == CoalesceRole::kHeldFollower);

        // Before deadline: still held (also safe against monotonic-clock-skew now < wait_start).
        CHECK(c.PollTimeouts(900).empty());
        CHECK(c.PollTimeouts(1249).empty());
        CHECK(c.IsFollowerHeld(1));

        // At deadline (1250 - 1000 == 250): released with kTimeout and 0 cloned tokens.
        const auto expired = c.PollTimeouts(1250);
        REQUIRE(expired.size() == 1u);
        CHECK(expired[0].follower_slot == 1u);
        CHECK(expired[0].reason == CoalesceReleaseReason::kTimeout);
        CHECK(expired[0].clone_prefix_len == 0u);
        CHECK(expired[0].tokens_saved == 0u);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.RoleOf(0) == CoalesceRole::kIndependent);
        CHECK(c.ActiveGroupCount() == 0u);
        CHECK(c.stats().coalesce_fallbacks_timeout == 1u);
        CHECK(c.stats().coalesced_tokens_saved == 0u);
        CHECK(c.CheckInvariants());
    }

    // 2. Timeout after producer already computed a partial chunk -> harvests partial KV before independent tail!
    {
        PrefixCoalescerConfig cfg;
        cfg.min_coalesce_prefix_tokens = 64;
        cfg.max_coalesce_wait_ms = 250;
        PrefixCoalescer c(cfg);

        const auto p0 = MakePrefixedPrompt(256, 10, 32, 1000);
        const auto f1 = MakePrefixedPrompt(256, 10, 32, 2000);
        (void)c.OnAdmit(0, p0, 0, 100);
        (void)c.OnAdmit(1, f1, 0, 100);
        CHECK(c.OnProducerProgress(0, 128).empty());  // chunk 1 computed, target 256 not yet reached

        const auto expired = c.PollTimeouts(350);
        REQUIRE(expired.size() == 1u);
        CHECK(expired[0].follower_slot == 1u);
        CHECK(expired[0].producer_slot == 0u);
        CHECK(expired[0].reason == CoalesceReleaseReason::kTimeout);
        CHECK(expired[0].clone_prefix_len == 128u);
        CHECK(expired[0].tokens_saved == 128u);
        CHECK(c.stats().coalesce_fallbacks_timeout == 1u);
        CHECK(c.stats().coalesced_tokens_saved == 128u);
        CHECK(c.CheckInvariants());
    }
}

TEST_CASE("llamacpp frontier: PrefixCoalescer follower cancellation cleans up without leaking group state") {
    PrefixCoalescerConfig cfg;
    cfg.min_coalesce_prefix_tokens = 64;
    PrefixCoalescer c(cfg);

    const auto p0 = MakePrefixedPrompt(200, 10, 30, 1000);
    const auto f1 = MakePrefixedPrompt(200, 10, 30, 2000);
    const auto f2 = MakePrefixedPrompt(200, 10, 30, 3000);
    (void)c.OnAdmit(0, p0, 0, 0);
    (void)c.OnAdmit(1, f1, 0, 0);
    (void)c.OnAdmit(2, f2, 0, 0);
    CHECK(c.ActiveGroupCount() == 1u);
    CHECK(c.HeldFollowerCount() == 2u);

    // Cancel follower 1 while held: group stays intact with only follower 2.
    CHECK(c.OnSlotRemoved(1).empty());
    CHECK_FALSE(c.IsFollowerHeld(1));
    CHECK(c.HeldFollowerCount() == 1u);
    CHECK(c.ActiveGroupCount() == 1u);
    CHECK(c.FollowersOf(0) == std::vector<std::size_t>{2u});
    CHECK(c.stats().coalesce_fallbacks_producer_failed == 0u);
    CHECK(c.stats().coalesce_fallbacks_timeout == 0u);
    CHECK(c.CheckInvariants());

    // Cancel follower 2 while held: producer cleanly reverts to kIndependent with zero group leak.
    CHECK(c.OnSlotRemoved(2).empty());
    CHECK_FALSE(c.IsFollowerHeld(2));
    CHECK(c.RoleOf(0) == CoalesceRole::kIndependent);
    CHECK(c.FollowersOf(0).empty());
    CHECK(c.ActiveGroupCount() == 0u);
    CHECK(c.HeldFollowerCount() == 0u);
    CHECK(c.ActiveSlotCount() == 1u);
    CHECK(c.CheckInvariants());

    // Producer completes its prompt -> zero active slots remaining.
    CHECK(c.OnProducerProgress(0, p0.size()).empty());
    CHECK(c.ActiveSlotCount() == 0u);
    CHECK(c.ActiveGroupCount() == 0u);
    CHECK(c.HeldFollowerCount() == 0u);
    CHECK(c.CheckInvariants());
}

TEST_CASE("llamacpp frontier: PrefixCoalescer bypasses hybrid/recurrent models and validates options") {
    const auto p0 = MakePrefixedPrompt(256, 10, 32, 1000);
    const auto p1 = MakePrefixedPrompt(256, 10, 32, 2000);

    // 1. Hybrid without valid checkpoint -> safely bypassed!
    {
        PrefixCoalescerConfig cfg;
        cfg.enabled = true;
        cfg.architecture = sonder::inference::ModelArchitecture::hybrid;
        cfg.has_valid_checkpoint = false;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        CHECK_FALSE(c.IsEnabled());
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, p1, 0, 0).role == CoalesceRole::kIndependent);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.ActiveSlotCount() == 0u);
        CHECK(c.stats().coalesced_groups == 0u);
        CHECK(c.stats().coalesced_followers == 0u);
    }

    // 2. Recurrent without valid checkpoint -> safely bypassed!
    {
        PrefixCoalescerConfig cfg;
        cfg.enabled = true;
        cfg.architecture = sonder::inference::ModelArchitecture::recurrent;
        cfg.has_valid_checkpoint = false;
        cfg.min_coalesce_prefix_tokens = 64;
        PrefixCoalescer c(cfg);
        CHECK_FALSE(c.IsEnabled());
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, p1, 0, 0).role == CoalesceRole::kIndependent);
        CHECK_FALSE(c.IsFollowerHeld(1));
        CHECK(c.ActiveSlotCount() == 0u);
    }

    // 3. Explicit checkpoint flag allows hybrid opt-in; enabled=false bypasses attention_only.
    CHECK(PrefixCoalescer::IsArchitectureEligible(sonder::inference::ModelArchitecture::attention_only, false));
    CHECK_FALSE(PrefixCoalescer::IsArchitectureEligible(sonder::inference::ModelArchitecture::hybrid, false));
    CHECK_FALSE(PrefixCoalescer::IsArchitectureEligible(sonder::inference::ModelArchitecture::recurrent, false));
    CHECK(PrefixCoalescer::IsArchitectureEligible(sonder::inference::ModelArchitecture::hybrid, true));

    {
        PrefixCoalescerConfig cfg;
        cfg.enabled = false;
        cfg.architecture = sonder::inference::ModelArchitecture::attention_only;
        PrefixCoalescer c(cfg);
        CHECK_FALSE(c.IsEnabled());
        CHECK(c.OnAdmit(0, p0, 0, 0).role == CoalesceRole::kIndependent);
        CHECK(c.OnAdmit(1, p1, 0, 0).role == CoalesceRole::kIndependent);
    }

    // 4. LoadOptions & LlamaCppBackendOptions validation for prefix coalescing knobs.
    LoadOptions lo;
    CHECK(lo.prefix_coalesce);
    CHECK(lo.min_coalesce_prefix_tokens == kDefaultMinCoalescePrefixTokens);
    CHECK(lo.max_followers_per_producer == kDefaultMaxCoalesceFollowers);
    CHECK(lo.max_coalesce_wait_ms == kDefaultMaxCoalesceWaitMs);
    CHECK(ValidateContextOptions(lo).ok());

    lo.min_coalesce_prefix_tokens = 0;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.min_coalesce_prefix_tokens = 64;

    lo.max_followers_per_producer = 0;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.max_followers_per_producer = kMaxSlots + 1;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.max_followers_per_producer = kMaxSlots;
    CHECK(ValidateContextOptions(lo).ok());

    lo.max_coalesce_wait_ms = 0;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.max_coalesce_wait_ms = kMaxCoalesceWaitMs + 1;
    CHECK(ValidateContextOptions(lo).code == ErrorCode::kInvalidArgument);
    lo.max_coalesce_wait_ms = 1000;
    CHECK(ValidateContextOptions(lo).ok());

    // When prefix_coalesce is disabled, zeroed tuning fields are ignored cleanly.
    lo.prefix_coalesce = false;
    lo.min_coalesce_prefix_tokens = 0;
    lo.max_followers_per_producer = 0;
    lo.max_coalesce_wait_ms = 0;
    CHECK(ValidateContextOptions(lo).ok());

    namespace si = sonder::inference;
    si::LlamaCppBackendOptions bo;
    bo.prefix_coalesce = true;
    bo.min_coalesce_prefix_tokens = 128;
    bo.max_followers_per_producer = 4;
    bo.max_coalesce_wait_ms = 2000;
    CHECK(si::validate_llamacpp_options(bo).ok());
    bo.max_followers_per_producer = kMaxSlots + 1;
    CHECK(si::validate_llamacpp_options(bo).code() == si::ErrorCode::invalid_argument);

    LlamaCppBackend be;
    const PrefixCoalescerStats initial_stats = be.GetPrefixCoalescerStats();
    CHECK(initial_stats.coalesced_groups == 0u);
    CHECK(initial_stats.coalesced_followers == 0u);
    CHECK(initial_stats.coalesced_tokens_saved == 0u);
    CHECK(initial_stats.coalesce_fallbacks_timeout == 0u);
    CHECK(initial_stats.coalesce_fallbacks_producer_failed == 0u);
}

TEST_CASE("llamacpp frontier: PrefixCoalescer randomized stress guarantees zero deadlock and strict invariants") {
    PrefixCoalescerConfig cfg;
    cfg.enabled = true;
    cfg.architecture = sonder::inference::ModelArchitecture::attention_only;
    cfg.min_coalesce_prefix_tokens = 16;
    cfg.max_followers_per_producer = 3;
    cfg.max_coalesce_wait_ms = 200;
    PrefixCoalescer c(cfg);

    constexpr std::size_t kSlots = 12;
    std::array<bool, kSlots> active{};
    std::array<std::vector<Token>, kSlots> prompts{};
    std::array<std::size_t, kSlots> offsets{};

    std::uint64_t rng = 0xC0A1E5CE00000001ULL;
    auto rnd = [&]() -> std::uint64_t {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };

    std::uint64_t now_ms = 0;
    for (int step = 0; step < 1000; ++step) {
        now_ms += 10u + (rnd() % 40u);
        const std::uint64_t op = rnd() % 100u;

        if (op < 40u) {
            // Admit into a random free slot using one of 3 shared prefix families (or unrelated).
            const std::size_t s = static_cast<std::size_t>(rnd() % kSlots);
            if (!active[s]) {
                const std::uint64_t family = rnd() % 4u;
                const std::size_t shared = family < 3u ? (32u + static_cast<std::size_t>((rnd() % 3u) * 16u)) : 0u;
                const std::size_t suffix = 16u + static_cast<std::size_t>(rnd() % 32u);
                prompts[s] = MakePrefixedPrompt(shared, static_cast<Token>(1000 * (family + 1u)), suffix,
                                                static_cast<Token>(100000 + step * 64));
                offsets[s] = 0;
                active[s] = true;
                (void)c.OnAdmit(s, prompts[s], 0, now_ms);
            }
        } else if (op < 75u) {
            // Advance a runnable (unheld) active slot by 16..48 tokens.
            std::vector<std::size_t> runnable;
            for (std::size_t s = 0; s < kSlots; ++s) {
                if (active[s] && !c.IsFollowerHeld(s)) runnable.push_back(s);
            }
            if (!runnable.empty()) {
                const std::size_t s = runnable[static_cast<std::size_t>(rnd() % runnable.size())];
                const std::size_t adv = 16u + static_cast<std::size_t>(rnd() % 33u);
                offsets[s] = std::min(prompts[s].size(), offsets[s] + adv);
                const auto released = c.OnProducerProgress(s, offsets[s]);
                for (const auto& act : released) {
                    REQUIRE(active[act.follower_slot]);
                    REQUIRE(act.clone_prefix_len < prompts[act.follower_slot].size());
                    offsets[act.follower_slot] = act.clone_prefix_len;
                }
                if (offsets[s] >= prompts[s].size()) {
                    active[s] = false;
                }
            }
        } else if (op < 90u) {
            // Cancel/fail a random active slot (producer OR held follower).
            const std::size_t s = static_cast<std::size_t>(rnd() % kSlots);
            if (active[s]) {
                (void)c.OnSlotRemoved(s);
                active[s] = false;
            }
        } else {
            // Poll follower timeouts.
            const auto timed_out = c.PollTimeouts(now_ms);
            for (const auto& act : timed_out) {
                REQUIRE(active[act.follower_slot]);
                if (act.clone_prefix_len > 0) {
                    offsets[act.follower_slot] = act.clone_prefix_len;
                }
            }
        }

        REQUIRE(c.CheckInvariants());

        // Zero-deadlock invariant: whenever any active slot is held, at least one active slot MUST be runnable!
        std::size_t any_active = 0;
        std::size_t any_runnable = 0;
        for (std::size_t s = 0; s < kSlots; ++s) {
            if (!active[s]) continue;
            ++any_active;
            if (!c.IsFollowerHeld(s)) ++any_runnable;
        }
        if (any_active > 0) {
            REQUIRE(any_runnable > 0);
        }
    }

    // Drain all remaining active slots cleanly and verify zero leaked state.
    for (std::size_t s = 0; s < kSlots; ++s) {
        if (active[s]) {
            (void)c.OnSlotRemoved(s);
            active[s] = false;
        }
    }
    CHECK(c.ActiveSlotCount() == 0u);
    CHECK(c.ActiveGroupCount() == 0u);
    CHECK(c.HeldFollowerCount() == 0u);
    CHECK(c.CheckInvariants());
}

TEST_CASE("llamacpp frontier: SuffixMatcher optional measured-cost tracking overrides high acceptance") {
    SuffixMatcher matcher(/*session_ring_capacity=*/4096, /*active_capacity=*/4096);
    std::vector<Token> prompt;
    for (Token t = 10; t < 40; ++t) prompt.push_back(t);
    matcher.BeginRequest(prompt);
    const std::vector<Token> tail{10, 11, 12, 13};

    // 1. Seed ordinary non-speculative decode baseline: 10.0 ms / committed token.
    matcher.ObserveBaselineStep(10.0, 1);
    matcher.ObserveBaselineStep(20.0, 2);
    // Invalid inputs are ignored safely without corrupting baseline EMA.
    matcher.ObserveBaselineStep(std::numeric_limits<double>::quiet_NaN(), 1);
    matcher.ObserveBaselineStep(-5.0, 1);
    matcher.ObserveBaselineStep(10.0, 0);
    CHECK(matcher.baseline_ms_per_token() == doctest::Approx(10.0));
    CHECK(matcher.cost_stats().baseline_cost_samples == 2u);

    // 2. When suffix verify beats baseline (4.0 ms/tok: 36ms / 9 committed tokens at 100% accept),
    //    draft_factor grows and suffix stays active.
    const double initial_factor = matcher.draft_factor();
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/36.0);
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/36.0);
    CHECK(matcher.suffix_ms_per_token() == doctest::Approx(4.0));
    CHECK(matcher.draft_factor() > initial_factor);
    CHECK_FALSE(matcher.paused());
    CHECK_FALSE(matcher.cost_paused());

    // 3. Reset request and test break-even hysteresis deadband (10.1 ms/tok vs 10.0 ms/tok):
    //    even at 100% acceptance, draft_factor must NOT inflate while break-even.
    matcher.BeginRequest(prompt);
    matcher.ObserveBaselineStep(10.0, 1);
    matcher.ObserveBaselineStep(10.0, 1);
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/90.9);  // 10.1 ms/tok
    const double factor_after_s1 = matcher.draft_factor();
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/90.9);  // 10.1 ms/tok
    CHECK(matcher.draft_factor() <= factor_after_s1);
    CHECK_FALSE(matcher.cost_paused());

    // 4. High acceptance (100%!) with WORSE ms/committed-token (13.0 ms/tok = 117ms / 9 committed
    //    vs 10.0 ms/tok baseline, ratio 1.30 >= 1.12) MUST shrink draft_factor and trigger cost pause!
    matcher.BeginRequest(prompt);
    matcher.ObserveBaselineStep(10.0, 1);
    matcher.ObserveBaselineStep(10.0, 1);
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/117.0);  // 13.0 ms/tok
    const double factor_before_regress = matcher.draft_factor();
    matcher.ObserveWithCost(/*drafted=*/8, /*accepted=*/8, /*elapsed_ms=*/117.0);  // 13.0 ms/tok
    CHECK(matcher.acceptance_ema() == doctest::Approx(1.0));
    CHECK(matcher.suffix_ms_per_token() == doctest::Approx(13.0));
    CHECK(matcher.draft_factor() < factor_before_regress);
    CHECK(matcher.paused());
    CHECK(matcher.cost_paused());
    CHECK(matcher.cost_stats().cost_regressions == 1u);
    CHECK(matcher.Draft(tail, 8).draft.empty());

    // 5. Draining cooldown via NoteSkippedStep clears cost_paused and restores drafting.
    for (int i = 0; i < 16 && matcher.paused(); ++i) {
        matcher.NoteSkippedStep();
    }
    CHECK_FALSE(matcher.paused());
    CHECK_FALSE(matcher.cost_paused());
    CHECK_FALSE(matcher.Draft(tail, 8).draft.empty());
    CHECK(matcher.CheckInvariants());
}
