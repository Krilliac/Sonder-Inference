// Frontier engine upgrade unit tests:
//   - Arctic-style SuffixMatcher (exact/variable-length/ring-eviction/hybrid MTP escalation)
//   - RadixPrefixCache (longest prefix match, LRU byte-budget eviction before alloc, split/overwrite)
//   - Continuous batching + chunked prefill scheduler state machine
//   - Option validation & plumbing (suffix, cache_ram_mib, moe_cache_mib, mmproj_path, media_blobs, logprobs)
//   - Exact numerically stable per-token logprob + top-K math
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "batch_scheduler.hpp"
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
