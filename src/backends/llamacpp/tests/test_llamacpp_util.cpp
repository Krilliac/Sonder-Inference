// Pure helper tests: no model, no llama.cpp calls.
#include <limits>
#include <string>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

TEST_CASE("llamacpp wrapper: CompleteUtf8Prefix finds the last complete code point") {
    CHECK(CompleteUtf8Prefix("") == 0u);
    CHECK(CompleteUtf8Prefix("abc") == 3u);
    const std::string euro = "\xE2\x82\xAC";  // U+20AC, 3 bytes
    CHECK(CompleteUtf8Prefix(euro) == 3u);
    CHECK(CompleteUtf8Prefix("a" + euro.substr(0, 1)) == 1u);
    CHECK(CompleteUtf8Prefix("a" + euro.substr(0, 2)) == 1u);
    const std::string emoji = "\xF0\x9F\x98\x80";  // U+1F600, 4 bytes
    CHECK(CompleteUtf8Prefix(emoji) == 4u);
    CHECK(CompleteUtf8Prefix("xy" + emoji.substr(0, 3)) == 2u);
    // Invalid bytes pass through untouched rather than stalling the stream.
    CHECK(CompleteUtf8Prefix("\xFF") == 1u);
    CHECK(CompleteUtf8Prefix("\x80\x80") == 2u);
}

TEST_CASE("llamacpp wrapper: Utf8StreamBuffer holds split multi-byte sequences") {
    Utf8StreamBuffer buf;
    const std::string emoji = "\xF0\x9F\x98\x80";
    std::string out;
    out += buf.Push("hi ");
    CHECK(out == "hi ");
    out += buf.Push(emoji.substr(0, 1));
    CHECK(out == "hi ");
    CHECK(buf.pending() == 1u);
    out += buf.Push(emoji.substr(1, 2));
    CHECK(buf.pending() == 3u);
    out += buf.Push(emoji.substr(3) + "!");
    CHECK(out == "hi " + emoji + "!");
    CHECK(buf.pending() == 0u);
    buf.Push("\xE2\x82");
    CHECK(buf.Flush() == "\xE2\x82");
    CHECK(buf.pending() == 0u);
}

TEST_CASE("llamacpp wrapper: SamplingParams validation and enum strings") {
    CHECK(Validate(SamplingParams{}).ok());
    SamplingParams p;
    p.top_p = 0.0F;
    CHECK(Validate(p).code == ErrorCode::kInvalidArgument);
    p = {};
    p.temperature = std::numeric_limits<float>::infinity();
    CHECK(!Validate(p).ok());
    CHECK(ToString(StopReason::kCancelled) == "cancelled");
    CHECK(ToString(StopReason::kEndOfGeneration) == "end_of_generation");
    CHECK(ToString(ErrorCode::kFileNotFound) == "file_not_found");
    CHECK(ToString(DeviceKind::kCpu) == "cpu");
}

TEST_CASE("llamacpp wrapper: GenerateResult throughput math") {
    GenerateResult r;
    CHECK(r.decode_tokens_per_second() == 0.0);
    r.generated_tokens = 50;
    r.decode_ms = 500.0;
    r.prompt_tokens = 10;
    r.prefill_ms = 2.0;
    CHECK(r.decode_tokens_per_second() == 100.0);
    CHECK(r.prefill_tokens_per_second() == 5000.0);
}

TEST_CASE("llamacpp wrapper: wrapper advertises its capabilities") {
    const Capabilities caps = LlamaCppBackend::GetCapabilities();
    CHECK(caps.tokenization);
    CHECK(caps.streaming_decode);
    CHECK(caps.cancellation);
    CHECK(caps.prefix_reuse);
    CHECK(caps.speculative_decode);
    CHECK(caps.structured_output);
    CHECK(caps.embeddings);
    CHECK_FALSE(caps.kv_export);
    CHECK(LlamaCppBackend::Name() == "llama.cpp");
}

TEST_CASE("llamacpp wrapper: KV cache type and flash attention names round-trip") {
    const KvCacheType all[] = {KvCacheType::kF16,  KvCacheType::kF32,  KvCacheType::kBF16,
                               KvCacheType::kQ8_0, KvCacheType::kQ5_1, KvCacheType::kQ5_0,
                               KvCacheType::kQ4_1, KvCacheType::kQ4_0, KvCacheType::kIQ4_NL};
    for (const KvCacheType t : all) {
        KvCacheType parsed = t == KvCacheType::kF16 ? KvCacheType::kF32 : KvCacheType::kF16;
        CHECK(ParseKvCacheType(ToString(t), parsed));
        CHECK(parsed == t);
    }
    KvCacheType t = KvCacheType::kF16;
    CHECK(ParseKvCacheType("Q8_0", t));
    CHECK(t == KvCacheType::kQ8_0);
    CHECK(ParseKvCacheType("IQ4_NL", t));
    CHECK(t == KvCacheType::kIQ4_NL);
    for (const char* bad : {"", "q8", "q4_k", "fp16", "q8_0 ", "f16x"}) {
        CAPTURE(bad);
        KvCacheType untouched = KvCacheType::kQ4_1;
        CHECK_FALSE(ParseKvCacheType(bad, untouched));
        CHECK(untouched == KvCacheType::kQ4_1);
    }
    CHECK_FALSE(IsQuantized(KvCacheType::kF16));
    CHECK_FALSE(IsQuantized(KvCacheType::kF32));
    CHECK_FALSE(IsQuantized(KvCacheType::kBF16));
    CHECK(IsQuantized(KvCacheType::kQ8_0));
    CHECK(IsQuantized(KvCacheType::kQ4_0));
    CHECK(IsQuantized(KvCacheType::kIQ4_NL));

    FlashAttention fa = FlashAttention::kAuto;
    CHECK(ParseFlashAttention("on", fa));
    CHECK(fa == FlashAttention::kEnabled);
    CHECK(ParseFlashAttention("OFF", fa));
    CHECK(fa == FlashAttention::kDisabled);
    CHECK(ParseFlashAttention("enabled", fa));
    CHECK(fa == FlashAttention::kEnabled);
    CHECK(ParseFlashAttention("disabled", fa));
    CHECK(fa == FlashAttention::kDisabled);
    CHECK(ParseFlashAttention("Auto", fa));
    CHECK(fa == FlashAttention::kAuto);
    CHECK_FALSE(ParseFlashAttention("1", fa));
    CHECK_FALSE(ParseFlashAttention("", fa));
    CHECK(ToString(FlashAttention::kAuto) == "auto");
    CHECK(ToString(FlashAttention::kEnabled) == "on");
    CHECK(ToString(FlashAttention::kDisabled) == "off");
}

TEST_CASE("llamacpp wrapper: context option defaults match llama.cpp and validate") {
    const LoadOptions d;
    CHECK(d.type_k == KvCacheType::kF16);
    CHECK(d.type_v == KvCacheType::kF16);
    CHECK(d.flash_attn == FlashAttention::kAuto);
    CHECK(d.n_ubatch == 0u);
    CHECK(d.n_batch == 512u);
    CHECK(ValidateContextOptions(d).ok());

    LoadOptions o;
    o.n_batch = 256;
    o.n_ubatch = 256;
    CHECK(ValidateContextOptions(o).ok());
    o.n_ubatch = 257;
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
    // n_batch == 0 is left to Load()'s own "n_batch must be > 0" error.
    o.n_batch = 0;
    CHECK(ValidateContextOptions(o).ok());

    // Quantized K works without flash attention; a quantized V cache does not.
    o = {};
    o.type_k = KvCacheType::kQ8_0;
    o.flash_attn = FlashAttention::kDisabled;
    CHECK(ValidateContextOptions(o).ok());
    for (const KvCacheType v : {KvCacheType::kQ8_0, KvCacheType::kQ5_1, KvCacheType::kQ5_0, KvCacheType::kQ4_1,
                                KvCacheType::kQ4_0, KvCacheType::kIQ4_NL}) {
        CAPTURE(ToString(v));
        o.type_v = v;
        o.flash_attn = FlashAttention::kDisabled;
        const Status s = ValidateContextOptions(o);
        CHECK(s.code == ErrorCode::kInvalidArgument);
        CHECK(s.message.find("flash attention") != std::string::npos);
        o.flash_attn = FlashAttention::kEnabled;
        CHECK(ValidateContextOptions(o).ok());
        o.flash_attn = FlashAttention::kAuto;  // llama.cpp enables it itself
        CHECK(ValidateContextOptions(o).ok());
    }
    for (const KvCacheType v : {KvCacheType::kF16, KvCacheType::kF32, KvCacheType::kBF16}) {
        o.type_v = v;
        o.flash_attn = FlashAttention::kDisabled;
        CHECK(ValidateContextOptions(o).ok());
    }

    // Out-of-range enum values (e.g. from a cast) are rejected, not mapped.
    o = {};
    o.type_v = static_cast<KvCacheType>(99);
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
    o = {};
    o.flash_attn = static_cast<FlashAttention>(-7);
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);

    // vocab-only loads create no context, so nothing is checked.
    o = {};
    o.vocab_only = true;
    o.type_v = KvCacheType::kQ4_0;
    o.flash_attn = FlashAttention::kDisabled;
    o.n_ubatch = 100000;
    CHECK(ValidateContextOptions(o).ok());
}

TEST_CASE("llamacpp wrapper: SplitMode names round-trip and reject invalid input") {
    const SplitMode all[] = {SplitMode::kNone, SplitMode::kLayer, SplitMode::kRow};
    for (const SplitMode m : all) {
        SplitMode parsed = m == SplitMode::kLayer ? SplitMode::kNone : SplitMode::kLayer;
        CHECK(ParseSplitMode(ToString(m), parsed));
        CHECK(parsed == m);
    }
    CHECK(ToString(SplitMode::kNone) == "none");
    CHECK(ToString(SplitMode::kLayer) == "layer");
    CHECK(ToString(SplitMode::kRow) == "row");
    CHECK(ToString(static_cast<SplitMode>(99)) == "unknown");

    SplitMode parsed = SplitMode::kLayer;
    CHECK(ParseSplitMode("NONE", parsed));
    CHECK(parsed == SplitMode::kNone);
    CHECK(ParseSplitMode("Layer", parsed));
    CHECK(parsed == SplitMode::kLayer);
    CHECK(ParseSplitMode("rOw", parsed));
    CHECK(parsed == SplitMode::kRow);

    for (const char* bad : {"", "tensor", "layers", "layer ", "gpu", "0"}) {
        CAPTURE(bad);
        SplitMode untouched = SplitMode::kRow;
        CHECK_FALSE(ParseSplitMode(bad, untouched));
        CHECK(untouched == SplitMode::kRow);
    }
}

TEST_CASE("llamacpp wrapper: placement option defaults and pure validation") {
    const LoadOptions d;
    CHECK(d.devices.empty());
    CHECK(d.main_gpu == 0);
    CHECK(d.split_mode == SplitMode::kLayer);
    CHECK(d.tensor_split.empty());
    CHECK_FALSE(d.use_mlock);
    CHECK(ValidatePlacementOptions(d).ok());

    // main_gpu must be >= 0.
    LoadOptions o;
    o.main_gpu = 2;
    CHECK(ValidatePlacementOptions(o).ok());
    o.main_gpu = -1;
    Status s = ValidatePlacementOptions(o);
    CHECK(s.code == ErrorCode::kInvalidArgument);
    CHECK(s.message.find("main_gpu") != std::string::npos);

    // Out-of-range split_mode enum is rejected.
    o = {};
    o.split_mode = static_cast<SplitMode>(42);
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);

    // Valid tensor_split ratios (including zero on one GPU + positive on another, and up to 16).
    o = {};
    o.tensor_split = {1.0F};
    CHECK(ValidatePlacementOptions(o).ok());
    o.tensor_split = {0.0F, 1.0F};
    CHECK(ValidatePlacementOptions(o).ok());
    o.tensor_split = {0.65F, 0.35F};
    CHECK(ValidatePlacementOptions(o).ok());
    o.tensor_split.assign(kMaxTensorSplitDevices, 1.0F);
    CHECK(ValidatePlacementOptions(o).ok());

    // Invalid tensor_split: NaN, +Inf, -Inf, negative, all-zero, > 16.
    o = {};
    o.tensor_split = {std::numeric_limits<float>::quiet_NaN(), 1.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split = {std::numeric_limits<float>::infinity(), 1.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split = {-std::numeric_limits<float>::infinity(), 1.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split = {-0.1F, 1.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split = {0.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split = {0.0F, 0.0F};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.tensor_split.assign(kMaxTensorSplitDevices + 1, 1.0F);
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);

    // Device list pure checks: non-empty strings, no NUL, max 16.
    o = {};
    o.devices = {"CUDA0", "CUDA1"};
    CHECK(ValidatePlacementOptions(o).ok());
    o.devices = {""};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.devices = {"CUDA0", ""};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    std::string with_nul = "CUDA0";
    with_nul[4] = '\0';
    o.devices = {with_nul};
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);
    o.devices.assign(kMaxTensorSplitDevices + 1, "CUDA0");
    CHECK(ValidatePlacementOptions(o).code == ErrorCode::kInvalidArgument);

    // use_mlock is always accepted by pure validation.
    o = {};
    o.use_mlock = true;
    CHECK(ValidatePlacementOptions(o).ok());
}

TEST_CASE("llamacpp wrapper: SpeculativeType names round-trip and new context options validate") {
    const SpeculativeType all[] = {SpeculativeType::kNone, SpeculativeType::kAuto, SpeculativeType::kNgram,
                                   SpeculativeType::kNgramMod, SpeculativeType::kDraftMtp};
    for (const SpeculativeType s : all) {
        SpeculativeType parsed = s == SpeculativeType::kNone ? SpeculativeType::kAuto : SpeculativeType::kNone;
        CHECK(ParseSpeculativeType(ToString(s), parsed));
        CHECK(parsed == s);
    }
    SpeculativeType sp = SpeculativeType::kNone;
    CHECK(ParseSpeculativeType("NGRAM-MOD", sp));
    CHECK(sp == SpeculativeType::kNgramMod);
    CHECK(ParseSpeculativeType("Draft-MTP", sp));
    CHECK(sp == SpeculativeType::kDraftMtp);
    for (const char* bad : {"", "mtp", "ngram_mod", "eagle", "none "}) {
        SpeculativeType untouched = SpeculativeType::kAuto;
        CHECK_FALSE(ParseSpeculativeType(bad, untouched));
        CHECK(untouched == SpeculativeType::kAuto);
    }

    // Extended KV cache type names parse cleanly; support matches compile-time ggml flags.
    KvCacheType ext = KvCacheType::kF16;
    CHECK(ParseKvCacheType("f8_e4m3", ext));
    CHECK(ext == KvCacheType::kF8_E4M3);
    CHECK(ParseKvCacheType("q2_0", ext));
    CHECK(ext == KvCacheType::kQ2_0);
    CHECK(ParseKvCacheType("PQ2_0", ext));
    CHECK(ext == KvCacheType::kPQ2_0);
    CHECK(ParseKvCacheType("pq2_0m", ext));
    CHECK(ext == KvCacheType::kPQ2_0);
    CHECK(IsQuantized(KvCacheType::kQ2_0));
    CHECK(IsQuantized(KvCacheType::kPQ2_0));

    LoadOptions o;
    CHECK(o.use_mmap);
    CHECK(o.kv_unified);
    CHECK(o.prefix_cache);
    CHECK(o.n_slots == 1u);
    CHECK(o.spec_type == "none");
    CHECK(o.spec_draft_max == 0);
    CHECK(ValidateContextOptions(o).ok());

    o.n_slots = 0;
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
    o.n_slots = kMaxSlots;
    CHECK(ValidateContextOptions(o).ok());
    o.n_slots = kMaxSlots + 1;
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);

    o = {};
    o.spec_type = "ngram-mod";
    o.spec_draft_max = 8;
    CHECK(ValidateContextOptions(o).ok());
    o.spec_type = "bogus";
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
    o.spec_type = "auto";
    o.spec_draft_max = -1;
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
    o.spec_draft_max = kMaxSpecDraftTokens;
    CHECK(ValidateContextOptions(o).ok());
    o.spec_draft_max = kMaxSpecDraftTokens + 1;
    CHECK(ValidateContextOptions(o).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("llamacpp wrapper: prefix cache LCP and multi-slot selection") {
    const std::vector<Token> a{1, 2, 3, 4, 5};
    const std::vector<Token> b{1, 2, 3, 9, 10};
    CHECK(LongestCommonPrefix(a, b) == 3u);
    CHECK(LongestCommonPrefix(a, a) == 5u);
    CHECK(LongestCommonPrefix({}, a) == 0u);
    const std::vector<Token> disjoint{9, 2};
    CHECK(LongestCommonPrefix(disjoint, a) == 0u);

    std::vector<PrefixSlotMeta> slots(3);
    slots[0] = {0, {10, 20, 30, 40, 50}, "sess-a", 10};
    slots[1] = {1, {10, 20, 30, 99}, "sess-b", 20};
    slots[2] = {2, {7, 8, 9}, "", 5};

    // Exact session_key match takes precedence even when another slot has a longer prefix.
    {
        const std::vector<Token> prompt{10, 20, 30, 40, 50, 60};
        const SlotSelection sel = SelectPrefixSlot(slots, prompt, "sess-b");
        CHECK(sel.slot_index == 1u);
        CHECK(sel.matched_prefix == 3u);
    }

    // Identical prompt clamps matched_prefix to prompt.size() - 1 so >= 1 token is decoded.
    {
        const std::vector<Token> prompt{10, 20, 30, 40, 50};
        const SlotSelection sel = SelectPrefixSlot(slots, prompt, "sess-a");
        CHECK(sel.slot_index == 0u);
        CHECK(sel.matched_prefix == 4u);
    }

    // Single-token prompt always clamps matched_prefix to 0.
    {
        const std::vector<Token> prompt{10};
        const SlotSelection sel = SelectPrefixSlot(slots, prompt, "sess-a");
        CHECK(sel.slot_index == 0u);
        CHECK(sel.matched_prefix == 0u);
    }

    // Without session_key, picks longest common prefix.
    {
        const std::vector<Token> prompt{10, 20, 30, 40, 77};
        const SlotSelection sel = SelectPrefixSlot(slots, prompt, "");
        CHECK(sel.slot_index == 0u);
        CHECK(sel.matched_prefix == 4u);
    }

    // With no session_key and no prefix overlap, evicts LRU slot (slot 2, last_used=5).
    {
        const std::vector<Token> prompt{99, 98, 97};
        const SlotSelection sel = SelectPrefixSlot(slots, prompt, "");
        CHECK(sel.slot_index == 2u);
        CHECK(sel.matched_prefix == 0u);
    }
}

TEST_CASE("llamacpp wrapper: NgramMatcher drafts 4-gram and 3-gram continuations") {
    NgramMatcher matcher(/*exact_scan_fallback=*/false);
    // History repeats [10, 20, 30, 40] -> [50, 60, 70] and [20, 30, 40] -> [99] earlier.
    const std::vector<Token> history{
        1, 20, 30, 40, 99, 2, 10, 20, 30, 40, 50, 60, 70, 3, 10, 20, 30, 40,
    };
    // Index everything strictly before the current tail position (history.size() - 1).
    matcher.IndexRange(history, 0, history.size() - 1);

    std::vector<Token> draft;
    matcher.Draft(history, 4, draft);
    REQUIRE(draft.size() == 4u);
    const std::vector<Token> expected4{50, 60, 70, 3};
    CHECK(draft == expected4);

    // Respects max_draft limit.
    matcher.Draft(history, 2, draft);
    const std::vector<Token> expected2{50, 60};
    CHECK(draft == expected2);

    // 3-gram fallback when 4-gram differs.
    const std::vector<Token> history3{
        5, 11, 12, 13, 41, 42, 43, 9, 11, 12, 13,
    };
    matcher.Reset();
    matcher.IndexRange(history3, 0, history3.size() - 1);
    matcher.Draft(history3, 3, draft);
    const std::vector<Token> expected3{41, 42, 43};
    CHECK(draft == expected3);

    // Short history (< 3 tokens) produces empty draft cleanly.
    matcher.Draft(std::vector<Token>{1, 2}, 4, draft);
    CHECK(draft.empty());
}

TEST_CASE("llamacpp wrapper: AdaptiveDraftController throttles on low acceptance and recovers") {
    AdaptiveDraftController ctrl(8);
    CHECK(ctrl.NextDraftLength() == 8);
    CHECK_FALSE(ctrl.paused());

    // Repeated zero acceptance triggers brief pause after warmup (>= 3 steps, >= 8 drafted).
    ctrl.Observe(4, 0);
    ctrl.Observe(4, 0);
    ctrl.Observe(4, 0);
    CHECK(ctrl.paused());
    CHECK(ctrl.NextDraftLength() == 0);

    // Stepping through cooldown resumes with a small probe.
    for (int i = 0; i < 8; ++i) ctrl.NoteSkippedStep();
    CHECK_FALSE(ctrl.paused());
    CHECK(ctrl.NextDraftLength() == 2);

    // Full acceptance on probe recovers and ramps back up toward max_draft.
    ctrl.Observe(2, 2);
    CHECK(ctrl.NextDraftLength() > 2);
    for (int i = 0; i < 4; ++i) ctrl.Observe(ctrl.NextDraftLength(), ctrl.NextDraftLength());
    CHECK(ctrl.NextDraftLength() == 8);
}

TEST_CASE("llamacpp wrapper: NgramModTable chains across turns, enforces n_min, and resets on low streak") {
    // Small table & match length for deterministic unit testing.
    NgramModTable table(/*n_match=*/4, /*table_size=*/4096, /*n_min=*/6, /*n_max=*/12);
    CHECK(table.n_match() == 4u);
    CHECK(table.n_min() == 6u);
    CHECK(table.n_max() == 12u);

    // Turn 1 indexes a sequence of 20 tokens [100..119].
    std::vector<Token> turn1;
    for (Token t = 100; t < 120; ++t) turn1.push_back(t);
    table.BeginRequest(turn1);
    CHECK(table.used() == 16u);

    // Turn 2 starts with the same 4-token seed [100, 101, 102] + id_last=103:
    // persistent table chains 12 tokens [104..115] across requests.
    const std::vector<Token> turn2_prompt{100, 101, 102};
    table.BeginRequest(turn2_prompt);
    std::vector<Token> draft;
    table.Draft(turn2_prompt, /*id_last=*/103, /*max_draft=*/12, draft);
    REQUIRE(draft.size() == 12u);
    for (std::size_t i = 0; i < draft.size(); ++i) {
        CHECK(draft[i] == static_cast<Token>(104 + i));
    }

    // Chain shorter than n_min (only 3 tokens available from seed [113,114,115]+116 -> 117,118,119) is rejected.
    const std::vector<Token> tail_hist{113, 114, 115};
    table.Draft(tail_hist, /*id_last=*/116, /*max_draft=*/12, draft);
    CHECK(draft.empty());

    // Five consecutive low-acceptance rounds (<25%) reset the table cleanly.
    for (int i = 0; i < 4; ++i) {
        CHECK_FALSE(table.Observe(12, 0));
        CHECK(table.used() > 0u);
    }
    CHECK(table.Observe(12, 0));
    CHECK(table.used() == 0u);
    table.Draft(turn2_prompt, /*id_last=*/103, /*max_draft=*/12, draft);
    CHECK(draft.empty());
}
