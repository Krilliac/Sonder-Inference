// Pure helper tests: no model, no llama.cpp calls.
#include <limits>
#include <string>

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
    CHECK_FALSE(caps.kv_export);
    CHECK_FALSE(caps.speculative_decode);
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
