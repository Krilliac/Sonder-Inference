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
