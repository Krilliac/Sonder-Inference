// libFuzzer target: Ollama NDJSON stream decoder
// (ollama::StreamDecoder in src/backends/ollama/ollama_client.cpp) and the
// single-line helper ollama::parse_stream_line (ollama_protocol.cpp).
//
// Input layout: byte 0 = control (bit 0: generate/chat, bits 1..7: seed for
// the chunk-split pattern); the rest is the raw HTTP response body. Seeds
// use printable control bytes ('0' generate, '1' chat, ...).
//
// Invariants:
//   * No crash/UB for arbitrary bodies.
//   * Split invariance: feeding the body in arbitrary chunks produces exactly
//     the same result (text, thinking, model, done, done_reason, server
//     timings, final status) as feeding it in one call.
//   * finish() ok implies the stream saw a done chunk.
#include <cstddef>
#include <cstdint>
#include <string>

#include "backends/ollama/ollama_protocol.hpp"
#include "fuzz_common.hpp"
#include "sonder/inference/backends/ollama.hpp"

using namespace sonder::inference;
using namespace sonder::inference::ollama;

namespace {

struct Outcome {
    bool feed_ok = true;
    Status status;
    StreamResult result;
};

Outcome run_whole(StreamKind kind, std::string_view body) {
    Outcome o;
    StreamDecoder d(kind, {});
    o.feed_ok = d.feed(body);
    o.status = d.finish();
    o.result = d.result();
    return o;
}

Outcome run_chunked(StreamKind kind, std::string_view body, std::uint32_t seed) {
    Outcome o;
    StreamDecoder d(kind, {});
    std::uint32_t state = seed * 2654435761u + 1u;
    while (!body.empty()) {
        state = state * 1103515245u + 12345u;
        std::size_t n = 1 + ((state >> 16) % 17u);  // 1..17 bytes
        if (n > body.size()) n = body.size();
        if (!d.feed(body.substr(0, n))) {
            o.feed_ok = false;
            break;
        }
        body.remove_prefix(n);
    }
    o.status = d.finish();
    o.result = d.result();
    return o;
}

bool same_timings(const OllamaTimings& a, const OllamaTimings& b) {
    return a.has_server_timings == b.has_server_timings && a.total_duration_ns == b.total_duration_ns &&
           a.load_duration_ns == b.load_duration_ns && a.prompt_eval_count == b.prompt_eval_count &&
           a.prompt_eval_cached_count == b.prompt_eval_cached_count &&
           a.prompt_eval_duration_ns == b.prompt_eval_duration_ns && a.eval_count == b.eval_count &&
           a.eval_duration_ns == b.eval_duration_ns && a.content_chunks == b.content_chunks;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) return 0;
    const StreamKind kind = (data[0] & 1u) ? StreamKind::chat : StreamKind::generate;
    const std::uint32_t seed = data[0] >> 1;
    const std::string_view body = sonder_fuzz::as_view(data + 1, size - 1);

    const Outcome whole = run_whole(kind, body);
    const Outcome chunked = run_chunked(kind, body, seed);

    SONDER_FUZZ_CHECK(whole.feed_ok == chunked.feed_ok);
    SONDER_FUZZ_CHECK(whole.status.code() == chunked.status.code());
    SONDER_FUZZ_CHECK(whole.result.text == chunked.result.text);
    SONDER_FUZZ_CHECK(whole.result.thinking == chunked.result.thinking);
    SONDER_FUZZ_CHECK(whole.result.model == chunked.result.model);
    SONDER_FUZZ_CHECK(whole.result.done == chunked.result.done);
    SONDER_FUZZ_CHECK(whole.result.done_reason == chunked.result.done_reason);
    SONDER_FUZZ_CHECK(same_timings(whole.result.timings, chunked.result.timings));
    if (whole.status.ok()) {
        SONDER_FUZZ_CHECK(whole.result.done);
    }
    // Derived rates must be non-negative and not NaN.
    const double pt = whole.result.timings.prompt_tokens_per_sec();
    const double dt = whole.result.timings.decode_tokens_per_sec();
    SONDER_FUZZ_CHECK(pt >= 0.0 && pt == pt);
    SONDER_FUZZ_CHECK(dt >= 0.0 && dt == dt);
    (void)timing_attributes(whole.result.timings);

    // Line-level helper used by the Backend adapter path.
    std::string_view rest = body;
    while (!rest.empty()) {
        const auto nl = rest.find('\n');
        const std::string_view line = rest.substr(0, nl);
        GenerateStats stats;
        auto r = parse_stream_line(line, stats);
        if (!r.ok()) {
            SONDER_FUZZ_CHECK(r.status().code() == ErrorCode::protocol_error ||
                              r.status().code() == ErrorCode::backend_error);
        }
        if (nl == std::string_view::npos) break;
        rest.remove_prefix(nl + 1);
    }
    return 0;
}
