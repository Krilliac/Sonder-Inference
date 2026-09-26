// libFuzzer target: benchmark corpus loading (bench::parse_corpus in
// bench/src/benchmark.cpp; bench::load_corpus is ifstream + parse_corpus).
//
// Invariants on a successfully parsed corpus:
//   * at least one prompt; every prompt has a non-empty id and prompt text;
//   * 1 <= max_tokens <= SamplingConfig::kMaxTokensLimit;
//   * fan-out children are non-empty and share the recorded prefix length;
//   * provenance hash is present.
#include <cstddef>
#include <cstdint>

#include "fuzz_common.hpp"
#include "sonder/inference/benchmark.hpp"

using namespace sonder::inference;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    auto parsed = bench::parse_corpus(sonder_fuzz::as_view(data, size));
    if (!parsed.ok()) return 0;
    const bench::Corpus& c = parsed.value();
    SONDER_FUZZ_CHECK(!c.prompts.empty());
    SONDER_FUZZ_CHECK(!c.fnv1a.empty());
    for (const auto& p : c.prompts) {
        SONDER_FUZZ_CHECK(!p.id.empty());
        SONDER_FUZZ_CHECK(p.max_tokens >= 1 && p.max_tokens <= SamplingConfig::kMaxTokensLimit);
        if (p.is_fanout()) {
            for (const auto& child : p.children) {
                SONDER_FUZZ_CHECK(!child.empty());
                SONDER_FUZZ_CHECK(child.size() >= p.shared_prefix_bytes);
            }
        } else {
            SONDER_FUZZ_CHECK(!p.prompt.empty());
        }
    }
    return 0;
}
