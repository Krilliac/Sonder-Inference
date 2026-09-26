// libFuzzer target: benchmark results rendering (bench/src/report.cpp).
// render_markdown() and default_result_stem() walk an arbitrary results
// document (e.g. a hand-edited or foreign bench/results/*.json).
//
// Invariants: no crash/UB; the stem is filesystem-safe (only [A-Za-z0-9._-],
// non-empty, no path separators).
#include <cstddef>
#include <cstdint>

#include "fuzz_common.hpp"
#include "sonder/inference/benchmark.hpp"

using namespace sonder::inference;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    auto parsed = json::parse(sonder_fuzz::as_view(data, size));
    if (!parsed.ok()) return 0;
    const std::string md = bench::render_markdown(parsed.value());
    SONDER_FUZZ_CHECK(!md.empty());
    const std::string stem = bench::default_result_stem(parsed.value());
    SONDER_FUZZ_CHECK(!stem.empty());
    for (char ch : stem) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                        ch == '.' || ch == '-' || ch == '_';
        SONDER_FUZZ_CHECK(ok);
    }
    return 0;
}
