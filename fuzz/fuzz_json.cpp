// libFuzzer target: core JSON parser/serializer (src/common/json.cpp).
// Every other parser (Ollama NDJSON, bench corpus, telemetry) sits on top of
// json::parse, so this target gets the deepest coverage of the shared code.
//
// Invariants:
//   * parse() never crashes, hangs, or trips ASan/UBSan on arbitrary bytes.
//   * dump() of a parsed document re-parses, and dump() is a fixed point
//     (parse(dump(v)).dump() == dump(v)).
//   * dump() is always a single line (telemetry sinks write it as JSONL).
//   * valid UTF-8 in -> valid UTF-8 out.
#include <cstddef>
#include <cstdint>
#include <string>

#include "fuzz_common.hpp"
#include "sonder/inference/json.hpp"

namespace json = sonder::inference::json;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view text = sonder_fuzz::as_view(data, size);
    auto parsed = json::parse(text);
    if (!parsed.ok()) {
        SONDER_FUZZ_CHECK(!parsed.status().message().empty());
        return 0;
    }
    const std::string once = parsed.value().dump();
    SONDER_FUZZ_CHECK(once.find('\n') == std::string::npos);
    auto again = json::parse(once);
    SONDER_FUZZ_CHECK(again.ok());
    SONDER_FUZZ_CHECK(again.value().type() == parsed.value().type() ||
                      // A non-finite double (e.g. 1e999) is serialized as null.
                      again.value().is_null());
    SONDER_FUZZ_CHECK(again.value().dump() == once);
    if (sonder_fuzz::is_valid_utf8(text)) {
        SONDER_FUZZ_CHECK(sonder_fuzz::is_valid_utf8(once));
    }
    return 0;
}
