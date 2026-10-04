// libFuzzer target: the GGUF header reader and the VRAM estimate behind
// `sonder-infer serve --profile` (src/server/src/vram_estimate.cpp).
//
// Input: raw bytes, read as the start of a model file (metadata and tensor
// table; no tensor data is read).
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary bytes, in the
//     reader or in estimate_vram on what it accepts (several profiles).
//   * a rejected header is invalid_argument with a "GGUF header: " message
//     of bounded size: file text quoted in errors is cut short, because the
//     message reaches /v1/models (`estimate_error`).
//   * an accepted header has a printable architecture of at most 256 bytes,
//     1..100000 blocks, per-block vectors of that length, and at most that
//     many MTP blocks; the estimate's total is the sum of its parts and it
//     never counts more GPU layers than blocks.
#include <cstddef>
#include <cstdint>
#include <string>

#include "fuzz_common.hpp"
#include "sonder/inference/launch_profile.hpp"

namespace si = sonder::inference;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    auto parsed = si::parse_gguf_model_info(sonder_fuzz::as_view(data, size));
    if (!parsed.ok()) {
        SONDER_FUZZ_CHECK(parsed.status().code() == si::ErrorCode::invalid_argument);
        SONDER_FUZZ_CHECK(parsed.status().message().rfind("GGUF header: ", 0) == 0);
        SONDER_FUZZ_CHECK(parsed.status().message().size() < 1024);
        return 0;
    }
    const si::GgufModelInfo& m = parsed.value();
    SONDER_FUZZ_CHECK(!m.architecture.empty() && m.architecture.size() <= 256);
    for (const char ch : m.architecture) SONDER_FUZZ_CHECK(ch >= 0x20 && ch < 0x7f);
    SONDER_FUZZ_CHECK(m.block_count >= 1 && m.block_count <= 100000);
    SONDER_FUZZ_CHECK(m.kv_heads.size() == m.block_count);
    SONDER_FUZZ_CHECK(m.block_bytes.size() == m.block_count);
    SONDER_FUZZ_CHECK(m.nextn_layers <= m.block_count);

    si::LaunchProfile p;
    p.name = "fuzz";
    p.backend = si::kLaunchProfileBackendLlamaServer;
    p.model = "fuzz.gguf";
    for (int variant = 0; variant < 4; ++variant) {
        if (variant == 1) {  // explicit slots, quantized KV, a fixed context
            p.parallel = 3;
            p.ctx_size = 4096;
            p.cache_type_k = "q8_0";
            p.cache_type_v = "q4_0";
        } else if (variant == 2) {  // MTP with rollback snapshots, partial offload
            p.speculative = si::SpeculativeSettings{{"draft-mtp"}, 3u, ""};
            p.n_gpu_layers = 1;
        } else if (variant == 3) {  // the direct backend
            p = si::LaunchProfile{};
            p.name = "fuzz";
            p.backend = si::kLaunchProfileBackendLlamaCpp;
            p.model = "fuzz.gguf";
        }
        const si::VramEstimate e = si::estimate_vram(m, p);
        // Unsigned sums wrap identically, so this holds for any input.
        SONDER_FUZZ_CHECK(e.total_bytes == e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes);
        SONDER_FUZZ_CHECK(e.gpu_layers <= m.block_count);
        SONDER_FUZZ_CHECK(e.attention_layers <= e.gpu_layers);
        SONDER_FUZZ_CHECK(e.total_mib() <= (e.total_bytes >> 20) + 1);
    }
    return 0;
}
