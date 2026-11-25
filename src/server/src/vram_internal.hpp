// Internal: shared arithmetic, excerpt formatting, and MoE tensor placement
// helpers used across gguf_header.cpp, moe_placement.cpp, and vram_estimate.cpp.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/launch_profile.hpp"

namespace sonder::inference::detail {

inline constexpr std::uint64_t kMiB = 1ull << 20;

// File text (a key, tensor name, or flag value) as quoted in an error: at most
// 64 bytes, non-printable bytes as '?'. Errors reach /v1/models
// (`estimate_error`) and the operator's log.
// (Not named `quoted`: ADL on a std::string argument would pick std::quoted.)
inline std::string excerpt(std::string_view text) {
    std::string out;
    for (const char ch : text.substr(0, 64)) out += (ch >= 0x20 && ch < 0x7f) ? ch : '?';
    if (text.size() > 64) out += "...";
    return out;
}

// `v` rounded to the nearest integer and saturated to [0, 2^64 - 1]: casting
// an out-of-range double is undefined behaviour, and crafted GGUF metadata
// can push the KV arithmetic (done in double) past 2^64.
inline std::uint64_t saturating_u64(double v) {
    const double r = std::floor(v + 0.5);
    if (!(r > 0.0)) return 0;  // also NaN
    if (r >= 18446744073709551616.0) return std::numeric_limits<std::uint64_t>::max();  // 2^64
    return static_cast<std::uint64_t>(r);
}

// a x b and a + b, saturated at 2^64 - 1 (the window comes from the file).
inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b ? std::numeric_limits<std::uint64_t>::max()
                                                                         : a * b;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

// ggml types llama.cpp b11195 can repack for CPU matmul (arch/repack.cpp):
// Q4_0 (2), Q8_0 (8), Q4_K (12), IQ4_NL (20).
inline bool repackable_ggml_type(std::uint32_t type) noexcept {
    return type == 2 || type == 8 || type == 12 || type == 20;
}

struct TensorPlacementBreakdown {
    bool user_overrides = false;
    bool tensor_table_usable = false;
    std::uint64_t all_expert_total = 0;      // routed-expert bytes of every counted block tensor
    std::uint64_t all_repack_expert = 0;     // of those, repackable types
    std::uint64_t all_repack_dense = 0;      // repackable-type non-expert block bytes
    std::vector<std::uint64_t> moved_to_host;  // bytes of an offloaded block placed on the host
    std::vector<std::uint64_t> moved_to_gpu;   // bytes of a host block placed on the GPU
    std::uint64_t host_expert_total = 0;       // routed-expert bytes that end up on the host
    std::uint64_t host_repacked_expert = 0;    // repackable-type expert bytes on the host
    std::uint64_t host_repacked_dense = 0;     // repackable-type non-expert block bytes on the host
};

// Evaluates `placement` rules against `m.tensors` (matching llama.cpp's
// first-matching-rule walk) and records per-block overrides and repack totals.
TensorPlacementBreakdown evaluate_tensor_placement(const GgufModelInfo& m, MoePlacement& placement,
                                                   std::uint32_t n_layer, std::uint32_t first_gpu,
                                                   std::uint32_t first_nextn, bool mtp,
                                                   std::vector<std::string>& notes);

// Dynamic SLRU MoE expert VRAM cache sizing decision (llama.cpp PR #29887 /
// #30966 / #30637 anti-regression rules).
struct MoeExpertCacheSizing {
    bool eligible = false;
    std::uint32_t recommended_cache_mib = 0;
    double recommended_cpu_share = 0.0;
    std::uint64_t offloaded_expert_mib = 0;
    std::uint64_t active_working_set_mib = 0;
    std::uint64_t spare_vram_mib = 0;
    std::string reason;
};

MoeExpertCacheSizing plan_moe_expert_cache(const GgufModelInfo& model, std::uint64_t host_expert_bytes,
                                           std::uint64_t spare_vram_mib,
                                           std::optional<std::uint64_t> available_host_ram_bytes);

}  // namespace sonder::inference::detail
