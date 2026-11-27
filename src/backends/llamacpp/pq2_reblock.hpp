// Lossless PQ2_0 (group 128, 34 B/block, private type 142) -> stock Q2_0
// (group 64, 18 B/block, mainline type 42) GGUF re-blocking and verification.
//
// Upstream llama.cpp v1.0.0 (commit b1b770d) natively supports GGML_TYPE_Q2_0
// (ggml/include/ggml.h:261, ggml/src/ggml-common.h:188-192, ggml-quants.c:442)
// and the full `prism.hadamard.*` loader + FWHT graph transforms
// (src/llama-arch.cpp:501-511, src/llama-model.cpp:2174-2500). Both PQ2_0 and
// Q2_0 use identical 2-bit codes (00=-1, 01=0, 10=+1, 11=+2) scaled by one
// IEEE-754 binary16 delta `d`, so splitting each 34-byte PQ2_0 block
// `[d (2B) | qs_lo (16B) | qs_hi (16B)]` into two 18-byte Q2_0 blocks
// `[d (2B) | qs_lo (16B)]` and `[d (2B) | qs_hi (16B)]` is bit-for-bit lossless
// (+5.88% weight bytes on PQ2_0 tensors) and lets Sonder Inference run Bonsai
// checkpoints on its single pinned llama.cpp engine without any external fork.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

inline constexpr std::uint32_t kGgmlTypeQ2_0 = 42;
inline constexpr std::uint32_t kGgmlTypePq2_0 = 142;
inline constexpr std::uint32_t kGgmlTypePtq1_0 = 143;
inline constexpr std::uint32_t kGgmlTypePq1_0 = 144;
inline constexpr std::uint32_t kGgmlTypePq2_0m = 145;

inline constexpr std::size_t kPq2BlockElements = 128;
inline constexpr std::size_t kPq2BlockBytes = 34;
inline constexpr std::size_t kQ2BlockElements = 64;
inline constexpr std::size_t kQ2BlockBytes = 18;

inline constexpr std::uint32_t kLlamaFtypeMostlyQ2_0 = 41;
inline constexpr std::uint32_t kPrismFtypeMostlyPq2_0 = 141;
inline constexpr std::uint32_t kPrismFtypeMostlyPq2_0Legacy = 142;

struct Pq2GgufInspection {
    std::uint32_t gguf_version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t pq2_tensor_count = 0;
    std::uint64_t q2_tensor_count = 0;
    std::uint64_t pq2_input_bytes = 0;
    std::uint64_t q2_output_bytes = 0;
    std::uint64_t other_tensor_bytes = 0;
    bool prism_hadamard = false;
    std::optional<std::uint32_t> file_type;
};

struct Pq2ReblockReport {
    Pq2GgufInspection inspection;
    std::uint64_t reblocked_tensors = 0;
    std::uint64_t copied_tensors = 0;
    std::uint64_t output_file_bytes = 0;
    std::string output_sha256;
};

// Dequantizes packed PQ2_0 blocks (`n_elements` must be a positive multiple of
// 128; `pq2_bytes.size()` must equal `(n_elements / 128) * 34`). Rejects NaN or
// Inf FP16 scales fail-closed.
[[nodiscard]] Status DequantizePq2Row(std::span<const std::uint8_t> pq2_bytes,
                                      std::size_t n_elements,
                                      std::vector<float>& out_floats);

// Losslessly re-blocks packed PQ2_0 blocks into stock group-64 Q2_0 blocks
// (`out_q2_bytes.size() == (n_elements / 64) * 18`). Rejects malformed sizes
// and NaN/Inf FP16 scales fail-closed.
[[nodiscard]] Status ReblockPq2RowToQ2(std::span<const std::uint8_t> pq2_bytes,
                                       std::size_t n_elements,
                                       std::vector<std::uint8_t>& out_q2_bytes);

// Header-and-bounds inspection of a GGUF v2/v3 file. Rejects symlinks,
// truncated files, oversized metadata/tensor counts, non-monotonic or
// overlapping tensor regions, unsupported private tensor types (PTQ1_0=143,
// PQ1_0=144, PQ2_0M=145), and legacy group-128-as-type-42 files whose smaller
// payload byte spans truncate the expected group-64 Q2_0 payload bounds.
[[nodiscard]] Status InspectPq2Gguf(const std::filesystem::path& path,
                                    Pq2GgufInspection& out_inspection);

// Streams `input_path` to a new GGUF at `output_path`, re-blocking every PQ2_0
// (type 142) tensor to stock Q2_0 (type 42), preserving all metadata keys
// (including `prism.hadamard.*`) and non-PQ2_0 tensors, updating
// `general.file_type` (141/142 -> 41), and writing atomically via a temporary
// sibling file. Never modifies `input_path`.
[[nodiscard]] Status ReblockPq2ToQ2Gguf(const std::filesystem::path& input_path,
                                        const std::filesystem::path& output_path,
                                        Pq2ReblockReport* out_report = nullptr);

// Streams `pq2_path` and `q2_path` side by side and verifies matching tensor
// counts, names, shapes, preserved metadata, byte-exact non-PQ2_0 tensors, and
// bit-exact dequantized float values across every re-blocked tensor.
[[nodiscard]] Status VerifyPq2ToQ2Gguf(const std::filesystem::path& pq2_path,
                                       const std::filesystem::path& q2_path);

}  // namespace sonder::backends::llamacpp
