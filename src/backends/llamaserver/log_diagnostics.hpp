// Structured performance diagnostics for a supervised llama-server child:
// an up-front check of the KV cache type pairing, and file tailing into the
// shared bounded parser in devices/log_diagnostics.hpp (docs/integration/
// vram-spill.md). Diagnostics only ever add warnings; they never change how
// the child is launched.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../devices/log_diagnostics.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

// ---- llama-server argv readers (read-only; the argv is caller-owned) ------

// Last value of -c / --ctx-size (also "--ctx-size=N"). nullopt when absent
// or not a positive integer.
std::optional<std::uint64_t> find_context_size(const std::vector<std::string> &args);

// Returns `args` with every context-size value replaced by `ctx`, or with
// "--ctx-size <ctx>" appended when none was present.
std::vector<std::string> with_context_size(std::vector<std::string> args, std::uint64_t ctx);

// Value of --log-file (also "--log-file=PATH"), when present.
std::optional<std::string> find_log_file(const std::vector<std::string> &args);

struct KvCacheConfig {
    std::string type_k = "f16";  // llama.cpp default
    std::string type_v = "f16";
    // "on", "off" or "auto" (llama.cpp's default when the flag is absent;
    // auto enables FlashAttention on CUDA when supported).
    std::string flash_attn = "auto";
};
KvCacheConfig read_kv_cache_config(const std::vector<std::string> &args);

// Up-front pairing check. With FlashAttention not explicitly off:
//  - K and V types that differ yield "kv_type_mismatch" (warning): llama.cpp
//    has no default FlashAttention vector kernel for mixed pairs and may
//    convert K and V to f16; measured on CUDA with q8_0/q5_1.
//  - q5_1/q5_1 yields "kv_type_no_vector_kernel" (warning); measured.
//  - other matching types outside f16/bf16/q8_0/q4_0 yield
//    "kv_type_kernel_unknown" (info): on CUDA such pairs may need
//    GGML_CUDA_FA_ALL_QUANTS=ON.
// When `backend_hint` is empty, the backend is inferred from `--device`/`-dev`
// in `args` (Vulkan*/ROCm*/HIP*/CUDA*), defaulting to CUDA wording.
std::vector<BackendWarning> check_kv_cache_pairing(const std::vector<std::string> &args,
                                                   std::string_view backend_hint = {});

// Incrementally reads a log file that a child appends to. Bounded per call;
// restarts from the beginning when the file shrinks (a new child truncated
// it). Never throws; I/O problems are reported through the Status.
class LogTail {
  public:
    explicit LogTail(std::string path, bool utf8 = false) : path_(std::move(path)), utf8_(utf8) {}
    [[nodiscard]] const std::string &path() const noexcept { return path_; }
    void rewind() noexcept { offset_ = 0; }
    Status poll(LogDiagnostics &sink, std::size_t max_bytes = 256 * 1024);

  private:
    std::string path_;
    std::uint64_t offset_ = 0;
    bool utf8_ = false; // default preserves existing native path interpretation
};

} // namespace sonder::inference::llamaserver
