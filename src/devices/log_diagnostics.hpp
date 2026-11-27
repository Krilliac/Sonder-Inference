// Bounded, I/O-free diagnostics for ggml backend log lines.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend_runtime.hpp"

namespace sonder::inference::llamaserver {

// ---- log parser -----------------------------------------------------------

// Recognised lines (matched as substrings, so log prefixes and timestamps
// do not matter):
//   kv_kernel_f16_fallback       "no FlashAttention vector kernel compiled for K/V types <k>-<v>, converting K and V to f16 instead (slow)"
//   mtp_tensors_ignored          "model has unused tensor blk.<n>.nextn.<name> (size = <bytes> bytes) -- ignoring"
//   partial_gpu_offload          "offloaded <n>/<total> layers to GPU" with n < total
//   cpu_buffer_fallback          "... cannot be used with preferred buffer type <buft>, using CPU instead"
//   no_gpu_device                "no usable GPU found"
//   gpu_init_failed              "failed to initialize CUDA", "failed to initialize HIP", "failed to initialize ROCm"
//   rocm_runtime_error           "HSA_STATUS_ERROR", "hipError"
//   vulkan_runtime_error         "failed to initialize Vulkan", "VK_ERROR_DEVICE_LOST"
//   gpu_oom                      CUDA/HIP/Vulkan out-of-memory / "OutOfDeviceMemoryError"
//   ternary_runtime_incompatible "unknown tensor type", "unsupported ggml type", "prism.hadamard" errors
// Repeated occurrences fold into one warning (count, and summed sizes for
// mtp_tensors_ignored). Lines are capped at kMaxLineBytes and the number of
// distinct warnings at kMaxWarnings.
class LogDiagnostics {
  public:
    static constexpr std::size_t kMaxLineBytes = 4096;
    static constexpr std::size_t kMaxWarnings = 32;

    void feed(std::string_view bytes);
    // Processes a trailing line without a newline.
    void finish();
    // Records the child command line at readiness.  The final verbosity
    // option wins; a level below four (or an invalid/missing value) makes
    // offload diagnostics blind and adds one diagnostics_blind warning.
    void ready(const std::vector<std::string> &args);
    void reset();
    [[nodiscard]] const std::vector<BackendWarning> &warnings() const noexcept { return warnings_; }
    [[nodiscard]] std::uint64_t lines() const noexcept { return lines_; }
    [[nodiscard]] std::string offload_status() const;

  private:
    void line(std::string_view text);
    BackendWarning *find(std::string_view code, std::string_view key);
    BackendWarning *add(BackendWarning warning, std::string key);

    std::string partial_;
    bool discarding_ = false;
    std::uint64_t lines_ = 0;
    std::vector<BackendWarning> warnings_;
    std::vector<std::string> keys_;  // parallel to warnings_: dedupe key
    std::uint64_t mtp_bytes_ = 0;
    bool ready_ = false;
    bool diagnostics_blind_ = false;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> offload_;
};

} // namespace sonder::inference::llamaserver
