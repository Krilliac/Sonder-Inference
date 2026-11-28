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
//   vulkan_runtime_error         "failed to initialize Vulkan", "VK_ERROR_DEVICE_LOST" (warning only), "ggml_vulkan: device lost on"
//   device_fault                 "ggml_vulkan: device fault on <name>: <description>" (VK_EXT_device_fault; the
//                                printing code is in the b11195 pin, the vendor-binary fix llama.cpp #32008 is
//                                not and first appears in v1.0.0) and fatal CUDA/HIP errors ("CUDA error: unspecified launch failure",
//                                "an illegal memory access was encountered", "device-side assert triggered", ...)
//   gpu_oom                      CUDA/HIP/Vulkan out-of-memory / "OutOfDeviceMemoryError"
//   ternary_runtime_incompatible "unknown tensor type", "unsupported ggml type", "prism.hadamard" errors
// Repeated occurrences fold into one warning (count, and summed sizes for
// mtp_tensors_ignored). Lines are capped at kMaxLineBytes and the number of
// distinct warnings at kMaxWarnings.
// A fatal accelerator fault seen in the child's log: the device cannot be
// trusted again until the host recovers it (for a hung OCuLink GPU that means a
// hard power cycle of the dock). The first fault wins for backend/reason; the
// device name and description are filled in by a later VK_EXT_device_fault
// line when the "device lost" line came first. Upstream prints the fault info
// first (print_device_lost_info calls print_device_fault_info before the lost
// line), so the real order is: fault line, address/vendor lines, lost line.
struct DeviceFault {
    std::string backend;      // "vulkan", "cuda" or "rocm"
    std::string reason;       // "device_lost", "device_fault" or "fatal_error"
    std::string device;       // device name when the log names it (may be empty)
    std::string description;  // driver/vendor text, bounded (may be empty)
    std::vector<std::string> details;  // bounded "address fault ..."/"vendor fault ..." lines
};

class LogDiagnostics {
  public:
    static constexpr std::size_t kMaxLineBytes = 4096;
    static constexpr std::size_t kMaxWarnings = 32;
    static constexpr std::size_t kMaxFaultDetails = 8;
    static constexpr std::size_t kMaxFaultText = 256;

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
    // The device fault seen since the last reset(), if any. Never cleared by
    // later healthy-looking lines: a lost device stays lost.
    [[nodiscard]] const std::optional<DeviceFault> &device_fault() const noexcept { return fault_; }

  private:
    void line(std::string_view text);
    BackendWarning *find(std::string_view code, std::string_view key);
    BackendWarning *add(BackendWarning warning, std::string key);
    void note_fault(std::string_view text);

    std::string partial_;
    bool discarding_ = false;
    std::uint64_t lines_ = 0;
    std::vector<BackendWarning> warnings_;
    std::vector<std::string> keys_;  // parallel to warnings_: dedupe key
    std::uint64_t mtp_bytes_ = 0;
    bool ready_ = false;
    bool diagnostics_blind_ = false;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> offload_;
    std::optional<DeviceFault> fault_;
    std::vector<std::string> pending_details_; // address/vendor lines seen before any fault line
};

} // namespace sonder::inference::llamaserver
