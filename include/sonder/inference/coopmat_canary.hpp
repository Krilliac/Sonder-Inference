// Sonder Inference: AMD Windows Vulkan cooperative-matrix child-process canary (NAT-16).
//
// Runs VK_KHR_cooperative_matrix property enumeration (#22750/#23508) and a
// generic 16x16x16 FP16->FP32 cooperative-matrix compute smoke test in an
// isolated child process so driver crashes or hangs never take down the server.
// Results are cached per (vendor_id, device_id, driver_build, ubatch); any
// failure activates the automatic GGML_VK_DISABLE_COOPMAT=1 fallback.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/device.hpp"

namespace sonder::inference {

enum class CoopmatCanaryStatus {
    not_applicable,        // No eligible AMD Windows GPU (or matrix_cores == "none")
    skipped_env_disabled,  // Operator already set GGML_VK_DISABLE_COOPMAT
    passed,                // All eligible AMD Windows GPUs passed (live or cached)
    failed,                // At least one eligible GPU crashed, hung, or failed verification
    error,                 // Canary runner could not execute (e.g. helper executable missing)
};

const char* to_string(CoopmatCanaryStatus status) noexcept;

struct CoopmatCanaryKey {
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::string driver_build;
    std::uint32_t ubatch = 64u;
};

struct CoopmatCanaryRecord {
    CoopmatCanaryKey key;
    std::string gpu_id;
    std::string gpu_name;
    bool is_integrated_uma = false;
    CoopmatCanaryStatus status = CoopmatCanaryStatus::not_applicable;
    bool disable_coopmat = false;
    bool from_cache = false;
    std::uint64_t elapsed_ms = 0;
    std::uint32_t exit_code = 0;
    std::string reason;
};

struct CoopmatCanaryDecision {
    CoopmatCanaryStatus status = CoopmatCanaryStatus::not_applicable;
    bool disable_coopmat = false;
    bool env_applied = false;
    std::string reason;
    std::vector<CoopmatCanaryRecord> records;
};

struct CoopmatCanaryProbeResult {
    bool ok = false;
    bool coopmat_supported = false;
    std::uint32_t shapes_found = 0;
    std::uint32_t ubatch_tested = 0;
    std::string reason;
};

struct CoopmatCanaryChildOutcome {
    bool completed = false;   // true if child exited normally with exit_code
    bool timed_out = false;   // true if child exceeded timeout and was killed
    bool spawn_error = false; // true if child could not be spawned (not cached)
    std::uint32_t exit_code = 0;
    std::uint64_t elapsed_ms = 0;
    std::string reason;
};

// Injectable child-process runner seam for deterministic unit testing.
using CanaryChildRunner =
    std::function<CoopmatCanaryChildOutcome(const CoopmatCanaryKey& key, std::chrono::milliseconds timeout)>;

struct CoopmatCanaryOptions {
#if defined(_WIN32)
    bool is_windows = true;
#else
    bool is_windows = false;
#endif
    // When nullopt, reads the process GGML_VK_DISABLE_COOPMAT environment variable.
    // Set explicitly in tests to avoid mutating or depending on host environment.
    std::optional<std::string> existing_disable_coopmat_env;
    // Persistent JSON cache file path. Empty + use_default_cache_path == false
    // disables disk caching.
    std::string cache_path;
    bool use_default_cache_path = true;
    // Explicit path to the `sonder-infer` binary used to run `coopmat-canary`.
    // Empty auto-discovers the current or sibling `sonder-infer` executable.
    std::string executable_path;
    // Tile-grid row count exercised by the generic coopmat compute smoke test (ceil(ubatch/16) tiles).
    std::uint32_t ubatch = 64u;
    std::chrono::milliseconds timeout{15000};
    // When true and disable_coopmat is triggered, sets GGML_VK_DISABLE_COOPMAT=1
    // in the current process environment.
    bool apply_env_fallback = true;
    // When true, a spawn/runner error fails safe by disabling coopmat (without caching).
    bool fail_closed_on_runner_error = true;
    // Optional injectable child runner (null uses the real isolated child spawner).
    CanaryChildRunner child_runner;
};

// True for AMD GPUs (both discrete e.g. RX 9060 XT and integrated UMA e.g. Radeon 780M)
// on Windows when Vulkan cooperative matrix is not already known absent ("none").
[[nodiscard]] bool needs_amd_windows_coopmat_canary(const GpuDeviceInfo& gpu, bool is_windows) noexcept;

// Deterministic, sanitized cache key combining vendor_id, device_id, ubatch, and driver_build.
[[nodiscard]] std::string canonical_coopmat_cache_key(const CoopmatCanaryKey& key);

// Platform default cache file path (e.g. %LOCALAPPDATA%/sonder-inference/coopmat-canary-v1.json).
[[nodiscard]] std::string default_coopmat_canary_cache_path();

// Resolves the `sonder-infer` executable used to launch the isolated canary child
// (checks explicit_path, SONDER_INFER_EXE, current executable if sonder-infer*, or sibling).
[[nodiscard]] std::string find_coopmat_canary_executable(const std::string& explicit_path = {});

// Strict, bounded persistent JSON cache I/O (rejects symlinks, `..` traversal,
// oversized or corrupt files; fails closed to an empty vector on load errors).
[[nodiscard]] std::vector<CoopmatCanaryRecord> load_coopmat_canary_cache(const std::string& path);
[[nodiscard]] bool save_coopmat_canary_cache(const std::string& path,
                                             const std::vector<CoopmatCanaryRecord>& entries);

// Executes the Vulkan cooperative-matrix canary directly in the CURRENT process:
//   Phase 1 (#22750/#23508): vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
//   Phase 2: generic 16x16x16 FP16->FP32 OpCooperativeMatrixMulAddKHR compute
//            smoke test over ceil(ubatch/16) workgroups with exact result verification.
//            (Does not reproduce upstream flash-attention coopmat1 flip-direct bugs.)
// Call ONLY from an isolated child process (`sonder-infer coopmat-canary`).
[[nodiscard]] CoopmatCanaryProbeResult run_vulkan_coopmat_canary_in_process(std::uint32_t vendor_id,
                                                                            std::uint32_t device_id,
                                                                            std::uint32_t ubatch = 64u);

// Evaluates all eligible AMD Windows GPUs (consulting the driver-build-keyed
// cache first, spawning an isolated child process on cache miss) and applies
// GGML_VK_DISABLE_COOPMAT=1 automatically when any eligible GPU fails.
[[nodiscard]] CoopmatCanaryDecision evaluate_amd_coopmat_canary(const std::vector<GpuDeviceInfo>& gpus,
                                                                const CoopmatCanaryOptions& options = {});

// Process-wide record of the most recent coopmat canary decision (surfaced by
// llamacpp runtime_status().warnings when fallback is active).
void record_coopmat_canary_decision(const CoopmatCanaryDecision& decision);
[[nodiscard]] std::optional<CoopmatCanaryDecision> last_coopmat_canary_decision();
void clear_coopmat_canary_decision() noexcept;

}  // namespace sonder::inference
