// Sonder Inference - in-process GPU device health state machine for native llama.cpp.
//
// Upstream llama.cpp v1.0.0 (b1b770d) failure-surfacing contract:
//   1. ggml_status (ggml/include/ggml.h:185-190) returned by
//      ggml_backend_graph_compute / ggml_backend_sched_graph_compute_async
//      (ggml/include/ggml-backend.h:108-110, 378-379; ggml/src/ggml-backend.cpp:799-808, 2466-2481):
//        GGML_STATUS_ALLOC_FAILED = -2  -> backend buffer/graph allocation failed
//        GGML_STATUS_FAILED       = -1  -> backend graph compute failed
//        GGML_STATUS_SUCCESS      =  0  -> success
//        GGML_STATUS_ABORTED      =  1  -> cooperative abort callback (not a device fault)
//   2. llama_decode / llama_encode / llama_process return codes
//      (include/llama.h:961-980, 1071-1075; src/llama-context.cpp:1974-1977, 2393-2396, 5079-5099, 5167-5173):
//         0    -> success
//         1    -> no KV slot available for batch (capacity warning, not a device fault)
//         2    -> aborted (GGML_STATUS_ABORTED, not a device fault)
//        -1    -> invalid input batch (caller argument error, not a device fault)
//        -2    -> GGML_STATUS_ALLOC_FAILED (fatal backend allocation / OOM failure)
//        <= -3 -> GGML_STATUS_FAILED (-3) or fatal backend execution failure
//   3. Vulkan backend (ggml-vulkan at v1.0.0):
//      - OOM / buffer alloc failure: ggml_vk_create_buffer (ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp:57, 159-183)
//        throws vk::OutOfDeviceMemoryError / vk::SystemError; ggml_backend_vk_buffer_type_alloc_buffer
//        (ggml/src/ggml-vulkan/ggml-vulkan.cpp:13598-13608) catches vk::SystemError and returns nullptr,
//        which process_ubatch (src/llama-context.cpp:1861-1865) turns into GGML_STATUS_ALLOC_FAILED (rc = -2).
//      - Device lost: VK_CHECK (ggml/src/ggml-vulkan/ggml-vulkan-types.h:196-212), ggml_vk_wait_for_fence
//        (ggml-vulkan.cpp:355-366), ggml_backend_vk_graph_compute (ggml-vulkan.cpp:14739, 14847-14853),
//        and vk_queue_handle_*::submit (ggml-vulkan.cpp:16952, 16968) rethrow vk::DeviceLostError
//        (std::system_error with VkResult VK_ERROR_DEVICE_LOST = -4) across llama_decode / llama_process.
//   4. CUDA backend (ggml-cuda at v1.0.0):
//      - OOM / device alloc failure: ggml_cuda_device_malloc (ggml/src/ggml-cuda/ggml-cuda.cu:194-230, 542-545)
//        and ggml_backend_cuda_buffer_type_alloc_buffer (ggml-cuda.cu:1204-1210) clear cudaGetLastError() on
//        err != cudaSuccess (e.g. cudaErrorMemoryAllocation) and return nullptr -> GGML_STATUS_ALLOC_FAILED (rc = -2).
//      - Synchronous runtime / kernel checks invoke CUDA_CHECK (ggml/src/ggml-cuda/common.cuh:203-211) ->
//        ggml_cuda_error (ggml-cuda.cu:105-114) -> GGML_ABORT / ggml_abort (ggml/src/ggml.c:253-273, noreturn).
//
// Security invariant:
//   Text from prompts, logs, or exception what() messages is NEVER inspected or treated as a fault signal.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference {
class Backend;
}  // namespace sonder::inference

namespace sonder::backends::llamacpp {

class LlamaCppBackend;

inline constexpr std::string_view kDeviceUnhealthyErrorCode = "device_unhealthy";
inline constexpr std::string_view kDeviceUnhealthyEventType = "backend.device_unhealthy";
inline constexpr std::size_t kMaxDeviceNameBytes = 128;
inline constexpr std::size_t kMaxTrackedUnhealthyDevices = 256;
inline constexpr std::size_t kMaxDeviceAliases = 256;
inline constexpr std::size_t kMaxPendingDeviceEvents = 64;
inline constexpr std::int32_t kDeviceUnhealthyDecodeSentinel = -1000;

// Standard Khronos VkResult negative error codes surfaced via vk::SystemError.
inline constexpr int kVkErrorOutOfHostMemory = -1;
inline constexpr int kVkErrorOutOfDeviceMemory = -2;
inline constexpr int kVkErrorInitializationFailed = -3;
inline constexpr int kVkErrorDeviceLost = -4;

// Mirrors enum ggml_status in ggml/include/ggml.h:185-190 (verified via static_assert in llamacpp_backend.cpp).
enum class GgmlComputeStatus : int {
    kAllocFailed = -2,
    kFailed = -1,
    kSuccess = 0,
    kAborted = 1,
};

enum class DeviceHealthState : std::uint8_t {
    kHealthy = 0,
    kUnhealthy = 1,
};

enum class DeviceFaultReason : std::uint8_t {
    kNone = 0,
    kDecodeAllocFailed,
    kDecodeComputeFailed,
    kGgmlAllocFailed,
    kGgmlComputeFailed,
    kDeviceLost,
    kDeviceOutOfMemory,
    kBackendDeviceError,
};

enum class DeviceHealthErrorKind : std::uint8_t {
    kOk = 0,
    kDeviceUnhealthy = 1,
    kInvalidDevice = 2,
};

[[nodiscard]] std::string_view ToString(DeviceHealthState state) noexcept;
[[nodiscard]] std::string_view ToString(DeviceFaultReason reason) noexcept;
[[nodiscard]] std::string_view ToString(DeviceHealthErrorKind kind) noexcept;

// Pure in-process status classifiers (never inspect strings).
[[nodiscard]] DeviceFaultReason ClassifyDecodeStatus(std::int32_t rc) noexcept;
[[nodiscard]] DeviceFaultReason ClassifyGgmlStatus(int ggml_status_code) noexcept;
[[nodiscard]] inline DeviceFaultReason ClassifyGgmlStatus(GgmlComputeStatus status) noexcept {
    return ClassifyGgmlStatus(static_cast<int>(status));
}
[[nodiscard]] DeviceFaultReason ClassifyBackendSystemError(int error_value) noexcept;

struct DeviceHealthEntry {
    std::string device;
    DeviceHealthState state = DeviceHealthState::kHealthy;
    DeviceFaultReason reason = DeviceFaultReason::kNone;
    std::int32_t raw_status = 0;
    std::uint64_t fault_count = 0;
};

struct DeviceHealthError {
    DeviceHealthErrorKind kind = DeviceHealthErrorKind::kOk;
    std::string device;
    DeviceFaultReason reason = DeviceFaultReason::kNone;
    std::int32_t raw_status = 0;

    [[nodiscard]] bool ok() const noexcept { return kind == DeviceHealthErrorKind::kOk; }
    [[nodiscard]] bool is_device_unhealthy() const noexcept {
        return kind == DeviceHealthErrorKind::kDeviceUnhealthy;
    }
    [[nodiscard]] std::string_view code() const noexcept { return ToString(kind); }
    [[nodiscard]] std::string FormatMessage() const;
    [[nodiscard]] sonder::inference::Status ToInferenceStatus() const;
};

[[nodiscard]] bool IsDeviceUnhealthyInferenceStatus(const sonder::inference::Status& status) noexcept;

class DeviceHealthTracker {
public:
    explicit DeviceHealthTracker(std::string backend_name = "llamacpp");

    [[nodiscard]] static bool IsValidDeviceName(std::string_view device) noexcept;

    // Maps an alias such as "gpu:0" onto a canonical ggml device name such as "Vulkan0".
    bool RegisterAlias(std::string_view alias, std::string_view canonical_device);

    // Non-blocking fail-fast admission checks.
    [[nodiscard]] DeviceHealthError CheckDevice(std::string_view device) const;
    [[nodiscard]] DeviceHealthError CheckDevices(std::span<const std::string> devices) const;
    [[nodiscard]] sonder::inference::Status CheckDevicesStatus(std::span<const std::string> devices) const;
    [[nodiscard]] DeviceHealthEntry GetDeviceState(std::string_view device) const;

    // In-process status observers. Non-fault codes never clear an existing kUnhealthy state.
    // Returns true iff the status indicates a device fault.
    bool ObserveDecodeStatus(std::string_view device, std::int32_t rc);
    bool ObserveDecodeStatusForDevices(std::span<const std::string> devices, std::int32_t rc);
    bool ObserveGgmlStatus(std::string_view device, int ggml_status_code);
    bool ObserveGgmlStatusForDevices(std::span<const std::string> devices, int ggml_status_code);
    bool ObserveSystemError(std::string_view device, int error_value);
    bool ObserveSystemErrorForDevices(std::span<const std::string> devices, int error_value);
    bool MarkDeviceUnhealthy(std::string_view device, DeviceFaultReason reason, std::int32_t raw_status = -1);

    // Runs `fn()` (returning llama_decode / llama_process int32_t rc) only if every device in `devices`
    // is healthy; otherwise fails fast with kDeviceUnhealthy without calling `fn()`.
    template <class Fn>
    DeviceHealthError ExecuteDecodeGuarded(std::span<const std::string> devices, Fn&& fn,
                                           std::int32_t* out_rc = nullptr) {
        if (DeviceHealthError pre = CheckDevices(devices); !pre.ok()) {
            if (out_rc != nullptr) *out_rc = kDeviceUnhealthyDecodeSentinel;
            return pre;
        }
        std::int32_t rc = 0;
        try {
            rc = fn();
        } catch (const std::system_error& sys_err) {
            (void)ObserveSystemErrorForDevices(devices, sys_err.code().value());
            if (out_rc != nullptr) *out_rc = kDeviceUnhealthyDecodeSentinel;
            return CheckDevices(devices);
        } catch (...) {
            (void)ObserveSystemErrorForDevices(devices, -1);
            if (out_rc != nullptr) *out_rc = kDeviceUnhealthyDecodeSentinel;
            return CheckDevices(devices);
        }
        if (out_rc != nullptr) *out_rc = rc;
        if (ObserveDecodeStatusForDevices(devices, rc)) {
            return CheckDevices(devices);
        }
        return DeviceHealthError{};
    }

    // Explicit recovery (never triggered automatically).
    sonder::inference::Status RecoverDevice(std::string_view device);
    void RecoverAll() noexcept;

    // Drains queued ("backend.device_unhealthy", {backend, device, reason}) telemetry events.
    [[nodiscard]] std::vector<std::pair<std::string, sonder::inference::json::Object>> DrainTelemetryEvents();
    [[nodiscard]] std::size_t PendingTelemetryEventCount() const noexcept;

private:
    struct UnhealthyRecord {
        std::string key;
        std::string display_name;
        DeviceFaultReason reason = DeviceFaultReason::kNone;
        std::int32_t raw_status = 0;
        std::uint64_t fault_count = 0;
    };

    struct AliasRecord {
        std::string alias_key;
        std::string canonical_display;
        std::string canonical_key;
    };

    [[nodiscard]] std::string ResolveCanonicalDisplayLocked(std::string_view device, std::string& out_key) const;
    bool RecordFaultLocked(std::string_view device, DeviceFaultReason reason, std::int32_t raw_status);

    std::string backend_name_;
    mutable std::mutex mutex_;
    std::vector<UnhealthyRecord> unhealthy_;
    std::vector<AliasRecord> aliases_;
    std::deque<std::pair<std::string, sonder::inference::json::Object>> pending_events_;
    bool overflow_latch_ = false;
    DeviceFaultReason overflow_reason_ = DeviceFaultReason::kNone;
    std::int32_t overflow_status_ = 0;
};

// Binding registry connecting LlamaCppBackend instances to a DeviceHealthTracker without
// exposing internal tracker types in the public llamacpp_backend.h header.
void BindBackendDeviceHealth(const LlamaCppBackend* backend,
                             std::shared_ptr<DeviceHealthTracker> tracker,
                             std::vector<std::string> gpu_devices);
void MoveBackendDeviceHealth(const LlamaCppBackend* from, const LlamaCppBackend* to) noexcept;
void UnbindBackendDeviceHealth(const LlamaCppBackend* backend) noexcept;
[[nodiscard]] DeviceHealthError CheckBoundBackendDeviceHealth(const LlamaCppBackend* backend);
[[nodiscard]] std::int32_t RunBoundBackendDecode(
    const LlamaCppBackend* backend, const std::function<std::int32_t()>& fn);

// Core Backend instance registry so unit/verifier tests and recover_llamacpp_device() work
// without widening Backend virtual methods or changing existing test factory signatures.
void BindCoreBackendDeviceHealth(
    const sonder::inference::Backend* backend,
    std::shared_ptr<DeviceHealthTracker> tracker,
    std::function<void(std::shared_ptr<DeviceHealthTracker>)> on_rebind = {});
void UnbindCoreBackendDeviceHealth(const sonder::inference::Backend* backend) noexcept;
[[nodiscard]] std::shared_ptr<DeviceHealthTracker> LookupCoreBackendDeviceHealth(
    const sonder::inference::Backend* backend);
sonder::inference::Status RecoverCoreBackendDevice(sonder::inference::Backend& backend, std::string_view device);

}  // namespace sonder::backends::llamacpp
