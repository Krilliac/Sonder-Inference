// Sonder Inference - in-process GPU device health state machine for native llama.cpp.
#include "device_health.hpp"

#include <algorithm>
#include <future>
#include <ios>
#include <limits>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

namespace {

std::string AsciiLower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            out.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

struct BackendHealthBinding {
    const LlamaCppBackend* backend = nullptr;
    std::shared_ptr<DeviceHealthTracker> tracker;
    std::vector<std::string> gpu_devices;
};

struct BindingRegistry {
    std::mutex mutex;
    std::vector<BackendHealthBinding> entries;
};

BindingRegistry& GetBindingRegistry() {
    static BindingRegistry reg;
    return reg;
}

struct CoreBackendHealthBinding {
    const sonder::inference::Backend* backend = nullptr;
    std::shared_ptr<DeviceHealthTracker> tracker;
    std::function<void(std::shared_ptr<DeviceHealthTracker>)> on_rebind;
};

struct CoreBindingRegistry {
    std::mutex mutex;
    std::vector<CoreBackendHealthBinding> entries;
};

CoreBindingRegistry& GetCoreBindingRegistry() {
    static CoreBindingRegistry reg;
    return reg;
}

}  // namespace

std::string_view ToString(DeviceHealthState state) noexcept {
    switch (state) {
        case DeviceHealthState::kHealthy: return "healthy";
        case DeviceHealthState::kUnhealthy: return "unhealthy";
    }
    return "unknown";
}

std::string_view ToString(DeviceFaultReason reason) noexcept {
    switch (reason) {
        case DeviceFaultReason::kNone: return "none";
        case DeviceFaultReason::kDecodeAllocFailed: return "decode_alloc_failed";
        case DeviceFaultReason::kDecodeComputeFailed: return "decode_compute_failed";
        case DeviceFaultReason::kGgmlAllocFailed: return "ggml_alloc_failed";
        case DeviceFaultReason::kGgmlComputeFailed: return "ggml_compute_failed";
        case DeviceFaultReason::kDeviceLost: return "device_lost";
        case DeviceFaultReason::kDeviceOutOfMemory: return "device_out_of_memory";
        case DeviceFaultReason::kBackendDeviceError: return "backend_device_error";
    }
    return "backend_device_error";
}

std::string_view ToString(DeviceHealthErrorKind kind) noexcept {
    switch (kind) {
        case DeviceHealthErrorKind::kOk: return "ok";
        case DeviceHealthErrorKind::kDeviceUnhealthy: return kDeviceUnhealthyErrorCode;
        case DeviceHealthErrorKind::kInvalidDevice: return "invalid_device";
    }
    return "unknown";
}

DeviceFaultReason ClassifyDecodeStatus(std::int32_t rc) noexcept {
    // llama.cpp v1.0.0 (include/llama.h:961-980, 1071-1075; src/llama-context.cpp:1974-1977, 2393-2396):
    //   0 = success, 1 = no KV slot warning, 2 = aborted, >2 = warning, -1 = invalid batch.
    //  -2 = GGML_STATUS_ALLOC_FAILED, <= -3 = GGML_STATUS_FAILED or fatal backend error.
    if (rc >= -1) {
        return DeviceFaultReason::kNone;
    }
    if (rc == -2) {
        return DeviceFaultReason::kDecodeAllocFailed;
    }
    return DeviceFaultReason::kDecodeComputeFailed;
}

DeviceFaultReason ClassifyGgmlStatus(int ggml_status_code) noexcept {
    // ggml/include/ggml.h:185-190:
    //   GGML_STATUS_SUCCESS (0) and GGML_STATUS_ABORTED (1) are non-fault outcomes.
    //   GGML_STATUS_ALLOC_FAILED (-2) and GGML_STATUS_FAILED (-1) / unknown codes are backend faults.
    if (ggml_status_code == static_cast<int>(GgmlComputeStatus::kSuccess) ||
        ggml_status_code == static_cast<int>(GgmlComputeStatus::kAborted)) {
        return DeviceFaultReason::kNone;
    }
    if (ggml_status_code == static_cast<int>(GgmlComputeStatus::kAllocFailed)) {
        return DeviceFaultReason::kGgmlAllocFailed;
    }
    return DeviceFaultReason::kGgmlComputeFailed;
}

class VulkanResultErrorCategoryImpl final : public std::error_category {
public:
    const char* name() const noexcept override { return "vk::Result"; }
    std::string message(int ev) const override {
        return "vk::Result(" + std::to_string(ev) + ")";
    }
};

bool IsVulkanErrorCategory(const std::error_category& category) noexcept {
    if (&category == &std::generic_category() ||
        &category == &std::system_category() ||
        &category == &std::iostream_category() ||
        &category == &std::future_category()) {
        return false;
    }
    const char* name = category.name();
    if (name == nullptr) {
        return false;
    }
    const std::string_view sv(name);
    if (sv == "vk::Result" || sv == "vulkan") {
        return true;
    }
    constexpr std::string_view kSuffix = "::Result";
    return sv.size() > kSuffix.size() && sv.substr(sv.size() - kSuffix.size()) == kSuffix;
}

const std::error_category& VulkanResultErrorCategory() noexcept {
    static const VulkanResultErrorCategoryImpl kCategory;
    return kCategory;
}

DeviceFaultReason ClassifyBackendSystemError(int error_value) noexcept {
    if (error_value == kVkErrorDeviceLost) {
        return DeviceFaultReason::kDeviceLost;
    }
    if (error_value == kVkErrorOutOfDeviceMemory) {
        return DeviceFaultReason::kDeviceOutOfMemory;
    }
    return DeviceFaultReason::kBackendDeviceError;
}

DeviceFaultReason ClassifyBackendSystemError(const std::error_code& ec) noexcept {
    if (IsVulkanErrorCategory(ec.category())) {
        return ClassifyBackendSystemError(ec.value());
    }
    return DeviceFaultReason::kBackendDeviceError;
}

std::string DeviceHealthError::FormatMessage() const {
    if (kind == DeviceHealthErrorKind::kOk) {
        return {};
    }
    if (kind == DeviceHealthErrorKind::kInvalidDevice) {
        return "invalid device identifier";
    }
    return std::string(kDeviceUnhealthyErrorCode) + ": device '" + device + "' is unhealthy (reason=" +
           std::string(ToString(reason)) + ", status=" + std::to_string(raw_status) + ")";
}

sonder::inference::Status DeviceHealthError::ToInferenceStatus() const {
    if (kind == DeviceHealthErrorKind::kOk) {
        return sonder::inference::Status::success();
    }
    if (kind == DeviceHealthErrorKind::kInvalidDevice) {
        return sonder::inference::Status(sonder::inference::ErrorCode::invalid_argument, FormatMessage());
    }
    return sonder::inference::Status(sonder::inference::ErrorCode::unavailable, FormatMessage());
}

bool IsDeviceUnhealthyInferenceStatus(const sonder::inference::Status& status) noexcept {
    if (status.code() != sonder::inference::ErrorCode::unavailable) {
        return false;
    }
    constexpr std::string_view kPrefix = "device_unhealthy:";
    return status.message().rfind(kPrefix, 0) == 0;
}

DeviceHealthTracker::DeviceHealthTracker(std::string backend_name)
    : backend_name_(IsValidDeviceName(backend_name) ? std::move(backend_name) : std::string("llamacpp")) {}

bool DeviceHealthTracker::IsValidDeviceName(std::string_view device) noexcept {
    if (device.empty() || device.size() > kMaxDeviceNameBytes) {
        return false;
    }
    bool has_non_space = false;
    for (unsigned char ch : device) {
        if (ch < 0x20u || ch >= 0x7Fu) {
            return false;
        }
        if (ch != ' ') {
            has_non_space = true;
        }
    }
    return has_non_space;
}

bool DeviceHealthTracker::RegisterAlias(std::string_view alias, std::string_view canonical_device) {
    if (!IsValidDeviceName(alias) || !IsValidDeviceName(canonical_device)) {
        return false;
    }
    std::string alias_key = AsciiLower(alias);
    std::lock_guard<std::mutex> lock(mutex_);
    std::string canonical_key;
    std::string resolved_display = ResolveCanonicalDisplayLocked(canonical_device, canonical_key);
    if (alias_key == canonical_key) {
        return true;
    }
    for (auto& existing : aliases_) {
        if (existing.alias_key == alias_key) {
            existing.canonical_display = std::move(resolved_display);
            existing.canonical_key = std::move(canonical_key);
            return true;
        }
    }
    if (aliases_.size() >= kMaxDeviceAliases) {
        return false;
    }
    aliases_.push_back(AliasRecord{std::move(alias_key), std::move(resolved_display), std::move(canonical_key)});
    return true;
}

std::string DeviceHealthTracker::ResolveCanonicalDisplayLocked(std::string_view device, std::string& out_key) const {
    std::string key = AsciiLower(device);
    for (const auto& a : aliases_) {
        if (a.alias_key == key) {
            out_key = a.canonical_key;
            for (const auto& u : unhealthy_) {
                if (u.key == out_key) {
                    return u.display_name;
                }
            }
            return a.canonical_display;
        }
    }
    for (const auto& u : unhealthy_) {
        if (u.key == key) {
            out_key = u.key;
            return u.display_name;
        }
    }
    out_key = std::move(key);
    return std::string(device);
}

DeviceHealthError DeviceHealthTracker::CheckDevice(std::string_view device) const {
    if (!IsValidDeviceName(device)) {
        return DeviceHealthError{DeviceHealthErrorKind::kInvalidDevice, {}, DeviceFaultReason::kNone, 0};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key;
    const std::string display = ResolveCanonicalDisplayLocked(device, key);
    for (const auto& u : unhealthy_) {
        if (u.key == key) {
            return DeviceHealthError{DeviceHealthErrorKind::kDeviceUnhealthy, u.display_name, u.reason, u.raw_status};
        }
    }
    if (overflow_latch_) {
        return DeviceHealthError{DeviceHealthErrorKind::kDeviceUnhealthy, display, overflow_reason_, overflow_status_};
    }
    return DeviceHealthError{};
}

DeviceHealthError DeviceHealthTracker::CheckDevices(std::span<const std::string> devices) const {
    for (const std::string& dev : devices) {
        if (DeviceHealthError err = CheckDevice(dev); !err.ok()) {
            return err;
        }
    }
    return DeviceHealthError{};
}

sonder::inference::Status DeviceHealthTracker::CheckDevicesStatus(std::span<const std::string> devices) const {
    return CheckDevices(devices).ToInferenceStatus();
}

DeviceHealthEntry DeviceHealthTracker::GetDeviceState(std::string_view device) const {
    DeviceHealthEntry out;
    if (!IsValidDeviceName(device)) {
        return out;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key;
    out.device = ResolveCanonicalDisplayLocked(device, key);
    for (const auto& u : unhealthy_) {
        if (u.key == key) {
            out.device = u.display_name;
            out.state = DeviceHealthState::kUnhealthy;
            out.reason = u.reason;
            out.raw_status = u.raw_status;
            out.fault_count = u.fault_count;
            return out;
        }
    }
    if (overflow_latch_) {
        out.state = DeviceHealthState::kUnhealthy;
        out.reason = overflow_reason_;
        out.raw_status = overflow_status_;
        out.fault_count = 1;
    }
    return out;
}

bool DeviceHealthTracker::RecordFaultLocked(std::string_view device, DeviceFaultReason reason,
                                            std::int32_t raw_status) {
    if (reason == DeviceFaultReason::kNone || !IsValidDeviceName(device)) {
        return false;
    }
    std::string key;
    std::string display = ResolveCanonicalDisplayLocked(device, key);

    for (auto& u : unhealthy_) {
        if (u.key == key) {
            if (u.fault_count < std::numeric_limits<std::uint64_t>::max()) {
                ++u.fault_count;
            }
            return true;
        }
    }

    auto enqueue_event = [&](const std::string& dev_name, DeviceFaultReason r) {
        if (pending_events_.size() >= kMaxPendingDeviceEvents) {
            pending_events_.pop_front();
        }
        sonder::inference::json::Object attrs;
        attrs.set("backend", backend_name_);
        attrs.set("device", dev_name);
        attrs.set("reason", std::string(ToString(r)));
        pending_events_.emplace_back(std::string(kDeviceUnhealthyEventType), std::move(attrs));
    };

    if (unhealthy_.size() >= kMaxTrackedUnhealthyDevices) {
        if (!overflow_latch_) {
            overflow_latch_ = true;
            overflow_reason_ = reason;
            overflow_status_ = raw_status;
            enqueue_event(display, reason);
        }
        return true;
    }

    enqueue_event(display, reason);
    unhealthy_.push_back(UnhealthyRecord{std::move(key), std::move(display), reason, raw_status, 1u});
    return true;
}

bool DeviceHealthTracker::ObserveDecodeStatus(std::string_view device, std::int32_t rc) {
    const DeviceFaultReason reason = ClassifyDecodeStatus(rc);
    if (reason == DeviceFaultReason::kNone) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return RecordFaultLocked(device, reason, rc);
}

bool DeviceHealthTracker::ObserveDecodeStatusForDevices(std::span<const std::string> devices, std::int32_t rc) {
    const DeviceFaultReason reason = ClassifyDecodeStatus(rc);
    if (reason == DeviceFaultReason::kNone || devices.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    bool any = false;
    for (const std::string& dev : devices) {
        any = RecordFaultLocked(dev, reason, rc) || any;
    }
    return any;
}

bool DeviceHealthTracker::ObserveGgmlStatus(std::string_view device, int ggml_status_code) {
    const DeviceFaultReason reason = ClassifyGgmlStatus(ggml_status_code);
    if (reason == DeviceFaultReason::kNone) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return RecordFaultLocked(device, reason, static_cast<std::int32_t>(ggml_status_code));
}

bool DeviceHealthTracker::ObserveGgmlStatusForDevices(std::span<const std::string> devices, int ggml_status_code) {
    const DeviceFaultReason reason = ClassifyGgmlStatus(ggml_status_code);
    if (reason == DeviceFaultReason::kNone || devices.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    bool any = false;
    for (const std::string& dev : devices) {
        any = RecordFaultLocked(dev, reason, static_cast<std::int32_t>(ggml_status_code)) || any;
    }
    return any;
}

bool DeviceHealthTracker::ObserveSystemError(std::string_view device, int error_value) {
    const DeviceFaultReason reason = ClassifyBackendSystemError(error_value);
    std::lock_guard<std::mutex> lock(mutex_);
    return RecordFaultLocked(device, reason, static_cast<std::int32_t>(error_value));
}

bool DeviceHealthTracker::ObserveSystemError(std::string_view device, const std::error_code& ec) {
    if (IsVulkanErrorCategory(ec.category())) {
        return ObserveSystemError(device, ec.value());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return RecordFaultLocked(device, DeviceFaultReason::kBackendDeviceError, -1);
}

bool DeviceHealthTracker::ObserveSystemErrorForDevices(std::span<const std::string> devices, int error_value) {
    if (devices.empty()) {
        return false;
    }
    const DeviceFaultReason reason = ClassifyBackendSystemError(error_value);
    std::lock_guard<std::mutex> lock(mutex_);
    bool any = false;
    for (const std::string& dev : devices) {
        any = RecordFaultLocked(dev, reason, static_cast<std::int32_t>(error_value)) || any;
    }
    return any;
}

bool DeviceHealthTracker::ObserveSystemErrorForDevices(std::span<const std::string> devices,
                                                       const std::error_code& ec) {
    if (devices.empty()) {
        return false;
    }
    if (IsVulkanErrorCategory(ec.category())) {
        return ObserveSystemErrorForDevices(devices, ec.value());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    bool any = false;
    for (const std::string& dev : devices) {
        any = RecordFaultLocked(dev, DeviceFaultReason::kBackendDeviceError, -1) || any;
    }
    return any;
}

bool DeviceHealthTracker::MarkDeviceUnhealthy(std::string_view device, DeviceFaultReason reason,
                                              std::int32_t raw_status) {
    if (reason == DeviceFaultReason::kNone) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return RecordFaultLocked(device, reason, raw_status);
}

sonder::inference::Status DeviceHealthTracker::RecoverDevice(std::string_view device) {
    if (!IsValidDeviceName(device)) {
        return sonder::inference::Status(sonder::inference::ErrorCode::invalid_argument,
                                         "invalid device name for recovery");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (overflow_latch_) {
        return sonder::inference::Status(
            sonder::inference::ErrorCode::invalid_state,
            "device health table overflowed; full backend recovery or restart is required");
    }
    std::string key;
    (void)ResolveCanonicalDisplayLocked(device, key);
    unhealthy_.erase(std::remove_if(unhealthy_.begin(), unhealthy_.end(),
                                    [&key](const UnhealthyRecord& r) { return r.key == key; }),
                     unhealthy_.end());
    return sonder::inference::Status::success();
}

void DeviceHealthTracker::RecoverAll() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    unhealthy_.clear();
    overflow_latch_ = false;
    overflow_reason_ = DeviceFaultReason::kNone;
    overflow_status_ = 0;
}

std::vector<std::pair<std::string, sonder::inference::json::Object>> DeviceHealthTracker::DrainTelemetryEvents() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::string, sonder::inference::json::Object>> out;
    out.reserve(pending_events_.size());
    for (auto& ev : pending_events_) {
        out.push_back(std::move(ev));
    }
    pending_events_.clear();
    return out;
}

std::size_t DeviceHealthTracker::PendingTelemetryEventCount() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_events_.size();
}

void BindBackendDeviceHealth(const LlamaCppBackend* backend, std::shared_ptr<DeviceHealthTracker> tracker,
                             std::vector<std::string> gpu_devices) {
    if (backend == nullptr) return;
    auto& reg = GetBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (auto& entry : reg.entries) {
        if (entry.backend == backend) {
            entry.tracker = std::move(tracker);
            entry.gpu_devices = std::move(gpu_devices);
            return;
        }
    }
    reg.entries.push_back(BackendHealthBinding{backend, std::move(tracker), std::move(gpu_devices)});
}

void MoveBackendDeviceHealth(const LlamaCppBackend* from, const LlamaCppBackend* to) noexcept {
    if (from == nullptr || to == nullptr || from == to) return;
    auto& reg = GetBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.entries.erase(std::remove_if(reg.entries.begin(), reg.entries.end(),
                                     [to](const BackendHealthBinding& b) { return b.backend == to; }),
                      reg.entries.end());
    for (auto& entry : reg.entries) {
        if (entry.backend == from) {
            entry.backend = to;
            return;
        }
    }
}

void UnbindBackendDeviceHealth(const LlamaCppBackend* backend) noexcept {
    if (backend == nullptr) return;
    auto& reg = GetBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.entries.erase(std::remove_if(reg.entries.begin(), reg.entries.end(),
                                     [backend](const BackendHealthBinding& b) { return b.backend == backend; }),
                      reg.entries.end());
}

bool HasBoundBackendDeviceTracker(const LlamaCppBackend* backend) noexcept {
    if (backend == nullptr) return false;
    auto& reg = GetBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (const auto& entry : reg.entries) {
        if (entry.backend == backend) {
            return entry.tracker != nullptr && !entry.gpu_devices.empty();
        }
    }
    return false;
}

DeviceHealthError CheckBoundBackendDeviceHealth(const LlamaCppBackend* backend) {
    if (backend == nullptr) return {};
    std::shared_ptr<DeviceHealthTracker> tracker;
    std::vector<std::string> devices;
    {
        auto& reg = GetBindingRegistry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (const auto& entry : reg.entries) {
            if (entry.backend == backend) {
                tracker = entry.tracker;
                devices = entry.gpu_devices;
                break;
            }
        }
    }
    if (!tracker || devices.empty()) {
        return {};
    }
    return tracker->CheckDevices(devices);
}

std::int32_t RunBoundBackendDecode(const LlamaCppBackend* backend, const std::function<std::int32_t()>& fn) {
    if (!fn) return 0;
    std::shared_ptr<DeviceHealthTracker> tracker;
    std::vector<std::string> devices;
    if (backend != nullptr) {
        auto& reg = GetBindingRegistry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (const auto& entry : reg.entries) {
            if (entry.backend == backend) {
                tracker = entry.tracker;
                devices = entry.gpu_devices;
                break;
            }
        }
    }
    if (tracker && !devices.empty()) {
        std::int32_t rc = 0;
        (void)tracker->ExecuteDecodeGuarded(devices, fn, &rc);
        return rc;
    }
    return fn();
}

bool RecoverOrInvalidateAfterDecodeFailure(
    const LlamaCppBackend* backend,
    std::int32_t rc,
    const std::function<bool()>& try_rollback,
    const std::function<void()>& clear_cache) {
    if ((ClassifyDecodeStatus(rc) != DeviceFaultReason::kNone) &&
        HasBoundBackendDeviceTracker(backend) &&
        !CheckBoundBackendDeviceHealth(backend).ok()) {
        if (clear_cache) clear_cache();
        return false;
    }
    return try_rollback ? try_rollback() : false;
}

void AppendUniqueNonCpuDevice(std::vector<std::string>& out, std::string_view candidate) {
    if (candidate.empty() || AsciiLower(candidate) == "cpu") return;
    const std::string low = AsciiLower(candidate);
    for (const auto& existing : out) {
        if (AsciiLower(existing) == low) return;
    }
    out.emplace_back(candidate);
}

std::vector<std::string> CollectInitialGpuDevices(const LoadOptions& lo, std::string_view device_id) {
    std::vector<std::string> out;
    if (device_id.rfind("cpu", 0) != 0) {
        for (const auto& name : lo.devices) {
            AppendUniqueNonCpuDevice(out, name);
        }
        if (device_id.rfind("gpu", 0) == 0) {
            AppendUniqueNonCpuDevice(out, device_id);
        }
    }
    for (const auto& ov : lo.tensor_overrides) {
        AppendUniqueNonCpuDevice(out, ov.device);
    }
    return out;
}

std::vector<std::string> CollectActiveGpuDevices(const LoadOptions& lo,
                                                 std::span<const DeviceInfo> enumerated_devices,
                                                 std::string_view device_id) {
    std::vector<std::string> out;
    if (lo.n_gpu_layers != 0) {
        for (const auto& name : lo.devices) {
            AppendUniqueNonCpuDevice(out, name);
        }
        if (out.empty()) {
            std::vector<const DeviceInfo*> gpus;
            for (const auto& d : enumerated_devices) {
                if (d.kind == DeviceKind::kGpu || d.kind == DeviceKind::kIntegratedGpu ||
                    d.kind == DeviceKind::kAccelerator) {
                    gpus.push_back(&d);
                }
            }
            if (lo.split_mode == SplitMode::kNone && lo.main_gpu >= 0 &&
                static_cast<std::size_t>(lo.main_gpu) < gpus.size()) {
                AppendUniqueNonCpuDevice(out, gpus[static_cast<std::size_t>(lo.main_gpu)]->name);
            } else {
                for (const auto* g : gpus) {
                    AppendUniqueNonCpuDevice(out, g->name);
                }
            }
        }
        if (out.empty() && device_id.rfind("gpu", 0) == 0) {
            AppendUniqueNonCpuDevice(out, device_id);
        }
    }
    for (const auto& ov : lo.tensor_overrides) {
        AppendUniqueNonCpuDevice(out, ov.device);
    }
    return out;
}

void BindCoreBackendDeviceHealth(
    const sonder::inference::Backend* backend,
    std::shared_ptr<DeviceHealthTracker> tracker,
    std::function<void(std::shared_ptr<DeviceHealthTracker>)> on_rebind) {
    if (backend == nullptr) return;
    std::function<void(std::shared_ptr<DeviceHealthTracker>)> cb;
    {
        auto& reg = GetCoreBindingRegistry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (auto& entry : reg.entries) {
            if (entry.backend == backend) {
                entry.tracker = tracker;
                if (on_rebind) {
                    entry.on_rebind = std::move(on_rebind);
                } else {
                    cb = entry.on_rebind;
                }
                goto notify;
            }
        }
        reg.entries.push_back(CoreBackendHealthBinding{backend, std::move(tracker), std::move(on_rebind)});
        return;
    }
notify:
    if (cb) {
        cb(std::move(tracker));
    }
}

void UnbindCoreBackendDeviceHealth(const sonder::inference::Backend* backend) noexcept {
    if (backend == nullptr) return;
    auto& reg = GetCoreBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.entries.erase(std::remove_if(reg.entries.begin(), reg.entries.end(),
                                     [backend](const CoreBackendHealthBinding& b) { return b.backend == backend; }),
                      reg.entries.end());
}

std::shared_ptr<DeviceHealthTracker> LookupCoreBackendDeviceHealth(const sonder::inference::Backend* backend) {
    if (backend == nullptr) return nullptr;
    auto& reg = GetCoreBindingRegistry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (const auto& entry : reg.entries) {
        if (entry.backend == backend) {
            return entry.tracker;
        }
    }
    return nullptr;
}

sonder::inference::Status RecoverCoreBackendDevice(sonder::inference::Backend& backend, std::string_view device) {
    const auto tracker = LookupCoreBackendDeviceHealth(&backend);
    if (!tracker) {
        return sonder::inference::Status(sonder::inference::ErrorCode::invalid_argument,
                                         "backend is not a llamacpp backend");
    }
    return tracker->RecoverDevice(device);
}

}  // namespace sonder::backends::llamacpp
