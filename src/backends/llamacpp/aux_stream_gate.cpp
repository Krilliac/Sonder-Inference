// In-process CUDA stream priorities, primary-context external EP stream helper
// (`user_compute_stream`), and Deficit Round Robin (DRR) decode-aware auxiliary gate.
#include "sonder/backends/llamacpp/aux_stream_gate.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#  include <dlfcn.h>
#endif

#include "embed_batch_planner.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder_cuda_prio_state.hpp"

namespace sonder::backends::llamacpp {

namespace {

bool EqualsAsciiIgnoreCase(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

std::string ToLowerAscii(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Extracts trailing decimal GPU ordinal from vendor backend names ("cuda0", "vulkan1",
// "rocm0", "hip2", "metal0", "sycl0", "musa0", "cann0", "opencl0", "webgpu0", "gpu:0")
// so backends referencing the same GPU ordinal canonicalize to the same domain token.
std::string CanonicalizeDeviceDomainToken(std::string_view raw_dev) {
    std::string lower = ToLowerAscii(raw_dev);
    if (lower.empty() || lower == "cpu") return "cpu";
    if (lower.rfind("gpu:", 0) == 0 && lower.size() > 4u) {
        const std::string_view tail(lower.data() + 4u, lower.size() - 4u);
        const bool all_digits = std::all_of(tail.begin(), tail.end(),
                                            [](unsigned char c) { return std::isdigit(c) != 0; });
        if (all_digits) return std::string(tail);
        return lower.substr(4u);
    }
    constexpr std::string_view kKnownPrefixes[] = {
        "cuda", "vulkan", "rocm", "hip", "metal", "sycl", "musa", "cann", "opencl", "webgpu"};
    for (std::string_view pfx : kKnownPrefixes) {
        if (lower.rfind(pfx, 0) == 0 && lower.size() > pfx.size()) {
            const std::string_view tail(lower.data() + pfx.size(), lower.size() - pfx.size());
            const bool all_digits = std::all_of(tail.begin(), tail.end(),
                                                [](unsigned char c) { return std::isdigit(c) != 0; });
            if (all_digits) {
                // Strip leading zeros except single "0".
                std::size_t first_nz = 0;
                while (first_nz + 1u < tail.size() && tail[first_nz] == '0') ++first_nz;
                return std::string(tail.substr(first_nz));
            }
        }
    }
    return lower;
}

bool IsValidStreamPriorityValue(StreamPriority p) noexcept {
    switch (p) {
        case StreamPriority::kDefault:
        case StreamPriority::kHigh:
        case StreamPriority::kLow: return true;
    }
    return false;
}

int ToHintCode(StreamPriority p) noexcept {
    switch (p) {
        case StreamPriority::kHigh: return SONDER_CUDA_STREAM_PRIO_HIGH;
        case StreamPriority::kLow: return SONDER_CUDA_STREAM_PRIO_LOW;
        case StreamPriority::kDefault:
        default: return SONDER_CUDA_STREAM_PRIO_DEFAULT;
    }
}

std::int64_t DefaultSteadyClockNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::int64_t SaturatingAddNs(std::int64_t a, std::int64_t b) noexcept {
    if (b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return a + b;
}

constexpr std::int64_t kMinCreditFloorNs = -60'000'000'000LL;  // -60s floor prevents underflow
constexpr std::uint64_t kMaxSttTotalEncoderTokens = 1ull << 20;

struct TransparentStringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
    std::size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

std::mutex& GetSyncObserverMutex() {
    static std::mutex mu;
    return mu;
}

AuxContextSyncObserver& GetSyncObserverSlot() {
    static AuxContextSyncObserver obs;
    return obs;
}

#if defined(_WIN32)
template <typename Fn>
Fn ResolveWinProc(HMODULE mod, const char* primary, const char* fallback = nullptr) noexcept {
    if (mod == nullptr) return nullptr;
    FARPROC p = ::GetProcAddress(mod, primary);
    if (p == nullptr && fallback != nullptr) {
        p = ::GetProcAddress(mod, fallback);
    }
    return reinterpret_cast<Fn>(p);
}
#elif defined(__linux__) || defined(__APPLE__)
template <typename Fn>
Fn ResolvePosixSym(void* handle, const char* primary, const char* fallback = nullptr) noexcept {
    if (handle == nullptr) return nullptr;
    void* p = ::dlsym(handle, primary);
    if (p == nullptr && fallback != nullptr) {
        p = ::dlsym(handle, fallback);
    }
    return reinterpret_cast<Fn>(p);
}
#endif

}  // namespace

std::string_view ToString(StreamPriority priority) noexcept {
    switch (priority) {
        case StreamPriority::kDefault: return "default";
        case StreamPriority::kHigh: return "high";
        case StreamPriority::kLow: return "low";
    }
    return "unknown";
}

bool ParseStreamPriority(std::string_view name, StreamPriority& out) noexcept {
    if (name.empty() || name.find('\0') != std::string_view::npos) return false;
    if (EqualsAsciiIgnoreCase(name, "default") || EqualsAsciiIgnoreCase(name, "normal")) {
        out = StreamPriority::kDefault;
        return true;
    }
    if (EqualsAsciiIgnoreCase(name, "high")) {
        out = StreamPriority::kHigh;
        return true;
    }
    if (EqualsAsciiIgnoreCase(name, "low")) {
        out = StreamPriority::kLow;
        return true;
    }
    return false;
}

std::string_view ToString(WorkloadRole role) noexcept {
    switch (role) {
        case WorkloadRole::kMainDecode: return "main_decode";
        case WorkloadRole::kAuxiliary: return "auxiliary";
    }
    return "unknown";
}

bool ParseWorkloadRole(std::string_view name, WorkloadRole& out) noexcept {
    if (name.empty() || name.find('\0') != std::string_view::npos) return false;
    if (EqualsAsciiIgnoreCase(name, "main_decode") || EqualsAsciiIgnoreCase(name, "main-decode") ||
        EqualsAsciiIgnoreCase(name, "main") || EqualsAsciiIgnoreCase(name, "decode")) {
        out = WorkloadRole::kMainDecode;
        return true;
    }
    if (EqualsAsciiIgnoreCase(name, "auxiliary") || EqualsAsciiIgnoreCase(name, "aux")) {
        out = WorkloadRole::kAuxiliary;
        return true;
    }
    return false;
}

int ResolveCudaNumericStreamPriority(int least_priority, int greatest_priority,
                                     StreamPriority requested) noexcept {
    const int hi = std::min(least_priority, greatest_priority);
    const int lo = std::max(least_priority, greatest_priority);
    switch (requested) {
        case StreamPriority::kHigh: return hi;
        case StreamPriority::kLow: return lo;
        case StreamPriority::kDefault:
        default: return std::clamp(0, hi, lo);
    }
}

StreamPriority EffectiveMainStreamPriority(WorkloadRole role, StreamPriority main_configured,
                                           StreamPriority aux_configured) noexcept {
    if (role == WorkloadRole::kAuxiliary) {
        return aux_configured;
    }
    return main_configured;
}

GgmlCudaStreamPriorityStats GetGgmlCudaStreamPriorityStats() noexcept {
    SonderGgmlCudaStreamStats raw{};
    sonder_ggml_cuda_get_stream_stats(&raw);
    GgmlCudaStreamPriorityStats out;
    out.high_priority_scopes = raw.high_priority_scopes;
    out.low_priority_scopes = raw.low_priority_scopes;
    out.default_priority_scopes = raw.default_priority_scopes;
    out.priority_streams_created = raw.priority_streams_created;
    out.fallback_streams_created = raw.fallback_streams_created;
    out.last_device = raw.last_device;
    out.last_numeric_priority = raw.last_numeric_priority;
    out.preinclude_hook_compiled = (raw.preinclude_hook_compiled != 0);
    return out;
}

void ResetGgmlCudaStreamPriorityStatsForTest() noexcept {
    sonder_ggml_cuda_reset_stream_stats();
}

ScopedGgmlCudaStreamPriority::ScopedGgmlCudaStreamPriority(StreamPriority priority) noexcept {
    const int hint = ToHintCode(priority);
    sonder_ggml_cuda_record_hint_scope(hint);
    prev_hint_ = sonder_ggml_cuda_set_stream_priority_hint(hint);
}

ScopedGgmlCudaStreamPriority::~ScopedGgmlCudaStreamPriority() noexcept {
    (void)sonder_ggml_cuda_set_stream_priority_hint(prev_hint_);
}

bool CudaDriverApi::complete() const noexcept {
    return init != nullptr && device_get_count != nullptr && device_get != nullptr &&
           primary_ctx_retain != nullptr && primary_ctx_release != nullptr &&
           ctx_push_current != nullptr && ctx_pop_current != nullptr &&
           ctx_get_stream_priority_range != nullptr && stream_create_with_priority != nullptr &&
           stream_get_priority != nullptr && stream_synchronize != nullptr && stream_destroy != nullptr;
}

const CudaDriverApi* GetSystemCudaDriverApi() noexcept {
    static std::once_flag once;
    static CudaDriverApi table{};
    static bool available = false;

    std::call_once(once, []() noexcept {
#if defined(_WIN32)
        HMODULE mod = ::LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (mod == nullptr) return;
        table.init = ResolveWinProc<decltype(table.init)>(mod, "cuInit");
        table.device_get_count = ResolveWinProc<decltype(table.device_get_count)>(mod, "cuDeviceGetCount");
        table.device_get = ResolveWinProc<decltype(table.device_get)>(mod, "cuDeviceGet");
        table.primary_ctx_retain =
            ResolveWinProc<decltype(table.primary_ctx_retain)>(mod, "cuDevicePrimaryCtxRetain");
        table.primary_ctx_release = ResolveWinProc<decltype(table.primary_ctx_release)>(
            mod, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
        table.ctx_push_current =
            ResolveWinProc<decltype(table.ctx_push_current)>(mod, "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
        table.ctx_pop_current =
            ResolveWinProc<decltype(table.ctx_pop_current)>(mod, "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
        table.ctx_get_stream_priority_range =
            ResolveWinProc<decltype(table.ctx_get_stream_priority_range)>(mod, "cuCtxGetStreamPriorityRange");
        table.stream_create_with_priority =
            ResolveWinProc<decltype(table.stream_create_with_priority)>(mod, "cuStreamCreateWithPriority");
        table.stream_get_priority =
            ResolveWinProc<decltype(table.stream_get_priority)>(mod, "cuStreamGetPriority");
        table.stream_synchronize =
            ResolveWinProc<decltype(table.stream_synchronize)>(mod, "cuStreamSynchronize");
        table.stream_destroy =
            ResolveWinProc<decltype(table.stream_destroy)>(mod, "cuStreamDestroy_v2", "cuStreamDestroy");
        available = table.complete();
#elif defined(__linux__)
        void* handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) return;
        table.init = ResolvePosixSym<decltype(table.init)>(handle, "cuInit");
        table.device_get_count = ResolvePosixSym<decltype(table.device_get_count)>(handle, "cuDeviceGetCount");
        table.device_get = ResolvePosixSym<decltype(table.device_get)>(handle, "cuDeviceGet");
        table.primary_ctx_retain =
            ResolvePosixSym<decltype(table.primary_ctx_retain)>(handle, "cuDevicePrimaryCtxRetain");
        table.primary_ctx_release = ResolvePosixSym<decltype(table.primary_ctx_release)>(
            handle, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
        table.ctx_push_current =
            ResolvePosixSym<decltype(table.ctx_push_current)>(handle, "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
        table.ctx_pop_current =
            ResolvePosixSym<decltype(table.ctx_pop_current)>(handle, "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
        table.ctx_get_stream_priority_range =
            ResolvePosixSym<decltype(table.ctx_get_stream_priority_range)>(handle, "cuCtxGetStreamPriorityRange");
        table.stream_create_with_priority =
            ResolvePosixSym<decltype(table.stream_create_with_priority)>(handle, "cuStreamCreateWithPriority");
        table.stream_get_priority =
            ResolvePosixSym<decltype(table.stream_get_priority)>(handle, "cuStreamGetPriority");
        table.stream_synchronize =
            ResolvePosixSym<decltype(table.stream_synchronize)>(handle, "cuStreamSynchronize");
        table.stream_destroy =
            ResolvePosixSym<decltype(table.stream_destroy)>(handle, "cuStreamDestroy_v2", "cuStreamDestroy");
        available = table.complete();
#endif
    });

    return available ? &table : nullptr;
}

std::optional<std::string_view> TrtRtxProviderOptions::Find(std::string_view key) const noexcept {
    for (const auto& [k, v] : entries) {
        if (k == key) return std::string_view(v);
    }
    return std::nullopt;
}

CudaPriorityStream::~CudaPriorityStream() noexcept { Reset(); }

CudaPriorityStream::CudaPriorityStream(CudaPriorityStream&& other) noexcept
    : api_(std::exchange(other.api_, nullptr)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)),
      cu_device_(std::exchange(other.cu_device_, 0)),
      primary_ctx_(std::exchange(other.primary_ctx_, nullptr)),
      stream_(std::exchange(other.stream_, nullptr)),
      requested_priority_(std::exchange(other.requested_priority_, StreamPriority::kDefault)),
      least_priority_(std::exchange(other.least_priority_, 0)),
      greatest_priority_(std::exchange(other.greatest_priority_, 0)),
      numeric_priority_(std::exchange(other.numeric_priority_, 0)) {}

CudaPriorityStream& CudaPriorityStream::operator=(CudaPriorityStream&& other) noexcept {
    if (this != &other) {
        Reset();
        api_ = std::exchange(other.api_, nullptr);
        device_ordinal_ = std::exchange(other.device_ordinal_, -1);
        cu_device_ = std::exchange(other.cu_device_, 0);
        primary_ctx_ = std::exchange(other.primary_ctx_, nullptr);
        stream_ = std::exchange(other.stream_, nullptr);
        requested_priority_ = std::exchange(other.requested_priority_, StreamPriority::kDefault);
        least_priority_ = std::exchange(other.least_priority_, 0);
        greatest_priority_ = std::exchange(other.greatest_priority_, 0);
        numeric_priority_ = std::exchange(other.numeric_priority_, 0);
    }
    return *this;
}

Status CudaPriorityStream::Create(int device_ordinal, StreamPriority priority, CudaPriorityStream& out,
                                  const CudaDriverApi* custom_api) {
    out.Reset();
    if (device_ordinal < 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "CUDA device_ordinal must be >= 0");
    }
    if (!IsValidStreamPriorityValue(priority)) {
        return Status::Error(ErrorCode::kInvalidArgument, "invalid CUDA StreamPriority value");
    }

    const CudaDriverApi* api = custom_api != nullptr ? custom_api : GetSystemCudaDriverApi();
    if (api == nullptr || !api->complete()) {
        return Status::Error(ErrorCode::kUnsupported, "CUDA Driver API is unavailable or incomplete");
    }
    if (const auto rc = api->init(0u); rc != CudaDriverApi::kSuccess) {
        return Status::Error(ErrorCode::kUnsupported, "cuInit failed (rc=" + std::to_string(rc) + ")");
    }
    int count = 0;
    if (const auto rc = api->device_get_count(&count); rc != CudaDriverApi::kSuccess || count <= 0) {
        return Status::Error(ErrorCode::kUnsupported, "no CUDA devices available");
    }
    if (device_ordinal >= count) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "CUDA device_ordinal (" + std::to_string(device_ordinal) + ") >= device count (" +
                                 std::to_string(count) + ")");
    }

    CudaDriverApi::CuDevice dev = 0;
    if (const auto rc = api->device_get(&dev, device_ordinal); rc != CudaDriverApi::kSuccess) {
        return Status::Error(ErrorCode::kInvalidArgument, "cuDeviceGet failed (rc=" + std::to_string(rc) + ")");
    }

    CudaDriverApi::CuContext pctx = nullptr;
    if (const auto rc = api->primary_ctx_retain(&pctx, dev);
        rc != CudaDriverApi::kSuccess || pctx == nullptr) {
        return Status::Error(ErrorCode::kDecodeFailed,
                             "cuDevicePrimaryCtxRetain failed (rc=" + std::to_string(rc) + ")");
    }

    struct PrimaryCtxGuard {
        const CudaDriverApi* api = nullptr;
        CudaDriverApi::CuDevice dev = 0;
        bool committed = false;
        ~PrimaryCtxGuard() noexcept {
            if (!committed && api != nullptr && api->primary_ctx_release != nullptr) {
                (void)api->primary_ctx_release(dev);
            }
        }
    } retain_guard{api, dev, false};

    if (const auto rc = api->ctx_push_current(pctx); rc != CudaDriverApi::kSuccess) {
        return Status::Error(ErrorCode::kDecodeFailed, "cuCtxPushCurrent failed (rc=" + std::to_string(rc) + ")");
    }

    struct PopCtxGuard {
        const CudaDriverApi* api = nullptr;
        ~PopCtxGuard() noexcept {
            if (api != nullptr && api->ctx_pop_current != nullptr) {
                CudaDriverApi::CuContext popped = nullptr;
                (void)api->ctx_pop_current(&popped);
            }
        }
    } pop_guard{api};

    int least = 0;
    int greatest = 0;
    if (api->ctx_get_stream_priority_range(&least, &greatest) != CudaDriverApi::kSuccess) {
        least = 0;
        greatest = 0;
    }
    const int target_prio = ResolveCudaNumericStreamPriority(least, greatest, priority);

    CudaDriverApi::CuStream stream = nullptr;
    if (const auto rc = api->stream_create_with_priority(&stream, CudaDriverApi::kStreamNonBlocking, target_prio);
        rc != CudaDriverApi::kSuccess || stream == nullptr) {
        return Status::Error(ErrorCode::kDecodeFailed,
                             "cuStreamCreateWithPriority failed (rc=" + std::to_string(rc) + ")");
    }

    int actual_prio = target_prio;
    if (api->stream_get_priority(stream, &actual_prio) != CudaDriverApi::kSuccess) {
        actual_prio = target_prio;
    }

    retain_guard.committed = true;
    out.api_ = api;
    out.device_ordinal_ = device_ordinal;
    out.cu_device_ = dev;
    out.primary_ctx_ = pctx;
    out.stream_ = stream;
    out.requested_priority_ = priority;
    out.least_priority_ = least;
    out.greatest_priority_ = greatest;
    out.numeric_priority_ = actual_prio;
    return Status::Ok();
}

void CudaPriorityStream::Reset() noexcept {
    if (api_ != nullptr) {
        if (stream_ != nullptr) {
            const bool pushed =
                (primary_ctx_ != nullptr && api_->ctx_push_current != nullptr &&
                 api_->ctx_push_current(primary_ctx_) == CudaDriverApi::kSuccess);
            if (api_->stream_destroy != nullptr) {
                (void)api_->stream_destroy(stream_);
            }
            if (pushed && api_->ctx_pop_current != nullptr) {
                CudaDriverApi::CuContext popped = nullptr;
                (void)api_->ctx_pop_current(&popped);
            }
            stream_ = nullptr;
        }
        if (primary_ctx_ != nullptr) {
            if (api_->primary_ctx_release != nullptr) {
                (void)api_->primary_ctx_release(cu_device_);
            }
            primary_ctx_ = nullptr;
        }
    }
    api_ = nullptr;
    device_ordinal_ = -1;
    cu_device_ = 0;
    requested_priority_ = StreamPriority::kDefault;
    least_priority_ = 0;
    greatest_priority_ = 0;
    numeric_priority_ = 0;
}

std::string CudaPriorityStream::FormatUserComputeStreamOption() const {
    return std::to_string(reinterpret_cast<std::uintptr_t>(stream_));
}

TrtRtxProviderOptions CudaPriorityStream::BuildTrtRtxProviderOptions(
    bool enable_cuda_graph, std::string_view runtime_cache_path) const {
    TrtRtxProviderOptions opts;
    opts.entries.reserve(runtime_cache_path.empty() ? 3u : 4u);
    opts.entries.emplace_back("device_id", std::to_string(std::max(0, device_ordinal_)));
    opts.entries.emplace_back("user_compute_stream", FormatUserComputeStreamOption());
    opts.entries.emplace_back("enable_cuda_graph", enable_cuda_graph ? "1" : "0");
    if (!runtime_cache_path.empty()) {
        opts.entries.emplace_back("nv_runtime_cache_path", std::string(runtime_cache_path));
    }
    return opts;
}

Status CudaPriorityStream::Synchronize() const {
    if (!valid() || api_ == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "CudaPriorityStream is not initialized");
    }
    const auto rc = api_->stream_synchronize(stream_);
    if (rc != CudaDriverApi::kSuccess) {
        return Status::Error(ErrorCode::kDecodeFailed,
                             "cuStreamSynchronize failed (rc=" + std::to_string(rc) + ")");
    }
    return Status::Ok();
}

Status ValidateAuxGateConfig(const AuxGateConfig& config) {
    if (!std::isfinite(config.max_aux_duty_cycle) || config.max_aux_duty_cycle <= 0.0 ||
        config.max_aux_duty_cycle > kMaxAllowedAuxDutyCycle) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "max_aux_duty_cycle must be finite and in (0.0, 0.50]");
    }
    if (config.active_slice_max_tokens < 1u || config.active_slice_max_tokens > kMaxAuxActiveSliceTokens) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "active_slice_max_tokens (" + std::to_string(config.active_slice_max_tokens) +
                                 ") must be in [1, " + std::to_string(kMaxAuxActiveSliceTokens) + "]");
    }
    if (config.active_slice_max_sequences < 1u || config.active_slice_max_sequences > kMaxEmbedSequences) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "active_slice_max_sequences (" + std::to_string(config.active_slice_max_sequences) +
                                 ") must be in [1, " + std::to_string(kMaxEmbedSequences) + "]");
    }
    if (config.max_decode_hold_ms < 1u || config.max_decode_hold_ms > kMaxAuxGateTimeoutMs) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "max_decode_hold_ms (" + std::to_string(config.max_decode_hold_ms) +
                                 ") must be in [1, " + std::to_string(kMaxAuxGateTimeoutMs) + "]");
    }
    if (config.handoff_pickup_timeout_ms < 1u || config.handoff_pickup_timeout_ms > kMaxAuxGateTimeoutMs) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "handoff_pickup_timeout_ms (" + std::to_string(config.handoff_pickup_timeout_ms) +
                                 ") must be in [1, " + std::to_string(kMaxAuxGateTimeoutMs) + "]");
    }
    if (config.max_starvation_wait_ms < 1u || config.max_starvation_wait_ms > kMaxAuxGateTimeoutMs) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "max_starvation_wait_ms (" + std::to_string(config.max_starvation_wait_ms) +
                                 ") must be in [1, " + std::to_string(kMaxAuxGateTimeoutMs) + "]");
    }
    if (config.cancel_poll_interval_ms < 1u || config.cancel_poll_interval_ms > kMaxAuxGateTimeoutMs) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "cancel_poll_interval_ms (" + std::to_string(config.cancel_poll_interval_ms) +
                                 ") must be in [1, " + std::to_string(kMaxAuxGateTimeoutMs) + "]");
    }
    return Status::Ok();
}

std::string ResolveAuxExecutionDomain(std::int32_t n_gpu_layers, std::int32_t main_gpu,
                                      const std::vector<std::string>& devices) {
    if (n_gpu_layers == 0) {
        return "cpu";
    }
    std::vector<std::string> tokens;
    tokens.reserve(devices.size());
    for (const std::string& d : devices) {
        if (d.empty()) continue;
        std::string tok = CanonicalizeDeviceDomainToken(d);
        if (tok == "cpu") continue;
        if (std::find(tokens.begin(), tokens.end(), tok) == tokens.end()) {
            tokens.push_back(std::move(tok));
        }
    }
    if (tokens.empty()) {
        for (const std::string& d : devices) {
            if (EqualsAsciiIgnoreCase(d, "cpu")) return "cpu";
        }
        return "gpu:" + std::to_string(std::max<std::int32_t>(0, main_gpu));
    }
    if (tokens.size() == 1u) {
        return "gpu:" + tokens[0];
    }
    std::string joined = "gpu:";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0u) joined.push_back('+');
        joined += tokens[i];
    }
    return joined;
}

struct DecodeAwareAuxGate::Impl {
    struct DomainState {
        mutable std::mutex mu;
        std::condition_variable cv;
        AuxGateStats stats{};
        std::int64_t aux_credit_ns = 0;
        bool handoff_armed = false;
        std::chrono::steady_clock::time_point handoff_deadline{};
    };

    mutable std::mutex table_mu;
    AuxGateConfig cfg{};
    ClockFn clock_fn;
    std::vector<std::unique_ptr<DomainState>> domains;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> domain_by_name;

    Impl(AuxGateConfig c, ClockFn clk)
        : cfg(ValidateAuxGateConfig(c).ok() ? c : AuxGateConfig{}),
          clock_fn(clk ? std::move(clk) : ClockFn(&DefaultSteadyClockNs)) {
        (void)GetOrCreateDomainIndex("default");
    }

    [[nodiscard]] std::int64_t NowNs() const noexcept {
        return clock_fn ? clock_fn() : DefaultSteadyClockNs();
    }

    [[nodiscard]] AuxGateConfig SnapshotConfig() const {
        std::lock_guard<std::mutex> lock(table_mu);
        return cfg;
    }

    [[nodiscard]] static std::string_view NormalizeDomainView(std::string_view raw) noexcept {
        return raw.empty() ? std::string_view("default") : raw;
    }

    [[nodiscard]] static bool IsBypassedCpuDomain(const AuxGateConfig& c, std::string_view dom_view) noexcept {
        return !c.gate_cpu_domain && dom_view == "cpu";
    }

    std::size_t GetOrCreateDomainIndex(std::string_view raw_domain) {
        const std::string_view key_view = NormalizeDomainView(raw_domain);
        std::lock_guard<std::mutex> lock(table_mu);
        const auto it = domain_by_name.find(key_view);
        if (it != domain_by_name.end()) {
            return it->second;
        }
        const std::size_t idx = domains.size();
        domains.push_back(std::make_unique<DomainState>());
        domain_by_name.emplace(std::string(key_view), idx);
        return idx;
    }

    [[nodiscard]] DomainState* FindDomain(std::string_view raw_domain) const {
        const std::string_view key_view = NormalizeDomainView(raw_domain);
        std::lock_guard<std::mutex> lock(table_mu);
        const auto it = domain_by_name.find(key_view);
        if (it == domain_by_name.end()) return nullptr;
        return domains[it->second].get();
    }
};

DecodeAwareAuxGate::DecodeAwareAuxGate(AuxGateConfig config, ClockFn clock_ns)
    : impl_(std::make_unique<Impl>(config, std::move(clock_ns))) {}

DecodeAwareAuxGate::~DecodeAwareAuxGate() = default;

Status DecodeAwareAuxGate::Reconfigure(const AuxGateConfig& config) {
    if (Status s = ValidateAuxGateConfig(config); !s.ok()) return s;
    std::vector<Impl::DomainState*> doms;
    {
        std::lock_guard<std::mutex> lock(impl_->table_mu);
        impl_->cfg = config;
        doms.reserve(impl_->domains.size());
        for (const auto& d : impl_->domains) {
            doms.push_back(d.get());
        }
    }
    for (Impl::DomainState* d : doms) {
        std::lock_guard<std::mutex> dlock(d->mu);
        d->cv.notify_all();
    }
    return Status::Ok();
}

AuxGateConfig DecodeAwareAuxGate::config() const { return impl_->SnapshotConfig(); }

bool DecodeAwareAuxGate::IsDecodeActive(std::string_view domain) const {
    if (!impl_) return false;
    const std::string_view key_view = Impl::NormalizeDomainView(domain);
    const AuxGateConfig c = impl_->SnapshotConfig();
    if (!c.enabled || Impl::IsBypassedCpuDomain(c, key_view)) return false;
    Impl::DomainState* dom = impl_->FindDomain(key_view);
    if (dom == nullptr) return false;
    std::lock_guard<std::mutex> lock(dom->mu);
    return dom->stats.active_decode_sessions > 0u || dom->stats.in_flight_decode_steps > 0u;
}

bool DecodeAwareAuxGate::IsDecodeStepInFlight(std::string_view domain) const {
    if (!impl_) return false;
    Impl::DomainState* dom = impl_->FindDomain(domain);
    if (dom == nullptr) return false;
    std::lock_guard<std::mutex> lock(dom->mu);
    return dom->stats.in_flight_decode_steps > 0u;
}

DecodeAwareAuxGate::DecodeSessionScope::DecodeSessionScope(DecodeAwareAuxGate* gate,
                                                           std::size_t domain_idx) noexcept
    : gate_(gate), domain_idx_(domain_idx) {}

DecodeAwareAuxGate::DecodeSessionScope::DecodeSessionScope(DecodeSessionScope&& other) noexcept
    : gate_(std::exchange(other.gate_, nullptr)), domain_idx_(other.domain_idx_) {}

DecodeAwareAuxGate::DecodeSessionScope& DecodeAwareAuxGate::DecodeSessionScope::operator=(
    DecodeSessionScope&& other) noexcept {
    if (this != &other) {
        Release();
        gate_ = std::exchange(other.gate_, nullptr);
        domain_idx_ = other.domain_idx_;
    }
    return *this;
}

DecodeAwareAuxGate::DecodeSessionScope::~DecodeSessionScope() noexcept { Release(); }

void DecodeAwareAuxGate::DecodeSessionScope::Release() noexcept {
    DecodeAwareAuxGate* g = std::exchange(gate_, nullptr);
    if (g == nullptr || !g->impl_) return;
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(g->impl_->table_mu);
        if (domain_idx_ < g->impl_->domains.size()) {
            dom = g->impl_->domains[domain_idx_].get();
        }
    }
    if (dom == nullptr) return;
    std::lock_guard<std::mutex> lock(dom->mu);
    if (dom->stats.active_decode_sessions > 0u) {
        --dom->stats.active_decode_sessions;
    }
    if (dom->stats.active_decode_sessions == 0u && dom->stats.in_flight_decode_steps == 0u) {
        dom->handoff_armed = false;
        dom->aux_credit_ns = 0;
    }
    dom->cv.notify_all();
}

DecodeAwareAuxGate::DecodeStepScope::DecodeStepScope(DecodeAwareAuxGate* gate, std::size_t domain_idx,
                                                     std::int64_t start_ns) noexcept
    : gate_(gate), domain_idx_(domain_idx), start_ns_(start_ns) {}

DecodeAwareAuxGate::DecodeStepScope::DecodeStepScope(DecodeStepScope&& other) noexcept
    : gate_(std::exchange(other.gate_, nullptr)),
      domain_idx_(other.domain_idx_),
      start_ns_(other.start_ns_),
      synced_(std::exchange(other.synced_, false)) {}

DecodeAwareAuxGate::DecodeStepScope& DecodeAwareAuxGate::DecodeStepScope::operator=(
    DecodeStepScope&& other) noexcept {
    if (this != &other) {
        Release();
        gate_ = std::exchange(other.gate_, nullptr);
        domain_idx_ = other.domain_idx_;
        start_ns_ = other.start_ns_;
        synced_ = std::exchange(other.synced_, false);
    }
    return *this;
}

DecodeAwareAuxGate::DecodeStepScope::~DecodeStepScope() noexcept { Release(); }

void DecodeAwareAuxGate::DecodeStepScope::MarkSynchronized() noexcept {
    if (gate_ != nullptr) {
        synced_ = true;
    }
}

void DecodeAwareAuxGate::DecodeStepScope::Release() noexcept {
    DecodeAwareAuxGate* g = std::exchange(gate_, nullptr);
    const bool was_synced = std::exchange(synced_, false);
    if (g == nullptr || !g->impl_) return;
    AuxGateConfig cfg{};
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(g->impl_->table_mu);
        cfg = g->impl_->cfg;
        if (domain_idx_ < g->impl_->domains.size()) {
            dom = g->impl_->domains[domain_idx_].get();
        }
    }
    if (dom == nullptr) return;

    const std::int64_t end_ns = g->impl_->NowNs();
    const std::int64_t step_ns = std::max<std::int64_t>(0, end_ns - start_ns_);

    std::lock_guard<std::mutex> lock(dom->mu);
    dom->stats.decode_busy_ns += static_cast<std::uint64_t>(step_ns);
    if (was_synced) {
        ++dom->stats.decode_steps_synced_before_release;
    }
    if (dom->stats.in_flight_decode_steps > 0u) {
        --dom->stats.in_flight_decode_steps;
    }

    if (cfg.enabled) {
        const double d = std::clamp(cfg.max_aux_duty_cycle, 1e-6, kMaxAllowedAuxDutyCycle);
        const double ratio = d / (1.0 - d);
        const auto proportional_ns = static_cast<std::int64_t>(std::llround(static_cast<double>(step_ns) * ratio));
        const std::int64_t add_ns = std::max<std::int64_t>(1LL, proportional_ns);
        // Cap positive credit at +1 ns so idle decode steps never pre-bank multi-slice bursts.
        dom->aux_credit_ns = std::min<std::int64_t>(1LL, SaturatingAddNs(dom->aux_credit_ns, add_ns));

        if (dom->stats.in_flight_decode_steps == 0u && dom->stats.waiting_aux_slices > 0u &&
            dom->aux_credit_ns > 0) {
            dom->handoff_armed = true;
            dom->handoff_deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.handoff_pickup_timeout_ms);
        }
    }
    dom->cv.notify_all();
}

DecodeAwareAuxGate::AuxSliceScope::AuxSliceScope(DecodeAwareAuxGate* gate, std::size_t domain_idx,
                                                 AuxAcquireStatus status, std::int64_t start_ns,
                                                 std::int64_t wait_ns) noexcept
    : gate_(gate), domain_idx_(domain_idx), status_(status), start_ns_(start_ns), wait_ns_(wait_ns) {}

DecodeAwareAuxGate::AuxSliceScope::AuxSliceScope(AuxSliceScope&& other) noexcept
    : gate_(std::exchange(other.gate_, nullptr)),
      domain_idx_(other.domain_idx_),
      status_(other.status_),
      start_ns_(other.start_ns_),
      wait_ns_(other.wait_ns_),
      synced_(std::exchange(other.synced_, false)) {}

DecodeAwareAuxGate::AuxSliceScope& DecodeAwareAuxGate::AuxSliceScope::operator=(
    AuxSliceScope&& other) noexcept {
    if (this != &other) {
        Release();
        gate_ = std::exchange(other.gate_, nullptr);
        domain_idx_ = other.domain_idx_;
        status_ = other.status_;
        start_ns_ = other.start_ns_;
        wait_ns_ = other.wait_ns_;
        synced_ = std::exchange(other.synced_, false);
    }
    return *this;
}

DecodeAwareAuxGate::AuxSliceScope::~AuxSliceScope() noexcept { Release(); }

void DecodeAwareAuxGate::AuxSliceScope::MarkSynchronized() noexcept {
    if (gate_ != nullptr && status_ != AuxAcquireStatus::kCancelled) {
        synced_ = true;
    }
}

void DecodeAwareAuxGate::AuxSliceScope::Release() noexcept {
    DecodeAwareAuxGate* g = std::exchange(gate_, nullptr);
    const bool was_synced = std::exchange(synced_, false);
    if (g == nullptr || !g->impl_ || status_ == AuxAcquireStatus::kCancelled) return;
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(g->impl_->table_mu);
        if (domain_idx_ < g->impl_->domains.size()) {
            dom = g->impl_->domains[domain_idx_].get();
        }
    }
    if (dom == nullptr) return;

    const std::int64_t end_ns = g->impl_->NowNs();
    const std::int64_t slice_ns = std::max<std::int64_t>(0, end_ns - start_ns_);

    std::lock_guard<std::mutex> lock(dom->mu);
    dom->stats.aux_busy_ns += static_cast<std::uint64_t>(slice_ns);
    if (was_synced) {
        ++dom->stats.aux_slices_synced_before_release;
    }
    if (dom->stats.in_flight_aux_slices > 0u) {
        --dom->stats.in_flight_aux_slices;
    }
    if (status_ == AuxAcquireStatus::kAcquiredBetweenSteps ||
        status_ == AuxAcquireStatus::kAcquiredStarvation) {
        const std::int64_t debit = std::max<std::int64_t>(1LL, slice_ns);
        dom->aux_credit_ns = std::max(kMinCreditFloorNs, SaturatingAddNs(dom->aux_credit_ns, -debit));
    }
    dom->cv.notify_all();
}

DecodeAwareAuxGate::DecodeSessionScope DecodeAwareAuxGate::EnterDecodeSession(std::string_view domain) {
    const std::string_view key_view = Impl::NormalizeDomainView(domain);
    const std::size_t idx = impl_->GetOrCreateDomainIndex(key_view);
    AuxGateConfig cfg{};
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(impl_->table_mu);
        cfg = impl_->cfg;
        dom = impl_->domains[idx].get();
    }
    if (Impl::IsBypassedCpuDomain(cfg, key_view)) {
        return DecodeSessionScope();
    }
    {
        std::lock_guard<std::mutex> lock(dom->mu);
        if (dom->stats.active_decode_sessions == 0u && dom->stats.in_flight_decode_steps == 0u) {
            dom->aux_credit_ns = 0;
            dom->handoff_armed = false;
        }
        ++dom->stats.active_decode_sessions;
        ++dom->stats.decode_sessions_total;
    }
    return DecodeSessionScope(this, idx);
}

DecodeAwareAuxGate::DecodeStepScope DecodeAwareAuxGate::EnterDecodeStep(std::string_view domain) {
    const std::string_view key_view = Impl::NormalizeDomainView(domain);
    const std::size_t idx = impl_->GetOrCreateDomainIndex(key_view);
    AuxGateConfig cfg{};
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(impl_->table_mu);
        cfg = impl_->cfg;
        dom = impl_->domains[idx].get();
    }
    if (Impl::IsBypassedCpuDomain(cfg, key_view)) {
        return DecodeStepScope();
    }

    std::unique_lock<std::mutex> lock(dom->mu);
    if (cfg.enabled) {
        const std::int64_t hold_begin_ns = impl_->NowNs();
        bool waited = false;

        // 1. Honor a freshly armed inter-step handoff reservation so a woken aux slice
        //    can claim its slot before the decode thread re-enters llama_decode.
        if (dom->handoff_armed && dom->stats.waiting_aux_slices > 0u &&
            dom->stats.in_flight_aux_slices == 0u) {
            waited = true;
            (void)dom->cv.wait_until(lock, dom->handoff_deadline, [&] {
                return !dom->handoff_armed || dom->stats.waiting_aux_slices == 0u ||
                       dom->stats.in_flight_aux_slices > 0u;
            });
            if (dom->stats.in_flight_aux_slices == 0u) {
                dom->handoff_armed = false;
            }
        }

        // 2. Wait for any in-flight AuxSliceScope (whose caller synchronizes device
        //    execution via llama_synchronize or CudaPriorityStream::Synchronize before
        //    releasing the scope) up to max_decode_hold_ms.
        if (dom->stats.in_flight_aux_slices > 0u) {
            waited = true;
            const auto hold_deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.max_decode_hold_ms);
            const bool drained = dom->cv.wait_until(lock, hold_deadline, [&] {
                return dom->stats.in_flight_aux_slices == 0u;
            });
            if (!drained && dom->stats.in_flight_aux_slices > 0u) {
                ++dom->stats.decode_hold_timeouts;
            }
        }

        if (waited) {
            const std::int64_t hold_ns = std::max<std::int64_t>(0, impl_->NowNs() - hold_begin_ns);
            dom->stats.decode_hold_wait_ns += static_cast<std::uint64_t>(hold_ns);
        }
    }

    dom->handoff_armed = false;
    ++dom->stats.in_flight_decode_steps;
    ++dom->stats.decode_steps_total;
    if (dom->stats.in_flight_aux_slices > 0u) {
        ++dom->stats.aux_slices_overlapped_decode;
    }
    return DecodeStepScope(this, idx, impl_->NowNs());
}

DecodeAwareAuxGate::AuxSliceScope DecodeAwareAuxGate::AcquireAuxSlice(
    std::string_view domain, const std::function<bool()>& should_cancel) {
    const std::string_view key_view = Impl::NormalizeDomainView(domain);
    const std::size_t idx = impl_->GetOrCreateDomainIndex(key_view);
    AuxGateConfig cfg{};
    Impl::DomainState* dom = nullptr;
    {
        std::lock_guard<std::mutex> tlock(impl_->table_mu);
        cfg = impl_->cfg;
        dom = impl_->domains[idx].get();
    }

    if (should_cancel && should_cancel()) {
        std::lock_guard<std::mutex> lock(dom->mu);
        ++dom->stats.aux_slices_cancelled;
        return AuxSliceScope();
    }

    std::unique_lock<std::mutex> lock(dom->mu);
    if (!cfg.enabled) {
        ++dom->stats.in_flight_aux_slices;
        ++dom->stats.aux_slices_total;
        if (dom->stats.in_flight_decode_steps > 0u) {
            ++dom->stats.aux_slices_overlapped_decode;
        }
        return AuxSliceScope(this, idx, AuxAcquireStatus::kBypassedDisabled, impl_->NowNs(), 0);
    }

    if (Impl::IsBypassedCpuDomain(cfg, key_view)) {
        ++dom->stats.in_flight_aux_slices;
        ++dom->stats.aux_slices_total;
        ++dom->stats.aux_slices_idle_fastpath;
        return AuxSliceScope(this, idx, AuxAcquireStatus::kAcquiredIdle, impl_->NowNs(), 0);
    }

    // Zero-wait idle fast path when no foreground decode session or step is active.
    if (dom->stats.active_decode_sessions == 0u && dom->stats.in_flight_decode_steps == 0u &&
        dom->stats.in_flight_aux_slices == 0u) {
        ++dom->stats.in_flight_aux_slices;
        ++dom->stats.aux_slices_total;
        ++dom->stats.aux_slices_idle_fastpath;
        return AuxSliceScope(this, idx, AuxAcquireStatus::kAcquiredIdle, impl_->NowNs(), 0);
    }

    const std::int64_t wait_begin_ns = impl_->NowNs();
    const auto steady_wait_begin = std::chrono::steady_clock::now();
    ++dom->stats.waiting_aux_slices;
    struct WaitingGuard {
        Impl::DomainState* dom = nullptr;
        ~WaitingGuard() noexcept {
            if (dom != nullptr && dom->stats.waiting_aux_slices > 0u) {
                --dom->stats.waiting_aux_slices;
                if (dom->stats.waiting_aux_slices == 0u && dom->handoff_armed) {
                    dom->handoff_armed = false;
                    dom->cv.notify_all();
                }
            }
        }
    } waiting_guard{dom};

    const auto poll_dur = std::chrono::milliseconds(std::max<std::uint32_t>(1u, cfg.cancel_poll_interval_ms));
    while (true) {
        if (should_cancel && should_cancel()) {
            const std::int64_t wait_ns = std::max<std::int64_t>(0, impl_->NowNs() - wait_begin_ns);
            dom->stats.aux_gate_wait_ns += static_cast<std::uint64_t>(wait_ns);
            ++dom->stats.aux_slices_cancelled;
            return AuxSliceScope();
        }

        const AuxGateConfig cur_cfg = impl_->SnapshotConfig();
        if (!cur_cfg.enabled || Impl::IsBypassedCpuDomain(cur_cfg, key_view)) {
            dom->handoff_armed = false;
            ++dom->stats.in_flight_aux_slices;
            ++dom->stats.aux_slices_total;
            if (Impl::IsBypassedCpuDomain(cur_cfg, key_view)) {
                ++dom->stats.aux_slices_idle_fastpath;
            } else if (dom->stats.in_flight_decode_steps > 0u) {
                ++dom->stats.aux_slices_overlapped_decode;
            }
            const std::int64_t wait_ns = std::max<std::int64_t>(0, impl_->NowNs() - wait_begin_ns);
            dom->stats.aux_gate_wait_ns += static_cast<std::uint64_t>(wait_ns);
            return AuxSliceScope(
                this, idx,
                Impl::IsBypassedCpuDomain(cur_cfg, key_view) ? AuxAcquireStatus::kAcquiredIdle
                                                             : AuxAcquireStatus::kBypassedDisabled,
                impl_->NowNs(), wait_ns);
        }

        if (dom->stats.in_flight_decode_steps == 0u && dom->stats.in_flight_aux_slices == 0u) {
            if (dom->stats.active_decode_sessions == 0u) {
                dom->handoff_armed = false;
                ++dom->stats.in_flight_aux_slices;
                ++dom->stats.aux_slices_total;
                ++dom->stats.aux_slices_idle_fastpath;
                const std::int64_t wait_ns = std::max<std::int64_t>(0, impl_->NowNs() - wait_begin_ns);
                dom->stats.aux_gate_wait_ns += static_cast<std::uint64_t>(wait_ns);
                return AuxSliceScope(this, idx, AuxAcquireStatus::kAcquiredIdle, impl_->NowNs(), wait_ns);
            }
            if (dom->aux_credit_ns > 0) {
                dom->handoff_armed = false;
                ++dom->stats.in_flight_aux_slices;
                ++dom->stats.aux_slices_total;
                ++dom->stats.aux_slices_gated_between_steps;
                const std::int64_t wait_ns = std::max<std::int64_t>(0, impl_->NowNs() - wait_begin_ns);
                dom->stats.aux_gate_wait_ns += static_cast<std::uint64_t>(wait_ns);
                return AuxSliceScope(this, idx, AuxAcquireStatus::kAcquiredBetweenSteps, impl_->NowNs(), wait_ns);
            }
            if (cur_cfg.max_starvation_wait_ms > 0u &&
                (std::chrono::steady_clock::now() - steady_wait_begin) >=
                    std::chrono::milliseconds(cur_cfg.max_starvation_wait_ms)) {
                dom->handoff_armed = false;
                ++dom->stats.in_flight_aux_slices;
                ++dom->stats.aux_slices_total;
                ++dom->stats.aux_slices_starvation_admitted;
                const std::int64_t wait_ns = std::max<std::int64_t>(0, impl_->NowNs() - wait_begin_ns);
                dom->stats.aux_gate_wait_ns += static_cast<std::uint64_t>(wait_ns);
                return AuxSliceScope(this, idx, AuxAcquireStatus::kAcquiredStarvation, impl_->NowNs(), wait_ns);
            }
        }

        (void)dom->cv.wait_for(lock, poll_dur);
    }
}

AuxGateStats DecodeAwareAuxGate::GetStats(std::string_view domain) const {
    AuxGateStats total{};
    if (!impl_) return total;
    if (!domain.empty()) {
        Impl::DomainState* dom = impl_->FindDomain(domain);
        if (dom == nullptr) return total;
        std::lock_guard<std::mutex> lock(dom->mu);
        return dom->stats;
    }

    std::vector<Impl::DomainState*> doms;
    {
        std::lock_guard<std::mutex> tlock(impl_->table_mu);
        doms.reserve(impl_->domains.size());
        for (const auto& d : impl_->domains) doms.push_back(d.get());
    }
    for (Impl::DomainState* dom : doms) {
        std::lock_guard<std::mutex> lock(dom->mu);
        total.decode_sessions_total += dom->stats.decode_sessions_total;
        total.decode_steps_total += dom->stats.decode_steps_total;
        total.decode_steps_synced_before_release += dom->stats.decode_steps_synced_before_release;
        total.decode_busy_ns += dom->stats.decode_busy_ns;
        total.decode_hold_wait_ns += dom->stats.decode_hold_wait_ns;
        total.decode_hold_timeouts += dom->stats.decode_hold_timeouts;
        total.aux_slices_total += dom->stats.aux_slices_total;
        total.aux_slices_synced_before_release += dom->stats.aux_slices_synced_before_release;
        total.aux_slices_idle_fastpath += dom->stats.aux_slices_idle_fastpath;
        total.aux_slices_gated_between_steps += dom->stats.aux_slices_gated_between_steps;
        total.aux_slices_starvation_admitted += dom->stats.aux_slices_starvation_admitted;
        total.aux_slices_cancelled += dom->stats.aux_slices_cancelled;
        total.aux_slices_overlapped_decode += dom->stats.aux_slices_overlapped_decode;
        total.aux_busy_ns += dom->stats.aux_busy_ns;
        total.aux_gate_wait_ns += dom->stats.aux_gate_wait_ns;
        total.active_decode_sessions += dom->stats.active_decode_sessions;
        total.in_flight_decode_steps += dom->stats.in_flight_decode_steps;
        total.in_flight_aux_slices += dom->stats.in_flight_aux_slices;
        total.waiting_aux_slices += dom->stats.waiting_aux_slices;
    }
    return total;
}

void DecodeAwareAuxGate::ResetStats() {
    if (!impl_) return;
    std::vector<Impl::DomainState*> doms;
    {
        std::lock_guard<std::mutex> tlock(impl_->table_mu);
        doms.reserve(impl_->domains.size());
        for (const auto& d : impl_->domains) doms.push_back(d.get());
    }
    for (Impl::DomainState* dom : doms) {
        std::lock_guard<std::mutex> lock(dom->mu);
        const std::size_t active_sess = dom->stats.active_decode_sessions;
        const std::size_t inflight_dec = dom->stats.in_flight_decode_steps;
        const std::size_t inflight_aux = dom->stats.in_flight_aux_slices;
        const std::size_t waiting_aux = dom->stats.waiting_aux_slices;
        dom->stats = AuxGateStats{};
        dom->stats.active_decode_sessions = active_sess;
        dom->stats.in_flight_decode_steps = inflight_dec;
        dom->stats.in_flight_aux_slices = inflight_aux;
        dom->stats.waiting_aux_slices = waiting_aux;
    }
}

std::shared_ptr<DecodeAwareAuxGate> DefaultProcessAuxGate() {
    static const std::shared_ptr<DecodeAwareAuxGate> kGate = std::make_shared<DecodeAwareAuxGate>();
    return kGate;
}

void SetAuxContextSyncObserverForTest(AuxContextSyncObserver observer) {
    std::lock_guard<std::mutex> lock(GetSyncObserverMutex());
    GetSyncObserverSlot() = std::move(observer);
}

void NotifyAuxContextSynchronized(const DecodeAwareAuxGate* gate, std::string_view domain,
                                  bool is_aux_slice) {
    AuxContextSyncObserver copy;
    {
        std::lock_guard<std::mutex> lock(GetSyncObserverMutex());
        copy = GetSyncObserverSlot();
    }
    if (!copy) return;

    AuxContextSyncObservation obs;
    obs.domain = std::string(domain);
    obs.is_aux_slice = is_aux_slice;
    if (gate != nullptr) {
        const AuxGateStats st = gate->GetStats(domain);
        obs.decode_step_in_flight_at_sync = (st.in_flight_decode_steps > 0u);
        obs.aux_slice_in_flight_at_sync = (st.in_flight_aux_slices > 0u);
    }
    copy(obs);
}

std::vector<PlannedEmbedBatch> SubslicePlannedBatch(const PlannedEmbedBatch& batch,
                                                    std::size_t max_slice_tokens,
                                                    std::size_t max_slice_sequences) {
    if (batch.items.empty()) return {};
    const std::size_t tok_cap = std::max<std::size_t>(1u, max_slice_tokens);
    const std::size_t seq_cap = std::max<std::size_t>(1u, max_slice_sequences);

    std::vector<PlannedEmbedBatch> out;
    PlannedEmbedBatch cur;
    auto flush_cur = [&] {
        if (!cur.items.empty()) {
            out.push_back(std::move(cur));
            cur = PlannedEmbedBatch{};
        }
    };

    for (const EmbedBatchItem& item : batch.items) {
        if (item.token_count > tok_cap) {
            flush_cur();
            PlannedEmbedBatch single;
            single.oversized_fallback = true;
            single.total_tokens = item.token_count;
            single.items.push_back(EmbedBatchItem{item.original_index, item.token_count, 0});
            out.push_back(std::move(single));
            continue;
        }
        if (!cur.items.empty() &&
            (cur.items.size() >= seq_cap || cur.total_tokens + item.token_count > tok_cap)) {
            flush_cur();
        }
        const auto seq_id = static_cast<std::int32_t>(cur.items.size());
        cur.items.push_back(EmbedBatchItem{item.original_index, item.token_count, seq_id});
        cur.total_tokens += item.token_count;
    }
    flush_cur();
    return out;
}

SttSlicePlan PlanSttAudioSlices(std::uint64_t total_pcm_samples, const SttSlicePlanConfig& config) {
    SttSlicePlan plan;
    auto fail = [&](std::string msg) {
        plan.ok = false;
        plan.error = std::move(msg);
        return plan;
    };

    if (total_pcm_samples == 0u) return fail("total_pcm_samples must be > 0");
    if (config.sample_rate_hz == 0u) return fail("sample_rate_hz must be > 0");
    if (config.frame_hop_samples == 0u) return fail("frame_hop_samples must be > 0");
    if (config.encoder_downsample == 0u) return fail("encoder_downsample must be > 0");
    if (config.idle_slice_max_tokens == 0u || config.idle_slice_max_tokens > kMaxAuxActiveSliceTokens) {
        return fail("idle_slice_max_tokens out of range");
    }
    if (config.active_slice_max_tokens == 0u || config.active_slice_max_tokens > kMaxAuxActiveSliceTokens) {
        return fail("active_slice_max_tokens out of range");
    }

    const std::uint64_t hop = config.frame_hop_samples;
    const std::uint64_t ds = config.encoder_downsample;
    if (hop > std::numeric_limits<std::uint64_t>::max() / ds) {
        return fail("samples_per_token overflow");
    }
    const std::uint64_t samples_per_token = hop * ds;
    const std::uint64_t total_tokens_u64 = (total_pcm_samples - 1u) / samples_per_token + 1u;
    if (total_tokens_u64 > kMaxSttTotalEncoderTokens) {
        return fail("total_pcm_samples exceeds maximum single-plan encoder token limit");
    }

    const auto total_tokens = static_cast<std::size_t>(total_tokens_u64);
    const std::size_t cap =
        config.decode_active ? config.active_slice_max_tokens : config.idle_slice_max_tokens;
    const std::size_t overlap = config.overlap_tokens;
    if (overlap >= cap) {
        return fail("overlap_tokens must be strictly less than slice token cap");
    }

    plan.total_encoder_tokens = total_tokens;
    std::size_t new_pos = 0;
    std::size_t slice_idx = 0;
    while (new_pos < total_tokens) {
        const std::size_t prefix_overlap = (slice_idx == 0u) ? 0u : std::min(overlap, new_pos);
        const std::size_t max_new = cap - prefix_overlap;
        const std::size_t new_tokens = std::min(max_new, total_tokens - new_pos);
        const std::size_t start_tok = new_pos - prefix_overlap;
        const std::size_t slice_toks = prefix_overlap + new_tokens;

        const std::uint64_t sample_off = static_cast<std::uint64_t>(start_tok) * samples_per_token;
        const std::uint64_t raw_end = static_cast<std::uint64_t>(start_tok + slice_toks) * samples_per_token;
        const std::uint64_t sample_end = std::min(total_pcm_samples, raw_end);

        SttAudioSlice s;
        s.slice_index = slice_idx;
        s.sample_offset = sample_off;
        s.sample_count = sample_end >= sample_off ? (sample_end - sample_off) : 0u;
        s.encoder_tokens = slice_toks;
        s.overlap_prefix_tokens = prefix_overlap;
        plan.slices.push_back(s);
        plan.max_slice_tokens_used = std::max(plan.max_slice_tokens_used, slice_toks);

        new_pos += new_tokens;
        ++slice_idx;
    }

    plan.ok = true;
    return plan;
}

Status RunGatedAuxSlices(
    DecodeAwareAuxGate& gate, std::string_view domain, std::size_t slice_count,
    const std::function<Status(std::size_t slice_index, bool decode_active)>& run_slice,
    const std::function<bool()>& should_cancel, StreamPriority stream_priority,
    const CudaPriorityStream* cuda_stream, AuxSliceRunSummary* out_summary) {
    if (out_summary != nullptr) *out_summary = {};
    if (slice_count == 0u) return Status::Ok();
    if (!run_slice) {
        return Status::Error(ErrorCode::kInvalidArgument, "run_slice callback is empty");
    }

    std::size_t executed = 0;
    std::size_t gated_count = 0;
    std::int64_t wait_ns_total = 0;

    for (std::size_t i = 0; i < slice_count; ++i) {
        auto scope = gate.AcquireAuxSlice(domain, should_cancel);
        if (!scope.acquired()) {
            if (out_summary != nullptr) {
                out_summary->slices_executed = executed;
                out_summary->gated_slices = gated_count;
                out_summary->gate_wait_ms = static_cast<double>(wait_ns_total) / 1e6;
            }
            return Status::Error(ErrorCode::kDecodeFailed, "auxiliary slice cancelled");
        }
        wait_ns_total += scope.wait_ns();
        if (scope.gated()) ++gated_count;

        Status slice_status = Status::Ok();
        {
            ScopedGgmlCudaStreamPriority prio_guard(stream_priority);
            const bool decode_active = scope.gated() || gate.IsDecodeActive(domain);
            slice_status = run_slice(i, decode_active);
            if (slice_status.ok() && cuda_stream != nullptr && cuda_stream->valid()) {
                slice_status = cuda_stream->Synchronize();
                if (slice_status.ok()) {
                    scope.MarkSynchronized();
                    NotifyAuxContextSynchronized(&gate, domain, /*is_aux_slice=*/true);
                }
            }
        }
        scope.Release();

        if (!slice_status.ok()) {
            if (out_summary != nullptr) {
                out_summary->slices_executed = executed;
                out_summary->gated_slices = gated_count;
                out_summary->gate_wait_ms = static_cast<double>(wait_ns_total) / 1e6;
            }
            return slice_status;
        }
        ++executed;
    }

    if (out_summary != nullptr) {
        out_summary->slices_executed = executed;
        out_summary->gated_slices = gated_count;
        out_summary->gate_wait_ms = static_cast<double>(wait_ns_total) / 1e6;
    }
    return Status::Ok();
}

}  // namespace sonder::backends::llamacpp
