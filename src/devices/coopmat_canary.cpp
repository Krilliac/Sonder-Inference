// Sonder Inference: AMD Windows Vulkan cooperative-matrix child-process canary (NAT-16).
//
// Design:
//   - Guards against AMD Windows Vulkan cooperative-matrix init regressions:
//       * Phase 1 (closed, unmerged PR llama.cpp#22750): Radeon 860M (RDNA 3.5)
//         amdvlk crash during vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
//         at backend init.
//       * Phase 2: generic 16x16x16 FP16->FP32 OpCooperativeMatrixMulAddKHR
//         compute smoke test over ceil(ubatch/16) tiles (does not reproduce
//         upstream flash-attention coopmat1 flip-direct shader bugs, which are
//         absent from pinned v1.0.0 and gated upstream by llama.cpp#32415).
//   - Runs the Vulkan probe strictly inside an isolated child process
//     (`sonder-infer coopmat-canary`), bounded by a Windows Job Object and wall
//     timeout, with crash dialogs suppressed via SetErrorMode.
//   - Caches definitive pass/fail verdicts atomically on disk keyed by
//     (vendor_id, device_id, driver_build, ubatch) so driver upgrades
//     automatically re-trigger verification.
//   - On failure (or optional fail-closed runner error), automatically activates
//     GGML_VK_DISABLE_COOPMAT=1 before llama_backend_init() and records the
//     decision for runtime_status().warnings (`amd_vk_coopmat_disabled`).

#include "sonder/inference/coopmat_canary.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "sonder/inference/json.hpp"

namespace sonder::inference {

namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kAmdVendorId = 0x1002u;
constexpr const char* kCacheSchema = "sonder.inference.coopmat_canary_cache/1";
constexpr std::size_t kMaxCacheBytes = 128u * 1024u;
constexpr std::size_t kMaxCacheEntries = 128u;
constexpr std::size_t kMaxDriverBuildLen = 128u;
constexpr std::size_t kMaxReasonLen = 256u;

std::mutex& cache_io_mutex() {
    static std::mutex m;
    return m;
}

std::mutex& decision_mutex() {
    static std::mutex m;
    return m;
}

std::optional<CoopmatCanaryDecision>& stored_decision() {
    static std::optional<CoopmatCanaryDecision> d;
    return d;
}

std::string read_env_var(const char* name) {
    if (name == nullptr || name[0] == '\0') return {};
#if defined(_WIN32)
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || buf == nullptr) {
        return {};
    }
    std::string out(buf);
    std::free(buf);
    return out;
#else
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string{};
#endif
}

bool set_env_var(const char* name, const char* value) noexcept {
    if (name == nullptr || value == nullptr) return false;
#if defined(_WIN32)
    return _putenv_s(name, value) == 0;
#else
    return ::setenv(name, value, 1) == 0;
#endif
}

bool is_env_truthy(std::string_view raw) noexcept {
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front()))) raw.remove_prefix(1);
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) raw.remove_suffix(1);
    if (raw.empty()) return false;
    if (raw == "0" || raw == "false" || raw == "FALSE" || raw == "off" || raw == "OFF" || raw == "no" || raw == "NO") {
        return false;
    }
    return true;
}

std::string sanitize_single_line(std::string_view raw, std::size_t max_len) {
    std::string out;
    out.reserve(std::min(raw.size(), max_len));
    for (const unsigned char c : raw) {
        if (c < 0x20u || c == 0x7Fu) continue;
        out.push_back(static_cast<char>(c));
    }
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    if (out.size() > max_len) {
        out.resize(max_len);
    }
    return out;
}

std::string format_hex_u32(std::uint32_t v, int min_width = 4) {
    std::ostringstream oss;
    oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(min_width) << v;
    return oss.str();
}

bool has_traversal_or_control(std::string_view p) noexcept {
    if (p.empty()) return true;
    for (const unsigned char c : p) {
        if (c < 0x20u || c == 0x7Fu) return true;
    }
    // Check raw path segments for ".." across both '/' and '\\'.
    std::size_t pos = 0;
    while (pos <= p.size()) {
        std::size_t sep = p.find_first_of("/\\", pos);
        if (sep == std::string_view::npos) sep = p.size();
        if (p.substr(pos, sep - pos) == "..") return true;
        if (sep == p.size()) break;
        pos = sep + 1;
    }
    return false;
}

bool is_safe_file_path(const std::string& raw, fs::path& out_path) {
    if (has_traversal_or_control(raw)) return false;
    try {
        fs::path p(raw);
        for (const auto& part : p) {
            if (part == "..") return false;
        }
        out_path = std::move(p);
        return true;
    } catch (...) {
        return false;
    }
}

std::string lower_ascii(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::optional<CoopmatCanaryStatus> parse_cached_status(std::string_view s) noexcept {
    if (s == "passed") return CoopmatCanaryStatus::passed;
    if (s == "failed") return CoopmatCanaryStatus::failed;
    return std::nullopt;
}

std::string current_executable_path() {
#if defined(_WIN32)
    std::wstring buf(32768u, L'\0');
    const DWORD n = ::GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0u || n >= buf.size()) return {};
    buf.resize(n);
    const int utf8_len = ::WideCharToMultiByte(CP_UTF8, 0, buf.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 1) return {};
    std::string out(static_cast<std::size_t>(utf8_len - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, buf.c_str(), -1, out.data(), utf8_len, nullptr, nullptr);
    return out;
#elif defined(__linux__)
    std::vector<char> buf(4096u, '\0');
    const ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1u);
    if (n <= 0) return {};
    return std::string(buf.data(), static_cast<std::size_t>(n));
#else
    return {};
#endif
}

bool is_regular_executable_file(const fs::path& p) {
    std::error_code ec;
    if (fs::is_symlink(p, ec)) return false;
    return fs::is_regular_file(p, ec);
}

#if defined(_WIN32)
std::wstring utf8_to_wide(std::string_view s) {
    if (s.empty()) return {};
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

struct ScopedWinHandle {
    HANDLE handle = nullptr;
    ScopedWinHandle() = default;
    explicit ScopedWinHandle(HANDLE h) noexcept : handle(h) {}
    ~ScopedWinHandle() noexcept {
        if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle);
        }
    }
    ScopedWinHandle(const ScopedWinHandle&) = delete;
    ScopedWinHandle& operator=(const ScopedWinHandle&) = delete;
    bool valid() const noexcept { return handle != nullptr && handle != INVALID_HANDLE_VALUE; }
};
#endif

CoopmatCanaryChildOutcome spawn_canary_child(const std::string& exe_path, const CoopmatCanaryKey& key,
                                             std::chrono::milliseconds timeout) {
    CoopmatCanaryChildOutcome out;
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed_ms = [&]() -> std::uint64_t {
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
        return dt.count() > 0 ? static_cast<std::uint64_t>(dt.count()) : 0u;
    };

    fs::path safe_exe;
    if (!is_safe_file_path(exe_path, safe_exe) || !is_regular_executable_file(safe_exe)) {
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "canary executable not found or invalid";
        return out;
    }

    const std::string vendor_arg = "0x" + format_hex_u32(key.vendor_id, 4);
    const std::string device_arg = "0x" + format_hex_u32(key.device_id, 4);
    const std::uint32_t ubatch = std::clamp(key.ubatch == 0u ? 64u : key.ubatch, 16u, 4096u);
    const std::string ubatch_arg = std::to_string(ubatch);
    const std::int64_t clamped_timeout_ms = std::clamp<std::int64_t>(timeout.count(), 250, 120000);

#if defined(_WIN32)
    ScopedWinHandle job(::CreateJobObjectW(nullptr, nullptr));
    if (!job.valid()) {
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "CreateJobObjectW failed";
        return out;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
    jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!::SetInformationJobObject(job.handle, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli))) {
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "SetInformationJobObject failed";
        return out;
    }

    const std::wstring exe_w = safe_exe.wstring();
    std::wstring cmdline = L"\"" + exe_w + L"\" coopmat-canary --vendor-id " + utf8_to_wide(vendor_arg) +
                           L" --device-id " + utf8_to_wide(device_arg) + L" --ubatch " + utf8_to_wide(ubatch_arg) +
                           L" --quiet";
    std::vector<wchar_t> cmd_buf(cmdline.begin(), cmdline.end());
    cmd_buf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    const DWORD flags = CREATE_SUSPENDED | CREATE_NO_WINDOW;
    if (!::CreateProcessW(exe_w.c_str(), cmd_buf.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi)) {
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "CreateProcessW failed (err=" + std::to_string(::GetLastError()) + ")";
        return out;
    }
    ScopedWinHandle proc_h(pi.hProcess);
    ScopedWinHandle thread_h(pi.hThread);

    if (!::AssignProcessToJobObject(job.handle, proc_h.handle)) {
        ::TerminateProcess(proc_h.handle, 1u);
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "AssignProcessToJobObject failed";
        return out;
    }
    if (::ResumeThread(thread_h.handle) == static_cast<DWORD>(-1)) {
        ::TerminateJobObject(job.handle, 1u);
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "ResumeThread failed";
        return out;
    }

    const DWORD wait_rc = ::WaitForSingleObject(proc_h.handle, static_cast<DWORD>(clamped_timeout_ms));
    out.elapsed_ms = elapsed_ms();
    if (wait_rc == WAIT_OBJECT_0) {
        DWORD exit_code = 1u;
        if (!::GetExitCodeProcess(proc_h.handle, &exit_code)) {
            out.spawn_error = true;
            out.reason = "GetExitCodeProcess failed";
            return out;
        }
        out.completed = true;
        out.exit_code = static_cast<std::uint32_t>(exit_code);
        if (exit_code == 0u) {
            out.reason = "ok";
        } else {
            out.reason = "canary child exited with code 0x" + format_hex_u32(out.exit_code, 8);
        }
        return out;
    }
    if (wait_rc == WAIT_TIMEOUT) {
        ::TerminateJobObject(job.handle, 1u);
        ::WaitForSingleObject(proc_h.handle, 2000u);
        out.timed_out = true;
        out.reason = "canary child timeout after " + std::to_string(clamped_timeout_ms) + "ms";
        return out;
    }
    ::TerminateJobObject(job.handle, 1u);
    out.spawn_error = true;
    out.reason = "WaitForSingleObject failed (rc=" + std::to_string(wait_rc) + ")";
    return out;
#else
    // Prepare argv strictly before fork() so the child performs only
    // async-signal-safe execv() / _exit().
    std::string exe_s = safe_exe.string();
    std::string cmd_s = "coopmat-canary";
    std::string f_vid = "--vendor-id";
    std::string f_did = "--device-id";
    std::string f_ub = "--ubatch";
    std::string f_q = "--quiet";
    std::string v_s = vendor_arg;
    std::string d_s = device_arg;
    std::string u_s = ubatch_arg;
    char* const argv[] = {exe_s.data(), cmd_s.data(), f_vid.data(), v_s.data(), f_did.data(),
                          d_s.data(),   f_ub.data(),  u_s.data(),   f_q.data(), nullptr};
    const pid_t pid = ::fork();
    if (pid < 0) {
        out.spawn_error = true;
        out.elapsed_ms = elapsed_ms();
        out.reason = "fork failed";
        return out;
    }
    if (pid == 0) {
        ::execv(exe_s.c_str(), argv);
        ::_exit(127);
    }

    const auto deadline = t0 + std::chrono::milliseconds(clamped_timeout_ms);
    int status = 0;
    while (true) {
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            out.completed = true;
            out.elapsed_ms = elapsed_ms();
            if (WIFEXITED(status)) {
                out.exit_code = static_cast<std::uint32_t>(WEXITSTATUS(status));
                out.reason = out.exit_code == 0u ? "ok" : ("canary child exited with code " + std::to_string(out.exit_code));
            } else if (WIFSIGNALED(status)) {
                const int sig = WTERMSIG(status);
                out.exit_code = 128u + static_cast<std::uint32_t>(sig);
                out.reason = "canary child killed by signal " + std::to_string(sig);
            } else {
                out.exit_code = 1u;
                out.reason = "canary child terminated abnormally";
            }
            return out;
        }
        if (r < 0) {
            out.spawn_error = true;
            out.elapsed_ms = elapsed_ms();
            out.reason = "waitpid failed";
            return out;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
            out.timed_out = true;
            out.elapsed_ms = elapsed_ms();
            out.reason = "canary child timeout after " + std::to_string(clamped_timeout_ms) + "ms";
            return out;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
#endif
}

// ---------------------------------------------------------------------------
// Minimal dynamic Vulkan 1.2 + VK_KHR_cooperative_matrix ABI (zero SDK dep).
// Struct layouts verified against Vulkan SDK 1.4.357.0 vulkan_core.h.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#define SONDER_VK_CALL __stdcall
#else
#define SONDER_VK_CALL
#endif

using VkResult = std::int32_t;
using VkBool32 = std::uint32_t;
using VkFlags = std::uint32_t;
using VkDeviceSize = std::uint64_t;
using VkInstance = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using VkDevice = struct VkDevice_T*;
using VkQueue = struct VkQueue_T*;
using VkCommandBuffer = struct VkCommandBuffer_T*;
// Non-dispatchable handles are always uint64_t on 64-bit and 32-bit Vulkan ABI.
using VkBuffer = std::uint64_t;
using VkDeviceMemory = std::uint64_t;
using VkShaderModule = std::uint64_t;
using VkDescriptorSetLayout = std::uint64_t;
using VkPipelineLayout = std::uint64_t;
using VkPipelineCache = std::uint64_t;
using VkPipeline = std::uint64_t;
using VkDescriptorPool = std::uint64_t;
using VkDescriptorSet = std::uint64_t;
using VkCommandPool = std::uint64_t;
using VkFence = std::uint64_t;
using VkSemaphore = std::uint64_t;
using VkSampler = std::uint64_t;

constexpr VkResult kVkSuccess = 0;
constexpr std::uint32_t kVkApiVersion12 = (1u << 22) | (2u << 12);
constexpr std::uint32_t kVkStypeApplicationInfo = 0u;
constexpr std::uint32_t kVkStypeInstanceCreateInfo = 1u;
constexpr std::uint32_t kVkStypeDeviceQueueCreateInfo = 2u;
constexpr std::uint32_t kVkStypeDeviceCreateInfo = 3u;
constexpr std::uint32_t kVkStypeSubmitInfo = 4u;
constexpr std::uint32_t kVkStypeMemoryAllocateInfo = 5u;
constexpr std::uint32_t kVkStypeBufferCreateInfo = 12u;
constexpr std::uint32_t kVkStypeShaderModuleCreateInfo = 16u;
constexpr std::uint32_t kVkStypePipelineShaderStageCreateInfo = 18u;
constexpr std::uint32_t kVkStypeComputePipelineCreateInfo = 29u;
constexpr std::uint32_t kVkStypePipelineLayoutCreateInfo = 30u;
constexpr std::uint32_t kVkStypeDescriptorSetLayoutCreateInfo = 32u;
constexpr std::uint32_t kVkStypeDescriptorPoolCreateInfo = 33u;
constexpr std::uint32_t kVkStypeDescriptorSetAllocateInfo = 34u;
constexpr std::uint32_t kVkStypeWriteDescriptorSet = 35u;
constexpr std::uint32_t kVkStypeCommandPoolCreateInfo = 39u;
constexpr std::uint32_t kVkStypeCommandBufferAllocateInfo = 40u;
constexpr std::uint32_t kVkStypeCommandBufferBeginInfo = 42u;
constexpr std::uint32_t kVkStypePhysicalDeviceFeatures2 = 1000059000u;
constexpr std::uint32_t kVkStypePhysicalDeviceShaderFloat16Int8Features = 1000082000u;
constexpr std::uint32_t kVkStypePhysicalDevice16BitStorageFeatures = 1000083000u;
constexpr std::uint32_t kVkStypePhysicalDeviceVulkanMemoryModelFeatures = 1000211000u;
constexpr std::uint32_t kVkStypePhysicalDeviceCoopMatFeaturesKHR = 1000506000u;
constexpr std::uint32_t kVkStypeCoopMatPropertiesKHR = 1000506001u;

constexpr std::uint32_t kVkComponentFloat16KHR = 0u;
constexpr std::uint32_t kVkComponentFloat32KHR = 1u;
constexpr std::uint32_t kVkScopeSubgroupKHR = 3u;

constexpr VkFlags kVkQueueComputeBit = 0x00000002u;
constexpr VkFlags kVkBufferUsageStorageBufferBit = 0x00000020u;
constexpr VkFlags kVkMemoryHostVisibleBit = 0x00000002u;
constexpr VkFlags kVkMemoryHostCoherentBit = 0x00000004u;
constexpr std::uint32_t kVkDescriptorTypeStorageBuffer = 7u;
constexpr VkFlags kVkShaderStageComputeBit = 0x00000020u;
constexpr std::uint32_t kVkPipelineBindPointCompute = 1u;
constexpr VkFlags kVkCommandBufferUsageOneTimeSubmitBit = 0x00000001u;
constexpr VkDeviceSize kVkWholeSize = ~VkDeviceSize{0};

struct VkApplicationInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    std::uint32_t applicationVersion;
    const char* pEngineName;
    std::uint32_t engineVersion;
    std::uint32_t apiVersion;
};

struct VkInstanceCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfoRaw* pApplicationInfo;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkPhysicalDevicePropertiesRaw {
    std::uint32_t apiVersion;
    std::uint32_t driverVersion;
    std::uint32_t vendorID;
    std::uint32_t deviceID;
    std::uint32_t deviceType;
    char deviceName[256];
    std::uint8_t pipelineCacheUUID[16];
    std::uint32_t limits_align_pad;
    std::uint64_t limits_and_sparse[66];
};

struct VkExtensionPropertiesRaw {
    char extensionName[256];
    std::uint32_t specVersion;
};

struct VkQueueFamilyPropertiesRaw {
    VkFlags queueFlags;
    std::uint32_t queueCount;
    std::uint32_t timestampValidBits;
    std::uint32_t minImageTransferGranularity[3];
};

struct VkPhysicalDeviceFeaturesRaw {
    VkBool32 flags[55];
};

struct VkPhysicalDeviceFeatures2Raw {
    std::uint32_t sType;
    void* pNext;
    VkPhysicalDeviceFeaturesRaw features;
};

struct VkPhysicalDeviceCoopMatFeaturesKHRRaw {
    std::uint32_t sType;
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
};

struct VkCoopMatPropertiesKHRRaw {
    std::uint32_t sType;
    void* pNext;
    std::uint32_t MSize;
    std::uint32_t NSize;
    std::uint32_t KSize;
    std::uint32_t AType;
    std::uint32_t BType;
    std::uint32_t CType;
    std::uint32_t ResultType;
    VkBool32 saturatingAccumulation;
    std::uint32_t scope;
};

struct VkPhysicalDevice16BitStorageFeaturesRaw {
    std::uint32_t sType;
    void* pNext;
    VkBool32 storageBuffer16BitAccess;
    VkBool32 uniformAndStorageBuffer16BitAccess;
    VkBool32 storagePushConstant16;
    VkBool32 storageInputOutput16;
};

struct VkPhysicalDeviceShaderFloat16Int8FeaturesRaw {
    std::uint32_t sType;
    void* pNext;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
};

struct VkPhysicalDeviceVulkanMemoryModelFeaturesRaw {
    std::uint32_t sType;
    void* pNext;
    VkBool32 vulkanMemoryModel;
    VkBool32 vulkanMemoryModelDeviceScope;
    VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
};

struct VkMemoryTypeRaw {
    VkFlags propertyFlags;
    std::uint32_t heapIndex;
};

struct VkMemoryHeapRaw {
    VkDeviceSize size;
    VkFlags flags;
};

struct VkPhysicalDeviceMemoryPropertiesRaw {
    std::uint32_t memoryTypeCount;
    VkMemoryTypeRaw memoryTypes[32];
    std::uint32_t memoryHeapCount;
    VkMemoryHeapRaw memoryHeaps[16];
};

struct VkDeviceQueueCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueFamilyIndex;
    std::uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfoRaw* pQueueCreateInfos;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeaturesRaw* pEnabledFeatures;
};

struct VkBufferCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    std::uint32_t sharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
};

struct VkMemoryRequirementsRaw {
    VkDeviceSize size;
    VkDeviceSize alignment;
    std::uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    std::uint32_t memoryTypeIndex;
};

struct VkShaderModuleCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::size_t codeSize;
    const std::uint32_t* pCode;
};

struct VkDescriptorSetLayoutBindingRaw {
    std::uint32_t binding;
    std::uint32_t descriptorType;
    std::uint32_t descriptorCount;
    VkFlags stageFlags;
    const VkSampler* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t bindingCount;
    const VkDescriptorSetLayoutBindingRaw* pBindings;
};

struct VkPipelineLayoutCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    std::uint32_t pushConstantRangeCount;
    const void* pPushConstantRanges;
};

struct VkPipelineShaderStageCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};

struct VkComputePipelineCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfoRaw stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    std::int32_t basePipelineIndex;
};

struct VkDescriptorPoolSizeRaw {
    std::uint32_t type;
    std::uint32_t descriptorCount;
};

struct VkDescriptorPoolCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t maxSets;
    std::uint32_t poolSizeCount;
    const VkDescriptorPoolSizeRaw* pPoolSizes;
};

struct VkDescriptorSetAllocateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    std::uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfoRaw {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkWriteDescriptorSetRaw {
    std::uint32_t sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    std::uint32_t dstBinding;
    std::uint32_t dstArrayElement;
    std::uint32_t descriptorCount;
    std::uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfoRaw* pBufferInfo;
    const void* pTexelBufferView;
};

struct VkCommandPoolCreateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    std::uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkCommandPool commandPool;
    std::uint32_t level;
    std::uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
};

struct VkSubmitInfoRaw {
    std::uint32_t sType;
    const void* pNext;
    std::uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    std::uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    std::uint32_t signalSemaphoreCount;
    const VkSemaphore* pSignalSemaphores;
};

// Validated SPIR-V 1.5 (Vulkan 1.2) compute shader compiled via glslangValidator
// and verified with spirv-val --target-env vulkan1.2:
//   #version 450 core
//   #extension GL_KHR_memory_scope_semantics : enable
//   #extension GL_KHR_cooperative_matrix : enable
//   #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
//   #extension GL_KHR_shader_subgroup_basic : enable
//   layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
//   layout(set = 0, binding = 0) readonly  buffer BufA { float16_t data[]; } bufA;
//   layout(set = 0, binding = 1) readonly  buffer BufB { float16_t data[]; } bufB;
//   layout(set = 0, binding = 2) writeonly buffer BufD { float     data[]; } bufD;
//   void main() {
//       if (gl_SubgroupID == 0u) {
//           uint tile = gl_WorkGroupID.x;
//           coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC =
//               coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
//           coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
//           coopMatLoad(matA, bufA.data, tile * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
//           coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
//           coopMatLoad(matB, bufB.data, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
//           matC = coopMatMulAdd(matA, matB, matC);
//           coopMatStore(matC, bufD.data, tile * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
//       }
//   }
constexpr std::array<std::uint32_t, 597> kCoopmatCanarySpv = {
    0x07230203u, 0x00010500u, 0x0008000bu, 0x0000004cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x00000009u, 0x00020011u, 0x0000003du, 0x00020011u, 0x00001151u, 0x00020011u, 0x000014e1u, 0x00020011u,
    0x00001786u, 0x0008000au, 0x5f565053u, 0x5f52484bu, 0x706f6f63u, 0x74617265u, 0x5f657669u, 0x7274616du,
    0x00007869u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu,
    0x00000000u, 0x00000003u, 0x000a000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000008u,
    0x00000013u, 0x00000026u, 0x00000038u, 0x00000043u, 0x00060010u, 0x00000004u, 0x00000011u, 0x00000040u,
    0x00000001u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x000d0004u, 0x455f4c47u, 0x735f5458u,
    0x65646168u, 0x78655f72u, 0x63696c70u, 0x615f7469u, 0x68746972u, 0x6974656du, 0x79745f63u, 0x5f736570u,
    0x616f6c66u, 0x00363174u, 0x00080004u, 0x4b5f4c47u, 0x635f5248u, 0x65706f6fu, 0x69746172u, 0x6d5f6576u,
    0x69727461u, 0x00000078u, 0x00090004u, 0x4b5f4c47u, 0x6d5f5248u, 0x726f6d65u, 0x63735f79u, 0x5f65706fu,
    0x616d6573u, 0x6369746eu, 0x00000073u, 0x00090004u, 0x4b5f4c47u, 0x735f5248u, 0x65646168u, 0x75735f72u,
    0x6f726762u, 0x625f7075u, 0x63697361u, 0x00000000u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00060005u, 0x00000008u, 0x535f6c67u, 0x72676275u, 0x4970756fu, 0x00000044u, 0x00040005u, 0x00000010u,
    0x656c6974u, 0x00000000u, 0x00060005u, 0x00000013u, 0x575f6c67u, 0x476b726fu, 0x70756f72u, 0x00004449u,
    0x00040005u, 0x0000001cu, 0x4374616du, 0x00000000u, 0x00040005u, 0x00000022u, 0x4174616du, 0x00000000u,
    0x00040005u, 0x00000024u, 0x41667542u, 0x00000000u, 0x00050006u, 0x00000024u, 0x00000000u, 0x61746164u,
    0x00000000u, 0x00040005u, 0x00000026u, 0x41667562u, 0x00000000u, 0x00040005u, 0x00000034u, 0x4274616du,
    0x00000000u, 0x00040005u, 0x00000036u, 0x42667542u, 0x00000000u, 0x00050006u, 0x00000036u, 0x00000000u,
    0x61746164u, 0x00000000u, 0x00040005u, 0x00000038u, 0x42667562u, 0x00000000u, 0x00040005u, 0x00000041u,
    0x44667542u, 0x00000000u, 0x00050006u, 0x00000041u, 0x00000000u, 0x61746164u, 0x00000000u, 0x00040005u,
    0x00000043u, 0x44667562u, 0x00000000u, 0x00040047u, 0x00000008u, 0x0000000bu, 0x00000028u, 0x00040047u,
    0x00000013u, 0x0000000bu, 0x0000001au, 0x00040047u, 0x00000023u, 0x00000006u, 0x00000002u, 0x00030047u,
    0x00000024u, 0x00000002u, 0x00040048u, 0x00000024u, 0x00000000u, 0x00000018u, 0x00050048u, 0x00000024u,
    0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000026u, 0x00000018u, 0x00040047u, 0x00000026u,
    0x00000021u, 0x00000000u, 0x00040047u, 0x00000026u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000035u,
    0x00000006u, 0x00000002u, 0x00030047u, 0x00000036u, 0x00000002u, 0x00040048u, 0x00000036u, 0x00000000u,
    0x00000018u, 0x00050048u, 0x00000036u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000038u,
    0x00000018u, 0x00040047u, 0x00000038u, 0x00000021u, 0x00000001u, 0x00040047u, 0x00000038u, 0x00000022u,
    0x00000000u, 0x00040047u, 0x00000040u, 0x00000006u, 0x00000004u, 0x00030047u, 0x00000041u, 0x00000002u,
    0x00040048u, 0x00000041u, 0x00000000u, 0x00000019u, 0x00050048u, 0x00000041u, 0x00000000u, 0x00000023u,
    0x00000000u, 0x00030047u, 0x00000043u, 0x00000019u, 0x00040047u, 0x00000043u, 0x00000021u, 0x00000002u,
    0x00040047u, 0x00000043u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000004bu, 0x0000000bu, 0x00000019u,
    0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u,
    0x00000000u, 0x00040020u, 0x00000007u, 0x00000001u, 0x00000006u, 0x0004003bu, 0x00000007u, 0x00000008u,
    0x00000001u, 0x0004002bu, 0x00000006u, 0x0000000au, 0x00000000u, 0x00020014u, 0x0000000bu, 0x00040020u,
    0x0000000fu, 0x00000007u, 0x00000006u, 0x00040017u, 0x00000011u, 0x00000006u, 0x00000003u, 0x00040020u,
    0x00000012u, 0x00000001u, 0x00000011u, 0x0004003bu, 0x00000012u, 0x00000013u, 0x00000001u, 0x00030016u,
    0x00000016u, 0x00000020u, 0x0004002bu, 0x00000006u, 0x00000017u, 0x00000003u, 0x0004002bu, 0x00000006u,
    0x00000018u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000019u, 0x00000002u, 0x00071168u, 0x0000001au,
    0x00000016u, 0x00000017u, 0x00000018u, 0x00000018u, 0x00000019u, 0x00040020u, 0x0000001bu, 0x00000007u,
    0x0000001au, 0x0004002bu, 0x00000016u, 0x0000001du, 0x00000000u, 0x0004002cu, 0x0000001au, 0x0000001eu,
    0x0000001du, 0x00030016u, 0x0000001fu, 0x00000010u, 0x00071168u, 0x00000020u, 0x0000001fu, 0x00000017u,
    0x00000018u, 0x00000018u, 0x0000000au, 0x00040020u, 0x00000021u, 0x00000007u, 0x00000020u, 0x0003001du,
    0x00000023u, 0x0000001fu, 0x0003001eu, 0x00000024u, 0x00000023u, 0x00040020u, 0x00000025u, 0x0000000cu,
    0x00000024u, 0x0004003bu, 0x00000025u, 0x00000026u, 0x0000000cu, 0x00040015u, 0x00000027u, 0x00000020u,
    0x00000001u, 0x0004002bu, 0x00000027u, 0x00000028u, 0x00000000u, 0x0004002bu, 0x00000006u, 0x0000002au,
    0x00000100u, 0x00040020u, 0x0000002cu, 0x0000000cu, 0x0000001fu, 0x0004002bu, 0x00000006u, 0x00000031u,
    0x00000001u, 0x00071168u, 0x00000032u, 0x0000001fu, 0x00000017u, 0x00000018u, 0x00000018u, 0x00000031u,
    0x00040020u, 0x00000033u, 0x00000007u, 0x00000032u, 0x0003001du, 0x00000035u, 0x0000001fu, 0x0003001eu,
    0x00000036u, 0x00000035u, 0x00040020u, 0x00000037u, 0x0000000cu, 0x00000036u, 0x0004003bu, 0x00000037u,
    0x00000038u, 0x0000000cu, 0x0003001du, 0x00000040u, 0x00000016u, 0x0003001eu, 0x00000041u, 0x00000040u,
    0x00040020u, 0x00000042u, 0x0000000cu, 0x00000041u, 0x0004003bu, 0x00000042u, 0x00000043u, 0x0000000cu,
    0x00040020u, 0x00000046u, 0x0000000cu, 0x00000016u, 0x0004002bu, 0x00000006u, 0x0000004au, 0x00000040u,
    0x0006002cu, 0x00000011u, 0x0000004bu, 0x0000004au, 0x00000031u, 0x00000031u, 0x00050036u, 0x00000002u,
    0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000000fu, 0x00000010u,
    0x00000007u, 0x0004003bu, 0x0000001bu, 0x0000001cu, 0x00000007u, 0x0004003bu, 0x00000021u, 0x00000022u,
    0x00000007u, 0x0004003bu, 0x00000033u, 0x00000034u, 0x00000007u, 0x0004003du, 0x00000006u, 0x00000009u,
    0x00000008u, 0x000500aau, 0x0000000bu, 0x0000000cu, 0x00000009u, 0x0000000au, 0x000300f7u, 0x0000000eu,
    0x00000000u, 0x000400fau, 0x0000000cu, 0x0000000du, 0x0000000eu, 0x000200f8u, 0x0000000du, 0x00050041u,
    0x00000007u, 0x00000014u, 0x00000013u, 0x0000000au, 0x0004003du, 0x00000006u, 0x00000015u, 0x00000014u,
    0x0003003eu, 0x00000010u, 0x00000015u, 0x0003003eu, 0x0000001cu, 0x0000001eu, 0x0004003du, 0x00000006u,
    0x00000029u, 0x00000010u, 0x00050084u, 0x00000006u, 0x0000002bu, 0x00000029u, 0x0000002au, 0x00060041u,
    0x0000002cu, 0x0000002du, 0x00000026u, 0x00000028u, 0x0000002bu, 0x0004003du, 0x00000006u, 0x0000002eu,
    0x00000010u, 0x00050084u, 0x00000006u, 0x0000002fu, 0x0000002eu, 0x0000002au, 0x00071169u, 0x00000020u,
    0x00000030u, 0x0000002du, 0x00000028u, 0x00000018u, 0x00000000u, 0x0003003eu, 0x00000022u, 0x00000030u,
    0x00060041u, 0x0000002cu, 0x00000039u, 0x00000038u, 0x00000028u, 0x0000000au, 0x00071169u, 0x00000032u,
    0x0000003au, 0x00000039u, 0x00000028u, 0x00000018u, 0x00000000u, 0x0003003eu, 0x00000034u, 0x0000003au,
    0x0004003du, 0x00000020u, 0x0000003bu, 0x00000022u, 0x0004003du, 0x00000032u, 0x0000003cu, 0x00000034u,
    0x0004003du, 0x0000001au, 0x0000003du, 0x0000001cu, 0x0006116bu, 0x0000001au, 0x0000003eu, 0x0000003bu,
    0x0000003cu, 0x0000003du, 0x0003003eu, 0x0000001cu, 0x0000003eu, 0x0004003du, 0x0000001au, 0x0000003fu,
    0x0000001cu, 0x0004003du, 0x00000006u, 0x00000044u, 0x00000010u, 0x00050084u, 0x00000006u, 0x00000045u,
    0x00000044u, 0x0000002au, 0x00060041u, 0x00000046u, 0x00000047u, 0x00000043u, 0x00000028u, 0x00000045u,
    0x0004003du, 0x00000006u, 0x00000048u, 0x00000010u, 0x00050084u, 0x00000006u, 0x00000049u, 0x00000048u,
    0x0000002au, 0x0006116au, 0x00000047u, 0x0000003fu, 0x00000028u, 0x00000018u, 0x00000000u, 0x000200f9u,
    0x0000000eu, 0x000200f8u, 0x0000000eu, 0x000100fdu, 0x00010038u,
};

struct VulkanLoader {
#if defined(_WIN32)
    HMODULE lib = nullptr;
#else
    void* lib = nullptr;
#endif
    using PfnVoid = void(SONDER_VK_CALL*)();
    using PfnGetInstanceProcAddr = PfnVoid(SONDER_VK_CALL*)(VkInstance, const char*);
    PfnGetInstanceProcAddr get_instance_proc_addr = nullptr;

    bool load() noexcept {
#if defined(_WIN32)
        lib = ::LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (lib == nullptr) return false;
        get_instance_proc_addr =
            reinterpret_cast<PfnGetInstanceProcAddr>(::GetProcAddress(lib, "vkGetInstanceProcAddr"));
#else
        lib = ::dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return false;
        get_instance_proc_addr = reinterpret_cast<PfnGetInstanceProcAddr>(::dlsym(lib, "vkGetInstanceProcAddr"));
#endif
        return get_instance_proc_addr != nullptr;
    }

    ~VulkanLoader() noexcept {
        if (lib != nullptr) {
#if defined(_WIN32)
            ::FreeLibrary(lib);
#else
            ::dlclose(lib);
#endif
        }
    }

    template <typename Fn>
    Fn resolve(VkInstance instance, const char* name) const noexcept {
        if (get_instance_proc_addr == nullptr) return nullptr;
        return reinterpret_cast<Fn>(get_instance_proc_addr(instance, name));
    }
};

}  // namespace

const char* to_string(CoopmatCanaryStatus status) noexcept {
    switch (status) {
        case CoopmatCanaryStatus::not_applicable: return "not_applicable";
        case CoopmatCanaryStatus::skipped_env_disabled: return "skipped_env_disabled";
        case CoopmatCanaryStatus::passed: return "passed";
        case CoopmatCanaryStatus::failed: return "failed";
        case CoopmatCanaryStatus::error: return "error";
    }
    return "unknown";
}

bool needs_amd_windows_coopmat_canary(const GpuDeviceInfo& gpu, bool is_windows) noexcept {
    if (!is_windows) return false;
    const bool is_amd = (gpu.vendor == GpuVendor::amd) || (gpu.vendor_id == kAmdVendorId);
    if (!is_amd) return false;
    if (gpu.matrix_cores == "none") return false;
    // Mirror ggml_vk_khr_cooperative_matrix_support (ggml-vulkan.cpp:16858-16863):
    // skip AMD GCN/RDNA1/RDNA2 GPUs that ggml-vulkan will never use for KHR coopmat,
    // so a failure on an unused iGPU cannot disable coopmat globally on other GPUs.
    if (gpu.ggml_coopmat_eligible.has_value() && !*gpu.ggml_coopmat_eligible) return false;
    return true;
}

std::string canonical_coopmat_cache_key(const CoopmatCanaryKey& key) {
    const std::string drv = sanitize_single_line(key.driver_build, kMaxDriverBuildLen);
    const std::uint32_t ub = key.ubatch == 0u ? 64u : key.ubatch;
    return "amd:" + format_hex_u32(key.vendor_id, 4) + ":" + format_hex_u32(key.device_id, 4) + ":ub" +
           std::to_string(ub) + ":" + (drv.empty() ? "unknown" : drv);
}

std::string default_coopmat_canary_cache_path() {
    const std::string env_cache = read_env_var("SONDER_COOPMAT_CANARY_CACHE");
    if (!env_cache.empty()) {
        fs::path custom_path;
        if (is_safe_file_path(env_cache, custom_path)) {
            return custom_path.string();
        }
    }
#if defined(_WIN32)
    std::string base = read_env_var("LOCALAPPDATA");
    if (base.empty()) base = read_env_var("TEMP");
    if (base.empty()) return {};
    return (fs::path(base) / "sonder-inference" / "coopmat-canary-v1.json").string();
#else
    std::string base = read_env_var("XDG_CACHE_HOME");
    if (base.empty()) {
        const std::string home = read_env_var("HOME");
        if (home.empty()) return {};
        return (fs::path(home) / ".cache" / "sonder-inference" / "coopmat-canary-v1.json").string();
    }
    return (fs::path(base) / "sonder-inference" / "coopmat-canary-v1.json").string();
#endif
}

std::string find_coopmat_canary_executable(const std::string& explicit_path) {
    if (!explicit_path.empty()) {
        fs::path p;
        if (is_safe_file_path(explicit_path, p) && is_regular_executable_file(p)) {
            return p.string();
        }
        return {};
    }
    const std::string env_exe = read_env_var("SONDER_INFER_EXE");
    if (!env_exe.empty()) {
        fs::path p;
        if (is_safe_file_path(env_exe, p) && is_regular_executable_file(p)) {
            return p.string();
        }
    }
    const std::string self_raw = current_executable_path();
    if (self_raw.empty()) return {};
    fs::path self;
    if (!is_safe_file_path(self_raw, self)) return {};
    const std::string stem = lower_ascii(self.stem().string());
    if (stem == "sonder-infer" || stem.rfind("sonder-infer-", 0) == 0) {
        if (is_regular_executable_file(self)) {
            return self.string();
        }
    }
#if defined(_WIN32)
    const fs::path sibling = self.parent_path() / "sonder-infer.exe";
#else
    const fs::path sibling = self.parent_path() / "sonder-infer";
#endif
    if (is_regular_executable_file(sibling)) {
        return sibling.string();
    }
    return {};
}

std::vector<CoopmatCanaryRecord> load_coopmat_canary_cache(const std::string& path) {
    fs::path file_path;
    if (!is_safe_file_path(path, file_path)) return {};
    std::error_code ec;
    if (fs::is_symlink(file_path, ec) || ec) return {};
    if (!fs::is_regular_file(file_path, ec) || ec) return {};
    const auto fsize = fs::file_size(file_path, ec);
    if (ec || fsize == 0u || fsize > kMaxCacheBytes) return {};

    std::ifstream in(file_path, std::ios::binary);
    if (!in) return {};
    std::string raw(static_cast<std::size_t>(fsize), '\0');
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (!in && !in.eof()) return {};
    raw.resize(static_cast<std::size_t>(in.gcount()));

    auto parsed = json::parse(raw);
    if (!parsed.ok() || !parsed.value().is_object()) return {};
    const auto& root = parsed.value().as_object();
    const json::Value* schema_val = root.find("schema");
    if (schema_val == nullptr || !schema_val->is_string() || schema_val->as_string() != kCacheSchema) {
        return {};
    }
    const json::Value* entries_val = root.find("entries");
    if (entries_val == nullptr || !entries_val->is_array()) {
        return {};
    }
    const auto& arr = entries_val->as_array();
    if (arr.size() > kMaxCacheEntries) return {};

    std::vector<CoopmatCanaryRecord> out;
    out.reserve(arr.size());
    for (const auto& item : arr) {
        if (!item.is_object()) continue;
        const auto& obj = item.as_object();
        auto get_u64 = [&](const char* k, std::uint64_t& dst) -> bool {
            const json::Value* v = obj.find(k);
            if (v == nullptr || !v->is_integer()) return false;
            const auto signed_v = v->try_as_int();
            if (!signed_v || *signed_v < 0) return false;
            dst = static_cast<std::uint64_t>(*signed_v);
            return true;
        };
        auto get_str = [&](const char* k, std::string& dst, std::size_t max_len) -> bool {
            const json::Value* v = obj.find(k);
            if (v == nullptr || !v->is_string()) return false;
            dst = sanitize_single_line(v->as_string(), max_len);
            return true;
        };
        auto get_bool = [&](const char* k, bool& dst) -> bool {
            const json::Value* v = obj.find(k);
            if (v == nullptr || !v->is_bool()) return false;
            dst = v->as_bool();
            return true;
        };

        std::uint64_t vid = 0;
        std::uint64_t did = 0;
        std::uint64_t ub = 0;
        std::string status_str;
        std::string drv;
        if (!get_u64("vendor_id", vid) || vid > 0xFFFFFFFFull) continue;
        if (!get_u64("device_id", did) || did > 0xFFFFFFFFull) continue;
        if (!get_u64("ubatch", ub) || ub == 0u || ub > 4096u) continue;
        if (!get_str("status", status_str, 32u)) continue;
        const auto parsed_status = parse_cached_status(status_str);
        if (!parsed_status) continue;
        (void)get_str("driver_build", drv, kMaxDriverBuildLen);

        CoopmatCanaryRecord rec;
        rec.key.vendor_id = static_cast<std::uint32_t>(vid);
        rec.key.device_id = static_cast<std::uint32_t>(did);
        rec.key.driver_build = std::move(drv);
        rec.key.ubatch = static_cast<std::uint32_t>(ub);
        rec.status = *parsed_status;
        rec.disable_coopmat = (rec.status == CoopmatCanaryStatus::failed);
        bool stored_disable = rec.disable_coopmat;
        if (get_bool("disable_coopmat", stored_disable)) {
            rec.disable_coopmat = stored_disable || (rec.status == CoopmatCanaryStatus::failed);
        }
        (void)get_str("gpu_id", rec.gpu_id, 64u);
        (void)get_str("gpu_name", rec.gpu_name, 128u);
        (void)get_bool("is_integrated_uma", rec.is_integrated_uma);
        std::uint64_t tmp_u64 = 0;
        if (get_u64("elapsed_ms", tmp_u64)) rec.elapsed_ms = tmp_u64;
        if (get_u64("exit_code", tmp_u64) && tmp_u64 <= 0xFFFFFFFFull) {
            rec.exit_code = static_cast<std::uint32_t>(tmp_u64);
        }
        (void)get_str("reason", rec.reason, kMaxReasonLen);
        rec.from_cache = true;
        out.push_back(std::move(rec));
    }
    return out;
}

bool save_coopmat_canary_cache(const std::string& path, const std::vector<CoopmatCanaryRecord>& entries) {
    fs::path file_path;
    if (!is_safe_file_path(path, file_path)) return false;
    std::error_code ec;
    if (fs::is_symlink(file_path, ec)) return false;

    const fs::path parent = file_path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) return false;
    }

    // Keep only definitive passed/failed entries, capped to kMaxCacheEntries.
    std::vector<CoopmatCanaryRecord> filtered;
    filtered.reserve(std::min(entries.size(), kMaxCacheEntries));
    for (const auto& e : entries) {
        if (e.status != CoopmatCanaryStatus::passed && e.status != CoopmatCanaryStatus::failed) continue;
        filtered.push_back(e);
    }
    if (filtered.size() > kMaxCacheEntries) {
        filtered.erase(filtered.begin(), filtered.end() - static_cast<std::ptrdiff_t>(kMaxCacheEntries));
    }

    auto serialize_entries = [&](std::size_t start_idx) -> std::string {
        json::Array arr;
        arr.reserve(filtered.size() - start_idx);
        for (std::size_t i = start_idx; i < filtered.size(); ++i) {
            const auto& r = filtered[i];
            arr.emplace_back(json::Object{
                {"key", canonical_coopmat_cache_key(r.key)},
                {"vendor_id", static_cast<std::int64_t>(r.key.vendor_id)},
                {"device_id", static_cast<std::int64_t>(r.key.device_id)},
                {"driver_build", sanitize_single_line(r.key.driver_build, kMaxDriverBuildLen)},
                {"ubatch", static_cast<std::int64_t>(r.key.ubatch == 0u ? 64u : r.key.ubatch)},
                {"gpu_id", sanitize_single_line(r.gpu_id, 64u)},
                {"gpu_name", sanitize_single_line(r.gpu_name, 128u)},
                {"is_integrated_uma", r.is_integrated_uma},
                {"status", to_string(r.status)},
                {"disable_coopmat", r.disable_coopmat},
                {"elapsed_ms", static_cast<std::int64_t>(r.elapsed_ms)},
                {"exit_code", static_cast<std::int64_t>(r.exit_code)},
                {"reason", sanitize_single_line(r.reason, kMaxReasonLen)},
            });
        }
        return json::Value(json::Object{{"schema", kCacheSchema}, {"entries", std::move(arr)}}).dump() + "\n";
    };

    std::size_t start_idx = 0;
    std::string payload = serialize_entries(start_idx);
    while (payload.size() > kMaxCacheBytes && start_idx < filtered.size()) {
        ++start_idx;
        payload = serialize_entries(start_idx);
    }
    if (payload.size() > kMaxCacheBytes) return false;

    static std::atomic<std::uint64_t> tmp_seq{0};
    const std::uint64_t seq = tmp_seq.fetch_add(1u, std::memory_order_relaxed);
#if defined(_WIN32)
    const std::uint32_t pid = static_cast<std::uint32_t>(::GetCurrentProcessId());
#else
    const std::uint32_t pid = static_cast<std::uint32_t>(::getpid());
#endif
    const fs::path tmp_path = file_path.parent_path() /
                              (file_path.filename().string() + ".tmp." + std::to_string(pid) + "." + std::to_string(seq));
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        out.flush();
        if (!out) {
            out.close();
            fs::remove(tmp_path, ec);
            return false;
        }
    }

#if defined(_WIN32)
    const BOOL moved =
        ::MoveFileExW(tmp_path.c_str(), file_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    if (!moved) {
        fs::remove(tmp_path, ec);
        return false;
    }
#else
    if (::rename(tmp_path.c_str(), file_path.c_str()) != 0) {
        fs::remove(tmp_path, ec);
        return false;
    }
#endif
    return true;
}

CoopmatCanaryProbeResult run_vulkan_coopmat_canary_in_process(std::uint32_t vendor_id, std::uint32_t device_id,
                                                              std::uint32_t ubatch) {
#if defined(_WIN32)
    ::SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
    CoopmatCanaryProbeResult res;
    const std::uint32_t clamped_ubatch = std::clamp(ubatch == 0u ? 64u : ubatch, 16u, 4096u);
    const std::uint32_t tiles = (clamped_ubatch + 15u) / 16u;
    res.ubatch_tested = tiles * 16u;

    VulkanLoader vk;
    if (!vk.load()) {
        res.reason = "Vulkan loader unavailable";
        return res;
    }

    using PfnCreateInstance = VkResult(SONDER_VK_CALL*)(const VkInstanceCreateInfoRaw*, const void*, VkInstance*);
    using PfnDestroyInstance = void(SONDER_VK_CALL*)(VkInstance, const void*);
    using PfnEnumPhysDevs = VkResult(SONDER_VK_CALL*)(VkInstance, std::uint32_t*, VkPhysicalDevice*);
    using PfnGetPhysProps = void(SONDER_VK_CALL*)(VkPhysicalDevice, VkPhysicalDevicePropertiesRaw*);
    using PfnEnumDevExts =
        VkResult(SONDER_VK_CALL*)(VkPhysicalDevice, const char*, std::uint32_t*, VkExtensionPropertiesRaw*);
    using PfnGetCoopMatProps =
        VkResult(SONDER_VK_CALL*)(VkPhysicalDevice, std::uint32_t*, VkCoopMatPropertiesKHRRaw*);
    using PfnGetPhysFeatures2 = void(SONDER_VK_CALL*)(VkPhysicalDevice, VkPhysicalDeviceFeatures2Raw*);
    using PfnGetQueueFamilies = void(SONDER_VK_CALL*)(VkPhysicalDevice, std::uint32_t*, VkQueueFamilyPropertiesRaw*);
    using PfnGetPhysMemProps = void(SONDER_VK_CALL*)(VkPhysicalDevice, VkPhysicalDeviceMemoryPropertiesRaw*);
    using PfnCreateDevice =
        VkResult(SONDER_VK_CALL*)(VkPhysicalDevice, const VkDeviceCreateInfoRaw*, const void*, VkDevice*);
    using PfnDestroyDevice = void(SONDER_VK_CALL*)(VkDevice, const void*);
    using PfnGetDeviceQueue = void(SONDER_VK_CALL*)(VkDevice, std::uint32_t, std::uint32_t, VkQueue*);
    using PfnCreateBuffer = VkResult(SONDER_VK_CALL*)(VkDevice, const VkBufferCreateInfoRaw*, const void*, VkBuffer*);
    using PfnDestroyBuffer = void(SONDER_VK_CALL*)(VkDevice, VkBuffer, const void*);
    using PfnGetBufMemReq = void(SONDER_VK_CALL*)(VkDevice, VkBuffer, VkMemoryRequirementsRaw*);
    using PfnAllocMem =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkMemoryAllocateInfoRaw*, const void*, VkDeviceMemory*);
    using PfnFreeMem = void(SONDER_VK_CALL*)(VkDevice, VkDeviceMemory, const void*);
    using PfnBindBufMem = VkResult(SONDER_VK_CALL*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
    using PfnMapMem = VkResult(SONDER_VK_CALL*)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
    using PfnUnmapMem = void(SONDER_VK_CALL*)(VkDevice, VkDeviceMemory);
    using PfnCreateShader =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkShaderModuleCreateInfoRaw*, const void*, VkShaderModule*);
    using PfnDestroyShader = void(SONDER_VK_CALL*)(VkDevice, VkShaderModule, const void*);
    using PfnCreateDescSetLayout = VkResult(SONDER_VK_CALL*)(VkDevice, const VkDescriptorSetLayoutCreateInfoRaw*,
                                                             const void*, VkDescriptorSetLayout*);
    using PfnDestroyDescSetLayout = void(SONDER_VK_CALL*)(VkDevice, VkDescriptorSetLayout, const void*);
    using PfnCreatePipeLayout =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkPipelineLayoutCreateInfoRaw*, const void*, VkPipelineLayout*);
    using PfnDestroyPipeLayout = void(SONDER_VK_CALL*)(VkDevice, VkPipelineLayout, const void*);
    using PfnCreateComputePipelines = VkResult(SONDER_VK_CALL*)(VkDevice, VkPipelineCache, std::uint32_t,
                                                                const VkComputePipelineCreateInfoRaw*, const void*,
                                                                VkPipeline*);
    using PfnDestroyPipeline = void(SONDER_VK_CALL*)(VkDevice, VkPipeline, const void*);
    using PfnCreateDescPool =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkDescriptorPoolCreateInfoRaw*, const void*, VkDescriptorPool*);
    using PfnDestroyDescPool = void(SONDER_VK_CALL*)(VkDevice, VkDescriptorPool, const void*);
    using PfnAllocDescSets =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkDescriptorSetAllocateInfoRaw*, VkDescriptorSet*);
    using PfnUpdateDescSets =
        void(SONDER_VK_CALL*)(VkDevice, std::uint32_t, const VkWriteDescriptorSetRaw*, std::uint32_t, const void*);
    using PfnCreateCmdPool =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkCommandPoolCreateInfoRaw*, const void*, VkCommandPool*);
    using PfnDestroyCmdPool = void(SONDER_VK_CALL*)(VkDevice, VkCommandPool, const void*);
    using PfnAllocCmdBufs =
        VkResult(SONDER_VK_CALL*)(VkDevice, const VkCommandBufferAllocateInfoRaw*, VkCommandBuffer*);
    using PfnBeginCmdBuf = VkResult(SONDER_VK_CALL*)(VkCommandBuffer, const VkCommandBufferBeginInfoRaw*);
    using PfnEndCmdBuf = VkResult(SONDER_VK_CALL*)(VkCommandBuffer);
    using PfnCmdBindPipeline = void(SONDER_VK_CALL*)(VkCommandBuffer, std::uint32_t, VkPipeline);
    using PfnCmdBindDescSets = void(SONDER_VK_CALL*)(VkCommandBuffer, std::uint32_t, VkPipelineLayout, std::uint32_t,
                                                     std::uint32_t, const VkDescriptorSet*, std::uint32_t,
                                                     const std::uint32_t*);
    using PfnCmdDispatch = void(SONDER_VK_CALL*)(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t);
    using PfnQueueSubmit = VkResult(SONDER_VK_CALL*)(VkQueue, std::uint32_t, const VkSubmitInfoRaw*, VkFence);
    using PfnQueueWaitIdle = VkResult(SONDER_VK_CALL*)(VkQueue);

    const auto create_instance = vk.resolve<PfnCreateInstance>(nullptr, "vkCreateInstance");
    if (create_instance == nullptr) {
        res.reason = "vkCreateInstance missing";
        return res;
    }

    VkApplicationInfoRaw app{};
    app.sType = kVkStypeApplicationInfo;
    app.pApplicationName = "sonder-coopmat-canary";
    app.applicationVersion = 1u;
    app.pEngineName = "sonder-inference";
    app.engineVersion = 1u;
    app.apiVersion = kVkApiVersion12;

    VkInstanceCreateInfoRaw ici{};
    ici.sType = kVkStypeInstanceCreateInfo;
    ici.pApplicationInfo = &app;

    VkInstance instance = nullptr;
    if (create_instance(&ici, nullptr, &instance) != kVkSuccess || instance == nullptr) {
        res.reason = "vkCreateInstance failed";
        return res;
    }

    const auto destroy_instance = vk.resolve<PfnDestroyInstance>(instance, "vkDestroyInstance");
    struct InstanceGuard {
        VkInstance inst = nullptr;
        PfnDestroyInstance destroy = nullptr;
        ~InstanceGuard() noexcept {
            if (inst != nullptr && destroy != nullptr) destroy(inst, nullptr);
        }
    } inst_guard{instance, destroy_instance};

    const auto enum_phys = vk.resolve<PfnEnumPhysDevs>(instance, "vkEnumeratePhysicalDevices");
    const auto get_props = vk.resolve<PfnGetPhysProps>(instance, "vkGetPhysicalDeviceProperties");
    const auto enum_exts = vk.resolve<PfnEnumDevExts>(instance, "vkEnumerateDeviceExtensionProperties");
    const auto get_coop_props =
        vk.resolve<PfnGetCoopMatProps>(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    const auto get_features2 = vk.resolve<PfnGetPhysFeatures2>(instance, "vkGetPhysicalDeviceFeatures2");
    const auto get_qfam = vk.resolve<PfnGetQueueFamilies>(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    const auto get_mem_props = vk.resolve<PfnGetPhysMemProps>(instance, "vkGetPhysicalDeviceMemoryProperties");
    const auto create_dev = vk.resolve<PfnCreateDevice>(instance, "vkCreateDevice");
    const auto destroy_dev = vk.resolve<PfnDestroyDevice>(instance, "vkDestroyDevice");
    if (enum_phys == nullptr || get_props == nullptr || enum_exts == nullptr || get_features2 == nullptr ||
        get_qfam == nullptr || get_mem_props == nullptr || create_dev == nullptr || destroy_dev == nullptr) {
        res.reason = "required Vulkan 1.2 instance entrypoints missing";
        return res;
    }

    std::uint32_t dev_count = 0;
    if (enum_phys(instance, &dev_count, nullptr) != kVkSuccess || dev_count == 0u) {
        res.reason = "no Vulkan physical devices";
        return res;
    }
    std::vector<VkPhysicalDevice> phys_devs(dev_count);
    if (enum_phys(instance, &dev_count, phys_devs.data()) != kVkSuccess) {
        res.reason = "vkEnumeratePhysicalDevices failed";
        return res;
    }

    VkPhysicalDevice target = nullptr;
    VkPhysicalDevicePropertiesRaw target_props{};
    for (VkPhysicalDevice pd : phys_devs) {
        VkPhysicalDevicePropertiesRaw props{};
        get_props(pd, &props);
        if (props.vendorID == vendor_id && props.deviceID == device_id) {
            target = pd;
            target_props = props;
            break;
        }
    }
    if (target == nullptr) {
        res.reason = "matching Vulkan physical device not found";
        return res;
    }
    if (target_props.apiVersion < kVkApiVersion12) {
        res.ok = true;
        res.coopmat_supported = false;
        res.reason = "device Vulkan API < 1.2 (coopmat not used)";
        return res;
    }

    std::uint32_t ext_count = 0;
    if (enum_exts(target, nullptr, &ext_count, nullptr) != kVkSuccess) {
        res.reason = "vkEnumerateDeviceExtensionProperties count failed";
        return res;
    }
    std::vector<VkExtensionPropertiesRaw> exts(ext_count);
    if (ext_count > 0u && enum_exts(target, nullptr, &ext_count, exts.data()) != kVkSuccess) {
        res.reason = "vkEnumerateDeviceExtensionProperties failed";
        return res;
    }
    bool has_coopmat_ext = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
            has_coopmat_ext = true;
            break;
        }
    }
    if (!has_coopmat_ext) {
        res.ok = true;
        res.coopmat_supported = false;
        res.reason = "VK_KHR_cooperative_matrix not advertised";
        return res;
    }
    res.coopmat_supported = true;

    // Phase 1 (PR #22750 entrypoint): query cooperative-matrix properties.
    if (get_coop_props == nullptr) {
        res.reason = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR entrypoint missing";
        return res;
    }
    std::uint32_t shape_count = 0;
    if (get_coop_props(target, &shape_count, nullptr) != kVkSuccess) {
        res.reason = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR count failed";
        return res;
    }
    res.shapes_found = shape_count;
    if (shape_count == 0u) {
        res.reason = "no cooperative matrix properties returned";
        return res;
    }
    std::vector<VkCoopMatPropertiesKHRRaw> shapes(shape_count);
    for (auto& s : shapes) {
        s = VkCoopMatPropertiesKHRRaw{};
        s.sType = kVkStypeCoopMatPropertiesKHR;
    }
    if (get_coop_props(target, &shape_count, shapes.data()) != kVkSuccess) {
        res.reason = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR query failed";
        return res;
    }
    bool has_16x16x16_f16_f32 = false;
    for (const auto& s : shapes) {
        if (s.MSize == 16u && s.NSize == 16u && s.KSize == 16u && s.AType == kVkComponentFloat16KHR &&
            s.BType == kVkComponentFloat16KHR && s.CType == kVkComponentFloat32KHR &&
            s.ResultType == kVkComponentFloat32KHR && s.scope == kVkScopeSubgroupKHR) {
            has_16x16x16_f16_f32 = true;
            break;
        }
    }
    if (!has_16x16x16_f16_f32) {
        res.reason = "no 16x16x16 FP16->FP32 subgroup cooperative matrix shape";
        return res;
    }

    // Phase 2: create logical device and run generic 16x16x16 FP16->FP32 coopMatMulAdd smoke dispatch.
    VkPhysicalDeviceCoopMatFeaturesKHRRaw cm_feat{};
    cm_feat.sType = kVkStypePhysicalDeviceCoopMatFeaturesKHR;
    VkPhysicalDevice16BitStorageFeaturesRaw s16_feat{};
    s16_feat.sType = kVkStypePhysicalDevice16BitStorageFeatures;
    s16_feat.pNext = &cm_feat;
    VkPhysicalDeviceShaderFloat16Int8FeaturesRaw f16_feat{};
    f16_feat.sType = kVkStypePhysicalDeviceShaderFloat16Int8Features;
    f16_feat.pNext = &s16_feat;
    VkPhysicalDeviceVulkanMemoryModelFeaturesRaw vmm_feat{};
    vmm_feat.sType = kVkStypePhysicalDeviceVulkanMemoryModelFeatures;
    vmm_feat.pNext = &f16_feat;
    VkPhysicalDeviceFeatures2Raw feat2{};
    feat2.sType = kVkStypePhysicalDeviceFeatures2;
    feat2.pNext = &vmm_feat;
    get_features2(target, &feat2);

    if (!cm_feat.cooperativeMatrix || !s16_feat.storageBuffer16BitAccess || !f16_feat.shaderFloat16 ||
        !vmm_feat.vulkanMemoryModel || !vmm_feat.vulkanMemoryModelDeviceScope) {
        res.reason = "required Vulkan 1.2 coopmat/fp16/memory-model features unsupported";
        return res;
    }

    std::uint32_t qf_count = 0;
    get_qfam(target, &qf_count, nullptr);
    std::vector<VkQueueFamilyPropertiesRaw> qfams(qf_count);
    if (qf_count > 0u) get_qfam(target, &qf_count, qfams.data());
    std::uint32_t compute_qf = UINT32_MAX;
    for (std::uint32_t i = 0; i < qf_count; ++i) {
        if ((qfams[i].queueFlags & kVkQueueComputeBit) != 0u && qfams[i].queueCount > 0u) {
            compute_qf = i;
            break;
        }
    }
    if (compute_qf == UINT32_MAX) {
        res.reason = "no Vulkan compute queue family";
        return res;
    }

    VkPhysicalDeviceCoopMatFeaturesKHRRaw en_cm{};
    en_cm.sType = kVkStypePhysicalDeviceCoopMatFeaturesKHR;
    en_cm.cooperativeMatrix = 1u;
    VkPhysicalDevice16BitStorageFeaturesRaw en_s16{};
    en_s16.sType = kVkStypePhysicalDevice16BitStorageFeatures;
    en_s16.pNext = &en_cm;
    en_s16.storageBuffer16BitAccess = 1u;
    VkPhysicalDeviceShaderFloat16Int8FeaturesRaw en_f16{};
    en_f16.sType = kVkStypePhysicalDeviceShaderFloat16Int8Features;
    en_f16.pNext = &en_s16;
    en_f16.shaderFloat16 = 1u;
    VkPhysicalDeviceVulkanMemoryModelFeaturesRaw en_vmm{};
    en_vmm.sType = kVkStypePhysicalDeviceVulkanMemoryModelFeatures;
    en_vmm.pNext = &en_f16;
    en_vmm.vulkanMemoryModel = 1u;
    en_vmm.vulkanMemoryModelDeviceScope = 1u;

    const float qprio = 1.0f;
    VkDeviceQueueCreateInfoRaw dqci{};
    dqci.sType = kVkStypeDeviceQueueCreateInfo;
    dqci.queueFamilyIndex = compute_qf;
    dqci.queueCount = 1u;
    dqci.pQueuePriorities = &qprio;

    const char* const dev_exts[] = {"VK_KHR_cooperative_matrix"};
    VkDeviceCreateInfoRaw dci{};
    dci.sType = kVkStypeDeviceCreateInfo;
    dci.pNext = &en_vmm;
    dci.queueCreateInfoCount = 1u;
    dci.pQueueCreateInfos = &dqci;
    dci.enabledExtensionCount = 1u;
    dci.ppEnabledExtensionNames = dev_exts;

    VkDevice device = nullptr;
    if (create_dev(target, &dci, nullptr, &device) != kVkSuccess || device == nullptr) {
        res.reason = "vkCreateDevice failed";
        return res;
    }

    const auto get_queue = vk.resolve<PfnGetDeviceQueue>(instance, "vkGetDeviceQueue");
    const auto create_buf = vk.resolve<PfnCreateBuffer>(instance, "vkCreateBuffer");
    const auto destroy_buf = vk.resolve<PfnDestroyBuffer>(instance, "vkDestroyBuffer");
    const auto get_buf_req = vk.resolve<PfnGetBufMemReq>(instance, "vkGetBufferMemoryRequirements");
    const auto alloc_mem = vk.resolve<PfnAllocMem>(instance, "vkAllocateMemory");
    const auto free_mem = vk.resolve<PfnFreeMem>(instance, "vkFreeMemory");
    const auto bind_buf = vk.resolve<PfnBindBufMem>(instance, "vkBindBufferMemory");
    const auto map_mem = vk.resolve<PfnMapMem>(instance, "vkMapMemory");
    const auto unmap_mem = vk.resolve<PfnUnmapMem>(instance, "vkUnmapMemory");
    const auto create_shader = vk.resolve<PfnCreateShader>(instance, "vkCreateShaderModule");
    const auto destroy_shader = vk.resolve<PfnDestroyShader>(instance, "vkDestroyShaderModule");
    const auto create_dsl = vk.resolve<PfnCreateDescSetLayout>(instance, "vkCreateDescriptorSetLayout");
    const auto destroy_dsl = vk.resolve<PfnDestroyDescSetLayout>(instance, "vkDestroyDescriptorSetLayout");
    const auto create_pl = vk.resolve<PfnCreatePipeLayout>(instance, "vkCreatePipelineLayout");
    const auto destroy_pl = vk.resolve<PfnDestroyPipeLayout>(instance, "vkDestroyPipelineLayout");
    const auto create_cp = vk.resolve<PfnCreateComputePipelines>(instance, "vkCreateComputePipelines");
    const auto destroy_pipe = vk.resolve<PfnDestroyPipeline>(instance, "vkDestroyPipeline");
    const auto create_dp = vk.resolve<PfnCreateDescPool>(instance, "vkCreateDescriptorPool");
    const auto destroy_dp = vk.resolve<PfnDestroyDescPool>(instance, "vkDestroyDescriptorPool");
    const auto alloc_ds = vk.resolve<PfnAllocDescSets>(instance, "vkAllocateDescriptorSets");
    const auto update_ds = vk.resolve<PfnUpdateDescSets>(instance, "vkUpdateDescriptorSets");
    const auto create_cmdpool = vk.resolve<PfnCreateCmdPool>(instance, "vkCreateCommandPool");
    const auto destroy_cmdpool = vk.resolve<PfnDestroyCmdPool>(instance, "vkDestroyCommandPool");
    const auto alloc_cmd = vk.resolve<PfnAllocCmdBufs>(instance, "vkAllocateCommandBuffers");
    const auto begin_cmd = vk.resolve<PfnBeginCmdBuf>(instance, "vkBeginCommandBuffer");
    const auto end_cmd = vk.resolve<PfnEndCmdBuf>(instance, "vkEndCommandBuffer");
    const auto cmd_bind_pipe = vk.resolve<PfnCmdBindPipeline>(instance, "vkCmdBindPipeline");
    const auto cmd_bind_ds = vk.resolve<PfnCmdBindDescSets>(instance, "vkCmdBindDescriptorSets");
    const auto cmd_dispatch = vk.resolve<PfnCmdDispatch>(instance, "vkCmdDispatch");
    const auto queue_submit = vk.resolve<PfnQueueSubmit>(instance, "vkQueueSubmit");
    const auto queue_wait = vk.resolve<PfnQueueWaitIdle>(instance, "vkQueueWaitIdle");

    struct ComputeResources {
        VkDevice dev = nullptr;
        PfnDestroyDevice destroy_dev_fn = nullptr;
        PfnDestroyBuffer destroy_buf_fn = nullptr;
        PfnFreeMem free_mem_fn = nullptr;
        PfnDestroyShader destroy_shader_fn = nullptr;
        PfnDestroyDescSetLayout destroy_dsl_fn = nullptr;
        PfnDestroyPipeLayout destroy_pl_fn = nullptr;
        PfnDestroyPipeline destroy_pipe_fn = nullptr;
        PfnDestroyDescPool destroy_dp_fn = nullptr;
        PfnDestroyCmdPool destroy_cp_fn = nullptr;

        VkCommandPool cmd_pool = 0;
        VkDescriptorPool desc_pool = 0;
        VkPipeline pipeline = 0;
        VkPipelineLayout pipe_layout = 0;
        VkDescriptorSetLayout desc_layout = 0;
        VkShaderModule shader = 0;
        VkBuffer bufs[3] = {0, 0, 0};
        VkDeviceMemory mems[3] = {0, 0, 0};

        ~ComputeResources() noexcept {
            if (dev == nullptr) return;
            if (cmd_pool != 0 && destroy_cp_fn != nullptr) destroy_cp_fn(dev, cmd_pool, nullptr);
            if (desc_pool != 0 && destroy_dp_fn != nullptr) destroy_dp_fn(dev, desc_pool, nullptr);
            if (pipeline != 0 && destroy_pipe_fn != nullptr) destroy_pipe_fn(dev, pipeline, nullptr);
            if (pipe_layout != 0 && destroy_pl_fn != nullptr) destroy_pl_fn(dev, pipe_layout, nullptr);
            if (desc_layout != 0 && destroy_dsl_fn != nullptr) destroy_dsl_fn(dev, desc_layout, nullptr);
            if (shader != 0 && destroy_shader_fn != nullptr) destroy_shader_fn(dev, shader, nullptr);
            for (int i = 0; i < 3; ++i) {
                if (bufs[i] != 0 && destroy_buf_fn != nullptr) destroy_buf_fn(dev, bufs[i], nullptr);
                if (mems[i] != 0 && free_mem_fn != nullptr) free_mem_fn(dev, mems[i], nullptr);
            }
            if (destroy_dev_fn != nullptr) destroy_dev_fn(dev, nullptr);
        }
    } r{device, destroy_dev, destroy_buf, free_mem, destroy_shader, destroy_dsl, destroy_pl, destroy_pipe, destroy_dp,
        destroy_cmdpool};

    if (get_queue == nullptr || create_buf == nullptr || destroy_buf == nullptr || get_buf_req == nullptr ||
        alloc_mem == nullptr || free_mem == nullptr || bind_buf == nullptr || map_mem == nullptr ||
        unmap_mem == nullptr || create_shader == nullptr || destroy_shader == nullptr || create_dsl == nullptr ||
        destroy_dsl == nullptr || create_pl == nullptr || destroy_pl == nullptr || create_cp == nullptr ||
        destroy_pipe == nullptr || create_dp == nullptr || destroy_dp == nullptr || alloc_ds == nullptr ||
        update_ds == nullptr || create_cmdpool == nullptr || destroy_cmdpool == nullptr || alloc_cmd == nullptr ||
        begin_cmd == nullptr || end_cmd == nullptr || cmd_bind_pipe == nullptr || cmd_bind_ds == nullptr ||
        cmd_dispatch == nullptr || queue_submit == nullptr || queue_wait == nullptr) {
        res.reason = "required Vulkan compute entrypoints missing";
        return res;
    }

    VkQueue queue = nullptr;
    get_queue(device, compute_qf, 0u, &queue);
    if (queue == nullptr) {
        res.reason = "vkGetDeviceQueue failed";
        return res;
    }

    VkPhysicalDeviceMemoryPropertiesRaw mem_props{};
    get_mem_props(target, &mem_props);
    auto find_host_mem_type = [&](std::uint32_t type_bits) -> std::uint32_t {
        const VkFlags wanted = kVkMemoryHostVisibleBit | kVkMemoryHostCoherentBit;
        for (std::uint32_t i = 0; i < mem_props.memoryTypeCount && i < 32u; ++i) {
            if ((type_bits & (1u << i)) != 0u && (mem_props.memoryTypes[i].propertyFlags & wanted) == wanted) {
                return i;
            }
        }
        return UINT32_MAX;
    };

    const std::size_t elems_a = static_cast<std::size_t>(tiles) * 256u;
    const std::size_t elems_b = 256u;
    const std::size_t elems_d = static_cast<std::size_t>(tiles) * 256u;
    const VkDeviceSize sizes[3] = {
        static_cast<VkDeviceSize>(elems_a * sizeof(std::uint16_t)),
        static_cast<VkDeviceSize>(elems_b * sizeof(std::uint16_t)),
        static_cast<VkDeviceSize>(elems_d * sizeof(float)),
    };

    for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfoRaw bci{};
        bci.sType = kVkStypeBufferCreateInfo;
        bci.size = sizes[i];
        bci.usage = kVkBufferUsageStorageBufferBit;
        bci.sharingMode = 0u;  // VK_SHARING_MODE_EXCLUSIVE
        if (create_buf(device, &bci, nullptr, &r.bufs[i]) != kVkSuccess) {
            res.reason = "vkCreateBuffer failed";
            return res;
        }
        VkMemoryRequirementsRaw req{};
        get_buf_req(device, r.bufs[i], &req);
        const std::uint32_t mt = find_host_mem_type(req.memoryTypeBits);
        if (mt == UINT32_MAX) {
            res.reason = "no host-visible coherent Vulkan memory type";
            return res;
        }
        VkMemoryAllocateInfoRaw mai{};
        mai.sType = kVkStypeMemoryAllocateInfo;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mt;
        if (alloc_mem(device, &mai, nullptr, &r.mems[i]) != kVkSuccess ||
            bind_buf(device, r.bufs[i], r.mems[i], 0u) != kVkSuccess) {
            res.reason = "Vulkan buffer allocation/bind failed";
            return res;
        }
    }

    // Fill BufA and BufB with FP16 1.0 (0x3C00) and BufD with 0.0f.
    constexpr std::uint16_t kFp16One = 0x3C00u;
    for (int i = 0; i < 2; ++i) {
        void* mapped = nullptr;
        if (map_mem(device, r.mems[i], 0u, sizes[i], 0u, &mapped) != kVkSuccess || mapped == nullptr) {
            res.reason = "vkMapMemory failed on input buffer";
            return res;
        }
        auto* dst = static_cast<std::uint16_t*>(mapped);
        const std::size_t count = (i == 0) ? elems_a : elems_b;
        for (std::size_t j = 0; j < count; ++j) dst[j] = kFp16One;
        unmap_mem(device, r.mems[i]);
    }
    {
        void* mapped = nullptr;
        if (map_mem(device, r.mems[2], 0u, sizes[2], 0u, &mapped) != kVkSuccess || mapped == nullptr) {
            res.reason = "vkMapMemory failed on output buffer";
            return res;
        }
        std::memset(mapped, 0, static_cast<std::size_t>(sizes[2]));
        unmap_mem(device, r.mems[2]);
    }

    VkShaderModuleCreateInfoRaw smci{};
    smci.sType = kVkStypeShaderModuleCreateInfo;
    smci.codeSize = kCoopmatCanarySpv.size() * sizeof(std::uint32_t);
    smci.pCode = kCoopmatCanarySpv.data();
    if (create_shader(device, &smci, nullptr, &r.shader) != kVkSuccess) {
        res.reason = "vkCreateShaderModule failed";
        return res;
    }

    VkDescriptorSetLayoutBindingRaw bindings[3]{};
    for (std::uint32_t i = 0; i < 3u; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = kVkDescriptorTypeStorageBuffer;
        bindings[i].descriptorCount = 1u;
        bindings[i].stageFlags = kVkShaderStageComputeBit;
    }
    VkDescriptorSetLayoutCreateInfoRaw dslci{};
    dslci.sType = kVkStypeDescriptorSetLayoutCreateInfo;
    dslci.bindingCount = 3u;
    dslci.pBindings = bindings;
    if (create_dsl(device, &dslci, nullptr, &r.desc_layout) != kVkSuccess) {
        res.reason = "vkCreateDescriptorSetLayout failed";
        return res;
    }

    VkPipelineLayoutCreateInfoRaw plci{};
    plci.sType = kVkStypePipelineLayoutCreateInfo;
    plci.setLayoutCount = 1u;
    plci.pSetLayouts = &r.desc_layout;
    if (create_pl(device, &plci, nullptr, &r.pipe_layout) != kVkSuccess) {
        res.reason = "vkCreatePipelineLayout failed";
        return res;
    }

    VkComputePipelineCreateInfoRaw cpci{};
    cpci.sType = kVkStypeComputePipelineCreateInfo;
    cpci.stage.sType = kVkStypePipelineShaderStageCreateInfo;
    cpci.stage.stage = kVkShaderStageComputeBit;
    cpci.stage.module = r.shader;
    cpci.stage.pName = "main";
    cpci.layout = r.pipe_layout;
    cpci.basePipelineIndex = -1;
    if (create_cp(device, 0u, 1u, &cpci, nullptr, &r.pipeline) != kVkSuccess) {
        res.reason = "vkCreateComputePipelines failed (coopmat shader compilation rejected)";
        return res;
    }

    VkDescriptorPoolSizeRaw pool_size{kVkDescriptorTypeStorageBuffer, 3u};
    VkDescriptorPoolCreateInfoRaw dpci{};
    dpci.sType = kVkStypeDescriptorPoolCreateInfo;
    dpci.maxSets = 1u;
    dpci.poolSizeCount = 1u;
    dpci.pPoolSizes = &pool_size;
    if (create_dp(device, &dpci, nullptr, &r.desc_pool) != kVkSuccess) {
        res.reason = "vkCreateDescriptorPool failed";
        return res;
    }

    VkDescriptorSet desc_set = 0;
    VkDescriptorSetAllocateInfoRaw dsai{};
    dsai.sType = kVkStypeDescriptorSetAllocateInfo;
    dsai.descriptorPool = r.desc_pool;
    dsai.descriptorSetCount = 1u;
    dsai.pSetLayouts = &r.desc_layout;
    if (alloc_ds(device, &dsai, &desc_set) != kVkSuccess) {
        res.reason = "vkAllocateDescriptorSets failed";
        return res;
    }

    VkDescriptorBufferInfoRaw dbi[3]{};
    VkWriteDescriptorSetRaw writes[3]{};
    for (std::uint32_t i = 0; i < 3u; ++i) {
        dbi[i] = VkDescriptorBufferInfoRaw{r.bufs[i], 0u, kVkWholeSize};
        writes[i].sType = kVkStypeWriteDescriptorSet;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = kVkDescriptorTypeStorageBuffer;
        writes[i].pBufferInfo = &dbi[i];
    }
    update_ds(device, 3u, writes, 0u, nullptr);

    VkCommandPoolCreateInfoRaw cmdpci{};
    cmdpci.sType = kVkStypeCommandPoolCreateInfo;
    cmdpci.queueFamilyIndex = compute_qf;
    if (create_cmdpool(device, &cmdpci, nullptr, &r.cmd_pool) != kVkSuccess) {
        res.reason = "vkCreateCommandPool failed";
        return res;
    }

    VkCommandBuffer cmd_buf = nullptr;
    VkCommandBufferAllocateInfoRaw cbai{};
    cbai.sType = kVkStypeCommandBufferAllocateInfo;
    cbai.commandPool = r.cmd_pool;
    cbai.level = 0u;  // VK_COMMAND_BUFFER_LEVEL_PRIMARY
    cbai.commandBufferCount = 1u;
    if (alloc_cmd(device, &cbai, &cmd_buf) != kVkSuccess || cmd_buf == nullptr) {
        res.reason = "vkAllocateCommandBuffers failed";
        return res;
    }

    VkCommandBufferBeginInfoRaw cbbi{};
    cbbi.sType = kVkStypeCommandBufferBeginInfo;
    cbbi.flags = kVkCommandBufferUsageOneTimeSubmitBit;
    if (begin_cmd(cmd_buf, &cbbi) != kVkSuccess) {
        res.reason = "vkBeginCommandBuffer failed";
        return res;
    }
    cmd_bind_pipe(cmd_buf, kVkPipelineBindPointCompute, r.pipeline);
    cmd_bind_ds(cmd_buf, kVkPipelineBindPointCompute, r.pipe_layout, 0u, 1u, &desc_set, 0u, nullptr);
    cmd_dispatch(cmd_buf, tiles, 1u, 1u);
    if (end_cmd(cmd_buf) != kVkSuccess) {
        res.reason = "vkEndCommandBuffer failed";
        return res;
    }

    VkSubmitInfoRaw si{};
    si.sType = kVkStypeSubmitInfo;
    si.commandBufferCount = 1u;
    si.pCommandBuffers = &cmd_buf;
    if (queue_submit(queue, 1u, &si, 0u) != kVkSuccess) {
        res.reason = "vkQueueSubmit failed";
        return res;
    }
    if (queue_wait(queue) != kVkSuccess) {
        res.reason = "vkQueueWaitIdle failed (device lost or compute fault)";
        return res;
    }

    void* out_mapped = nullptr;
    if (map_mem(device, r.mems[2], 0u, sizes[2], 0u, &out_mapped) != kVkSuccess || out_mapped == nullptr) {
        res.reason = "vkMapMemory failed on output verification";
        return res;
    }
    const auto* out_floats = static_cast<const float*>(out_mapped);
    bool mismatch = false;
    std::size_t bad_idx = 0;
    float bad_val = 0.0f;
    for (std::size_t i = 0; i < elems_d; ++i) {
        if (!std::isfinite(out_floats[i]) || std::fabs(out_floats[i] - 16.0f) > 1e-3f) {
            mismatch = true;
            bad_idx = i;
            bad_val = out_floats[i];
            break;
        }
    }
    unmap_mem(device, r.mems[2]);

    if (mismatch) {
        std::ostringstream oss;
        oss << "coopmat output mismatch at index " << bad_idx << " (got " << bad_val << ", expected 16.0)";
        res.reason = oss.str();
        return res;
    }

    res.ok = true;
    res.reason = "ok";
    return res;
}

CoopmatCanaryDecision evaluate_amd_coopmat_canary(const std::vector<GpuDeviceInfo>& gpus,
                                                  const CoopmatCanaryOptions& options) {
    CoopmatCanaryDecision decision;

    const std::string disable_env =
        options.existing_disable_coopmat_env ? *options.existing_disable_coopmat_env
                                             : read_env_var("GGML_VK_DISABLE_COOPMAT");
    if (is_env_truthy(disable_env)) {
        decision.status = CoopmatCanaryStatus::skipped_env_disabled;
        decision.disable_coopmat = true;
        decision.reason = "GGML_VK_DISABLE_COOPMAT already set by operator";
        return decision;
    }

    std::vector<const GpuDeviceInfo*> eligible;
    for (const auto& gpu : gpus) {
        if (needs_amd_windows_coopmat_canary(gpu, options.is_windows)) {
            eligible.push_back(&gpu);
        }
    }
    if (eligible.empty()) {
        decision.status = CoopmatCanaryStatus::not_applicable;
        decision.disable_coopmat = false;
        decision.reason = "no eligible AMD Windows GPU";
        return decision;
    }

    const std::uint32_t ubatch = std::clamp(options.ubatch == 0u ? 64u : options.ubatch, 16u, 4096u);
    std::string cache_path = options.cache_path;
    if (cache_path.empty() && options.use_default_cache_path) {
        cache_path = default_coopmat_canary_cache_path();
    }

    bool any_failed = false;
    bool any_error = false;
    std::string first_failure_reason;

    for (const GpuDeviceInfo* gpu : eligible) {
        CoopmatCanaryKey key;
        key.vendor_id = gpu->vendor_id != 0u ? gpu->vendor_id : kAmdVendorId;
        key.device_id = gpu->device_id;
        key.driver_build = sanitize_single_line(gpu->driver_build, kMaxDriverBuildLen);
        key.ubatch = ubatch;
        const std::string key_str = canonical_coopmat_cache_key(key);

        std::optional<CoopmatCanaryRecord> cached;
        if (!cache_path.empty()) {
            std::lock_guard<std::mutex> lock(cache_io_mutex());
            const auto entries = load_coopmat_canary_cache(cache_path);
            for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
                if (canonical_coopmat_cache_key(it->key) == key_str) {
                    cached = *it;
                    break;
                }
            }
        }

        CoopmatCanaryRecord rec;
        if (cached.has_value()) {
            rec = *cached;
            rec.gpu_id = gpu->id;
            rec.gpu_name = gpu->name;
            rec.is_integrated_uma = gpu->is_integrated_uma;
            rec.from_cache = true;
        } else {
            CoopmatCanaryChildOutcome outcome;
            if (options.child_runner) {
                outcome = options.child_runner(key, options.timeout);
            } else {
                const std::string exe = find_coopmat_canary_executable(options.executable_path);
                if (exe.empty()) {
                    outcome.spawn_error = true;
                    outcome.reason = "sonder-infer executable not found for coopmat canary";
                } else {
                    outcome = spawn_canary_child(exe, key, options.timeout);
                }
            }

            rec.key = key;
            rec.gpu_id = gpu->id;
            rec.gpu_name = gpu->name;
            rec.is_integrated_uma = gpu->is_integrated_uma;
            rec.from_cache = false;
            rec.elapsed_ms = outcome.elapsed_ms;
            rec.exit_code = outcome.exit_code;

            if (outcome.spawn_error) {
                rec.status = CoopmatCanaryStatus::error;
                rec.disable_coopmat = options.fail_closed_on_runner_error;
                rec.reason = outcome.reason.empty() ? "canary spawn error" : outcome.reason;
            } else if (outcome.timed_out) {
                rec.status = CoopmatCanaryStatus::failed;
                rec.disable_coopmat = true;
                rec.reason = outcome.reason.find("timeout") != std::string::npos
                                 ? outcome.reason
                                 : ("canary timeout: " + outcome.reason);
            } else if (outcome.completed && outcome.exit_code == 0u) {
                rec.status = CoopmatCanaryStatus::passed;
                rec.disable_coopmat = false;
                rec.reason = outcome.reason.empty() ? "ok" : outcome.reason;
            } else {
                rec.status = CoopmatCanaryStatus::failed;
                rec.disable_coopmat = true;
                rec.reason = outcome.reason.empty()
                                 ? ("canary child failed with exit_code=0x" + format_hex_u32(outcome.exit_code, 8))
                                 : outcome.reason;
            }

            // Persist only definitive pass/fail outcomes; never poison cache with transient spawn errors.
            if (!cache_path.empty() &&
                (rec.status == CoopmatCanaryStatus::passed || rec.status == CoopmatCanaryStatus::failed)) {
                std::lock_guard<std::mutex> lock(cache_io_mutex());
                auto entries = load_coopmat_canary_cache(cache_path);
                entries.erase(std::remove_if(entries.begin(), entries.end(),
                                             [&](const CoopmatCanaryRecord& existing) {
                                                 return canonical_coopmat_cache_key(existing.key) == key_str;
                                             }),
                              entries.end());
                entries.push_back(rec);
                (void)save_coopmat_canary_cache(cache_path, entries);
            }
        }

        if (rec.status == CoopmatCanaryStatus::failed) {
            any_failed = true;
            if (first_failure_reason.empty()) first_failure_reason = rec.reason;
        } else if (rec.status == CoopmatCanaryStatus::error) {
            any_error = true;
            if (first_failure_reason.empty()) first_failure_reason = rec.reason;
        }
        decision.records.push_back(std::move(rec));
    }

    if (any_failed) {
        decision.status = CoopmatCanaryStatus::failed;
        decision.disable_coopmat = true;
        decision.reason = first_failure_reason;
    } else if (any_error) {
        decision.status = CoopmatCanaryStatus::error;
        decision.disable_coopmat = options.fail_closed_on_runner_error;
        decision.reason = first_failure_reason;
    } else {
        decision.status = CoopmatCanaryStatus::passed;
        decision.disable_coopmat = false;
        decision.reason = "all eligible AMD Windows GPUs passed cooperative-matrix canary";
    }

    if (decision.disable_coopmat && options.apply_env_fallback) {
        decision.env_applied = set_env_var("GGML_VK_DISABLE_COOPMAT", "1");
    }
    return decision;
}

void record_coopmat_canary_decision(const CoopmatCanaryDecision& decision) {
    std::lock_guard<std::mutex> lock(decision_mutex());
    stored_decision() = decision;
}

std::optional<CoopmatCanaryDecision> last_coopmat_canary_decision() {
    std::lock_guard<std::mutex> lock(decision_mutex());
    return stored_decision();
}

void clear_coopmat_canary_decision() noexcept {
    std::lock_guard<std::mutex> lock(decision_mutex());
    stored_decision().reset();
}

}  // namespace sonder::inference
