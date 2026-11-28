#include "sonder/inference/device.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <dxgi1_4.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#else
#  if defined(__linux__)
#    include <dirent.h>
#    include <dlfcn.h>
#  endif
#  include <cstdlib>
#  include <fstream>
#  include <iterator>
#  include <sstream>
#  include <unistd.h>
#endif

namespace sonder::inference {

const char* to_string(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::cpu: return "cpu";
        case DeviceKind::gpu: return "gpu";
        case DeviceKind::npu: return "npu";
        case DeviceKind::remote: return "remote";
    }
    return "unknown";
}

const char* to_string(GpuVendor vendor) noexcept {
    switch (vendor) {
        case GpuVendor::unknown: return "unknown";
        case GpuVendor::nvidia: return "nvidia";
        case GpuVendor::amd: return "amd";
        case GpuVendor::intel: return "intel";
        case GpuVendor::apple: return "apple";
    }
    return "unknown";
}

GpuVendor gpu_vendor_from_pci_id(std::uint32_t vendor_id) noexcept {
    switch (vendor_id) {
        case 0x10DEu: return GpuVendor::nvidia;
        case 0x1002u: return GpuVendor::amd;
        case 0x8086u: return GpuVendor::intel;
        case 0x106Bu: return GpuVendor::apple;
        default: return GpuVendor::unknown;
    }
}

namespace {

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string_view trim_view(std::string_view sv) noexcept {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = sv.find_last_not_of(" \t\r\n");
    return sv.substr(first, last - first + 1);
}

char ascii_tolower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (ascii_tolower(a[i]) != ascii_tolower(b[i])) {
            return false;
        }
    }
    return true;
}

bool istarts_with(std::string_view sv, std::string_view prefix) noexcept {
    if (sv.size() < prefix.size()) {
        return false;
    }
    return iequals(sv.substr(0, prefix.size()), prefix);
}

std::optional<std::size_t> parse_index(std::string_view text) noexcept {
    if (text.empty() || text.size() > 20) {
        return std::nullopt;
    }
    std::size_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
        const auto digit = static_cast<std::size_t>(c - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint32_t> parse_hex_u32(std::string_view text) noexcept {
    text = trim_view(text);
    if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2);
    }
    if (text.empty() || text.size() > 8) {
        return std::nullopt;
    }
    std::uint32_t value = 0;
    for (const char c : text) {
        std::uint32_t nibble = 0;
        if (c >= '0' && c <= '9') {
            nibble = static_cast<std::uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            nibble = static_cast<std::uint32_t>(10 + (c - 'a'));
        } else if (c >= 'A' && c <= 'F') {
            nibble = static_cast<std::uint32_t>(10 + (c - 'A'));
        } else {
            return std::nullopt;
        }
        value = (value << 4) | nibble;
    }
    return value;
}

std::optional<std::uint64_t> parse_sysfs_u64(std::string_view text) noexcept {
    return parse_strict_u64(trim_view(text));
}

std::optional<std::uint32_t> parse_card_index(std::string_view name) noexcept {
    if (name.size() <= 4 || name.substr(0, 4) != "card") {
        return std::nullopt;
    }
    const auto idx = parse_index(name.substr(4));
    if (!idx || *idx > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(*idx);
}

std::string hex_u32_string(std::uint32_t value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out = "0x";
    bool started = false;
    for (int shift = 28; shift >= 0; shift -= 4) {
        const auto digit = (value >> shift) & 0xFu;
        if (digit != 0 || started || shift <= 12) {
            started = true;
            out.push_back(kHex[digit]);
        }
    }
    return out;
}

bool is_integrated_uma_memory(GpuVendor vendor, std::uint64_t dedicated_bytes, std::uint64_t shared_bytes) noexcept {
    if (dedicated_bytes == 0 && shared_bytes > 0) {
        return true;
    }
    if (vendor == GpuVendor::apple && shared_bytes > 0) {
        return true;
    }
    if ((vendor == GpuVendor::amd || vendor == GpuVendor::intel) && dedicated_bytes <= 2 * kGiB &&
        shared_bytes >= 4 * dedicated_bytes && shared_bytes > 0) {
        return true;
    }
    return false;
}

template <typename Pred>
std::optional<std::size_t> select_best_gpu(const std::vector<GpuDeviceInfo>& gpus, const Pred& pred,
                                           bool prefer_free, bool require_known_vram = true) noexcept {
    std::optional<std::size_t> best_discrete;
    std::uint64_t best_discrete_mib = 0;
    std::optional<std::size_t> best_uma;
    std::uint64_t best_uma_mib = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i])) {
            continue;
        }
        const auto mib_opt = gpu_effective_vram_mib(gpus[i], prefer_free);
        if (require_known_vram && !mib_opt) {
            continue;
        }
        const std::uint64_t mib = mib_opt.value_or(0);
        if (!gpus[i].is_integrated_uma) {
            if (!best_discrete || mib > best_discrete_mib) {
                best_discrete = i;
                best_discrete_mib = mib;
            }
        } else {
            if (!best_uma || mib > best_uma_mib) {
                best_uma = i;
                best_uma_mib = mib;
            }
        }
    }
    return best_discrete ? best_discrete : best_uma;
}

template <typename Pred>
std::optional<std::size_t> match_vendor_hint(const std::vector<GpuDeviceInfo>& gpus, std::string_view suffix,
                                             const Pred& pred, bool prefer_free,
                                             bool require_known_vram = true) noexcept {
    if (suffix.empty()) {
        return select_best_gpu(gpus, pred, prefer_free, require_known_vram);
    }
    if (suffix.front() == ':') {
        suffix.remove_prefix(1);
    }
    const auto target_idx = parse_index(suffix);
    if (!target_idx) {
        return std::nullopt;
    }
    std::size_t seen = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i])) {
            continue;
        }
        if (seen == *target_idx) {
            return i;
        }
        ++seen;
    }
    return std::nullopt;
}

std::optional<std::size_t> match_single_gpu_hint_impl(const std::vector<GpuDeviceInfo>& gpus, std::string_view token,
                                                      bool prefer_free, bool require_known_vram) noexcept {
    token = trim_view(token);
    if (token.empty()) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (iequals(gpus[i].id, token)) {
            return i;
        }
    }
    if (istarts_with(token, "gpu:")) {
        const auto idx = parse_index(token.substr(4));
        if (idx && *idx < gpus.size()) {
            return *idx;
        }
        return std::nullopt;
    }
    if (const auto idx = parse_index(token); idx && *idx < gpus.size()) {
        return *idx;
    }
    const auto is_nvidia = [](const GpuDeviceInfo& g) noexcept {
        return g.vendor == GpuVendor::nvidia || iequals(g.backend_hint, "cuda");
    };
    const auto is_amd = [](const GpuDeviceInfo& g) noexcept {
        return g.vendor == GpuVendor::amd || iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    };
    const bool any_explicit_rocm = std::any_of(gpus.begin(), gpus.end(), [](const GpuDeviceInfo& g) noexcept {
        return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    });
    const auto is_rocm_target = [any_explicit_rocm, &is_amd](const GpuDeviceInfo& g) noexcept {
        if (any_explicit_rocm) {
            return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
        }
        return is_amd(g);
    };
    const auto is_intel = [](const GpuDeviceInfo& g) noexcept { return g.vendor == GpuVendor::intel; };
    const auto is_apple = [](const GpuDeviceInfo& g) noexcept {
        return g.vendor == GpuVendor::apple || iequals(g.backend_hint, "metal");
    };
    const auto is_uma = [](const GpuDeviceInfo& g) noexcept { return g.is_integrated_uma; };
    const auto is_any = [](const GpuDeviceInfo&) noexcept { return true; };

    if (iequals(token, "uma") || iequals(token, "igpu") || iequals(token, "apu")) {
        return select_best_gpu(gpus, is_uma, prefer_free, require_known_vram);
    }

    if (istarts_with(token, "cuda")) {
        return match_vendor_hint(gpus, token.substr(4), is_nvidia, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "nvidia")) {
        return match_vendor_hint(gpus, token.substr(6), is_nvidia, prefer_free, require_known_vram);
    }
    const auto match_rocm_hint = [&](std::string_view suffix) noexcept -> std::optional<std::size_t> {
        std::string_view trimmed_suffix = trim_view(suffix);
        if (!trimmed_suffix.empty() && trimmed_suffix.front() == ':') {
            trimmed_suffix.remove_prefix(1);
        }
        if (const auto target_idx = parse_index(trimmed_suffix)) {
            for (std::size_t i = 0; i < gpus.size(); ++i) {
                if (gpus[i].rocm_runtime_ordinal && *gpus[i].rocm_runtime_ordinal == *target_idx) {
                    return i;
                }
            }
        }
        return match_vendor_hint(gpus, suffix, is_rocm_target, prefer_free, require_known_vram);
    };
    if (istarts_with(token, "rocm")) {
        return match_rocm_hint(token.substr(4));
    }
    if (istarts_with(token, "hip")) {
        return match_rocm_hint(token.substr(3));
    }
    if (istarts_with(token, "amd")) {
        return match_vendor_hint(gpus, token.substr(3), is_amd, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "vulkan")) {
        return match_vendor_hint(gpus, token.substr(6), is_any, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "vk")) {
        return match_vendor_hint(gpus, token.substr(2), is_any, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "intel")) {
        return match_vendor_hint(gpus, token.substr(5), is_intel, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "sycl")) {
        return match_vendor_hint(gpus, token.substr(4), is_intel, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "metal")) {
        return match_vendor_hint(gpus, token.substr(5), is_apple, prefer_free, require_known_vram);
    }
    if (istarts_with(token, "apple")) {
        return match_vendor_hint(gpus, token.substr(5), is_apple, prefer_free, require_known_vram);
    }
    return std::nullopt;
}

std::optional<std::size_t> map_rocm_visible_ordinal(std::size_t ordinal, std::string_view raw,
                                                    std::optional<std::string_view> device_uuid) noexcept {
    const std::string_view val = trim_view(raw);
    if (val.empty() || iequals(val, "-1") || iequals(val, "none") || iequals(val, "nodevfiles")) {
        return std::nullopt;
    }
    const std::string_view norm_uuid = device_uuid ? trim_view(*device_uuid) : std::string_view{};
    std::size_t mapped_pos = 0;
    std::size_t pos = 0;
    while (pos <= val.size()) {
        const std::size_t comma = val.find(',', pos);
        const std::string_view tok =
            trim_view(val.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? val.size() + 1 : comma + 1;
        if (tok.empty()) {
            continue;
        }
        if (tok.front() == '-') {
            break;  // HIP/ROCr terminates visible-device list parsing on negative entries
        }
        if (const auto idx = parse_index(tok)) {
            if (*idx == ordinal) {
                return mapped_pos;
            }
            ++mapped_pos;
        } else if (istarts_with(tok, "gpu-")) {
            if (!norm_uuid.empty() && iequals(tok, norm_uuid)) {
                return mapped_pos;
            }
            ++mapped_pos;
        } else {
            break;
        }
    }
    return std::nullopt;
}

std::string normalize_pci_bus_id(std::string_view raw) {
    const std::string_view sv = trim_view(raw);
    if (sv.empty()) {
        return {};
    }
    std::string lower;
    lower.reserve(sv.size());
    for (const char c : sv) {
        lower.push_back(ascii_tolower(c));
    }
    const std::size_t first_colon = lower.find(':');
    if (first_colon != std::string::npos && first_colon > 4u) {
        // NVML formats the PCI domain with 8 hex digits ("00000000:01:00.0")
        // whereas Vulkan and Linux sysfs use 4 ("0000:01:00.0"). Trim extra
        // leading zeros on the domain down to 4 digits for canonical matching.
        std::size_t trim_pos = 0;
        while (first_colon - trim_pos > 4u && lower[trim_pos] == '0') {
            ++trim_pos;
        }
        lower.erase(0, trim_pos);
    }
    return lower;
}

#if defined(_WIN32) || defined(__linux__)
#  if defined(_WIN32)
#    define SONDER_VKAPI_CALL __stdcall
#  else
#    define SONDER_VKAPI_CALL
#  endif

struct VkProbeAdapterLuid {
    bool valid = false;
    std::uint8_t bytes[8] = {};
};

struct VkPhysicalDeviceProbe {
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::string name;
    bool is_integrated_uma = false;
    VkProbeAdapterLuid luid;
    std::string pci_bus_id;
    std::string driver_name;
    std::string driver_info;
    std::optional<std::uint64_t> shared_memory_bytes;
    std::optional<std::uint32_t> warp_size;
    std::string matrix_cores;
    std::optional<bool> ggml_coopmat_eligible;
    std::vector<VulkanMemoryHeapInfo> heaps;
    std::optional<bool> rebar;
};

std::string format_pci_bdf(std::uint32_t domain, std::uint32_t bus, std::uint32_t device, std::uint32_t func) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(16);
    out.push_back(kHex[(domain >> 12) & 0xFu]);
    out.push_back(kHex[(domain >> 8) & 0xFu]);
    out.push_back(kHex[(domain >> 4) & 0xFu]);
    out.push_back(kHex[domain & 0xFu]);
    out.push_back(':');
    out.push_back(kHex[(bus >> 4) & 0xFu]);
    out.push_back(kHex[bus & 0xFu]);
    out.push_back(':');
    out.push_back(kHex[(device >> 4) & 0xFu]);
    out.push_back(kHex[device & 0xFu]);
    out.push_back('.');
    out.push_back(kHex[func & 0xFu]);
    return out;
}

// Optional, zero-SDK-dependency Vulkan 1.1/1.2 physical-device probe.
// Safety invariant (NAT-16 / #22750): never calls vkCreateDevice and never
// calls vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR in-process.
std::vector<VkPhysicalDeviceProbe> probe_vulkan_physical_devices() {
    std::vector<VkPhysicalDeviceProbe> out;
#  if defined(_WIN32)
    HMODULE lib = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) {
        return out;
    }
    struct ScopedVkLib {
        HMODULE h = nullptr;
        ~ScopedVkLib() {
            if (h) FreeLibrary(h);
        }
    } lib_guard{lib};
    auto load_sym = [&](const char* name, auto& fn) noexcept -> bool {
        const FARPROC raw = GetProcAddress(lib, name);
        if (!raw) return false;
        std::memcpy(&fn, &raw, sizeof(fn));
        return true;
    };
#  else
    void* lib = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (!lib) {
        return out;
    }
    struct ScopedVkLib {
        void* h = nullptr;
        ~ScopedVkLib() {
            if (h) dlclose(h);
        }
    } lib_guard{lib};
    auto load_sym = [&](const char* name, auto& fn) noexcept -> bool {
        void* raw = dlsym(lib, name);
        if (!raw) return false;
        std::memcpy(&fn, &raw, sizeof(fn));
        return true;
    };
#  endif

    using VkResult = std::int32_t;
    using VkBool32 = std::uint32_t;
    using VkFlags = std::uint32_t;
    using VkDeviceSize = std::uint64_t;
    using VkStructureType = std::int32_t;
    struct VkInstance_T;
    struct VkPhysicalDevice_T;
    using VkInstance = VkInstance_T*;
    using VkPhysicalDevice = VkPhysicalDevice_T*;
    using PFN_vkVoidFunction = void(SONDER_VKAPI_CALL*)();

    constexpr VkResult kVkSuccess = 0;
    constexpr VkStructureType kStAppInfo = 0;
    constexpr VkStructureType kStInstCreate = 1;
    constexpr VkStructureType kStProps2 = 1000059001;
    constexpr VkStructureType kStMemProps2 = 1000059006;
    constexpr VkStructureType kStIdProps = 1000071004;
    constexpr VkStructureType kStSubgroupProps = 1000094000;
    constexpr VkStructureType kStAmdShaderCoreProps = 1000185000;
    constexpr VkStructureType kStDriverProps = 1000196000;
    constexpr VkStructureType kStPciBusInfo = 1000212000;
    constexpr VkStructureType kStSubgroupSizeCtrlProps = 1000225000;
    constexpr VkStructureType kStMemBudget = 1000237000;
    constexpr VkStructureType kStIntDotProps = 1000280001;
    constexpr std::int32_t kVkDriverAmdProprietary = 1;
    constexpr std::int32_t kVkDriverAmdOpenSource = 2;
    constexpr std::uint32_t kVkApi11 = (1u << 22) | (1u << 12);
    constexpr std::uint32_t kVkApi12 = (1u << 22) | (2u << 12);
    constexpr std::uint32_t kMaxHeaps = 16u;
    constexpr std::uint32_t kMaxTypes = 32u;
    constexpr VkFlags kMemPropDeviceLocal = 0x00000001u;
    constexpr VkFlags kMemPropHostVisible = 0x00000002u;
    constexpr VkFlags kMemPropHostCoherent = 0x00000004u;
    constexpr VkFlags kMemHeapDeviceLocal = 0x00000001u;
    constexpr VkFlags kMemHeapMultiInstance = 0x00000002u;

    struct VkApplicationInfoRaw {
        VkStructureType sType;
        const void* pNext;
        const char* pApplicationName;
        std::uint32_t applicationVersion;
        const char* pEngineName;
        std::uint32_t engineVersion;
        std::uint32_t apiVersion;
    };
    struct VkInstanceCreateInfoRaw {
        VkStructureType sType;
        const void* pNext;
        VkFlags flags;
        const VkApplicationInfoRaw* pApplicationInfo;
        std::uint32_t enabledLayerCount;
        const char* const* ppEnabledLayerNames;
        std::uint32_t enabledExtensionCount;
        const char* const* ppEnabledExtensionNames;
    };
    struct VkExtensionPropertiesRaw {
        char extensionName[256];
        std::uint32_t specVersion;
    };
    struct VkPhysicalDevicePropertiesRaw {
        std::uint32_t apiVersion;
        std::uint32_t driverVersion;
        std::uint32_t vendorID;
        std::uint32_t deviceID;
        std::int32_t deviceType;
        char deviceName[256];
        std::uint8_t pipelineCacheUUID[16];
        // Natural 8-byte alignment pad before VkPhysicalDeviceLimits (which
        // contains VkDeviceSize / uint64_t).
        std::uint32_t limits_align_pad;
        // VkPhysicalDeviceLimits (504 bytes on 64-bit) +
        // VkPhysicalDeviceSparseProperties (20 bytes) + 4 bytes tail padding = 528 bytes (66 qwords).
        // maxComputeSharedMemorySize sits at byte offset 216 inside limits (512 in struct).
        std::uint64_t limits_and_sparse[66];
    };
    struct VkPhysicalDeviceProperties2Raw {
        VkStructureType sType;
        void* pNext;
        VkPhysicalDevicePropertiesRaw properties;
    };
    struct VkConformanceVersionRaw {
        std::uint8_t major;
        std::uint8_t minor;
        std::uint8_t subminor;
        std::uint8_t patch;
    };
    struct VkPhysicalDeviceDriverPropertiesRaw {
        VkStructureType sType;
        void* pNext;
        std::int32_t driverID;
        char driverName[256];
        char driverInfo[256];
        VkConformanceVersionRaw conformanceVersion;
    };
    struct VkPhysicalDeviceIDPropertiesRaw {
        VkStructureType sType;
        void* pNext;
        std::uint8_t deviceUUID[16];
        std::uint8_t driverUUID[16];
        std::uint8_t deviceLUID[8];
        std::uint32_t deviceNodeMask;
        VkBool32 deviceLUIDValid;
    };
    struct VkPhysicalDeviceSubgroupPropertiesRaw {
        VkStructureType sType;
        void* pNext;
        std::uint32_t subgroupSize;
        VkFlags supportedStages;
        VkFlags supportedOperations;
        VkBool32 quadOperationsInAllStages;
    };
    struct VkPhysicalDevicePCIBusInfoPropertiesEXTRaw {
        VkStructureType sType;
        void* pNext;
        std::uint32_t pciDomain;
        std::uint32_t pciBus;
        std::uint32_t pciDevice;
        std::uint32_t pciFunction;
    };
    struct VkPhysicalDeviceShaderCorePropertiesAMDRaw {
        VkStructureType sType;
        void* pNext;
        std::uint32_t shaderEngineCount;
        std::uint32_t shaderArraysPerEngineCount;
        std::uint32_t computeUnitsPerShaderArray;
        std::uint32_t simdPerComputeUnit;
        std::uint32_t wavefrontsPerSimd;
        std::uint32_t trailing[9];
    };
    struct VkPhysicalDeviceShaderIntegerDotProductPropertiesRaw {
        VkStructureType sType;
        void* pNext;
        VkBool32 flags[30];  // [5] == integerDotProduct4x8BitPackedMixedSignednessAccelerated
    };
    struct VkPhysicalDeviceSubgroupSizeControlPropertiesRaw {
        VkStructureType sType;
        void* pNext;
        std::uint32_t minSubgroupSize;
        std::uint32_t maxSubgroupSize;
        std::uint32_t maxComputeWorkgroupSubgroups;
        VkFlags requiredSubgroupSizeStages;
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
        VkMemoryTypeRaw memoryTypes[kMaxTypes];
        std::uint32_t memoryHeapCount;
        VkMemoryHeapRaw memoryHeaps[kMaxHeaps];
    };
    struct VkPhysicalDeviceMemoryProperties2Raw {
        VkStructureType sType;
        void* pNext;
        VkPhysicalDeviceMemoryPropertiesRaw memoryProperties;
    };
    struct VkPhysicalDeviceMemoryBudgetPropertiesEXTRaw {
        VkStructureType sType;
        void* pNext;
        VkDeviceSize heapBudget[kMaxHeaps];
        VkDeviceSize heapUsage[kMaxHeaps];
    };

    static_assert(sizeof(VkPhysicalDevicePropertiesRaw) == 824u);
    static_assert(offsetof(VkPhysicalDevicePropertiesRaw, limits_and_sparse) == 296u);

    using GetInstProcFn = PFN_vkVoidFunction(SONDER_VKAPI_CALL*)(VkInstance, const char*);
    using EnumInstVerFn = VkResult(SONDER_VKAPI_CALL*)(std::uint32_t*);
    using CreateInstFn = VkResult(SONDER_VKAPI_CALL*)(const VkInstanceCreateInfoRaw*, const void*, VkInstance*);
    using DestroyInstFn = void(SONDER_VKAPI_CALL*)(VkInstance, const void*);
    using EnumDevsFn = VkResult(SONDER_VKAPI_CALL*)(VkInstance, std::uint32_t*, VkPhysicalDevice*);
    using EnumExtsFn =
        VkResult(SONDER_VKAPI_CALL*)(VkPhysicalDevice, const char*, std::uint32_t*, VkExtensionPropertiesRaw*);
    using GetPropsFn = void(SONDER_VKAPI_CALL*)(VkPhysicalDevice, VkPhysicalDevicePropertiesRaw*);
    using GetProps2Fn = void(SONDER_VKAPI_CALL*)(VkPhysicalDevice, VkPhysicalDeviceProperties2Raw*);
    using GetMemProps2Fn = void(SONDER_VKAPI_CALL*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2Raw*);

    GetInstProcFn get_proc = nullptr;
    if (!load_sym("vkGetInstanceProcAddr", get_proc) || !get_proc) {
        return out;
    }
    auto resolve_inst = [&](VkInstance inst, const char* name, auto& fn) noexcept -> bool {
        const PFN_vkVoidFunction raw = get_proc(inst, name);
        if (!raw) return false;
        std::memcpy(&fn, &raw, sizeof(fn));
        return true;
    };

    EnumInstVerFn enum_ver = nullptr;
    CreateInstFn create_inst = nullptr;
    if (!resolve_inst(nullptr, "vkCreateInstance", create_inst)) {
        return out;
    }
    std::uint32_t inst_ver = kVkApi11;
    if (resolve_inst(nullptr, "vkEnumerateInstanceVersion", enum_ver)) {
        std::uint32_t avail = 0;
        if (enum_ver(&avail) == kVkSuccess) {
            if (avail < kVkApi11) {
                return out;  // Vulkan 1.1+ required for Properties2 / MemoryProperties2
            }
            inst_ver = avail >= kVkApi12 ? kVkApi12 : kVkApi11;
        }
    }

    VkApplicationInfoRaw app{};
    app.sType = kStAppInfo;
    app.pApplicationName = "sonder-inference";
    app.apiVersion = inst_ver;

    VkInstanceCreateInfoRaw ci{};
    ci.sType = kStInstCreate;
    ci.pApplicationInfo = &app;

    VkInstance instance = nullptr;
    if (create_inst(&ci, nullptr, &instance) != kVkSuccess || !instance) {
        return out;
    }

    DestroyInstFn destroy_inst = nullptr;
    (void)resolve_inst(instance, "vkDestroyInstance", destroy_inst);
    struct ScopedVkInst {
        VkInstance inst = nullptr;
        DestroyInstFn destroy = nullptr;
        ~ScopedVkInst() {
            if (inst && destroy) destroy(inst, nullptr);
        }
    } inst_guard{instance, destroy_inst};

    EnumDevsFn enum_devs = nullptr;
    EnumExtsFn enum_exts = nullptr;
    GetPropsFn get_props = nullptr;
    GetProps2Fn get_props2 = nullptr;
    GetMemProps2Fn get_mem_props2 = nullptr;
    if (!resolve_inst(instance, "vkEnumeratePhysicalDevices", enum_devs) ||
        !resolve_inst(instance, "vkEnumerateDeviceExtensionProperties", enum_exts) ||
        !resolve_inst(instance, "vkGetPhysicalDeviceProperties2", get_props2) ||
        !resolve_inst(instance, "vkGetPhysicalDeviceMemoryProperties2", get_mem_props2)) {
        return out;
    }
    (void)resolve_inst(instance, "vkGetPhysicalDeviceProperties", get_props);

    std::uint32_t dev_count = 0;
    if (enum_devs(instance, &dev_count, nullptr) != kVkSuccess || dev_count == 0) {
        return out;
    }
    dev_count = std::min<std::uint32_t>(dev_count, 32u);
    std::vector<VkPhysicalDevice> phys_devs(dev_count, nullptr);
    if (enum_devs(instance, &dev_count, phys_devs.data()) != kVkSuccess) {
        return out;
    }
    phys_devs.resize(dev_count);

    for (VkPhysicalDevice pd : phys_devs) {
        if (!pd) continue;

        bool has_mem_budget = false;
        bool has_pci_bus_info = false;
        bool has_driver_props_ext = false;
        bool has_coopmat = false;
        bool has_coopmat2 = false;
        bool has_amd_shader_core = false;
        bool has_int_dot_product = false;
        bool has_subgroup_size_control = false;
        bool has_shader_float8 = false;

        std::uint32_t ext_count = 0;
        if (enum_exts(pd, nullptr, &ext_count, nullptr) == kVkSuccess && ext_count > 0) {
            ext_count = std::min<std::uint32_t>(ext_count, 1024u);
            std::vector<VkExtensionPropertiesRaw> exts(ext_count);
            if (enum_exts(pd, nullptr, &ext_count, exts.data()) == kVkSuccess) {
                exts.resize(ext_count);
                for (const auto& e : exts) {
                    const std::string_view ename(e.extensionName,
                                                 strnlen(e.extensionName, sizeof(e.extensionName)));
                    if (ename == "VK_EXT_memory_budget") has_mem_budget = true;
                    else if (ename == "VK_EXT_pci_bus_info") has_pci_bus_info = true;
                    else if (ename == "VK_KHR_driver_properties") has_driver_props_ext = true;
                    else if (ename == "VK_KHR_cooperative_matrix") has_coopmat = true;
                    else if (ename == "VK_NV_cooperative_matrix2") has_coopmat2 = true;
                    else if (ename == "VK_AMD_shader_core_properties") has_amd_shader_core = true;
                    else if (ename == "VK_KHR_shader_integer_dot_product") has_int_dot_product = true;
                    else if (ename == "VK_EXT_subgroup_size_control") has_subgroup_size_control = true;
                    else if (ename == "VK_EXT_shader_float8") has_shader_float8 = true;
                }
            }
        }

        std::uint32_t dev_api_ver = 0;
        if (get_props) {
            VkPhysicalDevicePropertiesRaw base_props{};
            get_props(pd, &base_props);
            dev_api_ver = base_props.apiVersion;
        }
        const bool chain_driver_props = has_driver_props_ext || (inst_ver >= kVkApi12 && dev_api_ver >= kVkApi12);
        const bool chain_amd_arch_props = has_amd_shader_core && has_int_dot_product && has_subgroup_size_control;

        VkPhysicalDeviceProperties2Raw props2{};
        props2.sType = kStProps2;
        VkPhysicalDeviceIDPropertiesRaw id_props{};
        id_props.sType = kStIdProps;
        VkPhysicalDeviceSubgroupPropertiesRaw subgroup_props{};
        subgroup_props.sType = kStSubgroupProps;
        VkPhysicalDeviceDriverPropertiesRaw drv_props{};
        drv_props.sType = kStDriverProps;
        VkPhysicalDevicePCIBusInfoPropertiesEXTRaw pci_props{};
        pci_props.sType = kStPciBusInfo;
        VkPhysicalDeviceShaderCorePropertiesAMDRaw amd_core_props{};
        amd_core_props.sType = kStAmdShaderCoreProps;
        VkPhysicalDeviceShaderIntegerDotProductPropertiesRaw int_dot_props{};
        int_dot_props.sType = kStIntDotProps;
        VkPhysicalDeviceSubgroupSizeControlPropertiesRaw subgroup_ctrl_props{};
        subgroup_ctrl_props.sType = kStSubgroupSizeCtrlProps;

        void* pnext = nullptr;
        id_props.pNext = pnext;
        pnext = &id_props;
        subgroup_props.pNext = pnext;
        pnext = &subgroup_props;
        if (chain_driver_props) {
            drv_props.pNext = pnext;
            pnext = &drv_props;
        }
        if (has_pci_bus_info) {
            pci_props.pNext = pnext;
            pnext = &pci_props;
        }
        if (chain_amd_arch_props) {
            amd_core_props.pNext = pnext;
            pnext = &amd_core_props;
            int_dot_props.pNext = pnext;
            pnext = &int_dot_props;
            subgroup_ctrl_props.pNext = pnext;
            pnext = &subgroup_ctrl_props;
        }
        props2.pNext = pnext;
        get_props2(pd, &props2);

        // Skip CPU/virtual/software Vulkan adapters (e.g. llvmpipe).
        if (props2.properties.deviceType == 4 /*VK_PHYSICAL_DEVICE_TYPE_CPU*/ ||
            props2.properties.vendorID == 0x1414u) {
            continue;
        }

        VkPhysicalDeviceProbe p;
        p.vendor_id = props2.properties.vendorID;
        p.device_id = props2.properties.deviceID;
        p.is_integrated_uma = (props2.properties.deviceType == 1 /*INTEGRATED_GPU*/);
        p.name = sanitize_driver_build_string(std::string_view(
            props2.properties.deviceName, strnlen(props2.properties.deviceName, sizeof(props2.properties.deviceName))));
        if (id_props.deviceLUIDValid != 0) {
            p.luid.valid = true;
            std::memcpy(p.luid.bytes, id_props.deviceLUID, sizeof(p.luid.bytes));
        }
        if (has_pci_bus_info) {
            p.pci_bus_id = format_pci_bdf(pci_props.pciDomain, pci_props.pciBus, pci_props.pciDevice,
                                          pci_props.pciFunction);
        }
        if (chain_driver_props) {
            p.driver_name = sanitize_driver_build_string(
                std::string_view(drv_props.driverName, strnlen(drv_props.driverName, sizeof(drv_props.driverName))));
            p.driver_info = sanitize_driver_build_string(
                std::string_view(drv_props.driverInfo, strnlen(drv_props.driverInfo, sizeof(drv_props.driverInfo))));
        }
        std::uint32_t shared_mem = 0;
        std::memcpy(&shared_mem, reinterpret_cast<const std::uint8_t*>(props2.properties.limits_and_sparse) + 216u,
                    sizeof(shared_mem));
        if (shared_mem > 0) {
            p.shared_memory_bytes = static_cast<std::uint64_t>(shared_mem);
        }
        if (subgroup_props.subgroupSize > 0) {
            p.warp_size = subgroup_props.subgroupSize;
        }
        if (has_coopmat2) {
            p.matrix_cores = "NV_coopmat2";
        } else if (has_coopmat) {
            p.matrix_cores = "KHR_coopmat";
        } else {
            p.matrix_cores = "none";
        }
        // Mirror ggml_vk_khr_cooperative_matrix_support (ggml-vulkan.cpp:3-56, 16858-16863):
        // on AMD proprietary/open-source drivers, ggml enables KHR coopmat only on RDNA3/RDNA4.
        if (p.vendor_id == 0x1002u) {
            if (!has_coopmat) {
                p.ggml_coopmat_eligible = false;
            } else if (chain_driver_props &&
                       (drv_props.driverID == kVkDriverAmdProprietary || drv_props.driverID == kVkDriverAmdOpenSource)) {
                bool rdna3_or_4 = false;
                if (chain_amd_arch_props && subgroup_ctrl_props.maxSubgroupSize == 64u &&
                    subgroup_ctrl_props.minSubgroupSize == 32u && amd_core_props.wavefrontsPerSimd != 20u) {
                    const bool is_rdna4 = has_shader_float8;
                    const bool is_rdna3 = (int_dot_props.flags[5] != 0u);
                    rdna3_or_4 = is_rdna3 || is_rdna4;
                }
                p.ggml_coopmat_eligible = rdna3_or_4;
            } else if (chain_driver_props) {
                p.ggml_coopmat_eligible = true;
            }
        }

        VkPhysicalDeviceMemoryProperties2Raw mem2{};
        mem2.sType = kStMemProps2;
        VkPhysicalDeviceMemoryBudgetPropertiesEXTRaw budget{};
        budget.sType = kStMemBudget;
        if (has_mem_budget) {
            mem2.pNext = &budget;
        }
        get_mem_props2(pd, &mem2);

        const std::uint32_t heap_count = std::min<std::uint32_t>(mem2.memoryProperties.memoryHeapCount, kMaxHeaps);
        const std::uint32_t type_count = std::min<std::uint32_t>(mem2.memoryProperties.memoryTypeCount, kMaxTypes);
        p.heaps.reserve(heap_count);
        for (std::uint32_t hi = 0; hi < heap_count; ++hi) {
            VulkanMemoryHeapInfo h;
            h.heap_index = hi;
            h.size_bytes = mem2.memoryProperties.memoryHeaps[hi].size;
            h.device_local = (mem2.memoryProperties.memoryHeaps[hi].flags & kMemHeapDeviceLocal) != 0;
            h.multi_instance = (mem2.memoryProperties.memoryHeaps[hi].flags & kMemHeapMultiInstance) != 0;
            if (has_mem_budget) {
                h.budget_bytes = MemoryReading::known(budget.heapBudget[hi]);
                h.usage_bytes = MemoryReading::known(budget.heapUsage[hi]);
            }
            for (std::uint32_t ti = 0; ti < type_count; ++ti) {
                const auto& mt = mem2.memoryProperties.memoryTypes[ti];
                if (mt.heapIndex != hi) continue;
                if ((mt.propertyFlags & kMemPropHostVisible) != 0) {
                    h.host_visible = true;
                }
                constexpr VkFlags kHvv = kMemPropDeviceLocal | kMemPropHostVisible | kMemPropHostCoherent;
                if ((mt.propertyFlags & kHvv) == kHvv) {
                    h.host_coherent = true;
                }
            }
            p.heaps.push_back(std::move(h));
        }
        std::optional<std::uint32_t> first_device_local_heap;
        for (std::uint32_t ti = 0; ti < type_count; ++ti) {
            const auto& mt = mem2.memoryProperties.memoryTypes[ti];
            if ((mt.propertyFlags & kMemPropDeviceLocal) != 0 && mt.heapIndex < heap_count) {
                first_device_local_heap = mt.heapIndex;
                break;
            }
        }
        p.rebar = classify_vulkan_rebar(p.heaps, first_device_local_heap);
        out.push_back(std::move(p));
    }
    return out;
}

void enrich_gpus_with_vulkan(std::vector<GpuDeviceInfo>& gpus, const std::vector<VkProbeAdapterLuid>& adapter_luids) {
    if (gpus.empty()) {
        return;
    }
    // Cache the Vulkan physical-device probe for 30 s so 2 s health refreshes
    // and 10 s engine samples do not create/destroy a VkInstance on every call
    // (and no persistent VkInstance handle is kept alive across DLL unload).
    constexpr auto kVulkanProbeTtl = std::chrono::seconds(30);
    static bool vk_cache_valid = false;
    static std::chrono::steady_clock::time_point vk_cache_time{};
    static std::vector<VkPhysicalDeviceProbe> vk_cache_devs;
    const auto now = std::chrono::steady_clock::now();
    if (!vk_cache_valid || (now - vk_cache_time) >= kVulkanProbeTtl) {
        vk_cache_devs = probe_vulkan_physical_devices();
        vk_cache_time = now;
        vk_cache_valid = true;
    }
    const auto& vk_devs = vk_cache_devs;
    if (vk_devs.empty()) {
        return;
    }
    std::vector<bool> used(vk_devs.size(), false);
    auto apply_vk = [](GpuDeviceInfo& gpu, const VkPhysicalDeviceProbe& vk) {
        if (gpu.pci_bus_id.empty() && !vk.pci_bus_id.empty()) {
            gpu.pci_bus_id = vk.pci_bus_id;
        }
        if (gpu.driver_name.empty() && !vk.driver_name.empty()) {
            gpu.driver_name = vk.driver_name;
        }
        if (gpu.driver_build.empty() && !vk.driver_info.empty()) {
            gpu.driver_build = vk.driver_info;
        }
        if (!gpu.rebar.has_value() && vk.rebar.has_value()) {
            gpu.rebar = vk.rebar;
        }
        if (!gpu.shared_memory_bytes.has_value() && vk.shared_memory_bytes.has_value()) {
            gpu.shared_memory_bytes = vk.shared_memory_bytes;
        }
        if (!gpu.warp_size.has_value() && vk.warp_size.has_value()) {
            gpu.warp_size = vk.warp_size;
        }
        if (gpu.matrix_cores.empty() && !vk.matrix_cores.empty()) {
            gpu.matrix_cores = vk.matrix_cores;
        }
        if (!gpu.ggml_coopmat_eligible.has_value() && vk.ggml_coopmat_eligible.has_value()) {
            gpu.ggml_coopmat_eligible = vk.ggml_coopmat_eligible;
        }
        if (gpu.vulkan_heaps.empty() && !vk.heaps.empty()) {
            gpu.vulkan_heaps = vk.heaps;
        }
    };

    for (std::size_t gi = 0; gi < gpus.size(); ++gi) {
        auto& gpu = gpus[gi];
        std::optional<std::size_t> match;
        // 1. Exact Windows AdapterLuid match.
        if (gi < adapter_luids.size() && adapter_luids[gi].valid) {
            for (std::size_t vi = 0; vi < vk_devs.size(); ++vi) {
                if (!used[vi] && vk_devs[vi].luid.valid &&
                    std::memcmp(vk_devs[vi].luid.bytes, adapter_luids[gi].bytes, 8u) == 0) {
                    match = vi;
                    break;
                }
            }
        }
        // 2. Exact PCI BDF match (normalized across 4-digit and 8-digit domains).
        if (!match && !gpu.pci_bus_id.empty()) {
            const std::string norm_gpu_bdf = normalize_pci_bus_id(gpu.pci_bus_id);
            for (std::size_t vi = 0; vi < vk_devs.size(); ++vi) {
                if (!used[vi] && !vk_devs[vi].pci_bus_id.empty() &&
                    normalize_pci_bus_id(vk_devs[vi].pci_bus_id) == norm_gpu_bdf) {
                    match = vi;
                    break;
                }
            }
        }
        // 3. PCI vendor_id + device_id match.
        if (!match && gpu.vendor_id != 0 && gpu.device_id != 0) {
            for (std::size_t vi = 0; vi < vk_devs.size(); ++vi) {
                if (!used[vi] && vk_devs[vi].vendor_id == gpu.vendor_id && vk_devs[vi].device_id == gpu.device_id) {
                    match = vi;
                    break;
                }
            }
        }
        if (match) {
            used[*match] = true;
            apply_vk(gpu, vk_devs[*match]);
        }
    }
}
#endif

#if defined(_WIN32)
std::string cpu_name() {
    char buffer[256] = {};
    DWORD size = sizeof(buffer);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, buffer, &size) == ERROR_SUCCESS) {
        return trim(buffer);
    }
    return "cpu";
}

HostMemory live_host_memory() {
    HostMemory out;
    out.source = "GlobalMemoryStatusEx";
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        out.note = "GlobalMemoryStatusEx failed (error " + std::to_string(GetLastError()) + ")";
        return out;
    }
    out.total_physical = MemoryReading::known(status.ullTotalPhys);
    out.available_physical = MemoryReading::known(status.ullAvailPhys);
    out.available_commit = MemoryReading::known(status.ullAvailPageFile);
    return out;
}

constexpr GUID kIidDxgiFactory1 = {0x770aae78u, 0xf26fu, 0x4dbau, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
constexpr GUID kIidDxgiAdapter3 = {0x645967a4u, 0x1392u, 0x4310u, {0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd}};

struct ScopedLibrary {
    HMODULE handle = nullptr;
    explicit ScopedLibrary(HMODULE h) noexcept : handle(h) {}
    ~ScopedLibrary() {
        if (handle) {
            FreeLibrary(handle);
        }
    }
    ScopedLibrary(const ScopedLibrary&) = delete;
    ScopedLibrary& operator=(const ScopedLibrary&) = delete;
};

template <typename T>
struct ScopedComPtr {
    T* ptr = nullptr;
    ScopedComPtr() noexcept = default;
    ~ScopedComPtr() {
        if (ptr) {
            ptr->Release();
        }
    }
    ScopedComPtr(const ScopedComPtr&) = delete;
    ScopedComPtr& operator=(const ScopedComPtr&) = delete;
};

template <typename Fn>
bool load_win32_symbol(HMODULE handle, const char* symbol, Fn& out) noexcept {
    static_assert(sizeof(Fn) == sizeof(FARPROC));
    const FARPROC raw = GetProcAddress(handle, symbol);
    if (!raw) {
        return false;
    }
    std::memcpy(&out, &raw, sizeof(out));
    return true;
}

std::string utf8_from_wide(const wchar_t* text, std::size_t max_chars) {
    if (!text) {
        return {};
    }
    std::size_t len = 0;
    while (len < max_chars && text[len] != L'\0') {
        ++len;
    }
    if (len == 0) {
        return {};
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return {};
    }
    std::string out(static_cast<std::size_t>(needed), '\0');
    const int written =
        WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(len), out.data(), needed, nullptr, nullptr);
    if (written <= 0) {
        return {};
    }
    out.resize(static_cast<std::size_t>(written));
    return out;
}

std::optional<std::string> read_win32_env_utf8(const char* name) {
    SetLastError(ERROR_SUCCESS);
    char stack_buf[512] = {};
    const DWORD n = GetEnvironmentVariableA(name, stack_buf, static_cast<DWORD>(sizeof(stack_buf)));
    if (n == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        return std::string{};
    }
    if (n < sizeof(stack_buf)) {
        return std::string(stack_buf, static_cast<std::size_t>(n));
    }
    if (n > 32768u) {
        return std::nullopt;
    }
    std::string dyn(static_cast<std::size_t>(n), '\0');
    const DWORD n2 = GetEnvironmentVariableA(name, dyn.data(), n);
    if (n2 == 0 || n2 >= n) {
        return std::nullopt;
    }
    dyn.resize(static_cast<std::size_t>(n2));
    return dyn;
}

bool win32_file_exists(const std::wstring& path) noexcept {
    const DWORD attrs = GetFileAttributesW(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool detect_windows_hip_sdk() {
    for (const wchar_t* env_name : {L"HIP_PATH", L"ROCM_PATH"}) {
        wchar_t root_buf[MAX_PATH * 2] = {};
        const DWORD len = GetEnvironmentVariableW(env_name, root_buf, static_cast<DWORD>(sizeof(root_buf) / sizeof(wchar_t)));
        if (len == 0 || len >= sizeof(root_buf) / sizeof(wchar_t)) {
            continue;
        }
        std::wstring base(root_buf, static_cast<std::size_t>(len));
        while (!base.empty() && (base.back() == L'\\' || base.back() == L'/')) {
            base.pop_back();
        }
        if (base.empty()) {
            continue;
        }
        const std::wstring bin_dir = base + L"\\bin\\";
        // Require the HIP SDK toolkit bin directory (hipblas.dll / hipcc) rather
        // than bare System32 amdhip64_6.dll from consumer display drivers.
        if ((win32_file_exists(bin_dir + L"hipblas.dll") || win32_file_exists(bin_dir + L"amdhip64_7.dll") ||
             win32_file_exists(bin_dir + L"amdhip64_6.dll") || win32_file_exists(bin_dir + L"amdhip64.dll")) &&
            (win32_file_exists(bin_dir + L"hipblas.dll") || win32_file_exists(bin_dir + L"hipcc.exe") ||
             win32_file_exists(bin_dir + L"hipcc.bat") || win32_file_exists(bin_dir + L"hipcc.pl"))) {
            return true;
        }
    }
    return false;
}

std::vector<GpuDeviceInfo> enumerate_win32_nvml_devices() {
    std::vector<GpuDeviceInfo> out;
    ScopedLibrary nvml(LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (!nvml.handle) {
        return out;
    }
    using NvmlReturn = int;
    struct NvmlDeviceOpaque;
    using NvmlDevice = NvmlDeviceOpaque*;
    struct NvmlMemory {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };
    struct NvmlPciInfo {
        char bus_id_legacy[16];
        unsigned int domain;
        unsigned int bus;
        unsigned int device;
        unsigned int pci_device_id;
        unsigned int pci_subsystem_id;
        char bus_id[32];
    };
    constexpr NvmlReturn kNvmlSuccess = 0;
    constexpr int kNvmlDriverTcc = 1;

    NvmlReturn (*nvml_init)() = nullptr;
    NvmlReturn (*nvml_shutdown)() = nullptr;
    NvmlReturn (*nvml_get_count)(unsigned int*) = nullptr;
    NvmlReturn (*nvml_get_handle)(unsigned int, NvmlDevice*) = nullptr;
    NvmlReturn (*nvml_get_name)(NvmlDevice, char*, unsigned int) = nullptr;
    NvmlReturn (*nvml_get_memory)(NvmlDevice, NvmlMemory*) = nullptr;
    NvmlReturn (*nvml_get_pci)(NvmlDevice, NvmlPciInfo*) = nullptr;
    NvmlReturn (*nvml_get_driver_model)(NvmlDevice, int*, int*) = nullptr;
    NvmlReturn (*nvml_sys_driver_ver)(char*, unsigned int) = nullptr;
    NvmlReturn (*nvml_curr_pcie_gen)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_curr_pcie_width)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_max_pcie_gen)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_max_pcie_width)(NvmlDevice, unsigned int*) = nullptr;

    if (!load_win32_symbol(nvml.handle, "nvmlInit_v2", nvml_init) ||
        !load_win32_symbol(nvml.handle, "nvmlShutdown", nvml_shutdown) ||
        !load_win32_symbol(nvml.handle, "nvmlDeviceGetCount_v2", nvml_get_count) ||
        !load_win32_symbol(nvml.handle, "nvmlDeviceGetHandleByIndex_v2", nvml_get_handle) ||
        !load_win32_symbol(nvml.handle, "nvmlDeviceGetName", nvml_get_name) ||
        !load_win32_symbol(nvml.handle, "nvmlDeviceGetMemoryInfo", nvml_get_memory)) {
        return out;
    }
    if (!load_win32_symbol(nvml.handle, "nvmlDeviceGetPciInfo_v3", nvml_get_pci)) {
        (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetPciInfo", nvml_get_pci);
    }
    if (!load_win32_symbol(nvml.handle, "nvmlDeviceGetDriverModel_v2", nvml_get_driver_model)) {
        (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetDriverModel", nvml_get_driver_model);
    }
    (void)load_win32_symbol(nvml.handle, "nvmlSystemGetDriverVersion", nvml_sys_driver_ver);
    (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetCurrPcieLinkGeneration", nvml_curr_pcie_gen);
    (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetCurrPcieLinkWidth", nvml_curr_pcie_width);
    (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetMaxPcieLinkGeneration", nvml_max_pcie_gen);
    (void)load_win32_symbol(nvml.handle, "nvmlDeviceGetMaxPcieLinkWidth", nvml_max_pcie_width);
    if (nvml_init() != kNvmlSuccess) {
        return out;
    }
    struct ScopedNvml {
        NvmlReturn (*shutdown)() = nullptr;
        explicit ScopedNvml(NvmlReturn (*fn)()) noexcept : shutdown(fn) {}
        ~ScopedNvml() {
            if (shutdown) {
                (void)shutdown();
            }
        }
        ScopedNvml(const ScopedNvml&) = delete;
        ScopedNvml& operator=(const ScopedNvml&) = delete;
    } nvml_guard(nvml_shutdown);

    std::string nvml_driver_build;
    if (nvml_sys_driver_ver) {
        char drv_buf[96] = {};
        if (nvml_sys_driver_ver(drv_buf, static_cast<unsigned int>(sizeof(drv_buf))) == kNvmlSuccess) {
            drv_buf[sizeof(drv_buf) - 1] = '\0';
            nvml_driver_build = sanitize_driver_build_string(drv_buf);
        }
    }

    unsigned int count = 0;
    if (nvml_get_count(&count) != kNvmlSuccess || count == 0) {
        return out;
    }
    count = std::min<unsigned int>(count, 64u);
    for (unsigned int i = 0; i < count; ++i) {
        NvmlDevice dev = nullptr;
        if (nvml_get_handle(i, &dev) != kNvmlSuccess || !dev) {
            continue;
        }
        GpuDeviceInfo info;
        info.id = "gpu:" + std::to_string(out.size());
        char name_buf[128] = {};
        if (nvml_get_name(dev, name_buf, static_cast<unsigned int>(sizeof(name_buf))) == kNvmlSuccess) {
            name_buf[sizeof(name_buf) - 1] = '\0';
            info.name = trim(name_buf);
        }
        if (info.name.empty()) {
            info.name = "NVIDIA GPU " + std::to_string(i);
        }
        info.vendor = GpuVendor::nvidia;
        info.vendor_id = 0x10DEu;
        if (nvml_get_pci) {
            NvmlPciInfo pci{};
            if (nvml_get_pci(dev, &pci) == kNvmlSuccess) {
                info.device_id = static_cast<std::uint32_t>(pci.pci_device_id >> 16);
                pci.bus_id[sizeof(pci.bus_id) - 1] = '\0';
                info.pci_bus_id = trim(pci.bus_id);
            }
        }
        if (nvml_get_driver_model) {
            int cur_model = 0;
            int pending_model = 0;
            if (nvml_get_driver_model(dev, &cur_model, &pending_model) == kNvmlSuccess) {
                info.is_tcc_mode = (cur_model == kNvmlDriverTcc);
            }
        }
        if (!nvml_driver_build.empty()) {
            info.driver_name = "NVIDIA";
            info.driver_build = nvml_driver_build;
        }
        unsigned int pcie_val = 0;
        if (nvml_curr_pcie_gen && nvml_curr_pcie_gen(dev, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 6u) {
            info.pcie.current_gen = pcie_val;
        }
        if (nvml_curr_pcie_width && nvml_curr_pcie_width(dev, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 32u) {
            info.pcie.current_width = pcie_val;
        }
        if (nvml_max_pcie_gen && nvml_max_pcie_gen(dev, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 6u) {
            info.pcie.max_gen = pcie_val;
        }
        if (nvml_max_pcie_width && nvml_max_pcie_width(dev, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 32u) {
            info.pcie.max_width = pcie_val;
        }
        NvmlMemory mem{};
        if (nvml_get_memory(dev, &mem) == kNvmlSuccess) {
            info.dedicated_total = MemoryReading::known(static_cast<std::uint64_t>(mem.total));
            if (mem.total == 0 || mem.free > 0 || mem.used > 0) {
                info.dedicated_free = MemoryReading::known(static_cast<std::uint64_t>(mem.free));
            }
        }
        info.is_integrated_uma = false;
        info.backend_hint = "cuda";
        info.source = "nvml";
        out.push_back(std::move(info));
    }
    return out;
}

std::string read_reg_sz_value(HKEY key, const char* value_name) {
    char buf[512] = {};
    DWORD size = sizeof(buf);
    if (RegGetValueA(key, nullptr, value_name, RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS && size > 0) {
        buf[sizeof(buf) - 1] = '\0';
        return sanitize_driver_build_string(buf);
    }
    return {};
}

bool parse_ven_dev_from_pnp_id(std::string_view pnp_id, std::uint32_t& ven_out, std::uint32_t& dev_out) noexcept {
    auto find_hex_tag = [&](std::string_view tag, std::uint32_t& out) noexcept -> bool {
        for (std::size_t i = 0; i + tag.size() + 4 <= pnp_id.size(); ++i) {
            if (iequals(pnp_id.substr(i, tag.size()), tag)) {
                if (const auto v = parse_hex_u32(pnp_id.substr(i + tag.size(), 4))) {
                    out = *v;
                    return true;
                }
            }
        }
        return false;
    };
    return find_hex_tag("VEN_", ven_out) && find_hex_tag("DEV_", dev_out);
}

void enrich_win32_registry_drivers(std::vector<GpuDeviceInfo>& gpus) {
    if (gpus.empty()) return;
    HKEY class_key = nullptr;
    constexpr const char* kDisplayClass =
        "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, kDisplayClass, 0, KEY_READ, &class_key) != ERROR_SUCCESS || !class_key) {
        return;
    }
    struct ScopedRegKey {
        HKEY h = nullptr;
        ~ScopedRegKey() {
            if (h) RegCloseKey(h);
        }
    } class_guard{class_key};

    struct RegDriverEntry {
        std::uint32_t vendor_id = 0;
        std::uint32_t device_id = 0;
        std::string desc;
        std::string driver_build;
    };
    std::vector<RegDriverEntry> entries;
    for (DWORD idx = 0; idx < 64u; ++idx) {
        char sub_name[64] = {};
        DWORD sub_len = static_cast<DWORD>(sizeof(sub_name));
        if (RegEnumKeyExA(class_key, idx, sub_name, &sub_len, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) {
            break;
        }
        if (sub_len != 4 || sub_name[0] < '0' || sub_name[0] > '9') {
            continue;
        }
        HKEY sub_key = nullptr;
        if (RegOpenKeyExA(class_key, sub_name, 0, KEY_READ, &sub_key) != ERROR_SUCCESS || !sub_key) {
            continue;
        }
        ScopedRegKey sub_guard{sub_key};
        const std::string match_id = read_reg_sz_value(sub_key, "MatchingDeviceId");
        std::uint32_t ven = 0;
        std::uint32_t dev = 0;
        if (!parse_ven_dev_from_pnp_id(match_id, ven, dev)) {
            continue;
        }
        const std::string drv_ver = read_reg_sz_value(sub_key, "DriverVersion");
        std::string rel_ver = read_reg_sz_value(sub_key, "RadeonSoftwareVersion");
        if (rel_ver.empty()) {
            rel_ver = read_reg_sz_value(sub_key, "ReleaseVersion");
        }
        const std::string build = format_driver_build_string(drv_ver, rel_ver);
        if (build.empty()) {
            continue;
        }
        RegDriverEntry e;
        e.vendor_id = ven;
        e.device_id = dev;
        e.desc = read_reg_sz_value(sub_key, "DriverDesc");
        e.driver_build = build;
        entries.push_back(std::move(e));
    }

    for (auto& gpu : gpus) {
        if (!gpu.driver_build.empty() || gpu.vendor_id == 0 || gpu.device_id == 0) continue;
        for (const auto& e : entries) {
            if (e.vendor_id == gpu.vendor_id && e.device_id == gpu.device_id) {
                gpu.driver_build = e.driver_build;
                break;
            }
        }
    }
}

void enrich_win32_pcie_and_aspm(std::vector<GpuDeviceInfo>& gpus) {
    if (gpus.empty()) return;

    // 1. Windows PCIe link speed/width via cfgmgr32.dll (for non-NVIDIA or
    //    any GPU missing PCIe readings).
    ScopedLibrary cfgmgr(LoadLibraryExW(L"cfgmgr32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (cfgmgr.handle) {
        struct DevPropKeyRaw {
            GUID fmtid;
            ULONG pid;
        };
        constexpr GUID kPciDevicePropGuid = {
            0x3ab22e31u, 0x8264u, 0x4b4eu, {0x9au, 0xf5u, 0xa8u, 0xd2u, 0xd8u, 0xe3u, 0x3eu, 0x62u}};
        constexpr DevPropKeyRaw kKeyCurrSpeed{kPciDevicePropGuid, 9u};
        constexpr DevPropKeyRaw kKeyCurrWidth{kPciDevicePropGuid, 10u};
        constexpr DevPropKeyRaw kKeyMaxSpeed{kPciDevicePropGuid, 11u};
        constexpr DevPropKeyRaw kKeyMaxWidth{kPciDevicePropGuid, 12u};
        constexpr ULONG kCmGetIdListFilterClass = 0x00000200u;
        constexpr ULONG kCmGetIdListFilterPresent = 0x00000100u;
        constexpr ULONG kDevPropTypeUint32 = 0x00000007u;

        using GetListSizeFn = DWORD(WINAPI*)(ULONG*, const wchar_t*, ULONG);
        using GetListFn = DWORD(WINAPI*)(const wchar_t*, wchar_t*, ULONG, ULONG);
        using LocateFn = DWORD(WINAPI*)(DWORD*, wchar_t*, ULONG);
        using GetPropFn = DWORD(WINAPI*)(DWORD, const DevPropKeyRaw*, ULONG*, BYTE*, ULONG*, ULONG);

        GetListSizeFn get_list_size = nullptr;
        GetListFn get_list = nullptr;
        LocateFn locate = nullptr;
        GetPropFn get_prop = nullptr;
        if (load_win32_symbol(cfgmgr.handle, "CM_Get_Device_ID_List_SizeW", get_list_size) &&
            load_win32_symbol(cfgmgr.handle, "CM_Get_Device_ID_ListW", get_list) &&
            load_win32_symbol(cfgmgr.handle, "CM_Locate_DevNodeW", locate) &&
            load_win32_symbol(cfgmgr.handle, "CM_Get_DevNode_PropertyW", get_prop)) {
            const wchar_t* kDisplayGuid = L"{4d36e968-e325-11ce-bfc1-08002be10318}";
            ULONG chars = 0;
            if (get_list_size(&chars, kDisplayGuid, kCmGetIdListFilterClass | kCmGetIdListFilterPresent) == 0 &&
                chars > 1u && chars < 65536u) {
                std::vector<wchar_t> buf(chars, L'\0');
                if (get_list(kDisplayGuid, buf.data(), chars, kCmGetIdListFilterClass | kCmGetIdListFilterPresent) ==
                    0) {
                    auto read_u32_prop = [&](DWORD dev_inst, const DevPropKeyRaw& key) -> std::optional<std::uint32_t> {
                        ULONG prop_type = 0;
                        std::uint32_t value = 0;
                        ULONG sz = sizeof(value);
                        if (get_prop(dev_inst, &key, &prop_type, reinterpret_cast<BYTE*>(&value), &sz, 0) == 0 &&
                            prop_type == kDevPropTypeUint32 && sz == sizeof(value)) {
                            return value;
                        }
                        return std::nullopt;
                    };
                    const wchar_t* cur = buf.data();
                    const wchar_t* end = buf.data() + buf.size();
                    while (cur < end && *cur != L'\0') {
                        const std::size_t len = wcsnlen(cur, static_cast<std::size_t>(end - cur));
                        std::wstring id_w(cur, len);
                        cur += len + 1;
                        const std::string id_u8 = utf8_from_wide(id_w.c_str(), id_w.size());
                        std::uint32_t ven = 0;
                        std::uint32_t dev = 0;
                        if (!parse_ven_dev_from_pnp_id(id_u8, ven, dev)) {
                            continue;
                        }
                        DWORD dev_inst = 0;
                        if (locate(&dev_inst, id_w.data(), 0) != 0) {
                            continue;
                        }
                        const auto cgen = read_u32_prop(dev_inst, kKeyCurrSpeed);
                        const auto cwid = read_u32_prop(dev_inst, kKeyCurrWidth);
                        const auto mgen = read_u32_prop(dev_inst, kKeyMaxSpeed);
                        const auto mwid = read_u32_prop(dev_inst, kKeyMaxWidth);
                        for (auto& gpu : gpus) {
                            if (gpu.vendor_id != ven || gpu.device_id != dev) continue;
                            if (gpu.pcie.current_gen && gpu.pcie.current_width && gpu.pcie.max_gen &&
                                gpu.pcie.max_width) {
                                continue;
                            }
                            if (!gpu.pcie.current_gen && cgen && *cgen >= 1u && *cgen <= 6u) {
                                gpu.pcie.current_gen = *cgen;
                            }
                            if (!gpu.pcie.current_width && cwid && *cwid >= 1u && *cwid <= 32u) {
                                gpu.pcie.current_width = *cwid;
                            }
                            if (!gpu.pcie.max_gen && mgen && *mgen >= 1u && *mgen <= 6u) {
                                gpu.pcie.max_gen = *mgen;
                            }
                            if (!gpu.pcie.max_width && mwid && *mwid >= 1u && *mwid <= 32u) {
                                gpu.pcie.max_width = *mwid;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    // 2. Active Windows power-plan PCIe Link State Power Management (ASPM).
    ScopedLibrary powrprof(LoadLibraryExW(L"powrprof.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (powrprof.handle) {
        using GetActiveSchemeFn = DWORD(WINAPI*)(HKEY, GUID**);
        using ReadValueIdxFn = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, DWORD*);
        GetActiveSchemeFn get_scheme = nullptr;
        ReadValueIdxFn read_ac = nullptr;
        ReadValueIdxFn read_dc = nullptr;
        if (load_win32_symbol(powrprof.handle, "PowerGetActiveScheme", get_scheme) &&
            load_win32_symbol(powrprof.handle, "PowerReadACValueIndex", read_ac)) {
            (void)load_win32_symbol(powrprof.handle, "PowerReadDCValueIndex", read_dc);
            GUID* active_scheme = nullptr;
            if (get_scheme(nullptr, &active_scheme) == ERROR_SUCCESS && active_scheme) {
                struct ScopedLocalGuid {
                    GUID* p = nullptr;
                    ~ScopedLocalGuid() {
                        if (p) LocalFree(p);
                    }
                } scheme_guard{active_scheme};

                constexpr GUID kSubgroupPciExpress = {
                    0x501a4d13u, 0x42afu, 0x4429u, {0x9fu, 0xd1u, 0xa8u, 0x21u, 0x8cu, 0x26u, 0x8eu, 0x20u}};
                constexpr GUID kSettingAspmPolicy = {
                    0xee12f906u, 0xd277u, 0x404bu, {0xb6u, 0xdau, 0xe5u, 0xfau, 0x1au, 0x57u, 0x6du, 0xf5u}};

                bool on_dc = false;
                SYSTEM_POWER_STATUS sps{};
                if (GetSystemPowerStatus(&sps) && sps.ACLineStatus == 0) {
                    on_dc = true;
                }
                DWORD policy_idx = 0xFFFFFFFFu;
                const DWORD rc =
                    (on_dc && read_dc)
                        ? read_dc(nullptr, active_scheme, &kSubgroupPciExpress, &kSettingAspmPolicy, &policy_idx)
                        : read_ac(nullptr, active_scheme, &kSubgroupPciExpress, &kSettingAspmPolicy, &policy_idx);
                if (rc == ERROR_SUCCESS) {
                    const std::string_view aspm = windows_aspm_policy_name(policy_idx);
                    if (!aspm.empty()) {
                        for (auto& gpu : gpus) {
                            if (gpu.pcie.aspm_state.empty()) {
                                gpu.pcie.aspm_state = std::string(aspm);
                                gpu.pcie.aspm_source = "windows_power_plan";
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<GpuDeviceInfo> live_gpu_devices() {
    std::vector<GpuDeviceInfo> gpus;
    std::vector<VkProbeAdapterLuid> adapter_luids;
    const bool hip_sdk_available = detect_windows_hip_sdk();
    const auto hip_vis = read_win32_env_utf8("HIP_VISIBLE_DEVICES");
    const auto rocr_vis = read_win32_env_utf8("ROCR_VISIBLE_DEVICES");
    const std::optional<std::string_view> hip_vis_view =
        hip_vis ? std::optional<std::string_view>(*hip_vis) : std::nullopt;
    const std::optional<std::string_view> rocr_vis_view =
        rocr_vis ? std::optional<std::string_view>(*rocr_vis) : std::nullopt;

    ScopedLibrary dxgi(LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (dxgi.handle) {
        using CreateDXGIFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);
        CreateDXGIFactory1Fn create_factory = nullptr;
        if (load_win32_symbol(dxgi.handle, "CreateDXGIFactory1", create_factory)) {
            ScopedComPtr<IDXGIFactory1> factory;
            if (SUCCEEDED(create_factory(kIidDxgiFactory1, reinterpret_cast<void**>(&factory.ptr))) && factory.ptr) {
                std::size_t amd_ordinal = 0;
                for (UINT i = 0;; ++i) {
                    ScopedComPtr<IDXGIAdapter1> adapter;
                    const HRESULT hr = factory.ptr->EnumAdapters1(i, &adapter.ptr);
                    if (FAILED(hr) || !adapter.ptr) {
                        break;
                    }
                    DXGI_ADAPTER_DESC1 desc{};
                    if (FAILED(adapter.ptr->GetDesc1(&desc))) {
                        continue;
                    }
                    // Skip software/WARP adapters and Microsoft virtual/RDP display adapters (0x1414).
                    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 || desc.VendorId == 0x1414u) {
                        continue;
                    }
                    GpuDeviceInfo info;
                    info.id = "gpu:" + std::to_string(gpus.size());
                    info.name =
                        trim(utf8_from_wide(desc.Description, sizeof(desc.Description) / sizeof(desc.Description[0])));
                    info.vendor_id = static_cast<std::uint32_t>(desc.VendorId);
                    info.device_id = static_cast<std::uint32_t>(desc.DeviceId);
                    info.vendor = gpu_vendor_from_pci_id(info.vendor_id);
                    const auto dedicated_total = static_cast<std::uint64_t>(desc.DedicatedVideoMemory);
                    const auto shared_total = static_cast<std::uint64_t>(desc.SharedSystemMemory);
                    info.dedicated_total = MemoryReading::known(dedicated_total);
                    info.shared_total = MemoryReading::known(shared_total);
                    info.is_integrated_uma = is_integrated_uma_memory(info.vendor, dedicated_total, shared_total);
                    std::size_t cur_amd_ordinal = 0;
                    if (info.vendor == GpuVendor::amd && !info.is_integrated_uma) {
                        cur_amd_ordinal = amd_ordinal++;
                        if (hip_sdk_available) {
                            info.rocm_runtime_ordinal =
                                map_rocm_device_ordinal(cur_amd_ordinal, hip_vis_view, rocr_vis_view);
                        }
                    }
                    info.backend_hint = resolve_windows_gpu_backend_hint(
                        info.vendor, info.is_integrated_uma, hip_sdk_available, cur_amd_ordinal, hip_vis_view,
                        rocr_vis_view);
                    info.source = "dxgi";

                    ScopedComPtr<IDXGIAdapter3> adapter3;
                    if (SUCCEEDED(adapter.ptr->QueryInterface(kIidDxgiAdapter3, reinterpret_cast<void**>(&adapter3.ptr))) &&
                        adapter3.ptr) {
                        DXGI_QUERY_VIDEO_MEMORY_INFO local_info{};
                        if (SUCCEEDED(adapter3.ptr->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                                         &local_info))) {
                            std::uint64_t free_bytes = local_info.Budget > local_info.CurrentUsage
                                                           ? local_info.Budget - local_info.CurrentUsage
                                                           : 0;
                            bool free_complete = true;
                            bool has_wddm_budget = (local_info.Budget > 0 || local_info.CurrentUsage > 0 ||
                                                    dedicated_total == 0);
                            if (info.is_integrated_uma) {
                                DXGI_QUERY_VIDEO_MEMORY_INFO non_local_info{};
                                if (SUCCEEDED(adapter3.ptr->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                                                                                 &non_local_info))) {
                                    const std::uint64_t non_local_free =
                                        non_local_info.Budget > non_local_info.CurrentUsage
                                            ? non_local_info.Budget - non_local_info.CurrentUsage
                                            : 0;
                                    free_bytes = saturating_add(free_bytes, non_local_free);
                                    has_wddm_budget = (local_info.Budget > 0 || non_local_info.Budget > 0 ||
                                                       local_info.CurrentUsage > 0 || non_local_info.CurrentUsage > 0 ||
                                                       (dedicated_total == 0 && shared_total == 0));
                                } else {
                                    free_complete = false;
                                }
                            }
                            if (free_complete && has_wddm_budget) {
                                info.dedicated_free = MemoryReading::known(free_bytes);
                            }
                        }
                    }
                    VkProbeAdapterLuid luid;
                    luid.valid = (desc.AdapterLuid.LowPart != 0 || desc.AdapterLuid.HighPart != 0);
                    if (luid.valid) {
                        static_assert(sizeof(desc.AdapterLuid) == 8u);
                        std::memcpy(luid.bytes, &desc.AdapterLuid, 8u);
                    }
                    adapter_luids.push_back(luid);
                    gpus.push_back(std::move(info));
                }
            }
        }
    }

    // Always query System32 nvml.dll and merge any headless / TCC-mode NVIDIA
    // GPUs that DXGI does not enumerate, even when a WDDM NVIDIA card is present.
    const auto nvml_gpus = enumerate_win32_nvml_devices();
    merge_win32_nvml_devices(gpus, nvml_gpus);
    adapter_luids.resize(gpus.size());
    // Cache the static Windows display-registry, Vulkan, cfgmgr32 PCIe, and
    // powrprof ASPM enrichment for 30 s keyed on adapter identity so 2 s / 10 s
    // polling refreshes live DXGI/NVML memory without repeating expensive probes.
    struct Win32AdapterKey {
        std::uint32_t vendor_id = 0;
        std::uint32_t device_id = 0;
        bool luid_valid = false;
        std::uint8_t luid[8] = {};
        bool operator==(const Win32AdapterKey& o) const noexcept {
            return vendor_id == o.vendor_id && device_id == o.device_id && luid_valid == o.luid_valid &&
                   std::memcmp(luid, o.luid, sizeof(luid)) == 0;
        }
    };
    struct Win32StaticTelemetryEntry {
        std::string pci_bus_id;
        std::string driver_name;
        std::string driver_build;
        std::optional<bool> rebar;
        std::optional<std::uint64_t> shared_memory_bytes;
        std::optional<std::uint32_t> warp_size;
        std::string matrix_cores;
        std::optional<bool> ggml_coopmat_eligible;
        PcieLinkTelemetry pcie;
        std::vector<VulkanMemoryHeapInfo> vulkan_heaps;
    };
    static bool win32_cache_valid = false;
    static std::chrono::steady_clock::time_point win32_cache_time{};
    static std::vector<Win32AdapterKey> win32_cache_keys;
    static std::vector<Win32StaticTelemetryEntry> win32_cache_entries;

    std::vector<Win32AdapterKey> cur_keys;
    cur_keys.reserve(gpus.size());
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        Win32AdapterKey k;
        k.vendor_id = gpus[i].vendor_id;
        k.device_id = gpus[i].device_id;
        k.luid_valid = adapter_luids[i].valid;
        if (k.luid_valid) {
            std::memcpy(k.luid, adapter_luids[i].bytes, sizeof(k.luid));
        }
        cur_keys.push_back(k);
    }

    const auto now = std::chrono::steady_clock::now();
    constexpr auto kWin32StaticTtl = std::chrono::seconds(30);
    if (win32_cache_valid && (now - win32_cache_time) < kWin32StaticTtl && win32_cache_keys == cur_keys &&
        win32_cache_entries.size() == gpus.size()) {
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            const auto& c = win32_cache_entries[i];
            if (gpus[i].pci_bus_id.empty()) gpus[i].pci_bus_id = c.pci_bus_id;
            if (gpus[i].driver_name.empty()) gpus[i].driver_name = c.driver_name;
            if (gpus[i].driver_build.empty()) gpus[i].driver_build = c.driver_build;
            if (!gpus[i].rebar.has_value()) gpus[i].rebar = c.rebar;
            if (!gpus[i].shared_memory_bytes.has_value()) gpus[i].shared_memory_bytes = c.shared_memory_bytes;
            if (!gpus[i].warp_size.has_value()) gpus[i].warp_size = c.warp_size;
            if (gpus[i].matrix_cores.empty()) gpus[i].matrix_cores = c.matrix_cores;
            if (!gpus[i].ggml_coopmat_eligible.has_value()) gpus[i].ggml_coopmat_eligible = c.ggml_coopmat_eligible;
            if (!gpus[i].pcie.current_gen.has_value()) gpus[i].pcie.current_gen = c.pcie.current_gen;
            if (!gpus[i].pcie.current_width.has_value()) gpus[i].pcie.current_width = c.pcie.current_width;
            if (!gpus[i].pcie.max_gen.has_value()) gpus[i].pcie.max_gen = c.pcie.max_gen;
            if (!gpus[i].pcie.max_width.has_value()) gpus[i].pcie.max_width = c.pcie.max_width;
            if (gpus[i].pcie.aspm_state.empty()) {
                gpus[i].pcie.aspm_state = c.pcie.aspm_state;
                gpus[i].pcie.aspm_source = c.pcie.aspm_source;
            }
            if (gpus[i].vulkan_heaps.empty()) gpus[i].vulkan_heaps = c.vulkan_heaps;
        }
    } else {
        enrich_win32_registry_drivers(gpus);
        enrich_gpus_with_vulkan(gpus, adapter_luids);
        enrich_win32_pcie_and_aspm(gpus);
        win32_cache_keys = std::move(cur_keys);
        win32_cache_entries.clear();
        win32_cache_entries.reserve(gpus.size());
        for (const auto& g : gpus) {
            win32_cache_entries.push_back(Win32StaticTelemetryEntry{
                g.pci_bus_id, g.driver_name, g.driver_build, g.rebar, g.shared_memory_bytes, g.warp_size,
                g.matrix_cores, g.ggml_coopmat_eligible, g.pcie, g.vulkan_heaps});
        }
        win32_cache_time = now;
        win32_cache_valid = true;
    }
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        gpus[i].id = "gpu:" + std::to_string(i);
    }
    return gpus;
}
#elif defined(__APPLE__)
std::string cpu_name() {
    char buffer[256] = {};
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
        return trim(buffer);
    }
    return "cpu";
}

HostMemory live_host_memory() {
    HostMemory out;
    out.source = "sysctl";
    std::uint64_t mem = 0;
    size_t size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0) {
        out.total_physical = MemoryReading::known(mem);
    }
    // Available memory is not cheaply available without mach APIs: unknown.
    out.note = "available memory is not read on macOS";
    return out;
}

std::vector<GpuDeviceInfo> live_gpu_devices() { return {}; }
#else
std::string cpu_name() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                return trim(line.substr(colon + 1));
            }
        }
    }
    return "cpu";
}

// Reads at most `limit` bytes; nullopt when the file cannot be opened. A file
// longer than `limit` is returned truncated, which the strict parsers then
// refuse or ignore.
std::optional<std::string> read_small_file(const std::string& path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::string text(limit, '\0');
    in.read(text.data(), static_cast<std::streamsize>(limit));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

HostMemory live_host_memory() {
    const auto meminfo = read_small_file("/proc/meminfo", 1u << 16);
    if (!meminfo) {
        HostMemory out;
        out.source = "meminfo";
        out.note = "/proc/meminfo could not be read";
        return out;
    }
    const auto proc_cgroup = read_small_file("/proc/self/cgroup", 1u << 16);
    CgroupHeadroom cgroup;
    if (proc_cgroup) {
        cgroup = cgroup_v2_headroom(*proc_cgroup, [](const std::string& path) { return read_small_file(path, 64); });
    }
    return compose_linux_host_memory(*meminfo, cgroup);
}

#  if defined(__linux__)
template <typename Fn>
bool load_dl_symbol(void* handle, const char* symbol, Fn& out) noexcept {
    static_assert(sizeof(Fn) == sizeof(void*));
    void* raw = dlsym(handle, symbol);
    if (!raw) {
        return false;
    }
    std::memcpy(&out, &raw, sizeof(out));
    return true;
}

void append_amdgpu_sysfs_devices(std::vector<GpuDeviceInfo>& gpus) {
    DIR* dir = opendir("/sys/class/drm");
    if (!dir) {
        return;
    }
    std::vector<std::pair<std::uint32_t, std::string>> cards;
    while (const dirent* entry = readdir(dir)) {
        const std::string_view name(entry->d_name);
        if (const auto idx = parse_card_index(name)) {
            cards.emplace_back(*idx, std::string(name));
        }
    }
    closedir(dir);

    std::sort(cards.begin(), cards.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    const bool kfd_accessible = (::access("/dev/kfd", R_OK | W_OK) == 0);
    const char* raw_hip_vis = std::getenv("HIP_VISIBLE_DEVICES");
    const char* raw_rocr_vis = std::getenv("ROCR_VISIBLE_DEVICES");
    const std::optional<std::string_view> hip_vis =
        raw_hip_vis ? std::optional<std::string_view>(raw_hip_vis) : std::nullopt;
    const std::optional<std::string_view> rocr_vis =
        raw_rocr_vis ? std::optional<std::string_view>(raw_rocr_vis) : std::nullopt;

    const FileReader reader = [](const std::string& path) { return read_small_file(path, 4096); };
    std::size_t amd_ordinal = 0;
    for (const auto& card : cards) {
        std::string uuid_str;
        std::optional<std::string_view> uuid_view;
        if (const auto uid = reader("/sys/class/drm/" + card.second + "/device/unique_id")) {
            std::string_view hex = trim_view(*uid);
            if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
                hex.remove_prefix(2);
            }
            if (!hex.empty()) {
                uuid_str = "GPU-" + std::string(hex);
                uuid_view = uuid_str;
            }
        }
        const auto rocm_ord =
            kfd_accessible ? map_rocm_device_ordinal(amd_ordinal, hip_vis, rocr_vis, uuid_view) : std::nullopt;
        if (auto gpu = parse_amdgpu_sysfs_device(card.second, reader, rocm_ord.has_value())) {
            gpu->rocm_runtime_ordinal = rocm_ord;
            ++amd_ordinal;
            gpus.push_back(std::move(*gpu));
        }
    }
}

using NvmlReturn = int;
struct NvmlDeviceOpaque;
using NvmlDevice = NvmlDeviceOpaque*;
struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
struct NvmlPciInfo {
    char bus_id_legacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pci_device_id;
    unsigned int pci_subsystem_id;
    char bus_id[32];
};

void append_nvml_gpu_devices(std::vector<GpuDeviceInfo>& gpus) {
    struct ScopedDl {
        void* handle = nullptr;
        explicit ScopedDl(void* h) noexcept : handle(h) {}
        ~ScopedDl() {
            if (handle) {
                dlclose(handle);
            }
        }
        ScopedDl(const ScopedDl&) = delete;
        ScopedDl& operator=(const ScopedDl&) = delete;
    };

    void* dl_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!dl_handle) {
        // WSL2 mounts host NVIDIA driver libraries under /usr/lib/wsl/lib
        // alongside /dev/dxg, which is not always in the default loader cache.
        dl_handle = dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1", RTLD_LAZY);
    }
    ScopedDl lib(dl_handle);
    if (!lib.handle) {
        return;
    }

    constexpr NvmlReturn kNvmlSuccess = 0;

    NvmlReturn (*nvml_init)() = nullptr;
    NvmlReturn (*nvml_shutdown)() = nullptr;
    NvmlReturn (*nvml_get_count)(unsigned int*) = nullptr;
    NvmlReturn (*nvml_get_handle)(unsigned int, NvmlDevice*) = nullptr;
    NvmlReturn (*nvml_get_name)(NvmlDevice, char*, unsigned int) = nullptr;
    NvmlReturn (*nvml_get_memory)(NvmlDevice, NvmlMemory*) = nullptr;
    NvmlReturn (*nvml_get_pci)(NvmlDevice, NvmlPciInfo*) = nullptr;
    NvmlReturn (*nvml_sys_driver_ver)(char*, unsigned int) = nullptr;
    NvmlReturn (*nvml_curr_pcie_gen)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_curr_pcie_width)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_max_pcie_gen)(NvmlDevice, unsigned int*) = nullptr;
    NvmlReturn (*nvml_max_pcie_width)(NvmlDevice, unsigned int*) = nullptr;

    if (!load_dl_symbol(lib.handle, "nvmlInit_v2", nvml_init) ||
        !load_dl_symbol(lib.handle, "nvmlShutdown", nvml_shutdown) ||
        !load_dl_symbol(lib.handle, "nvmlDeviceGetCount_v2", nvml_get_count) ||
        !load_dl_symbol(lib.handle, "nvmlDeviceGetHandleByIndex_v2", nvml_get_handle) ||
        !load_dl_symbol(lib.handle, "nvmlDeviceGetName", nvml_get_name) ||
        !load_dl_symbol(lib.handle, "nvmlDeviceGetMemoryInfo", nvml_get_memory)) {
        return;
    }
    if (!load_dl_symbol(lib.handle, "nvmlDeviceGetPciInfo_v3", nvml_get_pci)) {
        (void)load_dl_symbol(lib.handle, "nvmlDeviceGetPciInfo", nvml_get_pci);
    }
    (void)load_dl_symbol(lib.handle, "nvmlSystemGetDriverVersion", nvml_sys_driver_ver);
    (void)load_dl_symbol(lib.handle, "nvmlDeviceGetCurrPcieLinkGeneration", nvml_curr_pcie_gen);
    (void)load_dl_symbol(lib.handle, "nvmlDeviceGetCurrPcieLinkWidth", nvml_curr_pcie_width);
    (void)load_dl_symbol(lib.handle, "nvmlDeviceGetMaxPcieLinkGeneration", nvml_max_pcie_gen);
    (void)load_dl_symbol(lib.handle, "nvmlDeviceGetMaxPcieLinkWidth", nvml_max_pcie_width);

    if (nvml_init() != kNvmlSuccess) {
        return;
    }
    struct ScopedNvml {
        NvmlReturn (*shutdown)() = nullptr;
        ~ScopedNvml() {
            if (shutdown) {
                (void)shutdown();
            }
        }
        ScopedNvml(const ScopedNvml&) = delete;
        ScopedNvml& operator=(const ScopedNvml&) = delete;
        explicit ScopedNvml(NvmlReturn (*fn)()) noexcept : shutdown(fn) {}
    } nvml_guard(nvml_shutdown);

    std::string nvml_driver_build;
    if (nvml_sys_driver_ver) {
        char drv_buf[96] = {};
        if (nvml_sys_driver_ver(drv_buf, static_cast<unsigned int>(sizeof(drv_buf))) == kNvmlSuccess) {
            drv_buf[sizeof(drv_buf) - 1] = '\0';
            nvml_driver_build = sanitize_driver_build_string(drv_buf);
        }
    }

    unsigned int count = 0;
    if (nvml_get_count(&count) != kNvmlSuccess || count == 0) {
        return;
    }
    count = std::min<unsigned int>(count, 64u);
    for (unsigned int i = 0; i < count; ++i) {
        NvmlDevice device = nullptr;
        if (nvml_get_handle(i, &device) != kNvmlSuccess || !device) {
            continue;
        }
        GpuDeviceInfo info;
        info.id = "gpu:" + std::to_string(gpus.size());
        char name_buf[128] = {};
        if (nvml_get_name(device, name_buf, static_cast<unsigned int>(sizeof(name_buf))) == kNvmlSuccess) {
            name_buf[sizeof(name_buf) - 1] = '\0';
            info.name = trim(name_buf);
        }
        if (info.name.empty()) {
            info.name = "NVIDIA GPU " + std::to_string(i);
        }
        info.vendor = GpuVendor::nvidia;
        info.vendor_id = 0x10DEu;
        if (nvml_get_pci) {
            NvmlPciInfo pci{};
            if (nvml_get_pci(device, &pci) == kNvmlSuccess) {
                info.device_id = static_cast<std::uint32_t>(pci.pci_device_id >> 16);
                pci.bus_id[sizeof(pci.bus_id) - 1] = '\0';
                info.pci_bus_id = trim(pci.bus_id);
            }
        }
        if (!nvml_driver_build.empty()) {
            info.driver_name = "NVIDIA";
            info.driver_build = nvml_driver_build;
        }
        unsigned int pcie_val = 0;
        if (nvml_curr_pcie_gen && nvml_curr_pcie_gen(device, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 6u) {
            info.pcie.current_gen = pcie_val;
        }
        if (nvml_curr_pcie_width && nvml_curr_pcie_width(device, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 32u) {
            info.pcie.current_width = pcie_val;
        }
        if (nvml_max_pcie_gen && nvml_max_pcie_gen(device, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 6u) {
            info.pcie.max_gen = pcie_val;
        }
        if (nvml_max_pcie_width && nvml_max_pcie_width(device, &pcie_val) == kNvmlSuccess && pcie_val >= 1u &&
            pcie_val <= 32u) {
            info.pcie.max_width = pcie_val;
        }
        NvmlMemory mem{};
        if (nvml_get_memory(device, &mem) == kNvmlSuccess) {
            info.dedicated_total = MemoryReading::known(static_cast<std::uint64_t>(mem.total));
            // Older WSL2 WDDM-paravirt drivers report total > 0 with free=0, used=0;
            // keep dedicated_free unknown in that case so callers fall back to total.
            if (mem.total == 0 || mem.free > 0 || mem.used > 0) {
                info.dedicated_free = MemoryReading::known(static_cast<std::uint64_t>(mem.free));
            }
        }
        info.is_integrated_uma = false;
        info.backend_hint = "cuda";
        info.source = "nvml";
        gpus.push_back(std::move(info));
    }
}

std::vector<GpuDeviceInfo> live_gpu_devices() {
    std::vector<GpuDeviceInfo> gpus;
    append_amdgpu_sysfs_devices(gpus);
    append_nvml_gpu_devices(gpus);
    enrich_gpus_with_vulkan(gpus, {});
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        gpus[i].id = "gpu:" + std::to_string(i);
    }
    return gpus;
}
#  else
std::vector<GpuDeviceInfo> live_gpu_devices() { return {}; }
#  endif
#endif

}  // namespace

std::optional<std::uint64_t> parse_strict_u64(std::string_view text) noexcept {
    if (!text.empty() && text.back() == '\n') {
        text.remove_suffix(1);
    }
    if (text.empty() || text.size() > 20 || (text.size() > 1 && text.front() == '0')) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint64_t> parse_meminfo_bytes(std::string_view meminfo, std::string_view key) noexcept {
    std::optional<std::uint64_t> result;
    std::size_t pos = 0;
    while (pos < meminfo.size()) {
        const auto end = meminfo.find('\n', pos);
        const std::string_view line = meminfo.substr(pos, end == std::string_view::npos ? end : end - pos);
        pos = end == std::string_view::npos ? meminfo.size() : end + 1;
        if (line.size() <= key.size() || line.substr(0, key.size()) != key || line[key.size()] != ':') {
            continue;
        }
        if (result) {
            return std::nullopt;  // duplicated key: do not guess which one is right
        }
        std::string_view rest = line.substr(key.size() + 1);
        const auto first = rest.find_first_not_of(' ');
        if (first == std::string_view::npos || first == 0) {
            return std::nullopt;
        }
        rest.remove_prefix(first);
        if (rest.size() < 4 || rest.substr(rest.size() - 3) != " kB") {
            return std::nullopt;
        }
        const auto kib = parse_strict_u64(rest.substr(0, rest.size() - 3));
        if (!kib || *kib > std::numeric_limits<std::uint64_t>::max() / 1024) {
            return std::nullopt;
        }
        result = *kib * 1024;
    }
    return result;
}

CgroupHeadroom cgroup_v2_headroom(std::string_view proc_cgroup, const FileReader& read_file,
                                  std::string_view cgroup_root) {
    CgroupHeadroom out;
    std::optional<std::string_view> entry;
    std::size_t pos = 0;
    while (pos < proc_cgroup.size()) {
        const auto end = proc_cgroup.find('\n', pos);
        const std::string_view line = proc_cgroup.substr(pos, end == std::string_view::npos ? end : end - pos);
        pos = end == std::string_view::npos ? proc_cgroup.size() : end + 1;
        if (line.substr(0, 3) == "0::") {
            if (entry) {
                out.malformed = true;
                out.note = "/proc/self/cgroup has more than one cgroup v2 entry";
                return out;
            }
            entry = line.substr(3);
        }
    }
    if (!entry) {
        out.note = "no cgroup v2 entry";
        return out;
    }
    // "." / ".." / empty components are refused so the walk cannot leave the
    // cgroup mount.
    if (entry->empty() || entry->front() != '/') {
        out.malformed = true;
        out.note = "cgroup path is not absolute";
        return out;
    }
    std::vector<std::string> components;
    for (std::size_t i = 1; i < entry->size();) {
        auto next = entry->find('/', i);
        if (next == std::string_view::npos) {
            next = entry->size();
        }
        const std::string_view part = entry->substr(i, next - i);
        if (part.empty() || part == "." || part == ".." || part.find('\0') != std::string_view::npos ||
            components.size() >= 64) {
            out.malformed = true;
            out.note = "cgroup path is not canonical";
            return out;
        }
        components.emplace_back(part);
        i = next + 1;
    }
    // depth 0 is the namespace-root cgroup itself (a container's own limit when
    // /proc/self/cgroup reads "0::/"); the host's real root has no memory.max.
    for (std::size_t depth = components.size() + 1; depth-- > 0;) {
        std::string dir(cgroup_root);
        for (std::size_t i = 0; i < depth; ++i) {
            dir += '/';
            dir += components[i];
        }
        const auto max_text = read_file(dir + "/memory.max");
        const auto cur_text = read_file(dir + "/memory.current");
        if (!max_text && !cur_text) {
            continue;  // memory controller not enabled at this level
        }
        const auto fail = [&](const char* what) {
            out.malformed = true;
            out.limited = false;
            out.headroom = MemoryReading::unknown();
            out.note = std::string(what) + " at cgroup level " + std::to_string(depth);
            return out;
        };
        if (!max_text || !cur_text) {
            return fail("memory.max or memory.current missing its pair");
        }
        if (max_text->size() > 32 || cur_text->size() > 32) {
            return fail("oversized memory.max or memory.current");
        }
        const auto current = parse_strict_u64(*cur_text);
        if (!current) {
            return fail("malformed memory.current");
        }
        if (*max_text == "max" || *max_text == "max\n") {
            continue;  // no limit at this level
        }
        const auto limit = parse_strict_u64(*max_text);
        if (!limit) {
            return fail("malformed memory.max");
        }
        const std::uint64_t room = *limit > *current ? *limit - *current : 0;
        if (!out.limited || room < out.headroom.bytes_or(0)) {
            out.headroom = MemoryReading::known(room);
        }
        out.limited = true;
    }
    if (!out.limited) {
        out.note = "no finite cgroup v2 memory limit";
    }
    return out;
}

HostMemory compose_linux_host_memory(std::string_view meminfo, const CgroupHeadroom& cgroup) {
    HostMemory out;
    out.source = cgroup.limited ? "meminfo+cgroup2" : "meminfo";
    if (const auto total = parse_meminfo_bytes(meminfo, "MemTotal")) {
        out.total_physical = MemoryReading::known(*total);
    }
    if (cgroup.malformed) {
        // A limit we cannot read could be tighter than MemAvailable: do not
        // report a number that may be too optimistic.
        out.note = "cgroup v2 limit unreadable: " + cgroup.note;
        return out;
    }
    const auto available = parse_meminfo_bytes(meminfo, "MemAvailable");
    if (available) {
        out.available_physical = MemoryReading::known(*available);
    } else {
        out.note = "MemAvailable missing or malformed in /proc/meminfo";
    }
    if (cgroup.limited && cgroup.headroom.is_known()) {
        const std::uint64_t room = cgroup.headroom.bytes_or(0);
        if (!available || room < *available) {
            out.available_physical = MemoryReading::known(room);
            out.note = "capped by cgroup v2 headroom";
        }
    }
    return out;
}

HostMemory read_host_memory() { return live_host_memory(); }

std::vector<DeviceInfo> enumerate_devices() {
    DeviceInfo cpu;
    cpu.id = "cpu:0";
    cpu.kind = DeviceKind::cpu;
    cpu.name = cpu_name();
    cpu.logical_cores = std::thread::hardware_concurrency();
    const HostMemory host = read_host_memory();
    cpu.total_memory_bytes = host.total_physical.bytes_or(0);
    cpu.available_memory_bytes = host.available_physical.bytes_or(0);
    cpu.available_physical = host.available_physical;
    cpu.available_commit = host.available_commit;
    return {cpu};
}

std::optional<std::size_t> map_rocm_device_ordinal(std::size_t amd_ordinal,
                                                   std::optional<std::string_view> hip_visible_devices,
                                                   std::optional<std::string_view> rocr_visible_devices,
                                                   std::optional<std::string_view> device_uuid) noexcept {
    std::size_t current = amd_ordinal;
    if (rocr_visible_devices.has_value()) {
        const auto mapped = map_rocm_visible_ordinal(current, *rocr_visible_devices, device_uuid);
        if (!mapped) {
            return std::nullopt;
        }
        current = *mapped;
    }
    if (hip_visible_devices.has_value()) {
        const auto mapped = map_rocm_visible_ordinal(current, *hip_visible_devices, device_uuid);
        if (!mapped) {
            return std::nullopt;
        }
        current = *mapped;
    }
    return current;
}

bool is_rocm_device_visible(std::size_t amd_ordinal,
                            std::optional<std::string_view> hip_visible_devices,
                            std::optional<std::string_view> rocr_visible_devices,
                            std::optional<std::string_view> device_uuid) noexcept {
    return map_rocm_device_ordinal(amd_ordinal, hip_visible_devices, rocr_visible_devices, device_uuid).has_value();
}

void merge_win32_nvml_devices(std::vector<GpuDeviceInfo>& gpus, const std::vector<GpuDeviceInfo>& nvml_gpus) {
    if (nvml_gpus.empty()) {
        return;
    }
    // Track which existing DXGI NVIDIA entries have already been matched to a
    // non-TCC NVML entry so identical dual-GPU setups are not collapsed while
    // WDDM cards already reported by DXGI are never duplicated.
    std::vector<bool> dxgi_matched(gpus.size(), false);
    for (const auto& nv : nvml_gpus) {
        if (nv.is_tcc_mode) {
            // TCC-mode adapters are never enumerated by DXGI; always append.
            gpus.push_back(nv);
            dxgi_matched.push_back(true);
            continue;
        }
        bool matched = false;
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            if (dxgi_matched[i] || gpus[i].vendor != GpuVendor::nvidia) {
                continue;
            }
            const bool same_bus = !nv.pci_bus_id.empty() && !gpus[i].pci_bus_id.empty() &&
                                  normalize_pci_bus_id(nv.pci_bus_id) == normalize_pci_bus_id(gpus[i].pci_bus_id);
            const bool same_dev = (nv.device_id != 0 && gpus[i].device_id == nv.device_id);
            const bool same_name = !nv.name.empty() && iequals(nv.name, gpus[i].name);
            if (same_bus || same_dev || same_name) {
                dxgi_matched[i] = true;
                if (gpus[i].pci_bus_id.empty() && !nv.pci_bus_id.empty()) {
                    gpus[i].pci_bus_id = nv.pci_bus_id;
                }
                if (gpus[i].driver_name.empty() && !nv.driver_name.empty()) {
                    gpus[i].driver_name = nv.driver_name;
                }
                if (gpus[i].driver_build.empty() && !nv.driver_build.empty()) {
                    gpus[i].driver_build = nv.driver_build;
                }
                if (!gpus[i].pcie.current_gen && nv.pcie.current_gen) {
                    gpus[i].pcie.current_gen = nv.pcie.current_gen;
                }
                if (!gpus[i].pcie.current_width && nv.pcie.current_width) {
                    gpus[i].pcie.current_width = nv.pcie.current_width;
                }
                if (!gpus[i].pcie.max_gen && nv.pcie.max_gen) {
                    gpus[i].pcie.max_gen = nv.pcie.max_gen;
                }
                if (!gpus[i].pcie.max_width && nv.pcie.max_width) {
                    gpus[i].pcie.max_width = nv.pcie.max_width;
                }
                matched = true;
                break;
            }
        }
        if (!matched) {
            gpus.push_back(nv);
            dxgi_matched.push_back(true);
        }
    }
}

std::string resolve_windows_gpu_backend_hint(GpuVendor vendor, bool is_integrated_uma, bool hip_sdk_available,
                                             std::size_t amd_ordinal,
                                             std::optional<std::string_view> hip_visible_devices,
                                             std::optional<std::string_view> rocr_visible_devices) {
    switch (vendor) {
        case GpuVendor::nvidia: return "cuda";
        case GpuVendor::amd:
            if (!is_integrated_uma && hip_sdk_available &&
                is_rocm_device_visible(amd_ordinal, hip_visible_devices, rocr_visible_devices)) {
                return "rocm";
            }
            return "vulkan";
        case GpuVendor::intel: return "vulkan";
        case GpuVendor::apple: return "metal";
        case GpuVendor::unknown: return {};
    }
    return {};
}

std::optional<GpuDeviceInfo> parse_amdgpu_sysfs_device(std::string_view card_id, const FileReader& read_file) {
    return parse_amdgpu_sysfs_device(card_id, read_file, true);
}

std::optional<GpuDeviceInfo> parse_amdgpu_sysfs_device(std::string_view card_id, const FileReader& read_file,
                                                       bool rocm_available) {
    if (card_id.empty() || !read_file) {
        return std::nullopt;
    }
    std::string dev_dir;
    std::string_view card_component;
    if (card_id.front() == '/') {
        while (card_id.size() > 1 && card_id.back() == '/') {
            card_id.remove_suffix(1);
        }
        dev_dir = std::string(card_id);
        std::string_view base = card_id;
        if (base.size() > 7 && base.substr(base.size() - 7) == "/device") {
            base.remove_suffix(7);
        }
        const auto slash = base.rfind('/');
        card_component = slash == std::string_view::npos ? base : base.substr(slash + 1);
    } else {
        if (card_id == "." || card_id == ".." || card_id.find('/') != std::string_view::npos ||
            card_id.find('\\') != std::string_view::npos || card_id.find('\0') != std::string_view::npos) {
            return std::nullopt;
        }
        card_component = card_id;
        dev_dir = "/sys/class/drm/" + std::string(card_id) + "/device";
    }

    const auto vendor_text = read_file(dev_dir + "/vendor");
    if (!vendor_text) {
        return std::nullopt;
    }
    const auto vendor_id = parse_hex_u32(*vendor_text);
    if (!vendor_id || gpu_vendor_from_pci_id(*vendor_id) != GpuVendor::amd) {
        return std::nullopt;
    }

    std::uint32_t uevent_device_id = 0;
    std::string uevent_pci_slot;
    if (const auto uevent = read_file(dev_dir + "/uevent")) {
        std::size_t pos = 0;
        const std::string_view text(*uevent);
        while (pos < text.size()) {
            const auto end = text.find('\n', pos);
            const std::string_view line =
                trim_view(text.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos));
            pos = end == std::string_view::npos ? text.size() : end + 1;
            if (line.size() > 7 && line.substr(0, 7) == "DRIVER=") {
                if (trim_view(line.substr(7)) != "amdgpu") {
                    return std::nullopt;
                }
            } else if (line.size() > 7 && line.substr(0, 7) == "PCI_ID=") {
                const std::string_view pci = trim_view(line.substr(7));
                const auto colon = pci.find(':');
                if (colon != std::string_view::npos) {
                    if (const auto parsed = parse_hex_u32(pci.substr(colon + 1))) {
                        uevent_device_id = *parsed;
                    }
                }
            } else if (line.size() > 14 && line.substr(0, 14) == "PCI_SLOT_NAME=") {
                uevent_pci_slot = sanitize_driver_build_string(line.substr(14));
            }
        }
    }

    std::uint64_t vram_total_bytes = 0;
    std::uint64_t vram_used_bytes = 0;
    std::uint64_t gtt_total_bytes = 0;
    std::uint64_t gtt_used_bytes = 0;
    bool has_vram_total = false;
    bool has_vram_used = false;
    bool has_gtt_total = false;
    bool has_gtt_used = false;
    const auto read_u64 = [&](const std::string& path, std::uint64_t& out, bool& present) {
        if (const auto text = read_file(path)) {
            if (const auto parsed = parse_sysfs_u64(*text)) {
                out = *parsed;
                present = true;
            }
        }
    };
    read_u64(dev_dir + "/mem_info_vram_total", vram_total_bytes, has_vram_total);
    read_u64(dev_dir + "/mem_info_vram_used", vram_used_bytes, has_vram_used);
    read_u64(dev_dir + "/mem_info_gtt_total", gtt_total_bytes, has_gtt_total);
    read_u64(dev_dir + "/mem_info_gtt_used", gtt_used_bytes, has_gtt_used);
    if (!has_vram_total && !has_gtt_total) {
        return std::nullopt;
    }

    GpuDeviceInfo info;
    const auto card_idx = parse_card_index(card_component);
    info.id = "gpu:" + std::to_string(card_idx.value_or(0));
    info.vendor = GpuVendor::amd;
    info.vendor_id = *vendor_id;
    if (const auto device_text = read_file(dev_dir + "/device")) {
        if (const auto parsed = parse_hex_u32(*device_text)) {
            info.device_id = *parsed;
        }
    }
    if (info.device_id == 0) {
        info.device_id = uevent_device_id;
    }

    if (const auto product_name = read_file(dev_dir + "/product_name")) {
        info.name = trim(*product_name);
    }
    if (info.name.empty()) {
        info.name = info.device_id != 0 ? "AMD Radeon (" + hex_u32_string(info.device_id) + ")" : "AMD GPU";
    }

    if (has_vram_total) {
        info.dedicated_total = MemoryReading::known(vram_total_bytes);
    }
    if (has_gtt_total) {
        info.shared_total = MemoryReading::known(gtt_total_bytes);
    }
    info.is_integrated_uma =
        is_integrated_uma_memory(GpuVendor::amd, info.dedicated_total.bytes_or(0), info.shared_total.bytes_or(0));

    if (!info.is_integrated_uma) {
        if (has_vram_total && has_vram_used) {
            info.dedicated_free =
                MemoryReading::known(vram_total_bytes > vram_used_bytes ? vram_total_bytes - vram_used_bytes : 0);
        }
    } else if (has_gtt_total && has_gtt_used) {
        const std::uint64_t gtt_free = gtt_total_bytes > gtt_used_bytes ? gtt_total_bytes - gtt_used_bytes : 0;
        std::uint64_t vram_free = 0;
        if (has_vram_total) {
            vram_free = has_vram_used ? (vram_total_bytes > vram_used_bytes ? vram_total_bytes - vram_used_bytes : 0)
                                      : vram_total_bytes;
        }
        info.dedicated_free = MemoryReading::known(saturating_add(vram_free, gtt_free));
    }

    if (!uevent_pci_slot.empty()) {
        info.pci_bus_id = std::move(uevent_pci_slot);
    }
    info.driver_name = "amdgpu";
    // /sys/module/amdgpu/version is populated by DKMS amdgpu builds; in-tree
    // mainline kernels omit it, leaving driver_build empty so Vulkan driverInfo
    // fills it via enrich_gpus_with_vulkan().
    if (const auto drv_ver = read_file("/sys/module/amdgpu/version")) {
        info.driver_build = sanitize_driver_build_string(*drv_ver);
    }
    if (const auto speed = read_file(dev_dir + "/current_link_speed")) {
        info.pcie.current_gen = parse_pcie_link_speed_gen(*speed);
    }
    if (const auto width = read_file(dev_dir + "/current_link_width")) {
        if (const auto w = parse_sysfs_u64(*width); w && *w >= 1u && *w <= 32u) {
            info.pcie.current_width = static_cast<std::uint32_t>(*w);
        }
    }
    if (const auto speed = read_file(dev_dir + "/max_link_speed")) {
        info.pcie.max_gen = parse_pcie_link_speed_gen(*speed);
    }
    if (const auto width = read_file(dev_dir + "/max_link_width")) {
        if (const auto w = parse_sysfs_u64(*width); w && *w >= 1u && *w <= 32u) {
            info.pcie.max_width = static_cast<std::uint32_t>(*w);
        }
    }
    auto read_sysfs_bool = [&](const std::string& path) -> std::optional<bool> {
        if (const auto text = read_file(path)) {
            const std::string_view v = trim_view(*text);
            if (v == "0") return false;
            if (v == "1") return true;
        }
        return std::nullopt;
    };
    const std::string link_aspm = format_linux_link_aspm(
        read_sysfs_bool(dev_dir + "/link/l0s_aspm"), read_sysfs_bool(dev_dir + "/link/l1_aspm"),
        read_sysfs_bool(dev_dir + "/link/l1_1_aspm"), read_sysfs_bool(dev_dir + "/link/l1_2_aspm"));
    if (!link_aspm.empty()) {
        info.pcie.aspm_state = link_aspm;
        info.pcie.aspm_source = "linux_sysfs_link";
    } else if (const auto policy = read_file("/sys/module/pcie_aspm/parameters/policy")) {
        const std::string pol = parse_linux_aspm_policy(*policy);
        if (!pol.empty()) {
            info.pcie.aspm_state = pol;
            info.pcie.aspm_source = "linux_pcie_aspm_policy";
        }
    }

    info.backend_hint = rocm_available ? "rocm" : "vulkan";
    info.source = "amdgpu-sysfs";
    return info;
}

std::vector<GpuDeviceInfo> enumerate_gpu_devices() {
    static std::mutex probe_mutex;
    std::lock_guard<std::mutex> lock(probe_mutex);
    std::vector<GpuDeviceInfo> gpus = live_gpu_devices();
    const auto ggml_devices = registered_ggml_device_telemetry();
    if (!ggml_devices.empty()) {
        merge_ggml_device_telemetry(gpus, ggml_devices);
    }
    return gpus;
}

std::optional<std::uint64_t> gpu_effective_vram_mib(const GpuDeviceInfo& gpu, bool prefer_free) noexcept {
    if (!gpu.is_integrated_uma) {
        if (prefer_free && gpu.dedicated_free.is_known()) {
            return gpu.dedicated_free.bytes_or(0) / kMiB;
        }
        if (gpu.dedicated_total.is_known()) {
            const std::uint64_t bytes = gpu.dedicated_total.bytes_or(0);
            if (bytes == 0) {
                return std::nullopt;
            }
            return bytes / kMiB;
        }
        return std::nullopt;
    }

    const bool has_pool = gpu.dedicated_total.is_known() || gpu.shared_total.is_known();
    const std::uint64_t total_pool =
        has_pool ? saturating_add(gpu.dedicated_total.bytes_or(0), gpu.shared_total.bytes_or(0)) : 0;

    if (prefer_free && gpu.dedicated_free.is_known()) {
        const std::uint64_t free_bytes = gpu.dedicated_free.bytes_or(0);
        const std::uint64_t capped = total_pool > 0 ? std::min(free_bytes, total_pool) : free_bytes;
        return capped / kMiB;
    }
    if (!has_pool || total_pool == 0) {
        return std::nullopt;
    }
    return total_pool / kMiB;
}

std::optional<std::uint64_t> select_gpu_vram_budget_mib(const std::vector<GpuDeviceInfo>& gpus,
                                                        std::string_view device_hint,
                                                        bool prefer_free) noexcept {
    if (gpus.empty()) {
        return std::nullopt;
    }
    const std::string_view trimmed = trim_view(device_hint);
    if (trimmed.empty()) {
        const auto best = select_best_gpu(gpus, [](const GpuDeviceInfo&) noexcept { return true; }, prefer_free);
        if (!best) {
            return std::nullopt;
        }
        return gpu_effective_vram_mib(gpus[*best], prefer_free);
    }

    bool seen[64] = {};
    bool matched_any = false;
    std::uint64_t total_mib = 0;
    std::size_t pos = 0;
    while (pos <= trimmed.size()) {
        const auto comma = trimmed.find(',', pos);
        const std::string_view token =
            trim_view(trimmed.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? trimmed.size() + 1 : comma + 1;
        if (token.empty()) {
            continue;
        }
        const auto idx = match_single_gpu_hint_impl(gpus, token, prefer_free, /*require_known_vram=*/true);
        if (!idx) {
            return std::nullopt;
        }
        const auto mib = gpu_effective_vram_mib(gpus[*idx], prefer_free);
        if (!mib) {
            return std::nullopt;
        }
        if (*idx >= 64 || !seen[*idx]) {
            if (*idx < 64) {
                seen[*idx] = true;
            }
            total_mib = saturating_add(total_mib, *mib);
        }
        matched_any = true;
    }
    return matched_any ? std::optional<std::uint64_t>(total_mib) : std::nullopt;
}

std::optional<std::size_t> match_single_gpu_hint(const std::vector<GpuDeviceInfo>& gpus,
                                                 std::string_view token,
                                                 bool prefer_free,
                                                 bool require_known_vram) noexcept {
    return match_single_gpu_hint_impl(gpus, token, prefer_free, require_known_vram);
}

std::optional<std::string_view> normalize_uma_mode(std::string_view mode) noexcept {
    if (mode == "on" || mode == "true") {
        return std::string_view{"on"};
    }
    if (mode == "off" || mode == "false") {
        return std::string_view{"off"};
    }
    if (mode == "auto") {
        return std::string_view{"auto"};
    }
    return std::nullopt;
}

bool is_valid_uma_mode(std::string_view mode) noexcept {
    return normalize_uma_mode(mode).has_value();
}

std::string_view extract_device_hint(const std::vector<std::string>& args) noexcept {
    std::string_view result;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--device" || arg == "-dev") {
            if (i + 1 < args.size()) {
                result = args[++i];
            } else {
                result = {};
            }
        } else if (arg.size() > 9 && arg.compare(0, 9, "--device=") == 0) {
            result = arg.substr(9);
        } else if (arg.size() > 5 && arg.compare(0, 5, "-dev=") == 0) {
            result = arg.substr(5);
        } else if (arg == "--device=" || arg == "-dev=") {
            result = {};
        }
    }
    return result;
}

UmaSelectionMode resolve_uma_selection(std::string_view mode_or_auto,
                                       const std::vector<GpuDeviceInfo>& gpus,
                                       std::string_view device_hint) noexcept {
    if (mode_or_auto == "on" || mode_or_auto == "true") {
        return UmaSelectionMode::uma_only;
    }
    if (mode_or_auto != "auto") {
        return UmaSelectionMode::discrete_only;
    }
    if (gpus.empty()) {
        return UmaSelectionMode::discrete_only;
    }

    const std::string_view trimmed = trim_view(device_hint);
    if (trimmed.empty()) {
        // Discrete GPU wins whenever present; return uma_only only when all
        // discovered GPUs are integrated UMA (e.g. Ryzen 780M/890M APU-only).
        for (const auto& gpu : gpus) {
            if (!gpu.is_integrated_uma) {
                return UmaSelectionMode::discrete_only;
            }
        }
        return UmaSelectionMode::uma_only;
    }

    bool matched_any = false;
    bool any_uma = false;
    bool any_discrete = false;
    std::size_t pos = 0;
    while (pos <= trimmed.size()) {
        const auto comma = trimmed.find(',', pos);
        const std::string_view token =
            trim_view(trimmed.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? trimmed.size() + 1 : comma + 1;
        if (token.empty()) {
            continue;
        }
        const auto idx = match_single_gpu_hint_impl(gpus, token, /*prefer_free=*/false, /*require_known_vram=*/false);
        if (!idx) {
            return UmaSelectionMode::discrete_only;
        }
        matched_any = true;
        if (gpus[*idx].is_integrated_uma) {
            any_uma = true;
        } else {
            any_discrete = true;
        }
    }
    if (!matched_any || !any_uma) {
        return UmaSelectionMode::discrete_only;
    }
    return any_discrete ? UmaSelectionMode::hybrid : UmaSelectionMode::uma_only;
}

bool resolve_uma_mode(std::string_view mode_or_auto,
                      const std::vector<GpuDeviceInfo>& gpus,
                      std::string_view device_hint) noexcept {
    return resolve_uma_selection(mode_or_auto, gpus, device_hint) != UmaSelectionMode::discrete_only;
}

namespace {

bool is_vulkan_pool_eligible(const GpuDeviceInfo& gpu) noexcept {
    if (gpu.is_tcc_mode) {
        return false;
    }
    if (gpu.vendor == GpuVendor::apple) {
        return false;
    }
    const auto eff = gpu_effective_vram_mib(gpu, false);
    return eff.has_value() && *eff > 0;
}

}  // namespace

GpuPoolSummary summarize_gpu_pools(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host) {
    GpuPoolSummary out;
    std::vector<std::string> discrete_vk;
    std::vector<std::string> uma_vk;

    for (std::size_t i = 0; i < gpus.size(); ++i) {
        const GpuDeviceInfo& g = gpus[i];
        const auto total_mib = gpu_effective_vram_mib(g, false);
        const auto free_mib = gpu_effective_vram_mib(g, true);
        const bool vk_ok = is_vulkan_pool_eligible(g);

        if (!g.is_integrated_uma) {
            ++out.discrete_gpu_count;
            if (total_mib) {
                out.discrete_vram_mib = saturating_add(out.discrete_vram_mib, *total_mib);
            }
            if (free_mib) {
                out.discrete_free_vram_mib = saturating_add(out.discrete_free_vram_mib, *free_mib);
            }
            if (vk_ok) {
                discrete_vk.push_back("Vulkan" + std::to_string(i));
            }
        } else {
            ++out.integrated_uma_gpu_count;
            if (total_mib) {
                out.integrated_uma_vram_mib = saturating_add(out.integrated_uma_vram_mib, *total_mib);
            }
            if (free_mib) {
                out.integrated_uma_free_vram_mib = saturating_add(out.integrated_uma_free_vram_mib, *free_mib);
            }
            if (vk_ok) {
                uma_vk.push_back("Vulkan" + std::to_string(i));
            }
        }
    }

    // All integrated UMA GPUs share the host's physical system RAM: cap their
    // combined contribution against host total/available physical RAM when known.
    if (host.total_physical.is_known()) {
        const std::uint64_t host_total_mib = host.total_physical.bytes_or(0) / kMiB;
        out.integrated_uma_vram_mib = std::min(out.integrated_uma_vram_mib, host_total_mib);
    }
    if (host.available_physical.is_known()) {
        const std::uint64_t host_avail_mib = host.available_physical.bytes_or(0) / kMiB;
        out.integrated_uma_free_vram_mib = std::min(out.integrated_uma_free_vram_mib, host_avail_mib);
    }
    if (out.integrated_uma_vram_mib > 0) {
        out.integrated_uma_free_vram_mib = std::min(out.integrated_uma_free_vram_mib, out.integrated_uma_vram_mib);
    } else {
        out.integrated_uma_free_vram_mib = 0;
    }

    out.combined_gpu_vram_mib = saturating_add(out.discrete_vram_mib, out.integrated_uma_vram_mib);
    out.combined_free_gpu_vram_mib = saturating_add(out.discrete_free_vram_mib, out.integrated_uma_free_vram_mib);

    if (!discrete_vk.empty() && !uma_vk.empty() && out.discrete_vram_mib > 0 && out.integrated_uma_vram_mib > 0) {
        out.hybrid_vulkan_pool_available = true;
        out.hybrid_vulkan_devices.reserve(discrete_vk.size() + uma_vk.size());
        for (auto& d : discrete_vk) {
            out.hybrid_vulkan_devices.push_back(std::move(d));
        }
        for (auto& u : uma_vk) {
            out.hybrid_vulkan_devices.push_back(std::move(u));
        }
    }
    return out;
}

std::uint64_t combined_gpu_vram_mib(const std::vector<GpuDeviceInfo>& gpus, bool prefer_free,
                                    const HostMemory& host) noexcept {
    std::uint64_t discrete_mib = 0;
    std::uint64_t uma_total_mib = 0;
    std::uint64_t uma_free_mib = 0;
    for (const auto& g : gpus) {
        if (!g.is_integrated_uma) {
            if (const auto mib = gpu_effective_vram_mib(g, prefer_free)) {
                discrete_mib = saturating_add(discrete_mib, *mib);
            }
        } else {
            if (const auto total = gpu_effective_vram_mib(g, false)) {
                uma_total_mib = saturating_add(uma_total_mib, *total);
            }
            if (const auto free = gpu_effective_vram_mib(g, true)) {
                uma_free_mib = saturating_add(uma_free_mib, *free);
            }
        }
    }
    if (host.total_physical.is_known()) {
        uma_total_mib = std::min(uma_total_mib, host.total_physical.bytes_or(0) / kMiB);
    }
    if (host.available_physical.is_known()) {
        uma_free_mib = std::min(uma_free_mib, host.available_physical.bytes_or(0) / kMiB);
    }
    if (uma_total_mib > 0) {
        uma_free_mib = std::min(uma_free_mib, uma_total_mib);
    } else {
        uma_free_mib = 0;
    }
    return saturating_add(discrete_mib, prefer_free ? uma_free_mib : uma_total_mib);
}

std::string host_platform() {
#if defined(_WIN32)
    std::string os = "windows";
#elif defined(__APPLE__)
    std::string os = "macos";
#elif defined(__linux__)
    std::string os = "linux";
#else
    std::string os = "unknown";
#endif
#if defined(_M_X64) || defined(__x86_64__)
    return os + "-x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return os + "-arm64";
#else
    return os + "-other";
#endif
}

std::string host_name() {
#if defined(_WIN32)
    char buffer[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD size = sizeof(buffer);
    if (GetComputerNameA(buffer, &size)) {
        return std::string(buffer, size);
    }
    return "unknown-host";
#else
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) == 0) {
        return buffer;
    }
    return "unknown-host";
#endif
}

namespace {

std::optional<bool> parse_bit_bool(std::string_view sv) noexcept {
    sv = trim_view(sv);
    if (sv == "0") return false;
    if (sv == "1") return true;
    return std::nullopt;
}

}  // namespace

std::string sanitize_driver_build_string(std::string_view raw) {
    constexpr std::size_t kMaxLen = 256u;
    std::string out;
    out.reserve(std::min(raw.size(), kMaxLen));
    for (const char c : raw) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20u || u == 0x7Fu) {
            continue;
        }
        out.push_back(c);
        if (out.size() >= kMaxLen) {
            break;
        }
    }
    return trim(std::move(out));
}

std::string format_driver_build_string(std::string_view driver_version, std::string_view release_version,
                                       std::string_view vulkan_driver_info) {
    const std::string drv = sanitize_driver_build_string(driver_version);
    const std::string rel = sanitize_driver_build_string(release_version);
    const std::string vk = sanitize_driver_build_string(vulkan_driver_info);
    if (!rel.empty() && !drv.empty()) {
        if (rel.find(drv) != std::string::npos) return rel;
        if (drv.find(rel) != std::string::npos) return drv;
        return sanitize_driver_build_string(rel + " (" + drv + ")");
    }
    if (!drv.empty()) return drv;
    if (!rel.empty()) return rel;
    return vk;
}

std::optional<GgmlDeviceTelemetry> parse_ggml_device_line(std::string_view line) {
    if (line.empty() || line.size() > 4096u || line.find('\0') != std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view trimmed = trim_view(line);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    constexpr std::string_view kVkPrefix = "ggml_vulkan:";
    const std::size_t prefix_pos = trimmed.find(kVkPrefix);
    if (prefix_pos == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view body = trim_view(trimmed.substr(prefix_pos + kVkPrefix.size()));
    if (body.empty()) {
        return std::nullopt;
    }

    // Split on '|': segment 0 is `<index> = <device name> (<driver name>)`,
    // followed by one or more `key: value` capability fields.
    std::vector<std::string_view> parts;
    std::size_t pos = 0;
    while (pos <= body.size()) {
        const std::size_t bar = body.find('|', pos);
        const std::string_view seg =
            trim_view(body.substr(pos, bar == std::string_view::npos ? std::string_view::npos : bar - pos));
        if (!seg.empty()) {
            parts.push_back(seg);
        }
        if (bar == std::string_view::npos) break;
        pos = bar + 1;
    }
    if (parts.size() < 2u) {
        return std::nullopt;
    }

    const std::string_view head = parts[0];
    const std::size_t eq = head.find('=');
    if (eq == std::string_view::npos) {
        return std::nullopt;
    }
    const auto idx = parse_strict_u64(trim_view(head.substr(0, eq)));
    if (!idx || *idx > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    std::string_view name_and_driver = trim_view(head.substr(eq + 1));
    if (name_and_driver.empty()) {
        return std::nullopt;
    }

    GgmlDeviceTelemetry out;
    out.backend = "vulkan";
    out.index = static_cast<std::uint32_t>(*idx);

    // Split trailing ` (<driver>)` using the last ` (` so names with embedded
    // parentheticals like `AMD Radeon(TM) 780M Graphics (AMD proprietary driver)`
    // keep `(TM)` intact.
    if (name_and_driver.back() == ')') {
        const std::size_t open = name_and_driver.rfind(" (");
        if (open != std::string_view::npos && open > 0) {
            const std::string_view dev_part = trim_view(name_and_driver.substr(0, open));
            const std::string_view drv_part =
                trim_view(name_and_driver.substr(open + 2, name_and_driver.size() - open - 3));
            if (!dev_part.empty()) {
                out.device_name = sanitize_driver_build_string(dev_part);
                out.driver_name = sanitize_driver_build_string(drv_part);
            }
        }
    }
    if (out.device_name.empty()) {
        out.device_name = sanitize_driver_build_string(name_and_driver);
    }
    if (out.device_name.empty()) {
        return std::nullopt;
    }

    bool any_recognized = false;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::size_t colon = parts[i].find(':');
        if (colon == std::string_view::npos) {
            return std::nullopt;
        }
        const std::string_view key = trim_view(parts[i].substr(0, colon));
        const std::string_view val = trim_view(parts[i].substr(colon + 1));
        if (key.empty() || val.empty()) {
            return std::nullopt;
        }
        if (iequals(key, "uma")) {
            const auto b = parse_bit_bool(val);
            if (!b) return std::nullopt;
            out.uma = *b;
            any_recognized = true;
        } else if (iequals(key, "rebar")) {
            const auto b = parse_bit_bool(val);
            if (!b) return std::nullopt;
            out.rebar = *b;
            any_recognized = true;
        } else if (iequals(key, "fp16")) {
            out.fp16 = sanitize_driver_build_string(val);
            if (out.fp16.empty()) return std::nullopt;
            any_recognized = true;
        } else if (iequals(key, "bf16")) {
            const auto b = parse_bit_bool(val);
            if (!b) return std::nullopt;
            out.bf16 = *b;
            any_recognized = true;
        } else if (iequals(key, "fp4")) {
            const auto b = parse_bit_bool(val);
            if (!b) return std::nullopt;
            out.fp4 = *b;
            any_recognized = true;
        } else if (iequals(key, "warp size")) {
            const auto w = parse_strict_u64(val);
            if (!w || *w == 0 || *w > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
            out.warp_size = static_cast<std::uint32_t>(*w);
            any_recognized = true;
        } else if (iequals(key, "shared memory")) {
            const auto s = parse_strict_u64(val);
            if (!s) return std::nullopt;
            out.shared_memory_bytes = *s;
            any_recognized = true;
        } else if (iequals(key, "int dot")) {
            const auto b = parse_bit_bool(val);
            if (!b) return std::nullopt;
            out.int_dot = *b;
            any_recognized = true;
        } else if (iequals(key, "matrix cores")) {
            out.matrix_cores = sanitize_driver_build_string(val);
            if (out.matrix_cores.empty()) return std::nullopt;
            any_recognized = true;
        }
    }
    if (!any_recognized) {
        return std::nullopt;
    }
    out.raw_line = std::string(trimmed.substr(0, std::min<std::size_t>(trimmed.size(), 1024u)));
    return out;
}

void apply_ggml_device_telemetry(GpuDeviceInfo& gpu, const GgmlDeviceTelemetry& ggml) {
    gpu.ggml_device = ggml;
    if (!ggml.driver_name.empty()) {
        gpu.driver_name = ggml.driver_name;
    }
    if (ggml.rebar.has_value()) {
        gpu.rebar = ggml.rebar;
    }
    if (ggml.shared_memory_bytes.has_value()) {
        gpu.shared_memory_bytes = ggml.shared_memory_bytes;
    }
    if (ggml.warp_size.has_value()) {
        gpu.warp_size = ggml.warp_size;
    }
    if (!ggml.matrix_cores.empty()) {
        gpu.matrix_cores = ggml.matrix_cores;
    }
}

namespace {

struct CoreGgmlDeviceRegistry {
    static constexpr std::size_t kMaxEntries = 64u;
    std::mutex mu;
    std::vector<GgmlDeviceTelemetry> entries;
};

CoreGgmlDeviceRegistry& core_ggml_registry() {
    static CoreGgmlDeviceRegistry reg;
    return reg;
}

}  // namespace

void register_ggml_device_telemetry(const GgmlDeviceTelemetry& entry) {
    if (entry.backend.empty() || entry.device_name.empty()) {
        return;
    }
    auto& reg = core_ggml_registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    for (auto& existing : reg.entries) {
        if (existing.backend == entry.backend && existing.index == entry.index &&
            existing.device_name == entry.device_name) {
            existing = entry;
            return;
        }
    }
    if (reg.entries.size() < CoreGgmlDeviceRegistry::kMaxEntries) {
        reg.entries.push_back(entry);
    }
}

std::vector<GgmlDeviceTelemetry> registered_ggml_device_telemetry() {
    auto& reg = core_ggml_registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    return reg.entries;
}

void clear_registered_ggml_device_telemetry() noexcept {
    auto& reg = core_ggml_registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.entries.clear();
}

void merge_ggml_device_telemetry(std::vector<GpuDeviceInfo>& gpus,
                                 const std::vector<GgmlDeviceTelemetry>& ggml_devices) {
    if (gpus.empty() || ggml_devices.empty()) {
        return;
    }
    std::vector<bool> matched_gpu(gpus.size(), false);
    for (const auto& ggml : ggml_devices) {
        std::optional<std::size_t> target;
        // 1. Exact case-insensitive device name match.
        if (!ggml.device_name.empty()) {
            for (std::size_t i = 0; i < gpus.size(); ++i) {
                if (!matched_gpu[i] && iequals(gpus[i].name, ggml.device_name)) {
                    target = i;
                    break;
                }
            }
        }
        // 2. Substring device name match (e.g. "AMD Radeon RX 9060 XT" vs
        //    "AMD Radeon RX 9060 XT (RADV ...)" or trimmed suffix).
        if (!target && !ggml.device_name.empty()) {
            std::string ggml_lower;
            ggml_lower.reserve(ggml.device_name.size());
            for (const char c : ggml.device_name) ggml_lower.push_back(ascii_tolower(c));
            for (std::size_t i = 0; i < gpus.size(); ++i) {
                if (matched_gpu[i] || gpus[i].name.empty()) continue;
                std::string gpu_lower;
                gpu_lower.reserve(gpus[i].name.size());
                for (const char c : gpus[i].name) gpu_lower.push_back(ascii_tolower(c));
                if (gpu_lower.find(ggml_lower) != std::string::npos ||
                    ggml_lower.find(gpu_lower) != std::string::npos) {
                    target = i;
                    break;
                }
            }
        }
        // 3. Fallback to ggml device index when UMA flag agrees (or unset).
        if (!target && ggml.index.has_value() && *ggml.index < gpus.size() && !matched_gpu[*ggml.index]) {
            if (!ggml.uma.has_value() || *ggml.uma == gpus[*ggml.index].is_integrated_uma) {
                target = *ggml.index;
            }
        }
        if (target) {
            matched_gpu[*target] = true;
            apply_ggml_device_telemetry(gpus[*target], ggml);
        }
    }
}

std::optional<bool> classify_vulkan_rebar(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept {
    for (const auto& h : heaps) {
        if (h.device_local) {
            return h.host_coherent;
        }
    }
    return std::nullopt;
}

std::optional<bool> classify_vulkan_rebar(const std::vector<VulkanMemoryHeapInfo>& heaps,
                                          std::optional<std::uint32_t> first_device_local_heap_index) noexcept {
    if (first_device_local_heap_index.has_value()) {
        for (const auto& h : heaps) {
            if (h.heap_index == *first_device_local_heap_index) {
                return h.host_coherent;
            }
        }
    }
    return classify_vulkan_rebar(heaps);
}

MemoryReading vulkan_device_local_budget_bytes(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept {
    bool any = false;
    std::uint64_t total = 0;
    for (const auto& h : heaps) {
        if (h.device_local && h.budget_bytes.is_known()) {
            total = saturating_add(total, h.budget_bytes.bytes_or(0));
            any = true;
        }
    }
    return any ? MemoryReading::known(total) : MemoryReading::unknown();
}

MemoryReading vulkan_device_local_usage_bytes(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept {
    bool any = false;
    std::uint64_t total = 0;
    for (const auto& h : heaps) {
        if (h.device_local && h.usage_bytes.is_known()) {
            total = saturating_add(total, h.usage_bytes.bytes_or(0));
            any = true;
        }
    }
    return any ? MemoryReading::known(total) : MemoryReading::unknown();
}

std::optional<std::uint32_t> parse_pcie_link_speed_gen(std::string_view speed_text) noexcept {
    speed_text = trim_view(speed_text);
    if (speed_text.empty() || speed_text.front() == '-') {
        return std::nullopt;
    }
    std::size_t end = 0;
    while (end < speed_text.size() && ((speed_text[end] >= '0' && speed_text[end] <= '9') || speed_text[end] == '.')) {
        ++end;
    }
    if (end == 0) {
        return std::nullopt;
    }
    const std::string_view num = speed_text.substr(0, end);
    const std::string_view rest = trim_view(speed_text.substr(end));
    if (!rest.empty() && !istarts_with(rest, "gt/s")) {
        return std::nullopt;
    }
    if (num == "2.5") return 1u;
    if (num == "5" || num == "5.0") return 2u;
    if (num == "8" || num == "8.0") return 3u;
    if (num == "16" || num == "16.0") return 4u;
    if (num == "32" || num == "32.0") return 5u;
    if (num == "64" || num == "64.0") return 6u;
    return std::nullopt;
}

std::string parse_linux_aspm_policy(std::string_view policy_text) {
    const std::size_t open = policy_text.find('[');
    if (open == std::string_view::npos) {
        return {};
    }
    const std::size_t close = policy_text.find(']', open + 1);
    if (close == std::string_view::npos || policy_text.find('[', open + 1) != std::string_view::npos) {
        return {};
    }
    const std::string_view token = trim_view(policy_text.substr(open + 1, close - open - 1));
    if (token.empty() || token.size() > 32u) {
        return {};
    }
    for (const char c : token) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-';
        if (!ok) return {};
    }
    return std::string(token);
}

std::string format_linux_link_aspm(std::optional<bool> l0s, std::optional<bool> l1, std::optional<bool> l1_1,
                                   std::optional<bool> l1_2) {
    if (!l0s.has_value() && !l1.has_value() && !l1_1.has_value() && !l1_2.has_value()) {
        return {};
    }
    std::string out;
    auto add = [&](std::optional<bool> flag, const char* label) {
        if (flag && *flag) {
            if (!out.empty()) out.push_back('+');
            out += label;
        }
    };
    add(l0s, "l0s");
    add(l1, "l1");
    add(l1_1, "l1.1");
    add(l1_2, "l1.2");
    return out.empty() ? std::string("off") : out;
}

std::string_view windows_aspm_policy_name(std::uint32_t power_index) noexcept {
    switch (power_index) {
        case 0u: return "off";
        case 1u: return "moderate";
        case 2u: return "maximum";
        default: return {};
    }
}

}  // namespace sonder::inference
