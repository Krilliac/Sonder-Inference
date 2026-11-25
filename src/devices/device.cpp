#include "sonder/inference/device.hpp"

#include <algorithm>
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

std::vector<GpuDeviceInfo> live_gpu_devices() {
    std::vector<GpuDeviceInfo> gpus;
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
                    gpus.push_back(std::move(info));
                }
            }
        }
    }

    // Always query System32 nvml.dll and merge any headless / TCC-mode NVIDIA
    // GPUs that DXGI does not enumerate, even when a WDDM NVIDIA card is present.
    const auto nvml_gpus = enumerate_win32_nvml_devices();
    merge_win32_nvml_devices(gpus, nvml_gpus);
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
            }
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
                                  iequals(nv.pci_bus_id, gpus[i].pci_bus_id);
            const bool same_dev = (nv.device_id != 0 && gpus[i].device_id == nv.device_id);
            const bool same_name = !nv.name.empty() && iequals(nv.name, gpus[i].name);
            if (same_bus || same_dev || same_name) {
                dxgi_matched[i] = true;
                if (gpus[i].pci_bus_id.empty() && !nv.pci_bus_id.empty()) {
                    gpus[i].pci_bus_id = nv.pci_bus_id;
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

    info.backend_hint = rocm_available ? "rocm" : "vulkan";
    info.source = "amdgpu-sysfs";
    return info;
}

std::vector<GpuDeviceInfo> enumerate_gpu_devices() {
    static std::mutex probe_mutex;
    std::lock_guard<std::mutex> lock(probe_mutex);
    return live_gpu_devices();
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

}  // namespace sonder::inference
