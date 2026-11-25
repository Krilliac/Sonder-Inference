#include "gpu_memory.hpp"
#include "gpu_residency.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>

#if defined(_WIN32)
#include <mutex>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
// windows.h must precede the PDH headers.
#include <pdh.h>
#include <pdhmsg.h>
#elif defined(__linux__)
#include <algorithm>
#include <cerrno>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#endif

namespace sonder::inference::llamaserver {
namespace {

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::string_view trim_view(std::string_view sv) noexcept {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    const auto last = sv.find_last_not_of(" \t\r\n");
    return sv.substr(first, last - first + 1);
}

std::optional<std::uint64_t> parse_unsigned_decimal(std::string_view text) noexcept {
    if (text.empty() || text.size() > 20)
        return std::nullopt;
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9')
            return std::nullopt;
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint64_t> parse_drm_memory_bytes(std::string_view text) noexcept {
    text = trim_view(text);
    if (text.empty())
        return std::nullopt;
    const auto sep = text.find_first_of(" \t");
    if (sep == std::string_view::npos)
        return std::nullopt;
    const std::string_view num_str = text.substr(0, sep);
    const std::string_view unit_str = trim_view(text.substr(sep + 1));
    std::uint64_t multiplier = 0;
    if (unit_str == "B")
        multiplier = 1ull;
    else if (unit_str == "KiB")
        multiplier = 1024ull;
    else if (unit_str == "MiB")
        multiplier = 1024ull * 1024ull;
    else if (unit_str == "GiB")
        multiplier = 1024ull * 1024ull * 1024ull;
    else
        return std::nullopt;
    const auto count = parse_unsigned_decimal(num_str);
    if (!count || *count > std::numeric_limits<std::uint64_t>::max() / multiplier)
        return std::nullopt;
    return *count * multiplier;
}

char ascii_lower_ch(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool parse_hex_u32_token(std::string_view text, std::size_t &pos, std::uint32_t &out) noexcept {
    if (pos + 2 >= text.size() || text[pos] != '0' || (text[pos + 1] != 'x' && text[pos + 1] != 'X'))
        return false;
    pos += 2;
    std::uint32_t value = 0;
    std::size_t digits = 0;
    while (pos < text.size()) {
        const char c = text[pos];
        std::uint32_t nibble = 0;
        if (c >= '0' && c <= '9') {
            nibble = static_cast<std::uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            nibble = static_cast<std::uint32_t>(10 + (c - 'a'));
        } else if (c >= 'A' && c <= 'F') {
            nibble = static_cast<std::uint32_t>(10 + (c - 'A'));
        } else {
            break;
        }
        if (++digits > 8)
            return false;
        value = (value << 4) | nibble;
        ++pos;
    }
    if (digits == 0)
        return false;
    out = value;
    return true;
}

#if defined(_WIN32)
constexpr GUID kIidDxgiFactory1 = {0x770aae78u, 0xf26fu, 0x4dbau, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};

struct ScopedWin32Library {
    HMODULE handle = nullptr;
    explicit ScopedWin32Library(HMODULE h) noexcept : handle(h) {}
    ~ScopedWin32Library() {
        if (handle)
            FreeLibrary(handle);
    }
    ScopedWin32Library(const ScopedWin32Library &) = delete;
    ScopedWin32Library &operator=(const ScopedWin32Library &) = delete;
};

template <typename T>
struct ScopedComPtr {
    T *ptr = nullptr;
    ScopedComPtr() noexcept = default;
    ~ScopedComPtr() {
        if (ptr)
            ptr->Release();
    }
    ScopedComPtr(const ScopedComPtr &) = delete;
    ScopedComPtr &operator=(const ScopedComPtr &) = delete;
};

template <typename Fn>
bool load_win32_symbol(HMODULE handle, const char *symbol, Fn &out) noexcept {
    static_assert(sizeof(Fn) == sizeof(FARPROC));
    const FARPROC raw = GetProcAddress(handle, symbol);
    if (!raw)
        return false;
    std::memcpy(&out, &raw, sizeof(out));
    return true;
}

// Mirrors src/devices/device.cpp is_integrated_uma_memory() so DXGI LUID
// classification stays identical to GPU inventory discovery.
bool is_dxgi_integrated_uma(UINT vendor_id, std::uint64_t dedicated_bytes, std::uint64_t shared_bytes) noexcept {
    if (dedicated_bytes == 0 && shared_bytes > 0)
        return true;
    if (vendor_id == 0x106Bu && shared_bytes > 0)
        return true;
    constexpr std::uint64_t kTwoGiB = 2ull * 1024ull * kMiB;
    if ((vendor_id == 0x1002u || vendor_id == 0x8086u) && dedicated_bytes <= kTwoGiB &&
        shared_bytes >= 4ull * dedicated_bytes && shared_bytes > 0)
        return true;
    return false;
}

void query_dxgi_adapter_luids(std::vector<AdapterLuidInfo> &hardware, std::vector<AdapterLuid> &non_hardware) {
    hardware.clear();
    non_hardware.clear();
    ScopedWin32Library dxgi(LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (!dxgi.handle)
        return;
    using CreateDXGIFactory1Fn = HRESULT(WINAPI *)(REFIID, void **);
    CreateDXGIFactory1Fn create_factory = nullptr;
    if (!load_win32_symbol(dxgi.handle, "CreateDXGIFactory1", create_factory))
        return;
    ScopedComPtr<IDXGIFactory1> factory;
    if (FAILED(create_factory(kIidDxgiFactory1, reinterpret_cast<void **>(&factory.ptr))) || !factory.ptr)
        return;
    for (UINT i = 0; i < 64u; ++i) {
        ScopedComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory.ptr->EnumAdapters1(i, &adapter.ptr)) || !adapter.ptr)
            break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter.ptr->GetDesc1(&desc)))
            continue;
        const AdapterLuid luid{static_cast<std::uint32_t>(desc.AdapterLuid.HighPart),
                               static_cast<std::uint32_t>(desc.AdapterLuid.LowPart)};
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 || desc.VendorId == 0x1414u) {
            non_hardware.push_back(luid);
            continue;
        }
        const bool is_uma = is_dxgi_integrated_uma(desc.VendorId, static_cast<std::uint64_t>(desc.DedicatedVideoMemory),
                                                   static_cast<std::uint64_t>(desc.SharedSystemMemory));
        hardware.push_back(AdapterLuidInfo{luid, is_uma});
    }
}

std::string hex_status(PDH_STATUS status) {
    static constexpr char kDigits[] = "0123456789ABCDEF";
    std::string out = "0x";
    for (int shift = 28; shift >= 0; shift -= 4)
        out.push_back(kDigits[(static_cast<std::uint32_t>(status) >> shift) & 0xFu]);
    return out;
}

std::string narrow(const wchar_t *text) {
    std::string out;
    if (!text)
        return out;
    // PDH instance names of this object are ASCII ("pid_<n>_luid_0x..._phys_<n>").
    for (; *text; ++text)
        out.push_back(*text < 0x80 ? static_cast<char>(*text) : '?');
    return out;
}

class PdhQuery {
  public:
    PdhQuery() { status_ = PdhOpenQueryW(nullptr, 0, &query_); }
    ~PdhQuery() {
        if (query_)
            PdhCloseQuery(query_);
    }
    PdhQuery(const PdhQuery &) = delete;
    PdhQuery &operator=(const PdhQuery &) = delete;
    [[nodiscard]] PDH_STATUS status() const { return status_; }
    [[nodiscard]] PDH_HQUERY get() const { return query_; }

  private:
    PDH_HQUERY query_ = nullptr;
    PDH_STATUS status_ = ERROR_SUCCESS;
};

// A wildcard instance with no matches is an empty (not failed) sample: a
// process that has not created a GPU context yet has no instances.
bool is_no_instance(PDH_STATUS status) {
    return status == static_cast<PDH_STATUS>(PDH_NO_DATA) || status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_INSTANCE) ||
           status == static_cast<PDH_STATUS>(PDH_INVALID_DATA);
}

Status read_array(PDH_HCOUNTER counter, std::vector<CounterInstance> &out) {
    DWORD bytes = 0;
    DWORD count = 0;
    PDH_STATUS status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count, nullptr);
    if (is_no_instance(status))
        return {};
    if (status != static_cast<PDH_STATUS>(PDH_MORE_DATA))
        return Status(ErrorCode::io_error, "PdhGetFormattedCounterArray failed: " + hex_status(status));
    // Bounded: a process has one instance per adapter/segment view.
    if (bytes == 0 || bytes > 1024u * 1024u)
        return Status(ErrorCode::io_error, "PDH counter array has an implausible size");
    std::vector<unsigned char> buffer(bytes);
    auto *items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W *>(buffer.data());
    status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count, items);
    if (is_no_instance(status))
        return {};
    if (status != ERROR_SUCCESS)
        return Status(ErrorCode::io_error, "PdhGetFormattedCounterArray failed: " + hex_status(status));
    for (DWORD i = 0; i < count; ++i) {
        const auto &item = items[i];
        if (item.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && item.FmtValue.CStatus != PDH_CSTATUS_NEW_DATA)
            continue;
        if (item.FmtValue.largeValue < 0)
            continue;
        out.push_back({narrow(item.szName), static_cast<std::uint64_t>(item.FmtValue.largeValue)});
    }
    return {};
}

class PdhCounterSource final : public GpuCounterSource {
  public:
    std::string name() const override { return "pdh"; }
    bool supported() const override { return true; }
    Result<GpuProcessCounters> read(std::uint32_t pid) override {
        if (pid == 0)
            return Status(ErrorCode::invalid_argument, "no process id to sample");
        PdhQuery query;
        if (query.status() != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhOpenQuery failed: " + hex_status(query.status()));
        const std::wstring prefix = L"\\GPU Process Memory(pid_" + std::to_wstring(pid) + L"_*)\\";
        PDH_HCOUNTER dedicated = nullptr;
        PDH_HCOUNTER shared = nullptr;
        PDH_STATUS status = PdhAddEnglishCounterW(query.get(), (prefix + L"Dedicated Usage").c_str(), 0, &dedicated);
        if (status == ERROR_SUCCESS)
            status = PdhAddEnglishCounterW(query.get(), (prefix + L"Shared Usage").c_str(), 0, &shared);
        if (status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_OBJECT) ||
            status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_COUNTER))
            return Status(ErrorCode::unsupported,
                          "the GPU Process Memory performance counters are not available (WDDM 2.x required)");
        if (status != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhAddEnglishCounter failed: " + hex_status(status));
        status = PdhCollectQueryData(query.get());
        GpuProcessCounters counters;
        if (is_no_instance(status))
            return counters;
        if (status != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhCollectQueryData failed: " + hex_status(status));
        if (auto st = read_array(dedicated, counters.dedicated); !st.ok())
            return st;
        if (auto st = read_array(shared, counters.shared); !st.ok())
            return st;
        classify_live(counters);
        return counters;
    }

  private:
    void classify_live(GpuProcessCounters &counters) {
        if (counters.dedicated.empty() && counters.shared.empty())
            return;
        std::lock_guard lock(dxgi_mutex_);
        if (!dxgi_loaded_) {
            query_dxgi_adapter_luids(hardware_luids_, non_hardware_luids_);
            dxgi_loaded_ = true;
        }
        const auto is_known = [&](const AdapterLuid &luid) noexcept {
            for (const auto &hw : hardware_luids_) {
                if (hw.luid == luid)
                    return true;
            }
            for (const auto &other : non_hardware_luids_) {
                if (other == luid)
                    return true;
            }
            return false;
        };
        const auto has_unseen_luid = [&](const std::vector<CounterInstance> &items) noexcept {
            for (const auto &item : items) {
                const auto luid = parse_pdh_adapter_luid(item.name);
                if (luid && !is_known(*luid))
                    return true;
            }
            return false;
        };
        if (has_unseen_luid(counters.dedicated) || has_unseen_luid(counters.shared)) {
            query_dxgi_adapter_luids(hardware_luids_, non_hardware_luids_);
            const auto record_remaining = [&](const std::vector<CounterInstance> &items) {
                for (const auto &item : items) {
                    if (non_hardware_luids_.size() >= 64u)
                        break;
                    const auto luid = parse_pdh_adapter_luid(item.name);
                    if (luid && !is_known(*luid))
                        non_hardware_luids_.push_back(*luid);
                }
            };
            record_remaining(counters.dedicated);
            record_remaining(counters.shared);
        }
        classify_pdh_counters(counters, hardware_luids_);
    }

    std::mutex dxgi_mutex_;
    bool dxgi_loaded_ = false;
    std::vector<AdapterLuidInfo> hardware_luids_;
    std::vector<AdapterLuid> non_hardware_luids_;
};
#else
class UnsupportedCounterSource final : public GpuCounterSource {
  public:
    std::string name() const override { return "none"; }
    bool supported() const override { return false; }
    Result<GpuProcessCounters> read(std::uint32_t) override {
        return Status(ErrorCode::unsupported, "per-process GPU memory counters are not available on this platform");
    }
};

#if defined(__linux__)
template <typename Fn>
bool load_dl_symbol(void *handle, const char *symbol, Fn &out) noexcept {
    static_assert(sizeof(Fn) == sizeof(void *));
    void *raw = dlsym(handle, symbol);
    if (!raw)
        return false;
    std::memcpy(&out, &raw, sizeof(out));
    return true;
}

bool is_drm_card_entry(std::string_view name) noexcept {
    if (name.size() <= 4 || name.substr(0, 4) != "card")
        return false;
    for (std::size_t i = 4; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9')
            return false;
    }
    return true;
}

bool has_amdgpu_sysfs() {
    DIR *dir = opendir("/sys/class/drm");
    if (!dir)
        return false;
    bool found = false;
    while (const dirent *entry = readdir(dir)) {
        const std::string_view name(entry->d_name);
        if (!is_drm_card_entry(name))
            continue;
        std::ifstream vram("/sys/class/drm/" + std::string(name) + "/device/mem_info_vram_total");
        if (vram.is_open()) {
            found = true;
            break;
        }
    }
    closedir(dir);
    return found;
}

std::optional<std::string> read_small_linux_file(const std::string &path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    std::string text(limit, '\0');
    in.read(text.data(), static_cast<std::streamsize>(limit));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

bool fdinfo_has_driver(std::string_view text, std::string_view expected_driver) noexcept {
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto end = text.find('\n', pos);
        std::string_view line =
            text.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos);
        pos = end == std::string_view::npos ? text.size() : end + 1;
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos)
            continue;
        if (trim_view(line.substr(0, colon)) == "drm-driver")
            return trim_view(line.substr(colon + 1)) == expected_driver;
    }
    return false;
}

bool read_proc_drm_fdinfo(std::uint32_t pid, std::string &out, bool &permission_denied, bool amdgpu_only) {
    permission_denied = false;
    out.clear();
    const std::string dir_path = "/proc/" + std::to_string(pid) + "/fdinfo";
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) {
        if (errno == EACCES || errno == EPERM)
            permission_denied = true;
        return false;
    }
    std::size_t scanned = 0;
    constexpr std::size_t kMaxFds = 4096;
    while (const dirent *entry = readdir(dir)) {
        const std::string_view name(entry->d_name);
        if (name == "." || name == "..")
            continue;
        if (++scanned > kMaxFds)
            break;
        const auto text = read_small_linux_file(dir_path + "/" + std::string(name), 4096);
        if (!text || text->find("drm-memory-") == std::string::npos)
            continue;
        if (amdgpu_only && !fdinfo_has_driver(*text, "amdgpu"))
            continue;
        if (!out.empty())
            out += "\n\n";
        out += *text;
    }
    closedir(dir);
    return true;
}

class NvmlProcessProbe {
  public:
    using NvmlReturn = int;
    struct NvmlDeviceOpaque;
    using NvmlDevice = NvmlDeviceOpaque *;

    struct NvmlProcessInfoV1 {
        unsigned int pid;
        unsigned long long usedGpuMemory;
    };
    struct NvmlProcessInfoV2 {
        unsigned int pid;
        unsigned long long usedGpuMemory;
        unsigned int gpuInstanceId;
        unsigned int computeInstanceId;
    };

    static constexpr NvmlReturn kNvmlSuccess = 0;
    static constexpr NvmlReturn kNvmlErrorInsufficientSize = 7;
    static constexpr unsigned long long kNvmlValueNotAvailable =
        std::numeric_limits<unsigned long long>::max();

    NvmlProcessProbe() {
        handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!handle_)
            return;
        if (!load_dl_symbol(handle_, "nvmlInit_v2", init_) ||
            !load_dl_symbol(handle_, "nvmlShutdown", shutdown_) ||
            !load_dl_symbol(handle_, "nvmlDeviceGetCount_v2", get_count_) ||
            !load_dl_symbol(handle_, "nvmlDeviceGetHandleByIndex_v2", get_handle_)) {
            cleanup();
            return;
        }
        if (!load_dl_symbol(handle_, "nvmlDeviceGetComputeRunningProcesses_v3", compute_v2_) &&
            !load_dl_symbol(handle_, "nvmlDeviceGetComputeRunningProcesses_v2", compute_v2_)) {
            (void)load_dl_symbol(handle_, "nvmlDeviceGetComputeRunningProcesses", compute_v1_);
        }
        if (!load_dl_symbol(handle_, "nvmlDeviceGetGraphicsRunningProcesses_v3", graphics_v2_) &&
            !load_dl_symbol(handle_, "nvmlDeviceGetGraphicsRunningProcesses_v2", graphics_v2_)) {
            (void)load_dl_symbol(handle_, "nvmlDeviceGetGraphicsRunningProcesses", graphics_v1_);
        }
        if (!compute_v2_ && !compute_v1_ && !graphics_v2_ && !graphics_v1_) {
            cleanup();
            return;
        }
        if (init_() != kNvmlSuccess) {
            cleanup();
            return;
        }
        initialized_ = true;
        unsigned int count = 0;
        if (get_count_(&count) != kNvmlSuccess || count == 0) {
            cleanup();
            return;
        }
        available_ = true;
    }

    ~NvmlProcessProbe() { cleanup(); }

    NvmlProcessProbe(const NvmlProcessProbe &) = delete;
    NvmlProcessProbe &operator=(const NvmlProcessProbe &) = delete;

    [[nodiscard]] bool available() const noexcept { return available_; }

    void read_pid(std::uint32_t pid, GpuProcessCounters &out) {
        if (!available_)
            return;
        std::lock_guard lock(mutex_);
        unsigned int count = 0;
        if (get_count_(&count) != kNvmlSuccess || count == 0)
            return;
        count = std::min<unsigned int>(count, 64u);
        for (unsigned int i = 0; i < count; ++i) {
            NvmlDevice dev = nullptr;
            if (get_handle_(i, &dev) != kNvmlSuccess || !dev)
                continue;
            std::uint64_t used_bytes = 0;
            bool has_valid_bytes = false;
            if (compute_v2_)
                has_valid_bytes |= query_list<NvmlProcessInfoV2>(compute_v2_, dev, pid, used_bytes);
            else if (compute_v1_)
                has_valid_bytes |= query_list<NvmlProcessInfoV1>(compute_v1_, dev, pid, used_bytes);
            if (graphics_v2_)
                has_valid_bytes |= query_list<NvmlProcessInfoV2>(graphics_v2_, dev, pid, used_bytes);
            else if (graphics_v1_)
                has_valid_bytes |= query_list<NvmlProcessInfoV1>(graphics_v1_, dev, pid, used_bytes);
            if (has_valid_bytes) {
                const std::string instance = "pid_" + std::to_string(pid) + "_nvml_" + std::to_string(i);
                out.dedicated.push_back({instance, used_bytes, CounterAdapterKind::discrete});
                out.shared.push_back({instance, 0, CounterAdapterKind::discrete});
            }
        }
    }

  private:
    template <typename Entry, typename Fn>
    static bool query_list(Fn fn, NvmlDevice dev, std::uint32_t pid, std::uint64_t &max_bytes) {
        if (!fn)
            return false;
        unsigned int count = 0;
        NvmlReturn rc = fn(dev, &count, nullptr);
        if (rc == kNvmlSuccess || rc != kNvmlErrorInsufficientSize || count == 0)
            return false;
        for (int attempt = 0; attempt < 2; ++attempt) {
            const unsigned int capacity = std::min<unsigned int>(count, 4088u) + 8u;
            std::vector<Entry> entries(capacity);
            unsigned int actual = capacity;
            rc = fn(dev, &actual, entries.data());
            if (rc == kNvmlErrorInsufficientSize) {
                count = actual;
                continue;
            }
            if (rc != kNvmlSuccess)
                return false;
            bool has_valid_bytes = false;
            const unsigned int limit = std::min(actual, capacity);
            for (unsigned int i = 0; i < limit; ++i) {
                if (entries[i].pid != pid)
                    continue;
                if (entries[i].usedGpuMemory != kNvmlValueNotAvailable) {
                    has_valid_bytes = true;
                    const auto bytes = static_cast<std::uint64_t>(entries[i].usedGpuMemory);
                    if (bytes > max_bytes)
                        max_bytes = bytes;
                }
            }
            return has_valid_bytes;
        }
        return false;
    }

    void cleanup() noexcept {
        if (initialized_ && shutdown_)
            (void)shutdown_();
        initialized_ = false;
        if (handle_)
            dlclose(handle_);
        handle_ = nullptr;
        available_ = false;
    }

    void *handle_ = nullptr;
    bool initialized_ = false;
    bool available_ = false;
    std::mutex mutex_;
    NvmlReturn (*init_)() = nullptr;
    NvmlReturn (*shutdown_)() = nullptr;
    NvmlReturn (*get_count_)(unsigned int *) = nullptr;
    NvmlReturn (*get_handle_)(unsigned int, NvmlDevice *) = nullptr;
    NvmlReturn (*compute_v2_)(NvmlDevice, unsigned int *, NvmlProcessInfoV2 *) = nullptr;
    NvmlReturn (*compute_v1_)(NvmlDevice, unsigned int *, NvmlProcessInfoV1 *) = nullptr;
    NvmlReturn (*graphics_v2_)(NvmlDevice, unsigned int *, NvmlProcessInfoV2 *) = nullptr;
    NvmlReturn (*graphics_v1_)(NvmlDevice, unsigned int *, NvmlProcessInfoV1 *) = nullptr;
};

class LinuxGpuCounterSource final : public GpuCounterSource {
  public:
    LinuxGpuCounterSource(bool amdgpu_available, std::unique_ptr<NvmlProcessProbe> nvml)
        : amdgpu_available_(amdgpu_available), nvml_(std::move(nvml)) {}

    std::string name() const override {
        if (amdgpu_available_ && nvml_)
            return "amdgpu-fdinfo+nvml";
        if (amdgpu_available_)
            return "amdgpu-fdinfo";
        return "nvml";
    }

    bool supported() const override { return amdgpu_available_ || nvml_ != nullptr; }

    Result<GpuProcessCounters> read(std::uint32_t pid) override {
        if (pid == 0)
            return Status(ErrorCode::invalid_argument, "no process id to sample");
        GpuProcessCounters counters;
        bool permission_denied = false;
        if (amdgpu_available_) {
            std::string fdinfo_text;
            const bool opened = read_proc_drm_fdinfo(pid, fdinfo_text, permission_denied, nvml_ != nullptr);
            if (!opened && permission_denied && !nvml_) {
                return Status(ErrorCode::io_error,
                              "cannot read /proc/" + std::to_string(pid) + "/fdinfo (permission denied)");
            }
            if (!fdinfo_text.empty()) {
                std::uint64_t vram = 0;
                std::uint64_t gtt = 0;
                if (parse_linux_drm_fdinfo(fdinfo_text, vram, gtt)) {
                    const std::string instance = "pid_" + std::to_string(pid) + "_drm_fdinfo";
                    counters.dedicated.push_back({instance, vram});
                    counters.shared.push_back({instance, gtt});
                }
            }
        }
        if (nvml_) {
            const std::size_t before = counters.dedicated.size();
            nvml_->read_pid(pid, counters);
            if (permission_denied && counters.dedicated.size() == before) {
                return Status(ErrorCode::io_error,
                              "cannot read /proc/" + std::to_string(pid) + "/fdinfo (permission denied)");
            }
        }
        return counters;
    }

  private:
    bool amdgpu_available_ = false;
    std::unique_ptr<NvmlProcessProbe> nvml_;
};
#endif
#endif

} // namespace

std::shared_ptr<GpuCounterSource> make_gpu_counter_source() {
#if defined(_WIN32)
    return std::make_shared<PdhCounterSource>();
#elif defined(__linux__)
    const bool amdgpu = has_amdgpu_sysfs();
    auto nvml = std::make_unique<NvmlProcessProbe>();
    if (!nvml->available())
        nvml.reset();
    if (amdgpu || nvml)
        return std::make_shared<LinuxGpuCounterSource>(amdgpu, std::move(nvml));
    return std::make_shared<UnsupportedCounterSource>();
#else
    return std::make_shared<UnsupportedCounterSource>();
#endif
}

bool parse_linux_drm_fdinfo(std::string_view fdinfo_text, std::uint64_t &vram_bytes, std::uint64_t &gtt_bytes) {
    vram_bytes = 0;
    gtt_bytes = 0;

    struct Record {
        std::string driver;
        std::string client_id;
        std::uint64_t vram = 0;
        std::uint64_t gtt = 0;
        bool has_driver = false;
        bool has_client_id = false;
        bool has_vram = false;
        bool has_gtt = false;
        bool has_any = false;
    };

    Record rec;
    std::unordered_set<std::string> seen_clients;
    bool parsed_any = false;

    const auto flush = [&]() {
        if (rec.has_vram || rec.has_gtt) {
            const bool is_nvidia_drm = rec.driver == "nvidia-drm" || rec.driver == "nvidia";
            if (!is_nvidia_drm) {
                bool duplicate = false;
                if (rec.has_client_id) {
                    std::string key = rec.driver + ":" + rec.client_id;
                    duplicate = !seen_clients.insert(std::move(key)).second;
                }
                if (!duplicate) {
                    vram_bytes = saturating_add(vram_bytes, rec.vram);
                    gtt_bytes = saturating_add(gtt_bytes, rec.gtt);
                    parsed_any = true;
                }
            }
        }
        rec = {};
    };

    std::size_t pos = 0;
    while (pos < fdinfo_text.size()) {
        const auto end = fdinfo_text.find('\n', pos);
        std::string_view line =
            fdinfo_text.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos);
        pos = end == std::string_view::npos ? fdinfo_text.size() : end + 1;
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        const std::string_view trimmed = trim_view(line);
        if (trimmed.empty()) {
            flush();
            continue;
        }
        const auto colon = trimmed.find(':');
        if (colon == std::string_view::npos)
            continue;
        const std::string_view key = trim_view(trimmed.substr(0, colon));
        const std::string_view val = trim_view(trimmed.substr(colon + 1));

        if (key == "pos") {
            if (rec.has_any)
                flush();
            rec.has_any = true;
        } else if (key == "drm-driver") {
            if (rec.has_driver || rec.has_client_id || rec.has_vram || rec.has_gtt)
                flush();
            rec.driver = std::string(val);
            rec.has_driver = true;
            rec.has_any = true;
        } else if (key == "drm-client-id") {
            if (rec.has_client_id)
                flush();
            if (!val.empty()) {
                rec.client_id = std::string(val);
                rec.has_client_id = true;
                rec.has_any = true;
            }
        } else if (key == "drm-memory-vram") {
            if (rec.has_vram)
                flush();
            if (const auto bytes = parse_drm_memory_bytes(val)) {
                rec.vram = *bytes;
                rec.has_vram = true;
                rec.has_any = true;
            }
        } else if (key == "drm-memory-gtt") {
            if (rec.has_gtt)
                flush();
            if (const auto bytes = parse_drm_memory_bytes(val)) {
                rec.gtt = *bytes;
                rec.has_gtt = true;
                rec.has_any = true;
            }
        } else {
            rec.has_any = true;
        }
    }
    flush();
    return parsed_any;
}

std::optional<AdapterLuid> parse_pdh_adapter_luid(std::string_view instance_name) noexcept {
    constexpr std::string_view kTag = "_luid_";
    if (instance_name.size() < kTag.size() + 7)
        return std::nullopt;
    std::size_t tag_pos = std::string_view::npos;
    for (std::size_t i = 0; i + kTag.size() <= instance_name.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < kTag.size(); ++j) {
            if (ascii_lower_ch(instance_name[i + j]) != kTag[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            tag_pos = i;
            break;
        }
    }
    if (tag_pos == std::string_view::npos)
        return std::nullopt;
    std::size_t pos = tag_pos + kTag.size();
    std::uint32_t high = 0;
    std::uint32_t low = 0;
    if (!parse_hex_u32_token(instance_name, pos, high))
        return std::nullopt;
    if (pos >= instance_name.size() || instance_name[pos] != '_')
        return std::nullopt;
    ++pos;
    if (!parse_hex_u32_token(instance_name, pos, low))
        return std::nullopt;
    if (pos < instance_name.size() && instance_name[pos] != '_')
        return std::nullopt;
    return AdapterLuid{high, low};
}

void classify_pdh_counters(GpuProcessCounters &counters, const std::vector<AdapterLuidInfo> &adapters) noexcept {
    if (adapters.empty())
        return;
    const auto classify_vec = [&](std::vector<CounterInstance> &items) noexcept {
        for (auto &item : items) {
            const auto luid = parse_pdh_adapter_luid(item.name);
            if (!luid)
                continue;
            for (const auto &adapter : adapters) {
                if (adapter.luid == *luid) {
                    item.adapter_kind = adapter.is_integrated_uma ? CounterAdapterKind::integrated_uma
                                                                  : CounterAdapterKind::discrete;
                    break;
                }
            }
        }
    };
    classify_vec(counters.dedicated);
    classify_vec(counters.shared);
}

bool is_process_instance(std::string_view name, std::uint32_t pid) {
    const std::string prefix = "pid_" + std::to_string(pid) + "_";
    return name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0;
}

GpuMemorySample summarize_gpu_counters(std::uint32_t pid, const GpuProcessCounters &counters) {
    GpuMemorySample sample;
    std::size_t dedicated_instances = 0;
    std::size_t shared_instances = 0;
    std::size_t discrete_dedicated_instances = 0;
    std::size_t discrete_shared_instances = 0;
    std::size_t uma_dedicated_instances = 0;
    std::size_t uma_shared_instances = 0;
    bool any_classified = false;
    bool any_nonzero_unknown = false;

    for (const auto &item : counters.dedicated) {
        if (!is_process_instance(item.name, pid))
            continue;
        sample.dedicated_bytes = saturating_add(sample.dedicated_bytes, item.value);
        ++dedicated_instances;
        switch (item.adapter_kind) {
        case CounterAdapterKind::discrete:
            sample.discrete_dedicated_bytes = saturating_add(sample.discrete_dedicated_bytes, item.value);
            ++discrete_dedicated_instances;
            any_classified = true;
            break;
        case CounterAdapterKind::integrated_uma:
            sample.uma_dedicated_bytes = saturating_add(sample.uma_dedicated_bytes, item.value);
            ++uma_dedicated_instances;
            any_classified = true;
            break;
        case CounterAdapterKind::unknown:
            if (item.value > 0)
                any_nonzero_unknown = true;
            break;
        }
    }
    for (const auto &item : counters.shared) {
        if (!is_process_instance(item.name, pid))
            continue;
        sample.shared_bytes = saturating_add(sample.shared_bytes, item.value);
        ++shared_instances;
        switch (item.adapter_kind) {
        case CounterAdapterKind::discrete:
            sample.discrete_shared_bytes = saturating_add(sample.discrete_shared_bytes, item.value);
            ++discrete_shared_instances;
            any_classified = true;
            break;
        case CounterAdapterKind::integrated_uma:
            sample.uma_shared_bytes = saturating_add(sample.uma_shared_bytes, item.value);
            ++uma_shared_instances;
            any_classified = true;
            break;
        case CounterAdapterKind::unknown:
            if (item.value > 0)
                any_nonzero_unknown = true;
            break;
        }
    }
    sample.instances = dedicated_instances > shared_instances ? dedicated_instances : shared_instances;
    sample.discrete_instances = discrete_dedicated_instances > discrete_shared_instances
                                    ? discrete_dedicated_instances
                                    : discrete_shared_instances;
    sample.uma_instances = uma_dedicated_instances > uma_shared_instances
                               ? uma_dedicated_instances
                               : uma_shared_instances;
    sample.has_adapter_classification = any_classified && !any_nonzero_unknown;
    return sample;
}

const char *to_string(SpillPolicy policy) noexcept {
    switch (policy) {
    case SpillPolicy::warn:
        return "warn";
    case SpillPolicy::refuse:
        return "refuse";
    case SpillPolicy::auto_fit:
        return "auto_fit";
    }
    return "warn";
}

std::optional<SpillPolicy> parse_spill_policy(std::string_view text) noexcept {
    if (text == "warn")
        return SpillPolicy::warn;
    if (text == "refuse")
        return SpillPolicy::refuse;
    if (text == "auto_fit")
        return SpillPolicy::auto_fit;
    return std::nullopt;
}

Status validate_spill_guard(const SpillGuardOptions &o) {
    const auto invalid = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: spill guard: ") + what);
    };
    if (o.policy != SpillPolicy::warn && o.policy != SpillPolicy::refuse && o.policy != SpillPolicy::auto_fit)
        return invalid("unknown policy");
    if (o.threshold_bytes == 0 || o.threshold_bytes > 1024ull * 1024 * kMiB)
        return invalid("threshold must be in (0, 1 TiB]");
    if (o.baseline_bytes > 1024ull * 1024 * kMiB)
        return invalid("baseline must be at most 1 TiB");
    if (o.baseline_bytes_per_1k_ctx > 1024ull * kMiB)
        return invalid("baseline growth per 1k context must be at most 1 GiB");
    if (o.sample_interval.count() < 100 || o.sample_interval > std::chrono::hours(1))
        return invalid("sample interval must be in [100 ms, 1 h]");
    if (!std::isfinite(o.step_factor) || o.step_factor <= 0.0 || o.step_factor >= 1.0)
        return invalid("step factor must be in (0, 1)");
    if (o.step_align == 0 || o.step_align > 1u << 20)
        return invalid("step alignment must be in [1, 1048576]");
    if (o.min_ctx == 0 || o.min_ctx > (1ull << 32))
        return invalid("minimum context must be in [1, 2^32]");
    if (o.max_attempts == 0 || o.max_attempts > 32)
        return invalid("max fit attempts must be in [1, 32]");
    if (o.residency.enabled)
        return validate_residency_guard(o.residency);
    return {};
}

std::uint64_t effective_baseline(const SpillGuardOptions &options, std::optional<std::uint64_t> ctx) {
    if (!ctx || options.baseline_bytes_per_1k_ctx == 0)
        return options.baseline_bytes;
    const std::uint64_t blocks = *ctx / 1024;
    const std::uint64_t growth =
        blocks != 0 && options.baseline_bytes_per_1k_ctx > std::numeric_limits<std::uint64_t>::max() / blocks
            ? std::numeric_limits<std::uint64_t>::max()
            : options.baseline_bytes_per_1k_ctx * blocks;
    return saturating_add(options.baseline_bytes, growth);
}

std::uint64_t effective_spill_shared_bytes(const GpuMemorySample &sample,
                                           const SpillGuardOptions &options) noexcept {
    if (options.uma_mode && !options.hybrid_uma_mode)
        return 0;
    if (sample.has_adapter_classification) {
        if (options.hybrid_uma_mode || sample.discrete_instances > 0)
            return sample.discrete_shared_bytes;
    }
    if (options.uma_mode || options.hybrid_uma_mode)
        return 0;
    return sample.shared_bytes;
}

bool is_spilled(const GpuMemorySample &sample, const SpillGuardOptions &options,
                std::optional<std::uint64_t> ctx) {
    if (options.uma_mode && !options.hybrid_uma_mode)
        return false;
    if (options.hybrid_uma_mode && !sample.has_adapter_classification)
        return false;
    const std::uint64_t spill_shared = effective_spill_shared_bytes(sample, options);
    const auto limit = saturating_add(effective_baseline(options, ctx), options.threshold_bytes);
    return spill_shared > limit;
}

std::optional<std::uint64_t> next_fit_context(std::uint64_t current, const SpillGuardOptions &o) {
    if (current <= o.min_ctx || o.step_align == 0)
        return std::nullopt;
    const long double scaled = static_cast<long double>(current) * static_cast<long double>(o.step_factor);
    std::uint64_t next = scaled >= static_cast<long double>(current) ? current : static_cast<std::uint64_t>(scaled);
    next -= next % o.step_align;
    // Always make progress by at least one alignment step.
    if (next + o.step_align > current)
        next = current > o.step_align ? current - o.step_align : 0;
    if (next < o.min_ctx)
        next = o.min_ctx;
    if (next >= current)
        return std::nullopt;
    return next;
}

} // namespace sonder::inference::llamaserver
