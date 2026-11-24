// Sonder Inference: device inventory.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sonder::inference {

enum class DeviceKind { cpu, gpu, npu, remote };

const char* to_string(DeviceKind kind) noexcept;

// One memory measurement with three distinct outcomes, so callers cannot
// confuse them:
//   known, > 0      the OS reported this many bytes
//   known, == 0     the OS reported that nothing is left: refuse to admit work
//   unknown         the platform has no reading, the read failed, or the
//                   source was malformed. This is NOT zero and NOT unlimited;
//                   each caller documents its own fallback (the llama-server
//                   supervisor admits with a warning and skips its floor guard).
// Trivially copyable value type; no thread affinity.
class MemoryReading {
public:
    constexpr MemoryReading() noexcept = default;  // unknown
    [[nodiscard]] static constexpr MemoryReading unknown() noexcept { return {}; }
    [[nodiscard]] static constexpr MemoryReading known(std::uint64_t bytes) noexcept {
        MemoryReading r;
        r.known_ = true;
        r.bytes_ = bytes;
        return r;
    }
    [[nodiscard]] constexpr bool is_known() const noexcept { return known_; }
    [[nodiscard]] constexpr bool is_known_zero() const noexcept { return known_ && bytes_ == 0; }
    [[nodiscard]] constexpr std::optional<std::uint64_t> value() const noexcept {
        return known_ ? std::optional<std::uint64_t>(bytes_) : std::nullopt;
    }
    [[nodiscard]] constexpr std::uint64_t bytes_or(std::uint64_t fallback) const noexcept {
        return known_ ? bytes_ : fallback;
    }
    friend constexpr bool operator==(const MemoryReading&, const MemoryReading&) noexcept = default;

private:
    bool known_ = false;
    std::uint64_t bytes_ = 0;  // 0 while unknown
};

// A snapshot of host memory. `available_physical` is memory a new process can
// use without paging live data out: Windows ullAvailPhys; Linux MemAvailable
// capped by the tightest cgroup v2 `memory.max - memory.current` from this
// process's cgroup up to the root. `available_commit` is Windows
// ullAvailPageFile (the commit this process can still take; Windows
// documents it as the smaller of the system and per-process limit); Linux and
// macOS report it unknown. macOS reports available_physical unknown.
struct HostMemory {
    MemoryReading total_physical;
    MemoryReading available_physical;
    MemoryReading available_commit;
    std::string source;  // e.g. "GlobalMemoryStatusEx", "meminfo+cgroup2"
    std::string note;    // why a reading is unknown or capped (diagnostic text)
};

// Injectable seam. A null function means "use read_host_memory()".
using HostMemoryProvider = std::function<HostMemory()>;

// Live OS reading. Cheap (one syscall on Windows; a few small file reads on
// Linux); callable from any thread; allocates only for the diagnostic strings.
HostMemory read_host_memory();

// Linux parsing, compiled on every platform so it is testable everywhere.
// Strict: a value that is not exactly what the kernel writes yields nullopt.
//  - parse_strict_u64: decimal digits only (no sign, spaces or leading zeros),
//    at most one trailing '\n', no overflow.
//  - parse_meminfo_bytes: `Key:<spaces><digits> kB` on exactly one line,
//    returns bytes; a missing, duplicated, malformed or overflowing field is
//    nullopt.
std::optional<std::uint64_t> parse_strict_u64(std::string_view text) noexcept;
std::optional<std::uint64_t> parse_meminfo_bytes(std::string_view meminfo, std::string_view key) noexcept;

// Reads a file for the cgroup walk: nullopt when the file does not exist.
using FileReader = std::function<std::optional<std::string>(const std::string& path)>;

// Headroom of the tightest memory limit among the cgroup v2 ancestors of the
// process whose /proc/<pid>/cgroup text is `proc_cgroup`, with the cgroup
// filesystem mounted at `cgroup_root` (normally /sys/fs/cgroup).
//   known(n)  the tightest finite `memory.max - memory.current` (0 when at or
//             over the limit)
//   unknown   no cgroup v2 entry, no finite limit on any ancestor, or any
//             memory.max/memory.current that is present but malformed (the
//             `note` says which). Callers combine this with MemAvailable: an
//             `unknown` here with `limited == false` means "no cgroup limit".
struct CgroupHeadroom {
    MemoryReading headroom;
    bool limited = false;    // a finite limit was found
    bool malformed = false;  // a present file could not be parsed strictly
    std::string note;
};
CgroupHeadroom cgroup_v2_headroom(std::string_view proc_cgroup, const FileReader& read_file,
                                  std::string_view cgroup_root = "/sys/fs/cgroup");

// Linux composition of /proc/meminfo text and the cgroup headroom. Pure.
HostMemory compose_linux_host_memory(std::string_view meminfo, const CgroupHeadroom& cgroup);

struct DeviceInfo {
    std::string id;          // stable within a process, e.g. "cpu:0"
    DeviceKind kind = DeviceKind::cpu;
    std::string name;        // human readable, e.g. CPU brand string
    std::uint32_t logical_cores = 0;
    std::uint64_t total_memory_bytes = 0;      // 0 when unknown
    std::uint64_t available_memory_bytes = 0;  // 0 when unknown (or known zero: see available_physical)
    // Tri-state companions of available_memory_bytes (cpu device only).
    MemoryReading available_physical;
    MemoryReading available_commit;
};

// Host inventory. Phase 1 reports the CPU (with system RAM) only; GPU/NPU
// discovery is deferred to backend capability probing.
std::vector<DeviceInfo> enumerate_devices();

enum class GpuVendor { unknown, nvidia, amd, intel, apple };
const char* to_string(GpuVendor vendor) noexcept;

// Pure PCI vendor ID classification (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel,
// 0x106B Apple). Compiled on all platforms for cross-platform testing.
GpuVendor gpu_vendor_from_pci_id(std::uint32_t vendor_id) noexcept;

struct GpuDeviceInfo {
    std::string id;                   // e.g. "gpu:0"
    std::string name;                 // human-readable adapter name
    GpuVendor vendor = GpuVendor::unknown;
    std::uint32_t vendor_id = 0;      // PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel)
    std::uint32_t device_id = 0;      // PCI device id
    MemoryReading dedicated_total;    // total dedicated VRAM
    MemoryReading dedicated_free;     // free/available budget VRAM when known
    MemoryReading shared_total;       // shared/GTT system memory pool when known
    bool is_integrated_uma = false;   // true on Ryzen APU/iGPU unified memory
    std::string backend_hint;         // "cuda", "rocm", "vulkan", "metal", or ""
    std::string source;               // "dxgi", "amdgpu-sysfs", "nvml", etc.
};

// Pure Linux amdgpu sysfs parser (/sys/class/drm/<card_id>/device/*).
// Compiled on all platforms so it is testable everywhere via an injected FileReader.
std::optional<GpuDeviceInfo> parse_amdgpu_sysfs_device(std::string_view card_id, const FileReader& read_file);

// Live GPU inventory: DXGI on Windows (dedicated + free budget + shared/UMA),
// amdgpu sysfs + optional dynamic libnvidia-ml.so.1 on Linux, empty elsewhere.
std::vector<GpuDeviceInfo> enumerate_gpu_devices();

// Effective usable VRAM pool in MiB. Discrete GPUs report dedicated_free when
// prefer_free && dedicated_free.is_known(), else dedicated_total. Integrated
// UMA GPUs (e.g. Ryzen APUs) include their shared/GTT pool rather than only
// the small BIOS dedicated carveout.
std::optional<std::uint64_t> gpu_effective_vram_mib(const GpuDeviceInfo& gpu, bool prefer_free = false) noexcept;

// Selects the effective VRAM budget in MiB for `device_hint` ("CUDA0",
// "ROCm0", "HIP0", "Vulkan0", "gpu:1", "nvidia", "amd", or comma-separated
// combinations), or the largest discrete GPU (preferring discrete over
// integrated UMA) when `device_hint` is empty.
std::optional<std::uint64_t> select_gpu_vram_budget_mib(const std::vector<GpuDeviceInfo>& gpus,
                                                        std::string_view device_hint = {},
                                                        bool prefer_free = false) noexcept;

// Current OS/arch label, e.g. "windows-x86_64".
std::string host_platform();
// Host name used as the telemetry producer node_id default.
std::string host_name();

}  // namespace sonder::inference
