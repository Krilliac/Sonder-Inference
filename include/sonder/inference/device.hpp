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

// Per-heap Vulkan memory snapshot (VkPhysicalDeviceMemoryProperties2 +
// VK_EXT_memory_budget when available).
struct VulkanMemoryHeapInfo {
    std::uint32_t heap_index = 0;
    std::uint64_t size_bytes = 0;
    MemoryReading budget_bytes;   // known when VK_EXT_memory_budget is supported
    MemoryReading usage_bytes;    // known when VK_EXT_memory_budget is supported
    bool device_local = false;    // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    bool multi_instance = false;  // VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
    bool host_visible = false;    // any memory type on this heap has HOST_VISIBLE
    bool host_coherent = false;   // any memory type on this heap has DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT
};

// Parsed ggml Vulkan (or accelerator) device capability line, e.g.:
//   ggml_vulkan: 0 = AMD Radeon RX 9060 XT (AMD proprietary driver) | uma: 0 | rebar: 1 |
//                fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 32768 |
//                int dot: 1 | matrix cores: KHR_coopmat
// Pre-b12171 lines omit `rebar`, leaving `rebar` as std::nullopt.
struct GgmlDeviceTelemetry {
    std::string backend;                              // e.g. "vulkan"
    std::optional<std::uint32_t> index;               // e.g. 0
    std::string device_name;                          // e.g. "AMD Radeon RX 9060 XT"
    std::string driver_name;                          // e.g. "AMD proprietary driver"
    std::optional<bool> uma;                          // uma: 0|1
    std::optional<bool> rebar;                        // rebar: 0|1 (absent before b12171)
    std::string fp16;                                 // "0", "1", "dot2", or ""
    std::optional<bool> bf16;                         // bf16: 0|1
    std::optional<bool> fp4;                          // fp4: 0|1
    std::optional<std::uint32_t> warp_size;           // subgroup / warp size
    std::optional<std::uint64_t> shared_memory_bytes; // workgroup shared memory in bytes
    std::optional<bool> int_dot;                      // int dot: 0|1
    std::string matrix_cores;                         // "none", "KHR_coopmat", "NV_coopmat2", "NV_coopmat2v", or ""
    std::string raw_line;                             // trimmed source line (bounded)
};

// PCIe link and ASPM telemetry for a GPU device. Unknown fields remain unset.
struct PcieLinkTelemetry {
    std::optional<std::uint32_t> current_gen;    // PCIe generation 1..6
    std::optional<std::uint32_t> current_width;  // PCIe lane width (1..32)
    std::optional<std::uint32_t> max_gen;        // max capable PCIe generation 1..6
    std::optional<std::uint32_t> max_width;      // max capable lane width (1..32)
    std::string aspm_state;                      // "off", "moderate", "maximum", "l0s", "l1", "l0s+l1", ...
    std::string aspm_source;                     // "windows_power_plan", "linux_sysfs_link", "linux_pcie_aspm_policy"
};

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
    bool is_tcc_mode = false;         // true for Windows NVIDIA TCC-mode compute adapters
    std::string backend_hint;         // "cuda", "rocm", "vulkan", "metal", or ""
    std::string source;               // "dxgi", "amdgpu-sysfs", "nvml", etc.
    std::string pci_bus_id;           // optional PCI BDF (e.g. "0000:01:00.0")
    std::optional<std::size_t> rocm_runtime_ordinal;  // mapped ROCm/HIP ordinal after visibility masks
    // Additive device telemetry (NAT-08; empty / nullopt when unavailable).
    std::string driver_name;                          // e.g. "AMD proprietary driver", "NVIDIA", "radv"
    std::string driver_build;                         // canonical driver build string (e.g. "25.10.2 (32.0.21025.1024)")
    std::optional<bool> rebar;                        // true when VRAM heap is host-visible+coherent or ggml rebar:1
    std::optional<std::uint64_t> shared_memory_bytes; // workgroup compute shared memory (bytes)
    std::optional<std::uint32_t> warp_size;           // subgroup / warp size
    std::string matrix_cores;                         // "KHR_coopmat", "NV_coopmat2", "NV_coopmat2v", "none", or ""
    PcieLinkTelemetry pcie;                           // PCIe link gen/width and ASPM state
    std::vector<VulkanMemoryHeapInfo> vulkan_heaps;   // per-heap Vulkan memory + VK_EXT_memory_budget
    std::optional<GgmlDeviceTelemetry> ggml_device;   // parsed ggml device line when captured/merged
};

// Pure NAT-08 device-telemetry parsers and classifiers (compiled on all platforms).
std::optional<GgmlDeviceTelemetry> parse_ggml_device_line(std::string_view line);
void apply_ggml_device_telemetry(GpuDeviceInfo& gpu, const GgmlDeviceTelemetry& ggml);
// Process-wide bounded registry of ggml device lines observed by in-process
// backends (e.g. llama.cpp RouteLog). enumerate_gpu_devices() merges matching
// entries onto GpuDeviceInfo so devices/node/health/device.memory.sample surface
// ggml_device automatically without coupling core to optional backend modules.
void register_ggml_device_telemetry(const GgmlDeviceTelemetry& entry);
std::vector<GgmlDeviceTelemetry> registered_ggml_device_telemetry();
void clear_registered_ggml_device_telemetry() noexcept;
void merge_ggml_device_telemetry(std::vector<GpuDeviceInfo>& gpus,
                                 const std::vector<GgmlDeviceTelemetry>& ggml_devices);
std::optional<bool> classify_vulkan_rebar(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept;
MemoryReading vulkan_device_local_budget_bytes(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept;
MemoryReading vulkan_device_local_usage_bytes(const std::vector<VulkanMemoryHeapInfo>& heaps) noexcept;
std::optional<std::uint32_t> parse_pcie_link_speed_gen(std::string_view speed_text) noexcept;
std::string parse_linux_aspm_policy(std::string_view policy_text);
std::string format_linux_link_aspm(std::optional<bool> l0s, std::optional<bool> l1,
                                   std::optional<bool> l1_1 = std::nullopt,
                                   std::optional<bool> l1_2 = std::nullopt);
std::string_view windows_aspm_policy_name(std::uint32_t power_index) noexcept;
std::string sanitize_driver_build_string(std::string_view raw);
std::string format_driver_build_string(std::string_view driver_version, std::string_view release_version = {},
                                       std::string_view vulkan_driver_info = {});

// Pure Linux amdgpu sysfs parser (/sys/class/drm/<card_id>/device/*).
// Compiled on all platforms so it is testable everywhere via an injected FileReader.
std::optional<GpuDeviceInfo> parse_amdgpu_sysfs_device(std::string_view card_id, const FileReader& read_file);
std::optional<GpuDeviceInfo> parse_amdgpu_sysfs_device(std::string_view card_id, const FileReader& read_file,
                                                       bool rocm_available);

// Pure ROCm/HIP device visibility & runtime ordinal mapping for AMD adapter
// ordinal `amd_ordinal` given optional HIP_VISIBLE_DEVICES and
// ROCR_VISIBLE_DEVICES values (std::nullopt when unset) and optional ROCm
// device UUID (e.g. "GPU-0123456789abcdef"). Empty, "-1", "none", or
// "NoDevFiles" disables ROCm; comma-separated integer lists filter and remap
// ordinals (ROCR applied before HIP when both are set); UUID tokens fail closed
// unless `device_uuid` is supplied and matches a token.
std::optional<std::size_t> map_rocm_device_ordinal(
    std::size_t amd_ordinal, std::optional<std::string_view> hip_visible_devices = std::nullopt,
    std::optional<std::string_view> rocr_visible_devices = std::nullopt,
    std::optional<std::string_view> device_uuid = std::nullopt) noexcept;

bool is_rocm_device_visible(std::size_t amd_ordinal,
                            std::optional<std::string_view> hip_visible_devices = std::nullopt,
                            std::optional<std::string_view> rocr_visible_devices = std::nullopt,
                            std::optional<std::string_view> device_uuid = std::nullopt) noexcept;

// Pure Windows DXGI + NVML merger: preserves all DXGI adapters and appends any
// NVML NVIDIA GPUs missing from DXGI (such as TCC-mode / headless cards on a
// mixed WDDM + TCC host) without duplicating WDDM cards already enumerated.
void merge_win32_nvml_devices(std::vector<GpuDeviceInfo>& gpus, const std::vector<GpuDeviceInfo>& nvml_gpus);

// Pure Windows GPU backend hint resolver: "cuda" for NVIDIA, "rocm" for
// discrete AMD RX XT when AMD HIP SDK is installed and the device is visible
// (else "vulkan"), "vulkan" for AMD Ryzen iGPU/APU UMA and Intel, "metal" for
// Apple, "" for unknown.
std::string resolve_windows_gpu_backend_hint(
    GpuVendor vendor, bool is_integrated_uma, bool hip_sdk_available = false, std::size_t amd_ordinal = 0,
    std::optional<std::string_view> hip_visible_devices = std::nullopt,
    std::optional<std::string_view> rocr_visible_devices = std::nullopt);

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

// Matches a single device-hint token ("CUDA0", "ROCm0", "rocm:0", "HIP0",
// "Vulkan1", "gpu:0", "uma", "igpu", "apu", "nvidia", "amd", "intel", "apple",
// numeric index, or exact GPU id) against `gpus`.
std::optional<std::size_t> match_single_gpu_hint(const std::vector<GpuDeviceInfo>& gpus,
                                                 std::string_view token,
                                                 bool prefer_free = false,
                                                 bool require_known_vram = false) noexcept;

// Classification of the selected GPU pool for UMA-aware spill and residency guards.
enum class UmaSelectionMode {
    discrete_only,  // No integrated UMA GPU selected (pure discrete or no GPU)
    uma_only,       // Every selected GPU is integrated UMA (pure APU / iGPU)
    hybrid          // Mixed discrete + integrated UMA pool (e.g. Vulkan0,Vulkan1)
};

// Normalizes a UMA mode string ("on"/"true" -> "on", "off"/"false" -> "off",
// "auto" -> "auto"), returning nullopt for any other string.
std::optional<std::string_view> normalize_uma_mode(std::string_view mode) noexcept;

// True when `mode` is one of "on", "off", "auto", "true", "false".
bool is_valid_uma_mode(std::string_view mode) noexcept;

// Extracts the last --device / -dev argument value from `args`, or empty view.
std::string_view extract_device_hint(const std::vector<std::string>& args) noexcept;

// Pure UMA pool classifier:
//  - "on" / "true"   -> UmaSelectionMode::uma_only
//  - "off" / "false" -> UmaSelectionMode::discrete_only
//  - "auto"          ->
//      * Empty `device_hint`: `uma_only` when `gpus` is non-empty and every
//        discovered GPU is integrated UMA; otherwise `discrete_only` (discrete
//        GPUs win when unhinted).
//      * Non-empty `device_hint`: parses comma-separated device tokens. Any
//        unmatched token fails closed to `discrete_only`. Otherwise returns
//        `uma_only` when all selected GPUs are integrated UMA, `hybrid` when
//        both discrete and integrated UMA GPUs are selected (e.g.
//        "Vulkan0,Vulkan1"), and `discrete_only` when only discrete GPUs are
//        selected.
UmaSelectionMode resolve_uma_selection(std::string_view mode_or_auto,
                                       const std::vector<GpuDeviceInfo>& gpus,
                                       std::string_view device_hint = {}) noexcept;

// Pure UMA mode resolver: true when resolve_uma_selection(...) != discrete_only
// (pure integrated UMA or hybrid discrete+iGPU pool).
bool resolve_uma_mode(std::string_view mode_or_auto,
                      const std::vector<GpuDeviceInfo>& gpus,
                      std::string_view device_hint = {}) noexcept;

// Summary of discrete, integrated UMA, and combined GPU VRAM pools on a host,
// including hybrid discrete + integrated UMA Vulkan spread eligibility (e.g.
// discrete RX 9060 XT + integrated Radeon 780M addressed jointly via
// Vulkan0,Vulkan1). Pure value type computed from GpuDeviceInfo + HostMemory.
struct GpuPoolSummary {
    std::uint64_t discrete_vram_mib = 0;
    std::uint64_t integrated_uma_vram_mib = 0;
    std::uint64_t combined_gpu_vram_mib = 0;
    std::uint64_t discrete_free_vram_mib = 0;
    std::uint64_t integrated_uma_free_vram_mib = 0;
    std::uint64_t combined_free_gpu_vram_mib = 0;
    std::size_t discrete_gpu_count = 0;
    std::size_t integrated_uma_gpu_count = 0;
    bool hybrid_vulkan_pool_available = false;
    // Ordered Vulkan device names ("Vulkan0", "Vulkan1", ...) placing discrete
    // GPU(s) before integrated UMA GPU(s) when hybrid_vulkan_pool_available is true.
    std::vector<std::string> hybrid_vulkan_devices;
};

// Summarizes discrete, integrated UMA, and combined GPU VRAM pools in MiB.
// When `host` physical memory readings are known, integrated UMA pool totals
// are capped at host total/available physical RAM so the combined pool never
// overstates what physical memory can back.
GpuPoolSummary summarize_gpu_pools(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host = {});

// Convenience helper returning the combined effective GPU VRAM pool in MiB
// across discrete and integrated UMA GPUs (honoring host RAM caps for UMA).
std::uint64_t combined_gpu_vram_mib(const std::vector<GpuDeviceInfo>& gpus, bool prefer_free = false,
                                    const HostMemory& host = {}) noexcept;

// Current OS/arch label, e.g. "windows-x86_64".
std::string host_platform();
// Host name used as the telemetry producer node_id default.
std::string host_name();

}  // namespace sonder::inference
