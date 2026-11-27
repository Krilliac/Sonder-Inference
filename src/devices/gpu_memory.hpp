// Per-process GPU memory probe and VRAM-spill classification for a
// process (native or supervised; docs/integration/vram-spill.md).
//
// On Windows (WDDM) an allocation that no longer fits in dedicated VRAM is
// placed in shared system memory instead of failing, so a llama-server run
// with `--fit off` can load "successfully" and then run 2-15x slower.
// nvidia-smi reports the same ~15.8 GB whether or not anything spilled; the
// reliable signal is the per-process PDH counters
// `\GPU Process Memory(pid_<PID>_*)\Shared Usage` and `Dedicated Usage`,
// summed over the process's adapter instances.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

inline constexpr std::uint64_t kMiB = 1024ull * 1024ull;

// Per-adapter classification carried by counter instances when the counter
// source (e.g. Windows PDH + DXGI LUID matching, or Linux NVML) can distinguish
// discrete GPUs from integrated UMA GPUs.
enum class CounterAdapterKind {
    unknown,
    discrete,
    integrated_uma
};

// One counter instance, e.g. {"pid_1234_luid_0x00000000_0x0000D1F5_phys_0", 351272960}.
struct CounterInstance {
    std::string name;
    std::uint64_t value = 0;
    CounterAdapterKind adapter_kind = CounterAdapterKind::unknown;
};

struct AdapterLuid {
    std::uint32_t high = 0;
    std::uint32_t low = 0;
    friend constexpr bool operator==(const AdapterLuid &, const AdapterLuid &) noexcept = default;
};

struct AdapterLuidInfo {
    AdapterLuid luid;
    bool is_integrated_uma = false;
};

struct GpuProcessCounters {
    std::vector<CounterInstance> dedicated;  // "Dedicated Usage" instances
    std::vector<CounterInstance> shared;     // "Shared Usage" instances
};

// Source of raw counter instances for a process. Injected in tests; the
// production source reads PDH on Windows, DRM fdinfo / optional NVML on Linux,
// and is unsupported elsewhere (or on Linux hosts without amdgpu/NVML).
class GpuCounterSource {
  public:
    virtual ~GpuCounterSource() = default;
    // Short identifier reported as GpuMemoryStatus::probe ("pdh", "amdgpu-fdinfo",
    // "nvml", "amdgpu-fdinfo+nvml", "none", ...).
    [[nodiscard]] virtual std::string name() const = 0;
    // False when the platform has no per-process GPU memory counters.
    [[nodiscard]] virtual bool supported() const = 0;
    // Reads the instances whose names start with "pid_<pid>_". Sources may
    // return extra instances; summarize_gpu_counters() filters them.
    virtual Result<GpuProcessCounters> read(std::uint32_t pid) = 0;
};

// Windows: PDH `GPU Process Memory`. Linux: amdgpu `/proc/<pid>/fdinfo/*` and/or
// dynamic `libnvidia-ml.so.1` when present, falling back cleanly to an
// unsupported source (`"none"`) when neither is present. Elsewhere: unsupported.
std::shared_ptr<GpuCounterSource> make_gpu_counter_source();

// Parses Linux DRM `/proc/<pid>/fdinfo/*` text (single entry or concatenated
// entries) for `drm-memory-vram:` and `drm-memory-gtt:` lines (`B`, `KiB`,
// `MiB`, `GiB`). Deduplicates repeated descriptors sharing the same
// `drm-client-id:` (scoped by `drm-driver:` when present). Compiled on all
// platforms so it is testable everywhere.
bool parse_linux_drm_fdinfo(std::string_view fdinfo_text, std::uint64_t &vram_bytes, std::uint64_t &gtt_bytes);

// Extracts the WDDM adapter LUID from a Windows PDH GPU Process Memory
// instance name (e.g. "pid_1234_luid_0x00000000_0x000104A4_phys_0"). Compiled
// on all platforms so per-adapter attribution is testable everywhere.
std::optional<AdapterLuid> parse_pdh_adapter_luid(std::string_view instance_name) noexcept;

// Pure helper: tags CounterInstance::adapter_kind on matching PDH LUID
// instances using `adapters`. Instances whose LUID is not in `adapters` remain
// unchanged.
void classify_pdh_counters(GpuProcessCounters &counters, const std::vector<AdapterLuidInfo> &adapters) noexcept;

struct GpuMemorySample {
    std::uint64_t dedicated_bytes = 0;
    std::uint64_t shared_bytes = 0;
    std::size_t instances = 0;  // matching adapter instances (max over the two counters)
    // Per-adapter breakdown when counter instances are classified into discrete
    // vs integrated UMA adapters. `has_adapter_classification` is true when at
    // least one matching instance is classified and no unclassified instance
    // reports non-zero usage.
    std::uint64_t discrete_dedicated_bytes = 0;
    std::uint64_t discrete_shared_bytes = 0;
    std::uint64_t uma_dedicated_bytes = 0;
    std::uint64_t uma_shared_bytes = 0;
    std::size_t discrete_instances = 0;
    std::size_t uma_instances = 0;
    bool has_adapter_classification = false;
};

// Sums the instances that belong to `pid` exactly ("pid_12_..." never
// matches pid 1 or pid 123). Saturates instead of wrapping.
GpuMemorySample summarize_gpu_counters(std::uint32_t pid, const GpuProcessCounters &counters);

// True when `name` is an instance of process `pid` ("pid_<pid>_" prefix).
bool is_process_instance(std::string_view name, std::uint32_t pid);

} // namespace sonder::inference::llamaserver
