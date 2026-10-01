// Per-process GPU memory probe and VRAM-spill classification for a
// supervised llama-server child (docs/integration/vram-spill.md).
//
// On Windows (WDDM) an allocation that no longer fits in dedicated VRAM is
// placed in shared system memory instead of failing, so a llama-server run
// with `--fit off` can load "successfully" and then run 2-15x slower.
// nvidia-smi reports the same ~15.8 GB whether or not anything spilled; the
// reliable signal is the per-process PDH counters
// `\GPU Process Memory(pid_<PID>_*)\Shared Usage` and `Dedicated Usage`,
// summed over the process's adapter instances.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/error.hpp"
#include "sonder/inference/backends/llamaserver_residency.hpp"

namespace sonder::inference::llamaserver {

inline constexpr std::uint64_t kMiB = 1024ull * 1024ull;

// One counter instance, e.g. {"pid_1234_luid_0x00000000_0x0000D1F5_phys_0", 351272960}.
struct CounterInstance {
    std::string name;
    std::uint64_t value = 0;
};

struct GpuProcessCounters {
    std::vector<CounterInstance> dedicated;  // "Dedicated Usage" instances
    std::vector<CounterInstance> shared;     // "Shared Usage" instances
};

// Source of raw counter instances for a process. Injected in tests; the
// production source reads PDH on Windows and is unsupported elsewhere.
class GpuCounterSource {
  public:
    virtual ~GpuCounterSource() = default;
    // Short identifier reported as GpuMemoryStatus::probe ("pdh", "none", ...).
    [[nodiscard]] virtual std::string name() const = 0;
    // False when the platform has no per-process GPU memory counters.
    [[nodiscard]] virtual bool supported() const = 0;
    // Reads the instances whose names start with "pid_<pid>_". Sources may
    // return extra instances; summarize_gpu_counters() filters them.
    virtual Result<GpuProcessCounters> read(std::uint32_t pid) = 0;
};

// Windows: PDH `GPU Process Memory`. Elsewhere: an unsupported source whose
// read() returns ErrorCode::unsupported (NVML is a possible later addition).
std::shared_ptr<GpuCounterSource> make_gpu_counter_source();

struct GpuMemorySample {
    std::uint64_t dedicated_bytes = 0;
    std::uint64_t shared_bytes = 0;
    std::size_t instances = 0;  // matching adapter instances (max over the two counters)
};

// Sums the instances that belong to `pid` exactly ("pid_12_..." never
// matches pid 1 or pid 123). Saturates instead of wrapping.
GpuMemorySample summarize_gpu_counters(std::uint32_t pid, const GpuProcessCounters &counters);

// True when `name` is an instance of process `pid` ("pid_<pid>_" prefix).
bool is_process_instance(std::string_view name, std::uint32_t pid);

// Spill guard configuration. The default threshold comes from the 2026-09-29
// RTX 5070 Ti (16 GB) measurements (docs/integration/vram-spill.md): clean
// Qwen3.8-27B Q3_K_XL runs kept 148-198 MiB in shared memory (CUDA context and
// pinned host buffers) at 49k-100k context, while every run whose dedicated
// usage had reached the ~15.2 GB ceiling showed 292-388 MiB (12-19% slower,
// 3x slower at 388 MiB) and the IQ4_XS spills 0.71-2.1 GB (2-15x slower).
// 256 MiB sits between the largest clean (198) and smallest spilled (292)
// sample.
enum class SpillPolicy { warn, refuse, auto_fit };
const char *to_string(SpillPolicy policy) noexcept;
std::optional<SpillPolicy> parse_spill_policy(std::string_view text) noexcept;

struct SpillGuardOptions {
    bool enabled = true;
    SpillPolicy policy = SpillPolicy::warn;
    // Spilled when shared usage exceeds baseline + threshold. The baseline is
    // an absolute byte count (default 0): a sample taken after the model has
    // loaded already contains any load-time spill, so it cannot serve as its
    // own baseline.
    std::uint64_t threshold_bytes = 256 * kMiB;
    std::uint64_t baseline_bytes = 0;
    // Optional growth of the clean baseline with context size, added per
    // 1,024 tokens of the child's --ctx-size (default 0 = fixed baseline).
    // Clean shared usage rises with context: about 1 MiB per 1k ctx without
    // speculation and about 2 MiB per 1k ctx (+~40 MiB after the first
    // prompt) with MTP (--spec-type draft-mtp), measured 2026-09-30.
    std::uint64_t baseline_bytes_per_1k_ctx = 0;
    // Periodic re-sampling while the child runs (monitor thread only).
    std::chrono::milliseconds sample_interval{5000};
    // auto_fit: next = align_down(ctx * step_factor, step_align), always at
    // least one step_align below ctx, never below min_ctx.
    double step_factor = 0.85;
    std::uint64_t step_align = 1024;
    std::uint64_t min_ctx = 8192;
    std::size_t max_attempts = 4;
    LlamaServerResidencyGuardOptions residency;
};

Status validate_spill_guard(const SpillGuardOptions &options);

// baseline_bytes + baseline_bytes_per_1k_ctx * ctx / 1024 (saturating); the
// fixed baseline when the context size is unknown.
std::uint64_t effective_baseline(const SpillGuardOptions &options, std::optional<std::uint64_t> ctx);

bool is_spilled(const GpuMemorySample &sample, const SpillGuardOptions &options,
                std::optional<std::uint64_t> ctx = std::nullopt);

// The next smaller context for auto_fit, or nullopt when `current` is already
// at (or below) the floor. Strictly decreasing, so repeated application
// terminates.
std::optional<std::uint64_t> next_fit_context(std::uint64_t current, const SpillGuardOptions &options);

} // namespace sonder::inference::llamaserver
