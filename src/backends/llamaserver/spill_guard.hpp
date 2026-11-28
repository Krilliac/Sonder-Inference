#pragma once

#include <chrono>
// Relative path on purpose: sonder-infer has no src/ include directory.
#include "../../devices/gpu_memory.hpp"
#include "sonder/inference/backends/llamaserver_residency.hpp"

namespace sonder::inference::llamaserver {

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
    // Integrated UMA GPUs (e.g. Ryzen APUs) legitimately allocate GPU buffers
    // in shared/GTT system memory; when true (and not hybrid_uma_mode),
    // is_spilled() never flags shared usage as a PCIe VRAM spill.
    bool uma_mode = false;
    // Hybrid discrete + integrated UMA pool (e.g. --device Vulkan0,Vulkan1 on
    // discrete RX 9060 XT + integrated Radeon 780M). When per-adapter counter
    // classification is available, is_spilled() evaluates ONLY the discrete
    // adapter(s)' shared usage against baseline + threshold, ignoring
    // intentional shared allocation on integrated UMA adapters. When only
    // aggregate unclassified counters are available, hybrid_uma_mode suppresses
    // false-positive spills caused by intentional UMA shared memory.
    bool hybrid_uma_mode = false;
};

Status validate_spill_guard(const SpillGuardOptions &options);

// baseline_bytes + baseline_bytes_per_1k_ctx * ctx / 1024 (saturating); the
// fixed baseline when the context size is unknown.
std::uint64_t effective_baseline(const SpillGuardOptions &options, std::optional<std::uint64_t> ctx);

// Shared bytes attributable to discrete-GPU VRAM spill under `options`.
// Returns 0 for pure UMA mode (or hybrid mode without per-adapter separation),
// `sample.discrete_shared_bytes` when per-adapter classification distinguishes
// discrete adapters from integrated UMA adapters, and `sample.shared_bytes`
// otherwise.
std::uint64_t effective_spill_shared_bytes(const GpuMemorySample &sample,
                                           const SpillGuardOptions &options) noexcept;

bool is_spilled(const GpuMemorySample &sample, const SpillGuardOptions &options,
                std::optional<std::uint64_t> ctx = std::nullopt);

// The next smaller context for auto_fit, or nullopt when `current` is already
// at (or below) the floor. Strictly decreasing, so repeated application
// terminates.
std::optional<std::uint64_t> next_fit_context(std::uint64_t current, const SpillGuardOptions &options);

} // namespace sonder::inference::llamaserver
