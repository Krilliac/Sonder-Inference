// Sonder Inference: backend runtime status (GPU memory, context fit and
// performance warnings) observed by a backend while it runs.
//
// Backend-neutral and optional: Backend::runtime_status() returns nullopt by
// default, and hosts (the HTTP server, the engine's periodic sampler) add
// these fields to their output only when a backend reports them. The first
// producer is the supervised llama-server backend (docs/integration/
// vram-spill.md); the struct is not shaped around it.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/json.hpp"

namespace sonder::inference {

// One structured performance warning. `code` is a stable identifier
// (e.g. "kv_kernel_f16_fallback"); `message` is human-readable.
struct BackendWarning {
    std::string code;
    std::string severity = "warning";  // "warning" or "info"
    std::string source;                // "config", "log" or "gpu_probe"
    std::string message;
    std::vector<std::pair<std::string, std::string>> details;  // ordered key/value facts
    std::uint64_t count = 1;           // occurrences folded into this warning
};

// Per-process GPU memory as measured by the backend's probe. Byte values are
// meaningful only when `status` is "ok".
struct GpuMemoryStatus {
    std::string probe;   // e.g. "pdh" (Windows GPU Process Memory counters) or "none"
    std::string status;  // "ok", "not_sampled", "unsupported" or "error"
    std::string error;   // set when status == "error" or "unsupported"
    std::uint64_t dedicated_bytes = 0;
    std::uint64_t shared_bytes = 0;
    std::uint64_t peak_shared_bytes = 0;
    std::uint64_t shared_baseline_bytes = 0;
    std::uint64_t spill_threshold_bytes = 0;
    bool spilled = false;
    std::uint64_t samples = 0;  // successful samples since the process started
};

// Context size the backend runs with, and whether it was reduced to fit.
struct ContextFitStatus {
    std::string policy;                      // "warn", "refuse" or "auto_fit"
    std::optional<std::uint64_t> configured_ctx;  // requested context (unknown when unset)
    std::optional<std::uint64_t> fitted_ctx;      // context of the running process
    std::uint64_t fit_attempts = 0;          // restarts with a smaller context
    std::string outcome;  // "not_needed", "fitted", "refused", "floor_reached" or "exhausted"
};

struct BackendRuntimeStatus {
    GpuMemoryStatus gpu_memory;
    ContextFitStatus context;
    std::vector<BackendWarning> warnings;
};

// Stable JSON shape shared by /v1/sonder/health, /v1/models and telemetry:
// {"gpu_memory":{...},"context":{...},"warnings":[...]}.
[[nodiscard]] json::Object to_json(const BackendRuntimeStatus& status);
[[nodiscard]] json::Object to_json(const GpuMemoryStatus& status);
[[nodiscard]] json::Object to_json(const BackendWarning& warning);

}  // namespace sonder::inference
