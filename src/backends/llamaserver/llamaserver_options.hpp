// Internal: LlamaServerBackendOptions validation, supervisor option mapping,
// and bearer API-key loading from environment variables.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "adaptive_speculation.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "supervisor.hpp"

namespace sonder::inference::llamaserver {

using sonder::inference::extract_device_hint;
using sonder::inference::is_valid_uma_mode;
using sonder::inference::resolve_uma_mode;
using sonder::inference::resolve_uma_selection;
using sonder::inference::UmaSelectionMode;

// Parse a JSON boolean (true/false) or string ("auto", "on", "off", "true",
// "false") into the canonical UMA mode setting ("on", "off", or "auto").
Status parse_uma_mode_json(const json::Value &value, std::string &out);

SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o);
SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o,
                                         const std::vector<std::string> &args);
SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o,
                                         const std::vector<std::string> &args,
                                         const std::vector<GpuDeviceInfo> &gpus);

// False when the policy name is unknown; `out` is then left partly filled.
bool to_host_memory_options(const LlamaServerHostMemoryOptions &o, HostMemoryOptions &out);

AdaptiveSpeculationConfig to_adaptive_speculation_config(const LlamaServerAdaptiveSpeculationOptions &o) noexcept;

// Reads the bearer token from the environment variable `name`. Messages name
// the variable, never the value. Refuses an unset/empty variable, a malformed
// variable name, and a value that is not a single visible-ASCII token.
Result<std::string> read_api_key(const std::string &name);

Status validate_backend_options(const LlamaServerBackendOptions &options);

}  // namespace sonder::inference::llamaserver
