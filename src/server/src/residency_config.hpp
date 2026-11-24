#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/device.hpp"

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "sonder/inference/backends/llamaserver.hpp"
#endif

namespace sonder::inference::detail {

// Parse a JSON boolean (true/false) or string ("auto", "on", "off", "true",
// "false") into the canonical UMA mode setting ("on", "off", or "auto").
Status parse_uma_mode_value(const json::Value &value, const char *field_path, std::string &out);

// True when `mode` is one of "on", "off", "auto", "true", "false".
bool is_valid_uma_mode(std::string_view mode) noexcept;

// Extracts the last --device / -dev argument value from `args`, or empty view.
std::string_view extract_device_hint(const std::vector<std::string> &args) noexcept;

// Pure UMA mode resolver:
//  - "on" / "true"   -> true
//  - "off" / "false" -> false
//  - "auto"          -> true only when `gpus` is non-empty and the selected
//                       GPU(s) (honoring `device_hint`, or preferring discrete
//                       over integrated UMA when unset) are all integrated UMA.
bool resolve_uma_mode(std::string_view mode_or_auto,
                      const std::vector<GpuDeviceInfo> &gpus,
                      std::string_view device_hint = {}) noexcept;

// Parse one spill_guard.residency object into the server-only config mirror.
// Parsing remains available in builds without the optional backend API.
Status parse_residency_config(const json::Value &value, BackendSetup &setup);

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
// Map the server-only mirror to the optional backend API, with checked MiB
// conversion and validation for callers that construct BackendSetup directly.
Status apply_residency_config(const BackendSetup &setup, LlamaServerResidencyGuardOptions &out);
Status apply_residency_config(const BackendSetup &setup,
                              LlamaServerResidencyGuardOptions &out,
                              const std::vector<GpuDeviceInfo> &gpus);
#endif

}  // namespace sonder::inference::detail
