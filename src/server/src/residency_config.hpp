#pragma once

#include "sonder/inference/backend_setup.hpp"

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "sonder/inference/backends/llamaserver.hpp"
#endif

namespace sonder::inference::detail {

// Parse one spill_guard.residency object into the server-only config mirror.
// Parsing remains available in builds without the optional backend API.
Status parse_residency_config(const json::Value &value, BackendSetup &setup);

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
// Map the server-only mirror to the optional backend API, with checked MiB
// conversion and validation for callers that construct BackendSetup directly.
Status apply_residency_config(const BackendSetup &setup, LlamaServerResidencyGuardOptions &out);
#endif

}  // namespace sonder::inference::detail
