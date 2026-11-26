// Internal: additive feature discovery for the actual bound backend.
#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server::detail {

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata);

json::Object health_features(std::string_view backend, const BackendCapabilities& capabilities,
                             const ServerOptions& options, json::Object metadata);

json::Object health_features(const std::shared_ptr<Backend>& backend, const ServerOptions& options,
                             json::Object metadata);

// Additive top-level capability map for `/v1/sonder/health` and
// `/v1/sonder/identity` (auto-detected by Sonder-runtime).
json::Object server_capabilities(const std::shared_ptr<Backend>& backend, const ServerOptions& options);

// Structured 5-state capability matrix (adapter_supports, build_supports,
// configured, observed_active, qualified_beneficial — never conflating a
// boolean capability flag with measured speedup; unmeasured states remain
// "unknown").
json::Object capability_status_matrix(const std::shared_ptr<Backend>& backend, const ServerOptions& options,
                                      const std::optional<BackendRuntimeStatus>& runtime = std::nullopt);

// Additive diagnostics object for `/v1/sonder/health` exposing the 5-state
// capability matrix alongside the bounded effective execution report without
// breaking existing top-level or `sonder` metadata fields.
json::Object effective_execution_diagnostics(const std::shared_ptr<Backend>& backend, const ServerOptions& options,
                                             const std::optional<BackendRuntimeStatus>& runtime = std::nullopt);

// Additive GPU inventory JSON array for `/v1/sonder/health`.
json::Array gpu_inventory_json(const std::vector<GpuDeviceInfo>& gpus);

}  // namespace sonder::inference::server::detail
