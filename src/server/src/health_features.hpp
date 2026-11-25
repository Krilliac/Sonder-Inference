// Internal: additive feature discovery for the actual bound backend.
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata);

json::Object health_features(std::string_view backend, const BackendCapabilities& capabilities,
                             const ServerOptions& options, json::Object metadata);

json::Object health_features(const std::shared_ptr<Backend>& backend, const ServerOptions& options,
                             json::Object metadata);

// Additive top-level capability map for `/v1/sonder/health` and
// `/v1/sonder/identity` (auto-detected by Sonder-runtime).
json::Object server_capabilities(const std::shared_ptr<Backend>& backend, const ServerOptions& options);

// Additive GPU inventory JSON array for `/v1/sonder/health`.
json::Array gpu_inventory_json(const std::vector<GpuDeviceInfo>& gpus);

}  // namespace sonder::inference::server::detail
