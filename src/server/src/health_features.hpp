// Internal: additive feature discovery for the actual bound backend.
#pragma once

#include <string_view>
#include <vector>

#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata);

// Additive GPU inventory JSON array for `/v1/sonder/health`.
json::Array gpu_inventory_json(const std::vector<GpuDeviceInfo>& gpus);

}  // namespace sonder::inference::server::detail
