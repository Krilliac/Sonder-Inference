// Internal: additive feature discovery for the actual bound backend.
#pragma once

#include <string_view>

#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata);

}  // namespace sonder::inference::server::detail
