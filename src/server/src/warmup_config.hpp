#pragma once

#include "sonder/inference/backend_setup.hpp"

namespace sonder::inference::detail {

// Parses the value of the llamaserver JSON "warmup" field into setup.
// Validation errors are ordinary invalid_argument usage errors.
Status parse_warmup_config(const json::Value& value, BackendSetup& setup);

}  // namespace sonder::inference::detail
