// Server-wide thinking pin application. Kept separate from request-path
// parsing so callers can preserve the legacy override policy or opt into
// request values taking precedence.
#pragma once

#include <string>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail::thinking_pins {

// Apply `pins` to `request`. In override_request mode an explicit conflicting
// request value is replaced and produces the historical warning. In
// default_value mode pins only fill fields that the request left unset.
[[nodiscard]] std::vector<std::string> apply(PinMode mode, ThinkingOptions& request,
                                             const ThinkingOptions& pins);

}  // namespace sonder::inference::server::detail::thinking_pins
