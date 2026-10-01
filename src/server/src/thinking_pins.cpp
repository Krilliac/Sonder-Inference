#include "thinking_pins.hpp"

namespace sonder::inference::server::detail::thinking_pins {

namespace {

const char* on_off(const bool value) { return value ? "on" : "off"; }

}  // namespace

std::vector<std::string> apply(const PinMode mode, ThinkingOptions& request, const ThinkingOptions& pins) {
    std::vector<std::string> warnings;
    const bool override = mode == PinMode::override_request;

    if (pins.enable_thinking) {
        if (override && request.enable_thinking && *request.enable_thinking != *pins.enable_thinking) {
            warnings.push_back(std::string("enable_thinking=") + on_off(*request.enable_thinking) +
                               " was overridden by the server pin (" + on_off(*pins.enable_thinking) +
                               ") that keeps the prompt prefix cacheable");
        }
        if (override || !request.enable_thinking) request.enable_thinking = pins.enable_thinking;
    }
    if (pins.reasoning_effort) {
        if (override && request.reasoning_effort && *request.reasoning_effort != *pins.reasoning_effort) {
            warnings.push_back("reasoning_effort=" + *request.reasoning_effort +
                               " was overridden by the server pin (" + *pins.reasoning_effort +
                               ") that keeps the prompt prefix cacheable");
        }
        if (override || !request.reasoning_effort) request.reasoning_effort = pins.reasoning_effort;
    }
    return warnings;
}

}  // namespace sonder::inference::server::detail::thinking_pins
