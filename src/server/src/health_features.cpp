#include "health_features.hpp"

namespace sonder::inference::server::detail {

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata) {
    json::Array features;
    if (backend == "llamaserver") {
        for (const auto* name : {"thinking", "chat_template_kwargs", "enable_thinking", "reasoning_effort",
                                 "reasoning_budget", "prompt_cache_key", "priority_classes"}) {
            features.emplace_back(name);
        }
    }
    metadata.set("features", std::move(features));
    json::Object pins{{"mode", options.pin_mode == PinMode::default_value ? "default" : "override"}};
    if (options.pin_enable_thinking) pins.set("enable_thinking", *options.pin_enable_thinking);
    if (options.pin_reasoning_effort) pins.set("reasoning_effort", *options.pin_reasoning_effort);
    if (options.pin_reasoning_budget_tokens) pins.set("reasoning_budget_tokens", *options.pin_reasoning_budget_tokens);
    if (options.pin_reasoning_budget_message) pins.set("reasoning_budget_message", *options.pin_reasoning_budget_message);
    metadata.set("pins", std::move(pins));
    return metadata;
}

}  // namespace sonder::inference::server::detail
