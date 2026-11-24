#include "health_features.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace sonder::inference::server::detail {

namespace {

constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;

json::Value reading_mib_json(const MemoryReading& reading) {
    if (!reading.is_known()) {
        return json::Value(nullptr);
    }
    return json::Value(reading.bytes_or(0) / kBytesPerMiB);
}

json::Value optional_u64_json(std::optional<std::uint64_t> value) {
    return value ? json::Value(*value) : json::Value(nullptr);
}

}  // namespace

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

json::Array gpu_inventory_json(const std::vector<GpuDeviceInfo>& gpus) {
    json::Array list;
    list.reserve(gpus.size());
    for (const auto& g : gpus) {
        list.emplace_back(json::Object{
            {"id", g.id},
            {"name", g.name},
            {"vendor", to_string(g.vendor)},
            {"vendor_id", g.vendor_id},
            {"device_id", g.device_id},
            {"dedicated_total_mib", reading_mib_json(g.dedicated_total)},
            {"dedicated_free_mib", reading_mib_json(g.dedicated_free)},
            {"shared_total_mib", reading_mib_json(g.shared_total)},
            {"effective_vram_mib", optional_u64_json(gpu_effective_vram_mib(g, false))},
            {"effective_free_vram_mib", optional_u64_json(gpu_effective_vram_mib(g, true))},
            {"is_integrated_uma", g.is_integrated_uma},
            {"backend_hint", g.backend_hint},
            {"source", g.source}});
    }
    return list;
}

}  // namespace sonder::inference::server::detail
