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

json::Object health_features_impl(std::string_view backend, const BackendCapabilities* capabilities,
                                  const ServerOptions& options, json::Object metadata) {
    json::Array features;
    if (backend == "llamaserver") {
        // Preserve the exact legacy 7-feature contract for llamaserver.
        for (const auto* name : {"thinking", "chat_template_kwargs", "enable_thinking", "reasoning_effort",
                                 "reasoning_budget", "prompt_cache_key", "priority_classes"}) {
            features.emplace_back(name);
        }
    } else if (!backend.empty() && backend != "mock") {
        // Derive non-mock backend features from server capabilities and backend Capability flags.
        features.emplace_back("native_tools");
        if (capabilities != nullptr && capabilities->has(Capability::tool_calling)) {
            features.emplace_back("backend_native_tools");
        }
        features.emplace_back("responses_api");
        features.emplace_back("priority_classes");
        if (capabilities != nullptr) {
            if (capabilities->has(Capability::streaming)) features.emplace_back("streaming");
            if (capabilities->has(Capability::tokenization)) features.emplace_back("tokenization");
            if (capabilities->has(Capability::batched_prefill)) features.emplace_back("batched_prefill");
            if (capabilities->has(Capability::continuous_batch_decode)) {
                features.emplace_back("continuous_batch_decode");
            }
            if (capabilities->has(Capability::prefix_reuse)) {
                features.emplace_back("prefix_reuse");
                features.emplace_back("prompt_cache_key");
            }
            if (capabilities->has(Capability::speculative_decode)) features.emplace_back("speculative_decode");
            if (capabilities->has(Capability::structured_output)) features.emplace_back("structured_output");
            if (capabilities->has(Capability::embeddings)) features.emplace_back("embeddings");
            if (capabilities->has(Capability::logprobs)) features.emplace_back("logprobs");
            if (capabilities->has(Capability::multimodal)) features.emplace_back("multimodal");
            if (capabilities->has(Capability::recurrent_checkpoint)) features.emplace_back("recurrent_checkpoint");
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

}  // namespace

json::Object health_features(std::string_view backend, const ServerOptions& options, json::Object metadata) {
    return health_features_impl(backend, nullptr, options, std::move(metadata));
}

json::Object health_features(std::string_view backend, const BackendCapabilities& capabilities,
                             const ServerOptions& options, json::Object metadata) {
    return health_features_impl(backend, &capabilities, options, std::move(metadata));
}

json::Object health_features(const std::shared_ptr<Backend>& backend, const ServerOptions& options,
                             json::Object metadata) {
    if (!backend) {
        return health_features_impl("", nullptr, options, std::move(metadata));
    }
    const BackendCapabilities caps = backend->capabilities();
    return health_features_impl(backend->name(), &caps, options, std::move(metadata));
}

json::Object server_capabilities(const std::shared_ptr<Backend>& backend, const ServerOptions& options) {
    const BackendCapabilities bc = backend ? backend->capabilities() : BackendCapabilities{};
    return json::Object{
        {"tools", true},
        {"native_tools", true},
        {"backend_native_tools", bc.has(Capability::tool_calling)},
        {"tool_choice", true},
        {"responses_api", true},
        {"response_store", !options.disable_response_store && options.response_history_ttl.count() > 0},
        {"sse_keepalive", options.stream_keepalive_interval.count() > 0},
        {"structured_output", bc.has(Capability::structured_output)},
        {"multimodal", bc.has(Capability::multimodal)},
        {"logprobs", bc.has(Capability::logprobs)},
        {"embeddings", bc.has(Capability::embeddings)}};
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
