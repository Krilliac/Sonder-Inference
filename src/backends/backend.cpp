#include "sonder/inference/backend.hpp"

#include <utility>

namespace sonder::inference {

std::vector<std::string> BackendCapabilities::names() const {
    static const std::pair<Capability, const char*> kNames[] = {
        {Capability::tokenization, "tokenization"},
        {Capability::streaming, "streaming"},
        {Capability::batched_prefill, "batched_prefill"},
        {Capability::continuous_batch_decode, "continuous_batch_decode"},
        {Capability::kv_export, "kv_export"},
        {Capability::kv_import, "kv_import"},
        {Capability::kv_copy, "kv_copy"},
        {Capability::kv_quantization, "kv_quantization"},
        {Capability::prefix_reuse, "prefix_reuse"},
        {Capability::speculative_decode, "speculative_decode"},
        {Capability::lora, "lora"},
        {Capability::layer_telemetry, "layer_telemetry"},
        {Capability::structured_output, "structured_output"},
        {Capability::embeddings, "embeddings"},
        {Capability::deterministic, "deterministic"},
        {Capability::remote_process, "remote_process"},
    };
    std::vector<std::string> out;
    for (const auto& [cap, name] : kNames) {
        if (has(cap)) {
            out.emplace_back(name);
        }
    }
    return out;
}

const char* to_string(StopReason reason) noexcept {
    switch (reason) {
        case StopReason::none: return "none";
        case StopReason::max_tokens: return "max_tokens";
        case StopReason::stop_sequence: return "stop_sequence";
        case StopReason::end_of_sequence: return "end_of_sequence";
        case StopReason::cancelled: return "cancelled";
        case StopReason::callback: return "callback";
        case StopReason::error: return "error";
    }
    return "unknown";
}

}  // namespace sonder::inference
