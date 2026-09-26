#include "sonder/inference/backend.hpp"

#include <cctype>
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
        {Capability::token_logits, "token_logits"},
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

bool is_known_chat_role(std::string_view role) noexcept {
    return role == "system" || role == "user" || role == "assistant" || role == "tool";
}

Status validate_chat_messages(const std::vector<ChatMessage>& messages) {
    if (messages.empty()) {
        return Status(ErrorCode::invalid_argument, "chat: at least one message is required");
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (!is_known_chat_role(messages[i].role)) {
            return Status(ErrorCode::invalid_argument, "chat: message " + std::to_string(i) + " has unknown role \"" +
                                                           messages[i].role +
                                                           "\" (expected system, user, assistant or tool)");
        }
    }
    const std::string& last = messages.back().role;
    if (last != "user" && last != "tool") {
        return Status(ErrorCode::invalid_argument, "chat: the last message must be from user or tool, got " + last);
    }
    return Status::success();
}

std::string format_chat_prompt(const std::vector<ChatMessage>& messages) {
    std::string out;
    for (const auto& m : messages) {
        std::string label = m.role;
        if (!label.empty()) {
            label[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(label[0])));
        }
        out += label;
        out += ": ";
        out += m.content;
        out += "\n\n";
    }
    out += "Assistant:";
    return out;
}

Result<GenerateStats> BackendModel::chat(const ChatRequest& request, const CancellationToken& cancel,
                                         const TokenCallback& on_chunk) {
    if (Status st = validate_chat_messages(request.messages); !st.ok()) {
        return st;
    }
    GenerateRequest g;
    g.request_id = request.request_id;
    g.prompt = format_chat_prompt(request.messages);
    g.sampling = request.sampling;
    return generate(g, cancel, on_chunk);
}

}  // namespace sonder::inference
