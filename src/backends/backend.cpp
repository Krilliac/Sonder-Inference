#include "sonder/inference/backend.hpp"

#include <cctype>
#include <cstdio>
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
        {Capability::recurrent_checkpoint, "recurrent_checkpoint"},
        {Capability::tool_calling, "tool_calling"},
        {Capability::multimodal, "multimodal"},
        {Capability::logprobs, "logprobs"},
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
        case StopReason::tool_calls: return "tool_calls";
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
        if (messages[i].content.find('\0') != std::string::npos) {
            return Status(ErrorCode::invalid_argument,
                          "chat: message " + std::to_string(i) + " content must not contain NUL");
        }
        if (messages[i].tool_call_id && messages[i].tool_call_id->find('\0') != std::string::npos) {
            return Status(ErrorCode::invalid_argument,
                          "chat: message " + std::to_string(i) + " tool_call_id must not contain NUL");
        }
        if (messages[i].name && messages[i].name->find('\0') != std::string::npos) {
            return Status(ErrorCode::invalid_argument,
                          "chat: message " + std::to_string(i) + " name must not contain NUL");
        }
        for (const auto& tc : messages[i].tool_calls) {
            if (tc.id.find('\0') != std::string::npos || tc.name.find('\0') != std::string::npos ||
                tc.arguments_json.find('\0') != std::string::npos) {
                return Status(ErrorCode::invalid_argument,
                              "chat: message " + std::to_string(i) + " tool_calls must not contain NUL");
            }
        }
    }
    const std::string& last = messages.back().role;
    if (last != "user" && last != "tool") {
        return Status(ErrorCode::invalid_argument, "chat: the last message must be from user or tool, got " + last);
    }
    return Status::success();
}

Result<std::vector<ChatPromptSegment>> build_chat_template_segments(
    const std::vector<ChatMessage>& messages,
    const std::function<Result<std::string>(const std::vector<ChatMessage>&)>& render_template) {
    if (Status st = validate_chat_messages(messages); !st.ok()) {
        return st;
    }
    if (!render_template) {
        return Status(ErrorCode::invalid_argument, "chat: render_template callback is empty");
    }

    std::uint64_t nonce = 0x7f3a9c1e5b2d4a09ULL;
    auto next_prefix = [&]() {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "SNDRMSGSLOT%016llxX", static_cast<unsigned long long>(nonce));
        nonce = nonce * 6364136223846793005ULL + 1442695040888963407ULL;
        return std::string(buf);
    };

    for (int attempt = 0; attempt < 8; ++attempt) {
        std::string prefix = next_prefix();
        auto prefix_in_messages = [&](std::string_view candidate) {
            for (const auto& m : messages) {
                if (m.role.find(candidate) != std::string::npos || m.content.find(candidate) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        while (prefix_in_messages(prefix)) {
            prefix = next_prefix();
        }

        std::vector<ChatMessage> placeholder_messages;
        placeholder_messages.reserve(messages.size());
        std::vector<std::string> placeholders(messages.size());
        for (std::size_t i = 0; i < messages.size(); ++i) {
            ChatMessage m;
            m.role = messages[i].role;
            if (!messages[i].content.empty()) {
                placeholders[i] = prefix + "_" + std::to_string(i) + "_Z";
                m.content = placeholders[i];
            }
            placeholder_messages.push_back(std::move(m));
        }

        auto rendered = render_template(placeholder_messages);
        if (!rendered.ok()) {
            return rendered.status();
        }
        const std::string& text = rendered.value();
        std::vector<ChatPromptSegment> segments;
        std::vector<std::size_t> seen(messages.size(), 0);
        bool collided = false;
        std::size_t cursor = 0;
        while (cursor < text.size()) {
            const std::size_t pos = text.find(prefix, cursor);
            if (pos == std::string::npos) {
                segments.push_back(ChatPromptSegment{text.substr(cursor), true});
                break;
            }
            const std::size_t after_prefix = pos + prefix.size();
            if (after_prefix >= text.size() || text[after_prefix] != '_') {
                collided = true;
                break;
            }
            const std::size_t digits_start = after_prefix + 1;
            std::size_t digits_end = digits_start;
            while (digits_end < text.size() && text[digits_end] >= '0' && text[digits_end] <= '9') {
                ++digits_end;
            }
            if (digits_end == digits_start || digits_end + 2 > text.size() || text[digits_end] != '_' ||
                text[digits_end + 1] != 'Z') {
                collided = true;
                break;
            }
            std::size_t idx = 0;
            bool overflow = false;
            for (std::size_t k = digits_start; k < digits_end; ++k) {
                const std::size_t digit = static_cast<std::size_t>(text[k] - '0');
                if (idx > (messages.size() - 1) / 10 + 1) {
                    overflow = true;
                    break;
                }
                idx = idx * 10 + digit;
            }
            if (overflow || idx >= messages.size() || placeholders[idx].empty() ||
                text.compare(pos, placeholders[idx].size(), placeholders[idx]) != 0) {
                collided = true;
                break;
            }
            if (pos > cursor) {
                segments.push_back(ChatPromptSegment{text.substr(cursor, pos - cursor), true});
            }
            segments.push_back(ChatPromptSegment{messages[idx].content, false});
            ++seen[idx];
            cursor = pos + placeholders[idx].size();
        }
        if (collided) {
            continue;
        }
        for (std::size_t i = 0; i < messages.size(); ++i) {
            if (!messages[i].content.empty() && seen[i] == 0) {
                return Status(ErrorCode::invalid_argument,
                              "chat: template did not preserve message boundary for message " + std::to_string(i));
            }
        }
        return segments;
    }
    return Status(ErrorCode::internal, "chat: failed to establish collision-free template boundary");
}

Status tokenize_chat_segments(
    const std::vector<ChatPromptSegment>& segments,
    const std::function<Status(std::string_view text, bool add_special, bool parse_special,
                               std::vector<TokenId>& out)>& tokenize_piece,
    std::vector<TokenId>& out) {
    out.clear();
    if (!tokenize_piece) {
        return Status(ErrorCode::invalid_argument, "chat: tokenize callback is empty");
    }
    bool first_non_empty = true;
    std::vector<TokenId> piece_tokens;
    for (const auto& seg : segments) {
        if (seg.text.empty()) {
            continue;
        }
        piece_tokens.clear();
        const bool add_special = first_non_empty;
        first_non_empty = false;
        if (Status st = tokenize_piece(seg.text, add_special, seg.parse_special, piece_tokens); !st.ok()) {
            out.clear();
            return st;
        }
        out.insert(out.end(), piece_tokens.begin(), piece_tokens.end());
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
        if (m.role == "tool" && m.tool_call_id && !m.tool_call_id->empty()) {
            out += " (";
            out += *m.tool_call_id;
            out += ")";
        }
        out += ": ";
        out += m.content;
        if (!m.tool_calls.empty()) {
            for (const auto& tc : m.tool_calls) {
                if (!m.content.empty()) out.push_back('\n');
                out += "<tool_call>{\"name\":";
                json::append_escaped(out, tc.name);
                out += ",\"arguments\":";
                out += tc.arguments_json.empty() ? "{}" : tc.arguments_json;
                out += "}</tool_call>";
            }
        }
        if (!m.media.empty()) {
            for (const auto& img : m.media) {
                if (!m.content.empty()) out.push_back(' ');
                std::uint64_t h = 1469598103934665603ull;
                for (const std::uint8_t b : img.bytes) {
                    h ^= b;
                    h *= 1099511628211ull;
                }
                char tag[96];
                std::snprintf(tag, sizeof(tag), "[image:%zu:%016llx]", img.bytes.size(),
                              static_cast<unsigned long long>(h));
                out += tag;
            }
        }
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
    g.session_key = request.session_key;
    g.grammar = request.grammar;
    return generate(g, cancel, on_chunk);
}

json::Object to_json(const GpuMemoryStatus& g) {
    json::Object out{{"probe", g.probe}, {"status", g.status}};
    if (!g.error.empty()) {
        out.set("error", g.error);
    }
    out.set("dedicated_bytes", g.dedicated_bytes);
    out.set("shared_bytes", g.shared_bytes);
    out.set("peak_shared_bytes", g.peak_shared_bytes);
    out.set("shared_baseline_bytes", g.shared_baseline_bytes);
    out.set("spill_threshold_bytes", g.spill_threshold_bytes);
    out.set("spilled", g.spilled);
    out.set("samples", g.samples);
    if (g.residency) {
        const auto &r = *g.residency;
        out.set("residency", json::Object{{"gpu_offload_missing", r.gpu_offload_missing},
                                           {"vram_evicted", r.vram_evicted},
                                           {"peak_dedicated_bytes", r.peak_dedicated_bytes},
                                           {"observed_dedicated_bytes", r.observed_dedicated_bytes},
                                           {"eviction_restarts", r.eviction_restarts},
                                           {"action", r.action}});
    }
    return out;
}

json::Object to_json(const BackendWarning& w) {
    json::Object details;
    for (const auto& [key, value] : w.details) {
        details.set(key, value);
    }
    return json::Object{{"code", w.code},     {"severity", w.severity},       {"source", w.source},
                        {"message", w.message}, {"details", std::move(details)}, {"count", w.count}};
}

json::Object to_json(const BackendWarmupSlotStatus& slot) {
    json::Object out{{"id_slot", slot.id_slot}, {"status", slot.status},
                     {"prompt_tokens", json::Value(slot.prompt_tokens)},
                     {"cache_n", json::Value(slot.cache_n)}, {"milliseconds", slot.milliseconds}};
    if (!slot.error.empty()) out.set("error", slot.error);
    return out;
}

json::Object to_json(const BackendWarmupStatus& status) {
    json::Array slots;
    for (const auto& slot : status.slots) slots.emplace_back(to_json(slot));
    return json::Object{{"generation", status.generation}, {"status", status.status}, {"slots", std::move(slots)}};
}

json::Object to_json(const BackendRuntimeStatus& status) {
    const auto& c = status.context;
    // Built with set() and the optional constructor (null when unset): a
    // conditional-operator temporary here trips GCC 13 -Wmaybe-uninitialized.
    json::Object context;
    context.set("policy", c.policy);
    context.set("configured_ctx", json::Value(c.configured_ctx));
    context.set("fitted_ctx", json::Value(c.fitted_ctx));
    context.set("fit_attempts", c.fit_attempts);
    context.set("outcome", c.outcome);
    json::Array warnings;
    for (const auto& w : status.warnings) {
        warnings.emplace_back(to_json(w));
    }
    json::Object out{{"gpu_memory", to_json(status.gpu_memory)},
                     {"context", std::move(context)},
                     {"warnings", std::move(warnings)}};
    if (status.warmup) out.set("warmup", to_json(*status.warmup));
    if (status.child) out.set("child", *status.child);
    if (status.stall) out.set("stall", *status.stall);
    if (status.diagnostics) out.set("diagnostics", *status.diagnostics);
    return out;
}

}  // namespace sonder::inference
