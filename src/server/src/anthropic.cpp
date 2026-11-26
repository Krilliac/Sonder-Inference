#include "anthropic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "tool_call_parser.hpp"

namespace sonder::inference::server::detail {
namespace {

bool present(const json::Value* v) { return v != nullptr && !v->is_null(); }
ApiError bad(std::string code, std::string message, std::optional<std::string> param = std::nullopt) {
    return make_error(400, std::move(code), std::move(message), std::move(param));
}

std::optional<ApiError> number(const json::Object& o, const char* key, float& out) {
    const auto* v = o.find(key);
    if (!present(v))
        return std::nullopt;
    if (!v->is_number() || !std::isfinite(v->as_double()) ||
        std::fabs(v->as_double()) > static_cast<double>(std::numeric_limits<float>::max())) {
        return bad("invalid_sampling", std::string(key) + " must be a finite number in range", key);
    }
    out = static_cast<float>(v->as_double());
    return std::nullopt;
}

std::optional<ApiError> integer(const json::Object& o, const char* key, std::int64_t lo, std::int64_t hi,
                                std::int64_t& out) {
    const auto* v = o.find(key);
    if (!present(v))
        return std::nullopt;
    // All supported bounds fit exactly in double and int64_t. Check the
    // requested range before conversion, including JSON's unsigned integers.
    const double value = v->as_double();
    if (!v->is_number() || !std::isfinite(value) || std::floor(value) != value) {
        return bad("invalid_sampling", std::string(key) + " must be an integer", key);
    }
    if (value < static_cast<double>(lo) || value > static_cast<double>(hi)) {
        return bad("invalid_sampling",
                   std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi),
                   key);
    }
    out = static_cast<std::int64_t>(value);
    return std::nullopt;
}

std::optional<ApiError> text_content(const json::Value& v, std::string& out, const std::string& where) {
    if (v.is_string()) {
        out = v.as_string();
        return std::nullopt;
    }
    if (!v.is_array())
        return bad("invalid_messages", where + " must be a string or text block array", where);
    out.clear();
    for (std::size_t i = 0; i < v.as_array().size(); ++i) {
        const auto& b = v.as_array()[i];
        const auto p = where + "[" + std::to_string(i) + "]";
        if (!b.is_object())
            return bad("invalid_messages", p + " must be an object", p);
        const auto* type = b.find("type");
        if (!present(type) || !type->is_string())
            return bad("invalid_messages", p + ".type is required", p + ".type");
        if (type->as_string() == "image")
            return bad("unsupported_parameter", "image content blocks are not supported by sonder-inference",
                       p);
        if (type->as_string() != "text")
            return bad("unsupported_parameter", p + " content block type is not supported", p + ".type");
        const auto* t = b.find("text");
        if (!present(t) || !t->is_string())
            return bad("invalid_messages", p + ".text must be a string", p + ".text");
        out += t->as_string();
    }
    return std::nullopt;
}

std::optional<ApiError> effort_value(const json::Value* v, ThinkingOptions& thinking, const char* field) {
    if (!present(v))
        return std::nullopt;
    if (!v->is_string())
        return bad("invalid_json", std::string(field) + " must be a string", field);
    const std::string value = v->as_string();
    if (value == "off")
        thinking.enable_thinking = false;
    else if (value == "low" || value == "medium") {
        thinking.enable_thinking = true;
        thinking.reasoning_effort = value;
    } else if (value == "high") {
        thinking.enable_thinking = true;
        thinking.reasoning_effort = "xhigh";
    } else
        return bad("invalid_json", std::string(field) + " must be off, low, medium, or high", field);
    return std::nullopt;
}

std::string stop_reason(const GenerationResult& result, bool has_tool_use) {
    if (has_tool_use || result.stats.stop_reason == StopReason::tool_calls || !result.tool_calls.empty())
        return "tool_use";
    if (result.stats.stop_reason == StopReason::max_tokens)
        return "max_tokens";
    if (result.stats.stop_reason == StopReason::stop_sequence)
        return "stop_sequence";
    return "end_turn";
}

json::Object usage(const GenerationResult& result) {
    std::uint64_t input = result.stats.prompt_tokens;
    if (result.stats.cached_tokens)
        input -= std::min(input, *result.stats.cached_tokens);
    json::Object out{{"input_tokens", input}, {"output_tokens", result.stats.completion_tokens}};
    if (result.stats.cached_tokens)
        out.set("cache_read_input_tokens", *result.stats.cached_tokens);
    return out;
}

std::string frame(std::string_view event, const json::Object& value) {
    auto data = value;
    data.set("type", std::string(event));
    return "event: " + std::string(event) + "\ndata: " + json::Value(data).dump() + "\n\n";
}

} // namespace

std::variant<ChatJob, ApiError> parse_messages_request(std::string_view body_text) {
    // Cap parsed value nodes on ingress requests to limit allocation/CPU amplification (M-17).
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok())
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    if (!parsed.value().is_object())
        return bad("invalid_json", "request body must be a JSON object");
    const auto& body = parsed.value().as_object();
    ChatJob out;
    if (const auto* v = body.find("model"); present(v)) {
        if (!v->is_string() || v->as_string().empty())
            return bad("invalid_json", "model must be a non-empty string", "model");
        out.model = v->as_string();
    }
    std::int64_t max = 0;
    if (auto e = integer(body, "max_tokens", 1, SamplingConfig::kMaxTokensLimit, max))
        return *e;
    if (!present(body.find("max_tokens")))
        return bad("invalid_request", "max_tokens is required", "max_tokens");
    out.sampling.max_tokens = static_cast<std::int32_t>(max);
    out.sampling.explicit_only = true;
    out.sampling.explicit_fields |= SamplingConfig::kMaxTokens;
    if (const auto* tools_v = body.find("tools"); present(tools_v)) {
        if (auto e = parse_anthropic_tools(*tools_v, out.tools))
            return *e;
    }
    if (const auto* tc_v = body.find("tool_choice"); present(tc_v)) {
        ToolChoice tc;
        if (auto e = parse_anthropic_tool_choice(*tc_v, out.tools, tc))
            return *e;
        out.tool_choice = std::move(tc);
    }
    if (const auto* v = body.find("stream"); present(v)) {
        if (!v->is_bool())
            return bad("invalid_json", "stream must be a boolean", "stream");
        out.stream = v->as_bool();
    }
    if (const auto* v = body.find("system"); present(v)) {
        std::string text;
        if (auto e = text_content(*v, text, "system"))
            return *e;
        out.messages.push_back({"system", std::move(text)});
    }
    const auto* messages = body.find("messages");
    if (!present(messages) || !messages->is_array() || messages->as_array().empty())
        return bad("invalid_messages", "messages must be a non-empty array", "messages");
    std::size_t total_images = 0;
    for (std::size_t i = 0; i < messages->as_array().size(); ++i) {
        const auto& m = messages->as_array()[i];
        const auto p = "messages[" + std::to_string(i) + "]";
        if (!m.is_object())
            return bad("invalid_messages", p + " must be an object", p);
        const auto* role = m.find("role");
        if (!present(role) || !role->is_string() ||
            (role->as_string() != "user" && role->as_string() != "assistant"))
            return bad("invalid_messages", p + ".role must be user or assistant", p + ".role");
        const auto* content = m.find("content");
        if (!present(content))
            return bad("invalid_messages", p + ".content is required", p + ".content");
        if (content->is_string()) {
            out.messages.push_back(ChatMessage{role->as_string(), content->as_string()});
            continue;
        }
        if (!content->is_array() || content->as_array().empty())
            return bad("invalid_messages", p + ".content must be a string or non-empty content block array",
                       p + ".content");

        const std::string& role_str = role->as_string();
        if (role_str == "user") {
            ChatMessage user_msg;
            user_msg.role = "user";
            std::vector<ChatMessage> tool_results;
            bool has_user_content = false;
            for (std::size_t j = 0; j < content->as_array().size(); ++j) {
                const auto& b = content->as_array()[j];
                const auto bp = p + ".content[" + std::to_string(j) + "]";
                if (!b.is_object())
                    return bad("invalid_messages", bp + " must be an object", bp);
                const auto* type = b.find("type");
                if (!present(type) || !type->is_string())
                    return bad("invalid_messages", bp + ".type is required", bp + ".type");
                const std::string& btype = type->as_string();
                if (btype == "text") {
                    const auto* t = b.find("text");
                    if (!present(t) || !t->is_string())
                        return bad("invalid_messages", bp + ".text must be a string", bp + ".text");
                    user_msg.content += t->as_string();
                    user_msg.parts.push_back(TextContentPart{t->as_string()});
                    has_user_content = true;
                } else if (btype == "image") {
                    if (++total_images > kMaxImagesPerRequest)
                        return bad("invalid_request",
                                   "request exceeds maximum of " + std::to_string(kMaxImagesPerRequest) + " images",
                                   "messages");
                    auto media = parse_anthropic_image_block(b.as_object(), bp);
                    if (auto* err = std::get_if<ApiError>(&media))
                        return *err;
                    MediaContent parsed_media = std::move(std::get<MediaContent>(media));
                    user_msg.parts.push_back(ImageContentPart{parsed_media});
                    user_msg.media.push_back(std::move(parsed_media));
                    has_user_content = true;
                } else if (btype == "tool_result") {
                    const auto* tuid = b.find("tool_use_id");
                    if (!present(tuid) || !tuid->is_string() || !is_valid_tool_call_id(tuid->as_string()))
                        return bad("invalid_messages",
                                   bp + ".tool_use_id must be a non-empty string of at most 128 bytes without control characters",
                                   bp + ".tool_use_id");
                    if (const auto* is_err = b.find("is_error"); present(is_err) && !is_err->is_bool())
                        return bad("invalid_messages", bp + ".is_error must be a boolean", bp + ".is_error");
                    std::string result_text;
                    if (const auto* rc = b.find("content"); present(rc)) {
                        if (auto e = text_content(*rc, result_text, bp + ".content"))
                            return *e;
                    }
                    ChatMessage tm;
                    tm.role = "tool";
                    tm.content = std::move(result_text);
                    tm.tool_call_id = tuid->as_string();
                    tool_results.push_back(std::move(tm));
                } else {
                    return bad("unsupported_parameter", bp + " content block type is not supported", bp + ".type");
                }
            }
            for (auto& tm : tool_results)
                out.messages.push_back(std::move(tm));
            if (has_user_content || tool_results.empty())
                out.messages.push_back(std::move(user_msg));
        } else {
            ChatMessage asst_msg;
            asst_msg.role = "assistant";
            for (std::size_t j = 0; j < content->as_array().size(); ++j) {
                const auto& b = content->as_array()[j];
                const auto bp = p + ".content[" + std::to_string(j) + "]";
                if (!b.is_object())
                    return bad("invalid_messages", bp + " must be an object", bp);
                const auto* type = b.find("type");
                if (!present(type) || !type->is_string())
                    return bad("invalid_messages", bp + ".type is required", bp + ".type");
                const std::string& btype = type->as_string();
                if (btype == "text") {
                    const auto* t = b.find("text");
                    if (!present(t) || !t->is_string())
                        return bad("invalid_messages", bp + ".text must be a string", bp + ".text");
                    asst_msg.content += t->as_string();
                } else if (btype == "thinking") {
                    const auto* t = b.find("thinking");
                    if (!present(t) || !t->is_string())
                        return bad("invalid_messages", bp + ".thinking must be a string", bp + ".thinking");
                    if (!asst_msg.reasoning_content)
                        asst_msg.reasoning_content = t->as_string();
                    else
                        *asst_msg.reasoning_content += t->as_string();
                } else if (btype == "tool_use") {
                    if (asst_msg.tool_calls.size() >= kMaxToolCallsPerMessage)
                        return bad("invalid_messages",
                                   p + " exceeds maximum of " + std::to_string(kMaxToolCallsPerMessage) +
                                       " tool_use blocks",
                                   p);
                    const auto* id = b.find("id");
                    if (!present(id) || !id->is_string() || !is_valid_tool_call_id(id->as_string()))
                        return bad("invalid_messages",
                                   bp + ".id must be a non-empty string of at most 128 bytes without control characters",
                                   bp + ".id");
                    const auto* name = b.find("name");
                    if (!present(name) || !name->is_string() || !is_valid_tool_name(name->as_string()))
                        return bad("invalid_messages", bp + ".name must be 1 to 128 characters of [A-Za-z0-9_.-]",
                                   bp + ".name");
                    std::string args_json = "{}";
                    if (const auto* input = b.find("input"); present(input)) {
                        if (!input->is_object())
                            return bad("invalid_messages", bp + ".input must be an object", bp + ".input");
                        args_json = input->dump();
                        if (args_json.size() > kMaxToolCallPayloadBytes)
                            return bad("invalid_messages",
                                       bp + ".input exceeds maximum size of " +
                                           std::to_string(kMaxToolCallPayloadBytes) + " bytes",
                                       bp + ".input");
                    }
                    asst_msg.tool_calls.push_back(ToolCall{id->as_string(), name->as_string(), std::move(args_json)});
                } else {
                    return bad("unsupported_parameter", bp + " content block type is not supported", bp + ".type");
                }
            }
            out.messages.push_back(std::move(asst_msg));
        }
    }
    if (Status st = validate_chat_messages(out.messages); !st.ok())
        return bad("invalid_messages", st.message(), "messages");
    if (const auto* p = body.find("priority")) {
        if (!p->is_string() || !parse_request_priority(p->as_string()))
            return bad("invalid_request", "priority must be interactive, subagent or background", "priority");
        out.priority_class = parse_request_priority(p->as_string());
    }
    if (const auto* d = body.find("deadline_ms")) {
        out.deadline_ms = parse_deadline_ms(*d);
        if (!out.deadline_ms)
            return bad("invalid_request", "deadline_ms must be a positive integer", "deadline_ms");
    }
    if (auto e = number(body, "temperature", out.sampling.temperature))
        return *e;
    if (present(body.find("temperature")))
        out.sampling.explicit_fields |= SamplingConfig::kTemperature;
    if (auto e = number(body, "top_p", out.sampling.top_p))
        return *e;
    if (present(body.find("top_p")))
        out.sampling.explicit_fields |= SamplingConfig::kTopP;
    std::int64_t n = 0;
    if (auto e = integer(body, "top_k", 0, 100000, n))
        return *e;
    if (present(body.find("top_k"))) {
        out.sampling.top_k = static_cast<std::int32_t>(n);
        out.sampling.explicit_fields |= SamplingConfig::kTopK;
    }
    if (const auto* stop = body.find("stop_sequences"); present(stop)) {
        if (!stop->is_array() || stop->as_array().size() > SamplingConfig::kMaxStopSequences)
            return bad("invalid_sampling", "stop_sequences must be an array of at most 16 strings",
                       "stop_sequences");
        for (const auto& s : stop->as_array())
            if (!s.is_string() || s.as_string().empty() ||
                s.as_string().size() > SamplingConfig::kMaxStopSequenceBytes)
                return bad("invalid_sampling",
                           "stop_sequences entries must be non-empty strings of at most 256 bytes",
                           "stop_sequences");
        for (const auto& s : stop->as_array())
            out.sampling.stop.push_back(s.as_string());
        out.sampling.explicit_fields |= SamplingConfig::kStop;
    }
    if (const auto* thinking = body.find("thinking"); present(thinking)) {
        if (!thinking->is_object())
            return bad("invalid_json", "thinking must be an object", "thinking");
        const auto* type = thinking->find("type");
        if (!present(type) || !type->is_string() ||
            (type->as_string() != "enabled" && type->as_string() != "disabled"))
            return bad("invalid_json", "thinking.type must be enabled or disabled", "thinking.type");
        out.thinking.enable_thinking = type->as_string() == "enabled";
        if (type->as_string() == "enabled") {
            std::int64_t budget = 0;
            if (auto e = integer(thinking->as_object(), "budget_tokens", 1, SamplingConfig::kMaxTokensLimit,
                                 budget))
                return *e;
            if (!present(thinking->find("budget_tokens")))
                return bad("invalid_json", "thinking.budget_tokens is required when enabled",
                           "thinking.budget_tokens");
            out.reasoning_budget_tokens = budget;
        }
    }
    const auto* direct_effort = body.find("reasoning_effort");
    const auto* output_config = body.find("output_config");
    if (present(output_config) && !output_config->is_object())
        return bad("invalid_json", "output_config must be an object", "output_config");
    const auto* nested_effort = present(output_config) ? output_config->find("effort") : nullptr;
    if (present(direct_effort) && present(nested_effort)) {
        if (!direct_effort->is_string() || !nested_effort->is_string())
            return bad("invalid_json", "reasoning_effort and output_config.effort must be strings",
                       "reasoning_effort");
        if (direct_effort->as_string() != nested_effort->as_string())
            return bad("invalid_json", "reasoning_effort and output_config.effort disagree",
                       "reasoning_effort");
    }
    const json::Value* effort = direct_effort;
    const char* effort_field = "reasoning_effort";
    if (!present(effort)) {
        if (present(output_config)) {
            effort = nested_effort;
            effort_field = "output_config.effort";
        }
    }
    if (present(effort)) {
        if (!effort->is_string())
            return bad("invalid_json", std::string(effort_field) + " must be a string", effort_field);
        const std::string requested = effort->as_string();
        if (out.thinking.enable_thinking && *out.thinking.enable_thinking && requested == "off")
            return bad("invalid_json", "thinking enabled conflicts with effort off", "thinking");
        if (out.thinking.enable_thinking && !*out.thinking.enable_thinking && requested != "off")
            return bad("invalid_json", "thinking disabled conflicts with a thinking effort", "thinking");
        if (auto e = effort_value(effort, out.thinking, effort_field))
            return *e;
    }
    if (const auto* meta = body.find("metadata"); present(meta)) {
        if (!meta->is_object())
            return bad("invalid_json", "metadata must be an object", "metadata");
        if (const auto* uid = meta->find("user_id"); present(uid)) {
            if (!uid->is_string() || uid->as_string().empty() || uid->as_string().size() > 256)
                return bad("invalid_json", "metadata.user_id must be a non-empty string of at most 256 bytes",
                           "metadata.user_id");
            out.session_key = uid->as_string();
        }
    }
    if (Status st = validate(out.sampling); !st.ok())
        return bad("invalid_sampling", st.message());
    return out;
}

json::Object anthropic_error_body(const ApiError& error) {
    const char* type = "invalid_request_error";
    switch (error.status) {
    case 401:
        type = "authentication_error";
        break;
    case 403:
        type = "permission_error";
        break;
    case 404:
        type = "not_found_error";
        break;
    case 413:
        type = "request_too_large";
        break;
    case 429:
        type = "rate_limit_error";
        break;
    case 500:
        type = "api_error";
        break;
    case 503:
        type = "overloaded_error";
        break;
    default:
        break;
    }
    return json::Object{{"type", "error"},
                        {"error", json::Object{{"type", type}, {"message", error.message}}}};
}

json::Object anthropic_message_response(std::string_view id, std::string_view model,
                                        const GenerationResult& result,
                                        const std::vector<std::string>& warnings,
                                        const std::vector<ToolCall>& tool_calls,
                                        std::optional<std::string_view> visible_text_override) {
    const std::vector<ToolCall>& raw_calls = tool_calls.empty() ? result.tool_calls : tool_calls;
    std::vector<const ToolCall*> valid_calls;
    valid_calls.reserve(raw_calls.size());
    for (const auto& tc : raw_calls) {
        if (!tc.name.empty()) {
            valid_calls.push_back(&tc);
        }
    }
    const std::string_view visible_text = visible_text_override.value_or(std::string_view(result.text));
    json::Array content;
    if (!result.reasoning.empty())
        content.push_back(json::Object{{"type", "thinking"}, {"thinking", result.reasoning}});
    if (!visible_text.empty() || valid_calls.empty())
        content.push_back(json::Object{{"type", "text"}, {"text", std::string(visible_text)}});
    for (std::size_t idx = 0; idx < valid_calls.size(); ++idx) {
        const ToolCall& tc = *valid_calls[idx];
        json::Value input_val = json::Object{};
        if (!tc.arguments_json.empty()) {
            auto parsed = json::parse(tc.arguments_json);
            if (parsed.ok() && parsed.value().is_object())
                input_val = std::move(parsed).value();
        }
        const std::string call_id = tc.id.empty() ? ("toolu_" + std::to_string(idx)) : tc.id;
        content.push_back(json::Object{{"type", "tool_use"},
                                       {"id", call_id},
                                       {"name", tc.name},
                                       {"input", std::move(input_val)}});
    }
    json::Object out{{"id", std::string(id)},
                     {"type", "message"},
                     {"role", "assistant"},
                     {"model", std::string(model)},
                     {"content", std::move(content)},
                     {"stop_reason", stop_reason(result, !valid_calls.empty())},
                     {"stop_sequence", nullptr},
                     {"usage", usage(result)}};
    if (result.stats.stop_reason == StopReason::stop_sequence && result.stats.matched_stop)
        out.set("stop_sequence", *result.stats.matched_stop);
    if (!warnings.empty()) {
        json::Array a;
        for (const auto& w : warnings)
            a.emplace_back(w);
        out.set("sonder", json::Object{{"warnings", std::move(a)}});
    }
    return out;
}

AnthropicStream::AnthropicStream(std::string id, std::string model)
    : id_(std::move(id)), model_(std::move(model)) {}
std::string AnthropicStream::start() {
    if (started_ || finished_)
        return {};
    started_ = true;
    return frame("message_start",
                 json::Object{{"message", json::Object{{"id", id_},
                                                       {"type", "message"},
                                                       {"role", "assistant"},
                                                       {"model", model_},
                                                       {"content", json::Array{}},
                                                       {"stop_reason", nullptr},
                                                       {"stop_sequence", nullptr},
                                                       {"usage", json::Object{{"input_tokens", 0},
                                                                              {"output_tokens", 0}}}}}});
}
void AnthropicStream::close_active_tool_block(std::string& out) {
    if (!tool_block_) return;
    out += frame("content_block_stop", json::Object{{"index", block_index_++}});
    tool_block_ = false;
    if (auto it = pending_tools_.find(active_tool_index_); it != pending_tools_.end()) {
        it->second.stopped = true;
        it->second.pending_args.clear();
    }
}
std::string AnthropicStream::token(const TokenChunk& chunk) {
    if (!started_ || finished_)
        return {};
    std::string out;
    if (!chunk.reasoning.empty()) {
        if (!thinking_block_) {
            if (text_block_) {
                out += frame("content_block_stop", json::Object{{"index", block_index_++}});
                text_block_ = false;
            }
            close_active_tool_block(out);
            out +=
                frame("content_block_start",
                      json::Object{{"index", block_index_},
                                   {"content_block", json::Object{{"type", "thinking"}, {"thinking", ""}}}});
            thinking_block_ = true;
        }
        out += frame("content_block_delta",
                     json::Object{{"index", block_index_},
                                  {"delta", json::Object{{"type", "thinking_delta"},
                                                         {"thinking", std::string(chunk.reasoning)}}}});
    }
    if (!chunk.text.empty()) {
        if (thinking_block_) {
            out += frame("content_block_stop", json::Object{{"index", block_index_++}});
            thinking_block_ = false;
        }
        close_active_tool_block(out);
        if (!text_block_) {
            out += frame("content_block_start",
                         json::Object{{"index", block_index_},
                                      {"content_block", json::Object{{"type", "text"}, {"text", ""}}}});
            text_block_ = true;
            saw_text_ = true;
        }
        out += frame(
            "content_block_delta",
            json::Object{{"index", block_index_},
                         {"delta", json::Object{{"type", "text_delta"}, {"text", std::string(chunk.text)}}}});
    }
    return out;
}
std::string AnthropicStream::tool_call(const ToolCall& call) {
    if (!started_ || finished_ || call.name.empty())
        return {};
    std::string out;
    if (thinking_block_) {
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
        thinking_block_ = false;
    }
    if (text_block_) {
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
        text_block_ = false;
    }
    close_active_tool_block(out);
    saw_tool_use_ = true;
    const std::string call_id = call.id.empty() ? ("toolu_" + std::to_string(block_index_)) : call.id;
    out += frame("content_block_start",
                 json::Object{{"index", block_index_},
                              {"content_block", json::Object{{"type", "tool_use"},
                                                             {"id", call_id},
                                                             {"name", call.name},
                                                             {"input", json::Object{}}}}});
    const std::string& partial = call.arguments_json.empty() ? std::string("{}") : call.arguments_json;
    out += frame("content_block_delta",
                 json::Object{{"index", block_index_},
                              {"delta", json::Object{{"type", "input_json_delta"}, {"partial_json", partial}}}});
    out += frame("content_block_stop", json::Object{{"index", block_index_++}});
    return out;
}
std::string AnthropicStream::flush_pending_tools() {
    std::string out;
    close_active_tool_block(out);
    for (auto& [idx, slot] : pending_tools_) {
        if (slot.stopped) continue;
        if (slot.name.empty()) {
            slot.stopped = true;
            slot.pending_args.clear();
            continue;
        }
        saw_tool_use_ = true;
        const std::string call_id = slot.id.empty() ? ("toolu_" + std::to_string(idx)) : slot.id;
        out += frame("content_block_start",
                     json::Object{{"index", block_index_},
                                  {"content_block", json::Object{{"type", "tool_use"},
                                                                 {"id", call_id},
                                                                 {"name", slot.name},
                                                                 {"input", json::Object{}}}}});
        if (!slot.pending_args.empty()) {
            out += frame("content_block_delta",
                         json::Object{{"index", block_index_},
                                      {"delta", json::Object{{"type", "input_json_delta"},
                                                             {"partial_json", slot.pending_args}}}});
            slot.pending_args.clear();
        }
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
        slot.started = true;
        slot.stopped = true;
    }
    return out;
}
std::string AnthropicStream::tool_call_delta(const ToolCallDelta& delta) {
    if (!started_ || finished_ || delta.index >= kMaxToolCallsPerMessage)
        return {};
    std::string out;
    auto& slot = pending_tools_[delta.index];
    if (slot.stopped) {
        return out;
    }
    if (!delta.id.empty()) slot.id = delta.id;
    if (!delta.name.empty()) slot.name = delta.name;

    // Stream directly when this index is the currently open block (or no other
    // tool block is open yet and we already have a non-empty tool name).
    // Interleaved parallel deltas or chunks arriving before `name` buffer per
    // index and flush cleanly in index order without duplicate starts.
    if (tool_block_ && delta.index == active_tool_index_) {
        if (!delta.arguments_delta.empty()) {
            out += frame("content_block_delta",
                         json::Object{{"index", block_index_},
                                      {"delta", json::Object{{"type", "input_json_delta"},
                                                             {"partial_json", delta.arguments_delta}}}});
        }
        return out;
    }
    if (!tool_block_ && !slot.started && !slot.stopped && !slot.name.empty() && pending_tools_.size() == 1u) {
        if (thinking_block_) {
            out += frame("content_block_stop", json::Object{{"index", block_index_++}});
            thinking_block_ = false;
        }
        if (text_block_) {
            out += frame("content_block_stop", json::Object{{"index", block_index_++}});
            text_block_ = false;
        }
        saw_tool_use_ = true;
        tool_block_ = true;
        active_tool_index_ = delta.index;
        slot.started = true;
        const std::string call_id = slot.id.empty() ? ("toolu_" + std::to_string(delta.index)) : slot.id;
        out += frame("content_block_start",
                     json::Object{{"index", block_index_},
                                  {"content_block", json::Object{{"type", "tool_use"},
                                                                 {"id", call_id},
                                                                 {"name", slot.name},
                                                                 {"input", json::Object{}}}}});
        std::string initial_args = std::move(slot.pending_args);
        slot.pending_args.clear();
        initial_args += delta.arguments_delta;
        if (!initial_args.empty()) {
            out += frame("content_block_delta",
                         json::Object{{"index", block_index_},
                                      {"delta", json::Object{{"type", "input_json_delta"},
                                                             {"partial_json", initial_args}}}});
        }
        return out;
    }
    if (!delta.arguments_delta.empty() && slot.pending_args.size() < kMaxToolCallPayloadBytes) {
        const std::size_t budget = kMaxToolCallPayloadBytes - slot.pending_args.size();
        slot.pending_args.append(delta.arguments_delta.data(), std::min(budget, delta.arguments_delta.size()));
    }
    return out;
}
std::string AnthropicStream::finish(const GenerationResult& result, const std::vector<std::string>& warnings,
                                    json::Object metadata) {
    if (!started_ || finished_)
        return {};
    std::string out;
    if (thinking_block_) {
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
        thinking_block_ = false;
    }
    if (text_block_) {
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
        text_block_ = false;
    }
    out += flush_pending_tools();
    if (!saw_text_ && !saw_tool_use_) {
        out += frame("content_block_start",
                     json::Object{{"index", block_index_},
                                  {"content_block", json::Object{{"type", "text"}, {"text", ""}}}});
        out += frame("content_block_stop", json::Object{{"index", block_index_++}});
    }
    const json::Value matched = result.stats.stop_reason == StopReason::stop_sequence
                                    ? json::Value(result.stats.matched_stop)
                                    : json::Value(nullptr);
    json::Object delta{
        {"delta", json::Object{{"stop_reason", stop_reason(result, saw_tool_use_)}, {"stop_sequence", matched}}},
        {"usage", usage(result)}};
    if (!warnings.empty())
        metadata.set("warnings", json::Array(warnings.begin(), warnings.end()));
    if (!metadata.empty())
        delta.set("sonder", std::move(metadata));
    finished_ = true;
    out += frame("message_delta", delta);
    out += frame("message_stop", json::Object{});
    return out;
}

} // namespace sonder::inference::server::detail
