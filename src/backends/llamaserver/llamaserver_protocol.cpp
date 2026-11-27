#include "llamaserver_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace sonder::inference::llamaserver {
namespace {
std::string encode_base64_bytes(std::span<const std::uint8_t> bytes) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((bytes.size() + 2u) / 3u) * 4u);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t b0 = bytes[i];
        const bool has1 = (i + 1 < bytes.size());
        const bool has2 = (i + 2 < bytes.size());
        const std::uint32_t b1 = has1 ? bytes[i + 1] : 0u;
        const std::uint32_t b2 = has2 ? bytes[i + 2] : 0u;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(triple >> 18) & 0x3Fu]);
        out.push_back(kAlphabet[(triple >> 12) & 0x3Fu]);
        out.push_back(has1 ? kAlphabet[(triple >> 6) & 0x3Fu] : '=');
        out.push_back(has2 ? kAlphabet[triple & 0x3Fu] : '=');
    }
    return out;
}

// A flagged field is always sent. Otherwise, unless explicit_only (set by
// parsers that know exactly what the caller supplied), a field is sent only
// when it differs from the local default, so the upstream keeps its own
// model defaults.
bool supplied(const SamplingConfig &s, SamplingConfig::Field field, bool changed) {
    return s.is_explicit(field) || (!s.explicit_only && changed);
}
void put_sampling(json::Object &o, const SamplingConfig &s, bool native, bool is_chat = false) {
    const SamplingConfig d{};
    if (supplied(s, SamplingConfig::kTemperature, s.temperature != d.temperature))
        o.set("temperature", static_cast<double>(s.temperature));
    if (supplied(s, SamplingConfig::kTopP, s.top_p != d.top_p))
        o.set("top_p", static_cast<double>(s.top_p));
    if (supplied(s, SamplingConfig::kTopK, s.top_k != d.top_k))
        o.set("top_k", s.top_k);
    if (supplied(s, SamplingConfig::kMinP, s.min_p != d.min_p))
        o.set("min_p", static_cast<double>(s.min_p));
    if (supplied(s, SamplingConfig::kRepeatPenalty, s.repeat_penalty != d.repeat_penalty))
        o.set("repeat_penalty", static_cast<double>(s.repeat_penalty));
    if (supplied(s, SamplingConfig::kTypicalP, s.typical_p != d.typical_p))
        o.set("typical_p", static_cast<double>(s.typical_p));
    if (supplied(s, SamplingConfig::kRepeatLastN, s.repeat_last_n != d.repeat_last_n))
        o.set("repeat_last_n", s.repeat_last_n);
    if (supplied(s, SamplingConfig::kPresencePenalty, s.presence_penalty != d.presence_penalty))
        o.set("presence_penalty", static_cast<double>(s.presence_penalty));
    if (supplied(s, SamplingConfig::kFrequencyPenalty, s.frequency_penalty != d.frequency_penalty))
        o.set("frequency_penalty", static_cast<double>(s.frequency_penalty));
    if (s.seed && supplied(s, SamplingConfig::kSeed, true))
        o.set("seed", static_cast<unsigned long long>(*s.seed));
    if (supplied(s, SamplingConfig::kMaxTokens, s.max_tokens != d.max_tokens))
        o.set(native ? "n_predict" : "max_tokens", s.max_tokens);
    if (supplied(s, SamplingConfig::kStop, !s.stop.empty())) {
        json::Array a;
        for (const auto &v : s.stop)
            a.emplace_back(v);
        o.set("stop", std::move(a));
    }
    if (supplied(s, SamplingConfig::kLogitBias, !s.logit_bias.empty())) {
        if (native) {
            json::Array bias;
            for (const auto &v : s.logit_bias) {
                bias.emplace_back(
                    json::Array{v.token, std::isinf(v.bias) ? json::Value(false) : json::Value(v.bias)});
            }
            o.set("logit_bias", std::move(bias));
        } else {
            json::Object bias;
            for (const auto &v : s.logit_bias)
                bias.set(std::to_string(v.token), std::isinf(v.bias) ? -100.0 : v.bias);
            o.set("logit_bias", std::move(bias));
        }
    }
    if (s.logprobs) {
        if (native) {
            o.set("n_probs", std::max<std::int32_t>(1, s.top_logprobs));
        } else if (is_chat) {
            o.set("logprobs", true);
            if (s.top_logprobs > 0) {
                o.set("top_logprobs", s.top_logprobs);
            }
        } else {
            o.set("logprobs", std::max<std::int32_t>(1, s.top_logprobs));
        }
    }
}
} // namespace

Status validate_sampling(const SamplingConfig &sampling, std::uint64_t served_ctx) {
    if (auto st = validate(sampling); !st.ok())
        return st;
    if (sampling.num_ctx > 0 && served_ctx > 0 && static_cast<std::uint64_t>(sampling.num_ctx) > served_ctx) {
        return Status(ErrorCode::invalid_argument,
                      "llamaserver: num_ctx " + std::to_string(sampling.num_ctx) + " exceeds the served context " +
                          std::to_string(served_ctx) +
                          " (llama-server's per-slot n_ctx); restart llama-server with a larger --ctx-size");
    }
    return {};
}

namespace {
void put_extras(json::Object &o, const RequestExtras &extras, bool emit_speculative) {
    if (extras.cache_prompt)
        o.set("cache_prompt", true);
    if (extras.id_slot)
        o.set("id_slot", static_cast<std::int64_t>(*extras.id_slot));
    if (emit_speculative && extras.spec_draft_n_max)
        o.set("speculative.n_max", static_cast<std::int64_t>(*extras.spec_draft_n_max));
    if (extras.telemetry_n_probs > 0) {
        o.set("n_probs", static_cast<std::int64_t>(extras.telemetry_n_probs));
        o.set("post_sampling_probs", true);
    }
}
} // namespace

json::Value build_completion_body(const std::string &model, const GenerateRequest &request,
                                  bool native_completion, std::string_view grammar,
                                  const RequestExtras &extras) {
    json::Object o;
    o.set("model", model);
    o.set("prompt", request.prompt);
    o.set("stream", true);
    if (!native_completion)
        o.set("stream_options", json::Object{{"include_usage", true}});
    put_sampling(o, request.sampling, native_completion);
    const std::string_view effective_grammar =
        !request.grammar.empty() ? std::string_view(request.grammar) : grammar;
    if (!effective_grammar.empty())
        o.set("grammar", std::string(effective_grammar));
    put_extras(o, extras, native_completion && !extras.generic_thinking);
    return o;
}

bool chat_tools_requested(const ChatRequest &request) noexcept {
    return !request.tools.empty() &&
           (!request.tool_choice.has_value() || request.tool_choice->mode != ToolChoiceMode::none);
}

json::Value build_chat_body(const std::string &model, const ChatRequest &request, std::string_view grammar,
                            const RequestExtras &extras) {
    json::Object o;
    o.set("model", model);
    o.set("stream", true);
    json::Array messages;
    for (const auto &m : request.messages) {
        json::Object message{{"role", m.role}};
        const auto append_image_part = [](json::Array &parts, const MediaContent &img) {
            const std::string data_uri = "data:" + img.mime_type + ";base64," + encode_base64_bytes(img.bytes);
            parts.emplace_back(
                json::Object{{"type", "image_url"}, {"image_url", json::Object{{"url", data_uri}}}});
        };
        const bool has_images = count_message_images(m) > 0;
        if (has_images && !m.parts.empty()) {
            json::Array parts;
            for (const auto &part : m.parts) {
                if (const auto *tp = std::get_if<TextContentPart>(&part)) {
                    if (!tp->text.empty()) {
                        parts.emplace_back(json::Object{{"type", "text"}, {"text", tp->text}});
                    }
                } else if (const auto *ip = std::get_if<ImageContentPart>(&part)) {
                    append_image_part(parts, ip->media);
                }
            }
            message.set("content", std::move(parts));
        } else if (!m.media.empty()) {
            json::Array parts;
            if (!m.content.empty()) {
                parts.emplace_back(json::Object{{"type", "text"}, {"text", m.content}});
            }
            for (const auto &img : m.media) {
                append_image_part(parts, img);
            }
            message.set("content", std::move(parts));
        } else {
            const std::string text_content = message_text_content(m);
            if (m.role == "assistant" && text_content.empty() && !m.tool_calls.empty()) {
                message.set("content", nullptr);
            } else {
                message.set("content", text_content);
            }
        }
        if (m.role == "assistant" && m.reasoning_content)
            message.set("reasoning_content", *m.reasoning_content);
        if (m.role == "assistant" && !m.tool_calls.empty()) {
            json::Array tcs;
            tcs.reserve(m.tool_calls.size());
            for (const auto &tc : m.tool_calls) {
                tcs.emplace_back(json::Object{
                    {"id", tc.id},
                    {"type", "function"},
                    {"function", json::Object{{"name", tc.name},
                                              {"arguments", tc.arguments_json.empty() ? "{}" : tc.arguments_json}}}});
            }
            message.set("tool_calls", std::move(tcs));
        }
        if (m.role == "tool" && m.tool_call_id && !m.tool_call_id->empty())
            message.set("tool_call_id", *m.tool_call_id);
        if (m.name && !m.name->empty())
            message.set("name", *m.name);
        messages.emplace_back(std::move(message));
    }
    o.set("messages", std::move(messages));
    o.set("stream_options", json::Object{{"include_usage", true}});
    put_sampling(o, request.sampling, false, true);
    const bool tools_active = chat_tools_requested(request);
    if (tools_active) {
        json::Array tools;
        tools.reserve(request.tools.size());
        for (const auto &tool : request.tools) {
            json::Object fn{{"name", tool.name}};
            if (!tool.description.empty()) {
                fn.set("description", tool.description);
            }
            bool has_params = false;
            if (!tool.parameters_json.empty()) {
                auto parsed_params = json::parse(tool.parameters_json);
                if (parsed_params.ok() && parsed_params.value().is_object() &&
                    !parsed_params.value().as_object().empty()) {
                    fn.set("parameters", std::move(parsed_params).value());
                    has_params = true;
                }
            }
            if (!has_params) {
                fn.set("parameters", json::Object{{"type", "object"}, {"properties", json::Object{}}});
            }
            if (tool.strict.has_value()) {
                fn.set("strict", *tool.strict);
            }
            tools.emplace_back(json::Object{{"type", "function"}, {"function", std::move(fn)}});
        }
        o.set("tools", std::move(tools));
        if (request.parallel_tool_calls.has_value()) {
            o.set("parallel_tool_calls", *request.parallel_tool_calls);
        }
        if (request.tool_choice) {
            switch (request.tool_choice->mode) {
                case ToolChoiceMode::auto_choice:
                    o.set("tool_choice", "auto");
                    break;
                case ToolChoiceMode::none:
                    break;
                case ToolChoiceMode::required:
                    o.set("tool_choice", "required");
                    break;
                case ToolChoiceMode::function:
                    o.set("tool_choice",
                          json::Object{{"type", "function"},
                                       {"function", json::Object{{"name", request.tool_choice->function_name}}}});
                    break;
            }
        }
    }
    bool emitted_response_format = false;
    if (request.response_format_type && !request.response_format_type->empty()) {
        const std::string &rf_type = *request.response_format_type;
        if (rf_type == "json_object" || rf_type == "text") {
            o.set("response_format", json::Object{{"type", rf_type}});
            emitted_response_format = true;
        } else if (rf_type == "json_schema" && request.response_format_schema) {
            auto parsed_schema = json::parse(*request.response_format_schema);
            if (parsed_schema.ok() && parsed_schema.value().is_object()) {
                o.set("response_format",
                      json::Object{{"type", "json_schema"},
                                   {"json_schema", json::Object{{"name", "response"},
                                                                {"schema", std::move(parsed_schema).value()},
                                                                {"strict", true}}}});
                emitted_response_format = true;
            }
        }
    }
    if (!emitted_response_format && !tools_active) {
        const std::string_view effective_grammar =
            !request.grammar.empty() ? std::string_view(request.grammar) : grammar;
        if (!effective_grammar.empty())
            o.set("grammar", std::string(effective_grammar));
    }
    if (!request.thinking.empty() && extras.generic_thinking) {
        if (request.thinking.enable_thinking)
            o.set("enable_thinking", *request.thinking.enable_thinking);
        if (request.thinking.reasoning_effort)
            o.set("reasoning_effort", *request.thinking.reasoning_effort);
    } else if (!request.thinking.empty()) {
        json::Object kwargs;
        if (request.thinking.enable_thinking)
            kwargs.set("enable_thinking", *request.thinking.enable_thinking);
        if (request.thinking.reasoning_effort)
            kwargs.set("reasoning_effort", *request.thinking.reasoning_effort);
        o.set("chat_template_kwargs", std::move(kwargs));
    }
    if (request.reasoning_budget_tokens)
        o.set("reasoning_budget_tokens", *request.reasoning_budget_tokens);
    if (request.reasoning_budget_message)
        o.set("reasoning_budget_message", *request.reasoning_budget_message);
    put_extras(o, extras, !extras.generic_thinking);
    return o;
}

Status parse_props(const json::Value &value, ServerProps &out) {
    out = {};
    if (!value.is_object())
        return Status(ErrorCode::protocol_error, "llamaserver: /props must be an object");
    const auto count = [](const json::Value *v, std::uint64_t max, std::uint64_t &dest) {
        if (!v || v->is_null())
            return true;
        if (!v->is_integer() || v->as_double() < 0 || v->as_uint() > max)
            return false;
        dest = v->as_uint();
        return true;
    };
    const json::Value *n_ctx = nullptr;
    if (const auto *settings = value.find("default_generation_settings"); settings && settings->is_object())
        n_ctx = settings->find("n_ctx");
    if (!n_ctx)
        n_ctx = value.find("n_ctx");
    std::uint64_t slots = 0;
    if (!count(n_ctx, std::uint64_t{1} << 32, out.n_ctx) ||
        !count(value.find("total_slots"), std::uint64_t{1} << 20, slots)) {
        out = {};
        return Status(ErrorCode::protocol_error, "llamaserver: invalid n_ctx or total_slots in /props");
    }
    out.total_slots = static_cast<std::uint32_t>(slots);
    return Status::success();
}

namespace {
Status parse_timings_into(const json::Value &value, Timings &out) {
    out = {};
    const json::Value *t = value.find("timings");
    if (t && !t->is_object() && !t->is_null())
        return Status(ErrorCode::protocol_error, "llamaserver: timings must be an object");
    if (!t || t->is_null())
        t = &value;
    const auto count = [&](const char *key, std::uint64_t &dest, bool &present) {
        const auto *v = t->find(key);
        if (!v)
            return true;
        if (!v->is_integer() || v->as_double() < 0)
            return false;
        dest = v->as_uint();
        present = true;
        out.present = true;
        return true;
    };
    if (!count("prompt_n", out.prompt_tokens, out.prompt_present) ||
        !count(t->find("cache_n") ? "cache_n" : "cached_tokens", out.cached_tokens, out.cache_present) ||
        !count(t->find("predicted_n") ? "predicted_n" : "completion_n", out.completion_tokens,
               out.completion_present) ||
        !count("draft_n", out.draft_tokens, out.draft_present) ||
        !count("draft_n_accepted", out.draft_accepted_tokens, out.draft_accepted_present)) {
        return Status(ErrorCode::protocol_error, "llamaserver: invalid timing token count");
    }
    if (out.cached_tokens > std::numeric_limits<std::uint64_t>::max() - out.prompt_tokens ||
        (out.draft_present && out.draft_accepted_present && out.draft_accepted_tokens > out.draft_tokens)) {
        return Status(ErrorCode::protocol_error, "llamaserver: inconsistent timing token counts");
    }
    if (const auto *speed = t->find("predicted_per_second")) {
        if (!speed->is_number() || !std::isfinite(speed->as_double()) || speed->as_double() < 0) {
            return Status(ErrorCode::protocol_error, "llamaserver: invalid token rate");
        }
        out.speed_present = true;
        out.present = true;
        out.predicted_tokens_per_second = speed->as_double();
    }
    const auto duration = [&](const char *key, std::uint64_t &ns) {
        const auto *v = t->find(key);
        if (!v)
            return true;
        const double ms = v->as_double(-1.0);
        if (!v->is_number() || !std::isfinite(ms) || ms < 0 ||
            ms >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()) / 1e6)
            return false;
        ns = static_cast<std::uint64_t>(ms * 1e6);
        out.present = true;
        return true;
    };
    if (!duration("prompt_ms", out.prompt_ns) || !duration("predicted_ms", out.eval_ns)) {
        return Status(ErrorCode::protocol_error, "llamaserver: invalid timing duration");
    }
    BackendTimings raw;
    if (auto status = parse_timings(value, raw); !status.ok())
        return status;
    if (raw.prompt_n || raw.cache_n || raw.prompt_ms || raw.predicted_n || raw.predicted_ms || raw.draft_n ||
        raw.draft_n_accepted)
        out.backend_timings = std::move(raw);
    return Status::success();
}
} // namespace

Status parse_timings(const json::Value &value, Timings &out) {
    auto status = parse_timings_into(value, out);
    if (!status.ok())
        out = {};
    return status;
}

StopReason map_finish_reason(std::string_view reason) {
    if (reason == "length" || reason == "limit")
        return StopReason::max_tokens;
    if (reason == "word")
        return StopReason::stop_sequence;
    if (reason == "stop" || reason == "eos" || reason == "end_of_sequence")
        return StopReason::end_of_sequence;
    return StopReason::none;
}
} // namespace sonder::inference::llamaserver
