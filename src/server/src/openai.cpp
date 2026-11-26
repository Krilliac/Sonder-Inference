#include "openai.hpp"
#include "json_schema_gbnf.hpp"
#include "reasoning_budget.hpp"
#include "tool_call_parser.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

const char* error_type(int status) noexcept {
    switch (status) {
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        case 429: return "rate_limit_error";
        case 500: return "server_error";
        case 501: return "not_implemented_error";
        case 503: return "service_unavailable";
        default: return "invalid_request_error";
    }
}

ApiError make_error(int status, std::string code, std::string message, std::optional<std::string> param) {
    ApiError e;
    e.status = status;
    e.code = std::move(code);
    e.message = std::move(message);
    e.param = std::move(param);
    return e;
}

json::Object error_body(const ApiError& error) {
    json::Object err{{"message", error.message},
                     {"type", error_type(error.status)},
                     {"code", error.code},
                     {"param", error.param}};
    return json::Object{{"error", std::move(err)}, {"sonder", json::Object{{"api_version", kApiVersion}}}};
}

namespace {

ApiError bad(std::string code, std::string message, std::optional<std::string> param = std::nullopt) {
    return make_error(400, std::move(code), std::move(message), std::move(param));
}

bool present(const json::Value* v) { return v != nullptr && !v->is_null(); }

std::optional<RequestPriority> parse_request_priority_impl(std::string_view value) noexcept {
    if (value == "interactive") return RequestPriority::interactive;
    if (value == "subagent") return RequestPriority::subagent;
    if (value == "background") return RequestPriority::background;
    return std::nullopt;
}

// Preserve the original X-Sonder-Priority wire contract.  This parser is
// intentionally separate from the class-name parser: numeric values carry
// the scheduler/telemetry integer, while class names only select admission.
std::optional<int> parse_numeric_priority_impl(std::string_view value) noexcept {
    bool negative = false;
    if (!value.empty() && (value.front() == '-' || value.front() == '+')) {
        negative = value.front() == '-';
        value.remove_prefix(1);
    }
    if (value.empty() || value.size() > 2 || value.find_first_not_of("0123456789") != std::string_view::npos) {
        return std::nullopt;
    }
    int parsed = 0;
    for (const char ch : value) parsed = parsed * 10 + (ch - '0');
    parsed = negative ? -parsed : parsed;
    if (parsed < -16 || parsed > 16) return std::nullopt;
    return parsed;
}

bool looks_numeric_priority(std::string_view value) noexcept {
    if (!value.empty() && (value.front() == '-' || value.front() == '+')) value.remove_prefix(1);
    return !value.empty() && value.find_first_not_of("0123456789") == std::string_view::npos;
}

std::optional<std::uint64_t> parse_deadline_ms_impl(const json::Value& value) noexcept {
    if (value.is_integer()) {
        const auto n = value.as_uint(std::numeric_limits<std::uint64_t>::max());
        if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) return std::nullopt;
        return n;
    }
    if (value.is_number() && std::isfinite(value.as_double()) && value.as_double() > 0.0 &&
        std::floor(value.as_double()) == value.as_double() &&
        value.as_double() <= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        return static_cast<std::uint64_t>(value.as_double());
    }
    return std::nullopt;
}

std::optional<ApiError> read_request_hints(const json::Object& body,
                                           std::optional<RequestPriority>& priority_class,
                                           std::optional<std::uint64_t>& deadline_ms) {
    if (const json::Value* p = body.find("priority")) {
        if (!p->is_string() || !parse_request_priority_impl(p->as_string())) {
            return bad("invalid_request", "priority must be interactive, subagent or background", "priority");
        }
        priority_class = parse_request_priority_impl(p->as_string());
    }
    if (const json::Value* d = body.find("deadline_ms")) {
        deadline_ms = parse_deadline_ms_impl(*d);
        if (!deadline_ms) {
            return bad("invalid_request", "deadline_ms must be a positive integer", "deadline_ms");
        }
    }
    return std::nullopt;
}

std::optional<ApiError> read_grammar_field(const json::Object& body, std::string& grammar_out) {
    if (const json::Value* g = body.find("grammar"); present(g)) {
        if (!g->is_string()) {
            return bad("invalid_request", "grammar must be a string", "grammar");
        }
        if (g->as_string().size() > kMaxRequestGrammarBytes) {
            return bad("invalid_request",
                       "grammar exceeds maximum length of " + std::to_string(kMaxRequestGrammarBytes) + " bytes",
                       "grammar");
        }
        grammar_out = g->as_string();
    }
    return std::nullopt;
}

std::optional<ApiError> read_response_format(const json::Object& body, ChatJob& job) {
    const json::Value* rf = body.find("response_format");
    if (!present(rf)) {
        return std::nullopt;
    }
    if (!rf->is_object()) {
        return bad("invalid_request", "response_format must be an object", "response_format");
    }
    const json::Value* type_v = rf->find("type");
    if (!present(type_v) || !type_v->is_string()) {
        return bad("invalid_request", "response_format.type must be a string", "response_format.type");
    }
    const std::string& type = type_v->as_string();
    if (type == "text") {
        return std::nullopt;
    }
    if (type == "json_object") {
        job.grammar = json_object_gbnf();
        job.response_format_type = "json_object";
        return std::nullopt;
    }
    if (type == "json_schema") {
        const json::Value* js = rf->find("json_schema");
        if (!present(js) || !js->is_object()) {
            return bad("invalid_request", "response_format.json_schema must be an object",
                       "response_format.json_schema");
        }
        if (const json::Value* name_v = js->find("name"); present(name_v)) {
            if (!name_v->is_string() || name_v->as_string().empty()) {
                return bad("invalid_request", "response_format.json_schema.name must be a non-empty string",
                           "response_format.json_schema.name");
            }
        }
        const json::Value* schema_v = js->find("schema");
        if (!present(schema_v) || !schema_v->is_object()) {
            return bad("invalid_request", "response_format.json_schema.schema must be a JSON object",
                       "response_format.json_schema.schema");
        }
        auto gbnf = json_schema_to_gbnf(*schema_v);
        if (!gbnf.ok()) {
            return bad("invalid_request", "invalid response_format.json_schema.schema: " + gbnf.status().message(),
                       "response_format.json_schema.schema");
        }
        job.grammar = std::move(gbnf).value();
        job.response_format_type = "json_schema";
        job.response_format_schema = schema_v->dump();
        return std::nullopt;
    }
    return bad("unsupported_parameter",
               "response_format.type '" + type + "' is not supported (expected text, json_object or json_schema)",
               "response_format");
}

// Numeric field as float; error when it is not a finite number.
std::optional<ApiError> read_float(const json::Object& body, const char* key, float& out) {
    const json::Value* v = body.find(key);
    if (!present(v)) {
        return std::nullopt;
    }
    if (!v->is_number()) {
        return bad("invalid_sampling", std::string(key) + " must be a number", key);
    }
    const double d = v->as_double();
    if (!std::isfinite(d) || std::fabs(d) > 1e6) {
        return bad("invalid_sampling", std::string(key) + " is out of range", key);
    }
    out = static_cast<float>(d);
    return std::nullopt;
}

// Integer field (a JSON number with an integral value) within [lo, hi].
std::optional<ApiError> read_int(const json::Object& body, const char* key, std::int64_t lo, std::int64_t hi,
                                 std::int64_t& out, bool& set) {
    set = false;
    const json::Value* v = body.find(key);
    if (!present(v)) {
        return std::nullopt;
    }
    std::int64_t n = 0;
    if (const auto exact = v->try_as_int()) {
        n = *exact;
    } else if (v->is_integer()) {
        return bad("invalid_sampling",
                   std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi), key);
    } else if (v->is_number() && std::isfinite(v->as_double()) && std::floor(v->as_double()) == v->as_double() &&
               std::fabs(v->as_double()) < 9.0e15) {
        n = static_cast<std::int64_t>(v->as_double());
    } else {
        return bad("invalid_sampling", std::string(key) + " must be an integer", key);
    }
    if (n < lo || n > hi) {
        return bad("invalid_sampling",
                   std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi), key);
    }
    out = n;
    set = true;
    return std::nullopt;
}

std::optional<ApiError> read_sampling(const json::Object& body, SamplingConfig& s) {
    // Record which sampler fields the request set, so a backend with per-model
    // defaults (Ollama) sends only those and leaves the rest to the model.
    s.explicit_only = true;
    struct FloatField {
        const char* key;
        float* target;
        std::uint32_t flag;  // 0 = not tracked
    };
    for (const auto& f : std::initializer_list<FloatField>{
             {"temperature", &s.temperature, SamplingConfig::kTemperature},
             {"top_p", &s.top_p, SamplingConfig::kTopP},
             {"min_p", &s.min_p, SamplingConfig::kMinP},
             {"typical_p", &s.typical_p, SamplingConfig::kTypicalP},
             {"presence_penalty", &s.presence_penalty, SamplingConfig::kPresencePenalty},
             {"frequency_penalty", &s.frequency_penalty, SamplingConfig::kFrequencyPenalty},
             {"repeat_penalty", &s.repeat_penalty, SamplingConfig::kRepeatPenalty}}) {
        if (auto e = read_float(body, f.key, *f.target)) {
            return e;
        }
        if (f.flag != 0 && present(body.find(f.key))) {
            s.explicit_fields |= f.flag;
        }
    }
    std::int64_t n = 0;
    bool set = false;
    if (auto e = read_int(body, "top_k", 0, 100000, n, set)) return e;
    if (set) {
        s.top_k = static_cast<std::int32_t>(n);
        s.explicit_fields |= SamplingConfig::kTopK;
    }
    if (auto e = read_int(body, "repeat_last_n", -1, SamplingConfig::kMaxContextLimit, n, set)) return e;
    if (set) {
        s.repeat_last_n = static_cast<std::int32_t>(n);
        s.explicit_fields |= SamplingConfig::kRepeatLastN;
    }
    if (auto e = read_int(body, "num_ctx", 0, SamplingConfig::kMaxContextLimit, n, set)) return e;
    if (set) {
        s.num_ctx = static_cast<std::int32_t>(n);
        s.explicit_fields |= SamplingConfig::kNumCtx;
    }
    if (auto e = read_int(body, "seed", 0, std::numeric_limits<std::int64_t>::max(), n, set)) return e;
    if (set) {
        s.seed = static_cast<std::uint64_t>(n);
        s.explicit_fields |= SamplingConfig::kSeed;
    }
    if (auto e = read_int(body, "max_tokens", 1, SamplingConfig::kMaxTokensLimit, n, set)) return e;
    if (set) {
        s.max_tokens = static_cast<std::int32_t>(n);
        s.explicit_fields |= SamplingConfig::kMaxTokens;
    }
    // The newer OpenAI name wins when both are present.
    if (auto e = read_int(body, "max_completion_tokens", 1, SamplingConfig::kMaxTokensLimit, n, set)) return e;
    if (set) {
        s.max_tokens = static_cast<std::int32_t>(n);
        s.explicit_fields |= SamplingConfig::kMaxTokens;
    }

    if (const json::Value* stop = body.find("stop"); present(stop)) {
        s.explicit_fields |= SamplingConfig::kStop;
        s.stop.clear();
        if (stop->is_string()) {
            s.stop.push_back(stop->as_string());
        } else if (stop->is_array()) {
            if (stop->as_array().size() > 4) {
                return bad("invalid_sampling", "stop accepts at most 4 sequences", "stop");
            }
            for (const auto& item : stop->as_array()) {
                if (!item.is_string()) {
                    return bad("invalid_sampling", "stop must be a string or an array of strings", "stop");
                }
                s.stop.push_back(item.as_string());
            }
        } else {
            return bad("invalid_sampling", "stop must be a string or an array of strings", "stop");
        }
    }

    if (const json::Value* bias = body.find("logit_bias"); present(bias)) {
        if (!bias->is_object()) {
            return bad("invalid_sampling", "logit_bias must be an object of token id to bias", "logit_bias");
        }
        s.explicit_fields |= SamplingConfig::kLogitBias;
        s.logit_bias.clear();
        for (const auto& [key, value] : bias->as_object()) {
            if (key.empty() || key.size() > 10 ||
                key.find_first_not_of("0123456789") != std::string::npos) {
                return bad("invalid_sampling", "logit_bias keys must be non-negative token ids", "logit_bias");
            }
            const long long token = std::stoll(key);
            if (token > std::numeric_limits<std::int32_t>::max()) {
                return bad("invalid_sampling", "logit_bias token id is out of range", "logit_bias");
            }
            if (!value.is_number() || !std::isfinite(value.as_double())) {
                return bad("invalid_sampling", "logit_bias values must be numbers", "logit_bias");
            }
            TokenLogitBias b;
            b.token = static_cast<std::int32_t>(token);
            b.bias = static_cast<float>(value.as_double());
            s.logit_bias.push_back(b);
        }
    }
    return std::nullopt;
}

bool reasoning_effort_ok(std::string_view v) {
    if (v.empty() || v.size() > 64) {
        return false;
    }
    for (const char c : v) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

// prompt_cache_key, chat_template_kwargs and think (docs/SERVER.md).
std::optional<ApiError> read_chat_extensions(const json::Object& body, ChatJob& job) {
    if (auto error = reasoning_budget::parse_body(body, job)) return error;
    if (const json::Value* key = body.find("prompt_cache_key"); present(key)) {
        if (!key->is_string() || key->as_string().empty() || key->as_string().size() > 256) {
            return bad("invalid_json", "prompt_cache_key must be a string of 1 to 256 bytes", "prompt_cache_key");
        }
        job.session_key = key->as_string();
    }
    if (const json::Value* kw = body.find("chat_template_kwargs"); present(kw)) {
        if (!kw->is_object()) {
            return bad("invalid_json", "chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        if (const json::Value* v = kw->find("enable_thinking"); present(v)) {
            if (!v->is_bool()) {
                return bad("invalid_json", "chat_template_kwargs.enable_thinking must be a boolean",
                           "chat_template_kwargs.enable_thinking");
            }
            job.thinking.enable_thinking = v->as_bool();
        }
        if (const json::Value* v = kw->find("reasoning_effort"); present(v)) {
            if (!v->is_string() || !reasoning_effort_ok(v->as_string())) {
                return bad("invalid_json",
                           "chat_template_kwargs.reasoning_effort must be a string of 1 to 64 [A-Za-z0-9._-]",
                           "chat_template_kwargs.reasoning_effort");
            }
            job.thinking.reasoning_effort = v->as_string();
        }
    }
    if (const json::Value* think = body.find("think"); present(think)) {
        if (!think->is_bool()) {
            return bad("invalid_json", "think must be a boolean", "think");
        }
        if (job.thinking.enable_thinking && *job.thinking.enable_thinking != think->as_bool()) {
            return bad("invalid_json", "think and chat_template_kwargs.enable_thinking disagree", "think");
        }
        job.thinking.enable_thinking = think->as_bool();
    }
    return std::nullopt;
}

}  // namespace

std::optional<RequestPriority> parse_request_priority(std::string_view value) noexcept {
    return parse_request_priority_impl(value);
}

std::optional<std::uint64_t> parse_deadline_ms(const json::Value& value) noexcept {
    return parse_deadline_ms_impl(value);
}

std::variant<ChatJob, ApiError> parse_chat_request(std::string_view body_text) {
    // Cap parsed value nodes on ingress requests to limit allocation/CPU amplification (M-17).
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    ChatJob job;

    // Legacy/unsupported OpenAI fields rejected explicitly rather than ignored.
    for (const char* key : {"functions", "function_call"}) {
        if (present(body.find(key))) {
            return bad("unsupported_parameter", std::string(key) + " is not supported by sonder-inference", key);
        }
    }
    if (const json::Value* n = body.find("n"); present(n)) {
        if (!n->is_number() || n->as_double() != 1.0) {
            return bad("unsupported_parameter", "only n = 1 is supported", "n");
        }
    }
    if (const json::Value* tools_v = body.find("tools"); present(tools_v)) {
        if (auto e = parse_openai_tools(*tools_v, job.tools, false)) {
            return *e;
        }
    }
    if (const json::Value* tc_v = body.find("tool_choice"); present(tc_v)) {
        ToolChoice tc;
        if (auto e = parse_openai_tool_choice(*tc_v, job.tools, tc, false)) {
            return *e;
        }
        job.tool_choice = std::move(tc);
    }

    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_json", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    if (const json::Value* stream = body.find("stream"); present(stream)) {
        if (!stream->is_bool()) {
            return bad("invalid_json", "stream must be a boolean", "stream");
        }
        job.stream = stream->as_bool();
    }
    if (const json::Value* so = body.find("stream_options"); present(so)) {
        if (!so->is_object()) {
            return bad("invalid_json", "stream_options must be an object", "stream_options");
        }
        if (const json::Value* iu = so->find("include_usage"); present(iu)) {
            if (!iu->is_bool()) {
                return bad("invalid_json", "stream_options.include_usage must be a boolean",
                           "stream_options.include_usage");
            }
            job.include_usage = iu->as_bool();
        }
    }

    const json::Value* messages = body.find("messages");
    if (!present(messages) || !messages->is_array() || messages->as_array().empty()) {
        return bad("invalid_messages", "messages must be a non-empty array", "messages");
    }
    std::size_t total_images = 0;
    for (std::size_t i = 0; i < messages->as_array().size(); ++i) {
        const json::Value& m = messages->as_array()[i];
        const std::string where = "messages[" + std::to_string(i) + "]";
        if (!m.is_object()) {
            return bad("invalid_messages", where + " must be an object", where);
        }
        const json::Value* role = m.find("role");
        if (!present(role) || !role->is_string() || !is_known_chat_role(role->as_string())) {
            return bad("invalid_messages", where + ".role must be system, user, assistant or tool", where + ".role");
        }
        ChatMessage msg;
        msg.role = role->as_string();
        if (const json::Value* name_v = m.find("name"); present(name_v)) {
            if (!name_v->is_string()) {
                return bad("invalid_messages", where + ".name must be a string", where + ".name");
            }
            msg.name = name_v->as_string();
        }
        if (const json::Value* tcs = m.find("tool_calls"); present(tcs)) {
            if (msg.role != "assistant") {
                return bad("invalid_messages", where + ".tool_calls is only valid on assistant messages",
                           where + ".tool_calls");
            }
            if (auto e = parse_openai_message_tool_calls(*tcs, msg.tool_calls, where + ".tool_calls")) {
                return *e;
            }
        }
        if (const json::Value* tcid = m.find("tool_call_id"); present(tcid)) {
            if (msg.role != "tool") {
                return bad("invalid_messages", where + ".tool_call_id is only valid on tool messages",
                           where + ".tool_call_id");
            }
            if (!tcid->is_string() || tcid->as_string().empty()) {
                return bad("invalid_messages", where + ".tool_call_id must be a non-empty string",
                           where + ".tool_call_id");
            }
            msg.tool_call_id = tcid->as_string();
        }
        const json::Value* content = m.find("content");
        if (!present(content)) {
            if (msg.role == "assistant" && !msg.tool_calls.empty()) {
                msg.content.clear();
            } else {
                return bad("invalid_messages", where + ".content must be a string or content parts array",
                           where + ".content");
            }
        } else if (content->is_string()) {
            msg.content = content->as_string();
        } else if (content->is_array()) {
            const auto& parts = content->as_array();
            for (std::size_t j = 0; j < parts.size(); ++j) {
                const std::string pwhere = where + ".content[" + std::to_string(j) + "]";
                if (!parts[j].is_object()) {
                    return bad("invalid_messages", pwhere + " must be an object", pwhere);
                }
                const json::Value* ptype = parts[j].find("type");
                if (!present(ptype) || !ptype->is_string()) {
                    return bad("invalid_messages", pwhere + ".type must be a string", pwhere + ".type");
                }
                if (ptype->as_string() == "text") {
                    const json::Value* text = parts[j].find("text");
                    if (!present(text) || !text->is_string()) {
                        return bad("invalid_messages", pwhere + ".text must be a string", pwhere + ".text");
                    }
                    msg.content += text->as_string();
                    msg.parts.push_back(TextContentPart{text->as_string()});
                } else if (ptype->as_string() == "image_url") {
                    const json::Value* iu = parts[j].find("image_url");
                    const json::Value* url = (iu != nullptr && iu->is_object()) ? iu->find("url") : nullptr;
                    if (!present(url) || !url->is_string()) {
                        return bad("invalid_request", pwhere + ".image_url.url must be a string",
                                   pwhere + ".image_url.url");
                    }
                    if (++total_images > kMaxImagesPerRequest) {
                        return bad("invalid_request",
                                   "request exceeds maximum of " + std::to_string(kMaxImagesPerRequest) + " images",
                                   "messages");
                    }
                    auto media = parse_data_image_uri(url->as_string(), pwhere + ".image_url.url");
                    if (auto* err = std::get_if<ApiError>(&media)) {
                        return *err;
                    }
                    MediaContent parsed_media = std::move(std::get<MediaContent>(media));
                    msg.parts.push_back(ImageContentPart{parsed_media});
                    msg.media.push_back(std::move(parsed_media));
                } else {
                    return bad("unsupported_parameter",
                               pwhere + ".type '" + ptype->as_string() + "' is not supported", pwhere + ".type");
                }
            }
        } else {
            return bad("invalid_messages", where + ".content must be a string or content parts array",
                       where + ".content");
        }
        if (msg.role == "assistant") {
            if (const auto* reasoning = m.find("reasoning_content"); present(reasoning)) {
                if (!reasoning->is_string()) {
                    return bad("invalid_messages", where + ".reasoning_content must be a string",
                               where + ".reasoning_content");
                }
                msg.reasoning_content = reasoning->as_string();
            }
        }
        job.messages.push_back(std::move(msg));
    }
    if (Status st = validate_chat_messages(job.messages); !st.ok()) {
        return bad("invalid_messages", st.message(), "messages");
    }
    if (auto e = read_chat_extensions(body, job)) return *e;
    if (auto e = read_response_format(body, job)) return *e;
    {
        std::string explicit_grammar;
        if (auto e = read_grammar_field(body, explicit_grammar)) return *e;
        if (!explicit_grammar.empty()) {
            if (job.response_format_type.has_value()) {
                return bad("invalid_request",
                           "grammar cannot be combined with response_format other than 'text'", "grammar");
            }
            job.grammar = std::move(explicit_grammar);
        }
    }
    if (auto e = read_request_hints(body, job.priority_class, job.deadline_ms)) {
        return *e;
    }

    job.sampling = SamplingConfig{};
    job.sampling.max_tokens = kDefaultMaxTokens;
    if (auto e = read_sampling(body, job.sampling)) {
        return *e;
    }
    if (const json::Value* lp = body.find("logprobs"); present(lp)) {
        if (!lp->is_bool()) {
            return bad("invalid_sampling", "logprobs must be a boolean", "logprobs");
        }
        job.sampling.logprobs = lp->as_bool();
        if (job.sampling.logprobs) {
            job.sampling.explicit_fields |= SamplingConfig::kLogprobs;
        }
    }
    if (const json::Value* tlp = body.find("top_logprobs"); present(tlp)) {
        std::int64_t k = 0;
        bool set = false;
        if (auto e = read_int(body, "top_logprobs", 0, SamplingConfig::kMaxTopLogprobs, k, set)) {
            return *e;
        }
        if (!job.sampling.logprobs) {
            return bad("invalid_sampling", "top_logprobs requires logprobs to be true", "top_logprobs");
        }
        job.sampling.top_logprobs = static_cast<std::int32_t>(k);
        job.sampling.explicit_fields |= SamplingConfig::kTopLogprobs;
    }
    if (Status st = validate(job.sampling); !st.ok()) {
        return bad("invalid_sampling", st.message());
    }
    return job;
}

std::variant<EmbedJob, ApiError> parse_embed_request(std::string_view body_text) {
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    EmbedJob job;
    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_request", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    if (const json::Value* fmt = body.find("encoding_format"); present(fmt)) {
        if (!fmt->is_string() || fmt->as_string() != "float") {
            return bad("unsupported_parameter", "only encoding_format 'float' is supported", "encoding_format");
        }
    }
    const json::Value* input = body.find("input");
    if (!present(input)) {
        return bad("invalid_request", "input is required (string or non-empty array of strings)", "input");
    }
    if (input->is_string()) {
        job.inputs.push_back(input->as_string());
    } else if (input->is_array()) {
        const auto& arr = input->as_array();
        if (arr.empty()) {
            return bad("invalid_request", "input array must not be empty", "input");
        }
        if (arr.size() > kMaxEmbedInputs) {
            return bad("invalid_request",
                       "input array exceeds maximum of " + std::to_string(kMaxEmbedInputs) + " items", "input");
        }
        job.inputs.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_string()) {
                return bad("invalid_request", "input[" + std::to_string(i) + "] must be a string", "input");
            }
            job.inputs.push_back(arr[i].as_string());
        }
    } else {
        return bad("invalid_request", "input must be a string or a non-empty array of strings", "input");
    }
    if (auto e = read_request_hints(body, job.priority_class, job.deadline_ms)) {
        return *e;
    }
    return job;
}

std::variant<CompletionJob, ApiError> parse_completion_request(std::string_view body_text) {
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    CompletionJob job;

    if (const json::Value* n = body.find("n"); present(n)) {
        if (!n->is_number() || n->as_double() != 1.0) {
            return bad("unsupported_parameter", "only n = 1 is supported", "n");
        }
    }
    if (const json::Value* best_of = body.find("best_of"); present(best_of)) {
        if (!best_of->is_number() || best_of->as_double() != 1.0) {
            return bad("unsupported_parameter", "only best_of = 1 is supported", "best_of");
        }
    }
    if (present(body.find("suffix"))) {
        return bad("unsupported_parameter", "suffix is not supported by sonder-inference", "suffix");
    }

    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_json", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    const json::Value* prompt = body.find("prompt");
    if (!present(prompt) || !prompt->is_string() || prompt->as_string().empty()) {
        return bad("invalid_request", "prompt must be a non-empty string", "prompt");
    }
    job.prompt = prompt->as_string();

    if (const json::Value* stream = body.find("stream"); present(stream)) {
        if (!stream->is_bool()) {
            return bad("invalid_json", "stream must be a boolean", "stream");
        }
        job.stream = stream->as_bool();
    }
    if (const json::Value* so = body.find("stream_options"); present(so)) {
        if (!so->is_object()) {
            return bad("invalid_json", "stream_options must be an object", "stream_options");
        }
        if (const json::Value* iu = so->find("include_usage"); present(iu)) {
            if (!iu->is_bool()) {
                return bad("invalid_json", "stream_options.include_usage must be a boolean",
                           "stream_options.include_usage");
            }
            job.include_usage = iu->as_bool();
        }
    }
    if (const json::Value* echo = body.find("echo"); present(echo)) {
        if (!echo->is_bool()) {
            return bad("invalid_request", "echo must be a boolean", "echo");
        }
        job.echo = echo->as_bool();
    }
    if (const json::Value* key = body.find("prompt_cache_key"); present(key)) {
        if (!key->is_string() || key->as_string().empty() || key->as_string().size() > 256) {
            return bad("invalid_json", "prompt_cache_key must be a string of 1 to 256 bytes", "prompt_cache_key");
        }
        job.session_key = key->as_string();
    }
    if (auto e = read_grammar_field(body, job.grammar)) return *e;
    if (auto e = read_request_hints(body, job.priority_class, job.deadline_ms)) return *e;

    job.sampling = SamplingConfig{};
    job.sampling.max_tokens = kDefaultMaxTokens;
    if (auto e = read_sampling(body, job.sampling)) return *e;
    if (const json::Value* lp = body.find("logprobs"); present(lp)) {
        if (lp->is_bool()) {
            job.sampling.logprobs = lp->as_bool();
            if (job.sampling.logprobs) {
                job.sampling.explicit_fields |= SamplingConfig::kLogprobs;
            }
        } else if (lp->is_integer()) {
            const auto exact = lp->try_as_int();
            if (!exact || *exact < 0 || *exact > SamplingConfig::kMaxTopLogprobs) {
                return bad("invalid_sampling",
                           "logprobs must be a boolean or an integer between 0 and " +
                               std::to_string(SamplingConfig::kMaxTopLogprobs),
                           "logprobs");
            }
            const std::int64_t k = *exact;
            if (k > 0) {
                job.sampling.logprobs = true;
                job.sampling.top_logprobs = static_cast<std::int32_t>(k);
                job.sampling.explicit_fields |= SamplingConfig::kLogprobs | SamplingConfig::kTopLogprobs;
            }
        } else {
            return bad("invalid_sampling", "logprobs must be a boolean or an integer between 0 and 20", "logprobs");
        }
    }
    if (const json::Value* tlp = body.find("top_logprobs"); present(tlp)) {
        std::int64_t k = 0;
        bool set = false;
        if (auto e = read_int(body, "top_logprobs", 0, SamplingConfig::kMaxTopLogprobs, k, set)) {
            return *e;
        }
        if (!job.sampling.logprobs) {
            return bad("invalid_sampling", "top_logprobs requires logprobs to be true", "top_logprobs");
        }
        job.sampling.top_logprobs = static_cast<std::int32_t>(k);
        job.sampling.explicit_fields |= SamplingConfig::kTopLogprobs;
    }
    if (Status st = validate(job.sampling); !st.ok()) {
        return bad("invalid_sampling", st.message());
    }
    return job;
}

std::variant<TokenizeJob, ApiError> parse_tokenize_request(std::string_view body_text) {
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    TokenizeJob job;
    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_request", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    const json::Value* content = body.find("content");
    if (!present(content)) {
        content = body.find("text");
    }
    if (!present(content) || !content->is_string()) {
        return bad("invalid_request", "content (or text) must be a string", "content");
    }
    job.content = content->as_string();
    return job;
}

std::variant<DetokenizeJob, ApiError> parse_detokenize_request(std::string_view body_text) {
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    DetokenizeJob job;
    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_request", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    const json::Value* tokens = body.find("tokens");
    if (!present(tokens) || !tokens->is_array()) {
        return bad("invalid_request", "tokens must be an array of integers", "tokens");
    }
    const auto& arr = tokens->as_array();
    job.tokens.reserve(arr.size());
    for (std::size_t i = 0; i < arr.size(); ++i) {
        if (!arr[i].is_integer()) {
            return bad("invalid_request", "tokens[" + std::to_string(i) + "] must be an integer", "tokens");
        }
        const auto exact = arr[i].try_as_int();
        if (!exact || *exact < std::numeric_limits<TokenId>::min() || *exact > std::numeric_limits<TokenId>::max()) {
            return bad("invalid_request", "tokens[" + std::to_string(i) + "] is out of 32-bit token id range",
                       "tokens");
        }
        job.tokens.push_back(static_cast<TokenId>(*exact));
    }
    return job;
}

std::optional<WorkloadClass> parse_workload(std::string_view name) noexcept {
    for (const auto w : {WorkloadClass::interactive_user, WorkloadClass::owner_orchestrator,
                         WorkloadClass::critic_verification, WorkloadClass::implementation_worker,
                         WorkloadClass::research_worker, WorkloadClass::background_indexing,
                         WorkloadClass::maintenance}) {
        if (name == to_string(w)) {
            return w;
        }
    }
    return std::nullopt;
}

std::variant<Correlation, ApiError> parse_correlation(const RequestHead& head) {
    for (const auto* name : {"x-sonder-priority", "x-sonder-deadline-ms"}) {
        std::size_t count = 0;
        for (const auto& field : head.headers) if (field.first == name) ++count;
        if (count > 1) return bad("invalid_correlation_header", std::string(name) + " must occur once", name);
    }
    Correlation c;
    if (auto error = reasoning_budget::parse_header(head, c)) return *error;
    const auto id_header = [&head](const char* lower, const char* display,
                                   std::optional<std::string>& out) -> std::optional<ApiError> {
        const std::string* v = head.header(lower);
        if (v == nullptr) {
            return std::nullopt;
        }
        if (!is_correlation_value(*v)) {
            return bad("invalid_correlation_header",
                       std::string(display) + " must match [A-Za-z0-9._:-]{1,128}", display);
        }
        out = *v;
        return std::nullopt;
    };
    if (auto e = id_header("x-sonder-run-id", "X-Sonder-Run-Id", c.run_id)) return *e;
    if (auto e = id_header("x-sonder-parent-request-id", "X-Sonder-Parent-Request-Id", c.parent_request_id)) return *e;
    if (auto e = id_header("x-sonder-agent-id", "X-Sonder-Agent-Id", c.agent_id)) return *e;
    if (auto e = id_header("x-sonder-task-id", "X-Sonder-Task-Id", c.task_id)) return *e;
    if (const std::string* w = head.header("x-sonder-workload")) {
        const auto parsed = parse_workload(*w);
        if (!parsed) {
            return bad("invalid_correlation_header",
                       "X-Sonder-Workload must be one of interactive_user, owner_orchestrator, critic_verification, "
                       "implementation_worker, research_worker, background_indexing, maintenance",
                       "X-Sonder-Workload");
        }
        c.workload = *parsed;
    }
    if (const std::string* p = head.header("x-sonder-priority")) {
        if (const auto numeric = parse_numeric_priority_impl(*p)) {
            c.priority = *numeric;
            c.numeric_priority = true;
            c.priority_class = RequestPriority::interactive;
        } else if (looks_numeric_priority(*p)) {
            return bad("invalid_correlation_header", "X-Sonder-Priority must be an integer from -16 to 16",
                       "X-Sonder-Priority");
        } else if (const auto class_name = parse_request_priority(*p)) {
            c.priority_class = *class_name;
        } else {
            return bad("invalid_correlation_header",
                       "X-Sonder-Priority must be an integer from -16 to 16 or interactive, subagent or background",
                       "X-Sonder-Priority");
        }
    }
    if (const std::string* d = head.header("x-sonder-deadline-ms")) {
        // Header parsing is deliberately strict decimal text; avoid floating
        // point conversion so uint64_t boundaries remain unambiguous.
        if (d->empty() || d->find_first_not_of("0123456789") != std::string::npos || d->size() > 20) {
            return bad("invalid_correlation_header", "X-Sonder-Deadline-Ms must be a positive integer", "X-Sonder-Deadline-Ms");
        }
        std::uint64_t n = 0;
        for (char ch : *d) {
            const auto digit = static_cast<unsigned>(ch - '0');
            if (n > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
                return bad("invalid_correlation_header", "X-Sonder-Deadline-Ms is out of range", "X-Sonder-Deadline-Ms");
            }
            n = n * 10 + digit;
        }
        if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            return bad("invalid_correlation_header", "X-Sonder-Deadline-Ms must be a positive integer within 32-bit milliseconds", "X-Sonder-Deadline-Ms");
        }
        c.deadline_ms = n;
    }
    return c;
}

const char* finish_reason(const GenerationResult& result) noexcept {
    if (result.outcome == RequestOutcome::cancelled) {
        return "cancelled";
    }
    if (result.stats.stop_reason == StopReason::tool_calls || !result.tool_calls.empty()) {
        return "tool_calls";
    }
    if (result.stats.stop_reason == StopReason::max_tokens) {
        return "length";
    }
    return "stop";
}

json::Object usage_json(const GenerationResult& result) {
    json::Object u{{"prompt_tokens", result.stats.prompt_tokens},
                   {"completion_tokens", result.stats.completion_tokens},
                   {"total_tokens", result.stats.prompt_tokens + result.stats.completion_tokens}};
    if (result.stats.cached_tokens) {
        u.set("prompt_tokens_details", json::Object{{"cached_tokens", *result.stats.cached_tokens}});
    }
    if (result.stats.backend_timings) {
        const auto& t = *result.stats.backend_timings;
        json::Object upstream;
        if (t.prompt_n) upstream.set("prompt_n", *t.prompt_n);
        if (t.cache_n) upstream.set("cache_n", *t.cache_n);
        if (t.prompt_ms) upstream.set("prompt_ms", *t.prompt_ms);
        if (t.predicted_n) upstream.set("predicted_n", *t.predicted_n);
        if (t.predicted_ms) upstream.set("predicted_ms", *t.predicted_ms);
        if (t.draft_n) upstream.set("draft_n", *t.draft_n);
        if (t.draft_n_accepted) upstream.set("draft_n_accepted", *t.draft_n_accepted);
        u.set("sonder", json::Object{{"timings", std::move(upstream)}});
        if (!result.stats.cached_tokens && t.cache_n) {
            u.set("prompt_tokens_details", json::Object{{"cached_tokens", *t.cache_n}});
        }
    }
    return u;
}

json::Object timings_json(const GenerationResult& result) {
    bool has_cached_tokens = false;
    std::uint64_t cached_tokens = 0;
    if (result.stats.cached_tokens) {
        has_cached_tokens = true;
        cached_tokens = *result.stats.cached_tokens;
    } else if (result.stats.backend_timings && result.stats.backend_timings->cache_n) {
        has_cached_tokens = true;
        cached_tokens = *result.stats.backend_timings->cache_n;
    }
    std::uint64_t evaluated_prompt_tokens = result.stats.prompt_tokens;
    if (result.stats.backend_timings && result.stats.backend_timings->prompt_n) {
        evaluated_prompt_tokens = *result.stats.backend_timings->prompt_n;
    } else if (has_cached_tokens) {
        evaluated_prompt_tokens -= std::min(evaluated_prompt_tokens, cached_tokens);
    }

    json::Object t{{"prompt_n", evaluated_prompt_tokens},
                   {"predicted_n", result.stats.completion_tokens},
                   {"total_ms", result.total_ms}};
    if (result.ttft_ms >= 0.0) {
        t.set("ttft_ms", result.ttft_ms);
    }
    if (result.stats.prompt_eval_ns > 0) {
        const double ms = static_cast<double>(result.stats.prompt_eval_ns) / 1e6;
        t.set("prompt_ms", ms);
        if (evaluated_prompt_tokens > 0 && ms > 0.0) {
            t.set("prompt_per_second", static_cast<double>(evaluated_prompt_tokens) * 1e3 / ms);
        }
    }
    if (result.stats.eval_ns > 0) {
        const double ms = static_cast<double>(result.stats.eval_ns) / 1e6;
        t.set("predicted_ms", ms);
        if (result.stats.completion_tokens > 0 && ms > 0.0) {
            t.set("predicted_per_second", static_cast<double>(result.stats.completion_tokens) * 1e3 / ms);
        }
    }
    // Backend observations: present only when reported (0 is a real value).
    if (has_cached_tokens) {
        t.set("cache_n", cached_tokens);
    }
    if (result.stats.draft_tokens) {
        t.set("draft_n", *result.stats.draft_tokens);
    }
    if (result.stats.draft_accepted_tokens) {
        t.set("draft_n_accepted", *result.stats.draft_accepted_tokens);
    }
    if (result.scheduling.scheduled) {
        t.set("queue_ms", result.scheduling.queue_ms);
    }
    return t;
}

namespace {

// "... does not support <field>" from a backend refusing a request field
// (for example "the ollama backend does not support logit_bias"). Returns
// the field when it is a plain identifier.
std::optional<std::string> unsupported_field(std::string_view message) {
    constexpr std::string_view kMarker = "does not support ";
    const std::size_t at = message.find(kMarker);
    if (at == std::string_view::npos) {
        return std::nullopt;
    }
    std::string_view rest = message.substr(at + kMarker.size());
    std::size_t n = 0;
    while (n < rest.size() && ((rest[n] >= 'a' && rest[n] <= 'z') || (rest[n] >= '0' && rest[n] <= '9') ||
                               rest[n] == '_')) {
        ++n;
    }
    if (n == 0 || (n < rest.size() && rest[n] != ' ' && rest[n] != ';' && rest[n] != ',' && rest[n] != '.')) {
        return std::nullopt;
    }
    return std::string(rest.substr(0, n));
}

}  // namespace

ApiError map_session_failure(const Status& st, bool scheduler_rejected) {
    if (scheduler_rejected && st.code() != ErrorCode::invalid_argument) {
        ApiError e = make_error(429, "overloaded", "the engine scheduler rejected the request: " + st.message());
        e.retry_after = 1;
        return e;
    }
    switch (st.code()) {
        case ErrorCode::invalid_argument:
            if (scheduler_rejected) {
                // The prompt can never fit the engine's KV pool.
                return make_error(400, "invalid_messages", st.message(), "messages");
            }
            if (st.message().rfind("sampling failed", 0) == 0) {
                return make_error(400, "invalid_sampling", st.message());
            }
            // parse_chat_request() already validated the messages, so this is
            // the engine or backend refusing a request parameter.
            if (auto field = unsupported_field(st.message())) {
                return make_error(400, "unsupported_parameter", st.message(), *field);
            }
            return make_error(400, "invalid_request", "the backend refused the request: " + st.message());
        case ErrorCode::not_found: return make_error(404, "model_not_found", st.message(), "model");
        case ErrorCode::unsupported: return make_error(400, "unsupported_parameter", st.message());
        case ErrorCode::unavailable:
        case ErrorCode::timeout:
        case ErrorCode::backend_error:
        case ErrorCode::protocol_error:
        case ErrorCode::io_error:
            return make_error(503, "backend_unavailable",
                              "backend failed while executing the request (not safe to replay): " + st.message());
        default: return make_error(500, "internal_error", st.message());
    }
}

}  // namespace sonder::inference::server::detail
