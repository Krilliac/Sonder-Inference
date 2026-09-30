#include "openai.hpp"

#include <cmath>
#include <limits>

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
    if (v->is_integer()) {
        n = v->as_int();
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

}  // namespace

std::variant<ChatJob, ApiError> parse_chat_request(std::string_view body_text) {
    auto parsed = json::parse(body_text);
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    ChatJob job;

    // Features this API does not provide (rejected rather than ignored, so a
    // client never silently loses tools or structured output).
    for (const char* key : {"tools", "tool_choice", "functions", "function_call", "response_format", "top_logprobs"}) {
        if (present(body.find(key))) {
            return bad("unsupported_parameter", std::string(key) + " is not supported by sonder-inference", key);
        }
    }
    if (const json::Value* lp = body.find("logprobs"); present(lp) && !(lp->is_bool() && !lp->as_bool())) {
        return bad("unsupported_parameter", "logprobs is not supported by sonder-inference", "logprobs");
    }
    if (const json::Value* n = body.find("n"); present(n)) {
        if (!n->is_number() || n->as_double() != 1.0) {
            return bad("unsupported_parameter", "only n = 1 is supported", "n");
        }
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
        if (present(m.find("tool_calls"))) {
            return bad("unsupported_parameter", where + ".tool_calls is not supported", where + ".tool_calls");
        }
        const json::Value* content = m.find("content");
        if (!present(content) || !content->is_string()) {
            return bad("unsupported_parameter", where + ".content must be a string (content parts are not supported)",
                       where + ".content");
        }
        job.messages.push_back(ChatMessage{role->as_string(), content->as_string()});
    }
    if (Status st = validate_chat_messages(job.messages); !st.ok()) {
        return bad("invalid_messages", st.message(), "messages");
    }

    job.sampling = SamplingConfig{};
    job.sampling.max_tokens = kDefaultMaxTokens;
    if (auto e = read_sampling(body, job.sampling)) {
        return *e;
    }
    if (Status st = validate(job.sampling); !st.ok()) {
        return bad("invalid_sampling", st.message());
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
    Correlation c;
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
        std::string_view text = *p;
        bool negative = false;
        if (!text.empty() && (text.front() == '-' || text.front() == '+')) {
            negative = text.front() == '-';
            text.remove_prefix(1);
        }
        if (text.empty() || text.size() > 2 || text.find_first_not_of("0123456789") != std::string_view::npos) {
            return bad("invalid_correlation_header", "X-Sonder-Priority must be an integer from -16 to 16",
                       "X-Sonder-Priority");
        }
        int v = std::stoi(std::string(text));
        v = negative ? -v : v;
        if (v < -16 || v > 16) {
            return bad("invalid_correlation_header", "X-Sonder-Priority must be an integer from -16 to 16",
                       "X-Sonder-Priority");
        }
        c.priority = v;
    }
    return c;
}

const char* finish_reason(const GenerationResult& result) noexcept {
    if (result.outcome == RequestOutcome::cancelled) {
        return "cancelled";
    }
    if (result.stats.stop_reason == StopReason::max_tokens) {
        return "length";
    }
    return "stop";
}

json::Object usage_json(const GenerationResult& result) {
    return json::Object{{"prompt_tokens", result.stats.prompt_tokens},
                        {"completion_tokens", result.stats.completion_tokens},
                        {"total_tokens", result.stats.prompt_tokens + result.stats.completion_tokens}};
}

json::Object timings_json(const GenerationResult& result) {
    json::Object t{{"prompt_n", result.stats.prompt_tokens},
                   {"predicted_n", result.stats.completion_tokens},
                   {"total_ms", result.total_ms}};
    if (result.ttft_ms >= 0.0) {
        t.set("ttft_ms", result.ttft_ms);
    }
    if (result.stats.prompt_eval_ns > 0) {
        const double ms = static_cast<double>(result.stats.prompt_eval_ns) / 1e6;
        t.set("prompt_ms", ms);
        if (result.stats.prompt_tokens > 0) {
            t.set("prompt_per_second", static_cast<double>(result.stats.prompt_tokens) * 1e3 / ms);
        }
    }
    if (result.stats.eval_ns > 0) {
        const double ms = static_cast<double>(result.stats.eval_ns) / 1e6;
        t.set("predicted_ms", ms);
        if (result.stats.completion_tokens > 0) {
            t.set("predicted_per_second", static_cast<double>(result.stats.completion_tokens) * 1e3 / ms);
        }
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
