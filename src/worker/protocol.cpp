#include "protocol.hpp"
#include "ipc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sonder::inference::worker {
namespace {
Status bad(std::string text) { return {ErrorCode::protocol_error, "worker: " + text}; }
bool unsigned_value(const json::Value &v) { return v.is_integer() && (!v.try_as_int() || *v.try_as_int() >= 0); }
bool real(const json::Value *v) { return v && v->is_number() && std::isfinite(v->as_double()); }
bool finite_float(const json::Value *v) {
    return real(v) && std::abs(v->as_double()) <= static_cast<double>(std::numeric_limits<float>::max());
}
bool signed32(const json::Value *v) {
    if (!v || !v->try_as_int())
        return false;
    const auto n = *v->try_as_int();
    return n >= INT32_MIN && n <= INT32_MAX;
}
} // namespace

json::Object message(std::string_view type, std::uint64_t id) {
    return {{"v", kProtocolVersion}, {"type", type}, {"id", id}};
}
Status fields(const json::Value &value, std::initializer_list<std::string_view> required) {
    if (!value.is_object() || value.as_object().size() != required.size())
        return bad("unexpected fields");
    for (const auto key : required)
        if (!value.find(key))
            return bad("missing " + std::string(key));
    return {};
}
Result<std::uint64_t> unsigned_field(const json::Value &v, std::string_view key) {
    const auto *p = v.find(key);
    if (!p || !unsigned_value(*p))
        return bad("invalid unsigned " + std::string(key));
    return p->as_uint();
}
Result<std::string> string_field(const json::Value &v, std::string_view key) {
    const auto *p = v.find(key);
    if (!p || !p->is_string())
        return bad("invalid string " + std::string(key));
    return p->as_string();
}
Result<json::Value> parse_message(std::string_view frame) {
    if (frame.empty() || frame.size() > kMaxFrameBytes)
        return bad("frame size");
    auto parsed = json::parse(frame, {24, 16384});
    if (!parsed || !parsed->is_object())
        return bad("invalid JSON object");
    // SI is both endpoints: require its canonical compact representation.
    // This also rejects duplicate keys, invalid UTF-8 and invalid surrogate
    // escapes instead of accepting the generic parser's repair semantics.
    if (parsed->dump() != frame)
        return bad("noncanonical JSON");
    const auto *v = parsed->find("v");
    auto type = string_field(parsed.value(), "type");
    auto id = unsigned_field(parsed.value(), "id");
    if (!v || !v->try_as_int() || *v->try_as_int() != kProtocolVersion || !type || type->empty() || !id)
        return bad("invalid protocol envelope");
    return parsed;
}
json::Value sampling_to_json(const SamplingConfig &s) {
    json::Array biases, stop;
    for (const auto &b : s.logit_bias)
        biases.emplace_back(json::Object{{"token", b.token},
                                         {"bias", std::isinf(b.bias) ? json::Value(nullptr) : json::Value(b.bias)}});
    for (const auto &p : s.stop)
        stop.emplace_back(p);
    return json::Object{{"temperature", s.temperature},
                        {"top_p", s.top_p},
                        {"top_k", s.top_k},
                        {"min_p", s.min_p},
                        {"repeat_penalty", s.repeat_penalty},
                        {"typical_p", s.typical_p},
                        {"repeat_last_n", s.repeat_last_n},
                        {"presence_penalty", s.presence_penalty},
                        {"frequency_penalty", s.frequency_penalty},
                        {"logit_bias", std::move(biases)},
                        {"num_ctx", s.num_ctx},
                        {"seed", s.seed},
                        {"max_tokens", s.max_tokens},
                        {"stop", std::move(stop)},
                        {"logprobs", s.logprobs},
                        {"top_logprobs", s.top_logprobs},
                        {"explicit_only", s.explicit_only},
                        {"explicit_fields", s.explicit_fields}};
}
Result<SamplingConfig> sampling_from_json(const json::Value &v) {
    auto st = fields(v, {"temperature", "top_p", "top_k", "min_p", "repeat_penalty", "typical_p", "repeat_last_n",
                         "presence_penalty", "frequency_penalty", "logit_bias", "num_ctx", "seed", "max_tokens", "stop",
                         "logprobs", "top_logprobs", "explicit_only", "explicit_fields"});
    if (!st.ok())
        return st;
    for (const auto key :
         {"temperature", "top_p", "min_p", "repeat_penalty", "typical_p", "presence_penalty", "frequency_penalty"})
        if (!finite_float(v.find(key)))
            return bad("invalid sampler number");
    for (const auto key : {"top_k", "repeat_last_n", "num_ctx", "max_tokens", "top_logprobs"})
        if (!signed32(v.find(key)))
            return bad("invalid sampler integer");
    if (!v.find("logprobs")->is_bool() || !v.find("explicit_only")->is_bool() || !v.find("stop")->is_array() ||
        !v.find("logit_bias")->is_array())
        return bad("invalid sampler shape");
    auto mask = unsigned_field(v, "explicit_fields");
    if (!mask || mask.value() > 0xffffu)
        return bad("unknown sampler fields");
    SamplingConfig s;
    s.temperature = static_cast<float>(v.find("temperature")->as_double());
    s.top_p = static_cast<float>(v.find("top_p")->as_double());
    s.top_k = static_cast<std::int32_t>(v.find("top_k")->as_int());
    s.min_p = static_cast<float>(v.find("min_p")->as_double());
    s.repeat_penalty = static_cast<float>(v.find("repeat_penalty")->as_double());
    s.typical_p = static_cast<float>(v.find("typical_p")->as_double());
    s.repeat_last_n = static_cast<std::int32_t>(v.find("repeat_last_n")->as_int());
    s.presence_penalty = static_cast<float>(v.find("presence_penalty")->as_double());
    s.frequency_penalty = static_cast<float>(v.find("frequency_penalty")->as_double());
    s.num_ctx = static_cast<std::int32_t>(v.find("num_ctx")->as_int());
    s.max_tokens = static_cast<std::int32_t>(v.find("max_tokens")->as_int());
    s.logprobs = v.find("logprobs")->as_bool();
    s.top_logprobs = static_cast<std::int32_t>(v.find("top_logprobs")->as_int());
    s.explicit_only = v.find("explicit_only")->as_bool();
    s.explicit_fields = static_cast<std::uint32_t>(mask.value());
    const auto *seed = v.find("seed");
    if (!seed->is_null()) {
        if (!unsigned_value(*seed))
            return bad("invalid seed");
        s.seed = seed->as_uint();
    }
    if (v.find("stop")->as_array().size() > SamplingConfig::kMaxStopSequences ||
        v.find("logit_bias")->as_array().size() > SamplingConfig::kMaxLogitBiasEntries)
        return bad("sampler array size");
    for (const auto &p : v.find("stop")->as_array()) {
        if (!p.is_string())
            return bad("invalid stop");
        s.stop.push_back(p.as_string());
    }
    for (const auto &b : v.find("logit_bias")->as_array()) {
        if (!fields(b, {"token", "bias"}).ok() || !signed32(b.find("token")) ||
            (!b.find("bias")->is_null() &&
             (!finite_float(b.find("bias")) ||
              std::abs(b.find("bias")->as_double()) > SamplingConfig::kMaxLogitBiasMagnitude)))
            return bad("invalid bias");
        s.logit_bias.push_back({static_cast<std::int32_t>(b.find("token")->as_int()),
                                b.find("bias")->is_null() ? -std::numeric_limits<float>::infinity()
                                                          : static_cast<float>(b.find("bias")->as_double())});
    }
    if (auto valid = validate(s); !valid.ok())
        return bad(valid.message());
    return s;
}
json::Value descriptor_to_json(const ModelDescriptor &d) {
    return json::Object{{"name", d.name},
                        {"backend", d.backend},
                        {"format", d.format},
                        {"family", d.family},
                        {"parameter_size", d.parameter_size},
                        {"quantization", d.quantization},
                        {"size_bytes", d.size_bytes},
                        {"context_length", d.context_length},
                        {"resident", d.resident}};
}
Result<ModelDescriptor> descriptor_from_json(const json::Value &v) {
    if (!fields(v, {"name", "backend", "format", "family", "parameter_size", "quantization", "size_bytes",
                    "context_length", "resident"})
             .ok())
        return bad("descriptor fields");
    for (const auto key : {"name", "backend", "format", "family", "parameter_size", "quantization"})
        if (!v.find(key)->is_string() || v.find(key)->as_string().size() > 4096 ||
            v.find(key)->as_string().find('\0') != std::string::npos)
            return bad("descriptor string");
    auto bytes = unsigned_field(v, "size_bytes"), ctx = unsigned_field(v, "context_length");
    if (!bytes || !ctx || !v.find("resident")->is_bool() || bytes.value() > kMaxModelBytes || ctx.value() == 0 ||
        ctx.value() > static_cast<std::uint64_t>(SamplingConfig::kMaxContextLimit) ||
        v.find("name")->as_string().empty() || v.find("backend")->as_string() != "mock")
        return bad("descriptor values");
    ModelDescriptor d;
    d.name = v.find("name")->as_string();
    d.backend = "remote-mock";
    d.format = v.find("format")->as_string();
    d.family = v.find("family")->as_string();
    d.parameter_size = v.find("parameter_size")->as_string();
    d.quantization = v.find("quantization")->as_string();
    d.size_bytes = bytes.value();
    d.context_length = ctx.value();
    d.resident = v.find("resident")->as_bool();
    return d;
}
json::Value stats_to_json(const GenerateStats &s) {
    return json::Object{{"prompt_tokens", s.prompt_tokens},
                        {"completion_tokens", s.completion_tokens},
                        {"chunks", s.chunks},
                        {"load_ns", s.load_ns},
                        {"prompt_eval_ns", s.prompt_eval_ns},
                        {"eval_ns", s.eval_ns},
                        {"token_counts_from_backend", s.token_counts_from_backend},
                        {"stop_reason", static_cast<int>(s.stop_reason)},
                        {"matched_stop", s.matched_stop}};
}
Result<GenerateStats> stats_from_json(const json::Value &v) {
    if (!fields(v, {"prompt_tokens", "completion_tokens", "chunks", "load_ns", "prompt_eval_ns", "eval_ns",
                    "token_counts_from_backend", "stop_reason", "matched_stop"})
             .ok())
        return bad("stats fields");
    for (const auto key : {"prompt_tokens", "completion_tokens", "chunks", "load_ns", "prompt_eval_ns", "eval_ns"})
        if (!unsigned_field(v, key))
            return bad("stats counter");
    if (v.find("prompt_tokens")->as_uint() > static_cast<std::uint64_t>(SamplingConfig::kMaxContextLimit) ||
        v.find("completion_tokens")->as_uint() > static_cast<std::uint64_t>(SamplingConfig::kMaxTokensLimit) ||
        v.find("chunks")->as_uint() > static_cast<std::uint64_t>(SamplingConfig::kMaxTokensLimit))
        return bad("stats token count exceeds limits");
    for (const auto key : {"load_ns", "prompt_eval_ns", "eval_ns"})
        if (v.find(key)->as_uint() > kMaxWorkerTimingNs)
            return bad("untrusted worker timing exceeds diagnostic limit");
    const auto reason = v.find("stop_reason")->try_as_int();
    if (!reason || *reason < 0 || *reason > static_cast<int>(StopReason::error) ||
        !v.find("token_counts_from_backend")->is_bool() ||
        (!v.find("matched_stop")->is_null() &&
         (!v.find("matched_stop")->is_string() ||
          v.find("matched_stop")->as_string().size() > SamplingConfig::kMaxStopSequenceBytes)))
        return bad("stats value");
    GenerateStats s;
    s.prompt_tokens = v.find("prompt_tokens")->as_uint();
    s.completion_tokens = v.find("completion_tokens")->as_uint();
    s.chunks = v.find("chunks")->as_uint();
    s.load_ns = v.find("load_ns")->as_uint();
    s.prompt_eval_ns = v.find("prompt_eval_ns")->as_uint();
    s.eval_ns = v.find("eval_ns")->as_uint();
    s.token_counts_from_backend = v.find("token_counts_from_backend")->as_bool();
    s.stop_reason = static_cast<StopReason>(*reason);
    if (!v.find("matched_stop")->is_null())
        s.matched_stop = v.find("matched_stop")->as_string();
    return s;
}
json::Value chunk_to_json(const TokenChunk &c) {
    json::Array top;
    for (const auto &p : c.top_logprobs)
        top.emplace_back(json::Object{{"token", p.token}, {"logprob", p.logprob}});
    return json::Object{{"text", c.text},
                        {"index", c.index},
                        {"reasoning", c.reasoning},
                        {"logprob", c.logprob},
                        {"top_logprobs", std::move(top)}};
}
Result<TokenChunk> chunk_from_json(const json::Value &v) {
    if (!fields(v, {"text", "index", "reasoning", "logprob", "top_logprobs"}).ok() || !v.find("text")->is_string() ||
        !v.find("reasoning")->is_string() || !unsigned_field(v, "index") ||
        (!v.find("logprob")->is_null() && !finite_float(v.find("logprob"))) || !v.find("top_logprobs")->is_array() ||
        v.find("top_logprobs")->as_array().size() > static_cast<std::size_t>(SamplingConfig::kMaxTopLogprobs))
        return bad("token fields");
    TokenChunk c;
    c.text = v.find("text")->as_string();
    c.index = v.find("index")->as_uint();
    c.reasoning = v.find("reasoning")->as_string();
    if (!v.find("logprob")->is_null())
        c.logprob = static_cast<float>(v.find("logprob")->as_double());
    for (const auto &p : v.find("top_logprobs")->as_array()) {
        if (!fields(p, {"token", "logprob"}).ok() || !p.find("token")->is_string() || !finite_float(p.find("logprob")))
            return bad("candidate fields");
        c.top_logprobs.push_back({p.find("token")->as_string(), static_cast<float>(p.find("logprob")->as_double())});
    }
    return c;
}
json::Value error_message(std::uint64_t id, const Status &s) {
    auto out = message("error", id);
    out.set("code", static_cast<int>(s.code()));
    out.set("message", s.message().substr(0, 2048));
    return out;
}
namespace {
json::Value priority_json(const std::optional<RequestPriority> &p) {
    return p ? json::Value(static_cast<int>(*p)) : json::Value(nullptr);
}
template <class R> Status read_common(const json::Value &v, R &r) {
    auto id = string_field(v, "request_id"), key = string_field(v, "session_key");
    if (!id || !key)
        return bad("request strings");
    auto sampling = sampling_from_json(*v.find("sampling"));
    if (!sampling)
        return sampling.status();
    r.request_id = std::move(id).value();
    r.session_key = std::move(key).value();
    r.sampling = std::move(sampling).value();
    const auto *priority = v.find("priority_class");
    if (!priority->is_null()) {
        auto n = priority->try_as_int();
        if (!n || *n < 0 || *n > static_cast<int>(RequestPriority::background))
            return bad("priority");
        r.priority_class = static_cast<RequestPriority>(*n);
    }
    for (const auto name : {"target_ttft_ms", "target_itl_ms"}) {
        const auto *p = v.find(name);
        if (!p->is_null() && (!real(p) || p->as_double() < 0))
            return bad("latency target");
    }
    if (!v.find("target_ttft_ms")->is_null())
        r.target_ttft_ms = v.find("target_ttft_ms")->as_double();
    if (!v.find("target_itl_ms")->is_null())
        r.target_itl_ms = v.find("target_itl_ms")->as_double();
    return {};
}
template <class R> json::Object common_request(const R &r) {
    return {{"request_id", r.request_id},
            {"session_key", r.session_key},
            {"sampling", sampling_to_json(r.sampling)},
            {"priority_class", priority_json(r.priority_class)},
            {"target_ttft_ms", r.target_ttft_ms},
            {"target_itl_ms", r.target_itl_ms}};
}
} // namespace
Result<json::Value> generate_to_json(const GenerateRequest &r) {
    if (!r.grammar.empty() || r.checkpoint || r.enable_stage_profiling || r.on_safe_point || r.suspended_state ||
        r.execution_resume || r.allow_preempt_spill)
        return Status(ErrorCode::unsupported, "worker v1 does not support grammar, profiling or execution state");
    if (auto st = validate(r.sampling); !st.ok())
        return st;
    auto out = common_request(r);
    out.set("prompt", r.prompt);
    out.set("priority_rank", r.priority_rank);
    json::Value encoded(std::move(out));
    // Apply the same checks as the worker before any bytes cross IPC. In
    // particular, validate() alone does not check priority, latency targets or
    // the protocol's sampling explicit-field mask.
    if (auto decoded = generate_from_json(encoded); !decoded)
        return Status(ErrorCode::invalid_argument, decoded.status().message());
    return encoded;
}
Result<GenerateRequest> generate_from_json(const json::Value &v) {
    if (!fields(v, {"request_id", "session_key", "sampling", "priority_class", "target_ttft_ms", "target_itl_ms",
                    "prompt", "priority_rank"})
             .ok())
        return bad("generate fields");
    GenerateRequest r;
    if (auto st = read_common(v, r); !st.ok())
        return st;
    auto prompt = string_field(v, "prompt");
    if (!prompt || !signed32(v.find("priority_rank")))
        return bad("generate payload");
    r.prompt = std::move(prompt).value();
    r.priority_rank = static_cast<int>(v.find("priority_rank")->as_int());
    return r;
}
Result<json::Value> chat_to_json(const ChatRequest &r) {
    // serve sets this output-routing hint even for plain text chat. It does not
    // enable thinking; preserve it on the wire while rejecting thinking controls.
    if (!r.thinking.empty() || r.reasoning_budget_tokens || r.reasoning_budget_message || !r.grammar.empty() ||
        (r.response_format_type && *r.response_format_type != "text") || r.response_format_schema || !r.tools.empty() ||
        r.tool_choice || r.parallel_tool_calls || r.enable_stage_profiling)
        return Status(ErrorCode::unsupported, "worker v1 supports text-only chat without tools, reasoning, "
                                              "grammar or profiling");
    if (auto st = validate(r.sampling); !st.ok())
        return st;
    if (r.messages.size() > kMaxChatMessages)
        return Status(ErrorCode::invalid_argument,
                      "worker chat allows at most " + std::to_string(kMaxChatMessages) + " messages");
    if (auto st = validate_chat_messages(r.messages); !st.ok())
        return st;
    std::vector<ChatMessage> flattened;
    flattened.reserve(r.messages.size());
    for (const auto &m : r.messages) {
        if (m.reasoning_content || !m.tool_calls.empty() || m.tool_call_id || m.name || count_message_images(m) > 0)
            return Status(ErrorCode::unsupported, "worker v1 supports plain text chat messages only");
        ChatMessage flat;
        flat.role = m.role;
        flat.content = message_text_content(m);
        flattened.push_back(std::move(flat));
    }
    // Validate the exact representation the worker receives, not just the
    // richer input (parts/media and legacy content can differ).
    if (auto st = validate_chat_messages(flattened); !st.ok())
        return st;
    json::Array messages;
    messages.reserve(flattened.size());
    for (const auto &m : flattened)
        messages.emplace_back(json::Object{{"role", m.role}, {"content", m.content}});
    auto out = common_request(r);
    out.set("messages", std::move(messages));
    out.set("separate_reasoning", r.separate_reasoning);
    json::Value encoded(std::move(out));
    if (auto decoded = chat_from_json(encoded); !decoded)
        return Status(ErrorCode::invalid_argument, decoded.status().message());
    return encoded;
}
Result<ChatRequest> chat_from_json(const json::Value &v) {
    if (!fields(v, {"request_id", "session_key", "sampling", "priority_class", "target_ttft_ms", "target_itl_ms",
                    "messages", "separate_reasoning"})
             .ok())
        return bad("chat fields");
    ChatRequest r;
    if (auto st = read_common(v, r); !st.ok())
        return st;
    const auto *separate_reasoning = v.find("separate_reasoning");
    if (!separate_reasoning->is_bool())
        return bad("chat separate_reasoning must be a boolean");
    r.separate_reasoning = separate_reasoning->as_bool();
    const auto *messages = v.find("messages");
    if (!messages->is_array() || messages->as_array().size() > kMaxChatMessages)
        return bad("chat messages");
    for (const auto &m : messages->as_array()) {
        if (!fields(m, {"role", "content"}).ok() || !m.find("role")->is_string() || !m.find("content")->is_string())
            return bad("chat message");
        ChatMessage next;
        next.role = m.find("role")->as_string();
        next.content = m.find("content")->as_string();
        r.messages.push_back(std::move(next));
    }
    if (auto st = validate_chat_messages(r.messages); !st.ok())
        return bad(st.message());
    return r;
}
Status status_from_message(const json::Value &v) {
    if (!fields(v, {"v", "type", "id", "code", "message"}).ok())
        return bad("error fields");
    const auto code = v.find("code")->try_as_int();
    if (!code || *code <= 0 || *code > static_cast<int>(ErrorCode::internal) || !v.find("message")->is_string() ||
        v.find("message")->as_string().size() > 2048)
        return bad("invalid error");
    return {static_cast<ErrorCode>(*code), v.find("message")->as_string()};
}
} // namespace sonder::inference::worker
