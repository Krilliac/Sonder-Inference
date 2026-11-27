// Ollama Cloud wire format: base URL parsing, request bodies, NDJSON stream
// decoding, /api/tags parsing and HTTP error mapping.
#include "ollama_cloud_protocol.hpp"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <string>
#include <utility>

#include "net/http_client.hpp"

namespace sonder::inference::ollama_cloud {

namespace {

std::string str_field(const json::Value& v, std::string_view key) {
    const json::Value* f = v.find(key);
    return (f != nullptr && f->is_string()) ? f->as_string() : std::string();
}

std::string_view str_view(const json::Value& v, std::string_view key) {
    const json::Value* f = v.find(key);
    if (f == nullptr || !f->is_string()) {
        return {};
    }
    return f->as_string();
}

std::int64_t int_field(const json::Value& v, std::string_view key) {
    const json::Value* f = v.find(key);
    if (f == nullptr || !f->is_number()) {
        return 0;
    }
    return f->as_int(0);  // 0 for out-of-range values (no UB cast)
}

bool starts_with_ci(std::string_view text, std::string_view prefix) {
    if (text.size() < prefix.size()) {
        return false;
    }
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(text[i])) != static_cast<unsigned char>(prefix[i])) {
            return false;
        }
    }
    return true;
}

std::string bounded(std::string_view text, std::size_t max) {
    std::string t(text.substr(0, std::min(text.size(), max)));
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ' || t.back() == '\t')) {
        t.pop_back();
    }
    if (text.size() > max) {
        t += "...";
    }
    return t;
}

}  // namespace

Result<BaseUrl> parse_base_url(std::string_view text) {
    constexpr std::string_view kScheme = "https://";
    if (!starts_with_ci(text, kScheme)) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: base_url must start with https://");
    }
    const std::string_view rest = text.substr(kScheme.size());
    if (rest.find('?') != std::string_view::npos) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: base_url must not contain a query");
    }
    // net::parse_url accepts https:// only in TLS builds; validate host, port
    // and path through its http:// grammar and apply the https default port.
    auto parsed = net::parse_url("http://" + std::string(rest));
    if (!parsed.ok()) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: invalid base_url: " + parsed.status().message());
    }
    const std::string_view authority = rest.substr(0, rest.find('/'));
    bool explicit_port = false;
    if (!authority.empty() && authority.front() == '[') {
        explicit_port = authority.find("]:") != std::string_view::npos;
    } else {
        explicit_port = authority.find(':') != std::string_view::npos;
    }
    BaseUrl out;
    out.host = parsed.value().host;
    out.port = explicit_port ? parsed.value().port : static_cast<std::uint16_t>(443);
    out.path = parsed.value().path.empty() ? std::string("/") : parsed.value().path;
    return out;
}

std::string join_target(const std::string& base_path, std::string_view api_path) {
    if (base_path.empty() || base_path == "/") {
        return std::string(api_path);
    }
    std::string t = base_path;
    if (t.back() == '/') {
        t.pop_back();
    }
    return t + std::string(api_path);
}

json::Object sampling_to_options(const SamplingConfig& s) {
    json::Object o;
    // With explicit_only, a field goes out only when the caller set it, so the
    // model keeps its own defaults for the rest.
    const auto send = [&](SamplingConfig::Field f) { return !s.explicit_only || s.is_explicit(f); };
    if (send(SamplingConfig::kTemperature)) {
        o.set("temperature", static_cast<double>(s.temperature));
    }
    if (send(SamplingConfig::kTopP)) {
        o.set("top_p", static_cast<double>(s.top_p));
    }
    if (send(SamplingConfig::kTopK)) {
        o.set("top_k", s.top_k);
    }
    if (send(SamplingConfig::kMinP)) {
        o.set("min_p", static_cast<double>(s.min_p));
    }
    if (send(SamplingConfig::kRepeatPenalty)) {
        o.set("repeat_penalty", static_cast<double>(s.repeat_penalty));
    }
    if (s.num_ctx > 0) {
        o.set("num_ctx", s.num_ctx);
    }
    if (s.explicit_only ? s.is_explicit(SamplingConfig::kRepeatLastN)
                        : s.repeat_last_n != SamplingConfig{}.repeat_last_n) {
        o.set("repeat_last_n", s.repeat_last_n);
    }
    if (s.explicit_only ? s.is_explicit(SamplingConfig::kPresencePenalty) : s.presence_penalty != 0.0f) {
        o.set("presence_penalty", static_cast<double>(s.presence_penalty));
    }
    if (s.explicit_only ? s.is_explicit(SamplingConfig::kFrequencyPenalty) : s.frequency_penalty != 0.0f) {
        o.set("frequency_penalty", static_cast<double>(s.frequency_penalty));
    }
    if (s.seed) {
        o.set("seed", static_cast<std::int64_t>(*s.seed & 0x7FFFFFFFFFFFFFFFull));  // Ollama seeds are signed
    }
    o.set("num_predict", s.max_tokens);
    if (!s.stop.empty()) {
        json::Array stops;
        for (const auto& st : s.stop) {
            stops.emplace_back(st);
        }
        o.set("stop", std::move(stops));
    }
    return o;
}

std::string build_chat_body(const std::string& model, const ChatRequest& request) {
    json::Object body;
    body.set("model", model);
    body.set("stream", true);
    if (request.thinking.enable_thinking) {
        body.set("think", *request.thinking.enable_thinking);
    }
    json::Array msgs;
    msgs.reserve(request.messages.size());
    for (const auto& m : request.messages) {
        json::Object msg{{"role", m.role}, {"content", message_text_content(m)}};
        if (m.role == "assistant" && !m.tool_calls.empty()) {
            json::Array tcs;
            tcs.reserve(m.tool_calls.size());
            for (const auto& tc : m.tool_calls) {
                json::Value args = json::Object{};
                if (!tc.arguments_json.empty()) {
                    auto parsed = json::parse(tc.arguments_json);
                    if (parsed.ok() && parsed.value().is_object()) {
                        args = std::move(parsed).value();
                    }
                }
                json::Object entry{{"function", json::Object{{"name", tc.name}, {"arguments", std::move(args)}}}};
                if (!tc.id.empty()) {
                    entry.set("id", tc.id);
                }
                tcs.emplace_back(std::move(entry));
            }
            msg.set("tool_calls", std::move(tcs));
        }
        if (m.role == "tool") {
            if (m.name && !m.name->empty()) {
                msg.set("tool_name", *m.name);
            }
            if (m.tool_call_id && !m.tool_call_id->empty()) {
                msg.set("tool_call_id", *m.tool_call_id);
            }
        }
        msgs.emplace_back(std::move(msg));
    }
    body.set("messages", std::move(msgs));
    const bool tool_choice_none = request.tool_choice && request.tool_choice->mode == ToolChoiceMode::none;
    if (!request.tools.empty() && !tool_choice_none) {
        const bool single = request.tool_choice && request.tool_choice->mode == ToolChoiceMode::function &&
                            !request.tool_choice->function_name.empty();
        json::Array tools;
        for (const auto& t : request.tools) {
            if (single && t.name != request.tool_choice->function_name) {
                continue;
            }
            json::Object fn{{"name", t.name}};
            if (!t.description.empty()) {
                fn.set("description", t.description);
            }
            json::Value params = json::Object{{"type", "object"}, {"properties", json::Object{}}};
            if (!t.parameters_json.empty()) {
                auto parsed = json::parse(t.parameters_json);
                if (parsed.ok() && parsed.value().is_object()) {
                    params = std::move(parsed).value();
                }
            }
            fn.set("parameters", std::move(params));
            tools.emplace_back(json::Object{{"type", "function"}, {"function", std::move(fn)}});
        }
        if (!tools.empty()) {
            body.set("tools", std::move(tools));
        }
    }
    const json::Object options = sampling_to_options(request.sampling);
    if (!options.empty()) {
        body.set("options", options);
    }
    return json::Value(std::move(body)).dump();
}

std::string build_generate_body(const std::string& model, const GenerateRequest& request) {
    json::Object body;
    body.set("model", model);
    body.set("stream", true);
    body.set("prompt", request.prompt);
    const json::Object options = sampling_to_options(request.sampling);
    if (!options.empty()) {
        body.set("options", options);
    }
    return json::Value(std::move(body)).dump();
}

// ---------------------------------------------------------------------------
// StreamDecoder

StreamDecoder::StreamDecoder(StreamKind kind, EventCallback on_event, std::size_t max_output_bytes,
                             TextScrubber scrub)
    : kind_(kind), on_event_(std::move(on_event)), scrub_(std::move(scrub)), max_output_bytes_(max_output_bytes) {}

bool StreamDecoder::feed(std::string_view bytes) {
    if (!error_.ok() || stopped_by_callback_) {
        return false;
    }
    while (!bytes.empty()) {
        const auto nl = bytes.find('\n');
        if (nl == std::string_view::npos) {
            if (buffer_.size() + bytes.size() > kMaxNdjsonLineBytes) {
                error_ = Status(ErrorCode::protocol_error, "ollama_cloud: NDJSON line exceeds 4 MiB");
                return false;
            }
            buffer_.append(bytes);
            return true;
        }
        bool keep_going = false;
        if (buffer_.empty()) {
            keep_going = on_line(bytes.substr(0, nl));
        } else {
            if (buffer_.size() + nl > kMaxNdjsonLineBytes) {
                error_ = Status(ErrorCode::protocol_error, "ollama_cloud: NDJSON line exceeds 4 MiB");
                return false;
            }
            buffer_.append(bytes.substr(0, nl));
            std::string line;
            line.swap(buffer_);
            keep_going = on_line(line);
        }
        if (!keep_going) {
            return false;
        }
        bytes.remove_prefix(nl + 1);
    }
    return true;
}

bool StreamDecoder::on_line(std::string_view line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
        line.remove_suffix(1);
    }
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
    }
    if (line.empty()) {
        return true;
    }
    auto parsed = json::parse(line);
    if (!parsed.ok() || !parsed.value().is_object()) {
        error_ = Status(ErrorCode::protocol_error, "ollama_cloud: malformed NDJSON line");
        return false;
    }
    const json::Value& j = parsed.value();
    if (const json::Value* e = j.find("error"); e != nullptr && !e->is_null()) {
        std::string text = e->is_string() ? e->as_string() : e->dump();
        if (scrub_) {
            scrub_(text);  // whole text first: a key across the cut must not leave a prefix
        }
        error_ = Status(ErrorCode::backend_error, "ollama_cloud: " + bounded(text, 512));
        return false;
    }
    StreamEvent ev;
    if (kind_ == StreamKind::generate) {
        ev.content = str_view(j, "response");
        ev.thinking = str_view(j, "thinking");
    } else if (const json::Value* m = j.find("message"); m != nullptr && m->is_object()) {
        ev.content = str_view(*m, "content");
        ev.thinking = str_view(*m, "thinking");
        if (const json::Value* tcs = m->find("tool_calls"); tcs != nullptr && tcs->is_array()) {
            for (const auto& item : tcs->as_array()) {
                const json::Value* fn = item.is_object() ? item.find("function") : nullptr;
                if (fn == nullptr || !fn->is_object()) {
                    continue;
                }
                std::string name = str_field(*fn, "name");
                if (name.empty()) {
                    continue;
                }
                ToolCallDelta delta;
                const json::Value* idx = fn->find("index");
                if (idx == nullptr || !idx->is_integer()) {
                    idx = item.find("index");
                }
                if (idx != nullptr && idx->is_integer() && idx->as_int(-1) >= 0 && idx->as_int(-1) < 64) {
                    delta.index = static_cast<std::uint32_t>(idx->as_int());
                    next_tool_call_index_ = std::max(next_tool_call_index_, delta.index + 1);
                } else {
                    delta.index = next_tool_call_index_++;
                }
                delta.id = str_field(item, "id");
                delta.name = std::move(name);
                if (const json::Value* args = fn->find("arguments"); args != nullptr && !args->is_null()) {
                    delta.arguments_delta = args->is_string() ? args->as_string() : args->dump();
                } else {
                    delta.arguments_delta = "{}";
                }
                ev.tool_calls.push_back(std::move(delta));
            }
        }
    }
    std::size_t bytes = ev.content.size() + ev.thinking.size();
    for (const auto& tc : ev.tool_calls) {
        bytes += tc.id.size() + tc.name.size() + tc.arguments_delta.size();
    }
    if (max_output_bytes_ > 0 && (bytes > max_output_bytes_ || output_bytes_ > max_output_bytes_ - bytes)) {
        error_ = Status(ErrorCode::protocol_error, "ollama_cloud: stream output exceeds the cumulative byte limit");
        return false;
    }
    output_bytes_ += bytes;
    const bool has_payload = !ev.content.empty() || !ev.thinking.empty() || !ev.tool_calls.empty();
    if (has_payload) {
        ++summary_.content_events;
    }
    if (!ev.tool_calls.empty()) {
        summary_.saw_tool_calls = true;
    }
    if (const json::Value* d = j.find("done"); d != nullptr && d->is_bool() && d->as_bool()) {
        summary_.done = true;
        summary_.done_reason = str_field(j, "done_reason");
        static constexpr std::string_view kKeys[] = {"total_duration", "load_duration", "prompt_eval_count",
                                                     "prompt_eval_duration", "eval_count", "eval_duration"};
        summary_.has_timings = std::any_of(std::begin(kKeys), std::end(kKeys),
                                           [&](std::string_view k) { return j.find(k) != nullptr; });
        summary_.total_duration_ns = int_field(j, "total_duration");
        summary_.load_duration_ns = int_field(j, "load_duration");
        summary_.prompt_eval_count = int_field(j, "prompt_eval_count");
        summary_.prompt_eval_duration_ns = int_field(j, "prompt_eval_duration");
        summary_.eval_count = int_field(j, "eval_count");
        summary_.eval_duration_ns = int_field(j, "eval_duration");
        if (j.find("prompt_eval_cached_count") != nullptr) {
            summary_.prompt_eval_cached_count = int_field(j, "prompt_eval_cached_count");
        }
    }
    if (has_payload && on_event_ && !on_event_(ev)) {
        stopped_by_callback_ = true;
        return false;
    }
    return true;
}

Status StreamDecoder::finish() {
    if (error_.ok() && !stopped_by_callback_ && !buffer_.empty()) {
        std::string line;
        line.swap(buffer_);
        on_line(line);
    }
    if (!error_.ok()) {
        return error_;
    }
    if (!summary_.done && !stopped_by_callback_) {
        return {ErrorCode::protocol_error, "ollama_cloud: stream ended without a done line"};
    }
    return Status::success();
}

// ---------------------------------------------------------------------------

Result<std::vector<TagEntry>> parse_tags(std::string_view body) {
    auto parsed = json::parse(body);
    if (!parsed.ok() || !parsed.value().is_object()) {
        return Status(ErrorCode::protocol_error, "ollama_cloud: /api/tags did not return a JSON object");
    }
    std::vector<TagEntry> out;
    const json::Value* models = parsed.value().find("models");
    if (models == nullptr || models->is_null()) {
        return out;
    }
    if (!models->is_array()) {
        return Status(ErrorCode::protocol_error, "ollama_cloud: /api/tags 'models' is not an array");
    }
    for (const auto& m : models->as_array()) {
        if (!m.is_object()) {
            continue;
        }
        TagEntry e;
        e.name = str_field(m, "name");
        if (e.name.empty()) {
            e.name = str_field(m, "model");
        }
        if (e.name.empty()) {
            continue;
        }
        e.size_bytes = static_cast<std::uint64_t>(std::max<std::int64_t>(0, int_field(m, "size")));
        if (const json::Value* d = m.find("details"); d != nullptr && d->is_object()) {
            e.format = str_field(*d, "format");
            e.family = str_field(*d, "family");
            e.parameter_size = str_field(*d, "parameter_size");
            e.quantization = str_field(*d, "quantization_level");
        }
        out.push_back(std::move(e));
    }
    return out;
}

Status http_status_error(int status, std::string_view body, const TextScrubber& scrub) {
    std::string detail;
    if (auto parsed = json::parse(body); parsed.ok()) {
        if (const json::Value* e = parsed.value().find("error"); e != nullptr) {
            detail = e->is_string() ? e->as_string() : e->dump();
        }
    } else {
        detail = std::string(body);
    }
    if (scrub) {
        scrub(detail);  // whole detail first: a key across the cut must not leave a prefix
    }
    detail = bounded(detail, 512);
    std::string text = "ollama_cloud HTTP " + std::to_string(status) + (detail.empty() ? "" : ": " + detail);
    switch (status) {
        case 400: return {ErrorCode::invalid_argument, std::move(text)};
        case 404: return {ErrorCode::not_found, std::move(text)};
        case 401:
        case 403:
            return {ErrorCode::unavailable, std::move(text) + " (check the configured API key)"};
        case 429:
            return {ErrorCode::unavailable, std::move(text) + " (rate limited)"};
        case 502:
        case 503:
        case 504: return {ErrorCode::unavailable, std::move(text)};
        default: return {ErrorCode::backend_error, std::move(text)};
    }
}

}  // namespace sonder::inference::ollama_cloud
