// Ollama HTTP client: NDJSON stream decoding, request building, model APIs.
#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

#include "net/http_client.hpp"
#include "sonder/inference/backends/ollama.hpp"

namespace sonder::inference::ollama {

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

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
    return f->is_integer() ? f->as_int() : static_cast<std::int64_t>(f->as_double());
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
        ++i;
    }
    return s.substr(i);
}

// Extracts {"error": "..."} or returns the (truncated) raw body.
std::string error_message_from_body(const std::string& body) {
    auto parsed = json::parse(body);
    if (parsed.ok()) {
        const json::Value* e = parsed.value().find("error");
        if (e != nullptr) {
            return e->is_string() ? e->as_string() : e->dump();
        }
    }
    std::string t = trim(body);
    constexpr std::size_t kMax = 512;
    if (t.size() > kMax) {
        t = t.substr(0, kMax) + "...";
    }
    return t;
}

Status http_status_error(int status, const std::string& body) {
    std::string msg = error_message_from_body(body);
    std::string text = "ollama HTTP " + std::to_string(status) + (msg.empty() ? "" : ": " + msg);
    if (status == 404) {
        return {ErrorCode::not_found, std::move(text)};
    }
    if (status == 400) {
        return {ErrorCode::invalid_argument, std::move(text)};
    }
    if (status == 503) {
        return {ErrorCode::unavailable, std::move(text)};
    }
    return {ErrorCode::backend_error, std::move(text)};
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

Result<net::HttpRequest> make_request(const OllamaConfig& config, std::string method, std::string_view api_path,
                                      std::string body) {
    auto url = net::parse_url(config.base_url);
    if (!url.ok()) {
        return url.status();
    }
    const net::Url& u = url.value();
    if (u.scheme != "http") {
        return Status(ErrorCode::unsupported,
                      "ollama: only http:// endpoints are supported by the internal client (got '" + u.scheme +
                          "'); front TLS workers with a local proxy");
    }
    if (!config.allow_remote && !net::is_loopback_host(u.host)) {
        return Status(ErrorCode::invalid_argument,
                      "ollama: refusing plain HTTP to non-loopback host '" + u.host +
                          "' (set allow_remote to override)");
    }
    net::HttpRequest req;
    req.method = std::move(method);
    req.host = u.host;
    req.port = u.port;
    req.target = join_target(u.path, api_path);
    req.body = std::move(body);
    req.content_type = "application/json";
    req.connect_timeout = config.connect_timeout;
    req.total_timeout = config.request_timeout;
    return req;
}

void apply_common(json::Object& body, const std::string& model, bool stream, const std::optional<bool>& think,
                  const std::string& keep_alive, const OllamaConfig& config, const json::Object& options) {
    body.set("model", model);
    body.set("stream", stream);
    if (think) {
        body.set("think", *think);
    }
    const std::string& ka = keep_alive.empty() ? config.keep_alive : keep_alive;
    if (!ka.empty()) {
        body.set("keep_alive", ka);
    }
    if (!options.empty()) {
        body.set("options", options);
    }
}

}  // namespace

double OllamaTimings::prompt_tokens_per_sec() const noexcept {
    if (prompt_eval_count <= 0 || prompt_eval_duration_ns <= 0) {
        return 0.0;
    }
    return static_cast<double>(prompt_eval_count) * 1e9 / static_cast<double>(prompt_eval_duration_ns);
}

double OllamaTimings::decode_tokens_per_sec() const noexcept {
    if (eval_count <= 0 || eval_duration_ns <= 0) {
        return 0.0;
    }
    return static_cast<double>(eval_count) * 1e9 / static_cast<double>(eval_duration_ns);
}

void apply_server_timings(const json::Value& j, OllamaTimings& t) {
    static constexpr std::string_view kKeys[] = {"total_duration",       "load_duration", "prompt_eval_count",
                                                 "prompt_eval_duration", "eval_count",    "eval_duration"};
    const bool any = std::any_of(std::begin(kKeys), std::end(kKeys), [&](std::string_view k) { return j.find(k) != nullptr; });
    if (!any) {
        return;
    }
    t.has_server_timings = true;
    t.total_duration_ns = int_field(j, "total_duration");
    t.load_duration_ns = int_field(j, "load_duration");
    t.prompt_eval_count = int_field(j, "prompt_eval_count");
    t.prompt_eval_duration_ns = int_field(j, "prompt_eval_duration");
    t.eval_count = int_field(j, "eval_count");
    t.eval_duration_ns = int_field(j, "eval_duration");
}

json::Object sampling_to_options(const SamplingConfig& s) {
    json::Object o;
    o.set("temperature", static_cast<double>(s.temperature));
    o.set("top_p", static_cast<double>(s.top_p));
    o.set("top_k", s.top_k);
    o.set("min_p", static_cast<double>(s.min_p));
    o.set("repeat_penalty", static_cast<double>(s.repeat_penalty));
    // Newer SamplingConfig fields are only sent when they differ from their
    // disabled/default value, so a default config produces the same options
    // object as before. logit_bias has no Ollama option and is not forwarded.
    if (s.num_ctx > 0) {
        o.set("num_ctx", s.num_ctx);
    }
    if (s.repeat_last_n != SamplingConfig{}.repeat_last_n) {
        o.set("repeat_last_n", s.repeat_last_n);
    }
    if (s.presence_penalty != 0.0f) {
        o.set("presence_penalty", static_cast<double>(s.presence_penalty));
    }
    if (s.frequency_penalty != 0.0f) {
        o.set("frequency_penalty", static_cast<double>(s.frequency_penalty));
    }
    if (s.typical_p != 1.0f) {
        o.set("typical_p", static_cast<double>(s.typical_p));
    }
    if (s.seed) {
        // Ollama takes a signed 64-bit seed.
        o.set("seed", static_cast<std::int64_t>(*s.seed & 0x7FFFFFFFFFFFFFFFull));
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

json::Value build_generate_body(const GenerateParams& p, const OllamaConfig& config) {
    json::Object body;
    apply_common(body, p.model, p.stream, p.think, p.keep_alive, config, p.options);
    body.set("prompt", p.prompt);
    if (!p.system.empty()) {
        body.set("system", p.system);
    }
    if (p.raw) {
        body.set("raw", true);
    }
    return body;
}

json::Value build_chat_body(const ChatParams& p, const OllamaConfig& config) {
    json::Object body;
    apply_common(body, p.model, p.stream, p.think, p.keep_alive, config, p.options);
    json::Array msgs;
    for (const auto& m : p.messages) {
        msgs.emplace_back(json::Object{{"role", m.role}, {"content", m.content}});
    }
    body.set("messages", std::move(msgs));
    return body;
}

// ---------------------------------------------------------------------------
// StreamDecoder

StreamDecoder::StreamDecoder(StreamKind kind, ChunkCallback on_chunk, Clock::time_point start)
    : kind_(kind), on_chunk_(std::move(on_chunk)), start_(start) {}

bool StreamDecoder::feed(std::string_view bytes) {
    if (!error_.ok()) {
        return false;
    }
    constexpr std::size_t kMaxLine = 16u << 20;
    while (!bytes.empty()) {
        const auto nl = bytes.find('\n');
        if (nl == std::string_view::npos) {
            if (buffer_.size() + bytes.size() > kMaxLine) {
                error_ = Status(ErrorCode::protocol_error, "ollama: NDJSON line exceeds 16 MiB");
                return false;
            }
            buffer_.append(bytes);
            return true;
        }
        bool keep_going;
        if (buffer_.empty()) {
            keep_going = on_line(bytes.substr(0, nl));
        } else {
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
        std::string snippet(line.substr(0, 200));
        error_ = Status(ErrorCode::protocol_error, "ollama: malformed NDJSON line: " + snippet);
        return false;
    }
    const json::Value& j = parsed.value();
    if (const json::Value* e = j.find("error"); e != nullptr && !e->is_null()) {
        error_ = Status(ErrorCode::backend_error, "ollama: " + (e->is_string() ? e->as_string() : e->dump()));
        return false;
    }
    StreamChunk c;
    c.raw = &j;
    if (kind_ == StreamKind::generate) {
        c.content = str_view(j, "response");
        c.thinking = str_view(j, "thinking");
    } else if (const json::Value* m = j.find("message"); m != nullptr && m->is_object()) {
        c.content = str_view(*m, "content");
        c.thinking = str_view(*m, "thinking");
    }
    if (const json::Value* d = j.find("done"); d != nullptr && d->is_bool()) {
        c.done = d->as_bool();
    }
    c.done_reason = str_view(j, "done_reason");
    if (!c.content.empty() || !c.thinking.empty()) {
        if (result_.timings.ttft_ms < 0) {
            result_.timings.ttft_ms = ms_between(start_, Clock::now());
        }
        ++result_.timings.content_chunks;
    }
    result_.text.append(c.content);
    result_.thinking.append(c.thinking);
    if (result_.model.empty()) {
        result_.model = str_field(j, "model");
    }
    if (c.done) {
        result_.done = true;
        result_.done_reason = std::string(c.done_reason);
        apply_server_timings(j, result_.timings);
    }
    if (on_chunk_ && !on_chunk_(c)) {
        result_.stopped_by_callback = true;
        return false;
    }
    return true;
}

Status StreamDecoder::finish() {
    if (error_.ok() && !buffer_.empty()) {
        std::string line;
        line.swap(buffer_);
        on_line(line);
    }
    if (!error_.ok()) {
        return error_;
    }
    if (!result_.done && !result_.stopped_by_callback) {
        return {ErrorCode::protocol_error, "ollama: stream ended without a done chunk"};
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// OllamaClient

OllamaClient::OllamaClient(OllamaConfig config) : config_(std::move(config)) {}

Result<json::Value> OllamaClient::get_json(const std::string& method, const std::string& path,
                                           const std::string& body) const {
    auto req = make_request(config_, method, path, body);
    if (!req.ok()) {
        return req.status();
    }
    req.value().total_timeout = std::min(config_.request_timeout, std::chrono::milliseconds(60000));
    std::string out;
    auto res = net::http_request_buffered(req.value(), out, {}, 64u * 1024u * 1024u);
    if (!res.ok()) {
        return res.status();
    }
    const int status = res.value().status;
    if (status < 200 || status >= 300) {
        return http_status_error(status, out);
    }
    auto parsed = json::parse(out);
    if (!parsed.ok() || !parsed.value().is_object()) {
        return Status(ErrorCode::protocol_error, "ollama: " + path + " did not return a JSON object");
    }
    return std::move(parsed).value();
}

Result<std::string> OllamaClient::version() const {
    auto j = get_json("GET", "/api/version", "");
    if (!j.ok()) {
        return j.status();
    }
    return str_field(j.value(), "version");
}

Result<std::vector<OllamaModelInfo>> OllamaClient::list_models() const {
    auto j = get_json("GET", "/api/tags", "");
    if (!j.ok()) {
        return j.status();
    }
    std::vector<OllamaModelInfo> out;
    const json::Value* models = j.value().find("models");
    if (models == nullptr) {
        return out;
    }
    for (const auto& m : models->as_array()) {
        if (!m.is_object()) {
            continue;
        }
        OllamaModelInfo info;
        info.name = str_field(m, "name");
        if (info.name.empty()) {
            info.name = str_field(m, "model");
        }
        info.digest = str_field(m, "digest");
        info.size_bytes = static_cast<std::uint64_t>(std::max<std::int64_t>(0, int_field(m, "size")));
        info.modified_at = str_field(m, "modified_at");
        if (const json::Value* d = m.find("details"); d != nullptr) {
            info.format = str_field(*d, "format");
            info.family = str_field(*d, "family");
            info.parameter_size = str_field(*d, "parameter_size");
            info.quantization_level = str_field(*d, "quantization_level");
        }
        out.push_back(std::move(info));
    }
    return out;
}

Result<std::vector<RunningModel>> OllamaClient::running_models() const {
    auto j = get_json("GET", "/api/ps", "");
    if (!j.ok()) {
        return j.status();
    }
    std::vector<RunningModel> out;
    if (const json::Value* models = j.value().find("models"); models != nullptr) {
        for (const auto& m : models->as_array()) {
            RunningModel r;
            r.name = str_field(m, "name");
            r.size_bytes = static_cast<std::uint64_t>(std::max<std::int64_t>(0, int_field(m, "size")));
            r.size_vram_bytes = static_cast<std::uint64_t>(std::max<std::int64_t>(0, int_field(m, "size_vram")));
            r.expires_at = str_field(m, "expires_at");
            out.push_back(std::move(r));
        }
    }
    return out;
}

Result<json::Value> OllamaClient::show_model(const std::string& name) const {
    const json::Value body = json::Object{{"model", name}};
    return get_json("POST", "/api/show", body.dump());
}

Result<StreamResult> OllamaClient::generate(const GenerateParams& params, const ChunkCallback& on_chunk,
                                            const CancellationToken& cancel) const {
    return stream(StreamKind::generate, "/api/generate", build_generate_body(params, config_).dump(), on_chunk, cancel);
}

Result<StreamResult> OllamaClient::chat(const ChatParams& params, const ChunkCallback& on_chunk,
                                        const CancellationToken& cancel) const {
    return stream(StreamKind::chat, "/api/chat", build_chat_body(params, config_).dump(), on_chunk, cancel);
}

Result<StreamResult> OllamaClient::stream(StreamKind kind, const std::string& path, const std::string& body,
                                          const ChunkCallback& on_chunk, const CancellationToken& cancel) const {
    if (cancel.cancelled()) {
        return Status(ErrorCode::cancelled, "ollama: cancelled before start");
    }
    auto req = make_request(config_, "POST", path, body);
    if (!req.ok()) {
        return req.status();
    }
    int http_status = 0;
    req.value().on_status = [&http_status](int s) { http_status = s; };

    const auto start = Clock::now();
    StreamDecoder decoder(kind, on_chunk, start);
    std::string error_body;
    auto res = net::http_request(
        req.value(),
        [&](std::string_view data) {
            auto& t = decoder.result().timings;
            if (t.ttfb_ms < 0) {
                t.ttfb_ms = ms_between(start, Clock::now());
            }
            if (http_status < 200 || http_status >= 300) {
                if (error_body.size() < 8192) {
                    error_body.append(data.substr(0, 8192 - error_body.size()));
                }
                return true;
            }
            return decoder.feed(data);
        },
        cancel);
    decoder.result().timings.wall_ms = ms_between(start, Clock::now());
    decoder.result().http_status = http_status;

    if (cancel.cancelled()) {
        return Status(ErrorCode::cancelled, "ollama: request cancelled");
    }
    if (!res.ok()) {
        return res.status();
    }
    if (http_status < 200 || http_status >= 300) {
        return http_status_error(http_status, error_body);
    }
    if (Status st = decoder.finish(); !st.ok()) {
        return st;
    }
    return std::move(decoder.result());
}

}  // namespace sonder::inference::ollama
