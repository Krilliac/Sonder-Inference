// Ollama compatibility backend ("Backend 0", docs/BACKENDS.md).
#include <chrono>

#include "net/http_client.hpp"
#include "ollama_protocol.hpp"
#include "sonder/inference/backends/ollama.hpp"

namespace sonder::inference {

namespace ollama {

json::Object build_generate_body(const std::string& model, const GenerateRequest& request,
                                 const std::string& keep_alive) {
    const SamplingConfig& s = request.sampling;
    json::Object options{{"temperature", s.temperature},
                         {"top_p", s.top_p},
                         {"top_k", s.top_k},
                         {"min_p", s.min_p},
                         {"repeat_penalty", s.repeat_penalty},
                         {"num_predict", s.max_tokens}};
    if (s.seed) {
        options.set("seed", static_cast<std::int64_t>(*s.seed & 0x7FFFFFFFFFFFFFFFull));
    }
    if (!s.stop.empty()) {
        json::Array stop;
        for (const auto& st : s.stop) {
            stop.emplace_back(st);
        }
        options.set("stop", std::move(stop));
    }
    json::Object body{{"model", model}, {"prompt", request.prompt}, {"stream", true}, {"options", std::move(options)}};
    if (!keep_alive.empty()) {
        body.set("keep_alive", keep_alive);
    }
    return body;
}

Result<StreamLine> parse_stream_line(std::string_view line, GenerateStats& stats) {
    StreamLine out;
    if (line.find_first_not_of(" \t\r") == std::string_view::npos) {
        return out;
    }
    auto parsed = json::parse(line);
    if (!parsed.ok()) {
        return parsed.status();
    }
    const json::Value& v = parsed.value();
    if (const json::Value* err = v.find("error")) {
        return Status(ErrorCode::backend_error, "ollama: " + err->as_string());
    }
    if (const json::Value* r = v.find("response")) {
        out.piece = r->as_string();
    }
    if (const json::Value* d = v.find("done")) {
        out.done = d->as_bool();
    }
    if (out.done) {
        auto u64 = [&](const char* key) -> std::uint64_t {
            const json::Value* f = v.find(key);
            const auto n = f ? f->as_int(0) : 0;
            return n > 0 ? static_cast<std::uint64_t>(n) : 0;
        };
        if (v.find("eval_count") || v.find("prompt_eval_count")) {
            stats.token_counts_from_backend = true;
        }
        stats.prompt_tokens = u64("prompt_eval_count");
        stats.completion_tokens = u64("eval_count");
        stats.load_ns = u64("load_duration");
        stats.prompt_eval_ns = u64("prompt_eval_duration");
        stats.eval_ns = u64("eval_duration");
        const json::Value* reason = v.find("done_reason");
        const std::string r = reason ? reason->as_string() : std::string();
        if (r == "length") {
            stats.stop_reason = StopReason::max_tokens;
        } else {
            stats.stop_reason = StopReason::end_of_sequence;
        }
    }
    return out;
}

ModelDescriptor descriptor_from_tag(const json::Value& entry) {
    ModelDescriptor d;
    d.name = entry.find("name") ? entry.find("name")->as_string() : std::string();
    if (d.name.empty() && entry.find("model")) {
        d.name = entry.find("model")->as_string();
    }
    d.backend = kOllamaBackendName;
    if (const json::Value* size = entry.find("size")) {
        const auto n = size->as_int(0);
        d.size_bytes = n > 0 ? static_cast<std::uint64_t>(n) : 0;
    }
    if (const json::Value* details = entry.find("details")) {
        auto str = [&](const char* key) {
            const json::Value* f = details->find(key);
            return f ? f->as_string() : std::string();
        };
        d.format = str("format");
        d.family = str("family");
        d.parameter_size = str("parameter_size");
        d.quantization = str("quantization_level");
    }
    return d;
}

}  // namespace ollama

namespace {

Status error_from_http(int http_status, const std::string& body) {
    std::string message = "HTTP " + std::to_string(http_status);
    if (auto parsed = json::parse(body); parsed.ok()) {
        if (const json::Value* e = parsed.value().find("error")) {
            message += ": " + e->as_string();
        }
    }
    const ErrorCode code = http_status == 404 ? ErrorCode::not_found : ErrorCode::backend_error;
    return Status(code, "ollama " + message);
}

class OllamaModel final : public BackendModel {
public:
    OllamaModel(ModelDescriptor d, net::Url url, OllamaBackendOptions options)
        : descriptor_(std::move(d)), url_(std::move(url)), options_(std::move(options)) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        net::HttpRequest http;
        http.method = "POST";
        http.host = url_.host;
        http.port = url_.port;
        http.target = (url_.path == "/" ? std::string() : url_.path) + "/api/generate";
        http.body = json::Value(ollama::build_generate_body(descriptor_.name, request, options_.keep_alive)).dump();
        http.connect_timeout = options_.connect_timeout;
        http.total_timeout = options_.request_timeout;

        GenerateStats stats;
        Status stream_error;
        bool stopped_by_callback = false;
        bool saw_done = false;
        std::string error_body;
        int http_status = 0;
        net::LineSplitter lines;

        auto on_line = [&](std::string_view line) -> bool {
            if (http_status != 200) {
                error_body.append(line);
                return true;
            }
            auto parsed = ollama::parse_stream_line(line, stats);
            if (!parsed.ok()) {
                stream_error = parsed.status();
                return false;
            }
            const auto& sl = parsed.value();
            if (!sl.piece.empty()) {
                const TokenChunk chunk{sl.piece, stats.chunks++};
                if (on_chunk && !on_chunk(chunk)) {
                    stopped_by_callback = true;
                    return false;
                }
            }
            if (sl.done) {
                saw_done = true;
                return false;
            }
            return true;
        };

        http.on_status = [&](int code) { http_status = code; };
        auto res = net::http_request(
            http, [&](std::string_view data) { return lines.feed(data, on_line); }, cancel);
        if (!res.ok()) {
            if (cancel.cancelled() || res.status().code() == ErrorCode::cancelled) {
                return Status(ErrorCode::cancelled, "cancelled by caller");
            }
            return res.status();
        }
        if (res.value().status != 200) {
            lines.finish(on_line);
            return error_from_http(res.value().status, error_body);
        }
        if (!stream_error.ok()) {
            return stream_error;
        }
        if (!saw_done && !stopped_by_callback) {
            lines.finish(on_line);
            if (!stream_error.ok()) {
                return stream_error;
            }
        }
        if (stopped_by_callback) {
            stats.stop_reason = StopReason::callback;
        } else if (!saw_done) {
            return Status(ErrorCode::protocol_error, "ollama stream ended without done=true");
        }
        if (!stats.token_counts_from_backend) {
            stats.completion_tokens = stats.chunks;
        }
        return stats;
    }

private:
    ModelDescriptor descriptor_;
    net::Url url_;
    OllamaBackendOptions options_;
};

class OllamaBackend final : public Backend {
public:
    explicit OllamaBackend(OllamaBackendOptions options) : options_(std::move(options)) {
        auto parsed = net::parse_url(options_.base_url);
        if (!parsed.ok()) {
            config_ = parsed.status();
            return;
        }
        url_ = parsed.value();
        if (!options_.allow_remote && !net::is_loopback_host(url_.host)) {
            config_ = Status(ErrorCode::invalid_argument,
                             "ollama backend refuses non-loopback host '" + url_.host +
                                 "' over plain HTTP (set allow_remote to override)");
        }
    }

    std::string name() const override { return kOllamaBackendName; }
    std::string description() const override { return "Ollama compatibility adapter at " + options_.base_url; }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming).add(Capability::remote_process);
        return c;
    }

    Result<std::string> probe() override {
        std::string body;
        auto res = get("/api/version", body);
        if (!res.ok()) {
            return res.status();
        }
        auto parsed = json::parse(body);
        if (!parsed.ok()) {
            return parsed.status();
        }
        const json::Value* v = parsed.value().find("version");
        return v ? v->as_string() : std::string("unknown");
    }

    Result<std::vector<ModelDescriptor>> list_models() override {
        std::string body;
        auto res = get("/api/tags", body);
        if (!res.ok()) {
            return res.status();
        }
        auto parsed = json::parse(body);
        if (!parsed.ok()) {
            return parsed.status();
        }
        std::vector<ModelDescriptor> out;
        if (const json::Value* models = parsed.value().find("models")) {
            for (const auto& entry : models->as_array()) {
                out.push_back(ollama::descriptor_from_tag(entry));
            }
        }
        return out;
    }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        auto models = list_models();
        if (!models.ok()) {
            return models.status();
        }
        for (const auto& d : models.value()) {
            if (d.name == options.model || d.name == options.model + ":latest") {
                return std::shared_ptr<BackendModel>(std::make_shared<OllamaModel>(d, url_, options_));
            }
        }
        return Status(ErrorCode::not_found, "ollama has no model named " + options.model);
    }

private:
    Result<int> get(const std::string& path, std::string& body) {
        if (!config_.ok()) {
            return config_;
        }
        net::HttpRequest http;
        http.method = "GET";
        http.host = url_.host;
        http.port = url_.port;
        http.target = (url_.path == "/" ? std::string() : url_.path) + path;
        http.connect_timeout = options_.connect_timeout;
        http.total_timeout = std::chrono::milliseconds(15000);
        auto res = net::http_request_buffered(http, body);
        if (!res.ok()) {
            return res.status();
        }
        if (res.value().status != 200) {
            return error_from_http(res.value().status, body);
        }
        return res.value().status;
    }

    OllamaBackendOptions options_;
    net::Url url_;
    Status config_;
};

}  // namespace

std::shared_ptr<Backend> make_ollama_backend(OllamaBackendOptions options) {
    return std::make_shared<OllamaBackend>(std::move(options));
}

}  // namespace sonder::inference
