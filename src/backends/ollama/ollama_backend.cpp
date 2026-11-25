// Backend adapter: exposes OllamaClient through the core Backend interface.
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "net/http_client.hpp"
#include "sonder/inference/backends/ollama.hpp"

namespace sonder::inference {

using ollama::OllamaClient;
using ollama::OllamaModelInfo;
using ollama::StreamChunk;
using ollama::StreamResult;

namespace {

std::uint64_t to_u64(std::int64_t v) { return v > 0 ? static_cast<std::uint64_t>(v) : 0u; }

StopReason map_done_reason(const std::string& reason) {
    if (reason == "length") {
        return StopReason::max_tokens;
    }
    if (reason == "stop") {
        return StopReason::end_of_sequence;
    }
    return StopReason::none;
}

std::uint64_t context_length_from_show(const json::Value& show) {
    const json::Value* info = show.find("model_info");
    if (info == nullptr || !info->is_object()) {
        return 0;
    }
    std::string arch;
    if (const json::Value* a = info->find("general.architecture"); a != nullptr && a->is_string()) {
        arch = a->as_string();
    }
    if (!arch.empty()) {
        if (const json::Value* c = info->find(arch + ".context_length"); c != nullptr && c->is_number()) {
            return to_u64(c->as_int());
        }
    }
    return 0;
}

// Classifies attention-only / hybrid / recurrent from /api/show model_info
// (e.g. Qwen3.8 reports qwen35.ssm.* next to qwen35.attention.*), so the
// cache limits prefix reuse to checkpoints for hybrid and recurrent models.
ModelArchitecture architecture_from_show(const json::Value& show) {
    ModelMetadata metadata;
    const json::Value* info = show.find("model_info");
    if (info == nullptr || !info->is_object()) {
        return ModelArchitecture::attention_only;
    }
    for (const auto& [key, value] : info->as_object()) {
        if (value.is_string()) {
            metadata.emplace_back(key, value.as_string());
        } else if (value.is_number() || value.is_bool()) {
            metadata.emplace_back(key, value.dump());
        }
    }
    return classify_model_architecture(metadata);
}

class OllamaModel final : public BackendModel {
public:
    OllamaModel(std::shared_ptr<const OllamaClient> client, ModelDescriptor descriptor)
        : client_(std::move(client)), descriptor_(std::move(descriptor)) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        // Ollama's API has no logit bias option. Refuse instead of silently
        // sampling without it.
        if (!request.sampling.logit_bias.empty()) {
            return Status(ErrorCode::invalid_argument, "the ollama backend does not support logit_bias");
        }
        // Ollama 0.34.1 removed typical_p and rejects requests that set it.
        if (request.sampling.typical_p != 1.0f) {
            return Status(ErrorCode::invalid_argument, "the ollama backend does not support typical_p");
        }
        if (!request.grammar.empty()) {
            return Status(ErrorCode::unsupported, "the ollama backend does not support raw GBNF grammar constraints");
        }
        ollama::GenerateParams params;
        params.model = descriptor_.name;
        params.prompt = request.prompt;
        params.retain_text = false;
        params.options = ollama::sampling_to_options(request.sampling);
        return run(cancel, on_chunk, [&](const ollama::ChunkCallback& cb) { return client_->generate(params, cb, cancel); });
    }

    // Native chat through POST /api/chat: the server applies the model's own
    // chat template, so no client-side prompt formatting is involved.
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        if (Status st = validate_chat_messages(request.messages); !st.ok()) {
            return st;
        }
        if (!request.sampling.logit_bias.empty()) {
            return Status(ErrorCode::invalid_argument, "the ollama backend does not support logit_bias");
        }
        // Ollama 0.34.1 removed typical_p and rejects requests that set it.
        if (request.sampling.typical_p != 1.0f) {
            return Status(ErrorCode::invalid_argument, "the ollama backend does not support typical_p");
        }
        ollama::ChatParams params;
        if (request.response_format_type) {
            if (*request.response_format_type == "json_object") {
                params.format = json::Value("json");
            } else if (*request.response_format_type == "json_schema") {
                if (!request.response_format_schema || request.response_format_schema->empty()) {
                    return Status(ErrorCode::invalid_argument,
                                  "ollama: json_schema response_format is missing its schema");
                }
                auto parsed = json::parse(*request.response_format_schema);
                if (!parsed.ok() || !parsed.value().is_object()) {
                    return Status(ErrorCode::invalid_argument,
                                  "ollama: json_schema response_format schema must be a JSON object");
                }
                params.format = std::move(parsed).value();
            } else {
                return Status(ErrorCode::unsupported,
                              "the ollama backend does not support response_format type '" +
                                  *request.response_format_type + "'");
            }
        } else if (!request.grammar.empty()) {
            return Status(ErrorCode::unsupported, "the ollama backend does not support raw GBNF grammar constraints");
        }
        params.model = descriptor_.name;
        params.messages = request.messages;
        params.tools = request.tools;
        params.tool_choice = request.tool_choice;
        params.retain_text = false;
        params.options = ollama::sampling_to_options(request.sampling);
        // Ollama's own switch; reasoning_effort has no portable Ollama field.
        params.think = request.thinking.enable_thinking;
        return run(cancel, on_chunk, [&](const ollama::ChunkCallback& cb) { return client_->chat(params, cb, cancel); },
                   request.separate_reasoning);
    }

    bool has_native_chat() const override { return true; }

    Result<EmbedResult> embed(const EmbedRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "ollama: request cancelled");
        }
        if (request.inputs.empty()) {
            return Status(ErrorCode::invalid_argument, "ollama: embed inputs must be non-empty");
        }
        const auto& cfg = client_->config();
        auto url = net::parse_url(cfg.base_url);
        if (!url.ok()) {
            return url.status();
        }
        const net::Url& u = url.value();
        const bool https = u.scheme == "https";
        if (u.scheme != "http" && !https) {
            return Status(ErrorCode::unsupported,
                          "ollama: only http:// endpoints are supported by the internal client (got '" + u.scheme +
                              "'); front TLS workers with a local proxy");
        }
        if (!cfg.allow_remote && !net::is_loopback_host(u.host)) {
            return Status(ErrorCode::invalid_argument,
                          std::string(https ? "ollama: refusing non-loopback host '"
                                            : "ollama: refusing plain HTTP to non-loopback host '") +
                              u.host + "' (set allow_remote to override)");
        }
        std::string target = "/api/embed";
        if (!u.path.empty() && u.path != "/") {
            std::string base = u.path;
            if (base.back() == '/') base.pop_back();
            target = base + target;
        }
        json::Array input_arr;
        input_arr.reserve(request.inputs.size());
        for (const auto& text : request.inputs) {
            input_arr.emplace_back(text);
        }
        json::Object body_obj{{"model", descriptor_.name}, {"input", std::move(input_arr)}};
        if (!cfg.keep_alive.empty()) {
            body_obj.set("keep_alive", cfg.keep_alive);
        }
        net::HttpRequest http_req;
        http_req.method = "POST";
        http_req.host = u.host;
        http_req.port = u.port;
        http_req.target = std::move(target);
        http_req.body = json::Value(std::move(body_obj)).dump();
        http_req.content_type = "application/json";
        http_req.connect_timeout = cfg.connect_timeout;
        http_req.total_timeout = cfg.request_timeout;
        if (https) {
            http_req.use_tls = true;
            http_req.tls.ca_bundle_path = cfg.tls.ca_bundle_path;
            http_req.tls.pinned_sha256 = cfg.tls.pinned_sha256;
            http_req.tls.pinned_cert_path = cfg.tls.pinned_cert_path;
            http_req.tls.insecure_skip_verify = cfg.tls.insecure_skip_verify;
            http_req.tls.server_name = cfg.tls.server_name;
            http_req.tls.handshake_timeout = cfg.tls.handshake_timeout;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string out;
        auto res = net::http_request_buffered(http_req, out, cancel, 64u * 1024u * 1024u);
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "ollama: request cancelled");
        }
        if (!res.ok()) {
            return res.status();
        }
        const int status = res.value().status;
        if (status < 200 || status >= 300) {
            std::string detail;
            if (auto err_parsed = json::parse(out); err_parsed.ok()) {
                if (const json::Value* e = err_parsed.value().find("error"); e != nullptr) {
                    detail = e->is_string() ? e->as_string() : e->dump();
                }
            }
            const std::string text = "ollama HTTP " + std::to_string(status) + (detail.empty() ? "" : ": " + detail);
            if (status == 501 || (status == 400 && detail.find("does not support") != std::string::npos)) {
                return Status(ErrorCode::unsupported, text);
            }
            if (status == 404) return Status(ErrorCode::not_found, text);
            if (status == 400) return Status(ErrorCode::invalid_argument, text);
            if (status == 503) return Status(ErrorCode::unavailable, text);
            return Status(ErrorCode::backend_error, text);
        }
        auto parsed = json::parse(out);
        if (!parsed.ok() || !parsed.value().is_object()) {
            return Status(ErrorCode::protocol_error, "ollama: /api/embed did not return a JSON object");
        }
        const json::Value* embs = parsed.value().find("embeddings");
        if (embs == nullptr || !embs->is_array()) {
            return Status(ErrorCode::protocol_error, "ollama: /api/embed response missing embeddings array");
        }
        const auto& rows = embs->as_array();
        if (rows.size() != request.inputs.size()) {
            return Status(ErrorCode::protocol_error, "ollama: /api/embed returned unexpected embedding count");
        }
        EmbedResult result;
        result.embeddings.reserve(rows.size());
        for (const auto& row : rows) {
            if (!row.is_array()) {
                return Status(ErrorCode::protocol_error, "ollama: /api/embed embedding entry must be an array");
            }
            std::vector<float> vec;
            vec.reserve(row.as_array().size());
            for (const auto& val : row.as_array()) {
                if (!val.is_number() || !std::isfinite(val.as_double())) {
                    return Status(ErrorCode::protocol_error, "ollama: /api/embed returned non-finite value");
                }
                vec.push_back(static_cast<float>(val.as_double()));
            }
            result.embeddings.push_back(std::move(vec));
        }
        if (const json::Value* pe = parsed.value().find("prompt_eval_count"); pe != nullptr && pe->is_number()) {
            result.prompt_tokens = to_u64(pe->as_int());
        }
        if (const json::Value* td = parsed.value().find("total_duration"); td != nullptr && td->is_number()) {
            result.eval_ns = to_u64(td->as_int());
        } else {
            result.eval_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        }
        return result;
    }

private:
    // Shared streaming path for /api/generate and /api/chat: maps Ollama
    // stream chunks onto TokenCallback and the final chunk onto GenerateStats.
    template <class Call>
    Result<GenerateStats> run(const CancellationToken& cancel, const TokenCallback& on_chunk, Call&& call,
                              bool separate_reasoning = false) {
        const bool emit_thinking = client_->config().emit_thinking_chunks;
        std::uint64_t index = 0;
        bool stopped_by_callback = false;
        bool saw_tool_calls = false;
        const ollama::ChunkCallback cb = [&](const StreamChunk& c) {
            if (cancel.cancelled()) {
                return false;
            }
            if (!c.tool_calls.empty()) {
                saw_tool_calls = true;
            }
            if (!on_chunk) {
                return true;
            }
            if (!c.tool_calls.empty() && !stopped_by_callback) {
                TokenChunk tc;
                tc.index = index;
                tc.tool_call_deltas = c.tool_calls;
                if (!on_chunk(tc)) {
                    stopped_by_callback = true;
                }
                ++index;
            }
            auto deliver = [&](std::string_view text, std::string_view reasoning = {}) {
                if ((text.empty() && reasoning.empty()) || stopped_by_callback) {
                    return;
                }
                if (!on_chunk(TokenChunk{text, index, reasoning})) {
                    stopped_by_callback = true;
                }
                ++index;
            };
            if (separate_reasoning) {
                // Both server protocols (Anthropic thinking blocks, OpenAI
                // reasoning_content) need thinking kept out of the visible
                // text stream. The old emit_thinking_chunks behavior
                // remains unchanged when this opt-in is false.
                deliver(c.content, c.thinking);
            } else {
                if (emit_thinking) {
                    deliver(c.thinking);
                }
                deliver(c.content);
            }
            return !stopped_by_callback;
        };
        Result<StreamResult> res = call(cb);
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "ollama: request cancelled");
        }
        if (!res.ok()) {
            return res.status();
        }
        const StreamResult& r = res.value();
        GenerateStats stats;
        stats.chunks = index;
        if (r.timings.has_server_timings) {
            stats.prompt_tokens = to_u64(r.timings.prompt_eval_count);
            stats.completion_tokens = to_u64(r.timings.eval_count);
            stats.load_ns = to_u64(r.timings.load_duration_ns);
            stats.prompt_eval_ns = to_u64(r.timings.prompt_eval_duration_ns);
            stats.eval_ns = to_u64(r.timings.eval_duration_ns);
            stats.token_counts_from_backend = true;
            if (r.timings.has_prompt_eval_cached_count) {
                stats.cached_tokens = to_u64(r.timings.prompt_eval_cached_count);
            }
        } else {
            stats.completion_tokens = static_cast<std::uint64_t>(r.timings.content_chunks);
        }
        if (stopped_by_callback || r.stopped_by_callback) {
            stats.stop_reason = StopReason::callback;
        } else if (saw_tool_calls || r.saw_tool_calls) {
            stats.stop_reason = StopReason::tool_calls;
        } else {
            stats.stop_reason = map_done_reason(r.done_reason);
        }
        return stats;
    }

    std::shared_ptr<const OllamaClient> client_;
    ModelDescriptor descriptor_;
};

class OllamaBackend final : public Backend {
public:
    explicit OllamaBackend(OllamaBackendOptions config) : client_(std::make_shared<OllamaClient>(std::move(config))) {}

    std::string name() const override { return kOllamaBackendName; }
    std::string description() const override {
        return "Ollama compatibility adapter at " + client_->config().base_url;
    }
    BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        caps.add(Capability::streaming)
            .add(Capability::remote_process)
            .add(Capability::embeddings)
            .add(Capability::structured_output)
            .add(Capability::tool_calling);
        return caps;
    }

    Result<std::string> probe() override { return client_->version(); }

    Result<std::vector<ModelDescriptor>> list_models() override {
        auto models = client_->list_models();
        if (!models.ok()) {
            return models.status();
        }
        std::vector<ModelDescriptor> out;
        out.reserve(models.value().size());
        for (const auto& m : models.value()) {
            out.push_back(to_descriptor(m));
        }
        return out;
    }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        if (options.model.empty()) {
            return Status(ErrorCode::invalid_argument, "ollama: model name is required");
        }
        // Ollama loads weights lazily on first request; here we only verify
        // the model exists and capture its metadata. No pulls are performed.
        auto show = client_->show_model(options.model);
        if (!show.ok()) {
            return show.status();
        }
        ModelDescriptor d;
        d.name = options.model;
        d.backend = kOllamaBackendName;
        d.resident = false;  // metadata only (/api/show); Ollama loads weights lazily
        if (const json::Value* det = show.value().find("details"); det != nullptr) {
            auto s = [&](std::string_view k) {
                const json::Value* v = det->find(k);
                return (v != nullptr && v->is_string()) ? v->as_string() : std::string();
            };
            d.format = s("format");
            d.family = s("family");
            d.parameter_size = s("parameter_size");
            d.quantization = s("quantization_level");
        }
        d.context_length = context_length_from_show(show.value());
        d.architecture = architecture_from_show(show.value());
        if (auto models = client_->list_models(); models.ok()) {
            for (const auto& m : models.value()) {
                if (m.name == options.model || m.name == options.model + ":latest") {
                    d.size_bytes = m.size_bytes;
                }
            }
        }
        return std::shared_ptr<BackendModel>(std::make_shared<OllamaModel>(client_, std::move(d)));
    }

private:
    static ModelDescriptor to_descriptor(const OllamaModelInfo& m) {
        ModelDescriptor d;
        d.name = m.name;
        d.backend = kOllamaBackendName;
        d.format = m.format;
        d.family = m.family;
        d.parameter_size = m.parameter_size;
        d.quantization = m.quantization_level;
        d.size_bytes = m.size_bytes;
        return d;
    }

    std::shared_ptr<OllamaClient> client_;
};

}  // namespace

std::shared_ptr<Backend> make_ollama_backend(OllamaBackendOptions options) {
    return std::make_shared<OllamaBackend>(std::move(options));
}

}  // namespace sonder::inference
