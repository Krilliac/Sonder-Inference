// Backend adapter: exposes OllamaClient through the core Backend interface.
#include <memory>
#include <string>
#include <utility>

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
        ollama::GenerateParams params;
        params.model = descriptor_.name;
        params.prompt = request.prompt;
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
        params.model = descriptor_.name;
        params.messages = request.messages;
        params.options = ollama::sampling_to_options(request.sampling);
        return run(cancel, on_chunk, [&](const ollama::ChunkCallback& cb) { return client_->chat(params, cb, cancel); });
    }

    bool has_native_chat() const override { return true; }

private:
    // Shared streaming path for /api/generate and /api/chat: maps Ollama
    // stream chunks onto TokenCallback and the final chunk onto GenerateStats.
    template <class Call>
    Result<GenerateStats> run(const CancellationToken& cancel, const TokenCallback& on_chunk, Call&& call) {
        const bool emit_thinking = client_->config().emit_thinking_chunks;
        std::uint64_t index = 0;
        bool stopped_by_callback = false;
        const ollama::ChunkCallback cb = [&](const StreamChunk& c) {
            if (cancel.cancelled()) {
                return false;
            }
            if (!on_chunk) {
                return true;
            }
            auto deliver = [&](std::string_view text) {
                if (text.empty() || stopped_by_callback) {
                    return;
                }
                if (!on_chunk(TokenChunk{text, index})) {
                    stopped_by_callback = true;
                }
                ++index;
            };
            if (emit_thinking) {
                deliver(c.thinking);
            }
            deliver(c.content);
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
        } else {
            stats.completion_tokens = static_cast<std::uint64_t>(r.timings.content_chunks);
        }
        stats.stop_reason = (stopped_by_callback || r.stopped_by_callback) ? StopReason::callback
                                                                           : map_done_reason(r.done_reason);
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
        caps.add(Capability::streaming).add(Capability::remote_process);
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
