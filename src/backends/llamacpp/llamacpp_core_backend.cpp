// Adapter: sonder::inference::Backend over the llama.cpp runtime wrapper.
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "llamacpp_internal.hpp"

namespace sonder::inference {
namespace {

namespace lc = sonder::backends::llamacpp;
namespace fs = std::filesystem;
using llamacpp_detail::PathFromUtf8;
using llamacpp_detail::PathToUtf8;
using llamacpp_detail::StopSequenceFilter;
using llamacpp_detail::ToLlamaSampling;

ErrorCode MapError(lc::ErrorCode code) {
    switch (code) {
        case lc::ErrorCode::kOk: return ErrorCode::ok;
        case lc::ErrorCode::kInvalidArgument: return ErrorCode::invalid_argument;
        case lc::ErrorCode::kFileNotFound: return ErrorCode::not_found;
        case lc::ErrorCode::kLoadFailed: return ErrorCode::backend_error;
        case lc::ErrorCode::kNotLoaded: return ErrorCode::invalid_state;
        case lc::ErrorCode::kTokenizeFailed: return ErrorCode::backend_error;
        case lc::ErrorCode::kContextOverflow: return ErrorCode::invalid_argument;
        case lc::ErrorCode::kDecodeFailed: return ErrorCode::backend_error;
    }
    return ErrorCode::internal;
}

Status ToStatus(const lc::Status& s) { return s.ok() ? Status::success() : Status(MapError(s.code), s.message); }

std::uint64_t MsToNs(double ms) { return ms > 0.0 ? static_cast<std::uint64_t>(ms * 1e6) : 0; }

bool IsGguf(const fs::path& p) {
    std::string ext = PathToUtf8(p.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".gguf";
}

class LlamaCppModel final : public BackendModel {
public:
    LlamaCppModel(ModelDescriptor desc, lc::LlamaCppBackend runtime, double load_ms)
        : desc_(std::move(desc)), runtime_(std::move(runtime)), load_ns_(MsToNs(load_ms)) {}

    [[nodiscard]] const ModelDescriptor& descriptor() const override { return desc_; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        if (Status v = validate(request.sampling); !v.ok()) return v;
        // One llama_context per model: serialize requests on it.
        std::lock_guard<std::mutex> lock(mutex_);
        return run(request.prompt, /*templated=*/false, request.sampling, cancel, on_chunk);
    }

    // Chat through the GGUF chat template (llama_chat_apply_template). Models
    // without a template, or with one llama.cpp does not recognise, use the
    // generic BackendModel::chat() prompt format instead.
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        if (Status v = validate_chat_messages(request.messages); !v.ok()) return v;
        if (Status v = validate(request.sampling); !v.ok()) return v;
        std::vector<lc::ChatTurn> turns;
        turns.reserve(request.messages.size());
        for (const auto& m : request.messages) turns.push_back(lc::ChatTurn{m.role, m.content});
        std::unique_lock<std::mutex> lock(mutex_);
        std::string prompt;
        if (!runtime_.ApplyChatTemplate(turns, /*add_assistant=*/true, prompt).ok()) {
            lock.unlock();
            return BackendModel::chat(request, cancel, on_chunk);
        }
        return run(prompt, /*templated=*/true, request.sampling, cancel, on_chunk);
    }

    [[nodiscard]] bool has_native_chat() const override {
        std::string ignored;
        std::lock_guard<std::mutex> lock(mutex_);
        return runtime_.ApplyChatTemplate({lc::ChatTurn{"user", "x"}}, true, ignored).ok();
    }

private:
    // Caller holds mutex_. `templated` prompts come from the model's chat
    // template and contain control-token text that must parse as specials.
    Result<GenerateStats> run(const std::string& prompt, bool templated, const SamplingConfig& sampling,
                              const CancellationToken& cancel, const TokenCallback& on_chunk) {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");

        lc::GenerateRequest req;
        if (lc::Status s = runtime_.Tokenize(prompt, /*add_special=*/true, req.prompt, /*parse_special=*/templated);
            !s.ok()) {
            return ToStatus(s);
        }
        if (req.prompt.empty()) return Status(ErrorCode::invalid_argument, "prompt produced no tokens");
        req.max_new_tokens = sampling.max_tokens;
        req.sampling = ToLlamaSampling(sampling);

        GenerateStats stats;
        stats.load_ns = load_ns_;
        stats.token_counts_from_backend = true;
        StopSequenceFilter stop(sampling.stop);
        bool callback_stopped = false;

        auto deliver = [&](std::string_view text) {
            if (text.empty() || callback_stopped) return;
            if (on_chunk && !on_chunk(TokenChunk{text, stats.chunks})) callback_stopped = true;
            ++stats.chunks;
        };

        const lc::GenerateResult r = runtime_.Generate(
            req,
            [&](const lc::TokenEvent& ev) {
                deliver(stop.Push(ev.text));
                return !callback_stopped && !stop.matched();
            },
            [&] { return cancel.cancelled(); });

        stats.prompt_tokens = static_cast<std::uint64_t>(r.prompt_tokens);
        stats.completion_tokens = static_cast<std::uint64_t>(r.generated_tokens);
        stats.prompt_eval_ns = MsToNs(r.prefill_ms);
        stats.eval_ns = MsToNs(r.decode_ms);
        if (!r.status.ok()) return ToStatus(r.status);

        switch (r.stop_reason) {
            case lc::StopReason::kEndOfGeneration: stats.stop_reason = StopReason::end_of_sequence; break;
            case lc::StopReason::kMaxTokens: stats.stop_reason = StopReason::max_tokens; break;
            case lc::StopReason::kContextFull: stats.stop_reason = StopReason::max_tokens; break;
            case lc::StopReason::kCancelled:
                if (stop.matched()) {
                    stats.stop_reason = StopReason::stop_sequence;
                } else if (callback_stopped) {
                    stats.stop_reason = StopReason::callback;
                } else {
                    return Status(ErrorCode::cancelled, "generation cancelled");
                }
                break;
            default: stats.stop_reason = StopReason::error; break;
        }
        if (stats.stop_reason != StopReason::stop_sequence && stats.stop_reason != StopReason::callback) {
            deliver(stop.Flush());
        }
        return stats;
    }

    ModelDescriptor desc_;
    lc::LlamaCppBackend runtime_;
    std::uint64_t load_ns_ = 0;
    mutable std::mutex mutex_;
};

class LlamaCppCoreBackend final : public Backend {
public:
    explicit LlamaCppCoreBackend(LlamaCppBackendOptions options) : options_(std::move(options)) {}

    [[nodiscard]] std::string name() const override { return kLlamaCppBackendName; }
    [[nodiscard]] std::string description() const override {
        return "direct llama.cpp/GGML (" + std::string(lc::LlamaCppBackend::UpstreamVersion()) + "), in-process GGUF";
    }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        caps.add(Capability::tokenization)
            .add(Capability::streaming)
            .add(Capability::batched_prefill)
            .add(Capability::deterministic);
        return caps;
    }

    Result<std::string> probe() override {
        std::string devices;
        for (const auto& d : lc::LlamaCppBackend::EnumerateDevices()) {
            if (!devices.empty()) devices += ", ";
            devices += d.name + "(" + std::string(lc::ToString(d.kind)) + ")";
        }
        return "llama.cpp " + std::string(lc::LlamaCppBackend::UpstreamVersion()) + " [" + devices + "]";
    }

    Result<std::vector<ModelDescriptor>> list_models() override {
        std::vector<ModelDescriptor> out;
        for (const auto& dir : options_.model_dirs) {
            std::error_code ec;
            for (fs::directory_iterator it(PathFromUtf8(dir), ec), end; !ec && it != end; it.increment(ec)) {
                if (!it->is_regular_file(ec) || !IsGguf(it->path())) continue;
                ModelDescriptor d;
                d.name = PathToUtf8(it->path().filename());
                d.backend = kLlamaCppBackendName;
                d.format = "gguf";
                d.size_bytes = it->file_size(ec);
                out.push_back(std::move(d));
            }
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
        return out;
    }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        if (options.model.empty()) return Status(ErrorCode::invalid_argument, "model path is empty");
        const std::string path = Resolve(options.model);

        lc::LoadOptions lo;
        lo.model_path = path;
        lo.n_ctx = options_.context_length;
        lo.n_batch = options_.batch_size;
        lo.n_threads = options_.threads;
        const std::string_view dev = options.device_id;
        if (dev.empty() || dev.rfind("cpu", 0) == 0) {
            lo.n_gpu_layers = 0;
        } else if (dev.rfind("gpu", 0) == 0) {
            bool has_gpu = false;
            for (const auto& d : lc::LlamaCppBackend::EnumerateDevices()) {
                has_gpu = has_gpu || d.kind == lc::DeviceKind::kGpu || d.kind == lc::DeviceKind::kIntegratedGpu;
            }
            if (!has_gpu) return Status(ErrorCode::unsupported, "this llama.cpp build has no GPU backend");
            lo.n_gpu_layers = options_.gpu_layers;
        } else {
            return Status(ErrorCode::unsupported, "unsupported device id for llamacpp: " + options.device_id);
        }

        lc::LlamaCppBackend runtime;
        double load_ms = 0.0;
        runtime.SetTelemetrySink([&](const lc::TelemetryEvent& e) {
            if (e.kind == lc::TelemetryKind::kModelLoaded) load_ms = e.elapsed_ms;
        });
        if (lc::Status s = runtime.Load(lo); !s.ok()) return ToStatus(s);
        runtime.SetTelemetrySink({});

        const lc::ModelInfo info = runtime.GetModelInfo();
        ModelDescriptor d;
        d.name = PathToUtf8(PathFromUtf8(path).filename());
        d.backend = kLlamaCppBackendName;
        d.format = "gguf";
        d.family = info.description;  // e.g. "llama 7B Q4_0"
        d.size_bytes = info.size_bytes;
        d.context_length = info.n_ctx;
        const std::size_t sp = info.description.find(' ');
        if (sp != std::string::npos) {
            d.family = info.description.substr(0, sp);
            const std::string rest = info.description.substr(sp + 1);
            const std::size_t sp2 = rest.find(' ');
            d.parameter_size = rest.substr(0, sp2);
            if (sp2 != std::string::npos) d.quantization = rest.substr(sp2 + 1);
        }
        return std::shared_ptr<BackendModel>(std::make_shared<LlamaCppModel>(std::move(d), std::move(runtime), load_ms));
    }

private:
    std::string Resolve(const std::string& model) const {
        std::error_code ec;
        if (fs::is_regular_file(PathFromUtf8(model), ec)) return model;
        for (const auto& dir : options_.model_dirs) {
            const fs::path candidate = PathFromUtf8(dir) / PathFromUtf8(model);
            if (fs::is_regular_file(candidate, ec)) return PathToUtf8(candidate);
        }
        return model;  // Load() reports not_found with the original name
    }

    LlamaCppBackendOptions options_;
};

}  // namespace

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options) {
    return std::make_shared<LlamaCppCoreBackend>(std::move(options));
}

}  // namespace sonder::inference
