// Adapter: sonder::inference::Backend over the llama.cpp runtime wrapper.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "llamacpp_internal.hpp"

namespace sonder::inference {

namespace llamacpp_detail {

// Resolves LoadOptions::n_gpu_layers from backend options and a load request's
// device_id. Explicit "cpu:*" always uses 0; "gpu:*" requires a GPU backend
// and uses options.gpu_layers; empty device_id honors options.gpu_layers when
// explicit ggml devices are configured, otherwise defaults to 0 (CPU).
Result<std::int32_t> ResolveGpuLayers(const LlamaCppBackendOptions& options, std::string_view device_id,
                                      bool has_gpu_backend);

Result<std::int32_t> ResolveGpuLayers(const LlamaCppBackendOptions& options, std::string_view device_id,
                                      bool has_gpu_backend) {
    if (device_id.rfind("cpu", 0) == 0) {
        return std::int32_t{0};
    }
    if (device_id.empty()) {
        return options.devices.empty() ? std::int32_t{0} : options.gpu_layers;
    }
    if (device_id.rfind("gpu", 0) == 0) {
        if (!has_gpu_backend) {
            return Status(ErrorCode::unsupported, "this llama.cpp build has no GPU backend");
        }
        return options.gpu_layers;
    }
    return Status(ErrorCode::unsupported, "unsupported device id for llamacpp: " + std::string(device_id));
}

}  // namespace llamacpp_detail

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

// Copies the context and placement options into `lo` and validates them with
// the wrapper's pure validators (the same checks Load() runs).
Status ApplyBackendOptions(const LlamaCppBackendOptions& o, lc::LoadOptions& lo) {
    lo.n_ctx = o.context_length;
    lo.n_batch = o.batch_size;
    lo.n_ubatch = o.ubatch_size;
    lo.n_threads = o.threads;
    lo.devices = o.devices;
    lo.main_gpu = o.main_gpu;
    lo.tensor_split = o.tensor_split;
    lo.use_mlock = o.use_mlock;
    if (!lc::ParseSplitMode(o.split_mode, lo.split_mode)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown split mode '" + o.split_mode + "' (expected none, layer or row)");
    }
    if (!lc::ParseKvCacheType(o.kv_cache_type_k, lo.type_k)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown K cache type '" + o.kv_cache_type_k +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0 or iq4_nl)");
    }
    if (!lc::ParseKvCacheType(o.kv_cache_type_v, lo.type_v)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown V cache type '" + o.kv_cache_type_v +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0 or iq4_nl)");
    }
    if (!lc::ParseFlashAttention(o.flash_attention, lo.flash_attn)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown flash attention mode '" + o.flash_attention + "' (expected auto, on or off)");
    }
    if (lc::Status s = lc::ValidatePlacementOptions(lo); !s.ok()) {
        return ToStatus(s);
    }
    return ToStatus(lc::ValidateContextOptions(lo));
}

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

    Result<std::vector<TokenId>> tokenize(std::string_view text) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<lc::Token> tokens;
        if (lc::Status s = runtime_.Tokenize(text, /*add_special=*/true, tokens, /*parse_special=*/false); !s.ok()) {
            return ToStatus(s);
        }
        return tokens;
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        if (Status v = validate(request.sampling); !v.ok()) return v;
        // The llama.cpp context size is fixed at load (ModelLoadOptions::context_length).
        // A smaller num_ctx is honoured by the engine's scheduling limit; a larger one
        // cannot be served without reloading.
        if (request.sampling.num_ctx > 0 && desc_.context_length > 0 &&
            static_cast<std::uint64_t>(request.sampling.num_ctx) > desc_.context_length) {
            return Status(ErrorCode::invalid_argument,
                          "num_ctx " + std::to_string(request.sampling.num_ctx) +
                              " exceeds the loaded llama.cpp context (" + std::to_string(desc_.context_length) +
                              "); reload with a larger context_length");
        }
        // One llama_context per model: serialize requests (including atomic
        // checkpoint restore and prefill-boundary capture) under one lock.
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<lc::Token> prompt_tokens;
        if (lc::Status s = runtime_.Tokenize(request.prompt, /*add_special=*/true, prompt_tokens, /*parse_special=*/false);
            !s.ok()) {
            return ToStatus(s);
        }
        return run_tokens(std::move(prompt_tokens), request.sampling, cancel, on_chunk, request.checkpoint);
    }

    // Chat through the GGUF chat template (llama_chat_apply_template). Models
    // without a template, or with one llama.cpp does not recognise, use the
    // generic BackendModel::chat() prompt format instead. Template scaffolding
    // is tokenized with parse_special=true while untrusted message content is
    // strictly tokenized with parse_special=false (G0028).
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        if (Status v = validate_chat_messages(request.messages); !v.ok()) return v;
        if (Status v = validate(request.sampling); !v.ok()) return v;
        if (request.sampling.num_ctx > 0 && desc_.context_length > 0 &&
            static_cast<std::uint64_t>(request.sampling.num_ctx) > desc_.context_length) {
            return Status(ErrorCode::invalid_argument,
                          "num_ctx " + std::to_string(request.sampling.num_ctx) +
                              " exceeds the loaded llama.cpp context (" + std::to_string(desc_.context_length) +
                              "); reload with a larger context_length");
        }
        std::unique_lock<std::mutex> lock(mutex_);
        bool template_unsupported = false;
        auto segments = build_chat_template_segments(
            request.messages,
            [&](const std::vector<ChatMessage>& placeholder_msgs) -> Result<std::string> {
                std::vector<lc::ChatTurn> turns;
                turns.reserve(placeholder_msgs.size());
                for (const auto& m : placeholder_msgs) turns.push_back(lc::ChatTurn{m.role, m.content});
                std::string rendered;
                const lc::Status s = runtime_.ApplyChatTemplate(turns, /*add_assistant=*/true, rendered);
                if (!s.ok()) {
                    template_unsupported = true;
                    return ToStatus(s);
                }
                return rendered;
            });
        if (!segments.ok()) {
            if (template_unsupported) {
                lock.unlock();
                return BackendModel::chat(request, cancel, on_chunk);
            }
            return segments.status();
        }
        std::vector<lc::Token> prompt_tokens;
        const Status tok_status = tokenize_chat_segments(
            segments.value(),
            [&](std::string_view piece, bool add_special, bool parse_special, std::vector<TokenId>& out) -> Status {
                return ToStatus(runtime_.Tokenize(piece, add_special, out, parse_special));
            },
            prompt_tokens);
        if (!tok_status.ok()) return tok_status;
        return run_tokens(std::move(prompt_tokens), request.sampling, cancel, on_chunk);
    }

    [[nodiscard]] bool has_native_chat() const override {
        std::string ignored;
        std::lock_guard<std::mutex> lock(mutex_);
        return runtime_.ApplyChatTemplate({lc::ChatTurn{"user", "x"}}, true, ignored).ok();
    }

    Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) override {
        if (token_position == 0) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint token_position must be > 0");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        RecurrentStateSnapshot snapshot;
        snapshot.token_position = token_position;
        if (lc::Status s = runtime_.CaptureSequenceState(snapshot.bytes, kMaxRecurrentCheckpointBytes); !s.ok()) {
            return ToStatus(s);
        }
        return snapshot;
    }

    Status restore_recurrent_state(std::span<const std::uint8_t> bytes) override {
        if (bytes.empty()) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint snapshot is empty");
        }
        if (bytes.size() > kMaxRecurrentCheckpointBytes) {
            return Status(ErrorCode::invalid_argument,
                          "recurrent checkpoint snapshot exceeds kMaxRecurrentCheckpointBytes");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return ToStatus(runtime_.RestoreSequenceState(bytes, kMaxRecurrentCheckpointBytes));
    }

private:
    // Caller holds mutex_. Prompt tokens are already produced across the
    // trusted/untrusted tokenization boundary.
    Result<GenerateStats> run_tokens(std::vector<lc::Token> prompt_tokens, const SamplingConfig& sampling,
                                     const CancellationToken& cancel, const TokenCallback& on_chunk,
                                     RecurrentCheckpointExchange* checkpoint = nullptr) {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");

        lc::GenerateRequest req;
        req.prompt = std::move(prompt_tokens);
        if (req.prompt.empty()) return Status(ErrorCode::invalid_argument, "prompt produced no tokens");
        req.max_new_tokens = sampling.max_tokens;
        req.sampling = ToLlamaSampling(sampling);
        if (checkpoint != nullptr) {
            if (!checkpoint->restore_bytes.empty() && checkpoint->restore_token_position > 0 &&
                checkpoint->restore_token_position < req.prompt.size()) {
                req.restore_state = checkpoint->restore_bytes;
                req.restore_tokens = static_cast<std::size_t>(checkpoint->restore_token_position);
                if (checkpoint->on_restored) {
                    req.on_state_restored = [checkpoint] {
                        checkpoint->restored = true;
                        checkpoint->on_restored();
                    };
                }
            }
            if (checkpoint->capture_at_token_position > 0 &&
                checkpoint->capture_at_token_position <= req.prompt.size()) {
                req.capture_at_tokens = static_cast<std::size_t>(checkpoint->capture_at_token_position);
                req.max_capture_bytes = kMaxRecurrentCheckpointBytes;
                req.on_state_captured = [checkpoint](std::vector<std::uint8_t> bytes) {
                    checkpoint->captured = true;
                    if (checkpoint->on_captured) {
                        checkpoint->on_captured(checkpoint->capture_at_token_position, std::move(bytes));
                    } else {
                        checkpoint->captured_bytes = std::move(bytes);
                    }
                };
            }
        }

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

        if (checkpoint != nullptr) {
            checkpoint->restored = r.state_restored;
            if (!r.restore_status.ok()) {
                checkpoint->restore_error = ToStatus(r.restore_status);
            }
            checkpoint->captured = r.state_captured;
            if (!r.capture_status.ok()) {
                checkpoint->capture_error = ToStatus(r.capture_status);
            }
        }

        const std::uint64_t restored_tokens =
            r.state_restored ? static_cast<std::uint64_t>(req.restore_tokens) : 0;
        stats.prompt_tokens = static_cast<std::uint64_t>(r.prompt_tokens) + restored_tokens;
        if (r.state_restored) {
            stats.cached_tokens = restored_tokens;
        }
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
        if (stats.stop_reason == StopReason::stop_sequence) {
            if (const auto matched = stop.matched_stop()) {
                stats.matched_stop = std::string(*matched);
            }
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
            .add(Capability::deterministic)
            .add(Capability::recurrent_checkpoint);
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
        if (Status s = ApplyBackendOptions(options_, lo); !s.ok()) return s;
        for (const auto& [pattern, device] : options_.tensor_overrides) {
            lo.tensor_overrides.push_back({pattern, device});
        }
        const std::string_view dev = options.device_id;
        bool has_gpu = false;
        if (dev.rfind("gpu", 0) == 0) {
            for (const auto& d : lc::LlamaCppBackend::EnumerateDevices()) {
                has_gpu = has_gpu || d.kind == lc::DeviceKind::kGpu || d.kind == lc::DeviceKind::kIntegratedGpu;
            }
        }
        const auto gpu_layers = llamacpp_detail::ResolveGpuLayers(options_, dev, has_gpu);
        if (!gpu_layers.ok()) return gpu_layers.status();
        lo.n_gpu_layers = gpu_layers.value();

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
        d.architecture = info.architecture;
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

Status validate_llamacpp_options(const LlamaCppBackendOptions& options) {
    lc::LoadOptions lo;
    return ApplyBackendOptions(options, lo);
}

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options) {
    return std::make_shared<LlamaCppCoreBackend>(std::move(options));
}

}  // namespace sonder::inference
