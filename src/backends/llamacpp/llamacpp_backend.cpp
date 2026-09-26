// Sonder Inference - llama.cpp backend implementation.
#include "sonder/backends/llamacpp/llamacpp_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <utility>

#include "ggml-backend.h"
#include "llama.h"

#ifndef SONDER_LLAMACPP_TAG_STRING
#define SONDER_LLAMACPP_TAG_STRING "unknown"
#endif

namespace sonder::backends::llamacpp {
namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// llama.cpp logs verbosely to stderr by default. Keep warnings and errors;
// set SONDER_LLAMACPP_VERBOSE=1 to see everything.
void LogCallback(ggml_log_level level, const char* text, void* /*user*/) {
    static const bool verbose = [] {
        const char* v = std::getenv("SONDER_LLAMACPP_VERBOSE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

void EnsureBackendInitialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(&LogCallback, nullptr);
        llama_backend_init();
    });
}

DeviceKind MapDeviceKind(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return DeviceKind::kCpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU: return DeviceKind::kGpu;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return DeviceKind::kIntegratedGpu;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return DeviceKind::kAccelerator;
        default: return DeviceKind::kOther;
    }
}

struct SamplerDeleter {
    void operator()(llama_sampler* s) const noexcept { llama_sampler_free(s); }
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

// Order follows llama.cpp's common sampler: logit bias, penalties, top-k,
// typical, top-p, min-p, temperature, then the selector.
SamplerPtr MakeSampler(const SamplingParams& p, int32_t n_vocab, int32_t n_ctx) {
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!p.logit_bias.empty()) {
        std::vector<llama_logit_bias> bias;
        bias.reserve(p.logit_bias.size());
        for (const auto& [token, value] : p.logit_bias) bias.push_back(llama_logit_bias{token, value});
        llama_sampler_chain_add(chain.get(), llama_sampler_init_logit_bias(n_vocab, static_cast<int32_t>(bias.size()),
                                                                           bias.data()));
    }
    const int32_t last_n = p.repeat_last_n < 0 ? n_ctx : p.repeat_last_n;
    const bool penalties = p.repeat_penalty != 1.0F || p.presence_penalty != 0.0F || p.frequency_penalty != 0.0F;
    if (penalties && last_n > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_penalties(n_vocab, last_n, p.repeat_penalty,
                                                                          p.frequency_penalty, p.presence_penalty));
    }
    if (p.temperature <= 0.0F) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        return chain;
    }
    if (p.top_k > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(p.top_k));
    if (p.typical_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_typical(p.typical_p, 1));
    if (p.top_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(p.top_p, 1));
    if (p.min_p > 0.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(p.min_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(p.temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(p.seed));
    return chain;
}

}  // namespace

struct LlamaCppBackend::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    LoadOptions options{};
    TelemetrySink telemetry;
    // Consulted by llama.cpp's abort callback during long llama_decode calls.
    const CancelPredicate* active_cancel = nullptr;
    std::atomic<bool> abort_requested{false};

    ~Impl() { Release(); }

    void Release() noexcept {
        if (ctx != nullptr) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
    }

    void Emit(TelemetryKind kind, std::int32_t tokens, double ms, std::string detail = {}) const {
        if (telemetry) telemetry(TelemetryEvent{kind, tokens, ms, std::move(detail)});
    }

    static bool AbortCallback(void* data) {
        auto* self = static_cast<Impl*>(data);
        if (self->abort_requested.load(std::memory_order_relaxed)) return true;
        if (self->active_cancel != nullptr && *self->active_cancel && (*self->active_cancel)()) {
            self->abort_requested.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    }
};

std::string_view LlamaCppBackend::UpstreamVersion() noexcept { return SONDER_LLAMACPP_TAG_STRING; }

std::vector<DeviceInfo> LlamaCppBackend::EnumerateDevices() {
    EnsureBackendInitialized();
    std::vector<DeviceInfo> out;
    const std::size_t count = ggml_backend_dev_count();
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        DeviceInfo info;
        if (const char* n = ggml_backend_dev_name(dev)) info.name = n;
        if (const char* d = ggml_backend_dev_description(dev)) info.description = d;
        info.kind = MapDeviceKind(ggml_backend_dev_type(dev));
        std::size_t free_b = 0;
        std::size_t total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        info.memory_free_bytes = free_b;
        info.memory_total_bytes = total_b;
        out.push_back(std::move(info));
    }
    return out;
}

std::string LlamaCppBackend::SystemInfo() {
    EnsureBackendInitialized();
    const char* s = llama_print_system_info();
    return s != nullptr ? std::string(s) : std::string();
}

LlamaCppBackend::LlamaCppBackend() : impl_(std::make_unique<Impl>()) {}
LlamaCppBackend::~LlamaCppBackend() = default;
LlamaCppBackend::LlamaCppBackend(LlamaCppBackend&&) noexcept = default;
LlamaCppBackend& LlamaCppBackend::operator=(LlamaCppBackend&&) noexcept = default;

Status LlamaCppBackend::Load(const LoadOptions& options) {
    if (options.model_path.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "model_path is empty");
    }
    std::error_code ec;
    const std::filesystem::path fs_path(std::u8string(options.model_path.begin(), options.model_path.end()));
    if (!std::filesystem::is_regular_file(fs_path, ec)) {
        return Status::Error(ErrorCode::kFileNotFound, "model file not found: " + options.model_path);
    }
    if (!options.vocab_only && options.n_batch == 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "n_batch must be > 0");
    }

    EnsureBackendInitialized();
    Unload();

    const auto start = Clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = options.n_gpu_layers;
    mp.vocab_only = options.vocab_only;
    if (!options.use_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model* model = llama_model_load_from_file(options.model_path.c_str(), mp);
    if (model == nullptr) {
        return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to load: " + options.model_path);
    }
    impl_->model = model;
    impl_->vocab = llama_model_get_vocab(model);
    impl_->options = options;

    if (!options.vocab_only) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = options.n_ctx;
        cp.n_batch = options.n_batch;
        cp.n_ubatch = std::min(cp.n_ubatch, options.n_batch);
        cp.n_seq_max = 1;
        if (options.n_threads > 0) {
            cp.n_threads = options.n_threads;
            cp.n_threads_batch = options.n_threads;
        }
        cp.abort_callback = &Impl::AbortCallback;
        cp.abort_callback_data = impl_.get();
        impl_->ctx = llama_init_from_model(model, cp);
        if (impl_->ctx == nullptr) {
            impl_->Release();
            return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to create context");
        }
    }

    impl_->Emit(TelemetryKind::kModelLoaded, 0, MsSince(start), options.model_path);
    return Status::Ok();
}

void LlamaCppBackend::Unload() noexcept {
    if (!impl_ || impl_->model == nullptr) return;
    impl_->Release();
    try {
        impl_->Emit(TelemetryKind::kModelUnloaded, 0, 0.0, impl_->options.model_path);
    } catch (...) {
        // Telemetry sinks must not break unload.
    }
}

bool LlamaCppBackend::IsLoaded() const noexcept { return impl_ && impl_->model != nullptr; }

ModelInfo LlamaCppBackend::GetModelInfo() const {
    ModelInfo info;
    if (!IsLoaded()) return info;
    char buf[256] = {};
    if (llama_model_desc(impl_->model, buf, sizeof(buf)) > 0) info.description = buf;
    info.size_bytes = llama_model_size(impl_->model);
    info.n_params = llama_model_n_params(impl_->model);
    info.n_vocab = llama_vocab_n_tokens(impl_->vocab);
    info.n_ctx = impl_->ctx != nullptr ? llama_n_ctx(impl_->ctx) : 0;
    return info;
}

Status LlamaCppBackend::Tokenize(std::string_view text, bool add_special, std::vector<Token>& out,
                                 bool parse_special) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (text.size() > static_cast<std::size_t>(INT32_MAX)) {
        return Status::Error(ErrorCode::kInvalidArgument, "text too long");
    }
    const auto len = static_cast<int32_t>(text.size());
    // Upper bound: one token per byte plus BOS/EOS.
    out.resize(static_cast<std::size_t>(len) + 2);
    int32_t n = llama_tokenize(impl_->vocab, text.data(), len, out.data(),
                               static_cast<int32_t>(out.size()), add_special, parse_special);
    if (n < 0 && n != INT32_MIN) {
        out.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(impl_->vocab, text.data(), len, out.data(),
                           static_cast<int32_t>(out.size()), add_special, parse_special);
    }
    if (n < 0) {
        out.clear();
        return Status::Error(ErrorCode::kTokenizeFailed, "llama_tokenize failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return Status::Ok();
}

Status LlamaCppBackend::TokenToPiece(Token token, std::string& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (token < 0 || token >= llama_vocab_n_tokens(impl_->vocab)) {
        return Status::Error(ErrorCode::kInvalidArgument, "token id out of range");
    }
    char stack_buf[64];
    int32_t n = llama_token_to_piece(impl_->vocab, token, stack_buf, sizeof(stack_buf), 0, /*special=*/false);
    if (n >= 0) {
        out.assign(stack_buf, static_cast<std::size_t>(n));
        return Status::Ok();
    }
    out.resize(static_cast<std::size_t>(-n));
    n = llama_token_to_piece(impl_->vocab, token, out.data(), static_cast<int32_t>(out.size()), 0, false);
    if (n < 0) {
        out.clear();
        return Status::Error(ErrorCode::kTokenizeFailed, "llama_token_to_piece failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return Status::Ok();
}

bool LlamaCppBackend::IsEndOfGeneration(Token token) const {
    return IsLoaded() && llama_vocab_is_eog(impl_->vocab, token);
}

std::string LlamaCppBackend::ChatTemplate() const {
    if (!IsLoaded()) return {};
    const char* tmpl = llama_model_chat_template(impl_->model, /*name=*/nullptr);
    return tmpl != nullptr ? std::string(tmpl) : std::string();
}

Status LlamaCppBackend::ApplyChatTemplate(const std::vector<ChatTurn>& messages, bool add_assistant,
                                          std::string& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    const std::string tmpl = ChatTemplate();
    if (tmpl.empty()) return Status::Error(ErrorCode::kInvalidArgument, "model has no chat template");
    return FormatChat(tmpl, messages, add_assistant, out);
}

Status LlamaCppBackend::FormatChat(std::string_view chat_template, const std::vector<ChatTurn>& messages,
                                   bool add_assistant, std::string& out) {
    out.clear();
    if (chat_template.empty()) return Status::Error(ErrorCode::kInvalidArgument, "empty chat template");
    if (messages.empty()) return Status::Error(ErrorCode::kInvalidArgument, "no chat messages");
    const std::string tmpl(chat_template);
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    std::size_t total = 0;
    for (const auto& m : messages) {
        chat.push_back(llama_chat_message{m.role.c_str(), m.content.c_str()});
        total += m.role.size() + m.content.size();
    }
    if (total > static_cast<std::size_t>(INT32_MAX / 4)) {
        return Status::Error(ErrorCode::kInvalidArgument, "chat too long");
    }
    std::vector<char> buf(total * 2 + 256);
    int32_t n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(), add_assistant, buf.data(),
                                          static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(), add_assistant, buf.data(),
                                      static_cast<int32_t>(buf.size()));
    }
    if (n < 0) return Status::Error(ErrorCode::kInvalidArgument, "chat template not supported by llama.cpp");
    out.assign(buf.data(), static_cast<std::size_t>(n));
    return Status::Ok();
}

void LlamaCppBackend::SetTelemetrySink(TelemetrySink sink) { impl_->telemetry = std::move(sink); }

GenerateResult LlamaCppBackend::Generate(const GenerateRequest& request, const TokenCallback& on_token,
                                         const CancelPredicate& should_cancel) {
    GenerateResult result;
    auto fail = [&](ErrorCode code, std::string msg) {
        result.status = Status::Error(code, std::move(msg));
        result.stop_reason = StopReason::kError;
        return result;
    };

    if (!IsLoaded()) return fail(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return fail(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    if (request.prompt.empty()) return fail(ErrorCode::kInvalidArgument, "prompt is empty");
    if (request.max_new_tokens < 0) return fail(ErrorCode::kInvalidArgument, "max_new_tokens < 0");
    if (Status s = Validate(request.sampling); !s.ok()) return fail(s.code, s.message);

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    for (Token t : request.prompt) {
        if (t < 0 || t >= n_vocab) return fail(ErrorCode::kInvalidArgument, "prompt token out of range");
    }
    for (const auto& entry : request.sampling.logit_bias) {
        if (entry.first < 0 || entry.first >= n_vocab) {
            return fail(ErrorCode::kInvalidArgument, "logit_bias token out of range");
        }
    }

    llama_context* ctx = impl_->ctx;
    llama_memory_t mem = llama_get_memory(ctx);
    if (request.reset_context) llama_memory_clear(mem, true);

    const auto n_ctx = static_cast<int64_t>(llama_n_ctx(ctx));
    const int64_t n_past = llama_memory_seq_pos_max(mem, 0) + 1;  // 0 when empty
    const auto n_prompt = static_cast<int64_t>(request.prompt.size());
    if (n_past + n_prompt > n_ctx) {
        return fail(ErrorCode::kContextOverflow, "prompt does not fit in the context window");
    }

    // Wire cancellation into llama.cpp's abort callback for the duration of
    // this call so a long prefill batch can also be interrupted.
    impl_->abort_requested.store(false, std::memory_order_relaxed);
    impl_->active_cancel = &should_cancel;
    struct ClearCancel {
        Impl* impl;
        ~ClearCancel() { impl->active_cancel = nullptr; }
    } clear_cancel{impl_.get()};

    auto cancelled = [&] {
        return impl_->abort_requested.load(std::memory_order_relaxed) || (should_cancel && should_cancel());
    };

    // ---- Prefill in n_batch chunks ----
    std::vector<Token> prompt = request.prompt;  // llama_batch_get_one takes a mutable pointer
    const auto n_batch = static_cast<int64_t>(std::max<uint32_t>(1, impl_->options.n_batch));
    const auto prefill_start = Clock::now();
    for (int64_t off = 0; off < n_prompt; off += n_batch) {
        if (cancelled()) {
            result.stop_reason = StopReason::kCancelled;
            result.prefill_ms = MsSince(prefill_start);
            return result;
        }
        const auto n = static_cast<int32_t>(std::min(n_batch, n_prompt - off));
        const int32_t rc = llama_decode(ctx, llama_batch_get_one(prompt.data() + off, n));
        if (rc == 2) {
            result.stop_reason = StopReason::kCancelled;
            result.prefill_ms = MsSince(prefill_start);
            return result;
        }
        if (rc != 0) {
            return fail(ErrorCode::kDecodeFailed, "llama_decode failed during prefill (rc=" + std::to_string(rc) + ")");
        }
        result.prompt_tokens += n;
    }
    result.prefill_ms = MsSince(prefill_start);
    impl_->Emit(TelemetryKind::kPrefillDone, result.prompt_tokens, result.prefill_ms);

    // ---- Streaming decode ----
    SamplerPtr sampler = MakeSampler(request.sampling, n_vocab, static_cast<int32_t>(n_ctx));
    Utf8StreamBuffer utf8;
    int64_t pos = n_past + n_prompt;
    const auto decode_start = Clock::now();
    auto finish = [&](StopReason reason) {
        result.stop_reason = reason;
        result.decode_ms = MsSince(decode_start);
        std::string tail = utf8.Flush();
        if (!tail.empty() && on_token && reason != StopReason::kError) {
            on_token(TokenEvent{-1, std::move(tail), result.generated_tokens});
        }
        impl_->Emit(TelemetryKind::kGenerationDone, result.generated_tokens, result.decode_ms,
                    std::string(ToString(reason)));
        return result;
    };

    while (true) {
        if (result.generated_tokens >= request.max_new_tokens) return finish(StopReason::kMaxTokens);
        if (cancelled()) return finish(StopReason::kCancelled);

        const auto token_start = Clock::now();
        Token tok = llama_sampler_sample(sampler.get(), ctx, -1);
        if (llama_vocab_is_eog(impl_->vocab, tok)) return finish(StopReason::kEndOfGeneration);

        std::string piece;
        if (Status s = TokenToPiece(tok, piece); !s.ok()) {
            result.status = s;
            return finish(StopReason::kError);
        }
        TokenEvent ev{tok, utf8.Push(piece), result.generated_tokens};
        ++result.generated_tokens;
        const bool keep_going = on_token ? on_token(ev) : true;
        if (!keep_going) return finish(StopReason::kCancelled);
        if (result.generated_tokens >= request.max_new_tokens) {
            impl_->Emit(TelemetryKind::kTokenDecoded, 1, MsSince(token_start));
            return finish(StopReason::kMaxTokens);
        }
        if (pos >= n_ctx) return finish(StopReason::kContextFull);

        const int32_t rc = llama_decode(ctx, llama_batch_get_one(&tok, 1));
        impl_->Emit(TelemetryKind::kTokenDecoded, 1, MsSince(token_start));
        if (rc == 2) return finish(StopReason::kCancelled);
        if (rc == 1) return finish(StopReason::kContextFull);
        if (rc != 0) {
            result.status =
                Status::Error(ErrorCode::kDecodeFailed, "llama_decode failed (rc=" + std::to_string(rc) + ")");
            return finish(StopReason::kError);
        }
        ++pos;
    }
}

}  // namespace sonder::backends::llamacpp
