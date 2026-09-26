// Sonder Inference - llama.cpp backend.
//
// Thin, policy-free adapter over the llama.cpp C API. It owns one model and one
// context, and exposes load/unload, tokenization, prefill + streaming decode
// with cooperative cancellation, device enumeration, and telemetry hooks.
//
// No llama.cpp header is included here; all llama.cpp types stay in the .cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::backends::llamacpp {

using Token = std::int32_t;

enum class ErrorCode {
    kOk = 0,
    kInvalidArgument,
    kFileNotFound,
    kLoadFailed,
    kNotLoaded,
    kTokenizeFailed,
    kContextOverflow,
    kDecodeFailed,
};

struct Status {
    ErrorCode code = ErrorCode::kOk;
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::kOk; }
    static Status Ok() { return {}; }
    static Status Error(ErrorCode c, std::string msg) { return {c, std::move(msg)}; }
};

[[nodiscard]] std::string_view ToString(ErrorCode code) noexcept;

struct LoadOptions {
    std::string model_path;       // GGUF file
    std::uint32_t n_ctx = 2048;   // 0 = model training context
    std::uint32_t n_batch = 512;  // max tokens per llama_decode during prefill
    std::int32_t n_threads = 0;   // 0 = llama.cpp default
    std::int32_t n_gpu_layers = 0;  // CPU-only by default; -1 = offload all
    bool vocab_only = false;      // tokenizer metadata only, no weights
    bool use_mmap = true;
};

struct SamplingParams {
    float temperature = 0.0F;  // <= 0 selects greedy decoding
    std::int32_t top_k = 40;   // <= 0 disables
    float top_p = 0.95F;       // >= 1 disables
    float min_p = 0.0F;        // <= 0 disables
    float repeat_penalty = 1.0F;       // 1 disables
    std::int32_t repeat_last_n = 64;   // penalty window; 0 disables, -1 = whole context
    float presence_penalty = 0.0F;     // 0 disables
    float frequency_penalty = 0.0F;    // 0 disables
    float typical_p = 1.0F;            // >= 1 disables
    std::vector<std::pair<Token, float>> logit_bias;  // token -> additive bias (-inf bans)
    std::uint32_t seed = 0xC0FFEEU;    // 0xFFFFFFFF = random
};

[[nodiscard]] Status Validate(const SamplingParams& params);

enum class StopReason {
    kNone = 0,
    kEndOfGeneration,  // model emitted an end-of-generation token
    kMaxTokens,
    kCancelled,
    kContextFull,
    kError,
};

[[nodiscard]] std::string_view ToString(StopReason reason) noexcept;

struct GenerateRequest {
    std::vector<Token> prompt;     // already tokenized; must be non-empty
    std::int32_t max_new_tokens = 128;
    SamplingParams sampling{};
    bool reset_context = true;     // clear KV memory before prefill
};

// One event per generated token. After the last token a final event with
// token == -1 may carry leftover bytes of an incomplete UTF-8 sequence.
struct TokenEvent {
    Token token = 0;
    std::string text;         // UTF-8 complete text released by this token (may be empty)
    std::int32_t index = 0;   // 0-based index among generated tokens
};

// Return false to stop generation (treated as cancellation).
using TokenCallback = std::function<bool(const TokenEvent&)>;
// Polled between tokens, between prefill batches, and from llama.cpp's abort
// callback inside llama_decode; return true to cancel. Must be thread-safe.
using CancelPredicate = std::function<bool()>;

struct GenerateResult {
    Status status{};
    StopReason stop_reason = StopReason::kNone;
    std::int32_t prompt_tokens = 0;
    std::int32_t generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;

    [[nodiscard]] double decode_tokens_per_second() const noexcept;
    [[nodiscard]] double prefill_tokens_per_second() const noexcept;
};

enum class DeviceKind { kCpu, kGpu, kIntegratedGpu, kAccelerator, kOther };

struct DeviceInfo {
    std::string name;
    std::string description;
    DeviceKind kind = DeviceKind::kOther;
    std::uint64_t memory_free_bytes = 0;
    std::uint64_t memory_total_bytes = 0;
};

[[nodiscard]] std::string_view ToString(DeviceKind kind) noexcept;

struct ModelInfo {
    std::string description;
    std::uint64_t size_bytes = 0;
    std::uint64_t n_params = 0;
    std::int32_t n_vocab = 0;
    std::uint32_t n_ctx = 0;  // context actually allocated (0 when vocab_only)
};

enum class TelemetryKind {
    kModelLoaded,
    kModelUnloaded,
    kPrefillDone,
    kTokenDecoded,
    kGenerationDone,
};

struct TelemetryEvent {
    TelemetryKind kind{};
    std::int32_t tokens = 0;   // tokens covered by this event
    double elapsed_ms = 0.0;   // wall time covered by this event
    std::string detail;
};

using TelemetrySink = std::function<void(const TelemetryEvent&)>;

// One chat message for chat-template formatting (role: system, user,
// assistant, tool).
struct ChatTurn {
    std::string role;
    std::string content;
};

// Backend capabilities advertised to the core (see docs/BACKENDS.md).
struct Capabilities {
    bool tokenization = true;
    bool batched_prefill = true;
    bool streaming_decode = true;
    bool cancellation = true;
    bool kv_export = false;
    bool kv_import = false;
    bool prefix_reuse = false;
    bool speculative_decode = false;
};

class LlamaCppBackend {
public:
    LlamaCppBackend();
    ~LlamaCppBackend();
    LlamaCppBackend(const LlamaCppBackend&) = delete;
    LlamaCppBackend& operator=(const LlamaCppBackend&) = delete;
    LlamaCppBackend(LlamaCppBackend&&) noexcept;
    LlamaCppBackend& operator=(LlamaCppBackend&&) noexcept;

    [[nodiscard]] static std::string_view Name() noexcept { return "llama.cpp"; }
    [[nodiscard]] static std::string_view UpstreamVersion() noexcept;  // pinned tag
    [[nodiscard]] static Capabilities GetCapabilities() noexcept { return {}; }
    // Devices ggml can see in this build (CPU always; CUDA/Vulkan/Metal when compiled in).
    [[nodiscard]] static std::vector<DeviceInfo> EnumerateDevices();
    [[nodiscard]] static std::string SystemInfo();

    Status Load(const LoadOptions& options);
    void Unload() noexcept;
    [[nodiscard]] bool IsLoaded() const noexcept;
    [[nodiscard]] ModelInfo GetModelInfo() const;

    // parse_special: treat control-token text (e.g. "<|im_start|>") as the
    // special token rather than plain text; needed for chat-templated prompts.
    Status Tokenize(std::string_view text, bool add_special, std::vector<Token>& out,
                    bool parse_special = false) const;
    Status TokenToPiece(Token token, std::string& out) const;
    [[nodiscard]] bool IsEndOfGeneration(Token token) const;

    // Chat template stored in the GGUF metadata (tokenizer.chat_template), or
    // empty when the model has none / nothing is loaded.
    [[nodiscard]] std::string ChatTemplate() const;
    // Formats `messages` with the model's chat template (llama.cpp's built-in
    // template matcher, not a Jinja engine). kInvalidArgument when the model
    // has no template or llama.cpp does not recognise it; callers fall back
    // to a generic prompt format in that case.
    Status ApplyChatTemplate(const std::vector<ChatTurn>& messages, bool add_assistant, std::string& out) const;
    // Same, with an explicit template: a Jinja template string or a built-in
    // name such as "chatml", "llama3", "mistral-v7". Needs no loaded model.
    static Status FormatChat(std::string_view chat_template, const std::vector<ChatTurn>& messages,
                             bool add_assistant, std::string& out);

    GenerateResult Generate(const GenerateRequest& request, const TokenCallback& on_token,
                            const CancelPredicate& should_cancel = {});

    void SetTelemetrySink(TelemetrySink sink);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- Helpers (pure; no llama.cpp dependency; unit-tested standalone) ----

// Splits a byte stream produced by token pieces into complete UTF-8 text.
// Token pieces can end in the middle of a multi-byte code point; this holds
// the incomplete tail until the next piece completes it.
class Utf8StreamBuffer {
public:
    // Appends bytes and returns the longest complete UTF-8 prefix accumulated.
    std::string Push(std::string_view bytes);
    // Returns whatever is left (possibly an invalid partial sequence).
    std::string Flush();
    [[nodiscard]] std::size_t pending() const noexcept { return pending_.size(); }

private:
    std::string pending_;
};

// Length of the longest prefix of `s` that does not end in an incomplete
// UTF-8 sequence. Invalid bytes are treated as complete single bytes.
[[nodiscard]] std::size_t CompleteUtf8Prefix(std::string_view s) noexcept;

}  // namespace sonder::backends::llamacpp
