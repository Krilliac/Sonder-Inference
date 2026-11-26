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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/model_architecture.hpp"
#include "sonder/inference/request_priority.hpp"
#include "sonder/inference/speculative_verifier.hpp"
#include "sonder/inference/swa_policy.hpp"

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

// Places weight tensors whose name matches `pattern` (ECMAScript regex, matched
// anywhere in the name as llama.cpp does) in `device`'s memory instead of the
// default layer placement. `device` is "cpu" or a llama.cpp device name such
// as "Vulkan0" or "CUDA0" (see EnumerateDevices). First match wins.
struct TensorOverride {
    std::string pattern;
    std::string device;
};

// MoE expert weights (llama.cpp's --cpu-moe): keep them in system RAM while
// attention and shared weights stay on the GPU. For MoE models larger than
// VRAM this beats offloading whole layers, since only a few experts run per
// token.
inline constexpr const char* kMoeExpertTensorPattern = R"(\.ffn_(up|down|gate)_(ch|)exps)";

// Element type of the K or V cache (llama.cpp's --cache-type-k/-v). ToString
// returns ggml's type names ("f16", "q8_0", "iq4_nl", ...), which llama.cpp's
// own CLI also accepts. Extended types (f8_e4m3, q2_0, pq2_0) require a
// llama.cpp/ggml build that defines the corresponding GGML_TYPE_* enum value.
enum class KvCacheType {
    kF16,
    kF32,
    kBF16,
    kQ8_0,
    kQ5_1,
    kQ5_0,
    kQ4_1,
    kQ4_0,
    kIQ4_NL,
    kF8_E4M3,
    kQ2_0,
    kPQ2_0,
};

// When llama.cpp uses Flash Attention (--flash-attn). kAuto lets llama.cpp
// decide per model and device; it is llama.cpp's default.
enum class FlashAttention { kAuto, kDisabled, kEnabled };

// How llama.cpp distributes model weights and KV cache across GPUs
// (--split-mode). kLayer matches llama.cpp's default.
enum class SplitMode { kNone, kLayer, kRow };

// In-process speculative decoding strategy.
enum class SpeculativeType { kNone, kAuto, kNgram, kNgramMod, kSuffix, kDraftMtp };

// Verification coupling policy for speculative decoding.
using SpeculativeVerificationMode = sonder::inference::SpeculativeVerificationMode;

// Maximum number of GPU entries accepted in LoadOptions::tensor_split and
// LoadOptions::devices (matches llama_max_devices() in pinned llama.cpp).
inline constexpr std::size_t kMaxTensorSplitDevices = 16;

// Maximum number of cached KV sequence slots (1..kMaxSlots).
inline constexpr std::uint32_t kMaxSlots = 16;

// Maximum draft tokens per speculative step and per-strategy defaults.
inline constexpr std::int32_t kMaxSpecDraftTokens = 64;
inline constexpr std::int32_t kDefaultNgramDraftMax = 8;
inline constexpr std::int32_t kDefaultNgramModDraftMax = 64;
inline constexpr std::int32_t kDefaultSuffixDraftMax = 16;
inline constexpr std::int32_t kDefaultMtpDraftMax = 3;

// Bounded L2 host-RAM KV/SSM prefix cache cap (MiB) and dynamic MoE expert cache cap (MiB).
inline constexpr std::uint32_t kMaxCacheRamMiB = 65536;
inline constexpr std::uint32_t kMaxMoeCacheMiB = 262144;

// Cold shared-prefix prefill coalescing defaults and bounds.
inline constexpr std::uint32_t kDefaultMinCoalescePrefixTokens = 64;
inline constexpr std::uint32_t kDefaultMaxCoalesceFollowers = 8;
inline constexpr std::uint32_t kDefaultMaxCoalesceWaitMs = 5000;
inline constexpr std::uint32_t kMaxCoalesceWaitMs = 600000;

// Maximum top-K alternative logprobs returned per generated token.
inline constexpr std::int32_t kMaxTopLogprobs = 20;

// Strict multimodal media payload limits (max 8 images, max 16 MiB each).
inline constexpr std::size_t kMaxMediaImages = 8;
inline constexpr std::size_t kMaxMediaBlobBytes = std::size_t{16} << 20;

[[nodiscard]] std::string_view ToString(KvCacheType type) noexcept;
[[nodiscard]] std::string_view ToString(FlashAttention mode) noexcept;
[[nodiscard]] std::string_view ToString(SplitMode mode) noexcept;
[[nodiscard]] std::string_view ToString(SpeculativeType type) noexcept;
[[nodiscard]] std::string_view ToString(SpeculativeVerificationMode mode) noexcept;
// Case-insensitive; returns false and leaves `out` untouched for an unknown name.
[[nodiscard]] bool ParseKvCacheType(std::string_view name, KvCacheType& out) noexcept;
// Accepts "auto", "on"/"enabled" and "off"/"disabled" (case-insensitive).
[[nodiscard]] bool ParseFlashAttention(std::string_view name, FlashAttention& out) noexcept;
// Accepts "none", "layer", and "row" (case-insensitive); leaves `out` untouched on failure.
[[nodiscard]] bool ParseSplitMode(std::string_view name, SplitMode& out) noexcept;
// Accepts "none", "auto", "ngram", "ngram-mod", "suffix", and "draft-mtp" (case-insensitive).
[[nodiscard]] bool ParseSpeculativeType(std::string_view name, SpeculativeType& out) noexcept;
// Accepts "target_match"/"target-match" and "stochastic_rejection"/"stochastic-rejection".
[[nodiscard]] bool ParseSpeculativeVerificationMode(std::string_view name,
                                                    SpeculativeVerificationMode& out) noexcept;
// True when the built-in proposer for `type` produces real stochastic proposal
// probability distributions `q` (false for deterministic suffix/n-gram and greedy MTP).
[[nodiscard]] bool ProposerSuppliesStochasticProbabilities(SpeculativeType type) noexcept;
// Resolves the effective verification mode for a loaded speculative proposer.
[[nodiscard]] SpeculativeVerificationMode ResolveSpeculativeVerificationMode(
    SpeculativeVerificationMode requested, SpeculativeType active_spec,
    bool proposer_has_probabilities = false) noexcept;
// True for block-quantized types (everything except f32, f16 and bf16).
[[nodiscard]] bool IsQuantized(KvCacheType type) noexcept;
// True if this llama.cpp/ggml build defines the corresponding GGML_TYPE_*.
[[nodiscard]] bool IsKvCacheTypeSupported(KvCacheType type) noexcept;

struct LoadOptions {
    std::string model_path;       // GGUF file
    std::uint32_t n_ctx = 2048;   // 0 = model training context
    std::uint32_t n_batch = 512;  // max tokens per llama_decode during prefill
    // Physical micro-batch (llama.cpp n_ubatch): tokens per compute graph
    // inside one llama_decode. 0 = llama.cpp's default (512) capped at
    // n_batch; otherwise it must be <= n_batch.
    std::uint32_t n_ubatch = 0;
    std::int32_t n_threads = 0;   // 0 = llama.cpp default
    std::int32_t n_gpu_layers = 0;  // CPU-only by default; -1 = offload all
    bool vocab_only = false;      // tokenizer metadata only, no weights
    bool use_mmap = true;
    std::vector<TensorOverride> tensor_overrides;  // applied in order
    // KV cache element types and Flash Attention; the defaults are llama.cpp's
    // defaults. A quantized V cache needs Flash Attention, so kDisabled with a
    // quantized type_v is rejected (kAuto lets llama.cpp switch it on).
    KvCacheType type_k = KvCacheType::kF16;
    KvCacheType type_v = KvCacheType::kF16;
    FlashAttention flash_attn = FlashAttention::kAuto;
    // Explicit ggml device names (e.g. "CUDA0", "ROCm0", "Vulkan0", "CUDA1")
    // passed as a null-terminated llama_model_params::devices list. Empty uses
    // llama.cpp's default device selection. Validated upfront before loading.
    std::vector<std::string> devices;
    // Primary GPU index used for the full model when split_mode == kNone and
    // for scratch / non-split tensors. Must be >= 0.
    std::int32_t main_gpu = 0;
    // Multi-GPU split strategy; kLayer matches llama.cpp's default.
    SplitMode split_mode = SplitMode::kLayer;
    // Relative proportions of layers/rows across GPUs (empty = llama.cpp
    // default). When non-empty: at most kMaxTensorSplitDevices (16) finite,
    // non-negative values with at least one strictly positive entry.
    std::vector<float> tensor_split;
    // Lock model weights in RAM. Mapped via llama.cpp b11195's load_mode:
    // use_mmap + use_mlock -> LLAMA_LOAD_MODE_MMAP_MLOCK; !use_mmap +
    // use_mlock -> LLAMA_LOAD_MODE_MLOCK; !use_mmap && !use_mlock ->
    // LLAMA_LOAD_MODE_NONE.
    bool use_mlock = false;
    // Use a unified KV buffer across slots/sequences when supported by llama.cpp.
    bool kv_unified = true;
    // Reuse longest matching prompt KV prefix across consecutive turns.
    bool prefix_cache = true;
    // Coalesce simultaneous cold shared-prefix prefills on attention-only models
    // (when n_slots > 1 and prefix_cache is enabled). Explicitly bypassed on
    // hybrid/recurrent models without a valid checkpoint.
    bool prefix_coalesce = true;
    std::uint32_t min_coalesce_prefix_tokens = kDefaultMinCoalescePrefixTokens;
    std::uint32_t max_followers_per_producer = kDefaultMaxCoalesceFollowers;
    std::uint32_t max_coalesce_wait_ms = kDefaultMaxCoalesceWaitMs;
    // Number of cached KV sequence slots (1..kMaxSlots).
    std::uint32_t n_slots = 1;
    // In-process speculative decoding strategy: "none", "auto", "ngram",
    // "ngram-mod", "suffix", or "draft-mtp".
    std::string spec_type = "none";
    // Max draft tokens per speculative step (0 = automatic default for spec_type).
    std::int32_t spec_draft_max = 0;
    // Verification policy for speculative decoding (TargetMatch default vs
    // opt-in StochasticRejection when a stochastic proposer supplies real q).
    SpeculativeVerificationMode spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
    // Bounded L2 host-RAM prefix state cache budget in MiB (0 = disabled, max kMaxCacheRamMiB).
    std::uint32_t cache_ram_mib = 0;
    // Dynamic GPU cache in MiB for host-resident MoE experts (0 = disabled, max kMaxMoeCacheMiB).
    std::uint32_t moe_cache_mib = 0;
    // Unused/rejected by the direct llamacpp backend (no execution consumer):
    // must remain 0.0; non-zero or non-finite values fail with invalid_argument.
    double moe_cache_cpu_share = 0.0;
    // Optional multimodal projector GGUF path (--mmproj); requires SONDER_WITH_MTMD=ON.
    std::string mmproj_path;
    // Sliding-window attention KV policy: Full (default, swa_full = true) or
    // CompactIfQualified (opt-in compact SWA with min_retained_pos rollback guard).
    sonder::inference::SwaKvMode swa_kv_mode = sonder::inference::SwaKvMode::Full;
    // Default inter-token latency (ITL) target in milliseconds applied to
    // interactive/foreground decoding slots when a request does not specify an
    // explicit `target_itl_ms` (0 disables the implicit foreground ITL cap).
    double default_foreground_itl_ms = 50.0;
};

// Checks placement options (main_gpu >= 0, split_mode, tensor_split finite /
// non-negative / at least one positive / max 16, non-empty device names).
// Pure: no file access and no llama.cpp calls. Load() runs this before
// touching the model file.
[[nodiscard]] Status ValidatePlacementOptions(const LoadOptions& options);

// Checks the context options above (micro-batch vs batch, quantized V cache
// vs Flash Attention). Pure: no file access and no llama.cpp calls. Load()
// runs it before touching the model file. Always Ok for vocab_only loads,
// which create no context.
[[nodiscard]] Status ValidateContextOptions(const LoadOptions& options);

// The llama_model_params values a Load() with given options passes to
// llama.cpp, read back from the real parameter struct. For tests and
// diagnostics.
struct ModelParamsSummary {
    std::int32_t n_gpu_layers = 0;
    std::int32_t main_gpu = 0;
    std::string split_mode;
    std::string load_mode;
    bool vocab_only = false;
    bool has_tensor_split = false;
    bool has_devices = false;
    bool load_mtp = false;
};

// The llama_context_params values a Load() with given options passes to
// llama.cpp, read back from the real parameter struct. Names come from
// llama.cpp itself (ggml_type_name; llama_flash_attn_type_name gives "auto",
// "enabled" or "disabled"). For tests and diagnostics.
struct ContextParamsSummary {
    std::uint32_t n_ctx = 0;
    std::uint32_t n_batch = 0;
    std::uint32_t n_ubatch = 0;
    std::uint32_t n_seq_max = 1;
    std::string type_k;
    std::string type_v;
    std::string flash_attn;
    bool offload_kqv = true;
    bool no_perf = true;
    bool kv_unified = false;
    bool op_offload = false;
    bool swa_full = true;
    std::uint64_t moe_cache_size = 0;
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
    std::string grammar;               // optional GBNF grammar constraining output (root rule)
    bool compute_logprobs = false;     // compute exact log-softmax logprob for each emitted token
    std::int32_t top_logprobs = 0;     // number of top alternative logprobs (0..kMaxTopLogprobs)
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

// Maximum serialized byte size accepted by CaptureSequenceState / RestoreSequenceState.
inline constexpr std::size_t kMaxSequenceStateBytes = std::size_t{512} << 20;

struct GenerateRequest {
    std::vector<Token> prompt;     // already tokenized; must be non-empty
    std::int32_t max_new_tokens = 128;
    SamplingParams sampling{};
    bool reset_context = true;     // clear KV memory before prefill (ignored when prefix_cache reuses a slot)
    // Optional conversation/slot affinity key for multi-slot KV prefix caching.
    std::string session_key;
    // Optional atomic sequence-state restore before suffix prefill. When
    // non-empty and 0 < restore_tokens < prompt.size(), Generate() clears
    // memory, restores sequence 0 state, and prefills only prompt[restore_tokens..).
    std::span<const std::uint8_t> restore_state{};
    std::size_t restore_tokens = 0;
    std::function<void()> on_state_restored;
    // Optional atomic prefill-boundary capture: when > n_past and <= prompt.size(),
    // Generate() splits prefill at `capture_at_tokens`, captures sequence 0 state
    // before decoding tail prompt or completion tokens, and invokes `on_state_captured`.
    std::size_t capture_at_tokens = 0;
    std::size_t max_capture_bytes = kMaxSequenceStateBytes;
    std::function<void(std::vector<std::uint8_t>)> on_state_captured;
    // Optional multimodal media blobs (e.g. PNG/JPEG bytes, max kMaxMediaImages,
    // max kMaxMediaBlobBytes each). Requires mmproj_path loaded with libmtmd.
    std::vector<std::vector<std::uint8_t>> media_blobs;
    // Optional prompt text with media markers when media_blobs is non-empty.
    // When empty, `prompt` is detokenized and missing markers are prepended.
    std::string media_prompt_text;
    // Preferred G0028-safe structured multimodal segments preserving per-part
    // `parse_special` boundaries and explicit media bitmap indices (no marker
    // string parsing on untrusted user text).
    struct MediaPromptPart {
        std::string text;
        bool parse_special = false;
        std::optional<std::size_t> media_index = std::nullopt;
    };
    std::vector<MediaPromptPart> media_prompt_parts;
    // Scheduling priority class and optional latency targets (ms) for the
    // latency-aware continuous-batching physical planner.
    sonder::inference::RequestPriority priority = sonder::inference::RequestPriority::subagent;
    std::optional<double> target_ttft_ms = std::nullopt;
    std::optional<double> target_itl_ms = std::nullopt;
};

// One event per generated token. After the last token a final event with
// token == -1 may carry leftover bytes of an incomplete UTF-8 sequence.
struct TokenEvent {
    Token token = 0;
    std::string text;         // UTF-8 complete text released by this token (may be empty)
    std::int32_t index = 0;   // 0-based index among generated tokens
    bool has_logprob = false;
    float logprob = 0.0F;
    std::vector<std::pair<Token, float>> top_logprobs{};
};

// Return false to stop generation (treated as cancellation).
using TokenCallback = std::function<bool(const TokenEvent&)>;
// Polled between tokens, between prefill batches, and from llama.cpp's abort
// callback inside llama_decode; return true to cancel. Must be thread-safe.
using CancelPredicate = std::function<bool()>;

struct GenerateResult {
    Status status{};
    StopReason stop_reason = StopReason::kNone;
    std::int32_t prompt_tokens = 0;          // newly evaluated prompt tokens
    std::int32_t cached_tokens = 0;          // prompt tokens reused from KV prefix cache
    std::int32_t generated_tokens = 0;
    std::int32_t draft_tokens = 0;           // speculative draft tokens proposed
    std::int32_t draft_accepted_tokens = 0;  // speculative draft tokens accepted
    SpeculativeVerificationMode spec_verifier_mode = SpeculativeVerificationMode::TargetMatch;
    std::uint64_t spec_residual_resamples = 0;
    double spec_verify_overhead_us = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    bool state_restored = false;
    Status restore_status{};
    bool state_captured = false;
    Status capture_status{};

    [[nodiscard]] double decode_tokens_per_second() const noexcept;
    [[nodiscard]] double prefill_tokens_per_second() const noexcept;
    [[nodiscard]] double spec_accept_rate() const noexcept;
};

struct EmbedOutput {
    std::vector<std::vector<float>> embeddings;
    std::int32_t prompt_tokens = 0;
    double eval_ms = 0.0;
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
    sonder::inference::ModelArchitecture architecture =
        sonder::inference::ModelArchitecture::attention_only;
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

// Telemetry counters for cold shared-prefix prefill coalescing.
struct PrefixCoalescerStats {
    std::uint64_t coalesced_groups = 0;
    std::uint64_t coalesced_followers = 0;
    std::uint64_t coalesced_tokens_saved = 0;
    std::uint64_t coalesce_fallbacks_timeout = 0;
    std::uint64_t coalesce_fallbacks_producer_failed = 0;
};

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
    bool prefix_reuse = true;
    bool speculative_decode = true;
    bool structured_output = true;
    bool embeddings = true;
    bool recurrent_checkpoint = true;
    bool continuous_batch_decode = true;
    bool logprobs = true;
};

// Severity of a llama.cpp/ggml log line. The values mirror ggml_log_level.
enum class LogLevel : int { kNone = 0, kDebug = 1, kInfo = 2, kWarn = 3, kError = 4, kCont = 5 };

// Destination of the llama.cpp log lines that pass the filter (warnings and
// errors; everything with SONDER_LLAMACPP_VERBOSE=1). `text` is a complete
// piece of text including its newline, valid for the call only. The sink is
// called from whatever thread llama.cpp logs on, so it must not block (queue
// the text and return) and must not throw. A null sink (the default) writes
// to stderr synchronously. Any thread.
using LogSink = void (*)(const char* text);
void SetLogSink(LogSink sink) noexcept;

// What llama.cpp's log callback does with one line: filters by `level`, then
// hands the text to the sink. Public so tests can drive it without a model.
void RouteLog(LogLevel level, const char* text);

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
    // Checks each ggml device name (non-empty, registered in this build via
    // ggml_backend_dev_by_name, at most llama_max_devices()). On success,
    // `resolved` (if non-null) receives one opaque ggml_backend_dev_t per
    // device, in order (not null-terminated). Needs no model file.
    [[nodiscard]] static Status ResolveDevices(const std::vector<std::string>& device_names,
                                               std::vector<void*>* resolved = nullptr);
    // Checks each override (non-empty valid regex, device present in this
    // build). On success, `buffer_types` (if non-null) receives one opaque
    // ggml buffer type per override, in order. Needs no model file.
    [[nodiscard]] static Status ResolveTensorOverrides(const std::vector<TensorOverride>& overrides,
                                                       std::vector<void*>* buffer_types = nullptr);
    [[nodiscard]] static std::string SystemInfo();
    // Model parameters Load() would use for `options`, and llama.cpp's own
    // llama_model_default_params(). Need no model file.
    [[nodiscard]] static ModelParamsSummary ModelParamsFor(const LoadOptions& options);
    [[nodiscard]] static ModelParamsSummary UpstreamModelDefaults();
    // Context parameters Load() would use for `options`, and llama.cpp's own
    // llama_context_default_params(). Need no model file.
    [[nodiscard]] static ContextParamsSummary ContextParamsFor(const LoadOptions& options);
    [[nodiscard]] static ContextParamsSummary UpstreamContextDefaults();

    Status Load(const LoadOptions& options);
    void Unload() noexcept;
    [[nodiscard]] bool IsLoaded() const noexcept;
    [[nodiscard]] bool HasMmproj() const noexcept;
    [[nodiscard]] std::uint32_t SlotCount() const noexcept;
    [[nodiscard]] SpeculativeVerificationMode RequestedSpecVerificationMode() const noexcept;
    [[nodiscard]] SpeculativeVerificationMode EffectiveSpecVerificationMode() const noexcept;
    [[nodiscard]] ModelInfo GetModelInfo() const;

    // parse_special: treat control-token text (e.g. "<|im_start|>") as the
    // special token rather than plain text; needed for chat-templated prompts.
    Status Tokenize(std::string_view text, bool add_special, std::vector<Token>& out,
                    bool parse_special = false) const;
    Status TokenToPiece(Token token, std::string& out) const;
    Status Detokenize(std::span<const Token> tokens, std::string& out, bool remove_special = false,
                      bool unparse_special = false) const;
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

    // Computes dense embeddings for one or more tokenized inputs using an
    // isolated lazy embedding context so live KV prefix caches stay intact.
    Status Embed(const std::vector<std::vector<Token>>& inputs, bool normalize, EmbedOutput& out,
                 const CancelPredicate& should_cancel = {});

    // Captures sequence 0 state (KV + recurrent/SSM state) via llama_state_seq_get_*.
    Status CaptureSequenceState(std::vector<std::uint8_t>& out_bytes,
                                std::size_t max_bytes = kMaxSequenceStateBytes);
    // Restores sequence 0 state via llama_state_seq_set_data.
    Status RestoreSequenceState(std::span<const std::uint8_t> bytes,
                                std::size_t max_bytes = kMaxSequenceStateBytes);

    void SetTelemetrySink(TelemetrySink sink);
    [[nodiscard]] PrefixCoalescerStats GetPrefixCoalescerStats() const noexcept;

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

// Length of the longest common prefix between two token spans.
[[nodiscard]] std::size_t LongestCommonPrefix(std::span<const Token> a, std::span<const Token> b) noexcept;

// Per-token exact log-softmax probability and top-K alternatives.
struct TokenLogprobInfo {
    bool valid = false;
    float logprob = 0.0F;
    std::vector<std::pair<Token, float>> top_logprobs;
};

// Computes exact log-softmax logprob for `chosen_token` and up to `top_k`
// highest-probability (token, logprob) pairs (descending logprob, tie-broken
// by smaller token id) using numerically stable max-subtraction.
[[nodiscard]] TokenLogprobInfo ComputeTokenLogprobs(std::span<const float> logits, Token chosen_token,
                                                    std::int32_t top_k);

// Validates multimodal media payloads against kMaxMediaImages (8) and
// kMaxMediaBlobBytes (16 MiB per image), rejecting empty blobs.
[[nodiscard]] Status ValidateMediaBlobs(const std::vector<std::vector<std::uint8_t>>& media_blobs);

// Per-slot prefix cache metadata used for slot selection.
struct PrefixSlotMeta {
    std::int32_t id = 0;
    std::vector<Token> tokens;
    std::string session_key;
    std::uint64_t last_used = 0;
};

struct SlotSelection {
    std::size_t slot_index = 0;
    std::size_t matched_prefix = 0;  // clamped to min(lcp, prompt.size() - 1)
};

// Selects a KV slot for `prompt` and optional `session_key`:
//   1. Exact non-empty `session_key` match
//   2. Longest common token prefix (> 0), tie-broken by most recently used
//   3. Least recently used slot (lowest `last_used`)
// `matched_prefix` is clamped to `prompt.size() - 1` (0 when prompt has <= 1
// token) so at least one prompt token is always decoded for fresh logits.
[[nodiscard]] SlotSelection SelectPrefixSlot(std::span<const PrefixSlotMeta> slots, std::span<const Token> prompt,
                                             std::string_view session_key) noexcept;

// Fast, allocation-light rolling 4-gram / 3-gram self-speculation matcher.
// Stores indices into the caller's token history in flat fixed-size hash tables
// (no per-token heap allocations).
class NgramMatcher {
public:
    explicit NgramMatcher(bool exact_scan_fallback = false);

    void Reset();
    // Indexes history positions [from_index, to_index). Callers must index only
    // positions strictly before the current tail position used for Draft().
    void IndexRange(std::span<const Token> history, std::size_t from_index, std::size_t to_index);
    // Appends up to `max_draft` candidate tokens into `out` (cleared first)
    // matching the 4-gram (or 3-gram fallback) ending at `history.back()`.
    void Draft(std::span<const Token> history, std::size_t max_draft, std::vector<Token>& out) const;

private:
    static constexpr std::size_t kTableBits = 14;
    static constexpr std::size_t kTableSize = std::size_t{1} << kTableBits;
    static constexpr std::size_t kTableMask = kTableSize - 1;
    static constexpr std::size_t kFallbackScanLimit = 2048;

    bool exact_scan_fallback_ = false;
    std::vector<std::int32_t> pos4_;
    std::vector<std::int32_t> pos3_;
};

// Persistent O(1) modular-hash n-gram table for `ngram-mod` self-speculation
// (matches PrismML / llama.cpp common_ngram_mod: 4M flat entries = 16 MiB RAM,
// n_match = 24, n_min = 48, n_max = 64, cross-request persistence with
// occupancy and low-acceptance streak reset, plus adaptive draft length).
class NgramModTable {
public:
    static constexpr Token kEmpty = -1;
    static constexpr std::size_t kDefaultTableSize = std::size_t{4} << 20;  // 4M entries (16 MiB)
    static constexpr std::size_t kDefaultMatchLen = 24;
    static constexpr std::size_t kDefaultMinDraft = 48;
    static constexpr std::size_t kDefaultMaxDraft = static_cast<std::size_t>(kDefaultNgramModDraftMax);

    explicit NgramModTable(std::size_t n_match = kDefaultMatchLen, std::size_t table_size = kDefaultTableSize,
                           std::size_t n_min = kDefaultMinDraft, std::size_t n_max = kDefaultMaxDraft);

    void Configure(std::size_t n_min, std::size_t n_max);
    void Reset();

    // Indexes prompt tokens into the persistent table and resets per-request
    // adaptive state. Resets the table only if occupancy exceeds 25%.
    void BeginRequest(std::span<const Token> prompt);
    // Indexes all n-gram windows whose start position lies in [from_index, to_index)
    // and whose next-token target lies inside `history`. Returns the next start index.
    std::size_t IndexRange(std::span<const Token> history, std::size_t from_index, std::size_t to_index);
    // Chains up to `min(max_draft, n_max)` tokens starting from the trailing
    // `n_match - 1` tokens of `history` plus `id_last`. Returns empty if fewer
    // than `min(n_min, max_draft)` tokens chain from the table.
    void Draft(std::span<const Token> history, Token id_last, std::size_t max_draft, std::vector<Token>& out);
    // Updates adaptive draft length and low-acceptance streak guard.
    // Returns true if the table was reset due to repeated low acceptance.
    bool Observe(std::size_t drafted, std::size_t accepted);

    [[nodiscard]] std::size_t n_match() const noexcept { return n_match_; }
    [[nodiscard]] std::size_t n_min() const noexcept { return n_min_; }
    [[nodiscard]] std::size_t n_max() const noexcept { return n_max_; }
    [[nodiscard]] std::size_t current_draft_cap() const noexcept { return n_cur_; }
    [[nodiscard]] std::size_t used() const noexcept { return used_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    [[nodiscard]] std::size_t IndexOf(const Token* tokens) const noexcept;
    void Add(const Token* tokens) noexcept;
    [[nodiscard]] Token Get(const Token* tokens) const noexcept;

    std::size_t n_match_ = kDefaultMatchLen;
    std::size_t n_min_ = kDefaultMinDraft;
    std::size_t n_max_ = kDefaultMaxDraft;
    std::size_t n_cur_ = kDefaultMaxDraft;
    std::size_t used_ = 0;
    std::size_t table_mask_ = 0;
    int low_streak_ = 0;
    std::vector<Token> entries_;
    std::vector<Token> scratch_;
};

// Adaptive speculative draft-length controller for n-gram self-speculation.
// Ramps up toward `max_draft` on high acceptance (>= 50%), shrinks on moderate
// acceptance, and pauses briefly (with periodic small probes) when rolling
// acceptance falls below 20% so unpredictable prose incurs near-zero overhead.
class AdaptiveDraftController {
public:
    explicit AdaptiveDraftController(std::int32_t max_draft = kDefaultNgramDraftMax);

    void Reset(std::int32_t max_draft);
    // Number of draft tokens to propose on the next step (0 while paused).
    [[nodiscard]] std::int32_t NextDraftLength() const noexcept;
    // Advances one non-speculative step while paused (counts down cooldown).
    void NoteSkippedStep() noexcept;
    // Records outcome of a speculative verification step (drafted > 0).
    void Observe(std::int32_t drafted, std::int32_t accepted) noexcept;

    [[nodiscard]] double acceptance_ema() const noexcept { return acceptance_ema_; }
    [[nodiscard]] std::int32_t max_draft() const noexcept { return max_draft_; }
    [[nodiscard]] bool paused() const noexcept { return cooldown_remaining_ > 0; }

private:
    std::int32_t max_draft_ = kDefaultNgramDraftMax;
    std::int32_t current_draft_ = kDefaultNgramDraftMax;
    std::int32_t cooldown_remaining_ = 0;
    std::int32_t cooldown_period_ = 8;
    std::int32_t steps_observed_ = 0;
    std::int32_t total_drafted_ = 0;
    double acceptance_ema_ = 0.5;
};

}  // namespace sonder::backends::llamacpp
