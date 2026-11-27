// Sonder Inference: execution backend interface.
//
// Backends own tensor execution; Sonder owns policy (sessions, scheduling,
// cache, lifecycle, telemetry). The interface is deliberately not shaped
// around a single backend (see docs/BACKENDS.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/model_architecture.hpp"
#include "sonder/inference/request_priority.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference {

// Capability advertisement (docs/BACKENDS.md). Sonder never assumes a
// capability that a backend does not declare.
enum class Capability : std::uint32_t {
    tokenization = 1u << 0,
    streaming = 1u << 1,
    batched_prefill = 1u << 2,
    continuous_batch_decode = 1u << 3,
    kv_export = 1u << 4,
    kv_import = 1u << 5,
    kv_copy = 1u << 6,
    kv_quantization = 1u << 7,
    prefix_reuse = 1u << 8,
    speculative_decode = 1u << 9,
    lora = 1u << 10,
    layer_telemetry = 1u << 11,
    structured_output = 1u << 12,
    embeddings = 1u << 13,
    deterministic = 1u << 14,  // identical inputs + seed => identical output
    remote_process = 1u << 15, // execution happens in another process/server
    token_logits = 1u << 16,   // exposes per-step logits (open_token_stream); Sonder samples
    recurrent_checkpoint = 1u << 17,  // supports recurrent/hybrid sequence state capture & restore
    tool_calling = 1u << 18,   // supports native function/tool calling
    multimodal = 1u << 19,     // supports multimodal media inputs (validated base64 images)
    logprobs = 1u << 20,       // supports per-token logprobs / top_logprobs
    reranking = 1u << 21,      // supports native cross-encoder reranking and pair scoring
    execution_checkpoint = 1u << 22,  // supports same-process execution-state capture & restore
    selective_preemption = 1u << 23,  // supports non-blocking per-request safe-point suspension & slot reclamation
};

// Maximum serialized byte size permitted for a single recurrent-state snapshot.
inline constexpr std::size_t kMaxRecurrentCheckpointBytes = std::size_t{512} << 20;

// Opaque serialized recurrent/SSM sequence state captured at an exact token position.
struct RecurrentStateSnapshot {
    std::uint64_t token_position = 0;
    std::vector<std::uint8_t> bytes;
};

// Optional atomic checkpoint exchange attached to GenerateRequest for
// hybrid/recurrent models. When provided, a capable backend executes restore
// and prefill-boundary capture under the same execution lock as prefill/decode
// so concurrent requests cannot clobber sequence state and captured bytes match
// the exact prefill boundary (never post-decode state).
struct RecurrentCheckpointExchange {
    // Input: checkpoint bytes to restore before prefilling the suffix.
    std::span<const std::uint8_t> restore_bytes{};
    // Input: exact token position represented by `restore_bytes` (0 = none).
    std::uint64_t restore_token_position = 0;
    // Input: optional callback invoked synchronously when restore succeeds
    // (before decode begins).
    std::function<void()> on_restored;
    // Output: true iff `restore_bytes` was applied before suffix prefill.
    bool restored = false;
    // Output: populated if restore was attempted and failed.
    std::optional<Status> restore_error = std::nullopt;

    // Input: exact prefill token position at which to capture a checkpoint
    // before decoding any tail prompt or completion tokens (0 = skip capture).
    std::uint64_t capture_at_token_position = 0;
    // Input: optional callback invoked synchronously at `capture_at_token_position`
    // with the captured snapshot bytes.
    std::function<void(std::uint64_t token_position, std::vector<std::uint8_t> bytes)> on_captured;
    // Output: true iff capture succeeded.
    bool captured = false;
    // Output: populated if capture was requested and failed.
    std::optional<Status> capture_error = std::nullopt;
    // Output: captured snapshot bytes when `on_captured` is empty.
    std::vector<std::uint8_t> captured_bytes;
};

// Vocabulary index as produced by a backend tokenizer.
using TokenId = std::int32_t;

// Largest number of alternatives a telemetry probability probe asks for per
// sampled token (GenerateRequest/ChatRequest::telemetry_top_k).
inline constexpr std::int32_t kMaxTelemetryTopK = 10;

// Non-blocking action returned at backend-owned safe points (between prefill
// chunks or after a decoded token) so a shared batch worker can park and
// reclaim ONE request's physical slot/state without stalling peer requests.
enum class SuspendAction : std::uint8_t {
    continue_running = 0,  // keep executing normally
    yield_recompute = 1,   // reclaim physical slot/KV; caller may requeue & recompute
    yield_spill = 2,       // spill/capture physical state blob, then reclaim physical slot/KV
};

const char* to_string(SuspendAction action) noexcept;

// Snapshot of execution progress passed by a backend at a safe point.
struct SafePointContext {
    std::uint64_t prompt_tokens = 0;
    std::uint64_t prefill_completed_tokens = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t kv_tokens = 0;
    std::size_t active_peers = 0;
    bool in_prefill = false;
};

// Non-blocking decision returned by a safe-point callback.
struct SuspendDecision {
    SuspendAction action = SuspendAction::continue_running;
    const char* reason = nullptr;  // optional static/diagnostic label for telemetry
    std::size_t max_spill_bytes = kMaxRecurrentCheckpointBytes;
};

// Must NEVER block or sleep; invoked inline on the backend execution thread.
using SuspendSafePointCallback = std::function<SuspendDecision(const SafePointContext&)>;

// Physical state returned when a backend parks one request at a safe point
// (StopReason::suspended). Strictly same-process / same-build only.
struct SuspendedExecutionState {
    SuspendAction action = SuspendAction::yield_recompute;
    std::uint64_t token_position = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t backend_step_index = 0;
    std::vector<TokenId> tokens;
    std::vector<TokenId> pending_draft_tokens;
    std::vector<std::uint8_t> backend_state_blob;
    std::string stop_filter_pending;
    bool stop_filter_stopped = false;
    bool physical_slot_reclaimed = false;
};

// Process-local execution resume payload attached to GenerateRequest for
// ExactResume continuation. Never portable across processes, builds, or models.
struct ExecutionResumeState {
    std::uint64_t token_position = 0;
    std::uint64_t generated_tokens = 0;
    std::uint64_t backend_step_index = 0;
    std::vector<TokenId> generated_token_ids;
    std::vector<TokenId> pending_draft_tokens;
    std::span<const std::uint8_t> backend_state_blob{};
    std::string stop_filter_pending;
    bool stop_filter_stopped = false;
};

// Optional stateful grammar/constraint hooks for ExactResume snapshot/restore
// and Branch reset (pure core types, independent of any grammar backend).
struct GrammarMatcherHooks {
    std::function<void(std::span<float> logits)> mask_logits;
    std::function<void(TokenId token)> accept_token;
    std::function<std::vector<std::uint8_t>()> snapshot_state;
    std::function<bool(std::span<const std::uint8_t>)> restore_state;
    std::function<void()> reset;
};

struct BackendCapabilities {
    std::uint32_t bits = 0;
    [[nodiscard]] bool has(Capability c) const noexcept { return (bits & static_cast<std::uint32_t>(c)) != 0; }
    BackendCapabilities& add(Capability c) noexcept {
        bits |= static_cast<std::uint32_t>(c);
        return *this;
    }
    [[nodiscard]] std::vector<std::string> names() const;
};

struct ModelDescriptor {
    std::string name;
    std::string backend;
    std::string format;        // e.g. "gguf", "mock"
    std::string family;        // architecture family when known
    std::string parameter_size;
    std::string quantization;  // e.g. "Q4_K_M"
    std::uint64_t size_bytes = 0;
    std::uint64_t context_length = 0;  // 0 when unknown
    // False when load_model() only fetched metadata and the weights are not
    // resident (e.g. Ollama loads lazily on first request).
    bool resident = true;
    ModelArchitecture architecture = ModelArchitecture::attention_only;
};

struct ModelLoadOptions {
    std::string model;
    std::string device_id = "cpu:0";
};

struct GenerateRequest {
    std::string request_id;
    std::string prompt;
    SamplingConfig sampling;
    // Optional conversation/slot affinity key for backends with prefix caching.
    std::string session_key{};
    // Optional GBNF grammar constraining output (empty = unconstrained).
    std::string grammar{};
    // Optional non-owning atomic checkpoint exchange for hybrid/recurrent models.
    // Must outlive the generate() / open_token_stream() call when non-null.
    RecurrentCheckpointExchange* checkpoint = nullptr;
    // Optional scheduling priority class and latency targets (ms) forwarded to
    // backends with latency-aware physical batch planning.
    std::optional<RequestPriority> priority_class = std::nullopt;
    std::optional<double> target_ttft_ms = std::nullopt;
    std::optional<double> target_itl_ms = std::nullopt;
    // Opt-in per-request stage timing hooks (false by default: zero extra clocks).
    bool enable_stage_profiling = false;
    // Optional non-blocking safe-point suspension hook (Item #9). Must not block.
    SuspendSafePointCallback on_safe_point{};
    // Optional non-owning output populated when StopReason::suspended occurs.
    SuspendedExecutionState* suspended_state = nullptr;
    // Optional non-owning ExactResume state payload (same-process/same-model only).
    const ExecutionResumeState* execution_resume = nullptr;
    // Scheduler priority rank (lower = more urgent) for physical victim choice.
    int priority_rank = 0;
    // Allow cost-aware spill-to-checkpoint when selectively preempted under pressure.
    bool allow_preempt_spill = false;
    // Telemetry-only probability probe, [0, kMaxTelemetryTopK]; 0 = off. Set
    // by the session when token telemetry is recorded. A backend that samples
    // upstream and can report probabilities (llamaserver against llama.cpp)
    // asks for the sampled token's probability and this many alternatives and
    // reports them in TokenChunk::sampled_probability. The caller-visible
    // TokenChunk::logprob / top_logprobs stay exactly as sampling.logprobs
    // asked. Other backends ignore it.
    std::int32_t telemetry_top_k = 0;
};

// Bounded batch limits for embedding, reranking, and pair-scoring requests.
inline constexpr std::size_t kMaxEmbedInputs = 256u;
inline constexpr std::size_t kMaxRerankDocuments = 256u;
inline constexpr std::size_t kMaxScorePairs = 256u;

enum class EmbedPoolingType { unspecified, none, mean, cls, last, rank };
const char* to_string(EmbedPoolingType pooling) noexcept;

struct EmbedRequest {
    std::string request_id;
    // Text inputs. Exactly one of `inputs` / `token_inputs` is non-empty.
    std::vector<std::string> inputs;
    // Pre-tokenized inputs (OpenAI `input` as token arrays). The ids are in the
    // loaded model's own vocabulary, so only backends that tokenize natively
    // can serve them; others fail the request.
    std::vector<std::vector<TokenId>> token_inputs;
    bool normalize = true;
    // Optional requested output dimension. Backends must fail closed if
    // unsupported or > native dimension; never silently truncate arbitrary models.
    std::optional<std::size_t> dimensions = std::nullopt;
    EmbedPoolingType pooling_type = EmbedPoolingType::unspecified;
    std::string encoding_format = "float";
};

struct EmbedResult {
    std::vector<std::vector<float>> embeddings;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t eval_ns = 0;
    std::size_t dimensions = 0;
    EmbedPoolingType pooling_type = EmbedPoolingType::unspecified;
    std::string encoding_format = "float";
    std::string model{};
};

struct RerankRequest {
    std::string request_id;
    std::string query;
    std::vector<std::string> documents;
    // Optional limit on returned top-ranked results (nullopt = return all documents).
    std::optional<std::size_t> top_n = std::nullopt;
    bool return_documents = false;
};

struct RerankItem {
    std::size_t index = 0;  // 0-based index into original RerankRequest::documents
    double relevance_score = 0.0;
    std::optional<std::string> document = std::nullopt;
};

struct RerankResult {
    std::vector<RerankItem> results;  // sorted descending by relevance_score (tie-broken by index asc)
    std::uint64_t prompt_tokens = 0;
    std::uint64_t eval_ns = 0;
    std::string model{};
};

struct ScorePair {
    std::string text_1;
    std::string text_2;
};

struct ScoreRequest {
    std::string request_id;
    std::vector<ScorePair> pairs;
};

struct ScoreItem {
    std::size_t index = 0;  // preserves exact input pair order 0..pairs.size()-1
    double score = 0.0;
};

struct ScoreResult {
    std::vector<ScoreItem> scores;  // exact original input pair order (never re-sorted)
    std::uint64_t prompt_tokens = 0;
    std::uint64_t eval_ns = 0;
    std::string model{};
};

Status validate_rerank_request(const RerankRequest& request);
Status validate_score_request(const ScoreRequest& request);
// Sorts items descending by relevance_score (NaN/non-finite last, tie-broken by
// ascending original index) and applies optional top_n truncation.
void finalize_rerank_items(std::vector<RerankItem>& items, std::optional<std::size_t> top_n);

struct TopLogprob {
    std::string token;
    float logprob = 0.0f;
};

struct TokenLogprob {
    std::string token;
    float logprob = 0.0f;
    std::vector<TopLogprob> top_logprobs{};
};

// Tool/function definition advertised to a model.
struct ToolDefinition {
    std::string name;
    std::string description;
    // Serialized JSON Schema object for function parameters (empty = unconstrained object).
    std::string parameters_json;
    // Optional strict schema enforcement flag (true fails closed if schema cannot be enforced).
    std::optional<bool> strict = std::nullopt;
};

enum class ToolChoiceMode { auto_choice, none, required, function };

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::auto_choice;
    // Populated when mode == ToolChoiceMode::function.
    std::string function_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Incremental streaming delta for a tool call (OpenAI delta.tool_calls item).
struct ToolCallDelta {
    std::uint32_t index = 0;
    std::string id;
    std::string name;
    std::string arguments_delta;
};

// Validated in-memory binary media attachment (decoded from data: base64 URI).
struct MediaContent {
    std::string mime_type;
    std::vector<std::uint8_t> bytes;
};

// One alternative considered for a backend-sampled token (telemetry only).
struct TokenCandidate {
    TokenId token_id = 0;
    std::string token;         // the token's text as the backend reported it
    float probability = 0.0f;  // [0, 1]
};

// Probability of the token a backend sampled itself, for telemetry
// (inference.token.generated / inference.sampling.candidates). Not part of
// the caller's logprobs.
struct SampledTokenProbability {
    TokenId token_id = 0;
    float probability = 0.0f;  // [0, 1]
    // True: after the backend's sampler chain (llama-server
    // post_sampling_probs). False: softmax of the raw logits (logprobs).
    bool post_sampling = false;
    // Most probable first, at most the requested telemetry_top_k entries.
    std::vector<TokenCandidate> candidates{};
};

struct TokenChunk {
    std::string_view text;
    std::uint64_t index = 0;  // 0-based chunk index within the request
    // Optional reasoning/thinking text. Backends populate this only when the
    // caller explicitly requests separate reasoning; legacy callers continue
    // to receive text-only chunks.
    std::string_view reasoning{};
    // Optional per-token log probability and top-K alternatives when requested.
    std::optional<float> logprob = std::nullopt;
    std::vector<TopLogprob> top_logprobs{};
    // Optional native upstream tool call deltas (e.g. from llama-server).
    std::vector<ToolCallDelta> tool_call_deltas{};
    // Optional, non-owning: the probability of the backend-sampled token that
    // produced this chunk, when the request set telemetry_top_k (or asked for
    // logprobs) and the backend reported token ids. Valid only during the
    // callback; copies of the chunk must not keep it past that.
    const SampledTokenProbability* sampled_probability = nullptr;
};

// Per-request timing observations reported by llama-server. Each member is
// optional because streaming and older server versions may omit individual
// counters.
struct BackendTimings {
    std::optional<std::uint64_t> prompt_n = std::nullopt;
    std::optional<std::uint64_t> cache_n = std::nullopt;
    std::optional<double> prompt_ms = std::nullopt;
    std::optional<std::uint64_t> predicted_n = std::nullopt;
    std::optional<double> predicted_ms = std::nullopt;
    std::optional<std::uint64_t> draft_n = std::nullopt;
    std::optional<std::uint64_t> draft_n_accepted = std::nullopt;
};

enum class StopReason { none, max_tokens, stop_sequence, end_of_sequence, cancelled, callback, error, tool_calls, suspended };
const char* to_string(StopReason reason) noexcept;

struct GenerateStats {
    std::uint64_t prompt_tokens = 0;
    std::uint64_t completion_tokens = 0;
    std::uint64_t chunks = 0;              // streamed chunks delivered to the callback
    std::uint64_t load_ns = 0;             // backend-reported, 0 when unknown
    std::uint64_t prompt_eval_ns = 0;      // backend-reported, 0 when unknown
    std::uint64_t eval_ns = 0;             // backend-reported, 0 when unknown
    bool token_counts_from_backend = false;
    StopReason stop_reason = StopReason::none;
    // Optional upstream observations, not Sonder's logical scheduler cache.
    // Absence means the upstream did not report the counter (zero is valid).
    std::optional<std::uint64_t> cached_tokens;
    std::optional<std::uint64_t> draft_tokens;
    std::optional<std::uint64_t> draft_accepted_tokens;
    std::optional<double> predicted_tokens_per_second;
    // The exact configured stop sequence, when the backend/sampler identified
    // it. A finish reason alone is not sufficient to populate this field.
    std::optional<std::string> matched_stop = std::nullopt;
    // Raw llama-server timing counters, when the upstream supplied them.
    // This is additive to the legacy aggregate fields above.
    std::optional<BackendTimings> backend_timings = std::nullopt;
    // Bounded effective execution report (configured vs effective speculation,
    // fallback reasons, committed vs proposed counts, and 3-tier cache reuse).
    std::optional<EffectiveExecutionReport> effective_execution = std::nullopt;
    // Opt-in per-request stage timing breakdown (populated only when opted in).
    std::optional<StageTimings> stage_timings = std::nullopt;
};

// Return false to stop generation early (StopReason::callback).
using TokenCallback = std::function<bool(const TokenChunk&)>;

// Token-level decode loop for backends that advertise Capability::token_logits.
// Sonder owns sampling: it asks for the next logits, samples a token with its
// own sampler chain (src/sampling) and commits it with accept().
class TokenStream {
public:
    virtual ~TokenStream() = default;
    [[nodiscard]] virtual std::size_t vocab_size() const = 0;
    // Prompt tokens as the backend tokenized them (after prefill).
    [[nodiscard]] virtual const std::vector<TokenId>& prompt_tokens() const = 0;
    // Logits for the next position (size == vocab_size()). The span stays
    // valid until the next call on this stream.
    virtual Result<std::span<const float>> next_logits(const CancellationToken& cancel) = 0;
    // Commits `token` as the next generated token and returns its text piece.
    virtual Status accept(TokenId token, std::string& piece) = 0;
    // True for end-of-generation tokens (EOS/EOT).
    [[nodiscard]] virtual bool is_end_of_generation(TokenId token) const = 0;
    // Optional: capture process-local physical execution state at the stream's
    // current position (strictly same-process / same-model / same-build only).
    virtual Result<std::vector<std::uint8_t>> capture_state(
        std::size_t max_bytes = kMaxRecurrentCheckpointBytes) {
        (void)max_bytes;
        return Status(ErrorCode::unsupported, "token stream does not expose state capture");
    }
    // Optional: restore process-local physical execution state captured from
    // the exact same model/build/tokenizer.
    virtual Status restore_state(std::span<const std::uint8_t> bytes, std::uint64_t token_position) {
        (void)bytes;
        (void)token_position;
        return Status(ErrorCode::unsupported, "token stream does not expose state restore");
    }
    // Optional: explicitly reclaim any physical slot/KV held by this stream
    // when selectively suspended (returns true if physical state was reclaimed).
    virtual bool reclaim_physical_state() { return false; }
};

// ---------------------------------------------------------------------------
// Chat. A conversation is an ordered list of (role, content) messages. Roles
// follow the common convention: "system", "user", "assistant", "tool".
// Backends with a native chat API (Ollama /api/chat) or a model chat template
// (llama.cpp GGUF metadata) override BackendModel::chat(); everything else
// inherits the default, which flattens the conversation with
// format_chat_prompt() and calls generate().
// ---------------------------------------------------------------------------
// Ordered multimodal content part inside a ChatMessage.
struct TextContentPart {
    std::string text;
};

struct ImageContentPart {
    MediaContent media;
};

using ContentPart = std::variant<TextContentPart, ImageContentPart>;

struct ChatMessage {
    std::string role;
    std::string content;
    // Native reasoning history is optional and is ignored by backends that do
    // not have a separate reasoning channel.
    std::optional<std::string> reasoning_content = std::nullopt;
    // Optional assistant tool calls (role == "assistant").
    std::vector<ToolCall> tool_calls{};
    // Optional tool result correlation id (role == "tool").
    std::optional<std::string> tool_call_id = std::nullopt;
    // Optional participant / tool name.
    std::optional<std::string> name = std::nullopt;
    // Optional validated in-memory media attachments (base64 images).
    std::vector<MediaContent> media{};
    // Preferred ordered content parts (TextContentPart / ImageContentPart). When
    // non-empty, preserves exact interleaved text/image ordering across ingress,
    // templating, direct execution, and external forwarding. When empty, readers
    // fall back cleanly to legacy `content` + `media`.
    std::vector<ContentPart> parts{};
};

// Reasoning ("thinking") controls for native-chat backends. Unset fields are
// not forwarded, so the model's chat template keeps its own default. On
// hybrid reasoning models (Qwen3.x) these change the top of the rendered
// prompt, so switching them between turns defeats upstream prefix reuse.
struct ThinkingOptions {
    // llama-server: chat_template_kwargs.enable_thinking; Ollama: think.
    std::optional<bool> enable_thinking;
    // llama-server: chat_template_kwargs.reasoning_effort (not sent to Ollama).
    std::optional<std::string> reasoning_effort;
    [[nodiscard]] bool empty() const noexcept { return !enable_thinking && !reasoning_effort; }
};

struct ChatRequest {
    std::string request_id;
    std::vector<ChatMessage> messages;
    SamplingConfig sampling;
    // Conversation key for upstream cache affinity (llamaserver pins a key to
    // one llama-server slot). Empty = no affinity.
    std::string session_key;
    ThinkingOptions thinking;
    std::optional<std::int64_t> reasoning_budget_tokens = std::nullopt;
    std::optional<std::string> reasoning_budget_message = std::nullopt;
    // Request reasoning as a separate stream from visible assistant text.
    // False preserves the historical text-only callback contract.
    bool separate_reasoning = false;
    // Optional GBNF grammar constraining output (empty = unconstrained).
    std::string grammar{};
    // Optional OpenAI response_format hint ("text", "json_object", "json_schema")
    // and serialized JSON schema (when type == "json_schema") for upstreams that
    // handle response_format natively (e.g. llama-server).
    std::optional<std::string> response_format_type = std::nullopt;
    std::optional<std::string> response_format_schema = std::nullopt;
    // Optional tool definitions, tool_choice policy, and parallel tool-call flag.
    std::vector<ToolDefinition> tools{};
    std::optional<ToolChoice> tool_choice = std::nullopt;
    std::optional<bool> parallel_tool_calls = std::nullopt;
    // Optional scheduling priority class and latency targets (ms) forwarded to
    // backends with latency-aware physical batch planning.
    std::optional<RequestPriority> priority_class = std::nullopt;
    std::optional<double> target_ttft_ms = std::nullopt;
    std::optional<double> target_itl_ms = std::nullopt;
    // Opt-in per-request stage timing hooks (false by default: zero extra clocks).
    bool enable_stage_profiling = false;
    // Telemetry-only probability probe; same contract as
    // GenerateRequest::telemetry_top_k.
    std::int32_t telemetry_top_k = 0;
};

// True for "system", "user", "assistant" and "tool".
[[nodiscard]] bool is_known_chat_role(std::string_view role) noexcept;

// invalid_argument when the list is empty, a role is unknown, a message
// contains an embedded NUL byte, or the final message is not from "user" or
// "tool" (there would be nothing to answer).
Status validate_chat_messages(const std::vector<ChatMessage>& messages);

// Number of image attachments in `message` (uses ordered `parts` when
// non-empty, otherwise legacy `media`).
[[nodiscard]] std::size_t count_message_images(const ChatMessage& message) noexcept;

// True if any message carries image attachments in `parts` or `media`.
[[nodiscard]] bool has_chat_media(const std::vector<ChatMessage>& messages) noexcept;

// Concatenated text of `message` (from `parts` when non-empty, else `content`).
[[nodiscard]] std::string message_text_content(const ChatMessage& message);

// Appends `suffix` to both `content` and `parts` (when non-empty).
void append_chat_message_text(ChatMessage& message, std::string_view suffix);

// One segment of a chat-templated prompt, preserving the boundary between
// template-owned scaffolding (parse_special == true) and untrusted message
// content (parse_special == false).
struct ChatPromptSegment {
    std::string text;
    bool parse_special = false;
};

// Renders `messages` through `render_template` using unique per-message
// placeholders so template-owned control-token scaffolding is strictly
// separated from untrusted message content. Every scaffolding segment has
// parse_special == true; every message-content segment has parse_special ==
// false. Fails closed if any non-empty message placeholder does not round-trip
// through the template.
Result<std::vector<ChatPromptSegment>> build_chat_template_segments(
    const std::vector<ChatMessage>& messages,
    const std::function<Result<std::string>(const std::vector<ChatMessage>&)>& render_template);

// Tokenizes segmented chat output across the hard special-token boundary:
// scaffolding segments run with parse_special == true and untrusted content
// segments run with parse_special == false. `add_special` is enabled only on
// the first non-empty segment.
Status tokenize_chat_segments(
    const std::vector<ChatPromptSegment>& segments,
    const std::function<Status(std::string_view text, bool add_special, bool parse_special,
                               std::vector<TokenId>& out)>& tokenize_piece,
    std::vector<TokenId>& out);

// Backend-neutral prompt used by the default BackendModel::chat():
//
//   System: <system content>\n\n
//   User: <user content>\n\n
//   Assistant: <assistant content>\n\n
//   ...
//   Assistant:
//
// Role labels are capitalised; unknown roles are emitted verbatim. The
// trailing "Assistant:" cue asks the model to produce the next reply.
[[nodiscard]] std::string format_chat_prompt(const std::vector<ChatMessage>& messages);

class BackendModel {
public:
    virtual ~BackendModel() = default;
    [[nodiscard]] virtual const ModelDescriptor& descriptor() const = 0;
    // Synchronous streaming generation. Implementations must poll `cancel`
    // at least once per produced chunk and while waiting on I/O, and return
    // ErrorCode::cancelled promptly once it trips.
    virtual Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                           const TokenCallback& on_chunk) = 0;
    // Optional (Capability::tokenization): exact prompt tokenization. The engine
    // falls back to an approximate accounting tokenizer when unsupported.
    virtual Result<std::vector<TokenId>> tokenize(std::string_view text) {
        (void)text;
        return Status(ErrorCode::unsupported, "backend does not expose tokenization");
    }
    // Optional (Capability::tokenization): decode token ids back to UTF-8 text.
    virtual Result<std::string> detokenize(std::span<const TokenId> tokens) {
        (void)tokens;
        return Status(ErrorCode::unsupported, "backend does not expose detokenization");
    }
    // Optional (Capability::embeddings): compute dense embeddings for one or more inputs.
    virtual Result<EmbedResult> embed(const EmbedRequest& request, const CancellationToken& cancel) {
        (void)request;
        (void)cancel;
        return Status(ErrorCode::unsupported, "backend does not expose embeddings");
    }
    // Optional (Capability::reranking): rank documents against a query with a cross-encoder.
    virtual Result<RerankResult> rerank(const RerankRequest& request, const CancellationToken& cancel) {
        (void)request;
        (void)cancel;
        return Status(ErrorCode::unsupported, "backend does not expose reranking");
    }
    // Optional (Capability::reranking): score ordered (text_1, text_2) pairs preserving input order.
    virtual Result<ScoreResult> score(const ScoreRequest& request, const CancellationToken& cancel) {
        (void)request;
        (void)cancel;
        return Status(ErrorCode::unsupported, "backend does not expose pair scoring");
    }
    // Optional (Capability::token_logits): open a token-level decode loop for
    // `request` (prompt prefilled). Sonder samples from the returned logits.
    virtual Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) {
        (void)request;
        return Status(ErrorCode::unsupported, "backend does not expose token logits");
    }

    // Synchronous streaming chat completion: streams the assistant reply to
    // the last message. Same cancellation and callback contract as
    // generate(). The default validates the messages, formats them with
    // format_chat_prompt() and delegates to generate(), so every backend
    // supports chat without extra work.
    virtual Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                                       const TokenCallback& on_chunk);

    // True when chat() uses a backend-native chat API or model chat template
    // instead of the generic format_chat_prompt() fallback.
    [[nodiscard]] virtual bool has_native_chat() const { return false; }

    // Optional (Capability::recurrent_checkpoint): capture serialized recurrent
    // state corresponding to `token_position`.
    virtual Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) {
        (void)token_position;
        return Status(ErrorCode::unsupported, "backend does not expose recurrent-state checkpoints");
    }

    // Optional (Capability::recurrent_checkpoint): restore a previously captured
    // recurrent-state snapshot prior to backend decode.
    virtual Status restore_recurrent_state(std::span<const std::uint8_t> bytes) {
        (void)bytes;
        return Status(ErrorCode::unsupported, "backend does not expose recurrent-state checkpoints");
    }

    // Optional (Capability::execution_checkpoint): capture/restore process-local
    // physical sequence state (KV + recurrent state) for continuation or spill.
    // Valid ONLY within the same process, build, model, and tokenizer.
    virtual Result<std::vector<std::uint8_t>> capture_execution_state(
        std::uint64_t token_position, std::size_t max_bytes = kMaxRecurrentCheckpointBytes) {
        (void)token_position;
        (void)max_bytes;
        return Status(ErrorCode::unsupported, "backend does not expose execution-state checkpoints");
    }

    virtual Status restore_execution_state(std::span<const std::uint8_t> bytes, std::uint64_t token_position) {
        (void)bytes;
        (void)token_position;
        return Status(ErrorCode::unsupported, "backend does not expose execution-state checkpoints");
    }

    // Diagnostic counters for physical slot occupancy and reclamation.
    [[nodiscard]] virtual std::size_t active_physical_slots() const { return 0; }
    [[nodiscard]] virtual std::uint64_t reclaimed_physical_slots() const { return 0; }
};

class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
    // Cheap reachability/health check. Returns a version string on success.
    virtual Result<std::string> probe() = 0;
    virtual Result<std::vector<ModelDescriptor>> list_models() = 0;
    virtual Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) = 0;
    // Optional runtime observations (GPU memory, context fit, performance
    // warnings). Must be a cheap snapshot: never blocks on backend I/O, and is
    // safe to call from any thread. nullopt (the default) means the backend
    // reports nothing, and hosts then add no runtime fields to their output.
    [[nodiscard]] virtual std::optional<BackendRuntimeStatus> runtime_status() const { return std::nullopt; }
    // Optional structured 5-state capability evidence (cheap, non-blocking,
    // safe from any thread). Empty by default; callers fall back to honest
    // adapter_supports + unknown for unmeasured states.
    [[nodiscard]] virtual std::vector<CapabilityStatusEntry> capability_statuses() const { return {}; }
    // Cheap, thread-safe snapshot of simultaneous generation slots. Zero means
    // unknown; HTTP hosts use this to keep waiting work out of upstream queues.
    [[nodiscard]] virtual std::size_t max_concurrent_requests() const { return 0; }
};

}  // namespace sonder::inference
