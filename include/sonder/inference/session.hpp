// Sonder Inference: generation session.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/model.hpp"
#include "sonder/inference/request_priority.hpp"
#include "sonder/inference/telemetry.hpp"

// Session::chat() exists (runs chat through the request runtime and emits
// session telemetry with request.queued.kind = "chat").
#define SONDER_HAS_SESSION_CHAT 1

namespace sonder::inference {

// idle -> running -> idle (repeatable); any -> closed (terminal).
enum class SessionState { idle, running, closed };
enum class RequestOutcome { none, completed, cancelled, failed };

// Workload classes (docs/SCHEDULER.md), most to least latency-sensitive.
// The numeric value is the scheduler's default priority rank (lower = sooner).
enum class WorkloadClass {
    interactive_user = 0,
    owner_orchestrator = 1,
    critic_verification = 2,
    implementation_worker = 3,
    research_worker = 4,
    background_indexing = 5,
    maintenance = 6,
};

const char* to_string(SessionState state) noexcept;
const char* to_string(RequestOutcome outcome) noexcept;
const char* to_string(WorkloadClass workload) noexcept;

struct SessionOptions {
    std::string session_id;  // generated when empty
    std::optional<std::string> run_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    SamplingConfig sampling;
    // Scheduling metadata (used when the engine schedules requests).
    WorkloadClass workload = WorkloadClass::implementation_worker;
    // Adjusts the class rank: effective rank = class rank - priority, so a
    // positive priority is more urgent. 0 keeps the class default.
    int priority = 0;
};

// How the engine scheduled a request (all zero when scheduling is inactive).
struct SchedulingInfo {
    bool scheduled = false;              // went through scheduler + KV cache
    std::uint64_t accounted_prompt_tokens = 0;  // tokens used for KV accounting
    bool exact_prompt_tokens = false;    // true when the backend tokenized the prompt
    std::uint64_t reused_prompt_tokens = 0;     // prompt tokens served by the prefix cache
    std::uint32_t preemptions = 0;
    double queue_ms = 0.0;               // submit -> first admission
    bool sonder_sampled = false;         // sampled by Sonder's sampler chain (token_logits)
    bool suspended = false;              // parked cleanly at a backend safe point
    bool physical_slot_reclaimed = false;  // verified backend released physical slot/KV
    SuspendAction suspend_action = SuspendAction::continue_running;
};

// ---------------------------------------------------------------------------
// Caller-visible Execution-State Checkpoints (Package 4 / Item #8).
//
// Scope: strictly SAME-PROCESS, SAME-BUILD, SAME-MODEL, SAME-TOKENIZER only.
// Execution checkpoints are NOT cross-host or cross-process portable; any
// compatibility mismatch is rejected on resume.
// ---------------------------------------------------------------------------

// Explicit continuation contract:
//   Branch:      Start a fresh branch from a valid token/prefix boundary.
//                Fresh sampler/RNG/penalty/stop/grammar state; isolated forked
//                logical KV via KvCacheManager::fork_sequence (+ truncate on rewind).
//   ExactResume: Resume an interrupted generation preserving model/build
//                compatibility fingerprint, token history, delivered output
//                offset, sampler RNG/penalty history, grammar matcher state,
//                stop-filter pending buffer, pending draft tokens, and
//                process-local backend physical/recurrent state blob.
enum class CheckpointMode : std::uint8_t {
    Branch = 0,
    ExactResume = 1,
};

const char* to_string(CheckpointMode mode) noexcept;

// Process-local compatibility identity enforced on checkpoint resume.
struct ModelCompatibilityFingerprint {
    std::uint64_t process_epoch = 0;
    std::string backend_name;
    std::string build_commit;
    std::string model_name;
    std::string model_format;
    std::string model_family;
    std::string parameter_size;
    std::string quantization;
    std::uint64_t model_size_bytes = 0;
    std::uint64_t context_length = 0;
    ModelArchitecture architecture = ModelArchitecture::attention_only;
    std::uint64_t tokenizer_hash = 0;
    std::uint64_t digest = 0;

    [[nodiscard]] bool compatible_with(const ModelCompatibilityFingerprint& other) const noexcept;
    [[nodiscard]] std::string mismatch_reason(const ModelCompatibilityFingerprint& other) const;
};

// Computes the process-local compatibility fingerprint for `model`.
[[nodiscard]] ModelCompatibilityFingerprint compute_model_compatibility_fingerprint(const Model& model);

// Bounded, owned, expirable caller-visible checkpoint handle.
struct ExecutionCheckpointHandle {
    std::string checkpoint_id;
    std::string owner_session_id;
    CheckpointMode mode = CheckpointMode::Branch;
    ModelCompatibilityFingerprint compatibility;
    std::uint64_t token_position = 0;
    std::uint64_t delivered_offset_bytes = 0;
    std::uint64_t delivered_chunks = 0;
    std::uint64_t delivered_completion_tokens = 0;
    std::size_t approx_bytes = 0;
    std::chrono::steady_clock::time_point created_at{};
    std::chrono::steady_clock::time_point expires_at{};
    // Documented & enforced scope: checkpoints are process-local only.
    bool same_process_only = true;

    [[nodiscard]] bool valid() const noexcept { return !checkpoint_id.empty(); }
    [[nodiscard]] bool expired(
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) const noexcept {
        return expires_at != std::chrono::steady_clock::time_point{} && now >= expires_at;
    }
};

// Complete continuation snapshot stored in CheckpointStore.
struct ExecutionCheckpointRecord {
    ExecutionCheckpointHandle handle;

    // Context & token history.
    std::string prompt;
    std::vector<TokenId> prompt_tokens;
    std::vector<std::string> context_token_pieces;
    std::vector<TokenId> generated_tokens;
    std::uint64_t logical_kv_sequence_id = 0;
    bool exact_tokens = false;

    // Delivered output continuity (ExactResume only; zeroed for Branch).
    std::uint64_t delivered_offset_bytes = 0;
    std::uint64_t delivered_chunks = 0;
    std::uint64_t delivered_completion_tokens = 0;
    std::string delivered_text;

    // Exact sampling / grammar / stop-filter / draft continuation state
    // (populated for ExactResume; fresh/empty for Branch).
    SamplingConfig sampling;
    std::array<std::uint64_t, 4> sampler_rng_state{};
    bool has_sampler_rng_state = false;
    std::vector<TokenId> sampler_penalty_history;
    std::string grammar;
    std::vector<std::uint8_t> grammar_matcher_state;
    std::string stop_filter_pending;
    bool stop_filter_stopped = false;
    std::vector<TokenId> pending_draft_tokens;

    // Process-local backend physical/recurrent state blob (never portable).
    std::vector<std::uint8_t> backend_state_blob;
    std::uint64_t backend_state_token_position = 0;
    std::uint64_t backend_step_index = 0;

    [[nodiscard]] std::size_t accounted_bytes() const noexcept;
};

struct ExecutionCheckpointStoreOptions {
    std::size_t max_bytes = std::size_t{64} << 20;  // 64 MiB bounded cap
    std::size_t max_entries = 256;
    std::chrono::milliseconds default_ttl{std::chrono::minutes{10}};
};

struct ExecutionCheckpointStoreStats {
    std::size_t entries = 0;
    std::size_t total_bytes = 0;
    std::size_t max_bytes = 0;
    std::size_t max_entries = 0;
    std::uint64_t evictions_capacity = 0;
    std::uint64_t evictions_expired = 0;
    std::uint64_t releases = 0;
};

// Bounded, thread-safe, TTL-expiring, LRU-evicting execution checkpoint store.
class CheckpointStore {
public:
    using ClockFn = std::function<std::chrono::steady_clock::time_point()>;
    using EvictCallback = std::function<void(const ExecutionCheckpointRecord&)>;

    explicit CheckpointStore(ExecutionCheckpointStoreOptions options = {}, ClockFn clock = {});
    ~CheckpointStore() = default;
    CheckpointStore(const CheckpointStore&) = delete;
    CheckpointStore& operator=(const CheckpointStore&) = delete;

    Result<ExecutionCheckpointHandle> put(ExecutionCheckpointRecord record,
                                          std::optional<std::chrono::milliseconds> ttl = std::nullopt);
    Result<ExecutionCheckpointRecord> get(const std::string& checkpoint_id);
    bool release(const std::string& checkpoint_id);
    std::size_t release_owned_by(std::string_view session_id);
    std::size_t sweep_expired();
    void clear();
    void set_evict_callback(EvictCallback cb);

    [[nodiscard]] ExecutionCheckpointStoreStats stats() const;
    [[nodiscard]] const ExecutionCheckpointStoreOptions& options() const noexcept { return options_; }

private:
    struct Entry {
        std::string id;
        ExecutionCheckpointRecord record;
        std::size_t bytes = 0;
    };

    std::chrono::steady_clock::time_point now() const;
    void collect_expired_locked(std::chrono::steady_clock::time_point ts,
                                std::vector<ExecutionCheckpointRecord>& evicted);
    void invoke_evicted(const std::vector<ExecutionCheckpointRecord>& evicted);

    ExecutionCheckpointStoreOptions options_;
    ClockFn clock_;
    mutable std::mutex mutex_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    std::size_t total_bytes_ = 0;
    std::uint64_t evictions_capacity_ = 0;
    std::uint64_t evictions_expired_ = 0;
    std::uint64_t releases_ = 0;
    EvictCallback evict_callback_;
};

// Per-request options that are not part of the session's configuration.
struct RequestOptions {
    // Engine-side request id (envelope request_id). Generated ("req-...") when
    // unset; a caller that supplies one must keep it unique, e.g. from
    // make_id("req"), so it can reference the request before it finishes.
    std::optional<std::string> request_id;
    // Caller-side request that caused this one (e.g. a Sonder Runtime turn id).
    // Emitted as attributes.parent_request_id on request.queued, started,
    // completed, cancelled and failed. The envelope request_id stays the
    // engine-generated id.
    std::optional<std::string> parent_request_id;
    // Chat on a native-chat backend only (ChatRequest::session_key and
    // ChatRequest::thinking); ignored by generate() and the generic template.
    std::string session_key;
    ThinkingOptions thinking;
    // Backend-native chat may return reasoning separately for protocols that
    // expose it (for example Anthropic-compatible responses).
    bool separate_reasoning = false;
    // Optional hosted-server admission class. The engine ignores this unless
    // strict priority admission is enabled.
    std::optional<RequestPriority> priority_class;
    // Admission time spent in an outer server queue, supplied additively to
    // the engine scheduler's own queue time for telemetry.
    double admission_queue_ms = 0.0;
    // Shared host ticket joins slot/class eligibility to logical KV admission.
    // Its queue time supersedes admission_queue_ms + the runtime's queue time.
    std::shared_ptr<RequestAdmission> admission;
    // Optional absolute deadline. A request waiting for admission expires
    // with timeout; a running request remains cooperatively cancellable.
    std::optional<std::chrono::steady_clock::time_point> deadline;
    // Called exactly once after scheduler admission and immediately before
    // the backend invocation, after checking the deadline. Must not throw.
    std::function<void()> on_backend_start;
    // Keep SessionOptions::workload - priority as the scheduler rank even
    // with strict class admission. Used by legacy numeric HTTP priorities,
    // including explicit zero; the host ticket still enforces class/FIFO.
    bool preserve_numeric_priority = false;
    // Native llama-server chat only. Unset leaves the upstream default intact.
    std::optional<std::int64_t> reasoning_budget_tokens = std::nullopt;
    std::optional<std::string> reasoning_budget_message = std::nullopt;
    // Cumulative byte cap on generated text + reasoning delivered in one
    // request (0 disables). Exceeding it stops streaming and fails the request
    // with ErrorCode::protocol_error.
    std::size_t max_response_bytes = 32u << 20;
    // When false (e.g. SSE streaming where chunks are flushed immediately),
    // Session skips retaining the full text/reasoning in GenerationResult.
    bool retain_text = true;
    // Optional GBNF grammar constraining output (empty = unconstrained).
    std::string grammar;
    // Optional OpenAI response_format hint ("text", "json_object", "json_schema")
    // and serialized JSON schema for native-format upstreams.
    std::optional<std::string> response_format_type = std::nullopt;
    std::optional<std::string> response_format_schema = std::nullopt;
    // Optional tool definitions, tool_choice policy, and parallel tool-call flag for chat requests.
    std::vector<ToolDefinition> tools{};
    std::optional<ToolChoice> tool_choice = std::nullopt;
    std::optional<bool> parallel_tool_calls = std::nullopt;
    // Optional latency targets (milliseconds) forwarded to execution backends
    // that support latency-aware physical batch planning.
    std::optional<double> target_ttft_ms = std::nullopt;
    std::optional<double> target_itl_ms = std::nullopt;
    // Opt-in per-request stage timing hooks (Doc 2 #25). False by default so
    // unprofiled requests incur zero extra stage clocks or timing events.
    bool enable_stage_profiling = false;
    // Optional non-blocking safe-point suspension hook (Item #9). Must not block.
    SuspendSafePointCallback on_safe_point{};
    // Force non-blocking scheduler gating inside backend callbacks (automatically
    // enabled on backends declaring selective_preemption or continuous_batch_decode).
    bool nonblocking_gate = false;
    // Automatically store an ExactResume checkpoint in Engine::checkpoint_store()
    // when generation stops cleanly with StopReason::suspended.
    bool auto_checkpoint_on_suspend = true;
    // Optional TTL override for auto-exported suspend checkpoints.
    std::optional<std::chrono::milliseconds> suspend_checkpoint_ttl = std::nullopt;
    // Allow cost-aware spill-to-checkpoint when selectively preempted under pressure.
    bool allow_preempt_spill = false;
    // Optional stateful grammar matcher hooks (for ExactResume state snapshot/restore
    // and Branch reset on token_logits backends).
    std::optional<GrammarMatcherHooks> grammar_hooks = std::nullopt;
};

struct ForkOptions {
    SessionOptions session_options{};
    bool inherit_session_options = true;
    // Optional token offset to fork from (<= parent context_token_count()).
    // std::nullopt forks from the full current context length.
    std::optional<std::size_t> prefix_token_offset = std::nullopt;
};

struct CheckpointExportOptions {
    CheckpointMode mode = CheckpointMode::Branch;
    // For Branch mode: optional prefix token boundary (<= current context length).
    std::optional<std::size_t> token_offset = std::nullopt;
    // Optional TTL override (defaults to CheckpointStoreOptions::default_ttl).
    std::optional<std::chrono::milliseconds> ttl = std::nullopt;
    // Optional pending speculative draft tokens to preserve in ExactResume mode.
    std::vector<TokenId> pending_draft_tokens{};
    // Optional external grammar matcher state snapshot override (ExactResume).
    std::vector<std::uint8_t> grammar_matcher_state{};
};

struct ResumeOptions {
    // Optional sampling override (valid for Branch mode; ExactResume preserves
    // the checkpointed sampling/RNG/penalty/stop/grammar state).
    std::optional<SamplingConfig> sampling_override = std::nullopt;
    // Optional prompt suffix appended in Branch mode (rejected in ExactResume).
    std::string branch_prompt_suffix{};
    // Optional remaining token budget override for ExactResume (unset continues
    // up to original sampling.max_tokens - delivered_completion_tokens).
    std::optional<std::int32_t> max_additional_tokens = std::nullopt;
    RequestOptions request_options{};
    // When true, releases the checkpoint from CheckpointStore once consumed.
    bool consume_checkpoint = false;
};

struct GenerationResult {
    std::string request_id;
    std::string text;
    GenerateStats stats;
    RequestOutcome outcome = RequestOutcome::none;
    double ttft_ms = -1.0;   // -1 when no chunk was produced
    double total_ms = 0.0;
    SchedulingInfo scheduling;
    std::string reasoning{};
    std::vector<TokenLogprob> logprobs{};
    std::vector<ToolCall> tool_calls{};
    // Bounded configured-vs-effective execution report with distinct 3-tier
    // cache counters (logical vs physical device vs physical host L2).
    std::optional<EffectiveExecutionReport> effective_execution = std::nullopt;
    // Opt-in per-request stage timing breakdown (absent when profiling is off).
    std::optional<StageTimings> stage_timings = std::nullopt;
    // Populated when generation parked at a safe point (StopReason::suspended)
    // and auto_checkpoint_on_suspend was true.
    std::optional<ExecutionCheckpointHandle> suspended_checkpoint = std::nullopt;
    std::uint64_t delivered_offset_bytes = 0;
    std::uint64_t delivered_chunks = 0;
};

class Engine;

class Session {
public:
    Session(Engine& engine, std::shared_ptr<Model> model, SessionOptions options);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] const std::string& id() const noexcept { return options_.session_id; }
    [[nodiscard]] SessionState state() const;
    [[nodiscard]] RequestOutcome last_outcome() const;
    [[nodiscard]] const SessionOptions& options() const noexcept { return options_; }
    [[nodiscard]] const std::shared_ptr<Model>& model() const noexcept { return model_; }
    [[nodiscard]] std::uint64_t requests_started() const;

    // Runs one request synchronously, streaming chunks to `on_chunk`.
    // A cancelled request returns a Result holding the partial output with
    // outcome == cancelled (not an error). Errors: invalid_argument (bad
    // sampling), invalid_state (closed or already running), backend errors.
    // Exceptions from on_chunk propagate unchanged after session-state recovery
    // and one runtime cleanup attempt (see docs/integration/request-callback-exceptions.md).
    // Delivered chunks and callback side effects are never replayed or undone.
    Result<GenerationResult> generate(const std::string& prompt, const TokenCallback& on_chunk = {},
                                      const std::optional<SamplingConfig>& sampling_override = std::nullopt,
                                      const RequestOptions& request_options = {});

    // Chat completion: answers the last message of `messages` through the same
    // request runtime (scheduler, KV accounting, sampler) and telemetry as
    // generate(), with request.queued.kind = "chat". Backends with a native
    // chat API or model chat template (BackendModel::has_native_chat()) get the
    // messages as-is; otherwise the conversation is flattened with
    // format_chat_prompt() and generated like a prompt (Sonder's sampler chain
    // applies on token_logits backends). Errors as generate(), plus
    // invalid_argument for invalid messages (validate_chat_messages()).
    Result<GenerationResult> chat(const std::vector<ChatMessage>& messages, const TokenCallback& on_chunk = {},
                                  const std::optional<SamplingConfig>& sampling_override = std::nullopt,
                                  const RequestOptions& request_options = {});

    // Embedding generation for one or more text inputs on backends that support
    // Capability::embeddings. The template overload constrains on EmbedRequest so
    // braced-init calls like embed({"hello"}) unambiguously bind to the vector overload.
    Result<EmbedResult> embed(const std::vector<std::string>& inputs, const RequestOptions& request_options = {});
    Result<EmbedResult> embed_request(const EmbedRequest& request, const RequestOptions& request_options = {});
    template <typename Req>
        requires std::same_as<std::remove_cvref_t<Req>, EmbedRequest>
    Result<EmbedResult> embed(const Req& request, const RequestOptions& request_options = {}) {
        return embed_request(request, request_options);
    }

    // Cross-encoder reranking for (query, documents[]) on backends that support
    // Capability::reranking. Preserves original document indices, sorts descending
    // by relevance_score, and honors bounded top_n and return_documents.
    Result<RerankResult> rerank(const RerankRequest& request, const RequestOptions& request_options = {});

    // Cross-encoder pair scoring for ordered (text_1, text_2) pairs on backends
    // that support Capability::reranking. Preserves exact input pair order.
    Result<ScoreResult> score(const ScoreRequest& request, const RequestOptions& request_options = {});

    // True when the last request failed because the engine scheduler rejected
    // it before any backend work started (nothing executed).
    [[nodiscard]] bool last_scheduler_rejected() const;

    // Cancels the in-flight request, if any. Safe from any thread, including
    // from inside the chunk callback. No effect on future requests.
    void cancel();
    // Requests cooperative selective suspension of the in-flight request at
    // the backend's next safe point (non-blocking; parks & reclaims THAT
    // request's physical slot without stalling peer requests).
    void request_suspend(SuspendAction action = SuspendAction::yield_spill);
    // Cancels any in-flight request, releases owned checkpoints and logical KV
    // sequence state, and makes the session terminal.
    void close();

    // Execution-state inspection & lifecycle (Package 4 / Item #8).
    [[nodiscard]] std::size_t context_token_count() const;
    [[nodiscard]] std::vector<TokenId> context_tokens() const;
    [[nodiscard]] std::string context_prompt() const;
    [[nodiscard]] std::uint64_t logical_kv_sequence_id() const;
    [[nodiscard]] ModelCompatibilityFingerprint compatibility_fingerprint() const;

    // Creates an isolated child Session branched from this session's current
    // context (or `options.prefix_token_offset`) using CheckpointMode::Branch
    // semantics: fresh sampler/stop/grammar state and isolated forked logical KV
    // via KvCacheManager::fork_sequence (+ truncate on rewind).
    Result<std::shared_ptr<Session>> fork(const ForkOptions& options = {});

    // Rewinds this session's tracked context and logical KV sequence to
    // `token_offset` (via KvCacheManager::truncate). Clears any stale
    // ExactResume continuation state past the rewind point.
    Status rewind(std::size_t token_offset);

    // Exports a bounded, owned, expirable checkpoint into Engine::checkpoint_store().
    // Strictly same-process / same-model-fingerprint only.
    Result<ExecutionCheckpointHandle> export_checkpoint(const CheckpointExportOptions& options = {});

    // Resumes execution from a checkpoint handle, enforcing same-process & exact
    // ModelCompatibilityFingerprint matching, expiry, and Branch vs ExactResume
    // semantics.
    Result<GenerationResult> resume_from_checkpoint(const ExecutionCheckpointHandle& handle,
                                                    const TokenCallback& on_chunk = {},
                                                    const ResumeOptions& resume_options = {});

    // Releases a checkpoint handle (freeing its memory and any forked logical KV).
    bool release_checkpoint(const ExecutionCheckpointHandle& handle);

    [[nodiscard]] TelemetryContext telemetry_context(const std::optional<std::string>& request_id = std::nullopt) const;

private:
    Result<GenerationResult> run_request(const char* kind, const std::string& prompt,
                                         const std::vector<ChatMessage>* messages, const TokenCallback& on_chunk,
                                         const std::optional<SamplingConfig>& sampling_override,
                                         const RequestOptions& request_options,
                                         const ExecutionCheckpointRecord* exact_resume = nullptr,
                                         std::optional<std::int32_t> max_additional_tokens = std::nullopt);

    Engine& engine_;
    std::shared_ptr<Model> model_;
    SessionOptions options_;

    mutable std::mutex mutex_;
    SessionState state_ = SessionState::idle;
    RequestOutcome last_outcome_ = RequestOutcome::none;
    bool last_scheduler_rejected_ = false;
    std::optional<CancellationSource> active_cancel_;
    std::uint64_t requests_started_ = 0;
    std::atomic<std::uint64_t> cancel_requested_ns_{0};
    std::atomic<std::uint8_t> suspend_requested_{0};

    // Tracked context & continuation state (protected by mutex_).
    std::string context_prompt_;
    std::vector<TokenId> context_tokens_;
    std::vector<std::string> context_token_pieces_;
    bool exact_context_tokens_ = false;
    bool branched_context_ = false;
    std::uint64_t logical_kv_seq_id_ = 0;
    std::optional<ExecutionCheckpointRecord> last_continuation_;
    std::vector<std::uint8_t> last_recurrent_state_blob_;
};

}  // namespace sonder::inference
