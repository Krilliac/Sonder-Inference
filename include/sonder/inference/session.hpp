// Sonder Inference: generation session.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/model.hpp"
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
};

struct GenerationResult {
    std::string request_id;
    std::string text;
    GenerateStats stats;
    RequestOutcome outcome = RequestOutcome::none;
    double ttft_ms = -1.0;   // -1 when no chunk was produced
    double total_ms = 0.0;
    SchedulingInfo scheduling;
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

    // True when the last request failed because the engine scheduler rejected
    // it before any backend work started (nothing executed).
    [[nodiscard]] bool last_scheduler_rejected() const;

    // Cancels the in-flight request, if any. Safe from any thread, including
    // from inside the chunk callback. No effect on future requests.
    void cancel();
    // Cancels any in-flight request and makes the session terminal.
    void close();

    [[nodiscard]] TelemetryContext telemetry_context(const std::optional<std::string>& request_id = std::nullopt) const;

private:
    Result<GenerationResult> run_request(const char* kind, const std::string& prompt,
                                         const std::vector<ChatMessage>* messages, const TokenCallback& on_chunk,
                                         const std::optional<SamplingConfig>& sampling_override,
                                         const RequestOptions& request_options);

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
};

}  // namespace sonder::inference
