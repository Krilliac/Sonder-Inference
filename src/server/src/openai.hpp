// Internal: OpenAI-compatible request mapping and response shaping for
// POST /v1/chat/completions (docs/SERVER.md). Pure functions.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "http.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/sampling.hpp"
#include "sonder/inference/session.hpp"
#include "sonder/inference/request_priority.hpp"

namespace sonder::inference::server::detail {

// {"error":{"message","type","code","param"}} plus the HTTP status and an
// optional Retry-After (seconds).
struct ApiError {
    int status = 500;
    std::string code;
    std::string message;
    std::optional<std::string> param;
    std::optional<int> retry_after;
};

// OpenAI error "type" for a status (400 invalid_request_error, 401
// authentication_error, 403 permission_error, 404 not_found_error, 429
// rate_limit_error, 500 server_error, 503 service_unavailable, ...).
const char* error_type(int status) noexcept;
ApiError make_error(int status, std::string code, std::string message, std::optional<std::string> param = std::nullopt);
// Error document, including "sonder":{"api_version":1}.
json::Object error_body(const ApiError& error);

// Default completion budget when the request sets neither max_tokens nor
// max_completion_tokens.
inline constexpr std::int32_t kDefaultMaxTokens = 1024;

struct ChatJob {
    std::string model = "default";
    std::vector<ChatMessage> messages;
    SamplingConfig sampling;
    bool stream = false;
    bool include_usage = false;
    // prompt_cache_key (OpenAI): conversation key for upstream cache affinity.
    std::optional<std::string> session_key;
    // chat_template_kwargs.enable_thinking / .reasoning_effort, and the
    // top-level think (same meaning as enable_thinking).
    ThinkingOptions thinking;
    // Optional request scheduling hints. The server resolves an absent class
    // to interactive and applies header values before these body values.
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

// Maps a request body to a ChatJob. Unknown top-level fields are ignored;
// user is accepted and ignored. chat_template_kwargs must be an object:
// enable_thinking (bool) and reasoning_effort (string, 1-64 of
// [A-Za-z0-9._-]) are forwarded, other keys ignored; think (bool) must agree
// with enable_thinking when both are set; prompt_cache_key is a 1-256 byte
// string (invalid_json otherwise). Rejected with
// unsupported_parameter: tools, tool_choice, functions, function_call,
// response_format, logprobs (true), top_logprobs, n != 1, non-string message
// content, tool_calls on a message. Sonder extensions: num_ctx, typical_p,
// repeat_last_n, repeat_penalty, top_k, min_p.
std::variant<ChatJob, ApiError> parse_chat_request(std::string_view body);

// Parses the additive scheduling hint values used by both JSON bodies and
// headers. Unknown values return nullopt; callers should map that to their
// endpoint-specific 400 error.
std::optional<RequestPriority> parse_request_priority(std::string_view value) noexcept;

// Positive integer deadline in milliseconds. Zero and malformed values are
// rejected. Values are bounded to INT32_MAX milliseconds so they can be
// represented safely by the server's chrono deadline arithmetic.
std::optional<std::uint64_t> parse_deadline_ms(const json::Value& value) noexcept;

struct Correlation {
    std::optional<std::string> run_id;
    std::optional<std::string> parent_request_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    WorkloadClass workload = WorkloadClass::interactive_user;
    int priority = 0;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
    // Explicit numeric zero also keeps the workload's legacy scheduler rank.
    bool numeric_priority = false;
};

// X-Sonder-Run-Id, -Parent-Request-Id, -Agent-Id, -Task-Id, -Workload,
// -Priority. X-Sonder-Priority accepts the legacy integer -16..16 (retained
// in Correlation::priority and mapped to interactive admission) or the
// additive class names interactive, subagent and background. Invalid values:
// 400 invalid_correlation_header naming the header.
std::variant<Correlation, ApiError> parse_correlation(const RequestHead& head);
std::optional<WorkloadClass> parse_workload(std::string_view name) noexcept;

// Maps a failed Session request to an API error: scheduler rejections before
// execution -> 429 overloaded (Retry-After 1), except a prompt that can
// never fit the KV pool -> 400 invalid_messages; other invalid_argument ->
// 400 invalid_sampling ("sampling failed ..."), unsupported_parameter with
// the field ("... does not support <field>", a backend refusal) or
// invalid_request (messages are validated before the session runs);
// not_found -> 404 model_not_found;
// unsupported -> 400 unsupported_parameter; backend-side failures -> 503
// backend_unavailable (executed, not safe to replay); anything else -> 500.
ApiError map_session_failure(const Status& status, bool scheduler_rejected);

// "stop" | "length" | "cancelled".
const char* finish_reason(const GenerationResult& result) noexcept;
// prompt/completion/total_tokens, plus prompt_tokens_details.cached_tokens
// when the backend reported a cached-prompt count.
json::Object usage_json(const GenerationResult& result);
// prompt_n, predicted_n, total_ms, plus only measured values: ttft_ms,
// prompt_ms, predicted_ms, prompt_per_second, predicted_per_second, and the
// backend's cache_n, draft_n, draft_n_accepted and the scheduler's queue_ms
// (scheduled requests only). A counter the backend did not report is
// omitted, never emitted as 0.
json::Object timings_json(const GenerationResult& result);

}  // namespace sonder::inference::server::detail
