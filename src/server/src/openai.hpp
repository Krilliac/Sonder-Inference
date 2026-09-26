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
};

// Maps a request body to a ChatJob. Unknown top-level fields are ignored;
// user and chat_template_kwargs are accepted and ignored. Rejected with
// unsupported_parameter: tools, tool_choice, functions, function_call,
// response_format, logprobs (true), top_logprobs, n != 1, non-string message
// content, tool_calls on a message. Sonder extensions: num_ctx, typical_p,
// repeat_last_n, repeat_penalty, top_k, min_p.
std::variant<ChatJob, ApiError> parse_chat_request(std::string_view body);

struct Correlation {
    std::optional<std::string> run_id;
    std::optional<std::string> parent_request_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    WorkloadClass workload = WorkloadClass::interactive_user;
    int priority = 0;
};

// X-Sonder-Run-Id, -Parent-Request-Id, -Agent-Id, -Task-Id, -Workload,
// -Priority. Invalid values: 400 invalid_correlation_header naming the header.
std::variant<Correlation, ApiError> parse_correlation(const RequestHead& head);
std::optional<WorkloadClass> parse_workload(std::string_view name) noexcept;

// Maps a failed Session request to an API error: scheduler rejections before
// execution -> 429 overloaded (Retry-After 1); invalid_argument -> 400
// (invalid_sampling / invalid_messages); not_found -> 404 model_not_found;
// unsupported -> 400 unsupported_parameter; backend-side failures -> 503
// backend_unavailable (executed, not safe to replay); anything else -> 500.
ApiError map_session_failure(const Status& status, bool scheduler_rejected);

// "stop" | "length" | "cancelled".
const char* finish_reason(const GenerationResult& result) noexcept;
json::Object usage_json(const GenerationResult& result);
// prompt_n, predicted_n, total_ms, plus only measured values: ttft_ms,
// prompt_ms, predicted_ms, prompt_per_second, predicted_per_second.
json::Object timings_json(const GenerationResult& result);

}  // namespace sonder::inference::server::detail
