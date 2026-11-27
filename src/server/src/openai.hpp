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
#include "json_schema_gbnf.hpp"
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
inline constexpr std::size_t kMaxRequestGrammarBytes = 64u * 1024u;
inline constexpr std::size_t kMaxEmbedInputs = sonder::inference::kMaxEmbedInputs;
inline constexpr std::size_t kMaxRerankDocuments = sonder::inference::kMaxRerankDocuments;
inline constexpr std::size_t kMaxScorePairs = sonder::inference::kMaxScorePairs;
inline constexpr std::size_t kMaxEmbedDimensions = 65536u;

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
    std::optional<std::int64_t> reasoning_budget_tokens = std::nullopt;
    std::optional<std::string> reasoning_budget_message = std::nullopt;
    // Optional GBNF grammar (from explicit `grammar` or converted from `response_format`).
    std::string grammar;
    // Optional native OpenAI response_format hint ("json_object" or "json_schema")
    // and serialized JSON schema when type == "json_schema".
    std::optional<std::string> response_format_type;
    std::optional<std::string> response_format_schema;
    // Deferred schema value + immutable compiled grammar artifact resolved against model/tokenizer identity.
    std::optional<json::Value> pending_response_schema;
    std::shared_ptr<const CompiledGrammar> compiled_grammar;
    // Optional tool definitions, tool_choice policy, and parallel tool-call flag.
    std::vector<ToolDefinition> tools;
    std::optional<ToolChoice> tool_choice;
    std::optional<bool> parallel_tool_calls;
};

// Per-request bounds for POST /v1/embeddings, checked at ingress (before any
// tokenization). The backend enforces the model-context bounds on top of these.
struct EmbedRouteLimits {
    std::size_t max_inputs = kMaxEmbedInputs;              // items in `input`
    std::size_t max_input_bytes = 256u * 1024u;            // one text input
    std::size_t max_total_bytes = 2u * 1024u * 1024u;      // all text inputs together
    std::size_t max_input_tokens = 65536u;                 // one token-array input
    std::size_t max_total_tokens = 262144u;                // all token-array inputs together
};

struct EmbedJob {
    std::string model = "default";
    std::vector<std::string> inputs;
    // Set instead of `inputs` when `input` is a token array / array of token arrays.
    std::vector<std::vector<TokenId>> token_inputs;
    std::optional<std::size_t> dimensions;
    EmbedPoolingType pooling_type = EmbedPoolingType::unspecified;
    std::string encoding_format = "float";  // "float" or "base64"
    bool normalize = true;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

struct RerankJob {
    std::string model = "default";
    std::string query;
    std::vector<std::string> documents;
    std::optional<std::size_t> top_n;
    bool return_documents = false;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

struct ScoreJob {
    std::string model = "default";
    std::vector<ScorePair> pairs;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

struct CompletionJob {
    std::string model = "default";
    std::string prompt;
    SamplingConfig sampling;
    bool stream = false;
    bool include_usage = false;
    bool echo = false;
    std::optional<std::string> session_key;
    std::string grammar;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

struct TokenizeJob {
    std::string model = "default";
    std::string content;
};

struct DetokenizeJob {
    std::string model = "default";
    std::vector<TokenId> tokens;
};

// Maps a request body to a ChatJob. Unknown top-level fields are ignored;
// user is accepted and ignored. chat_template_kwargs must be an object:
// enable_thinking (bool) and reasoning_effort (string, 1-64 of
// [A-Za-z0-9._-]) are forwarded, other keys ignored; think (bool) must agree
// with enable_thinking when both are set; prompt_cache_key is a 1-256 byte
// string (invalid_json otherwise). Supports response_format ("text",
// "json_object", "json_schema") and optional grammar (up to 64 KiB).
// Rejected with unsupported_parameter: functions, function_call, n != 1,
// unsupported content part types. Supports tools, tool_choice, tool_calls,
// role "tool" with tool_call_id, logprobs, top_logprobs (0..20), and
// inline base64 data:image/* content blocks (remote URLs rejected for SSRF).
// Sonder extensions: num_ctx, typical_p, repeat_last_n, repeat_penalty,
// top_k, min_p.
std::variant<ChatJob, ApiError> parse_chat_request(std::string_view body);
std::variant<ChatJob, ApiError> parse_chat_request_with_mode(std::string_view body, StructuredParseMode schema_mode);
std::variant<EmbedJob, ApiError> parse_embed_request(std::string_view body);
std::variant<EmbedJob, ApiError> parse_embed_request(std::string_view body, const EmbedRouteLimits& limits);
std::variant<RerankJob, ApiError> parse_rerank_request(std::string_view body);
std::variant<ScoreJob, ApiError> parse_score_request(std::string_view body);
std::variant<CompletionJob, ApiError> parse_completion_request(std::string_view body);
std::variant<TokenizeJob, ApiError> parse_tokenize_request(std::string_view body);
std::variant<DetokenizeJob, ApiError> parse_detokenize_request(std::string_view body);

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
    std::optional<std::int64_t> reasoning_budget_tokens = std::nullopt;
    // X-Sonder-No-Capture: forces the telemetry text-capture policy to off
    // for this request; it can never enable capture.
    bool no_text_capture = false;
    // X-Sonder-Local-Only: with the ollama_cloud backend the server refuses
    // the request before residency (no model load, no network I/O) and the
    // backend refuses it again; other backends ignore it.
    bool local_only = false;
};

// X-Sonder-Run-Id, -Parent-Request-Id, -Agent-Id, -Task-Id, -Workload,
// -Priority. X-Sonder-Priority accepts the legacy integer -16..16 (retained
// in Correlation::priority and mapped to interactive admission) or the
// additive class names interactive, subagent and background.
// X-Sonder-Local-Only accepts 1/true or 0/false (case-insensitive). Invalid values:
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
// when the backend reported a cached-prompt count, and usage.sonder.timings
// when raw llama-server timing fields are present. Existing cached counts win.
json::Object usage_json(const GenerationResult& result);
// prompt_n (uncached evaluated prompt tokens: backend_timings->prompt_n when
// reported, else prompt_tokens minus cached tokens), predicted_n, total_ms,
// plus only measured values: ttft_ms, prompt_ms, predicted_ms,
// prompt_per_second (computed from prompt_n / prompt_ms, omitted when
// prompt_n == 0), predicted_per_second, and the backend's cache_n
// (stats.cached_tokens else backend_timings->cache_n), draft_n,
// draft_n_accepted and the scheduler's queue_ms (scheduled requests only).
// A counter the backend did not report is omitted, never emitted as 0.
json::Object timings_json(const GenerationResult& result);

}  // namespace sonder::inference::server::detail
