// Ollama Cloud wire-format helpers (internal; unit-tested without a network).
//
// Self-contained on purpose: the local `ollama` backend (src/backends/ollama)
// is being retired, so this module does not link against it. The request
// body mapping and the NDJSON decoder follow the same Ollama API contract
// (https://docs.ollama.com/api/chat, https://docs.ollama.com/api/streaming).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference::ollama_cloud {

// Longest NDJSON line the decoder accepts before failing with protocol_error.
inline constexpr std::size_t kMaxNdjsonLineBytes = 4u << 20;
// Bound on a buffered (non-streaming) response body such as /api/tags.
inline constexpr std::size_t kMaxBufferedBodyBytes = 16u << 20;
// Bound on an error response body kept for the error message.
inline constexpr std::size_t kMaxErrorBodyBytes = 8192;

struct BaseUrl {
    std::string host;
    std::uint16_t port = 443;
    std::string path = "/";  // base path, no trailing slash except root
};

// Accepts only https:// (case-insensitive scheme) with an optional port and
// path; no userinfo, query or fragment. Host and path validation is the
// internal HTTP client's (net::parse_url). Works in builds without TLS.
Result<BaseUrl> parse_base_url(std::string_view text);
// base path + api path ("/" + "/api/chat" -> "/api/chat").
std::string join_target(const std::string& base_path, std::string_view api_path);

// SamplingConfig -> Ollama "options" (temperature, top_p, top_k, min_p,
// repeat_penalty, num_ctx, repeat_last_n, presence/frequency penalty, seed,
// num_predict, stop). typical_p and logit_bias are not forwarded.
json::Object sampling_to_options(const SamplingConfig& sampling);

// POST /api/chat body with stream=true. Does not apply the backend's refusals
// (see OllamaCloudModel::chat); it only maps fields.
std::string build_chat_body(const std::string& model, const ChatRequest& request);
// POST /api/generate body with stream=true.
std::string build_generate_body(const std::string& model, const GenerateRequest& request);

enum class StreamKind { generate, chat };

// Redacts secrets in place. Applied to upstream error text BEFORE it is
// bounded for a message, so a secret that straddles the cut is still found
// whole. Empty = no redaction. Called only on error paths.
using TextScrubber = std::function<void(std::string&)>;

struct StreamEvent {
    std::string_view content;   // "response" (generate) or "message.content" (chat)
    std::string_view thinking;  // "thinking" / "message.thinking"
    std::vector<ToolCallDelta> tool_calls;
};

// Values from the final ("done": true) line, exactly as the server sent them.
struct StreamSummary {
    bool done = false;
    std::string done_reason;
    bool saw_tool_calls = false;
    bool has_timings = false;
    std::int64_t total_duration_ns = 0;
    std::int64_t load_duration_ns = 0;
    std::int64_t prompt_eval_count = 0;
    std::optional<std::int64_t> prompt_eval_cached_count;
    std::int64_t prompt_eval_duration_ns = 0;
    std::int64_t eval_count = 0;
    std::int64_t eval_duration_ns = 0;
    std::uint64_t content_events = 0;  // lines that carried content/thinking/tool calls
};

// Incremental NDJSON decoder; bytes may be split at any boundary. Not
// thread-safe (one decoder per response). The callback's string_views are
// valid only during the call. `scrub` redacts an {"error": ...} line's text
// before it is bounded into error().
class StreamDecoder {
public:
    using EventCallback = std::function<bool(const StreamEvent&)>;

    StreamDecoder(StreamKind kind, EventCallback on_event, std::size_t max_output_bytes, TextScrubber scrub = {});

    // False when the callback asked to stop or the stream is invalid.
    bool feed(std::string_view bytes);
    // Flushes a trailing unterminated line. Errors: backend_error (an
    // {"error": ...} line, also mid-stream), protocol_error (malformed JSON,
    // oversized line or output, or no "done" line).
    Status finish();

    [[nodiscard]] const StreamSummary& summary() const noexcept { return summary_; }
    [[nodiscard]] bool stopped_by_callback() const noexcept { return stopped_by_callback_; }
    [[nodiscard]] const Status& error() const noexcept { return error_; }

private:
    bool on_line(std::string_view line);

    StreamKind kind_;
    EventCallback on_event_;
    TextScrubber scrub_;
    std::size_t max_output_bytes_;
    std::size_t output_bytes_ = 0;
    std::uint32_t next_tool_call_index_ = 0;
    std::string buffer_;
    StreamSummary summary_;
    Status error_;
    bool stopped_by_callback_ = false;
};

struct TagEntry {
    std::string name;
    std::string format;
    std::string family;
    std::string parameter_size;
    std::string quantization;
    std::uint64_t size_bytes = 0;
};

// Parses a GET /api/tags document ({"models": [...]}); entries without a
// name are skipped. Errors: protocol_error.
Result<std::vector<TagEntry>> parse_tags(std::string_view body);

// Maps a non-2xx response to a Status. The message carries the upstream
// {"error": ...} text (or the raw body), passed through `scrub` first and
// then bounded to 512 bytes.
Status http_status_error(int status, std::string_view body, const TextScrubber& scrub = {});

}  // namespace sonder::inference::ollama_cloud
