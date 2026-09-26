// Sonder Inference: Ollama compatibility backend (Backend 0, docs/BACKENDS.md).
//
// Talks to an Ollama server over its HTTP API:
//   * POST /api/generate and /api/chat, NDJSON streaming
//   * cancellation through sonder::inference::CancellationToken
//   * GET /api/tags, POST /api/show, GET /api/ps, GET /api/version
//   * server timing fields (load/prompt_eval/eval durations and counts) plus
//     client-measured TTFB/TTFT/wall time, mapped to GenerateStats and to
//     Observatory telemetry attributes.
//
// Available when the build defines SONDER_HAS_OLLAMA_BACKEND. Transport is the
// core internal HTTP client (src/net), which is plain HTTP; non-loopback hosts
// are refused unless allow_remote is set (policy: never plain-HTTP remote
// workers). TLS endpoints are not supported yet.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/sampling.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference {

// ---------------------------------------------------------------------------
// Ollama compatibility backend (docs/BACKENDS.md, "Backend 0"). Talks to an
// Ollama server over plain HTTP. Loopback only unless allow_remote is set;
// remote workers should sit behind TLS, which this adapter does not speak.
// ---------------------------------------------------------------------------
struct OllamaBackendOptions {
    std::string base_url = "http://127.0.0.1:11434";
    bool allow_remote = false;
    std::chrono::milliseconds connect_timeout{3000};
    // Whole-request budget (covers cold model load + long prefill + decode).
    std::chrono::milliseconds request_timeout{300000};
    std::string keep_alive;  // forwarded as keep_alive when non-empty, e.g. "5m"
    // Also deliver reasoning ("thinking") text to Backend token callbacks.
    // Off by default, so TTFT measures the first *content* token.
    bool emit_thinking_chunks = false;
};
inline constexpr const char* kOllamaBackendName = "ollama";
std::shared_ptr<Backend> make_ollama_backend(OllamaBackendOptions options = {});

}  // namespace sonder::inference

namespace sonder::inference::ollama {

using OllamaConfig = OllamaBackendOptions;

// Server-reported timings exactly as Ollama returns them in the final
// ("done": true) chunk, plus client-side steady-clock measurements.
struct OllamaTimings {
    std::int64_t total_duration_ns = 0;
    std::int64_t load_duration_ns = 0;
    std::int64_t prompt_eval_count = 0;
    std::int64_t prompt_eval_duration_ns = 0;
    std::int64_t eval_count = 0;
    std::int64_t eval_duration_ns = 0;
    bool has_server_timings = false;

    double ttfb_ms = -1.0;  // first response body byte
    double ttft_ms = -1.0;  // first non-empty content/thinking chunk
    double wall_ms = 0.0;   // request start -> stream end
    std::int64_t content_chunks = 0;

    [[nodiscard]] double prompt_tokens_per_sec() const noexcept;
    [[nodiscard]] double decode_tokens_per_sec() const noexcept;
};

enum class StreamKind { generate, chat };

struct StreamChunk {
    std::string_view content;   // "response" or "message.content"
    std::string_view thinking;  // reasoning text when the model emits it
    bool done = false;
    std::string_view done_reason;
    const json::Value* raw = nullptr;  // full parsed line (valid during callback)
};

// Return false to stop the stream early (StreamResult::stopped_by_callback).
using ChunkCallback = std::function<bool(const StreamChunk&)>;

struct StreamResult {
    std::string model;
    std::string text;
    std::string thinking;
    std::string done_reason;
    OllamaTimings timings;
    int http_status = 0;
    bool done = false;                 // saw the final "done": true chunk
    bool stopped_by_callback = false;  // callback returned false
};

// Incremental NDJSON stream decoder. Bytes may be split at any boundary.
// Used by OllamaClient and directly by fixture tests.
class StreamDecoder {
public:
    StreamDecoder(StreamKind kind, ChunkCallback on_chunk,
                  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now());

    // Returns false when the callback asked to stop or a hard error occurred.
    bool feed(std::string_view bytes);
    // Flushes a trailing unterminated line and validates completion.
    // Errors: backend_error ({"error": ...} line), protocol_error (malformed
    // JSON or stream ended without "done").
    Status finish();

    [[nodiscard]] const StreamResult& result() const noexcept { return result_; }
    [[nodiscard]] StreamResult& result() noexcept { return result_; }
    [[nodiscard]] const Status& error() const noexcept { return error_; }

private:
    bool on_line(std::string_view line);

    StreamKind kind_;
    ChunkCallback on_chunk_;
    std::chrono::steady_clock::time_point start_;
    std::string buffer_;
    StreamResult result_;
    Status error_;
};

struct ChatMessage {
    std::string role;  // system | user | assistant | tool
    std::string content;
};

struct GenerateParams {
    std::string model;
    std::string prompt;
    std::string system;
    bool raw = false;
    bool stream = true;
    std::optional<bool> think;
    std::string keep_alive;         // overrides OllamaConfig::keep_alive
    json::Object options;           // Ollama "options" (num_predict, seed, ...)
};

struct ChatParams {
    std::string model;
    std::vector<ChatMessage> messages;
    bool stream = true;
    std::optional<bool> think;
    std::string keep_alive;
    json::Object options;
};

struct OllamaModelInfo {
    std::string name;
    std::string digest;
    std::uint64_t size_bytes = 0;
    std::string format;
    std::string family;
    std::string parameter_size;
    std::string quantization_level;
    std::string modified_at;
};

struct RunningModel {
    std::string name;
    std::uint64_t size_bytes = 0;
    std::uint64_t size_vram_bytes = 0;
    std::string expires_at;
};

// SamplingConfig -> Ollama "options" (temperature, top_p, top_k, min_p,
// repeat_penalty, seed, num_predict, stop).
json::Object sampling_to_options(const SamplingConfig& sampling);
json::Value build_generate_body(const GenerateParams& params, const OllamaConfig& config = {});
json::Value build_chat_body(const ChatParams& params, const OllamaConfig& config = {});
// Applies timing fields from a "done": true chunk.
void apply_server_timings(const json::Value& done_chunk, OllamaTimings& timings);

class OllamaClient {
public:
    explicit OllamaClient(OllamaConfig config = {});

    [[nodiscard]] const OllamaConfig& config() const noexcept { return config_; }

    Result<std::string> version() const;
    Result<std::vector<OllamaModelInfo>> list_models() const;
    Result<std::vector<RunningModel>> running_models() const;
    // Raw /api/show document (details, model_info, parameters, template, ...).
    Result<json::Value> show_model(const std::string& name) const;

    // Thread-safe; each call opens its own connection. A tripped `cancel`
    // yields ErrorCode::cancelled. Partial output is delivered via callbacks.
    Result<StreamResult> generate(const GenerateParams& params, const ChunkCallback& on_chunk = {},
                                  const CancellationToken& cancel = {}) const;
    Result<StreamResult> chat(const ChatParams& params, const ChunkCallback& on_chunk = {},
                              const CancellationToken& cancel = {}) const;

private:
    Result<json::Value> get_json(const std::string& method, const std::string& path, const std::string& body) const;
    Result<StreamResult> stream(StreamKind kind, const std::string& path, const std::string& body,
                                const ChunkCallback& on_chunk, const CancellationToken& cancel) const;

    OllamaConfig config_;
};

// Telemetry: flat attributes for an Ollama request (only fields the server
// actually reported; nothing is synthesized).
json::Object timing_attributes(const OllamaTimings& timings);
// Emits backend.model.load.reported (when load_duration > 0; Ollama reports
// it even for warm models, so it is not a residency transition),
// backend.timing.prefill and backend.timing.decode. The names are distinct
// from the session's inference.* events so nothing is duplicated. Returns the
// number of events accepted by the bus.
int emit_timing_events(TelemetryBus& bus, const TelemetryContext& context, const OllamaTimings& timings,
                       std::string_view model);

}  // namespace sonder::inference::ollama
