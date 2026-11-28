// Sonder Inference: Ollama Cloud provider (docs/integration/ollama-cloud.md).
//
// Serves Ollama's hosted cloud models directly over HTTPS with no local
// Ollama daemon. Wire protocol: Ollama's native API at the configured base
// URL (default https://ollama.com):
//   * POST /api/chat and /api/generate, NDJSON streaming
//   * GET /api/tags (model list; also the reachability probe)
//   * Authorization: Bearer <API key>
//
// Policy:
//   * Opt-in. A model is a cloud model only when it is listed in
//     OllamaCloudOptions::models; anything else is refused at load_model().
//   * Requests flagged local-only (GenerateRequest::local_only,
//     ChatRequest::local_only) are refused before any network I/O.
//   * The API key comes from an environment variable name or a token file
//     path, never from a literal. It is never logged, never part of an error
//     message or telemetry attribute, and redacted_config() leaves it out.
//   * The key goes only to the Ollama Cloud origin (https://ollama.com:443)
//     unless allow_custom_endpoint names another host as trusted with it.
//     The configured provider is trusted with its own credential: successful
//     output is not filtered for it. Remote error bodies are never read or
//     propagated; errors carry the HTTP status and a fixed category.
//   * Certificate verification is always on (chain + host name against the
//     system trust store, or against ca_bundle_path). There is no option to
//     disable it and no pin-only mode.
//
// Available when the build defines SONDER_HAS_OLLAMA_CLOUD_BACKEND. Real
// HTTPS needs a SONDER_WITH_TLS=ON build (Schannel on Windows, OpenSSL
// elsewhere; docs/integration/tls.md). Without TLS, the default transport
// fails every request with ErrorCode::unsupported.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference {

inline constexpr const char* kOllamaCloudBackendName = "ollama_cloud";
inline constexpr const char* kOllamaCloudDefaultBaseUrl = "https://ollama.com";
// The canonical Ollama Cloud origin: the only host the API key is sent to
// unless OllamaCloudOptions::allow_custom_endpoint is set.
inline constexpr const char* kOllamaCloudCanonicalHost = "ollama.com";
inline constexpr std::uint16_t kOllamaCloudCanonicalPort = 443;
// Ceiling (and default) for OllamaCloudOptions::max_response_bytes.
inline constexpr std::size_t kOllamaCloudMaxResponseBytes = std::size_t{64} << 20;

// One HTTPS request as handed to an OllamaCloudTransport. `headers` carries
// the Authorization header, so a transport must never log header values.
struct OllamaCloudHttpRequest {
    std::string method;  // "GET" or "POST"
    std::string host;
    std::uint16_t port = 443;
    std::string target;  // origin-form, base path included (e.g. "/api/chat")
    std::string body;    // JSON for POST, empty for GET
    std::vector<std::pair<std::string, std::string>> headers;
    std::chrono::milliseconds connect_timeout{10000};
    std::chrono::milliseconds total_timeout{900000};
    // Maximum silence between body bytes once the response head arrived (0 = off).
    std::chrono::milliseconds idle_timeout{0};
    // PEM CA bundle that REPLACES the system trust store (it does not add to
    // it); empty = system trust store. Verification stays on.
    std::string ca_bundle_path;
    // Cap on all bytes received for this response (status line, headers,
    // framing and body), counted as they arrive; exceeding it fails the
    // request with protocol_error. Always set by the backend (never 0).
    std::size_t max_response_bytes = kOllamaCloudMaxResponseBytes;
};

// Pluggable transport. The default (make_ollama_cloud_https_transport) is the
// internal HTTP client over verified TLS; tests inject a fake. Implementations
// must be safe to call concurrently from several threads, call on_status once
// before any body bytes, stop when on_body returns false, poll `cancel` while
// waiting and then return ErrorCode::cancelled.
class OllamaCloudTransport {
public:
    virtual ~OllamaCloudTransport() = default;
    virtual Status send(const OllamaCloudHttpRequest& request, const std::function<void(int)>& on_status,
                        const std::function<bool(std::string_view)>& on_body, const CancellationToken& cancel) = 0;
};

// HTTPS over src/net with insecure_skip_verify forced off and no pins.
std::shared_ptr<OllamaCloudTransport> make_ollama_cloud_https_transport();

struct OllamaCloudOptions {
    // Must be https://. A path prefix is kept (e.g. https://host/prefix).
    // Without allow_custom_endpoint only the Ollama Cloud origin
    // (https://ollama.com, port 443) is accepted.
    std::string base_url = kOllamaCloudDefaultBaseUrl;
    // Accept a base_url on another host or port. The API key is sent to
    // that host on every request, so set this only for a host you trust
    // with the key (docs/integration/ollama-cloud.md, "Endpoint").
    bool allow_custom_endpoint = false;
    // Exactly one key source: the NAME of an environment variable (for
    // example OLLAMA_API_KEY) or the path of a file that holds only the key.
    std::string api_key_env;
    std::string api_key_file;
    // The cloud allowlist: model names exactly as GET /api/tags on the cloud
    // returns them (e.g. "gpt-oss:120b"). Empty means no model is cloud, so
    // every load_model() is refused.
    std::vector<std::string> models;
    std::chrono::milliseconds connect_timeout{10000};
    std::chrono::milliseconds request_timeout{900000};
    std::chrono::milliseconds idle_timeout{120000};  // 0 disables
    // PEM CA bundle used INSTEAD of the system trust store (net::TlsOptions);
    // a bundle that lacks ollama.com's roots makes https://ollama.com fail
    // verification. Empty (the default) = system trust store.
    std::string ca_bundle_path;
    // Also deliver thinking text to token callbacks when the caller did not
    // ask for separate reasoning (same meaning as the ollama backend option).
    bool emit_thinking_chunks = false;
    // Cumulative cap on streamed content + thinking + tool-call bytes (0 = off).
    std::size_t max_stream_output_bytes = 32u << 20;
    // Cap on the raw bytes of one response (head, framing and body, counted
    // as received, whether or not they carry output). Cannot be disabled:
    // 1..kOllamaCloudMaxResponseBytes, default the maximum (64 MiB).
    std::size_t max_response_bytes = kOllamaCloudMaxResponseBytes;
    // Null = make_ollama_cloud_https_transport().
    std::shared_ptr<OllamaCloudTransport> transport;
    // Environment lookup for api_key_env; null = the process environment.
    std::function<std::optional<std::string>(const char* name)> env_lookup;
};

// Configuration summary for logs, health and config dumps. Holds the base
// URL, key source kind and name ("env" + variable name, or "file" + path),
// models and timeouts. Never the key itself.
json::Object redacted_config(const OllamaCloudOptions& options);

// Validates the options and reads the key once. Errors: invalid_argument
// (bad URL, not https, no or two key sources, unreadable/empty/invalid key,
// empty or duplicate model names); the message never contains the key.
Result<std::shared_ptr<Backend>> make_ollama_cloud_backend(OllamaCloudOptions options);

}  // namespace sonder::inference
