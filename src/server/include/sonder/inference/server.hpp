// Sonder Inference: `sonder-infer serve`, a local HTTP/1.1 server (ADR-020).
//
// Serves an OpenAI-compatible chat subset plus Sonder extensions (health,
// models, backend identity) and the live Observatory telemetry streams
// (SSE and NDJSON, /.well-known/sonder-telemetry discovery). The API is
// versioned as api_version 1 under /v1; docs/SERVER.md is the reference.
//
// Available when the build defines SONDER_HAS_SERVER (module src/server).
// The server is written in-house over POSIX sockets / Winsock: no TLS (put a
// TLS-terminating proxy in front for remote use), one thread per connection,
// Connection: close on every response.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference {
class Engine;
}

namespace sonder::inference::server {

// HTTP API major version (X-Sonder-Inference-Api header, api_version fields).
inline constexpr int kApiVersion = 1;
inline constexpr std::uint16_t kDefaultPort = 11437;

enum class LogFormat { text, json };

struct ServerOptions {
    // Listen address. Non-loopback binds require `token`.
    std::string host = "127.0.0.1";
    // 0 picks an ephemeral port (see Server::port()).
    std::uint16_t port = kDefaultPort;

    // Execution backend and the model ids served on it. The first model is
    // the default and also answers to the alias "default". With the mock
    // backend an empty list serves the model "mock".
    BackendSetup backend;
    std::vector<std::string> models;

    // Bearer token required on every route except CORS preflight. Never
    // logged. Empty = no authentication (loopback binds only).
    std::string token;

    // CORS: exact-match origin allowlist. The defaults (Observatory dev,
    // preview and Tauri origins) apply to the read-only GET routes only; POST
    // routes need an origin from `cors_origins` or a configured token.
    std::vector<std::string> cors_origins;
    bool default_cors = true;

    // Limits (docs/SERVER.md "Limits").
    std::size_t max_connections = 64;
    std::size_t max_body_bytes = 4u * 1024u * 1024u;
    std::chrono::milliseconds read_timeout{10000};          // headers and body (408)
    std::chrono::milliseconds shutdown_grace{5000};         // in-flight requests on stop()
    std::chrono::milliseconds heartbeat_interval{15000};    // idle telemetry streams
    std::chrono::milliseconds write_stall_timeout{60000};   // peer not reading at all

    // Telemetry. The live hub is always attached; `extra_sinks` (e.g. the
    // --telemetry JSONL file) receive the same events.
    TelemetryLevel telemetry_level = TelemetryLevel::standard;
    std::size_t telemetry_buffer = 8192;      // retained ring, events
    std::size_t max_subscribers = 8;          // concurrent telemetry streams
    std::size_t subscriber_queue = 0;         // per-subscriber queue; 0 = telemetry_buffer
    bool capture_text = false;                // requires `token`
    std::vector<std::shared_ptr<TelemetrySink>> extra_sinks;

    // Access log and diagnostics (one line per request; never bodies or
    // header values). Null discards them.
    std::function<void(const std::string& line)> log;
    LogFormat log_format = LogFormat::text;
};

// Checks option combinations that the server refuses to start with
// (non-loopback bind without a token, --capture-text without a token, empty
// model ids, zero limits). Returns invalid_argument naming the option.
Status validate_options(const ServerOptions& options);

class Server {
public:
    explicit Server(ServerOptions options);
    ~Server();  // stop() + wait()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binds the socket, builds the engine (engine.started carries the listen
    // address), starts accepting (health reports "starting"), registers the
    // backend and loads the models, then reports "ready". `on_listening`
    // runs once the socket accepts connections, before models load.
    // Errors: invalid_argument (options), unavailable (address in use or
    // bind failure), and model load errors.
    Status start(const std::function<void()>& on_listening = {});

    // Graceful drain (docs/SERVER.md "Shutdown"): health turns 503 draining,
    // the listener closes, in-flight requests get `shutdown_grace` to finish
    // and are then cancelled, the engine stops (engine.stopped and a final
    // telemetry.dropped if needed), and telemetry streams end after
    // delivering what they hold. Idempotent and safe from any thread.
    void stop();
    // Blocks until stop() has completed.
    void wait();

    [[nodiscard]] std::uint16_t port() const noexcept;
    // "http://127.0.0.1:<port>" (IPv6 hosts in brackets).
    [[nodiscard]] std::string url() const;
    // Telemetry stream instance id (event_id prefix); empty before start().
    [[nodiscard]] std::string instance_id() const;
    // The hosted engine (null before start() or after stop()).
    [[nodiscard]] Engine* engine() noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// `sonder-infer serve [options]`: `args` excludes the program name and the
// word "serve". Handles --help itself. Blocks until request_shutdown() (or
// SIGINT/SIGTERM, whose handlers it installs while running). Returns 0 on a
// clean shutdown, 1 on a runtime error (e.g. port in use), 2 on a usage error.
int serve_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

// Shutdown hook for signal handlers and tests: asks the running serve_main()
// to drain and return. Async-signal-safe. serve_main() clears a pending
// request when it starts.
void request_shutdown() noexcept;

}  // namespace sonder::inference::server
