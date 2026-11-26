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
// bounded HTTP/1.1 keep-alive (configurable via ServerOptions).
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference {
class Engine;
}

namespace sonder::inference::server {

// HTTP API major version (X-Sonder-Inference-Api header, api_version fields).
inline constexpr int kApiVersion = 1;
inline constexpr std::uint16_t kDefaultPort = 11437;

enum class LogFormat { text, json };
enum class SchedulerPolicy { automatic, gate, account, off };
// Controls whether server-wide thinking pins override explicit request values
// (the historical default) or only supply values that are absent.
enum class PinMode { override_request, default_value };
// Controls the server-side class/capacity admission layer. `automatic` only
// enables it when an explicit class or queue cap is configured.
enum class PriorityAdmissionPolicy { automatic, on, off };

// Per-tier (VRAM or host RAM) memory footprint breakdown for one served model
// (Item #18: weights, KV/recurrent cache, compute/scratch workspaces,
// multimodal projectors, and adapters accounted separately).
struct ResourceFootprint {
    std::uint64_t weights_bytes = 0;
    std::uint64_t kv_cache_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t projector_bytes = 0;
    std::uint64_t adapter_bytes = 0;

    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        const auto sadd = [](std::uint64_t a, std::uint64_t b) noexcept {
            return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max()
                                                                     : a + b;
        };
        return sadd(sadd(sadd(weights_bytes, kv_cache_bytes), sadd(workspace_bytes, projector_bytes)), adapter_bytes);
    }
    [[nodiscard]] bool empty() const noexcept { return total_bytes() == 0; }
};

struct ModelResourceFootprint {
    ResourceFootprint vram;
    ResourceFootprint ram;

    [[nodiscard]] std::uint64_t vram_bytes() const noexcept { return vram.total_bytes(); }
    [[nodiscard]] std::uint64_t ram_bytes() const noexcept { return ram.total_bytes(); }
    [[nodiscard]] bool empty() const noexcept { return vram.empty() && ram.empty(); }
};

// A served model backed by a launch profile (launch_profile.hpp).
struct ModelProfileBinding {
    std::string model;      // served id; must also be listed in ServerOptions::models
    std::string load_name;  // passed to Backend::load_model (empty = `model`)
    // Applied to requests that leave these fields unset (apply_sampling_defaults).
    SamplingDefaults sampling_defaults;
    // Added to the model's /v1/models entry as `sonder.profile`.
    json::Object metadata;
    // Pre-load VRAM/RAM byte estimate derived from vram_estimate /
    // coresidency_planner for live residency reservation.
    ModelResourceFootprint estimated_footprint;
};

struct ServerOptions {
    // Listen address. Non-loopback binds require `token`.
    std::string host = "127.0.0.1";
    // 0 picks an ephemeral port (see Server::port()).
    std::uint16_t port = kDefaultPort;

    // Execution backend and the model ids served on it. The first model is
    // the default and also answers to the alias "default". With the mock
    // backend an empty list serves the model "mock". `backend.backend` must
    // name a backend compiled into this build (available_backend_names()).
    BackendSetup backend;
    std::vector<std::string> models;
    // Device every model is loaded on (e.g. "gpu:0" for llama.cpp GPU
    // offload); empty uses the backend default (CPU for llama.cpp).
    std::string device;
    // Embedding hosts and tests: a backend built by the caller, used instead
    // of make_backend(backend). Its name() decides the synthetic label.
    std::shared_ptr<Backend> backend_instance;
    // Launch profiles (docs/integration/launch-profiles.md). Without bindings
    // every model loads under its own id with no sampling defaults.
    std::vector<ModelProfileBinding> profile_bindings;
    // Every profile of the loaded file, listed additively as `sonder.profiles`
    // in the /v1/models response; empty leaves that field out.
    json::Array profile_catalog;

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
    std::chrono::milliseconds stream_keepalive_interval{15000};  // SSE `: keep-alive` during prefill/idle (0 = off)
    std::chrono::milliseconds write_stall_timeout{60000};   // peer not reading at all
    // Bounded HTTP/1.1 persistent connections (keep-alive).
    bool keep_alive = true;
    std::chrono::milliseconds keep_alive_timeout{15000};
    std::size_t max_keep_alive_requests = 256;

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
    //
    // Lines are queued and `log` is called on one logger thread, one call at
    // a time, in the order the server produced them. It is never called on a
    // request thread or the accept thread, so a slow or blocked sink costs
    // log lines, never requests: at most `log_queue_capacity` lines wait, and
    // while the queue is full new lines are dropped and counted (health
    // `log.dropped`, plus a "lines dropped" line once the sink takes lines
    // again). stop() gives the queue max(shutdown_grace, 100 ms) to drain. A
    // call that has not returned by then is abandoned: the lines behind it
    // are dropped, `log` is not entered again, but that one call may still be
    // running after stop() returns and after the Server is destroyed.
    // Whatever `log` uses must stay valid until the call returns (own it in
    // the callable, for example through a shared_ptr).
    std::function<void(const std::string& line)> log;
    LogFormat log_format = LogFormat::text;
    // Lines the log queue holds before it drops new ones; at least 1.
    std::size_t log_queue_capacity = 4096;

    // Model residency (docs/SERVER.md "Model residency"). The defaults keep
    // the behaviour from before these options existed: every model loads
    // during start() and stays loaded until stop().
    //
    // Register the models during start() without loading them (no backend
    // I/O) and load each on its first request. Concurrent first requests
    // share one load. A model that cannot load then fails its requests (404
    // model_not_found or 503 not_ready) instead of the start.
    bool lazy_models = false;
    // Unload a model once no request has used it for this long; the next
    // request loads it again. A request in flight pins its model. 0 = never.
    std::chrono::milliseconds model_idle_ttl{0};
    // Keep at most this many models loaded, evicting the least recently used
    // unpinned one first. Soft: pinned models are never evicted, so requests
    // in flight can exceed it until they finish. With eager loading only the
    // first N models load during start(). 0 = no cap.
    std::size_t max_resident_models = 0;
    // Byte budgets for live model residency (0 = unlimited). Pre-load byte
    // reservations are reconciled against observed allocations after load and
    // released immediately on error; unpinned victims are chosen by
    // (demand_score * reload_ms) / max(1, bytes_freed). Pinned models are
    // never evicted.
    std::uint64_t max_resident_vram_bytes = 0;
    std::uint64_t max_resident_ram_bytes = 0;
    // Request path (docs/SERVER.md "Scheduling" and "Thinking control").
    // --scheduler: automatic gates in-process backends per token and only
    // admits/accounts remote-process ones (llamaserver, ollama); off disables
    // the engine scheduler. --kv-pool-tokens: logical KV pool (0 = 65,536).
    SchedulerPolicy scheduler = SchedulerPolicy::automatic;
    std::uint64_t kv_pool_tokens = 0;
    // --pin-enable-thinking / --pin-reasoning-effort: forwarded on every chat
    // request; pin_mode controls whether conflicts are overridden or left
    // intact. Budget pins are default-only and applied by request glue.
    std::optional<bool> pin_enable_thinking;
    std::optional<std::string> pin_reasoning_effort;
    PinMode pin_mode = PinMode::override_request;
    std::optional<std::int64_t> pin_reasoning_budget_tokens;
    std::optional<std::string> pin_reasoning_budget_message;

    // Request-class admission limits, applied only when priority admission is
    // enabled. Zero means unlimited; engine scheduling is independent.
    std::size_t max_concurrent_subagent = 0;
    std::size_t max_concurrent_background = 0;
    std::size_t max_queue_per_class = 0;
    // 0 uses the backend's advertised capacity; unknown stays ungated.
    // A nonzero override applies only when priority admission is enabled.
    std::size_t backend_capacity = 0;
    PriorityAdmissionPolicy priority_admission = PriorityAdmissionPolicy::automatic;

    // Stateful /v1/responses previous_response_id history store controls.
    // When disable_response_store is true (or response_history_ttl is 0),
    // /v1/responses does not retain turns in memory and rejects
    // previous_response_id chaining.
    bool disable_response_store = false;
    std::chrono::milliseconds response_history_ttl{std::chrono::hours(1)};
};

// Checks option combinations that the server refuses to start with
// (non-loopback bind without a token, --capture-text without a token, a
// backend this build does not have, empty model ids, zero limits, a
// --cors-origin that is not a serialized origin). Returns invalid_argument
// naming the option.
Status validate_options(const ServerOptions& options);

// serve's launch-profile step (exposed so tests can drive the wiring): builds
// the `sonder.profiles` catalog, binds `active` into `options`, runs the VRAM
// fit check and, for an active llamaserver profile with or without GPU
// offload, copies the GGUF host-RAM estimate into
// options.backend.llamaserver_host_memory. On refusal `refusal` is non-empty.
json::Array profile_catalog(const std::vector<LaunchProfile>& profiles, const LaunchProfile& active,
                            ServerOptions& options, std::optional<std::uint64_t> budget_flag, bool allow_overcommit,
                            std::vector<std::string>& notes, std::string& refusal);

class Server {
public:
    explicit Server(ServerOptions options);
    ~Server();  // stop() + wait()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binds the socket, builds the engine (engine.started carries the listen
    // address), starts accepting (health reports "starting"), registers the
    // backend and the models and loads them (all of them unless
    // `lazy_models` or `max_resident_models` says otherwise), then reports
    // "ready". `on_listening`
    // runs once the socket accepts connections, before models load. No lock
    // is held across backend I/O: health answers 503 "starting" while a
    // model loads. Errors: invalid_argument (options), unavailable (address
    // in use or bind failure), model load errors, and cancelled when stop()
    // ran before start() finished.
    Status start(const std::function<void()>& on_listening = {});

    // Graceful drain (docs/SERVER.md "Shutdown"): health turns 503 draining,
    // the listener closes, in-flight requests get `shutdown_grace` to finish
    // and are then cancelled, the engine stops (engine.stopped and a final
    // telemetry.dropped if needed), and telemetry streams end after
    // delivering what they hold. Idempotent and safe from any thread,
    // including while start() runs (also from its on_listening callback):
    // start() then returns cancelled after the startup step in progress,
    // whose backend call (a model load, for example) stop() waits for.
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
