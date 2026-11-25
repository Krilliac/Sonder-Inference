// Internal: HTTP gateway policy, routing table, CORS/Host/auth middleware,
// connection I/O, response framing, and bounded JSON parse admission gate
// (M-17) for the server module.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "http.hpp"
#include "log_queue.hpp"
#include "openai.hpp"
#include "socket.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

using Headers = std::vector<std::pair<std::string, std::string>>;

// Observatory dev (5173), preview (4173) and Tauri origins (contract 2.4).
// Applied to the read-only GET routes only (see origin_allowed()).
const std::vector<std::string>& default_origins();

inline constexpr const char* kAllowHeaders =
    "Accept, Authorization, Cache-Control, Content-Type, Last-Event-ID, X-Sonder-Run-Id, "
    "X-Sonder-Parent-Request-Id, X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority, "
    "X-Sonder-Deadline-Ms, X-Sonder-Reasoning-Budget";
inline constexpr const char* kExposeHeaders = "X-Sonder-Inference-Api, X-Sonder-Request-Id, Retry-After";

enum class State { starting, ready, draining, stopped };

const char* state_name(State s) noexcept;

enum class Route {
    none,
    health,
    models,
    identity,
    sonder_node,
    sonder_plan,
    chat,
    completions,
    responses,
    messages,
    embeddings,
    tokenize,
    detokenize,
    discovery,
    telemetry,
    telemetry_sse,
    telemetry_ndjson
};

// Maximum JSON request body size accepted by POST /v1/sonder/plan.
inline constexpr std::size_t kPlanMaxBodyBytes = 64u * 1024u;

struct RouteInfo {
    Route route = Route::none;
    const char* method = "";  // the one method the route accepts
};

RouteInfo lookup_route(std::string_view path);

// A serialized origin as browsers send it in the Origin header (RFC 6454
// section 6.2): lowercase scheme "://" lowercase host, an optional numeric
// port, and nothing else (no path, query, fragment, userinfo or trailing
// slash). "null" and wildcards are refused: they would match too much.
bool is_serialized_origin(std::string_view origin);

// A bind that can only be reached from this machine. Literal addresses only
// (plus "localhost"): a DNS name starting with "127." may resolve to a LAN
// address, and such a bind must require a token like any other.
bool is_loopback_bind(std::string_view host);

// Origin policy (contract 2.4 with the review correction): explicit
// --cors-origin entries are allowed everywhere; the default allowlist only
// for read-only GET requests unless a token protects the server.
bool origin_allowed(const ServerOptions& opts, const std::string& origin, bool read_only);

double seconds_since(std::chrono::steady_clock::time_point t);

// Per-connection bookkeeping shared between the accept loop and the
// connection thread.
struct ConnFlags {
    std::atomic<bool> done{false};
    std::atomic<bool> streaming{false};  // telemetry stream: no engine access
    // Still reading the request head (or idle between keep-alive requests):
    // nothing active is in flight, so the connection may be evicted when the
    // server is at its limit.
    std::atomic<bool> in_head{true};
    std::atomic<bool> evict{false};  // set by the accept thread; read_head / idle wait gives up
    std::atomic<std::int64_t> head_started_ns{
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count()};
};

// A connection still sending its request head after this long may be evicted
// to make room for a new one when every slot is taken. Legitimate clients send
// their head in milliseconds; one that dribbles it cannot hold a slot against
// fresh clients (slowloris).
inline constexpr std::chrono::milliseconds kEvictableHeadAge{500};

// One request/response exchange on a connection.
struct Exchange {
    native_socket sock{};
    std::chrono::steady_clock::time_point started;
    std::string request_id;
    std::string method = "-";
    std::string path = "-";
    int status = 0;
    Headers cors;  // set once the Origin is known to be allowed
    // Request bytes the client may still be sending when the response goes
    // out (an unread body): the connection lingers to drain them.
    std::uint64_t unread_input = 0;
    // Keep-alive state for this exchange (controls Connection / Keep-Alive headers).
    bool keep_alive = false;
    std::chrono::milliseconds keep_alive_timeout{0};
    std::size_t keep_alive_remaining = 0;
    bool write_failed = false;
};

// True when any Connection header token (case-insensitive, comma-separated) is "close".
bool client_wants_close(const RequestHead& head);

// Backend reachability, refreshed off the request path (health and identity
// only read it). Shared with the refresher thread, which may outlive the
// Server when a probe is stuck in backend I/O at shutdown.
struct ProbeCache {
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false;
    bool exited = false;
    bool valid = false;
    bool ok = false;
    std::string version;
};

// How often the refresher re-probes the backend.
inline constexpr std::chrono::seconds kProbeInterval{2};
// Lingering close bounds (see drain_input): enough for a body the server
// refused with 401/403/413 to finish arriving on a local link.
inline constexpr std::uint64_t kLingerMaxBytes = 16u * 1024u * 1024u;
inline constexpr std::uint64_t kLingerUnknownLength = 1024u * 1024u;
inline constexpr std::chrono::milliseconds kLingerMaxTime{2000};
inline constexpr std::chrono::milliseconds kLingerIdle{500};
// Descriptors the process needs beyond one per connection (listener,
// telemetry file, backend clients, stdio, ...), for the RLIMIT_NOFILE check.
inline constexpr std::uint64_t kDescriptorHeadroom = 32;

// Bounded concurrent JSON parse admission gate (M-17): caps both concurrent
// parsers and aggregate in-flight raw body bytes before allocating DOM nodes.
struct ParseGate {
    std::mutex mu;
    std::condition_variable cv;
    std::size_t active = 0;
    std::size_t active_bytes = 0;
    std::size_t max_active = 4;
    std::size_t max_bytes = 16u << 20;
    bool closed = false;

    struct Guard {
        ParseGate* gate = nullptr;
        std::size_t bytes = 0;
        Guard() = default;
        Guard(ParseGate* g, std::size_t b) noexcept : gate(g), bytes(b) {}
        ~Guard() {
            if (gate != nullptr) {
                gate->release(bytes);
            }
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&& o) noexcept : gate(o.gate), bytes(o.bytes) {
            o.gate = nullptr;
            o.bytes = 0;
        }
        Guard& operator=(Guard&& o) noexcept {
            if (this != &o) {
                if (gate != nullptr) {
                    gate->release(bytes);
                }
                gate = o.gate;
                bytes = o.bytes;
                o.gate = nullptr;
                o.bytes = 0;
            }
            return *this;
        }
        explicit operator bool() const noexcept { return gate != nullptr; }
    };

    Guard acquire(std::size_t bytes, std::chrono::milliseconds timeout) {
        const std::size_t charge = std::min(bytes, max_bytes);
        std::unique_lock<std::mutex> lock(mu);
        const bool ok = cv.wait_for(lock, timeout, [&] {
            return closed || (active < max_active && active_bytes + charge <= max_bytes);
        });
        if (!ok || closed) {
            return {};
        }
        ++active;
        active_bytes += charge;
        return Guard{this, charge};
    }

    void release(std::size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (active > 0) --active;
            active_bytes = active_bytes >= bytes ? active_bytes - bytes : 0;
        }
        cv.notify_all();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu);
            closed = true;
        }
        cv.notify_all();
    }
};

// Text of the "lines dropped" report the log queue writes when it had to
// drop access-log or diagnostic lines, in the configured log format.
LogQueue::DropLine log_drop_reporter(bool as_json);
std::string format_log_message(LogFormat format, const char* level, const std::string& message);
std::string format_access_log(LogFormat format, const Exchange& ex);

// HTTP response helpers.
Headers base_headers(const Exchange& ex, const std::string& content_type);
bool send_raw(Exchange& ex, std::string_view data, std::chrono::milliseconds write_stall_timeout,
              const std::atomic<bool>& hard_stop);
void send_response(Exchange& ex, int status, const std::string& content_type, const std::string& body,
                   std::chrono::milliseconds write_stall_timeout, const std::atomic<bool>& hard_stop,
                   const Headers& extra = {});
void send_json(Exchange& ex, int status, const json::Object& body, std::chrono::milliseconds write_stall_timeout,
               const std::atomic<bool>& hard_stop, const Headers& extra = {});
void send_error(Exchange& ex, const ApiError& error, std::chrono::milliseconds write_stall_timeout,
                const std::atomic<bool>& hard_stop, Headers extra = {});

// Connection read & lingering-close helpers.
bool read_head(Exchange& ex, std::string& buffer, RequestHead& head, std::size_t& head_bytes, const ConnFlags& flags,
               const ServerOptions& opts, const std::atomic<bool>& stopping, const std::atomic<bool>& hard_stop);
bool read_body(Exchange& ex, std::string& buffer, std::size_t head_bytes, std::uint64_t length, std::string& body,
               const ServerOptions& opts, const std::atomic<bool>& hard_stop);
void linger(const Exchange& ex, const std::atomic<bool>& hard_stop);

struct PreDispatchResult {
    bool proceed = false;
    Route route = Route::none;
    std::string body;
};

// Enforces all uniform pre-dispatch invariants in exact order:
// loopback Host check -> CORS allowlist & OPTIONS preflight -> uniform
// constant-time digest auth -> route/method check -> Expect: 100-continue ->
// POST Content-Length/size check and body read.
PreDispatchResult run_request_guards(Exchange& ex, const RequestHead& head, std::string& buffer, std::size_t head_bytes,
                                     const ServerOptions& opts, bool loopback, bool has_token_digest,
                                     const std::array<std::uint8_t, 32>& token_digest,
                                     const std::atomic<bool>& hard_stop);

// Lingering-close 503 overloaded response sent directly on the accept thread.
Exchange reject_overloaded_socket(Socket& client, const std::string& reason, const std::atomic<bool>& hard_stop);

// Builds the local URL ("http://<host>:<port>") matching the bound listener.
std::string format_listen_url(const std::string& raw_host, std::uint16_t port, bool ipv6);

}  // namespace sonder::inference::server::detail
