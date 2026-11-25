#include "server_policy.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>

#include "../../common/loopback.hpp"
#include "anthropic.hpp"
#include "request_path.hpp"

namespace sonder::inference::server::detail {

const std::vector<std::string>& default_origins() {
    static const std::vector<std::string> kOrigins = {
        "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173",
        "http://localhost:4173", "tauri://localhost",     "http://tauri.localhost"};
    return kOrigins;
}

const char* state_name(State s) noexcept {
    switch (s) {
        case State::starting: return "starting";
        case State::ready: return "ready";
        case State::draining: return "draining";
        case State::stopped: return "draining";
    }
    return "draining";
}

RouteInfo lookup_route(std::string_view path) {
    if (path == "/v1/sonder/health") return {Route::health, "GET"};
    if (path == "/v1/models") return {Route::models, "GET"};
    if (path == "/v1/sonder/identity") return {Route::identity, "GET"};
    if (path == "/v1/sonder/node") return {Route::sonder_node, "GET"};
    if (path == "/v1/sonder/plan") return {Route::sonder_plan, "POST"};
    if (path == "/v1/chat/completions") return {Route::chat, "POST"};
    if (path == "/v1/completions") return {Route::completions, "POST"};
    if (path == "/v1/responses") return {Route::responses, "POST"};
    if (path == "/v1/messages") return {Route::messages, "POST"};
    if (path == "/v1/embeddings") return {Route::embeddings, "POST"};
    if (path == "/v1/tokenize") return {Route::tokenize, "POST"};
    if (path == "/v1/detokenize") return {Route::detokenize, "POST"};
    if (path == "/.well-known/sonder-telemetry") return {Route::discovery, "GET"};
    if (path == "/v1/telemetry") return {Route::telemetry, "GET"};
    if (path == "/v1/telemetry/sse") return {Route::telemetry_sse, "GET"};
    if (path == "/v1/telemetry/ndjson") return {Route::telemetry_ndjson, "GET"};
    return {};
}

bool client_wants_close(const RequestHead& head) {
    for (const auto& [k, v] : head.headers) {
        if (k != "connection") continue;
        std::size_t pos = 0;
        while (pos <= v.size()) {
            const std::size_t comma = v.find(',', pos);
            std::string_view tok =
                std::string_view(v).substr(pos, comma == std::string::npos ? std::string_view::npos : comma - pos);
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.remove_prefix(1);
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.remove_suffix(1);
            if (tok.size() == 5) {
                std::string lowered(tok);
                std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (lowered == "close") return true;
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    return false;
}

bool is_serialized_origin(std::string_view origin) {
    const std::size_t sep = origin.find("://");
    if (sep == std::string_view::npos || sep == 0) {
        return false;
    }
    const std::string_view scheme = origin.substr(0, sep);
    if (!(scheme.front() >= 'a' && scheme.front() <= 'z')) {
        return false;
    }
    for (const char c : scheme) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')) {
            return false;
        }
    }
    std::string_view rest = origin.substr(sep + 3);
    std::string_view host = rest;
    std::string_view port;
    if (!rest.empty() && rest.front() == '[') {
        const std::size_t close = rest.find(']');
        if (close == std::string_view::npos || close == 1) {
            return false;
        }
        host = rest.substr(0, close + 1);
        const std::string_view after = rest.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') {
                return false;
            }
            port = after.substr(1);
            if (port.empty()) {
                return false;
            }
        }
        for (const char c : host.substr(1, host.size() - 2)) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || c == ':' || c == '.')) {
                return false;
            }
        }
    } else {
        if (const std::size_t colon = rest.find(':'); colon != std::string_view::npos) {
            host = rest.substr(0, colon);
            port = rest.substr(colon + 1);
            if (port.empty()) {
                return false;
            }
        }
        if (host.empty()) {
            return false;
        }
        for (const char c : host) {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_')) {
                return false;
            }
        }
    }
    if (port.size() > 5) {
        return false;
    }
    for (const char c : port) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

bool is_loopback_bind(std::string_view host) { return sonder::inference::detail::is_loopback_literal(host); }

bool origin_allowed(const ServerOptions& opts, const std::string& origin, bool read_only) {
    if (std::find(opts.cors_origins.begin(), opts.cors_origins.end(), origin) != opts.cors_origins.end()) {
        return true;
    }
    if (!opts.default_cors) {
        return false;
    }
    const auto& d = default_origins();
    if (std::find(d.begin(), d.end(), origin) == d.end()) {
        return false;
    }
    return read_only || !opts.token.empty();
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Captures values only: the logger thread may call it after the Server is gone,
// and the logger thread and stop() may call it at the same time (see LogQueue).
// It keeps no state and utc_timestamp_now() is reentrant, so both are safe.
LogQueue::DropLine log_drop_reporter(bool as_json) {
    return [as_json](std::uint64_t dropped, bool final_report) -> std::string {
        std::string message = "log queue full; " + std::to_string(dropped) + " lines dropped";
        if (final_report) {
            message += " (total at shutdown)";
        }
        if (as_json) {
            return json::Value(
                       json::Object{{"time", utc_timestamp_now()}, {"level", "warning"}, {"message", message}})
                .dump();
        }
        return "sonder-infer serve: warning: " + message;
    };
}

std::string format_log_message(LogFormat format, const char* level, const std::string& message) {
    if (format == LogFormat::json) {
        return json::Value(json::Object{{"time", utc_timestamp_now()}, {"level", level}, {"message", message}}).dump();
    }
    return std::string("sonder-infer serve: ") + level + ": " + message;
}

std::string format_access_log(LogFormat format, const Exchange& ex) {
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ex.started).count();
    if (format == LogFormat::json) {
        return json::Value(json::Object{{"time", utc_timestamp_now()},
                                        {"method", ex.method},
                                        {"path", ex.path},
                                        {"status", ex.status},
                                        {"ms", ms},
                                        {"request_id", ex.request_id}})
            .dump();
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", ms);
    return utc_timestamp_now() + " " + ex.method + " " + ex.path + " " + std::to_string(ex.status) + " " + buf +
           "ms " + ex.request_id;
}

Headers base_headers(const Exchange& ex, const std::string& content_type) {
    Headers h;
    if (!content_type.empty()) {
        h.emplace_back("Content-Type", content_type);
    }
    h.emplace_back("X-Sonder-Inference-Api", std::to_string(kApiVersion));
    h.emplace_back("X-Sonder-Request-Id", ex.request_id);
    h.emplace_back("Cache-Control", "no-store");
    if (ex.keep_alive) {
        h.emplace_back("Connection", "keep-alive");
        const auto timeout_s =
            ex.keep_alive_timeout.count() > 0 ? std::max<long long>(1, (ex.keep_alive_timeout.count() + 999) / 1000) : 0;
        h.emplace_back("Keep-Alive",
                       "timeout=" + std::to_string(timeout_s) + ", max=" + std::to_string(ex.keep_alive_remaining));
    } else {
        h.emplace_back("Connection", "close");
    }
    for (const auto& kv : ex.cors) {
        h.push_back(kv);
    }
    return h;
}

bool send_raw(Exchange& ex, std::string_view data, std::chrono::milliseconds write_stall_timeout,
              const std::atomic<bool>& hard_stop) {
    const bool ok = send_all(ex.sock, data, write_stall_timeout, &hard_stop).ok();
    if (!ok) {
        ex.write_failed = true;
    }
    return ok;
}

void send_response(Exchange& ex, int status, const std::string& content_type, const std::string& body,
                   std::chrono::milliseconds write_stall_timeout, const std::atomic<bool>& hard_stop,
                   const Headers& extra) {
    Headers h = base_headers(ex, content_type);
    for (const auto& kv : extra) {
        h.push_back(kv);
    }
    h.emplace_back("Content-Length", std::to_string(body.size()));
    ex.status = status;
    std::string out = format_head(status, h);
    out += body;
    (void)send_raw(ex, out, write_stall_timeout, hard_stop);
}

void send_json(Exchange& ex, int status, const json::Object& body, std::chrono::milliseconds write_stall_timeout,
               const std::atomic<bool>& hard_stop, const Headers& extra) {
    send_response(ex, status, "application/json", json::Value(body).dump(), write_stall_timeout, hard_stop, extra);
}

void send_error(Exchange& ex, const ApiError& error, std::chrono::milliseconds write_stall_timeout,
                const std::atomic<bool>& hard_stop, Headers extra) {
    // Close persistent connections on framing/transport errors or unread input so
    // stream boundaries cannot desynchronize.
    if (ex.unread_input > 0 || error.status == 400 || error.status == 408 || error.status == 411 ||
        error.status == 413 || error.status == 431 || error.status == 503 || error.status == 505) {
        ex.keep_alive = false;
    }
    if (error.retry_after) {
        extra.emplace_back("Retry-After", std::to_string(*error.retry_after));
    }
    if (error.status == 401) {
        extra.emplace_back("WWW-Authenticate", "Bearer");
    }
    send_json(ex, error.status, ex.path == "/v1/messages" ? anthropic_error_body(error) : error_body(error),
              write_stall_timeout, hard_stop, extra);
}

bool read_head(Exchange& ex, std::string& buffer, RequestHead& head, std::size_t& head_bytes, const ConnFlags& flags,
               const ServerOptions& opts, const std::atomic<bool>& stopping, const std::atomic<bool>& hard_stop) {
    const auto deadline = ex.started + opts.read_timeout;
    char chunk[4096];
    for (;;) {
        if (flags.evict.load()) {
            // Evicted to make room for a new connection (at the limit) — refuse
            // even a complete pipelined head already sitting in the read buffer.
            ex.keep_alive = false;
            send_error(ex,
                       make_error(408, "request_timeout",
                                  "request headers not received in time (the server is at its connection limit)"),
                       opts.write_stall_timeout, hard_stop);
            return false;
        }
        const ParseResult pr = parse_request_head(buffer, head);
        if (pr.state == ParseState::complete) {
            head_bytes = pr.head_bytes;
            return true;
        }
        if (pr.state == ParseState::error) {
            // Header parsing may already have identified this endpoint.
            if (head.path == "/v1/messages") ex.path = head.path;
            const char* code = pr.status == 431   ? "request_header_fields_too_large"
                               : pr.status == 505 ? "http_version_not_supported"
                                                  : "malformed_request";
            // Whatever else the client sends is unread: linger briefly.
            ex.unread_input = kLingerUnknownLength;
            send_error(ex, make_error(pr.status, code, pr.message), opts.write_stall_timeout, hard_stop);
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        // Short slices so an eviction takes effect promptly.
        const auto slice = std::min(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now),
                                    std::chrono::milliseconds(50));
        const auto w = now >= deadline ? WaitResult::timeout : wait_socket(ex.sock, false, slice, &stopping);
        if (w == WaitResult::timeout) {
            if (std::chrono::steady_clock::now() < deadline) {
                continue;
            }
            send_error(ex, make_error(408, "request_timeout", "request headers not received in time"),
                       opts.write_stall_timeout, hard_stop);
            return false;
        }
        if (w != WaitResult::ready) {
            return false;
        }
        const long long n = recv_some(ex.sock, chunk, sizeof(chunk));
        if (n == 0 || n == -1) {
            return false;
        }
        if (n > 0) {
            buffer.append(chunk, static_cast<std::size_t>(n));
        }
    }
}

bool read_body(Exchange& ex, std::string& buffer, std::size_t head_bytes, std::uint64_t length, std::string& body,
               const ServerOptions& opts, const std::atomic<bool>& hard_stop) {
    const std::size_t want_total = static_cast<std::size_t>(length);
    const std::size_t available = buffer.size() >= head_bytes ? buffer.size() - head_bytes : 0;
    const std::size_t take = std::min(available, want_total);
    body.assign(buffer, head_bytes, take);
    // Erase the consumed head and body prefix, leaving any pipelined bytes intact.
    buffer.erase(0, head_bytes + take);
    ex.unread_input = length - body.size();
    const auto deadline = std::chrono::steady_clock::now() + opts.read_timeout;
    char chunk[16384];
    while (body.size() < want_total) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            send_error(ex, make_error(408, "request_timeout", "request body not received in time"),
                       opts.write_stall_timeout, hard_stop);
            return false;
        }
        const auto w = wait_socket(ex.sock, false,
                                   std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now), &hard_stop);
        if (w == WaitResult::timeout) {
            send_error(ex, make_error(408, "request_timeout", "request body not received in time"),
                       opts.write_stall_timeout, hard_stop);
            return false;
        }
        if (w != WaitResult::ready) {
            return false;
        }
        const long long n = recv_some(ex.sock, chunk, sizeof(chunk));
        if (n == 0 || n == -1) {
            return false;
        }
        if (n > 0) {
            const std::size_t want = want_total - body.size();
            const std::size_t got = static_cast<std::size_t>(n);
            const std::size_t use = std::min(want, got);
            body.append(chunk, use);
            if (got > use) {
                // Preserve any pipelined bytes past the end of this request body.
                buffer.append(chunk + use, got - use);
            }
            ex.unread_input = length - body.size();
        }
    }
    ex.unread_input = 0;
    return true;
}

void linger(const Exchange& ex, const std::atomic<bool>& hard_stop) {
    if (ex.unread_input == 0) {
        return;
    }
    drain_input(ex.sock, std::min<std::uint64_t>(ex.unread_input, kLingerMaxBytes), kLingerMaxTime, kLingerIdle,
                &hard_stop);
}

PreDispatchResult run_request_guards(Exchange& ex, const RequestHead& head, std::string& buffer, std::size_t head_bytes,
                                     const ServerOptions& opts, bool loopback, bool has_token_digest,
                                     const std::array<std::uint8_t, 32>& token_digest,
                                     const std::atomic<bool>& hard_stop) {
    const auto fail = [&](const ApiError& error, Headers extra = {}) -> PreDispatchResult {
        // Pre-dispatch rejections skip body reading; always close to avoid stream desync.
        ex.keep_alive = false;
        send_error(ex, error, opts.write_stall_timeout, hard_stop, std::move(extra));
        return {};
    };
    // DNS-rebinding defence on loopback binds.
    if (loopback) {
        const std::string* host = head.header("host");
        if (host == nullptr || !is_loopback_host_header(*host)) {
            return fail(make_error(403, "forbidden_host",
                                   "Host header must name 127.0.0.1, localhost or [::1] on a loopback bind"));
        }
    }
    const RouteInfo route = lookup_route(head.path);
    const bool preflight = head.method == "OPTIONS";
    const std::string* origin = head.header("origin");
    if (origin != nullptr) {
        std::string requested = head.method;
        if (preflight) {
            const std::string* m = head.header("access-control-request-method");
            requested = m != nullptr ? *m : std::string("GET");
        }
        const bool read_only = requested == "GET" || requested == "HEAD";
        if (!origin_allowed(opts, *origin, read_only)) {
            return fail(make_error(403, "forbidden_origin",
                                   "origin '" + *origin +
                                       "' is not allowed; add it with --cors-origin (POST routes need an "
                                       "explicit --cors-origin or a token)"));
        }
        ex.cors.emplace_back("Access-Control-Allow-Origin", *origin);
        ex.cors.emplace_back("Vary", "Origin");
        ex.cors.emplace_back("Access-Control-Expose-Headers", kExposeHeaders);
    }
    if (preflight) {
        if ((head.content_length && *head.content_length > 0) || head.has_transfer_encoding) {
            ex.keep_alive = false;
        } else {
            buffer.erase(0, head_bytes);
        }
        Headers h = base_headers(ex, "");
        h.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        h.emplace_back("Access-Control-Allow-Headers",
                       route.route == Route::messages
                           ? std::string(kAllowHeaders) + ", X-Api-Key, Anthropic-Version, Anthropic-Beta"
                           : kAllowHeaders);
        h.emplace_back("Access-Control-Max-Age", "600");
        if (origin != nullptr && head.header("access-control-request-private-network") != nullptr) {
            h.emplace_back("Access-Control-Allow-Private-Network", "true");
        }
        h.emplace_back("Content-Length", "0");
        ex.status = 204;
        (void)send_raw(ex, format_head(204, h), opts.write_stall_timeout, hard_stop);
        return {};
    }
    if (has_token_digest) {
        const std::string* auth = head.header("authorization");
        bool ok = false;
        if (auth != nullptr && auth->size() > 7) {
            std::string scheme = auth->substr(0, 7);
            std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            ok = scheme == "bearer " && constant_time_equals_digest(std::string_view(*auth).substr(7), token_digest);
        }
        if (!ok && route.route == Route::messages) {
            const std::string* key = head.header("x-api-key");
            ok = key != nullptr && constant_time_equals_digest(*key, token_digest);
        }
        if (!ok) {
            return fail(make_error(401, "unauthorized",
                                   route.route == Route::messages ? "a valid bearer token or x-api-key is required"
                                                                  : "a valid bearer token is required"));
        }
    }
    if (route.route == Route::none) {
        return fail(make_error(404, "not_found", "no route for " + head.path));
    }
    if (head.method != route.method) {
        return fail(make_error(405, "method_not_allowed", head.method + " is not allowed on " + head.path),
                    Headers{{"Allow", std::string(route.method) + ", OPTIONS"}});
    }
    // Expect (RFC 9110 section 10.1.1): only 100-continue is supported;
    // HTTP/1.0 requests must have it ignored.
    bool send_continue = false;
    if (const std::string* expect = head.header("expect"); expect != nullptr && head.minor_version >= 1) {
        std::string e = *expect;
        std::transform(e.begin(), e.end(), e.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (e != "100-continue") {
            return fail(make_error(417, "expectation_failed", "only \"Expect: 100-continue\" is supported", "Expect"));
        }
        send_continue = true;
    }
    std::string body;
    if (head.method == "POST") {
        if (head.has_transfer_encoding || !head.content_length) {
            return fail(make_error(411, "length_required",
                                   "POST requires Content-Length (chunked bodies are not accepted)"));
        }
        const std::size_t max_body =
            route.route == Route::sonder_plan ? std::min(opts.max_body_bytes, kPlanMaxBodyBytes) : opts.max_body_bytes;
        if (*head.content_length > max_body) {
            return fail(make_error(413, "payload_too_large",
                                   "request body exceeds " + std::to_string(max_body) + " bytes"));
        }
        // The request passed every check that does not need the body:
        // tell a waiting client (curl waits 1 s otherwise) to send it.
        if (send_continue && ex.unread_input > 0 &&
            !send_raw(ex, "HTTP/1.1 100 Continue\r\n\r\n", opts.write_stall_timeout, hard_stop)) {
            return {};
        }
        if (!read_body(ex, buffer, head_bytes, *head.content_length, body, opts, hard_stop)) {
            return {};
        }
    } else {
        if ((head.content_length && *head.content_length > 0) || head.has_transfer_encoding) {
            ex.keep_alive = false;
        } else {
            buffer.erase(0, head_bytes);
        }
    }
    return PreDispatchResult{true, route.route, std::move(body)};
}

Exchange reject_overloaded_socket(Socket& client, const std::string& reason, const std::atomic<bool>& hard_stop) {
    Exchange ex;
    ex.sock = client.get();
    ex.started = std::chrono::steady_clock::now();
    ex.request_id = make_id("http");
    ApiError e = make_error(503, "overloaded", reason);
    e.retry_after = 1;
    Headers h = base_headers(ex, "application/json");
    h.emplace_back("Retry-After", "1");
    const std::string body = json::Value(error_body(e)).dump();
    h.emplace_back("Content-Length", std::to_string(body.size()));
    (void)send_all(ex.sock, format_head(503, h) + body, std::chrono::milliseconds(200), &hard_stop);
    ex.status = 503;
    client.shutdown_write();
    drain_input(ex.sock, 64u * 1024u, std::chrono::milliseconds(200), std::chrono::milliseconds(100), &hard_stop);
    return ex;
}

std::string format_listen_url(const std::string& raw_host, std::uint16_t port, bool ipv6) {
    std::string host = raw_host;
    if (host == "localhost") {
        host = ipv6 ? "[::1]" : "127.0.0.1";
    } else if (host == "0.0.0.0") {
        host = "127.0.0.1";
    } else if (host == "::" || host == "[::]") {
        host = "[::1]";
    } else if (host.find(':') != std::string::npos && host.front() != '[') {
        host = "[" + host + "]";
    }
    return "http://" + host + ":" + std::to_string(port);
}

}  // namespace sonder::inference::server::detail

namespace sonder::inference::server {

Status validate_options(const ServerOptions& o) {
    if (o.host.empty()) {
        return Status(ErrorCode::invalid_argument, "--host must not be empty");
    }
    if (!detail::is_loopback_bind(o.host) && o.token.empty()) {
        return Status(ErrorCode::invalid_argument, "--host " + o.host +
                                                       " is not a loopback address; a non-loopback bind requires "
                                                       "--token-file");
    }
    if (o.capture_text && o.token.empty()) {
        return Status(ErrorCode::invalid_argument, "--capture-text exports generated text and requires --token-file");
    }
    const std::string backend_name = o.backend_instance ? o.backend_instance->name() : o.backend.backend;
    if (!o.backend_instance) {
        const std::vector<std::string> names = available_backend_names();
        std::string list;
        for (const auto& n : names) {
            list += (list.empty() ? "" : ", ") + n;
        }
        if (o.backend.backend.empty()) {
            return Status(ErrorCode::invalid_argument, "--backend is required (this build has: " + list + ")");
        }
        if (std::find(names.begin(), names.end(), o.backend.backend) == names.end()) {
            return Status(ErrorCode::invalid_argument,
                          "unknown or unavailable backend '" + o.backend.backend + "' (this build has: " + list + ")");
        }
        if (Status st = validate_backend_setup(o.backend); !st.ok()) {
            return st;
        }
    }
    if (o.models.empty() && !is_synthetic_backend(backend_name)) {
        return Status(ErrorCode::invalid_argument, "--model is required for backend " + backend_name +
                                                       " (list models with `sonder-infer models --backend " +
                                                       backend_name + "`)");
    }
    for (std::size_t i = 0; i < o.models.size(); ++i) {
        if (o.models[i].empty()) {
            return Status(ErrorCode::invalid_argument, "--model must not be empty");
        }
        if (o.models[i] == "default") {
            return Status(ErrorCode::invalid_argument, "--model 'default' is reserved as the alias of the first model");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (o.models[j] == o.models[i]) {
                return Status(ErrorCode::invalid_argument, "--model " + o.models[i] + " is given twice");
            }
        }
    }
    for (std::size_t i = 0; i < o.profile_bindings.size(); ++i) {
        const auto& b = o.profile_bindings[i];
        if (std::find(o.models.begin(), o.models.end(), b.model) == o.models.end()) {
            return Status(ErrorCode::invalid_argument, "launch profile '" + b.model + "' is not a served --model");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (o.profile_bindings[j].model == b.model) {
                return Status(ErrorCode::invalid_argument, "launch profile '" + b.model + "' is bound twice");
            }
        }
    }
    if (o.max_connections == 0 || o.max_body_bytes == 0 || o.telemetry_buffer == 0 || o.max_subscribers == 0) {
        return Status(ErrorCode::invalid_argument,
                      "--max-connections, --max-body-bytes and --telemetry-buffer must be positive");
    }
    if (o.log_queue_capacity == 0) {
        return Status(ErrorCode::invalid_argument, "log_queue_capacity must be at least 1");
    }
    if (o.model_idle_ttl.count() < 0) {
        return Status(ErrorCode::invalid_argument, "--model-idle-ttl must not be negative");
    }
    if (o.keep_alive_timeout.count() < 0) {
        return Status(ErrorCode::invalid_argument, "keep_alive_timeout must not be negative");
    }
    if (o.stream_keepalive_interval.count() < 0) {
        return Status(ErrorCode::invalid_argument, "--stream-keepalive-ms must not be negative");
    }
    for (const auto& origin : o.cors_origins) {
        if (!detail::is_serialized_origin(origin)) {
            return Status(ErrorCode::invalid_argument,
                          "--cors-origin '" + origin +
                              "' is not an origin a browser sends: use scheme://host[:port] in lowercase, with no "
                              "path, trailing slash or wildcard (for example http://127.0.0.1:4173)");
        }
    }
    return detail::validate_request_path_options(o);
}

}  // namespace sonder::inference::server
