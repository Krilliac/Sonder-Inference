#include "sonder/inference/server.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "http.hpp"
#include "identity.hpp"
#include "live_hub.hpp"
#include "openai.hpp"
#include "socket.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"
#include "sonder_inference.h"

namespace sonder::inference::server {

using namespace detail;  // NOLINT(google-build-using-namespace)

namespace {

using Headers = std::vector<std::pair<std::string, std::string>>;

// Observatory dev (5173), preview (4173) and Tauri origins (contract 2.4).
// Applied to the read-only GET routes only (see origin_allowed()).
const std::vector<std::string>& default_origins() {
    static const std::vector<std::string> kOrigins = {
        "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173",
        "http://localhost:4173", "tauri://localhost",     "http://tauri.localhost"};
    return kOrigins;
}

constexpr const char* kAllowHeaders =
    "Accept, Authorization, Cache-Control, Content-Type, Last-Event-ID, X-Sonder-Run-Id, "
    "X-Sonder-Parent-Request-Id, X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority";
constexpr const char* kExposeHeaders = "X-Sonder-Inference-Api, X-Sonder-Request-Id, Retry-After";

enum class State { starting, ready, draining, stopped };

const char* state_name(State s) noexcept {
    switch (s) {
        case State::starting: return "starting";
        case State::ready: return "ready";
        case State::draining: return "draining";
        case State::stopped: return "draining";
    }
    return "draining";
}

enum class Route { none, health, models, identity, chat, embeddings, discovery, telemetry, telemetry_sse, telemetry_ndjson };

struct RouteInfo {
    Route route = Route::none;
    const char* method = "";  // the one method the route accepts
};

RouteInfo lookup_route(std::string_view path) {
    if (path == "/v1/sonder/health") return {Route::health, "GET"};
    if (path == "/v1/models") return {Route::models, "GET"};
    if (path == "/v1/sonder/identity") return {Route::identity, "GET"};
    if (path == "/v1/chat/completions") return {Route::chat, "POST"};
    if (path == "/v1/embeddings") return {Route::embeddings, "POST"};
    if (path == "/.well-known/sonder-telemetry") return {Route::discovery, "GET"};
    if (path == "/v1/telemetry") return {Route::telemetry, "GET"};
    if (path == "/v1/telemetry/sse") return {Route::telemetry_sse, "GET"};
    if (path == "/v1/telemetry/ndjson") return {Route::telemetry_ndjson, "GET"};
    return {};
}

bool is_loopback_bind(std::string_view host) {
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    return host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::int64_t unix_seconds() {
    return static_cast<std::int64_t>(std::time(nullptr));
}

struct ServedModel {
    std::string id;
    std::string backend;
    bool is_default = false;
    std::shared_ptr<Model> model;
};

// Per-connection bookkeeping shared between the accept loop and the
// connection thread.
struct ConnFlags {
    std::atomic<bool> done{false};
    std::atomic<bool> streaming{false};  // telemetry stream: no engine access
};

// One request/response exchange on a connection.
struct Exchange {
    native_socket sock{};
    std::chrono::steady_clock::time_point started;
    std::string request_id;
    std::string method = "-";
    std::string path = "-";
    int status = 0;
    Headers cors;  // set once the Origin is known to be allowed
};

}  // namespace

// ============================================================ Server::Impl

struct Server::Impl {
    explicit Impl(ServerOptions o) : opts(std::move(o)) {}

    ServerOptions opts;
    std::atomic<State> state{State::starting};

    Socket listener;
    std::uint16_t bound_port = 0;
    std::string url;
    bool loopback = true;
    bool synthetic = false;

    std::shared_ptr<LiveTelemetryHub> hub;
    std::string instance_id;
    std::string node_id;
    std::chrono::steady_clock::time_point started_at;

    mutable std::mutex engine_mu;
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Backend> backend;
    mutable std::mutex models_mu;
    std::vector<ServedModel> served;

    // Backend probe cache (health must stay cheap).
    std::mutex probe_mu;
    std::chrono::steady_clock::time_point probe_at{};
    bool probe_valid = false;
    bool probe_ok = false;
    std::string probe_version;

    std::atomic<bool> stopping{false};   // listener closing; idle reads end
    std::atomic<bool> hard_stop{false};  // abort writes and streams
    std::thread accept_thread;
    bool started = false;

    std::mutex conns_mu;
    std::vector<std::pair<std::thread, std::shared_ptr<ConnFlags>>> conns;
    std::atomic<std::size_t> active{0};

    std::mutex inflight_mu;
    std::condition_variable inflight_cv;
    std::set<std::shared_ptr<Session>> inflight;

    std::once_flag stop_once;
    std::mutex stopped_mu;
    std::condition_variable stopped_cv;
    bool stopped = false;

    // ------------------------------------------------------------- logging

    void log_line(const std::string& line) const {
        if (opts.log) {
            opts.log(line);
        }
    }

    void log_message(const char* level, const std::string& message) const {
        if (opts.log_format == LogFormat::json) {
            log_line(json::Value(json::Object{{"time", utc_timestamp_now()}, {"level", level}, {"message", message}})
                         .dump());
        } else {
            log_line(std::string("sonder-infer serve: ") + level + ": " + message);
        }
    }

    void access_log(const Exchange& ex) const {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ex.started).count();
        if (opts.log_format == LogFormat::json) {
            log_line(json::Value(json::Object{{"time", utc_timestamp_now()},
                                              {"method", ex.method},
                                              {"path", ex.path},
                                              {"status", ex.status},
                                              {"ms", ms},
                                              {"request_id", ex.request_id}})
                         .dump());
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1f", ms);
            log_line(utc_timestamp_now() + " " + ex.method + " " + ex.path + " " + std::to_string(ex.status) + " " +
                     buf + "ms " + ex.request_id);
        }
    }

    // ------------------------------------------------------------ writing

    Headers base_headers(const Exchange& ex, const std::string& content_type) const {
        Headers h;
        if (!content_type.empty()) {
            h.emplace_back("Content-Type", content_type);
        }
        h.emplace_back("X-Sonder-Inference-Api", std::to_string(kApiVersion));
        h.emplace_back("X-Sonder-Request-Id", ex.request_id);
        h.emplace_back("Cache-Control", "no-store");
        h.emplace_back("Connection", "close");
        for (const auto& kv : ex.cors) {
            h.push_back(kv);
        }
        return h;
    }

    bool send_raw(Exchange& ex, std::string_view data) const {
        return send_all(ex.sock, data, opts.write_stall_timeout, &hard_stop).ok();
    }

    void send_response(Exchange& ex, int status, const std::string& content_type, const std::string& body,
                       const Headers& extra = {}) const {
        Headers h = base_headers(ex, content_type);
        for (const auto& kv : extra) {
            h.push_back(kv);
        }
        h.emplace_back("Content-Length", std::to_string(body.size()));
        ex.status = status;
        std::string out = format_head(status, h);
        out += body;
        (void)send_raw(ex, out);
    }

    void send_json(Exchange& ex, int status, const json::Object& body, const Headers& extra = {}) const {
        send_response(ex, status, "application/json", json::Value(body).dump(), extra);
    }

    void send_error(Exchange& ex, const ApiError& error, Headers extra = {}) const {
        if (error.retry_after) {
            extra.emplace_back("Retry-After", std::to_string(*error.retry_after));
        }
        if (error.status == 401) {
            extra.emplace_back("WWW-Authenticate", "Bearer");
        }
        send_json(ex, error.status, error_body(error), extra);
    }

    // ------------------------------------------------------------ helpers

    json::Object sonder_meta() const { return json::Object{{"api_version", kApiVersion}}; }

    std::vector<ServedModel> served_snapshot() const {
        std::lock_guard<std::mutex> lock(models_mu);
        return served;
    }

    std::optional<ServedModel> resolve_model(const std::string& id) const {
        std::lock_guard<std::mutex> lock(models_mu);
        for (const auto& m : served) {
            if (m.id == id || (id == "default" && m.is_default)) {
                return m;
            }
        }
        return std::nullopt;
    }

    // Probes the backend at most once every 2 s.
    void backend_status(bool& available, std::string& version) {
        std::shared_ptr<Backend> b;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
        }
        if (!b) {
            available = false;
            return;
        }
        std::lock_guard<std::mutex> lock(probe_mu);
        const auto now = std::chrono::steady_clock::now();
        if (!probe_valid || now - probe_at >= std::chrono::seconds(2)) {
            auto probed = b->probe();
            probe_ok = probed.ok();
            probe_version = probed.ok() ? probed.value() : std::string();
            probe_at = now;
            probe_valid = true;
        }
        available = probe_ok;
        version = probe_version;
    }

    std::string absolute(std::string_view path) const { return url + std::string(path); }

    // Origin policy (contract 2.4 with the review correction): explicit
    // --cors-origin entries are allowed everywhere; the default allowlist only
    // for read-only GET requests unless a token protects the server.
    bool origin_allowed(const std::string& origin, bool read_only) const {
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

    // ------------------------------------------------------------ routes

    void handle_health(Exchange& ex) {
        const State s = state.load();
        bool available = false;
        std::string version;
        backend_status(available, version);
        json::Array backends;
        {
            std::shared_ptr<Backend> b;
            {
                std::lock_guard<std::mutex> lock(engine_mu);
                b = backend;
            }
            if (b) {
                json::Array caps;
                for (const auto& c : b->capabilities().names()) {
                    caps.emplace_back(c);
                }
                json::Object entry{{"name", b->name()}, {"available", available}, {"capabilities", std::move(caps)}};
                if (!version.empty()) {
                    entry.set("version", version);
                }
                backends.emplace_back(std::move(entry));
            }
        }
        json::Array models;
        for (const auto& m : served_snapshot()) {
            models.emplace_back(json::Object{{"id", m.id}, {"backend", m.backend}, {"default", m.is_default}});
        }
        const HubStats hs = hub->stats();
        std::uint64_t emitted = 0;
        std::uint64_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (engine) {
                emitted = engine->telemetry().emitted_events();
                dropped = engine->telemetry().dropped_events();
            }
        }
        json::Object body{{"status", state_name(s)},
                          {"api_version", kApiVersion},
                          {"version", version_string()},
                          {"commit", build_commit()},
                          {"abi_version", SONDER_ABI_VERSION},
                          {"instance_id", instance_id},
                          {"node_id", node_id},
                          {"uptime_s", seconds_since(started_at)},
                          {"synthetic", synthetic},
                          {"auth_required", !opts.token.empty()},
                          {"backends", std::move(backends)},
                          {"models", std::move(models)},
                          {"telemetry", json::Object{{"level", to_string(opts.telemetry_level)},
                                                     {"subscribers", hs.subscribers},
                                                     {"retained", hs.retained},
                                                     {"capacity", hs.capacity},
                                                     {"emitted", emitted},
                                                     {"dropped", dropped},
                                                     {"subscriber_dropped_events", hs.subscriber_dropped_events}}},
                          {"sonder", sonder_meta()}};
        send_json(ex, s == State::ready ? 200 : 503, body);
    }

    bool require_ready(Exchange& ex) {
        const State s = state.load();
        if (s == State::ready) {
            return true;
        }
        send_error(ex, make_error(503, "not_ready",
                                  std::string("sonder-inference is ") + state_name(s) +
                                      "; nothing was executed, the request is safe to send elsewhere"));
        return false;
    }

    void handle_models(Exchange& ex) {
        if (!require_ready(ex)) {
            return;
        }
        json::Array data;
        for (const auto& m : served_snapshot()) {
            data.emplace_back(json::Object{
                {"id", m.id},
                {"object", "model"},
                {"owned_by", "sonder-inference"},
                {"sonder", json::Object{{"backend", m.backend}, {"default", m.is_default}, {"synthetic", synthetic}}}});
        }
        send_json(ex, 200, json::Object{{"object", "list"}, {"data", std::move(data)}, {"sonder", sonder_meta()}});
    }

    void handle_identity(Exchange& ex, const RequestHead& head) {
        if (!require_ready(ex)) {
            return;
        }
        std::string wanted = "default";
        if (!head.query.empty()) {
            if (const auto m = query_param(head.query, "model")) {
                wanted = *m;
            }
        }
        const auto served_model = resolve_model(wanted);
        if (!served_model) {
            send_error(ex, make_error(404, "model_not_found", "model '" + wanted + "' is not served", "model"));
            return;
        }
        bool available = false;
        std::string version;
        backend_status(available, version);
        const IdentityResult id =
            backend_identity(*served_model->model, available ? std::optional<std::string>(version) : std::nullopt);
        send_json(ex, 200,
                  json::Object{{"schema", "sonder.inference.identity/1"},
                               {"model", served_model->id},
                               {"synthetic", synthetic},
                               {"backend_identity", id.backend_identity},
                               {"reason", id.reason},
                               {"sonder", sonder_meta()}});
    }

    json::Object discovery_document() const {
        const HubStats hs = hub->stats();
        return json::Object{
            {"schema", "sonder.telemetry.producer/1"},
            {"producer", json::Object{{"name", "sonder-inference"},
                                      {"version", version_string()},
                                      {"node_id", node_id},
                                      {"instance_id", instance_id},
                                      {"role", "inference"},
                                      {"synthetic", synthetic}}},
            {"event_schema", std::string(kObservatorySchema)},
            {"streams", json::Array{json::Object{{"transport", "sse"}, {"url", "/v1/telemetry/sse"}},
                                    json::Object{{"transport", "ndjson"}, {"url", "/v1/telemetry/ndjson"}}}},
            {"resume", json::Object{{"header", "Last-Event-ID"},
                                    {"query", "last_event_id"},
                                    {"retained_events", hs.retained},
                                    {"oldest_sequence", hs.oldest_sequence},
                                    {"next_sequence", hs.next_sequence}}},
            {"auth", json::Object{{"required", !opts.token.empty()}, {"schemes", json::Array{"bearer"}}}},
            {"clock", json::Object{{"mono_ns", "host-monotonic"}}},
            {"sampling_level", to_string(opts.telemetry_level)},
            {"text_capture", opts.capture_text ? "on" : "off"},
            {"links", json::Object{{"health", "/v1/sonder/health"},
                                   {"identity", "/v1/sonder/identity"},
                                   {"models", "/v1/models"}}},
            {"vocabularies", json::Object{{"sonder.inference.events", 1}}},
            {"stats", json::Object{{"subscribers", hs.subscribers},
                                   {"max_subscribers", opts.max_subscribers},
                                   {"capacity", hs.capacity},
                                   {"subscriber_dropped_events", hs.subscriber_dropped_events}}},
            {"sonder", sonder_meta()}};
    }

    void handle_telemetry(Exchange& ex, const RequestHead& head, Route route, ConnFlags& flags) {
        bool sse = true;
        if (route == Route::telemetry_ndjson) {
            sse = false;
        } else if (route == Route::telemetry) {
            const auto format = head.query.empty() ? std::nullopt : query_param(head.query, "format");
            if (format && (*format == "ndjson" || *format == "sse")) {
                sse = *format == "sse";
            } else if (format) {
                send_error(ex, make_error(400, "invalid_request", "format must be sse or ndjson", "format"));
                return;
            } else if (const std::string* accept = head.header("accept")) {
                sse = accept->find("application/x-ndjson") == std::string::npos;
            }
        }
        ResumeRequest resume;
        if (const std::string* last = head.header("last-event-id")) {
            resume.last_event_id = *last;
        } else if (!head.query.empty()) {
            resume.last_event_id = query_param(head.query, "last_event_id");
        }
        if (!head.query.empty()) {
            const auto since = query_param(head.query, "since");
            resume.since_now = since && *since == "now";
        }
        auto subscribed = hub->subscribe(resume);
        if (subscribed.error == LiveTelemetryHub::SubscribeError::over_capacity) {
            ApiError e = make_error(429, "overloaded",
                                    "telemetry subscriber limit reached (" + std::to_string(opts.max_subscribers) +
                                        " streams)");
            e.retry_after = 1;
            send_error(ex, e);
            return;
        }
        if (subscribed.error == LiveTelemetryHub::SubscribeError::closed) {
            send_error(ex, make_error(503, "not_ready", "telemetry is shutting down"));
            return;
        }
        const auto sub = subscribed.subscription;
        // From here on the stream never touches the engine.
        flags.streaming.store(true);
        Headers h = base_headers(ex, sse ? "text/event-stream; charset=utf-8" : "application/x-ndjson");
        h.emplace_back("X-Accel-Buffering", "no");
        ex.status = 200;
        std::string preamble = format_head(200, h);
        if (sse) {
            preamble += "retry: 2000\n\n";
            if (!subscribed.gap_comment.empty()) {
                preamble += subscribed.gap_comment + "\n\n";
            }
        }
        bool ok = send_raw(ex, preamble);
        auto last_write = std::chrono::steady_clock::now();
        while (ok) {
            const auto since_write = std::chrono::steady_clock::now() - last_write;
            const auto wait = std::max<std::chrono::steady_clock::duration>(
                std::chrono::milliseconds(1),
                std::min<std::chrono::steady_clock::duration>(opts.heartbeat_interval - since_write,
                                                              std::chrono::milliseconds(250)));
            auto batch = sub->next(std::chrono::duration_cast<std::chrono::milliseconds>(wait), &hard_stop);
            std::string frame;
            if (batch.dropped > 0 && sse) {
                frame += ": dropped " + std::to_string(batch.dropped) + "\n\n";
            }
            for (const auto& ev : batch.events) {
                if (sse) {
                    frame += "id: " + instance_id + "-" + std::to_string(ev->sequence) + "\ndata: ";
                    frame += ev->line;
                    frame += "\n\n";
                } else {
                    frame += ev->line;
                    frame += "\n";
                }
            }
            if (!frame.empty()) {
                ok = send_raw(ex, frame);
                last_write = std::chrono::steady_clock::now();
            }
            if (!ok || batch.closed || hard_stop.load()) {
                break;
            }
            if (std::chrono::steady_clock::now() - last_write >= opts.heartbeat_interval) {
                ok = send_raw(ex, sse ? ": keepalive\n\n" : "\n");
                last_write = std::chrono::steady_clock::now();
            }
            if (peer_gone(ex.sock)) {
                break;
            }
        }
        hub->unsubscribe(sub);
    }

    void handle_chat(Exchange& ex, const RequestHead& head, const std::string& body) {
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            send_error(ex, *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        auto job_v = parse_chat_request(body);
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            send_error(ex, *e);
            return;
        }
        const ChatJob job = std::move(std::get<ChatJob>(job_v));
        if (!require_ready(ex)) {
            return;
        }
        const auto served_model = resolve_model(job.model);
        if (!served_model) {
            send_error(ex, make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"));
            return;
        }

        SessionOptions so;
        so.sampling = job.sampling;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                send_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(served_model->model, so);
            if (!created.ok()) {
                send_error(ex, map_session_failure(created.status(), false));
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                session->close();
                send_error(ex, make_error(503, "not_ready",
                                          "sonder-inference is draining; nothing was executed, the request is safe "
                                          "to send elsewhere"));
                return;
            }
            inflight.insert(session);
        }

        RequestOptions ro;
        ro.request_id = make_id("req");
        ro.parent_request_id = corr.parent_request_id;
        const std::string completion_id = "chatcmpl-" + *ro.request_id;
        // X-Sonder-Request-Id and the access log carry the engine request id.
        ex.request_id = *ro.request_id;
        const std::int64_t created_at = unix_seconds();

        // Client disconnect cancels the request (request.cancelled).
        std::mutex watch_mu;
        std::condition_variable watch_cv;
        bool watch_done = false;
        std::atomic<bool> disconnected{false};
        std::thread watcher([&] {
            std::unique_lock<std::mutex> lock(watch_mu);
            while (!watch_done) {
                lock.unlock();
                if (peer_gone(ex.sock)) {
                    disconnected.store(true);
                    session->cancel();
                    return;
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });

        bool headers_sent = false;
        bool write_failed = false;
        const auto chunk_object = [&](json::Object delta, json::Value finish) {
            return json::Object{{"id", completion_id},
                                {"object", "chat.completion.chunk"},
                                {"created", created_at},
                                {"model", served_model->id},
                                {"choices", json::Array{json::Object{{"index", 0},
                                                                     {"delta", std::move(delta)},
                                                                     {"finish_reason", std::move(finish)}}}}};
        };
        const auto send_event = [&](const json::Object& obj) {
            if (write_failed) {
                return false;
            }
            std::string frame = "data: ";
            frame += json::Value(obj).dump();
            frame += "\n\n";
            if (!send_raw(ex, frame)) {
                write_failed = true;
            }
            return !write_failed;
        };
        const auto ensure_stream_head = [&]() {
            if (headers_sent) {
                return !write_failed;
            }
            headers_sent = true;
            ex.status = 200;
            Headers h = base_headers(ex, "text/event-stream; charset=utf-8");
            h.emplace_back("X-Accel-Buffering", "no");
            if (!send_raw(ex, format_head(200, h))) {
                write_failed = true;
                return false;
            }
            return send_event(chunk_object(json::Object{{"role", "assistant"}, {"content", ""}}, nullptr));
        };

        TokenCallback on_chunk;
        if (job.stream) {
            on_chunk = [&](const TokenChunk& chunk) {
                if (!ensure_stream_head() ||
                    !send_event(chunk_object(json::Object{{"content", std::string(chunk.text)}}, nullptr))) {
                    session->cancel();
                    return false;
                }
                return true;
            };
        }

        auto result = session->chat(job.messages, on_chunk, job.sampling, ro);
        {
            std::lock_guard<std::mutex> lock(watch_mu);
            watch_done = true;
        }
        watch_cv.notify_all();
        watcher.join();
        const bool rejected = session->last_scheduler_rejected();
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            inflight.erase(session);
        }
        inflight_cv.notify_all();

        if (disconnected.load() || write_failed) {
            ex.status = 499;  // client closed the request (access log only)
            session->close();
            return;
        }
        if (!result.ok()) {
            session->close();
            const ApiError e = map_session_failure(result.status(), rejected);
            if (headers_sent) {
                (void)send_event(error_body(e));
            } else {
                send_error(ex, e);
            }
            return;
        }
        const GenerationResult& r = result.value();
        json::Object meta{{"api_version", kApiVersion},
                          {"request_id", r.request_id},
                          {"session_id", session->id()},
                          {"backend", served_model->backend},
                          {"synthetic", synthetic},
                          {"token_counts_from_backend", r.stats.token_counts_from_backend}};
        session->close();
        if (!job.stream) {
            json::Object doc{
                {"id", completion_id},
                {"object", "chat.completion"},
                {"created", created_at},
                {"model", served_model->id},
                {"choices",
                 json::Array{json::Object{{"index", 0},
                                          {"message", json::Object{{"role", "assistant"}, {"content", r.text}}},
                                          {"finish_reason", finish_reason(r)}}}},
                {"usage", usage_json(r)},
                {"timings", timings_json(r)},
                {"sonder", std::move(meta)}};
            send_json(ex, 200, doc);
            return;
        }
        if (!ensure_stream_head()) {
            return;
        }
        json::Object last = chunk_object(json::Object{}, finish_reason(r));
        if (job.include_usage) {
            last.set("usage", usage_json(r));
        }
        last.set("timings", timings_json(r));
        last.set("sonder", std::move(meta));
        if (send_event(last)) {
            (void)send_raw(ex, "data: [DONE]\n\n");
        }
    }

    // --------------------------------------------------------- connection

    // Reads until the head is complete. Returns false when an error response
    // was sent or the connection should just close.
    bool read_head(Exchange& ex, std::string& buffer, RequestHead& head, std::size_t& head_bytes) {
        const auto deadline = ex.started + opts.read_timeout;
        char chunk[4096];
        for (;;) {
            const ParseResult pr = parse_request_head(buffer, head);
            if (pr.state == ParseState::complete) {
                head_bytes = pr.head_bytes;
                return true;
            }
            if (pr.state == ParseState::error) {
                const char* code = pr.status == 431   ? "request_header_fields_too_large"
                                   : pr.status == 505 ? "http_version_not_supported"
                                                      : "malformed_request";
                send_error(ex, make_error(pr.status, code, pr.message));
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto w = now >= deadline
                               ? WaitResult::timeout
                               : wait_socket(ex.sock, false,
                                             std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now),
                                             &stopping);
            if (w == WaitResult::timeout) {
                send_error(ex, make_error(408, "request_timeout", "request headers not received in time"));
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

    bool read_body(Exchange& ex, std::string& buffer, std::size_t head_bytes, std::uint64_t length, std::string& body) {
        body = buffer.substr(head_bytes);
        if (body.size() > length) {
            body.resize(static_cast<std::size_t>(length));  // extra bytes after the body are ignored
        }
        const auto deadline = std::chrono::steady_clock::now() + opts.read_timeout;
        char chunk[16384];
        while (body.size() < length) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                send_error(ex, make_error(408, "request_timeout", "request body not received in time"));
                return false;
            }
            const auto w = wait_socket(ex.sock, false,
                                       std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now), &hard_stop);
            if (w == WaitResult::timeout) {
                send_error(ex, make_error(408, "request_timeout", "request body not received in time"));
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
                const std::size_t want = static_cast<std::size_t>(length) - body.size();
                body.append(chunk, std::min<std::size_t>(want, static_cast<std::size_t>(n)));
            }
        }
        return true;
    }

    void serve_connection(Socket sock, ConnFlags& flags) {
        Exchange ex;
        ex.sock = sock.get();
        ex.started = std::chrono::steady_clock::now();
        ex.request_id = make_id("http");
        std::string buffer;
        RequestHead head;
        std::size_t head_bytes = 0;
        if (!read_head(ex, buffer, head, head_bytes)) {
            if (ex.status != 0) {
                access_log(ex);
            }
            return;
        }
        ex.method = head.method;
        ex.path = head.path;
        dispatch(ex, head, buffer, head_bytes, flags);
        access_log(ex);
        // Let the peer read everything before the socket closes.
        sock.shutdown_write();
    }

    void dispatch(Exchange& ex, const RequestHead& head, std::string& buffer, std::size_t head_bytes,
                  ConnFlags& flags) {
        // DNS-rebinding defence on loopback binds.
        if (loopback) {
            const std::string* host = head.header("host");
            if (host == nullptr || !is_loopback_host_header(*host)) {
                send_error(ex, make_error(403, "forbidden_host",
                                          "Host header must name 127.0.0.1, localhost or [::1] on a loopback bind"));
                return;
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
            if (!origin_allowed(*origin, read_only)) {
                send_error(ex, make_error(403, "forbidden_origin",
                                          "origin '" + *origin +
                                              "' is not allowed; add it with --cors-origin (POST routes need an "
                                              "explicit --cors-origin or a token)"));
                return;
            }
            ex.cors.emplace_back("Access-Control-Allow-Origin", *origin);
            ex.cors.emplace_back("Vary", "Origin");
            ex.cors.emplace_back("Access-Control-Expose-Headers", kExposeHeaders);
        }
        if (preflight) {
            Headers h = base_headers(ex, "");
            h.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            h.emplace_back("Access-Control-Allow-Headers", kAllowHeaders);
            h.emplace_back("Access-Control-Max-Age", "600");
            if (origin != nullptr && head.header("access-control-request-private-network") != nullptr) {
                h.emplace_back("Access-Control-Allow-Private-Network", "true");
            }
            h.emplace_back("Content-Length", "0");
            ex.status = 204;
            (void)send_raw(ex, format_head(204, h));
            return;
        }
        if (!opts.token.empty()) {
            const std::string* auth = head.header("authorization");
            bool ok = false;
            if (auth != nullptr && auth->size() > 7) {
                std::string scheme = auth->substr(0, 7);
                std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                ok = scheme == "bearer " && constant_time_equals(std::string_view(*auth).substr(7), opts.token);
            }
            if (!ok) {
                send_error(ex, make_error(401, "unauthorized", "a valid bearer token is required"));
                return;
            }
        }
        if (route.route == Route::none) {
            send_error(ex, make_error(404, "not_found", "no route for " + head.path));
            return;
        }
        if (head.method != route.method) {
            send_error(ex, make_error(405, "method_not_allowed", head.method + " is not allowed on " + head.path),
                       Headers{{"Allow", std::string(route.method) + ", OPTIONS"}});
            return;
        }
        std::string body;
        if (head.method == "POST") {
            if (head.has_transfer_encoding || !head.content_length) {
                send_error(ex, make_error(411, "length_required",
                                          "POST requires Content-Length (chunked bodies are not accepted)"));
                return;
            }
            if (*head.content_length > opts.max_body_bytes) {
                send_error(ex, make_error(413, "payload_too_large",
                                          "request body exceeds " + std::to_string(opts.max_body_bytes) + " bytes"));
                return;
            }
            if (!read_body(ex, buffer, head_bytes, *head.content_length, body)) {
                return;
            }
        }
        switch (route.route) {
            case Route::health: handle_health(ex); break;
            case Route::models: handle_models(ex); break;
            case Route::identity: handle_identity(ex, head); break;
            case Route::chat: handle_chat(ex, head, body); break;
            case Route::embeddings:
                send_error(ex, make_error(501, "not_implemented", "embeddings are not implemented by sonder-inference"));
                break;
            case Route::discovery: send_json(ex, 200, discovery_document()); break;
            case Route::telemetry:
            case Route::telemetry_sse:
            case Route::telemetry_ndjson: handle_telemetry(ex, head, route.route, flags); break;
            case Route::none: break;
        }
    }

    // ------------------------------------------------------------ accept

    void reap_finished() {
        std::vector<std::thread> finished;
        {
            std::lock_guard<std::mutex> lock(conns_mu);
            for (auto it = conns.begin(); it != conns.end();) {
                if (it->second->done.load()) {
                    finished.push_back(std::move(it->first));
                    it = conns.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& t : finished) {
            t.join();
        }
    }

    void reject_overloaded(Socket client) {
        Exchange ex;
        ex.sock = client.get();
        ex.started = std::chrono::steady_clock::now();
        ex.request_id = make_id("http");
        ApiError e = make_error(503, "overloaded",
                                "connection limit reached (" + std::to_string(opts.max_connections) + ")");
        e.retry_after = 1;
        Headers h = base_headers(ex, "application/json");
        h.emplace_back("Retry-After", "1");
        const std::string body = json::Value(error_body(e)).dump();
        h.emplace_back("Content-Length", std::to_string(body.size()));
        (void)send_all(ex.sock, format_head(503, h) + body, std::chrono::milliseconds(200), &hard_stop);
        ex.status = 503;
        access_log(ex);
        client.shutdown_write();
    }

    void accept_loop() {
        while (!stopping.load()) {
            const auto w = wait_socket(listener.get(), false, std::chrono::milliseconds(100), &stopping);
            reap_finished();
            if (w != WaitResult::ready) {
                continue;
            }
            for (;;) {
                Socket client = accept_client(listener);
                if (!client.valid()) {
                    break;
                }
                if (active.load() >= opts.max_connections) {
                    reject_overloaded(std::move(client));
                    continue;
                }
                auto flags = std::make_shared<ConnFlags>();
                active.fetch_add(1);
                try {
                    std::thread t([this, flags, s = std::move(client)]() mutable {
                        serve_connection(std::move(s), *flags);
                        active.fetch_sub(1);
                        flags->done.store(true);
                    });
                    std::lock_guard<std::mutex> lock(conns_mu);
                    conns.emplace_back(std::move(t), flags);
                } catch (const std::system_error&) {
                    active.fetch_sub(1);
                    log_message("warning", "cannot start a connection thread; connection dropped");
                }
            }
        }
        listener.close();
    }

    // ------------------------------------------------------------- start

    Status start(const std::function<void()>& on_listening) {
        if (started) {
            return Status(ErrorCode::invalid_state, "server already started");
        }
        if (Status st = validate_options(opts); !st.ok()) {
            return st;
        }
        bool in_use = false;
        auto listened = listen_tcp(opts.host, opts.port, in_use);
        if (!listened.ok()) {
            return listened.status();
        }
        listener = std::move(listened.value().socket);
        bound_port = listened.value().port;
        loopback = is_loopback_bind(opts.host);
        std::string host = opts.host;
        if (host == "localhost") {
            host = listened.value().ipv6 ? "[::1]" : "127.0.0.1";
        } else if (host == "0.0.0.0") {
            host = "127.0.0.1";
        } else if (host == "::" || host == "[::]") {
            host = "[::1]";
        } else if (host.find(':') != std::string::npos && host.front() != '[') {
            host = "[" + host + "]";
        }
        url = "http://" + host + ":" + std::to_string(bound_port);

        synthetic = is_synthetic_backend(opts.backend.backend);
        hub = std::make_shared<LiveTelemetryHub>(opts.telemetry_buffer, opts.max_subscribers, opts.subscriber_queue);
        EngineOptions eo;
        eo.telemetry.level = opts.telemetry_level;
        eo.telemetry.capture_text = opts.capture_text;
        eo.telemetry.role = "inference";
        eo.telemetry.synthetic = synthetic;
        eo.telemetry.queue_capacity = std::max<std::size_t>(4096, opts.telemetry_buffer);
        eo.telemetry_sinks.push_back(hub);
        for (const auto& s : opts.extra_sinks) {
            eo.telemetry_sinks.push_back(s);
        }
        eo.server = EngineServerInfo{opts.host, bound_port, kApiVersion};
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            engine = std::make_unique<Engine>(std::move(eo));
            instance_id = engine->telemetry().instance_id();
            node_id = engine->telemetry().options().node_id;
        }
        hub->set_instance_id(instance_id);
        started_at = std::chrono::steady_clock::now();
        started = true;
        accept_thread = std::thread([this] { accept_loop(); });
        if (on_listening) {
            on_listening();
        }

        auto made = make_backend(opts.backend);
        if (!made.ok()) {
            return made.status();
        }
        std::vector<std::string> ids = opts.models;
        if (ids.empty()) {
            ids.emplace_back("mock");
        }
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (Status st = engine->register_backend(made.value()); !st.ok()) {
                return st;
            }
            backend = made.value();
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ModelLoadOptions lo;
            lo.model = ids[i];
            Result<std::shared_ptr<Model>> loaded = Status(ErrorCode::internal, "engine stopped");
            {
                std::lock_guard<std::mutex> lock(engine_mu);
                if (engine) {
                    loaded = engine->load_model(backend->name(), lo);
                }
            }
            if (!loaded.ok()) {
                return Status(loaded.status().code(),
                              "cannot load model '" + ids[i] + "' on " + backend->name() + ": " +
                                  loaded.status().message());
            }
            std::lock_guard<std::mutex> lock(models_mu);
            served.push_back(ServedModel{ids[i], backend->name(), i == 0, loaded.value()});
        }
        State expected = State::starting;
        state.compare_exchange_strong(expected, State::ready);
        return Status::success();
    }

    // -------------------------------------------------------------- stop

    // Waits until `pred` holds, polling; returns false on deadline.
    template <class Pred>
    bool wait_until(Pred pred, std::chrono::steady_clock::time_point deadline) {
        while (!pred()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

    bool all_connections_done(bool ignore_streams) {
        std::lock_guard<std::mutex> lock(conns_mu);
        return std::all_of(conns.begin(), conns.end(), [ignore_streams](const auto& c) {
            return c.second->done.load() || (ignore_streams && c.second->streaming.load());
        });
    }

    void do_stop() {
        if (!started) {
            std::lock_guard<std::mutex> lock(stopped_mu);
            stopped = true;
            stopped_cv.notify_all();
            return;
        }
        // 1. Health reports draining; new chats get 503 not_ready.
        state.store(State::draining);
        // 2. Stop accepting.
        stopping.store(true);
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        // 3. In-flight requests get the grace period, then are cancelled.
        {
            std::unique_lock<std::mutex> lock(inflight_mu);
            inflight_cv.wait_for(lock, opts.shutdown_grace, [this] { return inflight.empty(); });
            for (const auto& s : inflight) {
                s->cancel();
            }
        }
        const auto grace_deadline = std::chrono::steady_clock::now() + opts.shutdown_grace;
        if (!wait_until([this] { return all_connections_done(true); }, grace_deadline)) {
            hard_stop.store(true);  // responses to peers that stopped reading
            wait_until([this] { return all_connections_done(true); },
                       std::chrono::steady_clock::time_point::max());
        }
        // 4. engine.stopped (and a final telemetry.dropped) reach the hub.
        {
            std::lock_guard<std::mutex> models_lock(models_mu);
            served.clear();
        }
        std::unique_ptr<Engine> doomed;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            backend.reset();
            doomed = std::move(engine);
        }
        doomed.reset();
        // 5. Telemetry streams deliver what they hold, then end.
        hub->close();
        if (!wait_until([this] { return all_connections_done(false); },
                        std::chrono::steady_clock::now() + opts.shutdown_grace)) {
            hard_stop.store(true);
        }
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lock(conns_mu);
            for (auto& c : conns) {
                threads.push_back(std::move(c.first));
            }
            conns.clear();
        }
        for (auto& t : threads) {
            t.join();
        }
        state.store(State::stopped);
        std::lock_guard<std::mutex> lock(stopped_mu);
        stopped = true;
        stopped_cv.notify_all();
    }
};

// ================================================================= Server

Status validate_options(const ServerOptions& o) {
    if (o.host.empty()) {
        return Status(ErrorCode::invalid_argument, "--host must not be empty");
    }
    if (!is_loopback_bind(o.host) && o.token.empty()) {
        return Status(ErrorCode::invalid_argument, "--host " + o.host +
                                                       " is not a loopback address; a non-loopback bind requires "
                                                       "--token-file");
    }
    if (o.capture_text && o.token.empty()) {
        return Status(ErrorCode::invalid_argument, "--capture-text exports generated text and requires --token-file");
    }
    if (o.backend.backend.empty()) {
        return Status(ErrorCode::invalid_argument, "--backend is required (mock, ollama or llamacpp)");
    }
    if (o.models.empty() && !is_synthetic_backend(o.backend.backend)) {
        return Status(ErrorCode::invalid_argument, "--model is required for backend " + o.backend.backend +
                                                       " (list models with `sonder-infer models --backend " +
                                                       o.backend.backend + "`)");
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
    if (o.max_connections == 0 || o.max_body_bytes == 0 || o.telemetry_buffer == 0 || o.max_subscribers == 0) {
        return Status(ErrorCode::invalid_argument,
                      "--max-connections, --max-body-bytes and --telemetry-buffer must be positive");
    }
    for (const auto& origin : o.cors_origins) {
        if (origin.empty() || origin == "*" || origin.find_first_of(" \t\r\n,") != std::string::npos) {
            return Status(ErrorCode::invalid_argument,
                          "--cors-origin takes one exact origin such as http://127.0.0.1:4173 (no wildcards)");
        }
    }
    return Status::success();
}

Server::Server(ServerOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

Server::~Server() { stop(); }

Status Server::start(const std::function<void()>& on_listening) {
    Status st = impl_->start(on_listening);
    if (!st.ok() && impl_->started) {
        stop();
    }
    return st;
}

void Server::stop() {
    std::call_once(impl_->stop_once, [this] { impl_->do_stop(); });
    wait();
}

void Server::wait() {
    std::unique_lock<std::mutex> lock(impl_->stopped_mu);
    impl_->stopped_cv.wait(lock, [this] { return impl_->stopped; });
}

std::uint16_t Server::port() const noexcept { return impl_->bound_port; }
std::string Server::url() const { return impl_->url; }
std::string Server::instance_id() const { return impl_->instance_id; }

Engine* Server::engine() noexcept {
    std::lock_guard<std::mutex> lock(impl_->engine_mu);
    return impl_->engine.get();
}

}  // namespace sonder::inference::server
