// `sonder-infer serve`: startup banner, ready file, signal handling and
// graceful shutdown around sonder::inference::server::Server.
// CLI flag parsing and ServerOptions assembly live in serve_options.cpp;
// launch-profile catalog wiring lives in serve_profile_catalog.cpp.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <unistd.h>
#endif

#include "request_path.hpp"
#include "serve_options.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server {

namespace {

// Lock-free atomics are async-signal-safe; the flag is also read by the
// serve loop and set by request_shutdown() from other threads.
std::atomic<int> g_shutdown{0};
static_assert(std::atomic<int>::is_always_lock_free, "shutdown flag must be lock-free");

// The first SIGINT/SIGTERM asks for a graceful drain; the handler then
// restores the default action, so a second one terminates at once (for
// example while a model load or a drain is stuck on the backend).
extern "C" void on_shutdown_signal(int sig) {
    g_shutdown.store(1);
    std::signal(sig, SIG_DFL);
#if !defined(_WIN32)
    std::signal(sig == SIGINT ? SIGTERM : SIGINT, SIG_DFL);
#endif
}

int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

struct SignalGuard {
    using Handler = void (*)(int);
    Handler previous_int = SIG_DFL;
    Handler previous_term = SIG_DFL;
    SignalGuard() {
        previous_int = std::signal(SIGINT, on_shutdown_signal);
        previous_term = std::signal(SIGTERM, on_shutdown_signal);
    }
    ~SignalGuard() {
        std::signal(SIGINT, previous_int == SIG_ERR ? SIG_DFL : previous_int);
        std::signal(SIGTERM, previous_term == SIG_ERR ? SIG_DFL : previous_term);
    }
    SignalGuard(const SignalGuard&) = delete;
    SignalGuard& operator=(const SignalGuard&) = delete;
};

// Removes the ready file when serve_main() returns (error or shutdown), so
// discovery never points at a dead server.
struct ReadyFileGuard {
    std::string path;
    bool written = false;
    ~ReadyFileGuard() {
        if (written) {
            std::remove(path.c_str());
        }
    }
};

bool write_ready_file(const std::string& path, const std::string& content, std::string& error) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            error = "cannot write ready file " + tmp;
            return false;
        }
        out << content << "\n";
        if (!out) {
            error = "cannot write ready file " + tmp;
            return false;
        }
    }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        error = "cannot move ready file into place: " + path;
        return false;
    }
    return true;
}

}  // namespace

void request_shutdown() noexcept { g_shutdown.store(1); }

int serve_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    g_shutdown.store(0);
    auto log = std::make_shared<detail::LockedLog>(err);
    // Last to run on every return path (see LockedLogCloser): after it, no
    // thread writes to `err`, whichever way the server was stopped.
    const detail::LockedLogCloser close_log{log};

    detail::ParsedServeArgs a;
    std::string error;
    if (!detail::parse_serve_args(args, a, error)) {
        log->line("error: " + error + " (see `sonder-infer serve --help`)");
        return 2;
    }
    if (a.help) {
        out << detail::kServeUsage << '\n' << detail::kRequestPathUsage;
        out.flush();
        return 0;
    }

    detail::ServeCliConfig cfg;
    if (const int rc = detail::build_serve_options(a, log, cfg); rc != 0) {
        return rc;
    }
    ServerOptions& o = cfg.options;
    const bool json_log = o.log_format == LogFormat::json;
    o.log = [log](const std::string& line) { log->line(line); };
    const auto diag = [&](const char* level, const std::string& message) {
        if (json_log) {
            log->line(json::Value(json::Object{{"time", utc_timestamp_now()}, {"level", level}, {"message", message}})
                          .dump());
        } else {
            log->line(std::string("sonder-infer serve: ") + level + ": " + message);
        }
    };
    for (const auto& note : cfg.profile_notes) {
        diag("info", note);
    }
    if (!o.token.empty() && o.host != "127.0.0.1" && o.host != "localhost" && o.host != "::1" &&
        o.host != "[::1]" && o.host.rfind("127.", 0) != 0) {
        diag("warning", "listening on a non-loopback address without TLS: the bearer token and all traffic travel "
                        "in cleartext; put a TLS-terminating proxy in front");
    }
    if (const std::string notice = text_capture_startup_notice(combine_text_capture(o.text_capture, o.capture_text));
        !notice.empty()) {
        diag("warning", notice);
    }

    const std::string& ready_path = cfg.ready_path;
    std::string models;
    for (std::size_t i = 0; i < o.models.size(); ++i) {
        models += (i ? ", " : "") + o.models[i] + (i == 0 ? " (default)" : "");
    }
    if (models.empty()) {
        models = "mock (default)";
    }
    if (o.profile_bindings.size() > 1) {
        diag("info", "serving " + std::to_string(o.profile_bindings.size()) + " launch profiles: " + models);
    }
    const bool mock = is_synthetic_backend(o.backend.backend);
    // Residency summary for the banner; empty with the defaults, which keep
    // the banner exactly as it was before these options existed.
    std::string residency;
    if (o.lazy_models || o.model_idle_ttl.count() > 0 || o.max_resident_models > 0) {
        residency = std::string(o.lazy_models ? "lazy" : "eager") + ", idle TTL " +
                    (o.model_idle_ttl.count() > 0
                         ? std::to_string(std::chrono::duration_cast<std::chrono::seconds>(o.model_idle_ttl).count()) +
                               " s"
                         : std::string("off")) +
                    ", max resident " +
                    (o.max_resident_models > 0 ? std::to_string(o.max_resident_models) : std::string("unlimited"));
    }
    Server server(o);
    std::string ready_error;
    ReadyFileGuard ready_file{ready_path};
    SignalGuard signals;
    // Where the socket is bound, and the URL local clients use. They differ
    // for wildcard binds (0.0.0.0 / ::), which serve every interface.
    const auto bind_text = [&] {
        const std::string h = o.host.find(':') != std::string::npos && o.host.front() != '[' ? "[" + o.host + "]"
                                                                                              : o.host;
        return h + ":" + std::to_string(server.port());
    };
    const bool wildcard = o.host == "0.0.0.0" || o.host == "::" || o.host == "[::]";
    const auto on_listening = [&] {
        const std::string base = server.url();
        if (json_log) {
            json::Object listening_doc{{"time", utc_timestamp_now()},
                                       {"level", "info"},
                                       {"message", "listening"},
                                       {"url", base},
                                       {"bind", bind_text()},
                                       {"api_version", kApiVersion},
                                       {"backend", o.backend.backend},
                                       {"models", models},
                                       {"status", o.lazy_models ? "registering models" : "loading models"}};
            if (o.profile_bindings.size() > 1) {
                listening_doc.set("served_profiles", static_cast<std::uint64_t>(o.profile_bindings.size()));
            }
            log->line(json::Value(std::move(listening_doc)).dump());
        } else {
            log->line("sonder-infer serve: listening on " +
                      (wildcard ? bind_text() + " (all interfaces; local URL " + base + ")" : base) +
                      " (api_version " + std::to_string(kApiVersion) + ")");
            if (o.lazy_models) {
                log->line("sonder-infer serve: registering models on " + o.backend.backend + ": " + models +
                          " (lazy residency: each model loads on its first request)");
            } else {
                log->line("sonder-infer serve: loading models on " + o.backend.backend + ": " + models +
                          " (health reports 503 starting until they are loaded)");
            }
        }
        if (ready_path.empty()) {
            return;
        }
        const json::Object doc{{"url", base},
                               {"pid", current_pid()},
                               {"instance_id", server.instance_id()},
                               {"api_version", kApiVersion}};
        ready_file.written = write_ready_file(ready_path, json::Value(doc).dump(), ready_error);
    };
    // start() blocks while models load; a shutdown request (signal or hook)
    // meanwhile stops the server, which abandons the start after the backend
    // call in progress returns.
    std::atomic<bool> start_done{false};
    std::thread startup_watch([&] {
        while (!start_done.load() && g_shutdown.load() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!start_done.load()) {
            diag("info", "shutdown requested while starting; waiting for the backend call in progress to return "
                         "(send the signal again to exit immediately)");
            server.stop();
        }
    });
    Status started = server.start(on_listening);
    start_done.store(true);
    startup_watch.join();
    if (!ready_error.empty()) {
        diag("error", ready_error);
        server.stop();
        return 1;
    }
    if (!started.ok() && started.code() == ErrorCode::cancelled && g_shutdown.load() != 0) {
        server.stop();
        diag("info", "stopped before the models finished loading");
        return 0;
    }
    if (!started.ok()) {
        std::string message = started.message();
        if (started.code() == ErrorCode::unavailable && message.find("in use") != std::string::npos) {
            message += "; pick another --port, or --port 0 for an ephemeral port";
        }
        diag("error", message);
        return 1;
    }

    // Startup banner (the server is ready).
    const std::string base = server.url();
    const std::vector<std::string>& origins = o.cors_origins;
    if (json_log) {
        json::Array origin_list;
        for (const auto& s : origins) origin_list.emplace_back(s);
        json::Object ready_doc{{"time", utc_timestamp_now()},
                               {"level", "info"},
                               {"message", "ready"},
                               {"url", base},
                               {"bind", bind_text()},
                               {"api_version", kApiVersion},
                               {"auth", o.token.empty() ? "none" : "bearer"},
                               {"cors_origins", std::move(origin_list)},
                               {"default_cors", o.default_cors},
                               {"backend", o.backend.backend},
                               {"models", models},
                               {"telemetry_level", to_string(o.telemetry_level)},
                               {"discovery_url", base + "/.well-known/sonder-telemetry"},
                               {"sse_url", base + "/v1/telemetry/sse"},
                               {"ndjson_url", base + "/v1/telemetry/ndjson"},
                               {"synthetic", mock}};
        if (o.profile_bindings.size() > 1) {
            ready_doc.set("served_profiles", static_cast<std::uint64_t>(o.profile_bindings.size()));
        }
        if (!residency.empty()) {
            ready_doc.set("residency", residency);
        }
        log->line(json::Value(std::move(ready_doc)).dump());
    } else {
        std::string cors;
        for (const auto& s : origins) cors += (cors.empty() ? "" : ", ") + s;
        log->line("sonder-infer serve: ready on " + base + (wildcard ? " (bound to " + bind_text() + ")" : ""));
        log->line("  auth:      " + std::string(o.token.empty() ? "none (loopback only)" : "bearer token required"));
        log->line("  cors:      " + (cors.empty() ? std::string("no explicit origins") : cors) +
                  (o.default_cors ? "; default Observatory origins for GET routes" : "; defaults disabled"));
        log->line("  backend:   " + o.backend.backend + "; models: " + models);
        if (o.profile_bindings.size() > 1) {
            log->line("  profiles:  " + std::to_string(o.profile_bindings.size()) + " active");
        }
        if (!residency.empty()) {
            log->line("  residency: " + residency);
        }
        log->line("  telemetry: level " + std::string(to_string(o.telemetry_level)) + ", discovery " + base +
                  "/.well-known/sonder-telemetry, sse " + base + "/v1/telemetry/sse, ndjson " + base +
                  "/v1/telemetry/ndjson");
    }
    if (mock) {
        diag("warning", "MOCK BACKEND - synthetic output, not a quality or performance signal");
    }

    while (g_shutdown.load() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    diag("info", "shutting down (draining in-flight requests)");
    server.stop();
    diag("info", "stopped");
    return 0;
}

}  // namespace sonder::inference::server
