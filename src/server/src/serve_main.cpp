// `sonder-infer serve`: argument parsing, startup banner, ready file,
// signal handling and graceful shutdown around sonder::inference::server::Server.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif

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

extern "C" void on_shutdown_signal(int) { g_shutdown.store(1); }

constexpr const char* kServeUsage = R"(sonder-infer serve - local HTTP API and live telemetry (docs/SERVER.md)

Usage:
  sonder-infer serve --backend mock|ollama|llamacpp [--model ID]... [options]

Server:
  --host HOST             listen address (default 127.0.0.1; non-loopback needs --token-file)
  --port N                listen port (default 11437; 0 = ephemeral)
  --token-file PATH       require "Authorization: Bearer <token>" (token read from PATH)
  --cors-origin ORIGIN    allow a browser origin (repeatable, exact match)
  --no-default-cors       drop the default Observatory origins
  --max-connections N     concurrent connections (default 64)
  --max-body-bytes N      request body limit (default 4194304)
  --shutdown-grace-ms N   time in-flight requests get on shutdown (default 5000)
  --ready-file PATH       write {"url","pid","instance_id","api_version"} once listening
  --log-format text|json  access log and diagnostics on stderr (default text)

Backend:
  --backend NAME          mock (synthetic, tests only), ollama, llamacpp
                          (env SONDER_INFER_BACKEND)
  --model ID              model to serve (repeatable; the first is the default and
                          answers to "default"; env SONDER_INFER_MODEL). The mock
                          backend serves "mock" when no model is given.
  --ollama-url URL        Ollama base URL (env SONDER_OLLAMA_URL, then OLLAMA_HOST)
  --ollama-allow-remote   allow a non-loopback Ollama host (needs https, i.e. a
                          SONDER_WITH_TLS=ON build)
  --model-dir DIR         llama.cpp: directory with *.gguf files (repeatable)
  --mock-delay-ms N       mock backend per-token delay

Telemetry (Observatory envelope v1; live at /v1/telemetry/sse and /v1/telemetry/ndjson):
  --telemetry-level L     off|metrics|standard|deep (default standard)
  --telemetry-buffer N    retained events for resume (default 8192)
  --telemetry PATH        also write JSONL to PATH ('-' for stderr)
  --capture-text          include generated text in token events (requires --token-file)

Exit status: 0 after a clean shutdown (SIGINT/SIGTERM), 1 on a runtime error
(e.g. port in use), 2 on a usage error.
)";

enum class Kind { value, repeat, flag };

const std::vector<std::pair<std::string, Kind>>& spec() {
    static const std::vector<std::pair<std::string, Kind>> kSpec = {
        {"host", Kind::value},           {"port", Kind::value},
        {"backend", Kind::value},        {"model", Kind::repeat},
        {"ollama-url", Kind::value},     {"ollama-allow-remote", Kind::flag},
        {"model-dir", Kind::repeat},     {"token-file", Kind::value},
        {"cors-origin", Kind::repeat},   {"no-default-cors", Kind::flag},
        {"telemetry-level", Kind::value}, {"telemetry-buffer", Kind::value},
        {"telemetry", Kind::value},      {"capture-text", Kind::flag},
        {"max-connections", Kind::value}, {"max-body-bytes", Kind::value},
        {"shutdown-grace-ms", Kind::value}, {"ready-file", Kind::value},
        {"mock-delay-ms", Kind::value},  {"log-format", Kind::value},
    };
    return kSpec;
}

struct ParsedArgs {
    std::vector<std::pair<std::string, std::string>> values;  // in order
    std::set<std::string> flags;
    bool help = false;

    [[nodiscard]] std::optional<std::string> get(const std::string& key) const {
        for (const auto& [k, v] : values) {
            if (k == key) {
                return v;
            }
        }
        return std::nullopt;
    }
    [[nodiscard]] std::vector<std::string> all(const std::string& key) const {
        std::vector<std::string> out;
        for (const auto& [k, v] : values) {
            if (k == key) {
                out.push_back(v);
            }
        }
        return out;
    }
};

bool parse(const std::vector<std::string>& args, ParsedArgs& out, std::string& error) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            out.help = true;
            continue;
        }
        if (arg.rfind("--", 0) != 0 || arg.size() == 2) {
            error = "unexpected argument: " + arg;
            return false;
        }
        std::string key = arg.substr(2);
        std::optional<std::string> inline_value;
        if (const auto eq = key.find('='); eq != std::string::npos) {
            inline_value = key.substr(eq + 1);
            key.resize(eq);
        }
        const auto it = std::find_if(spec().begin(), spec().end(), [&](const auto& s) { return s.first == key; });
        if (it == spec().end()) {
            error = "unknown option --" + key;
            return false;
        }
        if (it->second == Kind::flag) {
            if (inline_value) {
                error = "--" + key + " takes no value";
                return false;
            }
            out.flags.insert(key);
            continue;
        }
        std::string value;
        if (inline_value) {
            value = *inline_value;
        } else {
            if (i + 1 >= args.size()) {
                error = "missing value for --" + key;
                return false;
            }
            value = args[++i];
        }
        if (it->second == Kind::value && out.get(key)) {
            error = "--" + key + " given more than once";
            return false;
        }
        out.values.emplace_back(key, std::move(value));
    }
    return true;
}

bool parse_uint(const std::string& text, std::uint64_t max, std::uint64_t& out) {
    if (text.empty() || text.size() > 19 || text.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    std::uint64_t n = 0;
    for (const char c : text) {
        n = n * 10 + static_cast<std::uint64_t>(c - '0');
    }
    if (n > max) {
        return false;
    }
    out = n;
    return true;
}

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

// Serializes writes to the shared error stream (access log, diagnostics and
// the optional '--telemetry -' sink come from different threads).
class LockedLog {
public:
    explicit LockedLog(std::ostream& stream) : stream_(stream) {}
    void line(std::string_view text) {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_.write(text.data(), static_cast<std::streamsize>(text.size()));
        stream_.put('\n');
        stream_.flush();
    }

private:
    std::mutex mutex_;
    std::ostream& stream_;
};

class LockedLogSink final : public TelemetrySink {
public:
    explicit LockedLogSink(std::shared_ptr<LockedLog> log) : log_(std::move(log)) {}
    void write(std::string_view json_line) override { log_->line(json_line); }

private:
    std::shared_ptr<LockedLog> log_;
};

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
    auto log = std::make_shared<LockedLog>(err);
    const auto usage_error = [&](const std::string& message) {
        log->line("error: " + message + " (see `sonder-infer serve --help`)");
        return 2;
    };

    ParsedArgs a;
    std::string error;
    if (!parse(args, a, error)) {
        return usage_error(error);
    }
    if (a.help) {
        out << kServeUsage;
        out.flush();
        return 0;
    }

    ServerOptions o;
    const BackendEnvDefaults env = backend_env_defaults();
    o.host = a.get("host").value_or("127.0.0.1");
    std::uint64_t n = 0;
    if (auto v = a.get("port")) {
        if (!parse_uint(*v, 65535, n)) return usage_error("--port must be an integer from 0 to 65535");
        o.port = static_cast<std::uint16_t>(n);
    }
    o.backend.backend = a.get("backend").value_or(env.backend.value_or(""));
    if (o.backend.backend.empty()) {
        return usage_error("--backend is required (mock, ollama or llamacpp; or set SONDER_INFER_BACKEND)");
    }
    o.models = a.all("model");
    if (o.models.empty() && env.model) {
        o.models.push_back(*env.model);
    }
    o.backend.ollama_url = a.get("ollama-url").value_or(env.ollama_url.value_or(""));
    o.backend.ollama_allow_remote = a.flags.count("ollama-allow-remote") != 0;
    o.backend.model_dirs = a.all("model-dir");
    if (auto v = a.get("mock-delay-ms")) {
        if (!parse_uint(*v, 60000, n)) return usage_error("--mock-delay-ms must be an integer from 0 to 60000");
        o.backend.mock_token_delay = std::chrono::milliseconds(n);
    }
    if (auto path = a.get("token-file")) {
        std::ifstream in(*path, std::ios::binary);
        if (!in) {
            return usage_error("cannot read --token-file " + *path);
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        o.token = trim(ss.str());
        if (o.token.empty() || o.token.find_first_of(" \t\r\n") != std::string::npos) {
            return usage_error("--token-file must contain one non-empty token");
        }
#if !defined(_WIN32)
        struct stat st {};
        if (::stat(path->c_str(), &st) == 0 && (st.st_mode & 077) != 0) {
            log->line("sonder-infer serve: warning: --token-file " + *path +
                      " is readable by group or others; restrict it with chmod 600");
        }
#endif
    }
    o.cors_origins = a.all("cors-origin");
    o.default_cors = a.flags.count("no-default-cors") == 0;
    if (auto v = a.get("telemetry-level")) {
        const auto level = parse_telemetry_level(*v);
        if (!level) return usage_error("--telemetry-level must be off, metrics, standard or deep");
        o.telemetry_level = *level;
    }
    if (auto v = a.get("telemetry-buffer")) {
        if (!parse_uint(*v, 1u << 20, n) || n == 0) return usage_error("--telemetry-buffer must be from 1 to 1048576");
        o.telemetry_buffer = static_cast<std::size_t>(n);
    }
    o.capture_text = a.flags.count("capture-text") != 0;
    if (auto v = a.get("max-connections")) {
        if (!parse_uint(*v, 100000, n) || n == 0) return usage_error("--max-connections must be from 1 to 100000");
        o.max_connections = static_cast<std::size_t>(n);
    }
    if (auto v = a.get("max-body-bytes")) {
        if (!parse_uint(*v, 1ull << 30, n) || n == 0) {
            return usage_error("--max-body-bytes must be from 1 to 1073741824");
        }
        o.max_body_bytes = static_cast<std::size_t>(n);
    }
    if (auto v = a.get("shutdown-grace-ms")) {
        if (!parse_uint(*v, 600000, n)) return usage_error("--shutdown-grace-ms must be from 0 to 600000");
        o.shutdown_grace = std::chrono::milliseconds(n);
    }
    if (auto v = a.get("log-format")) {
        if (*v != "text" && *v != "json") return usage_error("--log-format must be text or json");
        o.log_format = *v == "json" ? LogFormat::json : LogFormat::text;
    }
    if (auto path = a.get("telemetry")) {
        if (*path == "-") {
            o.extra_sinks.push_back(std::make_shared<LockedLogSink>(log));
        } else {
            Status st;
            auto sink = make_jsonl_file_sink(*path, false, &st);
            if (!sink) {
                log->line("error: " + st.message());
                return 1;
            }
            o.extra_sinks.push_back(std::shared_ptr<TelemetrySink>(std::move(sink)));
        }
    }
    if (Status st = validate_options(o); !st.ok()) {
        return usage_error(st.message());
    }
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
    if (!o.token.empty() && o.host != "127.0.0.1" && o.host != "localhost" && o.host != "::1" &&
        o.host != "[::1]" && o.host.rfind("127.", 0) != 0) {
        diag("warning", "listening on a non-loopback address without TLS: the bearer token and all traffic travel "
                        "in cleartext; put a TLS-terminating proxy in front");
    }

    const std::string ready_path = a.get("ready-file").value_or("");
    Server server(o);
    std::string ready_error;
    SignalGuard signals;
    Status started = server.start([&] {
        if (ready_path.empty()) {
            return;
        }
        const json::Object doc{{"url", server.url()},
                               {"pid", current_pid()},
                               {"instance_id", server.instance_id()},
                               {"api_version", kApiVersion}};
        (void)write_ready_file(ready_path, json::Value(doc).dump(), ready_error);
    });
    if (!ready_error.empty()) {
        diag("error", ready_error);
        server.stop();
        return 1;
    }
    if (!started.ok()) {
        std::string message = started.message();
        if (started.code() == ErrorCode::unavailable && message.find("in use") != std::string::npos) {
            message += "; pick another --port, or --port 0 for an ephemeral port";
        }
        diag("error", message);
        return 1;
    }

    // Startup banner.
    const std::string base = server.url();
    std::vector<std::string> origins = o.cors_origins;
    std::string models;
    for (std::size_t i = 0; i < o.models.size(); ++i) {
        models += (i ? ", " : "") + o.models[i] + (i == 0 ? " (default)" : "");
    }
    if (models.empty()) {
        models = "mock (default)";
    }
    const bool mock = is_synthetic_backend(o.backend.backend);
    if (json_log) {
        json::Array origin_list;
        for (const auto& s : origins) origin_list.emplace_back(s);
        log->line(json::Value(json::Object{
                                  {"time", utc_timestamp_now()},
                                  {"level", "info"},
                                  {"message", "listening"},
                                  {"url", base},
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
                                  {"synthetic", mock}})
                      .dump());
    } else {
        std::string cors;
        for (const auto& s : origins) cors += (cors.empty() ? "" : ", ") + s;
        log->line("sonder-infer serve: listening on " + base + " (api_version " + std::to_string(kApiVersion) + ")");
        log->line("  auth:      " + std::string(o.token.empty() ? "none (loopback only)" : "bearer token required"));
        log->line("  cors:      " + (cors.empty() ? std::string("no explicit origins") : cors) +
                  (o.default_cors ? "; default Observatory origins for GET routes" : "; defaults disabled"));
        log->line("  backend:   " + o.backend.backend + "; models: " + models);
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
