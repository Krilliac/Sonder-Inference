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
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#include "cli/cli_spec.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/server.hpp"
#include "sonder/inference/telemetry.hpp"
#include "request_path.hpp"

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

constexpr const char* kServeUsage = R"(sonder-infer serve - local HTTP API and live telemetry (docs/SERVER.md)

Usage:
  sonder-infer serve --backend mock|ollama|llamacpp|llamaserver [--model ID]... [options]

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
                          (models may still be loading: poll /v1/sonder/health for
                          200); removed again when the server exits
  --log-format text|json  access log and diagnostics on stderr (default text)

Backend:
  --backend NAME          mock (synthetic, tests only), ollama, llamacpp,
                          llamaserver (external llama-server or OpenAI upstream)
                          (env SONDER_INFER_BACKEND)
  --model ID              model to serve (repeatable; the first is the default and
                          answers to "default"; env SONDER_INFER_MODEL). The mock
                          backend serves "mock" when no model is given.
  --ollama-url URL        Ollama base URL (env SONDER_OLLAMA_URL, then OLLAMA_HOST)
  --ollama-allow-remote   allow a non-loopback Ollama host (needs https, i.e. a
                          SONDER_WITH_TLS=ON build)
  --model-dir DIR         llama.cpp: directory with *.gguf files (repeatable)
  --device ID             device to load models on (llama.cpp: gpu:0 offloads to
                          the GPU; default cpu)
  --gpu-layers N          llama.cpp: layers offloaded to the GPU (-1 = all, default)
  --context-length N      llama.cpp: context length (0 = model training context)
  --moe-experts WHERE     llama.cpp: cpu keeps MoE expert weights in system RAM
                          while attention stays on the GPU; gpu (default) leaves
                          placement to --gpu-layers
  --tensor-override P=D   llama.cpp: place tensors matching regex P on device D
                          (cpu, or a llama.cpp device such as Vulkan0); repeatable
  --mock-delay-ms N       mock backend per-token delay
  --llamaserver-config PATH  JSON config (mode, URL/executable, args, TLS and timeouts)
  --llamaserver-url URL     attach upstream URL (default http://127.0.0.1:8080)
  --llamaserver-executable PATH  spawn executable
  --llamaserver-arg ARG     repeatable spawn argument (passed verbatim)
  --llamaserver-mode MODE   attach or spawn

Telemetry (Observatory envelope v1; live at /v1/telemetry/sse and /v1/telemetry/ndjson):
  --telemetry-level L     off|metrics|standard|deep (default standard)
  --telemetry-buffer N    retained events for resume (default 8192)
  --telemetry PATH        also write JSONL to PATH ('-' for stderr)
  --capture-text          include generated text in token events (requires --token-file)

Exit status: 0 after a clean shutdown (SIGINT/SIGTERM; a second signal exits
at once), 1 on a runtime error (e.g. port in use), 2 on a usage error.
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
        {"llamaserver-config", Kind::value}, {"llamaserver-url", Kind::value},
        {"llamaserver-executable", Kind::value}, {"llamaserver-arg", Kind::repeat},
        {"llamaserver-mode", Kind::value},
        {"gpu-layers", Kind::value},     {"context-length", Kind::value},
        {"device", Kind::value},
        {"moe-experts", Kind::value},    {"tensor-override", Kind::repeat},
    };
    static const std::vector<std::pair<std::string, Kind>> kAll = [] {
        auto all = kSpec;
        for (const char* flag : detail::kRequestPathFlags) all.emplace_back(flag, Kind::value);
        return all;
    }();
    return kAll;
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
            // Same wording and did-you-mean suggestion as the other
            // sonder-infer commands (src/cli/cli_spec.hpp).
            std::vector<std::string_view> names;
            for (const auto& entry : spec()) {
                names.emplace_back(entry.first);
            }
            error = sonder::cli::unknown_option_message(key, names);
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
        out << kServeUsage << '\n' << detail::kRequestPathUsage;
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
        return usage_error("--backend is required (mock, ollama, llamacpp or llamaserver; or set SONDER_INFER_BACKEND)");
    }
    o.models = a.all("model");
    o.device = a.get("device").value_or("");
    if (o.models.empty() && env.model) {
        o.models.push_back(*env.model);
    }
    o.backend.ollama_url = a.get("ollama-url").value_or(env.ollama_url.value_or(""));
    o.backend.ollama_allow_remote = a.flags.count("ollama-allow-remote") != 0;
    if (auto path = a.get("llamaserver-config")) {
        if (auto st = load_llamaserver_config(*path, o.backend); !st.ok()) return usage_error(st.message());
    }
    if (auto v = a.get("llamaserver-mode")) o.backend.llamaserver_mode = *v;
    if (auto v = a.get("llamaserver-url")) o.backend.llamaserver_url = *v;
    if (auto v = a.get("llamaserver-executable")) o.backend.llamaserver_executable = *v;
    if (!a.all("llamaserver-arg").empty()) o.backend.llamaserver_args = a.all("llamaserver-arg");
    if (o.backend.backend == "llamaserver") {
        if (o.backend.llamaserver_mode == "spawn") {
            if (o.backend.llamaserver_executable.empty()) return usage_error("--llamaserver-executable is required in spawn mode");
            for (const auto& arg : o.backend.llamaserver_args) {
                if (arg == "--host" || arg.rfind("--host=", 0) == 0 || arg == "--port" || arg.rfind("--port=", 0) == 0)
                    return usage_error("llamaserver spawn args must not override --host or --port");
            }
        } else if (!o.backend.llamaserver_mode.empty() && o.backend.llamaserver_mode != "attach") {
            return usage_error("--llamaserver-mode must be attach or spawn");
        }
    }
    o.backend.model_dirs = a.all("model-dir");
    if (auto v = a.get("gpu-layers")) {
        if (*v == "-1") {
            o.backend.llamacpp_gpu_layers = -1;
        } else if (!parse_uint(*v, 100000, n)) {
            return usage_error("--gpu-layers must be -1 or a non-negative integer");
        } else {
            o.backend.llamacpp_gpu_layers = static_cast<std::int32_t>(n);
        }
    }
    if (auto v = a.get("context-length")) {
        if (!parse_uint(*v, 1u << 22, n)) return usage_error("--context-length must be an integer from 0 to 4194304");
        o.backend.llamacpp_context_length = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("moe-experts")) {
        if (*v == "cpu") {
            o.backend.llamacpp_tensor_overrides.push_back(std::string(kMoeExpertTensorOverridePattern) + "=cpu");
        } else if (*v != "gpu") {
            return usage_error("--moe-experts must be cpu or gpu");
        }
    }
    for (const std::string& spec : a.all("tensor-override")) {
        if (auto parsed = parse_tensor_override(spec); !parsed) return usage_error(parsed.status().message());
        o.backend.llamacpp_tensor_overrides.push_back(spec);
    }
    if (Status st = detail::parse_request_path_flags([&a](const std::string& k) { return a.get(k); }, o); !st.ok()) {
        return usage_error(st.message());
    }
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
    std::string models;
    for (std::size_t i = 0; i < o.models.size(); ++i) {
        models += (i ? ", " : "") + o.models[i] + (i == 0 ? " (default)" : "");
    }
    if (models.empty()) {
        models = "mock (default)";
    }
    const bool mock = is_synthetic_backend(o.backend.backend);
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
            log->line(json::Value(json::Object{{"time", utc_timestamp_now()},
                                               {"level", "info"},
                                               {"message", "listening"},
                                               {"url", base},
                                               {"bind", bind_text()},
                                               {"api_version", kApiVersion},
                                               {"backend", o.backend.backend},
                                               {"models", models},
                                               {"status", "loading models"}})
                          .dump());
        } else {
            log->line("sonder-infer serve: listening on " +
                      (wildcard ? bind_text() + " (all interfaces; local URL " + base + ")" : base) +
                      " (api_version " + std::to_string(kApiVersion) + ")");
            log->line("sonder-infer serve: loading models on " + o.backend.backend + ": " + models +
                      " (health reports 503 starting until they are loaded)");
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
    std::vector<std::string> origins = o.cors_origins;
    if (json_log) {
        json::Array origin_list;
        for (const auto& s : origins) origin_list.emplace_back(s);
        log->line(json::Value(json::Object{
                                  {"time", utc_timestamp_now()},
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
                                  {"synthetic", mock}})
                      .dump());
    } else {
        std::string cors;
        for (const auto& s : origins) cors += (cors.empty() ? "" : ", ") + s;
        log->line("sonder-infer serve: ready on " + base + (wildcard ? " (bound to " + bind_text() + ")" : ""));
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
