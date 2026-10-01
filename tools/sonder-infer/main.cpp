// sonder-infer: command-line interface over the Sonder Inference library.
// Reference: docs/CLI.md (commands, environment, exit codes, JSON shapes).
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <streambuf>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "sonder/inference.hpp"
#include "sonder_inference.h"
// Header-only CLI helpers (option specs, checked values, environment, chat
// files and the chat loop); shared with the unit tests (tests/test_cli_*.cpp).
#include "../../src/cli/cli_args.hpp"
#include "../../src/cli/cli_env.hpp"
#include "../../src/cli/cli_spec.hpp"
#include "../../src/cli/cli_values.hpp"
#include "../../src/cli/sonder_infer_commands.hpp"
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "../../src/backends/llamaserver/tune.hpp"
#endif
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/server.hpp"
#else
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
#include "sonder/inference/backends/llamacpp.hpp"
#endif
#endif
#if defined(SONDER_HAS_BENCH)
#include "sonder/inference/benchmark.hpp"
#endif
// After the Sonder headers, so Windows macros cannot reach them.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#endif

namespace si = sonder::inference;
namespace cli = sonder::cli;

namespace {

using cli::Args;

// ---------------------------------------------------------------------------
// stderr: status lines, stats, errors and the '--telemetry -' sink come from
// the main thread and the telemetry writer thread. Every complete line is
// written under one mutex, so JSON lines never interleave.
// ---------------------------------------------------------------------------
std::mutex& stderr_mutex() {
    static std::mutex m;
    return m;
}

class LineLockedBuf final : public std::streambuf {
public:
    explicit LineLockedBuf(std::ostream& target) : target_(target) {}

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
        const char c = traits_type::to_char_type(ch);
        pending_.push_back(c);
        if (c == '\n') flush_pending();
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i) overflow(traits_type::to_int_type(s[i]));
        return n;
    }
    int sync() override {
        flush_pending();
        return 0;
    }

private:
    void flush_pending() {
        if (pending_.empty()) return;
        std::lock_guard<std::mutex> lock(stderr_mutex());
        target_.write(pending_.data(), static_cast<std::streamsize>(pending_.size()));
        target_.flush();
        pending_.clear();
    }
    std::ostream& target_;
    std::string pending_;
};

std::ostream& err() {
    static LineLockedBuf buf(std::cerr);
    static std::ostream stream(&buf);
    return stream;
}

class StderrTelemetrySink final : public si::TelemetrySink {
public:
    void write(std::string_view json_line) override {
        std::lock_guard<std::mutex> lock(stderr_mutex());
        std::cerr.write(json_line.data(), static_cast<std::streamsize>(json_line.size()));
        std::cerr.put('\n');
        std::cerr.flush();
    }
};

bool stdin_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
}

bool stdout_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

// ANSI colors need a terminal that interprets escape sequences. Windows
// consoles (conhost) do so only with ENABLE_VIRTUAL_TERMINAL_PROCESSING,
// which is switched on here; if that fails, colors stay off.
bool stdout_supports_ansi() {
#if defined(_WIN32)
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out == nullptr || out == INVALID_HANDLE_VALUE) return false;
    DWORD mode = 0;
    if (GetConsoleMode(out, &mode) == 0) return false;
    if ((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0) return true;
    return SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    return true;
#endif
}

bool is_mock_backend(std::string_view name) { return name == si::kMockBackendName; }

// ---------------------------------------------------------------------------
// Ctrl-C: the signal handler only sets a lock-free atomic flag
// (async-signal-safe, and no data race with the watcher thread); a watcher
// thread turns it into a cancel callback (Session::cancel()). The flag stays
// latched for the watcher's lifetime (one request or chat turn) and the
// callback is repeated on every poll: Session::cancel() is a no-op until the
// request is active, so a Ctrl-C that arrives just before the request starts
// still cancels it. Repeated cancels of the same request are harmless.
// ---------------------------------------------------------------------------
std::atomic<int> g_interrupted{0};
static_assert(std::atomic<int>::is_always_lock_free, "the Ctrl-C flag must be lock-free");

extern "C" void on_sigint(int) { g_interrupted.store(1); }

class InterruptWatcher {
public:
    explicit InterruptWatcher(std::function<void()> on_interrupt) : on_interrupt_(std::move(on_interrupt)) {
        g_interrupted.store(0);
        std::signal(SIGINT, on_sigint);
        thread_ = std::thread([this] {
            while (!done_.load()) {
                if (g_interrupted.load() != 0) on_interrupt_();
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    ~InterruptWatcher() {
        done_.store(true);
        thread_.join();
        std::signal(SIGINT, SIG_DFL);
    }
    InterruptWatcher(const InterruptWatcher&) = delete;
    InterruptWatcher& operator=(const InterruptWatcher&) = delete;

private:
    std::function<void()> on_interrupt_;
    std::atomic<bool> done_{false};
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Errors and exit codes (docs/CLI.md "Exit codes").
// ---------------------------------------------------------------------------
int usage_error(std::string_view command, const std::string& message) {
    err() << "error: " << command << ": " << message << "\n(see 'sonder-infer " << command << " --help')\n";
    return cli::kExitUsage;
}

// Prints a runtime error; an unreachable Ollama also gets a hint.
void report_error(std::string_view command, const si::Status& status, const std::string& backend) {
    err() << "error: " << command << ": " << status.to_string() << "\n";
    if (backend == "ollama" && status.code() == si::ErrorCode::unavailable) {
        err() << "hint: Ollama is not reachable; start it with 'ollama serve', or pass --ollama-url URL "
                 "(or set SONDER_OLLAMA_URL / OLLAMA_HOST)\n";
    }
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------
constexpr std::string_view kKnownBackends[] = {"mock", "ollama", "llamacpp", "llamaserver"};

std::vector<std::string> compiled_backends() {
#if defined(SONDER_HAS_SERVER)
    return si::available_backend_names();
#else
    std::vector<std::string> names{si::kMockBackendName};
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    names.emplace_back("ollama");
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    names.emplace_back("llamacpp");
#endif
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    names.emplace_back("llamaserver");
#endif
    return names;
#endif
}

struct BackendChoice {
    std::string backend;  // empty when the command does not need one
    std::string model;
    std::string ollama_url;
    bool ollama_allow_remote = false;
    std::chrono::microseconds mock_delay{0};
    std::string llamaserver_config;
    std::string llamaserver_url;
    std::string llamaserver_executable;
    std::string llamaserver_mode;
    std::vector<std::string> llamaserver_args;
};

// Checks a backend name: unknown names and backends missing from this build
// are usage errors.
bool check_backend_name(std::string_view command, const std::string& name, int& rc) {
    const auto compiled = compiled_backends();
    for (const auto& n : compiled) {
        if (n == name) return true;
    }
    for (const auto known : kKnownBackends) {
        if (known == name) {
            rc = usage_error(command, "this build does not include the " + name + " backend" +
                                          (name == "llamacpp" ? " (configure with SONDER_WITH_LLAMA_CPP=ON)"
                                                              : " (module src/backends/" + name + ")"));
            return false;
        }
    }
    std::vector<std::string_view> candidates(compiled.begin(), compiled.end());
    std::string message = "unknown backend '" + name + "'";
    if (auto s = cli::suggest(name, candidates)) message += " (did you mean '" + *s + "'?)";
    std::string list;
    for (const auto& n : compiled) list += (list.empty() ? "" : ", ") + n;
    rc = usage_error(command, message + "; this build has: " + list);
    return false;
}

// --ollama-allow-remote never permits plain HTTP: prompts and replies would
// cross the network unencrypted. A remote Ollama host must use https://.
bool check_ollama_remote_url(std::string_view command, const std::string& url, int& rc) {
    if (!cli::is_plain_http_remote(url)) return true;
    rc = usage_error(command, "--ollama-allow-remote refuses plain http:// to a non-loopback host ('" + url +
                                  "'); use https:// (needs a SONDER_WITH_TLS=ON build)");
    return false;
}

// Resolves --backend/--model/--ollama-url from flags, then the environment.
bool resolve_backend(std::string_view command, const Args& a, const cli::EnvDefaults& env, bool need_backend,
                     bool need_model, BackendChoice& out, int& rc) {
    out.backend = a.get("backend").value_or(env.backend.value_or(""));
    out.model = a.get("model").value_or(env.model.value_or(""));
    out.ollama_url = a.get("ollama-url").value_or(env.ollama_url.value_or(""));
    out.ollama_allow_remote = a.has("ollama-allow-remote");
    out.llamaserver_config = a.get("llamaserver-config").value_or("");
    out.llamaserver_url = a.get("llamaserver-url").value_or("");
    out.llamaserver_executable = a.get("llamaserver-executable").value_or("");
    out.llamaserver_mode = a.get("llamaserver-mode").value_or("");
    out.llamaserver_args = a.all("llamaserver-arg");
    if (auto v = a.get("mock-delay-ms")) {
        int ms = 0;
        std::string error;
        if (!cli::parse_integer<int>("mock-delay-ms", *v, 0, 60000, ms, error)) {
            rc = usage_error(command, error);
            return false;
        }
        out.mock_delay = std::chrono::milliseconds(ms);
    }
    if (need_backend) {
        if (out.backend.empty()) {
            rc = usage_error(command, "--backend is required (mock, ollama or llamacpp; or set SONDER_INFER_BACKEND)");
            return false;
        }
        if (!check_backend_name(command, out.backend, rc)) return false;
    }
    if (need_model && out.model.empty()) {
        if (is_mock_backend(out.backend)) {
            out.model = "mock";
        } else {
            rc = usage_error(command, "--model is required (or set SONDER_INFER_MODEL); list models with "
                                      "'sonder-infer models --backend " +
                                          out.backend + "'");
            return false;
        }
    }
    if (out.ollama_allow_remote) {
        if (!check_ollama_remote_url(command, out.ollama_url, rc)) return false;
        // Security-relevant: printed even with --quiet.
        err() << "warning: --ollama-allow-remote: prompts may go to a non-loopback Ollama host (https:// only, "
                 "which needs a SONDER_WITH_TLS=ON build)\n";
    }
    return true;
}

void register_backends(si::Engine& engine, const BackendChoice& choice) {
#if defined(SONDER_HAS_SERVER)
    // Shared construction with `sonder-infer serve` (backend_setup.hpp).
    for (const auto& name : si::available_backend_names()) {
        si::BackendSetup setup;
        setup.backend = name;
        setup.ollama_url = choice.ollama_url;
        setup.ollama_allow_remote = choice.ollama_allow_remote;
        setup.mock_token_delay = choice.mock_delay;
        setup.llamaserver_config = choice.llamaserver_config;
        setup.llamaserver_url = choice.llamaserver_url;
        setup.llamaserver_executable = choice.llamaserver_executable;
        setup.llamaserver_mode = choice.llamaserver_mode;
        setup.llamaserver_args = choice.llamaserver_args;
        if (name == "llamaserver" && !setup.llamaserver_config.empty()) {
            if (auto st = si::load_llamaserver_config(setup.llamaserver_config, setup); !st.ok()) {
                err() << "warning: backend " << name << " config unavailable: " << st.to_string() << "\n";
                continue;
            }
            // Explicit CLI values override the file.
            if (!choice.llamaserver_url.empty()) setup.llamaserver_url = choice.llamaserver_url;
            if (!choice.llamaserver_executable.empty()) setup.llamaserver_executable = choice.llamaserver_executable;
            if (!choice.llamaserver_mode.empty()) setup.llamaserver_mode = choice.llamaserver_mode;
            if (!choice.llamaserver_args.empty()) setup.llamaserver_args = choice.llamaserver_args;
        }
        auto backend = si::make_backend(setup);
        if (backend.ok()) {
            (void)engine.register_backend(std::move(backend.value()));
        } else {
            err() << "warning: backend " << name << " unavailable: " << backend.status().to_string() << "\n";
        }
    }
#else
    si::MockBackendOptions mo;
    mo.token_delay = choice.mock_delay;
    (void)engine.register_backend(si::make_mock_backend(mo));
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    si::OllamaBackendOptions oo;
    if (!choice.ollama_url.empty()) oo.base_url = choice.ollama_url;
    oo.allow_remote = choice.ollama_allow_remote;
    (void)engine.register_backend(si::make_ollama_backend(oo));
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    // Direct llama.cpp backend: --model takes a GGUF path.
    (void)engine.register_backend(si::make_llamacpp_backend());
#endif
#endif
}

// Engine with the telemetry flags applied (commands that accept them).
bool build_engine(std::string_view command, const Args& a, bool with_telemetry, const BackendChoice& choice,
                  std::unique_ptr<si::Engine>& engine, int& rc) {
    si::EngineOptions eo;
    if (with_telemetry) {
        eo.telemetry.capture_text = a.has("capture-text");
        const std::string level_text = a.get("telemetry-level").value_or("standard");
        auto level = si::parse_telemetry_level(level_text);
        if (!level) {
            rc = usage_error(command, "invalid value '" + level_text +
                                          "' for --telemetry-level (expected off, metrics, standard or deep)");
            return false;
        }
        eo.telemetry.level = *level;
        if (auto path = a.get("telemetry")) {
            if (*path == "-") {
                eo.telemetry_sinks.push_back(std::make_shared<StderrTelemetrySink>());
            } else {
                si::Status st;
                auto sink = si::make_jsonl_file_sink(*path, false, &st);
                if (!sink) {
                    err() << "error: " << command << ": cannot open --telemetry " << *path << ": " << st.to_string()
                          << "\n";
                    rc = cli::kExitFailure;
                    return false;
                }
                eo.telemetry_sinks.push_back(std::shared_ptr<si::TelemetrySink>(std::move(sink)));
            }
        }
    }
    engine = std::make_unique<si::Engine>(std::move(eo));
    register_backends(*engine, choice);
    return true;
}

// ---------------------------------------------------------------------------
// Option values
// ---------------------------------------------------------------------------
bool sampling_from_args(const Args& a, si::SamplingConfig& s, std::string& error, bool upstream_defaults = false) {
    s = upstream_defaults ? si::SamplingConfig{} : si::SamplingConfig::greedy(128, 42);
    s.explicit_only = upstream_defaults;
    const auto mark = [&s](const char* flag) {
        static const std::pair<const char*, si::SamplingConfig::Field> kFlags[] = {
            {"max-tokens", si::SamplingConfig::kMaxTokens},
            {"temperature", si::SamplingConfig::kTemperature},
            {"top-p", si::SamplingConfig::kTopP},
            {"top-k", si::SamplingConfig::kTopK},
            {"min-p", si::SamplingConfig::kMinP},
            {"repeat-penalty", si::SamplingConfig::kRepeatPenalty},
            {"typical-p", si::SamplingConfig::kTypicalP},
            {"repeat-last-n", si::SamplingConfig::kRepeatLastN},
            {"presence-penalty", si::SamplingConfig::kPresencePenalty},
            {"frequency-penalty", si::SamplingConfig::kFrequencyPenalty},
            {"num-ctx", si::SamplingConfig::kNumCtx},
            {"seed", si::SamplingConfig::kSeed},
            {"logit-bias", si::SamplingConfig::kLogitBias},
            {"stop", si::SamplingConfig::kStop},
        };
        for (const auto& [name, field] : kFlags) {
            if (std::string_view(name) == flag) s.explicit_fields |= field;
        }
    };
    constexpr int kIntMax = std::numeric_limits<int>::max();
    const auto int_opt = [&](const char* flag, int min, auto& out) {
        if (auto v = a.get(flag)) {
            int value = 0;
            if (!cli::parse_integer<int>(flag, *v, min, kIntMax, value, error)) return false;
            out = static_cast<std::remove_reference_t<decltype(out)>>(value);
            mark(flag);
        }
        return true;
    };
    const auto float_opt = [&](const char* flag, float& out) {
        if (auto v = a.get(flag)) {
            if (!cli::parse_float(flag, *v, out, error)) return false;
            mark(flag);
        }
        return true;
    };
    if (!int_opt("max-tokens", 1, s.max_tokens) || !float_opt("temperature", s.temperature) ||
        !float_opt("top-p", s.top_p) || !int_opt("top-k", 0, s.top_k) || !float_opt("min-p", s.min_p) ||
        !float_opt("repeat-penalty", s.repeat_penalty) || !float_opt("typical-p", s.typical_p) ||
        !int_opt("repeat-last-n", -1, s.repeat_last_n) || !float_opt("presence-penalty", s.presence_penalty) ||
        !float_opt("frequency-penalty", s.frequency_penalty) || !int_opt("num-ctx", 0, s.num_ctx)) {
        return false;
    }
    if (auto v = a.get("seed")) {
        std::uint64_t seed = 0;
        if (!cli::parse_u64("seed", *v, seed, error)) return false;
        s.seed = seed;
        mark("seed");
    }
    if (auto v = a.get("logit-bias")) {
        if (!cli::parse_logit_bias(*v, s.logit_bias, error)) return false;
        mark("logit-bias");
    }
    s.stop = a.stops;
    if (!s.stop.empty()) mark("stop");
    if (auto st = si::validate(s); !st.ok()) {
        error = st.message();
        return false;
    }
    return true;
}

struct SessionMeta {
    std::optional<std::string> run_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    si::WorkloadClass workload = si::WorkloadClass::implementation_worker;
    int priority = 0;
};

bool session_meta_from_args(const Args& a, SessionMeta& m, std::string& error) {
    const auto id_opt = [&](const char* flag, std::optional<std::string>& out) {
        if (auto v = a.get(flag)) {
            std::string id;
            if (!cli::parse_correlation_id(flag, *v, id, error)) return false;
            out = std::move(id);
        }
        return true;
    };
    if (!id_opt("run-id", m.run_id) || !id_opt("agent-id", m.agent_id) || !id_opt("task-id", m.task_id)) {
        return false;
    }
    if (auto v = a.get("workload")) {
        auto w = cli::parse_workload(*v);
        if (!w) {
            error = "invalid value '" + *v + "' for --workload (expected one of " + cli::workload_names() + ")";
            return false;
        }
        m.workload = *w;
    }
    if (auto v = a.get("priority")) {
        if (!cli::parse_integer<int>("priority", *v, cli::kMinPriority, cli::kMaxPriority, m.priority, error)) {
            return false;
        }
    }
    return true;
}

si::SessionOptions session_options(const si::SamplingConfig& sampling, const SessionMeta& m) {
    si::SessionOptions so;
    so.sampling = sampling;
    so.run_id = m.run_id;
    so.agent_id = m.agent_id;
    so.task_id = m.task_id;
    so.workload = m.workload;
    so.priority = m.priority;
    return so;
}

// --stats: `fallback` by default, none with --quiet unless --stats is given.
bool stats_mode_from_args(const Args& a, cli::StatsMode fallback, cli::StatsMode& out, std::string& error) {
    out = a.has("quiet") ? cli::StatsMode::none : fallback;
    if (auto v = a.get("stats")) {
        auto m = cli::parse_stats_mode(*v);
        if (!m) {
            error = "invalid value '" + *v + "' for --stats (expected text, json or none)";
            return false;
        }
        out = *m;
    }
    return true;
}

void print_stats(cli::StatsMode mode, const cli::RequestStats& stats) {
    if (mode == cli::StatsMode::text) {
        err() << cli::format_stats_text(stats) << "\n";
    } else if (mode == cli::StatsMode::json) {
        err() << cli::format_stats_json(stats) << "\n";
    }
}

std::shared_ptr<si::Model> load_model(std::string_view command, si::Engine& engine, const BackendChoice& choice,
                                      int& rc) {
    si::ModelLoadOptions lo;
    lo.model = choice.model;
    auto loaded = engine.load_model(choice.backend, lo);
    if (!loaded.ok()) {
        report_error(command, loaded.status(), choice.backend);
        rc = cli::exit_code_for(loaded.status());
        return nullptr;
    }
    return loaded.value();
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
int cmd_version(const Args& a) {
    if (a.has("json")) {
        si::json::Array backends;
        for (const auto& n : compiled_backends()) backends.emplace_back(n);
#if defined(SONDER_HAS_SERVER)
        const si::json::Value api_version(si::server::kApiVersion);
        const bool server = true;
#else
        const si::json::Value api_version(nullptr);
        const bool server = false;
#endif
#if defined(SONDER_HAS_BENCH)
        const bool bench = true;
#else
        const bool bench = false;
#endif
        const si::json::Object doc{{"name", "sonder-infer"},
                                   {"version", si::version_string()},
                                   {"commit", si::build_commit()},
                                   {"abi_version", SONDER_ABI_VERSION},
                                   {"api_version", api_version},
                                   {"platform", si::host_platform()},
                                   {"backends", std::move(backends)},
                                   {"modules", si::json::Object{{"server", server}, {"bench", bench}}}};
        std::cout << si::json::Value(doc).dump() << "\n";
        return cli::kExitOk;
    }
    std::cout << "sonder-infer " << si::version_string() << " (" << si::build_commit() << ") C ABI "
              << SONDER_ABI_VERSION;
#if defined(SONDER_HAS_SERVER)
    std::cout << " HTTP API " << si::server::kApiVersion;
#endif
    std::cout << "\n";
    return cli::kExitOk;
}

int cmd_devices(const Args& a) {
    const auto devices = si::enumerate_devices();
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& d : devices) {
            list.emplace_back(si::json::Object{{"id", d.id},
                                               {"kind", si::to_string(d.kind)},
                                               {"name", d.name},
                                               {"logical_cores", d.logical_cores},
                                               {"total_memory_bytes", d.total_memory_bytes},
                                               {"available_memory_bytes", d.available_memory_bytes}});
        }
        std::cout << si::json::Value(si::json::Object{{"platform", si::host_platform()}, {"devices", std::move(list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& d : devices) {
        std::cout << d.id << "  kind=" << si::to_string(d.kind) << "  name=\"" << d.name << "\""
                  << "  logical_cores=" << d.logical_cores << "  total_memory_mib=" << (d.total_memory_bytes >> 20)
                  << "  available_memory_mib=" << (d.available_memory_bytes >> 20) << "\n";
    }
    std::cout << "platform=" << si::host_platform() << "\n";
    return cli::kExitOk;
}

int cmd_backends(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("backends", a, env, false, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("backends", a, false, choice, engine, rc)) return rc;
    const bool json = a.has("json");
    si::json::Array list;
    std::size_t available = 0;
    bool ollama_unreachable = false;
    for (const auto& name : engine->backend_names()) {
        auto b = engine->find_backend(name);
        auto probe = b->probe();
        if (probe.ok()) ++available;
        if (!probe.ok() && name == "ollama" && probe.status().code() == si::ErrorCode::unavailable) {
            ollama_unreachable = true;
        }
        if (json) {
            si::json::Array caps;
            for (const auto& c : b->capabilities().names()) caps.emplace_back(c);
            list.emplace_back(si::json::Object{
                {"name", name},
                {"available", probe.ok()},
                {"version", probe.ok() ? si::json::Value(probe.value()) : si::json::Value(nullptr)},
                {"error", probe.ok() ? si::json::Value(nullptr) : si::json::Value(probe.status().to_string())},
                {"description", b->description()},
                {"synthetic", is_mock_backend(name)},
                {"capabilities", std::move(caps)}});
            continue;
        }
        std::cout << name << "  "
                  << (probe.ok() ? "reachable version=" + probe.value()
                                 : "unavailable (" + probe.status().to_string() + ")")
                  << "\n    " << b->description() << "\n    capabilities:";
        for (const auto& c : b->capabilities().names()) std::cout << " " << c;
        std::cout << "\n";
    }
    if (json) {
        std::cout << si::json::Value(si::json::Object{{"backends", std::move(list)}, {"available", available}}).dump()
                  << "\n";
    } else if (ollama_unreachable) {
        std::cout << "hint: Ollama is not reachable; start it with 'ollama serve', or pass --ollama-url URL\n";
    }
    if (available == 0) {
        err() << "error: backends: no backend is available\n";
        return cli::kExitFailure;
    }
    return cli::kExitOk;
}

int cmd_models(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("models", a, env, true, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("models", a, false, choice, engine, rc)) return rc;
    auto models = engine->find_backend(choice.backend)->list_models();
    if (!models.ok()) {
        report_error("models", models.status(), choice.backend);
        return cli::exit_code_for(models.status());
    }
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& m : models.value()) {
            list.emplace_back(si::json::Object{{"name", m.name},
                                               {"format", m.format},
                                               {"family", m.family},
                                               {"parameter_size", m.parameter_size},
                                               {"quantization", m.quantization},
                                               {"size_bytes", m.size_bytes},
                                               {"context_length", m.context_length}});
        }
        std::cout << si::json::Value(si::json::Object{{"backend", choice.backend},
                                                      {"synthetic", is_mock_backend(choice.backend)},
                                                      {"models", std::move(list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& m : models.value()) {
        std::cout << m.name << "  format=" << m.format << "  family=" << m.family << "  params=" << m.parameter_size
                  << "  quant=" << m.quantization << "  size_mib=" << (m.size_bytes >> 20) << "\n";
    }
    return cli::kExitOk;
}

// Shared setup for generate, chat and bench: backend, sampling, session
// metadata, input files (`read_inputs`, usage errors), engine and model.
struct RequestSetup {
    BackendChoice choice;
    si::SamplingConfig sampling;
    SessionMeta meta;
    std::unique_ptr<si::Engine> engine;
    std::shared_ptr<si::Model> model;
};

bool prepare_request(std::string_view command, const Args& a, const cli::EnvDefaults& env, bool quiet,
                     RequestSetup& setup, const std::function<bool(int&)>& read_inputs, int& rc) {
    if (!resolve_backend(command, a, env, true, true, setup.choice, rc)) return false;
    std::string error;
    if (!sampling_from_args(a, setup.sampling, error, setup.choice.backend == "llamaserver") ||
        !session_meta_from_args(a, setup.meta, error)) {
        rc = usage_error(command, error);
        return false;
    }
    if (read_inputs && !read_inputs(rc)) return false;
    if (!build_engine(command, a, true, setup.choice, setup.engine, rc)) return false;
    if (!quiet && is_mock_backend(setup.choice.backend)) {
        err() << "[sonder-infer] MOCK BACKEND - synthetic output, not a quality or performance signal\n";
    }
    setup.model = load_model(command, *setup.engine, setup.choice, rc);
    return setup.model != nullptr;
}

int cmd_generate(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    if (!stats_mode_from_args(a, cli::StatsMode::text, stats_mode, error)) return usage_error("generate", error);
    const auto prompt = a.get("prompt");
    if (!prompt) return usage_error("generate", "--prompt is required");
    RequestSetup setup;
    if (!prepare_request("generate", a, env, quiet, setup, {}, rc)) return rc;
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("generate", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
    {
        si::Session& s = *session.value();
        InterruptWatcher watcher([&s] { s.cancel(); });
        res = s.generate(*prompt, [](const si::TokenChunk& c) {
            std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
            std::cout.flush();
            return true;
        });
    }
    std::cout << "\n";
    std::cout.flush();
    if (!res.ok()) {
        report_error("generate", res.status(), setup.choice.backend);
        return cli::exit_code_for(res.status());
    }
    print_stats(stats_mode, cli::make_request_stats("generate", setup.choice.backend, setup.model->descriptor().name,
                                                    is_mock_backend(setup.choice.backend), res.value()));
    return cli::exit_code_for(res.value().outcome);
}

int cmd_chat(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    const auto messages_path = a.get("messages");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    // One-shot chat prints stats by default; the interactive REPL only on
    // request (/stats, or --stats text|json).
    if (!stats_mode_from_args(a, messages_path ? cli::StatsMode::text : cli::StatsMode::none, stats_mode, error)) {
        return usage_error("chat", error);
    }
    std::vector<si::ChatMessage> file_messages;
    const auto read_messages = [&](int& code) {
        if (!messages_path) return true;
        auto loaded = cli::load_chat_messages(*messages_path);
        if (!loaded.ok()) {
            code = usage_error("chat", loaded.status().message());
            return false;
        }
        file_messages = std::move(loaded.value());
        return true;
    };
    RequestSetup setup;
    if (!prepare_request("chat", a, env, quiet, setup, read_messages, rc)) return rc;
    const std::string system = a.get("system").value_or("");
    const bool native = setup.model->backend_model().has_native_chat();
    const std::string model_name = setup.model->descriptor().name;
    const bool synthetic = is_mock_backend(setup.choice.backend);

#if defined(SONDER_HAS_SESSION_CHAT)
    // Every turn runs through the engine session (scheduler, KV accounting,
    // request telemetry with request.queued.kind = "chat").
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("chat", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Session& s = *session.value();
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&s] { s.cancel(); });
            res = s.chat(history, [](const si::TokenChunk& c) {
                std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
                std::cout.flush();
                return true;
            });
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().outcome == si::RequestOutcome::cancelled;
        report.stats = cli::make_request_stats("chat", setup.choice.backend, model_name, synthetic, res.value());
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#else
    // Library without Session::chat: turns go straight to the backend model
    // and emit no request telemetry.
    std::uint64_t turn_no = 0;
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::CancellationSource source;
        si::Result<cli::ChatTurnResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&source] { source.cancel(); });
            res = cli::run_chat_turn(setup.model->backend_model(), history, setup.sampling, source.token(), std::cout,
                                     "chat-" + std::to_string(++turn_no));
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().stats.stop_reason == si::StopReason::cancelled;
        report.stats.command = "chat";
        report.stats.backend = setup.choice.backend;
        report.stats.model = model_name;
        report.stats.synthetic = synthetic;
        report.stats.outcome = report.cancelled ? "cancelled" : "completed";
        report.stats.stop_reason = si::to_string(res.value().stats.stop_reason);
        report.stats.prompt_tokens = res.value().stats.prompt_tokens;
        report.stats.completion_tokens = res.value().stats.completion_tokens;
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#endif

    if (messages_path) {
        cli::apply_system_prompt(file_messages, system);
        auto res = turn(file_messages);
        std::cout << "\n";
        std::cout.flush();
        if (!res.ok()) {
            report_error("chat", res.status(), setup.choice.backend);
            return cli::exit_code_for(res.status());
        }
        print_stats(stats_mode, res.value().stats);
        return res.value().cancelled ? cli::kExitCancelled : cli::kExitOk;
    }

    const bool interactive = stdin_is_terminal();
    const bool terminal_out = stdout_is_terminal();
    if (!quiet) {
        err() << "[sonder-infer] chat with " << model_name << " on " << setup.choice.backend
              << (native ? " (native chat)" : " (generic prompt format)")
              << "; /help lists commands, /exit or EOF quits\n";
    }
    cli::ChatReplOptions ro;
    ro.system = system;
    ro.user_prompt = interactive ? "you> " : "";
    ro.assistant_label = interactive || terminal_out ? "assistant> " : "";
    ro.color = cli::color_enabled(terminal_out) && stdout_supports_ansi();
    ro.stats = stats_mode;
    return cli::run_chat_session(std::cin, std::cout, err(), ro, turn);
}

#if defined(SONDER_HAS_BENCH)
int cmd_bench(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    int rc = cli::kExitOk;
    const auto corpus_path = a.get("corpus");
    const auto out_path = a.get("out");
    if (!corpus_path || !out_path) return usage_error("bench", "--corpus and --out are required");
    si::bench::Options opts;
    std::string error;
    if (auto v = a.get("warmup"); v && !cli::parse_integer<int>("warmup", *v, 0, 100, opts.warmup_runs, error)) {
        return usage_error("bench", error);
    }
    if (auto v = a.get("runs"); v && !cli::parse_integer<int>("runs", *v, 1, 1000, opts.measured_runs, error)) {
        return usage_error("bench", error);
    }
    opts.label = a.get("label").value_or("");
    std::optional<si::bench::Corpus> corpus;
    const auto read_corpus = [&](int& code) {
        auto loaded = si::bench::load_corpus(*corpus_path);
        if (!loaded.ok()) {
            code = usage_error("bench", loaded.status().to_string());
            return false;
        }
        corpus = std::move(loaded.value());
        return true;
    };
    RequestSetup setup;
    if (!prepare_request("bench", a, env, quiet, setup, read_corpus, rc)) return rc;
    opts.sampling = setup.sampling;
    opts.run_id = setup.meta.run_id;
    opts.agent_id = setup.meta.agent_id;
    opts.task_id = setup.meta.task_id;
    opts.workload = setup.meta.workload;
    opts.priority = setup.meta.priority;
    auto doc = si::bench::run(*setup.engine, setup.model, *corpus, opts, quiet ? nullptr : &err());
    std::ofstream out(*out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err() << "error: bench: cannot write " << *out_path << "\n";
        return cli::kExitFailure;
    }
    out << si::json::Value(doc).dump() << "\n";
    out.close();
    if (a.has("markdown")) {
        const std::string markdown = si::bench::render_markdown(si::json::Value(doc));
        std::string md_path = *out_path;
        if (md_path.size() > 5 && md_path.compare(md_path.size() - 5, 5, ".json") == 0) {
            md_path.resize(md_path.size() - 5);
        }
        md_path += ".md";
        std::ofstream md(md_path, std::ios::binary | std::ios::trunc);
        if (!md) {
            err() << "error: bench: cannot write " << md_path << "\n";
            return cli::kExitFailure;
        }
        md << markdown;
        std::cout << markdown;
        if (!quiet) err() << "[bench] wrote " << md_path << "\n";
    }
    const auto* summary = doc.find("summary");
    if (!quiet) {
        err() << "[bench] wrote " << *out_path << " summary=" << (summary ? summary->dump() : "{}") << "\n";
    }
    const auto* failures = summary ? summary->find("failures") : nullptr;
    return failures && failures->as_int() > 0 ? cli::kExitFailure : cli::kExitOk;
}
#endif

bool wants_help(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

#if defined(SONDER_HAS_SERVER)
// The --ollama-allow-remote plain-HTTP refusal for serve, checked before
// serve_main() runs (src/server is outside this lane). Serve's parser takes
// "--ollama-url URL" and "--ollama-url=URL"; the environment fills in the
// URL as it does for serve (backend_env_defaults()).
bool serve_ollama_remote_ok(const std::vector<std::string>& args, int& rc) {
    bool allow_remote = false;
    std::optional<std::string> url;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--ollama-allow-remote") {
            allow_remote = true;
        } else if (arg == "--ollama-url" && i + 1 < args.size()) {
            url = args[++i];
        } else if (arg.rfind("--ollama-url=", 0) == 0) {
            url = arg.substr(std::string_view("--ollama-url=").size());
        }
    }
    if (!allow_remote) return true;
    if (!url) url = cli::read_env_defaults().ollama_url;
    return check_ollama_remote_url("serve", url.value_or(""), rc);
}
#endif

int cmd_serve(const std::vector<std::string>& args) {
#if defined(SONDER_HAS_SERVER)
    // --help wins over any other argument, as for the other commands.
    if (wants_help(args)) return si::server::serve_main({"--help"}, std::cout, std::cerr);
    int rc = cli::kExitOk;
    if (!serve_ollama_remote_ok(args, rc)) return rc;
    return si::server::serve_main(args, std::cout, std::cerr);
#else
    (void)args;
    err() << "error: serve: this build does not include the server module (src/server); "
             "build from a tree that contains it to use 'sonder-infer serve'\n";
    return cli::kExitUsage;
#endif
}

int cmd_help(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << cli::overview();
        return cli::kExitOk;
    }
    if (args.size() == 1 && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << cli::render_help(cli::kProgram, cli::help_command());
        return cli::kExitOk;
    }
    if (args.size() > 1) return usage_error("help", "expected at most one command name");
    const std::string& name = args.front();
    if (name == "serve") return cmd_serve({"--help"});
    if (const auto* spec = cli::find_command(name)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    std::string message = "unknown command '" + name + "'";
    if (auto s = cli::suggest(name, cli::command_names())) message += " (did you mean '" + *s + "'?)";
    err() << "error: help: " << message << "\n\n" << cli::short_usage();
    return cli::kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    if (args.empty()) {
        err() << cli::short_usage();
        return cli::kExitUsage;
    }
    const std::string cmd = args.front();
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (cmd == "help" || cmd == "--help" || cmd == "-h") return cmd_help(rest);
    if (cmd == "--version") return cmd_version(Args{});
    const cli::CommandSpec* spec = cli::find_command(cmd);
    if (spec == nullptr) {
        std::string message = "unknown command '" + cmd + "'";
        if (auto s = cli::suggest(cmd, cli::command_names())) message += " (did you mean '" + *s + "'?)";
        err() << "error: " << message << "\n\n" << cli::short_usage();
        return cli::kExitUsage;
    }
    if (cmd == "serve") return cmd_serve(rest);
    if (cmd == "tune") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        return si::llamaserver::tune::tune_main(rest, std::cout, std::cerr);
#else
        err() << "error: tune: this build does not include the llamaserver module\n";
        return cli::kExitUsage;
#endif
    }

    std::string error;
    auto parsed = cli::parse_command_args(*spec, rest, error);
    if (parsed ? parsed->help : wants_help(rest)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    if (!parsed) return usage_error(cmd, error);
    const Args& a = *parsed;
    const cli::EnvDefaults env = cli::read_env_defaults();

    if (cmd == "version") return cmd_version(a);
    if (cmd == "devices") return cmd_devices(a);
    if (cmd == "backends") return cmd_backends(a, env);
    if (cmd == "models") return cmd_models(a, env);
    if (cmd == "generate") return cmd_generate(a, env);
    if (cmd == "chat") return cmd_chat(a, env);
#if defined(SONDER_HAS_BENCH)
    if (cmd == "bench") return cmd_bench(a, env);
#else
    if (cmd == "bench") {
        err() << "error: bench: this build does not include the bench module (bench/)\n";
        return cli::kExitUsage;
    }
#endif
    err() << "error: unknown command '" << cmd << "'\n\n" << cli::short_usage();
    return cli::kExitUsage;
}
