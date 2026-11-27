#include "cli_runtime.hpp"

#include <csignal>
#include <cstdio>
#include <iostream>
#include <limits>
#include <streambuf>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "../../src/cli/cli_spec.hpp"
#if defined(SONDER_HAS_WORKER)
#include "../../src/worker/remote_backend.hpp"
#endif
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#else
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
#include "sonder/inference/backends/llamacpp.hpp"
#endif
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

namespace sonder::infer_cli {

namespace {

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

// Ctrl-C: the signal handler only sets a lock-free atomic flag
// (async-signal-safe, and no data race with the watcher thread); a watcher
// thread turns it into a cancel callback (Session::cancel()).
std::atomic<int> g_interrupted{0};
static_assert(std::atomic<int>::is_always_lock_free, "the Ctrl-C flag must be lock-free");

extern "C" void on_sigint(int) { g_interrupted.store(1); }

constexpr std::string_view kKnownBackends[] = {"mock", "remote-mock", "ollama", "llamacpp", "llamaserver"};

}  // namespace

std::mutex& stderr_mutex() {
    static std::mutex m;
    return m;
}

std::ostream& err() {
    static LineLockedBuf buf(std::cerr);
    static std::ostream stream(&buf);
    return stream;
}

void StderrTelemetrySink::write(std::string_view json_line) {
    std::lock_guard<std::mutex> lock(stderr_mutex());
    std::cerr.write(json_line.data(), static_cast<std::streamsize>(json_line.size()));
    std::cerr.put('\n');
    std::cerr.flush();
}

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

bool is_mock_backend(std::string_view name) { return si::is_mock_backend(name); }

InterruptWatcher::InterruptWatcher(std::function<void()> on_interrupt) : on_interrupt_(std::move(on_interrupt)) {
    g_interrupted.store(0);
    std::signal(SIGINT, on_sigint);
    thread_ = std::thread([this] {
        while (!done_.load()) {
            if (g_interrupted.load() != 0) on_interrupt_();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
}

InterruptWatcher::~InterruptWatcher() {
    done_.store(true);
    thread_.join();
    std::signal(SIGINT, SIG_DFL);
}

int usage_error(std::string_view command, const std::string& message) {
    err() << "error: " << command << ": " << message << "\n(see 'sonder-infer " << command << " --help')\n";
    return cli::kExitUsage;
}

void report_error(std::string_view command, const si::Status& status, const std::string& backend) {
    err() << "error: " << command << ": " << status.to_string() << "\n";
    if (backend == "ollama" && status.code() == si::ErrorCode::unavailable) {
        err() << "hint: Ollama is not reachable; start it with 'ollama serve', or pass --ollama-url URL "
                 "(or set SONDER_OLLAMA_URL / OLLAMA_HOST)\n";
    }
}

std::vector<std::string> compiled_backends() {
#if defined(SONDER_HAS_SERVER)
    return si::available_backend_names();
#else
    std::vector<std::string> names{si::kMockBackendName};
#if defined(SONDER_HAS_WORKER)
    names.emplace_back("remote-mock");
#endif
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

bool check_backend_name(std::string_view command, const std::string& name, int& rc) {
    const auto compiled = compiled_backends();
    for (const auto& n : compiled) {
        if (n == name) return true;
    }
    for (const auto known : kKnownBackends) {
        if (known == name) {
            rc = usage_error(command, "this build does not include the " + name + " backend" +
                                          (name == "llamacpp" ? " (configure with SONDER_WITH_LLAMA_CPP=ON)"
                                                              : (name == "remote-mock" ? " (module src/worker)"
                                                                  : " (module src/backends/" + name + ")")));
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

bool check_ollama_remote_url(std::string_view command, const std::string& url, int& rc) {
    if (!cli::is_plain_http_remote(url)) return true;
    rc = usage_error(command, "--ollama-allow-remote refuses plain http:// to a non-loopback host ('" + url +
                                  "'); use https:// (needs a SONDER_WITH_TLS=ON build)");
    return false;
}

bool resolve_backend(std::string_view command, const cli::Args& a, const cli::EnvDefaults& env, bool need_backend,
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
            rc = usage_error(command,
                             "--backend is required (mock, remote-mock, ollama, llamacpp or llamaserver; or set SONDER_INFER_BACKEND)");
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
        if (name == "remote-mock" && choice.backend != name) continue;
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
#if defined(SONDER_HAS_WORKER)
    if (choice.backend == "remote-mock") {
        si::worker::RemoteBackendOptions options;
        options.token_delay = std::chrono::duration_cast<std::chrono::milliseconds>(choice.mock_delay);
        (void)engine.register_backend(si::worker::make_remote_backend(std::move(options)));
    }
#endif
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

bool build_engine(std::string_view command, const cli::Args& a, bool with_telemetry, const BackendChoice& choice,
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

bool sampling_from_args(const cli::Args& a, si::SamplingConfig& s, std::string& error, bool upstream_defaults) {
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

bool session_meta_from_args(const cli::Args& a, SessionMeta& m, std::string& error) {
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

bool stats_mode_from_args(const cli::Args& a, cli::StatsMode fallback, cli::StatsMode& out, std::string& error) {
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

bool prepare_request(std::string_view command, const cli::Args& a, const cli::EnvDefaults& env, bool quiet,
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

}  // namespace sonder::infer_cli
