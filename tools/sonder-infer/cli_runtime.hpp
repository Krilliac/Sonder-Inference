// Internal: runtime I/O, signal handling, backend resolution/registration,
// and option-to-config helpers for the sonder-infer CLI executable.
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "sonder/inference.hpp"
#include "../../src/cli/cli_args.hpp"
#include "../../src/cli/cli_env.hpp"
#include "../../src/cli/cli_values.hpp"

namespace sonder::infer_cli {

namespace si = sonder::inference;
namespace cli = sonder::cli;

// Line-locked stderr shared by main thread diagnostics and `--telemetry -`.
std::mutex& stderr_mutex();
std::ostream& err();

class StderrTelemetrySink final : public si::TelemetrySink {
public:
    void write(std::string_view json_line) override;
};

bool stdin_is_terminal();
bool stdout_is_terminal();
bool stdout_supports_ansi();

bool is_mock_backend(std::string_view name);

// Async-signal-safe Ctrl-C watcher that repeatedly invokes `on_interrupt` once
// SIGINT arrives until destroyed.
class InterruptWatcher {
public:
    explicit InterruptWatcher(std::function<void()> on_interrupt);
    ~InterruptWatcher();
    InterruptWatcher(const InterruptWatcher&) = delete;
    InterruptWatcher& operator=(const InterruptWatcher&) = delete;

private:
    std::function<void()> on_interrupt_;
    std::atomic<bool> done_{false};
    std::thread thread_;
};

int usage_error(std::string_view command, const std::string& message);
void report_error(std::string_view command, const si::Status& status, const std::string& backend);

std::vector<std::string> compiled_backends();

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

bool check_backend_name(std::string_view command, const std::string& name, int& rc);
bool check_ollama_remote_url(std::string_view command, const std::string& url, int& rc);
bool resolve_backend(std::string_view command, const cli::Args& a, const cli::EnvDefaults& env, bool need_backend,
                     bool need_model, BackendChoice& out, int& rc);
void register_backends(si::Engine& engine, const BackendChoice& choice);
bool build_engine(std::string_view command, const cli::Args& a, bool with_telemetry, const BackendChoice& choice,
                  std::unique_ptr<si::Engine>& engine, int& rc);

bool sampling_from_args(const cli::Args& a, si::SamplingConfig& s, std::string& error, bool upstream_defaults = false);

struct SessionMeta {
    std::optional<std::string> run_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    si::WorkloadClass workload = si::WorkloadClass::implementation_worker;
    int priority = 0;
};

bool session_meta_from_args(const cli::Args& a, SessionMeta& m, std::string& error);
si::SessionOptions session_options(const si::SamplingConfig& sampling, const SessionMeta& m);

bool stats_mode_from_args(const cli::Args& a, cli::StatsMode fallback, cli::StatsMode& out, std::string& error);
void print_stats(cli::StatsMode mode, const cli::RequestStats& stats);

std::shared_ptr<si::Model> load_model(std::string_view command, si::Engine& engine, const BackendChoice& choice,
                                      int& rc);

// Shared setup for generate, chat and bench: backend, sampling, session
// metadata, input files (`read_inputs`, usage errors), engine and model.
struct RequestSetup {
    BackendChoice choice;
    si::SamplingConfig sampling;
    SessionMeta meta;
    std::unique_ptr<si::Engine> engine;
    std::shared_ptr<si::Model> model;
};

bool prepare_request(std::string_view command, const cli::Args& a, const cli::EnvDefaults& env, bool quiet,
                     RequestSetup& setup, const std::function<bool(int&)>& read_inputs, int& rc);

}  // namespace sonder::infer_cli
