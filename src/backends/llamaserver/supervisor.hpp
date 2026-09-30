#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu_memory.hpp"
#include "log_diagnostics.hpp"
#include "process.hpp"
#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/cancellation.hpp"

namespace sonder::inference::llamaserver {

class PrefixWarmup;

struct SupervisorOptions {
    std::string executable;
    std::vector<std::string> arguments;
    std::chrono::milliseconds readiness_timeout{60000};
    std::chrono::milliseconds health_poll_interval{50};
    std::chrono::milliseconds restart_initial_backoff{100};
    std::chrono::milliseconds restart_max_backoff{2000};
    std::chrono::milliseconds shutdown_timeout{2000};
    std::size_t max_restarts = 3;
    // One bounded probe. The supervisor owns polling and liveness checks.
    std::function<bool(std::uint16_t, std::chrono::milliseconds)> health_check;

    // Runtime diagnostics (docs/integration/vram-spill.md). They run on the
    // monitor thread only, never on the request path.
    SpillGuardOptions spill_guard;
    // Null uses make_gpu_counter_source() (PDH on Windows, unsupported elsewhere).
    std::shared_ptr<GpuCounterSource> gpu_counters;
    // Child log to scan for performance warnings. When set and `arguments`
    // has no --log-file, "--log-file <log_file>" is appended; an explicit
    // --log-file in `arguments` is always the file that is read.
    std::string log_file;
    // Warn up front about FlashAttention with mismatched K/V cache types.
    bool kv_pairing_check = true;
    std::vector<std::pair<std::string, std::string>> environment;
    std::string output_file;
    // Optional best-effort prefix replay, started after each accepted child.
    std::shared_ptr<PrefixWarmup> warmup;
};

class Supervisor {
  public:
    explicit Supervisor(SupervisorOptions options, std::unique_ptr<ProcessLauncher> launcher = {});
    ~Supervisor();
    Supervisor(const Supervisor &) = delete;
    Supervisor &operator=(const Supervisor &) = delete;
    // Lazy start or wait through a restart; never resets the retry budget.
    Result<std::uint16_t> start(const CancellationToken &cancel = {});
    // Terminal/idempotent; interrupts probes and backoff, then joins the owner.
    void stop();
    [[nodiscard]] bool running() const;
    [[nodiscard]] std::uint16_t port() const;
    [[nodiscard]] Status failure() const;
    // Snapshot of GPU memory, context fit and warnings. Cheap; never blocks
    // on the child or on I/O.
    [[nodiscard]] BackendRuntimeStatus runtime_status() const;
    // The argv the next (or current) child is launched with, including an
    // auto_fit context reduction and an appended --log-file.
    [[nodiscard]] std::vector<std::string> launch_arguments() const;

  private:
    enum class State { idle, starting, ready, backoff, failed, stopped };
    enum class GuardAction { serve, refit, refuse };
    struct GuardVerdict {
        GuardAction action = GuardAction::serve;
        Status status;
        std::uint64_t next_ctx = 0;
    };
    GuardVerdict guard_after_ready(Process &child);
    void observe(Process &child);
    void begin_child();
    Status wait_ready(Process &child, std::uint16_t port);
    Result<std::uint16_t> settle_after_deadline(std::unique_lock<std::mutex> &lock,
                                                const CancellationToken &cancel,
                                                std::chrono::steady_clock::time_point deadline);
    bool pause(std::chrono::milliseconds duration);
    void monitor();
    void fail(Status status);

    SupervisorOptions options_;
    std::unique_ptr<ProcessLauncher> launcher_;
    mutable std::mutex mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_;
    std::thread monitor_thread_;
    CancellationSource shutdown_;
    State state_ = State::idle;
    Status failure_;
    std::uint16_t port_ = 0;

    // Diagnostics. Guarded by mutex_ unless noted.
    std::vector<std::string> arguments_;  // current launch argv
    std::size_t epoch_ = 0;               // bumped per auto_fit relaunch; extends start()'s wait
    std::vector<BackendWarning> config_warnings_;
    std::vector<BackendWarning> log_warnings_;
    GpuMemoryStatus gpu_;
    ContextFitStatus context_;
    std::shared_ptr<GpuCounterSource> gpu_source_;  // monitor thread only after construction
    std::unique_ptr<LogTail> log_tail_;             // monitor thread only
    LogDiagnostics log_diagnostics_;                // monitor thread only
};

} // namespace sonder::inference::llamaserver
