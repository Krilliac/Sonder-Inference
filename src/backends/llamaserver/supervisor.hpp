#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "spill_guard.hpp"
#include "child_metrics.hpp"
#include "gpu_residency.hpp"
#include "host_memory.hpp"
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
    // A child that stayed ready at least this long before exiting clears the
    // crash-restart count and backoff, so only consecutive short-lived runs
    // exhaust max_restarts.
    std::chrono::milliseconds healthy_reset_after{60000};
    // One bounded probe. The supervisor owns polling and liveness checks.
    std::function<bool(std::uint16_t, std::chrono::milliseconds)> health_check;

    // Runtime diagnostics (docs/integration/vram-spill.md). They run on the
    // monitor thread only, never on the request path.
    SpillGuardOptions spill_guard;
    // Host RAM admission before each spawn and a floor guard on available
    // physical memory while the child runs (host_memory.hpp). Defaults are
    // inert: no estimate means no pre-spawn check, floor.enabled is false.
    // The floor guard samples on the monitor thread only and never blocks
    // decode; restart_smaller_ctx reuses spill_guard's context step/min_ctx.
    HostMemoryOptions host_memory;
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
    bool native_completion = true;
    ChildMetricsOptions stall_guard;
    // Attach observes this already-validated upstream but never owns or stops it.
    // target is its base path, not /metrics. TLS policy is copied unchanged.
    std::optional<net::HttpRequest> attached_endpoint;
    // Deterministic fake-only seams. Production uses HTTP and steady_clock.
    ChildMetricsPoll metrics_poll;
    ChildMetricsClock metrics_clock;
};

class Supervisor {
  public:
    explicit Supervisor(SupervisorOptions options, std::unique_ptr<ProcessLauncher> launcher = {});
    ~Supervisor();
    Supervisor(const Supervisor &) = delete;
    Supervisor &operator=(const Supervisor &) = delete;
    // Lazy start or wait through a restart; never resets the retry budget.
    Result<std::uint16_t> start(const CancellationToken &cancel = {});
    // A lease pins the ready child for the lifetime of an upstream request.
    // Maintenance closes admission, drains these leases, then stops the child.
    class RequestLease {
      public:
        RequestLease() = default;
        ~RequestLease();
        RequestLease(RequestLease &&other) noexcept;
        RequestLease &operator=(RequestLease &&other) noexcept;
        RequestLease(const RequestLease &) = delete;
        RequestLease &operator=(const RequestLease &) = delete;
        [[nodiscard]] std::uint16_t port() const { return port_; }

      private:
        friend class Supervisor;
        RequestLease(std::shared_ptr<std::atomic<std::size_t>> activity, std::uint16_t port)
            : activity_(std::move(activity)), port_(port) {}
        std::shared_ptr<std::atomic<std::size_t>> activity_;
        std::uint16_t port_ = 0;
    };
    Result<RequestLease> acquire_request(const CancellationToken &cancel = {});
    // Terminal/idempotent; interrupts probes and backoff, then joins the owner.
    void stop();
    [[nodiscard]] bool running() const;
    [[nodiscard]] std::uint16_t port() const;
    [[nodiscard]] Status failure() const;
    // Snapshot of GPU memory, context fit and warnings. Cheap; never blocks
    // on the child or on I/O.
    [[nodiscard]] BackendRuntimeStatus runtime_status() const;
    // Additive child/diagnostics/stall fields for runtime JSON. Kept separate
    // from the backend-neutral status until the integrator wires that type.
    [[nodiscard]] json::Object child_runtime_status() const;
    using RuntimeEvent = std::pair<std::string, json::Object>;
    // A shared source lets the engine drain events without health consuming
    // them or retaining a raw Supervisor pointer beyond its lifetime.
    struct RuntimeEvents {
        void push(std::string type, json::Object attrs);
        std::vector<RuntimeEvent> drain();
      private:
        std::mutex mutex;
        std::vector<RuntimeEvent> pending;
        std::uint64_t dropped = 0;
    };
    [[nodiscard]] std::shared_ptr<RuntimeEvents> runtime_events() const { return runtime_events_; }
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
    bool health_ok(std::uint16_t port, std::chrono::milliseconds budget);
    // Monitor thread. True when a device fault was logged and a second signal
    // (child gone, or /health failing) agrees.
    bool device_fault_confirmed(Process &child, std::uint16_t port);
    void note_unconfirmed_fault();
    Result<std::uint16_t> settle_after_deadline(std::unique_lock<std::mutex> &lock,
                                                const CancellationToken &cancel,
                                                std::chrono::steady_clock::time_point deadline);
    bool pause(std::chrono::milliseconds duration);
    void monitor();
    void monitor_attached();
    bool sample_child(net::HttpRequest endpoint);
    void publish_diagnostics();
    // Monitor thread. Reads whatever the child still wrote to its log, then
    // reports whether a fatal accelerator fault was seen for this child.
    bool drain_device_fault();
    // Monitor thread. Terminal: marks the device unhealthy (diagnostics.device,
    // a backend.device_unhealthy event), so nothing is scheduled on it and the
    // child is not relaunched into the same broken device.
    void fail_device_unhealthy();
    [[nodiscard]] std::chrono::steady_clock::time_point metrics_now() const;
    void fail(Status status);
    enum class ResidencyAction { none, restart, refuse };
    ResidencyAction residency_action();
    // Pre-spawn check; monitor thread. Records override/unknown warnings.
    Status admit_host_memory();
    // Floor guard step; monitor thread. True when the child must be stopped
    // and relaunched now (arguments_ already hold the smaller context).
    bool host_memory_tick();
    [[nodiscard]] std::chrono::steady_clock::time_point host_now() const;
    [[nodiscard]] HostMemory host_memory_now() const;
    void set_host_warning(BackendWarning warning);       // replaces one with the same code
    void clear_host_warning(const std::string &code);
    void record_residency();  // mutex_ held
    void append_residency_warnings(BackendRuntimeStatus &status) const;  // mutex_ held
    Status missing_offload_failure() const;  // mutex_ held

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
    // Shared with leases so destroying a stopped supervisor before its last
    // caller unwinds cannot leave a dangling lease owner.
    std::shared_ptr<std::atomic<std::size_t>> active_requests_ = std::make_shared<std::atomic<std::size_t>>(0);
    bool maintenance_pending_ = false;
    // A residency restart owns maintenance_pending_: the host floor guard must not clear it. Guarded by mutex_.
    bool residency_maintenance_ = false;

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
    GpuResidencyGuard residency_;  // guarded by mutex_, fed only after readiness
    HostMemoryFloorGuard host_floor_;       // monitor thread only
    bool host_admission_closed_ = false;    // guarded by mutex_; mirrors host_floor_.admitting()
    bool host_restart_maintenance_ = false; // guarded by mutex_; host_memory_tick closed admission for a restart
    std::vector<BackendWarning> host_warnings_;  // guarded by mutex_
    bool reported_missing_ = false;
    bool reported_eviction_ = false;
    std::size_t eviction_restarts_ = 0;  // lifetime budget; never reset per child
    ChildMetricsSampler child_metrics_;  // monitor thread only
    std::optional<json::Object> child_status_;
    std::optional<json::Object> stall_status_;
    std::optional<json::Object> diagnostics_status_;
    std::vector<BackendWarning> child_warnings_;
    bool diagnostics_warning_sent_ = false;
    const std::shared_ptr<RuntimeEvents> runtime_events_ = std::make_shared<RuntimeEvents>();
};

} // namespace sonder::inference::llamaserver
