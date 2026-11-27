#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference/backend.hpp"

namespace sonder::inference::worker {

struct ReapCompletion;

struct RemoteBackendOptions {
    std::string executable;
    std::string engine = "mock";
    std::chrono::milliseconds token_delay{0};
    std::chrono::milliseconds restart_initial_backoff{100};
    std::chrono::milliseconds restart_max_backoff{5000};
    std::size_t max_restarts_per_hour = 5;
    std::chrono::milliseconds startup_timeout{5000};
    std::chrono::milliseconds inactivity_timeout{300000};
    std::chrono::milliseconds cancel_grace{2000};
    std::chrono::milliseconds request_timeout{1800000}; // absolute, includes admission/backpressure
    // Available when the library is compiled with SONDER_WORKER_TESTING,
    // including the normal tests+CLI presets. Not a test-target-only guard.
    // Only the separate adversary executable accepts the flag; serve never
    // sets it and production worker rejects it. Keep struct layout stable.
    std::string test_worker_mode;
};

// Value-owned, single-thread-affine policy. Only entries aging out of the
// rolling hour release budget; uptime never clears it. No sleeping or I/O.
class RestartPolicy {
  public:
    using Clock = std::chrono::steady_clock;
    explicit RestartPolicy(std::size_t max_per_hour = 5,
                           std::chrono::milliseconds initial = std::chrono::milliseconds{100},
                           std::chrono::milliseconds maximum = std::chrono::milliseconds{5000});
    bool permit_restart(Clock::time_point now);
    [[nodiscard]] bool waiting_for_reap() const;
    // Supervisor supplies this output to spawn_process even for failed exec.
    std::shared_ptr<ReapCompletion> &predecessor() { return predecessor_; }
    std::size_t recent_restarts(Clock::time_point now);
    [[nodiscard]] std::chrono::milliseconds backoff() const;

  private:
    std::size_t limit_;
    std::chrono::milliseconds initial_, maximum_;
    std::deque<Clock::time_point> restarts_;
    std::size_t attempts_ = 0;
    std::shared_ptr<ReapCompletion> predecessor_;
};

// Shared ownership of an isolated worker. Operations serialize, while the
// monitor owns process lifetime. Diagnostics are thread-safe snapshots.
// Noncopyable/nonmovable because the monitor retains its implementation.
class RemoteBackend final : public Backend {
  public:
    explicit RemoteBackend(RemoteBackendOptions options);
    ~RemoteBackend() override;
    RemoteBackend(const RemoteBackend &) = delete;
    RemoteBackend &operator=(const RemoteBackend &) = delete;
    std::string name() const override;
    std::string description() const override;
    BackendCapabilities capabilities() const override;
    static BackendCapabilities advertised_capabilities();
    Result<std::string> probe() override;
    Result<std::vector<ModelDescriptor>> list_models() override;
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions &) override;
    std::size_t max_concurrent_requests() const override { return 1; }
    [[nodiscard]] std::uint32_t worker_pid() const;
    [[nodiscard]] std::size_t restart_count() const;
    [[nodiscard]] bool circuit_open() const;

  private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

std::shared_ptr<Backend> make_remote_backend(RemoteBackendOptions options);
int worker_main(const std::vector<std::string> &arguments);

} // namespace sonder::inference::worker
