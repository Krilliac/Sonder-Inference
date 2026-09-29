#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "process.hpp"
#include "sonder/inference/cancellation.hpp"

namespace sonder::inference::llamaserver {

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

  private:
    enum class State { idle, starting, ready, backoff, failed, stopped };
    Status wait_ready(Process &child, std::uint16_t port);
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
};

} // namespace sonder::inference::llamaserver
