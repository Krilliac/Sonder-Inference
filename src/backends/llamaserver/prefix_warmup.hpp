// Bounded, best-effort prefix replay. No model execution on caller threads.
#pragma once

#include <memory>
#include <mutex>
#include <thread>

#include "net/http_client.hpp"
#include "sonder/inference/backends/llamaserver.hpp"

namespace sonder::inference::llamaserver {

class PrefixWarmup {
  public:
    explicit PrefixWarmup(const LlamaServerBackendOptions &options);
    ~PrefixWarmup();
    PrefixWarmup(const PrefixWarmup &) = delete;
    PrefixWarmup &operator=(const PrefixWarmup &) = delete;
    // Called once per accepted child, after readiness. Attach mode probes
    // /health itself. Neither method is called by inference requests.
    void start(const std::string &base_url, bool wait_for_health = false) noexcept;
    void stop() noexcept;
    [[nodiscard]] std::optional<BackendWarmupStatus> status() const;
    [[nodiscard]] std::vector<BackendWarning> warnings() const;
    [[nodiscard]] std::vector<std::uint32_t> warmed_slots() const;

  private:
    void stop_locked();
    void run(std::string base_url, bool wait_for_health, CancellationToken cancel) noexcept;
    Status replay(const std::string &base_url, bool wait_for_health, const CancellationToken &cancel);
    void fail(const std::string &message);

    LlamaServerWarmupOptions options_;
    bool native_;
    net::HttpRequest request_;
    std::chrono::milliseconds startup_timeout_;
    std::chrono::milliseconds poll_interval_;
    std::mutex control_mutex_;
    std::unique_ptr<CancellationSource> cancel_;
    std::thread worker_;
    mutable std::mutex mutex_;
    std::optional<BackendWarmupStatus> status_;
    std::vector<BackendWarning> warnings_;
    std::vector<std::uint32_t> warmed_;
};

} // namespace sonder::inference::llamaserver
