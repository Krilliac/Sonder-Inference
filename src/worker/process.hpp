#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference::worker {

// Completion survives Process destruction and is published only after the
// child's PID is reaped (or waitpid confirms ECHILD). Linux launch failures
// also return this state when fork succeeded, so they cannot bypass restart
// admission. A null state means no deferred-reap barrier is needed.
struct ReapCompletion {
    std::atomic<bool> reaped{false};
};

// Owns one child and its kill-on-parent-exit lifetime container. Methods and
// destruction belong to the supervisor thread; the type is not thread-safe.
// stop() first waits for cooperative IPC shutdown, then forcibly kills/reaps.
// Linux forced reaping is bounded to 250ms. An unreapable child is handed to a
// process-lifetime background reaper, never joined by Process/backend teardown.
class Process {
  public:
    Process() = default;
    Process(const Process &) = delete;
    Process &operator=(const Process &) = delete;
    virtual ~Process() = default;
    [[nodiscard]] virtual bool running() const = 0;
    [[nodiscard]] virtual std::uint32_t pid() const = 0;
    virtual void stop(std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) = 0;
};

// No parent environment entries or handles/descriptors are inherited. The
// caller must supply an absolute executable; cwd is its parent directory.
// Linux requires close_range (kernel 5.9+); inability to close every inherited
// descriptor fails startup rather than using an incomplete descriptor bound.
// PR_SET_PDEATHSIG tracks the thread that forks the child, not just its
// process. That thread must outlive the worker: spawn only on the persistent
// supervisor thread, never a transient request/launch helper thread.
Result<std::unique_ptr<Process>> spawn_process(const std::string &executable,
                                               const std::vector<std::string> &arguments,
                                               std::shared_ptr<ReapCompletion> *completion = nullptr);
Result<std::string> current_executable();
// Fresh, exact allowlist; never consults user-supplied environment variables.
std::vector<std::pair<std::string, std::string>> worker_environment();

} // namespace sonder::inference::worker
