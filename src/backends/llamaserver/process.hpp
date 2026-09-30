#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

struct ProcessSpec {
    std::string executable;
    std::vector<std::string> arguments;
    std::uint16_t port = 0;
    std::chrono::milliseconds shutdown_timeout{2000};
};

class Process {
  public:
    virtual ~Process() = default;
    virtual bool running() const = 0;
    virtual void stop(std::chrono::milliseconds timeout = std::chrono::milliseconds{2000}) = 0;
    // OS process id of the server (not a watcher), or 0 when unknown. Used
    // only for per-process diagnostics such as GPU memory counters.
    [[nodiscard]] virtual std::uint32_t pid() const { return 0; }
};

class ProcessLauncher {
  public:
    virtual ~ProcessLauncher() = default;
    virtual Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) = 0;
};

// The production launcher uses CreateProcessW/Job Objects on Windows and a
// process group with a parent-liveness pipe watcher on POSIX (Linux/macOS).
// Tests can inject a launcher without creating a child.
std::unique_ptr<ProcessLauncher> make_process_launcher();

Status validate_process_arguments(const std::vector<std::string> &arguments);

} // namespace sonder::inference::llamaserver
