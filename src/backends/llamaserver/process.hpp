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
};

class ProcessLauncher {
  public:
    virtual ~ProcessLauncher() = default;
    virtual Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) = 0;
};

// The production launcher uses CreateProcessW/Job Objects on Windows and a
// process group on POSIX. Tests can inject a launcher without creating a child.
std::unique_ptr<ProcessLauncher> make_process_launcher();

Status validate_process_arguments(const std::vector<std::string> &arguments);

} // namespace sonder::inference::llamaserver
