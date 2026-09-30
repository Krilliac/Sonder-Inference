// Optional launch setup. Keeping it separate leaves the existing process
// ownership, watchdog and shutdown paths unchanged.
#pragma once
#include "process.hpp"
#include <cstddef>
#include <array>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <spawn.h>
#endif

namespace sonder::inference::llamaserver {
class ProcessOptions {
  public:
    ProcessOptions() = default;
    ~ProcessOptions();
    ProcessOptions(const ProcessOptions &) = delete;
    ProcessOptions &operator=(const ProcessOptions &) = delete;
    Status prepare(const ProcessSpec &spec);
#if defined(_WIN32)
    STARTUPINFOEXW startup{};
    std::vector<wchar_t> environment;
    bool redirect = false;
  private:
    HANDLE output_ = INVALID_HANDLE_VALUE;
    HANDLE input_ = INVALID_HANDLE_VALUE;
    std::vector<std::max_align_t> attributes_;
    std::array<HANDLE, 2> inherited_{}; // attribute values must outlive CreateProcessW
#else
    posix_spawn_file_actions_t actions{};
    bool redirect = false;
    std::vector<std::string> environment;
    std::vector<char *> envp;
#endif
};
} // namespace sonder::inference::llamaserver
