#include "process.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace sonder::inference::llamaserver {
namespace {

bool is_forbidden(std::string_view arg) {
    return arg == "--" || arg == "--host" || arg.rfind("--host=", 0) == 0 || arg == "--port" ||
           arg.rfind("--port=", 0) == 0 || arg.find("--host_") == 0 || arg.find("--port_") == 0;
}

bool contains_nul(std::string_view s) { return s.find('\0') != std::string_view::npos; }

#if defined(_WIN32)
std::wstring widen(const std::string &s) {
    if (s.empty() || s.size() > 32766)
        return {};
    const int n =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0)
        return {};
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), out.data(),
                            n) != n)
        return {};
    return out;
}

std::wstring quote_windows(const std::string &text) {
    // CommandLineToArgvW-compatible quoting; CreateProcess does not invoke a shell.
    const auto w = widen(text);
    if (w.empty())
        return L"\"\"";
    if (w.find_first_of(L" \t\"") == std::wstring::npos)
        return w;
    std::wstring out = L"\"";
    std::size_t slashes = 0;
    for (const wchar_t c : w) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        if (c == L'"')
            out.append(slashes * 2 + 1, L'\\');
        else
            out.append(slashes, L'\\');
        slashes = 0;
        out.push_back(c);
    }
    out.append(slashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

class WindowsProcess final : public Process {
  public:
    WindowsProcess(HANDLE process, HANDLE job, std::chrono::milliseconds timeout)
        : process_(process), job_(job), timeout_(timeout) {}
    ~WindowsProcess() override { stop(timeout_); }
    bool running() const override {
        if (!process_)
            return false;
        return WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
    }
    void stop(std::chrono::milliseconds timeout) override {
        if (job_) {
            TerminateJobObject(job_, 0);
            CloseHandle(job_);
            job_ = nullptr;
        }
        if (process_) {
            const auto wait_ms = std::clamp<std::int64_t>(timeout.count(), 1, 86400000);
            WaitForSingleObject(process_, static_cast<DWORD>(wait_ms));
            CloseHandle(process_);
            process_ = nullptr;
        }
    }

  private:
    HANDLE process_ = nullptr;
    HANDLE job_ = nullptr;
    std::chrono::milliseconds timeout_;
};

class WindowsLauncher final : public ProcessLauncher {
  public:
    Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
        if (spec.executable.empty() || spec.port == 0 || contains_nul(spec.executable))
            return Status(ErrorCode::invalid_argument, "invalid process spec");
        const auto executable = widen(spec.executable);
        if (executable.empty())
            return Status(ErrorCode::invalid_argument, "executable is not valid UTF-8 or is too long");
        std::wstring command = quote_windows(spec.executable);
        for (const auto &arg : spec.arguments) {
            if (contains_nul(arg))
                return Status(ErrorCode::invalid_argument, "argument contains NUL");
            if (!arg.empty() && widen(arg).empty())
                return Status(ErrorCode::invalid_argument, "argument is not valid UTF-8 or is too long");
            const auto quoted = quote_windows(arg);
            if (quoted.empty())
                return Status(ErrorCode::invalid_argument, "argument is not valid UTF-8");
            command.push_back(L' ');
            command += quoted;
        }
        if (command.size() >= 32767)
            return Status(ErrorCode::invalid_argument, "command line is too long");
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        auto mutable_command = command;
        if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
                            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, nullptr,
                            nullptr, &si, &pi)) {
            return Status(ErrorCode::unavailable, "CreateProcessW failed");
        }
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        const auto cleanup = [&] {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            if (job)
                CloseHandle(job);
        };
        if (!job) {
            cleanup();
            return Status(ErrorCode::io_error, "CreateJobObject failed");
        }
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            cleanup();
            return Status(ErrorCode::io_error, "SetInformationJobObject failed");
        }
        if (!AssignProcessToJobObject(job, pi.hProcess)) {
            cleanup();
            return Status(ErrorCode::io_error, "AssignProcessToJobObject failed");
        }
        if (ResumeThread(pi.hThread) == static_cast<DWORD>(-1)) {
            cleanup();
            return Status(ErrorCode::io_error, "ResumeThread failed");
        }
        CloseHandle(pi.hThread);
        return std::unique_ptr<Process>(new WindowsProcess(pi.hProcess, job, spec.shutdown_timeout));
    }
};
#else
class PosixProcess final : public Process {
  public:
    PosixProcess(pid_t pid, std::chrono::milliseconds timeout) : pid_(pid), timeout_(timeout) {}
    ~PosixProcess() override { stop(timeout_); }
    bool running() const override {
        if (pid_ <= 0)
            return false;
        siginfo_t info{};
        if (waitid(P_PID, static_cast<id_t>(pid_), &info, WEXITED | WNOHANG | WNOWAIT) != 0) {
            if (errno == ECHILD)
                pid_ = -1; // another reaper took ownership; never signal a recycled ID
            return errno == EINTR;
        }
        return info.si_pid == 0;
    }
    void stop(std::chrono::milliseconds timeout) override {
        (void)running(); // checks ownership without reaping
        if (pid_ <= 0)
            return;
        kill(-pid_, SIGTERM);
        const auto until = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < until) {
            siginfo_t info{};
            if (waitid(P_PID, static_cast<id_t>(pid_), &info, WEXITED | WNOHANG | WNOWAIT) != 0 &&
                errno == ECHILD) {
                pid_ = -1;
                return; // ownership lost; do not signal a potentially recycled group
            }
            // Give descendants the grace period even if the leader exited.
            std::this_thread::sleep_for(
                std::min(std::chrono::milliseconds(20), std::chrono::duration_cast<std::chrono::milliseconds>(
                                                            until - std::chrono::steady_clock::now())));
        }
        kill(-pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
        }
        pid_ = -1;
    }

  private:
    mutable pid_t pid_ = -1;
    std::chrono::milliseconds timeout_;
};

class PosixLauncher final : public ProcessLauncher {
  public:
    Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
        if (spec.executable.empty() || spec.port == 0 || contains_nul(spec.executable))
            return Status(ErrorCode::invalid_argument, "invalid process spec");
        std::vector<std::string> args;
        args.reserve(spec.arguments.size() + 1);
        args.push_back(spec.executable);
        for (const auto &arg : spec.arguments) {
            if (contains_nul(arg))
                return Status(ErrorCode::invalid_argument, "argument contains NUL");
            args.push_back(arg);
        }
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (auto &arg : args)
            argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawnattr_t attr{};
        if (posix_spawnattr_init(&attr) != 0)
            return Status(ErrorCode::io_error, "posix_spawnattr_init failed");
        if (posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP) != 0 ||
            posix_spawnattr_setpgroup(&attr, 0) != 0) {
            posix_spawnattr_destroy(&attr);
            return Status(ErrorCode::io_error, "posix_spawn process group configuration failed");
        }
        pid_t pid = -1;
        const int rc = posix_spawnp(&pid, spec.executable.c_str(), nullptr, &attr, argv.data(), environ);
        posix_spawnattr_destroy(&attr);
        if (rc != 0)
            return Status(ErrorCode::unavailable, "posix_spawnp failed");
        return std::unique_ptr<Process>(new PosixProcess(pid, spec.shutdown_timeout));
    }
};
#endif
} // namespace

Status validate_process_arguments(const std::vector<std::string> &arguments) {
    for (const auto &arg : arguments) {
        if (contains_nul(arg))
            return Status(ErrorCode::invalid_argument, "argument contains NUL");
        if (is_forbidden(arg))
            return Status(ErrorCode::invalid_argument, "llama-server host/port overrides are forbidden");
    }
    return Status::success();
}

std::unique_ptr<ProcessLauncher> make_process_launcher() {
#if defined(_WIN32)
    return std::make_unique<WindowsLauncher>();
#else
    return std::make_unique<PosixLauncher>();
#endif
}
} // namespace sonder::inference::llamaserver
