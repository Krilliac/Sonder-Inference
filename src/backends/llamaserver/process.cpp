#include "process.hpp"
#include "process_options.hpp"

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
#include <fcntl.h>
#include <mutex>
#include <pthread.h>
#include <spawn.h>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
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
    std::uint32_t pid() const override { return process_ ? static_cast<std::uint32_t>(GetProcessId(process_)) : 0; }
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
        ProcessOptions extra;
        if (auto st = extra.prepare(spec); !st.ok()) return st;
        PROCESS_INFORMATION pi{};
        auto mutable_command = command;
        if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr, nullptr, extra.redirect ? TRUE : FALSE,
                            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW |
                                (extra.redirect ? EXTENDED_STARTUPINFO_PRESENT : 0),
                            extra.environment.empty() ? nullptr : extra.environment.data(),
                            nullptr, &extra.startup.StartupInfo, &pi)) {
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
struct Pipe {
    int read_fd = -1;
    int write_fd = -1;
    ~Pipe() {
        if (read_fd >= 0)
            close(read_fd);
        if (write_fd >= 0)
            close(write_fd);
    }
    bool open() {
        int fds[2];
#if defined(__linux__)
        if (pipe2(fds, O_CLOEXEC) != 0)
            return false;
#else
        if (pipe(fds) != 0)
            return false;
#endif
        read_fd = fds[0];
        write_fd = fds[1];
        return fcntl(read_fd, F_SETFD, FD_CLOEXEC) == 0 && fcntl(write_fd, F_SETFD, FD_CLOEXEC) == 0;
    }
};

// Only async-signal-safe operations are allowed in the forked watcher. In
// particular, never use C++ allocation, locks, logging or exit() here.
[[noreturn]] void watch_parent(int lifetime_fd, int ready_fd, int fd_limit) {
    const pid_t group = getpid();
    if (setpgid(0, 0) != 0)
        _exit(1);
    // Do not retain sockets, files, or another launcher's liveness writer. The
    // latter would keep that server alive after the real parent disappeared.
    for (int fd = 0; fd < fd_limit; ++fd) {
        if (fd != lifetime_fd && fd != ready_fd)
            close(fd);
    }
    const char ready = 'R';
    ssize_t n;
    do {
        n = write(ready_fd, &ready, 1);
    } while (n < 0 && errno == EINTR);
    close(ready_fd);
    if (n == 1) {
        char byte;
        do {
            n = read(lifetime_fd, &byte, 1);
        } while (n > 0 || (n < 0 && errno == EINTR));
    }
    // EOF is kernel-delivered even after a crash or SIGKILL. Use SIGKILL so a
    // server ignoring SIGTERM cannot keep model/GPU resources indefinitely.
    kill(-group, SIGKILL);
    _exit(0);
}

bool owns_child(pid_t pid) {
    siginfo_t info{};
    int rc;
    do {
        rc = waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT);
    } while (rc < 0 && errno == EINTR);
    return rc == 0;
}

void reap_child(pid_t pid) {
    if (pid > 0) {
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
        }
    }
}

class PosixProcess final : public Process {
  public:
    explicit PosixProcess(std::chrono::milliseconds timeout) : timeout_(timeout) {}
    void adopt_watcher(pid_t watcher, int lifetime_fd) {
        watcher_ = watcher;
        lifetime_fd_ = lifetime_fd;
    }
    void adopt_server(pid_t pid) { pid_ = pid; }
    ~PosixProcess() override { stop(timeout_); }
    std::uint32_t pid() const override { return pid_ > 0 ? static_cast<std::uint32_t>(pid_) : 0; }
    bool running() const override {
        if (pid_ <= 0)
            return false;
        siginfo_t watcher_info{};
        int rc;
        do {
            rc = waitid(P_PID, static_cast<id_t>(watcher_), &watcher_info, WEXITED | WNOHANG | WNOWAIT);
        } while (rc < 0 && errno == EINTR);
        if (rc != 0 || watcher_info.si_pid != 0)
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
        if (watcher_ <= 0)
            return;
        // The watcher PID reserves the server PGID even after the server has
        // been reaped. If another reaper took the watcher, do not signal a
        // potentially recycled group ID.
        if (owns_child(watcher_)) {
            kill(-watcher_, SIGTERM);
            const auto until =
                std::chrono::steady_clock::now() + (pid_ > 0 ? timeout : std::chrono::milliseconds(0));
            for (;;) {
                if (pid_ > 0) {
                    const auto reaped = waitpid(pid_, nullptr, WNOHANG);
                    if (reaped == pid_ || (reaped < 0 && errno == ECHILD))
                        pid_ = -1;
                }
                // Reap the exited server first: a zombie still makes kill(0)
                // report that its group exists. Descendants get the remaining
                // grace period, but an empty group needs no further delay.
                if (kill(-watcher_, 0) < 0 && errno == ESRCH)
                    break;
                if (std::chrono::steady_clock::now() >= until)
                    break;
                std::this_thread::sleep_for(std::min(std::chrono::milliseconds(20),
                                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                                         until - std::chrono::steady_clock::now())));
            }
            kill(-watcher_, SIGKILL);
        } else if (pid_ > 0 && owns_child(pid_)) {
            // A foreign reaper took the watchdog. The owned server can still
            // be stopped safely, but we no longer own the numeric group ID.
            kill(pid_, SIGKILL);
        }
        if (lifetime_fd_ >= 0) {
            close(lifetime_fd_);
            lifetime_fd_ = -1;
        }
        reap_child(pid_);
        reap_child(watcher_);
        pid_ = -1;
        watcher_ = -1;
    }

  private:
    mutable pid_t pid_ = -1;
    pid_t watcher_ = -1;
    int lifetime_fd_ = -1;
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
        ProcessOptions extra;
        if (auto st = extra.prepare(spec); !st.ok()) return st;

        // Establish the watchdog BEFORE spawning the server, so there is no
        // parent-death window between launching the server and arming cleanup.
        // A pipe works on Linux and macOS and follows process lifetime, unlike
        // Linux PR_SET_PDEATHSIG, which follows the creating THREAD's lifetime.
        int fd_limit = 0;
#if defined(__APPLE__)
        // macOS commonly reports an unlimited hard rlimit. The kernel's
        // per-process ceiling supplies a finite bound even in that case.
        std::size_t limit_size = sizeof(fd_limit);
        if (sysctlbyname("kern.maxfilesperproc", &fd_limit, &limit_size, nullptr, 0) != 0 || fd_limit <= 0)
            return Status(ErrorCode::io_error, "cannot determine watcher descriptor limit");
#else
        struct rlimit limit{};
        if (getrlimit(RLIMIT_NOFILE, &limit) != 0 || limit.rlim_max == RLIM_INFINITY ||
            limit.rlim_max > static_cast<rlim_t>(std::numeric_limits<int>::max()))
            return Status(ErrorCode::io_error, "cannot determine watcher descriptor limit");
        fd_limit = static_cast<int>(limit.rlim_max);
#endif
        // Serialize this launcher's pipe()+fcntl() fallback on macOS with
        // other launches, so they cannot exec an as-yet inheritable writer.
        static std::mutex launch_mutex;
        const std::lock_guard lock(launch_mutex);
        Pipe lifetime, ready;
        if (!lifetime.open() || !ready.open())
            return Status(ErrorCode::io_error, "watcher pipe creation failed");
        auto process = std::make_unique<PosixProcess>(spec.shutdown_timeout);
        sigset_t blocked, previous;
        sigfillset(&blocked);
        // Block before fork so no inherited handler can run in the watcher.
        // The server is spawned after restoring the caller's signal mask.
        if (pthread_sigmask(SIG_SETMASK, &blocked, &previous) != 0)
            return Status(ErrorCode::io_error, "watcher signal mask failed");
        const pid_t watcher = fork();
        if (watcher == 0)
            watch_parent(lifetime.read_fd, ready.write_fd, fd_limit);
        const int mask_rc = pthread_sigmask(SIG_SETMASK, &previous, nullptr);
        if (watcher < 0)
            return Status(ErrorCode::unavailable, "watcher fork failed");
        process->adopt_watcher(watcher, lifetime.write_fd);
        lifetime.write_fd = -1;
        close(ready.write_fd);
        ready.write_fd = -1;
        char reply = 0;
        ssize_t received;
        do {
            received = read(ready.read_fd, &reply, 1);
        } while (received < 0 && errno == EINTR);
        if (received != 1 || reply != 'R' || mask_rc != 0)
            return Status(ErrorCode::io_error, "watcher startup failed");

        posix_spawnattr_t attr{};
        if (posix_spawnattr_init(&attr) != 0)
            return Status(ErrorCode::io_error, "posix_spawnattr_init failed");
        if (posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP) != 0 ||
            posix_spawnattr_setpgroup(&attr, watcher) != 0) {
            posix_spawnattr_destroy(&attr);
            return Status(ErrorCode::io_error, "posix_spawn process group configuration failed");
        }
        pid_t pid = -1;
        const int rc = posix_spawnp(&pid, spec.executable.c_str(), extra.redirect ? &extra.actions : nullptr,
                                    &attr, argv.data(), extra.envp.empty() ? environ : extra.envp.data());
        posix_spawnattr_destroy(&attr);
        if (rc != 0)
            return Status(ErrorCode::unavailable, "posix_spawnp failed");
        process->adopt_server(pid);
        // The server now holds the group open. Move the watcher out so it
        // survives SIGTERM during normal shutdown and does not keep an empty
        // server group alive. Its unreaped PID still prevents PGID reuse.
        if (setpgid(watcher, getpgrp()) != 0)
            return Status(ErrorCode::io_error, "watcher group detachment failed");
        return std::unique_ptr<Process>(std::move(process));
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
