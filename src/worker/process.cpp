#include "process.hpp"

#include <algorithm>
#include <cerrno>
#include <exception>
#include <filesystem>
#include <limits>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include "process_reap.hpp"
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace sonder::inference::worker {
namespace {
using Milliseconds = std::chrono::milliseconds;
using Clock = std::chrono::steady_clock;

#if defined(_WIN32)
std::wstring widen(const std::string &text) {
    if (text.empty() || text.size() > 32766 || text.find('\0') != std::string::npos)
        return {};
    const int size =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0)
        return {};
    std::wstring out(static_cast<std::size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), out.data(),
                            size) != size)
        return {};
    return out;
}
std::string narrow(std::wstring_view text) {
    if (text.empty() || text.size() > 32766)
        return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                                         nullptr, 0, nullptr, nullptr);
    if (size <= 0)
        return {};
    std::string out(static_cast<std::size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), out.data(), size,
                            nullptr, nullptr) != size)
        return {};
    return out;
}
std::wstring quote(const std::wstring &text) {
    std::wstring out = L"\"";
    std::size_t slashes = 0;
    for (const auto character : text) {
        if (character == L'\\') {
            ++slashes;
            continue;
        }
        out.append(character == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        slashes = 0;
        out += character;
    }
    out.append(slashes * 2, L'\\');
    out += L'"';
    return out;
}

class WindowsProcess final : public Process {
  public:
    WindowsProcess(HANDLE process, HANDLE job) : process_(process), job_(job) {}
    ~WindowsProcess() override { stop(Milliseconds{0}); }
    bool running() const override { return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT; }
    std::uint32_t pid() const override { return process_ ? GetProcessId(process_) : 0; }
    void stop(Milliseconds timeout) override {
        if (!process_)
            return;
        const auto wait = static_cast<DWORD>(std::clamp<std::int64_t>(timeout.count(), 0, 60000));
        WaitForSingleObject(process_, wait);
        // Closing the job kills the child and any descendants, including after
        // the direct child has exited. The job handle itself is never inherited.
        if (job_) {
            TerminateJobObject(job_, 1);
            CloseHandle(job_);
            job_ = nullptr;
        }
        WaitForSingleObject(process_, 5000);
        CloseHandle(process_);
        process_ = nullptr;
    }

  private:
    HANDLE process_;
    HANDLE job_;
};
#elif defined(__linux__)
enum class ChildState { running, exited, gone, unknown };
ChildState observe_child(pid_t pid, Clock::time_point deadline) {
    siginfo_t info{};
    const auto [rc, error] = detail::retry_interrupted(
        deadline,
        [&] {
            const int result = waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT);
            const int captured_errno = errno;
            return std::pair{result, captured_errno};
        },
        [] { return Clock::now(); });
    if (rc == 0)
        return info.si_pid == 0 ? ChildState::running : ChildState::exited;
    return error == ECHILD ? ChildState::gone : ChildState::unknown;
}

detail::DeferredReaper &deferred_reaper() {
    // The pool and its one sweeper live until process exit. No backend joins
    // the thread. Initialization (including thread creation) precedes fork.
    static auto *instance = [] {
        auto pool = std::make_unique<detail::DeferredReaper>(detail::terminate_and_poll);
        std::thread sweeper([ptr = pool.get()] {
            for (;;) {
                ptr->sweep();
                std::this_thread::sleep_for(Milliseconds{100});
            }
        });
        sweeper.detach();
        return pool.release();
    }();
    return *instance;
}

class LinuxProcess final : public Process {
  public:
    LinuxProcess()
        : reaper_(deferred_reaper()), completion_(std::make_shared<ReapCompletion>()),
          ticket_(reaper_.reserve(completion_)) {}
    bool has_lease() const { return ticket_ != nullptr; }
    const std::shared_ptr<ReapCompletion> &completion() const { return completion_; }
    void adopt(pid_t pid) {
        pid_ = pid;
        ticket_->pid = pid;
    }
    ~LinuxProcess() override { stop(Milliseconds{0}); }
    bool running() const override {
        if (pid_ <= 0)
            return false;
        const auto state = observe_child(pid_, Clock::now() + Milliseconds{10});
        // An interrupted observation does not establish child exit.
        return state == ChildState::running || state == ChildState::unknown;
    }
    std::uint32_t pid() const override { return pid_ > 0 ? static_cast<std::uint32_t>(pid_) : 0; }
    void stop(Milliseconds timeout) override {
        if (!ticket_)
            return; // already reaped or handed off
        if (pid_ <= 0) {
            reaper_.release(std::exchange(ticket_, nullptr)); // pre-fork failure
            return;
        }
        const auto deadline = Clock::now() + std::clamp(timeout, Milliseconds{0}, Milliseconds{60000});
        while (Clock::now() < deadline) {
            const auto state = observe_child(pid_, std::min(deadline, Clock::now() + Milliseconds{10}));
            if (state != ChildState::running && state != ChildState::unknown)
                break;
            const auto remaining = std::chrono::ceil<Milliseconds>(deadline - Clock::now());
            if (remaining > Milliseconds{0})
                std::this_thread::sleep_for(std::min(Milliseconds{10}, remaining));
        }
        const auto forced_deadline = Clock::now() + detail::kForcedReapBudget;
        const auto state = observe_child(pid_, forced_deadline);
        // WNOWAIT retains PID ownership while signalling. Unknown ownership
        // is never treated as ECHILD and never releases the lease or PID.
        if (state == ChildState::running || state == ChildState::exited) {
            kill(-pid_, SIGKILL);
            kill(pid_, SIGKILL);
        }
        const auto remaining = std::max(
            Milliseconds{0}, std::chrono::duration_cast<Milliseconds>(forced_deadline - Clock::now()));
        (void)reaper_.reap_or_defer(
            ticket_, remaining, detail::poll_reaped, [] { return Clock::now(); },
            [](Milliseconds duration) { std::this_thread::sleep_for(duration); });
        pid_ = -1;
    }

  private:
    pid_t pid_ = -1;
    detail::DeferredReaper &reaper_;
    std::shared_ptr<ReapCompletion> completion_;
    detail::DeferredReaper::Ticket *ticket_;
};

// Runs only in the post-fork child: no allocations, locks, C++ destructors or
// stdio. The sole temporary inherited fd is the CLOEXEC error-report pipe (3).
[[noreturn]] void child_error(int error, int report_fd = 3) {
    const auto *bytes = reinterpret_cast<const char *>(&error);
    std::size_t offset = 0;
    while (offset < sizeof(error)) {
        const ssize_t count = write(report_fd, bytes + offset, sizeof(error) - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            break;
        offset += static_cast<std::size_t>(count);
    }
    _exit(127);
}
#endif
} // namespace

std::vector<std::pair<std::string, std::string>> worker_environment() {
#if defined(_WIN32)
    // GetWindowsDirectory reads the OS installation path, not the parent's
    // potentially attacker-controlled SystemRoot environment entry.
    std::vector<wchar_t> directory(32768, L'\0');
    const UINT size = GetWindowsDirectoryW(directory.data(), static_cast<UINT>(directory.size()));
    if (size == 0 || size >= directory.size())
        return {};
    auto root = narrow(std::wstring_view(directory.data(), size));
    if (root.empty())
        return {};
    return {{"NoDefaultCurrentDirectoryInExePath", "1"}, {"SystemRoot", std::move(root)}};
#elif defined(__linux__)
    return {{"LANG", "C.UTF-8"}};
#else
    return {};
#endif
}

Result<std::string> current_executable() {
#if defined(_WIN32)
    std::vector<wchar_t> path(32768, L'\0');
    const DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (size == 0 || size >= path.size())
        return Status(ErrorCode::io_error, "worker executable path unavailable");
    auto result = narrow(std::wstring_view(path.data(), size));
    if (result.empty())
        return Status(ErrorCode::io_error, "worker executable path is not valid Unicode");
    return result;
#elif defined(__linux__)
    std::vector<char> path(4096, '\0');
    const auto size = readlink("/proc/self/exe", path.data(), path.size());
    if (size <= 0 || static_cast<std::size_t>(size) >= path.size())
        return Status(ErrorCode::io_error, "worker executable path unavailable");
    return std::string(path.data(), static_cast<std::size_t>(size));
#else
    return Status(ErrorCode::unsupported, "worker processes support Windows and Linux only");
#endif
}

Result<std::unique_ptr<Process>> spawn_process(const std::string &executable,
                                               const std::vector<std::string> &arguments,
                                               std::shared_ptr<ReapCompletion> *completion) {
    if (completion)
        completion->reset();
    if (executable.empty() || executable.find('\0') != std::string::npos || arguments.size() > 128)
        return Status(ErrorCode::invalid_argument, "invalid worker process arguments");
    for (const auto &argument : arguments) {
        if (argument.find('\0') != std::string::npos || argument.size() > 32766)
            return Status(ErrorCode::invalid_argument, "invalid worker process argument");
    }
#if defined(_WIN32)
    const auto wide = widen(executable);
    if (wide.empty())
        return Status(ErrorCode::invalid_argument, "worker executable is not valid UTF-8");
    const std::filesystem::path path(wide);
    if (!path.is_absolute())
        return Status(ErrorCode::invalid_argument, "worker executable must be absolute");
    const auto directory = path.parent_path().wstring();
    std::wstring command = quote(wide);
    for (const auto &argument : arguments) {
        const auto arg = argument.empty() ? std::wstring{} : widen(argument);
        if (!argument.empty() && arg.empty())
            return Status(ErrorCode::invalid_argument, "worker argument is not valid UTF-8");
        command += L' ';
        command += quote(arg);
    }
    if (command.size() >= 32767)
        return Status(ErrorCode::invalid_argument, "worker command line too long");
    const auto entries = worker_environment();
    if (entries.size() != 2)
        return Status(ErrorCode::io_error, "worker OS environment unavailable");
    std::vector<wchar_t> environment;
    for (const auto &[key, value] : entries) {
        const auto entry = widen(key + "=" + value);
        if (entry.empty())
            return Status(ErrorCode::io_error, "worker OS environment invalid");
        environment.insert(environment.end(), entry.begin(), entry.end());
        environment.push_back(L'\0');
    }
    environment.push_back(L'\0');
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job)
        return Status(ErrorCode::io_error, "worker job creation failed");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        CloseHandle(job);
        return Status(ErrorCode::io_error, "worker job configuration failed");
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(wide.c_str(), command.data(), nullptr, nullptr, FALSE,
                        CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED | CREATE_NO_WINDOW, environment.data(),
                        directory.c_str(), &startup, &child)) {
        CloseHandle(job);
        return Status(ErrorCode::unavailable, "worker CreateProcessW failed");
    }
    if (!AssignProcessToJobObject(job, child.hProcess) || ResumeThread(child.hThread) == static_cast<DWORD>(-1)) {
        TerminateProcess(child.hProcess, 1);
        WaitForSingleObject(child.hProcess, 5000);
        CloseHandle(child.hThread);
        CloseHandle(child.hProcess);
        CloseHandle(job);
        return Status(ErrorCode::io_error, "worker job assignment or resume failed");
    }
    CloseHandle(child.hThread);
    return std::unique_ptr<Process>(new WindowsProcess(child.hProcess, job));
#elif defined(__linux__)
    if (executable.front() != '/')
        return Status(ErrorCode::invalid_argument, "worker executable must be absolute");
    // Establish the long-lived reaper and reserve its ticket before a child can
    // exist. Resource failure here is a launch error, never a shutdown exception.
    std::unique_ptr<LinuxProcess> process;
    try {
        process = std::make_unique<LinuxProcess>();
        if (!process->has_lease())
            return Status(ErrorCode::unavailable, "worker child/reap lease cap reached (64 process-wide)");
    } catch (const std::exception &) {
        return Status(ErrorCode::unavailable, "worker deferred reaper initialization failed");
    }
    const std::string directory = std::filesystem::path(executable).parent_path().string();
    std::vector<std::string> argv_storage{executable};
    argv_storage.insert(argv_storage.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    for (auto &item : argv_storage)
        argv.push_back(item.data());
    argv.push_back(nullptr);
    std::vector<std::string> env_storage;
    for (const auto &[key, value] : worker_environment())
        env_storage.push_back(key + "=" + value);
    std::vector<char *> envp;
    for (auto &item : env_storage)
        envp.push_back(item.data());
    envp.push_back(nullptr);
    int errors[2];
    if (pipe2(errors, O_CLOEXEC) != 0)
        return Status(ErrorCode::io_error, "worker exec error pipe failed");
    // Keep control descriptors out of stdio even when the parent was started
    // with stdin/stdout/stderr closed.
    for (int index = 0; index < 2; ++index) {
        if (errors[index] >= 3)
            continue;
        const int duplicate = fcntl(errors[index], F_DUPFD_CLOEXEC, 3);
        if (duplicate < 0) {
            close(errors[0]);
            close(errors[1]);
            return Status(ErrorCode::io_error, "worker exec error descriptor setup failed");
        }
        close(errors[index]);
        errors[index] = duplicate;
    }
    int null_fd = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (null_fd < 0) {
        close(errors[0]);
        close(errors[1]);
        return Status(ErrorCode::io_error, "worker null stdio failed");
    }
    if (null_fd < 4) {
        const int duplicate = fcntl(null_fd, F_DUPFD_CLOEXEC, 4);
        close(null_fd);
        if (duplicate < 0) {
            close(errors[0]);
            close(errors[1]);
            return Status(ErrorCode::io_error, "worker null stdio descriptor setup failed");
        }
        null_fd = duplicate;
    }
    const pid_t parent = getpid();
    const pid_t child = fork();
    if (child < 0) {
        close(errors[0]);
        close(errors[1]);
        close(null_fd);
        return Status(ErrorCode::unavailable, "worker fork failed");
    }
    if (child == 0) {
        // Establish error reporting before any fallible child setup. null_fd
        // is >=4, so replacing fd 3 cannot clobber the controlled stdio source.
        if (errors[1] != 3 && dup2(errors[1], 3) < 0)
            child_error(errno, errors[1]);
        if (fcntl(3, F_SETFD, FD_CLOEXEC) < 0)
            child_error(errno);
        for (int fd = 0; fd < 3; ++fd) {
            if (dup2(null_fd, fd) < 0 || fcntl(fd, F_SETFD, 0) < 0)
                child_error(errno);
        }
#if defined(SYS_close_range)
        if (syscall(SYS_close_range, 4u, std::numeric_limits<unsigned int>::max(), 0u) != 0)
            child_error(errno);
#else
        child_error(ENOSYS);
#endif
        if (setpgid(0, 0) != 0 || prctl(PR_SET_PDEATHSIG, SIGKILL) != 0)
            child_error(errno);
        if (getppid() != parent)
            child_error(ESRCH);
        if (chdir(directory.c_str()) != 0)
            child_error(errno);
        execve(executable.c_str(), argv.data(), envp.data());
        child_error(errno);
    }
    process->adopt(child);
    if (completion)
        *completion = process->completion();
    close(errors[1]);
    close(null_fd);
    int child_errno = 0;
    auto *bytes = reinterpret_cast<char *>(&child_errno);
    std::size_t count = 0;
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    for (;;) {
        const auto remaining = std::chrono::duration_cast<Milliseconds>(deadline - Clock::now()).count();
        if (remaining <= 0) {
            child_errno = ETIMEDOUT;
            break;
        }
        pollfd report{errors[0], POLLIN, 0};
        const int ready = poll(&report, 1, static_cast<int>(remaining));
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready == 0) {
            child_errno = ETIMEDOUT;
            break;
        }
        if (ready < 0 || (report.revents & (POLLERR | POLLNVAL))) {
            child_errno = EIO;
            break;
        }
        const auto n = read(errors[0], bytes + count, sizeof(child_errno) - count);
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0) {
            child_errno = EIO;
            break;
        }
        if (n == 0)
            break;
        count += static_cast<std::size_t>(n);
        if (count == sizeof(child_errno))
            break;
    }
    close(errors[0]);
    if (count != 0 || child_errno != 0) {
        process->stop(Milliseconds{0});
        return Status(ErrorCode::unavailable, "worker exec failed (errno " + std::to_string(child_errno) + ")");
    }
    return std::unique_ptr<Process>(std::move(process));
#else
    (void)arguments;
    return Status(ErrorCode::unsupported, "worker processes support Windows and Linux only");
#endif
}
} // namespace sonder::inference::worker
