#include "ipc.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// These SDK headers require the Windows type definitions above. Keep this
// separate include group so formatters cannot sort them before windows.h.
#include <bcrypt.h>
#include <sddl.h>
#elif defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace sonder::inference::worker {
namespace {
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

Status io_failure() { return {ErrorCode::io_error, "worker IPC connection closed or failed"}; }
Status timed_out() { return {ErrorCode::timeout, "worker IPC timeout"}; }
Status cancelled() { return {ErrorCode::cancelled, "worker IPC cancelled"}; }

// Framing has no shared mutable state: one reader and one writer use their own
// stack headers and payloads. The derived transport owns the cancellation flag.
class FramedChannel : public Channel {
  public:
    Result<std::string> read_frame(Milliseconds timeout, const CancellationToken &cancel) final {
        const auto deadline = Clock::now() + std::max(timeout, Milliseconds{0});
        std::array<unsigned char, 4> header{};
        std::size_t transferred = 0;
        auto st = transfer(false, header.data(), header.size(), deadline, cancel, transferred);
        if (!st.ok()) {
            if (st.code() != ErrorCode::timeout || transferred != 0)
                close();
            return st;
        }
        const auto size = (std::uint32_t{header[0]} << 24) | (std::uint32_t{header[1]} << 16) |
                          (std::uint32_t{header[2]} << 8) | std::uint32_t{header[3]};
        if (size == 0 || size > kMaxFrameBytes) {
            close();
            return Status(ErrorCode::protocol_error, "worker IPC frame size is invalid");
        }
        std::string payload(size, '\0');
        const auto payload_deadline = Clock::now() + std::max(timeout, Milliseconds{0});
        st = transfer(false, payload.data(), payload.size(), payload_deadline, cancel, transferred);
        if (!st.ok()) {
            close();
            return st;
        }
        return payload;
    }
    Status write_frame(std::string_view payload, Milliseconds timeout, const CancellationToken &cancel) final {
        if (payload.empty() || payload.size() > kMaxFrameBytes) {
            close();
            return {ErrorCode::protocol_error, "worker IPC frame size is invalid"};
        }
        const auto size = static_cast<std::uint32_t>(payload.size());
        std::vector<unsigned char> frame(payload.size() + 4);
        frame[0] = static_cast<unsigned char>(size >> 24);
        frame[1] = static_cast<unsigned char>(size >> 16);
        frame[2] = static_cast<unsigned char>(size >> 8);
        frame[3] = static_cast<unsigned char>(size);
        std::memcpy(frame.data() + 4, payload.data(), payload.size());
        const auto deadline = Clock::now() + std::max(timeout, Milliseconds{0});
        std::size_t transferred = 0;
        // One contiguous write avoids a scheduling gap between prefix and body.
        auto st = transfer(true, frame.data(), frame.size(), deadline, cancel, transferred);
        if (!st.ok())
            close();
        return st;
    }

  protected:
    virtual Status transfer(bool writing, void *data, std::size_t size, Clock::time_point deadline,
                            const CancellationToken &cancel, std::size_t &transferred) = 0;
};

#if defined(_WIN32)
std::wstring widen(std::string_view text) {
    if (text.empty() || text.size() > 32766 || text.find('\0') != std::string_view::npos)
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

// Owns the protected DACL backing memory until CreateNamedPipe returns.
struct UserSecurity {
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    ~UserSecurity() {
        if (descriptor)
            LocalFree(descriptor);
    }
    bool prepare() {
        HANDLE token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
            return false;
        DWORD size = 0;
        GetTokenInformation(token, TokenUser, nullptr, 0, &size);
        std::vector<unsigned char> bytes(size);
        const bool got = size != 0 && GetTokenInformation(token, TokenUser, bytes.data(), size, &size);
        CloseHandle(token);
        if (!got)
            return false;
        LPWSTR sid = nullptr;
        if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(bytes.data())->User.Sid, &sid))
            return false;
        const std::wstring sddl = L"D:P(A;;GA;;;" + std::wstring(sid) + L")";
        LocalFree(sid);
        return ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &descriptor,
                                                                    nullptr) != FALSE;
    }
};

// Each overlapped operation owns its event and stack OVERLAPPED until its OS
// completion is drained, including cancellation. The handle closes only after
// callers release the channel, avoiding close/reuse races with concurrent I/O.
class WindowsChannel final : public FramedChannel {
  public:
    explicit WindowsChannel(HANDLE handle) : handle_(handle) {}
    ~WindowsChannel() override {
        close();
        CloseHandle(handle_);
    }
    void close() override {
        closed_.store(true, std::memory_order_release);
        CancelIoEx(handle_, nullptr);
    }

  protected:
    Status transfer(bool writing, void *data, std::size_t size, Clock::time_point deadline,
                    const CancellationToken &cancel, std::size_t &transferred) override {
        auto *bytes = static_cast<unsigned char *>(data);
        std::size_t offset = 0;
        while (offset < size) {
            if (closed_.load(std::memory_order_acquire))
                return io_failure();
            if (cancel.cancelled())
                return cancelled();
            if (Clock::now() >= deadline)
                return timed_out();
            OVERLAPPED operation{};
            operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!operation.hEvent)
                return io_failure();
            DWORD count = 0;
            const auto remaining = static_cast<DWORD>(size - offset);
            const BOOL immediate = writing ? WriteFile(handle_, bytes + offset, remaining, &count, &operation)
                                           : ReadFile(handle_, bytes + offset, remaining, &count, &operation);
            Status result;
            if (!immediate) {
                if (GetLastError() != ERROR_IO_PENDING)
                    result = io_failure();
                else {
                    for (;;) {
                        const auto wait = WaitForSingleObject(operation.hEvent, 10);
                        if (wait == WAIT_OBJECT_0)
                            break;
                        if (wait != WAIT_TIMEOUT || closed_.load(std::memory_order_acquire)) {
                            result = io_failure();
                            break;
                        }
                        if (cancel.cancelled()) {
                            result = cancelled();
                            break;
                        }
                        if (Clock::now() >= deadline) {
                            result = timed_out();
                            break;
                        }
                    }
                    if (!result.ok())
                        CancelIoEx(handle_, &operation);
                    const BOOL completed = GetOverlappedResult(handle_, &operation, &count, TRUE);
                    // Completion can win the race with CancelIoEx. Consumed
                    // bytes are real progress even if our wait timed out first.
                    if (completed && count > 0)
                        result = Status::success();
                    else if (result.ok() && !completed)
                        result = io_failure();
                }
            }
            CloseHandle(operation.hEvent);
            // A timed-out operation can have completed while being cancelled.
            // Account for these bytes so a partial frame never looks retryable.
            transferred += count;
            offset += count;
            if (!result.ok())
                return result;
            if (count == 0)
                return io_failure();
        }
        return Status::success();
    }

  private:
    const HANDLE handle_;
    std::atomic<bool> closed_{false};
};

class WindowsListener final : public Listener {
  public:
    WindowsListener(std::string endpoint, HANDLE pipe) : endpoint_(std::move(endpoint)), pipe_(pipe) {}
    ~WindowsListener() override {
        if (pipe_ != INVALID_HANDLE_VALUE)
            CloseHandle(pipe_);
    }
    std::string endpoint() const override { return endpoint_; }
    Result<std::shared_ptr<Channel>> accept(Milliseconds timeout, const CancellationToken &cancel) override {
        if (pipe_ == INVALID_HANDLE_VALUE)
            return Status(ErrorCode::invalid_state, "worker IPC already accepted");
        OVERLAPPED operation{};
        operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!operation.hEvent)
            return io_failure();
        const BOOL immediate = ConnectNamedPipe(pipe_, &operation);
        const DWORD error = immediate ? ERROR_SUCCESS : GetLastError();
        Status st;
        if (error == ERROR_IO_PENDING) {
            const auto deadline = Clock::now() + std::max(timeout, Milliseconds{0});
            for (;;) {
                const auto wait = WaitForSingleObject(operation.hEvent, 10);
                if (wait == WAIT_OBJECT_0)
                    break;
                if (wait != WAIT_TIMEOUT) {
                    st = io_failure();
                    break;
                }
                if (cancel.cancelled()) {
                    st = cancelled();
                    break;
                }
                if (Clock::now() >= deadline) {
                    st = timed_out();
                    break;
                }
            }
            if (!st.ok())
                CancelIoEx(pipe_, &operation);
            DWORD ignored = 0;
            const BOOL completed = GetOverlappedResult(pipe_, &operation, &ignored, TRUE);
            if (completed)
                st = Status::success();
            else if (st.ok())
                st = io_failure();
        } else if (error != ERROR_SUCCESS && error != ERROR_PIPE_CONNECTED)
            st = io_failure();
        CloseHandle(operation.hEvent);
        if (!st.ok())
            return st;
        const HANDLE accepted = std::exchange(pipe_, INVALID_HANDLE_VALUE);
        return std::shared_ptr<Channel>(new WindowsChannel(accepted));
    }

  private:
    const std::string endpoint_;
    HANDLE pipe_;
};

#elif defined(__linux__)
Status wait_socket(int fd, short events, Clock::time_point deadline, const CancellationToken &cancel,
                   const std::atomic<bool> *closed = nullptr) {
    for (;;) {
        if (closed && closed->load(std::memory_order_acquire))
            return io_failure();
        if (cancel.cancelled())
            return cancelled();
        const auto now = Clock::now();
        if (now >= deadline)
            return timed_out();
        // A positive fractional millisecond still permits polling queued data.
        // Flooring it to zero made the supervisor's 1ms idle reads never poll.
        const auto remaining = std::chrono::ceil<Milliseconds>(deadline - now).count();
        pollfd item{fd, events, 0};
        const int rc = poll(&item, 1, static_cast<int>(std::min<std::int64_t>(remaining, 20)));
        if (rc < 0 && errno == EINTR)
            continue;
        if (rc < 0)
            return io_failure();
        if (rc == 0)
            continue;
        if (item.revents & events)
            return Status::success();
        if (item.revents & (POLLERR | POLLHUP | POLLNVAL))
            return io_failure();
    }
}
bool same_user(int fd) {
    ucred peer{};
    socklen_t size = sizeof(peer);
    return getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0 && size == sizeof(peer) && peer.uid == geteuid();
}
class UnixChannel final : public FramedChannel {
  public:
    explicit UnixChannel(int fd) : fd_(fd) {}
    ~UnixChannel() override {
        close();
        ::close(fd_);
    }
    void close() override {
        closed_.store(true, std::memory_order_release);
        shutdown(fd_, SHUT_RDWR);
    }

  protected:
    Status transfer(bool writing, void *data, std::size_t size, Clock::time_point deadline,
                    const CancellationToken &cancel, std::size_t &transferred) override {
        auto *bytes = static_cast<unsigned char *>(data);
        std::size_t offset = 0;
        while (offset < size) {
            auto st = wait_socket(fd_, writing ? POLLOUT : POLLIN, deadline, cancel, &closed_);
            if (!st.ok())
                return st;
            const ssize_t count = writing ? send(fd_, bytes + offset, size - offset, MSG_NOSIGNAL)
                                          : recv(fd_, bytes + offset, size - offset, 0);
            if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                continue;
            if (count <= 0)
                return io_failure();
            offset += static_cast<std::size_t>(count);
            transferred += static_cast<std::size_t>(count);
        }
        return Status::success();
    }

  private:
    const int fd_;
    std::atomic<bool> closed_{false};
};
class UnixListener final : public Listener {
  public:
    UnixListener(std::string directory, std::string endpoint, int fd)
        : directory_(std::move(directory)), endpoint_(std::move(endpoint)), fd_(fd) {}
    ~UnixListener() override {
        ::close(fd_);
        unlink(endpoint_.c_str());
        rmdir(directory_.c_str());
    }
    std::string endpoint() const override { return endpoint_; }
    Result<std::shared_ptr<Channel>> accept(Milliseconds timeout, const CancellationToken &cancel) override {
        if (accepted_)
            return Status(ErrorCode::invalid_state, "worker IPC already accepted");
        const auto deadline = Clock::now() + std::max(timeout, Milliseconds{0});
        for (;;) {
            auto st = wait_socket(fd_, POLLIN, deadline, cancel);
            if (!st.ok())
                return st;
            const int client = accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (client < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                continue;
            if (client < 0)
                return io_failure();
            if (!same_user(client)) {
                ::close(client);
                return Status(ErrorCode::protocol_error, "worker IPC peer uid mismatch");
            }
            accepted_ = true;
            return std::shared_ptr<Channel>(new UnixChannel(client));
        }
    }

  private:
    const std::string directory_;
    const std::string endpoint_;
    const int fd_;
    bool accepted_ = false;
};
#endif

Result<std::string> random_suffix() {
    std::array<unsigned char, 16> bytes{};
#if defined(_WIN32)
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        return io_failure();
#elif defined(__linux__)
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = getrandom(bytes.data() + offset, bytes.size() - offset, 0);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return io_failure();
        offset += static_cast<std::size_t>(count);
    }
#else
    return Status(ErrorCode::unsupported, "worker IPC supports Windows and Linux only");
#endif
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const auto byte : bytes) {
        out += hex[byte >> 4];
        out += hex[byte & 15];
    }
    return out;
}
} // namespace

Result<std::unique_ptr<Listener>> make_listener() {
    auto suffix = random_suffix();
    if (!suffix)
        return suffix.status();
#if defined(_WIN32)
    UserSecurity security;
    if (!security.prepare())
        return Status(ErrorCode::io_error, "worker IPC current-user DACL failed");
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), security.descriptor, FALSE};
    const std::string endpoint = "\\\\.\\pipe\\sonder-worker-" + suffix.value();
    const auto wide = widen(endpoint);
    HANDLE pipe = CreateNamedPipeW(
        wide.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536, 65536, 0, &attributes);
    if (pipe == INVALID_HANDLE_VALUE)
        return Status(ErrorCode::io_error, "worker IPC named pipe creation failed");
    return std::unique_ptr<Listener>(new WindowsListener(endpoint, pipe));
#elif defined(__linux__)
    const std::string directory = "/tmp/sonder-worker-" + suffix.value();
    const std::string endpoint = directory + "/ipc";
    if (mkdir(directory.c_str(), 0700) != 0)
        return io_failure();
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        rmdir(directory.c_str());
        return io_failure();
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, endpoint.c_str(), endpoint.size() + 1);
    if (bind(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0 ||
        chmod(endpoint.c_str(), 0600) != 0 || listen(fd, 1) != 0) {
        ::close(fd);
        unlink(endpoint.c_str());
        rmdir(directory.c_str());
        return io_failure();
    }
    return std::unique_ptr<Listener>(new UnixListener(directory, endpoint, fd));
#else
    return Status(ErrorCode::unsupported, "worker IPC supports Windows and Linux only");
#endif
}

Result<std::shared_ptr<Channel>> connect(std::string_view endpoint, Milliseconds timeout,
                                         const CancellationToken &cancel) {
    const auto deadline = Clock::now() + std::max(timeout, Milliseconds{0});
#if defined(_WIN32)
    constexpr std::string_view prefix = "\\\\.\\pipe\\sonder-worker-";
    if (!endpoint.starts_with(prefix) || endpoint.size() != prefix.size() + 32 ||
        endpoint.substr(prefix.size()).find_first_not_of("0123456789abcdef") != std::string_view::npos)
        return Status(ErrorCode::invalid_argument, "invalid worker pipe endpoint");
    const auto wide = widen(endpoint);
    if (wide.empty())
        return Status(ErrorCode::invalid_argument, "invalid worker pipe endpoint");
    for (;;) {
        if (cancel.cancelled())
            return cancelled();
        if (Clock::now() >= deadline)
            return timed_out();
        const HANDLE pipe =
            CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                        FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (pipe != INVALID_HANDLE_VALUE)
            return std::shared_ptr<Channel>(new WindowsChannel(pipe));
        const auto error = GetLastError();
        if (error != ERROR_PIPE_BUSY && error != ERROR_FILE_NOT_FOUND)
            return Status(ErrorCode::io_error,
                          "worker IPC CreateFileW failed (Windows error " + std::to_string(error) + ")");
        std::this_thread::sleep_for(Milliseconds{10});
    }
#elif defined(__linux__)
    constexpr std::string_view prefix = "/tmp/sonder-worker-";
    if (!endpoint.starts_with(prefix) || !endpoint.ends_with("/ipc") || endpoint.size() != prefix.size() + 32 + 4 ||
        endpoint.substr(prefix.size(), 32).find_first_not_of("0123456789abcdef") != std::string_view::npos)
        return Status(ErrorCode::invalid_argument, "invalid worker socket endpoint");
    const std::string name(endpoint);
    const std::string directory = name.substr(0, name.size() - 4);
    struct stat parent{}, node{};
    if (lstat(directory.c_str(), &parent) != 0 || !S_ISDIR(parent.st_mode) || parent.st_uid != geteuid() ||
        (parent.st_mode & 0777) != 0700 || lstat(name.c_str(), &node) != 0 || !S_ISSOCK(node.st_mode) ||
        node.st_uid != geteuid() || (node.st_mode & 0777) != 0600)
        return Status(ErrorCode::protocol_error, "worker socket ownership or permissions invalid");
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0)
        return io_failure();
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, name.c_str(), name.size() + 1);
    const int rc = ::connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address));
    if (rc < 0 && errno != EINPROGRESS) {
        ::close(fd);
        return io_failure();
    }
    if (rc < 0) {
        auto st = wait_socket(fd, POLLOUT, deadline, cancel);
        int error = 0;
        socklen_t size = sizeof(error);
        if (!st.ok() || getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) != 0 || error != 0) {
            ::close(fd);
            return st.ok() ? io_failure() : st;
        }
    }
    if (!same_user(fd)) {
        ::close(fd);
        return Status(ErrorCode::protocol_error, "worker IPC peer uid mismatch");
    }
    return std::shared_ptr<Channel>(new UnixChannel(fd));
#else
    (void)endpoint;
    (void)deadline;
    (void)cancel;
    return Status(ErrorCode::unsupported, "worker IPC supports Windows and Linux only");
#endif
}
} // namespace sonder::inference::worker
