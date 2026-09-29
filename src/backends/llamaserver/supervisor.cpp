#include "supervisor.hpp"

#include <algorithm>
#include <exception>
#include <utility>

#include "net/http_client.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace sonder::inference::llamaserver {
namespace {
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

Result<std::uint16_t> ephemeral_port() {
#if defined(_WIN32)
    struct WinsockScope {
        WSADATA data{};
        int result = WSAStartup(MAKEWORD(2, 2), &data);
        ~WinsockScope() {
            if (result == 0)
                WSACleanup();
        }
    } winsock;
    if (winsock.result != 0)
        return Status(ErrorCode::io_error, "llamaserver: WSAStartup failed");
    const SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET)
        return Status(ErrorCode::io_error, "llamaserver: socket failed");
    const auto close_socket = [&] { closesocket(fd); };
#else
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return Status(ErrorCode::io_error, "llamaserver: socket failed");
    const auto close_socket = [&] { close(fd); };
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        close_socket();
        return Status(ErrorCode::unavailable, "llamaserver: cannot reserve loopback port");
    }
#if defined(_WIN32)
    int size = sizeof(addr);
#else
    socklen_t size = sizeof(addr);
#endif
    const int result = getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &size);
    close_socket();
    if (result != 0 || addr.sin_port == 0)
        return Status(ErrorCode::io_error, "llamaserver: getsockname failed");
    // llama-server cannot inherit this listener. A bind handoff is unavoidable.
    return ntohs(addr.sin_port);
}
} // namespace

Supervisor::Supervisor(SupervisorOptions options, std::unique_ptr<ProcessLauncher> launcher)
    : options_(std::move(options)), launcher_(launcher ? std::move(launcher) : make_process_launcher()) {}
Supervisor::~Supervisor() { stop(); }

Result<std::uint16_t> Supervisor::start(const CancellationToken &cancel) {
    std::unique_lock lock(mutex_);
    if (cancel.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
    if (shutdown_.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: supervisor stopped");
    if (state_ == State::idle) {
        const auto valid_duration = [](Milliseconds value) {
            return value.count() > 0 && value <= std::chrono::hours(24);
        };
        Status valid = validate_process_arguments(options_.arguments);
        if (options_.executable.empty() || options_.executable.find('\0') != std::string::npos ||
            !valid_duration(options_.readiness_timeout) || !valid_duration(options_.health_poll_interval) ||
            !valid_duration(options_.restart_initial_backoff) ||
            !valid_duration(options_.restart_max_backoff) || !valid_duration(options_.shutdown_timeout) ||
            options_.restart_initial_backoff > options_.restart_max_backoff || options_.max_restarts > 1000) {
            valid = Status(ErrorCode::invalid_argument, "llamaserver: invalid process supervision options");
        }
        if (!valid.ok()) {
            state_ = State::failed;
            failure_ = valid;
            return valid;
        }
        state_ = State::starting;
        try {
            monitor_thread_ = std::thread(&Supervisor::monitor, this);
        } catch (const std::exception &e) {
            state_ = State::failed;
            failure_ = Status(ErrorCode::unavailable,
                              std::string("llamaserver: monitor startup failed: ") + e.what());
        }
    }
    const auto deadline = Clock::now() + options_.readiness_timeout + options_.restart_max_backoff;
    for (;;) {
        if (cancel.cancelled() || shutdown_.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: readiness wait cancelled");
        if (state_ == State::ready)
            return port_;
        if (state_ == State::failed)
            return failure_;
        if (Clock::now() >= deadline)
            return Status(ErrorCode::timeout, "llamaserver: readiness wait timed out");
        wake_.wait_until(lock, std::min(deadline, Clock::now() + Milliseconds(10)));
    }
}

bool Supervisor::pause(Milliseconds duration) {
    std::unique_lock lock(mutex_);
    return wake_.wait_for(lock, duration, [&] { return shutdown_.cancelled(); });
}

Status Supervisor::wait_ready(Process &child, std::uint16_t port) {
    const auto deadline = Clock::now() + options_.readiness_timeout;
    while (!shutdown_.cancelled()) {
        if (!child.running())
            return Status(ErrorCode::unavailable, "llamaserver: child exited during startup");
        const auto remaining = std::chrono::duration_cast<Milliseconds>(deadline - Clock::now());
        if (remaining.count() <= 0)
            return Status(ErrorCode::timeout, "llamaserver: /health readiness timed out");
        const auto budget = std::min(remaining, Milliseconds(250));
        bool healthy = false;
        if (options_.health_check) {
            healthy = options_.health_check(port, budget);
        } else {
            net::HttpRequest request;
            request.host = "127.0.0.1";
            request.port = port;
            request.target = "/health";
            request.connect_timeout = budget;
            request.total_timeout = budget;
            std::string body;
            const auto response = net::http_request_buffered(request, body, shutdown_.token(), 4096);
            healthy = response.ok() && response->status == 200;
        }
        if (healthy && child.running())
            return Status::success();
        const auto left = std::chrono::duration_cast<Milliseconds>(deadline - Clock::now());
        if (left.count() > 0)
            pause(std::min(left, options_.health_poll_interval));
    }
    return Status(ErrorCode::cancelled, "llamaserver: startup cancelled");
}

void Supervisor::fail(Status status) {
    std::lock_guard lock(mutex_);
    failure_ = std::move(status);
    port_ = 0;
    state_ = State::failed;
    wake_.notify_all();
}

void Supervisor::monitor() {
    // Only this thread accesses Process: native wait, reap and stop never race.
    std::size_t restarts = 0;
    auto backoff = options_.restart_initial_backoff;
    try {
        while (!shutdown_.cancelled()) {
            Status result;
            auto selected = ephemeral_port();
            if (!selected.ok()) {
                result = selected.status();
            } else {
                ProcessSpec spec{options_.executable, options_.arguments, selected.value(),
                                 options_.shutdown_timeout};
                spec.arguments.insert(spec.arguments.end(),
                                      {"--host", "127.0.0.1", "--port", std::to_string(spec.port)});
                if (shutdown_.cancelled())
                    break;
                auto launched = launcher_->start(spec);
                if (!launched.ok()) {
                    result = launched.status();
                } else if (!launched.value()) {
                    result = Status(ErrorCode::internal, "llamaserver: launcher returned no process");
                } else {
                    auto child = std::move(launched.value());
                    result = wait_ready(*child, spec.port);
                    if (result.ok() && !shutdown_.cancelled()) {
                        {
                            std::lock_guard lock(mutex_);
                            port_ = spec.port;
                            state_ = State::ready;
                            failure_ = Status::success();
                            wake_.notify_all();
                        }
                        while (!shutdown_.cancelled() && child->running())
                            pause(options_.health_poll_interval);
                        result = Status(ErrorCode::unavailable, "llamaserver: child exited");
                    }
                    {
                        std::lock_guard lock(mutex_);
                        port_ = 0;
                        state_ = State::backoff;
                    }
                    child->stop(options_.shutdown_timeout);
                }
            }
            if (shutdown_.cancelled())
                break;
            // A live child that never becomes ready has timed out; do not
            // repeatedly load a model behind the caller's back in that case.
            if (result.code() == ErrorCode::timeout || result.code() == ErrorCode::invalid_argument ||
                restarts == options_.max_restarts) {
                fail(result);
                return;
            }
            ++restarts;
            {
                std::lock_guard lock(mutex_);
                state_ = State::backoff;
                failure_ = result;
                wake_.notify_all();
            }
            if (pause(backoff))
                break;
            backoff = std::min(options_.restart_max_backoff, backoff * 2);
            {
                std::lock_guard lock(mutex_);
                state_ = State::starting;
            }
        }
    } catch (const std::exception &e) {
        fail(Status(ErrorCode::internal, std::string("llamaserver: supervisor failure: ") + e.what()));
        return;
    } catch (...) {
        fail(Status(ErrorCode::internal, "llamaserver: supervisor failure"));
        return;
    }
    std::lock_guard lock(mutex_);
    state_ = State::stopped;
    port_ = 0;
    wake_.notify_all();
}

void Supervisor::stop() {
    std::lock_guard stop_lock(stop_mutex_);
    {
        std::lock_guard lock(mutex_);
        shutdown_.cancel();
        wake_.notify_all();
    }
    if (monitor_thread_.joinable())
        monitor_thread_.join();
    std::lock_guard lock(mutex_);
    state_ = State::stopped;
    port_ = 0;
}

bool Supervisor::running() const {
    std::lock_guard lock(mutex_);
    return state_ == State::ready;
}
std::uint16_t Supervisor::port() const {
    std::lock_guard lock(mutex_);
    return port_;
}
Status Supervisor::failure() const {
    std::lock_guard lock(mutex_);
    return failure_;
}

} // namespace sonder::inference::llamaserver
