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
    : options_(std::move(options)), launcher_(launcher ? std::move(launcher) : make_process_launcher()) {
    arguments_ = options_.arguments;
    auto log_path = find_log_file(arguments_);
    if (!log_path && !options_.log_file.empty()) {
        arguments_.push_back("--log-file");
        arguments_.push_back(options_.log_file);
        log_path = options_.log_file;
    }
    if (log_path)
        log_tail_ = std::make_unique<LogTail>(*log_path);
    if (options_.kv_pairing_check)
        config_warnings_ = check_kv_cache_pairing(arguments_);
    const auto &guard = options_.spill_guard;
    if (guard.enabled) {
        gpu_source_ = options_.gpu_counters ? options_.gpu_counters : make_gpu_counter_source();
        gpu_.probe = gpu_source_->name();
        gpu_.status = gpu_source_->supported() ? "not_sampled" : "unsupported";
        if (!gpu_source_->supported())
            gpu_.error = "per-process GPU memory counters are not available on this platform";
    } else {
        gpu_.probe = "none";
        gpu_.status = "disabled";
    }
    gpu_.shared_baseline_bytes = effective_baseline(guard, find_context_size(options_.arguments));
    gpu_.spill_threshold_bytes = guard.threshold_bytes;
    context_.policy = to_string(guard.policy);
    context_.configured_ctx = find_context_size(options_.arguments);
    context_.fitted_ctx = context_.configured_ctx;
    context_.outcome = "not_needed";
}
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
        if (valid.ok() && options_.spill_guard.enabled)
            valid = validate_spill_guard(options_.spill_guard);
        if (valid.ok() && options_.log_file.find('\0') != std::string::npos)
            valid = Status(ErrorCode::invalid_argument, "llamaserver: log file path contains NUL");
        if (valid.ok() && options_.spill_guard.enabled && options_.spill_guard.policy == SpillPolicy::auto_fit &&
            !context_.configured_ctx)
            valid = Status(ErrorCode::invalid_argument,
                           "llamaserver: spill_guard.policy=auto_fit needs an explicit --ctx-size/-c argument");
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
    auto deadline = Clock::now() + options_.readiness_timeout + options_.restart_max_backoff;
    auto seen_epoch = epoch_;
    for (;;) {
        if (cancel.cancelled() || shutdown_.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: readiness wait cancelled");
        if (epoch_ != seen_epoch) {
            // An auto_fit relaunch loads the model again: allow it a full
            // readiness budget. Bounded by spill_guard.max_attempts.
            seen_epoch = epoch_;
            deadline = Clock::now() + options_.readiness_timeout + options_.restart_max_backoff;
        }
        if (state_ == State::ready)
            return port_;
        if (state_ == State::failed)
            return failure_;
        if (Clock::now() >= deadline)
            return settle_after_deadline(lock, cancel, deadline);
        wake_.wait_until(lock, std::min(deadline, Clock::now() + Milliseconds(10)));
    }
}

Result<std::uint16_t> Supervisor::settle_after_deadline(std::unique_lock<std::mutex> &lock,
                                                        const CancellationToken &cancel,
                                                        Clock::time_point deadline) {
    // The monitor may still be inside a health probe (up to one 250 ms budget)
    // or stopping the child. Give it a bounded grace to publish its outcome so
    // a timed-out start() never returns while the child is still running.
    const auto grace_end = deadline + Milliseconds(250) + options_.shutdown_timeout;
    while (state_ != State::ready && state_ != State::failed && !cancel.cancelled() &&
           !shutdown_.cancelled() && Clock::now() < grace_end)
        wake_.wait_until(lock, std::min(grace_end, Clock::now() + Milliseconds(10)));
    if (state_ == State::ready)
        return port_;
    if (state_ == State::failed)
        return failure_;
    if (cancel.cancelled() || shutdown_.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: readiness wait cancelled");
    return Status(ErrorCode::timeout, "llamaserver: readiness wait timed out");
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
                ProcessSpec spec{options_.executable, launch_arguments(), selected.value(), options_.shutdown_timeout};
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
                    begin_child();
                    result = wait_ready(*child, spec.port);
                    if (result.ok() && !shutdown_.cancelled()) {
                        const auto verdict = guard_after_ready(*child);
                        if (verdict.action == GuardAction::refuse) {
                            child->stop(options_.shutdown_timeout);
                            fail(verdict.status);
                            return;
                        }
                        if (verdict.action == GuardAction::refit) {
                            // Relaunch with a smaller context. This neither uses
                            // the crash-restart budget nor waits for its backoff;
                            // refits are bounded by spill_guard.max_attempts.
                            child->stop(options_.shutdown_timeout);
                            std::lock_guard lock(mutex_);
                            arguments_ = with_context_size(arguments_, verdict.next_ctx);
                            context_.fitted_ctx = verdict.next_ctx;
                            ++context_.fit_attempts;
                            ++epoch_;
                            port_ = 0;
                            state_ = State::starting;
                            wake_.notify_all();
                            continue;
                        }
                        {
                            std::lock_guard lock(mutex_);
                            port_ = spec.port;
                            state_ = State::ready;
                            failure_ = Status::success();
                            wake_.notify_all();
                        }
                        auto next_sample = Clock::now() + options_.spill_guard.sample_interval;
                        while (!shutdown_.cancelled() && child->running()) {
                            pause(options_.health_poll_interval);
                            if (Clock::now() >= next_sample && !shutdown_.cancelled()) {
                                observe(*child);
                                next_sample = Clock::now() + options_.spill_guard.sample_interval;
                            }
                        }
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

std::vector<std::string> Supervisor::launch_arguments() const {
    std::lock_guard lock(mutex_);
    return arguments_;
}

void Supervisor::begin_child() {
    // A new child truncates (or appends to) its log; parse it from the start.
    log_diagnostics_.reset();
    if (log_tail_)
        log_tail_->rewind();
    std::lock_guard lock(mutex_);
    log_warnings_.clear();
    if (gpu_.status == "ok" || gpu_.status == "error") {
        gpu_.status = "not_sampled";
        gpu_.error.clear();
    }
    gpu_.dedicated_bytes = 0;
    gpu_.shared_bytes = 0;
    gpu_.peak_shared_bytes = 0;
    gpu_.spilled = false;
    gpu_.samples = 0;
}

void Supervisor::observe(Process &child) {
    if (log_tail_) {
        // Bounded read per tick. A missing file is not an error: the child may
        // not have created it yet.
        (void)log_tail_->poll(log_diagnostics_);
        std::lock_guard lock(mutex_);
        log_warnings_ = log_diagnostics_.warnings();
    }
    if (!gpu_source_ || !gpu_source_->supported())
        return;
    const auto pid = child.pid();
    auto read = gpu_source_->read(pid);
    std::lock_guard lock(mutex_);
    if (!read.ok()) {
        gpu_.status = read.status().code() == ErrorCode::unsupported ? "unsupported" : "error";
        gpu_.error = read.status().message();
        return;
    }
    const auto sample = summarize_gpu_counters(pid, read.value());
    gpu_.status = "ok";
    gpu_.error.clear();
    gpu_.dedicated_bytes = sample.dedicated_bytes;
    gpu_.shared_bytes = sample.shared_bytes;
    gpu_.peak_shared_bytes = std::max(gpu_.peak_shared_bytes, sample.shared_bytes);
    const auto ctx = find_context_size(arguments_);
    gpu_.shared_baseline_bytes = effective_baseline(options_.spill_guard, ctx);
    gpu_.spilled = is_spilled(sample, options_.spill_guard, ctx);
    ++gpu_.samples;
}

namespace {
std::string mib_text(std::uint64_t bytes) { return std::to_string((bytes + kMiB / 2) / kMiB) + " MiB"; }
} // namespace

Supervisor::GuardVerdict Supervisor::guard_after_ready(Process &child) {
    observe(child);
    GuardVerdict verdict;
    std::lock_guard lock(mutex_);
    const auto &guard = options_.spill_guard;
    // A probe error or an unsupported platform cannot classify the child: it
    // is served (and the error is reported), never refused or refitted.
    if (!guard.enabled || gpu_.status != "ok" || !gpu_.spilled) {
        if (context_.fit_attempts > 0 && gpu_.status == "ok" && !gpu_.spilled)
            context_.outcome = "fitted";
        return verdict;
    }
    if (guard.policy == SpillPolicy::warn)
        return verdict;
    const auto current = find_context_size(arguments_);
    const std::string numbers = "shared GPU memory " + mib_text(gpu_.shared_bytes) + " exceeds the " +
                                mib_text(guard.threshold_bytes) + " spill threshold above a " +
                                mib_text(effective_baseline(guard, current)) + " baseline (dedicated " +
                                mib_text(gpu_.dedicated_bytes) +
                                (current ? ", ctx " + std::to_string(*current) : std::string()) + ")";
    if (guard.policy == SpillPolicy::refuse) {
        context_.outcome = "refused";
        verdict.action = GuardAction::refuse;
        verdict.status = Status(ErrorCode::unavailable,
                                "llamaserver: VRAM spill detected: " + numbers +
                                    "; refusing to serve (spill_guard.policy=refuse). Lower --ctx-size or the KV "
                                    "cache precision, or use policy auto_fit");
        return verdict;
    }
    const auto next = current ? next_fit_context(*current, guard) : std::nullopt;
    if (!next) {
        context_.outcome = "floor_reached";
        verdict.action = GuardAction::refuse;
        verdict.status = Status(ErrorCode::unavailable, "llamaserver: VRAM spill persists at the minimum context " +
                                                            std::to_string(guard.min_ctx) + " after " +
                                                            std::to_string(context_.fit_attempts) +
                                                            " context reduction(s): " + numbers);
        return verdict;
    }
    if (context_.fit_attempts >= guard.max_attempts) {
        context_.outcome = "exhausted";
        verdict.action = GuardAction::refuse;
        verdict.status = Status(ErrorCode::unavailable, "llamaserver: VRAM spill persists after " +
                                                            std::to_string(context_.fit_attempts) +
                                                            " context reduction(s) (spill_guard.max_attempts): " +
                                                            numbers);
        return verdict;
    }
    verdict.action = GuardAction::refit;
    verdict.next_ctx = *next;
    return verdict;
}

BackendRuntimeStatus Supervisor::runtime_status() const {
    std::lock_guard lock(mutex_);
    BackendRuntimeStatus status;
    status.gpu_memory = gpu_;
    status.context = context_;
    status.warnings = config_warnings_;
    status.warnings.insert(status.warnings.end(), log_warnings_.begin(), log_warnings_.end());
    if (gpu_.status == "ok" && gpu_.spilled) {
        std::vector<std::pair<std::string, std::string>> details{
            {"dedicated_bytes", std::to_string(gpu_.dedicated_bytes)},
            {"shared_bytes", std::to_string(gpu_.shared_bytes)},
            {"shared_baseline_bytes", std::to_string(gpu_.shared_baseline_bytes)},
            {"spill_threshold_bytes", std::to_string(gpu_.spill_threshold_bytes)}};
        if (context_.fitted_ctx)
            details.emplace_back("ctx", std::to_string(*context_.fitted_ctx));
        status.warnings.push_back(BackendWarning{
            "vram_spill", "warning", "gpu_probe",
            "llama-server keeps " + mib_text(gpu_.shared_bytes) + " in shared system memory (threshold " +
                mib_text(gpu_.spill_threshold_bytes) + " above a " + mib_text(gpu_.shared_baseline_bytes) +
                " baseline): VRAM overflowed into system RAM (measured spills ran 12% to 15x slower). Lower "
                "--ctx-size or the KV cache precision, or set spill_guard.policy to auto_fit or refuse",
            std::move(details), 1});
    }
    return status;
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
