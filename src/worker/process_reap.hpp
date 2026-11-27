#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <functional>
#include <memory>
#include <utility>

#include "process.hpp"

#if defined(__linux__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace sonder::inference::worker::detail {

inline constexpr std::chrono::milliseconds kForcedReapBudget{250};
inline constexpr std::size_t kMaxWorkerChildren = 64;

// Internal injectable wait/reap seam. poll must be nonblocking (WNOHANG in
// production); defer transfers ownership without waiting for the child.
template <class Poll, class Defer, class Now, class Pause>
bool reap_until_deadline(int pid, std::chrono::milliseconds budget, Poll poll, Defer defer, Now now,
                         Pause pause) {
    using Milliseconds = std::chrono::milliseconds;
    const auto deadline = now() + std::max(budget, Milliseconds{0});
    for (;;) {
        if (poll(pid))
            return true;
        const auto current = now();
        if (current >= deadline) {
            defer(pid);
            return false;
        }
        pause(std::min(Milliseconds{10}, std::chrono::ceil<Milliseconds>(deadline - current)));
    }
}

// A fixed process-wide lease pool covers reservations, live children and
// deferred children together. reserve() runs before fork. Only the owner may
// release an owned slot; only the single sweeper may release a deferred slot.
// Tests inject a nonblocking poll and call sweep(), the production run body.
class DeferredReaper {
  public:
    enum class State { free, owned, deferred };
    struct Ticket {
        std::atomic<State> state{State::free};
        int pid = -1;
        std::shared_ptr<ReapCompletion> completion;
    };
    using Poll = std::function<bool(int)>;
    explicit DeferredReaper(Poll poll) : poll_(std::move(poll)) {}
    Ticket *reserve(const std::shared_ptr<ReapCompletion> &completion) noexcept {
        for (auto &ticket : tickets_) {
            auto expected = State::free;
            if (ticket.state.compare_exchange_strong(expected, State::owned)) {
                ticket.completion = completion;
                return &ticket;
            }
        }
        return nullptr;
    }
    // No allocation or mutex acquisition. Publishing the slot transfers
    // ownership atomically; callers must not access it after this call.
    void adopt(Ticket *ticket) noexcept { ticket->state.store(State::deferred); }
    void release(Ticket *ticket) noexcept {
        auto completion = std::move(ticket->completion);
        ticket->pid = -1;
        ticket->state.store(State::free);
        // Make the lease available before allowing a replacement to spawn.
        completion->reaped.store(true);
    }
    template <class ChildPoll, class Now, class Pause>
    bool reap_or_defer(Ticket *&ticket, std::chrono::milliseconds budget, ChildPoll poll, Now now,
                       Pause pause) {
        const bool reaped = reap_until_deadline(
            ticket->pid, budget, poll, [&](int) { adopt(std::exchange(ticket, nullptr)); }, now, pause);
        if (reaped)
            release(std::exchange(ticket, nullptr));
        return reaped;
    }
    void sweep() {
        for (auto &ticket : tickets_)
            if (ticket.state.load() == State::deferred && poll_(ticket.pid))
                release(&ticket);
    }
    [[nodiscard]] std::size_t pending() const noexcept { return count(State::deferred); }
    [[nodiscard]] std::size_t leased() const noexcept { return tickets_.size() - count(State::free); }

  private:
    std::size_t count(State state) const noexcept {
        std::size_t total = 0;
        for (const auto &ticket : tickets_)
            total += ticket.state.load() == state ? 1u : 0u;
        return total;
    }
    Poll poll_;
    std::array<Ticket, kMaxWorkerChildren> tickets_{};
};

// Operation returns {return code, captured errno}. Repeated EINTR must not
// turn a nominally nonblocking waitid observation into an unbounded loop.
template <class Operation, class Now>
auto retry_interrupted(std::chrono::steady_clock::time_point deadline, Operation operation, Now now) {
    auto result = operation();
    while (result.first < 0 && result.second == EINTR && now() < deadline)
        result = operation();
    return result;
}

#if defined(__linux__)
inline bool poll_reaped(int pid) {
    const auto result = waitpid(pid, nullptr, WNOHANG);
    const int error = errno;
    return result == pid || (result < 0 && error == ECHILD);
}
inline bool terminate_and_poll(int pid) {
    // The foreground ownership check can exhaust its EINTR budget before
    // signalling. Retry that work in bounded, single-shot sweeps rather than
    // leaving an otherwise live child permanently occupying a lease.
    siginfo_t info{};
    if (waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0) {
        (void)kill(-pid, SIGKILL);
        (void)kill(pid, SIGKILL);
    }
    return poll_reaped(pid);
}
#endif

} // namespace sonder::inference::worker::detail
