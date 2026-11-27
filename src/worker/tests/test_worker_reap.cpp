#include "worker/process_reap.hpp"
#include "worker/remote_backend.hpp"

#include <cerrno>
#include <doctest/doctest.h>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
namespace detail = sonder::inference::worker::detail;
namespace worker = sonder::inference::worker;
} // namespace

TEST_CASE("worker: Linux unreapable child is handed off at the monotonic bound") {
    auto clock = Clock::time_point{};
    const auto start = clock;
    int polls = 0, handoffs = 0, handed_pid = -1;
    const bool reaped = detail::reap_until_deadline(
        123, 25ms,
        [&](int pid) {
            CHECK(pid == 123);
            ++polls;
            return false;
        }, // WNOHANG: still in D state
        [&](int pid) {
            ++handoffs;
            handed_pid = pid;
        },
        [&] { return clock; }, [&](auto wait) { clock += wait; });
    CHECK_FALSE(reaped);
    CHECK(clock - start == 25ms);
    CHECK(polls == 4);
    CHECK(handoffs == 1);
    CHECK(handed_pid == 123);
}

#if defined(__linux__)
TEST_CASE("worker: Linux normal child exit is reaped without deferred ownership") {
    const pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0)
        _exit(0);
    pid_t observed = 0;
    bool deferred = false;
    const bool reaped = detail::reap_until_deadline(
        child, 2000ms,
        [&](int pid) {
            observed = waitpid(pid, nullptr, WNOHANG);
            return observed == pid || (observed < 0 && errno == ECHILD);
        },
        [&](int pid) {
            deferred = true;
            // Failure cleanup stays off the test's waiting path as well.
            (void)kill(pid, SIGKILL);
            std::thread([pid] {
                for (;;) {
                    const auto result = waitpid(pid, nullptr, WNOHANG);
                    if (result == pid || (result < 0 && errno == ECHILD))
                        return;
                    std::this_thread::sleep_for(10ms);
                }
            }).detach();
        },
        [] { return Clock::now(); }, [](auto wait) { std::this_thread::sleep_for(wait); });
    CHECK(reaped);
    CHECK_FALSE(deferred);
    CHECK(observed == child);
    if (reaped && !deferred) {
        const auto result = waitpid(child, nullptr, WNOHANG);
        const int captured_errno = errno;
        CHECK(result == -1);
        CHECK(captured_errno == ECHILD);
    }
}
#endif

TEST_CASE("worker: deferred reaper leases cap all owned and pending children") {
    bool exits = false;
    std::size_t polls = 0;
    detail::DeferredReaper reaper([&](int) {
        ++polls;
        return exits;
    });
    std::vector<std::shared_ptr<worker::ReapCompletion>> completions;
    for (std::size_t i = 0; i < detail::kMaxWorkerChildren; ++i) {
        auto completion = std::make_shared<worker::ReapCompletion>();
        auto *ticket = reaper.reserve(completion);
        REQUIRE(ticket != nullptr);
        ticket->pid = static_cast<int>(i + 1);
        completions.push_back(completion);
        // Mix owned and pending leases; the cap applies to their sum.
        if (i == 0) {
            CHECK(reaper.leased() == 1);
            reaper.sweep();
            CHECK(polls == 0);
        }
        reaper.adopt(ticket);
    }
    for (int attempt = 0; attempt < 256; ++attempt) {
        CHECK(reaper.reserve(std::make_shared<worker::ReapCompletion>()) == nullptr);
        const auto before = polls;
        reaper.sweep();
        CHECK(polls - before == detail::kMaxWorkerChildren);
        CHECK(reaper.pending() == detail::kMaxWorkerChildren);
        CHECK(reaper.leased() == detail::kMaxWorkerChildren);
    }
    for (const auto &completion : completions)
        CHECK_FALSE(completion->reaped.load());
    exits = true;
    reaper.sweep();
    CHECK(reaper.pending() == 0);
    CHECK(reaper.leased() == 0);
    for (const auto &completion : completions)
        CHECK(completion->reaped.load());
    // Reuse a slot without resetting a prior incarnation's completion.
    auto replacement = std::make_shared<worker::ReapCompletion>();
    auto *ticket = reaper.reserve(replacement);
    REQUIRE(ticket != nullptr);
    CHECK_FALSE(replacement->reaped.load());
    CHECK(completions.front()->reaped.load());
    reaper.release(ticket); // fork failure releases a reservation too
    CHECK(reaper.leased() == 0);
}

TEST_CASE("worker: restart admission waits for each deferred predecessor "
          "across rolling hours") {
    bool exits = false;
    detail::DeferredReaper reaper([&](int) { return exits; });
    worker::RestartPolicy policy(5, 1ms, 1ms);
    auto now = Clock::time_point{};
    std::size_t replacements = 0;
    for (int generation = 0; generation < 8; ++generation) {
        exits = false;
        auto completion = std::make_shared<worker::ReapCompletion>();
        auto *ticket = reaper.reserve(completion);
        REQUIRE(ticket != nullptr);
        ticket->pid = generation + 1;
        policy.predecessor() = completion; // same output used by spawn_process
        auto clock = Clock::time_point{};
        CHECK_FALSE(reaper.reap_or_defer(
            ticket, 25ms, [](int) { return false; }, [&] { return clock; },
            [&](auto wait) { clock += wait; }));
        CHECK(ticket == nullptr);
        for (int attempt = 0; attempt < 32; ++attempt) {
            now += 1h;
            reaper.sweep();
            CHECK(policy.waiting_for_reap());
            CHECK_FALSE(policy.permit_restart(now));
            CHECK(policy.recent_restarts(now) == 0);
            CHECK(reaper.pending() == 1);
            CHECK(reaper.leased() == 1);
        }
        CHECK(replacements == static_cast<std::size_t>(generation));
        exits = true;
        reaper.sweep();
        REQUIRE_FALSE(policy.waiting_for_reap());
        REQUIRE(policy.permit_restart(now));
        ++replacements;
        CHECK(policy.recent_restarts(now) == 1);
        CHECK(reaper.pending() == 0);
        CHECK(reaper.leased() == 0);
    }
    CHECK(replacements == 8);
}

TEST_CASE("worker: interrupted child observations stop retrying at the deadline") {
    auto now = Clock::time_point{};
    const auto deadline = now + 10ms;
    int calls = 0;
    const auto result = detail::retry_interrupted(
        deadline,
        [&] {
            ++calls;
            now += 1ms;
            return std::pair{-1, EINTR};
        },
        [&] { return now; });
    CHECK(result.first == -1);
    CHECK(result.second == EINTR);
    CHECK(calls == 10);
    CHECK(now == deadline);
}

#if defined(__linux__)
TEST_CASE("worker: production deferred sweep reaps a real child and releases "
          "its lease") {
    auto reaper = std::make_shared<detail::DeferredReaper>(detail::terminate_and_poll);
    auto completion = std::make_shared<worker::ReapCompletion>();
    auto *ticket = reaper->reserve(completion);
    REQUIRE(ticket != nullptr);
    const pid_t child = fork();
    if (child < 0)
        reaper->release(ticket);
    REQUIRE(child >= 0);
    if (child == 0)
        _exit(0);
    ticket->pid = child;
    // A zero-budget forced handoff executes the production adoption/sweep
    // path even if the child exits immediately after fork.
    CHECK_FALSE(reaper->reap_or_defer(
        ticket, 0ms, [](int) { return false; }, [] { return Clock::now(); }, [](auto) {}));
    CHECK(ticket == nullptr);
    const auto deadline = Clock::now() + 2s;
    while (!completion->reaped.load() && Clock::now() < deadline) {
        reaper->sweep();
        std::this_thread::sleep_for(1ms);
    }
    const bool reaped_in_time = completion->reaped.load();
    if (!reaped_in_time) {
        (void)kill(child, SIGKILL);
        // Failure cleanup retains the pool off the test's bounded wait path.
        std::thread([reaper, completion] {
            while (!completion->reaped.load()) {
                reaper->sweep();
                std::this_thread::sleep_for(10ms);
            }
        }).detach();
    }
    REQUIRE(reaped_in_time);
    CHECK(reaper->pending() == 0);
    CHECK(reaper->leased() == 0);
    const auto result = waitpid(child, nullptr, WNOHANG);
    const int captured_errno = errno;
    CHECK(result == -1);
    CHECK(captured_errno == ECHILD);
    CHECK(detail::poll_reaped(child)); // ECHILD is a completed disposition too
}
#endif
