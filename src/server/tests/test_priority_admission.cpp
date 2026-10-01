// Admission is independent of GPU/model execution and keeps account-mode
// requests resident until the whole upstream call ends.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "src/priority_admission.hpp"

namespace {
using sonder::inference::RequestPriority;
using sonder::inference::server::detail::PriorityAdmission;
using Outcome = PriorityAdmission::Outcome;
using namespace std::chrono_literals;

template <class Predicate>
bool wait_for(Predicate pred) {
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (!pred() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
    return pred();
}
}

TEST_CASE("priority admission: unlimited defaults do not queue or cap any class") {
    PriorityAdmission admission;
    admission.configure(0, 0, 0, 0);
    for (int n = 0; n < 100; ++n) {
        for (const auto cls : {RequestPriority::interactive, RequestPriority::subagent, RequestPriority::background}) {
            const auto r = admission.acquire(cls, [] { return false; });
            CHECK(r.outcome == Outcome::admitted);
            CHECK(r.queue_ms >= 0.0);
        }
    }
    CHECK(admission.queued() == std::array<std::size_t, 3>{0, 0, 0});
    for (int n = 0; n < 100; ++n) {
        for (const auto cls : {RequestPriority::interactive, RequestPriority::subagent, RequestPriority::background})
            admission.release(cls);
    }
}

TEST_CASE("priority admission: strict classes, FIFO and slot lifetime") {
    PriorityAdmission admission;
    admission.configure(1, 0, 0, 0);
    REQUIRE(admission.acquire(RequestPriority::background, [] { return false; }).outcome == Outcome::admitted);
    std::atomic<bool> cancel{false};
    const auto enqueue = [&](RequestPriority cls) {
        return std::async(std::launch::async, [&, cls] {
            return admission.acquire(cls, [&] { return cancel.load(); }, PriorityAdmission::Clock::now() + 3s);
        });
    };
    auto bg = enqueue(RequestPriority::background);
    CHECK(wait_for([&] { return admission.queued()[2] == 1; }));
    auto sub1 = enqueue(RequestPriority::subagent);
    CHECK(wait_for([&] { return admission.queued()[1] == 1; }));
    auto sub2 = enqueue(RequestPriority::subagent);
    CHECK(wait_for([&] { return admission.queued()[1] == 2; }));
    auto chat = enqueue(RequestPriority::interactive);
    CHECK(wait_for([&] { return admission.queued()[0] == 1; }));
    admission.release(RequestPriority::background);
    const auto chat_result = chat.get();
    CHECK(chat_result.outcome == Outcome::admitted);
    CHECK(chat_result.queue_ms >= 0.0);
    CHECK(sub1.wait_for(0ms) == std::future_status::timeout);
    CHECK(bg.wait_for(0ms) == std::future_status::timeout);
    if (chat_result.outcome == Outcome::admitted) admission.release(RequestPriority::interactive);
    const auto first = sub1.get();
    CHECK(first.outcome == Outcome::admitted);
    CHECK(sub2.wait_for(0ms) == std::future_status::timeout);
    if (first.outcome == Outcome::admitted) admission.release(RequestPriority::subagent);
    const auto second = sub2.get();
    CHECK(second.outcome == Outcome::admitted);
    if (second.outcome == Outcome::admitted) admission.release(RequestPriority::subagent);
    const auto last = bg.get();
    CHECK(last.outcome == Outcome::admitted);
    if (last.outcome == Outcome::admitted) admission.release(RequestPriority::background);
    CHECK(admission.queued() == std::array<std::size_t, 3>{0, 0, 0});
}

TEST_CASE("priority admission: caps count running requests, queue cap counts only waiting requests") {
    PriorityAdmission admission;
    admission.configure(0, 1, 1, 1);
    REQUIRE(admission.acquire(RequestPriority::subagent, [] { return false; }).outcome == Outcome::admitted);
    std::atomic<bool> cancel{false};
    auto pending = std::async(std::launch::async, [&] {
        return admission.acquire(RequestPriority::subagent, [&] { return cancel.load(); },
                                 PriorityAdmission::Clock::now() + 3s);
    });
    CHECK(wait_for([&] { return admission.queued()[1] == 1; }));
    CHECK(admission.acquire(RequestPriority::subagent, [] { return false; }).outcome == Outcome::full);
    // A capped subagent must not block an eligible lower class or interactive.
    CHECK(admission.acquire(RequestPriority::background, [] { return false; }).outcome == Outcome::admitted);
    CHECK(admission.acquire(RequestPriority::interactive, [] { return false; }).outcome == Outcome::admitted);
    admission.release(RequestPriority::background);
    admission.release(RequestPriority::interactive);
    cancel.store(true);
    CHECK(pending.get().outcome == Outcome::cancelled);
    CHECK(admission.queued()[1] == 0);
    admission.release(RequestPriority::subagent);
}

TEST_CASE("priority admission: deadlines and cancellation remove waiters without consuming capacity") {
    PriorityAdmission admission;
    admission.configure(1, 0, 0, 1);
    REQUIRE(admission.acquire(RequestPriority::background, [] { return false; }).outcome == Outcome::admitted);
    const auto expired = admission.acquire(RequestPriority::interactive, [] { return false; },
                                           PriorityAdmission::Clock::now() + 30ms);
    CHECK(expired.outcome == Outcome::expired);
    CHECK(expired.queue_ms > 0.0);
    CHECK(admission.queued()[0] == 0);
    CHECK(admission.acquire(RequestPriority::subagent, [] { return true; }).outcome == Outcome::cancelled);
    admission.release(RequestPriority::background);
    CHECK(admission.acquire(RequestPriority::interactive, [] { return false; },
                            PriorityAdmission::Clock::now() - 1ms).outcome == Outcome::expired);
    CHECK(admission.acquire(RequestPriority::interactive, [] { return false; }).outcome == Outcome::admitted);
    admission.release(RequestPriority::interactive);
}

TEST_CASE("priority tickets keep their FIFO position while another class is capped") {
    PriorityAdmission admission;
    admission.configure(4, 1, 0, 0);
    auto running = admission.enqueue(RequestPriority::subagent, [] { return false; });
    REQUIRE(running->try_acquire());
    auto capped = admission.enqueue(RequestPriority::subagent, [] { return false; });
    auto background = admission.enqueue(RequestPriority::background, [] { return false; });
    CHECK_FALSE(capped->ready());
    CHECK(background->ready());
    CHECK(background->try_acquire());
    auto first = admission.enqueue(RequestPriority::interactive, [] { return false; });
    auto second = admission.enqueue(RequestPriority::interactive, [] { return false; });
    CHECK(first->order() < second->order());
    CHECK_FALSE(second->ready());
    CHECK(first->try_acquire());
    CHECK(second->try_acquire());
}
