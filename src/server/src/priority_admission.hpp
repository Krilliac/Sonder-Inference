// Internal request admission. Permits cover the entire backend request, also
// in account mode where the engine releases its logical scheduling grant early.
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <memory>

#include "sonder/inference/request_priority.hpp"

namespace sonder::inference::server::detail {

class PriorityAdmission {
public:
    using Clock = std::chrono::steady_clock;
    enum class Outcome { admitted, full, cancelled, expired };
    struct Result {
        Outcome outcome = Outcome::cancelled;
        double queue_ms = 0.0;
    };

    // A ticket stays queued until BOTH logical KV and a backend slot are
    // available. The scheduler calls try_acquire only after reserving KV;
    // off/account-bypass requests call wait directly.
    class Ticket final : public RequestAdmission {
    public:
        ~Ticket() override;
        [[nodiscard]] bool ready() const override;
        [[nodiscard]] std::uint64_t order() const noexcept override { return order_; }
        bool try_acquire() override;
        Status wait(const CancellationToken& cancel) override;
        [[nodiscard]] double queue_ms() const override;
        void release();
        [[nodiscard]] Outcome outcome() const;

    private:
        friend class PriorityAdmission;
        Ticket(PriorityAdmission& owner, RequestPriority priority, std::function<bool()> cancelled,
               Clock::time_point deadline);
        bool try_locked();
        bool ready_locked() const;
        PriorityAdmission& owner_;
        std::size_t cls_;
        std::function<bool()> cancelled_;
        Clock::time_point deadline_;
        Clock::time_point started_ = Clock::now();
        double elapsed_ms_ = 0.0;
        Outcome outcome_ = Outcome::cancelled;
        bool queued_ = false;
        bool held_ = false;
        std::uint64_t order_ = 0;
    };

    // Zero means unlimited. Configure before accepting inference requests.
    void configure(std::size_t capacity, std::size_t subagent, std::size_t background,
                   std::size_t queue_per_class);
    std::shared_ptr<Ticket> enqueue(RequestPriority priority, std::function<bool()> cancelled,
                                    Clock::time_point deadline = Clock::time_point::max());
    Result acquire(RequestPriority priority, const std::function<bool()>& cancelled,
                   Clock::time_point deadline = Clock::time_point::max());
    void release(RequestPriority priority);
    [[nodiscard]] std::array<std::size_t, 3> queued() const;

private:
    bool eligible(std::size_t cls) const;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t capacity_ = 0;
    std::size_t queue_cap_ = 0;
    std::size_t total_running_ = 0;
    std::uint64_t next_order_ = 0;
    std::array<std::size_t, 3> caps_{};
    std::array<std::size_t, 3> running_{};
    // Pointers identify live tickets; release/destruction removes the entry.
    std::array<std::deque<const void*>, 3> waiting_;
};

}  // namespace sonder::inference::server::detail
