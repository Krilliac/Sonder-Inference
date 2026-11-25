// Clock abstraction so scheduling is deterministic and testable without a
// real backend.
#pragma once

#include "sonder/inference/scheduler/types.hpp"

#include <limits>

namespace sonder::inference::scheduler {

class Clock {
public:
    virtual ~Clock() = default;
    [[nodiscard]] virtual TimeUs now() const noexcept = 0;
};

/// Manually advanced clock. Time only moves when the owner calls advance(),
/// so repeated simulations produce bit-identical results.
class SimClock final : public Clock {
public:
    explicit SimClock(TimeUs start = 0) noexcept : now_(start) {}
    [[nodiscard]] TimeUs now() const noexcept override { return now_; }
    void advance(TimeUs delta) noexcept {
        if (delta <= 0) return;
        constexpr TimeUs kMax = std::numeric_limits<TimeUs>::max();
        now_ = now_ > kMax - delta ? kMax : now_ + delta;
    }
    void advance_to(TimeUs t) noexcept {
        if (t > now_) now_ = t;
    }

private:
    TimeUs now_;
};

}  // namespace sonder::inference::scheduler
