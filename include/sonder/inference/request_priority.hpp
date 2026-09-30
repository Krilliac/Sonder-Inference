// Request priority classes used by hosted schedulers.
#pragma once

#include <memory>
#include <cstdint>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference {

enum class RequestPriority { interactive, subagent, background };

inline const char* to_string(RequestPriority priority) noexcept {
    switch (priority) {
        case RequestPriority::interactive: return "interactive";
        case RequestPriority::subagent: return "subagent";
        case RequestPriority::background: return "background";
    }
    return "unknown";
}

// Optional host admission ticket. The host owns its lifetime and releases the
// permit after the session call returns. Implementations must not block in
// ready()/try_acquire(): these run on the engine's scheduler coordinator.
class RequestAdmission {
public:
    virtual ~RequestAdmission() = default;
    // Non-reserving eligibility check, followed by try_acquire after logical
    // capacity is reserved. Both must be cheap and safe on the coordinator.
    [[nodiscard]] virtual bool ready() const = 0;
    [[nodiscard]] virtual std::uint64_t order() const noexcept = 0;
    virtual bool try_acquire() = 0;
    virtual Status wait(const CancellationToken& cancel) = 0;
    [[nodiscard]] virtual double queue_ms() const = 0;
};

}  // namespace sonder::inference
