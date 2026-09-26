// Sonder Inference: cooperative cancellation.
#pragma once

#include <atomic>
#include <memory>

namespace sonder::inference {

// Read side of a cancellation flag. Cheap to copy and to poll from hot paths
// (one relaxed atomic load). A default-constructed token is never cancelled.
class CancellationToken {
public:
    CancellationToken() = default;

    [[nodiscard]] bool cancelled() const noexcept {
        return flag_ && flag_->load(std::memory_order_acquire);
    }
    [[nodiscard]] bool can_be_cancelled() const noexcept { return static_cast<bool>(flag_); }

private:
    friend class CancellationSource;
    explicit CancellationToken(std::shared_ptr<const std::atomic<bool>> flag) : flag_(std::move(flag)) {}
    std::shared_ptr<const std::atomic<bool>> flag_;
};

// Write side. Cancellation is sticky and idempotent; safe from any thread.
class CancellationSource {
public:
    CancellationSource() : flag_(std::make_shared<std::atomic<bool>>(false)) {}

    void cancel() noexcept { flag_->store(true, std::memory_order_release); }
    [[nodiscard]] bool cancelled() const noexcept { return flag_->load(std::memory_order_acquire); }
    [[nodiscard]] CancellationToken token() const { return CancellationToken(flag_); }

private:
    std::shared_ptr<std::atomic<bool>> flag_;
};

}  // namespace sonder::inference
