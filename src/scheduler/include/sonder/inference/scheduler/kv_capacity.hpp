// Abstract KV-capacity interface used for admission control.
//
// The scheduler never touches the cache implementation (src/cache/). The
// engine adapts the real KV block manager to this interface; tests use
// FixedKvCapacity.
#pragma once

#include "sonder/inference/scheduler/types.hpp"

#include <unordered_map>

namespace sonder::inference::scheduler {

class KvCapacity {
public:
    virtual ~KvCapacity() = default;

    /// Tokens stored per logical KV block.
    [[nodiscard]] virtual TokenCount block_size_tokens() const noexcept = 0;
    /// Total blocks the scheduler may ever use (for "never fits" checks).
    [[nodiscard]] virtual BlockCount total_blocks() const noexcept = 0;
    /// Blocks currently free for new reservations.
    [[nodiscard]] virtual BlockCount free_blocks() const noexcept = 0;
    /// Reserve `blocks` additional blocks for `id`. Must be all-or-nothing.
    [[nodiscard]] virtual bool try_reserve(RequestId id, BlockCount blocks) = 0;
    /// Release every block held by `id` (no-op if none).
    virtual void release(RequestId id) = 0;
    /// Blocks currently held by `id`.
    [[nodiscard]] virtual BlockCount blocks_held(RequestId id) const noexcept = 0;
};

/// Blocks needed to hold `tokens` tokens.
[[nodiscard]] constexpr BlockCount blocks_for_tokens(TokenCount tokens,
                                                     TokenCount block_size) noexcept {
    return block_size == 0 ? 0
                           : (static_cast<BlockCount>(tokens) + block_size - 1) / block_size;
}

/// Simple fixed-size pool; reference implementation for tests/simulation.
class FixedKvCapacity final : public KvCapacity {
public:
    FixedKvCapacity(BlockCount total_blocks, TokenCount block_size_tokens) noexcept
        : total_(total_blocks), block_size_(block_size_tokens) {}

    [[nodiscard]] TokenCount block_size_tokens() const noexcept override { return block_size_; }
    [[nodiscard]] BlockCount total_blocks() const noexcept override { return total_; }
    [[nodiscard]] BlockCount free_blocks() const noexcept override { return total_ - used_; }
    [[nodiscard]] bool try_reserve(RequestId id, BlockCount blocks) override;
    void release(RequestId id) override;
    [[nodiscard]] BlockCount blocks_held(RequestId id) const noexcept override;

    [[nodiscard]] BlockCount used_blocks() const noexcept { return used_; }
    [[nodiscard]] BlockCount peak_used_blocks() const noexcept { return peak_; }

private:
    BlockCount total_;
    TokenCount block_size_;
    BlockCount used_ = 0;
    BlockCount peak_ = 0;
    std::unordered_map<RequestId, BlockCount> held_;
};

}  // namespace sonder::inference::scheduler
