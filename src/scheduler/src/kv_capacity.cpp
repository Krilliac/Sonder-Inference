#include "sonder/inference/scheduler/kv_capacity.hpp"

#include <algorithm>

namespace sonder::inference::scheduler {

bool FixedKvCapacity::try_reserve(RequestId id, BlockCount blocks) {
    if (blocks == 0) return true;
    if (blocks > total_ - used_) return false;
    used_ += blocks;
    peak_ = std::max(peak_, used_);
    held_[id] += blocks;
    return true;
}

void FixedKvCapacity::release(RequestId id) {
    auto it = held_.find(id);
    if (it == held_.end()) return;
    used_ -= it->second;
    held_.erase(it);
}

BlockCount FixedKvCapacity::blocks_held(RequestId id) const noexcept {
    auto it = held_.find(id);
    return it == held_.end() ? 0 : it->second;
}

}  // namespace sonder::inference::scheduler
