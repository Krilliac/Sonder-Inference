#include "slot_affinity.hpp"

#include <algorithm>
#include <utility>

namespace sonder::inference::llamaserver {

SlotAffinity::Lease &SlotAffinity::Lease::operator=(Lease &&other) noexcept {
    if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        slot_ = std::exchange(other.slot_, std::nullopt);
        generation_ = std::exchange(other.generation_, std::uint64_t{0});
        outcome_ = std::exchange(other.outcome_, std::string_view{"none"});
    }
    return *this;
}

void SlotAffinity::Lease::reset() noexcept {
    if (owner_ && slot_)
        owner_->release(*slot_, generation_);
    owner_ = nullptr;
    slot_.reset();
    generation_ = 0;
    outcome_ = "none";
}

void SlotAffinity::release(std::uint32_t slot, std::uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    // A lease from before a slot-count change refers to the old layout.
    if (generation != generation_ || slot >= slots_.size())
        return;
    if (slots_[slot].busy > 0)
        --slots_[slot].busy;
}

void SlotAffinity::reset_slots(std::uint32_t n_slots) {
    slots_.assign(n_slots, Slot{});
    lru_.clear();
    by_key_.clear();
    ++generation_;
}

void SlotAffinity::assign(const std::string &key, std::uint32_t slot) {
    Slot &s = slots_[slot];
    if (s.owned) {
        by_key_.erase(*s.owner);
        lru_.erase(s.owner);
    }
    lru_.push_front(key);
    s.owned = true;
    s.owner = lru_.begin();
    by_key_[key] = Entry{slot, lru_.begin()};
}

SlotAffinity::Lease SlotAffinity::acquire(std::string_view key_view, std::uint32_t n_slots,
                                          const std::vector<std::uint32_t> &preferred_slots) {
    Lease lease;
    if (key_view.empty() || n_slots == 0)
        return lease;
    n_slots = std::min(n_slots, kMaxSlots);
    const std::string key(key_view);
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_.size() != n_slots)
        reset_slots(n_slots);
    if (n_slots == 1) {
        lease.outcome_ = "single_slot";
        return lease;
    }
    std::optional<std::uint32_t> chosen;
    if (auto it = by_key_.find(key); it != by_key_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second.lru);
        if (slots_[it->second.slot].busy != 0) {
            lease.outcome_ = "busy_unpinned";
            return lease;
        }
        chosen = it->second.slot;
        lease.outcome_ = "hit";
    } else {
        // Warmed slots are preferred for a new session, because using one
        // avoids immediately discarding the prefix populated by warm-up.
        // Keep this advisory: malformed/out-of-range entries and owned slots
        // are ignored, then the historical lowest-unowned policy applies.
        for (const auto slot : preferred_slots) {
            if (slot < n_slots && !slots_[slot].owned && slots_[slot].busy == 0) {
                chosen = slot;
                break;
            }
        }
        for (std::uint32_t i = 0; i < n_slots && !chosen; ++i) {
            if (!slots_[i].owned && slots_[i].busy == 0)
                chosen = i;
        }
        // Least recently used owner whose slot is idle.
        for (auto lru = lru_.rbegin(); lru != lru_.rend() && !chosen; ++lru) {
            const std::uint32_t slot = by_key_.at(*lru).slot;
            if (slots_[slot].busy == 0)
                chosen = slot;
        }
        if (!chosen) {
            lease.outcome_ = "busy_unpinned";
            return lease;  // every slot is owned and busy: let the upstream pick
        }
        lease.outcome_ = slots_[*chosen].owned ? "stolen" : "new";
        assign(key, *chosen);
    }
    ++slots_[*chosen].busy;
    lease.owner_ = this;
    lease.slot_ = chosen;
    lease.generation_ = generation_;
    return lease;
}

std::optional<std::uint32_t> SlotAffinity::owned_slot(std::string_view key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = by_key_.find(std::string(key));
    if (it == by_key_.end())
        return std::nullopt;
    return it->second.slot;
}

std::size_t SlotAffinity::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return by_key_.size();
}

} // namespace sonder::inference::llamaserver
