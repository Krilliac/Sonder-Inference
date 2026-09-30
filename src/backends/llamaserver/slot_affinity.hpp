// Conversation -> llama-server slot affinity (docs/integration/llama-server.md
// "Prompt cache and slot affinity").
//
// llama-server keeps one KV cache per slot and reuses a prompt prefix only
// inside the slot that last processed it. Without `id_slot` it picks an idle
// slot by prompt similarity, which a concurrent conversation can take over.
// SlotAffinity pins each conversation key to one slot so its next turn lands
// on the slot that holds its prefix.
//
// Policy: a known key keeps its slot (even while busy: turns of one
// conversation are sequential, and waiting for its own slot beats a full
// re-read elsewhere). A new key takes the lowest slot no key owns, otherwise
// the slot of the least recently used key whose slot is idle; when every
// slot is owned and busy the request is not pinned (the upstream picks).
// With one slot every keyed request uses slot 0. The map holds at most one
// key per slot, so it is bounded by the slot count (capped at kMaxSlots).
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonder::inference::llamaserver {

class SlotAffinity {
  public:
    static constexpr std::uint32_t kMaxSlots = 1024;

    // Releases its slot's busy mark on destruction. Movable, not copyable.
    class Lease {
      public:
        Lease() = default;
        Lease(Lease &&other) noexcept { *this = std::move(other); }
        Lease &operator=(Lease &&other) noexcept;
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        ~Lease() { reset(); }
        [[nodiscard]] std::optional<std::uint32_t> slot() const noexcept { return slot_; }
        void reset() noexcept;

      private:
        friend class SlotAffinity;
        SlotAffinity *owner_ = nullptr;
        std::optional<std::uint32_t> slot_;
        std::uint64_t generation_ = 0;
    };

    // Slot for `key` among `n_slots` upstream slots. No slot (and no busy
    // mark) for an empty key or an unknown slot count (0).
    // New keys may prefer slots whose prefix was warmed by the backend. The
    // preference is advisory: existing ownership and busy-slot rules still
    // win, and an empty list preserves the historical selection policy.
    [[nodiscard]] Lease acquire(std::string_view key, std::uint32_t n_slots,
                                const std::vector<std::uint32_t> &preferred_slots = {});

    // Test observation: the slot a key currently owns.
    [[nodiscard]] std::optional<std::uint32_t> owned_slot(std::string_view key) const;
    [[nodiscard]] std::size_t keys() const;

  private:
    struct Slot {
        std::uint32_t busy = 0;
        bool owned = false;
        std::list<std::string>::iterator owner;
    };
    void release(std::uint32_t slot, std::uint64_t generation) noexcept;
    void reset_slots(std::uint32_t n_slots);
    void assign(const std::string &key, std::uint32_t slot);

    mutable std::mutex mutex_;
    std::uint64_t generation_ = 1;  // bumped when the slot count changes
    std::vector<Slot> slots_;
    struct Entry {
        std::uint32_t slot = 0;
        std::list<std::string>::iterator lru;
    };
    std::list<std::string> lru_;  // front = most recent
    std::unordered_map<std::string, Entry> by_key_;
};

} // namespace sonder::inference::llamaserver
