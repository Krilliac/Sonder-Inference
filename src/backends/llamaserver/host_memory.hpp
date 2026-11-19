#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "sonder/inference/device.hpp"
#include "sonder/inference/error.hpp"

// Host RAM admission and floor guard for very large (mostly MoE) models whose
// experts are kept in system memory or mmapped from disk. Pure policy: no
// threads, no I/O and no clock reads of its own, so every decision is testable
// with a fake HostMemory and fake time points. The supervisor owns the single
// monitor thread that calls it.
namespace sonder::inference::llamaserver {

inline constexpr std::uint64_t kHostMiB = 1ull << 20;
inline constexpr std::uint64_t kHostGiB = 1ull << 30;

// What the runtime guard does once available physical memory stays below the
// floor.
enum class HostFloorPolicy {
    warn,                 // warning only; requests keep being admitted
    stop_admission,       // warning, and acquire_request() refuses new requests until memory recovers
    restart_smaller_ctx   // as stop_admission, then relaunch the child with a smaller --ctx-size once idle
};
const char *to_string(HostFloorPolicy policy) noexcept;

struct HostMemoryAdmissionOptions {
    // Estimated host-RAM resident bytes of the child (VramEstimate::
    // host_resident_bytes). 0 means no estimate was supplied: the pre-spawn
    // check is skipped.
    std::uint64_t estimated_resident_bytes = 0;
    // Estimated private (committed) bytes, VramEstimate::host_private_bytes.
    // Compared with available commit when the platform reports it. 0 skips.
    std::uint64_t estimated_commit_bytes = 0;
    // Kept free for the OS and other processes on top of the estimate.
    std::uint64_t reserve_bytes = 2 * kHostGiB;
    // Start anyway when the check would refuse (a warning is recorded).
    bool override_refusal = false;
    // The estimate is a lower bound (llama-server --fit decides the offload and
    // no VRAM budget is known): the supervisor raises a warning.
    bool estimate_lower_bound = false;
};

struct HostMemoryFloorOptions {
    bool enabled = false;
    HostFloorPolicy policy = HostFloorPolicy::warn;
    // Pressure starts when available physical memory is below this for
    // `consecutive_samples` samples in a row (a known 0 counts).
    std::uint64_t floor_bytes = 0;
    // Pressure ends when available memory is at or above this for
    // `consecutive_samples` samples in a row. 0 = floor + floor / 4.
    std::uint64_t resume_bytes = 0;
    std::size_t consecutive_samples = 2;
    std::chrono::milliseconds sample_interval{5000};
    // restart_smaller_ctx: minimum time between restarts and their lifetime cap.
    std::chrono::milliseconds restart_cooldown{60000};
    std::size_t max_restarts = 1;
};

struct HostMemoryOptions {
    HostMemoryAdmissionOptions admission;
    HostMemoryFloorOptions floor;
    // Null uses read_host_memory(). Fake-only seams for tests; production
    // leaves both null.
    HostMemoryProvider provider;
    std::function<std::chrono::steady_clock::time_point()> clock;
};

Status validate_host_memory_options(const HostMemoryOptions &options);

// resume_bytes with its default applied (saturating).
std::uint64_t effective_resume_bytes(const HostMemoryFloorOptions &options) noexcept;

struct HostAdmissionVerdict {
    enum class Outcome {
        skipped,           // no estimate supplied
        admitted,          // enough physical memory (and commit)
        admitted_unknown,  // availability unknown: admitted, documented fallback
        admitted_override, // would refuse; override_refusal set
        refused
    };
    Outcome outcome = Outcome::skipped;
    std::uint64_t required_bytes = 0;   // estimate + reserve (saturating)
    std::uint64_t available_bytes = 0;  // 0 unless a reading was used
    // Refusal text (or the override/unknown warning text); empty otherwise.
    // It names the estimate, the reserve and the available memory in MiB.
    std::string message;
    [[nodiscard]] bool admit() const noexcept { return outcome != Outcome::refused; }
};

// Compares estimate + reserve with available physical memory, and the private
// estimate + reserve with available commit when both are known.
//   known zero available   refuse (never "unknown")
//   unknown available      admit, outcome admitted_unknown (the fallback: a
//                          probe that cannot measure must not block a start)
HostAdmissionVerdict check_host_admission(const HostMemoryAdmissionOptions &options, const HostMemory &memory);

class HostMemoryFloorGuard {
  public:
    enum class State { ok, pressure };
    struct Update {
        bool entered_pressure = false;
        bool left_pressure = false;
        bool restart_requested = false;  // restart_smaller_ctx only; once per request
        bool unknown_sample = false;
    };

    HostMemoryFloorGuard() = default;
    explicit HostMemoryFloorGuard(HostMemoryFloorOptions options) : options_(options) {}

    // True when the guard is enabled and `now` is at or past the next sample time.
    [[nodiscard]] bool due(std::chrono::steady_clock::time_point now) const noexcept {
        return options_.enabled && now >= next_sample_;
    }
    // Feeds one reading and schedules the next sample at now + sample_interval.
    // An unknown reading changes neither streak nor state.
    Update observe(const MemoryReading &available, std::chrono::steady_clock::time_point now);
    // A new child starts clean: state ok, streaks cleared, sampling due at
    // once. The restart budget is lifetime and is kept.
    void reset(std::chrono::steady_clock::time_point now) noexcept;

    [[nodiscard]] State state() const noexcept { return state_; }
    // False only while in pressure under stop_admission / restart_smaller_ctx.
    [[nodiscard]] bool admitting() const noexcept {
        return !(state_ == State::pressure && options_.policy != HostFloorPolicy::warn);
    }
    [[nodiscard]] bool restart_pending() const noexcept { return restart_pending_; }
    // The supervisor relaunched the child (or gave up): clears the pending request.
    void restart_performed(std::chrono::steady_clock::time_point now) noexcept;
    void restart_abandoned() noexcept { restart_pending_ = false; }
    [[nodiscard]] std::size_t restarts() const noexcept { return restarts_; }
    [[nodiscard]] std::uint64_t last_available_bytes() const noexcept { return last_available_; }
    [[nodiscard]] std::size_t unknown_samples() const noexcept { return unknown_samples_; }

  private:
    HostMemoryFloorOptions options_;
    State state_ = State::ok;
    std::size_t low_streak_ = 0;
    std::size_t recovered_streak_ = 0;
    std::size_t unknown_samples_ = 0;
    std::size_t restarts_ = 0;
    bool restart_pending_ = false;
    bool restarted_before_ = false;
    std::uint64_t last_available_ = 0;
    std::chrono::steady_clock::time_point next_sample_{};
    std::chrono::steady_clock::time_point last_restart_{};
};

} // namespace sonder::inference::llamaserver
