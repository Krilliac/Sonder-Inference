#include "host_memory.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace sonder::inference::llamaserver {
namespace {

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

// Round up for what is needed, down for what is available, so a message never
// shows equal numbers for a refusal.
std::uint64_t mib_up(std::uint64_t bytes) noexcept { return bytes / kHostMiB + (bytes % kHostMiB != 0 ? 1 : 0); }
std::uint64_t mib_down(std::uint64_t bytes) noexcept { return bytes / kHostMiB; }

std::string need_text(std::uint64_t estimate, std::uint64_t reserve, std::uint64_t required) {
    return std::to_string(mib_up(estimate)) + " MiB estimated host RAM + " + std::to_string(mib_up(reserve)) +
           " MiB reserve = " + std::to_string(mib_up(required)) + " MiB";
}

} // namespace

const char *to_string(HostFloorPolicy policy) noexcept {
    switch (policy) {
    case HostFloorPolicy::warn:
        return "warn";
    case HostFloorPolicy::stop_admission:
        return "stop_admission";
    case HostFloorPolicy::restart_smaller_ctx:
        return "restart_smaller_ctx";
    }
    return "unknown";
}

std::uint64_t effective_resume_bytes(const HostMemoryFloorOptions &options) noexcept {
    if (options.resume_bytes != 0)
        return options.resume_bytes;
    return saturating_add(options.floor_bytes, options.floor_bytes / 4);
}

Status validate_host_memory_options(const HostMemoryOptions &options) {
    const auto bad = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: invalid host_memory options: ") + what);
    };
    const auto &admission = options.admission;
    if (admission.estimated_commit_bytes > admission.estimated_resident_bytes &&
        admission.estimated_resident_bytes != 0)
        return bad("estimated_commit_bytes exceeds estimated_resident_bytes");
    const auto &floor = options.floor;
    if (!floor.enabled)
        return Status::success();
    if (floor.floor_bytes == 0)
        return bad("floor.floor_bytes must be above 0");
    if (floor.resume_bytes != 0 && floor.resume_bytes < floor.floor_bytes)
        return bad("floor.resume_bytes must not be below floor.floor_bytes");
    if (floor.consecutive_samples == 0 || floor.consecutive_samples > 100)
        return bad("floor.consecutive_samples must be 1..100");
    if (floor.sample_interval < std::chrono::milliseconds(10) || floor.sample_interval > std::chrono::hours(1))
        return bad("floor.sample_interval must be 10 ms..1 h");
    if (floor.restart_cooldown < std::chrono::milliseconds(0) || floor.restart_cooldown > std::chrono::hours(24))
        return bad("floor.restart_cooldown must be 0..24 h");
    if (floor.max_restarts > 1000)
        return bad("floor.max_restarts must be at most 1000");
    return Status::success();
}

HostAdmissionVerdict check_host_admission(const HostMemoryAdmissionOptions &options, const HostMemory &memory) {
    HostAdmissionVerdict verdict;
    if (options.estimated_resident_bytes == 0)
        return verdict;  // skipped
    verdict.required_bytes = saturating_add(options.estimated_resident_bytes, options.reserve_bytes);
    const auto &available = memory.available_physical;
    verdict.available_bytes = available.bytes_or(0);
    const bool physical_short = available.is_known() && *available.value() < verdict.required_bytes;

    const std::uint64_t commit_required = saturating_add(options.estimated_commit_bytes, options.reserve_bytes);
    const bool commit_short = options.estimated_commit_bytes != 0 && memory.available_commit.is_known() &&
                              *memory.available_commit.value() < commit_required;

    if (!physical_short && !commit_short) {
        if (!available.is_known()) {
            verdict.outcome = HostAdmissionVerdict::Outcome::admitted_unknown;
            verdict.message = "host memory availability is unknown" +
                              (memory.note.empty() ? std::string() : " (" + memory.note + ")") +
                              "; the pre-spawn check for " +
                              need_text(options.estimated_resident_bytes, options.reserve_bytes, verdict.required_bytes) +
                              " was skipped";
        } else {
            verdict.outcome = HostAdmissionVerdict::Outcome::admitted;
        }
        return verdict;
    }

    std::string text = "the estimated host-RAM need is " +
                       need_text(options.estimated_resident_bytes, options.reserve_bytes, verdict.required_bytes);
    if (physical_short) {
        text += " but only " + std::to_string(mib_down(verdict.available_bytes)) + " MiB of physical memory is available";
        if (!memory.source.empty())
            text += " (" + memory.source + (memory.note.empty() ? std::string() : ": " + memory.note) + ")";
    }
    if (commit_short) {
        text += std::string(physical_short ? "; " : " but ") + "available commit is " +
                std::to_string(mib_down(*memory.available_commit.value())) + " MiB against a " +
                std::to_string(mib_up(commit_required)) + " MiB private estimate + reserve";
    }
    if (options.override_refusal) {
        verdict.outcome = HostAdmissionVerdict::Outcome::admitted_override;
        verdict.message = text + "; starting anyway (host_memory.admission.override_refusal)";
        return verdict;
    }
    verdict.outcome = HostAdmissionVerdict::Outcome::refused;
    verdict.message = "llamaserver: host memory admission refused: " + text +
                      ". Offload more layers, lower --ctx-size, free memory, or set "
                      "host_memory.admission.override_refusal to start anyway";
    return verdict;
}

HostMemoryFloorGuard::Update HostMemoryFloorGuard::observe(const MemoryReading &available,
                                                           std::chrono::steady_clock::time_point now) {
    Update update;
    next_sample_ = now + options_.sample_interval;
    if (!available.is_known()) {
        // Neither enters nor leaves pressure: an unmeasurable host is not low
        // and not recovered.
        ++unknown_samples_;
        update.unknown_sample = true;
        return update;
    }
    const std::uint64_t value = *available.value();
    last_available_ = value;
    const std::size_t needed = std::max<std::size_t>(options_.consecutive_samples, 1);
    if (state_ == State::ok) {
        if (value < options_.floor_bytes) {
            if (++low_streak_ >= needed) {
                state_ = State::pressure;
                low_streak_ = 0;
                recovered_streak_ = 0;
                update.entered_pressure = true;
            }
        } else {
            low_streak_ = 0;
        }
    } else {
        if (value >= effective_resume_bytes(options_)) {
            if (++recovered_streak_ >= needed) {
                state_ = State::ok;
                low_streak_ = 0;
                recovered_streak_ = 0;
                restart_pending_ = false;
                update.left_pressure = true;
            }
        } else {
            recovered_streak_ = 0;
        }
    }
    if (state_ == State::pressure && options_.policy == HostFloorPolicy::restart_smaller_ctx && !restart_pending_ &&
        restarts_ < options_.max_restarts &&
        (!restarted_before_ || now - last_restart_ >= options_.restart_cooldown)) {
        restart_pending_ = true;
        update.restart_requested = true;
    }
    return update;
}

void HostMemoryFloorGuard::reset(std::chrono::steady_clock::time_point now) noexcept {
    state_ = State::ok;
    low_streak_ = 0;
    recovered_streak_ = 0;
    unknown_samples_ = 0;
    restart_pending_ = false;
    next_sample_ = now;
}

void HostMemoryFloorGuard::restart_performed(std::chrono::steady_clock::time_point now) noexcept {
    restart_pending_ = false;
    ++restarts_;
    restarted_before_ = true;
    last_restart_ = now;
}

} // namespace sonder::inference::llamaserver
