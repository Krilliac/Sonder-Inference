// Supervisor integration of the host RAM admission check and floor guard
// (host_memory.hpp). Everything here runs on the monitor thread; nothing runs
// on the request path except the single bool read in acquire_request().
#include <algorithm>
#include <utility>

#include "prefix_warmup.hpp"
#include "supervisor.hpp"

namespace sonder::inference::llamaserver {
namespace {
std::string mib(std::uint64_t bytes) { return std::to_string(bytes / kHostMiB) + " MiB"; }
} // namespace

std::chrono::steady_clock::time_point Supervisor::host_now() const {
    return options_.host_memory.clock ? options_.host_memory.clock() : std::chrono::steady_clock::now();
}

HostMemory Supervisor::host_memory_now() const {
    return options_.host_memory.provider ? options_.host_memory.provider() : read_host_memory();
}

void Supervisor::set_host_warning(BackendWarning warning) {
    // Surfaced through runtime_status().warnings; the engine's sampler turns
    // those into backend.warning telemetry, so no event is queued here.
    std::lock_guard lock(mutex_);
    const auto existing = std::find_if(host_warnings_.begin(), host_warnings_.end(),
                                       [&](const BackendWarning &w) { return w.code == warning.code; });
    if (existing != host_warnings_.end())
        *existing = std::move(warning);
    else
        host_warnings_.push_back(std::move(warning));
}

void Supervisor::clear_host_warning(const std::string &code) {
    std::lock_guard lock(mutex_);
    std::erase_if(host_warnings_, [&](const BackendWarning &warning) { return warning.code == code; });
}

Status Supervisor::admit_host_memory() {
    const auto &admission = options_.host_memory.admission;
    if (admission.estimated_resident_bytes == 0)
        return Status::success();
    if (admission.estimate_lower_bound) {
        set_host_warning(BackendWarning{
            "host_memory_estimate_lower_bound",
            "warning",
            "host_probe",
            "the host RAM estimate is a lower bound: llama-server --fit decides the offload and no VRAM budget is known",
            {{"estimated_resident_bytes", std::to_string(admission.estimated_resident_bytes)}},
            1});
    }
    const auto verdict = check_host_admission(admission, host_memory_now());
    using Outcome = HostAdmissionVerdict::Outcome;
    switch (verdict.outcome) {
    case Outcome::refused:
        return Status(ErrorCode::unavailable, verdict.message);
    case Outcome::admitted_override:
    case Outcome::admitted_unknown:
        set_host_warning(BackendWarning{
            verdict.outcome == Outcome::admitted_override ? "host_memory_admission_override" : "host_memory_unknown",
            "warning",
            "host_probe",
            verdict.message,
            {{"estimated_resident_bytes", std::to_string(admission.estimated_resident_bytes)},
             {"reserve_bytes", std::to_string(admission.reserve_bytes)},
             {"required_bytes", std::to_string(verdict.required_bytes)},
             {"available_bytes", std::to_string(verdict.available_bytes)}},
            1});
        break;
    case Outcome::admitted:
    case Outcome::skipped:
        clear_host_warning("host_memory_admission_override");
        clear_host_warning("host_memory_unknown");
        break;
    }
    return Status::success();
}

bool Supervisor::host_memory_tick() {
    const auto &floor = options_.host_memory.floor;
    if (!floor.enabled)
        return false;
    const auto now = host_now();

    // Relaunch with a smaller context once a restart was requested and the
    // child is idle (requests and warm-up drain first, like residency).
    const auto restart_if_idle = [&]() -> bool {
        std::lock_guard lock(mutex_);
        maintenance_pending_ = true;  // closes admission and makes start() wait
        host_restart_maintenance_ = true;
        const auto warmup = options_.warmup ? options_.warmup->status() : std::nullopt;
        if (active_requests_->load() != 0 || (warmup && (warmup->status == "pending" || warmup->status == "warming")))
            return false;
        const auto current = find_context_size(arguments_);
        const auto next = current ? next_fit_context(*current, options_.spill_guard) : std::nullopt;
        if (!next) {
            // Nothing smaller to try: keep serving under the floor policy.
            host_floor_.restart_abandoned();
            if (!residency_maintenance_) maintenance_pending_ = false;
            host_restart_maintenance_ = false;
            wake_.notify_all();
            return false;
        }
        arguments_ = with_context_size(arguments_, *next);
        context_.fitted_ctx = *next;
        ++epoch_;
        port_ = 0;
        state_ = State::starting;
        host_floor_.restart_performed(now);
        host_admission_closed_ = false;
        host_restart_maintenance_ = false;  // maintenance_pending_ clears when the new child is ready
        runtime_events_->push("backend.restart", {{"reason", "host_memory_low"},
                                                   {"ctx", *next},
                                                   {"attempt", static_cast<std::uint64_t>(host_floor_.restarts())}});
        wake_.notify_all();
        return true;
    };

    // A pending restart keeps being sampled: if memory recovers while a
    // request still holds the child busy, observe() drops the request and the
    // block below reopens admission instead of restarting for no reason.
    if (!host_floor_.due(now))
        return host_floor_.restart_pending() ? restart_if_idle() : false;

    const HostMemory memory = host_memory_now();
    const auto update = host_floor_.observe(memory.available_physical, now);
    if (update.unknown_sample) {
        // Documented fallback: an unmeasurable host neither trips nor clears
        // the guard. The warning is refreshed on every unknown sample (a known
        // sample clears it) and states what the guard is doing meanwhile.
        set_host_warning(BackendWarning{"host_memory_unknown", "warning", "host_probe",
                                        "host available memory could not be measured" +
                                            (memory.note.empty() ? std::string() : " (" + memory.note + ")") +
                                            "; the floor guard keeps its last state (" +
                                            (host_floor_.admitting() ? "admitting" : "admission closed") +
                                            ") until a reading succeeds",
                                        {{"floor_bytes", std::to_string(floor.floor_bytes)}}, 1});
        return host_floor_.restart_pending() ? restart_if_idle() : false;
    }
    clear_host_warning("host_memory_unknown");
    if (update.entered_pressure) {
        set_host_warning(BackendWarning{
            "host_memory_low", "warning", "host_probe",
            "available physical memory " + mib(host_floor_.last_available_bytes()) + " is below the " +
                mib(floor.floor_bytes) + " floor (policy " + to_string(floor.policy) +
                "); the child may be paging model weights or about to be killed by the OS",
            {{"available_bytes", std::to_string(host_floor_.last_available_bytes())},
             {"floor_bytes", std::to_string(floor.floor_bytes)},
             {"resume_bytes", std::to_string(effective_resume_bytes(floor))},
             {"policy", to_string(floor.policy)}},
            1});
    } else if (update.left_pressure) {
        clear_host_warning("host_memory_low");
    }
    {
        std::lock_guard lock(mutex_);
        host_admission_closed_ = !host_floor_.admitting();
        if (host_restart_maintenance_ && !host_floor_.restart_pending()) {
            // The pressure that asked for a restart ended first: reopen.
            host_restart_maintenance_ = false;
            if (!residency_maintenance_) maintenance_pending_ = false;
            wake_.notify_all();
        }
    }
    return host_floor_.restart_pending() ? restart_if_idle() : false;
}

} // namespace sonder::inference::llamaserver
