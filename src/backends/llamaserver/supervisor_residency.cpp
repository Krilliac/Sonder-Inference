#include "supervisor.hpp"

#include <utility>

#include "prefix_warmup.hpp"

namespace sonder::inference::llamaserver {

Supervisor::RequestLease::~RequestLease() {
    if (activity_)
        activity_->fetch_sub(1);
}

Supervisor::RequestLease::RequestLease(RequestLease &&other) noexcept
    : activity_(std::move(other.activity_)), port_(std::exchange(other.port_, std::uint16_t{0})) {}

Supervisor::RequestLease &Supervisor::RequestLease::operator=(RequestLease &&other) noexcept {
    if (this != &other) {
        if (activity_)
            activity_->fetch_sub(1);
        activity_ = std::move(other.activity_);
        port_ = std::exchange(other.port_, std::uint16_t{0});
    }
    return *this;
}

Result<Supervisor::RequestLease> Supervisor::acquire_request(const CancellationToken &cancel) {
    for (;;) {
        auto ready = start(cancel);
        if (!ready.ok())
            return ready.status();
        std::lock_guard lock(mutex_);
        if (cancel.cancelled() || shutdown_.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
        // Atomic with maintenance admission closure. The caller uses this
        // lease's port, never a second start() that could wait behind itself.
        if (state_ == State::ready && !maintenance_pending_ && port_ == ready.value()) {
            active_requests_->fetch_add(1);
            return RequestLease(active_requests_, port_);
        }
    }
}

void Supervisor::record_residency() {
    const bool missing = residency_.gpu_offload_missing();
    const bool evicted = residency_.vram_evicted();
    if ((!missing || reported_missing_) && (!evicted || reported_eviction_))
        return;
    reported_missing_ = missing;
    reported_eviction_ = evicted;
    // Keep the incident after a relaunch; a short restart must not erase it
    // before health or the engine's periodic telemetry sampler can see it.
    if (!gpu_.residency)
        gpu_.residency.emplace();
    auto &event = *gpu_.residency;
    event.gpu_offload_missing |= residency_.gpu_offload_missing();
    event.vram_evicted |= residency_.vram_evicted();
    event.peak_dedicated_bytes = residency_.peak_dedicated_bytes();
    event.observed_dedicated_bytes = residency_.observed_dedicated_bytes();
    event.eviction_restarts = static_cast<std::uint64_t>(eviction_restarts_);
    event.action = "warn";
}

Status Supervisor::missing_offload_failure() const {
    return Status(ErrorCode::unavailable,
                  "llamaserver: gpu_offload_missing: dedicated GPU memory " +
                      std::to_string(gpu_.dedicated_bytes) + " bytes remained below " +
                      std::to_string(options_.spill_guard.residency.min_dedicated_bytes) + " bytes for " +
                      std::to_string(options_.spill_guard.residency.consecutive_samples) +
                      " consecutive post-readiness samples despite requested GPU layers; refusing to serve "
                      "(spill_guard.policy=refuse). Check CUDA backend availability and the child's PATH");
}

Supervisor::ResidencyAction Supervisor::residency_action() {
    std::lock_guard lock(mutex_);
    const auto &guard = options_.spill_guard;
    if (!guard.enabled || !guard.residency.enabled || !gpu_.residency)
        return ResidencyAction::none;
    const bool refuse = residency_.gpu_offload_missing() && guard.policy == SpillPolicy::refuse;
    const bool restart = residency_.vram_evicted() &&
                         guard.residency.on_eviction == LlamaServerEvictionPolicy::restart;
    if (!refuse && !restart)
        return ResidencyAction::none;
    auto &event = *gpu_.residency;
    if (!refuse && eviction_restarts_ >= guard.residency.max_eviction_restarts) {
        event.action = "restart_exhausted";
        return ResidencyAction::none;
    }
    maintenance_pending_ = true;
    event.action = "waiting_for_idle";
    // Warm-up owns its own HTTP operations. Do not cancel one merely to
    // recover residency; it must finish before the child can be replaced.
    const auto warmup = options_.warmup ? options_.warmup->status() : std::nullopt;
    if (active_requests_->load() != 0 || (warmup && (warmup->status == "pending" || warmup->status == "warming")))
        return ResidencyAction::none;
    state_ = State::starting;
    port_ = 0;
    if (refuse) {
        event.action = "refused";
        return ResidencyAction::refuse;
    }
    ++eviction_restarts_;
    ++epoch_;
    event.eviction_restarts = static_cast<std::uint64_t>(eviction_restarts_);
    event.action = "restarting";
    wake_.notify_all();
    return ResidencyAction::restart;
}

void Supervisor::append_residency_warnings(BackendRuntimeStatus &status) const {
    if (!gpu_.residency)
        return;
    const auto &event = *gpu_.residency;
    const std::vector<std::pair<std::string, std::string>> details{
        {"peak_dedicated_bytes", std::to_string(event.peak_dedicated_bytes)},
        {"observed_dedicated_bytes", std::to_string(event.observed_dedicated_bytes)},
        {"dedicated_bytes", std::to_string(gpu_.dedicated_bytes)},
        {"action", event.action}};
    if (event.gpu_offload_missing)
        status.warnings.push_back({
            "gpu_offload_missing", "warning", "gpu_probe",
            "GPU offload was requested but dedicated usage remained below " +
                std::to_string(options_.spill_guard.residency.min_dedicated_bytes) +
                " bytes after readiness. Check CUDA backend availability and the child's PATH", details, 1});
    if (event.vram_evicted)
        status.warnings.push_back({
            "vram_evicted", "warning", "gpu_probe",
            "Dedicated GPU memory dropped from a post-readiness peak of " +
                std::to_string(event.peak_dedicated_bytes) + " to " +
                std::to_string(event.observed_dedicated_bytes) +
                " bytes. Possible WDDM eviction; release competing GPU allocations before reloading the model",
            details, 1});
}

} // namespace sonder::inference::llamaserver
