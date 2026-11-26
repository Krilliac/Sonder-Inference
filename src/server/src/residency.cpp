#include "residency.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace sonder::inference::server::detail {

namespace {

constexpr double kLn2 = 0.693147180559945309417;

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::uint64_t entry_vram_bytes(const ModelResourceFootprint& fp) noexcept { return fp.vram_bytes(); }
std::uint64_t entry_ram_bytes(const ModelResourceFootprint& fp) noexcept { return fp.ram_bytes(); }

}  // namespace

const char* residency_state_name(ResidencyState state) noexcept {
    switch (state) {
        case ResidencyState::unloaded: return "unloaded";
        case ResidencyState::loading: return "loading";
        case ResidencyState::resident: return "resident";
        case ResidencyState::unloading: return "unloading";
    }
    return "unloaded";
}

// ------------------------------------------------------------------ Pin

ModelResidency::Pin::Pin(std::shared_ptr<ModelResidency> owner, std::size_t index, std::shared_ptr<Model> model,
                         std::string id, std::string backend)
    : owner_(std::move(owner)), index_(index), model_(std::move(model)), id_(std::move(id)),
      backend_(std::move(backend)) {}

ModelResidency::Pin::Pin(Pin&& other) noexcept
    : owner_(std::move(other.owner_)), index_(other.index_), model_(std::move(other.model_)),
      id_(std::move(other.id_)), backend_(std::move(other.backend_)) {
    other.owner_.reset();
}

ModelResidency::Pin& ModelResidency::Pin::operator=(Pin&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = std::move(other.owner_);
        other.owner_.reset();
        index_ = other.index_;
        model_ = std::move(other.model_);
        id_ = std::move(other.id_);
        backend_ = std::move(other.backend_);
    }
    return *this;
}

ModelResidency::Pin::~Pin() { release(); }

void ModelResidency::Pin::release() noexcept {
    if (!owner_) {
        return;
    }
    // Drop our model reference before the owner may evict it.
    model_.reset();
    const std::shared_ptr<ModelResidency> owner = std::move(owner_);
    owner_.reset();
    owner->release(index_);
}

// ------------------------------------------------------- ModelResidency

ModelResidency::ModelResidency(ResidencyConfig config, Loader loader, Evictor evictor, FootprintObserver observer,
                               FootprintEstimator estimator)
    : config_(config),
      loader_(std::move(loader)),
      evictor_(std::move(evictor)),
      observer_(std::move(observer)),
      estimator_(std::move(estimator)) {}

ModelResidency::~ModelResidency() { stop_sweeper(); }

double ModelResidency::compute_eviction_cost(double demand_score, double reload_ms,
                                             std::uint64_t bytes_freed) noexcept {
    const double eff_demand = (std::isfinite(demand_score) && demand_score > 0.0) ? demand_score : 1e-6;
    const double eff_reload = (std::isfinite(reload_ms) && reload_ms > 0.0) ? reload_ms : 1.0;
    const std::uint64_t denom = std::max<std::uint64_t>(1u, bytes_freed);
    return (eff_demand * eff_reload) / static_cast<double>(denom);
}

bool ModelResidency::add(const std::string& id, const std::string& backend, bool is_default,
                         ModelResourceFootprint estimated) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& e : entries_) {
        if (e.id == id) {
            return false;
        }
    }
    Entry e;
    e.id = id;
    e.backend = backend;
    e.is_default = is_default;
    e.estimated_footprint = estimated;
    entries_.push_back(std::move(e));
    return true;
}

bool ModelResidency::set_estimated_footprint(const std::string& id, ModelResourceFootprint estimated) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.id == id) {
            e.estimated_footprint = estimated;
            return true;
        }
    }
    return false;
}

bool ModelResidency::reconcile_footprint(const std::string& id, ModelResourceFootprint observed,
                                         Clock::time_point now) {
    std::vector<Eviction> victims;
    {
        std::lock_guard<std::mutex> lock(mu_);
        std::optional<std::size_t> idx;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].id == id) {
                idx = i;
                break;
            }
        }
        if (!idx || entries_[*idx].state != ResidencyState::resident) {
            return false;
        }
        Entry& e = entries_[*idx];
        e.observed_footprint = observed;
        e.last_observed_footprint = observed;
        e.has_last_observed = !observed.empty();
        if (has_byte_budget()) {
            collect_over_budget_locked(*idx, now, victims);
        }
    }
    run_evictions(victims);
    return true;
}

bool ModelResidency::set_reload_ms(const std::string& id, double reload_ms) {
    if (!std::isfinite(reload_ms) || reload_ms < 0.0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.id == id) {
            e.reload_ms = reload_ms;
            return true;
        }
    }
    return false;
}

bool ModelResidency::record_demand(const std::string& id, double weight, Clock::time_point now) {
    if (!std::isfinite(weight) || weight < 0.0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& e : entries_) {
        if (e.id == id) {
            bump_demand_locked(e, weight, now);
            return true;
        }
    }
    return false;
}

double ModelResidency::decayed_demand_locked(const Entry& e, Clock::time_point now) const noexcept {
    if (e.demand_score <= 0.0) {
        return 0.0;
    }
    if (!e.has_demand_timestamp || config_.demand_decay_half_life.count() <= 0) {
        return e.demand_score;
    }
    const double elapsed_ms =
        std::max(0.0, std::chrono::duration<double, std::milli>(now - e.demand_updated_at).count());
    const double half_life_ms = static_cast<double>(config_.demand_decay_half_life.count());
    return e.demand_score * std::exp2(-elapsed_ms / half_life_ms);
}

double ModelResidency::demand_rate_per_min_locked(double decayed_demand) const noexcept {
    if (decayed_demand <= 0.0 || config_.demand_decay_half_life.count() <= 0) {
        return 0.0;
    }
    const double half_life_min = static_cast<double>(config_.demand_decay_half_life.count()) / 60000.0;
    return decayed_demand * (kLn2 / half_life_min);
}

void ModelResidency::bump_demand_locked(Entry& e, double weight, Clock::time_point now) noexcept {
    e.demand_score = decayed_demand_locked(e, now) + weight;
    e.demand_updated_at = now;
    e.has_demand_timestamp = true;
}

void ModelResidency::active_bytes_locked(std::uint64_t& vram_out, std::uint64_t& ram_out) const noexcept {
    vram_out = 0;
    ram_out = 0;
    for (const auto& e : entries_) {
        if (e.state == ResidencyState::resident) {
            vram_out = saturating_add(vram_out, entry_vram_bytes(e.observed_footprint));
            ram_out = saturating_add(ram_out, entry_ram_bytes(e.observed_footprint));
        } else if (e.state == ResidencyState::loading) {
            vram_out = saturating_add(vram_out, entry_vram_bytes(e.reserved_footprint));
            ram_out = saturating_add(ram_out, entry_ram_bytes(e.reserved_footprint));
        }
    }
}

std::optional<std::size_t> ModelResidency::find_locked(const std::string& id_or_alias) const {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].id == id_or_alias || (id_or_alias == "default" && entries_[i].is_default)) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<std::string> ModelResidency::resolve(const std::string& id_or_alias) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto i = find_locked(id_or_alias)) {
        return entries_[*i].id;
    }
    return std::nullopt;
}

Result<ModelResidency::Pin> ModelResidency::acquire(const std::string& id_or_alias) {
    std::unique_lock<std::mutex> lock(mu_);
    const auto found = find_locked(id_or_alias);
    if (!found) {
        return Status(ErrorCode::not_found, "model '" + id_or_alias + "' is not served");
    }
    const std::size_t i = *found;
    // Pinned from here on, including while the load runs: never evicted.
    ++entries_[i].pins;
    std::optional<std::uint64_t> joined;  // generation of the load this request waits for
    for (;;) {
        Entry& e = entries_[i];  // re-read after every unlock: add() may grow the vector
        if (e.state == ResidencyState::resident) {
            bump_demand_locked(e, 1.0, Clock::now());
            return Pin(shared_from_this(), i, e.model, e.id, e.backend);
        }
        if (joined && e.failed_generation >= *joined) {
            Status failed = e.last_error;
            --e.pins;
            lock.unlock();
            sweep_cv_.notify_all();
            return failed;
        }
        if (e.state == ResidencyState::unloading) {
            // Wait until the evicted handle is released, then load anew.
            load_cv_.wait(lock, [&] { return entries_[i].state != ResidencyState::unloading; });
            continue;
        }
        if (e.state == ResidencyState::loading) {
            // Join the load in progress instead of starting another.
            joined = e.load_generation;
            load_cv_.wait(lock, [&] {
                const Entry& x = entries_[i];
                return x.state != ResidencyState::loading || x.load_generation != *joined;
            });
            continue;
        }
        // Unloaded: resolve pre-load footprint estimate (optional estimator
        // runs unlocked; re-check state afterwards in case another request won).
        const std::string id = e.id;
        std::optional<ModelResourceFootprint> dynamic_est;
        if (estimator_) {
            lock.unlock();
            try {
                dynamic_est = estimator_(id);
            } catch (...) {
                dynamic_est = std::nullopt;
            }
            lock.lock();
            if (entries_[i].state != ResidencyState::unloaded) {
                continue;
            }
        }
        Entry& cur = entries_[i];
        const ModelResourceFootprint reservation =
            dynamic_est ? *dynamic_est
                        : (cur.has_last_observed ? cur.last_observed_footprint : cur.estimated_footprint);

        // Pre-load byte-budget admission check: never evict unpinned models if
        // pinned/loading models + incoming reservation still exceed the budget.
        if (has_byte_budget()) {
            std::uint64_t pinned_vram = 0;
            std::uint64_t pinned_ram = 0;
            for (std::size_t j = 0; j < entries_.size(); ++j) {
                if (j == i) continue;
                const Entry& other = entries_[j];
                if (other.state == ResidencyState::loading) {
                    pinned_vram = saturating_add(pinned_vram, entry_vram_bytes(other.reserved_footprint));
                    pinned_ram = saturating_add(pinned_ram, entry_ram_bytes(other.reserved_footprint));
                } else if (other.state == ResidencyState::resident && other.pins > 0) {
                    pinned_vram = saturating_add(pinned_vram, entry_vram_bytes(other.observed_footprint));
                    pinned_ram = saturating_add(pinned_ram, entry_ram_bytes(other.observed_footprint));
                }
            }
            const std::uint64_t req_vram = entry_vram_bytes(reservation);
            const std::uint64_t req_ram = entry_ram_bytes(reservation);
            const bool vram_exceeded =
                config_.max_resident_vram_bytes > 0 &&
                saturating_add(pinned_vram, req_vram) > config_.max_resident_vram_bytes;
            const bool ram_exceeded =
                config_.max_resident_ram_bytes > 0 &&
                saturating_add(pinned_ram, req_ram) > config_.max_resident_ram_bytes;
            if (vram_exceeded || ram_exceeded) {
                ++cur.admission_rejections;
                ++total_admission_rejections_;
                --cur.pins;
                std::string msg = "insufficient resident ";
                if (vram_exceeded) {
                    msg += "VRAM budget for model '" + id + "' (need " + std::to_string(req_vram) +
                           " B, pinned/reserved " + std::to_string(pinned_vram) + " B, budget " +
                           std::to_string(config_.max_resident_vram_bytes) + " B)";
                } else {
                    msg += "RAM budget for model '" + id + "' (need " + std::to_string(req_ram) +
                           " B, pinned/reserved " + std::to_string(pinned_ram) + " B, budget " +
                           std::to_string(config_.max_resident_ram_bytes) + " B)";
                }
                lock.unlock();
                sweep_cv_.notify_all();
                return Status(ErrorCode::unavailable, std::move(msg));
            }
        }

        // Reserve estimated bytes before unlocking so concurrent loads account
        // for this in-flight allocation.
        const std::uint64_t generation = ++cur.load_generation;
        cur.state = ResidencyState::loading;
        cur.reserved_footprint = reservation;
        const Clock::time_point now = Clock::now();
        std::vector<Eviction> victims;
        if (config_.max_resident != 0) {
            collect_over_cap_locked(config_.max_resident - 1, i, now, victims);
        }
        if (has_byte_budget()) {
            collect_over_budget_locked(i, now, victims);
        }
        lock.unlock();
        // Free the evicted models before loading the new one.
        run_evictions(victims);
        victims.clear();

        const Clock::time_point load_start = Clock::now();
        Result<std::shared_ptr<Model>> loaded = Status(ErrorCode::internal, "model load did not run");
        try {
            loaded = loader_(id);
            if (loaded.ok() && !loaded.value()) {
                loaded = Status(ErrorCode::internal, "model load returned no model");
            }
        } catch (const std::exception& ex) {
            loaded = Status(ErrorCode::internal, std::string("model load threw: ") + ex.what());
        } catch (...) {
            loaded = Status(ErrorCode::internal, "model load threw an exception");
        }
        const double load_ms =
            std::max(0.0, std::chrono::duration<double, std::milli>(Clock::now() - load_start).count());

        // Reconcile observed allocations post-load (unlocked).
        ModelResourceFootprint observed = reservation;
        if (loaded.ok()) {
            std::optional<ModelResourceFootprint> measured;
            if (observer_) {
                try {
                    measured = observer_(id, loaded.value(), reservation);
                } catch (...) {
                    measured = std::nullopt;
                }
            }
            if (measured) {
                observed = *measured;
            } else if (observed.empty()) {
                const auto& desc = loaded.value()->descriptor();
                if (desc.size_bytes > 0) {
                    if (loaded.value()->device_id().rfind("gpu:", 0) == 0) {
                        observed.vram.weights_bytes = desc.size_bytes;
                    } else {
                        observed.ram.weights_bytes = desc.size_bytes;
                    }
                }
            }
        }

        std::vector<Eviction> post_victims;
        lock.lock();
        Entry& done = entries_[i];
        // Pre-load reservation is always released immediately on both success
        // (replaced by observed_footprint) and error (cleared to zero).
        done.reserved_footprint = {};
        if (loaded.ok()) {
            done.model = std::move(loaded).value();
            done.descriptor = done.model->descriptor();
            done.observed_footprint = observed;
            done.last_observed_footprint = observed;
            done.has_last_observed = !observed.empty();
            done.state = ResidencyState::resident;
            ++done.loads;
            ++total_loads_;
            done.reload_ms =
                (done.loads == 1 || done.reload_ms <= 0.0) ? load_ms : (0.5 * done.reload_ms + 0.5 * load_ms);
            const Clock::time_point done_now = Clock::now();
            done.last_used = done_now;
            if (has_byte_budget()) {
                collect_over_budget_locked(i, done_now, post_victims);
            }
        } else {
            done.observed_footprint = {};
            done.state = ResidencyState::unloaded;
            done.failed_generation = generation;
            done.last_error = loaded.status();
            ++done.load_failures;
            ++total_load_failures_;
        }
        joined = generation;
        load_cv_.notify_all();
        if (!post_victims.empty()) {
            lock.unlock();
            run_evictions(post_victims);
            post_victims.clear();
            lock.lock();
        }
    }
}

void ModelResidency::release(std::size_t index) noexcept {
    std::vector<Eviction> victims;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (index >= entries_.size()) {
            return;
        }
        Entry& e = entries_[index];
        if (e.pins > 0) {
            --e.pins;
        }
        const Clock::time_point now = Clock::now();
        e.last_used = now;
        if (config_.max_resident != 0) {
            // Pinned models may have pushed the count over the cap.
            collect_over_cap_locked(config_.max_resident, std::nullopt, now, victims);
        }
        if (has_byte_budget()) {
            // Reconciled footprints of pinned models may have pushed bytes over budget.
            collect_over_budget_locked(std::nullopt, now, victims);
        }
    }
    sweep_cv_.notify_all();
    run_evictions(victims);
}

void ModelResidency::collect_over_cap_locked(std::size_t keep, std::optional<std::size_t> except,
                                             Clock::time_point now, std::vector<Eviction>& out) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (except && *except == i) {
            continue;
        }
        // An unloading model is already on its way out: not counted.
        if (entries_[i].state == ResidencyState::resident || entries_[i].state == ResidencyState::loading) {
            ++count;
        }
    }
    const bool use_cost = has_byte_budget();
    while (count > keep) {
        std::optional<std::size_t> victim;
        double best_cost = 0.0;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
            if ((except && *except == i) || e.state != ResidencyState::resident || e.pins != 0) {
                continue;
            }
            if (!use_cost) {
                if (!victim || e.last_used < entries_[*victim].last_used) {
                    victim = i;
                }
            } else {
                const std::uint64_t bytes =
                    saturating_add(entry_vram_bytes(e.observed_footprint), entry_ram_bytes(e.observed_footprint));
                const double cost = compute_eviction_cost(decayed_demand_locked(e, now), e.reload_ms, bytes);
                if (!victim || cost < best_cost ||
                    (cost == best_cost && e.last_used < entries_[*victim].last_used)) {
                    victim = i;
                    best_cost = cost;
                }
            }
        }
        if (!victim) {
            return;  // everything left is pinned or loading: soft cap
        }
        out.push_back(evict_locked(entries_[*victim], kEvictMaxResident, now, best_cost));
        --count;
    }
}

void ModelResidency::collect_over_budget_locked(std::optional<std::size_t> except, Clock::time_point now,
                                                std::vector<Eviction>& out) {
    std::uint64_t active_vram = 0;
    std::uint64_t active_ram = 0;
    active_bytes_locked(active_vram, active_ram);

    for (;;) {
        const bool vram_over =
            config_.max_resident_vram_bytes > 0 && active_vram > config_.max_resident_vram_bytes;
        const bool ram_over =
            config_.max_resident_ram_bytes > 0 && active_ram > config_.max_resident_ram_bytes;
        if (!vram_over && !ram_over) {
            return;
        }
        std::optional<std::size_t> victim;
        double best_cost = 0.0;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
            if ((except && *except == i) || e.state != ResidencyState::resident || e.pins != 0) {
                continue;
            }
            std::uint64_t bytes_freed = 0;
            if (vram_over) {
                bytes_freed = saturating_add(bytes_freed, entry_vram_bytes(e.observed_footprint));
            }
            if (ram_over) {
                bytes_freed = saturating_add(bytes_freed, entry_ram_bytes(e.observed_footprint));
            }
            if (bytes_freed == 0) {
                continue;  // freeing this entry does not relieve any pressured tier
            }
            const double cost = compute_eviction_cost(decayed_demand_locked(e, now), e.reload_ms, bytes_freed);
            if (!victim || cost < best_cost || (cost == best_cost && e.last_used < entries_[*victim].last_used)) {
                victim = i;
                best_cost = cost;
            }
        }
        if (!victim) {
            return;  // everything left on pressured tiers is pinned or loading
        }
        const std::uint64_t freed_vram = entry_vram_bytes(entries_[*victim].observed_footprint);
        const std::uint64_t freed_ram = entry_ram_bytes(entries_[*victim].observed_footprint);
        active_vram = active_vram >= freed_vram ? (active_vram - freed_vram) : 0;
        active_ram = active_ram >= freed_ram ? (active_ram - freed_ram) : 0;
        const char* reason = vram_over ? kEvictVramBudget : kEvictRamBudget;
        out.push_back(evict_locked(entries_[*victim], reason, now, best_cost));
    }
}

Eviction ModelResidency::evict_locked(Entry& e, const char* reason, Clock::time_point now, double eviction_cost) {
    Eviction ev;
    ev.id = e.id;
    ev.backend = e.backend;
    ev.model = std::move(e.model);
    ev.index = static_cast<std::size_t>(&e - entries_.data());
    ev.reason = reason;
    ev.idle_ms = std::max(0.0, std::chrono::duration<double, std::milli>(now - e.last_used).count());
    ev.freed_footprint = e.observed_footprint;
    ev.eviction_cost = eviction_cost;
    e.model.reset();
    e.observed_footprint = {};
    e.reserved_footprint = {};
    e.state = ResidencyState::unloading;  // unloaded once run_evictions() released it
    ++e.evictions;
    ++total_evictions_;
    return ev;
}

void ModelResidency::run_evictions(std::vector<Eviction>& evictions) noexcept {
    for (auto& ev : evictions) {
        if (evictor_) {
            try {
                evictor_(ev);
            } catch (...) {
                // Eviction bookkeeping already happened; the handle drops below.
            }
        }
        ev.model.reset();
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (ev.index < entries_.size() && entries_[ev.index].state == ResidencyState::unloading) {
                entries_[ev.index].state = ResidencyState::unloaded;
            }
        }
        load_cv_.notify_all();
    }
}

std::size_t ModelResidency::sweep(Clock::time_point now) {
    if (config_.idle_ttl.count() <= 0) {
        return 0;
    }
    std::vector<Eviction> victims;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& e : entries_) {
            if (e.state == ResidencyState::resident && e.pins == 0 && now - e.last_used >= config_.idle_ttl) {
                victims.push_back(evict_locked(e, kEvictIdleTtl, now));
            }
        }
    }
    run_evictions(victims);
    return victims.size();
}

std::optional<ModelResidency::Clock::time_point> ModelResidency::next_deadline_locked() const {
    std::optional<Clock::time_point> next;
    for (const auto& e : entries_) {
        if (e.state == ResidencyState::resident && e.pins == 0) {
            const Clock::time_point d = e.last_used + config_.idle_ttl;
            if (!next || d < *next) {
                next = d;
            }
        }
    }
    return next;
}

void ModelResidency::sweeper_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!sweeper_stop_) {
        const auto deadline = next_deadline_locked();
        if (deadline) {
            // Woken early by release() (a new deadline) or stop_sweeper().
            sweep_cv_.wait_until(lock, *deadline);
        } else {
            sweep_cv_.wait(lock);
        }
        if (sweeper_stop_) {
            break;
        }
        lock.unlock();
        (void)sweep(Clock::now());
        lock.lock();
    }
}

void ModelResidency::start_sweeper() {
    if (config_.idle_ttl.count() <= 0 || sweeper_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        sweeper_stop_ = false;
    }
    sweeper_ = std::thread([this] { sweeper_loop(); });
}

void ModelResidency::stop_sweeper() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        sweeper_stop_ = true;
    }
    sweep_cv_.notify_all();
    if (sweeper_.joinable()) {
        sweeper_.join();
    }
}

std::vector<ResidencySnapshot> ModelResidency::snapshot(Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ResidencySnapshot> out;
    out.reserve(entries_.size());
    for (const auto& e : entries_) {
        ResidencySnapshot s;
        s.id = e.id;
        s.backend = e.backend;
        s.is_default = e.is_default;
        s.state = e.state;
        s.pins = e.pins;
        s.loads = e.loads;
        s.load_failures = e.load_failures;
        s.evictions = e.evictions;
        s.admission_rejections = e.admission_rejections;
        s.estimated_footprint = e.estimated_footprint;
        s.reserved_footprint = e.reserved_footprint;
        s.observed_footprint = e.observed_footprint;
        s.reload_ms = e.reload_ms;
        s.demand_score = decayed_demand_locked(e, now);
        s.demand_rate_per_min = demand_rate_per_min_locked(s.demand_score);
        if (e.state == ResidencyState::resident) {
            s.model_instance_id = e.model->instance_id();
            if (e.pins == 0) {
                s.idle_s = std::max(0.0, std::chrono::duration<double>(now - e.last_used).count());
            }
        }
        out.push_back(std::move(s));
    }
    return out;
}

ResidencyTotals ModelResidency::totals() const {
    std::lock_guard<std::mutex> lock(mu_);
    ResidencyTotals t;
    t.registered = entries_.size();
    for (const auto& e : entries_) {
        if (e.state == ResidencyState::resident) {
            ++t.resident;
            t.resident_vram_bytes = saturating_add(t.resident_vram_bytes, entry_vram_bytes(e.observed_footprint));
            t.resident_ram_bytes = saturating_add(t.resident_ram_bytes, entry_ram_bytes(e.observed_footprint));
        } else if (e.state == ResidencyState::loading) {
            ++t.loading;
            t.reserved_vram_bytes = saturating_add(t.reserved_vram_bytes, entry_vram_bytes(e.reserved_footprint));
            t.reserved_ram_bytes = saturating_add(t.reserved_ram_bytes, entry_ram_bytes(e.reserved_footprint));
        } else if (e.state == ResidencyState::unloading) {
            ++t.unloading;
        }
    }
    t.loads = total_loads_;
    t.load_failures = total_load_failures_;
    t.evictions = total_evictions_;
    t.admission_rejections = total_admission_rejections_;
    return t;
}

std::shared_ptr<Model> ModelResidency::resident_model(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& e : entries_) {
        if (e.id == id) {
            return e.state == ResidencyState::resident ? e.model : nullptr;
        }
    }
    return nullptr;
}

std::optional<ModelDescriptor> ModelResidency::last_descriptor(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& e : entries_) {
        if (e.id == id) {
            return e.descriptor;
        }
    }
    return std::nullopt;
}

void ModelResidency::clear() {
    std::vector<std::shared_ptr<Model>> doomed;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& e : entries_) {
            if (e.model) {
                doomed.push_back(std::move(e.model));
            }
            e.model.reset();
            e.observed_footprint = {};
            e.reserved_footprint = {};
            if (e.state == ResidencyState::resident) {
                e.state = ResidencyState::unloaded;
            }
        }
    }
    doomed.clear();
}

}  // namespace sonder::inference::server::detail
