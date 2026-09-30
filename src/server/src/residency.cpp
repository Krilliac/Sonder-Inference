#include "residency.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace sonder::inference::server::detail {

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

ModelResidency::ModelResidency(ResidencyConfig config, Loader loader, Evictor evictor)
    : config_(config), loader_(std::move(loader)), evictor_(std::move(evictor)) {}

ModelResidency::~ModelResidency() { stop_sweeper(); }

bool ModelResidency::add(const std::string& id, const std::string& backend, bool is_default) {
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
    entries_.push_back(std::move(e));
    return true;
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
        // Unloaded: this request loads it; later ones join.
        const std::uint64_t generation = ++e.load_generation;
        e.state = ResidencyState::loading;
        const std::string id = e.id;
        std::vector<Eviction> victims;
        if (config_.max_resident != 0) {
            collect_over_cap_locked(config_.max_resident - 1, i, Clock::now(), victims);
        }
        lock.unlock();
        // Free the evicted models before loading the new one.
        run_evictions(victims);
        victims.clear();
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
        lock.lock();
        Entry& done = entries_[i];
        if (loaded.ok()) {
            done.model = std::move(loaded).value();
            done.descriptor = done.model->descriptor();
            done.state = ResidencyState::resident;
            ++done.loads;
            ++total_loads_;
            done.last_used = Clock::now();
        } else {
            done.state = ResidencyState::unloaded;
            done.failed_generation = generation;
            done.last_error = loaded.status();
            ++done.load_failures;
            ++total_load_failures_;
        }
        joined = generation;
        load_cv_.notify_all();
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
    while (count > keep) {
        std::optional<std::size_t> lru;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
            if ((except && *except == i) || e.state != ResidencyState::resident || e.pins != 0) {
                continue;
            }
            if (!lru || e.last_used < entries_[*lru].last_used) {
                lru = i;
            }
        }
        if (!lru) {
            return;  // everything left is pinned or loading: soft cap
        }
        out.push_back(evict_locked(entries_[*lru], kEvictMaxResident, now));
        --count;
    }
}

Eviction ModelResidency::evict_locked(Entry& e, const char* reason, Clock::time_point now) {
    Eviction ev;
    ev.id = e.id;
    ev.backend = e.backend;
    ev.model = std::move(e.model);
    ev.index = static_cast<std::size_t>(&e - entries_.data());
    ev.reason = reason;
    ev.idle_ms = std::max(0.0, std::chrono::duration<double, std::milli>(now - e.last_used).count());
    e.model.reset();
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
        } else if (e.state == ResidencyState::loading) {
            ++t.loading;
        } else if (e.state == ResidencyState::unloading) {
            ++t.unloading;
        }
    }
    t.loads = total_loads_;
    t.load_failures = total_load_failures_;
    t.evictions = total_evictions_;
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
            if (e.state == ResidencyState::resident) {
                e.state = ResidencyState::unloaded;
            }
        }
    }
    doomed.clear();
}

}  // namespace sonder::inference::server::detail
