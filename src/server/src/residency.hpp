// Internal: model residency for `sonder-infer serve` (docs/SERVER.md "Model
// residency", Item #18 measured-byte/reload-cost policy).
//
// Every served model is registered up front (id, backend, default flag, plus
// optional pre-load ResourceFootprint estimate; no backend I/O). A request
// acquires a Pin: the first acquire loads the model (concurrent first requests
// share one load), and while any Pin is alive the model is never evicted.
// Unpinned models may be evicted after an idle TTL, to honour a max-resident
// count cap, or to stay within configured VRAM/RAM byte budgets; the next
// acquire loads them again.
//
// When byte budgets (`max_resident_vram_bytes` / `max_resident_ram_bytes`) are
// configured:
//   - Pre-load byte reservations are taken before backend loading begins and
//     released immediately if loading fails.
//   - Post-load allocations are reconciled against observed footprints (via an
//     optional FootprintObserver callback or explicit reconcile_footprint()).
//   - Cold loads that cannot fit alongside pinned/loading models even after
//     evicting every unpinned model are refused cleanly without thrashing.
//   - Unpinned victims are selected minimizing
//     `eviction_cost = (demand_score * reload_ms) / max(1, bytes_freed)`.
//
// Defaults (idle_ttl 0, max_resident 0, byte budgets 0) never evict, so a
// model that was loaded stays loaded for the life of the server.
//
// Thread safety: every member is safe to call from any thread. The loader,
// observer, estimator, and evictor are called without the residency lock held.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "sonder/inference/error.hpp"
#include "sonder/inference/model.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

using sonder::inference::server::ModelResourceFootprint;
using sonder::inference::server::ResourceFootprint;

// `unloading`: evicted, and the evictor is still releasing the old handle
// (telemetry, engine unload). The model reports `unloaded` only once that has
// finished, and a request for it waits for the release before loading it
// again, so an old and a new copy are never held at the same time.
enum class ResidencyState { unloaded, loading, resident, unloading };

// "unloaded", "loading", "resident", "unloading".
const char* residency_state_name(ResidencyState state) noexcept;

struct ResidencyConfig {
    // Evict a model after it has been unpinned this long; 0 = never.
    std::chrono::milliseconds idle_ttl{0};
    // Evict least recently used (or lowest eviction-cost when a byte budget is
    // configured) unpinned models to keep at most this many loaded or loading;
    // 0 = no cap. A soft cap: a pinned model is never evicted, so pinned models
    // alone can exceed it until they are released.
    std::size_t max_resident = 0;
    // Byte budgets for live VRAM and host-RAM residency (0 = unlimited).
    std::uint64_t max_resident_vram_bytes = 0;
    std::uint64_t max_resident_ram_bytes = 0;
    // Half-life for exponential decay of per-model recent demand score.
    std::chrono::milliseconds demand_decay_half_life{std::chrono::minutes(5)};
};

// Why a model was evicted (telemetry `model.evicted` attribute `reason`).
inline constexpr const char* kEvictIdleTtl = "idle_ttl";
inline constexpr const char* kEvictMaxResident = "max_resident";
inline constexpr const char* kEvictVramBudget = "vram_budget";
inline constexpr const char* kEvictRamBudget = "ram_budget";

struct Eviction {
    std::string id;
    std::string backend;
    std::shared_ptr<Model> model;
    std::size_t index = 0;  // entry index, to mark it unloaded afterwards
    const char* reason = kEvictIdleTtl;
    double idle_ms = 0.0;
    ModelResourceFootprint freed_footprint;
    double eviction_cost = 0.0;
};

struct ResidencySnapshot {
    std::string id;
    std::string backend;
    bool is_default = false;
    ResidencyState state = ResidencyState::unloaded;
    std::uint64_t pins = 0;           // requests holding the model now
    std::uint64_t loads = 0;          // successful loads
    std::uint64_t load_failures = 0;  // failed loads
    std::uint64_t evictions = 0;
    std::optional<double> idle_s;     // resident and unpinned: seconds since last use
    std::string model_instance_id;    // empty unless resident
    ModelResourceFootprint estimated_footprint;
    ModelResourceFootprint reserved_footprint;  // non-empty while loading
    ModelResourceFootprint observed_footprint;  // reconciled bytes while resident
    double reload_ms = 0.0;                     // measured load duration EMA (ms)
    double demand_score = 0.0;                  // decayed recent acquire demand
    double demand_rate_per_min = 0.0;           // recent demand rate (req/min)
    std::uint64_t admission_rejections = 0;
};

struct ResidencyTotals {
    std::size_t registered = 0;
    std::size_t resident = 0;
    std::size_t loading = 0;
    std::size_t unloading = 0;
    std::uint64_t loads = 0;
    std::uint64_t load_failures = 0;
    std::uint64_t evictions = 0;
    std::uint64_t resident_vram_bytes = 0;
    std::uint64_t resident_ram_bytes = 0;
    std::uint64_t reserved_vram_bytes = 0;
    std::uint64_t reserved_ram_bytes = 0;
    std::uint64_t admission_rejections = 0;
};

class ModelResidency : public std::enable_shared_from_this<ModelResidency> {
public:
    using Clock = std::chrono::steady_clock;
    // Loads model `id`. Called without any residency lock held.
    using Loader = std::function<Result<std::shared_ptr<Model>>(const std::string& id)>;
    // Releases an evicted model (engine unload, telemetry). Called without any
    // residency lock held, while the model is marked unloading; it becomes
    // unloaded when this returns. It may take (move out) `model`, so that the
    // residency's reference is gone before the engine unloads it.
    using Evictor = std::function<void(Eviction&)>;
    // Optional post-load footprint observer invoked without the residency lock
    // held right after `Loader` succeeds to reconcile the pre-load reservation
    // against observed allocations. Returning nullopt keeps the reservation.
    using FootprintObserver = std::function<std::optional<ModelResourceFootprint>(
        const std::string& id, const std::shared_ptr<Model>& model, const ModelResourceFootprint& reserved)>;
    // Optional dynamic pre-load footprint estimator invoked without the
    // residency lock held before reserving bytes for a cold load.
    using FootprintEstimator = std::function<std::optional<ModelResourceFootprint>(const std::string& id)>;

    ModelResidency(ResidencyConfig config, Loader loader, Evictor evictor, FootprintObserver observer = {},
                   FootprintEstimator estimator = {});
    ~ModelResidency();
    ModelResidency(const ModelResidency&) = delete;
    ModelResidency& operator=(const ModelResidency&) = delete;

    [[nodiscard]] const ResidencyConfig& config() const noexcept { return config_; }

    // Registers a model (no backend I/O). False when `id` is already known.
    bool add(const std::string& id, const std::string& backend, bool is_default,
             ModelResourceFootprint estimated = {});

    // Updates the pre-load estimated footprint for a registered model.
    bool set_estimated_footprint(const std::string& id, ModelResourceFootprint estimated);

    // Reconciles a resident model's active footprint against live observed
    // allocations (e.g. GPU memory probe, KV pool resize, adapter attach/detach).
    // If the updated footprint pushes active bytes over budget, evicts unpinned
    // models (never pinned models). Returns false if `id` is not registered or
    // not currently resident.
    bool reconcile_footprint(const std::string& id, ModelResourceFootprint observed,
                             Clock::time_point now = Clock::now());

    // Sets a measured/prior reload duration in milliseconds for `id` (ignores
    // negative or non-finite values). Returns false when `id` is unknown.
    bool set_reload_ms(const std::string& id, double reload_ms);

    // Records an explicit demand observation for `id` at `now` (ignores
    // negative or non-finite weights). Returns false when `id` is unknown.
    bool record_demand(const std::string& id, double weight = 1.0, Clock::time_point now = Clock::now());

    // Computes cost-per-byte eviction score:
    //   eviction_cost = (demand_score * reload_ms) / max(1, bytes_freed)
    // Lower scores are evicted first.
    [[nodiscard]] static double compute_eviction_cost(double demand_score, double reload_ms,
                                                      std::uint64_t bytes_freed) noexcept;

    // The registered id for `id_or_alias` ("default" names the default model).
    [[nodiscard]] std::optional<std::string> resolve(const std::string& id_or_alias) const;

    // A request's hold on a loaded model. Move-only; releasing it (destructor
    // or release()) makes the model evictable again.
    class Pin {
    public:
        Pin() = default;
        Pin(Pin&& other) noexcept;
        Pin& operator=(Pin&& other) noexcept;
        Pin(const Pin&) = delete;
        Pin& operator=(const Pin&) = delete;
        ~Pin();

        [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }
        [[nodiscard]] const std::shared_ptr<Model>& model() const noexcept { return model_; }
        [[nodiscard]] const std::string& id() const noexcept { return id_; }
        [[nodiscard]] const std::string& backend() const noexcept { return backend_; }
        void release() noexcept;

    private:
        friend class ModelResidency;
        Pin(std::shared_ptr<ModelResidency> owner, std::size_t index, std::shared_ptr<Model> model, std::string id,
            std::string backend);

        std::shared_ptr<ModelResidency> owner_;
        std::size_t index_ = 0;
        std::shared_ptr<Model> model_;
        std::string id_;
        std::string backend_;
    };

    // Pins `id_or_alias`, loading it first when it is not resident. A load
    // already in progress is joined, not repeated; its failure is returned
    // to every request that joined it. Errors: not_found (not registered),
    // unavailable (byte budget cannot admit alongside pinned/loading models),
    // or the loader's status.
    Result<Pin> acquire(const std::string& id_or_alias);

    // Evicts models idle for at least the TTL at `now`. Returns how many.
    std::size_t sweep(Clock::time_point now = Clock::now());

    // Runs sweep() on a background thread whenever the next idle deadline
    // passes (no thread when idle_ttl is 0). stop_sweeper() joins it.
    void start_sweeper();
    void stop_sweeper();

    [[nodiscard]] std::vector<ResidencySnapshot> snapshot(Clock::time_point now = Clock::now()) const;
    [[nodiscard]] ResidencyTotals totals() const;

    // The loaded model, or the descriptor it had when it was last loaded
    // (kept across evictions); nullopt when it has never loaded.
    [[nodiscard]] std::shared_ptr<Model> resident_model(const std::string& id) const;
    [[nodiscard]] std::optional<ModelDescriptor> last_descriptor(const std::string& id) const;

    // Shutdown: drops every handle without calling the evictor.
    void clear();

private:
    struct Entry {
        std::string id;
        std::string backend;
        bool is_default = false;
        ResidencyState state = ResidencyState::unloaded;
        std::shared_ptr<Model> model;
        std::optional<ModelDescriptor> descriptor;
        std::uint64_t pins = 0;
        std::uint64_t loads = 0;
        std::uint64_t load_failures = 0;
        std::uint64_t evictions = 0;
        std::uint64_t admission_rejections = 0;
        std::uint64_t load_generation = 0;    // incremented when a load starts
        std::uint64_t failed_generation = 0;  // generation of the last failed load
        Status last_error;
        Clock::time_point last_used{};
        ModelResourceFootprint estimated_footprint;
        ModelResourceFootprint reserved_footprint;
        ModelResourceFootprint observed_footprint;
        ModelResourceFootprint last_observed_footprint;
        bool has_last_observed = false;
        double reload_ms = 0.0;
        double demand_score = 0.0;
        Clock::time_point demand_updated_at{};
        bool has_demand_timestamp = false;
    };

    [[nodiscard]] bool has_byte_budget() const noexcept {
        return config_.max_resident_vram_bytes > 0 || config_.max_resident_ram_bytes > 0;
    }
    [[nodiscard]] double decayed_demand_locked(const Entry& e, Clock::time_point now) const noexcept;
    [[nodiscard]] double demand_rate_per_min_locked(double decayed_demand) const noexcept;
    void bump_demand_locked(Entry& e, double weight, Clock::time_point now) noexcept;
    void active_bytes_locked(std::uint64_t& vram_out, std::uint64_t& ram_out) const noexcept;
    void release(std::size_t index) noexcept;
    std::optional<std::size_t> find_locked(const std::string& id_or_alias) const;
    // Picks unpinned resident models until at most `keep` models are loaded or
    // loading (excluding `except`). Uses LRU without byte budgets, or cost-per-
    // byte ordering when byte budgets are configured.
    void collect_over_cap_locked(std::size_t keep, std::optional<std::size_t> except, Clock::time_point now,
                                 std::vector<Eviction>& out);
    // Picks unpinned resident victims optimizing eviction_cost until active
    // VRAM and RAM bytes fit within configured budgets (or no unpinned victim
    // with bytes on a pressured tier remains).
    void collect_over_budget_locked(std::optional<std::size_t> except, Clock::time_point now,
                                    std::vector<Eviction>& out);
    Eviction evict_locked(Entry& e, const char* reason, Clock::time_point now, double eviction_cost = 0.0);
    void run_evictions(std::vector<Eviction>& evictions) noexcept;
    std::optional<Clock::time_point> next_deadline_locked() const;
    void sweeper_loop();

    const ResidencyConfig config_;
    const Loader loader_;
    const Evictor evictor_;
    const FootprintObserver observer_;
    const FootprintEstimator estimator_;

    mutable std::mutex mu_;
    std::condition_variable load_cv_;
    std::vector<Entry> entries_;
    std::uint64_t total_loads_ = 0;
    std::uint64_t total_load_failures_ = 0;
    std::uint64_t total_evictions_ = 0;
    std::uint64_t total_admission_rejections_ = 0;

    std::condition_variable sweep_cv_;
    bool sweeper_stop_ = false;
    std::thread sweeper_;
};

}  // namespace sonder::inference::server::detail
