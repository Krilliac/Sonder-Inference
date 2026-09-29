// Internal: model residency for `sonder-infer serve` (docs/SERVER.md "Model
// residency").
//
// Every served model is registered up front (id, backend, default flag; no
// backend I/O). A request acquires a Pin: the first acquire loads the model
// (concurrent first requests share one load), and while any Pin is alive the
// model is never evicted. Unpinned models may be evicted after an idle TTL or
// to honour a max-resident cap; the next acquire loads them again.
//
// Defaults (idle_ttl 0, max_resident 0) never evict, so a model that was
// loaded stays loaded for the life of the server, as before lazy residency.
//
// Thread safety: every member is safe to call from any thread. The loader and
// the evictor are called without the residency lock held (they do backend
// work and emit telemetry). An eviction may still be running its evictor when
// a new request starts the next load of the same model: the two work on
// different model instances.
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

namespace sonder::inference::server::detail {

enum class ResidencyState { unloaded, loading, resident };

// "unloaded", "loading", "resident".
const char* residency_state_name(ResidencyState state) noexcept;

struct ResidencyConfig {
    // Evict a model after it has been unpinned this long; 0 = never.
    std::chrono::milliseconds idle_ttl{0};
    // Evict least recently used unpinned models to keep at most this many
    // loaded or loading; 0 = no cap. A soft cap: a pinned model is never
    // evicted, so pinned models alone can exceed it until they are released.
    std::size_t max_resident = 0;
};

// Why a model was evicted (telemetry `model.evicted` attribute `reason`).
inline constexpr const char* kEvictIdleTtl = "idle_ttl";
inline constexpr const char* kEvictMaxResident = "max_resident";

struct Eviction {
    std::string id;
    std::string backend;
    std::shared_ptr<Model> model;
    const char* reason = kEvictIdleTtl;
    double idle_ms = 0.0;
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
};

struct ResidencyTotals {
    std::size_t registered = 0;
    std::size_t resident = 0;
    std::size_t loading = 0;
    std::uint64_t loads = 0;
    std::uint64_t load_failures = 0;
    std::uint64_t evictions = 0;
};

class ModelResidency : public std::enable_shared_from_this<ModelResidency> {
public:
    using Clock = std::chrono::steady_clock;
    // Loads model `id`. Called without any residency lock held.
    using Loader = std::function<Result<std::shared_ptr<Model>>(const std::string& id)>;
    // Releases an evicted model (engine unload, telemetry). Called without any
    // residency lock held, after the model was marked unloaded. It may take
    // (move out) `model`, so that the residency's reference is gone before
    // the engine unloads it.
    using Evictor = std::function<void(Eviction&)>;

    ModelResidency(ResidencyConfig config, Loader loader, Evictor evictor);
    ~ModelResidency();
    ModelResidency(const ModelResidency&) = delete;
    ModelResidency& operator=(const ModelResidency&) = delete;

    [[nodiscard]] const ResidencyConfig& config() const noexcept { return config_; }

    // Registers a model (no backend I/O). False when `id` is already known.
    bool add(const std::string& id, const std::string& backend, bool is_default);

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
        std::uint64_t load_generation = 0;    // incremented when a load starts
        std::uint64_t failed_generation = 0;  // generation of the last failed load
        Status last_error;
        Clock::time_point last_used{};
    };

    void release(std::size_t index) noexcept;
    std::optional<std::size_t> find_locked(const std::string& id_or_alias) const;
    // Picks least recently used unpinned resident models until at most
    // `keep` models are loaded or loading (excluding `except`).
    void collect_over_cap_locked(std::size_t keep, std::optional<std::size_t> except, Clock::time_point now,
                                 std::vector<Eviction>& out);
    Eviction evict_locked(Entry& e, const char* reason, Clock::time_point now);
    void run_evictions(std::vector<Eviction>& evictions) noexcept;
    std::optional<Clock::time_point> next_deadline_locked() const;
    void sweeper_loop();

    const ResidencyConfig config_;
    const Loader loader_;
    const Evictor evictor_;

    mutable std::mutex mu_;
    std::condition_variable load_cv_;
    std::vector<Entry> entries_;
    std::uint64_t total_loads_ = 0;
    std::uint64_t total_load_failures_ = 0;
    std::uint64_t total_evictions_ = 0;

    std::condition_variable sweep_cv_;
    bool sweeper_stop_ = false;
    std::thread sweeper_;
};

}  // namespace sonder::inference::server::detail
