// Sonder Inference: engine (device inventory, backend registry, model
// registry, sessions, telemetry).
#pragma once

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/model.hpp"
#include "sonder/inference/session.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference {

// Library version, e.g. "0.1.0".
const char* version_string() noexcept;
// Git commit the library was configured from, or "unknown".
const char* build_commit() noexcept;

// Request scheduling and logical KV accounting. Effective only when the
// scheduler (src/scheduler) and KV-cache (src/cache) modules are built; see
// Engine::scheduling_active().
struct SchedulingOptions {
    bool enabled = true;
    // Logical KV pool: kv_num_blocks blocks of kv_block_size_tokens tokens.
    std::uint32_t kv_block_size_tokens = 16;
    std::uint32_t kv_num_blocks = 4096;
    // Keep full blocks addressable by prefix after their request ends.
    bool prefix_caching = true;
    // Continuous-batching budgets (SchedulerConfig).
    std::uint32_t max_running_sequences = 64;
    std::uint32_t max_step_sequences = 64;
    std::uint32_t max_step_tokens = 2048;
    std::uint32_t prefill_chunk_tokens = 512;
    std::uint32_t admission_watermark_blocks = 4;
    // A request preempted more than this many times fails (ErrorCode::unavailable).
    std::uint32_t max_requeue_count = 8;
    bool enable_priority_preemption = true;
    // Longest a scheduling step waits for a granted request to produce its
    // token (0 = 1 ms). A request that misses it is left out of the step
    // barrier until it produces the token, so one stalled request (a
    // reasoning model thinking, a cold model load, a session blocked on a
    // slow client) delays the others by at most this much.
    std::uint32_t step_stall_timeout_ms = 250;
};

// Snapshot of the engine's logical KV pool (all zero when inactive).
struct KvUsage {
    bool active = false;
    std::uint32_t block_size_tokens = 0;
    std::uint64_t total_blocks = 0;
    std::uint64_t free_blocks = 0;    // never used or fully released
    std::uint64_t cached_blocks = 0;  // unreferenced, reusable by prefix, evictable
    std::uint64_t pinned_blocks = 0;  // referenced by a live request
    std::uint64_t shared_blocks = 0;  // referenced by more than one request
    std::uint64_t sequences = 0;      // live request sequences
    std::uint64_t prefix_hit_blocks = 0;
    std::uint64_t avoided_prefill_tokens = 0;
    std::uint64_t evictions = 0;
};

namespace detail {
class RequestRuntime;
}

// Network listener that hosts the engine (`sonder-infer serve`). When set,
// engine.started carries it as attributes.server {host, port, api_version}.
struct EngineServerInfo {
    std::string host;
    std::uint16_t port = 0;
    int api_version = 1;
};

struct EngineOptions {
    TelemetryOptions telemetry;
    // Optional: reported on engine.started (docs/TELEMETRY.md).
    std::optional<EngineServerInfo> server;
    // Attached before any engine event is emitted.
    std::vector<std::shared_ptr<TelemetrySink>> telemetry_sinks;
    // Emit a device.memory.sample for each device at startup.
    bool sample_devices_on_start = true;
    // Periodic device.memory.sample for every device (0 disables).
    std::chrono::milliseconds device_sample_interval{10000};
    SchedulingOptions scheduling;
};

class Engine {
public:
    explicit Engine(EngineOptions options = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Correlation scope for engine-level events (model load/unload, device
    // samples); the envelope requires a session_id on every event.
    [[nodiscard]] const std::string& engine_id() const noexcept { return engine_id_; }
    [[nodiscard]] TelemetryBus& telemetry() noexcept { return *telemetry_; }
    [[nodiscard]] const std::vector<DeviceInfo>& devices() const noexcept { return devices_; }

    Status register_backend(std::shared_ptr<Backend> backend);
    [[nodiscard]] std::shared_ptr<Backend> find_backend(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> backend_names() const;

    Result<std::shared_ptr<Model>> load_model(const std::string& backend_name, const ModelLoadOptions& options);
    Status unload_model(const std::string& model_instance_id);
    [[nodiscard]] std::vector<std::shared_ptr<Model>> loaded_models() const;

    Result<std::shared_ptr<Session>> create_session(const std::shared_ptr<Model>& model, SessionOptions options = {});

    // Engine-scope telemetry context (session_id = run_id = engine id).
    [[nodiscard]] TelemetryContext engine_context() const;

    // True when requests go through the scheduler and the logical KV cache.
    [[nodiscard]] bool scheduling_active() const noexcept { return runtime_ != nullptr; }
    [[nodiscard]] KvUsage kv_usage() const;

    // Internal: request runtime used by Session (null when inactive).
    [[nodiscard]] detail::RequestRuntime* request_runtime() noexcept { return runtime_.get(); }

private:
    void sample_devices(const std::vector<DeviceInfo>& devices);
    // backend.gpu_memory.sample / backend.warning for backends that report
    // Backend::runtime_status() (sampler thread only).
    void sample_backend_runtime();
    void device_sampler_loop();

    EngineOptions options_;
    std::string engine_id_;
    std::unique_ptr<TelemetryBus> telemetry_;
    std::vector<DeviceInfo> devices_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Backend>> backends_;
    std::map<std::string, std::shared_ptr<Model>> models_;
    // Declared last: stopped before the telemetry bus and registries go away.
    std::unique_ptr<detail::RequestRuntime> runtime_;
    // Periodic device sampler.
    std::mutex sampler_mutex_;
    std::condition_variable sampler_cv_;
    bool sampler_stop_ = false;
    // Sampler thread only: last reported GPU sample count and the warnings
    // already emitted, per backend name.
    std::map<std::string, std::uint64_t> runtime_samples_seen_;
    std::map<std::string, std::vector<std::string>> runtime_warnings_seen_;
    std::thread sampler_;
};

}  // namespace sonder::inference
