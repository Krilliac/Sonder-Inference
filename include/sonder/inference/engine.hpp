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

// How a scheduled request is paced (docs/integration/engine-wiring.md).
//   gate:      every generated chunk waits for a scheduler grant, and KV is
//              accounted for prompt and output (continuous batching of an
//              in-process backend).
//   account:   admission, priority and logical prefix accounting of the
//              prompt, then the backend streams without per-chunk waits. A
//              prompt larger than the logical KV pool runs unscheduled
//              instead of being rejected (the real KV lives upstream).
//   automatic: gate for in-process backends, account for backends that
//              declare Capability::remote_process (llamaserver, ollama).
enum class SchedulerMode { automatic, gate, account };
const char* to_string(SchedulerMode mode) noexcept;

// Request scheduling and logical KV accounting. Effective only when the
// scheduler (src/scheduler) and KV-cache (src/cache) modules are built; see
// Engine::scheduling_active().
struct SchedulingOptions {
    bool enabled = true;
    SchedulerMode mode = SchedulerMode::automatic;
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
    // Anti-starvation cap: shield repeatedly preempted requests from priority
    // preemption once they reach this requeue count (0 disables).
    std::uint32_t anti_starvation_preemption_cap = 3;
    bool enable_priority_preemption = true;
    // Cost-aware preemption policy (Recompute default; SpillToCheckpoint when
    // enabled and spill+restore cost < recompute cost within checkpoint budget).
    bool enable_spill_preemption = false;
    std::uint32_t spill_min_tokens = 1024;
    std::uint64_t kv_bytes_per_token = 4096;
    double prefill_tokens_per_sec = 2000.0;
    double spill_bandwidth_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    double restore_bandwidth_bytes_per_sec = 4.0 * 1024.0 * 1024.0 * 1024.0;
    // Longest a scheduling step waits for a granted request to produce its
    // token (0 = 1 ms). A request that misses it is left out of the step
    // barrier until it produces the token, so one stalled request (a
    // reasoning model thinking, a cold model load, a session blocked on a
    // slow client) delays the others by at most this much.
    std::uint32_t step_stall_timeout_ms = 250;
    // Hosted server opt-in. When enabled, admission uses strict
    // interactive > subagent > background ordering without aging, lookahead,
    // or urgency preemption. Legacy library scheduling remains unchanged.
    bool strict_priority_admission = false;
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
    // Bounded, TTL-expiring store for caller-visible execution checkpoints.
    ExecutionCheckpointStoreOptions checkpoints;
    // Enumerate live GPUs at startup and emit GPU device.memory.sample events.
    bool sample_gpu_devices = true;
};

// Pure builder for GPU `device.memory.sample` event attributes
// (`sonder.observatory.event/1`). Exposed for deterministic cross-platform
// testing of discrete and integrated UMA GPU telemetry payloads.
[[nodiscard]] json::Object gpu_memory_sample_attributes(const GpuDeviceInfo& gpu);

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
    [[nodiscard]] std::vector<GpuDeviceInfo> gpu_devices() const;

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
    [[nodiscard]] const SchedulingOptions& scheduling_options() const noexcept { return options_.scheduling; }
    [[nodiscard]] KvUsage kv_usage() const;

    // Internal: request runtime used by Session (null when inactive).
    [[nodiscard]] detail::RequestRuntime* request_runtime() noexcept { return runtime_.get(); }

    // Bounded, owned, expirable store for caller-visible execution checkpoints.
    [[nodiscard]] CheckpointStore& checkpoint_store() noexcept { return checkpoint_store_; }
    [[nodiscard]] const CheckpointStore& checkpoint_store() const noexcept { return checkpoint_store_; }

private:
    void sample_devices(const std::vector<DeviceInfo>& devices);
    void sample_gpu_devices(const std::vector<GpuDeviceInfo>& gpus);
    // backend.gpu_memory.sample / backend.warning for backends that report
    // Backend::runtime_status() (sampler thread only).
    void sample_backend_runtime();
    void device_sampler_loop();

    EngineOptions options_;
    std::string engine_id_;
    std::unique_ptr<TelemetryBus> telemetry_;
    std::vector<DeviceInfo> devices_;

    mutable std::mutex gpu_devices_mutex_;
    std::vector<GpuDeviceInfo> gpu_devices_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Backend>> backends_;
    std::map<std::string, std::shared_ptr<Model>> models_;
    // Declared last: stopped before the telemetry bus and registries go away.
    std::unique_ptr<detail::RequestRuntime> runtime_;
    // Destroyed before runtime_ so eviction callbacks can safely release logical KV.
    CheckpointStore checkpoint_store_;
    // Periodic device sampler.
    std::mutex sampler_mutex_;
    std::condition_variable sampler_cv_;
    bool sampler_stop_ = false;
    // Sampler thread only: last reported GPU sample count and the warnings
    // already emitted, per backend name.
    std::map<std::string, std::uint64_t> runtime_samples_seen_;
    std::map<std::string, std::vector<std::string>> runtime_warnings_seen_;
    std::map<std::string, std::pair<std::uint64_t, std::vector<std::uint32_t>>> runtime_warmups_seen_;
    std::thread sampler_;
};

}  // namespace sonder::inference
