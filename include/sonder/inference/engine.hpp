// Sonder Inference: engine (device inventory, backend registry, model
// registry, sessions, telemetry).
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
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

struct EngineOptions {
    TelemetryOptions telemetry;
    // Attached before any engine event is emitted.
    std::vector<std::shared_ptr<TelemetrySink>> telemetry_sinks;
    // Emit a device.memory.sample for each device at startup.
    bool sample_devices_on_start = true;
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

    // Engine-scope telemetry context (session_id = engine id).
    [[nodiscard]] TelemetryContext engine_context() const;

private:
    EngineOptions options_;
    std::string engine_id_;
    std::unique_ptr<TelemetryBus> telemetry_;
    std::vector<DeviceInfo> devices_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Backend>> backends_;
    std::map<std::string, std::shared_ptr<Model>> models_;
};

}  // namespace sonder::inference
