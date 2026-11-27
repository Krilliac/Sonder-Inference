// Internal: read-only status, model listing, backend identity, node capability
// inventory, safe placement planning, discovery, and SSE/NDJSON live telemetry
// streaming endpoints for the server module.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <variant>

#include "http.hpp"
#include "json_schema_gbnf.hpp"
#include "live_hub.hpp"
#include "common/log_queue.hpp"
#include "residency.hpp"
#include "server_policy.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/placement_planner.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

json::Object sonder_meta();

// The launch profile bound to served model `id`, if any (validate_options
// guarantees at most one per model).
const ModelProfileBinding* find_profile_binding(const ServerOptions& opts, const std::string& id);

// Load options for served model `id`: the profile's GGUF path and device
// allowlist (a non-empty allowlist forces a gpu:* device_id, since cpu:* ids
// offload no layers), and --device otherwise.
ModelLoadOptions make_model_load_options(const ServerOptions& opts, const std::string& id);

// When the backend reports a fitted --ctx-size, the served profile reports
// that running value as `context_length` and keeps the configured one as
// `configured_context_length`. Unchanged otherwise.
json::Object profile_with_runtime_context(json::Object profile, const std::optional<BackendRuntimeStatus>& runtime);

// A lazy load that failed before anything executed.
ApiError load_failure_error(const std::string& id, const Status& st);

json::Array build_health_models(const ModelResidency& residency);
json::Object build_residency_summary(const ServerOptions& opts, const ModelResidency& residency);

struct HealthSnapshot {
    const ServerOptions& opts;
    State state = State::starting;
    bool backend_available = false;
    std::string backend_version;
    std::shared_ptr<Backend> backend;
    const ModelResidency& residency;
    HubStats hub_stats;
    std::uint64_t telemetry_emitted = 0;
    std::uint64_t telemetry_dropped = 0;
    std::array<std::uint64_t, 3> queued_by_class{0, 0, 0};
    LogQueueStats log_stats;
    const std::string& instance_id;
    const std::string& node_id;
    std::chrono::steady_clock::time_point started_at;
    bool synthetic = false;
    CompiledGrammarCacheStats grammar_cache{};
};

json::Object build_health_body(const HealthSnapshot& snapshot);

json::Object build_models_body(const ServerOptions& opts, const ModelResidency& residency,
                               const std::shared_ptr<Backend>& backend, bool synthetic);

json::Object build_identity_body(const ModelResidency& residency, const std::string& served_id, bool synthetic,
                                 const std::optional<std::string>& probed);

struct NodeSnapshot {
    const ServerOptions& opts;
    bool backend_available = false;
    std::string backend_version;
    std::shared_ptr<Backend> backend;
    const ModelResidency& residency;
    const std::string& instance_id;
    const std::string& node_id;
    bool synthetic = false;
};

json::Object build_node_body(const NodeSnapshot& snapshot);

struct PlanHttpRequest {
    bool has_profile = false;
    std::string profile;
    bool has_model = false;
    std::string model;
    PlacementPlanRequest plan_request;
};

// Strictly parses and validates a POST /v1/sonder/plan JSON request body.
std::variant<PlanHttpRequest, ApiError> parse_plan_request(std::string_view body);

// Resolves the requested profile or model strictly inside configured server
// bindings or model_dirs, reads only the GGUF header, runs the placement
// planner, and redacts any host paths to basenames before returning JSON.
std::variant<json::Object, ApiError> execute_plan_request(const ServerOptions& opts, const PlanHttpRequest& request);

json::Object build_discovery_document(const ServerOptions& opts, const LiveTelemetryHub& hub,
                                      const std::string& instance_id, const std::string& node_id, bool synthetic);

// Periodic off-request backend reachability refresher. Health and identity read
// the cached snapshot without ever blocking on backend I/O.
class BackendProbeRefresher {
public:
    void status(bool& available, std::string& version) const;
    void record(const Result<std::string>& probed);
    void start(std::shared_ptr<Backend> backend);
    // Returns false when the refresher thread stayed blocked in backend I/O
    // beyond `patience` and was detached.
    bool stop(std::chrono::milliseconds patience);

private:
    std::shared_ptr<ProbeCache> cache_ = std::make_shared<ProbeCache>();
    std::thread thread_;
};

void handle_telemetry_stream(Exchange& ex, const RequestHead& head, Route route, ConnFlags& flags,
                             const ServerOptions& opts, LiveTelemetryHub& hub, const std::string& instance_id,
                             const std::atomic<bool>& hard_stop);

// Options-aware identity builder: resolves live `llamacpp` GGUF paths via
// profile bindings and `opts.backend.model_dirs` and reuses the server's
// cached GPU inventory.
json::Object build_identity_body(const ServerOptions& opts, const ModelResidency& residency,
                                 const std::string& served_id, bool synthetic,
                                 const std::optional<std::string>& probed,
                                 const std::shared_ptr<Backend>& backend = nullptr);

}  // namespace sonder::inference::server::detail
