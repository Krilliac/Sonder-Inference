#include "sonder/inference/backends/llamaserver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "adaptive_speculation.hpp"
#include "llamaserver_options.hpp"
#include "llamaserver_protocol.hpp"
#include "llamaserver_stream.hpp"
#include "net/http_client.hpp"
#include "prefix_warmup.hpp"
#include "slot_affinity.hpp"
#include "supervisor.hpp"

namespace sonder::inference {
namespace {

bool contains_code_or_diff_markers(std::string_view text) noexcept {
    return text.find("```") != std::string_view::npos ||
           text.find("diff --git") != std::string_view::npos ||
           text.find("@@ ") != std::string_view::npos;
}

bool chat_thinking_enabled(const ChatRequest &request) noexcept {
    if (request.separate_reasoning)
        return true;
    if (request.thinking.enable_thinking.has_value())
        return *request.thinking.enable_thinking;
    if (request.thinking.reasoning_effort.has_value() && !request.thinking.reasoning_effort->empty() &&
        *request.thinking.reasoning_effort != "none")
        return true;
    return false;
}

llamaserver::SpeculationWorkload classify_generate_workload(const GenerateRequest &request,
                                                            std::string_view grammar) noexcept {
    if (!grammar.empty() || !request.grammar.empty())
        return llamaserver::SpeculationWorkload::tool_json;
    if (contains_code_or_diff_markers(request.prompt))
        return llamaserver::SpeculationWorkload::code_edit;
    return llamaserver::SpeculationWorkload::general;
}

llamaserver::SpeculationWorkload classify_chat_workload(const ChatRequest &request,
                                                        std::string_view grammar) noexcept {
    if (!grammar.empty() || !request.grammar.empty() || llamaserver::chat_tools_requested(request) ||
        (request.response_format_type && *request.response_format_type != "text"))
        return llamaserver::SpeculationWorkload::tool_json;
    if (chat_thinking_enabled(request))
        return llamaserver::SpeculationWorkload::reasoning;
    for (auto it = request.messages.rbegin(); it != request.messages.rend(); ++it) {
        if (it->role == "user") {
            if (contains_code_or_diff_markers(it->content))
                return llamaserver::SpeculationWorkload::code_edit;
            break;
        }
    }
    return llamaserver::SpeculationWorkload::general;
}

class ProbeGuard {
  public:
    ProbeGuard(llamaserver::AdaptiveSpeculationPolicy *policy,
               llamaserver::SpeculationWorkload workload,
               bool active,
               std::uint64_t probe_id) noexcept
        : policy_(policy), workload_(workload), active_(active), probe_id_(probe_id) {}
    ~ProbeGuard() noexcept {
        if (active_ && policy_ != nullptr) {
            try {
                policy_->cancel_probe(workload_, probe_id_);
            } catch (...) {
            }
        }
    }
    ProbeGuard(const ProbeGuard &) = delete;
    ProbeGuard &operator=(const ProbeGuard &) = delete;
    void disarm() noexcept { active_ = false; }

  private:
    llamaserver::AdaptiveSpeculationPolicy *policy_ = nullptr;
    llamaserver::SpeculationWorkload workload_ = llamaserver::SpeculationWorkload::general;
    bool active_ = false;
    std::uint64_t probe_id_ = 0;
};

void record_speculation_observation(llamaserver::AdaptiveSpeculationPolicy *policy,
                                    const GenerateStats &stats,
                                    llamaserver::SpeculationWorkload workload,
                                    bool is_probe,
                                    std::uint64_t probe_id,
                                    ProbeGuard &probe_guard) {
    if (policy == nullptr || !stats.backend_timings.has_value())
        return;
    auto obs = llamaserver::observation_from_timings(*stats.backend_timings, workload);
    if (obs.decode_tps <= 0.0 && stats.predicted_tokens_per_second.has_value() &&
        std::isfinite(*stats.predicted_tokens_per_second) && *stats.predicted_tokens_per_second > 0.0) {
        obs.decode_tps = *stats.predicted_tokens_per_second;
    }
    obs.is_probe = is_probe;
    obs.probe_id = probe_id;
    probe_guard.disarm();
    policy->observe(obs);
}

class LlamaServerModel;
class LlamaServerBackendImpl final : public LlamaServerBackend,
                                     public std::enable_shared_from_this<LlamaServerBackendImpl> {
  public:
    explicit LlamaServerBackendImpl(LlamaServerBackendOptions options)
        : options_(std::move(options)), validation_(llamaserver::validate_backend_options(options_)) {
        if (validation_.ok() && options_.adaptive_speculation.enabled && options_.native_completion) {
            adaptive_spec_ = std::make_unique<llamaserver::AdaptiveSpeculationPolicy>(
                llamaserver::to_adaptive_speculation_config(options_.adaptive_speculation));
        }
        if (validation_.ok() && !options_.warmup.messages_file.empty())
            warmup_ = std::make_shared<llamaserver::PrefixWarmup>(options_);
        if (validation_.ok() && options_.mode == LlamaServerMode::spawn) {
            llamaserver::SupervisorOptions s;
            s.environment = options_.environment;
            s.executable = options_.executable;
            s.arguments = options_.args;
            s.readiness_timeout = options_.startup_timeout;
            s.health_poll_interval = options_.poll_interval;
            s.restart_initial_backoff = options_.restart_backoff;
            s.restart_max_backoff = options_.max_restart_backoff;
            s.shutdown_timeout = options_.shutdown_timeout;
            s.max_restarts = options_.max_restarts;
            s.spill_guard = llamaserver::to_spill_guard_options(options_.spill_guard, options_.args);
            (void)llamaserver::to_host_memory_options(options_.host_memory, s.host_memory);  // policy validated above
            s.log_file = options_.diagnostics.log_file;
            s.kv_pairing_check = options_.diagnostics.kv_pairing_check;
            s.warmup = warmup_;
            s.native_completion = options_.native_completion;
            s.stall_guard.enabled = options_.stall_guard.enabled;
            s.stall_guard.stall_seconds = options_.stall_guard.stall_seconds;
            s.stall_guard.policy = options_.stall_guard.policy;
            supervisor_ = std::make_unique<llamaserver::Supervisor>(std::move(s));
        }
        if (validation_.ok() && warmup_ && options_.mode == LlamaServerMode::attach)
            warmup_->start(options_.base_url, true);
        if (validation_.ok() && options_.native_completion && options_.mode == LlamaServerMode::attach) {
            // prepare preserves the validated URL and all existing TLS policy.
            const auto request = prepare("GET", "", {}, {});
            if (request.ok()) {
                llamaserver::SupervisorOptions s;
                s.attached_endpoint = request.value();
                s.arguments = options_.args;
                s.spill_guard.enabled = false;
                s.spill_guard.sample_interval = options_.spill_guard.sample_interval;
                s.kv_pairing_check = false;
                s.stall_guard.enabled = options_.stall_guard.enabled;
                s.stall_guard.stall_seconds = options_.stall_guard.stall_seconds;
                s.stall_guard.policy = options_.stall_guard.policy;
                attached_monitor_ = std::make_unique<llamaserver::Supervisor>(std::move(s));
                // Observer startup failure remains diagnostic, never changes requests.
                (void)attached_monitor_->start();
            }
        }
    }
    ~LlamaServerBackendImpl() override {
        if (supervisor_)
            supervisor_->stop();
        if (attached_monitor_)
            attached_monitor_->stop();
        if (warmup_)
            warmup_->stop();
    }
    std::string name() const override { return kLlamaServerBackendName; }
    std::string description() const override {
        return options_.mode == LlamaServerMode::spawn ? "Supervised external llama-server"
                                                       : "OpenAI-compatible HTTP upstream";
    }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming)
            .add(Capability::remote_process)
            .add(Capability::embeddings)
            .add(Capability::tool_calling)
            .add(Capability::multimodal)
            .add(Capability::logprobs);
        if (options_.upstream_grammar || !options_.grammar.empty())
            c.add(Capability::structured_output);
        // These indicate delegation support, not measured acceleration or KV ownership.
        if (options_.native_completion)
            c.add(Capability::speculative_decode).add(Capability::prefix_reuse);
        return c;
    }
    // /props is fetched during model discovery and cached per upstream
    // instance.  Admission may use this snapshot without issuing network I/O
    // from a const query; zero means that the upstream did not report slots.
    std::size_t max_concurrent_requests() const override {
        std::lock_guard<std::mutex> lock(props_mutex_);
        return props_fetched_ ? static_cast<std::size_t>(props_.total_slots) : 0;
    }
    Result<std::string> probe() override {
        auto models = list_models();
        if (!models.ok())
            return models.status();
        return std::string("OpenAI-compatible HTTP API");  // no fabricated upstream version
    }
    Result<std::vector<ModelDescriptor>> list_models() override {
        auto response = request("GET", "/v1/models", {});
        if (!response.ok())
            return response.status();
        auto parsed = json::parse(response.value());
        if (!parsed.ok())
            return llamaserver::protocol_error("malformed /v1/models response");
        const auto *data = parsed->find("data");
        if (!data || !data->is_array())
            return llamaserver::protocol_error("missing model data array");
        std::vector<ModelDescriptor> models;
        for (const auto &item : data->as_array()) {
            const auto *id = item.find("id");
            if (!id || !id->is_string() || id->as_string().empty())
                return llamaserver::protocol_error("missing model id");
            ModelDescriptor model;
            model.name = id->as_string();
            model.backend = kLlamaServerBackendName;
            model.resident = false;
            models.push_back(std::move(model));
        }
        return models;
    }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions &options) override;
    Status save_slot(std::uint32_t slot, const std::string &filename) override {
        return slot_action(slot, filename, "save");
    }
    Status restore_slot(std::uint32_t slot, const std::string &filename) override {
        return slot_action(slot, filename, "restore");
    }
    const LlamaServerBackendOptions &options() const { return options_; }
    llamaserver::AdaptiveSpeculationPolicy *adaptive_speculation() const noexcept { return adaptive_spec_.get(); }
    // Native attach observes the upstream without owning its process. Generic
    // OpenAI attach stays silent unless prefix warm-up was configured.
    std::optional<BackendRuntimeStatus> runtime_status() const override {
        const bool generic_report = options_.stall_guard.generic_progress || !options_.api_key_env.empty();
        if (!supervisor_ && !attached_monitor_ && !warmup_ && !adaptive_spec_ && !generic_report)
            return std::nullopt;
        auto status = supervisor_       ? supervisor_->runtime_status()
                      : attached_monitor_ ? attached_monitor_->runtime_status()
                                          : BackendRuntimeStatus{};
        if (!options_.api_key_env.empty()) {
            // Name of the variable only; the header value is always redacted.
            if (!status.diagnostics)
                status.diagnostics.emplace();
            status.diagnostics->set("upstream_auth",
                                    json::Object{{"header", "Authorization"},
                                                 {"api_key_env", options_.api_key_env},
                                                 {"value", net::redacted_header_value("Authorization", "")}});
        }
        if (adaptive_spec_) {
            if (!status.diagnostics)
                status.diagnostics.emplace();
            status.diagnostics->set("adaptive_speculation", adaptive_spec_->status_json());
        }
        if (options_.stall_guard.generic_progress) {
            std::lock_guard<std::mutex> lock(stall_mutex_);
            json::Object stall{{"generic_progress", true},
                               {"policy", options_.stall_guard.policy},
                               {"stall_seconds", static_cast<std::int64_t>(options_.stall_guard.stall_seconds.count())},
                               {"events", static_cast<std::int64_t>(stall_events_)}};
            status.stall = std::move(stall);
            if (stall_events_ != 0)
                status.warnings.push_back(stall_warning_);
        }
        if (warmup_) {
            status.warmup = warmup_->status();
            const auto warnings = warmup_->warnings();
            status.warnings.insert(status.warnings.end(), warnings.begin(), warnings.end());
        }
        return status;
    }

    Result<std::string> endpoint(const CancellationToken &cancel) {
        if (!validation_.ok())
            return validation_;
        if (cancel.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
        if (!supervisor_)
            return options_.base_url;
        auto port = supervisor_->start(cancel);
        if (!port.ok())
            return port.status();
        return "http://127.0.0.1:" + std::to_string(port.value());
    }
    Result<net::HttpRequest> prepare(std::string_view method, std::string_view path, std::string body,
                                     const CancellationToken &cancel, std::uint16_t leased_port = 0) {
        auto base = leased_port != 0 ? Result<std::string>("http://127.0.0.1:" + std::to_string(leased_port))
                                     : endpoint(cancel);
        if (!base.ok())
            return base.status();
        auto url = net::parse_url(base.value());
        if (!url.ok())
            return url.status();
        net::HttpRequest request;
        request.method = method;
        request.host = url->host;
        request.port = url->port;
        request.target = (url->path == "/" ? "" : url->path) + std::string(path);
        request.body = std::move(body);
        request.connect_timeout = options_.connect_timeout;
        request.total_timeout = options_.request_timeout;
        request.use_tls = url->scheme == "https";
        request.tls.ca_bundle_path = options_.tls.ca_bundle_path;
        request.tls.pinned_sha256 = options_.tls.pinned_sha256;
        request.tls.pinned_cert_path = options_.tls.pinned_cert_path;
        request.tls.insecure_skip_verify = options_.tls.insecure_skip_verify;
        request.tls.server_name = options_.tls.server_name;
        request.tls.handshake_timeout = options_.tls.handshake_timeout;
        if (!options_.api_key_env.empty() && !supervisor_) {
            // Read per request so a rotated token is picked up; never cached
            // in options_, never logged.
            auto key = llamaserver::read_api_key(options_.api_key_env);
            if (!key.ok())
                return key.status();
            request.headers.emplace_back("Authorization", "Bearer " + key.value());
        }
        return request;
    }
    Result<std::string> request(std::string_view method, std::string_view path, std::string body) {
        auto activity = acquire_request({});
        if (!activity.ok())
            return activity.status();
        auto prepared = prepare(method, path, std::move(body), {}, activity->port());
        if (!prepared.ok())
            return prepared.status();
        std::string response_body;
        const auto response = net::http_request_buffered(prepared.value(), response_body, {}, 4 * 1024 * 1024);
        if (!response.ok())
            return response.status();
        if (response->status != 200)
            return llamaserver::status_for_http(response->status);
        return response_body;
    }
    Result<std::string> request_embeddings(std::string body, const CancellationToken &cancel) {
        auto activity = acquire_request(cancel);
        if (!activity.ok())
            return activity.status();
        auto prepared = prepare("POST", "/v1/embeddings", std::move(body), cancel, activity->port());
        if (!prepared.ok())
            return prepared.status();
        std::string response_body;
        const auto response = net::http_request_buffered(prepared.value(), response_body, cancel, 64u * 1024u * 1024u);
        if (cancel.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
        if (!response.ok())
            return response.status();
        if (response->status == 501)
            return Status(ErrorCode::unsupported, "llamaserver: upstream does not support embeddings");
        if (response->status != 200)
            return llamaserver::status_for_http(response->status);
        return response_body;
    }
    Result<GenerateStats> stream(std::string_view path, std::string body, const CancellationToken &cancel,
                                 const TokenCallback &callback, bool chat, bool separate_reasoning = false,
                                 bool tools_requested = false) {
        auto activity = acquire_request(cancel);
        if (!activity.ok())
            return activity.status();
        auto prepared = prepare("POST", path, std::move(body), cancel, activity->port());
        if (!prepared.ok())
            return prepared.status();
        auto request = std::move(prepared.value());
        // Frames that carried content or reasoning text (visible to the caller or
        // not). Empty keepalive deltas never count. Drives the first-content
        // timeout and the generic stall watchdog.
        std::uint64_t progress_events = 0;
        if (options_.mode == LlamaServerMode::attach) {
            request.unlimited_total = options_.unlimited_request_timeout;
            request.idle_timeout = options_.idle_timeout;
            request.first_progress_timeout = options_.first_content_timeout;
            if (options_.stall_guard.generic_progress) {
                request.progress_stall_timeout = options_.stall_guard.stall_seconds;
                if (options_.stall_guard.policy == "warn")
                    request.on_progress_stall = [this](std::chrono::milliseconds stalled) { record_stall(stalled); };
            }
            if (request.first_progress_timeout.count() > 0 || request.progress_stall_timeout.count() > 0)
                request.progress_counter = [&progress_events] { return progress_events; };
        }
        const bool native = !chat && options_.native_completion;
        return llamaserver::run_completion_stream(request, chat, native, separate_reasoning, cancel, callback,
                                                  progress_events, tools_requested);
    }

    // Generic-mode progress watchdog (warn policy), called from the request
    // thread by the HTTP client. Folds repeats into one bounded warning.
    void record_stall(std::chrono::milliseconds stalled) {
        std::lock_guard<std::mutex> lock(stall_mutex_);
        ++stall_events_;
        if (stall_warning_.code.empty()) {
            stall_warning_.code = "generic_stream_stalled";
            stall_warning_.source = "config";
            stall_warning_.message =
                "an upstream stream produced no content or reasoning text for longer than stall_guard.stall_seconds "
                "(keepalive-only periods do not count as progress)";
        }
        stall_warning_.count = stall_events_;
        stall_warning_.details = {{"last_stalled_ms", std::to_string(stalled.count())}};
    }

    // Served context and slot count: GET /props once per upstream instance
    // (endpoint, plus the launch argv in spawn mode, so a restart or an
    // auto_fit relaunch re-reads it). A missing or malformed /props caches
    // "unknown" (0) for that instance; options.context_length fills in n_ctx.
    llamaserver::ServerProps served_props(const CancellationToken &cancel) {
        auto ep = endpoint(cancel);
        llamaserver::ServerProps props;
        if (ep.ok()) {
            std::string key = ep.value();
            if (supervisor_) {
                for (const auto &arg : supervisor_->launch_arguments()) {
                    key += '\n';
                    key += arg;
                }
            }
            bool cached = false;
            {
                std::lock_guard<std::mutex> lock(props_mutex_);
                if (props_fetched_ && props_key_ == key) {
                    props = props_;
                    cached = true;
                }
            }
            if (!cached) {
                bool definitive = false;
                props = fetch_props(cancel, definitive);
                // Transport failures, timeouts and transient statuses (e.g. 503
                // while loading) are retried on the next request, never cached.
                if (definitive && !cancel.cancelled()) {
                    std::lock_guard<std::mutex> lock(props_mutex_);
                    props_ = props;
                    props_key_ = std::move(key);
                    props_fetched_ = true;
                }
            }
        }
        if (props.n_ctx == 0)
            props.n_ctx = options_.context_length;
        return props;
    }
    // Request fields only llama.cpp understands; generic OpenAI mode gets none.
    llamaserver::RequestExtras extras(const llamaserver::ServerProps &props, std::string_view session_key,
                                      llamaserver::SlotAffinity::Lease &lease) {
        llamaserver::RequestExtras e;
        if (!options_.native_completion) {
            e.generic_thinking = true;
            return e;
        }
        e.cache_prompt = true;
        if (options_.slot_affinity) {
            lease = slots_.acquire(session_key, props.total_slots,
                                   warmup_ ? warmup_->warmed_slots() : std::vector<std::uint32_t>{});
            e.id_slot = lease.slot();
        }
        return e;
    }

  private:
    Result<llamaserver::Supervisor::RequestLease> acquire_request(const CancellationToken &cancel) {
        if (supervisor_)
            return supervisor_->acquire_request(cancel);
        return llamaserver::Supervisor::RequestLease{};
    }
    // `definitive` is set only when the upstream gave a final answer (200 or
    // 404); anything else leaves it false so the caller does not cache it.
    llamaserver::ServerProps fetch_props(const CancellationToken &cancel, bool &definitive) {
        definitive = false;
        llamaserver::ServerProps props;
        auto activity = acquire_request(cancel);
        if (!activity.ok())
            return props;
        auto prepared = prepare("GET", "/props", {}, cancel, activity->port());
        if (!prepared.ok())
            return props;
        prepared->total_timeout = std::min(options_.request_timeout, std::chrono::milliseconds(30000));
        std::string body;
        const auto response = net::http_request_buffered(prepared.value(), body, cancel, 1024 * 1024);
        if (!response.ok())
            return props;
        if (response->status == 404)
            definitive = true;
        if (response->status != 200)
            return props;
        definitive = true;
        auto parsed = json::parse(body);
        if (!parsed.ok() || !llamaserver::parse_props(parsed.value(), props).ok())
            return {};
        return props;
    }
    Status slot_action(std::uint32_t slot, const std::string &filename, std::string_view action) {
        if (!validation_.ok())
            return validation_;
        if (!options_.native_completion)
            return {ErrorCode::unsupported, "llamaserver: slot API requires native mode"};
        if (filename.empty() || filename == "." || filename == ".." || filename.size() > 255 ||
            filename.find_first_of("/\\:") != std::string::npos || filename.back() == '.' ||
            filename.back() == ' ' || std::any_of(filename.begin(), filename.end(), [](unsigned char c) {
                return c < 32 || c == 127;
            })) {
            return {ErrorCode::invalid_argument, "llamaserver: slot filename must be a basename"};
        }
        const auto response =
            request("POST", "/slots/" + std::to_string(slot) + "?action=" + std::string(action),
                    json::Value(json::Object{{"filename", filename}}).dump());
        if (!response.ok())
            return response.status();
        auto result = json::parse(response.value());
        if (!result.ok() || !result->is_object())
            return llamaserver::protocol_error("malformed slot response");
        if (result->find("error"))
            return {ErrorCode::backend_error, "llamaserver: slot operation failed"};
        return {};
    }
    LlamaServerBackendOptions options_;
    Status validation_;
    std::unique_ptr<llamaserver::Supervisor> supervisor_;
    std::unique_ptr<llamaserver::Supervisor> attached_monitor_;
    mutable std::mutex props_mutex_;
    bool props_fetched_ = false;
    std::string props_key_;
    llamaserver::ServerProps props_;
    llamaserver::SlotAffinity slots_;
    std::shared_ptr<llamaserver::PrefixWarmup> warmup_;
    std::unique_ptr<llamaserver::AdaptiveSpeculationPolicy> adaptive_spec_;
    mutable std::mutex stall_mutex_;
    std::uint64_t stall_events_ = 0;
    BackendWarning stall_warning_;
};

class LlamaServerModel final : public BackendModel {
  public:
    LlamaServerModel(std::shared_ptr<LlamaServerBackendImpl> backend, ModelDescriptor descriptor)
        : backend_(std::move(backend)), descriptor_(std::move(descriptor)) {}
    const ModelDescriptor &descriptor() const override { return descriptor_; }
    bool has_native_chat() const override { return true; }
    Result<GenerateStats> generate(const GenerateRequest &request, const CancellationToken &cancel,
                                   const TokenCallback &callback) override {
        if (auto status = validate(request.sampling); !status.ok())
            return status;
        const auto props = backend_->served_props(cancel);
        if (auto status = llamaserver::validate_sampling(request.sampling, props.n_ctx); !status.ok())
            return status;
        const bool native = backend_->options().native_completion;
        const auto workload = classify_generate_workload(request, backend_->options().grammar);
        auto *policy = backend_->adaptive_speculation();
        bool has_spec_decision = false;
        llamaserver::SpeculationDecision spec_decision;
        if (policy != nullptr && !cancel.cancelled()) {
            spec_decision = policy->recommend(workload);
            has_spec_decision = true;
        }
        ProbeGuard probe_guard(policy, workload, has_spec_decision && spec_decision.is_probe,
                               has_spec_decision ? spec_decision.probe_id : 0u);
        llamaserver::SlotAffinity::Lease lease;
        auto req_extras = backend_->extras(props, {}, lease);
        if (has_spec_decision)
            req_extras.spec_draft_n_max = spec_decision.draft_n_max;
        auto body = llamaserver::build_completion_body(descriptor_.name, request, native, backend_->options().grammar,
                                                       req_extras);
        auto result =
            backend_->stream(native ? "/completion" : "/v1/completions", body.dump(), cancel, callback, false);
        if (result.ok() && has_spec_decision) {
            record_speculation_observation(policy, result.value(), workload, spec_decision.is_probe,
                                           spec_decision.probe_id, probe_guard);
        }
        return result;
    }
    Result<GenerateStats> chat(const ChatRequest &request, const CancellationToken &cancel,
                               const TokenCallback &callback) override {
        if (auto status = validate_chat_messages(request.messages); !status.ok())
            return status;
        if (auto status = validate(request.sampling); !status.ok())
            return status;
        const auto props = backend_->served_props(cancel);
        if (auto status = llamaserver::validate_sampling(request.sampling, props.n_ctx); !status.ok())
            return status;
        const auto workload = classify_chat_workload(request, backend_->options().grammar);
        auto *policy = backend_->adaptive_speculation();
        bool has_spec_decision = false;
        llamaserver::SpeculationDecision spec_decision;
        if (policy != nullptr && !cancel.cancelled()) {
            spec_decision = policy->recommend(workload);
            has_spec_decision = true;
        }
        ProbeGuard probe_guard(policy, workload, has_spec_decision && spec_decision.is_probe,
                               has_spec_decision ? spec_decision.probe_id : 0u);
        // The lease keeps the slot marked busy until the stream ends.
        llamaserver::SlotAffinity::Lease lease;
        auto req_extras = backend_->extras(props, request.session_key, lease);
        if (has_spec_decision)
            req_extras.spec_draft_n_max = spec_decision.draft_n_max;
        const bool tools_requested = llamaserver::chat_tools_requested(request);
        auto body = llamaserver::build_chat_body(descriptor_.name, request, backend_->options().grammar, req_extras);
        auto result = backend_->stream("/v1/chat/completions", body.dump(), cancel, callback, true,
                                       request.separate_reasoning, tools_requested);
        if (result.ok() && has_spec_decision) {
            record_speculation_observation(policy, result.value(), workload, spec_decision.is_probe,
                                           spec_decision.probe_id, probe_guard);
        }
        return result;
    }
    Result<EmbedResult> embed(const EmbedRequest &request, const CancellationToken &cancel) override {
        if (cancel.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
        if (request.inputs.empty())
            return Status(ErrorCode::invalid_argument, "llamaserver: embed inputs must be non-empty");
        json::Array input_arr;
        input_arr.reserve(request.inputs.size());
        for (const auto &text : request.inputs)
            input_arr.emplace_back(text);
        json::Object req_body{{"model", descriptor_.name}, {"input", std::move(input_arr)}};
        const auto t0 = std::chrono::steady_clock::now();
        auto response = backend_->request_embeddings(json::Value(std::move(req_body)).dump(), cancel);
        if (!response.ok())
            return response.status();
        auto parsed = json::parse(response.value());
        if (!parsed.ok() || !parsed->is_object())
            return llamaserver::protocol_error("malformed /v1/embeddings response");
        const auto *data = parsed->find("data");
        if (!data || !data->is_array())
            return llamaserver::protocol_error("missing data array in /v1/embeddings response");
        const auto &items = data->as_array();
        if (items.size() != request.inputs.size())
            return llamaserver::protocol_error("embedding count mismatch in /v1/embeddings response");
        EmbedResult result;
        result.embeddings.resize(items.size());
        std::vector<bool> populated(items.size(), false);
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto &item = items[i];
            if (!item.is_object())
                return llamaserver::protocol_error("malformed embedding entry");
            std::size_t slot = i;
            if (const auto *idx = item.find("index"); idx && idx->is_integer()) {
                const std::int64_t v = idx->as_int(-1);
                if (v < 0 || static_cast<std::size_t>(v) >= items.size())
                    return llamaserver::protocol_error("out-of-range embedding index");
                slot = static_cast<std::size_t>(v);
            }
            if (populated[slot])
                return llamaserver::protocol_error("duplicate embedding index");
            const auto *emb = item.find("embedding");
            if (!emb || !emb->is_array())
                return llamaserver::protocol_error("missing embedding vector");
            std::vector<float> vec;
            vec.reserve(emb->as_array().size());
            for (const auto &num : emb->as_array()) {
                if (!num.is_number() || !std::isfinite(num.as_double()))
                    return llamaserver::protocol_error("non-finite value in embedding vector");
                vec.push_back(static_cast<float>(num.as_double()));
            }
            result.embeddings[slot] = std::move(vec);
            populated[slot] = true;
        }
        if (const auto *usage = parsed->find("usage"); usage && usage->is_object()) {
            if (const auto *pt = usage->find("prompt_tokens"); pt && pt->is_integer() && pt->as_int(-1) >= 0) {
                result.prompt_tokens = pt->as_uint();
            }
        }
        result.eval_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        return result;
    }

  private:
    std::shared_ptr<LlamaServerBackendImpl> backend_;
    ModelDescriptor descriptor_;
};

Result<std::shared_ptr<BackendModel>> LlamaServerBackendImpl::load_model(const ModelLoadOptions &options) {
    if (!validation_.ok())
        return validation_;
    if (options.model.empty())
        return Status(ErrorCode::invalid_argument, "llamaserver: model is required");
    // Verify readiness and the alias through discovery; never download or load
    // an unrelated model implicitly on a generic OpenAI upstream.
    auto models = list_models();
    if (!models.ok())
        return models.status();
    for (const auto &model : models.value()) {
        if (model.name == options.model) {
            // Readiness is proven: read the served context and slot count now,
            // so the first request does not pay for it.
            ModelDescriptor descriptor = model;
            descriptor.context_length = served_props({}).n_ctx;
            return std::shared_ptr<BackendModel>(
                std::make_shared<LlamaServerModel>(shared_from_this(), std::move(descriptor)));
        }
    }
    return Status(ErrorCode::not_found, "llamaserver: model is not advertised by upstream");
}

}  // namespace

std::shared_ptr<LlamaServerBackend> make_llamaserver_backend(LlamaServerBackendOptions options) {
    return std::make_shared<LlamaServerBackendImpl>(std::move(options));
}

}  // namespace sonder::inference
