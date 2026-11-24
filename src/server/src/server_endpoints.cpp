#include "server_endpoints.hpp"

#include <algorithm>

#include "health_features.hpp"
#include "identity.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder_inference.h"

namespace sonder::inference::server::detail {

json::Object sonder_meta() { return json::Object{{"api_version", kApiVersion}}; }

const ModelProfileBinding* find_profile_binding(const ServerOptions& opts, const std::string& id) {
    for (const auto& b : opts.profile_bindings) {
        if (b.model == id) {
            return &b;
        }
    }
    return nullptr;
}

json::Object profile_with_runtime_context(json::Object profile, const std::optional<BackendRuntimeStatus>& runtime) {
    if (!runtime || !runtime->context.fitted_ctx) {
        return profile;
    }
    const json::Value* configured = profile.find("context_length");
    const std::uint64_t fitted = *runtime->context.fitted_ctx;
    if (configured != nullptr && configured->is_integer() &&
        static_cast<std::uint64_t>(configured->as_int()) == fitted) {
        return profile;
    }
    const json::Value configured_value = configured != nullptr ? *configured : json::Value();
    profile.set("context_length", json::Value(fitted));
    profile.set("configured_context_length", configured_value);
    return profile;
}

ApiError load_failure_error(const std::string& id, const Status& st) {
    if (st.code() == ErrorCode::not_found) {
        return make_error(404, "model_not_found", "cannot load model '" + id + "': " + st.message(), "model");
    }
    return make_error(503, "not_ready",
                      "cannot load model '" + id + "': " + st.message() +
                          "; nothing was executed, the request is safe to send elsewhere");
}

json::Array build_health_models(const ModelResidency& residency) {
    json::Array models;
    for (const auto& m : residency.snapshot()) {
        json::Object entry{{"id", m.id}, {"backend", m.backend}, {"default", m.is_default}};
        // Additive residency fields (docs/SERVER.md "Model residency").
        entry.set("state", residency_state_name(m.state));
        entry.set("resident", m.state == ResidencyState::resident);
        entry.set("in_flight", m.pins);
        entry.set("loads", m.loads);
        entry.set("load_failures", m.load_failures);
        entry.set("evictions", m.evictions);
        entry.set("idle_s", m.idle_s ? json::Value(*m.idle_s) : json::Value(nullptr));
        entry.set("model_instance_id",
                  m.model_instance_id.empty() ? json::Value(nullptr) : json::Value(m.model_instance_id));
        models.emplace_back(std::move(entry));
    }
    return models;
}

json::Object build_residency_summary(const ServerOptions& opts, const ModelResidency& residency) {
    const ResidencyTotals t = residency.totals();
    const ResidencyConfig& c = residency.config();
    return json::Object{
        {"mode", opts.lazy_models ? "lazy" : "eager"},
        {"idle_ttl_s", c.idle_ttl.count() > 0 ? json::Value(std::chrono::duration<double>(c.idle_ttl).count())
                                              : json::Value(nullptr)},
        {"max_resident", c.max_resident > 0 ? json::Value(c.max_resident) : json::Value(nullptr)},
        {"registered", t.registered},
        {"resident", t.resident},
        {"loading", t.loading},
        {"unloading", t.unloading},
        {"loads", t.loads},
        {"load_failures", t.load_failures},
        {"evictions", t.evictions}};
}

json::Object build_health_body(const HealthSnapshot& s) {
    json::Array backends;
    std::string bound_backend;
    if (s.backend) {
        bound_backend = s.backend->name();
        json::Array caps;
        for (const auto& c : s.backend->capabilities().names()) {
            caps.emplace_back(c);
        }
        json::Object entry{
            {"name", s.backend->name()}, {"available", s.backend_available}, {"capabilities", std::move(caps)}};
        if (!s.backend_version.empty()) {
            entry.set("version", s.backend_version);
        }
        // Additive: only backends that report runtime observations
        // (e.g. a spawned llama-server) get this field.
        if (const auto runtime = s.backend->runtime_status()) {
            entry.set("runtime", to_json(*runtime));
        }
        backends.emplace_back(std::move(entry));
    }
    return json::Object{
        {"status", state_name(s.state)},
        {"api_version", kApiVersion},
        {"version", version_string()},
        {"commit", build_commit()},
        {"abi_version", SONDER_ABI_VERSION},
        {"instance_id", s.instance_id},
        {"node_id", s.node_id},
        {"uptime_s", seconds_since(s.started_at)},
        {"synthetic", s.synthetic},
        {"auth_required", !s.opts.token.empty()},
        {"backends", std::move(backends)},
        {"models", build_health_models(s.residency)},
        {"queued_by_class", json::Object{{"interactive", s.queued_by_class[0]},
                                         {"subagent", s.queued_by_class[1]},
                                         {"background", s.queued_by_class[2]}}},
        {"residency", build_residency_summary(s.opts, s.residency)},
        {"telemetry", json::Object{{"level", to_string(s.opts.telemetry_level)},
                                   {"subscribers", s.hub_stats.subscribers},
                                   {"retained", s.hub_stats.retained},
                                   {"capacity", s.hub_stats.capacity},
                                   {"emitted", s.telemetry_emitted},
                                   {"dropped", s.telemetry_dropped},
                                   {"subscriber_dropped_events", s.hub_stats.subscriber_dropped_events}}},
        {"log", json::Object{{"queued", s.log_stats.queued},
                             {"capacity", s.log_stats.capacity},
                             {"written", s.log_stats.written},
                             {"dropped", s.log_stats.dropped}}},
        {"sonder", health_features(bound_backend, s.opts, sonder_meta())}};
}

json::Object build_models_body(const ServerOptions& opts, const ModelResidency& residency,
                               const std::shared_ptr<Backend>& backend, bool synthetic) {
    const std::optional<BackendRuntimeStatus> runtime = backend ? backend->runtime_status() : std::nullopt;
    json::Array data;
    for (const auto& m : residency.snapshot()) {
        json::Object ext{{"backend", m.backend}, {"default", m.is_default}, {"synthetic", synthetic}};
        // Additive: context fit, GPU memory and warnings of the backend
        // process serving this model, when the backend reports them. This
        // describes the backend process, not the model handle, so it is
        // reported whatever the model's residency state (a lazy or
        // evicted model is still served by that process).
        if (runtime && backend->name() == m.backend) {
            ext.set("runtime", to_json(*runtime));
        }
        // Additive: the launch profile behind this model (whatever its
        // residency state, like `runtime` above).
        if (const ModelProfileBinding* binding = find_profile_binding(opts, m.id)) {
            const bool own_runtime = backend && backend->name() == m.backend;
            ext.set("profile",
                    profile_with_runtime_context(binding->metadata,
                                                 own_runtime ? runtime : std::optional<BackendRuntimeStatus>{}));
        }
        data.emplace_back(json::Object{
            {"id", m.id}, {"object", "model"}, {"owned_by", "sonder-inference"}, {"sonder", std::move(ext)}});
    }
    json::Object meta = health_features(backend ? backend->name() : "", opts, sonder_meta());
    if (!opts.profile_catalog.empty()) {
        json::Array catalog;
        for (const auto& entry : opts.profile_catalog) {
            const json::Value* is_served = entry.find("served");
            if (entry.is_object() && is_served != nullptr && is_served->is_bool() && is_served->as_bool()) {
                catalog.emplace_back(profile_with_runtime_context(entry.as_object(), runtime));
            } else {
                catalog.push_back(entry);
            }
        }
        meta.set("profiles", std::move(catalog));
    }
    return json::Object{{"object", "list"}, {"data", std::move(data)}, {"sonder", std::move(meta)}};
}

json::Object build_identity_body(const ModelResidency& residency, const std::string& served_id, bool synthetic,
                                 const std::optional<std::string>& probed) {
    // Identity never loads a model (no backend I/O here): a model that is
    // not resident reports the descriptor of its last load, if any.
    IdentityResult id;
    if (const std::shared_ptr<Model> model = residency.resident_model(served_id)) {
        id = backend_identity(*model, probed);
    } else {
        std::string backend_name;
        for (const auto& m : residency.snapshot()) {
            if (m.id == served_id) {
                backend_name = m.backend;
            }
        }
        const std::optional<ModelDescriptor> last = residency.last_descriptor(served_id);
        id = backend_identity(backend_name, served_id, last ? &*last : nullptr, probed);
    }
    return json::Object{{"schema", "sonder.inference.identity/1"},
                        {"model", served_id},
                        {"synthetic", synthetic},
                        {"backend_identity", id.backend_identity},
                        {"reason", id.reason},
                        {"sonder", sonder_meta()}};
}

json::Object build_discovery_document(const ServerOptions& opts, const LiveTelemetryHub& hub,
                                      const std::string& instance_id, const std::string& node_id, bool synthetic) {
    const HubStats hs = hub.stats();
    return json::Object{
        {"schema", "sonder.telemetry.producer/1"},
        {"producer", json::Object{{"name", "sonder-inference"},
                                  {"version", version_string()},
                                  {"node_id", node_id},
                                  {"instance_id", instance_id},
                                  {"role", "inference"},
                                  {"synthetic", synthetic}}},
        {"event_schema", std::string(kObservatorySchema)},
        {"streams", json::Array{json::Object{{"transport", "sse"}, {"url", "/v1/telemetry/sse"}},
                                json::Object{{"transport", "ndjson"}, {"url", "/v1/telemetry/ndjson"}}}},
        {"resume", json::Object{{"header", "Last-Event-ID"},
                                {"query", "last_event_id"},
                                {"retained_events", hs.retained},
                                {"oldest_sequence", hs.oldest_sequence},
                                {"next_sequence", hs.next_sequence}}},
        {"auth", json::Object{{"required", !opts.token.empty()}, {"schemes", json::Array{"bearer"}}}},
        {"clock", json::Object{{"mono_ns", "host-monotonic"}}},
        {"sampling_level", to_string(opts.telemetry_level)},
        {"text_capture", opts.capture_text ? "on" : "off"},
        {"links", json::Object{{"health", "/v1/sonder/health"},
                               {"identity", "/v1/sonder/identity"},
                               {"models", "/v1/models"}}},
        {"vocabularies", json::Object{{"sonder.inference.events", 1}}},
        {"stats", json::Object{{"subscribers", hs.subscribers},
                               {"max_subscribers", opts.max_subscribers},
                               {"capacity", hs.capacity},
                               {"subscriber_dropped_events", hs.subscriber_dropped_events}}},
        {"sonder", sonder_meta()}};
}

void handle_telemetry_stream(Exchange& ex, const RequestHead& head, Route route, ConnFlags& flags,
                             const ServerOptions& opts, LiveTelemetryHub& hub, const std::string& instance_id,
                             const std::atomic<bool>& hard_stop) {
    bool sse = true;
    if (route == Route::telemetry_ndjson) {
        sse = false;
    } else if (route == Route::telemetry) {
        const auto format = head.query.empty() ? std::nullopt : query_param(head.query, "format");
        if (format && (*format == "ndjson" || *format == "sse")) {
            sse = *format == "sse";
        } else if (format) {
            send_error(ex, make_error(400, "invalid_request", "format must be sse or ndjson", "format"),
                       opts.write_stall_timeout, hard_stop);
            return;
        } else if (const std::string* accept = head.header("accept")) {
            sse = accept->find("application/x-ndjson") == std::string::npos;
        }
    }
    ResumeRequest resume;
    if (const std::string* last = head.header("last-event-id")) {
        resume.last_event_id = *last;
    } else if (!head.query.empty()) {
        resume.last_event_id = query_param(head.query, "last_event_id");
    }
    if (!head.query.empty()) {
        const auto since = query_param(head.query, "since");
        resume.since_now = since && *since == "now";
    }
    auto subscribed = hub.subscribe(resume);
    if (subscribed.error == LiveTelemetryHub::SubscribeError::over_capacity) {
        ApiError e = make_error(429, "overloaded",
                                "telemetry subscriber limit reached (" + std::to_string(opts.max_subscribers) +
                                    " streams)");
        e.retry_after = 1;
        send_error(ex, e, opts.write_stall_timeout, hard_stop);
        return;
    }
    if (subscribed.error == LiveTelemetryHub::SubscribeError::closed) {
        send_error(ex, make_error(503, "not_ready", "telemetry is shutting down"), opts.write_stall_timeout, hard_stop);
        return;
    }
    const auto sub = subscribed.subscription;
    // From here on the stream never touches the engine.
    flags.streaming.store(true);
    Headers h = base_headers(ex, sse ? "text/event-stream; charset=utf-8" : "application/x-ndjson");
    h.emplace_back("X-Accel-Buffering", "no");
    ex.status = 200;
    std::string preamble = format_head(200, h);
    if (sse) {
        preamble += "retry: 2000\n\n";
        if (!subscribed.gap_comment.empty()) {
            preamble += subscribed.gap_comment + "\n\n";
        }
    }
    bool ok = send_raw(ex, preamble, opts.write_stall_timeout, hard_stop);
    auto last_write = std::chrono::steady_clock::now();
    while (ok) {
        const auto since_write = std::chrono::steady_clock::now() - last_write;
        const auto wait = std::max<std::chrono::steady_clock::duration>(
            std::chrono::milliseconds(1),
            std::min<std::chrono::steady_clock::duration>(opts.heartbeat_interval - since_write,
                                                          std::chrono::milliseconds(250)));
        auto batch = sub->next(std::chrono::duration_cast<std::chrono::milliseconds>(wait), &hard_stop);
        std::string frame;
        if (batch.dropped > 0 && sse) {
            frame += ": dropped " + std::to_string(batch.dropped) + "\n\n";
        }
        for (const auto& ev : batch.events) {
            if (sse) {
                frame += "id: " + instance_id + "-" + std::to_string(ev->sequence) + "\ndata: ";
                frame += ev->line;
                frame += "\n\n";
            } else {
                frame += ev->line;
                frame += "\n";
            }
        }
        if (!frame.empty()) {
            ok = send_raw(ex, frame, opts.write_stall_timeout, hard_stop);
            last_write = std::chrono::steady_clock::now();
        }
        if (!ok || batch.closed || hard_stop.load()) {
            break;
        }
        if (std::chrono::steady_clock::now() - last_write >= opts.heartbeat_interval) {
            ok = send_raw(ex, sse ? ": keepalive\n\n" : "\n", opts.write_stall_timeout, hard_stop);
            last_write = std::chrono::steady_clock::now();
        }
        if (peer_gone(ex.sock)) {
            break;
        }
    }
    hub.unsubscribe(sub);
}

void BackendProbeRefresher::status(bool& available, std::string& version) const {
    std::lock_guard<std::mutex> lock(cache_->mu);
    available = cache_->valid && cache_->ok;
    version = cache_->ok ? cache_->version : std::string();
}

void BackendProbeRefresher::record(const Result<std::string>& probed) {
    std::lock_guard<std::mutex> lock(cache_->mu);
    cache_->valid = true;
    cache_->ok = probed.ok();
    cache_->version = probed.ok() ? probed.value() : std::string();
}

void BackendProbeRefresher::start(std::shared_ptr<Backend> backend) {
    thread_ = std::thread([cache = cache_, b = std::move(backend)] {
        std::unique_lock<std::mutex> lock(cache->mu);
        while (!cache->stop) {
            cache->cv.wait_for(lock, kProbeInterval, [&] { return cache->stop; });
            if (cache->stop) {
                break;
            }
            lock.unlock();
            const Result<std::string> probed = b->probe();
            {
                std::lock_guard<std::mutex> store_lock(cache->mu);
                cache->valid = true;
                cache->ok = probed.ok();
                cache->version = probed.ok() ? probed.value() : std::string();
            }
            lock.lock();
        }
        cache->exited = true;
        cache->cv.notify_all();
    });
}

bool BackendProbeRefresher::stop(std::chrono::milliseconds patience) {
    if (!thread_.joinable()) {
        return true;
    }
    bool exited = false;
    {
        std::unique_lock<std::mutex> lock(cache_->mu);
        cache_->stop = true;
        cache_->cv.notify_all();
        exited = cache_->cv.wait_for(lock, patience, [this] { return cache_->exited; });
    }
    if (exited) {
        thread_.join();
        return true;
    }
    thread_.detach();
    return false;
}

}  // namespace sonder::inference::server::detail
