#include "sonder/inference/server.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "anthropic.hpp"
#include "chat_handler.hpp"
#include "http.hpp"
#include "json_schema_gbnf.hpp"
#include "live_hub.hpp"
#include "common/log_queue.hpp"
#include "openai.hpp"
#include "priority_admission.hpp"
#include "request_path.hpp"
#include "residency.hpp"
#include "responses_handler.hpp"
#include "server_endpoints.hpp"
#include "server_policy.hpp"
#include "socket.hpp"
#include "tool_call_parser.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::server {

using namespace detail;  // NOLINT(google-build-using-namespace)

// ============================================================ Server::Impl

struct Server::Impl {
    explicit Impl(ServerOptions o)
        : opts(std::move(o)),
          log_queue(opts.log, opts.log_queue_capacity, log_drop_reporter(opts.log_format == LogFormat::json)),
          response_history(ResponseHistoryStore::kMaxEntries, ResponseHistoryStore::kMaxTotalBytes,
                           opts.disable_response_store ? std::chrono::milliseconds(0) : opts.response_history_ttl) {
        if (!opts.token.empty()) {
            token_digest = sha256_digest(opts.token);
            has_token_digest = true;
        }
        parse_gate.max_active = std::clamp<std::size_t>(opts.max_connections / 4, 4, 16);
        parse_gate.max_bytes = std::max<std::size_t>(opts.max_body_bytes * 4, 16u << 20);
        ResidencyConfig residency_cfg;
        residency_cfg.idle_ttl = opts.model_idle_ttl;
        residency_cfg.max_resident = opts.max_resident_models;
        residency_cfg.max_resident_vram_bytes = opts.max_resident_vram_bytes;
        residency_cfg.max_resident_ram_bytes = opts.max_resident_ram_bytes;
        residency = std::make_shared<ModelResidency>(
            residency_cfg,
            [this](const std::string& id) { return load_served_model(id); },
            [this](Eviction& ev) { on_model_evicted(ev); },
            [this](const std::string& id, const std::shared_ptr<Model>& model,
                   const ModelResourceFootprint& reserved) { return observe_model_footprint(id, model, reserved); });
    }

    ServerOptions opts;
    std::array<std::uint8_t, 32> token_digest{};
    bool has_token_digest = false;
    ParseGate parse_gate;
    // Access-log and diagnostic lines on their way to `opts.log`, which is
    // only ever called on the queue's logger thread (never on a request or
    // accept thread). Declared right after `opts` (it is built from them) and
    // before everything else, so it is destroyed last and anything that still
    // logs while the other members go away finds it alive. Mutable: the
    // logging helpers are const and the queue synchronizes itself.
    mutable LogQueue log_queue;
    std::atomic<State> state{State::starting};

    Socket listener;
    // Spare descriptor for answering connections at EMFILE (accept thread
    // only after start; acquired before that thread exists).
    ReserveDescriptor reserve;
    std::uint16_t bound_port = 0;
    std::string url;
    bool loopback = true;
    bool synthetic = false;

    std::shared_ptr<LiveTelemetryHub> hub;
    std::string instance_id;
    std::string node_id;
    std::chrono::steady_clock::time_point started_at;

    mutable std::mutex engine_mu;
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Backend> backend;
    // Served models: registered during start(), loaded eagerly (default) or
    // on first use, pinned by in-flight requests, optionally evicted when
    // idle (docs/SERVER.md "Model residency").
    std::shared_ptr<ModelResidency> residency;

    // Backend probe cache (health must stay cheap: it never does backend I/O).
    BackendProbeRefresher probe;

    std::atomic<bool> stopping{false};   // listener closing; idle reads end
    std::atomic<bool> hard_stop{false};  // abort writes and streams
    std::thread accept_thread;
    std::atomic<bool> started{false};

    // Startup steps (bind, backend construction, each model load) run one at
    // a time. stop() sets cancel_start and waits for the step in progress
    // (bounded by the backend's own timeouts); start() checks cancel_start
    // before every step. Engine and backend locks are never held across
    // backend I/O, so health keeps answering "starting" meanwhile.
    std::mutex startup_mu;
    std::condition_variable startup_cv;
    bool cancel_start = false;
    bool in_step = false;

    std::mutex conns_mu;
    std::vector<std::pair<std::thread, std::shared_ptr<ConnFlags>>> conns;
    std::atomic<std::size_t> active{0};

    std::mutex inflight_mu;
    std::condition_variable inflight_cv;
    std::set<std::shared_ptr<Session>> inflight;
    PriorityAdmission admission;
    ResponseHistoryStore response_history;
    mutable CompiledGrammarCache grammar_cache;

    std::once_flag stop_once;
    std::mutex stopped_mu;
    std::condition_variable stopped_cv;
    bool stopped = false;

    // ------------------------------------------------------------- logging

    void log_line(std::string line) const {
        if (opts.log) {
            (void)log_queue.push(std::move(line));
        }
    }

    void stop_log_queue() {
        (void)log_queue.stop(std::max(opts.shutdown_grace, std::chrono::milliseconds(100)));
    }

    void log_message(const char* level, const std::string& message) const {
        log_line(format_log_message(opts.log_format, level, message));
    }

    void access_log(const Exchange& ex) const { log_line(format_access_log(opts.log_format, ex)); }

    // ------------------------------------------------------------ writing

    bool write_raw(Exchange& ex, std::string_view data) const {
        return send_raw(ex, data, opts.write_stall_timeout, hard_stop);
    }

    void write_json(Exchange& ex, int status, const json::Object& body, const Headers& extra = {}) const {
        if (stopping.load()) ex.keep_alive = false;
        send_json(ex, status, body, opts.write_stall_timeout, hard_stop, extra);
    }

    void write_error(Exchange& ex, const ApiError& error, Headers extra = {}) const {
        if (stopping.load()) ex.keep_alive = false;
        send_error(ex, error, opts.write_stall_timeout, hard_stop, std::move(extra));
    }

    // ------------------------------------------------------ model residency

    // Loads a served model on the engine (residency loader; no lock held).
    Result<std::shared_ptr<Model>> load_served_model(const std::string& id) {
        Engine* e = nullptr;
        std::string backend_name;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            e = engine.get();
            if (backend) {
                backend_name = backend->name();
            }
        }
        if (e == nullptr || backend_name.empty()) {
            return Status(ErrorCode::unavailable, "the engine is not running");
        }
        // A lazy load that would start after stop() began only delays the
        // shutdown: refuse it (503 not_ready, safe to retry elsewhere).
        if (stopping.load()) {
            return Status(ErrorCode::unavailable, "sonder-inference is draining");
        }
        // Every load goes through here: eager, lazy and after an eviction.
        return e->load_model(backend_name, make_model_load_options(opts, id));
    }

    // Reconciles a freshly loaded model's pre-load byte reservation against
    // backend runtime GPU memory observations and model descriptor metadata.
    std::optional<ModelResourceFootprint> observe_model_footprint(const std::string& id,
                                                                  const std::shared_ptr<Model>& model,
                                                                  const ModelResourceFootprint& reserved) {
        (void)id;
        ModelResourceFootprint observed = reserved;
        if (model && observed.vram.weights_bytes == 0 && observed.ram.weights_bytes == 0 &&
            model->descriptor().size_bytes > 0) {
            if (model->device_id().rfind("gpu:", 0) == 0) {
                observed.vram.weights_bytes = model->descriptor().size_bytes;
            } else {
                observed.ram.weights_bytes = model->descriptor().size_bytes;
            }
        }
        std::shared_ptr<Backend> b;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
        }
        if (b) {
            if (const auto runtime = b->runtime_status()) {
                if (runtime->gpu_memory.status == "ok" && runtime->gpu_memory.dedicated_bytes > 0) {
                    const std::uint64_t other_vram = residency ? residency->totals().resident_vram_bytes : 0;
                    if (runtime->gpu_memory.dedicated_bytes > other_vram) {
                        const std::uint64_t observed_delta = runtime->gpu_memory.dedicated_bytes - other_vram;
                        const std::uint64_t est_vram = observed.vram_bytes();
                        if (est_vram == 0) {
                            observed.vram.weights_bytes = observed_delta;
                        } else if (observed_delta > est_vram) {
                            const std::uint64_t extra = observed_delta - est_vram;
                            const std::uint64_t max_u64 = std::numeric_limits<std::uint64_t>::max();
                            observed.vram.workspace_bytes =
                                observed.vram.workspace_bytes > max_u64 - extra
                                    ? max_u64
                                    : observed.vram.workspace_bytes + extra;
                        }
                    }
                }
            }
        }
        return observed;
    }

    // Residency evictor: `model.evicted`, then the engine's `model.unload`.
    void on_model_evicted(Eviction& ev) {
        if (!ev.model) {
            return;
        }
        const std::string model_instance = ev.model->instance_id();
        const std::string device_id = ev.model->device_id();
        ev.model.reset();
        Engine* e = engine_ptr();
        if (e == nullptr) {
            return;
        }
        TelemetryContext ctx = e->engine_context();
        ctx.model_instance_id = model_instance;
        if (!device_id.empty()) {
            ctx.device_id = device_id;
        }
        e->telemetry().emit("model.evicted", ctx,
                            json::Object{{"backend", ev.backend},
                                         {"model", ev.id},
                                         {"reason", std::string(ev.reason)},
                                         {"idle_ms", ev.idle_ms},
                                         {"freed_vram_bytes", ev.freed_footprint.vram_bytes()},
                                         {"freed_ram_bytes", ev.freed_footprint.ram_bytes()},
                                         {"eviction_cost", ev.eviction_cost}},
                            TelemetryLevel::metrics);
        (void)e->unload_model(model_instance);
        log_message("info", "unloaded model '" + ev.id + "' (" + ev.reason + ")");
    }

    // ------------------------------------------------------------ routes

    bool backend_has_capability(Capability cap) {
        std::lock_guard<std::mutex> lock(engine_mu);
        return backend && backend->capabilities().has(cap);
    }

    static ApiError not_ready_error(State s) {
        return make_error(503, "not_ready",
                          std::string("sonder-inference is ") + state_name(s) +
                              "; nothing was executed, the request is safe to send elsewhere");
    }

    bool require_ready(Exchange& ex) {
        const State s = state.load();
        if (s == State::ready) {
            return true;
        }
        write_error(ex, not_ready_error(s));
        return false;
    }

    void fail_inference_request(Exchange& ex, std::string_view kind, const ApiError& err,
                                const Correlation& corr = {},
                                std::optional<RequestPriority> job_prio = std::nullopt,
                                std::string_view resolved_model = {},
                                std::string_view requested_model = {},
                                const std::shared_ptr<Model>& loaded_model = nullptr,
                                const std::shared_ptr<Session>& session = nullptr) {
        if (Engine* e = engine_ptr()) {
            TelemetryContext ctx = session ? session->telemetry_context(ex.request_id) : e->engine_context();
            ctx.request_id = ex.request_id;
            if (!session) {
                if (loaded_model) {
                    ctx.model_instance_id = loaded_model->instance_id();
                    ctx.device_id = loaded_model->device_id();
                    ctx.synthetic_work = is_synthetic_backend(loaded_model->backend_name());
                } else {
                    ctx.synthetic_work = synthetic;
                }
            }
            emit_pre_execution_failure(e->telemetry(), std::move(ctx), kind, err.code, corr, job_prio,
                                       resolved_model, requested_model);
        }
        write_error(ex, err);
    }

    void handle_health(Exchange& ex) {
        const State s = state.load();
        bool available = false;
        std::string version;
        probe.status(available, version);
        std::shared_ptr<Backend> b;
        std::uint64_t emitted = 0;
        std::uint64_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
            if (engine) {
                emitted = engine->telemetry().emitted_events();
                dropped = engine->telemetry().dropped_events();
            }
        }
        const HealthSnapshot snapshot{
            opts, s, available, std::move(version), std::move(b), *residency, hub->stats(), emitted, dropped,
            admission.queued(), log_queue.stats(), instance_id, node_id, started_at, synthetic,
            grammar_cache.stats()};
        write_json(ex, s == State::ready ? 200 : 503, build_health_body(snapshot));
    }

    void handle_models(Exchange& ex) {
        if (!require_ready(ex)) {
            return;
        }
        std::shared_ptr<Backend> b;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
        }
        write_json(ex, 200, build_models_body(opts, *residency, b, synthetic));
    }

    void handle_identity(Exchange& ex, const RequestHead& head) {
        if (!require_ready(ex)) {
            return;
        }
        std::string wanted = "default";
        if (!head.query.empty()) {
            if (const auto m = query_param(head.query, "model")) {
                wanted = *m;
            }
        }
        const auto served_id = residency->resolve(wanted);
        if (!served_id) {
            write_error(ex, make_error(404, "model_not_found", "model '" + wanted + "' is not served", "model"));
            return;
        }
        bool available = false;
        std::string version;
        probe.status(available, version);
        std::shared_ptr<Backend> b;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
        }
        const std::optional<std::string> probed = available ? std::optional<std::string>(version) : std::nullopt;
        write_json(ex, 200, build_identity_body(opts, *residency, *served_id, synthetic, probed, b));
    }

    void handle_node(Exchange& ex) {
        if (!require_ready(ex)) {
            return;
        }
        bool available = false;
        std::string version;
        probe.status(available, version);
        std::shared_ptr<Backend> b;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            b = backend;
        }
        const NodeSnapshot snapshot{opts,       available,   std::move(version), std::move(b),
                                    *residency, instance_id, node_id,            synthetic};
        write_json(ex, 200, build_node_body(snapshot));
    }

    void handle_plan(Exchange& ex, const std::string& body) {
        std::variant<PlanHttpRequest, ApiError> parsed = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (!require_ready(ex)) {
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                write_error(ex, e);
                return;
            }
            parsed = parse_plan_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&parsed)) {
            write_error(ex, *e);
            return;
        }
        if (!require_ready(ex)) {
            return;
        }
        auto planned = execute_plan_request(opts, std::get<PlanHttpRequest>(parsed));
        if (auto* e = std::get_if<ApiError>(&planned)) {
            write_error(ex, *e);
            return;
        }
        write_json(ex, 200, std::get<json::Object>(planned));
    }

    void handle_chat(Exchange& ex, const RequestHead& head, const std::string& body,
                     ChatProtocol protocol = ChatProtocol::openai) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "chat", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<ChatJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "chat", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "chat", e, corr);
                return;
            }
            job_v = protocol == ChatProtocol::anthropic
                        ? parse_messages_request(body)
                        : parse_chat_request_with_mode(body, StructuredParseMode::defer_grammar_compile);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "chat", *e, corr);
            return;
        }
        ChatJob job = std::move(std::get<ChatJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "chat", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "chat", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        // Pins the model for the whole request (never evicted meanwhile);
        // loads it first when it is not resident (lazy residency, or evicted
        // after its idle TTL). Released after the response, when the session
        // (declared later, destroyed earlier) is gone.
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "chat", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();
        if ((job.response_format_type.has_value() || !job.grammar.empty()) &&
            !backend_has_capability(Capability::structured_output)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support structured output / grammar constraints",
                           job.response_format_type ? "response_format" : "grammar"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        if (job.sampling.logprobs && !backend_has_capability(Capability::logprobs)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support logprobs", "logprobs"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        const bool has_media = has_chat_media(job.messages);
        if (has_media && !backend_has_capability(Capability::multimodal)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support multimodal image inputs", "messages"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        const ModelDescriptor* desc_ptr = pin.model() ? &pin.model()->descriptor() : nullptr;
        const std::string tokenizer_id = build_tokenizer_identity_key(pin.backend(), pin.id(), desc_ptr);
        if (auto err = prepare_chat_grammar(job, tokenizer_id,
                                            backend_has_capability(Capability::structured_output),
                                            backend_has_capability(Capability::tool_calling),
                                            pin.backend(), &grammar_cache)) {
            fail_inference_request(ex, "chat", *err, corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        // Launch profile defaults fill only the fields the request left unset
        // (both chat protocols; "default" resolves to the served id first).
        if (const ModelProfileBinding* binding = find_profile_binding(opts, *served_id)) {
            apply_sampling_defaults(binding->sampling_defaults, job.sampling);
        }

        SessionOptions so;
        so.sampling = job.sampling;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "chat", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "chat", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] {
                ex.status = 200;
                ex.keep_alive = false;
                Headers h = base_headers(ex, "text/event-stream; charset=utf-8");
                h.emplace_back("X-Accel-Buffering", "no");
                return write_raw(ex, format_head(200, h));
            },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; chat request refused (503)"); }};
        execute_chat(exchange, job, corr, pin.id(), pin.backend(), session, protocol,
                     backend_has_capability(Capability::tool_calling), &grammar_cache, tokenizer_id,
                     backend_has_capability(Capability::structured_output));
    }

    void handle_responses(Exchange& ex, const RequestHead& head, const std::string& body) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "chat", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<ResponsesJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "chat", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "chat", e, corr);
                return;
            }
            job_v = parse_responses_request_with_mode(body, StructuredParseMode::defer_grammar_compile);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "chat", *e, corr);
            return;
        }
        ResponsesJob job = std::move(std::get<ResponsesJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "chat", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "chat", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "chat", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();
        if ((job.response_format_type.has_value() || !job.grammar.empty()) &&
            !backend_has_capability(Capability::structured_output)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support structured output / grammar constraints",
                           "text.format"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        if (job.sampling.logprobs && !backend_has_capability(Capability::logprobs)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support logprobs", "logprobs"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        const bool has_media = has_chat_media(job.input_messages);
        if (has_media && !backend_has_capability(Capability::multimodal)) {
            fail_inference_request(
                ex, "chat",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support multimodal image inputs", "input"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        const ModelDescriptor* desc_ptr = pin.model() ? &pin.model()->descriptor() : nullptr;
        const std::string tokenizer_id = build_tokenizer_identity_key(pin.backend(), pin.id(), desc_ptr);
        if (auto err = prepare_responses_grammar(job, tokenizer_id,
                                                 backend_has_capability(Capability::structured_output),
                                                 backend_has_capability(Capability::tool_calling),
                                                 pin.backend(), &grammar_cache)) {
            fail_inference_request(ex, "chat", *err, corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        if (const ModelProfileBinding* binding = find_profile_binding(opts, *served_id)) {
            apply_sampling_defaults(binding->sampling_defaults, job.sampling);
        }

        SessionOptions so;
        so.sampling = job.sampling;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "chat", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "chat", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] {
                ex.status = 200;
                ex.keep_alive = false;
                Headers h = base_headers(ex, "text/event-stream; charset=utf-8");
                h.emplace_back("X-Accel-Buffering", "no");
                return write_raw(ex, format_head(200, h));
            },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; responses request refused (503)"); }};
        const std::string caller_scope = derive_responses_caller_scope(head, corr);
        execute_responses(exchange, job, corr, pin.id(), pin.backend(), session, response_history,
                          backend_has_capability(Capability::tool_calling),
                          backend_has_capability(Capability::multimodal), caller_scope, &grammar_cache,
                          tokenizer_id, backend_has_capability(Capability::structured_output));
    }

    void handle_completions(Exchange& ex, const RequestHead& head, const std::string& body) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "generate", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<CompletionJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "generate", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "generate", e, corr);
                return;
            }
            job_v = parse_completion_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "generate", *e, corr);
            return;
        }
        CompletionJob job = std::move(std::get<CompletionJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "generate", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "generate", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "generate", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();
        if (!job.grammar.empty() && !backend_has_capability(Capability::structured_output)) {
            fail_inference_request(
                ex, "generate",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support grammar constraints", "grammar"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        if (job.sampling.logprobs && !backend_has_capability(Capability::logprobs)) {
            fail_inference_request(
                ex, "generate",
                make_error(400, "unsupported_parameter",
                           "backend '" + pin.backend() + "' does not support logprobs", "logprobs"),
                corr, job.priority_class, pin.id(), job.model, pin.model());
            return;
        }
        if (const ModelProfileBinding* binding = find_profile_binding(opts, *served_id)) {
            apply_sampling_defaults(binding->sampling_defaults, job.sampling);
        }

        SessionOptions so;
        so.sampling = job.sampling;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "generate", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "generate", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] {
                ex.status = 200;
                ex.keep_alive = false;
                Headers h = base_headers(ex, "text/event-stream; charset=utf-8");
                h.emplace_back("X-Accel-Buffering", "no");
                return write_raw(ex, format_head(200, h));
            },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; completion request refused (503)"); }};
        execute_completion(exchange, job, corr, pin.id(), pin.backend(), session);
    }

    void handle_embeddings(Exchange& ex, const RequestHead& head, const std::string& body) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "embed", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<EmbedJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "embed", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "embed", e, corr);
                return;
            }
            job_v = parse_embed_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "embed", *e, corr);
            return;
        }
        const EmbedJob job = std::move(std::get<EmbedJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "embed", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "embed", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "embed", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();

        SessionOptions so;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "embed", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "embed", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] { return false; },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; embeddings request refused (503)"); }};
        execute_embeddings(exchange, job, corr, pin.id(), pin.backend(), session);
    }

    void handle_rerank(Exchange& ex, const RequestHead& head, const std::string& body) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "rerank", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<RerankJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "rerank", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "rerank", e, corr);
                return;
            }
            job_v = parse_rerank_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "rerank", *e, corr);
            return;
        }
        const RerankJob job = std::move(std::get<RerankJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "rerank", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "rerank", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "rerank", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();

        SessionOptions so;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "rerank", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "rerank", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] { return false; },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; rerank request refused (503)"); }};
        execute_rerank(exchange, job, corr, pin.id(), pin.backend(), session);
    }

    void handle_score(Exchange& ex, const RequestHead& head, const std::string& body) {
        ex.request_id = make_id("req");
        auto corr_v = parse_correlation(head);
        if (auto* e = std::get_if<ApiError>(&corr_v)) {
            fail_inference_request(ex, "score", *e);
            return;
        }
        const Correlation corr = std::get<Correlation>(corr_v);
        std::variant<ScoreJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (const State s = state.load(); s != State::ready) {
                    fail_inference_request(ex, "score", not_ready_error(s), corr);
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                fail_inference_request(ex, "score", e, corr);
                return;
            }
            job_v = parse_score_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            fail_inference_request(ex, "score", *e, corr);
            return;
        }
        const ScoreJob job = std::move(std::get<ScoreJob>(job_v));
        if (const State s = state.load(); s != State::ready) {
            fail_inference_request(ex, "score", not_ready_error(s), corr, job.priority_class, job.model, job.model);
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            fail_inference_request(
                ex, "score", make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"),
                corr, job.priority_class, job.model, job.model);
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            fail_inference_request(ex, "score", load_failure_error(*served_id, pinned.status()), corr,
                                   job.priority_class, *served_id, job.model);
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();

        SessionOptions so;
        so.run_id = corr.run_id;
        so.agent_id = corr.agent_id;
        so.task_id = corr.task_id;
        so.workload = corr.workload;
        so.priority = corr.priority;
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            if (!engine) {
                write_error(ex, make_error(503, "not_ready", "sonder-inference is draining"));
                return;
            }
            auto created = engine->create_session(pin.model(), so);
            if (!created.ok()) {
                const ApiError err = map_session_failure(created.status(), false);
                TelemetryContext ctx = engine->engine_context();
                ctx.request_id = ex.request_id;
                ctx.model_instance_id = pin.model()->instance_id();
                ctx.device_id = pin.model()->device_id();
                ctx.synthetic_work = is_synthetic_backend(pin.model()->backend_name());
                emit_pre_execution_failure(engine->telemetry(), std::move(ctx), "score", err.code, corr,
                                           job.priority_class, pin.id(), job.model);
                write_error(ex, err);
                return;
            }
            session = created.value();
        }
        {
            std::lock_guard<std::mutex> lock(inflight_mu);
            if (state.load() != State::ready) {
                const ApiError err = make_error(
                    503, "not_ready",
                    "sonder-inference is draining; nothing was executed, the request is safe to send elsewhere");
                fail_inference_request(ex, "score", err, corr, job.priority_class, pin.id(), job.model, pin.model(),
                                       session);
                session->close();
                return;
            }
            inflight.insert(session);
        }

        ChatExchange exchange{
            opts, ex.sock, ex.request_id, ex.status, synthetic,
            ex.started, admission, hard_stop, engine_ptr()->telemetry(),
            [&] {
                {
                    std::lock_guard<std::mutex> lock(inflight_mu);
                    inflight.erase(session);
                }
                inflight_cv.notify_all();
            },
            [&](std::string_view data) { return write_raw(ex, data); },
            [&] { return false; },
            [&](int status, const json::Object& doc) { write_json(ex, status, doc); },
            [&](const ApiError& error) { write_error(ex, error); },
            [&] { log_message("warning", "cannot start a request thread; score request refused (503)"); }};
        execute_score(exchange, job, corr, pin.id(), pin.backend(), session);
    }

    void handle_tokenize(Exchange& ex, const std::string& body) {
        std::variant<TokenizeJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (!require_ready(ex)) {
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                write_error(ex, e);
                return;
            }
            job_v = parse_tokenize_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            write_error(ex, *e);
            return;
        }
        const TokenizeJob job = std::move(std::get<TokenizeJob>(job_v));
        if (!require_ready(ex)) {
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            write_error(ex, make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"));
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            write_error(ex, load_failure_error(*served_id, pinned.status()));
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();
        auto res = pin.model()->backend_model().tokenize(job.content);
        if (!res.ok()) {
            if (res.status().code() == ErrorCode::unsupported) {
                write_error(ex, make_error(501, "not_implemented", res.status().message()));
            } else {
                write_error(ex, map_session_failure(res.status(), false));
            }
            return;
        }
        json::Array toks;
        toks.reserve(res.value().size());
        for (const TokenId id : res.value()) {
            toks.emplace_back(static_cast<std::int64_t>(id));
        }
        json::Object meta{{"api_version", kApiVersion},
                          {"request_id", ex.request_id},
                          {"backend", pin.backend()},
                          {"synthetic", synthetic}};
        write_json(ex, 200,
                   json::Object{{"tokens", std::move(toks)},
                                {"count", static_cast<std::uint64_t>(res.value().size())},
                                {"model", pin.id()},
                                {"sonder", std::move(meta)}});
    }

    void handle_detokenize(Exchange& ex, const std::string& body) {
        std::variant<DetokenizeJob, ApiError> job_v = make_error(500, "internal_error", "uninitialized");
        {
            const auto parse_timeout = std::min(opts.read_timeout, std::chrono::milliseconds(2000));
            auto parse_guard = parse_gate.acquire(body.size(), parse_timeout);
            if (!parse_guard) {
                if (!require_ready(ex)) {
                    return;
                }
                ApiError e = make_error(503, "overloaded", "request parser concurrency limit reached; retry later");
                e.retry_after = 1;
                write_error(ex, e);
                return;
            }
            job_v = parse_detokenize_request(body);
        }
        if (auto* e = std::get_if<ApiError>(&job_v)) {
            write_error(ex, *e);
            return;
        }
        const DetokenizeJob job = std::move(std::get<DetokenizeJob>(job_v));
        if (!require_ready(ex)) {
            return;
        }
        const auto served_id = residency->resolve(job.model);
        if (!served_id) {
            write_error(ex, make_error(404, "model_not_found", "model '" + job.model + "' is not served", "model"));
            return;
        }
        auto pinned = residency->acquire(*served_id);
        if (!pinned.ok()) {
            write_error(ex, load_failure_error(*served_id, pinned.status()));
            return;
        }
        const ModelResidency::Pin pin = std::move(pinned).value();
        auto res = pin.model()->backend_model().detokenize(job.tokens);
        if (!res.ok()) {
            if (res.status().code() == ErrorCode::unsupported) {
                write_error(ex, make_error(501, "not_implemented", res.status().message()));
            } else {
                write_error(ex, map_session_failure(res.status(), false));
            }
            return;
        }
        json::Object meta{{"api_version", kApiVersion},
                          {"request_id", ex.request_id},
                          {"backend", pin.backend()},
                          {"synthetic", synthetic}};
        write_json(ex, 200,
                   json::Object{{"content", std::move(res).value()}, {"model", pin.id()}, {"sonder", std::move(meta)}});
    }

    // --------------------------------------------------------- connection

    void serve_connection(Socket sock, ConnFlags& flags) {
        std::string buffer;
        std::size_t requests_on_conn = 0;
        for (;;) {
            if (requests_on_conn > 0) {
                if (stopping.load() || hard_stop.load()) {
                    break;
                }
                const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();
                flags.head_started_ns.store(now_ns, std::memory_order_release);
                flags.in_head.store(true, std::memory_order_release);
                if (buffer.empty()) {
                    const auto idle_deadline = std::chrono::steady_clock::now() + opts.keep_alive_timeout;
                    bool got_bytes = false;
                    char chunk[4096];
                    for (;;) {
                        if (stopping.load() || hard_stop.load() || flags.evict.load()) {
                            break;
                        }
                        const auto now = std::chrono::steady_clock::now();
                        if (now >= idle_deadline) {
                            break;
                        }
                        const auto rem_ms = std::max<std::chrono::milliseconds>(
                            std::chrono::milliseconds(1),
                            std::chrono::duration_cast<std::chrono::milliseconds>(idle_deadline - now));
                        const auto slice = std::min(rem_ms, std::chrono::milliseconds(50));
                        const auto w = wait_socket(sock.get(), false, slice, &stopping);
                        if (w == WaitResult::timeout) {
                            continue;
                        }
                        if (w != WaitResult::ready) {
                            break;
                        }
                        const long long n = recv_some(sock.get(), chunk, sizeof(chunk));
                        if (n > 0) {
                            buffer.append(chunk, static_cast<std::size_t>(n));
                            got_bytes = true;
                            break;
                        }
                        if (n == 0 || n == -1) {
                            break;
                        }
                    }
                    if (!got_bytes) {
                        flags.in_head.store(false, std::memory_order_release);
                        break;
                    }
                }
                // Refresh head_started_ns once the next request actually begins
                // arriving so idle keep-alive time does not cause spurious 408 eviction.
                const auto req_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now().time_since_epoch())
                                              .count();
                flags.head_started_ns.store(req_start_ns, std::memory_order_release);
            }
            Exchange ex;
            ex.sock = sock.get();
            ex.started = std::chrono::steady_clock::now();
            ex.request_id = make_id("http");
            RequestHead head;
            std::size_t head_bytes = 0;
            const bool head_ok = read_head(ex, buffer, head, head_bytes, flags, opts, stopping, hard_stop);
            flags.in_head.store(false, std::memory_order_release);
            if (!head_ok) {
                if (ex.status != 0) {
                    access_log(ex);
                    sock.shutdown_write();
                    linger(ex, hard_stop);
                }
                break;
            }
            ++requests_on_conn;
            ex.method = head.method;
            ex.path = head.path;
            const RouteInfo route_info = lookup_route(head.path);
            const bool is_streaming_route = route_info.route == Route::telemetry ||
                                            route_info.route == Route::telemetry_sse ||
                                            route_info.route == Route::telemetry_ndjson;
            const bool can_keep_alive = opts.keep_alive && !stopping.load() && !hard_stop.load() &&
                                        head.minor_version >= 1 && !client_wants_close(head) &&
                                        !is_streaming_route && requests_on_conn < opts.max_keep_alive_requests;
            ex.keep_alive = can_keep_alive;
            ex.keep_alive_timeout = opts.keep_alive_timeout;
            ex.keep_alive_remaining =
                opts.max_keep_alive_requests > requests_on_conn ? (opts.max_keep_alive_requests - requests_on_conn) : 0;

            // Until read_body() consumes it, the declared body counts as unread.
            const std::uint64_t buffered = buffer.size() - head_bytes;
            if (head.has_transfer_encoding) {
                ex.unread_input = kLingerUnknownLength;
            } else if (head.content_length && *head.content_length > buffered) {
                ex.unread_input = *head.content_length - buffered;
            }
            dispatch(ex, head, buffer, head_bytes, flags);
            access_log(ex);

            const bool continue_alive = ex.keep_alive && !stopping.load() && !hard_stop.load() &&
                                        !flags.evict.load() && ex.unread_input == 0 && !ex.write_failed &&
                                        !flags.streaming.load() && ex.status != 499 &&
                                        requests_on_conn < opts.max_keep_alive_requests;
            if (!continue_alive) {
                // Let the peer read everything before the socket closes.
                sock.shutdown_write();
                linger(ex, hard_stop);
                break;
            }
        }
    }

    void dispatch(Exchange& ex, const RequestHead& head, std::string& buffer, std::size_t head_bytes,
                  ConnFlags& flags) {
        PreDispatchResult checked =
            run_request_guards(ex, head, buffer, head_bytes, opts, loopback, has_token_digest, token_digest, hard_stop);
        if (!checked.proceed) {
            return;
        }
        switch (checked.route) {
            case Route::health: handle_health(ex); break;
            case Route::models: handle_models(ex); break;
            case Route::identity: handle_identity(ex, head); break;
            case Route::sonder_node: handle_node(ex); break;
            case Route::sonder_plan: handle_plan(ex, checked.body); break;
            case Route::chat: handle_chat(ex, head, checked.body); break;
            case Route::completions: handle_completions(ex, head, checked.body); break;
            case Route::responses: handle_responses(ex, head, checked.body); break;
            case Route::messages: handle_chat(ex, head, checked.body, ChatProtocol::anthropic); break;
            case Route::embeddings: handle_embeddings(ex, head, checked.body); break;
            case Route::rerank: handle_rerank(ex, head, checked.body); break;
            case Route::score: handle_score(ex, head, checked.body); break;
            case Route::tokenize: handle_tokenize(ex, checked.body); break;
            case Route::detokenize: handle_detokenize(ex, checked.body); break;
            case Route::discovery:
                write_json(ex, 200, build_discovery_document(opts, *hub, instance_id, node_id, synthetic));
                break;
            case Route::telemetry:
            case Route::telemetry_sse:
            case Route::telemetry_ndjson:
                handle_telemetry_stream(ex, head, checked.route, flags, opts, *hub, instance_id, hard_stop);
                break;
            case Route::none: break;
        }
    }

    // ------------------------------------------------------------ accept

    // At the connection limit: makes room by evicting the oldest connection
    // that is still sending its request head after kEvictableHeadAge. Evicted
    // connections leave within one 50 ms wait slice; until then they may take
    // the process over the limit, never by more than the limit itself.
    // Connections past their head (authorized work, streams) are never
    // evicted. Returns true when the new connection may be served.
    bool evict_slow_head() {
        std::lock_guard<std::mutex> lock(conns_mu);
        std::size_t evicting = 0;
        ConnFlags* oldest = nullptr;
        std::int64_t oldest_ns = 0;
        const std::int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();
        const std::int64_t min_age_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(kEvictableHeadAge).count();
        for (const auto& [conn_thread, flags] : conns) {
            (void)conn_thread;
            if (flags->done.load()) {
                continue;
            }
            if (flags->evict.load()) {
                ++evicting;
                continue;
            }
            if (flags->in_head.load(std::memory_order_acquire)) {
                const std::int64_t started_ns = flags->head_started_ns.load(std::memory_order_acquire);
                if (now_ns - started_ns >= min_age_ns && (oldest == nullptr || started_ns < oldest_ns)) {
                    oldest = flags.get();
                    oldest_ns = started_ns;
                }
            }
        }
        if (evicting >= opts.max_connections) {
            return false;
        }
        if (active.load() < opts.max_connections + evicting) {
            return true;  // an eviction already in progress frees this slot
        }
        if (oldest == nullptr) {
            return false;
        }
        oldest->evict.store(true);
        return true;
    }

    void reap_finished() {
        std::vector<std::thread> finished;
        {
            std::lock_guard<std::mutex> lock(conns_mu);
            for (auto it = conns.begin(); it != conns.end();) {
                if (it->second->done.load()) {
                    finished.push_back(std::move(it->first));
                    it = conns.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& t : finished) {
            t.join();
        }
    }

    void reject_overloaded(Socket client, const std::string& reason) {
        access_log(reject_overloaded_socket(client, reason, hard_stop));
    }

    // Sleeps `d` on the accept thread, waking early when stopping.
    void backoff(std::chrono::milliseconds d) const {
        const auto deadline = std::chrono::steady_clock::now() + d;
        while (!stopping.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    void accept_loop() {
        bool exhausted_reported = false;
        bool failure_reported = false;
        while (!stopping.load()) {
            const auto w = wait_socket(listener.get(), false, std::chrono::milliseconds(100), &stopping);
            reap_finished();
            if (w != WaitResult::ready) {
                continue;
            }
            constexpr std::size_t kMaxAcceptBatch = 32;
            for (std::size_t batch = 0; batch < kMaxAcceptBatch && !stopping.load(); ++batch) {
                Accepted a = accept_client(listener);
                if (a.status == AcceptStatus::none_pending) {
                    break;
                }
                if (a.status == AcceptStatus::out_of_descriptors) {
                    if (!exhausted_reported) {
                        exhausted_reported = true;
                        log_message("warning",
                                    "out of file descriptors (accept failed, errno " + std::to_string(a.native_error) +
                                        "); answering new connections with 503 overloaded until descriptors free up. "
                                        "Raise the limit (ulimit -n) or lower --max-connections");
                    }
                    if (reserve.held() || reserve.acquire()) {
                        reserve.release();
                        Accepted spare = accept_client(listener);
                        if (spare.status == AcceptStatus::accepted) {
                            reject_overloaded(std::move(spare.socket), "out of file descriptors");
                        }
                        const bool reacquired = reserve.acquire();
                        if (spare.status == AcceptStatus::accepted && reacquired) {
                            continue;
                        }
                    }
                    backoff(std::chrono::milliseconds(50));
                    break;
                }
                if (a.status == AcceptStatus::failed) {
                    if (!failure_reported) {
                        failure_reported = true;
                        log_message("warning", "accept failed (errno " + std::to_string(a.native_error) +
                                                   "); backing off");
                    }
                    backoff(std::chrono::milliseconds(50));
                    break;
                }
                exhausted_reported = false;
                failure_reported = false;
                (void)reserve.acquire();
                Socket client = std::move(a.socket);
                if (active.load() >= opts.max_connections && !evict_slow_head()) {
                    reject_overloaded(std::move(client),
                                      "connection limit reached (" + std::to_string(opts.max_connections) + ")");
                    continue;
                }
                auto flags = std::make_shared<ConnFlags>();
                active.fetch_add(1);
                bool slot_reserved = false;
                try {
                    {
                        std::lock_guard<std::mutex> lock(conns_mu);
                        conns.emplace_back(std::thread{}, flags);
                        slot_reserved = true;
                    }
                    std::thread t([this, flags, s = std::move(client)]() mutable {
                        try {
                            serve_connection(std::move(s), *flags);
                        } catch (const std::exception& e) {
                            log_message("error", std::string("request failed with an exception: ") + e.what());
                        } catch (...) {
                            log_message("error", "request failed with an unknown exception");
                        }
                        active.fetch_sub(1);
                        flags->done.store(true);
                    });
                    std::lock_guard<std::mutex> lock(conns_mu);
                    conns.back().first = std::move(t);  // noexcept; slot already allocated
                } catch (const std::system_error&) {
                    if (slot_reserved) {
                        std::lock_guard<std::mutex> lock(conns_mu);
                        if (!conns.empty() && conns.back().second == flags && !conns.back().first.joinable()) {
                            conns.pop_back();
                        }
                    }
                    active.fetch_sub(1);
                    log_message("warning", "cannot start a connection thread; connection dropped");
                } catch (const std::bad_alloc&) {
                    active.fetch_sub(1);
                    log_message("warning", "cannot allocate connection slot; connection dropped");
                }
            }
            reap_finished();
        }
        listener.close();
    }

    // ------------------------------------------------------------- start

    bool begin_step() {
        std::lock_guard<std::mutex> lock(startup_mu);
        if (cancel_start) {
            return false;
        }
        in_step = true;
        return true;
    }

    void end_step() {
        {
            std::lock_guard<std::mutex> lock(startup_mu);
            in_step = false;
        }
        startup_cv.notify_all();
    }

    struct StepGuard {
        Impl& impl;
        ~StepGuard() { impl.end_step(); }
    };

    static Status cancelled_start() {
        return Status(ErrorCode::cancelled, "the server was stopped while starting");
    }

    Engine* engine_ptr() const {
        std::lock_guard<std::mutex> lock(engine_mu);
        return engine.get();
    }

    // Step 1: bind, build the engine, start accepting.
    Status open_listener() {
        bool in_use = false;
        auto listened = listen_tcp(opts.host, opts.port, in_use);
        if (!listened.ok()) {
            return listened.status();
        }
        listener = std::move(listened.value().socket);
        bound_port = listened.value().port;
        loopback = is_loopback_bind(opts.host);
        url = format_listen_url(opts.host, bound_port, listened.value().ipv6);

        synthetic = is_synthetic_backend(opts.backend_instance ? opts.backend_instance->name() : opts.backend.backend);
        hub = std::make_shared<LiveTelemetryHub>(opts.telemetry_buffer, opts.max_subscribers, opts.subscriber_queue);
        EngineOptions eo;
        eo.telemetry.level = opts.telemetry_level;
        eo.telemetry.capture_text = opts.capture_text;
        eo.telemetry.text_capture = opts.text_capture;
        eo.telemetry.role = "inference";
        eo.telemetry.synthetic = synthetic;
        eo.telemetry.queue_capacity = std::max<std::size_t>(4096, opts.telemetry_buffer);
        eo.telemetry_sinks.push_back(hub);
        for (const auto& s : opts.extra_sinks) {
            eo.telemetry_sinks.push_back(s);
        }
        eo.server = EngineServerInfo{opts.host, bound_port, kApiVersion};
        apply_scheduling(opts, eo.scheduling);
        // Preserve the origin scheduler's default ordering unless the
        // operator actually enabled the server-side priority policy.
        eo.scheduling.strict_priority_admission = priority_admission_enabled(opts);
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            engine = std::make_unique<Engine>(std::move(eo));
            instance_id = engine->telemetry().instance_id();
            node_id = engine->telemetry().options().node_id;
        }
        hub->set_instance_id(instance_id);
        started_at = std::chrono::steady_clock::now();
        if (const std::uint64_t limit = descriptor_limit();
            limit != 0 && opts.max_connections + kDescriptorHeadroom > limit) {
            log_message("warning", "--max-connections " + std::to_string(opts.max_connections) +
                                       " plus headroom exceeds the open-file limit (" + std::to_string(limit) +
                                       "); connections beyond it get 503 overloaded. Raise ulimit -n or lower "
                                       "--max-connections");
        }
        {
            std::lock_guard<std::mutex> lock(startup_mu);
            started.store(true);
        }
#if !defined(_WIN32)
        if (!reserve.acquire()) {
            log_message("warning", "cannot reserve a spare descriptor; at the open-file limit new connections wait "
                                   "in the backlog instead of getting 503");
        }
#endif
        accept_thread = std::thread([this] { accept_loop(); });
        return Status::success();
    }

    Status start(const std::function<void()>& on_listening) {
        if (started.load()) {
            return Status(ErrorCode::invalid_state, "server already started");
        }
        if (Status st = validate_options(opts); !st.ok()) {
            return st;
        }
        if (!begin_step()) {
            return cancelled_start();
        }
        {
            StepGuard step{*this};
            if (Status st = open_listener(); !st.ok()) {
                return st;
            }
        }
        if (on_listening) {
            on_listening();
        }

        std::shared_ptr<Backend> made;
        {
            if (!begin_step()) {
                return cancelled_start();
            }
            StepGuard step{*this};
            if (opts.backend_instance) {
                made = opts.backend_instance;
            } else {
                auto built = make_backend(opts.backend);
                if (!built.ok()) {
                    return built.status();
                }
                made = built.value();
            }
            if (Status st = engine_ptr()->register_backend(made); !st.ok()) {
                return st;
            }
            std::lock_guard<std::mutex> lock(engine_mu);
            backend = made;
        }
        std::vector<std::string> ids = opts.models;
        if (ids.empty()) {
            ids.emplace_back("mock");
        }
        const std::string backend_name = made->name();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (!begin_step()) {
                return cancelled_start();
            }
            StepGuard step{*this};
            ModelResourceFootprint estimated;
            if (const ModelProfileBinding* binding = find_profile_binding(opts, ids[i])) {
                estimated = binding->estimated_footprint;
            }
            (void)residency->add(ids[i], backend_name, i == 0, std::move(estimated));
            if (opts.lazy_models || (opts.max_resident_models != 0 && i >= opts.max_resident_models)) {
                continue;
            }
            auto loaded = residency->acquire(ids[i]);
            if (!loaded.ok()) {
                return Status(loaded.status().code(), "cannot load model '" + ids[i] + "' on " + backend_name + ": " +
                                                          loaded.status().message());
            }
        }
        {
            if (!begin_step()) {
                return cancelled_start();
            }
            StepGuard step{*this};
            probe.record(made->probe());
            probe.start(made);
            residency->start_sweeper();
        }
        if (priority_admission_enabled(opts)) {
            const std::size_t capacity = opts.backend_capacity != 0 ? opts.backend_capacity
                                                                   : made->max_concurrent_requests();
            admission.configure(capacity, opts.max_concurrent_subagent, opts.max_concurrent_background,
                                opts.max_queue_per_class);
        }
        State expected = State::starting;
        state.compare_exchange_strong(expected, State::ready);
        return Status::success();
    }

    // -------------------------------------------------------------- stop

    template <class Pred>
    bool wait_until(Pred pred, std::chrono::steady_clock::time_point deadline) {
        while (!pred()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

    bool all_connections_done(bool ignore_streams) {
        std::lock_guard<std::mutex> lock(conns_mu);
        return std::all_of(conns.begin(), conns.end(), [ignore_streams](const auto& c) {
            return c.second->done.load() || (ignore_streams && c.second->streaming.load());
        });
    }

    void do_stop() {
        // 0. Abandon a start() in progress: health reports draining at once,
        // and the startup step running now (a model load, say) is the last.
        state.store(State::draining);
        bool was_started = false;
        {
            std::unique_lock<std::mutex> lock(startup_mu);
            cancel_start = true;
            startup_cv.wait(lock, [this] { return !in_step; });
            was_started = started.load();
        }
        if (!was_started) {
            stop_log_queue();
            state.store(State::stopped);
            std::lock_guard<std::mutex> lock(stopped_mu);
            stopped = true;
            stopped_cv.notify_all();
            return;
        }
        // 1. New chats get 503 not_ready.
        // 2. Stop accepting.
        stopping.store(true);
        parse_gate.close();
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        // 3. In-flight requests get the grace period, then are cancelled.
        {
            std::unique_lock<std::mutex> lock(inflight_mu);
            inflight_cv.wait_for(lock, opts.shutdown_grace, [this] { return inflight.empty(); });
            for (const auto& s : inflight) {
                s->cancel();
            }
        }
        const auto grace_deadline = std::chrono::steady_clock::now() + opts.shutdown_grace;
        if (!wait_until([this] { return all_connections_done(true); }, grace_deadline)) {
            hard_stop.store(true);  // responses to peers that stopped reading
            wait_until([this] { return all_connections_done(true); },
                       std::chrono::steady_clock::time_point::max());
        }
        // 4. engine.stopped (and a final telemetry.dropped) reach the hub.
        if (!probe.stop(std::max(opts.shutdown_grace, std::chrono::milliseconds(100)))) {
            log_message("warning", "backend probe still blocked in backend I/O at shutdown; not waiting for it");
        }
        residency->stop_sweeper();
        residency->clear();
        std::unique_ptr<Engine> doomed;
        {
            std::lock_guard<std::mutex> lock(engine_mu);
            backend.reset();
            doomed = std::move(engine);
        }
        doomed.reset();
        // 5. Telemetry streams deliver what they hold, then end.
        hub->close();
        if (!wait_until([this] { return all_connections_done(false); },
                        std::chrono::steady_clock::now() + opts.shutdown_grace)) {
            hard_stop.store(true);
        }
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lock(conns_mu);
            for (auto& c : conns) {
                threads.push_back(std::move(c.first));
            }
            conns.clear();
        }
        for (auto& t : threads) {
            t.join();
        }
        // 6. Nothing logs any more: the queued access-log lines are written,
        // unless the sink is blocked, which shutdown does not wait out.
        stop_log_queue();
        state.store(State::stopped);
        std::lock_guard<std::mutex> lock(stopped_mu);
        stopped = true;
        stopped_cv.notify_all();
    }
};

// ================================================================= Server

Server::Server(ServerOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

Server::~Server() { stop(); }

Status Server::start(const std::function<void()>& on_listening) {
    Status st = impl_->start(on_listening);
    if (!st.ok() && impl_->started.load()) {
        stop();
    }
    return st;
}

void Server::stop() {
    std::call_once(impl_->stop_once, [this] { impl_->do_stop(); });
    wait();
}

void Server::wait() {
    std::unique_lock<std::mutex> lock(impl_->stopped_mu);
    impl_->stopped_cv.wait(lock, [this] { return impl_->stopped; });
}

std::uint16_t Server::port() const noexcept { return impl_->bound_port; }
std::string Server::url() const { return impl_->url; }
std::string Server::instance_id() const { return impl_->instance_id; }

Engine* Server::engine() noexcept {
    std::lock_guard<std::mutex> lock(impl_->engine_mu);
    return impl_->engine.get();
}

}  // namespace sonder::inference::server
