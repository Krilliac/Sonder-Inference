#include "sonder/inference/session.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "engine/accounting_tokens.hpp"
#include "engine/request_runtime.hpp"
#include "sonder/inference/backends.hpp"
#include "sonder/inference/engine.hpp"

#if defined(SONDER_HAS_SAMPLER_CHAIN)
#include "sonder/sampling/core_bridge.hpp"
#include "sonder/sampling/stop.hpp"
#endif

namespace sonder::inference {

const char* to_string(SessionState state) noexcept {
    switch (state) {
        case SessionState::idle: return "idle";
        case SessionState::running: return "running";
        case SessionState::closed: return "closed";
    }
    return "unknown";
}

const char* to_string(WorkloadClass workload) noexcept {
    switch (workload) {
        case WorkloadClass::interactive_user: return "interactive_user";
        case WorkloadClass::owner_orchestrator: return "owner_orchestrator";
        case WorkloadClass::critic_verification: return "critic_verification";
        case WorkloadClass::implementation_worker: return "implementation_worker";
        case WorkloadClass::research_worker: return "research_worker";
        case WorkloadClass::background_indexing: return "background_indexing";
        case WorkloadClass::maintenance: return "maintenance";
    }
    return "unknown";
}

const char* to_string(RequestOutcome outcome) noexcept {
    switch (outcome) {
        case RequestOutcome::none: return "none";
        case RequestOutcome::completed: return "completed";
        case RequestOutcome::cancelled: return "cancelled";
        case RequestOutcome::failed: return "failed";
    }
    return "unknown";
}

namespace {
// These are backend observations. Keep their names distinct from Sonder's
// logical reused_prompt_tokens, which do not prove physical KV reuse.
void backend_observations(json::Object& attrs, const GenerateStats& stats) {
    if (stats.cached_tokens) attrs.set("backend_cached_tokens", *stats.cached_tokens);
    if (stats.draft_tokens) attrs.set("backend_draft_tokens", *stats.draft_tokens);
    if (stats.draft_accepted_tokens) attrs.set("backend_draft_accepted_tokens", *stats.draft_accepted_tokens);
    if (stats.draft_tokens && *stats.draft_tokens > 0 && stats.draft_accepted_tokens) {
        attrs.set("backend_draft_acceptance_ratio",
                  static_cast<double>(*stats.draft_accepted_tokens) / static_cast<double>(*stats.draft_tokens));
    }
    if (stats.predicted_tokens_per_second) {
        attrs.set("backend_predicted_tokens_per_second", *stats.predicted_tokens_per_second);
    }
}

json::Value sampling_json(const SamplingConfig& s) {
    // With explicit_only, a field the caller did not set was never sent to a
    // model-default backend, so it is recorded as null ("model default"),
    // not as this struct's placeholder value.
    const auto field = [&](SamplingConfig::Field f, json::Value v) {
        return (!s.explicit_only || s.is_explicit(f)) ? std::move(v) : json::Value(nullptr);
    };
    json::Object o{{"temperature", field(SamplingConfig::kTemperature, s.temperature)},
                   {"top_p", field(SamplingConfig::kTopP, s.top_p)},
                   {"top_k", field(SamplingConfig::kTopK, s.top_k)},
                   {"min_p", field(SamplingConfig::kMinP, s.min_p)},
                   {"repeat_penalty", field(SamplingConfig::kRepeatPenalty, s.repeat_penalty)},
                   {"typical_p", s.typical_p},
                   {"repeat_last_n", field(SamplingConfig::kRepeatLastN, s.repeat_last_n)},
                   {"presence_penalty", field(SamplingConfig::kPresencePenalty, s.presence_penalty)},
                   {"frequency_penalty", field(SamplingConfig::kFrequencyPenalty, s.frequency_penalty)},
                   {"explicit_only", s.explicit_only},
                   {"logit_bias_count", static_cast<std::int64_t>(s.logit_bias.size())},
                   {"num_ctx", s.num_ctx},
                   {"seed", s.seed},
                   {"max_tokens", s.max_tokens}};
    return o;
}

// Compatibility fingerprint for prefix sharing: requests may share cached KV
// only for the same backend model (name, format, family, quantization).
std::uint64_t model_fingerprint(const Model& m) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::string& part) {
        for (const char c : part) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        h ^= 0xFFu;
        h *= 1099511628211ull;
    };
    const auto& d = m.descriptor();
    mix(m.backend_name());
    mix(d.name);
    mix(d.format);
    mix(d.family);
    mix(d.quantization);
    return h;
}

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

Session::Session(Engine& engine, std::shared_ptr<Model> model, SessionOptions options)
    : engine_(engine), model_(std::move(model)), options_(std::move(options)) {
    engine_.telemetry().emit("session.created", telemetry_context(),
                             json::Object{{"model", model_->descriptor().name},
                                          {"backend", model_->backend_name()},
                                          {"priority", options_.priority},
                                          {"workload", to_string(options_.workload)},
                                          {"text_capture", engine_.telemetry().options().capture_text ? "on" : "off"},
                                          {"sampling", sampling_json(options_.sampling)}},
                             TelemetryLevel::metrics);
}

Session::~Session() { close(); }

SessionState Session::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

RequestOutcome Session::last_outcome() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_outcome_;
}

std::uint64_t Session::requests_started() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_started_;
}

TelemetryContext Session::telemetry_context(const std::optional<std::string>& request_id) const {
    TelemetryContext ctx;
    ctx.session_id = options_.session_id;
    // Defaults to the engine id so session/request events group with the
    // engine's own events (Observatory run grouping).
    ctx.run_id = options_.run_id.value_or(engine_.engine_id());
    ctx.request_id = request_id;
    ctx.agent_id = options_.agent_id;
    ctx.task_id = options_.task_id;
    ctx.model_instance_id = model_->instance_id();
    ctx.device_id = model_->device_id();
    ctx.synthetic_work = model_->backend_name() == kMockBackendName;
    return ctx;
}

void Session::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_cancel_) {
        std::uint64_t expected = 0;
        cancel_requested_ns_.compare_exchange_strong(expected, monotonic_ns());
        active_cancel_->cancel();
    }
}

void Session::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return;
        }
        if (active_cancel_) {
            std::uint64_t expected = 0;
            cancel_requested_ns_.compare_exchange_strong(expected, monotonic_ns());
            active_cancel_->cancel();
        }
        state_ = SessionState::closed;
    }
    engine_.telemetry().emit("session.closed", telemetry_context(),
                             json::Object{{"requests", requests_started()}}, TelemetryLevel::metrics);
}

#if defined(SONDER_HAS_SAMPLER_CHAIN)
namespace {

// Token-level decode loop with Sonder's sampler chain (Capability::token_logits).
// NoViableCandidates maps to invalid_argument: it only happens when the
// request's own policy (bias/penalties/constraints) excludes every
// token, so retrying the same request cannot succeed. EmptyLogits is a backend
// fault (backend_error).
template <class Gate, class Produced, class Deliver>
Result<GenerateStats> run_sonder_sampling(TokenStream& stream, const SamplingConfig& config,
                                          const CancellationToken& cancel, TelemetryBus& bus,
                                          const TelemetryContext& ctx, const Gate& gate, const Produced& produced,
                                          const Deliver& deliver) {
    namespace smp = sampling;
    GenerateStats stats;
    stats.prompt_tokens = stream.prompt_tokens().size();
    stats.token_counts_from_backend = true;

    auto built = smp::make_chain(config, nullptr, stream.vocab_size());
    if (!built.ok()) {
        return built.status();
    }
    smp::SamplerChain& chain = built.value();
    chain.accept_prompt(stream.prompt_tokens());
    {
        json::Array stages;
        for (const auto name : chain.stage_names()) {
            stages.emplace_back(std::string(name));
        }
        bus.emit("sampling.configured", ctx,
                 json::Object{{"sampler", "sonder"},
                              {"selector", config.temperature == 0.0f ? "greedy" : "distribution"},
                              {"stages", std::move(stages)},
                              {"seed", config.seed},
                              {"vocab_size", stream.vocab_size()}},
                 TelemetryLevel::metrics);
    }

    smp::StopSequenceMatcher stop(config.stop);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int32_t i = 0; i < config.max_tokens; ++i) {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }
        if (!gate()) {
            stats.stop_reason = StopReason::callback;
            return stats;
        }
        auto logits = stream.next_logits(cancel);
        if (!logits.ok()) {
            return logits.status();
        }
        const smp::SampleResult sampled = chain.sample(logits.value());
        if (!sampled.ok()) {
            const bool no_viable = sampled.status == smp::SampleStatus::NoViableCandidates;
            const ErrorCode code = no_viable ? ErrorCode::invalid_argument : ErrorCode::backend_error;
            bus.emit("sampling.failed", ctx,
                     json::Object{{"status", no_viable ? "no_viable_candidates" : "empty_logits"},
                                  {"error_code", to_string(code)},
                                  {"position", i}},
                     TelemetryLevel::metrics);
            return Status(code, no_viable ? "sampling failed: no viable candidates (bias, penalties or "
                                            "constraints excluded every token)"
                                          : "sampling failed: backend returned empty logits");
        }
        chain.accept(sampled.token);
        produced(sampled.token);
        if (stream.is_end_of_generation(sampled.token) || chain.is_stop_token(sampled.token)) {
            stats.stop_reason = StopReason::end_of_sequence;
            break;
        }
        std::string piece;
        if (Status st = stream.accept(sampled.token, piece); !st.ok()) {
            return st;
        }
        ++stats.completion_tokens;
        const smp::StopCheck check = stop.feed(piece);
        if (!check.emit.empty()) {
            const TokenChunk chunk{check.emit, stats.chunks++};
            if (!deliver(chunk, sampled.token, sampled.probability)) {
                stats.stop_reason = StopReason::callback;
                break;
            }
        }
        if (check.stopped) {
            stats.stop_reason = StopReason::stop_sequence;
            if (check.stop_index < config.stop.size()) {
                stats.matched_stop = config.stop[check.stop_index];
            }
            break;
        }
    }
    if (stats.stop_reason == StopReason::none) {
        stats.stop_reason = StopReason::max_tokens;
    }
    if (stats.stop_reason == StopReason::max_tokens || stats.stop_reason == StopReason::end_of_sequence) {
        const std::string rest = stop.flush();
        if (!rest.empty()) {
            const TokenChunk chunk{rest, stats.chunks++};
            (void)deliver(chunk, std::nullopt, 0.0f);
        }
    }
    stats.eval_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    return stats;
}

}  // namespace
#endif

bool Session::last_scheduler_rejected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_scheduler_rejected_;
}

Result<GenerationResult> Session::generate(const std::string& prompt, const TokenCallback& on_chunk,
                                           const std::optional<SamplingConfig>& sampling_override,
                                           const RequestOptions& request_options) {
    return run_request("generate", prompt, nullptr, on_chunk, sampling_override, request_options);
}

Result<GenerationResult> Session::chat(const std::vector<ChatMessage>& messages, const TokenCallback& on_chunk,
                                       const std::optional<SamplingConfig>& sampling_override,
                                       const RequestOptions& request_options) {
    if (Status st = validate_chat_messages(messages); !st.ok()) {
        return st;
    }
    return run_request("chat", format_chat_prompt(messages), &messages, on_chunk, sampling_override,
                       request_options);
}

Result<GenerationResult> Session::run_request(const char* kind, const std::string& prompt,
                                              const std::vector<ChatMessage>* messages, const TokenCallback& on_chunk,
                                              const std::optional<SamplingConfig>& sampling_override,
                                              const RequestOptions& request_options) {
    SamplingConfig sampling = sampling_override.value_or(options_.sampling);
    if (auto st = validate(sampling); !st.ok()) {
        return st;
    }
    GenerateRequest request;
    request.request_id = request_options.request_id && !request_options.request_id->empty()
                             ? *request_options.request_id
                             : make_id("req");
    request.prompt = prompt;
    request.sampling = sampling;

    CancellationToken token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        state_ = SessionState::running;
        active_cancel_.emplace();
        token = active_cancel_->token();
        cancel_requested_ns_.store(0);
        ++requests_started_;
    }

    TelemetryBus& bus = engine_.telemetry();
    const auto ctx = telemetry_context(request.request_id);
    const bool emit_tokens = bus.enabled(TelemetryLevel::standard);
    const bool capture_text = bus.options().capture_text;
    BackendModel& backend_model = model_->backend_model();
    // Chat with a backend-native template bypasses the generic prompt (and
    // therefore Sonder's token-level sampler, which works on that prompt).
    const bool native_chat = messages != nullptr && backend_model.has_native_chat();
    const auto& parent = request_options.parent_request_id;
    const auto with_parent = [&parent](json::Object attrs) {
        if (parent) {
            attrs.set("parent_request_id", *parent);
        }
        return attrs;
    };
    const auto with_admission = [&](json::Object attrs, double queue_ms = 0.0) {
        if (request_options.priority_class) {
            attrs.set("priority_class", to_string(*request_options.priority_class));
            attrs.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                   {"queue_ms", queue_ms}});
        }
        return attrs;
    };

    {
        json::Object queued{{"kind", kind},
                            {"priority", options_.priority},
                            {"workload", to_string(options_.workload)},
                            {"prompt_bytes", prompt.size()}};
        if (messages != nullptr) {
            queued.set("messages", messages->size());
        }
        bus.emit("request.queued", ctx,
                 with_parent(with_admission(std::move(queued), request_options.admission_queue_ms)),
                 TelemetryLevel::metrics);
    }
    const auto t_start = std::chrono::steady_clock::now();

    GenerationResult result;
    result.request_id = request.request_id;
    result.scheduling.queue_ms = request_options.admission_queue_ms;
    Status failure;
    bool pre_failed = false;
    bool scheduler_rejected = false;

    // Sampling path: Sonder's sampler chain when the backend exposes logits.
    std::unique_ptr<TokenStream> stream;
    const auto backend = engine_.find_backend(model_->backend_name());
    const bool defer_stream_open = engine_.scheduling_options().strict_priority_admission &&
                                   !native_chat && backend &&
                                   backend->capabilities().has(Capability::token_logits);
#if defined(SONDER_HAS_SAMPLER_CHAIN)
    if (!defer_stream_open && !native_chat && backend && backend->capabilities().has(Capability::token_logits)) {
        auto opened = backend_model.open_token_stream(request);
        if (opened.ok()) {
            stream = std::move(opened).value();
        } else {
            failure = opened.status();
            pre_failed = true;
        }
    }
#endif
    result.scheduling.sonder_sampled = stream != nullptr;

    // Scheduling: the request enters the scheduler queue and holds KV blocks.
    detail::RequestRuntime* runtime = engine_.request_runtime();
    std::optional<std::uint64_t> sched_id;
    // Per-chunk grants pace an in-process backend's decode; a remote process
    // batches on its own, so by default it is only admitted and accounted.
    const SchedulerMode mode = engine_.scheduling_options().mode;
    const bool remote = backend && backend->capabilities().has(Capability::remote_process);
    const bool gated = mode == SchedulerMode::gate || (mode == SchedulerMode::automatic && !remote);
    if (runtime != nullptr && !pre_failed) {
        detail::RuntimeRequestSpec spec;
        spec.context = ctx;
        spec.workload = options_.workload;
        spec.priority = options_.priority;
        spec.priority_class = request_options.priority_class;
        spec.preserve_numeric_priority = request_options.preserve_numeric_priority;
        spec.admission = request_options.admission;
        if (stream) {
            spec.prompt_tokens = stream->prompt_tokens();
            spec.exact_tokens = true;
        } else if (!native_chat) {
            // Exact only for the prompt the backend actually receives: a native
            // chat template differs from the generic prompt, so native chat is
            // accounted approximately below.
            if (auto tk = backend_model.tokenize(prompt); tk.ok() && !tk.value().empty()) {
                spec.prompt_tokens = std::move(tk).value();
                spec.exact_tokens = true;
            }
        }
        if (spec.prompt_tokens.empty()) {
            spec.prompt_tokens = detail::accounting_tokenize(prompt);
            spec.exact_tokens = false;
        }
        spec.max_new_tokens = static_cast<std::uint32_t>(sampling.max_tokens);
        // A requested num_ctx narrows the model's context window (0 = model default).
        std::uint64_t ctx_limit = model_->descriptor().context_length;
        if (sampling.num_ctx > 0) {
            const auto requested = static_cast<std::uint64_t>(sampling.num_ctx);
            ctx_limit = ctx_limit == 0 ? requested : std::min(ctx_limit, requested);
        }
        spec.context_limit = static_cast<std::uint32_t>(std::min<std::uint64_t>(ctx_limit, 0xFFFFFFFFull));
        spec.fingerprint = model_fingerprint(*model_);
        spec.architecture = model_->descriptor().architecture;
        spec.gated = gated;
        result.scheduling.accounted_prompt_tokens = spec.prompt_tokens.size();
        result.scheduling.exact_prompt_tokens = spec.exact_tokens;
        const bool exact = spec.exact_tokens;
        auto submitted = runtime->submit(std::move(spec));
        if (submitted.ok() && submitted.value().bypassed) {
            // Ungated prompt beyond the logical KV pool: runs unscheduled.
            result.scheduling.accounted_prompt_tokens = 0;
            result.scheduling.exact_prompt_tokens = false;
        } else if (submitted.ok()) {
            sched_id = submitted.value().id;
            result.scheduling.scheduled = true;
            // The runtime clamps the budget to the context window (num_ctx)
            // and the KV pool. Generation honours it: past it the scheduler
            // stops gating, so nothing else would stop the request short of
            // max_tokens. Only with exact prompt tokens; an approximate count
            // must not truncate a reply the backend itself would allow.
            const auto budget = static_cast<std::int32_t>(
                std::min<std::uint32_t>(submitted.value().max_new_tokens, 0x7FFFFFFFu));
            if (exact && budget > 0 && budget < sampling.max_tokens) {
                sampling.max_tokens = budget;
                request.sampling.max_tokens = budget;
            }
        } else {
            failure = submitted.status();
            pre_failed = true;
            scheduler_rejected = true;
        }
    }

    {
        json::Object started{{"kind", kind},
                             {"sampling", sampling_json(sampling)},
                             {"scheduled", sched_id.has_value()},
                             {"sampler", (stream || defer_stream_open) ? "sonder" : "backend"}};
        if (sched_id) {
            started.set("scheduler_mode", gated ? "gate" : "account");
        }
        if (messages != nullptr) {
            started.set("chat_template", native_chat ? "native" : "generic");
        }
        bus.emit("request.started", ctx,
                 with_parent(with_admission(std::move(started), request_options.admission_queue_ms)),
                 TelemetryLevel::metrics);
    }

    std::chrono::steady_clock::time_point t_first{};
    bool have_first = false;
    std::uint64_t delivered = 0;

    // Telemetry + user callback for one visible chunk (not gated).
    const auto deliver = [&](const TokenChunk& chunk, std::optional<TokenId> token_id, float probability) -> bool {
        const auto now = std::chrono::steady_clock::now();
        if (!have_first) {
            have_first = true;
            t_first = now;
            result.ttft_ms = ms_between(t_start, now);
            bus.emit("inference.decode.started", ctx, json::Object{{"ttft_ms", result.ttft_ms}},
                     TelemetryLevel::metrics);
        }
        if (!chunk.text.empty()) {
            result.text.append(chunk.text.data(), chunk.text.size());
        }
        if (!chunk.reasoning.empty()) {
            result.reasoning.append(chunk.reasoning.data(), chunk.reasoning.size());
        }
        ++delivered;
        if (emit_tokens) {
            json::Object attrs{{"index", chunk.index},
                               {"bytes", chunk.text.size()},
                               {"elapsed_ms", ms_between(t_start, now)}};
            if (token_id) {
                // Sonder-sampled: exactly one token per event.
                attrs.set("unit", "token");
                attrs.set("count", 1);
                attrs.set("token_id", *token_id);
                attrs.set("probability", probability);
            } else {
                // Backend-streamed chunk: token count per chunk is unknown.
                attrs.set("unit", "chunk");
            }
            if (capture_text) {
                attrs.set("text", std::string(chunk.text));
            }
            bus.emit("inference.token.generated", ctx, std::move(attrs), TelemetryLevel::standard);
        }
        if (on_chunk) {
            return on_chunk(chunk);
        }
        return true;
    };

    // Scheduler gate: every generated token waits for a grant (gated), or
    // the request waits once for admission (ungated, before the backend call).
    std::optional<Status> gate_failure;
    const auto prepare_backend = [&]() -> Status {
        // Library callers retain the old per-chunk gate unless they explicitly
        // enable strict admission. Account mode always waits once up front.
        if (sched_id && (!gated || engine_.scheduling_options().strict_priority_admission)) {
            Status st = gated ? runtime->admit_backend(*sched_id, token, request_options.deadline)
                              : runtime->acquire_token(*sched_id, token, request_options.deadline);
            if (!st.ok()) return st;
        } else if (!sched_id && request_options.admission) {
            Status st = request_options.admission->wait(token);
            if (!st.ok()) return st;
        }
        if (request_options.deadline && std::chrono::steady_clock::now() >= *request_options.deadline)
            return Status(ErrorCode::timeout, "request deadline exceeded before backend start");
        if (token.cancelled()) return Status(ErrorCode::cancelled, "cancelled before backend start");
        if (request_options.on_backend_start) request_options.on_backend_start();
        // The host can notice a disconnect at the exact start boundary and
        // cancel here, even if its earlier cancel raced session initialization.
        if (token.cancelled()) return Status(ErrorCode::cancelled, "cancelled before backend start");
        return Status::success();
    };
    const auto gate = [&]() -> bool {
        if (!sched_id || !gated) {
            return true;
        }
        Status st = runtime->acquire_token(*sched_id, token);
        if (st.ok()) {
            return true;
        }
        if (st.code() != ErrorCode::cancelled) {
            gate_failure = std::move(st);
        }
        return false;
    };
    const auto produced = [&](TokenId t) {
        if (sched_id && gated) {
            runtime->token_produced(*sched_id, t);
        }
    };

    Result<GenerateStats> generated = GenerateStats{};
    Status prepared = pre_failed ? failure : prepare_backend();
    if (!prepared.ok()) {
        generated = prepared;
    } else {
#if defined(SONDER_HAS_SAMPLER_CHAIN)
        if (defer_stream_open) {
            auto opened = backend_model.open_token_stream(request);
            if (opened.ok()) {
                stream = std::move(opened).value();
                result.scheduling.sonder_sampled = true;
            } else {
                prepared = opened.status();
            }
        }
#endif
        if (!prepared.ok()) {
            generated = prepared;
        } else if (stream) {
#if defined(SONDER_HAS_SAMPLER_CHAIN)
            generated = run_sonder_sampling(*stream, sampling, token, bus, ctx, gate, produced, deliver);
#endif
        } else {
            const auto on_backend_chunk = [&](const TokenChunk& chunk) -> bool {
                if (!gated || !sched_id) return deliver(chunk, std::nullopt, 0.0f);
                if (!gate()) return false;
                produced(detail::accounting_chunk_token(chunk.text));
                return deliver(chunk, std::nullopt, 0.0f);
            };
            if (native_chat) {
                ChatRequest chat_request;
                chat_request.request_id = request.request_id;
                chat_request.messages = *messages;
                chat_request.sampling = sampling;
                chat_request.session_key = request_options.session_key;
                chat_request.thinking = request_options.thinking;
                chat_request.separate_reasoning = request_options.separate_reasoning;
                chat_request.reasoning_budget_tokens = request_options.reasoning_budget_tokens;
                chat_request.reasoning_budget_message = request_options.reasoning_budget_message;
                generated = backend_model.chat(chat_request, token, on_backend_chunk);
            } else {
                generated = backend_model.generate(request, token, on_backend_chunk);
            }
        }
    }
    const auto t_end = std::chrono::steady_clock::now();
    result.total_ms = ms_between(t_start, t_end);

    const bool was_cancelled =
        token.cancelled() || (!generated.ok() && generated.status().code() == ErrorCode::cancelled);

    RequestOutcome outcome = RequestOutcome::completed;
    if (gate_failure && !token.cancelled()) {
        outcome = RequestOutcome::failed;
        failure = *gate_failure;
        if (generated.ok()) {
            result.stats = generated.value();
        }
    } else if (was_cancelled) {
        outcome = RequestOutcome::cancelled;
        if (generated.ok()) {
            result.stats = generated.value();
        }
        result.stats.stop_reason = StopReason::cancelled;
        if (result.stats.chunks == 0) {
            result.stats.chunks = delivered;
        }
        if (!result.stats.token_counts_from_backend && result.stats.completion_tokens == 0) {
            result.stats.completion_tokens = delivered;
        }
    } else if (!generated.ok()) {
        outcome = RequestOutcome::failed;
        failure = generated.status();
    } else {
        result.stats = generated.value();
    }
    result.outcome = outcome;

    if (sched_id) {
        const detail::RuntimeRequestSummary sched = runtime->finish(*sched_id, outcome);
        result.scheduling.preemptions = sched.preemptions;
        result.scheduling.queue_ms = request_options.admission_queue_ms + sched.queue_ms;
        result.scheduling.reused_prompt_tokens = sched.reused_prompt_tokens;
    }

    if (request_options.admission) result.scheduling.queue_ms = request_options.admission->queue_ms();

    const std::uint64_t cancel_ns = cancel_requested_ns_.load();
    const std::uint64_t now_ns = monotonic_ns();

    json::Object summary{{"outcome", to_string(outcome)},
                         {"stop_reason", to_string(result.stats.stop_reason)},
                         {"prompt_tokens", result.stats.prompt_tokens},
                         {"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"token_counts_from_backend", result.stats.token_counts_from_backend},
                         {"ttft_ms", result.ttft_ms},
                         {"total_ms", result.total_ms},
                         {"scheduled", result.scheduling.scheduled},
                         {"sampler", result.scheduling.sonder_sampled ? "sonder" : "backend"}};
    backend_observations(summary, result.stats);
    if (request_options.reasoning_budget_tokens) summary.set("reasoning_budget", *request_options.reasoning_budget_tokens);
    if (result.stats.backend_timings) {
        const auto& t = *result.stats.backend_timings;
        if (t.cache_n) summary.set("backend_cache_n", *t.cache_n);
        if (t.prompt_n) summary.set("backend_prompt_n", *t.prompt_n);
        if (t.predicted_n) summary.set("backend_predicted_n", *t.predicted_n);
        if (t.draft_n) summary.set("backend_draft_n", *t.draft_n);
        if (t.draft_n_accepted) summary.set("backend_draft_accepted", *t.draft_n_accepted);
    }
    if (result.scheduling.scheduled) {
        summary.set("queue_ms", result.scheduling.queue_ms);
        summary.set("preemptions", result.scheduling.preemptions);
        summary.set("accounted_prompt_tokens", result.scheduling.accounted_prompt_tokens);
        summary.set("reused_prompt_tokens", result.scheduling.reused_prompt_tokens);
    }
    if (request_options.priority_class) {
        summary.set("priority_class", to_string(*request_options.priority_class));
        summary.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                {"queue_ms", result.scheduling.queue_ms}});
    }

    if (outcome == RequestOutcome::completed) {
        json::Object pre{{"prompt_tokens", result.stats.prompt_tokens},
                         {"token_counts_from_backend", result.stats.token_counts_from_backend},
                         {"ttft_ms", result.ttft_ms}};
        if (result.stats.cached_tokens) pre.set("backend_cached_tokens", *result.stats.cached_tokens);
        if (result.stats.prompt_eval_ns > 0) {
            pre.set("backend_prompt_eval_ms", static_cast<double>(result.stats.prompt_eval_ns) / 1e6);
        }
        if (result.scheduling.scheduled) {
            pre.set("reused_prompt_tokens", result.scheduling.reused_prompt_tokens);
        }
        bus.emit("inference.prefill.completed", ctx, std::move(pre), TelemetryLevel::metrics);
        const double decode_ms = have_first ? ms_between(t_first, t_end) : 0.0;
        json::Object dec{{"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"decode_wall_ms", decode_ms}};
        backend_observations(dec, result.stats);
        if (result.stats.eval_ns > 0) {
            dec.set("backend_eval_ms", static_cast<double>(result.stats.eval_ns) / 1e6);
            dec.set("backend_tokens_per_sec",
                    static_cast<double>(result.stats.completion_tokens) * 1e9 / static_cast<double>(result.stats.eval_ns));
        }
        if (result.stats.prompt_eval_ns > 0) {
            dec.set("backend_prompt_eval_ms", static_cast<double>(result.stats.prompt_eval_ns) / 1e6);
        }
        bus.emit("inference.decode.completed", ctx, std::move(dec), TelemetryLevel::metrics);
        bus.emit("request.completed", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    } else if (outcome == RequestOutcome::cancelled) {
        if (cancel_ns != 0 && now_ns >= cancel_ns) {
            summary.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
        }
        bus.emit("request.cancelled", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    } else {
        summary.set("error_code", to_string(failure.code()));
        summary.set("error", failure.message());
        if (scheduler_rejected) {
            summary.set("scheduler_rejected", true);
        }
        bus.emit("request.failed", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        last_outcome_ = outcome;
        last_scheduler_rejected_ = outcome == RequestOutcome::failed && scheduler_rejected;
        if (state_ == SessionState::running) {
            state_ = SessionState::idle;
        }
    }

    if (outcome == RequestOutcome::failed) {
        return failure;
    }
    return result;
}

}  // namespace sonder::inference
