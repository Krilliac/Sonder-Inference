#include "sonder/inference/session.hpp"

#include <chrono>

#include "sonder/inference/engine.hpp"

namespace sonder::inference {

const char* to_string(SessionState state) noexcept {
    switch (state) {
        case SessionState::idle: return "idle";
        case SessionState::running: return "running";
        case SessionState::closed: return "closed";
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
json::Value sampling_json(const SamplingConfig& s) {
    json::Object o{{"temperature", s.temperature}, {"top_p", s.top_p},     {"top_k", s.top_k},
                   {"min_p", s.min_p},             {"repeat_penalty", s.repeat_penalty},
                   {"seed", s.seed},               {"max_tokens", s.max_tokens}};
    return o;
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
    ctx.run_id = options_.run_id;
    ctx.request_id = request_id;
    ctx.agent_id = options_.agent_id;
    ctx.task_id = options_.task_id;
    ctx.model_instance_id = model_->instance_id();
    ctx.device_id = model_->device_id();
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

Result<GenerationResult> Session::generate(const std::string& prompt, const TokenCallback& on_chunk,
                                           const std::optional<SamplingConfig>& sampling_override) {
    const SamplingConfig sampling = sampling_override.value_or(options_.sampling);
    if (auto st = validate(sampling); !st.ok()) {
        return st;
    }
    GenerateRequest request;
    request.request_id = make_id("req");
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

    bus.emit("request.queued", ctx,
             json::Object{{"kind", "generate"}, {"priority", options_.priority}, {"prompt_bytes", prompt.size()}},
             TelemetryLevel::metrics);
    const auto t_start = std::chrono::steady_clock::now();
    bus.emit("request.started", ctx, json::Object{{"kind", "generate"}, {"sampling", sampling_json(sampling)}},
             TelemetryLevel::metrics);

    GenerationResult result;
    result.request_id = request.request_id;
    std::chrono::steady_clock::time_point t_first{};
    bool have_first = false;
    std::uint64_t delivered = 0;

    const TokenCallback wrapped = [&](const TokenChunk& chunk) -> bool {
        const auto now = std::chrono::steady_clock::now();
        if (!have_first) {
            have_first = true;
            t_first = now;
            result.ttft_ms = ms_between(t_start, now);
            bus.emit("inference.decode.started", ctx, json::Object{{"ttft_ms", result.ttft_ms}},
                     TelemetryLevel::metrics);
        }
        result.text.append(chunk.text.data(), chunk.text.size());
        ++delivered;
        if (emit_tokens) {
            json::Object attrs{{"index", chunk.index},
                               {"bytes", chunk.text.size()},
                               {"elapsed_ms", ms_between(t_start, now)}};
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

    auto generated = model_->backend_model().generate(request, token, wrapped);
    const auto t_end = std::chrono::steady_clock::now();
    result.total_ms = ms_between(t_start, t_end);

    const bool was_cancelled =
        token.cancelled() || (!generated.ok() && generated.status().code() == ErrorCode::cancelled);

    RequestOutcome outcome = RequestOutcome::completed;
    Status failure;
    if (was_cancelled) {
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

    const std::uint64_t cancel_ns = cancel_requested_ns_.load();
    const std::uint64_t now_ns = monotonic_ns();

    json::Object summary{{"outcome", to_string(outcome)},
                         {"stop_reason", to_string(result.stats.stop_reason)},
                         {"prompt_tokens", result.stats.prompt_tokens},
                         {"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"token_counts_from_backend", result.stats.token_counts_from_backend},
                         {"ttft_ms", result.ttft_ms},
                         {"total_ms", result.total_ms}};

    if (outcome == RequestOutcome::completed) {
        const double decode_ms = have_first ? ms_between(t_first, t_end) : 0.0;
        json::Object dec{{"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"decode_wall_ms", decode_ms}};
        if (result.stats.eval_ns > 0) {
            dec.set("backend_eval_ms", static_cast<double>(result.stats.eval_ns) / 1e6);
            dec.set("backend_tokens_per_sec",
                    static_cast<double>(result.stats.completion_tokens) * 1e9 / static_cast<double>(result.stats.eval_ns));
        }
        if (result.stats.prompt_eval_ns > 0) {
            dec.set("backend_prompt_eval_ms", static_cast<double>(result.stats.prompt_eval_ns) / 1e6);
        }
        bus.emit("inference.decode.completed", ctx, std::move(dec), TelemetryLevel::metrics);
        bus.emit("request.completed", ctx, std::move(summary), TelemetryLevel::metrics);
    } else if (outcome == RequestOutcome::cancelled) {
        if (cancel_ns != 0 && now_ns >= cancel_ns) {
            summary.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
        }
        bus.emit("request.cancelled", ctx, std::move(summary), TelemetryLevel::metrics);
    } else {
        summary.set("error_code", to_string(failure.code()));
        summary.set("error", failure.message());
        bus.emit("request.failed", ctx, std::move(summary), TelemetryLevel::metrics);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        last_outcome_ = outcome;
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
