#include "engine/request_runtime.hpp"

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>

#include "engine/kv_capacity_adapter.hpp"
#include "sonder/inference/cache/kv_cache_manager.hpp"
#include "sonder/inference/scheduler/scheduler.hpp"

namespace sonder::inference::detail {
namespace {

namespace sch = scheduler;
namespace kvc = cache;

class SteadyClock final : public sch::Clock {
public:
    SteadyClock() : origin_(std::chrono::steady_clock::now()) {}
    [[nodiscard]] sch::TimeUs now() const noexcept override {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - origin_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point origin_;
};

std::uint32_t at_least_one(std::uint32_t v) { return v == 0 ? 1u : v; }

kvc::KvCacheConfig make_cache_config(const SchedulingOptions& o) {
    kvc::KvCacheConfig c;
    c.block_size_tokens = at_least_one(o.kv_block_size_tokens);
    c.num_blocks = at_least_one(o.kv_num_blocks);
    c.enable_prefix_caching = o.prefix_caching;
    return c;
}

sch::SchedulerConfig make_scheduler_config(const SchedulingOptions& o) {
    sch::SchedulerConfig c;
    c.max_running_sequences = at_least_one(o.max_running_sequences);
    c.max_step_sequences = at_least_one(o.max_step_sequences);
    c.max_step_tokens = at_least_one(o.max_step_tokens);
    c.prefill_chunk_tokens = at_least_one(o.prefill_chunk_tokens);
    c.max_step_prefill_tokens = std::max(c.prefill_chunk_tokens, c.max_step_tokens / 2);
    c.admission_watermark_blocks = o.admission_watermark_blocks;
    c.max_requeue_count = o.max_requeue_count;
    c.enable_priority_preemption = o.enable_priority_preemption;
    return c;
}

std::string str(std::string_view v) { return std::string(v); }

double ms_since(sch::TimeUs from_us, sch::TimeUs to_us) { return static_cast<double>(to_us - from_us) / 1000.0; }

struct Req {
    sch::RequestId id = 0;
    TelemetryContext ctx;
    std::vector<kvc::TokenId> tokens;  // prompt + produced tokens
    std::size_t prompt_tokens = 0;
    kvc::CacheFingerprint fingerprint{};
    kvc::Priority cache_priority = 0;
    sch::TimeUs submitted_at = 0;
    std::uint32_t credits = 0;  // grants not yet acquired by the session
    std::uint32_t pending = 0;  // grants not yet produced
    bool done = false;          // session finished; no more tokens will come
    bool passthrough = false;   // scheduler completed it; later tokens are not gated
    std::optional<Status> failure;
    bool in_cache = false;
    std::size_t cached_tokens = 0;
    std::uint64_t admission_reused = 0;
    bool admitted_once = false;
    RuntimeRequestSummary summary;
};

class SchedulingRuntime final : public RequestRuntime {
public:
    SchedulingRuntime(const SchedulingOptions& options, TelemetryBus& bus, TelemetryContext engine_context)
        : bus_(bus),
          engine_ctx_(std::move(engine_context)),
          cache_(make_cache_config(options)),
          adapter_(cache_),
          scheduler_(make_scheduler_config(options), clock_, adapter_) {
        cache_.set_event_listener([this](const kvc::CacheEvent& e) { on_cache_event(e); });
        // Free a request's sequence the moment the scheduler drops its
        // reservation so preemption sees the blocks come back (recompute).
        adapter_.set_release_hook([this](sch::RequestId id) {
            if (Req* r = find(id)) {
                free_cache(*r, release_reason_);
            }
        });
        thread_ = std::thread([this] { loop(); });
    }

    ~SchedulingRuntime() override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    Result<std::uint64_t> submit(RuntimeRequestSpec spec) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (spec.prompt_tokens.empty()) {
            return Status(ErrorCode::invalid_argument, "request has no prompt tokens");
        }
        const std::uint64_t block = cache_.config().block_size_tokens;
        const std::uint64_t capacity = block * cache_.config().num_blocks;
        const std::uint64_t prompt = spec.prompt_tokens.size();
        const sch::RequestId id = next_id_++;

        if (prompt + 1 > capacity) {
            bus_.emit("scheduler.rejected", spec.context,
                      json::Object{{"scheduler_request_id", id},
                                   {"reason", "never_fits"},
                                   {"prompt_tokens", prompt},
                                   {"kv_capacity_tokens", capacity}},
                      TelemetryLevel::metrics);
            return Status(ErrorCode::invalid_argument, "prompt needs " + std::to_string(prompt + 1) +
                                                           " KV tokens but the engine KV pool holds " +
                                                           std::to_string(capacity));
        }
        std::uint64_t max_new = std::max<std::uint32_t>(1, spec.max_new_tokens);
        if (spec.context_limit > 0) {
            max_new = std::min<std::uint64_t>(max_new, spec.context_limit > prompt ? spec.context_limit - prompt : 1);
        }
        max_new = std::max<std::uint64_t>(1, std::min<std::uint64_t>(max_new, capacity - prompt));

        const int rank = static_cast<int>(spec.workload) - spec.priority;
        sch::RequestSpec rs;
        rs.id = id;
        rs.workload = static_cast<sch::WorkloadClass>(static_cast<int>(spec.workload));
        if (spec.priority != 0) {
            rs.priority_override = rank;
        }
        rs.task_id = spec.context.task_id.value_or("");
        rs.prompt_tokens = static_cast<sch::TokenCount>(prompt);
        rs.max_new_tokens = static_cast<sch::TokenCount>(max_new);
        rs.cancellable = true;
        const sch::SubmitResult submitted = scheduler_.submit(rs);
        if (!submitted.accepted) {
            bus_.emit("scheduler.rejected", spec.context,
                      json::Object{{"scheduler_request_id", id},
                                   {"reason", str(sch::to_string(submitted.reason))},
                                   {"prompt_tokens", prompt}},
                      TelemetryLevel::metrics);
            const bool caller_error = submitted.reason == sch::FailureReason::NeverFits ||
                                      submitted.reason == sch::FailureReason::InvalidRequest;
            return Status(caller_error ? ErrorCode::invalid_argument : ErrorCode::internal,
                          "scheduler rejected the request: " + str(sch::to_string(submitted.reason)));
        }

        Req r;
        r.id = id;
        r.ctx = spec.context;
        r.tokens = std::move(spec.prompt_tokens);
        r.prompt_tokens = r.tokens.size();
        r.fingerprint = kvc::CacheFingerprint{spec.fingerprint};
        r.cache_priority = static_cast<kvc::Priority>(std::clamp(6 - rank, 0, 255));
        r.submitted_at = clock_.now();
        bus_.emit("scheduler.enqueued", r.ctx,
                  json::Object{{"scheduler_request_id", id},
                               {"workload", str(sch::to_string(rs.workload))},
                               {"priority_rank", rank},
                               {"prompt_tokens", prompt},
                               {"exact_prompt_tokens", spec.exact_tokens},
                               {"max_new_tokens", max_new},
                               {"context_limit", spec.context_limit}},
                  TelemetryLevel::metrics);
        reqs_.emplace(id, std::move(r));
        ++generation_;
        cv_.notify_all();
        return static_cast<std::uint64_t>(id);
    }

    Status acquire_token(std::uint64_t id, const CancellationToken& cancel) override {
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            auto it = reqs_.find(id);
            if (it == reqs_.end()) {
                return Status(ErrorCode::internal, "unknown scheduled request");
            }
            Req& r = it->second;
            if (r.failure) {
                return *r.failure;
            }
            if (r.passthrough) {
                return Status::success();
            }
            if (r.credits > 0) {
                --r.credits;
                return Status::success();
            }
            if (cancel.cancelled()) {
                return Status(ErrorCode::cancelled, "cancelled while waiting for the scheduler");
            }
            if (stopping_) {
                return Status(ErrorCode::unavailable, "engine is shutting down");
            }
            // Cancellation is cooperative (token polled), grants are notified.
            cv_.wait_for(lock, std::chrono::milliseconds(2));
        }
    }

    void token_produced(std::uint64_t id, TokenId token) override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = reqs_.find(id);
            if (it == reqs_.end()) {
                return;
            }
            Req& r = it->second;
            r.tokens.push_back(token);
            if (r.pending > 0) {
                --r.pending;
            }
            ++generation_;
        }
        cv_.notify_all();
    }

    RuntimeRequestSummary finish(std::uint64_t id, RequestOutcome outcome) override {
        RuntimeRequestSummary summary;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = reqs_.find(id);
            if (it == reqs_.end()) {
                return summary;
            }
            Req& r = it->second;
            r.done = true;
            const auto state = scheduler_.state(id);
            bool terminal = !state || sch::is_terminal(*state);
            if (!terminal) {
                const bool running = *state == sch::RequestState::Prefill || *state == sch::RequestState::Decode;
                if (outcome != RequestOutcome::completed || !running) {
                    // Releases the KV reservation (and the sequence) immediately.
                    release_reason_ = to_string(outcome);
                    (void)scheduler_.cancel(id);
                    terminal = true;
                }
                // Otherwise the next step reports it as finished early and the
                // scheduler records a normal completion.
            }
            free_cache(r, to_string(outcome));
            summary = r.summary;
            if (terminal) {
                reqs_.erase(it);
            }
            ++generation_;
        }
        cv_.notify_all();
        return summary;
    }

    KvUsage kv_usage() const override {
        std::lock_guard<std::mutex> lock(mu_);
        const kvc::KvCacheStats s = cache_.stats();
        KvUsage u;
        u.active = true;
        u.block_size_tokens = s.block_size_tokens;
        u.total_blocks = s.total_blocks;
        u.free_blocks = s.free_blocks;
        u.cached_blocks = s.cached_blocks;
        u.pinned_blocks = s.pinned_blocks;
        u.shared_blocks = s.shared_blocks;
        u.sequences = s.sequences;
        u.prefix_hit_blocks = s.prefix_hit_blocks;
        u.avoided_prefill_tokens = s.avoided_prefill_tokens;
        u.evictions = s.evictions;
        return u;
    }

private:
    Req* find(sch::RequestId id) {
        auto it = reqs_.find(id);
        return it == reqs_.end() ? nullptr : &it->second;
    }

    void on_cache_event(const kvc::CacheEvent& e) {
        if (e.kind == kvc::CacheEvent::Kind::evicted) {
            ++evicted_pending_;
        } else if (e.kind == kvc::CacheEvent::Kind::pressure_changed) {
            pressure_pending_ = e.pressure;
        }
    }

    // Emits aggregated eviction/pressure events after a cache mutation.
    void flush_cache_events(const TelemetryContext& ctx, sch::RequestId trigger) {
        if (evicted_pending_ > 0) {
            bus_.emit("kv.evicted", ctx,
                      json::Object{{"blocks", evicted_pending_},
                                   {"trigger_scheduler_request_id", trigger},
                                   {"total_evictions", cache_.stats().evictions}},
                      TelemetryLevel::metrics);
            evicted_pending_ = 0;
        }
        if (pressure_pending_) {
            const auto s = cache_.stats();
            bus_.emit("kv.pressure", engine_ctx_,
                      json::Object{{"level", kvc::to_string(*pressure_pending_)},
                                   {"utilization", s.utilization},
                                   {"pinned_blocks", s.pinned_blocks},
                                   {"total_blocks", s.total_blocks}},
                      TelemetryLevel::metrics);
            pressure_pending_.reset();
        }
    }

    void free_cache(Req& r, const char* reason) {
        if (!r.in_cache) {
            return;
        }
        const std::size_t blocks = cache_.block_table(r.id).size();
        (void)cache_.free_sequence(r.id);
        r.in_cache = false;
        r.cached_tokens = 0;
        bus_.emit("kv.freed", r.ctx,
                  json::Object{{"scheduler_request_id", r.id}, {"blocks", blocks}, {"reason", reason}},
                  TelemetryLevel::metrics);
        flush_cache_events(r.ctx, r.id);
    }

    void fail(Req& r, Status status) {
        if (!r.failure) {
            r.failure = std::move(status);
        }
        free_cache(r, "failed");
    }

    // Appends tokens [offset, offset + n) of the request to its cache sequence.
    bool append(Req& r, std::size_t offset, std::size_t n, kvc::AppendResult& result) {
        if (!r.in_cache) {
            kvc::SequenceOptions so;
            so.fingerprint = r.fingerprint;
            so.priority = r.cache_priority;
            if (Status st = cache_.add_sequence(r.id, so); !st.ok()) {
                fail(r, Status(ErrorCode::internal, "KV cache: " + st.message()));
                return false;
            }
            r.in_cache = true;
            r.cached_tokens = 0;
        }
        if (offset != r.cached_tokens || offset >= r.tokens.size()) {
            return true;  // nothing to account (e.g. finished request)
        }
        n = std::min(n, r.tokens.size() - offset);
        const std::span<const kvc::TokenId> span(r.tokens.data() + offset, n);
        Status st = cache_.append_tokens(r.id, span, &result);
        flush_cache_events(r.ctx, r.id);
        if (!st.ok()) {
            release_reason_ = "failed";
            (void)scheduler_.cancel(r.id);
            fail(r, Status(ErrorCode::unavailable, "KV cache exhausted: " + st.message()));
            return false;
        }
        r.cached_tokens += n;
        if (result.blocks_allocated > 0) {
            bus_.emit("kv.allocated", r.ctx,
                      json::Object{{"scheduler_request_id", r.id},
                                   {"blocks", result.blocks_allocated},
                                   {"tokens", result.tokens_appended}},
                      TelemetryLevel::standard);
        }
        return true;
    }

    void grant(Req& r) {
        ++r.credits;
        ++r.pending;
    }

    void apply_plan(const sch::StepPlan& plan) {
        const sch::TimeUs now = clock_.now();
        for (const sch::PreemptionEvent& ev : plan.preempted) {
            Req* r = find(ev.id);
            if (r == nullptr) {
                continue;
            }
            ++r->summary.preemptions;
            free_cache(*r, "preempted");
            bus_.emit("scheduler.preempted", r->ctx,
                      json::Object{{"scheduler_request_id", ev.id},
                                   {"reason", str(sch::to_string(ev.reason))},
                                   {"mode", str(sch::to_string(ev.mode))},
                                   {"kv_tokens", ev.kv_tokens},
                                   {"beneficiary_scheduler_request_id", ev.beneficiary},
                                   {"preemptions", r->summary.preemptions},
                                   {"failed", ev.failed}},
                      TelemetryLevel::metrics);
            if (ev.failed) {
                fail(*r, Status(ErrorCode::unavailable,
                                "request preempted more than max_requeue_count times under KV pressure"));
            }
        }
        for (const sch::RequestId id : plan.admitted) {
            Req* r = find(id);
            if (r == nullptr) {
                continue;
            }
            const double queue_ms = ms_since(r->submitted_at, now);
            if (!r->admitted_once) {
                r->admitted_once = true;
                r->summary.queue_ms = queue_ms;
            }
            r->admission_reused = 0;
            bus_.emit("scheduler.admitted", r->ctx,
                      json::Object{{"scheduler_request_id", id},
                                   {"step", plan.step_index},
                                   {"queue_ms", queue_ms},
                                   {"resumed", r->summary.preemptions > 0},
                                   {"reserved_blocks", adapter_.blocks_held(id)}},
                      TelemetryLevel::metrics);
        }
        for (const sch::ScheduledWork& w : plan.work) {
            Req* r = find(w.id);
            if (r == nullptr || r->done || r->failure) {
                continue;
            }
            kvc::AppendResult ar;
            if (w.phase == sch::Phase::Prefill) {
                if (!append(*r, w.context_offset, w.num_tokens, ar)) {
                    continue;
                }
                r->admission_reused += ar.tokens_reused;
                const bool recompute = r->summary.preemptions > 0;
                if (ar.tokens_reused > 0) {
                    bus_.emit("kv.reused", r->ctx,
                              json::Object{{"scheduler_request_id", w.id},
                                           {"tokens", ar.tokens_reused},
                                           {"blocks", ar.blocks_reused},
                                           {"recompute", recompute}},
                              TelemetryLevel::metrics);
                }
                bus_.emit("inference.prefill.chunk", r->ctx,
                          json::Object{{"scheduler_request_id", w.id},
                                       {"step", plan.step_index},
                                       {"tokens", w.num_tokens},
                                       {"context_offset", w.context_offset},
                                       {"reused_tokens", ar.tokens_reused},
                                       {"completes_prefill", w.completes_prefill},
                                       {"recompute", recompute}},
                          TelemetryLevel::standard);
                if (w.completes_prefill) {
                    if (!recompute) {
                        r->summary.reused_prompt_tokens = r->admission_reused;
                    }
                    bus_.emit("inference.prefill.completed", r->ctx,
                              json::Object{{"scheduler_request_id", w.id},
                                           {"context_tokens", r->cached_tokens},
                                           {"prompt_tokens", r->prompt_tokens},
                                           {"reused_tokens", r->admission_reused},
                                           {"recompute", recompute}},
                              TelemetryLevel::metrics);
                    grant(*r);
                }
            } else {
                if (!append(*r, w.context_offset, 1, ar)) {
                    continue;
                }
                grant(*r);
            }
        }
        if (!plan.empty() || !plan.preempted.empty()) {
            bus_.emit("scheduler.batch.formed", engine_ctx_,
                      json::Object{{"step", plan.step_index},
                                   {"sequences", plan.work.size()},
                                   {"prefill_tokens", plan.prefill_tokens},
                                   {"decode_tokens", plan.decode_tokens},
                                   {"admitted", plan.admitted.size()},
                                   {"preempted", plan.preempted.size()},
                                   {"running", scheduler_.running().size()}},
                          TelemetryLevel::standard);
        }
    }

    bool step_consumed(const sch::StepPlan& plan) {
        for (const sch::ScheduledWork& w : plan.work) {
            const Req* r = find(w.id);
            if (r == nullptr || r->done || r->failure || r->passthrough) {
                continue;
            }
            if (r->pending > 0) {
                return false;
            }
        }
        return true;
    }

    void post_step(const sch::StepPlan& plan, sch::TimeUs started) {
        std::size_t finished = 0;
        auto settle = [&](sch::RequestId id) {
            auto it = reqs_.find(id);
            if (it == reqs_.end()) {
                return;
            }
            Req& r = it->second;
            const auto state = scheduler_.state(id);
            if (state && !sch::is_terminal(*state)) {
                return;
            }
            if (state == sch::RequestState::Completed) {
                r.passthrough = true;
                ++finished;
            } else if (state == sch::RequestState::Failed && !r.failure) {
                fail(r, Status(ErrorCode::unavailable, "request failed in the scheduler"));
            }
            if (r.done) {
                free_cache(r, "completed");
                reqs_.erase(it);
            }
        };
        for (const auto& w : plan.work) settle(w.id);
        for (const auto& ev : plan.preempted) settle(ev.id);
        if (!plan.empty()) {
            bus_.emit("scheduler.batch.completed", engine_ctx_,
                      json::Object{{"step", plan.step_index},
                                   {"duration_ms", ms_since(started, clock_.now())},
                                   {"finished", finished}},
                      TelemetryLevel::standard);
        }
    }

    void loop() {
        std::unique_lock<std::mutex> lock(mu_);
        while (!stopping_) {
            if (scheduler_.idle()) {
                cv_.wait(lock, [&] { return stopping_ || !scheduler_.idle(); });
                continue;
            }
            const sch::TimeUs started = clock_.now();
            release_reason_ = "preempted";
            const sch::StepPlan plan = scheduler_.plan_step();
            release_reason_ = "completed";
            apply_plan(plan);
            if (plan.empty()) {
                scheduler_.complete_step(plan);
                post_step(plan, started);
                cv_.notify_all();
                if (!plan.preempted.empty() || !plan.admitted.empty() || !plan.failed.empty()) {
                    continue;  // state changed; plan again
                }
                // Nothing runnable (e.g. admission blocked on KV): wait for a
                // submit, a finish or produced tokens to change the picture.
                const std::uint64_t seen = generation_;
                cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                continue;
            }
            cv_.notify_all();
            cv_.wait(lock, [&] { return stopping_ || step_consumed(plan); });
            if (stopping_) {
                break;
            }
            sch::StepOutcome outcome;
            for (const auto& w : plan.work) {
                const Req* r = find(w.id);
                if (r == nullptr || r->done) {
                    outcome.finished_early.push_back(w.id);
                }
            }
            scheduler_.complete_step(plan, outcome);
            post_step(plan, started);
            cv_.notify_all();
        }
        // Wake sessions blocked in acquire_token().
        cv_.notify_all();
    }

    TelemetryBus& bus_;
    TelemetryContext engine_ctx_;
    SteadyClock clock_;
    kvc::KvCacheManager cache_;
    KvCacheCapacityAdapter adapter_;
    sch::Scheduler scheduler_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::map<sch::RequestId, Req> reqs_;
    sch::RequestId next_id_ = 1;
    std::uint64_t generation_ = 0;
    std::size_t evicted_pending_ = 0;
    std::optional<kvc::PressureLevel> pressure_pending_;
    const char* release_reason_ = "completed";  // kv.freed reason for the release hook
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace

std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions& options, TelemetryBus& bus,
                                                     TelemetryContext engine_context) {
    if (!options.enabled) {
        return nullptr;
    }
    return std::make_unique<SchedulingRuntime>(options, bus, std::move(engine_context));
}

}  // namespace sonder::inference::detail

#else  // modules not built

namespace sonder::inference::detail {

std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions&, TelemetryBus&, TelemetryContext) {
    return nullptr;
}

}  // namespace sonder::inference::detail

#endif
