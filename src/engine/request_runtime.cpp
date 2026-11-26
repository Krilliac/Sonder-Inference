#include "engine/request_runtime.hpp"

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
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

kvc::KvCacheConfig make_cache_config(const SchedulingOptions& o, const CheckpointStoreOptions& cp) {
    kvc::KvCacheConfig c;
    c.block_size_tokens = at_least_one(o.kv_block_size_tokens);
    c.num_blocks = at_least_one(o.kv_num_blocks);
    c.enable_prefix_caching = o.prefix_caching;
    c.max_checkpoint_bytes = cp.max_bytes;
    c.max_checkpoints = cp.max_checkpoints;
    return c;
}

// Bounded, leaf-mutex-protected store for serialized recurrent-state blobs.
// Entries are keyed 1:1 with KvCacheManager checkpoints via CheckpointHandle
// and evicted synchronously whenever KvCacheManager emits checkpoint_evicted.
class CheckpointBlobStore {
public:
    explicit CheckpointBlobStore(std::uint64_t max_bytes) noexcept : max_bytes_(max_bytes) {}

    [[nodiscard]] std::uint64_t max_bytes() const noexcept { return max_bytes_; }

    bool put(const CheckpointHandle& handle, std::shared_ptr<const std::vector<std::uint8_t>> blob) {
        if (!blob || blob->empty()) {
            return false;
        }
        const auto size = static_cast<std::uint64_t>(blob->size());
        std::lock_guard<std::mutex> lock(mutex_);
        if (size > max_bytes_) {
            return false;
        }
        auto existing = blobs_.find(handle);
        std::uint64_t existing_bytes = 0;
        if (existing != blobs_.end() && existing->second) {
            existing_bytes = static_cast<std::uint64_t>(existing->second->size());
        }
        const std::uint64_t base_bytes = total_bytes_ >= existing_bytes ? (total_bytes_ - existing_bytes) : 0;
        if (size > max_bytes_ - base_bytes) {
            return false;
        }
        if (existing != blobs_.end()) {
            existing->second = std::move(blob);
        } else {
            blobs_.emplace(handle, std::move(blob));
        }
        total_bytes_ = base_bytes + size;
        return true;
    }

    [[nodiscard]] std::shared_ptr<const std::vector<std::uint8_t>> get(const CheckpointHandle& handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = blobs_.find(handle);
        return it == blobs_.end() ? nullptr : it->second;
    }

    bool erase(const CheckpointHandle& handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = blobs_.find(handle);
        if (it == blobs_.end()) {
            return false;
        }
        const std::uint64_t sz = it->second ? static_cast<std::uint64_t>(it->second->size()) : 0;
        total_bytes_ = total_bytes_ >= sz ? (total_bytes_ - sz) : 0;
        blobs_.erase(it);
        ++evictions_;
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        blobs_.clear();
        total_bytes_ = 0;
    }

    [[nodiscard]] CheckpointStoreUsage usage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CheckpointStoreUsage u;
        u.entries = blobs_.size();
        u.total_bytes = total_bytes_;
        u.max_bytes = max_bytes_;
        u.evictions = evictions_;
        return u;
    }

private:
    std::uint64_t max_bytes_ = 0;
    mutable std::mutex mutex_;
    std::map<CheckpointHandle, std::shared_ptr<const std::vector<std::uint8_t>>> blobs_;
    std::uint64_t total_bytes_ = 0;
    std::uint64_t evictions_ = 0;
};

sch::SchedulerConfig make_scheduler_config(const SchedulingOptions& o) {
    sch::SchedulerConfig c;
    c.strict_priority_admission = o.strict_priority_admission;
    c.max_running_sequences = at_least_one(o.max_running_sequences);
    c.max_step_sequences = at_least_one(o.max_step_sequences);
    c.max_step_tokens = at_least_one(o.max_step_tokens);
    c.prefill_chunk_tokens = at_least_one(o.prefill_chunk_tokens);
    c.max_step_prefill_tokens = std::max(c.prefill_chunk_tokens, c.max_step_tokens / 2);
    c.admission_watermark_blocks = o.admission_watermark_blocks;
    c.max_requeue_count = o.max_requeue_count;
    c.anti_starvation_preemption_cap = o.anti_starvation_preemption_cap;
    c.enable_priority_preemption = o.enable_priority_preemption;
    c.enable_swap = o.enable_spill_preemption;
    c.swap_min_tokens = o.spill_min_tokens;
    c.preemption_cost.kv_bytes_per_token = o.kv_bytes_per_token;
    c.preemption_cost.prefill_tokens_per_sec = o.prefill_tokens_per_sec;
    c.preemption_cost.spill_bandwidth_bytes_per_sec = o.spill_bandwidth_bytes_per_sec;
    c.preemption_cost.restore_bandwidth_bytes_per_sec = o.restore_bandwidth_bytes_per_sec;
    if (o.strict_priority_admission) {
        c.admission_lookahead = 0;
        c.aging_interval_us = std::numeric_limits<sch::TimeUs>::max();
        c.starvation_threshold_us = std::numeric_limits<sch::TimeUs>::max();
        c.enable_priority_preemption = false;
    }
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
    ModelArchitecture architecture = ModelArchitecture::attention_only;
    kvc::Priority cache_priority = 0;
    sch::TimeUs submitted_at = 0;
    std::uint32_t credits = 0;  // grants not yet acquired by the session
    std::uint32_t pending = 0;  // grants not yet produced
    bool done = false;          // session finished; no more tokens will come
    bool passthrough = false;   // scheduler completed it; later tokens are not gated
    // Missed a step deadline with a grant outstanding (pending > 0). Left out
    // of the step barrier, and not granted again, until it produces that token.
    bool lagging = false;
    std::optional<Status> failure;
    bool in_cache = false;
    std::size_t cached_tokens = 0;
    std::uint64_t admission_reused = 0;
    bool admitted_once = false;
    std::shared_ptr<RequestAdmission> admission;
    // false: one admission grant at the end of the logical prefill, then the
    // session streams without grants (SchedulerMode::account).
    bool gated = true;
    bool retain_for_checkpoint = false;
    std::optional<RestoredCheckpoint> restored_checkpoint;
    bool preempted_pending = false;
    sch::PreemptionMode last_preemption_mode = sch::PreemptionMode::Recompute;
    const char* last_preemption_reason = nullptr;
    RuntimeRequestSummary summary;
};

class SchedulingRuntime final : public RequestRuntime {
public:
    SchedulingRuntime(const SchedulingOptions& options, TelemetryBus& bus, TelemetryContext engine_context,
                      CheckpointStoreOptions checkpoint_options)
        : bus_(bus),
          engine_ctx_(std::move(engine_context)),
          blob_store_(checkpoint_options.max_bytes),
          cache_(make_cache_config(options, checkpoint_options)),
          adapter_(cache_),
          scheduler_(make_scheduler_config(options), clock_, adapter_),
          stall_timeout_(std::max<std::uint32_t>(1, options.step_stall_timeout_ms)),
          strict_priority_(options.strict_priority_admission) {
        cache_.set_event_listener([this](const kvc::CacheEvent& e) { on_cache_event(e); });
        // Free a request's sequence the moment the scheduler drops its
        // reservation so preemption sees the blocks come back (recompute).
        // Checkpoint-enabled requests keep their sequence attached on normal
        // scheduler completion until Session registers any captured checkpoint
        // and calls finish().
        adapter_.set_release_hook([this](sch::RequestId id) {
            if (Req* r = find(id)) {
                if (r->retain_for_checkpoint && !r->done && release_reason_ != nullptr &&
                    std::string_view(release_reason_) == "completed") {
                    return;
                }
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
        cache_.set_event_listener({});
        blob_store_.clear();
    }

    Result<RuntimeSubmission> submit(RuntimeRequestSpec spec) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (spec.prompt_tokens.empty()) {
            return Status(ErrorCode::invalid_argument, "request has no prompt tokens");
        }
        const std::uint64_t block = cache_.config().block_size_tokens;
        const std::uint64_t capacity = block * cache_.config().num_blocks;
        const std::uint64_t prompt = spec.prompt_tokens.size();
        const sch::RequestId id = next_id_++;

        if (prompt + 1 > capacity && !spec.gated) {
            // Ungated (remote-process) requests keep their KV upstream: a
            // prompt beyond the logical pool runs unscheduled instead of
            // being refused for a limit the upstream does not have.
            bus_.emit("scheduler.bypassed", spec.context,
                      json::Object{{"reason", "exceeds_kv_pool"},
                                   {"prompt_tokens", prompt},
                                   {"kv_capacity_tokens", capacity}},
                      TelemetryLevel::metrics);
            RuntimeSubmission bypass;
            bypass.bypassed = true;
            bypass.max_new_tokens = std::max<std::uint32_t>(1, spec.max_new_tokens);
            if (spec.context_limit > 0) {
                bypass.max_new_tokens = static_cast<std::uint32_t>(std::max<std::uint64_t>(
                    1, std::min<std::uint64_t>(bypass.max_new_tokens,
                                               spec.context_limit > prompt ? spec.context_limit - prompt : 1)));
            }
            return bypass;
        }
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
        if (spec.exact_tokens && spec.context_limit > 0 && prompt >= spec.context_limit) {
            bus_.emit("scheduler.rejected", spec.context,
                      json::Object{{"scheduler_request_id", id},
                                   {"reason", "invalid_request"},
                                   {"prompt_tokens", prompt}},
                      TelemetryLevel::metrics);
            return Status(ErrorCode::invalid_argument, "prompt leaves no room for generation in the context window");
        }
        std::uint64_t max_new = std::max<std::uint32_t>(1, spec.max_new_tokens);
        if (spec.context_limit > 0) {
            max_new = std::min<std::uint64_t>(max_new, spec.context_limit > prompt ? spec.context_limit - prompt : 1);
        }
        if (spec.gated) {
            // Ungated output is not accounted in the pool (it leaves the
            // scheduler after its prefill), so only gated requests are capped.
            max_new = std::max<std::uint64_t>(1, std::min<std::uint64_t>(max_new, capacity - prompt));
        }

        int rank = static_cast<int>(spec.workload) - spec.priority;
        // Numeric HTTP headers retain the legacy scheduler rank, including
        // explicit zero. Their host ticket still enforces interactive FIFO.
        if (strict_priority_ && !spec.preserve_numeric_priority) {
            // Hosted requests have three explicit classes. Missing metadata
            // is deliberately interactive, matching the server default.
            rank = spec.priority_class == RequestPriority::background ? 2
                  : spec.priority_class == RequestPriority::subagent ? 1 : 0;
        }
        sch::RequestSpec rs;
        rs.id = id;
        rs.workload = static_cast<sch::WorkloadClass>(static_cast<int>(spec.workload));
        if (strict_priority_ || spec.priority != 0) {
            rs.priority_override = rank;
        }
        rs.task_id = spec.context.task_id.value_or("");
        rs.prompt_tokens = static_cast<sch::TokenCount>(prompt);
        // An ungated request completes in the scheduler with its first token,
        // i.e. at the end of its logical prefill (admission + prompt KV).
        rs.max_new_tokens = static_cast<sch::TokenCount>(spec.gated ? max_new : 1);
        rs.cancellable = true;
        if (spec.admission) {
            rs.try_admit = [admission = spec.admission] { return admission->try_acquire(); };
            rs.admission_ready = [admission = spec.admission] { return admission->ready(); };
            rs.admission_order = spec.admission->order();
        }
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
        r.architecture = spec.architecture;
        r.cache_priority = static_cast<kvc::Priority>(std::clamp(6 - rank, 0, 255));
        r.submitted_at = clock_.now();
        r.gated = spec.gated;
        r.retain_for_checkpoint = spec.retain_for_checkpoint;
        r.admission = std::move(spec.admission);
        bus_.emit("scheduler.enqueued", r.ctx,
                  json::Object{{"scheduler_request_id", id},
                               {"workload", str(sch::to_string(rs.workload))},
                               {"priority_rank", rank},
                               {"prompt_tokens", prompt},
                               {"exact_prompt_tokens", spec.exact_tokens},
                               {"max_new_tokens", max_new},
                               {"context_limit", spec.context_limit},
                               {"gated", spec.gated}},
                  TelemetryLevel::metrics);
        reqs_.emplace(id, std::move(r));
        ++generation_;
        cv_.notify_all();
        RuntimeSubmission submission;
        submission.id = static_cast<std::uint64_t>(id);
        submission.max_new_tokens = static_cast<std::uint32_t>(max_new);
        return submission;
    }

    Status acquire_token(std::uint64_t id, const CancellationToken& cancel,
                         std::optional<std::chrono::steady_clock::time_point> deadline) override {
        return wait_for_grant(id, cancel, deadline, true);
    }

    GatePollResult try_acquire_token_nonblocking(std::uint64_t id, const CancellationToken& cancel) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = reqs_.find(id);
        if (it == reqs_.end()) {
            return GatePollResult{GatePollStatus::failed, Status(ErrorCode::internal, "unknown scheduled request"),
                                  "unknown_request"};
        }
        Req& r = it->second;
        if (r.failure) {
            return GatePollResult{GatePollStatus::failed, *r.failure, "scheduler_failed"};
        }
        if (cancel.cancelled()) {
            return GatePollResult{GatePollStatus::cancelled,
                                  Status(ErrorCode::cancelled, "cancelled during non-blocking gate poll"),
                                  "cancelled"};
        }
        if (r.preempted_pending) {
            const GatePollStatus st = r.last_preemption_mode == sch::PreemptionMode::SpillToCheckpoint
                                          ? GatePollStatus::yield_spill
                                          : GatePollStatus::yield_recompute;
            return GatePollResult{st, Status::success(),
                                  r.last_preemption_reason ? r.last_preemption_reason : "preempted"};
        }
        if (r.passthrough) {
            return GatePollResult{GatePollStatus::granted, Status::success(), nullptr};
        }
        if (r.credits > 0) {
            --r.credits;
            return GatePollResult{GatePollStatus::granted, Status::success(), nullptr};
        }
        if (stopping_) {
            return GatePollResult{GatePollStatus::failed, Status(ErrorCode::unavailable, "engine is shutting down"),
                                  "stopping"};
        }
        // No credit immediately available; do NOT block shared worker threads.
        return GatePollResult{GatePollStatus::yield_recompute, Status::success(), "gate_wait"};
    }

    SuspendDecision poll_suspend_decision(std::uint64_t id, const CancellationToken& cancel) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (cancel.cancelled() || stopping_) {
            return SuspendDecision{};
        }
        auto it = reqs_.find(id);
        if (it == reqs_.end()) {
            return SuspendDecision{};
        }
        const Req& r = it->second;
        if (!r.preempted_pending) {
            return SuspendDecision{};
        }
        SuspendDecision d;
        d.action = r.last_preemption_mode == sch::PreemptionMode::SpillToCheckpoint
                       ? SuspendAction::yield_spill
                       : SuspendAction::yield_recompute;
        d.reason = r.last_preemption_reason ? r.last_preemption_reason : "preempted";
        return d;
    }

    Status admit_backend(std::uint64_t id, const CancellationToken& cancel,
                         std::optional<std::chrono::steady_clock::time_point> deadline) override {
        return wait_for_grant(id, cancel, deadline, false);
    }

    // The first backend admission and later token gates share the same wait.
    // Only the initial admission has a deadline argument: running deadlines
    // cancel through the session's token, preserving partial-result semantics.
    Status wait_for_grant(std::uint64_t id, const CancellationToken& cancel,
                          std::optional<std::chrono::steady_clock::time_point> deadline, bool consume) {
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            auto it = reqs_.find(id);
            if (it == reqs_.end()) return Status(ErrorCode::internal, "unknown scheduled request");
            Req& r = it->second;
            if (r.failure) return *r.failure;
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled while waiting for the scheduler");
            if (deadline && std::chrono::steady_clock::now() >= *deadline) {
                return Status(ErrorCode::timeout, "request deadline exceeded while waiting for admission");
            }
            if (r.passthrough) return Status::success();
            if (r.credits > 0) {
                if (consume) --r.credits;
                return Status::success();
            }
            if (stopping_) return Status(ErrorCode::unavailable, "engine is shutting down");
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
            if (r.pending == 0) {
                r.lagging = false;
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
            r.restored_checkpoint.reset();
            const auto state = scheduler_.state(id);
            bool terminal = !state || sch::is_terminal(*state);
            if (!terminal) {
                const bool running = *state == sch::RequestState::Prefill || *state == sch::RequestState::Decode;
                if (outcome != RequestOutcome::completed || !running) {
                    // Releases the KV reservation (and the sequence) immediately.
                    ScopedReleaseReason guard(release_reason_, to_string(outcome));
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
                (void)scheduler_.forget(id);
            }
            ++generation_;
        }
        cv_.notify_all();
        return summary;
    }

    Status register_checkpoint(std::uint64_t id, std::uint64_t token_position,
                               std::vector<std::uint8_t> bytes) override {
        if (token_position == 0) {
            return Status(ErrorCode::invalid_argument, "checkpoint token_position must be > 0");
        }
        if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
            if (token_position > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                return Status(ErrorCode::invalid_argument, "checkpoint token_position exceeds addressable range");
            }
        }
        if (bytes.empty()) {
            return Status(ErrorCode::invalid_argument, "checkpoint state is empty");
        }
        if (bytes.size() > kMaxRecurrentCheckpointBytes) {
            return Status(ErrorCode::invalid_argument, "checkpoint state exceeds kMaxRecurrentCheckpointBytes");
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (static_cast<std::uint64_t>(bytes.size()) > blob_store_.max_bytes()) {
            return Status(ErrorCode::invalid_argument, "checkpoint state exceeds checkpoint store budget");
        }
        Req* r = find(id);
        if (r == nullptr || !r->in_cache) {
            return Status(ErrorCode::not_found, "active cache sequence not found for checkpoint registration");
        }
        if (r->architecture == ModelArchitecture::attention_only) {
            return Status(ErrorCode::invalid_argument,
                          "recurrent checkpoints require hybrid or recurrent architecture");
        }
        const auto pos = static_cast<std::size_t>(token_position);
        const auto byte_size = static_cast<std::uint64_t>(bytes.size());
        Status st = cache_.save_checkpoint(id, pos, byte_size);
        if (!st.ok()) {
            flush_cache_events(r->ctx, id);
            return st;
        }
        const CheckpointHandle handle{id, token_position};
        auto blob = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
        if (!blob_store_.put(handle, std::move(blob))) {
            // Roll back cache bookkeeping without emitting a spurious
            // kv.checkpoint.evicted event for a blob that was never stored.
            const std::size_t prev_evicted = checkpoint_evicted_pending_;
            const std::uint64_t prev_bytes = checkpoint_evicted_bytes_pending_;
            (void)cache_.remove_checkpoint(id, pos);
            checkpoint_evicted_pending_ = prev_evicted;
            checkpoint_evicted_bytes_pending_ = prev_bytes;
            flush_cache_events(r->ctx, id);
            return Status(ErrorCode::unavailable, "checkpoint blob store budget exceeded");
        }
        flush_cache_events(r->ctx, id);
        return Status::success();
    }

    std::optional<RestoredCheckpoint> lookup_restored_checkpoint(std::uint64_t id) const override {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = reqs_.find(id);
        if (it == reqs_.end()) {
            return std::nullopt;
        }
        return it->second.restored_checkpoint;
    }

    void release_checkpoint_retention(std::uint64_t id) override {
        std::lock_guard<std::mutex> lock(mu_);
        Req* r = find(id);
        if (r == nullptr || !r->retain_for_checkpoint) {
            return;
        }
        r->retain_for_checkpoint = false;
        const auto state = scheduler_.state(id);
        const bool terminal = !state || sch::is_terminal(*state);
        if (terminal || !r->gated) {
            free_cache(*r, "completed");
        }
    }

    CheckpointStoreUsage checkpoint_store_usage() const override { return blob_store_.usage(); }

    Result<LogicalBranchInfo> ensure_logical_sequence(
        std::uint64_t existing_seq_id, std::span<const TokenId> tokens, std::uint64_t fingerprint,
        ModelArchitecture architecture, std::int32_t priority,
        std::span<const std::uint8_t> recurrent_checkpoint_blob) override {
        std::lock_guard<std::mutex> lock(mu_);
        kvc::SequenceId seq_id = existing_seq_id;
        if (seq_id != 0 && cache_.has_sequence(seq_id)) {
            if (cache_.num_tokens(seq_id) == tokens.size()) {
                return branch_info_locked(seq_id, 0);
            }
            (void)cache_.free_sequence(seq_id);
        }
        seq_id = next_branch_seq_id_++;
        kvc::SequenceOptions so;
        so.fingerprint = kvc::CacheFingerprint{fingerprint};
        so.priority = static_cast<kvc::Priority>(std::clamp(priority, 0, 255));
        so.architecture = architecture;
        if (Status st = cache_.add_sequence(seq_id, so); !st.ok()) {
            return st;
        }
        kvc::AppendResult ar;
        if (!tokens.empty()) {
            if (Status st = cache_.append_tokens(seq_id, tokens, &ar); !st.ok()) {
                (void)cache_.free_sequence(seq_id);
                return st;
            }
            if (architecture != ModelArchitecture::attention_only && !recurrent_checkpoint_blob.empty()) {
                (void)cache_.save_checkpoint(seq_id, tokens.size(),
                                             static_cast<std::uint64_t>(recurrent_checkpoint_blob.size()));
            }
        }
        return branch_info_locked(seq_id, ar.copy_on_write);
    }

    Result<LogicalBranchInfo> fork_logical_sequence(
        std::uint64_t parent_seq_id, std::optional<std::size_t> rewind_to_tokens,
        std::optional<std::int32_t> child_priority) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (parent_seq_id == 0 || !cache_.has_sequence(parent_seq_id)) {
            return Status(ErrorCode::not_found, "parent logical KV sequence not found");
        }
        const kvc::SequenceId child_id = next_branch_seq_id_++;
        std::optional<kvc::Priority> prio;
        if (child_priority) {
            prio = static_cast<kvc::Priority>(std::clamp(*child_priority, 0, 255));
        }
        if (Status st = cache_.fork_sequence(parent_seq_id, child_id, prio); !st.ok()) {
            return st;
        }
        if (rewind_to_tokens.has_value()) {
            if (Status st = cache_.truncate(child_id, *rewind_to_tokens); !st.ok()) {
                (void)cache_.free_sequence(child_id);
                return st;
            }
        }
        return branch_info_locked(child_id, 0);
    }

    Result<LogicalBranchInfo> truncate_logical_sequence(std::uint64_t seq_id, std::size_t new_length) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (seq_id == 0 || !cache_.has_sequence(seq_id)) {
            return Status(ErrorCode::not_found, "logical KV sequence not found");
        }
        if (Status st = cache_.truncate(seq_id, new_length); !st.ok()) {
            return st;
        }
        return branch_info_locked(seq_id, 0);
    }

    Result<LogicalBranchInfo> append_logical_sequence(std::uint64_t seq_id,
                                                      std::span<const TokenId> suffix_tokens) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (seq_id == 0 || !cache_.has_sequence(seq_id)) {
            return Status(ErrorCode::not_found, "logical KV sequence not found");
        }
        kvc::AppendResult ar;
        if (!suffix_tokens.empty()) {
            if (Status st = cache_.append_tokens(seq_id, suffix_tokens, &ar); !st.ok()) {
                return st;
            }
        }
        return branch_info_locked(seq_id, ar.copy_on_write);
    }

    void release_logical_sequence(std::uint64_t seq_id) override {
        if (seq_id == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (cache_.has_sequence(seq_id)) {
            (void)cache_.free_sequence(seq_id);
        }
    }

    std::optional<LogicalBranchInfo> inspect_logical_sequence(std::uint64_t seq_id) const override {
        if (seq_id == 0) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (!cache_.has_sequence(seq_id)) {
            return std::nullopt;
        }
        return branch_info_locked(seq_id, 0);
    }

    std::size_t tracked_requests() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return std::max(reqs_.size(), scheduler_.tracked());
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
    class ScopedReleaseReason {
    public:
        ScopedReleaseReason(const char*& slot, const char* next) noexcept : slot_(slot), prev_(slot) { slot_ = next; }
        ~ScopedReleaseReason() noexcept { slot_ = prev_; }
        ScopedReleaseReason(const ScopedReleaseReason&) = delete;
        ScopedReleaseReason& operator=(const ScopedReleaseReason&) = delete;

    private:
        const char*& slot_;
        const char* prev_;
    };

    Req* find(sch::RequestId id) {
        auto it = reqs_.find(id);
        return it == reqs_.end() ? nullptr : &it->second;
    }

    void on_cache_event(const kvc::CacheEvent& e) {
        if (e.kind == kvc::CacheEvent::Kind::evicted) {
            ++evicted_pending_;
        } else if (e.kind == kvc::CacheEvent::Kind::pressure_changed) {
            pressure_pending_ = e.pressure;
        } else if (e.kind == kvc::CacheEvent::Kind::checkpoint_evicted && e.checkpoint) {
            const CheckpointHandle handle{static_cast<std::uint64_t>(e.checkpoint->owner),
                                          static_cast<std::uint64_t>(e.checkpoint->position)};
            blob_store_.erase(handle);
            ++checkpoint_evicted_pending_;
            checkpoint_evicted_bytes_pending_ += e.checkpoint->state_size;
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
        if (checkpoint_evicted_pending_ > 0) {
            bus_.emit("kv.checkpoint.evicted", ctx,
                      json::Object{{"checkpoints", checkpoint_evicted_pending_},
                                   {"bytes", checkpoint_evicted_bytes_pending_},
                                   {"trigger_scheduler_request_id", trigger},
                                   {"total_checkpoint_evictions", cache_.stats().checkpoint_evictions}},
                      TelemetryLevel::metrics);
            checkpoint_evicted_pending_ = 0;
            checkpoint_evicted_bytes_pending_ = 0;
        }
        if (pressure_pending_) {
            const auto s = cache_.stats();
            bus_.emit("kv.pressure", engine_ctx_,
                      json::Object{{"level", kvc::to_string(*pressure_pending_)},
                                   {"occupancy", s.utilization},
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
            so.architecture = r.architecture;
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
            {
                ScopedReleaseReason guard(release_reason_, "failed");
                (void)scheduler_.cancel(r.id);
            }
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

    // At most one grant is outstanding per request: a request still owing a
    // token from an earlier step (lagging) is not granted another.
    void grant(Req& r) {
        if (r.pending > 0) {
            r.lagging = true;
            return;
        }
        ++r.credits;
        ++r.pending;
    }

    // Returns the ids of planned requests that were lagging when the plan was
    // applied: their work is not executed (no KV append, no grant) until they
    // produce the token they already owe.
    std::vector<sch::RequestId> apply_plan(const sch::StepPlan& plan) {
        std::vector<sch::RequestId> lagging;
        const sch::TimeUs now = clock_.now();
        for (const sch::PreemptionEvent& ev : plan.preempted) {
            Req* r = find(ev.id);
            if (r == nullptr) {
                continue;
            }
            ++r->summary.preemptions;
            r->preempted_pending = true;
            r->last_preemption_mode = ev.mode;
            r->last_preemption_reason =
                ev.reason == sch::PreemptionReason::KvPressure ? "kv_pressure" : "priority_admission";
            r->restored_checkpoint.reset();
            free_cache(*r, "preempted");
            bus_.emit("scheduler.preempted", r->ctx,
                      json::Object{{"scheduler_request_id", ev.id},
                                   {"reason", str(sch::to_string(ev.reason))},
                                   {"mode", str(sch::to_string(ev.mode))},
                                   {"kv_tokens", ev.kv_tokens},
                                   {"recompute_cost_us", ev.recompute_cost_us},
                                   {"spill_restore_cost_us", ev.spill_restore_cost_us},
                                   {"estimated_spill_bytes", ev.estimated_spill_bytes},
                                   {"beneficiary_scheduler_request_id", ev.beneficiary},
                                   {"preemptions", r->summary.preemptions},
                                   {"queue_ms", r->summary.queue_ms},
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
            r->preempted_pending = false;
            r->last_preemption_reason = nullptr;
            const double queue_ms = ms_since(r->submitted_at, now);
            if (!r->admitted_once) {
                r->admitted_once = true;
                r->summary.queue_ms = r->admission ? r->admission->queue_ms() : queue_ms;
            }
            r->admission_reused = 0;
            r->restored_checkpoint.reset();
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
            if (r->lagging) {
                lagging.push_back(w.id);
                continue;
            }
            kvc::AppendResult ar;
            if (w.phase == sch::Phase::Prefill) {
                if (!append(*r, w.context_offset, w.num_tokens, ar)) {
                    continue;
                }
                if (ar.checkpoint) {
                    const CheckpointHandle handle{static_cast<std::uint64_t>(ar.checkpoint->owner),
                                                  static_cast<std::uint64_t>(ar.checkpoint->position)};
                    if (auto blob = blob_store_.get(handle)) {
                        r->restored_checkpoint = RestoredCheckpoint{handle, std::move(blob)};
                    }
                }
                r->admission_reused += ar.tokens_reused;
                const bool recompute = r->summary.preemptions > 0;
                if (ar.tokens_reused > 0) {
                    bus_.emit("kv.reused", r->ctx,
                              json::Object{{"scheduler_request_id", w.id},
                                           {"tokens", ar.tokens_reused},
                                           {"avoided_prefill_tokens", ar.tokens_reused},
                                           {"blocks", ar.blocks_reused},
                                           {"recompute", recompute}},
                              TelemetryLevel::metrics);
                }
                bus_.emit("scheduler.prefill.chunk", r->ctx,
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
                    bus_.emit("scheduler.prefill.completed", r->ctx,
                              json::Object{{"scheduler_request_id", w.id},
                                           {"context_tokens", r->cached_tokens},
                                           {"prompt_tokens", r->prompt_tokens},
                                           {"reused_tokens", r->admission_reused},
                                           {"recompute", recompute}},
                              TelemetryLevel::metrics);
                    if (r->gated) {
                        grant(*r);
                    } else {
                        // Admission grant only: nothing is owed back, so the
                        // step barrier never waits on an ungated request.
                        ++r->credits;
                    }
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
                                   {"batch_size", plan.work.size()},
                                   {"sequences", plan.work.size()},
                                   {"prefill_tokens", plan.prefill_tokens},
                                   {"decode_tokens", plan.decode_tokens},
                                   {"admitted", plan.admitted.size()},
                                   {"preempted", plan.preempted.size()},
                                   {"running", scheduler_.running().size()}},
                          TelemetryLevel::standard);
        }
        return lagging;
    }

    bool step_consumed(const sch::StepPlan& plan) {
        for (const sch::ScheduledWork& w : plan.work) {
            const Req* r = find(w.id);
            if (r == nullptr || r->done || r->failure || r->passthrough || r->lagging) {
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
                (void)scheduler_.forget(id);
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

    void wait_for_progress(std::unique_lock<std::mutex>& lock) {
        const std::uint64_t seen = generation_;
        const auto changed = [&] { return stopping_ || generation_ != seen; };
        const bool host_waiting = std::any_of(reqs_.begin(), reqs_.end(), [](const auto& entry) {
            const Req& r = entry.second;
            return r.admission && !r.admitted_once && !r.done && !r.failure;
        });
        if (host_waiting) {
            // Host permits outlive finish(): releasing one signals the host's
            // condition variable, not this runtime's generation counter. Also
            // recheck when a queued host ticket is removed without an engine
            // event. Keep the legacy event-only wait for unticketed requests.
            cv_.wait_for(lock, std::chrono::milliseconds(20), changed);
        } else {
            cv_.wait(lock, changed);
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
            const sch::StepPlan plan = [&] {
                ScopedReleaseReason guard(release_reason_, "preempted");
                return scheduler_.plan_step();
            }();
            const std::vector<sch::RequestId> lagging = apply_plan(plan);
            if (plan.empty()) {
                {
                    ScopedReleaseReason guard(release_reason_, "completed");
                    scheduler_.complete_step(plan);
                }
                post_step(plan, started);
                cv_.notify_all();
                if (!plan.preempted.empty() || !plan.admitted.empty() || !plan.failed.empty()) {
                    continue;  // state changed; plan again
                }
                // Nothing runnable: either engine progress or a host permit
                // becoming available can make the next plan runnable.
                wait_for_progress(lock);
                continue;
            }
            cv_.notify_all();
            // Bounded barrier: wait for this step's grants, but never longer
            // than the stall timeout. A step whose planned requests are all
            // still catching up on earlier grants has nothing to wait for but
            // their progress (or a submit/finish), so it sleeps until then
            // instead of spinning.
            const bool anything_runnable = std::any_of(plan.work.begin(), plan.work.end(), [&](const auto& w) {
                const Req* r = find(w.id);
                return r == nullptr || !r->lagging;
            });
            if (anything_runnable) {
                const auto deadline = std::chrono::steady_clock::now() + stall_timeout_;
                cv_.wait_until(lock, deadline, [&] { return stopping_ || step_consumed(plan); });
            } else {
                wait_for_progress(lock);
            }
            if (stopping_) {
                break;
            }
            sch::StepOutcome outcome;
            for (const auto& w : plan.work) {
                Req* r = find(w.id);
                if (r == nullptr || r->done) {
                    outcome.finished_early.push_back(w.id);
                    continue;
                }
                if (r->failure || r->passthrough) {
                    continue;
                }
                const bool was_lagging = std::find(lagging.begin(), lagging.end(), w.id) != lagging.end();
                if (r->pending > 0) {
                    // Missed the deadline (or still owes an earlier token):
                    // the work did not happen this step.
                    outcome.stalled.push_back(w.id);
                    if (!r->lagging) {
                        r->lagging = true;
                        bus_.emit("scheduler.stalled", r->ctx,
                                  json::Object{{"scheduler_request_id", w.id},
                                               {"step", plan.step_index},
                                               {"phase", w.phase == sch::Phase::Prefill ? "prefill" : "decode"},
                                               {"timeout_ms", static_cast<std::uint64_t>(stall_timeout_.count())}},
                                  TelemetryLevel::metrics);
                    }
                } else if (was_lagging) {
                    // Caught up during this step: the token it produced is
                    // this step's work. Account its KV now (append skips
                    // anything already accounted).
                    kvc::AppendResult ar;
                    (void)append(*r, w.context_offset, w.phase == sch::Phase::Prefill ? w.num_tokens : 1, ar);
                }
            }
            {
                ScopedReleaseReason guard(release_reason_, "completed");
                scheduler_.complete_step(plan, outcome);
            }
            post_step(plan, started);
            cv_.notify_all();
        }
        // Wake sessions blocked in acquire_token().
        cv_.notify_all();
    }

    LogicalBranchInfo branch_info_locked(kvc::SequenceId seq_id, std::size_t cow_count) const {
        LogicalBranchInfo info;
        info.sequence_id = static_cast<std::uint64_t>(seq_id);
        info.num_tokens = cache_.num_tokens(seq_id);
        const auto bt = cache_.block_table(seq_id);
        info.block_count = bt.size();
        for (const kvc::BlockId b : bt) {
            if (cache_.ref_count(b) > 1) {
                ++info.shared_block_count;
            }
        }
        info.copy_on_write_count = cow_count;
        return info;
    }

    TelemetryBus& bus_;
    TelemetryContext engine_ctx_;
    SteadyClock clock_;
    CheckpointBlobStore blob_store_;
    kvc::KvCacheManager cache_;
    KvCacheCapacityAdapter adapter_;
    sch::Scheduler scheduler_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::map<sch::RequestId, Req> reqs_;
    sch::RequestId next_id_ = 1;
    kvc::SequenceId next_branch_seq_id_ = (std::uint64_t{1} << 62) + 1;
    std::uint64_t generation_ = 0;
    std::size_t evicted_pending_ = 0;
    std::size_t checkpoint_evicted_pending_ = 0;
    std::uint64_t checkpoint_evicted_bytes_pending_ = 0;
    std::optional<kvc::PressureLevel> pressure_pending_;
    const char* release_reason_ = "completed";  // kv.freed reason for the release hook
    std::chrono::milliseconds stall_timeout_;
    bool strict_priority_ = false;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace

std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions& options, TelemetryBus& bus,
                                                     TelemetryContext engine_context,
                                                     CheckpointStoreOptions checkpoint_options) {
    if (!options.enabled) {
        return nullptr;
    }
    return std::make_unique<SchedulingRuntime>(options, bus, std::move(engine_context), checkpoint_options);
}

}  // namespace sonder::inference::detail

#else  // modules not built

namespace sonder::inference::detail {

std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions&, TelemetryBus&, TelemetryContext,
                                                     CheckpointStoreOptions) {
    return nullptr;
}

}  // namespace sonder::inference::detail

#endif
