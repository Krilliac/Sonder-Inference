// Internal: routes session requests through the scheduler (src/scheduler) and
// the logical KV cache (src/cache). Only active when both modules are built.
//
// Model: sessions run their backend call on their own thread; the runtime's
// coordinator thread plans iteration-level steps with the scheduler and gates
// every generated token on a scheduler grant (the chunk that completes a
// prefill, or one decode slot). A step waits for its grants for at most
// SchedulingOptions::step_stall_timeout_ms: a request that has not produced
// its token by then (a backend still thinking or loading, or a session
// blocked writing to a slow client) is left out of the barrier until it
// catches up, so it can never freeze the other requests. KV blocks are appended to the cache as the
// scheduler plans prefill/decode work, and freed when the request ends or is
// preempted (recompute). See docs/integration/engine-wiring.md.
#pragma once

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/session.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::detail {

struct RuntimeRequestSpec {
    TelemetryContext context;  // request-scoped (request_id set)
    WorkloadClass workload = WorkloadClass::implementation_worker;
    int priority = 0;
    std::optional<RequestPriority> priority_class;
    std::shared_ptr<RequestAdmission> admission;
    std::vector<TokenId> prompt_tokens;  // non-empty
    bool exact_tokens = false;
    std::uint32_t max_new_tokens = 1;
    std::uint32_t context_limit = 0;  // 0 = unknown
    std::uint64_t fingerprint = 0;    // model compatibility (prefix sharing)
    ModelArchitecture architecture = ModelArchitecture::attention_only;
    // false (SchedulerMode::account): the request is admitted and its prompt
    // accounted, then leaves the scheduler at the end of its logical prefill;
    // the session acquires one admission grant and streams without per-chunk
    // grants. A prompt larger than the KV pool is bypassed, not rejected.
    bool gated = true;
    bool preserve_numeric_priority = false;
    // Keep the cache sequence live across scheduler completion until Session
    // registers any captured recurrent-state checkpoint and calls finish().
    // Preemption and cancellation still release blocks immediately.
    bool retain_for_checkpoint = false;
};

// Opaque handle identifying a registered recurrent-state checkpoint blob in the
// runtime store (maps 1:1 onto KvCacheManager's saved checkpoint).
struct CheckpointHandle {
    std::uint64_t owner_request_id = 0;
    std::uint64_t token_position = 0;

    friend bool operator==(const CheckpointHandle&, const CheckpointHandle&) = default;
    friend bool operator<(const CheckpointHandle& a, const CheckpointHandle& b) noexcept {
        return a.owner_request_id < b.owner_request_id ||
               (a.owner_request_id == b.owner_request_id && a.token_position < b.token_position);
    }
};

struct RestoredCheckpoint {
    CheckpointHandle handle{};
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
};

struct CheckpointStoreOptions {
    // Bounded total serialized bytes retained across all saved checkpoints.
    std::uint64_t max_bytes = 64ULL * 1024ULL * 1024ULL;  // 64 MiB
    std::size_t max_checkpoints = 1024;
};

struct CheckpointStoreUsage {
    std::size_t entries = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t max_bytes = 0;
    std::uint64_t evictions = 0;
};

struct RuntimeSubmission {
    std::uint64_t id = 0;
    // Tokens the request may generate: max_new_tokens clamped to the context
    // limit and to the KV pool, minus the prompt. Callers cap generation to
    // this so a smaller num_ctx (or the pool size) is actually honoured.
    std::uint32_t max_new_tokens = 1;
    // Ungated only: the prompt exceeds the KV pool, so nothing was scheduled
    // (id is 0) and the request runs unscheduled (scheduler.bypassed).
    bool bypassed = false;
};

struct RuntimeRequestSummary {
    std::uint32_t preemptions = 0;
    std::uint64_t reused_prompt_tokens = 0;
    double queue_ms = 0.0;
};

// Non-blocking gate poll result for backends that run multiple requests on a
// shared worker thread (prevents blocking peer slots on logical pauses).
enum class GatePollStatus : std::uint8_t {
    granted = 0,
    yield_recompute = 1,
    yield_spill = 2,
    cancelled = 3,
    failed = 4,
};

struct GatePollResult {
    GatePollStatus status = GatePollStatus::granted;
    Status error;
    const char* reason = nullptr;
};

// Snapshot of a logical KV branch sequence owned by a Session or Checkpoint.
struct LogicalBranchInfo {
    std::uint64_t sequence_id = 0;
    std::size_t num_tokens = 0;
    std::size_t block_count = 0;
    std::size_t shared_block_count = 0;
    std::size_t copy_on_write_count = 0;
};

class RequestRuntime {
public:
    virtual ~RequestRuntime() = default;

    // Registers a request. Errors: invalid_argument (can never fit the KV
    // pool / invalid), internal.
    virtual Result<RuntimeSubmission> submit(RuntimeRequestSpec spec) = 0;
    // Blocks until the scheduler grants the next token. Returns cancelled
    // when `cancel` trips while waiting, unavailable when the request was
    // failed by the scheduler (requeue limit, KV exhaustion) or on shutdown.
    virtual Status acquire_token(std::uint64_t id, const CancellationToken& cancel,
                                 std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt) = 0;
    // Non-blocking token gate poll: consumes a credit if immediately available
    // (or passthrough), or returns yield_recompute / yield_spill without
    // sleeping/blocking so shared batch workers never stall peer requests.
    virtual GatePollResult try_acquire_token_nonblocking(std::uint64_t id, const CancellationToken& cancel) {
        (void)id;
        if (cancel.cancelled()) {
            return GatePollResult{GatePollStatus::cancelled, Status(ErrorCode::cancelled, "cancelled"), "cancelled"};
        }
        return GatePollResult{GatePollStatus::granted, Status::success(), nullptr};
    }
    // Non-blocking safe-point preemption check (does not consume a token credit).
    virtual SuspendDecision poll_suspend_decision(std::uint64_t id, const CancellationToken& cancel) {
        (void)id;
        (void)cancel;
        return SuspendDecision{};
    }
    // Waits for admission without consuming a token grant, leaving
    // the first gate credit available for the backend's first token.
    virtual Status admit_backend(std::uint64_t id, const CancellationToken& cancel,
                                 std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt) = 0;
    // Records the token produced after a successful acquire_token().
    virtual void token_produced(std::uint64_t id, TokenId token) = 0;
    // Ends the request and frees its KV blocks. Safe to call once per id.
    virtual RuntimeRequestSummary finish(std::uint64_t id, RequestOutcome outcome) = 0;

    // Session/checkpoint logical KV lifecycle via KvCacheManager::fork_sequence & truncate.
    virtual Result<LogicalBranchInfo> ensure_logical_sequence(
        std::uint64_t existing_seq_id, std::span<const TokenId> tokens, std::uint64_t fingerprint,
        ModelArchitecture architecture, std::int32_t priority,
        std::span<const std::uint8_t> recurrent_checkpoint_blob = {}) {
        (void)existing_seq_id;
        (void)tokens;
        (void)fingerprint;
        (void)architecture;
        (void)priority;
        (void)recurrent_checkpoint_blob;
        return LogicalBranchInfo{};
    }
    virtual Result<LogicalBranchInfo> fork_logical_sequence(
        std::uint64_t parent_seq_id, std::optional<std::size_t> rewind_to_tokens = std::nullopt,
        std::optional<std::int32_t> child_priority = std::nullopt) {
        (void)parent_seq_id;
        (void)rewind_to_tokens;
        (void)child_priority;
        return LogicalBranchInfo{};
    }
    virtual Result<LogicalBranchInfo> truncate_logical_sequence(std::uint64_t seq_id, std::size_t new_length) {
        (void)seq_id;
        (void)new_length;
        return LogicalBranchInfo{};
    }
    virtual Result<LogicalBranchInfo> append_logical_sequence(std::uint64_t seq_id,
                                                              std::span<const TokenId> suffix_tokens) {
        (void)seq_id;
        (void)suffix_tokens;
        return LogicalBranchInfo{};
    }
    virtual void release_logical_sequence(std::uint64_t seq_id) { (void)seq_id; }
    [[nodiscard]] virtual std::optional<LogicalBranchInfo> inspect_logical_sequence(std::uint64_t seq_id) const {
        (void)seq_id;
        return std::nullopt;
    }

    // Registers a backend-captured recurrent-state checkpoint for an active
    // request and records it in KvCacheManager::save_checkpoint(). No-op when
    // scheduling/cache is disabled.
    virtual Status register_checkpoint(std::uint64_t id, std::uint64_t token_position,
                                       std::vector<std::uint8_t> bytes) {
        (void)id;
        (void)token_position;
        (void)bytes;
        return Status::success();
    }
    // Returns the saved checkpoint matched during logical prefill for `id`, if
    // any. Safe to call concurrently with evictions.
    [[nodiscard]] virtual std::optional<RestoredCheckpoint> lookup_restored_checkpoint(std::uint64_t id) const {
        (void)id;
        return std::nullopt;
    }
    // Releases any checkpoint-induced cache sequence retention for `id` once
    // prefill capture has registered (or been skipped/failed). For ungated
    // (account-mode) or already-completed requests this frees the sequence
    // immediately so KV blocks are not pinned during decode.
    virtual void release_checkpoint_retention(std::uint64_t id) { (void)id; }
    // Current occupancy of the bounded checkpoint blob store.
    [[nodiscard]] virtual CheckpointStoreUsage checkpoint_store_usage() const { return {}; }

    [[nodiscard]] virtual KvUsage kv_usage() const = 0;
    // Requests the runtime and its scheduler still hold state for. Returns to
    // zero once every submitted request has finished (no per-request leak).
    [[nodiscard]] virtual std::size_t tracked_requests() const = 0;
};

// Returns null when scheduling is disabled or the modules are not built.
std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions& options, TelemetryBus& bus,
                                                     TelemetryContext engine_context,
                                                     CheckpointStoreOptions checkpoint_options = {});

}  // namespace sonder::inference::detail
