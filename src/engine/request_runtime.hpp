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
    // Waits for admission without consuming a token grant, leaving
    // the first gate credit available for the backend's first token.
    virtual Status admit_backend(std::uint64_t id, const CancellationToken& cancel,
                                 std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt) = 0;
    // Records the token produced after a successful acquire_token().
    virtual void token_produced(std::uint64_t id, TokenId token) = 0;
    // Ends the request and frees its KV blocks. Safe to call once per id.
    virtual RuntimeRequestSummary finish(std::uint64_t id, RequestOutcome outcome) = 0;

    [[nodiscard]] virtual KvUsage kv_usage() const = 0;
    // Requests the runtime and its scheduler still hold state for. Returns to
    // zero once every submitted request has finished (no per-request leak).
    [[nodiscard]] virtual std::size_t tracked_requests() const = 0;
};

// Returns null when scheduling is disabled or the modules are not built.
std::unique_ptr<RequestRuntime> make_request_runtime(const SchedulingOptions& options, TelemetryBus& bus,
                                                     TelemetryContext engine_context);

}  // namespace sonder::inference::detail
