// Core value types for the Sonder inference scheduler (Phase 2 policy).
//
// The scheduler is pure policy: it decides which sequences prefill/decode in
// each engine step. It never executes a backend and never inspects prompt text;
// all intent arrives as explicit metadata (docs/SCHEDULER.md).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sonder::inference::scheduler {

using RequestId = std::uint64_t;
using TokenCount = std::uint32_t;
using BlockCount = std::uint64_t;
/// Simulation/engine time in microseconds since an arbitrary epoch.
using TimeUs = std::int64_t;

/// Workload classes from docs/SCHEDULER.md, ordered from most to least
/// latency-sensitive. The numeric value is the default priority rank
/// (lower = more urgent); runtime policy may override per request.
enum class WorkloadClass : std::uint8_t {
    InteractiveUser = 0,
    OwnerOrchestrator = 1,
    CriticVerification = 2,
    ImplementationWorker = 3,  // typical agent fan-out
    ResearchWorker = 4,
    BackgroundIndexing = 5,
    Maintenance = 6,
};

inline constexpr std::size_t kWorkloadClassCount = 7;

[[nodiscard]] std::string_view to_string(WorkloadClass c) noexcept;
[[nodiscard]] constexpr std::size_t index_of(WorkloadClass c) noexcept {
    return static_cast<std::size_t>(c);
}

/// Request lifecycle (docs/SCHEDULER.md "Request state machine").
enum class RequestState : std::uint8_t {
    Created,
    WaitingAdmission,
    Admitted,
    Prefill,
    Decode,
    Completed,
    Preempted,
    Cancelled,
    Failed,
    Rejected,
};

[[nodiscard]] std::string_view to_string(RequestState s) noexcept;
[[nodiscard]] bool is_terminal(RequestState s) noexcept;

enum class Phase : std::uint8_t { Prefill, Decode };

/// How a preempted sequence will be resumed.
enum class PreemptionMode : std::uint8_t {
    Recompute,  // drop KV, re-prefill prompt + generated tokens later
    Swap,       // move KV to a lower tier, swap back in on resume (stub)
};

enum class PreemptionReason : std::uint8_t {
    KvPressure,        // a running decode needed a block and none were free
    PriorityAdmission, // an urgent waiting request displaced a lower class
};

[[nodiscard]] std::string_view to_string(PreemptionMode m) noexcept;
[[nodiscard]] std::string_view to_string(PreemptionReason r) noexcept;

/// Why a request ended in a non-completed terminal state.
enum class FailureReason : std::uint8_t {
    None,
    RequeueLimit,       // bounded requeue count exceeded (no-progress guard)
    NeverFits,          // prompt can never fit the KV capacity
    DuplicateSequence,  // identical sequence already active
    DuplicateId,
    InvalidRequest,
};

[[nodiscard]] std::string_view to_string(FailureReason r) noexcept;

/// Explicit metadata supplied by Sonder Runtime for one generation request.
struct RequestSpec {
    RequestId id = 0;
    WorkloadClass workload = WorkloadClass::ImplementationWorker;
    /// Opaque agent/task identifier (never interpreted, used for telemetry).
    std::string task_id;
    /// Optional override of the class-derived priority rank (lower = urgent).
    std::optional<int> priority_override;
    TokenCount prompt_tokens = 0;
    TokenCount max_new_tokens = 0;
    bool cancellable = true;
    /// Optional identity of the exact sequence (model + prompt hash, etc.).
    /// When non-empty, a second active request with the same fingerprint is
    /// refused unless allow_duplicate is set.
    std::string sequence_fingerprint;
    bool allow_duplicate = false;
};

/// Per-class counters container.
template <typename T>
using PerClass = std::array<T, kWorkloadClassCount>;

}  // namespace sonder::inference::scheduler
