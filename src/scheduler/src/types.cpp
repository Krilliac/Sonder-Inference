#include "sonder/inference/scheduler/types.hpp"

namespace sonder::inference::scheduler {

std::string_view to_string(WorkloadClass c) noexcept {
    switch (c) {
        case WorkloadClass::InteractiveUser: return "interactive_user";
        case WorkloadClass::OwnerOrchestrator: return "owner_orchestrator";
        case WorkloadClass::CriticVerification: return "critic_verification";
        case WorkloadClass::ImplementationWorker: return "implementation_worker";
        case WorkloadClass::ResearchWorker: return "research_worker";
        case WorkloadClass::BackgroundIndexing: return "background_indexing";
        case WorkloadClass::Maintenance: return "maintenance";
    }
    return "unknown";
}

std::string_view to_string(RequestState s) noexcept {
    switch (s) {
        case RequestState::Created: return "created";
        case RequestState::WaitingAdmission: return "waiting_admission";
        case RequestState::Admitted: return "admitted";
        case RequestState::Prefill: return "prefill";
        case RequestState::Decode: return "decode";
        case RequestState::Completed: return "completed";
        case RequestState::Preempted: return "preempted";
        case RequestState::Cancelled: return "cancelled";
        case RequestState::Failed: return "failed";
        case RequestState::Rejected: return "rejected";
    }
    return "unknown";
}

bool is_terminal(RequestState s) noexcept {
    return s == RequestState::Completed || s == RequestState::Cancelled ||
           s == RequestState::Failed || s == RequestState::Rejected;
}

std::string_view to_string(PreemptionMode m) noexcept {
    return m == PreemptionMode::Swap ? "swap" : "recompute";
}

std::string_view to_string(PreemptionReason r) noexcept {
    return r == PreemptionReason::KvPressure ? "kv_pressure" : "priority_admission";
}

std::string_view to_string(FailureReason r) noexcept {
    switch (r) {
        case FailureReason::None: return "none";
        case FailureReason::RequeueLimit: return "requeue_limit";
        case FailureReason::NeverFits: return "never_fits";
        case FailureReason::DuplicateSequence: return "duplicate_sequence";
        case FailureReason::DuplicateId: return "duplicate_id";
        case FailureReason::InvalidRequest: return "invalid_request";
    }
    return "unknown";
}

}  // namespace sonder::inference::scheduler
