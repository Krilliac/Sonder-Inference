#include "sonder/inference/error.hpp"

namespace sonder::inference {

const char* to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::ok: return "ok";
        case ErrorCode::invalid_argument: return "invalid_argument";
        case ErrorCode::invalid_state: return "invalid_state";
        case ErrorCode::not_found: return "not_found";
        case ErrorCode::unavailable: return "unavailable";
        case ErrorCode::cancelled: return "cancelled";
        case ErrorCode::timeout: return "timeout";
        case ErrorCode::backend_error: return "backend_error";
        case ErrorCode::protocol_error: return "protocol_error";
        case ErrorCode::io_error: return "io_error";
        case ErrorCode::unsupported: return "unsupported";
        case ErrorCode::internal: return "internal";
    }
    return "unknown";
}

std::string Status::to_string() const {
    if (ok()) {
        return "ok";
    }
    std::string out = inference::to_string(code_);
    if (!message_.empty()) {
        out += ": ";
        out += message_;
    }
    return out;
}

}  // namespace sonder::inference
