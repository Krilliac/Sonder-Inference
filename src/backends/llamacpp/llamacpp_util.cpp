// Pure helpers for the llama.cpp backend (no llama.cpp dependency).
#include "sonder/backends/llamacpp/llamacpp_backend.h"

#include <cmath>
#include <limits>

namespace sonder::backends::llamacpp {

std::string_view ToString(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kOk: return "ok";
        case ErrorCode::kInvalidArgument: return "invalid_argument";
        case ErrorCode::kFileNotFound: return "file_not_found";
        case ErrorCode::kLoadFailed: return "load_failed";
        case ErrorCode::kNotLoaded: return "not_loaded";
        case ErrorCode::kTokenizeFailed: return "tokenize_failed";
        case ErrorCode::kContextOverflow: return "context_overflow";
        case ErrorCode::kDecodeFailed: return "decode_failed";
    }
    return "unknown";
}

std::string_view ToString(StopReason reason) noexcept {
    switch (reason) {
        case StopReason::kNone: return "none";
        case StopReason::kEndOfGeneration: return "end_of_generation";
        case StopReason::kMaxTokens: return "max_tokens";
        case StopReason::kCancelled: return "cancelled";
        case StopReason::kContextFull: return "context_full";
        case StopReason::kError: return "error";
    }
    return "unknown";
}

std::string_view ToString(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::kCpu: return "cpu";
        case DeviceKind::kGpu: return "gpu";
        case DeviceKind::kIntegratedGpu: return "igpu";
        case DeviceKind::kAccelerator: return "accel";
        case DeviceKind::kOther: return "other";
    }
    return "unknown";
}

Status Validate(const SamplingParams& params) {
    if (!std::isfinite(params.temperature)) {
        return Status::Error(ErrorCode::kInvalidArgument, "temperature must be finite");
    }
    if (!std::isfinite(params.top_p) || params.top_p <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "top_p must be finite and > 0");
    }
    if (!std::isfinite(params.min_p) || params.min_p > 1.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "min_p must be finite and <= 1");
    }
    if (!std::isfinite(params.repeat_penalty) || params.repeat_penalty <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "repeat_penalty must be finite and > 0");
    }
    if (params.repeat_last_n < -1) {
        return Status::Error(ErrorCode::kInvalidArgument, "repeat_last_n must be >= -1");
    }
    if (!std::isfinite(params.presence_penalty) || !std::isfinite(params.frequency_penalty)) {
        return Status::Error(ErrorCode::kInvalidArgument, "presence/frequency penalties must be finite");
    }
    if (!std::isfinite(params.typical_p) || params.typical_p <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "typical_p must be finite and > 0");
    }
    for (const auto& entry : params.logit_bias) {
        if (std::isnan(entry.second) || entry.second == std::numeric_limits<float>::infinity()) {
            return Status::Error(ErrorCode::kInvalidArgument, "logit_bias values must be finite or -inf");
        }
    }
    return Status::Ok();
}

double GenerateResult::decode_tokens_per_second() const noexcept {
    return decode_ms > 0.0 ? generated_tokens * 1000.0 / decode_ms : 0.0;
}

double GenerateResult::prefill_tokens_per_second() const noexcept {
    return prefill_ms > 0.0 ? prompt_tokens * 1000.0 / prefill_ms : 0.0;
}

namespace {

// Expected total length of a UTF-8 sequence from its lead byte; 0 if the byte
// is a continuation byte or an invalid lead.
std::size_t SequenceLength(unsigned char lead) noexcept {
    if (lead < 0x80U) return 1;
    if ((lead & 0xE0U) == 0xC0U) return lead >= 0xC2U ? 2 : 0;
    if ((lead & 0xF0U) == 0xE0U) return 3;
    if ((lead & 0xF8U) == 0xF0U) return lead <= 0xF4U ? 4 : 0;
    return 0;
}

}  // namespace

std::size_t CompleteUtf8Prefix(std::string_view s) noexcept {
    // Only the last (up to) 3 bytes can belong to an incomplete sequence.
    const std::size_t n = s.size();
    const std::size_t lookback = n < 4 ? n : 4;
    for (std::size_t back = 1; back <= lookback; ++back) {
        const auto byte = static_cast<unsigned char>(s[n - back]);
        if ((byte & 0xC0U) == 0x80U) continue;  // continuation byte, keep scanning
        const std::size_t need = SequenceLength(byte);
        if (need == 0) return n;  // invalid lead: pass bytes through unchanged
        return need > back ? n - back : n;
    }
    return n;  // only continuation bytes in lookback: malformed, pass through
}

std::string Utf8StreamBuffer::Push(std::string_view bytes) {
    pending_.append(bytes.data(), bytes.size());
    const std::size_t keep = CompleteUtf8Prefix(pending_);
    std::string out = pending_.substr(0, keep);
    pending_.erase(0, keep);
    return out;
}

std::string Utf8StreamBuffer::Flush() {
    std::string out;
    out.swap(pending_);
    return out;
}

}  // namespace sonder::backends::llamacpp
