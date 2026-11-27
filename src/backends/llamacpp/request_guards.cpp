// Request-boundary validators and exception-barrier helpers for the in-process
// llama.cpp backend (NAT-02).
#include "request_guards.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <regex>
#include <stdexcept>

namespace sonder::backends::llamacpp {

namespace {

std::string TruncateAscii(std::string_view text, std::size_t max_bytes) {
    const std::size_t n = std::min(text.size(), max_bytes);
    std::string out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto ch = static_cast<unsigned char>(text[i]);
        out.push_back((ch >= 0x20U && ch <= 0x7EU) ? static_cast<char>(ch) : '?');
    }
    return out;
}

std::string_view ClassifyException(const std::exception* ex) noexcept {
    if (ex == nullptr) return "unknown internal exception";
    if (dynamic_cast<const std::bad_alloc*>(ex) != nullptr) return "out of memory";
    if (dynamic_cast<const std::regex_error*>(ex) != nullptr) return "invalid regex pattern";
    if (dynamic_cast<const std::out_of_range*>(ex) != nullptr) return "index out of range";
    if (dynamic_cast<const std::length_error*>(ex) != nullptr) return "length limit exceeded";
    if (dynamic_cast<const std::invalid_argument*>(ex) != nullptr) return "invalid argument";
    if (dynamic_cast<const std::runtime_error*>(ex) != nullptr) return "runtime error";
    return "internal exception";
}

}  // namespace

Status ValidateTokenIds(std::span<const Token> tokens, std::int32_t n_vocab, std::string_view label) {
    if (n_vocab <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "vocabulary size must be positive");
    }
    for (const Token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            return Status::Error(ErrorCode::kInvalidArgument, std::string(label) + " token id out of range");
        }
    }
    return Status::Ok();
}

Status ValidatePromptTokens(std::span<const Token> prompt, std::int32_t n_vocab) {
    if (prompt.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "prompt is empty");
    }
    return ValidateTokenIds(prompt, n_vocab, "prompt");
}

Status ValidateLogitBias(const std::vector<std::pair<Token, float>>& logit_bias, std::int32_t n_vocab) {
    if (n_vocab <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "vocabulary size must be positive");
    }
    if (logit_bias.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return Status::Error(ErrorCode::kInvalidArgument, "logit_bias list is too large");
    }
    for (const auto& [token, bias] : logit_bias) {
        if (token < 0 || token >= n_vocab) {
            return Status::Error(ErrorCode::kInvalidArgument, "logit_bias token out of range");
        }
        const bool is_neg_inf = std::isinf(bias) && bias < 0.0F;
        if (!std::isfinite(bias) && !is_neg_inf) {
            return Status::Error(ErrorCode::kInvalidArgument, "logit_bias values must be finite or -inf");
        }
    }
    return Status::Ok();
}

Status ValidateTopK(std::int32_t top_k, std::int32_t n_vocab) {
    if (n_vocab <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "vocabulary size must be positive");
    }
    if (top_k <= 0) {
        return Status::Ok();
    }
    if (top_k > n_vocab) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "top_k (" + std::to_string(top_k) + ") exceeds vocabulary size (" +
                                 std::to_string(n_vocab) + ")");
    }
    if (top_k > kMaxBackendProbeCandidates) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "top_k (" + std::to_string(top_k) + ") exceeds backend sampler limit (" +
                                 std::to_string(kMaxBackendProbeCandidates) + ")");
    }
    return Status::Ok();
}

Status ValidateGrammarInputs(const SamplingParams& params, std::int32_t n_vocab) {
    if (n_vocab <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "vocabulary size must be positive");
    }
    if (params.grammar.size() > kMaxGrammarBytes) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "grammar exceeds maximum byte length (" + std::to_string(kMaxGrammarBytes) + ")");
    }
    if (params.grammar.find('\0') != std::string::npos) {
        return Status::Error(ErrorCode::kInvalidArgument, "grammar contains embedded NUL byte");
    }
    if (!params.grammar_prefill.empty() && (params.grammar.empty() || params.grammar_lazy)) {
        return Status::Error(ErrorCode::kInvalidArgument, "grammar_prefill requires a non-lazy grammar");
    }
    if (params.grammar_lazy) {
        if (params.grammar.empty() || (params.trigger_patterns.empty() && params.trigger_tokens.empty())) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "lazy grammar requires a grammar and at least one trigger pattern or token");
        }
        if (Status s = ValidateTokenIds(params.trigger_tokens, n_vocab, "grammar trigger"); !s.ok()) {
            return s;
        }
        for (const std::string& pat : params.trigger_patterns) {
            if (pat.empty()) {
                return Status::Error(ErrorCode::kInvalidArgument, "lazy grammar trigger pattern must be non-empty");
            }
            if (pat.find('\0') != std::string::npos) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "lazy grammar trigger pattern contains embedded NUL byte");
            }
            try {
                const std::regex compiled(pat);
                (void)compiled;
            } catch (const std::exception& ex) {
                return ExceptionToStatus(ErrorCode::kInvalidArgument, "lazy_grammar_trigger_pattern", &ex);
            } catch (...) {
                return ExceptionToStatus(ErrorCode::kInvalidArgument, "lazy_grammar_trigger_pattern", nullptr);
            }
        }
    }
    for (const std::string& text : params.preserved_tokens) {
        if (text.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "preserved token text must be non-empty");
        }
        if (text.find('\0') != std::string::npos) {
            return Status::Error(ErrorCode::kInvalidArgument, "preserved token text contains embedded NUL byte");
        }
    }
    return Status::Ok();
}

Status ValidateSamplingRequestGuards(const SamplingParams& params, std::int32_t n_vocab, std::int32_t n_ctx) {
    (void)n_ctx;
    if (Status s = Validate(params); !s.ok()) {
        return s;
    }
    if (n_vocab <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "vocabulary size must be positive");
    }
    if (Status s = ValidateLogitBias(params.logit_bias, n_vocab); !s.ok()) {
        return s;
    }
    if (params.repeat_last_n > kMaxRepeatLastN) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "repeat_last_n (" + std::to_string(params.repeat_last_n) + ") exceeds maximum (" +
                                 std::to_string(kMaxRepeatLastN) + ")");
    }
    if (params.temperature > 0.0F) {
        if (Status s = ValidateTopK(params.top_k, n_vocab); !s.ok()) {
            return s;
        }
    }
    return ValidateGrammarInputs(params, n_vocab);
}

Status ValidateSeqId(std::int32_t seq_id, std::uint32_t n_seq_max) {
    if (n_seq_max == 0 || n_seq_max > kLlamaMaxSeq) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "n_seq_max (" + std::to_string(n_seq_max) + ") must be in [1, " +
                                 std::to_string(kLlamaMaxSeq) + "]");
    }
    if (seq_id < 0 || static_cast<std::uint32_t>(seq_id) >= n_seq_max) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "sequence id (" + std::to_string(seq_id) + ") out of range [0, " +
                                 std::to_string(n_seq_max) + ")");
    }
    return Status::Ok();
}

Status ValidateContextPosition(std::int64_t n_past, std::int64_t n_tokens, std::int64_t n_ctx) {
    if (n_ctx <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "context length must be positive");
    }
    if (n_past < 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "n_past must be non-negative");
    }
    if (n_tokens <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "token count must be positive");
    }
    if (n_past >= n_ctx || n_tokens > n_ctx - n_past) {
        return Status::Error(ErrorCode::kContextOverflow, "prompt or position exceeds context window");
    }
    return Status::Ok();
}

Status ValidateBatchBounds(std::int32_t n_tokens, std::int32_t n_batch, std::int32_t n_ubatch,
                           bool require_ubatch_fit) {
    if (n_tokens <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "batch token count must be positive");
    }
    if (n_batch <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "n_batch must be positive");
    }
    if (n_tokens > n_batch) {
        return Status::Error(ErrorCode::kContextOverflow,
                             "batch size (" + std::to_string(n_tokens) + ") exceeds n_batch (" +
                                 std::to_string(n_batch) + ")");
    }
    if (require_ubatch_fit) {
        if (n_ubatch <= 0) {
            return Status::Error(ErrorCode::kInvalidArgument, "n_ubatch must be positive");
        }
        if (n_tokens > n_ubatch) {
            return Status::Error(ErrorCode::kContextOverflow,
                                 "batch size (" + std::to_string(n_tokens) + ") exceeds n_ubatch (" +
                                     std::to_string(n_ubatch) + ")");
        }
    }
    return Status::Ok();
}

Status ValidateMultimodalRequest(const GenerateRequest& request, bool has_mmproj, bool has_mtmd_build) {
    bool has_media_ref = !request.media_blobs.empty();
    for (const auto& part : request.media_prompt_parts) {
        if (part.media_index.has_value()) {
            has_media_ref = true;
            break;
        }
    }
    if (!has_media_ref) {
        return Status::Ok();
    }
    if (!has_mtmd_build) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "media_blobs provided but this build has no libmtmd support");
    }
    if (!has_mmproj) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "media_blobs provided but no mmproj multimodal projector is loaded (unsupported)");
    }
    if (Status s = ValidateMediaBlobs(request.media_blobs); !s.ok()) {
        return s;
    }
    for (const auto& part : request.media_prompt_parts) {
        if (part.media_index.has_value() && *part.media_index >= request.media_blobs.size()) {
            return Status::Error(ErrorCode::kInvalidArgument, "multimodal media_index out of range");
        }
    }
    return Status::Ok();
}

Status ValidateMultimodalEvalBounds(std::int64_t total_tokens, std::int64_t total_pos, std::int64_t n_ctx,
                                    std::int32_t n_batch) {
    if (n_batch <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "multimodal n_batch must be positive");
    }
    if (total_tokens <= 0 || total_pos <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "multimodal prompt produced no tokens");
    }
    if (n_ctx <= 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "context length must be positive");
    }
    if (total_pos > n_ctx) {
        return Status::Error(ErrorCode::kContextOverflow, "multimodal prompt exceeds context window");
    }
    return Status::Ok();
}

Status ValidateStopSequences(std::span<const std::string> stops) {
    if (stops.size() > kMaxStopSequences) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "too many stop sequences (maximum is " + std::to_string(kMaxStopSequences) + ")");
    }
    for (const std::string& s : stops) {
        if (s.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "stop sequence must be non-empty");
        }
        if (s.size() > kMaxStopSequenceBytes) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "stop sequence exceeds maximum length (" + std::to_string(kMaxStopSequenceBytes) +
                                     " bytes)");
        }
        if (s.find('\0') != std::string::npos) {
            return Status::Error(ErrorCode::kInvalidArgument, "stop sequence contains embedded NUL byte");
        }
    }
    return Status::Ok();
}

std::string FormatBoundedExceptionMessage(std::string_view stage, const std::exception* ex) noexcept {
    try {
        const std::string_view reason = ClassifyException(ex);
        const std::string_view safe_stage = stage.empty() ? std::string_view("llamacpp") : stage;
        const std::size_t suffix_len = 2u + reason.size();
        const std::size_t stage_budget =
            kMaxErrorMessageBytes > suffix_len ? (kMaxErrorMessageBytes - suffix_len) : 0u;
        if (stage_budget == 0u) {
            return TruncateAscii(reason, kMaxErrorMessageBytes);
        }
        std::string msg = TruncateAscii(safe_stage, stage_budget);
        msg += ": ";
        msg += reason;
        if (msg.size() > kMaxErrorMessageBytes) {
            msg.resize(kMaxErrorMessageBytes);
        }
        return msg;
    } catch (...) {
        return {};
    }
}

Status ExceptionToStatus(ErrorCode code, std::string_view stage, const std::exception* ex) noexcept {
    return Status::Error(code, FormatBoundedExceptionMessage(stage, ex));
}

}  // namespace sonder::backends::llamacpp
