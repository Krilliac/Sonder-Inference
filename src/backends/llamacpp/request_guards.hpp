// Request-boundary validators and exception-barrier helpers for the in-process
// llama.cpp backend (NAT-02). Pure C++20 with no llama.cpp header dependency so
// every guard can be unit-tested without loading a model.
#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

// Upstream llama.cpp v1.0.0 structural ceilings verified against source:
// - kMaxBackendProbeCandidates: probe graph candidate count in
//   llama_sampler_backend_support (src/llama-sampler.cpp:648), which feeds
//   ggml_top_k's GGML_ASSERT(a->ne[0] >= k) at ggml/src/ggml.c:5374.
// - kLlamaMaxSeq: LLAMA_MAX_SEQ in src/llama-cparams.h:8.
inline constexpr std::int32_t kMaxBackendProbeCandidates = 128 * 1024;
inline constexpr std::uint32_t kLlamaMaxSeq = 256u;
inline constexpr std::int32_t kMaxRepeatLastN = 1 << 24;
inline constexpr std::size_t kMaxGrammarBytes = std::size_t{1} << 20;
inline constexpr std::size_t kMaxStopSequences = 16u;
inline constexpr std::size_t kMaxStopSequenceBytes = 256u;
inline constexpr std::size_t kMaxErrorMessageBytes = 256u;

// Validates that every token id in `tokens` lies in `[0, n_vocab)`.
// Prevents std::out_of_range in llama_vocab::impl::token_get_attr / cache.at
// (src/llama-vocab.cpp:3603,4108) and llama_grammar_accept_impl (:1477).
[[nodiscard]] Status ValidateTokenIds(std::span<const Token> tokens, std::int32_t n_vocab,
                                      std::string_view label = "token");

// Validates that `prompt` is non-empty and all tokens lie in `[0, n_vocab)`.
// Prevents GGML_ASSERT(n_tok > 0) in llama_batch_allocr::init (src/llama-batch.cpp:41).
[[nodiscard]] Status ValidatePromptTokens(std::span<const Token> prompt, std::int32_t n_vocab);

// Validates logit-bias token ids `[0, n_vocab)` and bias values (finite or -inf).
// Prevents GGML_ASSERT(lb.token >= 0 && lb.token < n_vocab) in
// llama_sampler_logit_bias_backend_set_input (src/llama-sampler.cpp:3991).
[[nodiscard]] Status ValidateLogitBias(const std::vector<std::pair<Token, float>>& logit_bias,
                                       std::int32_t n_vocab);

// Validates top_k against `kMaxBackendProbeCandidates` (128*1024) when enabled (> 0)
// and checks `n_vocab > 0`. Values in `(n_vocab, kMaxBackendProbeCandidates]` are
// valid and clamped to `n_vocab` via `ClampTopK` at sampler init to match upstream
// CPU top-k clamping (src/llama-sampler.cpp:331) while preventing GGML_ASSERT(a->ne[0] >= k)
// in ggml_top_k (ggml/src/ggml.c:5374) via llama_sampler_backend_support (:648) and :1493.
[[nodiscard]] Status ValidateTopK(std::int32_t top_k, std::int32_t n_vocab);

// Clamps an enabled `top_k` (> 0) to `std::min(top_k, n_vocab)` for `llama_sampler_init_top_k`.
[[nodiscard]] std::int32_t ClampTopK(std::int32_t top_k, std::int32_t n_vocab) noexcept;

// Validates GBNF grammar size/NUL, lazy-grammar triggers (non-empty/NUL patterns,
// trigger_tokens in `[0, n_vocab)`), grammar_prefill preconditions, and preserved_tokens.
[[nodiscard]] Status ValidateGrammarInputs(const SamplingParams& params, std::int32_t n_vocab);

// Runs all request-time sampling guards that require `n_vocab`
// (logit_bias token bounds, top_k probe ceiling, repeat_last_n cap, top_logprobs, grammar inputs).
[[nodiscard]] Status ValidateSamplingRequestGuards(const SamplingParams& params, std::int32_t n_vocab);

// Validates sequence id in `[0, n_seq_max)` with `1 <= n_seq_max <= kLlamaMaxSeq`.
// Prevents GGML_ASSERT(seq_id >= 0 && seq_id < seq_to_stream.size()) in
// src/llama-kv-cache.cpp:447,515,516,607,723,736,1301,1307,2325,2871.
[[nodiscard]] Status ValidateSeqId(std::int32_t seq_id, std::uint32_t n_seq_max);

// Validates KV position / n_past bounds without signed overflow:
// `n_ctx > 0`, `n_past >= 0`, `n_tokens > 0`, and `n_past + n_tokens <= n_ctx`.
[[nodiscard]] Status ValidateContextPosition(std::int64_t n_past, std::int64_t n_tokens,
                                             std::int64_t n_ctx);

// Validates batch token count against `n_batch` and (when `require_ubatch_fit` is true,
// e.g. encoder-only or non-causal contexts) `n_ubatch`.
// Prevents GGML_ASSERT(n_tok > 0) (src/llama-batch.cpp:41),
// GGML_ASSERT(n_tokens_all <= cparams.n_batch) (src/llama-context.cpp:2227), and
// GGML_ASSERT(cparams.n_ubatch >= n_tokens) (src/llama-context.cpp:1928,2229).
[[nodiscard]] Status ValidateBatchBounds(std::int32_t n_tokens, std::int32_t n_batch,
                                         std::int32_t n_ubatch, bool require_ubatch_fit);

// Validates multimodal request fields before calling libmtmd: rejects requests carrying
// media blobs or media_prompt_parts image references when no mmproj is loaded (or mtmd
// is not built), checks blob bounds, and checks every `media_index < media_blobs.size()`.
[[nodiscard]] Status ValidateMultimodalRequest(const GenerateRequest& request, bool has_mmproj,
                                               bool has_mtmd_build);

// Validates post-tokenization multimodal chunk totals before `mtmd_helper_eval_chunks`.
// Prevents GGML_ASSERT(n_batch > 0) (tools/mtmd/mtmd-helper.cpp:131,212), context
// overflow, and empty-chunk silent no-op followed by Debug GGML_ABORT in
// llama_get_logits_ith (src/llama-context.cpp:1143).
[[nodiscard]] Status ValidateMultimodalEvalBounds(std::int64_t total_tokens, std::int64_t total_pos,
                                                  std::int64_t n_ctx, std::int32_t n_batch);

// Validates stop-sequence count (`<= kMaxStopSequences`), non-empty, NUL-free,
// and byte length (`<= kMaxStopSequenceBytes`).
[[nodiscard]] Status ValidateStopSequences(std::span<const std::string> stops);

// Formats a bounded (`<= kMaxErrorMessageBytes`), ASCII, prompt-free error message
// describing an exception caught at `stage`. Never echoes `ex->what()` verbatim
// because llama.cpp exceptions can embed generated or prompt text.
[[nodiscard]] std::string FormatBoundedExceptionMessage(std::string_view stage,
                                                        const std::exception* ex = nullptr) noexcept;

// Converts a caught exception into a typed Status with a bounded prompt-free message.
[[nodiscard]] Status ExceptionToStatus(ErrorCode code, std::string_view stage,
                                       const std::exception* ex = nullptr) noexcept;

// Narrow exception barrier for calling potentially-throwing llama.cpp / ggml / mtmd code.
// Optional `oom_code` overrides `error_code` specifically for `std::bad_alloc` so OOM
// is never misclassified as an invalid-argument or decode error.
template <typename Fn>
[[nodiscard]] Status RunGuarded(ErrorCode error_code, std::string_view stage, Fn&& fn,
                                std::optional<ErrorCode> oom_code = std::nullopt) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc& ex) {
        return ExceptionToStatus(oom_code.value_or(error_code), stage, &ex);
    } catch (const std::exception& ex) {
        return ExceptionToStatus(error_code, stage, &ex);
    } catch (...) {
        return ExceptionToStatus(error_code, stage, nullptr);
    }
}

}  // namespace sonder::backends::llamacpp
