// Unit tests for NAT-02 request guards and exception/abort barrier.
// Pure validator tests require no model weights; vocab-only grammar tests use
// llama.cpp's bundled vocab fixture (matching test_llamacpp_nomodel.cpp /
// test_llamacpp_util.cpp).
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "llamacpp_internal.hpp"
#include "request_guards.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

TEST_CASE("request guards: token ids outside vocab range are rejected") {
    const std::vector<Token> good{0, 1, 31999};
    CHECK(ValidateTokenIds(good, 32000, "prompt").ok());
    CHECK(ValidateTokenIds({}, 32000, "prompt").ok());

    CHECK(ValidateTokenIds(good, 0, "prompt").code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTokenIds(good, -5, "prompt").code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTokenIds(std::vector<Token>{-1}, 32000, "prompt").code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTokenIds(std::vector<Token>{32000}, 32000, "prompt").code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTokenIds(std::vector<Token>{1, 2, std::numeric_limits<Token>::max()}, 32000, "prompt").code ==
          ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: empty prompts and out-of-vocab prompt tokens are rejected") {
    CHECK(ValidatePromptTokens({}, 32000).code == ErrorCode::kInvalidArgument);
    CHECK(ValidatePromptTokens(std::vector<Token>{1, 2, 3}, 32000).ok());
    CHECK(ValidatePromptTokens(std::vector<Token>{1, -1}, 32000).code == ErrorCode::kInvalidArgument);
    CHECK(ValidatePromptTokens(std::vector<Token>{32000}, 32000).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: logit-bias token ids and bias values are validated") {
    const std::vector<std::pair<Token, float>> ok_bias{
        {0, 1.5F},
        {31999, -std::numeric_limits<float>::infinity()},
    };
    CHECK(ValidateLogitBias(ok_bias, 32000).ok());
    CHECK(ValidateLogitBias({}, 32000).ok());

    CHECK(ValidateLogitBias({{-1, 1.0F}}, 32000).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLogitBias({{32000, 1.0F}}, 32000).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateLogitBias({{1, std::numeric_limits<float>::quiet_NaN()}}, 32000).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateLogitBias({{1, std::numeric_limits<float>::infinity()}}, 32000).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateLogitBias({{0, 1.0F}}, 0).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: top_k bounds guard ggml_top_k against vocab and backend probe ceilings") {
    CHECK(ValidateTopK(0, 32000).ok());
    CHECK(ValidateTopK(-1, 32000).ok());
    CHECK(ValidateTopK(40, 32000).ok());
    CHECK(ValidateTopK(32000, 32000).ok());
    CHECK(ValidateTopK(kMaxBackendProbeCandidates, 151936).ok());

    // top_k > n_vocab (up to kMaxBackendProbeCandidates) is accepted and clamped at sampler init.
    CHECK(ValidateTopK(32001, 32000).ok());
    CHECK(ValidateTopK(32000 + 1000, 32000).ok());
    CHECK(ClampTopK(0, 32000) == 0);
    CHECK(ClampTopK(-1, 32000) == -1);
    CHECK(ClampTopK(40, 32000) == 40);
    CHECK(ClampTopK(32001, 32000) == 32000);
    CHECK(ClampTopK(32000 + 1000, 32000) == 32000);
    CHECK(ClampTopK(kMaxBackendProbeCandidates, 151936) == kMaxBackendProbeCandidates);
    // top_k > 128*1024 reaches GGML_ASSERT in llama_sampler_backend_support probe (src/llama-sampler.cpp:648).
    CHECK(ValidateTopK(kMaxBackendProbeCandidates + 1, 151936).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTopK(kMaxBackendProbeCandidates + 1, 32000).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateTopK(10, 0).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: sampling request guards enforce top_k, repeat_last_n, top_logprobs, and logit_bias") {
    SamplingParams p;
    p.temperature = 0.8F;
    p.top_k = 40;
    CHECK(ValidateSamplingRequestGuards(p, 32000).ok());

    // Greedy (temperature <= 0) does not attach top_k, so default top_k=40 on tiny vocab stays valid.
    SamplingParams greedy;
    greedy.temperature = 0.0F;
    greedy.top_k = 40;
    CHECK(ValidateSamplingRequestGuards(greedy, 16).ok());

    // Stochastic with top_k > n_vocab (within backend probe ceiling) is accepted and clamped.
    SamplingParams clamped_topk = p;
    clamped_topk.top_k = 32000 + 1000;
    CHECK(ValidateSamplingRequestGuards(clamped_topk, 32000).ok());
    CHECK(ClampTopK(clamped_topk.top_k, 32000) == 32000);
    clamped_topk.top_k = 50000;
    CHECK(ValidateSamplingRequestGuards(clamped_topk, 32000).ok());
    CHECK(ClampTopK(clamped_topk.top_k, 32000) == 32000);

    // Stochastic with top_k > kMaxBackendProbeCandidates is rejected.
    SamplingParams bad_topk = p;
    bad_topk.top_k = kMaxBackendProbeCandidates + 1;
    CHECK(ValidateSamplingRequestGuards(bad_topk, 32000).code == ErrorCode::kInvalidArgument);

    // Huge repeat_last_n cannot force multi-GB ring_buffer allocation.
    SamplingParams huge_window = p;
    huge_window.repeat_penalty = 1.1F;
    huge_window.repeat_last_n = kMaxRepeatLastN + 1;
    CHECK(ValidateSamplingRequestGuards(huge_window, 32000).code == ErrorCode::kInvalidArgument);

    // top_logprobs bounds.
    SamplingParams bad_lp = p;
    bad_lp.compute_logprobs = true;
    bad_lp.top_logprobs = kMaxTopLogprobs + 1;
    CHECK(ValidateSamplingRequestGuards(bad_lp, 32000).code == ErrorCode::kInvalidArgument);

    // Out-of-range logit_bias token.
    SamplingParams bad_bias = p;
    bad_bias.logit_bias = {{32000, -1.0F}};
    CHECK(ValidateSamplingRequestGuards(bad_bias, 32000).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: grammar and lazy-grammar trigger inputs are validated") {
    SamplingParams p;
    p.grammar = "root ::= [a-z]+";
    CHECK(ValidateGrammarInputs(p, 32000).ok());

    // Oversized grammar rejected.
    SamplingParams big;
    big.grammar.assign(kMaxGrammarBytes + 1, 'a');
    CHECK(ValidateGrammarInputs(big, 32000).code == ErrorCode::kInvalidArgument);

    // Embedded NUL in grammar rejected.
    SamplingParams nul;
    nul.grammar = std::string("root ::= \"a", 11);
    nul.grammar.push_back('\0');
    nul.grammar += "\"";
    CHECK(ValidateGrammarInputs(nul, 32000).code == ErrorCode::kInvalidArgument);

    // Lazy grammar without triggers rejected.
    SamplingParams no_trig = p;
    no_trig.grammar_lazy = true;
    CHECK(ValidateGrammarInputs(no_trig, 32000).code == ErrorCode::kInvalidArgument);

    // Out-of-range trigger token rejected.
    SamplingParams bad_trig_tok = p;
    bad_trig_tok.grammar_lazy = true;
    bad_trig_tok.trigger_tokens = {32000};
    CHECK(ValidateGrammarInputs(bad_trig_tok, 32000).code == ErrorCode::kInvalidArgument);
    bad_trig_tok.trigger_tokens = {-1};
    CHECK(ValidateGrammarInputs(bad_trig_tok, 32000).code == ErrorCode::kInvalidArgument);

    // Valid trigger token accepted.
    bad_trig_tok.trigger_tokens = {42};
    CHECK(ValidateGrammarInputs(bad_trig_tok, 32000).ok());

    // Empty or NUL-containing trigger_patterns rejected.
    SamplingParams empty_pat = p;
    empty_pat.grammar_lazy = true;
    empty_pat.trigger_patterns = {""};
    CHECK(ValidateGrammarInputs(empty_pat, 32000).code == ErrorCode::kInvalidArgument);

    SamplingParams nul_pat = p;
    nul_pat.grammar_lazy = true;
    std::string pat_with_nul = "ab";
    pat_with_nul[1] = '\0';
    nul_pat.trigger_patterns = {pat_with_nul};
    CHECK(ValidateGrammarInputs(nul_pat, 32000).code == ErrorCode::kInvalidArgument);

    // Preserved token empty or containing NUL rejected.
    SamplingParams bad_pres = p;
    bad_pres.preserved_tokens = {""};
    CHECK(ValidateGrammarInputs(bad_pres, 32000).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: sequence id bounds against n_seq_max and LLAMA_MAX_SEQ") {
    CHECK(ValidateSeqId(0, 1).ok());
    CHECK(ValidateSeqId(15, 16).ok());
    CHECK(ValidateSeqId(255, kLlamaMaxSeq).ok());

    CHECK(ValidateSeqId(-1, 4).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateSeqId(4, 4).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateSeqId(0, 0).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateSeqId(0, kLlamaMaxSeq + 1).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: n_past and context position bounds reject overflow and out-of-context ranges") {
    CHECK(ValidateContextPosition(0, 10, 4096).ok());
    CHECK(ValidateContextPosition(4000, 96, 4096).ok());

    CHECK(ValidateContextPosition(-1, 1, 4096).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateContextPosition(0, 0, 4096).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateContextPosition(0, -3, 4096).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateContextPosition(0, 1, 0).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateContextPosition(4096, 1, 4096).code == ErrorCode::kContextOverflow);
    CHECK(ValidateContextPosition(4000, 97, 4096).code == ErrorCode::kContextOverflow);
    CHECK(ValidateContextPosition(std::numeric_limits<std::int64_t>::max(), 1, 4096).code ==
          ErrorCode::kContextOverflow);
}

TEST_CASE("request guards: batch sizes beyond n_batch and n_ubatch are rejected") {
    CHECK(ValidateBatchBounds(1, 512, 256, /*require_ubatch_fit=*/false).ok());
    CHECK(ValidateBatchBounds(512, 512, 256, /*require_ubatch_fit=*/false).ok());
    CHECK(ValidateBatchBounds(256, 512, 256, /*require_ubatch_fit=*/true).ok());

    // Empty batch -> GGML_ASSERT(n_tok > 0) at src/llama-batch.cpp:41.
    CHECK(ValidateBatchBounds(0, 512, 256, false).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateBatchBounds(-1, 512, 256, false).code == ErrorCode::kInvalidArgument);
    // Exceeding n_batch -> GGML_ASSERT(n_tokens_all <= cparams.n_batch) at src/llama-context.cpp:2227.
    CHECK(ValidateBatchBounds(513, 512, 256, false).code == ErrorCode::kContextOverflow);
    // Exceeding n_ubatch when required -> GGML_ASSERT(cparams.n_ubatch >= n_tokens) at src/llama-context.cpp:1928,2229.
    CHECK(ValidateBatchBounds(257, 512, 256, true).code == ErrorCode::kContextOverflow);
    CHECK(ValidateBatchBounds(1, 0, 256, false).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateBatchBounds(1, 512, 0, true).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: multimodal request rejects media without mmproj, bad indices, and empty eval") {
    GenerateRequest plain;
    plain.prompt = {1, 2};
    CHECK(ValidateMultimodalRequest(plain, /*has_mmproj=*/false, /*has_mtmd_build=*/false).ok());

    // Media blobs with no mmproj loaded.
    GenerateRequest with_img = plain;
    with_img.media_blobs = {{0x89, 0x50, 0x4E, 0x47}};
    CHECK(ValidateMultimodalRequest(with_img, /*has_mmproj=*/false, /*has_mtmd_build=*/true).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateMultimodalRequest(with_img, /*has_mmproj=*/false, /*has_mtmd_build=*/false).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateMultimodalRequest(with_img, /*has_mmproj=*/true, /*has_mtmd_build=*/true).ok());

    // media_prompt_parts referencing an image when media_blobs is empty or mmproj not loaded.
    GenerateRequest dangling_part = plain;
    dangling_part.media_prompt_parts = {GenerateRequest::MediaPromptPart{{}, false, std::size_t{0}}};
    CHECK(ValidateMultimodalRequest(dangling_part, /*has_mmproj=*/false, /*has_mtmd_build=*/true).code ==
          ErrorCode::kInvalidArgument);
    CHECK(ValidateMultimodalRequest(dangling_part, /*has_mmproj=*/true, /*has_mtmd_build=*/true).code ==
          ErrorCode::kInvalidArgument);

    // Out-of-range media_index with blobs present.
    GenerateRequest bad_idx = with_img;
    bad_idx.media_prompt_parts = {GenerateRequest::MediaPromptPart{{}, false, std::size_t{1}}};
    CHECK(ValidateMultimodalRequest(bad_idx, /*has_mmproj=*/true, /*has_mtmd_build=*/true).code ==
          ErrorCode::kInvalidArgument);

    // Eval bounds: empty chunks, n_batch <= 0, context overflow.
    CHECK(ValidateMultimodalEvalBounds(10, 10, 4096, 512).ok());
    CHECK(ValidateMultimodalEvalBounds(0, 0, 4096, 512).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateMultimodalEvalBounds(10, 0, 4096, 512).code == ErrorCode::kInvalidArgument);
    CHECK(ValidateMultimodalEvalBounds(10, 4097, 4096, 512).code == ErrorCode::kContextOverflow);
    CHECK(ValidateMultimodalEvalBounds(10, 10, 4096, 0).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("request guards: stop-sequence bounds and StopSequenceFilter empty-string safety") {
    const std::vector<std::string> ok_stops{"\n\n", "END"};
    CHECK(ValidateStopSequences(ok_stops).ok());
    CHECK(ValidateStopSequences({}).ok());

    // Empty stop string rejected.
    const std::vector<std::string> empty_stop{""};
    CHECK(ValidateStopSequences(empty_stop).code == ErrorCode::kInvalidArgument);

    // Too many stop sequences rejected.
    std::vector<std::string> too_many(kMaxStopSequences + 1, "x");
    CHECK(ValidateStopSequences(too_many).code == ErrorCode::kInvalidArgument);

    // Overlength stop sequence rejected.
    std::vector<std::string> too_long{std::string(kMaxStopSequenceBytes + 1, 'a')};
    CHECK(ValidateStopSequences(too_long).code == ErrorCode::kInvalidArgument);

    // Stop sequence with embedded NUL rejected.
    std::string with_nul = "ab";
    with_nul[1] = '\0';
    CHECK(ValidateStopSequences(std::vector<std::string>{with_nul}).code == ErrorCode::kInvalidArgument);

    // StopSequenceFilter must not underflow s.size() - 1 if constructed with an empty stop.
    sonder::inference::llamacpp_detail::StopSequenceFilter filter(empty_stop);
    CHECK(filter.Push("hello") == "hello");
    CHECK_FALSE(filter.matched());
}

TEST_CASE("request guards: RunGuarded and FormatBoundedExceptionMessage bound messages and strip prompt text") {
    const std::string secret_prompt = "TOP_SECRET_USER_PROMPT_" + std::string(400, 'Z');

    const Status rt = RunGuarded(ErrorCode::kDecodeFailed, "llama_grammar_accept", [&]() -> Status {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + secret_prompt);
    });
    CHECK(rt.code == ErrorCode::kDecodeFailed);
    CHECK(rt.message.size() <= kMaxErrorMessageBytes);
    CHECK(rt.message.find("TOP_SECRET_USER_PROMPT") == std::string::npos);
    CHECK(rt.message.find("llama_grammar_accept") != std::string::npos);

    const Status oom = RunGuarded(
        ErrorCode::kInvalidArgument, "llama_chat_apply_template", [&]() -> Status { throw std::bad_alloc(); },
        ErrorCode::kTokenizeFailed);
    CHECK(oom.code == ErrorCode::kTokenizeFailed);
    CHECK(oom.message.size() <= kMaxErrorMessageBytes);
    CHECK(oom.message.find("out of memory") != std::string::npos);

    const Status sampler_oom = RunGuarded(
        ErrorCode::kDecodeFailed, "sampler_init", [&]() -> Status { throw std::bad_alloc(); }, ErrorCode::kLoadFailed);
    CHECK(sampler_oom.code == ErrorCode::kLoadFailed);
    CHECK(sampler_oom.message.size() <= kMaxErrorMessageBytes);
    CHECK(sampler_oom.message.find("out of memory") != std::string::npos);

    const Status rx = RunGuarded(ErrorCode::kInvalidArgument, "grammar_init", [&]() -> Status {
        throw std::regex_error(std::regex_constants::error_paren);
    });
    CHECK(rx.code == ErrorCode::kInvalidArgument);
    CHECK(rx.message.size() <= kMaxErrorMessageBytes);
    CHECK(rx.message.find("regex") != std::string::npos);

    const Status unk = RunGuarded(ErrorCode::kDecodeFailed, "llama_decode", [&]() -> Status {
        throw 42;
    });
    CHECK(unk.code == ErrorCode::kDecodeFailed);
    CHECK(unk.message.size() <= kMaxErrorMessageBytes);
    CHECK_FALSE(unk.message.empty());

    // Long stage name is still truncated to kMaxErrorMessageBytes.
    const std::string long_stage(512, 's');
    const std::string bounded = FormatBoundedExceptionMessage(long_stage, nullptr);
    CHECK(bounded.size() <= kMaxErrorMessageBytes);
}

#ifdef SONDER_LLAMACPP_VOCAB_DIR
TEST_CASE("request guards: ProbeGrammarMask blocks EOG abort, OOB tokens, OOB logit_bias, and OOB top_k (vocab-only)") {
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-bpe.gguf";
    opts.vocab_only = true;
    REQUIRE(be.Load(opts).ok());
    const std::int32_t n_vocab = be.GetModelInfo().n_vocab;
    REQUIRE(n_vocab > 0);

    std::vector<Token> eog;
    REQUIRE(be.Tokenize("<|eot_id|>", false, eog, /*parse_special=*/true).ok());
    REQUIRE(eog.size() == 1u);

    SamplingParams gram;
    gram.grammar = "root ::= \"hello\" [a-z]+";

    // Previously reached GGML_ABORT("fatal error") at src/llama-grammar.cpp:1527!
    std::vector<std::uint8_t> mask;
    Status st = Status::Ok();
    CHECK_NOTHROW(st = be.ProbeGrammarMask(gram, eog, mask));
    CHECK(st.code == ErrorCode::kDecodeFailed);
    CHECK(mask.empty());

    // Out-of-range accepted token returns kInvalidArgument (not std::out_of_range or kDecodeFailed).
    CHECK(be.ProbeGrammarMask(gram, std::vector<Token>{-1}, mask).code == ErrorCode::kInvalidArgument);
    CHECK(be.ProbeGrammarMask(gram, std::vector<Token>{n_vocab}, mask).code == ErrorCode::kInvalidArgument);

    // Out-of-range logit_bias token rejected in BuildSampler / ProbeGrammarMask.
    SamplingParams bad_bias = gram;
    bad_bias.logit_bias = {{n_vocab, 1.0F}};
    CHECK(be.ProbeGrammarMask(bad_bias, {}, mask).code == ErrorCode::kInvalidArgument);

    // top_k = n_vocab + 1000 (within kMaxBackendProbeCandidates) is accepted and clamped at sampler init.
    SamplingParams clamped_topk = gram;
    clamped_topk.temperature = 0.8F;
    clamped_topk.top_k = n_vocab + 1000;
    CHECK(be.ProbeGrammarMask(clamped_topk, {}, mask).ok());

    // top_k > kMaxBackendProbeCandidates is rejected before backend probe.
    SamplingParams bad_topk = gram;
    bad_topk.temperature = 0.8F;
    bad_topk.top_k = kMaxBackendProbeCandidates + 1;
    CHECK(be.ProbeGrammarMask(bad_topk, {}, mask).code == ErrorCode::kInvalidArgument);

    // Out-of-range lazy trigger token rejected.
    SamplingParams bad_trig = gram;
    bad_trig.grammar_lazy = true;
    bad_trig.trigger_tokens = {n_vocab};
    CHECK(be.ProbeGrammarMask(bad_trig, {}, mask).code == ErrorCode::kInvalidArgument);

    // Invalid lazy trigger regex rejected without echoing the pattern.
    SamplingParams bad_re = gram;
    bad_re.grammar_lazy = true;
    bad_re.trigger_patterns = {"([SECRET_PROMPT_MARKER"};
    const Status re_st = be.ProbeGrammarMask(bad_re, {}, mask);
    CHECK(re_st.code == ErrorCode::kInvalidArgument);
    CHECK(re_st.message.find("SECRET_PROMPT_MARKER") == std::string::npos);

    // Preserved token mapping failure does not echo the token text.
    SamplingParams bad_pres = gram;
    bad_pres.preserved_tokens = {"MULTI_TOKEN_SECRET_VALUE"};
    const Status pres_st = be.ProbeGrammarMask(bad_pres, {}, mask);
    CHECK(pres_st.code == ErrorCode::kInvalidArgument);
    CHECK(pres_st.message.find("MULTI_TOKEN_SECRET_VALUE") == std::string::npos);

    be.Unload();
}
#endif
