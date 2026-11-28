#include "llguidance_sampler.hpp"

#include "llama.h"

#ifdef SONDER_LLAMACPP_HAS_LLGUIDANCE

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llguidance.h"

// Adapted from llama.cpp common/llguidance.cpp (MIT, pinned with the same llguidance commit). Changes:
// no process-wide tokenizer cache (it was keyed on the vocab pointer, which a model reload can
// reuse), errors instead of GGML_ABORT, and a matcher error is never swallowed (see the header).

namespace sonder::backends::llamacpp {

namespace {

struct LlgCtx {
    const llama_vocab* vocab = nullptr;
    std::string kind;
    std::string data;
    LlgTokenizer* tokenizer = nullptr;
    LlgMatcher* matcher = nullptr;
    std::size_t mask_bytes = 0;
};

std::size_t TokenizeCallback(const void* user_data, const uint8_t* bytes, std::size_t bytes_len, uint32_t* out,
                             std::size_t out_len) {
    // Called from Rust: nothing may unwind across the boundary.
    try {
        const auto* vocab = static_cast<const llama_vocab*>(user_data);
        const int r = llama_tokenize(vocab, reinterpret_cast<const char*>(bytes), static_cast<int32_t>(bytes_len),
                                     reinterpret_cast<int32_t*>(out), static_cast<int32_t>(out_len),
                                     /*add_special=*/false, /*parse_special=*/true);
        return r < 0 ? static_cast<std::size_t>(-static_cast<std::int64_t>(r)) : static_cast<std::size_t>(r);
    } catch (...) {
        return 0;
    }
}

LlgTokenizer* NewTokenizer(const llama_vocab* vocab, std::string& error) {
    llama_token eos = llama_vocab_eot(vocab);
    if (eos == LLAMA_TOKEN_NULL) eos = llama_vocab_eos(vocab);
    if (eos == LLAMA_TOKEN_NULL) {
        error = "llguidance needs a vocabulary with an end-of-generation token";
        return nullptr;
    }
    const std::size_t n = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    constexpr std::size_t kMaxTokenBytes = 1024;
    std::vector<uint32_t> lens(n);
    std::vector<uint8_t> bytes;
    bytes.reserve(n * 8 + 4096);
    for (std::size_t i = 0; i < n; ++i) {
        const llama_token tok = static_cast<llama_token>(i);
        const std::size_t off = bytes.size();
        bytes.resize(off + kMaxTokenBytes);
        char* dp = reinterpret_cast<char*>(bytes.data() + off);
        int size = llama_detokenize(vocab, &tok, 1, dp, static_cast<int32_t>(kMaxTokenBytes), false, false);
        if (size < 0) {
            error = "llama_detokenize failed for token " + std::to_string(i);
            return nullptr;
        }
        if (size == 0) {
            size = llama_detokenize(vocab, &tok, 1, dp + 1, static_cast<int32_t>(kMaxTokenBytes - 1), false, true);
            if (size < 0) {
                error = "llama_detokenize failed for special token " + std::to_string(i);
                return nullptr;
            }
            if (size != 0) {
                *dp = static_cast<char>(0xff);  // special-token prefix marker
                size += 1;
            }
        }
        lens[i] = static_cast<uint32_t>(size);
        bytes.resize(off + static_cast<std::size_t>(size));
    }
    LlgTokenizerInit init{};
    init.vocab_size = static_cast<uint32_t>(n);
    init.tok_eos = static_cast<uint32_t>(eos);
    init.token_lens = lens.data();
    init.token_bytes = bytes.data();
    init.tokenizer_json = nullptr;
    init.tokenize_assumes_string = true;
    init.tokenize_fn = TokenizeCallback;
    init.use_approximate_greedy_tokenize_fn = false;
    init.tokenize_user_data = vocab;
    init.slices = nullptr;
    char err[1024] = {0};
    LlgTokenizer* tok = llg_new_tokenizer(&init, err, sizeof(err));
    if (tok == nullptr) error = std::string("llguidance tokenizer error: ") + err;
    return tok;
}

LlgMatcher* NewMatcher(LlgTokenizer* tokenizer, const std::string& kind, const std::string& data,
                       std::string& error) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    // Keep inference threads off stderr (matching llamacpp_core_backend.cpp); callers inspect
    // llg_matcher_get_error() instead.
    cinit.log_stderr_level = 0;
    LlgMatcher* m = llg_new_matcher(&cinit, kind.c_str(), data.c_str());
    if (m == nullptr) {
        error = "llguidance could not create a matcher";
        return nullptr;
    }
    if (const char* e = llg_matcher_get_error(m)) {
        error = std::string("llguidance rejected the schema: ") + e;
        llg_free_matcher(m);
        return nullptr;
    }
    return m;
}

[[noreturn]] void Fail(LlgCtx* c, const char* what) {
    const char* e = c->matcher != nullptr ? llg_matcher_get_error(c->matcher) : nullptr;
    throw std::runtime_error(std::string("llguidance ") + what + (e != nullptr ? std::string(": ") + e : ""));
}

const char* Name(const llama_sampler*) { return "llguidance"; }

void Accept(llama_sampler* s, llama_token token) {
    auto* c = static_cast<LlgCtx*>(s->ctx);
    if (c->matcher == nullptr) return;
    if (llg_matcher_consume_token(c->matcher, static_cast<uint32_t>(token)) != 0 ||
        llg_matcher_get_error(c->matcher) != nullptr) {
        Fail(c, "rejected a generated token");
    }
}

void Apply(llama_sampler* s, llama_token_data_array* cur_p) {
    auto* c = static_cast<LlgCtx*>(s->ctx);
    if (c->matcher == nullptr) return;
    const uint32_t* mask = llg_matcher_get_mask(c->matcher);
    if (mask == nullptr) {
        if (llg_matcher_compute_mask(c->matcher) != 0) Fail(c, "failed to compute the token mask");
        mask = llg_matcher_get_mask(c->matcher);
        if (mask == nullptr) Fail(c, "produced no token mask");
    }
    for (std::size_t i = 0; i < cur_p->size; ++i) {
        const llama_token t = cur_p->data[i].id;
        if (t < 0 || (static_cast<std::size_t>(t) / 32 + 1) * 4 > c->mask_bytes ||
            (mask[t / 32] & (1U << (t % 32))) == 0) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

void Reset(llama_sampler* s) {
    auto* c = static_cast<LlgCtx*>(s->ctx);
    if (c->matcher != nullptr) llg_matcher_reset(c->matcher);
}

void Free(llama_sampler* s);

llama_sampler* Clone(const llama_sampler* s);

llama_sampler_i kIface = {
    /* .name              = */ Name,
    /* .accept            = */ Accept,
    /* .apply             = */ Apply,
    /* .reset             = */ Reset,
    /* .clone             = */ Clone,
    /* .free              = */ Free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

void Free(llama_sampler* s) {
    auto* c = static_cast<LlgCtx*>(s->ctx);
    if (c->matcher != nullptr) llg_free_matcher(c->matcher);
    if (c->tokenizer != nullptr) llg_free_tokenizer(c->tokenizer);
    delete c;
}

llama_sampler* Clone(const llama_sampler* s) {
    const auto* c = static_cast<const LlgCtx*>(s->ctx);
    auto* out = new LlgCtx(*c);
    out->matcher = c->matcher != nullptr ? llg_clone_matcher(c->matcher) : nullptr;
    out->tokenizer = c->tokenizer != nullptr ? llg_clone_tokenizer(c->tokenizer) : nullptr;
    return llama_sampler_init(&kIface, out);
}

}  // namespace

bool LlguidanceAvailable() noexcept { return true; }

llama_sampler* MakeLlguidanceJsonSchemaSampler(const llama_vocab* vocab, std::string_view json_schema,
                                               std::string& error) {
    if (vocab == nullptr) {
        error = "llguidance needs a loaded vocabulary";
        return nullptr;
    }
    auto ctx = std::make_unique<LlgCtx>();
    ctx->vocab = vocab;
    ctx->kind = "json_schema";
    ctx->data = std::string(json_schema);
    ctx->tokenizer = NewTokenizer(vocab, error);
    if (ctx->tokenizer == nullptr) return nullptr;
    ctx->matcher = NewMatcher(ctx->tokenizer, ctx->kind, ctx->data, error);
    if (ctx->matcher == nullptr) {
        llg_free_tokenizer(ctx->tokenizer);
        return nullptr;
    }
    ctx->mask_bytes = llg_matcher_get_mask_byte_size(ctx->matcher);
    const std::size_t want = ((static_cast<std::size_t>(llama_vocab_n_tokens(vocab)) + 31) / 32) * 4;
    if (ctx->mask_bytes != want) {
        error = "llguidance token mask size does not match the vocabulary";
        llg_free_matcher(ctx->matcher);
        llg_free_tokenizer(ctx->tokenizer);
        return nullptr;
    }
    return llama_sampler_init(&kIface, ctx.release());
}

}  // namespace sonder::backends::llamacpp

#else  // !SONDER_LLAMACPP_HAS_LLGUIDANCE

namespace sonder::backends::llamacpp {

bool LlguidanceAvailable() noexcept { return false; }

llama_sampler* MakeLlguidanceJsonSchemaSampler(const llama_vocab*, std::string_view, std::string& error) {
    error = "this build has no llguidance support (configure with -DSONDER_WITH_LLGUIDANCE=ON)";
    return nullptr;
}

}  // namespace sonder::backends::llamacpp

#endif
