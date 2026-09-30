// Backend tests that need llama.cpp linked but no model weights. Tokenization
// uses llama.cpp's bundled vocab-only GGUF fixtures (tokenizer metadata only).
#include <regex>
#include <string>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

#ifndef SONDER_LLAMACPP_VOCAB_DIR
#error "SONDER_LLAMACPP_VOCAB_DIR must be defined by the build"
#endif

TEST_CASE("llamacpp wrapper: pinned version and CPU device") {
    CHECK(LlamaCppBackend::UpstreamVersion() == "b11195");
    const auto devices = LlamaCppBackend::EnumerateDevices();
    bool has_cpu = false;
    for (const auto& d : devices) has_cpu = has_cpu || d.kind == DeviceKind::kCpu;
    CHECK(has_cpu);
    CHECK(!LlamaCppBackend::SystemInfo().empty());
}

TEST_CASE("llamacpp wrapper: operations before load fail cleanly") {
    LlamaCppBackend be;
    CHECK(!be.IsLoaded());
    std::vector<Token> toks;
    CHECK(be.Tokenize("hi", true, toks).code == ErrorCode::kNotLoaded);
    std::string piece;
    CHECK(be.TokenToPiece(1, piece).code == ErrorCode::kNotLoaded);
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    const GenerateResult r = be.Generate(req, {});
    CHECK(r.status.code == ErrorCode::kNotLoaded);
    CHECK(r.stop_reason == StopReason::kError);
    be.Unload();  // no-op, must not crash
    CHECK(be.GetModelInfo().n_vocab == 0);
}

TEST_CASE("llamacpp wrapper: load rejects bad paths") {
    LlamaCppBackend be;
    LoadOptions opts;
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.model_path = "this/file/does/not/exist.gguf";
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    opts.model_path = SONDER_LLAMACPP_VOCAB_DIR;  // a directory, not a file
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    CHECK(!be.IsLoaded());
}

TEST_CASE("llamacpp wrapper: vocab-only GGUF tokenizes and detokenizes") {
    LlamaCppBackend be;
    int loaded = 0;
    int unloaded = 0;
    be.SetTelemetrySink([&](const TelemetryEvent& e) {
        if (e.kind == TelemetryKind::kModelLoaded) ++loaded;
        if (e.kind == TelemetryKind::kModelUnloaded) ++unloaded;
    });
    LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    opts.vocab_only = true;
    const Status s = be.Load(opts);
    REQUIRE(s.ok());
    CHECK(be.IsLoaded());
    CHECK(loaded == 1);
    const ModelInfo info = be.GetModelInfo();
    CHECK(info.n_vocab == 32000);
    CHECK(info.n_ctx == 0u);

    std::vector<Token> with_bos;
    std::vector<Token> no_bos;
    REQUIRE(be.Tokenize("Hello world", true, with_bos).ok());
    REQUIRE(be.Tokenize("Hello world", false, no_bos).ok());
    CHECK(!no_bos.empty());
    CHECK(with_bos.size() == no_bos.size() + 1);

    std::string text;
    for (Token t : no_bos) {
        std::string piece;
        REQUIRE(be.TokenToPiece(t, piece).ok());
        text += piece;
    }
    // SentencePiece adds a leading space to the first word.
    CHECK((text == " Hello world" || text == "Hello world"));

    std::string piece;
    CHECK(be.TokenToPiece(-1, piece).code == ErrorCode::kInvalidArgument);
    CHECK(be.TokenToPiece(info.n_vocab, piece).code == ErrorCode::kInvalidArgument);

    // Multi-byte text round-trips through the UTF-8 stream buffer.
    std::vector<Token> uni;
    REQUIRE(be.Tokenize("caf\xC3\xA9 \xE2\x82\xAC", false, uni).ok());
    Utf8StreamBuffer buf;
    std::string streamed;
    for (Token t : uni) {
        REQUIRE(be.TokenToPiece(t, piece).ok());
        streamed += buf.Push(piece);
    }
    streamed += buf.Flush();
    CHECK(streamed.find("caf\xC3\xA9") != std::string::npos);
    CHECK(streamed.find("\xE2\x82\xAC") != std::string::npos);

    GenerateRequest req;
    req.prompt = no_bos;
    CHECK(be.Generate(req, {}).status.code == ErrorCode::kNotLoaded);  // no context

    be.Unload();
    CHECK(!be.IsLoaded());
    CHECK(unloaded == 1);
}

TEST_CASE("llamacpp wrapper: tensor overrides are validated before any load") {
    CHECK(LlamaCppBackend::ResolveTensorOverrides({}).ok());
    std::vector<void*> buffers;
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{kMoeExpertTensorPattern, "cpu"}}, &buffers).ok());
    CHECK(buffers.size() == 1);
    CHECK(buffers.front() != nullptr);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"x", "CPU"}}).ok());  // device name is case-insensitive for cpu
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"", "cpu"}}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"(unclosed", "cpu"}}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"x", "NoSuchDevice9"}}).code == ErrorCode::kInvalidArgument);

    // Load reports a bad override even when the model path does not exist.
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.tensor_overrides = {{"x", "NoSuchDevice9"}};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("llamacpp wrapper: the MoE expert pattern matches expert tensors only") {
    const std::regex re(kMoeExpertTensorPattern);
    CHECK(std::regex_search("blk.3.ffn_up_exps.weight", re));
    CHECK(std::regex_search("blk.12.ffn_down_exps.weight", re));
    CHECK(std::regex_search("blk.0.ffn_gate_exps.weight", re));
    CHECK(std::regex_search("blk.7.ffn_up_chexps.weight", re));
    CHECK_FALSE(std::regex_search("blk.3.ffn_up.weight", re));        // dense FFN
    CHECK_FALSE(std::regex_search("blk.3.ffn_up_shexp.weight", re));  // shared expert stays on GPU
    CHECK_FALSE(std::regex_search("blk.3.attn_q.weight", re));
    CHECK_FALSE(std::regex_search("blk.3.ffn_gate_inp.weight", re));  // router
}

TEST_CASE("llamacpp wrapper: default context params are unchanged and match llama.cpp's") {
    const ContextParamsSummary upstream = LlamaCppBackend::UpstreamContextDefaults();
    // Pinned b11195 defaults; if an upgrade moves them, Sonder's defaults move too.
    CHECK(upstream.type_k == "f16");
    CHECK(upstream.type_v == "f16");
    CHECK(upstream.flash_attn == "auto");
    CHECK(upstream.n_ubatch == 512u);

    // Default LoadOptions pass exactly what Load() passed before these options
    // existed: n_ubatch = min(llama.cpp default, n_batch), llama.cpp's KV types
    // and flash attention mode.
    LoadOptions d;
    ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(d);
    CHECK(s.n_ctx == 2048u);
    CHECK(s.n_batch == 512u);
    CHECK(s.n_ubatch == 512u);
    CHECK(s.type_k == upstream.type_k);
    CHECK(s.type_v == upstream.type_v);
    CHECK(s.flash_attn == upstream.flash_attn);

    d.n_batch = 64;  // smaller batch caps the micro-batch, as before
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 64u);
    d.n_batch = 2048;  // larger batch keeps llama.cpp's micro-batch
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 512u);
    d.n_ubatch = 1024;  // explicit micro-batch is passed through
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 1024u);
}

TEST_CASE("llamacpp wrapper: KV cache types and flash attention reach llama_context_params") {
    const KvCacheType all[] = {KvCacheType::kF16,  KvCacheType::kF32,  KvCacheType::kBF16,
                               KvCacheType::kQ8_0, KvCacheType::kQ5_1, KvCacheType::kQ5_0,
                               KvCacheType::kQ4_1, KvCacheType::kQ4_0, KvCacheType::kIQ4_NL};
    for (const KvCacheType t : all) {
        CAPTURE(ToString(t));
        LoadOptions o;
        o.type_k = t;
        o.type_v = t;
        const ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(o);
        // ggml's own type name for the value we pass must be our name for it.
        CHECK(s.type_k == ToString(t));
        CHECK(s.type_v == ToString(t));
    }
    LoadOptions o;
    o.type_k = KvCacheType::kQ8_0;
    o.type_v = KvCacheType::kQ4_0;
    ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(o);
    CHECK(s.type_k == "q8_0");
    CHECK(s.type_v == "q4_0");
    o.flash_attn = FlashAttention::kEnabled;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "enabled");
    o.flash_attn = FlashAttention::kDisabled;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "disabled");
    o.flash_attn = FlashAttention::kAuto;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "auto");
}

TEST_CASE("llamacpp wrapper: Load rejects bad context options before touching the file") {
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.type_v = KvCacheType::kQ8_0;
    opts.flash_attn = FlashAttention::kDisabled;
    const Status s = be.Load(opts);
    CHECK(s.code == ErrorCode::kInvalidArgument);  // not kFileNotFound
    CHECK(s.message.find("flash attention") != std::string::npos);
    opts = {};
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.n_batch = 128;
    opts.n_ubatch = 256;
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    // Valid options still reach the file check.
    opts.n_ubatch = 128;
    opts.type_k = KvCacheType::kQ4_0;
    opts.type_v = KvCacheType::kQ4_0;
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    CHECK(!be.IsLoaded());
}
