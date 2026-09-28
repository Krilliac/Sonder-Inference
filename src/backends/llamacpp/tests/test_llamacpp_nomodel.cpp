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
