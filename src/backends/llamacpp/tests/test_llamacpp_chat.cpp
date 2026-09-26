// Chat-template support in the llama.cpp wrapper (no model weights): built-in
// template formatting, special-token parsing for templated prompts, and the
// vocab-only GGUF fixtures bundled with llama.cpp.
#include <string>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

#ifndef SONDER_LLAMACPP_VOCAB_DIR
#error "SONDER_LLAMACPP_VOCAB_DIR must be defined by the build"
#endif

TEST_CASE("llamacpp chat: built-in chatml template") {
    const std::vector<ChatTurn> msgs{{"system", "Be brief."}, {"user", "Hi"}};
    std::string out;
    REQUIRE(LlamaCppBackend::FormatChat("chatml", msgs, /*add_assistant=*/true, out).ok());
    CHECK(out == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    REQUIRE(LlamaCppBackend::FormatChat("chatml", msgs, /*add_assistant=*/false, out).ok());
    CHECK(out == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n");
}

TEST_CASE("llamacpp chat: long conversations grow the buffer") {
    std::vector<ChatTurn> msgs;
    for (int i = 0; i < 50; ++i) {
        msgs.push_back({"user", std::string(40, 'u')});
        msgs.push_back({"assistant", std::string(40, 'a')});
    }
    msgs.push_back({"user", "last"});
    std::string out;
    REQUIRE(LlamaCppBackend::FormatChat("chatml", msgs, true, out).ok());
    CHECK(out.size() > 50u * 2u * 40u);
    CHECK(out.find("<|im_start|>user\nlast<|im_end|>\n<|im_start|>assistant\n") != std::string::npos);
}

TEST_CASE("llamacpp chat: invalid inputs") {
    std::string out = "stale";
    CHECK(LlamaCppBackend::FormatChat("", {{"user", "x"}}, true, out).code == ErrorCode::kInvalidArgument);
    CHECK(out.empty());
    CHECK(LlamaCppBackend::FormatChat("chatml", {}, true, out).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::FormatChat("definitely-not-a-template", {{"user", "x"}}, true, out).code ==
          ErrorCode::kInvalidArgument);

    LlamaCppBackend unloaded;
    CHECK(unloaded.ChatTemplate().empty());
    CHECK(unloaded.ApplyChatTemplate({{"user", "x"}}, true, out).code == ErrorCode::kNotLoaded);
}

TEST_CASE("llamacpp chat: vocab-only GGUF template lookup and special-token parsing") {
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-bpe.gguf";
    opts.vocab_only = true;
    REQUIRE(be.Load(opts).ok());

    // Whatever the fixture carries, template lookup and formatting agree.
    std::string out;
    const Status applied = be.ApplyChatTemplate({{"user", "Hi"}}, true, out);
    if (be.ChatTemplate().empty()) {
        CHECK(applied.code == ErrorCode::kInvalidArgument);
    } else if (applied.ok()) {
        CHECK(out.find("Hi") != std::string::npos);
    }

    // Templated prompts carry control tokens as text; parse_special maps them
    // to single special tokens instead of splitting them into pieces.
    std::vector<Token> plain;
    std::vector<Token> special;
    REQUIRE(be.Tokenize("<|begin_of_text|>", false, plain, /*parse_special=*/false).ok());
    REQUIRE(be.Tokenize("<|begin_of_text|>", false, special, /*parse_special=*/true).ok());
    CHECK(special.size() == 1u);
    CHECK(plain.size() > 1u);
}
