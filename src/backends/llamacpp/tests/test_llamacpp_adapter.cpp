// Core-interface adapter tests (no model weights needed).
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "llamacpp_internal.hpp"
#include "sonder/inference/backends/llamacpp.hpp"

using namespace sonder::inference;
using llamacpp_detail::StopSequenceFilter;
using llamacpp_detail::ToLlamaSampling;

TEST_CASE("llamacpp adapter: stop filter passes text through without stops") {
    StopSequenceFilter f({});
    CHECK(f.Push("hello ") == "hello ");
    CHECK(f.Push("world") == "world");
    CHECK(f.Flush().empty());
    CHECK_FALSE(f.matched());
}

TEST_CASE("llamacpp adapter: stop filter holds partial matches and cuts on a hit") {
    StopSequenceFilter f({"\nUser:", "###"});
    std::string out;
    out += f.Push("Answer is 4.");
    out += f.Push("\nUs");  // could be the start of "\nUser:"
    CHECK(out == "Answer is 4.");
    out += f.Push("er: next");
    CHECK(f.matched());
    CHECK(out == "Answer is 4.");
    CHECK(f.Push("more").empty());
    CHECK(f.Flush().empty());
}

TEST_CASE("llamacpp adapter: stop filter releases held text that diverges") {
    StopSequenceFilter f({"###"});
    std::string out = f.Push("a#");
    CHECK(out == "a");
    out += f.Push("#b");
    CHECK(out == "a##b");
    CHECK_FALSE(f.matched());
    out += f.Push("##");
    CHECK(out == "a##b");
    CHECK(f.Flush() == "##");
}

TEST_CASE("llamacpp adapter: sampling config mapping") {
    SamplingConfig greedy = SamplingConfig::greedy(32, 42);
    auto p = ToLlamaSampling(greedy);
    CHECK(p.temperature == 0.0f);
    CHECK(p.top_k == 0);
    CHECK(p.top_p == 1.0f);
    CHECK(p.repeat_penalty == 1.0f);
    CHECK(p.seed == 42u);

    SamplingConfig c;
    c.seed.reset();
    CHECK(ToLlamaSampling(c).seed == 0xFFFFFFFFu);  // llama.cpp random seed
    c.seed = 0xFFFFFFFFull;                           // must not collide with "random"
    CHECK(ToLlamaSampling(c).seed != 0xFFFFFFFFu);
    c.seed = 0x0000000100000002ull;
    CHECK(ToLlamaSampling(c).seed == 3u);
}

TEST_CASE("llamacpp adapter: backend identity, capabilities, probe") {
    auto be = make_llamacpp_backend();
    REQUIRE(be != nullptr);
    CHECK(be->name() == kLlamaCppBackendName);
    CHECK(be->description().find("b11195") != std::string::npos);
    const auto caps = be->capabilities();
    CHECK(caps.has(Capability::tokenization));
    CHECK(caps.has(Capability::streaming));
    CHECK(caps.has(Capability::batched_prefill));
    CHECK(caps.has(Capability::deterministic));
    CHECK_FALSE(caps.has(Capability::remote_process));
    CHECK_FALSE(caps.has(Capability::kv_export));
    auto probe = be->probe();
    REQUIRE(probe.ok());
    CHECK(probe.value().find("llama.cpp b11195") != std::string::npos);
    CHECK(probe.value().find("cpu") != std::string::npos);
}

TEST_CASE("llamacpp adapter: list_models scans model_dirs for gguf files") {
    LlamaCppBackendOptions opts;
    opts.model_dirs = {SONDER_LLAMACPP_VOCAB_DIR, "does/not/exist"};
    auto be = make_llamacpp_backend(opts);
    auto models = be->list_models();
    REQUIRE(models.ok());
    bool found = false;
    for (const auto& m : models.value()) {
        CHECK(m.format == "gguf");
        CHECK(m.backend == kLlamaCppBackendName);
        found = found || m.name == "ggml-vocab-llama-spm.gguf";
    }
    CHECK(found);
}

TEST_CASE("llamacpp adapter: load_model error mapping") {
    auto be = make_llamacpp_backend();
    ModelLoadOptions o;
    CHECK(be->load_model(o).status().code() == ErrorCode::invalid_argument);
    o.model = "missing-model.gguf";
    CHECK(be->load_model(o).status().code() == ErrorCode::not_found);
    o.model = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    o.device_id = "npu:0";
    CHECK(be->load_model(o).status().code() == ErrorCode::unsupported);
}
