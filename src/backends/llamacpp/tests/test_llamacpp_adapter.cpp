// Core-interface adapter tests (no model weights needed).
#include <doctest/doctest.h>

#include <limits>
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

TEST_CASE("llamacpp adapter: new sampling fields are mapped and validated") {
    SamplingConfig c;
    auto d = ToLlamaSampling(c);
    CHECK(d.repeat_last_n == 64);
    CHECK(d.presence_penalty == 0.0f);
    CHECK(d.frequency_penalty == 0.0f);
    CHECK(d.typical_p == 1.0f);
    CHECK(d.logit_bias.empty());

    c.repeat_last_n = -1;
    c.presence_penalty = 0.5f;
    c.frequency_penalty = -0.25f;
    c.typical_p = 0.9f;
    c.logit_bias = {{7, 2.5f}, {9, -std::numeric_limits<float>::infinity()}};
    auto p = ToLlamaSampling(c);
    CHECK(p.repeat_last_n == -1);
    CHECK(p.presence_penalty == 0.5f);
    CHECK(p.frequency_penalty == -0.25f);
    CHECK(p.typical_p == doctest::Approx(0.9f));
    REQUIRE(p.logit_bias.size() == 2u);
    CHECK(p.logit_bias[0].first == 7);
    CHECK(p.logit_bias[0].second == 2.5f);
    CHECK(p.logit_bias[1].second == -std::numeric_limits<float>::infinity());
    CHECK(sonder::backends::llamacpp::Validate(p).ok());

    p.repeat_last_n = -2;
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
    p.repeat_last_n = 64;
    p.typical_p = 0.0f;
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
    p.typical_p = 1.0f;
    p.logit_bias = {{1, std::numeric_limits<float>::quiet_NaN()}};
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
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

TEST_CASE("llamacpp adapter: context options default to today's behaviour and validate") {
    const LlamaCppBackendOptions d;
    CHECK(d.batch_size == 512u);
    CHECK(d.ubatch_size == 0u);
    CHECK(d.kv_cache_type_k == "f16");
    CHECK(d.kv_cache_type_v == "f16");
    CHECK(d.flash_attention == "auto");
    CHECK(validate_llamacpp_options(d).ok());

    LlamaCppBackendOptions o;
    o.kv_cache_type_k = "q8_0";
    o.kv_cache_type_v = "q8_0";
    CHECK(validate_llamacpp_options(o).ok());  // flash attention auto
    o.flash_attention = "on";
    CHECK(validate_llamacpp_options(o).ok());
    o.flash_attention = "off";
    Status s = validate_llamacpp_options(o);
    CHECK(s.code() == ErrorCode::invalid_argument);
    CHECK(s.message().find("flash attention") != std::string::npos);
    o.kv_cache_type_v = "f16";  // quantized K alone is fine without it
    CHECK(validate_llamacpp_options(o).ok());

    o = {};
    o.kv_cache_type_k = "q4_k";
    s = validate_llamacpp_options(o);
    CHECK(s.code() == ErrorCode::invalid_argument);
    CHECK(s.message().find("K cache type 'q4_k'") != std::string::npos);
    o = {};
    o.kv_cache_type_v = "";
    CHECK(validate_llamacpp_options(o).message().find("V cache type") != std::string::npos);
    o = {};
    o.flash_attention = "maybe";
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o = {};
    o.batch_size = 256;
    o.ubatch_size = 512;
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.ubatch_size = 128;
    CHECK(validate_llamacpp_options(o).ok());
}

TEST_CASE("llamacpp adapter: load_model rejects invalid context options") {
    LlamaCppBackendOptions opts;
    opts.kv_cache_type_v = "q4_0";
    opts.flash_attention = "off";
    auto be = make_llamacpp_backend(opts);
    ModelLoadOptions o;
    o.model = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    const auto r = be->load_model(o);
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(r.status().message().find("flash attention") != std::string::npos);
}
