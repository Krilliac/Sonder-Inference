// Backend-interface adapter tests against the fake Ollama server.
#include <doctest/doctest.h>

#include <string>

#include "test_support.hpp"

using namespace sonder::inference;
using namespace sonder::inference::ollama;
using sonder_test::config_for;
using sonder_test::FakeOllamaServer;
using sonder_test::fixture;
using sonder_test::StreamScript;

TEST_CASE("backend: capabilities, probe, list_models") {
    FakeOllamaServer srv;
    srv.set_tags(fixture("tags.json"));
    auto be = make_ollama_backend(config_for(srv));
    CHECK(be->name() == "ollama");
    CHECK(be->capabilities().has(Capability::streaming));
    CHECK(be->capabilities().has(Capability::remote_process));
    CHECK_FALSE(be->capabilities().has(Capability::kv_export));
    auto v = be->probe();
    REQUIRE_MESSAGE(v.ok(), v.status().to_string());
    auto models = be->list_models();
    REQUIRE(models.ok());
    REQUIRE(models->size() == 2u);
    CHECK(models->at(0).backend == "ollama");
    CHECK(models->at(0).quantization == "Q4_K_M");
}

TEST_CASE("backend: load_model uses /api/show metadata; missing model fails") {
    FakeOllamaServer srv;
    srv.set_tags(fixture("tags.json"));
    srv.set_show("llama3.2:3b", fixture("show.json"));
    auto be = make_ollama_backend(config_for(srv));
    auto m = be->load_model({"llama3.2:3b", "cpu:0"});
    REQUIRE_MESSAGE(m.ok(), m.status().to_string());
    const auto& d = m.value()->descriptor();
    CHECK(d.name == "llama3.2:3b");
    CHECK(d.family == "llama");
    CHECK(d.quantization == "Q4_K_M");
    CHECK(d.context_length == 131072u);
    CHECK(d.size_bytes == 2019393189u);
    CHECK(be->load_model({"absent:1b", "cpu:0"}).status().code() == ErrorCode::not_found);
    CHECK(be->load_model({"", "cpu:0"}).status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("backend: generate maps chunks and GenerateStats") {
    FakeOllamaServer srv;
    srv.set_show("llama3.2:3b", fixture("show.json"));
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    s.chunk_bytes = 5;
    srv.set_generate(s);
    auto be = make_ollama_backend(config_for(srv));
    auto m = be->load_model({"llama3.2:3b", "cpu:0"});
    REQUIRE(m.ok());

    GenerateRequest req;
    req.request_id = "req-1";
    req.prompt = "Why is the sky blue?";
    req.sampling = SamplingConfig::greedy(32, 1);
    std::string text;
    std::uint64_t last_index = 0;
    auto st = m.value()->generate(req, {}, [&](const TokenChunk& c) {
        text.append(c.text);
        last_index = c.index;
        return true;
    });
    REQUIRE_MESSAGE(st.ok(), st.status().to_string());
    CHECK(text == "The sky is blue.");
    CHECK(last_index == 3u);
    CHECK(st->chunks == 4u);
    CHECK(st->prompt_tokens == 26u);
    CHECK(st->completion_tokens == 4u);
    CHECK(st->load_ns == 150000000u);
    CHECK(st->prompt_eval_ns == 130000000u);
    CHECK(st->eval_ns == 40000000u);
    CHECK(st->token_counts_from_backend);
    CHECK(st->stop_reason == StopReason::end_of_sequence);
    auto sent = json::parse(srv.last_generate_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("options")->find("num_predict")->as_int() == 32);
    CHECK(sent->find("options")->find("temperature")->as_double() == doctest::Approx(0.0));
}

TEST_CASE("backend: callback stop and cancellation") {
    FakeOllamaServer srv;
    srv.set_show("llama3.2:3b", fixture("show.json"));
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    srv.set_generate(s);
    auto be = make_ollama_backend(config_for(srv));
    auto m = be->load_model({"llama3.2:3b", "cpu:0"});
    REQUIRE(m.ok());
    GenerateRequest req;
    req.prompt = "x";
    int n = 0;
    auto st = m.value()->generate(req, {}, [&](const TokenChunk&) { return ++n < 2; });
    REQUIRE(st.ok());
    CHECK(st->stop_reason == StopReason::callback);
    CHECK(st->chunks == 2u);

    CancellationSource source;
    auto cancelled = m.value()->generate(req, source.token(), [&](const TokenChunk&) {
        source.cancel();
        return true;
    });
    CHECK(cancelled.status().code() == ErrorCode::cancelled);
}
