// OllamaClient against an in-process fake Ollama server (no live Ollama).
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "test_support.hpp"

using namespace sonder::inference;
using namespace sonder::inference::ollama;
using sonder_test::config_for;
using sonder_test::FakeOllamaServer;
using sonder_test::fixture;
using sonder_test::StreamScript;

TEST_CASE("client: generate streams chunked NDJSON and reports timings") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    s.chunk_bytes = 7;  // lines split at arbitrary byte boundaries
    srv.set_generate(s);
    OllamaClient cli(config_for(srv));

    GenerateParams p;
    p.model = "llama3.2:3b";
    p.prompt = "Why is the sky blue?";
    p.options = sampling_to_options(SamplingConfig::greedy(64, 42));
    std::string streamed;
    auto r = cli.generate(p, [&](const StreamChunk& c) {
        streamed.append(c.content);
        return true;
    });
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r->http_status == 200);
    CHECK(r->text == "The sky is blue.");
    CHECK(streamed == r->text);
    CHECK(r->timings.eval_count == 4);
    CHECK(r->timings.decode_tokens_per_sec() == doctest::Approx(100.0));
    CHECK(r->timings.ttfb_ms >= 0.0);
    CHECK(r->timings.ttft_ms >= 0.0);
    CHECK(r->timings.wall_ms >= r->timings.ttft_ms);

    auto sent = json::parse(srv.last_generate_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("model")->as_string() == "llama3.2:3b");
    CHECK(sent->find("prompt")->as_string() == "Why is the sky blue?");
    CHECK(sent->find("options")->find("seed")->as_int() == 42);
    CHECK(sent->find("options")->find("num_predict")->as_int() == 64);
}

TEST_CASE("client: chat streams content and thinking") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    s.chunk_bytes = 1;
    srv.set_chat(s);
    OllamaClient cli(config_for(srv));
    ChatParams p;
    p.model = "qwen3:8b";
    p.messages = {{"user", "Say hi"}};
    p.think = true;
    auto r = cli.chat(p);
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r->text == "Hello, Nate!");
    CHECK(r->thinking == "User wants a greeting.");
    auto sent = json::parse(srv.last_chat_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("messages")->as_array().size() == 1u);
    CHECK(sent->find("think")->as_bool());
}

TEST_CASE("client: HTTP errors map to status codes") {
    FakeOllamaServer srv;
    OllamaClient cli(config_for(srv));
    GenerateParams p;
    p.model = "nope";

    StreamScript s404;
    s404.status = 404;
    s404.body = R"({"error":"model \"nope\" not found, try pulling it first"})";
    s404.content_type = "application/json";
    srv.set_generate(s404);
    auto r = cli.generate(p);
    CHECK(r.status().code() == ErrorCode::not_found);
    CHECK(r.status().message().find("not found") != std::string::npos);

    StreamScript s500;
    s500.status = 500;
    s500.body = "internal failure\n";
    s500.content_type = "text/plain";
    srv.set_generate(s500);
    r = cli.generate(p);
    CHECK(r.status().code() == ErrorCode::backend_error);
    CHECK(r.status().message().find("internal failure") != std::string::npos);
}

TEST_CASE("client: mid-stream error and truncation") {
    FakeOllamaServer srv;
    OllamaClient cli(config_for(srv));
    GenerateParams p;
    p.model = "m";
    StreamScript s;
    s.body = fixture("generate_error_midstream.ndjson");
    srv.set_generate(s);
    CHECK(cli.generate(p).status().code() == ErrorCode::backend_error);
    s.body = fixture("generate_truncated.ndjson");
    srv.set_generate(s);
    CHECK(cli.generate(p).status().code() == ErrorCode::protocol_error);
}

TEST_CASE("client: callback stop returns partial result") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    srv.set_generate(s);
    OllamaClient cli(config_for(srv));
    GenerateParams p;
    p.model = "m";
    int seen = 0;
    auto r = cli.generate(p, [&](const StreamChunk&) { return ++seen < 2; });
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r->stopped_by_callback);
    CHECK(r->text == "The sky");
}

TEST_CASE("client: cancellation from another thread aborts a stalled stream") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    s.stall_after_bytes = s.body.find('\n') + 1;  // one line, then stall
    s.stall_max = std::chrono::milliseconds(5000);
    srv.set_generate(s);
    OllamaClient cli(config_for(srv));
    GenerateParams p;
    p.model = "m";
    CancellationSource source;
    std::atomic<bool> first{false};
    std::thread canceller([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!first && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        source.cancel();
    });
    const auto t0 = std::chrono::steady_clock::now();
    auto r = cli.generate(
        p,
        [&](const StreamChunk&) {
            first = true;
            return true;
        },
        source.token());
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    canceller.join();
    CHECK(r.status().code() == ErrorCode::cancelled);
    CHECK(elapsed < std::chrono::milliseconds(2500));
}

TEST_CASE("client: cancellation during upstream prefill closes the response") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("generate_stream.ndjson");
    s.chunk_bytes = 1;
    s.initial_delay = std::chrono::milliseconds(300);
    srv.set_generate(s);
    OllamaClient cli(config_for(srv));
    GenerateParams p;
    p.model = "m";
    CancellationSource source;
    std::thread canceller([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (srv.request_count() == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        source.cancel();
    });
    auto r = cli.generate(p, {}, source.token());
    canceller.join();
    REQUIRE(srv.request_count() > 0);
    REQUIRE(r.status().code() == ErrorCode::cancelled);
    // The provider wakes after its simulated prefill and must observe that
    // the client's socket was closed by the cancelled HTTP request.
    for (int i = 0; i < 100 && srv.disconnects() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(srv.disconnects() > 0);
}

TEST_CASE("client: pre-cancelled token never sends a request") {
    FakeOllamaServer srv;
    OllamaClient cli(config_for(srv));
    CancellationSource source;
    source.cancel();
    GenerateParams p;
    p.model = "m";
    CHECK(cli.generate(p, {}, source.token()).status().code() == ErrorCode::cancelled);
    CHECK(srv.request_count() == 0);
}

TEST_CASE("client: version, tags, ps, show") {
    FakeOllamaServer srv;
    srv.set_tags(fixture("tags.json"));
    srv.set_show("llama3.2:3b", fixture("show.json"));
    OllamaClient cli(config_for(srv));

    auto v = cli.version();
    REQUIRE_MESSAGE(v.ok(), v.status().to_string());
    CHECK(v.value() == "0.12.0-fake");

    auto models = cli.list_models();
    REQUIRE(models.ok());
    REQUIRE(models->size() == 2u);
    CHECK(models->at(0).name == "llama3.2:3b");
    CHECK(models->at(0).quantization_level == "Q4_K_M");
    CHECK(models->at(0).parameter_size == "3.2B");
    CHECK(models->at(0).size_bytes == 2019393189u);
    CHECK(models->at(1).family == "nomic-bert");

    auto ps = cli.running_models();
    REQUIRE(ps.ok());
    REQUIRE(ps->size() == 1u);
    CHECK(ps->at(0).size_vram_bytes == 3400000000u);

    auto show = cli.show_model("llama3.2:3b");
    REQUIRE(show.ok());
    CHECK(show->find("details")->find("family")->as_string() == "llama");
    auto missing = cli.show_model("absent:1b");
    CHECK(missing.status().code() == ErrorCode::not_found);
    CHECK(missing.status().message().find("model not found") != std::string::npos);
}

TEST_CASE("client: connection refused -> unavailable") {
    int port = 0;
    {
        FakeOllamaServer tmp;
        port = tmp.port();
    }
    OllamaConfig c;
    c.base_url = "http://127.0.0.1:" + std::to_string(port);
    c.connect_timeout = std::chrono::milliseconds(500);
    OllamaClient cli(c);
    GenerateParams p;
    p.model = "m";
    auto r = cli.generate(p);
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() != ErrorCode::ok);
    CHECK_FALSE(cli.list_models().ok());
}

TEST_CASE("client: policy refuses plain HTTP to remote hosts and https") {
    OllamaConfig c;
    c.base_url = "http://10.77.0.2:11434";
    OllamaClient remote(c);
    CHECK(remote.version().status().code() == ErrorCode::invalid_argument);
    c.base_url = "https://10.77.0.2:8443";
    OllamaClient tls(c);
#if defined(SONDER_HAS_TLS)
    // TLS builds accept https:// but still refuse non-loopback hosts without
    // allow_remote, before any connection attempt.
    CHECK(tls.version().status().code() == ErrorCode::invalid_argument);
#else
    CHECK(tls.version().status().code() == ErrorCode::unsupported);
#endif
}
