#include <doctest/doctest.h>

#include <cstdlib>

#include "../ollama_protocol.hpp"
#include "net/http_client.hpp"
#include "sonder/inference.hpp"
#include "sonder/inference/backends/ollama.hpp"

using namespace sonder::inference;

TEST_SUITE("ollama_adapter") {
TEST_CASE("parses base URLs") {
    auto u = net::parse_url("http://127.0.0.1:11434");
    REQUIRE(u.ok());
    CHECK(u.value().host == "127.0.0.1");
    CHECK(u.value().port == 11434);
    CHECK(u.value().path == "/");
    auto v = net::parse_url("http://localhost/ollama/");
    REQUIRE(v.ok());
    CHECK(v.value().port == 80);
    CHECK(v.value().path == "/ollama");
    auto w = net::parse_url("http://[::1]:8080");
    REQUIRE(w.ok());
    CHECK(w.value().host == "::1");
#if defined(SONDER_HAS_TLS)
    CHECK(net::parse_url("https://127.0.0.1").value().port == 443);
#else
    CHECK(net::parse_url("https://127.0.0.1").status().code() == ErrorCode::unsupported);
#endif
    CHECK_FALSE(net::parse_url("127.0.0.1:11434").ok());
    CHECK_FALSE(net::parse_url("http://host:99999").ok());
    CHECK(net::is_loopback_host("127.0.0.1"));
    CHECK(net::is_loopback_host("localhost"));
    CHECK_FALSE(net::is_loopback_host("10.77.0.2"));
}

TEST_CASE("refuses non-loopback hosts without touching the network") {
    OllamaBackendOptions o;
    o.base_url = "http://10.77.0.2:11434";
    auto b = make_ollama_backend(o);
    CHECK(b->probe().status().code() == ErrorCode::invalid_argument);
    CHECK(b->capabilities().has(Capability::remote_process));
    CHECK_FALSE(b->capabilities().has(Capability::deterministic));
}

TEST_CASE("maps sampling to the generate request body") {
    GenerateRequest req;
    req.prompt = "hi";
    req.sampling = SamplingConfig::greedy(32, 42);
    req.sampling.stop = {"\n\n"};
    const auto body = json::Value(ollama::build_generate_body("qwen3:4b", req, "5m"));
    CHECK(body.find("model")->as_string() == "qwen3:4b");
    CHECK(body.find("stream")->as_bool() == true);
    CHECK(body.find("keep_alive")->as_string() == "5m");
    const auto* opts = body.find("options");
    REQUIRE(opts);
    CHECK(opts->find("num_predict")->as_int() == 32);
    CHECK(opts->find("temperature")->as_double() == doctest::Approx(0.0));
    CHECK(opts->find("seed")->as_int() == 42);
    CHECK(opts->find("stop")->as_array().at(0).as_string() == "\n\n");
}

TEST_CASE("parses streaming NDJSON lines") {
    GenerateStats stats;
    auto a = ollama::parse_stream_line(R"({"model":"m","response":"Hel","done":false})", stats);
    REQUIRE(a.ok());
    CHECK(a.value().piece == "Hel");
    CHECK_FALSE(a.value().done);
    auto done = ollama::parse_stream_line(
        R"({"model":"m","response":"","done":true,"done_reason":"length","prompt_eval_count":12,)"
        R"("eval_count":32,"load_duration":1000,"prompt_eval_duration":2000000,"eval_duration":400000000})",
        stats);
    REQUIRE(done.ok());
    CHECK(done.value().done);
    CHECK(stats.prompt_tokens == 12);
    CHECK(stats.completion_tokens == 32);
    CHECK(stats.eval_ns == 400000000u);
    CHECK(stats.token_counts_from_backend);
    CHECK(stats.stop_reason == StopReason::max_tokens);
    auto err = ollama::parse_stream_line(R"({"error":"model not found"})", stats);
    CHECK(err.status().code() == ErrorCode::backend_error);
    CHECK(ollama::parse_stream_line("{not json", stats).status().code() == ErrorCode::protocol_error);
    CHECK(ollama::parse_stream_line("   ", stats).ok());
}

TEST_CASE("decodes chunked transfer encoding across arbitrary splits") {
    const std::string wire = "4\r\nWiki\r\n6;ext=1\r\npedia \r\nE\r\nin \r\n\r\nchunks.\r\n0\r\n\r\n";
    for (std::size_t split = 1; split < wire.size(); ++split) {
        net::ChunkedDecoder d;
        std::string out;
        auto sink = [&](std::string_view s) {
            out.append(s);
            return true;
        };
        REQUIRE(d.feed(std::string_view(wire).substr(0, split), sink).ok());
        REQUIRE(d.feed(std::string_view(wire).substr(split), sink).ok());
        CHECK(d.done());
        CHECK(out == "Wikipedia in \r\n\r\nchunks.");
    }
    net::ChunkedDecoder bad;
    CHECK_FALSE(bad.feed("zz\r\n", [](std::string_view) { return true; }).ok());
}

TEST_CASE("splits NDJSON lines across reads") {
    net::LineSplitter s;
    std::vector<std::string> lines;
    auto on = [&](std::string_view l) {
        lines.emplace_back(l);
        return true;
    };
    s.feed("{\"a\":1}\r\n{\"b\"", on);
    s.feed(":2}\n{\"c\":3}", on);
    s.finish(on);
    REQUIRE(lines.size() == 3);
    CHECK(lines[0] == "{\"a\":1}");
    CHECK(lines[1] == "{\"b\":2}");
    CHECK(lines[2] == "{\"c\":3}");
}

TEST_CASE("maps /api/tags entries to descriptors") {
    auto v = json::parse(
        R"({"name":"qwen3:4b","size":2500000000,"details":{"format":"gguf","family":"qwen3",)"
        R"("parameter_size":"4.0B","quantization_level":"Q4_K_M"}})");
    REQUIRE(v.ok());
    const auto d = ollama::descriptor_from_tag(v.value());
    CHECK(d.name == "qwen3:4b");
    CHECK(d.format == "gguf");
    CHECK(d.quantization == "Q4_K_M");
    CHECK(d.size_bytes == 2500000000u);
    CHECK(d.backend == "ollama");
}

TEST_CASE("unreachable server reports unavailable, not a crash") {
    OllamaBackendOptions o;
    o.base_url = "http://127.0.0.1:9";  // discard port; nothing listens here in CI
    o.connect_timeout = std::chrono::milliseconds(1500);
    auto b = make_ollama_backend(o);
    auto p = b->probe();
    REQUIRE_FALSE(p.ok());
    CHECK(p.status().code() == ErrorCode::unavailable);
}

// Optional live check: set SONDER_TEST_OLLAMA_MODEL (e.g. qwen3:0.6b) to run
// against a local Ollama at SONDER_TEST_OLLAMA_URL or the default URL.
TEST_CASE("live ollama generate (opt-in)") {
    const char* model = std::getenv("SONDER_TEST_OLLAMA_MODEL");
    if (model == nullptr || model[0] == '\0') {
        MESSAGE("SONDER_TEST_OLLAMA_MODEL not set; live Ollama check skipped");
        return;
    }
    OllamaBackendOptions o;
    if (const char* url = std::getenv("SONDER_TEST_OLLAMA_URL")) o.base_url = url;
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(make_ollama_backend(o)).ok());
    ModelLoadOptions lo;
    lo.model = model;
    auto m = engine.load_model("ollama", lo);
    REQUIRE_MESSAGE(m.ok(), m.status().to_string());
    SessionOptions so;
    so.sampling = SamplingConfig::greedy(16);
    auto s = engine.create_session(m.value(), so);
    REQUIRE(s.ok());
    auto r = s.value()->generate("Reply with one word: ready");
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK(r.value().stats.completion_tokens > 0);
}
}
