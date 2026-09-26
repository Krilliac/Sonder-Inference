// Recorded-fixture tests for the NDJSON stream decoder (no network).
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "test_support.hpp"

using namespace sonder::inference;
using namespace sonder::inference::ollama;
using sonder_test::fixture;

TEST_CASE("decoder: generate fixture fed byte-by-byte") {
    const std::string body = fixture("generate_stream.ndjson");
    std::string streamed;
    int chunks = 0;
    StreamDecoder d(StreamKind::generate, [&](const StreamChunk& c) {
        streamed.append(c.content);
        ++chunks;
        return true;
    });
    for (char ch : body) {
        REQUIRE(d.feed(std::string_view(&ch, 1)));
    }
    REQUIRE(d.finish().ok());
    const auto& r = d.result();
    CHECK(r.text == "The sky is blue.");
    CHECK(streamed == r.text);
    CHECK(chunks == 5);
    CHECK(r.model == "llama3.2:3b");
    CHECK(r.done);
    CHECK(r.done_reason == "stop");
    CHECK(r.timings.has_server_timings);
    CHECK(r.timings.total_duration_ns == 1250000000);
    CHECK(r.timings.load_duration_ns == 150000000);
    CHECK(r.timings.prompt_eval_count == 26);
    CHECK(r.timings.prompt_eval_duration_ns == 130000000);
    CHECK(r.timings.eval_count == 4);
    CHECK(r.timings.eval_duration_ns == 40000000);
    CHECK(r.timings.decode_tokens_per_sec() == doctest::Approx(100.0));
    CHECK(r.timings.prompt_tokens_per_sec() == doctest::Approx(200.0));
    CHECK(r.timings.content_chunks == 4);
    CHECK(r.timings.ttft_ms >= 0.0);
}

TEST_CASE("decoder: chat fixture with thinking") {
    StreamDecoder d(StreamKind::chat, {});
    REQUIRE(d.feed(fixture("chat_stream.ndjson")));
    REQUIRE(d.finish().ok());
    CHECK(d.result().text == "Hello, Nate!");
    CHECK(d.result().thinking == "User wants a greeting.");
    CHECK(d.result().timings.eval_count == 8);
    CHECK(d.result().timings.load_duration_ns == 0);
}

TEST_CASE("decoder: CRLF, blank lines, missing final newline") {
    StreamDecoder d(StreamKind::generate, {});
    REQUIRE(d.feed("{\"response\":\"a\",\"done\":false}\r\n\r\n  \n"));
    REQUIRE(d.feed("{\"response\":\"b\",\"done\":true,\"eval_count\":2,\"eval_duration\":1000}"));
    REQUIRE(d.finish().ok());
    CHECK(d.result().text == "ab");
    CHECK(d.result().timings.eval_count == 2);
}

TEST_CASE("decoder: mid-stream error line -> backend_error") {
    StreamDecoder d(StreamKind::generate, {});
    CHECK_FALSE(d.feed(fixture("generate_error_midstream.ndjson")));
    Status st = d.finish();
    CHECK(st.code() == ErrorCode::backend_error);
    CHECK(st.message().find("unexpected EOF") != std::string::npos);
    CHECK(d.result().text == "Partial");
}

TEST_CASE("decoder: truncated stream -> protocol_error") {
    StreamDecoder d(StreamKind::generate, {});
    REQUIRE(d.feed(fixture("generate_truncated.ndjson")));
    CHECK(d.finish().code() == ErrorCode::protocol_error);
    CHECK(d.result().text == "Only part");
}

TEST_CASE("decoder: malformed JSON -> protocol_error") {
    StreamDecoder d(StreamKind::generate, {});
    CHECK_FALSE(d.feed("{\"response\":\"a\"\n"));
    CHECK(d.finish().code() == ErrorCode::protocol_error);
}

TEST_CASE("decoder: callback stop") {
    int seen = 0;
    StreamDecoder d(StreamKind::generate, [&](const StreamChunk&) { return ++seen < 2; });
    CHECK_FALSE(d.feed(fixture("generate_stream.ndjson")));
    CHECK(d.finish().ok());
    CHECK(d.result().stopped_by_callback);
    CHECK(d.result().text == "The sky");
}

TEST_CASE("request bodies: sampling mapping and options") {
    SamplingConfig s = SamplingConfig::greedy(64, 7);
    s.stop = {"</s>"};
    json::Object opts = sampling_to_options(s);
    CHECK(opts.find("num_predict")->as_int() == 64);
    CHECK(opts.find("seed")->as_int() == 7);
    CHECK(opts.find("temperature")->as_double() == doctest::Approx(s.temperature));
    CHECK(opts.find("stop")->as_array().size() == 1u);

    GenerateParams g;
    g.model = "m";
    g.prompt = "p";
    g.system = "sys";
    g.raw = true;
    g.options = opts;
    OllamaConfig cfg;
    cfg.keep_alive = "5m";
    json::Value b = build_generate_body(g, cfg);
    CHECK(b.find("model")->as_string() == "m");
    CHECK(b.find("stream")->as_bool());
    CHECK(b.find("system")->as_string() == "sys");
    CHECK(b.find("raw")->as_bool());
    CHECK(b.find("keep_alive")->as_string() == "5m");
    CHECK(b.find("options")->find("num_predict")->as_int() == 64);

    GenerateParams bare;
    bare.model = "m";
    json::Value bb = build_generate_body(bare);
    CHECK(bb.find("options") == nullptr);
    CHECK(bb.find("keep_alive") == nullptr);
    CHECK(bb.find("system") == nullptr);

    ChatParams c;
    c.model = "m";
    c.messages = {{"system", "Be brief."}, {"user", "hi"}};
    c.think = false;
    json::Value cb = build_chat_body(c);
    CHECK(cb.find("messages")->as_array().size() == 2u);
    CHECK(cb.find("think")->as_bool(true) == false);
}

TEST_CASE("telemetry: timing attributes only include server-reported fields") {
    OllamaTimings t;
    t.wall_ms = 12.5;
    json::Object a = timing_attributes(t);
    CHECK(a.find("eval_count") == nullptr);
    CHECK(a.find("ttft_ms") == nullptr);
    CHECK(a.find("wall_ms")->as_double() == doctest::Approx(12.5));

    t.has_server_timings = true;
    t.eval_count = 20;
    t.eval_duration_ns = 400000000;
    t.ttft_ms = 30.0;
    a = timing_attributes(t);
    CHECK(a.find("eval_count")->as_int() == 20);
    CHECK(a.find("decode_tokens_per_sec")->as_double() == doctest::Approx(50.0));
    CHECK(a.find("ttft_ms")->as_double() == doctest::Approx(30.0));
}

TEST_CASE("telemetry: emit_timing_events maps load/prefill/decode") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    TelemetryBus bus;
    bus.add_sink(sink);
    TelemetryContext ctx;
    ctx.session_id = "sess-1";
    ctx.request_id = "req-1";

    OllamaTimings t;
    t.has_server_timings = true;
    t.load_duration_ns = 5;
    t.prompt_eval_count = 10;
    t.prompt_eval_duration_ns = 100;
    t.eval_count = 4;
    t.eval_duration_ns = 40;
    CHECK(emit_timing_events(bus, ctx, t, "llama3.2:3b") == 3);
    t.load_duration_ns = 0;
    CHECK(emit_timing_events(bus, ctx, t, "llama3.2:3b") == 2);
    CHECK(emit_timing_events(bus, ctx, OllamaTimings{}, "x") == 0);
    bus.flush();
    auto lines = sink->lines();
    REQUIRE(lines.size() == 5u);
    CHECK(lines[0].find("model.load.completed") != std::string::npos);
    CHECK(lines[1].find("inference.prefill.completed") != std::string::npos);
    CHECK(lines[2].find("inference.decode.completed") != std::string::npos);
    CHECK(lines[2].find("req-1") != std::string::npos);
}
