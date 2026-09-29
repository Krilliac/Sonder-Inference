// Recorded-fixture tests for the NDJSON stream decoder (no network).
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "backends/ollama/ollama_protocol.hpp"
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

TEST_CASE("request bodies: default sampling options are unchanged") {
    const json::Object opts = sampling_to_options(SamplingConfig{});
    CHECK(opts.size() == 6u);  // temperature, top_p, top_k, min_p, repeat_penalty, num_predict
    CHECK_FALSE(opts.contains("num_ctx"));
    CHECK_FALSE(opts.contains("repeat_last_n"));
    CHECK_FALSE(opts.contains("presence_penalty"));
    CHECK_FALSE(opts.contains("frequency_penalty"));
    CHECK_FALSE(opts.contains("typical_p"));
    CHECK_FALSE(opts.contains("logit_bias"));
    CHECK_FALSE(opts.contains("seed"));
    CHECK(json::Value(sampling_to_options(SamplingConfig::greedy(8, 1))).dump() ==
          R"({"temperature":0.0,"top_p":1.0,"top_k":0,"min_p":0.0,"repeat_penalty":1.0,"seed":1,"num_predict":8})");
}

TEST_CASE("request bodies: num_ctx and penalties reach Ollama options") {
    SamplingConfig s = SamplingConfig::greedy(32, 3);
    s.num_ctx = 8192;
    s.repeat_last_n = 128;
    s.presence_penalty = 0.5f;
    s.frequency_penalty = -0.25f;
    s.typical_p = 0.9f;
    s.repeat_penalty = 1.15f;
    s.logit_bias = {{42, 5.0f}};
    const json::Object opts = sampling_to_options(s);
    REQUIRE(opts.contains("num_ctx"));
    CHECK(opts.find("num_ctx")->as_int() == 8192);
    CHECK(opts.find("repeat_last_n")->as_int() == 128);
    CHECK(opts.find("presence_penalty")->as_double() == doctest::Approx(0.5));
    CHECK(opts.find("frequency_penalty")->as_double() == doctest::Approx(-0.25));
    // Ollama 0.34.1+ rejects any request that sets typical_p, so it is never sent.
    CHECK_FALSE(opts.contains("typical_p"));
    CHECK(opts.find("repeat_penalty")->as_double() == doctest::Approx(1.15));
    CHECK_FALSE(opts.contains("logit_bias"));  // no Ollama equivalent

    GenerateParams g;
    g.model = "m";
    g.prompt = "p";
    g.options = opts;
    const json::Value body = build_generate_body(g);
    CHECK(body.find("options")->find("num_ctx")->as_int() == 8192);

    // repeat_last_n = 0 (window off) differs from the default and is sent.
    s = SamplingConfig{};
    s.repeat_last_n = 0;
    CHECK(sampling_to_options(s).find("repeat_last_n")->as_int() == 0);
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

TEST_CASE("request bodies: explicit_only sends only the fields the caller set") {
    SamplingConfig s;  // struct defaults: temperature 0.8, repeat_penalty 1.1, ...
    s.explicit_only = true;
    s.max_tokens = 16;
    json::Object opts = sampling_to_options(s);
    // Nothing set explicitly: the model's own (GGUF) defaults apply.
    CHECK_FALSE(opts.contains("temperature"));
    CHECK_FALSE(opts.contains("top_p"));
    CHECK_FALSE(opts.contains("top_k"));
    CHECK_FALSE(opts.contains("min_p"));
    CHECK_FALSE(opts.contains("repeat_penalty"));
    CHECK(opts.find("num_predict")->as_int() == 16);

    s.temperature = 0.8f;  // equal to the struct default, but explicitly set
    s.repeat_last_n = 64;  // equal to the default window, but explicitly set
    s.explicit_fields = SamplingConfig::kTemperature | SamplingConfig::kRepeatLastN;
    opts = sampling_to_options(s);
    CHECK(opts.find("temperature")->as_double() == doctest::Approx(0.8));
    CHECK(opts.find("repeat_last_n")->as_int() == 64);
    CHECK_FALSE(opts.contains("repeat_penalty"));
    CHECK_FALSE(opts.contains("presence_penalty"));
}

TEST_CASE("decoder: prompt_eval_cached_count is parsed and reported") {
    OllamaTimings t;
    apply_server_timings(json::parse(R"({"prompt_eval_count":35,"prompt_eval_cached_count":34,"eval_count":4})").value(),
                         t);
    CHECK(t.has_server_timings);
    CHECK(t.prompt_eval_count == 35);
    CHECK(t.prompt_eval_cached_count == 34);
    CHECK(timing_attributes(t).find("prompt_eval_cached_count")->as_int() == 34);

    OllamaTimings old;  // servers before 0.33.3 omit the field
    apply_server_timings(json::parse(R"({"prompt_eval_count":35})").value(), old);
    CHECK(old.prompt_eval_cached_count == 0);
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
    CHECK(lines[0].find("backend.model.load.reported") != std::string::npos);
    CHECK(lines[1].find("backend.timing.prefill") != std::string::npos);
    CHECK(lines[2].find("backend.timing.decode") != std::string::npos);
    CHECK(lines[2].find("req-1") != std::string::npos);
}

// Regression tests for docs/integration/hardening.md B6 and the line cap.
// Repro inputs also live in fuzz/corpus/ollama_stream.

TEST_CASE("parse_stream_line agrees with StreamDecoder (B6)") {
    struct Case {
        const char* line;
        bool ok;
        ErrorCode code;
    };
    const Case cases[] = {
        {"[1,2]", false, ErrorCode::protocol_error},
        {"\"text\"", false, ErrorCode::protocol_error},
        {"42", false, ErrorCode::protocol_error},
        {"{\"error\":null,\"response\":\"a\"}", true, ErrorCode::ok},
        {"{\"error\":\"boom\"}", false, ErrorCode::backend_error},
        {"{\"response\":\"a\"", false, ErrorCode::protocol_error},
    };
    for (const auto& c : cases) {
        CAPTURE(c.line);
        GenerateStats stats;
        auto r = parse_stream_line(c.line, stats);
        CHECK(r.ok() == c.ok);
        StreamDecoder d(StreamKind::generate, {});
        CHECK(d.feed(std::string(c.line) + "\n") == c.ok);
        if (!c.ok) {
            CHECK(r.status().code() == c.code);
            CHECK(d.error().code() == c.code);
        }
    }
    GenerateStats stats;
    auto r = parse_stream_line("{\"error\":null,\"response\":\"a\"}", stats);
    REQUIRE(r.ok());
    CHECK(r.value().piece == "a");
}

TEST_CASE("decoder: lines longer than the cap are rejected") {
    StreamDecoder d(StreamKind::generate, {});
    const std::string chunk(1u << 20, ' ');
    bool ok = true;
    for (std::size_t sent = 0; ok && sent <= kMaxNdjsonLineBytes; sent += chunk.size()) {
        ok = d.feed(chunk);
    }
    CHECK_FALSE(ok);
    CHECK(d.error().code() == ErrorCode::protocol_error);
    CHECK(d.error().message().find("4 MiB") != std::string::npos);

    // A large but legal final chunk (big "context" array) still decodes.
    std::string line = "{\"response\":\"\",\"done\":true,\"context\":[";
    for (int i = 0; i < 150000; ++i) line += (i ? ",123456" : "123456");
    line += "]}\n";
    REQUIRE(line.size() < kMaxNdjsonLineBytes);
    StreamDecoder big(StreamKind::generate, {});
    CHECK(big.feed(line));
    CHECK(big.finish().ok());
}
