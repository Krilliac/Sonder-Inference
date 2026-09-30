// BackendModel::chat() on the Ollama backend: native /api/chat against the
// in-process fake Ollama server (recorded NDJSON fixture, no live Ollama).
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "test_support.hpp"

using namespace sonder::inference;
using sonder_test::config_for;
using sonder_test::FakeOllamaServer;
using sonder_test::fixture;
using sonder_test::StreamScript;

namespace {

std::shared_ptr<BackendModel> load(FakeOllamaServer& srv, OllamaBackendOptions options) {
    srv.set_show("qwen3:8b", fixture("show.json"));
    auto be = make_ollama_backend(std::move(options));
    auto m = be->load_model({"qwen3:8b", "cpu:0"});
    REQUIRE_MESSAGE(m.ok(), m.status().to_string());
    return m.value();
}

}  // namespace

TEST_CASE("backend chat: native /api/chat streams content and maps stats") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    s.chunk_bytes = 7;  // split NDJSON lines at awkward boundaries
    srv.set_chat(s);
    auto m = load(srv, config_for(srv));
    CHECK(m->has_native_chat());

    ChatRequest req;
    req.request_id = "chat-1";
    req.messages = {{"system", "Be friendly."}, {"user", "Say hi to Nate"}};
    req.sampling = SamplingConfig::greedy(24, 7);
    std::string text;
    std::vector<std::uint64_t> indices;
    auto st = m->chat(req, {}, [&](const TokenChunk& c) {
        text.append(c.text);
        indices.push_back(c.index);
        return true;
    });
    REQUIRE_MESSAGE(st.ok(), st.status().to_string());
    CHECK(text == "Hello, Nate!");  // thinking is not delivered by default
    CHECK(indices == std::vector<std::uint64_t>{0u, 1u});
    CHECK(st->chunks == 2u);
    CHECK(st->prompt_tokens == 12u);
    CHECK(st->completion_tokens == 8u);
    CHECK(st->prompt_eval_ns == 60000000u);
    CHECK(st->eval_ns == 100000000u);
    CHECK(st->token_counts_from_backend);
    CHECK(st->stop_reason == StopReason::end_of_sequence);

    // The request went to /api/chat (not /api/generate) with the messages and
    // sampling options intact, and no client-side prompt formatting.
    CHECK(srv.last_generate_body().empty());
    auto sent = json::parse(srv.last_chat_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("model")->as_string() == "qwen3:8b");
    CHECK(sent->find("prompt") == nullptr);
    const auto& msgs = sent->find("messages")->as_array();
    REQUIRE(msgs.size() == 2u);
    CHECK(msgs[0].find("role")->as_string() == "system");
    CHECK(msgs[0].find("content")->as_string() == "Be friendly.");
    CHECK(msgs[1].find("role")->as_string() == "user");
    CHECK(msgs[1].find("content")->as_string() == "Say hi to Nate");
    CHECK(sent->find("options")->find("num_predict")->as_int() == 24);
    CHECK(sent->find("options")->find("seed")->as_int() == 7);
    CHECK(sent->find("options")->find("temperature")->as_double() == doctest::Approx(0.0));
}

TEST_CASE("backend chat: thinking chunks when enabled") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    srv.set_chat(s);
    auto cfg = config_for(srv);
    cfg.emit_thinking_chunks = true;
    auto m = load(srv, cfg);
    ChatRequest req;
    req.messages = {{"user", "hi"}};
    std::vector<std::string> pieces;
    auto st = m->chat(req, {}, [&](const TokenChunk& c) {
        pieces.emplace_back(c.text);
        return true;
    });
    REQUIRE(st.ok());
    CHECK(pieces == std::vector<std::string>{"User wants a greeting.", "Hello", ", Nate!"});
}

TEST_CASE("backend chat: callback stop, cancellation, invalid messages, server error") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    srv.set_chat(s);
    auto m = load(srv, config_for(srv));
    ChatRequest req;
    req.messages = {{"user", "hi"}};

    auto stopped = m->chat(req, {}, [&](const TokenChunk&) { return false; });
    REQUIRE(stopped.ok());
    CHECK(stopped->stop_reason == StopReason::callback);
    CHECK(stopped->chunks == 1u);

    CancellationSource source;
    auto cancelled = m->chat(req, source.token(), [&](const TokenChunk&) {
        source.cancel();
        return true;
    });
    CHECK(cancelled.status().code() == ErrorCode::cancelled);

    // Validation happens client-side: nothing is sent for a bad conversation.
    const int before = srv.request_count();
    ChatRequest bad;
    bad.messages = {{"user", "x"}, {"assistant", "y"}};
    CHECK(m->chat(bad, {}, {}).status().code() == ErrorCode::invalid_argument);
    bad.messages.clear();
    CHECK(m->chat(bad, {}, {}).status().code() == ErrorCode::invalid_argument);
    CHECK(srv.request_count() == before);

    StreamScript err;
    err.status = 404;
    err.body = R"({"error":"model 'qwen3:8b' not found"})";
    err.content_type = "application/json";
    srv.set_chat(err);
    auto failed = m->chat(req, {}, {});
    CHECK_FALSE(failed.ok());
}

TEST_CASE("backend chat: stalled stream is cancellable") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    s.stall_after_bytes = s.body.find('\n', s.body.find("Hello")) + 1;  // after the "Hello" line
    srv.set_chat(s);
    auto m = load(srv, config_for(srv));
    ChatRequest req;
    req.messages = {{"user", "hi"}};
    CancellationSource source;
    std::string text;
    auto st = m->chat(req, source.token(), [&](const TokenChunk& c) {
        text.append(c.text);
        source.cancel();
        return true;
    });
    CHECK(st.status().code() == ErrorCode::cancelled);
    CHECK(text == "Hello");
}

TEST_CASE("backend chat: logit_bias is rejected, not silently dropped") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = fixture("chat_stream.ndjson");
    srv.set_chat(s);
    auto m = load(srv, config_for(srv));
    const int before = srv.request_count();
    ChatRequest req;
    req.messages = {{"user", "hi"}};
    req.sampling.logit_bias = {{42, 5.0f}};
    auto st = m->chat(req, {}, {});
    REQUIRE_FALSE(st.ok());
    CHECK(st.status().code() == ErrorCode::invalid_argument);
    CHECK(st.status().message().find("logit_bias") != std::string::npos);
    CHECK(srv.request_count() == before);  // nothing was sent
}

TEST_CASE("backend chat: thinking control becomes think and cached prompt tokens reach the stats") {
    FakeOllamaServer srv;
    StreamScript s;
    s.body = R"({"model":"qwen3:8b","message":{"role":"assistant","content":"ok"},"done":false})" "\n"
             R"({"model":"qwen3:8b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",)"
             R"("prompt_eval_count":40,"prompt_eval_cached_count":32,"eval_count":1})" "\n";
    srv.set_chat(s);
    auto m = load(srv, config_for(srv));
    ChatRequest req;
    req.messages = {{"user", "hi"}};
    auto st = m->chat(req, {}, {});
    REQUIRE_MESSAGE(st.ok(), st.status().to_string());
    CHECK(st->cached_tokens == std::optional<std::uint64_t>(32));
    auto sent = json::parse(srv.last_chat_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("think") == nullptr);  // unset: the model default applies

    req.thinking.enable_thinking = false;
    req.thinking.reasoning_effort = "high";  // no portable Ollama field: not sent
    REQUIRE(m->chat(req, {}, {}).ok());
    sent = json::parse(srv.last_chat_body());
    REQUIRE(sent.ok());
    REQUIRE(sent->find("think") != nullptr);
    CHECK_FALSE(sent->find("think")->as_bool());
    CHECK(sent->find("reasoning_effort") == nullptr);

    // A server that omits the counter reports no cached count (not 0).
    s.body = R"({"model":"qwen3:8b","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop",)"
             R"("prompt_eval_count":40,"eval_count":1})" "\n";
    srv.set_chat(s);
    auto plain = m->chat(req, {}, {});
    REQUIRE(plain.ok());
    CHECK_FALSE(plain->cached_tokens.has_value());
}
