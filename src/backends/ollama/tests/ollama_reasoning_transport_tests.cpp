#include <doctest/doctest.h>

#include <string>

#include "test_support.hpp"

using namespace sonder::inference;

namespace {
std::shared_ptr<BackendModel> load_reasoning_model(sonder_test::FakeOllamaServer &server,
                                                   OllamaBackendOptions options) {
    server.set_show("qwen3:8b", sonder_test::fixture("show.json"));
    auto backend = make_ollama_backend(std::move(options));
    auto model = backend->load_model({"qwen3:8b", "cpu:0"});
    REQUIRE(model.ok());
    return model.value();
}
} // namespace

TEST_CASE("ollama opt-in reasoning is separate while legacy mode stays text-only") {
    sonder_test::FakeOllamaServer server;
    sonder_test::StreamScript script;
    script.body = sonder_test::fixture("chat_stream.ndjson");
    server.set_chat(script);
    auto model = load_reasoning_model(server, sonder_test::config_for(server));

    ChatRequest request;
    request.messages = {{"user", "hi"}};
    request.separate_reasoning = true;
    std::string text;
    std::string reasoning;
    auto result = model->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        reasoning += chunk.reasoning;
        return true;
    });
    REQUIRE(result.ok());
    CHECK(text == "Hello, Nate!");
    CHECK(reasoning == "User wants a greeting.");

    request.separate_reasoning = false;
    text.clear();
    reasoning.clear();
    result = model->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        reasoning += chunk.reasoning;
        return true;
    });
    REQUIRE(result.ok());
    CHECK(text == "Hello, Nate!");
    CHECK(reasoning.empty());
}

TEST_CASE("ollama legacy emit_thinking_chunks still flattens thinking into text") {
    sonder_test::FakeOllamaServer server;
    sonder_test::StreamScript script;
    script.body = sonder_test::fixture("chat_stream.ndjson");
    server.set_chat(script);
    auto config = sonder_test::config_for(server);
    config.emit_thinking_chunks = true;
    auto model = load_reasoning_model(server, config);
    ChatRequest request;
    request.messages = {{"user", "hi"}};
    std::string text;
    auto result = model->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        CHECK(chunk.reasoning.empty());
        return true;
    });
    REQUIRE(result.ok());
    CHECK(text == "User wants a greeting.Hello, Nate!");
}
