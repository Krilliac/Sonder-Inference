#include <doctest/doctest.h>

#include <string>

#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"

using namespace sonder::inference;

TEST_CASE("llama-server keeps opt-in reasoning separate from visible text") {
    sonder_test::FakeLlamaServer server;
    server.set_body("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n"
                    "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n");
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.native_completion = false;
    auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
    REQUIRE(model.ok());

    ChatRequest request;
    request.messages = {{"user", "hi"}};
    request.separate_reasoning = true;
    std::string text;
    std::string reasoning;
    auto result = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        reasoning += chunk.reasoning;
        return true;
    });
    REQUIRE_MESSAGE(result.ok(), result.status().to_string());
    CHECK(text == "answer");
    CHECK(reasoning == "think");
}

TEST_CASE("llama-server legacy chat ignores malformed reasoning extensions") {
    sonder_test::FakeLlamaServer server;
    server.set_body("data: {\"choices\":[{\"delta\":{\"reasoning_content\":42,\"content\":\"answer\"}}]}\n\n"
                    "data: {\"choices\":[{\"finish_reason\":\"length\"}]}\n\n"
                    "data: [DONE]\n\n");
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.native_completion = false;
    auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
    REQUIRE(model.ok());
    ChatRequest request;
    request.messages = {{"user", "hi"}};
    std::string text;
    auto result = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        CHECK(chunk.reasoning.empty());
        return true;
    });
    REQUIRE(result.ok());
    CHECK(text == "answer");
}

TEST_CASE("llama-server separated reasoning leaves text and counts unchanged and adds none when absent") {
    sonder_test::FakeLlamaServer server;
    server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"No, 91 \"}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"content\":\"is not prime.\"}}]}\n\n"
                    "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n");
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.native_completion = false;
    auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
    REQUIRE(model.ok());

    ChatRequest request;
    request.messages = {{"user", "hi"}};
    std::string text_plain;
    auto plain = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        text_plain += chunk.text;
        return true;
    });
    REQUIRE(plain.ok());

    request.separate_reasoning = true;
    std::string text;
    auto result = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        CHECK(chunk.reasoning.empty());
        return true;
    });
    REQUIRE_MESSAGE(result.ok(), result.status().to_string());
    CHECK(text == "No, 91 is not prime.");
    CHECK(text == text_plain);
    CHECK(result.value().chunks == plain.value().chunks);
    CHECK(result.value().stop_reason == plain.value().stop_reason);
}

TEST_CASE("llama-server separated reasoning accepts the reasoning key used by some builds") {
    sonder_test::FakeLlamaServer server;
    server.set_body("data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\"}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n"
                    "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n");
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.native_completion = false;
    auto model = make_llamaserver_backend(options)->load_model({"fake-model", "cpu:0"});
    REQUIRE(model.ok());
    ChatRequest request;
    request.messages = {{"user", "hi"}};
    request.separate_reasoning = true;
    std::string text;
    std::string reasoning;
    auto result = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        text += chunk.text;
        reasoning += chunk.reasoning;
        return true;
    });
    REQUIRE(result.ok());
    CHECK(text == "answer");
    CHECK(reasoning == "think");
}
