// Actual C ABI -> Session -> native backend -> loopback mock HTTP server.
#include <doctest/doctest.h>

#include <memory>
#include <string>

#include "sonder/inference/json.hpp"
#include "sonder_inference.h"
#include "test_support.hpp"

using namespace sonder_test;
using namespace sonder::inference;

TEST_CASE("C ABI chat routes structured messages to native /api/chat and decodes UTF-8") {
    FakeOllamaServer server;
    server.set_show("qwen3:8b", fixture("show.json"));
    StreamScript script;
    script.body = R"({"message":{"role":"assistant","content":"h\u00e9llo \u2713"},"done":false})" "\n"
                  R"({"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":12,"eval_count":8})" "\n";
    script.chunk_bytes = 1;
    server.set_chat(script);
    sonder_engine_options options;
    sonder_engine_options_init(&options);
    options.telemetry_level = SONDER_TELEMETRY_OFF;
    sonder_engine* raw_engine = nullptr;
    REQUIRE(sonder_engine_create(&options, &raw_engine) == SONDER_OK);
    std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine(raw_engine, sonder_engine_destroy);
    REQUIRE(sonder_engine_register_ollama_backend(engine.get(), server.base_url().c_str()) == SONDER_OK);
    sonder_model* raw_model = nullptr;
    REQUIRE(sonder_model_load(engine.get(), "ollama", "qwen3:8b", &raw_model) == SONDER_OK);
    std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model(raw_model, sonder_model_release);
    sonder_sampling_config config;
    sonder_sampling_config_init(&config);
    config.max_tokens = 24;
    config.temperature = 0;
    sonder_session* raw_session = nullptr;
    REQUIRE(sonder_session_create(engine.get(), model.get(), &config, &raw_session) == SONDER_OK);
    std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session(raw_session, sonder_session_destroy);
    const sonder_chat_message records[] = {
        {sizeof(sonder_chat_message), "system", "Be concise"},
        {sizeof(sonder_chat_message), "user", "h\xc3\xa9llo \xe2\x9c\x93"}};
    const sonder_chat_message* messages[] = {&records[0], &records[1]};
    std::string text;
    const auto callback = [](void* user, const char* chunk, size_t length) {
        static_cast<std::string*>(user)->append(chunk, length);
        return 0;
    };
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE_MESSAGE(sonder_session_chat(session.get(), messages, 2, callback, &text, &stats) == SONDER_OK,
                    sonder_last_error_message());
    CHECK(text == "h\xc3\xa9llo \xe2\x9c\x93");
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(stats.prompt_tokens == 12);
    CHECK(stats.completion_tokens == 8);
    CHECK(stats.chunks == 1);
    CHECK(server.last_generate_body().empty());
    const auto sent = json::parse(server.last_chat_body());
    REQUIRE(sent.ok());
    CHECK(sent->find("prompt") == nullptr);
    const auto& received = sent->find("messages")->as_array();
    REQUIRE(received.size() == 2);
    CHECK(received[0].find("role")->as_string() == "system");
    CHECK(received[0].find("content")->as_string() == records[0].content);
    CHECK(received[1].find("role")->as_string() == "user");
    CHECK(received[1].find("content")->as_string() == records[1].content);
    CHECK(sent->find("options")->find("num_predict")->as_int() == 24);
    CHECK(sent->find("options")->find("temperature")->as_double() == doctest::Approx(0));

    // A provider failure crosses the ABI as FAILED without poisoning reuse.
    StreamScript failed;
    failed.status = 500;
    failed.content_type = "application/json";
    failed.body = R"({"error":"synthetic provider failure"})";
    server.set_chat(failed);
    CHECK(sonder_session_chat(session.get(), messages, 2, nullptr, nullptr, &stats) != SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_FAILED);
    CHECK_FALSE(std::string(sonder_last_error_message()).empty());
    server.set_chat(script);
    REQUIRE(sonder_session_chat(session.get(), messages, 2, nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(std::string(sonder_last_error_message()).empty());
}
