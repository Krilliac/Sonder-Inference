#include <doctest/doctest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "sonder_inference.h"

namespace {
struct ChatFixture {
    std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine{nullptr, sonder_engine_destroy};
    std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model{nullptr, sonder_model_release};
    std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session{nullptr, sonder_session_destroy};
    ChatFixture() {
        sonder_engine_options eo;
        sonder_engine_options_init(&eo);
        eo.telemetry_level = SONDER_TELEMETRY_OFF;
        sonder_engine* e = nullptr;
        REQUIRE(sonder_engine_create(&eo, &e) == SONDER_OK);
        engine.reset(e);
        REQUIRE(sonder_engine_register_mock_backend(e) == SONDER_OK);
        sonder_model* m = nullptr;
        REQUIRE(sonder_model_load(e, "mock", "mock:tiny", &m) == SONDER_OK);
        model.reset(m);
        sonder_sampling_config cfg;
        sonder_sampling_config_init(&cfg);
        cfg.temperature = 0;
        cfg.max_tokens = 5;
        cfg.has_seed = 1;
        cfg.seed = 7;
        sonder_session* s = nullptr;
        REQUIRE(sonder_session_create(e, m, &cfg, &s) == SONDER_OK);
        session.reset(s);
    }
};
struct ChatOutput {
    std::string text;
    int chunks = 0;
    int stop_at = 0;
    sonder_session* cancel = nullptr;
};
int chat_token(void* user, const char* text, size_t length) {
    auto& output = *static_cast<ChatOutput*>(user);
    output.text.append(text, length);
    ++output.chunks;
    if (output.cancel) {
        CHECK(sonder_session_cancel(output.cancel) == SONDER_OK);
    }
    return output.stop_at != 0 && output.chunks >= output.stop_at ? 1 : 0;
}
}  // namespace

TEST_SUITE("c_abi_chat") {
TEST_CASE("text chat uses the engine fallback and preserves handle lifetime") {
    ChatFixture f;
    const sonder_chat_message records[] = {
        {sizeof(sonder_chat_message), "system", "Be concise"},
        {sizeof(sonder_chat_message), "user", "hello"},
        {sizeof(sonder_chat_message), "assistant", "hi"},
        {sizeof(sonder_chat_message), "tool", "ready"}};
    const sonder_chat_message* messages[] = {&records[0], &records[1], &records[2], &records[3]};
    f.model.reset();
    f.engine.reset();
    ChatOutput chat, generate;
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_chat(f.session.get(), messages, 4, chat_token, &chat, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(stats.completion_tokens == 5);
    CHECK(stats.chunks == 5);
    CHECK(stats.prompt_tokens > 0);
    CHECK(stats.ttft_ms >= 0);
    CHECK(stats.total_ms >= stats.ttft_ms);
    REQUIRE(sonder_session_generate(f.session.get(),
        "System: Be concise\n\nUser: hello\n\nAssistant: hi\n\nTool: ready\n\nAssistant:",
        chat_token, &generate, nullptr) == SONDER_OK);
    CHECK(chat.text == generate.text);
    CHECK(chat.chunks == generate.chunks);
    CHECK(std::string(sonder_last_error_message()).empty());
}

TEST_CASE("text chat validates versioned records and conversation without poisoning reuse") {
    ChatFixture f;
    sonder_chat_message record{sizeof(sonder_chat_message), "user", "hi"};
    const sonder_chat_message* messages[] = {&record};
    auto run = [&] { return sonder_session_chat(f.session.get(), messages, 1, nullptr, nullptr, nullptr); };
    CHECK(sonder_session_chat(nullptr, messages, 1, nullptr, nullptr, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_session_chat(f.session.get(), nullptr, 1, nullptr, nullptr, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_session_chat(f.session.get(), messages, 0, nullptr, nullptr, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    messages[0] = nullptr;
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    messages[0] = &record;
    record.struct_size = offsetof(sonder_chat_message, content);
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    record.struct_size = sizeof(record);
    record.role = nullptr;
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    record.role = "user";
    record.content = nullptr;
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    record.content = "";
    record.role = "unknown";
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    record.role = "assistant";
    CHECK(run() == SONDER_ERROR_INVALID_ARGUMENT);
    record.role = "user";
    sonder_generation_stats too_small{};
    too_small.struct_size = 4;
    CHECK(sonder_session_chat(f.session.get(), messages, 1, nullptr, nullptr, &too_small) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(run() == SONDER_OK);
}

TEST_CASE("text chat accepts the exact count bound and ignores appended record tails") {
    ChatFixture f;
    struct FutureMessage {
        sonder_chat_message prefix;
        const char* future;
    };
    std::vector<FutureMessage> records(SONDER_MAX_CHAT_MESSAGES + 1);
    std::vector<const sonder_chat_message*> messages;
    for (auto& record : records) {
        record.prefix = {sizeof(FutureMessage), "user", ""};
        record.future = "ignored";
        messages.push_back(&record.prefix);
    }
    CHECK(sonder_session_chat(f.session.get(), messages.data(), SONDER_MAX_CHAT_MESSAGES,
                             nullptr, nullptr, nullptr) == SONDER_OK);
    CHECK(sonder_session_chat(f.session.get(), messages.data(), messages.size(),
                             nullptr, nullptr, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_session_generate(f.session.get(), "still usable", nullptr, nullptr, nullptr) == SONDER_OK);
}

TEST_CASE("text chat callback stop and cancellation leave a reusable session") {
    ChatFixture f;
    const sonder_chat_message record{sizeof(sonder_chat_message), "user", "hi"};
    const sonder_chat_message* messages[] = {&record};
    ChatOutput early;
    early.stop_at = 2;
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_chat(f.session.get(), messages, 1, chat_token, &early, &stats) == SONDER_OK);
    CHECK(early.chunks == 2);
    ChatOutput cancelled;
    cancelled.cancel = f.session.get();
    REQUIRE(sonder_session_chat(f.session.get(), messages, 1, chat_token, &cancelled, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_CANCELLED);
    CHECK(stats.completion_tokens < 5);
    REQUIRE(sonder_session_chat(f.session.get(), messages, 1, nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(stats.chunks == 5);
}
}
