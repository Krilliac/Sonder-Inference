// Chat through the Backend interface: message validation, the generic prompt
// format, and the default BackendModel::chat() path on the MOCK backend.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {

std::shared_ptr<BackendModel> load_mock(MockBackendOptions o = {}) {
    auto b = make_mock_backend(o);
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    return b->load_model(lo).value();
}

std::vector<ChatMessage> conversation() {
    return {{"system", "Be brief."}, {"user", "Hi"}, {"assistant", "Hello."}, {"user", "What is a KV cache?"}};
}

}  // namespace

TEST_CASE("chat: roles and message validation") {
    CHECK(is_known_chat_role("system"));
    CHECK(is_known_chat_role("user"));
    CHECK(is_known_chat_role("assistant"));
    CHECK(is_known_chat_role("tool"));
    CHECK_FALSE(is_known_chat_role("User"));
    CHECK_FALSE(is_known_chat_role(""));

    CHECK(validate_chat_messages(conversation()).ok());
    CHECK(validate_chat_messages({{"user", "x"}}).ok());
    CHECK(validate_chat_messages({{"user", "x"}, {"assistant", "y"}, {"tool", "{}"}}).ok());
    CHECK(validate_chat_messages({}).code() == ErrorCode::invalid_argument);
    CHECK(validate_chat_messages({{"narrator", "x"}}).code() == ErrorCode::invalid_argument);
    const Status last_assistant = validate_chat_messages({{"user", "x"}, {"assistant", "y"}});
    CHECK(last_assistant.code() == ErrorCode::invalid_argument);
    CHECK(last_assistant.message().find("last message") != std::string::npos);
}

TEST_CASE("chat: generic prompt format") {
    CHECK(format_chat_prompt({{"user", "Hi"}}) == "User: Hi\n\nAssistant:");
    CHECK(format_chat_prompt(conversation()) ==
          "System: Be brief.\n\nUser: Hi\n\nAssistant: Hello.\n\nUser: What is a KV cache?\n\nAssistant:");
    CHECK(format_chat_prompt({}) == "Assistant:");
}

TEST_CASE("chat: mock backend uses the default path (formatted prompt -> generate)") {
    auto model = load_mock();
    CHECK_FALSE(model->has_native_chat());

    ChatRequest chat;
    chat.request_id = "chat-1";
    chat.messages = conversation();
    chat.sampling = SamplingConfig::greedy(8);
    std::string chat_text;
    std::vector<std::uint64_t> indices;
    auto cs = model->chat(chat, {}, [&](const TokenChunk& c) {
        chat_text.append(c.text);
        indices.push_back(c.index);
        return true;
    });
    REQUIRE_MESSAGE(cs.ok(), cs.status().to_string());

    GenerateRequest gen;
    gen.request_id = "gen-1";
    gen.prompt = format_chat_prompt(chat.messages);
    gen.sampling = chat.sampling;
    std::string gen_text;
    auto gs = model->generate(gen, {}, [&](const TokenChunk& c) {
        gen_text.append(c.text);
        return true;
    });
    REQUIRE(gs.ok());

    CHECK_FALSE(chat_text.empty());
    CHECK(chat_text == gen_text);
    CHECK(cs->completion_tokens == 8u);
    CHECK(cs->prompt_tokens == gs->prompt_tokens);
    CHECK(cs->stop_reason == StopReason::max_tokens);
    REQUIRE(indices.size() == 8u);
    CHECK(indices.front() == 0u);
    CHECK(indices.back() == 7u);

    // A different conversation produces a different (deterministic) reply.
    chat.messages.back().content = "Something else entirely";
    std::string other;
    REQUIRE(model->chat(chat, {}, [&](const TokenChunk& c) {
                     other.append(c.text);
                     return true;
                 }).ok());
    CHECK(other != chat_text);
}

TEST_CASE("chat: invalid messages fail before generation") {
    auto model = load_mock();
    ChatRequest chat;
    chat.sampling = SamplingConfig::greedy(4);
    int chunks = 0;
    auto cb = [&](const TokenChunk&) {
        ++chunks;
        return true;
    };
    CHECK(model->chat(chat, {}, cb).status().code() == ErrorCode::invalid_argument);
    chat.messages = {{"user", "x"}, {"assistant", "y"}};
    CHECK(model->chat(chat, {}, cb).status().code() == ErrorCode::invalid_argument);
    chat.messages = {{"wizard", "x"}};
    CHECK(model->chat(chat, {}, cb).status().code() == ErrorCode::invalid_argument);
    CHECK(chunks == 0);
}

TEST_CASE("chat: callback stop, cancellation and backend errors propagate") {
    auto model = load_mock();
    ChatRequest chat;
    chat.messages = {{"user", "stream please"}};
    chat.sampling = SamplingConfig::greedy(16);

    int n = 0;
    auto stopped = model->chat(chat, {}, [&](const TokenChunk&) { return ++n < 3; });
    REQUIRE(stopped.ok());
    CHECK(stopped->stop_reason == StopReason::callback);
    CHECK(stopped->chunks == 3u);

    CancellationSource source;
    auto cancelled = model->chat(chat, source.token(), [&](const TokenChunk&) {
        source.cancel();
        return true;
    });
    CHECK(cancelled.status().code() == ErrorCode::cancelled);

    MockBackendOptions failing;
    failing.fail_after_tokens = 2;
    auto broken = load_mock(failing);
    CHECK(broken->chat(chat, {}, {}).status().code() == ErrorCode::backend_error);
}

TEST_CASE("chat: stop sequences from SamplingConfig apply to chat") {
    auto model = load_mock();
    ChatRequest chat;
    chat.messages = {{"user", "Hi"}};
    chat.sampling = SamplingConfig::greedy(16);
    std::string full;
    REQUIRE(model->chat(chat, {}, [&](const TokenChunk& c) {
                     full.append(c.text);
                     return true;
                 }).ok());
    const auto space = full.find(' ', 1);
    REQUIRE(space != std::string::npos);
    chat.sampling.stop = {full.substr(space, 3)};
    std::string cut;
    auto st = model->chat(chat, {}, [&](const TokenChunk& c) {
        cut.append(c.text);
        return true;
    });
    REQUIRE(st.ok());
    CHECK(st->stop_reason == StopReason::stop_sequence);
    CHECK(cut == full.substr(0, space));
}
