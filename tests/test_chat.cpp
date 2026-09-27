// Chat through the Backend interface: message validation, the generic prompt
// format, and the default BackendModel::chat() path on the MOCK backend.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
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

// ---------------------------------------------------------------------------
// Session::chat: chat through the request runtime with session telemetry.
// ---------------------------------------------------------------------------

namespace {

// Backend model with a native chat API: records the messages it receives.
class NativeChatModel final : public BackendModel {
public:
    NativeChatModel() {
        descriptor_.name = "native:test";
        descriptor_.backend = "native-test";
        descriptor_.format = "test";
        descriptor_.context_length = 1024;
    }
    const ModelDescriptor& descriptor() const override { return descriptor_; }
    Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken&, const TokenCallback&) override {
        ++generate_calls;
        return Status(ErrorCode::internal, "native chat model: generate() must not be used for chat");
    }
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken&,
                               const TokenCallback& on_chunk) override {
        received = request.messages;
        GenerateStats stats;
        stats.prompt_tokens = 3;
        stats.token_counts_from_backend = true;
        for (const char* piece : {"native", " reply"}) {
            ++stats.completion_tokens;
            if (on_chunk && !on_chunk(TokenChunk{piece, stats.chunks++})) {
                stats.stop_reason = StopReason::callback;
                return stats;
            }
        }
        stats.stop_reason = StopReason::end_of_sequence;
        return stats;
    }
    bool has_native_chat() const override { return true; }

    std::vector<ChatMessage> received;
    int generate_calls = 0;

private:
    ModelDescriptor descriptor_;
};

class NativeChatBackend final : public Backend {
public:
    explicit NativeChatBackend(std::shared_ptr<NativeChatModel> model) : model_(std::move(model)) {}
    std::string name() const override { return "native-test"; }
    std::string description() const override { return "test backend with native chat"; }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming);
        return c;
    }
    Result<std::string> probe() override { return std::string("test"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
        return std::shared_ptr<BackendModel>(model_);
    }

private:
    std::shared_ptr<NativeChatModel> model_;
};

std::vector<std::string> types_for(sonder_test::Harness& h, const std::string& request_id) {
    std::vector<std::string> out;
    for (const auto& e : h.events()) {
        const json::Value* rid = e.find("request_id");
        if (rid != nullptr && rid->is_string() && rid->as_string() == request_id) {
            out.push_back(e.find("event_type")->as_string());
        }
    }
    return out;
}

}  // namespace

TEST_CASE("session chat: request lifecycle with kind chat, generic template on the mock") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(6));
    REQUIRE(session);
    std::string streamed;
    auto res = session->chat(conversation(), [&](const TokenChunk& c) {
        streamed.append(c.text);
        return true;
    });
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    CHECK(res->outcome == RequestOutcome::completed);
    CHECK(res->text == streamed);
    CHECK(res->stats.completion_tokens == 6u);

    // Same text as generating from the formatted prompt: the mock has no
    // native chat, so chat goes through format_chat_prompt().
    auto gen_session = h.session(SamplingConfig::greedy(6));
    auto gen = gen_session->generate(format_chat_prompt(conversation()));
    REQUIRE(gen.ok());
    CHECK(gen->text == res->text);

    const auto types = types_for(h, res->request_id);
    std::vector<std::string> lifecycle;
    for (const auto& t : types) {
        if (t.rfind("request.", 0) == 0) lifecycle.push_back(t);
    }
    CHECK(lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});
    CHECK(types.front() == "request.queued");
    CHECK(types.back() == "request.completed");
    for (const char* t : {"inference.decode.started", "inference.token.generated", "inference.prefill.completed",
                          "inference.decode.completed"}) {
        CHECK_MESSAGE(std::find(types.begin(), types.end(), t) != types.end(), t);
    }
    bool checked_queued = false;
    for (const auto& e : h.events_of("request.queued")) {
        if (e.find("request_id")->as_string() != res->request_id) continue;
        checked_queued = true;
        CHECK(e.find("attributes")->find("kind")->as_string() == "chat");
        CHECK(e.find("attributes")->find("messages")->as_int() == 4);
        CHECK(e.find("attributes")->find("parent_request_id") == nullptr);
    }
    CHECK(checked_queued);
    for (const auto& e : h.events_of("request.started")) {
        if (e.find("request_id")->as_string() == res->request_id) {
            CHECK(e.find("attributes")->find("kind")->as_string() == "chat");
            CHECK(e.find("attributes")->find("chat_template")->as_string() == "generic");
        } else {
            CHECK(e.find("attributes")->find("kind")->as_string() == "generate");
        }
    }
}

TEST_CASE("session chat: parent_request_id and a caller-supplied request id") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(3));
    RequestOptions ro;
    ro.parent_request_id = "rt-turn-1";
    ro.request_id = make_id("req");
    auto res = session->chat({{"user", "hi"}}, {}, std::nullopt, ro);
    REQUIRE(res.ok());
    CHECK(res->request_id == *ro.request_id);
    int with_parent = 0;
    for (const auto& e : h.events()) {
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        CHECK(e.find("request_id")->as_string() == *ro.request_id);
        CHECK(e.find("attributes")->find("parent_request_id")->as_string() == "rt-turn-1");
        ++with_parent;
    }
    CHECK(with_parent == 3);  // queued, started, completed

    // generate() takes the same options; cancelled and failed carry it too.
    MockBackendOptions slow;
    slow.token_delay = std::chrono::milliseconds(5);
    slow.fail_after_tokens = 3;
    sonder_test::Harness f(slow);
    auto fs = f.session(SamplingConfig::greedy(8));
    RequestOptions fo;
    fo.parent_request_id = "rt-turn-2";
    CHECK_FALSE(fs->generate("x", {}, std::nullopt, fo).ok());
    auto failed = f.events_of("request.failed");
    REQUIRE(failed.size() == 1);
    CHECK(failed[0].find("attributes")->find("parent_request_id")->as_string() == "rt-turn-2");
    CHECK_FALSE(fs->last_scheduler_rejected());

    auto cs = f.session(SamplingConfig::greedy(2));
    auto cancelled = cs->chat({{"user", "x"}}, [&](const TokenChunk&) {
        cs->cancel();
        return true;
    }, std::nullopt, fo);
    REQUIRE(cancelled.ok());
    CHECK(cancelled->outcome == RequestOutcome::cancelled);
    auto cev = f.events_of("request.cancelled");
    REQUIRE(cev.size() == 1);
    CHECK(cev[0].find("attributes")->find("parent_request_id")->as_string() == "rt-turn-2");
}

TEST_CASE("session chat: invalid messages fail before any request event") {
    sonder_test::Harness h;
    auto session = h.session();
    CHECK(session->chat({}).status().code() == ErrorCode::invalid_argument);
    CHECK(session->chat({{"user", "x"}, {"assistant", "y"}}).status().code() == ErrorCode::invalid_argument);
    CHECK(h.events_of("request.queued").empty());
    CHECK(session->state() == SessionState::idle);
    CHECK(session->chat({{"user", "ok"}}).ok());
}

TEST_CASE("session chat: native chat backends receive the messages") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    Engine engine(std::move(eo));
    auto native = std::make_shared<NativeChatModel>();
    REQUIRE(engine.register_backend(std::make_shared<NativeChatBackend>(native)).ok());
    ModelLoadOptions lo;
    lo.model = "native:test";
    auto model = engine.load_model("native-test", lo);
    REQUIRE(model.ok());
    auto session = engine.create_session(model.value(), SessionOptions{});
    REQUIRE(session.ok());
    auto res = session.value()->chat(conversation());
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    CHECK(res->text == "native reply");
    CHECK(native->generate_calls == 0);
    REQUIRE(native->received.size() == conversation().size());
    CHECK(native->received.back().content == "What is a KV cache?");
    engine.telemetry().flush();
    bool native_template = false;
    for (const auto& line : sink->lines()) {
        auto v = json::parse(line);
        REQUIRE(v.ok());
        if (v.value().find("event_type")->as_string() == "request.started") {
            native_template = v.value().find("attributes")->find("chat_template")->as_string() == "native";
        }
    }
    CHECK(native_template);
}
