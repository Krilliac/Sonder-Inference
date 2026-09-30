// Anthropic Messages compatibility endpoint, exercised end to end against
// the deterministic mock backend.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/backends.hpp"
#include "src/test_hooks.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

class FirstChunkGate {
public:
    void reached(bool delivered) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            reached_ = true;
            delivered_ = delivered;
        }
        cv_.notify_all();
    }
    bool wait_for_reached(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] { return reached_; });
    }
    bool delivered() const {
        std::lock_guard<std::mutex> lock(mu_);
        return delivered_;
    }
    void wait_for_release() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return released_; });
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool reached_ = false;
    bool delivered_ = false;
    bool released_ = false;
};

class GatedModel final : public si::BackendModel {
public:
    GatedModel(std::shared_ptr<si::BackendModel> inner, std::shared_ptr<FirstChunkGate> gate)
        : inner_(std::move(inner)), gate_(std::move(gate)) {}

    const si::ModelDescriptor& descriptor() const override { return inner_->descriptor(); }
    si::Result<si::GenerateStats> generate(const si::GenerateRequest& request,
                                           const si::CancellationToken& cancel,
                                           const si::TokenCallback& on_chunk) override {
        bool first = true;
        return inner_->generate(request, cancel, [&](const si::TokenChunk& chunk) {
            const bool accepted = !on_chunk || on_chunk(chunk);
            if (first) {
                first = false;
                gate_->reached(accepted);
                gate_->wait_for_release();
            }
            return accepted;
        });
    }
    si::Result<std::vector<si::TokenId>> tokenize(std::string_view text) override {
        return inner_->tokenize(text);
    }

private:
    std::shared_ptr<si::BackendModel> inner_;
    std::shared_ptr<FirstChunkGate> gate_;
};

class GatedBackend final : public si::Backend {
public:
    explicit GatedBackend(std::shared_ptr<FirstChunkGate> gate)
        : inner_(si::make_mock_backend()), gate_(std::move(gate)) {}

    std::atomic<int> loads{0};

    std::string name() const override { return inner_->name(); }
    std::string description() const override { return "gated mock backend"; }
    si::BackendCapabilities capabilities() const override { return inner_->capabilities(); }
    si::Result<std::string> probe() override { return inner_->probe(); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner_->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& options) override {
        ++loads;
        auto loaded = inner_->load_model(options);
        if (!loaded.ok() || options.model != "mock:a") return loaded;
        return std::shared_ptr<si::BackendModel>(std::make_shared<GatedModel>(loaded.value(), gate_));
    }

private:
    std::shared_ptr<si::Backend> inner_;
    std::shared_ptr<FirstChunkGate> gate_;
};

std::string messages_body(const std::string& extra = "") {
    return R"({"model":"default","max_tokens":6,"messages":[{"role":"user","content":"hello sonder"}])" +
           extra + "}";
}

struct SseEvent {
    std::string name;
    std::string data;
};

std::vector<SseEvent> sse_events(const std::string& body) {
    std::vector<SseEvent> out;
    std::size_t pos = 0;
    while (pos < body.size()) {
        const auto end = body.find("\n\n", pos);
        const std::string frame = body.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const auto event_at = frame.find("event: ");
        const auto data_at = frame.find("data: ");
        if (event_at != std::string::npos && data_at != std::string::npos) {
            const auto event_end = frame.find('\n', event_at);
            const auto data_end = frame.find('\n', data_at);
            out.push_back(
                {frame.substr(event_at + 7, event_end - event_at - 7),
                 frame.substr(data_at + 6,
                              data_end == std::string::npos ? std::string::npos : data_end - data_at - 6)});
        }
        if (end == std::string::npos)
            break;
        pos = end + 2;
    }
    return out;
}

void check_anthropic_error(const Reply& r, int status, const std::string& type,
                           const std::string& message_fragment = {}) {
    REQUIRE_MESSAGE(r.status == status, r.body);
    const auto doc = r.json();
    CHECK(doc.find("type")->as_string() == "error");
    const auto* error = doc.find("error");
    REQUIRE(error != nullptr);
    CHECK(error->find("type")->as_string() == type);
    if (!message_fragment.empty())
        CHECK(error->find("message")->as_string().find(message_fragment) != std::string::npos);
}

} // namespace

TEST_CASE("anthropic messages: non-streaming response and usage") {
    Fixture f;
    const Reply r = post(f.port, "/v1/messages", messages_body());
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto doc = r.json();
    CHECK(doc.find("id")->as_string().rfind("msg_", 0) == 0);
    CHECK(doc.find("type")->as_string() == "message");
    CHECK(doc.find("role")->as_string() == "assistant");
    CHECK(doc.find("model")->as_string() == "mock:tiny");
    const auto& content = doc.find("content")->as_array();
    REQUIRE(content.size() == 1);
    CHECK(content[0].find("type")->as_string() == "text");
    CHECK_FALSE(content[0].find("text")->as_string().empty());
    CHECK(doc.find("stop_reason")->as_string() == "max_tokens");
    CHECK((doc.find("stop_sequence") == nullptr || doc.find("stop_sequence")->is_null()));
    const auto* usage = doc.find("usage");
    REQUIRE(usage != nullptr);
    CHECK(usage->find("input_tokens")->as_int() > 0);
    CHECK(usage->find("output_tokens")->as_int() == 6);
    CHECK(usage->find("cache_read_input_tokens") == nullptr);
}

TEST_CASE("anthropic messages: string and text-block content, system variants, metadata affinity") {
    Fixture f;
    const Reply string_system = post(
        f.port, "/v1/messages",
        R"({"model":"default","max_tokens":2,"system":"be concise","messages":[{"role":"user","content":"hello"}]})");
    REQUIRE_MESSAGE(string_system.status == 200, string_system.body);
    const std::string blocks =
        R"({"model":"default","max_tokens":3,"system":[{"type":"text","text":"be concise"}],"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]},{"role":"assistant","content":"hi"},{"role":"user","content":[{"type":"text","text":"again"}]}],"metadata":{"user_id":"anthropic-user-7"},"temperature":0.2,"top_p":0.9,"top_k":20,"stop_sequences":["<END>"],"thinking":{"type":"disabled"}})";
    const Reply r = post(f.port, "/v1/messages", blocks, {{"X-Sonder-Run-Id", "run-from-header"}});
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(r.json().find("content")->as_array().size() == 1);
}

TEST_CASE("anthropic messages: streaming follows the Messages SSE sequence") {
    Fixture f;
    const Reply r = post(f.port, "/v1/messages", messages_body(R"(,"stream":true)"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(r.header("content-type") == "text/event-stream; charset=utf-8");
    const auto frames = sse_events(r.body);
    REQUIRE(frames.size() >= 7);
    std::vector<std::string> names;
    for (const auto& frame : frames)
        names.push_back(frame.name);
    CHECK(names.front() == "message_start");
    CHECK(names[1] == "content_block_start");
    CHECK(names[names.size() - 3] == "content_block_stop");
    CHECK(names[names.size() - 2] == "message_delta");
    CHECK(names.back() == "message_stop");
    const auto block_start = json::parse(frames[1].data);
    REQUIRE(block_start.ok());
    CHECK(block_start.value().find("content_block")->find("type")->as_string() == "text");
    std::string text;
    for (const auto& frame : frames) {
        const auto parsed = json::parse(frame.data);
        REQUIRE(parsed.ok());
        REQUIRE(parsed->find("type") != nullptr);
        CHECK(parsed->find("type")->as_string() == frame.name);
        if (frame.name == "content_block_delta") {
            const auto* delta = parsed.value().find("delta");
            REQUIRE(delta != nullptr);
            CHECK(delta->find("type") != nullptr);
            if (delta->find("type")->as_string() == "text_delta")
                text += delta->find("text")->as_string();
        }
    }
    CHECK_FALSE(text.empty());
    const auto stop = json::parse(frames[frames.size() - 2].data);
    REQUIRE(stop.ok());
    CHECK(stop.value().find("delta")->find("stop_reason")->as_string() == "max_tokens");
    CHECK(stop.value().find("usage")->find("output_tokens")->as_int() == 6);
    const Reply plain = post(f.port, "/v1/messages", messages_body());
    REQUIRE(plain.status == 200);
    CHECK(text == plain.json().find("content")->as_array()[0].find("text")->as_string());
}

TEST_CASE("anthropic messages: thinking and effort controls, including pins") {
    Fixture f;
    CHECK(post(f.port, "/v1/messages", messages_body(R"(,"thinking":{"type":"disabled"})")).status == 200);
    CHECK(post(f.port, "/v1/messages", messages_body(R"(,"reasoning_effort":"low")")).status == 200);
    CHECK(post(f.port, "/v1/messages", messages_body(R"(,"output_config":{"effort":"high"})")).status == 200);
    CHECK(
        post(f.port, "/v1/messages",
             messages_body(R"(,"thinking":{"type":"enabled","budget_tokens":128},"reasoning_effort":"low")"))
            .status == 200);
    check_anthropic_error(
        post(f.port, "/v1/messages",
             messages_body(R"(,"thinking":{"type":"enabled","budget_tokens":128},"reasoning_effort":"off")")),
        400, "invalid_request_error", "thinking");
    check_anthropic_error(
        post(f.port, "/v1/messages",
             messages_body(R"(,"thinking":{"type":"disabled"},"reasoning_effort":"medium")")),
        400, "invalid_request_error", "thinking");

    auto options = Fixture::defaults();
    options.pin_enable_thinking = false;
    options.pin_reasoning_effort = "medium";
    Fixture pinned(options);
    const Reply r = post(
        pinned.port, "/v1/messages",
        messages_body(R"(,"thinking":{"type":"enabled","budget_tokens":128},"reasoning_effort":"high")"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto response = r.json();
    const auto* warnings = response.find("sonder")->find("warnings");
    REQUIRE(warnings != nullptr);
    CHECK(warnings->as_array().size() >= 1);
}

TEST_CASE("anthropic messages: stream pin warning is present in final event") {
    auto options = Fixture::defaults();
    options.pin_enable_thinking = false;
    Fixture f(options);
    const Reply r = post(f.port, "/v1/messages",
                         messages_body(R"(,"stream":true,"thinking":{"type":"enabled","budget_tokens":64})"));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    const auto frames = sse_events(r.body);
    REQUIRE(frames.size() >= 2);
    const auto final = json::parse(frames[frames.size() - 2].data);
    REQUIRE(final.ok());
    const auto* warnings = final.value().find("sonder")->find("warnings");
    REQUIRE(warnings != nullptr);
    CHECK(warnings->as_array().size() >= 1);
}

TEST_CASE("anthropic messages: unsupported tools, images and missing max_tokens are clear errors") {
    Fixture f;
    check_anthropic_error(
        post(f.port, "/v1/messages", R"({"model":"default","messages":[{"role":"user","content":"x"}]})"),
        400, "invalid_request_error", "max_tokens");
    check_anthropic_error(post(f.port, "/v1/messages", messages_body(R"(,"tools":[])")), 400,
                          "invalid_request_error", "tools");
    check_anthropic_error(post(f.port, "/v1/messages", messages_body(R"(,"tool_choice":{"type":"auto"})")),
                          400, "invalid_request_error", "tool_choice");
    check_anthropic_error(post(f.port, "/v1/messages",
                               R"({"model":"default","max_tokens":2,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]})"),
                          400, "invalid_request_error", "image");
}

TEST_CASE("anthropic messages: error envelope and x-api-key authentication") {
    auto options = Fixture::defaults();
    options.token = "s3cret-token";
    Fixture f(options);
    check_anthropic_error(post(f.port, "/v1/messages", messages_body()), 401, "authentication_error");
    check_anthropic_error(
        post(f.port, "/v1/messages", messages_body(), {{"Authorization", "Bearer wrong-token"}}), 401,
        "authentication_error");
    check_anthropic_error(post(f.port, "/v1/messages", messages_body(), {{"x-api-key", "wrong-token"}}), 401,
                          "authentication_error");
    const Reply r = post(f.port, "/v1/messages", messages_body(), {{"x-api-key", "s3cret-token"}});
    REQUIRE_MESSAGE(r.status == 200, r.body);
    CHECK(post(f.port, "/v1/messages", messages_body(), {{"Authorization", "Bearer s3cret-token"}}).status ==
          200);
    CHECK(post(f.port, "/v1/messages", messages_body(),
               {{"Authorization", "Bearer wrong"}, {"x-api-key", "s3cret-token"}})
              .status == 200);
    const Reply chat_api_key = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":1)"),
                                    {{"x-api-key", "s3cret-token"}});
    CHECK(chat_api_key.status == 401);
    check_anthropic_error(post(f.port, "/v1/messages", "{not json}", {{"x-api-key", "s3cret-token"}}), 400,
                          "invalid_request_error", "JSON");
}

TEST_CASE("anthropic messages: endpoint HTTP errors use Anthropic error envelopes") {
    Fixture f;
    check_anthropic_error(roundtrip(f.port, build_request("GET", "/v1/messages", f.port)), 405,
                          "invalid_request_error", "not allowed");
    check_anthropic_error(
        post(f.port, "/v1/messages", messages_body(), {{"Origin", "http://not-allowed.example"}}), 403,
        "permission_error");

    auto limited = Fixture::defaults();
    limited.max_body_bytes = 32;
    Fixture small(limited);
    check_anthropic_error(post(small.port, "/v1/messages", messages_body()), 413, "request_too_large",
                          "request body exceeds");

    const Reply missing_length = roundtrip(f.port, "POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                                   "Content-Type: application/json\r\n\r\n");
    check_anthropic_error(missing_length, 411, "invalid_request_error", "Content-Length");
    check_anthropic_error(post(f.port, "/v1/messages", messages_body(), {{"X-Sonder-Run-Id", "bad value!"}}),
                          400, "invalid_request_error", "X-Sonder-Run-Id");
    check_anthropic_error(
        post(f.port, "/v1/messages",
             R"({"model":"does-not-exist","max_tokens":1,"messages":[{"role":"user","content":"x"}]})"),
        404, "not_found_error");
}

TEST_CASE("anthropic messages: exact stop sequence survives nonstream and empty stream output") {
    Fixture f;
    const auto baseline = post(f.port, "/v1/messages", messages_body());
    REQUIRE(baseline.status == 200);
    const std::string text = baseline.json().find("content")->as_array()[0].find("text")->as_string();
    REQUIRE_FALSE(text.empty());
    const std::string stop = text.substr(0, text.find(' '));
    REQUIRE_FALSE(stop.empty());
    auto parsed = json::parse(messages_body());
    REQUIRE(parsed.ok());
    auto body = parsed->as_object();
    body.set("stop_sequences", json::Array{stop});
    for (bool streaming : {false, true}) {
        body.set("stream", streaming);
        const Reply r = post(f.port, "/v1/messages", json::Value(body).dump());
        REQUIRE_MESSAGE(r.status == 200, r.body);
        if (!streaming) {
            const auto document = r.json();
            CHECK(document.find("stop_reason")->as_string() == "stop_sequence");
            CHECK(document.find("stop_sequence")->as_string() == stop);
        } else {
            const auto frames = sse_events(r.body);
            REQUIRE(frames.size() >= 5);
            CHECK(frames[1].name == "content_block_start");
            const auto final = json::parse(frames[frames.size() - 2].data);
            REQUIRE(final.ok());
            CHECK(final->find("delta")->find("stop_reason")->as_string() == "stop_sequence");
            CHECK(final->find("delta")->find("stop_sequence")->as_string() == stop);
            CHECK(final->find("sonder")->find("synthetic")->as_bool());
        }
    }
}

TEST_CASE("anthropic messages: watcher refusal and malformed headers use endpoint errors") {
    Fixture f;
    det::fail_watcher_spawns_for_test().store(1);
    const Reply failed = post(f.port, "/v1/messages", messages_body());
    check_anthropic_error(failed, 503, "overloaded_error");
    CHECK(failed.header("retry-after") == "1");
    CHECK(post(f.port, "/v1/messages", messages_body()).status == 200);
    check_anthropic_error(roundtrip(f.port, "POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                            "Host: localhost\r\nContent-Length: 0\r\n\r\n"),
                          400, "invalid_request_error", "Host");
}

TEST_CASE("anthropic messages: a streaming lazy request pins its model during competing eviction") {
    auto gate = std::make_shared<FirstChunkGate>();
    auto backend = std::make_shared<GatedBackend>(gate);
    auto options = Fixture::defaults();
    options.models = {"mock:a", "mock:b"};
    options.lazy_models = true;
    options.max_resident_models = 1;
    options.backend.backend.clear();
    options.backend_instance = backend;
    Fixture f(options);
    CHECK(backend->loads.load() == 0);

    Conn stream(f.port);
    const std::string request_body =
        R"({"model":"mock:a","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hold this request"}]})";
    struct GateCleanup {
        FirstChunkGate& gate;
        ~GateCleanup() { gate.release(); }
    } cleanup{*gate};
    stream.send(build_request("POST", "/v1/messages", f.port, {}, request_body));

    REQUIRE(gate->wait_for_reached(std::chrono::milliseconds(5000)));
    CHECK(gate->delivered());  // the first Anthropic SSE delta was written before the gate closed
    std::string wire;
    REQUIRE(stream.read_until(wire, "event: content_block_delta\n"));
    const auto pinned_health = get(f.port, "/v1/sonder/health").json();
    bool pinned = false;
    for (const auto& model : pinned_health.find("models")->as_array()) {
        if (model.find("id")->as_string() == "mock:a") {
            pinned = model.find("state")->as_string() == "resident" && model.find("in_flight")->as_int() > 0;
            CHECK(model.find("in_flight")->as_int() == 1);
            CHECK(model.find("loads")->as_int() == 1);
        }
    }
    REQUIRE(pinned);

    const Reply competing = post(
        f.port, "/v1/messages",
        R"({"model":"mock:b","max_tokens":1,"messages":[{"role":"user","content":"load the competitor"}]})");
    CHECK_MESSAGE(competing.status == 200, competing.body);
    // The soft cap must not unload the model held by the streaming request.
    const auto during = get(f.port, "/v1/sonder/health").json();
    for (const auto& model : during.find("models")->as_array()) {
        if (model.find("id")->as_string() == "mock:a")
            CHECK(model.find("state")->as_string() == "resident");
    }
    for (const auto& eviction : f.of_type("model.evicted")) {
        CHECK(eviction.find("attributes")->find("model")->as_string() != "mock:a");
    }

    gate->release();
    wire += stream.read_all();
    const Reply completed = parse_reply(wire);
    REQUIRE_MESSAGE(completed.status == 200, completed.body);
    CHECK(stream.closed());
    const auto frames = sse_events(completed.body);
    REQUIRE_FALSE(frames.empty());
    CHECK(frames.back().name == "message_stop");
    // A is now unpinned; loading B again forces the cap to evict A.
    const Reply reload_competing = post(
        f.port, "/v1/messages",
        R"({"model":"mock:b","max_tokens":1,"messages":[{"role":"user","content":"reload the competitor"}]})");
    REQUIRE_MESSAGE(reload_competing.status == 200, reload_competing.body);
    CHECK(eventually([&] {
        const auto health = get(f.port, "/v1/sonder/health").json();
        for (const auto& model : health.find("models")->as_array()) {
            if (model.find("id")->as_string() == "mock:a")
                return model.find("state")->as_string() == "unloaded";
        }
        return false;
    }));
    for (const auto& unload : f.of_type("model.unload")) {
        CHECK(unload.find("attributes")->find("outstanding_references")->as_int() == 0);
    }
}
