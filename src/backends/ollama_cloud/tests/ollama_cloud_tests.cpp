// Ollama Cloud provider tests. A scripted in-process transport stands in for
// HTTPS, so there is no network, no TLS endpoint and no real API key. The key
// used here is an obvious fake.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backends/ollama_cloud/ollama_cloud_protocol.hpp"
#include "backends/ollama_cloud/ollama_cloud_secret.hpp"
#include "net/tls.hpp"
#include "sonder/inference.hpp"
#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/backends/ollama_cloud.hpp"

using namespace sonder::inference;
namespace fs = std::filesystem;

namespace {

constexpr const char* kFakeKey = "sk-FAKE-test-key-0123456789abcdef";
constexpr const char* kTagsBody =
    R"({"models":[{"name":"gpt-oss:120b","model":"gpt-oss:120b","size":65000000000,)"
    R"("details":{"format":"","family":"gptoss","parameter_size":"116.8B","quantization_level":"MXFP4"}},)"
    R"({"name":"qwen3-coder:480b","size":1},{"name":"not-allowlisted:7b","size":2}]})";

struct Route {
    int status = 200;
    std::vector<std::string> chunks;
    Status error;                // returned before any status/body when not ok
    bool block_after_chunks = false;  // then wait for cancellation
};

// Thread-safe scripted transport. Records every request it is handed.
class FakeTransport final : public OllamaCloudTransport {
public:
    void route(const std::string& target, Route r) {
        std::lock_guard<std::mutex> lock(mu_);
        routes_[target] = std::move(r);
    }
    std::vector<OllamaCloudHttpRequest> requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return requests_;
    }
    std::size_t count(const std::string& target) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::size_t n = 0;
        for (const auto& r : requests_) n += r.target == target ? 1u : 0u;
        return n;
    }
    // Signalled once a blocking route has delivered its chunks.
    void wait_blocked() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, std::chrono::seconds(5), [this] { return blocked_; });
    }

    Status send(const OllamaCloudHttpRequest& request, const std::function<void(int)>& on_status,
                const std::function<bool(std::string_view)>& on_body, const CancellationToken& cancel) override {
        Route r;
        {
            std::lock_guard<std::mutex> lock(mu_);
            requests_.push_back(request);
            auto it = routes_.find(request.target);
            if (it == routes_.end()) {
                return Status(ErrorCode::unavailable, "fake transport: no route for " + request.target);
            }
            r = it->second;
        }
        if (!r.error.ok()) {
            return r.error;
        }
        if (on_status) on_status(r.status);
        for (const auto& c : r.chunks) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "fake transport: cancelled");
            if (!on_body(c)) return Status::success();
        }
        if (r.block_after_chunks) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                blocked_ = true;
            }
            cv_.notify_all();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!cancel.cancelled()) {
                if (std::chrono::steady_clock::now() > deadline) {
                    return Status(ErrorCode::timeout, "fake transport: never cancelled");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            return Status(ErrorCode::cancelled, "fake transport: cancelled");
        }
        return Status::success();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool blocked_ = false;
    std::map<std::string, Route> routes_;
    std::vector<OllamaCloudHttpRequest> requests_;
};

OllamaCloudOptions options_with(const std::shared_ptr<FakeTransport>& t) {
    OllamaCloudOptions o;
    o.api_key_env = "SONDER_TEST_OLLAMA_CLOUD_KEY";
    o.env_lookup = [](const char* name) -> std::optional<std::string> {
        if (std::string(name) == "SONDER_TEST_OLLAMA_CLOUD_KEY") return std::string(kFakeKey);
        return std::nullopt;
    };
    o.models = {"gpt-oss:120b", "qwen3-coder:480b"};
    o.transport = t;
    return o;
}

std::shared_ptr<FakeTransport> fake_with_tags() {
    auto t = std::make_shared<FakeTransport>();
    t->route("/api/tags", Route{200, {kTagsBody}, {}, false});
    return t;
}

const std::string kChatStream =
    "{\"model\":\"gpt-oss:120b\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"plan\"},\"done\":false}\n"
    "{\"model\":\"gpt-oss:120b\",\"message\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"done\":false}\n"
    "{\"model\":\"gpt-oss:120b\",\"message\":{\"role\":\"assistant\",\"content\":\"lo\"},\"done\":false}\n"
    "{\"model\":\"gpt-oss:120b\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,"
    "\"done_reason\":\"stop\",\"total_duration\":900,\"load_duration\":5,\"prompt_eval_count\":11,"
    "\"prompt_eval_duration\":100,\"eval_count\":3,\"eval_duration\":200}\n";

std::shared_ptr<BackendModel> load(const std::shared_ptr<Backend>& be, const std::string& name = "gpt-oss:120b") {
    ModelLoadOptions lo;
    lo.model = name;
    auto m = be->load_model(lo);
    REQUIRE_MESSAGE(m.ok(), m.status().to_string());
    return m.value();
}

ChatRequest user_chat(const std::string& text) {
    ChatRequest r;
    r.messages.push_back(ChatMessage{"user", text});
    r.sampling = SamplingConfig::greedy(16);
    return r;
}

std::string header(const OllamaCloudHttpRequest& r, const std::string& name) {
    for (const auto& [k, v] : r.headers) {
        if (k == name) return v;
    }
    return {};
}

bool contains_key(std::string_view text) { return text.find(kFakeKey) != std::string_view::npos; }

fs::path temp_file(const std::string& name, const std::string& content) {
    const fs::path p = fs::temp_directory_path() / ("sonder-ollama-cloud-" + name);
    std::ofstream(p, std::ios::binary) << content;
    return p;
}

}  // namespace

TEST_CASE("options: https only, exactly one key source, sane model list") {
    auto t = fake_with_tags();
    {
        auto o = options_with(t);
        o.base_url = "http://ollama.com";
        auto r = make_ollama_cloud_backend(o);
        REQUIRE_FALSE(r.ok());
        CHECK(r.status().code() == ErrorCode::invalid_argument);
        CHECK(r.status().message().find("https://") != std::string::npos);
    }
    {
        auto o = options_with(t);
        o.api_key_env.clear();
        CHECK(make_ollama_cloud_backend(o).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto o = options_with(t);
        o.api_key_file = "some/path";
        CHECK(make_ollama_cloud_backend(o).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto o = options_with(t);
        o.env_lookup = [](const char*) -> std::optional<std::string> { return std::nullopt; };
        auto r = make_ollama_cloud_backend(o);
        REQUIRE_FALSE(r.ok());
        CHECK(r.status().message().find("SONDER_TEST_OLLAMA_CLOUD_KEY is not set") != std::string::npos);
    }
    {
        auto o = options_with(t);
        o.models = {"a", "a"};
        CHECK(make_ollama_cloud_backend(o).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto o = options_with(t);
        o.base_url = "https://ollama.com/?x=1";
        CHECK(make_ollama_cloud_backend(o).status().code() == ErrorCode::invalid_argument);
    }
    {
        // A malformed key is refused and never quoted back.
        auto o = options_with(t);
        o.env_lookup = [](const char*) -> std::optional<std::string> { return std::string("sk-FAKE bad\tkey"); };
        auto r = make_ollama_cloud_backend(o);
        REQUIRE_FALSE(r.ok());
        CHECK(r.status().message().find("sk-FAKE") == std::string::npos);
    }
    CHECK(make_ollama_cloud_backend(options_with(t)).ok());
}

TEST_CASE("base url: default port 443, explicit port and path prefix") {
    auto a = ollama_cloud::parse_base_url("https://ollama.com");
    REQUIRE(a.ok());
    CHECK(a->host == "ollama.com");
    CHECK(a->port == 443);
    CHECK(ollama_cloud::join_target(a->path, "/api/chat") == "/api/chat");
    auto b = ollama_cloud::parse_base_url("HTTPS://example.test:8443/prefix/");
    REQUIRE(b.ok());
    CHECK(b->port == 8443);
    CHECK(ollama_cloud::join_target(b->path, "/api/tags") == "/prefix/api/tags");
    auto c = ollama_cloud::parse_base_url("https://[::1]");
    REQUIRE(c.ok());
    CHECK(c->port == 443);
    CHECK_FALSE(ollama_cloud::parse_base_url("https://user@ollama.com").ok());
    CHECK_FALSE(ollama_cloud::parse_base_url("ftp://ollama.com").ok());
}

TEST_CASE("key file: trimmed, sent as a bearer token to the configured host") {
    const auto path = temp_file("key.txt", std::string("  ") + kFakeKey + "\r\n");
    auto t = fake_with_tags();
    auto o = options_with(t);
    o.api_key_env.clear();
    o.env_lookup = nullptr;
    o.api_key_file = path.string();
    o.base_url = "https://cloud.example.test:8443/base";
    t->route("/base/api/tags", Route{200, {kTagsBody}, {}, false});
    auto be = make_ollama_cloud_backend(o);
    std::error_code ec;
    fs::remove(path, ec);
    REQUIRE_MESSAGE(be.ok(), be.status().to_string());
    auto probe = be.value()->probe();
    REQUIRE_MESSAGE(probe.ok(), probe.status().to_string());
    const auto reqs = t->requests();
    REQUIRE(reqs.size() == 1u);
    CHECK(reqs[0].method == "GET");
    CHECK(reqs[0].host == "cloud.example.test");
    CHECK(reqs[0].port == 8443);
    CHECK(reqs[0].target == "/base/api/tags");
    CHECK(header(reqs[0], "Authorization") == std::string("Bearer ") + kFakeKey);

    auto missing = options_with(t);
    missing.api_key_env.clear();
    missing.api_key_file = (fs::temp_directory_path() / "sonder-ollama-cloud-no-such-key.txt").string();
    CHECK(make_ollama_cloud_backend(missing).status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("policy: only allowlisted models are listed and loadable") {
    auto t = fake_with_tags();
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    CHECK(be->name() == "ollama_cloud");
    CHECK(be->capabilities().has(Capability::remote_process));
    CHECK(be->capabilities().has(Capability::streaming));
    CHECK_FALSE(be->capabilities().has(Capability::structured_output));
    auto models = be->list_models();
    REQUIRE(models.ok());
    REQUIRE(models->size() == 2u);
    CHECK(models->at(0).name == "gpt-oss:120b");
    CHECK(models->at(0).backend == "ollama_cloud");
    CHECK(models->at(0).quantization == "MXFP4");
    CHECK_FALSE(models->at(0).resident);
    CHECK(models->at(1).name == "qwen3-coder:480b");

    ModelLoadOptions lo;
    lo.model = "not-allowlisted:7b";  // the cloud offers it, but it is not configured as cloud
    auto refused = be->load_model(lo);
    REQUIRE_FALSE(refused.ok());
    CHECK(refused.status().code() == ErrorCode::invalid_argument);
    CHECK(refused.status().message().find("not configured as a cloud model") != std::string::npos);

    auto o = options_with(t);
    o.models = {"gone:1b"};
    auto be2 = make_ollama_cloud_backend(o).value();
    lo.model = "gone:1b";
    CHECK(be2->load_model(lo).status().code() == ErrorCode::not_found);

    auto empty = options_with(t);
    empty.models.clear();
    auto be3 = make_ollama_cloud_backend(empty).value();
    lo.model = "gpt-oss:120b";
    CHECK(be3->load_model(lo).status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("chat: streams content, maps the request body and server timings") {
    auto t = fake_with_tags();
    // Split mid-line to exercise reassembly.
    t->route("/api/chat", Route{200, {kChatStream.substr(0, 37), kChatStream.substr(37)}, {}, false});
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);
    CHECK(model->has_native_chat());
    auto req = user_chat("hi");
    req.thinking.enable_thinking = true;
    std::string text;
    auto stats = model->chat(req, {}, [&](const TokenChunk& c) {
        text.append(c.text);
        CHECK(c.reasoning.empty());
        return true;
    });
    REQUIRE_MESSAGE(stats.ok(), stats.status().to_string());
    CHECK(text == "Hello");
    CHECK(stats->prompt_tokens == 11u);
    CHECK(stats->completion_tokens == 3u);
    CHECK(stats->token_counts_from_backend);
    CHECK(stats->stop_reason == StopReason::end_of_sequence);

    const auto reqs = t->requests();
    const OllamaCloudHttpRequest* chat = nullptr;
    for (const auto& r : reqs) {
        if (r.target == "/api/chat") chat = &r;
    }
    REQUIRE(chat != nullptr);
    CHECK(chat->method == "POST");
    CHECK(header(*chat, "Authorization") == std::string("Bearer ") + kFakeKey);
    auto body = json::parse(chat->body);
    REQUIRE(body.ok());
    CHECK(body->find("model")->as_string() == "gpt-oss:120b");
    CHECK(body->find("stream")->as_bool());
    CHECK(body->find("think")->as_bool());
    REQUIRE(body->find("messages")->as_array().size() == 1u);
    CHECK(body->find("messages")->as_array()[0].find("content")->as_string() == "hi");
    CHECK(body->find("options")->find("num_predict")->as_int() == 16);
    CHECK_FALSE(contains_key(chat->body));
}

TEST_CASE("chat: separate reasoning, tool calls and generate") {
    auto t = fake_with_tags();
    t->route("/api/chat", Route{200, {kChatStream}, {}, false});
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);
    auto req = user_chat("hi");
    req.separate_reasoning = true;
    std::string text;
    std::string reasoning;
    REQUIRE(model->chat(req, {}, [&](const TokenChunk& c) {
                     text.append(c.text);
                     reasoning.append(c.reasoning);
                     return true;
                 }).ok());
    CHECK(text == "Hello");
    CHECK(reasoning == "plan");

    t->route("/api/chat",
             Route{200,
                   {"{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":"
                    "\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}}]},\"done\":false}\n"
                    "{\"message\":{\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n"},
                   {},
                   false});
    std::vector<ToolCallDelta> calls;
    auto tool_stats = model->chat(user_chat("weather?"), {}, [&](const TokenChunk& c) {
        calls.insert(calls.end(), c.tool_call_deltas.begin(), c.tool_call_deltas.end());
        return true;
    });
    REQUIRE(tool_stats.ok());
    CHECK(tool_stats->stop_reason == StopReason::tool_calls);
    REQUIRE(calls.size() == 1u);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_delta == R"({"city":"Paris"})");

    t->route("/api/generate", Route{200,
                                    {"{\"response\":\"ab\",\"done\":false}\n{\"response\":\"c\",\"done\":true,"
                                     "\"done_reason\":\"length\",\"eval_count\":2}\n"},
                                    {},
                                    false});
    GenerateRequest g;
    g.prompt = "x";
    g.sampling = SamplingConfig::greedy(2);
    std::string out;
    auto gs = model->generate(g, {}, [&](const TokenChunk& c) {
        out.append(c.text);
        return true;
    });
    REQUIRE(gs.ok());
    CHECK(out == "abc");
    CHECK(gs->stop_reason == StopReason::max_tokens);
}

TEST_CASE("refusals: local-only, structured output, grammar and images never reach the network") {
    auto t = fake_with_tags();
    t->route("/api/chat", Route{200, {kChatStream}, {}, false});
    t->route("/api/generate", Route{200, {kChatStream}, {}, false});
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);

    auto local = user_chat("secret plans");
    local.local_only = true;
    auto r = model->chat(local, {}, {});
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(r.status().message().find("local-only") != std::string::npos);
    CHECK(r.status().message().find("gpt-oss:120b") != std::string::npos);

    GenerateRequest g;
    g.prompt = "secret";
    g.local_only = true;
    CHECK(model->generate(g, {}, {}).status().code() == ErrorCode::invalid_argument);

    auto fmt = user_chat("json please");
    fmt.response_format_type = "json_object";
    CHECK(model->chat(fmt, {}, {}).status().code() == ErrorCode::unsupported);
    auto gram = user_chat("x");
    gram.grammar = "root ::= \"a\"";
    CHECK(model->chat(gram, {}, {}).status().code() == ErrorCode::unsupported);
    auto img = user_chat("look");
    img.messages[0].media.push_back(MediaContent{});
    CHECK(model->chat(img, {}, {}).status().code() == ErrorCode::unsupported);

    CHECK(t->count("/api/chat") == 0u);
    CHECK(t->count("/api/generate") == 0u);

    // The refusal is visible in telemetry as a cloud-provider event.
    auto status = be->runtime_status();
    REQUIRE(status.has_value());
    REQUIRE(status->take_events);
    auto events = status->take_events();
    REQUIRE(events.size() == 2u);
    CHECK(events[0].first == "backend.remote.request");
    CHECK(events[0].second.find("outcome")->as_string() == "refused");
    CHECK(events[0].second.find("provider")->as_string() == "ollama_cloud");
    CHECK(events[0].second.find("locality")->as_string() == "cloud");
    CHECK(events[0].second.find("remote")->as_bool());
    CHECK(status->diagnostics->find("refused_local_only")->as_uint() == 2u);
}

TEST_CASE("errors: upstream HTTP, mid-stream and transport errors") {
    auto t = fake_with_tags();
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);
    t->route("/api/chat", Route{401, {R"({"error":"unauthorized"})"}, {}, false});
    auto unauth = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(unauth.ok());
    CHECK(unauth.status().code() == ErrorCode::unavailable);
    CHECK(unauth.status().message().find("HTTP 401") != std::string::npos);

    t->route("/api/chat", Route{429, {R"({"error":"too many requests"})"}, {}, false});
    CHECK(model->chat(user_chat("x"), {}, {}).status().code() == ErrorCode::unavailable);

    t->route("/api/chat", Route{200,
                                {"{\"message\":{\"content\":\"a\"},\"done\":false}\n"
                                 "{\"error\":\"an error was encountered while running the model\"}\n"},
                                {},
                                false});
    auto mid = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(mid.ok());
    CHECK(mid.status().code() == ErrorCode::backend_error);

    t->route("/api/chat", Route{200, {"{\"message\":{\"content\":\"a\"},\"done\":false}\n"}, {}, false});
    CHECK(model->chat(user_chat("x"), {}, {}).status().code() == ErrorCode::protocol_error);

    t->route("/api/chat", Route{200, {}, Status(ErrorCode::unavailable, "connect failed"), false});
    CHECK(model->chat(user_chat("x"), {}, {}).status().code() == ErrorCode::unavailable);
}

TEST_CASE("secrets: the key never appears in errors, telemetry or config dumps") {
    auto t = fake_with_tags();
    auto o = options_with(t);
    auto be = make_ollama_cloud_backend(o).value();
    auto model = load(be);
    const std::string echo = std::string("invalid key ") + kFakeKey;

    // An upstream that echoes the key back, as an HTTP error body ...
    t->route("/api/chat", Route{401, {R"({"error":")" + echo + R"("})"}, {}, false});
    auto a = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(a.ok());
    CHECK_FALSE(contains_key(a.status().message()));
    CHECK(a.status().message().find("[redacted]") != std::string::npos);
    // ... mid-stream ...
    t->route("/api/chat", Route{200, {R"({"error":")" + echo + "\"}\n"}, {}, false});
    auto b = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(b.ok());
    CHECK_FALSE(contains_key(b.status().message()));
    // ... in a non-JSON error body ...
    t->route("/api/chat", Route{500, {echo}, {}, false});
    auto c = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(c.ok());
    CHECK_FALSE(contains_key(c.status().message()));
    // ... or through a transport error, and on the model list path.
    t->route("/api/chat", Route{200, {}, Status(ErrorCode::io_error, "boom " + echo), false});
    auto d = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(d.ok());
    CHECK_FALSE(contains_key(d.status().message()));
    t->route("/api/tags", Route{403, {echo}, {}, false});
    auto e = be->list_models();
    REQUIRE_FALSE(e.ok());
    CHECK_FALSE(contains_key(e.status().message()));

    auto status = be->runtime_status();
    REQUIRE(status.has_value());
    for (const auto& [type, attrs] : status->take_events()) {
        CHECK_FALSE(contains_key(json::Value(attrs).dump()));
    }
    CHECK_FALSE(contains_key(json::Value(*status->diagnostics).dump()));
    CHECK_FALSE(contains_key(json::Value(to_json(*status)).dump()));
    const std::string dump = json::Value(redacted_config(o)).dump();
    CHECK_FALSE(contains_key(dump));
    CHECK(dump.find("SONDER_TEST_OLLAMA_CLOUD_KEY") != std::string::npos);
    CHECK_FALSE(contains_key(be->description()));
}

TEST_CASE("secrets: a key echoed across the 512-byte error cut leaves no prefix behind") {
    // The key starts 12 bytes before the cut, so a cut-then-redact order
    // would leak "sk-FAKE-test" (12 bytes of the key) into the message.
    const std::string key(kFakeKey);
    const std::string prefix = key.substr(0, 12);
    const std::string detail = std::string(500, 'x') + key + " trailing text";
    REQUIRE(detail.size() > 512u);
    const auto leaks = [&](const std::string& message) {
        return message.find(prefix) != std::string::npos || contains_key(message);
    };

    auto t = fake_with_tags();
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);

    // HTTP error path, JSON {"error": "..."} body.
    t->route("/api/chat", Route{401, {R"({"error":")" + detail + R"("})"}, {}, false});
    auto http_json = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(http_json.ok());
    CHECK_FALSE(leaks(http_json.status().message()));
    CHECK(http_json.status().message().find("[redacted]") != std::string::npos);

    // HTTP error path, raw (non-JSON) body, on the generate endpoint.
    t->route("/api/generate", Route{500, {detail}, {}, false});
    GenerateRequest gen;
    gen.prompt = "x";
    gen.sampling = SamplingConfig::greedy(4);
    auto http_raw = model->generate(gen, {}, {});
    REQUIRE_FALSE(http_raw.ok());
    CHECK_FALSE(leaks(http_raw.status().message()));
    CHECK(http_raw.status().message().find("[redacted]") != std::string::npos);

    // Mid-stream {"error": ...} line after a 200 status.
    t->route("/api/chat", Route{200,
                                {"{\"message\":{\"content\":\"a\"},\"done\":false}\n",
                                 R"({"error":")" + detail + "\"}\n"},
                                {},
                                false});
    auto mid = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(mid.ok());
    CHECK(mid.status().code() == ErrorCode::backend_error);
    CHECK_FALSE(leaks(mid.status().message()));
    CHECK(mid.status().message().find("[redacted]") != std::string::npos);

    // Buffered GET path (/api/tags behind list_models).
    t->route("/api/tags", Route{403, {R"({"error":")" + detail + R"("})"}, {}, false});
    auto tags = be->list_models();
    REQUIRE_FALSE(tags.ok());
    CHECK_FALSE(leaks(tags.status().message()));
    CHECK(tags.status().message().find("[redacted]") != std::string::npos);
}

TEST_CASE("secrets: a non-string error value re-serialised with JSON escapes is redacted") {
    // Printable ASCII, so ApiKey accepts it; '"' and '\' are escaped by dump().
    const std::string key = R"(sk-FAKE"quote\back-0123456789)";
    const std::string escaped = R"(sk-FAKE\"quote\\back-0123456789)";
    auto t = fake_with_tags();
    auto o = options_with(t);
    o.env_lookup = [key](const char* name) -> std::optional<std::string> {
        if (std::string(name) == "SONDER_TEST_OLLAMA_CLOUD_KEY") return key;
        return std::nullopt;
    };
    auto be = make_ollama_cloud_backend(o).value();
    auto model = load(be);
    // {"error": {"detail": "<key>"}}: the JSON text carries the escaped form.
    const std::string body = R"({"error":{"detail":")" + escaped + R"("}})";
    const auto leaks = [&](const std::string& message) {
        return message.find(key) != std::string::npos || message.find(escaped) != std::string::npos ||
               message.find("quote") != std::string::npos;
    };

    t->route("/api/chat", Route{400, {body}, {}, false});
    auto http = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(http.ok());
    CHECK_FALSE(leaks(http.status().message()));
    CHECK(http.status().message().find("[redacted]") != std::string::npos);

    t->route("/api/chat", Route{200, {body + "\n"}, {}, false});
    auto mid = model->chat(user_chat("x"), {}, {});
    REQUIRE_FALSE(mid.ok());
    CHECK_FALSE(leaks(mid.status().message()));
    CHECK(mid.status().message().find("[redacted]") != std::string::npos);

    // Direct: raw and escaped occurrences in one text.
    auto direct = ollama_cloud::ApiKey::from_value(std::string(key), "test");
    REQUIRE(direct.ok());
    std::string text = "a " + key + " b " + escaped + " c";
    direct.value().scrub(text);
    CHECK(text == "a [redacted] b [redacted] c");
}

TEST_CASE("secrets: end to end through the engine, no telemetry line carries the key") {
    auto t = fake_with_tags();
    t->route("/api/chat", Route{200, {kChatStream}, {}, false});
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::deep;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_gpu_devices = false;
    eo.device_sample_interval = std::chrono::milliseconds(20);
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(make_ollama_cloud_backend(options_with(t)).value()).ok());
    ModelLoadOptions lo;
    lo.model = "gpt-oss:120b";
    auto model = engine.load_model("ollama_cloud", lo);
    REQUIRE_MESSAGE(model.ok(), model.status().to_string());
    auto session = engine.create_session(model.value());
    REQUIRE(session.ok());

    RequestOptions local;
    local.local_only = true;
    auto refused = session.value()->chat({ChatMessage{"user", "hi"}}, {}, std::nullopt, local);
    CHECK((!refused.ok() || refused->outcome == RequestOutcome::failed));
    CHECK(t->count("/api/chat") == 0u);

    auto ok = session.value()->chat({ChatMessage{"user", "hi"}}, {}, std::nullopt, {});
    REQUIRE_MESSAGE(ok.ok(), ok.status().to_string());
    CHECK(ok->text == "Hello");
    CHECK(t->count("/api/chat") == 1u);

    // The engine sampler drains the provider's events onto the bus.
    bool saw_remote = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!saw_remote && std::chrono::steady_clock::now() < deadline) {
        for (const auto& line : sink->lines()) {
            if (line.find("backend.remote.request") != std::string::npos &&
                line.find("\"locality\":\"cloud\"") != std::string::npos) {
                saw_remote = true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(saw_remote);
    const auto lines = sink->lines();
    CHECK_FALSE(lines.empty());
    for (const auto& line : lines) {
        CHECK_FALSE(contains_key(line));
    }
}

TEST_CASE("cancellation: mid-stream cancel stops the request promptly") {
    auto t = fake_with_tags();
    t->route("/api/chat", Route{200, {"{\"message\":{\"content\":\"first\"},\"done\":false}\n"}, {}, true});
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);
    CancellationSource source;
    std::string got;
    std::thread canceller([&] {
        t->wait_blocked();
        source.cancel();
    });
    const auto start = std::chrono::steady_clock::now();
    auto r = model->chat(user_chat("long answer"), source.token(), [&](const TokenChunk& c) {
        got.append(c.text);
        return true;
    });
    canceller.join();
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::cancelled);
    CHECK(got == "first");
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));

    // Already cancelled: nothing is sent.
    const std::size_t before = t->count("/api/chat");
    CHECK(model->chat(user_chat("x"), source.token(), {}).status().code() == ErrorCode::cancelled);
    CHECK(t->count("/api/chat") == before);

    auto status = be->runtime_status();
    REQUIRE(status.has_value());
    auto events = status->take_events();
    REQUIRE_FALSE(events.empty());
    CHECK(events.front().second.find("outcome")->as_string() == "cancelled");
    CHECK(status->diagnostics->find("cancelled")->as_uint() == 2u);
}

TEST_CASE("telemetry: bounded queue drops and counts") {
    auto t = fake_with_tags();
    auto be = make_ollama_cloud_backend(options_with(t)).value();
    auto model = load(be);
    auto local = user_chat("x");
    local.local_only = true;
    for (int i = 0; i < 300; ++i) {
        (void)model->chat(local, {}, {});
    }
    auto status = be->runtime_status();
    REQUIRE(status.has_value());
    CHECK(status->take_events().size() == 256u);
    CHECK(be->runtime_status()->diagnostics->find("telemetry_dropped")->as_uint() == 44u);
    CHECK(be->runtime_status()->take_events().empty());
}

TEST_CASE("https transport: refuses cleanly in a build without TLS") {
    if (net::kTlsCompiledIn) {
        // Real TLS would need a network peer; covered by tests/tls for the client.
        return;
    }
    auto transport = make_ollama_cloud_https_transport();
    OllamaCloudHttpRequest req;
    req.method = "GET";
    req.host = "ollama.com";
    req.target = "/api/tags";
    const Status st = transport->send(req, {}, [](std::string_view) { return true; }, {});
    CHECK(st.code() == ErrorCode::unsupported);
    CHECK(st.message().find("SONDER_WITH_TLS") != std::string::npos);
}

TEST_CASE("protocol: decoder bounds and tags parsing") {
    ollama_cloud::StreamDecoder big(ollama_cloud::StreamKind::chat, {}, 4);
    CHECK_FALSE(big.feed("{\"message\":{\"content\":\"too long\"},\"done\":false}\n"));
    CHECK(big.finish().code() == ErrorCode::protocol_error);

    ollama_cloud::StreamDecoder stop(ollama_cloud::StreamKind::generate,
                                     [](const ollama_cloud::StreamEvent&) { return false; }, 0);
    CHECK_FALSE(stop.feed("{\"response\":\"a\",\"done\":false}\n"));
    CHECK(stop.stopped_by_callback());
    CHECK(stop.finish().ok());

    auto tags = ollama_cloud::parse_tags(kTagsBody);
    REQUIRE(tags.ok());
    CHECK(tags->size() == 3u);
    CHECK_FALSE(ollama_cloud::parse_tags("[]").ok());
    CHECK(ollama_cloud::parse_tags("{}").value().empty());
    CHECK(ollama_cloud::http_status_error(404, "{\"error\":\"model not found\"}").code() == ErrorCode::not_found);
}
