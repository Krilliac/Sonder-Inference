// Attach mode against a slow OpenAI-compatible upstream (docs/integration/
// colibri.md): bearer token from the environment, idle / first-content /
// progress-stall timeouts, header-injection refusal and generic thinking
// fields. A loopback cpp-httplib server scripts the upstream; there is no
// network service, model, or third-party runtime involved.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fake_server.hpp"
#include "net/http_client.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/backends/llamaserver.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

namespace {

constexpr const char *kTokenVar = "SONDER_TEST_SLOW_UPSTREAM_KEY";
constexpr const char *kTokenValue = "tok-4f9c2a71-not-a-real-secret";

class ScopedEnv {
  public:
    ScopedEnv(std::string name, std::optional<std::string> value) : name_(std::move(name)) {
        if (const char *old = std::getenv(name_.c_str()))
            previous_ = old;
        apply(value);
    }
    ~ScopedEnv() { apply(previous_); }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

  private:
    void apply(const std::optional<std::string> &value) const {
#if defined(_WIN32)
        (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
        if (value)
            (void)setenv(name_.c_str(), value->c_str(), 1);
        else
            (void)unsetenv(name_.c_str());
#endif
    }
    std::string name_;
    std::optional<std::string> previous_;
};

// Generic empty delta (some other upstreams send this shape).
const std::string kKeepalive = "data: {\"choices\":[{\"delta\":{}}]}\n\n";
// colibri's real chat keepalive (openai_server.py _keepalive_choice): an empty
// reasoning_content delta. It must never count as progress.
const std::string kColibriKeepalive = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\"}}]}\n\n";
// What COLI_VISIBLE_KEEPALIVE=1 sends instead: non-empty, so it IS progress.
const std::string kColibriVisibleKeepalive = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\".\"}}]}\n\n";
std::string content(const std::string &text) {
    return "data: {\"choices\":[{\"delta\":{\"content\":\"" + text + "\"}}]}\n\n";
}
const std::string kFinish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

struct Step {
    std::chrono::milliseconds wait;
    std::string bytes;
};

// Scripted OpenAI-compatible upstream. Each step sleeps, then writes; after the
// script, `hold_open` keeps the connection silent (a stalled upstream) until
// the client disconnects or the server stops.
class SlowUpstream {
  public:
    SlowUpstream() {
        svr_.Get("/v1/models", [this](const httplib::Request &req, httplib::Response &res) {
            if (!authorized(req, res))
                return;
            res.set_content(R"({"object":"list","data":[{"id":"slow-model","object":"model"}]})",
                            "application/json");
        });
        svr_.Post("/v1/chat/completions", [this](const httplib::Request &req, httplib::Response &res) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                last_body_ = req.body;
                ++chat_requests_;
            }
            if (!authorized(req, res))
                return;
            std::vector<Step> steps;
            bool hold = false;
            {
                std::lock_guard<std::mutex> lock(mu_);
                steps = steps_;
                hold = hold_open_;
            }
            res.set_chunked_content_provider(
                "text/event-stream", [this, steps, hold](std::size_t, httplib::DataSink &sink) {
                    for (const auto &step : steps) {
                        const auto until = std::chrono::steady_clock::now() + step.wait;
                        while (std::chrono::steady_clock::now() < until) {
                            if (stopping_.load() || !sink.is_writable())
                                return false;
                            std::this_thread::sleep_for(10ms);
                        }
                        if (!sink.write(step.bytes.data(), step.bytes.size()))
                            return false;
                    }
                    if (hold) {
                        // Silent for up to 10 s unless the client goes away.
                        for (int i = 0; i < 1000 && !stopping_.load(); ++i) {
                            if (!sink.is_writable())
                                return false;
                            std::this_thread::sleep_for(10ms);
                        }
                        return false;
                    }
                    sink.done();
                    return true;
                });
        });
        port_ = svr_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0)
            throw std::runtime_error("slow upstream bind failed");
        thread_ = std::thread([this] { svr_.listen_after_bind(); });
        svr_.wait_until_ready();
    }
    ~SlowUpstream() {
        stopping_.store(true);
        svr_.stop();
        if (thread_.joinable())
            thread_.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void script(std::vector<Step> steps, bool hold_open) {
        std::lock_guard<std::mutex> lock(mu_);
        steps_ = std::move(steps);
        hold_open_ = hold_open;
    }
    void require_token(std::string token) {
        std::lock_guard<std::mutex> lock(mu_);
        token_ = std::move(token);
    }
    std::string last_body() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_body_;
    }
    int chat_requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return chat_requests_;
    }
    int unauthorized_count() const { return unauthorized_.load(); }
    int total_requests() const { return total_.load(); }

  private:
    bool authorized(const httplib::Request &req, httplib::Response &res) {
        total_.fetch_add(1);
        std::string token;
        {
            std::lock_guard<std::mutex> lock(mu_);
            token = token_;
        }
        if (token.empty() || req.get_header_value("Authorization") == "Bearer " + token)
            return true;
        unauthorized_.fetch_add(1);
        res.status = 401;
        res.set_content(R"({"error":"unauthorized"})", "application/json");
        return false;
    }

    httplib::Server svr_;
    std::thread thread_;
    int port_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<int> unauthorized_{0};
    std::atomic<int> total_{0};
    mutable std::mutex mu_;
    std::vector<Step> steps_;
    bool hold_open_ = false;
    std::string token_;
    std::string last_body_;
    int chat_requests_ = 0;
};

LlamaServerBackendOptions slow_options(const SlowUpstream &upstream) {
    LlamaServerBackendOptions o;
    o.base_url = upstream.url();
    o.native_completion = false;
    o.slot_affinity = false;
    // The scripted upstream sends its "keepalives" every 100 ms, so declare
    // that gap rather than colibri's 10 s to keep the tests fast.
    o.keepalive_gap = 100ms;
    o.request_timeout = 30s;
    return o;
}

ChatRequest hello() {
    ChatRequest r;
    r.messages = {{"user", "hi"}};
    return r;
}

struct Outcome {
    Status status;
    std::string text;
    std::chrono::milliseconds elapsed{0};
};

Outcome run_chat(const LlamaServerBackendOptions &options, const ChatRequest &request = hello()) {
    Outcome out;
    auto backend = make_llamaserver_backend(options);
    auto model = backend->load_model({"slow-model", "cpu:0"});
    if (!model.ok()) {
        out.status = model.status();
        return out;
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = model.value()->chat(request, {}, [&](const TokenChunk &chunk) {
        out.text += chunk.text;
        return true;
    });
    out.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    out.status = result.ok() ? Status::success() : result.status();
    return out;
}

} // namespace

TEST_SUITE("llamaserver slow upstream") {
    TEST_CASE("bearer token comes from the environment, is required, and never leaks") {
        ScopedEnv env(kTokenVar, std::string(kTokenValue));
        SlowUpstream upstream;
        upstream.require_token(kTokenValue);
        upstream.script({{0ms, content("ok")}, {0ms, kFinish}}, false);

        auto options = slow_options(upstream);
        options.api_key_env = kTokenVar;
        auto backend = make_llamaserver_backend(options);
        auto model = backend->load_model({"slow-model", "cpu:0"});
        REQUIRE_MESSAGE(model.ok(), model.status().to_string());
        std::string text;
        auto result = model.value()->chat(hello(), {}, [&](const TokenChunk &c) {
            text += c.text;
            return true;
        });
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(text == "ok");
        CHECK(upstream.unauthorized_count() == 0);
        CHECK(upstream.total_requests() >= 2); // /v1/models and the completion

        // Runtime status reports the variable name and a redacted value only.
        const auto status = backend->runtime_status();
        REQUIRE(status.has_value());
        const auto rendered = json::Value(to_json(*status)).dump();
        CHECK(rendered.find(kTokenVar) != std::string::npos);
        CHECK(rendered.find("[redacted]") != std::string::npos);
        CHECK(rendered.find(kTokenValue) == std::string::npos);

        // Without the header the upstream answers 401 and the error is clean.
        auto anonymous = slow_options(upstream);
        const auto before = upstream.unauthorized_count();
        const auto refused = run_chat(anonymous);
        CHECK_FALSE(refused.status.ok());
        CHECK(upstream.unauthorized_count() > before);
        CHECK(refused.status.to_string().find(kTokenValue) == std::string::npos);
        CHECK(refused.status.to_string().find("401") != std::string::npos);
    }

    TEST_CASE("a missing, malformed or misplaced token variable is refused at build time") {
        SlowUpstream upstream;
        {
            ScopedEnv env(kTokenVar, std::nullopt);
            auto options = slow_options(upstream);
            options.api_key_env = kTokenVar;
            const auto outcome = run_chat(options);
            CHECK(outcome.status.code() == ErrorCode::invalid_argument);
            CHECK(upstream.total_requests() == 0);
        }
        {
            ScopedEnv env(kTokenVar, std::string("two words"));
            auto options = slow_options(upstream);
            options.api_key_env = kTokenVar;
            const auto outcome = run_chat(options);
            CHECK(outcome.status.code() == ErrorCode::invalid_argument);
            CHECK(outcome.status.to_string().find("two words") == std::string::npos);
        }
        {
            ScopedEnv env(kTokenVar, std::string(kTokenValue));
            auto options = slow_options(upstream);
            options.api_key_env = "not a name";
            CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
            options.api_key_env = kTokenVar;
            options.mode = LlamaServerMode::spawn;
            options.executable = "never-started.exe";
            CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        }
    }

    TEST_CASE("timeout configuration is validated") {
        SlowUpstream upstream;
        auto options = slow_options(upstream);
        options.keepalive_gap = 10s;
        options.idle_timeout = 10s; // not above the keepalive gap
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        options.idle_timeout = 30s;
        options.unlimited_request_timeout = true; // no first-content bound
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        options.first_content_timeout = 600s;
        upstream.script({{0ms, content("ok")}, {0ms, kFinish}}, false);
        const auto ok = run_chat(options);
        CHECK_MESSAGE(ok.status.ok(), ok.status.to_string());
        options.stall_guard.policy = "cancel"; // cancel needs generic_progress
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        options.stall_guard.generic_progress = true;
        CHECK_MESSAGE(run_chat(options).status.ok(), "cancel policy with generic_progress");
        options.grammar = "root ::= \"x\"";
        options.upstream_grammar = false;
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);

        // generic_progress needs the stall guard to be enabled.
        options.grammar.clear();
        options.upstream_grammar = true;
        options.stall_guard.enabled = false;
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        options.stall_guard.enabled = true;
        CHECK_MESSAGE(run_chat(options).status.ok(), "generic_progress with the guard enabled");
    }

    TEST_CASE("unlimited_request_timeout needs an idle bound") {
        SlowUpstream upstream;
        upstream.script({{0ms, content("ok")}, {0ms, kFinish}}, false);
        auto options = slow_options(upstream);
        options.unlimited_request_timeout = true;
        options.first_content_timeout = 600s; // idle_timeout left at 0
        auto refused = run_chat(options);
        CHECK(refused.status.code() == ErrorCode::invalid_argument);
        CHECK(refused.status.message().find("idle_timeout") != std::string::npos);
        CHECK(upstream.chat_requests() == 0);
        options.stall_guard.generic_progress = true; // strict rule: cancel does not replace idle
        options.stall_guard.policy = "cancel";
        CHECK(run_chat(options).status.code() == ErrorCode::invalid_argument);
        options.idle_timeout = 30s; // above keepalive_gap (100 ms in slow_options)
        const auto ok = run_chat(options);
        CHECK_MESSAGE(ok.status.ok(), ok.status.to_string());
    }

    TEST_CASE("keepalive deltas keep the idle timeout quiet while slow content still completes") {
        SlowUpstream upstream;
        // 15 keepalives 100 ms apart, then content, all within a 1 s idle bound
        // per gap even though the whole response takes ~1.8 s. Only meaningful
        // together with the mid-way stall case, which proves the bound does fire.
        std::vector<Step> steps;
        for (int i = 0; i < 15; ++i)
            steps.push_back({100ms, kKeepalive});
        steps.push_back({150ms, content("slow")});
        steps.push_back({150ms, content("ly")});
        steps.push_back({0ms, kFinish});
        upstream.script(std::move(steps), false);
        auto options = slow_options(upstream);
        options.idle_timeout = 1s;
        options.first_content_timeout = 5s;
        options.unlimited_request_timeout = true;
        const auto outcome = run_chat(options);
        REQUIRE_MESSAGE(outcome.status.ok(), outcome.status.to_string());
        CHECK(outcome.text == "slowly");
        CHECK(outcome.elapsed >= 1500ms); // the stream outlived the idle bound
    }

    TEST_CASE("first-content timeout fires on keepalives alone, not on bytes") {
        SlowUpstream upstream;
        std::vector<Step> steps;
        for (int i = 0; i < 40; ++i)
            steps.push_back({100ms, kKeepalive});
        upstream.script(std::move(steps), false);
        auto options = slow_options(upstream);
        options.idle_timeout = 1500ms; // never trips: bytes arrive every 100 ms
        options.first_content_timeout = 600ms;
        const auto outcome = run_chat(options);
        CHECK(outcome.status.code() == ErrorCode::protocol_error);
        CHECK(outcome.status.message().find("first-content") != std::string::npos);
        CHECK(outcome.elapsed < 3s);
        CHECK(outcome.text.empty());
    }

    TEST_CASE("colibri's real keepalive (empty reasoning_content) is not progress, non-empty is") {
        for (const bool separate : {false, true}) {
            CAPTURE(separate);
            // Keepalives alone: the first-content bound must still fire.
            {
                SlowUpstream upstream;
                std::vector<Step> steps;
                for (int i = 0; i < 40; ++i)
                    steps.push_back({100ms, kColibriKeepalive});
                upstream.script(std::move(steps), false);
                auto options = slow_options(upstream);
                options.idle_timeout = 1500ms;
                options.first_content_timeout = 600ms;
                auto request = hello();
                request.separate_reasoning = separate;
                const auto outcome = run_chat(options, request);
                CHECK(outcome.status.code() == ErrorCode::protocol_error);
                CHECK(outcome.status.message().find("first-content") != std::string::npos);
                CHECK(outcome.elapsed < 3s);
                CHECK(outcome.text.empty());
            }
            // Content once, then only real keepalives: the stall watchdog cancels.
            {
                SlowUpstream upstream;
                std::vector<Step> steps{{50ms, content("a")}};
                for (int i = 0; i < 40; ++i)
                    steps.push_back({100ms, kColibriKeepalive});
                upstream.script(std::move(steps), false);
                auto options = slow_options(upstream);
                options.idle_timeout = 5s;
                options.stall_guard.generic_progress = true;
                options.stall_guard.stall_seconds = 1s;
                options.stall_guard.policy = "cancel";
                auto request = hello();
                request.separate_reasoning = separate;
                const auto outcome = run_chat(options, request);
                CHECK(outcome.status.code() == ErrorCode::protocol_error);
                CHECK(outcome.status.message().find("stall") != std::string::npos);
                CHECK(outcome.text == "a");
            }
            // Positive control: the COLI_VISIBLE_KEEPALIVE shape counts as
            // progress, so the same bounds do NOT fire on it.
            {
                SlowUpstream upstream;
                std::vector<Step> steps;
                for (int i = 0; i < 15; ++i)
                    steps.push_back({100ms, kColibriVisibleKeepalive});
                steps.push_back({0ms, content("done")});
                steps.push_back({0ms, kFinish});
                upstream.script(std::move(steps), false);
                auto options = slow_options(upstream);
                options.idle_timeout = 1500ms;
                options.first_content_timeout = 600ms; // stream lasts ~1.5 s
                auto request = hello();
                request.separate_reasoning = separate;
                const auto outcome = run_chat(options, request);
                CHECK_MESSAGE(outcome.status.ok(), outcome.status.to_string());
                CHECK(outcome.text == "done");
                CHECK(outcome.elapsed >= 1200ms);
            }
            // Same control against the stall watchdog: visible keepalives are
            // progress, so the 1 s stall bound must not cancel a ~1.5 s stream.
            {
                SlowUpstream upstream;
                std::vector<Step> steps;
                for (int i = 0; i < 15; ++i)
                    steps.push_back({100ms, kColibriVisibleKeepalive});
                steps.push_back({0ms, content("done")});
                steps.push_back({0ms, kFinish});
                upstream.script(std::move(steps), false);
                auto options = slow_options(upstream);
                options.idle_timeout = 5s;
                options.first_content_timeout = 5s;
                options.stall_guard.generic_progress = true;
                options.stall_guard.stall_seconds = 1s;
                options.stall_guard.policy = "cancel";
                auto request = hello();
                request.separate_reasoning = separate;
                const auto outcome = run_chat(options, request);
                CHECK_MESSAGE(outcome.status.ok(), outcome.status.to_string());
                CHECK(outcome.text == "done");
                CHECK(outcome.elapsed >= 1200ms);
            }
        }
    }

    TEST_CASE("reasoning text counts as first content") {
        SlowUpstream upstream;
        const std::string thinking = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n";
        upstream.script({{100ms, kKeepalive}, {100ms, thinking}, {2000ms, content("done")}, {0ms, kFinish}}, false);
        auto options = slow_options(upstream);
        options.first_content_timeout = 1500ms; // content arrives at ~2.2 s, reasoning at ~200 ms
        options.idle_timeout = 5s;
        const auto outcome = run_chat(options);
        CHECK_MESSAGE(outcome.status.ok(), outcome.status.to_string());
        CHECK(outcome.text == "done");
    }

    TEST_CASE("a stream that stalls mid-way ends with a clean error from the idle timeout") {
        SlowUpstream upstream;
        upstream.script({{50ms, content("partial")}}, true); // then silent
        auto options = slow_options(upstream);
        options.idle_timeout = 400ms;
        options.first_content_timeout = 2s;
        options.unlimited_request_timeout = true;
        const auto outcome = run_chat(options);
        CHECK(outcome.status.code() == ErrorCode::protocol_error);
        CHECK(outcome.status.message().find("idle timeout") != std::string::npos);
        CHECK(outcome.text == "partial");
        CHECK(outcome.elapsed >= 350ms);
        CHECK(outcome.elapsed < 4s);
    }

    TEST_CASE("generic progress watchdog: cancel ends a keepalive-only stall, warn reports it") {
        SlowUpstream upstream;
        // Content once, then only keepalives: bytes keep the idle timeout quiet.
        std::vector<Step> steps{{50ms, content("a")}};
        for (int i = 0; i < 40; ++i)
            steps.push_back({100ms, kKeepalive});
        upstream.script(steps, false);
        auto options = slow_options(upstream);
        options.idle_timeout = 5s; // not under test here; wide so a scheduling hiccup cannot trip it
        options.stall_guard.generic_progress = true;
        options.stall_guard.stall_seconds = 1s;
        options.stall_guard.policy = "cancel";
        const auto cancelled = run_chat(options);
        CHECK(cancelled.status.code() == ErrorCode::protocol_error);
        CHECK(cancelled.status.message().find("stall") != std::string::npos);
        CHECK(cancelled.text == "a");

        // Warn: the stream is left running (here to its natural end) and the
        // stall surfaces as a runtime warning.
        std::vector<Step> warn_steps{{50ms, content("a")}};
        for (int i = 0; i < 14; ++i)
            warn_steps.push_back({100ms, kKeepalive});
        warn_steps.push_back({0ms, content("b")});
        warn_steps.push_back({0ms, kFinish});
        upstream.script(std::move(warn_steps), false);
        options.stall_guard.policy = "warn";
        auto backend = make_llamaserver_backend(options);
        auto model = backend->load_model({"slow-model", "cpu:0"});
        REQUIRE(model.ok());
        std::string text;
        auto result = model.value()->chat(hello(), {}, [&](const TokenChunk &c) {
            text += c.text;
            return true;
        });
        REQUIRE_MESSAGE(result.ok(), result.status().to_string());
        CHECK(text == "ab");
        const auto status = backend->runtime_status();
        REQUIRE(status.has_value());
        bool warned = false;
        for (const auto &warning : status->warnings)
            warned = warned || warning.code == "generic_stream_stalled";
        CHECK(warned);
        REQUIRE(status->stall.has_value());
    }

    TEST_CASE("header injection is refused before anything is sent") {
        SlowUpstream upstream;
        const auto port = static_cast<std::uint16_t>(std::stoi(upstream.url().substr(upstream.url().rfind(':') + 1)));
        const std::vector<std::pair<std::string, std::string>> bad[] = {
            {{"X-Test", "ok\r\nX-Injected: 1"}},
            {{"X-Test", "ok\nX-Injected: 1"}},
            {{"X-Test", std::string("nul\0byte", 8)}},
            {{"X-Test\r\nX-Injected", "1"}},
            {{"Bad Name", "1"}},
            {{"", "1"}},
            {{"Content-Length", "0"}},
            {{"Host", "evil.example"}},
        };
        for (const auto &headers : bad) {
            sonder::inference::net::HttpRequest request;
            request.host = "127.0.0.1";
            request.port = port;
            request.target = "/v1/models";
            request.headers = headers;
            std::string body;
            const auto result = sonder::inference::net::http_request_buffered(request, body);
            REQUIRE_FALSE(result.ok());
            CHECK(result.status().code() == ErrorCode::invalid_argument);
        }
        // Supporting evidence only: the invalid_argument checks above are the proof.
        CHECK(upstream.total_requests() == 0);

        // A valid extra header goes through, and credential values are redacted
        // in diagnostics.
        sonder::inference::net::HttpRequest request;
        request.host = "127.0.0.1";
        request.port = port;
        request.target = "/v1/models";
        request.headers = {{"X-Request-Tag", "alpha"}};
        std::string body;
        const auto result = sonder::inference::net::http_request_buffered(request, body);
        REQUIRE(result.ok());
        CHECK(result->status == 200);
        CHECK(sonder::inference::net::redacted_header_value("Authorization", "Bearer abc") == "[redacted]");
        CHECK(sonder::inference::net::redacted_header_value("X-Api-Key", "abc") == "[redacted]");
        CHECK(sonder::inference::net::redacted_header_value("X-Request-Tag", "alpha") == "alpha");
    }

    TEST_CASE("generic mode sends thinking controls top-level and no chat_template_kwargs") {
        SlowUpstream upstream;
        upstream.script({{0ms, content("ok")}, {0ms, kFinish}}, false);
        auto options = slow_options(upstream);
        auto request = hello();
        request.thinking.enable_thinking = false;
        request.thinking.reasoning_effort = "low";
        const auto outcome = run_chat(options, request);
        REQUIRE_MESSAGE(outcome.status.ok(), outcome.status.to_string());
        auto sent = json::parse(upstream.last_body());
        REQUIRE(sent.ok());
        const auto *enable = sent->find("enable_thinking");
        REQUIRE(enable);
        CHECK_FALSE(enable->as_bool(true));
        REQUIRE(sent->find("reasoning_effort"));
        CHECK(sent->find("reasoning_effort")->as_string() == "low");
        CHECK(sent->find("chat_template_kwargs") == nullptr);
        CHECK(sent->find("cache_prompt") == nullptr);
    }
}
