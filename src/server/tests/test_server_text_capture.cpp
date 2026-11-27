// Telemetry text-capture policy at the server: discovery declaration, the
// --text-capture option and its environment default, the startup notice, and
// the X-Sonder-No-Capture request header.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/text_capture.hpp"
#include "src/openai.hpp"
#include "src/serve_options.hpp"

using namespace server_test;
namespace json = sonder::inference::json;
namespace fs = std::filesystem;

namespace {

// Fake bearer token (tests only).
constexpr const char* kToken = "test-token-not-a-secret";

const std::vector<std::pair<std::string, std::string>> kAuth = {{"Authorization", std::string("Bearer ") + kToken}};

srv::ServerOptions capture_options(si::TextCapturePolicy policy) {
    auto o = Fixture::defaults();
    o.token = kToken;
    o.text_capture = policy;
    return o;
}

const json::Value* attr(const json::Value& event, const char* name) {
    return event.find("attributes")->find(name);
}

std::vector<std::pair<std::string, std::string>> with_headers(
    std::vector<std::pair<std::string, std::string>> extra) {
    auto h = kAuth;
    for (auto& e : extra) h.push_back(std::move(e));
    return h;
}

// Sets an environment variable for one scope and restores the old state.
struct ScopedEnv {
    std::string name;
    std::string old;
    bool had = false;
    ScopedEnv(std::string n, const char* value) : name(std::move(n)) {
        if (const char* v = std::getenv(name.c_str())) {
            had = true;
            old = v;
        }
        set(value);
    }
    ~ScopedEnv() { set(had ? old.c_str() : nullptr); }
    void set(const char* value) const {
#if defined(_WIN32)
        _putenv_s(name.c_str(), value != nullptr ? value : "");
#else
        if (value != nullptr) {
            setenv(name.c_str(), value, 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }
};

// When set, build() adds `--token-file <path>` (on/full need a token).
std::string g_token_file;

int build(std::vector<std::string> args, det::ServeCliConfig& cfg, std::string& err_text) {
    if (!g_token_file.empty()) {
        args.push_back("--token-file");
        args.push_back(g_token_file);
    }
    det::ParsedServeArgs parsed;
    std::string parse_err;
    REQUIRE_MESSAGE(det::parse_serve_args(args, parsed, parse_err), parse_err);
    std::ostringstream err;
    auto log = std::make_shared<det::LockedLog>(err);
    const int rc = det::build_serve_options(parsed, log, cfg);
    log->close();
    err_text = err.str();
    return rc;
}

// Runs a chat request with a run id and returns the request's events.
struct RunEvents {
    std::vector<json::Value> started;
    std::vector<json::Value> tokens;
};

RunEvents chat_with(Fixture& f, const std::string& run_id,
                    std::vector<std::pair<std::string, std::string>> extra_headers = {}) {
    extra_headers.push_back({"X-Sonder-Run-Id", run_id});
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)", "pangolin zeppelin"),
                         with_headers(std::move(extra_headers)));
    REQUIRE_MESSAGE(r.status == 200, r.body);
    RunEvents out;
    for (auto& e : f.of_type("request.started")) {
        if (e.find("run_id")->as_string() == run_id) out.started.push_back(e);
    }
    for (auto& e : f.of_type("inference.token.generated")) {
        if (e.find("run_id")->as_string() == run_id) out.tokens.push_back(e);
    }
    return out;
}

const json::Value* discovery_capture(const json::Value& d) { return d.find("text_capture"); }

}  // namespace

TEST_CASE("text capture server: discovery declares the policy next to sampling_level") {
    for (const auto policy : {si::TextCapturePolicy::off, si::TextCapturePolicy::on, si::TextCapturePolicy::full}) {
        CAPTURE(si::to_string(policy));
        Fixture f(capture_options(policy));
        const Reply r = get(f.port, "/.well-known/sonder-telemetry", kAuth);
        REQUIRE(r.status == 200);
        const json::Value d = r.json();
        REQUIRE(d.find("sampling_level") != nullptr);
        REQUIRE(discovery_capture(d) != nullptr);
        CHECK(discovery_capture(d)->as_string() == si::to_string(policy));
    }
}

TEST_CASE("text capture server: the legacy capture_text option is declared as on") {
    auto o = Fixture::defaults();
    o.token = kToken;
    o.capture_text = true;
    Fixture f(o);
    const json::Value d = get(f.port, "/.well-known/sonder-telemetry", kAuth).json();
    CHECK(discovery_capture(d)->as_string() == "on");
}

TEST_CASE("text capture server: the default is off and nothing is captured") {
    Fixture f;
    const json::Value d = get(f.port, "/.well-known/sonder-telemetry").json();
    CHECK(discovery_capture(d)->as_string() == "off");
    // Plain requests (no token needed with no capture) carry no text anywhere.
    const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)", "pangolin zeppelin"),
                         {{"X-Sonder-Run-Id", "plain-1"}});
    REQUIRE(r.status == 200);
    for (const auto& e : f.of_type("inference.token.generated")) CHECK(attr(e, "text") == nullptr);
    for (const auto& e : f.of_type("request.started")) CHECK(attr(e, "prompt") == nullptr);
}

TEST_CASE("text capture server: on exports output text, full also the prompt") {
    {
        Fixture f(capture_options(si::TextCapturePolicy::on));
        const RunEvents ev = chat_with(f, "cap-on");
        REQUIRE(ev.started.size() == 1);
        CHECK(attr(ev.started[0], "prompt") == nullptr);
        REQUIRE_FALSE(ev.tokens.empty());
        for (const auto& e : ev.tokens) CHECK(attr(e, "text") != nullptr);
    }
    {
        Fixture f(capture_options(si::TextCapturePolicy::full));
        const RunEvents ev = chat_with(f, "cap-full");
        REQUIRE(ev.started.size() == 1);
        REQUIRE(attr(ev.started[0], "prompt") != nullptr);
        CHECK(attr(ev.started[0], "prompt")->as_string().find("pangolin zeppelin") != std::string::npos);
        REQUIRE_FALSE(ev.tokens.empty());
        for (const auto& e : ev.tokens) CHECK(attr(e, "text") != nullptr);
    }
}

TEST_CASE("text capture server: X-Sonder-No-Capture forces off for that request only") {
    Fixture f(capture_options(si::TextCapturePolicy::full));
    const RunEvents quiet = chat_with(f, "quiet-1", {{"X-Sonder-No-Capture", "1"}});
    REQUIRE(quiet.started.size() == 1);
    CHECK(attr(quiet.started[0], "prompt") == nullptr);
    REQUIRE(attr(quiet.started[0], "text_capture_opt_out") != nullptr);
    REQUIRE_FALSE(quiet.tokens.empty());
    for (const auto& e : quiet.tokens) CHECK(attr(e, "text") == nullptr);

    // "true" works too, and "0" is an explicit no-op.
    const RunEvents quiet_true = chat_with(f, "quiet-2", {{"X-Sonder-No-Capture", "true"}});
    for (const auto& e : quiet_true.tokens) CHECK(attr(e, "text") == nullptr);
    const RunEvents loud = chat_with(f, "loud-1", {{"X-Sonder-No-Capture", "0"}});
    REQUIRE(loud.started.size() == 1);
    CHECK(attr(loud.started[0], "prompt") != nullptr);
    for (const auto& e : loud.tokens) CHECK(attr(e, "text") != nullptr);

    // The next request without the header is captured again.
    const RunEvents after = chat_with(f, "after-1");
    CHECK(attr(after.started[0], "prompt") != nullptr);
}

TEST_CASE("text capture server: X-Sonder-No-Capture on a streaming request") {
    Fixture f(capture_options(si::TextCapturePolicy::on));
    const Reply r = post(f.port, "/v1/chat/completions",
                         chat_body(R"(,"max_tokens":4,"stream":true)", "pangolin zeppelin"),
                         with_headers({{"X-Sonder-Run-Id", "stream-quiet"}, {"X-Sonder-No-Capture", "1"}}));
    REQUIRE(r.status == 200);
    std::size_t tokens = 0;
    for (const auto& e : f.of_type("inference.token.generated")) {
        if (e.find("run_id")->as_string() != "stream-quiet") continue;
        ++tokens;
        CHECK(attr(e, "text") == nullptr);
    }
    CHECK(tokens > 0);
}

TEST_CASE("text capture server: a malformed or repeated opt-out header is a 400") {
    Fixture f;
    for (const char* bad : {"yes", "", "2", "on"}) {
        CAPTURE(bad);
        const Reply r = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)"),
                             {{"X-Sonder-No-Capture", bad}});
        CHECK(r.status == 400);
        CHECK(r.body.find("invalid_correlation_header") != std::string::npos);
    }
    // Two copies are ambiguous for a privacy header: refused, not guessed.
    const Reply dup = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)"),
                           {{"X-Sonder-No-Capture", "0"}, {"X-Sonder-No-Capture", "1"}});
    CHECK(dup.status == 400);
}

TEST_CASE("text capture server: parse_correlation reads the header") {
    det::RequestHead head;
    const auto plain = det::parse_correlation(head);
    REQUIRE(std::holds_alternative<det::Correlation>(plain));
    CHECK_FALSE(std::get<det::Correlation>(plain).no_text_capture);
    for (const auto& [value, expected] :
         std::vector<std::pair<std::string, bool>>{{"1", true}, {"true", true}, {"0", false}, {"false", false}}) {
        CAPTURE(value);
        det::RequestHead h;
        h.headers.push_back({"x-sonder-no-capture", value});
        const auto parsed = det::parse_correlation(h);
        REQUIRE(std::holds_alternative<det::Correlation>(parsed));
        CHECK(std::get<det::Correlation>(parsed).no_text_capture == expected);
    }
    det::RequestHead bad;
    bad.headers.push_back({"x-sonder-no-capture", "maybe"});
    CHECK(std::holds_alternative<det::ApiError>(det::parse_correlation(bad)));
    det::RequestHead twice;
    twice.headers.push_back({"x-sonder-no-capture", "0"});
    twice.headers.push_back({"x-sonder-no-capture", "1"});
    CHECK(std::holds_alternative<det::ApiError>(det::parse_correlation(twice)));
}

TEST_CASE("text capture server: --text-capture parsing, environment default and conflicts") {
    // on/full need a token; a fake one in a temp file (never a real secret).
    const auto token_path = fs::temp_directory_path() / (si::make_id("sonder-capture-opts") + "-token");
    {
        std::ofstream(token_path) << "tok-capture-opts\n";
    }
    fs::permissions(token_path, fs::perms::owner_read | fs::perms::owner_write);
    g_token_file = token_path.string();
    struct Reset {
        ~Reset() { g_token_file.clear(); }
    } reset_token_file;
    std::string err;
    {
        det::ServeCliConfig c;
        REQUIRE(build({"--backend", "mock"}, c, err) == 0);
        CHECK(c.options.text_capture == si::TextCapturePolicy::off);
        CHECK_FALSE(c.options.capture_text);
    }
    for (const auto& [value, expected] : std::vector<std::pair<std::string, si::TextCapturePolicy>>{
             {"off", si::TextCapturePolicy::off}, {"on", si::TextCapturePolicy::on},
             {"full", si::TextCapturePolicy::full}, {"FULL", si::TextCapturePolicy::full}}) {
        CAPTURE(value);
        det::ServeCliConfig c;
        REQUIRE(build({"--backend", "mock", "--text-capture", value}, c, err) == 0);
        CHECK(c.options.text_capture == expected);
    }
    {
        det::ServeCliConfig c;
        CHECK(build({"--backend", "mock", "--text-capture", "everything"}, c, err) == 2);
        CHECK(err.find("--text-capture must be off, on or full") != std::string::npos);
    }
    {
        det::ServeCliConfig c;
        REQUIRE(build({"--backend", "mock", "--capture-text", "--text-capture", "full"}, c, err) == 0);
        CHECK(c.options.text_capture == si::TextCapturePolicy::full);
        CHECK(c.options.capture_text);
    }
    {
        det::ServeCliConfig c;
        CHECK(build({"--backend", "mock", "--capture-text", "--text-capture", "off"}, c, err) == 2);
        CHECK(err.find("conflicts") != std::string::npos);
    }
    // Environment is the config default; the flag wins; a typo is an error.
    {
        ScopedEnv env("SONDER_INFER_TEXT_CAPTURE", "full");
        det::ServeCliConfig c;
        REQUIRE(build({"--backend", "mock"}, c, err) == 0);
        CHECK(c.options.text_capture == si::TextCapturePolicy::full);
        det::ServeCliConfig c2;
        REQUIRE(build({"--backend", "mock", "--text-capture", "off"}, c2, err) == 0);
        CHECK(c2.options.text_capture == si::TextCapturePolicy::off);
    }
    {
        ScopedEnv env("SONDER_INFER_TEXT_CAPTURE", "sometimes");
        det::ServeCliConfig c;
        CHECK(build({"--backend", "mock"}, c, err) == 2);
        CHECK(err.find("SONDER_INFER_TEXT_CAPTURE must be off, on or full") != std::string::npos);
    }
    {
        ScopedEnv env("SONDER_INFER_TEXT_CAPTURE", "");
        det::ServeCliConfig c;
        REQUIRE(build({"--backend", "mock"}, c, err) == 0);
        CHECK(c.options.text_capture == si::TextCapturePolicy::off);
    }
    // Without a token, on and full are refused at option build time.
    g_token_file.clear();
    for (const char* policy : {"on", "full"}) {
        CAPTURE(policy);
        det::ServeCliConfig c;
        CHECK(build({"--backend", "mock", "--text-capture", policy}, c, err) == 2);
        CHECK(err.find("--token-file") != std::string::npos);
    }
    std::error_code ec;
    fs::remove(token_path, ec);
}

TEST_CASE("text capture server: on and full require a token") {
    auto o = Fixture::defaults();
    CHECK(srv::validate_options(o).ok());
    for (const auto policy : {si::TextCapturePolicy::on, si::TextCapturePolicy::full}) {
        o.text_capture = policy;
        o.token.clear();
        const auto st = srv::validate_options(o);
        CHECK(st.code() == si::ErrorCode::invalid_argument);
        CHECK(st.message().find("--text-capture") != std::string::npos);
        CHECK(st.message().find("--token-file") != std::string::npos);
        o.token = kToken;
        CHECK(srv::validate_options(o).ok());
    }
    o.text_capture = si::TextCapturePolicy::off;
    o.token.clear();
    CHECK(srv::validate_options(o).ok());
}

TEST_CASE("text capture server: the startup log says so plainly, and only when not off") {
    const auto token = fs::temp_directory_path() / (si::make_id("sonder-capture-test") + "-token");
    {
        std::ofstream(token) << "tok-capture\n";
    }
    fs::permissions(token, fs::perms::owner_read | fs::perms::owner_write);
    // A model the mock backend refuses makes serve_main exit 1 after the
    // startup diagnostics, so no shutdown hook is needed.
    const auto serve = [&](const std::string& policy, std::string& err_text) {
        std::ostringstream out;
        std::ostringstream err;
        const int rc = srv::serve_main({"--backend", "mock", "--model", "not-a-mock-model", "--port", "0",
                                        "--token-file", token.string(), "--text-capture", policy},
                                       out, err);
        err_text = err.str();
        return rc;
    };
    std::string err_text;
    CHECK(serve("full", err_text) == 1);
    CHECK(err_text.find("text capture is FULL") != std::string::npos);
    CHECK(err_text.find("prompt text") != std::string::npos);
    CHECK(serve("on", err_text) == 1);
    CHECK(err_text.find("text capture is ON") != std::string::npos);
    CHECK(serve("off", err_text) == 1);
    CHECK(err_text.find("text capture is") == std::string::npos);
    std::error_code ec;
    fs::remove(token, ec);
}
