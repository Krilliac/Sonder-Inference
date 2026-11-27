// Telemetry text-capture policy: the pure policy helpers, then the policy
// applied by a real Session on the mock backend (docs/TELEMETRY.md "Text
// capture policy").
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "sonder/inference.hpp"
#include "sonder/inference/text_capture.hpp"

using namespace sonder::inference;

namespace {

// Engine + mock session with an explicit text-capture setting.
struct CaptureHarness {
    std::shared_ptr<MemoryTelemetrySink> sink = std::make_shared<MemoryTelemetrySink>();
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Model> model;
    std::shared_ptr<Session> session;

    explicit CaptureHarness(TextCapturePolicy policy, bool legacy_capture_text = false) {
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::standard;
        eo.telemetry.text_capture = policy;
        eo.telemetry.capture_text = legacy_capture_text;
        eo.telemetry_sinks.push_back(sink);
        engine = std::make_unique<Engine>(std::move(eo));
        (void)engine->register_backend(make_mock_backend({}));
        ModelLoadOptions lo;
        lo.model = "mock:tiny";
        auto loaded = engine->load_model(kMockBackendName, lo);
        REQUIRE(loaded.ok());
        model = loaded.value();
        SessionOptions so;
        so.sampling = SamplingConfig::greedy(8);
        auto s = engine->create_session(model, so);
        REQUIRE(s.ok());
        session = s.value();
    }

    std::vector<json::Value> events_of(const std::string& type) {
        engine->telemetry().flush();
        std::vector<json::Value> out;
        for (const auto& line : sink->lines()) {
            auto v = json::parse(line);
            if (v.ok() && v.value().find("event_type")->as_string() == type) out.push_back(v.value());
        }
        return out;
    }
};

const json::Value* attr(const json::Value& event, const char* name) {
    return event.find("attributes")->find(name);
}

constexpr const char* kPrompt = "zebra quartz marmalade";

}  // namespace

TEST_CASE("text capture: the policy names parse exactly and round-trip") {
    CHECK(parse_text_capture_policy("off") == TextCapturePolicy::off);
    CHECK(parse_text_capture_policy("on") == TextCapturePolicy::on);
    CHECK(parse_text_capture_policy("full") == TextCapturePolicy::full);
    CHECK(parse_text_capture_policy("FULL") == TextCapturePolicy::full);
    CHECK(parse_text_capture_policy("On") == TextCapturePolicy::on);
    for (const char* bad : {"", " on", "on ", "true", "1", "yes", "onn", "fulll", "of", "off\n", "ful"}) {
        CAPTURE(bad);
        CHECK_FALSE(parse_text_capture_policy(bad).has_value());
    }
    for (const auto p : {TextCapturePolicy::off, TextCapturePolicy::on, TextCapturePolicy::full}) {
        CHECK(parse_text_capture_policy(to_string(p)) == p);
    }
}

TEST_CASE("text capture: default policy is off") {
    CHECK(TelemetryOptions{}.text_capture == TextCapturePolicy::off);
    CHECK_FALSE(TelemetryOptions{}.capture_text);
    CHECK(resolved_text_capture(TelemetryOptions{}) == TextCapturePolicy::off);
    CHECK_FALSE(RequestOptions{}.no_text_capture);
}

TEST_CASE("text capture: which policy exports what") {
    CHECK_FALSE(captures_output_text(TextCapturePolicy::off));
    CHECK(captures_output_text(TextCapturePolicy::on));
    CHECK(captures_output_text(TextCapturePolicy::full));
    CHECK_FALSE(captures_prompt_text(TextCapturePolicy::off));
    CHECK_FALSE(captures_prompt_text(TextCapturePolicy::on));
    CHECK(captures_prompt_text(TextCapturePolicy::full));
}

TEST_CASE("text capture: the legacy flag only ever raises off to on") {
    CHECK(combine_text_capture(TextCapturePolicy::off, false) == TextCapturePolicy::off);
    CHECK(combine_text_capture(TextCapturePolicy::off, true) == TextCapturePolicy::on);
    CHECK(combine_text_capture(TextCapturePolicy::on, true) == TextCapturePolicy::on);
    CHECK(combine_text_capture(TextCapturePolicy::full, true) == TextCapturePolicy::full);
    CHECK(combine_text_capture(TextCapturePolicy::full, false) == TextCapturePolicy::full);
    TelemetryOptions legacy;
    legacy.capture_text = true;
    CHECK(resolved_text_capture(legacy) == TextCapturePolicy::on);
}

TEST_CASE("text capture: a request opt-out can only lower the policy") {
    for (const auto p : {TextCapturePolicy::off, TextCapturePolicy::on, TextCapturePolicy::full}) {
        CHECK(apply_request_opt_out(p, true) == TextCapturePolicy::off);
        CHECK(apply_request_opt_out(p, false) == p);
    }
}

TEST_CASE("text capture: X-Sonder-No-Capture header values") {
    CHECK(parse_no_capture_header("1") == true);
    CHECK(parse_no_capture_header("true") == true);
    CHECK(parse_no_capture_header("TRUE") == true);
    CHECK(parse_no_capture_header("0") == false);
    CHECK(parse_no_capture_header("false") == false);
    for (const char* bad : {"", "yes", "no", "2", "-1", " 1", "1 ", "on", "tru"}) {
        CAPTURE(bad);
        CHECK_FALSE(parse_no_capture_header(bad).has_value());
    }
}

TEST_CASE("text capture: startup notice is silent for off and plain otherwise") {
    CHECK(text_capture_startup_notice(TextCapturePolicy::off).empty());
    const std::string on = text_capture_startup_notice(TextCapturePolicy::on);
    CHECK(on.find("text capture is ON") != std::string::npos);
    CHECK(on.find("prompts are not") != std::string::npos);
    const std::string full = text_capture_startup_notice(TextCapturePolicy::full);
    CHECK(full.find("text capture is FULL") != std::string::npos);
    CHECK(full.find("prompt text") != std::string::npos);
    CHECK(full.find("X-Sonder-No-Capture") != std::string::npos);
}

TEST_CASE("text capture: truncate_utf8 never splits a sequence") {
    CHECK(truncate_utf8("hello", 10) == "hello");
    CHECK(truncate_utf8("hello", 5) == "hello");
    CHECK(truncate_utf8("hello", 3) == "hel");
    CHECK(truncate_utf8("hello", 0).empty());
    CHECK(truncate_utf8("", 4).empty());
    const std::string two = "a\xC3\xA9";                  // a + e-acute (2 bytes)
    CHECK(truncate_utf8(two, 3) == two);
    CHECK(truncate_utf8(two, 2) == "a");                  // would end inside the pair
    const std::string three = "a\xE2\x82\xAC";            // a + euro sign (3 bytes)
    CHECK(truncate_utf8(three, 3) == "a");
    CHECK(truncate_utf8(three, 2) == "a");
    CHECK(truncate_utf8(three, 1) == "a");
    const std::string four = "a\xF0\x9F\x98\x80";         // a + emoji (4 bytes)
    CHECK(truncate_utf8(four, 4) == "a");
    CHECK(truncate_utf8(four, 2) == "a");
    CHECK(truncate_utf8(four, 5) == four);
    CHECK(truncate_utf8("\xF0\x9F\x98\x80", 3).empty());  // nothing fits
    // Invalid input (only continuation bytes) must terminate and stay bounded.
    const std::string junk(10, '\x80');
    CHECK(truncate_utf8(junk, 5).size() <= 5);
    CHECK(kMaxCapturedPromptBytes == 64u * 1024u);
}

TEST_CASE("text capture: off emits no text fields and declares off") {
    CaptureHarness h(TextCapturePolicy::off);
    REQUIRE(h.session->generate(kPrompt).ok());
    const auto tokens = h.events_of("inference.token.generated");
    REQUIRE_FALSE(tokens.empty());
    for (const auto& e : tokens) CHECK(attr(e, "text") == nullptr);
    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 1);
    CHECK(attr(started[0], "prompt") == nullptr);
    CHECK(attr(started[0], "prompt_truncated") == nullptr);
    CHECK(attr(started[0], "text_capture_opt_out") == nullptr);
    CHECK(attr(h.events_of("engine.started")[0], "text_capture")->as_string() == "off");
    CHECK(attr(h.events_of("session.created")[0], "text_capture")->as_string() == "off");
}

TEST_CASE("text capture: on emits output text only") {
    CaptureHarness h(TextCapturePolicy::on);
    REQUIRE(h.session->generate(kPrompt).ok());
    const auto tokens = h.events_of("inference.token.generated");
    REQUIRE_FALSE(tokens.empty());
    for (const auto& e : tokens) {
        REQUIRE(attr(e, "text") != nullptr);
        CHECK(attr(e, "text")->is_string());
    }
    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 1);
    CHECK(attr(started[0], "prompt") == nullptr);
    CHECK(attr(h.events_of("engine.started")[0], "text_capture")->as_string() == "on");
    CHECK(attr(h.events_of("session.created")[0], "text_capture")->as_string() == "on");
}

TEST_CASE("text capture: full emits output text and the prompt") {
    CaptureHarness h(TextCapturePolicy::full);
    REQUIRE(h.session->generate(kPrompt).ok());
    const auto tokens = h.events_of("inference.token.generated");
    REQUIRE_FALSE(tokens.empty());
    for (const auto& e : tokens) CHECK(attr(e, "text") != nullptr);
    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 1);
    REQUIRE(attr(started[0], "prompt") != nullptr);
    CHECK(attr(started[0], "prompt")->as_string() == kPrompt);
    CHECK_FALSE(attr(started[0], "prompt_truncated")->as_bool());
    CHECK(attr(h.events_of("engine.started")[0], "text_capture")->as_string() == "full");
    CHECK(attr(h.events_of("session.created")[0], "text_capture")->as_string() == "full");
}

TEST_CASE("text capture: full bounds the prompt copied into request.started") {
    CaptureHarness h(TextCapturePolicy::full);
    const std::string big(kMaxCapturedPromptBytes + 1000, 'q');
    REQUIRE(h.session->generate(big).ok());
    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 1);
    CHECK(attr(started[0], "prompt")->as_string().size() == kMaxCapturedPromptBytes);
    CHECK(attr(started[0], "prompt_truncated")->as_bool());
}

TEST_CASE("text capture: the legacy capture_text flag still means on") {
    CaptureHarness h(TextCapturePolicy::off, /*legacy_capture_text=*/true);
    REQUIRE(h.session->generate(kPrompt).ok());
    for (const auto& e : h.events_of("inference.token.generated")) CHECK(attr(e, "text") != nullptr);
    CHECK(attr(h.events_of("request.started")[0], "prompt") == nullptr);
    CHECK(attr(h.events_of("engine.started")[0], "text_capture")->as_string() == "on");
}

TEST_CASE("text capture: the per-request opt-out forces off for that request only") {
    CaptureHarness h(TextCapturePolicy::full);
    RequestOptions quiet;
    quiet.no_text_capture = true;
    REQUIRE(h.session->generate("first secret prompt", {}, std::nullopt, quiet).ok());
    REQUIRE(h.session->generate("second public prompt").ok());

    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 2);
    // First request: no prompt, marked as opted out.
    CHECK(attr(started[0], "prompt") == nullptr);
    REQUIRE(attr(started[0], "text_capture_opt_out") != nullptr);
    CHECK(attr(started[0], "text_capture_opt_out")->as_bool());
    // Second request: captured as usual.
    REQUIRE(attr(started[1], "prompt") != nullptr);
    CHECK(attr(started[1], "prompt")->as_string() == "second public prompt");

    // No token event of the first request carries text; the second's do.
    const std::string first_request = started[0].find("request_id")->as_string();
    std::size_t first_tokens = 0;
    std::size_t second_tokens_with_text = 0;
    for (const auto& e : h.events_of("inference.token.generated")) {
        if (e.find("request_id")->as_string() == first_request) {
            ++first_tokens;
            CHECK(attr(e, "text") == nullptr);
        } else if (attr(e, "text") != nullptr) {
            ++second_tokens_with_text;
        }
    }
    CHECK(first_tokens > 0);
    CHECK(second_tokens_with_text > 0);
}

TEST_CASE("text capture: an opt-out never turns capture on") {
    CaptureHarness h(TextCapturePolicy::off);
    RequestOptions explicit_no;
    explicit_no.no_text_capture = false;
    REQUIRE(h.session->generate(kPrompt, {}, std::nullopt, explicit_no).ok());
    for (const auto& e : h.events_of("inference.token.generated")) CHECK(attr(e, "text") == nullptr);
    CHECK(attr(h.events_of("request.started")[0], "prompt") == nullptr);
    // Opting out while off adds no marker either: the stream stays unchanged.
    CaptureHarness h2(TextCapturePolicy::off);
    RequestOptions quiet;
    quiet.no_text_capture = true;
    REQUIRE(h2.session->generate(kPrompt, {}, std::nullopt, quiet).ok());
    CHECK(attr(h2.events_of("request.started")[0], "text_capture_opt_out") == nullptr);
}

TEST_CASE("text capture: chat prompts are captured under full and suppressed by the opt-out") {
    CaptureHarness h(TextCapturePolicy::full);
    const std::vector<ChatMessage> messages = {{"user", "chat marker walrus"}};
    REQUIRE(h.session->chat(messages).ok());
    RequestOptions quiet;
    quiet.no_text_capture = true;
    REQUIRE(h.session->chat(messages, {}, std::nullopt, quiet).ok());
    const auto started = h.events_of("request.started");
    REQUIRE(started.size() == 2);
    REQUIRE(attr(started[0], "prompt") != nullptr);
    CHECK(attr(started[0], "prompt")->as_string().find("chat marker walrus") != std::string::npos);
    CHECK(attr(started[1], "prompt") == nullptr);
}
