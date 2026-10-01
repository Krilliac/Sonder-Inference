// Request-path units: prompt_cache_key / chat_template_kwargs / think parsing,
// thinking pins, the conversation key, --scheduler / --kv-pool-tokens, and
// the cached-token usage and timings fields.
#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "sonder/inference.hpp"
#include "sonder/inference/server.hpp"
#include "src/health_features.hpp"
#include "src/http.hpp"
#include "src/openai.hpp"
#include "src/reasoning_budget.hpp"
#include "src/request_path.hpp"

namespace si = sonder::inference;
namespace srv = sonder::inference::server;
namespace det = sonder::inference::server::detail;

namespace {

det::ChatJob job_of(const std::string& body) {
    auto v = det::parse_chat_request(body);
    REQUIRE_MESSAGE(std::holds_alternative<det::ChatJob>(v),
                    (std::holds_alternative<det::ApiError>(v) ? std::get<det::ApiError>(v).message : ""));
    return std::get<det::ChatJob>(v);
}

det::ApiError error_of(const std::string& body) {
    auto v = det::parse_chat_request(body);
    REQUIRE(std::holds_alternative<det::ApiError>(v));
    return std::get<det::ApiError>(v);
}

si::Status flags(const std::map<std::string, std::string>& values, srv::ServerOptions& o) {
    return det::parse_request_path_flags(
        [&](const std::string& k) -> std::optional<std::string> {
            auto it = values.find(k);
            return it == values.end() ? std::nullopt : std::optional<std::string>(it->second);
        },
        o);
}

}  // namespace

TEST_CASE("chat extensions: prompt_cache_key, chat_template_kwargs and think") {
    const std::string msgs = R"("messages":[{"role":"user","content":"hi"}])";
    auto plain = job_of("{" + msgs + "}");
    CHECK_FALSE(plain.session_key);
    CHECK(plain.thinking.empty());

    auto j = job_of(R"({"prompt_cache_key":"conv-1","chat_template_kwargs":{"enable_thinking":false,)"
                    R"("reasoning_effort":"high","other":{"x":1}},)" + msgs + "}");
    CHECK(j.session_key == std::optional<std::string>("conv-1"));
    REQUIRE(j.thinking.enable_thinking);
    CHECK_FALSE(*j.thinking.enable_thinking);
    CHECK(j.thinking.reasoning_effort == std::optional<std::string>("high"));

    auto t = job_of(R"({"think":true,)" + msgs + "}");
    CHECK(t.thinking.enable_thinking == std::optional<bool>(true));
    // think agreeing with enable_thinking is fine; disagreeing is refused.
    CHECK(job_of(R"({"think":false,"chat_template_kwargs":{"enable_thinking":false},)" + msgs + "}")
              .thinking.enable_thinking == std::optional<bool>(false));
    CHECK(error_of(R"({"think":false,"chat_template_kwargs":{"enable_thinking":true},)" + msgs + "}").param ==
          std::optional<std::string>("think"));

    CHECK(error_of(R"({"think":"on",)" + msgs + "}").status == 400);
    CHECK(error_of(R"({"chat_template_kwargs":"x",)" + msgs + "}").param ==
          std::optional<std::string>("chat_template_kwargs"));
    CHECK(error_of(R"({"chat_template_kwargs":{"enable_thinking":1},)" + msgs + "}").param ==
          std::optional<std::string>("chat_template_kwargs.enable_thinking"));
    CHECK(error_of(R"({"chat_template_kwargs":{"reasoning_effort":"a b"},)" + msgs + "}").param ==
          std::optional<std::string>("chat_template_kwargs.reasoning_effort"));
    CHECK(error_of(R"({"prompt_cache_key":"",)" + msgs + "}").param == std::optional<std::string>("prompt_cache_key"));
    CHECK(error_of(R"({"prompt_cache_key":7,)" + msgs + "}").status == 400);
    // Still accepted: an empty kwargs object (Runtime bridges send one).
    CHECK(job_of(R"({"chat_template_kwargs":{},)" + msgs + "}").thinking.empty());
}

TEST_CASE("the conversation key prefers prompt_cache_key, then run and agent ids") {
    det::ChatJob job;
    det::Correlation c;
    CHECK(det::chat_session_key(job, c).empty());
    c.run_id = "r1";
    CHECK(det::chat_session_key(job, c) == "run=r1");
    c.agent_id = "a1";
    CHECK(det::chat_session_key(job, c) == "run=r1;agent=a1");
    c.run_id.reset();
    CHECK(det::chat_session_key(job, c) == "agent=a1");
    job.session_key = "conv";
    CHECK(det::chat_session_key(job, c) == "conv");
}

TEST_CASE("thinking pins fill unset fields and override conflicts with a warning") {
    srv::ServerOptions o;
    si::ThinkingOptions t;
    CHECK(det::apply_thinking_pins(o, t).empty());
    CHECK(t.empty());  // no pins: nothing forwarded

    o.pin_enable_thinking = false;
    o.pin_reasoning_effort = "low";
    CHECK(det::apply_thinking_pins(o, t).empty());
    CHECK(t.enable_thinking == std::optional<bool>(false));
    CHECK(t.reasoning_effort == std::optional<std::string>("low"));

    si::ThinkingOptions conflict;
    conflict.enable_thinking = true;
    conflict.reasoning_effort = "high";
    const auto warnings = det::apply_thinking_pins(o, conflict);
    CHECK(warnings.size() == 2);
    CHECK(conflict.enable_thinking == std::optional<bool>(false));
    CHECK(conflict.reasoning_effort == std::optional<std::string>("low"));
}

TEST_CASE("thinking pin override mode preserves the explicit false override warning") {
    srv::ServerOptions o;
    o.pin_enable_thinking = true;
    si::ThinkingOptions explicit_off;
    explicit_off.enable_thinking = false;
    const auto off_warnings = det::apply_thinking_pins(o, explicit_off);
    REQUIRE(off_warnings.size() == 1);
    CHECK(off_warnings.front().find("enable_thinking=off") != std::string::npos);
    CHECK(explicit_off.enable_thinking == std::optional<bool>(true));
}

TEST_CASE("thinking pin default mode preserves explicit request values without warnings") {
    srv::ServerOptions o;
    o.pin_mode = srv::PinMode::default_value;
    o.pin_enable_thinking = true;
    o.pin_reasoning_effort = "medium";

    si::ThinkingOptions explicit_request;
    explicit_request.enable_thinking = false;
    explicit_request.reasoning_effort = "low";
    CHECK(det::apply_thinking_pins(o, explicit_request).empty());
    CHECK(explicit_request.enable_thinking == std::optional<bool>(false));
    CHECK(explicit_request.reasoning_effort == std::optional<std::string>("low"));

    si::ThinkingOptions absent_request;
    CHECK(det::apply_thinking_pins(o, absent_request).empty());
    CHECK(absent_request.enable_thinking == std::optional<bool>(true));
    CHECK(absent_request.reasoning_effort == std::optional<std::string>("medium"));
}

TEST_CASE("--scheduler and --kv-pool-tokens map onto the engine scheduling options") {
    srv::ServerOptions o;
    REQUIRE(flags({}, o).ok());
    si::SchedulingOptions s;
    det::apply_scheduling(o, s);
    CHECK(s.enabled);
    CHECK(s.mode == si::SchedulerMode::automatic);
    CHECK(s.kv_num_blocks == 4096);  // default pool unchanged

    REQUIRE(flags({{"scheduler", "account"}, {"kv-pool-tokens", "262144"}}, o).ok());
    si::SchedulingOptions a;
    det::apply_scheduling(o, a);
    CHECK(a.mode == si::SchedulerMode::account);
    CHECK(a.kv_num_blocks == 262144 / 16);

    REQUIRE(flags({{"scheduler", "gate"}, {"kv-pool-tokens", "17"}}, o).ok());
    si::SchedulingOptions g;
    det::apply_scheduling(o, g);
    CHECK(g.mode == si::SchedulerMode::gate);
    CHECK(g.kv_num_blocks == 2);  // rounded up to whole blocks

    REQUIRE(flags({{"scheduler", "off"}}, o).ok());
    si::SchedulingOptions off;
    det::apply_scheduling(o, off);
    CHECK_FALSE(off.enabled);

    srv::ServerOptions bad;
    CHECK(flags({{"scheduler", "fast"}}, bad).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"kv-pool-tokens", "8"}}, bad).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"kv-pool-tokens", "x"}}, bad).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"pin-enable-thinking", "maybe"}}, bad).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"pin-reasoning-effort", "a b"}}, bad).code() == si::ErrorCode::invalid_argument);
    srv::ServerOptions pins;
    REQUIRE(flags({{"pin-enable-thinking", "off"}, {"pin-reasoning-effort", "medium"}}, pins).ok());
    CHECK(pins.pin_enable_thinking == std::optional<bool>(false));
    CHECK(pins.pin_reasoning_effort == std::optional<std::string>("medium"));
}

TEST_CASE("thinking and budget pin flags validate their values") {
    srv::ServerOptions mode;
    REQUIRE(flags({{"pin-mode", "default"}, {"pin-reasoning-budget", "512"},
                   {"pin-reasoning-budget-message", "short"}}, mode)
                .ok());
    CHECK(mode.pin_mode == srv::PinMode::default_value);
    CHECK(mode.pin_reasoning_budget_tokens == std::optional<std::int64_t>(512));
    CHECK(mode.pin_reasoning_budget_message == std::optional<std::string>("short"));
    CHECK(flags({{"pin-mode", "bad"}}, mode).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"pin-reasoning-budget", "-2"}}, mode).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"pin-reasoning-budget", "nope"}}, mode).code() == si::ErrorCode::invalid_argument);
    CHECK(flags({{"pin-reasoning-budget-message", std::string(513, 'x')}}, mode).code() ==
          si::ErrorCode::invalid_argument);
}

TEST_CASE("usage and timings add cached, draft and queue fields only when measured") {
    si::GenerationResult r;
    r.stats.prompt_tokens = 12;
    r.stats.completion_tokens = 3;
    auto usage = si::json::Value(det::usage_json(r));
    auto timings = si::json::Value(det::timings_json(r));
    CHECK(usage.find("prompt_tokens_details") == nullptr);
    CHECK(timings.find("cache_n") == nullptr);
    CHECK(timings.find("draft_n") == nullptr);
    CHECK(timings.find("draft_n_accepted") == nullptr);
    CHECK(timings.find("queue_ms") == nullptr);

    r.stats.cached_tokens = 0;  // a reported zero is a real miss
    r.stats.draft_tokens = 8;
    r.stats.draft_accepted_tokens = 5;
    r.scheduling.scheduled = true;
    r.scheduling.queue_ms = 1.5;
    usage = si::json::Value(det::usage_json(r));
    timings = si::json::Value(det::timings_json(r));
    REQUIRE(usage.find("prompt_tokens_details") != nullptr);
    CHECK(usage.find("prompt_tokens_details")->find("cached_tokens")->as_int() == 0);
    CHECK(usage.find("prompt_tokens")->as_int() == 12);
    CHECK(timings.find("cache_n")->as_int() == 0);
    CHECK(timings.find("draft_n")->as_int() == 8);
    CHECK(timings.find("draft_n_accepted")->as_int() == 5);
    CHECK(timings.find("queue_ms")->as_double() == doctest::Approx(1.5));
}

TEST_CASE("upstream usage timings preserve fields and respect the existing cached token count") {
    si::GenerationResult r;
    r.stats.cached_tokens = 0;
    si::BackendTimings backend;
    backend.prompt_n = 20;
    backend.cache_n = 19;
    backend.prompt_ms = 2.5;
    backend.predicted_n = 7;
    backend.predicted_ms = 3.5;
    backend.draft_n = 4;
    backend.draft_n_accepted = 3;
    r.stats.backend_timings = backend;
    auto usage = si::json::Value(det::usage_json(r));
    const auto* sonder = usage.find("sonder");
    REQUIRE(sonder != nullptr);
    const auto* upstream = sonder->find("timings");
    REQUIRE(upstream != nullptr);
    CHECK(upstream->find("prompt_n")->as_int() == 20);
    CHECK(upstream->find("cache_n")->as_int() == 19);
    CHECK(upstream->find("prompt_ms")->as_double() == doctest::Approx(2.5));
    CHECK(upstream->find("predicted_n")->as_int() == 7);
    CHECK(upstream->find("predicted_ms")->as_double() == doctest::Approx(3.5));
    CHECK(upstream->find("draft_n")->as_int() == 4);
    CHECK(upstream->find("draft_n_accepted")->as_int() == 3);
    CHECK(usage.find("prompt_tokens_details")->find("cached_tokens")->as_int() == 0);

    r.stats.cached_tokens.reset();
    usage = si::json::Value(det::usage_json(r));
    CHECK(usage.find("prompt_tokens_details")->find("cached_tokens")->as_int() == 19);
    si::BackendTimings partial;
    partial.prompt_n = 20;
    r.stats.backend_timings = partial;
    usage = si::json::Value(det::usage_json(r));
    CHECK(usage.find("sonder")->find("timings")->find("cache_n") == nullptr);
}

TEST_CASE("reasoning budget parser accepts canonical, alias, zero and signed limits") {
    CHECK(det::reasoning_budget::parse_integer("-1") == std::optional<std::int64_t>(-1));
    CHECK(det::reasoning_budget::parse_integer("0") == std::optional<std::int64_t>(0));
    CHECK(det::reasoning_budget::parse_integer("-0") == std::optional<std::int64_t>(0));
    CHECK(det::reasoning_budget::parse_integer("-01") == std::optional<std::int64_t>(-1));
    CHECK(det::reasoning_budget::parse_integer("9223372036854775807") ==
          std::optional<std::int64_t>(INT64_MAX));
    CHECK_FALSE(det::reasoning_budget::parse_integer("+1"));
    CHECK_FALSE(det::reasoning_budget::parse_integer("9223372036854775808"));

    auto parsed = si::json::parse(R"({"thinking_budget_tokens":0})");
    REQUIRE(parsed.ok());
    det::ChatJob alias_job;
    CHECK_FALSE(det::reasoning_budget::parse_body(parsed->as_object(), alias_job));
    CHECK(alias_job.reasoning_budget_tokens == std::optional<std::int64_t>(0));

    parsed = si::json::parse(R"({"reasoning_budget_tokens":-1})");
    REQUIRE(parsed.ok());
    det::ChatJob canonical_job;
    CHECK_FALSE(det::reasoning_budget::parse_body(parsed->as_object(), canonical_job));
    CHECK(canonical_job.reasoning_budget_tokens == std::optional<std::int64_t>(-1));

    det::RequestHead head;
    head.headers = {{"x-sonder-reasoning-budget", "9223372036854775807"}};
    det::Correlation correlation;
    CHECK_FALSE(det::reasoning_budget::parse_header(head, correlation));
    CHECK(correlation.reasoning_budget_tokens == std::optional<std::int64_t>(INT64_MAX));
}

TEST_CASE("reasoning budget header wins and pins fill only absent request values") {
    det::ChatJob job;
    job.reasoning_budget_tokens = 0;
    job.reasoning_budget_message = "request";
    det::Correlation header;
    header.reasoning_budget_tokens = 1024;
    srv::ServerOptions pins;
    pins.pin_reasoning_budget_tokens = 512;
    pins.pin_reasoning_budget_message = "pinned";
    si::RequestOptions request;
    std::vector<std::string> warnings;
    det::reasoning_budget::apply(job, header, pins, "llamaserver", request, warnings);
    CHECK(request.reasoning_budget_tokens == std::optional<std::int64_t>(1024));
    CHECK(request.reasoning_budget_message == std::optional<std::string>("request"));
    CHECK(warnings.empty());

    det::ChatJob absent;
    si::RequestOptions filled;
    det::reasoning_budget::apply(absent, {}, pins, "llamaserver", filled, warnings);
    CHECK(filled.reasoning_budget_tokens == std::optional<std::int64_t>(512));
    CHECK(filled.reasoning_budget_message == std::optional<std::string>("pinned"));
}

TEST_CASE("reasoning budget request parser preserves byte boundaries and canonical precedence") {
    std::string message;
    for (int i = 0; i < 256; ++i) message += "\xC3\xA9";
    si::json::Object body{{"messages", si::json::Array{si::json::Object{{"role", "user"}, {"content", "hi"}}}},
                          {"reasoning_budget_tokens", 512}, {"thinking_budget_tokens", 99},
                          {"reasoning_budget_message", message}};
    const auto job = job_of(si::json::Value(body).dump());
    CHECK(job.reasoning_budget_tokens == std::optional<std::int64_t>(512));
    CHECK(job.reasoning_budget_message == std::optional<std::string>(message));
    body.set("thinking_budget_tokens", -2);
    CHECK(error_of(si::json::Value(body).dump()).param == std::optional<std::string>("thinking_budget_tokens"));

    det::RequestHead duplicate;
    duplicate.headers = {{"x-sonder-reasoning-budget", "512"}, {"x-sonder-reasoning-budget", "512"}};
    const auto parsed = det::parse_correlation(duplicate);
    REQUIRE(std::holds_alternative<det::ApiError>(parsed));
    CHECK(std::get<det::ApiError>(parsed).status == 400);
    CHECK(std::get<det::ApiError>(parsed).param == std::optional<std::string>("X-Sonder-Reasoning-Budget"));
}

TEST_CASE("reasoning budget rejects invalid body and header values and warns on unsupported backends") {
    for (const auto* value : {"-2", "1.5", "true", "9223372036854775808"}) {
        auto parsed = si::json::parse(std::string("{\"reasoning_budget_tokens\":") + value + "}");
        REQUIRE(parsed.ok());
        det::ChatJob job;
        const auto error = det::reasoning_budget::parse_body(parsed->as_object(), job);
        REQUIRE(error);
        CHECK(error->status == 400);
        CHECK(error->param == std::optional<std::string>("reasoning_budget_tokens"));
    }
    det::RequestHead head;
    head.headers = {{"x-sonder-reasoning-budget", "-2"}};
    det::Correlation correlation;
    CHECK(det::reasoning_budget::parse_header(head, correlation)->status == 400);

    det::ChatJob job;
    job.reasoning_budget_tokens = 1;
    job.reasoning_budget_message = "hint";
    si::RequestOptions request;
    std::vector<std::string> warnings;
    det::reasoning_budget::apply(job, {}, {}, "mock", request, warnings);
    CHECK_FALSE(request.reasoning_budget_tokens);
    CHECK_FALSE(request.reasoning_budget_message);
    REQUIRE(warnings.size() == 2);
    CHECK(warnings[0].find("reasoning_budget_tokens") != std::string::npos);
    CHECK(warnings[1].find("reasoning_budget_message") != std::string::npos);
}

TEST_CASE("health features are exact by backend and report pin mode") {
    const auto mock = det::health_features("mock", {}, {});
    REQUIRE(mock.find("features") != nullptr);
    CHECK(mock.find("features")->as_array().empty());

    srv::ServerOptions options;
    options.pin_mode = srv::PinMode::default_value;
    const auto llama = det::health_features("llamaserver", options, {});
    const auto& features = llama.find("features")->as_array();
    const std::vector<std::string> expected = {"thinking", "chat_template_kwargs", "enable_thinking",
                                               "reasoning_effort", "reasoning_budget", "prompt_cache_key",
                                               "priority_classes"};
    REQUIRE(features.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK(features[i].as_string() == expected[i]);
    CHECK(llama.find("pins")->find("mode")->as_string() == "default");
    options.pin_mode = srv::PinMode::override_request;
    CHECK(det::health_features("llamaserver", options, {}).find("pins")->find("mode")->as_string() ==
          "override");
}

TEST_CASE("serve help advertises thinking pin and budget flags") {
    std::ostringstream out;
    std::ostringstream err;
    CHECK(srv::serve_main({"--help"}, out, err) == 0);
    CHECK(out.str().find("--pin-mode override|default") != std::string::npos);
    CHECK(out.str().find("--pin-reasoning-budget N") != std::string::npos);
    CHECK(out.str().find("--pin-reasoning-budget-message TEXT") != std::string::npos);
}
