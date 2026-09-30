// Request-path units: prompt_cache_key / chat_template_kwargs / think parsing,
// thinking pins, the conversation key, --scheduler / --kv-pool-tokens, and
// the cached-token usage and timings fields.
#include <doctest/doctest.h>

#include <map>
#include <optional>
#include <string>
#include <variant>

#include "sonder/inference.hpp"
#include "sonder/inference/server.hpp"
#include "src/openai.hpp"
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
