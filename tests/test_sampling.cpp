#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {
bool rejects(SamplingConfig c) {
    const auto st = validate(c);
    return !st.ok() && st.code() == ErrorCode::invalid_argument && !st.message().empty();
}
}  // namespace

TEST_SUITE("sampling") {
TEST_CASE("defaults and greedy preset are valid") {
    CHECK(validate(SamplingConfig{}).ok());
    const auto g = SamplingConfig::greedy(64, 7);
    CHECK(validate(g).ok());
    CHECK(g.temperature == 0.0f);
    CHECK(g.max_tokens == 64);
    REQUIRE(g.seed.has_value());
    CHECK(*g.seed == 7u);
}

TEST_CASE("rejects out-of-range fields") {
    SamplingConfig c;
    c.temperature = -0.1f; CHECK(rejects(c));
    c.temperature = 10.5f; CHECK(rejects(c));
    c.temperature = std::numeric_limits<float>::quiet_NaN(); CHECK(rejects(c));
    c = {}; c.top_p = 0.0f; CHECK(rejects(c));
    c = {}; c.top_p = 1.01f; CHECK(rejects(c));
    c = {}; c.top_k = -1; CHECK(rejects(c));
    c = {}; c.min_p = 1.5f; CHECK(rejects(c));
    c = {}; c.repeat_penalty = 0.0f; CHECK(rejects(c));
    c = {}; c.max_tokens = 0; CHECK(rejects(c));
    c = {}; c.max_tokens = SamplingConfig::kMaxTokensLimit + 1; CHECK(rejects(c));
    c = {}; c.stop = {""}; CHECK(rejects(c));
    c = {}; c.stop = std::vector<std::string>(SamplingConfig::kMaxStopSequences + 1, "x"); CHECK(rejects(c));
    c = {}; c.stop = {std::string(SamplingConfig::kMaxStopSequenceBytes + 1, 'x')}; CHECK(rejects(c));
}

TEST_CASE("accepts boundary values") {
    SamplingConfig c;
    c.temperature = 0.0f; c.top_p = 1.0f; c.top_k = 0; c.min_p = 0.0f; c.max_tokens = 1;
    CHECK(validate(c).ok());
    c.temperature = 10.0f; c.min_p = 1.0f; c.max_tokens = SamplingConfig::kMaxTokensLimit;
    CHECK(validate(c).ok());
}

TEST_CASE("new fields default to values that change nothing") {
    const SamplingConfig d;
    // Pre-existing defaults are untouched.
    CHECK(d.temperature == 0.8f);
    CHECK(d.top_p == 0.95f);
    CHECK(d.top_k == 40);
    CHECK(d.min_p == 0.0f);
    CHECK(d.repeat_penalty == 1.1f);
    CHECK_FALSE(d.seed.has_value());
    CHECK(d.max_tokens == 256);
    CHECK(d.stop.empty());
    // New fields are disabled / backend-default.
    CHECK(d.typical_p == 1.0f);
    CHECK(d.repeat_last_n == 64);
    CHECK(d.presence_penalty == 0.0f);
    CHECK(d.frequency_penalty == 0.0f);
    CHECK(d.logit_bias.empty());
    CHECK(d.num_ctx == 0);

    const auto g = SamplingConfig::greedy(64, 7);
    CHECK(g.typical_p == 1.0f);
    CHECK(g.presence_penalty == 0.0f);
    CHECK(g.frequency_penalty == 0.0f);
    CHECK(g.logit_bias.empty());
    CHECK(g.num_ctx == 0);
}

TEST_CASE("rejects out-of-range new fields with a field-specific message") {
    auto message_for = [](const SamplingConfig& c) {
        const auto st = validate(c);
        CHECK(st.code() == ErrorCode::invalid_argument);
        return st.message();
    };
    SamplingConfig c;
    c.typical_p = 0.0f; CHECK(message_for(c).find("typical_p") != std::string::npos);
    c.typical_p = 1.5f; CHECK(rejects(c));
    c.typical_p = std::numeric_limits<float>::quiet_NaN(); CHECK(rejects(c));
    c = {}; c.repeat_last_n = -2; CHECK(message_for(c).find("repeat_last_n") != std::string::npos);
    c = {}; c.repeat_last_n = SamplingConfig::kMaxContextLimit + 1; CHECK(rejects(c));
    c = {}; c.presence_penalty = 2.5f; CHECK(message_for(c).find("presence_penalty") != std::string::npos);
    c = {}; c.presence_penalty = std::numeric_limits<float>::infinity(); CHECK(rejects(c));
    c = {}; c.frequency_penalty = -2.5f; CHECK(message_for(c).find("frequency_penalty") != std::string::npos);
    c = {}; c.frequency_penalty = std::numeric_limits<float>::quiet_NaN(); CHECK(rejects(c));
    c = {}; c.num_ctx = -1; CHECK(message_for(c).find("num_ctx") != std::string::npos);
    c = {}; c.num_ctx = SamplingConfig::kMaxContextLimit + 1; CHECK(rejects(c));
    c = {}; c.logit_bias = {{-1, 1.0f}}; CHECK(message_for(c).find("logit_bias") != std::string::npos);
    c = {}; c.logit_bias = {{3, 101.0f}}; CHECK(rejects(c));
    c = {}; c.logit_bias = {{3, std::numeric_limits<float>::infinity()}}; CHECK(rejects(c));
    c = {}; c.logit_bias = {{3, std::numeric_limits<float>::quiet_NaN()}}; CHECK(rejects(c));
    c = {}; c.logit_bias = {{3, 1.0f}, {3, 2.0f}}; CHECK(message_for(c).find("more than once") != std::string::npos);
    c = {};
    c.logit_bias.resize(SamplingConfig::kMaxLogitBiasEntries + 1);
    for (std::size_t i = 0; i < c.logit_bias.size(); ++i) {
        c.logit_bias[i].token = static_cast<std::int32_t>(i);
    }
    CHECK(rejects(c));
}

TEST_CASE("accepts boundary values of new fields") {
    SamplingConfig c;
    c.typical_p = 1.0f; c.repeat_last_n = -1; c.presence_penalty = -2.0f; c.frequency_penalty = 2.0f;
    c.num_ctx = SamplingConfig::kMaxContextLimit;
    c.logit_bias = {{0, -100.0f}, {1, 100.0f}, {2, -std::numeric_limits<float>::infinity()}};
    CHECK(validate(c).ok());
    c.typical_p = 0.01f; c.repeat_last_n = 0; c.presence_penalty = 2.0f; c.frequency_penalty = -2.0f;
    c.num_ctx = 1;
    c.logit_bias.assign(SamplingConfig::kMaxLogitBiasEntries, TokenLogitBias{});
    for (std::size_t i = 0; i < c.logit_bias.size(); ++i) {
        c.logit_bias[i].token = static_cast<std::int32_t>(i);
    }
    CHECK(validate(c).ok());
}

TEST_CASE("engine and session enforce validation") {
    sonder_test::Harness h;
    REQUIRE(h.model);
    SessionOptions bad;
    bad.sampling.top_p = 2.0f;
    auto s = h.engine->create_session(h.model, bad);
    CHECK(s.status().code() == ErrorCode::invalid_argument);

    auto session = h.session();
    REQUIRE(session);
    SamplingConfig override_cfg = SamplingConfig::greedy();
    override_cfg.max_tokens = -5;
    auto r = session->generate("hi", {}, override_cfg);
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(session->requests_started() == 0);  // rejected before starting
    CHECK(session->state() == SessionState::idle);
}
}
