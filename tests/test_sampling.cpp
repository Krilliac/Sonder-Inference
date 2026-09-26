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
