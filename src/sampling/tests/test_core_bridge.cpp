#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "sonder/sampling/core_bridge.hpp"
#include "sampling_test_util.hpp"

using namespace sonder::inference::sampling;
namespace core = sonder::inference;

TEST_CASE("from_core maps every shared field") {
    core::SamplingConfig in;
    in.temperature = 0.7f;
    in.top_p = 0.9f;
    in.top_k = 20;
    in.min_p = 0.1f;
    in.repeat_penalty = 1.2f;
    in.seed = 5;
    in.stop = {"###"};
    const SamplerConfig out = from_core(in);
    CHECK_NEAR(out.temperature, 0.7, 1e-6);
    CHECK(!out.greedy);
    CHECK_NEAR(out.top_p, 0.9, 1e-6);
    CHECK_EQ(out.top_k, 20);
    CHECK_NEAR(out.min_p, 0.1, 1e-6);
    CHECK_NEAR(out.repeat_penalty, 1.2, 1e-6);
    CHECK(out.seed == std::optional<std::uint64_t>{5});
    CHECK((out.stop_sequences == std::vector<std::string>{"###"}));
    CHECK_NEAR(out.typical_p, 1.0, 0.0);
    CHECK_NEAR(out.frequency_penalty, 0.0, 0.0);
    CHECK_NEAR(out.presence_penalty, 0.0, 0.0);
    CHECK(out.stage_order == default_stage_order());
    CHECK(validate(out).ok());
}

TEST_CASE("core greedy preset maps to the greedy selector") {
    const SamplerConfig out = from_core(core::SamplingConfig::greedy());
    CHECK(out.greedy);
    auto chain = make_chain(core::SamplingConfig::greedy());
    REQUIRE(chain.ok());
    CHECK_EQ(chain.value().sample(std::vector<float>{0.0f, 3.0f, 1.0f}).token, 1);
}

TEST_CASE("make_chain rejects invalid core configs with invalid_argument") {
    core::SamplingConfig bad;
    bad.top_p = 0.0f;  // core requires (0, 1]
    auto r = make_chain(bad);
    CHECK(!r.ok());
    CHECK(r.status().code() == core::ErrorCode::invalid_argument);
    CHECK(r.status().message().find("top_p") != std::string::npos);
}

TEST_CASE("to_status lists every failing field") {
    SamplerConfig c;
    c.top_k = -1;
    c.typical_p = 2.0f;
    const auto st = to_status(validate(c));
    CHECK(st.code() == core::ErrorCode::invalid_argument);
    CHECK(st.message().find("top_k") != std::string::npos);
    CHECK(st.message().find("typical_p") != std::string::npos);
    CHECK(to_status(validate(SamplerConfig{})).ok());
}

TEST_CASE("seeded core config samples reproducibly through make_chain") {
    core::SamplingConfig in;
    in.seed = 42;
    in.repeat_penalty = 1.0f;
    auto a = make_chain(in);
    auto b = make_chain(in);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const std::vector<float> logits = {1.0f, 2.0f, 0.5f, 1.5f, -1.0f, 0.0f, 2.2f, 1.1f};
    for (int i = 0; i < 64; ++i) {
        CHECK_EQ(a.value().sample(logits).token, b.value().sample(logits).token);
    }
}

TEST_CASE("from_core maps typical_p, penalties and logit bias") {
    core::SamplingConfig in;
    in.typical_p = 0.8f;
    in.repeat_last_n = -1;
    in.presence_penalty = 0.4f;
    in.frequency_penalty = -0.3f;
    in.logit_bias = {{7, 2.0f}, {2, -std::numeric_limits<float>::infinity()}};
    in.num_ctx = 4096;  // backend option; no chain equivalent
    const SamplerConfig out = from_core(in);
    CHECK_NEAR(out.typical_p, 0.8, 1e-6);
    CHECK_EQ(out.penalty_last_n, -1);
    CHECK_NEAR(out.presence_penalty, 0.4, 1e-6);
    CHECK_NEAR(out.frequency_penalty, -0.3, 1e-6);
    REQUIRE_EQ(out.logit_bias.size(), 2u);
    CHECK_EQ(out.logit_bias[0].token, 7);
    CHECK_NEAR(out.logit_bias[0].bias, 2.0, 0.0);
    CHECK_EQ(out.logit_bias[1].token, 2);
    CHECK(std::isinf(out.logit_bias[1].bias));
    CHECK(validate(out).ok());
}

TEST_CASE("default core config maps exactly as before the new fields") {
    const SamplerConfig out = from_core(core::SamplingConfig{});
    CHECK_NEAR(out.typical_p, 1.0, 0.0);
    CHECK_EQ(out.penalty_last_n, 64);
    CHECK_NEAR(out.presence_penalty, 0.0, 0.0);
    CHECK_NEAR(out.frequency_penalty, 0.0, 0.0);
    CHECK(out.logit_bias.empty());
    CHECK_EQ(out.min_keep, 0);
    CHECK(out.stage_order == default_stage_order());
}

TEST_CASE("logit bias from the core config steers the chain") {
    core::SamplingConfig in = core::SamplingConfig::greedy();
    in.logit_bias = {{0, 10.0f}, {1, -std::numeric_limits<float>::infinity()}};
    auto chain = make_chain(in);
    REQUIRE(chain.ok());
    // Token 1 is banned; token 0 gets +10 and wins over token 2.
    CHECK_EQ(chain.value().sample(std::vector<float>{0.0f, 5.0f, 3.0f}).token, 0);
}

TEST_CASE("core penalties change which token the chain picks") {
    core::SamplingConfig in = core::SamplingConfig::greedy();
    in.presence_penalty = 2.0f;
    auto chain = make_chain(in);
    REQUIRE(chain.ok());
    const std::vector<float> logits = {1.0f, 2.0f, 0.0f};
    auto first = chain.value().sample(logits);
    REQUIRE(first.ok());
    CHECK_EQ(first.token, 1);
    chain.value().accept(first.token);
    // 2.0 - presence 2.0 = 0.0 < 1.0, so token 0 now wins.
    CHECK_EQ(chain.value().sample(logits).token, 0);
}

TEST_CASE("make_chain rejects invalid new core fields and out-of-vocab bias") {
    core::SamplingConfig bad;
    bad.frequency_penalty = 5.0f;
    auto r = make_chain(bad);
    CHECK(r.status().code() == core::ErrorCode::invalid_argument);
    CHECK(r.status().message().find("frequency_penalty") != std::string::npos);

    core::SamplingConfig oov;
    oov.logit_bias = {{100, 1.0f}};
    auto r2 = make_chain(oov, nullptr, std::size_t{8});
    CHECK(r2.status().code() == core::ErrorCode::invalid_argument);
    CHECK(r2.status().message().find("logit_bias") != std::string::npos);
}

TEST_CASE("seeded runs with every new field set stay deterministic") {
    core::SamplingConfig in;
    in.seed = 1234;
    in.temperature = 1.0f;
    in.top_k = 0;
    in.top_p = 1.0f;
    in.typical_p = 0.95f;
    in.repeat_last_n = 16;
    in.presence_penalty = 0.3f;
    in.frequency_penalty = 0.2f;
    in.logit_bias = {{3, 1.5f}, {6, -2.0f}};
    in.num_ctx = 2048;
    auto run = [&] {
        auto chain = make_chain(in);
        REQUIRE(chain.ok());
        const std::vector<float> logits = {1.0f, 2.0f, 0.5f, 1.5f, -1.0f, 0.0f, 2.2f, 1.1f, 0.3f, 0.9f};
        std::vector<TokenId> out;
        for (int i = 0; i < 128; ++i) {
            const auto res = chain.value().sample(logits);
            REQUIRE(res.ok());
            chain.value().accept(res.token);
            out.push_back(res.token);
        }
        return out;
    };
    const auto a = run();
    const auto b = run();
    CHECK(a == b);
    // The draw is not degenerate: penalties spread picks over several tokens.
    std::vector<TokenId> distinct(a);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    CHECK(distinct.size() > 2u);
}
