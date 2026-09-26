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
