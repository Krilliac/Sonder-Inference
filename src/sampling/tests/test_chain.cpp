#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "sonder/sampling/sampling.hpp"
#include "sampling_test_util.hpp"

using namespace sonder::inference::sampling;

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

SamplerConfig plain(std::uint64_t seed) {
    SamplerConfig c;
    c.seed = seed;
    c.temperature = 1.0f;
    c.top_k = 0;
    c.top_p = 1.0f;
    c.min_p = 0.0f;
    return c;
}

std::vector<TokenId> draw(SamplerChain& chain, const std::vector<float>& logits, int n) {
    std::vector<TokenId> out;
    for (int i = 0; i < n; ++i) {
        const auto r = chain.sample(logits);
        out.push_back(r.token);
    }
    return out;
}

const std::vector<float> kLogits = {1.0f, 2.0f, 0.5f, 1.5f, -1.0f, 0.0f, 2.2f, 1.1f};

class CountingConstraint final : public Constraint {
public:
    std::string_view name() const noexcept override { return "counting"; }
    void apply(Candidates&) override { ++applied; }
    void accept(TokenId) override { ++accepted; }
    void reset() override { ++resets; }
    int applied = 0;
    int accepted = 0;
    int resets = 0;
};
}  // namespace

TEST_CASE("greedy picks argmax with lowest-id tie break") {
    SamplerConfig c = plain(1);
    c.greedy = true;
    auto chain = build_chain(c);
    CHECK_EQ(chain.sample(kLogits).token, 6);
    const std::vector<float> ties = {-kInf, 3.0f, 3.0f};
    const auto r = chain.sample(ties);
    CHECK_EQ(r.token, 1);
    CHECK_NEAR(r.probability, 0.5, 1e-6);
}

TEST_CASE("greedy chain contains only logit bias and penalties") {
    SamplerConfig c;
    c.greedy = true;
    c.logit_bias = {{0, 1.0f}};
    auto chain = build_chain(c);
    const auto names = chain.stage_names();
    CHECK_EQ(names.size(), std::size_t{2});
    CHECK(names[0] == "logit_bias");
    CHECK(names[1] == "penalties");
}

TEST_CASE("temperature 0 is deterministic regardless of seed") {
    for (std::uint64_t seed : {1ULL, 2ULL, 999ULL}) {
        SamplerConfig c;
        c.seed = seed;
        c.temperature = 0.0f;
        auto chain = build_chain(c);
        for (int i = 0; i < 20; ++i) CHECK_EQ(chain.sample(kLogits).token, 6);
    }
}

TEST_CASE("fixed seed reproduces the exact token sequence") {
    auto a = build_chain(plain(1234));
    auto b = build_chain(plain(1234));
    CHECK(draw(a, kLogits, 500) == draw(b, kLogits, 500));
}

TEST_CASE("different seeds produce different sequences") {
    auto a = build_chain(plain(1));
    auto b = build_chain(plain(2));
    CHECK(draw(a, kLogits, 200) != draw(b, kLogits, 200));
}

TEST_CASE("reseed and copy continue identically") {
    auto a = build_chain(plain(77));
    const auto first = draw(a, kLogits, 50);
    a.reseed(77);
    CHECK(draw(a, kLogits, 50) == first);

    auto b = build_chain(plain(5));
    (void)draw(b, kLogits, 10);
    SamplerChain fork = b;
    CHECK(draw(b, kLogits, 100) == draw(fork, kLogits, 100));
}

TEST_CASE("golden sequence for seed 42 with default config") {
    // Cross-checked against an independent Python reference of the same
    // semantics. Pinned so any change to RNG, ordering or stage semantics is noticed and
    // so Windows/Linux CI prove cross-platform reproducibility.
    SamplerConfig c;
    c.seed = 42;
    auto chain = build_chain(c);
    const auto got = draw(chain, kLogits, 16);
    const std::vector<TokenId> golden = {6, 1, 3, 0, 2, 3, 3, 7, 3, 1, 3, 6, 7, 6, 3, 0};
    CHECK(got == golden);
}

TEST_CASE("empirical frequencies match softmax with p = 1 and k > vocab") {
    SamplerConfig c = plain(2024);
    c.top_k = 1000;  // larger than vocab
    c.top_p = 1.0f;
    auto chain = build_chain(c, nullptr, kLogits.size());
    std::map<TokenId, int> counts;
    const int n = 40000;
    for (int i = 0; i < n; ++i) ++counts[chain.sample(kLogits).token];
    double z = 0.0;
    for (float l : kLogits) z += std::exp(static_cast<double>(l));
    for (std::size_t i = 0; i < kLogits.size(); ++i) {
        const double expected = std::exp(static_cast<double>(kLogits[i])) / z;
        CHECK_NEAR(static_cast<double>(counts[static_cast<TokenId>(i)]) / n, expected, 0.01);
    }
}

TEST_CASE("top_k = 1 always yields argmax") {
    SamplerConfig c = plain(3);
    c.top_k = 1;
    auto chain = build_chain(c);
    for (int i = 0; i < 50; ++i) CHECK_EQ(chain.sample(kLogits).token, 6);
}

TEST_CASE("empty logits and all -inf report errors instead of a token") {
    auto chain = build_chain(plain(1));
    const auto e = chain.sample(std::vector<float>{});
    CHECK(e.status == SampleStatus::EmptyLogits);
    CHECK_EQ(e.token, kInvalidToken);
    const auto n = chain.sample(std::vector<float>{-kInf, -kInf, -kInf});
    CHECK(n.status == SampleStatus::NoViableCandidates);
    CHECK(!n.ok());
    CHECK(to_string(n.status) != to_string(SampleStatus::Ok));
}

TEST_CASE("banning every token through logit bias yields no viable candidates") {
    SamplerConfig c = plain(1);
    c.logit_bias = {{0, -kInf}, {1, -kInf}};
    auto chain = build_chain(c);
    CHECK(chain.sample(std::vector<float>{1.0f, 2.0f}).status == SampleStatus::NoViableCandidates);
}

TEST_CASE("NaN logits are treated as -inf") {
    auto chain = build_chain(plain(9));
    for (int i = 0; i < 20; ++i) {
        const auto r = chain.sample(std::vector<float>{kNaN, 0.0f, kNaN});
        CHECK(r.ok());
        CHECK_EQ(r.token, 1);
        CHECK_NEAR(r.probability, 1.0, 1e-6);
    }
}

TEST_CASE("+inf logit always wins") {
    auto chain = build_chain(plain(11));
    for (int i = 0; i < 20; ++i) CHECK_EQ(chain.sample(std::vector<float>{100.0f, kInf, 5.0f}).token, 1);
}

TEST_CASE("single token vocab always returns it") {
    SamplerConfig c;
    c.seed = 1;
    auto chain = build_chain(c);
    const auto r = chain.sample(std::vector<float>{-3.0f});
    CHECK(r.ok());
    CHECK_EQ(r.token, 0);
    CHECK_NEAR(r.probability, 1.0, 1e-6);
}

TEST_CASE("accepted tokens are penalised in later steps") {
    SamplerConfig c = plain(1);
    c.greedy = true;
    c.repeat_penalty = 1.5f;
    auto chain = build_chain(c);
    const std::vector<float> logits = {5.0f, 4.9f};
    CHECK_EQ(chain.sample(logits).token, 0);
    chain.accept(0);
    CHECK_EQ(chain.sample(logits).token, 1);
    chain.reset();
    CHECK_EQ(chain.sample(logits).token, 0);
}

TEST_CASE("accept_prompt feeds penalties but not the constraint") {
    SamplerConfig c = plain(1);
    c.greedy = true;
    c.presence_penalty = 10.0f;
    auto constraint = std::make_shared<CountingConstraint>();
    auto chain = build_chain(c, constraint);
    const std::vector<TokenId> prompt = {0, 0};
    chain.accept_prompt(prompt);
    CHECK_EQ(constraint->accepted, 0);
    CHECK_EQ(chain.sample(std::vector<float>{5.0f, 4.0f}).token, 1);
    chain.accept(1);
    CHECK_EQ(constraint->accepted, 1);
    CHECK(constraint->applied >= 1);
    chain.reset();
    CHECK_EQ(constraint->resets, 1);
}

TEST_CASE("custom stage order is honoured") {
    SamplerConfig c = plain(1);
    c.stage_order = {StageKind::Temperature, StageKind::MinP, StageKind::TopK};
    auto chain = build_chain(c);
    const auto names = chain.stage_names();
    CHECK_EQ(names.size(), std::size_t{3});
    CHECK(names[0] == "temperature");
    CHECK(names[1] == "min_p");
    CHECK(names[2] == "top_k");
}

TEST_CASE("default order follows llama.cpp") {
    SamplerConfig c;
    c.seed = 1;
    auto chain = build_chain(c);
    const auto names = chain.stage_names();
    const std::vector<std::string_view> expected = {"penalties", "top_k", "typical_p",
                                                    "top_p",     "min_p", "temperature"};
    CHECK(names == expected);
}

TEST_CASE("order matters: temperature before min_p changes the kept set") {
    // Logits {2, 0}: p = {0.881, 0.119}; min_p 0.2 -> threshold 0.176 drops token 1.
    // At temperature 4 first: p = {0.622, 0.378} -> token 1 survives.
    const std::vector<float> logits = {2.0f, 0.0f};
    SamplerConfig c = plain(1);
    c.min_p = 0.2f;
    c.temperature = 4.0f;
    c.stage_order = {StageKind::MinP, StageKind::Temperature};
    auto late = build_chain(c);
    (void)late.sample(logits);
    CHECK_EQ(late.last_candidates().size(), std::size_t{1});
    c.stage_order = {StageKind::Temperature, StageKind::MinP};
    auto early = build_chain(c);
    (void)early.sample(logits);
    CHECK_EQ(early.last_candidates().size(), std::size_t{2});
}

TEST_CASE("stop tokens are reported") {
    SamplerConfig c;
    c.stop_tokens = {2, 7};
    auto chain = build_chain(c);
    CHECK(chain.is_stop_token(2));
    CHECK(chain.is_stop_token(7));
    CHECK(!chain.is_stop_token(3));
}

TEST_CASE("manually composed chain works and rejects null stages") {
    SamplerChain chain(5, Selector::Greedy);
    chain.add(std::make_unique<LogitBiasSampler>(std::vector<LogitBias>{{0, 10.0f}}))
        .add(std::make_unique<TopKSampler>(2));
    CHECK_EQ(chain.stage_count(), std::size_t{2});
    CHECK_EQ(chain.sample(kLogits).token, 0);
    CHECK_THROWS((void)chain.add(nullptr));
}

TEST_CASE("unseeded config still samples valid tokens") {
    SamplerConfig c;  // seed = nullopt -> entropy seed
    auto chain = build_chain(c);
    for (int i = 0; i < 20; ++i) {
        const auto r = chain.sample(kLogits);
        CHECK(r.ok());
        CHECK((r.token >= 0 && r.token < static_cast<TokenId>(kLogits.size())));
    }
}
