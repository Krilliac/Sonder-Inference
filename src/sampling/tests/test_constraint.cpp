#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "sonder/sampling/sampling.hpp"
#include "sampling_test_util.hpp"

using namespace sonder::inference::sampling;

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();

// Toy stateful constraint: forces a fixed token sequence, then completes.
class SequenceConstraint final : public Constraint {
public:
    explicit SequenceConstraint(std::vector<TokenId> seq) : seq_(std::move(seq)) {}
    std::string_view name() const noexcept override { return "sequence"; }
    void apply(Candidates& c) override {
        for (auto& t : c.data) {
            if (pos_ >= seq_.size() || t.id != seq_[pos_]) t.logit = -kInf;
        }
        c.sorted = false;
    }
    void accept(TokenId) override { ++pos_; }
    void reset() override { pos_ = 0; }
    bool is_complete() const noexcept override { return pos_ >= seq_.size(); }

private:
    std::vector<TokenId> seq_;
    std::size_t pos_ = 0;
};

SamplerConfig random_config(std::uint64_t seed) {
    SamplerConfig c;
    c.seed = seed;
    c.temperature = 1.5f;
    c.top_k = 0;
    c.top_p = 1.0f;
    c.min_p = 0.0f;
    return c;
}
}  // namespace

TEST_CASE("allowlist masks every other token") {
    TokenAllowlistConstraint a({1, 3});
    Candidates c;
    for (TokenId i = 0; i < 5; ++i) c.data.push_back({i, 1.0f, 0.0f});
    a.apply(c);
    CHECK(c.data[0].logit == -kInf);
    CHECK(c.data[1].logit == 1.0f);
    CHECK(c.data[2].logit == -kInf);
    CHECK(c.data[3].logit == 1.0f);
    CHECK(c.data[4].logit == -kInf);
    CHECK(a.name() == "allowlist");
    CHECK(!a.is_complete());
}

TEST_CASE("chain with allowlist only samples allowed tokens") {
    auto chain = build_chain(random_config(8), std::make_shared<TokenAllowlistConstraint>(std::vector<TokenId>{2, 5}));
    const std::vector<float> logits = {9.0f, 8.0f, 0.0f, 7.0f, 6.0f, 0.1f};
    std::set<TokenId> seen;
    for (int i = 0; i < 500; ++i) {
        const auto r = chain.sample(logits);
        CHECK(r.ok());
        seen.insert(r.token);
    }
    CHECK((seen == std::set<TokenId>{2, 5}));
}

TEST_CASE("constraint runs before truncation stages") {
    SamplerConfig c = random_config(1);
    c.top_k = 1;  // would pick token 0 without the constraint
    auto chain = build_chain(c, std::make_shared<TokenAllowlistConstraint>(std::vector<TokenId>{3}));
    CHECK_EQ(chain.sample(std::vector<float>{10.0f, 1.0f, 1.0f, -5.0f}).token, 3);
}

TEST_CASE("over-constrained sampling reports no viable candidates") {
    auto chain = build_chain(random_config(1), std::make_shared<TokenAllowlistConstraint>(std::vector<TokenId>{42}));
    CHECK(chain.sample(std::vector<float>{1.0f, 2.0f}).status == SampleStatus::NoViableCandidates);
}

TEST_CASE("stateful constraint advances on accept and resets") {
    auto seq = std::make_shared<SequenceConstraint>(std::vector<TokenId>{4, 1, 3});
    auto chain = build_chain(random_config(99), seq);
    const std::vector<float> logits(6, 0.0f);
    std::vector<TokenId> out;
    while (!seq->is_complete()) {
        const auto r = chain.sample(logits);
        REQUIRE(r.ok());  // REQUIRE stops the loop on failure
        out.push_back(r.token);
        chain.accept(r.token);
    }
    CHECK((out == std::vector<TokenId>{4, 1, 3}));
    CHECK(chain.sample(logits).status == SampleStatus::NoViableCandidates);
    chain.reset();
    CHECK(!seq->is_complete());
    CHECK_EQ(chain.sample(logits).token, 4);
}

TEST_CASE("constraint can be replaced or cleared") {
    auto chain = build_chain(random_config(3), std::make_shared<TokenAllowlistConstraint>(std::vector<TokenId>{0}));
    const std::vector<float> logits = {0.0f, 50.0f};
    CHECK_EQ(chain.sample(logits).token, 0);
    chain.set_constraint(nullptr);
    CHECK_EQ(chain.sample(logits).token, 1);
}
