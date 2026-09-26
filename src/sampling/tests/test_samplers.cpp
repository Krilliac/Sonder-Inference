#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "sonder/sampling/sampling.hpp"
#include "sampling_test_util.hpp"

using namespace sonder::inference::sampling;

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();

Candidates make(const std::vector<float>& logits) {
    Candidates c;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        c.data.push_back({static_cast<TokenId>(i), logits[i], 0.0f});
    }
    return c;
}

// Logits whose softmax is exactly the given probabilities.
Candidates from_probs(const std::vector<double>& probs) {
    std::vector<float> logits;
    for (double p : probs) logits.push_back(static_cast<float>(std::log(p)));
    return make(logits);
}

std::vector<TokenId> ids(const Candidates& c) {
    std::vector<TokenId> out;
    for (const auto& t : c.data) out.push_back(t.id);
    std::sort(out.begin(), out.end());
    return out;
}

float logit_of(const Candidates& c, TokenId id) {
    for (const auto& t : c.data) {
        if (t.id == id) return t.logit;
    }
    return std::numeric_limits<float>::quiet_NaN();
}
}  // namespace

// ------------------------------------------------------------------ helpers
TEST_CASE("softmax normalises and orders") {
    auto c = make({1.0f, 2.0f, 3.0f});
    CHECK(softmax(c));
    double sum = 0.0;
    for (const auto& t : c.data) sum += t.p;
    CHECK_NEAR(sum, 1.0, 1e-6);
    CHECK((c.data[2].p > c.data[1].p && c.data[1].p > c.data[0].p));
}

TEST_CASE("softmax is stable for huge logits") {
    auto c = make({10000.0f, 10000.0f, -10000.0f});
    CHECK(softmax(c));
    CHECK_NEAR(c.data[0].p, 0.5, 1e-6);
    CHECK_NEAR(c.data[2].p, 0.0, 1e-12);
}

TEST_CASE("softmax with all -inf reports no viable candidate") {
    auto c = make({-kInf, -kInf});
    CHECK(!softmax(c));
    CHECK((c.data[0].p == 0.0f && c.data[1].p == 0.0f));
    CHECK_EQ(viable_count(c), std::size_t{0});
    Candidates empty;
    CHECK(!softmax(empty));
}

TEST_CASE("softmax with +inf is uniform over the +inf tokens") {
    auto c = make({kInf, 3.0f, kInf, -kInf});
    CHECK(softmax(c));
    CHECK_NEAR(c.data[0].p, 0.5, 1e-7);
    CHECK_NEAR(c.data[2].p, 0.5, 1e-7);
    CHECK(c.data[1].p == 0.0f);
}

TEST_CASE("canonical sort breaks ties by lower token id") {
    auto c = make({1.0f, 5.0f, 5.0f, 2.0f});
    sort_canonical(c);
    CHECK_EQ(c.data[0].id, 1);
    CHECK_EQ(c.data[1].id, 2);
    CHECK_EQ(c.data[2].id, 3);
    CHECK_EQ(c.data[3].id, 0);
}

// ------------------------------------------------------------------- top-k
TEST_CASE("top_k keeps the k highest logits") {
    auto c = make({0.1f, 3.0f, 2.0f, -1.0f, 2.5f});
    TopKSampler(2).apply(c);
    CHECK((ids(c) == std::vector<TokenId>{1, 4}));
    CHECK(c.sorted);
}

TEST_CASE("top_k 0 disables and k > vocab keeps everything") {
    auto a = make({1.0f, 2.0f, 3.0f});
    TopKSampler(0).apply(a);
    CHECK_EQ(a.size(), std::size_t{3});
    auto b = make({1.0f, 2.0f, 3.0f});
    TopKSampler(1000).apply(b);
    CHECK_EQ(b.size(), std::size_t{3});
    auto d = make({1.0f, 2.0f, 3.0f});
    TopKSampler(3).apply(d);
    CHECK_EQ(d.size(), std::size_t{3});
}

TEST_CASE("top_k ties keep the lower token id") {
    auto c = make({1.0f, 1.0f, 1.0f});
    TopKSampler(1).apply(c);
    CHECK_EQ(c.size(), std::size_t{1});
    CHECK_EQ(c.data[0].id, 0);
}

// ------------------------------------------------------------------- top-p
TEST_CASE("top_p keeps smallest prefix reaching p (inclusive)") {
    auto a = make({0.0f, 0.0f});  // exactly 0.5 / 0.5
    TopPSampler(0.5f, 1).apply(a);
    CHECK((ids(a) == std::vector<TokenId>{0}));
    auto b = from_probs({0.5, 0.3, 0.2});
    TopPSampler(0.6f, 1).apply(b);
    CHECK((ids(b) == std::vector<TokenId>{0, 1}));
    auto d = from_probs({0.2, 0.5, 0.3});
    TopPSampler(0.75f, 1).apply(d);
    CHECK((ids(d) == std::vector<TokenId>{1, 2}));
}

TEST_CASE("top_p = 1 is a no-op") {
    auto c = from_probs({0.97, 0.01, 0.01, 0.01});
    TopPSampler(1.0f, 1).apply(c);
    CHECK_EQ(c.size(), std::size_t{4});
}

TEST_CASE("top_p = 0 keeps min_keep (at least one)") {
    auto a = from_probs({0.4, 0.35, 0.25});
    TopPSampler(0.0f, 0).apply(a);
    CHECK((ids(a) == std::vector<TokenId>{0}));
    auto b = from_probs({0.4, 0.35, 0.25});
    TopPSampler(0.0f, 2).apply(b);
    CHECK((ids(b) == std::vector<TokenId>{0, 1}));
}

TEST_CASE("top_p ignores -inf tokens and tolerates all -inf") {
    auto a = make({0.0f, -kInf, 0.0f, -kInf});
    TopPSampler(0.9f, 1).apply(a);
    CHECK((ids(a) == std::vector<TokenId>{0, 2}));
    auto b = make({-kInf, -kInf});
    TopPSampler(0.5f, 1).apply(b);
    CHECK_EQ(b.size(), std::size_t{2});
}

// ------------------------------------------------------------------- min-p
TEST_CASE("min_p keeps tokens with p >= min_p * p_max") {
    auto c = from_probs({0.5, 0.3, 0.2});
    MinPSampler(0.5f, 1).apply(c);  // threshold 0.25
    CHECK((ids(c) == std::vector<TokenId>{0, 1}));
}

TEST_CASE("min_p = 0 disables min_p = 1 keeps only ties with max") {
    auto a = from_probs({0.5, 0.3, 0.2});
    MinPSampler(0.0f, 1).apply(a);
    CHECK_EQ(a.size(), std::size_t{3});
    auto b = make({2.0f, 2.0f, 1.0f});
    MinPSampler(1.0f, 1).apply(b);
    CHECK((ids(b) == std::vector<TokenId>{0, 1}));
}

TEST_CASE("min_p honours min_keep") {
    auto c = from_probs({0.9, 0.05, 0.05});
    MinPSampler(0.5f, 2).apply(c);
    CHECK_EQ(c.size(), std::size_t{2});
    CHECK_EQ(c.data[0].id, 0);
}

// --------------------------------------------------------------- typical-p
TEST_CASE("typical_p = 1 is a no-op") {
    auto c = from_probs({0.7, 0.2, 0.1});
    TypicalPSampler(1.0f, 1).apply(c);
    CHECK_EQ(c.size(), std::size_t{3});
}

TEST_CASE("typical_p on a uniform distribution cuts by strict cumulative mass") {
    auto c = from_probs({0.25, 0.25, 0.25, 0.25});
    TypicalPSampler(0.5f, 1).apply(c);  // cum .25, .5, .75 -> first > .5 is 3 tokens
    CHECK((ids(c) == std::vector<TokenId>{0, 1, 2}));
}

TEST_CASE("typical_p can drop the most likely token") {
    // H = 1.0889; |-log p - H|: 0.4 -> 0.173, 0.3 -> 0.115. Tokens 1,2 are more typical.
    auto c = from_probs({0.4, 0.3, 0.3});
    TypicalPSampler(0.5f, 1).apply(c);
    CHECK((ids(c) == std::vector<TokenId>{1, 2}));
    CHECK(c.sorted);
}

TEST_CASE("typical_p keeps the peak on skewed distributions and handles -inf") {
    auto c = from_probs({0.7, 0.2, 0.1});
    TypicalPSampler(0.5f, 1).apply(c);
    CHECK((ids(c) == std::vector<TokenId>{0}));
    auto d = make({0.0f, -kInf, 0.0f});
    TypicalPSampler(0.9f, 1).apply(d);
    CHECK((ids(d) == std::vector<TokenId>{0, 2}));
    auto e = make({-kInf, -kInf});
    TypicalPSampler(0.5f, 1).apply(e);
    CHECK_EQ(e.size(), std::size_t{2});
}

// ------------------------------------------------------------- temperature
TEST_CASE("temperature divides logits") {
    auto c = make({2.0f, -4.0f, -kInf});
    TemperatureSampler(2.0f).apply(c);
    CHECK_NEAR(logit_of(c, 0), 1.0, 1e-6);
    CHECK_NEAR(logit_of(c, 1), -2.0, 1e-6);
    CHECK(logit_of(c, 2) == -kInf);
    auto d = make({2.0f});
    TemperatureSampler(1.0f).apply(d);
    CHECK_NEAR(logit_of(d, 0), 2.0, 0.0);
}

TEST_CASE("temperature <= 0 keeps only the argmax (lowest id on ties)") {
    auto c = make({1.0f, 3.0f, 3.0f, 2.0f});
    TemperatureSampler(0.0f).apply(c);
    CHECK_EQ(c.size(), std::size_t{1});
    CHECK_EQ(c.data[0].id, 1);
    auto all_neg = make({-kInf, -kInf});
    TemperatureSampler(0.0f).apply(all_neg);
    CHECK_EQ(all_neg.size(), std::size_t{2});
}

// --------------------------------------------------------------- penalties
TEST_CASE("repeat penalty divides positive and multiplies negative logits") {
    PenaltiesSampler s(64, 2.0f, 0.0f, 0.0f);
    s.accept(0);
    s.accept(1);
    auto c = make({4.0f, -2.0f, 4.0f});
    s.apply(c);
    CHECK_NEAR(logit_of(c, 0), 2.0, 1e-6);
    CHECK_NEAR(logit_of(c, 1), -4.0, 1e-6);
    CHECK_NEAR(logit_of(c, 2), 4.0, 1e-6);  // unseen token untouched
}

TEST_CASE("frequency scales with count presence applies once") {
    PenaltiesSampler s(64, 1.0f, 0.5f, 1.0f);
    s.accept(0);
    s.accept(0);
    s.accept(0);
    s.accept(1);
    CHECK_EQ(s.count(0), 3);
    auto c = make({10.0f, 10.0f, 10.0f});
    s.apply(c);
    CHECK_NEAR(logit_of(c, 0), 10.0 - 1.5 - 1.0, 1e-6);
    CHECK_NEAR(logit_of(c, 1), 10.0 - 0.5 - 1.0, 1e-6);
    CHECK_NEAR(logit_of(c, 2), 10.0, 1e-6);
}

TEST_CASE("penalty window evicts old tokens - 0 disables - -1 unbounded") {
    PenaltiesSampler w(2, 2.0f, 0.0f, 0.0f);
    w.accept(5);
    w.accept(6);
    w.accept(7);
    CHECK_EQ(w.count(5), 0);
    CHECK_EQ(w.count(7), 1);

    PenaltiesSampler off(0, 2.0f, 1.0f, 1.0f);
    off.accept(0);
    auto c = make({4.0f});
    off.apply(c);
    CHECK_NEAR(logit_of(c, 0), 4.0, 0.0);

    PenaltiesSampler all(-1, 1.0f, 1.0f, 0.0f);
    for (int i = 0; i < 1000; ++i) all.accept(3);
    CHECK_EQ(all.count(3), 1000);
}

TEST_CASE("penalties reset and clone are independent") {
    PenaltiesSampler s(8, 2.0f, 0.0f, 0.0f);
    s.accept(1);
    auto copy = s.clone();
    s.reset();
    CHECK_EQ(s.count(1), 0);
    auto c = make({0.0f, 4.0f});
    copy->apply(c);
    CHECK_NEAR(logit_of(c, 1), 2.0, 1e-6);
}

// -------------------------------------------------------------- logit bias
TEST_CASE("logit bias adds accumulates bans and ignores unknown ids") {
    LogitBiasSampler s({{0, 1.0f}, {0, 0.5f}, {1, -kInf}, {99, 3.0f}});
    auto c = make({1.0f, 1.0f, 1.0f});
    s.apply(c);
    CHECK_NEAR(logit_of(c, 0), 2.5, 1e-6);
    CHECK(logit_of(c, 1) == -kInf);
    CHECK_NEAR(logit_of(c, 2), 1.0, 1e-6);
}

TEST_CASE("make_stage builds the right stage for each kind") {
    SamplerConfig cfg;
    CHECK(make_stage(StageKind::Penalties, cfg)->name() == "penalties");
    CHECK(make_stage(StageKind::TopK, cfg)->name() == "top_k");
    CHECK(make_stage(StageKind::TypicalP, cfg)->name() == "typical_p");
    CHECK(make_stage(StageKind::TopP, cfg)->name() == "top_p");
    CHECK(make_stage(StageKind::MinP, cfg)->name() == "min_p");
    CHECK(make_stage(StageKind::Temperature, cfg)->name() == "temperature");
}

TEST_CASE("stages tolerate empty candidate sets") {
    Candidates c;
    TopKSampler(3).apply(c);
    TopPSampler(0.5f, 1).apply(c);
    MinPSampler(0.5f, 1).apply(c);
    TypicalPSampler(0.5f, 1).apply(c);
    TemperatureSampler(0.0f).apply(c);
    TemperatureSampler(0.7f).apply(c);
    CHECK(c.empty());
}
