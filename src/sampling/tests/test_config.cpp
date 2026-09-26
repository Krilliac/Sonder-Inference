#include <limits>
#include <stdexcept>
#include <string>

#include "sonder/sampling/sampling.hpp"
#include "sampling_test_util.hpp"

using namespace sonder::inference::sampling;

namespace {
bool has_error(const ValidationResult& r, const std::string& field) {
    for (const auto& e : r.errors) {
        if (e.field == field) return true;
    }
    return false;
}
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();
}  // namespace

TEST_CASE("default config is valid and matches llama.cpp defaults") {
    SamplerConfig c;
    CHECK(validate(c).ok());
    CHECK(validate(c, 32000).ok());
    CHECK_NEAR(c.temperature, 0.8, 1e-6);
    CHECK_EQ(c.top_k, 40);
    CHECK_NEAR(c.top_p, 0.95, 1e-6);
    CHECK_NEAR(c.min_p, 0.05, 1e-6);
    CHECK_NEAR(c.typical_p, 1.0, 1e-6);
    CHECK_EQ(c.penalty_last_n, 64);
    CHECK_NEAR(c.repeat_penalty, 1.0, 1e-6);
    CHECK(c.stage_order == default_stage_order());
}

TEST_CASE("temperature must be finite and non-negative") {
    SamplerConfig c;
    c.temperature = -0.1f;
    CHECK(has_error(validate(c), "temperature"));
    c.temperature = kNaN;
    CHECK(has_error(validate(c), "temperature"));
    c.temperature = kInf;
    CHECK(has_error(validate(c), "temperature"));
    c.temperature = 0.0f;
    CHECK(validate(c).ok());
}

TEST_CASE("top_k must be non-negative large values allowed") {
    SamplerConfig c;
    c.top_k = -1;
    CHECK(has_error(validate(c), "top_k"));
    c.top_k = 0;
    CHECK(validate(c).ok());
    c.top_k = 1 << 30;
    CHECK(validate(c, 10).ok());
}

TEST_CASE("probability parameters must be within 0 and 1") {
    for (float bad : {-0.01f, 1.01f, kNaN, kInf}) {
        SamplerConfig c;
        c.top_p = bad;
        c.min_p = bad;
        c.typical_p = bad;
        const auto r = validate(c);
        CHECK(has_error(r, "top_p"));
        CHECK(has_error(r, "min_p"));
        CHECK(has_error(r, "typical_p"));
    }
    SamplerConfig edge;
    edge.top_p = 0.0f;
    edge.min_p = 1.0f;
    edge.typical_p = 0.0f;
    CHECK(validate(edge).ok());
}

TEST_CASE("penalty fields are validated") {
    SamplerConfig c;
    c.penalty_last_n = -2;
    c.repeat_penalty = 0.0f;
    c.frequency_penalty = kNaN;
    c.presence_penalty = kInf;
    c.min_keep = -1;
    const auto r = validate(c);
    CHECK(has_error(r, "penalty_last_n"));
    CHECK(has_error(r, "repeat_penalty"));
    CHECK(has_error(r, "frequency_penalty"));
    CHECK(has_error(r, "presence_penalty"));
    CHECK(has_error(r, "min_keep"));
    CHECK_EQ(r.errors.size(), std::size_t{5});

    SamplerConfig ok;
    ok.penalty_last_n = -1;
    ok.frequency_penalty = -2.0f;  // negative penalties encourage repetition; allowed
    ok.presence_penalty = 2.0f;
    CHECK(validate(ok).ok());
}

TEST_CASE("logit bias token ids and values are validated") {
    SamplerConfig c;
    c.logit_bias = {{-1, 1.0f}, {5, kNaN}, {6, kInf}, {7, -kInf}, {100, 1.0f}};
    const auto no_vocab = validate(c);
    CHECK(has_error(no_vocab, "logit_bias[0]"));
    CHECK(has_error(no_vocab, "logit_bias[1]"));
    CHECK(has_error(no_vocab, "logit_bias[2]"));
    CHECK(!has_error(no_vocab, "logit_bias[3]"));  // -inf bans the token: allowed
    CHECK(!has_error(no_vocab, "logit_bias[4]"));  // vocab unknown
    const auto with_vocab = validate(c, 50);
    CHECK(has_error(with_vocab, "logit_bias[4]"));
}

TEST_CASE("stop sequences and stop tokens are validated") {
    SamplerConfig c;
    c.stop_sequences = {"ok", ""};
    c.stop_tokens = {2, -5, 99};
    const auto r = validate(c, 10);
    CHECK(!has_error(r, "stop_sequences[0]"));
    CHECK(has_error(r, "stop_sequences[1]"));
    CHECK(!has_error(r, "stop_tokens[0]"));
    CHECK(has_error(r, "stop_tokens[1]"));
    CHECK(has_error(r, "stop_tokens[2]"));
}

TEST_CASE("stage order rejects duplicates accepts subsets and empty") {
    SamplerConfig c;
    c.stage_order = {StageKind::TopK, StageKind::Temperature, StageKind::TopK};
    CHECK(has_error(validate(c), "stage_order[2]"));
    c.stage_order = {StageKind::Temperature, StageKind::MinP};
    CHECK(validate(c).ok());
    c.stage_order.clear();
    CHECK(validate(c).ok());
}

TEST_CASE("zero vocab size is rejected") {
    SamplerConfig c;
    CHECK(has_error(validate(c, 0), "vocab_size"));
}

TEST_CASE("error string names every failing field") {
    SamplerConfig c;
    c.top_p = 2.0f;
    c.top_k = -3;
    const std::string s = validate(c).to_string();
    CHECK(s.find("sampling config invalid") != std::string::npos);
    CHECK(s.find("top_p") != std::string::npos);
    CHECK(s.find("top_k") != std::string::npos);
    CHECK(s.find("got 2") != std::string::npos);
    CHECK(validate(SamplerConfig{}).to_string() == "sampling config valid");
}

TEST_CASE("build_chain throws invalid_argument with the validation message") {
    SamplerConfig c;
    c.min_p = -1.0f;
    bool threw = false;
    try {
        (void)build_chain(c);
    } catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("min_p") != std::string::npos);
    }
    CHECK(threw);
    SamplerConfig v;
    v.logit_bias = {{20, 1.0f}};
    CHECK_THROWS((void)build_chain(v, nullptr, 10));
}

TEST_CASE("stage names round-trip through parse_stage") {
    for (const auto kind : default_stage_order()) {
        const auto parsed = parse_stage(to_string(kind));
        CHECK(parsed.has_value());
        CHECK(*parsed == kind);
    }
    CHECK(parse_stage("typical_p") == StageKind::TypicalP);
    CHECK(parse_stage("temp") == StageKind::Temperature);
    CHECK(!parse_stage("xtc").has_value());
    CHECK(!parse_stage("").has_value());
}
