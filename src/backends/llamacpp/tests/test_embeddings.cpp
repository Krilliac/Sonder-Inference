// Embeddings helpers: L2 normalisation maths, request limits, pooling choice.
// No model weights needed.
#include <doctest/doctest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "embeddings.hpp"

namespace lc = sonder::backends::llamacpp;

namespace {

double norm_of(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
    return std::sqrt(s);
}

lc::EmbedLimitCheck check(std::vector<std::size_t> lengths, lc::EmbedLimits limits) {
    return lc::CheckEmbedLimits(lengths, limits);
}

}  // namespace

// ---------------------------------------------------------------- L2 maths

TEST_CASE("embeddings: L2Normalize scales to unit length and returns the original norm") {
    std::vector<float> v{3.0f, 4.0f};
    const double n = lc::L2Normalize(v);
    CHECK(n == doctest::Approx(5.0));
    CHECK(v[0] == doctest::Approx(0.6f));
    CHECK(v[1] == doctest::Approx(0.8f));
    CHECK(norm_of(v) == doctest::Approx(1.0).epsilon(1e-6));

    std::vector<float> w(768);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(static_cast<float>(i) * 0.37f) * 12.5f + 0.25f;
    const double before = norm_of(w);
    CHECK(lc::L2Normalize(w) == doctest::Approx(before).epsilon(1e-9));
    CHECK(norm_of(w) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("embeddings: L2Normalize is idempotent and keeps direction") {
    std::vector<float> v{1.0f, -2.0f, 2.0f};
    lc::L2Normalize(v);
    const std::vector<float> once = v;
    CHECK(lc::L2Normalize(v) == doctest::Approx(1.0).epsilon(1e-6));
    for (std::size_t i = 0; i < v.size(); ++i) CHECK(v[i] == doctest::Approx(once[i]).epsilon(1e-6));
    CHECK(once[0] == doctest::Approx(1.0f / 3.0f));
    CHECK(once[1] == doctest::Approx(-2.0f / 3.0f));
}

TEST_CASE("embeddings: L2Normalize survives extreme magnitudes") {
    std::vector<float> big{3e30f, 4e30f};  // squares overflow float, not double
    CHECK(lc::L2Normalize(big) == doctest::Approx(5e30).epsilon(1e-6));
    CHECK(big[0] == doctest::Approx(0.6f));
    std::vector<float> tiny{3e-30f, 4e-30f};  // squares underflow float, not double
    CHECK(lc::L2Normalize(tiny) == doctest::Approx(5e-30).epsilon(1e-6));
    CHECK(tiny[1] == doctest::Approx(0.8f));
}

TEST_CASE("embeddings: L2Normalize leaves empty, zero and non-finite vectors alone") {
    std::vector<float> empty;
    CHECK(lc::L2Normalize(empty) == 0.0);

    std::vector<float> zero(8, 0.0f);
    CHECK(lc::L2Normalize(zero) == 0.0);
    for (float x : zero) CHECK(x == 0.0f);

    std::vector<float> with_nan{1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};
    CHECK(lc::L2Normalize(with_nan) == 0.0);
    CHECK(with_nan[0] == 1.0f);
    CHECK(with_nan[2] == 2.0f);

    std::vector<float> with_inf{1.0f, std::numeric_limits<float>::infinity()};
    CHECK(lc::L2Normalize(with_inf) == 0.0);
    CHECK(with_inf[0] == 1.0f);
}

// ---------------------------------------------------------------- limits

TEST_CASE("embeddings: CheckEmbedLimits accepts a request inside every limit") {
    lc::EmbedLimits lim;
    lim.max_inputs = 3;
    lim.max_input_tokens = 10;
    lim.max_total_tokens = 25;
    const auto r = check({10, 10, 5}, lim);  // 3 inputs, longest 10, total exactly 25
    CHECK(r.ok);
    CHECK(r.violation == lc::EmbedLimitViolation::none);
    CHECK(r.message.empty());
}

TEST_CASE("embeddings: CheckEmbedLimits names each violation") {
    lc::EmbedLimits lim;
    lim.max_inputs = 2;
    lim.max_input_tokens = 8;
    lim.max_total_tokens = 12;

    auto none = check({}, lim);
    CHECK_FALSE(none.ok);
    CHECK(none.violation == lc::EmbedLimitViolation::no_inputs);

    auto empty_one = check({4, 0}, lim);
    CHECK_FALSE(empty_one.ok);
    CHECK(empty_one.violation == lc::EmbedLimitViolation::empty_input);
    CHECK(empty_one.index == 1u);

    auto many = check({1, 1, 1}, lim);
    CHECK_FALSE(many.ok);
    CHECK(many.violation == lc::EmbedLimitViolation::too_many_inputs);
    CHECK(many.message.find("2") != std::string::npos);

    auto longone = check({3, 9}, lim);
    CHECK_FALSE(longone.ok);
    CHECK(longone.violation == lc::EmbedLimitViolation::input_too_long);
    CHECK(longone.index == 1u);
    CHECK(longone.message.find("8") != std::string::npos);

    auto total = check({8, 5}, lim);  // each fine, 13 > 12
    CHECK_FALSE(total.ok);
    CHECK(total.violation == lc::EmbedLimitViolation::total_too_long);
    CHECK(total.message.find("12") != std::string::npos);
}

TEST_CASE("embeddings: CheckEmbedLimits treats 0 as no limit and cannot be wrapped around") {
    lc::EmbedLimits open;
    open.max_inputs = 0;
    open.max_input_tokens = 0;
    open.max_total_tokens = 0;
    CHECK(check(std::vector<std::size_t>(1000, 5000), open).ok);

    lc::EmbedLimits lim;
    lim.max_inputs = 0;
    lim.max_input_tokens = 0;
    lim.max_total_tokens = 100;
    const std::size_t huge = std::numeric_limits<std::size_t>::max() - 1;
    const auto r = check({huge, huge, 5}, lim);  // naive sum wraps to a small number
    CHECK_FALSE(r.ok);
    CHECK(r.violation == lc::EmbedLimitViolation::total_too_long);
}

// ---------------------------------------------------------------- pooling

TEST_CASE("embeddings: pooling metadata key and value parsing") {
    CHECK(lc::PoolingMetadataKey("bert") == "bert.pooling_type");
    CHECK(lc::PoolingMetadataKey("qwen3") == "qwen3.pooling_type");
    CHECK(lc::PoolingMetadataKey("").empty());

    CHECK(lc::ParsePoolingMetadata("0") == std::optional<lc::PoolingType>(lc::PoolingType::kNone));
    CHECK(lc::ParsePoolingMetadata("1") == std::optional<lc::PoolingType>(lc::PoolingType::kMean));
    CHECK(lc::ParsePoolingMetadata("2") == std::optional<lc::PoolingType>(lc::PoolingType::kCls));
    CHECK(lc::ParsePoolingMetadata("3") == std::optional<lc::PoolingType>(lc::PoolingType::kLast));
    CHECK(lc::ParsePoolingMetadata("4") == std::optional<lc::PoolingType>(lc::PoolingType::kRank));
    for (const char* bad : {"", "5", "-1", "10", "1 ", " 1", "mean", "1x", "01"}) {
        CAPTURE(bad);
        CHECK_FALSE(lc::ParsePoolingMetadata(bad).has_value());
    }
}

TEST_CASE("embeddings: ResolveEmbedPooling honours the request, then the model, then the fallback") {
    using P = lc::PoolingType;
    // Explicit request always wins, whatever the model says.
    CHECK(lc::ResolveEmbedPooling(P::kCls, P::kLast, false) == P::kCls);
    CHECK(lc::ResolveEmbedPooling(P::kMean, std::nullopt, true) == P::kMean);
    CHECK(lc::ResolveEmbedPooling(P::kNone, P::kMean, false) == P::kNone);

    // Unspecified: the model's declared pooling is used (Qwen3-Embedding is "last", BERT is "cls"/"mean").
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kLast, false) == P::kLast);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kCls, true) == P::kCls);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kMean, false) == P::kMean);

    // Model declares rank pooling: preserved so dense embeddings fail closed for both encoder and decoder rerankers.
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kRank, false) == P::kRank);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kRank, true) == P::kRank);

    // Model declares nothing usable: decoder-only -> mean, encoder -> leave to llama.cpp.
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, std::nullopt, false) == P::kMean);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kNone, false) == P::kMean);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, std::nullopt, true) == P::kUnspecified);
    CHECK(lc::ResolveEmbedPooling(P::kUnspecified, P::kNone, true) == P::kUnspecified);
}
