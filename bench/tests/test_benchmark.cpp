#include <doctest/doctest.h>

#include "sonder/inference/benchmark.hpp"
#include "test_helpers.hpp"

using namespace sonder::inference;

TEST_SUITE("benchmark") {
TEST_CASE("percentiles interpolate") {
    CHECK(bench::percentile({}, 50) == 0.0);
    CHECK(bench::percentile({5.0}, 95) == 5.0);
    CHECK(bench::percentile({1, 2, 3, 4}, 50) == doctest::Approx(2.5));
    CHECK(bench::percentile({4, 1, 3, 2}, 100) == doctest::Approx(4.0));
}

TEST_CASE("corpus validation") {
    CHECK_FALSE(bench::parse_corpus("{}").ok());
    CHECK_FALSE(bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[]})").ok());
    CHECK_FALSE(bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[{"id":"a"}]})").ok());
    auto ok = bench::parse_corpus(
        R"({"schema":"sonder.inference.corpus/1","name":"t","prompts":[{"id":"a","prompt":"hi","max_tokens":4}]})");
    REQUIRE(ok.ok());
    CHECK(ok.value().prompts.at(0).max_tokens == 4);
}

TEST_CASE("repository smoke corpus loads") {
    auto c = bench::load_corpus(SONDER_TEST_CORPUS);
    REQUIRE_MESSAGE(c.ok(), c.status().to_string());
    CHECK(c.value().prompts.size() >= 3);
}

TEST_CASE("runs a corpus against the mock backend") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    bench::Corpus corpus;
    corpus.name = "unit";
    corpus.prompts = {{"p1", "interactive_chat", "hello there", 5}, {"p2", "coding", "write a loop", 7}};
    bench::Options o;
    o.warmup_runs = 1;
    o.measured_runs = 2;
    o.label = "unit-test";
    const auto doc = json::Value(bench::run(*h.engine, h.model, corpus, o));
    CHECK(doc.find("schema")->as_string() == "sonder.inference.bench/1");
    CHECK(doc.find("backend")->find("name")->as_string() == "mock");
    CHECK(doc.find("model")->find("name")->as_string() == "mock:tiny");
    CHECK(doc.find("runs")->as_array().size() == 6);
    const auto* summary = doc.find("summary");
    CHECK(summary->find("measured_requests")->as_int() == 4);
    CHECK(summary->find("failures")->as_int() == 0);
    CHECK(summary->find("total_ms")->find("n")->as_int() == 4);
    for (const auto& run : doc.find("runs")->as_array()) {
        CHECK(run.find("outcome")->as_string() == "completed");
        CHECK(run.find("stop_reason")->as_string() == "max_tokens");
    }
    // Round-trips through the serializer.
    CHECK(json::parse(doc.dump()).ok());
}
}
