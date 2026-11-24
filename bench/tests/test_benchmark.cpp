#include <doctest/doctest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

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

TEST_CASE("an oversized corpus file is refused before it is read") {
    const std::string path = "bench-oversized-corpus.json";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        // Valid JSON of limit + 1 bytes: '{' then spaces then '}'.
        std::string chunk(1u << 20, ' ');
        out.put('{');
        std::size_t left = bench::kMaxCorpusFileBytes - 1;
        while (left > 0) {
            const std::size_t n = std::min(left, chunk.size());
            out.write(chunk.data(), static_cast<std::streamsize>(n));
            left -= n;
        }
        out.put('}');
    }
    auto c = bench::load_corpus(path);
    std::remove(path.c_str());
    REQUIRE_FALSE(c.ok());
    CHECK(c.status().code() == ErrorCode::invalid_argument);
    CHECK(c.status().message().find("too large") != std::string::npos);
    CHECK_FALSE(bench::parse_corpus(std::string(bench::kMaxCorpusFileBytes + 1, ' ')).ok());
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

TEST_CASE("default concurrency 1 emits additive summary throughput and config fields") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    bench::Corpus corpus;
    corpus.name = "unit-c1";
    corpus.prompts = {{"p1", "interactive_chat", "hello there", 5}, {"p2", "coding", "write a loop", 7}};
    bench::Options o;
    o.warmup_runs = 1;
    o.measured_runs = 2;
    const auto doc = json::Value(bench::run(*h.engine, h.model, corpus, o));
    REQUIRE(doc.find("config") != nullptr);
    CHECK(doc.find("config")->find("concurrency")->as_int() == 1);
    const auto* summary = doc.find("summary");
    REQUIRE(summary != nullptr);
    CHECK(summary->find("concurrency")->as_int() == 1);
    CHECK(summary->find("wall_seconds")->as_double() > 0.0);
    CHECK(summary->find("total_completion_tokens")->as_int() == 2 * (5 + 7));
    CHECK(summary->find("aggregate_decode_tokens_per_sec")->as_double() > 0.0);
    for (const auto& r : doc.find("runs")->as_array()) {
        CHECK(r.find("worker_index") == nullptr);
    }

    // Direct C++ Options::concurrency <= 0 clamps safely to 1.
    bench::Options clamped;
    clamped.warmup_runs = 0;
    clamped.measured_runs = 1;
    clamped.concurrency = 0;
    clamped.capture_gpu_snapshot = false;
    const auto clamped_doc = json::Value(bench::run(*h.engine, h.model, corpus, clamped));
    CHECK(clamped_doc.find("config")->find("concurrency")->as_int() == 1);
    CHECK(clamped_doc.find("summary")->find("measured_requests")->as_int() == 2);
}

TEST_CASE("gpu snapshot arrays are well-formed and omitted when disabled") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    bench::Corpus corpus;
    corpus.name = "unit-gpu";
    corpus.prompts = {{"p1", "interactive_chat", "hello", 4}};

    bench::Options enabled;
    enabled.warmup_runs = 0;
    enabled.measured_runs = 1;
    enabled.capture_gpu_snapshot = true;
    const auto doc = json::Value(bench::run(*h.engine, h.model, corpus, enabled));
    CHECK(doc.find("config")->find("capture_gpu_snapshot")->as_bool());
    const auto* before = doc.find("gpus_before");
    const auto* after = doc.find("gpus_after");
    REQUIRE(before != nullptr);
    REQUIRE(after != nullptr);
    CHECK(before->is_array());
    CHECK(after->is_array());
    const auto check_gpu_entry = [](const json::Value& g, bool allow_delta) {
        REQUIRE(g.is_object());
        for (const char* str_key : {"id", "name", "vendor", "backend_hint", "source"}) {
            CAPTURE(str_key);
            const auto* f = g.find(str_key);
            REQUIRE(f != nullptr);
            CHECK(f->is_string());
        }
        for (const char* mib_key :
             {"dedicated_total_mib", "dedicated_free_mib", "effective_vram_mib", "effective_free_vram_mib"}) {
            CAPTURE(mib_key);
            const auto* f = g.find(mib_key);
            REQUIRE(f != nullptr);
            CHECK((f->is_null() || f->is_integer()));
        }
        if (g.find("dedicated_free_mib")->is_null()) {
            CHECK(g.find("effective_free_vram_mib")->is_null());
        }
        const auto* uma = g.find("is_integrated_uma");
        REQUIRE(uma != nullptr);
        CHECK(uma->is_bool());
        if (const auto* delta = g.find("peak_vram_delta_mib")) {
            CHECK(allow_delta);
            CHECK(delta->is_integer());
            CHECK(delta->as_int() > 0);
        }
    };
    for (const auto& g : before->as_array()) check_gpu_entry(g, false);
    for (const auto& g : after->as_array()) check_gpu_entry(g, true);

    bench::Options disabled = enabled;
    disabled.capture_gpu_snapshot = false;
    const auto doc_no_gpu = json::Value(bench::run(*h.engine, h.model, corpus, disabled));
    CHECK_FALSE(doc_no_gpu.find("config")->find("capture_gpu_snapshot")->as_bool(true));
    CHECK(doc_no_gpu.find("gpus_before") == nullptr);
    CHECK(doc_no_gpu.find("gpus_after") == nullptr);
}
}
