// Extended harness tests: fillers/fan-out corpus, per-prompt summaries,
// fan-out batches, budget, markdown/file output. Mock backend only (harness
// behavior, not performance).
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "sonder/inference/benchmark.hpp"
#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {
std::string slurp(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
}  // namespace

TEST_SUITE("benchmark_ext") {
TEST_CASE("fillers, shared prefix and fan-out children") {
    auto c = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"t","version":"1",
        "fillers":{"f":"XY"},
        "prompts":[{"id":"long","workload":"long_context","context":{"filler":"f","repeat":2},"prompt":"Q"},
                   {"id":"fan","workload":"agent_fanout","shared_prefix":"S","context":{"filler":"f","repeat":1},
                    "children":["a","b"]}]})");
    REQUIRE_MESSAGE(c.ok(), c.status().to_string());
    CHECK(c->version == "1");
    CHECK(c->fnv1a.size() == 16u);
    CHECK(c->prompts[0].prompt == "XY\n\nXY\n\nQ");
    CHECK_FALSE(c->prompts[0].is_fanout());
    REQUIRE(c->prompts[1].is_fanout());
    CHECK(c->prompts[1].children.size() == 2u);
    CHECK(c->prompts[1].children[0] == "S\n\nXY\n\na");
    CHECK(c->prompts[1].children[1] == "S\n\nXY\n\nb");
    CHECK(c->prompts[1].shared_prefix_bytes == std::string("S\n\nXY").size());

    CHECK_FALSE(bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[
        {"id":"x","prompt":"p","context":{"filler":"missing"}}]})").ok());
    CHECK_FALSE(bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[
        {"id":"x","children":["ok",""]}]})").ok());
}

TEST_CASE("repository baseline corpus covers interactive, long-context and fan-out") {
    auto c = bench::load_corpus(SONDER_TEST_BASELINE_CORPUS);
    REQUIRE_MESSAGE(c.ok(), c.status().to_string());
    bool interactive = false, long_ctx = false, fanout = false;
    for (const auto& p : c->prompts) {
        interactive = interactive || p.workload == "interactive_chat";
        if (p.workload == "long_context") {
            long_ctx = true;
            CHECK(p.prompt.size() > 8000u);
        }
        if (p.is_fanout()) {
            fanout = true;
            CHECK(p.children.size() >= 4u);
            for (const auto& ch : p.children) {
                CHECK(ch.compare(0, p.shared_prefix_bytes, p.children[0], 0, p.shared_prefix_bytes) == 0);
            }
        }
    }
    CHECK(interactive);
    CHECK(long_ctx);
    CHECK(fanout);
}

TEST_CASE("fan-out runs concurrently and summaries are per prompt") {
    MockBackendOptions mo;
    // Large per-token delay so child latency dominates thread/session
    // overhead even under sanitizers or slow CI runners.
    mo.token_delay = std::chrono::milliseconds(20);
    sonder_test::Harness h(mo, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"u","prompts":[
        {"id":"one","prompt":"hello","max_tokens":6},
        {"id":"fan","workload":"agent_fanout","shared_prefix":"ctx","children":["a","b","c"],"max_tokens":6}]})");
    REQUIRE(corpus.ok());
    bench::Options o;
    o.warmup_runs = 1;
    o.measured_runs = 2;
    o.hardware = "unit";
    std::ostringstream progress;
    const json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o, &progress);
    CHECK(doc.find("runs")->as_array().size() == 3u * 1u + 3u * 3u);  // (1 warmup + 2) x (1 + 3 children)
    CHECK(doc.find("summary")->find("measured_requests")->as_int() == 2 + 2 * 3);
    CHECK(doc.find("summary")->find("failures")->as_int() == 0);
    CHECK(doc.find("host")->find("hardware")->as_string() == "unit");
    CHECK_FALSE(doc.find("truncated_by_budget")->as_bool());
    CHECK(doc.find("cold_first_request_ms")->as_double() > 0.0);

    const auto& pp = doc.find("per_prompt")->as_array();
    REQUIRE(pp.size() == 2u);
    CHECK(pp[0].find("prompt_id")->as_string() == "one");
    CHECK(pp[0].find("measured_requests")->as_int() == 2);
    CHECK(pp[0].find("ttft_ms")->find("n")->as_int() == 2);
    CHECK(pp[0].find("fanout_makespan_ms") == nullptr);
    CHECK(pp[1].find("measured_requests")->as_int() == 6);
    CHECK(pp[1].find("fanout_makespan_ms")->find("n")->as_int() == 2);
    CHECK(pp[1].find("requests_per_iteration")->as_int() == 3);

    const auto& batches = doc.find("fanout_batches")->as_array();
    REQUIRE(batches.size() == 3u);  // 1 warmup + 2 measured
    CHECK(batches[0].find("warmup")->as_bool());
    CHECK(batches[1].find("children")->as_int() == 3);
    // Concurrent children: makespan well below the sum of child latencies.
    double child_sum = 0;
    double child_max = 0;
    for (const auto& r : doc.find("runs")->as_array()) {
        if (r.find("child") && r.find("iteration")->as_int() == 1) {
            child_sum += r.find("total_ms")->as_double();
            child_max = std::max(child_max, r.find("total_ms")->as_double());
        }
    }
    const double makespan = batches[1].find("makespan_ms")->as_double();
    CHECK(makespan >= child_max);
    CHECK(makespan < 0.8 * child_sum);
    CHECK(progress.str().find("fan-out x3") != std::string::npos);
}

TEST_CASE("prompt filter and budget truncation") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[
        {"id":"a","prompt":"x","max_tokens":4},{"id":"b","prompt":"y","max_tokens":4}]})");
    REQUIRE(corpus.ok());
    bench::Options o;
    o.warmup_runs = 0;
    o.measured_runs = 1;
    o.prompt_ids = {"b"};
    json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o);
    REQUIRE(doc.find("per_prompt")->as_array().size() == 1u);
    CHECK(doc.find("per_prompt")->as_array()[0].find("prompt_id")->as_string() == "b");

    o.prompt_ids.clear();
    o.budget_seconds = 1e-9;
    doc = bench::run(*h.engine, h.model, corpus.value(), o);
    CHECK(doc.find("truncated_by_budget")->as_bool());
    CHECK(doc.find("runs")->as_array().empty());
}

TEST_CASE("failures are recorded, warmup failures excluded") {
    MockBackendOptions mo;
    mo.fail_after_tokens = 1;
    sonder_test::Harness h(mo, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[{"id":"a","prompt":"x"}]})");
    REQUIRE(corpus.ok());
    bench::Options o;
    o.warmup_runs = 1;
    o.measured_runs = 2;
    const json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o);
    CHECK(doc.find("summary")->find("failures")->as_int() == 2);
    CHECK(doc.find("runs")->as_array().size() == 3u);
    CHECK(doc.find("runs")->as_array()[0].find("error") != nullptr);
}

TEST_CASE("markdown summary and result files") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"md","version":"3","prompts":[
        {"id":"one","workload":"interactive_chat","prompt":"hello","max_tokens":5},
        {"id":"fan","workload":"agent_fanout","children":["a","b"],"max_tokens":5}]})");
    REQUIRE(corpus.ok());
    bench::Options o;
    o.warmup_runs = 0;
    o.measured_runs = 1;
    o.hardware = "Test GPU 8GB";
    const json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o);
    const std::string md = bench::render_markdown(doc);
    CHECK(md.find("MOCK BACKEND") != std::string::npos);
    CHECK(md.find("Test GPU 8GB") != std::string::npos);
    CHECK(md.find("| one | interactive_chat | 1 | 0 |") != std::string::npos);
    CHECK(md.find("Agent fan-out") != std::string::npos);
    CHECK(md.find("md v3") != std::string::npos);

    const std::string stem = bench::default_result_stem(doc);
    CHECK(stem.find("-mock-mock_tiny-md") != std::string::npos);
    const auto dir = std::filesystem::temp_directory_path() / "sonder_bench_ext_test";
    std::filesystem::remove_all(dir);
    std::string jp, mp;
    REQUIRE(bench::write_results(doc, dir.string(), stem, &jp, &mp).ok());
    auto parsed = json::parse(slurp(jp));
    REQUIRE(parsed.ok());
    CHECK(parsed->find("schema")->as_string() == "sonder.inference.bench/1");
    CHECK(slurp(mp) == md);
    std::filesystem::remove_all(dir);
}

// Regression tests for docs/integration/hardening.md B2, B3 and markdown
// escaping. Repro inputs also live in fuzz/corpus/bench_corpus.

TEST_CASE("max_tokens is range-checked before narrowing (B2)") {
    const auto corpus_with = [](const std::string& max_tokens) {
        return bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","prompts":[{"id":"p","prompt":"x","max_tokens":)" +
                                   max_tokens + "}]}");
    };
    CHECK_FALSE(corpus_with("4294967297").ok());  // used to wrap to 1
    CHECK_FALSE(corpus_with("4294967360").ok());  // used to wrap to 64
    CHECK_FALSE(corpus_with("-4294967295").ok());
    CHECK_FALSE(corpus_with("0").ok());
    auto ok = corpus_with("64");
    REQUIRE(ok.ok());
    CHECK(ok.value().prompts[0].max_tokens == 64);
}

TEST_CASE("corpus expansion is capped (B3)") {
    const std::string filler(1000, 'x');
    std::string kids;
    for (int i = 0; i < 20; ++i) kids += std::string(i ? "," : "") + "\"k\"";
    // 1.2 KB of JSON that used to expand to ~210 MB of prompt text.
    const std::string bomb = R"({"schema":"sonder.inference.corpus/1","fillers":{"f":")" + filler +
                             R"("},"prompts":[{"id":"p","context":{"filler":"f","repeat":10000},"children":[)" + kids +
                             "]}]}";
    auto r = bench::parse_corpus(bomb);
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(r.status().message().find("size limit") != std::string::npos);

    // A single 10 MB prefix is fine; copying it into two children is not.
    const std::string single = R"({"schema":"sonder.inference.corpus/1","fillers":{"f":")" + filler +
                               R"("},"prompts":[{"id":"p","prompt":"q","context":{"filler":"f","repeat":10000}}]})";
    CHECK(bench::parse_corpus(single).ok());
    const std::string two = R"({"schema":"sonder.inference.corpus/1","fillers":{"f":")" + filler +
                            R"("},"prompts":[{"id":"p","context":{"filler":"f","repeat":10000},"children":["a","b"]}]})";
    CHECK_FALSE(bench::parse_corpus(two).ok());

    // Total across prompts is capped too (7 x ~10 MB > 64 MiB).
    std::string many = R"({"schema":"sonder.inference.corpus/1","fillers":{"f":")" + filler + R"("},"prompts":[)";
    for (int i = 0; i < 7; ++i) {
        many += std::string(i ? "," : "") + R"({"id":"p)" + std::to_string(i) +
                R"(","prompt":"q","context":{"filler":"f","repeat":10000}})";
    }
    many += "]}";
    CHECK_FALSE(bench::parse_corpus(many).ok());
}

TEST_CASE("render_markdown escapes table-breaking characters") {
    auto doc = json::parse(
        R"({"backend":{"name":"mock"},"model":{"name":"evil|model\nnext"},"label":"a\\b|c",)"
        R"("summary":{"measured_requests":1,"ttft_ms":{"n":1,"p50":1e300,"p95":2.0}}})");
    REQUIRE(doc.ok());
    const std::string md = bench::render_markdown(doc.value());
    CHECK(md.find("evil\\|model next") != std::string::npos);
    CHECK(md.find("| Label | a\\\\b\\|c |") != std::string::npos);
    CHECK(md.find("evil|model") == std::string::npos);
    CHECK(md.find("1.0e+300") != std::string::npos);  // not a truncated 301-digit number
    // The file stem still uses the raw value, made filesystem-safe.
    CHECK(bench::default_result_stem(doc.value()) == "undated-mock-evil_model_next--");
}
}
