// Extended harness tests: fillers/fan-out corpus, per-prompt summaries,
// fan-out batches, budget, markdown/file output. Mock backend only (harness
// behavior, not performance).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

TEST_CASE("concurrency 4 executes parallel sessions and computes aggregate throughput") {
    MockBackendOptions mo;
    mo.token_delay = std::chrono::milliseconds(20);
    sonder_test::Harness h(mo, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"conc","prompts":[
        {"id":"p1","workload":"interactive_chat","prompt":"hello world","max_tokens":6},
        {"id":"p2","workload":"coding","prompt":"write code","max_tokens":6}]})");
    REQUIRE_MESSAGE(corpus.ok(), corpus.status().to_string());

    bench::Options o;
    o.warmup_runs = 0;
    o.measured_runs = 2;
    o.concurrency = 4;
    o.capture_gpu_snapshot = false;
    std::ostringstream progress;
    const json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o, &progress);

    REQUIRE(doc.find("config") != nullptr);
    CHECK(doc.find("config")->find("concurrency")->as_int() == 4);
    const auto* summary = doc.find("summary");
    REQUIRE(summary != nullptr);
    CHECK(summary->find("concurrency")->as_int() == 4);
    CHECK(summary->find("measured_requests")->as_int() == 16);  // 2 prompts * 2 runs * 4 workers
    CHECK(summary->find("failures")->as_int() == 0);
    CHECK(summary->find("total_completion_tokens")->as_int() == 16 * 6);
    const double measured_wall_s = summary->find("wall_seconds")->as_double();
    const double agg_tps = summary->find("aggregate_decode_tokens_per_sec")->as_double();
    CHECK(measured_wall_s > 0.0);
    CHECK(agg_tps > 0.0);

    const auto& rows = doc.find("runs")->as_array();
    REQUIRE(rows.size() == 16u);  // 4 * measured_runs per prompt
    double total_ms_sum = 0.0;
    std::map<std::pair<std::string, std::int64_t>, std::set<std::int64_t>> seen_workers;
    for (const auto& row : rows) {
        CHECK(row.find("outcome")->as_string() == "completed");
        CHECK_FALSE(row.find("warmup")->as_bool(true));
        REQUIRE(row.find("worker_index") != nullptr);
        const std::int64_t widx = row.find("worker_index")->as_int(-1);
        CHECK((widx >= 0 && widx < 4));
        seen_workers[{row.find("prompt_id")->as_string(), row.find("iteration")->as_int()}].insert(widx);
        total_ms_sum += row.find("total_ms")->as_double();
    }
    CHECK(seen_workers.size() == 4u);  // 2 prompts * 2 measured iterations
    for (const auto& [key, workers] : seen_workers) {
        CAPTURE(key.first);
        CAPTURE(key.second);
        CHECK(workers == std::set<std::int64_t>{0, 1, 2, 3});
    }
    // 4 concurrent workers complete well below the sum of individual latencies.
    CHECK(measured_wall_s * 1000.0 < 0.8 * total_ms_sum);

    const auto check_dist = [](const json::Value* dist, std::int64_t expected_n) {
        REQUIRE(dist != nullptr);
        CHECK(dist->find("n")->as_int() == expected_n);
        const double dmin = dist->find("min")->as_double();
        const double dp50 = dist->find("p50")->as_double();
        const double dp95 = dist->find("p95")->as_double();
        const double dp99 = dist->find("p99")->as_double();
        const double dmax = dist->find("max")->as_double();
        CHECK(dp50 > 0.0);
        CHECK(dmin <= dp50);
        CHECK(dp50 <= dp95);
        CHECK(dp95 <= dp99);
        CHECK(dp99 <= dmax);
    };
    check_dist(summary->find("ttft_ms"), 16);
    check_dist(summary->find("total_ms"), 16);
    check_dist(summary->find("decode_tokens_per_sec"), 16);
    check_dist(summary->find("client_decode_tokens_per_sec"), 16);
    CHECK(agg_tps > summary->find("client_decode_tokens_per_sec")->find("p50")->as_double());

    const auto& pp = doc.find("per_prompt")->as_array();
    REQUIRE(pp.size() == 2u);
    for (const auto& p : pp) {
        CHECK(p.find("requests_per_iteration")->as_int() == 4);
        CHECK(p.find("measured_requests")->as_int() == 8);
        CHECK(p.find("wall_seconds")->as_double() > 0.0);
        CHECK(p.find("aggregate_decode_tokens_per_sec")->as_double() > 0.0);
    }
    CHECK(progress.str().find("concurrency x4") != std::string::npos);

    const std::string md = bench::render_markdown(doc);
    CHECK(md.find("| Concurrency | 4 |") != std::string::npos);
    CHECK(md.find("Aggregate decode tok/s") != std::string::npos);
    CHECK(md.find("MOCK BACKEND") != std::string::npos);
    CHECK(md.find("## GPUs") == std::string::npos);

    // Sequential warmup + concurrent measured runs emit 1 warmup + 4*M measured rows.
    sonder_test::Harness fast_h({}, TelemetryLevel::metrics);
    bench::Corpus single;
    single.name = "warm-conc";
    single.prompts = {{"p1", "interactive_chat", "hi", 4}};
    bench::Options warm_o;
    warm_o.warmup_runs = 1;
    warm_o.measured_runs = 2;
    warm_o.concurrency = 4;
    warm_o.capture_gpu_snapshot = false;
    const json::Value warm_doc = bench::run(*fast_h.engine, fast_h.model, single, warm_o);
    REQUIRE(warm_doc.find("runs")->as_array().size() == 1u + 4u * 2u);
    CHECK(warm_doc.find("runs")->as_array()[0].find("warmup")->as_bool());
    CHECK(warm_doc.find("runs")->as_array()[0].find("worker_index") == nullptr);
    CHECK(warm_doc.find("summary")->find("measured_requests")->as_int() == 8);

    // Worker failures under concurrency never deadlock and record all failures.
    MockBackendOptions fail_mo;
    fail_mo.fail_after_tokens = 1;
    sonder_test::Harness fail_h(fail_mo, TelemetryLevel::metrics);
    bench::Options fail_o;
    fail_o.warmup_runs = 0;
    fail_o.measured_runs = 1;
    fail_o.concurrency = 4;
    fail_o.capture_gpu_snapshot = false;
    const json::Value fail_doc = bench::run(*fail_h.engine, fail_h.model, single, fail_o);
    CHECK(fail_doc.find("summary")->find("failures")->as_int() == 4);
    CHECK(fail_doc.find("summary")->find("aggregate_decode_tokens_per_sec")->as_double() == doctest::Approx(0.0));
}

TEST_CASE("gpu snapshot delta calculation and markdown GPUs table") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;

    GpuDeviceInfo d_before;
    d_before.id = "gpu:0";
    d_before.name = "RTX Test | 16GB";
    d_before.vendor = GpuVendor::nvidia;
    d_before.dedicated_total = MemoryReading::known(16384ull * kMiB);
    d_before.dedicated_free = MemoryReading::known(14000ull * kMiB);
    d_before.backend_hint = "cuda";
    d_before.source = "dxgi";

    GpuDeviceInfo uma_before;
    uma_before.id = "gpu:1";
    uma_before.name = "Radeon iGPU";
    uma_before.vendor = GpuVendor::amd;
    uma_before.is_integrated_uma = true;
    uma_before.dedicated_total = MemoryReading::known(512ull * kMiB);
    uma_before.shared_total = MemoryReading::known(15360ull * kMiB);
    uma_before.dedicated_free = MemoryReading::known(12000ull * kMiB);
    uma_before.backend_hint = "rocm";
    uma_before.source = "amdgpu-sysfs";

    GpuDeviceInfo unk_before;
    unk_before.id = "gpu:2";
    unk_before.name = "Intel Arc";
    unk_before.vendor = GpuVendor::intel;
    unk_before.dedicated_total = MemoryReading::known(8192ull * kMiB);
    unk_before.dedicated_free = MemoryReading::unknown();
    unk_before.backend_hint = "vulkan";
    unk_before.source = "dxgi";

    GpuDeviceInfo d_after = d_before;
    d_after.dedicated_free = MemoryReading::known(11500ull * kMiB);  // dropped 2500 MiB

    GpuDeviceInfo uma_after = uma_before;
    uma_after.dedicated_free = MemoryReading::known(12500ull * kMiB);  // increased; no delta

    GpuDeviceInfo unk_after = unk_before;  // still unknown free VRAM

    const std::vector<GpuDeviceInfo> before_vec{d_before, uma_before, unk_before};
    const std::vector<GpuDeviceInfo> after_vec{d_after, uma_after, unk_after};
    json::Array before_json = bench::gpu_snapshot_to_json(before_vec, nullptr);
    json::Array after_json = bench::gpu_snapshot_to_json(after_vec, &before_vec);
    REQUIRE(before_json.size() == 3u);
    REQUIRE(after_json.size() == 3u);

    CHECK(before_json[0].find("peak_vram_delta_mib") == nullptr);
    REQUIRE(after_json[0].find("peak_vram_delta_mib") != nullptr);
    CHECK(after_json[0].find("peak_vram_delta_mib")->as_int() == 2500);
    CHECK(after_json[0].find("dedicated_total_mib")->as_int() == 16384);
    CHECK(after_json[0].find("dedicated_free_mib")->as_int() == 11500);
    CHECK(after_json[0].find("effective_vram_mib")->as_int() == 16384);
    CHECK(after_json[0].find("effective_free_vram_mib")->as_int() == 11500);
    CHECK_FALSE(after_json[0].find("is_integrated_uma")->as_bool(true));

    CHECK(after_json[1].find("peak_vram_delta_mib") == nullptr);
    CHECK(after_json[1].find("is_integrated_uma")->as_bool());
    CHECK(after_json[1].find("effective_vram_mib")->as_int() == 512 + 15360);

    CHECK(before_json[2].find("dedicated_free_mib")->is_null());
    CHECK(before_json[2].find("effective_free_vram_mib")->is_null());
    CHECK(after_json[2].find("dedicated_free_mib")->is_null());
    CHECK(after_json[2].find("effective_free_vram_mib")->is_null());
    CHECK(after_json[2].find("effective_vram_mib")->as_int() == 8192);
    CHECK(after_json[2].find("peak_vram_delta_mib") == nullptr);

    json::Object doc;
    doc.set("backend", json::Object{{"name", "mock"}});
    doc.set("model", json::Object{{"name", "mock:tiny"}});
    doc.set("config", json::Object{{"concurrency", 2}});
    doc.set("summary", json::Object{{"measured_requests", 4}, {"aggregate_decode_tokens_per_sec", 123.4}});
    doc.set("gpus_before", std::move(before_json));
    doc.set("gpus_after", std::move(after_json));

    const std::string md = bench::render_markdown(json::Value(doc));
    CHECK(md.find("## GPUs") != std::string::npos);
    CHECK(md.find("RTX Test \\| 16GB") != std::string::npos);
    CHECK(md.find("RTX Test | 16GB") == std::string::npos);
    CHECK(md.find("| gpu:0 | RTX Test \\| 16GB | nvidia | 16384 | 14000 | 11500 | 2500 | cuda | dxgi |") !=
          std::string::npos);
    CHECK(md.find("| gpu:2 | Intel Arc | intel | 8192 | - | - | - | vulkan | dxgi |") != std::string::npos);

    // Empty GPU arrays omit the ## GPUs section cleanly.
    json::Object empty_doc;
    empty_doc.set("backend", json::Object{{"name", "mock"}});
    empty_doc.set("model", json::Object{{"name", "mock:tiny"}});
    empty_doc.set("gpus_before", json::Array{});
    empty_doc.set("gpus_after", json::Array{});
    CHECK(bench::render_markdown(json::Value(empty_doc)).find("## GPUs") == std::string::npos);
}

TEST_CASE("sonder-bench CLI option validation for concurrency and gpu snapshot") {
    bench::BenchCliArgs parsed;
    std::string error;

    REQUIRE_MESSAGE(bench::parse_bench_cli_args({"--backend", "mock", "--model", "mock:tiny"}, parsed, error), error);
    CHECK_FALSE(parsed.help);
    CHECK(parsed.options.concurrency == 1);
    CHECK(parsed.options.capture_gpu_snapshot);

    REQUIRE_MESSAGE(
        bench::parse_bench_cli_args(
            {"--backend", "mock", "--model", "mock:tiny", "--concurrency", "4", "--no-gpu-snapshot"}, parsed, error),
        error);
    CHECK(parsed.options.concurrency == 4);
    CHECK_FALSE(parsed.options.capture_gpu_snapshot);

    REQUIRE_MESSAGE(bench::parse_bench_cli_args({"--model", "m", "--concurrency", "1"}, parsed, error), error);
    CHECK(parsed.options.concurrency == 1);
    REQUIRE_MESSAGE(bench::parse_bench_cli_args({"--model", "m", "--concurrency=64"}, parsed, error), error);
    CHECK(parsed.options.concurrency == 64);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "mock", "--model", "mock:tiny", "--concurrency", "0"}, parsed,
                                            error));
    CHECK(error.find("--concurrency") != std::string::npos);
    CHECK(error.find("out of range") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "mock", "--model", "mock:tiny", "--concurrency", "65"},
                                            parsed, error));
    CHECK(error.find("--concurrency") != std::string::npos);
    CHECK(error.find("out of range") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--model", "m", "--concurrency", "-1"}, parsed, error));
    CHECK(error.find("--concurrency") != std::string::npos);
    CHECK_FALSE(bench::parse_bench_cli_args({"--model", "m", "--concurrency", "abc"}, parsed, error));
    CHECK(error.find("--concurrency") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--model", "m", "--no-gpu-snapshot=1"}, parsed, error));
    CHECK(error.find("--no-gpu-snapshot") != std::string::npos);

    const std::string help = bench::render_bench_cli_help();
    CHECK(help.find("--concurrency N") != std::string::npos);
    CHECK(help.find("--no-gpu-snapshot") != std::string::npos);
    CHECK(help.find("MOCK BACKEND banner") != std::string::npos);
    CHECK(help.find("Exit status: 0 ok") != std::string::npos);
}

TEST_CASE("single-child fan-out records child index 0") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"fan1","prompts":[
        {"id":"fan-single","workload":"agent_fanout","shared_prefix":"ctx","children":["only-task"],"max_tokens":4}]})");
    REQUIRE_MESSAGE(corpus.ok(), corpus.status().to_string());

    bench::Options o;
    o.warmup_runs = 0;
    o.measured_runs = 1;
    o.concurrency = 4;
    o.capture_gpu_snapshot = false;
    const json::Value doc = bench::run(*h.engine, h.model, corpus.value(), o);

    const auto& rows = doc.find("runs")->as_array();
    REQUIRE(rows.size() == 1u);
    REQUIRE(rows[0].find("child") != nullptr);
    CHECK(rows[0].find("child")->as_int() == 0);
    CHECK(rows[0].find("worker_index") == nullptr);
    CHECK(rows[0].find("outcome")->as_string() == "completed");

    const auto& batches = doc.find("fanout_batches")->as_array();
    REQUIRE(batches.size() == 1u);
    CHECK(batches[0].find("children")->as_int() == 1);
}
}
