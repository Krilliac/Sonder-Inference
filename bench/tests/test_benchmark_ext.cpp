// Extended harness tests: fillers/fan-out corpus, per-prompt summaries,
// fan-out batches, budget, markdown/file output. Mock backend only (harness
// behavior, not performance).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cli/cli_args.hpp"
#include "cli/sonder_infer_commands.hpp"
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

struct SpecScriptedReply {
    std::string text = "ok";
    std::uint64_t prompt_tokens = 32;
    std::uint64_t completion_tokens = 8;
    std::uint64_t prompt_eval_ns = 10'000'000;
    std::uint64_t eval_ns = 20'000'000;
    // When true, populate only GenerateStats::backend_timings so tests verify
    // raw BackendTimings extraction even when top-level optional fields are unset.
    bool timings_only = false;
    bool fail_request = false;
    bool cancel_request = false;
    std::optional<std::uint64_t> draft_n;
    std::optional<std::uint64_t> draft_n_accepted;
    std::optional<std::uint64_t> cache_n;
};

class SpecStubModel final : public BackendModel {
public:
    SpecStubModel(ModelDescriptor desc, std::vector<SpecScriptedReply> replies,
                  std::shared_ptr<std::vector<SamplingConfig>> seen_sampling)
        : descriptor_(std::move(desc)), replies_(std::move(replies)), seen_sampling_(std::move(seen_sampling)) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled");
        }
        SpecScriptedReply reply;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (seen_sampling_) {
                seen_sampling_->push_back(request.sampling);
            }
            if (index_ < replies_.size()) {
                reply = replies_[index_++];
            } else if (!replies_.empty()) {
                reply = replies_.back();
            }
        }
        if (reply.fail_request) {
            return Status(ErrorCode::backend_error, "injected backend failure");
        }
        if (reply.cancel_request) {
            return Status(ErrorCode::cancelled, "injected cancellation");
        }
        GenerateStats stats;
        stats.prompt_tokens = reply.prompt_tokens;
        stats.completion_tokens = reply.completion_tokens;
        stats.prompt_eval_ns = reply.prompt_eval_ns;
        stats.eval_ns = reply.eval_ns;
        stats.token_counts_from_backend = true;
        stats.stop_reason = StopReason::max_tokens;
        if (reply.draft_n.has_value() || reply.draft_n_accepted.has_value() || reply.cache_n.has_value()) {
            BackendTimings bt;
            bt.prompt_n = reply.prompt_tokens;
            bt.predicted_n = reply.completion_tokens;
            bt.draft_n = reply.draft_n;
            bt.draft_n_accepted = reply.draft_n_accepted;
            bt.cache_n = reply.cache_n;
            stats.backend_timings = bt;
            if (!reply.timings_only) {
                stats.draft_tokens = reply.draft_n;
                stats.draft_accepted_tokens = reply.draft_n_accepted;
                stats.cached_tokens = reply.cache_n;
            }
        }
        if (on_chunk) {
            (void)on_chunk(TokenChunk{reply.text, 0});
        }
        stats.chunks = 1;
        return stats;
    }

private:
    ModelDescriptor descriptor_;
    std::vector<SpecScriptedReply> replies_;
    std::shared_ptr<std::vector<SamplingConfig>> seen_sampling_;
    std::mutex mu_;
    std::size_t index_ = 0;
};

class SpecStubBackend final : public Backend {
public:
    explicit SpecStubBackend(std::vector<SpecScriptedReply> replies,
                             std::shared_ptr<std::vector<SamplingConfig>> seen_sampling = nullptr)
        : replies_(std::move(replies)), seen_sampling_(std::move(seen_sampling)) {}

    std::string name() const override { return "llamaserver"; }
    std::string description() const override { return "Stub llamaserver with speculative timings"; }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming)
            .add(Capability::remote_process)
            .add(Capability::speculative_decode)
            .add(Capability::prefix_reuse);
        return c;
    }
    Result<std::string> probe() override { return std::string("stub-llamaserver/1"); }
    Result<std::vector<ModelDescriptor>> list_models() override {
        ModelDescriptor m;
        m.name = "bonsai-2-q1";
        m.backend = "llamaserver";
        m.format = "gguf";
        m.quantization = "Q1_0_G128";
        return std::vector<ModelDescriptor>{m};
    }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        ModelDescriptor m;
        m.name = options.model;
        m.backend = "llamaserver";
        m.format = "gguf";
        m.quantization = "Q1_0_G128";
        return std::shared_ptr<BackendModel>(std::make_shared<SpecStubModel>(std::move(m), replies_, seen_sampling_));
    }

private:
    std::vector<SpecScriptedReply> replies_;
    std::shared_ptr<std::vector<SamplingConfig>> seen_sampling_;
};
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
    // Unique per run: several worktrees run ctest at once, and a shared fixed
    // directory let one process remove_all() another's results mid-test.
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
                     ("sonder_bench_ext_test-" + std::to_string(unique));
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

TEST_CASE("sonder-bench defaults to llamacpp and accepts native options without weights") {
    bench::BenchCliArgs parsed;
    CHECK(parsed.backend == "llamacpp");
    std::string error;
    REQUIRE_MESSAGE(bench::parse_bench_cli_args({"--model", "models/test model.gguf"}, parsed, error), error);
    CHECK(parsed.backend == "llamacpp");
    CHECK(parsed.model == "models/test model.gguf");
    CHECK(parsed.context_length == 4096);
    CHECK(parsed.threads == 0);
    CHECK(parsed.gpu_layers == -1);

    REQUIRE_MESSAGE(bench::parse_bench_cli_args(
        {"--backend", "llamacpp", "--model", "model.gguf", "--ctx-size", "8192", "--threads", "4",
         "--gpu-layers", "0", "--warmup", "0", "--runs", "1", "--no-gpu-snapshot"}, parsed, error), error);
    CHECK(parsed.backend == "llamacpp");
    CHECK(parsed.context_length == 8192);
    CHECK(parsed.threads == 4);
    CHECK(parsed.gpu_layers == 0);
    CHECK(parsed.options.warmup_runs == 0);
    CHECK(parsed.options.measured_runs == 1);
    CHECK_FALSE(parsed.options.capture_gpu_snapshot);

    REQUIRE_MESSAGE(bench::parse_bench_cli_args(
        {"--model", "model.gguf", "--ctx-size", "0", "--threads", "0", "--gpu-layers", "-1"}, parsed, error), error);
    CHECK(parsed.context_length == 0);
    CHECK(parsed.threads == 0);
    CHECK(parsed.gpu_layers == -1);
    const auto help = bench::render_bench_cli_help();
    CHECK(help.find("default: llamacpp") != std::string::npos);
    CHECK(help.find("--ctx-size N") != std::string::npos);
    CHECK(help.find("--threads N") != std::string::npos);
    CHECK(help.find("--gpu-layers N") != std::string::npos);
}

TEST_CASE("sonder-bench rejects invalid native options and backend combinations") {
    bench::BenchCliArgs parsed;
    std::string error;
    for (const char* option : {"--ctx-size", "--threads", "--gpu-layers"}) {
        for (const char* bad : {"-2", "nope", "1.5", "2147483648", ""}) {
            CAPTURE(option);
            CAPTURE(bad);
            CHECK_FALSE(bench::parse_bench_cli_args({"--model", "model.gguf", option, bad}, parsed, error));
            CHECK(error.find(option) != std::string::npos);
        }
        for (const char* backend : {"mock", "ollama", "llamaserver"}) {
            CAPTURE(option);
            CAPTURE(backend);
            CHECK_FALSE(bench::parse_bench_cli_args({"--backend", backend, option, "1"}, parsed, error));
            CHECK(error.find("requires --backend llamacpp") != std::string::npos);
        }
    }
    for (const char* option : {"--ctx-size", "--threads"}) {
        CHECK_FALSE(bench::parse_bench_cli_args({option, "-1"}, parsed, error));
    }
    CHECK_FALSE(bench::parse_bench_cli_args({"--require-idle"}, parsed, error));
    CHECK(error.find("--backend ollama") != std::string::npos);
    CHECK_FALSE(bench::parse_bench_cli_args({"--ollama-url", "http://127.0.0.1:11434"}, parsed, error));
    CHECK(error.find("--backend ollama") != std::string::npos);
}

TEST_CASE("sonder-bench CLI options for llamaserver backend") {
    bench::BenchCliArgs parsed;
    std::string error;

    REQUIRE_MESSAGE(
        bench::parse_bench_cli_args(
            {"--backend", "llamaserver", "--llamaserver-url", "http://127.0.0.1:18080", "--model", "bonsai-2"}, parsed,
            error),
        error);
    CHECK(parsed.backend == "llamaserver");
    CHECK(parsed.llamaserver_url == "http://127.0.0.1:18080");
    CHECK(parsed.llamaserver_config.empty());
    CHECK(parsed.model == "bonsai-2");
    CHECK_FALSE(parsed.list_models);

    REQUIRE_MESSAGE(
        bench::parse_bench_cli_args(
            {"--backend", "llamaserver", "--llamaserver-config", "server.json", "--model", "bonsai-2"}, parsed, error),
        error);
    CHECK(parsed.backend == "llamaserver");
    CHECK(parsed.llamaserver_config == "server.json");
    CHECK(parsed.llamaserver_url.empty());

    // --llamaserver-url and --llamaserver-config are strictly mutually exclusive.
    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "llamaserver", "--llamaserver-config", "server.json",
                                             "--llamaserver-url", "http://127.0.0.1:18080", "--model", "bonsai-2"},
                                            parsed, error));
    CHECK(error.find("cannot be used together") != std::string::npos);

    REQUIRE_MESSAGE(
        bench::parse_bench_cli_args(
            {"--backend", "llamaserver", "--llamaserver-url", "http://127.0.0.1:18080", "--list-models"}, parsed,
            error),
        error);
    CHECK(parsed.backend == "llamaserver");
    CHECK(parsed.list_models);
    CHECK(parsed.model.empty());

    REQUIRE_MESSAGE(bench::parse_bench_cli_args({"--backend", "llamaserver", "--list-models"}, parsed, error), error);
    CHECK(parsed.backend == "llamaserver");
    CHECK(parsed.list_models);

    // Ollama defaults and --require-idle stay intact.
    REQUIRE_MESSAGE(
        bench::parse_bench_cli_args(
            {"--backend", "ollama", "--ollama-url", "http://127.0.0.1:11434", "--require-idle", "--model", "qwen3:0.6b"},
            parsed, error),
        error);
    CHECK(parsed.backend == "ollama");
    CHECK(parsed.require_idle);
    CHECK(parsed.ollama_url == "http://127.0.0.1:11434");

    // Conflicting and invalid backend/option combinations are rejected.
    CHECK_FALSE(bench::parse_bench_cli_args({"--llamaserver-url", "http://127.0.0.1:18080", "--model", "m"}, parsed,
                                            error));
    CHECK(error.find("--llamaserver-url") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args(
        {"--backend", "ollama", "--llamaserver-url", "http://127.0.0.1:18080", "--model", "m"}, parsed, error));
    CHECK(error.find("--llamaserver-url") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args(
        {"--backend", "mock", "--llamaserver-url", "http://127.0.0.1:18080", "--model", "mock:tiny"}, parsed, error));
    CHECK(error.find("--llamaserver-url") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args(
        {"--backend", "ollama", "--llamaserver-config", "server.json", "--model", "m"}, parsed, error));
    CHECK(error.find("--llamaserver-config") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args(
        {"--backend", "mock", "--llamaserver-config", "server.json", "--model", "mock:tiny"}, parsed, error));
    CHECK(error.find("--llamaserver-config") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args(
        {"--backend", "llamaserver", "--ollama-url", "http://127.0.0.1:11434", "--model", "m"}, parsed, error));
    CHECK(error.find("--ollama-url") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "llamaserver", "--require-idle", "--model", "m"}, parsed,
                                            error));
    CHECK(error.find("--require-idle") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "mock", "--require-idle", "--model", "mock:tiny"}, parsed,
                                            error));
    CHECK(error.find("--require-idle") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "llamaserver", "--llamaserver-url", "", "--model", "m"},
                                            parsed, error));
    CHECK(error.find("--llamaserver-url") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "llamaserver", "--llamaserver-config", "", "--model", "m"},
                                            parsed, error));
    CHECK(error.find("--llamaserver-config") != std::string::npos);

    CHECK_FALSE(bench::parse_bench_cli_args({"--backend", "bogus", "--model", "m"}, parsed, error));
    CHECK(error.find("unsupported backend") != std::string::npos);

    const std::string help = bench::render_bench_cli_help();
    CHECK(help.find("llamaserver") != std::string::npos);
    CHECK(help.find("--llamaserver-url") != std::string::npos);
    CHECK(help.find("--llamaserver-config") != std::string::npos);
}

TEST_CASE("llamaserver bench backend enforces loopback containment and config validation") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    // Non-loopback URLs are always rejected before creating a backend.
    for (const char* remote_url :
         {"http://192.0.2.1:18080", "http://127.0.0.1.attacker.example:18080", "http://0.0.0.0:18080"}) {
        CAPTURE(remote_url);
        bench::LlamaServerBenchOptions opts;
        opts.url = remote_url;
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
    {
        bench::LlamaServerBenchOptions opts;
        opts.url = "https://example.com:18080";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK((res.status().code() == ErrorCode::invalid_argument || res.status().code() == ErrorCode::unsupported));
    }
    {
        bench::LlamaServerBenchOptions opts;
        opts.url = "not-a-valid-url";
        auto res = bench::make_llamaserver_bench_backend(opts);
        CHECK_FALSE(res.ok());
    }

    // Valid loopback URLs construct the llamaserver backend cleanly.
    {
        bench::LlamaServerBenchOptions opts;
        opts.url = "http://127.0.0.1:18080";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        REQUIRE(res.value() != nullptr);
        CHECK(res.value()->name() == "llamaserver");
    }

    // Invalid spawn options fail fast with invalid_argument (not deferred to probe()).
    {
        bench::LlamaServerBenchOptions opts;
        opts.mode = "spawn";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
    {
        bench::LlamaServerBenchOptions opts;
        opts.mode = "spawn";
        opts.executable = "llama-server.exe";
        opts.args = {"--host", "0.0.0.0"};
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
    {
        bench::LlamaServerBenchOptions opts;
        opts.mode = "bogus";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }

#if defined(SONDER_HAS_SERVER)
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto cfg_path = std::filesystem::temp_directory_path() /
                          ("sonder-bench-llamaserver-cfg-" + std::to_string(unique) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{cfg_path};
    const auto write_cfg = [&](const std::string& body) {
        std::ofstream out(cfg_path, std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out << body;
    };

    // Valid loopback attach config loads cleanly.
    write_cfg(R"({"mode":"attach","base_url":"http://127.0.0.1:18080","native_completion":false,"connect_timeout_ms":100})");
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        CHECK(res.value()->name() == "llamaserver");
    }

    // Passing both config_path and url is strictly refused (mutually exclusive).
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        opts.url = "http://[::1]:18080";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        opts.url = "http://192.0.2.1:18080";
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }

    // Spawn config with empty executable or forbidden --port arg fails eagerly.
    write_cfg(R"({"mode":"spawn","executable":""})");
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
    write_cfg(R"({"mode":"spawn","executable":"llama-server.exe","args":["--port","8080"]})");
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }

    // Config enabling allow_remote is strictly refused in benchmark mode.
    write_cfg(R"({"mode":"attach","base_url":"http://127.0.0.1:18080","allow_remote":true})");
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }

    // Config pointing to a non-loopback URL is refused.
    write_cfg(R"({"mode":"attach","base_url":"http://192.0.2.1:18080","allow_remote":false})");
    {
        bench::LlamaServerBenchOptions opts;
        opts.config_path = cfg_path.string();
        auto res = bench::make_llamaserver_bench_backend(opts);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::invalid_argument);
    }
#endif
#else
    bench::LlamaServerBenchOptions opts;
    opts.url = "http://127.0.0.1:18080";
    auto res = bench::make_llamaserver_bench_backend(opts);
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::unsupported);
#endif
}

TEST_CASE("sonder-infer bench command spec exposes list-models and llamaserver options") {
    const auto& spec = sonder::cli::bench_command();
    REQUIRE(spec.find("list-models") != nullptr);
    CHECK(spec.find("list-models")->kind == sonder::cli::OptionKind::flag);
    REQUIRE(spec.find("llamaserver-url") != nullptr);
    REQUIRE(spec.find("llamaserver-config") != nullptr);
    REQUIRE(spec.find("concurrency") != nullptr);
    REQUIRE(spec.find("no-gpu-snapshot") != nullptr);
    REQUIRE(spec.find("hardware") != nullptr);

    std::string error;
    auto parsed = sonder::cli::parse_command_args(
        spec, {"--backend", "llamaserver", "--llamaserver-url", "http://127.0.0.1:18080", "--list-models"}, error);
    REQUIRE_MESSAGE(parsed.has_value(), error);
    CHECK(parsed->get("backend") == std::optional<std::string>("llamaserver"));
    CHECK(parsed->get("llamaserver-url") == std::optional<std::string>("http://127.0.0.1:18080"));
    CHECK(parsed->has("list-models"));
}

TEST_CASE("speculative decoding and cached prompt metrics aggregate weighted rate and render markdown") {
    // Prompt p1: 1 warmup + 2 measured; Prompt p2: 1 warmup + 2 measured.
    // Warmup values are intentionally skewed to verify they stay on runs[] and never pollute summaries.
    std::vector<SpecScriptedReply> replies;
    {
        SpecScriptedReply r;
        r.timings_only = true;
        r.draft_n = 100;
        r.draft_n_accepted = 10;
        r.cache_n = 999;
        replies.push_back(r);  // p1 warmup
    }
    {
        SpecScriptedReply r;
        r.timings_only = true;  // raw BackendTimings only (draft_n / draft_n_accepted / cache_n)
        r.draft_n = 10;
        r.draft_n_accepted = 7;
        r.cache_n = 0;  // explicit zero is a real measurement, not absence
        replies.push_back(r);  // p1 measured #1 (rate 0.7)
    }
    {
        SpecScriptedReply r;
        r.timings_only = false;
        r.draft_n = 20;
        r.draft_n_accepted = 17;
        r.cache_n = 64;
        replies.push_back(r);  // p1 measured #2 (rate 0.85; p1 weighted = 24/30 = 0.80)
    }
    {
        SpecScriptedReply r;
        replies.push_back(r);  // p2 warmup (no draft or cache counters reported)
    }
    {
        SpecScriptedReply r;
        r.timings_only = false;
        r.draft_n = 10;
        r.draft_n_accepted = 6;
        r.cache_n = 128;
        replies.push_back(r);  // p2 measured #1 (rate 0.60)
    }
    {
        SpecScriptedReply r;
        r.timings_only = true;
        r.draft_n = 30;
        r.draft_n_accepted = 24;
        r.cache_n = 192;
        replies.push_back(r);  // p2 measured #2 (rate 0.80; p2 weighted = 30/40 = 0.75)
    }

    auto seen_sampling = std::make_shared<std::vector<SamplingConfig>>();
    EngineOptions eo;
    eo.sample_devices_on_start = false;
    eo.telemetry.level = TelemetryLevel::metrics;
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(replies, seen_sampling)).ok());

    ModelLoadOptions lo;
    lo.model = "bonsai-2-q1";
    auto model = engine.load_model("llamaserver", lo);
    REQUIRE_MESSAGE(model.ok(), model.status().to_string());

    auto corpus = bench::parse_corpus(R"({"schema":"sonder.inference.corpus/1","name":"spec","version":"1","prompts":[
        {"id":"p1","workload":"coding","prompt":"implement add","max_tokens":12},
        {"id":"p2","workload":"interactive_chat","prompt":"hello","max_tokens":16}]})");
    REQUIRE_MESSAGE(corpus.ok(), corpus.status().to_string());

    bench::Options opts;
    opts.warmup_runs = 1;
    opts.measured_runs = 2;
    opts.capture_gpu_snapshot = false;
    opts.sampling = SamplingConfig{};
    opts.sampling.explicit_only = true;  // mirror `sonder-infer bench --backend llamaserver`

    std::ostringstream progress;
    const json::Value doc = bench::run(engine, model.value(), corpus.value(), opts, &progress);

    // Per-prompt max_tokens is flagged explicit even when caller sampling uses explicit_only,
    // and sampling provenance honestly reports explicit_only=true with null unset fields.
    REQUIRE(seen_sampling->size() == 6u);
    CHECK((*seen_sampling)[0].max_tokens == 12);
    CHECK((*seen_sampling)[0].is_explicit(SamplingConfig::kMaxTokens));
    CHECK((*seen_sampling)[3].max_tokens == 16);
    CHECK((*seen_sampling)[3].is_explicit(SamplingConfig::kMaxTokens));
    const auto* sampling_doc = doc.find("sampling");
    REQUIRE(sampling_doc != nullptr);
    REQUIRE(sampling_doc->find("explicit_only") != nullptr);
    CHECK(sampling_doc->find("explicit_only")->as_bool());
    REQUIRE(sampling_doc->find("temperature") != nullptr);
    CHECK(sampling_doc->find("temperature")->is_null());
    REQUIRE(sampling_doc->find("seed") != nullptr);
    CHECK(sampling_doc->find("seed")->is_null());

    const auto& rows = doc.find("runs")->as_array();
    REQUIRE(rows.size() == 6u);

    // p1 warmup row preserves its own per-request measurements.
    CHECK(rows[0].find("warmup")->as_bool());
    REQUIRE(rows[0].find("draft_tokens") != nullptr);
    REQUIRE(rows[0].find("draft_accepted_tokens") != nullptr);
    REQUIRE(rows[0].find("draft_acceptance_rate") != nullptr);
    REQUIRE(rows[0].find("cached_prompt_tokens") != nullptr);
    CHECK(rows[0].find("draft_tokens")->as_uint() == 100u);
    CHECK(rows[0].find("draft_accepted_tokens")->as_uint() == 10u);
    CHECK(rows[0].find("draft_acceptance_rate")->as_double() == doctest::Approx(0.1));
    CHECK(rows[0].find("cached_prompt_tokens")->as_uint() == 999u);

    // p1 measured #1 (BackendTimings only, explicit cache_n=0).
    CHECK_FALSE(rows[1].find("warmup")->as_bool(true));
    REQUIRE(rows[1].find("draft_tokens") != nullptr);
    REQUIRE(rows[1].find("draft_accepted_tokens") != nullptr);
    REQUIRE(rows[1].find("draft_acceptance_rate") != nullptr);
    REQUIRE(rows[1].find("cached_prompt_tokens") != nullptr);
    CHECK(rows[1].find("draft_tokens")->as_uint() == 10u);
    CHECK(rows[1].find("draft_accepted_tokens")->as_uint() == 7u);
    CHECK(rows[1].find("draft_acceptance_rate")->as_double() == doctest::Approx(0.7));
    CHECK(rows[1].find("cached_prompt_tokens")->as_uint() == 0u);

    // p1 measured #2.
    REQUIRE(rows[2].find("draft_tokens") != nullptr);
    REQUIRE(rows[2].find("draft_accepted_tokens") != nullptr);
    REQUIRE(rows[2].find("draft_acceptance_rate") != nullptr);
    REQUIRE(rows[2].find("cached_prompt_tokens") != nullptr);
    CHECK(rows[2].find("draft_tokens")->as_uint() == 20u);
    CHECK(rows[2].find("draft_accepted_tokens")->as_uint() == 17u);
    CHECK(rows[2].find("draft_acceptance_rate")->as_double() == doctest::Approx(0.85));
    CHECK(rows[2].find("cached_prompt_tokens")->as_uint() == 64u);

    // p2 warmup had no draft or cache counters reported: all four optional fields omitted.
    CHECK(rows[3].find("draft_tokens") == nullptr);
    CHECK(rows[3].find("draft_accepted_tokens") == nullptr);
    CHECK(rows[3].find("draft_acceptance_rate") == nullptr);
    CHECK(rows[3].find("cached_prompt_tokens") == nullptr);

    // p2 measured #1 and #2.
    REQUIRE(rows[4].find("draft_tokens") != nullptr);
    REQUIRE(rows[4].find("draft_accepted_tokens") != nullptr);
    REQUIRE(rows[4].find("draft_acceptance_rate") != nullptr);
    REQUIRE(rows[4].find("cached_prompt_tokens") != nullptr);
    CHECK(rows[4].find("draft_tokens")->as_uint() == 10u);
    CHECK(rows[4].find("draft_accepted_tokens")->as_uint() == 6u);
    CHECK(rows[4].find("draft_acceptance_rate")->as_double() == doctest::Approx(0.6));
    CHECK(rows[4].find("cached_prompt_tokens")->as_uint() == 128u);

    REQUIRE(rows[5].find("draft_tokens") != nullptr);
    REQUIRE(rows[5].find("draft_accepted_tokens") != nullptr);
    REQUIRE(rows[5].find("draft_acceptance_rate") != nullptr);
    REQUIRE(rows[5].find("cached_prompt_tokens") != nullptr);
    CHECK(rows[5].find("draft_tokens")->as_uint() == 30u);
    CHECK(rows[5].find("draft_accepted_tokens")->as_uint() == 24u);
    CHECK(rows[5].find("draft_acceptance_rate")->as_double() == doctest::Approx(0.8));
    CHECK(rows[5].find("cached_prompt_tokens")->as_uint() == 192u);

    // Per-prompt weighted aggregates (warmups excluded).
    const auto& pp = doc.find("per_prompt")->as_array();
    REQUIRE(pp.size() == 2u);
    REQUIRE(pp[0].find("draft_tokens_total") != nullptr);
    REQUIRE(pp[0].find("draft_accepted_total") != nullptr);
    REQUIRE(pp[0].find("draft_acceptance_rate") != nullptr);
    REQUIRE(pp[0].find("cached_prompt_tokens_total") != nullptr);
    CHECK(pp[0].find("draft_tokens_total")->as_uint() == 30u);
    CHECK(pp[0].find("draft_accepted_total")->as_uint() == 24u);
    CHECK(pp[0].find("draft_acceptance_rate")->as_double() == doctest::Approx(24.0 / 30.0));
    CHECK(pp[0].find("cached_prompt_tokens_total")->as_uint() == 64u);

    REQUIRE(pp[1].find("draft_tokens_total") != nullptr);
    REQUIRE(pp[1].find("draft_accepted_total") != nullptr);
    REQUIRE(pp[1].find("draft_acceptance_rate") != nullptr);
    REQUIRE(pp[1].find("cached_prompt_tokens_total") != nullptr);
    CHECK(pp[1].find("draft_tokens_total")->as_uint() == 40u);
    CHECK(pp[1].find("draft_accepted_total")->as_uint() == 30u);
    CHECK(pp[1].find("draft_acceptance_rate")->as_double() == doctest::Approx(30.0 / 40.0));
    CHECK(pp[1].find("cached_prompt_tokens_total")->as_uint() == 320u);

    // Overall summary weighted aggregate: (24 + 30) / (30 + 40) = 54 / 70.
    const auto* summary = doc.find("summary");
    REQUIRE(summary != nullptr);
    REQUIRE(summary->find("draft_tokens_total") != nullptr);
    REQUIRE(summary->find("draft_accepted_total") != nullptr);
    REQUIRE(summary->find("draft_acceptance_rate") != nullptr);
    REQUIRE(summary->find("cached_prompt_tokens_total") != nullptr);
    CHECK(summary->find("draft_tokens_total")->as_uint() == 70u);
    CHECK(summary->find("draft_accepted_total")->as_uint() == 54u);
    CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(54.0 / 70.0));
    CHECK(summary->find("cached_prompt_tokens_total")->as_uint() == 384u);

    // Markdown renders the Speculation acceptance row with accepted / total (XX.X%).
    const std::string md = bench::render_markdown(doc);
    CHECK(md.find("| Speculation | 54 / 70 (77.1%) |") != std::string::npos);
    CHECK(md.find("MOCK BACKEND") == std::string::npos);
    CHECK(progress.str().find("draft_acc=") != std::string::npos);
}

TEST_CASE("speculation metrics omitted cleanly when draft tokens are absent or zero") {
    // Case 1: mock backend reports neither draft nor cached prompt tokens.
    {
        sonder_test::Harness h({}, TelemetryLevel::metrics);
        bench::Corpus corpus;
        corpus.name = "no-spec";
        corpus.prompts = {{"p1", "interactive_chat", "hello", 4}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 1;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(*h.engine, h.model, corpus, opts);
        const auto& row = doc.find("runs")->as_array().front();
        CHECK(row.find("draft_tokens") == nullptr);
        CHECK(row.find("draft_accepted_tokens") == nullptr);
        CHECK(row.find("draft_acceptance_rate") == nullptr);
        CHECK(row.find("cached_prompt_tokens") == nullptr);

        const auto* summary = doc.find("summary");
        CHECK(summary->find("draft_tokens_total") == nullptr);
        CHECK(summary->find("draft_accepted_total") == nullptr);
        CHECK(summary->find("draft_acceptance_rate") == nullptr);
        CHECK(summary->find("cached_prompt_tokens_total") == nullptr);

        const auto& pp0 = doc.find("per_prompt")->as_array().front();
        CHECK(pp0.find("draft_tokens_total") == nullptr);
        CHECK(pp0.find("draft_accepted_total") == nullptr);
        CHECK(pp0.find("draft_acceptance_rate") == nullptr);
        CHECK(pp0.find("cached_prompt_tokens_total") == nullptr);

        CHECK(bench::render_markdown(doc).find("Speculation") == std::string::npos);
    }

    // Case 2: backend reports draft_n=0, draft_n_accepted=0, cache_n=32.
    {
        SpecScriptedReply r;
        r.timings_only = true;
        r.draft_n = 0;
        r.draft_n_accepted = 0;
        r.cache_n = 32;

        EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.telemetry.level = TelemetryLevel::metrics;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(std::vector<SpecScriptedReply>{r})).ok());

        ModelLoadOptions lo;
        lo.model = "bonsai-2-q1";
        auto model = engine.load_model("llamaserver", lo);
        REQUIRE(model.ok());

        bench::Corpus corpus;
        corpus.name = "zero-draft";
        corpus.prompts = {{"p1", "coding", "hi", 8}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 1;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(engine, model.value(), corpus, opts);
        const auto& row = doc.find("runs")->as_array().front();
        REQUIRE(row.find("draft_tokens") != nullptr);
        CHECK(row.find("draft_tokens")->as_uint() == 0u);
        REQUIRE(row.find("draft_accepted_tokens") != nullptr);
        CHECK(row.find("draft_accepted_tokens")->as_uint() == 0u);
        CHECK(row.find("draft_acceptance_rate") == nullptr);
        REQUIRE(row.find("cached_prompt_tokens") != nullptr);
        CHECK(row.find("cached_prompt_tokens")->as_uint() == 32u);

        const auto* summary = doc.find("summary");
        CHECK(summary->find("draft_tokens_total") == nullptr);
        CHECK(summary->find("draft_accepted_total") == nullptr);
        CHECK(summary->find("draft_acceptance_rate") == nullptr);
        REQUIRE(summary->find("cached_prompt_tokens_total") != nullptr);
        CHECK(summary->find("cached_prompt_tokens_total")->as_uint() == 32u);

        const auto& pp0 = doc.find("per_prompt")->as_array().front();
        CHECK(pp0.find("draft_tokens_total") == nullptr);
        CHECK(pp0.find("draft_accepted_total") == nullptr);
        CHECK(pp0.find("draft_acceptance_rate") == nullptr);
        REQUIRE(pp0.find("cached_prompt_tokens_total") != nullptr);
        CHECK(pp0.find("cached_prompt_tokens_total")->as_uint() == 32u);

        CHECK(bench::render_markdown(doc).find("Speculation") == std::string::npos);
    }
}

TEST_CASE("speculative decoding and cached prompt metrics under concurrency 4") {
    std::vector<SpecScriptedReply> replies;
    // Sequential warmup runs first before concurrent measured waves.
    {
        SpecScriptedReply warmup;
        warmup.timings_only = true;
        warmup.draft_n = 100;
        warmup.draft_n_accepted = 10;
        warmup.cache_n = 999;
        replies.push_back(warmup);
    }
    // 2 measured runs x 4 concurrent workers = 8 measured requests.
    for (int i = 0; i < 8; ++i) {
        SpecScriptedReply r;
        r.timings_only = (i % 2 == 0);
        r.draft_n = 12;
        r.draft_n_accepted = 9;
        r.cache_n = 40;
        replies.push_back(r);
    }

    EngineOptions eo;
    eo.sample_devices_on_start = false;
    eo.telemetry.level = TelemetryLevel::metrics;
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(replies)).ok());

    ModelLoadOptions lo;
    lo.model = "bonsai-2-q1";
    auto model = engine.load_model("llamaserver", lo);
    REQUIRE_MESSAGE(model.ok(), model.status().to_string());

    bench::Corpus corpus;
    corpus.name = "spec-conc4";
    corpus.prompts = {{"p1", "coding", "write parser", 12}};

    bench::Options opts;
    opts.warmup_runs = 1;
    opts.measured_runs = 2;
    opts.concurrency = 4;
    opts.capture_gpu_snapshot = false;

    const json::Value doc = bench::run(engine, model.value(), corpus, opts);
    REQUIRE(doc.find("runs")->as_array().size() == 9u);

    const auto* sampling_doc = doc.find("sampling");
    REQUIRE(sampling_doc != nullptr);
    REQUIRE(sampling_doc->find("explicit_only") != nullptr);
    CHECK_FALSE(sampling_doc->find("explicit_only")->as_bool(true));
    CHECK(sampling_doc->find("temperature")->is_number());
    CHECK(sampling_doc->find("seed")->is_integer());

    const auto* summary = doc.find("summary");
    REQUIRE(summary != nullptr);
    CHECK(summary->find("measured_requests")->as_int() == 8);
    REQUIRE(summary->find("draft_tokens_total") != nullptr);
    REQUIRE(summary->find("draft_accepted_total") != nullptr);
    REQUIRE(summary->find("draft_acceptance_rate") != nullptr);
    REQUIRE(summary->find("cached_prompt_tokens_total") != nullptr);
    CHECK(summary->find("draft_tokens_total")->as_uint() == 96u);
    CHECK(summary->find("draft_accepted_total")->as_uint() == 72u);
    CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(0.75));
    CHECK(summary->find("cached_prompt_tokens_total")->as_uint() == 320u);

    const auto& pp0 = doc.find("per_prompt")->as_array().front();
    CHECK(pp0.find("draft_tokens_total")->as_uint() == 96u);
    CHECK(pp0.find("draft_accepted_total")->as_uint() == 72u);
    CHECK(pp0.find("draft_acceptance_rate")->as_double() == doctest::Approx(0.75));
    CHECK(pp0.find("cached_prompt_tokens_total")->as_uint() == 320u);

    const std::string md = bench::render_markdown(doc);
    CHECK(md.find("| Speculation | 72 / 96 (75.0%) |") != std::string::npos);
}

TEST_CASE("speculation clamping denominator parity saturation and failed or cancelled exclusion") {
    // 1. Clamp draft_accepted_tokens <= draft_tokens (including draft_tokens == 0).
    {
        std::vector<SpecScriptedReply> replies;
        {
            SpecScriptedReply r;
            r.timings_only = true;
            r.draft_n = 10;
            r.draft_n_accepted = 25;  // exceeds draft_n; must clamp to 10
            replies.push_back(r);
        }
        {
            SpecScriptedReply r;
            r.timings_only = true;
            r.draft_n = 0;
            r.draft_n_accepted = 5;  // draft_n == 0; must clamp accepted to 0 and omit rate
            replies.push_back(r);
        }

        EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.telemetry.level = TelemetryLevel::metrics;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(replies)).ok());

        ModelLoadOptions lo;
        lo.model = "bonsai-2-q1";
        auto model = engine.load_model("llamaserver", lo);
        REQUIRE(model.ok());

        bench::Corpus corpus;
        corpus.name = "clamp";
        corpus.prompts = {{"p1", "coding", "hi", 8}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 2;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(engine, model.value(), corpus, opts);
        const auto& rows = doc.find("runs")->as_array();
        REQUIRE(rows.size() == 2u);
        REQUIRE(rows[0].find("draft_tokens") != nullptr);
        REQUIRE(rows[0].find("draft_accepted_tokens") != nullptr);
        REQUIRE(rows[0].find("draft_acceptance_rate") != nullptr);
        CHECK(rows[0].find("draft_tokens")->as_uint() == 10u);
        CHECK(rows[0].find("draft_accepted_tokens")->as_uint() == 10u);
        CHECK(rows[0].find("draft_acceptance_rate")->as_double() == doctest::Approx(1.0));

        REQUIRE(rows[1].find("draft_tokens") != nullptr);
        REQUIRE(rows[1].find("draft_accepted_tokens") != nullptr);
        CHECK(rows[1].find("draft_tokens")->as_uint() == 0u);
        CHECK(rows[1].find("draft_accepted_tokens")->as_uint() == 0u);
        CHECK(rows[1].find("draft_acceptance_rate") == nullptr);

        const auto* summary = doc.find("summary");
        REQUIRE(summary != nullptr);
        CHECK(summary->find("draft_tokens_total")->as_uint() == 10u);
        CHECK(summary->find("draft_accepted_total")->as_uint() == 10u);
        CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(1.0));

        // render_markdown clamps out-of-range summary rate and accepted > total too.
        auto malformed = json::parse(
            R"({"backend":{"name":"llamaserver"},"model":{"name":"m"},)"
            R"("summary":{"measured_requests":1,"draft_tokens_total":40,"draft_accepted_total":99,"draft_acceptance_rate":1.5}})");
        REQUIRE(malformed.ok());
        CHECK(bench::render_markdown(malformed.value()).find("| Speculation | 40 / 40 (100.0%) |") !=
              std::string::npos);
    }

    // 2. Partial acceptance reporting emits draft_tokens_evaluated_total so Markdown fraction matches rate.
    {
        std::vector<SpecScriptedReply> replies;
        {
            SpecScriptedReply r;
            r.timings_only = true;
            r.draft_n = 20;
            r.draft_n_accepted = 10;
            replies.push_back(r);
        }
        {
            SpecScriptedReply r;
            r.timings_only = true;
            r.draft_n = 100;  // accepted absent on this request
            replies.push_back(r);
        }

        EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.telemetry.level = TelemetryLevel::metrics;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(replies)).ok());

        ModelLoadOptions lo;
        lo.model = "bonsai-2-q1";
        auto model = engine.load_model("llamaserver", lo);
        REQUIRE(model.ok());

        bench::Corpus corpus;
        corpus.name = "partial-eval";
        corpus.prompts = {{"p1", "coding", "hi", 8}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 2;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(engine, model.value(), corpus, opts);
        const auto* summary = doc.find("summary");
        REQUIRE(summary != nullptr);
        REQUIRE(summary->find("draft_tokens_total") != nullptr);
        REQUIRE(summary->find("draft_tokens_evaluated_total") != nullptr);
        REQUIRE(summary->find("draft_accepted_total") != nullptr);
        REQUIRE(summary->find("draft_acceptance_rate") != nullptr);
        CHECK(summary->find("draft_tokens_total")->as_uint() == 120u);
        CHECK(summary->find("draft_tokens_evaluated_total")->as_uint() == 20u);
        CHECK(summary->find("draft_accepted_total")->as_uint() == 10u);
        CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(0.5));
        CHECK(bench::render_markdown(doc).find("| Speculation | 10 / 20 (50.0%) |") != std::string::npos);
    }

    // 3. Saturating uint64_t totals near UINT64_MAX preserve accurate long-double acceptance rate
    //    even when both draft_tokens_total and draft_accepted_total saturate at UINT64_MAX.
    {
        constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t big_draft = kMaxU64 - 10u;
        const std::uint64_t big_acc = (kMaxU64 / 2u) + 1000u;
        SpecScriptedReply r;
        r.timings_only = true;
        r.draft_n = big_draft;
        r.draft_n_accepted = big_acc;
        r.cache_n = big_draft;

        EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.telemetry.level = TelemetryLevel::metrics;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(std::vector<SpecScriptedReply>{r, r})).ok());

        ModelLoadOptions lo;
        lo.model = "bonsai-2-q1";
        auto model = engine.load_model("llamaserver", lo);
        REQUIRE(model.ok());

        bench::Corpus corpus;
        corpus.name = "saturate";
        corpus.prompts = {{"p1", "coding", "hi", 8}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 2;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(engine, model.value(), corpus, opts);
        const auto* summary = doc.find("summary");
        REQUIRE(summary != nullptr);
        REQUIRE(summary->find("draft_tokens_total") != nullptr);
        REQUIRE(summary->find("draft_accepted_total") != nullptr);
        REQUIRE(summary->find("draft_acceptance_rate") != nullptr);
        REQUIRE(summary->find("cached_prompt_tokens_total") != nullptr);
        CHECK(summary->find("draft_tokens_total")->as_uint() == kMaxU64);
        CHECK(summary->find("draft_accepted_total")->as_uint() == kMaxU64);
        CHECK(summary->find("cached_prompt_tokens_total")->as_uint() == kMaxU64);
        CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(0.5));
    }

    // 4. Failed and cancelled measured requests are excluded from summary and per_prompt aggregates.
    {
        std::vector<SpecScriptedReply> replies;
        {
            SpecScriptedReply ok;
            ok.timings_only = true;
            ok.draft_n = 10;
            ok.draft_n_accepted = 8;
            ok.cache_n = 20;
            replies.push_back(ok);
        }
        {
            SpecScriptedReply failed;
            failed.fail_request = true;
            failed.draft_n = 500;
            failed.draft_n_accepted = 500;
            failed.cache_n = 5000;
            replies.push_back(failed);
        }
        {
            SpecScriptedReply cancelled;
            cancelled.cancel_request = true;
            cancelled.draft_n = 600;
            cancelled.draft_n_accepted = 600;
            cancelled.cache_n = 6000;
            replies.push_back(cancelled);
        }

        EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.telemetry.level = TelemetryLevel::metrics;
        Engine engine(std::move(eo));
        REQUIRE(engine.register_backend(std::make_shared<SpecStubBackend>(replies)).ok());

        ModelLoadOptions lo;
        lo.model = "bonsai-2-q1";
        auto model = engine.load_model("llamaserver", lo);
        REQUIRE(model.ok());

        bench::Corpus corpus;
        corpus.name = "fail-cancel";
        corpus.prompts = {{"p1", "coding", "hi", 8}};
        bench::Options opts;
        opts.warmup_runs = 0;
        opts.measured_runs = 3;
        opts.capture_gpu_snapshot = false;

        const json::Value doc = bench::run(engine, model.value(), corpus, opts);
        const auto* summary = doc.find("summary");
        REQUIRE(summary != nullptr);
        CHECK(summary->find("measured_requests")->as_int() == 3);
        CHECK(summary->find("failures")->as_int() == 1);
        CHECK(summary->find("cancelled")->as_int() == 1);
        REQUIRE(summary->find("draft_tokens_total") != nullptr);
        REQUIRE(summary->find("draft_accepted_total") != nullptr);
        REQUIRE(summary->find("draft_acceptance_rate") != nullptr);
        REQUIRE(summary->find("cached_prompt_tokens_total") != nullptr);
        CHECK(summary->find("draft_tokens_total")->as_uint() == 10u);
        CHECK(summary->find("draft_accepted_total")->as_uint() == 8u);
        CHECK(summary->find("draft_acceptance_rate")->as_double() == doctest::Approx(0.8));
        CHECK(summary->find("cached_prompt_tokens_total")->as_uint() == 20u);

        const auto& pp0 = doc.find("per_prompt")->as_array().front();
        CHECK(pp0.find("draft_tokens_total")->as_uint() == 10u);
        CHECK(pp0.find("draft_accepted_total")->as_uint() == 8u);
        CHECK(pp0.find("draft_acceptance_rate")->as_double() == doctest::Approx(0.8));
        CHECK(pp0.find("cached_prompt_tokens_total")->as_uint() == 20u);
    }
}
}
