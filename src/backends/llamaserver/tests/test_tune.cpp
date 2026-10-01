#include "../tune.hpp"
#include "../tune_model.hpp"
#include "../log_diagnostics.hpp"
#include <doctest/doctest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
namespace tune = sonder::inference::llamaserver::tune;
namespace {
tune::Measurement clean(tune::Candidate c, tune::Phase phase = tune::Phase::benchmark) {
    tune::Measurement m;
    m.candidate = std::move(c);
    m.phase = phase;
    m.verdict = tune::Verdict::clean;
    m.memory_released = true;
    m.safety_probe_ctx = m.candidate.ctx + tune::kMargin;
    m.dedicated_bytes = 14000 * kMiB;
    m.shared_bytes = 182 * kMiB;
    m.short_tps = 60;
    m.prefill_tps = 1000;
    m.workloads = {{"prose", 0.4, 60.0, 10, 5}, {"code_edit", 0.3, 60.0, 10, 5},
        {"tool_json", 0.2, 60.0, 10, 5}, {"reasoning", 0.1, 60.0, 10, 5}};
    return m;
}
void integer(std::string &out, std::uint64_t n, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) out.push_back(static_cast<char>((n >> (8 * i)) & 255));
}
void string(std::string &out, const std::string &s) { integer(out, s.size(), 8); out += s; }
std::string gguf(bool nextn) {
    std::string out = "GGUF";
    integer(out, 3, 4); integer(out, 1, 8); integer(out, 2, 8);
    string(out, "general.architecture"); integer(out, 8, 4); string(out, "qwen3_next");
    // Tokenizer array is skipped, not mistaken for a tensor or allocated.
    string(out, "tokenizer.ggml.tokens"); integer(out, 9, 4); integer(out, 8, 4); integer(out, 2, 8);
    string(out, "blk.64.nextn.decoy"); string(out, "another token");
    string(out, nextn ? "blk.64.nextn.eh_proj.weight" : "blk.0.attn_q.weight");
    integer(out, 2, 4); integer(out, 32, 8); integer(out, 32, 8); integer(out, 0, 4); integer(out, 0, 8);
    return out;
}
} // namespace

TEST_CASE("tune bisection finds every clean boundary without duplicate probes") {
    const std::vector<std::uint64_t> contexts{32768, 36864, 40960, 45056, 49152, 53248, 57344};
    for (std::size_t count = 0; count <= contexts.size(); ++count) {
        tune::Bisection search(contexts);
        std::set<std::uint64_t> visited;
        while (const auto ctx = search.next()) {
            REQUIRE(visited.insert(*ctx).second);
            search.observe(count != 0 && *ctx <= contexts[count - 1]);
        }
        CHECK(visited.size() <= 3);
        CHECK(search.edge() == (count ? std::optional<std::uint64_t>(contexts[count - 1]) : std::nullopt));
        search.observe(true); // exhausted search remains exhausted
        CHECK_FALSE(search.next());
    }
    CHECK_FALSE(tune::Bisection({}).next());
}
TEST_CASE("tune default grid and capability gate are bounded") {
    tune::Grid grid;
    REQUIRE(tune::validate_grid(grid).ok());
    CHECK(grid.ctx.front() == 32768);
    CHECK(grid.ctx.back() == 139264);
    CHECK(tune::planned_candidates(grid, true).size() == 4212);
    const auto gated = tune::planned_candidates(grid, false);
    CHECK(gated.size() == 324);
    for (const auto &c : gated) { CHECK(c.mtp == 0); CHECK(c.ubatch <= grid.batch); }
    CHECK(grid.kv[2] == tune::KvPair{"q8_0", "q4_0"});
}
TEST_CASE("tune grid rejects typos fractions duplicate contexts and unbounded work") {
    for (const auto *text : {R"({"unknown":1})", R"({"ctx":[]})", R"({"ctx":[32768,32768]})",
            R"({"ctx":[65536,32768]})", R"({"ctx":[32769]})", R"({"ctx":[32768.5]})",
            R"({"ctx":[9223372036854775808]})", R"({"mtp":[1,2]})", R"({"mtp":[0,4]})",
            R"({"ubatch":[4096],"batch":2048})", R"({"kv":[["made_up","q4_0"]]})",
            R"({"spill_threshold_mib":0})", R"({"batch":-1})", R"({"ubatch":[512,512]})"}) {
        CAPTURE(text);
        const auto json = json::parse(text);
        REQUIRE(json.ok());
        CHECK_FALSE(tune::parse_grid(json.value()).ok());
    }
    auto json = json::parse(R"({"kv":[["q8_0","q4_0"]],"ctx":[65536,69632],"ubatch":[1024],"mtp":[0,2],"batch":4096,"shared_baseline_mib":32})");
    REQUIRE(json.ok());
    auto parsed = tune::parse_grid(json.value());
    REQUIRE(parsed.ok());
    CHECK(parsed.value().spill.baseline_bytes == 32 * kMiB);
    CHECK(parsed.value().batch == 4096);
}
TEST_CASE("tune margin is a full 4096 tokens and never underflows") {
    CHECK_FALSE(tune::margin_context(0));
    CHECK_FALSE(tune::margin_context(12288));
    CHECK(tune::margin_context(16384) == 12288);
    CHECK(tune::margin_context(77824) == 73728);
    CHECK(tune::margin_context(65536) == 61440);
}
TEST_CASE("tune search bisects then validates each recommendation and tunes only leaders") {
    tune::Options o;
    o.grid.kv = {{"q4_0", "q4_0"}};
    o.grid.ctx = {65536, 69632, 73728, 77824, 81920};
    auto now = tune::Clock::now();
    const auto deadline = now + std::chrono::minutes(10);
    const auto report = tune::search(o, true, deadline, [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline until) {
        CHECK(now < until);
        now += std::chrono::seconds(1);
        auto m = clean(c, phase);
        if (c.ctx > 77824 || (c.ubatch == 2048 && c.ctx > 69632)) m.verdict = tune::Verdict::spill;
        m.short_tps = 50.0 + (c.mtp == 2 ? 30.0 : 0.0) + (c.ubatch == 1024 ? 5.0 : 0.0);
        for (auto &work : m.workloads) work.tok_s = *m.short_tps;
        return m;
    }, [&] { return now; });
    REQUIRE(report.fast_default);
    REQUIRE(report.long_context);
    const auto &fast = report.results[*report.fast_default];
    const auto &longest = report.results[*report.long_context];
    CHECK(fast.candidate.mtp == 2);
    CHECK(fast.candidate.ubatch == 1024);
    CHECK(longest.candidate.ctx == 73728);
    for (const auto index : {*report.fast_default, *report.long_context}) {
        const auto &winner = report.results[index];
        auto safety = winner.candidate;
        safety.ctx += 4096;
        CHECK(std::any_of(report.results.begin(), report.results.end(), [&](const auto &m) {
            return m.phase == tune::Phase::probe && m.candidate == safety && m.verdict == tune::Verdict::clean;
        }));
    }
}
TEST_CASE("tune failures do not become spill bounds and release failures stop the run") {
    tune::Options o;
    o.grid.kv = {{"q4_0", "q4_0"}};
    o.grid.ctx = {65536, 69632, 73728};
    for (const auto verdict : {tune::Verdict::error, tune::Verdict::slow_kernel, tune::Verdict::release_failed}) {
        unsigned calls = 0;
        auto report = tune::search(o, false, tune::Clock::now() + std::chrono::minutes(1),
            [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline) {
                ++calls;
                auto m = clean(c, phase);
                m.verdict = verdict;
                if (verdict == tune::Verdict::release_failed) m.memory_released = false;
                return m;
            });
        CHECK(calls == 1);
        CHECK_FALSE(report.fast_default);
        CHECK_FALSE(report.long_context);
        REQUIRE(report.results.size() == 1);
        CHECK(report.results.front().verdict == verdict);
    }
}
TEST_CASE("tune deadline stops launching candidates and retains partial results") {
    tune::Options o;
    auto now = tune::Clock::now();
    const auto deadline = now + std::chrono::seconds(10);
    unsigned calls = 0;
    auto report = tune::search(o, false, deadline, [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline until) {
        ++calls;
        now = until;
        auto m = clean(c, phase);
        m.verdict = tune::Verdict::timed_out;
        return m;
    }, [&] { return now; });
    CHECK(calls == 1);
    CHECK_FALSE(report.fast_default);
    now = deadline;
    auto empty = tune::search(o, false, deadline, [&](const auto &, auto, auto) { ++calls; return tune::Measurement{}; }, [&] { return now; });
    CHECK(empty.budget_exhausted);
    CHECK(empty.results.empty());
    CHECK(calls == 1);
}
TEST_CASE("tune ranking excludes unverified measurements and distinguishes fast and long") {
    tune::Report report;
    auto fast = clean({{"q4_0", "q4_0"}, 65536, 1024, 2});
    fast.short_tps = 85;
    for (auto &work : fast.workloads) work.tok_s = *fast.short_tps;
    auto longest = clean({{"q4_0", "q4_0"}, 126976, 512, 0});
    longest.short_tps = 29;
    for (auto &work : longest.workloads) work.tok_s = *longest.short_tps;
    report.results = {fast, longest};
    for (int kind = 0; kind < 6; ++kind) {
        auto invalid = clean({{"q8_0", "q4_0"}, 139264, 512, 0});
        invalid.short_tps = 999;
        if (kind == 0) invalid.verdict = tune::Verdict::slow_kernel;
        if (kind == 1) invalid.memory_released = false;
        if (kind == 2) invalid.safety_probe_ctx = invalid.candidate.ctx;
        if (kind == 3) invalid.shared_bytes.reset();
        if (kind == 4) invalid.phase = tune::Phase::probe;
        if (kind == 5) invalid.short_tps = std::numeric_limits<double>::quiet_NaN();
        report.results.push_back(invalid);
    }
    tune::rank(report);
    CHECK(report.fast_default == 0);
    CHECK(report.long_context == 1);
    report.results = {clean({{"q4_0", "q4_0"}, 61440, 512, 0})};
    tune::rank(report);
    CHECK_FALSE(report.fast_default);
    CHECK(report.long_context == 0);
}
TEST_CASE("tune judges MTP candidates against the MTP clean line") {
    tune::Grid grid;
    const tune::Candidate plain{{"q4_0", "q4_0"}, 73728, 512, 0};
    const tune::Candidate mtp{{"q4_0", "q4_0"}, 73728, 512, 2};
    CHECK(&tune::spill_guard_for(grid, plain) == &grid.spill);
    CHECK(&tune::spill_guard_for(grid, mtp) == &grid.spill_mtp);
    GpuMemorySample s;
    // A healthy MTP load at 73,728 (measured ~310 MiB shared): clean for the
    // MTP rule; the same sample without speculation is a spill.
    s.shared_bytes = 310 * kMiB;
    CHECK_FALSE(is_spilled(s, tune::spill_guard_for(grid, mtp), mtp.ctx));
    CHECK(is_spilled(s, tune::spill_guard_for(grid, plain), plain.ctx));
    // 33+ MiB above the MTP line was measurably slower: a spill.
    s.shared_bytes = (166 + 144 + 33) * kMiB;
    CHECK(is_spilled(s, tune::spill_guard_for(grid, mtp), mtp.ctx));
}

TEST_CASE("tune output is a loadable spawn configuration with complete additive results") {
    tune::Options options;
    options.executable = "C:/a b/llama-server.exe";
    options.model = "C:/model.gguf";
    options.mmproj = "C:/vision.gguf";
    options.environment = {{"TUNE_TEST", "quoted\"value"}};
    tune::Report report;
    report.results = {clean({{"q4_0", "q4_0"}, 73728, 1024, 2})};
    tune::rank(report);
    const auto document = tune::report_json(options, report);
    const auto roundtrip = json::parse(document.dump());
    REQUIRE(roundtrip.ok());
    CHECK(roundtrip.value().find("mode")->as_string() == "spawn");
    const auto *guard = roundtrip.value().find("spill_guard");
    CHECK(guard->find("policy")->as_string() == "auto_fit");
    // The recommended candidate uses MTP (n=2), so its profile carries the MTP
    // clean line, not the fixed 0 + 256 MiB rule.
    CHECK(guard->find("baseline_mib")->as_uint() == 166);
    CHECK(guard->find("baseline_per_1k_ctx_mib")->as_uint() == 2);
    CHECK(guard->find("threshold_mib")->as_uint() == 32);
    CHECK(guard->find("fit_min_ctx")->as_uint() == 32768);
    CHECK(roundtrip.value().find("max_restarts")->as_uint() == 3);
    CHECK(roundtrip.value().find("results")->find("candidates")->as_array().size() == 1);
    CHECK(roundtrip.value().find("results")->find("long_context")->find("config") != nullptr);
    const auto argv = tune::arguments(options, report.results.front().candidate);
    std::vector<std::string> written_args;
    for (const auto &arg : roundtrip.value().find("args")->as_array()) written_args.push_back(arg.as_string());
    CHECK(written_args == argv);
    const auto flag_value = [&](const std::string &flag) {
        const auto it = std::find(written_args.begin(), written_args.end(), flag);
        return it != written_args.end() && std::next(it) != written_args.end() ? *std::next(it) : std::string{};
    };
    CHECK(flag_value("--parallel") == "1");
    CHECK(flag_value("--cache-ram") == "2048");
    CHECK(flag_value("--ctx-checkpoints") == "8");
    CHECK(flag_value("--checkpoint-min-step") == "8192");
    CHECK(flag_value("--reasoning-format") == "deepseek");
    CHECK(flag_value("--min-p") == "0.000000");
    for (const auto *flag : {"--no-kv-unified", "--jinja", "--metrics"})
        CHECK(std::find(written_args.begin(), written_args.end(), flag) != written_args.end());
    CHECK(find_context_size(argv) == 73728);
    CHECK(read_kv_cache_config(argv).flash_attn == "on");
    CHECK(std::find(argv.begin(), argv.end(), "--mmproj") != argv.end());
    CHECK(validate_process_arguments(argv).ok());
#if defined(SONDER_HAS_SERVER)
    const auto path = std::filesystem::temp_directory_path() / ("sonder-tune-config-" + std::to_string(tune::Clock::now().time_since_epoch().count()) + ".json");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); } } cleanup{path};
    { std::ofstream file(path, std::ios::binary); file << document.dump(); }
    BackendSetup setup;
    auto loaded = load_llamaserver_config(path.string(), setup);
    REQUIRE_MESSAGE(loaded.ok(), loaded.message());
    CHECK(setup.llamaserver_executable == options.executable);
    CHECK(setup.llamaserver_args == argv);
    CHECK(setup.llamaserver_environment == options.environment);
    CHECK(setup.llamaserver_spill_policy == "auto_fit");
    CHECK(setup.llamaserver_spill_baseline_per_1k_ctx_mib == 2);
    for (const auto *invalid : {R"({"env":{"BAD=KEY":"x"}})", R"({"env":{"A":1}})",
                              R"({"results":{}})", R"({"surprise":true})"}) {
        { std::ofstream file(path, std::ios::binary); file << invalid; }
        CHECK_FALSE(load_llamaserver_config(path.string(), setup).ok());
        CHECK(setup.llamaserver_args == argv); // loader remains transactional
    }
#endif
    tune::Report empty;
    CHECK(tune::report_json(options, empty).find("mode") == nullptr);
    CHECK(tune::markdown(empty).find("unavailable") != std::string::npos);
}
TEST_CASE("tune dry run never opens model or executable and respects CLI errors") {
    std::ostringstream out, err;
    CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf", "--dry-run",
        "--grid", R"({"kv":[["q4_0","q4_0"]],"ctx":[65536],"ubatch":[512],"mtp":[0]})"}, out, err) == 0);
    CHECK(out.str().find("| q4_0/q4_0 | 65536 | 512 | 0 | none | 0 |") != std::string::npos);
    CHECK(err.str().empty());
    CHECK(tune::tune_main({"--nonsense", "--help"}, out, err) == 0);
    for (const auto &args : std::vector<std::vector<std::string>>{
            {"--dry-run"}, {"--executable", "x", "--model", "m", "--budget-minutes", "nan", "--dry-run"},
            {"--executable", "x", "--model", "m", "--env", "=bad", "--dry-run"},
            {"--executable", "x", "--model", "m", "--budget-minutes", "0", "--dry-run"}})
        CHECK(tune::tune_main(args, out, err) == 2);
}
TEST_CASE("tune GGUF scanner reuses architecture classification and gates actual nextn tensors") {
    for (const bool nextn : {false, true}) {
        const auto bytes = gguf(nextn);
        std::istringstream input(bytes);
        auto info = tune::read_model_info(input);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK(info.value().architecture == ModelArchitecture::hybrid);
        CHECK(info.value().has_nextn == nextn);
        // Every truncation within the directory must fail, even after nextn was found.
        for (std::size_t size = 0; size < bytes.size(); ++size) {
            std::istringstream truncated(bytes.substr(0, size));
            CHECK_FALSE(tune::read_model_info(truncated).ok());
        }
        std::istringstream cancelled(bytes);
        CHECK_FALSE(tune::read_model_info(cancelled, [] { return true; }).ok());
    }
    auto bad = gguf(true);
    bad[4] = 9;
    std::istringstream input(bad);
    CHECK_FALSE(tune::read_model_info(input).ok());
}
TEST_CASE("tune MTP help gate requires the spec type section and draft limit flag") {
    CHECK(tune::help_supports_mtp("--spec-type TYPE\n    possible: none, draft-mtp\n--spec-draft-n-max N\n"));
    CHECK_FALSE(tune::help_supports_mtp("--spec-type none\n  --other draft-mtp\n--spec-draft-n-max N\n"));
    CHECK_FALSE(tune::help_supports_mtp("--spec-type draft-mtp\n"));
    CHECK_FALSE(tune::help_supports_mtp("--spec-typex draft-mtp\n--spec-draft-n-max N\n"));
}

TEST_CASE("tune MTP grid enumerates spec type and p-min cross product while baseline stays singular") {
    tune::Grid grid;
    grid.kv = {{"q4_0", "q4_0"}};
    grid.ctx = {65536};
    grid.ubatch = {512};
    grid.mtp = {0, 2};
    grid.spec_type = {"draft-mtp", "draft-mtp,ngram-mod"};
    grid.p_min = {0.0, 0.5};
    const auto rows = tune::planned_candidates(grid, true);
    REQUIRE(rows.size() == 5);
    CHECK(std::count_if(rows.begin(), rows.end(), [](const auto &c) { return c.mtp == 0; }) == 1);
    CHECK(std::count_if(rows.begin(), rows.end(), [](const auto &c) { return c.mtp != 0; }) == 4);
    CHECK(std::count_if(rows.begin(), rows.end(), [](const auto &c) {
        return c.spec_type == "draft-mtp,ngram-mod" && c.p_min == 0.5;
    }) == 1);
}

TEST_CASE("tune search measures every MTP spec and p-min variant with safety probes") {
    tune::Options options;
    options.grid.kv = {{"q4_0", "q4_0"}};
    options.grid.ctx = {65536, 69632};
    options.grid.ubatch = {512};
    options.grid.mtp = {0, 1};
    options.grid.spec_type = {"draft-mtp", "draft-mtp,ngram-mod"};
    options.grid.p_min = {0.0, 0.5};
    auto now = tune::Clock::now();
    const auto report = tune::search(options, true, now + std::chrono::minutes(2),
        [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline) {
            auto m = clean(c, phase);
            m.short_tps = 50.0 + c.mtp + c.p_min;
            for (auto &work : m.workloads) work.tok_s = *m.short_tps;
            return m;
        }, [&] { return now; });
    const auto variants = std::count_if(report.results.begin(), report.results.end(), [](const auto &m) {
        return m.phase == tune::Phase::benchmark && m.candidate.mtp == 1;
    });
    CHECK(variants >= 4);
    for (const auto &type : options.grid.spec_type)
        for (const auto p_min : options.grid.p_min)
            CHECK(std::any_of(report.results.begin(), report.results.end(), [&](const auto &m) {
                return m.phase == tune::Phase::benchmark && m.candidate.mtp == 1 &&
                    m.candidate.spec_type == type && m.candidate.p_min == p_min;
            }));
    for (const auto &m : report.results)
        if (m.phase == tune::Phase::benchmark)
            CHECK(std::any_of(report.results.begin(), report.results.end(), [&](const auto &probe) {
                return probe.phase == tune::Phase::probe && probe.candidate ==
                    tune::Candidate{m.candidate.kv, m.candidate.ctx + tune::kMargin, m.candidate.ubatch,
                        m.candidate.mtp, m.candidate.spec_type, m.candidate.p_min};
            }));
}

TEST_CASE("tune profile emits bounded drop-in flags and typed overrides") {
    tune::Options options;
    options.model = "model.gguf";
    auto argv = tune::arguments(options, {{"q4_0", "q4_0"}, 65536, 512, 0});
    const auto has = [&](const std::string &value) { return std::find(argv.begin(), argv.end(), value) != argv.end(); };
    CHECK(has("--parallel")); CHECK(has("1")); CHECK(has("--no-kv-unified")); CHECK(has("--cache-ram"));
    CHECK(has("--ctx-checkpoints")); CHECK(has("--checkpoint-min-step")); CHECK(has("--jinja"));
    CHECK(has("--reasoning-format")); CHECK(has("deepseek")); CHECK(has("--min-p")); CHECK(has("--metrics"));

    auto parsed = json::parse(R"({"profile":{"parallel":1,"no_kv_unified":false,"cache_ram":1024,"ctx_checkpoints":4,"checkpoint_min_step":4096,"jinja":false,"reasoning_format":"none","min_p":0.25,"metrics":false},"spec_type":["draft-mtp"],"p_min":[0,0.5]})");
    REQUIRE(parsed.ok());
    auto grid = tune::parse_grid(parsed.value());
    REQUIRE(grid.ok());
    options.grid = grid.value();
    argv = tune::arguments(options, {{"q4_0", "q4_0"}, 65536, 512, 2, "draft-mtp", 0.5});
    CHECK(std::find(argv.begin(), argv.end(), "--no-kv-unified") == argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--kv-unified") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--no-jinja") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--parallel") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--metrics") == argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--reasoning-format") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "none") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--spec-draft-p-min") != argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "0.500000") != argv.end());
    CHECK(std::count(argv.begin(), argv.end(), "--min-p") == 1);

    for (const auto *text : {R"({"profile":{"parallel":2}})", R"({"profile":{"cache_ram":-1}})",
            R"({"profile":{"checkpoint_min_step":18446744073709551616}})"}) {
        auto bad = json::parse(text);
        REQUIRE(bad.ok());
        CHECK_FALSE(tune::parse_grid(bad.value()).ok());
    }
}

TEST_CASE("tune weighted workload ranking and serialization require all natural classes") {
    const std::vector<tune::WorkloadMeasurement> workloads{
        {"prose", 0.4, 100.0, 10, 5}, {"code_edit", 0.3, 80.0, 10, 4},
        {"tool_json", 0.2, 60.0, 10, 3}, {"reasoning", 0.1, 40.0, 10, 2}};
    const auto weighted = tune::weighted_decode_tps(workloads);
    REQUIRE(weighted);
    CHECK(*weighted == doctest::Approx(80.0));
    auto missing = workloads;
    missing.pop_back();
    CHECK_FALSE(tune::weighted_decode_tps(missing));
    tune::Measurement m = clean({{"q4_0", "q4_0"}, 65536, 512, 0});
    m.workloads = workloads;
    m.short_tps = 1.0;
    auto slower_weighted = clean({{"q8_0", "q4_0"}, 65536, 512, 0});
    slower_weighted.short_tps = 999.0;
    slower_weighted.workloads = {{"prose", 0.4, 70.0, 10, 5}, {"code_edit", 0.3, 70.0, 10, 5},
        {"tool_json", 0.2, 70.0, 10, 5}, {"reasoning", 0.1, 70.0, 10, 5}};
    tune::Report report;
    report.results = {m, slower_weighted};
    tune::rank(report);
    REQUIRE(report.fast_default);
    CHECK(*report.fast_default == 0);
    const auto doc = tune::report_json(tune::Options{}, report);
    const auto *row = doc.find("results")->find("candidates")->as_array().front().find("workloads");
    REQUIRE(row != nullptr);
    CHECK(row->as_array().size() == 4);
    CHECK(doc.find("results")->find("candidates")->as_array().front().find("weighted_tok_s")->as_double() == doctest::Approx(80.0));
    CHECK(row->as_array().front().find("acceptance_ratio")->as_double() == doctest::Approx(0.5));
}

TEST_CASE("tune workload counters distinguish zero acceptance from unavailable ratios") {
    auto measurement = clean({{"q4_0", "q4_0"}, 65536, 512, 2});
    measurement.workloads[0].draft_n_accepted = 0;
    measurement.workloads[1].draft_n = 0;
    measurement.workloads[1].draft_n_accepted = 0;
    measurement.workloads[2].draft_n_accepted.reset();
    measurement.workloads[3].draft_n.reset();
    tune::Report report;
    report.results.push_back(measurement);
    const auto document = tune::report_json({}, report);
    const auto &rows = document.find("results")->find("candidates")->as_array()[0].find("workloads")->as_array();
    CHECK(rows[0].find("acceptance_ratio")->as_double(-1) == 0);
    for (std::size_t i = 1; i < rows.size(); ++i) CHECK(rows[i].find("acceptance_ratio")->is_null());
    CHECK(rows[2].find("draft_n_accepted")->is_null());
    CHECK(rows[3].find("draft_n")->is_null());
}

TEST_CASE("tune rejects missing duplicate nonfinite or incorrectly weighted classes") {
    const auto valid = clean({{"q4_0", "q4_0"}, 65536, 512, 0});
    for (int failure = 0; failure < 6; ++failure) {
        auto measurement = valid;
        if (failure == 0) measurement.workloads.clear();
        if (failure == 1) measurement.workloads.pop_back();
        if (failure == 2) measurement.workloads[1].name = "prose";
        if (failure == 3) measurement.workloads[1].tok_s = std::numeric_limits<double>::infinity();
        if (failure == 4) measurement.workloads[1].tok_s = 0;
        if (failure == 5) measurement.workloads[1].weight = 0.9;
        CHECK_FALSE(tune::weighted_decode_tps(measurement.workloads));
        tune::Report report;
        report.results.push_back(measurement);
        tune::rank(report);
        CHECK_FALSE(report.fast_default);
        CHECK_FALSE(report.long_context);
    }
}
