#include "../tune.hpp"
#include "../tune_model.hpp"
#include "../adaptive_speculation.hpp"
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
            R"({"ctx":[9223372036854775808]})", R"({"mtp":[1,2]})", R"({"mtp":[0,17]})",
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
    CHECK(out.str().find("| q4_0/q4_0 | on | 65536 | 512 | 0 | none | - |") != std::string::npos);
    CHECK(err.str().empty());
    out.str("");
    CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf", "--dry-run",
        "--grid", R"({"kv":[["f16","f16"]],"flash_attn":["on","auto"],"ctx":[65536],"ubatch":[512],"mtp":[0,2],"spec_type":["ngram-mod"]})"}, out, err) == 0);
    CHECK(out.str().find("| f16/f16 | auto | 65536 | 512 | 2 | ngram-mod | - |") != std::string::npos);
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

    // --spec-type support is detected independently of draft-mtp support (e.g. ngram-mod-only builds).
    CHECK(tune::help_supports_spec_type("--spec-type TYPE\n    possible: none, ngram-mod\n"));
    CHECK(tune::help_supports_spec_type("  -s, --spec-type=TYPE\n"));
    CHECK_FALSE(tune::help_supports_spec_type("--spec-typex TYPE\n"));
    CHECK_FALSE(tune::help_supports_spec_type("--other-flag\n"));
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

TEST_CASE("tune supports standalone ngram-mod speculation without MTP tensors") {
    auto parsed = json::parse(
        R"({"kv":[["q4_0","q4_0"]],"ctx":[65536,69632],"ubatch":[512],"mtp":[0,2],"spec_type":["ngram-mod"],"p_min":[0.0,0.5]})");
    REQUIRE(parsed.ok());
    auto grid_res = tune::parse_grid(parsed.value());
    REQUIRE_MESSAGE(grid_res.ok(), grid_res.status().message());
    const auto &grid = grid_res.value();

    // Non-MTP models (mtp_available == false) plan baseline + standalone ngram-mod
    // without multiplying ngram-mod by p_min.
    const auto planned = tune::planned_candidates(grid, false);
    REQUIRE(planned.size() == 4); // 2 ctx * (1 baseline + 1 standalone ngram-mod)
    CHECK(std::count_if(planned.begin(), planned.end(), [](const auto &c) { return c.mtp == 0; }) == 2);
    CHECK(std::count_if(planned.begin(), planned.end(), [](const auto &c) {
        return c.mtp == 2 && c.spec_type == "ngram-mod";
    }) == 2);

    // Mixed draft-mtp + ngram-mod skips draft-mtp when mtp_available is false,
    // and only multiplies draft-mtp variants by p_min when mtp_available is true.
    tune::Grid mixed = grid;
    mixed.ctx = {65536};
    mixed.spec_type = {"draft-mtp", "draft-mtp,ngram-mod", "ngram-mod"};
    REQUIRE(tune::validate_grid(mixed).ok());
    const auto gated = tune::planned_candidates(mixed, false);
    REQUIRE(gated.size() == 2); // 1 baseline + 1 standalone ngram-mod
    for (const auto &c : gated) {
        if (c.mtp != 0) CHECK(c.spec_type == "ngram-mod");
    }
    const auto all_with_mtp = tune::planned_candidates(mixed, true);
    CHECK(all_with_mtp.size() == 6); // 1 baseline + 2 draft-mtp + 2 hybrid + 1 standalone ngram-mod

    // uses_mtp, spill_guard_for, and Candidate::operator== distinguish baseline,
    // standalone ngram-mod, and MTP cleanly.
    const tune::Candidate base_a{{"q4_0", "q4_0"}, 65536, 512, 0, "draft-mtp", 0.0};
    const tune::Candidate base_b{{"q4_0", "q4_0"}, 65536, 512, 0, "ngram-mod", 0.5};
    const tune::Candidate ngram_a{{"q4_0", "q4_0"}, 65536, 512, 2, "ngram-mod", 0.0};
    const tune::Candidate ngram_b{{"q4_0", "q4_0"}, 65536, 512, 2, "ngram-mod", 0.5};
    const tune::Candidate mtp_a{{"q4_0", "q4_0"}, 65536, 512, 2, "draft-mtp", 0.0};
    const tune::Candidate mtp_b{{"q4_0", "q4_0"}, 65536, 512, 2, "draft-mtp", 0.5};
    const tune::Candidate hybrid{{"q4_0", "q4_0"}, 65536, 512, 2, "draft-mtp,ngram-mod", 0.0};
    CHECK_FALSE(tune::uses_mtp(base_a));
    CHECK_FALSE(tune::uses_mtp(ngram_a));
    CHECK(tune::uses_mtp(mtp_a));
    CHECK(tune::uses_mtp(hybrid));
    CHECK(&tune::spill_guard_for(grid, ngram_a) == &grid.spill);
    CHECK(&tune::spill_guard_for(grid, mtp_a) == &grid.spill_mtp);
    CHECK(base_a == base_b);
    CHECK(ngram_a == ngram_b);
    CHECK_FALSE(mtp_a == mtp_b);

    // Standalone ngram-mod emits --spec-type ngram-mod --spec-ngram-mod-n-max N
    // and never emits --spec-draft-n-max or --spec-draft-p-min.
    tune::Options options;
    options.model = "ternary-bonsai-2-27b.gguf";
    options.grid = grid;
    const auto argv = tune::arguments(options, ngram_b);
    const auto spec_it = std::find(argv.begin(), argv.end(), "--spec-type");
    REQUIRE(spec_it != argv.end());
    REQUIRE(std::next(spec_it) != argv.end());
    CHECK(*std::next(spec_it) == "ngram-mod");
    const auto ngram_n_it = std::find(argv.begin(), argv.end(), "--spec-ngram-mod-n-max");
    REQUIRE(ngram_n_it != argv.end());
    REQUIRE(std::next(ngram_n_it) != argv.end());
    CHECK(*std::next(ngram_n_it) == "2");
    CHECK(std::find(argv.begin(), argv.end(), "--spec-draft-n-max") == argv.end());
    CHECK(std::find(argv.begin(), argv.end(), "--spec-draft-p-min") == argv.end());

    auto now = tune::Clock::now();
    unsigned ngram_benchmarks = 0;
    const auto report = tune::search(
        options, false, now + std::chrono::minutes(2),
        [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline) {
            if (phase == tune::Phase::benchmark && c.mtp == 2 && c.spec_type == "ngram-mod") ++ngram_benchmarks;
            auto m = clean(c, phase);
            m.short_tps = 40.0 + (c.mtp == 2 && c.spec_type == "ngram-mod" ? 25.0 : 0.0);
            for (auto &work : m.workloads) work.tok_s = *m.short_tps;
            return m;
        },
        [&] { return now; });
    CHECK(ngram_benchmarks == 1);
    REQUIRE(report.fast_default);
    const auto &winner = report.results[*report.fast_default];
    CHECK(winner.candidate.mtp == 2);
    CHECK(winner.candidate.spec_type == "ngram-mod");
    const auto doc = tune::report_json(options, report);
    const auto &rows = doc.find("results")->find("candidates")->as_array();
    CHECK(rows[*report.fast_default].find("p_min")->is_null());
}

TEST_CASE("tune sweeps flash_attn and skips quantized V cache when flash_attn is off") {
    auto parsed = json::parse(
        R"({"kv":[["f16","f16"],["q8_0","f16"],["q4_0","q4_0"]],"flash_attn":["on","off","auto"],"ctx":[65536,69632],"ubatch":[512],"mtp":[0]})");
    REQUIRE(parsed.ok());
    auto grid_res = tune::parse_grid(parsed.value());
    REQUIRE_MESSAGE(grid_res.ok(), grid_res.status().message());
    const auto &grid = grid_res.value();

    // f16/f16 (3 FA modes) + q8_0/f16 (3 FA modes) + q4_0/q4_0 (2 FA modes: on, auto) = 8 valid lanes * 2 ctx = 16.
    const auto planned = tune::planned_candidates(grid, false);
    CHECK(planned.size() == 16);
    CHECK(std::none_of(planned.begin(), planned.end(), [](const auto &c) {
        return c.flash_attn == "off" && c.kv.v == "q4_0";
    }));
    CHECK(std::any_of(planned.begin(), planned.end(), [](const auto &c) {
        return c.flash_attn == "off" && c.kv == tune::KvPair{"q8_0", "f16"};
    }));

    tune::Options options;
    options.model = "model.gguf";
    options.grid = grid;
    for (const std::string fa : {"on", "off", "auto"}) {
        const auto argv = tune::arguments(options, {{"f16", "f16"}, 65536, 512, 0, "draft-mtp", 0.0, fa});
        CHECK(read_kv_cache_config(argv).flash_attn == fa);
    }

    auto now = tune::Clock::now();
    bool saw_invalid = false;
    bool saw_off = false;
    bool saw_auto = false;
    const auto report = tune::search(
        options, false, now + std::chrono::minutes(2),
        [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline) {
            if (c.flash_attn == "off" && c.kv.v == "q4_0") saw_invalid = true;
            if (c.flash_attn == "off") saw_off = true;
            if (c.flash_attn == "auto") saw_auto = true;
            auto m = clean(c, phase);
            m.short_tps = c.flash_attn == "auto" && c.kv == tune::KvPair{"q4_0", "q4_0"} ? 90.0 : 50.0;
            for (auto &work : m.workloads) work.tok_s = *m.short_tps;
            return m;
        },
        [&] { return now; });
    CHECK_FALSE(saw_invalid);
    CHECK(saw_off);
    CHECK(saw_auto);
    REQUIRE(report.fast_default);
    CHECK(report.results[*report.fast_default].candidate.flash_attn == "auto");
    const auto doc = tune::report_json(options, report);
    const auto &rows = doc.find("results")->find("candidates")->as_array();
    REQUIRE_FALSE(rows.empty());
    CHECK(rows.front().find("flash_attn") != nullptr);

    // Tie-breaker in rank(): when decode/prefill/ctx/mtp tie, flash_attn prefers on > auto > off.
    {
        tune::Report tied;
        tied.results = {
            clean({{"f16", "f16"}, 65536, 512, 0, "draft-mtp", 0.0, "off"}),
            clean({{"f16", "f16"}, 65536, 512, 0, "draft-mtp", 0.0, "auto"}),
            clean({{"f16", "f16"}, 65536, 512, 0, "draft-mtp", 0.0, "on"}),
        };
        tune::rank(tied);
        REQUIRE(tied.fast_default);
        CHECK(*tied.fast_default == 2);
        tied.results.pop_back(); // only off vs auto remain
        tune::rank(tied);
        REQUIRE(tied.fast_default);
        CHECK(*tied.fast_default == 1);
    }

    // Invalid flash_attn grids must be rejected.
    for (const auto *bad_json : {R"({"flash_attn":[]})", R"({"flash_attn":["maybe"]})",
                                  R"({"flash_attn":["on","on"]})",
                                  R"({"kv":[["q4_0","q4_0"]],"flash_attn":["off"]})"}) {
        CAPTURE(bad_json);
        const auto bad = json::parse(bad_json);
        REQUIRE(bad.ok());
        CHECK_FALSE(tune::parse_grid(bad.value()).ok());
    }
}

TEST_CASE("tune accepts 8192 context floor for multi-model presets while enforcing alignment") {
    auto ok_json = json::parse(R"({"kv":[["q4_0","q4_0"]],"ctx":[8192,12288,16384,20480,24576],"ubatch":[512],"mtp":[0]})");
    REQUIRE(ok_json.ok());
    auto grid_res = tune::parse_grid(ok_json.value());
    REQUIRE_MESSAGE(grid_res.ok(), grid_res.status().message());

    tune::Options options;
    options.grid = grid_res.value();
    auto now = tune::Clock::now();
    const auto report = tune::search(
        options, false, now + std::chrono::minutes(1),
        [&](const tune::Candidate &c, tune::Phase phase, tune::Deadline) { return clean(c, phase); },
        [&] { return now; });
    CHECK_FALSE(report.fast_default); // below 64k fast threshold
    REQUIRE(report.long_context);
    CHECK(report.results[*report.long_context].candidate.ctx == 20480);

    for (const auto *bad_json : {R"({"ctx":[4096]})", R"({"ctx":[10000]})"}) {
        CAPTURE(bad_json);
        const auto bad = json::parse(bad_json);
        REQUIRE(bad.ok());
        CHECK_FALSE(tune::parse_grid(bad.value()).ok());
    }
}

TEST_CASE("tune GGUF scanner detects nextn_predict_layers metadata and prism.hadamard arrays") {
    const auto make_meta_gguf = [](bool nextn_before_arch, const std::string &nextn_key, std::uint32_t nextn_type,
                                   std::uint64_t nextn_val, bool with_prism) {
        std::string out = "GGUF";
        const std::uint64_t kv_count = (nextn_key.empty() ? 1ull : 2ull) + (with_prism ? 2ull : 0ull);
        integer(out, 3, 4);
        integer(out, 1, 8);        // 1 tensor
        integer(out, kv_count, 8); // kv count
        const auto emit_arch = [&] {
            string(out, "general.architecture");
            integer(out, 8, 4);
            string(out, "qwen35");
        };
        const auto emit_nextn = [&] {
            if (nextn_key.empty()) return;
            string(out, nextn_key);
            integer(out, nextn_type, 4);
            integer(out, nextn_val, nextn_type == 10 || nextn_type == 11 ? 8u : 4u);
        };
        if (nextn_before_arch) {
            emit_nextn();
            emit_arch();
        } else {
            emit_arch();
            emit_nextn();
        }
        if (with_prism) {
            string(out, "prism.hadamard.block_size");
            integer(out, 4, 4); // uint32
            integer(out, 128, 4);
            // Large int8 sign_values array (28,672 bytes) must fast-skip safely.
            string(out, "prism.hadamard.sign_values");
            integer(out, 9, 4); // array
            integer(out, 1, 4); // int8
            integer(out, 28672, 8);
            out.append(28672, '\x01');
        }
        string(out, "blk.0.attn_q.weight");
        integer(out, 2, 4);
        integer(out, 32, 8);
        integer(out, 32, 8);
        integer(out, 0, 4);
        integer(out, 0, 8);
        return out;
    };

    // Positive <arch>.nextn_predict_layers works both after and before general.architecture.
    for (const bool before : {false, true}) {
        std::istringstream in(make_meta_gguf(before, "qwen35.nextn_predict_layers", 4, 1, false));
        const auto info = tune::read_model_info(in);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK(info.value().has_nextn);
        CHECK_FALSE(info.value().prism_hadamard);
    }

    // Zero, negative signed int32, and wrong-architecture keys do not enable MTP.
    {
        std::istringstream in(make_meta_gguf(false, "qwen35.nextn_predict_layers", 4, 0, false));
        const auto info = tune::read_model_info(in);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK_FALSE(info.value().has_nextn);
    }
    {
        std::istringstream in(make_meta_gguf(false, "qwen35.nextn_predict_layers", 5, 0xffffffffu, false));
        const auto info = tune::read_model_info(in);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK_FALSE(info.value().has_nextn);
    }
    {
        std::istringstream in(make_meta_gguf(false, "other.nextn_predict_layers", 4, 2, false));
        const auto info = tune::read_model_info(in);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK_FALSE(info.value().has_nextn);
    }

    // PrismML Hadamard metadata + large sign_values array sets prism_hadamard cleanly.
    {
        std::istringstream in(make_meta_gguf(false, "", 4, 0, true));
        const auto info = tune::read_model_info(in);
        REQUIRE_MESSAGE(info.ok(), info.status().message());
        CHECK(info.value().prism_hadamard);
        CHECK_FALSE(info.value().has_nextn);
    }
}

TEST_CASE("tune accepts expanded KV types extended mtp range and dflash or draft spec types") {
    auto parsed = json::parse(
        R"({"kv":[["f8_e4m3","f8_e4m3"],["q5_0","q5_0"],["q2_0","q2_0"],["pq2_0m","pq2_0m"]],)"
        R"("ctx":[8192,16384],"ubatch":[512],"mtp":[0,4,8,16],)"
        R"("spec_type":["draft-mtp","draft-mtp,ngram-mod","ngram-mod","dflash","draft-dflash","draft","draft-simple"],"p_min":[0.0,0.5]})");
    REQUIRE(parsed.ok());
    auto grid_res = tune::parse_grid(parsed.value());
    REQUIRE_MESSAGE(grid_res.ok(), grid_res.status().message());
    const auto &grid = grid_res.value();
    CHECK(grid.kv.size() == 4);
    CHECK(grid.ctx == std::vector<std::uint64_t>{8192, 16384});
    CHECK(grid.mtp == std::vector<std::uint32_t>{0, 4, 8, 16});
    CHECK(grid.spec_type.size() == 7);

    // Non-MTP models still plan standalone ngram-mod, dflash/draft-dflash, and draft/draft-simple without requiring nextn tensors.
    const auto planned_no_mtp = tune::planned_candidates(grid, false);
    CHECK(std::any_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp == 8 && c.spec_type == "dflash" && c.p_min == 0.5;
    }));
    CHECK(std::any_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp == 8 && c.spec_type == "draft-dflash" && c.p_min == 0.5;
    }));
    CHECK(std::any_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp == 16 && c.spec_type == "draft" && c.p_min == 0.0;
    }));
    CHECK(std::any_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp == 16 && c.spec_type == "draft-simple" && c.p_min == 0.0;
    }));
    CHECK(std::any_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp == 8 && c.spec_type == "ngram-mod";
    }));
    CHECK(std::none_of(planned_no_mtp.begin(), planned_no_mtp.end(), [](const auto &c) {
        return c.mtp != 0 && (c.spec_type == "draft-mtp" || c.spec_type == "draft-mtp,ngram-mod");
    }));

    // Quantized / FP8 V caches still reject flash_attn=off alone; out-of-range mtp/spec_type rejected.
    for (const auto *bad_json : {
             R"({"kv":[["f8_e4m3","f8_e4m3"]],"flash_attn":["off"]})",
             R"({"kv":[["q2_0","q2_0"]],"flash_attn":["off"]})",
             R"({"kv":[["pq2_0m","pq2_0m"]],"flash_attn":["off"]})",
             R"({"kv":[["q5_0","q5_0"]],"flash_attn":["off"]})",
             R"({"mtp":[0,17]})",
             R"({"spec_type":["unknown-spec"]})",
         }) {
        CAPTURE(bad_json);
        const auto bad = json::parse(bad_json);
        REQUIRE(bad.ok());
        CHECK_FALSE(tune::parse_grid(bad.value()).ok());
    }

    const tune::Candidate dflash_cand{{"q2_0", "q2_0"}, 8192, 512, 8, "dflash", 0.5, "on"};
    const tune::Candidate draft_dflash_cand{{"q2_0", "q2_0"}, 8192, 512, 8, "draft-dflash", 0.5, "on"};
    const tune::Candidate draft_cand{{"f8_e4m3", "f8_e4m3"}, 16384, 512, 8, "draft", 0.25, "on"};
    const tune::Candidate draft_simple_cand{{"f8_e4m3", "f8_e4m3"}, 16384, 512, 8, "draft-simple", 0.25, "on"};
    const tune::Candidate ngram_cand{{"pq2_0m", "pq2_0m"}, 8192, 512, 8, "ngram-mod", 0.0, "on"};
    CHECK_FALSE(tune::uses_mtp(dflash_cand));
    CHECK_FALSE(tune::uses_mtp(draft_dflash_cand));
    CHECK_FALSE(tune::uses_mtp(draft_cand));
    CHECK_FALSE(tune::uses_mtp(draft_simple_cand));
    CHECK(tune::uses_draft_p_min(dflash_cand));
    CHECK(tune::uses_draft_p_min(draft_dflash_cand));
    CHECK(tune::uses_draft_p_min(draft_cand));
    CHECK(tune::uses_draft_p_min(draft_simple_cand));
    CHECK_FALSE(tune::uses_draft_p_min(ngram_cand));
    CHECK(&tune::spill_guard_for(grid, dflash_cand) == &grid.spill);
    CHECK(&tune::spill_guard_for(grid, draft_dflash_cand) == &grid.spill);
    CHECK(&tune::spill_guard_for(grid, draft_cand) == &grid.spill);
    CHECK(&tune::spill_guard_for(grid, draft_simple_cand) == &grid.spill);

    tune::Options options;
    options.model = "bonsai.gguf";
    options.grid = grid;
    const auto flag_val = [](const std::vector<std::string> &argv, const std::string &flag) -> std::string {
        const auto it = std::find(argv.begin(), argv.end(), flag);
        if (it != argv.end() && std::next(it) != argv.end()) return *std::next(it);
        return {};
    };
    const auto dflash_argv = tune::arguments(options, dflash_cand);
    CHECK(flag_val(dflash_argv, "--spec-type") == "dflash");
    CHECK(flag_val(dflash_argv, "--spec-draft-n-max") == "8");
    CHECK(flag_val(dflash_argv, "--spec-draft-p-min") == "0.500000");
    const auto draft_dflash_argv = tune::arguments(options, draft_dflash_cand);
    CHECK(flag_val(draft_dflash_argv, "--spec-type") == "draft-dflash");
    CHECK(flag_val(draft_dflash_argv, "--spec-draft-n-max") == "8");
    CHECK(flag_val(draft_dflash_argv, "--spec-draft-p-min") == "0.500000");
    const auto draft_simple_argv = tune::arguments(options, draft_simple_cand);
    CHECK(flag_val(draft_simple_argv, "--spec-type") == "draft-simple");
    CHECK(flag_val(draft_simple_argv, "--spec-draft-n-max") == "8");
    CHECK(flag_val(draft_simple_argv, "--spec-draft-p-min") == "0.250000");
    const auto ngram_argv = tune::arguments(options, ngram_cand);
    CHECK(flag_val(ngram_argv, "--spec-type") == "ngram-mod");
    CHECK(flag_val(ngram_argv, "--spec-ngram-mod-n-max") == "8");
}

TEST_CASE("tune grid_for_gpu selects vendor VRAM UMA and ternary presets") {
    constexpr std::uint64_t kGiB = 1024ull * kMiB;
    const auto has_kv = [](const tune::Grid &g, const std::string &k, const std::string &v) {
        return std::find(g.kv.begin(), g.kv.end(), tune::KvPair{k, v}) != g.kv.end();
    };
    const auto has_spec = [](const tune::Grid &g, const std::string &spec) {
        return std::find(g.spec_type.begin(), g.spec_type.end(), spec) != g.spec_type.end();
    };

    // 1. NVIDIA RTX 5070 Ti 16 GiB (Blackwell/Ada discrete)
    GpuDeviceInfo rtx;
    rtx.id = "gpu:0";
    rtx.name = "NVIDIA GeForce RTX 5070 Ti";
    rtx.vendor = GpuVendor::nvidia;
    rtx.dedicated_total = MemoryReading::known(16ull * kGiB);
    rtx.is_integrated_uma = false;
    const auto rtx_grid = tune::grid_for_gpu(rtx, false);
    REQUIRE_MESSAGE(tune::validate_grid(rtx_grid).ok(), tune::validate_grid(rtx_grid).message());
    CHECK(has_kv(rtx_grid, "f8_e4m3", "f8_e4m3"));
    CHECK(has_kv(rtx_grid, "q8_0", "q8_0"));
    CHECK(has_kv(rtx_grid, "q8_0", "q4_0"));
    CHECK(has_kv(rtx_grid, "q4_0", "q4_0"));
    CHECK_FALSE(has_kv(rtx_grid, "q2_0", "q2_0"));
    CHECK(rtx_grid.ctx.front() == 32768);
    CHECK(rtx_grid.ctx.back() == 139264);
    CHECK_FALSE(rtx_grid.spill.uma_mode);

    const auto rtx_ternary = tune::grid_for_gpu(rtx, true);
    REQUIRE_MESSAGE(tune::validate_grid(rtx_ternary).ok(), tune::validate_grid(rtx_ternary).message());
    CHECK(has_kv(rtx_ternary, "f8_e4m3", "f8_e4m3"));
    CHECK(has_kv(rtx_ternary, "q2_0", "q2_0"));
    CHECK(rtx_ternary.kv.size() <= 8);
    CHECK(has_spec(rtx_ternary, "ngram-mod"));

    // 2. AMD RX 9070 XT 16 GiB discrete (no unsupported FP8 KV)
    GpuDeviceInfo rx_xt;
    rx_xt.id = "gpu:0";
    rx_xt.name = "AMD Radeon RX 9070 XT";
    rx_xt.vendor = GpuVendor::amd;
    rx_xt.dedicated_total = MemoryReading::known(16ull * kGiB);
    rx_xt.is_integrated_uma = false;
    const auto xt_grid = tune::grid_for_gpu(rx_xt, false);
    REQUIRE_MESSAGE(tune::validate_grid(xt_grid).ok(), tune::validate_grid(xt_grid).message());
    CHECK_FALSE(has_kv(xt_grid, "f8_e4m3", "f8_e4m3"));
    CHECK(has_kv(xt_grid, "q8_0", "q8_0"));
    CHECK(has_kv(xt_grid, "q8_0", "q4_0"));
    CHECK(has_kv(xt_grid, "q4_0", "q4_0"));
    CHECK(has_kv(xt_grid, "f16", "f16"));
    CHECK(xt_grid.ctx.front() == 32768);
    CHECK_FALSE(xt_grid.spill.uma_mode);

    const auto xt_ternary = tune::grid_for_gpu(rx_xt, true);
    REQUIRE_MESSAGE(tune::validate_grid(xt_ternary).ok(), tune::validate_grid(xt_ternary).message());
    CHECK_FALSE(has_kv(xt_ternary, "f8_e4m3", "f8_e4m3"));
    CHECK(has_kv(xt_ternary, "q2_0", "q2_0"));
    CHECK(has_spec(xt_ternary, "ngram-mod"));

    // 3. AMD Ryzen APU (Unified Memory / UMA)
    GpuDeviceInfo apu;
    apu.id = "gpu:0";
    apu.name = "AMD Radeon 780M Graphics";
    apu.vendor = GpuVendor::amd;
    apu.dedicated_total = MemoryReading::known(512ull * kMiB);
    apu.shared_total = MemoryReading::known(16ull * kGiB);
    apu.is_integrated_uma = true;
    const auto apu_grid = tune::grid_for_gpu(apu, false);
    REQUIRE_MESSAGE(tune::validate_grid(apu_grid).ok(), tune::validate_grid(apu_grid).message());
    CHECK(apu_grid.ctx.front() == 8192);
    CHECK(apu_grid.ctx.back() == 65536);
    CHECK(apu_grid.ctx.size() == 15);
    CHECK(apu_grid.ubatch == std::vector<std::uint32_t>{256, 512, 1024});
    CHECK(apu_grid.kv == std::vector<tune::KvPair>{{"q8_0", "q8_0"}, {"q4_0", "q4_0"}});
    CHECK(apu_grid.spill.uma_mode);
    CHECK(apu_grid.spill_mtp.uma_mode);

    const auto apu_ternary = tune::grid_for_gpu(apu, true);
    REQUIRE_MESSAGE(tune::validate_grid(apu_ternary).ok(), tune::validate_grid(apu_ternary).message());
    CHECK(has_kv(apu_ternary, "q2_0", "q2_0"));
    CHECK(has_spec(apu_ternary, "ngram-mod"));
    CHECK(apu_ternary.spill.uma_mode);

    // 4. Low-VRAM 8 GiB discrete GPUs (RTX 4060 & RX 7600)
    GpuDeviceInfo rtx_8g;
    rtx_8g.id = "gpu:0";
    rtx_8g.name = "NVIDIA GeForce RTX 4060";
    rtx_8g.vendor = GpuVendor::nvidia;
    rtx_8g.dedicated_total = MemoryReading::known(8ull * kGiB);
    rtx_8g.is_integrated_uma = false;
    const auto rtx_8g_grid = tune::grid_for_gpu(rtx_8g, false);
    REQUIRE_MESSAGE(tune::validate_grid(rtx_8g_grid).ok(), tune::validate_grid(rtx_8g_grid).message());
    CHECK(rtx_8g_grid.ctx.front() == 8192);
    CHECK(rtx_8g_grid.ctx.back() == 65536);
    CHECK(has_kv(rtx_8g_grid, "f8_e4m3", "f8_e4m3"));
    CHECK_FALSE(rtx_8g_grid.spill.uma_mode);

    GpuDeviceInfo rx_8g;
    rx_8g.id = "gpu:0";
    rx_8g.name = "AMD Radeon RX 7600";
    rx_8g.vendor = GpuVendor::amd;
    rx_8g.dedicated_total = MemoryReading::known(8ull * kGiB);
    rx_8g.is_integrated_uma = false;
    const auto rx_8g_grid = tune::grid_for_gpu(rx_8g, false);
    REQUIRE_MESSAGE(tune::validate_grid(rx_8g_grid).ok(), tune::validate_grid(rx_8g_grid).message());
    CHECK(rx_8g_grid.ctx.front() == 8192);
    CHECK(rx_8g_grid.ctx.back() == 65536);
    CHECK_FALSE(has_kv(rx_8g_grid, "f8_e4m3", "f8_e4m3"));

    // 5. UMA mode parsing, resolution, and host-RAM pool ceiling helpers
    CHECK(tune::parse_uma_mode("off") == tune::UmaMode::off);
    CHECK(tune::parse_uma_mode("on") == tune::UmaMode::on);
    CHECK(tune::parse_uma_mode("auto") == tune::UmaMode::auto_detect);
    CHECK_FALSE(tune::parse_uma_mode("bogus").has_value());
    CHECK(std::string(tune::to_string(tune::UmaMode::off)) == "off");
    CHECK(std::string(tune::to_string(tune::UmaMode::on)) == "on");
    CHECK(std::string(tune::to_string(tune::UmaMode::auto_detect)) == "auto");

    CHECK(tune::resolve_tune_uma_mode(tune::UmaMode::on, {rtx}));
    CHECK_FALSE(tune::resolve_tune_uma_mode(tune::UmaMode::off, {apu}));
    CHECK(tune::resolve_tune_uma_mode(tune::UmaMode::auto_detect, {apu}));
    CHECK_FALSE(tune::resolve_tune_uma_mode(tune::UmaMode::auto_detect, {rtx, apu}));
    CHECK_FALSE(tune::resolve_tune_uma_mode(tune::UmaMode::auto_detect, {}));
    GpuDeviceInfo apu_second = apu;
    apu_second.id = "gpu:1";
    CHECK(tune::resolve_tune_uma_mode(tune::UmaMode::auto_detect, {rtx, apu_second}, {"--device", "gpu:1"}));

    HostMemory host;
    host.total_physical = MemoryReading::known(32ull * kGiB);
    host.available_physical = MemoryReading::known(24ull * kGiB);
    const auto apu_cap = tune::uma_pool_ceiling_bytes(host, {apu});
    REQUIRE(apu_cap.has_value());
    // GPU pool (512 MiB + 16 GiB = 16.5 GiB) is tighter than host ceiling (24 GiB - 2 GiB = 22 GiB).
    CHECK(*apu_cap == 512ull * kMiB + 16ull * kGiB);

    HostMemory tight_host;
    tight_host.total_physical = MemoryReading::known(16ull * kGiB);
    tight_host.available_physical = MemoryReading::known(10ull * kGiB);
    const auto tight_cap = tune::uma_pool_ceiling_bytes(tight_host, {apu});
    REQUIRE(tight_cap.has_value());
    // Host ceiling (10 GiB - 2 GiB = 8 GiB) caps the 16.5 GiB GPU pool.
    CHECK(*tight_cap == 8ull * kGiB);

    HostMemory zero_host;
    zero_host.available_physical = MemoryReading::known(0);
    const auto zero_cap = tune::uma_pool_ceiling_bytes(zero_host, {apu});
    REQUIRE(zero_cap.has_value());
    CHECK(*zero_cap == 0);

    // Zero-sized UMA GPU pools (0 dedicated + 0 shared) are skipped rather than clamping host ceiling to 0.
    GpuDeviceInfo empty_uma = apu;
    empty_uma.dedicated_total = MemoryReading::known(0);
    empty_uma.shared_total = MemoryReading::known(0);
    const auto skipped_zero_gpu = tune::uma_pool_ceiling_bytes(host, {empty_uma});
    REQUIRE(skipped_zero_gpu.has_value());
    CHECK(*skipped_zero_gpu == 22ull * kGiB);
    CHECK_FALSE(tune::uma_pool_ceiling_bytes(HostMemory{}, {empty_uma}).has_value());

    CHECK_FALSE(tune::uma_pool_ceiling_bytes(HostMemory{}, {}).has_value());
}

TEST_CASE("tune spawn_config emits uma_mode and adaptive_speculation only when configured") {
    tune::Options options;
    options.executable = "C:/bin/llama-server.exe";
    options.model = "C:/models/bonsai.gguf";

    // Default options omit both spill_guard.uma_mode and adaptive_speculation.
    {
        tune::Report report;
        report.results = {clean({{"q4_0", "q4_0"}, 73728, 1024, 2})};
        tune::rank(report);
        const auto doc = tune::report_json(options, report);
        const auto *guard = doc.find("spill_guard");
        REQUIRE(guard != nullptr);
        CHECK(guard->find("uma_mode") == nullptr);
        CHECK(doc.find("adaptive_speculation") == nullptr);
    }

    // UmaMode::on emits boolean true; UmaMode::auto_detect emits string "auto".
    for (const auto mode : {tune::UmaMode::on, tune::UmaMode::auto_detect}) {
        tune::Options uma_opts = options;
        uma_opts.uma_mode = mode;
        tune::Report report;
        report.results = {clean({{"q4_0", "q4_0"}, 65536, 512, 0})};
        tune::rank(report);
        const auto doc = tune::report_json(uma_opts, report);
        const auto *guard = doc.find("spill_guard");
        REQUIRE(guard != nullptr);
        const auto *uma_val = guard->find("uma_mode");
        REQUIRE(uma_val != nullptr);
        if (mode == tune::UmaMode::on) {
            CHECK(uma_val->is_bool());
            CHECK(uma_val->as_bool());
        } else {
            CHECK(uma_val->is_string());
            CHECK(uma_val->as_string() == "auto");
        }
#if defined(SONDER_HAS_SERVER)
        const std::string tag = mode == tune::UmaMode::on ? "-on" : "-auto";
        const auto path = std::filesystem::temp_directory_path() /
            ("sonder-tune-uma-" + std::to_string(tune::Clock::now().time_since_epoch().count()) + tag + ".json");
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p, ec); } } cleanup{path};
        { std::ofstream file(path, std::ios::binary); file << doc.dump(); }
        BackendSetup setup;
        auto loaded = load_llamaserver_config(path.string(), setup);
        REQUIRE_MESSAGE(loaded.ok(), loaded.message());
        CHECK(setup.llamaserver_spill_uma_mode == (mode == tune::UmaMode::on ? "on" : "auto"));
#endif
    }

    // C++ callers setting options.grid = grid_for_gpu(apu) emit spill_guard.uma_mode = true
    // even when options.uma_mode remains at its default UmaMode::off.
    {
        GpuDeviceInfo apu;
        apu.id = "gpu:0";
        apu.name = "AMD Radeon 890M Graphics";
        apu.vendor = GpuVendor::amd;
        apu.dedicated_total = MemoryReading::known(512ull * kMiB);
        apu.shared_total = MemoryReading::known(16384ull * kMiB);
        apu.is_integrated_uma = true;

        tune::Options apu_opts = options;
        apu_opts.grid = tune::grid_for_gpu(apu, false);
        tune::Report report;
        report.results = {clean({{"q4_0", "q4_0"}, 65536, 512, 0})};
        tune::rank(report);
        const auto doc = tune::report_json(apu_opts, report);
        const auto *guard = doc.find("spill_guard");
        REQUIRE(guard != nullptr);
        const auto *uma_val = guard->find("uma_mode");
        REQUIRE(uma_val != nullptr);
        CHECK(uma_val->is_bool());
        CHECK(uma_val->as_bool());
    }

    // emit_adaptive_speculation emits valid schema only when candidate.mtp > 0.
    tune::Options spec_opts = options;
    spec_opts.emit_adaptive_speculation = true;
    {
        tune::Report report;
        report.results = {clean({{"q4_0", "q4_0"}, 65536, 512, 0})};
        tune::rank(report);
        const auto doc = tune::report_json(spec_opts, report);
        CHECK(doc.find("adaptive_speculation") == nullptr);
    }
    for (const std::uint32_t mtp : {1u, 2u, 8u}) {
        CAPTURE(mtp);
        tune::Report report;
        report.results = {clean({{"q4_0", "q4_0"}, 65536, 512, mtp, mtp == 8u ? "dflash" : "draft-mtp", 0.5})};
        tune::rank(report);
        const auto doc = tune::report_json(spec_opts, report);
        const auto *spec = doc.find("adaptive_speculation");
        REQUIRE(spec != nullptr);
        REQUIRE(spec->is_object());
        CHECK(spec->as_object().size() == 5);
        CHECK(spec->find("enabled")->as_bool());
        CHECK(spec->find("min_draft_n")->as_uint() == 0);
        CHECK(spec->find("initial_draft_n")->as_uint() == mtp);
        CHECK(spec->find("max_draft_n")->as_uint() == std::max<std::uint32_t>(mtp, 4u));
        CHECK(spec->find("probe_draft_n")->as_uint() == std::min<std::uint32_t>(mtp, 2u));

        AdaptiveSpeculationConfig cfg;
        cfg.enabled = spec->find("enabled")->as_bool();
        cfg.min_draft_n = static_cast<std::uint32_t>(spec->find("min_draft_n")->as_uint());
        cfg.initial_draft_n = static_cast<std::uint32_t>(spec->find("initial_draft_n")->as_uint());
        cfg.max_draft_n = static_cast<std::uint32_t>(spec->find("max_draft_n")->as_uint());
        cfg.probe_draft_n = static_cast<std::uint32_t>(spec->find("probe_draft_n")->as_uint());
        CHECK(validate_adaptive_speculation_config(cfg).ok());
    }
}

TEST_CASE("tune CLI accepts auto-gpu ternary uma-mode and adaptive-speculation flags") {
    std::ostringstream out, err;
    CHECK(tune::tune_main({"--help"}, out, err) == 0);
    CHECK(out.str().find("--auto-gpu") != std::string::npos);
    CHECK(out.str().find("--ternary") != std::string::npos);
    CHECK(out.str().find("--uma-mode") != std::string::npos);
    CHECK(out.str().find("--adaptive-speculation") != std::string::npos);

    out.str("");
    err.str("");
    CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf",
                           "--ternary", "--dry-run"}, out, err) == 0);
    CHECK(out.str().find("q2_0/q2_0") != std::string::npos);
    CHECK(out.str().find("ngram-mod") != std::string::npos);
    CHECK(err.str().empty());

    out.str("");
    err.str("");
    CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf",
                           "--auto-gpu", "--uma-mode", "auto", "--adaptive-speculation", "--dry-run"}, out, err) == 0);
    CHECK(err.str().empty());

    for (const auto *mode : {"on", "off"}) {
        out.str("");
        err.str("");
        CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf",
                               "--uma-mode", mode, "--dry-run"}, out, err) == 0);
        CHECK(err.str().empty());
    }

    out.str("");
    err.str("");
    CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf",
                           "--uma-mode", "invalid", "--dry-run"}, out, err) == 2);

    // Combining --grid with --auto-gpu or --ternary is rejected with a usage error.
    for (const auto *conflict_flag : {"--auto-gpu", "--ternary"}) {
        out.str("");
        err.str("");
        CHECK(tune::tune_main({"--executable", "does-not-exist.exe", "--model", "absent.gguf",
                               "--grid", R"({"kv":[["q4_0","q4_0"]],"ctx":[65536],"ubatch":[512],"mtp":[0]})",
                               conflict_flag, "--dry-run"}, out, err) == 2);
        CHECK(err.str().find("--grid cannot be combined") != std::string::npos);
    }
}
