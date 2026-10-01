#include "../tune_runner.hpp"
#include "../tune_model.hpp"
#include "../log_diagnostics.hpp"
#include "../process_options.hpp"
#include "fake_process.hpp"
#include "fake_gpu_memory.hpp"
#include <doctest/doctest.h>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;
namespace tune = sonder::inference::llamaserver::tune;
namespace {
struct Fixture {
    std::shared_ptr<LaunchState> state = std::make_shared<LaunchState>();
    tune::Options options;
    tune::Dependencies dependencies;
    tune::Candidate candidate{{"q4_0", "q4_0"}, 73728, 1024, 0};
    std::filesystem::path log_path = std::filesystem::temp_directory_path() /
        ("sonder-tune-runner-" + std::to_string(tune::Clock::now().time_since_epoch().count()) + ".log");
    std::vector<std::size_t> prompt_sizes;
    std::vector<std::uint64_t> decode_sizes;
    std::vector<json::Value> chat_bodies;
    std::atomic<bool> spill{false};
    bool spill_during_prefill = false;
    bool retain_memory = false;
    bool release_error = false;
    bool wrong_pid = false;
    bool missing_timings = false;
    bool wrong_props = false;
    bool slow_kernel = false;
    bool ignored_mtp = false;
    bool truncated_workload = false;
    bool missing_acceptance = false;
    bool empty_chat = false;
    bool cached_chat = false;
    bool zero_chat_tokens = false;
    bool overlong_chat = false;
    bool invalid_finish = false;
    bool chat_error = false;
    bool unexpected_drafts = false;
    Fixture() {
        state->next_pid = 1000;
        options.executable = "fake-llama-server";
        options.model = "fake-model.gguf";
        options.environment = {{"GGML_TEST_OVERRIDE", "hello world"}};
        dependencies.launcher = [this] { return std::make_unique<FakeLauncher>(state); };
        dependencies.release_timeout = std::chrono::milliseconds(250);
        dependencies.sample_interval = std::chrono::milliseconds(10);
        dependencies.counters = std::make_shared<FakeCounters>([this](std::uint32_t pid) -> Result<GpuProcessCounters> {
            std::lock_guard lock(state->mutex);
            if (state->children.empty()) return GpuProcessCounters{};
            const bool alive = state->children.back()->alive.load();
            if (!alive && release_error) return Status(ErrorCode::io_error, "release PDH read failed");
            const bool held = alive || retain_memory;
            const auto instance = "pid_" + std::to_string(wrong_pid ? pid + 1 : pid) + "_luid_test";
            return GpuProcessCounters{{{instance, held ? 14000 * kMiB : 0}},
                                      {{instance, held ? (spill.load() ? 388 : 182) * kMiB : 0}}};
        });
        dependencies.exchange = [this](std::uint16_t, const std::string &, const std::string &path,
                                      const json::Value &body, tune::Deadline, const CancellationToken &cancel) -> Result<json::Value> {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "fake request cancelled");
            if (path == "/health") return json::Value(json::Object{{"status", "ok"}});
            if (path == "/props") {
                std::lock_guard lock(state->mutex);
                const auto ctx = find_context_size(state->specs.back().arguments).value_or(0);
                return json::Value(json::Object{{"default_generation_settings", json::Object{{"n_ctx", wrong_props ? ctx / 2 : ctx}}},
                                               {"total_slots", 1}});
            }
            if (path == "/tokenize") return json::Value(json::Object{{"tokens", json::Array{12, 25, 39}}});
            if (path == "/v1/chat/completions") {
                chat_bodies.push_back(body);
                if (chat_error) return Status(ErrorCode::backend_error, "chat failed");
                const std::uint64_t predicted = zero_chat_tokens ? 0 :
                    (overlong_chat ? body.find("max_tokens")->as_uint() + 1 : 100);
                json::Object timing{{"prompt_n", 400}, {"cache_n", cached_chat ? 100 : 0},
                    {"predicted_n", predicted}, {"prompt_ms", 100},
                    {"predicted_ms", 1000 * static_cast<int>(chat_bodies.size())},
                    {"draft_n", (candidate.mtp != 0 && !ignored_mtp) || unexpected_drafts ? 128 : 0}};
                if (!missing_acceptance) timing.set("draft_n_accepted", candidate.mtp != 0 && !ignored_mtp ? 96 : 0);
                json::Object response{{"choices", json::Array{json::Object{
                    {"message", json::Object{{"role", "assistant"}, {"content", empty_chat ? "" : "Mock natural answer."}}},
                    {"finish_reason", invalid_finish ? "tool_calls" : "stop"}}}}};
                if (!missing_timings) response.set("timings", std::move(timing));
                return json::Value(std::move(response));
            }
            if (path != "/completion") return Status(ErrorCode::not_found, "unexpected fake endpoint");
            const auto prompt = body.find("prompt")->as_array().size();
            const auto predicted = body.find("n_predict")->as_uint();
            // These assertions run on the test thread (never the monitor).
            CHECK_FALSE(body.find("cache_prompt")->as_bool());
            CHECK(body.find("ignore_eos")->as_bool());
            CHECK(body.find("id_slot")->as_uint() == 0);
            prompt_sizes.push_back(prompt);
            decode_sizes.push_back(predicted);
            if (spill_during_prefill && prompt == 8192) spill.store(true);
            if (slow_kernel) {
                std::ofstream log(log_path, std::ios::binary | std::ios::app);
                log << "no FlashAttention vector kernel compiled for K/V types q8_0-q4_0, "
                       "converting K and V to f16 instead (slow)"; // no trailing newline
            }
            if (missing_timings) return json::Value(json::Object{});
            return json::Value(json::Object{{"timings", json::Object{
                {"prompt_n", static_cast<std::uint64_t>(prompt)}, {"cache_n", 0},
                {"predicted_n", truncated_workload ? 1 : predicted}, {"prompt_ms", 8000}, {"predicted_ms", 1000},
                {"draft_n", candidate.mtp != 0 && !ignored_mtp ? 128 : 0}}}});
        };
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove(log_path, ec); }
    tune::Measurement run(tune::Phase phase = tune::Phase::benchmark) {
        return tune::run_candidate(options, candidate, phase, tune::Clock::now() + std::chrono::seconds(3),
                                   log_path.string(), dependencies);
    }
};
} // namespace

TEST_CASE("tune decode replaces the repeated sentence with four natural chat requests") {
    Fixture f;
    auto exchange = f.dependencies.exchange;
    std::string tokenized_text;
    unsigned chats = 0;
    f.dependencies.exchange = [&](std::uint16_t port, const std::string &method, const std::string &path,
                                  const json::Value &body, tune::Deadline deadline,
                                  const CancellationToken &cancel) -> Result<json::Value> {
        if (path == "/tokenize") tokenized_text = body.find("content")->as_string();
        if (path == "/v1/chat/completions") ++chats;
        return exchange(port, method, path, body, deadline, cancel);
    };
    const auto result = f.run();
    REQUIRE_MESSAGE(result.verdict == tune::Verdict::clean, result.detail);
    CHECK(tokenized_text.find("The sun rises over a quiet river.") == std::string::npos);
    CHECK(chats == 4);
}

TEST_CASE("tune runner uses supervisor fake launcher exact workloads and confirmed release") {
    Fixture f;
    f.options.thorough = true;
    const auto result = f.run();
    REQUIRE_MESSAGE(result.verdict == tune::Verdict::clean, result.detail);
    CHECK(result.memory_released);
    CHECK(result.served_ctx == 73728);
    CHECK(result.shared_bytes == 182 * kMiB);
    CHECK(result.dedicated_bytes == 14000 * kMiB);
    REQUIRE(result.short_tps);
    CHECK(*result.short_tps == doctest::Approx(0.4 * 100 + 0.3 * 50 + 0.2 * (100.0 / 3) + 0.1 * 25));
    CHECK(result.prefill_tps == 1024);
    CHECK(result.long_prefill_tps.has_value());
    CHECK(f.prompt_sizes == std::vector<std::size_t>{8192, 72704});
    CHECK(f.decode_sizes == std::vector<std::uint64_t>{1, 1});
    CHECK(f.chat_bodies.size() == 4);
    REQUIRE(f.state->specs.size() == 1);
    CHECK(f.state->specs[0].environment == f.options.environment);
    CHECK(f.state->specs[0].output_file == f.log_path.string());
    CHECK(f.state->specs[0].port != 0);
    CHECK_FALSE(f.state->children.front()->alive.load());
    CHECK(f.state->children.front()->stops.load() >= 1);
}
TEST_CASE("tune cheap probe skips decode and thorough workload") {
    Fixture f;
    f.options.thorough = true;
    const auto result = f.run(tune::Phase::probe);
    REQUIRE(result.verdict == tune::Verdict::clean);
    CHECK_FALSE(result.short_tps);
    CHECK_FALSE(result.long_prefill_tps);
    CHECK(f.chat_bodies.empty());
    CHECK(f.prompt_sizes == std::vector<std::size_t>{8192});
    CHECK(f.decode_sizes == std::vector<std::uint64_t>{1});
}
TEST_CASE("tune rejects measured load and workload spill above the independent clean line") {
    for (const bool during_prefill : {false, true}) {
        Fixture f;
        f.spill.store(!during_prefill);
        f.spill_during_prefill = during_prefill;
        const auto result = f.run();
        CHECK(result.verdict == tune::Verdict::spill);
        CHECK(result.memory_released);
        CHECK(result.shared_bytes == 388 * kMiB);
    }
    Fixture baseline;
    baseline.options.grid.spill.baseline_bytes = 150 * kMiB;
    baseline.spill.store(true); // 388 < 150 + 256
    CHECK(baseline.run().verdict == tune::Verdict::clean);
}
TEST_CASE("tune records slow kernels instead of selecting them even without newline") {
    Fixture f;
    f.slow_kernel = true;
    const auto result = f.run();
    CHECK(result.verdict == tune::Verdict::slow_kernel);
    CHECK(result.slow_kernel);
    CHECK(result.memory_released);
}
TEST_CASE("tune stops when memory release cannot be proved") {
    for (const bool error : {false, true}) {
        Fixture f;
        f.retain_memory = !error;
        f.release_error = error;
        const auto result = f.run();
        CHECK(result.verdict == tune::Verdict::release_failed);
        CHECK_FALSE(result.memory_released);
        CHECK(result.stop_search);
        CHECK_FALSE(f.state->children.front()->alive.load());
    }
}
TEST_CASE("tune unsupported probes cannot produce a recommendation or start a child") {
    Fixture f;
    f.dependencies.counters = std::make_shared<FakeCounters>([](std::uint32_t) -> Result<GpuProcessCounters> {
        return Status(ErrorCode::unsupported, "no probe");
    }, false);
    const auto result = f.run();
    CHECK(result.verdict == tune::Verdict::unsupported);
    CHECK(result.stop_search);
    CHECK(f.state->specs.empty());
    Fixture empty;
    empty.wrong_pid = true;
    CHECK(empty.run().verdict == tune::Verdict::unsupported);
}
TEST_CASE("tune validates served context actual decode length timings and draft activity") {
    for (int failure = 0; failure != 4; ++failure) {
        Fixture f;
        f.wrong_props = failure == 0;
        f.missing_timings = failure == 1;
        f.zero_chat_tokens = failure == 2;
        if (failure == 3) { f.candidate.mtp = 2; f.ignored_mtp = true; }
        const auto result = f.run();
        CHECK(result.verdict == tune::Verdict::error);
        CHECK(result.memory_released);
    }
    Fixture mtp;
    mtp.candidate.mtp = 2;
    CHECK(mtp.run().verdict == tune::Verdict::clean);
}
TEST_CASE("tune natural chat requests use templates and allow natural EOS") {
    Fixture f;
    const auto result = f.run();
    REQUIRE_MESSAGE(result.verdict == tune::Verdict::clean, result.detail);
    REQUIRE(f.chat_bodies.size() == 4);
    for (const auto &body : f.chat_bodies) {
        REQUIRE(body.find("messages"));
        REQUIRE(body.find("messages")->as_array().size() == 2);
        CHECK(body.find("messages")->as_array()[0].find("role")->as_string() == "system");
        CHECK(body.find("messages")->as_array()[1].find("role")->as_string() == "user");
        CHECK_FALSE(body.find("chat_template_kwargs")->find("enable_thinking")->as_bool(true));
        CHECK_FALSE(body.find("ignore_eos"));
        CHECK_FALSE(body.find("id_slot"));
        CHECK_FALSE(body.find("cache_prompt")->as_bool());
        CHECK_FALSE(body.find("stream")->as_bool());
        CHECK(body.find("temperature")->as_double() == 0);
        CHECK(body.find("seed")->as_uint() == 42);
    }
    const auto prompt = [&](std::size_t i) -> const std::string & {
        return f.chat_bodies[i].find("messages")->as_array()[1].find("content")->as_string();
    };
    CHECK(prompt(0).find("200-token essay") != std::string::npos);
    CHECK(prompt(1).find("rename") != std::string::npos);
    CHECK(prompt(1).find("calculate_total") != std::string::npos);
    const auto begin = prompt(1).find("```python\n");
    const auto end = prompt(1).rfind("```");
    REQUIRE(begin != std::string::npos);
    REQUIRE(end > begin);
    const auto file = prompt(1).substr(begin + 10, end - begin - 10);
    CHECK(std::count(file.begin(), file.end(), '\n') == 120);
    CHECK(prompt(2).find("file_read") != std::string::npos);
    CHECK(prompt(2).find("JSON") != std::string::npos);
    CHECK(prompt(3).find("kilometres") != std::string::npos);
    REQUIRE(result.workloads.size() == 4);
    CHECK(result.workloads[0].name == "prose");
    CHECK(result.workloads[1].name == "code_edit");
    CHECK(result.workloads[2].name == "tool_json");
    CHECK(result.workloads[3].name == "reasoning");
}
TEST_CASE("tune retains measured acceptance for each natural workload") {
    Fixture f;
    f.candidate.mtp = 2;
    const auto result = f.run();
    REQUIRE_MESSAGE(result.verdict == tune::Verdict::clean, result.detail);
    REQUIRE(result.workloads.size() == 4);
    for (const auto &row : result.workloads) {
        CHECK(row.draft_n == 128);
        CHECK(row.draft_n_accepted == 96);
        CHECK(row.tok_s > 0);
    }
}
TEST_CASE("tune rejects unusable natural responses without recommending partial classes") {
    for (int failure = 0; failure != 8; ++failure) {
        Fixture f;
        f.candidate.mtp = failure == 0 ? 2U : 0U;
        f.missing_acceptance = failure == 0;
        f.empty_chat = failure == 1;
        f.cached_chat = failure == 2;
        f.zero_chat_tokens = failure == 3;
        f.overlong_chat = failure == 4;
        f.invalid_finish = failure == 5;
        f.chat_error = failure == 6;
        f.unexpected_drafts = failure == 7;
        const auto result = f.run();
        CHECK(result.verdict == tune::Verdict::error);
        CHECK_FALSE(result.short_tps);
        CHECK(result.memory_released);
    }
}
TEST_CASE("tune preserves partial class evidence but has no weighted score after a failed class") {
    Fixture f;
    const auto exchange = f.dependencies.exchange;
    unsigned chats = 0;
    f.dependencies.exchange = [&](std::uint16_t port, const std::string &method, const std::string &path,
                                  const json::Value &body, tune::Deadline deadline,
                                  const CancellationToken &cancel) -> Result<json::Value> {
        if (path == "/v1/chat/completions" && ++chats == 3)
            return Status(ErrorCode::backend_error, "third class failed");
        return exchange(port, method, path, body, deadline, cancel);
    };
    const auto result = f.run();
    CHECK(result.verdict == tune::Verdict::error);
    CHECK(result.workloads.size() == 2);
    CHECK_FALSE(result.short_tps);
    CHECK(result.detail.find("tool_json: third class failed") != std::string::npos);
    CHECK(result.memory_released);
}
TEST_CASE("tune honours interruption between natural classes and releases the child") {
    Fixture f;
    std::atomic<bool> interrupted{false};
    f.dependencies.interrupted = [&] { return interrupted.load(); };
    const auto exchange = f.dependencies.exchange;
    f.dependencies.exchange = [&](std::uint16_t port, const std::string &method, const std::string &path,
                                  const json::Value &body, tune::Deadline deadline,
                                  const CancellationToken &cancel) -> Result<json::Value> {
        auto response = exchange(port, method, path, body, deadline, cancel);
        if (path == "/v1/chat/completions") interrupted.store(true);
        return response;
    };
    const auto result = f.run();
    CHECK(result.verdict == tune::Verdict::cancelled);
    CHECK(result.workloads.size() == 1);
    CHECK(f.chat_bodies.size() == 1);
    CHECK_FALSE(result.short_tps);
    CHECK(result.memory_released);
}
TEST_CASE("tune deadline and cancellation before launch are side effect free") {
    Fixture f;
    auto result = tune::run_candidate(f.options, f.candidate, tune::Phase::probe, tune::Clock::now(),
                                      f.log_path.string(), f.dependencies);
    CHECK(result.verdict == tune::Verdict::timed_out);
    CHECK(f.state->specs.empty());
    f.dependencies.interrupted = [] { return true; };
    CHECK(f.run().verdict == tune::Verdict::cancelled);
    CHECK(f.state->specs.empty());
}
TEST_CASE("tune readiness timeout still stops the child and verifies release") {
    Fixture f;
    f.dependencies.exchange = [](std::uint16_t, const std::string &, const std::string &, const json::Value &,
                                  tune::Deadline, const CancellationToken &) -> Result<json::Value> {
        return Status(ErrorCode::unavailable, "not ready");
    };
    const auto result = tune::run_candidate(f.options, f.candidate, tune::Phase::probe,
        tune::Clock::now() + std::chrono::milliseconds(80), f.log_path.string(), f.dependencies);
    CHECK(result.verdict == tune::Verdict::timed_out);
    CHECK(result.memory_released);
    REQUIRE(f.state->children.size() == 1);
    CHECK_FALSE(f.state->children.front()->alive.load());
}
TEST_CASE("tune retains peak spill even if the final sample recovers") {
    Fixture f;
    auto underlying = f.dependencies.counters;
    std::atomic<bool> first{true};
    f.dependencies.counters = std::make_shared<FakeCounters>([&](std::uint32_t pid) -> Result<GpuProcessCounters> {
        auto raw = underlying->read(pid);
        if (raw.ok() && !raw.value().dedicated.empty() && raw.value().dedicated.front().value != 0 && first.exchange(false))
            raw.value().shared.front().value = 388 * kMiB;
        return raw;
    });
    const auto result = f.run();
    CHECK(result.verdict == tune::Verdict::spill);
    CHECK(result.shared_bytes == 388 * kMiB);
    CHECK(result.memory_released);
}
TEST_CASE("tune help detection reuses launcher capture and child environment") {
    Fixture f;
    class HelpLauncher final : public ProcessLauncher {
      public:
        explicit HelpLauncher(std::shared_ptr<LaunchState> state) : state_(std::move(state)) {}
        Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
            FakeLauncher launcher(state_);
            auto process = launcher.start(spec);
            { std::ofstream file(spec.output_file, std::ios::binary);
              file << "--spec-type TYPE (none, draft-mtp)\n--spec-draft-n-max N\n"; }
            child_at(state_, 0)->alive.store(false);
            return process;
        }
      private:
        std::shared_ptr<LaunchState> state_;
    };
    f.dependencies.launcher = [&] { return std::make_unique<HelpLauncher>(f.state); };
    auto help = tune::executable_help(f.options, tune::Clock::now() + std::chrono::seconds(1), f.log_path.string(), f.dependencies);
    REQUIRE(help.ok());
    CHECK(tune::help_supports_mtp(help.value()));
    REQUIRE(f.state->specs.size() == 1);
    CHECK(f.state->specs.front().arguments == std::vector<std::string>{"--help"});
    CHECK(f.state->specs.front().environment == f.options.environment);
    CHECK(f.state->children.front()->stops.load() >= 1);
}
TEST_CASE("process optional environment and capture preserve the no-option launch") {
    ProcessSpec spec;
    ProcessOptions unchanged;
    REQUIRE(unchanged.prepare(spec).ok());
    CHECK(unchanged.environment.empty());
    CHECK_FALSE(unchanged.redirect);
    CHECK(validate_process_environment({{"TUNE_TEST", "has spaces=and equals"}}).ok());
    CHECK_FALSE(validate_process_environment({{"", "x"}}).ok());
    CHECK_FALSE(validate_process_environment({{"A=B", "x"}}).ok());
    CHECK_FALSE(validate_process_environment({{"A", std::string("a\0b", 3)}}).ok());
    Fixture fixture;
    spec.environment = {{"SONDER_TUNE_PROCESS_TEST", "first"}, {"SONDER_TUNE_PROCESS_TEST", "last value"}};
    spec.output_file = fixture.log_path.string();
    ProcessOptions overrides;
    REQUIRE(overrides.prepare(spec).ok());
    CHECK(overrides.redirect);
#if defined(_WIN32)
    const std::wstring environment(overrides.environment.begin(), overrides.environment.end());
    CHECK(environment.find(L"SONDER_TUNE_PROCESS_TEST=last value") != std::wstring::npos);
    CHECK(environment.find(L"SONDER_TUNE_PROCESS_TEST=first") == std::wstring::npos);
    CHECK(overrides.startup.lpAttributeList != nullptr);
#else
    CHECK(std::find(overrides.environment.begin(), overrides.environment.end(), "SONDER_TUNE_PROCESS_TEST=last value") != overrides.environment.end());
    CHECK(overrides.envp.back() == nullptr);
#endif
}
