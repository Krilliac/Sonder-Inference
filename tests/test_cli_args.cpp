// sonder-infer argument parsing, chat message files and the interactive chat
// loop (src/cli/cli_args.hpp), exercised with the MOCK backend.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "cli/cli_args.hpp"
#include "cli/sonder_infer_commands.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "test_helpers.hpp"

using namespace sonder::inference;
namespace cli = sonder::cli;

namespace {

std::optional<cli::Args> parse(std::vector<const char*> argv, std::string& error) {
    argv.insert(argv.begin(), "sonder-infer");
    return cli::parse_args(static_cast<int>(argv.size()), argv.data(), error);
}

}  // namespace

TEST_CASE("cli: parse chat subcommand with values, stops and flags") {
    std::string error;
    auto a = parse({"chat", "--backend", "mock", "--model", "mock:tiny", "--messages", "conv.json", "--system",
                    "be brief", "--stop", "User:", "--stop", "\n\n", "--capture-text", "--max-tokens", "12"},
                   error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->command == "chat");
    CHECK(a->get("backend") == std::optional<std::string>("mock"));
    CHECK(a->get("model") == std::optional<std::string>("mock:tiny"));
    CHECK(a->get("messages") == std::optional<std::string>("conv.json"));
    CHECK(a->get("system") == std::optional<std::string>("be brief"));
    CHECK(a->get("max-tokens") == std::optional<std::string>("12"));
    CHECK_FALSE(a->get("prompt").has_value());
    REQUIRE(a->stops.size() == 2u);
    CHECK(a->stops[0] == "User:");
    CHECK(a->stops[1] == "\n\n");
    CHECK(a->has("capture-text"));
    CHECK_FALSE(a->has("markdown"));
}

TEST_CASE("cli: parse llamaserver configuration options and verbatim repeatable args") {
    std::string error;
    auto a = cli::parse_command_args(cli::generate_command(),
                    {"--backend", "llamaserver", "--llamaserver-config", "server.json",
                     "--llamaserver-mode", "spawn", "--llamaserver-executable", "llama-server.exe",
                     "--llamaserver-arg", "--spec-type", "--llamaserver-arg", "draft-mtp",
                     "--llamaserver-arg", "--spec-draft-n-max", "--llamaserver-arg", "3"}, error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->get("backend") == std::optional<std::string>("llamaserver"));
    CHECK(a->get("llamaserver-config") == std::optional<std::string>("server.json"));
    CHECK(a->get("llamaserver-mode") == std::optional<std::string>("spawn"));
    CHECK(a->all("llamaserver-arg") == std::vector<std::string>{"--spec-type", "draft-mtp", "--spec-draft-n-max", "3"});
}

TEST_CASE("cli: bench --markdown is a boolean flag anywhere in the list") {
    std::string error;
    auto a = parse({"bench", "--markdown", "--backend", "mock", "--out", "r.json"}, error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->has("markdown"));
    CHECK(a->get("backend") == std::optional<std::string>("mock"));
    CHECK(a->get("out") == std::optional<std::string>("r.json"));

    auto b = parse({"bench", "--backend", "mock", "--markdown"}, error);
    REQUIRE_MESSAGE(b.has_value(), error);
    CHECK(b->has("markdown"));
    CHECK(cli::is_boolean_flag("--markdown"));
    CHECK(cli::is_boolean_flag("--capture-text"));
    CHECK_FALSE(cli::is_boolean_flag("--messages"));
}

TEST_CASE("cli: parse errors") {
    std::string error;
    CHECK_FALSE(parse({}, error).has_value());
    CHECK(error == "missing command");
    CHECK_FALSE(parse({"chat", "--messages"}, error).has_value());
    CHECK(error == "missing value for --messages");
    CHECK_FALSE(parse({"chat", "stray"}, error).has_value());
    CHECK(error == "unexpected argument: stray");
    CHECK_FALSE(parse({"chat", "--"}, error).has_value());
    CHECK(error == "unexpected argument: --");
    // The last value wins for repeated non-repeatable options.
    auto a = parse({"chat", "--model", "a", "--model", "b"}, error);
    REQUIRE(a.has_value());
    CHECK(a->get("model") == std::optional<std::string>("b"));
}

TEST_CASE("cli: chat message files") {
    auto obj = cli::parse_chat_messages(
        R"({"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi \"there\""}]})");
    REQUIRE_MESSAGE(obj.ok(), obj.status().to_string());
    REQUIRE(obj->size() == 2u);
    CHECK(obj->at(0).role == "system");
    CHECK(obj->at(1).content == "hi \"there\"");

    auto arr = cli::parse_chat_messages(R"([{"role":"user","content":"one"}])");
    REQUIRE(arr.ok());
    CHECK(arr->size() == 1u);

    CHECK(cli::parse_chat_messages("not json").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"({"msgs":[]})").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"("text")").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"user"}])").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"user","content":7}])").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages("[]").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"bard","content":"x"}])").status().code() ==
          ErrorCode::invalid_argument);
    CHECK(cli::load_chat_messages("definitely/not/here.json").status().code() == ErrorCode::not_found);
}

TEST_CASE("cli: system prompt is prepended only when absent") {
    std::vector<ChatMessage> m{{"user", "hi"}};
    cli::apply_system_prompt(m, "");
    CHECK(m.size() == 1u);
    cli::apply_system_prompt(m, "be kind");
    REQUIRE(m.size() == 2u);
    CHECK(m[0].role == "system");
    CHECK(m[0].content == "be kind");
    cli::apply_system_prompt(m, "other");
    CHECK(m.size() == 2u);
    CHECK(m[0].content == "be kind");
}

TEST_CASE("cli: run_chat_turn streams the mock reply") {
    auto backend = make_mock_backend();
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto model = backend->load_model(lo).value();
    std::ostringstream out;
    auto turn = cli::run_chat_turn(*model, {{"user", "hello"}}, SamplingConfig::greedy(5), {}, out);
    REQUIRE_MESSAGE(turn.ok(), turn.status().to_string());
    CHECK_FALSE(turn->text.empty());
    CHECK(out.str() == turn->text);
    CHECK(turn->stats.completion_tokens == 5u);

    std::ostringstream none;
    CHECK(cli::run_chat_turn(*model, {}, SamplingConfig::greedy(5), {}, none).status().code() ==
          ErrorCode::invalid_argument);
    CHECK(none.str().empty());
}

TEST_CASE("cli: interactive chat loop keeps history and handles commands") {
    std::vector<std::vector<ChatMessage>> seen;
    int reply_no = 0;
    auto turn = [&](const std::vector<ChatMessage>& history) -> Result<std::string> {
        seen.push_back(history);
        if (history.back().content == "fail") return Status(ErrorCode::backend_error, "boom");
        return "reply" + std::to_string(++reply_no);
    };
    std::istringstream in("first\n\n   \nsecond\r\nfail\n/reset\nthird\n/exit\nnever\n");
    std::ostringstream out;
    std::ostringstream err;
    cli::ChatReplOptions ro;
    ro.system = "sys";
    const int rc = cli::run_chat_repl(in, out, err, ro, turn);
    CHECK(rc == 1);  // the "fail" turn
    REQUIRE(seen.size() == 4u);
    // Turn 1: system + first.
    REQUIRE(seen[0].size() == 2u);
    CHECK(seen[0][0].role == "system");
    CHECK(seen[0][1].content == "first");
    // Turn 2 sees the first exchange; CR is stripped.
    REQUIRE(seen[1].size() == 4u);
    CHECK(seen[1][2].role == "assistant");
    CHECK(seen[1][2].content == "reply1");
    CHECK(seen[1][3].content == "second");
    // The failed turn is dropped from history, then /reset clears it.
    CHECK(seen[2].size() == 6u);
    REQUIRE(seen[3].size() == 2u);
    CHECK(seen[3][0].content == "sys");
    CHECK(seen[3][1].content == "third");
    CHECK(err.str().find("boom") != std::string::npos);
    CHECK(out.str().find("[history cleared]") != std::string::npos);
    CHECK(out.str().find("> ") != std::string::npos);
}

TEST_CASE("cli: interactive chat loop with the mock backend ends cleanly at EOF") {
    auto backend = make_mock_backend();
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto model = backend->load_model(lo).value();
    std::istringstream in("hello\nagain\n");
    std::ostringstream out;
    std::ostringstream err;
    cli::ChatReplOptions ro;
    ro.user_prompt.clear();
    const int rc = cli::run_chat_repl(in, out, err, ro, [&](const std::vector<ChatMessage>& h) -> Result<std::string> {
        auto t = cli::run_chat_turn(*model, h, SamplingConfig::greedy(3), {}, out);
        if (!t.ok()) return t.status();
        return t->text;
    });
    CHECK(rc == 0);
    CHECK(err.str().empty());
    // Two replies, each followed by a newline.
    const std::string s = out.str();
    CHECK(std::count(s.begin(), s.end(), '\n') == 2);
}

TEST_CASE("cli: --logit-bias parsing") {
    std::vector<TokenLogitBias> out;
    std::string error;
    REQUIRE(cli::parse_logit_bias("42:-100,7:2.5,3:-inf", out, error));
    REQUIRE(out.size() == 3u);
    CHECK(out[0].token == 42);
    CHECK(out[0].bias == doctest::Approx(-100.0f));
    CHECK(out[1].token == 7);
    CHECK(out[1].bias == doctest::Approx(2.5f));
    CHECK(out[2].token == 3);
    CHECK(std::isinf(out[2].bias));
    CHECK(out[2].bias < 0.0f);
    for (const char* bad : {"", "42", ":1", "42:", "x:1", "4x:1", "1:y", "1:2,", "1:2,,3:4"}) {
        CAPTURE(bad);
        CHECK_FALSE(cli::parse_logit_bias(bad, out, error));
        CHECK_FALSE(error.empty());
    }
}

TEST_CASE("cli: parse devices subcommand with optional --json flag") {
    std::string error;
    const auto plain = cli::parse_command_args(cli::devices_command(), {}, error);
    REQUIRE_MESSAGE(plain.has_value(), error);
    CHECK(plain->command == "devices");
    CHECK_FALSE(plain->has("json"));
    CHECK_FALSE(plain->help);

    const auto as_json = cli::parse_command_args(cli::devices_command(), {"--json"}, error);
    REQUIRE_MESSAGE(as_json.has_value(), error);
    CHECK(as_json->command == "devices");
    CHECK(as_json->has("json"));
    CHECK(cli::is_boolean_flag("--json"));

    const auto legacy = parse({"devices", "--json"}, error);
    REQUIRE_MESSAGE(legacy.has_value(), error);
    CHECK(legacy->command == "devices");
    CHECK(legacy->has("json"));

    const auto help = cli::parse_command_args(cli::devices_command(), {"--help"}, error);
    REQUIRE_MESSAGE(help.has_value(), error);
    CHECK(help->help);

    CHECK_FALSE(cli::parse_command_args(cli::devices_command(), {"--unknown-flag"}, error).has_value());
    CHECK_FALSE(error.empty());
    error.clear();
    CHECK_FALSE(cli::parse_command_args(cli::devices_command(), {"gpu:0"}, error).has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("cli: devices inventory helpers cover discrete, UMA and unknown memory readings") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // Backwards-compatible CPU device inventory always includes cpu:0.
    const auto devices = enumerate_devices();
    REQUIRE_FALSE(devices.empty());
    CHECK(devices.front().id == "cpu:0");
    CHECK(devices.front().kind == DeviceKind::cpu);

    // Live GPU enumeration succeeds cleanly on both GPU hosts and headless 0-GPU CI.
    const auto live_gpus = enumerate_gpu_devices();
    for (const auto& g : live_gpus) {
        CHECK_FALSE(g.id.empty());
        CHECK(std::string(to_string(g.vendor)).size() > 0u);
        CHECK_FALSE(g.source.empty());
    }

    // Verify MiB + effective VRAM contract across discrete NVIDIA/AMD, UMA APU, and unknown readings.
    GpuDeviceInfo rtx;
    rtx.id = "gpu:0";
    rtx.name = "NVIDIA GeForce RTX 5070 Ti";
    rtx.vendor = GpuVendor::nvidia;
    rtx.vendor_id = 0x10DEu;
    rtx.device_id = 0x2C05u;
    rtx.dedicated_total = MemoryReading::known(16303ull * kMiB);
    rtx.dedicated_free = MemoryReading::known(14800ull * kMiB);
    rtx.shared_total = MemoryReading::known(16ull * kGiB);
    rtx.is_integrated_uma = false;
    rtx.backend_hint = "cuda";
    rtx.source = "dxgi";
    CHECK(std::string(to_string(rtx.vendor)) == "nvidia");
    CHECK(gpu_effective_vram_mib(rtx, false) == std::optional<std::uint64_t>(16303ull));
    CHECK(gpu_effective_vram_mib(rtx, true) == std::optional<std::uint64_t>(14800ull));

    GpuDeviceInfo rx_xt;
    rx_xt.id = "gpu:1";
    rx_xt.name = "AMD Radeon RX 7900 XT";
    rx_xt.vendor = GpuVendor::amd;
    rx_xt.vendor_id = 0x1002u;
    rx_xt.device_id = 0x744Cu;
    rx_xt.dedicated_total = MemoryReading::known(20ull * kGiB);
    rx_xt.dedicated_free = MemoryReading::known(18ull * kGiB);
    rx_xt.shared_total = MemoryReading::known(16ull * kGiB);
    rx_xt.is_integrated_uma = false;
    rx_xt.backend_hint = "rocm";
    rx_xt.source = "amdgpu-sysfs";
    CHECK(std::string(to_string(rx_xt.vendor)) == "amd");
    CHECK(gpu_effective_vram_mib(rx_xt, false) == std::optional<std::uint64_t>(20480ull));
    CHECK(gpu_effective_vram_mib(rx_xt, true) == std::optional<std::uint64_t>(18432ull));

    GpuDeviceInfo apu;
    apu.id = "gpu:2";
    apu.name = "AMD Radeon 780M Graphics";
    apu.vendor = GpuVendor::amd;
    apu.vendor_id = 0x1002u;
    apu.device_id = 0x15BFu;
    apu.dedicated_total = MemoryReading::known(512ull * kMiB);
    apu.dedicated_free = MemoryReading::known(384ull * kMiB + 14ull * kGiB);
    apu.shared_total = MemoryReading::known(16ull * kGiB);
    apu.is_integrated_uma = true;
    apu.backend_hint = "rocm";
    apu.source = "amdgpu-sysfs";
    CHECK(gpu_effective_vram_mib(apu, false) == std::optional<std::uint64_t>(16896ull));
    CHECK(gpu_effective_vram_mib(apu, true) == std::optional<std::uint64_t>(14720ull));

    GpuDeviceInfo unknown_gpu;
    unknown_gpu.id = "gpu:3";
    unknown_gpu.name = "Unknown GPU";
    unknown_gpu.vendor = GpuVendor::unknown;
    unknown_gpu.dedicated_total = MemoryReading::unknown();
    unknown_gpu.dedicated_free = MemoryReading::unknown();
    unknown_gpu.shared_total = MemoryReading::unknown();
    CHECK(std::string(to_string(unknown_gpu.vendor)) == "unknown");
    CHECK_FALSE(gpu_effective_vram_mib(unknown_gpu, false).has_value());
    CHECK_FALSE(gpu_effective_vram_mib(unknown_gpu, true).has_value());
    CHECK(json::Value(gpu_effective_vram_mib(unknown_gpu, false)).is_null());
    CHECK(json::Value(gpu_effective_vram_mib(unknown_gpu, true)).is_null());
}
