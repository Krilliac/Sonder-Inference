// sonder-infer option specs, did-you-mean suggestions, checked values,
// environment defaults and stats lines (src/cli/cli_spec.hpp,
// cli_values.hpp, cli_env.hpp, sonder_infer_commands.hpp).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "cli/cli_args.hpp"
#include "cli/cli_env.hpp"
#include "cli/cli_spec.hpp"
#include "cli/cli_values.hpp"
#include "cli/sonder_infer_commands.hpp"
#include "net/http_client.hpp"
#include "sonder/inference.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif

using namespace sonder::inference;
namespace net = sonder::inference::net;
namespace cli = sonder::cli;

namespace {

const cli::CommandSpec& demo_spec() {
    static const cli::CommandSpec spec{"demo",
                                       "demo command",
                                       {"demo --name X"},
                                       {{"Group",
                                         {{"name", cli::OptionKind::value, "X", "a value"},
                                          {"max-tokens", cli::OptionKind::value, "N", "a number"},
                                          {"tag", cli::OptionKind::repeat, "T", "repeatable"},
                                          {"json", cli::OptionKind::flag, "", "a flag\nsecond line"}}}},
                                       {"A note."},
                                       {"demo --name x"}};
    return spec;
}

bool parse_demo(std::vector<std::string> args, cli::ParsedCommand& out, std::string& error) {
    return cli::parse_command(demo_spec(), args, out, error);
}

cli::EnvLookup env_of(std::map<std::string, std::string> vars) {
    return [vars](const char* name) -> std::optional<std::string> {
        auto it = vars.find(name);
        if (it == vars.end()) return std::nullopt;
        return it->second;
    };
}

}  // namespace

TEST_CASE("cli spec: edit distance and suggestions") {
    CHECK(cli::edit_distance("", "") == 0u);
    CHECK(cli::edit_distance("abc", "") == 3u);
    CHECK(cli::edit_distance("max-token", "max-tokens") == 1u);
    CHECK(cli::edit_distance("tpo-k", "top-k") == 1u);  // transposition
    CHECK(cli::edit_distance("kitten", "sitting") == 3u);

    const std::vector<std::string_view> names{"max-tokens", "top-p", "top-k", "temperature", "prompt"};
    CHECK(cli::suggest("max-token", names) == std::optional<std::string>("max-tokens"));
    CHECK(cli::suggest("temprature", names) == std::optional<std::string>("temperature"));
    CHECK(cli::suggest("promt", names) == std::optional<std::string>("prompt"));
    CHECK(cli::suggest("top-q", names) == std::optional<std::string>("top-p"));  // tie: first candidate
    CHECK_FALSE(cli::suggest("bogus", names).has_value());
    CHECK_FALSE(cli::suggest("x", {"y"}).has_value());  // never closer than the word's own length
    CHECK_FALSE(cli::suggest("seed", names).has_value());

    CHECK(cli::unknown_option_message("max-token", names) == "unknown option --max-token (did you mean --max-tokens?)");
    CHECK(cli::unknown_option_message("bogus", names) == "unknown option --bogus");
}

#if defined(SONDER_HAS_SERVER)
TEST_CASE("llamaserver config: strict spawn schema preserves argv and rejects malformed values") {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::current_path() /
                      ("sonder-llamaserver-cli-test-" + std::to_string(unique) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    const auto write = [&](const std::string& body) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << body;
    };
    sonder::inference::BackendSetup setup;
    write(R"({"mode":"spawn","executable":"llama-server.exe","args":["--model","m.gguf","--spec-type","draft-mtp","--spec-draft-n-max","3"],"startup_timeout_ms":1234,"allow_remote":false})");
    auto status = sonder::inference::load_llamaserver_config(path.string(), setup);
    REQUIRE_MESSAGE(status.ok(), status.to_string());
    CHECK(setup.llamaserver_mode == "spawn");
    CHECK(setup.llamaserver_executable == "llama-server.exe");
    CHECK(setup.llamaserver_args == std::vector<std::string>{"--model", "m.gguf", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"});
    CHECK(setup.llamaserver_startup_timeout_ms == 1234);
    write(R"({"base_url":"http://127.0.0.1:8080"})");
    sonder::inference::BackendSetup attach;
    REQUIRE(sonder::inference::load_llamaserver_config(path.string(), attach).ok());
    CHECK(attach.llamaserver_mode.empty());  // empty means the backend default: attach
    for (const auto& body : {R"({"bogus":1})", R"({"allow_remote":"false"})", R"({"startup_timeout_ms":-1})",
                             R"({"startup_timeout_ms":0})", R"({"mode":"bogus"})", R"({"mode":""})",
                             R"({"args":[1]})", R"({"max_restarts":1001})", R"({"restart_backoff_ms":20,"max_restart_backoff_ms":10})",
                             R"({"tls":{"handshake_timeout_ms":0}})", R"({"executable":"bad\u0000path"})"}) {
        write(body);
        sonder::inference::BackendSetup invalid;
        CHECK_FALSE(sonder::inference::load_llamaserver_config(path.string(), invalid).ok());
    }
}
#endif

TEST_CASE("cli spec: parse_command accepts values, inline values, repeats, flags and help") {
    cli::ParsedCommand p;
    std::string error;
    REQUIRE_MESSAGE(parse_demo({"--name", "a", "--tag", "x", "--json", "--tag=y", "--max-tokens=12"}, p, error),
                    error);
    CHECK(p.get("name") == std::optional<std::string>("a"));
    CHECK(p.get("max-tokens") == std::optional<std::string>("12"));
    CHECK(p.all("tag") == std::vector<std::string>{"x", "y"});
    CHECK(p.has("json"));
    CHECK_FALSE(p.help);
    CHECK(p.all("missing").empty());

    REQUIRE(parse_demo({"--name", "a", "-h"}, p, error));
    CHECK(p.help);
    REQUIRE(parse_demo({"--help"}, p, error));
    CHECK(p.help);
    // An inline value may be empty or contain '='.
    REQUIRE(parse_demo({"--name=k=v"}, p, error));
    CHECK(p.get("name") == std::optional<std::string>("k=v"));
    // A value may start with dashes when it is the separate next argument.
    REQUIRE(parse_demo({"--name", "--json"}, p, error));
    CHECK(p.get("name") == std::optional<std::string>("--json"));
    CHECK_FALSE(p.has("json"));
}

TEST_CASE("cli spec: parse_command errors") {
    cli::ParsedCommand p;
    std::string error;
    CHECK_FALSE(parse_demo({"--max-token", "5"}, p, error));
    CHECK(error == "unknown option --max-token (did you mean --max-tokens?)");
    CHECK_FALSE(parse_demo({"--bogus", "1"}, p, error));
    CHECK(error == "unknown option --bogus");
    CHECK_FALSE(parse_demo({"stray"}, p, error));
    CHECK(error == "unexpected argument 'stray'");
    CHECK_FALSE(parse_demo({"--"}, p, error));
    CHECK(error == "unexpected argument '--'");
    CHECK_FALSE(parse_demo({"-x"}, p, error));
    CHECK(error == "unexpected argument '-x'");
    CHECK_FALSE(parse_demo({"--name"}, p, error));
    CHECK(error == "missing value for --name");
    CHECK_FALSE(parse_demo({"--json=1"}, p, error));
    CHECK(error == "--json takes no value");
    CHECK_FALSE(parse_demo({"--name", "a", "--name", "b"}, p, error));
    CHECK(error == "--name given more than once");
}

TEST_CASE("cli spec: parse_command_args maps --stop and other repeatables") {
    static const cli::CommandSpec spec{"x",
                                       "x",
                                       {},
                                       {{"G",
                                         {{"stop", cli::OptionKind::repeat, "T", ""},
                                          {"tag", cli::OptionKind::repeat, "T", ""},
                                          {"quiet", cli::OptionKind::flag, "", ""}}}},
                                       {},
                                       {}};
    std::string error;
    auto a = cli::parse_command_args(spec, {"--stop", "A", "--tag", "t", "--stop", "B", "--quiet", "-h"}, error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->command == "x");
    CHECK(a->stops == std::vector<std::string>{"A", "B"});
    CHECK(a->all("stop") == std::vector<std::string>{"A", "B"});
    CHECK(a->all("tag") == std::vector<std::string>{"t"});
    CHECK(a->has("quiet"));
    CHECK(a->help);
    CHECK_FALSE(cli::parse_command_args(spec, {"--stp", "A"}, error).has_value());
    CHECK(error == "unknown option --stp (did you mean --stop?)");
}

TEST_CASE("cli spec: help rendering") {
    const std::string help = cli::render_help("prog", demo_spec());
    CHECK(help.rfind("prog demo - demo command\n\nUsage:\n  prog demo --name X\n", 0) == 0);
    CHECK(help.find("\nGroup:\n") != std::string::npos);
    CHECK(help.find("  --name X ") != std::string::npos);
    CHECK(help.find("  --tag T... ") != std::string::npos);
    // Continuation lines are aligned under the help column.
    const auto first = help.find("a flag\n");
    REQUIRE(first != std::string::npos);
    const auto line_start = help.rfind('\n', first) + 1;
    const auto column = first - line_start;
    const auto second = help.find("second line");
    REQUIRE(second != std::string::npos);
    CHECK(second - (help.rfind('\n', second) + 1) == column);
    CHECK(help.find("\nA note.\n") != std::string::npos);
    CHECK(help.find("Examples:\n  prog demo --name x\n") != std::string::npos);

    const cli::CommandSpec unnamed{"", "tool", {"--x"}, {}, {}, {}};
    CHECK(cli::render_help("tool-bin", unnamed).rfind("tool-bin - tool\n\nUsage:\n  tool-bin --x\n", 0) == 0);
}

TEST_CASE("cli commands: table integrity and help content") {
    std::set<std::string_view> names;
    for (const auto* c : cli::all_commands()) {
        CAPTURE(c->name);
        CHECK(names.insert(c->name).second);
        CHECK_FALSE(c->summary.empty());
        CHECK_FALSE(c->synopsis.empty());
        CHECK(cli::find_command(c->name) == c);
        std::set<std::string_view> options;
        for (const auto o : c->option_names()) {
            CAPTURE(o);
            CHECK(options.insert(o).second);  // no duplicate option within a command
            CHECK(o.rfind("--", 0) != 0);
        }
        const std::string help = cli::render_help(cli::kProgram, *c);
        CHECK(help.find("Usage:") != std::string::npos);
    }
    for (const char* expected : {"version", "devices", "backends", "models", "generate", "chat", "bench", "serve",
                                 "help"}) {
        CAPTURE(expected);
        CHECK(cli::find_command(expected) != nullptr);
    }
    CHECK(cli::find_command("nope") == nullptr);

    // The mock naming rule and exit codes are stated in the help of commands
    // that take a backend.
    for (const auto* c : {&cli::generate_command(), &cli::chat_command(), &cli::bench_command(),
                          &cli::models_command()}) {
        const std::string help = cli::render_help(cli::kProgram, *c);
        CAPTURE(c->name);
        CHECK(help.find("'mock' and 'mock:<anything>'") != std::string::npos);
        CHECK(help.find("about 16 tokens") != std::string::npos);
        CHECK(help.find("130 cancelled") != std::string::npos);
        CHECK(help.find("SONDER_INFER_BACKEND") != std::string::npos);
    }
    for (const auto* c : {&cli::generate_command(), &cli::chat_command(), &cli::bench_command()}) {
        CAPTURE(c->name);
        for (const char* option : {"llamaserver-config", "llamaserver-mode", "llamaserver-url",
                                   "llamaserver-executable", "llamaserver-arg"}) {
            CAPTURE(option);
            CHECK(c->find(option) != nullptr);
        }
    }
    // Correlation, output and JSON flags per command.
    for (const auto* c : {&cli::generate_command(), &cli::chat_command(), &cli::bench_command()}) {
        CAPTURE(c->name);
        for (const char* o : {"run-id", "agent-id", "task-id", "workload", "priority", "quiet"}) {
            CAPTURE(o);
            CHECK(c->find(o) != nullptr);
        }
    }
    for (const auto* c : {&cli::generate_command(), &cli::chat_command()}) {
        REQUIRE(c->find("stats") != nullptr);
        CHECK(c->find("stats")->kind == cli::OptionKind::value);
        CHECK(c->find("stop")->kind == cli::OptionKind::repeat);
    }
    for (const auto* c : {&cli::version_command(), &cli::devices_command(), &cli::backends_command(),
                          &cli::models_command()}) {
        CAPTURE(c->name);
        REQUIRE(c->find("json") != nullptr);
        CHECK(c->find("json")->kind == cli::OptionKind::flag);
    }
    CHECK(cli::generate_command().find("ollama-allow-remote")->kind == cli::OptionKind::flag);

    std::string error;
    CHECK_FALSE(cli::parse_command_args(cli::generate_command(), {"--max-token", "5"}, error).has_value());
    CHECK(error == "unknown option --max-token (did you mean --max-tokens?)");
    CHECK_FALSE(cli::parse_command_args(cli::chat_command(), {"--mesages", "f"}, error).has_value());
    CHECK(error == "unknown option --mesages (did you mean --messages?)");

    const std::string usage = cli::short_usage();
    for (const auto* c : cli::all_commands()) CHECK(usage.find("  " + std::string(c->name) + " ") != std::string::npos);
    CHECK(cli::overview().find("Exit status: 0 ok, 1 runtime or backend failure, 2 usage error, 130 cancelled") !=
          std::string::npos);
    CHECK(cli::suggest("genrate", cli::command_names()) == std::optional<std::string>("generate"));
}

TEST_CASE("cli values: checked integers") {
    std::string error;
    int v = 0;
    CHECK(cli::parse_integer<int>("runs", "12", 1, 1000, v, error));
    CHECK(v == 12);
    CHECK(cli::parse_integer<int>("priority", "-16", -16, 16, v, error));
    CHECK(v == -16);
    for (const char* bad : {"", "abc", "12x", " 1", "1 ", "+1", "1.5", "0x10"}) {
        CAPTURE(bad);
        CHECK_FALSE(cli::parse_integer<int>("runs", bad, 1, 1000, v, error));
        CHECK(error == "invalid value '" + std::string(bad) + "' for --runs (expected an integer from 1 to 1000)");
    }
    CHECK_FALSE(cli::parse_integer<int>("runs", "0", 1, 1000, v, error));
    CHECK(error == "value '0' for --runs is out of range (expected an integer from 1 to 1000)");
    CHECK_FALSE(cli::parse_integer<int>("priority", "17", -16, 16, v, error));
    CHECK(error.find("--priority") != std::string::npos);
    CHECK_FALSE(cli::parse_integer<int>("warmup", "99999999999999999999", 0, 100, v, error));
    CHECK(error.find("out of range") != std::string::npos);
    CHECK(v == -16);  // unchanged on failure

    std::uint64_t seed = 0;
    CHECK(cli::parse_u64("seed", "18446744073709551615", seed, error));
    CHECK(seed == std::numeric_limits<std::uint64_t>::max());
    CHECK_FALSE(cli::parse_u64("seed", "-1", seed, error));
    CHECK(error.find("'-1' for --seed") != std::string::npos);
    CHECK_FALSE(cli::parse_u64("seed", "18446744073709551616", seed, error));
}

TEST_CASE("cli values: checked numbers") {
    std::string error;
    double d = 0;
    constexpr double inf = std::numeric_limits<double>::infinity();
    CHECK(cli::parse_number("temperature", "0.7", -inf, inf, d, error));
    CHECK(d == doctest::Approx(0.7));
    CHECK(cli::parse_number("temperature", "-2", -inf, inf, d, error));
    CHECK(cli::parse_number("temperature", "1e-3", -inf, inf, d, error));
    CHECK(cli::parse_number("temperature", ".5", -inf, inf, d, error));
    CHECK(d == doctest::Approx(0.5));
    CHECK(cli::parse_number("temperature", "5.", -inf, inf, d, error));
    CHECK(cli::parse_number("temperature", "2E+1", -inf, inf, d, error));
    CHECK(d == doctest::Approx(20.0));
    // Only plain decimal syntax: no leading '+' (parse_integer rejects it
    // too), no hex floats, no bare dot or dangling exponent.
    for (const char* bad : {"", "abc", "0.5x", " 1", "1 ", "nan", "inf", "-inf", "1e999", "+0.5", "+1", "0x0.8p0",
                            "0X1", "-0x1p-1", ".", "-", "1e", "1e+", "e5", "--1", "1.2.3"}) {
        CAPTURE(bad);
        CHECK_FALSE(cli::parse_number("top-p", bad, -inf, inf, d, error));
        CHECK(error == "invalid value '" + std::string(bad) + "' for --top-p (expected a number)");
    }
    CHECK_FALSE(cli::parse_number("budget-seconds", "-1", 0.0, 10.0, d, error));
    CHECK(error == "value '-1' for --budget-seconds is out of range (expected a number from 0 to 10)");

    float f = 0;
    CHECK(cli::parse_float("top-p", "0.9", f, error));
    CHECK(f == doctest::Approx(0.9f));
    CHECK_FALSE(cli::parse_float("top-p", "abc", f, error));
    CHECK(error == "invalid value 'abc' for --top-p (expected a number)");
    CHECK_FALSE(cli::parse_float("top-p", "1e300", f, error));
    CHECK(error.find("--top-p") != std::string::npos);
}

TEST_CASE("cli values: correlation ids, workloads and stats modes") {
    CHECK(cli::is_valid_correlation_id("run-1.a_b:c"));
    CHECK(cli::is_valid_correlation_id(std::string(128, 'x')));
    CHECK_FALSE(cli::is_valid_correlation_id(""));
    CHECK_FALSE(cli::is_valid_correlation_id(std::string(129, 'x')));
    CHECK_FALSE(cli::is_valid_correlation_id("has space"));
    CHECK_FALSE(cli::is_valid_correlation_id("slash/"));
    std::string id, error;
    CHECK_FALSE(cli::parse_correlation_id("run-id", "a b", id, error));
    CHECK(error.find("'a b' for --run-id") != std::string::npos);
    CHECK(cli::parse_correlation_id("run-id", "ok", id, error));
    CHECK(id == "ok");

    CHECK(cli::parse_workload("interactive_user") == WorkloadClass::interactive_user);
    CHECK(cli::parse_workload("maintenance") == WorkloadClass::maintenance);
    CHECK(cli::parse_workload("research_worker") == WorkloadClass::research_worker);
    CHECK_FALSE(cli::parse_workload("interactive").has_value());
    CHECK(cli::workload_names() ==
          "interactive_user, owner_orchestrator, critic_verification, implementation_worker, research_worker, "
          "background_indexing, maintenance");

    CHECK(cli::parse_stats_mode("text") == cli::StatsMode::text);
    CHECK(cli::parse_stats_mode("json") == cli::StatsMode::json);
    CHECK(cli::parse_stats_mode("none") == cli::StatsMode::none);
    CHECK_FALSE(cli::parse_stats_mode("yaml").has_value());
}

TEST_CASE("cli values: exit codes") {
    CHECK(cli::kExitOk == 0);
    CHECK(cli::kExitFailure == 1);
    CHECK(cli::kExitUsage == 2);
    CHECK(cli::kExitCancelled == 130);
    CHECK(cli::exit_code_for(Status::success()) == 0);
    CHECK(cli::exit_code_for(Status(ErrorCode::cancelled, "x")) == 130);
    CHECK(cli::exit_code_for(Status(ErrorCode::unavailable, "x")) == 1);
    CHECK(cli::exit_code_for(Status(ErrorCode::not_found, "x")) == 1);
    CHECK(cli::exit_code_for(RequestOutcome::completed) == 0);
    CHECK(cli::exit_code_for(RequestOutcome::cancelled) == 130);
    CHECK(cli::exit_code_for(RequestOutcome::failed) == 1);
    CHECK(cli::exit_code_for(RequestOutcome::none) == 1);
}

TEST_CASE("cli values: stats lines") {
    GenerationResult r;
    r.request_id = "req-1";
    r.outcome = RequestOutcome::completed;
    r.stats.stop_reason = StopReason::max_tokens;
    r.stats.prompt_tokens = 9;
    r.stats.completion_tokens = 4;
    r.stats.token_counts_from_backend = true;
    r.ttft_ms = 1.5;
    r.total_ms = 7.25;
    auto s = cli::make_request_stats("chat", "mock", "mock:tiny", true, r);
    s.native_chat = false;
    s.messages = 4;
    const std::string text = cli::format_stats_text(s);
    CHECK(text.find("[sonder-infer] chat native=no messages=4 outcome=completed stop=max_tokens") == 0);
    CHECK(text.find("prompt_tokens=9 completion_tokens=4") != std::string::npos);

    auto parsed = json::parse(cli::format_stats_json(s));
    REQUIRE(parsed.ok());
    const auto& j = parsed.value();
    CHECK(j.find("command")->as_string() == "chat");
    CHECK(j.find("backend")->as_string() == "mock");
    CHECK(j.find("model")->as_string() == "mock:tiny");
    CHECK(j.find("synthetic")->as_bool());
    CHECK(j.find("request_id")->as_string() == "req-1");
    CHECK(j.find("outcome")->as_string() == "completed");
    CHECK(j.find("stop_reason")->as_string() == "max_tokens");
    CHECK(j.find("prompt_tokens")->as_int() == 9);
    CHECK(j.find("completion_tokens")->as_int() == 4);
    CHECK(j.find("ttft_ms")->as_double() == doctest::Approx(1.5));
    CHECK(j.find("total_ms")->as_double() == doctest::Approx(7.25));
    CHECK(j.find("messages")->as_int() == 4);
    CHECK_FALSE(j.find("native_chat")->as_bool(true));
    CHECK(j.find("turn") == nullptr);
    CHECK(cli::format_stats_json(s).find('\n') == std::string::npos);

    r.ttft_ms = -1;
    auto g = cli::make_request_stats("generate", "mock", "mock:tiny", true, r);
    CHECK(cli::format_stats_text(g).rfind("[sonder-infer] outcome=completed stop=max_tokens", 0) == 0);
    auto gj = json::parse(cli::format_stats_json(g));
    REQUIRE(gj.ok());
    CHECK(gj.value().find("ttft_ms")->is_null());
    CHECK(gj.value().find("messages") == nullptr);
}

TEST_CASE("cli env: defaults from the environment") {
    auto d = cli::read_env_defaults(env_of({{"SONDER_INFER_BACKEND", "mock"}, {"SONDER_INFER_MODEL", "mock:tiny"}}));
    CHECK(d.backend == std::optional<std::string>("mock"));
    CHECK(d.model == std::optional<std::string>("mock:tiny"));
    CHECK_FALSE(d.ollama_url.has_value());

    // Empty variables count as unset.
    d = cli::read_env_defaults(env_of({{"SONDER_INFER_BACKEND", ""}, {"OLLAMA_HOST", ""}}));
    CHECK_FALSE(d.backend.has_value());
    CHECK_FALSE(d.ollama_url.has_value());

    // SONDER_OLLAMA_URL wins over OLLAMA_HOST; it is used verbatim.
    d = cli::read_env_defaults(env_of({{"SONDER_OLLAMA_URL", "http://127.0.0.1:9999"}, {"OLLAMA_HOST", "other:1"}}));
    CHECK(d.ollama_url == std::optional<std::string>("http://127.0.0.1:9999"));

    // OLLAMA_HOST: host:port and URL forms.
    d = cli::read_env_defaults(env_of({{"OLLAMA_HOST", "127.0.0.1:11500"}}));
    CHECK(d.ollama_url == std::optional<std::string>("http://127.0.0.1:11500"));
    d = cli::read_env_defaults(env_of({{"OLLAMA_HOST", "http://localhost:11434/"}}));
    CHECK(d.ollama_url == std::optional<std::string>("http://localhost:11434"));
}

TEST_CASE("cli env: OLLAMA_HOST forms, shared and local implementations agree") {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"127.0.0.1:11500", "http://127.0.0.1:11500"},
        {"localhost", "http://localhost:11434"},
        {"0.0.0.0", "http://127.0.0.1:11434"},
        {"0.0.0.0:8080", "http://127.0.0.1:8080"},
        {":11434", "http://127.0.0.1:11434"},
        {"http://localhost:11434/", "http://localhost:11434"},
        {"http://127.0.0.1:11434", "http://127.0.0.1:11434"},
        {"https://gpu.example", "https://gpu.example:443"},
        {"https://gpu.example/ollama/", "https://gpu.example:443/ollama"},
        {"[::1]:9000", "http://[::1]:9000"},
        {"[::]", "http://127.0.0.1:11434"},
        {"[::1]", "http://[::1]:11434"},
    };
    for (const auto& [in, expected] : cases) {
        CAPTURE(in);
        CHECK(cli::normalize_ollama_host(in) == expected);
        CHECK(cli::detail::normalize_ollama_host_local(in) == expected);
    }
    // Both environment readers agree on every combination of variables.
    const std::vector<std::map<std::string, std::string>> envs = {
        {},
        {{"SONDER_INFER_BACKEND", "ollama"}, {"SONDER_INFER_MODEL", "qwen3:0.6b"}},
        {{"OLLAMA_HOST", "0.0.0.0:1234"}},
        {{"SONDER_OLLAMA_URL", ""}, {"OLLAMA_HOST", "https://h"}},
        {{"SONDER_OLLAMA_URL", "http://127.0.0.1:1"}, {"OLLAMA_HOST", "x"}},
    };
    for (const auto& vars : envs) {
        const auto shared = cli::read_env_defaults(env_of(vars));
        const auto local = cli::detail::read_env_defaults_local(env_of(vars));
        CHECK(shared.backend == local.backend);
        CHECK(shared.model == local.model);
        CHECK(shared.ollama_url == local.ollama_url);
    }
}

TEST_CASE("cli env: --ollama-allow-remote refuses plain http to remote hosts") {
    CHECK(cli::is_plain_http_remote("http://192.0.2.1:11434"));
    CHECK(cli::is_plain_http_remote("http://192.0.2.1"));
    CHECK(cli::is_plain_http_remote("HTTP://ollama.example:11434/base"));
    CHECK(cli::is_plain_http_remote("http://[2001:db8::1]:11434"));
    CHECK(cli::is_plain_http_remote("http://LOCALHOST:11434"));  // the client's loopback test is case-sensitive
    CHECK(cli::is_plain_http_remote("http://0.0.0.0:11434"));
    CHECK_FALSE(cli::is_plain_http_remote("https://ollama.example"));
    CHECK_FALSE(cli::is_plain_http_remote("http://127.0.0.1:11434"));
    CHECK_FALSE(cli::is_plain_http_remote("http://127.1.2.3"));
    // Only IPv4 literals in 127/8 are loopback, not DNS names starting "127.".
    CHECK(cli::is_plain_http_remote("http://127.0.0.1.attacker.example:11434"));
    CHECK(cli::is_plain_http_remote("http://127.evil/"));
    CHECK_FALSE(cli::is_plain_http_remote("http://localhost:11434/"));
    CHECK_FALSE(cli::is_plain_http_remote("http://[::1]:11434"));
    CHECK_FALSE(cli::is_plain_http_remote(""));
    CHECK_FALSE(cli::is_plain_http_remote("192.0.2.1:11434"));  // no scheme: the client rejects it
    CHECK_FALSE(cli::is_plain_http_remote("http://"));
    CHECK_FALSE(cli::is_plain_http_remote("http://[::1"));
    // Agrees with the Ollama client's own URL parser and loopback test for
    // every http:// URL the client accepts.
    for (const char* url : {"http://192.0.2.1:11434", "http://ollama.example/x/", "http://[2001:db8::1]:1",
                            "http://LOCALHOST", "http://127.0.0.1:11434", "http://localhost", "http://[::1]:11434",
                            "http://127.9.9.9:80/api", "HTTP://10.0.0.1", "http://0.0.0.0",
                            "http://127.0.0.1.attacker.example"}) {
        CAPTURE(url);
        auto parsed = net::parse_url(url);
        REQUIRE(parsed.ok());
        CHECK(cli::is_plain_http_remote(url) == !net::is_loopback_host(parsed.value().host));
    }
}

TEST_CASE("cli env: color only on a terminal without NO_COLOR") {
    CHECK(cli::color_enabled(true, env_of({})));
    CHECK_FALSE(cli::color_enabled(false, env_of({})));
    CHECK_FALSE(cli::color_enabled(true, env_of({{"NO_COLOR", "1"}})));
    CHECK(cli::color_enabled(true, env_of({{"NO_COLOR", ""}})));  // empty NO_COLOR is unset
    CHECK_FALSE(cli::color_enabled(true, env_of({{"TERM", "dumb"}})));
    CHECK(cli::color_enabled(true, env_of({{"TERM", "xterm-256color"}})));
}
