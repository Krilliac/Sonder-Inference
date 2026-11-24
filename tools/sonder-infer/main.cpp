// sonder-infer: command-line interface over the Sonder Inference library.
// Reference: docs/CLI.md (commands, environment, exit codes, JSON shapes).
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli_runtime.hpp"
#include "sonder/inference.hpp"
#include "sonder_inference.h"
// Header-only CLI helpers (option specs, checked values, environment, chat
// files and the chat loop); shared with the unit tests (tests/test_cli_*.cpp).
#include "../../src/cli/cli_args.hpp"
#include "../../src/cli/cli_env.hpp"
#include "../../src/cli/cli_spec.hpp"
#include "../../src/cli/cli_values.hpp"
#include "../../src/cli/sonder_infer_commands.hpp"
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "../../src/backends/llamaserver/tune.hpp"
#endif
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/server.hpp"
#endif
#if defined(SONDER_HAS_BENCH)
#include "sonder/inference/benchmark.hpp"
#endif
#if defined(SONDER_HAS_WEIGHT_MANIFEST)
#include "sonder/inference/weight_manifest.hpp"
#endif

namespace si = sonder::inference;
namespace cli = sonder::cli;

namespace {

using cli::Args;
using namespace sonder::infer_cli;

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
int cmd_version(const Args& a) {
    if (a.has("json")) {
        si::json::Array backends;
        for (const auto& n : compiled_backends()) backends.emplace_back(n);
#if defined(SONDER_HAS_SERVER)
        const si::json::Value api_version(si::server::kApiVersion);
        const bool server = true;
#else
        const si::json::Value api_version(nullptr);
        const bool server = false;
#endif
#if defined(SONDER_HAS_BENCH)
        const bool bench = true;
#else
        const bool bench = false;
#endif
        const si::json::Object doc{{"name", "sonder-infer"},
                                   {"version", si::version_string()},
                                   {"commit", si::build_commit()},
                                   {"abi_version", SONDER_ABI_VERSION},
                                   {"api_version", api_version},
                                   {"platform", si::host_platform()},
                                   {"backends", std::move(backends)},
                                   {"modules", si::json::Object{{"server", server}, {"bench", bench}}}};
        std::cout << si::json::Value(doc).dump() << "\n";
        return cli::kExitOk;
    }
    std::cout << "sonder-infer " << si::version_string() << " (" << si::build_commit() << ") C ABI "
              << SONDER_ABI_VERSION;
#if defined(SONDER_HAS_SERVER)
    std::cout << " HTTP API " << si::server::kApiVersion;
#endif
    std::cout << "\n";
    return cli::kExitOk;
}

int cmd_devices(const Args& a) {
    const auto devices = si::enumerate_devices();
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& d : devices) {
            list.emplace_back(si::json::Object{{"id", d.id},
                                               {"kind", si::to_string(d.kind)},
                                               {"name", d.name},
                                               {"logical_cores", d.logical_cores},
                                               {"total_memory_bytes", d.total_memory_bytes},
                                               {"available_memory_bytes", d.available_memory_bytes}});
        }
        std::cout << si::json::Value(si::json::Object{{"platform", si::host_platform()}, {"devices", std::move(list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& d : devices) {
        std::cout << d.id << "  kind=" << si::to_string(d.kind) << "  name=\"" << d.name << "\""
                  << "  logical_cores=" << d.logical_cores << "  total_memory_mib=" << (d.total_memory_bytes >> 20)
                  << "  available_memory_mib=" << (d.available_memory_bytes >> 20) << "\n";
    }
    std::cout << "platform=" << si::host_platform() << "\n";
    return cli::kExitOk;
}

int cmd_backends(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("backends", a, env, false, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("backends", a, false, choice, engine, rc)) return rc;
    const bool json = a.has("json");
    si::json::Array list;
    std::size_t available = 0;
    bool ollama_unreachable = false;
    for (const auto& name : engine->backend_names()) {
        auto b = engine->find_backend(name);
        auto probe = b->probe();
        if (probe.ok()) ++available;
        if (!probe.ok() && name == "ollama" && probe.status().code() == si::ErrorCode::unavailable) {
            ollama_unreachable = true;
        }
        if (json) {
            si::json::Array caps;
            for (const auto& c : b->capabilities().names()) caps.emplace_back(c);
            list.emplace_back(si::json::Object{
                {"name", name},
                {"available", probe.ok()},
                {"version", probe.ok() ? si::json::Value(probe.value()) : si::json::Value(nullptr)},
                {"error", probe.ok() ? si::json::Value(nullptr) : si::json::Value(probe.status().to_string())},
                {"description", b->description()},
                {"synthetic", is_mock_backend(name)},
                {"capabilities", std::move(caps)}});
            continue;
        }
        std::cout << name << "  "
                  << (probe.ok() ? "reachable version=" + probe.value()
                                 : "unavailable (" + probe.status().to_string() + ")")
                  << "\n    " << b->description() << "\n    capabilities:";
        for (const auto& c : b->capabilities().names()) std::cout << " " << c;
        std::cout << "\n";
    }
    if (json) {
        std::cout << si::json::Value(si::json::Object{{"backends", std::move(list)}, {"available", available}}).dump()
                  << "\n";
    } else if (ollama_unreachable) {
        std::cout << "hint: Ollama is not reachable; start it with 'ollama serve', or pass --ollama-url URL\n";
    }
    if (available == 0) {
        err() << "error: backends: no backend is available\n";
        return cli::kExitFailure;
    }
    return cli::kExitOk;
}

int cmd_models(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("models", a, env, true, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("models", a, false, choice, engine, rc)) return rc;
    auto models = engine->find_backend(choice.backend)->list_models();
    if (!models.ok()) {
        report_error("models", models.status(), choice.backend);
        return cli::exit_code_for(models.status());
    }
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& m : models.value()) {
            list.emplace_back(si::json::Object{{"name", m.name},
                                               {"format", m.format},
                                               {"family", m.family},
                                               {"parameter_size", m.parameter_size},
                                               {"quantization", m.quantization},
                                               {"size_bytes", m.size_bytes},
                                               {"context_length", m.context_length}});
        }
        std::cout << si::json::Value(si::json::Object{{"backend", choice.backend},
                                                      {"synthetic", is_mock_backend(choice.backend)},
                                                      {"models", std::move(list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& m : models.value()) {
        std::cout << m.name << "  format=" << m.format << "  family=" << m.family << "  params=" << m.parameter_size
                  << "  quant=" << m.quantization << "  size_mib=" << (m.size_bytes >> 20) << "\n";
    }
    return cli::kExitOk;
}

int cmd_generate(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    if (!stats_mode_from_args(a, cli::StatsMode::text, stats_mode, error)) return usage_error("generate", error);
    const auto prompt = a.get("prompt");
    if (!prompt) return usage_error("generate", "--prompt is required");
    RequestSetup setup;
    if (!prepare_request("generate", a, env, quiet, setup, {}, rc)) return rc;
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("generate", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
    {
        si::Session& s = *session.value();
        InterruptWatcher watcher([&s] { s.cancel(); });
        res = s.generate(*prompt, [](const si::TokenChunk& c) {
            std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
            std::cout.flush();
            return true;
        });
    }
    std::cout << "\n";
    std::cout.flush();
    if (!res.ok()) {
        report_error("generate", res.status(), setup.choice.backend);
        return cli::exit_code_for(res.status());
    }
    print_stats(stats_mode, cli::make_request_stats("generate", setup.choice.backend, setup.model->descriptor().name,
                                                    is_mock_backend(setup.choice.backend), res.value()));
    return cli::exit_code_for(res.value().outcome);
}

int cmd_chat(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    const auto messages_path = a.get("messages");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    // One-shot chat prints stats by default; the interactive REPL only on
    // request (/stats, or --stats text|json).
    if (!stats_mode_from_args(a, messages_path ? cli::StatsMode::text : cli::StatsMode::none, stats_mode, error)) {
        return usage_error("chat", error);
    }
    std::vector<si::ChatMessage> file_messages;
    const auto read_messages = [&](int& code) {
        if (!messages_path) return true;
        auto loaded = cli::load_chat_messages(*messages_path);
        if (!loaded.ok()) {
            code = usage_error("chat", loaded.status().message());
            return false;
        }
        file_messages = std::move(loaded.value());
        return true;
    };
    RequestSetup setup;
    if (!prepare_request("chat", a, env, quiet, setup, read_messages, rc)) return rc;
    const std::string system = a.get("system").value_or("");
    const bool native = setup.model->backend_model().has_native_chat();
    const std::string model_name = setup.model->descriptor().name;
    const bool synthetic = is_mock_backend(setup.choice.backend);

#if defined(SONDER_HAS_SESSION_CHAT)
    // Every turn runs through the engine session (scheduler, KV accounting,
    // request telemetry with request.queued.kind = "chat").
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("chat", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Session& s = *session.value();
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&s] { s.cancel(); });
            res = s.chat(history, [](const si::TokenChunk& c) {
                std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
                std::cout.flush();
                return true;
            });
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().outcome == si::RequestOutcome::cancelled;
        report.stats = cli::make_request_stats("chat", setup.choice.backend, model_name, synthetic, res.value());
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#else
    // Library without Session::chat: turns go straight to the backend model
    // and emit no request telemetry.
    std::uint64_t turn_no = 0;
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::CancellationSource source;
        si::Result<cli::ChatTurnResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&source] { source.cancel(); });
            res = cli::run_chat_turn(setup.model->backend_model(), history, setup.sampling, source.token(), std::cout,
                                     "chat-" + std::to_string(++turn_no));
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().stats.stop_reason == si::StopReason::cancelled;
        report.stats.command = "chat";
        report.stats.backend = setup.choice.backend;
        report.stats.model = model_name;
        report.stats.synthetic = synthetic;
        report.stats.outcome = report.cancelled ? "cancelled" : "completed";
        report.stats.stop_reason = si::to_string(res.value().stats.stop_reason);
        report.stats.prompt_tokens = res.value().stats.prompt_tokens;
        report.stats.completion_tokens = res.value().stats.completion_tokens;
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#endif

    if (messages_path) {
        cli::apply_system_prompt(file_messages, system);
        auto res = turn(file_messages);
        std::cout << "\n";
        std::cout.flush();
        if (!res.ok()) {
            report_error("chat", res.status(), setup.choice.backend);
            return cli::exit_code_for(res.status());
        }
        print_stats(stats_mode, res.value().stats);
        return res.value().cancelled ? cli::kExitCancelled : cli::kExitOk;
    }

    const bool interactive = stdin_is_terminal();
    const bool terminal_out = stdout_is_terminal();
    if (!quiet) {
        err() << "[sonder-infer] chat with " << model_name << " on " << setup.choice.backend
              << (native ? " (native chat)" : " (generic prompt format)")
              << "; /help lists commands, /exit or EOF quits\n";
    }
    cli::ChatReplOptions ro;
    ro.system = system;
    ro.user_prompt = interactive ? "you> " : "";
    ro.assistant_label = interactive || terminal_out ? "assistant> " : "";
    ro.color = cli::color_enabled(terminal_out) && stdout_supports_ansi();
    ro.stats = stats_mode;
    return cli::run_chat_session(std::cin, std::cout, err(), ro, turn);
}

#if defined(SONDER_HAS_BENCH)
int cmd_bench(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    int rc = cli::kExitOk;
    const auto corpus_path = a.get("corpus");
    const auto out_path = a.get("out");
    if (!corpus_path || !out_path) return usage_error("bench", "--corpus and --out are required");
    si::bench::Options opts;
    std::string error;
    if (auto v = a.get("warmup"); v && !cli::parse_integer<int>("warmup", *v, 0, 100, opts.warmup_runs, error)) {
        return usage_error("bench", error);
    }
    if (auto v = a.get("runs"); v && !cli::parse_integer<int>("runs", *v, 1, 1000, opts.measured_runs, error)) {
        return usage_error("bench", error);
    }
    opts.label = a.get("label").value_or("");
    std::optional<si::bench::Corpus> corpus;
    const auto read_corpus = [&](int& code) {
        auto loaded = si::bench::load_corpus(*corpus_path);
        if (!loaded.ok()) {
            code = usage_error("bench", loaded.status().to_string());
            return false;
        }
        corpus = std::move(loaded.value());
        return true;
    };
    RequestSetup setup;
    if (!prepare_request("bench", a, env, quiet, setup, read_corpus, rc)) return rc;
    opts.sampling = setup.sampling;
    opts.run_id = setup.meta.run_id;
    opts.agent_id = setup.meta.agent_id;
    opts.task_id = setup.meta.task_id;
    opts.workload = setup.meta.workload;
    opts.priority = setup.meta.priority;
    auto doc = si::bench::run(*setup.engine, setup.model, *corpus, opts, quiet ? nullptr : &err());
    std::ofstream out(*out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err() << "error: bench: cannot write " << *out_path << "\n";
        return cli::kExitFailure;
    }
    out << si::json::Value(doc).dump() << "\n";
    out.close();
    if (a.has("markdown")) {
        const std::string markdown = si::bench::render_markdown(si::json::Value(doc));
        std::string md_path = *out_path;
        if (md_path.size() > 5 && md_path.compare(md_path.size() - 5, 5, ".json") == 0) {
            md_path.resize(md_path.size() - 5);
        }
        md_path += ".md";
        std::ofstream md(md_path, std::ios::binary | std::ios::trunc);
        if (!md) {
            err() << "error: bench: cannot write " << md_path << "\n";
            return cli::kExitFailure;
        }
        md << markdown;
        std::cout << markdown;
        if (!quiet) err() << "[bench] wrote " << md_path << "\n";
    }
    const auto* summary = doc.find("summary");
    if (!quiet) {
        err() << "[bench] wrote " << *out_path << " summary=" << (summary ? summary->dump() : "{}") << "\n";
    }
    const auto* failures = summary ? summary->find("failures") : nullptr;
    return failures && failures->as_int() > 0 ? cli::kExitFailure : cli::kExitOk;
}
#endif

bool wants_help(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

#if defined(SONDER_HAS_SERVER)
// The --ollama-allow-remote plain-HTTP refusal for serve, checked before
// serve_main() runs (src/server is outside this lane). Serve's parser takes
// "--ollama-url URL" and "--ollama-url=URL"; the environment fills in the
// URL as it does for serve (backend_env_defaults()).
bool serve_ollama_remote_ok(const std::vector<std::string>& args, int& rc) {
    bool allow_remote = false;
    std::optional<std::string> url;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--ollama-allow-remote") {
            allow_remote = true;
        } else if (arg == "--ollama-url" && i + 1 < args.size()) {
            url = args[++i];
        } else if (arg.rfind("--ollama-url=", 0) == 0) {
            url = arg.substr(std::string_view("--ollama-url=").size());
        }
    }
    if (!allow_remote) return true;
    if (!url) url = cli::read_env_defaults().ollama_url;
    return check_ollama_remote_url("serve", url.value_or(""), rc);
}
#endif

int cmd_serve(const std::vector<std::string>& args) {
#if defined(SONDER_HAS_SERVER)
    // --help wins over any other argument, as for the other commands.
    if (wants_help(args)) return si::server::serve_main({"--help"}, std::cout, std::cerr);
    int rc = cli::kExitOk;
    if (!serve_ollama_remote_ok(args, rc)) return rc;
    return si::server::serve_main(args, std::cout, std::cerr);
#else
    (void)args;
    err() << "error: serve: this build does not include the server module (src/server); "
             "build from a tree that contains it to use 'sonder-infer serve'\n";
    return cli::kExitUsage;
#endif
}

int cmd_help(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << cli::overview();
        return cli::kExitOk;
    }
    if (args.size() == 1 && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << cli::render_help(cli::kProgram, cli::help_command());
        return cli::kExitOk;
    }
    if (args.size() > 1) return usage_error("help", "expected at most one command name");
    const std::string& name = args.front();
    if (name == "serve") return cmd_serve({"--help"});
    if (const auto* spec = cli::find_command(name)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    std::string message = "unknown command '" + name + "'";
    if (auto s = cli::suggest(name, cli::command_names())) message += " (did you mean '" + *s + "'?)";
    err() << "error: help: " << message << "\n\n" << cli::short_usage();
    return cli::kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    if (args.empty()) {
        err() << cli::short_usage();
        return cli::kExitUsage;
    }
    const std::string cmd = args.front();
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (cmd == "help" || cmd == "--help" || cmd == "-h") return cmd_help(rest);
    if (cmd == "--version") return cmd_version(Args{});
    const cli::CommandSpec* spec = cli::find_command(cmd);
    if (spec == nullptr) {
        std::string message = "unknown command '" + cmd + "'";
        if (auto s = cli::suggest(cmd, cli::command_names())) message += " (did you mean '" + *s + "'?)";
        err() << "error: " << message << "\n\n" << cli::short_usage();
        return cli::kExitUsage;
    }
    if (cmd == "serve") return cmd_serve(rest);
    if (cmd == "weights") {
#if defined(SONDER_HAS_WEIGHT_MANIFEST)
        return si::weights::weights_main(rest, std::cout, std::cerr);
#else
        err() << "error: weights: this build does not include the artifacts module (src/artifacts)\n";
        return 4;  // `weights` usage code (cli::kExitUsage is 2, which `weights` uses for "mismatch")
#endif
    }
    if (cmd == "tune") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        return si::llamaserver::tune::tune_main(rest, std::cout, std::cerr);
#else
        err() << "error: tune: this build does not include the llamaserver module\n";
        return cli::kExitUsage;
#endif
    }

    std::string error;
    auto parsed = cli::parse_command_args(*spec, rest, error);
    if (parsed ? parsed->help : wants_help(rest)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    if (!parsed) return usage_error(cmd, error);
    const Args& a = *parsed;
    const cli::EnvDefaults env = cli::read_env_defaults();

    if (cmd == "version") return cmd_version(a);
    if (cmd == "devices") return cmd_devices(a);
    if (cmd == "backends") return cmd_backends(a, env);
    if (cmd == "models") return cmd_models(a, env);
    if (cmd == "generate") return cmd_generate(a, env);
    if (cmd == "chat") return cmd_chat(a, env);
#if defined(SONDER_HAS_BENCH)
    if (cmd == "bench") return cmd_bench(a, env);
#else
    if (cmd == "bench") {
        err() << "error: bench: this build does not include the bench module (bench/)\n";
        return cli::kExitUsage;
    }
#endif
    err() << "error: unknown command '" << cmd << "'\n\n" << cli::short_usage();
    return cli::kExitUsage;
}
