// sonder-infer command table: every command's options, help text and
// examples (docs/CLI.md). Header-only; used by tools/sonder-infer/main.cpp
// and tests/test_cli_spec.cpp.
#pragma once

#include <string_view>
#include <vector>

#include "cli_spec.hpp"

namespace sonder::cli {

inline constexpr std::string_view kProgram = "sonder-infer";

namespace commands_detail {

inline constexpr std::string_view kBackendNote =
    "Backends: mock is the deterministic MOCK backend for tests and harness work only\n"
    "(no inference; never a quality or performance signal). It serves the model names\n"
    "'mock' and 'mock:<anything>' and stops after about 16 tokens by design.\n"
    "ollama talks to a local Ollama server; llamacpp (SONDER_WITH_LLAMA_CPP=ON builds)\n"
    "takes a GGUF path as --model.";

inline constexpr std::string_view kEnvNote =
    "Environment (flags win): SONDER_INFER_BACKEND (--backend), SONDER_INFER_MODEL\n"
    "(--model), SONDER_OLLAMA_URL, then OLLAMA_HOST (host, host:port or URL) (--ollama-url).";

inline constexpr std::string_view kExitNote =
    "Exit status: 0 ok, 1 runtime or backend failure, 2 usage error, 130 cancelled (Ctrl-C).";

inline OptionGroup output_json() {
    return {"Output", {{"json", OptionKind::flag, "", "print one JSON document instead of text"}}};
}

inline OptionGroup ollama_options() {
    return {"Ollama",
            {{"ollama-url", OptionKind::value, "URL", "Ollama base URL (default http://127.0.0.1:11434)"},
             {"ollama-allow-remote", OptionKind::flag, "",
              "allow a non-loopback Ollama host (warns; remote hosts need https://,\n"
              "which needs a SONDER_WITH_TLS=ON build)"}}};
}

inline OptionGroup backend_options() {
    return {"Backend",
            {{"backend", OptionKind::value, "NAME", "mock | ollama | llamacpp (env SONDER_INFER_BACKEND)"},
             {"model", OptionKind::value, "MODEL",
              "model id (env SONDER_INFER_MODEL; the mock backend defaults to 'mock')"},
             {"ollama-url", OptionKind::value, "URL", "Ollama base URL (env SONDER_OLLAMA_URL, then OLLAMA_HOST)"},
             {"ollama-allow-remote", OptionKind::flag, "",
              "allow a non-loopback Ollama host (warns; remote hosts need https://,\n"
              "which needs a SONDER_WITH_TLS=ON build)"},
             {"mock-delay-ms", OptionKind::value, "N", "mock backend per-token delay, 0 to 60000 (default 0)"}}};
}

inline OptionGroup sampling_options() {
    return {"Sampling",
            {{"max-tokens", OptionKind::value, "N", "completion token limit (default 128)"},
             {"temperature", OptionKind::value, "F", "default 0 = greedy"},
             {"top-p", OptionKind::value, "F", "nucleus sampling (default 1 = off)"},
             {"top-k", OptionKind::value, "N", "default 0 = off"},
             {"min-p", OptionKind::value, "F", "default 0 = off"},
             {"repeat-penalty", OptionKind::value, "F", "default 1 = off"},
             {"seed", OptionKind::value, "N", "default 42"},
             {"typical-p", OptionKind::value, "F", "default 1 = off"},
             {"repeat-last-n", OptionKind::value, "N", "penalty window (default 64; -1 = whole context, 0 = off)"},
             {"presence-penalty", OptionKind::value, "F", "range [-2, 2] (default 0 = off)"},
             {"frequency-penalty", OptionKind::value, "F", "range [-2, 2] (default 0 = off)"},
             {"num-ctx", OptionKind::value, "N", "context window request (default 0 = model default)"},
             {"logit-bias", OptionKind::value, "LIST",
              "TOKEN:BIAS[,TOKEN:BIAS...]; BIAS may be -inf (not supported by ollama)"},
             {"stop", OptionKind::repeat, "TEXT", "stop sequence (repeatable)"}}};
}

inline OptionGroup telemetry_options() {
    return {"Telemetry (Observatory envelope sonder.observatory.event/1, JSONL)",
            {{"telemetry", OptionKind::value, "PATH", "write events to PATH ('-' for stderr)"},
             {"telemetry-level", OptionKind::value, "L", "off | metrics | standard | deep (default standard)"},
             {"capture-text", OptionKind::flag, "", "include generated text in token events"}}};
}

inline OptionGroup correlation_options() {
    return {"Correlation and scheduling (session metadata on every event)",
            {{"run-id", OptionKind::value, "ID", "envelope run_id ([A-Za-z0-9._:-], at most 128)"},
             {"agent-id", OptionKind::value, "ID", "envelope agent_id"},
             {"task-id", OptionKind::value, "ID", "envelope task_id"},
             {"workload", OptionKind::value, "CLASS",
              "interactive_user | owner_orchestrator | critic_verification |\n"
              "implementation_worker | research_worker | background_indexing |\n"
              "maintenance (default implementation_worker)"},
             {"priority", OptionKind::value, "N", "-16 to 16; positive runs sooner within the class (default 0)"}}};
}

inline OptionGroup stats_output_options() {
    return {"Output",
            {{"stats", OptionKind::value, "MODE",
              "text | json | none: per-request stats on stderr (default text;\n"
              "json writes one JSON object per request)"},
             {"quiet", OptionKind::flag, "",
              "suppress human status lines on stderr (stats default to none);\n"
              "with --telemetry - stderr carries only JSON lines"}}};
}

}  // namespace commands_detail

inline const CommandSpec& version_command() {
    using namespace commands_detail;
    static const CommandSpec spec{"version",
                                  "print version, commit, C ABI and HTTP API versions",
                                  {"version [--json]"},
                                  {output_json()},
                                  {kExitNote},
                                  {"version", "version --json"}};
    return spec;
}

inline const CommandSpec& devices_command() {
    using namespace commands_detail;
    static const CommandSpec spec{"devices",
                                  "list host devices (CPU and memory inventory)",
                                  {"devices [--json]"},
                                  {output_json()},
                                  {kExitNote},
                                  {"devices", "devices --json"}};
    return spec;
}

inline const CommandSpec& backends_command() {
    using namespace commands_detail;
    static const CommandSpec spec{
        "backends",
        "probe the backends in this build",
        {"backends [--json] [--ollama-url URL]"},
        {output_json(), ollama_options()},
        {kBackendNote, kEnvNote,
         "Exits 1 when no backend is available (an unreachable Ollama alone does not fail\n"
         "the command while the mock backend is available).",
         kExitNote},
        {"backends", "backends --json --ollama-url http://127.0.0.1:11434"}};
    return spec;
}

inline const CommandSpec& models_command() {
    using namespace commands_detail;
    static const CommandSpec spec{
        "models",
        "list the models a backend can serve",
        {"models --backend NAME [--json] [--ollama-url URL]"},
        {{"Backend",
          {{"backend", OptionKind::value, "NAME", "mock | ollama | llamacpp (env SONDER_INFER_BACKEND)"}}},
         output_json(),
         ollama_options()},
        {kBackendNote, kEnvNote, kExitNote},
        {"models --backend mock --json", "models --backend ollama"}};
    return spec;
}

inline const CommandSpec& generate_command() {
    using namespace commands_detail;
    static const CommandSpec spec{
        "generate",
        "generate a completion for one prompt",
        {"generate --backend NAME --model MODEL --prompt TEXT [options]"},
        {backend_options(),
         {"Input", {{"prompt", OptionKind::value, "TEXT", "prompt text (required)"}}},
         sampling_options(),
         telemetry_options(),
         correlation_options(),
         stats_output_options()},
        {kBackendNote, kEnvNote,
         "The completion streams to stdout; status and stats go to stderr. Ctrl-C cancels\n"
         "the request (request.cancelled is emitted) and exits 130.",
         kExitNote},
        {"generate --backend mock --model mock:tiny --prompt \"hello sonder\" --max-tokens 8",
         "generate --backend ollama --model qwen3:0.6b --prompt \"Say hi\" --telemetry events.jsonl",
         "generate --prompt hi --quiet --stats json --telemetry -   (with SONDER_INFER_BACKEND=mock)"}};
    return spec;
}

inline const CommandSpec& chat_command() {
    using namespace commands_detail;
    static const CommandSpec spec{
        "chat",
        "chat with a model (one-shot from a file, or interactive)",
        {"chat --backend NAME --model MODEL [--messages FILE] [--system TEXT] [options]"},
        {backend_options(),
         {"Chat",
          {{"messages", OptionKind::value, "FILE",
            "one-shot: answer the conversation in FILE (JSON array of\n"
            "{\"role\",\"content\"} or {\"messages\":[...]}) and exit"},
           {"system", OptionKind::value, "TEXT", "system prompt (ignored when the conversation has one)"}}},
         sampling_options(),
         telemetry_options(),
         correlation_options(),
         stats_output_options()},
        {kBackendNote, kEnvNote,
         "Without --messages, chat reads one user message per line from stdin. REPL commands:\n"
         "/help, /stats (last turn and totals), /reset (clear history, keep the system prompt),\n"
         "/exit or /quit (EOF also ends the chat); start a message with // to send a literal\n"
         "leading /. Ctrl-C cancels the current turn. Every turn runs through the engine\n"
         "session, so --telemetry records request.queued (kind chat) through request.completed.\n"
         "Interactive stats default to none (use /stats or --stats text|json). Labels are\n"
         "colored only on a terminal and never when NO_COLOR is set.",
         kExitNote},
        {"chat --backend mock --model mock:tiny --messages conversation.json",
         "chat --backend ollama --model qwen3:0.6b --system \"Be brief.\" --telemetry chat.jsonl"}};
    return spec;
}

inline const CommandSpec& bench_command() {
    using namespace commands_detail;
    static const CommandSpec spec{
        "bench",
        "run the benchmark harness over a corpus (JSON results)",
        {"bench --backend NAME --model MODEL --corpus FILE --out FILE [--markdown] [options]"},
        {backend_options(),
         {"Bench",
          {{"corpus", OptionKind::value, "FILE", "corpus JSON (e.g. bench/corpus/smoke.json; required)"},
           {"out", OptionKind::value, "FILE", "results JSON, schema sonder.inference.bench/1 (required)"},
           {"markdown", OptionKind::flag, "",
            "also print the markdown summary and write it next to --out\n"
            "(same name, .md extension)"},
           {"warmup", OptionKind::value, "N", "warmup runs per prompt, 0 to 100 (default 1)"},
           {"runs", OptionKind::value, "N", "measured runs per prompt, 1 to 1000 (default 3)"},
           {"label", OptionKind::value, "TEXT", "free-form label stored in results (default run_id)"}}},
         sampling_options(),
         telemetry_options(),
         correlation_options(),
         {"Output", {{"quiet", OptionKind::flag, "", "suppress progress lines on stderr"}}}},
        {kBackendNote, kEnvNote,
         "MOCK results are labelled in the markdown summary and are never a performance claim.",
         "Exits 1 when any request failed.", kExitNote},
        {"bench --backend mock --model mock:tiny --corpus bench/corpus/smoke.json --out r.json --warmup 0 --runs 1"}};
    return spec;
}

// `serve` options are parsed by serve_main() (src/server); its help comes
// from `sonder-infer serve --help`.
inline const CommandSpec& serve_command() {
    static const CommandSpec spec{"serve",
                                  "local HTTP API (OpenAI-compatible subset) and live telemetry (docs/SERVER.md)",
                                  {"serve --backend mock|ollama|llamacpp [--model ID]... [options]"},
                                  {},
                                  {},
                                  {"serve --backend mock --port 11437"}};
    return spec;
}

inline const CommandSpec& help_command() {
    static const CommandSpec spec{"help",
                                  "show help for a command",
                                  {"help [COMMAND]", "COMMAND --help"},
                                  {},
                                  {},
                                  {"help chat", "generate --help"}};
    return spec;
}

// All commands, in display order.
inline const std::vector<const CommandSpec*>& all_commands() {
    static const std::vector<const CommandSpec*> list{&generate_command(), &chat_command(),     &serve_command(),
                                                      &bench_command(),    &models_command(),   &backends_command(),
                                                      &devices_command(),  &version_command(),  &help_command()};
    return list;
}

inline const CommandSpec* find_command(std::string_view name) {
    for (const auto* c : all_commands()) {
        if (c->name == name) return c;
    }
    return nullptr;
}

inline std::vector<std::string_view> command_names() {
    std::vector<std::string_view> out;
    for (const auto* c : all_commands()) out.push_back(c->name);
    return out;
}

// Short list printed for a bare `sonder-infer` or an unknown command.
inline std::string short_usage() {
    return "usage: sonder-infer <command> [options]\n\nCommands:\n" + render_command_list(all_commands()) +
           "\nRun 'sonder-infer help <command>' (or '<command> --help') for options and examples.\n";
}

// Full overview for `sonder-infer help` / --help.
inline std::string overview() {
    return std::string("sonder-infer - Sonder Inference CLI (docs/CLI.md)\n\n") + short_usage() + "\n" +
           std::string(commands_detail::kBackendNote) + "\n\n" + std::string(commands_detail::kEnvNote) + "\n\n" +
           std::string(commands_detail::kExitNote) + "\n";
}

}  // namespace sonder::cli
