// sonder-infer: small CLI over the Sonder Inference library.
#include <atomic>
#include <chrono>
#include <thread>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sonder/inference.hpp"
#include "sonder_inference.h"
// Header-only CLI helpers (argument parsing, chat files, chat loop); shared
// with tests/test_cli_args.cpp.
#include "../../src/cli/cli_args.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif
#if defined(SONDER_HAS_BENCH)
#include "sonder/inference/benchmark.hpp"
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
#include "sonder/inference/backends/llamacpp.hpp"
#endif

namespace si = sonder::inference;

namespace {

constexpr const char* kUsage = R"(sonder-infer - Sonder Inference CLI

Usage:
  sonder-infer version
  sonder-infer devices
  sonder-infer backends [--ollama-url URL]
  sonder-infer models   --backend NAME [--ollama-url URL]
  sonder-infer generate --backend NAME --model MODEL --prompt TEXT [options]
  sonder-infer chat     --backend NAME --model MODEL [--messages FILE] [--system TEXT] [options]
  sonder-infer bench    --backend NAME --model MODEL --corpus FILE --out FILE [--markdown] [options]

Backends:
  mock     deterministic MOCK backend for tests only (no inference performed)
  ollama   Ollama compatibility adapter (module; default http://127.0.0.1:11434)
  llamacpp direct llama.cpp backend (module; build with SONDER_WITH_LLAMA_CPP=ON; --model is a GGUF path)

Generation options:
  --max-tokens N        (default 128)
  --temperature F       (default 0 = greedy)
  --top-p F  --top-k N  --min-p F  --repeat-penalty F
  --seed N              (default 42)
  --stop TEXT           (repeatable)

Telemetry options (Observatory envelope v1, JSONL):
  --telemetry PATH      write events to PATH ('-' for stderr)
  --telemetry-level L   off|metrics|standard|deep (default standard)
  --capture-text        include generated text in token events

Chat options:
  --messages FILE       one-shot: answer the conversation in FILE (JSON array of
                        {"role","content"} or {"messages":[...]}) and exit.
                        Without it, chat reads user lines from stdin
                        interactively (/reset clears history, /exit quits).
  --system TEXT         system prompt (ignored if the conversation has one)

Bench options:
  --warmup N            warmup runs per prompt (default 1)
  --runs N              measured runs per prompt (default 3)
  --label TEXT          free-form label stored in results
  --markdown            also print the markdown summary to stdout and write it
                        next to --out (same name, .md extension)

Other:
  --ollama-url URL      Ollama base URL (loopback only)
  --mock-delay-ms N     per-token delay for the mock backend
)";

using sonder::cli::Args;
using sonder::cli::parse_args;

// The signal handler only sets a lock-free flag (async-signal-safe); a
// watcher thread turns it into a cancel callback (Session::cancel() or a
// CancellationSource for chat turns).
volatile std::sig_atomic_t g_interrupted = 0;

extern "C" void on_sigint(int) { g_interrupted = 1; }

class InterruptWatcher {
public:
    explicit InterruptWatcher(std::function<void()> on_interrupt) : on_interrupt_(std::move(on_interrupt)) {
        g_interrupted = 0;
        std::signal(SIGINT, on_sigint);
        thread_ = std::thread([this] {
            while (!done_.load()) {
                if (g_interrupted != 0) {
                    on_interrupt_();
                    g_interrupted = 0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    ~InterruptWatcher() {
        done_.store(true);
        thread_.join();
        std::signal(SIGINT, SIG_DFL);
    }
    InterruptWatcher(const InterruptWatcher&) = delete;
    InterruptWatcher& operator=(const InterruptWatcher&) = delete;

private:
    std::function<void()> on_interrupt_;
    std::atomic<bool> done_{false};
    std::thread thread_;
};

struct Runtime {
    std::unique_ptr<si::Engine> engine;
};

bool build_engine(const Args& a, Runtime& rt) {
    si::EngineOptions eo;
    eo.telemetry.capture_text = a.has("capture-text");
    const std::string level_text = a.get("telemetry-level").value_or("standard");
    auto level = si::parse_telemetry_level(level_text);
    if (!level) {
        std::cerr << "error: invalid --telemetry-level " << level_text << "\n";
        return false;
    }
    eo.telemetry.level = *level;
    if (auto path = a.get("telemetry")) {
        if (*path == "-") {
            eo.telemetry_sinks.push_back(std::shared_ptr<si::TelemetrySink>(si::make_ostream_sink(std::cerr)));
        } else {
            si::Status st;
            auto sink = si::make_jsonl_file_sink(*path, false, &st);
            if (!sink) {
                std::cerr << "error: " << st.to_string() << "\n";
                return false;
            }
            eo.telemetry_sinks.push_back(std::shared_ptr<si::TelemetrySink>(std::move(sink)));
        }
    }
    rt.engine = std::make_unique<si::Engine>(std::move(eo));

    si::MockBackendOptions mo;
    if (auto d = a.get("mock-delay-ms")) {
        mo.token_delay = std::chrono::milliseconds(std::atoi(d->c_str()));
    }
    rt.engine->register_backend(si::make_mock_backend(mo));
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    si::OllamaBackendOptions oo;
    if (auto url = a.get("ollama-url")) {
        oo.base_url = *url;
    }
    rt.engine->register_backend(si::make_ollama_backend(oo));
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    // Direct llama.cpp backend: --model takes a GGUF path.
    rt.engine->register_backend(si::make_llamacpp_backend());
#endif
    return true;
}

bool sampling_from_args(const Args& a, si::SamplingConfig& s) {
    s = si::SamplingConfig::greedy(128, 42);
    try {
        if (auto v = a.get("max-tokens")) s.max_tokens = std::stoi(*v);
        if (auto v = a.get("temperature")) s.temperature = std::stof(*v);
        if (auto v = a.get("top-p")) s.top_p = std::stof(*v);
        if (auto v = a.get("top-k")) s.top_k = std::stoi(*v);
        if (auto v = a.get("min-p")) s.min_p = std::stof(*v);
        if (auto v = a.get("repeat-penalty")) s.repeat_penalty = std::stof(*v);
        if (auto v = a.get("seed")) s.seed = std::stoull(*v);
    } catch (const std::exception&) {
        std::cerr << "error: invalid numeric option\n";
        return false;
    }
    s.stop = a.stops;
    if (auto st = si::validate(s); !st.ok()) {
        std::cerr << "error: " << st.to_string() << "\n";
        return false;
    }
    return true;
}

int cmd_devices() {
    for (const auto& d : si::enumerate_devices()) {
        std::cout << d.id << "  kind=" << si::to_string(d.kind) << "  name=\"" << d.name << "\""
                  << "  logical_cores=" << d.logical_cores << "  total_memory_mib=" << (d.total_memory_bytes >> 20)
                  << "  available_memory_mib=" << (d.available_memory_bytes >> 20) << "\n";
    }
    std::cout << "platform=" << si::host_platform() << "\n";
    return 0;
}

int cmd_backends(Runtime& rt) {
    for (const auto& name : rt.engine->backend_names()) {
        auto b = rt.engine->find_backend(name);
        auto probe = b->probe();
        std::cout << name << "  " << (probe.ok() ? "reachable version=" + probe.value() : "unavailable (" + probe.status().to_string() + ")")
                  << "\n    " << b->description() << "\n    capabilities:";
        for (const auto& c : b->capabilities().names()) std::cout << " " << c;
        std::cout << "\n";
    }
    return 0;
}

int cmd_models(const Args& a, Runtime& rt) {
    auto name = a.get("backend");
    if (!name) {
        std::cerr << "error: --backend required\n";
        return 2;
    }
    auto b = rt.engine->find_backend(*name);
    if (!b) {
        std::cerr << "error: unknown backend " << *name << "\n";
        return 2;
    }
    auto models = b->list_models();
    if (!models.ok()) {
        std::cerr << "error: " << models.status().to_string() << "\n";
        return 1;
    }
    for (const auto& m : models.value()) {
        std::cout << m.name << "  format=" << m.format << "  family=" << m.family << "  params=" << m.parameter_size
                  << "  quant=" << m.quantization << "  size_mib=" << (m.size_bytes >> 20) << "\n";
    }
    return 0;
}

std::shared_ptr<si::Model> load(const Args& a, Runtime& rt) {
    auto backend = a.get("backend");
    auto model = a.get("model");
    if (!backend || !model) {
        std::cerr << "error: --backend and --model are required\n";
        return nullptr;
    }
    si::ModelLoadOptions lo;
    lo.model = *model;
    auto loaded = rt.engine->load_model(*backend, lo);
    if (!loaded.ok()) {
        std::cerr << "error: " << loaded.status().to_string() << "\n";
        return nullptr;
    }
    return loaded.value();
}

int cmd_generate(const Args& a, Runtime& rt) {
    auto prompt = a.get("prompt");
    if (!prompt) {
        std::cerr << "error: --prompt required\n";
        return 2;
    }
    si::SamplingConfig sampling;
    if (!sampling_from_args(a, sampling)) return 2;
    auto model = load(a, rt);
    if (!model) return 1;
    si::SessionOptions so;
    so.sampling = sampling;
    auto session = rt.engine->create_session(model, so);
    if (!session.ok()) {
        std::cerr << "error: " << session.status().to_string() << "\n";
        return 1;
    }
    si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
    {
        si::Session& s = *session.value();
        InterruptWatcher watcher([&s] { s.cancel(); });
        res = session.value()->generate(*prompt, [](const si::TokenChunk& c) {
            std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
            std::cout.flush();
            return true;
        });
    }
    std::cout << "\n";
    if (!res.ok()) {
        std::cerr << "error: " << res.status().to_string() << "\n";
        return 1;
    }
    const auto& r = res.value();
    std::cerr << "[sonder-infer] outcome=" << si::to_string(r.outcome) << " stop=" << si::to_string(r.stats.stop_reason)
              << " prompt_tokens=" << r.stats.prompt_tokens << " completion_tokens=" << r.stats.completion_tokens
              << " ttft_ms=" << r.ttft_ms << " total_ms=" << r.total_ms << "\n";
    return r.outcome == si::RequestOutcome::completed ? 0 : 130;
}

// Chat goes straight to BackendModel::chat() (native chat on ollama and
// llamacpp, generic prompt formatting elsewhere). Session has no chat entry
// point yet, so chat turns do not emit Observatory session telemetry; see
// docs/integration/chat-cli.md.
int cmd_chat(const Args& a, Runtime& rt) {
    si::SamplingConfig sampling;
    if (!sampling_from_args(a, sampling)) return 2;
    std::vector<si::ChatMessage> file_messages;
    const auto messages_path = a.get("messages");
    if (messages_path) {
        auto loaded = sonder::cli::load_chat_messages(*messages_path);
        if (!loaded.ok()) {
            std::cerr << "error: " << loaded.status().to_string() << "\n";
            return 2;
        }
        file_messages = std::move(loaded.value());
    }
    const std::string system = a.get("system").value_or("");
    auto model = load(a, rt);
    if (!model) return 1;
    si::BackendModel& backend_model = model->backend_model();
    std::uint64_t turn_no = 0;

    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<sonder::cli::ChatTurnResult> {
        si::CancellationSource source;
        InterruptWatcher watcher([&source] { source.cancel(); });
        return sonder::cli::run_chat_turn(backend_model, history, sampling, source.token(), std::cout,
                                          "chat-" + std::to_string(++turn_no));
    };

    if (messages_path) {
        sonder::cli::apply_system_prompt(file_messages, system);
        auto res = turn(file_messages);
        std::cout << "\n";
        if (!res.ok()) {
            std::cerr << "error: " << res.status().to_string() << "\n";
            return res.status().code() == si::ErrorCode::cancelled ? 130 : 1;
        }
        const auto& st = res.value().stats;
        std::cerr << "[sonder-infer] chat native=" << (backend_model.has_native_chat() ? "yes" : "no")
                  << " messages=" << file_messages.size() << " stop=" << si::to_string(st.stop_reason)
                  << " prompt_tokens=" << st.prompt_tokens << " completion_tokens=" << st.completion_tokens << "\n";
        return 0;
    }

    std::cerr << "[sonder-infer] chat with " << model->descriptor().name << " on " << model->backend_name()
              << (backend_model.has_native_chat() ? " (native chat)" : " (generic prompt format)")
              << "; /reset clears history, /exit or EOF quits\n";
    sonder::cli::ChatReplOptions ro;
    ro.system = system;
    return sonder::cli::run_chat_repl(std::cin, std::cout, std::cerr, ro,
                                      [&](const std::vector<si::ChatMessage>& history) -> si::Result<std::string> {
                                          auto res = turn(history);
                                          if (!res.ok()) return res.status();
                                          return std::move(res.value().text);
                                      });
}

#if defined(SONDER_HAS_BENCH)
int cmd_bench(const Args& a, Runtime& rt) {
    auto corpus_path = a.get("corpus");
    auto out_path = a.get("out");
    if (!corpus_path || !out_path) {
        std::cerr << "error: --corpus and --out are required\n";
        return 2;
    }
    auto corpus = si::bench::load_corpus(*corpus_path);
    if (!corpus.ok()) {
        std::cerr << "error: " << corpus.status().to_string() << "\n";
        return 2;
    }
    si::bench::Options opts;
    if (!sampling_from_args(a, opts.sampling)) return 2;
    opts.warmup_runs = std::atoi(a.get("warmup").value_or("1").c_str());
    opts.measured_runs = std::atoi(a.get("runs").value_or("3").c_str());
    opts.label = a.get("label").value_or("");
    auto model = load(a, rt);
    if (!model) return 1;
    auto doc = si::bench::run(*rt.engine, model, corpus.value(), opts, &std::cerr);
    std::ofstream out(*out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "error: cannot write " << *out_path << "\n";
        return 1;
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
            std::cerr << "error: cannot write " << md_path << "\n";
            return 1;
        }
        md << markdown;
        std::cout << markdown;
        std::cerr << "[bench] wrote " << md_path << "\n";
    }
    const auto* summary = doc.find("summary");
    std::cerr << "[bench] wrote " << *out_path << " summary=" << (summary ? summary->dump() : "{}") << "\n";
    const auto* failures = summary ? summary->find("failures") : nullptr;
    return failures && failures->as_int() > 0 ? 1 : 0;
}

#endif

}  // namespace

int main(int argc, char** argv) {
    std::string error;
    auto args = parse_args(argc, argv, error);
    if (!args) {
        std::cerr << "error: " << error << "\n\n" << kUsage;
        return 2;
    }
    const std::string& cmd = args->command;
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        std::cout << kUsage;
        return 0;
    }
    if (cmd == "version" || cmd == "--version") {
        std::cout << "sonder-infer " << si::version_string() << " (" << si::build_commit() << ") C ABI "
                  << SONDER_ABI_VERSION << "\n";
        return 0;
    }
    if (cmd == "devices") return cmd_devices();
    Runtime rt;
    if (!build_engine(*args, rt)) return 2;
    if (cmd == "backends") return cmd_backends(rt);
    if (cmd == "models") return cmd_models(*args, rt);
    if (cmd == "generate") return cmd_generate(*args, rt);
    if (cmd == "chat") return cmd_chat(*args, rt);
    if (cmd == "bench") {
#if defined(SONDER_HAS_BENCH)
        return cmd_bench(*args, rt);
#else
        std::cerr << "error: built without the bench module (bench/)\n";
        return 2;
#endif
    }
    std::cerr << "error: unknown command " << cmd << "\n\n" << kUsage;
    return 2;
}
