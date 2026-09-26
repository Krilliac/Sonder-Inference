// sonder-bench: standalone benchmark runner (docs/BENCHMARK_PLAN.md).
// Writes <out-dir>/<stem>.json (sonder.inference.bench/1) and <stem>.md.
// `sonder-infer bench` runs the same harness (JSON, plus --markdown).
// Options, checked values and exit codes: docs/CLI.md "sonder-bench".
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Header-only CLI helpers shared with sonder-infer (src/cli).
#include "../../src/cli/cli_spec.hpp"
#include "../../src/cli/cli_values.hpp"

#include "sonder/inference.hpp"
#include "sonder/inference/benchmark.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif

namespace si = sonder::inference;
namespace cli = sonder::cli;

namespace {

// Exit codes (docs/CLI.md "sonder-bench"): 0 ok, 1 a request failed or an
// output file could not be written, 2 usage error, 3 backend unreachable or
// model load failed, 4 --require-idle refused to run.
constexpr int kExitBackend = 3;
constexpr int kExitNotIdle = 4;

const cli::CommandSpec& bench_spec() {
    static const cli::CommandSpec spec{
        "",
        "standalone benchmark runner (docs/BENCHMARK_PLAN.md)",
        {"--model NAME [options]", "--backend ollama --list-models"},
        {{"Backend",
          {{"backend", cli::OptionKind::value, "NAME", "ollama | mock (default: ollama)"},
           {"ollama-url", cli::OptionKind::value, "URL",
            "Ollama base URL (default: http://127.0.0.1:11434, loopback only)"},
           {"model", cli::OptionKind::value, "NAME", "model to benchmark (required unless --list-models)"},
           {"require-idle", cli::OptionKind::flag, "",
            "ollama: refuse to run if a different model is resident\n(avoids competing with live traffic for the GPU)"},
           {"list-models", cli::OptionKind::flag, "", "list models on the backend and exit"}}},
         {"Run",
          {{"corpus", cli::OptionKind::value, "PATH", "corpus JSON (default: bench/corpus/baseline.json)"},
           {"prompts", cli::OptionKind::value, "a,b,c", "run only these prompt ids"},
           {"warmup", cli::OptionKind::value, "N", "discarded warmup runs per prompt, 0 to 100 (default: 1)"},
           {"runs", cli::OptionKind::value, "N", "measured runs per prompt, 1 to 1000 (default: 3)"},
           {"seed", cli::OptionKind::value, "N", "sampling seed (greedy decoding; default: 42)"},
           {"budget-seconds", cli::OptionKind::value, "S",
            "stop issuing requests after S seconds, 0 = unlimited (default: 540)"},
           {"dry-run", cli::OptionKind::flag, "", "print the plan and exit without sending requests"}}},
         {"Results",
          {{"hardware", cli::OptionKind::value, "TEXT",
            "hardware description recorded in results (GPU/VRAM/CPU/RAM/driver)"},
           {"label", cli::OptionKind::value, "TEXT", "free-form run label"},
           {"out-dir", cli::OptionKind::value, "DIR", "results directory (default: bench/results)"},
           {"stem", cli::OptionKind::value, "NAME", "output file stem (default: <date>-<backend>-<model>-<corpus>)"},
           {"telemetry", cli::OptionKind::value, "FILE", "also write Observatory telemetry JSONL"}}}},
        {"MOCK backend runs are harness checks only: stderr shows a MOCK BACKEND banner and the\n"
         "markdown summary is headed 'MOCK BACKEND: not a performance claim'.",
         "Exit status: 0 ok, 1 a request failed or an output file could not be written, 2 usage error,\n"
         "3 backend unreachable or model load failed, 4 --require-idle refused to run."},
        {"--backend ollama --list-models",
         "--backend ollama --model qwen3:0.6b --warmup 1 --runs 3 --require-idle --hardware \"RTX 4090\"",
         "--backend mock --model mock:tiny --warmup 0 --runs 1"}};
    return spec;
}

void usage(std::ostream& out) { out << cli::render_help("sonder-bench", bench_spec()); }

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int usage_error(const std::string& message) {
    std::cerr << "error: " << message << "\n(see 'sonder-bench --help')\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    if (args.empty()) {
        usage(std::cerr);
        return 2;
    }
    cli::ParsedCommand a;
    std::string error;
    const bool parsed = cli::parse_command(bench_spec(), args, a, error);
    if (parsed ? a.help : std::find(args.begin(), args.end(), "--help") != args.end() ||
                              std::find(args.begin(), args.end(), "-h") != args.end()) {
        usage(std::cout);
        return 0;
    }
    if (!parsed) return usage_error(error);

    const std::string backend_name = a.get("backend").value_or("ollama");
    const std::string ollama_url = a.get("ollama-url").value_or("");
    const std::string model_name = a.get("model").value_or("");
    const std::string corpus_path = a.get("corpus").value_or("bench/corpus/baseline.json");
    const std::string out_dir = a.get("out-dir").value_or("bench/results");
    std::string stem = a.get("stem").value_or("");
    const std::string telemetry_path = a.get("telemetry").value_or("");
    const bool list_models = a.has("list-models");
    const bool dry_run = a.has("dry-run");
    const bool require_idle = a.has("require-idle");
    si::bench::Options opts;
    opts.budget_seconds = 540.0;
    std::uint64_t seed = 42;
    if (auto v = a.get("prompts")) opts.prompt_ids = split_csv(*v);
    if (auto v = a.get("warmup"); v && !cli::parse_integer<int>("warmup", *v, 0, 100, opts.warmup_runs, error)) {
        return usage_error(error);
    }
    if (auto v = a.get("runs"); v && !cli::parse_integer<int>("runs", *v, 1, 1000, opts.measured_runs, error)) {
        return usage_error(error);
    }
    if (auto v = a.get("seed"); v && !cli::parse_u64("seed", *v, seed, error)) return usage_error(error);
    if (auto v = a.get("budget-seconds");
        v && !cli::parse_number("budget-seconds", *v, 0.0, 31536000.0, opts.budget_seconds, error)) {
        return usage_error(error);
    }
    opts.hardware = a.get("hardware").value_or("");
    opts.label = a.get("label").value_or("");
    opts.sampling = si::SamplingConfig::greedy(128, seed);
    if (backend_name == "mock") {
        std::cerr << "[sonder-bench] MOCK BACKEND - synthetic output, not a quality or performance signal\n";
    }

    si::EngineOptions eo;
    eo.sample_devices_on_start = false;
    if (!telemetry_path.empty()) {
        si::Status st;
        auto sink = si::make_jsonl_file_sink(telemetry_path, false, &st);
        if (!st.ok() || !sink) {
            std::cerr << "error: telemetry sink: " << st.to_string() << "\n";
            return 1;
        }
        eo.telemetry_sinks.push_back(std::shared_ptr<si::TelemetrySink>(std::move(sink)));
    }
    si::Engine engine(std::move(eo));

    if (backend_name == "mock") {
        si::MockBackendOptions mo;
        mo.token_delay = std::chrono::microseconds(500);
        mo.default_completion_tokens = 48;
        (void)engine.register_backend(si::make_mock_backend(mo));
    }
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    else if (backend_name == "ollama") {
        si::OllamaBackendOptions oo;
        if (!ollama_url.empty()) oo.base_url = ollama_url;
        if (require_idle && !model_name.empty()) {
            si::ollama::OllamaClient client(oo);
            auto running = client.running_models();
            if (!running.ok()) {
                std::cerr << "error: ollama unreachable: " << running.status().to_string() << "\n";
                return kExitBackend;
            }
            for (const auto& m : running.value()) {
                if (m.name != model_name && m.name != model_name + ":latest") {
                    std::cerr << "refusing to run: ollama has '" << m.name
                              << "' resident (possible live traffic); omit --require-idle to override\n";
                    return kExitNotIdle;
                }
            }
        }
        (void)engine.register_backend(si::make_ollama_backend(oo));
    }
#endif
    else {
        return usage_error("unsupported backend '" + backend_name + "' (expected ollama or mock)");
    }

    auto backend = engine.find_backend(backend_name);
    auto version = backend->probe();
    if (!version.ok()) {
        std::cerr << "error: " << backend_name << " probe failed: " << version.status().to_string() << "\n";
        if (backend_name == "ollama") {
            std::cerr << "hint: start Ollama with 'ollama serve', or pass --ollama-url URL\n";
        }
        return kExitBackend;
    }
    if (list_models) {
        auto models = backend->list_models();
        if (!models.ok()) {
            std::cerr << "error: " << models.status().to_string() << "\n";
            return kExitBackend;
        }
        for (const auto& m : models.value()) {
            std::cout << m.name << "\t" << m.parameter_size << "\t" << m.quantization << "\t" << m.size_bytes << "\n";
        }
        return 0;
    }
    if (model_name.empty()) {
        std::cerr << "error: --model is required\n";
        return 2;
    }
    auto corpus = si::bench::load_corpus(corpus_path);
    if (!corpus.ok()) {
        std::cerr << "error: " << corpus.status().to_string() << "\n";
        return 2;
    }
    if (dry_run) {
        std::cout << backend_name << " " << version.value() << ", model " << model_name << ", corpus "
                  << corpus.value().name << " v" << corpus.value().version << "\n";
        for (const auto& p : corpus.value().prompts) {
            if (!opts.prompt_ids.empty() &&
                std::find(opts.prompt_ids.begin(), opts.prompt_ids.end(), p.id) == opts.prompt_ids.end()) {
                continue;
            }
            std::cout << "  " << p.id << " [" << p.workload << "] requests/iteration="
                      << (p.is_fanout() ? p.children.size() : 1u)
                      << " prompt_bytes=" << (p.is_fanout() ? p.children.front().size() : p.prompt.size())
                      << " max_tokens=" << p.max_tokens << "\n";
        }
        std::cout << "warmup " << opts.warmup_runs << ", runs " << opts.measured_runs << ", budget "
                  << opts.budget_seconds << " s\n";
        return 0;
    }

    si::ModelLoadOptions lo;
    lo.model = model_name;
    auto model = engine.load_model(backend_name, lo);
    if (!model.ok()) {
        std::cerr << "error: load_model: " << model.status().to_string() << "\n";
        return kExitBackend;
    }
    const si::json::Value doc = si::bench::run(engine, model.value(), corpus.value(), opts, &std::cerr);
    if (stem.empty()) stem = si::bench::default_result_stem(doc);
    std::string jp, mp;
    auto st = si::bench::write_results(doc, out_dir, stem, &jp, &mp);
    if (!st.ok()) {
        std::cerr << "error: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << si::bench::render_markdown(doc) << "\nwrote " << jp << "\n      " << mp << "\n";
    const si::json::Value* failures = doc.find("summary") ? doc.find("summary")->find("failures") : nullptr;
    return failures && failures->as_int() > 0 ? 1 : 0;
}
