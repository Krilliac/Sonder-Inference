// sonder-bench: standalone benchmark runner (docs/BENCHMARK_PLAN.md).
// Writes <out-dir>/<stem>.json (sonder.inference.bench/1) and <stem>.md.
// `sonder-infer bench` runs the same harness (JSON, plus --markdown).
// Options, checked values and exit codes: docs/CLI.md "sonder-bench".
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "sonder/inference.hpp"
#include "sonder/inference/benchmark.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif

namespace si = sonder::inference;

namespace {

// Exit codes (docs/CLI.md "sonder-bench"): 0 ok, 1 a request failed or an
// output file could not be written, 2 usage error, 3 backend unreachable or
// model load failed, 4 --require-idle refused to run.
constexpr int kExitBackend = 3;
#if defined(SONDER_HAS_OLLAMA_BACKEND)
constexpr int kExitNotIdle = 4;
#endif

void usage(std::ostream& out) { out << si::bench::render_bench_cli_help(); }

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
    si::bench::BenchCliArgs parsed;
    std::string error;
    if (!si::bench::parse_bench_cli_args(args, parsed, error)) {
        return usage_error(error);
    }
    if (parsed.help) {
        usage(std::cout);
        return 0;
    }

    const std::string& backend_name = parsed.backend;
    const std::string& model_name = parsed.model;
    const std::string& corpus_path = parsed.corpus_path;
    const std::string& out_dir = parsed.out_dir;
    std::string stem = parsed.stem;
    const std::string& telemetry_path = parsed.telemetry_path;
    const bool list_models = parsed.list_models;
    const bool dry_run = parsed.dry_run;
    const si::bench::Options& opts = parsed.options;
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
        if (!parsed.ollama_url.empty()) oo.base_url = parsed.ollama_url;
        if (parsed.require_idle && !model_name.empty()) {
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
    else if (backend_name == "llamaserver") {
        si::bench::LlamaServerBenchOptions lso;
        lso.url = parsed.llamaserver_url;
        lso.config_path = parsed.llamaserver_config;
        auto made = si::bench::make_llamaserver_bench_backend(lso);
        if (!made.ok()) {
            return usage_error(made.status().to_string());
        }
        (void)engine.register_backend(std::move(made.value()));
    } else {
        return usage_error("unsupported backend '" + backend_name + "' (expected ollama, llamaserver or mock)");
    }

    auto backend = engine.find_backend(backend_name);
    if (!backend) {
        std::cerr << "error: backend '" << backend_name << "' was not registered\n";
        return kExitBackend;
    }
    auto version = backend->probe();
    if (!version.ok()) {
        std::cerr << "error: " << backend_name << " probe failed: " << version.status().to_string() << "\n";
        if (backend_name == "ollama") {
            std::cerr << "hint: start Ollama with 'ollama serve', or pass --ollama-url URL\n";
        } else if (backend_name == "llamaserver") {
            std::cerr << "hint: start llama-server on loopback, or pass --llamaserver-url URL / --llamaserver-config "
                         "PATH\n";
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
                      << (p.is_fanout() ? p.children.size() : static_cast<std::size_t>(opts.concurrency))
                      << " prompt_bytes=" << (p.is_fanout() ? p.children.front().size() : p.prompt.size())
                      << " max_tokens=" << p.max_tokens << "\n";
        }
        std::cout << "warmup " << opts.warmup_runs << ", runs " << opts.measured_runs << ", concurrency "
                  << opts.concurrency << ", gpu_snapshot " << (opts.capture_gpu_snapshot ? "on" : "off")
                  << ", budget " << opts.budget_seconds << " s\n";
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
