// sonder-bench: standalone benchmark runner (docs/BENCHMARK_PLAN.md).
// Writes <out-dir>/<stem>.json (sonder.inference.bench/1) and <stem>.md.
// `sonder-infer bench` runs the same harness but writes JSON only.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference.hpp"
#include "sonder/inference/benchmark.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif

namespace si = sonder::inference;

namespace {

void usage() {
    std::cout << R"(usage: sonder-bench --model NAME [options]

  --backend NAME        ollama | mock (default: ollama)
  --ollama-url URL      Ollama base URL (default: http://127.0.0.1:11434, loopback only)
  --model NAME          model to benchmark (required unless --list-models)
  --corpus PATH         corpus JSON (default: bench/corpus/baseline.json)
  --prompts a,b,c       run only these prompt ids
  --warmup N            discarded warmup runs per prompt (default: 1)
  --runs N              measured runs per prompt (default: 3)
  --seed N              sampling seed (greedy decoding; default: 42)
  --budget-seconds S    stop issuing requests after S seconds (default: 540)
  --hardware TEXT       hardware description recorded in results (GPU/VRAM/CPU/RAM/driver)
  --label TEXT          free-form run label
  --out-dir DIR         results directory (default: bench/results)
  --stem NAME           output file stem (default: <date>-<backend>-<model>-<corpus>)
  --telemetry FILE      also write Observatory telemetry JSONL
  --require-idle        ollama: refuse to run if a different model is resident
                        (avoids competing with live traffic for the GPU)
  --list-models         list models on the backend and exit
  --dry-run             print the plan and exit without sending requests
  -h, --help
)";
}

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

}  // namespace

int main(int argc, char** argv) {
    std::string backend_name = "ollama";
    std::string ollama_url;
    std::string model_name;
    std::string corpus_path = "bench/corpus/baseline.json";
    std::string out_dir = "bench/results";
    std::string stem;
    std::string telemetry_path;
    bool list_models = false;
    bool dry_run = false;
    bool require_idle = false;
    si::bench::Options opts;
    opts.budget_seconds = 540.0;
    std::uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for " << a << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--backend") {
            backend_name = next();
        } else if (a == "--ollama-url") {
            ollama_url = next();
        } else if (a == "--model") {
            model_name = next();
        } else if (a == "--corpus") {
            corpus_path = next();
        } else if (a == "--prompts") {
            opts.prompt_ids = split_csv(next());
        } else if (a == "--warmup") {
            opts.warmup_runs = std::atoi(next().c_str());
        } else if (a == "--runs") {
            opts.measured_runs = std::atoi(next().c_str());
        } else if (a == "--seed") {
            seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (a == "--budget-seconds") {
            opts.budget_seconds = std::atof(next().c_str());
        } else if (a == "--hardware") {
            opts.hardware = next();
        } else if (a == "--label") {
            opts.label = next();
        } else if (a == "--out-dir") {
            out_dir = next();
        } else if (a == "--stem") {
            stem = next();
        } else if (a == "--telemetry") {
            telemetry_path = next();
        } else if (a == "--require-idle") {
            require_idle = true;
        } else if (a == "--list-models") {
            list_models = true;
        } else if (a == "--dry-run") {
            dry_run = true;
        } else {
            std::cerr << "error: unknown option " << a << "\n";
            usage();
            return 2;
        }
    }
    opts.sampling = si::SamplingConfig::greedy(128, seed);

    si::EngineOptions eo;
    eo.sample_devices_on_start = false;
    if (!telemetry_path.empty()) {
        si::Status st;
        auto sink = si::make_jsonl_file_sink(telemetry_path, false, &st);
        if (!st.ok() || !sink) {
            std::cerr << "error: telemetry sink: " << st.to_string() << "\n";
            return 2;
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
                return 3;
            }
            for (const auto& m : running.value()) {
                if (m.name != model_name && m.name != model_name + ":latest") {
                    std::cerr << "refusing to run: ollama has '" << m.name
                              << "' resident (possible live traffic); omit --require-idle to override\n";
                    return 4;
                }
            }
        }
        (void)engine.register_backend(si::make_ollama_backend(oo));
    }
#endif
    else {
        std::cerr << "error: unsupported backend '" << backend_name << "'\n";
        return 2;
    }

    auto backend = engine.find_backend(backend_name);
    auto version = backend->probe();
    if (!version.ok()) {
        std::cerr << "error: " << backend_name << " probe failed: " << version.status().to_string() << "\n";
        return 3;
    }
    if (list_models) {
        auto models = backend->list_models();
        if (!models.ok()) {
            std::cerr << "error: " << models.status().to_string() << "\n";
            return 3;
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
        return 3;
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
