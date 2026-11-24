#include "sonder/inference/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include "cli/cli_spec.hpp"
#include "cli/cli_values.hpp"

namespace sonder::inference::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string fnv1a_hex(std::string_view s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Expanded prompt text limits (filler repeats are multiplied into every
// fan-out child, so a tiny corpus could otherwise expand to many GB).
constexpr std::size_t kMaxPromptBytes = 16u << 20;   // one prompt incl. all children
constexpr std::size_t kMaxCorpusBytes = 64u << 20;   // whole corpus after expansion

std::string str_or(const json::Value& v, std::string_view key, std::string fallback = {}) {
    const json::Value* f = v.find(key);
    return (f != nullptr && f->is_string()) ? f->as_string() : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// Corpus

Result<Corpus> parse_corpus(std::string_view json_text) {
    if (json_text.size() > kMaxCorpusFileBytes) {
        return Status(ErrorCode::invalid_argument, "corpus is too large (over 64 MiB of JSON)");
    }
    auto parsed = json::parse(json_text);
    if (!parsed.ok()) {
        return parsed.status();
    }
    const json::Value& root = parsed.value();
    if (!root.is_object()) {
        return Status(ErrorCode::invalid_argument, "corpus must be a JSON object");
    }
    const json::Value* schema = root.find("schema");
    if (!schema || schema->as_string() != kCorpusSchema) {
        return Status(ErrorCode::invalid_argument, "corpus schema must be " + std::string(kCorpusSchema));
    }
    Corpus corpus;
    corpus.name = str_or(root, "name", "unnamed");
    corpus.description = str_or(root, "description");
    corpus.version = str_or(root, "version");
    corpus.fnv1a = fnv1a_hex(json_text);

    std::map<std::string, std::string, std::less<>> fillers;
    if (const json::Value* f = root.find("fillers")) {
        for (const auto& [name, text] : f->as_object()) {
            fillers[name] = text.as_string();
        }
    }

    const json::Value* prompts = root.find("prompts");
    if (!prompts || !prompts->is_array() || prompts->as_array().empty()) {
        return Status(ErrorCode::invalid_argument, "corpus needs a non-empty prompts array");
    }
    std::size_t corpus_bytes = 0;
    for (const auto& p : prompts->as_array()) {
        Prompt pr;
        pr.id = str_or(p, "id");
        pr.workload = str_or(p, "workload", "interactive_chat");
        if (pr.id.empty()) {
            return Status(ErrorCode::invalid_argument, "every prompt needs non-empty id and prompt");
        }
        // Range-check the 64-bit value before narrowing (4294967297 must not become 1).
        const std::int64_t max_tokens = p.find("max_tokens") ? p.find("max_tokens")->as_int(64) : 64;
        if (max_tokens < 1 || max_tokens > SamplingConfig::kMaxTokensLimit) {
            return Status(ErrorCode::invalid_argument, "prompt " + pr.id + " has invalid max_tokens");
        }
        pr.max_tokens = static_cast<std::int32_t>(max_tokens);
        const auto too_large = [&] {
            return Status(ErrorCode::invalid_argument,
                          "prompt " + pr.id + " expands beyond the corpus size limit (16 MiB per prompt, 64 MiB total)");
        };

        // Prefix = shared_prefix + repeated filler context.
        std::string prefix = str_or(p, "shared_prefix");
        if (const json::Value* ctx = p.find("context")) {
            const std::string name = str_or(*ctx, "filler");
            auto it = fillers.find(name);
            if (it == fillers.end()) {
                return Status(ErrorCode::invalid_argument, "prompt " + pr.id + " references unknown filler '" + name + "'");
            }
            const std::int64_t repeat = ctx->find("repeat") ? ctx->find("repeat")->as_int(1) : 1;
            if (repeat < 1 || repeat > 10000) {
                return Status(ErrorCode::invalid_argument, "prompt " + pr.id + " has invalid context.repeat");
            }
            for (std::int64_t i = 0; i < repeat; ++i) {
                if (prefix.size() + 2 + it->second.size() > kMaxPromptBytes) {
                    return too_large();
                }
                if (!prefix.empty()) prefix += "\n\n";
                prefix += it->second;
            }
        }
        auto with_prefix = [&](const std::string& tail) {
            if (prefix.empty()) return tail;
            if (tail.empty()) return prefix;
            return prefix + "\n\n" + tail;
        };

        // Size of with_prefix(tail) without building it.
        const auto expanded_size = [&](std::size_t tail) {
            if (prefix.empty()) return tail;
            if (tail == 0) return prefix.size();
            return prefix.size() + 2 + tail;
        };
        std::size_t prompt_bytes = 0;
        if (const json::Value* children = p.find("children"); children && !children->as_array().empty()) {
            for (const auto& c : children->as_array()) {
                if (c.as_string().empty()) {
                    return Status(ErrorCode::invalid_argument, "prompt " + pr.id + " has an empty child prompt");
                }
                prompt_bytes += expanded_size(c.as_string().size());
                if (prompt_bytes > kMaxPromptBytes) {
                    return too_large();
                }
            }
            prompt_bytes += expanded_size(str_or(p, "prompt").size());
            if (prompt_bytes > kMaxPromptBytes || corpus_bytes + prompt_bytes > kMaxCorpusBytes) {
                return too_large();
            }
            for (const auto& c : children->as_array()) {
                pr.children.push_back(with_prefix(c.as_string()));
            }
            pr.shared_prefix_bytes = prefix.size();
            pr.prompt = with_prefix(str_or(p, "prompt"));
        } else {
            const std::string text = str_or(p, "prompt");
            if (text.empty()) {
                return Status(ErrorCode::invalid_argument, "every prompt needs non-empty id and prompt");
            }
            prompt_bytes = expanded_size(text.size());
            if (prompt_bytes > kMaxPromptBytes || corpus_bytes + prompt_bytes > kMaxCorpusBytes) {
                return too_large();
            }
            pr.prompt = with_prefix(text);
        }
        corpus_bytes += prompt_bytes;
        corpus.prompts.push_back(std::move(pr));
    }
    return corpus;
}

Result<Corpus> load_corpus(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status(ErrorCode::io_error, "cannot open corpus: " + path);
    }
    // Bound the read before allocating: the file size is checked first, and
    // the read itself stops one byte past the limit (a file that grows while
    // being read, or a size the stream cannot report).
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size > static_cast<std::streamoff>(kMaxCorpusFileBytes)) {
        return Status(ErrorCode::invalid_argument, "corpus is too large (over 64 MiB): " + path);
    }
    in.clear();
    in.seekg(0, std::ios::beg);
    std::string text;
    char buf[64 * 1024];
    while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
        text.append(buf, static_cast<std::size_t>(in.gcount()));
        if (text.size() > kMaxCorpusFileBytes) {
            return Status(ErrorCode::invalid_argument, "corpus is too large (over 64 MiB): " + path);
        }
    }
    return parse_corpus(text);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    p = std::clamp(p, 0.0, 100.0);
    const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(rank);
    const auto hi = std::min(lo + 1, values.size() - 1);
    const double frac = rank - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * frac;
}

// ---------------------------------------------------------------------------
// Runner

namespace {

json::Object distribution(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    return json::Object{{"n", v.size()},
                        {"mean", v.empty() ? 0.0 : sum / static_cast<double>(v.size())},
                        {"p50", percentile(v, 50)},
                        {"p95", percentile(v, 95)},
                        {"p99", percentile(v, 99)},
                        {"min", v.empty() ? 0.0 : *std::min_element(v.begin(), v.end())},
                        {"max", v.empty() ? 0.0 : *std::max_element(v.begin(), v.end())}};
}

struct Metrics {
    std::vector<double> ttft, total, decode_tps, client_decode_tps, prompt_tps, completion_tokens;
    std::vector<double> makespan, aggregate_tps;  // fan-out batches
    std::uint64_t requests = 0;
    std::uint64_t failures = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t total_completion_tokens = 0;
    double measured_wall_ms = 0.0;
};

struct Outcome {
    bool completed = false;
    bool cancelled = false;
    bool failed = false;
    double ttft = -1, total = 0, dtps = 0, client_dtps = 0, ptps = 0;
    std::uint64_t completion_tokens = 0;
    double load_ms = 0;
};

constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;

json::Value reading_mib_json(const MemoryReading& reading) {
    if (!reading.is_known()) {
        return json::Value(nullptr);
    }
    return json::Value(reading.bytes_or(0) / kBytesPerMiB);
}

json::Value optional_u64_json(std::optional<std::uint64_t> value) {
    if (!value) {
        return json::Value(nullptr);
    }
    return json::Value(*value);
}

const GpuDeviceInfo* find_matching_gpu(const std::vector<GpuDeviceInfo>& baseline, const GpuDeviceInfo& gpu,
                                       std::size_t index) {
    if (!gpu.id.empty()) {
        for (const auto& candidate : baseline) {
            if (candidate.id == gpu.id) {
                return &candidate;
            }
        }
    }
    if (index < baseline.size() && baseline[index].id == gpu.id && baseline[index].name == gpu.name) {
        return &baseline[index];
    }
    return nullptr;
}

// Coordinates simultaneous release of all worker threads in a concurrent batch.
// Uses mutex + condition_variable for portability across toolchains and TSan,
// and supports abort() so a partial thread-spawn failure can never deadlock.
struct StartLatch {
    std::mutex mu;
    std::condition_variable cv;
    std::size_t ready = 0;
    bool go = false;
    bool aborted = false;

    bool arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mu);
        ++ready;
        cv.notify_all();
        cv.wait(lock, [&] { return go || aborted; });
        return go && !aborted;
    }

    Clock::time_point wait_for_ready_and_release(std::size_t expected) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return ready >= expected || aborted; });
        const auto start_time = Clock::now();
        if (!aborted) {
            go = true;
        }
        cv.notify_all();
        return start_time;
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mu);
        aborted = true;
        cv.notify_all();
    }
};

struct BatchThreadGuard {
    StartLatch& latch;
    std::vector<std::thread>& threads;
    bool released = false;

    BatchThreadGuard(StartLatch& l, std::vector<std::thread>& t) noexcept : latch(l), threads(t) {}
    BatchThreadGuard(const BatchThreadGuard&) = delete;
    BatchThreadGuard& operator=(const BatchThreadGuard&) = delete;

    ~BatchThreadGuard() noexcept {
        if (!released) {
            try {
                latch.abort();
            } catch (...) {
            }
        }
        for (auto& thr : threads) {
            if (thr.joinable()) {
                try {
                    thr.join();
                } catch (...) {
                }
            }
        }
    }
};

json::Object make_failed_row(const Prompt& prompt, std::int64_t iteration, int child, int worker_index, bool warmup,
                             const std::string& error_message) {
    json::Object row{{"prompt_id", prompt.id}, {"workload", prompt.workload}, {"iteration", iteration},
                     {"warmup", warmup}};
    if (child >= 0) {
        row.set("child", child);
    }
    if (worker_index >= 0) {
        row.set("worker_index", worker_index);
    }
    row.set("outcome", "failed");
    row.set("error", error_message);
    return row;
}

json::Object execute(Engine& engine, const std::shared_ptr<Model>& model, const Prompt& prompt,
                     const std::string& text, std::int64_t iteration, int child, int worker_index, bool warmup,
                     const Options& options, Outcome& o) {
    SamplingConfig sampling = options.sampling;
    sampling.max_tokens = prompt.max_tokens;
    SessionOptions so;
    so.sampling = sampling;
    if (options.run_id) {
        so.run_id = options.run_id;
    } else if (!options.label.empty()) {
        so.run_id = options.label;
    }
    so.agent_id = options.agent_id;
    so.task_id = options.task_id;
    so.workload = options.workload;
    so.priority = options.priority;
    json::Object row{{"prompt_id", prompt.id}, {"workload", prompt.workload}, {"iteration", iteration},
                     {"warmup", warmup}};
    if (child >= 0) {
        row.set("child", child);
    }
    if (worker_index >= 0) {
        row.set("worker_index", worker_index);
    }
    auto session = engine.create_session(model, so);
    if (!session.ok()) {
        row.set("outcome", "failed");
        row.set("error", session.status().to_string());
        o.failed = true;
        return row;
    }
    auto res = session.value()->generate(text);
    if (!res.ok()) {
        row.set("outcome", "failed");
        row.set("error", res.status().to_string());
        o.failed = true;
        return row;
    }
    const auto& r = res.value();
    const auto& s = r.stats;
    row.set("outcome", to_string(r.outcome));
    row.set("stop_reason", to_string(s.stop_reason));
    row.set("ttft_ms", r.ttft_ms);
    row.set("total_ms", r.total_ms);
    row.set("prompt_tokens", s.prompt_tokens);
    row.set("completion_tokens", s.completion_tokens);
    row.set("chunks", s.chunks);
    row.set("token_counts_from_backend", s.token_counts_from_backend);
    row.set("output_bytes", r.text.size());
    double client = 0;
    if (r.ttft_ms >= 0 && r.total_ms > r.ttft_ms && s.completion_tokens > 1) {
        client = static_cast<double>(s.completion_tokens - 1) * 1000.0 / (r.total_ms - r.ttft_ms);
    }
    double dtps = 0;
    if (s.eval_ns > 0 && s.completion_tokens > 0) {
        dtps = static_cast<double>(s.completion_tokens) * 1e9 / static_cast<double>(s.eval_ns);
        row.set("decode_tokens_per_sec_source", "backend_eval_duration");
    } else if (client > 0) {
        dtps = client;
        row.set("decode_tokens_per_sec_source", "wall_clock_after_first_chunk");
    }
    row.set("decode_tokens_per_sec", dtps);
    row.set("client_decode_tokens_per_sec", client);
    double ptps = 0;
    if (s.prompt_eval_ns > 0 && s.prompt_tokens > 0) {
        ptps = static_cast<double>(s.prompt_tokens) * 1e9 / static_cast<double>(s.prompt_eval_ns);
    }
    row.set("prompt_tokens_per_sec", ptps);
    row.set("backend_load_ms", static_cast<double>(s.load_ns) / 1e6);

    o.completed = r.outcome == RequestOutcome::completed;
    o.cancelled = r.outcome == RequestOutcome::cancelled;
    o.ttft = r.ttft_ms;
    o.total = r.total_ms;
    o.dtps = dtps;
    o.client_dtps = client;
    o.ptps = ptps;
    o.completion_tokens = s.completion_tokens;
    o.load_ms = static_cast<double>(s.load_ns) / 1e6;
    return row;
}

void record(Metrics& m, const Outcome& o) {
    ++m.requests;
    if (o.failed) {
        ++m.failures;
        return;
    }
    if (o.cancelled) {
        ++m.cancelled;
        return;
    }
    if (!o.completed) {
        return;
    }
    if (o.ttft >= 0) m.ttft.push_back(o.ttft);
    m.total.push_back(o.total);
    if (o.dtps > 0) m.decode_tps.push_back(o.dtps);
    if (o.client_dtps > 0) m.client_decode_tps.push_back(o.client_dtps);
    if (o.ptps > 0) m.prompt_tps.push_back(o.ptps);
    m.completion_tokens.push_back(static_cast<double>(o.completion_tokens));
}

const sonder::cli::CommandSpec& bench_cli_spec() {
    namespace cli = sonder::cli;
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
           {"concurrency", cli::OptionKind::value, "N",
            "parallel worker sessions per measured run, 1 to 64 (default: 1)"},
           {"seed", cli::OptionKind::value, "N", "sampling seed (greedy decoding; default: 42)"},
           {"budget-seconds", cli::OptionKind::value, "S",
            "stop issuing requests after S seconds, 0 = unlimited (default: 540)"},
           {"no-gpu-snapshot", cli::OptionKind::flag, "",
            "skip before/after GPU VRAM snapshot enumeration"},
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
         "--backend mock --model mock:tiny --warmup 0 --runs 1 --concurrency 4"}};
    return spec;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  // namespace

json::Array gpu_snapshot_to_json(const std::vector<GpuDeviceInfo>& gpus, const std::vector<GpuDeviceInfo>* before) {
    json::Array out;
    out.reserve(gpus.size());
    for (std::size_t idx = 0; idx < gpus.size(); ++idx) {
        const auto& g = gpus[idx];
        const auto eff_total = gpu_effective_vram_mib(g, false);
        std::optional<std::uint64_t> eff_free;
        if (g.dedicated_free.is_known()) {
            eff_free = gpu_effective_vram_mib(g, true);
        }
        json::Object entry{{"id", g.id},
                           {"name", g.name},
                           {"vendor", to_string(g.vendor)},
                           {"dedicated_total_mib", reading_mib_json(g.dedicated_total)},
                           {"dedicated_free_mib", reading_mib_json(g.dedicated_free)},
                           {"effective_vram_mib", optional_u64_json(eff_total)},
                           {"effective_free_vram_mib", optional_u64_json(eff_free)},
                           {"is_integrated_uma", g.is_integrated_uma},
                           {"backend_hint", g.backend_hint},
                           {"source", g.source}};
        if (before != nullptr && eff_free) {
            if (const GpuDeviceInfo* prev = find_matching_gpu(*before, g, idx);
                prev != nullptr && prev->dedicated_free.is_known()) {
                const auto prev_free = gpu_effective_vram_mib(*prev, true);
                if (prev_free && *prev_free > *eff_free) {
                    entry.set("peak_vram_delta_mib", *prev_free - *eff_free);
                }
            }
        }
        out.emplace_back(std::move(entry));
    }
    return out;
}

std::string render_bench_cli_help() { return sonder::cli::render_help("sonder-bench", bench_cli_spec()); }

bool parse_bench_cli_args(const std::vector<std::string>& args, BenchCliArgs& out, std::string& error) {
    namespace cli = sonder::cli;
    out = BenchCliArgs{};
    out.options.budget_seconds = 540.0;
    error.clear();

    cli::ParsedCommand a;
    const bool parsed = cli::parse_command(bench_cli_spec(), args, a, error);
    if (parsed ? a.help
               : std::find(args.begin(), args.end(), "--help") != args.end() ||
                     std::find(args.begin(), args.end(), "-h") != args.end()) {
        out.help = true;
        error.clear();
        return true;
    }
    if (!parsed) {
        return false;
    }

    out.backend = a.get("backend").value_or("ollama");
    out.ollama_url = a.get("ollama-url").value_or("");
    out.model = a.get("model").value_or("");
    out.corpus_path = a.get("corpus").value_or("bench/corpus/baseline.json");
    out.out_dir = a.get("out-dir").value_or("bench/results");
    out.stem = a.get("stem").value_or("");
    out.telemetry_path = a.get("telemetry").value_or("");
    out.list_models = a.has("list-models");
    out.dry_run = a.has("dry-run");
    out.require_idle = a.has("require-idle");
    out.options.capture_gpu_snapshot = !a.has("no-gpu-snapshot");

    if (auto v = a.get("prompts")) {
        out.options.prompt_ids = split_csv(*v);
    }
    if (auto v = a.get("warmup"); v && !cli::parse_integer<int>("warmup", *v, 0, 100, out.options.warmup_runs, error)) {
        return false;
    }
    if (auto v = a.get("runs"); v && !cli::parse_integer<int>("runs", *v, 1, 1000, out.options.measured_runs, error)) {
        return false;
    }
    if (auto v = a.get("concurrency");
        v && !cli::parse_integer<int>("concurrency", *v, kMinConcurrency, kMaxConcurrency, out.options.concurrency,
                                      error)) {
        return false;
    }
    if (auto v = a.get("seed"); v && !cli::parse_u64("seed", *v, out.seed, error)) {
        return false;
    }
    if (auto v = a.get("budget-seconds");
        v && !cli::parse_number("budget-seconds", *v, 0.0, 31536000.0, out.options.budget_seconds, error)) {
        return false;
    }
    out.options.hardware = a.get("hardware").value_or("");
    out.options.label = a.get("label").value_or("");
    out.options.sampling = SamplingConfig::greedy(128, out.seed);
    return true;
}

json::Object run(Engine& engine, const std::shared_ptr<Model>& model, const Corpus& corpus, const Options& options,
                 std::ostream* progress) {
    const auto t0 = Clock::now();
    const int concurrency = std::clamp(options.concurrency, kMinConcurrency, kMaxConcurrency);
    const auto& d = model->descriptor();
    auto backend = engine.find_backend(model->backend_name());
    std::string backend_version = "unknown";
    std::string backend_description;
    if (backend) {
        backend_description = backend->description();
        if (auto v = backend->probe(); v.ok()) {
            backend_version = v.value();
        }
    }

    json::Array host_devices;
    for (const auto& dev : engine.devices()) {
        host_devices.emplace_back(json::Object{{"id", dev.id},
                                               {"kind", to_string(dev.kind)},
                                               {"name", dev.name},
                                               {"logical_cores", dev.logical_cores},
                                               {"total_memory_bytes", dev.total_memory_bytes}});
    }

    std::vector<GpuDeviceInfo> gpus_before;
    if (options.capture_gpu_snapshot) {
        gpus_before = enumerate_gpu_devices();
    }

    json::Array runs;
    json::Array batches;
    json::Array per_prompt;
    Metrics global;
    bool truncated = false;
    double cold_first_ms = -1;
    double cold_load_ms = 0;
    auto over_budget = [&] { return options.budget_seconds > 0 && ms_since(t0) > options.budget_seconds * 1000.0; };

    for (const auto& prompt : corpus.prompts) {
        if (!options.prompt_ids.empty() &&
            std::find(options.prompt_ids.begin(), options.prompt_ids.end(), prompt.id) == options.prompt_ids.end()) {
            continue;
        }
        Metrics pm;
        const std::int64_t warmups = std::max<std::int64_t>(0, options.warmup_runs);
        const std::int64_t measured = std::max<std::int64_t>(1, options.measured_runs);
        const std::int64_t total_runs = warmups + measured;
        for (std::int64_t i = 0; i < total_runs; ++i) {
            if (over_budget()) {
                truncated = true;
                break;
            }
            const bool warmup = i < warmups;
            const bool concurrent_non_fanout = !prompt.is_fanout() && !warmup && concurrency > 1;
            const std::size_t batch_size =
                prompt.is_fanout() ? prompt.children.size()
                                   : (concurrent_non_fanout ? static_cast<std::size_t>(concurrency) : std::size_t{1});
            const auto text_for = [&](std::size_t slot) -> const std::string& {
                return prompt.is_fanout() ? prompt.children[slot] : prompt.prompt;
            };
            std::vector<json::Object> rows(batch_size);
            std::vector<Outcome> outcomes(batch_size);
            Clock::time_point b0 = Clock::now();
            if (batch_size == 1) {
                const int child_idx = prompt.is_fanout() ? 0 : -1;
                try {
                    rows[0] = execute(engine, model, prompt, text_for(0), i, child_idx, -1, warmup, options, outcomes[0]);
                } catch (const std::exception& ex) {
                    outcomes[0] = Outcome{};
                    outcomes[0].failed = true;
                    rows[0] = make_failed_row(prompt, i, child_idx, -1, warmup, ex.what());
                } catch (...) {
                    outcomes[0] = Outcome{};
                    outcomes[0].failed = true;
                    rows[0] = make_failed_row(prompt, i, child_idx, -1, warmup, "unknown exception");
                }
            } else {
                StartLatch latch;
                std::vector<std::thread> threads;
                threads.reserve(batch_size);
                BatchThreadGuard guard{latch, threads};
                std::string spawn_error;
                try {
                    for (std::size_t c = 0; c < batch_size; ++c) {
                        const int child_idx = prompt.is_fanout() ? static_cast<int>(c) : -1;
                        const int worker_idx = concurrent_non_fanout ? static_cast<int>(c) : -1;
                        threads.emplace_back([&, c, child_idx, worker_idx] {
                            try {
                                if (!latch.arrive_and_wait()) {
                                    outcomes[c] = Outcome{};
                                    outcomes[c].failed = true;
                                    rows[c] = make_failed_row(prompt, i, child_idx, worker_idx, warmup,
                                                              "concurrent batch aborted before start");
                                    return;
                                }
                                rows[c] = execute(engine, model, prompt, text_for(c), i, child_idx, worker_idx, warmup,
                                                  options, outcomes[c]);
                            } catch (const std::exception& ex) {
                                outcomes[c] = Outcome{};
                                outcomes[c].failed = true;
                                try {
                                    rows[c] = make_failed_row(prompt, i, child_idx, worker_idx, warmup, ex.what());
                                } catch (...) {
                                }
                            } catch (...) {
                                outcomes[c] = Outcome{};
                                outcomes[c].failed = true;
                                try {
                                    rows[c] = make_failed_row(prompt, i, child_idx, worker_idx, warmup,
                                                              "unknown exception");
                                } catch (...) {
                                }
                            }
                        });
                    }
                    b0 = latch.wait_for_ready_and_release(threads.size());
                    guard.released = true;
                } catch (const std::exception& ex) {
                    latch.abort();
                    spawn_error = std::string("failed to spawn worker thread: ") + ex.what();
                } catch (...) {
                    latch.abort();
                    spawn_error = "failed to spawn worker thread: unknown exception";
                }
                for (auto& thr : threads) {
                    if (thr.joinable()) {
                        thr.join();
                    }
                }
                if (!spawn_error.empty()) {
                    for (std::size_t c = threads.size(); c < batch_size; ++c) {
                        const int child_idx = prompt.is_fanout() ? static_cast<int>(c) : -1;
                        const int worker_idx = concurrent_non_fanout ? static_cast<int>(c) : -1;
                        outcomes[c] = Outcome{};
                        outcomes[c].failed = true;
                        rows[c] = make_failed_row(prompt, i, child_idx, worker_idx, warmup, spawn_error);
                    }
                }
            }
            const double makespan = ms_since(b0);
            if (cold_first_ms < 0 && !outcomes.empty()) {
                cold_first_ms = outcomes[0].total;
                cold_load_ms = outcomes[0].load_ms;
            }
            std::uint64_t batch_tokens = 0;
            std::uint64_t batch_completed_tokens = 0;
            std::uint64_t batch_failures = 0;
            for (std::size_t c = 0; c < batch_size; ++c) {
                if (!warmup) {
                    record(pm, outcomes[c]);
                    record(global, outcomes[c]);
                }
                batch_tokens += outcomes[c].completion_tokens;
                if (outcomes[c].completed) {
                    batch_completed_tokens += outcomes[c].completion_tokens;
                }
                batch_failures += outcomes[c].failed ? 1u : 0u;
                if (progress) {
                    *progress << "[bench] " << prompt.id << " #" << i;
                    if (prompt.is_fanout()) {
                        *progress << "." << c;
                    } else if (concurrent_non_fanout) {
                        *progress << " [w" << c << "]";
                    }
                    const json::Value* out_v = rows[c].find("outcome");
                    *progress << (warmup ? " (warmup)" : "") << ": "
                              << (out_v != nullptr ? out_v->as_string() : std::string("failed"));
                    if (const json::Value* tv = rows[c].find("ttft_ms")) *progress << " ttft_ms=" << tv->as_double();
                    if (const json::Value* tv = rows[c].find("total_ms")) *progress << " total_ms=" << tv->as_double();
                    if (const json::Value* tv = rows[c].find("decode_tokens_per_sec")) {
                        *progress << " decode_tps=" << tv->as_double();
                    }
                    if (const json::Value* ev = rows[c].find("error")) *progress << " error=" << ev->as_string();
                    *progress << "\n";
                }
                runs.emplace_back(std::move(rows[c]));
            }
            if (!warmup) {
                pm.measured_wall_ms += makespan;
                pm.total_completion_tokens += batch_completed_tokens;
                global.measured_wall_ms += makespan;
                global.total_completion_tokens += batch_completed_tokens;
            }
            if (prompt.is_fanout()) {
                const double agg = makespan > 0 ? static_cast<double>(batch_tokens) * 1000.0 / makespan : 0.0;
                batches.emplace_back(json::Object{{"prompt_id", prompt.id},
                                                  {"iteration", i},
                                                  {"warmup", warmup},
                                                  {"children", batch_size},
                                                  {"failures", batch_failures},
                                                  {"makespan_ms", makespan},
                                                  {"completion_tokens", batch_tokens},
                                                  {"aggregate_tokens_per_sec", agg}});
                if (!warmup) {
                    pm.makespan.push_back(makespan);
                    if (agg > 0) pm.aggregate_tps.push_back(agg);
                }
                if (progress) {
                    *progress << "[bench] " << prompt.id << " #" << i << " fan-out x" << batch_size
                              << " makespan_ms=" << makespan << " aggregate_tps=" << agg << "\n";
                }
            } else if (concurrent_non_fanout && progress) {
                const double agg = makespan > 0 ? static_cast<double>(batch_completed_tokens) * 1000.0 / makespan : 0.0;
                *progress << "[bench] " << prompt.id << " #" << i << " concurrency x" << batch_size
                          << " makespan_ms=" << makespan << " aggregate_tps=" << agg << "\n";
            }
        }
        const double pm_wall_seconds = pm.measured_wall_ms / 1000.0;
        const double pm_aggregate_tps =
            pm_wall_seconds > 0.0 ? static_cast<double>(pm.total_completion_tokens) / pm_wall_seconds : 0.0;
        json::Object ps{{"prompt_id", prompt.id},
                        {"workload", prompt.workload},
                        {"requests_per_iteration",
                         prompt.is_fanout() ? prompt.children.size() : static_cast<std::size_t>(concurrency)},
                        {"prompt_bytes", prompt.is_fanout() ? prompt.children.front().size() : prompt.prompt.size()},
                        {"max_tokens", prompt.max_tokens},
                        {"measured_requests", pm.requests},
                        {"failures", pm.failures},
                        {"cancelled", pm.cancelled},
                        {"wall_seconds", pm_wall_seconds},
                        {"aggregate_decode_tokens_per_sec", pm_aggregate_tps},
                        {"ttft_ms", distribution(pm.ttft)},
                        {"total_ms", distribution(pm.total)},
                        {"decode_tokens_per_sec", distribution(pm.decode_tps)},
                        {"client_decode_tokens_per_sec", distribution(pm.client_decode_tps)},
                        {"prompt_tokens_per_sec", distribution(pm.prompt_tps)},
                        {"completion_tokens", distribution(pm.completion_tokens)}};
        if (prompt.is_fanout()) {
            ps.set("shared_prefix_bytes", prompt.shared_prefix_bytes);
            ps.set("fanout_makespan_ms", distribution(pm.makespan));
            ps.set("fanout_aggregate_tokens_per_sec", distribution(pm.aggregate_tps));
        }
        per_prompt.emplace_back(std::move(ps));
        if (truncated) {
            break;
        }
    }

    std::vector<GpuDeviceInfo> gpus_after;
    if (options.capture_gpu_snapshot) {
        gpus_after = enumerate_gpu_devices();
    }

    const double measured_wall_seconds = global.measured_wall_ms / 1000.0;
    const double aggregate_decode_tps =
        measured_wall_seconds > 0.0 ? static_cast<double>(global.total_completion_tokens) / measured_wall_seconds : 0.0;

    const auto& s = options.sampling;
    json::Object doc;
    doc.set("schema", std::string(kResultSchema));
    doc.set("created_at", utc_timestamp_now());
    doc.set("label", options.label);
    doc.set("engine", json::Object{{"name", "sonder-inference"}, {"version", version_string()}, {"commit", build_commit()}});
    doc.set("host", json::Object{{"platform", host_platform()},
                                 {"node_id", host_name()},
                                 {"hardware", options.hardware},
                                 {"devices", std::move(host_devices)}});
    doc.set("backend", json::Object{{"name", model->backend_name()},
                                    {"version", backend_version},
                                    {"description", backend_description}});
    doc.set("model", json::Object{{"name", d.name},
                                  {"format", d.format},
                                  {"family", d.family},
                                  {"parameter_size", d.parameter_size},
                                  {"quantization", d.quantization},
                                  {"size_bytes", d.size_bytes},
                                  {"context_length", d.context_length}});
    doc.set("sampling", json::Object{{"temperature", s.temperature},
                                     {"top_p", s.top_p},
                                     {"top_k", s.top_k},
                                     {"min_p", s.min_p},
                                     {"repeat_penalty", s.repeat_penalty},
                                     {"typical_p", s.typical_p},
                                     {"repeat_last_n", s.repeat_last_n},
                                     {"presence_penalty", s.presence_penalty},
                                     {"frequency_penalty", s.frequency_penalty},
                                     {"logit_bias_count", static_cast<std::int64_t>(s.logit_bias.size())},
                                     {"num_ctx", s.num_ctx},
                                     {"seed", s.seed},
                                     {"max_tokens", "per-prompt"}});
    doc.set("config", json::Object{{"warmup_runs", options.warmup_runs},
                                   {"measured_runs", options.measured_runs},
                                   {"concurrency", concurrency},
                                   {"concurrency_note", "agent_fanout prompts run one session per child concurrently"},
                                   {"capture_gpu_snapshot", options.capture_gpu_snapshot},
                                   {"cache_state", options.warmup_runs > 0 ? "warm after per-prompt warmup (backend-managed)"
                                                                           : "backend-default (not controlled)"},
                                   {"budget_seconds", options.budget_seconds}});
    doc.set("corpus", json::Object{{"name", corpus.name},
                                   {"version", corpus.version},
                                   {"fnv1a", corpus.fnv1a},
                                   {"prompts", corpus.prompts.size()}});
    if (options.capture_gpu_snapshot) {
        doc.set("gpus_before", gpu_snapshot_to_json(gpus_before, nullptr));
        doc.set("gpus_after", gpu_snapshot_to_json(gpus_after, &gpus_before));
    }
    doc.set("truncated_by_budget", truncated);
    doc.set("cold_first_request_ms", cold_first_ms);
    doc.set("cold_first_request_backend_load_ms", cold_load_ms);
    doc.set("wall_seconds", ms_since(t0) / 1000.0);
    doc.set("summary", json::Object{{"concurrency", concurrency},
                                    {"measured_requests", global.requests},
                                    {"failures", global.failures},
                                    {"cancelled", global.cancelled},
                                    {"wall_seconds", measured_wall_seconds},
                                    {"total_completion_tokens", global.total_completion_tokens},
                                    {"aggregate_decode_tokens_per_sec", aggregate_decode_tps},
                                    {"ttft_ms", distribution(global.ttft)},
                                    {"total_ms", distribution(global.total)},
                                    {"decode_tokens_per_sec", distribution(global.decode_tps)},
                                    {"client_decode_tokens_per_sec", distribution(global.client_decode_tps)},
                                    {"prompt_tokens_per_sec", distribution(global.prompt_tps)}});
    doc.set("per_prompt", std::move(per_prompt));
    doc.set("fanout_batches", std::move(batches));
    doc.set("runs", std::move(runs));
    return doc;
}

}  // namespace sonder::inference::bench
