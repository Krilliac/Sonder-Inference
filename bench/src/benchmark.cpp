#include "sonder/inference/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#include "cli/cli_spec.hpp"
#include "cli/cli_values.hpp"
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "backends/llamaserver/llamaserver_options.hpp"
#include "net/http_client.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif
#endif

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

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    return b > kMax - a ? kMax : a + b;
}

struct Metrics {
    std::vector<double> ttft, total, decode_tps, client_decode_tps, prompt_tps, completion_tokens;
    std::vector<double> makespan, aggregate_tps;  // fan-out batches
    std::uint64_t requests = 0;
    std::uint64_t failures = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t total_completion_tokens = 0;
    double measured_wall_ms = 0.0;
    std::uint64_t draft_tokens_total = 0;
    std::uint64_t draft_tokens_with_accepted = 0;
    std::uint64_t draft_accepted_total = 0;
    long double draft_tokens_with_accepted_ld = 0.0L;
    long double draft_accepted_ld = 0.0L;
    bool has_draft_accepted = false;
    bool has_cached_prompt_tokens = false;
    std::uint64_t cached_prompt_tokens_total = 0;
};

struct Outcome {
    bool completed = false;
    bool cancelled = false;
    bool failed = false;
    double ttft = -1, total = 0, dtps = 0, client_dtps = 0, ptps = 0;
    std::uint64_t completion_tokens = 0;
    double load_ms = 0;
    bool has_draft_tokens = false;
    std::uint64_t draft_tokens = 0;
    bool has_draft_accepted_tokens = false;
    std::uint64_t draft_accepted_tokens = 0;
    bool has_cached_prompt_tokens = false;
    std::uint64_t cached_prompt_tokens = 0;
};

std::optional<std::uint64_t> extract_draft_tokens(const GenerateStats& s) noexcept {
    if (s.draft_tokens.has_value()) {
        return *s.draft_tokens;
    }
    if (s.backend_timings.has_value() && s.backend_timings->draft_n.has_value()) {
        return *s.backend_timings->draft_n;
    }
    return std::nullopt;
}

std::optional<std::uint64_t> extract_draft_accepted_tokens(const GenerateStats& s) noexcept {
    if (s.draft_accepted_tokens.has_value()) {
        return *s.draft_accepted_tokens;
    }
    if (s.backend_timings.has_value() && s.backend_timings->draft_n_accepted.has_value()) {
        return *s.backend_timings->draft_n_accepted;
    }
    return std::nullopt;
}

std::optional<std::uint64_t> extract_cached_prompt_tokens(const GenerateStats& s) noexcept {
    if (s.cached_tokens.has_value()) {
        return *s.cached_tokens;
    }
    if (s.backend_timings.has_value() && s.backend_timings->cache_n.has_value()) {
        return *s.backend_timings->cache_n;
    }
    return std::nullopt;
}

void append_speculation_and_cache_metrics(json::Object& target, const Metrics& m) {
    if (m.draft_tokens_total > 0) {
        target.set("draft_tokens_total", m.draft_tokens_total);
        if (m.has_draft_accepted && m.draft_tokens_with_accepted > 0 && m.draft_tokens_with_accepted_ld > 0.0L) {
            if (m.draft_tokens_with_accepted != m.draft_tokens_total) {
                target.set("draft_tokens_evaluated_total", m.draft_tokens_with_accepted);
            }
            target.set("draft_accepted_total", m.draft_accepted_total);
            const double rate =
                std::clamp(static_cast<double>(m.draft_accepted_ld / m.draft_tokens_with_accepted_ld), 0.0, 1.0);
            if (std::isfinite(rate)) {
                target.set("draft_acceptance_rate", rate);
            }
        }
    }
    if (m.has_cached_prompt_tokens) {
        target.set("cached_prompt_tokens_total", m.cached_prompt_tokens_total);
    }
}

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
    sampling.explicit_fields |= SamplingConfig::kMaxTokens;
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

    const std::optional<std::uint64_t> draft_tokens = extract_draft_tokens(s);
    const std::optional<std::uint64_t> draft_accepted_raw = extract_draft_accepted_tokens(s);
    const std::optional<std::uint64_t> cached_prompt_tokens = extract_cached_prompt_tokens(s);
    if (draft_tokens.has_value()) {
        row.set("draft_tokens", *draft_tokens);
        o.has_draft_tokens = true;
        o.draft_tokens = *draft_tokens;
    }
    if (draft_accepted_raw.has_value()) {
        std::uint64_t accepted = *draft_accepted_raw;
        if (draft_tokens.has_value()) {
            accepted = std::min(accepted, *draft_tokens);
        }
        row.set("draft_accepted_tokens", accepted);
        o.has_draft_accepted_tokens = true;
        o.draft_accepted_tokens = accepted;
        if (draft_tokens.has_value() && *draft_tokens > 0) {
            const double rate =
                std::clamp(static_cast<double>(accepted) / static_cast<double>(*draft_tokens), 0.0, 1.0);
            if (std::isfinite(rate)) {
                row.set("draft_acceptance_rate", rate);
            }
        }
    }
    if (cached_prompt_tokens.has_value()) {
        row.set("cached_prompt_tokens", *cached_prompt_tokens);
        o.has_cached_prompt_tokens = true;
        o.cached_prompt_tokens = *cached_prompt_tokens;
    }

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
    if (o.has_draft_tokens && o.draft_tokens > 0) {
        m.draft_tokens_total = saturating_add(m.draft_tokens_total, o.draft_tokens);
        if (o.has_draft_accepted_tokens) {
            const std::uint64_t accepted = std::min(o.draft_accepted_tokens, o.draft_tokens);
            m.draft_tokens_with_accepted = saturating_add(m.draft_tokens_with_accepted, o.draft_tokens);
            m.draft_accepted_total = saturating_add(m.draft_accepted_total, accepted);
            m.draft_tokens_with_accepted_ld += static_cast<long double>(o.draft_tokens);
            m.draft_accepted_ld += static_cast<long double>(accepted);
            m.has_draft_accepted = true;
        }
    }
    if (o.has_cached_prompt_tokens) {
        m.has_cached_prompt_tokens = true;
        m.cached_prompt_tokens_total = saturating_add(m.cached_prompt_tokens_total, o.cached_prompt_tokens);
    }
}

const sonder::cli::CommandSpec& bench_cli_spec() {
    namespace cli = sonder::cli;
    static const cli::CommandSpec spec{
        "",
        "standalone benchmark runner (docs/BENCHMARK_PLAN.md)",
        {"--model NAME [options]", "--backend ollama --list-models",
         "--backend llamaserver --llamaserver-url URL --list-models"},
        {{"Backend",
          {{"backend", cli::OptionKind::value, "NAME", "ollama | llamaserver | mock (default: ollama)"},
           {"ollama-url", cli::OptionKind::value, "URL",
            "Ollama base URL (default: http://127.0.0.1:11434, loopback only)"},
           {"llamaserver-url", cli::OptionKind::value, "URL",
            "llama-server base URL (default: http://127.0.0.1:8080, loopback only)"},
           {"llamaserver-config", cli::OptionKind::value, "PATH",
            "llama-server JSON config (attach or spawn, loopback only)"},
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
         "--backend llamaserver --llamaserver-url http://127.0.0.1:8080 --list-models",
         "--backend llamaserver --llamaserver-url http://127.0.0.1:8080 --model bonsai-2 --warmup 1 --runs 3",
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

Result<std::shared_ptr<Backend>> make_llamaserver_bench_backend(const LlamaServerBenchOptions& opts) {
#if !defined(SONDER_HAS_LLAMASERVER_BACKEND)
    (void)opts;
    return Status(ErrorCode::unsupported,
                  "this build does not include the llamaserver backend (src/backends/llamaserver)");
#else
    if (!opts.url.empty() && !opts.config_path.empty()) {
        return Status(ErrorCode::invalid_argument,
                      "llamaserver: --llamaserver-url and --llamaserver-config cannot be used together");
    }

    const auto validate_loopback_url = [](const std::string& url) -> Status {
        auto parsed = net::parse_url(url);
        if (!parsed.ok()) {
            return parsed.status();
        }
        if (!net::is_loopback_host(parsed->host)) {
            return Status(ErrorCode::invalid_argument,
                          "llamaserver: non-loopback upstream is not allowed in benchmark mode (loopback only)");
        }
        return Status::success();
    };

    if (!opts.url.empty()) {
        if (auto st = validate_loopback_url(opts.url); !st.ok()) {
            return st;
        }
    }

#if defined(SONDER_HAS_SERVER)
    BackendSetup setup;
    setup.backend = kLlamaServerBackendName;
    if (!opts.config_path.empty()) {
        if (auto st = load_llamaserver_config(opts.config_path, setup); !st.ok()) {
            return st;
        }
    }
    if (!opts.url.empty()) {
        setup.llamaserver_url = opts.url;
    }
    if (!opts.mode.empty()) {
        setup.llamaserver_mode = opts.mode;
    }
    if (!opts.executable.empty()) {
        setup.llamaserver_executable = opts.executable;
    }
    if (!opts.args.empty()) {
        setup.llamaserver_args = opts.args;
    }
    if (setup.llamaserver_allow_remote) {
        return Status(ErrorCode::invalid_argument,
                      "llamaserver: allow_remote=true is not allowed in benchmark mode (loopback only)");
    }
    setup.llamaserver_allow_remote = false;
    if (!setup.llamaserver_url.empty()) {
        if (auto st = validate_loopback_url(setup.llamaserver_url); !st.ok()) {
            return st;
        }
    }

    // Eagerly mirror BackendSetup into LlamaServerBackendOptions and run
    // validate_backend_options() so invalid attach/spawn configs fail fast
    // with invalid_argument (exit 2) instead of deferring to probe() (exit 3).
    LlamaServerBackendOptions check_lo;
    check_lo.allow_remote = false;
    if (!setup.llamaserver_mode.empty()) {
        if (setup.llamaserver_mode == "attach") {
            check_lo.mode = LlamaServerMode::attach;
        } else if (setup.llamaserver_mode == "spawn") {
            check_lo.mode = LlamaServerMode::spawn;
        } else {
            return Status(ErrorCode::invalid_argument, "llamaserver: mode must be attach or spawn");
        }
    }
    if (!setup.llamaserver_url.empty()) {
        check_lo.base_url = setup.llamaserver_url;
    }
    check_lo.executable = setup.llamaserver_executable;
    check_lo.args = setup.llamaserver_args;
    check_lo.environment = setup.llamaserver_environment;
    check_lo.native_completion = setup.llamaserver_native_completion;
    check_lo.grammar = setup.llamaserver_grammar;
    check_lo.upstream_grammar = setup.llamaserver_upstream_grammar;
    check_lo.context_length = setup.llamaserver_context_length;
    check_lo.connect_timeout = std::chrono::milliseconds(setup.llamaserver_connect_timeout_ms);
    check_lo.request_timeout = std::chrono::milliseconds(setup.llamaserver_request_timeout_ms);
    check_lo.startup_timeout = std::chrono::milliseconds(setup.llamaserver_startup_timeout_ms);
    check_lo.poll_interval = std::chrono::milliseconds(setup.llamaserver_poll_interval_ms);
    check_lo.shutdown_timeout = std::chrono::milliseconds(setup.llamaserver_shutdown_timeout_ms);
    check_lo.restart_backoff = std::chrono::milliseconds(setup.llamaserver_restart_backoff_ms);
    check_lo.max_restart_backoff = std::chrono::milliseconds(setup.llamaserver_max_restart_backoff_ms);
    check_lo.max_restarts = static_cast<std::size_t>(setup.llamaserver_max_restarts);
    check_lo.tls.handshake_timeout = std::chrono::milliseconds(setup.llamaserver_handshake_timeout_ms);
    check_lo.diagnostics.log_file = setup.llamaserver_log_file;
    check_lo.stall_guard.enabled = setup.llamaserver_stall_guard;
    check_lo.stall_guard.stall_seconds = std::chrono::seconds(setup.llamaserver_stall_seconds);
    check_lo.stall_guard.policy = setup.llamaserver_stall_policy;
    check_lo.stall_guard.generic_progress = setup.llamaserver_stall_generic_progress;
    check_lo.api_key_env = setup.llamaserver_api_key_env;
    check_lo.idle_timeout = std::chrono::milliseconds(setup.llamaserver_idle_timeout_ms);
    check_lo.first_content_timeout = std::chrono::milliseconds(setup.llamaserver_first_content_timeout_ms);
    check_lo.keepalive_gap = std::chrono::milliseconds(setup.llamaserver_keepalive_gap_ms);
    check_lo.unlimited_request_timeout = setup.llamaserver_unlimited_request_timeout;
    check_lo.warmup.messages_file = setup.llamaserver_warmup_messages_file;
    check_lo.spill_guard.enabled = setup.llamaserver_spill_guard;
    if (setup.llamaserver_spill_policy == "warn") {
        check_lo.spill_guard.policy = LlamaServerSpillPolicy::warn;
    } else if (setup.llamaserver_spill_policy == "refuse") {
        check_lo.spill_guard.policy = LlamaServerSpillPolicy::refuse;
    } else if (setup.llamaserver_spill_policy == "auto_fit") {
        check_lo.spill_guard.policy = LlamaServerSpillPolicy::auto_fit;
    } else {
        return Status(ErrorCode::invalid_argument, "llamaserver: spill policy must be warn, refuse or auto_fit");
    }
    if (setup.llamaserver_spill_threshold_mib > 1048576 || setup.llamaserver_spill_baseline_mib > 1048576) {
        return Status(ErrorCode::invalid_argument, "llamaserver: spill threshold/baseline must be at most 1048576 MiB");
    }
    check_lo.spill_guard.threshold_bytes = setup.llamaserver_spill_threshold_mib * kBytesPerMiB;
    check_lo.spill_guard.baseline_bytes = setup.llamaserver_spill_baseline_mib * kBytesPerMiB;
    if (setup.llamaserver_spill_baseline_per_1k_ctx_mib > 1024) {
        return Status(ErrorCode::invalid_argument,
                      "llamaserver: spill baseline growth must be at most 1024 MiB per 1k ctx");
    }
    check_lo.spill_guard.baseline_bytes_per_1k_ctx = setup.llamaserver_spill_baseline_per_1k_ctx_mib * kBytesPerMiB;
    check_lo.spill_guard.sample_interval = std::chrono::milliseconds(setup.llamaserver_spill_sample_interval_ms);
    check_lo.spill_guard.fit_step_factor = setup.llamaserver_fit_step_factor;
    check_lo.spill_guard.fit_step_align = setup.llamaserver_fit_step_align;
    check_lo.spill_guard.fit_min_ctx = setup.llamaserver_fit_min_ctx;
    check_lo.spill_guard.fit_max_attempts = static_cast<std::size_t>(setup.llamaserver_fit_max_attempts);
    check_lo.spill_guard.uma_mode_policy = setup.llamaserver_spill_uma_mode;
    {
        const auto& rc_cfg = setup.llamaserver_residency;
        check_lo.spill_guard.residency.enabled = rc_cfg.enabled;
        check_lo.spill_guard.residency.expect_gpu = rc_cfg.expect_gpu;
        check_lo.spill_guard.residency.min_dedicated_bytes = rc_cfg.min_dedicated_mib * kBytesPerMiB;
        check_lo.spill_guard.residency.consecutive_samples = static_cast<std::size_t>(rc_cfg.consecutive_samples);
        check_lo.spill_guard.residency.eviction_fraction = rc_cfg.eviction_fraction;
        check_lo.spill_guard.residency.eviction_bytes = rc_cfg.eviction_mib * kBytesPerMiB;
        if (rc_cfg.on_eviction == "warn") {
            check_lo.spill_guard.residency.on_eviction = LlamaServerEvictionPolicy::warn;
        } else if (rc_cfg.on_eviction == "restart") {
            check_lo.spill_guard.residency.on_eviction = LlamaServerEvictionPolicy::restart;
        } else {
            return Status(ErrorCode::invalid_argument, "llamaserver: residency.on_eviction must be warn or restart");
        }
        check_lo.spill_guard.residency.max_eviction_restarts = static_cast<std::size_t>(rc_cfg.max_eviction_restarts);
    }
    {
        const auto& hm = setup.llamaserver_host_memory;
        check_lo.host_memory.estimated_resident_bytes = hm.estimated_resident_bytes;
        check_lo.host_memory.estimated_private_bytes = hm.estimated_private_bytes;
        check_lo.host_memory.estimate_lower_bound = hm.estimate_lower_bound;
        check_lo.host_memory.reserve_bytes = hm.reserve_mib * kBytesPerMiB;
        check_lo.host_memory.override_refusal = hm.override_refusal;
        check_lo.host_memory.floor.enabled = hm.floor_enabled;
        check_lo.host_memory.floor.policy = hm.floor_policy;
        check_lo.host_memory.floor.floor_bytes = hm.floor_mib * kBytesPerMiB;
        check_lo.host_memory.floor.resume_bytes = hm.resume_mib * kBytesPerMiB;
        check_lo.host_memory.floor.consecutive_samples = static_cast<std::size_t>(hm.consecutive_samples);
        check_lo.host_memory.floor.sample_interval = std::chrono::milliseconds(hm.sample_interval_ms);
        check_lo.host_memory.floor.restart_cooldown = std::chrono::milliseconds(hm.restart_cooldown_ms);
        check_lo.host_memory.floor.max_restarts = static_cast<std::size_t>(hm.max_restarts);
    }
    {
        const auto& sp = setup.llamaserver_adaptive_speculation;
        check_lo.adaptive_speculation.enabled = sp.enabled;
        check_lo.adaptive_speculation.min_draft_n = sp.min_draft_n;
        check_lo.adaptive_speculation.initial_draft_n = sp.initial_draft_n;
        check_lo.adaptive_speculation.max_draft_n = sp.max_draft_n;
        check_lo.adaptive_speculation.window_size = sp.window_size;
        check_lo.adaptive_speculation.min_samples = sp.min_samples;
        check_lo.adaptive_speculation.pause_acceptance_threshold = sp.pause_acceptance_threshold;
        check_lo.adaptive_speculation.step_down_acceptance_threshold = sp.step_down_acceptance_threshold;
        check_lo.adaptive_speculation.step_up_acceptance_threshold = sp.step_up_acceptance_threshold;
        check_lo.adaptive_speculation.cooldown_requests = sp.cooldown_requests;
        check_lo.adaptive_speculation.probe_draft_n = sp.probe_draft_n;
        check_lo.adaptive_speculation.min_draft_tokens_per_sample = sp.min_draft_tokens_per_sample;
    }
    if (auto st = llamaserver::validate_backend_options(check_lo); !st.ok()) {
        return st;
    }
    return make_backend(setup);
#else
    if (!opts.config_path.empty()) {
        return Status(ErrorCode::unsupported,
                      "llamaserver: --llamaserver-config requires a build with the server module (SONDER_HAS_SERVER)");
    }
    LlamaServerBackendOptions lo;
    lo.allow_remote = false;
    if (!opts.mode.empty()) {
        if (opts.mode == "attach") {
            lo.mode = LlamaServerMode::attach;
        } else if (opts.mode == "spawn") {
            lo.mode = LlamaServerMode::spawn;
        } else {
            return Status(ErrorCode::invalid_argument, "llamaserver: mode must be attach or spawn");
        }
    }
    if (!opts.url.empty()) {
        lo.base_url = opts.url;
    }
    lo.executable = opts.executable;
    lo.args = opts.args;
    if (auto st = llamaserver::validate_backend_options(lo); !st.ok()) {
        return st;
    }
    return std::shared_ptr<Backend>(make_llamaserver_backend(std::move(lo)));
#endif
#endif
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
    if (out.backend != "ollama" && out.backend != "llamaserver" && out.backend != "mock") {
        error = "unsupported backend '" + out.backend + "' (expected ollama, llamaserver or mock)";
        return false;
    }

    const auto ollama_url = a.get("ollama-url");
    const auto llamaserver_url = a.get("llamaserver-url");
    const auto llamaserver_config = a.get("llamaserver-config");
    if (llamaserver_url.has_value() && llamaserver_config.has_value()) {
        error = "--llamaserver-url and --llamaserver-config cannot be used together";
        return false;
    }
    if (llamaserver_url.has_value()) {
        if (out.backend != "llamaserver") {
            error = "--llamaserver-url requires --backend llamaserver";
            return false;
        }
        if (llamaserver_url->empty()) {
            error = "--llamaserver-url must not be empty";
            return false;
        }
        out.llamaserver_url = *llamaserver_url;
    }
    if (llamaserver_config.has_value()) {
        if (out.backend != "llamaserver") {
            error = "--llamaserver-config requires --backend llamaserver";
            return false;
        }
        if (llamaserver_config->empty()) {
            error = "--llamaserver-config must not be empty";
            return false;
        }
        out.llamaserver_config = *llamaserver_config;
    }
    if (ollama_url.has_value()) {
        if (out.backend != "ollama") {
            error = "--ollama-url requires --backend ollama";
            return false;
        }
        if (ollama_url->empty()) {
            error = "--ollama-url must not be empty";
            return false;
        }
        out.ollama_url = *ollama_url;
    }

    out.model = a.get("model").value_or("");
    out.corpus_path = a.get("corpus").value_or("bench/corpus/baseline.json");
    out.out_dir = a.get("out-dir").value_or("bench/results");
    out.stem = a.get("stem").value_or("");
    out.telemetry_path = a.get("telemetry").value_or("");
    out.list_models = a.has("list-models");
    out.dry_run = a.has("dry-run");
    out.require_idle = a.has("require-idle");
    if (out.require_idle && out.backend != "ollama") {
        error = "--require-idle is only supported with --backend ollama";
        return false;
    }
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
                    if (const json::Value* dv = rows[c].find("draft_acceptance_rate")) {
                        *progress << " draft_acc=" << dv->as_double();
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
        append_speculation_and_cache_metrics(ps, pm);
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
    const auto sampling_field = [&](SamplingConfig::Field f, json::Value v) -> json::Value {
        if (!s.explicit_only || s.is_explicit(f)) {
            return v;
        }
        return json::Value(nullptr);
    };
    doc.set("sampling", json::Object{
                            {"temperature", sampling_field(SamplingConfig::kTemperature, s.temperature)},
                            {"top_p", sampling_field(SamplingConfig::kTopP, s.top_p)},
                            {"top_k", sampling_field(SamplingConfig::kTopK, s.top_k)},
                            {"min_p", sampling_field(SamplingConfig::kMinP, s.min_p)},
                            {"repeat_penalty", sampling_field(SamplingConfig::kRepeatPenalty, s.repeat_penalty)},
                            {"typical_p", sampling_field(SamplingConfig::kTypicalP, s.typical_p)},
                            {"repeat_last_n", sampling_field(SamplingConfig::kRepeatLastN, s.repeat_last_n)},
                            {"presence_penalty", sampling_field(SamplingConfig::kPresencePenalty, s.presence_penalty)},
                            {"frequency_penalty",
                             sampling_field(SamplingConfig::kFrequencyPenalty, s.frequency_penalty)},
                            {"explicit_only", s.explicit_only},
                            {"logit_bias_count", static_cast<std::int64_t>(s.logit_bias.size())},
                            {"num_ctx", sampling_field(SamplingConfig::kNumCtx, s.num_ctx)},
                            {"seed", sampling_field(SamplingConfig::kSeed, json::Value(s.seed))},
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
    json::Object summary{{"concurrency", concurrency},
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
                         {"prompt_tokens_per_sec", distribution(global.prompt_tps)}};
    append_speculation_and_cache_metrics(summary, global);
    doc.set("summary", std::move(summary));
    doc.set("per_prompt", std::move(per_prompt));
    doc.set("fanout_batches", std::move(batches));
    doc.set("runs", std::move(runs));
    return doc;
}

}  // namespace sonder::inference::bench
