#include "sonder/inference/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <thread>

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
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_corpus(ss.str());
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
};

struct Outcome {
    bool completed = false;
    bool cancelled = false;
    bool failed = false;
    double ttft = -1, total = 0, dtps = 0, client_dtps = 0, ptps = 0;
    std::uint64_t completion_tokens = 0;
    double load_ms = 0;
};

json::Object execute(Engine& engine, const std::shared_ptr<Model>& model, const Prompt& prompt,
                     const std::string& text, int iteration, int child, bool warmup, const Options& options,
                     Outcome& o) {
    SamplingConfig sampling = options.sampling;
    sampling.max_tokens = prompt.max_tokens;
    SessionOptions so;
    so.sampling = sampling;
    so.run_id = options.label.empty() ? std::optional<std::string>() : options.label;
    json::Object row{{"prompt_id", prompt.id}, {"workload", prompt.workload}, {"iteration", iteration},
                     {"warmup", warmup}};
    if (child >= 0) {
        row.set("child", child);
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

}  // namespace

json::Object run(Engine& engine, const std::shared_ptr<Model>& model, const Corpus& corpus, const Options& options,
                 std::ostream* progress) {
    const auto t0 = Clock::now();
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
        const int warmups = std::max(0, options.warmup_runs);
        const int total_runs = warmups + std::max(1, options.measured_runs);
        for (int i = 0; i < total_runs; ++i) {
            if (over_budget()) {
                truncated = true;
                break;
            }
            const bool warmup = i < warmups;
            std::vector<std::string> texts = prompt.is_fanout() ? prompt.children : std::vector<std::string>{prompt.prompt};
            std::vector<json::Object> rows(texts.size());
            std::vector<Outcome> outcomes(texts.size());
            const auto b0 = Clock::now();
            if (texts.size() == 1) {
                rows[0] = execute(engine, model, prompt, texts[0], i, -1, warmup, options, outcomes[0]);
            } else {
                std::vector<std::thread> threads;
                threads.reserve(texts.size());
                for (std::size_t c = 0; c < texts.size(); ++c) {
                    threads.emplace_back([&, c] {
                        rows[c] = execute(engine, model, prompt, texts[c], i, static_cast<int>(c), warmup, options,
                                          outcomes[c]);
                    });
                }
                for (auto& t : threads) t.join();
            }
            const double makespan = ms_since(b0);
            if (cold_first_ms < 0) {
                cold_first_ms = outcomes[0].total;
                cold_load_ms = outcomes[0].load_ms;
            }
            std::uint64_t batch_tokens = 0;
            std::uint64_t batch_failures = 0;
            for (std::size_t c = 0; c < texts.size(); ++c) {
                if (!warmup) {
                    record(pm, outcomes[c]);
                    record(global, outcomes[c]);
                }
                batch_tokens += outcomes[c].completion_tokens;
                batch_failures += outcomes[c].failed ? 1u : 0u;
                if (progress) {
                    *progress << "[bench] " << prompt.id << " #" << i;
                    if (prompt.is_fanout()) *progress << "." << c;
                    *progress << (warmup ? " (warmup)" : "") << ": " << rows[c].find("outcome")->as_string();
                    if (const json::Value* t = rows[c].find("ttft_ms")) *progress << " ttft_ms=" << t->as_double();
                    if (const json::Value* t = rows[c].find("total_ms")) *progress << " total_ms=" << t->as_double();
                    if (const json::Value* t = rows[c].find("decode_tokens_per_sec")) {
                        *progress << " decode_tps=" << t->as_double();
                    }
                    if (const json::Value* e = rows[c].find("error")) *progress << " error=" << e->as_string();
                    *progress << "\n";
                }
                runs.emplace_back(std::move(rows[c]));
            }
            if (prompt.is_fanout()) {
                const double agg = makespan > 0 ? static_cast<double>(batch_tokens) * 1000.0 / makespan : 0.0;
                batches.emplace_back(json::Object{{"prompt_id", prompt.id},
                                                  {"iteration", i},
                                                  {"warmup", warmup},
                                                  {"children", texts.size()},
                                                  {"failures", batch_failures},
                                                  {"makespan_ms", makespan},
                                                  {"completion_tokens", batch_tokens},
                                                  {"aggregate_tokens_per_sec", agg}});
                if (!warmup) {
                    pm.makespan.push_back(makespan);
                    if (agg > 0) pm.aggregate_tps.push_back(agg);
                }
                if (progress) {
                    *progress << "[bench] " << prompt.id << " #" << i << " fan-out x" << texts.size()
                              << " makespan_ms=" << makespan << " aggregate_tps=" << agg << "\n";
                }
            }
        }
        json::Object ps{{"prompt_id", prompt.id},
                        {"workload", prompt.workload},
                        {"requests_per_iteration", prompt.is_fanout() ? prompt.children.size() : std::size_t{1}},
                        {"prompt_bytes", prompt.is_fanout() ? prompt.children.front().size() : prompt.prompt.size()},
                        {"max_tokens", prompt.max_tokens},
                        {"measured_requests", pm.requests},
                        {"failures", pm.failures},
                        {"cancelled", pm.cancelled},
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
                                   {"concurrency", "1 (agent_fanout prompts: one session per child, concurrent)"},
                                   {"cache_state", options.warmup_runs > 0 ? "warm after per-prompt warmup (backend-managed)"
                                                                           : "backend-default (not controlled)"},
                                   {"budget_seconds", options.budget_seconds}});
    doc.set("corpus", json::Object{{"name", corpus.name},
                                   {"version", corpus.version},
                                   {"fnv1a", corpus.fnv1a},
                                   {"prompts", corpus.prompts.size()}});
    doc.set("truncated_by_budget", truncated);
    doc.set("cold_first_request_ms", cold_first_ms);
    doc.set("cold_first_request_backend_load_ms", cold_load_ms);
    doc.set("wall_seconds", ms_since(t0) / 1000.0);
    doc.set("summary", json::Object{{"measured_requests", global.requests},
                                    {"failures", global.failures},
                                    {"cancelled", global.cancelled},
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
