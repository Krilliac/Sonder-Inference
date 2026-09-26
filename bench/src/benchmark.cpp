#include "sonder/inference/benchmark.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace sonder::inference::bench {

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
    corpus.name = root.find("name") ? root.find("name")->as_string() : "unnamed";
    corpus.description = root.find("description") ? root.find("description")->as_string() : "";
    const json::Value* prompts = root.find("prompts");
    if (!prompts || !prompts->is_array() || prompts->as_array().empty()) {
        return Status(ErrorCode::invalid_argument, "corpus needs a non-empty prompts array");
    }
    for (const auto& p : prompts->as_array()) {
        Prompt pr;
        pr.id = p.find("id") ? p.find("id")->as_string() : "";
        pr.workload = p.find("workload") ? p.find("workload")->as_string() : "interactive_chat";
        pr.prompt = p.find("prompt") ? p.find("prompt")->as_string() : "";
        pr.max_tokens = static_cast<std::int32_t>(p.find("max_tokens") ? p.find("max_tokens")->as_int(64) : 64);
        if (pr.id.empty() || pr.prompt.empty()) {
            return Status(ErrorCode::invalid_argument, "every prompt needs non-empty id and prompt");
        }
        if (pr.max_tokens < 1 || pr.max_tokens > SamplingConfig::kMaxTokensLimit) {
            return Status(ErrorCode::invalid_argument, "prompt " + pr.id + " has invalid max_tokens");
        }
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

namespace {

json::Object distribution(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    return json::Object{{"n", v.size()},
                        {"mean", v.empty() ? 0.0 : sum / static_cast<double>(v.size())},
                        {"p50", percentile(v, 50)},
                        {"p95", percentile(v, 95)},
                        {"min", v.empty() ? 0.0 : *std::min_element(v.begin(), v.end())},
                        {"max", v.empty() ? 0.0 : *std::max_element(v.begin(), v.end())}};
}

}  // namespace

json::Object run(Engine& engine, const std::shared_ptr<Model>& model, const Corpus& corpus, const Options& options,
                 std::ostream* progress) {
    const auto& d = model->descriptor();
    auto backend = engine.find_backend(model->backend_name());
    std::string backend_version = "unknown";
    if (backend) {
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
    std::vector<double> ttft, total, decode_tps, prompt_tps;
    std::uint64_t failures = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t measured = 0;

    for (const auto& prompt : corpus.prompts) {
        const int total_runs = std::max(0, options.warmup_runs) + std::max(1, options.measured_runs);
        for (int i = 0; i < total_runs; ++i) {
            const bool warmup = i < options.warmup_runs;
            SamplingConfig sampling = options.sampling;
            sampling.max_tokens = prompt.max_tokens;
            SessionOptions so;
            so.sampling = sampling;
            so.run_id = options.label.empty() ? std::optional<std::string>() : options.label;
            json::Object row{{"prompt_id", prompt.id}, {"workload", prompt.workload}, {"iteration", i},
                             {"warmup", warmup}};
            auto session = engine.create_session(model, so);
            if (!session.ok()) {
                row.set("outcome", "failed");
                row.set("error", session.status().to_string());
                ++failures;
                runs.emplace_back(std::move(row));
                continue;
            }
            auto res = session.value()->generate(prompt.prompt);
            if (!res.ok()) {
                row.set("outcome", "failed");
                row.set("error", res.status().to_string());
                if (!warmup) ++failures;
            } else {
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
                double dtps = 0;
                if (s.eval_ns > 0 && s.completion_tokens > 0) {
                    dtps = static_cast<double>(s.completion_tokens) * 1e9 / static_cast<double>(s.eval_ns);
                    row.set("decode_tokens_per_sec_source", "backend_eval_duration");
                } else if (r.ttft_ms >= 0 && r.total_ms > r.ttft_ms && s.completion_tokens > 1) {
                    dtps = static_cast<double>(s.completion_tokens - 1) * 1000.0 / (r.total_ms - r.ttft_ms);
                    row.set("decode_tokens_per_sec_source", "wall_clock_after_first_chunk");
                }
                row.set("decode_tokens_per_sec", dtps);
                double ptps = 0;
                if (s.prompt_eval_ns > 0 && s.prompt_tokens > 0) {
                    ptps = static_cast<double>(s.prompt_tokens) * 1e9 / static_cast<double>(s.prompt_eval_ns);
                }
                row.set("prompt_tokens_per_sec", ptps);
                row.set("backend_load_ms", static_cast<double>(s.load_ns) / 1e6);
                if (!warmup) {
                    ++measured;
                    if (r.outcome == RequestOutcome::completed) {
                        if (r.ttft_ms >= 0) ttft.push_back(r.ttft_ms);
                        total.push_back(r.total_ms);
                        if (dtps > 0) decode_tps.push_back(dtps);
                        if (ptps > 0) prompt_tps.push_back(ptps);
                    } else if (r.outcome == RequestOutcome::cancelled) {
                        ++cancelled;
                    }
                }
            }
            if (progress) {
                *progress << "[bench] " << prompt.id << " #" << i << (warmup ? " (warmup)" : "") << ": "
                          << row.find("outcome")->as_string();
                if (const json::Value* t = row.find("total_ms")) {
                    *progress << " total_ms=" << t->as_double();
                }
                if (const json::Value* t = row.find("decode_tokens_per_sec")) {
                    *progress << " decode_tps=" << t->as_double();
                }
                *progress << "\n";
            }
            runs.emplace_back(std::move(row));
        }
    }

    const auto& s = options.sampling;
    json::Object doc;
    doc.set("schema", std::string(kResultSchema));
    doc.set("created_at", utc_timestamp_now());
    doc.set("label", options.label);
    doc.set("engine", json::Object{{"name", "sonder-inference"}, {"version", version_string()}, {"commit", build_commit()}});
    doc.set("host", json::Object{{"platform", host_platform()}, {"node_id", host_name()}, {"devices", std::move(host_devices)}});
    doc.set("backend", json::Object{{"name", model->backend_name()}, {"version", backend_version}});
    doc.set("model", json::Object{{"name", d.name},
                                  {"format", d.format},
                                  {"family", d.family},
                                  {"parameter_size", d.parameter_size},
                                  {"quantization", d.quantization},
                                  {"size_bytes", d.size_bytes}});
    doc.set("sampling", json::Object{{"temperature", s.temperature},
                                     {"top_p", s.top_p},
                                     {"top_k", s.top_k},
                                     {"min_p", s.min_p},
                                     {"repeat_penalty", s.repeat_penalty},
                                     {"seed", s.seed},
                                     {"max_tokens", "per-prompt"}});
    doc.set("config", json::Object{{"warmup_runs", options.warmup_runs},
                                   {"measured_runs", options.measured_runs},
                                   {"concurrency", 1},
                                   {"cache_state", "backend-default (not controlled)"}});
    doc.set("corpus", json::Object{{"name", corpus.name}, {"prompts", corpus.prompts.size()}});
    doc.set("summary", json::Object{{"measured_requests", measured},
                                    {"failures", failures},
                                    {"cancelled", cancelled},
                                    {"ttft_ms", distribution(ttft)},
                                    {"total_ms", distribution(total)},
                                    {"decode_tokens_per_sec", distribution(decode_tps)},
                                    {"prompt_tokens_per_sec", distribution(prompt_tps)}});
    doc.set("runs", std::move(runs));
    return doc;
}

}  // namespace sonder::inference::bench
