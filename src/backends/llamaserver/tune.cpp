#include "tune.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>

namespace sonder::inference::llamaserver::tune {
namespace {
Status invalid(const char *message) { return {ErrorCode::invalid_argument, message}; }
bool eligible(const Measurement &m) {
    const auto weighted = weighted_decode_tps(m.workloads);
    return m.phase == Phase::benchmark && m.verdict == Verdict::clean && m.memory_released &&
           !m.slow_kernel && m.dedicated_bytes && m.shared_bytes && m.short_tps && m.prefill_tps &&
           std::isfinite(*m.short_tps) && *m.short_tps > 0 && std::isfinite(*m.prefill_tps) &&
           *m.prefill_tps > 0 && m.safety_probe_ctx >= m.candidate.ctx + kMargin && weighted.has_value();
}
bool faster(const Measurement &a, const Measurement &b) {
    const auto aw = weighted_decode_tps(a.workloads);
    const auto bw = weighted_decode_tps(b.workloads);
    if (aw && bw && *aw != *bw) return *aw > *bw;
    if (a.prefill_tps != b.prefill_tps) return a.prefill_tps > b.prefill_tps;
    if (a.candidate.ctx != b.candidate.ctx) return a.candidate.ctx > b.candidate.ctx;
    return a.candidate.mtp < b.candidate.mtp;
}
json::Object spawn_config(const Options &options, const Candidate &candidate) {
    json::Array args;
    for (const auto &arg : arguments(options, candidate)) args.emplace_back(arg);
    json::Object env;
    for (const auto &[key, value] : options.environment) env.set(key, value);
    return {{"mode", "spawn"}, {"executable", options.executable}, {"args", std::move(args)},
            // A profile for daily use: restart after a crash, and shrink the
            // context (not refuse) if the desktop takes more VRAM later.
            {"env", std::move(env)}, {"max_restarts", 3}, {"startup_timeout_ms", 120000},
            {"spill_guard", json::Object{{"enabled", true}, {"policy", "auto_fit"},
                {"threshold_mib", spill_guard_for(options.grid, candidate).threshold_bytes / kMiB},
                {"baseline_mib", spill_guard_for(options.grid, candidate).baseline_bytes / kMiB},
                {"baseline_per_1k_ctx_mib", spill_guard_for(options.grid, candidate).baseline_bytes_per_1k_ctx / kMiB},
                {"fit_min_ctx", std::min<std::uint64_t>(candidate.ctx, 32768)}}}};
}
json::Value measurement_json(const Measurement &m) {
    const auto mib = [](const std::optional<std::uint64_t> &n) -> json::Value {
        return n ? json::Value(static_cast<double>(*n) / static_cast<double>(kMiB)) : json::Value{};
    };
    json::Array workloads;
    for (const auto &w : m.workloads) {
        std::optional<double> acceptance;
        if (w.draft_n && *w.draft_n != 0 && w.draft_n_accepted)
            acceptance = static_cast<double>(*w.draft_n_accepted) / static_cast<double>(*w.draft_n);
        workloads.emplace_back(json::Object{{"name", w.name}, {"weight", w.weight}, {"tok_s", w.tok_s},
            {"draft_n", w.draft_n ? json::Value(*w.draft_n) : json::Value{}},
            {"draft_n_accepted", w.draft_n_accepted ? json::Value(*w.draft_n_accepted) : json::Value{}},
            {"acceptance_ratio", acceptance ? json::Value(*acceptance) : json::Value{}}});
    }
    const auto weighted = weighted_decode_tps(m.workloads);
    return json::Object{{"ctx", m.candidate.ctx}, {"k", m.candidate.kv.k}, {"v", m.candidate.kv.v},
        {"ubatch", m.candidate.ubatch}, {"mtp_n", m.candidate.mtp},
        {"spec_type", m.candidate.mtp == 0 ? "none" : m.candidate.spec_type},
        {"p_min", m.candidate.mtp == 0 ? json::Value{} : json::Value(m.candidate.p_min)},
        {"phase", m.phase == Phase::probe ? "probe" : "benchmark"},
        {"shared_mib", mib(m.shared_bytes)}, {"dedicated_mib", mib(m.dedicated_bytes)},
        {"short_tok_s", m.short_tps}, {"prefill_tok_s", m.prefill_tps},
        {"weighted_tok_s", weighted ? json::Value(*weighted) : json::Value{}},
        {"workloads", std::move(workloads)},
        {"long_prefill_tok_s", m.long_prefill_tps}, {"verdict", to_string(m.verdict)},
        {"slow_kernel", m.slow_kernel}, {"served_ctx", m.served_ctx},
        {"safety_probe_ctx", m.safety_probe_ctx}, {"memory_released", m.memory_released}, {"detail", m.detail}};
}
} // namespace

Grid::Grid() {
    for (std::uint64_t ctx_size = 32768; ctx_size <= 139264; ctx_size += kMargin) ctx.push_back(ctx_size);
    spill_mtp.baseline_bytes = 166 * kMiB;
    spill_mtp.baseline_bytes_per_1k_ctx = 2 * kMiB;
    spill_mtp.threshold_bytes = 32 * kMiB;
}
std::optional<double> weighted_decode_tps(const std::vector<WorkloadMeasurement> &workloads) {
    static const std::array<const char *, 4> names{"prose", "code_edit", "tool_json", "reasoning"};
    static const std::array<double, 4> weights{0.4, 0.3, 0.2, 0.1};
    if (workloads.size() != names.size()) return {};
    double total = 0.0;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto it = std::find_if(workloads.begin(), workloads.end(), [&](const auto &w) {
            return w.name == names[i];
        });
        if (it == workloads.end() || it->weight != weights[i] || !std::isfinite(it->tok_s) || it->tok_s <= 0) return {};
        total += weights[i] * it->tok_s;
    }
    return std::isfinite(total) ? std::optional<double>(total) : std::nullopt;
}
const SpillGuardOptions &spill_guard_for(const Grid &grid, const Candidate &candidate) {
    return candidate.mtp > 0 ? grid.spill_mtp : grid.spill;
}
const char *to_string(Verdict verdict) {
    switch (verdict) {
    case Verdict::clean: return "clean";
    case Verdict::spill: return "spill";
    case Verdict::slow_kernel: return "slow kernel";
    case Verdict::error: return "error";
    case Verdict::unsupported: return "unsupported";
    case Verdict::timed_out: return "budget exhausted";
    case Verdict::cancelled: return "cancelled";
    case Verdict::release_failed: return "memory release unconfirmed";
    }
    return "error";
}
Bisection::Bisection(std::vector<std::uint64_t> contexts)
    : contexts_(std::move(contexts)), upper_(contexts_.size()) {}
std::optional<std::uint64_t> Bisection::next() const {
    if (lower_ >= upper_) return {};
    return contexts_[lower_ + (upper_ - lower_) / 2];
}
void Bisection::observe(bool clean) {
    if (!next()) return;
    const auto mid = lower_ + (upper_ - lower_) / 2;
    if (clean) {
        edge_ = contexts_[mid];
        lower_ = mid + 1;
    } else upper_ = mid;
}
std::optional<std::uint64_t> margin_context(std::uint64_t edge) {
    // Always reserve room for the 8192-token workload plus decode and a margin.
    if (edge < 16384) return {};
    return edge - kMargin;
}
Status validate_grid(const Grid &g) {
    if (g.kv.empty() || g.kv.size() > 8 || g.ctx.empty() || g.ctx.size() > 128 ||
        g.ubatch.empty() || g.ubatch.size() > 8 || g.mtp.empty() || g.mtp.size() > 4)
        return invalid("grid arrays are empty or exceed their bounds");
    const auto base_count = g.kv.size() * g.ctx.size() * g.ubatch.size() * g.mtp.size();
    if (base_count > 4096) return invalid("grid exceeds 4096 base candidate combinations");
    if (g.spec_type.empty() || g.spec_type.size() > 4 || g.p_min.empty() || g.p_min.size() > 4)
        return invalid("spec_type and p_min arrays are empty or exceed their bounds");
    const auto spec_variants = g.spec_type.size() * g.p_min.size();
    if (g.kv.size() * g.ctx.size() * g.ubatch.size() * (1 + (g.mtp.size() - 1) * spec_variants) > 16384)
        return invalid("grid exceeds 16384 candidate combinations");
    if (g.batch == 0 || g.batch > 8192) return invalid("batch must be in [1, 8192]");
    const std::set<std::string> types{"q4_0", "q8_0", "q5_1", "f16", "bf16"};
    std::set<std::pair<std::string, std::string>> pairs;
    for (const auto &pair : g.kv) {
        if (!types.contains(pair.k) || !types.contains(pair.v) || !pairs.emplace(pair.k, pair.v).second)
            return invalid("KV types must be supported names and pairs must be unique");
    }
    if (!std::is_sorted(g.ctx.begin(), g.ctx.end()) ||
        std::adjacent_find(g.ctx.begin(), g.ctx.end()) != g.ctx.end())
        return invalid("ctx must be strictly increasing");
    for (auto n : g.ctx)
        if (n < 16384 || n > 1048576 || n % kMargin != 0)
            return invalid("ctx must be 4096-aligned in [16384, 1048576]");
    std::set<std::uint32_t> batches, drafts;
    for (auto n : g.ubatch)
        if (n < 32 || n > g.batch || !batches.insert(n).second)
            return invalid("ubatch must be unique, at least 32 and at most batch");
    for (auto n : g.mtp)
        if (n > 3 || !drafts.insert(n).second) return invalid("mtp must contain unique integers in [0, 3]");
    if (!drafts.contains(0)) return invalid("mtp must include 0 for the non-speculative baseline");
    std::set<std::string> spec_types;
    for (const auto &type : g.spec_type)
        if ((type != "draft-mtp" && type != "draft-mtp,ngram-mod") || !spec_types.insert(type).second)
            return invalid("spec_type must contain unique supported values");
    std::set<double> pmins;
    for (const auto p : g.p_min)
        if (!std::isfinite(p) || p < 0 || p > 1 || !pmins.insert(p).second)
            return invalid("p_min must contain unique values in [0, 1]");
    const auto &p = g.profile;
    if (p.parallel != 1 || p.cache_ram > 1048576 || p.ctx_checkpoints == 0 ||
        p.ctx_checkpoints > 128 || p.checkpoint_min_step == 0 || p.checkpoint_min_step > 1048576 ||
        !std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1 ||
        (p.reasoning_format != "deepseek" && p.reasoning_format != "none"))
        return invalid("profile values are outside their bounds");
    return validate_spill_guard(g.spill);
}
Result<Grid> parse_grid(const json::Value &value) {
    if (!value.is_object()) return invalid("grid must be a JSON object");
    Grid grid;
    for (const auto &[key, item] : value.as_object()) {
        if (key == "kv") {
            if (!item.is_array()) return invalid("kv must be an array of [K,V] pairs");
            grid.kv.clear();
            for (const auto &pair : item.as_array()) {
                if (!pair.is_array() || pair.as_array().size() != 2 ||
                    !pair.as_array()[0].is_string() || !pair.as_array()[1].is_string())
                    return invalid("kv must be an array of [K,V] string pairs");
                grid.kv.push_back({pair.as_array()[0].as_string(), pair.as_array()[1].as_string()});
            }
        } else if (key == "ctx" || key == "ubatch" || key == "mtp") {
            if (!item.is_array()) return invalid("ctx, ubatch and mtp must be arrays");
            if (key == "ctx") grid.ctx.clear();
            if (key == "ubatch") grid.ubatch.clear();
            if (key == "mtp") grid.mtp.clear();
            for (const auto &n : item.as_array()) {
                if (!n.is_integer() || n.as_int(-1) < 0 || n.as_uint() > 1048576)
                    return invalid("grid arrays require bounded nonnegative integers");
                if (key == "ctx") grid.ctx.push_back(n.as_uint());
                if (key == "ubatch") grid.ubatch.push_back(static_cast<std::uint32_t>(n.as_uint()));
                if (key == "mtp") grid.mtp.push_back(static_cast<std::uint32_t>(n.as_uint()));
            }
        } else if (key == "spec_type" || key == "p_min") {
            if (!item.is_array()) return invalid("spec_type and p_min must be arrays");
            if (key == "spec_type") {
                grid.spec_type.clear();
                for (const auto &v : item.as_array()) {
                    if (!v.is_string()) return invalid("spec_type entries must be strings");
                    grid.spec_type.push_back(v.as_string());
                }
            } else {
                grid.p_min.clear();
                for (const auto &v : item.as_array()) {
                    if (!v.is_number()) return invalid("p_min entries must be numbers");
                    grid.p_min.push_back(v.as_double());
                }
            }
        } else if (key == "profile") {
            if (!item.is_object()) return invalid("profile must be an object");
            for (const auto &[profile_key, profile_value] : item.as_object()) {
                if (profile_key == "parallel" && profile_value.is_integer() && profile_value.as_int(-1) >= 0 && profile_value.as_uint() <= 1)
                    grid.profile.parallel = static_cast<std::uint32_t>(profile_value.as_uint());
                else if (profile_key == "no_kv_unified" && profile_value.is_bool()) grid.profile.no_kv_unified = profile_value.as_bool();
                else if (profile_key == "cache_ram" && profile_value.is_integer() && profile_value.as_int(-1) >= 0 && profile_value.as_uint() <= 1048576)
                    grid.profile.cache_ram = static_cast<std::uint32_t>(profile_value.as_uint());
                else if (profile_key == "ctx_checkpoints" && profile_value.is_integer() && profile_value.as_int(-1) > 0 && profile_value.as_uint() <= 128)
                    grid.profile.ctx_checkpoints = static_cast<std::uint32_t>(profile_value.as_uint());
                else if (profile_key == "checkpoint_min_step" && profile_value.is_integer() && profile_value.as_int(-1) > 0 && profile_value.as_uint() <= 1048576)
                    grid.profile.checkpoint_min_step = profile_value.as_uint();
                else if (profile_key == "jinja" && profile_value.is_bool()) grid.profile.jinja = profile_value.as_bool();
                else if (profile_key == "reasoning_format" && profile_value.is_string()) grid.profile.reasoning_format = profile_value.as_string();
                else if (profile_key == "min_p" && profile_value.is_number()) grid.profile.min_p = profile_value.as_double();
                else if (profile_key == "metrics" && profile_value.is_bool()) grid.profile.metrics = profile_value.as_bool();
                else return invalid("unknown or invalid profile key");
            }
        } else if (key == "batch" || key == "spill_threshold_mib" || key == "shared_baseline_mib") {
            if (!item.is_integer() || item.as_int(-1) < 0 || item.as_uint() > 1048576)
                return invalid("grid scalar is not a bounded nonnegative integer");
            if (key == "batch") grid.batch = static_cast<std::uint32_t>(item.as_uint());
            if (key == "spill_threshold_mib") grid.spill.threshold_bytes = item.as_uint() * kMiB;
            if (key == "shared_baseline_mib") grid.spill.baseline_bytes = item.as_uint() * kMiB;
        } else return invalid("unknown grid key");
    }
    if (auto status = validate_grid(grid); !status.ok()) return status;
    return grid;
}
std::vector<Candidate> planned_candidates(const Grid &grid, bool mtp_available) {
    std::vector<Candidate> candidates;
    for (const auto &kv : grid.kv)
        for (auto ctx : grid.ctx)
            for (auto ubatch : grid.ubatch)
                for (auto mtp : grid.mtp) {
                    if (mtp != 0 && !mtp_available) continue;
                    if (mtp == 0) candidates.push_back({kv, ctx, ubatch, mtp});
                    else for (const auto &type : grid.spec_type)
                        for (const auto p : grid.p_min) candidates.push_back({kv, ctx, ubatch, mtp, type, p});
                }
    return candidates;
}
std::vector<std::string> arguments(const Options &o, const Candidate &c) {
    std::vector<std::string> args{"--model", o.model, "--ctx-size", std::to_string(c.ctx),
        "--cache-type-k", c.kv.k, "--cache-type-v", c.kv.v, "--ubatch-size", std::to_string(c.ubatch),
        "--batch-size", std::to_string(o.grid.batch), "--flash-attn", "on", "--parallel", std::to_string(o.grid.profile.parallel),
        "--n-gpu-layers", "999", "--fit", "off"};
    args.emplace_back(o.grid.profile.no_kv_unified ? "--no-kv-unified" : "--kv-unified");
    args.insert(args.end(), {"--cache-ram", std::to_string(o.grid.profile.cache_ram), "--ctx-checkpoints",
        std::to_string(o.grid.profile.ctx_checkpoints), "--checkpoint-min-step", std::to_string(o.grid.profile.checkpoint_min_step)});
    args.emplace_back(o.grid.profile.jinja ? "--jinja" : "--no-jinja");
    args.insert(args.end(), {"--reasoning-format", o.grid.profile.reasoning_format});
    args.insert(args.end(), {"--min-p", std::to_string(o.grid.profile.min_p)});
    if (o.grid.profile.metrics) args.emplace_back("--metrics");
    if (!o.mmproj.empty()) args.insert(args.end(), {"--mmproj", o.mmproj});
    // Explicit none avoids inheriting a speculative default from the environment.
    // Older executables without --spec-type support use the ordinary default.
    if (c.mtp != 0)
        args.insert(args.end(), {"--spec-type", c.spec_type, "--spec-draft-n-max", std::to_string(c.mtp),
            "--spec-draft-p-min", std::to_string(c.p_min)});
    else if (o.spec_type_supported) args.insert(args.end(), {"--spec-type", "none"});
    return args;
}

Report search(const Options &o, bool mtp_available, Deadline deadline, const Trial &trial, const Now &now) {
    Report report;
    if (auto status = validate_grid(o.grid); !status.ok()) {
        report.stopped_reason = status.message();
        return report;
    }
    const auto began = now();
    const auto probe_deadline = began + (deadline - began) * 3 / 5;
    const auto base_ubatch = *std::min_element(o.grid.ubatch.begin(), o.grid.ubatch.end());
    bool stopped = false;
    const auto run = [&](const Candidate &c, Phase phase, Deadline until) -> std::optional<std::size_t> {
        if (stopped || now() >= until) return {};
        auto m = trial(c, phase, until);
        m.candidate = c;
        m.phase = phase;
        if (m.stop_search || m.verdict == Verdict::release_failed || m.verdict == Verdict::cancelled ||
            m.verdict == Verdict::unsupported || !m.memory_released) {
            stopped = true;
            report.stopped_reason = m.detail.empty() ? to_string(m.verdict) : m.detail;
        }
        report.results.push_back(std::move(m));
        return report.results.size() - 1;
    };
    std::vector<Bisection> lanes;
    std::vector<bool> abandoned(o.grid.kv.size(), false);
    for (std::size_t i = 0; i < o.grid.kv.size(); ++i) lanes.emplace_back(o.grid.ctx);
    // Round-robin so one pair cannot monopolize the cheap-probe budget.
    bool progress = true;
    while (progress && !stopped && now() < probe_deadline) {
        progress = false;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const auto ctx = lanes[i].next();
            if (abandoned[i] || !ctx) continue;
            const auto index = run({o.grid.kv[i], *ctx, base_ubatch, 0}, Phase::probe, probe_deadline);
            if (!index) break;
            progress = true;
            const auto verdict = report.results[*index].verdict;
            if (verdict == Verdict::clean || verdict == Verdict::spill) lanes[i].observe(verdict == Verdict::clean);
            else abandoned[i] = true;
        }
    }
    const auto measure = [&](Candidate c) {
        // The SAME profile must be clean one step above the recommendation.
        Candidate safety = c;
        safety.ctx += kMargin;
        bool proved = false;
        for (const auto &m : report.results)
            if (m.candidate == safety && m.verdict == Verdict::clean && m.memory_released) proved = true;
        if (!proved) {
            const auto p = run(safety, Phase::probe, deadline);
            if (!p || report.results[*p].verdict != Verdict::clean) return;
        }
        for (const auto &m : report.results)
            if (m.candidate == c && m.phase == Phase::benchmark) return;
        const auto index = run(c, Phase::benchmark, deadline);
        if (index) report.results[*index].safety_probe_ctx = safety.ctx;
    };
    // Time only the measured clean edges and the >=64k decode baseline.
    for (std::size_t i = 0; i < lanes.size() && !stopped; ++i) {
        if (abandoned[i] || !lanes[i].edge()) continue;
        const auto ctx = margin_context(*lanes[i].edge());
        if (!ctx) continue;
        if (*ctx >= kFastContext) measure({o.grid.kv[i], kFastContext, base_ubatch, 0});
        measure({o.grid.kv[i], *ctx, base_ubatch, 0});
    }
    rank(report);
    std::vector<Candidate> leaders;
    if (report.fast_default) leaders.push_back(report.results[*report.fast_default].candidate);
    if (report.long_context) {
        const auto c = report.results[*report.long_context].candidate;
        if (leaders.empty() || !(leaders.front() == c)) leaders.push_back(c);
    }
    // At most two leaders; interleave profiles to keep MTP and ubatch coverage
    // useful under a short budget. Every variant gets its own safety probe.
    for (auto ubatch : o.grid.ubatch)
        for (auto mtp : o.grid.mtp)
            if (mtp == 0 || mtp_available)
                for (auto c : leaders) {
                    c.ubatch = ubatch;
                    c.mtp = mtp;
                    if (mtp == 0) {
                        c.spec_type = "draft-mtp";
                        c.p_min = 0.0;
                        measure(c);
                    } else {
                        for (const auto &spec_type : o.grid.spec_type)
                            for (const auto p_min : o.grid.p_min) {
                                c.spec_type = spec_type;
                                c.p_min = p_min;
                                measure(c);
                            }
                    }
                }
    report.budget_exhausted = now() >= deadline || std::any_of(report.results.begin(), report.results.end(),
        [](const Measurement &m) { return m.verdict == Verdict::timed_out; });
    rank(report);
    return report;
}
void rank(Report &report) {
    report.fast_default.reset();
    report.long_context.reset();
    for (std::size_t i = 0; i < report.results.size(); ++i) {
        const auto &m = report.results[i];
        if (!eligible(m)) continue;
        if (m.candidate.ctx >= kFastContext &&
            (!report.fast_default || faster(m, report.results[*report.fast_default]))) report.fast_default = i;
        if (!report.long_context || m.candidate.ctx > report.results[*report.long_context].candidate.ctx ||
            (m.candidate.ctx == report.results[*report.long_context].candidate.ctx &&
             faster(m, report.results[*report.long_context]))) report.long_context = i;
    }
}
json::Value report_json(const Options &o, const Report &report) {
    const auto chosen = report.fast_default ? report.fast_default : report.long_context;
    json::Object doc = chosen ? spawn_config(o, report.results[*chosen].candidate) : json::Object{};
    json::Array rows;
    for (const auto &m : report.results) rows.push_back(measurement_json(m));
    const auto recommendation = [&](std::optional<std::size_t> index) -> json::Value {
        if (!index) return {};
        return json::Object{{"result_index", static_cast<std::uint64_t>(*index)},
            {"config", spawn_config(o, report.results[*index].candidate)}};
    };
    doc.set("results", json::Object{{"schema", "sonder.inference.tune/1"},
        {"margin_tokens", kMargin}, {"budget_exhausted", report.budget_exhausted},
        {"stopped_reason", report.stopped_reason}, {"thorough", o.thorough},
        {"fast_default", recommendation(report.fast_default)}, {"long_context", recommendation(report.long_context)},
        {"candidates", std::move(rows)}});
    return doc;
}
std::string markdown(const Report &report) {
    std::ostringstream out;
    out << "# Calibration results\n\n";
    const auto recommendation = [&](const char *name, std::optional<std::size_t> index) {
        out << "- **" << name << "**: ";
        if (!index) { out << "unavailable (no eligible measurement).\n"; return; }
        const auto &m = report.results[*index];
        out << m.candidate.kv.k << '/' << m.candidate.kv.v << ", ctx " << m.candidate.ctx
            << ", ubatch " << m.candidate.ubatch << ", MTP " << m.candidate.mtp
            << ", spec_type " << (m.candidate.mtp == 0 ? "none" : m.candidate.spec_type);
        if (m.candidate.mtp != 0) out << ", p_min " << m.candidate.p_min;
        out << ", " << std::fixed << std::setprecision(1) << *weighted_decode_tps(m.workloads)
            << " weighted decode tok/s.\n";
    };
    recommendation("Fast default (>=64k)", report.fast_default);
    recommendation("Long context", report.long_context);
    out << "\nRecommendations reserve 4096 tokens below a clean probe of the same profile.\n"
           "Fastest means fastest measured within this budget; the search is not exhaustive.\n\n"
           "| Phase | K/V | ctx | ubatch | MTP | spec_type | p_min | shared MiB | dedicated MiB | weighted decode tok/s | prefill tok/s | Verdict |\n"
           "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|\n";
    const auto number = [&](const auto &n, double divisor) {
        if (n) out << std::fixed << std::setprecision(1) << static_cast<double>(*n) / divisor;
        else out << "-";
    };
    for (const auto &m : report.results) {
        out << "| " << (m.phase == Phase::probe ? "probe" : "bench") << " | " << m.candidate.kv.k << '/'
            << m.candidate.kv.v << " | " << m.candidate.ctx << " | " << m.candidate.ubatch << " | "
            << m.candidate.mtp << " | " << (m.candidate.mtp == 0 ? "none" : m.candidate.spec_type) << " | ";
        if (m.candidate.mtp != 0) out << m.candidate.p_min;
        else out << "-";
        out << " | ";
        number(m.shared_bytes, static_cast<double>(kMiB)); out << " | ";
        number(m.dedicated_bytes, static_cast<double>(kMiB)); out << " | ";
        number(weighted_decode_tps(m.workloads), 1); out << " | "; number(m.prefill_tps, 1);
        out << " | " << to_string(m.verdict);
        if (m.slow_kernel && m.verdict != Verdict::slow_kernel) out << "; slow kernel";
        out << " |\n";
    }
    if (report.budget_exhausted) out << "\nBudget exhausted; unmeasured profiles are not recommendations.\n";
    if (!report.stopped_reason.empty()) out << "\nStopped: " << report.stopped_reason << '\n';
    return out.str();
}
} // namespace sonder::inference::llamaserver::tune
