// Markdown rendering and file output for benchmark results documents.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/benchmark.hpp"

namespace sonder::inference::bench {

namespace {

const json::Value* path(const json::Value& v, std::initializer_list<std::string_view> keys) {
    const json::Value* cur = &v;
    for (auto k : keys) {
        cur = cur->find(k);
        if (cur == nullptr) return nullptr;
    }
    return cur;
}

std::string raw_text(const json::Value& v, std::initializer_list<std::string_view> keys) {
    const json::Value* f = path(v, keys);
    if (f == nullptr || f->is_null()) return "-";
    if (f->is_string()) return f->as_string().empty() ? "-" : f->as_string();
    return f->dump();
}

// Makes a value safe inside a Markdown table cell or heading: '|' would end
// the cell, a line break would end the row, and a backslash could escape the
// next character, so escape/flatten them.
std::string md(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '|': out += "\\|"; break;
            case '\\': out += "\\\\"; break;
            case '\n':
            case '\r': out.push_back(' '); break;
            default: out.push_back(c);
        }
    }
    return out;
}

std::string text(const json::Value& v, std::initializer_list<std::string_view> keys) {
    return md(raw_text(v, keys));
}

std::string num(double x, int prec = 1) {
    if (!std::isfinite(x)) return "-";
    char buf[64];
    // Fixed notation for huge values would not fit (1e300 has 301 digits).
    std::snprintf(buf, sizeof buf, std::fabs(x) < 1e15 ? "%.*f" : "%.*e", prec, x);
    return buf;
}

// "p50" etc. of a distribution object, "-" when n == 0.
std::string stat(const json::Value& dist_parent, std::string_view dist, std::string_view which, int prec = 1) {
    const json::Value* d = dist_parent.find(dist);
    if (d == nullptr || d->find("n") == nullptr || d->find("n")->as_int() == 0) return "-";
    const json::Value* f = d->find(which);
    return f ? num(f->as_double(), prec) : "-";
}

std::string safe(std::string s) {
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
        if (!ok) c = '_';
    }
    return s.empty() ? std::string("unknown") : s;
}

std::string mib_cell(const json::Value* gpu, std::string_view primary, std::string_view fallback = {}) {
    if (gpu == nullptr || !gpu->is_object()) return "-";
    const json::Value* f = gpu->find(primary);
    if ((f == nullptr || f->is_null()) && !fallback.empty()) {
        f = gpu->find(fallback);
    }
    if (f == nullptr || f->is_null()) return "-";
    if (f->is_integer()) return f->dump();
    if (f->is_number()) return num(f->as_double(), 0);
    return "-";
}

// Free VRAM cell: only shows a number when dedicated_free_mib is actually
// reported (non-null), never substituting total VRAM when free is unknown.
std::string free_mib_cell(const json::Value* gpu) {
    if (gpu == nullptr || !gpu->is_object()) return "-";
    const json::Value* dedicated_free = gpu->find("dedicated_free_mib");
    if (dedicated_free == nullptr || dedicated_free->is_null()) return "-";
    return mib_cell(gpu, "effective_free_vram_mib", "dedicated_free_mib");
}

const json::Value* find_gpu_entry(const json::Array& arr, const std::string& id, std::size_t idx) {
    if (!id.empty()) {
        for (const auto& item : arr) {
            if (!item.is_object()) continue;
            const json::Value* item_id = item.find("id");
            if (item_id != nullptr && item_id->is_string() && item_id->as_string() == id) {
                return &item;
            }
        }
    }
    if (idx < arr.size() && arr[idx].is_object()) {
        const json::Value* item_id = arr[idx].find("id");
        const std::string item_id_str = (item_id != nullptr && item_id->is_string()) ? item_id->as_string() : std::string{};
        if (item_id_str == id) {
            return &arr[idx];
        }
    }
    return nullptr;
}

}  // namespace

std::string render_markdown(const json::Value& r) {
    std::ostringstream o;
    const std::string backend = text(r, {"backend", "name"});
    o << "# Sonder bench: " << text(r, {"model", "name"}) << " via " << backend << "\n\n";
    if (backend == "mock") {
        o << "> **MOCK BACKEND: not a performance claim.** Harness check only.\n\n";
    }
    if (const json::Value* t = r.find("truncated_by_budget"); t && t->as_bool()) {
        o << "> **Truncated:** the time budget was reached before all iterations ran.\n\n";
    }
    std::string hw = text(r, {"host", "hardware"});
    std::string cpu = "-";
    if (const json::Value* devs = path(r, {"host", "devices"}); devs && !devs->as_array().empty()) {
        const auto& d0 = devs->as_array().front();
        cpu = text(d0, {"name"}) + ", " + text(d0, {"logical_cores"}) + " logical cores, " +
              num(static_cast<double>(d0.find("total_memory_bytes") ? d0.find("total_memory_bytes")->as_int() : 0) /
                      (1024.0 * 1024.0 * 1024.0),
                  1) +
              " GiB RAM";
    }
    o << "| Field | Value |\n|---|---|\n";
    o << "| Date (UTC) | " << text(r, {"created_at"}) << " |\n";
    o << "| Label | " << text(r, {"label"}) << " |\n";
    o << "| Backend | " << backend << " " << text(r, {"backend", "version"}) << " |\n";
    o << "| Model | " << text(r, {"model", "name"}) << " (" << text(r, {"model", "parameter_size"}) << ", "
      << text(r, {"model", "quantization"}) << ", " << text(r, {"model", "format"}) << ") |\n";
    o << "| Hardware | " << hw << " |\n";
    o << "| Host | " << text(r, {"host", "platform"}) << "; " << cpu << " |\n";
    o << "| Engine | sonder-inference " << text(r, {"engine", "version"}) << " @ " << text(r, {"engine", "commit"})
      << " |\n";
    o << "| Corpus | " << text(r, {"corpus", "name"}) << " v" << text(r, {"corpus", "version"}) << " (fnv1a "
      << text(r, {"corpus", "fnv1a"}) << ") |\n";
    o << "| Runs | warmup " << text(r, {"config", "warmup_runs"}) << " + measured "
      << text(r, {"config", "measured_runs"}) << " per prompt; temperature " << text(r, {"sampling", "temperature"})
      << ", seed " << text(r, {"sampling", "seed"}) << "; cache: " << text(r, {"config", "cache_state"}) << " |\n";
    const json::Value* conc = path(r, {"config", "concurrency"});
    if (conc == nullptr) conc = path(r, {"summary", "concurrency"});
    if (conc == nullptr) conc = path(r, {"options", "concurrency"});
    if (conc != nullptr && !conc->is_null()) {
        std::string conc_text;
        if (conc->is_integer()) {
            conc_text = conc->dump();
        } else if (conc->is_number()) {
            conc_text = num(conc->as_double(), 0);
        } else if (conc->is_string() && !conc->as_string().empty()) {
            conc_text = md(conc->as_string());
        }
        if (!conc_text.empty()) {
            o << "| Concurrency | " << conc_text << " |\n";
        }
    }
    const json::Value* cold = r.find("cold_first_request_ms");
    const json::Value* cold_load = r.find("cold_first_request_backend_load_ms");
    o << "| Cold first request | " << (cold ? num(cold->as_double()) : "-") << " ms (backend load "
      << (cold_load ? num(cold_load->as_double()) : "-") << " ms) |\n";
    const json::Value* wall = r.find("wall_seconds");
    o << "| Wall time | " << (wall ? num(wall->as_double()) : "-") << " s |\n";
    if (const json::Value* s = r.find("summary")) {
        const json::Value* draft_total_v = s->find("draft_tokens_total");
        if (draft_total_v != nullptr && draft_total_v->is_integer() && draft_total_v->as_uint(0) > 0) {
            const std::uint64_t draft_total = draft_total_v->as_uint(0);
            std::uint64_t denom = draft_total;
            const json::Value* draft_eval_v = s->find("draft_tokens_evaluated_total");
            if (draft_eval_v != nullptr && draft_eval_v->is_integer() && draft_eval_v->as_uint(0) > 0) {
                denom = draft_eval_v->as_uint(0);
            }
            const json::Value* draft_acc_v = s->find("draft_accepted_total");
            const json::Value* rate_v = s->find("draft_acceptance_rate");
            o << "| Speculation | ";
            if (draft_acc_v != nullptr && draft_acc_v->is_integer() && draft_acc_v->as_double(-1.0) >= 0.0) {
                const std::uint64_t draft_acc = std::min(draft_acc_v->as_uint(0), denom);
                double pct = static_cast<double>(draft_acc) * 100.0 / static_cast<double>(denom);
                if (rate_v != nullptr && rate_v->is_number()) {
                    const double rate = rate_v->as_double();
                    if (std::isfinite(rate) && rate >= 0.0 && rate <= 1.0) {
                        pct = rate * 100.0;
                    }
                }
                pct = std::clamp(pct, 0.0, 100.0);
                o << draft_acc << " / " << denom << " (" << num(pct, 1) << "%)";
            } else {
                o << draft_total << " draft tokens";
            }
            o << " |\n";
        }
    }
    o << "\n";

    if (const json::Value* s = r.find("summary")) {
        const json::Value* agg = s->find("aggregate_decode_tokens_per_sec");
        const bool has_agg = agg != nullptr && agg->is_number();
        o << "## Overall (measured requests)\n\n";
        o << "| Requests | Failures | Cancelled | TTFT p50 ms | TTFT p95 ms | Total p50 ms | Total p95 ms | "
             "Decode tok/s p50 |";
        if (has_agg) {
            o << " Aggregate decode tok/s |";
        }
        o << " Prompt tok/s p50 |\n|---:|---:|---:|---:|---:|---:|---:|---:|";
        if (has_agg) {
            o << "---:|";
        }
        o << "---:|\n";
        o << "| " << text(*s, {"measured_requests"}) << " | " << text(*s, {"failures"}) << " | "
          << text(*s, {"cancelled"}) << " | " << stat(*s, "ttft_ms", "p50") << " | " << stat(*s, "ttft_ms", "p95")
          << " | " << stat(*s, "total_ms", "p50") << " | " << stat(*s, "total_ms", "p95") << " | "
          << stat(*s, "decode_tokens_per_sec", "p50") << " | ";
        if (has_agg) {
            const json::Value* mr = s->find("measured_requests");
            const bool any_req = mr == nullptr || !mr->is_integer() || mr->as_uint(1) > 0;
            o << (any_req ? num(agg->as_double()) : "-") << " | ";
        }
        o << stat(*s, "prompt_tokens_per_sec", "p50") << " |\n\n";
    }

    if (const json::Value* pp = r.find("per_prompt"); pp && !pp->as_array().empty()) {
        o << "## Per prompt (p50 / p95)\n\n";
        o << "| Prompt | Workload | Req | Fail | TTFT ms | Total ms | Decode tok/s (backend) | Decode tok/s (client) | "
             "Prompt tok/s | Output tok |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        bool any_fanout = false;
        for (const auto& p : pp->as_array()) {
            any_fanout = any_fanout || p.find("fanout_makespan_ms") != nullptr;
            o << "| " << text(p, {"prompt_id"}) << " | " << text(p, {"workload"}) << " | "
              << text(p, {"measured_requests"}) << " | " << text(p, {"failures"}) << " | "
              << stat(p, "ttft_ms", "p50") << " / " << stat(p, "ttft_ms", "p95") << " | "
              << stat(p, "total_ms", "p50") << " / " << stat(p, "total_ms", "p95") << " | "
              << stat(p, "decode_tokens_per_sec", "p50") << " | " << stat(p, "client_decode_tokens_per_sec", "p50")
              << " | " << stat(p, "prompt_tokens_per_sec", "p50") << " | " << stat(p, "completion_tokens", "p50", 0)
              << " |\n";
        }
        if (any_fanout) {
            o << "\n## Agent fan-out (children issued concurrently)\n\n";
            o << "| Prompt | Children | Shared prefix bytes | Makespan ms p50 | Makespan ms p95 | Aggregate tok/s p50 |\n"
                 "|---|---:|---:|---:|---:|---:|\n";
            for (const auto& p : pp->as_array()) {
                if (p.find("fanout_makespan_ms") == nullptr) continue;
                o << "| " << text(p, {"prompt_id"}) << " | " << text(p, {"requests_per_iteration"}) << " | "
                  << text(p, {"shared_prefix_bytes"}) << " | " << stat(p, "fanout_makespan_ms", "p50") << " | "
                  << stat(p, "fanout_makespan_ms", "p95") << " | "
                  << stat(p, "fanout_aggregate_tokens_per_sec", "p50") << " |\n";
            }
        }
    }

    const json::Value* gpus_before = r.find("gpus_before");
    const json::Value* gpus_after = r.find("gpus_after");
    const bool has_before = gpus_before != nullptr && gpus_before->is_array() && !gpus_before->as_array().empty();
    const bool has_after = gpus_after != nullptr && gpus_after->is_array() && !gpus_after->as_array().empty();
    if (has_before || has_after) {
        struct GpuRow {
            const json::Value* before = nullptr;
            const json::Value* after = nullptr;
        };
        std::vector<GpuRow> gpu_rows;
        const json::Array empty_gpus;
        const json::Array& before_arr = has_before ? gpus_before->as_array() : empty_gpus;
        const json::Array& after_arr = has_after ? gpus_after->as_array() : empty_gpus;
        if (has_after) {
            for (std::size_t idx = 0; idx < after_arr.size(); ++idx) {
                const auto& g = after_arr[idx];
                if (!g.is_object()) continue;
                const json::Value* id_val = g.find("id");
                const std::string id_str = (id_val != nullptr && id_val->is_string()) ? id_val->as_string() : std::string{};
                gpu_rows.push_back(GpuRow{find_gpu_entry(before_arr, id_str, idx), &g});
            }
            for (std::size_t idx = 0; idx < before_arr.size(); ++idx) {
                const auto& g = before_arr[idx];
                if (!g.is_object()) continue;
                const json::Value* id_val = g.find("id");
                const std::string id_str = (id_val != nullptr && id_val->is_string()) ? id_val->as_string() : std::string{};
                if (find_gpu_entry(after_arr, id_str, idx) == nullptr) {
                    gpu_rows.push_back(GpuRow{&g, nullptr});
                }
            }
        } else {
            for (const auto& g : before_arr) {
                if (!g.is_object()) continue;
                gpu_rows.push_back(GpuRow{&g, nullptr});
            }
        }
        if (!gpu_rows.empty()) {
            o << "\n## GPUs\n\n";
            o << "| ID | Name | Vendor | Total MiB | Free Before MiB | Free After MiB | Peak Delta MiB | Backend | "
                 "Source |\n"
                 "|---|---|---|---:|---:|---:|---:|---|---|\n";
            for (const auto& row : gpu_rows) {
                const json::Value& primary = row.after != nullptr ? *row.after : *row.before;
                o << "| " << text(primary, {"id"}) << " | " << text(primary, {"name"}) << " | "
                  << text(primary, {"vendor"}) << " | "
                  << mib_cell(&primary, "effective_vram_mib", "dedicated_total_mib") << " | "
                  << free_mib_cell(row.before) << " | " << free_mib_cell(row.after) << " | "
                  << mib_cell(row.after, "peak_vram_delta_mib") << " | "
                  << text(primary, {"backend_hint"}) << " | " << text(primary, {"source"}) << " |\n";
            }
        }
    }

    o << "\nTTFT and total latency are client-measured by the session (request start to first chunk / completion). "
         "Backend decode tok/s = eval_count / eval_duration when the backend reports it. Client decode tok/s = "
         "(output tokens - 1) / (total - TTFT). Do not compare across prompt lengths, quantizations, or "
         "concurrency levels (docs/BENCHMARK_PLAN.md).\n";
    return o.str();
}

std::string default_result_stem(const json::Value& r) {
    std::string date = raw_text(r, {"created_at"});
    date = date.size() >= 10 ? date.substr(0, 10) : std::string("undated");
    return safe(date) + "-" + safe(raw_text(r, {"backend", "name"})) + "-" + safe(raw_text(r, {"model", "name"})) +
           "-" + safe(raw_text(r, {"corpus", "name"}));
}

Status write_results(const json::Value& r, const std::string& dir, const std::string& stem, std::string* json_path,
                     std::string* markdown_path) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return {ErrorCode::io_error, "cannot create " + dir + ": " + ec.message()};
    }
    const std::string jp = (std::filesystem::path(dir) / (stem + ".json")).string();
    const std::string mp = (std::filesystem::path(dir) / (stem + ".md")).string();
    {
        std::ofstream out(jp, std::ios::binary | std::ios::trunc);
        out << r.dump() << "\n";
        if (!out) return {ErrorCode::io_error, "cannot write " + jp};
    }
    {
        std::ofstream out(mp, std::ios::binary | std::ios::trunc);
        out << render_markdown(r);
        if (!out) return {ErrorCode::io_error, "cannot write " + mp};
    }
    if (json_path) *json_path = jp;
    if (markdown_path) *markdown_path = mp;
    return Status::success();
}

}  // namespace sonder::inference::bench
