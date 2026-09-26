// Markdown rendering and file output for benchmark results documents.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

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

std::string text(const json::Value& v, std::initializer_list<std::string_view> keys) {
    const json::Value* f = path(v, keys);
    if (f == nullptr || f->is_null()) return "-";
    if (f->is_string()) return f->as_string().empty() ? "-" : f->as_string();
    return f->dump();
}

std::string num(double x, int prec = 1) {
    if (!std::isfinite(x)) return "-";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", prec, x);
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
    const json::Value* cold = r.find("cold_first_request_ms");
    const json::Value* cold_load = r.find("cold_first_request_backend_load_ms");
    o << "| Cold first request | " << (cold ? num(cold->as_double()) : "-") << " ms (backend load "
      << (cold_load ? num(cold_load->as_double()) : "-") << " ms) |\n";
    const json::Value* wall = r.find("wall_seconds");
    o << "| Wall time | " << (wall ? num(wall->as_double()) : "-") << " s |\n\n";

    if (const json::Value* s = r.find("summary")) {
        o << "## Overall (measured requests)\n\n";
        o << "| Requests | Failures | Cancelled | TTFT p50 ms | TTFT p95 ms | Total p50 ms | Total p95 ms | "
             "Decode tok/s p50 | Prompt tok/s p50 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        o << "| " << text(*s, {"measured_requests"}) << " | " << text(*s, {"failures"}) << " | "
          << text(*s, {"cancelled"}) << " | " << stat(*s, "ttft_ms", "p50") << " | " << stat(*s, "ttft_ms", "p95")
          << " | " << stat(*s, "total_ms", "p50") << " | " << stat(*s, "total_ms", "p95") << " | "
          << stat(*s, "decode_tokens_per_sec", "p50") << " | " << stat(*s, "prompt_tokens_per_sec", "p50") << " |\n\n";
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
    o << "\nTTFT and total latency are client-measured by the session (request start to first chunk / completion). "
         "Backend decode tok/s = eval_count / eval_duration when the backend reports it. Client decode tok/s = "
         "(output tokens - 1) / (total - TTFT). Do not compare across prompt lengths, quantizations, or "
         "concurrency levels (docs/BENCHMARK_PLAN.md).\n";
    return o.str();
}

std::string default_result_stem(const json::Value& r) {
    std::string date = text(r, {"created_at"});
    date = date.size() >= 10 ? date.substr(0, 10) : std::string("undated");
    return safe(date) + "-" + safe(text(r, {"backend", "name"})) + "-" + safe(text(r, {"model", "name"})) + "-" +
           safe(text(r, {"corpus", "name"}));
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
