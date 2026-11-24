// sonder-infer: command-line interface over the Sonder Inference library.
// Reference: docs/CLI.md (commands, environment, exit codes, JSON shapes).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cli_runtime.hpp"
#include "sonder/inference.hpp"
#include "sonder_inference.h"
// Header-only CLI helpers (option specs, checked values, environment, chat
// files and the chat loop); shared with the unit tests (tests/test_cli_*.cpp).
#include "../../src/cli/cli_args.hpp"
#include "../../src/cli/cli_env.hpp"
#include "../../src/cli/cli_spec.hpp"
#include "../../src/cli/cli_values.hpp"
#include "../../src/cli/sonder_infer_commands.hpp"
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "../../src/backends/llamaserver/tune.hpp"
#endif
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/coresidency_planner.hpp"
#include "sonder/inference/placement_planner.hpp"
#include "sonder/inference/server.hpp"
#endif
#if defined(SONDER_HAS_BENCH)
#include "sonder/inference/benchmark.hpp"
#endif
#if defined(SONDER_HAS_WEIGHT_MANIFEST)
#include "sonder/inference/weight_manifest.hpp"
#endif

namespace si = sonder::inference;
namespace cli = sonder::cli;

namespace {

using cli::Args;
using namespace sonder::infer_cli;

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
int cmd_version(const Args& a) {
    if (a.has("json")) {
        si::json::Array backends;
        for (const auto& n : compiled_backends()) backends.emplace_back(n);
#if defined(SONDER_HAS_SERVER)
        const si::json::Value api_version(si::server::kApiVersion);
        const bool server = true;
#else
        const si::json::Value api_version(nullptr);
        const bool server = false;
#endif
#if defined(SONDER_HAS_BENCH)
        const bool bench = true;
#else
        const bool bench = false;
#endif
        const si::json::Object doc{{"name", "sonder-infer"},
                                   {"version", si::version_string()},
                                   {"commit", si::build_commit()},
                                   {"abi_version", SONDER_ABI_VERSION},
                                   {"api_version", api_version},
                                   {"platform", si::host_platform()},
                                   {"backends", std::move(backends)},
                                   {"modules", si::json::Object{{"server", server}, {"bench", bench}}}};
        std::cout << si::json::Value(doc).dump() << "\n";
        return cli::kExitOk;
    }
    std::cout << "sonder-infer " << si::version_string() << " (" << si::build_commit() << ") C ABI "
              << SONDER_ABI_VERSION;
#if defined(SONDER_HAS_SERVER)
    std::cout << " HTTP API " << si::server::kApiVersion;
#endif
    std::cout << "\n";
    return cli::kExitOk;
}

constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;

si::json::Value reading_mib_json(const si::MemoryReading& reading) {
    if (!reading.is_known()) return si::json::Value(nullptr);
    return si::json::Value(reading.bytes_or(0) / kBytesPerMiB);
}

si::json::Value reading_bytes_json(const si::MemoryReading& reading) {
    if (!reading.is_known()) return si::json::Value(nullptr);
    return si::json::Value(reading.bytes_or(0));
}

si::json::Value optional_u64_json(std::optional<std::uint64_t> value) {
    return value ? si::json::Value(*value) : si::json::Value(nullptr);
}

std::string format_reading_mib(const si::MemoryReading& reading) {
    if (!reading.is_known()) return "unknown";
    return std::to_string(reading.bytes_or(0) / kBytesPerMiB);
}

si::json::Object gpu_to_json(const si::GpuDeviceInfo& g) {
    return si::json::Object{
        {"id", g.id},
        {"name", g.name},
        {"vendor", si::to_string(g.vendor)},
        {"vendor_id", g.vendor_id},
        {"device_id", g.device_id},
        {"dedicated_total_mib", reading_mib_json(g.dedicated_total)},
        {"dedicated_free_mib", reading_mib_json(g.dedicated_free)},
        {"shared_total_mib", reading_mib_json(g.shared_total)},
        {"effective_vram_mib", optional_u64_json(si::gpu_effective_vram_mib(g, false))},
        {"effective_free_vram_mib", optional_u64_json(si::gpu_effective_vram_mib(g, true))},
        {"is_integrated_uma", g.is_integrated_uma},
        {"backend_hint", g.backend_hint},
        {"source", g.source}};
}

int cmd_devices(const Args& a) {
    const auto devices = si::enumerate_devices();
    const auto gpus = si::enumerate_gpu_devices();
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& d : devices) {
            list.emplace_back(si::json::Object{{"id", d.id},
                                               {"kind", si::to_string(d.kind)},
                                               {"name", d.name},
                                               {"logical_cores", d.logical_cores},
                                               {"total_memory_bytes", d.total_memory_bytes},
                                               {"available_memory_bytes", d.available_memory_bytes}});
        }
        si::json::Array gpu_list;
        gpu_list.reserve(gpus.size());
        for (const auto& g : gpus) {
            gpu_list.emplace_back(gpu_to_json(g));
        }
        std::cout << si::json::Value(si::json::Object{{"platform", si::host_platform()},
                                                      {"devices", std::move(list)},
                                                      {"gpus", std::move(gpu_list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& d : devices) {
        std::cout << d.id << "  kind=" << si::to_string(d.kind) << "  name=\"" << d.name << "\""
                  << "  logical_cores=" << d.logical_cores << "  total_memory_mib=" << (d.total_memory_bytes >> 20)
                  << "  available_memory_mib=" << (d.available_memory_bytes >> 20) << "\n";
    }
    for (const auto& g : gpus) {
        std::cout << g.id << "  kind=gpu  vendor=" << si::to_string(g.vendor) << "  name=\"" << g.name << "\""
                  << "  dedicated_total_mib=" << format_reading_mib(g.dedicated_total)
                  << "  dedicated_free_mib=" << format_reading_mib(g.dedicated_free)
                  << "  shared_total_mib=" << format_reading_mib(g.shared_total)
                  << "  uma=" << (g.is_integrated_uma ? "true" : "false")
                  << "  backend_hint=" << (g.backend_hint.empty() ? "none" : g.backend_hint)
                  << "  source=" << (g.source.empty() ? "unknown" : g.source) << "\n";
    }
    std::cout << "platform=" << si::host_platform() << "\n";
    return cli::kExitOk;
}

int cmd_node(const Args& a) {
    const auto devices = si::enumerate_devices();
    const auto host = si::read_host_memory();
    const auto gpus = si::enumerate_gpu_devices();
    const auto backend_names = compiled_backends();
#if defined(SONDER_HAS_SERVER)
    const bool has_server = true;
#else
    const bool has_server = false;
#endif
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    const bool has_adaptive_spec = true;
#else
    const bool has_adaptive_spec = false;
#endif

    if (a.has("json")) {
#if defined(SONDER_HAS_SERVER)
        const si::json::Value api_version(si::server::kApiVersion);
#else
        const si::json::Value api_version(nullptr);
#endif
        si::json::Array dev_list;
        dev_list.reserve(devices.size());
        for (const auto& d : devices) {
            dev_list.emplace_back(si::json::Object{{"id", d.id},
                                                   {"kind", si::to_string(d.kind)},
                                                   {"name", d.name},
                                                   {"logical_cores", d.logical_cores},
                                                   {"total_memory_bytes", d.total_memory_bytes},
                                                   {"available_memory_bytes", d.available_memory_bytes}});
        }
        si::json::Array gpu_list;
        gpu_list.reserve(gpus.size());
        for (const auto& g : gpus) {
            gpu_list.emplace_back(gpu_to_json(g));
        }
        si::json::Array backends;
        backends.reserve(backend_names.size());
        for (const auto& n : backend_names) backends.emplace_back(n);

        si::json::Object host_mem{{"total_physical_bytes", reading_bytes_json(host.total_physical)},
                                  {"available_physical_bytes", reading_bytes_json(host.available_physical)},
                                  {"available_commit_bytes", reading_bytes_json(host.available_commit)},
                                  {"source", host.source}};
        si::json::Object caps{{"ternary_gguf", has_server},
                              {"placement_planner", has_server},
                              {"coresidency_planner", has_server},
                              {"multi_gpu", true},
                              {"uma_telemetry", true},
                              {"adaptive_speculation", has_adaptive_spec}};
        const si::json::Object doc{{"schema", "sonder.inference.node/1"},
                                   {"node_id", si::host_name()},
                                   {"platform", si::host_platform()},
                                   {"version", si::version_string()},
                                   {"commit", si::build_commit()},
                                   {"abi_version", SONDER_ABI_VERSION},
                                   {"api_version", api_version},
                                   {"host_memory", std::move(host_mem)},
                                   {"devices", std::move(dev_list)},
                                   {"gpus", std::move(gpu_list)},
                                   {"backends", std::move(backends)},
                                   {"capabilities", std::move(caps)}};
        std::cout << si::json::Value(doc).dump() << "\n";
        return cli::kExitOk;
    }

    std::cout << "node=" << si::host_name() << "  platform=" << si::host_platform()
              << "  version=" << si::version_string() << " (" << si::build_commit() << ")\n";
    std::cout << "host_memory: total_mib=" << format_reading_mib(host.total_physical)
              << "  available_mib=" << format_reading_mib(host.available_physical)
              << "  available_commit_mib=" << format_reading_mib(host.available_commit)
              << "  source=" << (host.source.empty() ? "unknown" : host.source) << "\n";
    for (const auto& d : devices) {
        std::cout << d.id << "  kind=" << si::to_string(d.kind) << "  name=\"" << d.name << "\""
                  << "  logical_cores=" << d.logical_cores << "  total_memory_mib=" << (d.total_memory_bytes >> 20)
                  << "  available_memory_mib=" << (d.available_memory_bytes >> 20) << "\n";
    }
    for (const auto& g : gpus) {
        std::cout << g.id << "  kind=gpu  vendor=" << si::to_string(g.vendor) << "  name=\"" << g.name << "\""
                  << "  dedicated_total_mib=" << format_reading_mib(g.dedicated_total)
                  << "  dedicated_free_mib=" << format_reading_mib(g.dedicated_free)
                  << "  shared_total_mib=" << format_reading_mib(g.shared_total)
                  << "  uma=" << (g.is_integrated_uma ? "true" : "false")
                  << "  backend_hint=" << (g.backend_hint.empty() ? "none" : g.backend_hint)
                  << "  source=" << (g.source.empty() ? "unknown" : g.source) << "\n";
    }
    std::cout << "backends:";
    for (const auto& n : backend_names) std::cout << " " << n;
    std::cout << "\n";
    std::cout << "capabilities:"
              << " ternary_gguf=" << (has_server ? "true" : "false")
              << " placement_planner=" << (has_server ? "true" : "false")
              << " coresidency_planner=" << (has_server ? "true" : "false") << " multi_gpu=true uma_telemetry=true"
              << " adaptive_speculation=" << (has_adaptive_spec ? "true" : "false") << "\n";
    return cli::kExitOk;
}

#if defined(SONDER_HAS_SERVER)
std::string make_coresidency_profile_name(const std::optional<std::string>& name_prefix, const std::string& model_path,
                                          std::size_t index, std::set<std::string>& used_names) {
    constexpr std::size_t kMaxNameLen = 128;
    std::string raw;
    if (name_prefix) {
        raw = *name_prefix + "-" + std::to_string(index + 1);
    } else {
        raw = std::filesystem::path(model_path).stem().string();
    }
    std::string base;
    base.reserve(std::min(raw.size(), kMaxNameLen));
    for (const char c : raw) {
        if (base.size() >= kMaxNameLen) break;
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-' || c == ':';
        base.push_back(ok ? c : '-');
    }
    if (base.empty() || base == "default") {
        base = "model-" + std::to_string(index + 1);
    }
    std::string candidate = base;
    for (std::size_t suffix = 2; used_names.count(candidate) != 0; ++suffix) {
        const std::string tag = "-" + std::to_string(suffix);
        const std::size_t keep = base.size() + tag.size() > kMaxNameLen ? (kMaxNameLen - tag.size()) : base.size();
        candidate = base.substr(0, keep) + tag;
    }
    used_names.insert(candidate);
    return candidate;
}

void print_plan_speculative(const si::PlacementPlan& plan) {
    if (plan.profile.speculative && !plan.profile.speculative->types.empty()) {
        std::cout << "  spec=";
        for (std::size_t i = 0; i < plan.profile.speculative->types.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << plan.profile.speculative->types[i];
        }
        if (plan.profile.speculative->draft_n_max) {
            std::cout << "(n=" << *plan.profile.speculative->draft_n_max << ")";
        }
    }
}
#endif

int cmd_plan(const Args& a) {
#if defined(SONDER_HAS_SERVER)
    const auto models = a.all("model");
    if (models.empty()) return usage_error("plan", "--model is required");
    for (const auto& m : models) {
        if (m.empty()) return usage_error("plan", "--model must not be empty");
    }

    const auto name = a.get("name");
    if (name && name->empty()) return usage_error("plan", "--name must not be empty");

    const auto mmproj = a.get("mmproj");
    if (models.size() > 1 && mmproj.has_value()) {
        return usage_error("plan",
                           "--mmproj applies only to single-model 'plan' (cannot be used with multiple --model flags)");
    }

    si::PlacementPlanRequest base_req;
    if (name) {
        base_req.profile_name = *name;
    }
    std::string parse_err;
    if (const auto ctx = a.get("ctx")) {
        if (!cli::parse_integer<std::uint32_t>("ctx", *ctx, 128u, 1048576u, base_req.target_ctx_size, parse_err)) {
            return usage_error("plan", parse_err);
        }
    }
    if (const auto min_ctx = a.get("min-ctx")) {
        if (!cli::parse_integer<std::uint32_t>("min-ctx", *min_ctx, 128u, 1048576u, base_req.min_ctx_size, parse_err)) {
            return usage_error("plan", parse_err);
        }
    }
    if (base_req.min_ctx_size > base_req.target_ctx_size) {
        base_req.min_ctx_size = base_req.target_ctx_size;
    }
    if (const auto parallel = a.get("parallel")) {
        if (!cli::parse_integer<std::uint32_t>("parallel", *parallel, 1u, 256u, base_req.parallel, parse_err)) {
            return usage_error("plan", parse_err);
        }
    }
    if (const auto headroom = a.get("headroom-mib")) {
        if (!cli::parse_integer<std::uint64_t>("headroom-mib", *headroom, 0ull, 65536ull, base_req.vram_headroom_mib,
                                               parse_err)) {
            return usage_error("plan", parse_err);
        }
    }
    if (const auto device = a.get("device")) base_req.device_hint = *device;
    if (a.has("total-vram")) base_req.prefer_free_vram = false;
    if (a.has("no-quantized-kv")) base_req.allow_quantized_kv = false;
    if (a.has("no-speculative")) base_req.allow_speculative = false;
    if (a.has("no-moe-offload")) base_req.allow_moe_cpu_offload = false;

    if (models.size() == 1) {
        si::PlacementPlanRequest req = base_req;
        req.model_path = models.front();
        if (mmproj) {
            if (mmproj->empty()) return usage_error("plan", "--mmproj must not be empty");
            std::error_code ec;
            const auto sz = std::filesystem::file_size(*mmproj, ec);
            if (ec) {
                err() << "error: plan: cannot stat --mmproj '" << *mmproj << "': " << ec.message() << "\n";
                return cli::kExitFailure;
            }
            req.mmproj_path = *mmproj;
            req.mmproj_bytes = static_cast<std::uint64_t>(sz);
        }

        const auto info = si::read_gguf_model_info(req.model_path);
        if (!info.ok()) {
            report_error("plan", info.status(), "");
            return cli::exit_code_for(info.status());
        }
        const auto gpus = si::enumerate_gpu_devices();
        const auto host = si::read_host_memory();
        const auto plan = si::plan_model_placement(info.value(), gpus, host, req);

        if (a.has("json")) {
            std::cout << si::json::Value(si::placement_plan_to_json(plan)).dump() << "\n";
            return cli::kExitOk;
        }

        std::cout << "profile=" << plan.profile.name << "  vendor=" << si::to_string(plan.selected_vendor)
                  << "  backend_hint=" << plan.recommended_backend_hint
                  << "  uma=" << (plan.uma_mode ? "true" : "false")
                  << "  prism_runtime=" << (plan.requires_prism_runtime ? "true" : "false") << "\n";
        std::cout << "fits_vram=" << (plan.fits_in_vram ? "true" : "false")
                  << "  fits_host_ram=" << (plan.fits_in_host_ram ? "true" : "false")
                  << "  budget_mib=" << plan.budget_mib
                  << "  est_vram_mib=" << plan.estimate.total_mib()
                  << "  est_host_ram_mib=" << plan.estimate.host_resident_mib() << "\n";
        std::cout << "ctx=" << plan.profile.ctx_size.value_or(0)
                  << "  kv=" << plan.profile.cache_type_k.value_or("f16") << "/"
                  << plan.profile.cache_type_v.value_or("f16")
                  << "  flash_attn=" << plan.profile.flash_attn.value_or("auto")
                  << "  n_gpu_layers="
                  << (plan.profile.n_gpu_layers ? std::to_string(*plan.profile.n_gpu_layers) : "auto");
        if (plan.profile.device) std::cout << "  device=" << *plan.profile.device;
        if (plan.profile.split_mode) std::cout << "  split_mode=" << *plan.profile.split_mode;
        print_plan_speculative(plan);
        std::cout << "\n";
        for (const auto& kv : plan.recommended_env) {
            std::cout << "env: " << kv.first << "=" << kv.second << "\n";
        }
        for (const auto& w : plan.warnings) {
            std::cout << "warning: " << w << "\n";
        }
        for (const auto& r : plan.reasons) {
            std::cout << "reason: " << r << "\n";
        }
        return cli::kExitOk;
    }

    // Multi-model co-residency planning (2+ --model flags).
    std::vector<si::CoResidencyEntry> entries;
    entries.reserve(models.size());
    std::set<std::string> used_names;
    for (std::size_t i = 0; i < models.size(); ++i) {
        const auto info = si::read_gguf_model_info(models[i]);
        if (!info.ok()) {
            report_error("plan", info.status(), "");
            return cli::exit_code_for(info.status());
        }
        si::CoResidencyEntry entry;
        entry.model = info.value();
        entry.request = base_req;
        entry.request.model_path = models[i];
        entry.request.profile_name = make_coresidency_profile_name(name, models[i], i, used_names);
        entries.push_back(std::move(entry));
    }

    const auto gpus = si::enumerate_gpu_devices();
    const auto host = si::read_host_memory();
    const auto co_plan = si::plan_coresident_models(entries, gpus, host);

    if (a.has("json")) {
        std::cout << si::json::Value(si::coresidency_plan_to_json(co_plan)).dump() << "\n";
        return cli::kExitOk;
    }

    std::cout << "models=" << co_plan.models.size()
              << "  all_fit_coresident=" << (co_plan.all_fit_coresident ? "true" : "false")
              << "  aggregate_vram_mib=" << co_plan.aggregate_vram_mib
              << "  aggregate_host_ram_mib=" << co_plan.aggregate_host_ram_mib
              << "  remaining_host_ram_mib=" << co_plan.remaining_host_ram_mib << "\n";
    if (!co_plan.remaining_gpu_vram_mib.empty()) {
        std::cout << "remaining_gpu_vram_mib:";
        for (const auto& [gpu_id, mib] : co_plan.remaining_gpu_vram_mib) {
            std::cout << " " << gpu_id << "=" << mib;
        }
        std::cout << "\n";
    }
    for (std::size_t i = 0; i < co_plan.models.size(); ++i) {
        const auto& plan = co_plan.models[i];
        std::cout << "[" << i << "] profile=" << plan.profile.name
                  << "  vendor=" << si::to_string(plan.selected_vendor)
                  << "  backend_hint=" << plan.recommended_backend_hint
                  << "  uma=" << (plan.uma_mode ? "true" : "false")
                  << "  fits_vram=" << (plan.fits_in_vram ? "true" : "false")
                  << "  fits_host_ram=" << (plan.fits_in_host_ram ? "true" : "false")
                  << "  budget_mib=" << plan.budget_mib
                  << "  est_vram_mib=" << plan.estimate.total_mib()
                  << "  est_host_ram_mib=" << plan.estimate.host_resident_mib()
                  << "  ctx=" << plan.profile.ctx_size.value_or(0)
                  << "  kv=" << plan.profile.cache_type_k.value_or("f16") << "/"
                  << plan.profile.cache_type_v.value_or("f16")
                  << "  n_gpu_layers="
                  << (plan.profile.n_gpu_layers ? std::to_string(*plan.profile.n_gpu_layers) : "auto");
        if (plan.profile.device) std::cout << "  device=" << *plan.profile.device;
        if (plan.profile.split_mode) std::cout << "  split_mode=" << *plan.profile.split_mode;
        print_plan_speculative(plan);
        std::cout << "\n";
        for (const auto& kv : plan.recommended_env) {
            std::cout << "  env[" << i << "]: " << kv.first << "=" << kv.second << "\n";
        }
        for (const auto& w : plan.warnings) {
            std::cout << "  warning[" << i << "]: " << w << "\n";
        }
    }
    for (const auto& n : co_plan.notes) {
        std::cout << "note: " << n << "\n";
    }
    for (const auto& w : co_plan.warnings) {
        std::cout << "warning: " << w << "\n";
    }
    return cli::kExitOk;
#else
    (void)a;
    err() << "error: plan: this build does not include the server module (src/server)\n";
    return cli::kExitUsage;
#endif
}

int cmd_backends(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("backends", a, env, false, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("backends", a, false, choice, engine, rc)) return rc;
    const bool json = a.has("json");
    si::json::Array list;
    std::size_t available = 0;
    bool ollama_unreachable = false;
    for (const auto& name : engine->backend_names()) {
        auto b = engine->find_backend(name);
        auto probe = b->probe();
        if (probe.ok()) ++available;
        if (!probe.ok() && name == "ollama" && probe.status().code() == si::ErrorCode::unavailable) {
            ollama_unreachable = true;
        }
        if (json) {
            si::json::Array caps;
            for (const auto& c : b->capabilities().names()) caps.emplace_back(c);
            list.emplace_back(si::json::Object{
                {"name", name},
                {"available", probe.ok()},
                {"version", probe.ok() ? si::json::Value(probe.value()) : si::json::Value(nullptr)},
                {"error", probe.ok() ? si::json::Value(nullptr) : si::json::Value(probe.status().to_string())},
                {"description", b->description()},
                {"synthetic", is_mock_backend(name)},
                {"capabilities", std::move(caps)}});
            continue;
        }
        std::cout << name << "  "
                  << (probe.ok() ? "reachable version=" + probe.value()
                                 : "unavailable (" + probe.status().to_string() + ")")
                  << "\n    " << b->description() << "\n    capabilities:";
        for (const auto& c : b->capabilities().names()) std::cout << " " << c;
        std::cout << "\n";
    }
    if (json) {
        std::cout << si::json::Value(si::json::Object{{"backends", std::move(list)}, {"available", available}}).dump()
                  << "\n";
    } else if (ollama_unreachable) {
        std::cout << "hint: Ollama is not reachable; start it with 'ollama serve', or pass --ollama-url URL\n";
    }
    if (available == 0) {
        err() << "error: backends: no backend is available\n";
        return cli::kExitFailure;
    }
    return cli::kExitOk;
}

int cmd_models(const Args& a, const cli::EnvDefaults& env) {
    int rc = cli::kExitOk;
    BackendChoice choice;
    if (!resolve_backend("models", a, env, true, false, choice, rc)) return rc;
    std::unique_ptr<si::Engine> engine;
    if (!build_engine("models", a, false, choice, engine, rc)) return rc;
    auto models = engine->find_backend(choice.backend)->list_models();
    if (!models.ok()) {
        report_error("models", models.status(), choice.backend);
        return cli::exit_code_for(models.status());
    }
    if (a.has("json")) {
        si::json::Array list;
        for (const auto& m : models.value()) {
            list.emplace_back(si::json::Object{{"name", m.name},
                                               {"format", m.format},
                                               {"family", m.family},
                                               {"parameter_size", m.parameter_size},
                                               {"quantization", m.quantization},
                                               {"size_bytes", m.size_bytes},
                                               {"context_length", m.context_length}});
        }
        std::cout << si::json::Value(si::json::Object{{"backend", choice.backend},
                                                      {"synthetic", is_mock_backend(choice.backend)},
                                                      {"models", std::move(list)}})
                         .dump()
                  << "\n";
        return cli::kExitOk;
    }
    for (const auto& m : models.value()) {
        std::cout << m.name << "  format=" << m.format << "  family=" << m.family << "  params=" << m.parameter_size
                  << "  quant=" << m.quantization << "  size_mib=" << (m.size_bytes >> 20) << "\n";
    }
    return cli::kExitOk;
}

int cmd_generate(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    if (!stats_mode_from_args(a, cli::StatsMode::text, stats_mode, error)) return usage_error("generate", error);
    const auto prompt = a.get("prompt");
    if (!prompt) return usage_error("generate", "--prompt is required");
    RequestSetup setup;
    if (!prepare_request("generate", a, env, quiet, setup, {}, rc)) return rc;
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("generate", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
    {
        si::Session& s = *session.value();
        InterruptWatcher watcher([&s] { s.cancel(); });
        res = s.generate(*prompt, [](const si::TokenChunk& c) {
            std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
            std::cout.flush();
            return true;
        });
    }
    std::cout << "\n";
    std::cout.flush();
    if (!res.ok()) {
        report_error("generate", res.status(), setup.choice.backend);
        return cli::exit_code_for(res.status());
    }
    print_stats(stats_mode, cli::make_request_stats("generate", setup.choice.backend, setup.model->descriptor().name,
                                                    is_mock_backend(setup.choice.backend), res.value()));
    return cli::exit_code_for(res.value().outcome);
}

int cmd_chat(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    const auto messages_path = a.get("messages");
    int rc = cli::kExitOk;
    cli::StatsMode stats_mode = cli::StatsMode::text;
    std::string error;
    // One-shot chat prints stats by default; the interactive REPL only on
    // request (/stats, or --stats text|json).
    if (!stats_mode_from_args(a, messages_path ? cli::StatsMode::text : cli::StatsMode::none, stats_mode, error)) {
        return usage_error("chat", error);
    }
    std::vector<si::ChatMessage> file_messages;
    const auto read_messages = [&](int& code) {
        if (!messages_path) return true;
        auto loaded = cli::load_chat_messages(*messages_path);
        if (!loaded.ok()) {
            code = usage_error("chat", loaded.status().message());
            return false;
        }
        file_messages = std::move(loaded.value());
        return true;
    };
    RequestSetup setup;
    if (!prepare_request("chat", a, env, quiet, setup, read_messages, rc)) return rc;
    const std::string system = a.get("system").value_or("");
    const bool native = setup.model->backend_model().has_native_chat();
    const std::string model_name = setup.model->descriptor().name;
    const bool synthetic = is_mock_backend(setup.choice.backend);

#if defined(SONDER_HAS_SESSION_CHAT)
    // Every turn runs through the engine session (scheduler, KV accounting,
    // request telemetry with request.queued.kind = "chat").
    auto session = setup.engine->create_session(setup.model, session_options(setup.sampling, setup.meta));
    if (!session.ok()) {
        report_error("chat", session.status(), setup.choice.backend);
        return cli::exit_code_for(session.status());
    }
    si::Session& s = *session.value();
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::Result<si::GenerationResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&s] { s.cancel(); });
            res = s.chat(history, [](const si::TokenChunk& c) {
                std::cout.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
                std::cout.flush();
                return true;
            });
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().outcome == si::RequestOutcome::cancelled;
        report.stats = cli::make_request_stats("chat", setup.choice.backend, model_name, synthetic, res.value());
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#else
    // Library without Session::chat: turns go straight to the backend model
    // and emit no request telemetry.
    std::uint64_t turn_no = 0;
    auto turn = [&](const std::vector<si::ChatMessage>& history) -> si::Result<cli::ChatTurnReport> {
        si::CancellationSource source;
        si::Result<cli::ChatTurnResult> res = si::Status(si::ErrorCode::internal, "not run");
        {
            InterruptWatcher watcher([&source] { source.cancel(); });
            res = cli::run_chat_turn(setup.model->backend_model(), history, setup.sampling, source.token(), std::cout,
                                     "chat-" + std::to_string(++turn_no));
        }
        if (!res.ok()) return res.status();
        cli::ChatTurnReport report;
        report.cancelled = res.value().stats.stop_reason == si::StopReason::cancelled;
        report.stats.command = "chat";
        report.stats.backend = setup.choice.backend;
        report.stats.model = model_name;
        report.stats.synthetic = synthetic;
        report.stats.outcome = report.cancelled ? "cancelled" : "completed";
        report.stats.stop_reason = si::to_string(res.value().stats.stop_reason);
        report.stats.prompt_tokens = res.value().stats.prompt_tokens;
        report.stats.completion_tokens = res.value().stats.completion_tokens;
        report.stats.native_chat = native;
        report.stats.messages = history.size();
        report.text = std::move(res.value().text);
        return report;
    };
#endif

    if (messages_path) {
        cli::apply_system_prompt(file_messages, system);
        auto res = turn(file_messages);
        std::cout << "\n";
        std::cout.flush();
        if (!res.ok()) {
            report_error("chat", res.status(), setup.choice.backend);
            return cli::exit_code_for(res.status());
        }
        print_stats(stats_mode, res.value().stats);
        return res.value().cancelled ? cli::kExitCancelled : cli::kExitOk;
    }

    const bool interactive = stdin_is_terminal();
    const bool terminal_out = stdout_is_terminal();
    if (!quiet) {
        err() << "[sonder-infer] chat with " << model_name << " on " << setup.choice.backend
              << (native ? " (native chat)" : " (generic prompt format)")
              << "; /help lists commands, /exit or EOF quits\n";
    }
    cli::ChatReplOptions ro;
    ro.system = system;
    ro.user_prompt = interactive ? "you> " : "";
    ro.assistant_label = interactive || terminal_out ? "assistant> " : "";
    ro.color = cli::color_enabled(terminal_out) && stdout_supports_ansi();
    ro.stats = stats_mode;
    return cli::run_chat_session(std::cin, std::cout, err(), ro, turn);
}

#if defined(SONDER_HAS_BENCH)
int cmd_bench(const Args& a, const cli::EnvDefaults& env) {
    const bool quiet = a.has("quiet");
    const bool list_models = a.has("list-models");
    int rc = cli::kExitOk;

    const auto check_backend_option_conflicts = [&](const std::string& backend_name) -> bool {
        const auto ls_url = a.get("llamaserver-url");
        const auto ls_cfg = a.get("llamaserver-config");
        const auto ls_mode = a.get("llamaserver-mode");
        const auto ls_exe = a.get("llamaserver-executable");
        const bool has_ls_args = !a.all("llamaserver-arg").empty();
        if (backend_name != "llamaserver") {
            if (ls_url.has_value()) {
                rc = usage_error("bench", "--llamaserver-url requires --backend llamaserver");
                return false;
            }
            if (ls_cfg.has_value()) {
                rc = usage_error("bench", "--llamaserver-config requires --backend llamaserver");
                return false;
            }
            if (ls_mode.has_value() || ls_exe.has_value() || has_ls_args) {
                rc = usage_error("bench", "--llamaserver-* options require --backend llamaserver");
                return false;
            }
        } else {
            if (ls_url.has_value() && ls_cfg.has_value()) {
                rc = usage_error("bench", "--llamaserver-url and --llamaserver-config cannot be used together");
                return false;
            }
            if (ls_url.has_value() && ls_url->empty()) {
                rc = usage_error("bench", "--llamaserver-url must not be empty");
                return false;
            }
            if (ls_cfg.has_value() && ls_cfg->empty()) {
                rc = usage_error("bench", "--llamaserver-config must not be empty");
                return false;
            }
        }
        if (backend_name != "ollama") {
            if (a.get("ollama-url").has_value()) {
                rc = usage_error("bench", "--ollama-url requires --backend ollama");
                return false;
            }
            if (a.has("ollama-allow-remote")) {
                rc = usage_error("bench", "--ollama-allow-remote requires --backend ollama");
                return false;
            }
        }
        return true;
    };

    const auto make_llamaserver = [&](const BackendChoice& choice) -> std::shared_ptr<si::Backend> {
        si::bench::LlamaServerBenchOptions lso;
        lso.url = choice.llamaserver_url;
        lso.config_path = choice.llamaserver_config;
        lso.mode = choice.llamaserver_mode;
        lso.executable = choice.llamaserver_executable;
        lso.args = choice.llamaserver_args;
        auto made = si::bench::make_llamaserver_bench_backend(lso);
        if (!made.ok()) {
            rc = usage_error("bench", made.status().to_string());
            return nullptr;
        }
        return std::move(made.value());
    };

    const std::string effective_backend = a.get("backend").value_or(env.backend.value_or(""));

    if (list_models) {
        if (!effective_backend.empty()) {
            if (!check_backend_name("bench", effective_backend, rc)) return rc;
            if (!check_backend_option_conflicts(effective_backend)) return rc;
        }
        BackendChoice choice;
        if (!resolve_backend("bench", a, env, true, false, choice, rc)) return rc;
        std::shared_ptr<si::Backend> backend;
        std::unique_ptr<si::Engine> engine;
        if (choice.backend == "llamaserver") {
            backend = make_llamaserver(choice);
            if (!backend) return rc;
        } else {
            if (!build_engine("bench", a, false, choice, engine, rc)) return rc;
            backend = engine->find_backend(choice.backend);
            if (!backend) {
                err() << "error: bench: backend '" << choice.backend << "' is unavailable\n";
                return cli::kExitFailure;
            }
        }
        auto models = backend->list_models();
        if (!models.ok()) {
            report_error("bench", models.status(), choice.backend);
            return cli::exit_code_for(models.status());
        }
        for (const auto& m : models.value()) {
            std::cout << m.name << "\t" << m.parameter_size << "\t" << m.quantization << "\t" << m.size_bytes << "\n";
        }
        return cli::kExitOk;
    }

    const auto corpus_path = a.get("corpus");
    const auto out_path = a.get("out");
    if (!corpus_path || !out_path) return usage_error("bench", "--corpus and --out are required");
    si::bench::Options opts;
    std::string error;
    if (auto v = a.get("warmup"); v && !cli::parse_integer<int>("warmup", *v, 0, 100, opts.warmup_runs, error)) {
        return usage_error("bench", error);
    }
    if (auto v = a.get("runs"); v && !cli::parse_integer<int>("runs", *v, 1, 1000, opts.measured_runs, error)) {
        return usage_error("bench", error);
    }
    if (auto v = a.get("concurrency");
        v && !cli::parse_integer<int>("concurrency", *v, si::bench::kMinConcurrency, si::bench::kMaxConcurrency,
                                      opts.concurrency, error)) {
        return usage_error("bench", error);
    }
    opts.capture_gpu_snapshot = !a.has("no-gpu-snapshot");
    opts.hardware = a.get("hardware").value_or("");
    opts.label = a.get("label").value_or("");
    if (!effective_backend.empty()) {
        if (!check_backend_name("bench", effective_backend, rc)) return rc;
        if (!check_backend_option_conflicts(effective_backend)) return rc;
    }
    RequestSetup setup;
    if (!resolve_backend("bench", a, env, true, true, setup.choice, rc)) return rc;
    if (!sampling_from_args(a, setup.sampling, error, false) || !session_meta_from_args(a, setup.meta, error)) {
        return usage_error("bench", error);
    }
    auto corpus = si::bench::load_corpus(*corpus_path);
    if (!corpus.ok()) return usage_error("bench", corpus.status().to_string());
    if (setup.choice.backend == "llamaserver") {
        si::EngineOptions eo;
        eo.telemetry.capture_text = a.has("capture-text");
        const std::string level_text = a.get("telemetry-level").value_or("standard");
        auto level = si::parse_telemetry_level(level_text);
        if (!level) {
            return usage_error("bench", "invalid value '" + level_text +
                                            "' for --telemetry-level (expected off, metrics, standard or deep)");
        }
        eo.telemetry.level = *level;
        if (auto path = a.get("telemetry")) {
            if (*path == "-") {
                eo.telemetry_sinks.push_back(std::make_shared<StderrTelemetrySink>());
            } else {
                si::Status st;
                auto sink = si::make_jsonl_file_sink(*path, false, &st);
                if (!sink) {
                    err() << "error: bench: cannot open --telemetry " << *path << ": " << st.to_string() << "\n";
                    return cli::kExitFailure;
                }
                eo.telemetry_sinks.push_back(std::shared_ptr<si::TelemetrySink>(std::move(sink)));
            }
        }
        auto backend = make_llamaserver(setup.choice);
        if (!backend) return rc;
        setup.engine = std::make_unique<si::Engine>(std::move(eo));
        (void)setup.engine->register_backend(std::move(backend));
    } else {
        if (!build_engine("bench", a, true, setup.choice, setup.engine, rc)) return rc;
        if (!quiet && is_mock_backend(setup.choice.backend)) {
            err() << "[sonder-infer] MOCK BACKEND - synthetic output, not a quality or performance signal\n";
        }
    }
    setup.model = load_model("bench", *setup.engine, setup.choice, rc);
    if (!setup.model) return rc;
    opts.sampling = setup.sampling;
    opts.run_id = setup.meta.run_id;
    opts.agent_id = setup.meta.agent_id;
    opts.task_id = setup.meta.task_id;
    opts.workload = setup.meta.workload;
    opts.priority = setup.meta.priority;
    auto doc = si::bench::run(*setup.engine, setup.model, corpus.value(), opts, quiet ? nullptr : &err());
    std::ofstream out(*out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err() << "error: bench: cannot write " << *out_path << "\n";
        return cli::kExitFailure;
    }
    out << si::json::Value(doc).dump() << "\n";
    out.close();
    if (a.has("markdown")) {
        const std::string markdown = si::bench::render_markdown(si::json::Value(doc));
        std::string md_path = *out_path;
        if (md_path.size() > 5 && md_path.compare(md_path.size() - 5, 5, ".json") == 0) {
            md_path.resize(md_path.size() - 5);
        }
        md_path += ".md";
        std::ofstream md(md_path, std::ios::binary | std::ios::trunc);
        if (!md) {
            err() << "error: bench: cannot write " << md_path << "\n";
            return cli::kExitFailure;
        }
        md << markdown;
        std::cout << markdown;
        if (!quiet) err() << "[bench] wrote " << md_path << "\n";
    }
    const auto* summary = doc.find("summary");
    if (!quiet) {
        err() << "[bench] wrote " << *out_path << " summary=" << (summary ? summary->dump() : "{}") << "\n";
    }
    const auto* failures = summary ? summary->find("failures") : nullptr;
    return failures && failures->as_int() > 0 ? cli::kExitFailure : cli::kExitOk;
}
#endif

bool wants_help(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

#if defined(SONDER_HAS_SERVER)
// The --ollama-allow-remote plain-HTTP refusal for serve, checked before
// serve_main() runs (src/server is outside this lane). Serve's parser takes
// "--ollama-url URL" and "--ollama-url=URL"; the environment fills in the
// URL as it does for serve (backend_env_defaults()).
bool serve_ollama_remote_ok(const std::vector<std::string>& args, int& rc) {
    bool allow_remote = false;
    std::optional<std::string> url;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--ollama-allow-remote") {
            allow_remote = true;
        } else if (arg == "--ollama-url" && i + 1 < args.size()) {
            url = args[++i];
        } else if (arg.rfind("--ollama-url=", 0) == 0) {
            url = arg.substr(std::string_view("--ollama-url=").size());
        }
    }
    if (!allow_remote) return true;
    if (!url) url = cli::read_env_defaults().ollama_url;
    return check_ollama_remote_url("serve", url.value_or(""), rc);
}
#endif

int cmd_serve(const std::vector<std::string>& args) {
#if defined(SONDER_HAS_SERVER)
    // --help wins over any other argument, as for the other commands.
    if (wants_help(args)) return si::server::serve_main({"--help"}, std::cout, std::cerr);
    int rc = cli::kExitOk;
    if (!serve_ollama_remote_ok(args, rc)) return rc;
    return si::server::serve_main(args, std::cout, std::cerr);
#else
    (void)args;
    err() << "error: serve: this build does not include the server module (src/server); "
             "build from a tree that contains it to use 'sonder-infer serve'\n";
    return cli::kExitUsage;
#endif
}

int cmd_help(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << cli::overview();
        return cli::kExitOk;
    }
    if (args.size() == 1 && (args.front() == "--help" || args.front() == "-h")) {
        std::cout << cli::render_help(cli::kProgram, cli::help_command());
        return cli::kExitOk;
    }
    if (args.size() > 1) return usage_error("help", "expected at most one command name");
    const std::string& name = args.front();
    if (name == "serve") return cmd_serve({"--help"});
    if (const auto* spec = cli::find_command(name)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    std::string message = "unknown command '" + name + "'";
    if (auto s = cli::suggest(name, cli::command_names())) message += " (did you mean '" + *s + "'?)";
    err() << "error: help: " << message << "\n\n" << cli::short_usage();
    return cli::kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    if (args.empty()) {
        err() << cli::short_usage();
        return cli::kExitUsage;
    }
    const std::string cmd = args.front();
    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (cmd == "help" || cmd == "--help" || cmd == "-h") return cmd_help(rest);
    if (cmd == "--version") return cmd_version(Args{});
    const cli::CommandSpec* spec = cli::find_command(cmd);
    if (spec == nullptr) {
        std::string message = "unknown command '" + cmd + "'";
        if (auto s = cli::suggest(cmd, cli::command_names())) message += " (did you mean '" + *s + "'?)";
        err() << "error: " << message << "\n\n" << cli::short_usage();
        return cli::kExitUsage;
    }
    if (cmd == "serve") return cmd_serve(rest);
    if (cmd == "weights") {
#if defined(SONDER_HAS_WEIGHT_MANIFEST)
        return si::weights::weights_main(rest, std::cout, std::cerr);
#else
        err() << "error: weights: this build does not include the artifacts module (src/artifacts)\n";
        return 4;  // `weights` usage code (cli::kExitUsage is 2, which `weights` uses for "mismatch")
#endif
    }
    if (cmd == "tune") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        return si::llamaserver::tune::tune_main(rest, std::cout, std::cerr);
#else
        err() << "error: tune: this build does not include the llamaserver module\n";
        return cli::kExitUsage;
#endif
    }

    std::string error;
    auto parsed = cli::parse_command_args(*spec, rest, error);
    if (parsed ? parsed->help : wants_help(rest)) {
        std::cout << cli::render_help(cli::kProgram, *spec);
        return cli::kExitOk;
    }
    if (!parsed) return usage_error(cmd, error);
    const Args& a = *parsed;
    const cli::EnvDefaults env = cli::read_env_defaults();

    if (cmd == "version") return cmd_version(a);
    if (cmd == "devices") return cmd_devices(a);
    if (cmd == "node") return cmd_node(a);
    if (cmd == "plan") return cmd_plan(a);
    if (cmd == "backends") return cmd_backends(a, env);
    if (cmd == "models") return cmd_models(a, env);
    if (cmd == "generate") return cmd_generate(a, env);
    if (cmd == "chat") return cmd_chat(a, env);
#if defined(SONDER_HAS_BENCH)
    if (cmd == "bench") return cmd_bench(a, env);
#else
    if (cmd == "bench") {
        err() << "error: bench: this build does not include the bench module (bench/)\n";
        return cli::kExitUsage;
    }
#endif
    err() << "error: unknown command '" << cmd << "'\n\n" << cli::short_usage();
    return cli::kExitUsage;
}
