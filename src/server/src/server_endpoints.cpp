#include "server_endpoints.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "health_features.hpp"
#include "identity.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/placement_planner.hpp"
#include "sonder_inference.h"

namespace sonder::inference::server::detail {

namespace {

// Short-lived serialized cache so concurrent `/v1/sonder/health` bursts do not
// race platform GPU loader calls (DXGI/NVML) or stall health polling while
// still reflecting free-VRAM changes within a couple of seconds.
constexpr auto kGpuInventoryCacheTtl = std::chrono::seconds(2);

std::vector<GpuDeviceInfo> cached_gpu_inventory() {
    struct Cache {
        std::mutex mu;
        bool populated = false;
        std::chrono::steady_clock::time_point refreshed_at{};
        std::vector<GpuDeviceInfo> gpus;
    };
    static Cache cache;
    std::lock_guard<std::mutex> lock(cache.mu);
    const auto now = std::chrono::steady_clock::now();
    if (!cache.populated || (now - cache.refreshed_at) >= kGpuInventoryCacheTtl) {
        cache.gpus = enumerate_gpu_devices();
        cache.refreshed_at = std::chrono::steady_clock::now();
        cache.populated = true;
    }
    return cache.gpus;
}

json::Value memory_reading_bytes_json(const MemoryReading& reading) {
    if (!reading.is_known()) {
        return json::Value(nullptr);
    }
    return json::Value(reading.bytes_or(0));
}

bool has_control_or_nul(std::string_view sv) noexcept {
    for (const char c : sv) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20u || u == 0x7Fu) {
            return true;
        }
    }
    return false;
}

std::string sanitize_key_for_error(std::string_view key) {
    constexpr std::size_t kMaxLen = 64;
    std::string out;
    out.reserve(std::min(key.size(), kMaxLen) + 3);
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (out.size() >= kMaxLen) {
            out += "...";
            break;
        }
        const auto u = static_cast<unsigned char>(key[i]);
        out.push_back((u < 0x20u || u >= 0x7Fu) ? '?' : static_cast<char>(u));
    }
    return out;
}

bool is_safe_profile_name(std::string_view name) noexcept {
    if (name.empty() || name.size() > 128) {
        return false;
    }
    if (name.find("..") != std::string_view::npos) {
        return false;
    }
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-' || c == ':';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool is_safe_model_basename(std::string_view name) noexcept {
    if (name.empty() || name.size() > 255) {
        return false;
    }
    if (name == "." || name == "..") {
        return false;
    }
    if (name.find("..") != std::string_view::npos) {
        return false;
    }
    for (const char c : name) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20u || u == 0x7Fu) {
            return false;
        }
        if (c == '/' || c == '\\' || c == ':') {
            return false;
        }
    }
    return true;
}

bool ends_with_gguf_ci(std::string_view name) noexcept {
    if (name.size() <= 5) {
        return false;
    }
    const std::string_view tail = name.substr(name.size() - 5);
    const auto lower = [](char c) noexcept {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
    };
    return tail[0] == '.' && lower(tail[1]) == 'g' && lower(tail[2]) == 'g' && lower(tail[3]) == 'u' &&
           lower(tail[4]) == 'f';
}

std::string path_basename(std::string_view path) {
    const std::size_t slash = path.find_last_of("/\\");
    std::string_view name = slash == std::string_view::npos ? path : path.substr(slash + 1);
    const std::size_t colon = name.rfind(':');
    if (colon != std::string_view::npos) {
        name = name.substr(colon + 1);
    }
    return std::string(name);
}

bool parse_strict_u64_range(const json::Value& value, std::uint64_t lo, std::uint64_t hi, std::uint64_t& out) noexcept {
    if (!value.is_integer()) {
        return false;
    }
    constexpr std::uint64_t kSentinel = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t n = value.as_uint(kSentinel);
    if (n == kSentinel || n < lo || n > hi) {
        return false;
    }
    out = n;
    return true;
}

bool is_canonical_child_of(const std::filesystem::path& file, const std::filesystem::path& dir) {
    auto dir_it = dir.begin();
    auto file_it = file.begin();
    for (; dir_it != dir.end(); ++dir_it) {
        if (dir_it->empty()) {
            continue;
        }
        while (file_it != file.end() && file_it->empty()) {
            ++file_it;
        }
        if (file_it == file.end() || *file_it != *dir_it) {
            return false;
        }
        ++file_it;
    }
    while (file_it != file.end() && file_it->empty()) {
        ++file_it;
    }
    return file_it != file.end();
}

bool resolve_model_in_dirs(const std::vector<std::string>& model_dirs, const std::string& basename,
                           std::string& resolved_out) {
    if (!is_safe_model_basename(basename) || !ends_with_gguf_ci(basename)) {
        return false;
    }
    for (const auto& dir_str : model_dirs) {
        if (dir_str.empty()) {
            continue;
        }
        std::error_code ec;
        const std::filesystem::path canonical_dir = std::filesystem::canonical(std::filesystem::path(dir_str), ec);
        if (ec || !std::filesystem::is_directory(canonical_dir, ec) || ec) {
            continue;
        }
        const std::filesystem::path candidate = canonical_dir / std::filesystem::path(basename);
        const std::filesystem::file_status sym_status = std::filesystem::symlink_status(candidate, ec);
        if (ec || !std::filesystem::exists(sym_status)) {
            continue;
        }
        const std::filesystem::path canonical_file = std::filesystem::canonical(candidate, ec);
        if (ec) {
            continue;
        }
        if (!is_canonical_child_of(canonical_file, canonical_dir)) {
            // Symlink escapes its configured model_dir: refuse immediately.
            return false;
        }
        if (!std::filesystem::is_regular_file(canonical_file, ec) || ec) {
            continue;
        }
        if (!ends_with_gguf_ci(canonical_file.filename().string())) {
            return false;
        }
        resolved_out = canonical_file.string();
        return true;
    }
    return false;
}

void extract_llamaserver_profile_paths(const std::vector<std::string>& args, std::string_view profile_name,
                                       bool single_binding, std::string& model_out, std::string& mmproj_out) {
    std::string arg_model;
    std::string arg_alias;
    std::string arg_mmproj;
    const auto take_value = [&args](std::size_t& idx, std::string& dst) {
        const std::string& raw = args[idx];
        const std::size_t eq = raw.find('=');
        if (eq != std::string::npos) {
            dst = raw.substr(eq + 1);
            return;
        }
        if (idx + 1 < args.size()) {
            dst = args[++idx];
        }
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string flag = llamaserver_flag_name(args[i]);
        if (flag == "--model" || flag == "-m") {
            take_value(i, arg_model);
        } else if (flag == "--alias" || flag == "-a") {
            take_value(i, arg_alias);
        } else if (flag == "--mmproj" || flag == "-mm") {
            take_value(i, arg_mmproj);
        }
    }
    if (!arg_model.empty() && (arg_alias == profile_name || (arg_alias.empty() && single_binding))) {
        model_out = std::move(arg_model);
        mmproj_out = std::move(arg_mmproj);
    }
}

bool resolve_profile_paths(const ServerOptions& opts, const std::string& profile_name, std::string& model_out,
                           std::string& mmproj_out) {
    const ModelProfileBinding* binding = find_profile_binding(opts, profile_name);
    if (binding == nullptr) {
        return false;
    }
    // 1. Active llamaserver profiles carry their operator-configured --model / --mmproj in llamaserver_args.
    extract_llamaserver_profile_paths(opts.backend.llamaserver_args, profile_name, opts.profile_bindings.size() == 1,
                                      model_out, mmproj_out);
    if (!model_out.empty()) {
        std::error_code ec;
        if (ends_with_gguf_ci(model_out) && std::filesystem::is_regular_file(std::filesystem::path(model_out), ec) &&
            !ec) {
            if (!mmproj_out.empty()) {
                std::error_code mmproj_ec;
                if (!std::filesystem::is_regular_file(std::filesystem::path(mmproj_out), mmproj_ec) || mmproj_ec) {
                    mmproj_out.clear();
                }
            }
            return true;
        }
        model_out.clear();
        mmproj_out.clear();
    }
    // 2. llamacpp / direct bindings store the GGUF path (or a model_dir basename) in load_name.
    const std::string& target = binding->load_name.empty() ? binding->model : binding->load_name;
    if (is_safe_model_basename(target)) {
        if (resolve_model_in_dirs(opts.backend.model_dirs, target, model_out)) {
            return true;
        }
    }
    if (!binding->load_name.empty() && binding->load_name != binding->model) {
        std::error_code ec;
        if (ends_with_gguf_ci(binding->load_name) &&
            std::filesystem::is_regular_file(std::filesystem::path(binding->load_name), ec) && !ec) {
            model_out = binding->load_name;
            return true;
        }
    }
    return false;
}

// Mirrors LlamaCppCoreBackend::Resolve() using UTF-8 std::filesystem paths:
// direct regular file first, then the first matching regular file under
// `model_dirs`. Never falls back to an unrelated CWD basename.
std::string resolve_llamacpp_gguf_path(const std::string& load_name, const std::vector<std::string>& model_dirs) {
    if (load_name.empty()) {
        return {};
    }
    const std::filesystem::path direct = path_from_utf8(load_name);
    std::error_code direct_ec;
    const std::filesystem::file_status direct_st = std::filesystem::status(direct, direct_ec);
    if (!direct_ec && std::filesystem::is_regular_file(direct_st)) {
        return load_name;
    }
    if (direct.is_absolute()) {
        return {};
    }
    for (const std::string& dir : model_dirs) {
        if (dir.empty()) {
            continue;
        }
        const std::filesystem::path candidate = path_from_utf8(dir) / direct;
        std::error_code dir_ec;
        const std::filesystem::file_status st = std::filesystem::status(candidate, dir_ec);
        if (!dir_ec && std::filesystem::is_regular_file(st)) {
            return path_to_utf8(candidate);
        }
    }
    return {};
}

}  // namespace

json::Object sonder_meta() { return json::Object{{"api_version", kApiVersion}}; }

const ModelProfileBinding* find_profile_binding(const ServerOptions& opts, const std::string& id) {
    for (const auto& b : opts.profile_bindings) {
        if (b.model == id) {
            return &b;
        }
    }
    return nullptr;
}

json::Object profile_with_runtime_context(json::Object profile, const std::optional<BackendRuntimeStatus>& runtime) {
    if (!runtime || !runtime->context.fitted_ctx) {
        return profile;
    }
    const json::Value* configured = profile.find("context_length");
    const std::uint64_t fitted = *runtime->context.fitted_ctx;
    if (configured != nullptr && configured->is_integer() &&
        static_cast<std::uint64_t>(configured->as_int()) == fitted) {
        return profile;
    }
    const json::Value configured_value = configured != nullptr ? *configured : json::Value();
    profile.set("context_length", json::Value(fitted));
    profile.set("configured_context_length", configured_value);
    return profile;
}

ApiError load_failure_error(const std::string& id, const Status& st) {
    if (st.code() == ErrorCode::not_found) {
        return make_error(404, "model_not_found", "cannot load model '" + id + "': " + st.message(), "model");
    }
    return make_error(503, "not_ready",
                      "cannot load model '" + id + "': " + st.message() +
                          "; nothing was executed, the request is safe to send elsewhere");
}

json::Object resource_footprint_json(const ResourceFootprint& f) {
    return json::Object{{"weights_bytes", f.weights_bytes},
                        {"kv_cache_bytes", f.kv_cache_bytes},
                        {"workspace_bytes", f.workspace_bytes},
                        {"projector_bytes", f.projector_bytes},
                        {"adapter_bytes", f.adapter_bytes},
                        {"total_bytes", f.total_bytes()}};
}

json::Object model_footprint_json(const ModelResourceFootprint& fp) {
    return json::Object{{"vram", resource_footprint_json(fp.vram)},
                        {"ram", resource_footprint_json(fp.ram)}};
}

json::Array build_health_models(const ModelResidency& residency) {
    json::Array models;
    for (const auto& m : residency.snapshot()) {
        json::Object entry{{"id", m.id}, {"backend", m.backend}, {"default", m.is_default}};
        // Additive residency fields (docs/SERVER.md "Model residency").
        entry.set("state", residency_state_name(m.state));
        entry.set("resident", m.state == ResidencyState::resident);
        entry.set("in_flight", m.pins);
        entry.set("loads", m.loads);
        entry.set("load_failures", m.load_failures);
        entry.set("evictions", m.evictions);
        entry.set("admission_rejections", m.admission_rejections);
        entry.set("idle_s", m.idle_s ? json::Value(*m.idle_s) : json::Value(nullptr));
        entry.set("reload_ms", m.reload_ms);
        entry.set("demand_score", m.demand_score);
        entry.set("demand_rate_per_min", m.demand_rate_per_min);
        entry.set("footprint",
                  model_footprint_json(m.state == ResidencyState::loading ? m.reserved_footprint
                                                                          : m.observed_footprint));
        entry.set("model_instance_id",
                  m.model_instance_id.empty() ? json::Value(nullptr) : json::Value(m.model_instance_id));
        models.emplace_back(std::move(entry));
    }
    return models;
}

json::Object build_residency_summary(const ServerOptions& opts, const ModelResidency& residency) {
    const ResidencyTotals t = residency.totals();
    const ResidencyConfig& c = residency.config();
    return json::Object{
        {"mode", opts.lazy_models ? "lazy" : "eager"},
        {"idle_ttl_s", c.idle_ttl.count() > 0 ? json::Value(std::chrono::duration<double>(c.idle_ttl).count())
                                              : json::Value(nullptr)},
        {"max_resident", c.max_resident > 0 ? json::Value(c.max_resident) : json::Value(nullptr)},
        {"max_resident_vram_bytes",
         c.max_resident_vram_bytes > 0 ? json::Value(c.max_resident_vram_bytes) : json::Value(nullptr)},
        {"max_resident_ram_bytes",
         c.max_resident_ram_bytes > 0 ? json::Value(c.max_resident_ram_bytes) : json::Value(nullptr)},
        {"registered", t.registered},
        {"resident", t.resident},
        {"loading", t.loading},
        {"unloading", t.unloading},
        {"loads", t.loads},
        {"load_failures", t.load_failures},
        {"evictions", t.evictions},
        {"admission_rejections", t.admission_rejections},
        {"resident_vram_bytes", t.resident_vram_bytes},
        {"resident_ram_bytes", t.resident_ram_bytes},
        {"reserved_vram_bytes", t.reserved_vram_bytes},
        {"reserved_ram_bytes", t.reserved_ram_bytes}};
}

json::Object build_health_body(const HealthSnapshot& s) {
    json::Array backends;
    if (s.backend) {
        json::Array caps;
        for (const auto& c : s.backend->capabilities().names()) {
            caps.emplace_back(c);
        }
        json::Object entry{
            {"name", s.backend->name()}, {"available", s.backend_available}, {"capabilities", std::move(caps)}};
        if (!s.backend_version.empty()) {
            entry.set("version", s.backend_version);
        }
        // Additive: only backends that report runtime observations
        // (e.g. a spawned llama-server) get this field.
        if (const auto runtime = s.backend->runtime_status()) {
            entry.set("runtime", to_json(*runtime));
        }
        backends.emplace_back(std::move(entry));
    }
    return json::Object{
        {"status", state_name(s.state)},
        {"api_version", kApiVersion},
        {"version", version_string()},
        {"commit", build_commit()},
        {"abi_version", SONDER_ABI_VERSION},
        {"instance_id", s.instance_id},
        {"node_id", s.node_id},
        {"uptime_s", seconds_since(s.started_at)},
        {"synthetic", s.synthetic},
        {"auth_required", !s.opts.token.empty()},
        {"backends", std::move(backends)},
        {"models", build_health_models(s.residency)},
        {"queued_by_class", json::Object{{"interactive", s.queued_by_class[0]},
                                         {"subagent", s.queued_by_class[1]},
                                         {"background", s.queued_by_class[2]}}},
        {"residency", build_residency_summary(s.opts, s.residency)},
        {"telemetry", json::Object{{"level", to_string(s.opts.telemetry_level)},
                                   {"subscribers", s.hub_stats.subscribers},
                                   {"retained", s.hub_stats.retained},
                                   {"capacity", s.hub_stats.capacity},
                                   {"emitted", s.telemetry_emitted},
                                   {"dropped", s.telemetry_dropped},
                                   {"subscriber_dropped_events", s.hub_stats.subscriber_dropped_events}}},
        {"log", json::Object{{"queued", s.log_stats.queued},
                             {"capacity", s.log_stats.capacity},
                             {"written", s.log_stats.written},
                             {"dropped", s.log_stats.dropped}}},
        {"gpus", gpu_inventory_json(cached_gpu_inventory())},
        {"grammar_cache", json::Object{{"entries", static_cast<std::uint64_t>(s.grammar_cache.entries)},
                                       {"used_bytes", static_cast<std::uint64_t>(s.grammar_cache.used_bytes)},
                                       {"max_entries", static_cast<std::uint64_t>(s.grammar_cache.max_entries)},
                                       {"max_bytes", static_cast<std::uint64_t>(s.grammar_cache.max_bytes)},
                                       {"hits", s.grammar_cache.hits},
                                       {"misses", s.grammar_cache.misses},
                                       {"inserts", s.grammar_cache.inserts},
                                       {"evictions", s.grammar_cache.evictions},
                                       {"bypassed_oversize", s.grammar_cache.bypassed_oversize},
                                       {"compile_failures", s.grammar_cache.compile_failures}}},
        {"capabilities", server_capabilities(s.backend, s.opts)},
        {"sonder", health_features(s.backend, s.opts, sonder_meta())}};
}

json::Object build_models_body(const ServerOptions& opts, const ModelResidency& residency,
                               const std::shared_ptr<Backend>& backend, bool synthetic) {
    const std::optional<BackendRuntimeStatus> runtime = backend ? backend->runtime_status() : std::nullopt;
    json::Array data;
    for (const auto& m : residency.snapshot()) {
        json::Object ext{{"backend", m.backend}, {"default", m.is_default}, {"synthetic", synthetic}};
        // Additive: context fit, GPU memory and warnings of the backend
        // process serving this model, when the backend reports them. This
        // describes the backend process, not the model handle, so it is
        // reported whatever the model's residency state (a lazy or
        // evicted model is still served by that process).
        if (runtime && backend->name() == m.backend) {
            ext.set("runtime", to_json(*runtime));
        }
        // Additive: the launch profile behind this model (whatever its
        // residency state, like `runtime` above).
        if (const ModelProfileBinding* binding = find_profile_binding(opts, m.id)) {
            const bool own_runtime = backend && backend->name() == m.backend;
            ext.set("profile",
                    profile_with_runtime_context(binding->metadata,
                                                 own_runtime ? runtime : std::optional<BackendRuntimeStatus>{}));
        }
        data.emplace_back(json::Object{
            {"id", m.id}, {"object", "model"}, {"owned_by", "sonder-inference"}, {"sonder", std::move(ext)}});
    }
    json::Object meta = health_features(backend, opts, sonder_meta());
    if (!opts.profile_catalog.empty()) {
        json::Array catalog;
        for (const auto& entry : opts.profile_catalog) {
            const json::Value* is_served = entry.find("served");
            if (entry.is_object() && is_served != nullptr && is_served->is_bool() && is_served->as_bool()) {
                catalog.emplace_back(profile_with_runtime_context(entry.as_object(), runtime));
            } else {
                catalog.push_back(entry);
            }
        }
        meta.set("profiles", std::move(catalog));
    }
    return json::Object{{"object", "list"}, {"data", std::move(data)}, {"sonder", std::move(meta)}};
}

json::Object build_identity_body(const ModelResidency& residency, const std::string& served_id, bool synthetic,
                                 const std::optional<std::string>& probed) {
    // Identity never loads a model (no backend I/O here): a model that is
    // not resident reports the descriptor of its last load, if any.
    IdentityResult id;
    if (const std::shared_ptr<Model> model = residency.resident_model(served_id)) {
        id = backend_identity(*model, probed);
    } else {
        std::string backend_name;
        for (const auto& m : residency.snapshot()) {
            if (m.id == served_id) {
                backend_name = m.backend;
            }
        }
        const std::optional<ModelDescriptor> last = residency.last_descriptor(served_id);
        id = backend_identity(backend_name, served_id, last ? &*last : nullptr, probed);
    }
    return json::Object{{"schema", "sonder.inference.identity/1"},
                        {"model", served_id},
                        {"synthetic", synthetic},
                        {"backend_identity", id.backend_identity},
                        {"reason", id.reason},
                        {"capabilities", server_capabilities(nullptr, ServerOptions{})},
                        {"sonder", sonder_meta()}};
}

json::Object build_node_body(const NodeSnapshot& s) {
    const HostMemory host = read_host_memory();
    json::Object host_memory{
        {"total_physical_bytes", memory_reading_bytes_json(host.total_physical)},
        {"available_physical_bytes", memory_reading_bytes_json(host.available_physical)},
        {"available_commit_bytes", memory_reading_bytes_json(host.available_commit)},
        {"source", host.source.empty() ? json::Value(nullptr) : json::Value(host.source)}};

    json::Array devices;
    for (const auto& d : enumerate_devices()) {
        devices.emplace_back(json::Object{{"id", d.id},
                                          {"kind", to_string(d.kind)},
                                          {"name", d.name},
                                          {"logical_cores", d.logical_cores},
                                          {"total_memory_bytes", d.total_memory_bytes},
                                          {"available_memory_bytes", d.available_memory_bytes}});
    }

    json::Array backends;
    if (s.backend) {
        json::Array caps;
        for (const auto& c : s.backend->capabilities().names()) {
            caps.emplace_back(c);
        }
        json::Object entry{
            {"name", s.backend->name()}, {"available", s.backend_available}, {"capabilities", std::move(caps)}};
        if (!s.backend_version.empty()) {
            entry.set("version", s.backend_version);
        }
        backends.emplace_back(std::move(entry));
    }

    json::Array models;
    for (const auto& m : s.residency.snapshot()) {
        models.emplace_back(json::Object{{"id", m.id},
                                         {"backend", m.backend},
                                         {"state", residency_state_name(m.state)},
                                         {"resident", m.state == ResidencyState::resident},
                                         {"default", m.is_default}});
    }

    json::Array profiles;
    if (!s.opts.profile_catalog.empty()) {
        const std::optional<BackendRuntimeStatus> runtime = s.backend ? s.backend->runtime_status() : std::nullopt;
        for (const auto& entry : s.opts.profile_catalog) {
            const json::Value* is_served = entry.find("served");
            if (entry.is_object() && is_served != nullptr && is_served->is_bool() && is_served->as_bool()) {
                profiles.emplace_back(profile_with_runtime_context(entry.as_object(), runtime));
            } else {
                profiles.push_back(entry);
            }
        }
    }

    const std::vector<GpuDeviceInfo> gpus = cached_gpu_inventory();
    const GpuPoolSummary gpu_pool = summarize_gpu_pools(gpus, host);

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    constexpr bool kHasAdaptiveSpeculation = true;
#else
    constexpr bool kHasAdaptiveSpeculation = false;
#endif
    json::Object capabilities{{"ternary_gguf", true},
                              {"placement_planner", true},
                              {"coresidency_planner", true},
                              {"multi_gpu", true},
                              {"hybrid_gpu_spread", true},
                              {"uma_telemetry", true},
                              {"adaptive_speculation", kHasAdaptiveSpeculation}};

    return json::Object{{"schema", "sonder.inference.node/1"},
                        {"node_id", s.node_id},
                        {"instance_id", s.instance_id},
                        {"platform", host_platform()},
                        {"version", version_string()},
                        {"commit", build_commit()},
                        {"abi_version", SONDER_ABI_VERSION},
                        {"api_version", kApiVersion},
                        {"synthetic", s.synthetic},
                        {"host_memory", std::move(host_memory)},
                        {"devices", std::move(devices)},
                        {"gpus", gpu_inventory_json(gpus)},
                        {"combined_gpu_vram_mib", gpu_pool.combined_gpu_vram_mib},
                        {"gpu_pool", gpu_pool_summary_to_json(gpu_pool)},
                        {"backends", std::move(backends)},
                        {"models", std::move(models)},
                        {"profiles", std::move(profiles)},
                        {"capabilities", std::move(capabilities)},
                        {"sonder", sonder_meta()}};
}

std::variant<PlanHttpRequest, ApiError> parse_plan_request(std::string_view body) {
    const Result<json::Value> parsed = json::parse(body, json::ParseLimits{32, 4096});
    if (!parsed.ok()) {
        return make_error(400, "invalid_request", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return make_error(400, "invalid_request", "request body must be a JSON object");
    }

    PlanHttpRequest out;
    const json::Object& obj = parsed.value().as_object();
    for (const auto& [key, val] : obj) {
        if (key == "profile") {
            if (!val.is_string()) {
                return make_error(400, "invalid_request", "profile must be a string", "profile");
            }
            if (!is_safe_profile_name(val.as_string())) {
                return make_error(400, "invalid_request",
                                  "profile must be a non-empty identifier using [A-Za-z0-9._:-] (max 128 bytes, no '..')",
                                  "profile");
            }
            out.has_profile = true;
            out.profile = val.as_string();
        } else if (key == "model") {
            if (!val.is_string()) {
                return make_error(400, "invalid_request", "model must be a string", "model");
            }
            if (!is_safe_model_basename(val.as_string())) {
                return make_error(
                    400, "invalid_request",
                    "model must be a single clean basename without path separators, '..', ':', or control characters",
                    "model");
            }
            out.has_model = true;
            out.model = val.as_string();
        } else if (key == "ctx_size") {
            std::uint64_t n = 0;
            if (!parse_strict_u64_range(val, 128ull, 1048576ull, n)) {
                return make_error(400, "invalid_request", "ctx_size must be an integer between 128 and 1048576",
                                  "ctx_size");
            }
            out.plan_request.target_ctx_size = static_cast<std::uint32_t>(n);
        } else if (key == "min_ctx_size") {
            std::uint64_t n = 0;
            if (!parse_strict_u64_range(val, 128ull, 1048576ull, n)) {
                return make_error(400, "invalid_request", "min_ctx_size must be an integer between 128 and 1048576",
                                  "min_ctx_size");
            }
            out.plan_request.min_ctx_size = static_cast<std::uint32_t>(n);
        } else if (key == "parallel") {
            std::uint64_t n = 0;
            if (!parse_strict_u64_range(val, 1ull, 256ull, n)) {
                return make_error(400, "invalid_request", "parallel must be an integer between 1 and 256", "parallel");
            }
            out.plan_request.parallel = static_cast<std::uint32_t>(n);
        } else if (key == "vram_headroom_mib") {
            std::uint64_t n = 0;
            if (!parse_strict_u64_range(val, 0ull, 65536ull, n)) {
                return make_error(400, "invalid_request", "vram_headroom_mib must be an integer between 0 and 65536",
                                  "vram_headroom_mib");
            }
            out.plan_request.vram_headroom_mib = n;
        } else if (key == "device_hint") {
            if (!val.is_string()) {
                return make_error(400, "invalid_request", "device_hint must be a string", "device_hint");
            }
            const std::string& hint = val.as_string();
            if (hint.size() > 128 || has_control_or_nul(hint)) {
                return make_error(400, "invalid_request",
                                  "device_hint must be at most 128 bytes with no control characters", "device_hint");
            }
            out.plan_request.device_hint = hint;
        } else if (key == "prefer_free_vram") {
            if (!val.is_bool()) {
                return make_error(400, "invalid_request", "prefer_free_vram must be a boolean", "prefer_free_vram");
            }
            out.plan_request.prefer_free_vram = val.as_bool();
        } else if (key == "allow_quantized_kv") {
            if (!val.is_bool()) {
                return make_error(400, "invalid_request", "allow_quantized_kv must be a boolean", "allow_quantized_kv");
            }
            out.plan_request.allow_quantized_kv = val.as_bool();
        } else if (key == "allow_speculative") {
            if (!val.is_bool()) {
                return make_error(400, "invalid_request", "allow_speculative must be a boolean", "allow_speculative");
            }
            out.plan_request.allow_speculative = val.as_bool();
        } else if (key == "allow_moe_cpu_offload") {
            if (!val.is_bool()) {
                return make_error(400, "invalid_request", "allow_moe_cpu_offload must be a boolean",
                                  "allow_moe_cpu_offload");
            }
            out.plan_request.allow_moe_cpu_offload = val.as_bool();
        } else if (key == "allow_hybrid_gpu_spread") {
            if (!val.is_bool()) {
                return make_error(400, "invalid_request", "allow_hybrid_gpu_spread must be a boolean",
                                  "allow_hybrid_gpu_spread");
            }
            out.plan_request.allow_hybrid_gpu_spread = val.as_bool();
        } else {
            const std::string safe_key = sanitize_key_for_error(key);
            return make_error(400, "invalid_request", "unknown field '" + safe_key + "'", safe_key);
        }
    }

    if (!out.has_profile && !out.has_model) {
        return make_error(400, "invalid_request", "either 'profile' or 'model' is required");
    }
    if (out.has_profile && out.has_model) {
        return make_error(400, "invalid_request", "specify either 'profile' or 'model', not both");
    }
    if (out.plan_request.min_ctx_size > out.plan_request.target_ctx_size) {
        out.plan_request.min_ctx_size = out.plan_request.target_ctx_size;
    }
    return out;
}

std::variant<json::Object, ApiError> execute_plan_request(const ServerOptions& opts, const PlanHttpRequest& request) {
    std::string resolved_model_path;
    std::string resolved_mmproj_path;
    std::string profile_name = "planned";

    if (request.has_model) {
        if (!resolve_model_in_dirs(opts.backend.model_dirs, request.model, resolved_model_path)) {
            return make_error(404, "not_found", "model '" + request.model + "' was not found in configured model_dirs",
                              "model");
        }
        if (request.model.size() > 5) {
            profile_name = request.model.substr(0, request.model.size() - 5);
        }
    } else if (request.has_profile) {
        if (!resolve_profile_paths(opts, request.profile, resolved_model_path, resolved_mmproj_path)) {
            return make_error(404, "not_found", "profile '" + request.profile + "' was not found", "profile");
        }
        profile_name = request.profile;
    } else {
        return make_error(400, "invalid_request", "either 'profile' or 'model' is required");
    }

    const Result<GgufModelInfo> info = read_gguf_model_info(resolved_model_path);
    if (!info.ok()) {
        if (info.status().code() == ErrorCode::not_found) {
            return make_error(404, "not_found", "model file is not available");
        }
        const std::string redacted = redact_model_path(info.status(), resolved_model_path);
        if (info.status().code() == ErrorCode::invalid_argument) {
            return make_error(400, "invalid_request", "cannot parse GGUF header: " + redacted);
        }
        return make_error(500, "internal_error", "cannot read GGUF header: " + redacted);
    }

    PlacementPlanRequest plan_req = request.plan_request;
    plan_req.profile_name = profile_name;
    const std::string safe_model_basename = path_basename(resolved_model_path);
    plan_req.model_path = safe_model_basename.empty() ? std::string("model.gguf") : safe_model_basename;
    if (!resolved_mmproj_path.empty()) {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(std::filesystem::path(resolved_mmproj_path), ec);
        const std::string safe_mmproj_basename = path_basename(resolved_mmproj_path);
        if (!ec && !safe_mmproj_basename.empty()) {
            plan_req.mmproj_path = safe_mmproj_basename;
            plan_req.mmproj_bytes = static_cast<std::uint64_t>(sz);
        }
    }

    PlacementPlan plan = plan_model_placement(info.value(), cached_gpu_inventory(), read_host_memory(), plan_req);

    // Strictly redact any filesystem paths in the HTTP response to basename only.
    plan.profile.model = path_basename(plan.profile.model);
    if (plan.profile.model.empty()) {
        plan.profile.model = "model.gguf";
    }
    if (!plan.profile.mmproj.empty()) {
        plan.profile.mmproj = path_basename(plan.profile.mmproj);
    }
    if (plan.profile.speculative && !plan.profile.speculative->draft_model.empty()) {
        plan.profile.speculative->draft_model = path_basename(plan.profile.speculative->draft_model);
    }

    json::Object out;
    out.set("schema", "sonder.inference.plan/1");
    const json::Object planned = placement_plan_to_json(plan);
    for (const auto& [k, v] : planned) {
        out.set(k, v);
    }
    out.set("sonder", sonder_meta());
    return out;
}

json::Object build_identity_body(const ServerOptions& opts, const ModelResidency& residency,
                                 const std::string& served_id, bool synthetic,
                                 const std::optional<std::string>& probed,
                                 const std::shared_ptr<Backend>& backend) {
    // Identity never loads a model (no backend I/O here): a model that is
    // not resident reports the descriptor of its last load, if any.
    IdentityResult id;
    const std::shared_ptr<Model> resident = residency.resident_model(served_id);
    std::string backend_name;
    std::optional<ModelDescriptor> last_desc;
    const ModelDescriptor* descriptor = nullptr;
    if (resident != nullptr) {
        backend_name = resident->backend_name();
        descriptor = &resident->descriptor();
    } else {
        for (const auto& m : residency.snapshot()) {
            if (m.id == served_id) {
                backend_name = m.backend;
            }
        }
        last_desc = residency.last_descriptor(served_id);
        if (last_desc.has_value()) {
            descriptor = &*last_desc;
        }
    }
    if (backend_name == "llamacpp") {
        if (descriptor == nullptr) {
            id = backend_identity(backend_name, served_id, nullptr, probed);
        } else {
            std::string load_name = served_id;
            if (const ModelProfileBinding* binding = find_profile_binding(opts, served_id);
                binding != nullptr && !binding->load_name.empty()) {
                load_name = binding->load_name;
            }
            const std::string gguf_path = resolve_llamacpp_gguf_path(load_name, opts.backend.model_dirs);
            if (gguf_path.empty()) {
                id.reason = "llamacpp: GGUF file '" + load_name + "' could not be resolved to a readable regular file";
            } else {
                const std::vector<GpuDeviceInfo> gpus = cached_gpu_inventory();
                id = gguf_backend_identity(backend_name, served_id, gguf_path, descriptor->context_length, probed,
                                           &gpus);
            }
        }
    } else if (resident != nullptr) {
        id = backend_identity(*resident, probed);
    } else {
        id = backend_identity(backend_name, served_id, descriptor, probed);
    }
    return json::Object{{"schema", "sonder.inference.identity/1"},
                        {"model", served_id},
                        {"synthetic", synthetic},
                        {"backend_identity", id.backend_identity},
                        {"reason", id.reason},
                        {"capabilities", server_capabilities(backend, opts)},
                        {"sonder", sonder_meta()}};
}
json::Object build_discovery_document(const ServerOptions& opts, const LiveTelemetryHub& hub,
                                      const std::string& instance_id, const std::string& node_id, bool synthetic) {
    const HubStats hs = hub.stats();
    return json::Object{
        {"schema", "sonder.telemetry.producer/1"},
        {"producer", json::Object{{"name", "sonder-inference"},
                                  {"version", version_string()},
                                  {"node_id", node_id},
                                  {"instance_id", instance_id},
                                  {"role", "inference"},
                                  {"synthetic", synthetic}}},
        {"event_schema", std::string(kObservatorySchema)},
        {"streams", json::Array{json::Object{{"transport", "sse"}, {"url", "/v1/telemetry/sse"}},
                                json::Object{{"transport", "ndjson"}, {"url", "/v1/telemetry/ndjson"}}}},
        {"resume", json::Object{{"header", "Last-Event-ID"},
                                {"query", "last_event_id"},
                                {"retained_events", hs.retained},
                                {"oldest_sequence", hs.oldest_sequence},
                                {"next_sequence", hs.next_sequence}}},
        {"auth", json::Object{{"required", !opts.token.empty()}, {"schemes", json::Array{"bearer"}}}},
        {"clock", json::Object{{"mono_ns", "host-monotonic"}}},
        {"sampling_level", to_string(opts.telemetry_level)},
        {"text_capture", opts.capture_text ? "on" : "off"},
        {"links", json::Object{{"health", "/v1/sonder/health"},
                               {"identity", "/v1/sonder/identity"},
                               {"models", "/v1/models"},
                               {"node", "/v1/sonder/node"},
                               {"plan", "/v1/sonder/plan"}}},
        {"vocabularies", json::Object{{"sonder.inference.events", 1}}},
        {"stats", json::Object{{"subscribers", hs.subscribers},
                               {"max_subscribers", opts.max_subscribers},
                               {"capacity", hs.capacity},
                               {"subscriber_dropped_events", hs.subscriber_dropped_events}}},
        {"sonder", sonder_meta()}};
}

void handle_telemetry_stream(Exchange& ex, const RequestHead& head, Route route, ConnFlags& flags,
                             const ServerOptions& opts, LiveTelemetryHub& hub, const std::string& instance_id,
                             const std::atomic<bool>& hard_stop) {
    bool sse = true;
    if (route == Route::telemetry_ndjson) {
        sse = false;
    } else if (route == Route::telemetry) {
        const auto format = head.query.empty() ? std::nullopt : query_param(head.query, "format");
        if (format && (*format == "ndjson" || *format == "sse")) {
            sse = *format == "sse";
        } else if (format) {
            send_error(ex, make_error(400, "invalid_request", "format must be sse or ndjson", "format"),
                       opts.write_stall_timeout, hard_stop);
            return;
        } else if (const std::string* accept = head.header("accept")) {
            sse = accept->find("application/x-ndjson") == std::string::npos;
        }
    }
    ResumeRequest resume;
    if (const std::string* last = head.header("last-event-id")) {
        resume.last_event_id = *last;
    } else if (!head.query.empty()) {
        resume.last_event_id = query_param(head.query, "last_event_id");
    }
    if (!head.query.empty()) {
        const auto since = query_param(head.query, "since");
        resume.since_now = since && *since == "now";
    }
    auto subscribed = hub.subscribe(resume);
    if (subscribed.error == LiveTelemetryHub::SubscribeError::over_capacity) {
        ApiError e = make_error(429, "overloaded",
                                "telemetry subscriber limit reached (" + std::to_string(opts.max_subscribers) +
                                    " streams)");
        e.retry_after = 1;
        send_error(ex, e, opts.write_stall_timeout, hard_stop);
        return;
    }
    if (subscribed.error == LiveTelemetryHub::SubscribeError::closed) {
        send_error(ex, make_error(503, "not_ready", "telemetry is shutting down"), opts.write_stall_timeout, hard_stop);
        return;
    }
    const auto sub = subscribed.subscription;
    // From here on the stream never touches the engine.
    flags.streaming.store(true);
    ex.keep_alive = false;
    Headers h = base_headers(ex, sse ? "text/event-stream; charset=utf-8" : "application/x-ndjson");
    h.emplace_back("X-Accel-Buffering", "no");
    ex.status = 200;
    std::string preamble = format_head(200, h);
    if (sse) {
        preamble += "retry: 2000\n\n";
        if (!subscribed.gap_comment.empty()) {
            preamble += subscribed.gap_comment + "\n\n";
        }
    }
    bool ok = send_raw(ex, preamble, opts.write_stall_timeout, hard_stop);
    auto last_write = std::chrono::steady_clock::now();
    while (ok) {
        const auto since_write = std::chrono::steady_clock::now() - last_write;
        const auto wait = std::max<std::chrono::steady_clock::duration>(
            std::chrono::milliseconds(1),
            std::min<std::chrono::steady_clock::duration>(opts.heartbeat_interval - since_write,
                                                          std::chrono::milliseconds(250)));
        auto batch = sub->next(std::chrono::duration_cast<std::chrono::milliseconds>(wait), &hard_stop);
        std::string frame;
        if (batch.dropped > 0 && sse) {
            frame += ": dropped " + std::to_string(batch.dropped) + "\n\n";
        }
        for (const auto& ev : batch.events) {
            if (sse) {
                frame += "id: " + instance_id + "-" + std::to_string(ev->sequence) + "\ndata: ";
                frame += ev->line;
                frame += "\n\n";
            } else {
                frame += ev->line;
                frame += "\n";
            }
        }
        if (!frame.empty()) {
            ok = send_raw(ex, frame, opts.write_stall_timeout, hard_stop);
            last_write = std::chrono::steady_clock::now();
        }
        if (!ok || batch.closed || hard_stop.load()) {
            break;
        }
        if (std::chrono::steady_clock::now() - last_write >= opts.heartbeat_interval) {
            ok = send_raw(ex, sse ? ": keepalive\n\n" : "\n", opts.write_stall_timeout, hard_stop);
            last_write = std::chrono::steady_clock::now();
        }
        if (peer_gone(ex.sock)) {
            break;
        }
    }
    hub.unsubscribe(sub);
}

void BackendProbeRefresher::status(bool& available, std::string& version) const {
    std::lock_guard<std::mutex> lock(cache_->mu);
    available = cache_->valid && cache_->ok;
    version = cache_->ok ? cache_->version : std::string();
}

void BackendProbeRefresher::record(const Result<std::string>& probed) {
    std::lock_guard<std::mutex> lock(cache_->mu);
    cache_->valid = true;
    cache_->ok = probed.ok();
    cache_->version = probed.ok() ? probed.value() : std::string();
}

void BackendProbeRefresher::start(std::shared_ptr<Backend> backend) {
    thread_ = std::thread([cache = cache_, b = std::move(backend)] {
        std::unique_lock<std::mutex> lock(cache->mu);
        while (!cache->stop) {
            cache->cv.wait_for(lock, kProbeInterval, [&] { return cache->stop; });
            if (cache->stop) {
                break;
            }
            lock.unlock();
            const Result<std::string> probed = b->probe();
            {
                std::lock_guard<std::mutex> store_lock(cache->mu);
                cache->valid = true;
                cache->ok = probed.ok();
                cache->version = probed.ok() ? probed.value() : std::string();
            }
            lock.lock();
        }
        cache->exited = true;
        cache->cv.notify_all();
    });
}

bool BackendProbeRefresher::stop(std::chrono::milliseconds patience) {
    if (!thread_.joinable()) {
        return true;
    }
    bool exited = false;
    {
        std::unique_lock<std::mutex> lock(cache_->mu);
        cache_->stop = true;
        cache_->cv.notify_all();
        exited = cache_->cv.wait_for(lock, patience, [this] { return cache_->exited; });
    }
    if (exited) {
        thread_.join();
        return true;
    }
    thread_.detach();
    return false;
}

}  // namespace sonder::inference::server::detail
