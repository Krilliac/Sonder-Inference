#include "residency_config.hpp"

#include "config_text.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace sonder::inference::detail {
namespace {

using Config = BackendSetup::LlamaServerResidencyConfig;

Status bad(const char *key, const char *range) {
    return Status(ErrorCode::invalid_argument,
                  std::string("llamaserver config: spill_guard.residency.") + key + " must be " + range);
}

}  // namespace

Status parse_uma_mode_value(const json::Value &value, const char *field_path, std::string &out) {
    if (value.is_bool()) {
        out = value.as_bool() ? "on" : "off";
        return {};
    }
    if (value.is_string()) {
        if (const auto normalized = sonder::inference::normalize_uma_mode(value.as_string())) {
            out.assign(*normalized);
            return {};
        }
    }
    return Status(ErrorCode::invalid_argument,
                  std::string("llamaserver config: ") + field_path +
                      " must be boolean or \"auto\", \"on\", \"off\", \"true\", \"false\"");
}

Status parse_residency_config(const json::Value &value, BackendSetup &setup) {
    if (!value.is_object())
        return Status(ErrorCode::invalid_argument, "llamaserver config: spill_guard.residency must be an object");

    static constexpr std::string_view keys[] = {"enabled", "expect_gpu", "min_dedicated_mib",
                                                "consecutive_samples", "eviction_fraction", "eviction_mib",
                                                "on_eviction", "max_eviction_restarts", "uma_mode"};
    for (const auto &member : value.as_object()) {
        if (std::find(std::begin(keys), std::end(keys), member.first) == std::end(keys))
            return Status(ErrorCode::invalid_argument,
                          "llamaserver config: unknown spill_guard.residency field '" + bounded_for_message(member.first) + "'");
    }

    Config parsed = setup.llamaserver_residency;
    if (const auto *v = value.find("enabled")) {
        if (!v->is_bool()) return bad("enabled", "boolean");
        parsed.enabled = v->as_bool();
    }
    if (const auto *v = value.find("expect_gpu")) {
        if (!v->is_bool()) return bad("expect_gpu", "boolean");
        parsed.expect_gpu = v->as_bool();
    }
    const auto uint_in = [&](const char *key, std::uint64_t lo, std::uint64_t hi,
                             std::uint64_t &out, const char *range) -> Status {
        if (const auto *v = value.find(key)) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < lo || v->as_uint() > hi)
                return bad(key, range);
            out = v->as_uint();
        }
        return {};
    };
    if (auto st = uint_in("min_dedicated_mib", 1, 1048576, parsed.min_dedicated_mib,
                          "an integer in [1, 1048576]"); !st.ok()) return st;
    if (auto st = uint_in("consecutive_samples", 1, 1000, parsed.consecutive_samples,
                          "an integer in [1, 1000]"); !st.ok()) return st;
    if (auto st = uint_in("eviction_mib", 0, 1048576, parsed.eviction_mib,
                          "an integer in [0, 1048576]"); !st.ok()) return st;
    if (auto st = uint_in("max_eviction_restarts", 0, 32, parsed.max_eviction_restarts,
                          "an integer in [0, 32]"); !st.ok()) return st;
    if (const auto *v = value.find("eviction_fraction")) {
        if (!v->is_number() || !std::isfinite(v->as_double()) || v->as_double() < 0.0 || v->as_double() > 1.0)
            return bad("eviction_fraction", "a finite number in [0, 1]");
        parsed.eviction_fraction = v->as_double();
    }
    if (const auto *v = value.find("on_eviction")) {
        if (!v->is_string() || (v->as_string() != "warn" && v->as_string() != "restart"))
            return bad("on_eviction", "warn or restart");
        parsed.on_eviction = v->as_string();
    }
    if (const auto *v = value.find("uma_mode")) {
        if (auto st = parse_uma_mode_value(*v, "spill_guard.residency.uma_mode", parsed.uma_mode); !st.ok())
            return st;
    }
    if (parsed.eviction_fraction == 0.0 && parsed.eviction_mib == 0)
        return Status(ErrorCode::invalid_argument,
                      "llamaserver config: spill_guard.residency.eviction_fraction and eviction_mib cannot both be zero");
    setup.llamaserver_residency = parsed;
    return {};
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
Status apply_residency_config(const BackendSetup &setup,
                              LlamaServerResidencyGuardOptions &out,
                              const std::vector<GpuDeviceInfo> &gpus) {
    const auto &in = setup.llamaserver_residency;
    if (in.min_dedicated_mib == 0 || in.min_dedicated_mib > 1048576)
        return Status(ErrorCode::invalid_argument, "llamaserver residency min_dedicated_mib must be in [1, 1048576]");
    if (in.consecutive_samples == 0 || in.consecutive_samples > 1000)
        return Status(ErrorCode::invalid_argument, "llamaserver residency consecutive_samples must be in [1, 1000]");
    if (!std::isfinite(in.eviction_fraction) || in.eviction_fraction < 0.0 || in.eviction_fraction > 1.0)
        return Status(ErrorCode::invalid_argument, "llamaserver residency eviction_fraction must be finite and in [0, 1]");
    if (in.eviction_mib > 1048576)
        return Status(ErrorCode::invalid_argument, "llamaserver residency eviction_mib must be in [0, 1048576]");
    if (in.max_eviction_restarts > 32)
        return Status(ErrorCode::invalid_argument, "llamaserver residency max_eviction_restarts must be in [0, 32]");
    if (in.eviction_fraction == 0.0 && in.eviction_mib == 0)
        return Status(ErrorCode::invalid_argument, "llamaserver residency eviction thresholds cannot both be zero");
    if (in.on_eviction != "warn" && in.on_eviction != "restart")
        return Status(ErrorCode::invalid_argument, "llamaserver residency on_eviction must be warn or restart");
    if (!is_valid_uma_mode(setup.llamaserver_spill_uma_mode))
        return Status(ErrorCode::invalid_argument,
                      "llamaserver spill_guard.uma_mode must be on, off, auto, true or false");
    if (!in.uma_mode.empty() && !is_valid_uma_mode(in.uma_mode))
        return Status(ErrorCode::invalid_argument,
                      "llamaserver residency uma_mode must be empty or on, off, auto, true or false");

    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    const std::string_view effective_uma_mode =
        in.uma_mode.empty() ? std::string_view(setup.llamaserver_spill_uma_mode)
                            : std::string_view(in.uma_mode);
    const std::string_view device_hint = extract_device_hint(setup.llamaserver_args);

    out.enabled = in.enabled;
    out.expect_gpu = in.expect_gpu;
    out.min_dedicated_bytes = in.min_dedicated_mib * kMiB;
    out.consecutive_samples = static_cast<std::size_t>(in.consecutive_samples);
    out.eviction_fraction = in.eviction_fraction;
    out.eviction_bytes = in.eviction_mib * kMiB;
    out.on_eviction = in.on_eviction == "restart" ? LlamaServerEvictionPolicy::restart
                                                    : LlamaServerEvictionPolicy::warn;
    out.max_eviction_restarts = static_cast<std::size_t>(in.max_eviction_restarts);
    out.uma_mode = resolve_uma_mode(effective_uma_mode, gpus, device_hint);
    return {};
}

Status apply_residency_config(const BackendSetup &setup, LlamaServerResidencyGuardOptions &out) {
    const auto &in = setup.llamaserver_residency;
    const std::string_view effective_uma_mode =
        in.uma_mode.empty() ? std::string_view(setup.llamaserver_spill_uma_mode)
                            : std::string_view(in.uma_mode);
    if (is_valid_uma_mode(setup.llamaserver_spill_uma_mode) &&
        (in.uma_mode.empty() || is_valid_uma_mode(in.uma_mode)) &&
        effective_uma_mode == "auto") {
        return apply_residency_config(setup, out, enumerate_gpu_devices());
    }
    return apply_residency_config(setup, out, {});
}
#endif

namespace {

std::uint64_t sat_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::uint64_t sat_sub(std::uint64_t a, std::uint64_t b) noexcept { return a >= b ? (a - b) : 0; }

}  // namespace

server::ModelResourceFootprint footprint_from_vram_estimate(const VramEstimate &estimate,
                                                            std::uint64_t projector_vram_bytes,
                                                            std::uint64_t projector_ram_bytes,
                                                            std::uint64_t adapter_vram_bytes,
                                                            std::uint64_t adapter_ram_bytes) noexcept {
    server::ModelResourceFootprint fp;

    // VRAM tier: VramEstimate::weights_bytes already includes offloaded mmproj
    // bytes when present, so split projector_vram_bytes out cleanly without
    // double-counting.
    const std::uint64_t vram_proj = std::min(projector_vram_bytes, estimate.weights_bytes);
    fp.vram.weights_bytes = sat_sub(estimate.weights_bytes, vram_proj);
    fp.vram.kv_cache_bytes = sat_add(estimate.kv_bytes, estimate.recurrent_bytes);
    fp.vram.workspace_bytes = sat_add(estimate.compute_bytes, estimate.moe_cache_bytes);
    fp.vram.projector_bytes = projector_vram_bytes;
    fp.vram.adapter_bytes = adapter_vram_bytes;

    // Host RAM tier: preserve exact conservation with host_resident_bytes.
    const std::uint64_t host_non_weight =
        sat_add(sat_add(estimate.host_kv_bytes, estimate.host_recurrent_bytes), estimate.host_compute_bytes);
    const std::uint64_t host_weight_resident = sat_sub(estimate.host_resident_bytes, host_non_weight);
    const std::uint64_t ram_proj = std::min(projector_ram_bytes, host_weight_resident);
    fp.ram.weights_bytes = sat_sub(host_weight_resident, ram_proj);
    fp.ram.kv_cache_bytes = sat_add(estimate.host_kv_bytes, estimate.host_recurrent_bytes);
    fp.ram.workspace_bytes = estimate.host_compute_bytes;
    fp.ram.projector_bytes = projector_ram_bytes;
    fp.ram.adapter_bytes = adapter_ram_bytes;
    return fp;
}

server::ModelResourceFootprint footprint_from_placement_plan(const PlacementPlan &plan,
                                                             std::uint64_t projector_bytes,
                                                             std::uint64_t adapter_vram_bytes,
                                                             std::uint64_t adapter_ram_bytes) noexcept {
    const bool mmproj_on_gpu = !plan.profile.mmproj.empty() && plan.estimate.gpu_layers > 0 &&
                               plan.profile.mmproj_offload.value_or(true);
    const std::uint64_t proj_vram = mmproj_on_gpu ? projector_bytes : 0;
    const std::uint64_t proj_ram = (!plan.profile.mmproj.empty() && !mmproj_on_gpu) ? projector_bytes : 0;
    server::ModelResourceFootprint fp =
        footprint_from_vram_estimate(plan.estimate, proj_vram, proj_ram, adapter_vram_bytes, adapter_ram_bytes);
    if (plan.uma_mode) {
        // Integrated UMA allocations come out of unified physical RAM as well
        // as the logical GPU pool.
        fp.ram.weights_bytes = sat_add(fp.ram.weights_bytes, fp.vram.weights_bytes);
        fp.ram.kv_cache_bytes = sat_add(fp.ram.kv_cache_bytes, fp.vram.kv_cache_bytes);
        fp.ram.workspace_bytes = sat_add(fp.ram.workspace_bytes, fp.vram.workspace_bytes);
        fp.ram.projector_bytes = sat_add(fp.ram.projector_bytes, fp.vram.projector_bytes);
        fp.ram.adapter_bytes = sat_add(fp.ram.adapter_bytes, fp.vram.adapter_bytes);
    }
    return fp;
}

std::vector<std::pair<std::string, server::ModelResourceFootprint>> footprints_from_coresidency_plan(
    const CoResidencyPlan &plan) {
    std::vector<std::pair<std::string, server::ModelResourceFootprint>> out;
    out.reserve(plan.models.size());
    for (const auto &model_plan : plan.models) {
        out.emplace_back(model_plan.profile.name, footprint_from_placement_plan(model_plan));
    }
    return out;
}

Status parse_live_residency_budgets(const json::Value &value, std::uint64_t &max_vram_bytes,
                                    std::uint64_t &max_ram_bytes) {
    if (!value.is_object()) {
        return Status(ErrorCode::invalid_argument, "residency budget config must be a JSON object");
    }
    static constexpr std::string_view kKeys[] = {"max_resident_vram_mib", "max_resident_ram_mib",
                                                 "max_resident_vram_bytes", "max_resident_ram_bytes"};
    for (const auto &member : value.as_object()) {
        if (std::find(std::begin(kKeys), std::end(kKeys), member.first) == std::end(kKeys)) {
            return Status(ErrorCode::invalid_argument,
                          "residency budget config: unknown field '" + bounded_for_message(member.first) + "'");
        }
    }
    const bool has_vram_mib = value.find("max_resident_vram_mib") != nullptr;
    const bool has_vram_bytes = value.find("max_resident_vram_bytes") != nullptr;
    if (has_vram_mib && has_vram_bytes) {
        return Status(ErrorCode::invalid_argument,
                      "residency budget config: specify only one of max_resident_vram_mib or max_resident_vram_bytes");
    }
    const bool has_ram_mib = value.find("max_resident_ram_mib") != nullptr;
    const bool has_ram_bytes = value.find("max_resident_ram_bytes") != nullptr;
    if (has_ram_mib && has_ram_bytes) {
        return Status(ErrorCode::invalid_argument,
                      "residency budget config: specify only one of max_resident_ram_mib or max_resident_ram_bytes");
    }
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kMaxMiB = 16777216ull;  // 16 TiB
    constexpr std::uint64_t kMaxBytes = kMaxMiB * kMiB;

    std::uint64_t parsed_vram = max_vram_bytes;
    std::uint64_t parsed_ram = max_ram_bytes;
    const auto parse_u64 = [&](const char *key, std::uint64_t hi, std::uint64_t scale, std::uint64_t &dst) -> Status {
        if (const auto *v = value.find(key)) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > hi) {
                return Status(ErrorCode::invalid_argument,
                              std::string("residency budget config: ") + key + " must be an integer in [0, " +
                                  std::to_string(hi) + "]");
            }
            dst = v->as_uint() * scale;
        }
        return {};
    };
    if (auto st = parse_u64("max_resident_vram_mib", kMaxMiB, kMiB, parsed_vram); !st.ok()) return st;
    if (auto st = parse_u64("max_resident_ram_mib", kMaxMiB, kMiB, parsed_ram); !st.ok()) return st;
    if (auto st = parse_u64("max_resident_vram_bytes", kMaxBytes, 1u, parsed_vram); !st.ok()) return st;
    if (auto st = parse_u64("max_resident_ram_bytes", kMaxBytes, 1u, parsed_ram); !st.ok()) return st;
    max_vram_bytes = parsed_vram;
    max_ram_bytes = parsed_ram;
    return {};
}

}  // namespace sonder::inference::detail
