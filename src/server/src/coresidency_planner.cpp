// Pure, deterministic multi-model co-residency VRAM planner
// (coresidency_planner.hpp).
#include "sonder/inference/coresidency_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference {

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) noexcept {
    if (a == 0 || b == 0) return 0;
    return a > std::numeric_limits<std::uint64_t>::max() / b ? std::numeric_limits<std::uint64_t>::max() : a * b;
}

std::uint64_t saturating_u64(double v) noexcept {
    const double r = std::floor(v + 0.5);
    if (!(r > 0.0)) return 0;
    if (r >= 18446744073709551616.0) return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t>(r);
}

std::uint64_t bytes_to_mib_ceil(std::uint64_t bytes) noexcept {
    return (bytes >> 20) + ((bytes & (kMiB - 1)) != 0 ? 1ull : 0ull);
}

std::string_view trim_view(std::string_view sv) noexcept {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const auto last = sv.find_last_not_of(" \t\r\n");
    return sv.substr(first, last - first + 1);
}

char ascii_tolower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (ascii_tolower(a[i]) != ascii_tolower(b[i])) return false;
    }
    return true;
}

bool istarts_with(std::string_view sv, std::string_view prefix) noexcept {
    return sv.size() >= prefix.size() && iequals(sv.substr(0, prefix.size()), prefix);
}

std::string sanitize_for_warning(std::string_view sv) {
    constexpr std::size_t kMaxLen = 64;
    std::string out;
    out.reserve(std::min(sv.size(), kMaxLen) + 3);
    for (std::size_t i = 0; i < sv.size(); ++i) {
        if (out.size() >= kMaxLen) {
            out += "...";
            break;
        }
        const unsigned char ch = static_cast<unsigned char>(sv[i]);
        out.push_back((ch < 0x20u || ch == 0x7Fu) ? ' ' : static_cast<char>(ch));
    }
    return out;
}

std::optional<std::size_t> parse_index(std::string_view text) noexcept {
    if (text.empty() || text.size() > 20) return std::nullopt;
    std::size_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') return std::nullopt;
        const auto digit = static_cast<std::size_t>(c - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}

bool is_nvidia_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::nvidia || iequals(g.backend_hint, "cuda");
}

bool is_amd_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::amd || iequals(g.backend_hint, "rocm");
}

bool is_intel_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::intel || iequals(g.backend_hint, "sycl");
}

bool is_apple_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::apple || iequals(g.backend_hint, "metal");
}

std::string gpu_id_or_fallback(const std::vector<GpuDeviceInfo>& gpus, std::size_t idx) {
    if (idx < gpus.size() && !gpus[idx].id.empty()) return gpus[idx].id;
    return "gpu:" + std::to_string(idx);
}

std::string gpu_label(const std::vector<GpuDeviceInfo>& gpus, std::size_t idx) {
    const std::string id = gpu_id_or_fallback(gpus, idx);
    if (idx < gpus.size() && !gpus[idx].name.empty()) {
        return id + " (" + gpus[idx].name + ")";
    }
    return id;
}

// Matches compute_gpu_budget() in placement_planner.cpp so the single OS/desktop
// reserve deducted on a GPU's first occupancy is identical to single-model plan.
std::uint64_t compute_initial_reserve_mib(const GpuDeviceInfo& gpu, const HostMemory& host,
                                          const PlacementPlanRequest& request) noexcept {
    const auto raw = gpu_effective_vram_mib(gpu, request.prefer_free_vram);
    if (!raw || *raw == 0 || request.vram_headroom_mib == 0) return 0;

    if (!gpu.is_integrated_uma) {
        return std::min(*raw, request.vram_headroom_mib);
    }

    std::uint64_t pool = *raw;
    if (host.available_physical.is_known()) {
        pool = std::min(pool, host.available_physical.bytes_or(0) / kMiB);
    }
    if (pool == 0) return 0;
    const std::uint64_t proportional = pool / 8;  // 12.5%
    const std::uint64_t floor_reserve = std::min<std::uint64_t>(512, pool / 4);
    return std::min(request.vram_headroom_mib, std::max(floor_reserve, proportional));
}

void deduct_uma_total_pool_bytes(GpuDeviceInfo& gpu, std::uint64_t bytes) noexcept {
    std::uint64_t rem = bytes;
    if (gpu.dedicated_total.is_known()) {
        const std::uint64_t cur = gpu.dedicated_total.bytes_or(0);
        const std::uint64_t take = std::min(cur, rem);
        gpu.dedicated_total = MemoryReading::known(cur - take);
        rem -= take;
    }
    if (rem > 0 && gpu.shared_total.is_known()) {
        const std::uint64_t cur = gpu.shared_total.bytes_or(0);
        const std::uint64_t take = std::min(cur, rem);
        gpu.shared_total = MemoryReading::known(cur - take);
    }
}

void deduct_gpu_bytes(GpuDeviceInfo& gpu, std::uint64_t bytes) noexcept {
    if (bytes == 0) return;
    if (!gpu.is_integrated_uma) {
        if (gpu.dedicated_free.is_known()) {
            const std::uint64_t cur = gpu.dedicated_free.bytes_or(0);
            gpu.dedicated_free = MemoryReading::known(cur > bytes ? cur - bytes : 0);
        }
        if (gpu.dedicated_total.is_known()) {
            const std::uint64_t cur = gpu.dedicated_total.bytes_or(0);
            gpu.dedicated_total = MemoryReading::known(cur > bytes ? cur - bytes : 0);
        }
        return;
    }
    if (gpu.dedicated_free.is_known()) {
        const std::uint64_t cur = gpu.dedicated_free.bytes_or(0);
        gpu.dedicated_free = MemoryReading::known(cur > bytes ? cur - bytes : 0);
    }
    deduct_uma_total_pool_bytes(gpu, bytes);
}

void cap_gpu_bytes(GpuDeviceInfo& gpu, std::uint64_t cap_bytes) noexcept {
    if (!gpu.is_integrated_uma) {
        if (gpu.dedicated_free.is_known()) {
            gpu.dedicated_free = MemoryReading::known(std::min(gpu.dedicated_free.bytes_or(0), cap_bytes));
        }
        if (gpu.dedicated_total.is_known()) {
            gpu.dedicated_total = MemoryReading::known(std::min(gpu.dedicated_total.bytes_or(0), cap_bytes));
        }
        return;
    }
    if (gpu.dedicated_free.is_known()) {
        gpu.dedicated_free = MemoryReading::known(std::min(gpu.dedicated_free.bytes_or(0), cap_bytes));
    }
    const std::uint64_t cur_pool = saturating_add(gpu.dedicated_total.bytes_or(0), gpu.shared_total.bytes_or(0));
    if (cur_pool > cap_bytes) {
        deduct_uma_total_pool_bytes(gpu, cur_pool - cap_bytes);
    }
}

std::uint64_t effective_usable_mib(const GpuDeviceInfo& gpu, const HostMemory& host,
                                   const PlacementPlanRequest& request) noexcept {
    const auto raw = gpu_effective_vram_mib(gpu, request.prefer_free_vram);
    if (!raw || *raw == 0) return 0;
    const std::uint64_t reserve = compute_initial_reserve_mib(gpu, host, request);
    std::uint64_t pool = *raw;
    if (gpu.is_integrated_uma && host.available_physical.is_known()) {
        pool = std::min(pool, host.available_physical.bytes_or(0) / kMiB);
    }
    return pool > reserve ? pool - reserve : 0;
}

std::uint64_t remaining_gpu_summary_mib(const GpuDeviceInfo& gpu, const HostMemory& host, bool prefer_free,
                                        std::uint64_t reserved_mib) noexcept {
    const auto raw = gpu_effective_vram_mib(gpu, prefer_free);
    std::uint64_t rem = raw.value_or(0);
    if (gpu.is_integrated_uma && host.available_physical.is_known()) {
        const std::uint64_t host_avail_mib = host.available_physical.bytes_or(0) / kMiB;
        const std::uint64_t host_cap = host_avail_mib > reserved_mib ? host_avail_mib - reserved_mib : 0;
        rem = std::min(rem, host_cap);
    }
    return rem;
}

template <typename Pred>
std::optional<std::size_t> nth_matching_gpu(const std::vector<GpuDeviceInfo>& gpus, std::size_t local_idx,
                                            const Pred& pred) noexcept {
    std::size_t seen = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i])) continue;
        if (seen == local_idx) return i;
        ++seen;
    }
    return std::nullopt;
}

std::optional<std::size_t> resolve_ggml_device_token(const std::vector<GpuDeviceInfo>& gpus,
                                                     std::string_view token) noexcept {
    token = trim_view(token);
    if (token.empty()) return std::nullopt;

    if (istarts_with(token, "cuda")) {
        const auto local = parse_index(token.substr(4));
        if (!local) return std::nullopt;
        return nth_matching_gpu(gpus, *local, is_nvidia_gpu);
    }
    if (istarts_with(token, "rocm")) {
        const auto local = parse_index(token.substr(4));
        if (!local) return std::nullopt;
        return nth_matching_gpu(gpus, *local, is_amd_gpu);
    }
    if (istarts_with(token, "sycl")) {
        const auto local = parse_index(token.substr(4));
        if (!local) return std::nullopt;
        return nth_matching_gpu(gpus, *local, is_intel_gpu);
    }
    if (istarts_with(token, "metal")) {
        const auto local = parse_index(token.substr(5));
        if (!local) return std::nullopt;
        return nth_matching_gpu(gpus, *local, is_apple_gpu);
    }
    if (istarts_with(token, "vulkan")) {
        const auto global_idx = parse_index(token.substr(6));
        if (global_idx && *global_idx < gpus.size()) return *global_idx;
        return std::nullopt;
    }
    return std::nullopt;
}

// Inverse of ggml_device_name_for() / join_device_names() in placement_planner.cpp.
std::vector<std::size_t> resolve_placed_gpu_indices(const std::vector<GpuDeviceInfo>& gpus, const PlacementPlan& plan) {
    std::vector<std::size_t> indices;
    if (!plan.profile.device.has_value() || plan.profile.device->empty()) return indices;
    if (plan.profile.n_gpu_layers.has_value() && *plan.profile.n_gpu_layers == 0) return indices;
    if (plan.estimate.total_bytes == 0) return indices;

    const std::string_view dev = *plan.profile.device;
    std::size_t pos = 0;
    while (pos <= dev.size()) {
        const std::size_t comma = dev.find(',', pos);
        const std::string_view token =
            trim_view(dev.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? dev.size() + 1 : comma + 1;
        if (token.empty()) continue;
        if (const auto idx = resolve_ggml_device_token(gpus, token)) {
            if (std::find(indices.begin(), indices.end(), *idx) == indices.end()) {
                indices.push_back(*idx);
            }
        }
    }
    return indices;
}

// Computes the per-GPU VRAM footprint (in MiB, rounded up) for each placed GPU,
// matching vram_estimate.cpp and placement_planner.cpp's multi-GPU shard math.
std::vector<std::uint64_t> compute_per_gpu_vram_mib(const PlacementPlan& plan,
                                                    const std::vector<std::size_t>& placed_indices) {
    std::vector<std::uint64_t> per_gpu(placed_indices.size(), 0);
    if (placed_indices.empty() || plan.estimate.total_bytes == 0) return per_gpu;

    const VramEstimate& e = plan.estimate;
    const LaunchProfile& p = plan.profile;
    const std::string split_mode = p.split_mode.value_or("layer");

    if (placed_indices.size() == 1 || p.tensor_split.size() < 2 || split_mode == "none") {
        std::size_t target = 0;
        if (placed_indices.size() > 1 && p.main_gpu.has_value() && *p.main_gpu >= 0 &&
            static_cast<std::size_t>(*p.main_gpu) < placed_indices.size()) {
            target = static_cast<std::size_t>(*p.main_gpu);
        }
        per_gpu[target] = e.total_mib();
        return per_gpu;
    }

    const std::size_t n = std::min(placed_indices.size(), p.tensor_split.size());
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = p.tensor_split[i];
        if (std::isfinite(w) && w > 0.0) weight_sum += w;
    }

    if (split_mode == "row") {
        const std::uint64_t splittable = saturating_add(e.weights_bytes, e.recurrent_bytes);
        const std::size_t main_pos =
            (p.main_gpu.has_value() && *p.main_gpu >= 0 && static_cast<std::size_t>(*p.main_gpu) < n)
                ? static_cast<std::size_t>(*p.main_gpu)
                : 0u;
        for (std::size_t i = 0; i < n; ++i) {
            const double share = (weight_sum > 0.0 && std::isfinite(p.tensor_split[i]) && p.tensor_split[i] > 0.0)
                                     ? (p.tensor_split[i] / weight_sum)
                                     : (1.0 / static_cast<double>(n));
            const std::uint64_t shard =
                std::min(splittable, saturating_u64(std::ceil(static_cast<double>(splittable) * share)));
            std::uint64_t gpu_bytes = saturating_add(shard, e.compute_bytes);
            if (i == main_pos) {
                gpu_bytes = saturating_add(gpu_bytes, e.kv_bytes);
            }
            per_gpu[i] = bytes_to_mib_ceil(gpu_bytes);
        }
        return per_gpu;
    }

    // Default split_mode == "layer": proportional weights + KV + recurrent + per-GPU compute.
    const std::uint64_t splittable =
        saturating_add(e.weights_bytes, saturating_add(e.kv_bytes, e.recurrent_bytes));
    for (std::size_t i = 0; i < n; ++i) {
        const double share = (weight_sum > 0.0 && std::isfinite(p.tensor_split[i]) && p.tensor_split[i] > 0.0)
                                 ? (p.tensor_split[i] / weight_sum)
                                 : (1.0 / static_cast<double>(n));
        const std::uint64_t shard =
            std::min(splittable, saturating_u64(std::ceil(static_cast<double>(splittable) * share)));
        const std::uint64_t gpu_bytes = saturating_add(shard, e.compute_bytes);
        per_gpu[i] = bytes_to_mib_ceil(gpu_bytes);
    }
    return per_gpu;
}

std::string gpu_selector_token(const std::vector<GpuDeviceInfo>& gpus, std::size_t idx) {
    if (idx < gpus.size() && !gpus[idx].id.empty()) {
        std::size_t matches = 0;
        for (const auto& g : gpus) {
            if (iequals(g.id, gpus[idx].id)) ++matches;
        }
        if (matches == 1) return gpus[idx].id;
    }
    return "gpu:" + std::to_string(idx);
}

std::optional<std::size_t> select_best_uma_gpu(const std::vector<GpuDeviceInfo>& candidate_gpus,
                                               const HostMemory& working_host,
                                               const PlacementPlanRequest& eval_req) noexcept {
    std::optional<std::size_t> best;
    std::uint64_t best_usable = 0;
    for (std::size_t i = 0; i < candidate_gpus.size(); ++i) {
        if (!candidate_gpus[i].is_integrated_uma) continue;
        const std::uint64_t usable = effective_usable_mib(candidate_gpus[i], working_host, eval_req);
        if (usable == 0) continue;
        if (!best || usable > best_usable) {
            best = i;
            best_usable = usable;
        }
    }
    return best;
}

}  // namespace

CoResidencyPlan plan_coresident_models(const std::vector<CoResidencyEntry>& entries,
                                       const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host) {
    CoResidencyPlan result;

    if (entries.empty()) {
        result.all_fit_coresident = true;
        result.remaining_host_ram_mib =
            host.available_physical.is_known() ? (host.available_physical.bytes_or(0) / kMiB) : 0;
        result.remaining_gpu_vram_mib.reserve(gpus.size());
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            result.remaining_gpu_vram_mib.emplace_back(
                gpu_id_or_fallback(gpus, i), remaining_gpu_summary_mib(gpus[i], host, /*prefer_free=*/true, 0));
        }
        result.notes.emplace_back("no models supplied; reported initial available GPU VRAM and host RAM");
        return result;
    }

    std::vector<GpuDeviceInfo> working_gpus = gpus;
    HostMemory working_host = host;
    std::vector<std::uint64_t> reserved_mib(gpus.size(), 0);
    result.models.reserve(entries.size());

    bool all_fit = true;

    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
        const CoResidencyEntry& entry = entries[idx];
        const PlacementPlanRequest& orig_req = entry.request;

        // Determine per-GPU OS/desktop reserve still needed if this model is the
        // first occupant of that GPU; GPUs already occupied reuse their reserve.
        std::vector<std::uint64_t> pending_reserve_mib(working_gpus.size(), 0);
        bool any_gpu_already_reserved = false;
        for (std::size_t k = 0; k < working_gpus.size(); ++k) {
            if (reserved_mib[k] > 0) {
                any_gpu_already_reserved = true;
            } else {
                pending_reserve_mib[k] = compute_initial_reserve_mib(working_gpus[k], working_host, orig_req);
            }
        }

        std::vector<GpuDeviceInfo> candidate_gpus = working_gpus;
        PlacementPlanRequest eval_req = orig_req;
        if (any_gpu_already_reserved) {
            // Apply per-GPU reserve selectively so occupied GPUs do not pay a
            // second OS reserve while unoccupied GPUs still reserve headroom once.
            eval_req.vram_headroom_mib = 0;
            for (std::size_t k = 0; k < candidate_gpus.size(); ++k) {
                if (candidate_gpus[k].is_integrated_uma && working_host.available_physical.is_known()) {
                    const std::uint64_t host_avail_mib = working_host.available_physical.bytes_or(0) / kMiB;
                    const std::uint64_t effective_reserve =
                        reserved_mib[k] > 0 ? reserved_mib[k] : pending_reserve_mib[k];
                    const std::uint64_t raw_mib =
                        gpu_effective_vram_mib(candidate_gpus[k], orig_req.prefer_free_vram).value_or(0);
                    const std::uint64_t host_cap =
                        host_avail_mib > effective_reserve ? host_avail_mib - effective_reserve : 0;
                    const std::uint64_t gpu_cap =
                        reserved_mib[k] > 0 ? raw_mib
                                            : (raw_mib > pending_reserve_mib[k] ? raw_mib - pending_reserve_mib[k] : 0);
                    cap_gpu_bytes(candidate_gpus[k], saturating_mul(std::min(gpu_cap, host_cap), kMiB));
                } else if (pending_reserve_mib[k] > 0) {
                    deduct_gpu_bytes(candidate_gpus[k], saturating_mul(pending_reserve_mib[k], kMiB));
                }
            }
        }

        PlacementPlan model_plan = plan_model_placement(entry.model, candidate_gpus, working_host, eval_req);

        const std::string_view orig_hint = trim_view(orig_req.device_hint);
        const bool explicit_cpu = iequals(orig_hint, "cpu");
        const bool pinned_gpu = !orig_hint.empty() && !explicit_cpu;

        // If the caller left device_hint empty and remaining discrete GPU VRAM
        // could not fit all layers, check whether an integrated UMA iGPU can
        // take the overflow cleanly with better GPU layer coverage.
        const bool full_gpu_offload =
            model_plan.fits_in_vram && model_plan.profile.n_gpu_layers.has_value() && *model_plan.profile.n_gpu_layers < 0;
        if (orig_hint.empty() && !full_gpu_offload) {
            if (const auto uma_idx = select_best_uma_gpu(candidate_gpus, working_host, eval_req)) {
                const auto current_indices = resolve_placed_gpu_indices(working_gpus, model_plan);
                const bool already_on_uma =
                    current_indices.size() == 1 && current_indices.front() == *uma_idx;
                if (!already_on_uma) {
                    PlacementPlanRequest uma_req = eval_req;
                    uma_req.device_hint = gpu_selector_token(candidate_gpus, *uma_idx);
                    PlacementPlan uma_plan = plan_model_placement(entry.model, candidate_gpus, working_host, uma_req);

                    const bool uma_full = uma_plan.fits_in_vram && uma_plan.fits_in_host_ram &&
                                          uma_plan.profile.n_gpu_layers.has_value() &&
                                          *uma_plan.profile.n_gpu_layers < 0;
                    const bool primary_partial = model_plan.fits_in_vram &&
                                                 model_plan.profile.n_gpu_layers.has_value() &&
                                                 *model_plan.profile.n_gpu_layers > 0;
                    const bool uma_better =
                        uma_plan.fits_in_vram && uma_plan.fits_in_host_ram &&
                        (!model_plan.fits_in_vram || uma_full ||
                         (primary_partial && uma_plan.estimate.gpu_layers > model_plan.estimate.gpu_layers));
                    if (uma_better) {
                        const std::string spill_msg =
                            "model[" + std::to_string(idx) + "] '" + uma_plan.profile.name +
                            "' spilled from saturated discrete GPU to integrated UMA " +
                            gpu_label(working_gpus, *uma_idx);
                        uma_plan.reasons.push_back(
                            "co-residency planner spilled model to integrated UMA GPU because discrete GPU "
                            "remaining VRAM could not fit full offload");
                        result.notes.push_back(spill_msg);
                        model_plan = std::move(uma_plan);
                    }
                }
            }
        }

        // If the caller did not pin a GPU and remaining GPU VRAM could not fit
        // even 1 layer, re-plan cleanly as CPU-only so host-RAM sizing applies.
        bool cpu_spill_warned = false;
        if (orig_hint.empty() && !model_plan.fits_in_vram &&
            (!model_plan.profile.device.has_value() ||
             (model_plan.profile.n_gpu_layers.has_value() && *model_plan.profile.n_gpu_layers == 0)) &&
            model_plan.recommended_backend_hint != "cpu") {
            PlacementPlanRequest cpu_req = orig_req;
            cpu_req.device_hint = "cpu";
            PlacementPlan cpu_plan = plan_model_placement(entry.model, candidate_gpus, working_host, cpu_req);
            cpu_plan.warnings.emplace_back(
                "remaining GPU VRAM could not fit even 1 layer; spilled to CPU-only execution");
            result.warnings.push_back("model[" + std::to_string(idx) + "] '" + cpu_plan.profile.name +
                                      "' spilled to CPU-only execution because remaining GPU VRAM was exhausted");
            model_plan = std::move(cpu_plan);
            cpu_spill_warned = true;
        }

        // Deduct GPU VRAM footprint (and first-occupancy OS reserve once per GPU).
        const std::vector<std::size_t> placed_indices = resolve_placed_gpu_indices(working_gpus, model_plan);
        const std::vector<std::uint64_t> per_gpu_mib = compute_per_gpu_vram_mib(model_plan, placed_indices);
        bool vram_overflow = false;
        for (std::size_t j = 0; j < placed_indices.size(); ++j) {
            const std::size_t gpu_idx = placed_indices[j];
            const std::uint64_t used_mib = per_gpu_mib[j];
            const std::uint64_t avail_before_mib =
                effective_usable_mib(candidate_gpus[gpu_idx], working_host, eval_req);
            if (used_mib > avail_before_mib) {
                vram_overflow = true;
            }

            const std::uint64_t newly_reserved_mib = pending_reserve_mib[gpu_idx];
            if (newly_reserved_mib > 0) {
                reserved_mib[gpu_idx] = saturating_add(reserved_mib[gpu_idx], newly_reserved_mib);
                result.notes.push_back("reserved " + std::to_string(newly_reserved_mib) +
                                       " MiB OS/desktop headroom once on " + gpu_label(working_gpus, gpu_idx));
            } else if (reserved_mib[gpu_idx] > 0 && orig_req.vram_headroom_mib > 0) {
                result.notes.push_back("reused existing " + std::to_string(reserved_mib[gpu_idx]) +
                                       " MiB OS reserve on " + gpu_label(working_gpus, gpu_idx) +
                                       " for co-resident model[" + std::to_string(idx) + "] '" +
                                       model_plan.profile.name + "'");
            }

            deduct_gpu_bytes(working_gpus[gpu_idx],
                             saturating_mul(saturating_add(newly_reserved_mib, used_mib), kMiB));
            result.aggregate_vram_mib = saturating_add(result.aggregate_vram_mib, used_mib);
        }

        // Deduct host RAM footprint (on UMA, GPU pool + host-side bytes both come
        // from unified physical RAM without double-counting either component).
        const std::uint64_t host_req_bytes =
            model_plan.uma_mode
                ? saturating_add(model_plan.estimate.total_bytes, model_plan.estimate.host_resident_bytes)
                : model_plan.estimate.host_resident_bytes;
        const std::uint64_t host_used_mib = bytes_to_mib_ceil(host_req_bytes);
        bool host_overflow = !model_plan.fits_in_host_ram;
        if (working_host.available_physical.is_known()) {
            const std::uint64_t avail_bytes = working_host.available_physical.bytes_or(0);
            if (host_req_bytes > avail_bytes) {
                host_overflow = true;
            }
            const std::uint64_t deduct_bytes = saturating_mul(host_used_mib, kMiB);
            working_host.available_physical =
                MemoryReading::known(avail_bytes > deduct_bytes ? avail_bytes - deduct_bytes : 0);
        }
        result.aggregate_host_ram_mib = saturating_add(result.aggregate_host_ram_mib, host_used_mib);

        if (model_plan.uma_mode) {
            result.notes.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                   "' uses integrated UMA (" + std::to_string(model_plan.estimate.total_mib()) +
                                   " MiB GPU pool, " + std::to_string(model_plan.estimate.host_resident_mib()) +
                                   " MiB host side, " + std::to_string(host_used_mib) +
                                   " MiB unified physical RAM)");
        }

        const bool pinned_gpu_failed = pinned_gpu && !model_plan.fits_in_vram;
        const bool gpu_assigned_failed = !placed_indices.empty() && !model_plan.fits_in_vram;
        const bool entry_fits = !vram_overflow && !host_overflow && !pinned_gpu_failed && !gpu_assigned_failed;
        if (!entry_fits) {
            all_fit = false;
            if (pinned_gpu_failed) {
                result.warnings.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                          "' could not fit pinned device_hint '" + sanitize_for_warning(orig_hint) +
                                          "' in remaining VRAM");
            } else if (vram_overflow || gpu_assigned_failed) {
                result.warnings.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                          "' exceeded remaining GPU VRAM budget");
            }
            if (host_overflow) {
                result.warnings.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                          "' exceeded remaining physical host RAM budget");
            }
        } else {
            if (model_plan.profile.n_gpu_layers.has_value() && *model_plan.profile.n_gpu_layers > 0 &&
                static_cast<std::uint32_t>(*model_plan.profile.n_gpu_layers) < entry.model.block_count) {
                result.warnings.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                          "' uses partial GPU offload (" +
                                          std::to_string(*model_plan.profile.n_gpu_layers) + "/" +
                                          std::to_string(entry.model.block_count) + " layers on GPU)");
            } else if (!cpu_spill_warned && !explicit_cpu && !gpus.empty() &&
                       model_plan.recommended_backend_hint == "cpu") {
                result.warnings.push_back("model[" + std::to_string(idx) + "] '" + model_plan.profile.name +
                                          "' spilled to CPU-only execution because remaining GPU VRAM was exhausted");
            }
        }

        result.models.push_back(std::move(model_plan));
    }

    result.all_fit_coresident = all_fit;
    result.remaining_host_ram_mib =
        working_host.available_physical.is_known() ? (working_host.available_physical.bytes_or(0) / kMiB) : 0;

    const bool prefer_free_summary =
        std::any_of(entries.begin(), entries.end(),
                    [](const CoResidencyEntry& e) noexcept { return e.request.prefer_free_vram; });
    result.remaining_gpu_vram_mib.reserve(working_gpus.size());
    for (std::size_t i = 0; i < working_gpus.size(); ++i) {
        result.remaining_gpu_vram_mib.emplace_back(
            gpu_id_or_fallback(working_gpus, i),
            remaining_gpu_summary_mib(working_gpus[i], working_host, prefer_free_summary, reserved_mib[i]));
    }

    return result;
}

json::Object coresidency_plan_to_json(const CoResidencyPlan& plan) {
    json::Object rem_gpu;
    for (const auto& [id, mib] : plan.remaining_gpu_vram_mib) {
        rem_gpu.set(id, mib);
    }

    json::Array models_arr;
    models_arr.reserve(plan.models.size());
    for (const auto& m : plan.models) {
        models_arr.emplace_back(placement_plan_to_json(m));
    }

    json::Array notes_arr;
    notes_arr.reserve(plan.notes.size());
    for (const auto& n : plan.notes) {
        notes_arr.emplace_back(n);
    }

    json::Array warnings_arr;
    warnings_arr.reserve(plan.warnings.size());
    for (const auto& w : plan.warnings) {
        warnings_arr.emplace_back(w);
    }

    json::Object out;
    out.set("schema", "sonder.inference.coresidency_plan/1");
    out.set("all_fit_coresident", plan.all_fit_coresident);
    out.set("model_count", static_cast<std::uint64_t>(plan.models.size()));
    out.set("aggregate_vram_mib", plan.aggregate_vram_mib);
    out.set("aggregate_host_ram_mib", plan.aggregate_host_ram_mib);
    out.set("remaining_gpu_vram_mib", std::move(rem_gpu));
    out.set("remaining_host_ram_mib", plan.remaining_host_ram_mib);
    out.set("models", std::move(models_arr));
    out.set("notes", std::move(notes_arr));
    out.set("warnings", std::move(warnings_arr));
    return out;
}

}  // namespace sonder::inference
