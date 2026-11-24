// Pure, deterministic hardware placement planner from GGUF metadata,
// GPU inventory, and host memory (placement_planner.hpp).
#include "sonder/inference/placement_planner.hpp"

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

#include "backends/llamaserver/env_policy.hpp"

namespace sonder::inference {

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint32_t kMaxContext = 1u << 24;
constexpr std::uint32_t kDefaultTargetContext = 32768u;
constexpr std::uint32_t kDefaultMinContext = 8192u;
constexpr std::uint32_t kCtxAlign = 4096u;
constexpr std::uint32_t kMaxParallel = 256u;
constexpr std::uint32_t kDefaultMtpDraftNMax = 3u;
constexpr std::size_t kMaxMultiGpu = 16u;
constexpr std::uint64_t kMaxVramBudgetMib = 1ull << 24;

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
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

std::string lower_copy(std::string_view sv) {
    std::string out;
    out.reserve(sv.size());
    for (const char c : sv) out.push_back(ascii_tolower(c));
    return out;
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

bool has_invalid_path_char(std::string_view sv) noexcept {
    for (const char c : sv) {
        if (c == '\0' || c == '\r' || c == '\n') return true;
    }
    return false;
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

bool is_valid_profile_name(std::string_view name) noexcept {
    if (name.empty() || name.size() > 128 || name == "default") return false;
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-' || c == ':';
        if (!ok) return false;
    }
    return true;
}

std::string sanitize_profile_name(std::string_view name) {
    if (is_valid_profile_name(name)) return std::string(name);
    std::string out;
    out.reserve(std::min<std::size_t>(name.size(), 128));
    for (const char c : name) {
        if (out.size() >= 128) break;
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-' || c == ':';
        out.push_back(ok ? c : '-');
    }
    if (!is_valid_profile_name(out)) return "planned";
    return out;
}

std::optional<std::uint32_t> model_training_context(const GgufModelInfo& model) {
    if (model.architecture.empty()) return std::nullopt;
    const std::string key = model.architecture + ".context_length";
    std::optional<std::uint32_t> found;
    for (const auto& [k, v] : model.metadata) {
        if (k == key) {
            const auto parsed = parse_strict_u64(trim_view(v));
            if (parsed && *parsed > 0) {
                found = static_cast<std::uint32_t>(std::min<std::uint64_t>(*parsed, kMaxContext));
            }
        }
    }
    return found;
}

bool is_nvidia_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::nvidia || iequals(g.backend_hint, "cuda");
}

bool is_amd_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::amd || iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
}

bool has_explicit_rocm_gpu(const std::vector<GpuDeviceInfo>& gpus) noexcept {
    return std::any_of(gpus.begin(), gpus.end(), [](const GpuDeviceInfo& g) noexcept {
        return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    });
}

bool is_rocm_target_gpu(const GpuDeviceInfo& g, bool explicit_rocm_present) noexcept {
    if (explicit_rocm_present) {
        return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    }
    return is_amd_gpu(g);
}

bool is_intel_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::intel || iequals(g.backend_hint, "sycl");
}

bool is_apple_gpu(const GpuDeviceInfo& g) noexcept {
    return g.vendor == GpuVendor::apple || iequals(g.backend_hint, "metal");
}

int vendor_priority(GpuVendor v) noexcept {
    switch (v) {
        case GpuVendor::nvidia: return 4;
        case GpuVendor::amd: return 3;
        case GpuVendor::apple: return 2;
        case GpuVendor::intel: return 1;
        case GpuVendor::unknown: return 0;
    }
    return 0;
}

struct GpuBudget {
    std::uint64_t raw_mib = 0;
    std::uint64_t reserve_mib = 0;
    std::uint64_t usable_mib = 0;
    bool uma_capped_by_host = false;
};

std::optional<GpuBudget> compute_gpu_budget(const GpuDeviceInfo& gpu, const HostMemory& host,
                                            const PlacementPlanRequest& request) noexcept {
    const auto raw = gpu_effective_vram_mib(gpu, request.prefer_free_vram);
    if (!raw || *raw == 0) return std::nullopt;

    GpuBudget b;
    b.raw_mib = *raw;
    if (!gpu.is_integrated_uma) {
        b.reserve_mib = request.vram_headroom_mib;
        b.usable_mib = b.raw_mib > b.reserve_mib ? b.raw_mib - b.reserve_mib : 0;
        return b;
    }

    // Integrated UMA: budget against effective UMA pool, never exceeding
    // available physical host RAM, with a smaller/proportional OS reserve.
    std::uint64_t pool = b.raw_mib;
    if (host.available_physical.is_known()) {
        const std::uint64_t host_avail_mib = host.available_physical.bytes_or(0) / kMiB;
        if (host_avail_mib < pool) {
            pool = host_avail_mib;
            b.uma_capped_by_host = true;
        }
    }
    if (request.vram_headroom_mib > 0 && pool > 0) {
        const std::uint64_t proportional = pool / 8;  // 12.5%
        const std::uint64_t floor_reserve = std::min<std::uint64_t>(512, pool / 4);
        b.reserve_mib = std::min(request.vram_headroom_mib, std::max(floor_reserve, proportional));
    }
    b.usable_mib = pool > b.reserve_mib ? pool - b.reserve_mib : 0;
    return b;
}

template <typename Pred>
std::optional<std::size_t> select_best_gpu(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host,
                                           const PlacementPlanRequest& request, const Pred& pred) noexcept {
    std::optional<std::size_t> best_discrete;
    std::uint64_t best_discrete_usable = 0;
    std::uint64_t best_discrete_raw = 0;
    std::optional<std::size_t> best_uma;
    std::uint64_t best_uma_usable = 0;
    std::uint64_t best_uma_raw = 0;

    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i])) continue;
        const auto budget = compute_gpu_budget(gpus[i], host, request);
        if (!budget) continue;
        if (!gpus[i].is_integrated_uma) {
            if (!best_discrete || budget->usable_mib > best_discrete_usable ||
                (budget->usable_mib == best_discrete_usable && budget->raw_mib > best_discrete_raw)) {
                best_discrete = i;
                best_discrete_usable = budget->usable_mib;
                best_discrete_raw = budget->raw_mib;
            }
        } else {
            if (!best_uma || budget->usable_mib > best_uma_usable ||
                (budget->usable_mib == best_uma_usable && budget->raw_mib > best_uma_raw)) {
                best_uma = i;
                best_uma_usable = budget->usable_mib;
                best_uma_raw = budget->raw_mib;
            }
        }
    }
    return best_discrete ? best_discrete : best_uma;
}

template <typename Pred>
std::optional<std::size_t> match_vendor_token(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host,
                                              const PlacementPlanRequest& request, std::string_view suffix,
                                              const Pred& pred, bool& indexed_out) noexcept {
    if (suffix.empty()) {
        indexed_out = false;
        return select_best_gpu(gpus, host, request, pred);
    }
    if (suffix.front() == ':') suffix.remove_prefix(1);
    const auto target_idx = parse_index(suffix);
    if (!target_idx) return std::nullopt;
    indexed_out = true;
    std::size_t seen = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i])) continue;
        if (seen == *target_idx) {
            return compute_gpu_budget(gpus[i], host, request) ? std::optional<std::size_t>(i) : std::nullopt;
        }
        ++seen;
    }
    return std::nullopt;
}

struct TokenMatch {
    std::optional<std::size_t> index;
    std::string backend_override;
    bool indexed = false;
};

TokenMatch match_single_hint(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host,
                             const PlacementPlanRequest& request, std::string_view token) {
    TokenMatch out;
    token = trim_view(token);
    if (token.empty()) return out;

    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (iequals(gpus[i].id, token)) {
            if (compute_gpu_budget(gpus[i], host, request)) {
                out.index = i;
                out.indexed = true;
            }
            return out;
        }
    }
    if (istarts_with(token, "gpu:")) {
        if (const auto idx = parse_index(token.substr(4));
            idx && *idx < gpus.size() && compute_gpu_budget(gpus[*idx], host, request)) {
            out.index = *idx;
            out.indexed = true;
        }
        return out;
    }
    if (const auto idx = parse_index(token);
        idx && *idx < gpus.size() && compute_gpu_budget(gpus[*idx], host, request)) {
        out.index = *idx;
        out.indexed = true;
        return out;
    }

    const auto is_any = [](const GpuDeviceInfo&) noexcept { return true; };

    if (istarts_with(token, "cuda")) {
        out.backend_override = "cuda";
        out.index = match_vendor_token(gpus, host, request, token.substr(4), is_nvidia_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "nvidia")) {
        out.index = match_vendor_token(gpus, host, request, token.substr(6), is_nvidia_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "rocm")) {
        out.backend_override = "rocm";
        const bool explicit_rocm = has_explicit_rocm_gpu(gpus);
        out.index = match_vendor_token(
            gpus, host, request, token.substr(4),
            [explicit_rocm](const GpuDeviceInfo& g) noexcept { return is_rocm_target_gpu(g, explicit_rocm); },
            out.indexed);
        return out;
    }
    if (istarts_with(token, "hip")) {
        out.backend_override = "rocm";
        const bool explicit_rocm = has_explicit_rocm_gpu(gpus);
        out.index = match_vendor_token(
            gpus, host, request, token.substr(3),
            [explicit_rocm](const GpuDeviceInfo& g) noexcept { return is_rocm_target_gpu(g, explicit_rocm); },
            out.indexed);
        return out;
    }
    if (istarts_with(token, "amd")) {
        out.index = match_vendor_token(gpus, host, request, token.substr(3), is_amd_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "vulkan")) {
        out.backend_override = "vulkan";
        out.index = match_vendor_token(gpus, host, request, token.substr(6), is_any, out.indexed);
        return out;
    }
    if (istarts_with(token, "vk")) {
        out.backend_override = "vulkan";
        out.index = match_vendor_token(gpus, host, request, token.substr(2), is_any, out.indexed);
        return out;
    }
    if (istarts_with(token, "intel")) {
        out.index = match_vendor_token(gpus, host, request, token.substr(5), is_intel_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "sycl")) {
        out.backend_override = "sycl";
        out.index = match_vendor_token(gpus, host, request, token.substr(4), is_intel_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "metal")) {
        out.backend_override = "metal";
        out.index = match_vendor_token(gpus, host, request, token.substr(5), is_apple_gpu, out.indexed);
        return out;
    }
    if (istarts_with(token, "apple")) {
        out.index = match_vendor_token(gpus, host, request, token.substr(5), is_apple_gpu, out.indexed);
        return out;
    }
    return out;
}

std::vector<std::size_t> collect_same_vendor_discrete_peers(const std::vector<GpuDeviceInfo>& gpus,
                                                            const HostMemory& host,
                                                            const PlacementPlanRequest& request,
                                                            std::size_t primary_idx) {
    std::vector<std::size_t> peers;
    if (primary_idx >= gpus.size() || gpus[primary_idx].is_integrated_uma) return peers;
    const GpuDeviceInfo& primary = gpus[primary_idx];
    const bool primary_is_rocm = iequals(primary.backend_hint, "rocm") || iequals(primary.backend_hint, "hip");
    for (std::size_t i = 0; i < gpus.size() && peers.size() < kMaxMultiGpu; ++i) {
        if (gpus[i].is_integrated_uma) continue;
        const bool same_vendor = primary.vendor != GpuVendor::unknown
                                     ? gpus[i].vendor == primary.vendor
                                     : (gpus[i].vendor == GpuVendor::unknown &&
                                        iequals(gpus[i].backend_hint, primary.backend_hint));
        if (!same_vendor) continue;
        if (primary_is_rocm && iequals(gpus[i].backend_hint, "vulkan")) continue;
        const auto budget = compute_gpu_budget(gpus[i], host, request);
        if (!budget || budget->usable_mib == 0) continue;
        peers.push_back(i);
    }
    return peers;
}

std::vector<std::size_t> select_best_multi_gpu_group(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host,
                                                     const PlacementPlanRequest& request) {
    std::vector<std::size_t> best_group;
    std::uint64_t best_total_usable = 0;
    int best_prio = -1;

    constexpr GpuVendor kVendors[] = {GpuVendor::nvidia, GpuVendor::amd, GpuVendor::apple, GpuVendor::intel,
                                      GpuVendor::unknown};
    for (const GpuVendor v : kVendors) {
        std::vector<std::size_t> group;
        std::uint64_t total_usable = 0;
        bool has_rocm_peer = false;
        if (v == GpuVendor::amd) {
            for (const auto& g : gpus) {
                if (!g.is_integrated_uma && g.vendor == GpuVendor::amd &&
                    (iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip"))) {
                    has_rocm_peer = true;
                    break;
                }
            }
        }
        for (std::size_t i = 0; i < gpus.size() && group.size() < kMaxMultiGpu; ++i) {
            if (gpus[i].is_integrated_uma || gpus[i].vendor != v) continue;
            if (has_rocm_peer && iequals(gpus[i].backend_hint, "vulkan")) continue;
            const auto budget = compute_gpu_budget(gpus[i], host, request);
            if (!budget || budget->usable_mib == 0) continue;
            group.push_back(i);
            total_usable = saturating_add(total_usable, budget->usable_mib);
        }
        if (group.size() < 2) continue;
        const int prio = vendor_priority(v);
        if (best_group.empty() || total_usable > best_total_usable ||
            (total_usable == best_total_usable && prio > best_prio)) {
            best_group = std::move(group);
            best_total_usable = total_usable;
            best_prio = prio;
        }
    }
    return best_group;
}

struct DeviceSelection {
    std::vector<std::size_t> indices;
    std::string backend_override;
    bool explicit_cpu = false;
    bool explicit_multi = false;
    bool multi_unavailable = false;
    bool pinned_single = false;
    bool hint_unmatched = false;
};

DeviceSelection resolve_device_selection(const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host,
                                         const PlacementPlanRequest& request) {
    DeviceSelection sel;
    const std::string_view hint = trim_view(request.device_hint);
    if (hint.empty()) {
        if (const auto best = select_best_gpu(gpus, host, request, [](const GpuDeviceInfo&) noexcept { return true; })) {
            sel.indices.push_back(*best);
        }
        return sel;
    }
    if (iequals(hint, "cpu")) {
        sel.explicit_cpu = true;
        return sel;
    }
    if (iequals(hint, "multi")) {
        sel.explicit_multi = true;
        sel.indices = select_best_multi_gpu_group(gpus, host, request);
        if (sel.indices.size() < 2) {
            sel.multi_unavailable = true;
            sel.indices.clear();
            if (const auto best =
                    select_best_gpu(gpus, host, request, [](const GpuDeviceInfo&) noexcept { return true; })) {
                sel.indices.push_back(*best);
            }
        }
        return sel;
    }
    if (iequals(hint, "uma") || iequals(hint, "igpu") || iequals(hint, "apu")) {
        const auto best_uma =
            select_best_gpu(gpus, host, request, [](const GpuDeviceInfo& g) noexcept { return g.is_integrated_uma; });
        if (best_uma) {
            sel.indices.push_back(*best_uma);
            sel.pinned_single = true;
        } else {
            sel.hint_unmatched = true;
        }
        return sel;
    }

    bool any_indexed = false;
    std::size_t pos = 0;
    while (pos <= hint.size()) {
        const std::size_t comma = hint.find(',', pos);
        const std::string_view token =
            trim_view(hint.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? hint.size() + 1 : comma + 1;
        if (token.empty()) continue;

        const TokenMatch m = match_single_hint(gpus, host, request, token);
        if (!m.index) {
            sel.indices.clear();
            sel.hint_unmatched = true;
            return sel;
        }
        if (sel.backend_override.empty() && !m.backend_override.empty()) {
            sel.backend_override = m.backend_override;
        }
        any_indexed = any_indexed || m.indexed;
        if (std::find(sel.indices.begin(), sel.indices.end(), *m.index) == sel.indices.end() &&
            sel.indices.size() < kMaxMultiGpu) {
            sel.indices.push_back(*m.index);
        }
    }
    if (sel.indices.empty()) {
        sel.hint_unmatched = true;
    } else if (sel.indices.size() >= 2) {
        sel.explicit_multi = true;
    } else {
        sel.pinned_single = any_indexed;
    }
    return sel;
}

std::string resolve_backend_hint(const GpuDeviceInfo& gpu, const std::string& backend_override) {
    if (!backend_override.empty()) return backend_override;
    if (!gpu.backend_hint.empty()) {
        const std::string lower = lower_copy(trim_view(gpu.backend_hint));
        if (lower == "hip") return "rocm";
        if (lower == "vk") return "vulkan";
        if (!lower.empty()) return lower;
    }
    switch (gpu.vendor) {
        case GpuVendor::nvidia: return "cuda";
        case GpuVendor::amd: return gpu.is_integrated_uma ? "vulkan" : "rocm";
        case GpuVendor::intel: return "vulkan";
        case GpuVendor::apple: return "metal";
        case GpuVendor::unknown: return "vulkan";
    }
    return "vulkan";
}

std::string ggml_device_name_for(const std::vector<GpuDeviceInfo>& gpus, std::size_t idx,
                                 std::string_view backend_hint) {
    if (backend_hint == "cuda") {
        std::size_t local = 0;
        for (std::size_t i = 0; i < idx && i < gpus.size(); ++i) {
            if (is_nvidia_gpu(gpus[i])) ++local;
        }
        return "CUDA" + std::to_string(local);
    }
    if (backend_hint == "rocm") {
        const bool explicit_rocm = has_explicit_rocm_gpu(gpus);
        std::size_t local = 0;
        for (std::size_t i = 0; i < idx && i < gpus.size(); ++i) {
            if (is_rocm_target_gpu(gpus[i], explicit_rocm)) ++local;
        }
        return "ROCm" + std::to_string(local);
    }
    if (backend_hint == "sycl") {
        std::size_t local = 0;
        for (std::size_t i = 0; i < idx && i < gpus.size(); ++i) {
            if (is_intel_gpu(gpus[i])) ++local;
        }
        return "SYCL" + std::to_string(local);
    }
    if (backend_hint == "metal") {
        std::size_t local = 0;
        for (std::size_t i = 0; i < idx && i < gpus.size(); ++i) {
            if (is_apple_gpu(gpus[i])) ++local;
        }
        return "Metal" + std::to_string(local);
    }
    return "Vulkan" + std::to_string(idx);
}

std::string join_device_names(const std::vector<GpuDeviceInfo>& gpus, const std::vector<std::size_t>& indices,
                              std::string_view backend_hint) {
    std::string out;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i > 0) out.push_back(',');
        out += ggml_device_name_for(gpus, indices[i], backend_hint);
    }
    return out;
}

std::vector<std::string> strip_n_cpu_moe_args(const std::vector<std::string>& args) {
    std::vector<std::string> out;
    out.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string flag = llamaserver_flag_name(args[i]);
        if (flag == "--n-cpu-moe" || flag == "-ncmoe") {
            if (args[i].find('=') == std::string::npos && i + 1 < args.size()) {
                ++i;
            }
            continue;
        }
        out.push_back(args[i]);
    }
    return out;
}

LaunchProfile with_n_cpu_moe(const LaunchProfile& base, std::uint32_t n) {
    LaunchProfile next = base;
    next.extra_args = strip_n_cpu_moe_args(base.extra_args);
    if (n > 0) {
        next.extra_args.emplace_back("--n-cpu-moe");
        next.extra_args.push_back(std::to_string(n));
    }
    return next;
}

bool has_usable_expert_tensors(const GgufModelInfo& model) noexcept {
    if (model.tensors.empty() || model.tensors_truncated || model.block_count == 0) return false;
    return std::any_of(model.tensors.begin(), model.tensors.end(), [&model](const GgufTensorInfo& t) noexcept {
        return t.expert && t.block != kNoBlock && t.block < model.block_count && t.bytes > 0;
    });
}

void add_allowed_env(std::vector<std::pair<std::string, std::string>>& env, std::string key, std::string value) {
    if (!llamaserver::is_allowed_profile_env_override(key)) return;
    for (const auto& entry : env) {
        if (iequals(entry.first, key)) return;
    }
    std::vector<std::pair<std::string, std::string>> candidate = env;
    candidate.emplace_back(std::move(key), std::move(value));
    if (llamaserver::check_child_environment_overrides(candidate).ok()) {
        env = std::move(candidate);
    }
}

// Builds the ordered grid of candidate contexts strictly below `target_ctx`,
// starting at `min_ctx` and aligned to `kCtxAlign` (4096) above it.
std::vector<std::uint32_t> context_stepdown_candidates(std::uint32_t min_ctx, std::uint32_t target_ctx) {
    std::vector<std::uint32_t> candidates;
    if (min_ctx >= target_ctx) return candidates;
    candidates.push_back(min_ctx);
    const std::uint32_t first_aligned = (min_ctx / kCtxAlign + 1u) * kCtxAlign;
    for (std::uint32_t c = first_aligned; c < target_ctx; c += kCtxAlign) {
        candidates.push_back(c);
        if (target_ctx - c <= kCtxAlign) break;
    }
    return candidates;
}

std::string format_mib_list(const std::vector<std::uint64_t>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(values[i]);
    }
    return out;
}

}  // namespace

PlacementPlan plan_model_placement(const GgufModelInfo& model, const std::vector<GpuDeviceInfo>& gpus,
                                   const HostMemory& host, const PlacementPlanRequest& request) {
    PlacementPlan plan;

    // --------------------------------------------------------- 1. Base profile
    LaunchProfile profile;
    profile.name = sanitize_profile_name(request.profile_name);
    if (profile.name != request.profile_name) {
        plan.warnings.push_back("profile_name was adjusted to '" + profile.name +
                                "' to satisfy LaunchProfile naming rules");
    }
    profile.backend = kLaunchProfileBackendLlamaServer;
    if (!request.model_path.empty() && !has_invalid_path_char(request.model_path)) {
        profile.model = request.model_path;
    } else {
        profile.model = "model.gguf";
        plan.warnings.emplace_back(
            "model_path was empty or contained NUL/CR/LF; using placeholder 'model.gguf' for LaunchProfile validation");
    }
    if (!request.mmproj_path.empty()) {
        if (!has_invalid_path_char(request.mmproj_path)) {
            profile.mmproj = request.mmproj_path;
        } else {
            plan.warnings.emplace_back("mmproj_path contained NUL/CR/LF and was ignored");
        }
    }

    const std::uint32_t parallel = std::clamp<std::uint32_t>(request.parallel == 0 ? 1u : request.parallel, 1u, kMaxParallel);
    profile.parallel = parallel;
    profile.kv_unified = true;
    profile.fit = false;

    // Context clamping against model training context when known.
    std::uint32_t target_ctx =
        std::min<std::uint32_t>(request.target_ctx_size > 0 ? request.target_ctx_size : kDefaultTargetContext, kMaxContext);
    if (const auto train_ctx = model_training_context(model)) {
        if (target_ctx > *train_ctx) {
            plan.reasons.push_back("clamped target_ctx_size from " + std::to_string(target_ctx) +
                                   " to model training context " + std::to_string(*train_ctx));
            target_ctx = *train_ctx;
        }
    }
    std::uint32_t min_ctx =
        std::min<std::uint32_t>(request.min_ctx_size > 0 ? request.min_ctx_size : kDefaultMinContext, kMaxContext);
    if (min_ctx > target_ctx) min_ctx = target_ctx;
    profile.ctx_size = target_ctx;

    // ------------------------------------- 2. PrismML & speculative detection
    if (model.has_prism_private_tensors || model.prism_hadamard) {
        plan.requires_prism_runtime = true;
        std::string warn =
            "model uses PrismML Hadamard / private ternary tensors (PQ2_0/PTQ1_0";
        if (!model.prism_hadamard_transform.empty()) {
            warn += ", transform=" + model.prism_hadamard_transform;
        }
        if (model.prism_hadamard_block_size > 0) {
            warn += ", block_size=" + std::to_string(model.prism_hadamard_block_size);
        }
        warn += "); requires PrismML's Hadamard-capable llama-server (stock llama-server cannot load it)";
        plan.warnings.push_back(std::move(warn));
    }

    if (request.allow_speculative) {
        if (model.nextn_layers > 0) {
            SpeculativeSettings spec;
            spec.types = {"draft-mtp"};
            spec.draft_n_max = kDefaultMtpDraftNMax;
            profile.speculative = std::move(spec);
            plan.reasons.push_back("enabled speculative decoding draft-mtp (draft_n_max=" +
                                   std::to_string(kDefaultMtpDraftNMax) + ") for model with " +
                                   std::to_string(model.nextn_layers) + " MTP nextn layer(s)");
        } else if (model.has_ternary_tensors || model.has_one_bit_tensors) {
            SpeculativeSettings spec;
            spec.types = {"ngram-mod"};
            profile.speculative = std::move(spec);
            plan.reasons.emplace_back(
                "enabled lightweight speculative decoding ngram-mod for ternary/1-bit model");
        }
    }

    // Initial KV cache type.
    if (request.allow_quantized_kv) {
        profile.cache_type_k = "q8_0";
        profile.cache_type_v = "q8_0";
    } else {
        profile.cache_type_k = "f16";
        profile.cache_type_v = "f16";
    }

    // --------------------------------------------------- 3. Device selection
    const DeviceSelection sel = resolve_device_selection(gpus, host, request);
    if (sel.hint_unmatched) {
        plan.warnings.push_back("device_hint '" + sanitize_for_warning(request.device_hint) +
                                "' matched no usable GPU; falling back to CPU-only execution");
    }
    if (sel.multi_unavailable) {
        plan.warnings.emplace_back(
            "device_hint 'multi' requested but fewer than 2 usable discrete GPUs of the same vendor were found");
    }

    const auto evaluate = [&](const LaunchProfile& p, std::optional<std::uint64_t> per_gpu_budget_mib) -> VramEstimate {
        VramEstimateInputs inputs{};
        inputs.mmproj_bytes = !p.mmproj.empty() ? request.mmproj_bytes : 0;
        if (per_gpu_budget_mib && *per_gpu_budget_mib > 0) {
            inputs.fit_vram_budget_bytes =
                *per_gpu_budget_mib > (std::numeric_limits<std::uint64_t>::max() >> 20)
                    ? std::numeric_limits<std::uint64_t>::max()
                    : (*per_gpu_budget_mib << 20);
        }
        return estimate_vram(model, p, inputs);
    };

    const auto check_host_fit = [&](const VramEstimate& e, bool is_uma) -> bool {
        if (!host.available_physical.is_known()) return true;
        const std::uint64_t avail = host.available_physical.bytes_or(0);
        const std::uint64_t required =
            is_uma ? saturating_add(e.total_bytes, e.host_resident_bytes) : e.host_resident_bytes;
        return required <= avail;
    };

    // ------------------------------------------------- 4. CPU-only fallback
    if (sel.explicit_cpu || sel.indices.empty()) {
        plan.selected_vendor = GpuVendor::unknown;
        plan.recommended_backend_hint = "cpu";
        plan.uma_mode = false;
        plan.fits_in_vram = false;

        profile.n_gpu_layers = 0;
        if (!profile.mmproj.empty()) {
            profile.mmproj_offload = false;
        }
        profile.device = std::nullopt;
        profile.split_mode = std::nullopt;
        profile.tensor_split.clear();
        profile.vram_budget_mib = std::nullopt;
        profile.flash_attn = "auto";
        profile.load_mode = "auto";

        if (sel.explicit_cpu) {
            plan.reasons.emplace_back("device_hint 'cpu' selected CPU-only execution (n_gpu_layers=0)");
        } else {
            plan.reasons.emplace_back("no usable GPU available; configured CPU-only execution (n_gpu_layers=0)");
        }

        VramEstimate est = evaluate(profile, std::nullopt);
        if (host.available_physical.is_known()) {
            const std::uint64_t avail_bytes = host.available_physical.bytes_or(0);
            plan.budget_mib = avail_bytes / kMiB;
            plan.reasons.push_back("sized CPU-only execution against " + std::to_string(plan.budget_mib) +
                                   " MiB available physical host RAM");

            if (est.host_resident_bytes > avail_bytes && request.allow_quantized_kv &&
                !(profile.cache_type_k && *profile.cache_type_k == "q4_0")) {
                LaunchProfile q4 = profile;
                q4.cache_type_k = "q4_0";
                q4.cache_type_v = "q4_0";
                const VramEstimate est_q4 = evaluate(q4, std::nullopt);
                if (est_q4.host_kv_bytes < est.host_kv_bytes) {
                    profile = std::move(q4);
                    est = est_q4;
                    plan.reasons.push_back("stepped host KV cache to q4_0/q4_0 (host resident " +
                                           std::to_string(est.host_resident_mib()) + " MiB)");
                }
            }
            if (est.host_resident_bytes > avail_bytes && !profile.no_repack.value_or(false)) {
                LaunchProfile nr = profile;
                nr.no_repack = true;
                const VramEstimate est_nr = evaluate(nr, std::nullopt);
                if (est_nr.host_resident_bytes < est.host_resident_bytes) {
                    profile = std::move(nr);
                    est = est_nr;
                    plan.reasons.push_back("enabled no_repack to reduce private host RAM to " +
                                           std::to_string(est.host_resident_mib()) + " MiB");
                }
            }
            if (est.host_resident_bytes > avail_bytes && *profile.ctx_size > min_ctx) {
                const auto candidates = context_stepdown_candidates(min_ctx, *profile.ctx_size);
                if (!candidates.empty()) {
                    LaunchProfile try_min = profile;
                    try_min.ctx_size = candidates.front();
                    const VramEstimate est_min = evaluate(try_min, std::nullopt);
                    if (est_min.host_resident_bytes <= avail_bytes) {
                        std::size_t lo = 0;
                        std::size_t hi = candidates.size() - 1;
                        std::size_t best_idx = 0;
                        VramEstimate best_est = est_min;
                        while (lo <= hi) {
                            const std::size_t mid = lo + (hi - lo) / 2;
                            LaunchProfile probe = profile;
                            probe.ctx_size = candidates[mid];
                            const VramEstimate e = evaluate(probe, std::nullopt);
                            if (e.host_resident_bytes <= avail_bytes) {
                                best_idx = mid;
                                best_est = e;
                                lo = mid + 1;
                            } else {
                                if (mid == 0) break;
                                hi = mid - 1;
                            }
                        }
                        plan.reasons.push_back("stepped CPU ctx_size from " + std::to_string(*profile.ctx_size) +
                                               " to " + std::to_string(candidates[best_idx]) +
                                               " to fit available host RAM");
                        profile.ctx_size = candidates[best_idx];
                        est = best_est;
                    } else {
                        plan.reasons.push_back("stepped CPU ctx_size down to min_ctx_size (" +
                                               std::to_string(min_ctx) + ")");
                        profile.ctx_size = min_ctx;
                        est = est_min;
                    }
                }
            }
            plan.fits_in_host_ram = est.host_resident_bytes <= avail_bytes;
            if (!plan.fits_in_host_ram) {
                plan.warnings.push_back("estimated host RAM (" + std::to_string(est.host_resident_mib()) +
                                        " MiB) exceeds available physical RAM (" + std::to_string(plan.budget_mib) +
                                        " MiB)");
            }
        } else {
            plan.budget_mib = 0;
            plan.fits_in_host_ram = true;
            plan.warnings.emplace_back("host available physical RAM is unknown; CPU-only fit could not be verified");
        }

        if (profile.cache_type_v && *profile.cache_type_v == "q4_0") {
            plan.warnings.emplace_back(
                "4-bit V cache (q4_0) enabled to fit memory budget; verify generation quality on this model");
        }

        plan.profile = std::move(profile);
        plan.estimate = evaluate(plan.profile, std::nullopt);
        return plan;
    }

    // ----------------------------------------- 5. GPU vendor & budget setup
    std::vector<std::size_t> active_indices = sel.indices;
    const GpuDeviceInfo& primary_gpu = gpus[active_indices.front()];
    plan.selected_vendor = primary_gpu.vendor;
    plan.uma_mode = active_indices.size() == 1 && primary_gpu.is_integrated_uma;
    plan.recommended_backend_hint = resolve_backend_hint(primary_gpu, sel.backend_override);
    if (active_indices.size() >= 2 && plan.recommended_backend_hint != "vulkan") {
        bool mixed_vendors = false;
        for (std::size_t k = 1; k < active_indices.size(); ++k) {
            if (gpus[active_indices[k]].vendor != primary_gpu.vendor) {
                mixed_vendors = true;
                break;
            }
        }
        if (mixed_vendors) {
            plan.recommended_backend_hint = "vulkan";
            plan.warnings.emplace_back(
                "selected GPUs span multiple vendors; using Vulkan multi-GPU device naming");
        }
    }

    if (plan.uma_mode) {
        profile.flash_attn = "auto";
        profile.load_mode = "auto";
        plan.warnings.emplace_back(
            "integrated UMA GPU shares system memory bandwidth; long contexts and large weights may be "
            "memory-bandwidth bound");
        if (plan.selected_vendor == GpuVendor::amd && plan.recommended_backend_hint == "rocm") {
            add_allowed_env(plan.recommended_env, "GPU_MAX_ALLOC_PERCENT", "100");
            plan.reasons.emplace_back("suggested GPU_MAX_ALLOC_PERCENT=100 for AMD ROCm/HIP UMA GTT allocation");
        }
    } else if (plan.selected_vendor == GpuVendor::nvidia) {
        profile.flash_attn = "on";
        profile.load_mode = "none";
        plan.reasons.emplace_back(
            "NVIDIA RTX defaults: flash_attn=on, kv_unified=true, load_mode=none for resident VRAM weights");
    } else if (plan.selected_vendor == GpuVendor::amd) {
        profile.flash_attn = "auto";
        profile.load_mode = "auto";
        add_allowed_env(plan.recommended_env, "GPU_MAX_ALLOC_PERCENT", "100");
        plan.reasons.emplace_back(
            "AMD Radeon RX discrete defaults: flash_attn=auto, kv_unified=true, GPU_MAX_ALLOC_PERCENT=100 for "
            "ROCm/HIP full VRAM allocation");
    } else {
        profile.flash_attn = "auto";
        profile.load_mode = "auto";
        plan.reasons.push_back(std::string("GPU vendor ") + to_string(plan.selected_vendor) +
                               " defaults: flash_attn=auto, kv_unified=true");
    }

    profile.n_gpu_layers = -1;

    std::vector<std::uint64_t> active_usable_mib;
    std::vector<std::uint64_t> active_raw_mib;
    std::uint64_t total_usable_mib = 0;
    std::uint64_t max_single_usable_mib = 0;
    bool any_uma_capped_by_host = false;

    const auto apply_active_devices = [&](const std::vector<std::size_t>& indices) {
        active_indices = indices;
        active_usable_mib.clear();
        active_raw_mib.clear();
        total_usable_mib = 0;
        max_single_usable_mib = 0;
        any_uma_capped_by_host = false;

        std::size_t largest_pos = 0;
        for (std::size_t k = 0; k < active_indices.size(); ++k) {
            const auto b = compute_gpu_budget(gpus[active_indices[k]], host, request);
            const std::uint64_t usable = b ? b->usable_mib : 0;
            const std::uint64_t raw = b ? b->raw_mib : 0;
            if (b && b->uma_capped_by_host) any_uma_capped_by_host = true;
            active_usable_mib.push_back(usable);
            active_raw_mib.push_back(raw);
            total_usable_mib = saturating_add(total_usable_mib, usable);
            if (usable > max_single_usable_mib) {
                max_single_usable_mib = usable;
                largest_pos = k;
            }
        }

        profile.device = join_device_names(gpus, active_indices, plan.recommended_backend_hint);
        if (active_indices.size() >= 2 && total_usable_mib > 0) {
            profile.split_mode = "layer";
            profile.tensor_split.clear();
            profile.tensor_split.reserve(active_indices.size());
            for (const std::uint64_t usable : active_usable_mib) {
                const double w = static_cast<double>(usable) / static_cast<double>(total_usable_mib);
                profile.tensor_split.push_back(w > 0.0 ? w : 1e-6);
            }
            if (largest_pos > 0) {
                profile.main_gpu = static_cast<std::int32_t>(largest_pos);
            } else {
                profile.main_gpu = std::nullopt;
            }
            plan.budget_mib = total_usable_mib;
            profile.vram_budget_mib =
                max_single_usable_mib > 0
                    ? std::optional<std::uint64_t>(std::min(max_single_usable_mib, kMaxVramBudgetMib))
                    : std::nullopt;
        } else {
            profile.split_mode = std::nullopt;
            profile.tensor_split.clear();
            profile.main_gpu = std::nullopt;
            plan.budget_mib = total_usable_mib;
            profile.vram_budget_mib =
                total_usable_mib > 0 ? std::optional<std::uint64_t>(std::min(total_usable_mib, kMaxVramBudgetMib))
                                     : std::nullopt;
        }
    };

    const auto fits_vram = [&](const VramEstimate& e, const LaunchProfile& p) -> bool {
        if (active_usable_mib.empty() || total_usable_mib == 0) return false;
        if (e.total_bytes == 0) return true;
        if (active_usable_mib.size() == 1 || p.tensor_split.size() < 2 ||
            (p.split_mode && *p.split_mode == "none")) {
            return e.total_mib() <= active_usable_mib.front();
        }
        double weight_sum = 0.0;
        for (const double w : p.tensor_split) {
            if (std::isfinite(w) && w > 0.0) weight_sum += w;
        }
        if (!(weight_sum > 0.0)) return false;
        const std::uint64_t splittable =
            saturating_add(e.weights_bytes, saturating_add(e.kv_bytes, e.recurrent_bytes));
        const std::size_t n = std::min(active_usable_mib.size(), p.tensor_split.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double share = p.tensor_split[i] / weight_sum;
            const std::uint64_t shard =
                std::min(splittable, saturating_u64(std::ceil(static_cast<double>(splittable) * share)));
            const std::uint64_t gpu_bytes = saturating_add(shard, e.compute_bytes);
            if (bytes_to_mib_ceil(gpu_bytes) > active_usable_mib[i]) return false;
        }
        return e.max_gpu_mib() <= max_single_usable_mib;
    };

    apply_active_devices(active_indices);

    if (active_indices.size() == 1) {
        const auto single_budget = compute_gpu_budget(gpus[active_indices[0]], host, request);
        const std::uint64_t reserve_mib = single_budget ? single_budget->reserve_mib : 0;
        const std::uint64_t effective_pool_mib =
            any_uma_capped_by_host ? saturating_add(active_usable_mib[0], reserve_mib) : active_raw_mib[0];
        std::string reason = "selected " + gpus[active_indices[0]].name + " (" + *profile.device +
                             (plan.uma_mode ? ", integrated UMA" : ", discrete") + "): effective pool " +
                             std::to_string(effective_pool_mib) + " MiB minus " +
                             std::to_string(reserve_mib) + " MiB reserve = " +
                             std::to_string(active_usable_mib[0]) + " MiB budget";
        if (any_uma_capped_by_host) {
            reason += " (capped by available host RAM from " + std::to_string(active_raw_mib[0]) + " MiB)";
        }
        plan.reasons.push_back(std::move(reason));
    } else {
        plan.reasons.push_back("selected " + std::to_string(active_indices.size()) + " GPUs (" + *profile.device +
                               ") with split_mode=layer, total usable VRAM " + std::to_string(total_usable_mib) +
                               " MiB (per-GPU budgets: " + format_mib_list(active_usable_mib) + " MiB)");
    }

    // ----------------------------------------- 6. Auto multi-GPU expansion
    VramEstimate est = evaluate(profile, profile.vram_budget_mib);
    if (active_indices.size() == 1 && !sel.pinned_single && !plan.uma_mode && !fits_vram(est, profile)) {
        const auto peers = collect_same_vendor_discrete_peers(gpus, host, request, active_indices[0]);
        if (peers.size() >= 2) {
            const std::uint64_t single_usable = active_usable_mib[0];
            const std::uint64_t single_needed = est.total_mib();
            apply_active_devices(peers);
            est = evaluate(profile, profile.vram_budget_mib);
            plan.reasons.push_back("single-GPU budget (" + std::to_string(single_usable) +
                                   " MiB) could not fit initial estimate (" + std::to_string(single_needed) +
                                   " MiB); expanded to " + std::to_string(active_indices.size()) +
                                   "-GPU layer split (" + *profile.device + ", total usable " +
                                   std::to_string(total_usable_mib) + " MiB, per-GPU budgets: " +
                                   format_mib_list(active_usable_mib) + " MiB)");
        }
    }

    // ----------------------------------------- 7. Iterative fit solver
    bool moe_offload_active = false;
    bool stepped_kv_to_q4 = false;

    // Step 1: quantize KV cache to q4_0/q4_0.
    if (!fits_vram(est, profile) && request.allow_quantized_kv &&
        !(profile.cache_type_k && *profile.cache_type_k == "q4_0")) {
        LaunchProfile q4 = profile;
        q4.cache_type_k = "q4_0";
        q4.cache_type_v = "q4_0";
        const VramEstimate est_q4 = evaluate(q4, q4.vram_budget_mib);
        if (est_q4.kv_bytes < est.kv_bytes) {
            profile = std::move(q4);
            est = est_q4;
            stepped_kv_to_q4 = true;
            plan.reasons.push_back("stepped KV cache from q8_0/q8_0 to q4_0/q4_0 (estimated VRAM " +
                                   std::to_string(est.total_mib()) + " MiB)");
        }
    }

    // Step 2: selective MoE expert CPU offload via --n-cpu-moe N.
    const bool model_is_moe = model.routed_experts.has_value() && model.routed_experts->total > 0;
    if (!fits_vram(est, profile) && request.allow_moe_cpu_offload && model_is_moe && model.block_count > 0) {
        if (!has_usable_expert_tensors(model)) {
            plan.warnings.emplace_back(
                "model declares routed experts but GGUF tensor table is empty or truncated; cannot search --n-cpu-moe");
        } else {
            LaunchProfile moe_base = profile;
            if (moe_base.load_mode && *moe_base.load_mode == "none") {
                moe_base.load_mode = "auto";
            }
            const LaunchProfile all_cpu_moe = with_n_cpu_moe(moe_base, model.block_count);
            const VramEstimate est_all = evaluate(all_cpu_moe, all_cpu_moe.vram_budget_mib);
            if (est_all.total_bytes < est.total_bytes) {
                if (fits_vram(est_all, all_cpu_moe)) {
                    std::uint32_t lo = 1;
                    std::uint32_t hi = model.block_count;
                    std::uint32_t best_n = model.block_count;
                    VramEstimate best_est = est_all;
                    while (lo <= hi) {
                        const std::uint32_t mid = lo + (hi - lo) / 2;
                        const LaunchProfile probe = with_n_cpu_moe(moe_base, mid);
                        const VramEstimate e = evaluate(probe, probe.vram_budget_mib);
                        if (fits_vram(e, probe)) {
                            best_n = mid;
                            best_est = e;
                            if (mid == 1) break;
                            hi = mid - 1;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    profile = with_n_cpu_moe(moe_base, best_n);
                    est = best_est;
                    moe_offload_active = true;

                    // If we stepped KV to q4_0 earlier and q8_0 fits at the exact same N, restore q8_0 quality.
                    if (stepped_kv_to_q4) {
                        LaunchProfile try_q8 = profile;
                        try_q8.cache_type_k = "q8_0";
                        try_q8.cache_type_v = "q8_0";
                        const VramEstimate est_q8 = evaluate(try_q8, try_q8.vram_budget_mib);
                        if (fits_vram(est_q8, try_q8)) {
                            profile = std::move(try_q8);
                            est = est_q8;
                            stepped_kv_to_q4 = false;
                            plan.reasons.emplace_back(
                                "restored q8_0/q8_0 KV cache with sufficient headroom after --n-cpu-moe offload");
                        }
                    }

                    plan.reasons.push_back("offloaded MoE routed experts for first " + std::to_string(best_n) + "/" +
                                           std::to_string(model.block_count) + " blocks to host RAM via --n-cpu-moe " +
                                           std::to_string(best_n) + " (VRAM " + std::to_string(est.total_mib()) +
                                           " MiB, host resident " + std::to_string(est.host_resident_mib()) + " MiB)");
                } else {
                    // Keep all experts on CPU so ctx step-down / partial layer offload operates on dense weights.
                    profile = all_cpu_moe;
                    est = est_all;
                    moe_offload_active = true;
                    plan.reasons.push_back("offloaded all " + std::to_string(model.block_count) +
                                           " MoE expert blocks to host RAM via --n-cpu-moe " +
                                           std::to_string(model.block_count) + " (VRAM reduced to " +
                                           std::to_string(est.total_mib()) + " MiB; continuing fit)");
                }
            }
        }
    }

    // Step 2b: if MTP rollback snapshots are inflating recurrent VRAM on a tight hybrid model,
    // step draft_n_max down before shrinking context below target_ctx.
    if (!fits_vram(est, profile) && profile.speculative && est.recurrent_snapshots > 1 &&
        profile.speculative->draft_n_max && *profile.speculative->draft_n_max > 1) {
        const std::uint32_t orig_n = *profile.speculative->draft_n_max;
        for (std::uint32_t n = orig_n - 1; n >= 1; --n) {
            LaunchProfile probe = profile;
            probe.speculative->draft_n_max = n;
            const VramEstimate e = evaluate(probe, probe.vram_budget_mib);
            if (e.recurrent_bytes < est.recurrent_bytes) {
                profile = std::move(probe);
                est = e;
                if (fits_vram(est, profile)) {
                    break;
                }
            }
            if (n == 1) break;
        }
        if (*profile.speculative->draft_n_max < orig_n) {
            plan.reasons.push_back("reduced speculative draft_n_max from " + std::to_string(orig_n) + " to " +
                                   std::to_string(*profile.speculative->draft_n_max) +
                                   " to reduce MTP recurrent rollback snapshot VRAM (" +
                                   std::to_string(est.total_mib()) + " MiB)");
        }
    }

    // Step 3: step down ctx_size (aligned to 4096, down to min_ctx_size).
    if (!fits_vram(est, profile) && *profile.ctx_size > min_ctx) {
        const std::uint32_t prev_ctx = *profile.ctx_size;
        const auto candidates = context_stepdown_candidates(min_ctx, prev_ctx);
        if (!candidates.empty()) {
            LaunchProfile try_min = profile;
            try_min.ctx_size = candidates.front();
            const VramEstimate est_min = evaluate(try_min, try_min.vram_budget_mib);
            if (fits_vram(est_min, try_min)) {
                std::size_t lo = 0;
                std::size_t hi = candidates.size() - 1;
                std::size_t best_idx = 0;
                VramEstimate best_est = est_min;
                while (lo <= hi) {
                    const std::size_t mid = lo + (hi - lo) / 2;
                    LaunchProfile probe = profile;
                    probe.ctx_size = candidates[mid];
                    const VramEstimate e = evaluate(probe, probe.vram_budget_mib);
                    if (fits_vram(e, probe)) {
                        best_idx = mid;
                        best_est = e;
                        lo = mid + 1;
                    } else {
                        if (mid == 0) break;
                        hi = mid - 1;
                    }
                }
                profile.ctx_size = candidates[best_idx];
                est = best_est;
                plan.reasons.push_back("stepped ctx_size down from " + std::to_string(prev_ctx) + " to " +
                                       std::to_string(*profile.ctx_size) + " (4096-aligned) to fit VRAM budget (" +
                                       std::to_string(est.total_mib()) + " MiB)");

                // Re-minimize --n-cpu-moe if we had pushed all experts to CPU before ctx step-down.
                if (moe_offload_active && model.block_count > 1) {
                    std::uint32_t m_lo = 0;
                    std::uint32_t m_hi = model.block_count;
                    std::uint32_t m_best = model.block_count;
                    VramEstimate m_est = est;
                    while (m_lo <= m_hi) {
                        const std::uint32_t mid = m_lo + (m_hi - m_lo) / 2;
                        LaunchProfile probe = profile;
                        if (mid == 0) {
                            probe.extra_args = strip_n_cpu_moe_args(probe.extra_args);
                        } else {
                            probe = with_n_cpu_moe(profile, mid);
                        }
                        const VramEstimate e = evaluate(probe, probe.vram_budget_mib);
                        if (fits_vram(e, probe)) {
                            m_best = mid;
                            m_est = e;
                            if (mid == 0) break;
                            m_hi = mid - 1;
                        } else {
                            m_lo = mid + 1;
                        }
                    }
                    if (m_best == 0) {
                        profile.extra_args = strip_n_cpu_moe_args(profile.extra_args);
                        est = m_est;
                        plan.reasons.emplace_back("removed --n-cpu-moe after ctx_size step-down");
                    } else if (m_best < model.block_count) {
                        profile = with_n_cpu_moe(profile, m_best);
                        est = m_est;
                        plan.reasons.push_back("tightened --n-cpu-moe to " + std::to_string(m_best) + "/" +
                                               std::to_string(model.block_count) + " after ctx_size step-down");
                    }
                }
            } else {
                profile.ctx_size = min_ctx;
                est = est_min;
                plan.reasons.push_back("stepped ctx_size down from " + std::to_string(prev_ctx) +
                                       " to min_ctx_size (" + std::to_string(min_ctx) + ") before partial offload");
            }
        }
    }

    // Step 3b: if MTP still prevents fitting at min_ctx, test dropping MTP before dropping GPU layers.
    if (!fits_vram(est, profile) && profile.speculative &&
        std::find(profile.speculative->types.begin(), profile.speculative->types.end(), "draft-mtp") !=
            profile.speculative->types.end()) {
        LaunchProfile no_mtp = profile;
        if (model.has_ternary_tensors || model.has_one_bit_tensors) {
            SpeculativeSettings ngram;
            ngram.types = {"ngram-mod"};
            no_mtp.speculative = std::move(ngram);
        } else {
            no_mtp.speculative = std::nullopt;
        }
        const VramEstimate est_no_mtp = evaluate(no_mtp, no_mtp.vram_budget_mib);
        if (fits_vram(est_no_mtp, no_mtp)) {
            profile = std::move(no_mtp);
            est = est_no_mtp;
            plan.reasons.emplace_back(
                "disabled draft-mtp to recover MTP layer and recurrent rollback snapshot VRAM");
            plan.warnings.emplace_back("draft-mtp disabled due to tight VRAM budget");
        }
    }

    // Step 3c: if mmproj offload still prevents fitting at min_ctx, move mmproj to CPU before dropping LLM layers.
    if (!fits_vram(est, profile) && !profile.mmproj.empty() && profile.mmproj_offload.value_or(true)) {
        LaunchProfile cpu_mmproj = profile;
        cpu_mmproj.mmproj_offload = false;
        const VramEstimate est_cpu_mmproj = evaluate(cpu_mmproj, cpu_mmproj.vram_budget_mib);
        if (fits_vram(est_cpu_mmproj, cpu_mmproj)) {
            profile = std::move(cpu_mmproj);
            est = est_cpu_mmproj;
            plan.reasons.emplace_back("kept multimodal projector (mmproj) on CPU to keep all LLM layers in VRAM");
            plan.warnings.emplace_back("mmproj_offload disabled due to tight VRAM budget");
        }
    }

    // Step 4: last resort — reduce n_gpu_layers to the maximum offloaded layer count that fits.
    if (!fits_vram(est, profile)) {
        LaunchProfile partial_base = profile;
        if (partial_base.load_mode && *partial_base.load_mode == "none") {
            partial_base.load_mode = "auto";
        }
        std::uint32_t best_layers = 0;
        const std::uint32_t max_layers =
            std::min<std::uint32_t>(model.block_count, static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()));
        if (max_layers > 0) {
            std::uint32_t lo = 1;
            std::uint32_t hi = max_layers;
            while (lo <= hi) {
                const std::uint32_t mid = lo + (hi - lo) / 2;
                LaunchProfile probe = partial_base;
                probe.n_gpu_layers = static_cast<std::int32_t>(mid);
                const VramEstimate e = evaluate(probe, probe.vram_budget_mib);
                if (fits_vram(e, probe)) {
                    best_layers = mid;
                    lo = mid + 1;
                } else {
                    if (mid == 1) break;
                    hi = mid - 1;
                }
            }
        }

        if (best_layers == 0) {
            profile = std::move(partial_base);
            profile.n_gpu_layers = 0;
            if (!profile.mmproj.empty()) {
                profile.mmproj_offload = false;
            }
            profile.extra_args = strip_n_cpu_moe_args(profile.extra_args);
            profile.device = std::nullopt;
            profile.split_mode = std::nullopt;
            profile.tensor_split.clear();
            profile.main_gpu = std::nullopt;
            profile.vram_budget_mib = std::nullopt;
            plan.recommended_env.clear();
            plan.recommended_backend_hint = "cpu";
            est = evaluate(profile, std::nullopt);
            plan.fits_in_vram = false;
            plan.reasons.push_back("even 1 GPU layer exceeds VRAM budget (" + std::to_string(plan.budget_mib) +
                                   " MiB); fell back to n_gpu_layers=0");
            plan.warnings.push_back("partial GPU offload fell back to 0 GPU layers (CPU-only); VRAM budget " +
                                    std::to_string(plan.budget_mib) + " MiB is too small for 1 layer + compute");
        } else {
            profile = std::move(partial_base);
            profile.n_gpu_layers = static_cast<std::int32_t>(best_layers);
            est = evaluate(profile, profile.vram_budget_mib);
            plan.fits_in_vram = true;
            plan.reasons.push_back("reduced n_gpu_layers to " + std::to_string(best_layers) + "/" +
                                   std::to_string(model.block_count) + " (maximum layers fitting VRAM budget: " +
                                   std::to_string(est.total_mib()) + " MiB)");
            plan.warnings.push_back("partial GPU offload (" + std::to_string(best_layers) + "/" +
                                    std::to_string(model.block_count) + " layers on GPU" +
                                    (best_layers == model.block_count ? ", output tensor on host" : "") +
                                    "); remaining weights run on CPU");
        }
    } else {
        plan.fits_in_vram = true;
    }

    // If host-resident weights exceed available host RAM (e.g. repackable MoE
    // experts or partial CPU layers), try no_repack to avoid private-RAM copies.
    if (!check_host_fit(est, plan.uma_mode) && !profile.no_repack.value_or(false)) {
        LaunchProfile nr = profile;
        nr.no_repack = true;
        if (nr.load_mode && *nr.load_mode == "none") nr.load_mode = "auto";
        const VramEstimate est_nr = evaluate(nr, nr.vram_budget_mib);
        if (est_nr.host_resident_bytes < est.host_resident_bytes) {
            profile = std::move(nr);
            est = est_nr;
            plan.reasons.push_back("enabled no_repack to reduce host resident RAM to " +
                                   std::to_string(est.host_resident_mib()) + " MiB");
        }
    }

    // --------------------------------------------------- 8. Finalize & audit
    plan.profile = std::move(profile);
    plan.estimate = evaluate(plan.profile, plan.profile.vram_budget_mib);
    plan.fits_in_host_ram = check_host_fit(plan.estimate, plan.uma_mode);

    if (!plan.fits_in_host_ram && host.available_physical.is_known()) {
        const std::uint64_t avail_mib = host.available_physical.bytes_or(0) / kMiB;
        const std::uint64_t required_mib =
            plan.uma_mode ? bytes_to_mib_ceil(saturating_add(plan.estimate.total_bytes, plan.estimate.host_resident_bytes))
                          : plan.estimate.host_resident_mib();
        plan.warnings.push_back("estimated host RAM demand (" + std::to_string(required_mib) +
                                " MiB) exceeds available physical RAM (" + std::to_string(avail_mib) + " MiB)");
    }

    if (plan.fits_in_vram && plan.budget_mib > 0) {
        const bool multi_split = plan.estimate.max_gpu_bytes > 0;
        const std::uint64_t used_mib = multi_split ? plan.estimate.max_gpu_mib() : plan.estimate.total_mib();
        const std::uint64_t ref_budget_mib =
            (multi_split && plan.profile.vram_budget_mib) ? *plan.profile.vram_budget_mib : plan.budget_mib;
        if (ref_budget_mib > 0) {
            const std::uint64_t headroom_left = ref_budget_mib > used_mib ? ref_budget_mib - used_mib : 0;
            const std::uint64_t pct = used_mib > std::numeric_limits<std::uint64_t>::max() / 100ull
                                          ? 100ull
                                          : (used_mib * 100ull) / ref_budget_mib;
            if (pct >= 90 || headroom_left < 512) {
                plan.warnings.push_back("tight VRAM headroom: estimated " + std::to_string(used_mib) +
                                        " MiB uses " + std::to_string(pct) + "% of " +
                                        std::to_string(ref_budget_mib) + " MiB budget (" +
                                        std::to_string(headroom_left) + " MiB remaining)");
            }
        }
    }

    if (plan.profile.cache_type_v && *plan.profile.cache_type_v == "q4_0") {
        plan.warnings.emplace_back(
            "4-bit V cache (q4_0) enabled to fit VRAM budget; verify generation quality on this model");
    }

    return plan;
}

json::Object placement_plan_to_json(const PlacementPlan& plan) {
    // Emit `profile` in the exact schema accepted by parse_launch_profiles().
    json::Object prof;
    prof.set("name", plan.profile.name);
    prof.set("backend", plan.profile.backend);
    prof.set("model", plan.profile.model);
    if (!plan.profile.mmproj.empty()) prof.set("mmproj", plan.profile.mmproj);
    if (plan.profile.mmproj_offload) prof.set("mmproj_offload", *plan.profile.mmproj_offload);
    if (plan.profile.image_max_tokens) prof.set("image_max_tokens", *plan.profile.image_max_tokens);
    if (plan.profile.ctx_size) prof.set("ctx_size", *plan.profile.ctx_size);
    if (plan.profile.n_gpu_layers) prof.set("n_gpu_layers", *plan.profile.n_gpu_layers);
    if (plan.profile.device) prof.set("device", *plan.profile.device);
    if (plan.profile.main_gpu) prof.set("main_gpu", *plan.profile.main_gpu);
    if (plan.profile.split_mode) prof.set("split_mode", *plan.profile.split_mode);
    if (!plan.profile.tensor_split.empty()) {
        json::Array ts;
        ts.reserve(plan.profile.tensor_split.size());
        for (const double w : plan.profile.tensor_split) ts.emplace_back(w);
        prof.set("tensor_split", std::move(ts));
    }
    if (plan.profile.load_mode) prof.set("load_mode", *plan.profile.load_mode);
    if (plan.profile.no_repack) prof.set("no_repack", *plan.profile.no_repack);
    if (plan.profile.batch_size) prof.set("batch_size", *plan.profile.batch_size);
    if (plan.profile.ubatch_size) prof.set("ubatch_size", *plan.profile.ubatch_size);
    if (plan.profile.parallel) prof.set("parallel", *plan.profile.parallel);
    if (plan.profile.kv_unified) prof.set("kv_unified", *plan.profile.kv_unified);
    if (plan.profile.flash_attn) prof.set("flash_attn", *plan.profile.flash_attn);
    if (plan.profile.cache_type_k) prof.set("cache_type_k", *plan.profile.cache_type_k);
    if (plan.profile.cache_type_v) prof.set("cache_type_v", *plan.profile.cache_type_v);
    if (plan.profile.fit) prof.set("fit", *plan.profile.fit);
    if (plan.profile.speculative) {
        json::Object spec;
        json::Array types;
        types.reserve(plan.profile.speculative->types.size());
        for (const auto& t : plan.profile.speculative->types) types.emplace_back(t);
        spec.set("types", std::move(types));
        if (plan.profile.speculative->draft_n_max) {
            spec.set("draft_n_max", *plan.profile.speculative->draft_n_max);
        }
        if (!plan.profile.speculative->draft_model.empty()) {
            spec.set("draft_model", plan.profile.speculative->draft_model);
        }
        prof.set("speculative", std::move(spec));
    }
    if (!plan.profile.extra_args.empty()) {
        json::Array extra;
        extra.reserve(plan.profile.extra_args.size());
        for (const auto& arg : plan.profile.extra_args) extra.emplace_back(arg);
        prof.set("extra_args", std::move(extra));
    }
    if (plan.profile.vram_budget_mib) prof.set("vram_budget_mib", *plan.profile.vram_budget_mib);

    json::Object est;
    est.set("total_bytes", plan.estimate.total_bytes);
    est.set("total_mib", plan.estimate.total_mib());
    est.set("weights_bytes", plan.estimate.weights_bytes);
    est.set("kv_bytes", plan.estimate.kv_bytes);
    est.set("recurrent_bytes", plan.estimate.recurrent_bytes);
    est.set("compute_bytes", plan.estimate.compute_bytes);
    est.set("max_gpu_bytes", plan.estimate.max_gpu_bytes);
    est.set("max_gpu_mib", plan.estimate.max_gpu_mib());
    est.set("gpu_layers", plan.estimate.gpu_layers);
    est.set("attention_layers", plan.estimate.attention_layers);
    est.set("swa_attention_layers", plan.estimate.swa_attention_layers);
    est.set("context", plan.estimate.context);
    est.set("context_per_sequence", plan.estimate.context_per_sequence);
    est.set("sequences", plan.estimate.sequences);
    est.set("recurrent_sequences", plan.estimate.recurrent_sequences);
    est.set("recurrent_snapshots", plan.estimate.recurrent_snapshots);
    est.set("host_resident_bytes", plan.estimate.host_resident_bytes);
    est.set("host_resident_mib", plan.estimate.host_resident_mib());
    est.set("host_private_bytes", plan.estimate.host_private_bytes);
    est.set("host_page_cache_bytes", plan.estimate.host_page_cache_bytes);
    est.set("host_dense_weight_bytes", plan.estimate.host_dense_weight_bytes);
    est.set("host_expert_bytes", plan.estimate.host_expert_bytes);
    est.set("host_repacked_bytes", plan.estimate.host_repacked_bytes);
    est.set("host_weights_mmapped", plan.estimate.host_weights_mmapped);
    est.set("prism_hadamard", plan.estimate.prism_hadamard);
    est.set("requires_prism_runtime", plan.estimate.requires_prism_runtime);
    json::Array est_notes;
    est_notes.reserve(plan.estimate.notes.size());
    for (const auto& note : plan.estimate.notes) est_notes.emplace_back(note);
    est.set("notes", std::move(est_notes));

    json::Object env_obj;
    for (const auto& [k, v] : plan.recommended_env) env_obj.set(k, v);

    json::Array reasons;
    reasons.reserve(plan.reasons.size());
    for (const auto& r : plan.reasons) reasons.emplace_back(r);

    json::Array warnings;
    warnings.reserve(plan.warnings.size());
    for (const auto& w : plan.warnings) warnings.emplace_back(w);

    json::Object out;
    out.set("budget_mib", plan.budget_mib);
    out.set("fits_in_vram", plan.fits_in_vram);
    out.set("fits_in_host_ram", plan.fits_in_host_ram);
    out.set("uma_mode", plan.uma_mode);
    out.set("requires_prism_runtime", plan.requires_prism_runtime);
    out.set("selected_vendor", to_string(plan.selected_vendor));
    out.set("recommended_backend_hint", plan.recommended_backend_hint);
    out.set("profile", std::move(prof));
    out.set("estimate", std::move(est));
    out.set("recommended_env", std::move(env_obj));
    out.set("reasons", std::move(reasons));
    out.set("warnings", std::move(warnings));
    return out;
}

}  // namespace sonder::inference
