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

std::optional<std::string_view> normalize_uma_mode(std::string_view text) noexcept {
    if (text == "on" || text == "true")
        return std::string_view{"on"};
    if (text == "off" || text == "false")
        return std::string_view{"off"};
    if (text == "auto")
        return std::string_view{"auto"};
    return std::nullopt;
}

std::string_view trim_ascii(std::string_view sv) noexcept {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r' || sv.front() == '\n'))
        sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r' || sv.back() == '\n'))
        sv.remove_suffix(1);
    return sv;
}

char ascii_lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (ascii_lower(a[i]) != ascii_lower(b[i]))
            return false;
    }
    return true;
}

bool istarts_with(std::string_view text, std::string_view prefix) noexcept {
    return text.size() >= prefix.size() && iequals(text.substr(0, prefix.size()), prefix);
}

std::optional<std::size_t> parse_index(std::string_view text) noexcept {
    if (text.empty())
        return std::nullopt;
    std::size_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9')
            return std::nullopt;
        const auto digit = static_cast<std::size_t>(c - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10)
            return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}

template <typename Pred>
std::optional<std::size_t> select_best_gpu(const std::vector<GpuDeviceInfo> &gpus, const Pred &pred) noexcept {
    std::optional<std::size_t> best_discrete;
    std::uint64_t best_discrete_mib = 0;
    std::optional<std::size_t> best_uma;
    std::uint64_t best_uma_mib = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i]))
            continue;
        const std::uint64_t mib = gpu_effective_vram_mib(gpus[i], false).value_or(0);
        if (!gpus[i].is_integrated_uma) {
            if (!best_discrete || mib > best_discrete_mib) {
                best_discrete = i;
                best_discrete_mib = mib;
            }
        } else {
            if (!best_uma || mib > best_uma_mib) {
                best_uma = i;
                best_uma_mib = mib;
            }
        }
    }
    return best_discrete ? best_discrete : best_uma;
}

template <typename Pred>
std::optional<std::size_t> match_vendor_hint(const std::vector<GpuDeviceInfo> &gpus,
                                             std::string_view suffix,
                                             const Pred &pred) noexcept {
    if (suffix.empty())
        return select_best_gpu(gpus, pred);
    if (suffix.front() == ':')
        suffix.remove_prefix(1);
    const auto target_idx = parse_index(suffix);
    if (!target_idx)
        return std::nullopt;
    std::size_t seen = 0;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (!pred(gpus[i]))
            continue;
        if (seen == *target_idx)
            return i;
        ++seen;
    }
    return std::nullopt;
}

std::optional<std::size_t> match_single_gpu_hint(const std::vector<GpuDeviceInfo> &gpus,
                                                 std::string_view token) noexcept {
    token = trim_ascii(token);
    if (token.empty())
        return std::nullopt;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (iequals(gpus[i].id, token))
            return i;
    }
    if (istarts_with(token, "gpu:")) {
        const auto idx = parse_index(token.substr(4));
        if (idx && *idx < gpus.size())
            return *idx;
        return std::nullopt;
    }
    if (const auto idx = parse_index(token); idx && *idx < gpus.size())
        return *idx;

    const auto is_nvidia = [](const GpuDeviceInfo &g) noexcept {
        return g.vendor == GpuVendor::nvidia || iequals(g.backend_hint, "cuda");
    };
    const auto is_amd = [](const GpuDeviceInfo &g) noexcept {
        return g.vendor == GpuVendor::amd || iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    };
    const bool any_explicit_rocm = std::any_of(gpus.begin(), gpus.end(), [](const GpuDeviceInfo &g) noexcept {
        return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
    });
    const auto is_rocm_target = [any_explicit_rocm, &is_amd](const GpuDeviceInfo &g) noexcept {
        if (any_explicit_rocm)
            return iequals(g.backend_hint, "rocm") || iequals(g.backend_hint, "hip");
        return is_amd(g);
    };
    const auto is_intel = [](const GpuDeviceInfo &g) noexcept { return g.vendor == GpuVendor::intel; };
    const auto is_apple = [](const GpuDeviceInfo &g) noexcept {
        return g.vendor == GpuVendor::apple || iequals(g.backend_hint, "metal");
    };
    const auto is_uma = [](const GpuDeviceInfo &g) noexcept { return g.is_integrated_uma; };
    const auto is_any = [](const GpuDeviceInfo &) noexcept { return true; };

    if (iequals(token, "uma") || iequals(token, "igpu") || iequals(token, "apu"))
        return select_best_gpu(gpus, is_uma);

    if (istarts_with(token, "cuda"))
        return match_vendor_hint(gpus, token.substr(4), is_nvidia);
    if (istarts_with(token, "nvidia"))
        return match_vendor_hint(gpus, token.substr(6), is_nvidia);
    const auto match_rocm_hint = [&](std::string_view suffix) noexcept -> std::optional<std::size_t> {
        std::string_view trimmed_suffix = trim_ascii(suffix);
        if (!trimmed_suffix.empty() && trimmed_suffix.front() == ':')
            trimmed_suffix.remove_prefix(1);
        if (const auto target_idx = parse_index(trimmed_suffix)) {
            for (std::size_t i = 0; i < gpus.size(); ++i) {
                if (gpus[i].rocm_runtime_ordinal && *gpus[i].rocm_runtime_ordinal == *target_idx)
                    return i;
            }
        }
        return match_vendor_hint(gpus, suffix, is_rocm_target);
    };
    if (istarts_with(token, "rocm"))
        return match_rocm_hint(token.substr(4));
    if (istarts_with(token, "hip"))
        return match_rocm_hint(token.substr(3));
    if (istarts_with(token, "amd"))
        return match_vendor_hint(gpus, token.substr(3), is_amd);
    if (istarts_with(token, "vulkan"))
        return match_vendor_hint(gpus, token.substr(6), is_any);
    if (istarts_with(token, "vk"))
        return match_vendor_hint(gpus, token.substr(2), is_any);
    if (istarts_with(token, "intel"))
        return match_vendor_hint(gpus, token.substr(5), is_intel);
    if (istarts_with(token, "sycl"))
        return match_vendor_hint(gpus, token.substr(4), is_intel);
    if (istarts_with(token, "metal"))
        return match_vendor_hint(gpus, token.substr(5), is_apple);
    if (istarts_with(token, "apple"))
        return match_vendor_hint(gpus, token.substr(5), is_apple);
    return std::nullopt;
}

}  // namespace

Status parse_uma_mode_value(const json::Value &value, const char *field_path, std::string &out) {
    if (value.is_bool()) {
        out = value.as_bool() ? "on" : "off";
        return {};
    }
    if (value.is_string()) {
        if (const auto normalized = normalize_uma_mode(value.as_string())) {
            out.assign(*normalized);
            return {};
        }
    }
    return Status(ErrorCode::invalid_argument,
                  std::string("llamaserver config: ") + field_path +
                      " must be boolean or \"auto\", \"on\", \"off\", \"true\", \"false\"");
}

bool is_valid_uma_mode(std::string_view mode) noexcept {
    return normalize_uma_mode(mode).has_value();
}

std::string_view extract_device_hint(const std::vector<std::string> &args) noexcept {
    std::string_view result;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--device" || arg == "-dev") {
            if (i + 1 < args.size()) {
                result = args[++i];
            } else {
                result = {};
            }
        } else if (arg.size() > 9 && arg.compare(0, 9, "--device=") == 0) {
            result = arg.substr(9);
        } else if (arg.size() > 5 && arg.compare(0, 5, "-dev=") == 0) {
            result = arg.substr(5);
        } else if (arg == "--device=" || arg == "-dev=") {
            result = {};
        }
    }
    return result;
}

bool resolve_uma_mode(std::string_view mode_or_auto,
                      const std::vector<GpuDeviceInfo> &gpus,
                      std::string_view device_hint) noexcept {
    if (mode_or_auto == "on" || mode_or_auto == "true")
        return true;
    if (mode_or_auto != "auto")
        return false;
    if (gpus.empty())
        return false;

    const std::string_view trimmed = trim_ascii(device_hint);
    if (trimmed.empty()) {
        // Discrete GPU wins whenever present; return true only when all
        // discovered GPUs are integrated UMA (e.g. Ryzen 780M/890M APU-only).
        for (const auto &gpu : gpus) {
            if (!gpu.is_integrated_uma)
                return false;
        }
        return true;
    }

    bool matched_any = false;
    bool any_uma = false;
    std::size_t pos = 0;
    while (pos <= trimmed.size()) {
        const auto comma = trimmed.find(',', pos);
        const std::string_view token =
            trim_ascii(trimmed.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
        pos = comma == std::string_view::npos ? trimmed.size() + 1 : comma + 1;
        if (token.empty())
            continue;
        const auto idx = match_single_gpu_hint(gpus, token);
        if (!idx)
            return false;
        matched_any = true;
        if (gpus[*idx].is_integrated_uma)
            any_uma = true;
    }
    return matched_any && any_uma;
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

}  // namespace sonder::inference::detail
