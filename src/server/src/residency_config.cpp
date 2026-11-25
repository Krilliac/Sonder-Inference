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

}  // namespace sonder::inference::detail
