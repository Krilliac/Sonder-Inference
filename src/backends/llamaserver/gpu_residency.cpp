#include "gpu_residency.hpp"

#include <cmath>
#include <limits>

namespace sonder::inference::llamaserver {
namespace {

constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;

bool is_gpu_flag(std::string_view arg, std::string_view &inline_value, bool &has_inline_value) {
    constexpr std::string_view flags[] = {"--n-gpu-layers", "--gpu-layers", "-ngl"};
    for (const auto flag : flags) {
        if (arg == flag) {
            inline_value = {};
            has_inline_value = false;
            return true;
        }
        if (arg.size() >= flag.size() + 1 && arg.compare(0, flag.size(), flag) == 0 && arg[flag.size()] == '=') {
            inline_value = arg.substr(flag.size() + 1);
            has_inline_value = true;
            return true;
        }
    }
    return false;
}

bool positive_layers(std::string_view value) {
    if (value == "all" || value == "-1")
        return true;
    if (value.empty())
        return false;
    std::uint64_t number = 0;
    for (const char c : value) {
        if (c < '0' || c > '9')
            return false;
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (number > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            return false;
        number = number * 10 + digit;
    }
    return number != 0;
}

}  // namespace

Status validate_residency_guard(const LlamaServerResidencyGuardOptions &o) {
    const auto invalid = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: residency guard: ") + what);
    };
    if (o.on_eviction != LlamaServerEvictionPolicy::warn && o.on_eviction != LlamaServerEvictionPolicy::restart)
        return invalid("unknown eviction policy");
    if (o.consecutive_samples == 0 || o.consecutive_samples > 1000)
        return invalid("consecutive samples must be in [1, 1000]");
    if (!std::isfinite(o.eviction_fraction) || o.eviction_fraction < 0.0 || o.eviction_fraction > 1.0)
        return invalid("eviction fraction must be finite and in [0, 1]");
    if (o.eviction_fraction == 0.0 && o.eviction_bytes == 0)
        return invalid("at least one eviction threshold must be enabled");
    if (o.min_dedicated_bytes == 0 || o.min_dedicated_bytes > kGiB * 1024)
        return invalid("minimum dedicated usage must be in (0, 1 TiB]");
    if (o.eviction_bytes > kGiB * 1024)
        return invalid("eviction bytes must be at most 1 TiB");
    if (o.max_eviction_restarts > 32)
        return invalid("maximum eviction restarts must be at most 32");
    return {};
}

bool expects_gpu_offload(const std::vector<std::string> &args, bool assume_gpu) {
    bool expected = assume_gpu;
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view inline_value;
        bool has_inline_value = false;
        if (!is_gpu_flag(args[i], inline_value, has_inline_value))
            continue;
        std::string_view value = inline_value;
        if (!has_inline_value && i + 1 < args.size()) {
            std::string_view next_inline;
            bool next_has_inline_value = false;
            if (!is_gpu_flag(args[i + 1], next_inline, next_has_inline_value))
                value = args[++i];
        }
        // An explicit malformed or missing value is fail-closed, and because
        // this is the last occurrence it intentionally overrides earlier flags.
        expected = positive_layers(value);
    }
    return expected;
}

GpuResidencyGuard::GpuResidencyGuard(LlamaServerResidencyGuardOptions options, bool gpu_expected)
    : options_(options), gpu_expected_(gpu_expected) {}

void GpuResidencyGuard::reset() {
    gpu_offload_missing_ = false;
    vram_evicted_ = false;
    event_observed_ = false;
    missing_streak_ = 0;
    eviction_streak_ = 0;
    peak_dedicated_bytes_ = 0;
    observed_dedicated_bytes_ = 0;
}

void GpuResidencyGuard::unavailable() {
    missing_streak_ = 0;
    eviction_streak_ = 0;
}

void GpuResidencyGuard::observe(std::uint64_t dedicated_bytes) {
    if (!options_.enabled)
        return;

    if (dedicated_bytes > peak_dedicated_bytes_)
        peak_dedicated_bytes_ = dedicated_bytes;
    if (!event_observed_)
        observed_dedicated_bytes_ = dedicated_bytes;

    if (gpu_expected_ && options_.min_dedicated_bytes != 0 && dedicated_bytes < options_.min_dedicated_bytes) {
        if (missing_streak_ < options_.consecutive_samples)
            ++missing_streak_;
        if (!gpu_offload_missing_ && missing_streak_ >= options_.consecutive_samples) {
            gpu_offload_missing_ = true;
            event_observed_ = true;
            observed_dedicated_bytes_ = dedicated_bytes;
        }
    } else {
        missing_streak_ = 0;
    }

    const bool sufficient_peak = peak_dedicated_bytes_ >= options_.min_dedicated_bytes;
    const std::uint64_t drop = peak_dedicated_bytes_ > dedicated_bytes ? peak_dedicated_bytes_ - dedicated_bytes : 0;
    const bool fraction_triggered = options_.eviction_fraction != 0.0 &&
                                    static_cast<long double>(drop) >
                                        static_cast<long double>(peak_dedicated_bytes_) * options_.eviction_fraction;
    const bool bytes_triggered = options_.eviction_bytes != 0 && drop > options_.eviction_bytes;
    if (sufficient_peak && (fraction_triggered || bytes_triggered)) {
        if (eviction_streak_ < options_.consecutive_samples)
            ++eviction_streak_;
        if (!vram_evicted_ && eviction_streak_ >= options_.consecutive_samples) {
            vram_evicted_ = true;
            event_observed_ = true;
            observed_dedicated_bytes_ = dedicated_bytes;
        }
    } else {
        eviction_streak_ = 0;
    }
}

}  // namespace sonder::inference::llamaserver
