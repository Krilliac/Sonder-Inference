#include "spill_guard.hpp"
#include "gpu_residency.hpp"

#include <cmath>
#include <limits>

namespace sonder::inference::llamaserver {
namespace {

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

} // namespace

const char *to_string(SpillPolicy policy) noexcept {
    switch (policy) {
    case SpillPolicy::warn:
        return "warn";
    case SpillPolicy::refuse:
        return "refuse";
    case SpillPolicy::auto_fit:
        return "auto_fit";
    }
    return "warn";
}

std::optional<SpillPolicy> parse_spill_policy(std::string_view text) noexcept {
    if (text == "warn")
        return SpillPolicy::warn;
    if (text == "refuse")
        return SpillPolicy::refuse;
    if (text == "auto_fit")
        return SpillPolicy::auto_fit;
    return std::nullopt;
}

Status validate_spill_guard(const SpillGuardOptions &o) {
    const auto invalid = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: spill guard: ") + what);
    };
    if (o.policy != SpillPolicy::warn && o.policy != SpillPolicy::refuse && o.policy != SpillPolicy::auto_fit)
        return invalid("unknown policy");
    if (o.threshold_bytes == 0 || o.threshold_bytes > 1024ull * 1024 * kMiB)
        return invalid("threshold must be in (0, 1 TiB]");
    if (o.baseline_bytes > 1024ull * 1024 * kMiB)
        return invalid("baseline must be at most 1 TiB");
    if (o.baseline_bytes_per_1k_ctx > 1024ull * kMiB)
        return invalid("baseline growth per 1k context must be at most 1 GiB");
    if (o.sample_interval.count() < 100 || o.sample_interval > std::chrono::hours(1))
        return invalid("sample interval must be in [100 ms, 1 h]");
    if (!std::isfinite(o.step_factor) || o.step_factor <= 0.0 || o.step_factor >= 1.0)
        return invalid("step factor must be in (0, 1)");
    if (o.step_align == 0 || o.step_align > 1u << 20)
        return invalid("step alignment must be in [1, 1048576]");
    if (o.min_ctx == 0 || o.min_ctx > (1ull << 32))
        return invalid("minimum context must be in [1, 2^32]");
    if (o.max_attempts == 0 || o.max_attempts > 32)
        return invalid("max fit attempts must be in [1, 32]");
    if (o.residency.enabled)
        return validate_residency_guard(o.residency);
    return {};
}

std::uint64_t effective_baseline(const SpillGuardOptions &options, std::optional<std::uint64_t> ctx) {
    if (!ctx || options.baseline_bytes_per_1k_ctx == 0)
        return options.baseline_bytes;
    const std::uint64_t blocks = *ctx / 1024;
    const std::uint64_t growth =
        blocks != 0 && options.baseline_bytes_per_1k_ctx > std::numeric_limits<std::uint64_t>::max() / blocks
            ? std::numeric_limits<std::uint64_t>::max()
            : options.baseline_bytes_per_1k_ctx * blocks;
    return saturating_add(options.baseline_bytes, growth);
}

std::uint64_t effective_spill_shared_bytes(const GpuMemorySample &sample,
                                           const SpillGuardOptions &options) noexcept {
    if (options.uma_mode && !options.hybrid_uma_mode)
        return 0;
    if (sample.has_adapter_classification) {
        if (options.hybrid_uma_mode || sample.discrete_instances > 0)
            return sample.discrete_shared_bytes;
    }
    if (options.uma_mode || options.hybrid_uma_mode)
        return 0;
    return sample.shared_bytes;
}

bool is_spilled(const GpuMemorySample &sample, const SpillGuardOptions &options,
                std::optional<std::uint64_t> ctx) {
    if (options.uma_mode && !options.hybrid_uma_mode)
        return false;
    if (options.hybrid_uma_mode && !sample.has_adapter_classification)
        return false;
    const std::uint64_t spill_shared = effective_spill_shared_bytes(sample, options);
    const auto limit = saturating_add(effective_baseline(options, ctx), options.threshold_bytes);
    return spill_shared > limit;
}

std::optional<std::uint64_t> next_fit_context(std::uint64_t current, const SpillGuardOptions &o) {
    if (current <= o.min_ctx || o.step_align == 0)
        return std::nullopt;
    const long double scaled = static_cast<long double>(current) * static_cast<long double>(o.step_factor);
    std::uint64_t next = scaled >= static_cast<long double>(current) ? current : static_cast<std::uint64_t>(scaled);
    next -= next % o.step_align;
    // Always make progress by at least one alignment step.
    if (next + o.step_align > current)
        next = current > o.step_align ? current - o.step_align : 0;
    if (next < o.min_ctx)
        next = o.min_ctx;
    if (next >= current)
        return std::nullopt;
    return next;
}

} // namespace sonder::inference::llamaserver
