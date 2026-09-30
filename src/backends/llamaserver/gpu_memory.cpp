#include "gpu_memory.hpp"

#include <cmath>
#include <limits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h must precede the PDH headers.
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace sonder::inference::llamaserver {
namespace {

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

#if defined(_WIN32)
std::string hex_status(PDH_STATUS status) {
    static constexpr char kDigits[] = "0123456789ABCDEF";
    std::string out = "0x";
    for (int shift = 28; shift >= 0; shift -= 4)
        out.push_back(kDigits[(static_cast<std::uint32_t>(status) >> shift) & 0xFu]);
    return out;
}

std::string narrow(const wchar_t *text) {
    std::string out;
    if (!text)
        return out;
    // PDH instance names of this object are ASCII ("pid_<n>_luid_0x..._phys_<n>").
    for (; *text; ++text)
        out.push_back(*text < 0x80 ? static_cast<char>(*text) : '?');
    return out;
}

class PdhQuery {
  public:
    PdhQuery() { status_ = PdhOpenQueryW(nullptr, 0, &query_); }
    ~PdhQuery() {
        if (query_)
            PdhCloseQuery(query_);
    }
    PdhQuery(const PdhQuery &) = delete;
    PdhQuery &operator=(const PdhQuery &) = delete;
    [[nodiscard]] PDH_STATUS status() const { return status_; }
    [[nodiscard]] PDH_HQUERY get() const { return query_; }

  private:
    PDH_HQUERY query_ = nullptr;
    PDH_STATUS status_ = ERROR_SUCCESS;
};

// A wildcard instance with no matches is an empty (not failed) sample: a
// process that has not created a GPU context yet has no instances.
bool is_no_instance(PDH_STATUS status) {
    return status == static_cast<PDH_STATUS>(PDH_NO_DATA) || status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_INSTANCE) ||
           status == static_cast<PDH_STATUS>(PDH_INVALID_DATA);
}

Status read_array(PDH_HCOUNTER counter, std::vector<CounterInstance> &out) {
    DWORD bytes = 0;
    DWORD count = 0;
    PDH_STATUS status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count, nullptr);
    if (is_no_instance(status))
        return {};
    if (status != static_cast<PDH_STATUS>(PDH_MORE_DATA))
        return Status(ErrorCode::io_error, "PdhGetFormattedCounterArray failed: " + hex_status(status));
    // Bounded: a process has one instance per adapter/segment view.
    if (bytes == 0 || bytes > 1024u * 1024u)
        return Status(ErrorCode::io_error, "PDH counter array has an implausible size");
    std::vector<unsigned char> buffer(bytes);
    auto *items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W *>(buffer.data());
    status = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count, items);
    if (is_no_instance(status))
        return {};
    if (status != ERROR_SUCCESS)
        return Status(ErrorCode::io_error, "PdhGetFormattedCounterArray failed: " + hex_status(status));
    for (DWORD i = 0; i < count; ++i) {
        const auto &item = items[i];
        if (item.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && item.FmtValue.CStatus != PDH_CSTATUS_NEW_DATA)
            continue;
        if (item.FmtValue.largeValue < 0)
            continue;
        out.push_back({narrow(item.szName), static_cast<std::uint64_t>(item.FmtValue.largeValue)});
    }
    return {};
}

class PdhCounterSource final : public GpuCounterSource {
  public:
    std::string name() const override { return "pdh"; }
    bool supported() const override { return true; }
    Result<GpuProcessCounters> read(std::uint32_t pid) override {
        if (pid == 0)
            return Status(ErrorCode::invalid_argument, "no process id to sample");
        PdhQuery query;
        if (query.status() != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhOpenQuery failed: " + hex_status(query.status()));
        const std::wstring prefix = L"\\GPU Process Memory(pid_" + std::to_wstring(pid) + L"_*)\\";
        PDH_HCOUNTER dedicated = nullptr;
        PDH_HCOUNTER shared = nullptr;
        PDH_STATUS status = PdhAddEnglishCounterW(query.get(), (prefix + L"Dedicated Usage").c_str(), 0, &dedicated);
        if (status == ERROR_SUCCESS)
            status = PdhAddEnglishCounterW(query.get(), (prefix + L"Shared Usage").c_str(), 0, &shared);
        if (status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_OBJECT) ||
            status == static_cast<PDH_STATUS>(PDH_CSTATUS_NO_COUNTER))
            return Status(ErrorCode::unsupported,
                          "the GPU Process Memory performance counters are not available (WDDM 2.x required)");
        if (status != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhAddEnglishCounter failed: " + hex_status(status));
        status = PdhCollectQueryData(query.get());
        GpuProcessCounters counters;
        if (is_no_instance(status))
            return counters;
        if (status != ERROR_SUCCESS)
            return Status(ErrorCode::io_error, "PdhCollectQueryData failed: " + hex_status(status));
        if (auto st = read_array(dedicated, counters.dedicated); !st.ok())
            return st;
        if (auto st = read_array(shared, counters.shared); !st.ok())
            return st;
        return counters;
    }
};
#else
class UnsupportedCounterSource final : public GpuCounterSource {
  public:
    std::string name() const override { return "none"; }
    bool supported() const override { return false; }
    Result<GpuProcessCounters> read(std::uint32_t) override {
        return Status(ErrorCode::unsupported, "per-process GPU memory counters are only implemented on Windows");
    }
};
#endif

} // namespace

std::shared_ptr<GpuCounterSource> make_gpu_counter_source() {
#if defined(_WIN32)
    return std::make_shared<PdhCounterSource>();
#else
    return std::make_shared<UnsupportedCounterSource>();
#endif
}

bool is_process_instance(std::string_view name, std::uint32_t pid) {
    const std::string prefix = "pid_" + std::to_string(pid) + "_";
    return name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0;
}

GpuMemorySample summarize_gpu_counters(std::uint32_t pid, const GpuProcessCounters &counters) {
    GpuMemorySample sample;
    std::size_t dedicated_instances = 0;
    std::size_t shared_instances = 0;
    for (const auto &item : counters.dedicated) {
        if (!is_process_instance(item.name, pid))
            continue;
        sample.dedicated_bytes = saturating_add(sample.dedicated_bytes, item.value);
        ++dedicated_instances;
    }
    for (const auto &item : counters.shared) {
        if (!is_process_instance(item.name, pid))
            continue;
        sample.shared_bytes = saturating_add(sample.shared_bytes, item.value);
        ++shared_instances;
    }
    sample.instances = dedicated_instances > shared_instances ? dedicated_instances : shared_instances;
    return sample;
}

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

bool is_spilled(const GpuMemorySample &sample, const SpillGuardOptions &options,
                std::optional<std::uint64_t> ctx) {
    const auto limit = saturating_add(effective_baseline(options, ctx), options.threshold_bytes);
    return sample.shared_bytes > limit;
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
