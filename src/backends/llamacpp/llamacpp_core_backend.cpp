// Adapter: sonder::inference::Backend over the llama.cpp runtime wrapper.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#  if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    include <windows.h>
#  else
#    include <unistd.h>
#  endif
#endif

#include "common/diag_log.hpp"
#include "device_health.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/json.hpp"
#include "llamacpp_internal.hpp"

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#  include "backends/llamaserver/gpu_memory.hpp"
#endif

namespace sonder::inference {

namespace llamacpp_detail {

namespace {

std::uint64_t SaturatingAddU64(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

std::uint64_t SafeMsToNs(double ms) noexcept {
    if (!std::isfinite(ms) || ms <= 0.0) return 0;
    constexpr double kMaxMs = static_cast<double>(std::numeric_limits<std::uint64_t>::max()) / 1e6;
    if (ms >= kMaxMs) return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t>(ms * 1e6);
}

bool DeviceMatchesConfigured(const sonder::backends::llamacpp::DeviceInfo& d,
                             const std::vector<std::string>& configured_devices) {
    if (configured_devices.empty()) return true;
    const std::string name_lower = LowerAscii(d.name);
    for (const auto& cfg : configured_devices) {
        if (LowerAscii(cfg) == name_lower) return true;
    }
    return false;
}

}  // namespace

// Resolves LoadOptions::n_gpu_layers from backend options and a load request's
// device_id. Explicit "cpu:*" always uses 0; "gpu:*" requires a GPU backend
// and uses options.gpu_layers; empty device_id honors options.gpu_layers when
// explicit ggml devices are configured, otherwise defaults to 0 (CPU).
Result<std::int32_t> ResolveGpuLayers(const LlamaCppBackendOptions& options, std::string_view device_id,
                                      bool has_gpu_backend);

Result<std::int32_t> ResolveGpuLayers(const LlamaCppBackendOptions& options, std::string_view device_id,
                                      bool has_gpu_backend) {
    if (device_id.rfind("cpu", 0) == 0) {
        return std::int32_t{0};
    }
    if (device_id.empty()) {
        return options.devices.empty() ? std::int32_t{0} : options.gpu_layers;
    }
    if (device_id.rfind("gpu", 0) == 0) {
        if (!has_gpu_backend) {
            return Status(ErrorCode::unsupported, "this llama.cpp build has no GPU backend");
        }
        return options.gpu_layers;
    }
    return Status(ErrorCode::unsupported, "unsupported device id for llamacpp: " + std::string(device_id));
}

ProcessGpuMemoryReading SummarizeGgmlDeviceMemory(
    const std::vector<sonder::backends::llamacpp::DeviceInfo>& devices,
    const std::vector<std::string>& configured_devices) {
    namespace lc = sonder::backends::llamacpp;
    std::uint64_t dedicated_used = 0;
    std::uint64_t shared_used = 0;
    bool any_valid = false;
    for (const auto& d : devices) {
        if (d.kind != lc::DeviceKind::kGpu && d.kind != lc::DeviceKind::kIntegratedGpu) continue;
        if (!DeviceMatchesConfigured(d, configured_devices)) continue;
        if (d.memory_total_bytes == 0 || d.memory_free_bytes > d.memory_total_bytes) continue;
        const std::uint64_t used = d.memory_total_bytes - d.memory_free_bytes;
        if (d.kind == lc::DeviceKind::kGpu) {
            dedicated_used = SaturatingAddU64(dedicated_used, used);
        } else {
            shared_used = SaturatingAddU64(shared_used, used);
        }
        any_valid = true;
    }

    ProcessGpuMemoryReading out;
    if (any_valid) {
        out.probe = "ggml";
        out.status = "ok";
        out.dedicated_bytes = dedicated_used;
        out.shared_bytes = shared_used;
    } else {
        out.probe = "none";
        out.status = "unsupported";
        out.error = "per-process GPU memory counters are not available on this platform";
    }
    return out;
}

bool HasIntegratedUmaTarget(const std::vector<sonder::backends::llamacpp::DeviceInfo>& devices,
                            const std::vector<std::string>& configured_devices) {
    namespace lc = sonder::backends::llamacpp;
    for (const auto& d : devices) {
        if (d.kind == lc::DeviceKind::kIntegratedGpu && DeviceMatchesConfigured(d, configured_devices)) {
            return true;
        }
    }
    return false;
}

void PopulateGenerateStats(const sonder::backends::llamacpp::GenerateResult& result,
                           std::size_t restore_tokens, std::uint64_t load_ns,
                           GenerateStats& stats) noexcept {
    stats.load_ns = load_ns;
    stats.token_counts_from_backend = true;

    const std::uint64_t evaluated =
        result.prompt_tokens > 0 ? static_cast<std::uint64_t>(result.prompt_tokens) : 0u;
    const std::uint64_t reported_cached =
        result.cached_tokens > 0 ? static_cast<std::uint64_t>(result.cached_tokens) : 0u;
    const std::uint64_t cached =
        result.state_restored ? (restore_tokens > 0 ? static_cast<std::uint64_t>(restore_tokens) : reported_cached)
                              : reported_cached;
    const std::uint64_t total_prompt = SaturatingAddU64(evaluated, cached);
    const std::uint64_t generated =
        result.generated_tokens > 0 ? static_cast<std::uint64_t>(result.generated_tokens) : 0u;
    const std::uint64_t draft =
        result.draft_tokens > 0 ? static_cast<std::uint64_t>(result.draft_tokens) : 0u;
    const std::uint64_t draft_accepted = std::min(
        draft, result.draft_accepted_tokens > 0 ? static_cast<std::uint64_t>(result.draft_accepted_tokens) : 0u);

    // Total prompt tokens (evaluated + cached) in stats.prompt_tokens, cached
    // reuse in stats.cached_tokens, and strictly uncached evaluated prompt
    // tokens in backend_timings.prompt_n.
    stats.prompt_tokens = total_prompt;
    stats.cached_tokens = cached;
    if (draft > 0) {
        stats.draft_tokens = draft;
        stats.draft_accepted_tokens = draft_accepted;
    } else {
        stats.draft_tokens.reset();
        stats.draft_accepted_tokens.reset();
    }
    stats.completion_tokens = generated;

    const double prefill_ms = (std::isfinite(result.prefill_ms) && result.prefill_ms > 0.0) ? result.prefill_ms : 0.0;
    const double decode_ms = (std::isfinite(result.decode_ms) && result.decode_ms > 0.0) ? result.decode_ms : 0.0;
    stats.prompt_eval_ns = SafeMsToNs(prefill_ms);
    stats.eval_ns = SafeMsToNs(decode_ms);
    if (decode_ms > 0.0 && generated > 0) {
        const double tps = static_cast<double>(generated) * 1000.0 / decode_ms;
        if (std::isfinite(tps) && tps > 0.0) {
            stats.predicted_tokens_per_second = tps;
        } else {
            stats.predicted_tokens_per_second.reset();
        }
    } else {
        stats.predicted_tokens_per_second.reset();
    }

    BackendTimings timings;
    timings.prompt_n = evaluated;
    timings.cache_n = cached;
    timings.prompt_ms = prefill_ms;
    timings.predicted_n = generated;
    timings.predicted_ms = decode_ms;
    if (draft > 0) {
        timings.draft_n = draft;
        timings.draft_n_accepted = draft_accepted;
    }
    stats.backend_timings = timings;

    EffectiveExecutionReport eff;
    eff.configured_spec_method = result.configured_spec_method;
    eff.effective_spec_method = result.effective_spec_method;
    if (result.configured_draft_limit > 0 || !result.configured_spec_method.empty()) {
        eff.configured_draft_limit =
            static_cast<std::uint32_t>(std::max<std::int32_t>(0, result.configured_draft_limit));
    }
    if (result.effective_draft_limit > 0 || !result.effective_spec_method.empty()) {
        eff.effective_draft_limit =
            static_cast<std::uint32_t>(std::max<std::int32_t>(0, result.effective_draft_limit));
    }
    eff.fallback_reason = result.spec_fallback_reason;
    eff.draft_identity = result.draft_identity;
    eff.backend_version = "llama.cpp " + std::string(sonder::backends::llamacpp::LlamaCppBackend::UpstreamVersion());
    eff.committed_tokens = generated;
    if (draft > 0 || !result.effective_spec_method.empty()) {
        eff.proposed_draft_tokens = draft;
        eff.accepted_draft_tokens = draft_accepted;
    }
    if (draft > 0) {
        eff.speculation_observed_active = CapabilityState::yes;
    } else if (!result.effective_spec_method.empty()) {
        eff.speculation_observed_active = CapabilityState::no;
    } else {
        eff.speculation_observed_active = CapabilityState::unknown;
    }
    eff.transitioned_during_request = result.spec_transitioned_during_request;

    // Physical 3-tier cache breakdown (logical_cache_hit_tokens stays unset
    // here because Session owns engine KvCacheManager accounting).
    if (result.host_l2_restored_tokens > 0) {
        eff.cache.host_l2_restored_tokens = static_cast<std::uint64_t>(result.host_l2_restored_tokens);
        if (std::isfinite(result.host_l2_restore_ms) && result.host_l2_restore_ms >= 0.0) {
            eff.cache.host_l2_restore_ms = result.host_l2_restore_ms;
        }
        eff.cache.device_prefix_reused_tokens =
            static_cast<std::uint64_t>(std::max<std::int32_t>(0, result.device_prefix_reused_tokens));
        if (result.device_cross_slot_reused_tokens > 0) {
            eff.cache.device_cross_slot_reused_tokens =
                static_cast<std::uint64_t>(result.device_cross_slot_reused_tokens);
        }
    } else if (result.device_prefix_reused_tokens > 0) {
        eff.cache.device_prefix_reused_tokens = static_cast<std::uint64_t>(result.device_prefix_reused_tokens);
        if (result.device_cross_slot_reused_tokens > 0) {
            eff.cache.device_cross_slot_reused_tokens =
                static_cast<std::uint64_t>(result.device_cross_slot_reused_tokens);
        }
        eff.cache.host_l2_restored_tokens = 0u;
    } else if (result.state_restored) {
        eff.cache.device_prefix_reused_tokens = 0u;
        eff.cache.host_l2_restored_tokens = cached;
    } else {
        eff.cache.device_prefix_reused_tokens = reported_cached;
        eff.cache.host_l2_restored_tokens = 0u;
    }
    stats.effective_execution = bound_effective_execution_report(std::move(eff));

    if (result.stage_timings.enabled) {
        StageTimings st;
        st.enabled = true;
        st.cache_lookup_ms = result.stage_timings.cache_lookup_ms;
        st.l2_restore_ms = result.stage_timings.l2_restore_ms;
        st.prefill_ms = result.stage_timings.prefill_ms;
        st.draft_ms = result.stage_timings.draft_ms;
        st.verify_ms = result.stage_timings.verify_ms;
        st.rollback_ms = result.stage_timings.rollback_ms;
        st.sampling_ms = result.stage_timings.sampling_ms;
        st.stages_may_overlap = result.stage_timings.stages_may_overlap;
        stats.stage_timings = std::move(st);
    } else {
        stats.stage_timings.reset();
    }
}

}  // namespace llamacpp_detail

namespace {

namespace lc = sonder::backends::llamacpp;
namespace fs = std::filesystem;
using llamacpp_detail::ApplyNoThinkScaffolding;
using llamacpp_detail::PathFromUtf8;
using llamacpp_detail::PathToUtf8;
using llamacpp_detail::StopSequenceFilter;
using llamacpp_detail::ThinkingSplitResult;
using llamacpp_detail::ThinkingStreamSplitter;
using llamacpp_detail::ToLlamaSampling;

ErrorCode MapError(lc::ErrorCode code) {
    switch (code) {
        case lc::ErrorCode::kOk: return ErrorCode::ok;
        case lc::ErrorCode::kInvalidArgument: return ErrorCode::invalid_argument;
        case lc::ErrorCode::kFileNotFound: return ErrorCode::not_found;
        case lc::ErrorCode::kLoadFailed: return ErrorCode::backend_error;
        case lc::ErrorCode::kNotLoaded: return ErrorCode::invalid_state;
        case lc::ErrorCode::kTokenizeFailed: return ErrorCode::backend_error;
        case lc::ErrorCode::kContextOverflow: return ErrorCode::invalid_argument;
        case lc::ErrorCode::kDecodeFailed: return ErrorCode::backend_error;
        case lc::ErrorCode::kUnsupported: return ErrorCode::unsupported;
    }
    return ErrorCode::internal;
}

Status ToStatus(const lc::Status& s) { return s.ok() ? Status::success() : Status(MapError(s.code), s.message); }

void ForwardLogToDiagnosticQueue(const char* text) {
    if (text != nullptr) (void)diag::log(std::string(text));
}

std::uint64_t MsToNs(double ms) noexcept {
    if (!std::isfinite(ms) || ms <= 0.0) return 0;
    constexpr double kMaxMs = static_cast<double>(std::numeric_limits<std::uint64_t>::max()) / 1e6;
    if (ms >= kMaxMs) return std::numeric_limits<std::uint64_t>::max();
    return static_cast<std::uint64_t>(ms * 1e6);
}

std::uint64_t SaturatingAddBytes(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
std::uint32_t CurrentProcessId() noexcept {
#  if defined(_WIN32)
    return static_cast<std::uint32_t>(GetCurrentProcessId());
#  else
    const auto pid = ::getpid();
    return pid > 0 ? static_cast<std::uint32_t>(pid) : 0u;
#  endif
}
#endif

std::string FormatMiBText(std::uint64_t bytes) {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    return std::to_string(SaturatingAddBytes(bytes, kMiB / 2) / kMiB) + " MiB";
}

llamacpp_detail::ProcessGpuMemoryProbe MakeDefaultGpuMemoryProbe(std::vector<std::string> configured_devices) {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    auto os_source = llamaserver::make_gpu_counter_source();
#endif
    return [configured_devices = std::move(configured_devices)
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
            ,
            os_source = std::move(os_source)
#endif
    ]() -> llamacpp_detail::ProcessGpuMemoryReading {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        if (os_source && os_source->supported()) {
            const std::uint32_t pid = CurrentProcessId();
            const auto counters = os_source->read(pid);
            if (counters.ok()) {
                const auto sample = llamaserver::summarize_gpu_counters(pid, counters.value());
                llamacpp_detail::ProcessGpuMemoryReading reading;
                reading.probe = os_source->name();
                reading.status = "ok";
                reading.dedicated_bytes = sample.dedicated_bytes;
                reading.shared_bytes = sample.shared_bytes;
                return reading;
            }
            if (counters.status().code() != ErrorCode::unsupported) {
                llamacpp_detail::ProcessGpuMemoryReading reading;
                reading.probe = os_source->name();
                reading.status = "error";
                reading.error = counters.status().message();
                return reading;
            }
        }
#endif
        return llamacpp_detail::SummarizeGgmlDeviceMemory(lc::LlamaCppBackend::EnumerateDevices(), configured_devices);
    };
}

struct ActiveModelEntry {
    std::uint64_t id = 0;
    std::uint32_t n_ctx = 0;
    bool gpu_offload = false;
    bool uma_target = false;
    std::string target_identity;
    lc::SpeculativeLoadInfo spec_info;
};

struct RuntimeState {
    static constexpr std::uint64_t kDefaultSpillThresholdBytes = 256ull * 1024ull * 1024ull;

    mutable std::mutex mutex;
    llamacpp_detail::ProcessGpuMemoryProbe probe;
    std::chrono::milliseconds sample_interval{1000};
    std::uint32_t configured_ctx = 0;
    bool force_uma_mode = false;
    std::optional<bool> force_gpu_offload;

    // Configured direct-backend feature knobs for honest 5-state capability reporting.
    std::string configured_spec_type = "none";
    std::int32_t configured_spec_draft_max = 0;
    bool configured_prefix_cache = true;
    std::uint32_t configured_slots = 1;
    std::uint32_t configured_cache_ram_mib = 0;
    bool configured_mmproj = false;

    bool has_sampled = false;
    bool probe_in_flight = false;
    std::chrono::steady_clock::time_point last_sample_time{};
    GpuMemoryStatus gpu_memory;
    std::uint64_t next_model_id = 1;
    std::vector<ActiveModelEntry> active_models;

    // Observed runtime evidence (updated on actual request completions).
    bool observed_speculation_active = false;
    bool observed_device_prefix_reuse = false;
    bool observed_host_l2_restore = false;
    bool observed_concurrent_batch = false;
    std::optional<EffectiveExecutionReport> last_execution_report;
    std::shared_ptr<lc::DeviceHealthTracker> device_health;

    RuntimeState(llamacpp_detail::ProcessGpuMemoryProbe probe_fn, std::chrono::milliseconds interval,
                 const LlamaCppBackendOptions& opts, bool uma_override, std::optional<bool> gpu_offload_override)
        : probe(std::move(probe_fn)),
          sample_interval(interval.count() < 0 ? std::chrono::milliseconds{0} : interval),
          configured_ctx(opts.context_length),
          force_uma_mode(uma_override),
          force_gpu_offload(gpu_offload_override),
          configured_spec_type(opts.spec_type),
          configured_spec_draft_max(opts.spec_draft_max),
          configured_prefix_cache(opts.prefix_cache),
          configured_slots(opts.slots),
          configured_cache_ram_mib(opts.cache_ram_mib),
          configured_mmproj(!opts.mmproj_path.empty()),
          device_health(std::make_shared<lc::DeviceHealthTracker>(kLlamaCppBackendName)) {
        gpu_memory.probe = "none";
        gpu_memory.status = "not_sampled";
        gpu_memory.spill_threshold_bytes = kDefaultSpillThresholdBytes;
        gpu_memory.shared_baseline_bytes = 0;
    }

    std::shared_ptr<lc::DeviceHealthTracker> DeviceHealth() const {
        std::lock_guard<std::mutex> lock(mutex);
        return device_health;
    }

    void SetDeviceHealth(std::shared_ptr<lc::DeviceHealthTracker> dh) {
        std::lock_guard<std::mutex> lock(mutex);
        device_health = std::move(dh);
    }

    std::uint64_t RegisterModel(std::uint32_t n_ctx, bool gpu_offload, bool uma_target, std::string target_identity,
                                lc::SpeculativeLoadInfo spec_info) {
        std::lock_guard<std::mutex> lock(mutex);
        const std::uint64_t id = next_model_id++;
        active_models.push_back(
            ActiveModelEntry{id, n_ctx, gpu_offload, uma_target, std::move(target_identity), std::move(spec_info)});
        return id;
    }

    void UnregisterModel(std::uint64_t id) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        active_models.erase(std::remove_if(active_models.begin(), active_models.end(),
                                           [id](const ActiveModelEntry& e) noexcept { return e.id == id; }),
                            active_models.end());
    }

    void RecordExecution(const EffectiveExecutionReport& report) {
        std::lock_guard<std::mutex> lock(mutex);
        if (report.speculation_observed_active == CapabilityState::yes) {
            observed_speculation_active = true;
        }
        if (report.cache.device_prefix_reused_tokens.has_value() && *report.cache.device_prefix_reused_tokens > 0u) {
            observed_device_prefix_reuse = true;
        }
        if (report.cache.host_l2_restored_tokens.has_value() && *report.cache.host_l2_restored_tokens > 0u) {
            observed_host_l2_restore = true;
        }
        if (report.fallback_reason == kSpecReasonConcurrencyBatch) {
            observed_concurrent_batch = true;
        }
        last_execution_report = bound_effective_execution_report(report);
    }

    std::vector<CapabilityStatusEntry> BuildCapabilityStatusesLocked() const {
        const ActiveModelEntry* latest = active_models.empty() ? nullptr : &active_models.back();
        const bool spec_configured = !configured_spec_type.empty() && configured_spec_type != "none";
        const bool mtp_requested = configured_spec_type == "draft-mtp";
        const bool mtp_build = lc::LlamaCppBackend::HasMtpBuildSupport();
        const bool mtmd_build = lc::LlamaCppBackend::HasMtmdBuildSupport();

        std::vector<CapabilityStatusEntry> out;
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            if (mtp_requested) {
                s.build_supports = capability_state_from_bool(mtp_build);
                if (!mtp_build) {
                    s.note = "mtp_not_in_build";
                } else if (latest != nullptr && !latest->spec_info.mtp_model_supported) {
                    s.note = "mtp_not_in_model";
                }
            } else {
                s.build_supports = CapabilityState::yes;
            }
            s.configured = capability_state_from_bool(spec_configured);
            if (observed_speculation_active) {
                s.observed_active = CapabilityState::yes;
            } else if (!spec_configured) {
                s.observed_active = CapabilityState::no;
            } else if (latest != nullptr && latest->spec_info.effective_method == "none") {
                s.observed_active = CapabilityState::no;
                if (s.note.empty() && !latest->spec_info.fallback_reason.empty()) {
                    s.note = latest->spec_info.fallback_reason;
                }
            } else {
                s.observed_active = CapabilityState::unknown;
            }
            s.qualified_beneficial = CapabilityState::unknown;
            out.emplace_back("speculative_decode", std::move(s));
        }
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            s.build_supports = CapabilityState::yes;
            s.configured = capability_state_from_bool(configured_prefix_cache);
            if (observed_device_prefix_reuse) {
                s.observed_active = CapabilityState::yes;
            } else if (!configured_prefix_cache) {
                s.observed_active = CapabilityState::no;
            } else {
                s.observed_active = CapabilityState::unknown;
            }
            s.qualified_beneficial = CapabilityState::unknown;
            out.emplace_back("prefix_reuse", std::move(s));
        }
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            s.build_supports = CapabilityState::yes;
            s.configured = capability_state_from_bool(configured_cache_ram_mib > 0u);
            if (observed_host_l2_restore) {
                s.observed_active = CapabilityState::yes;
            } else if (configured_cache_ram_mib == 0u) {
                s.observed_active = CapabilityState::no;
            } else {
                s.observed_active = CapabilityState::unknown;
            }
            s.qualified_beneficial = CapabilityState::unknown;
            out.emplace_back("host_l2_cache", std::move(s));
        }
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            s.build_supports = CapabilityState::yes;
            s.configured = capability_state_from_bool(configured_slots > 1u);
            if (observed_concurrent_batch) {
                s.observed_active = CapabilityState::yes;
            } else if (configured_slots <= 1u) {
                s.observed_active = CapabilityState::no;
            } else {
                s.observed_active = CapabilityState::unknown;
            }
            s.qualified_beneficial = CapabilityState::unknown;
            out.emplace_back("continuous_batch_decode", std::move(s));
        }
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            s.build_supports = capability_state_from_bool(mtmd_build);
            s.configured = capability_state_from_bool(configured_mmproj);
            s.observed_active = (configured_mmproj && mtmd_build) ? CapabilityState::unknown : CapabilityState::no;
            s.qualified_beneficial = CapabilityState::unknown;
            if (!mtmd_build) s.note = "libmtmd_not_in_build";
            out.emplace_back("multimodal", std::move(s));
        }
        for (const char* cap_name :
             {"structured_output", "embeddings", "reranking", "logprobs", "recurrent_checkpoint",
              "execution_checkpoint", "selective_preemption"}) {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::yes;
            s.build_supports = CapabilityState::yes;
            s.configured = CapabilityState::yes;
            s.observed_active = CapabilityState::unknown;
            s.qualified_beneficial = CapabilityState::unknown;
            out.emplace_back(cap_name, std::move(s));
        }
        {
            CapabilityStatus s;
            s.adapter_supports = CapabilityState::no;
            s.build_supports = CapabilityState::no;
            s.configured = CapabilityState::no;
            s.observed_active = CapabilityState::no;
            s.qualified_beneficial = CapabilityState::unknown;
            s.note = "server_prompt_injected_tools";
            out.emplace_back("tool_calling", std::move(s));
        }
        return out;
    }

    std::vector<CapabilityStatusEntry> CapabilityStatuses() const {
        std::lock_guard<std::mutex> lock(mutex);
        return BuildCapabilityStatusesLocked();
    }

    EffectiveExecutionReport BuildEffectiveExecutionLocked() const {
        if (last_execution_report.has_value()) {
            return *last_execution_report;
        }
        EffectiveExecutionReport eff;
        eff.backend_version = "llama.cpp " + std::string(lc::LlamaCppBackend::UpstreamVersion());
        eff.configured_spec_method = configured_spec_type.empty() ? "none" : configured_spec_type;
        if (configured_spec_draft_max > 0) {
            eff.configured_draft_limit = static_cast<std::uint32_t>(configured_spec_draft_max);
        }
        if (!active_models.empty()) {
            const ActiveModelEntry& latest = active_models.back();
            eff.target_identity = latest.target_identity;
            eff.configured_spec_method = latest.spec_info.configured_method;
            eff.effective_spec_method = latest.spec_info.effective_method;
            eff.configured_draft_limit =
                static_cast<std::uint32_t>(std::max<std::int32_t>(0, latest.spec_info.configured_draft_max));
            eff.effective_draft_limit =
                static_cast<std::uint32_t>(std::max<std::int32_t>(0, latest.spec_info.effective_draft_max));
            eff.fallback_reason = latest.spec_info.fallback_reason;
            eff.draft_identity = latest.spec_info.draft_identity;
            eff.speculation_observed_active =
                (latest.spec_info.effective_method == "none") ? CapabilityState::no : CapabilityState::unknown;
        } else if (eff.configured_spec_method == "none") {
            eff.effective_spec_method = "none";
            eff.configured_draft_limit = 0u;
            eff.effective_draft_limit = 0u;
            eff.speculation_observed_active = CapabilityState::no;
        } else {
            // Model not yet loaded: effective method is unknown until load resolves it.
            eff.speculation_observed_active = CapabilityState::unknown;
        }
        return bound_effective_execution_report(std::move(eff));
    }

    void RefreshGpuMemory(bool force) {
        llamacpp_detail::ProcessGpuMemoryProbe probe_fn;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!probe || probe_in_flight) return;
            const auto now = std::chrono::steady_clock::now();
            if (!force && has_sampled && sample_interval.count() > 0 && (now - last_sample_time) < sample_interval) {
                return;
            }
            probe_in_flight = true;
            probe_fn = probe;
        }

        struct InFlightGuard {
            RuntimeState& state;
            bool committed = false;
            ~InFlightGuard() noexcept {
                if (!committed) {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    state.probe_in_flight = false;
                }
            }
        } guard{*this};

        llamacpp_detail::ProcessGpuMemoryReading reading;
        try {
            reading = probe_fn();
        } catch (const std::exception& ex) {
            reading.probe = "none";
            reading.status = "error";
            reading.error = ex.what();
        } catch (...) {
            reading.probe = "none";
            reading.status = "error";
            reading.error = "GPU memory probe failed";
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            has_sampled = true;
            last_sample_time = std::chrono::steady_clock::now();
            gpu_memory.probe = reading.probe.empty() ? "none" : reading.probe;
            if (reading.status == "ok") {
                gpu_memory.status = "ok";
                gpu_memory.error.clear();
                gpu_memory.dedicated_bytes = reading.dedicated_bytes;
                gpu_memory.shared_bytes = reading.shared_bytes;
                gpu_memory.peak_shared_bytes = std::max(gpu_memory.peak_shared_bytes, reading.shared_bytes);
                ++gpu_memory.samples;
            } else if (reading.status == "unsupported") {
                gpu_memory.status = "unsupported";
                gpu_memory.error = reading.error;
            } else {
                gpu_memory.status = "error";
                gpu_memory.error = reading.error.empty() ? "GPU memory probe failed" : reading.error;
            }
            probe_in_flight = false;
            guard.committed = true;
        }
    }

    BackendRuntimeStatus Snapshot() {
        RefreshGpuMemory(/*force=*/false);
        std::lock_guard<std::mutex> lock(mutex);

        bool any_gpu_offload = false;
        bool any_uma = false;
        std::optional<std::uint64_t> fitted_ctx;
        for (auto it = active_models.rbegin(); it != active_models.rend(); ++it) {
            if (!fitted_ctx.has_value() && it->n_ctx > 0) {
                fitted_ctx = it->n_ctx;
            }
            any_gpu_offload = any_gpu_offload || it->gpu_offload;
            any_uma = any_uma || it->uma_target;
        }
        const bool gpu_offload = force_gpu_offload.value_or(any_gpu_offload);
        const bool uma_mode = force_uma_mode || any_uma;

        BackendRuntimeStatus status;
        status.gpu_memory = gpu_memory;
        const std::uint64_t spill_limit =
            SaturatingAddBytes(status.gpu_memory.shared_baseline_bytes, status.gpu_memory.spill_threshold_bytes);
        status.gpu_memory.spilled = (status.gpu_memory.status == "ok") && gpu_offload && !uma_mode &&
                                    (status.gpu_memory.shared_bytes > spill_limit);

        status.context.policy = "warn";
        if (configured_ctx > 0) {
            status.context.configured_ctx = configured_ctx;
        }
        status.context.fitted_ctx = fitted_ctx;
        status.context.fit_attempts = 0;
        status.context.outcome = "not_needed";

        if (status.gpu_memory.spilled) {
            std::vector<std::pair<std::string, std::string>> details{
                {"dedicated_bytes", std::to_string(status.gpu_memory.dedicated_bytes)},
                {"shared_bytes", std::to_string(status.gpu_memory.shared_bytes)},
                {"shared_baseline_bytes", std::to_string(status.gpu_memory.shared_baseline_bytes)},
                {"spill_threshold_bytes", std::to_string(status.gpu_memory.spill_threshold_bytes)}};
            if (fitted_ctx) {
                details.emplace_back("ctx", std::to_string(*fitted_ctx));
            }
            status.warnings.push_back(BackendWarning{
                "vram_spill", "warning", "gpu_probe",
                "llamacpp keeps " + FormatMiBText(status.gpu_memory.shared_bytes) +
                    " in shared system memory (threshold " + FormatMiBText(status.gpu_memory.spill_threshold_bytes) +
                    " above a " + FormatMiBText(status.gpu_memory.shared_baseline_bytes) +
                    " baseline): VRAM overflowed into system RAM; lower context_length or KV cache precision",
                std::move(details), 1});
        }
        status.capability_statuses = BuildCapabilityStatusesLocked();
        status.effective_execution = BuildEffectiveExecutionLocked();
        if (device_health) {
            status.take_events = [dh = device_health] { return dh->DrainTelemetryEvents(); };
        }
        return status;
    }
};

// Copies the context and placement options into `lo` and validates them with
// the wrapper's pure validators (the same checks Load() runs).
Status ApplyBackendOptions(const LlamaCppBackendOptions& o, lc::LoadOptions& lo) {
    lo.n_ctx = o.context_length;
    lo.n_batch = o.batch_size;
    lo.n_ubatch = o.ubatch_size;
    lo.n_threads = o.threads;
    lo.devices = o.devices;
    lo.main_gpu = o.main_gpu;
    lo.tensor_split = o.tensor_split;
    lo.use_mlock = o.use_mlock;
    lo.use_mmap = o.use_mmap;
    lo.kv_unified = o.kv_unified;
    lo.prefix_cache = o.prefix_cache;
    lo.prefix_coalesce = o.prefix_coalesce;
    lo.min_coalesce_prefix_tokens = o.min_coalesce_prefix_tokens;
    lo.max_followers_per_producer = o.max_followers_per_producer;
    lo.max_coalesce_wait_ms = o.max_coalesce_wait_ms;
    lo.n_slots = o.slots;
    lo.spec_type = o.spec_type;
    lo.spec_draft_max = o.spec_draft_max;
    lo.spec_verification_mode = o.spec_verification_mode;
    lo.cache_ram_mib = o.cache_ram_mib;
    lo.moe_cache_mib = o.moe_cache_mib;
    lo.moe_cache_cpu_share = o.moe_cache_cpu_share;
    lo.mmproj_path = o.mmproj_path;
    lo.swa_kv_mode = o.swa_kv_mode;
    lo.default_foreground_itl_ms = o.default_foreground_itl_ms;
    lo.spec_reactivation_cooldown_steps = o.spec_reactivation_cooldown_steps;
    lo.spec_batch_enabled = o.spec_batch_enabled;
    lo.spec_batch_max_slots = o.spec_batch_max_slots;
    lo.spec_batch_max_aggregate_draft_tokens = o.spec_batch_max_aggregate_draft_tokens;
    lo.embed_batch_tokens = o.embed_batch_tokens;
    lo.embed_max_sequences = o.embed_max_sequences;
    if (!lc::ParseSplitMode(o.split_mode, lo.split_mode)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown split mode '" + o.split_mode + "' (expected none, layer or row)");
    }
    if (!lc::ParseKvCacheType(o.kv_cache_type_k, lo.type_k)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown K cache type '" + o.kv_cache_type_k +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl, f8_e4m3, q2_0 or pq2_0)");
    }
    if (!lc::ParseKvCacheType(o.kv_cache_type_v, lo.type_v)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown V cache type '" + o.kv_cache_type_v +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl, f8_e4m3, q2_0 or pq2_0)");
    }
    if (!lc::ParseFlashAttention(o.flash_attention, lo.flash_attn)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown flash attention mode '" + o.flash_attention + "' (expected auto, on or off)");
    }
    if (lc::Status s = lc::ValidatePlacementOptions(lo); !s.ok()) {
        return ToStatus(s);
    }
    return ToStatus(lc::ValidateContextOptions(lo));
}

bool IsGguf(const fs::path& p) {
    std::string ext = PathToUtf8(p.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".gguf";
}

class LlamaCppModel final : public BackendModel {
public:
    LlamaCppModel(ModelDescriptor desc, lc::LlamaCppBackend runtime, double load_ms, std::uint32_t slots,
                  std::shared_ptr<RuntimeState> runtime_state = nullptr, std::uint64_t runtime_model_id = 0,
                  std::vector<std::string> gpu_devices = {})
        : desc_(std::move(desc)),
          runtime_(std::move(runtime)),
          load_ns_(MsToNs(load_ms)),
          slots_(std::max<std::uint32_t>(1u, slots)),
          runtime_state_(std::move(runtime_state)),
          runtime_model_id_(runtime_model_id),
          gpu_devices_(std::move(gpu_devices)) {
        if (runtime_state_ && !gpu_devices_.empty()) {
            if (auto dh = runtime_state_->DeviceHealth()) {
                lc::BindBackendDeviceHealth(&runtime_, std::move(dh), gpu_devices_);
            }
        }
        // The chat template is fixed once the model is loaded, so decide here
        // rather than under mutex_, which is held for a whole generation.
        std::string ignored;
        native_chat_ = runtime_.ApplyChatTemplate({lc::ChatTurn{"user", "x"}}, true, ignored).ok();
    }

    ~LlamaCppModel() override {
        lc::UnbindBackendDeviceHealth(&runtime_);
        if (runtime_state_ && runtime_model_id_ != 0) {
            runtime_state_->UnregisterModel(runtime_model_id_);
        }
    }

    [[nodiscard]] const ModelDescriptor& descriptor() const override { return desc_; }

    Result<std::vector<TokenId>> tokenize(std::string_view text) override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<lc::Token> tokens;
        if (lc::Status s = runtime_.Tokenize(text, /*add_special=*/true, tokens, /*parse_special=*/false); !s.ok()) {
            return ToStatus(s);
        }
        return tokens;
    }

    Result<std::string> detokenize(std::span<const TokenId> tokens) override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::string text;
        if (lc::Status s = runtime_.Detokenize(tokens, text); !s.ok()) {
            return ToStatus(s);
        }
        return text;
    }

    static lc::PoolingType ToLcPooling(EmbedPoolingType pooling) noexcept {
        switch (pooling) {
            case EmbedPoolingType::unspecified: return lc::PoolingType::kUnspecified;
            case EmbedPoolingType::none: return lc::PoolingType::kNone;
            case EmbedPoolingType::mean: return lc::PoolingType::kMean;
            case EmbedPoolingType::cls: return lc::PoolingType::kCls;
            case EmbedPoolingType::last: return lc::PoolingType::kLast;
            case EmbedPoolingType::rank: return lc::PoolingType::kRank;
        }
        return lc::PoolingType::kUnspecified;
    }

    static EmbedPoolingType FromLcPooling(lc::PoolingType pooling) noexcept {
        switch (pooling) {
            case lc::PoolingType::kNone: return EmbedPoolingType::none;
            case lc::PoolingType::kMean: return EmbedPoolingType::mean;
            case lc::PoolingType::kCls: return EmbedPoolingType::cls;
            case lc::PoolingType::kLast: return EmbedPoolingType::last;
            case lc::PoolingType::kRank: return EmbedPoolingType::rank;
            case lc::PoolingType::kUnspecified:
            default: return EmbedPoolingType::unspecified;
        }
    }

    Result<EmbedResult> embed(const EmbedRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (request.inputs.empty()) {
            return Status(ErrorCode::invalid_argument, "embed request inputs are empty");
        }
        if (request.pooling_type == EmbedPoolingType::rank) {
            return Status(ErrorCode::invalid_argument,
                          "llamacpp backend does not support pooling_type 'rank' for dense embeddings");
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::vector<lc::Token>> tokenized;
        tokenized.reserve(request.inputs.size());
        for (const std::string& input : request.inputs) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "embedding cancelled");
            std::vector<lc::Token> toks;
            if (lc::Status s = runtime_.Tokenize(input, /*add_special=*/true, toks, /*parse_special=*/false); !s.ok()) {
                return ToStatus(s);
            }
            if (toks.empty()) {
                return Status(ErrorCode::invalid_argument, "embed input produced no tokens");
            }
            tokenized.push_back(std::move(toks));
        }
        lc::EmbedOptions emb_opts;
        emb_opts.normalize = request.normalize;
        emb_opts.dimensions = request.dimensions;
        emb_opts.pooling = ToLcPooling(request.pooling_type);
        lc::EmbedOutput emb_out;
        const lc::Status st =
            runtime_.Embed(tokenized, emb_opts, emb_out, [&] { return cancel.cancelled(); });
        if (!st.ok()) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "embedding cancelled");
            if (Status dh = check_device_health(); !dh.ok()) return dh;
            return ToStatus(st);
        }
        EmbedResult result;
        result.embeddings = std::move(emb_out.embeddings);
        result.prompt_tokens = static_cast<std::uint64_t>(emb_out.prompt_tokens);
        result.eval_ns = MsToNs(emb_out.eval_ms);
        result.dimensions = emb_out.dimensions;
        result.pooling_type = FromLcPooling(emb_out.pooling);
        result.encoding_format = request.encoding_format.empty() ? std::string("float") : request.encoding_format;
        result.model = desc_.name;
        return result;
    }

    Result<RerankResult> rerank(const RerankRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (Status st = validate_rerank_request(request); !st.ok()) return st;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!runtime_.SupportsReranking()) {
            return Status(ErrorCode::unsupported,
                          "model '" + desc_.name + "' does not support reranking (RANK pooling is required)");
        }
        std::vector<std::vector<lc::Token>> tokenized_pairs;
        tokenized_pairs.reserve(request.documents.size());
        for (const std::string& doc : request.documents) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "reranking cancelled");
            std::vector<lc::Token> toks;
            if (lc::Status s = runtime_.FormatRerankPair(request.query, doc, toks); !s.ok()) {
                return ToStatus(s);
            }
            tokenized_pairs.push_back(std::move(toks));
        }
        lc::ScoreOutput score_out;
        const lc::Status st =
            runtime_.ScorePairs(tokenized_pairs, score_out, [&] { return cancel.cancelled(); });
        if (!st.ok()) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "reranking cancelled");
            if (Status dh = check_device_health(); !dh.ok()) return dh;
            return ToStatus(st);
        }
        if (score_out.scores.size() != request.documents.size()) {
            return Status(ErrorCode::backend_error, "llamacpp returned mismatched rerank score count");
        }
        RerankResult result;
        result.model = desc_.name;
        result.prompt_tokens = static_cast<std::uint64_t>(score_out.prompt_tokens);
        result.eval_ns = MsToNs(score_out.eval_ms);
        result.results.reserve(request.documents.size());
        for (std::size_t i = 0; i < request.documents.size(); ++i) {
            RerankItem item;
            item.index = i;
            item.relevance_score = score_out.scores[i];
            if (request.return_documents) {
                item.document = request.documents[i];
            }
            result.results.push_back(std::move(item));
        }
        finalize_rerank_items(result.results, request.top_n);
        return result;
    }

    Result<ScoreResult> score(const ScoreRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (Status st = validate_score_request(request); !st.ok()) return st;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!runtime_.SupportsReranking()) {
            return Status(ErrorCode::unsupported,
                          "model '" + desc_.name + "' does not support pair scoring (RANK pooling is required)");
        }
        std::vector<std::vector<lc::Token>> tokenized_pairs;
        tokenized_pairs.reserve(request.pairs.size());
        for (const auto& pair : request.pairs) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "pair scoring cancelled");
            std::vector<lc::Token> toks;
            if (lc::Status s = runtime_.FormatRerankPair(pair.text_1, pair.text_2, toks); !s.ok()) {
                return ToStatus(s);
            }
            tokenized_pairs.push_back(std::move(toks));
        }
        lc::ScoreOutput score_out;
        const lc::Status st =
            runtime_.ScorePairs(tokenized_pairs, score_out, [&] { return cancel.cancelled(); });
        if (!st.ok()) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "pair scoring cancelled");
            if (Status dh = check_device_health(); !dh.ok()) return dh;
            return ToStatus(st);
        }
        if (score_out.scores.size() != request.pairs.size()) {
            return Status(ErrorCode::backend_error, "llamacpp returned mismatched pair score count");
        }
        ScoreResult result;
        result.model = desc_.name;
        result.prompt_tokens = static_cast<std::uint64_t>(score_out.prompt_tokens);
        result.eval_ns = MsToNs(score_out.eval_ms);
        result.scores.reserve(request.pairs.size());
        for (std::size_t i = 0; i < request.pairs.size(); ++i) {
            result.scores.push_back(ScoreItem{i, score_out.scores[i]});
        }
        return result;
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (Status v = validate(request.sampling); !v.ok()) return v;
        // The llama.cpp context size is fixed at load (ModelLoadOptions::context_length).
        // A smaller num_ctx is honoured by the engine's scheduling limit; a larger one
        // cannot be served without reloading.
        if (request.sampling.num_ctx > 0 && desc_.context_length > 0 &&
            static_cast<std::uint64_t>(request.sampling.num_ctx) > desc_.context_length) {
            return Status(ErrorCode::invalid_argument,
                          "num_ctx " + std::to_string(request.sampling.num_ctx) +
                              " exceeds the loaded llama.cpp context (" + std::to_string(desc_.context_length) +
                              "); reload with a larger context_length");
        }
        // When n_slots > 1 and no atomic checkpoint exchange is requested,
        // take a shared_lock so concurrent HTTP requests enter Generate()
        // simultaneously and are packed by the backend's continuous-batching
        // worker loop. Keep unique_lock for n_slots == 1 or atomic checkpoints.
        const bool concurrent_batch = (slots_ > 1u) && (request.checkpoint == nullptr);
        std::shared_lock<std::shared_mutex> shared_guard;
        std::unique_lock<std::shared_mutex> exclusive_guard;
        if (concurrent_batch) {
            shared_guard = std::shared_lock<std::shared_mutex>(mutex_);
        } else {
            exclusive_guard = std::unique_lock<std::shared_mutex>(mutex_);
        }
        std::vector<lc::Token> prompt_tokens;
        if (lc::Status s = runtime_.Tokenize(request.prompt, /*add_special=*/true, prompt_tokens, /*parse_special=*/false);
            !s.ok()) {
            return ToStatus(s);
        }
        return run_tokens(std::move(prompt_tokens), request.sampling, cancel, on_chunk, request.session_key,
                          request.grammar, /*separate_reasoning=*/false, request.checkpoint, {}, {}, {},
                          request.priority_class, request.target_ttft_ms, request.target_itl_ms,
                          request.enable_stage_profiling, request.on_safe_point, request.suspended_state,
                          request.execution_resume, request.priority_rank, request.allow_preempt_spill);
    }

    // Chat through the GGUF chat template (llama_chat_apply_template). Models
    // without a template, or with one llama.cpp does not recognise, use the
    // generic BackendModel::chat() prompt format instead. Template scaffolding
    // is tokenized with parse_special=true while untrusted message content is
    // strictly tokenized with parse_special=false (G0028).
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (Status v = validate_chat_messages(request.messages); !v.ok()) return v;
        if (Status v = validate(request.sampling); !v.ok()) return v;
        if (request.sampling.num_ctx > 0 && desc_.context_length > 0 &&
            static_cast<std::uint64_t>(request.sampling.num_ctx) > desc_.context_length) {
            return Status(ErrorCode::invalid_argument,
                          "num_ctx " + std::to_string(request.sampling.num_ctx) +
                              " exceeds the loaded llama.cpp context (" + std::to_string(desc_.context_length) +
                              "); reload with a larger context_length");
        }
        const bool concurrent_batch = (slots_ > 1u);
        std::shared_lock<std::shared_mutex> shared_guard;
        std::unique_lock<std::shared_mutex> exclusive_guard;
        if (concurrent_batch) {
            shared_guard = std::shared_lock<std::shared_mutex>(mutex_);
        } else {
            exclusive_guard = std::unique_lock<std::shared_mutex>(mutex_);
        }
        llamacpp_detail::DirectChatPromptPlan plan = llamacpp_detail::BuildDirectChatPromptPlan(
            request,
            [&](const std::vector<lc::ChatTurn>& turns, bool add_assistant) -> Result<std::string> {
                std::string rendered;
                const lc::Status s = runtime_.ApplyChatTemplate(turns, add_assistant, rendered);
                if (!s.ok()) return ToStatus(s);
                return rendered;
            });
        if (!plan.status.ok()) {
            if (plan.template_unsupported) {
                if (shared_guard.owns_lock()) shared_guard.unlock();
                if (exclusive_guard.owns_lock()) exclusive_guard.unlock();
                if (!plan.media_blobs.empty()) {
                    return Status(ErrorCode::invalid_argument,
                                  "multimodal chat requires a model with a chat template");
                }
                return BackendModel::chat(request, cancel, on_chunk);
            }
            return plan.status;
        }
        std::vector<lc::Token> prompt_tokens;
        const Status tok_status = tokenize_chat_segments(
            plan.segments,
            [&](std::string_view piece, bool add_special, bool parse_special, std::vector<TokenId>& out) -> Status {
                return ToStatus(runtime_.Tokenize(piece, add_special, out, parse_special));
            },
            prompt_tokens);
        if (!tok_status.ok()) return tok_status;
        return run_tokens(std::move(prompt_tokens), request.sampling, cancel, on_chunk, request.session_key,
                          request.grammar, request.separate_reasoning, /*checkpoint=*/nullptr,
                          std::move(plan.media_blobs), {}, std::move(plan.media_prompt_parts),
                          request.priority_class, request.target_ttft_ms, request.target_itl_ms,
                          request.enable_stage_profiling, {}, nullptr, nullptr, 0, false,
                          request.separate_reasoning && llamacpp_detail::PromptEndsInsideThink(plan.segments));
    }

    [[nodiscard]] bool has_native_chat() const override { return native_chat_; }

    Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) override {
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (token_position == 0) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint token_position must be > 0");
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        RecurrentStateSnapshot snapshot;
        snapshot.token_position = token_position;
        if (lc::Status s = runtime_.CaptureSequenceState(snapshot.bytes, kMaxRecurrentCheckpointBytes); !s.ok()) {
            return ToStatus(s);
        }
        return snapshot;
    }

    Status restore_recurrent_state(std::span<const std::uint8_t> bytes) override {
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (bytes.empty()) {
            return Status(ErrorCode::invalid_argument, "recurrent checkpoint snapshot is empty");
        }
        if (bytes.size() > kMaxRecurrentCheckpointBytes) {
            return Status(ErrorCode::invalid_argument,
                          "recurrent checkpoint snapshot exceeds kMaxRecurrentCheckpointBytes");
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return ToStatus(runtime_.RestoreSequenceState(bytes, kMaxRecurrentCheckpointBytes));
    }

    Result<std::vector<std::uint8_t>> capture_execution_state(std::uint64_t token_position,
                                                              std::size_t max_bytes) override {
        (void)token_position;
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        const std::size_t cap = std::min(max_bytes, kMaxRecurrentCheckpointBytes);
        if (cap == 0) {
            return Status(ErrorCode::invalid_argument, "execution state max_bytes must be > 0");
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::uint8_t> bytes;
        if (lc::Status s = runtime_.CaptureSequenceState(bytes, cap); !s.ok()) {
            return ToStatus(s);
        }
        return bytes;
    }

    Status restore_execution_state(std::span<const std::uint8_t> bytes, std::uint64_t token_position) override {
        (void)token_position;
        if (Status dh = check_device_health(); !dh.ok()) return dh;
        if (bytes.empty()) {
            return Status(ErrorCode::invalid_argument, "execution state snapshot is empty");
        }
        if (bytes.size() > kMaxRecurrentCheckpointBytes) {
            return Status(ErrorCode::invalid_argument,
                          "execution state snapshot exceeds kMaxRecurrentCheckpointBytes");
        }
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return ToStatus(runtime_.RestoreSequenceState(bytes, kMaxRecurrentCheckpointBytes));
    }

private:
    // Caller holds mutex_. Prompt tokens are already produced across the
    // trusted/untrusted tokenization boundary.
    Result<GenerateStats> run_tokens(std::vector<lc::Token> prompt_tokens, const SamplingConfig& sampling,
                                     const CancellationToken& cancel, const TokenCallback& on_chunk,
                                     std::string_view session_key = {}, std::string_view grammar = {},
                                     bool separate_reasoning = false,
                                     RecurrentCheckpointExchange* checkpoint = nullptr,
                                     std::vector<std::vector<std::uint8_t>> media_blobs = {},
                                     std::string media_prompt_text = {},
                                     std::vector<lc::GenerateRequest::MediaPromptPart> media_prompt_parts = {},
                                     std::optional<RequestPriority> priority_class = std::nullopt,
                                     std::optional<double> target_ttft_ms = std::nullopt,
                                     std::optional<double> target_itl_ms = std::nullopt,
                                     bool enable_stage_profiling = false,
                                     const SuspendSafePointCallback& on_safe_point = {},
                                     SuspendedExecutionState* suspended_state = nullptr,
                                     const ExecutionResumeState* execution_resume = nullptr,
                                     int priority_rank = 0, bool allow_preempt_spill = false,
                                     bool start_in_thinking = false) {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");

        lc::GenerateRequest req;
        req.prompt = std::move(prompt_tokens);
        if (req.prompt.empty()) return Status(ErrorCode::invalid_argument, "prompt produced no tokens");
        req.max_new_tokens = sampling.max_tokens;
        req.sampling = ToLlamaSampling(sampling);
        req.sampling.grammar = std::string(grammar);
        req.session_key = std::string(session_key);
        req.media_blobs = std::move(media_blobs);
        req.media_prompt_text = std::move(media_prompt_text);
        req.media_prompt_parts = std::move(media_prompt_parts);
        req.priority = priority_class.value_or(RequestPriority::subagent);
        req.target_ttft_ms = target_ttft_ms;
        req.target_itl_ms = target_itl_ms;
        req.enable_stage_profiling = enable_stage_profiling;
        req.priority_rank = priority_rank;
        req.allow_preempt_spill = allow_preempt_spill;
        if (execution_resume != nullptr && !execution_resume->generated_token_ids.empty()) {
            req.prompt.insert(req.prompt.end(), execution_resume->generated_token_ids.begin(),
                              execution_resume->generated_token_ids.end());
            if (!execution_resume->backend_state_blob.empty() && execution_resume->token_position > 0 &&
                execution_resume->token_position < req.prompt.size()) {
                req.restore_state = execution_resume->backend_state_blob;
                req.restore_tokens = static_cast<std::size_t>(execution_resume->token_position);
            }
        }
        if (on_safe_point) {
            req.on_safe_point = [&on_safe_point](const lc::SafePointInfo& info) -> lc::SafePointDecision {
                SafePointContext ctx;
                ctx.prompt_tokens = info.prompt_tokens;
                ctx.prefill_completed_tokens = info.prefill_completed_tokens;
                ctx.generated_tokens = static_cast<std::uint64_t>(std::max<std::int32_t>(0, info.generated_tokens));
                ctx.kv_tokens = info.kv_pos > 0 ? static_cast<std::uint64_t>(info.kv_pos) : 0;
                ctx.active_peers = info.active_peers;
                ctx.in_prefill = info.in_prefill;
                const SuspendDecision d = on_safe_point(ctx);
                lc::SafePointDecision out;
                out.max_spill_bytes = std::min(d.max_spill_bytes, lc::kMaxSequenceStateBytes);
                switch (d.action) {
                    case SuspendAction::continue_running: out.action = lc::SuspendAction::kContinue; break;
                    case SuspendAction::yield_recompute: out.action = lc::SuspendAction::kYieldRecompute; break;
                    case SuspendAction::yield_spill: out.action = lc::SuspendAction::kYieldSpill; break;
                }
                return out;
            };
        }
        if (checkpoint != nullptr) {
            if (!checkpoint->restore_bytes.empty() && checkpoint->restore_token_position > 0 &&
                checkpoint->restore_token_position < req.prompt.size()) {
                req.restore_state = checkpoint->restore_bytes;
                req.restore_tokens = static_cast<std::size_t>(checkpoint->restore_token_position);
                if (checkpoint->on_restored) {
                    req.on_state_restored = [checkpoint] {
                        checkpoint->restored = true;
                        checkpoint->on_restored();
                    };
                }
            }
            if (checkpoint->capture_at_token_position > 0 &&
                checkpoint->capture_at_token_position <= req.prompt.size()) {
                req.capture_at_tokens = static_cast<std::size_t>(checkpoint->capture_at_token_position);
                req.max_capture_bytes = kMaxRecurrentCheckpointBytes;
                req.on_state_captured = [checkpoint](std::vector<std::uint8_t> bytes) {
                    checkpoint->captured = true;
                    if (checkpoint->on_captured) {
                        checkpoint->on_captured(checkpoint->capture_at_token_position, std::move(bytes));
                    } else {
                        checkpoint->captured_bytes = std::move(bytes);
                    }
                };
            }
        }

        GenerateStats stats;
        stats.load_ns = load_ns_;
        stats.token_counts_from_backend = true;
        StopSequenceFilter stop(sampling.stop);
        if (execution_resume != nullptr && !execution_resume->stop_filter_pending.empty()) {
            stop.RestoreState(execution_resume->stop_filter_pending, execution_resume->stop_filter_stopped);
        }
        ThinkingStreamSplitter thinking_splitter(start_in_thinking);
        bool callback_stopped = false;

        auto deliver_split = [&](std::string_view text_part, std::string_view reasoning_part,
                                 std::optional<float> logprob = std::nullopt,
                                 std::vector<TopLogprob> top_logprobs = {}) {
            if (!reasoning_part.empty() && !callback_stopped) {
                if (on_chunk && !on_chunk(TokenChunk{std::string_view{}, stats.chunks, reasoning_part})) {
                    callback_stopped = true;
                }
                ++stats.chunks;
            }
            if (!text_part.empty() && !callback_stopped) {
                TokenChunk chunk{text_part, stats.chunks, std::string_view{}};
                chunk.logprob = logprob;
                chunk.top_logprobs = std::move(top_logprobs);
                if (on_chunk && !on_chunk(chunk)) {
                    callback_stopped = true;
                }
                ++stats.chunks;
            }
        };

        auto deliver = [&](std::string_view text, std::optional<float> logprob = std::nullopt,
                           std::vector<TopLogprob> top_logprobs = {}) {
            if (text.empty() || callback_stopped) return;
            if (!separate_reasoning) {
                TokenChunk chunk{text, stats.chunks};
                chunk.logprob = logprob;
                chunk.top_logprobs = std::move(top_logprobs);
                if (on_chunk && !on_chunk(chunk)) callback_stopped = true;
                ++stats.chunks;
                return;
            }
            const ThinkingSplitResult split = thinking_splitter.Push(text);
            deliver_split(split.text, split.reasoning, logprob, std::move(top_logprobs));
        };

        const lc::GenerateResult r = runtime_.Generate(
            req,
            [&](const lc::TokenEvent& ev) {
                std::optional<float> lp = ev.has_logprob ? std::optional<float>(ev.logprob) : std::nullopt;
                std::vector<TopLogprob> top_lps;
                if (ev.has_logprob && !ev.top_logprobs.empty()) {
                    top_lps.reserve(ev.top_logprobs.size());
                    for (const auto& [tok, tok_lp] : ev.top_logprobs) {
                        std::string piece;
                        const lc::Token arr[1] = {tok};
                        if (!runtime_.Detokenize(std::span<const lc::Token>(arr, 1), piece).ok()) {
                            piece = "<tok:" + std::to_string(tok) + ">";
                        }
                        top_lps.push_back(TopLogprob{std::move(piece), tok_lp});
                    }
                }
                deliver(stop.Push(ev.text), lp, std::move(top_lps));
                return !callback_stopped && !stop.matched();
            },
            [&] { return cancel.cancelled(); });

        if (checkpoint != nullptr) {
            checkpoint->restored = r.state_restored;
            if (!r.restore_status.ok()) {
                checkpoint->restore_error = ToStatus(r.restore_status);
            }
            checkpoint->captured = r.state_captured;
            if (!r.capture_status.ok()) {
                checkpoint->capture_error = ToStatus(r.capture_status);
            }
        }

        llamacpp_detail::PopulateGenerateStats(r, req.restore_tokens, load_ns_, stats);
        if (stats.effective_execution) {
            stats.effective_execution->target_identity = desc_.name;
            if (runtime_state_) {
                runtime_state_->RecordExecution(*stats.effective_execution);
            }
        }

        if (!r.status.ok()) {
            if (Status dh = check_device_health(); !dh.ok()) return dh;
            return ToStatus(r.status);
        }

        switch (r.stop_reason) {
            case lc::StopReason::kEndOfGeneration: stats.stop_reason = StopReason::end_of_sequence; break;
            case lc::StopReason::kMaxTokens: stats.stop_reason = StopReason::max_tokens; break;
            case lc::StopReason::kContextFull: stats.stop_reason = StopReason::max_tokens; break;
            case lc::StopReason::kSuspended:
                stats.stop_reason = StopReason::suspended;
                if (suspended_state != nullptr) {
                    suspended_state->action = r.suspend_action == lc::SuspendAction::kYieldSpill
                                                  ? SuspendAction::yield_spill
                                                  : SuspendAction::yield_recompute;
                    suspended_state->token_position = r.suspended_token_position;
                    suspended_state->generated_tokens =
                        static_cast<std::uint64_t>(std::max<std::int32_t>(0, r.generated_tokens));
                    suspended_state->tokens = r.suspended_tokens;
                    suspended_state->backend_state_blob = r.suspended_state_blob;
                    suspended_state->stop_filter_pending = stop.held();
                    suspended_state->stop_filter_stopped = stop.matched();
                    suspended_state->physical_slot_reclaimed = r.physical_slot_reclaimed;
                }
                break;
            case lc::StopReason::kCancelled:
                if (stop.matched()) {
                    stats.stop_reason = StopReason::stop_sequence;
                } else if (callback_stopped) {
                    stats.stop_reason = StopReason::callback;
                } else {
                    return Status(ErrorCode::cancelled, "generation cancelled");
                }
                break;
            default: stats.stop_reason = StopReason::error; break;
        }
        if (stats.stop_reason == StopReason::stop_sequence) {
            if (const auto matched = stop.matched_stop()) {
                stats.matched_stop = std::string(*matched);
            }
        }
        if (stats.stop_reason != StopReason::stop_sequence && stats.stop_reason != StopReason::callback &&
            stats.stop_reason != StopReason::suspended) {
            deliver(stop.Flush());
            if (separate_reasoning && !callback_stopped) {
                const ThinkingSplitResult tail = thinking_splitter.Flush();
                deliver_split(tail.text, tail.reasoning);
            }
        } else if (stats.stop_reason == StopReason::stop_sequence && separate_reasoning && !callback_stopped) {
            const ThinkingSplitResult tail = thinking_splitter.Flush();
            deliver_split(tail.text, tail.reasoning);
        }
        return stats;
    }

    [[nodiscard]] Status check_device_health() const {
        if (!runtime_state_ || gpu_devices_.empty()) {
            return Status::success();
        }
        const auto dh = runtime_state_->DeviceHealth();
        return dh ? dh->CheckDevicesStatus(gpu_devices_) : Status::success();
    }

    ModelDescriptor desc_;
    lc::LlamaCppBackend runtime_;
    std::uint64_t load_ns_ = 0;
    std::uint32_t slots_ = 1;
    std::shared_ptr<RuntimeState> runtime_state_;
    std::uint64_t runtime_model_id_ = 0;
    std::vector<std::string> gpu_devices_;
    bool native_chat_ = false;  // set in the constructor, read-only afterwards
    mutable std::shared_mutex mutex_;
};

class LlamaCppCoreBackend final : public Backend {
public:
    explicit LlamaCppCoreBackend(LlamaCppBackendOptions options,
                                 llamacpp_detail::ProcessGpuMemoryProbe probe = {},
                                 std::chrono::milliseconds sample_interval = std::chrono::milliseconds{1000},
                                 bool force_uma_mode = false,
                                 std::optional<bool> force_gpu_offload = std::nullopt)
        : options_(std::move(options)),
          runtime_state_(std::make_shared<RuntimeState>(
              probe ? std::move(probe) : MakeDefaultGpuMemoryProbe(options_.devices), sample_interval, options_,
              force_uma_mode, force_gpu_offload)) {
        lc::BindCoreBackendDeviceHealth(
            this, runtime_state_->DeviceHealth(),
            [rs = runtime_state_](std::shared_ptr<lc::DeviceHealthTracker> dh) {
                if (dh) rs->SetDeviceHealth(std::move(dh));
            });
        // llama.cpp's log callback runs on inference threads: send its lines
        // through the non-blocking diagnostic queue, never to stderr directly.
        lc::SetLogSink(&ForwardLogToDiagnosticQueue);
    }

    ~LlamaCppCoreBackend() override { lc::UnbindCoreBackendDeviceHealth(this); }

    [[nodiscard]] std::string name() const override { return kLlamaCppBackendName; }
    [[nodiscard]] std::string description() const override {
        return "direct llama.cpp/GGML (" + std::string(lc::LlamaCppBackend::UpstreamVersion()) + "), in-process GGUF";
    }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        // Intentionally omit Capability::tool_calling: direct llama.cpp renders
        // chat via llama_chat_apply_template() and executes tool constraints when
        // the server layer synthesizes tool prompts + GBNF grammars on top of
        // Capability::structured_output (so health reports native_tools=true and
        // backend_native_tools=false accurately).
        caps.add(Capability::tokenization)
            .add(Capability::streaming)
            .add(Capability::batched_prefill)
            .add(Capability::prefix_reuse)
            .add(Capability::speculative_decode)
            .add(Capability::structured_output)
            .add(Capability::embeddings)
            .add(Capability::deterministic)
            .add(Capability::recurrent_checkpoint)
            .add(Capability::logprobs)
            .add(Capability::reranking)
            .add(Capability::execution_checkpoint)
            .add(Capability::selective_preemption);
        if (options_.slots > 1u) {
            caps.add(Capability::continuous_batch_decode);
        }
        if (!options_.mmproj_path.empty()) {
            caps.add(Capability::multimodal);
        }
        return caps;
    }
    [[nodiscard]] std::optional<BackendRuntimeStatus> runtime_status() const override {
        return runtime_state_->Snapshot();
    }
    [[nodiscard]] std::vector<CapabilityStatusEntry> capability_statuses() const override {
        return runtime_state_->CapabilityStatuses();
    }
    [[nodiscard]] std::size_t max_concurrent_requests() const override {
        return std::max<std::size_t>(1u, static_cast<std::size_t>(options_.slots));
    }

    Result<std::string> probe() override {
        std::string devices;
        for (const auto& d : lc::LlamaCppBackend::EnumerateDevices()) {
            if (!devices.empty()) devices += ", ";
            devices += d.name + "(" + std::string(lc::ToString(d.kind)) + ")";
        }
        return "llama.cpp " + std::string(lc::LlamaCppBackend::UpstreamVersion()) + " [" + devices + "]";
    }

    Result<std::vector<ModelDescriptor>> list_models() override {
        std::vector<ModelDescriptor> out;
        for (const auto& dir : options_.model_dirs) {
            std::error_code ec;
            for (fs::directory_iterator it(PathFromUtf8(dir), ec), end; !ec && it != end; it.increment(ec)) {
                if (!it->is_regular_file(ec) || !IsGguf(it->path())) continue;
                ModelDescriptor d;
                d.name = PathToUtf8(it->path().filename());
                d.backend = kLlamaCppBackendName;
                d.format = "gguf";
                d.size_bytes = it->file_size(ec);
                out.push_back(std::move(d));
            }
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
        return out;
    }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        if (options.model.empty()) return Status(ErrorCode::invalid_argument, "model path is empty");
        const std::string path = Resolve(options.model);

        lc::LoadOptions lo;
        lo.model_path = path;
        if (Status s = ApplyBackendOptions(options_, lo); !s.ok()) return s;
        for (const auto& [pattern, device] : options_.tensor_overrides) {
            lo.tensor_overrides.push_back({pattern, device});
        }
        const std::string_view dev = options.device_id;
        const auto health = runtime_state_ ? runtime_state_->DeviceHealth() : nullptr;
        if (health) {
            if (Status dh = health->CheckDevicesStatus(lc::CollectInitialGpuDevices(lo, dev)); !dh.ok()) {
                return dh;
            }
        }
        bool has_gpu = false;
        std::vector<lc::DeviceInfo> enumerated_devices;
        if (dev.rfind("gpu", 0) == 0) {
            enumerated_devices = lc::LlamaCppBackend::EnumerateDevices();
            for (const auto& d : enumerated_devices) {
                has_gpu = has_gpu || d.kind == lc::DeviceKind::kGpu || d.kind == lc::DeviceKind::kIntegratedGpu;
            }
        }
        const auto gpu_layers = llamacpp_detail::ResolveGpuLayers(options_, dev, has_gpu);
        if (!gpu_layers.ok()) return gpu_layers.status();
        lo.n_gpu_layers = gpu_layers.value();
        if (lo.n_gpu_layers != 0) {
            if (enumerated_devices.empty() && lo.devices.empty()) {
                enumerated_devices = lc::LlamaCppBackend::EnumerateDevices();
            }
            const std::vector<GpuDeviceInfo> os_gpus =
                (dev.rfind("gpu:", 0) == 0 && lo.devices.empty() && lo.tensor_split.empty())
                    ? enumerate_gpu_devices()
                    : std::vector<GpuDeviceInfo>{};
            if (Status s = llamacpp_detail::ResolveGpuDevicePin(dev, enumerated_devices, lo.devices, lo.main_gpu,
                                                                lo.tensor_split, os_gpus);
                !s.ok()) {
                return s;
            }
        }
        std::vector<std::string> active_gpu_devices = lc::CollectActiveGpuDevices(lo, enumerated_devices, dev);
        if (health && !active_gpu_devices.empty()) {
            if (dev.rfind("gpu:", 0) == 0 && active_gpu_devices.size() == 1u) {
                (void)health->RegisterAlias(dev, active_gpu_devices[0]);
            }
            if (Status dh = health->CheckDevicesStatus(active_gpu_devices); !dh.ok()) {
                return dh;
            }
        }

        lc::LlamaCppBackend runtime;
        if (health && !active_gpu_devices.empty()) {
            lc::BindBackendDeviceHealth(&runtime, health, active_gpu_devices);
        }
        double load_ms = 0.0;
        runtime.SetTelemetrySink([&](const lc::TelemetryEvent& e) {
            if (e.kind == lc::TelemetryKind::kModelLoaded) load_ms = e.elapsed_ms;
        });
        if (lc::Status s = runtime.Load(lo); !s.ok()) {
            if (health && !active_gpu_devices.empty()) {
                if (Status dh = health->CheckDevicesStatus(active_gpu_devices); !dh.ok()) {
                    return dh;
                }
            }
            return ToStatus(s);
        }
        runtime.SetTelemetrySink({});

        const lc::ModelInfo info = runtime.GetModelInfo();
        ModelDescriptor d;
        d.name = PathToUtf8(PathFromUtf8(path).filename());
        d.backend = kLlamaCppBackendName;
        d.format = "gguf";
        d.family = info.description;  // e.g. "llama 7B Q4_0"
        d.architecture = info.architecture;
        d.size_bytes = info.size_bytes;
        d.context_length = info.n_ctx;
        const std::size_t sp = info.description.find(' ');
        if (sp != std::string::npos) {
            d.family = info.description.substr(0, sp);
            const std::string rest = info.description.substr(sp + 1);
            const std::size_t sp2 = rest.find(' ');
            d.parameter_size = rest.substr(0, sp2);
            if (sp2 != std::string::npos) d.quantization = rest.substr(sp2 + 1);
        }
        bool uma_target = false;
        if (lo.n_gpu_layers != 0) {
            if (enumerated_devices.empty()) {
                enumerated_devices = lc::LlamaCppBackend::EnumerateDevices();
            }
            uma_target = llamacpp_detail::HasIntegratedUmaTarget(enumerated_devices, lo.devices);
        }
        const std::uint32_t loaded_slots = std::max<std::uint32_t>(1u, runtime.SlotCount());
        const lc::SpeculativeLoadInfo spec_info = runtime.GetSpeculativeInfo();
        const std::uint64_t model_id =
            runtime_state_->RegisterModel(info.n_ctx, lo.n_gpu_layers != 0, uma_target, d.name, spec_info);
        runtime_state_->RefreshGpuMemory(/*force=*/true);
        return std::shared_ptr<BackendModel>(std::make_shared<LlamaCppModel>(
            std::move(d), std::move(runtime), load_ms, loaded_slots, runtime_state_, model_id,
            std::move(active_gpu_devices)));
    }

    Status RecoverDevice(std::string_view device) {
        const auto dh = runtime_state_ ? runtime_state_->DeviceHealth() : nullptr;
        if (!dh) {
            return Status(ErrorCode::internal, "device health tracker is not initialized");
        }
        return dh->RecoverDevice(device);
    }

private:
    std::string Resolve(const std::string& model) const {
        std::error_code ec;
        if (fs::is_regular_file(PathFromUtf8(model), ec)) return model;
        for (const auto& dir : options_.model_dirs) {
            const fs::path candidate = PathFromUtf8(dir) / PathFromUtf8(model);
            if (fs::is_regular_file(candidate, ec)) return PathToUtf8(candidate);
        }
        return model;  // Load() reports not_found with the original name
    }

    LlamaCppBackendOptions options_;
    std::shared_ptr<RuntimeState> runtime_state_;
};

}  // namespace

namespace llamacpp_detail {

std::shared_ptr<Backend> MakeLlamaCppBackendForTest(LlamaCppBackendOptions options, ProcessGpuMemoryProbe probe,
                                                    std::chrono::milliseconds sample_interval, bool force_uma_mode,
                                                    std::optional<bool> force_gpu_offload) {
    return std::make_shared<LlamaCppCoreBackend>(std::move(options), std::move(probe), sample_interval, force_uma_mode,
                                                 force_gpu_offload);
}

}  // namespace llamacpp_detail

bool is_llamacpp_device_unhealthy_error(const Status& status) noexcept {
    return lc::IsDeviceUnhealthyInferenceStatus(status);
}

Status recover_llamacpp_device(Backend& backend, std::string_view device) {
    auto* core = dynamic_cast<LlamaCppCoreBackend*>(&backend);
    if (core == nullptr) {
        return lc::RecoverCoreBackendDevice(backend, device);
    }
    return core->RecoverDevice(device);
}

Status validate_llamacpp_options(const LlamaCppBackendOptions& options) {
    lc::LoadOptions lo;
    return ApplyBackendOptions(options, lo);
}

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options) {
    return std::make_shared<LlamaCppCoreBackend>(std::move(options));
}

}  // namespace sonder::inference
