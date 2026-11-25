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
    }
    return ErrorCode::internal;
}

Status ToStatus(const lc::Status& s) { return s.ok() ? Status::success() : Status(MapError(s.code), s.message); }

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
};

struct RuntimeState {
    static constexpr std::uint64_t kDefaultSpillThresholdBytes = 256ull * 1024ull * 1024ull;

    mutable std::mutex mutex;
    llamacpp_detail::ProcessGpuMemoryProbe probe;
    std::chrono::milliseconds sample_interval{1000};
    std::uint32_t configured_ctx = 0;
    bool force_uma_mode = false;
    std::optional<bool> force_gpu_offload;

    bool has_sampled = false;
    bool probe_in_flight = false;
    std::chrono::steady_clock::time_point last_sample_time{};
    GpuMemoryStatus gpu_memory;
    std::uint64_t next_model_id = 1;
    std::vector<ActiveModelEntry> active_models;

    RuntimeState(llamacpp_detail::ProcessGpuMemoryProbe probe_fn, std::chrono::milliseconds interval,
                 std::uint32_t ctx_len, bool uma_override, std::optional<bool> gpu_offload_override)
        : probe(std::move(probe_fn)),
          sample_interval(interval.count() < 0 ? std::chrono::milliseconds{0} : interval),
          configured_ctx(ctx_len),
          force_uma_mode(uma_override),
          force_gpu_offload(gpu_offload_override) {
        gpu_memory.probe = "none";
        gpu_memory.status = "not_sampled";
        gpu_memory.spill_threshold_bytes = kDefaultSpillThresholdBytes;
        gpu_memory.shared_baseline_bytes = 0;
    }

    std::uint64_t RegisterModel(std::uint32_t n_ctx, bool gpu_offload, bool uma_target) {
        std::lock_guard<std::mutex> lock(mutex);
        const std::uint64_t id = next_model_id++;
        active_models.push_back(ActiveModelEntry{id, n_ctx, gpu_offload, uma_target});
        return id;
    }

    void UnregisterModel(std::uint64_t id) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        active_models.erase(std::remove_if(active_models.begin(), active_models.end(),
                                           [id](const ActiveModelEntry& e) noexcept { return e.id == id; }),
                            active_models.end());
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
    lo.n_slots = o.slots;
    lo.spec_type = o.spec_type;
    lo.spec_draft_max = o.spec_draft_max;
    lo.cache_ram_mib = o.cache_ram_mib;
    lo.moe_cache_mib = o.moe_cache_mib;
    lo.moe_cache_cpu_share = o.moe_cache_cpu_share;
    lo.mmproj_path = o.mmproj_path;
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
                  std::shared_ptr<RuntimeState> runtime_state = nullptr, std::uint64_t runtime_model_id = 0)
        : desc_(std::move(desc)),
          runtime_(std::move(runtime)),
          load_ns_(MsToNs(load_ms)),
          slots_(std::max<std::uint32_t>(1u, slots)),
          runtime_state_(std::move(runtime_state)),
          runtime_model_id_(runtime_model_id) {}

    ~LlamaCppModel() override {
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

    Result<EmbedResult> embed(const EmbedRequest& request, const CancellationToken& cancel) override {
        if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled before start");
        if (request.inputs.empty()) {
            return Status(ErrorCode::invalid_argument, "embed request inputs are empty");
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
        lc::EmbedOutput emb_out;
        const lc::Status st =
            runtime_.Embed(tokenized, request.normalize, emb_out, [&] { return cancel.cancelled(); });
        if (!st.ok()) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "embedding cancelled");
            return ToStatus(st);
        }
        EmbedResult result;
        result.embeddings = std::move(emb_out.embeddings);
        result.prompt_tokens = static_cast<std::uint64_t>(emb_out.prompt_tokens);
        result.eval_ns = MsToNs(emb_out.eval_ms);
        return result;
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
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
                          request.grammar, /*separate_reasoning=*/false, request.checkpoint);
    }

    // Chat through the GGUF chat template (llama_chat_apply_template). Models
    // without a template, or with one llama.cpp does not recognise, use the
    // generic BackendModel::chat() prompt format instead. Template scaffolding
    // is tokenized with parse_special=true while untrusted message content is
    // strictly tokenized with parse_special=false (G0028).
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
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
        std::vector<std::vector<std::uint8_t>> media_blobs;
        std::vector<std::pair<std::string, std::size_t>> media_sentinels;
        std::vector<ChatMessage> normalized_messages;
        const std::vector<ChatMessage>* template_messages = &request.messages;
        bool needs_normalized = false;
        for (const auto& m : request.messages) {
            if (!m.media.empty() || !m.tool_calls.empty()) {
                needs_normalized = true;
                break;
            }
        }
        if (needs_normalized) {
            normalized_messages = request.messages;
            std::uint64_t media_nonce = 0x51a7c9e2f03b8d11ULL;
            auto next_media_prefix = [&]() {
                char buf[36];
                std::snprintf(buf, sizeof(buf), "SNDRMM%016llxM", static_cast<unsigned long long>(media_nonce));
                media_nonce = media_nonce * 6364136223846793005ULL + 1442695040888963407ULL;
                return std::string(buf);
            };
            std::string sentinel_prefix = next_media_prefix();
            auto prefix_collides = [&](std::string_view pfx) {
                for (const auto& m : normalized_messages) {
                    if (m.content.find(pfx) != std::string::npos || m.role.find(pfx) != std::string::npos) return true;
                }
                return false;
            };
            while (prefix_collides(sentinel_prefix)) {
                sentinel_prefix = next_media_prefix();
            }
            for (auto& m : normalized_messages) {
                if (!m.media.empty()) {
                    std::string markers;
                    for (const auto& blob : m.media) {
                        const std::size_t blob_idx = media_blobs.size();
                        media_blobs.push_back(blob.bytes);
                        std::string tag = sentinel_prefix + "_" + std::to_string(blob_idx) + "_END";
                        media_sentinels.emplace_back(tag, blob_idx);
                        markers += tag;
                    }
                    m.content = markers + m.content;
                }
                if (m.role == "assistant" && !m.tool_calls.empty()) {
                    std::string tool_text;
                    for (const auto& tc : m.tool_calls) {
                        if (!tool_text.empty()) tool_text.push_back('\n');
                        tool_text += "<tool_call>{\"name\":";
                        tool_text += json::Value(tc.name).dump();
                        tool_text += ",\"arguments\":";
                        tool_text += tc.arguments_json.empty() ? "{}" : tc.arguments_json;
                        tool_text += "}</tool_call>";
                    }
                    if (!m.content.empty()) {
                        m.content += "\n" + tool_text;
                    } else {
                        m.content = std::move(tool_text);
                    }
                }
            }
            template_messages = &normalized_messages;
        }
        bool template_unsupported = false;
        auto segments = build_chat_template_segments(
            *template_messages,
            [&](const std::vector<ChatMessage>& placeholder_msgs) -> Result<std::string> {
                std::vector<lc::ChatTurn> turns;
                turns.reserve(placeholder_msgs.size());
                for (const auto& m : placeholder_msgs) turns.push_back(lc::ChatTurn{m.role, m.content});
                std::string rendered;
                const lc::Status s = runtime_.ApplyChatTemplate(turns, /*add_assistant=*/true, rendered);
                if (!s.ok()) {
                    template_unsupported = true;
                    return ToStatus(s);
                }
                return rendered;
            });
        if (!segments.ok()) {
            if (template_unsupported) {
                if (shared_guard.owns_lock()) shared_guard.unlock();
                if (exclusive_guard.owns_lock()) exclusive_guard.unlock();
                if (!media_blobs.empty()) {
                    return Status(ErrorCode::invalid_argument,
                                  "multimodal chat requires a model with a chat template");
                }
                return BackendModel::chat(request, cancel, on_chunk);
            }
            return segments.status();
        }
        if (request.thinking.enable_thinking.has_value() && !*request.thinking.enable_thinking) {
            ApplyNoThinkScaffolding(segments.value(), /*enable_thinking=*/false);
        }
        // Build G0028-safe structured multimodal parts preserving per-segment
        // parse_special boundaries: trusted template scaffolding stays
        // parse_special=true, untrusted message text stays parse_special=false,
        // and media sentinels become explicit bitmap parts (zero marker parsing).
        std::vector<lc::GenerateRequest::MediaPromptPart> media_prompt_parts;
        if (!media_blobs.empty()) {
            for (const auto& seg : segments.value()) {
                if (seg.parse_special) {
                    if (!seg.text.empty()) {
                        media_prompt_parts.push_back(
                            lc::GenerateRequest::MediaPromptPart{seg.text, /*parse_special=*/true, std::nullopt});
                    }
                    continue;
                }
                std::size_t cursor = 0;
                while (cursor < seg.text.size()) {
                    std::size_t best_pos = std::string::npos;
                    std::size_t best_len = 0;
                    std::size_t best_blob = 0;
                    for (const auto& [tag, bidx] : media_sentinels) {
                        const std::size_t p = seg.text.find(tag, cursor);
                        if (p != std::string::npos && p < best_pos) {
                            best_pos = p;
                            best_len = tag.size();
                            best_blob = bidx;
                        }
                    }
                    if (best_pos == std::string::npos) {
                        media_prompt_parts.push_back(lc::GenerateRequest::MediaPromptPart{
                            seg.text.substr(cursor), /*parse_special=*/false, std::nullopt});
                        break;
                    }
                    if (best_pos > cursor) {
                        media_prompt_parts.push_back(lc::GenerateRequest::MediaPromptPart{
                            seg.text.substr(cursor, best_pos - cursor), /*parse_special=*/false, std::nullopt});
                    }
                    media_prompt_parts.push_back(
                        lc::GenerateRequest::MediaPromptPart{{}, /*parse_special=*/false, best_blob});
                    cursor = best_pos + best_len;
                }
            }
        }
        std::vector<lc::Token> prompt_tokens;
        const Status tok_status = tokenize_chat_segments(
            segments.value(),
            [&](std::string_view piece, bool add_special, bool parse_special, std::vector<TokenId>& out) -> Status {
                return ToStatus(runtime_.Tokenize(piece, add_special, out, parse_special));
            },
            prompt_tokens);
        if (!tok_status.ok()) return tok_status;
        return run_tokens(std::move(prompt_tokens), request.sampling, cancel, on_chunk, request.session_key,
                          request.grammar, request.separate_reasoning, /*checkpoint=*/nullptr,
                          std::move(media_blobs), {}, std::move(media_prompt_parts));
    }

    [[nodiscard]] bool has_native_chat() const override {
        std::string ignored;
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return runtime_.ApplyChatTemplate({lc::ChatTurn{"user", "x"}}, true, ignored).ok();
    }

    Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) override {
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
                                     std::vector<lc::GenerateRequest::MediaPromptPart> media_prompt_parts = {}) {
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
        ThinkingStreamSplitter thinking_splitter;
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

        if (!r.status.ok()) return ToStatus(r.status);

        switch (r.stop_reason) {
            case lc::StopReason::kEndOfGeneration: stats.stop_reason = StopReason::end_of_sequence; break;
            case lc::StopReason::kMaxTokens: stats.stop_reason = StopReason::max_tokens; break;
            case lc::StopReason::kContextFull: stats.stop_reason = StopReason::max_tokens; break;
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
        if (stats.stop_reason != StopReason::stop_sequence && stats.stop_reason != StopReason::callback) {
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

    ModelDescriptor desc_;
    lc::LlamaCppBackend runtime_;
    std::uint64_t load_ns_ = 0;
    std::uint32_t slots_ = 1;
    std::shared_ptr<RuntimeState> runtime_state_;
    std::uint64_t runtime_model_id_ = 0;
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
              probe ? std::move(probe) : MakeDefaultGpuMemoryProbe(options_.devices), sample_interval,
              options_.context_length, force_uma_mode, force_gpu_offload)) {}

    [[nodiscard]] std::string name() const override { return kLlamaCppBackendName; }
    [[nodiscard]] std::string description() const override {
        return "direct llama.cpp/GGML (" + std::string(lc::LlamaCppBackend::UpstreamVersion()) + "), in-process GGUF";
    }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        caps.add(Capability::tokenization)
            .add(Capability::streaming)
            .add(Capability::batched_prefill)
            .add(Capability::prefix_reuse)
            .add(Capability::speculative_decode)
            .add(Capability::structured_output)
            .add(Capability::embeddings)
            .add(Capability::deterministic)
            .add(Capability::recurrent_checkpoint)
            .add(Capability::logprobs);
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

        lc::LlamaCppBackend runtime;
        double load_ms = 0.0;
        runtime.SetTelemetrySink([&](const lc::TelemetryEvent& e) {
            if (e.kind == lc::TelemetryKind::kModelLoaded) load_ms = e.elapsed_ms;
        });
        if (lc::Status s = runtime.Load(lo); !s.ok()) return ToStatus(s);
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
        const std::uint64_t model_id = runtime_state_->RegisterModel(info.n_ctx, lo.n_gpu_layers != 0, uma_target);
        runtime_state_->RefreshGpuMemory(/*force=*/true);
        return std::shared_ptr<BackendModel>(std::make_shared<LlamaCppModel>(
            std::move(d), std::move(runtime), load_ms, loaded_slots, runtime_state_, model_id));
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

Status validate_llamacpp_options(const LlamaCppBackendOptions& options) {
    lc::LoadOptions lo;
    return ApplyBackendOptions(options, lo);
}

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options) {
    return std::make_shared<LlamaCppCoreBackend>(std::move(options));
}

}  // namespace sonder::inference
