// In-process CUDA stream priorities, primary-context external EP stream helper
// (ONNX Runtime `NvTensorRTRTXExecutionProvider` `user_compute_stream` scaffolding),
// and decode-aware auxiliary execution gate for slicing embeddings, reranking, and
// auxiliary workloads between main LLM decode steps.
//
// Zero llama.cpp or CUDA SDK headers are included here so callers and unit tests
// build cleanly across CPU-only, Vulkan, ROCm, and CUDA presets.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::backends::llamacpp {

enum class ErrorCode;
struct Status;
struct PlannedEmbedBatch;

// Logical GPU stream priority class within a single CUDA context.
// In CUDA, numerically lower values represent higher hardware scheduling priority
// (`greatestPriority <= leastPriority` from `cudaDeviceGetStreamPriorityRange`).
enum class StreamPriority : std::uint8_t {
    kDefault = 0,
    kHigh = 1,
    kLow = 2,
};

[[nodiscard]] std::string_view ToString(StreamPriority priority) noexcept;
[[nodiscard]] bool ParseStreamPriority(std::string_view name, StreamPriority& out) noexcept;

// Execution role of a loaded model inside Sonder Inference.
//   kMainDecode: foreground generative LLM; owns high-priority decode streams and
//                registers active decode sessions/steps on DecodeAwareAuxGate.
//   kAuxiliary:  co-resident auxiliary model (embeddings, reranker, encoder/seq2seq);
//                runs on low-priority streams and slices work between decode steps.
enum class WorkloadRole : std::uint8_t {
    kMainDecode = 0,
    kAuxiliary = 1,
};

[[nodiscard]] std::string_view ToString(WorkloadRole role) noexcept;
[[nodiscard]] bool ParseWorkloadRole(std::string_view name, WorkloadRole& out) noexcept;

// Resolves the numeric CUDA stream priority in [min(least, greatest), max(least, greatest)].
// Handles CUDA's negative-is-higher convention (`greatest <= least`) as well as
// degenerate (`least == greatest`) or inverted mock ranges safely.
[[nodiscard]] int ResolveCudaNumericStreamPriority(int least_priority, int greatest_priority,
                                                   StreamPriority requested) noexcept;

// Effective primary context stream priority for a model's main `llama_context`:
// auxiliary-role models use `aux_configured` (`kLow` by default); main-decode
// models use `main_configured` (`kHigh` by default).
[[nodiscard]] StreamPriority EffectiveMainStreamPriority(
    WorkloadRole role, StreamPriority main_configured,
    StreamPriority aux_configured = StreamPriority::kLow) noexcept;

struct GgmlCudaStreamPriorityStats {
    std::uint64_t high_priority_scopes = 0;
    std::uint64_t low_priority_scopes = 0;
    std::uint64_t default_priority_scopes = 0;
    std::uint64_t priority_streams_created = 0;
    std::uint64_t fallback_streams_created = 0;
    int last_device = -1;
    int last_numeric_priority = 0;
    bool preinclude_hook_compiled = false;
};

[[nodiscard]] GgmlCudaStreamPriorityStats GetGgmlCudaStreamPriorityStats() noexcept;
void ResetGgmlCudaStreamPriorityStatsForTest() noexcept;

// RAII thread-local priority hint guard consulted by `sonder_ggml_cuda_stream_priority.cuh`
// whenever `ggml-cuda` lazily creates a `cudaStream_t` during context init or graph execution.
class ScopedGgmlCudaStreamPriority {
public:
    explicit ScopedGgmlCudaStreamPriority(StreamPriority priority) noexcept;
    ~ScopedGgmlCudaStreamPriority() noexcept;

    ScopedGgmlCudaStreamPriority(const ScopedGgmlCudaStreamPriority&) = delete;
    ScopedGgmlCudaStreamPriority& operator=(const ScopedGgmlCudaStreamPriority&) = delete;

private:
    int prev_hint_ = 0;
};

// Minimal CUDA Driver API table (`nvcuda.dll` / `libcuda.so.1`) used to create
// non-blocking priority streams on a device's primary CUDA context without a
// compile-time CUDA Toolkit dependency. Injectable for deterministic unit tests.
struct CudaDriverApi {
    using CuResult = int;
    using CuDevice = int;
    using CuContext = void*;
    using CuStream = void*;

    static constexpr CuResult kSuccess = 0;
    static constexpr unsigned int kStreamNonBlocking = 0x1u;

    CuResult (*init)(unsigned int flags) = nullptr;
    CuResult (*device_get_count)(int* count) = nullptr;
    CuResult (*device_get)(CuDevice* device, int ordinal) = nullptr;
    CuResult (*primary_ctx_retain)(CuContext* pctx, CuDevice dev) = nullptr;
    CuResult (*primary_ctx_release)(CuDevice dev) = nullptr;
    CuResult (*ctx_push_current)(CuContext ctx) = nullptr;
    CuResult (*ctx_pop_current)(CuContext* pctx) = nullptr;
    CuResult (*ctx_get_stream_priority_range)(int* least_priority, int* greatest_priority) = nullptr;
    CuResult (*stream_create_with_priority)(CuStream* ph_stream, unsigned int flags, int priority) = nullptr;
    CuResult (*stream_get_priority)(CuStream h_stream, int* priority) = nullptr;
    CuResult (*stream_synchronize)(CuStream h_stream) = nullptr;
    CuResult (*stream_destroy)(CuStream h_stream) = nullptr;

    [[nodiscard]] bool complete() const noexcept;
};

// Loads the system CUDA Driver API (`LOAD_LIBRARY_SEARCH_SYSTEM32` on Windows,
// `libcuda.so.1` on Linux). Returns nullptr when no NVIDIA driver is present.
[[nodiscard]] const CudaDriverApi* GetSystemCudaDriverApi() noexcept;

// Key-value provider options formatted for ONNX Runtime TensorRT-RTX Execution Provider
// (`NvTensorRTRTXExecutionProvider` / `NvTensorRtRtx`) sharing the process primary CUDA
// context. API scaffolding: Sonder Inference has no in-tree ONNX Runtime consumer yet.
struct TrtRtxProviderOptions {
    std::vector<std::pair<std::string, std::string>> entries;

    [[nodiscard]] std::optional<std::string_view> Find(std::string_view key) const noexcept;
};

// Move-only RAII non-blocking CUDA stream bound to the device's primary CUDA context
// (the same context `cudart` and `ggml-cuda` use in-process). Scaffolding helper exposing
// `user_compute_stream()` for future external EP / custom auxiliary kernel integrations.
class CudaPriorityStream {
public:
    CudaPriorityStream() = default;
    ~CudaPriorityStream() noexcept;

    CudaPriorityStream(CudaPriorityStream&& other) noexcept;
    CudaPriorityStream& operator=(CudaPriorityStream&& other) noexcept;
    CudaPriorityStream(const CudaPriorityStream&) = delete;
    CudaPriorityStream& operator=(const CudaPriorityStream&) = delete;

    // Creates a primary-context stream at `priority` on `device_ordinal`.
    // Uses `custom_api` when non-null (unit tests), otherwise `GetSystemCudaDriverApi()`.
    [[nodiscard]] static Status Create(int device_ordinal, StreamPriority priority, CudaPriorityStream& out,
                                       const CudaDriverApi* custom_api = nullptr);

    void Reset() noexcept;

    [[nodiscard]] bool valid() const noexcept { return stream_ != nullptr && primary_ctx_ != nullptr; }
    [[nodiscard]] int device_ordinal() const noexcept { return device_ordinal_; }
    [[nodiscard]] StreamPriority requested_priority() const noexcept { return requested_priority_; }
    [[nodiscard]] int numeric_priority() const noexcept { return numeric_priority_; }
    [[nodiscard]] int least_priority() const noexcept { return least_priority_; }
    [[nodiscard]] int greatest_priority() const noexcept { return greatest_priority_; }
    [[nodiscard]] bool priority_range_supported() const noexcept { return least_priority_ != greatest_priority_; }

    // Opaque `cudaStream_t` / `CUstream` handle for ORT `NvTensorRTRTXExecutionProvider` `user_compute_stream`.
    [[nodiscard]] void* user_compute_stream() const noexcept { return stream_; }

    // Formats `user_compute_stream` as decimal uintptr string expected by ORT C API provider maps.
    [[nodiscard]] std::string FormatUserComputeStreamOption() const;

    // Builds `NvTensorRTRTXExecutionProvider` options (`device_id`, `user_compute_stream`,
    // `enable_cuda_graph`, and optional `nv_runtime_cache_path`). Defaults `enable_cuda_graph`
    // to false on auxiliary streams because TRT-RTX enables CUDA graphs by default and graph
    // capture/replay can interact with stream priority unless explicitly validated.
    [[nodiscard]] TrtRtxProviderOptions BuildTrtRtxProviderOptions(
        bool enable_cuda_graph = false, std::string_view runtime_cache_path = {}) const;

    // Synchronizes pending GPU work on this stream so a gated slice measures true
    // device execution duration and finishes before the next decode step starts.
    [[nodiscard]] Status Synchronize() const;

private:
    const CudaDriverApi* api_ = nullptr;
    int device_ordinal_ = -1;
    CudaDriverApi::CuDevice cu_device_ = 0;
    CudaDriverApi::CuContext primary_ctx_ = nullptr;
    CudaDriverApi::CuStream stream_ = nullptr;
    StreamPriority requested_priority_ = StreamPriority::kDefault;
    int least_priority_ = 0;
    int greatest_priority_ = 0;
    int numeric_priority_ = 0;
};

// ---- Decode-Aware Auxiliary Execution Gate ----

inline constexpr double kDefaultMaxAuxDutyCycle = 0.05;
inline constexpr double kMaxAllowedAuxDutyCycle = 0.50;
inline constexpr std::uint32_t kDefaultAuxActiveSliceTokens = 512;
inline constexpr std::uint32_t kMaxAuxActiveSliceTokens = 65536;
inline constexpr std::uint32_t kDefaultAuxActiveMaxSequences = 4;
inline constexpr std::uint32_t kDefaultMaxDecodeHoldMs = 25;
inline constexpr std::uint32_t kDefaultHandoffPickupTimeoutMs = 5;
inline constexpr std::uint32_t kDefaultMaxStarvationWaitMs = 1000;
inline constexpr std::uint32_t kDefaultCancelPollIntervalMs = 2;
inline constexpr std::uint32_t kMaxAuxGateTimeoutMs = 60000;

struct AuxGateConfig {
    bool enabled = true;
    // When false (default), CPU-only execution (`domain == "cpu"`, `n_gpu_layers == 0`)
    // bypasses decode-aware slicing and throttling so CPU deployments keep existing
    // behavior unless explicitly opted in (e.g. deterministic CPU tests).
    bool gate_cpu_domain = false;
    // Configured DRR upper bound on the fraction of active-decode GPU busy time granted
    // to auxiliary slices (default 0.05 targets >= 95% main decode retention; end-to-end
    // throughput impact on full-scale models remains a target to measure on hardware).
    double max_aux_duty_cycle = kDefaultMaxAuxDutyCycle;
    // Maximum tokens packed per auxiliary slice while foreground decode is active (<= 512 default).
    std::uint32_t active_slice_max_tokens = kDefaultAuxActiveSliceTokens;
    // Maximum sequences packed per auxiliary slice while foreground decode is active.
    std::uint32_t active_slice_max_sequences = kDefaultAuxActiveMaxSequences;
    // Upper bound on how long EnterDecodeStep() holds for one in-flight aux slice (ms).
    std::uint32_t max_decode_hold_ms = kDefaultMaxDecodeHoldMs;
    // Upper bound for a woken aux thread to claim an inter-step handoff ticket (ms).
    std::uint32_t handoff_pickup_timeout_ms = kDefaultHandoffPickupTimeoutMs;
    // Anti-starvation ceiling in [1, kMaxAuxGateTimeoutMs] ms (must be >= 1 to prevent
    // infinite wait if an aux request is issued while a decode session is paused).
    std::uint32_t max_starvation_wait_ms = kDefaultMaxStarvationWaitMs;
    // Cancellation polling period while an aux caller waits at the gate (ms).
    std::uint32_t cancel_poll_interval_ms = kDefaultCancelPollIntervalMs;
};

[[nodiscard]] Status ValidateAuxGateConfig(const AuxGateConfig& config);

// Maps placement settings onto a canonical device execution domain ("cpu", "gpu:0",
// "gpu:1", "gpu:0+1", ...) canonicalizing vendor ordinal names such as "CUDA0" or
// "Vulkan0" onto "gpu:0" so co-resident models on the same GPU coordinate even when
// one uses `main_gpu` and another specifies an explicit `devices` allowlist.
[[nodiscard]] std::string ResolveAuxExecutionDomain(std::int32_t n_gpu_layers, std::int32_t main_gpu,
                                                    const std::vector<std::string>& devices);

enum class AuxAcquireStatus : std::uint8_t {
    kAcquiredIdle = 0,
    kAcquiredBetweenSteps = 1,
    kAcquiredStarvation = 2,
    kBypassedDisabled = 3,
    kCancelled = 4,
};

struct AuxGateStats {
    std::uint64_t decode_sessions_total = 0;
    std::uint64_t decode_steps_total = 0;
    std::uint64_t decode_steps_synced_before_release = 0;
    std::uint64_t decode_busy_ns = 0;
    std::uint64_t decode_hold_wait_ns = 0;
    std::uint64_t decode_hold_timeouts = 0;
    std::uint64_t aux_slices_total = 0;
    std::uint64_t aux_slices_synced_before_release = 0;
    std::uint64_t aux_slices_idle_fastpath = 0;
    std::uint64_t aux_slices_gated_between_steps = 0;
    std::uint64_t aux_slices_starvation_admitted = 0;
    std::uint64_t aux_slices_cancelled = 0;
    std::uint64_t aux_slices_overlapped_decode = 0;
    std::uint64_t aux_busy_ns = 0;
    std::uint64_t aux_gate_wait_ns = 0;
    std::size_t active_decode_sessions = 0;
    std::size_t in_flight_decode_steps = 0;
    std::size_t in_flight_aux_slices = 0;
    std::size_t waiting_aux_slices = 0;
};

// Thread-safe Deficit Round Robin (DRR) inter-step gate coordinating foreground
// decode steps and auxiliary slices per execution domain. Callers must synchronize
// device work (`llama_synchronize` or `CudaPriorityStream::Synchronize`) and call
// `MarkSynchronized()` before releasing `DecodeStepScope` / `AuxSliceScope` so the
// gate orders actual GPU execution rather than asynchronous host submission.
class DecodeAwareAuxGate {
public:
    using ClockFn = std::function<std::int64_t()>;

    explicit DecodeAwareAuxGate(AuxGateConfig config = {}, ClockFn clock_ns = {});
    ~DecodeAwareAuxGate();

    DecodeAwareAuxGate(const DecodeAwareAuxGate&) = delete;
    DecodeAwareAuxGate& operator=(const DecodeAwareAuxGate&) = delete;

    Status Reconfigure(const AuxGateConfig& config);
    [[nodiscard]] AuxGateConfig config() const;

    [[nodiscard]] bool IsDecodeActive(std::string_view domain = "default") const;
    [[nodiscard]] bool IsDecodeStepInFlight(std::string_view domain = "default") const;

    class DecodeSessionScope {
    public:
        DecodeSessionScope() = default;
        DecodeSessionScope(DecodeSessionScope&& other) noexcept;
        DecodeSessionScope& operator=(DecodeSessionScope&& other) noexcept;
        DecodeSessionScope(const DecodeSessionScope&) = delete;
        DecodeSessionScope& operator=(const DecodeSessionScope&) = delete;
        ~DecodeSessionScope() noexcept;

        void Release() noexcept;
        [[nodiscard]] bool active() const noexcept { return gate_ != nullptr; }

    private:
        friend class DecodeAwareAuxGate;
        DecodeSessionScope(DecodeAwareAuxGate* gate, std::size_t domain_idx) noexcept;

        DecodeAwareAuxGate* gate_ = nullptr;
        std::size_t domain_idx_ = 0;
    };

    class DecodeStepScope {
    public:
        DecodeStepScope() = default;
        DecodeStepScope(DecodeStepScope&& other) noexcept;
        DecodeStepScope& operator=(DecodeStepScope&& other) noexcept;
        DecodeStepScope(const DecodeStepScope&) = delete;
        DecodeStepScope& operator=(const DecodeStepScope&) = delete;
        ~DecodeStepScope() noexcept;

        void MarkSynchronized() noexcept;
        void Release() noexcept;
        [[nodiscard]] bool active() const noexcept { return gate_ != nullptr; }
        [[nodiscard]] bool synchronized() const noexcept { return synced_; }

    private:
        friend class DecodeAwareAuxGate;
        DecodeStepScope(DecodeAwareAuxGate* gate, std::size_t domain_idx, std::int64_t start_ns) noexcept;

        DecodeAwareAuxGate* gate_ = nullptr;
        std::size_t domain_idx_ = 0;
        std::int64_t start_ns_ = 0;
        bool synced_ = false;
    };

    class AuxSliceScope {
    public:
        AuxSliceScope() = default;
        AuxSliceScope(AuxSliceScope&& other) noexcept;
        AuxSliceScope& operator=(AuxSliceScope&& other) noexcept;
        AuxSliceScope(const AuxSliceScope&) = delete;
        AuxSliceScope& operator=(const AuxSliceScope&) = delete;
        ~AuxSliceScope() noexcept;

        void MarkSynchronized() noexcept;
        void Release() noexcept;
        [[nodiscard]] AuxAcquireStatus status() const noexcept { return status_; }
        [[nodiscard]] bool acquired() const noexcept { return status_ != AuxAcquireStatus::kCancelled; }
        [[nodiscard]] bool synchronized() const noexcept { return synced_; }
        [[nodiscard]] bool gated() const noexcept {
            return status_ == AuxAcquireStatus::kAcquiredBetweenSteps ||
                   status_ == AuxAcquireStatus::kAcquiredStarvation;
        }
        [[nodiscard]] std::int64_t wait_ns() const noexcept { return wait_ns_; }

    private:
        friend class DecodeAwareAuxGate;
        AuxSliceScope(DecodeAwareAuxGate* gate, std::size_t domain_idx, AuxAcquireStatus status,
                      std::int64_t start_ns, std::int64_t wait_ns) noexcept;

        DecodeAwareAuxGate* gate_ = nullptr;
        std::size_t domain_idx_ = 0;
        AuxAcquireStatus status_ = AuxAcquireStatus::kCancelled;
        std::int64_t start_ns_ = 0;
        std::int64_t wait_ns_ = 0;
        bool synced_ = false;
    };

    [[nodiscard]] DecodeSessionScope EnterDecodeSession(std::string_view domain = "default");
    [[nodiscard]] DecodeStepScope EnterDecodeStep(std::string_view domain = "default");
    [[nodiscard]] AuxSliceScope AcquireAuxSlice(std::string_view domain = "default",
                                                const std::function<bool()>& should_cancel = {});

    // Empty `domain` returns totals summed across all domains; non-empty returns that domain.
    [[nodiscard]] AuxGateStats GetStats(std::string_view domain = {}) const;
    void ResetStats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Shared process-wide default gate used when co-resident `LlamaCppBackend` instances
// do not supply an explicit isolated gate.
[[nodiscard]] std::shared_ptr<DecodeAwareAuxGate> DefaultProcessAuxGate();

// Test inspection hook invoked immediately after `llama_synchronize` inside a gated
// decode step or auxiliary slice so unit tests can assert GPU synchronization runs
// while the gate scope is still held (before `Release()`).
struct AuxContextSyncObservation {
    std::string domain;
    bool is_aux_slice = false;
    bool decode_step_in_flight_at_sync = false;
    bool aux_slice_in_flight_at_sync = false;
};

using AuxContextSyncObserver = std::function<void(const AuxContextSyncObservation&)>;
void SetAuxContextSyncObserverForTest(AuxContextSyncObserver observer);
void NotifyAuxContextSynchronized(const DecodeAwareAuxGate* gate, std::string_view domain,
                                  bool is_aux_slice);

// Splits a `PlannedEmbedBatch` into bounded sub-batches of at most `max_slice_tokens`
// and `max_slice_sequences` (used dynamically when foreground decode starts mid-request).
// Single oversized items (`token_count > max_slice_tokens`) remain isolated 1-item slices.
[[nodiscard]] std::vector<PlannedEmbedBatch> SubslicePlannedBatch(const PlannedEmbedBatch& batch,
                                                                  std::size_t max_slice_tokens,
                                                                  std::size_t max_slice_sequences);

// ---- STT / Streaming Audio Encoder Slice Planner & Runner (API Scaffolding) ----
// Sonder Inference has no in-tree STT or ONNX Runtime consumer yet; these helpers
// provide tested slice-planning and stream-synchronized gate execution primitives
// for future in-process audio/encoder integrations.

struct SttSlicePlanConfig {
    std::uint32_t sample_rate_hz = 16000;
    std::uint32_t frame_hop_samples = 160;      // 10 ms at 16 kHz
    std::uint32_t encoder_downsample = 4;       // e.g. FastConformer/Parakeet 4x-8x subsampling
    std::uint32_t overlap_tokens = 0;           // optional left-context overlap per slice
    std::uint32_t idle_slice_max_tokens = 2048; // full batch when main LLM decode is idle
    std::uint32_t active_slice_max_tokens = kDefaultAuxActiveSliceTokens;  // <= 512 during decode
    bool decode_active = false;
};

struct SttAudioSlice {
    std::size_t slice_index = 0;
    std::uint64_t sample_offset = 0;
    std::uint64_t sample_count = 0;
    std::size_t encoder_tokens = 0;
    std::size_t overlap_prefix_tokens = 0;
};

struct SttSlicePlan {
    bool ok = false;
    std::string error;
    std::vector<SttAudioSlice> slices;
    std::size_t total_encoder_tokens = 0;
    std::size_t max_slice_tokens_used = 0;
};

[[nodiscard]] SttSlicePlan PlanSttAudioSlices(std::uint64_t total_pcm_samples,
                                              const SttSlicePlanConfig& config = {});

struct AuxSliceRunSummary {
    std::size_t slices_executed = 0;
    std::size_t gated_slices = 0;
    double gate_wait_ms = 0.0;
};

// Executes `slice_count` auxiliary work slices through `gate` on `domain` under
// `stream_priority`. When `cuda_stream` is non-null and valid, `cuda_stream->Synchronize()`
// runs inside each `AuxSliceScope` before releasing the gate so async GPU kernels complete
// within the measured slice budget. Cancellation returns `ErrorCode::kDecodeFailed`
// (matching `LlamaCppBackend::Embed` / `ScorePairs`).
[[nodiscard]] Status RunGatedAuxSlices(
    DecodeAwareAuxGate& gate, std::string_view domain, std::size_t slice_count,
    const std::function<Status(std::size_t slice_index, bool decode_active)>& run_slice,
    const std::function<bool()>& should_cancel = {},
    StreamPriority stream_priority = StreamPriority::kLow,
    const CudaPriorityStream* cuda_stream = nullptr,
    AuxSliceRunSummary* out_summary = nullptr);

}  // namespace sonder::backends::llamacpp
