// Sonder Inference: direct llama.cpp/GGML backend factory (module
// src/backends/llamacpp, built when SONDER_WITH_LLAMA_CPP=ON).
//
// Available only when SONDER_HAS_LLAMACPP_BACKEND is defined.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/speculative_verifier.hpp"
#include "sonder/inference/swa_policy.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference {

inline constexpr const char* kLlamaCppBackendName = "llamacpp";

// One LoRA adapter file + scale attached to a named LoRA alias.
struct LlamaCppLoraAdapterOption {
    std::string path;
    float scale = 1.0f;
};

// Named LoRA alias mapping an alias string to a base model + one or more adapters.
struct LlamaCppLoraAliasOption {
    std::string alias;
    std::string base_model;
    std::vector<LlamaCppLoraAdapterOption> adapters;
};

struct LlamaCppBackendOptions {
    // Directories scanned (non-recursively) by list_models() for *.gguf files.
    // load_model() accepts either a GGUF path or a file name found here.
    std::vector<std::string> model_dirs;
    std::uint32_t context_length = 4096;  // 0 = model training context
    std::uint32_t batch_size = 512;       // max prompt tokens per llama_decode
    std::int32_t threads = 0;             // 0 = llama.cpp default
    // Layers offloaded for "gpu:*" device ids (-1 = all). "cpu:*" always uses 0.
    std::int32_t gpu_layers = -1;
    // Tensor placement overrides as (regex pattern, device) pairs, e.g.
    // {"\.ffn_(up|down|gate)_(ch|)exps", "cpu"} keeps MoE experts in RAM.
    // Device is "cpu" or a llama.cpp device name ("Vulkan0", "CUDA0").
    std::vector<std::pair<std::string, std::string>> tensor_overrides;
    // Physical micro-batch (llama.cpp n_ubatch). 0 = llama.cpp's default (512)
    // capped at batch_size; otherwise must be <= batch_size.
    std::uint32_t ubatch_size = 0;
    // KV cache element types: "auto" (default: resolves per model to q8_0 when
    // the model's head dims are multiples of 32 and flash_attention != "off",
    // otherwise f16), or an explicit type: q8_0, f16, f32, bf16, q5_1, q5_0,
    // q4_1, q4_0 (K requires kv_k_q4_allowed + model rotation), iq4_nl, f8_e4m3 (CUDA-only).
    std::string kv_cache_type_k = "auto";
    std::string kv_cache_type_v = "auto";
    // Explicit opt-in permitting q4_0 for K when the model supports Hadamard K rotation.
    bool kv_k_q4_allowed = false;
    // Flash Attention: auto (default, llama.cpp decides), on or off. A
    // quantized V cache needs it: "off" with a quantized V type is rejected.
    std::string flash_attention = "auto";
    // Explicit ggml device names (e.g. "CUDA0", "ROCm0", "Vulkan0", "CUDA1")
    // for vendor-targeted or multi-GPU offloading. Non-empty is an exact
    // allowlist (case-insensitive names; an iGPU is used only when named; two
    // backend views of one card collapse to the first named). Empty = automatic:
    // discrete GPUs only, never an iGPU, CPU-only when there is none. A load
    // request's ModelLoadOptions::devices overrides this per model.
    std::vector<std::string> devices;
    // Primary GPU index for single-GPU mode (split_mode == "none") and
    // non-split / scratch tensors. Must be >= 0.
    std::int32_t main_gpu = 0;
    // Multi-GPU split mode: "layer" (default), "none" (single GPU), or "row".
    std::string split_mode = "layer";
    // Relative split proportions across GPUs (empty = equal / llama.cpp
    // default). When non-empty: at most 16 finite, non-negative entries with
    // at least one strictly positive value.
    std::vector<float> tensor_split;
    // Lock model weights in RAM (llama.cpp load_mode mlock / mmap+mlock).
    bool use_mlock = false;
    // Memory-map GGUF weights (true = mmap default; false = direct read, e.g. load-mode=none).
    bool use_mmap = true;
    // Use a unified KV buffer across slots/sequences when supported by llama.cpp.
    bool kv_unified = true;
    // Reuse longest matching prompt KV prefix across consecutive turns.
    bool prefix_cache = true;
    // Coalesce simultaneous cold shared-prefix prefills on attention-only models
    // (requires slots > 1 and prefix_cache = true; bypassed on hybrid/recurrent).
    bool prefix_coalesce = true;
    std::uint32_t min_coalesce_prefix_tokens = 64;
    std::uint32_t max_followers_per_producer = 8;
    std::uint32_t max_coalesce_wait_ms = 5000;
    // Number of cached KV sequence slots (1..16). Values > 1 allow interleaved
    // conversations (matched by session_key or longest common prefix) without
    // evicting each other's prefix cache.
    std::uint32_t slots = 1;
    // In-process speculative decoding strategy: "none" (default), "auto"
    // (draft-mtp if model has MTP layers, else ngram-mod), "ngram",
    // "ngram-mod", "suffix", or "draft-mtp".
    std::string spec_type = "none";
    // Max draft tokens per speculative step (0 = automatic default for spec_type).
    std::int32_t spec_draft_max = 0;
    // Minimum draft-token probability for model-based speculation ([0, 1], default 0.0).
    double spec_p_min = 0.0;
    // Per-route / per-task-class speculation rules (<selector>=<policy>), parsed by ParseSpecRule().
    std::vector<std::string> spec_route_rules{};
    // Determinism mode: disables model-based drafting (draft-model / draft-mtp).
    bool deterministic = false;
    // Speculative decoding verification policy: TargetMatch (default reference
    // path) vs opt-in StochasticRejection (Leviathan et al. sum min(p, q) +
    // residual (p - q)_+ resampling). StochasticRejection activates only when
    // a stochastic proposer supplies real post-processor proposal probabilities;
    // greedy MTP and deterministic suffix/n-gram proposers stay on TargetMatch.
    SpeculativeVerificationMode spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
    // Bounded L2 host-RAM prefix state cache budget in MiB (0 = disabled, max 65536).
    std::uint32_t cache_ram_mib = 0;
    // Context checkpoint spacing in tokens (normal / dense for hybrid & recurrent models) and per-slot cap.
    std::uint32_t checkpoint_spacing_tokens = 2048;
    std::uint32_t checkpoint_dense_spacing_tokens = 1024;
    std::uint32_t checkpoint_max_per_slot = 8;
    // Dynamic GPU cache in MiB for host-resident MoE experts (0 = disabled, max 262144).
    std::uint32_t moe_cache_mib = 0;
    // Unused/rejected by the direct llamacpp backend (no execution consumer):
    // must remain 0.0; non-zero or non-finite values fail with invalid_argument.
    double moe_cache_cpu_share = 0.0;
    // Optional multimodal projector GGUF path (--mmproj); requires libmtmd build.
    std::string mmproj_path;
    // Sliding-window attention KV policy: Full (default, safe for arbitrary
    // prefix trims/rewinds/speculative rollbacks) or CompactIfQualified
    // (opt-in compact SWA cache guarded by min_retained_pos watermark checks).
    SwaKvMode swa_kv_mode = SwaKvMode::Full;
    // Default inter-token latency (ITL) target in milliseconds for foreground
    // (`interactive`) decoding requests without an explicit `target_itl_ms`
    // (0 disables the implicit foreground ITL cap).
    double default_foreground_itl_ms = 50.0;
    // Consecutive sustained single-request decode steps required after batch
    // suspension before reactivating per-request speculation (0..1024).
    std::uint32_t spec_reactivation_cooldown_steps = 4;
    // Opt-in bounded small-batch concurrent linear speculation (default false).
    bool spec_batch_enabled = false;
    // Maximum active decode slots eligible for concurrent linear speculation (1..16).
    std::uint32_t spec_batch_max_slots = 4;
    // Maximum aggregate draft tokens verified across all concurrent slots per step (0 = default 16).
    std::int32_t spec_batch_max_aggregate_draft_tokens = 16;
    // Configurable multi-sequence embedding/rerank batch token budget (0 = auto).
    std::uint32_t embed_batch_tokens = 2048;
    // Maximum sequences packed per embedding/rerank forward pass (0 = auto, max 64).
    std::uint32_t embed_max_sequences = 16;
    // In-process workload role ("main_decode" or "auxiliary"), CUDA stream priorities
    // ("high", "default", "low"), and decode-aware auxiliary gate settings (NAT-12).
    std::string workload_role = "main_decode";
    std::string main_stream_priority = "high";
    std::string aux_stream_priority = "low";
    bool aux_gate_enabled = true;
    double max_aux_duty_cycle = 0.05;
    std::uint32_t aux_active_slice_tokens = 512;
    std::uint32_t aux_active_max_sequences = 4;
    std::uint32_t aux_max_decode_hold_ms = 25;
    // Optional root directories constraining allowed LoRA adapter files (defaults to model_dirs when empty).
    std::vector<std::string> lora_roots{};
    // Named LoRA aliases mapping alias -> base model + one or more (adapter, scale) pairs.
    std::vector<LlamaCppLoraAliasOption> lora_aliases{};
    // Opt-in startup greedy smoke test after load (default off).
    bool startup_smoke_test = false;
    // Per-request embedding limits on the tokenized inputs (0 = no limit). A single input is
    // always bounded by the context window as well.
    std::uint32_t embed_max_total_tokens = 262144;
    std::uint32_t embed_max_input_tokens = 0;
    // Telemetry verbosity for backend-internal instrumentation. Layer and
    // operator graph telemetry (and Capability::layer_telemetry) activate
    // strictly at TelemetryLevel::deep; below deep no ggml callback is installed.
    TelemetryLevel telemetry_level = TelemetryLevel::standard;
    // Sample graph eval telemetry every N evals when telemetry_level == deep (>= 1).
    std::uint32_t graph_telemetry_sample_every = 1;
    // Collect per-layer / per-op wall timings on sampled evals at deep level.
    bool graph_telemetry_timings = true;
    // Optional non-owning TelemetryBus receiving backend.layer.timing and
    // backend.operator.activity events at TelemetryLevel::deep.
    TelemetryBus* telemetry_bus = nullptr;
    // Optional correlation context for emitted graph telemetry events.
    TelemetryContext telemetry_context{};
};

// Checks the llama.cpp context and placement options (KV cache types,
// flash_attention, ubatch_size vs batch_size, split_mode, main_gpu,
// tensor_split, device name syntax) without loading anything. load_model()
// applies the same checks; hosts call this to reject bad flags at startup.
// Errors: invalid_argument.
Status validate_llamacpp_options(const LlamaCppBackendOptions& options);

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options = {});

#define SONDER_HAS_LLAMACPP_DEVICE_HEALTH_API 1
inline constexpr std::string_view kLlamaCppDeviceUnhealthyError = "device_unhealthy";
[[nodiscard]] bool is_llamacpp_device_unhealthy_error(const Status& status) noexcept;
Status recover_llamacpp_device(Backend& backend, std::string_view device);

}  // namespace sonder::inference
