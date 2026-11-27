// Sonder Inference: direct llama.cpp/GGML backend factory (module
// src/backends/llamacpp, built when SONDER_WITH_LLAMA_CPP=ON).
//
// Available only when SONDER_HAS_LLAMACPP_BACKEND is defined.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/speculative_verifier.hpp"
#include "sonder/inference/swa_policy.hpp"

namespace sonder::inference {

inline constexpr const char* kLlamaCppBackendName = "llamacpp";

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
    // KV cache element types: q8_0 (default), f16, f32, bf16, q5_1, q5_0,
    // q4_1, q4_0 (K requires kv_k_q4_allowed + model rotation), iq4_nl, f8_e4m3 (CUDA-only).
    std::string kv_cache_type_k = "q8_0";
    std::string kv_cache_type_v = "q8_0";
    // Explicit opt-in permitting q4_0 for K when the model supports Hadamard K rotation.
    bool kv_k_q4_allowed = false;
    // Flash Attention: auto (default, llama.cpp decides), on or off. A
    // quantized V cache needs it: "off" with a quantized V type is rejected.
    std::string flash_attention = "auto";
    // Explicit ggml device names (e.g. "CUDA0", "ROCm0", "Vulkan0", "CUDA1")
    // for vendor-targeted or multi-GPU offloading. Empty = llama.cpp default.
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
    // Speculative decoding verification policy: TargetMatch (default reference
    // path) vs opt-in StochasticRejection (Leviathan et al. sum min(p, q) +
    // residual (p - q)_+ resampling). StochasticRejection activates only when
    // a stochastic proposer supplies real post-processor proposal probabilities;
    // greedy MTP and deterministic suffix/n-gram proposers stay on TargetMatch.
    SpeculativeVerificationMode spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
    // Bounded L2 host-RAM prefix state cache budget in MiB (0 = disabled, max 65536).
    std::uint32_t cache_ram_mib = 0;
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
};

// Checks the llama.cpp context and placement options (KV cache types,
// flash_attention, ubatch_size vs batch_size, split_mode, main_gpu,
// tensor_split, device name syntax) without loading anything. load_model()
// applies the same checks; hosts call this to reject bad flags at startup.
// Errors: invalid_argument.
Status validate_llamacpp_options(const LlamaCppBackendOptions& options);

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options = {});

}  // namespace sonder::inference
