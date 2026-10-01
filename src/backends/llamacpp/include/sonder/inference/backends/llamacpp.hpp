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
    // KV cache element types: f16 (default), f32, bf16, q8_0, q5_1, q5_0,
    // q4_1, q4_0, iq4_nl. q8_0 roughly halves KV memory, q4_0 roughly quarters
    // it (docs/PLACEMENT.md, "KV cache types").
    std::string kv_cache_type_k = "f16";
    std::string kv_cache_type_v = "f16";
    // Flash Attention: auto (default, llama.cpp decides), on or off. A
    // quantized V cache needs it: "off" with a quantized V type is rejected.
    std::string flash_attention = "auto";
};

// Checks the llama.cpp context options (KV cache types, flash_attention,
// ubatch_size vs batch_size) without loading anything. load_model() applies
// the same checks; hosts call this to reject bad flags at startup.
// Errors: invalid_argument.
Status validate_llamacpp_options(const LlamaCppBackendOptions& options);

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options = {});

}  // namespace sonder::inference
