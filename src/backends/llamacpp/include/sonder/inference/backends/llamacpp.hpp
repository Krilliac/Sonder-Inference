// Sonder Inference: direct llama.cpp/GGML backend factory (module
// src/backends/llamacpp, built when SONDER_WITH_LLAMA_CPP=ON).
//
// Available only when SONDER_HAS_LLAMACPP_BACKEND is defined.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
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
};

std::shared_ptr<Backend> make_llamacpp_backend(LlamaCppBackendOptions options = {});

}  // namespace sonder::inference
