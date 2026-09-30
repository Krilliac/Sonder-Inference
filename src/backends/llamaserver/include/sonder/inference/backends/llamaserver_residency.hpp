// Pure policy for detecting missing CUDA offload and post-readiness VRAM
// eviction.  Counter collection and process supervision remain outside this
// header so the policy can be tested without a GPU or a child process.
#pragma once

#include <cstddef>
#include <cstdint>

namespace sonder::inference {

enum class LlamaServerEvictionPolicy { warn, restart };

struct LlamaServerResidencyGuardOptions {
    bool enabled = true;
    // Used only when argv has no explicit GPU-layer setting; zero wins.
    bool expect_gpu = false;
    std::uint64_t min_dedicated_bytes = 512ull * 1024 * 1024;
    std::size_t consecutive_samples = 3;
    // Strict fraction OR byte drop from the post-readiness high-water mark.
    // Zero disables one dimension; at least one must remain nonzero.
    double eviction_fraction = 0.25;
    std::uint64_t eviction_bytes = 2ull * 1024 * 1024 * 1024;
    LlamaServerEvictionPolicy on_eviction = LlamaServerEvictionPolicy::warn;
    // Independent supervisor-lifetime budget, never reset by relaunches.
    std::size_t max_eviction_restarts = 1;
};

}  // namespace sonder::inference
