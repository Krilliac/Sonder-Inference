// Sonder Inference: pure, deterministic per-model hardware placement planner
// (docs/PLACEMENT.md step 2).
//
// Given a parsed GGUF header (GgufModelInfo), discovered GPU inventory
// (GpuDeviceInfo), host memory snapshot (HostMemory), and caller preferences
// (PlacementPlanRequest), computes a validated LaunchProfile and exact
// VramEstimate covering NVIDIA RTX, AMD Radeon RX XT discrete, AMD Ryzen APU
// UMA, multi-GPU layer split, MoE selective CPU expert offload (--n-cpu-moe),
// ternary / PrismML Bonsai runtime requirements, speculative decoding, and
// CPU-only fallback.
//
// Pure function: performs no filesystem I/O, no GPU probes, and no process
// launches; emits no fabricated throughput numbers.
// Part of the server module (SONDER_HAS_SERVER).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"

namespace sonder::inference {

struct PlacementPlanRequest {
    std::string profile_name = "planned";
    std::string model_path;
    std::string mmproj_path;
    std::uint64_t mmproj_bytes = 0;
    std::uint32_t target_ctx_size = 32768;  // clamped to model training context if smaller and known
    std::uint32_t min_ctx_size = 8192;
    std::uint32_t parallel = 1;
    std::uint64_t vram_headroom_mib = 1536;  // desktop/OS reserve per discrete GPU; smaller/proportional on UMA
    bool prefer_free_vram = true;
    bool allow_quantized_kv = true;
    bool allow_speculative = true;
    bool allow_moe_cpu_offload = true;
    std::string device_hint;  // optional e.g. "CUDA0", "ROCm0", "Vulkan0", "multi", "cpu"
};

struct PlacementPlan {
    LaunchProfile profile;  // ready-to-validate LaunchProfile (passes validate_launch_profile)
    VramEstimate estimate;  // exact estimate_vram() output for profile
    std::uint64_t budget_mib = 0;  // selected per-GPU, aggregate multi-GPU, UMA, or CPU host budget
    bool fits_in_vram = false;
    bool fits_in_host_ram = true;
    bool uma_mode = false;
    bool requires_prism_runtime = false;
    GpuVendor selected_vendor = GpuVendor::unknown;
    std::string recommended_backend_hint;  // "cuda", "rocm", "vulkan", "metal", "sycl", "cpu"
    // Strictly env vars permitted by backends/llamaserver/env_policy.hpp.
    std::vector<std::pair<std::string, std::string>> recommended_env;
    std::vector<std::string> reasons;   // concrete measured/arithmetic rationale for every decision
    std::vector<std::string> warnings;  // runtime, headroom, UMA bandwidth, or partial-offload cautions
};

PlacementPlan plan_model_placement(const GgufModelInfo& model, const std::vector<GpuDeviceInfo>& gpus,
                                   const HostMemory& host, const PlacementPlanRequest& request = {});

json::Object placement_plan_to_json(const PlacementPlan& plan);

}  // namespace sonder::inference
