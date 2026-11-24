// Sonder Inference: pure, deterministic multi-model co-residency VRAM planner.
//
// Plans simultaneous co-residency of multiple GGUF models across the host's
// GPUs (discrete NVIDIA RTX / AMD RX XT, multi-GPU rigs, integrated UMA APUs)
// and host RAM by scheduling entries sequentially in input (priority) order
// through plan_model_placement() with live remaining-VRAM and host-RAM
// accounting:
//   - OS/desktop vram_headroom_mib is reserved once per GPU on first occupancy
//     rather than double-counted for subsequent co-resident models on the same
//     GPU.
//   - Discrete GPU allocations deduct from GPU VRAM while host_resident_bytes
//     deducts from host RAM; integrated UMA allocations deduct consistently
//     from both the UMA pool and shared physical host RAM without double-
//     counting.
//   - When a discrete GPU fills up and the entry did not pin a device_hint,
//     subsequent models naturally spill to a second GPU, an integrated UMA
//     iGPU (if it fits more layers), or CPU-only execution.
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
#include "sonder/inference/placement_planner.hpp"

namespace sonder::inference {

struct CoResidencyEntry {
    GgufModelInfo model;
    PlacementPlanRequest request;
};

struct CoResidencyPlan {
    // Per-model placement plans, scheduled sequentially in input (priority)
    // order and returned in the exact same input order.
    std::vector<PlacementPlan> models;
    // True when every model fits its assigned GPU/CPU budget simultaneously
    // without overflowing remaining VRAM or remaining host RAM (vacuously true
    // when `entries` is empty).
    bool all_fit_coresident = false;
    std::uint64_t aggregate_vram_mib = 0;
    std::uint64_t aggregate_host_ram_mib = 0;
    // Remaining usable MiB per GPU `id` after all placed models (and after the
    // single per-GPU OS reserve for each occupied GPU).
    std::vector<std::pair<std::string, std::uint64_t>> remaining_gpu_vram_mib;
    std::uint64_t remaining_host_ram_mib = 0;
    std::vector<std::string> notes;
    std::vector<std::string> warnings;
};

CoResidencyPlan plan_coresident_models(const std::vector<CoResidencyEntry>& entries,
                                       const std::vector<GpuDeviceInfo>& gpus, const HostMemory& host);

json::Object coresidency_plan_to_json(const CoResidencyPlan& plan);

}  // namespace sonder::inference
