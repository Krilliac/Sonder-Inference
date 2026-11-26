#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/coresidency_planner.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/placement_planner.hpp"
#include "sonder/inference/server.hpp"

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "sonder/inference/backends/llamaserver.hpp"
#endif

namespace sonder::inference::detail {

// Parse a JSON boolean (true/false) or string ("auto", "on", "off", "true",
// "false") into the canonical UMA mode setting ("on", "off", or "auto").
Status parse_uma_mode_value(const json::Value &value, const char *field_path, std::string &out);

using sonder::inference::extract_device_hint;
using sonder::inference::is_valid_uma_mode;
using sonder::inference::resolve_uma_mode;

// Parse one spill_guard.residency object into the server-only config mirror.
// Parsing remains available in builds without the optional backend API.
Status parse_residency_config(const json::Value &value, BackendSetup &setup);

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
// Map the server-only mirror to the optional backend API, with checked MiB
// conversion and validation for callers that construct BackendSetup directly.
Status apply_residency_config(const BackendSetup &setup, LlamaServerResidencyGuardOptions &out);
Status apply_residency_config(const BackendSetup &setup,
                              LlamaServerResidencyGuardOptions &out,
                              const std::vector<GpuDeviceInfo> &gpus);
#endif

// Builds a live ModelResidency pre-load ResourceFootprint breakdown from a
// VramEstimate (splitting projector bytes out of weights so nothing is
// double-counted, and preserving KV/recurrent, workspace/compute/MoE-cache,
// projector, and adapter components across both VRAM and host RAM tiers).
server::ModelResourceFootprint footprint_from_vram_estimate(const VramEstimate &estimate,
                                                            std::uint64_t projector_vram_bytes = 0,
                                                            std::uint64_t projector_ram_bytes = 0,
                                                            std::uint64_t adapter_vram_bytes = 0,
                                                            std::uint64_t adapter_ram_bytes = 0) noexcept;

// Builds a live ModelResidency pre-load footprint from one PlacementPlan
// (accounting for integrated UMA shared physical RAM without double-counting).
server::ModelResourceFootprint footprint_from_placement_plan(const PlacementPlan &plan,
                                                             std::uint64_t projector_bytes = 0,
                                                             std::uint64_t adapter_vram_bytes = 0,
                                                             std::uint64_t adapter_ram_bytes = 0) noexcept;

// Projects a multi-model CoResidencyPlan into ordered (model_id, footprint)
// entries ready to seed live ModelResidency pre-load byte reservations.
std::vector<std::pair<std::string, server::ModelResourceFootprint>> footprints_from_coresidency_plan(
    const CoResidencyPlan &plan);

// Parses optional live ModelResidency byte-budget JSON controls
// (`max_resident_vram_mib`, `max_resident_ram_mib`, `max_resident_vram_bytes`,
// `max_resident_ram_bytes`). Rejects unknown fields and conflicting units.
Status parse_live_residency_budgets(const json::Value &value, std::uint64_t &max_vram_bytes,
                                    std::uint64_t &max_ram_bytes);

}  // namespace sonder::inference::detail
