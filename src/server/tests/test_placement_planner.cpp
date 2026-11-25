// Unit tests for the per-model hardware placement planner
// (sonder/inference/placement_planner.hpp).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backends/llamaserver/env_policy.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/placement_planner.hpp"

namespace si = sonder::inference;
namespace json = sonder::inference::json;

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

bool any_contains(const std::vector<std::string>& list, const std::string& needle) {
    return std::any_of(list.begin(), list.end(), [&](const std::string& s) { return contains(s, needle); });
}

std::optional<std::string> find_env(const std::vector<std::pair<std::string, std::string>>& env,
                                    const std::string& key) {
    for (const auto& [k, v] : env) {
        if (k == key) return v;
    }
    return std::nullopt;
}

void verify_plan_contracts(const si::PlacementPlan& plan) {
    const si::Status valid = si::validate_launch_profile(plan.profile);
    CHECK_MESSAGE(valid.ok(), valid.message());
    const auto argv = si::llamaserver_arguments(plan.profile);
    CHECK_MESSAGE(argv.ok(), argv.status().message());

    // Every recommended_env entry must satisfy env_policy.hpp.
    for (const auto& [k, v] : plan.recommended_env) {
        (void)v;
        CHECK_MESSAGE(si::llamaserver::is_allowed_profile_env_override(k), "disallowed env var: " << k);
    }
    const si::Status env_st = si::llamaserver::check_child_environment_overrides(plan.recommended_env);
    CHECK_MESSAGE(env_st.ok(), env_st.message());

    // placement_plan_to_json().profile must round-trip through parse_launch_profiles().
    const json::Object doc = si::placement_plan_to_json(plan);
    const json::Value* prof_val = doc.find("profile");
    REQUIRE(prof_val != nullptr);
    json::Object wrapper;
    json::Array list;
    list.push_back(*prof_val);
    wrapper.set("profiles", std::move(list));
    const auto reparsed = si::parse_launch_profiles(json::Value(std::move(wrapper)));
    CHECK_MESSAGE(reparsed.ok(), reparsed.status().message());
}

si::HostMemory host_ram(std::uint64_t total_gib, std::uint64_t avail_gib) {
    si::HostMemory h;
    h.total_physical = si::MemoryReading::known(total_gib * kGiB);
    h.available_physical = si::MemoryReading::known(avail_gib * kGiB);
    h.source = "test";
    return h;
}

si::GpuDeviceInfo make_rtx_5070_ti(std::string id = "gpu:0", std::uint64_t free_mib = 15360) {
    si::GpuDeviceInfo g;
    g.id = std::move(id);
    g.name = "NVIDIA GeForce RTX 5070 Ti";
    g.vendor = si::GpuVendor::nvidia;
    g.vendor_id = 0x10DEu;
    g.device_id = 0x2C05u;
    g.dedicated_total = si::MemoryReading::known(16 * kGiB);
    g.dedicated_free = si::MemoryReading::known(free_mib * kMiB);
    g.shared_total = si::MemoryReading::known(16 * kGiB);
    g.is_integrated_uma = false;
    g.backend_hint = "cuda";
    g.source = "dxgi";
    return g;
}

si::GpuDeviceInfo make_amd_discrete(std::string id, std::string name, std::uint64_t total_gib,
                                    std::uint64_t free_mib, std::string backend_hint) {
    si::GpuDeviceInfo g;
    g.id = std::move(id);
    g.name = std::move(name);
    g.vendor = si::GpuVendor::amd;
    g.vendor_id = 0x1002u;
    g.dedicated_total = si::MemoryReading::known(total_gib * kGiB);
    g.dedicated_free = si::MemoryReading::known(free_mib * kMiB);
    g.shared_total = si::MemoryReading::known(16 * kGiB);
    g.is_integrated_uma = false;
    g.backend_hint = std::move(backend_hint);
    g.source = "amdgpu-sysfs";
    return g;
}

si::GpuDeviceInfo make_ryzen_780m_apu(std::string id = "gpu:0", std::string backend_hint = "vulkan") {
    si::GpuDeviceInfo g;
    g.id = std::move(id);
    g.name = "AMD Radeon 780M Graphics";
    g.vendor = si::GpuVendor::amd;
    g.vendor_id = 0x1002u;
    g.device_id = 0x15BFu;
    g.dedicated_total = si::MemoryReading::known(512 * kMiB);
    g.shared_total = si::MemoryReading::known(24 * kGiB);
    g.is_integrated_uma = true;
    g.backend_hint = std::move(backend_hint);
    g.source = "dxgi";
    return g;
}

// Ternary Bonsai-2-27B: PQ2_0 + prism_hadamard + hybrid qwen35 DeltaNet (~7.2 GiB weights).
si::GgufModelInfo bonsai_2_27b_model() {
    si::GgufModelInfo m;
    m.architecture = "qwen35";
    m.kind = si::ModelArchitecture::hybrid;
    m.block_count = 64;
    m.nextn_layers = 0;
    m.embedding_length = 5120;
    m.vocab_size = 152064;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads.assign(64, 0);
    for (std::uint32_t i = 3; i < 64; i += 4) m.kv_heads[i] = 4;  // 16 full-attention blocks, 48 DeltaNet
    m.block_bytes.assign(64, 112 * kMiB);
    m.output_bytes = 256 * kMiB;
    m.embedding_bytes = 256 * kMiB;
    m.recurrent_state_values = 512ull * 128ull;  // 256 KiB per recurrent block per sequence/snapshot
    m.metadata = {{"general.architecture", "qwen35"}, {"qwen35.context_length", "131072"}};
    m.prism_hadamard = true;
    m.prism_hadamard_transform = "fwht";
    m.prism_hadamard_block_size = 4096;
    m.has_ternary_tensors = true;
    m.has_prism_private_tensors = true;
    return m;
}

// Ternary Bonsai-2-9B-MTP: PQ2_0 + nextn_layers=1 + hybrid qwen35 DeltaNet (~2.5 GiB weights).
si::GgufModelInfo bonsai_2_9b_mtp_model() {
    si::GgufModelInfo m;
    m.architecture = "qwen35";
    m.kind = si::ModelArchitecture::hybrid;
    m.block_count = 33;  // 32 base + 1 trailing MTP block
    m.nextn_layers = 1;
    m.embedding_length = 4096;
    m.vocab_size = 152064;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads.assign(33, 0);
    for (std::uint32_t i = 3; i < 32; i += 4) m.kv_heads[i] = 4;  // 8 attention, 24 recurrent
    m.kv_heads[32] = 4;  // MTP trailing block
    m.block_bytes.assign(33, 76 * kMiB);
    m.output_bytes = 128 * kMiB;
    m.embedding_bytes = 128 * kMiB;
    m.recurrent_state_values = 1024ull * 128ull;  // 512 KiB per recurrent block per sequence/snapshot
    m.metadata = {{"general.architecture", "qwen35"}, {"qwen35.context_length", "131072"}};
    m.prism_hadamard = true;
    m.prism_hadamard_transform = "fwht";
    m.prism_hadamard_block_size = 4096;
    m.has_ternary_tensors = true;
    m.has_prism_private_tensors = true;
    return m;
}

si::GgufModelInfo dense_model(std::uint32_t blocks, std::uint64_t block_mib, std::uint32_t kv_heads,
                              std::uint32_t train_ctx = 65536) {
    si::GgufModelInfo m;
    m.architecture = "llama";
    m.kind = si::ModelArchitecture::attention_only;
    m.block_count = blocks;
    m.embedding_length = 4096;
    m.vocab_size = 32000;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads.assign(blocks, kv_heads);
    m.block_bytes.assign(blocks, block_mib * kMiB);
    m.output_bytes = 256 * kMiB;
    m.embedding_bytes = 256 * kMiB;
    m.metadata = {{"general.architecture", "llama"}, {"llama.context_length", std::to_string(train_ctx)}};
    return m;
}

// 12-block MoE (~18.75 GiB total: 100 MiB attention + 1500 MiB routed experts per block).
si::GgufModelInfo large_moe_model() {
    si::GgufModelInfo m;
    m.architecture = "qwen3moe";
    m.kind = si::ModelArchitecture::attention_only;
    m.block_count = 12;
    m.embedding_length = 2048;
    m.vocab_size = 32000;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads.assign(12, 4);
    m.metadata = {{"general.architecture", "qwen3moe"}, {"qwen3moe.context_length", "65536"}};
    m.routed_experts = si::RoutedExpertCounts{8, 2};
    m.block_bytes.assign(12, 0);
    const auto add = [&](const std::string& name, std::uint64_t mib, std::uint32_t blk, bool expert,
                         std::uint32_t type) {
        si::GgufTensorInfo t;
        t.name = name;
        t.bytes = mib * kMiB;
        t.block = blk;
        t.expert = expert;
        t.type = type;
        m.block_bytes[blk] += t.bytes;
        m.tensors.push_back(std::move(t));
    };
    for (std::uint32_t b = 0; b < 12; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        add(p + "attn_q.weight", 100, b, false, 12);
        add(p + "ffn_up_exps.weight", 750, b, true, 12);
        add(p + "ffn_down_exps.weight", 750, b, true, 12);
    }
    m.output_bytes = 200 * kMiB;
    m.embedding_bytes = 200 * kMiB;
    return m;
}

}  // namespace

TEST_CASE("placement planner: Ternary Bonsai-2-27B on RTX 5070 Ti fits on GPU with PrismML and ngram-mod") {
    const si::GgufModelInfo model = bonsai_2_27b_model();
    // Put Ryzen iGPU first to prove discrete RTX 5070 Ti is still preferred by default.
    const std::vector<si::GpuDeviceInfo> gpus{make_ryzen_780m_apu("gpu:0"), make_rtx_5070_ti("gpu:1")};
    si::PlacementPlanRequest req;
    req.profile_name = "bonsai-2-27b";
    req.model_path = "D:/models/Bonsai-2-27B-PQ2_0.gguf";
    req.target_ctx_size = 32768;

    const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(32, 24), req);
    verify_plan_contracts(plan);

    CHECK(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK_FALSE(plan.uma_mode);
    CHECK(plan.requires_prism_runtime);
    CHECK(plan.selected_vendor == si::GpuVendor::nvidia);
    CHECK(plan.recommended_backend_hint == "cuda");
    CHECK(plan.budget_mib == 15360u - 1536u);
    CHECK(plan.profile.n_gpu_layers == -1);
    CHECK(plan.profile.ctx_size == 32768u);
    CHECK(plan.profile.flash_attn == std::string("on"));
    CHECK(plan.profile.cache_type_k == std::string("q8_0"));
    CHECK(plan.profile.cache_type_v == std::string("q8_0"));
    CHECK(plan.profile.load_mode == std::string("none"));
    CHECK(plan.profile.kv_unified == true);
    CHECK(plan.profile.device == std::string("CUDA0"));
    REQUIRE(plan.profile.speculative.has_value());
    CHECK(plan.profile.speculative->types == std::vector<std::string>{"ngram-mod"});
    CHECK(plan.estimate.total_mib() <= plan.budget_mib);
    CHECK(plan.estimate.requires_prism_runtime);
    CHECK(any_contains(plan.warnings, "PrismML"));
    CHECK(any_contains(plan.warnings, "PQ2_0/PTQ1_0"));
    CHECK_FALSE(plan.reasons.empty());
}

TEST_CASE("placement planner: Ternary Bonsai-2-9B-MTP on RTX 5070 Ti enables draft-mtp and rollback snapshots") {
    const si::GgufModelInfo model = bonsai_2_9b_mtp_model();
    const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0")};
    si::PlacementPlanRequest req;
    req.profile_name = "bonsai-2-9b-mtp";
    req.model_path = "D:/models/Bonsai-2-9B-MTP-PQ2_0.gguf";
    req.target_ctx_size = 32768;

    const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(32, 24), req);
    verify_plan_contracts(plan);

    CHECK(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK(plan.requires_prism_runtime);
    CHECK(plan.profile.n_gpu_layers == -1);
    REQUIRE(plan.profile.speculative.has_value());
    CHECK(plan.profile.speculative->types == std::vector<std::string>{"draft-mtp"});
    CHECK(plan.profile.speculative->draft_n_max == 3u);
    CHECK(plan.estimate.recurrent_snapshots == 4u);
    CHECK(plan.estimate.gpu_layers == 33u);

    // Compare directly against allow_speculative=false to prove MTP snapshots and block are accounted for.
    si::PlacementPlanRequest no_spec_req = req;
    no_spec_req.allow_speculative = false;
    const si::PlacementPlan no_spec = si::plan_model_placement(model, gpus, host_ram(32, 24), no_spec_req);
    verify_plan_contracts(no_spec);
    CHECK_FALSE(no_spec.profile.speculative.has_value());
    CHECK(no_spec.estimate.recurrent_snapshots == 1u);
    CHECK(no_spec.estimate.gpu_layers == 32u);
    CHECK(plan.estimate.recurrent_bytes == 4u * no_spec.estimate.recurrent_bytes);
    CHECK(plan.estimate.weights_bytes > no_spec.estimate.weights_bytes);
}

TEST_CASE("placement planner: AMD RX 7900 XT and RX 9070 XT discrete defaults and GPU_MAX_ALLOC_PERCENT") {
    const si::GgufModelInfo model = dense_model(32, 220, 4);  // ~7 GiB weights
    si::PlacementPlanRequest req;
    req.profile_name = "amd-xt";
    req.model_path = "/models/model-q4.gguf";

    // RX 7900 XT (20 GiB, ROCm)
    {
        const std::vector<si::GpuDeviceInfo> gpus{
            make_amd_discrete("gpu:0", "AMD Radeon RX 7900 XT", 20, 19456, "rocm")};
        const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(64, 48), req);
        verify_plan_contracts(plan);

        CHECK(plan.fits_in_vram);
        CHECK(plan.fits_in_host_ram);
        CHECK_FALSE(plan.uma_mode);
        CHECK(plan.selected_vendor == si::GpuVendor::amd);
        CHECK(plan.recommended_backend_hint == "rocm");
        CHECK(plan.profile.flash_attn == std::string("auto"));
        CHECK(plan.profile.device == std::string("ROCm0"));
        CHECK(plan.profile.cache_type_k == std::string("q8_0"));
        CHECK(plan.profile.cache_type_v == std::string("q8_0"));
        CHECK(find_env(plan.recommended_env, "GPU_MAX_ALLOC_PERCENT") == std::string("100"));
    }

    // RX 9070 XT (16 GiB, Vulkan)
    {
        const std::vector<si::GpuDeviceInfo> gpus{
            make_amd_discrete("gpu:0", "AMD Radeon RX 9070 XT", 16, 15360, "vulkan")};
        const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(32, 24), req);
        verify_plan_contracts(plan);

        CHECK(plan.fits_in_vram);
        CHECK_FALSE(plan.uma_mode);
        CHECK(plan.selected_vendor == si::GpuVendor::amd);
        CHECK(plan.recommended_backend_hint == "vulkan");
        CHECK(plan.profile.flash_attn == std::string("auto"));
        CHECK(plan.profile.device == std::string("Vulkan0"));
        CHECK_FALSE(find_env(plan.recommended_env, "GPU_MAX_ALLOC_PERCENT").has_value());
    }

    // Reordered ROCm runtime ordinal (e.g. ROCR_VISIBLE_DEVICES=1,0) must emit the mapped ROCmN name.
    {
        si::GpuDeviceInfo card0 = make_amd_discrete("gpu:0", "AMD Radeon RX 7900 XT", 20, 19456, "rocm");
        card0.rocm_runtime_ordinal = 1;
        si::GpuDeviceInfo card1 = make_amd_discrete("gpu:1", "AMD Radeon RX 7900 XT", 20, 19456, "rocm");
        card1.rocm_runtime_ordinal = 0;
        si::PlacementPlanRequest pinned = req;
        pinned.device_hint = "gpu:0";
        const si::PlacementPlan plan = si::plan_model_placement(model, {card0, card1}, host_ram(64, 48), pinned);
        verify_plan_contracts(plan);
        CHECK(plan.profile.device == std::string("ROCm1"));
    }
}

TEST_CASE("placement planner: AMD Ryzen 780M APU uses UMA pool and respects available host RAM") {
    const si::GgufModelInfo model = dense_model(32, 280, 4);  // ~9 GiB weights (far above 512 MiB carveout)
    const std::vector<si::GpuDeviceInfo> gpus{make_ryzen_780m_apu("gpu:0", "vulkan")};
    si::PlacementPlanRequest req;
    req.profile_name = "ryzen-780m";
    req.model_path = "D:/models/llama-14b.gguf";
    req.target_ctx_size = 32768;

    const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(32, 26), req);
    verify_plan_contracts(plan);

    CHECK(plan.uma_mode);
    CHECK(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK(plan.selected_vendor == si::GpuVendor::amd);
    CHECK(plan.recommended_backend_hint == "vulkan");
    CHECK(plan.profile.flash_attn == std::string("auto"));
    CHECK(plan.profile.load_mode == std::string("auto"));
    CHECK(plan.profile.n_gpu_layers == -1);
    CHECK(plan.budget_mib > 20000u);  // Uses 512 MiB + 24 GiB UMA pool, NOT 512 MiB carveout
    CHECK(plan.estimate.total_mib() > 8000u);
    CHECK(plan.estimate.total_mib() <= plan.budget_mib);
    CHECK(any_contains(plan.warnings, "bandwidth"));

    // UMA budget must never exceed available physical host RAM.
    const si::PlacementPlan capped = si::plan_model_placement(model, gpus, host_ram(32, 12), req);
    verify_plan_contracts(capped);
    CHECK(capped.uma_mode);
    CHECK(capped.budget_mib <= 12u * 1024u);
    CHECK(any_contains(capped.reasons, "capped by available host RAM"));

    // Explicit device_hint can select the iGPU even when a discrete RTX is present.
    const std::vector<si::GpuDeviceInfo> mixed{make_rtx_5070_ti("gpu:0"), make_ryzen_780m_apu("gpu:1", "vulkan")};
    si::PlacementPlanRequest igpu_req = req;
    igpu_req.device_hint = "Vulkan1";
    const si::PlacementPlan targeted = si::plan_model_placement(model, mixed, host_ram(32, 26), igpu_req);
    verify_plan_contracts(targeted);
    CHECK(targeted.uma_mode);
    CHECK(targeted.selected_vendor == si::GpuVendor::amd);
    CHECK(targeted.profile.device == std::string("Vulkan1"));
}

TEST_CASE("placement planner: Dual RTX splits 22 GB model with split_mode layer and proportional tensor_split") {
    const si::GgufModelInfo model = dense_model(44, 512, 4);  // 22 GiB weights
    const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 15872), make_rtx_5070_ti("gpu:1", 15872)};
    si::PlacementPlanRequest req;
    req.profile_name = "dual-rtx-22g";
    req.model_path = "D:/models/model-22g.gguf";
    req.target_ctx_size = 32768;

    const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(64, 48), req);
    verify_plan_contracts(plan);

    CHECK(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK(plan.selected_vendor == si::GpuVendor::nvidia);
    CHECK(plan.profile.n_gpu_layers == -1);
    CHECK(plan.profile.device == std::string("CUDA0,CUDA1"));
    CHECK(plan.profile.split_mode == std::string("layer"));
    REQUIRE(plan.profile.tensor_split.size() == 2u);
    CHECK(plan.profile.tensor_split[0] == doctest::Approx(0.5));
    CHECK(plan.profile.tensor_split[1] == doctest::Approx(0.5));
    CHECK(plan.estimate.max_gpu_bytes > 0u);
    REQUIRE(plan.profile.vram_budget_mib.has_value());
    CHECK(plan.estimate.max_gpu_mib() <= *plan.profile.vram_budget_mib);
    CHECK(plan.budget_mib == 2u * (15872u - 1536u));

    // Heterogeneous Dual RTX (24 GiB + 12 GiB) produces proportional tensor_split.
    si::GpuDeviceInfo rtx_24 = make_rtx_5070_ti("gpu:0", 24576);
    rtx_24.name = "NVIDIA GeForce RTX 4090";
    rtx_24.dedicated_total = si::MemoryReading::known(24 * kGiB);
    si::GpuDeviceInfo rtx_12 = make_rtx_5070_ti("gpu:1", 12288);
    rtx_12.name = "NVIDIA GeForce RTX 3060";
    rtx_12.dedicated_total = si::MemoryReading::known(12 * kGiB);
    si::PlacementPlanRequest multi_req = req;
    multi_req.device_hint = "multi";
    const si::PlacementPlan hetero =
        si::plan_model_placement(model, std::vector<si::GpuDeviceInfo>{rtx_24, rtx_12}, host_ram(64, 48), multi_req);
    verify_plan_contracts(hetero);
    CHECK(hetero.fits_in_vram);
    CHECK(hetero.profile.split_mode == std::string("layer"));
    REQUIRE(hetero.profile.tensor_split.size() == 2u);
    const double u0 = static_cast<double>(24576u - 1536u);
    const double u1 = static_cast<double>(12288u - 1536u);
    CHECK(hetero.profile.tensor_split[0] == doctest::Approx(u0 / (u0 + u1)));
    CHECK(hetero.profile.tensor_split[1] == doctest::Approx(u1 / (u0 + u1)));

    // A small model on Dual RTX without device_hint="multi" stays single-GPU.
    const si::GgufModelInfo small_m = dense_model(16, 256, 4);  // 4 GiB
    const si::PlacementPlan single = si::plan_model_placement(small_m, gpus, host_ram(64, 48), req);
    verify_plan_contracts(single);
    CHECK(single.fits_in_vram);
    CHECK(single.profile.device == std::string("CUDA0"));
    CHECK_FALSE(single.profile.split_mode.has_value());
    CHECK(single.profile.tensor_split.empty());
}

TEST_CASE("placement planner: MoE overflowing 16 GB VRAM picks --n-cpu-moe to fit VRAM and host RAM") {
    const si::GgufModelInfo model = large_moe_model();
    const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 15872)};
    si::PlacementPlanRequest req;
    req.profile_name = "qwen3-moe";
    req.model_path = "D:/models/qwen3-moe.gguf";
    req.target_ctx_size = 32768;

    const si::PlacementPlan plan = si::plan_model_placement(model, gpus, host_ram(32, 28), req);
    verify_plan_contracts(plan);

    CHECK(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK(plan.profile.n_gpu_layers == -1);
    CHECK(plan.profile.ctx_size == 32768u);
    CHECK(plan.profile.load_mode == std::string("auto"));
    CHECK(plan.estimate.total_mib() <= plan.budget_mib);
    CHECK(plan.estimate.host_expert_bytes > 0u);

    const auto parsed = si::parse_moe_placement(plan.profile.extra_args);
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    REQUIRE(parsed->rules.size() == 1u);
    CHECK(parsed->rules[0].kind == si::TensorPlacementRule::Kind::first_n_experts);
    CHECK(parsed->rules[0].count >= 1u);
    CHECK(parsed->rules[0].count < model.block_count);
    CHECK(any_contains(plan.reasons, "--n-cpu-moe"));
}

TEST_CASE("placement planner: solver steps KV to q4_0, aligns ctx_size to 4096, and falls back to partial layers") {
    const std::vector<si::GpuDeviceInfo> rtx{make_rtx_5070_ti("gpu:0", 15360)};  // usable = 13824 MiB

    // 1. Fits at target_ctx after stepping KV from q8_0/q8_0 to q4_0/q4_0.
    {
        const si::GgufModelInfo m = dense_model(32, 350, 8);
        si::PlacementPlanRequest req;
        req.profile_name = "kv-q4-fit";
        req.model_path = "m.gguf";
        req.target_ctx_size = 32768;
        const si::PlacementPlan plan = si::plan_model_placement(m, rtx, host_ram(32, 24), req);
        verify_plan_contracts(plan);

        CHECK(plan.fits_in_vram);
        CHECK(plan.profile.n_gpu_layers == -1);
        CHECK(plan.profile.ctx_size == 32768u);
        CHECK(plan.profile.cache_type_k == std::string("q4_0"));
        CHECK(plan.profile.cache_type_v == std::string("q4_0"));
        CHECK(plan.estimate.total_mib() <= plan.budget_mib);
        CHECK(any_contains(plan.warnings, "q4_0"));
    }

    // 2. Steps ctx_size down on 4096 boundary when q4_0 at target_ctx still overflows.
    {
        const si::GgufModelInfo m = dense_model(32, 380, 8);
        si::PlacementPlanRequest req;
        req.profile_name = "ctx-stepdown";
        req.model_path = "m.gguf";
        req.target_ctx_size = 32768;
        req.min_ctx_size = 8192;
        const si::PlacementPlan plan = si::plan_model_placement(m, rtx, host_ram(32, 24), req);
        verify_plan_contracts(plan);

        CHECK(plan.fits_in_vram);
        CHECK(plan.profile.n_gpu_layers == -1);
        REQUIRE(plan.profile.ctx_size.has_value());
        CHECK(*plan.profile.ctx_size < 32768u);
        CHECK(*plan.profile.ctx_size >= 8192u);
        CHECK((*plan.profile.ctx_size % 4096u) == 0u);
        CHECK(plan.estimate.total_mib() <= plan.budget_mib);
    }

    // 3. Training context clamp: target_ctx_size > training context clamps cleanly.
    {
        const si::GgufModelInfo m = dense_model(16, 200, 4, 16384);
        si::PlacementPlanRequest req;
        req.profile_name = "train-clamp";
        req.model_path = "m.gguf";
        req.target_ctx_size = 65536;
        const si::PlacementPlan plan = si::plan_model_placement(m, rtx, host_ram(32, 24), req);
        verify_plan_contracts(plan);
        CHECK(plan.profile.ctx_size == 16384u);
        CHECK(any_contains(plan.reasons, "training context"));
    }

    // 4. Last resort: oversized dense model reduces n_gpu_layers to max fitting count.
    {
        si::GpuDeviceInfo gpu_8g = make_rtx_5070_ti("gpu:0", 8192);
        gpu_8g.dedicated_total = si::MemoryReading::known(8 * kGiB);
        const si::GgufModelInfo m = dense_model(20, 500, 4);  // 10 GiB weights on 6.5 GiB budget
        si::PlacementPlanRequest req;
        req.profile_name = "partial-ngl";
        req.model_path = "m.gguf";
        req.target_ctx_size = 32768;
        req.min_ctx_size = 8192;
        const si::PlacementPlan plan =
            si::plan_model_placement(m, std::vector<si::GpuDeviceInfo>{gpu_8g}, host_ram(32, 24), req);
        verify_plan_contracts(plan);

        CHECK(plan.fits_in_vram);
        REQUIRE(plan.profile.n_gpu_layers.has_value());
        CHECK(*plan.profile.n_gpu_layers > 0);
        CHECK(*plan.profile.n_gpu_layers < 20);
        CHECK(plan.profile.ctx_size == 8192u);
        CHECK(plan.profile.load_mode == std::string("auto"));
        CHECK(plan.estimate.total_mib() <= plan.budget_mib);
        CHECK(any_contains(plan.warnings, "partial GPU offload"));

        // One more GPU layer must strictly exceed the budget (proves optimality).
        si::LaunchProfile plus_one = plan.profile;
        plus_one.n_gpu_layers = *plan.profile.n_gpu_layers + 1;
        CHECK(si::estimate_vram(m, plus_one).total_mib() > plan.budget_mib);
    }
}

TEST_CASE("placement planner: CPU-only fallback and placement_plan_to_json serialization") {
    const si::GgufModelInfo model = dense_model(16, 256, 4);  // ~4 GiB weights
    si::PlacementPlanRequest req;
    req.profile_name = "cpu-fallback";
    req.model_path = "/models/small.gguf";
    req.target_ctx_size = 16384;

    const si::PlacementPlan plan = si::plan_model_placement(model, {}, host_ram(32, 16), req);
    verify_plan_contracts(plan);

    CHECK_FALSE(plan.fits_in_vram);
    CHECK(plan.fits_in_host_ram);
    CHECK_FALSE(plan.uma_mode);
    CHECK(plan.selected_vendor == si::GpuVendor::unknown);
    CHECK(plan.recommended_backend_hint == "cpu");
    CHECK(plan.profile.n_gpu_layers == 0);
    CHECK_FALSE(plan.profile.device.has_value());
    CHECK_FALSE(plan.profile.vram_budget_mib.has_value());
    CHECK(plan.budget_mib == 16u * 1024u);
    CHECK(plan.estimate.total_bytes == 0u);
    CHECK(plan.estimate.host_resident_bytes > 0u);

    // Tight host RAM flags fits_in_host_ram = false cleanly.
    const si::PlacementPlan oom_host = si::plan_model_placement(model, {}, host_ram(4, 2), req);
    verify_plan_contracts(oom_host);
    CHECK_FALSE(oom_host.fits_in_vram);
    CHECK_FALSE(oom_host.fits_in_host_ram);
    CHECK(any_contains(oom_host.warnings, "host RAM"));

    // Default PlacementPlanRequest{} (empty model_path) still produces a valid LaunchProfile.
    const si::PlacementPlan default_req_plan = si::plan_model_placement(model, {}, host_ram(32, 16));
    verify_plan_contracts(default_req_plan);
    CHECK(default_req_plan.profile.name == "planned");

    // Full JSON schema and round-trip parse verification.
    const json::Object obj = si::placement_plan_to_json(plan);
    const std::string dumped = json::Value(obj).dump();
    const auto reparsed = json::parse(dumped);
    REQUIRE_MESSAGE(reparsed.ok(), reparsed.status().message());
    const json::Object& root = reparsed->as_object();
    REQUIRE(root.find("budget_mib") != nullptr);
    CHECK(root.find("budget_mib")->as_uint() == plan.budget_mib);
    REQUIRE(root.find("fits_in_vram") != nullptr);
    CHECK_FALSE(root.find("fits_in_vram")->as_bool());
    REQUIRE(root.find("fits_in_host_ram") != nullptr);
    CHECK(root.find("fits_in_host_ram")->as_bool());
    REQUIRE(root.find("uma_mode") != nullptr);
    CHECK_FALSE(root.find("uma_mode")->as_bool());
    REQUIRE(root.find("requires_prism_runtime") != nullptr);
    CHECK_FALSE(root.find("requires_prism_runtime")->as_bool());
    REQUIRE(root.find("selected_vendor") != nullptr);
    CHECK(root.find("selected_vendor")->as_string() == "unknown");
    REQUIRE(root.find("recommended_backend_hint") != nullptr);
    CHECK(root.find("recommended_backend_hint")->as_string() == "cpu");
    REQUIRE(root.find("profile") != nullptr);
    CHECK(root.find("profile")->is_object());
    REQUIRE(root.find("estimate") != nullptr);
    CHECK(root.find("estimate")->is_object());
    REQUIRE(root.find("estimate")->find("host_resident_mib") != nullptr);
    CHECK(root.find("estimate")->find("host_resident_mib")->as_uint() == plan.estimate.host_resident_mib());
    REQUIRE(root.find("recommended_env") != nullptr);
    CHECK(root.find("recommended_env")->is_object());
    REQUIRE(root.find("reasons") != nullptr);
    CHECK_FALSE(root.find("reasons")->as_array().empty());
    REQUIRE(root.find("warnings") != nullptr);
    CHECK(root.find("warnings")->is_array());
}

TEST_CASE("placement planner disables mmproj offload on CPU fallback and uses Vulkan for mixed-vendor hints") {
    const si::GgufModelInfo model = bonsai_2_9b_mtp_model();

    // CPU-only plan with mmproj must disable mmproj_offload so GPU VRAM stays 0.
    si::PlacementPlanRequest cpu_req;
    cpu_req.model_path = "/models/vl.gguf";
    cpu_req.mmproj_path = "/models/mmproj.gguf";
    cpu_req.mmproj_bytes = 512 * kMiB;
    cpu_req.device_hint = "cpu";
    const si::PlacementPlan cpu_plan = si::plan_model_placement(model, {make_rtx_5070_ti()}, host_ram(32, 24), cpu_req);
    verify_plan_contracts(cpu_plan);
    CHECK(cpu_plan.profile.n_gpu_layers == std::optional<std::int32_t>(0));
    CHECK(cpu_plan.profile.mmproj_offload == std::optional<bool>(false));
    CHECK(cpu_plan.estimate.total_bytes == 0u);

    // Mixed-vendor explicit multi-GPU hint switches to Vulkan so device names stay distinct.
    si::PlacementPlanRequest mixed_req;
    mixed_req.model_path = "/models/model.gguf";
    mixed_req.device_hint = "gpu:1,gpu:0";
    const std::vector<si::GpuDeviceInfo> mixed_gpus{
        make_ryzen_780m_apu("gpu:0", "vulkan"),
        make_rtx_5070_ti("gpu:1", 15360),
    };
    const si::PlacementPlan mixed_plan = si::plan_model_placement(model, mixed_gpus, host_ram(32, 24), mixed_req);
    verify_plan_contracts(mixed_plan);
    CHECK(mixed_plan.recommended_backend_hint == "vulkan");
    CHECK(mixed_plan.profile.device == std::optional<std::string>("Vulkan1,Vulkan0"));
    CHECK(any_contains(mixed_plan.warnings, "multiple vendors"));

    // Control characters in unmatched device_hint are sanitized in warnings.
    si::PlacementPlanRequest ctrl_req;
    ctrl_req.model_path = "/models/model.gguf";
    ctrl_req.device_hint = "bad\nhint\r\t";
    const si::PlacementPlan ctrl_plan = si::plan_model_placement(model, {make_rtx_5070_ti()}, host_ram(32, 24), ctrl_req);
    verify_plan_contracts(ctrl_plan);
    REQUIRE(any_contains(ctrl_plan.warnings, "device_hint"));
    for (const auto& w : ctrl_plan.warnings) {
        CHECK(w.find('\n') == std::string::npos);
        CHECK(w.find('\r') == std::string::npos);
    }
}

TEST_CASE("placement planner: cross-platform Linux/WSL Blackwell, hybrid AMD iGPU+RX XT ROCm0, and no-KFD APU") {
    // 1. Linux / WSL2 NVIDIA RTX 5070 Ti (Blackwell sm_120, NVML source) + Bonsai-2-9B-MTP.
    {
        si::GpuDeviceInfo linux_blackwell = make_rtx_5070_ti("gpu:0", 14848);
        linux_blackwell.source = "nvml";
        linux_blackwell.shared_total = si::MemoryReading::unknown();

        si::PlacementPlanRequest req;
        req.profile_name = "linux-blackwell-bonsai";
        req.model_path = "/models/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf";
        req.target_ctx_size = 32768;

        const si::PlacementPlan plan =
            si::plan_model_placement(bonsai_2_9b_mtp_model(), {linux_blackwell}, host_ram(32, 22), req);
        verify_plan_contracts(plan);
        CHECK(plan.fits_in_vram);
        CHECK(plan.requires_prism_runtime);
        CHECK(plan.selected_vendor == si::GpuVendor::nvidia);
        CHECK(plan.recommended_backend_hint == "cuda");
        CHECK(plan.profile.device == std::optional<std::string>("CUDA0"));
        CHECK(plan.profile.flash_attn == std::optional<std::string>("on"));
        CHECK(plan.profile.load_mode == std::optional<std::string>("none"));
        REQUIRE(plan.profile.speculative.has_value());
        CHECK(plan.profile.speculative->types == std::vector<std::string>{"draft-mtp"});
    }

    // 2. Hybrid AMD workstation: gpu:0 = Ryzen 9 9900X3D iGPU (UMA, backend_hint="vulkan") +
    //    gpu:1 = Radeon RX 9070 XT discrete (backend_hint="rocm").
    //    Default AND explicit "ROCm0" must select gpu:1 and emit --device ROCm0 (never ROCm1!).
    {
        si::GpuDeviceInfo ryzen_igpu = make_ryzen_780m_apu("gpu:0", "vulkan");
        ryzen_igpu.name = "AMD Radeon(TM) Graphics";
        ryzen_igpu.device_id = 0x13C0u;
        const si::GpuDeviceInfo rx9070xt = make_amd_discrete("gpu:1", "AMD Radeon RX 9070 XT", 16, 15360, "rocm");
        const std::vector<si::GpuDeviceInfo> hybrid_amd{ryzen_igpu, rx9070xt};

        si::PlacementPlanRequest req;
        req.profile_name = "hybrid-amd-bonsai";
        req.model_path = "D:/models/Ternary-Bonsai-2-27B-PQ2_0.gguf";
        req.target_ctx_size = 32768;

        const si::PlacementPlan auto_plan =
            si::plan_model_placement(bonsai_2_27b_model(), hybrid_amd, host_ram(32, 24), req);
        verify_plan_contracts(auto_plan);
        CHECK(auto_plan.fits_in_vram);
        CHECK_FALSE(auto_plan.uma_mode);
        CHECK(auto_plan.selected_vendor == si::GpuVendor::amd);
        CHECK(auto_plan.recommended_backend_hint == "rocm");
        CHECK(auto_plan.profile.device == std::optional<std::string>("ROCm0"));
        CHECK(find_env(auto_plan.recommended_env, "GPU_MAX_ALLOC_PERCENT") == std::optional<std::string>("100"));

        si::PlacementPlanRequest rocm0_req = req;
        rocm0_req.device_hint = "ROCm0";
        const si::PlacementPlan rocm0_plan =
            si::plan_model_placement(bonsai_2_27b_model(), hybrid_amd, host_ram(32, 24), rocm0_req);
        verify_plan_contracts(rocm0_plan);
        CHECK_FALSE(rocm0_plan.uma_mode);
        CHECK(rocm0_plan.profile.device == std::optional<std::string>("ROCm0"));
        CHECK(rocm0_plan.budget_mib == 15360u - 1536u);

        si::PlacementPlanRequest uma_req = req;
        uma_req.device_hint = "apu";
        const si::PlacementPlan uma_plan =
            si::plan_model_placement(bonsai_2_27b_model(), hybrid_amd, host_ram(32, 24), uma_req);
        verify_plan_contracts(uma_plan);
        CHECK(uma_plan.uma_mode);
        CHECK(uma_plan.recommended_backend_hint == "vulkan");
        CHECK(uma_plan.profile.device == std::optional<std::string>("Vulkan0"));
    }

    // 3. Linux AMD Ryzen 780M APU without /dev/kfd (amdgpu-sysfs with backend_hint="vulkan")
    //    uses Vulkan0 UMA rather than emitting broken ROCm0.
    {
        si::GpuDeviceInfo linux_apu_no_kfd = make_ryzen_780m_apu("gpu:0", "vulkan");
        linux_apu_no_kfd.source = "amdgpu-sysfs";

        si::PlacementPlanRequest req;
        req.profile_name = "linux-apu-vulkan";
        req.model_path = "/models/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf";
        const si::PlacementPlan plan =
            si::plan_model_placement(bonsai_2_9b_mtp_model(), {linux_apu_no_kfd}, host_ram(32, 24), req);
        verify_plan_contracts(plan);
        CHECK(plan.uma_mode);
        CHECK(plan.fits_in_vram);
        CHECK(plan.recommended_backend_hint == "vulkan");
        CHECK(plan.profile.device == std::optional<std::string>("Vulkan0"));
        CHECK_FALSE(find_env(plan.recommended_env, "GPU_MAX_ALLOC_PERCENT").has_value());
    }

    // 4. Dual discrete AMD cards where card1 is hidden from ROCm (gpu:0="rocm", gpu:1="vulkan"):
    //    ROCm auto-expansion must NOT split across the Vulkan-only peer.
    {
        const std::vector<si::GpuDeviceInfo> masked_amd{
            make_amd_discrete("gpu:0", "AMD Radeon RX 7900 XT", 20, 18432, "rocm"),
            make_amd_discrete("gpu:1", "AMD Radeon RX 7900 XT", 20, 18432, "vulkan"),
        };
        const si::GgufModelInfo big = dense_model(44, 512, 4);  // 22 GiB > single 18 GiB budget
        si::PlacementPlanRequest req;
        req.profile_name = "masked-rocm";
        req.model_path = "/models/model-22g.gguf";
        const si::PlacementPlan plan = si::plan_model_placement(big, masked_amd, host_ram(64, 48), req);
        verify_plan_contracts(plan);
        CHECK(plan.recommended_backend_hint == "rocm");
        CHECK(plan.profile.device == std::optional<std::string>("ROCm0"));
        CHECK_FALSE(plan.profile.split_mode.has_value());
    }
}
