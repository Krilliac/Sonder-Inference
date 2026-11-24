// Unit tests for the multi-model co-residency VRAM planner
// (sonder/inference/coresidency_planner.hpp).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backends/llamaserver/env_policy.hpp"
#include "sonder/inference/coresidency_planner.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/placement_planner.hpp"

namespace si = sonder::inference;
namespace json = sonder::inference::json;

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

std::uint64_t bytes_to_mib_ceil(std::uint64_t bytes) noexcept {
    return (bytes >> 20) + ((bytes & (kMiB - 1)) != 0 ? 1ull : 0ull);
}

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

bool any_contains(const std::vector<std::string>& list, const std::string& needle) {
    return std::any_of(list.begin(), list.end(), [&](const std::string& s) { return contains(s, needle); });
}

void verify_plan_contracts(const si::PlacementPlan& plan) {
    const si::Status valid = si::validate_launch_profile(plan.profile);
    CHECK_MESSAGE(valid.ok(), valid.message());
    const auto argv = si::llamaserver_arguments(plan.profile);
    CHECK_MESSAGE(argv.ok(), argv.status().message());

    for (const auto& [k, v] : plan.recommended_env) {
        (void)v;
        CHECK_MESSAGE(si::llamaserver::is_allowed_profile_env_override(k), "disallowed env var: " << k);
    }
    const si::Status env_st = si::llamaserver::check_child_environment_overrides(plan.recommended_env);
    CHECK_MESSAGE(env_st.ok(), env_st.message());
}

void verify_coresidency_contracts(const si::CoResidencyPlan& co_plan) {
    json::Array profile_list;
    profile_list.reserve(co_plan.models.size());
    for (const auto& m : co_plan.models) {
        verify_plan_contracts(m);
        const json::Object doc = si::placement_plan_to_json(m);
        const json::Value* prof_val = doc.find("profile");
        REQUIRE(prof_val != nullptr);
        profile_list.push_back(*prof_val);
    }
    if (!profile_list.empty()) {
        json::Object wrapper;
        wrapper.set("profiles", std::move(profile_list));
        const auto reparsed = si::parse_launch_profiles(json::Value(std::move(wrapper)));
        CHECK_MESSAGE(reparsed.ok(), reparsed.status().message());
    }
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

// Ternary Bonsai-2-27B: PQ2_0 + prism_hadamard + hybrid qwen35 DeltaNet (~8.4 GiB total VRAM at 32K ctx).
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
    for (std::uint32_t i = 3; i < 64; i += 4) m.kv_heads[i] = 4;  // 16 full-attention, 48 recurrent
    m.block_bytes.assign(64, 112 * kMiB);
    m.output_bytes = 256 * kMiB;
    m.embedding_bytes = 256 * kMiB;
    m.recurrent_state_values = 512ull * 128ull;
    m.metadata = {{"general.architecture", "qwen35"}, {"qwen35.context_length", "131072"}};
    m.prism_hadamard = true;
    m.prism_hadamard_transform = "fwht";
    m.prism_hadamard_block_size = 4096;
    m.has_ternary_tensors = true;
    m.has_prism_private_tensors = true;
    return m;
}

// Ternary Bonsai-2-9B-MTP: PQ2_0 + nextn_layers=1 + hybrid qwen35 DeltaNet (~3.5 GiB total VRAM at 32K ctx).
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
    m.kv_heads[32] = 4;
    m.block_bytes.assign(33, 76 * kMiB);
    m.output_bytes = 128 * kMiB;
    m.embedding_bytes = 128 * kMiB;
    m.recurrent_state_values = 1024ull * 128ull;
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

si::CoResidencyEntry make_entry(si::GgufModelInfo model, std::string name, std::string path,
                                std::uint32_t ctx = 32768) {
    si::CoResidencyEntry e;
    e.model = std::move(model);
    e.request.profile_name = std::move(name);
    e.request.model_path = std::move(path);
    e.request.target_ctx_size = ctx;
    return e;
}

}  // namespace

TEST_CASE("coresidency planner: Ternary Bonsai-2-27B and Bonsai-2-9B-MTP share one RTX 5070 Ti with single reserve") {
    const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 15360)};
    const std::vector<si::CoResidencyEntry> entries{
        make_entry(bonsai_2_27b_model(), "bonsai-2-27b", "D:/models/Bonsai-2-27B-PQ2_0.gguf", 32768),
        make_entry(bonsai_2_9b_mtp_model(), "bonsai-2-9b-mtp", "D:/models/Bonsai-2-9B-MTP-PQ2_0.gguf", 32768),
    };

    const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(32, 24));
    verify_coresidency_contracts(plan);

    CHECK(plan.all_fit_coresident);
    REQUIRE(plan.models.size() == 2u);
    CHECK(plan.models[0].profile.name == "bonsai-2-27b");
    CHECK(plan.models[1].profile.name == "bonsai-2-9b-mtp");

    for (const auto& m : plan.models) {
        CHECK(m.fits_in_vram);
        CHECK(m.fits_in_host_ram);
        CHECK_FALSE(m.uma_mode);
        CHECK(m.requires_prism_runtime);
        CHECK(m.selected_vendor == si::GpuVendor::nvidia);
        CHECK(m.profile.n_gpu_layers == std::optional<std::int32_t>(-1));
        CHECK(m.profile.ctx_size == std::optional<std::uint32_t>(32768u));
        CHECK(m.profile.device == std::optional<std::string>("CUDA0"));
    }

    const std::uint64_t expected_vram = plan.models[0].estimate.total_mib() + plan.models[1].estimate.total_mib();
    const std::uint64_t expected_host =
        plan.models[0].estimate.host_resident_mib() + plan.models[1].estimate.host_resident_mib();
    CHECK(plan.aggregate_vram_mib == expected_vram);
    CHECK(plan.aggregate_host_ram_mib == expected_host);

    // Single 1536 MiB OS reserve deducted once across both co-resident models.
    REQUIRE(plan.remaining_gpu_vram_mib.size() == 1u);
    CHECK(plan.remaining_gpu_vram_mib[0].first == "gpu:0");
    CHECK(plan.remaining_gpu_vram_mib[0].second == 15360u - 1536u - expected_vram);
    CHECK(plan.remaining_gpu_vram_mib[0].second > 1000u);
    CHECK(plan.remaining_host_ram_mib == 24u * 1024u - expected_host);
    CHECK(any_contains(plan.notes, "reused existing 1536 MiB OS reserve"));
}

TEST_CASE("coresidency planner: three models exceeding 16 GB RTX spill to Ryzen iGPU UMA or degrade cleanly") {
    // 1. RTX 5070 Ti + Ryzen 780M iGPU UMA: 3rd model spills cleanly to Ryzen iGPU.
    {
        const std::vector<si::GpuDeviceInfo> gpus{
            make_rtx_5070_ti("gpu:0", 15360),
            make_ryzen_780m_apu("gpu:1", "vulkan"),
        };
        const std::vector<si::CoResidencyEntry> entries{
            make_entry(bonsai_2_27b_model(), "bonsai-2-27b", "D:/models/Bonsai-2-27B.gguf", 32768),
            make_entry(bonsai_2_9b_mtp_model(), "bonsai-2-9b-mtp", "D:/models/Bonsai-2-9B-MTP.gguf", 32768),
            make_entry(dense_model(32, 200, 4), "helper-12b", "D:/models/helper-12b.gguf", 32768),
        };

        const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(48, 36));
        verify_coresidency_contracts(plan);

        CHECK(plan.all_fit_coresident);
        REQUIRE(plan.models.size() == 3u);
        CHECK(plan.models[0].profile.device == std::optional<std::string>("CUDA0"));
        CHECK(plan.models[1].profile.device == std::optional<std::string>("CUDA0"));
        CHECK(plan.models[2].uma_mode);
        CHECK(plan.models[2].fits_in_vram);
        CHECK(plan.models[2].fits_in_host_ram);
        CHECK(plan.models[2].profile.n_gpu_layers == std::optional<std::int32_t>(-1));
        CHECK(plan.models[2].profile.device == std::optional<std::string>("Vulkan1"));
        CHECK(any_contains(plan.notes, "spilled from saturated discrete GPU to integrated UMA"));
        REQUIRE(plan.remaining_gpu_vram_mib.size() == 2u);
        CHECK(plan.remaining_gpu_vram_mib[0].first == "gpu:0");
        CHECK(plan.remaining_gpu_vram_mib[1].first == "gpu:1");
    }

    // 2. Single RTX 5070 Ti + tight host RAM: 3rd oversized model overflows host RAM -> all_fit_coresident == false.
    {
        const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 15360)};
        const std::vector<si::CoResidencyEntry> entries{
            make_entry(bonsai_2_27b_model(), "bonsai-2-27b", "D:/models/Bonsai-2-27B.gguf", 32768),
            make_entry(bonsai_2_9b_mtp_model(), "bonsai-2-9b-mtp", "D:/models/Bonsai-2-9B-MTP.gguf", 32768),
            make_entry(dense_model(40, 320, 4), "oversized-third", "D:/models/oversized.gguf", 32768),
        };

        const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(16, 10));
        verify_coresidency_contracts(plan);

        CHECK_FALSE(plan.all_fit_coresident);
        REQUIRE(plan.models.size() == 3u);
        CHECK_FALSE(plan.models[2].fits_in_host_ram);
        CHECK(any_contains(plan.warnings, "host RAM"));
    }

    // 3. Single RTX 5070 Ti filled to < 1 layer remaining: pinned CUDA0 fails fit; unpinned spills cleanly to CPU.
    {
        const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 15360)};
        si::CoResidencyEntry first = make_entry(dense_model(24, 512, 4), "heavy-primary", "m0.gguf", 16384);
        si::CoResidencyEntry pinned = make_entry(dense_model(16, 400, 4), "pinned-second", "m1.gguf", 16384);
        pinned.request.device_hint = "CUDA0";

        const si::CoResidencyPlan pinned_plan =
            si::plan_coresident_models({first, pinned}, gpus, host_ram(64, 48));
        verify_coresidency_contracts(pinned_plan);
        CHECK_FALSE(pinned_plan.all_fit_coresident);
        REQUIRE(pinned_plan.models.size() == 2u);
        CHECK_FALSE(pinned_plan.models[1].fits_in_vram);
        CHECK(any_contains(pinned_plan.warnings, "pinned device_hint"));

        si::CoResidencyEntry unpinned = pinned;
        unpinned.request.device_hint.clear();
        unpinned.request.profile_name = "cpu-spilled-second";
        const si::CoResidencyPlan cpu_spill_plan =
            si::plan_coresident_models({first, unpinned}, gpus, host_ram(64, 48));
        verify_coresidency_contracts(cpu_spill_plan);
        CHECK(cpu_spill_plan.all_fit_coresident);
        REQUIRE(cpu_spill_plan.models.size() == 2u);
        CHECK(cpu_spill_plan.models[1].recommended_backend_hint == "cpu");
        CHECK(cpu_spill_plan.models[1].profile.n_gpu_layers == std::optional<std::int32_t>(0));
        CHECK(cpu_spill_plan.models[1].fits_in_host_ram);
        CHECK(any_contains(cpu_spill_plan.warnings, "CPU-only"));
    }
}

TEST_CASE("coresidency planner: dual-GPU rig places Model A on GPU 0 and Model B on GPU 1 when GPU 0 fills") {
    const std::vector<si::GpuDeviceInfo> gpus{
        make_rtx_5070_ti("gpu:0", 15360),
        make_rtx_5070_ti("gpu:1", 15360),
    };
    const std::vector<si::CoResidencyEntry> entries{
        make_entry(dense_model(32, 280, 4), "model-a", "D:/models/model-a.gguf", 32768),
        make_entry(dense_model(32, 280, 4), "model-b", "D:/models/model-b.gguf", 32768),
        make_entry(dense_model(8, 200, 4), "model-c-small", "D:/models/model-c.gguf", 16384),
    };

    const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(64, 48));
    verify_coresidency_contracts(plan);

    CHECK(plan.all_fit_coresident);
    REQUIRE(plan.models.size() == 3u);
    CHECK(plan.models[0].profile.device == std::optional<std::string>("CUDA0"));
    CHECK(plan.models[1].profile.device == std::optional<std::string>("CUDA1"));
    CHECK(plan.models[2].profile.device == std::optional<std::string>("CUDA0"));
    for (const auto& m : plan.models) {
        CHECK(m.fits_in_vram);
        CHECK(m.fits_in_host_ram);
        CHECK(m.profile.n_gpu_layers == std::optional<std::int32_t>(-1));
    }

    const std::uint64_t gpu0_used = plan.models[0].estimate.total_mib() + plan.models[2].estimate.total_mib();
    const std::uint64_t gpu1_used = plan.models[1].estimate.total_mib();
    CHECK(plan.aggregate_vram_mib == gpu0_used + gpu1_used);
    REQUIRE(plan.remaining_gpu_vram_mib.size() == 2u);
    CHECK(plan.remaining_gpu_vram_mib[0].first == "gpu:0");
    CHECK(plan.remaining_gpu_vram_mib[0].second == 15360u - 1536u - gpu0_used);
    CHECK(plan.remaining_gpu_vram_mib[1].first == "gpu:1");
    CHECK(plan.remaining_gpu_vram_mib[1].second == 15360u - 1536u - gpu1_used);
}

TEST_CASE("coresidency planner: multi-GPU tensor_split model deducts proportionally across GPUs") {
    const std::vector<si::GpuDeviceInfo> gpus{
        make_rtx_5070_ti("gpu:0", 15872),
        make_rtx_5070_ti("gpu:1", 15872),
    };
    const std::vector<si::CoResidencyEntry> entries{
        make_entry(dense_model(44, 512, 4), "split-22g", "D:/models/split-22g.gguf", 32768),
        make_entry(dense_model(6, 160, 4), "tiny-helper", "D:/models/tiny-helper.gguf", 8192),
    };

    const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(64, 48));
    verify_coresidency_contracts(plan);

    CHECK(plan.all_fit_coresident);
    REQUIRE(plan.models.size() == 2u);
    CHECK(plan.models[0].profile.device == std::optional<std::string>("CUDA0,CUDA1"));
    CHECK(plan.models[0].profile.split_mode == std::optional<std::string>("layer"));
    CHECK(plan.models[1].fits_in_vram);
    CHECK(plan.models[1].profile.n_gpu_layers == std::optional<std::int32_t>(-1));
    CHECK(plan.models[1].profile.device == std::optional<std::string>("CUDA0"));

    REQUIRE(plan.remaining_gpu_vram_mib.size() == 2u);
    // Both GPUs had the 22 GB model's shard deducted, and GPU 0 also deducted tiny-helper.
    CHECK(plan.remaining_gpu_vram_mib[1].second < 15872u - 1536u - 10000u);
    CHECK(plan.remaining_gpu_vram_mib[0].second < plan.remaining_gpu_vram_mib[1].second);
    CHECK(plan.remaining_gpu_vram_mib[1].second - plan.remaining_gpu_vram_mib[0].second ==
          plan.models[1].estimate.total_mib());
}

TEST_CASE("coresidency planner: UMA APU co-residency deducts shared pool and host RAM without double-counting") {
    const std::vector<si::GpuDeviceInfo> gpus{make_ryzen_780m_apu("gpu:0", "vulkan")};
    const std::vector<si::CoResidencyEntry> entries{
        make_entry(dense_model(24, 220, 4), "uma-model-a", "D:/models/uma-a.gguf", 16384),
        make_entry(dense_model(16, 200, 4), "uma-model-b", "D:/models/uma-b.gguf", 16384),
    };

    // 1. Uncapped by host RAM (36 GiB free > 24.5 GiB UMA pool).
    {
        const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(48, 36));
        verify_coresidency_contracts(plan);

        CHECK(plan.all_fit_coresident);
        REQUIRE(plan.models.size() == 2u);
        for (const auto& m : plan.models) {
            CHECK(m.uma_mode);
            CHECK(m.fits_in_vram);
            CHECK(m.fits_in_host_ram);
            CHECK(m.profile.n_gpu_layers == std::optional<std::int32_t>(-1));
        }

        const std::uint64_t gpu_used = plan.models[0].estimate.total_mib() + plan.models[1].estimate.total_mib();
        const std::uint64_t unified_m0 =
            bytes_to_mib_ceil(plan.models[0].estimate.total_bytes + plan.models[0].estimate.host_resident_bytes);
        const std::uint64_t unified_m1 =
            bytes_to_mib_ceil(plan.models[1].estimate.total_bytes + plan.models[1].estimate.host_resident_bytes);
        const std::uint64_t unified_total = unified_m0 + unified_m1;

        CHECK(plan.aggregate_vram_mib == gpu_used);
        CHECK(plan.aggregate_host_ram_mib == unified_total);
        // Single 1536 MiB reserve on 25088 MiB UMA pool.
        REQUIRE(plan.remaining_gpu_vram_mib.size() == 1u);
        CHECK(plan.remaining_gpu_vram_mib[0].second == (512u + 24u * 1024u) - 1536u - gpu_used);
        CHECK(plan.remaining_host_ram_mib == 36u * 1024u - unified_total);
    }

    // 2. Host-capped UMA (14 GiB available physical RAM < 24.5 GiB UMA pool):
    // remaining UMA pool stays strictly bounded by (remaining host RAM - single reserve).
    {
        const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host_ram(24, 14));
        verify_coresidency_contracts(plan);

        CHECK(plan.all_fit_coresident);
        REQUIRE(plan.models.size() == 2u);
        CHECK(plan.models[0].uma_mode);
        CHECK(plan.models[1].uma_mode);
        CHECK(plan.remaining_host_ram_mib == 14u * 1024u - plan.aggregate_host_ram_mib);
        REQUIRE(plan.remaining_gpu_vram_mib.size() == 1u);
        CHECK(plan.remaining_gpu_vram_mib[0].second + 1536u == plan.remaining_host_ram_mib);
    }
}

TEST_CASE("coresidency planner: prefer_free_vram false uses dedicated_total and preserves input order") {
    // Only 6 GiB free reported, but 16 GiB dedicated total.
    const std::vector<si::GpuDeviceInfo> gpus{make_rtx_5070_ti("gpu:0", 6144)};
    si::CoResidencyEntry small_first =
        make_entry(bonsai_2_9b_mtp_model(), "small-first", "D:/models/Bonsai-2-9B-MTP.gguf", 32768);
    small_first.request.prefer_free_vram = false;
    si::CoResidencyEntry large_second =
        make_entry(bonsai_2_27b_model(), "large-second", "D:/models/Bonsai-2-27B.gguf", 32768);
    large_second.request.prefer_free_vram = false;

    const si::CoResidencyPlan plan =
        si::plan_coresident_models({small_first, large_second}, gpus, host_ram(32, 24));
    verify_coresidency_contracts(plan);

    CHECK(plan.all_fit_coresident);
    REQUIRE(plan.models.size() == 2u);
    CHECK(plan.models[0].profile.name == "small-first");
    CHECK(plan.models[1].profile.name == "large-second");
    REQUIRE(plan.remaining_gpu_vram_mib.size() == 1u);
    CHECK(plan.remaining_gpu_vram_mib[0].second == 16u * 1024u - 1536u - plan.aggregate_vram_mib);
}

TEST_CASE("coresidency planner: empty entries list and coresidency_plan_to_json full schema round-trip") {
    const std::vector<si::GpuDeviceInfo> gpus{
        make_rtx_5070_ti("gpu:0", 15360),
        make_ryzen_780m_apu("gpu:1", "vulkan"),
    };
    const si::HostMemory host = host_ram(32, 24);

    // 1. Empty entries list.
    {
        const si::CoResidencyPlan empty_plan = si::plan_coresident_models({}, gpus, host);
        CHECK(empty_plan.models.empty());
        CHECK(empty_plan.all_fit_coresident);
        CHECK(empty_plan.aggregate_vram_mib == 0u);
        CHECK(empty_plan.aggregate_host_ram_mib == 0u);
        CHECK(empty_plan.remaining_host_ram_mib == 24u * 1024u);
        REQUIRE(empty_plan.remaining_gpu_vram_mib.size() == 2u);
        CHECK(empty_plan.remaining_gpu_vram_mib[0].first == "gpu:0");
        CHECK(empty_plan.remaining_gpu_vram_mib[0].second == 15360u);
        CHECK(empty_plan.remaining_gpu_vram_mib[1].first == "gpu:1");
        CHECK(empty_plan.remaining_gpu_vram_mib[1].second == 24u * 1024u);
        CHECK_FALSE(empty_plan.notes.empty());

        const std::string dumped = json::Value(si::coresidency_plan_to_json(empty_plan)).dump();
        const auto reparsed = json::parse(dumped);
        REQUIRE_MESSAGE(reparsed.ok(), reparsed.status().message());
        const json::Object& root = reparsed->as_object();
        REQUIRE(root.find("schema") != nullptr);
        CHECK(root.find("schema")->as_string() == "sonder.inference.coresidency_plan/1");
        REQUIRE(root.find("all_fit_coresident") != nullptr);
        CHECK(root.find("all_fit_coresident")->as_bool());
        REQUIRE(root.find("model_count") != nullptr);
        CHECK(root.find("model_count")->as_uint() == 0u);
        REQUIRE(root.find("models") != nullptr);
        CHECK(root.find("models")->as_array().empty());
    }

    // 2. Populated 2-model co-residency plan full JSON schema round-trip.
    {
        const std::vector<si::CoResidencyEntry> entries{
            make_entry(bonsai_2_27b_model(), "bonsai-2-27b", "D:/models/Bonsai-2-27B.gguf", 32768),
            make_entry(bonsai_2_9b_mtp_model(), "bonsai-2-9b-mtp", "D:/models/Bonsai-2-9B-MTP.gguf", 32768),
        };
        const si::CoResidencyPlan plan = si::plan_coresident_models(entries, gpus, host);
        const json::Object obj = si::coresidency_plan_to_json(plan);
        const std::string dumped = json::Value(obj).dump();
        const auto reparsed = json::parse(dumped);
        REQUIRE_MESSAGE(reparsed.ok(), reparsed.status().message());
        const json::Object& root = reparsed->as_object();

        REQUIRE(root.find("schema") != nullptr);
        CHECK(root.find("schema")->as_string() == "sonder.inference.coresidency_plan/1");
        REQUIRE(root.find("all_fit_coresident") != nullptr);
        CHECK(root.find("all_fit_coresident")->as_bool() == plan.all_fit_coresident);
        REQUIRE(root.find("model_count") != nullptr);
        CHECK(root.find("model_count")->as_uint() == 2u);
        REQUIRE(root.find("aggregate_vram_mib") != nullptr);
        CHECK(root.find("aggregate_vram_mib")->as_uint() == plan.aggregate_vram_mib);
        REQUIRE(root.find("aggregate_host_ram_mib") != nullptr);
        CHECK(root.find("aggregate_host_ram_mib")->as_uint() == plan.aggregate_host_ram_mib);
        REQUIRE(root.find("remaining_host_ram_mib") != nullptr);
        CHECK(root.find("remaining_host_ram_mib")->as_uint() == plan.remaining_host_ram_mib);

        const json::Value* rem_gpu = root.find("remaining_gpu_vram_mib");
        REQUIRE(rem_gpu != nullptr);
        REQUIRE(rem_gpu->is_object());
        REQUIRE(rem_gpu->find("gpu:0") != nullptr);
        CHECK(rem_gpu->find("gpu:0")->as_uint() == plan.remaining_gpu_vram_mib[0].second);
        REQUIRE(rem_gpu->find("gpu:1") != nullptr);
        CHECK(rem_gpu->find("gpu:1")->as_uint() == plan.remaining_gpu_vram_mib[1].second);

        const json::Value* models = root.find("models");
        REQUIRE(models != nullptr);
        REQUIRE(models->is_array());
        REQUIRE(models->as_array().size() == 2u);

        json::Array profiles_out;
        for (const auto& item : models->as_array()) {
            REQUIRE(item.is_object());
            const json::Value* prof = item.find("profile");
            REQUIRE(prof != nullptr);
            profiles_out.push_back(*prof);
        }
        json::Object wrapper;
        wrapper.set("profiles", std::move(profiles_out));
        const auto parsed_profiles = si::parse_launch_profiles(json::Value(std::move(wrapper)));
        REQUIRE_MESSAGE(parsed_profiles.ok(), parsed_profiles.status().message());
        REQUIRE(parsed_profiles->size() == 2u);
        CHECK(parsed_profiles->at(0).name == "bonsai-2-27b");
        CHECK(parsed_profiles->at(1).name == "bonsai-2-9b-mtp");

        REQUIRE(root.find("notes") != nullptr);
        CHECK(root.find("notes")->is_array());
        CHECK_FALSE(root.find("notes")->as_array().empty());
        REQUIRE(root.find("warnings") != nullptr);
        CHECK(root.find("warnings")->is_array());
    }
}
