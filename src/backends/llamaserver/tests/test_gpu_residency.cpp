#include <doctest/doctest.h>

#include "../spill_guard.hpp"
#include "../gpu_residency.hpp"

#include <limits>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;

namespace {
constexpr std::uint64_t MiB = 1024ull * 1024ull;
constexpr std::uint64_t GiB = 1024ull * MiB;
}

TEST_CASE("GPU layer parsing is explicit, last occurrence wins, and fail closed") {
    CHECK(expects_gpu_offload({"--n-gpu-layers", "4"}));
    CHECK(expects_gpu_offload({"--gpu-layers=all"}));
    CHECK(expects_gpu_offload({"-ngl=999"}));
    CHECK(expects_gpu_offload({"-ngl", "-1"}));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers", "0"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers=bad"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers="}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers", "4", "-ngl", "0"}));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers=", "4"}));
    CHECK_FALSE(expects_gpu_offload({}));
    CHECK(expects_gpu_offload({}, true));
}

TEST_CASE("residency guard validates bounded finite configuration") {
    LlamaServerResidencyGuardOptions o;
    CHECK(validate_residency_guard(o).ok());
    o.consecutive_samples = 0;
    CHECK(validate_residency_guard(o).code() == ErrorCode::invalid_argument);
    o = {};
    o.eviction_fraction = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = std::numeric_limits<double>::infinity();
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = 1.01;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = 0;
    o.eviction_bytes = 0;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_bytes = 1;
    o.max_eviction_restarts = 33;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.max_eviction_restarts = 1;
    o.consecutive_samples = 1001;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.consecutive_samples = 3;
    o.min_dedicated_bytes = 0;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.min_dedicated_bytes = 512 * MiB;
    o.eviction_bytes = 1024ull * GiB + 1;
    CHECK_FALSE(validate_residency_guard(o).ok());
}

TEST_CASE("missing GPU offload requires consecutive low samples and latches") {
    LlamaServerResidencyGuardOptions o;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    g.observe(0);
    CHECK(g.gpu_offload_missing());
    CHECK(g.observed_dedicated_bytes() == 0);
    g.observe(2 * GiB);
    CHECK(g.gpu_offload_missing());
    g.reset();
    CHECK_FALSE(g.gpu_offload_missing());
    CHECK(g.peak_dedicated_bytes() == 0);
}

TEST_CASE("healthy samples reset low and eviction streaks") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 2;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    g.observe(2 * GiB);
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    g.observe(2 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
    g.unavailable();
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
}

TEST_CASE("eviction uses strict fraction or byte thresholds and high water") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 2;
    GpuResidencyGuard g(o, false);
    g.observe(8 * GiB);
    g.observe(6 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(5 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(5 * GiB);
    CHECK(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 8 * GiB);
    CHECK(g.observed_dedicated_bytes() == 5 * GiB);
}

TEST_CASE("eviction can independently disable fraction or byte dimensions") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    o.eviction_fraction = 0;
    o.eviction_bytes = 1 * GiB;
    GpuResidencyGuard bytes(o, false);
    bytes.observe(8 * GiB);
    bytes.observe(7 * GiB);
    CHECK_FALSE(bytes.vram_evicted());
    bytes.observe(6 * GiB);
    CHECK(bytes.vram_evicted());

    o.eviction_fraction = 0.25;
    o.eviction_bytes = 0;
    GpuResidencyGuard fraction(o, false);
    fraction.observe(8 * GiB);
    fraction.observe(6 * GiB);
    CHECK_FALSE(fraction.vram_evicted());
    fraction.observe(5 * GiB);
    CHECK(fraction.vram_evicted());
}

TEST_CASE("tiny peaks do not create eviction events under the floor") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, false);
    g.observe(400 * MiB);
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 400 * MiB);
}

TEST_CASE("missing and eviction findings may coexist and unavailable preserves latches") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, true);
    g.observe(2 * GiB);
    g.observe(0);
    CHECK(g.gpu_offload_missing());
    CHECK(g.vram_evicted());
    g.unavailable();
    CHECK(g.gpu_offload_missing());
    CHECK(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 2 * GiB);
}

TEST_CASE("a later eviction records its own sample after an earlier missing offload incident") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    REQUIRE(g.gpu_offload_missing());
    g.observe(8 * GiB);
    g.observe(3 * GiB);
    REQUIRE(g.vram_evicted());
    CHECK(g.observed_dedicated_bytes() == 3 * GiB);
}

TEST_CASE("explicit CPU intent overrides expect_gpu at detector construction") {
    LlamaServerResidencyGuardOptions o;
    o.expect_gpu = true;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, expects_gpu_offload({"-ngl", "0"}, o.expect_gpu));
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    CHECK_FALSE(g.vram_evicted());
}

TEST_CASE("parse_linux_drm_fdinfo parses units, deduplicates client ids, and ignores malformed input") {
    std::uint64_t vram = 123;
    std::uint64_t gtt = 456;

    // Single amdgpu fdinfo with tab-separated KiB values.
    const std::string single =
        "pos:\t0\n"
        "flags:\t02100002\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t42\n"
        "drm-pdev:\t0000:03:00.0\n"
        "drm-memory-vram:\t15421440 KiB\n"
        "drm-memory-gtt:\t196608 KiB\n";
    REQUIRE(parse_linux_drm_fdinfo(single, vram, gtt));
    CHECK(vram == 15421440ull * 1024ull);
    CHECK(gtt == 196608ull * 1024ull);

    // MiB, GiB, and B units with CRLF line endings.
    const std::string mixed_units =
        "drm-driver: amdgpu\r\n"
        "drm-client-id: 7\r\n"
        "drm-memory-vram: 12 GiB\r\n"
        "drm-memory-gtt: 256 MiB\r\n"
        "\r\n"
        "drm-driver: amdgpu\r\n"
        "drm-client-id: 8\r\n"
        "drm-memory-vram: 512 B\r\n"
        "drm-memory-gtt: 0 B\r\n";
    REQUIRE(parse_linux_drm_fdinfo(mixed_units, vram, gtt));
    CHECK(vram == 12ull * GiB + 512ull);
    CHECK(gtt == 256ull * MiB);

    // Duplicate drm-client-id entries (both blank-line separated and adjacent) are deduplicated,
    // while distinct client ids and distinct drivers are summed.
    const std::string dedup =
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t42\n"
        "drm-memory-vram:\t2048 MiB\n"
        "drm-memory-gtt:\t128 MiB\n"
        "\n"
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t42\n"
        "drm-memory-vram:\t2048 MiB\n"
        "drm-memory-gtt:\t128 MiB\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t42\n"
        "drm-memory-vram:\t2048 MiB\n"
        "drm-memory-gtt:\t128 MiB\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t43\n"
        "drm-memory-vram:\t512 MiB\n"
        "drm-memory-gtt:\t64 MiB\n"
        "drm-driver:\ti915\n"
        "drm-client-id:\t42\n"
        "drm-memory-vram:\t100 MiB\n"
        "drm-memory-gtt:\t32 MiB\n";
    REQUIRE(parse_linux_drm_fdinfo(dedup, vram, gtt));
    CHECK(vram == (2048ull + 512ull + 100ull) * MiB);
    CHECK(gtt == (128ull + 64ull + 32ull) * MiB);

    // Malformed / non-DRM inputs return false and reset outputs to 0.
    vram = 99;
    gtt = 99;
    CHECK_FALSE(parse_linux_drm_fdinfo("", vram, gtt));
    CHECK(vram == 0);
    CHECK(gtt == 0);

    // Hybrid AMD+NVIDIA Linux host: nvidia-drm/nvidia fdinfo entries are excluded
    // so NVIDIA VRAM is never double-counted alongside NVML.
    const std::string nvidia_only =
        "drm-driver:\tnvidia-drm\n"
        "drm-client-id:\t1\n"
        "drm-memory-vram:\t14000 MiB\n"
        "drm-memory-gtt:\t128 MiB\n";
    vram = 55;
    gtt = 66;
    CHECK_FALSE(parse_linux_drm_fdinfo(nvidia_only, vram, gtt));
    CHECK(vram == 0);
    CHECK(gtt == 0);

    const std::string hybrid =
        "pos:\t0\n"
        "drm-driver:\tnvidia-drm\n"
        "drm-client-id:\t1\n"
        "drm-memory-vram:\t14000 MiB\n"
        "drm-memory-gtt:\t128 MiB\n"
        "\n"
        "pos:\t0\n"
        "drm-driver:\tnvidia\n"
        "drm-client-id:\t2\n"
        "drm-memory-vram:\t2000 MiB\n"
        "drm-memory-gtt:\t64 MiB\n"
        "\n"
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        "drm-client-id:\t3\n"
        "drm-memory-vram:\t1024 MiB\n"
        "drm-memory-gtt:\t96 MiB\n";
    REQUIRE(parse_linux_drm_fdinfo(hybrid, vram, gtt));
    CHECK(vram == 1024ull * MiB);
    CHECK(gtt == 96ull * MiB);

    const std::string malformed =
        "pos:\t0\n"
        "flags:\t0100000\n"
        "drm-memory-vram:\t-10 MiB\n"
        "drm-memory-gtt:\tabc KiB\n"
        "drm-memory-vram:\t1024\n"
        "drm-memory-gtt:\t50 XB\n"
        "drm-memory-vram:\t99999999999999999999 GiB\n";
    vram = 77;
    gtt = 88;
    CHECK_FALSE(parse_linux_drm_fdinfo(malformed, vram, gtt));
    CHECK(vram == 0);
    CHECK(gtt == 0);
}

TEST_CASE("is_spilled suppresses shared-memory spill false positives in uma_mode") {
    GpuMemorySample sample;
    sample.dedicated_bytes = 512 * MiB;
    sample.shared_bytes = 12 * GiB;

    SpillGuardOptions discrete;
    CHECK_FALSE(discrete.uma_mode);
    CHECK(is_spilled(sample, discrete));

    SpillGuardOptions uma = discrete;
    uma.uma_mode = true;
    CHECK(validate_spill_guard(uma).ok());
    CHECK_FALSE(is_spilled(sample, uma));
    CHECK_FALSE(is_spilled(sample, uma, 65536u));
}

TEST_CASE("residency guard counts shared memory as active GPU memory only in uma_mode") {
    LlamaServerResidencyGuardOptions uma;
    uma.uma_mode = true;
    uma.consecutive_samples = 2;
    uma.min_dedicated_bytes = 512 * MiB;
    GpuResidencyGuard apu(uma, true);

    // Ryzen APU: small dedicated carveout (256 MiB) + 8 GiB in GTT/shared is healthy.
    apu.observe(256 * MiB, 8 * GiB);
    apu.observe(256 * MiB, 8 * GiB);
    CHECK_FALSE(apu.gpu_offload_missing());
    CHECK_FALSE(apu.vram_evicted());
    CHECK(apu.peak_dedicated_bytes() == 256 * MiB + 8 * GiB);

    // A steep drop in combined UMA usage still detects eviction.
    apu.observe(128 * MiB, 100 * MiB);
    CHECK_FALSE(apu.vram_evicted());
    apu.observe(128 * MiB, 100 * MiB);
    CHECK(apu.vram_evicted());
    CHECK(apu.gpu_offload_missing());

    // Pure CPU fallback on an APU (0 dedicated + 0 shared) triggers gpu_offload_missing.
    GpuResidencyGuard cpu_fallback(uma, true);
    cpu_fallback.observe(0, 0);
    CHECK_FALSE(cpu_fallback.gpu_offload_missing());
    cpu_fallback.observe(0, 0);
    CHECK(cpu_fallback.gpu_offload_missing());

    // Saturating addition guards against uint64_t overflow.
    GpuResidencyGuard overflow_check(uma, true);
    overflow_check.observe(std::numeric_limits<std::uint64_t>::max() - 10, 100);
    CHECK_FALSE(overflow_check.gpu_offload_missing());
    CHECK(overflow_check.peak_dedicated_bytes() == std::numeric_limits<std::uint64_t>::max());

    // Discrete mode (uma_mode == false): shared memory must NOT hide missing dedicated offload.
    LlamaServerResidencyGuardOptions discrete;
    discrete.uma_mode = false;
    discrete.consecutive_samples = 2;
    GpuResidencyGuard dgpu(discrete, true);
    dgpu.observe(100 * MiB, 8 * GiB);
    dgpu.observe(100 * MiB, 8 * GiB);
    CHECK(dgpu.gpu_offload_missing());
    CHECK(dgpu.peak_dedicated_bytes() == 100 * MiB);
}

TEST_CASE("parse_pdh_adapter_luid extracts WDDM high/low LUID pairs and rejects malformed names") {
    const auto std_luid = parse_pdh_adapter_luid("pid_1234_luid_0x00000000_0x000104A4_phys_0");
    REQUIRE(std_luid.has_value());
    CHECK(std_luid->high == 0x00000000u);
    CHECK(std_luid->low == 0x000104A4u);

    const auto short_luid = parse_pdh_adapter_luid("pid_9_LUID_0x0_0x1");
    REQUIRE(short_luid.has_value());
    CHECK(short_luid->high == 0u);
    CHECK(short_luid->low == 1u);

    const auto upper_luid = parse_pdh_adapter_luid("pid_77_luid_0XFFFFFFFF_0x1234abcd_phys_1");
    REQUIRE(upper_luid.has_value());
    CHECK(upper_luid->high == 0xFFFFFFFFu);
    CHECK(upper_luid->low == 0x1234ABCDu);

    CHECK_FALSE(parse_pdh_adapter_luid("").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_luid_test").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_drm_fdinfo").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_nvml_0").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_luid_0x123456789_0x1_phys_0").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_luid_0x0_0x1xyz").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_luid_0x_0x1_phys_0").has_value());
    CHECK_FALSE(parse_pdh_adapter_luid("pid_1234_luid_0x0_phys_0").has_value());
}

TEST_CASE("classify_pdh_counters and summarize_gpu_counters separate discrete and UMA adapters") {
    constexpr AdapterLuid kDiscreteLuid{0x00000000u, 0x000104A4u};  // e.g. RX 9060 XT
    constexpr AdapterLuid kUmaLuid{0x00000000u, 0x000118C2u};       // e.g. Radeon 780M
    const std::vector<AdapterLuidInfo> adapters{
        {kDiscreteLuid, false},
        {kUmaLuid, true},
    };

    GpuProcessCounters c;
    c.dedicated = {
        {"pid_42_luid_0x00000000_0x000104A4_phys_0", 15 * GiB},
        {"pid_42_luid_0x00000000_0x000118C2_phys_0", 512 * MiB},
        {"pid_42_luid_0x00000000_0x00099999_phys_0", 0},         // 0-byte software/WARP adapter
        {"pid_420_luid_0x00000000_0x000104A4_phys_0", 10 * GiB}, // different PID
    };
    c.shared = {
        {"pid_42_luid_0x00000000_0x000104A4_phys_0", 160 * MiB},
        {"pid_42_luid_0x00000000_0x000118C2_phys_0", 25 * GiB},
        {"pid_42_luid_0x00000000_0x00099999_phys_0", 0},
        {"pid_420_luid_0x00000000_0x000118C2_phys_0", 8 * GiB},
    };

    classify_pdh_counters(c, adapters);
    const auto s = summarize_gpu_counters(42, c);
    CHECK(s.has_adapter_classification);
    CHECK(s.instances == 3);
    CHECK(s.discrete_instances == 1);
    CHECK(s.uma_instances == 1);
    CHECK(s.dedicated_bytes == 15 * GiB + 512 * MiB);
    CHECK(s.shared_bytes == 25 * GiB + 160 * MiB);
    CHECK(s.discrete_dedicated_bytes == 15 * GiB);
    CHECK(s.discrete_shared_bytes == 160 * MiB);
    CHECK(s.uma_dedicated_bytes == 512 * MiB);
    CHECK(s.uma_shared_bytes == 25 * GiB);

    // Non-zero unknown adapter fails safe and disables per-adapter classification.
    c.shared.push_back({"pid_42_luid_0x00000000_0x00088888_phys_0", 400 * MiB});
    const auto partial = summarize_gpu_counters(42, c);
    CHECK_FALSE(partial.has_adapter_classification);
    CHECK(partial.shared_bytes == 25 * GiB + 560 * MiB);
}

TEST_CASE("is_spilled distinguishes pure discrete, pure UMA, and hybrid discrete+iGPU pools") {
    constexpr AdapterLuid kDiscreteLuid{0x00000000u, 0x000104A4u};
    constexpr AdapterLuid kDiscrete2Luid{0x00000000u, 0x000105B5u};
    constexpr AdapterLuid kUmaLuid{0x00000000u, 0x000118C2u};

    const auto make_classified = [](std::uint32_t pid,
                                    const std::vector<AdapterLuidInfo> &adapters,
                                    std::vector<CounterInstance> dedicated,
                                    std::vector<CounterInstance> shared) {
        GpuProcessCounters c{std::move(dedicated), std::move(shared)};
        classify_pdh_counters(c, adapters);
        return summarize_gpu_counters(pid, c);
    };

    const std::vector<AdapterLuidInfo> hybrid_adapters{{kDiscreteLuid, false}, {kUmaLuid, true}};
    const std::vector<AdapterLuidInfo> dual_discrete_adapters{{kDiscreteLuid, false}, {kDiscrete2Luid, false}};

    // 1. Pure discrete (single GPU & dual discrete): clean passes, spill triggers.
    {
        SpillGuardOptions discrete;
        const auto clean = make_classified(
            10, hybrid_adapters,
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 15 * GiB}},
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 180 * MiB}});
        CHECK(clean.has_adapter_classification);
        CHECK(effective_spill_shared_bytes(clean, discrete) == 180 * MiB);
        CHECK_FALSE(is_spilled(clean, discrete));

        const auto spilled = make_classified(
            10, hybrid_adapters,
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 15 * GiB}},
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 388 * MiB}});
        CHECK(effective_spill_shared_bytes(spilled, discrete) == 388 * MiB);
        CHECK(is_spilled(spilled, discrete));

        const auto dual_spilled = make_classified(
            10, dual_discrete_adapters,
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 15 * GiB},
             {"pid_10_luid_0x00000000_0x000105B5_phys_0", 15 * GiB}},
            {{"pid_10_luid_0x00000000_0x000104A4_phys_0", 120 * MiB},
             {"pid_10_luid_0x00000000_0x000105B5_phys_0", 300 * MiB}});
        CHECK(effective_spill_shared_bytes(dual_spilled, discrete) == 420 * MiB);
        CHECK(is_spilled(dual_spilled, discrete));
    }

    // 2. Pure iGPU UMA: uma_mode=true suppresses spill; explicit uma_mode=false keeps legacy check.
    {
        const auto apu_sample = make_classified(
            20, hybrid_adapters,
            {{"pid_20_luid_0x00000000_0x000118C2_phys_0", 512 * MiB}},
            {{"pid_20_luid_0x00000000_0x000118C2_phys_0", 25 * GiB}});
        CHECK(apu_sample.has_adapter_classification);
        CHECK(apu_sample.discrete_instances == 0);
        CHECK(apu_sample.uma_instances == 1);

        SpillGuardOptions uma;
        uma.uma_mode = true;
        CHECK(effective_spill_shared_bytes(apu_sample, uma) == 0);
        CHECK_FALSE(is_spilled(apu_sample, uma));

        SpillGuardOptions explicit_off;
        explicit_off.uma_mode = false;
        CHECK(effective_spill_shared_bytes(apu_sample, explicit_off) == 25 * GiB);
        CHECK(is_spilled(apu_sample, explicit_off));
    }

    // 3. Hybrid discrete + iGPU pool (Node1 RX 9060 XT + Radeon 780M):
    //    ~25 GiB intentional iGPU UMA shared allocation does NOT trigger spill,
    //    while genuine discrete-GPU PCIe spill (> threshold) STILL triggers!
    {
        SpillGuardOptions hybrid;
        hybrid.uma_mode = true;
        hybrid.hybrid_uma_mode = true;

        const auto healthy_hybrid = make_classified(
            30, hybrid_adapters,
            {{"pid_30_luid_0x00000000_0x000104A4_phys_0", 15 * GiB},
             {"pid_30_luid_0x00000000_0x000118C2_phys_0", 512 * MiB}},
            {{"pid_30_luid_0x00000000_0x000104A4_phys_0", 160 * MiB},
             {"pid_30_luid_0x00000000_0x000118C2_phys_0", 25 * GiB}});
        CHECK(effective_spill_shared_bytes(healthy_hybrid, hybrid) == 160 * MiB);
        CHECK_FALSE(is_spilled(healthy_hybrid, hybrid));

        // Defense in depth: even when uma_mode=false on a dual-adapter sample,
        // per-adapter classification attributes spill solely to discrete adapters.
        SpillGuardOptions legacy_discrete;
        CHECK(effective_spill_shared_bytes(healthy_hybrid, legacy_discrete) == 160 * MiB);
        CHECK_FALSE(is_spilled(healthy_hybrid, legacy_discrete));

        const auto discrete_spilled_hybrid = make_classified(
            30, hybrid_adapters,
            {{"pid_30_luid_0x00000000_0x000104A4_phys_0", 15 * GiB},
             {"pid_30_luid_0x00000000_0x000118C2_phys_0", 512 * MiB}},
            {{"pid_30_luid_0x00000000_0x000104A4_phys_0", 950 * MiB},
             {"pid_30_luid_0x00000000_0x000118C2_phys_0", 25 * GiB}});
        CHECK(effective_spill_shared_bytes(discrete_spilled_hybrid, hybrid) == 950 * MiB);
        CHECK(is_spilled(discrete_spilled_hybrid, hybrid));
        CHECK(is_spilled(discrete_spilled_hybrid, legacy_discrete));

        // Explicit pure uma_mode=true (hybrid_uma_mode=false, e.g. user set "on")
        // suppresses spill unconditionally.
        SpillGuardOptions forced_on;
        forced_on.uma_mode = true;
        forced_on.hybrid_uma_mode = false;
        CHECK_FALSE(is_spilled(discrete_spilled_hybrid, forced_on));

        // Unclassified aggregate counters in hybrid mode suppress false-positive kills.
        GpuMemorySample aggregate_hybrid;
        aggregate_hybrid.dedicated_bytes = 15 * GiB + 512 * MiB;
        aggregate_hybrid.shared_bytes = 25 * GiB + 160 * MiB;
        aggregate_hybrid.instances = 2;
        CHECK_FALSE(aggregate_hybrid.has_adapter_classification);
        CHECK(effective_spill_shared_bytes(aggregate_hybrid, hybrid) == 0);
        CHECK_FALSE(is_spilled(aggregate_hybrid, hybrid));
    }
}
