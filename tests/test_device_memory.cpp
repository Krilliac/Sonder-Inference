#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "sonder/inference/device.hpp"

using namespace sonder::inference;

namespace {

FileReader fake_files(std::unordered_map<std::string, std::string> files) {
    return [files = std::move(files)](const std::string& path) -> std::optional<std::string> {
        const auto it = files.find(path);
        if (it == files.end()) {
            return std::nullopt;
        }
        return it->second;
    };
}

}  // namespace

TEST_CASE("MemoryReading keeps unknown, known zero and known value distinct") {
    const MemoryReading unknown;
    const auto zero = MemoryReading::known(0);
    const auto some = MemoryReading::known(42);
    CHECK_FALSE(unknown.is_known());
    CHECK_FALSE(unknown.is_known_zero());
    CHECK_FALSE(unknown.value().has_value());
    CHECK(unknown.bytes_or(7) == 7);
    CHECK(zero.is_known());
    CHECK(zero.is_known_zero());
    CHECK(zero.value() == std::optional<std::uint64_t>(0));
    CHECK(zero.bytes_or(7) == 0);
    CHECK(some.is_known());
    CHECK_FALSE(some.is_known_zero());
    CHECK(some.bytes_or(7) == 42);
    CHECK(unknown != zero);
    CHECK(zero != some);
    CHECK(MemoryReading::unknown() == unknown);
}

TEST_CASE("live host memory reading is known and positive on supported hosts") {
    // A reader that silently returned "unknown" or 0 would make every
    // admission check below vacuous, so the live value must be real here.
    const HostMemory live = read_host_memory();
#if defined(_WIN32) || defined(__linux__)
    REQUIRE_MESSAGE(live.available_physical.is_known(), live.note);
    CHECK(live.available_physical.bytes_or(0) > 0);
    CHECK(live.total_physical.bytes_or(0) > 0);
    CHECK(live.total_physical.bytes_or(0) >= live.available_physical.bytes_or(0));
    CHECK_FALSE(live.source.empty());
    const auto devices = enumerate_devices();
    REQUIRE(devices.size() == 1);
    CHECK(devices[0].available_physical.is_known());
    CHECK(devices[0].available_memory_bytes > 0);
#else
    CHECK_FALSE(live.available_physical.is_known());  // documented: unknown off Windows/Linux
#endif
#if defined(_WIN32)
    REQUIRE(live.available_commit.is_known());
    CHECK(live.available_commit.bytes_or(0) > 0);
#else
    CHECK_FALSE(live.available_commit.is_known());
#endif
}

TEST_CASE("parse_strict_u64 accepts only canonical decimal") {
    CHECK(parse_strict_u64("0") == std::optional<std::uint64_t>(0));
    CHECK(parse_strict_u64("12345\n") == std::optional<std::uint64_t>(12345));
    CHECK(parse_strict_u64("18446744073709551615") == std::optional<std::uint64_t>(18446744073709551615ull));
    CHECK_FALSE(parse_strict_u64("18446744073709551616").has_value());
    CHECK_FALSE(parse_strict_u64("99999999999999999999").has_value());
    CHECK_FALSE(parse_strict_u64("").has_value());
    CHECK_FALSE(parse_strict_u64("\n").has_value());
    CHECK_FALSE(parse_strict_u64("max").has_value());
    CHECK_FALSE(parse_strict_u64("-1").has_value());
    CHECK_FALSE(parse_strict_u64("+1").has_value());
    CHECK_FALSE(parse_strict_u64(" 1").has_value());
    CHECK_FALSE(parse_strict_u64("1 ").has_value());
    CHECK_FALSE(parse_strict_u64("1\n\n").has_value());
    CHECK_FALSE(parse_strict_u64("007").has_value());
    CHECK_FALSE(parse_strict_u64("0x10").has_value());
    CHECK_FALSE(parse_strict_u64("1e3").has_value());
}

TEST_CASE("parse_meminfo_bytes is strict about shape and duplicates") {
    const std::string good =
        "MemTotal:       32768000 kB\nMemFree:         1000 kB\nMemAvailable:   16384000 kB\nSwapTotal:             0 kB\n";
    CHECK(parse_meminfo_bytes(good, "MemTotal") == std::optional<std::uint64_t>(32768000ull * 1024));
    CHECK(parse_meminfo_bytes(good, "MemAvailable") == std::optional<std::uint64_t>(16384000ull * 1024));
    CHECK_FALSE(parse_meminfo_bytes(good, "MemAvail").has_value());  // a prefix is not the key
    CHECK_FALSE(parse_meminfo_bytes(good, "Missing").has_value());
    CHECK(parse_meminfo_bytes("MemAvailable: 0 kB\n", "MemAvailable") == std::optional<std::uint64_t>(0));
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: 1 kB\nMemAvailable: 2 kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable:1 kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: 1 MB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: 1kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: x kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: -5 kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: 18446744073709551615 kB\n", "MemAvailable").has_value());
    CHECK_FALSE(parse_meminfo_bytes("MemAvailable: kB\n", "MemAvailable").has_value());
}

TEST_CASE("cgroup v2 headroom takes the tightest limit over ancestors") {
    const auto files = fake_files({
        {"/cg/a/b/memory.max", "1000\n"},
        {"/cg/a/b/memory.current", "400\n"},
        {"/cg/a/memory.max", "500\n"},
        {"/cg/a/memory.current", "300\n"},
    });
    const auto h = cgroup_v2_headroom("0::/a/b\n", files, "/cg");
    CHECK(h.limited);
    CHECK_FALSE(h.malformed);
    CHECK(h.headroom == MemoryReading::known(200));  // ancestor a: 500 - 300 beats b: 600
}

TEST_CASE("cgroup v2 headroom reads the namespace-root cgroup for a private cgroup namespace") {
    // Docker's default on cgroup v2: /proc/self/cgroup is "0::/" and the
    // container limit sits in <root>/memory.max.
    const auto files = fake_files({{"/cg/memory.max", "1000\n"}, {"/cg/memory.current", "400\n"}});
    const auto h = cgroup_v2_headroom("0::/\n", files, "/cg");
    CHECK(h.limited);
    CHECK_FALSE(h.malformed);
    CHECK(h.headroom == MemoryReading::known(600));

    // A nested path still honours a tighter limit at the root.
    const auto nested = fake_files({{"/cg/memory.max", "500\n"},
                                    {"/cg/memory.current", "450\n"},
                                    {"/cg/a/memory.max", "1000\n"},
                                    {"/cg/a/memory.current", "100\n"}});
    CHECK(cgroup_v2_headroom("0::/a\n", nested, "/cg").headroom == MemoryReading::known(50));

    // The host's real root cgroup has no memory.max/current: not limited.
    const auto none = cgroup_v2_headroom("0::/\n", fake_files({}), "/cg");
    CHECK_FALSE(none.limited);
    CHECK_FALSE(none.malformed);
}

TEST_CASE("cgroup v2 headroom: unlimited levels are skipped and a limit at or over max is known zero") {
    const auto unlimited = fake_files({{"/cg/a/memory.max", "max\n"}, {"/cg/a/memory.current", "5\n"}});
    const auto none = cgroup_v2_headroom("0::/a\n", unlimited, "/cg");
    CHECK_FALSE(none.limited);
    CHECK_FALSE(none.malformed);
    CHECK_FALSE(none.headroom.is_known());

    const auto over = fake_files({{"/cg/a/memory.max", "100\n"}, {"/cg/a/memory.current", "150\n"}});
    const auto zero = cgroup_v2_headroom("0::/a\n", over, "/cg");
    CHECK(zero.limited);
    CHECK(zero.headroom.is_known_zero());

    // v1-only host: no "0::" line, so no cgroup constraint and no error.
    const auto v1 = cgroup_v2_headroom("4:memory:/x\n", unlimited, "/cg");
    CHECK_FALSE(v1.limited);
    CHECK_FALSE(v1.malformed);
}

TEST_CASE("cgroup v2 headroom refuses malformed files and paths") {
    const auto bad_max = fake_files({{"/cg/a/memory.max", "10 \n"}, {"/cg/a/memory.current", "5\n"}});
    CHECK(cgroup_v2_headroom("0::/a\n", bad_max, "/cg").malformed);
    const auto bad_cur = fake_files({{"/cg/a/memory.max", "10\n"}, {"/cg/a/memory.current", "five\n"}});
    CHECK(cgroup_v2_headroom("0::/a\n", bad_cur, "/cg").malformed);
    const auto lonely = fake_files({{"/cg/a/memory.max", "10\n"}});
    CHECK(cgroup_v2_headroom("0::/a\n", lonely, "/cg").malformed);
    const auto huge = fake_files({{"/cg/a/memory.max", "99999999999999999999\n"}, {"/cg/a/memory.current", "5\n"}});
    CHECK(cgroup_v2_headroom("0::/a\n", huge, "/cg").malformed);
    // Valid, readable, tight limits at /cg/a and /cg/b: with the canonical-path
    // refusal removed the walk would reach /cg/a and report a limit, so only
    // that refusal can produce "malformed" here.
    const auto valid = fake_files({{"/cg/a/memory.max", "100\n"},
                                   {"/cg/a/memory.current", "5\n"},
                                   {"/cg/b/memory.max", "100\n"},
                                   {"/cg/b/memory.current", "5\n"}});
    const auto dotdot = cgroup_v2_headroom("0::/a/../b\n", valid, "/cg");
    CHECK(dotdot.malformed);
    CHECK_FALSE(dotdot.limited);
    CHECK(dotdot.note == "cgroup path is not canonical");
    const auto empty_part = cgroup_v2_headroom("0::/a//b\n", valid, "/cg");
    CHECK(empty_part.malformed);
    CHECK_FALSE(empty_part.limited);
    CHECK(empty_part.note == "cgroup path is not canonical");
    const auto dot_part = cgroup_v2_headroom("0::/a/./b\n", valid, "/cg");
    CHECK(dot_part.malformed);
    CHECK_FALSE(dot_part.limited);
    const auto relative = cgroup_v2_headroom("0::a\n", valid, "/cg");
    CHECK(relative.malformed);
    CHECK_FALSE(relative.limited);
    CHECK(relative.note == "cgroup path is not absolute");
    CHECK(cgroup_v2_headroom("0::/a\n0::/b\n", bad_max, "/cg").malformed);
}

TEST_CASE("linux composition caps MemAvailable by the cgroup and never guesses past a malformed one") {
    const std::string meminfo = "MemTotal: 1000 kB\nMemAvailable: 800 kB\n";
    const CgroupHeadroom none;
    const auto plain = compose_linux_host_memory(meminfo, none);
    CHECK(plain.total_physical == MemoryReading::known(1000 * 1024));
    CHECK(plain.available_physical == MemoryReading::known(800 * 1024));
    CHECK_FALSE(plain.available_commit.is_known());

    CgroupHeadroom tight;
    tight.limited = true;
    tight.headroom = MemoryReading::known(100 * 1024);
    CHECK(compose_linux_host_memory(meminfo, tight).available_physical == MemoryReading::known(100 * 1024));

    CgroupHeadroom loose = tight;
    loose.headroom = MemoryReading::known(900 * 1024);
    CHECK(compose_linux_host_memory(meminfo, loose).available_physical == MemoryReading::known(800 * 1024));

    CgroupHeadroom exhausted = tight;
    exhausted.headroom = MemoryReading::known(0);
    CHECK(compose_linux_host_memory(meminfo, exhausted).available_physical.is_known_zero());

    CgroupHeadroom broken;
    broken.malformed = true;
    broken.note = "x";
    CHECK_FALSE(compose_linux_host_memory(meminfo, broken).available_physical.is_known());

    // MemAvailable absent: unknown, unless a cgroup limit still bounds it.
    CHECK_FALSE(compose_linux_host_memory("MemTotal: 1000 kB\n", none).available_physical.is_known());
    CHECK(compose_linux_host_memory("MemTotal: 1000 kB\n", tight).available_physical ==
          MemoryReading::known(100 * 1024));
    // MemAvailable of exactly 0 is known zero, not unknown.
    CHECK(compose_linux_host_memory("MemAvailable: 0 kB\n", none).available_physical.is_known_zero());
}

TEST_CASE("GPU vendor PCI IDs map to canonical vendors and strings") {
    CHECK(gpu_vendor_from_pci_id(0x10DEu) == GpuVendor::nvidia);
    CHECK(gpu_vendor_from_pci_id(0x1002u) == GpuVendor::amd);
    CHECK(gpu_vendor_from_pci_id(0x8086u) == GpuVendor::intel);
    CHECK(gpu_vendor_from_pci_id(0x106Bu) == GpuVendor::apple);
    CHECK(gpu_vendor_from_pci_id(0x0000u) == GpuVendor::unknown);
    CHECK(gpu_vendor_from_pci_id(0x1AF4u) == GpuVendor::unknown);

    CHECK(std::string(to_string(GpuVendor::nvidia)) == "nvidia");
    CHECK(std::string(to_string(GpuVendor::amd)) == "amd");
    CHECK(std::string(to_string(GpuVendor::intel)) == "intel");
    CHECK(std::string(to_string(GpuVendor::apple)) == "apple");
    CHECK(std::string(to_string(GpuVendor::unknown)) == "unknown");
}

TEST_CASE("parse_amdgpu_sysfs_device parses discrete RX XT and integrated Ryzen APU UMA devices") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // Discrete AMD Radeon RX 7900 XT (20 GiB VRAM, 16 GiB GTT).
    const auto rx7900xt_files = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/device", "0x744c\n"},
        {"/sys/class/drm/card0/device/product_name", "AMD Radeon RX 7900 XT\n"},
        {"/sys/class/drm/card0/device/uevent", "DRIVER=amdgpu\nPCI_ID=1002:744C\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(20 * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_used", std::to_string(2 * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_gtt_used", std::to_string(1 * kGiB) + "\n"},
    });
    const auto rx7900xt = parse_amdgpu_sysfs_device("card0", rx7900xt_files);
    REQUIRE(rx7900xt.has_value());
    CHECK(rx7900xt->id == "gpu:0");
    CHECK(rx7900xt->name == "AMD Radeon RX 7900 XT");
    CHECK(rx7900xt->vendor == GpuVendor::amd);
    CHECK(rx7900xt->vendor_id == 0x1002u);
    CHECK(rx7900xt->device_id == 0x744cu);
    CHECK_FALSE(rx7900xt->is_integrated_uma);
    CHECK(rx7900xt->dedicated_total == MemoryReading::known(20 * kGiB));
    CHECK(rx7900xt->dedicated_free == MemoryReading::known(18 * kGiB));
    CHECK(rx7900xt->shared_total == MemoryReading::known(16 * kGiB));
    CHECK(rx7900xt->backend_hint == "rocm");
    CHECK(rx7900xt->source == "amdgpu-sysfs");
    CHECK(gpu_effective_vram_mib(*rx7900xt, false) == std::optional<std::uint64_t>(20480));
    CHECK(gpu_effective_vram_mib(*rx7900xt, true) == std::optional<std::uint64_t>(18432));

    // Discrete AMD Radeon RX 9070 XT (16 GiB VRAM, fallback name from PCI device id).
    const auto rx9070xt_files = fake_files({
        {"/sys/class/drm/card2/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card2/device/uevent", "DRIVER=amdgpu\nPCI_ID=1002:7550\n"},
        {"/sys/class/drm/card2/device/mem_info_vram_total", std::to_string(16 * kGiB) + "\n"},
        {"/sys/class/drm/card2/device/mem_info_vram_used", std::to_string(512 * kMiB) + "\n"},
        {"/sys/class/drm/card2/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
    });
    const auto rx9070xt = parse_amdgpu_sysfs_device("/sys/class/drm/card2/device", rx9070xt_files);
    REQUIRE(rx9070xt.has_value());
    CHECK(rx9070xt->id == "gpu:2");
    CHECK(rx9070xt->device_id == 0x7550u);
    CHECK(rx9070xt->name == "AMD Radeon (0x7550)");
    CHECK_FALSE(rx9070xt->is_integrated_uma);
    CHECK(gpu_effective_vram_mib(*rx9070xt, false) == std::optional<std::uint64_t>(16384));
    CHECK(gpu_effective_vram_mib(*rx9070xt, true) == std::optional<std::uint64_t>(15872));

    // Integrated Ryzen 780M APU UMA (512 MiB BIOS carveout + 16 GiB GTT).
    const auto apu_files = fake_files({
        {"/sys/class/drm/card1/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card1/device/device", "0x15bf\n"},
        {"/sys/class/drm/card1/device/product_name", "AMD Radeon 780M Graphics\n"},
        {"/sys/class/drm/card1/device/mem_info_vram_total", std::to_string(512 * kMiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_vram_used", std::to_string(128 * kMiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_gtt_used", std::to_string(2 * kGiB) + "\n"},
    });
    const auto apu = parse_amdgpu_sysfs_device("card1", apu_files);
    REQUIRE(apu.has_value());
    CHECK(apu->id == "gpu:1");
    CHECK(apu->name == "AMD Radeon 780M Graphics");
    CHECK(apu->is_integrated_uma);
    CHECK(apu->dedicated_total == MemoryReading::known(512 * kMiB));
    CHECK(apu->shared_total == MemoryReading::known(16 * kGiB));
    CHECK(apu->dedicated_free == MemoryReading::known(384 * kMiB + 14 * kGiB));
    CHECK(gpu_effective_vram_mib(*apu, false) == std::optional<std::uint64_t>(512 + 16384));
    CHECK(gpu_effective_vram_mib(*apu, true) == std::optional<std::uint64_t>(384 + 14336));

    // Integrated APU without mem_info_gtt_used: prefer_free falls back to full UMA pool, not 384 MiB carveout.
    const auto apu_no_gtt_used = fake_files({
        {"/sys/class/drm/card1/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card1/device/mem_info_vram_total", std::to_string(512 * kMiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_vram_used", std::to_string(128 * kMiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
    });
    const auto apu2 = parse_amdgpu_sysfs_device("card1", apu_no_gtt_used);
    REQUIRE(apu2.has_value());
    CHECK(apu2->is_integrated_uma);
    CHECK_FALSE(apu2->dedicated_free.is_known());
    CHECK(gpu_effective_vram_mib(*apu2, true) == std::optional<std::uint64_t>(512 + 16384));

    // Reject non-AMD vendors, non-amdgpu drivers, path traversal, and missing memory info.
    const auto nvidia_card = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x10de\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(16 * kGiB) + "\n"},
    });
    CHECK_FALSE(parse_amdgpu_sysfs_device("card0", nvidia_card).has_value());

    const auto legacy_radeon = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/uevent", "DRIVER=radeon\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(4 * kGiB) + "\n"},
    });
    CHECK_FALSE(parse_amdgpu_sysfs_device("card0", legacy_radeon).has_value());
    CHECK_FALSE(parse_amdgpu_sysfs_device("../card0", rx7900xt_files).has_value());
    CHECK_FALSE(parse_amdgpu_sysfs_device("card0", fake_files({{"/sys/class/drm/card0/device/vendor", "0x1002\n"}}))
                    .has_value());
}

TEST_CASE("select_gpu_vram_budget_mib prefers discrete over iGPU and honors device hints") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    GpuDeviceInfo rtx;
    rtx.id = "gpu:0";
    rtx.name = "NVIDIA GeForce RTX 5070 Ti";
    rtx.vendor = GpuVendor::nvidia;
    rtx.vendor_id = 0x10DEu;
    rtx.dedicated_total = MemoryReading::known(16303 * kMiB);
    rtx.dedicated_free = MemoryReading::known(14800 * kMiB);
    rtx.shared_total = MemoryReading::known(16 * kGiB);
    rtx.is_integrated_uma = false;
    rtx.backend_hint = "cuda";
    rtx.source = "dxgi";

    GpuDeviceInfo ryzen_igpu;
    ryzen_igpu.id = "gpu:1";
    ryzen_igpu.name = "AMD Radeon(TM) Graphics";
    ryzen_igpu.vendor = GpuVendor::amd;
    ryzen_igpu.vendor_id = 0x1002u;
    ryzen_igpu.dedicated_total = MemoryReading::known(512 * kMiB);
    ryzen_igpu.dedicated_free = MemoryReading::known(15000 * kMiB);
    ryzen_igpu.shared_total = MemoryReading::known(16 * kGiB);
    ryzen_igpu.is_integrated_uma = true;
    ryzen_igpu.backend_hint = "vulkan";
    ryzen_igpu.source = "dxgi";

    // Even when the iGPU's UMA pool (16896 MiB) exceeds the discrete GPU's dedicated
    // VRAM (16303 MiB), default selection picks the discrete RTX regardless of order.
    const std::vector<GpuDeviceInfo> rtx_first{rtx, ryzen_igpu};
    CHECK(select_gpu_vram_budget_mib(rtx_first) == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(rtx_first, {}, true) == std::optional<std::uint64_t>(14800));

    GpuDeviceInfo igpu_zero = ryzen_igpu;
    igpu_zero.id = "gpu:0";
    GpuDeviceInfo rtx_one = rtx;
    rtx_one.id = "gpu:1";
    const std::vector<GpuDeviceInfo> igpu_first{igpu_zero, rtx_one};
    CHECK(select_gpu_vram_budget_mib(igpu_first) == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "CUDA0") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "CUDA0", true) == std::optional<std::uint64_t>(14800));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "ROCm0") == std::optional<std::uint64_t>(16896));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "HIP0", true) == std::optional<std::uint64_t>(15000));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "amd") == std::optional<std::uint64_t>(16896));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "nvidia") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "Vulkan0") == std::optional<std::uint64_t>(16896));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "Vulkan1") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "gpu:1") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib(igpu_first, "gpu:0") == std::optional<std::uint64_t>(16896));

    // Explicit hints fail closed when no matching device exists.
    CHECK_FALSE(select_gpu_vram_budget_mib(rtx_first, "CUDA1").has_value());
    CHECK_FALSE(select_gpu_vram_budget_mib(rtx_first, "gpu:5").has_value());
    CHECK_FALSE(select_gpu_vram_budget_mib(rtx_first, "intel").has_value());
    CHECK_FALSE(select_gpu_vram_budget_mib({}).has_value());

    // Multi-device hints sum unique GPUs; UMA-only hosts report the full UMA pool.
    CHECK(select_gpu_vram_budget_mib(rtx_first, "CUDA0,ROCm0") == std::optional<std::uint64_t>(16303 + 16896));
    CHECK(select_gpu_vram_budget_mib(rtx_first, "CUDA0,CUDA0") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib({ryzen_igpu}) == std::optional<std::uint64_t>(16896));
    CHECK(select_gpu_vram_budget_mib({ryzen_igpu}, {}, true) == std::optional<std::uint64_t>(15000));

    // Dual discrete GPUs pick the largest discrete GPU by default.
    GpuDeviceInfo rx7900xt;
    rx7900xt.id = "gpu:1";
    rx7900xt.name = "AMD Radeon RX 7900 XT";
    rx7900xt.vendor = GpuVendor::amd;
    rx7900xt.vendor_id = 0x1002u;
    rx7900xt.dedicated_total = MemoryReading::known(20 * kGiB);
    rx7900xt.dedicated_free = MemoryReading::known(19 * kGiB);
    rx7900xt.is_integrated_uma = false;
    rx7900xt.backend_hint = "rocm";
    CHECK(select_gpu_vram_budget_mib({rtx, rx7900xt}) == std::optional<std::uint64_t>(20480));
    CHECK(select_gpu_vram_budget_mib({rtx, rx7900xt}, "CUDA0") == std::optional<std::uint64_t>(16303));
    CHECK(select_gpu_vram_budget_mib({rtx, rx7900xt}, "ROCm0") == std::optional<std::uint64_t>(20480));
}

TEST_CASE("enumerate_gpu_devices returns well-formed entries without crashing") {
    const auto gpus = enumerate_gpu_devices();
    for (const auto& gpu : gpus) {
        CHECK_FALSE(gpu.id.empty());
        CHECK_FALSE(gpu.source.empty());
        CHECK(std::string(to_string(gpu.vendor)).size() > 0);
    }
}
