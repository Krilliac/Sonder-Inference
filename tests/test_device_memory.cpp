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

    // Integrated APU with 2 GiB BIOS carveout + 16 GiB GTT is still detected as UMA.
    const auto apu_2g_files = fake_files({
        {"/sys/class/drm/card1/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card1/device/device", "0x15bf\n"},
        {"/sys/class/drm/card1/device/mem_info_vram_total", std::to_string(2 * kGiB) + "\n"},
        {"/sys/class/drm/card1/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
    });
    const auto apu_2g = parse_amdgpu_sysfs_device("card1", apu_2g_files);
    REQUIRE(apu_2g.has_value());
    CHECK(apu_2g->is_integrated_uma);
    CHECK(gpu_effective_vram_mib(*apu_2g, false) == std::optional<std::uint64_t>(2048 + 16384));

    // 4 GiB discrete AMD Radeon RX 6500 XT on a 32 GB host (16 GiB GTT) must NOT be misclassified as UMA.
    const auto rx6500xt_files = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/device", "0x743f\n"},
        {"/sys/class/drm/card0/device/product_name", "AMD Radeon RX 6500 XT\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(4 * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_used", std::to_string(512 * kMiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_gtt_used", std::to_string(256 * kMiB) + "\n"},
    });
    const auto rx6500xt = parse_amdgpu_sysfs_device("card0", rx6500xt_files);
    REQUIRE(rx6500xt.has_value());
    CHECK_FALSE(rx6500xt->is_integrated_uma);
    CHECK(rx6500xt->dedicated_total == MemoryReading::known(4 * kGiB));
    CHECK(rx6500xt->dedicated_free == MemoryReading::known(4 * kGiB - 512 * kMiB));
    CHECK(gpu_effective_vram_mib(*rx6500xt, false) == std::optional<std::uint64_t>(4096));
    CHECK(gpu_effective_vram_mib(*rx6500xt, true) == std::optional<std::uint64_t>(3584));

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

TEST_CASE("ROCm/KFD availability, visible-device filtering, and Windows AMD backend hints") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // 1. Linux amdgpu sysfs: without /dev/kfd (rocm_available=false), AMD APU and RX XT hint "vulkan".
    const auto apu_files = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/device", "0x15bf\n"},
        {"/sys/class/drm/card0/device/product_name", "AMD Radeon 780M Graphics\n"},
        {"/sys/class/drm/card0/device/uevent", "DRIVER=amdgpu\nPCI_ID=1002:15BF\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(512 * kMiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_gtt_total", std::to_string(16 * kGiB) + "\n"},
    });
    const auto apu_with_kfd = parse_amdgpu_sysfs_device("card0", apu_files, true);
    const auto apu_no_kfd = parse_amdgpu_sysfs_device("card0", apu_files, false);
    REQUIRE(apu_with_kfd.has_value());
    REQUIRE(apu_no_kfd.has_value());
    CHECK(apu_with_kfd->backend_hint == "rocm");
    CHECK(apu_no_kfd->backend_hint == "vulkan");
    CHECK(apu_no_kfd->is_integrated_uma);

    // 2. HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES ordinal filtering and composition.
    CHECK(is_rocm_device_visible(0));
    CHECK(is_rocm_device_visible(3));
    CHECK_FALSE(is_rocm_device_visible(0, std::string_view("")));
    CHECK_FALSE(is_rocm_device_visible(0, std::string_view("-1")));
    CHECK_FALSE(is_rocm_device_visible(0, std::string_view("none")));
    CHECK_FALSE(is_rocm_device_visible(0, std::nullopt, std::string_view("NoDevFiles")));
    CHECK(is_rocm_device_visible(0, std::string_view("0")));
    CHECK_FALSE(is_rocm_device_visible(1, std::string_view("0")));
    CHECK_FALSE(is_rocm_device_visible(0, std::string_view("1")));
    CHECK(is_rocm_device_visible(1, std::string_view("1")));
    // Negative entry terminates HIP visible list ("0,-1,1" exposes only ordinal 0).
    CHECK(is_rocm_device_visible(0, std::string_view("0,-1,1")));
    CHECK_FALSE(is_rocm_device_visible(1, std::string_view("0,-1,1")));
    // ROCR filters physical ordinals first; HIP indexes into ROCR's visible list.
    CHECK_FALSE(is_rocm_device_visible(0, std::string_view("1"), std::string_view("1,2")));
    CHECK_FALSE(is_rocm_device_visible(1, std::string_view("1"), std::string_view("1,2")));
    CHECK(is_rocm_device_visible(2, std::string_view("1"), std::string_view("1,2")));
    // Reordered ROCR_VISIBLE_DEVICES preserves mapped runtime ordinal: card 1 -> ROCm0, card 0 -> ROCm1.
    CHECK(map_rocm_device_ordinal(1, std::nullopt, std::string_view("1,0")) == std::optional<std::size_t>(0));
    CHECK(map_rocm_device_ordinal(0, std::nullopt, std::string_view("1,0")) == std::optional<std::size_t>(1));
    // ROCr UUID tokens fail closed without a matching device UUID and match only the named card.
    CHECK_FALSE(is_rocm_device_visible(0, std::nullopt, std::string_view("GPU-0123456789abcdef")));
    CHECK_FALSE(is_rocm_device_visible(1, std::nullopt, std::string_view("GPU-0123456789abcdef"),
                                       std::string_view("GPU-fedcba9876543210")));
    CHECK(map_rocm_device_ordinal(1, std::nullopt, std::string_view("GPU-0123456789abcdef"),
                                  std::string_view("GPU-0123456789abcdef")) == std::optional<std::size_t>(0));

    // Windows mixed WDDM + TCC NVIDIA enumeration merges TCC adapters without duplicating WDDM cards.
    {
        GpuDeviceInfo dxgi_wddm;
        dxgi_wddm.id = "gpu:0";
        dxgi_wddm.name = "NVIDIA GeForce RTX 5070 Ti";
        dxgi_wddm.vendor = GpuVendor::nvidia;
        dxgi_wddm.vendor_id = 0x10DEu;
        dxgi_wddm.device_id = 0x2C05u;
        dxgi_wddm.source = "dxgi";

        GpuDeviceInfo nvml_wddm = dxgi_wddm;
        nvml_wddm.source = "nvml";
        nvml_wddm.pci_bus_id = "0000:01:00.0";
        nvml_wddm.is_tcc_mode = false;

        GpuDeviceInfo nvml_tcc;
        nvml_tcc.name = "NVIDIA L40S";
        nvml_tcc.vendor = GpuVendor::nvidia;
        nvml_tcc.vendor_id = 0x10DEu;
        nvml_tcc.device_id = 0x26B9u;
        nvml_tcc.pci_bus_id = "0000:02:00.0";
        nvml_tcc.is_tcc_mode = true;
        nvml_tcc.backend_hint = "cuda";
        nvml_tcc.source = "nvml";

        std::vector<GpuDeviceInfo> merged{dxgi_wddm};
        merge_win32_nvml_devices(merged, {nvml_wddm, nvml_tcc});
        REQUIRE(merged.size() == 2u);
        CHECK(merged[0].source == "dxgi");
        CHECK(merged[0].pci_bus_id == "0000:01:00.0");
        CHECK(merged[1].name == "NVIDIA L40S");
        CHECK(merged[1].is_tcc_mode);
    }

    // 3. Windows GPU backend hint matrix (NVIDIA, AMD discrete w/ and w/o HIP SDK, AMD Ryzen iGPU UMA).
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::nvidia, false, false) == "cuda");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::amd, false, false, 0) == "vulkan");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::amd, false, true, 0) == "rocm");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::amd, false, true, 0, std::string_view("1")) == "vulkan");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::amd, false, true, 1, std::string_view("1")) == "rocm");
    // Integrated UMA Ryzen iGPU on Windows hints "vulkan" even when HIP SDK is installed.
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::amd, true, true, 0) == "vulkan");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::intel, true, false) == "vulkan");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::apple, true, false) == "metal");
    CHECK(resolve_windows_gpu_backend_hint(GpuVendor::unknown, false, false).empty());

    // 4. Hybrid AMD host (gpu:0 = non-ROCm Ryzen iGPU "vulkan", gpu:1 = discrete RX 9070 XT "rocm"):
    // "ROCm0" / "HIP0" budget selection targets the ROCm-capable discrete RX 9070 XT, not gpu:0!
    GpuDeviceInfo ryzen_igpu;
    ryzen_igpu.id = "gpu:0";
    ryzen_igpu.name = "AMD Radeon(TM) Graphics";
    ryzen_igpu.vendor = GpuVendor::amd;
    ryzen_igpu.vendor_id = 0x1002u;
    ryzen_igpu.dedicated_total = MemoryReading::known(512 * kMiB);
    ryzen_igpu.shared_total = MemoryReading::known(16 * kGiB);
    ryzen_igpu.is_integrated_uma = true;
    ryzen_igpu.backend_hint = "vulkan";
    ryzen_igpu.source = "dxgi";

    GpuDeviceInfo rx9070xt;
    rx9070xt.id = "gpu:1";
    rx9070xt.name = "AMD Radeon RX 9070 XT";
    rx9070xt.vendor = GpuVendor::amd;
    rx9070xt.vendor_id = 0x1002u;
    rx9070xt.dedicated_total = MemoryReading::known(16 * kGiB);
    rx9070xt.dedicated_free = MemoryReading::known(15 * kGiB);
    rx9070xt.shared_total = MemoryReading::known(16 * kGiB);
    rx9070xt.is_integrated_uma = false;
    rx9070xt.backend_hint = "rocm";
    rx9070xt.source = "dxgi";

    const std::vector<GpuDeviceInfo> hybrid_amd{ryzen_igpu, rx9070xt};
    CHECK(select_gpu_vram_budget_mib(hybrid_amd) == std::optional<std::uint64_t>(16384));
    CHECK(select_gpu_vram_budget_mib(hybrid_amd, "ROCm0") == std::optional<std::uint64_t>(16384));
    CHECK(select_gpu_vram_budget_mib(hybrid_amd, "HIP0", true) == std::optional<std::uint64_t>(15360));
    CHECK_FALSE(select_gpu_vram_budget_mib(hybrid_amd, "ROCm1").has_value());
    CHECK(select_gpu_vram_budget_mib(hybrid_amd, "Vulkan0") == std::optional<std::uint64_t>(512 + 16384));
    CHECK(select_gpu_vram_budget_mib(hybrid_amd, "Vulkan1") == std::optional<std::uint64_t>(16384));
}

TEST_CASE("summarize_gpu_pools and combined_gpu_vram_mib model discrete-only, UMA-only, and hybrid Vulkan pools") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    auto make_host = [&](std::uint64_t total_gib, std::uint64_t avail_gib) {
        HostMemory h;
        h.total_physical = MemoryReading::known(total_gib * kGiB);
        h.available_physical = MemoryReading::known(avail_gib * kGiB);
        h.source = "test";
        return h;
    };

    GpuDeviceInfo rx9060xt;
    rx9060xt.id = "gpu:0";
    rx9060xt.name = "AMD Radeon RX 9060 XT";
    rx9060xt.vendor = GpuVendor::amd;
    rx9060xt.vendor_id = 0x1002u;
    rx9060xt.device_id = 0x7590u;
    rx9060xt.dedicated_total = MemoryReading::known(16288ull * kMiB);  // ~15.9 GiB
    rx9060xt.dedicated_free = MemoryReading::known(15900ull * kMiB);
    rx9060xt.shared_total = MemoryReading::known(24ull * kGiB);
    rx9060xt.is_integrated_uma = false;
    rx9060xt.backend_hint = "vulkan";
    rx9060xt.source = "dxgi";

    GpuDeviceInfo radeon780m;
    radeon780m.id = "gpu:1";
    radeon780m.name = "AMD Radeon 780M Graphics";
    radeon780m.vendor = GpuVendor::amd;
    radeon780m.vendor_id = 0x1002u;
    radeon780m.device_id = 0x15BFu;
    radeon780m.dedicated_total = MemoryReading::known(512ull * kMiB);
    radeon780m.shared_total = MemoryReading::known(32051ull * kMiB);   // total 32563 MiB (~31.8 GiB)
    radeon780m.dedicated_free = MemoryReading::known(31800ull * kMiB);
    radeon780m.is_integrated_uma = true;
    radeon780m.backend_hint = "vulkan";
    radeon780m.source = "dxgi";

    // 1. Empty GPU list.
    {
        const GpuPoolSummary empty = summarize_gpu_pools({}, make_host(48, 44));
        CHECK(empty.discrete_vram_mib == 0u);
        CHECK(empty.integrated_uma_vram_mib == 0u);
        CHECK(empty.combined_gpu_vram_mib == 0u);
        CHECK(empty.combined_free_gpu_vram_mib == 0u);
        CHECK(empty.discrete_gpu_count == 0u);
        CHECK(empty.integrated_uma_gpu_count == 0u);
        CHECK_FALSE(empty.hybrid_vulkan_pool_available);
        CHECK(empty.hybrid_vulkan_devices.empty());
        CHECK(combined_gpu_vram_mib({}, false, make_host(48, 44)) == 0u);
    }

    // 2. Discrete-only host (no integrated UMA).
    {
        const GpuPoolSummary disc = summarize_gpu_pools({rx9060xt}, make_host(48, 44));
        CHECK(disc.discrete_vram_mib == 16288u);
        CHECK(disc.discrete_free_vram_mib == 15900u);
        CHECK(disc.integrated_uma_vram_mib == 0u);
        CHECK(disc.combined_gpu_vram_mib == 16288u);
        CHECK(disc.combined_free_gpu_vram_mib == 15900u);
        CHECK(disc.discrete_gpu_count == 1u);
        CHECK(disc.integrated_uma_gpu_count == 0u);
        CHECK_FALSE(disc.hybrid_vulkan_pool_available);
        CHECK(disc.hybrid_vulkan_devices.empty());
        CHECK(combined_gpu_vram_mib({rx9060xt}, false, make_host(48, 44)) == 16288u);
        CHECK(combined_gpu_vram_mib({rx9060xt}, true, make_host(48, 44)) == 15900u);
    }

    // 3. Integrated UMA-only host.
    {
        const GpuPoolSummary uma = summarize_gpu_pools({radeon780m}, make_host(48, 44));
        CHECK(uma.discrete_vram_mib == 0u);
        CHECK(uma.integrated_uma_vram_mib == 32563u);
        CHECK(uma.integrated_uma_free_vram_mib == 31800u);
        CHECK(uma.combined_gpu_vram_mib == 32563u);
        CHECK(uma.combined_free_gpu_vram_mib == 31800u);
        CHECK(uma.discrete_gpu_count == 0u);
        CHECK(uma.integrated_uma_gpu_count == 1u);
        CHECK_FALSE(uma.hybrid_vulkan_pool_available);
        CHECK(uma.hybrid_vulkan_devices.empty());
    }

    // 4. Node1 hybrid discrete RX 9060 XT (Vulkan0) + integrated Radeon 780M (Vulkan1, UMA).
    {
        const std::vector<GpuDeviceInfo> node1{rx9060xt, radeon780m};
        const HostMemory host48 = make_host(48, 44);
        const GpuPoolSummary pool = summarize_gpu_pools(node1, host48);
        CHECK(pool.discrete_vram_mib == 16288u);
        CHECK(pool.integrated_uma_vram_mib == 32563u);
        CHECK(pool.combined_gpu_vram_mib == 16288u + 32563u);
        CHECK(pool.discrete_free_vram_mib == 15900u);
        CHECK(pool.integrated_uma_free_vram_mib == 31800u);
        CHECK(pool.combined_free_gpu_vram_mib == 15900u + 31800u);
        CHECK(pool.discrete_gpu_count == 1u);
        CHECK(pool.integrated_uma_gpu_count == 1u);
        CHECK(pool.hybrid_vulkan_pool_available);
        CHECK(pool.hybrid_vulkan_devices == std::vector<std::string>{"Vulkan0", "Vulkan1"});
        CHECK(combined_gpu_vram_mib(node1, false, host48) == 16288u + 32563u);
        CHECK(combined_gpu_vram_mib(node1, true, host48) == 15900u + 31800u);
    }

    // 5. Reversed adapter order places discrete GPU first in hybrid_vulkan_devices.
    {
        GpuDeviceInfo igpu0 = radeon780m;
        igpu0.id = "gpu:0";
        GpuDeviceInfo dgpu1 = rx9060xt;
        dgpu1.id = "gpu:1";
        const GpuPoolSummary pool = summarize_gpu_pools({igpu0, dgpu1}, make_host(48, 44));
        CHECK(pool.hybrid_vulkan_pool_available);
        CHECK(pool.hybrid_vulkan_devices == std::vector<std::string>{"Vulkan1", "Vulkan0"});
    }

    // 6. Tight host RAM caps UMA contribution in combined_gpu_vram_mib (both total and free).
    {
        const std::vector<GpuDeviceInfo> node1{rx9060xt, radeon780m};
        const HostMemory tight_host = make_host(24, 16);
        const GpuPoolSummary capped = summarize_gpu_pools(node1, tight_host);
        CHECK(capped.discrete_vram_mib == 16288u);
        CHECK(capped.integrated_uma_vram_mib == 24u * 1024u);
        CHECK(capped.combined_gpu_vram_mib == 16288u + 24u * 1024u);
        CHECK(capped.integrated_uma_free_vram_mib == 16u * 1024u);
        CHECK(capped.combined_free_gpu_vram_mib == 15900u + 16u * 1024u);
        CHECK(combined_gpu_vram_mib(node1, false, tight_host) == 16288u + 24u * 1024u);
        CHECK(combined_gpu_vram_mib(node1, true, tight_host) == 15900u + 16u * 1024u);
    }

    // 7. TCC-mode discrete GPU and Apple Metal GPU are excluded from Vulkan hybrid spread.
    {
        GpuDeviceInfo tcc_gpu = rx9060xt;
        tcc_gpu.vendor = GpuVendor::nvidia;
        tcc_gpu.vendor_id = 0x10DEu;
        tcc_gpu.is_tcc_mode = true;
        tcc_gpu.backend_hint = "cuda";
        const GpuPoolSummary tcc_pool = summarize_gpu_pools({tcc_gpu, radeon780m}, make_host(48, 44));
        CHECK_FALSE(tcc_pool.hybrid_vulkan_pool_available);
        CHECK(tcc_pool.hybrid_vulkan_devices.empty());
    }
}

TEST_CASE("shared match_single_gpu_hint and resolve_uma_selection unify GPU-hint and UMA classification") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;

    GpuDeviceInfo rx9060xt;
    rx9060xt.id = "gpu:0";
    rx9060xt.name = "AMD Radeon RX 9060 XT";
    rx9060xt.vendor = GpuVendor::amd;
    rx9060xt.dedicated_total = MemoryReading::known(16384ull * kMiB);
    rx9060xt.is_integrated_uma = false;
    rx9060xt.backend_hint = "rocm";
    rx9060xt.rocm_runtime_ordinal = 0;

    GpuDeviceInfo radeon780m;
    radeon780m.id = "gpu:1";
    radeon780m.name = "AMD Radeon 780M Graphics";
    radeon780m.vendor = GpuVendor::amd;
    radeon780m.is_integrated_uma = true;
    radeon780m.backend_hint = "vulkan";

    const std::vector<GpuDeviceInfo> hybrid{rx9060xt, radeon780m};

    CHECK(match_single_gpu_hint(hybrid, "uma") == std::optional<std::size_t>(1));
    CHECK(match_single_gpu_hint(hybrid, "igpu") == std::optional<std::size_t>(1));
    CHECK(match_single_gpu_hint(hybrid, "apu") == std::optional<std::size_t>(1));
    CHECK(match_single_gpu_hint(hybrid, "rocm:0") == std::optional<std::size_t>(0));
    CHECK(match_single_gpu_hint(hybrid, "Vulkan0") == std::optional<std::size_t>(0));
    CHECK(match_single_gpu_hint(hybrid, "Vulkan1") == std::optional<std::size_t>(1));
    CHECK_FALSE(match_single_gpu_hint(hybrid, "bogus").has_value());
    // require_known_vram=true skips unknown-VRAM GPUs (used by select_gpu_vram_budget_mib).
    CHECK_FALSE(match_single_gpu_hint(hybrid, "uma", false, true).has_value());

    CHECK(normalize_uma_mode("on") == std::optional<std::string_view>("on"));
    CHECK(normalize_uma_mode("true") == std::optional<std::string_view>("on"));
    CHECK(normalize_uma_mode("off") == std::optional<std::string_view>("off"));
    CHECK(normalize_uma_mode("false") == std::optional<std::string_view>("off"));
    CHECK(normalize_uma_mode("auto") == std::optional<std::string_view>("auto"));
    CHECK_FALSE(normalize_uma_mode("maybe").has_value());
    CHECK(is_valid_uma_mode("auto"));
    CHECK_FALSE(is_valid_uma_mode("invalid"));

    CHECK(extract_device_hint({"--port", "8080", "--device", "Vulkan0", "-dev=Vulkan0,Vulkan1"}) ==
          "Vulkan0,Vulkan1");
    CHECK(resolve_uma_selection("auto", hybrid) == UmaSelectionMode::discrete_only);
    CHECK(resolve_uma_selection("auto", hybrid, "Vulkan0") == UmaSelectionMode::discrete_only);
    CHECK(resolve_uma_selection("auto", hybrid, "Vulkan1") == UmaSelectionMode::uma_only);
    CHECK(resolve_uma_selection("auto", hybrid, "Vulkan0,Vulkan1") == UmaSelectionMode::hybrid);
    CHECK(resolve_uma_selection("auto", hybrid, "Vulkan0,bogus") == UmaSelectionMode::discrete_only);
    CHECK(resolve_uma_mode("auto", hybrid, "Vulkan0,Vulkan1"));
    CHECK_FALSE(resolve_uma_mode("auto", hybrid, "Vulkan0"));
}

TEST_CASE("NAT-08: parse_ggml_device_line parses v1.0.0 with rebar, pre-b12171 without rebar, and rejects bad lines") {
    // 1. v1.0.0 / b12171+ discrete RX 9060 XT line with rebar: 1 and KHR_coopmat.
    const auto rx = parse_ggml_device_line(
        "ggml_vulkan: 0 = AMD Radeon RX 9060 XT (AMD proprietary driver) | uma: 0 | rebar: 1 | "
        "fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | "
        "matrix cores: KHR_coopmat\n");
    REQUIRE(rx.has_value());
    CHECK(rx->backend == "vulkan");
    CHECK(rx->index == std::optional<std::uint32_t>(0u));
    CHECK(rx->device_name == "AMD Radeon RX 9060 XT");
    CHECK(rx->driver_name == "AMD proprietary driver");
    CHECK(rx->uma == std::optional<bool>(false));
    CHECK(rx->rebar == std::optional<bool>(true));
    CHECK(rx->fp16 == "1");
    CHECK(rx->bf16 == std::optional<bool>(true));
    CHECK(rx->fp4 == std::optional<bool>(false));
    CHECK(rx->warp_size == std::optional<std::uint32_t>(64u));
    CHECK(rx->shared_memory_bytes == std::optional<std::uint64_t>(32768ull));
    CHECK(rx->int_dot == std::optional<bool>(true));
    CHECK(rx->matrix_cores == "KHR_coopmat");

    // 2. Pre-b12171 line without `rebar:` leaves rebar unset (nullopt), never false.
    const auto pre_b12171 = parse_ggml_device_line(
        "ggml_vulkan: 1 = AMD Radeon(TM) 780M Graphics (AMD proprietary driver) | uma: 1 | "
        "fp16: dot2 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | "
        "matrix cores: none\r\n");
    REQUIRE(pre_b12171.has_value());
    CHECK(pre_b12171->index == std::optional<std::uint32_t>(1u));
    CHECK(pre_b12171->device_name == "AMD Radeon(TM) 780M Graphics");
    CHECK(pre_b12171->driver_name == "AMD proprietary driver");
    CHECK(pre_b12171->uma == std::optional<bool>(true));
    CHECK_FALSE(pre_b12171->rebar.has_value());
    CHECK(pre_b12171->fp16 == "dot2");
    CHECK(pre_b12171->bf16 == std::optional<bool>(false));
    CHECK(pre_b12171->warp_size == std::optional<std::uint32_t>(32u));
    CHECK(pre_b12171->shared_memory_bytes == std::optional<std::uint64_t>(32768ull));
    CHECK(pre_b12171->matrix_cores == "none");

    // 3. NVIDIA NV_coopmat2v with rebar: 0 and extra future fields is accepted.
    const auto nv = parse_ggml_device_line(
        "ggml_vulkan: 0 = NVIDIA GeForce RTX 5070 Ti (NVIDIA) | uma: 0 | rebar: 0 | "
        "fp16: 1 | bf16: 1 | fp4: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | "
        "matrix cores: NV_coopmat2v | future_flag: 42");
    REQUIRE(nv.has_value());
    CHECK(nv->rebar == std::optional<bool>(false));
    CHECK(nv->fp4 == std::optional<bool>(true));
    CHECK(nv->shared_memory_bytes == std::optional<std::uint64_t>(49152ull));
    CHECK(nv->matrix_cores == "NV_coopmat2v");

    // 4. apply_ggml_device_telemetry merges onto GpuDeviceInfo cleanly.
    GpuDeviceInfo gpu;
    gpu.id = "gpu:0";
    gpu.name = "AMD Radeon RX 9060 XT";
    apply_ggml_device_telemetry(gpu, *rx);
    REQUIRE(gpu.ggml_device.has_value());
    CHECK(gpu.driver_name == "AMD proprietary driver");
    CHECK(gpu.rebar == std::optional<bool>(true));
    CHECK(gpu.shared_memory_bytes == std::optional<std::uint64_t>(32768ull));
    CHECK(gpu.warp_size == std::optional<std::uint32_t>(64u));
    CHECK(gpu.matrix_cores == "KHR_coopmat");

    // 5. Rejects unrelated, empty, malformed, negative, or non-device ggml_vulkan lines.
    CHECK_FALSE(parse_ggml_device_line("").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: Found 2 Vulkan devices:").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: Warning: Device type is CPU.").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: -1 = Bad | uma: 0 | shared memory: 32768").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: 0 = | uma: 0").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: 0 = Foo (Bar) | uma: 2").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: 0 = Foo (Bar) | shared memory: -5").has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: 0 = Foo (Bar) | warp size: 99999999999999").has_value());
    CHECK_FALSE(parse_ggml_device_line(std::string(5000u, 'a')).has_value());
    CHECK_FALSE(parse_ggml_device_line(std::string_view("ggml_vulkan: 0 = Foo\0Bar | uma: 0", 31)).has_value());
    CHECK_FALSE(parse_ggml_device_line("ggml_vulkan: 0 = Foo (Bar) | unknown_only: 1").has_value());
}

TEST_CASE("NAT-08: Vulkan heap ReBAR, memory budget, PCIe speed, ASPM, and driver build helpers") {
    constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;

    // 1. Per-heap VK_EXT_memory_budget and ReBAR classification.
    VulkanMemoryHeapInfo vram_rebar;
    vram_rebar.heap_index = 0;
    vram_rebar.size_bytes = 16ull * kGiB;
    vram_rebar.budget_bytes = MemoryReading::known(15ull * kGiB);
    vram_rebar.usage_bytes = MemoryReading::known(1ull * kGiB);
    vram_rebar.device_local = true;
    vram_rebar.host_visible = true;
    vram_rebar.host_coherent = true;

    VulkanMemoryHeapInfo host_heap;
    host_heap.heap_index = 1;
    host_heap.size_bytes = 24ull * kGiB;
    host_heap.budget_bytes = MemoryReading::known(20ull * kGiB);
    host_heap.usage_bytes = MemoryReading::known(2ull * kGiB);
    host_heap.device_local = false;
    host_heap.host_visible = true;
    host_heap.host_coherent = false;

    CHECK(classify_vulkan_rebar({vram_rebar, host_heap}) == std::optional<bool>(true));
    CHECK(vulkan_device_local_budget_bytes({vram_rebar, host_heap}) == MemoryReading::known(15ull * kGiB));
    CHECK(vulkan_device_local_usage_bytes({vram_rebar, host_heap}) == MemoryReading::known(1ull * kGiB));

    VulkanMemoryHeapInfo vram_no_rebar = vram_rebar;
    vram_no_rebar.host_visible = false;
    vram_no_rebar.host_coherent = false;
    vram_no_rebar.budget_bytes = MemoryReading::unknown();
    vram_no_rebar.usage_bytes = MemoryReading::unknown();
    CHECK(classify_vulkan_rebar({vram_no_rebar, host_heap}) == std::optional<bool>(false));
    CHECK_FALSE(vulkan_device_local_budget_bytes({vram_no_rebar, host_heap}).is_known());
    CHECK_FALSE(vulkan_device_local_usage_bytes({vram_no_rebar, host_heap}).is_known());
    CHECK_FALSE(classify_vulkan_rebar({}).has_value());
    CHECK_FALSE(classify_vulkan_rebar({host_heap}).has_value());

    // 2. PCIe link speed string -> generation mapping.
    CHECK(parse_pcie_link_speed_gen("2.5 GT/s PCIe\n") == std::optional<std::uint32_t>(1u));
    CHECK(parse_pcie_link_speed_gen("5.0 GT/s PCIe") == std::optional<std::uint32_t>(2u));
    CHECK(parse_pcie_link_speed_gen("8.0 GT/s PCIe") == std::optional<std::uint32_t>(3u));
    CHECK(parse_pcie_link_speed_gen("16.0 GT/s PCIe") == std::optional<std::uint32_t>(4u));
    CHECK(parse_pcie_link_speed_gen("32.0 GT/s PCIe") == std::optional<std::uint32_t>(5u));
    CHECK(parse_pcie_link_speed_gen("64.0 GT/s PCIe") == std::optional<std::uint32_t>(6u));
    CHECK_FALSE(parse_pcie_link_speed_gen("Unknown").has_value());
    CHECK_FALSE(parse_pcie_link_speed_gen("").has_value());
    CHECK_FALSE(parse_pcie_link_speed_gen("-16.0 GT/s").has_value());

    // 3. Linux ASPM policy & per-link state + Windows ASPM policy mapping.
    CHECK(parse_linux_aspm_policy("[default] performance powersave powersupersave\n") == "default");
    CHECK(parse_linux_aspm_policy("default [performance] powersave powersupersave\n") == "performance");
    CHECK(parse_linux_aspm_policy("default performance [powersave] powersupersave") == "powersave");
    CHECK(parse_linux_aspm_policy("no brackets here").empty());
    CHECK(parse_linux_aspm_policy("[]").empty());

    CHECK(format_linux_link_aspm(false, false, false, false) == "off");
    CHECK(format_linux_link_aspm(true, false) == "l0s");
    CHECK(format_linux_link_aspm(false, true) == "l1");
    CHECK(format_linux_link_aspm(true, true) == "l0s+l1");
    CHECK(format_linux_link_aspm(true, true, true, true) == "l0s+l1+l1.1+l1.2");
    CHECK(format_linux_link_aspm(std::nullopt, std::nullopt).empty());

    CHECK(windows_aspm_policy_name(0u) == "off");
    CHECK(windows_aspm_policy_name(1u) == "moderate");
    CHECK(windows_aspm_policy_name(2u) == "maximum");
    CHECK(windows_aspm_policy_name(3u).empty());

    // 4. Driver build string sanitization and formatting.
    CHECK(sanitize_driver_build_string("  32.0.21025.1024\r\n") == "32.0.21025.1024");
    CHECK(sanitize_driver_build_string("25.10.2\x01\x02-ok") == "25.10.2-ok");
    CHECK(format_driver_build_string("32.0.21025.1024", "25.10.2", "") == "25.10.2 (32.0.21025.1024)");
    CHECK(format_driver_build_string("32.0.21025.1024", "", "") == "32.0.21025.1024");
    CHECK(format_driver_build_string("", "", "Mesa 25.1.0") == "Mesa 25.1.0");
    CHECK(format_driver_build_string("", "", "").empty());

    // 5. Linux amdgpu sysfs parser extracts PCIe gen/width, ASPM, pci_bus_id, and driver_build.
    const auto sysfs = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/device", "0x7590\n"},
        {"/sys/class/drm/card0/device/product_name", "AMD Radeon RX 9060 XT\n"},
        {"/sys/class/drm/card0/device/uevent",
         "DRIVER=amdgpu\nPCI_ID=1002:7590\nPCI_SLOT_NAME=0000:03:00.0\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(16ull * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_used", std::to_string(1ull * kGiB) + "\n"},
        {"/sys/class/drm/card0/device/current_link_speed", "32.0 GT/s PCIe\n"},
        {"/sys/class/drm/card0/device/current_link_width", "16\n"},
        {"/sys/class/drm/card0/device/max_link_speed", "32.0 GT/s PCIe\n"},
        {"/sys/class/drm/card0/device/max_link_width", "16\n"},
        {"/sys/class/drm/card0/device/link/l0s_aspm", "0\n"},
        {"/sys/class/drm/card0/device/link/l1_aspm", "1\n"},
        {"/sys/module/amdgpu/version", "6.10.5.60204-2057491\n"},
    });
    const auto parsed = parse_amdgpu_sysfs_device("card0", sysfs);
    REQUIRE(parsed.has_value());
    CHECK(parsed->pci_bus_id == "0000:03:00.0");
    CHECK(parsed->driver_name == "amdgpu");
    CHECK(parsed->driver_build == "6.10.5.60204-2057491");
    CHECK(parsed->pcie.current_gen == std::optional<std::uint32_t>(5u));
    CHECK(parsed->pcie.current_width == std::optional<std::uint32_t>(16u));
    CHECK(parsed->pcie.max_gen == std::optional<std::uint32_t>(5u));
    CHECK(parsed->pcie.max_width == std::optional<std::uint32_t>(16u));
    CHECK(parsed->pcie.aspm_state == "l1");
    CHECK(parsed->pcie.aspm_source == "linux_sysfs_link");

    // In-tree mainline amdgpu (no /sys/module/amdgpu/version) leaves driver_build empty
    // so Vulkan driverInfo can populate it; global pcie_aspm policy is used as fallback.
    const auto sysfs_global_aspm = fake_files({
        {"/sys/class/drm/card0/device/vendor", "0x1002\n"},
        {"/sys/class/drm/card0/device/mem_info_vram_total", std::to_string(16ull * kGiB) + "\n"},
        {"/sys/module/pcie_aspm/parameters/policy", "default [performance] powersave powersupersave\n"},
    });
    const auto parsed_global = parse_amdgpu_sysfs_device("card0", sysfs_global_aspm);
    REQUIRE(parsed_global.has_value());
    CHECK(parsed_global->driver_build.empty());
    CHECK(parsed_global->pcie.aspm_state == "performance");
    CHECK(parsed_global->pcie.aspm_source == "linux_pcie_aspm_policy");

    // 6. Core ggml device registry merges captured lines into GPU inventory end-to-end.
    clear_registered_ggml_device_telemetry();
    const auto g0 = parse_ggml_device_line(
        "ggml_vulkan: 0 = AMD Radeon RX 9060 XT (AMD proprietary driver) | uma: 0 | rebar: 1 | "
        "fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | "
        "matrix cores: KHR_coopmat");
    const auto g1 = parse_ggml_device_line(
        "ggml_vulkan: 1 = AMD Radeon 780M Graphics (AMD proprietary driver) | uma: 1 | "
        "fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | "
        "matrix cores: none");
    REQUIRE(g0.has_value());
    REQUIRE(g1.has_value());
    register_ggml_device_telemetry(*g0);
    register_ggml_device_telemetry(*g1);
    CHECK(registered_ggml_device_telemetry().size() == 2u);

    GpuDeviceInfo d_rx;
    d_rx.id = "gpu:0";
    d_rx.name = "AMD Radeon RX 9060 XT";
    d_rx.is_integrated_uma = false;
    GpuDeviceInfo d_igpu;
    d_igpu.id = "gpu:1";
    d_igpu.name = "AMD Radeon 780M Graphics";
    d_igpu.is_integrated_uma = true;
    std::vector<GpuDeviceInfo> inv{d_rx, d_igpu};
    merge_ggml_device_telemetry(inv, registered_ggml_device_telemetry());
    REQUIRE(inv[0].ggml_device.has_value());
    CHECK(inv[0].ggml_device->matrix_cores == "KHR_coopmat");
    CHECK(inv[0].rebar == std::optional<bool>(true));
    REQUIRE(inv[1].ggml_device.has_value());
    CHECK(inv[1].ggml_device->matrix_cores == "none");
    CHECK_FALSE(inv[1].ggml_device->rebar.has_value());

    // Live enumerate_gpu_devices() also merges any matching registered ggml line.
    const auto live_before = enumerate_gpu_devices();
    clear_registered_ggml_device_telemetry();
    if (!live_before.empty()) {
        GgmlDeviceTelemetry live_entry;
        live_entry.backend = "vulkan";
        live_entry.index = 0u;
        live_entry.device_name = live_before[0].name;
        live_entry.driver_name = "test-driver";
        live_entry.uma = live_before[0].is_integrated_uma;
        live_entry.rebar = true;
        live_entry.shared_memory_bytes = 32768ull;
        live_entry.matrix_cores = "KHR_coopmat";
        register_ggml_device_telemetry(live_entry);
        const auto live_after = enumerate_gpu_devices();
        clear_registered_ggml_device_telemetry();
        REQUIRE_FALSE(live_after.empty());
        REQUIRE(live_after[0].ggml_device.has_value());
        CHECK(live_after[0].ggml_device->driver_name == "test-driver");
    }

    // 7. PCI BDF normalization matches 8-digit NVML domains ("00000000:01:00.0") with 4-digit ("0000:01:00.0").
    {
        GpuDeviceInfo dxgi_nv;
        dxgi_nv.vendor = GpuVendor::nvidia;
        dxgi_nv.pci_bus_id = "0000:01:00.0";
        dxgi_nv.source = "dxgi";
        GpuDeviceInfo nvml_nv;
        nvml_nv.vendor = GpuVendor::nvidia;
        nvml_nv.pci_bus_id = "00000000:01:00.0";
        nvml_nv.driver_build = "610.88";
        nvml_nv.pcie.current_gen = 5u;
        nvml_nv.pcie.current_width = 16u;
        nvml_nv.source = "nvml";
        std::vector<GpuDeviceInfo> merged{dxgi_nv};
        merge_win32_nvml_devices(merged, {nvml_nv});
        REQUIRE(merged.size() == 1u);
        CHECK(merged[0].driver_build == "610.88");
        CHECK(merged[0].pcie.current_gen == std::optional<std::uint32_t>(5u));
        CHECK(merged[0].pcie.current_width == std::optional<std::uint32_t>(16u));
    }
}

TEST_CASE("NAT-08: classify_vulkan_rebar mirrors ggml_vk_has_rebar first DEVICE_LOCAL memory type") {
    VulkanMemoryHeapInfo h0;
    h0.heap_index = 0u;
    h0.device_local = true;
    h0.host_coherent = false;

    VulkanMemoryHeapInfo h1;
    h1.heap_index = 1u;
    h1.device_local = true;
    h1.host_coherent = true;

    const std::vector<VulkanMemoryHeapInfo> heaps{h0, h1};
    // Without memory-type hint, falls back to first DEVICE_LOCAL heap (heap 0 -> false).
    CHECK(classify_vulkan_rebar(heaps) == std::optional<bool>(false));
    CHECK(classify_vulkan_rebar(heaps, std::nullopt) == std::optional<bool>(false));
    // With ggml_vk_has_rebar hint pointing at heap 1, classifies heap 1 -> true.
    CHECK(classify_vulkan_rebar(heaps, std::optional<std::uint32_t>(1u)) == std::optional<bool>(true));
    // Out-of-range hint falls back cleanly to first DEVICE_LOCAL heap.
    CHECK(classify_vulkan_rebar(heaps, std::optional<std::uint32_t>(99u)) == std::optional<bool>(false));
}
