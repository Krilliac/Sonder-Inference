#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

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
