// Sonder Inference: device inventory.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sonder::inference {

enum class DeviceKind { cpu, gpu, npu, remote };

const char* to_string(DeviceKind kind) noexcept;

// One memory measurement with three distinct outcomes, so callers cannot
// confuse them:
//   known, > 0      the OS reported this many bytes
//   known, == 0     the OS reported that nothing is left: refuse to admit work
//   unknown         the platform has no reading, the read failed, or the
//                   source was malformed. This is NOT zero and NOT unlimited;
//                   each caller documents its own fallback (the llama-server
//                   supervisor admits with a warning and skips its floor guard).
// Trivially copyable value type; no thread affinity.
class MemoryReading {
public:
    constexpr MemoryReading() noexcept = default;  // unknown
    [[nodiscard]] static constexpr MemoryReading unknown() noexcept { return {}; }
    [[nodiscard]] static constexpr MemoryReading known(std::uint64_t bytes) noexcept {
        MemoryReading r;
        r.known_ = true;
        r.bytes_ = bytes;
        return r;
    }
    [[nodiscard]] constexpr bool is_known() const noexcept { return known_; }
    [[nodiscard]] constexpr bool is_known_zero() const noexcept { return known_ && bytes_ == 0; }
    [[nodiscard]] constexpr std::optional<std::uint64_t> value() const noexcept {
        return known_ ? std::optional<std::uint64_t>(bytes_) : std::nullopt;
    }
    [[nodiscard]] constexpr std::uint64_t bytes_or(std::uint64_t fallback) const noexcept {
        return known_ ? bytes_ : fallback;
    }
    friend constexpr bool operator==(const MemoryReading&, const MemoryReading&) noexcept = default;

private:
    bool known_ = false;
    std::uint64_t bytes_ = 0;  // 0 while unknown
};

// A snapshot of host memory. `available_physical` is memory a new process can
// use without paging live data out: Windows ullAvailPhys; Linux MemAvailable
// capped by the tightest cgroup v2 `memory.max - memory.current` from this
// process's cgroup up to the root. `available_commit` is Windows
// ullAvailPageFile (the commit this process can still take; Windows
// documents it as the smaller of the system and per-process limit); Linux and
// macOS report it unknown. macOS reports available_physical unknown.
struct HostMemory {
    MemoryReading total_physical;
    MemoryReading available_physical;
    MemoryReading available_commit;
    std::string source;  // e.g. "GlobalMemoryStatusEx", "meminfo+cgroup2"
    std::string note;    // why a reading is unknown or capped (diagnostic text)
};

// Injectable seam. A null function means "use read_host_memory()".
using HostMemoryProvider = std::function<HostMemory()>;

// Live OS reading. Cheap (one syscall on Windows; a few small file reads on
// Linux); callable from any thread; allocates only for the diagnostic strings.
HostMemory read_host_memory();

// Linux parsing, compiled on every platform so it is testable everywhere.
// Strict: a value that is not exactly what the kernel writes yields nullopt.
//  - parse_strict_u64: decimal digits only (no sign, spaces or leading zeros),
//    at most one trailing '\n', no overflow.
//  - parse_meminfo_bytes: `Key:<spaces><digits> kB` on exactly one line,
//    returns bytes; a missing, duplicated, malformed or overflowing field is
//    nullopt.
std::optional<std::uint64_t> parse_strict_u64(std::string_view text) noexcept;
std::optional<std::uint64_t> parse_meminfo_bytes(std::string_view meminfo, std::string_view key) noexcept;

// Reads a file for the cgroup walk: nullopt when the file does not exist.
using FileReader = std::function<std::optional<std::string>(const std::string& path)>;

// Headroom of the tightest memory limit among the cgroup v2 ancestors of the
// process whose /proc/<pid>/cgroup text is `proc_cgroup`, with the cgroup
// filesystem mounted at `cgroup_root` (normally /sys/fs/cgroup).
//   known(n)  the tightest finite `memory.max - memory.current` (0 when at or
//             over the limit)
//   unknown   no cgroup v2 entry, no finite limit on any ancestor, or any
//             memory.max/memory.current that is present but malformed (the
//             `note` says which). Callers combine this with MemAvailable: an
//             `unknown` here with `limited == false` means "no cgroup limit".
struct CgroupHeadroom {
    MemoryReading headroom;
    bool limited = false;    // a finite limit was found
    bool malformed = false;  // a present file could not be parsed strictly
    std::string note;
};
CgroupHeadroom cgroup_v2_headroom(std::string_view proc_cgroup, const FileReader& read_file,
                                  std::string_view cgroup_root = "/sys/fs/cgroup");

// Linux composition of /proc/meminfo text and the cgroup headroom. Pure.
HostMemory compose_linux_host_memory(std::string_view meminfo, const CgroupHeadroom& cgroup);

struct DeviceInfo {
    std::string id;          // stable within a process, e.g. "cpu:0"
    DeviceKind kind = DeviceKind::cpu;
    std::string name;        // human readable, e.g. CPU brand string
    std::uint32_t logical_cores = 0;
    std::uint64_t total_memory_bytes = 0;      // 0 when unknown
    std::uint64_t available_memory_bytes = 0;  // 0 when unknown (or known zero: see available_physical)
    // Tri-state companions of available_memory_bytes (cpu device only).
    MemoryReading available_physical;
    MemoryReading available_commit;
};

// Host inventory. Phase 1 reports the CPU (with system RAM) only; GPU/NPU
// discovery is deferred to backend capability probing.
std::vector<DeviceInfo> enumerate_devices();

// Current OS/arch label, e.g. "windows-x86_64".
std::string host_platform();
// Host name used as the telemetry producer node_id default.
std::string host_name();

}  // namespace sonder::inference
