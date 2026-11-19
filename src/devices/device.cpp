#include "sonder/inference/device.hpp"

#include <algorithm>
#include <limits>
#include <thread>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#else
#  include <fstream>
#  include <iterator>
#  include <sstream>
#  include <unistd.h>
#endif

namespace sonder::inference {

const char* to_string(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::cpu: return "cpu";
        case DeviceKind::gpu: return "gpu";
        case DeviceKind::npu: return "npu";
        case DeviceKind::remote: return "remote";
    }
    return "unknown";
}

namespace {

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

#if defined(_WIN32)
std::string cpu_name() {
    char buffer[256] = {};
    DWORD size = sizeof(buffer);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, buffer, &size) == ERROR_SUCCESS) {
        return trim(buffer);
    }
    return "cpu";
}

HostMemory live_host_memory() {
    HostMemory out;
    out.source = "GlobalMemoryStatusEx";
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        out.note = "GlobalMemoryStatusEx failed (error " + std::to_string(GetLastError()) + ")";
        return out;
    }
    out.total_physical = MemoryReading::known(status.ullTotalPhys);
    out.available_physical = MemoryReading::known(status.ullAvailPhys);
    out.available_commit = MemoryReading::known(status.ullAvailPageFile);
    return out;
}
#elif defined(__APPLE__)
std::string cpu_name() {
    char buffer[256] = {};
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
        return trim(buffer);
    }
    return "cpu";
}

HostMemory live_host_memory() {
    HostMemory out;
    out.source = "sysctl";
    std::uint64_t mem = 0;
    size_t size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0) {
        out.total_physical = MemoryReading::known(mem);
    }
    // Available memory is not cheaply available without mach APIs: unknown.
    out.note = "available memory is not read on macOS";
    return out;
}
#else
std::string cpu_name() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                return trim(line.substr(colon + 1));
            }
        }
    }
    return "cpu";
}

// Reads at most `limit` bytes; nullopt when the file cannot be opened. A file
// longer than `limit` is returned truncated, which the strict parsers then
// refuse or ignore.
std::optional<std::string> read_small_file(const std::string& path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::string text(limit, '\0');
    in.read(text.data(), static_cast<std::streamsize>(limit));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

HostMemory live_host_memory() {
    const auto meminfo = read_small_file("/proc/meminfo", 1u << 16);
    if (!meminfo) {
        HostMemory out;
        out.source = "meminfo";
        out.note = "/proc/meminfo could not be read";
        return out;
    }
    const auto proc_cgroup = read_small_file("/proc/self/cgroup", 1u << 16);
    CgroupHeadroom cgroup;
    if (proc_cgroup) {
        cgroup = cgroup_v2_headroom(*proc_cgroup, [](const std::string& path) { return read_small_file(path, 64); });
    }
    return compose_linux_host_memory(*meminfo, cgroup);
}
#endif

}  // namespace

std::optional<std::uint64_t> parse_strict_u64(std::string_view text) noexcept {
    if (!text.empty() && text.back() == '\n') {
        text.remove_suffix(1);
    }
    if (text.empty() || text.size() > 20 || (text.size() > 1 && text.front() == '0')) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint64_t> parse_meminfo_bytes(std::string_view meminfo, std::string_view key) noexcept {
    std::optional<std::uint64_t> result;
    std::size_t pos = 0;
    while (pos < meminfo.size()) {
        const auto end = meminfo.find('\n', pos);
        const std::string_view line = meminfo.substr(pos, end == std::string_view::npos ? end : end - pos);
        pos = end == std::string_view::npos ? meminfo.size() : end + 1;
        if (line.size() <= key.size() || line.substr(0, key.size()) != key || line[key.size()] != ':') {
            continue;
        }
        if (result) {
            return std::nullopt;  // duplicated key: do not guess which one is right
        }
        std::string_view rest = line.substr(key.size() + 1);
        const auto first = rest.find_first_not_of(' ');
        if (first == std::string_view::npos || first == 0) {
            return std::nullopt;
        }
        rest.remove_prefix(first);
        if (rest.size() < 4 || rest.substr(rest.size() - 3) != " kB") {
            return std::nullopt;
        }
        const auto kib = parse_strict_u64(rest.substr(0, rest.size() - 3));
        if (!kib || *kib > std::numeric_limits<std::uint64_t>::max() / 1024) {
            return std::nullopt;
        }
        result = *kib * 1024;
    }
    return result;
}

CgroupHeadroom cgroup_v2_headroom(std::string_view proc_cgroup, const FileReader& read_file,
                                  std::string_view cgroup_root) {
    CgroupHeadroom out;
    std::optional<std::string_view> entry;
    std::size_t pos = 0;
    while (pos < proc_cgroup.size()) {
        const auto end = proc_cgroup.find('\n', pos);
        const std::string_view line = proc_cgroup.substr(pos, end == std::string_view::npos ? end : end - pos);
        pos = end == std::string_view::npos ? proc_cgroup.size() : end + 1;
        if (line.substr(0, 3) == "0::") {
            if (entry) {
                out.malformed = true;
                out.note = "/proc/self/cgroup has more than one cgroup v2 entry";
                return out;
            }
            entry = line.substr(3);
        }
    }
    if (!entry) {
        out.note = "no cgroup v2 entry";
        return out;
    }
    // "." / ".." / empty components are refused so the walk cannot leave the
    // cgroup mount.
    if (entry->empty() || entry->front() != '/') {
        out.malformed = true;
        out.note = "cgroup path is not absolute";
        return out;
    }
    std::vector<std::string> components;
    for (std::size_t i = 1; i < entry->size();) {
        auto next = entry->find('/', i);
        if (next == std::string_view::npos) {
            next = entry->size();
        }
        const std::string_view part = entry->substr(i, next - i);
        if (part.empty() || part == "." || part == ".." || part.find('\0') != std::string_view::npos ||
            components.size() >= 64) {
            out.malformed = true;
            out.note = "cgroup path is not canonical";
            return out;
        }
        components.emplace_back(part);
        i = next + 1;
    }
    // depth 0 is the namespace-root cgroup itself (a container's own limit when
    // /proc/self/cgroup reads "0::/"); the host's real root has no memory.max.
    for (std::size_t depth = components.size() + 1; depth-- > 0;) {
        std::string dir(cgroup_root);
        for (std::size_t i = 0; i < depth; ++i) {
            dir += '/';
            dir += components[i];
        }
        const auto max_text = read_file(dir + "/memory.max");
        const auto cur_text = read_file(dir + "/memory.current");
        if (!max_text && !cur_text) {
            continue;  // memory controller not enabled at this level
        }
        const auto fail = [&](const char* what) {
            out.malformed = true;
            out.limited = false;
            out.headroom = MemoryReading::unknown();
            out.note = std::string(what) + " at cgroup level " + std::to_string(depth);
            return out;
        };
        if (!max_text || !cur_text) {
            return fail("memory.max or memory.current missing its pair");
        }
        if (max_text->size() > 32 || cur_text->size() > 32) {
            return fail("oversized memory.max or memory.current");
        }
        const auto current = parse_strict_u64(*cur_text);
        if (!current) {
            return fail("malformed memory.current");
        }
        if (*max_text == "max" || *max_text == "max\n") {
            continue;  // no limit at this level
        }
        const auto limit = parse_strict_u64(*max_text);
        if (!limit) {
            return fail("malformed memory.max");
        }
        const std::uint64_t room = *limit > *current ? *limit - *current : 0;
        if (!out.limited || room < out.headroom.bytes_or(0)) {
            out.headroom = MemoryReading::known(room);
        }
        out.limited = true;
    }
    if (!out.limited) {
        out.note = "no finite cgroup v2 memory limit";
    }
    return out;
}

HostMemory compose_linux_host_memory(std::string_view meminfo, const CgroupHeadroom& cgroup) {
    HostMemory out;
    out.source = cgroup.limited ? "meminfo+cgroup2" : "meminfo";
    if (const auto total = parse_meminfo_bytes(meminfo, "MemTotal")) {
        out.total_physical = MemoryReading::known(*total);
    }
    if (cgroup.malformed) {
        // A limit we cannot read could be tighter than MemAvailable: do not
        // report a number that may be too optimistic.
        out.note = "cgroup v2 limit unreadable: " + cgroup.note;
        return out;
    }
    const auto available = parse_meminfo_bytes(meminfo, "MemAvailable");
    if (available) {
        out.available_physical = MemoryReading::known(*available);
    } else {
        out.note = "MemAvailable missing or malformed in /proc/meminfo";
    }
    if (cgroup.limited && cgroup.headroom.is_known()) {
        const std::uint64_t room = cgroup.headroom.bytes_or(0);
        if (!available || room < *available) {
            out.available_physical = MemoryReading::known(room);
            out.note = "capped by cgroup v2 headroom";
        }
    }
    return out;
}

HostMemory read_host_memory() { return live_host_memory(); }

std::vector<DeviceInfo> enumerate_devices() {
    DeviceInfo cpu;
    cpu.id = "cpu:0";
    cpu.kind = DeviceKind::cpu;
    cpu.name = cpu_name();
    cpu.logical_cores = std::thread::hardware_concurrency();
    const HostMemory host = read_host_memory();
    cpu.total_memory_bytes = host.total_physical.bytes_or(0);
    cpu.available_memory_bytes = host.available_physical.bytes_or(0);
    cpu.available_physical = host.available_physical;
    cpu.available_commit = host.available_commit;
    return {cpu};
}

std::string host_platform() {
#if defined(_WIN32)
    std::string os = "windows";
#elif defined(__APPLE__)
    std::string os = "macos";
#elif defined(__linux__)
    std::string os = "linux";
#else
    std::string os = "unknown";
#endif
#if defined(_M_X64) || defined(__x86_64__)
    return os + "-x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return os + "-arm64";
#else
    return os + "-other";
#endif
}

std::string host_name() {
#if defined(_WIN32)
    char buffer[MAX_COMPUTERNAME_LENGTH + 1] = {};
    DWORD size = sizeof(buffer);
    if (GetComputerNameA(buffer, &size)) {
        return std::string(buffer, size);
    }
    return "unknown-host";
#else
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) == 0) {
        return buffer;
    }
    return "unknown-host";
#endif
}

}  // namespace sonder::inference
