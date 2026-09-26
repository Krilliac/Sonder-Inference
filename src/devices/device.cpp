#include "sonder/inference/device.hpp"

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

void memory(std::uint64_t& total, std::uint64_t& avail) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        total = status.ullTotalPhys;
        avail = status.ullAvailPhys;
    }
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

void memory(std::uint64_t& total, std::uint64_t& avail) {
    std::uint64_t mem = 0;
    size_t size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0) {
        total = mem;
    }
    avail = 0;  // not cheaply available without mach APIs; reported as unknown
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

void memory(std::uint64_t& total, std::uint64_t& avail) {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "MemTotal:") {
            total = value * 1024;
        } else if (key == "MemAvailable:") {
            avail = value * 1024;
        }
    }
}
#endif

}  // namespace

std::vector<DeviceInfo> enumerate_devices() {
    DeviceInfo cpu;
    cpu.id = "cpu:0";
    cpu.kind = DeviceKind::cpu;
    cpu.name = cpu_name();
    cpu.logical_cores = std::thread::hardware_concurrency();
    memory(cpu.total_memory_bytes, cpu.available_memory_bytes);
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
