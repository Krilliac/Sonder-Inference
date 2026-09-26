// Sonder Inference: device inventory.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sonder::inference {

enum class DeviceKind { cpu, gpu, npu, remote };

const char* to_string(DeviceKind kind) noexcept;

struct DeviceInfo {
    std::string id;          // stable within a process, e.g. "cpu:0"
    DeviceKind kind = DeviceKind::cpu;
    std::string name;        // human readable, e.g. CPU brand string
    std::uint32_t logical_cores = 0;
    std::uint64_t total_memory_bytes = 0;      // 0 when unknown
    std::uint64_t available_memory_bytes = 0;  // 0 when unknown
};

// Host inventory. Phase 1 reports the CPU (with system RAM) only; GPU/NPU
// discovery is deferred to backend capability probing.
std::vector<DeviceInfo> enumerate_devices();

// Current OS/arch label, e.g. "windows-x86_64".
std::string host_platform();
// Host name used as the telemetry producer node_id default.
std::string host_name();

}  // namespace sonder::inference
