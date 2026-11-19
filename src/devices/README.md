# Devices

Responsibility: Capability discovery and inference placement policy.

Implemented: `device.cpp` reports the host CPU (brand string, logical cores,
total/available RAM) on Windows, Linux, and macOS, plus host platform/name.

`read_host_memory()` returns a tri-state `MemoryReading` (known value, known
zero, unknown) for total/available physical memory and available commit.
Windows uses `GlobalMemoryStatusEx` (`ullAvailPhys`, `ullAvailPageFile`); Linux
uses `MemAvailable` capped by the tightest cgroup v2 `memory.max -
memory.current` up the ancestors, parsed strictly (a malformed cgroup file makes
the reading unknown rather than optimistic); macOS reports available memory
unknown. Callers choose their own fallback for unknown.

Deferred: GPU/NPU discovery (expected via backend capability probing) and the
placement planner.

See [source workspace](../README.md).
