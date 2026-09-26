# Devices

Responsibility: Capability discovery and inference placement policy.

Implemented: `device.cpp` reports the host CPU (brand string, logical cores,
total/available RAM) on Windows, Linux, and macOS, plus host platform/name.

Deferred: GPU/NPU discovery (expected via backend capability probing) and the
placement planner.

See [source workspace](../README.md).
