# Devices

Responsibility: Capability discovery and inference placement policy.

Implemented: `device.cpp` reports the host CPU (brand string, logical cores,
total/available RAM) on Windows, Linux, and macOS, plus host platform/name,
and GPU inventory (`enumerate_gpu_devices()`) via Windows DXGI + dynamic NVML,
Linux `amdgpu` sysfs + dynamic NVML, and optional dynamic Vulkan physical-device
probing (`VK_EXT_memory_budget` per-heap readings, ReBAR host-visible VRAM
classification, compute shared memory, warp/subgroup size, driver name/build,
and advertised matrix-core extensions without creating a Vulkan device or
calling `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`).

Device telemetry (NAT-08) also collects driver build strings (Windows display
class registry / NVML / Vulkan / Linux sysfs), PCIe current/max link gen and
width (NVML, Windows `cfgmgr32.dll`, Linux sysfs), ASPM state (Windows active
power plan via `powrprof.dll`, Linux per-link sysfs and `pcie_aspm` policy),
and provides pure parsers for ggml device capability lines
(`parse_ggml_device_line()`, handling both v1.0.0/b12171+ `rebar` and older
lines where `rebar` is absent).

`coopmat_canary.cpp` implements the AMD Windows Vulkan cooperative-matrix
child-process canary (NAT-16): before `llama_backend_init()`, eligible AMD
Windows GPUs (both discrete e.g. RX 9060 XT and integrated UMA e.g. Radeon
780M) are probed in an isolated `sonder-infer coopmat-canary` child process
using dynamic `vulkan-1.dll` loading (Phase 1 `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`
for `llama.cpp#22750/#23508`; Phase 2 generic 16x16x16 FP16->FP32 compute smoke
test over `ceil(ubatch/16)` tiles — note that Phase 2 is a generic KHR coopmat
compute smoke test and does not reproduce upstream flash-attention coopmat1
flip-direct shader bugs, which are absent from pinned `v1.0.0` and gated
upstream by `llama.cpp#32415`). Verdicts are cached atomically per
`(vendor_id, device_id, driver_build, ubatch)`; any crash, hang, or mismatch
activates `GGML_VK_DISABLE_COOPMAT=1` automatically and surfaces an
`amd_vk_coopmat_disabled` runtime warning. Automatic probing requires the
`sonder-infer` executable (self, sibling, or `SONDER_INFER_EXE`); callers
without it alongside skip automatic probing unless `SONDER_INFER_EXE` is set.

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
