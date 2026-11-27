# GPU, Ternary/1-Bit, and APU Hardware Support Guide

This guide is the operator reference for running Sonder Inference across discrete NVIDIA RTX GPUs, discrete AMD Radeon RX XT GPUs, integrated AMD Ryzen APUs (UMA), and compact ternary, 1-bit, and FP4/FP8 GGUF models.

Every flag, preset, tensor layout, probe, and API documented here corresponds to implemented code across `CMakePresets.json`, `src/devices/`, `src/backends/`, `src/server/`, `bench/`, `include/sonder_inference.h`, and `bindings/python/`. No throughput numbers are assumed or fabricated; measure target hardware with `sonder-infer tune` or `sonder-bench`.

## 1. Ternary, 1-Bit, and FP4/FP8 GGUF Formats

Sonder's header-only GGUF reader (`src/server/src/gguf_header.cpp`) recognizes the block layouts below without reading tensor payloads, so `estimate_vram()`, `sonder-infer plan`, and launch-profile fit checks size weights and KV caches accurately. Executing a model still requires a matching runtime build whose GGML implementation supports those tensor types.

### Weight and KV `ggml_type` Layouts

| `ggml_type` ID | Name | Elements / block | Bytes / block | Bits / value | Header flag | Runtime requirement |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `34` | `TQ1_0` | 256 | 54 | 1.6875 | `has_ternary_tensors` | Pinned `llama.cpp b11195` or newer |
| `35` | `TQ2_0` | 256 | 66 | 2.0625 | `has_ternary_tensors` | Pinned `llama.cpp b11195` or newer |
| `39` | `MXFP4` | 32 | 17 | 4.25 | `has_mxfp4_tensors`, `has_fp4_or_fp8_tensors` | Pinned `llama.cpp b11195` or newer (block-scaled FP4) |
| `40` | `NVFP4` | 64 | 36 | 4.50 | `has_nvfp4_tensors`, `has_fp4_or_fp8_tensors` | Mainline `ggml-org/llama.cpp >= b11556` (Blackwell native FP4; software dequant on older GPUs) |
| `41` | `Q1_0` | 128 | 18 | 1.125 | `has_one_bit_tensors` | Mainline `ggml-org/llama.cpp >= b11556` (1-bit Bonsai-1) |
| `42` | `Q2_0` | 64 | 18 | 2.25 | `has_ternary_tensors` | Mainline `ggml-org/llama.cpp >= b11556` (2-bit ternary) |
| `43` | `F8_E4M3` | 1 | 1 | 8.00 | `has_fp8_tensors`, `has_fp4_or_fp8_tensors` | Mainline `ggml-org/llama.cpp >= b11556` (Ada/Blackwell/RDNA 4 FP8) |
| `142` | `PQ2_0` | 128 | 34 | 2.125 | `has_ternary_tensors`, `has_prism_private_tensors` | `PrismML-Eng/llama.cpp` (`prism` branch; Bonsai-2) |
| `143` | `PTQ1_0` | 128 | 28 | 1.75 | `has_ternary_tensors`, `has_prism_private_tensors` | `PrismML-Eng/llama.cpp` (`prism` branch; dense ternary) |
| `145` | `PQ2_0M` | 128 | 36 | 2.25 | `has_ternary_tensors`, `has_prism_private_tensors` | `PrismML-Eng/llama.cpp` (`prism` branch; KV-only) |

### Quantized KV Cache Types

`kv_cache_type_bytes()` (`src/server/src/launch_profile.cpp`) models both standard and low-bit KV formats:

| Type | Bytes / cached value | Relative to `f16` | Supported backends |
| --- | ---: | ---: | --- |
| `f32` | 4.0000 | 2.00x | `llamacpp`, `llamaserver` |
| `f16`, `bf16` | 2.0000 | 1.00x | `llamacpp`, `llamaserver` |
| `q8_0` | 1.0625 (34/32) | 0.53x | `llamacpp`, `llamaserver` |
| `f8_e4m3` | 1.0000 | 0.50x | `llamaserver` (upstream `>= b11556`) |
| `q5_1` | 0.7500 (24/32) | 0.38x | `llamacpp`, `llamaserver` |
| `q5_0` | 0.6875 (22/32) | 0.34x | `llamacpp`, `llamaserver` |
| `q4_1` | 0.6250 (20/32) | 0.31x | `llamacpp`, `llamaserver` |
| `q4_0`, `iq4_nl` | 0.5625 (18/32) | 0.28x | `llamacpp`, `llamaserver` |
| `q2_0` | 0.28125 (18/64) | 0.14x | `llamaserver` (upstream `>= b11556`) |
| `pq2_0m` | 0.28125 (36/128) | 0.14x | `llamaserver` (PrismML `prism` branch) |

Rules enforced at startup before any model file is loaded:

- Any quantized V cache (`is_quantized_kv_cache_type()` — everything except `f32`, `f16`, `bf16`) requires `flash_attn` `on` or `auto`. Pairing a quantized V cache with `flash_attn: "off"` fails fast with `invalid_argument`.
- Direct `llamacpp` profiles and `serve` CLI flags reject `f8_e4m3`, `q2_0`, and `pq2_0m` as KV cache types (`is_llamacpp_kv_cache_type()` in `launch_profile.cpp`); use `llamaserver` with a matching `llama-server` binary. The default `v1.0.0` pin does define `GGML_TYPE_F8_E4M3`, and direct `LoadOptions` accept `f8_e4m3` there, but the profile and CLI paths still reject it.
- Compact ternary/1-bit weights reduce weight VRAM sharply, making KV cache dtype, context length, FlashAttention kernel pairing, and concurrent slot count the primary VRAM levers on 16 GiB cards.

### PrismML Hadamard Metadata (`prism.hadamard.*`) and Runtime Overrides

When a GGUF header contains any `prism.hadamard.*` metadata key (such as `prism.hadamard.transform`, `prism.hadamard.block_size`, or `prism.hadamard.sign_values`) or any PrismML private tensor (`PQ2_0`, `PTQ1_0`, `PQ2_0M`):

- `gguf_header.cpp` records `prism_hadamard = true`, captures `prism_hadamard_transform` and `prism_hadamard_block_size`, and streams past large sign arrays without bloating `GgufModelInfo::metadata`.
- `VramEstimate`, `PlacementPlan`, and `/v1/models` (`sonder.profile`) set `"prism_hadamard": true` and `"requires_prism_runtime": true`.
- `plan_model_placement()` adds an explicit warning that stock `llama-server` cannot load the file and requires PrismML's Hadamard-capable runtime.
- If a spawned `llama-server` fails to load unknown tensor types or reports a `prism.hadamard` error in its log file, `LogDiagnostics` emits warning code `ternary_runtime_incompatible`.

To build the direct `llamacpp` backend against a newer mainline or PrismML `llama.cpp` tree without editing the default pin in `cmake/llamacpp-pin.json` (see also [BACKENDS.md](BACKENDS.md#upstream-llamacpp-pin-and-ternary--prismml-overrides)):

```powershell
# Windows one-liner (normalizes path, enters VS x64 DevShell, builds & tests):
powershell -NoProfile -File scripts\build.ps1 `
  -Preset windows-cuda-blackwell `
  -LlamaCppSourceDir D:\ollama\runtimes\PrismML-Eng\llama.cpp `
  -Jobs 8 -Test
```

```sh
# Linux / WSL2 one-liner (probes GPU toolchains, builds & tests):
./scripts/build.sh \
  --preset linux-cuda-blackwell \
  --llamacpp-src /path/to/PrismML-Eng/llama.cpp \
  --jobs 8 --test

# Direct CMake invocation or explicit upstream tag + SHA-256 (fails closed on mismatch):
cmake --preset linux-cuda-rtx \
  -DSONDER_LLAMACPP_TAG=b11556 \
  -DSONDER_LLAMACPP_SHA256=<sha256-of-tarball>
```

Both wrapper scripts also forward arbitrary CMake cache flags (`-ExtraCMakeArgs @('-DSONDER_WARNINGS_AS_ERRORS=ON')` in `scripts/build.ps1`; `-- -DSONDER_WARNINGS_AS_ERRORS=ON` or `-D...` in `scripts/build.sh`). For WSL/Linux CI or CPU-only direct verification without GPU toolchains, use `--preset linux-llamacpp` (`SONDER_WITH_LLAMA_CPP=ON`, `RelWithDebInfo`).

The `llamaserver` backend requires no `llama.cpp` compile-time linkage: point `--llamaserver-executable` (or `executable` in JSON config) at a mainline `>= b11556` or PrismML `llama-server` binary.

### PrismML Bonsai GPU Acceleration Matrix

| Host OS | Accelerator | Preset (`CMakePresets.json`) | Runtime / Toolchain | Supported Bonsai GGUF Families |
| --- | --- | --- | --- | --- |
| Windows 11 | NVIDIA Blackwell (`sm_120`, RTX 50-series) | `windows-cuda-blackwell` | CUDA 12.8+ / 13.x + MSVC (`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (`PQ2_0`, `PTQ1_0`, `draft-mtp` via PrismML tree) |
| Windows 11 | NVIDIA RTX (`sm_86;89;120`, RTX 30/40/50) | `windows-cuda-rtx` | CUDA 12.8+ / 13.x + MSVC (`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (`PQ2_0`, `PTQ1_0`, `draft-mtp`) |
| Linux / WSL2 | NVIDIA Blackwell (`sm_120`, RTX 50-series) | `linux-cuda-blackwell` | CUDA 12.8+ / 13.x + GCC/Clang (`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (`PQ2_0`, `PTQ1_0`, `draft-mtp`) |
| Linux / WSL2 | NVIDIA RTX (`sm_86;89;120`, RTX 30/40/50) | `linux-cuda-rtx` | CUDA 12.8+ / 13.x + GCC/Clang (`GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (`PQ2_0`, `PTQ1_0`, `draft-mtp`) |
| Windows 11 | AMD Radeon RX XT (RDNA 2/3/4) & Ryzen APU | `windows-rocm-xt` | AMD HIP SDK (`HIP_PATH`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (with matching `llama.cpp` tree) |
| Linux | AMD Radeon RX XT (RDNA 2/3/4) & Ryzen APU | `linux-rocm-xt` | ROCm 6.x+ (`/dev/kfd`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON`) | Bonsai-1 (`Q1_0`), Bonsai (`Q2_0`), Bonsai-2 (with matching `llama.cpp` tree) |
| Windows / Linux | AMD RX XT, Ryzen APU UMA, Intel, Cross-Vendor | `windows-vulkan-xt` / `linux-vulkan-xt` | Vulkan SDK (`GGML_VULKAN=ON`) | Cross-vendor Vulkan compute (including systems without `/dev/kfd` or HIP SDK) |
| Linux / WSL2 | CPU (x86_64 / ARM64) | `linux-llamacpp` | GCC/Clang + Ninja (`SONDER_WITH_LLAMA_CPP=ON`) | Direct CPU execution and CI compile/test verification |

---

## 2. NVIDIA RTX (Ampere, Ada, Blackwell)

### Build Presets (`CMakePresets.json`)

| Preset | Platform | Flags |
| --- | --- | --- |
| `windows-cuda-blackwell` | Windows (MSVC + Ninja, `Release`) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON` |
| `linux-cuda-blackwell` | Linux / WSL2 (GCC/Clang + Ninja, `Release`) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON` |
| `windows-cuda-rtx` | Windows (MSVC + Ninja, `RelWithDebInfo`) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86;89;120`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON` |
| `linux-cuda-rtx` | Linux / WSL2 (GCC/Clang + Ninja, `RelWithDebInfo`) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86;89;120`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON` |

- `windows-cuda-blackwell` and `linux-cuda-blackwell` target RTX 50-series Blackwell (`sm_120`, such as RTX 5070 Ti, RTX 5080, RTX 5090) in pure `Release` mode (requiring CUDA Toolkit 12.8+ or 13.x), cutting multi-arch NVCC compile time while enabling `GGML_CUDA_GRAPHS=ON` for low-overhead single-token decode launches.
- `86;89;120` in `*-cuda-rtx` covers RTX 30-series (Ampere `sm_86`), RTX 40-series (Ada `sm_89`), and RTX 50-series (Blackwell `sm_120`) in one portable binary.
- `GGML_CUDA_FA_ALL_QUANTS=ON` compiles CUDA FlashAttention kernels for quantized KV pairs (`q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl`) so `llama.cpp` does not silently dequantize K/V back to `f16` at runtime.

### VRAM Discovery and Process Telemetry

- **Windows**: `enumerate_gpu_devices()` loads `dxgi.dll` from System32 (`LOAD_LIBRARY_SEARCH_SYSTEM32`), filters software and Microsoft virtual display adapters (`0x1414`), and reads `DedicatedVideoMemory`, `SharedSystemMemory`, and live free budget via `IDXGIAdapter3::QueryVideoMemoryInfo` (`Budget - CurrentUsage` on `DXGI_MEMORY_SEGMENT_GROUP_LOCAL`), tagged `source: "dxgi"`, `backend_hint: "cuda"`. On headless or TCC-mode NVIDIA hosts where DXGI reports no NVIDIA adapters, it falls back to System32 `nvml.dll`. For spawned `llamaserver` children, the spill and residency guards poll Windows PDH `\GPU Process Memory(pid_<PID>_*)\Dedicated Usage` and `Shared Usage` (`probe: "pdh"`). See [integration/vram-spill.md](integration/vram-spill.md) for Windows WDDM shared-memory spill behavior and residency guard tuning.
- **Linux & WSL2**: `enumerate_gpu_devices()` dynamically loads `libnvidia-ml.so.1` via `dlopen` — falling back automatically to `/usr/lib/wsl/lib/libnvidia-ml.so.1` inside WSL2 (`/dev/dxg`) when the WSL driver directory is not in the default dynamic-linker cache — and queries `nvmlDeviceGetMemoryInfo` plus PCI device IDs (`source: "nvml"`, `backend_hint: "cuda"`). Spawned `llamaserver` monitoring and `sonder-infer tune` query per-PID NVML compute/graphics process memory (`probe: "nvml"`, or `"amdgpu-fdinfo+nvml"` on mixed AMD + NVIDIA hosts).

### Placement Defaults and Multi-GPU Splitting

For discrete NVIDIA RTX adapters, `plan_model_placement()` selects:

- `flash_attn: "on"`, `kv_unified: true`, and `load_mode: "none"` for fully VRAM-resident models (automatically relaxed to `load_mode: "auto"` when `--n-cpu-moe`, `no_repack`, or partial `n_gpu_layers` is applied).
- Initial KV cache `q8_0/q8_0` (or `f8_e4m3/f8_e4m3` when `PlacementPlanRequest::prefer_fp8_kv` is enabled on discrete NVIDIA RTX GPUs with upstream `llama-server >= b11556`), stepping down through the 3-step ladder **`q8_0/q8_0` -> asymmetric `q8_0/q4_0` -> `q4_0/q4_0`** before shrinking context so keys retain 8-bit precision whenever the ~24% value-only savings are enough to fit VRAM (unless `--no-quantized-kv` is set).
- **Blackwell (`sm_120`) FP4/FP8 sizing**: `gguf_header.cpp` and `vram_estimate.cpp` identify `MXFP4` (`4.25 bpw`, 17 B / 32 elems), `NVFP4` (`4.50 bpw`, 36 B / 64 elems), and `F8_E4M3` (`8.00 bpw` weights / `1.0 byte/elem` KV). On Blackwell GPUs (`sm_120`, RTX 50-series), `plan_model_placement()` notes native FP4 tensor-core acceleration for `NVFP4`; on older or non-NVIDIA GPUs it emits a software-dequantization warning.
- **Dynamic SLRU MoE Expert Cache (`--moe-cache-mib` + `--moe-cache-cpu-share`) & Selective CPU Repacking**: when a MoE model fits 100% in VRAM, `--moe-cache-mib` is **never** enabled (avoiding ~10% graph-splitting overhead). When a MoE overflows VRAM via `--n-cpu-moe`, the planner computes remaining spare VRAM headroom after pinned GPU layers + KV + compute buffers and enables `--moe-cache-mib` + `--moe-cache-cpu-share` (`0.25–0.50` when host DDR5 RAM is ample) **only** when spare VRAM >= 512 MiB and covers the active CPU-offloaded expert working set (avoiding <75% hit-rate PCIe thrash). Global `--repack` is never emitted while GPU layers remain active (`repack_policy: "cpu_only_selective"`), and CPU-offloaded MoE decode recommends physical core count (`recommended_threads`, e.g. `logical_cores / 2` clamped >= 4) rather than oversubscribing SMT threads on DDR5 bandwidth.
- Speculative decoding: `draft-mtp` (`draft_n_max: 3`) when GGUF `<arch>.nextn_predict_layers > 0` (such as `Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf` and `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`), or lightweight `ngram-mod` for ternary/1-bit models without MTP layers (unless `--no-speculative` is set); launch profiles and `serve --spec-type` also support `suffix` (Arctic Suffix-Tree speculation). Tied LM-head embeddings (`prism.hadamard.tied_output = 1` / `token_embd.weight` without `output.weight`) and trailing dense-attention MTP blocks are sized accurately in `estimate_vram()`.
- **Multi-GPU (`split_mode: "layer"` + `tensor_split`)**: when two or more discrete NVIDIA GPUs are available and a single GPU cannot fit the model (or when `--device multi` / `--device CUDA0,CUDA1` is requested), the planner sets `device: "CUDA0,CUDA1"`, `split_mode: "layer"`, proportional `tensor_split` weights sized to each GPU's usable VRAM budget, and `main_gpu` to the largest shard when index > 0. `VramEstimate::max_gpu_mib()` validates the heaviest single-GPU shard plus compute buffers. Direct `llamacpp` users can configure the same `devices`, `main_gpu`, `split_mode`, `tensor_split`, and `use_mlock` fields on `LlamaCppBackendOptions` / `LoadOptions` (see [PLACEMENT.md](PLACEMENT.md#multi-gpu-vendor-targeting-and-memory-locking-implemented)).

---

## 3. AMD Radeon RX XT (RDNA 2 / 3 / 4)

### Build Presets (`CMakePresets.json`)

| Preset | Platform | Accelerator | Flags |
| --- | --- | --- | --- |
| `windows-rocm-xt` | Windows (AMD HIP SDK Clang + Ninja) | ROCm/HIP (`ROCm0`, ...) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON`, `AMDGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1200;gfx1201` |
| `linux-rocm-xt` | Linux (GCC/Clang + Ninja) | ROCm/HIP (`ROCm0`, ...) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON`, `AMDGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1200;gfx1201` |
| `windows-vulkan-xt` | Windows (MSVC + Ninja) | Vulkan (`Vulkan0`, ...) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_VULKAN=ON` |
| `linux-vulkan-xt` | Linux (GCC/Clang + Ninja) | Vulkan (`Vulkan0`, ...) | `SONDER_WITH_LLAMA_CPP=ON`, `GGML_VULKAN=ON` |

`AMDGPU_TARGETS` covers RDNA 2 (`gfx1030`, RX 6800/6900 XT), RDNA 3 discrete (`gfx1100`, `gfx1101`, `gfx1102`, RX 7900/7800/7700/7600 XT), RDNA 3/3.5 Ryzen APUs (`gfx1103` Radeon 780M, `gfx1150` Radeon 890M), and RDNA 4 (`gfx1200`, `gfx1201`, RX 9060 / RX 9070 XT). `GGML_HIP_GRAPHS=ON` enables HIP graph capture for lower kernel-launch overhead during decode, and `GGML_HIP_ROCWMMA_FATTN=ON` enables rocWMMA FlashAttention kernels on RDNA 3/4 (`gfx1100`–`gfx1201`).

### RDNA 4 (`gfx1200`/`gfx1201`) Vulkan Cooperative Matrix vs ROCm Advisory

On RDNA 4 discrete cards (RX 9060 `gfx1200`, RX 9070 / 9070 XT `gfx1201`), Vulkan's `VK_KHR_cooperative_matrix` path is a competitive or preferred fallback alongside ROCm (especially on Windows hosts without the HIP SDK or when evaluating FlashAttention kernel parity):

- `plan_model_placement()` recognizes RDNA 4 PCI IDs (`0x7550`, `0x7551`, `0x7590`) and names (`9060`, `9070`, `gfx1200`, `gfx1201`, `RDNA 4`) and adds an explicit backend advisory in `plan.reasons` (noting `VK_KHR_cooperative_matrix` under Vulkan and `GGML_HIP_ROCWMMA_FATTN=ON` plus the Vulkan CoopMat fallback under ROCm).
- `sonder-infer tune` (`grid_for_gpu()`) is aware of RDNA 4 and explicit `vulkan`/`rocm` GPU hints, sweeping `flash_attn` across `{"on", "auto"}` so operators can benchmark Vulkan CoopMat against ROCm WMMA FlashAttention head-to-head on their target model.
- **AMD Windows Vulkan cooperative-matrix canary (`NAT-16`)**: before `llama_backend_init()` on Windows, Sonder Inference probes eligible AMD GPUs (both discrete e.g. RX 9060 XT and integrated UMA e.g. Radeon 780M) in an isolated child process (`sonder-infer coopmat-canary --vendor-id ... --device-id ... --ubatch 64`). The child runs `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (guarding `llama.cpp#22750`/`#23508` iGPU init crashes) followed by a `16x16x16` FP16->FP32 `OpCooperativeMatrixMulAddKHR` compute dispatch over `ubatch >= 64` rows (guarding `llama.cpp#32447` `gfx1201` compute crashes). Results are cached per `(vendor_id, device_id, driver_build, ubatch)` under `%LOCALAPPDATA%/sonder-inference/coopmat-canary-v1.json`; driver updates automatically invalidate the cache. Any child crash, timeout, or verification failure automatically sets `GGML_VK_DISABLE_COOPMAT=1` before `ggml-vulkan` initializes and surfaces the `amd_vk_coopmat_disabled` warning in `runtime_status()` / `backend.warning`. Setting `GGML_VK_DISABLE_COOPMAT=1` skips the canary; setting `SONDER_SKIP_COOPMAT_CANARY=1` bypasses automatic probing.

### Discovery, Per-Process Counters, and Environment Policy

- **Linux sysfs discovery (`/sys/class/drm` + `/dev/kfd`)**: `parse_amdgpu_sysfs_device()` scans `/sys/class/drm/cardN/device/` for PCI vendor `0x1002` with `DRIVER=amdgpu` in `uevent`, reading `mem_info_vram_total`, `mem_info_vram_used`, `mem_info_gtt_total`, `mem_info_gtt_used`, `product_name`, and `device` (`source: "amdgpu-sysfs"`). Live enumeration checks `/dev/kfd` R/W accessibility and `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` (`is_rocm_device_visible()`): cards with usable ROCm access report `backend_hint: "rocm"`, while systems/containers without `/dev/kfd` (such as Vulkan-only Linux Ryzen APUs or unprivileged containers) or cards masked by `HIP_VISIBLE_DEVICES` automatically fall back to `backend_hint: "vulkan"`.
- **Linux per-process VRAM/GTT (`/proc/<pid>/fdinfo`)**: `LinuxGpuCounterSource` scans `/proc/<pid>/fdinfo/*` for `drm-memory-vram` and `drm-memory-gtt`, deduplicating shared DRM file descriptors by `drm-driver:drm-client-id` (`probe: "amdgpu-fdinfo"`). This feeds `llamaserver` spill/residency checks and `sonder-infer tune` on Linux AMD hosts.
- **Windows discovery & counters**: DXGI reports dedicated and shared adapter memory, and PDH reads per-process dedicated/shared usage. When the AMD HIP SDK (`HIP_PATH` / `ROCM_PATH`) is installed and visible via `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`, discrete Radeon RX XT cards report `backend_hint: "rocm"` (pairing cleanly with `windows-rocm-xt` and numbering `ROCm0` accurately even on Ryzen desktops whose integrated iGPU uses `"vulkan"`); without the HIP SDK (or on integrated Ryzen UMA iGPUs), Windows AMD adapters default to `backend_hint: "vulkan"`.
- **`GPU_MAX_ALLOC_PERCENT=100`**: `plan_model_placement()` sets `flash_attn: "auto"`, `load_mode: "auto"`, `kv_unified: true`, and recommends `GPU_MAX_ALLOC_PERCENT=100` in `recommended_env` for discrete AMD ROCm/HIP cards (and AMD ROCm UMA) so the ROCm allocator permits single allocations up to full VRAM/GTT capacity.
- **Child environment containment (`src/backends/llamaserver/env_policy.hpp`)**: `GPU_MAX_ALLOC_PERCENT`, `GPU_MAX_HEAP_SIZE`, `GPU_SINGLE_ALLOC_PERCENT`, `GPU_FORCE_64BIT_PTR`, `GPU_USE_SYNC_OBJECTS`, `DRI_PRIME`, `VK_ICD_FILENAMES`, `AMD_LOG_LEVEL`, and safe vendor tuning prefixes (`ROCR_*`, `ROCM_*`, `HSA_*`, `HIP_*`, `AMD_*`, `VK_*`, `RADV_*`, `MESA_*`, `CUDA_*`, `GGML_*`, `OMP_*`) are permitted. Loader/layer injection and file-write log hooks (`GGML_BACKEND_PATH`, `CUDA_INJECTION64_PATH`, `CUDA_INJECTION32_PATH`, `HSA_TOOLS_LIB`, `VK_LAYER_PATH`, `VK_ADD_LAYER_PATH`, `VK_INSTANCE_LAYERS`, `VK_LOADER_LAYERS_ENABLE`, `VK_DRIVER_FILES`, `VK_ADD_DRIVER_FILES`, `MESA_LOADER_DRIVER_OVERRIDE`, `AMD_LOG_LEVEL_FILE`, `HIPBLASLT_LOG_FILE`, `ROCBLAS_LOG_*`) and credential-suffixed variables are always refused as overrides and stripped from inherited child environments.

### ROCm/HIP and Vulkan Log Diagnostics

When `log_file` (or `--log-file` in `args`) is enabled on a spawned `llamaserver` child, `LogDiagnostics` (`src/backends/llamaserver/log_diagnostics.cpp`) classifies vendor errors and KV fallback lines into `backends[].runtime.warnings` and `backend.warning` telemetry:

| Warning code | Triggered by | Details / notes |
| --- | --- | --- |
| `gpu_init_failed` | `failed to initialize CUDA`, `failed to initialize HIP`, `failed to initialize ROCm` | `backend: "rocm"` for HIP/ROCm failures |
| `gpu_oom` | `cuda error: out of memory`, `OutOfDeviceMemoryError`, `VK_ERROR_OUT_OF_DEVICE_MEMORY`, `hipErrorOutOfMemory`, `hipErrorMemoryAllocation`, or vendor-qualified `out of memory` | `backend`: `"cuda"`, `"rocm"`, or `"vulkan"` |
| `rocm_runtime_error` | `HSA_STATUS_ERROR`, `hipError` (non-OOM) | `backend: "rocm"` |
| `vulkan_runtime_error` | `failed to initialize Vulkan` (`reason: "init_failed"`), `ggml_vulkan: device lost on <device>` (`reason: "device_lost"`), and `VK_ERROR_DEVICE_LOST` (warning only, no unhealthy marking) | `backend: "vulkan"` |
| `device_fault` | `ggml_vulkan: device fault on <device>: <text>` (VK_EXT_device_fault, `reason: "device_fault"`), fatal CUDA/HIP errors (`CUDA error: ` / `ROCm error: ` lines) such as `unspecified launch failure` or `illegal memory access` (`reason: "fatal_error"`) | `backend`: `"vulkan"`, `"cuda"`, or `"rocm"`. Together with `ggml_vulkan: device lost on` this marks the device unhealthy once the child has exited or fails `/health`: the supervisor stops, does not relaunch, and reports `diagnostics.device`. See `docs/runbooks/node1-oculink-gpu-hang.md` |
| `ternary_runtime_incompatible` | `unknown tensor type`, `unsupported ggml type`, `unsupported tensor type`, or error-qualified `prism.hadamard` | Recommends PrismML binary or `llama.cpp >= b11556` |
| `kv_type_mismatch`, `kv_type_no_vector_kernel`, `kv_type_kernel_unknown` | Up-front KV cache pairing check (`check_kv_cache_pairing`) | Backend-aware messaging for `cuda`, `rocm`, and `vulkan` |

---

## 4. AMD Ryzen APU / Integrated UMA

Integrated APUs such as Ryzen 780M (`gfx1103`) and 890M (`gfx1150`) expose only a small BIOS dedicated carveout (commonly 512 MiB–2 GiB) while allocating GPU buffers from a much larger shared/GTT system memory pool.

### Detection and Host-RAM Pool Cap

- `enumerate_gpu_devices()` sets `is_integrated_uma = true` when `dedicated == 0 && shared > 0`, on Apple GPUs with `shared > 0`, or on AMD/Intel GPUs with `dedicated <= 2 GiB && shared >= 4 * dedicated && shared > 0`.
- `gpu_effective_vram_mib()` computes the usable UMA pool as `dedicated + shared` (total) or `local_free + non_local_free` (DXGI) / `vram_free + gtt_free` (`amdgpu-sysfs`) when `prefer_free` is set.
- `plan_model_placement()` caps the UMA pool at `host.available_physical` (`uma_capped_by_host`), applies a proportional OS reserve (`min(vram_headroom_mib, max(min(512, pool/4), pool/8))`), sets `uma_mode: true`, `flash_attn: "auto"`, `load_mode: "auto"`, and validates `total_bytes + host_resident_bytes <= host.available_physical` so GPU and host allocations sharing physical RAM do not overcommit the system.
- On hybrid desktop/laptop hosts that have both a discrete GPU (RTX or RX XT) and a Ryzen iGPU, automatic selection strictly prefers the discrete GPU. Target the APU explicitly with `--device uma` (or `igpu`, `apu`, or `Vulkan1`).

### `llamaserver` Spill & Residency Guard `"uma_mode"`

Because UMA allocations live in shared/GTT memory by design, treating shared usage as a discrete-GPU PCIe spill would false-positive on every healthy APU load. Configure `uma_mode` in `--llamaserver-config` under `spill_guard` (and optionally override independently under `spill_guard.residency`):

```json
{
  "mode": "spawn",
  "executable": "/path/to/llama-server",
  "args": ["--model", "model.gguf", "--ctx-size", "32768", "--device", "Vulkan0"],
  "spill_guard": {
    "uma_mode": "auto",
    "residency": {
      "uma_mode": "auto"
    }
  }
}
```

- Accepted JSON values: boolean `true` or `false`, or strings `"auto"`, `"on"`, `"off"`, `"true"`, `"false"`.
- `"auto"` inspects `enumerate_gpu_devices()` and any `--device`/`-dev` hint in `args`: it resolves to `true` only when every targeted GPU (or every detected GPU when unhinted) has `is_integrated_uma == true`, and `false` whenever a discrete GPU is present/targeted.
- When resolved `true`: `is_spilled()` returns `false` (no shared-memory spill warning/refusal/step-down), and the residency guard counts `dedicated_bytes + shared_bytes` as active GPU residency.

### Bandwidth Caveats

Integrated UMA GPUs share DDR/LPDDR memory bandwidth with the CPU. Even when a model and long KV cache fit comfortably inside a 32–64 GiB unified memory pool, prefill and decode speeds remain memory-bandwidth bound compared to discrete GDDR VRAM. `plan_model_placement()` always emits a warning to that effect in UMA mode; use compact quantizations (ternary/1-bit or Q4) and quantized KV caches to reduce bytes read per token, and verify real latency with `sonder-bench`.

---

## 5. CLI, HTTP, Engine Telemetry, C ABI, and Python Tooling

### CLI (`sonder-infer` and `sonder-bench`)

- **Device and node inventory**:
  - `sonder-infer devices [--json]` lists CPU cores/RAM plus every discovered GPU (`dedicated_total_mib`, `dedicated_free_mib`, `shared_total_mib`, `effective_vram_mib`, `effective_free_vram_mib`, `is_integrated_uma`, `backend_hint`, `source`).
  - `sonder-infer node [--json]` prints a unified `sonder.inference.node/1` snapshot with `host_memory` (including Windows `available_commit_bytes`), `devices`, `gpus`, compiled `backends`, and `capabilities` (`ternary_gguf`, `placement_planner`, `coresidency_planner`, `multi_gpu`, `uma_telemetry`, `adaptive_speculation`).
- **Single-model and multi-model co-residency planning**:
  - `sonder-infer plan --model model.gguf [--ctx 32768] [--device CUDA0|ROCm0|Vulkan0|uma|multi|cpu] [--json]` reads only the GGUF header and live GPU/RAM inventory and emits a validated `LaunchProfile`, `VramEstimate`, `recommended_env`, `reasons`, and `warnings`.
  - Repeat `--model` (`sonder-infer plan --model a.gguf --model b.gguf --json`) to run `plan_coresident_models()` (`sonder.inference.coresidency_plan/1`): plans models sequentially in input order, deducts per-GPU OS headroom once per occupied GPU, tracks remaining VRAM and host RAM across single-GPU co-residency, multi-GPU splits, and UMA overlap, and overflows unpinned models to a second GPU, iGPU, or CPU when GPU 0 fills.
- **Calibration (`sonder-infer tune`)**:
  - Supports Windows PDH, Linux `amdgpu-fdinfo`, and Linux NVML counters.
  - Grid axes include `kv`, `flash_attn` (`on`, `auto`, `off`), `ctx`, `ubatch`, `mtp`, `spec_type` (`draft-mtp`, `draft-mtp,ngram-mod`, and standalone `ngram-mod` emitting `--spec-type ngram-mod --spec-ngram-mod-n-max N` even when a model has no `nextn` tensors), and `p_min`.
- **Multi-profile serving (`sonder-infer serve`)**:
  - Repeat `--profile NAME` or pass `--all-profiles` with `--profiles FILE` to serve multiple profiles behind one `serve` instance (first profile is `default`).
  - Validates aggregate co-residency VRAM across active offloading profiles (summed eagerly, or top-K under `--max-resident-models`; `--lazy-models --max-resident-models 1` validates peak single-model fit) and refuses overcommit unless `--allow-overcommit` is passed.
  - Supports multi-model in-process `llamacpp` residency, multi-profile `llamaserver` attach mode (e.g. an upstream router), and shared-GGUF `llamaserver` spawn aliases.
- **Concurrent benchmarking (`sonder-bench`)**:
  - `--concurrency N` (1 to 64, default 1) runs `N` barrier-synchronized worker sessions per measured iteration for non-fanout prompts, reporting `aggregate_decode_tokens_per_sec`, batch `wall_seconds`, and per-request p50/p95/p99 distributions.
  - Captures lightweight before/after `gpus_before` and `gpus_after` snapshots (with `peak_vram_delta_mib` when free VRAM drops) and renders a `## GPUs` Markdown table unless `--no-gpu-snapshot` is passed.
  - Use `--corpus bench/corpus/ternary_agent.json` to benchmark ternary/1-bit models, speculative decoding (`code_edit`, `tool_json`, `reasoning`, `prose`), ~4k hybrid long-context prefill, and 4-way shared-prefix agent fan-out.

### HTTP API (`sonder-infer serve`)

- `GET /v1/sonder/health` includes the 2 s TTL-cached top-level `gpus` array alongside `backends[].runtime` (which carries `gpu_memory`, `context`, `warnings`, and `diagnostics.adaptive_speculation` when live adaptive speculation is enabled).
- `GET /v1/sonder/node` returns `sonder.inference.node/1` (hardware, host memory, GPUs, backends, residency state, profiles, and feature flags), advertised in `/.well-known/sonder-telemetry` under `links.node`.
- `POST /v1/sonder/plan` returns `sonder.inference.plan/1` (capped at 64 KiB request body, bearer-auth gated when `--token-file` is set, advertised under `links.plan`). For containment, it never accepts arbitrary host paths: pass either `"profile"` (resolved only against configured profiles) or `"model"` (a clean `.gguf` basename canonicalized inside `--model-dir`, rejecting traversal and escaping symlinks), plus optional tuning overrides (`ctx_size`, `min_ctx_size`, `parallel`, `vram_headroom_mib`, `device_hint`, `prefer_free_vram`, `allow_quantized_kv`, `allow_speculative`, `allow_moe_cpu_offload`). All paths in the response are redacted to basenames.
- `GET /v1/sonder/identity[?model=ID]` computes real 9-key GGUF identity for `llamacpp`: streaming 64 KiB full-file SHA-256 (`model_digest`), single-pass canonical 12-key `tokenizer.ggml.*` hash (`tokenizer_digest`), raw `tokenizer.chat_template` hash (`template_digest`), `general.file_type` / dominant-tensor `quantization`, and GPU+CPU `hardware`, cached in a bounded 16-entry `(path, size, mtime)` LRU cache. If required fields such as `tokenizer.chat_template` are absent, it honestly returns `null` with an explanatory `reason` instead of fabricating digests.

### Live Adaptive Speculation (`llamaserver`)

Enable per-workload dynamic speculation control in `--llamaserver-config` (opt-in; requires native `llama-server` mode, `native_completion: true`):

```json
{
  "mode": "attach",
  "base_url": "http://127.0.0.1:8080",
  "adaptive_speculation": {
    "enabled": true,
    "min_draft_n": 0,
    "initial_draft_n": 3,
    "max_draft_n": 6,
    "window_size": 16,
    "min_samples": 4,
    "pause_acceptance_threshold": 0.25,
    "step_down_acceptance_threshold": 0.45,
    "step_up_acceptance_threshold": 0.72,
    "cooldown_requests": 8,
    "probe_draft_n": 1,
    "min_draft_tokens_per_sample": 4
  }
}
```

- Classifies requests deterministically into `tool_json` (grammar set), `reasoning` (`separate_reasoning` / thinking enabled), `code_edit` (fenced code/diff markers), `prose`, or `general`.
- Injects `"speculative.n_max"` on native `llama-server` requests, updates per-workload rolling acceptance windows from upstream `timings`, steps `draft_n_max` up/down with hysteresis, pauses low-acceptance buckets when `min_draft_n == 0`, and re-probes with monotonic `probe_id` tracking and RAII error/cancel cleanup.
- Exposes live per-bucket state under `runtime_status()->diagnostics["adaptive_speculation"]` (visible on `/v1/sonder/health` and `/v1/models`).

### Engine GPU Telemetry and Hybrid Recurrent Checkpoints

- **Live GPU telemetry**: `EngineOptions::sample_gpu_devices` (default `true`) populates `Engine::gpu_devices()`, reports `gpu_device_count` on `engine.started`, and emits `device.memory.sample` Observatory events (`kind: "gpu"`, `vendor`, `is_integrated_uma`, `dedicated_total_bytes`, `dedicated_free_bytes`, `shared_total_bytes`, `effective_vram_mib`, `effective_free_vram_mib`) on startup and periodic sampling.
- **Hybrid/recurrent state checkpoints**: `BackendModel::capture_recurrent_state()` and `restore_recurrent_state()` (advertised via `Capability::recurrent_checkpoint`, `1u << 17`, `"recurrent_checkpoint"`; implemented for `llamacpp` via `llama_state_seq_get_size` / `llama_state_seq_get_data` / `llama_state_seq_set_data` with a 512 MiB snapshot cap) pair with a bounded checkpoint blob store in `SchedulingRuntime` and `KvCacheManager::save_checkpoint()`, emitting `kv.checkpoint.saved` and `kv.checkpoint.restored` telemetry.

### C ABI v1 and Python Bindings

All additions preserve `SONDER_ABI_VERSION == 1` via `struct_size` versioning (`include/sonder_inference.h`, `bindings/python/src/sonder_inference/`):

| Capability | C ABI v1 (`sonder_inference.h`) | Python (`sonder_inference`) |
| --- | --- | --- |
| Live GPU inventory | `sonder_gpu_info`, `sonder_gpu_device_count()`, `sonder_gpu_device_info()` | `GpuInfo`, `list_gpu_devices()` |
| ADR-020 scheduling metadata | `sonder_workload_class` (`SONDER_WORKLOAD_DEFAULT = 0`, classes `1..7`), appended `workload_class` & `priority` (`[-16, 16]`) on `sonder_request_metadata` | `WorkloadClass`, `RequestMetadata(parent_request_id, workload_class, priority)` |
| Live telemetry callback | `sonder_telemetry_callback`, `sonder_engine_set_telemetry_callback()` | `Engine(..., telemetry_callback=...)`, `Engine.set_telemetry_callback()` |

See [CLI.md](CLI.md), [SERVER.md](SERVER.md), [PLACEMENT.md](PLACEMENT.md), and [BACKENDS.md](BACKENDS.md) for command syntax, HTTP schemas, and build details.
