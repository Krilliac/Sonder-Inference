# Model placement across memory tiers

Status: **steps 1 and 2 implemented** (tensor placement overrides, multi-GPU
placement, and the header-only per-model and multi-model co-residency placement
planner). Steps 3–4 remain design only (local node capability inventory is
implemented in `sonder-infer node` / `GET /v1/sonder/node`). See
[GPU_SUPPORT.md](GPU_SUPPORT.md) for the complete hardware, ternary/1-bit, RTX,
AMD RX XT, and Ryzen APU UMA operator guide.

Scope note: the direct placement, MoE cache, and KV-cache flags on this page
(`--gpu-layers`, `--context-length`, `--moe-experts`, `--tensor-override`,
`--moe-cache-mib`, `--moe-cache-cpu-share`, `--cache-ram-mib`, `--mmproj`,
`--spec-type`, `--batch-size`, `--ubatch-size`, `--cache-type-k`,
`--cache-type-v`, `--flash-attn`) apply to the `llamacpp` backend only; `serve`
rejects them for any other backend with `option --<flag> applies only to --backend llamacpp`
(`src/server/src/serve_options.cpp`). `--device` is not one of them: every
backend accepts it and reports it as the model's device id in telemetry and
health. Build the `llamacpp` backend with `-DSONDER_WITH_LLAMA_CPP=ON`
([README](../README.md#build)); installation and run steps are in
[SETUP.md](SETUP.md).

## Why

VRAM is the scarcest resource on a Sonder host. The reference workstation has
16 GB of discrete VRAM, and a second host has an integrated GPU sharing 48 GB of
system RAM. The common way to run a model larger than VRAM is to offload whole
*layers* to the CPU. That is expensive for every token, because every token
passes through every layer.

Mixture-of-Experts (MoE) models change the trade-off. Each token activates only
a few experts per layer, and the experts hold most of the weights. The
attention, router and shared weights are small, and every token uses them, so
they belong in VRAM. The expert weights are large and each token touches only a
few of them, so they can live in system RAM (and, later, on NVMe) with
bounded cost.

The same idea applies to models that are not too big for VRAM. Moving experts
out frees VRAM for more context (KV cache) or for a second resident model.

## Step 1 — tensor placement overrides (implemented)

The llama.cpp backend accepts ordered `(regex, device)` overrides, applied at
load time through llama.cpp's `tensor_buft_overrides`. Tensor names matching a
pattern are allocated on that device. The first match wins, and everything else
follows `--gpu-layers`.

```sh
# All layers on the GPU, MoE expert weights in system RAM.
sonder-infer serve --backend llamacpp --model-dir <models-dir> --model qwen3-30b-a3b.gguf \
    --gpu-layers -1 --moe-experts cpu

# The general form: any regex, any llama.cpp device ("cpu", "Vulkan0", "CUDA0").
sonder-infer serve ... --tensor-override "blk\.(3[0-9]|4[0-7])\.ffn_.*_exps=cpu"
```

- **`--moe-experts cpu`** expands to the pattern `\.ffn_(up|down|gate)_(ch|)exps` on `cpu`. This matches llama.cpp's own `--cpu-moe` choice: routed experts move to RAM, while the router (`ffn_gate_inp`) and the shared experts (`*_shexp`) stay on the GPU.
- **Validation happens before any file access.** The pattern must be a valid regex and the device must exist in this build; a bad flag fails at startup with `invalid_argument`. The number of overrides is capped by `llama_max_tensor_buft_overrides()`.
- **Library users** set `LlamaCppBackendOptions::tensor_overrides`. At the wrapper level the field is `LoadOptions::tensor_overrides`.

### Measurements

See the "Measured" section at the end of this file. Numbers are recorded only
from real runs.

## KV cache types, flash attention and micro-batch (implemented)

The KV cache competes with weights for VRAM, and its size grows linearly with
context. The llama.cpp backend exposes llama.cpp's own context knobs so an
operator can trade a little accuracy for a lot of KV memory. With no KV-type
flags, `serve` resolves `auto` per model (`q8_0` K and V when the model's head
dimensions are multiples of 32 and flash attention is not `off`, otherwise `f16`;
flash attention `auto`, micro-batch = llama.cpp's 512 capped at the batch size).

| `serve` flag | `LlamaCppBackendOptions` | Values | Default |
|---|---|---|---|
| `--cache-type-k T` | `kv_cache_type_k` | `auto`, `f16`, `f32`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0` (needs `--allow-q4-k-cache` + rotation), `iq4_nl`, `f8_e4m3` (CUDA-only) | `auto` (`q8_0` if head dims % 32 == 0 and `--flash-attn != off`, else `f16`) |
| `--cache-type-v T` | `kv_cache_type_v` | `auto`, `f16`, `f32`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`, `f8_e4m3` (CUDA-only) | `auto` (`q8_0` if head dims % 32 == 0 and `--flash-attn != off`, else `f16`) |
| `--allow-q4-k-cache` | `kv_k_q4_allowed` | opt-in flag enabling `q4_0` for K when model supports Hadamard K rotation | `false` |
| `--flash-attn MODE` | `flash_attention` | `auto`, `on`, `off` | `auto` |
| `--batch-size N` | `batch_size` | tokens per `llama_decode` during prefill | 512 |
| `--ubatch-size N` | `ubatch_size` | physical micro-batch, `<= --batch-size` | llama.cpp's 512, capped at the batch size |
| `--moe-cache-mib N` | `moe_cache_mib` | dynamic SLRU MoE expert VRAM cache in MiB (`0..262144`) | `0` (disabled) |
| `--moe-cache-cpu-share F` | `moe_cache_cpu_share` (`llamaserver` profile/args; rejected on direct `llamacpp`) | fraction (`0.0..1.0`) of SLRU cache misses executed directly on CPU (`llama-server` only) | unset |
| `--cache-ram-mib N` | `cache_ram_mib` | host RAM prompt-cache budget in MiB (`0..65536`) | unset |
| `--mmproj PATH` | `mmproj_path` | multimodal projector GGUF path | unset |
| `--spec-type TYPE` | `spec_type` | `none`, `ngram-simple`, `ngram-mod`, `ngram-cache`, `suffix`, `draft` | `none` |

Bytes per cached value, from ggml's block layouts (32 values per block), and
the effect on the 384 MiB f16 KV cache measured below (4096 context):

| Type | Bytes/value | vs f16 | 384 MiB f16 becomes |
|---|---|---|---|
| `f32` | 4 | 2× | 768 MiB |
| `f16`, `bf16` | 2 | 1× | 384 MiB |
| `q8_0` | 1.0625 | ≈ ½ | ≈ 204 MiB |
| `f8_e4m3` (`llamaserver`) | 1.0000 | ½ | 192 MiB |
| `q5_1` | 0.75 | ≈ 0.38 | ≈ 144 MiB |
| `q5_0` | 0.6875 | ≈ 0.34 | ≈ 132 MiB |
| `q4_1` | 0.625 | ≈ 0.31 | ≈ 120 MiB |
| `q4_0`, `iq4_nl` | 0.5625 | ≈ ¼ | ≈ 108 MiB |

`llamaserver` launch profiles additionally support `f8_e4m3` (1.0 B/value, ½ of
`f16`; selectable in `plan_model_placement()` via `prefer_fp8_kv = true` on
discrete NVIDIA RTX GPUs), `q2_0` (0.28125 B/value, 18/64), and `pq2_0m`
(0.28125 B/value, 36/128, PrismML KV-only) when backed by a matching
`llama-server` build; direct `llamacpp` (`b11195`) keeps the nine types in the
table above. See [GPU_SUPPORT.md](GPU_SUPPORT.md#quantized-kv-cache-types).

K and V are sized independently, so an **asymmetric `q8_0/q4_0` KV cache**
(`--cache-type-k q8_0 --cache-type-v q4_0`) saves ~24% KV VRAM relative to
`q8_0/q8_0` while keeping attention keys at 8-bit precision (keys are far more
quantization-sensitive than values). The placement planner always tries
`q8_0/q4_0` as an intermediate step before dropping both K and V to `q4_0/q4_0`.
The table is arithmetic from the formats, not a measurement; quality effects are
model dependent and not yet measured here.
One data point (a second host, CPU, a 491 MB GGUF, 512 context, greedy, 16 tokens,
2026-09-30): `q8_0` K and V produced the same text as `f16`; `q4_0` K and V
with flash attention on degenerated into a repeated token. Treat 4-bit V
caches as something to evaluate per model, not a free default.

Rules, checked at startup (`invalid_argument`, before any model file is read):

- **A quantized V cache needs flash attention.** llama.cpp `b11195`
  (`llama_init_from_model`) switches `auto` to enabled for a quantized V
  cache and refuses one with flash attention disabled, so Sonder rejects
  `--cache-type-v q8_0 --flash-attn off` up front. A quantized K cache alone
  works without flash attention.
- **`--ubatch-size` must not exceed `--batch-size`.**
- Unknown type or mode names are rejected (case-insensitive; `on`/`enabled`
  and `off`/`disabled` are synonyms).
- **These five flags apply to `--backend llamacpp` only.** Any other backend
  would run without them, so `serve` exits with status 2 and
  `option --X applies only to --backend llamacpp; for llamaserver set it in the JSON args`.
  A spawned llama-server takes `--cache-type-k` and the rest in its
  `--llamaserver-config` `args`, or from a
  [launch profile](integration/launch-profiles.md)'s typed fields; with
  `--profile` the five flags are refused because the profile sets them.

Some limits depend on the model and are left to llama.cpp, which then fails
the load: MLA models (DeepSeek-style) need identical K and V types, a
quantized type's block size must divide the head size, and Grok models force
flash attention off (so a quantized V cache cannot load for them).

```sh
# Half the KV bytes, flash attention decided by llama.cpp (on for quantized V).
sonder-infer serve --backend llamacpp ... --cache-type-k q8_0 --cache-type-v q8_0
```

## Multi-GPU, vendor targeting, and memory locking (implemented)

The direct `llamacpp` backend exposes multi-GPU and vendor device placement on `LlamaCppBackendOptions` (`sonder/inference/backends/llamacpp.hpp`) and `LoadOptions` (`sonder/backends/llamacpp/llamacpp_backend.h`). All fields are additive and default to `llama.cpp`'s existing behavior. (`llamaserver` launch profiles pass the corresponding `--device`, `--main-gpu`, `--split-mode`, `--tensor-split`, and `--load-mode` flags to `llama-server`.)

| `LlamaCppBackendOptions` | `LoadOptions` | Values / constraints | Default | `llama_model_params` (`v1.0.0`) |
| --- | --- | --- | --- | --- |
| `devices` | `devices` | exact ggml device allowlist (`CUDA0`, `CUDA1`, `ROCm0`, `Vulkan0`, `CPU`), case-insensitive, max 16; per-model override via `ModelLoadOptions::devices` and launch profile `devices` | empty (`SelectDevices` auto: discrete GPUs only, never iGPUs, native backend preferred over Vulkan) | null-terminated `mp.devices` |
| `main_gpu` | `main_gpu` | integer `>= 0`; primary GPU for `split_mode = none` and non-split/scratch tensors (re-indexed when duplicates/iGPUs are filtered) | `0` | `mp.main_gpu` |
| `split_mode` | `split_mode` | `layer`, `none`, `row` (case-insensitive) | `layer` | `mp.split_mode` (`LLAMA_SPLIT_MODE_*`) |
| `tensor_split` | `tensor_split` | up to 16 finite floats `>= 0`, at least one `> 0` when non-empty (re-indexed when duplicates/iGPUs are filtered; rejected when no discrete GPU is selected) | empty (equal / default) | `mp.tensor_split` (zero-padded to `llama_max_devices()`) |
| `use_mlock` | `use_mlock` | boolean; locks weights in RAM | `false` | `mp.load_mode` (`LLAMA_LOAD_MODE_MMAP_MLOCK` with `use_mmap`, `LLAMA_LOAD_MODE_MLOCK` without) |

Rules, checked before any model file is read (`invalid_argument`):

- **Pure validation (`validate_llamacpp_options` / `ValidatePlacementOptions`)** rejects `main_gpu < 0`, unknown `split_mode`, empty or NUL-containing device names, more than 16 devices or split entries, and `tensor_split` entries that are NaN, infinite, negative, or all zero.
- **Explicit allowlist, automatic iGPU exclusion, and cross-backend deduplication (`SelectDevices` / `ResolveModelDevices` / `ResolveGpuDevicePin`)**:
  - When `devices` is empty, `SelectDevices` chooses discrete GPUs (`kGpu`) only — never an integrated UMA GPU (`kIntegratedGpu`, e.g. Radeon 780M) — and collapses duplicate views of the same physical card across backends (e.g. `CUDA0` + `Vulkan0`, or `ROCm0` + `Vulkan0`, matched via normalized PCI bus ID or cross-backend description + memory) in favor of the native vendor backend. `gpu:N` ordinals index only this de-duplicated discrete GPU set (and non-UMA OS GPUs).
  - When `devices` is non-empty (via `LlamaCppBackendOptions::devices`, per-model `ModelLoadOptions::devices`, or launch profile `devices`), names match case-insensitively, explicitly named integrated GPUs are honored, and duplicate views of the same card collapse to the first named entry.
  - `tensor_split` and `main_gpu` are re-indexed onto the surviving selected device list whenever duplicate views or unlisted iGPUs are filtered out, and `tensor_split` or `main_gpu > 0` without a discrete GPU (and without an explicit allowlist) fails fast with `invalid_argument`. Multimodal projectors (`--mmproj`) are pinned inside the resolved accelerator allowlist via `ProjectorDevice`.
- **Load mode mapping in `b11195`**: `llama.cpp` `b11195` folds mmap and mlock into `llama_model_params::load_mode`. Leaving `use_mmap = true` and `use_mlock = false` preserves `LLAMA_LOAD_MODE_AUTO` (so backends without `mmap_support` still fall back automatically); `use_mmap = false, use_mlock = false` sets `LLAMA_LOAD_MODE_NONE`; `use_mlock = true` selects `LLAMA_LOAD_MODE_MMAP_MLOCK` (with `use_mmap = true`) or `LLAMA_LOAD_MODE_MLOCK` (with `use_mmap = false`).

## Vendor and low-bit placement notes

- **NVIDIA RTX (Ampere `sm_86`, Ada `sm_89`, Blackwell `sm_120`)**: build with `windows-cuda-rtx` or `linux-cuda-rtx` (`GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86;89;120`, `GGML_CUDA_FA_ALL_QUANTS=ON`). `GGML_CUDA_FA_ALL_QUANTS=ON` ensures quantized KV caches (`q8_0`, `q4_0`, etc.) execute with CUDA Flash Attention rather than failing or falling back. On mixed RTX + iGPU workstations, `llama.cpp`'s default selection skips integrated GPUs when a discrete GPU is present; set `devices = {"CUDA0"}` (or `{"CUDA0", "CUDA1"}` with `split_mode` and `tensor_split`) to pin placement explicitly.
- **AMD Radeon RX XT (RDNA 2/3/4) and Ryzen APU UMA**: build with `windows-rocm-xt` / `linux-rocm-xt` (`GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON`, targeting `gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1200;gfx1201`) for native ROCm (`ROCm0`) with WMMA FlashAttention on RDNA 3/4, or `windows-vulkan-xt` / `linux-vulkan-xt` (`GGML_VULKAN=ON`, `Vulkan0`) on Windows and non-ROCm Linux hosts. On RDNA 4 (`gfx1200`/`gfx1201`, RX 9060 / RX 9070 XT), Vulkan `VK_KHR_cooperative_matrix` is a competitive or preferred fallback alongside ROCm; `plan_model_placement()` surfaces this advisory and `sonder-infer tune` sweeps `flash_attn` (`on`/`auto`) across both hints. Discrete RX XT cards budget against dedicated VRAM, whereas Ryzen APUs (`gfx1103`, e.g. Radeon 780M/890M) expose only a small BIOS carveout as dedicated VRAM and run primarily out of shared UMA/GTT memory; UMA hosts should not treat shared GPU allocations as PCIe spill, and `LLAMA_LOAD_MODE_AUTO` handles devices without `mmap_support` automatically.
- **Ternary, 1-bit, and FP4/FP8 GGUFs**: compact block formats shrink weight footprints dramatically — `Q1_0` (18 B / 128 elems, ~1.125 bpw), `PTQ1_0` (28 B / 128 elems, ~1.75 bpw), `PQ2_0` (34 B / 128 elems, ~2.125 bpw), `PQ2_0M` (36 B / 128 elems, KV-only), `Q2_0` (18 B / 64 elems, ~2.25 bpw), `MXFP4` (17 B / 32 elems, 4.25 bpw), `NVFP4` (36 B / 64 elems, 4.50 bpw), and `F8_E4M3` (1 B / elem, 8.00 bpw weights / 1.0 B/elem KV). Because weights occupy much less VRAM per parameter, KV cache sizing (`--cache-type-k`/`--cache-type-v`, Flash Attention) and context budget become the dominant VRAM lever on 16 GB cards. Running these files requires a matching runtime build: mainline `llama.cpp >= b11556` for stock `Q2_0`/`Q1_0`/`NVFP4`/`F8_E4M3`, or `PrismML-Eng/llama.cpp` (`prism` branch) for Hadamard-transformed `PQ2_0`/`PTQ1_0`/`PQ2_0M` (`prism.hadamard.*`); see [GPU_SUPPORT.md](GPU_SUPPORT.md) and [BACKENDS.md](BACKENDS.md#upstream-llamacpp-pin-and-ternary--prismml-overrides).

## Step 2 — per-model and co-residency placement planner (implemented)

Header-only placement planning is implemented in
`src/server/include/sonder/inference/placement_planner.hpp`
(`plan_model_placement()`, `placement_plan_to_json()`) and
`src/server/include/sonder/inference/coresidency_planner.hpp`
(`plan_coresident_models()`, `coresidency_plan_to_json()`), exposed through
`sonder-infer plan` ([CLI.md](CLI.md)) and `POST /v1/sonder/plan`
([SERVER.md](SERVER.md#post-v1sonderplan)). It reads only GGUF headers plus
live GPU inventory (DXGI, NVML, or `amdgpu` sysfs) and host memory without
loading model weights:

| Field | Source |
|---|---|
| total weight bytes, per tensor class (attention, dense FFN, experts, shared, embeddings) | GGUF tensor table (`gguf_header.cpp`) |
| experts per layer / experts used per token | `<arch>.expert_count`, `<arch>.expert_used_count` (`routed_experts`) |
| KV bytes per token at a given context | attention-layer count × KV head dims × dtype, with sliding-window layers sized to their window and recurrent layers excluded |
| recurrent-state & rollback snapshot bytes | `<arch>.ssm.*` state per sequence × `(1 + draft_n_max)` when rollback speculation (`draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`) is enabled |
| PrismML Hadamard & low-bit tensor traits | `prism.hadamard.*` keys and `ggml_type` IDs (`40`–`43`, `142`, `143`, `145`) |

- **Single-model planner (`plan_model_placement`)**: selects the target GPU
  (preferring discrete GPUs over integrated UMA unless pinned by `device_hint`),
  applies vendor defaults (NVIDIA RTX `flash_attn=on`, `load_mode=none`; AMD RX
  XT `flash_attn=auto`, `GPU_MAX_ALLOC_PERCENT=100`; Ryzen APU `uma_mode=true`
  with host-RAM pool cap and bandwidth warning), auto-expands to same-vendor
  multi-GPU `split_mode=layer` + proportional `tensor_split` when a single GPU
  cannot fit, flags `requires_prism_runtime` on Hadamard/PrismML GGUFs, enables
  `draft-mtp` (when `nextn_layers > 0`) or `ngram-mod` (on ternary/1-bit
  models), and runs a deterministic iterative fit solver via `estimate_vram()`:
  asymmetric `q8_0/q4_0` KV -> symmetric `q4_0/q4_0` KV -> binary-searched
  `--n-cpu-moe N` (restoring `q8_0/q8_0`, or else `q8_0/q4_0`, if it fits at
  the same `N`) -> MTP rollback trim -> 4096-aligned `ctx_size` step-down ->
  MTP drop / CPU `mmproj` -> binary-searched `n_gpu_layers` -> dynamic SLRU
  MoE expert cache (`--moe-cache-mib` + `--moe-cache-cpu-share`) -> `no_repack`
  / CPU-only fallback -> selective CPU-only repacking guard & physical-core
  thread recommendation (`recommended_threads`).
- **Dynamic SLRU MoE Expert Cache (`--moe-cache-mib` + `--moe-cache-cpu-share`)**:
  enforces two anti-regression guards from upstream `llama.cpp` PR #29887 /
  #30966 / #30637:
  1. **Never enabled on 100% VRAM-resident MoE models** (`host_expert_bytes == 0`), avoiding ~10% graph-splitting overhead.
  2. **Enabled on `--n-cpu-moe` VRAM overflow only when spare VRAM >= 512 MiB and covers the active CPU-offloaded expert working set** (avoiding `<75%` hit-rate PCIe thrash), skipping integrated UMA where GPU and host share memory. On heavily overflowing fine-grained sparse MoEs (e.g., an 80B MoE on 16 GB RX 9060 / RTX 5070 Ti + 48 GB RAM), the planner upgrades `<50%` static block pinning to full `--n-cpu-moe` + dynamic SLRU cache and sets `moe_cache_cpu_share` (`0.25–0.50` when host DDR5 RAM is ample).
- **Selective CPU-only repacking & physical-core decode threads**:
  `plan_model_placement()` never emits global `--repack` (`no_repack = false`)
  while GPU layers remain active (`repack_policy: "cpu_only_selective"`),
  avoiding row-interleaved GPU CPU-fallback regressions. Whenever MoE experts
  are CPU-offloaded (`--n-cpu-moe > 0`), it recommends physical core count
  (`recommended_threads` and `profile.threads`: `cpu_physical_cores`, or
  `logical_cores / 2` clamped `>= 4` and `<= logical_cores`) so memory-bandwidth-bound
  DDR5 decode does not oversubscribe SMT hyperthreads.
- **Multi-model co-residency planner (`plan_coresident_models`)**: invoked via
  repeatable `sonder-infer plan --model A.gguf --model B.gguf`
  (`sonder.inference.coresidency_plan/1`) and mirrored by multi-profile
  aggregate VRAM gating in `sonder-infer serve --profile A --profile B` /
  `--all-profiles`. Schedules models sequentially in input order, reserves
  OS/desktop `vram_headroom_mib` once per occupied GPU (not double-counted per
  co-resident model), deducts discrete VRAM, multi-GPU split shards, and UMA
  shared host RAM without double-counting, and overflows unpinned models to a
  second GPU, iGPU, or CPU when GPU 0 fills.
- **Telemetry follow-up**: publishing the chosen per-tensor-class placement in
  `model.loaded` telemetry for Observatory remains a follow-up.

```sh
# Single-model LaunchProfile + VRAM fit plan (header-only)
sonder-infer plan --model model.gguf --ctx 32768 --json

# Multi-model co-residency plan across host GPUs and RAM
sonder-infer plan --model bonsai-27b.gguf --model bonsai-9b-mtp.gguf --json
```

### Routed-expert header observations

`GgufModelInfo::routed_experts` is an optional typed pair of declared
`total` (`expert_count`) and `active` (`expert_used_count`) counts. The parser
requires exactly one string `general.architecture` with namespace syntax
`[A-Za-z0-9_][A-Za-z0-9_.-]*` (at most 256 bytes), then exactly one scalar GGUF
integer for each literal `<architecture>.expert_count` and
`<architecture>.expert_used_count` key. It requires
`0 < active <= total <= UINT32_MAX`. Missing, duplicate, wrongly typed or
out-of-range fields leave the pair unset. Arrays and the string metadata
view's array maxima are not scalar count evidence. Dots and case in the
namespace are literal; `qwen3.8` is not an alias for `qwen3_8`.

This observation is independent of `ModelArchitecture`, which describes the
attention/hybrid/recurrent cache and state axis. A previously unknown family
can supply valid count metadata without changing its existing cache
classification. An unset pair means unknown, not dense; model and family
names do not establish sparsity. Counts do not establish layer topology,
active parameter totals or measured routing frequencies.

The pair describes only the supplied GGUF header bytes. Neither
`parse_gguf_model_info` nor `read_gguf_model_info` binds those bytes to an
immutable full-model revision, and a header hash would not be a full-model
hash. `plan_model_placement()` uses `routed_experts` together with the GGUF
tensor table solely to decide whether to search `--n-cpu-moe` expert-offload
candidates (and `estimate_vram()` uses the declared active/total ratio for
mmapped host-expert residency estimates); the pair is not propagated to
`ModelDescriptor`, the C ABI, or Runtime routing policy, and does not establish
measured routing frequencies or model/task quality. Backend execution
capability bits and Runtime task-quality evidence remain separate from physical
model traits. No model-quality, inference-speed or hardware recommendation
follows from synthetic header tests.

## Step 3 — NVMe as a cold tier (design)

With `mmap`, expert tensors that are never paged in cost no RAM. For very large
MoE models (for example 150 GB with 10–15 GB active per token), cold experts can
stay on NVMe and be paged in on first use. PCIe 4 NVMe reads at about 5–7 GB/s,
roughly 10× slower than DDR5. That is only viable for experts the router rarely
selects, so this step needs expert-usage telemetry (routing histograms) first.
Without data it is guesswork.

## Step 4 — Sonder-wide device scheduling (design)

Local node capability inventory is now implemented (`sonder-infer node` and
`GET /v1/sonder/node`, schema `sonder.inference.node/1`) so each host reports
its CPU, host memory, GPUs, backends, models, profiles, and feature flags.
Multi-host scheduling remains design work:

In the runtime, the Ollama pool already schedules whole requests across hosts
(for example, the reasoning tier served by a second host's integrated GPU while the workstation
keeps its resident dense model). That pattern of distributed agents, rather
than tensors split across the network, remains the right one over 2.5 GbE. The
link is roughly 300 MB/s, against hundreds of GB/s for VRAM.

The extension is a placement-aware scheduler that knows, per device:

- compute (GPU/iGPU/NPU/CPU);
- memory (VRAM, shared iGPU memory, RAM, NVMe);
- links between devices (PCIe, the node link).

It then chooses both *which host* serves a request and *how* that host places
the model. Small always-on models (embeddings, reranker, router, speculative
draft) are natural tenants for an iGPU or NPU, which frees the discrete GPU for
the main model.

## Measured

**2026-09-27, workstation.** A 16 GB discrete GPU, llama.cpp `b11195` with Vulkan,
MSVC RelWithDebInfo. Model: `qwen3-coder:30b-a3b` Q4_K_M, a 17.3 GiB GGUF (larger
than VRAM). Command:

`serve --backend llamacpp --device gpu:0 --gpu-layers -1 --context-length 4096 --moe-experts cpu`

| | Result |
|---|---|
| Layers offloaded | 49/49 |
| Weights in VRAM (`Vulkan0` model buffer) | **784 MiB** |
| KV cache in VRAM (4096 ctx) | 384 MiB |
| Expert weights in mapped system memory | 17,681 MiB |
| Output (temperature 0, 2 runs) | correct and identical |
| Decode, warm | about 3.6 tok/s (71 tokens in 19.9 s) — **not representative** |

**Placement works as designed.** A 17 GB MoE runs with about 1.2 GB of VRAM,
leaving room for a resident dense model or a much longer context.

**The speed figure is memory-starved.** Only about 3 GB of RAM was free (a WSL
test VM held 14 GB), so expert pages streamed from NVMe on every token. A
representative number needs about 18 GB of free RAM. The next measurement to
take is this same model with free RAM, comparing `--moe-experts cpu` against
layer offload (`--gpu-layers N` with no override) at equal VRAM.

**This also found a gap.** `serve` never passed a device to model loads, so a
llama.cpp model served by `sonder-infer serve` always ran CPU-only. `--device
gpu:0` now fixes that.

## Opt-in inspected-content foundation

The private server-module `server::detail::observe_gguf_artifact` entrypoint
binds the existing literal GGUF header interpretation and optional typed
routed-expert counts to a caller-trusted full SHA256 of the same inspected
stream. Success returns the matching digest and inspected byte length; missing
or ambiguous counts remain unknown. The existing header-only APIs still make
unbound header observations. See [the contract and local qualification](integration/gguf-artifact-observation.md).

The observer is opt-in and has no default serving or placement call site. It
does not populate model descriptors, backend identities, HTTP metadata,
telemetry or cache/routing policy. Its strict logical read/header allowances
and cooperative cancellation/deadline checks do not establish OS-prefetch,
heap/RSS or hard-latency bounds. A matching digest does not validate tensor
extents or model loadability, authenticate the expectation, or associate the
inspected content with bytes actually loaded by a backend.

Backend-loaded artifact association and separately qualified descriptor/HTTP
projection are still prerequisites for consuming this evidence in placement
or routing. Existing placement rules and residency controls remain unchanged;
physical expert counts do not imply model/task quality or execution speed.
