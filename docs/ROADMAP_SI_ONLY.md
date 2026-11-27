# SI-Only Inference Roadmap

This document tracks the transition to **Sonder Inference (SI) as the sole
local inference engine** across the Sonder ecosystem.

## Decision and rationale

As of 2026-11-26, the local inference architecture consolidates onto SI's
in-process `llama.cpp`/GGML backend (`src/backends/llamacpp`):

- **Sole local engine:** `src/backends/llamacpp`, moving to a single vendored,
  pinned `ggml`/`llama.cpp` source tree compiled into `sonder-infer` with SI's
  own patch series on top (no separate forks or standalone server binaries).
- **Retiring:** local Ollama daemon usage, SI's `ollama` and `llamaserver`
  backends (both spawned and attached external `llama-server` processes), and
  standalone fork/router servers.
- **Kept:** Ollama Cloud models only, reached directly by SI over opt-in HTTPS
  with no local Ollama daemon.
- **Why:** SI must own scheduling, memory budgets, and telemetry end to end.
  Piggy-backed external servers double-load or obscure memory ownership and
  hide execution internals (KV pool state, token probabilities, layer/operator
  activity) from the Sonder Observatory.

Whole-request placement across multiple SI instances remains owned by Sonder
Runtime per ADR-021
([integration/federation-worker-contract.md](integration/federation-worker-contract.md)).

## Worklist status

Statuses below are checked against `git log origin/main` (at `df2fb11`,
2026-11-26). An item is marked `merged` only when the full item has landed on
`origin/main`.

| Item | What | Status | Current state and references |
| --- | --- | --- | --- |
| **RET-01** | Native backend as sole local engine on CUDA (`cuda-blackwell`) and Vulkan (`vulkan-xt`) | in progress | Direct `llamacpp` backend, CUDA/Vulkan presets, continuous batching, L2 radix cache, speculation, tool chat (`#124`), lazy grammar (`#128`, `#130`), `<think>` splitter (`#129`), and the `v1.0.0` upstream pin move (`#134`, see RET-01a below) are on `origin/main`. Live `cuda-blackwell` and `vulkan-xt` parity qualification remain. See [GPU_SUPPORT.md](GPU_SUPPORT.md) and [integration/llamacpp.md](integration/llamacpp.md). |
| **RET-01a (NAT-01)** | Move `llama.cpp` upstream pin past `b11195` to `v1.0.0` | merged | Merged on `origin/main` (`df2fb11`, `#134`). Pin moved to `v1.0.0` (commit `b1b770d`) via `cmake/llamacpp-pin.json` with configure-time `expected_features` enforcement; `b11195` survives only as a legacy tag in `src/backends/llamacpp/CMakeLists.txt`. |
| **RET-02** | Ollama Cloud provider over HTTPS | in progress | Not merged on `origin/main`. Opt-in direct HTTPS client to Ollama Cloud with no local daemon, building on optional `src/net` TLS ([integration/tls.md](integration/tls.md)). |
| **RET-03** | Native embeddings endpoint | in progress | `POST /v1/embeddings` and batched direct `llamacpp` embeddings (`src/backends/llamacpp/embed_batch_planner.hpp`, plus `/v1/rerank` and `/v1/score`) are on `origin/main` (`46c8a3e`, `#121`; [SERVER.md](SERVER.md#post-v1embeddings)). Low-priority stream gating and cutover of Runtime's embedding model to native GGUF remain. |
| **RET-04** | Model store importing an existing Ollama library in place | planned | Not on `origin/main`. Resolves existing Ollama manifests and blobs into an SI registry in place (no weight copies). |
| **RET-05** | Sonder-runtime migration off local Ollama | planned | Cross-repo migration in Sonder Runtime (embeddings, residency, cloud routing, training LoRA deployment, config); depends on RET-01 through RET-04 and RET-06. Worker contract: [integration/federation-worker-contract.md](integration/federation-worker-contract.md). |
| **RET-06** | LoRA adapters in the native backend | planned | Not on `origin/main` (`Capability::lora` is declared in `include/sonder/inference/backend.hpp`, not yet implemented in `src/backends/llamacpp`). Required by Runtime's training path in RET-05. |
| **RET-07** | Removal of the `ollama` and `llamaserver` backends | planned | Both backends are marked deprecated in [README.md](../README.md) and [src/backends/README.md](../src/backends/README.md). Code removal follows RET-01 through RET-06, keeping only the RET-02 cloud provider. |
| **RET-08** | Cutover and bring-up | planned | Final operational cutover to native SI across main and secondary hosts, disabling local Ollama and verifying all Observatory inference panels ([OBSERVATORY_CONTRACT.md](OBSERVATORY_CONTRACT.md), [TELEMETRY.md](TELEMETRY.md)). |
| **RET-09** | One vendored, pinned `ggml`/`llama.cpp` tree with SI's own patch series; no separate forks or binaries; stock `Q2_0` preferred for Bonsai if it reaches parity, otherwise `PQ2_0` kernels carried as SI patches | planned | Current state on `origin/main`: FetchContent of `v1.0.0` (commit `b1b770d`, `#134`) via the pin manifest (`cmake/llamacpp-pin.json`), with `b11195` surviving only as a legacy tag in `src/backends/llamacpp/CMakeLists.txt`. Replaces standalone fork/server binaries with one SI-owned vendored tree and patch series; licensing review and owner approval are tracked as pending in [LICENSE_REVIEW.md](LICENSE_REVIEW.md). |

## RET-07 — helper extraction and retirement

PR-1 is helper extraction only. Keep the placement planner's default backend
as `llamaserver` (`src/server/src/placement_planner.cpp:978`); defer the
`llamacpp` default flip until the native profile mapping below is supported
and tested.

### Follow-up: native profile mapping before the default flip

A backend-name-only change produces profiles that the native loader rejects.
The blocking findings, verified against the PR-1 source at `df2fb11`, are:

- **Fit:** the planner always assigns `profile.fit = false`
  (`src/server/src/placement_planner.cpp:997`).
  `apply_llamacpp_profile` rejects any engaged `fit` optional, including
  `false` (`src/server/src/launch_profile.cpp:820`).
- **Device and split:** the planner emits `device`, `split_mode`,
  `tensor_split`, and sometimes `main_gpu`
  (`src/server/src/placement_planner.cpp:1385`,
  `src/server/src/placement_planner.cpp:1387`,
  `src/server/src/placement_planner.cpp:1392`,
  `src/server/src/placement_planner.cpp:1395`).
  The native loader rejects these fields
  (`src/server/src/launch_profile.cpp:793`,
  `src/server/src/launch_profile.cpp:794`,
  `src/server/src/launch_profile.cpp:795`,
  `src/server/src/launch_profile.cpp:796`).
- **Loading and repacking:** planner paths set `load_mode` and `no_repack`
  (`src/server/src/placement_planner.cpp:1111`,
  `src/server/src/placement_planner.cpp:1274`,
  `src/server/src/placement_planner.cpp:2233`).
  The native loader rejects both
  (`src/server/src/launch_profile.cpp:797`,
  `src/server/src/launch_profile.cpp:798`).
- **MoE placement and cache:** the planner emits `--n-cpu-moe` in
  `extra_args` and sets `moe_cache_mib` / `moe_cache_cpu_share`
  (`src/server/src/placement_planner.cpp:916`,
  `src/server/src/placement_planner.cpp:2210`,
  `src/server/src/placement_planner.cpp:2211`).
  The native loader rejects the cache fields and non-empty `extra_args`
  (`src/server/src/launch_profile.cpp:799`,
  `src/server/src/launch_profile.cpp:800`,
  `src/server/src/launch_profile.cpp:836`).

Before changing the default, implement equivalent native mappings or make
profile generation backend-aware without silently dropping placement or
memory-policy semantics. Add coverage that passes generated profiles directly
to `apply_llamacpp_profile`, including CPU, single-GPU, multi-GPU, and MoE
cases. Converting a copy back to `llamaserver` for argv checks does not prove
native compatibility.

## Sequencing

1. **Core engine consolidation (RET-01, RET-09):** build on the merged `v1.0.0`
   pin move (`#134`) to vendor the pinned `ggml`/`llama.cpp` tree with SI's
   patch series, validate stock `Q2_0` versus carried `PQ2_0` patches for
   Bonsai, and qualify native `cuda-blackwell` and `vulkan-xt` parity.
2. **Feature closure (RET-02, RET-03, RET-04, RET-06):** land the Ollama Cloud
   HTTPS provider, finish native embeddings cutover, import existing Ollama
   model libraries in place, and add native LoRA adapter lifecycle.
3. **Ecosystem migration (RET-05):** move Sonder Runtime off local Ollama onto
   SI native endpoints, LoRA loading, and the SI cloud provider.
4. **Retirement and cutover (RET-07, RET-08):** remove the deprecated `ollama`
   and `llamaserver` backends, bring up native-only SI across hosts, and verify
   Observatory telemetry end to end.

## Related documentation

- [GPU, ternary/1-bit, and APU hardware support](GPU_SUPPORT.md) — build
  presets, GGUF tensor formats, and CUDA/ROCm/Vulkan/UMA details
- [Upstream license and dependency review](LICENSE_REVIEW.md) — Apache-2.0
  project license, MIT terms for upstream `ggml`/`llama.cpp`, pending
  vendoring/patch-series review, and third-party review rules
- [General roadmap](ROADMAP.md) and [backend strategy](BACKENDS.md) — broader
  engine milestones and capability model
- [HTTP server reference](SERVER.md) — current `/v1/*` routes and serve flags
- Integration contracts:
  [llamacpp](integration/llamacpp.md),
  [llama-server](integration/llama-server.md) (deprecated),
  [launch profiles](integration/launch-profiles.md),
  [VRAM spill](integration/vram-spill.md),
  [TLS](integration/tls.md), and
  [federation worker contract](integration/federation-worker-contract.md)
