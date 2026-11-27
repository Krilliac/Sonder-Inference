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

Statuses below are checked against `git log origin/main` (at `c3df44d`,
2026-11-26). An item is marked `merged` only when the full item has landed on
`origin/main`.

| Item | What | Status | Current state and references |
| --- | --- | --- | --- |
| **RET-01** | Native backend as sole local engine on CUDA (RTX 5070 Ti) and Vulkan (RX 9060 XT) | in progress | Direct `llamacpp` backend, CUDA/Vulkan presets, continuous batching, L2 radix cache, speculation, tool chat (`#124`), lazy grammar (`#128`, `#130`), and `<think>` splitter (`#129`) are on `origin/main`. Upstream pin rebase past `b11195` and live CUDA/Vulkan parity qualification remain. See [GPU_SUPPORT.md](GPU_SUPPORT.md) and [integration/llamacpp.md](integration/llamacpp.md). |
| **RET-02** | Ollama Cloud provider over HTTPS | in progress | Not merged on `origin/main`. Opt-in direct HTTPS client to Ollama Cloud with no local daemon, building on optional `src/net` TLS ([integration/tls.md](integration/tls.md)). |
| **RET-03** | Native embeddings endpoint | in progress | `POST /v1/embeddings` and batched direct `llamacpp` embeddings (`src/backends/llamacpp/embed_batch_planner.hpp`, plus `/v1/rerank` and `/v1/score`) are on `origin/main` (`46c8a3e`, `#121`; [SERVER.md](SERVER.md#post-v1embeddings)). Low-priority stream gating and cutover of Runtime's embedding model to native GGUF remain. |
| **RET-04** | Model store importing an existing Ollama library in place | planned | Not on `origin/main`. Resolves existing Ollama manifests and blobs into an SI registry in place (no weight copies). |
| **RET-05** | Sonder-runtime migration off local Ollama | planned | Cross-repo migration in Sonder Runtime (embeddings, residency, cloud routing, training LoRA deployment, config); depends on RET-01 through RET-04 and RET-06. Worker contract: [integration/federation-worker-contract.md](integration/federation-worker-contract.md). |
| **RET-06** | LoRA adapters in the native backend | planned | Not on `origin/main` (`Capability::lora` is declared in `include/sonder/inference/backend.hpp`, not yet implemented in `src/backends/llamacpp`). Required by Runtime's training path in RET-05. |
| **RET-07** | Removal of the `ollama` and `llamaserver` backends | planned | Both backends are marked deprecated in [README.md](../README.md) and [src/backends/README.md](../src/backends/README.md). Code removal follows RET-01 through RET-06, keeping only the RET-02 cloud provider. |
| **RET-08** | Cutover and bring-up | planned | Final operational cutover to native SI across main and secondary hosts, disabling local Ollama and verifying all Observatory inference panels ([OBSERVATORY_CONTRACT.md](OBSERVATORY_CONTRACT.md), [TELEMETRY.md](TELEMETRY.md)). |
| **RET-09** | One vendored, pinned `ggml`/`llama.cpp` tree with SI's own patch series; no separate forks or binaries; stock `Q2_0` preferred for Bonsai if it reaches parity, otherwise `PQ2_0` kernels carried as SI patches | planned | Not on `origin/main` (`src/backends/llamacpp/CMakeLists.txt` still FetchContents tag `b11195`). Replaces standalone fork/server binaries with one SI-owned build and patch series. Licensing terms are tracked in [LICENSE_REVIEW.md](LICENSE_REVIEW.md). |

## Sequencing

1. **Core engine consolidation (RET-01, RET-09):** upgrade and vendor the
   pinned `ggml`/`llama.cpp` tree with SI's patch series, validate stock `Q2_0`
   versus carried `PQ2_0` patches for Bonsai, and qualify native CUDA and
   Vulkan parity.
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
  project license, MIT terms for `ggml`/`llama.cpp` and SI patches, and
  third-party review rules
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
