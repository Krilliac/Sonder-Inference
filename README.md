# Sonder Inference

**Sonder Inference** is the proposed native inference runtime for the Sonder ecosystem.

The goal is not to immediately rewrite CUDA/Vulkan linear algebra from scratch. The goal is to let Sonder own **inference policy**: model lifecycle, sessions, scheduling, batching, KV/context management, memory pressure, device placement, speculative decoding policy, adapters, distributed execution, and telemetry—while initially reusing proven execution backends and kernels.

## Architectural thesis

```text
Sonder Runtime
      |
      v
+----------------------+
|   Sonder Inference   |
|----------------------|
| Session Manager      |
| Scheduler            |
| KV/Context Manager   |
| Model/Adapter Manager|
| Device Planner       |
| Sampling/Speculation |
| Telemetry            |
+----------+-----------+
           |
   +-------+------------------+
   |                          |
   v                          v
llama.cpp/GGML          specialized backends
(initial baseline)      FlashInfer / native later
   |                          |
 CPU/CUDA/Vulkan/HIP      CUDA/Vulkan/etc.
```

Inference engine != GPU kernel library. Sonder can become the owner of inference without first becoming the owner of every GEMM implementation.

## Why build it

Ollama is a strong convenience layer, but Sonder increasingly needs direct control over concerns that are difficult to express through an opaque request/response boundary:

- agent-aware priority scheduling and continuous batching
- shared/persistent KV and prefix reuse
- context compaction and session continuation
- hot/cold model and adapter residency
- memory-aware CPU/GPU/offload policy
- heterogeneous and distributed nodes
- speculative decoding chosen by workload/hardware
- MoE expert placement
- bounded retry/recovery and preemption
- deep, trustworthy Observatory telemetry

## Research baseline

The research catalog covers major engine/runtime families including:

- llama.cpp / GGML
- vLLM
- SGLang
- TensorRT-LLM
- Hugging Face TGI
- LMDeploy / TurboMind
- LightLLM
- DeepSpeed-MII / FastGen
- MLC LLM / WebLLM
- ExLlama
- KTransformers
- NInfer
- ik_llama.cpp
- ONNX Runtime GenAI
- ExecuTorch
- MLX / MLX-LM
- PowerInfer
- BitNet
- FastFlowLM
- Lemon MLX Engine
- MAX / Modular
- Candle / Rust-native engines
- LocalAI / vllm.cpp
- LMCache / Mooncake
- FlashInfer / FlashAttention
- research systems and papers including Orca, PagedAttention, Sarathi, DistServe, Splitwise, FastServe, FlexGen, Punica, S-LoRA, Preble, INFERCEPT, and related work

See `docs/RESEARCH_CATALOG.md` and `docs/RESEARCH_SOURCES.md`.

## Proposed phases

**Phase 0 — measurement and compatibility**
- retain Ollama path
- build benchmark corpus
- define engine API and Observatory protocol

**Phase 1 — Sonder-owned runtime**
- embed/control llama.cpp or GGML backend directly
- own model/session lifecycle and sampling
- expose structured telemetry

**Phase 2 — scheduler/cache**
- continuous batching
- prefix/KV reuse
- cache tiers and eviction policy
- agent/task priority scheduling
- adapter management
- speculative decode policy

**Phase 3 — heterogeneous/distributed**
- CPU/GPU placement
- Main/Node scheduling
- prefill/decode specialization where justified
- MoE hot/cold expert policy
- remote cache/session transport

**Phase 4 — selective native acceleration**
- replace individual operators or paths only when profiling demonstrates a material win
- consume proven kernel libraries where possible
- maintain fallback backends

## Non-goals

- Reimplement every model architecture and quantization before the runtime provides value
- Replace proven CUDA/Vulkan kernels for prestige
- Hard-code the architecture around one GPU vendor
- Sacrifice correctness/reproducibility for benchmark screenshots
- Make Observatory or agent orchestration a dependency of the low-level execution loop

## Repository map

- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_CATALOG.md`
- `docs/RESEARCH_SOURCES.md`
- `docs/SCHEDULER.md`
- `docs/KV_CACHE.md`
- `docs/BACKENDS.md`
- `docs/BENCHMARK_PLAN.md`
- `docs/OBSERVATORY_CONTRACT.md`
- `docs/DESIGN_DECISIONS.md`
- `docs/ROADMAP.md`
- `docs/LICENSE_REVIEW.md`
- `docs/SCAFFOLD.md`

## Status

Implementation has started (2026-09-26). The first slice provides:

- C++20 engine core: `Engine`, `Device` (CPU inventory), `Model` handle,
  `Session`, `Backend` interface with capability advertisement,
  `SamplingConfig` validation, cooperative cancellation, `Status`/`Result`.
- Stable C ABI: [`include/sonder_inference.h`](include/sonder_inference.h).
- Backends: a deterministic **mock** backend (tests only, performs no
  inference) and an **Ollama** compatibility adapter (loopback HTTP, streaming,
  cancellable).
- Observatory telemetry: JSONL events in the `sonder.observatory.event/1`
  envelope with a bounded, non-blocking queue.
- `sonder-infer` CLI and a benchmark harness skeleton.
- doctest + CTest suite; GitHub Actions on Windows and Linux.

An optional direct llama.cpp/GGML backend is available with
`SONDER_WITH_LLAMA_CPP=ON` (off by default). See [ROADMAP](docs/ROADMAP.md)
for exact status. No upstream project is copied or linked until its license is
verified and recorded in [LICENSE_REVIEW](docs/LICENSE_REVIEW.md).

## Build and test

Requirements: CMake 3.21+, Ninja, and a C++20 compiler (MSVC 2022+, GCC 11+,
or Clang 14+). The first configure downloads doctest for the tests (pinned
hash).

Windows (MSVC + Ninja; the script enters a VS developer environment):

```powershell
powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test
```

Linux/macOS:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

Presets: `msvc-debug`, `msvc-release`, `linux-debug`, `linux-release`, plus
`ci-windows`/`ci-linux` (warnings as errors). Build output goes to
`build/<preset>/`.

## Quickstart

```bash
# Inventory and backends
build/linux-debug/sonder-infer devices
build/linux-debug/sonder-infer backends

# Mock backend (deterministic, no model needed) with telemetry
build/linux-debug/sonder-infer generate --backend mock --model mock:tiny \
    --prompt "hello sonder" --max-tokens 16 --telemetry events.jsonl

# Local Ollama (http://127.0.0.1:11434) with a model you already pulled
build/linux-debug/sonder-infer models --backend ollama
build/linux-debug/sonder-infer generate --backend ollama --model qwen3:0.6b \
    --prompt "Say hi" --max-tokens 32 --telemetry events.jsonl

# Benchmark harness
build/linux-debug/sonder-infer bench --backend ollama --model qwen3:0.6b \
    --corpus bench/corpus/smoke.json --out results.json --warmup 1 --runs 3
```

`Ctrl-C` during `generate` cancels the in-flight request and emits a
`request.cancelled` event with the observed cancel latency.

## Layout

- `include/sonder/inference/` public C++ API; `include/sonder_inference.h` C ABI
- `src/` implementation (see [src/README.md](src/README.md))
- `tools/sonder-infer/` CLI
- `tests/` doctest suite and CTest CLI checks
- `bench/corpus/` prompt corpora; `bench/results/` small reviewed snapshots
- `cmake/`, `CMakePresets.json`, `scripts/build.ps1` build tooling
- Optional modules (`src/cache`, `src/scheduler`, `src/sampling`,
  `src/backends/llamacpp`, `src/backends/ollama`, `bench`) are auto-included
  when present; see [docs/MODULES.md](docs/MODULES.md)

## Local scaffold

The original scaffold notes are kept in [scaffold status](docs/SCAFFOLD.md),
now updated for the implementation layout.

## License

Sonder Inference is released under the [MIT License](LICENSE)
(Copyright (c) 2026 Krilliac). Third-party dependencies keep their own
licenses; each is recorded in [LICENSE_REVIEW](docs/LICENSE_REVIEW.md). All
currently adopted dependencies (llama.cpp/GGML, doctest, cpp-httplib) are MIT
and compatible with the project license.
