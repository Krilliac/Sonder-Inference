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

## Status

Research/architecture foundation. No upstream project should be copied or linked as a dependency until its current license and compatibility are verified.
