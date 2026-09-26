# Architecture

## Principle

Sonder Inference owns **inference policy and lifecycle** while execution backends own optimized tensor/operator implementations.

Do not start by rebuilding every kernel.

```text
Sonder Runtime
      |
      | Inference API
      v
+--------------------------------------------------+
|                 Sonder Inference                 |
|--------------------------------------------------|
| API / Session Manager                            |
| Scheduler / Admission / Priority / Preemption    |
| Context + KV Cache Manager                       |
| Model + Adapter Residency Manager                |
| Sampling + Structured Output                     |
| Speculative Decode Policy                        |
| Device / Placement Planner                       |
| Distributed Transport                            |
| Telemetry / Benchmark Hooks                      |
+---------------------------+----------------------+
                            |
                    execution interface
                            |
       +--------------------+---------------------+
       |                    |                     |
       v                    v                     v
 llama.cpp/GGML       optimized libraries      native paths
 initial baseline       e.g. FlashInfer       only if justified
       |                    |                     |
 CPU/CUDA/Vulkan/HIP     CUDA/etc.            CUDA/Vulkan/etc.
```

## Public runtime concepts

### Engine

Owns:
- device discovery
- backend registry
- global scheduler
- model registry
- cache pools
- telemetry
- global resource limits

### Model

A loaded/partially resident model with:
- immutable model descriptor
- quantization
- architecture
- backend representation
- residency map
- adapters
- compatible cache formats

### Session

A logical generation state:
- request lineage
- sampling policy
- context lineage
- KV references
- priority/class
- cancellation
- telemetry correlation

### Request

A schedulable operation:
- prefill
- decode
- embed/rerank
- adapter operation
- cache operation
- model load/unload

### Device

CPU/GPU/NPU/remote execution target with:
- memory capacity/pressure
- supported operations/formats
- bandwidth/latency estimates
- health and thermal/utilization signals

## Core managers

### Scheduler

Responsibilities:
- continuous/iteration-level batching
- admission control
- latency/throughput QoS
- agent-aware priorities
- preemption
- fairness
- chunked prefill
- speculative work budgeting

See `SCHEDULER.md`.

### KV / Context Manager

Responsibilities:
- block/page allocation
- prefix lookup
- fork/share
- eviction
- CPU/GPU/disk tiers
- quantization
- remote transfer
- compaction lineage
- accounting

See `KV_CACHE.md`.

### Model Residency Manager

Responsibilities:
- model load/unload
- partial/offloaded residency
- adapter hot-swap
- backend artifacts
- memory-pressure decisions
- warm/cold model policies

### Device Planner

Starts simple:
- backend capability check
- fit model/cache/workspace
- choose GPU vs hybrid CPU/GPU

Later:
- heterogeneous placement
- MoE expert placement
- remote-node placement
- prefill/decode placement
- transfer-cost-aware scheduling

### Backend interface

The first useful Sonder-native engine can use llama.cpp/GGML under this interface.

Backend responsibilities:
- tokenize/detokenize or expose tokenizer
- model load/materialization
- prefill
- decode step/batch
- cache import/export/copy where supported
- sampling primitives where backend-owned
- capability advertisement
- deep telemetry hooks where available

The interface must not be shaped around only one backend's quirks.

## Threading / hot path

Rules:
- no blocking Observatory I/O in decode
- cancellation is cheap and checked predictably
- scheduler data structures remain bounded
- telemetry queues are bounded
- allocations in hot decode path are minimized
- backend synchronization is explicit
- deterministic/reproducible mode exists for tests

## Compatibility path

During migration:

```text
Sonder Runtime
  |
  +-- Ollama adapter                (existing/fallback)
  |
  +-- Sonder Inference API
          |
          +-- llama.cpp backend     (first direct backend)
          +-- future backends
```

Do not remove Ollama until feature and quality gates demonstrate that the new path is a reliable replacement for supported workloads.

## Distributed boundary

Do not begin with arbitrary tensor parallelism across slow links.

Preferred progression:

1. route whole requests/models to nodes
2. use background/low-priority work on secondary nodes
3. add cache/context transfer with cost accounting
4. add P/D separation only where network and workload justify it
5. add model/tensor/expert partitioning only with measured benefit

## Observatory

Sonder Inference is a telemetry producer. Observatory is never linked into the critical execution path.

See `OBSERVATORY_CONTRACT.md`.
