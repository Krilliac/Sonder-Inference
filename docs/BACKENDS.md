# Backend Strategy

## Rule

Sonder Inference should own a stable execution interface and evolve backends independently.

## Backend 0: Ollama compatibility

Purpose:
- preserve current behavior during migration
- A/B comparison
- fallback for unsupported models

Limitations:
- less direct cache/scheduler/device control
- telemetry limited to what the external server exposes

## Backend 1: direct llama.cpp / GGML

Preferred first native backend because it offers:
- C/C++ embeddability
- GGUF ecosystem
- CPU plus multiple accelerator backends
- quantization breadth
- consumer-hardware orientation
- an existing optimized implementation to benchmark against

Initial objective is **control**, not beating llama.cpp at its own kernels.

## Optional specialized backends

### NVIDIA

Candidates/reference paths:
- ExLlamaV3 for consumer quantized inference
- FlashInfer for kernels/primitives
- TensorRT-LLM for deeply optimized production NVIDIA path

A Sonder-native CUDA operator should be added only after profiling identifies a stable bottleneck and a benchmark demonstrates a meaningful win.

### Compiler-backed portability

Study:
- MLC
- ONNX Runtime
- OpenVINO

Possible use:
- NPU/edge targets
- graph-compiled model variants
- platform-specific accelerators

### Heterogeneous MoE

Study KTransformers/PowerInfer/ik_llama.cpp.

Potential Sonder feature:
- hot experts on GPU
- cold experts/system components in RAM/CPU
- NUMA-aware placement
- dynamic promotion based on measured expert frequency

This is later-phase work.

## Capability advertisement

A backend declares capabilities rather than Sonder assuming them:

```text
tokenization
batched_prefill
continuous_batch_decode
kv_export
kv_import
kv_copy
kv_quantization
prefix_reuse
speculative_decode
lora
moe_routing_telemetry
layer_telemetry
grammar/structured_output
embeddings
vision/audio
tensor_parallel
pipeline_parallel
```

## Backend selection

Selection inputs:
- model/format support
- hardware
- quality mode
- measured throughput/latency
- memory fit
- feature requirements
- reliability history

Do not select by brand/name alone.

## Native kernels

Native code is justified when:
1. profiling shows a material bottleneck;
2. an existing dependency cannot solve it acceptably;
3. the improvement survives representative workloads;
4. correctness matches reference output within defined tolerance;
5. fallback remains available.

## Implementation status (v0.1)

| Backend | Status | Capabilities advertised |
| --- | --- | --- |
| `mock` | implemented; **tests only**, performs no inference | tokenization, streaming, deterministic |
| `ollama` | implemented; loopback HTTP, streaming, cancellable | streaming, remote_process |
| `llamacpp` | implemented, optional (`SONDER_WITH_LLAMA_CPP=ON`, pinned b11195); CPU by default, GPU via `GGML_*` | tokenization, streaming, batched_prefill, deterministic |

The interface lives in `include/sonder/inference/backend.hpp`. Capabilities
are a bitset (`Capability`), matching the list above plus `deterministic`
and `remote_process`.
