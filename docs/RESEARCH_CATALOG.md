# Inference Engine Research Catalog

Snapshot: **2026-09-26**.

This catalog is a design/reference inventory, not a dependency list. Verify current license, maintenance status, supported hardware, and API stability before adopting code.

## A. Major execution / serving engines

| Project | Layer / focus | Ideas to study for Sonder |
|---|---|---|
| llama.cpp / GGML | native local runtime | GGUF, broad CPU/GPU portability, quantization, hybrid offload, embeddability |
| vLLM | high-throughput serving | PagedAttention, continuous batching, prefix caching, scheduler, chunked prefill, distributed serving |
| SGLang | serving/runtime | RadixAttention/prefix reuse, structured workloads, scheduling, P/D work |
| TensorRT-LLM | NVIDIA optimized runtime | fused kernels, quantization, multi-GPU, MoE, speculative decode, P/D |
| Hugging Face TGI | production serving | batching, telemetry, quantization, multi-GPU/server design |
| LMDeploy / TurboMind | CUDA serving | persistent batching, blocked KV, prefix/cache optimization |
| LightLLM | Python/Triton serving | token-level KV management, dynamic batching |
| DeepSpeed-MII / FastGen | distributed serving | blocked KV, Dynamic SplitFuse, distributed replicas |
| Aphrodite Engine | vLLM-derived serving | broad model/quantization support and serving ergonomics |
| MLC LLM | compiler-backed portable runtime | compilation across CUDA/ROCm/Vulkan/Metal/WebGPU |
| WebLLM | browser WebGPU runtime | portable client-side WebGPU execution |
| ExLlamaV3 | consumer NVIDIA specialization | quantization, cache quantization, continuous batching, speculation |
| ExLlamaV2 | predecessor/reference | historical EXL2 design; superseded for new work |
| KTransformers | heterogeneous CPU/GPU MoE | hot/cold expert placement, CPU AMX/AVX, NUMA awareness |
| NInfer | specialized C++/CUDA | value of intentionally narrow specialization |
| ik_llama.cpp | optimized llama.cpp derivative | CPU/MoE optimization, repacking, experimental execution paths |
| ONNX Runtime GenAI | graph/runtime | hardware provider abstraction + generation state machine |
| ExecuTorch | edge runtime | mobile/embedded inference and accelerator delegates |
| MLX / MLX-LM | array/runtime + LLM tooling | Apple-centric optimized execution and compact runtime design |
| PowerInfer | heterogeneous research runtime | activation locality, hot/cold neuron placement |
| BitNet / bitnet.cpp | specialized low-bit runtime | architecture-specific extreme quantization |
| FastFlowLM | NPU-oriented runtime | Ryzen AI/XDNA-style accelerator execution |
| Lemon MLX Engine | native C++ MLX server/runtime | native multi-model serving without a Python dependency |
| Modular MAX | compiler/runtime/serving | graph compiler + scheduler + custom-kernel model |
| LitGPT | readable model/reference stack | understandable implementations and correctness baselines |
| Candle | Rust tensor framework | native Rust tensor/runtime foundation |
| atoma-infer | Rust/CUDA engine | Rust-native inference architecture |
| rLLM | Rust/Candle server | paged attention, batching, prefix cache, observability |
| vllm.cpp / LocalAI work | native C++ vLLM-inspired path | from-scratch native implementation of paging/batching concepts |

## B. Serving / orchestration layers

These may sit above an engine and should not be confused with a tensor runtime.

| Project | What to study |
|---|---|
| Ollama | current Sonder baseline; model lifecycle and simple local API ergonomics |
| LocalAI | multi-backend local gateway and packaging |
| Lemonade | local multi-backend orchestration |
| BentoML / OpenLLM | packaging/deployment interfaces |
| Triton Inference Server | backend model, batching, production server concerns |
| Ray Serve LLM | distributed routing/autoscaling around serving engines |
| NVIDIA Dynamo | distributed inference orchestration, KV-aware routing, disaggregation |
| llm-d | Kubernetes-native distributed inference |
| TabbyAPI | ExLlama serving/API layer |
| llamafile | portable packaging/deployment of llama.cpp |

## C. KV/cache systems

| Project | Why it matters |
|---|---|
| LMCache | engine-independent KV tiers/reuse and storage/transport |
| Mooncake | KV-centric distributed serving and P/D architecture |
| vLLM prefix caching | practical prefix block reuse |
| TensorRT-LLM KV reuse | production-oriented cache reuse mechanisms |

## D. Kernel / compiler libraries

| Project | Role |
|---|---|
| FlashInfer | LLM inference kernels: paged/ragged KV, sampling, MoE, speculation primitives |
| FlashAttention | optimized attention family |
| MInference | long-context/sparse attention optimization |
| GGML | low-level tensor/runtime foundation used by llama.cpp |
| ONNX Runtime | broad execution-provider abstraction |
| OpenVINO | Intel CPU/iGPU/dGPU/NPU compiler/runtime |

## E. Research systems and papers

| Work | Key concept to study |
|---|---|
| Orca | iteration-level scheduling / continuous batching foundations |
| PagedAttention / vLLM paper | paged KV-memory management |
| SGLang paper | structured generation + RadixAttention |
| Sarathi / Sarathi-Serve | chunked prefill and decode-friendly scheduling |
| DistServe | disaggregated prefill/decode |
| Splitwise | hardware-aware prefill/decode separation |
| FastServe | token-granularity preemption / MLFQ-style scheduling |
| vAttention | virtual-memory alternative to explicit KV paging |
| FlexGen | GPU/CPU/SSD memory hierarchy |
| Punica | multi-LoRA serving |
| S-LoRA | scalable adapter serving |
| DeepSpeed-FastGen | Dynamic SplitFuse |
| Preble | distributed prompt/prefix-aware scheduling |
| P/D-Serve | production-scale P/D separation research |
| INFERCEPT | interruption/dynamic inference workloads |
| Mooncake paper | KV-centric disaggregated serving |

## F. Secondary research backlog

These surfaced in broad surveys and should be investigated independently before we rely on them:

- NanoFlow
- Project Zero
- EIE
- XTLLM
- PrefillOnly
- other experimental inference-engine repositories cataloged by community/academic surveys

## What not to do

Do not build a "Franken-engine" by copying internals from multiple projects. Use the catalog to identify **concepts and measurable requirements**. Where code reuse is desired, choose explicit dependencies only after license/API/maintenance review.

## Current study priority

1. llama.cpp/GGML
2. vLLM
3. SGLang
4. LMCache + Mooncake
5. KTransformers
6. ExLlamaV3 + NInfer + ik_llama.cpp
7. FlashInfer
8. TensorRT-LLM
9. MLC
10. FlexGen / Sarathi / DistServe / Preble / FastServe papers
