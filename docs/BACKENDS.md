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

### GPU build presets (`CMakePresets.json`)

Default CI presets (`ci-windows`, `ci-linux`) keep `SONDER_WITH_LLAMA_CPP=OFF` and never pull or compile GPU SDKs. Opt-in presets enable the direct `llamacpp` backend for consumer discrete and integrated GPUs:

| Preset | Platform | Accelerator | Key flags |
| --- | --- | --- | --- |
| `windows-cuda-blackwell` / `linux-cuda-blackwell` | Windows (MSVC) / Linux (GCC/Clang), `Release` | NVIDIA CUDA (Blackwell) | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120a`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `windows-cuda-blackwell-120f` / `linux-cuda-blackwell-120f` | Windows / Linux, `Release` | NVIDIA CUDA (Blackwell `sm_12x` family) | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120f`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `windows-cuda-rtx` | Windows (MSVC + Ninja) | NVIDIA CUDA | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86;89;120a`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `windows-vulkan-xt` | Windows (MSVC + Ninja) | Vulkan (AMD RX XT / APU / cross-vendor) | `GGML_VULKAN=ON` |
| `linux-cuda-rtx` | Linux (GCC/Clang + Ninja) | NVIDIA CUDA | `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86;89;120a`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `linux-rocm-xt` | Linux (GCC/Clang + Ninja) | AMD ROCm/HIP | `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201` |
| `linux-vulkan-xt` | Linux (GCC/Clang + Ninja) | Vulkan (AMD RX XT / APU / cross-vendor) | `GGML_VULKAN=ON` |

Target coverage:
- **NVIDIA RTX (`86;89;120a`, `120a`, `120f`)**: Ampere (RTX 30-series, `sm_86`), Ada Lovelace (RTX 40-series, `sm_89`), and Blackwell (RTX 50-series, `sm_120a` non-portable cubins or `sm_120f` family-portable cubins). Configure-time checks (`cmake/SonderCudaCheck.cmake`) require CUDA Toolkit `>= 13.4` (`13.5` preferred), reject bare `120` targets (which fail `ptxas` on block-scaled FP4 MMA), and refuse `cuBLAS 13.6.0` (`CUB-10445`); see [GPU_SUPPORT.md](GPU_SUPPORT.md#2-nvidia-rtx-ampere-ada-blackwell). `GGML_CUDA_FA_ALL_QUANTS=ON` compiles Flash Attention kernels for quantized KV cache types (`q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl`).
- **AMD Radeon RX XT & Ryzen APU**: RDNA 2 (`gfx1030`), RDNA 3 discrete (`gfx1100`, `gfx1101`, `gfx1102`), RDNA 3 Ryzen APU UMA (`gfx1103`, e.g. Radeon 780M/890M), and RDNA 4 (`gfx1200`, `gfx1201`). On Linux, `linux-rocm-xt` targets ROCm/HIP directly (`ROCm0`); on Windows (and on Linux hosts without ROCm), `windows-vulkan-xt` / `linux-vulkan-xt` use Vulkan (`Vulkan0`).

### Upstream llama.cpp pin and Ternary / PrismML overrides

`src/backends/llamacpp/CMakeLists.txt` defaults to the tag pinned in `cmake/llamacpp-pin.json`, currently **`v1.0.0`** (commit `b1b770d`), with a verified SHA-256 (recorded in [LICENSE_REVIEW.md](LICENSE_REVIEW.md)) so default and CI builds stay deterministic. The previous pin was `b11195`; `v1.0.0` contains mainline `b11556`. To run newer low-bit or Hadamard-ternary GGUF formats with direct `llamacpp`, override the source tree at configure time without editing the default pin:

```sh
# Local checkout (offline builds, newer mainline, or PrismML fork):
cmake --preset windows-cuda-rtx -DFETCHCONTENT_SOURCE_DIR_LLAMACPP=/path/to/llama.cpp

# Or override the fetched tarball tag + matching SHA-256 together (fails closed on mismatch):
cmake --preset linux-cuda-rtx \
  -DSONDER_LLAMACPP_TAG=b11556 \
  -DSONDER_LLAMACPP_SHA256=<sha256-of-b11556-tarball>
```

Format-to-runtime matrix:
- **Pinned `v1.0.0` (default)**: standard GGUF K-quants, I-quants, `MXFP4` (type 39), ternary `TQ1_0`/`TQ2_0` (types 34–35), and the mainline `>= b11556` types below (`ggml.h` at the pin defines types 40–43; the previous `b11195` pin defined 40–42 but not `F8_E4M3`).
- **Mainline `ggml-org/llama.cpp >= b11556`**: adds stock `NVFP4` (Blackwell FP4, type 40), `Q1_0` (1-bit Bonsai-1, type 41), `Q2_0` (2-bit ternary, type 42), and `F8_E4M3` (FP8, type 43).
- **`PrismML-Eng/llama.cpp` (`prism` branch)**: adds Hadamard-transformed Bonsai-2 ternary formats (`PQ2_0` type 142, `PTQ1_0` type 143, `PQ2_0M` KV-only type 145) and `prism.hadamard.*` GGUF metadata handling.

**RPC server at this pin.** Sonder does not build or start the ggml RPC server (`GGML_RPC` stays off). `v1.0.0` predates upstream `b12658` ("rpc : typed request helpers and a single validation step on the server", #32651), which first ships in `v1.1.0`. Any `rpc-server` built from this tree (for example for `llama-server --rpc`) must stay on a private link until the pin moves to `v1.1.0` or later.

The `llamaserver` backend can also spawn or attach to an external mainline `>= b11556` or PrismML `llama-server` binary while keeping the core build independent of llama.cpp. Sonder's GGUF header inspection recognizes all of these block layouts for fit estimation, but model execution requires a runtime build whose GGML implementation supports the file's tensor types.

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
| `mock` | implemented; **tests only**, performs no inference | tokenization, streaming, deterministic; token_logits when `MockBackendOptions::token_logits` |
| `ollama` | implemented; loopback HTTP, streaming, cancellable | streaming, remote_process |
| `ollama_cloud` | implemented (`src/backends/ollama_cloud`, 2026-11-26): Ollama's hosted models over verified HTTPS (`/api/chat`, `/api/generate`, `/api/tags`, bearer key from env var name or token file), no local daemon. Opt-in allowlist of cloud models; refuses local-only requests; real HTTPS needs `SONDER_WITH_TLS=ON`. See [integration/ollama-cloud.md](integration/ollama-cloud.md) | streaming, remote_process, tool_calling |
| `llamacpp` | implemented, optional (`SONDER_WITH_LLAMA_CPP=ON`, pinned `v1.0.0` default (`cmake/llamacpp-pin.json`), overridable via `FETCHCONTENT_SOURCE_DIR_LLAMACPP`); CPU by default, GPU presets for CUDA RTX / ROCm XT / Vulkan XT; supports tensor overrides, KV quantization, and multi-GPU placement (`devices`, `main_gpu`, `split_mode`, `tensor_split`, `use_mlock`) | tokenization, streaming, batched_prefill, deterministic |
| `llamaserver` | implemented, always built (`src/backends/llamaserver`); no llama.cpp linked. Attach to a running llama-server or any OpenAI-compatible upstream (default `http://127.0.0.1:8080`), or spawn and supervise a llama-server child. `"native_completion": false` selects generic OpenAI mode. Status 2026-11-20. See [integration/llama-server.md](integration/llama-server.md) | streaming, remote_process; plus speculative_decode and prefix_reuse in native mode (not generic OpenAI mode); structured_output when a grammar is configured |

The interface lives in `include/sonder/inference/backend.hpp`. Capabilities
are a bitset (`Capability`), matching the list above plus `deterministic`,
`remote_process`, `token_logits` (the backend exposes per-step logits via
`BackendModel::open_token_stream()` and Sonder samples with its own chain; see
[integration/engine-wiring.md](integration/engine-wiring.md)).
`ModelDescriptor::resident` is false when `load_model()` only fetched
metadata (Ollama); telemetry reports it on `model.load.completed`.

Sampling fields per backend: the Sonder chain (backends with `token_logits`)
and llama.cpp apply every `SamplingConfig` field. Ollama forwards all of them
as request options except `logit_bias`. A request with a non-empty
`logit_bias` fails with `invalid_argument` on Ollama. llama.cpp rejects a
`num_ctx` larger than the context it was loaded with. See
[integration/sampling-config.md](integration/sampling-config.md).
