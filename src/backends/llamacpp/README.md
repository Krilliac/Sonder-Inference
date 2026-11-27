# llama.cpp backend

Direct llama.cpp/GGML execution backend, built only with
`-DSONDER_WITH_LLAMA_CPP=ON` (default OFF). llama.cpp is fetched with
FetchContent at a pinned tag (`b11195`, SHA-256 checked) and linked statically.
See `docs/LICENSE_REVIEW.md` for the dependency record.

| Path | Purpose |
|---|---|
| `include/sonder/backends/llamacpp/llamacpp_backend.h` | policy-free wrapper over the llama.cpp C API |
| `include/sonder/backends/llamacpp/aux_stream_gate.hpp` | in-process CUDA stream priorities, `DecodeAwareAuxGate`, and `CudaPriorityStream` / STT slice-planning API scaffolding |
| `include/sonder/inference/backends/llamacpp.hpp` | `make_llamacpp_backend()` factory for the core `Backend` interface |
| `llamacpp_backend.cpp`, `llamacpp_util.cpp`, `aux_stream_gate.cpp` | wrapper, stream priority, and DRR auxiliary gate implementation (`llamacpp_backend.cpp` is the only file that includes `llama.h`) |
| `sonder_ggml_cuda_stream_priority.cuh`, `sonder_cuda_prio_state.*` | thread-scoped `ggml-cuda` `cudaStreamCreateWithPriority` shim (HIGH for main decode, LOW for auxiliary/embed/rerank) |
| `llamacpp_core_backend.cpp`, `llamacpp_internal.hpp` | core adapter (sampling mapping, stop sequences, error mapping, shared `DecodeAwareAuxGate`) |
| `tests/` | doctest unit tests (no weights) + opt-in real-model integration test |

```sh
cmake --preset linux-release -DSONDER_WITH_LLAMA_CPP=ON -DGGML_NATIVE=OFF
cmake --build build/linux-release -j 4
ctest --test-dir build/linux-release -R llamacpp
SONDER_TEST_GGUF=/path/to/model.gguf ctest --test-dir build/linux-release -R llamacpp.integration -V
```

CPU is the default device. GPU offload works when llama.cpp is configured with
an accelerator (`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`, ...) and the model is loaded
with a `gpu:*` device id. Never commit model weights.

Tensor placement: `LoadOptions::tensor_overrides` (and `serve --moe-experts cpu` /
`--tensor-override PATTERN=DEVICE`) place matching weight tensors on a chosen
device, e.g. MoE expert weights in system RAM while attention stays on the GPU.
See `docs/PLACEMENT.md`.

Co-resident auxiliary models (`aux_stream_gate.hpp`): on GPU domains (`gpu:*`),
main LLM decode runs on high-priority CUDA streams (`WorkloadRole::kMainDecode`),
while in-process embeddings, cross-encoder reranking, and `WorkloadRole::kAuxiliary`
models run on low-priority streams (`StreamPriority::kLow`) and slice batches
(`<= 512` tokens / `<= 4` sequences while foreground decode is active) between
decode steps via `DecodeAwareAuxGate`. Each gated decode step and auxiliary slice
calls `llama_synchronize` before releasing its scope so the gate orders device
execution rather than host kernel launch; CPU-only loads (`gate_cpu_domain = false`
by default) bypass slicing and throttling unless explicitly enabled. The default
`max_aux_duty_cycle = 0.05` (`>= 95%` main-decode duty share) is a configured DRR
target whose end-to-end throughput retention on full-scale models remains to be
benchmarked on hardware. `CudaPriorityStream` / `BuildTrtRtxProviderOptions`
(`NvTensorRTRTXExecutionProvider` `user_compute_stream`), `PlanSttAudioSlices`,
and `RunGatedAuxSlices` are tested API scaffolding with no in-tree STT or ONNX
Runtime consumer yet.
