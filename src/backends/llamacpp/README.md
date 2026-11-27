# llama.cpp backend

Direct llama.cpp/GGML execution backend, built only with
`-DSONDER_WITH_LLAMA_CPP=ON` (default OFF). llama.cpp is fetched with
FetchContent at a pinned tag (`b11195`, SHA-256 checked) and linked statically.
See `docs/LICENSE_REVIEW.md` for the dependency record.

| Path | Purpose |
|---|---|
| `include/sonder/backends/llamacpp/llamacpp_backend.h` | policy-free wrapper over the llama.cpp C API |
| `include/sonder/inference/backends/llamacpp.hpp` | `make_llamacpp_backend()` factory for the core `Backend` interface |
| `llamacpp_backend.cpp`, `llamacpp_util.cpp` | wrapper implementation (`llamacpp_backend.cpp` is the only file that includes `llama.h`) |
| `llamacpp_core_backend.cpp`, `llamacpp_internal.hpp` | core adapter (sampling mapping, stop sequences, error mapping) |
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

## Hadamard-folded (Bonsai / Prism) load guard

`Load()` reads the GGUF metadata keys before handing the file to llama.cpp. A model
carrying any `prism.hadamard.*` key (upstream llama.cpp PR #29600) is refused with
`ErrorCode::kUnsupported` when the linked llama.cpp has no prism.hadamard / FWHT runtime
(CMake defines `SONDER_LLAMACPP_HAS_PRISM_HADAMARD` when `src/llama-arch.cpp`, `llama-model.cpp` or
`llama-context.cpp` mentions `prism.hadamard.version`: mainline >= b11556 / v1.0.0 and the PrismML
fork do; older trees such as b11195 do not). Override with
`-DSONDER_LLAMACPP_FORCE_PRISM_HADAMARD=ON|OFF` (any other value is a configure error). Warning:
`ON` only claims runtime support exists; if the linked library lacks it, Hadamard-folded models
load and emit garbage again. Use it only when you know the library has FWHT.

The guard runs before the previously loaded model is unloaded, so a refused load keeps it.

`LoadOptions::startup_smoke_test` (default off; `LlamaCppBackendOptions::startup_smoke_test`,
`sonder-infer serve --startup-smoke-test`, and launch-profile key `startup_smoke_test`) runs an
8-token greedy decode of a fixed prompt after load and fails the load with `ErrorCode::kLoadFailed`
on non-finite logits, degenerate uniform logits, immediate end-of-generation (zero non-EOG content
tokens produced), or a single repeated token (at least 4 steps). It is skipped for `vocab_only`
loads, encoder-only models, and pooled contexts (`llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE`,
covering embedding MEAN/CLS/LAST and reranker RANK), and always clears the KV cache before
returning. Note: this is a lightweight greedy sanity check (not KLD against a reference model;
varied garbage or periodic multi-token cycles such as A/B/A/B are not rejected).
