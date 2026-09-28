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
