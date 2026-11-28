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

## Optional llguidance JSON-Schema path (`SONDER_WITH_LLGUIDANCE`)

`-DSONDER_WITH_LLGUIDANCE=ON` (default OFF; needs `SONDER_WITH_LLAMA_CPP=ON`, `cargo` and `git`)
builds [llguidance](https://github.com/guidance-ai/llguidance) at the commit llama.cpp v1.0.0 pins
and links it into the backend. Configure fails with a message about `cargo` when the option is ON
and the Rust toolchain is missing; with the option OFF nothing changes. See
`docs/LICENSE_REVIEW.md` for the dependency record.

With it ON, a chat request with `response_format: {type: json_schema}` is constrained by llguidance
instead of the GBNF the server compiled. The schema is first normalised
(`json_schema_normalize.hpp`): every object schema that does not set `additionalProperties`
(or `unevaluatedProperties`) gets `additionalProperties: false`. Objects with `$ref`, schemas with
`allOf` and their direct `allOf` branches are left open, because closing them would forbid
properties that come from the referenced or sibling schemas. Hostile schemas are bounded (depth 64,
100000 schema nodes, 4 MiB). A schema that cannot be normalised, or that llguidance rejects, fails the
request with `invalid_argument`; it never falls back to unconstrained output. `json_object`,
`grammar`, tool-call grammars and every other request keep the GBNF path.

With it ON, `prepare_chat_grammar` in `src/server/src/chat_handler.cpp` skips GBNF compilation
for `llamacpp` chat requests with `response_format: {type: json_schema}` and hands the normalised
schema straight to the native backend (which enforces it through llguidance). With the option OFF,
`chat_handler.cpp` behaviour is unchanged and `json_schema` takes the existing GBNF path.
