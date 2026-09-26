# Modules and parallel work streams

The core library (`sonder_inference`) is built from `src/{common,engine,
devices,sessions,telemetry,net}` plus `src/backends/{backend,mock_backend}.cpp`.
Optional areas are **modules**: the root `CMakeLists.txt` adds each one with
`add_subdirectory` only if its `CMakeLists.txt` exists, so module work never
needs to edit root files.

| Module directory | Owner branch | Status on main |
| --- | --- | --- |
| `src/cache/` | `feat/kv-cache` | merged (PR #4, 34 tests) — notes: `docs/integration/kv-cache.md`; scheduler adapter in `src/engine/kv_capacity_adapter.hpp`; driven by the engine request runtime (`docs/integration/engine-wiring.md`) |
| `src/scheduler/` | `feat/scheduler` | merged (PR #2, 33 tests) — notes: `docs/integration/scheduler.md`; wired into Engine/Session (`src/engine/request_runtime.cpp`, ADR-016) |
| `src/sampling/` | `feat/sampling` | merged (PR #3, 90 tests; headers under `sonder/sampling/`) — notes: `docs/integration/sampling.md`; core keeps `SamplingConfig` + validation; applied per request for `token_logits` backends (header path kept, ADR-018) |
| `src/backends/llamacpp/` | `feat/llamacpp-backend` | merged (PR #5, 16 unit tests + opt-in GGUF integration); built only with `SONDER_WITH_LLAMA_CPP=ON`; license approved 2026-09-26 (MIT, compatible with the project's Apache-2.0) — notes: `docs/integration/llamacpp.md` |
| `src/backends/ollama/` | `feat/ollama-bench` | extended (PR #6, 34 tests): `OllamaClient` generate/chat/tags/show/ps/version, NDJSON decoder, HTTP error mapping, timing telemetry — notes: `docs/integration/ollama-bench.md` |
| `src/server/` | `eco/inf-serve` | new (ecosystem contract v1, ADR-020): `sonder-infer serve` HTTP/1.1 server (OpenAI-compatible chat through `Session::chat`, health, models, identity), live telemetry hub (SSE/NDJSON, discovery, resume), shared backend factory `sonder/inference/backend_setup.hpp`; `SONDER_HAS_SERVER`; tests `sonder.server.*` — reference: `docs/SERVER.md`; integrator changes: `INTEGRATION_NOTES.md` |
| `bench/` | `feat/ollama-bench` | extended (PR #6, 12 tests): `sonder-bench` runner, baseline corpus with agent fan-out, markdown output; Ollama baseline still pending |

## Module contract

In the module's `CMakeLists.txt` use the helpers from
`cmake/SonderModules.cmake`:

```cmake
sonder_module_sources(foo.cpp bar.cpp)          # compiled into sonder_inference
sonder_module_include_directories(include)      # module-owned public headers
sonder_module_define(SONDER_HAS_FOO)            # feature macro for core/CLI (#if)
sonder_add_module_tests(foo tests/test_foo.cpp) # doctest exe, CTest "sonder.foo.*"
```

- Module public headers go under `<module>/include/sonder/inference/...`.
- Module tests may include `test_helpers.hpp` (from `tests/`) and internal
  headers under `src/`.
- Core code and the CLI reference module features only behind the module's
  `SONDER_HAS_*` macro, so the core builds with any subset of modules.
- Tests must not need network services or model weights; live checks are
  opt-in via environment variables.

## Integration notes

If a module needs anything outside its directory (root CMake, public core
headers, C ABI, CLI wiring, CI), list the exact change in
`INTEGRATION_NOTES.md` at the root of the feature branch. The integrator
applies it on main when merging and removes the notes file.
