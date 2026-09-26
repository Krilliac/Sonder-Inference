# Modules and parallel work streams

The core library (`sonder_inference`) is built from `src/{common,engine,
devices,sessions,telemetry,net}` plus `src/backends/{backend,mock_backend}.cpp`.
Optional areas are **modules**: the root `CMakeLists.txt` adds each one with
`add_subdirectory` only if its `CMakeLists.txt` exists, so module work never
needs to edit root files.

| Module directory | Owner branch | Status on main |
| --- | --- | --- |
| `src/cache/` | `feat/kv-cache` | reserved |
| `src/scheduler/` | `feat/scheduler` | merged (PR #2, 33 tests) — notes: `docs/integration/scheduler.md` |
| `src/sampling/` | `feat/sampling` | merged (PR #3, 90 tests; headers under `sonder/sampling/`) — notes: `docs/integration/sampling.md`; core keeps `SamplingConfig` + validation |
| `src/backends/llamacpp/` | `feat/llamacpp-backend` | reserved (license review required first) |
| `src/backends/ollama/` | `feat/ollama-bench` | initial adapter landed with the foundation |
| `bench/` | `feat/ollama-bench` | harness skeleton + smoke corpus landed with the foundation |

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
