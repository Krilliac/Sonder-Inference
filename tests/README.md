# Tests

Core doctest suite (`sonder_core_tests`, registered per test case with CTest
as `sonder.core.*`) plus CTest-driven C ABI and CLI checks. Modules register
their own suites (`sonder.<module>.*`, see ../docs/MODULES.md). Run with `ctest --preset <preset>`.

Coverage:

- `test_session.cpp`: session lifecycle, concurrent-request rejection, failure
  handling, correlated event ordering, model unload while in use.
- `test_cancellation.cpp`: token semantics, cancel from callback, cancel from
  another thread (bounded latency), idle cancel, close while running.
- `test_sampling.cpp`: sampling config validation and enforcement.
- `test_telemetry.cpp`: every event checked against the Observatory envelope
  v1 shape; levels; text capture; bounded queue drops; JSONL file sink.
- `test_mock_backend.cpp`: deterministic streaming, seed/greedy behaviour,
  stop sequences, max tokens, injected failures.
- `test_c_api.cpp`, `c_abi_smoke.c`: the C ABI from C++ and from plain C11.

Module suites on main: `src/backends/ollama/tests/` (URL policy, request
mapping, NDJSON and chunked parsing, unreachable server; live check only with
`SONDER_TEST_OLLAMA_MODEL`) and `bench/tests/` (corpus validation,
percentiles, harness output shape).

Tests never require network services or model weights.
