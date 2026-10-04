# Stream error qualification integration

The bounded sink fix lives in `telemetry.cpp`. It also changes root telemetry
tests, the reusable stress harness and contract/qualification documentation.
The Ollama module changes are test-only: a bounded prefill gate synchronizes an
existing cancellation/disconnect control. Production HTTP cancellation,
backend behavior, public headers and ABI are unchanged.

Routine stress qualification also changes `tests/CMakeLists.txt`, the reusable
`scripts/stress_telemetry.cpp` receipt label, and telemetry documentation. The
tests link the actual library and inherit its configured instrumentation;
existing full-suite CI lanes run them without workflow changes. Builds with
`SONDER_BUILD_TESTS=OFF` omit the executable and its tests.
