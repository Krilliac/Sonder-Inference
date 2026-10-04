# Stream error qualification integration

The bounded sink fix lives in `telemetry.cpp`. It also changes root telemetry
tests, the reusable stress harness and contract/qualification documentation.
The Ollama module changes are test-only: a bounded prefill gate synchronizes an
existing cancellation/disconnect control. Production HTTP cancellation,
backend behavior, public headers and ABI are unchanged.
