# Source workspace

C++20 implementation of the `sonder_inference` library (ADR-011). Public
headers live in [`../include/sonder/inference/`](../include/sonder/inference/)
and the C ABI in [`../include/sonder_inference.h`](../include/sonder_inference.h).
Ownership remains provisional. Optional areas are modules; see
[docs/MODULES.md](../docs/MODULES.md).

- `engine/`: `Engine` (device inventory, backend and model registries, session
  factory) and the C ABI implementation.
- `sessions/`: Generation sessions and request lifecycle.
- `scheduler/`: Inference admission, priorities, batching, and preemption (reserved).
- `cache/`: Logical context and KV cache ownership and accounting (reserved).
- `models/`: Model and adapter lifecycle and residency (handle only so far).
- `backends/`: Execution backend interface and mock backend; `ollama/` and `llamacpp/` are modules. Subject to upstream license review.
- `devices/`: Capability discovery and inference placement policy (CPU inventory).
- `sampling/`: Sampling, structured output, and speculation policy (module, reserved; `SamplingConfig` validation lives in `engine/`).
- `telemetry/`: Bounded inference events for external observers.
- `common/`: Error types and the small JSON value type.
- `net/`: Minimal internal HTTP/1.1 client used by the Ollama adapter.
