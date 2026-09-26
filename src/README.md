# Source workspace

Language-neutral ownership placeholders; there are no source modules or build
targets yet. See [scaffold status](../docs/SCAFFOLD.md).

- `sessions/`: Generation sessions and request lifecycle.
- `scheduler/`: Inference admission, priorities, batching, and preemption.
- `cache/`: Logical context and KV cache ownership and accounting.
- `models/`: Model and adapter lifecycle and residency.
- `backends/`: Execution backend adapters, subject to upstream license review.
- `devices/`: Capability discovery and inference placement policy.
- `sampling/`: Sampling, structured output, and speculation policy.
- `telemetry/`: Bounded inference events for external observers.
