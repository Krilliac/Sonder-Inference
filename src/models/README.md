# Models

Responsibility: Model and adapter lifecycle and residency.

Implemented: the `Model` handle (`include/sonder/inference/model.hpp`) and the
engine's model registry with `model.load.*`/`model.unload` telemetry
(`src/engine/engine.cpp`).

Deferred: residency manager, adapters, memory-pressure policy.

See [source workspace](../README.md).
