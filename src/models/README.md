# Models

Responsibility: Model and adapter lifecycle and residency.

Implemented: the `Model` handle (`include/sonder/inference/model.hpp`) and the
engine's model registry with `model.load.*`/`model.unload` telemetry
(`src/engine/engine.cpp`).

Implemented here (`SONDER_HAS_MODEL_STORE`): the model store. It reads an existing
Ollama models directory read-only and in place (manifests-v2 and legacy trees, blobs,
manifest lists), resolves `name[:tag]` to GGUF/projector/adapter paths plus params,
template and system records, reports unsupported entries (cloud stubs, safetensors,
missing or truncated blobs), opens every file without following links, hashes small
layers on every read and weights on first use, and exposes `ModelRegistry` for the
native backend plus `sonder-infer models import-ollama`.
Reference: [model store](../../docs/integration/model-store.md). The native backend does
not use it yet.

Deferred: residency manager, adapters, memory-pressure policy.

See [source workspace](../README.md).
