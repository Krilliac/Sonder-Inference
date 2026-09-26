# Backends

Responsibility: Execution backend adapters, subject to upstream license review.

Implemented:

- `backend.cpp`: capability names and stop reasons for the interface in
  `include/sonder/inference/backend.hpp`.
- `mock_backend.cpp`: **MOCK** deterministic backend for tests and harness
  development only. It performs no inference.
- `ollama/` (module): Ollama compatibility adapter
  (`/api/version`, `/api/tags`, streaming `/api/generate`), loopback-only by
  default, cancellable mid-stream.

Deferred: direct llama.cpp/GGML backend (see ../../docs/LICENSE_REVIEW.md).

See [source workspace](../README.md).
