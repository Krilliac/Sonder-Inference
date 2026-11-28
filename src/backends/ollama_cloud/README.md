# Ollama Cloud provider (`ollama_cloud`)

Serves Ollama's hosted cloud models directly over HTTPS (`https://ollama.com`,
Ollama's native `/api/chat`, `/api/generate` and `/api/tags`, bearer API key).
No local Ollama daemon is involved. Full reference, policy and the upstream
API sources: [docs/integration/ollama-cloud.md](../../../docs/integration/ollama-cloud.md).

| File | Purpose |
| --- | --- |
| `include/sonder/inference/backends/ollama_cloud.hpp` | Public API: `OllamaCloudOptions`, `make_ollama_cloud_backend`, the pluggable `OllamaCloudTransport`, `redacted_config` |
| `ollama_cloud_backend.cpp` | Backend and model, HTTPS transport (verification always on), allowlist and local-only policy, telemetry queue |
| `ollama_cloud_protocol.*` | Base URL parsing, request bodies, NDJSON decoder, `/api/tags` parsing, HTTP error mapping |
| `ollama_cloud_secret.*` | API key loading (env var name or token file), redaction, wiping |
| `tests/` | `sonder_ollama_cloud_tests` with a fake transport (no network) |

Feature macro: `SONDER_HAS_OLLAMA_CLOUD_BACKEND`. Real HTTPS needs a
`SONDER_WITH_TLS=ON` build.
