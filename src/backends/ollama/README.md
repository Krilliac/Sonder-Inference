# Ollama backend (Backend 0)

Compatibility/fallback adapter over the Ollama HTTP API (docs/BACKENDS.md,
ADR-002). Compiled into `sonder_inference` when this directory's
`CMakeLists.txt` exists; defines `SONDER_HAS_OLLAMA_BACKEND`.

Public header: `include/sonder/inference/backends/ollama.hpp`.

| Piece | What it does |
|---|---|
| `OllamaClient` | `/api/generate` + `/api/chat` NDJSON streaming, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` |
| `StreamDecoder` | Incremental NDJSON decoder (arbitrary byte splits, CRLF, error lines, truncation detection) |
| `make_ollama_backend(OllamaBackendOptions)` | Core `Backend` adapter: `probe`, `list_models`, `load_model` (metadata via `/api/show`, no pulls), `generate` |
| `ollama_protocol.hpp` (internal) | Wire helpers used by the core-facing tests: `build_generate_body`, `parse_stream_line`, `descriptor_from_tag` |
| `timing_attributes()` / `emit_timing_events()` | Maps `load_duration`, `prompt_eval_*`, `eval_*` plus client TTFB/TTFT/wall to Observatory telemetry |

Cancellation: `CancellationToken` is polled by the core HTTP client while
waiting on I/O and per chunk; a tripped token returns `ErrorCode::cancelled`.
A callback returning `false` stops the stream and keeps the partial result.

Transport policy: non-loopback hosts are refused unless
`OllamaBackendOptions::allow_remote` is set (for http and https alike).
`https://` works only in builds configured with `-DSONDER_WITH_TLS=ON`
(OpenSSL on Linux/macOS, Schannel on Windows) and is configured through
`OllamaBackendOptions::tls`; in the default TLS-off build it returns
`unsupported`. See docs/integration/tls.md.

Tests (`tests/`) use recorded NDJSON fixtures and an in-process fake Ollama
server; no live Ollama is needed.
