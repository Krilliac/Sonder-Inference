# Backends

Responsibility: execution backend adapters for the `Backend` interface in
`include/sonder/inference/backend.hpp`, subject to upstream license review
([docs/LICENSE_REVIEW.md](../../docs/LICENSE_REVIEW.md)).

## Direction

The native in-process `llama.cpp`/GGML backend (`src/backends/llamacpp`) is the
**target and only long-term local inference engine** for Sonder Inference. The
project is consolidating onto a single vendored, pinned `ggml`/`llama.cpp`
source tree with SI's own patch series rather than relying on external servers
or separate forks ([docs/ROADMAP_SI_ONLY.md](../../docs/ROADMAP_SI_ONLY.md)).

The `ollama` and `llamaserver` backends are **deprecated and scheduled for
removal** (RET-07). Sonder Inference must own scheduling, memory management,
and telemetry end to end; piggy-backed external servers obscure residency and
hide execution internals (KV cache state, token logits, layer and operator
activity) from the Sonder Observatory.

An **Ollama Cloud** provider (RET-02) is **in progress (not yet merged on
`main`)** to reach Ollama Cloud models directly over HTTPS without a local
Ollama daemon, keeping cloud access opt-in and built on optional `src/net` TLS
([docs/integration/tls.md](../../docs/integration/tls.md)).

## Implemented

- `backend.cpp`: capability names and stop reasons for
  `include/sonder/inference/backend.hpp`.
- `mock_backend.cpp`: **MOCK** deterministic backend for tests and harness
  development only. It performs no inference.
- `llamacpp/` (module, **target local engine**): direct, in-process
  `llama.cpp`/GGML backend built with `SONDER_WITH_LLAMA_CPP=ON`. Supports
  streaming decode, batched prefill, continuous batching, prefix reuse,
  speculative decoding, structured output, embeddings, reranking, and
  recurrent/execution checkpoints
  ([llamacpp/README.md](llamacpp/README.md),
  [docs/GPU_SUPPORT.md](../../docs/GPU_SUPPORT.md),
  [docs/integration/llamacpp.md](../../docs/integration/llamacpp.md)).
- `ollama/` (module, **deprecated, scheduled for removal**): Ollama HTTP
  compatibility adapter (`/api/version`, `/api/tags`, `/api/show`, `/api/ps`,
  streaming `/api/generate` and `/api/chat`), loopback-only by default,
  cancellable mid-stream ([ollama/README.md](ollama/README.md)).
- `llamaserver/` (module, **deprecated, scheduled for removal**): adapter that
  spawns or attaches to an external `llama-server` or OpenAI-compatible HTTP
  upstream without linking `llama.cpp`
  ([docs/integration/llama-server.md](../../docs/integration/llama-server.md)).

See [docs/ROADMAP_SI_ONLY.md](../../docs/ROADMAP_SI_ONLY.md),
[docs/BACKENDS.md](../../docs/BACKENDS.md), and
[source workspace](../README.md).
