# Design Decisions

## ADR-001 — Build an inference runtime before a kernel ecosystem

**Decision:** Sonder owns scheduling, sessions, cache, device policy, model lifecycle, and telemetry first. Existing backends provide tensor execution.

**Reason:** this captures most of the Sonder-specific value early while avoiding years of architecture/quantization/kernel reimplementation.

## ADR-002 — Keep Ollama during migration

**Decision:** retain an Ollama adapter until the direct path clears compatibility/reliability gates.

**Reason:** rollout safety and A/B benchmarking.

## ADR-003 — Direct llama.cpp/GGML is the initial native execution candidate

**Decision:** start evaluation with direct embedding/control rather than building CUDA kernels first.

**Reason:** broad consumer hardware and model/quantization support, C/C++ embedding, GGUF ecosystem.

**Not permanent:** the backend interface remains independent.

## ADR-004 — KV/context is a first-class resource

**Decision:** Sonder owns logical KV allocation/reuse/eviction/accounting above backend-specific physical representation.

**Influences:** vLLM PagedAttention, LMCache, Mooncake, SGLang prefix reuse.

## ADR-005 — Scheduler is agent-aware through metadata, not model-name hacks

**Decision:** Runtime passes workload class/priority/deadline/cancellability.

**Reason:** keep Inference generic and prevent hard-coded orchestration model names.

## ADR-006 — Continuous batching + chunked prefill are target capabilities

**Influences:** Orca, vLLM, Sarathi.

**Gate:** implement incrementally and benchmark interactive latency, not just aggregate throughput.

## ADR-007 — Distributed work starts request-level

**Decision:** whole-model/request routing before tensor/model sharding across ordinary LAN links.

**Reason:** simpler correctness and often better cost/benefit on heterogeneous consumer hardware.

## ADR-008 — Native operators are benchmark-driven

**Decision:** replace backend operators only when profiling and reproducible benchmarks justify them.

**Reason:** avoid maintaining redundant kernels without a measured benefit.

## ADR-009 — Observatory is external

**Decision:** telemetry producer only; no 3D renderer in inference process.

**Reason:** isolation and hot-path safety.

## ADR-010 — Research ideas are not automatically code dependencies

**Decision:** concept adoption and code reuse are separate decisions.

Before code reuse:
- license review
- API/ABI stability review
- maintenance status
- security review
- benchmark/correctness validation

## ADR-011 — Engine core in C++20 with CMake presets and a stable C ABI

**Date:** 2026-09-26. **Status:** accepted (decided by the owner's chief of staff; owner requested implementation).

**Decision:** implement the engine core in C++20, built with CMake presets
(MSVC + Ninja on Windows, GCC/Clang + Ninja elsewhere). The public C++ API
lives in `include/sonder/inference/`. A separate, stable C ABI
(`include/sonder_inference.h`) is the boundary for Rust, Python, C#, and other
bindings.

**Reason:**
- llama.cpp/GGML (ADR-003) is C/C++; embedding it directly avoids an FFI layer
  on the hottest path.
- The owner is a C++ engine developer; the codebase should match the team's
  strongest toolchain.
- A C ABI (opaque handles, `struct_size`-versioned structs, append-only status
  codes, thread-local error text) gives future Rust or managed bindings a
  stable surface without freezing the C++ API.

**Consequences:**
- C++ types never cross the C ABI; `SONDER_ABI_VERSION` increments on any
  incompatible change.
- Rust is not excluded: a Rust front end or components can bind the C ABI later.
- No exceptions cross the C ABI; C++ API errors use `Status`/`Result<T>`.

## ADR-012 — Telemetry uses the Observatory envelope v1 directly

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** every event is emitted as one JSON object per line conforming to
`sonder.observatory.event/1`
([schema](https://github.com/Krilliac/Sonder-Observatory/blob/main/protocol/observatory-events.schema.json)).
Emission goes through a bounded queue drained by a writer thread; overflow is
dropped and counted, and a final `telemetry.dropped` event reports the count.

**Details:**
- Engine-level events (engine, device, backend, model lifecycle) use the engine
  id as `session_id`, because the envelope requires a session on every event.
- Levels: `metrics` (lifecycle and request summaries), `standard` (adds
  per-chunk `inference.token.generated`), `deep` (reserved for backend
  layer/operator events when a backend exposes them).
- Token text is excluded unless explicitly enabled (`capture_text`), keeping
  raw-text capture independent from structural events.
- Additional event types beyond the contract list are documented in
  [OBSERVATORY_CONTRACT.md](OBSERVATORY_CONTRACT.md#implementation-status-v01).

## ADR-013 — Minimal in-house HTTP/JSON for the Ollama adapter

**Date:** 2026-09-26. **Status:** accepted (revisit when TLS or HTTP/2 is needed).

**Decision:** the Ollama adapter uses a small internal HTTP/1.1 client
(Content-Length, chunked, streaming, cancellable) and a small JSON value type
rather than third-party libraries.

**Reason:** keeps the library dependency-free while licensing (ADR-010) and
package policy are unsettled; the surface needed is tiny.

**Constraints:** plain HTTP only; loopback hosts only unless `allow_remote` is
set explicitly. Remote workers belong behind TLS, which this adapter does not
implement.

## ADR-014 — Test framework: doctest fetched at configure time

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** unit tests use doctest (MIT), downloaded by CMake FetchContent at
a pinned version and SHA-256, used by the test executable only, and never
vendored into the repository or linked into the library. Tests are registered
with CTest per test case. See [LICENSE_REVIEW.md](LICENSE_REVIEW.md#doctest).

## ADR-015 — Project license: Apache-2.0

**Date:** 2026-09-26. **Status:** accepted (revised the same day: Nate
overrode the initial MIT choice with Apache-2.0).

**Decision:** Sonder Inference is licensed under the Apache License, Version
2.0. The root `LICENSE` holds the full standard text. `NOTICE` reads
"Sonder-Inference, Copyright 2026 Nate Witkowski" and credits the fetched MIT
third-party code (llama.cpp b11195, cpp-httplib v0.58.0 test-only, doctest
test-only).

**Reason:** Nate's explicit choice. It adds an express patent grant and a
NOTICE convention for attribution. All adopted dependencies are MIT, which
is compatible with Apache-2.0 as long as the MIT notices are kept (they are
in `NOTICE`). History: PR #7 first adopted MIT from a survey of Krilliac's
repositories; this revision supersedes it. See
[LICENSE_REVIEW.md](LICENSE_REVIEW.md#project-license).

## ADR-016 — Engine-owned request runtime (scheduler + logical KV)

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** When the cache and scheduler modules are built, the `Engine`
owns a request runtime. It is a coordinator thread that drives
`scheduler::Scheduler` over `cache::KvCacheManager`, getting capacity through
`KvCacheCapacityAdapter`. Sessions submit requests and wait for a per-token
grant before each token. The runtime is on by default
(`EngineOptions::scheduling.enabled`) and does nothing in a core-only build.
Preemption uses recompute mode, and the adapter's release hook frees the
preempted sequence synchronously.

**Reason:** makes batching, admission, preemption and prefix reuse
observable and testable with the mock backend now, without waiting for a
native backend with KV control. Output is unchanged by scheduling.

**Constraints:** KV is logical until backends expose KV control. The token
gate is lockstep per step. See
[integration/engine-wiring.md](integration/engine-wiring.md).

## ADR-017 — NoViableCandidates maps to `invalid_argument`

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** When Sonder's sampler chain finds no viable candidate, the
request fails with `ErrorCode::invalid_argument`. Empty backend logits are
`backend_error`.

**Reason:** with valid logits, only the request's own policy (bias,
penalties or constraints) can exclude every token, so retrying cannot help. That is a
caller error, not a backend or availability failure.

## ADR-018 — Sampling module keeps `sonder/sampling/` include path

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** The sampling module's public headers stay at
`src/sampling/include/sonder/sampling/`, an exception to the
`sonder/inference/<module>/` convention. New modules follow the convention.

**Reason:** moving would touch 39 include lines in 25 files for no functional
gain. If the public include set is frozen for 1.0, it can be done
mechanically with forwarding headers.

## ADR-019 — Telemetry stream identity and per-event level

**Date:** 2026-09-26. **Status:** accepted.

**Decision:** Every envelope carries `producer.instance_id` (the telemetry
bus id), and `event_id` is `<instance_id>-<sequence>` by contract.
`sampling.level` is the level the event was emitted at. `run_id` defaults to
the engine id. The Ollama timing helper uses `backend.*` names so it never
duplicates the session's `inference.*` events.

**Reason:** Observatory change requests (its `docs/telemetry-schema.md` @
f5e3ff5): one counter numbers all of an engine's events, so gap detection
must key on the bus instance, and consumers need to know what a lower level
would drop. All additions are compatible with envelope v1. Full status in
[TELEMETRY.md](TELEMETRY.md).

## ADR-020 — Local HTTP boundary with Sonder Runtime (`sonder-infer serve`)

**Date:** 2026-09-26. **Status:** proposed, awaiting owner sign-off
(ecosystem integration contract v1; the contract review asks for sign-off
by the owners of all three repositories before this is marked accepted).

**Decision:** Sonder Runtime reaches Sonder Inference over local HTTP served
by `sonder-infer serve` (module `src/server`, `SONDER_HAS_SERVER`), not
in-process through the C ABI. The API is an OpenAI-compatible subset plus
Sonder extensions (health, models, backend identity), versioned as
`api_version` 1 under `/v1` ([SERVER.md](SERVER.md)). The same listener
serves live Observatory telemetry as Server-Sent Events and NDJSON with a
`/.well-known/sonder-telemetry` discovery document. The server is written
in-house over POSIX sockets and Winsock, like the ADR-013 client.

**Reasons:** a native crash must not take down the Python runtime; one
listener serves both inference and live telemetry; Runtime already has an
OpenAI-compatible transport to reuse; and no third-party server code is
approved (ADR-010, LICENSE_REVIEW.md; cpp-httplib stays test-only).

**Constraints:**

- Loopback by default; a non-loopback bind needs a bearer token. There is no
  TLS server in v1: remote use needs a TLS-terminating proxy.
- No WebSocket in v1: browsers cannot send `Authorization` on a WebSocket
  handshake, and SSE/NDJSON avoid in-house WebSocket framing.
- Inference never emulates the Ollama API and never presents itself as
  Ollama.
- The C ABI is untouched (`SONDER_ABI_VERSION` stays 1). Chat, session
  metadata and a telemetry callback in the C ABI remain follow-ups.
- Telemetry stays off the critical path: the live hub is a `TelemetrySink`
  on the bus writer thread with a bounded ring and bounded per-subscriber
  queues (drop-oldest, counted).

This resolves the transport item that [SCAFFOLD.md](SCAFFOLD.md) listed as
undecided.


## Python token iterator delivery bound — 2026-10-04

Bound pending iterator output to 64 chunks with lossless consumer backpressure.
A private cancellation epoch and nonblocking completion signal release stalled
callback delivery during cancel/close, including requests that have not entered
the native call yet. This changes no public API or C ABI; response text remains
retained and the bound does not imply a total byte-memory cap. See
[integration/python-stream-backpressure.md](integration/python-stream-backpressure.md).


## SDK session correlation prefix — 2026-10-04

Add a separate ABI-v1 versioned metadata record and session-create export for
caller session/run/agent/task IDs. Reuse the HTTP ASCII 1–128-byte correlation
policy, copy ownership at creation, and preserve original-create defaults and
older-library loading. IDs are telemetry metadata independent of text capture;
no prompt-derived IDs, scheduling hints or telemetry callback are introduced.
See [integration/cabi-session-metadata.md](integration/cabi-session-metadata.md).

## Positive mock event provenance — 2026-10-04

Carry known mock work on each telemetry context instead of changing the whole
engine's classification when it registers a backend. Positive synthetic
evidence overrides a host-wide false setting; unknown contexts preserve the
host policy. This gives SDK recordings the same existing envelope label used
by the HTTP server, while supporting mixed engines and keeping C ABI layouts,
capture and cursor contracts unchanged. See
[integration/sdk-mock-provenance.md](integration/sdk-mock-provenance.md).
