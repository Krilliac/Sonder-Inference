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
