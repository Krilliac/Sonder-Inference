# GGUF inspected-content observation

The opt-in internal `server::detail::observe_gguf_artifact` entrypoint binds
native GGUF header observations to a caller-trusted full SHA256 over the same
sequential stream. Existing header-only APIs, estimators, backend identities,
HTTP metadata and placement policy retain their contracts.

Integration documentation needed outside this module:

- Append the inspected-content boundary and limitations to `docs/PLACEMENT.md`.
- Record the qualified foundation in `docs/ROADMAP.md`, leaving loader identity,
  descriptor/HTTP propagation and placement follow-ups incomplete.
- Add `docs/integration/gguf-artifact-observation.md` with actual qualification
  receipts and strict read-cap/cooperative cancellation/deadline semantics.

These updates require completed qualification. This note is not an execution
receipt. No tensor extent validation, model loadability, immutable filesystem,
trusted-digest provenance or backend-loaded association is certified.
