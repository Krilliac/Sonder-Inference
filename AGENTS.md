# Repository instructions

## Current phase

Implementation, as of 2026-09-26 (owner request: "start building these out").
The first bounded slice (Phase 0 contract items plus the start of Phase 1) is
in place: a C++20 engine core with a stable C ABI, a deterministic mock backend
for tests, an Ollama compatibility adapter, Observatory telemetry, a benchmark
harness skeleton, and a doctest/CTest suite. See [ROADMAP](docs/ROADMAP.md) for
exactly what is done and [SCAFFOLD](docs/SCAFFOLD.md) for the layout.

## Working rules

- Read README.md, docs/ARCHITECTURE.md, docs/DESIGN_DECISIONS.md, and
  docs/ROADMAP.md before changing structure or public API.
- Engine core is C++20 built with CMake presets (ADR-011). The C ABI in
  `include/sonder_inference.h` is append-only; bump `SONDER_ABI_VERSION` on
  any incompatible change.
- Build and test before committing:
  - Windows: `powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test`
  - Linux/macOS: `cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug`
- Optional areas are modules auto-included by root CMake (docs/MODULES.md).
  Feature branches stay inside their module directory and list any other
  needed change in `INTEGRATION_NOTES.md` for the integrator.
- Tests must exercise real behaviour. Do not report empty or placeholder tests
  as passing. Tests must not require network services or model weights; live
  checks (e.g. Ollama) are opt-in through environment variables.
- The mock backend is for tests and harness development only. Keep it clearly
  labelled and never cite it for quality or performance claims.
- No upstream code may be vendored, linked, or fetched until its license is
  checked at a pinned revision and recorded in docs/LICENSE_REVIEW.md.
- Keep secrets, model weights, generated outputs, and local runtime state out
  of Git. Only small, reviewed benchmark snapshots belong in `bench/results/`.
- Telemetry must stay off the critical path: bounded queues, drop-and-count
  under pressure, no blocking I/O in decode.
- Preserve existing architecture and research documents; extend them.
- Record unresolved design choices explicitly rather than inventing contracts.
- Run `git diff --check` before committing; follow .editorconfig and
  .gitattributes (LF for source, CRLF only for `.ps1`).
