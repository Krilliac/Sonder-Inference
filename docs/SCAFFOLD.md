# Scaffold status

Created 2026-09-26 as documentation and structure only. **Superseded the same
day by the first implementation slice** at the owner's request; this page now
records the layout and what remains undecided.

## Included

- [Source ownership](../src/README.md) with implemented modules.
- [Test suite](../tests/README.md) (doctest + CTest).
- [Contract workspace](contracts/README.md).
- [Proposed ecosystem boundaries](BOUNDARIES.md).
- Build system: CMake presets (`CMakePresets.json`), `cmake/`, `scripts/build.ps1`.
- CI: `.github/workflows/ci.yml` (windows-latest + ubuntu-latest).
- CLI: `tools/sonder-infer/`. Benchmark corpora and snapshots: `bench/`.
- Repository editing conventions and contribution instructions.

## Decided since the scaffold

- Implementation language and build system: C++20 + CMake presets (ADR-011).
- API boundary: public C++ headers plus a stable C ABI (ADR-011).
- Telemetry format: Observatory envelope v1 JSONL (ADR-012).
- Test framework: doctest via FetchContent, test-only (ADR-014).
- CI jobs: configure/build/test on Windows (MSVC) and Linux.
- Project license: Apache-2.0 (ADR-015, [LICENSE](../LICENSE), [NOTICE](../NOTICE)).
- Transport between Sonder Runtime and Inference: local HTTP served by
  `sonder-infer serve`, OpenAI-compatible subset plus Sonder extensions, with
  live telemetry over SSE/NDJSON on the same listener (ADR-020,
  [SERVER.md](SERVER.md)).

## Still undecided

Package/distribution model and the first supported model/quantization
matrix. Directory
ownership remains provisional; module work streams are listed in
[MODULES.md](MODULES.md).

## Next design work

1. Pin and license-review a llama.cpp revision; implement the direct backend
   behind `SONDER_WITH_LLAMA_CPP`.
2. Agree the first supported Runtime use case on the ADR-020 boundary.
3. Define the first model/quantization matrix and extend the benchmark corpus
   to the workload families in [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md).
4. Start the Phase 2 scheduler and logical KV manager.
