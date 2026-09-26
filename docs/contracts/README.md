# Contract workspace

Reviewed interface specifications that other repositories may depend on.

Defined in code (v0.1, 2026-09-26):

- **C ABI** — [`include/sonder_inference.h`](../../include/sonder_inference.h),
  `SONDER_ABI_VERSION 1`. Append-only status codes, `struct_size`-versioned
  structs, opaque handles, thread-local error text (ADR-011).
- **Telemetry** — Observatory envelope `sonder.observatory.event/1`, owned by
  [Sonder Observatory](https://github.com/Krilliac/Sonder-Observatory); the
  events this repository emits are listed in
  [OBSERVATORY_CONTRACT.md](../OBSERVATORY_CONTRACT.md#implementation-status-v01).
- **Benchmark results** — `sonder.inference.bench/1` and corpus
  `sonder.inference.corpus/1` (see [BENCHMARK_PLAN.md](../BENCHMARK_PLAN.md#harness-v01)).

Not yet agreed: transport between Sonder Runtime and Inference, authorization,
budgets/resource limits across repositories, and compatibility policy beyond
the ABI version number.
