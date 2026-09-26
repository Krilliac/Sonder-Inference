# Integration notes: `feat/scheduler`

Owner area: `src/scheduler/` only (plus this file). No root CMake, presets, CI,
core headers, or other modules were edited.

## Build wiring

- `src/scheduler/CMakeLists.txt` follows the root module convention: when
  `sonder_inference` and the `SonderModules.cmake` helpers exist it calls
  `sonder_module_sources(...)`, `sonder_module_include_directories(include)`,
  `sonder_module_define(SONDER_HAS_SCHEDULER)` and
  `sonder_add_module_tests(scheduler ...)` (doctest, `sonder.scheduler.<case>`).
- It also creates `sonder::scheduler` as an alias of `sonder_inference` for
  explicit linking.
- Standalone mode (`cmake -S src/scheduler`) builds a `sonder_scheduler` static
  library and fetches the same pinned doctest 2.5.3 (same SHA-256).
- The module has no dependency on the core library or other modules; it only
  uses the C++20 standard library.

## Public API (namespace `sonder::inference::scheduler`)

Headers: `#include "sonder/inference/scheduler/scheduler.hpp"` (and
`simulator.hpp`, `kv_capacity.hpp`, `stats.hpp`, `config.hpp`, `clock.hpp`,
`types.hpp`).

- `Scheduler(SchedulerConfig, const Clock&, KvCapacity&)`
- `submit(RequestSpec)`, `cancel(id)`, `plan_step()`, `complete_step(plan, outcome)`
- `stats()`, `timeline(id)`, `timelines()`, `queue_order()`, `effective_rank(id)`,
  `decide_preemption_mode(kv_tokens)`
- `Simulator` for deterministic workload runs (no backend).

## Seams for other work streams

- **KV cache (`src/cache/`)**: the scheduler talks only to the abstract
  `KvCapacity` interface (`block_size_tokens`, `total_blocks`, `free_blocks`,
  `try_reserve(id, blocks)` all-or-nothing, `release(id)`, `blocks_held(id)`).
  The engine (or the cache owner) should provide a thin adapter from the block
  manager to this interface. `FixedKvCapacity` is a test/simulation pool only.
  Prefix-sharing discounts are not modelled yet; an adapter can report
  shared-prefix blocks as already held.
- **Engine/Session**: map `SessionOptions::priority` / runtime metadata into
  `RequestSpec` (`workload`, `priority_override`, `task_id`, `cancellable`,
  `sequence_fingerprint`). The engine drives `plan_step()` / `complete_step()`
  per backend step and passes EOS via `StepOutcome::finished_early`.
- **Swap preemption**: `PreemptionMode::Swap` is a decision stub. The engine or
  cache must perform the actual KV move; the scheduler assumes swapped KV is
  restorable and re-reserves its blocks on resume.
- **Telemetry**: `SchedulerStats` / `RequestTimeline` carry the metrics listed in
  docs/SCHEDULER.md (queue time, TTFT, ITL, occupancy, prefill/decode share,
  preemptions by reason and mode, starvation age). Mapping them to Observatory
  events is left to the telemetry owner.

## Tests

33 doctest cases (27 unit, 6 simulation). All pass in these builds on the box:
GCC 14 Debug standalone (`-Werror -Wconversion -Wshadow`), Release,
ASan+UBSan, and a mock root project using the real `cmake/Sonder*.cmake`
helpers with `SONDER_WARNINGS_AS_ERRORS=ON`. MSVC is untested locally; CI
will cover it.

The simulation tests check these invariants: step budgets and KV capacity are
never exceeded, every request completes, interactive TTFT p95 is lower than
the worker and background p50, interactive ITL stays bounded with zero
interactive preemptions, batching averages more than 4 sequences per step,
repeated runs are deterministic, cancellation frees capacity, and the
starvation guard keeps background queue wait under 700 ms (versus more than
8 s without the guard).

## Open questions

- Default budgets and aging/starvation constants are placeholders until
  benchmarked (docs/BENCHMARK_PLAN.md).
- Not implemented yet: prefix-aware placement, multi-device placement,
  deadline classes beyond rank, speculative allowance.
