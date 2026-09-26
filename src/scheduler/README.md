# Scheduler

Phase 2 inference scheduling policy (see [docs/SCHEDULER.md](../../docs/SCHEDULER.md)).
Pure policy: it decides *what* runs each engine step and never executes a backend.

Namespace `sonder::inference::scheduler`, headers under
`include/sonder/inference/scheduler/`.

| Header | Purpose |
| --- | --- |
| `types.hpp` | Workload classes, request state machine, `RequestSpec` metadata |
| `clock.hpp` | `Clock` interface and deterministic `SimClock` |
| `kv_capacity.hpp` | Abstract `KvCapacity` admission interface + `FixedKvCapacity` test pool |
| `config.hpp` | Budgets, chunk size, watermark, aging/starvation, preemption knobs |
| `scheduler.hpp` | `Scheduler`: queue, step planner, admission, preemption, cancellation |
| `stats.hpp` | Telemetry snapshot: queue time, TTFT, ITL, occupancy, preemptions, starvation |
| `simulator.hpp` | Deterministic step simulator with a synthetic cost model |

## Step protocol

```text
plan = scheduler.plan_step();          // decode first, then prefill chunks, then admission
backend executes plan.work             // outside this module
clock advances by the step duration
scheduler.complete_step(plan, outcome) // tokens emitted, EOS, completions
```

## Policy summary

- **Priority queue**: rank = workload class (or override) minus aging; requests
  older than `starvation_threshold_us` jump to the front and cannot be bypassed.
- **Continuous batching**: running decodes are scheduled first every step, so
  prefill never starves decode; per-step token, prefill-token and sequence
  budgets are enforced.
- **Chunked prefill**: prompts are split into `prefill_chunk_tokens` chunks
  interleaved with decode; the chunk that finishes prefill yields the first token.
- **Admission**: the whole (re)prefill context is reserved up front through
  `KvCapacity`, keeping `admission_watermark_blocks` free for decode growth.
  Bounded head-of-line bypass (`admission_lookahead`), never past an
  interactive or starving head.
- **Preemption**: KV pressure preempts the lowest-priority, newest sequence;
  urgent requests may displace lower classes (`priority_preemption_min_rank_gap`).
  `decide_preemption_mode()` picks recompute vs swap (stub). Every event records
  its reason, mode and beneficiary.
- **No-progress guards**: bounded requeue count, duplicate-sequence refusal,
  never-fits rejection at submit, one plan in flight at a time.

## Build and test

Integrated: the root `CMakeLists.txt` adds this directory automatically; tests
register as `sonder.scheduler.<case>`.

Standalone:

```sh
cmake -S src/scheduler -B build/scheduler -G Ninja
cmake --build build/scheduler -j 4
ctest --test-dir build/scheduler
```
