# C++ token callback exception recovery

A public C++ `TokenCallback` is not `noexcept`. An exception from first-chunk
or later delivery previously skipped the request epilogue, leaving a session
running and any acquired scheduler reservation outstanding. A local failure
guard in `Session::run_request` now recovers this lifecycle; it does not
retry a callback, request body, backend call or runtime finalization.

The guard exists before activation and is armed only after the cancellation
source is constructed and running is published. It finalizes an acquired
scheduler id with `failed` once, before reopening the session. The attempt is
marked before calling `RequestRuntime::finish`, which is not `noexcept`.
Secondary runtime cleanup and telemetry exceptions are contained independently
so they cannot replace the original callback exception or skip the separate
session-state reset. Running becomes idle, cancellation state is cleared,
last outcome is failed and scheduler-rejected is false. A concurrently closed
session remains closed. The original exception is neither converted to a
successful partial result nor changed into a cancellation outcome.

Generic failure telemetry uses the existing request/session/parent lineage,
existing summary field types and fixed internal classification. It never
copies the callback exception message. The chunk count records deliveries
already attempted; unreturned backend statistics remain their defaults. Each
terminal emission is attempted at most once, and the bounded telemetry bus
may drop it. No delivery guarantee or cursor/batching change is implied.

Callback-visible chunks and arbitrary user or provider effects remain
already delivered. They are not rolled back or replayed. Runtime effect
journals, explicit reconciliation, owner epochs and upgrade rollback remain
owned by Runtime. Logical KV sequence/reservation release is distinct from
physical backend state and cached evictable prefixes may remain.

The existing C ABI exception barrier still converts C++ exceptions to
`SONDER_ERROR_INTERNAL`. The existing thread-local last-error message policy
is unchanged; this change adds no exception-message telemetry. C ABI layouts,
version, Python cancellation epochs/iterator bounds and close synchronization
are untouched.

Limits: this is ordinary callback-exception lifecycle recovery, not blanket
allocator, synchronization or backend exception safety. `finish` may itself
partially fail; its failure is contained, not retried or represented as proven
complete runtime cleanup. A mutex failure is contained to preserve the
original exception but cannot guarantee restored state. Runtime submission
that partially commits before returning an id is a separate boundary. Backend
worker teardown and process death are outside this slice.

The regression controls use only the deterministic mock backend: first/mid
chunk exceptions, generic and native-chat routes, sampler/backend streams,
gate/account/disabled modes, unchanged exception identity (standard and
nonstandard), preserved side effects, immediate reuse, cancel/close, exact
logical sequence release, private lineage, independent sessions and the
existing C ABI barrier. Existing queued cancellation, callback-false stop,
backend Status failure and ordinary success controls remain mandatory.

## Local qualification — 2026-10-06

An isolated offline `linux-debug` configure and build completed with GCC
14.2.0, TLS and llama.cpp disabled, using the previously reviewed cached
dependencies. All six new CTest nodes passed; the full suite passed all 945
tests, including those six, with no errors, failures or skips in 68.31 seconds.
The full suite also includes the existing synthetic telemetry stress controls.

A separate standalone driver linked the unchanged qualified baseline archive
and newly built candidate archive against their respective headers. The two
baseline controls reproduced the first/mid-chunk exception defect: the session
remained running, logical reservations remained, and immediate reuse returned
`invalid_state`. The candidate completed 512 failure/reuse pairs across four
independent sessions sharing each engine in gate and account modes. It verified
4,096 reuse callback chunks, 128 cancel-before-throw pairs, two additional
close-before-throw controls, and exactly one failed terminal and logical free
per each of the 514 failed request IDs. Final tracked requests, sequences and
pinned blocks were zero; all workers joined. The driver checked the original
exception object, type and payload inside each live catch and checked that its
private marker text never entered telemetry.

Normal-call cost used telemetry and scheduling off, 32 warmups per dataset
and four batches of 128 measurements per dataset and binary. These are
synthetic Debug/O0 session/mock wall times; checks and output serialization
occurred outside the timed intervals.

| Dataset | Baseline median / p95 (microseconds) | Candidate median / p95 (microseconds) |
| --- | --- | --- |
| No callback | 74.719 / 105.256 | 74.319 / 125.393 |
| Normal callback | 75.430 / 104.434 | 75.180 / 156.492 |

The candidate's observed tails were higher. Fixed binary/dataset order, Debug
code, host variability and CPU-clock quantization prevent a production cost
bound or causal speedup claim. Raw wall nanoseconds and process CPU ticks were
retained for all 2,048 measured requests. This is lifecycle/session-wrapper
qualification, with no live provider calls, model quality or model throughput
measurement.

All twelve supervised configure/build/test/driver stages completed with their
expected zero exits, unchanged preservation guards and strict owned-process
cleanup. The baseline archive embeds build stamp `e15ea89b31d6`: it was built
during the prior GGUF source qualification before its public commit adoption.
Its frozen archive and source provenance were reused, without rebuilding or
rewriting it. The candidate embeds base stamp `cc9d2ce9aaac` plus the seven-file
callback patch; these local results do not assert a later public revision's CI.

Local qualification receipt SHA-256 values:

- Native gates: `5e4317ae02c09bc6fe08b9e11f5a1ece27b6964240027202022112d2155dae44`.
- Standalone driver: `42a0ae3b1f6432f36e4cda93d2d91f30bb6f09dda64e49dffd87fd90b322e5e7`.
