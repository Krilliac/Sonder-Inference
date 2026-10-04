# Python stream backlog and lifecycle

An unchanged ABI-v1 shared library can deliver a long upstream response faster
than a Python iterator consumes it. Previously `TokenStream` used an unbounded
queue. A loopback synthetic `/api/generate` response containing 2,048 Unicode
chunks reproduced 2,049 queued entries (including completion) with a paused
consumer on main `52d314c9c872`.

The iterator now holds at most 64 queued entries and uses bounded polling to
backpressure its producer without dropping normal output. Native cancellation
cannot interrupt Python while it waits inside a callback, so each Session tracks
a private cancellation epoch. Stream callbacks observe epoch changes, stream
close and session close. A completion event plus a best-effort sentinel avoids
blocking the worker's finalizer when the queue is full. Errors propagate after
queued chunks drain. The public API and C ABI are unchanged.

The bound covers pending delivery count only; chunks have variable byte lengths
and the final result retains response text. Consumer stalls can retain session
and scheduler admission resources. Closing abandoned streams is required,
just as before. Telemetry privacy/capture, producer cursors, engine batching,
Runtime effect recovery and rollback remain owned by their existing layers.

## Reproduce

Build the shared library and install the binding's test extra, then run:

```sh
PYTHONPATH=bindings/python/src SONDER_INFERENCE_LIBRARY=build/shared/libsonder_inference.so python -m pytest bindings/python/tests -q
python scripts/stress_python_stream.py --library build/shared/libsonder_inference.so --cycles 4 --out /tmp/python-stream-stress.json
```

Eight subprocess cases use actual library handles: lossless ordered Unicode
HTTP delivery, stalled-consumer stream close/session cancel/session close/engine
close, an injected worker exception after queued output, and cancellation/stream
close before native request entry. The bounded stress driver repeats the cohort
with 1, 2 and 4 concurrent workers, no retries and process deadlines. The upstream
is synthetic loopback HTTP; elapsed time includes a deliberate paused consumer
and test startup and cannot measure provider/model throughput or quality.

Cross-module integration: the root integrator changes only Python API internals,
its tests and this reproducible script, with the README and this document. No
schema, dependency, C header, backend or scheduler changes are required.

## Initial qualification

The paused-consumer regression failed on unchanged main with 2,049 queued
entries. The candidate's eight delivery/lifecycle cases pass; its full real
shared-library Python suite passes 96 tests and strict native CTest passes all
924 tests. A first stress-driver run (`--cycles 1`) passes 56 cases without
retries across 1/2/4 workers. Exact committed-revision qualification and hosted
platform results are recorded in the pull request; these initial working-tree
receipts do not authorize a merge by themselves.
