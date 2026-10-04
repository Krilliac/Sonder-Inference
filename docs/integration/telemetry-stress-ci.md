# Routine telemetry stress qualification

With `SONDER_BUILD_TESTS=ON`, CMake builds `sonder_telemetry_stress` from the
published [harness](../../scripts/stress_telemetry.cpp), linked to the actual
`sonder::inference` library. The target uses the repository warning policy and
the build's existing sanitizer instrumentation. It adds no dependencies,
provider calls, model weights, or production configuration.

Three CTest controls run in the existing full-suite platform and sanitizer
lanes. Each has a 60-second timeout, `RUN_SERIAL=TRUE`, and the labels
`stress`, `synthetic`, and `telemetry`:

| CTest suffix (`sonder.telemetry.stress.`) | Harness mode | Bounded workload |
| --- | --- | --- |
| `healthy_stream` | `healthy-stream` | Eight lifecycles, one producer, 4,096 events and queue capacity 4,096 per lifecycle; built-in ostream plus two memory sinks |
| `callback_pressure` | `pressure` | Eight repetitions of healthy, callback-write failure, callback-flush failure, and all-sinks-failed controls |
| `stream_pressure` | `stream-pressure` | Eight repetitions of healthy, default-mask short write, default-mask flush failure, and all-sinks-failed controls |

Each pressure lifecycle uses six producers, 512 attempts per producer and a
queue capacity of 64. Each pressure test therefore exercises 32 lifecycles and
98,304 attempts. Assertions check accepted-event accounting, healthy sibling
delivery, original producer cursors and event IDs, synthetic labels, final drop
reports, distinct failed-sink retirement, no retry, and repeatable drain and
shutdown. The healthy stream control additionally checks identical JSONL bytes,
unchanged exception mask, and zero queue drops. Failures return a nonzero exit
code; timings and per-lifecycle counters are emitted as JSON in CTest output.

To run only these controls after a normal configure/build:

```sh
ctest --test-dir build/linux-debug -L stress --output-on-failure
```

For verbose JSON output use `-V`; for a preserved machine-readable test receipt
add `--output-junit telemetry-stress.xml`. Ordinary full-suite runs include all
three controls without an opt-in flag. `SONDER_BUILD_TESTS=OFF` omits both the
target and registration, including in the shared-library Python CI lanes.

This qualifies synthetic transport correctness, bounded pressure and lifecycle
stability. It does not measure provider throughput, model quality, or native
inference performance. Serial CTest execution isolates these controls from
other tests in that invocation; it does not reserve the host or eliminate
unrelated system load. No production API, privacy/consent, queue, producer
cursor, effect recovery or rollback contract changes.
