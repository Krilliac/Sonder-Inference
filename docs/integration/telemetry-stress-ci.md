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

## Qualification observations (2026-10-04)

Implementation revision `fbf4f6310280b645821d9ba27013e9c04cdc521a` passed a
strict Linux Debug build and all 919 CTest cases in 65.49 seconds. A Release
shared-library build with tests disabled passed and omitted the stress target
and registration. Repeating the registered controls with CTest `-j 8 -V` 24
times passed all 72 tests: 1,728 lifecycles and 5,505,024 synthetic attempts in
383.345 seconds, within a 720-second driver budget. Each invocation also had a
205-second process-group budget around the three 60-second CTest limits. Peak
observed child-tree RSS was 28,956 KiB, including CTest and its launched processes.

| Local Debug control | Samples | Median seconds | Maximum seconds |
| --- | --- | --- | --- |
| Healthy stream | 24 | 5.425 | 5.523 |
| Callback pressure | 24 | 5.082 | 6.284 |
| Stream pressure | 24 | 5.310 | 6.003 |

The three controls also passed individually in all seven hosted full-suite
lanes: Linux, Windows, OpenSSL, Schannel, ASan+UBSan, TSan and optional native
Windows. TSan passed all 919 cases in 24.15 seconds; its three stress controls
took 1.98, 2.82 and 3.00 seconds. ASan+UBSan passed all 919 cases in 24.42
seconds. Native Windows registered 946 cases: 945 passed and the existing
model-dependent integration fixture skipped; none of these stress controls
skipped. These are host/configuration-specific test costs, not a throughput
comparison or an inference speed claim.

Actual mock HTTP/SSE qualification passed 204 requests across three lifecycles
and twelve active shutdown drains with zero errors. All 30 ecosystem gates
passed in 70.8 seconds with Runtime `e42aefbd` and Observatory `9ea5f788`;
all fourteen process groups stopped and Git/home guards stayed unchanged.

The initial revision's three CodeQL analysis jobs and C++ SARIF upload passed,
but GitHub's PR summary remained neutral because its C++ comparison
configuration was unavailable. GitHub refused workflow and job reruns with
HTTP 403. That result is preserved and is not a clean comparison receipt.
These observations precede the documentation revision; fresh exact-revision
checks remain necessary before merging.
