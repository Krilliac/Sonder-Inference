# Telemetry sink exception isolation qualification

A sink exception previously escaped the writer thread and terminated the host.
The writer now retires that distinct sink object, including duplicate
registrations, and continues delivering to healthy siblings. The C++
`failed_sinks()` counter is separate from queue-pressure drops. There is no
automatic retry, exception-message capture, C ABI change or envelope change.
C++ clients must rebuild because `TelemetryBus` has a new private counter.

## Regression and stability controls

On 2026-10-04, the ordinary throwing-write public-API regression compiled with
production sources from main `e4aa2ae343ec52d3e98b45ba0b990a0fa1f4c1c9`
terminated with SIGABRT. Core dumps were disabled. Eight new cases, including
the repaired regression, pass: write/event override, flush, non-standard
exception, duplicate aliases, all-sinks-failed backlog, destruction outside the
bus mutex, throwing ostream, bounded pressure/drop accounting and mock engine
generation with raw text capture disabled. Several behaviours share a case.

The Linux Debug build with `SONDER_WARNINGS_AS_ERRORS=ON` passed all 911 CTest
cases in 13.65 seconds. These local observations precede publication; hosted
checks on the exact final revision are required before merge.

## Bounded concurrent qualification

The standalone harness [stress_telemetry.cpp](../../scripts/stress_telemetry.cpp)
links the actual library. It uses numeric synthetic event IDs and no model or
provider calls. Run from the repository root after building `linux-debug`:

```sh
c++ -std=c++20 -Wall -Wextra -Werror -pthread -Iinclude \
  scripts/stress_telemetry.cpp build/linux-debug/libsonder_inference.a \
  -o /tmp/sonder-telemetry-stress
timeout 45s /tmp/sonder-telemetry-stress pressure > /tmp/telemetry-pressure.json
timeout 45s /tmp/sonder-telemetry-stress > /tmp/telemetry-healthy.json
```

Pressure mode runs eight lifecycles each of healthy-only, throwing write,
throwing flush and all-sinks-failed configurations: 32 lifecycles, six emitter
threads, 512 attempts per thread and queue capacity 64, totaling 98,304 attempts.
It checks accepted IDs against delivery, byte-identical healthy siblings,
contiguous sequence/event IDs, producer identity, the synthetic label, exact
drop accounting, retirement without retry, and repeated flush/shutdown.
All-sinks-failed mode distinguishes queue drops from later filtered emissions.
Every admitted backlog drains; it does not claim delivery to failed sinks.

The local run passed all controls. Healthy-sibling runs admitted 9,205, 9,483
and 10,318 events respectively; the remaining attempts were counted queue
drops. All-sinks-failed runs admitted 833 events, counted 4,440 queue drops and
filtered 19,303 later attempts. Maximum final drain was 0.533 ms. Admission
counts vary with scheduling and are not pass thresholds or throughput targets.

## Healthy-path timing

For a healthy-only baseline control, compile the same harness against main
`e4aa2ae` with `-DSONDER_BASELINE_TELEMETRY=1`, using its matching headers and
library. That macro omits the new counter and disables fault modes. Run the
baseline and candidate executables in alternating baseline/candidate and
candidate/baseline order. Default mode uses one emitter, two memory sinks and
queue capacity 4,096, with eight independent lifecycles per process.

Sixteen lifecycles per build delivered all 65,536 admitted events with zero
drops. Baseline median elapsed time per 4,096 events was 288.807 ms (range
281.680–305.909); candidate median was 281.939 ms (277.389–286.640). This is a
local Debug transport check, including serialization, emitter-thread startup
and drain; timing starts after bus and sink construction. The
overlapping ranges do not establish a performance improvement. Neither these
timings nor mock HTTP results measure provider/model quality or throughput.

Separately, the existing mock HTTP/SSE lifecycle qualification completed 204
requests across three restarts, with zero errors, all twelve admitted streams
drained on SIGINT and all three ready files removed.

## Limits

This isolates exceptions reported by sink callbacks. Silent ostream fail bits,
allocation failures outside callbacks, indefinitely blocking or reentrant sink
callbacks/destructors and recovery/re-enabling of a retired sink remain outside
this change. Raw text consent, bounded batching, producer cursors, effect
recovery and rollback contracts are unchanged. See [TELEMETRY](../TELEMETRY.md)
for the delivery and capture contract.
