# Built-in telemetry stream error qualification

File and ostream sinks now check stream state after writing an envelope and
its newline, and after flushing. Failed operations enter the existing writer
retirement path even with the default exception mask. A failed destination is
counted once, duplicate registrations are retired together, and healthy sinks
continue receiving the original producer cursors. Caller-owned ostream masks
are never changed. Open-error status, append behavior, JSONL bytes, queue/drop
accounting, capture consent, public headers, class layout and C ABI are unchanged.

The bus does not retry a failed sink or capture its exception contents.
Custom sinks must still signal failures by throwing. This adds no recovery for
retired sinks or interruption of arbitrary blocking callbacks/destructors.

## Regression controls

On 2026-10-04, the ordinary short-write/flush regression on unchanged main
`fa11d36e8601b8dfa36865d87bcce750f1e9ca0b` failed four retirement-counter
assertions: both default-mask streams entered a bad state while
`failed_sinks()` remained zero. Healthy delivery, masks and cursor checks
passed. With the writer-only checks, all 212 assertions in that case pass.

Five added CTest cases cover default-mask short writes and failed flushes,
already-failed streams, healthy file/ostream byte parity and append/open errors,
Linux `/dev/full` I/O failure, and continued eight-token mock generation with
private prompt and token-text capture disabled. The Linux file-error control
uses append mode and does not fill a disk. It is compiled only on Linux;
portable stream and healthy-file controls also run on other platforms.

The local Linux Debug build with `SONDER_WARNINGS_AS_ERRORS=ON` passed all 916
CTest cases in 53.58 seconds. These observations precede publication; hosted
checks on the exact final revision are required before merging.

## Bounded stress and healthy timing

The actual-library harness [stress_telemetry.cpp](../../scripts/stress_telemetry.cpp)
retains its original modes and adds `stream-pressure` and `healthy-stream`:

```sh
c++ -std=c++20 -Wall -Wextra -Werror -pthread -Iinclude \
  scripts/stress_telemetry.cpp build/linux-debug/libsonder_inference.a \
  -o /tmp/sonder-telemetry-stress
timeout 45s /tmp/sonder-telemetry-stress stream-pressure > /tmp/stream-pressure.json
timeout 45s /tmp/sonder-telemetry-stress healthy-stream > /tmp/healthy-stream.json
```

Stream pressure runs eight lifecycles each of healthy, short-write, failed-flush
and all-sinks-failed configurations. All 32 lifecycles passed: six emitter
threads, 512 attempts per thread, queue capacity 64 and 98,304 total attempts.
Controls verify admitted IDs against delivery, healthy sibling bytes, cursors,
synthetic producer identity, separate pressure/filtered accounting, default
exception masks, no retry and repeated flush/shutdown. The initial run admitted
26,760 events, counted 49,787 queue drops and filtered 21,757 later attempts;
maximum final drain was 0.533 ms. These scheduling-dependent counts and times
are observations, not performance targets.

Healthy-stream mode adds a real built-in ostream destination to two memory
siblings and verifies byte-identical JSONL. A paired comparison used the same
harness against actual main `fa11d36e` and the candidate library in alternating
baseline/candidate and candidate/baseline order. Each build completed sixteen
lifecycles of 4,096 events: 65,536 accepted, zero drops. Baseline median elapsed
time was 283.990 ms (range 278.311–303.855); candidate median was 281.069 ms
(272.184–302.902). These local Debug runs overlapped full CTest activity; the
overlapping ranges do not establish a performance improvement.

All inputs are synthetic. Timing includes serialization, emitter-thread startup
and drain after sink/bus construction. Neither these controls nor mock engine
tokens measure model, GPU or provider quality or service throughput. Privacy,
bounded batching, producer cursors, effect recovery and rollback contracts remain
unchanged. See [TELEMETRY](../TELEMETRY.md) for delivery and capture limits.
