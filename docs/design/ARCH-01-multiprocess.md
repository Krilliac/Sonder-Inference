# ARCH-01: local engine worker processes

Date: 2026-11-26. First implementation slice: isolated **mock** execution on
Windows and Linux. Mock output is synthetic, not inference or performance
evidence. This design extends [ARCHITECTURE](../ARCHITECTURE.md),
[TELEMETRY](../TELEMETRY.md) and [GPU_SUPPORT](../GPU_SUPPORT.md).

## Ownership and rollout

`sonder-infer serve` is the front door: HTTP API, authentication, admission,
scheduling, logical KV accounting, session/request identity, the telemetry bus
and live hub, and worker supervision remain in this process. A
`RemoteBackend` implements the existing C++ `Backend` interface. It forwards
execution over private local IPC to `sonder-infer worker` in the **same
executable**. The worker never opens a network listener.

The destination architecture is one worker per GPU, and later per engine
build. Its native engine is the in-process backend in `src/backends/llamacpp`.
A GPU-driver fault or GGML abort then loses that worker and its requests, while
the API process and other GPUs survive. This slice supplies one mock worker
per RemoteBackend, selected explicitly:

```text
sonder-infer serve --backend remote-mock --model mock:tiny --port 11437
```

The server resolves its own executable with the OS, not `argv[0]`, a shell,
`PATH`, or the current directory. `--mock-delay-ms` also applies to remote
mock, to exercise streaming and cancellation without weights or a GPU. Direct
`--backend mock` remains useful for existing unit tests. Existing native and
compatibility backends remain during this bounded slice; removing them is not
part of this diff. The worker engine factory is the future native insertion
point. This slice does not change `src/backends/llamacpp` or the C ABI.

The proxy deliberately does not advertise `token_logits`. Sampling and logits
stay in the worker; only selected text/token chunks, optional bounded top-K
candidate information and aggregate execution statistics cross the boundary.
The proxy advertises only operations its protocol actually implements. It
does not transfer KV snapshots, tensors, full logits, arbitrary telemetry
objects or executable code. Native chat dispatch forwards structured text
messages into the worker. Unsupported richer request features fail explicitly.

## Transport and trust boundary

The parent creates a fresh endpoint for every worker incarnation, before
starting the child. The child connects to it as a client.

| Platform | Endpoint | Access and process containment |
| --- | --- | --- |
| Windows | Local byte-mode named pipe with a cryptographically random name | Protected DACL granting only the current user access; first-instance creation; remote clients rejected. Child starts suspended, is assigned to a kill-on-close Job Object, then resumes. |
| Linux | Filesystem AF_UNIX stream socket in a private random directory under `/tmp` | Directory mode 0700 and socket mode 0600; peer UID checked. Child has its own process group and `PR_SET_PDEATHSIG(SIGKILL)` with the parent-race check. |

**ARCH-01 provides fault isolation, not a security sandbox.** The worker runs
with the parent's identity. Code execution in the worker is **not contained**:
it can act as the user, kill the parent, and on Linux escape
`PR_SET_PDEATHSIG` by forking (the signal is not inherited by forked children).
No AppContainer, namespaces or seccomp sandbox is implemented in this slice.

The guarantee is at the IPC boundary: a crashing, hung or misbehaving worker
cannot crash, hang or exhaust the parent **through the IPC channel**. The parent
validates metadata, bounds frames/output/counters, enforces deadlines and
retires invalid connections. This does not constrain arbitrary code using other
OS interfaces, nor forcibly unwind a blocked callback supplied by the parent.
Endpoint names are not authentication credentials; the same-user OS ACL
prevents unintended cross-user access. There is no TCP fallback or worker
network listener.

Follow-up: real sandbox (AppContainer/LPAC on Windows; separate uid,
namespaces, no_new_privs, seccomp/Landlock, process-tree containment on Linux).

Launch is always by absolute executable path with its directory as the
working directory. The child environment is built from an **exact allowlist**,
not copied and filtered. Windows supplies only the OS-derived `SystemRoot`
and `NoDefaultCurrentDirectoryInExePath=1`; Linux supplies only `LANG=C.UTF-8`.
No inherited `PATH`, `PATHEXT`, `COMSPEC`, `LD_LIBRARY_PATH`, `LD_PRELOAD`,
`DYLD_*`, `VK_ICD_FILENAMES`, `VK_LAYER_PATH`, `PYTHON*`, credentials, home
directory or GPU-loader overrides are forwarded. Future per-GPU selection must
add narrowly validated, explicit configuration rather than wildcard prefixes.

Windows starts with handle inheritance disabled. POSIX closes inherited file
descriptors, including server listeners and unrelated files, and installs
controlled `/dev/null` standard descriptors. Only the launch-error pipe is
temporarily retained, close-on-exec, so exec errors are distinguishable from
successful launch. Linux requires `close_range` (kernel 5.9 or newer); if the
syscall is absent or denied, launch fails rather than risk leaking descriptors
above a lowered file-descriptor limit. Exec confirmation is bounded to five
seconds. IPC is connected after exec. Failure to establish process
containment, a secure endpoint or a clean launch is fatal to that startup.

## Framing and protocol version 1

Every frame is a four-byte unsigned **big-endian** payload length followed by
exactly that many UTF-8 JSON bytes. Length must be in `1..1,048,576` inclusive.
Check the prefix before allocating the body. A zero or oversized length,
truncation or partial-frame timeout closes the broken connection. There is no
resynchronization by skipping bytes. A timeout before any frame byte arrives
can be retried. Writers submit prefix and payload in one contiguous buffer;
readers start a fresh payload deadline after receiving the complete header.
Reads/writes are deadline- and cancellation-aware.

After a valid handshake, a complete but invalid request is rejected with a
typed error and the worker stays alive. Decoder/schema failures become
`invalid_argument`, so the parent does not retire the worker or spend restart
budget. An unparseable envelope uses error ID 0 because it cannot be correlated.
Invalid controls/operations are also rejected without executing them. Broken
channels and failed handshakes still terminate the worker; successful shutdown
is the orderly exit path.

The common envelope is `{ "v": 1, "type": "...", "id": <uint64> }`.
On the wire v1 requires the existing JSON serializer's compact canonical form
(including its key order); reparsing and serializing must reproduce the exact
payload. This rejects duplicate keys, noncanonical numbers/whitespace and text
that the parser would otherwise repair. Both endpoints use that serializer.
Each operation has an exact set of fields and checked scalar types/ranges;
unknown fields are rejected rather than silently ignored. A version change
requires a new handshake; no permissive downgrade or mixed-version exchange
is supported. The parent and worker currently come from the same executable.

Handshake completes before the worker becomes available: the worker sends
`hello` with protocol version and engine, then the parent validates and returns
`ready` confirming both. Invalid
or incomplete handshakes are startup failures. Request IDs correlate all
responses and control messages; token indices must be contiguous. Model
descriptors returned across IPC are metadata, never process-local pointers.

The protocol covers discovery/probe, model load, generate, chat, token delivery,
credit acknowledgement, callback `stop`, cancellation, terminal success/error
and shutdown. Callback stop preserves successful statistics and reports
`StopReason::callback`; cancellation reports a cancelled request.
Sampling is serialized field by field, including explicit-field masks, exact
64-bit seeds, stop strings, biases and logprob options. A `null` bias denotes
negative infinity (a ban), because JSON numbers must be finite. The
worker validates the decoded SamplingConfig. Request fields unsupported by
this first version are refused before execution, not dropped during encoding.
The parent uses the worker decoder checks before sending both generate and
chat requests, including priority, finite nonnegative latency targets and
sampling-mask validation. Chat has one shared 1,024-message cap; the parent
validates the flattened `{role, content}` messages that cross IPC. The complete
outgoing execution envelope is also checked against the worker's parser limits.
Text chat preserves the boolean `separate_reasoning` output-routing hint:
`serve` sets it even for ordinary text requests. It does not enable thinking;
thinking controls, reasoning budgets, tools, grammar and profiling remain
unsupported. The worker rejects non-boolean values for this field.

The implementation is the authoritative v1 schema:
`src/worker/protocol.cpp`, `worker_main.cpp` and `remote_backend.cpp`. It uses
the existing bounded in-house JSON parser and introduces no wire dependency.

## Lifecycle, failure and restart budget

The supervisor owns endpoint creation, spawn, handshake, process monitoring,
restart pacing and final teardown. Readiness waits are bounded by
`startup_timeout + restart_max_backoff` (10 seconds by default), clipped by a
generation request's absolute deadline. A dead worker closes the stream and
fails the request with `ErrorCode::backend_error`; malformed worker responses
instead produce `protocol_error`. Already emitted tokens are not
retracted. **In-flight requests are never replayed**, even if no token was
observed: the parent cannot prove the old worker did not execute them.

A replacement gets a new endpoint and connection. Proxy models retain
immutable load options and resolve/load in the new worker, so no stale worker
pointer or model handle can be reused. Recovery permits a later request to
succeed without restarting `serve`. Idle crashes are monitored too.

Default restart policy:

- Initial launch does not consume a restart. Every subsequent launch attempt
  consumes one, including failed launches/handshakes.
- Delay grows exponentially from 100 ms, capped at 5 seconds. Waiting is
  interruptible by shutdown.
- At most five restarts are admitted in a rolling hour, using a monotonic
  clock and timestamp deque. A worker remaining healthy does **not** clear
  this history. Entries expire only when they age out of that hour.
- While the circuit is open, execution is unavailable and the monitor waits
  for budget to expire. It does not busy-loop spawning children.

Orderly shutdown stops admission, asks the worker to exit, waits a bounded
grace period and forcibly terminates the contained child if necessary. The
monitor is joined and IPC resources are released. If `serve` crashes or is
killed, the OS lifetime mechanism kills the worker independently of graceful
cleanup. Linux abnormal parent death can leave a private socket directory;
its name is never reused by a new incarnation. The Linux parent-death signal
kills the worker itself; worker engines must not launch detached subprocesses.
On both shutdown **and restart**, Linux gives forced termination/reaping a
250ms budget, including interrupted ownership observations, and uses nonblocking
`waitpid(..., WNOHANG)`. If the child is still unreapable (for example, in
uninterruptible D state), its PID and preallocated ownership slot transfer to
one process-lifetime detached reaper. Backend shutdown never joins that reaper.
Handoff does not mean that the child has released kernel/GPU resources.

On Linux only (Windows has no such pool), the process-wide pool has **64 slots total**, shared by pre-fork reservations,
live worker children and deferred children. A slot is reserved before `fork`;
when exhausted, launch fails with `unavailable` before creating a child. A
successful fork keeps its slot until `waitpid` returns the PID or `ECHILD`,
including failed exec/handshake paths and backend destruction. Pre-fork failures
return their reservation. Handoff allocates nothing and cannot throw; the
reaper sweeps the fixed 64 slots every 100ms, so retained ownership and polling
work stay bounded even across backend creation/destruction.

Each supervisor also retains its predecessor's reap completion, including when
launch fails after fork. While that completion is pending it creates no endpoint
or replacement child and consumes no restart attempts, regardless of rolling-hour
budget expiry. Requests return `unavailable` with an explicit unreaped-predecessor
message. Once the reaper confirms completion, the monitor observes it within its
20ms wait interval and resumes normal backoff/circuit admission. Shutdown can
interrupt this wait; the slot stays owned by the process-wide reaper. A later
exit is therefore reaped without allowing an unreaped predecessor and its
replacement to coexist.

## Back-pressure, scheduling and cancellation

Version 1 admits one active operation per worker and advertises a concurrency
limit of one. `serve` retains its existing admission/scheduling policy. This
slice does not implement GPU batching, multiplexed execution or independent
per-GPU queues.

Token delivery uses one-token credit: the worker cannot send another token
until the parent acknowledges consumption. There is no unbounded token queue
when an HTTP client is slow. OS buffers and the maximum frame size bound wire
buffering. The worker's control reader remains independent of generation and
can receive cancellation while generation is waiting for token credit.

Generation admission waits honor caller cancellation and shutdown. Metadata
commands (`probe`, `list`, `load`) use the same timed-lock/shutdown polling;
their admission wait is bounded by `request_timeout`, followed by the readiness
bound above. Their Backend interface has no caller cancellation parameter.
Callback refusal, client disconnect or explicit session cancellation sends
cancel to the worker. Cancellation is sticky and request-scoped. A terminal
response must be drained or the connection retired before another request
can use it. A worker that fails to stop within the bounded cancellation grace
is retired; a replacement cannot inherit the cancelled request. Token callback
exceptions also unwind the request and retire the connection safely.

Parent-enforced v1 stream limits (even if the worker ignores token credit):

- At most the requested `max_tokens` frames, with contiguous indices. Terminal
  chunk and completion counts must equal the number of observed frames; the
  parent supplies those observed counts to Session. Under the ACK/control writer
  lock, cancel/stop snapshots whether the worker still held credit (the initial
  request's credit or an ACK already sent). If so, at most one token at the next
  expected index may be drained without delivery or ACK; it still consumes the
  byte/token budgets. A control that replaces an ACK allows zero further tokens.
  A second token, an invalid index or a cap violation retires the connection.
  The terminal frame must still arrive within the original cancellation grace.
  After a terminal response, an idle supervisor read retires unsolicited frames;
  a later request also rejects an old ID.
- At most 64 KiB of text, reasoning and candidate strings/float payloads per
  token, and `min(max_tokens * 64 KiB, 64 MiB)` cumulatively per request. Checked
  subtraction enforces the byte budget before callbacks/Session accumulation.
- `RemoteBackendOptions::inactivity_timeout` defaults to 300 seconds and resets
  after the consumer callback and ACK, excluding consumer backpressure.
- `cancel_grace` defaults to 2 seconds, measured from cancel/stop dispatch;
  subsequent worker activity cannot extend it.
- `request_timeout` defaults to an absolute 30 minutes, including admission,
  readiness and consumer backpressure. A separate request watchdog closes IPC
  on absolute/grace expiry even if reads or writes are blocked. These three
  durations are configurable in RemoteBackendOptions, not client HTTP fields.
  A watchdog deadline retires the worker and therefore consumes restart budget
  on replacement, even when the worker was merely slow rather than faulty.

Loaded descriptors must name exactly the requested model and the expected
worker engine (`mock` in v1). Descriptor strings are bounded to 4 KiB, context
length to 1..16,777,216 tokens, and model size to 1 TiB. Prompt counters are
capped at 16,777,216; completion/chunk counters at 1,048,576 plus the tighter
per-request observed-frame checks. Worker timing counters are untrusted
diagnostics bounded to 24 hours in nanoseconds, not clocks used for policy.
Session checks additions involving returned counts and resume offsets.

Normal CLI commands register RemoteBackend only when `remote-mock` is selected.
`backends --json` still lists its synthetic metadata with `probed:false` and
`available:false`, without creating a worker. `backends --backend remote-mock`
explicitly probes it. CLI model defaults, warnings and statistics classify it
as synthetic, and the server identity endpoint uses the mock identity digests
while retaining the `remote-mock` backend name.

## Telemetry

`serve` remains the single Observatory producer and owns the bounded telemetry
bus/live hub. Worker responses carry aggregate counts and timings through
`GenerateStats`; the existing Session emits request, prefill/decode and token
events with the front door's request/session correlation. No second worker
engine telemetry stream duplicates those lifecycle events. Raw prompts and
token text are not added to telemetry by IPC; the existing capture-text policy
continues to govern the parent bus.

Remote mock is positively classified as synthetic at backend registration,
model load/unload and session/request emission, including mixed-engine hosts.
Worker PID, restart count and circuit state are local diagnostic snapshots.
Optional cache/speculation/effective-execution reports and stage timings are
not transported in v1; those need explicit bounded fields with the native slice.
Future native device summaries must use a bounded drop-and-count channel;
decode must never block on an Observatory subscriber or telemetry writer.

## Verification and follow-up boundary

The module's doctest suite exercises real private IPC and child processes,
HTTP generate/chat streaming, mid-stream client cancellation, worker death and
recovery, rolling-hour restart policy, exact child environment/working
directory, and malformed/oversized messages. Policy tests use a supplied clock
value rather than waiting an hour. No GPU, model weights or external inference
service is required. CMake compiles platform implementations on Windows/MSVC
and Linux; execution evidence must be reported separately for each platform.

Next slices:

1. Wire the native in-process llama.cpp engine factory, preserve its sampling
   and model lifecycle contracts inside the worker, and test real driver/GGML
   failures. Logits must still never reach `serve`.
2. Add explicit per-GPU placement, worker identities/build selection,
   residency, concurrent slots, health visibility and GPU summaries. Review
   any additional child environment entry individually.
3. Consider authenticated encrypted cross-host RPC separately. This private
   IPC protocol is not network authentication or remote routing. Existing
   [ADR-021](../DESIGN_DECISIONS.md) keeps whole-request federation in Sonder
   Runtime; cross-host SI workers require a new decision, not an implicit
   extension of local endpoint names.
