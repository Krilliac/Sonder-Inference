# SDK per-request parent lineage

The native `RequestOptions::parent_request_id` and HTTP correlation header
already bind a caller turn to an engine request. The C ABI and Python SDK now
expose that same parent through a separate ABI-v1 `sonder_request_metadata`
record and optional `sonder_session_generate_with_metadata` /
`sonder_session_chat_with_metadata` exports. Existing signatures, layouts,
ABI major, session run IDs and engine-generated request IDs are unchanged.
There is no per-request run override, caller-selected engine request ID,
scheduling hint, deadline or telemetry callback in this record.

Each record carries `struct_size` and a nullable `parent_request_id`.
Complete prefixes are required; larger records ignore unknown tails. NULL
metadata or parent retains no parent. Non-null IDs must match
`[A-Za-z0-9._:-]{1,128}`; empty, control/non-ASCII and overlong values fail
before submission or callbacks, with errors that never echo the input. C
validation inspects at most 129 bytes including the terminator. Callers still
provide valid storage unchanged until return; this bound does not validate
pointer safety. The wrapper copies the parent into owned native request
options before submission, including asynchronous telemetry delivery.

Python exports frozen `RequestMetadata` and accepts keyword-only `metadata`
on `Session.generate`, `chat`, `stream` and `chat_stream`. Existing positional
callbacks and the public `TokenStream(session, prompt)` remain supported.
Default calls use existing exports. Explicit metadata, including an empty
record, requires the corresponding new export and raises `UnsupportedError`
against older ABI-v1 libraries. Both iterator methods validate metadata,
inputs, export support and closed state on the calling thread before starting
a worker. Prepared native parent and conversation bytes are owned snapshots.

The parent appears only as `attributes.parent_request_id` on the five request
lifecycle events: queued, started, completed, cancelled and failed. It does
not replace envelope `request_id` or become a token/scheduler/KV attribute.
Unset requests on reused sessions inherit no earlier parent. Session
session/run/agent/task IDs continue to come from `SessionMetadata`; native
`run_id` remains immutable session configuration. The caller owns the parent
reference's meaning; no identity registry or authorization is introduced.

Identifiers are explicit telemetry metadata, visible independently of text
capture. Do not put prompts, secrets or personal data in IDs. Validation is a
structural bound, not a semantic privacy filter. Metadata never enables text
capture. Producer identity/global sequence cursors, known mock provenance,
the 64-chunk queue, 25 ms cancellation polling, independent completion,
drain-before-error and native input lifetime remain unchanged. Result text
and exception tracebacks can retain data; this is not a total byte-memory cap.
Batching, admission, Runtime consent/effects and rollback retain their
existing contracts, including the Runtime's HTTP boundary.

## Qualification

Native tests exercise C11 calls, prefix versioning, exact ID boundaries,
sanitized invalid inputs without submission/callbacks, asynchronous telemetry
ownership and generate/chat/cancel/default reuse. Python controls cover all
four methods, real native error/cancellation lifecycle, eager preparation,
snapshot lifetime, optional symbols and the full-queue/close/cancel/error
matrix with and without metadata.

```sh
python scripts/qualify_request_metadata_legacy.py --library /tmp/old/libsonder_inference.so --out /tmp/legacy.json
python scripts/stress_request_metadata.py --library build/shared/libsonder_inference.so --cycles 8 --requests-per-worker 16 --out /tmp/parents.json
python scripts/make_request_metadata_fixture.py --library build/shared/libsonder_inference.so --out-dir /tmp/parent-fixtures
```

The real older-library control supplements the collected missing-symbol
proxy. An original-header compiled C caller must still run against the new
library. The bounded stress driver uses 1/2/4 workers and independent sessions
with mixed generate/chat and iterator requests, alternating/unset parents,
per-session correlation, unique engine IDs, contiguous producer cursors and
capture-off private prompt exclusion. Timings describe synthetic SDK/engine
overhead on that host, not provider/model speed or quality. Generated logs,
recordings and receipts stay outside Git.

The fixture driver generates actual native JSONL with explicit capture off/on
controls, exact parent/session expectations and known mock labels for
Observatory import/export/reopen qualification. The native producer manifest
does not claim consumer success; that requires a separate consumer receipt.
Full tests and required platform checks must qualify the exact reviewed
revision before merge. Scheduling/request-ID metadata and a telemetry
callback remain incomplete roadmap scope.
