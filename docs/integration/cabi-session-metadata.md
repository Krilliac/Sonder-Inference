# SDK session correlation metadata

The HTTP server and C++ SessionOptions already carry correlation IDs. The C
ABI and Python could only create sessions with generated defaults. Add the
ABI-v1 export `sonder_session_create_with_metadata` and an independent
`struct_size`-versioned `sonder_session_metadata` prefix containing nullable
session/run/agent/task IDs. Existing structs/symbols/ABI-major are unchanged.
The original create export delegates with null metadata and retains sampling,
engine/model lifetime, cancellation and session state contracts.

Non-null IDs match the existing HTTP policy `[A-Za-z0-9._:-]{1,128}`. Empty,
non-ASCII, control/slash characters and overlong IDs fail before session
creation, with errors that never echo input. C validation reads at most 129
bytes including the terminator; C callers still provide valid NUL-terminated
storage. Larger records ignore unknown tails; truncated prefixes fail. All
strings are copied into immutable session options before returning. Callers
keep supplied session IDs unique; this feature introduces no identity registry.

Python offers frozen `SessionMetadata` and a keyword-only `metadata` argument.
Default calls use the original export. The loader declares the new export only
when available; explicit metadata against an older ABI-v1 library fails with
UnsupportedError. The archive compatibility check uses an actual pre-feature
shared library, in addition to the normally collected missing-symbol proxy.

Identifiers are explicit caller metadata, visible in producer envelopes even
when text capture is off. They must not contain prompts, secrets or personal
data. Metadata never derives from text and does not enable capture. Existing
producer cursor/instance identity, queue bounds, batching, scheduling priorities,
Runtime consent, effects and rollback remain unchanged. Workload/priority hints,
request metadata and a telemetry callback remain future SDK additions.

## Validation

Native tests exercise C11 calls, prefix versioning, owned ID copies, engine/model
retention, generate/chat correlation, privacy and bounds. Python tests qualify
generate/chat/stream envelopes, global sequence/event-ID uniqueness, private
text exclusion, defaults, invalid inputs, and older-library fallback.

```sh
python -m pytest bindings/python/tests -q
python scripts/stress_session_metadata.py --library build/shared/libsonder_inference.so --cycles 8 --requests-per-worker 16 --out /tmp/sdk-metadata.json
```

The bounded stress driver uses actual library sessions and the deterministic
mock backend with 1/2/4 workers. It requires every queued request, correct
per-session IDs, unique monotonic cursors and no text capture after each engine
drains. Synthetic session-creation and request timings measure SDK/engine/
telemetry overhead on this host; they establish no provider/model speed or
quality. Generated logs stay outside Git.

Root integration touches the stable C header/core wrapper, Python mirrors and
API, C/C++/Python tests and this stress script. No schema or dependency change
is needed. This document records those cross-module integration changes.

## Initial local qualification

Before commit, strict native CTest passes 926 cases, including the two metadata
controls and the updated C11 smoke call. The real shared-library Python suite
passes 106 tests. A bounded 1/2/4-worker run passes 896 requests across 56
sessions, checking queued request counts, per-session correlation, cursors and
private text exclusion. An archived pre-feature shared library still handles
default generate/chat/stream and rejects explicit metadata as Unsupported.
Exact committed-revision/platform qualification is recorded in the pull request.
These are synthetic SDK/runtime measurements, not provider/model quality.
