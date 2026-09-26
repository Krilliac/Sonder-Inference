# Sessions

Responsibility: Generation sessions and request lifecycle.

Implemented: `Session` (`session.cpp`): idle/running/closed state machine,
one in-flight request per session, per-request cancellation, request telemetry
(`request.*`, `inference.decode.*`, `inference.token.generated`).

Deferred: fork/share, context lineage, KV references.

See [source workspace](../README.md).
