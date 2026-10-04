# Structured Python chat iteration

`Session.chat_stream(messages)` fills the documented SDK chat-iterator gap.
It uses `sonder_session_chat` and the existing `TokenStream` implementation;
it introduces no native ABI symbol, layout, backend or scheduling path.

Chat-message preparation is shared with synchronous `chat`: 1–1,024 records,
at most 1,025 consumed iterable elements, text role/content only, no NULs or
silently ignored mapping fields. Encoded native buffers are owned before the
worker starts, so subsequent mutation of caller mappings cannot change the
conversation. Older ABI-v1 libraries without chat raise `UnsupportedError`
before starting a worker. Native conversation validation and backend errors
propagate from iteration after already queued chunks drain.

The existing 64-pending-chunk bound, ordered UTF-8 delivery, cancellation epoch,
nonblocking completion signal, session/engine close and worker join semantics
apply to both iterator methods. Counts bound chunk/message objects, not text
bytes. Final result text is retained. A paused consumer holds session/admission
resources until resumed or closed; use the iterator's context manager when
abandoning delivery.

Native chat telemetry remains `kind=chat`, with the structural message count,
session correlation IDs, producer cursor and known-mock synthetic provenance.
Input content stays absent; captured output requires existing explicit consent.
Batching, effect recovery and rollback contracts are unchanged. Tests and
qualification use actual shared-library mock calls and loopback synthetic
transport; they do not measure provider/model quality. Qualification receipts
belong on the exact reviewed revision, not in committed recordings or outputs.
