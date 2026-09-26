# Observatory Contract

Sonder Inference produces telemetry. Sonder Observatory consumes it.

Canonical Observatory schema and UX contract live in:
- https://github.com/Krilliac/Sonder-Observatory

## Required inference event families

### Lifecycle
- model.load.started/completed
- model.unload
- request.queued/started/completed/failed/cancelled

### Execution
- inference.prefill.started/chunk/completed
- inference.decode.started/completed
- inference.token.generated
- inference.speculation.*

### Cache/context
- context.created/appended/forked
- kv.allocated/reused/evicted/moved/quantized/pressure

### Scheduler
- scheduler.enqueued/admitted/preempted
- scheduler.batch.formed/completed

### Device
- device.memory.sample
- device.compute.sample
- device.transfer.started/completed

### Deep backend mode
- backend.layer.*
- backend.operator.*

Only emit layer/operator events when the backend exposes a meaningful mapping. Do not fake architecture detail for visualization.

## Correlation

Inference must preserve:
- session_id
- run_id
- request_id
- model_instance_id
- device_id

Sonder Runtime may additionally attach:
- agent_id
- task_id
- route IDs

## Performance

Telemetry is never a synchronous requirement for token generation.

- bounded queue
- drop/sample under pressure
- emit dropped-event counters
- configurable levels
- raw text capture independent from structural events

## Replay

Events must contain enough stable state to reconstruct:
- scheduler transitions
- model/device residency changes
- cache allocation/reuse/eviction
- request/token timeline

Replay is observational; it is not required to reproduce the exact model generation unless deterministic inputs/model/backend are also retained.

## Implementation status (v0.1)

As of 2026-09-26 the library emits JSONL envelopes
(`schema: sonder.observatory.event/1`) with `event_id`, contiguous `sequence`,
RFC 3339 UTC `wall_time`, steady-clock `mono_ns`, `producer`
(`sonder-inference`, version, node id), `sampling.level`, and the correlation
ids `session_id`, `run_id`, `request_id`, `agent_id`, `task_id`,
`model_instance_id`, `device_id` (null when unknown).

The full event list, with every attribute and the stability rules, is in
[TELEMETRY.md](TELEMETRY.md). Sonder Observatory's `docs/telemetry-schema.md`
is written from it.

Emitted today:

- Lifecycle: `model.load.started/completed/failed`, `model.unload`, and
  `request.queued/started/completed/failed/cancelled`.
- Execution: `inference.prefill.completed`,
  `inference.decode.started/completed`, and `inference.token.generated`.
- Cache: `kv.allocated/reused/evicted/pressure` and `kv.freed` (logical KV;
  engine wiring, ADR-016).
- Scheduler: `scheduler.enqueued/admitted/preempted/rejected`,
  `scheduler.prefill.chunk/completed`, `scheduler.batch.formed/completed`,
  and `scheduler.configured`.
- Sampling: `sampling.configured` and `sampling.failed`.
- Device: `device.memory.sample` (CPU/RAM at engine start and every
  `device_sample_interval`, default 10 s).
- Other (ADR-012): `engine.started/stopped`, `backend.registered`,
  `session.created/closed`, and `telemetry.dropped` (live and final).

Envelope additions (compatible with v1, `additionalProperties` allowed):
`producer.instance_id`, `sampling.level` = the event's own level, and
`run_id` defaulting to the engine id. Ecosystem contract v1 adds
`producer.role` (`inference`) and `producer.synthetic` (true only with the
MOCK backend), `request.queued.kind = "chat"`, and `parent_request_id` on the
request lifecycle events. See TELEMETRY.md for the Observatory change-request
status.

## Live transport and cross-producer correlation (ecosystem contract v1)

`sonder-infer serve` serves the stream live (ADR-020,
[SERVER.md](SERVER.md#live-telemetry)): discovery at
`/.well-known/sonder-telemetry` (`sonder.telemetry.producer/1`, owned by
Observatory's `protocol/`), SSE at `/v1/telemetry/sse` and NDJSON at
`/v1/telemetry/ndjson`, with `Last-Event-ID` resume and a bounded ring. No
WebSocket. Observatory connects to Inference directly; Sonder Runtime does not
relay Inference telemetry.

For a request sent by Sonder Runtime, Runtime's turn id R arrives as
`X-Sonder-Run-Id` and `X-Sonder-Parent-Request-Id`. Inference events for that
request then carry `run_id = R`, their own `request_id`, and
`attributes.parent_request_id = R` on the request lifecycle events, so
Observatory can group by `run_id` and link Runtime's `request_id` to
Inference's. `mono_ns` is comparable across producers on the same Linux host
only (see TELEMETRY.md).

Not emitted yet: `inference.prefill.started`, speculation, `context.*`,
`kv.moved/quantized`, `device.compute.sample`, device transfer, and deep
backend events. For the Ollama adapter, streamed chunks approximate tokens;
authoritative counts come from the backend's final `eval_count`
(`token_counts_from_backend: true`).
