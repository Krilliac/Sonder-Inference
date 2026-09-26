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

Emitted today:

| Event | Level | Notes |
| --- | --- | --- |
| `model.load.started` / `model.load.completed` | metrics | duration, format, quantization |
| `model.unload` | metrics | outstanding references |
| `request.queued` / `request.started` | metrics | kind, priority, sampling |
| `request.completed` / `request.cancelled` / `request.failed` | metrics | tokens, TTFT, total latency, stop reason, cancel latency, error code |
| `inference.decode.started` | metrics | at first streamed chunk, with TTFT |
| `inference.decode.completed` | metrics | wall decode time, backend eval time and tokens/s when reported |
| `inference.token.generated` | standard | per streamed chunk: index, bytes, elapsed; text only with `capture_text` |
| `device.memory.sample` | metrics | CPU/RAM at engine start |

Additional event types (not in the list above, recorded here per ADR-012):
`engine.started`, `engine.stopped`, `backend.registered`, `session.created`,
`session.closed`, `model.load.failed`, and `telemetry.dropped` (final drop
count). Engine-scoped events use the engine id as `session_id`.

Not emitted yet: prefill chunk events, speculation, context/KV, scheduler,
device transfer, and deep backend events. For the Ollama adapter, streamed
chunks approximate tokens; authoritative counts come from the backend's final
`eval_count` (`token_counts_from_backend: true`).
