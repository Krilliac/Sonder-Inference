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
