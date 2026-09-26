# Telemetry events (schema `sonder.observatory.event/1`)

This is the authoritative list of the events Sonder Inference emits, with
their attributes. Sonder Observatory's `docs/telemetry-schema.md` is written
from this file. Keep the two in step.

## Stability rules

- Event names and attribute names in this file are **stable** for schema
  `sonder.observatory.event/1`. Renaming or removing one, or changing an
  attribute's type or meaning, needs a schema bump and a matching Observatory
  change.
- Adding a new event, or a new optional attribute on an existing event, is
  allowed within `/1`. Update this file in the same PR. Consumers must ignore
  attributes they don't recognise.
- An attribute listed as *optional* may be absent. Every other listed
  attribute is always present when the event is emitted.
- Tests pin the envelope and the Observatory requests
  (`tests/test_telemetry.cpp`) and the event order (`tests/test_session.cpp`,
  `tests/test_engine_runtime.cpp`).

## Envelope

Each line of the JSONL stream is one object:

| Field | Type | Notes |
| --- | --- | --- |
| `schema` | string | `sonder.observatory.event/1` |
| `event_id` | string | `<producer.instance_id>-<sequence>` (contract, see below) |
| `sequence` | int | contiguous per producer instance, starting at 0. Dropped events take no number |
| `event_type` | string | one of the names below |
| `wall_time` | string | RFC 3339 UTC, millisecond precision |
| `mono_ns` | int | steady clock in ns; see the precision note below |
| `session_id` | string | engine id (`engine-…`) for engine-scoped events, session id (`sess-…`) for session and request events, bus instance id (`tel-…`) for `telemetry.dropped` |
| `run_id` | string or null | the host's `SessionOptions::run_id`; **defaults to the engine id** on engine, session, request, scheduler and KV events, so one engine's events group as one run. Null only on `telemetry.dropped` |
| `request_id`, `agent_id`, `task_id`, `model_instance_id`, `device_id` | string or null | correlation ids; null when unknown |
| `producer` | object | `name` (`sonder-inference`), `version`, `node_id` (host name), `instance_id` (`tel-…`, one per telemetry bus), `role` (`inference`; additive, from `TelemetryOptions::role`), `synthetic` bool (additive, optional; `TelemetryOptions::synthetic`: true when the events describe synthetic work, i.e. `sonder-infer serve` with the MOCK backend, false when the host knows they do not; absent means unknown, so hosts that do not decide it, such as the CLI, the C ABI and bench, never claim that mock output is real) |
| `sampling` | object | `level`: the level **this event was emitted at** (`metrics` / `standard` / `deep`), so a consumer knows what a lower setting would drop; `sampled`: always true |
| `attributes` | object | per event, listed below |

**Stream identity contract.** One `TelemetryBus` (one engine) numbers all of
its events, engine, session and request alike, from a single counter.
`producer.instance_id` names that stream, and `event_id` is always
`<instance_id>-<sequence>`. Consumers key gap detection on
`producer.instance_id`, not on `session_id`.

**`mono_ns` clock.** `mono_ns` is `std::chrono::steady_clock` since its
platform epoch (boot on Linux and Windows). On Linux that is
`CLOCK_MONOTONIC`, the same clock as Python's `time.monotonic_ns()`, so
producers **on the same Linux host** (Sonder Runtime and Sonder Inference)
can be merged by `mono_ns` (ecosystem contract v1, section 6.1). Across hosts,
and on macOS or Windows where the libraries may use different clocks (after
sleep, or Python before 3.13 on Windows), `mono_ns` values are not
comparable; within one producer always order by `sequence`. Values pass 2^53
after about 104 days of uptime; a JavaScript consumer parsing it as a double
then loses sub-microsecond precision. Use `mono_ns` differences for
durations.

Levels: `metrics` events are always recorded when telemetry is enabled.
`standard` adds per-token and per-step detail. `off` records nothing.

Types used below: `int` (JSON integer), `num` (JSON number), `str`,
`bool`, `obj`, `arr`.

## Engine and models

| Event | Level | Attributes |
| --- | --- | --- |
| `engine.started` | metrics | `version` str, `commit` str, `platform` str, `device_count` int, `text_capture` str (`on` / `off`); optional `server` obj `{host` str, `port` int, `api_version` int`}` when the engine is hosted by `sonder-infer serve` (`EngineOptions::server`) |
| `engine.stopped` | metrics | none |
| `scheduler.configured` | metrics | emitted once at engine start when scheduling is active: `kv_block_size_tokens` int, `kv_num_blocks` int, `prefix_caching` bool, `max_running_sequences` int, `max_step_sequences` int, `max_step_tokens` int, `prefill_chunk_tokens` int, `admission_watermark_blocks` int, `max_requeue_count` int |
| `device.memory.sample` | metrics | at start (`sample_devices_on_start`) and every `EngineOptions::device_sample_interval` (default 10 s, 0 disables): `kind` str, `name` str, `logical_cores` int, `total_bytes` int, `available_bytes` int; optional `used_bytes` int. Host memory only; no backend reports VRAM yet |
| `backend.registered` | metrics | `backend` str, `description` str, `capabilities` arr of str |
| `model.load.started` | metrics | `backend` str, `model` str |
| `model.load.completed` | metrics | `backend`, `model`, `format`, `family`, `parameter_size`, `quantization` str; `size_bytes` int; `resident` bool (false when only metadata was fetched, e.g. Ollama `/api/show`; the weights load on first request); `duration_ms` num |
| `model.load.failed` | metrics | `backend`, `model` str; `duration_ms` num; `error_code`, `error` str |
| `model.unload` | metrics | `backend`, `model` str; `outstanding_references` int |
| `telemetry.dropped` | metrics (bypasses the queue limit) | `dropped_events` int (cumulative), `emitted_events` int, `queue_capacity` int, `final` bool. Emitted as soon as the writer catches up after the first drop, then at most once per `TelemetryOptions::drop_report_interval` (default 1 s) while drops continue, and once at shutdown (`final: true`) if drops are still unreported |

## Sessions and requests

| Event | Level | Attributes |
| --- | --- | --- |
| `session.created` | metrics | `model`, `backend` str; `priority` int; `workload` str (see below); `text_capture` str (`on` / `off`); `sampling` obj |
| `session.closed` | metrics | `requests` int |
| `request.queued` | metrics | `kind` str (`generate` from `Session::generate`, `chat` from `Session::chat`), `priority` int, `workload` str, `prompt_bytes` int (for chat: the generic formatted prompt); `chat` adds `messages` int |
| `request.started` | metrics | `kind` str, `sampling` obj, `scheduled` bool, `sampler` str (`sonder` or `backend`); `chat` adds `chat_template` str (`native`: the backend's chat API or model template received the messages; `generic`: `format_chat_prompt()`) |
| `request.completed` / `request.cancelled` / `request.failed` | metrics | `outcome`, `stop_reason` str; `prompt_tokens`, `completion_tokens`, `chunks` int; `token_counts_from_backend` bool; `ttft_ms`, `total_ms` num; `scheduled` bool; `sampler` str. When `scheduled`: `queue_ms` num, `preemptions` int, `accounted_prompt_tokens` int, `reused_prompt_tokens` int. `cancelled` optionally adds `cancel_latency_ms` num. `failed` adds `error_code`, `error` str, and `scheduler_rejected` bool (true) when the scheduler refused the request before any backend work |

All five request lifecycle events carry the optional `parent_request_id` str
when the caller set `RequestOptions::parent_request_id` (for `sonder-infer
serve`: the `X-Sonder-Parent-Request-Id` header, i.e. the Sonder Runtime turn
id). The envelope `request_id` is always the engine's own id.

`sampling` objects hold `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `typical_p`, `repeat_last_n`, `presence_penalty`,
`frequency_penalty`, `logit_bias_count` (int; the biases themselves are not
recorded), `num_ctx` (int, 0 = model default), `max_tokens` and `seed` (int
or null). The fields after `repeat_penalty` were added with
`feat/sampling-config`; consumers of older streams must treat them as
optional.

`workload` values, highest scheduling class first: `interactive_user`,
`owner_orchestrator`, `critic_verification`, `implementation_worker`
(default), `research_worker`, `background_indexing`, `maintenance`.

## Execution

| Event | Level | Attributes |
| --- | --- | --- |
| `inference.decode.started` | metrics | `ttft_ms` num (at the first streamed chunk) |
| `inference.token.generated` | standard | one per delivered chunk: `index` int, `bytes` int, `elapsed_ms` num, `unit` str. `unit: "token"` (Sonder sampler) means exactly one token and adds `count` int (= 1), `token_id` int, `probability` num. `unit: "chunk"` (backend-sampled stream) means the token count per chunk is unknown; use `request.completed.completion_tokens`. Optional `text` str (only with `capture_text`) |
| `inference.prefill.completed` | metrics | once per completed request, emitted at request end: `prompt_tokens` int, `token_counts_from_backend` bool, `ttft_ms` num; optional `backend_prompt_eval_ms` num (backend-reported), `reused_prompt_tokens` int (scheduled requests) |
| `inference.decode.completed` | metrics | once per completed request: `completion_tokens` int, `chunks` int, `decode_wall_ms` num; optional `backend_eval_ms`, `backend_tokens_per_sec`, `backend_prompt_eval_ms` num |

The Ollama timing helper (`ollama::emit_timing_events`) never reuses these
names. It emits `backend.model.load.reported` (`backend`, `model`,
`load_duration_ns` int; Ollama reports a load time even for a warm model, so
this is not a residency transition), `backend.timing.prefill` (`backend`,
`model`, `prompt_eval_count` int, `prompt_eval_duration_ns` int,
`prompt_tokens_per_sec` num) and `backend.timing.decode` (`backend`,
`model`, `eval_count` int, `eval_duration_ns` int, `decode_tokens_per_sec`
num, optional `ttft_ms` num). The engine does not call it yet.

## Sampling

| Event | Level | Attributes |
| --- | --- | --- |
| `sampling.configured` | metrics | per request, when Sonder's sampler chain samples (backend exposes `token_logits`): `sampler` str (`sonder`), `selector` str (`greedy` / `distribution`), `stages` arr of str (chain order), `seed` int or null, `vocab_size` int |
| `sampling.failed` | metrics | `status` str (`no_viable_candidates` / `empty_logits`), `error_code` str (`invalid_argument` / `backend_error`), `position` int (generated-token index) |

## Scheduler

Only emitted when scheduling is active (the cache and scheduler modules are
built and `EngineOptions.scheduling.enabled`). `scheduler_request_id` is the
scheduler's numeric id; the envelope's `request_id` carries the request
correlation id on every per-request event.

| Event | Level | Attributes |
| --- | --- | --- |
| `scheduler.enqueued` | metrics | `scheduler_request_id` int, `workload` str, `priority_rank` int, `prompt_tokens` int, `exact_prompt_tokens` bool, `max_new_tokens` int, `context_limit` int |
| `scheduler.rejected` | metrics | `scheduler_request_id` int, `reason` str (`never_fits`, `invalid_request`, `duplicate_id`, `duplicate_sequence`), `prompt_tokens` int; optional `kv_capacity_tokens` int |
| `scheduler.admitted` | metrics | `scheduler_request_id` int, `step` int, `queue_ms` num, `resumed` bool (true after a preemption), `reserved_blocks` int |
| `scheduler.preempted` | metrics | `scheduler_request_id` int, `reason` str (`kv_pressure` / `priority_admission`), `mode` str (`recompute`), `kv_tokens` int, `beneficiary_scheduler_request_id` int, `preemptions` int, `queue_ms` num (first admission wait), `failed` bool (requeue limit exceeded) |
| `scheduler.prefill.chunk` | standard | `scheduler_request_id` int, `step` int, `tokens` int, `context_offset` int, `reused_tokens` int, `completes_prefill` bool, `recompute` bool |
| `scheduler.prefill.completed` | metrics | the scheduler finished planning the prompt (the grant for the first token): `scheduler_request_id` int, `context_tokens` int, `prompt_tokens` int, `reused_tokens` int, `recompute` bool |
| `scheduler.batch.formed` | standard | `step` int, `batch_size` int, `sequences` int (same value, kept for compatibility), `prefill_tokens` int, `decode_tokens` int, `admitted` int, `preempted` int, `running` int |
| `scheduler.batch.completed` | standard | `step` int, `duration_ms` num, `finished` int |

## KV cache (logical)

The engine's KV accounting is logical: blocks of `kv_block_size_tokens`
tokens tracked by `src/cache`. Backends still own physical KV memory.
Per-request KV events carry the request's `request_id` in the envelope.

| Event | Level | Attributes |
| --- | --- | --- |
| `kv.allocated` | standard | cache miss: `scheduler_request_id` int, `blocks` int (newly allocated), `tokens` int (appended) |
| `kv.reused` | metrics | prefix-cache hit: `scheduler_request_id` int, `tokens` int, `avoided_prefill_tokens` int (same value), `blocks` int, `recompute` bool |
| `kv.evicted` | metrics | aggregated per append: `blocks` int, `trigger_scheduler_request_id` int, `total_evictions` int |
| `kv.pressure` | metrics | engine-scoped, on watermark level change: `level` str (`normal` / `high` / `critical`), `occupancy` num (0..1), `utilization` num (same value), `pinned_blocks` int, `total_blocks` int |
| `kv.freed` | metrics | `scheduler_request_id` int, `blocks` int, `reason` str (`completed`, `cancelled`, `failed`, `preempted`) |

## Typical order for one scheduled request

`request.queued` → `scheduler.enqueued` → `request.started` →
`scheduler.admitted` → (`kv.reused`) → `scheduler.prefill.chunk`… →
`scheduler.prefill.completed` → `inference.decode.started` →
`inference.token.generated`… → (`scheduler.preempted` → `kv.freed` →
`scheduler.admitted` …) → `inference.prefill.completed` →
`inference.decode.completed` → `request.completed` → `kv.freed`.
Engine-scoped `scheduler.batch.*`, `kv.evicted`, `kv.pressure`,
`device.memory.sample` and `telemetry.dropped` are interleaved. Events from
different threads are ordered by `sequence`, not by `mono_ns`.

## Envelope additions for the ecosystem contract (v1)

Additive within `/1`: `producer.role` and `producer.synthetic`;
`request.queued.kind = "chat"` and `request.started.chat_template`;
`parent_request_id` on the request lifecycle events;
`request.failed.scheduler_rejected`; `engine.started.server`. Discovery
advertises this vocabulary as `vocabularies: {"sonder.inference.events": 1}`.

## Observatory change requests (Observatory `docs/telemetry-schema.md` @ f5e3ff5)

| # | Request | Status |
| --- | --- | --- |
| 1 | name the stream | done: `producer.instance_id`, and the `event_id` format is a contract |
| 2 | group a run | done: `run_id` defaults to the engine id |
| 3 | capture policy | done: `text_capture` on `engine.started` and `session.created` |
| 4 | live drop reports | done: `telemetry.dropped` while running, `final` flag |
| 5 | periodic memory | done for host memory (`device_sample_interval`); VRAM waits for a backend that reports it |
| 6 | tokens vs chunks | done: `unit` (`token` / `chunk`) and `count` on Sonder-sampled tokens; name unchanged |
| 7 | Ollama helper duplicates | done: helper renamed to `backend.*`; session emits `inference.prefill.completed` |
| 8 | metadata-only loads | done: `resident` on `model.load.completed` |
| 9 | per-event level | done |
| 10 | `mono_ns` range | documented (above); base unchanged |
| 11 | KV and scheduler names | done: names above are settled; `occupancy`, `avoided_prefill_tokens`, `batch_size`, `queue_ms` added |

## Live transport (`sonder-infer serve`)

`sonder-infer serve` (module `src/server`, ADR-020) serves this stream live
over HTTP. [SERVER.md](SERVER.md) is the reference; the transport contract
itself is Observatory's (`protocol/producer-discovery.schema.json`,
`docs/TELEMETRY_PROTOCOL.md`) and ecosystem contract v1 sections 5.1 to 5.4.

- Discovery: `GET /.well-known/sonder-telemetry` (`sonder.telemetry.producer/1`)
  names the streams, the resume window and the auth mode.
- `GET /v1/telemetry/sse` (Server-Sent Events) and `GET /v1/telemetry/ndjson`
  (one envelope per line) carry the envelopes above unchanged.
  `GET /v1/telemetry` picks the format from `?format=sse|ndjson`, then
  `Accept`. There is no WebSocket endpoint in v1.
- SSE framing: the first write is `retry: 2000`; each event is
  `id: <event_id>` then `data: <envelope>` then a blank line, with no
  `event:` field. Idle streams get a `: keepalive` comment (NDJSON: a blank
  line) every 15 s.
- Resume: the `Last-Event-ID` header wins over `?last_event_id=`. The id is
  split at its last `-` into instance and sequence.
  - Same instance and still retained: replay from the next sequence, then
    live.
  - Same instance but older than the retained window: an SSE comment
    `: resume-gap <from>-<to>` names the missing range, then the whole window
    is replayed.
  - Unknown or different instance (**the producer restarted**; this settles
    the open question recorded here earlier), a malformed id, or no id:
    replay the whole retained window, then live.
  - `?since=now`: live only.
- Backpressure: the server keeps a ring of the last `--telemetry-buffer`
  events (default 8192) and a bounded queue per subscriber (at most 8
  subscribers; the ninth gets 429). A subscriber that falls behind loses its
  oldest undelivered events. That loss is visible as a sequence gap, is
  announced to that subscriber only (SSE comment `: dropped <n>`), and is
  counted as `subscriber_dropped_events` in health and discovery. It is
  **not** reported as `telemetry.dropped`, which keeps meaning bus-queue drops
  shared by every consumer. Emission never waits for a subscriber.

## Not emitted yet

`inference.prefill.started`, `inference.speculation.*`, `context.*`,
`kv.moved`, `kv.quantized`, `device.compute.sample`, `device.transfer.*`,
VRAM samples, and `backend.layer.*` / `backend.operator.*`.
