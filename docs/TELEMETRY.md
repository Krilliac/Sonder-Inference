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
| `event_id` | string | `<producer.instance_id>-<sequence>` on the main stream, `<producer.instance_id>-s<sequence>` on the samples stream (contract, see below) |
| `sequence` | int | contiguous per (producer instance, sequence stream), starting at 0 in each stream. Dropped events take no number |
| `event_type` | string | one of the names below |
| `wall_time` | string | RFC 3339 UTC, millisecond precision |
| `mono_ns` | int | steady clock in ns; see the precision note below |
| `session_id` | string | engine id (`engine-…`) for engine-scoped events, session id (`sess-…`) for session and request events, bus instance id (`tel-…`) for `telemetry.dropped` |
| `run_id` | string or null | the host's `SessionOptions::run_id`; **defaults to the engine id** on engine, session, request, scheduler and KV events, so one engine's events group as one run. Null only on `telemetry.dropped` |
| `request_id`, `agent_id`, `task_id`, `model_instance_id`, `device_id` | string or null | correlation ids; null when unknown |
| `producer` | object | `name` (`sonder-inference`), `version`, `node_id` (host name), `instance_id` (`tel-…`, one per telemetry bus), `stream` (additive, optional: `"samples"` on periodic sample events, absent on every other event; see the stream identity contract), `role` (`inference`; additive, from `TelemetryOptions::role`), `synthetic` bool (additive, optional; `TelemetryOptions::synthetic`: true when the events describe synthetic work, i.e. `sonder-infer serve` with the MOCK backend, false when the host knows they do not; absent means unknown, so hosts that do not decide it, such as the CLI, the C ABI and bench, never claim that mock output is real) |
| `sampling` | object | `level`: the level **this event was emitted at** (`metrics` / `standard` / `deep`), so a consumer knows what a lower setting would drop; `sampled`: always true |
| `attributes` | object | per event, listed below |

**Stream identity contract.** One `TelemetryBus` (one engine) has two
sequence streams, each numbered from 0 by its own counter:

- the **main stream** (the default; events carry no `producer.stream`): every
  event except periodic samples, engine, session and request alike.
  `event_id` is `<instance_id>-<sequence>`.
- the **samples stream** (`producer.stream` is `"samples"`): events whose type
  ends in `.sample` (`device.memory.sample`, `backend.metrics.sample`,
  `backend.gpu_memory.sample`). `event_id` is `<instance_id>-s<sequence>`.

`producer.instance_id` names the producer instance; `sequence` is contiguous
per (instance, stream), so it is unique per (instance, stream) but not per
instance alone. Stream membership is decided by the producer and read from
`producer.stream` only; consumers never infer it from `event_type`. Consumers
key gap detection on `producer.instance_id` plus `producer.stream`, not on
`session_id`. Other producers (Sonder Runtime, the fake producer) have one
stream and send no `producer.stream`.

Why two streams: lifecycle events and periodic samples need different
retention. A retained set that keeps old lifecycle events but drops old
samples would be non-contiguous in one shared number space, which consumers
read as loss. With one stream each, every retained window is a contiguous
suffix of its own stream and no notices are needed.

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

## Synthetic provenance

Known mock backend/model/session work carries `producer.synthetic = true`
through the C++ engine, C ABI and Python SDK. Positive event context overrides
a host-wide false value. Unclassified contexts inherit the optional host
setting, keeping unknown absent; backend identity, rather than model names,
decides known mock work. This preserves engine/device observations and mixed
backend engines. See [SDK mock provenance](integration/sdk-mock-provenance.md).

## Sink failure isolation

The [routine stress controls](integration/telemetry-stress-ci.md) exercise
healthy delivery, callback failures and default-mask stream failures in the
normal platform and sanitizer CTest suites. They use synthetic transport data
and make no provider or model calls.

The writer catches exceptions from each sink's `write_event()` and `flush()`.
It retires the failed sink and all duplicate registrations of that object,
without retrying it or logging its exception message, event contents, or
destination. Healthy siblings continue receiving the original envelopes in
sequence order with the same producer instance and event IDs. No failure event
is recursively queued, and the envelope schema and C ABI are unchanged.

The C++ getter `TelemetryBus::failed_sinks()` counts distinct objects retired
after an exception, once per object registered before emission. This counter is
separate from `dropped_events()`, which retains its queue-pressure meaning.
The existing queue capacity and drop-report rules remain in effect. Sink I/O
and retirement stay on the writer; the final failed-sink reference is released
outside the bus mutex so a destructor cannot hold up the emitting thread while
holding that mutex.

`flush()` waits for processing of already-admitted events, including a backlog
left after the last sink fails. Further emission is disabled when no sink
survives; `shutdown()` still drains and terminates the writer. Processing does
not guarantee delivery to a sink that failed. Stream/file factories retain
their existing open-error behavior.

`shutdown()` waits at most `TelemetryOptions::shutdown_timeout` (default 5 s; zero or negative
means do not wait) for the writer. A sink stuck inside `write()` or `flush()` (a stalled
disk or pipe) therefore cannot hang the process: on expiry `shutdown()` returns `false`,
discards the events still queued, and detaches the writer. The detached writer owns the
queue state and the sinks it was handed, starts no further sink call, and exits when the
stuck call returns; it never touches the destroyed bus. A repeated call returns the first
result.

Built-in file and ostream sinks check stream state after writing the line and
newline, and after flushing. A failed operation retires the sink even with the
default exception mask; caller-owned ostream masks are never changed. This
also contains exceptions from streams with an explicit exception mask.
Custom sinks must signal failure by throwing; the bus cannot detect a custom
callback that silently discards output. Indefinitely blocking callbacks or destructors
remain the sink implementation's responsibility; the bus cannot interrupt
arbitrary user code. Tests exercise real writer-thread failures, healthy
sibling cursor parity, alias retirement, queue pressure, backlog drain,
destructor queries, and continued mock generation with raw-text capture off.
Mock and synthetic I/O controls provide no provider/model quality measurement.
The [exception qualification note](integration/telemetry-sink-isolation.md) and
[built-in stream follow-up](integration/telemetry-stream-failbits.md) record
regressions, bounded concurrent controls, timing comparisons and remaining limits.

## Engine and models

| Event | Level | Attributes |
| --- | --- | --- |
| `engine.started` | metrics | `version` str, `commit` str, `platform` str, `device_count` int, `text_capture` str (`on` / `off`); optional `server` obj `{host` str, `port` int, `api_version` int`}` when the engine is hosted by `sonder-infer serve` (`EngineOptions::server`) |
| `engine.stopped` | metrics | none |
| `scheduler.configured` | metrics | emitted once at engine start when scheduling is active: `mode` str (`automatic`, `gate`, `account`), `kv_block_size_tokens` int, `kv_num_blocks` int, `prefix_caching` bool, `max_running_sequences` int, `max_step_sequences` int, `max_step_tokens` int, `prefill_chunk_tokens` int, `admission_watermark_blocks` int, `max_requeue_count` int, `step_stall_timeout_ms` int |
| `device.memory.sample` | metrics | at start (`sample_devices_on_start`) and every `EngineOptions::device_sample_interval` (default 10 s, 0 disables): `kind` str, `name` str, `logical_cores` int, `total_bytes` int, `available_bytes` int; optional `used_bytes` int. Host memory only; per-process VRAM of a spawned llama-server is `backend.gpu_memory.sample` |
| `backend.registered` | metrics | `backend` str, `description` str, `capabilities` arr of str |
| `backend.gpu_memory.sample` | metrics | from the engine's periodic sampler, once per new sample of a backend that reports `Backend::runtime_status()` (a spawned `llamaserver` child): `backend`, `probe`, `status` str; `dedicated_bytes`, `shared_bytes`, `peak_shared_bytes`, `shared_baseline_bytes` (effective, including any per-1k-ctx growth), `spill_threshold_bytes`, `samples` int; `spilled` bool; `fitted_ctx` int or null; `fit_outcome` str ([vram-spill](integration/vram-spill.md)) |
| `backend.metrics.sample` | metrics | each attempted native child poll: `backend`, `port` int, `status` str (`ok`, `error`, or `unavailable`), `sampled_at` monotonic milliseconds; on `ok`, `metrics`, `slots`, and `speculation` directly in attributes, matching the child health object below. Failure details are in the accompanying warning. A 404 is reported once per child |
| `backend.warning` | metrics | once per distinct runtime warning: `backend`, `code`, `severity`, `source`, `message` str; `details` obj of str; `count` int |
| `backend.metrics.dropped` | metrics | `backend` str, `dropped_events` int (observations dropped since the previous drain of the 64-event pending queue) |
| `backend.restart` | metrics | child restart: `backend`, `reason` (`backend_stalled` for the stall policy), `attempt` int |
| `backend.device_unhealthy` | metrics | a spawned child logged a fatal accelerator fault (`ggml_vulkan: device lost on`, VK_EXT_device_fault, fatal CUDA/HIP error) and the child then exited or failed `/health`: `backend`, `reason`, `device`, `description` str. Emitted once; the supervisor then fails terminally instead of restarting, and runtime health adds `diagnostics.device` (`status: "unhealthy"`, plus `backend`, `reason`, `device`, `description`, `details`) |
| `backend.warmup` | metrics | once per terminal warm-up slot per generation: `backend` str, `generation` int, `id_slot` int, `status` str (`complete`, `error`, `cancelled`), `prompt_tokens` and `cache_n` int or null, `milliseconds` num; optional sanitized `error` str |
| `model.load.started` | metrics | `backend` str, `model` str |
| `model.load.completed` | metrics | `backend`, `model`, `format`, `family`, `parameter_size`, `quantization` str; `size_bytes` int; `resident` bool (false when only metadata was fetched, e.g. Ollama `/api/show`; the weights load on first request); `duration_ms` num |
| `model.load.failed` | metrics | `backend`, `model` str; `duration_ms` num; `error_code`, `error` str |
| `model.unload` | metrics | `backend`, `model` str; `outstanding_references` int |
| `model.evicted` | metrics | `sonder-infer serve` model residency only (`--model-idle-ttl`, `--max-resident-models`; never with the defaults): `backend`, `model` str; `reason` str (`idle_ttl` or `max_resident`); `idle_ms` num (time since the model's last request finished). Always followed by `model.unload` for the same `model_instance_id` |
| `telemetry.dropped` | metrics (bypasses the queue limit) | `dropped_events` int (cumulative), `emitted_events` int, `queue_capacity` int, `final` bool. Emitted as soon as the writer catches up after the first drop, then at most once per `TelemetryOptions::drop_report_interval` (default 1 s) while drops continue, and once at shutdown (`final: true`) if drops are still unreported |

### Native child observation

For a native `llamaserver` child, health may include the additive
`backends[].runtime.child` object. The monitor polls `/metrics` and `/slots`
at `max(5000, spill_guard.sample_interval_ms)` milliseconds, with a bounded
250 ms HTTP budget and a 1 MiB response limit per endpoint. `sampled_at` is a
monotonic millisecond timestamp; it is not wall time and is never a per-request
delta. Token and draft totals below are cumulative process counters; rates,
queue depth, busy slots and KV utilization are upstream gauges. These are
upstream observations, not independently measured request speed:

```json
{
  "port": 36515,
  "metrics": {
    "prompt_tokens_seconds": 1050.5,
    "predicted_tokens_seconds": 86.8,
    "prompt_tokens_total": 2500,
    "tokens_predicted_total": 1500,
    "requests_processing": 1,
    "requests_deferred": 2,
    "n_busy_slots_per_decode": 1.5,
    "n_decode_total": 1000,
    "spec_decode_num_draft_tokens_total": 2000,
    "spec_decode_accepted_tokens_total": 1500,
    "spec_decode_drafts_total": 1000,
    "spec_decode_accepted_tokens_per_pos_total": [900, 600],
    "kv_cache_usage_ratio": 0.25
  },
  "slots": [{"id": 0, "is_processing": true, "n_ctx": 77824}],
  "speculation": {
    "mean_accepted_len": 1.5,
    "acceptance_by_position": [0.9, 0.6],
    "speedup_est": 1.1363636363636362
  },
  "sampled_at": 1234567890
}
```

This synthetic example uses `--spec-draft-n-max 2`. Missing counters are
`null`; positional arrays use the upstream `position` label as their index,
with `null` for missing positions and a maximum of 256 entries. Raw metric
names use the `llamacpp:` prefix upstream; JSON omits that prefix.
`mean_accepted_len` is accepted tokens divided by drafts, and each cumulative
position acceptance is that position's accepted tokens divided by drafts.
Both are `null` when the denominator is absent or zero.
The speculation speedup heuristic is
`(1 + mean_accepted_len) / (1 + 0.6 * n_max)`, where `0.6` is the documented
default draft-cost coefficient and `n_max` comes from explicit
`--spec-draft-n-max`; when that option is absent or not an integer,
`speedup_est` is `null`. This coefficient is a heuristic, not a measured
speedup or a calibrated hardware claim. A sample is emitted only after
the poll completes; pending samples are bounded at 64 and are drained by the
engine sampler. A drop emits additive `backend.metrics.dropped` telemetry.

An endpoint 404 disables polling for that child and produces one
`metrics_unavailable` warning. Other failures produce a warning and reset
stall evidence; they are never fatal. A `backend.metrics.sample` event still
records `status: "error"` or `status: "unavailable"` for each attempted poll.

Runtime diagnostics are additive health state: `diagnostics.offload` is
`blind`, `pending`, `ok`, or `partial`. Offload and CPU-buffer classification
requires child verbosity `-lv 4` or higher; a lower or missing verbosity emits
one `diagnostics_blind` warning and is not evidence that offload is healthy.
`backend.warning` retains `code` and adds the matching `kind` for
`diagnostics_blind`, `backend_stalled`, `metrics_unavailable`, and
`metrics_scrape_failed`. `-lv 4` without observed offload evidence stays
`pending`; `ok` requires an observed complete offload line, and `partial`
requires N < M. A configured log file is still needed to read those lines.

The optional `stall_guard` defaults to enabled with a 90-second threshold and
`warn` policy. A stall requires `requests_processing > 0` and no movement in
either token counter for the threshold. Both counters must be present; a
counter reset, scrape failure, or idle sample breaks the evidence window.
It adds `backend_stalled`, with numeric `processing`, nullable `deferred`,
and duration `since_ms`; health adds `stall` with
`detected_at` and `since_ms` while retaining `ready`. `restart` uses the
existing crash-restart path and consumes `max_restarts`; attach mode warns
only because the external process is not owned by the supervisor.
`detected_at` is monotonic milliseconds at first detection; progress clears
the active stall. Attach monitoring is passive and does not replace the
existing backend health probe. Event delivery follows the engine sampler
interval (default 10 s); health reads the latest sample immediately. Setting
`device_sample_interval` to zero disables this delivery, as with existing
GPU and warning events. The pending queue remains bounded and counts drops.

## Sessions and requests

| Event | Level | Attributes |
| --- | --- | --- |
| `session.created` | metrics | `model`, `backend` str; `priority` int; `workload` str (see below); `text_capture` str (`on` / `off`); `sampling` obj |
| `session.closed` | metrics | `requests` int |
| `request.queued` | metrics | `kind` str (`generate` from `Session::generate`, `chat` from `Session::chat`), `priority` int (legacy scheduler rank, preserved), `priority_class` str (`interactive`, `subagent`, `background`), `workload` str, `prompt_bytes` int (for chat: the generic formatted prompt); `chat` adds `messages` int |
| `request.started` | metrics | `kind` str, `sampling` obj, `scheduled` bool, `sampler` str (`sonder` or `backend`); when scheduled, `scheduler_mode` str (`gate`: per-chunk grants; `account`: admission and prompt accounting only); `chat` adds `chat_template` str (`native`: the backend's chat API or model template received the messages; `generic`: `format_chat_prompt()`) |
| `request.completed` / `request.cancelled` / `request.failed` | metrics | `outcome`, `stop_reason` str; `prompt_tokens`, `completion_tokens`, `chunks` int; `token_counts_from_backend` bool; `ttft_ms`, `total_ms` num; `scheduled` bool; `sampler` str. For hosted requests: additive `priority_class` str and `admission` obj with `priority` str and `queue_ms` num. When scheduled: the existing `queue_ms`, `preemptions`, `accounted_prompt_tokens`, and `reused_prompt_tokens` remain. Backend request attributes are optional and additive: `backend_cached_tokens`, `backend_draft_tokens`, `backend_draft_accepted_tokens`, `backend_draft_acceptance_ratio`, and `backend_predicted_tokens_per_second`. `cancelled` optionally adds `cancel_latency_ms` num. `failed` adds `error_code`, `error` str, and `scheduler_rejected` bool (true) when the scheduler refused the request before any backend work |

Hosted request admission fields are additive: the historical numeric
`request.queued.attributes.priority` remains unchanged. Class names live in
`priority_class` and `admission.priority` (`interactive`, `subagent`,
`background`). `admission.queue_ms` is the measured wait as of the event;
terminal events include the full queue wait, including logical KV admission,
without double-counting scheduler time. It remains available with scheduling
off or an oversized account-mode prompt that bypasses the logical pool.
A queue refusal before a session request begins emits `request.failed` with
`backend_started: false`, `error_code`, class, queue time and correlation.
Health exposes `queued_by_class` as instantaneous counts from these tickets.
There is no separate Prometheus metrics endpoint.

All five request lifecycle events carry the optional `parent_request_id` str
when the caller set `RequestOptions::parent_request_id` (for `sonder-infer
serve`: the `X-Sonder-Parent-Request-Id` header, i.e. the Sonder Runtime turn
id). The C ABI/Python SDK can also set this parent with the additive
[request metadata record](integration/cabi-request-parent.md), without
changing the session's run ID. The envelope `request_id` is always the
engine's own id.

`sampling` objects hold `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `typical_p`, `repeat_last_n`, `presence_penalty`,
`frequency_penalty`, `logit_bias_count` (int; the biases themselves are not
recorded), `num_ctx` (int, 0 = model default), `max_tokens` and `seed` (int
or null), plus `explicit_only` bool. When `explicit_only` is true (requests
from the OpenAI-compatible server), a sampler field the caller did not set is
null: a model-default backend such as Ollama applied the model's own value.
The fields after `repeat_penalty` were added with
`feat/sampling-config`; consumers of older streams must treat them as
optional.

`workload` values, highest scheduling class first: `interactive_user`,
`owner_orchestrator`, `critic_verification`, `implementation_worker`
(default), `research_worker`, `background_indexing`, `maintenance`.

## Execution

| Event | Level | Attributes |
| --- | --- | --- |
| `inference.decode.started` | metrics | `ttft_ms` num (at the first streamed chunk) |
| `inference.token.generated` | standard | one per delivered chunk: `index` int, `bytes` int, `elapsed_ms` num, `unit` str. `unit: "token"` means one sampled token and adds `count` int (= 1), `token_id` int, `probability` num, `probability_basis` str (`post_sampling` / `pre_sampling`): always for Sonder's sampler (`post_sampling`), and for a backend-sampled token whose probability the backend reported (native `llamacpp` and deprecated `llamaserver`, see [Token probabilities and top alternatives](#token-probabilities-and-top-alternatives)). `unit: "chunk"` (other backend-sampled streams) means the token count per chunk is unknown; use `request.completed.completion_tokens`. Optional `text` str (only with `capture_text`) |
| `inference.prefill.completed` | metrics | once per completed request, emitted at request end: `prompt_tokens` int, `token_counts_from_backend` bool, `ttft_ms` num; optional `backend_prompt_eval_ms` num (backend-reported), `reused_prompt_tokens` int (scheduled requests) |
| `inference.decode.completed` | metrics | once per completed request: `completion_tokens` int, `chunks` int, `decode_wall_ms` num; optional `backend_eval_ms`, `backend_tokens_per_sec`, `backend_prompt_eval_ms` num |

### Token probabilities and top alternatives

Observatory's "Token probabilities" and "Top alternatives" panels consume
`inference.token.generated` (`unit: "token"`, `probability`) and
`inference.sampling.candidates`. Candidate emission is active when `standard`
events are recorded (a sink is attached and the level is `standard` or `deep`)
and `TelemetryOptions::token_probability_top_k` (default 5, clamped to 10) is
not 0:

- **Sonder's sampler (`token_logits` backends).** `SamplerChain` always emits
  `inference.token.generated` with `unit: "token"`, `token_id`, `probability`,
  and `probability_basis: "post_sampling"`, and emits
  `inference.sampling.candidates` from its post-sampling candidate distribution
  (`SamplerChain::top_candidates`, bounded by `token_probability_top_k`,
  `token_candidates_stride`, and `token_candidates_max_events`). Setting
  `token_probability_top_k` to 0 disables `inference.sampling.candidates` while
  keeping per-token `probability` on `inference.token.generated`.

#### Backend-sampled token probabilities

- **Native `llamacpp` backend.** When `token_probability_top_k > 0` and the
  caller did not request logprobs, `LlamaCppBackend` reads the post-sampling
  distribution (`probability_basis: "post_sampling"`) directly from its
  in-process llama.cpp sampler chain (without populating caller
  `TokenChunk::logprob` / `top_logprobs` or `GenerationResult::logprobs`). When
  the caller requested `sampling.logprobs`, telemetry reuses the pre-sampling
  log-softmax (`probability_basis: "pre_sampling"`) without altering the
  caller's `top_logprobs`. Setting `token_probability_top_k = 0` or level
  `metrics`/`off` turns off the probe and preserves GPU backend sampling
  (`llama_set_sampler`).
- **Legacy `llamaserver` probe (deprecated).** Against llama.cpp
  (`native_completion`, the default) a request whose caller did not ask for
  logprobs gets `"n_probs": K` and `"post_sampling_probs": true` (native
  `/completion`, and the OpenAI chat endpoint, which passes both through). The
  probabilities never reach the caller: `TokenChunk::logprob` / `top_logprobs`
  and `GenerationResult::logprobs` stay empty, as without the probe. A caller
  that asked for logprobs gets its request unchanged, and telemetry reuses that
  data (`probability_basis: "pre_sampling"`, the softmax of the raw logits). No
  probe is added for chat with tools (llama-server refuses logprobs with tools
  and streaming) or for a generic OpenAI upstream (`native_completion = false`);
  those requests keep `unit: "chunk"` events unless the caller's own logprobs
  carry token ids.
- **Events.** Each delivered chunk carrying a sampled probability emits
  `inference.token.generated` with `unit: "token"`, `count` 1, `token_id`,
  `probability` and `probability_basis` (`post_sampling` after the sampler
  chain, or `pre_sampling` from raw logits). Frames without one stay
  `unit: "chunk"`. Hidden reasoning tokens produce no visible text chunk, so
  authoritative counts still come from `request.completed`.
- **Cost.** Post-sampling probabilities reuse the sampler's candidate list;
  pre-sampling ones (the caller's logprobs) cost a pass over the vocabulary per
  token. On native `llamacpp`, enabling the probe (`token_probability_top_k > 0`
  at `standard`/`deep`) keeps logits readable on CPU instead of binding GPU
  backend sampling (`llama_set_sampler`).

| Event | Level | Attributes |
| --- | --- | --- |
| `inference.sampling.candidates` | standard | per sampled token (Sonder's sampler, native `llamacpp`, or `llamaserver`) with a reported probability and at least one candidate, bounded per request (below): `index` int (the `inference.token.generated` `index` of the same token), `sampled_token_id` int, `probability_basis` str (`post_sampling` / `pre_sampling`), `candidates` arr of `{token_id` int, `probability` num, optional `text` str (only with `capture_text`)`}`, most probable first, at most `token_probability_top_k` entries. Alternatives without a token id are left out |

Volume bounds (`TelemetryOptions`): `token_candidates_stride` (default 1;
emit for every n-th probability-bearing token of the request, counting from
the first; 0 is read as 1) and `token_candidates_max_events` (default 256 per
request; 0 = no cap). The shape follows Observatory's proposal in its
`docs/TELEMETRY_PROTOCOL.md` ("3D Inference: reserved and proposed events"),
with two differences: it is emitted at `standard` rather than `deep` (the
bounds above keep the volume down, and the panels then work at the default
level), and it adds `probability_basis`.

The Ollama timing helper (`ollama::emit_timing_events`) never reuses these
names. It emits `backend.model.load.reported` (`backend`, `model`,
`load_duration_ns` int; Ollama reports a load time even for a warm model, so
this is not a residency transition), `backend.timing.prefill` (`backend`,
`model`, `prompt_eval_count` int, `prompt_eval_cached_count` int (prompt
tokens served from Ollama's prompt cache; 0 before Ollama 0.33.3),
`prompt_eval_duration_ns` int,
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
| `scheduler.enqueued` | metrics | `scheduler_request_id` int, `workload` str, `priority_rank` int, `prompt_tokens` int, `exact_prompt_tokens` bool, `max_new_tokens` int, `context_limit` int, `gated` bool |
| `scheduler.bypassed` | metrics | an ungated (`account`) request whose prompt exceeds the logical KV pool runs unscheduled: `reason` str (`exceeds_kv_pool`), `prompt_tokens` int, `kv_capacity_tokens` int |
| `scheduler.rejected` | metrics | `scheduler_request_id` int, `reason` str (`never_fits`, `invalid_request`, `duplicate_id`, `duplicate_sequence`), `prompt_tokens` int; optional `kv_capacity_tokens` int |
| `scheduler.admitted` | metrics | `scheduler_request_id` int, `step` int, `queue_ms` num, `resumed` bool (true after a preemption), `reserved_blocks` int |
| `scheduler.preempted` | metrics | `scheduler_request_id` int, `reason` str (`kv_pressure` / `priority_admission`), `mode` str (`recompute`), `kv_tokens` int, `beneficiary_scheduler_request_id` int, `preemptions` int, `queue_ms` num (first admission wait), `failed` bool (requeue limit exceeded) |
| `scheduler.prefill.chunk` | standard | `scheduler_request_id` int, `step` int, `tokens` int, `context_offset` int, `reused_tokens` int, `completes_prefill` bool, `recompute` bool |
| `scheduler.prefill.completed` | metrics | the scheduler finished planning the prompt (the grant for the first token): `scheduler_request_id` int, `context_tokens` int, `prompt_tokens` int, `reused_tokens` int, `recompute` bool |
| `scheduler.stalled` | metrics | a granted request did not produce its token within `step_stall_timeout_ms` (a backend still thinking or loading, or a session blocked writing to a slow client); it leaves the step barrier until it produces that token, so the other requests keep running: `scheduler_request_id` int, `step` int, `phase` str (`prefill` / `decode`), `timeout_ms` int |
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
`request.failed.scheduler_rejected`; `engine.started.server`;
`scheduler.configured.mode`, `request.started.scheduler_mode`,
`scheduler.enqueued.gated`, `scheduler.bypassed`, request `priority_class`,
request `admission.priority`, request `admission.queue_ms`, and the server
health `queued_by_class` gauge. Discovery
advertises this vocabulary as `vocabularies: {"sonder.inference.events": 1}`.

## Observatory change requests (Observatory `docs/telemetry-schema.md` @ f5e3ff5)

| # | Request | Status |
| --- | --- | --- |
| 1 | name the stream | done: `producer.instance_id`, and the `event_id` format is a contract |
| 2 | group a run | done: `run_id` defaults to the engine id |
| 3 | capture policy | done: `text_capture` on `engine.started` and `session.created` |
| 4 | live drop reports | done: `telemetry.dropped` while running, `final` flag |
| 5 | periodic memory | done for host memory (`device_sample_interval`); per-process VRAM for spawned llama-server children (`backend.gpu_memory.sample`, Windows) |
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
  line) every 15 s. NDJSON carries one envelope per line and nothing else.
- Resume: the `Last-Event-ID` header wins over `?last_event_id=`. The cursor is
  one event id, or several separated by `,` (at most one per stream, any
  order, optional spaces), for example `tel-ab12-41,tel-ab12-s97`. Each member
  is split at its last `-`: the tail is `<sequence>` (main stream) or
  `s<sequence>` (samples stream). The rules apply to each stream
  independently:
  - Same instance and still retained: replay from the next sequence of that
    stream, then live.
  - Same instance but older than the retained window of that stream: an SSE
    comment `: resume-gap <from>-<to>` (main) or `: resume-gap s<from>-s<to>`
    (samples) names the missing range, then the whole window of that stream
    is replayed. The comment is its own block directly before the first
    replayed event of that stream; there is at most one per stream per
    connection and a fresh connection never gets one. NDJSON has no comment
    channel and carries none.
  - No member for a stream, a member of another instance (**the producer
    restarted**; this settles the open question recorded here earlier), or a
    malformed member: replay the whole retained window of that stream, then
    live.
  - `?since=now`: nothing replayed from either stream, live only.
  - The replay interleaves the two streams in original emission order. A
    client that sends only the last id it saw (a plain EventSource) still
    works: the other stream replays its whole window and the client drops
    duplicates by `event_id`.
- Retention: two rings of `--telemetry-buffer` events each, one per stream.
  Each holds a contiguous suffix of its stream, so a replay is contiguous per
  stream, samples never evict main events and main events never evict
  samples. The samples stream is a sliding window with no latest-per-key
  guarantee: a sample type has to be re-emitted periodically to stay visible
  to a late subscriber. Discovery reports the main stream in
  `resume.retained_events`, `resume.oldest_sequence` and
  `resume.next_sequence`, and the samples stream in
  `resume.streams.samples` with the same three fields. Lines written to the
  hub by producers other than the telemetry bus name their stream in
  `producer.stream`; any value other than `"samples"` is the main stream.
- Backpressure: the server keeps the two rings above and a bounded queue per
  subscriber (at most 8 subscribers; the ninth gets 429). The default queue
  holds a full replay (`2 x --telemetry-buffer`), so a fresh subscriber never
  loses retained events. A subscriber that falls behind loses its
  oldest undelivered events. That loss is visible as a sequence gap, is
  announced to that subscriber only (SSE comment `: dropped <n>`), and is
  counted as `subscriber_dropped_events` in health and discovery. It is
  **not** reported as `telemetry.dropped`, which keeps meaning bus-queue drops
  shared by every consumer. Emission never waits for a subscriber.

## Not emitted yet

`inference.prefill.started`, `inference.speculation.*`, `context.*`,
`kv.moved`, `kv.quantized`, `device.compute.sample`, `device.transfer.*`,
VRAM samples, and `backend.layer.*` / `backend.operator.*`.
