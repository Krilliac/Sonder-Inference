# Engine wiring: scheduler + KV cache + sampler

Status: implemented on `feat/engine-wiring` (PR #9, 2026-09-26). This connects
the Phase 2 policy modules (`src/scheduler`, `src/cache`, `src/sampling`) to
`Engine` and `Session`. Everything below is compiled only when the modules are
built (`SONDER_HAS_KV_CACHE`, `SONDER_HAS_SCHEDULER`,
`SONDER_HAS_SAMPLER_CHAIN`). A core-only build keeps the old direct path.

## Request path

```
Session::generate
  ├─ open TokenStream   (backend has Capability::token_logits → Sonder samples)
  ├─ prompt tokens      (TokenStream → BackendModel::tokenize → accounting tokenizer)
  ├─ RequestRuntime::submit   → scheduler.enqueued
  ├─ backend decode loop
  │    └─ before every token: acquire_token()  (waits for the scheduler's grant)
  │       after every token:  token_produced(id)
  └─ RequestRuntime::finish   → frees the cache sequence (kv.freed)
```

`detail::RequestRuntime` (`src/engine/request_runtime.{hpp,cpp}`) is owned by
the `Engine`. It holds a `cache::KvCacheManager`, the
`KvCacheCapacityAdapter` and a `scheduler::Scheduler`, and runs one coordinator
thread that repeats:

1. `plan_step()`. The scheduler admits, preempts and picks prefill chunks and
   decode tokens within `max_step_tokens` / `max_step_sequences`.
2. Apply the plan to the cache. Preempted requests lose their sequence
   (recompute mode). Prefill chunks append prompt tokens, which is where prefix
   reuse happens (`kv.reused`). Decode work appends the previous token. Each
   request that completes prefill, or gets a decode slot, receives one token
   credit.
3. Wait until every granted token has been produced, or its request has
   finished or failed, for at most `SchedulingOptions::step_stall_timeout_ms`
   (default 250 ms). A request that misses the deadline (a backend still
   thinking before its first chunk, a cold model load, a session blocked
   writing to a slow client) is reported as stalled (`scheduler.stalled`): it
   gets no new grant and is left out of later barriers until it produces the
   token it owes, so it can never freeze the other requests. A step in which
   every planned request is stalled sleeps until one of them makes progress.
4. `complete_step()`. Requests that finished early (EOS, stop sequence,
   cancel) are reported as `finished_early`; stalled work is reported as
   `stalled` and is not accounted (planned again next step). Finished requests
   are then dropped from the scheduler (`Scheduler::forget`), and inter-token
   latency samples are a bounded ring (`SchedulerConfig::latency_sample_window`),
   so a long-running engine holds state only for live requests.

Capacity comes only from the adapter: the scheduler sees free blocks as
"cache free + evictable − outstanding reservations". The adapter's release
hook frees a request's cache sequence *synchronously* when the scheduler drops
its reservation. The scheduler re-reads free capacity after each victim, so
without the hook one pressure event would preempt every running request.

### Sessions and blocks

- A request's cache sequence lives from admission until it finishes. When a
  request completes, is cancelled, fails or is preempted, its blocks are
  released. Full blocks stay in the prefix index as evictable cache.
- Prefix sharing is fingerprinted per model (backend, name, format, family,
  quantization). Two concurrent requests with the same prompt prefix share
  blocks (`KvUsage::shared_blocks`). A later request reuses the blocks while
  they are still cached (`SchedulingInfo::reused_prompt_tokens`).
- A cancelled request frees its blocks immediately, whether it was running or
  still queued. A cancel while queued returns `outcome == cancelled` with no
  output.
- A prompt that can never fit (`prompt + 1 > kv_num_blocks × block`) is
  rejected with `invalid_argument` (`scheduler.rejected`, `never_fits`).
- A request preempted more than `max_requeue_count` times fails with
  `unavailable` (`scheduler.preempted` with `failed: true`).
- `max_new_tokens` is clamped to the context limit and to the pool. The
  context limit is the model's `context_length`, narrowed by
  `SamplingConfig::num_ctx` when that is set. When the prompt tokens are exact
  (sampler-chain path, or a backend that tokenizes), the session caps
  generation at that budget (`SamplingConfig::max_tokens` for the backend
  call and the sampler loop), so the reply stops with `max_tokens` instead of
  running on ungated. With an approximate prompt count the backend's own
  context handling applies.

### Priority

`SessionOptions::workload` (default `implementation_worker`) selects the
scheduler class. `SessionOptions::priority` shifts it: rank = class −
priority, so a larger priority wins. The cache eviction priority follows the
rank.

### Sampling

When the backend advertises `Capability::token_logits`,
`BackendModel::open_token_stream()` returns a `TokenStream`. The session then
builds a `sampling::SamplerChain` from the request's `SamplingConfig` via
`core_bridge::make_chain`, and samples every token with it (logit bias,
repeat/presence/frequency penalties over `repeat_last_n`, top-k, typical-p,
top-p, min-p, temperature, then greedy or seeded selection; every
`SamplingConfig` field is mapped since #9, see sampling-config.md). It also
applies `StopSequenceMatcher`. Output is deterministic for a given seed and
does not depend on scheduling or batching (`tests/test_engine_runtime.cpp`).
Backends without `token_logits` (Ollama, llama.cpp today) receive
`SamplingConfig` and sample themselves. `request.started.sampler` and
`SchedulingInfo::sonder_sampled` show which path ran.

The mock backend exposes `token_logits` when
`MockBackendOptions::token_logits` is set. It then has 24 words plus EOS
(id 24), deterministic pseudo-logits, and EOS at `default_completion_tokens`.
`MockBackendOptions::ban_all_tokens` makes every logit `-inf`, which is how
the tests exercise `NoViableCandidates`.

## Decisions

### NoViableCandidates → `ErrorCode::invalid_argument`

`SampleStatus::NoViableCandidates` means every logit was `-inf` after the
chain ran. Given valid backend logits, the only things that can do that are
the request's own policy (bias, penalties or constraints). Retrying the same
request cannot succeed, and the backend did nothing wrong. So it is a caller
error: `invalid_argument`, not `backend_error` or `unavailable`.
`SampleStatus::EmptyLogits`, where the backend produced nothing, maps to
`backend_error`. Both emit `sampling.failed` with a stable `status`
(`no_viable_candidates` / `empty_logits`) and the `error_code`. ADR-017.

### Telemetry

Scheduler, KV and sampling events, and the Observatory change requests
(stream identity, run grouping, capture policy, live drop reports, periodic
memory samples, token units, per-event level), are specified in
[../TELEMETRY.md](../TELEMETRY.md). The scheduler's prefill events are
`scheduler.prefill.chunk` / `scheduler.prefill.completed`; the session emits
one `inference.prefill.completed` per completed request.

### Ownership boundaries (parallel workers)

New `SamplingConfig` fields and their Ollama/llama.cpp mapping belong to
`feat/sampling-config`; chat on the Backend interface and the CLI belong to
`feat/chat-cli`. This PR does not touch either.

### Sampling header path stays `sonder/sampling/…`

The module convention is `<module>/include/sonder/inference/<module>/`. The
sampling module ships `src/sampling/include/sonder/sampling/` (namespace
`sonder::inference::sampling`). Moving it would touch 39 include lines in 25
files (module sources, tests, README, docs, session) and give no functional
benefit, so it is **kept**. New modules follow the convention. If the public
include set gets frozen for 1.0, the move becomes one mechanical PR with
forwarding headers. ADR-018.

## Limitations and open items

- KV accounting is **logical**. Backends still own physical KV (Ollama
  server-side; llama.cpp in its own context). Prefix reuse saves Sonder
  bookkeeping, not backend prefill, until a backend exposes KV control
  (`kv_export` / sequence APIs).
- For backends without `tokenize()` (the mock without `token_logits`, and
  Ollama) prompt tokens come from an approximate accounting tokenizer (BOS +
  one token per whitespace word). Text chunks count as one token each. Block
  counts are then estimates.
- The loop is lockstep: a slow chunk consumer delays the whole step for every
  sequence in it.
- The scheduler's per-request table (`seqs_`) is never pruned. Memory grows
  with the number of requests over the engine lifetime.
- There is no session-level KV retention or fork across turns. Blocks outlive
  a request only as evictable prefix cache.
- The llama.cpp backend doesn't expose `token_logits` yet, so it samples with
  llama.cpp's own chain.
- The C ABI lacks the scheduling options.
