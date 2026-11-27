# Benchmark and Validation Plan

A custom engine is only useful if it is measurably better for Sonder's workloads or gives capabilities the baseline cannot provide.

## Baselines

At minimum compare against:
- current Ollama path
- same model through direct llama.cpp where possible
- Sonder Inference backend under test

For relevant hardware, optionally include:
- vLLM/SGLang
- ExLlamaV3
- TensorRT-LLM
- specialized engine under evaluation

## Workload families

### Interactive chat
- short/medium prompt
- streaming decode
- concurrency 1
- latency-sensitive

### Long-context
- large prompt/prefill
- moderate output
- context continuation

### Agent fan-out
- shared large prefix
- 4/8/16 child requests
- measures prefix/KV sharing benefit

### Coding
- long system/project context
- medium/long output
- tool interruptions and resume

### Background batch
- high concurrency
- throughput-oriented

### Compaction
- context grows to threshold
- compaction
- continue generation
- verify bounded memory and semantic continuity

### Multi-model
- interactive model + small worker/embed/rerank
- model residency churn

### Failure/recovery
- cancellation
- backend error
- OOM pressure simulation
- node disconnect
- cache eviction
- repeated failure/no-progress guard

## Metrics

Performance:
- TTFT
- time per output token
- tokens/sec
- prompt tokens/sec
- throughput under concurrency
- p50/p95/p99 latency
- GPU utilization
- CPU utilization
- VRAM/RAM
- transfer bytes/time
- model load time
- cache hit benefit

Correctness:
- deterministic greedy token match against reference where applicable
- tokenizer round-trip
- logit/probability tolerance
- grammar/structured-output validity
- KV reuse equivalence
- resume/fork equivalence
- quantization quality suite

Reliability:
- crash rate
- request failure rate
- memory leaks
- cache corruption
- cancellation latency
- starvation
- retry loops
- long soak tests

## Benchmark hygiene

Record:
- exact engine commit/version
- model revision
- quantization
- context/output lengths
- sampling
- batch/concurrency
- cache state
- warm/cold run
- hardware/drivers
- backend flags
- power/thermal state where relevant

Never compare headline tokens/sec from different prompt lengths, quantizations, or concurrency configurations as if they were equivalent.

## Promotion gates

A new backend/feature is not promoted because it is novel.

Promotion requires:
- correctness suite green
- representative benchmark win or unique required capability
- bounded failure behavior
- observability present
- fallback path
- documented hardware/model support

## Harness (v0.1)

`sonder-infer bench` runs a corpus (`sonder.inference.corpus/1`, e.g.
[`bench/corpus/smoke.json`](../bench/corpus/smoke.json)) against one backend and
model and writes a `sonder.inference.bench/1` results document:

```bash
sonder-infer bench --backend ollama --model <model> \
    --corpus bench/corpus/smoke.json --out results.json --warmup 1 --runs 3
```

Recorded per the hygiene list: engine version and commit, host platform and
CPU/RAM, backend name and version, model name/format/family/parameter
size/quantization, sampling (greedy, seed 42 by default), warmup/measured
runs, concurrency (1), and cache state (not controlled yet). Per-run rows carry
TTFT, total latency, prompt/completion tokens, decode and prompt tokens/s (with
the source of the number), and stop reason. The summary gives
n/mean/p50/p95/min/max.

`sonder-bench` (target built from `bench/tools/sonder_bench.cpp`) is the
standalone runner. It writes JSON plus a markdown summary to `bench/results/`
and supports `--budget-seconds` (default 540), `--prompts`, `--require-idle`
(refuses to run when a different Ollama model is resident), `--telemetry FILE`,
`--list-models` and `--dry-run`:

```bash
sonder-bench --backend ollama --model <model> --corpus bench/corpus/baseline.json --require-idle
```

[`bench/corpus/baseline.json`](../bench/corpus/baseline.json) covers a short
fact, coding, medium reasoning, a ~2.4k-token long-context prompt, and a 4-way
agent fan-out with a shared ~600-token prefix. Fan-out children run
concurrently, each in its own session; results add `per_prompt` distributions,
`fanout_batches` (makespan and aggregate tok/s), p99, host hardware, the corpus
hash, and cold first-request/load time.

Limitations: concurrency is 1 except for fan-out prompts; no context-window
control (`SamplingConfig` has no `num_ctx`, so the long-context case is sized
for Ollama's default 4096 window); no GPU/VRAM sampling; mock-backend numbers
are meaningless by design. Small reviewed snapshots go in
[`bench/results/`](../bench/results/). The Ollama baseline is still pending
(`bench/results/PENDING_ollama_baseline.md`): no live Ollama was reachable when
the harness was built.

## HTTP agent and contention harness

[`bench/http/bench_http.py`](../bench/http/bench_http.py) is the Python stdlib
HTTP harness; [its README](../bench/http/README.md) defines the scenarios and
result fields. Its fake-server tests exercise transport, cache accounting and
failure checks without model weights or a GPU. They establish harness
behavior only, not inference quality or performance.

The following measurements support the 2026-09-30 serving experiments:

| Experiment | Scenario and evidence | Acceptance |
| --- | --- | --- |
| E02 busy-slot regression | `concurrency-stall`: three clients with different cache keys; per-request wall/status plus content and reasoning canaries; child `selected slot by id` and repeated `progress = 1.00` counters | No request over 250 s, no 503, no cross-client marker leak. A timeout or missing response fails; diagnostic-source failures are explicit. |
| E07 proxy versus raw backend | `agent-stable`, `agent-volatile-top`, `fanout-sweep`, plus `agent-alternate`; per-turn `cache_n`, `prompt_n`, TTFT and cached-token provenance | Reuse through the proxy must agree with raw llama-server for the same model, pins, prompt and cache configuration. Missing cache data is not a zero-hit measurement. |
| E08 prompt-cache RAM sizing | `agent-alternate`: A ~22k, B ~5k, C ~30k tokens, strict A/B/C order; separate histories and keys; count eviction/full-reprocessing log messages | A's turn-2+ `cache_n` (or reported cached-token equivalent) is at least 0.9 times measured prompt length. Record system commit and free RAM separately; this harness does not change cache RAM settings. |
| E11 speculative draft length | `--metrics-url` samples accepted-token counters by position around each request and scenario | Compare the same sampling and prompt classes. Server-wide deltas from overlapping request windows must not be summed. Counter resets/missing positions remain explicit. |
| Priority policy comparison | `priority-contention`: cold ~20k-token background prefill and ~500-token interactive request | Record interactive TTFT and actual overlap. Compare policy configurations with the same workload; this is not a claim that a running prefill is preemptible. |

Repeatable `--header K=V` and per-scenario `headers`, `prompt_cache_key` and
`priority` let each arm use the real session/priority paths. Preserve the
existing baseline scenarios and their settings when comparing old results.
Pinned-thinking warnings raise recall/agent default output headroom to at
least 1,500 tokens, with explicit output overrides preserved; record any
retry and whether the recalled fact was found only in `reasoning_content`.

The GPU operator owns live runs and stores reviewed E02/E07/E08/E11 artifacts
under `<bench-output-dir>`. The lane author runs only fake-server tests;
real GPU acceptance, deployment, commits and PR operations belong to the
integrator. Each result should include exact server revision, model, launch
configuration, pins, sampling, context size and the arm's scenario JSON.

## Evidence rules

These rules apply to every number that goes into a report, a commit message or
a decision. They are Sonder's own wording of practices used by other projects'
benchmarking notes; no text or data was copied.

1. **Same-session ABBA for A/B speed claims.** Compare two configurations by
   running A, B, B, A (then repeat the block) in one session, same model load
   state, same machine, no other GPU work. Report each arm's runs, not only the
   mean, and the A-vs-A drift (first A against last A). If drift is as large as
   the A/B difference the claim is not supported. Numbers from different
   sessions, days or thermal states are never an A/B pair.
2. **A correctness line beside every speed row.** Each speed row carries
   either token-ID parity against the reference (only where the arithmetic is
   the same: same weights, quantization, kernels and sampling at temperature 0)
   or, where arithmetic differs, a score or perplexity on a named fixed set. A
   speed row without a correctness line is marked `UNCHECKED` and cannot back a
   promotion.
3. **Cache state is a column.** Label every row `cold` (model not resident,
   page cache dropped or first load), `warm` (resident, prompt cache empty) or
   `warm-cache` (prompt/KV cache populated). State how cold was produced.
4. **Exact token counts.** Record prompt and completion token counts as the
   server or tokenizer reports them, plus their source (`usage`, tokenizer, or
   streamed-chunk count as a fallback). Never derive tok/s from characters or
   words, and never compare rows with different counts as if equal.
5. **Retractions stay in place.** A wrong or superseded result is struck
   through or marked `RETRACTED <date>: <reason>` where it stood, with a link
   to the corrected row. Do not delete it and do not silently rewrite it.
6. **Ledger of rejected ideas.** Every idea tried and dropped gets a row:

   | Idea | Date | Evidence (run, commit, numbers) | Why rejected | Reopen when |
   | --- | --- | --- | --- | --- |

   `Reopen when` is a concrete, checkable condition (a driver version, a new
   kernel, a hardware change, a different model family), not "later". An idea
   whose condition is met is re-tested before it is built.
7. **A check that stops checking is not a pass.** A run with zero completed
   requests, zero host samples, or a fault schedule in which no fault fired
   fails. Say what the number would look like if the harness were broken.

## Reliability run recipe

For hour-long (or longer) runs of slow models, where the question is whether
the stack stays up, not how fast it is. The driver is
[`scripts/reliability_run.py`](../scripts/reliability_run.py) (stdlib only;
`--self-test` checks the driver against a built-in fake server and is part of
CTest as `sonder.reliability_run.self_test`).

**Setup.** One model, one launch configuration, fixed prompts, temperature 0,
`max_tokens` fixed, one client unless concurrency is the thing under test.
Record the exact server and model revision, launch flags, context size, host
and driver versions, and whether the run starts cold or warm.

**Sampling at 1 Hz** (`samples.jsonl`), for the server process and any child it
supervises:

| Signal | Windows | Linux |
| --- | --- | --- |
| Process private bytes (not comparable across OSes: Windows `PrivateUsage` is commit charge, Linux is resident private pages) | `GetProcessMemoryInfo` `PrivateUsage` | `smaps_rollup` Private_Clean+Private_Dirty (fallback `RssAnon`) |
| Process working set | `WorkingSetSize` | `VmRSS` |
| Host available memory | `GlobalMemoryStatusEx` `ullAvailPhys` | `MemAvailable` |
| Host commit (used / limit) | `ullTotalPageFile` minus `ullAvailPageFile` / `ullTotalPageFile` | `Committed_AS` / `CommitLimit` |
| Process read bytes/s | `GetProcessIoCounters` `ReadTransferCount` delta: all I/O reads of the process, sockets included | `/proc/<pid>/io` `read_bytes` delta: reads from the storage layer |
| VRAM used | `nvidia-smi` | `nvidia-smi` |
| Restarts | pid change in `--pid-file` (no supervisor counter is read) | same |

A value that cannot be read is `null`, never `0`. On Windows the process read
rate includes network and pipe reads, so it says nothing about paging by
itself; a real disk or paging signal needs a host counter (for example the
`\PhysicalDisk` or `\Memory\Pages Input/sec` performance counters), which the
script does not sample. On Linux the rate is storage reads, so a sustained
non-zero value on a model that should be resident suggests weights are being
re-read.

**Per request** (`requests.jsonl`): index, prompt id, TTFT, total time, decode
tok/s, prompt and completion token counts with their source, finish reason,
output SHA-256, HTTP status and error kind.

**Fault schedule.** Decide the faults before the run and write them down with
their offsets, for example a 60-minute run with: child kill mid-stream at 10
and 40 minutes, cancel burst during a long prefill at 25 minutes, client stall
at 50 minutes. Inject with `--fault SECONDS:COMMAND` (or the operator's own
tooling) and log each injection time. Every injection must show evidence that
it fired (a child-side log line, a pid change, a dropped connection).
The fault unit tests under `tests/fault/` show the pattern: the supervisor's
outcome is asserted together with proof of the injection.

**Endpoint and headers.** `--base-url` may be the server root or end in `/v1`
(a trailing `/v1` is stripped; the driver always posts to
`<root>/v1/chat/completions`). `--header K=V` must contain `=`; to keep a
token off the command line use `--header-env K=ENVVAR`, which reads the value
from the environment. Error messages never include a header value.

**Pass/fail criteria.** `scripts/reliability_run.py` enforces exactly these
(fail on any). Each appears in `summary.json` `criteria` with its observed
value; optional ones appear only when their flag is given.

- `requests_completed`: at least one completed request;
- `host_samples`: at least 80% of the expected 1 Hz samples;
- `faults_fired`: every `--fault` command ran and exited 0. This proves only
  the exit code: a fault that changed nothing still counts as fired unless
  `--pid-file` with `--expected-restarts` (or a child-side event log) shows
  an effect;
- `fault_recovery`: after each fired fault, a request that starts after the
  fault and finishes within `--recovery-budget-s` (default 120) succeeds;
- `error_rate`: failed fraction of requests outside fault windows at most
  `--max-error-rate` (default 0). A fault window runs from the fault command
  start to its end plus `--fault-window` seconds (default 30); a request that
  overlaps it is an expected error, listed under `requests.expected_fault_failures`
  and not counted;
- `restarts_explained` (with `--pid-file`): pid changes equal
  `--expected-restarts` exactly (default 0, so declare the restarts your
  schedule causes); a kill command that exits 0 but restarts nothing fails;
- `deterministic_output`: the temperature-0 output hash for a prompt never
  changes (unless `--allow-nondeterminism`);
- `exact_token_counts` (with `--require-usage-tokens`): at least one request
  completed and every completed request took its token counts from the
  server's `usage` event, not the one-token-per-chunk fallback;
- `ttft_p95` (with `--max-ttft-p95`), `restarts` (with `--max-restarts`);
- `private_bytes_growth` (with `--max-private-growth-mib-per-hour`): slope per
  sampled pid using only samples after `--warmup-s` (default 600);
- `host_available_floor` (with `--min-available-mib`) and
  `host_commit_headroom_floor` (with `--min-commit-headroom-mib`): the
  minimum over the run must stay above the floor.

`--request-timeout` is a total per-request deadline. It is exact over the
response body: the socket timeout is shrunk to the time remaining before every
body read, so a server that trickles body bytes still trips it. Before the
body (sending the request and reading the status line and headers) each socket
operation is bounded by the time remaining, so only a server that trickles
header bytes can overrun it. A request that outlasts it is a failed request
(`deadline_exceeded`) and so counts
against `error_rate` like any other. Keep the raw JSONL with the result; small
reviewed snapshots may go in `bench/results/`.

## AgentX coding-agent trace benchmark (AGX-01)

Modeled on SemiAnalysis InferenceX AgentX to evaluate Sonder Inference's
serving stack (prefix caching, L2 `RadixPrefixCache`, continuous batching
slots, KV quantization, and speculative decoding) on traffic shaped like real
coding-agent sessions rather than single-turn prompts.

### Headline metric and reporting

- **Headline metric:** **P90 interactivity tok/s**
  (`p90_interactivity_tps = 1000 / client_observed_ms.itl_ms.p90` in
  `bench/http/httpbench/metrics.py`), pooled across token gaps of `ok` requests
  per concurrency level — the **slow tail** per-client decode speed matching
  SemiAnalysis InferenceX AgentX methodology
  (`https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide`).
  `p50_interactivity_tps` (`1000 / itl_ms.p50`) is reported alongside it.
- **Secondary per-request decode distribution:** `output_tps_per_user`
  summarizes per-request `(committed_output_tokens - 1) / (e2e_s - ttft_s)`
  with `p50`/`p90`/`p95`/`p99` (plus `output_tps_by_session` when `session_id`
  is present). Its `p90` (`output_tps_per_user_p90` / `req_decode_tps_p90_fast_tail`)
  is the **fast tail** and is retained only as a secondary diagnostic, never as
  the AgentX headline or primary A/B ranker.
- **Percentiles:** `summarize()` reports `p50`, `p90`, `p95`, and `p99`
  (linear interpolation) across TTFT, ITL, e2e, and `output_tps_per_user`.
- **Realized prefix-hit rate:** token-weighted
  `sum(cached_tokens) / sum(prompt_tokens)` over `ok` requests reporting
  `usage.prompt_tokens_details.cached_tokens`, `usage.sonder.timings.cache_n`
  (both emitted by `src/server/src/openai.cpp` `usage_json()`), or
  `timings.cache_n`. Unreported cache counts remain `null` (`unknown`), never
  `0`.

### Trace-replay workload (`semianalysisai/cc-traces-weka-062126`)

[`bench/http/httpbench/agentx.py`](../bench/http/httpbench/agentx.py) replays
local JSON/JSONL traces in the Weka schema of the public Hugging Face dataset
`semianalysisai/cc-traces-weka-062126` (which ships a single `traces.jsonl`
file containing the with-subagents corpus). The loader streams `traces.jsonl`
line by line and fits or discards each trace immediately so multi-gigabyte
corpora do not retain unused `hash_ids` in memory. The harness never downloads
datasets automatically; operators fetch the corpus locally for live runs, and
unit tests run offline against the small checked-in fixture
[`bench/http/fixtures/agentx_synthetic_traces.jsonl`](../bench/http/fixtures/agentx_synthetic_traces.jsonl).

Replay properties preserved:

- **Input length & growing shared prefix:** each 64-token KV-block `hash_id`
  maps deterministically (keyed by per-user `session_salt`) to synthetic text,
  so later turns and descendant subagents share byte-identical prefix blocks
  while distinct concurrent users are cache-busted from one another.
- **Request types and zero-output turns:** both `type: "s"` (streaming) and
  `type: "n"` (non-streaming) rows are replayed as main-agent turns (optional
  `ttft` fields are parsed and preserved). Real traces contain `out: 0` turns;
  because SI's `/v1/chat/completions` requires `max_tokens >= 1`
  (`src/server/src/openai.cpp`), those requests send `max_tokens: 1` and record
  `target_output_tokens: 0`, `effective_output_tokens: 1`, and
  `output_tokens_clamped: true`.
- **Subagent branches:** consecutive `type: "subagent"` blocks spawn
  concurrently as child branches (carrying `"priority": "subagent"`) and join
  before the parent agent's next main turn.
- **Idle gaps:** inter-turn think/idle gaps are preserved and capped at **10 s**
  (`--idle-gap-cap 10`).
- **Context cap (`--max-context-length`, default `32768`) and `--trace-fit {scale,drop}` (default `scale`):**
  the median trace request in `cc-traces-weka-062126` is ~142K tokens (most
  real traces exceed 32K after turn 1).
  - `--trace-fit scale` (**default**): proportionally shrinks each oversized
    trace's `in`/`out` lengths and stride-subsamples its `hash_ids` so peak
    `in + max(1, out) <= max_context_length` while preserving turn/subagent
    topology and prefix-share ratios, enabling real traces to run on a 16 GB
    GPU at 32K. The JSON and Markdown summaries record `trace_fit`, each
    trace's `scale_factor`, and `original_peak_tokens -> scaled_peak_tokens`.
  - `--trace-fit drop`: skips whole traces whose peak exceeds
    `--max-context-length` and lists them in `skipped_traces`, matching
    AIPerf's exact-length `--max-context-length` filter.
- **Concurrency sweep:** defaults to concurrency **1, 2, 4, and 8**
  (`--concurrency-levels 1,2,4,8`).

Example live-server invocation (pending live server bring-up):

```bash
python bench/http/bench_http.py run \
    --base-url http://127.0.0.1:11437/v1 \
    --trace-file /path/to/cc-traces-weka-062126/traces.jsonl \
    --max-context-length 32768 \
    --trace-fit scale \
    --concurrency-levels 1,2,4,8 \
    --label agentx-baseline
```

### NVIDIA AIPerf `agentx` (`inferencex-agentx-mvp`) compatibility finding

Reviewed against NVIDIA's documentation at
`docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark`
and SI's server implementation (`src/server/src/openai.cpp`,
`src/server/src/chat_handler.cpp`,
`src/server/include/sonder/inference/server.hpp`), without adding any AIPerf
dependency:

- **Verdict:** **Partially compatible — AIPerf can target SI's
  `POST /v1/chat/completions` and collect streaming/usage metrics, but SI does
  not enforce AIPerf's locked `ignore_eos: true` invariant today.**
- **What works out of the box:**
  - `--endpoint-type chat --streaming` posts to `/v1/chat/completions` with
    `stream: true` and `stream_options.include_usage: true`
    (`server_policy.cpp`, `openai.cpp:464-481`).
  - `--use-server-token-count` consumes `usage.prompt_tokens`,
    `usage.completion_tokens`, and `usage.prompt_tokens_details.cached_tokens`,
    all emitted by `usage_json()` (`openai.cpp:1218-1241`) on the final SSE
    chunk (`chat_handler.cpp:598-600`).
  - `ServerOptions::max_body_bytes` defaults to 4 MiB
    (`src/server/include/sonder/inference/server.hpp`), enough for 32K–128K
    token chat payloads (raise `--max-body-bytes` if testing near-1M contexts).
  - `parse_chat_request_with_mode()` ignores unknown top-level JSON keys
    instead of rejecting them, so `--extra-inputs ignore_eos:true` returns HTTP
    200 rather than HTTP 400.
- **Gaps blocking strict `inferencex-agentx-mvp` parity:**
  1. **`ignore_eos: true` is silently ignored:** AIPerf's
     `--scenario inferencex-agentx-mvp` locks `--extra-inputs ignore_eos:true`
     so every turn generates up to its recorded trace `out` length rather than
     terminating early on EOS. SI's `SamplingConfig` and `read_sampling()`
     (`openai.cpp:241-344`) have no `ignore_eos` knob, so early EOS stops will
     shorten outputs and skew throughput versus `submission_valid: true`
     InferenceX runs until `ignore_eos` is plumbed into `SamplingConfig` and
     the native `llamacpp` backend.
  2. **Explicit HF `--tokenizer` required:** AIPerf needs a Hugging Face
     tokenizer to synthesize prompts from Weka `hash_ids` blocks. Because SI
     serves local GGUF model IDs rather than HF repo IDs, operators must pass
     `--tokenizer <matching-hf-repo-or-local-dir>` explicitly even with
     `--use-server-token-count`.
  3. **Context cap override:** Pass `--max-context-length 32768` on 16 GB GPUs
     (AIPerf examples default to `128000`).
- **Recommendation:** Use `bench/http/bench_http.py` (`agentx_trace`) as the
  zero-dependency harness for SI A/B runs now; if external AIPerf
  `inferencex-agentx-mvp` cross-validation is desired later, first add
  `ignore_eos` support to `SamplingConfig`, `openai.cpp`, and `llamacpp`.

### Live-server A/B matrix (run when inference services are up)

Run each axis as a same-session **ABBA** block per the [Evidence rules](#evidence-rules)
above, holding model GGUF SHA-256, `--context-length 32768`,
`--max-context-length 32768`, `--concurrency-levels 1,2,4,8`, temperature `0`,
and trace file identical across A and B. Compare arms with
`python bench/http/bench_http.py compare A.json B.json` and record a
correctness line beside every speed row.

| Axis | Arm A (baseline) | Arm B (candidate) | `sonder-infer serve --backend llamacpp` flags | Primary metrics to compare |
| --- | --- | --- | --- | --- |
| Speculative decoding (`spec_type`) | `none` | `suffix` and `ngram` (also `ngram-mod`) | `--spec-type none` vs `--spec-type suffix` / `--spec-type ngram` | P90 interactivity tok/s (`1000 / itl_ms.p90`, `c1..c8`), ITL p50/p90, draft acceptance, committed vs proposed draft tokens |
| KV-cache quant + Flash Attention (`kv_cache_type_k`/`_v`) | `f16` / `f16` (`flash_attn=on`) | `q8_0` / `q8_0` (`flash_attn=on`, required for quantized V) | `--cache-type-k f16 --cache-type-v f16 --flash-attn on` vs `--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on` | P90 interactivity tok/s (`1000 / itl_ms.p90`), TTFT p50/p90, peak dedicated/shared VRAM (spill check), correctness parity |
| Concurrent KV slots (`n_slots` / `--slots`) | `1` slot | `4` slots | `--slots 1` vs `--slots 4` | P90 interactivity tok/s (`1000 / itl_ms.p90`) at `c2`/`c4`/`c8`, aggregate output tok/s, makespan, TTFT p90 under concurrency |
| L2 host-RAM `RadixPrefixCache` (`cache_ram_mib`) | off (`0`) | on (`8192` MiB, subject to host commit headroom) | `--cache-ram-mib 0` vs `--cache-ram-mib 8192` | Realized prefix-hit rate (`c2`/`c4`/`c8`), TTFT p50/p90 on warm turns/subagents, host private bytes |

*Status:* Live A/B runs are blocked while inference services are intentionally
down; only offline fixture tests have been executed.
