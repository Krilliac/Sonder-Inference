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
under `D:/sonder-eco/wf/bench/`. The lane author runs only fake-server tests;
real GPU acceptance, deployment, commits and PR operations belong to the
integrator. Each result should include exact server revision, model, launch
configuration, pins, sampling, context size and the arm's scenario JSON.
