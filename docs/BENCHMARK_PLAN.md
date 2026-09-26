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
