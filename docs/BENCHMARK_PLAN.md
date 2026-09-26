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
