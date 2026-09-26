# Benchmark harness

Implements the first slice of docs/BENCHMARK_PLAN.md: interactive chat,
coding, long-context prefill, and agent fan-out. The harness runs through
Engine sessions, so the Ollama path, the mock backend, and future native
backends share one code path, and Session telemetry is emitted as usual.

- Library: `sonder/inference/benchmark.hpp` (`bench::run`,
  `render_markdown`, `write_results`), compiled into `sonder_inference`.
- `sonder-infer bench ...`: the CLI subcommand. It writes a JSON results
  document.
- `sonder-bench`: a standalone runner. It writes JSON plus a markdown summary
  and adds a time budget, prompt filtering, `--require-idle`, and a telemetry
  file option.

Run from the repository root:

```
sonder-bench --backend ollama --list-models
sonder-bench --backend ollama --model <name> --warmup 1 --runs 3 --require-idle \
             --hardware "<GPU + VRAM, CPU, RAM, driver>" --label ollama-baseline-v1
sonder-bench --backend mock --model mock:tiny     # harness check only; NOT a perf claim
sonder-bench --help
```

Defaults are corpus `bench/corpus/baseline.json`, output directory
`bench/results/`, and a budget of 540 s. Output files are named
`<date>-<backend>-<model>-<corpus>.json` and `.md`.

## Metrics

**Per request** (session-measured): TTFT, total latency, prompt and completion
tokens, stop reason, and errors. Decode tok/s comes from the backend's
`eval_count / eval_duration` when reported (Ollama). Otherwise it falls back
to wall clock after the first chunk. Client decode tok/s is always computed as
(tokens - 1) / (total - TTFT). Prompt tok/s is computed from
`prompt_eval_count / prompt_eval_duration`.

**Per prompt and overall:** n, mean, p50, p95, p99, min, and max.

**Agent fan-out:** all children run concurrently (one session each). Each
batch records its makespan and aggregate tok/s.

**Provenance:** engine version and commit, backend version, model
family/parameters/quantization/format/context length, hardware string, host
platform and devices, corpus name/version/hash, sampling (greedy, fixed seed),
warmup and measured runs, cache state, cold first-request time and backend
load time, wall time, and `truncated_by_budget`.

## Corpora

| File | Contents |
|---|---|
| `corpus/smoke.json` | Tiny interactive set for harness bring-up and CI |
| `corpus/baseline.json` | Baseline workload families (short, coding, medium, ~2.4k-token long context, 4-way fan-out with a shared ~600-token prefix) |

Schema `sonder.inference.corpus/1`. The baseline corpus adds optional
`fillers`, and prompts can set `context: {filler, repeat}`, `shared_prefix`,
and `children` (for agent fan-out). Bump `version` whenever prompts change.

Caveat: `SamplingConfig` has no `num_ctx`, so Ollama runs use the server's
default context window. The long-context prompt is sized to fit a 4096
default.
