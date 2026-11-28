# Benchmark harness

Implements the first slice of docs/BENCHMARK_PLAN.md: interactive chat,
coding, long-context prefill, and agent fan-out. The harness runs through
Engine sessions, so native `llamacpp`, the compatibility backends, and the
mock backend share one code path, and Session telemetry is emitted as usual.

- Library: `sonder/inference/benchmark.hpp` (`bench::run`,
  `render_markdown`, `write_results`), compiled into `sonder_inference`.
- `sonder-infer bench ...`: the CLI subcommand. It writes a JSON results
  document.
- `sonder-bench`: a standalone runner. It writes JSON plus a markdown summary
  and adds a time budget, prompt filtering, `--require-idle`, and a telemetry
  file option.

For benchmarking a server over HTTP (raw `llama-server`, `sonder-infer
serve`, Sonder Runtime `/v1`), with cached-token, inter-token-latency,
concurrency, long-context recall and GPU-spill measurements, see
[`bench/http/`](http/README.md).

Run from the repository root:

```
sonder-bench --model <path.gguf> --ctx-size 4096 --threads 4 --gpu-layers 0 \
             --warmup 1 --runs 3 --hardware "<CPU, RAM>" --label native-baseline-v1
sonder-bench --backend llamacpp --model <path.gguf> --gpu-layers -1
sonder-bench --backend ollama --list-models       # compatibility option
sonder-bench --backend ollama --model <name> --require-idle
sonder-bench --backend mock --model mock:tiny     # harness check only; NOT a perf claim
sonder-bench --help
```

The default backend is `llamacpp`; build with `SONDER_WITH_LLAMA_CPP=ON`
(for example, `pwsh -NoProfile -File scripts/build.ps1 -Preset msvc-debug -Test
-ExtraCMakeArgs '-DSONDER_WITH_LLAMA_CPP=ON'`). A build without native support
reports a configuration hint and exits 2; it does not fall back to Ollama.
`--model` takes a local GGUF path. Native-only options are `--ctx-size N`
(default 4096; 0 uses the model training context), `--threads N` (default 0
uses the backend default), and `--gpu-layers N` (default -1 offloads all
layers when a GPU backend is compiled in; 0 selects CPU). `--list-models`
does not scan arbitrary directories; use an explicit GGUF path for native
runs. `--require-idle` remains Ollama-only and requires `--backend ollama`.

Other defaults are corpus `bench/corpus/baseline.json`, output directory
`bench/results/`, and a budget of 540 s. Output files are named
`<date>-<backend>-<model>-<corpus>.json` and `.md`.

## Metrics

**Per request** (session-measured): TTFT, total latency, prompt and completion
tokens, stop reason, and errors. Decode tok/s comes from the backend's
reported token counts and evaluation duration (including native llama.cpp
timings). Otherwise it falls back
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
| `corpus/bonsai_head2head.json` | Retained Bonsai regression corpus; select prompts and a context size appropriate for your GGUF |

Schema `sonder.inference.corpus/1`. The baseline corpus adds optional
`fillers`, and prompts can set `context: {filler, repeat}`, `shared_prefix`,
and `children` (for agent fan-out). Bump `version` whenever prompts change.

Caveat: `SamplingConfig` has no `num_ctx`, so Ollama runs use the server's
default context window. The long-context prompt is sized to fit a 4096
default.

The bench unit suite checks CLI defaults, native options, and invalid
combinations without weights. CTest `sonder.bench.tool_llamacpp` runs the
standalone CLI with the default backend when `SONDER_TEST_GGUF` names a local
GGUF and native support is enabled; otherwise it reports **Skipped**. It
forces CPU execution with one thread and a 512-token context, so use a small
compatible test model. This smoke test is not performance evidence.

## Reliability runs

Long, slow-model runs use [`scripts/reliability_run.py`](../scripts/reliability_run.py)
and the evidence and reliability-run rules in
[`docs/BENCHMARK_PLAN.md`](../docs/BENCHMARK_PLAN.md). It writes
`requests.jsonl`, `samples.jsonl`, `events.jsonl` and `summary.json`, and exits
non-zero when no request completed. `python scripts/reliability_run.py
--self-test` needs no server or weights. Supervisor fault-injection tests live
in `tests/fault/` and run under CTest as `sonder.fault.*`.
