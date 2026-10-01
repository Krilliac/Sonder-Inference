# Calibrating llama-server

`sonder-infer tune` searches for the fastest **measured** non-spilling profile
for a local model within a wall-clock budget. It writes a spawn configuration
that `sonder-infer serve --llamaserver-config` can read directly. It does not
change the configuration or processes of an already running server.

Live calibration **spawns the supplied executable and uses the GPU**. Use a
trusted executable and close other GPU applications first. Desktop VRAM use
varies (515–680 MiB in the owner's measurements); a result is a measurement
under current conditions, not a guarantee for every future workload. No model,
executable, dependency, or package is downloaded. The subprocess supervisor
keeps each server on loopback with a separately chosen ephemeral port.

```powershell
sonder-infer tune --executable C:/llama/llama-server.exe `
  --model D:/models/model.gguf --out tuned.json --budget-minutes 20

# Preview the search space without starting any process, reading the model,
# probing the GPU, or writing an output file (MTP rows remain conditional).
sonder-infer tune --executable C:/llama/llama-server.exe `
  --model D:/models/model.gguf --dry-run

# Use the resulting best measured profile. Specify the backend's model ID as
# usual; the spawn config's --model argument is the actual local GGUF path.
sonder-infer serve --backend llamaserver --model default --llamaserver-config tuned.json
```

Options:

| Option | Meaning |
|---|---|
| `--executable PATH` | Required trusted llama-server executable. Paths with spaces must be quoted. |
| `--model GGUF` | Required single-file GGUF v2/v3 model. |
| `--mmproj PATH` | Load the optional projector in every candidate and include it in the output. Workloads remain text-only. |
| `--out PATH` | Output spawn config, default `tuned.json`. Parent directory must exist. |
| `--budget-minutes N` | Work budget, default 20; accepts decimals from 0.1 to 1440. |
| `--grid JSON` | Inline JSON, a JSON file path, or `@PATH`; at most 64 KiB. |
| `--env KEY=VALUE` | Repeatable child-only overrides, inherited by help and candidate processes and recorded in the config's `env` object. Does not change the parent environment. Avoid secrets: the output stores these values. |
| `--thorough` | Also measure a prefill of `ctx - 1024` tokens for timed candidates. |
| `--dry-run` | Print the candidate grid and adaptive measurement rules, then exit. |

Live memory qualification currently requires Windows WDDM's PDH **GPU Process
Memory** counters. Other platforms support planning and GPU-free unit tests;
a live run fails before spawning when the probe is unsupported. Missing or
unreadable counters never count as a clean sample. Multi-shard and big-endian
GGUF files are rejected rather than guessing whether another shard has MTP.

## Search and measurements

The defaults are K/V `q4_0/q4_0`, `q8_0/q8_0`, `q8_0/q4_0` (k8v4), and
`q5_1/q5_1`; context 32768 through 139264 in 4096-token increments; ubatch
512, 1024, 2048; batch 2048; and MTP draft limits 0, 1, 2, 3. FlashAttention is
on, GPU layers are requested in full, automatic upstream fitting is off, and
there is one slot so `/props` describes the actual context being tested.

1. Inspect only the GGUF directory, without loading tensor data. Reuse the
   engine's architecture classifier. Nonzero MTP requires an actual `.nextn.`
   tensor name **and** `--spec-type draft-mtp`/`--spec-draft-n-max` advertised
   by a bounded `--help` process. A model architecture name alone is insufficient.
2. Reserve 60% of the remaining budget for round-robin context bisection, one
   lane per K/V pair at the smallest ubatch and no MTP. Each cheap probe loads
   the model, waits for readiness, verifies `/props` context and slot count,
   and processes exactly 8192 prompt tokens with one completion token.
   Only clean and spill outcomes move a bisection bound. Other failures abandon
   that lane; failed probes are retained in the report.
3. Benchmark the clean-edge-minus-4096 and 65536-token profiles where feasible.
   The short decode uses 128 prompt tokens and 256 generated tokens; the prefill
   uses 8192 prompt tokens and one generated token. Then explore ubatch/MTP
   variants of at most two leaders: fastest decode at >=65536 context, and
   largest qualified context. Every variant must also pass a clean probe at
   **its own context plus 4096**, before its timed run. A variant that spills is
   rejected; the tuner does not infer a margin from a different variant.
4. Tokenize fixed text once per child, repeating the resulting valid token IDs
   to exact prompt lengths. Requests set greedy sampling, seed 42, `ignore_eos`,
   and `cache_prompt: false`. Cached/truncated work or missing/zero timings is
   not accepted as a throughput measurement. MTP benchmarks must report draft
   activity; the baseline must not. Rates come from upstream measured token
   counts and durations, not model-quality claims.
5. Sample the existing PDH source during loading and workloads (200 ms), keeping
   peak dedicated/shared bytes. The threshold is the spill guard's independent
   clean line plus threshold: by default `0 + 256 MiB` for candidates without
   speculation. MTP candidates use their own rule (`spill_mtp`: `166 MiB + 2 MiB`
   per 1k ctx, `+ 32 MiB`), because MTP raises a healthy load's shared usage to
   about 126 MiB + 2 MiB per 1k ctx (+~40 MiB after the first prompt; about
   310 MiB at 73,728, measured 2026-09-30). The fixed rule would reject every
   healthy MTP load near 73k. A loaded model is never
   used as its own baseline, which could hide load-time spill. The strict `>`
   comparison matches the spill guard. The existing log parser detects
   `kv_kernel_f16_fallback`; these rows are reported as **slow kernel**, retained,
   and excluded from recommendations. Such kernels may depend on executable
   build options, so mismatched pairs are tested rather than rejected by name.
6. Stop each child through the supervisor. Require two consecutive zero-memory
   samples for that PID before starting another candidate. Any release-read
   error or memory remaining after 10 seconds stops the whole search. Work and
   HTTP operations are cancelled on the deadline or Ctrl-C. Shutdown and this
   bounded release check may extend beyond the work budget by about 12 seconds;
   OS process creation, file I/O and counter calls are synchronous.

A finite grid, binary search and a fixed two-leader refinement bound the work
even with a very large time budget. The optimum is limited to profiles actually
measured before the budget expires. Performance is a short synthetic workload
estimate; use the HTTP benchmark harness for representative agent traffic,
prefix reuse, concurrency and long-context quality checks.

## Custom grid

Arrays replace defaults. Unknown keys, duplicate entries, fractional integers,
unsupported KV names and inconsistent batch sizes are rejected. `ctx` must be
strictly increasing, 4096-aligned and in [16384, 1048576]. At most 4096 grid
combinations are accepted. `mtp` must include 0 for the baseline. Recommendations
and their safety probes can use derived contexts outside the explicit `ctx`
array: the measured edge minus 4096 and the fixed >=64k comparison point.

```json
{
  "kv": [["q4_0", "q4_0"], ["q8_0", "q4_0"]],
  "ctx": [65536, 69632, 73728, 77824, 81920, 131072, 135168],
  "ubatch": [512, 1024, 2048],
  "batch": 2048,
  "mtp": [0, 1, 2, 3],
  "spill_threshold_mib": 256,
  "shared_baseline_mib": 0
}
```

The fixed 4096 margin still applies to sparse custom context arrays; an edge
means the largest **observed clean** grid point, not a claim about untested gaps.
Only change the independent shared-memory baseline with separate clean-line
evidence. Enlarging it to match a spilling run defeats spill detection.

## Output and failures

Stdout is a Markdown summary with both recommendations and all candidate rows.
Progress goes to stderr. `tuned.json` has ordinary `mode`, `executable`, `args`,
`env`, `max_restarts: 3`, `startup_timeout_ms: 120000` and `spill_guard` keys, plus an inert `results` object
with schema `sonder.inference.tune/1`. The profile is meant for daily use: its
spill guard carries the rule the winning candidate was judged by (including
`baseline_per_1k_ctx_mib` for MTP) with policy `auto_fit` and `fit_min_ctx`
32768, so if the desktop later takes more VRAM the server shrinks the context
instead of spilling silently or refusing to start.

`results.candidates` records K/V, ctx, ubatch, MTP n, phase, peak shared/dedicated
MiB, decode and prefill tok/s, optional long-prefill tok/s, served context,
safety-probe context, memory-release confirmation, verdict, and diagnostic
detail. Unmeasured numbers are JSON `null`. `results.fast_default` and
`results.long_context` each contain a result index and a complete `config`;
either may be null. The root config selects fast default when available,
otherwise long context. To use the alternative, save its `config` object as
another file. Configs still reject unknown top-level execution keys.

When no profile qualifies, an existing `tuned.json` is left intact and only
`tuned.json.results.json` is written. That report is deliberately not loadable
as a spawn config. Scratch help/child logs are removed on ordinary completion
and cancellation; forced termination can leave the `.sonder-tune-*` directory.

Exit codes follow the CLI: 0 for a dry run or a successful measured profile,
1 for runtime failure/no eligible profile/unsafe cleanup, 2 for invalid input,
130 for Ctrl-C. Budget-limited runs may return 0 if a verified profile already
exists; the summary and JSON disclose whether the work budget was exhausted.

## Tests

`sonder.llamaserver.*` includes policy, GGUF directory, CLI, JSON round-trip,
and supervisor/workload tests. The runner tests reuse `FakeLauncher` and the
spill guard's `FakeCounters`; fake HTTP responses provide counts/timings and
faults. They do not spawn a real llama-server, read weights, or use a GPU.
Full build/CTest and a separately authorized live calibration are still
necessary before claiming real-machine performance or compatibility.
