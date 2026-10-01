# GPU residency, VRAM spill and KV cache kernel pairing (llamaserver)

When `llamaserver` spawns llama-server, it checks two performance problems
that the 2026-09-29 benchmarks on an RTX 5070 Ti (16 GB) found. Neither
problem shows up in normal tools:

1. **VRAM spill.** On Windows, a llama-server run with `--fit off` that does
   not fit in VRAM still loads. The WDDM driver places the overflow in shared
   system memory. `nvidia-smi` shows the same ~15.8 GB whether or not
   anything spilled, but the spilled run is 12% to 15x slower.
2. **K/V cache types without a FlashAttention kernel.** llama.cpp's default
   CUDA build has no FlashAttention vector kernel for mixed K/V cache types,
   such as `q8_0`/`q5_1`, or for `q5_1`/`q5_1`. llama-server converts K and V
   to f16 at run time and prints one log line about it.

These checks and the dedicated-residency checks below only report by default.
Child lifecycle changes require `refuse`, `auto_fit`, or the opt-in eviction
restart policy. Attach mode has
no child process to measure, so it reports nothing, and its responses are
unchanged.

## What was measured

All runs used Ollama's bundled llama-server (0.4.1-dev) with Qwen3.8-27B,
`--flash-attn on --fit off --n-gpu-layers 999`. Full reports:
`bench-iq4xs.md` (IQ4_XS) and `bench-kv-fit` (Q3_K_XL probes), both in the
sonder-eco workspace.

Q3_K_XL, 8k-token prompt. "Shared" is the process's
`GPU Process Memory\Shared Usage` after load.

| K/V | ctx | dedicated | shared | short decode | 8k decode |
|---|---|---|---|---|---|
| q8_0/q5_1 | 49152 | 13764 MiB | 148 MiB | 47.5 tok/s | 43.1 tok/s |
| q8_0/q5_1 | 91904 | 15184 MiB | 190 MiB | 49.8 tok/s | 44.8 tok/s |
| q8_0/q5_1 | 96000 | 15222 MiB | **292 MiB** | 44.0 tok/s | 37.2 tok/s |
| q8_0/q5_1 | 100096 | 15266 MiB | **388 MiB** | **15.9 tok/s** | **17.9 tok/s** |
| q8_0/q8_0 | 83712 | 15320 MiB | 182 MiB | 49.8 tok/s | 43.7 tok/s |
| q8_0/q8_0 | 87808 | 15314 MiB | **348 MiB** | 40.7 tok/s | 39.7 tok/s |
| q5_1/q5_1 | 100096 | 14966 MiB | 198 MiB | 50.0 tok/s | 44.3 tok/s |
| q5_1/q5_1 | 116480 | 15302 MiB | **342 MiB** | 43.1 tok/s | 38.6 tok/s |

IQ4_XS with q8_0/q5_1:

| ctx | shared | effect |
|---|---|---|
| 65536 | 0.33 GB | fastest IQ4_XS run (36 tok/s short decode) |
| 73728 | 0.71 GB | 17 tok/s short decode, 4.7 tok/s decode at 16k |
| 81920 | 0.95 GB | decode at 72k: 1.26 tok/s |
| 100096 | 2.1 GB | 2-15x slower overall |

Every spill appears after dedicated usage reaches the same ~15.2-15.3 GB
ceiling. From that point, dedicated usage stops growing and the overflow
lands in shared memory.

## Spill guard

After llama-server reports ready on `/health`, the supervisor sums the
child's `\GPU Process Memory(pid_<PID>_*)\Dedicated Usage` and
`Shared Usage` PDH counters over all of the process's adapter instances.
It samples again every `sample_interval_ms`. Sampling runs on the
supervisor's monitor thread and never on a request path.

The child counts as **spilled** when `shared > baseline + threshold_mib`,
where `baseline = baseline_mib + baseline_per_1k_ctx_mib × (--ctx-size / 1024)`.
`baseline_per_1k_ctx_mib` defaults to 0 (a fixed baseline).

- The default threshold is **256 MiB** above a baseline of **0**. In the Q3
  runs, the largest clean sample was 198 MiB and the smallest spilled sample
  was 292 MiB.
- The IQ4_XS run at 65536 (0.33 GB shared) is above the threshold. Its
  dedicated usage was also at the ceiling. It was the fastest IQ4_XS
  configuration measured, but there was no smaller-context IQ4_XS run to
  compare it with. That makes it a possible mild spill, like the Q3 runs at
  292-348 MiB, which were 12-19% slower.
- The baseline is an absolute number, not a sample. A sample taken after the
  model loads already contains any spill that happened during the load, so
  it cannot serve as its own baseline. Raise `baseline_mib` if your
  configuration keeps more pinned host memory when it is clean.
- **MTP speculation raises the clean line.** With `--spec-type draft-mtp
  --spec-draft-n-max 2` (Q3_K_XL, q4_0/q4_0 KV), the clean shared usage
  measured 2026-09-30 was about 126 MiB + 2 MiB per 1k ctx, plus about
  40 MiB after the first prompt: roughly 310 MiB at 73,728, already above
  the default 256 MiB limit. Runs 33 MiB or more above that line were
  slower. For such a profile use `"baseline_mib": 166,
  "baseline_per_1k_ctx_mib": 2, "threshold_mib": 32`. Otherwise `auto_fit`
  shrinks the context of a healthy child. Without speculation the clean
  line grows about 1 MiB per 1k ctx, which the default 256 MiB threshold
  already absorbs up to about 140k.
- `shared_baseline_bytes` in the runtime status and telemetry reports the
  effective baseline at the child's current `--ctx-size`.

| policy | on a spilled child after readiness |
|---|---|
| `warn` (default) | serve it; report `spilled: true` and a `vram_spill` warning |
| `refuse` | stop it and fail startup: `llamaserver: VRAM spill detected: shared GPU memory 972 MiB exceeds the 256 MiB spill threshold above a 0 MiB baseline (dedicated 15100 MiB, ctx 100096); refusing to serve …` |
| `auto_fit` | stop it and relaunch with a smaller `--ctx-size` until it is not spilled |

How `auto_fit` steps down the context:

- Each step is `ctx → align_down(ctx × fit_step_factor, fit_step_align)`.
  With the defaults (0.85 and 1024), 100096 → 84992 → 71680 → 60416.
- Each step is at least one alignment unit smaller than the last.
- The context never goes below `fit_min_ctx` (default 8192).
- There are at most `fit_max_attempts` relaunches (default 4).

When the floor or the attempt limit is reached and the child still spills,
startup fails with the numbers. `auto_fit` never loops. `auto_fit` requires
an explicit `--ctx-size`/`-c` in `args`; without one, the configuration is
rejected before anything is launched. Relaunches do not use the crash-restart
budget. A crash restart after a fit keeps the fitted context. Each relaunch
gets a full `startup_timeout_ms` to load.

The shared-memory spill policy only acts on the sample taken at readiness. A spill that appears
later, for example because another application takes VRAM, is reported but
never kills a child that is already serving. A probe error does not refuse
or refit the child either. Neither does a platform without the counters:
non-Windows platforms report `status: "unsupported"`, and NVML is a possible
later addition. The separate residency policies below can act on confirmed
dedicated-memory loss after readiness.

## Missing offload and VRAM eviction

Two further incidents were reported by the owner on **2026-09-30**, on an
RTX 5070 Ti (16 GB) under Windows WDDM with Qwen3.8-27B (16 attention and
48 DeltaNet layers). These are owner measurements, not results from the
GPU-free regression suite:

- A PATH problem prevented the spawned server from loading its CUDA backend.
  The 27B ran on the CPU at **2.2 tok/s decode**, **0.25 tok/s prefill**, with
  about **9 GB working set**. `/health` was ready, dedicated and shared GPU
  usage were both zero, and no warning appeared because no log file was set.
- When Ollama loaded another **13.4 GB** model, the server's dedicated usage
  fell from **14,736 MiB to 2,889 MiB**. Shared usage was zero at observation
  (peak shared **424 MiB**), `spilled` was false, and performance stayed
  degraded until the server was restarted. A shared-usage threshold alone
  cannot detect this loss of residency.

The residency guard uses the same post-readiness PDH samples and leaves the
existing shared-memory spill calculation unchanged. It has two independent
warnings:

- `gpu_offload_missing`: dedicated usage stays **below 512 MiB for three
  consecutive successful samples** while GPU offload is expected. The first
  sample is taken after `/health` becomes ready; with the default five-second
  interval, confirmation normally takes another ten seconds. A zero-instance
  successful PDH read counts as zero usage. Failed/unsupported probes and an
  unknown PID do not constitute evidence and interrupt unconfirmed streaks.
- `vram_evicted`: dedicated usage stays below its post-readiness high-water
  mark by **more than 25% OR more than 2 GiB** for three consecutive samples.
  Equality is not a trigger. The peak must be at least the dedicated floor.
  A normal sample interrupts an unconfirmed streak. Either threshold can be
  disabled with zero; at least one must be enabled. This is evidence of a
  residency loss, not proof that WDDM was its cause.

GPU intent is conservative: explicit `--n-gpu-layers`, `--gpu-layers` or
`-ngl` (separate or `=` values) with a positive count, `all`, or `-1` enables
the missing-offload check. The last occurrence wins; explicit zero or an
unrecognized value suppresses it. The supervisor cannot prove that an
arbitrary external executable is CUDA-capable. With no GPU-layer flag, set
`expect_gpu: true` only when that is known; the default is false. Eviction
detection needs no flag because the measured high-water mark establishes
prior dedicated residency. For intentionally tiny GPU allocations, lower the
floor or disable this nested guard.

Missing offload warns by default, including under `auto_fit` (reducing context
does not repair a missing CUDA backend). Under `spill_guard.policy: "refuse"`,
it closes request admission, drains tracked requests and warm-up, stops the
child and enters a terminal failure with the measured numbers and a CUDA/PATH
hint. For `consecutive_samples: 1`, it can refuse before exposing readiness.
Otherwise readiness may already have been returned before confirmation.

`on_eviction: "restart"` opts into reloading the child with the same arguments
and fitted context. It closes admission atomically with the active-request
check and waits for **all Sonder-tracked HTTP requests and prefix warm-up**
to finish. New requests wait through recovery (subject to existing readiness
timeouts and cancellation). A slow/hung request can delay recovery until its
own request timeout; the guard does not kill it to force an idle period.
Clients that bypass Sonder and call the private child port directly cannot
be tracked; use this policy only when Sonder owns access to that child.

There is at most **one eviction relaunch per supervisor lifetime** by default,
independent of the existing crash and context-fit budgets. Repeated drops,
including in a replacement child, cannot reset this budget. Exhaustion falls
back to warnings and keeps serving. Release competing GPU allocations before
reloading; a restart cannot guarantee that enough VRAM is available. Existing
crash-restart rules still apply if a replacement crashes or fails to launch.

Findings are latched per child to avoid flapping; a new child resets the
detector's peak and streaks. The last incident remains visible across
restarts. Only when an incident has occurred does `gpu_memory` gain the
additive `residency` object:

```json
"residency": {
  "gpu_offload_missing": false,
  "vram_evicted": true,
  "peak_dedicated_bytes": 15451815936,
  "observed_dedicated_bytes": 3029336064,
  "eviction_restarts": 1,
  "action": "restarted"
}
```

The booleans record incidents observed during this supervisor's lifetime;
they do not claim that the replacement is still degraded. The peak and
observed bytes describe the latest detected child incident; current usage
remains in `gpu_memory.dedicated_bytes`. Actions are `warn`,
`waiting_for_idle`, `refused`, `restarting`, `restarted` or
`restart_exhausted`. Warnings include peak, observed and current dedicated
bytes. Health (`backends[].runtime`), `/v1/models` (`sonder.runtime`) and
`backend.gpu_memory.sample` telemetry use this same serialization;
`backend.warning` carries the two new warning codes. No residency object
or warning is added when no incident occurs. Existing GPU status strings,
spill flags, and context-fit fields retain their meanings.

## KV cache pairing and log diagnostics

The up-front check reads `--flash-attn`/`-fa`, `--cache-type-k`/`-ctk` and
`--cache-type-v`/`-ctv` from `args`. It warns unless FlashAttention is
explicitly off:

- `kv_type_mismatch` (warning): K and V types differ. Use a matched pair such
  as `--cache-type-k q8_0 --cache-type-v q8_0`.
- `kv_type_no_vector_kernel` (warning): `q5_1`/`q5_1`. It also logs the
  conversion.
- `kv_type_kernel_unknown` (info): another matched type outside
  f16/bf16/q8_0/q4_0.

In the table above, the conversion itself cost little at an 8k prompt. On
q8_0/q5_1 at 83712 decode ran at 44.8 tok/s, and on q8_0/q8_0 at 43.7 tok/s.
Its cost at longer prompts was not measured separately. The spill was what
made these runs slow: the smaller q5_1 K cache is what let 100096 fit
without spilling.

Set `log_file` to have the supervisor pass `--log-file <path>` to the child.
If `args` already contains `--log-file`, that file is used instead. The
supervisor reads the log in bounded chunks (at most 256 KiB per tick, lines
capped at 4 KiB, at most 32 distinct warnings) and looks for these lines:

| code | log line (substring) | requires `-lv 4` |
|---|---|---|
| `kv_kernel_f16_fallback` | `no FlashAttention vector kernel compiled for K/V types q8_0-q5_1, converting K and V to f16 instead (slow)`; `details.k_type`/`v_type` | no |
| `mtp_tensors_ignored` | `model has unused tensor blk.64.nextn.… -- ignoring`; folded, with `details.layer`, `tensors`, `ignored_bytes` | no |
| `partial_gpu_offload` | `offloaded N/M layers to GPU` with N < M | yes |
| `cpu_buffer_fallback` | `cannot be used with preferred buffer type …, using CPU instead` | yes |
| `no_gpu_device` | `no usable GPU found` | no |
| `gpu_init_failed` | `failed to initialize CUDA` | no |

The first two appear exactly like this in the 2026-09-29 logs. The last four
are taken from llama.cpp's source strings; the benchmark runs offloaded every
layer, so those lines did not appear.

Log diagnostics remain **opt-in through `log_file` or `--log-file`**. There
is no separate enabled-without-a-file setting today. Default stderr capture
would change both native launchers: inherited Windows handles/Job Object
lifetime and POSIX spawn file actions/parent-liveness pipe ownership, plus
a continuously drained, bounded pipe and shutdown/join handling to prevent
the child from blocking on a full pipe. That wider process-lifecycle change
is deferred. The dedicated-memory guard above detects the observed CPU
fallback without requiring logs; it cannot identify the exact CUDA failure.
Set a log file for `no_gpu_device`, `gpu_init_failed` and
`cpu_buffer_fallback` classification. Existing log read/line/warning bounds
are unchanged.

## Stall guard

The native child monitor also polls `/metrics` and `/slots` at the existing
sample cadence, with a minimum interval of five seconds. A stall is present
only when `requests_processing > 0` and neither cumulative token counter has
moved for `stall_seconds`. The default is enabled, 90 seconds, and `warn`.
Warnings include `processing`, `deferred`, and monotonic `since_ms`; health
adds `runtime.stall.detected_at` and `since_ms` while retaining `ready`.
Scrape failures are warnings and clear stall evidence. A 404 disables polling
for that child after one `metrics_unavailable` warning. `restart` uses the
existing crash-restart path and counts against `max_restarts`; attach mode
warns only because the supervisor does not own the external process.

The observed child metrics are cumulative process counters, not per-request
deltas or measured speed. Speculation records cumulative accepted and draft
totals and positional totals. `mean_accepted_len` is reported when available;
`speedup_est` uses `(1 + mean_accepted_len) / (1 + 0.6 * n_max)`, with
`n_max` taken from explicit `--spec-draft-n-max` and `null` when unknown.

## Configuration

GPU and log-file diagnostics apply to spawn mode. `stall_guard` also applies
to native attach, where it can only warn. Each value shown is the default;
`stall_seconds` accepts integers in [1, 86400].

```json
{
  "mode": "spawn",
  "executable": "C:/llama/llama-server.exe",
  "args": ["--model", "m.gguf", "--ctx-size", "100096", "--flash-attn", "on",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--fit", "off"],
  "log_file": "",
  "kv_pairing_check": true,
  "spill_guard": {
    "enabled": true,
    "policy": "warn",
    "threshold_mib": 256,
    "baseline_mib": 0,
    "baseline_per_1k_ctx_mib": 0,
    "sample_interval_ms": 5000,
    "fit_step_factor": 0.85,
    "fit_step_align": 1024,
    "fit_min_ctx": 8192,
    "fit_max_attempts": 4,
    "residency": {
      "enabled": true,
      "expect_gpu": false,
      "min_dedicated_mib": 512,
      "consecutive_samples": 3,
      "eviction_fraction": 0.25,
      "eviction_mib": 2048,
      "on_eviction": "warn",
      "max_eviction_restarts": 1
    }
  },
  "stall_guard": {
    "enabled": true,
    "stall_seconds": 90,
    "policy": "warn"
  }
}
```

In C++, set `LlamaServerBackendOptions::spill_guard`
(`LlamaServerSpillGuardOptions`, with thresholds in bytes) and
`LlamaServerBackendOptions::diagnostics` (`LlamaServerDiagnosticsOptions`).
The status comes from `Backend::runtime_status()`. It is backend-neutral:
other backends return `std::nullopt` by default.
Residency options live in `LlamaServerSpillGuardOptions::residency`
(`LlamaServerResidencyGuardOptions`); the C++ floor/amount are byte counts.
`spill_guard.enabled: false` disables all counter checks;
`spill_guard.residency.enabled: false` disables only the new detectors.

## Where it shows up

All of these are additive. A backend that reports no runtime status
produces byte-identical responses.

- `GET /v1/sonder/health`: `backends[].runtime`
- `GET /v1/models`: `data[].sonder.runtime` for models served by that backend

```json
"runtime": {
  "gpu_memory": {"probe": "pdh", "status": "ok", "dedicated_bytes": 16007561216,
                 "shared_bytes": 406847488, "peak_shared_bytes": 406847488,
                 "shared_baseline_bytes": 0, "spill_threshold_bytes": 268435456,
                 "spilled": true, "samples": 3},
  "context": {"policy": "warn", "configured_ctx": 100096, "fitted_ctx": 100096,
              "fit_attempts": 0, "outcome": "not_needed"},
  "warnings": [{"code": "kv_kernel_f16_fallback", "severity": "warning", "source": "log",
                "message": "…", "details": {"k_type": "q8_0", "v_type": "q5_1"}, "count": 1}]
}
```

- `gpu_memory.status` is one of `ok`, `not_sampled`, `unsupported`, `error`
  or `disabled`.
- `context.outcome` is one of `not_needed`, `fitted`, `refused`,
  `floor_reached` or `exhausted`.
- The context sizes are `null` when `args` has no `--ctx-size`.
- Telemetry, from the engine's periodic sampler
  (`EngineOptions::device_sample_interval`): `backend.gpu_memory.sample` is
  emitted for each new sample. It carries the `gpu_memory` fields plus
  `backend`, `fitted_ctx` and `fit_outcome`. `backend.warning` is emitted once
  for each distinct warning. See [TELEMETRY](../TELEMETRY.md).
