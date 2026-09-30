# VRAM spill and KV cache kernel pairing (llamaserver)

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

Both checks only report by default. Nothing about how the child is started
changes unless you choose the `refuse` or `auto_fit` policy. Attach mode has
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

The guard only acts on the sample taken at readiness. A spill that appears
later, for example because another application takes VRAM, is reported but
never kills a child that is already serving. A probe error does not refuse
or refit the child either. Neither does a platform without the counters:
non-Windows platforms report `status: "unsupported"`, and NVML is a possible
later addition.

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

| code | log line (substring) |
|---|---|
| `kv_kernel_f16_fallback` | `no FlashAttention vector kernel compiled for K/V types q8_0-q5_1, converting K and V to f16 instead (slow)`; `details.k_type`/`v_type` |
| `mtp_tensors_ignored` | `model has unused tensor blk.64.nextn.… -- ignoring`; folded, with `details.layer`, `tensors`, `ignored_bytes` |
| `partial_gpu_offload` | `offloaded N/M layers to GPU` with N < M |
| `cpu_buffer_fallback` | `cannot be used with preferred buffer type …, using CPU instead` |
| `no_gpu_device` | `no usable GPU found` |
| `gpu_init_failed` | `failed to initialize CUDA` |

The first two appear exactly like this in the 2026-09-29 logs. The last four
are taken from llama.cpp's source strings; the benchmark runs offloaded every
layer, so those lines did not appear.

## Configuration

These `llamaserver` JSON keys apply to spawn mode only. Each value shown is
the default.

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
    "fit_max_attempts": 4
  }
}
```

In C++, set `LlamaServerBackendOptions::spill_guard`
(`LlamaServerSpillGuardOptions`, with thresholds in bytes) and
`LlamaServerBackendOptions::diagnostics` (`LlamaServerDiagnosticsOptions`).
The status comes from `Backend::runtime_status()`. It is backend-neutral:
other backends return `std::nullopt` by default.

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
