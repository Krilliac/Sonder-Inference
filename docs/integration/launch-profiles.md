# Launch profiles

A launch profile is a named, typed description of how to serve one model:
the GGUF file, the context and KV cache setup, offload, batching, prompt
cache, speculative decoding, chat template handling, and default sampling.
`sonder-infer serve --profiles FILE --profile NAME` serves it. Every setting
is a checked field, not a free-form argument string, so a typo or an
impossible combination fails at startup instead of silently changing what
runs.

The same profile works for both serving backends:

- **`llamaserver`** (the [external llama-server backend](llama-server.md) in
  spawn mode). The fields become a llama-server argument vector. The flag
  names were checked against the installed `llama-server --help` (Ollama's
  bundled build, `version: 0.4.1-dev (commit 161755f29)`). The supervisor
  still appends its own `--host 127.0.0.1 --port <free port>`.
- **`llamacpp`** (the direct, in-process [llama.cpp backend](llamacpp.md)).
  Fields map onto the backend's own options. A field the direct backend
  cannot honour is rejected with
  `'<field>' is not supported by the llamacpp backend; use llamaserver`.
  It is never silently dropped.

Nothing changes for a `serve` without `--profile`.

## The measured 100k profile

This profile is measured on the owner's RTX 5070 Ti 16 GB (Windows, 16,303
MiB usable). It runs Qwen3.8-27B UD-Q3_K_XL (12.24 GiB file):

```json
{
  "profiles": [
    {
      "name": "qwen3.8-27b-q3-100k",
      "backend": "llamaserver",
      "model": "D:/models/Qwen3.8-27B-UD-Q3_K_XL.gguf",
      "n_gpu_layers": 999,
      "ctx_size": 100096,
      "parallel": 1,
      "kv_unified": false,
      "batch_size": 2048,
      "ubatch_size": 512,
      "flash_attn": "on",
      "cache_type_k": "q8_0",
      "cache_type_v": "q5_1",
      "fit": false,
      "cache_ram_mib": 4096,
      "ctx_checkpoints": 4,
      "checkpoint_min_step": 8192,
      "context_shift": false,
      "jinja": true,
      "reasoning_format": "deepseek",
      "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
      "vram_budget_mib": 15600
    }
  ]
}
```

```powershell
sonder-infer serve --profiles profiles.json --profile qwen3.8-27b-q3-100k `
  --llamaserver-executable "$env:LOCALAPPDATA/Programs/Ollama/lib/ollama/llama-server.exe"
```

It spawns llama-server with exactly the measured command line:

```text
--model D:/models/Qwen3.8-27B-UD-Q3_K_XL.gguf --alias qwen3.8-27b-q3-100k
--ctx-size 100096 --n-gpu-layers 999 --batch-size 2048 --ubatch-size 512 --parallel 1
--no-kv-unified --flash-attn on --cache-type-k q8_0 --cache-type-v q5_1
--ctx-checkpoints 4 --checkpoint-min-step 8192 --cache-ram 4096 --no-context-shift
--fit off --jinja --reasoning-format deepseek
```

The model is served as `qwen3.8-27b-q3-100k` (and as `default`).

Measured with this setup:

| | |
|---|---|
| VRAM (nvidia-smi) | 15.9 of 16.3 GB in use; the llama-server process held 15,266 MiB dedicated and 388 MiB shared (a later probe with 774 MiB used by the desktop) |
| decode, short prompt | 30 tok/s |
| decode at 90k context | 16 tok/s |
| prefill | 814 tok/s |
| repeated 90k prompt | 89,785 tokens reused from the prompt cache in 1.0 s |

`vram_budget_mib` is 15,600 because this configuration needs more than the
default budget (device memory minus 1.5 GiB = 14,767 MiB). It only fits
while the desktop uses less than about 0.8 GB of VRAM. The default budget
would refuse it; see [VRAM fit check](#vram-fit-check).

**This configuration is at the spill edge.** The 388 MiB of shared memory
above is more than the llamaserver spill guard's 256 MiB threshold, so the
guard reports this run as spilled. The [VRAM spill](vram-spill.md) benchmark
ran the same q8_0/q5_1 setup at 100,096 and measured 15.9 tok/s short
decode, against 50.0 tok/s for q5_1/q5_1 at the same context, which did not
spill. The spawned llama-server also gets a `kv_type_mismatch` warning from
the KV pairing check, because q8_0/q5_1 has no FlashAttention vector kernel
in a default CUDA build and runs converted to f16. See
[Spill guard and KV pairing check](#spill-guard-and-kv-pairing-check) for the
variant that fits and the budget that matches the measured spill line.

## The community IQ4_XS profile

The community configuration for the UD-IQ4_XS file (14.25 GB) adds a vision
projector kept in system RAM, plus Qwen's recommended sampling:

```json
{
  "name": "qwen3.8-27b-iq4xs-vision",
  "backend": "llamaserver",
  "model": "D:/models/Qwen3.8-27B-UD-IQ4_XS.gguf",
  "mmproj": "D:/models/mmproj-F16.gguf",
  "mmproj_offload": false,
  "image_max_tokens": 2400,
  "n_gpu_layers": 999,
  "ctx_size": 65536,
  "parallel": 1,
  "kv_unified": false,
  "flash_attn": "on",
  "cache_type_k": "q8_0",
  "cache_type_v": "q5_1",
  "fit": false,
  "jinja": true,
  "reasoning_format": "deepseek",
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
  "vram_budget_mib": 15600
}
```

On 16 GB, IQ4_XS fits 65,536 tokens of q8_0/q5_1 KV. At that size it
measured 36 tok/s short, 1,112 tok/s prefill, 21 tok/s decode at 55k, with
0.33 GB shared (no spill). At 73,728, 81,920 and 100,096 it spilled 0.71,
0.95 and 2.1 GB into shared system memory. The driver does this silently
under `--fit off`. That run was 2 to 15 times slower, and decode collapsed
to about 1 tok/s at 72k. For long context, keep Q3_K_XL at 100k.

## Fields

| Field | llama-server flag | llamacpp backend | Values |
|---|---|---|---|
| `name` | `--alias` | served id | `[A-Za-z0-9._:-]`, 1-128 chars, not `default` |
| `backend` | | | `llamaserver` or `llamacpp` |
| `model` | `--model` | GGUF path passed to `load_model` | path |
| `mmproj` | `--mmproj` | rejected | path |
| `mmproj_offload` | `--mmproj-offload` / `--no-mmproj-offload` | rejected | bool; needs `mmproj` |
| `image_max_tokens` | `--image-max-tokens` | rejected | 1-65536; needs `mmproj` |
| `ctx_size` | `--ctx-size` | `context_length` (at most 4194304) | 0 (model default) to 16777216 |
| `n_gpu_layers` | `--n-gpu-layers` (`-1` is `all`) | `gpu_layers`; sets `--device gpu:0` when none is given, refuses a cpu device | -1 to 100000 |
| `batch_size` | `--batch-size` | `batch_size` * | 1 to 4194304 |
| `ubatch_size` | `--ubatch-size` | `ubatch_size` * | at most `batch_size` (upstream default: 2048 for llama-server, 512 for llamacpp) |
| `parallel` | `--parallel` | rejected | 1-256 |
| `kv_unified` | `--kv-unified` / `--no-kv-unified` | rejected | bool |
| `flash_attn` | `--flash-attn` | `flash_attention` * | `on`, `off`, `auto` (also `enabled`, `disabled`) |
| `cache_type_k`, `cache_type_v` | `--cache-type-k`, `--cache-type-v` | `kv_cache_type_k`, `kv_cache_type_v` * | `f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` |
| `ctx_checkpoints` | `--ctx-checkpoints` | rejected | 0-1024 |
| `checkpoint_min_step` | `--checkpoint-min-step` | rejected | 0 to 16777216 |
| `cache_ram_mib` | `--cache-ram` | rejected | -1 (no limit), 0 (off), up to 16777216 |
| `context_shift` | `--context-shift` / `--no-context-shift` | rejected | bool |
| `fit` | `--fit on/off` | rejected | bool |
| `speculative.types` | `--spec-type a,b` | rejected | `none draft-simple draft-eagle3 draft-mtp draft-dflash draft-dspark ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache` |
| `speculative.draft_n_max` | `--spec-draft-n-max` | rejected | 1-64 |
| `speculative.draft_model` | `--spec-draft-model` | rejected | path |
| `threads`, `threads_batch` | `--threads`, `--threads-batch` | rejected | -1 to 1024 |
| `jinja` | `--jinja` / `--no-jinja` | rejected | bool |
| `reasoning_format` | `--reasoning-format` | rejected | `none deepseek deepseek-legacy auto` |
| `sampling.*` | (per request) | (per request) | `temperature top_p top_k min_p presence_penalty repeat_penalty` |
| `extra_args` | appended verbatim | rejected | see below |
| `vram_budget_mib` | | | fit-check budget, MiB |

\* The llamacpp column uses the option names and values of the direct
backend's KV cache lane (`feat/kv-cache-types-flash-attn-ubatch`:
`LlamaCppBackendOptions::kv_cache_type_k/_v`, `flash_attention`,
`ubatch_size`, and `BackendSetup::llamacpp_*`). Until that lane is merged,
`BackendSetup` has no such fields, and a llamacpp profile that sets them is
rejected with the "not supported by the llamacpp backend" error. The check
is a compile-time test for the fields
(`kLlamaCppContextOptionsAvailable`), so the mapping switches on by itself
when they appear.

Rules checked at load time (`invalid_argument`, before anything starts):

- Unknown keys, wrong types and out-of-range values are errors.
- Profile names are unique within a file (at most 64 profiles, 1 MiB).
- **A quantized V cache needs flash attention.** `cache_type_v` other than
  f32/f16/bf16 together with `flash_attn: off` is refused, as llama.cpp
  refuses it. `auto` is allowed, because llama.cpp then turns flash
  attention on.
- `ubatch_size` must not exceed `batch_size` (or the upstream default).
- `mmproj_offload` and `image_max_tokens` need `mmproj`.
- Speculative types must be known and unique. `none` cannot be combined
  with other types.
- Sampling defaults must pass the same range checks as request values.
- `extra_args` may not contain `--host`, `--port` (or `--host_*`/`--port_*`),
  `--`, `--api-key`, `--api-key-file`, `--ssl-key-file`, `--ssl-cert-file`,
  download flags (`-hf`, `--hf-repo`, `--hf-file`, `--hf-token`,
  `--model-url`, `--docker-repo`, `--mmproj-url`, `--hf-repo-draft`,
  `--spec-draft-hf`), llama-server's built-in model presets (every
  `--*-default` and `--*-spec` flag, such as `--gpt-oss-20b-default`,
  `--fim-qwen-7b-spec` and `--spec-default`: they replace the profile's model
  and settings and can download weights), `--log-file`,
  `--log-prompts-dir` and `-lcd`/`--lookup-cache-dynamic` (a shared profile
  must not record prompts to disk; `--slot-save-path` stays allowed, see
  llama-server.md),
  or `--models-dir`/`--models-preset`. They may not
  turn on llama-server's agent tools, MCP or local file serving either:
  `--tools` (which includes `exec_shell_command` and `write_file`),
  `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`,
  `-ag`/`--agent`, `--ui-mcp-proxy`/`--webui-mcp-proxy`, `--path` and
  `--media-path`. Their `--no-*` forms are allowed. Profiles are meant to be
  copied from shared configurations, so a profile file must not be able to
  switch on shell execution in the child. (`--llamaserver-arg` without a
  profile is an operator flag and is not filtered this way.) The spawn-mode loopback
  rules still apply: the child only ever listens on 127.0.0.1. Any spelling
  of a flag that a typed field owns (for example `-c`, `--ctx-size=`,
  `--temp`, `-ngl`) is refused with the field to use instead.

With `--profile`, `serve` also refuses `--backend` or `--model` values that
differ from the profile, `--gpu-layers`, `--context-length`,
`--llamaserver-arg`, `--llamaserver-url`, attach mode, and `args` in a
`--llamaserver-config` file. Everything else, such as the executable,
timeouts, restart policy, TLS, `--device`, `--moe-experts` and
`--tensor-override`, is taken from the usual options.

## Default sampling

`sampling` values apply only when a request does not set that field.
`apply_sampling_defaults` sets the value and marks it explicit, using the
`explicit_fields` presence mechanism the llama-server and Ollama adapters already use. The upstream therefore receives it exactly as if
the caller had sent it, including `min_p: 0.0`, which equals a Sonder
default. A request's own `temperature` always wins. Models served without a
profile get no defaults, and their requests are unchanged.

## VRAM fit check

`serve --profile` reads the GGUF header (metadata and tensor table only,
about 11 MB for the Qwen files; no weights). It estimates the process's VRAM
and compares it with a budget:

- **budget** = the profile's `vram_budget_mib`, else `--vram-budget-mib`,
  else the largest adapter's dedicated memory (DXGI on Windows) minus 1,536
  MiB. Other platforms need an explicit budget.
- **weights** = tensor bytes of the blocks on the GPU. llama.cpp offloads the
  last `n_gpu_layers` blocks, and adds the output tensors when every block
  is offloaded (`-1`/`all`, or more layers than blocks). `token_embd` stays
  on the CPU. MTP blocks (`nextn_predict_layers`) count only when
  `speculative.types` has `draft-mtp`, because llama.cpp skips them
  otherwise. The mmproj and draft files count when offloaded.
- **KV** = sum over GPU attention layers of n_kv_heads × (key_length ×
  bytes(K) + value_length × bytes(V)) × ctx_seq × sequences. Bytes per value
  come from ggml's block layouts: f16 2, q8_0 1.0625, q5_1 0.75, q4_0 0.5625.
  ctx is padded to 256. With `kv_unified` (the default when `parallel` is
  unset) there is one pool: sequences = 1 and ctx_seq = ctx. Without it,
  sequences = `parallel` and ctx_seq = ctx / `parallel` padded to 256,
  because `--ctx-size` is the total and llama.cpp splits it into one stream
  per sequence (`n_ctx_seq` in `llama_context`). Either way the KV holds
  about ctx tokens in total; `parallel` does not multiply it.
  **Hybrid models count attention layers only.** They are found from
  per-block `head_count_kv` arrays, then `attn_k` tensors, then
  `full_attention_interval`, using the architecture classification in `model_architecture.hpp`.
- **recurrent state** = for each GPU block without attention in a hybrid or
  recurrent model: ((conv_kernel − 1) × (inner + 2 × groups × state) +
  inner × state) × 4 bytes × `parallel`.
- **compute** = n_vocab × ubatch × 4 (logits) + 4 × n_embd × ubatch × 4 +
  256 MiB runtime allowance.

`serve` logs the breakdown. It exits with status 2 when the estimate exceeds
the budget, unless `--allow-overcommit` is given. When llama-server sizes
the offload itself (`fit` not false and `n_gpu_layers` unset), the estimate
is only reported. A profile with `n_gpu_layers: 0`, or a llamacpp profile
on a cpu device, skips the check.

Validation against the measured 100k profile (estimator run on the real
Q3_K_XL header on Node1):

| Part | Estimate |
|---|---|
| weights (64 blocks + output; MTP block skipped) | 11,671 MiB |
| KV: 16 attention layers × 29,696 B/token × 100,096 | 2,835 MiB |
| recurrent state: 48 delta-net layers | 150 MiB |
| compute | 781 MiB |
| **total** | **15,437 MiB** |

Measured: 15,266 MiB dedicated to the process, so the estimate is **+1.1 %**
high. Counting the 388 MiB of shared memory too (15,654 MiB), it is 1.4 %
low. An earlier reading of "about 14.3 GB for the process" was the
nvidia-smi total minus about 1.6 GB of desktop use. Against that figure the
estimate is 8 % high. The per-process counter above is the better
reference. Without the hybrid rule (all 64 blocks as attention), KV alone
would be 11,339 MiB and the total about 23,790 MiB.

The only full-configuration datapoint so far is this one, so the compute
allowance is a heuristic, not a fitted model. The same estimator on the
IQ4_XS header gives:

| ctx (q8_0 K / q5_1 V, parallel 1) | Q3_K_XL estimate | IQ4_XS estimate | IQ4_XS measured (shared memory; about 0.35 GB is the no-spill baseline) |
|---|---|---|---|
| 65,536 | 14,458 MiB | 15,513 MiB | 0.33 GB, full speed |
| 73,728 | 14,690 MiB | 15,745 MiB | 0.71 GB, spilled |
| 81,920 | 14,922 MiB | 15,977 MiB | 0.95 GB, spilled |
| 100,096 | 15,437 MiB | 16,492 MiB | 2.1 GB, spilled |

The IQ4_XS runs had 0.67 to 1.2 GB of desktop VRAM in use, which leaves about
15,100 to 15,600 MiB. The estimate puts 65,536 inside that range and every
larger size above it, the same line the measurements drew. The measured spill
grew faster than the estimated overshoot. At 100,096 about 1.75 GB spilled
beyond the baseline, against a 0.9 to 1.4 GiB overshoot. So on this card,
treat the estimate as a lower bound near the limit. With
`vram_budget_mib: 15600`, the check accepts IQ4_XS at 65,536 and refuses it
at 73,728 and above.

The known limits:

- MLA (DeepSeek-style) and sliding-window attention are counted as full
  attention, which overestimates.
- Placement flags in `extra_args` (`-ot`, `--cpu-moe`, `--tensor-split`,
  and so on) are not modelled. The estimate says so.
- Vision encoder compute buffers and a draft model's KV cache are not
  included.
- Windows does not fail an overcommitted allocation under `--fit off`. It
  spills into shared memory and runs 2 to 15 times slower. That is why the
  check refuses by default.

## `/v1/models`

The served model's entry gains `sonder.profile`, and the list gains
`sonder.profiles` with every profile of the file (`served` marks the active
one). All existing fields are unchanged, and without `--profile` neither
field appears.

```json
{"object":"list","data":[{"id":"qwen3.8-27b-q3-100k","object":"model","owned_by":"sonder-inference",
  "sonder":{"backend":"llamaserver","default":true,"synthetic":false,
    "profile":{"name":"qwen3.8-27b-q3-100k","backend":"llamaserver","context_length":100096,"parallel":1,
      "cache_type_k":"q8_0","cache_type_v":"q5_1","flash_attn":"on",
      "capabilities":[],"upstream_capabilities":["tools"],
      "estimated_vram_mib":15437,"vram_budget_mib":15600,"served":true}}}],
 "sonder":{"api_version":1,"profiles":[{"name":"qwen3.8-27b-q3-100k","...":"...","served":true}]}}
```

`capabilities` lists only what a request to this Sonder endpoint can use:
`speculative` when speculative types other than `none` are set (it is
transparent to the caller). `upstream_capabilities` lists what the upstream
llama-server supports: `vision` when `mmproj` is set, `tools` for llamaserver
profiles with jinja on (llama-server's default), and `speculative`. Sonder's
`/v1/chat/completions` rejects `tools`, `tool_choice`, `functions` and
non-string message content (image parts) with `unsupported_parameter`, so
`vision` and `tools` are not in `capabilities` until Sonder forwards them. A
consumer that reads `capabilities` never builds a request Sonder refuses.
`estimated_vram_mib` is null when no estimate was possible, and
`estimate_error` then says why. The model's filesystem path is replaced by
`<model path>` there; the full message, with the path, goes to the
operator's log.

`context_length` is the profile's `ctx_size`, unless the backend reports
that the running process uses a different context. The llamaserver spill
guard's `auto_fit` policy relaunches llama-server with a smaller
`--ctx-size` when it spills. Then `context_length` (in `sonder.profile` and
in the served entry of `sonder.profiles`) is the running context, and
`configured_context_length` holds the profile's value. `estimated_vram_mib`
stays the estimate for the configured context. The backend's own
`sonder.runtime` object (context fit, GPU memory, warnings) sits beside
`sonder.profile` on the same entry, unchanged.

## Spill guard and KV pairing check

The spawned llama-server runs under the llamaserver backend's
[spill guard and KV pairing check](vram-spill.md). A profile does not change
either. Configure them in the `--llamaserver-config` file (`spill_guard`,
`kv_pairing_check`, `log_file`); only `args` is refused there with
`--profile`.

- **KV pairing.** A profile with different `cache_type_k` and
  `cache_type_v` (such as the measured q8_0/q5_1) gets a `kv_type_mismatch`
  warning in `sonder.runtime.warnings`, and q5_1/q5_1 gets
  `kv_type_no_vector_kernel`. Both are converted to f16 inside
  FlashAttention in a default CUDA build. The profile still loads; the
  warning is advice, not a refusal.
- **`auto_fit`.** It can shrink the context of a profile's llama-server at
  run time. `/v1/models` then reports the running context (see above). The
  fit check at startup still judged the configured context.
- **Spill threshold vs fit budget.** The guard calls a process spilled when
  its shared GPU memory exceeds the baseline plus 256 MiB. The fit check
  compares the estimate with dedicated memory only. The measured 100k run
  held 388 MiB shared, so by the guard's rule it spilled. Against dedicated
  plus shared (15,654 MiB) the estimate of 15,437 MiB is 1.4 % low.

The VRAM spill benchmark measured seven Q3_K_XL configurations with the
per-process counters. The estimates below apply this estimator to them.
Only the KV term depends on the context and the cache types, so each one is
the measured profile's 15,437 MiB with its KV term replaced.

| K/V | ctx | estimate | measured dedicated | shared | spilled |
|---|---|---|---|---|---|
| q5_1/q5_1 | 100,096 | 14,948 MiB | 14,966 MiB | 198 MiB | no |
| q8_0/q5_1 | 91,904 | 15,205 MiB | 15,184 MiB | 190 MiB | no |
| q8_0/q5_1 | 96,000 | 15,321 MiB | 15,222 MiB | 292 MiB | yes |
| q5_1/q5_1 | 116,480 | 15,332 MiB | 15,302 MiB | 342 MiB | yes |
| q8_0/q8_0 | 83,712 | 15,382 MiB | 15,320 MiB | 182 MiB | no |
| q8_0/q5_1 | 100,096 | 15,437 MiB | 15,266 MiB | 388 MiB | yes |
| q8_0/q8_0 | 87,808 | 15,518 MiB | 15,314 MiB | 348 MiB | yes |

Dedicated usage stops near 15.2-15.3 GB, and the overflow goes to shared
memory. Every spilled run has an estimate of 15,321 MiB or more. So on this
card, **`vram_budget_mib: 15300`** refuses every spilled configuration. It
also refuses q8_0/q8_0 at 83,712, which did not spill. The 15,600 in the
measured profile accepts three spilled configurations, including that
profile itself. For 100k context on 16 GB, the configuration that measured
clean is q5_1/q5_1:

```json
{"name": "qwen3.8-27b-q3-100k-q51", "...": "same as the measured profile",
 "cache_type_k": "q5_1", "cache_type_v": "q5_1", "vram_budget_mib": 15300}
```

It measured 50.0 tok/s short decode and 44.3 tok/s at an 8k prompt. It
still gets the `kv_type_no_vector_kernel` advice. The benchmark measured
little conversion cost at 8k, and did not measure it at long prompts.

## Trade-offs

- **KV type vs quality and VRAM.** Relative to f16, q8_0 halves the KV bytes
  and q5_1 cuts them to 0.375×. The K cache is more sensitive to
  quantization than V. The measured profile keeps K at q8_0 and puts V at
  q5_1: 1.8125 bytes per K+V value pair instead of 4, so 29,696 instead of
  65,536 bytes per token for this model. That is how 100k tokens fit next
  to 11.4 GiB of weights. Quality effects are model
  dependent and have not been measured here. This llama-server build warns
  that it has no FlashAttention vector kernel for the `q8_0-q5_1` pair and
  converts to f16 in that path (`GGML_CUDA_FA_QUANTS`). The measured decode
  speeds already include that cost. The KV pairing check reports it as
  `kv_type_mismatch`. q5_1/q5_1 at the same context uses 489 MiB less and
  did not spill; see [Spill guard and KV pairing
  check](#spill-guard-and-kv-pairing-check).
- **Checkpoints vs RAM.** Hybrid models cannot trim their recurrent state,
  so prefix reuse needs context checkpoints. `ctx_checkpoints` limits how
  many a slot keeps (llama-server's default is 32), and
  `checkpoint_min_step` spaces them. Each checkpoint holds the recurrent
  state plus the attention KV up to its position, in host RAM under the
  prompt cache (`cache_ram_mib`, default 8,192 MiB). The measured profile
  keeps 4 checkpoints at least 8,192 tokens apart within a 4 GiB cache. That
  was enough to reuse 89,785 of about 90k prompt tokens in 1.0 s.
- **MTP on 16 GB.** Qwen3.8's MTP block adds a 335 MiB block plus its KV
  layer. On this card, MTP pushed 7 to 11 % of the layers out of VRAM. In
  Ollama it was slower: 46 → 23 tok/s at n = 3. Keep `speculative` unset on
  16 GB unless a smaller context frees the room. The fit check counts the
  MTP block only when `draft-mtp` is set.
- **`fit: false`** makes the configuration exact and reproducible. It also
  lets Windows spill silently. Keep the budget honest, or leave `fit` on and
  `n_gpu_layers` unset to let llama-server shrink the offload itself.

## Binding Runtime tiers to profiles

Sonder Runtime chooses a model id per tier with
`SONDER_INFERENCE_TIER_MODELS` (see the Runtime's
`docs/architecture/sonder-inference-provider.md`). A profile's `name` is its
served model id, so a tier binds to a profile by name:

```powershell
$env:SONDER_MODEL_BACKEND = "sonder-inference"
$env:SONDER_INFERENCE_BASE_URL = "http://127.0.0.1:11437"
$env:SONDER_INFERENCE_TIER_MODELS = "general=qwen3.8-27b-q3-100k,fast=qwen3.8-27b-q3-100k"
```

The Runtime can read `sonder.profile` from `/v1/models` to see the context
length, cache types and capabilities behind each id. It can also check a
tier's context needs against `context_length` before it routes. One `serve`
hosts one profile today, because one spawned llama-server serves one model.
Tiers that need different profiles need separate `serve` instances. The
Runtime currently takes a single base URL. Multi-profile hosting is the
next step (see the roadmap).

## Tests

`sonder.server.*` in CTest covers the following, with no model, GPU or
llama-server needed:

- schema parsing and every validation rule
- argv golden tests (the measured profile and one that sets every toggle)
- direct-backend mapping and each rejection
- the GGUF reader on synthetic headers (hybrid by tensors and by interval,
  per-layer head counts, truncation)
- estimator arithmetic, including the hybrid rule, parallel and unified KV,
  MTP and partial offload
- sampling defaults reaching the backend
- the additive `/v1/models` fields, and their absence without a profile
- endpoint vs upstream capabilities, and a context reduced at run time
  (`auto_fit`) replacing `context_length` next to the backend's `runtime`
- `serve` refusals

Set `SONDER_TEST_GGUF` to a GGUF file, or to a copy of its header, to print
the estimate for a real model.
