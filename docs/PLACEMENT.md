# Model placement across memory tiers

Status: **step 1 implemented** (tensor placement overrides in the llama.cpp
backend). Steps 2–4 are design only.

## Why

VRAM is the scarcest resource on a Sonder host. The workstation has 16 GB on an
RTX 5070 Ti, and Node1 has a Radeon 780M sharing 48 GB of system RAM. The
common way to run a model larger than VRAM is to offload whole *layers* to the
CPU. That is expensive for every token, because every token passes through
every layer.

Mixture-of-Experts (MoE) models change the trade-off. Each token activates only
a few experts per layer, and the experts hold most of the weights. The
attention, router and shared weights are small, and every token uses them, so
they belong in VRAM. The expert weights are large and each token touches only a
few of them, so they can live in system RAM (and, later, on NVMe) with
bounded cost.

The same idea applies to models that are not too big for VRAM. Moving experts
out frees VRAM for more context (KV cache) or for a second resident model.

## Step 1 — tensor placement overrides (implemented)

The llama.cpp backend accepts ordered `(regex, device)` overrides, applied at
load time through llama.cpp's `tensor_buft_overrides`. Tensor names matching a
pattern are allocated on that device. The first match wins, and everything else
follows `--gpu-layers`.

```sh
# All layers on the GPU, MoE expert weights in system RAM.
sonder-infer serve --backend llamacpp --model-dir D:/models --model qwen3-30b-a3b.gguf \
    --gpu-layers -1 --moe-experts cpu

# The general form: any regex, any llama.cpp device ("cpu", "Vulkan0", "CUDA0").
sonder-infer serve ... --tensor-override "blk\.(3[0-9]|4[0-7])\.ffn_.*_exps=cpu"
```

- **`--moe-experts cpu`** expands to the pattern `\.ffn_(up|down|gate)_(ch|)exps` on `cpu`. This matches llama.cpp's own `--cpu-moe` choice: routed experts move to RAM, while the router (`ffn_gate_inp`) and the shared experts (`*_shexp`) stay on the GPU.
- **Validation happens before any file access.** The pattern must be a valid regex and the device must exist in this build; a bad flag fails at startup with `invalid_argument`. The number of overrides is capped by `llama_max_tensor_buft_overrides()`.
- **Library users** set `LlamaCppBackendOptions::tensor_overrides`. At the wrapper level the field is `LoadOptions::tensor_overrides`.

### Measurements

See the "Measured" section at the end of this file. Numbers are recorded only
from real runs.

## KV cache types, flash attention and micro-batch (implemented)

The KV cache competes with weights for VRAM, and its size grows linearly with
context. The llama.cpp backend exposes llama.cpp's own context knobs so an
operator can trade a little accuracy for a lot of KV memory. This is plumbing
only: with no flags, `serve` passes exactly what it passed before (f16 K and V,
flash attention `auto`, micro-batch = llama.cpp's 512 capped at the batch size).

| `serve` flag | `LlamaCppBackendOptions` | Values | Default |
|---|---|---|---|
| `--cache-type-k T` | `kv_cache_type_k` | `f16`, `f32`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl` | `f16` |
| `--cache-type-v T` | `kv_cache_type_v` | same | `f16` |
| `--flash-attn MODE` | `flash_attention` | `auto`, `on`, `off` | `auto` |
| `--batch-size N` | `batch_size` | tokens per `llama_decode` during prefill | 512 |
| `--ubatch-size N` | `ubatch_size` | physical micro-batch, `<= --batch-size` | llama.cpp's 512, capped at the batch size |

Bytes per cached value, from ggml's block layouts (32 values per block), and
the effect on the 384 MiB f16 KV cache measured below (4096 context):

| Type | Bytes/value | vs f16 | 384 MiB f16 becomes |
|---|---|---|---|
| `f32` | 4 | 2× | 768 MiB |
| `f16`, `bf16` | 2 | 1× | 384 MiB |
| `q8_0` | 1.0625 | ≈ ½ | ≈ 204 MiB |
| `q5_1` | 0.75 | ≈ 0.38 | ≈ 144 MiB |
| `q5_0` | 0.6875 | ≈ 0.34 | ≈ 132 MiB |
| `q4_1` | 0.625 | ≈ 0.31 | ≈ 120 MiB |
| `q4_0`, `iq4_nl` | 0.5625 | ≈ ¼ | ≈ 108 MiB |

K and V are sized independently, so `--cache-type-k q8_0` alone saves about a
quarter of the total. The table is arithmetic from the formats, not a
measurement; quality effects are model dependent and not yet measured here.
One data point (Node1, CPU, a 491 MB GGUF, 512 context, greedy, 16 tokens,
2026-09-30): `q8_0` K and V produced the same text as `f16`; `q4_0` K and V
with flash attention on degenerated into a repeated token. Treat 4-bit V
caches as something to evaluate per model, not a free default.

Rules, checked at startup (`invalid_argument`, before any model file is read):

- **A quantized V cache needs flash attention.** llama.cpp `b11195`
  (`llama_init_from_model`) switches `auto` to enabled for a quantized V
  cache and refuses one with flash attention disabled, so Sonder rejects
  `--cache-type-v q8_0 --flash-attn off` up front. A quantized K cache alone
  works without flash attention.
- **`--ubatch-size` must not exceed `--batch-size`.**
- Unknown type or mode names are rejected (case-insensitive; `on`/`enabled`
  and `off`/`disabled` are synonyms).
- **These five flags apply to `--backend llamacpp` only.** Any other backend
  would run without them, so `serve` exits with status 2 and
  `option --X applies only to --backend llamacpp; for llamaserver set it in the JSON args`.
  A spawned llama-server takes `--cache-type-k` and the rest in its
  `--llamaserver-config` `args`, or from a
  [launch profile](integration/launch-profiles.md)'s typed fields; with
  `--profile` the five flags are refused because the profile sets them.

Some limits depend on the model and are left to llama.cpp, which then fails
the load: MLA models (DeepSeek-style) need identical K and V types, a
quantized type's block size must divide the head size, and Grok models force
flash attention off (so a quantized V cache cannot load for them).

```sh
# Half the KV bytes, flash attention decided by llama.cpp (on for quantized V).
sonder-infer serve --backend llamacpp ... --cache-type-k q8_0 --cache-type-v q8_0
```

## Step 2 — per-model placement profiles (design)

Today the operator picks overrides per `serve` invocation. [Launch profiles](integration/launch-profiles.md)
now read the GGUF header for a VRAM fit estimate (weights of the offloaded
blocks, hybrid-aware KV bytes per token with sliding-window layers at their
window-sized cache, recurrent state, compute), but do not choose a
placement. The next step is a placement profile per model, derived from the
GGUF metadata:

| Field | Source |
|---|---|
| total weight bytes, per tensor class (attention, dense FFN, experts, shared, embeddings) | GGUF tensor table |
| experts per layer / experts used per token | `*.expert_count`, `*.expert_used_count` |
| KV bytes per token at a given context | attention-layer count × KV head dims × dtype (exclude recurrent layers) |
| recurrent-state checkpoint bytes | backend-reported state size per saved position, budgeted separately from token-indexed KV |

### Routed-expert header observations

`GgufModelInfo::routed_experts` is an optional typed pair of declared
`total` (`expert_count`) and `active` (`expert_used_count`) counts. The parser
requires exactly one string `general.architecture` with namespace syntax
`[A-Za-z0-9_][A-Za-z0-9_.-]*` (at most 256 bytes), then exactly one scalar GGUF
integer for each literal `<architecture>.expert_count` and
`<architecture>.expert_used_count` key. It requires
`0 < active <= total <= UINT32_MAX`. Missing, duplicate, wrongly typed or
out-of-range fields leave the pair unset. Arrays and the string metadata
view's array maxima are not scalar count evidence. Dots and case in the
namespace are literal; `qwen3.8` is not an alias for `qwen3_8`.

This observation is independent of `ModelArchitecture`, which describes the
attention/hybrid/recurrent cache and state axis. A previously unknown family
can supply valid count metadata without changing its existing cache
classification. An unset pair means unknown, not dense; model and family
names do not establish sparsity. Counts do not establish layer topology,
active parameter totals or measured routing frequencies.

The pair describes only the supplied GGUF header bytes. Neither
`parse_gguf_model_info` nor `read_gguf_model_info` binds those bytes to an
immutable full-model revision, and a header hash would not be a full-model
hash. The observation is not propagated to `ModelDescriptor`, the C ABI,
HTTP metadata or Runtime and is not consumed by placement, cache reuse or
routing policy. Those follow-ups require an exact same-artifact revision
binding and separate qualification. Backend execution capability bits and
Runtime task-quality evidence remain separate from physical model traits.
No model-quality, inference-speed or hardware recommendation follows from
synthetic header tests. The current VRAM arithmetic and manual tensor
placement rules remain unchanged.

A planner then chooses the cheapest placement that fits. It fills VRAM in
priority order: hot shared weights, then KV cache for the requested context,
then as many expert blocks as fit (whole layers' experts at a time). The rest
goes to RAM. The chosen placement should be published in the `model.loaded`
telemetry event, so the Observatory can show where each tensor class lives
rather than guessing.

## Step 3 — NVMe as a cold tier (design)

With `mmap`, expert tensors that are never paged in cost no RAM. For very large
MoE models (for example 150 GB with 10–15 GB active per token), cold experts can
stay on NVMe and be paged in on first use. PCIe 4 NVMe reads at about 5–7 GB/s,
roughly 10× slower than DDR5. That is only viable for experts the router rarely
selects, so this step needs expert-usage telemetry (routing histograms) first.
Without data it is guesswork.

## Step 4 — Sonder-wide device scheduling (design)

In the runtime, the Ollama pool already schedules whole requests across hosts
(for example, the reasoning tier served by Node1's 780M while the workstation
keeps its resident dense model). That pattern of distributed agents, rather
than tensors split across the network, remains the right one over 2.5 GbE. The
link is roughly 300 MB/s, against hundreds of GB/s for VRAM.

The extension is a placement-aware scheduler that knows, per device:

- compute (GPU/iGPU/NPU/CPU);
- memory (VRAM, shared iGPU memory, RAM, NVMe);
- links between devices (PCIe, the node link).

It then chooses both *which host* serves a request and *how* that host places
the model. Small always-on models (embeddings, reranker, router, speculative
draft) are natural tenants for an iGPU or NPU, which frees the discrete GPU for
the main model.

## Measured

**2026-09-27, workstation.** RTX 5070 Ti 16 GB, llama.cpp `b11195` with Vulkan,
MSVC RelWithDebInfo. Model: `qwen3-coder:30b-a3b` Q4_K_M, a 17.3 GiB GGUF (larger
than VRAM). Command:

`serve --backend llamacpp --device gpu:0 --gpu-layers -1 --context-length 4096 --moe-experts cpu`

| | Result |
|---|---|
| Layers offloaded | 49/49 |
| Weights in VRAM (`Vulkan0` model buffer) | **784 MiB** |
| KV cache in VRAM (4096 ctx) | 384 MiB |
| Expert weights in mapped system memory | 17,681 MiB |
| Output (temperature 0, 2 runs) | correct and identical |
| Decode, warm | about 3.6 tok/s (71 tokens in 19.9 s) — **not representative** |

**Placement works as designed.** A 17 GB MoE runs with about 1.2 GB of VRAM,
leaving room for a resident dense model or a much longer context.

**The speed figure is memory-starved.** Only about 3 GB of RAM was free (a WSL
test VM held 14 GB), so expert pages streamed from NVMe on every token. A
representative number needs about 18 GB of free RAM. The next measurement to
take is this same model with free RAM, comparing `--moe-experts cpu` against
layer offload (`--gpu-layers N` with no override) at equal VRAM.

**This also found a gap.** `serve` never passed a device to model loads, so a
llama.cpp model served by `sonder-infer serve` always ran CPU-only. `--device
gpu:0` now fixes that.

## Opt-in inspected-content foundation

The private server-module `server::detail::observe_gguf_artifact` entrypoint
binds the existing literal GGUF header interpretation and optional typed
routed-expert counts to a caller-trusted full SHA256 of the same inspected
stream. Success returns the matching digest and inspected byte length; missing
or ambiguous counts remain unknown. The existing header-only APIs still make
unbound header observations. See [the contract and local qualification](integration/gguf-artifact-observation.md).

The observer is opt-in and has no default serving or placement call site. It
does not populate model descriptors, backend identities, HTTP metadata,
telemetry or cache/routing policy. Its strict logical read/header allowances
and cooperative cancellation/deadline checks do not establish OS-prefetch,
heap/RSS or hard-latency bounds. A matching digest does not validate tensor
extents or model loadability, authenticate the expectation, or associate the
inspected content with bytes actually loaded by a backend.

Backend-loaded artifact association and separately qualified descriptor/HTTP
projection are still prerequisites for consuming this evidence in placement
or routing. Existing placement rules and residency controls remain unchanged;
physical expert counts do not imply model/task quality or execution speed.
