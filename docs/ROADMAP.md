# Roadmap

## Phase 0 — research + contract

- [x] seed research catalog
- [x] seed architecture
- [x] define scheduler/KV/backend direction
- [x] define Observatory contract
- [x] define C/C++/Rust API boundary (C++20 API + stable C ABI `sonder_inference.h`; Rust binds the C ABI; ADR-011)
- [x] choose implementation language for engine core (C++20 + CMake presets; ADR-011)
- [x] establish benchmark harness (`sonder-infer bench` and `sonder-bench`, `smoke`/`baseline`/`ternary_agent` corpora, results schema `sonder.inference.bench/1`, `--concurrency 1..64` parallel worker runs, and before/after GPU VRAM snapshots)
- [ ] snapshot Ollama baseline
- [ ] define first supported model/quantization matrix (supported GGUF layouts and hardware presets are documented in `docs/GPU_SUPPORT.md`; measured quality/performance matrix pending)

## Phase 1 — direct native inference path

- [x] Engine / Device / Model / Session abstractions (CPU + cross-platform GPU discovery via DXGI/NVML/ROCm/DRM/sysfs with UMA detection, `Engine::gpu_devices()`, live GPU `device.memory.sample` telemetry, surfaced on `sonder-infer devices`, `sonder-infer node`, `/v1/sonder/health`, and `/v1/sonder/node`)
- [x] llama.cpp/GGML backend (optional build, `SONDER_WITH_LLAMA_CPP=ON`, GPU presets `windows-cuda-rtx` / `linux-cuda-rtx` / `linux-rocm-xt` / `windows-vulkan-xt` / `linux-vulkan-xt`, multi-GPU placement, single pinned archive in `cmake/llamacpp-pin.json` + `third_party/ggml-patches/`, and offline `SONDER_LLAMACPP_URL` mirror support; license approved 2026-09-26; see `docs/GPU_SUPPORT.md`)
- [ ] tokenizer + prompt processing
- [x] streaming decode (session API; via mock and Ollama backends; native backend pending)
- [ ] sampling parity (sampling config + validation done; Sonder sampler applied for `token_logits` backends; new fields in `feat/sampling-config`; parity checks need a native backend)
- [x] cancellation (cooperative tokens; session cancel/close; Ctrl-C in CLI; cancel latency reported)
- [ ] model load/unload (engine registry + telemetry done; native residency pending)
- [x] telemetry (Observatory envelope v1 JSONL; lifecycle, request, decode, token, CPU/GPU `device.memory.sample`, scheduler, KV + `kv.checkpoint.saved`/`restored`, sampling, drop accounting; see `docs/TELEMETRY.md`)
- [x] side-by-side Ollama adapter (loopback HTTP, streaming, cancellation)
- [x] local HTTP API for Sonder Runtime (module `src/server`: `serve_main()` and `Server`; OpenAI-compatible chat subset plus Anthropic Messages `POST /v1/messages` through `Session::chat`, health with live GPU inventory, models, backend identity, `GET /v1/sonder/node`, `POST /v1/sonder/plan`, and multi-profile serving; ADR-020 proposed, `docs/SERVER.md`)
- [x] request priority classes and bounded admission (interactive/subagent/background FIFO ordering, per-class concurrency and queue caps, deadlines, cancellation propagation, and additive telemetry; see `docs/SERVER.md` and `docs/TELEMETRY.md`)
- [x] `sonder-infer serve` command-line dispatch (`src/cli/sonder_infer_commands.hpp`; exercised in CI by the Linux smoke step, which starts `serve --backend mock --port 0 --ready-file` and drains it on SIGINT)
- [x] live telemetry transport: SSE and NDJSON with discovery, resume and per-subscriber backpressure (`docs/TELEMETRY.md`)
- [x] additive text chat C ABI and Python Session.chat (versioned messages, 1–1,024 count bound; `docs/integration/cabi-chat.md`)
- [x] bounded Python structured chat iterator (`Session.chat_stream`; shared 64-chunk delivery/cancellation path; `docs/integration/python-chat-stream.md`)
- [x] additive C ABI/Python session correlation IDs (`docs/integration/cabi-session-metadata.md`)
- [x] additive C ABI/Python per-request parent lineage (four methods, reused-session isolation; `docs/integration/cabi-request-parent.md`)
- [x] C ABI v1 GPU inventory (`sonder_gpu_info`, `sonder_gpu_device_count`, `sonder_gpu_device_info`), ADR-020 request scheduling metadata (`sonder_workload_class`, appended `workload_class`/`priority` on `sonder_request_metadata`), and live telemetry callback sink (`sonder_engine_set_telemetry_callback`), with full Python parity (`GpuInfo`, `list_gpu_devices`, `WorkloadClass`, `RequestMetadata`, `Engine.set_telemetry_callback`)
- [x] backend identity for `llamacpp` via real GGUF hashing (streaming 64 KiB SHA-256 `model_digest`, single-pass canonical 12-key `tokenizer.ggml.*` `tokenizer_digest`, raw `tokenizer.chat_template` `template_digest`, quantization, GPU+CPU hardware, bounded 16-entry `(path, size, mtime)` LRU cache; honestly returns `null` with `reason` when required metadata is absent)
- [ ] backend identity for Ollama (Ollama API does not expose a measurable tokenizer digest; stays `null` by honesty contract)

Gate: correctness + reliability parity for selected models.

## Phase 2 — Sonder-owned cache and scheduler

- [x] block/page logical KV manager (`src/cache`, PR #4; driven by the engine request runtime, ADR-016)
- [x] continuous batching (policy: `src/scheduler`, PR #2; wired into Engine/Session)
- [x] chunked prefill (policy: `src/scheduler`; wired, logical KV accounting)
- [x] prefix reuse (logical, fingerprinted: `src/cache`; shared across concurrent requests by the engine; backend KV reuse pending)
- [x] architecture-aware logical prefix policy: attention-only, hybrid, and recurrent descriptors; hybrid/recurrent reuse requires a saved state checkpoint (`docs/KV_CACHE.md`)
- [x] backend recurrent-state capture/restore and engine checkpoint registration (`Capability::recurrent_checkpoint`, `BackendModel::capture_recurrent_state`/`restore_recurrent_state`, `llama.cpp` `llama_state_seq_get_size`/`get_data`/`set_data` with 512 MiB cap, opt-in deterministic mock hybrid support, bounded `SchedulingRuntime` checkpoint blob store wired to `KvCacheManager::save_checkpoint()` and `Session` with `kv.checkpoint.saved`/`restored` telemetry; live real-model hybrid quality and speedup benchmarking pending)
- [x] upgrade the optional llama.cpp pin beyond b11195: single pinned archive is `v1.0.0` in `cmake/llamacpp-pin.json` with configure-time `third_party/ggml-patches/` verification and `expected_features` enforcement (external fork/checkout overrides retired; offline builds use `SONDER_LLAMACPP_URL`; a GPU run with an `f8_e4m3` KV cache and the stories15M integration cases are still unverified at this pin; see `docs/KV_CACHE.md` and `docs/GPU_SUPPORT.md`)
- [x] session fork/share (logical fork + copy-on-write in `src/cache`; session API wiring pending)
- [x] cache pressure/eviction (watermarks, priority-aware/LRU eviction in `src/cache`)
- [x] workload priorities (7 classes, aging, starvation guard in `src/scheduler`)
- [x] preemption (policy; recompute vs swap decision, swap is a stub)
- [ ] adapter lifecycle

Gate: measurable win on agent fan-out/long-context workloads without interactive-latency regression.

## Phase 3 — speculation + memory hierarchy

- [x] external backend speculation via [`llamaserver`](integration/llama-server.md)
  (passes MTP/DFlash/n-gram configuration to llama-server; reports upstream
  draft acceptance and cache reuse; Sonder-owned adaptive policy remains below)
- [x] llamaserver VRAM-spill guard (Windows PDH + Linux `amdgpu-fdinfo` and NVML
  per-process counters; warn / refuse / bounded `auto_fit` context step-down;
  Ryzen APU `"uma_mode": true|false|"auto"`), backend-aware KV cache pairing
  check, and CUDA/ROCm/HIP/Vulkan/ternary log diagnostics, surfaced additively
  on health, `/v1/models` and telemetry ([vram-spill](integration/vram-spill.md),
  [GPU_SUPPORT.md](GPU_SUPPORT.md))
- [x] llamaserver slot prefix warm-up after readiness and supervised restart
  (bounded, per-slot, cancellable; `docs/integration/llama-server.md`)
- [x] adaptive speculation policy (`AdaptiveSpeculationPolicy` wired live into
  `LlamaServerBackend`: per-workload rolling acceptance across `general`,
  `code_edit`, `tool_json`, `prose`, and `reasoning`, native `"speculative.n_max"`
  injection, hysteresis step-up/down, pause + monotonic `probe_id` cooldown
  re-probe with RAII cancel guard, opt-in JSON `"adaptive_speculation"` config,
  and `runtime_status()->diagnostics["adaptive_speculation"]`)
- [x] `sonder-infer tune`: bounded llama-server calibration (Windows PDH, Linux
  `amdgpu-fdinfo`, and Linux NVML counters), per-profile context margin,
  `flash_attn` grid axis, `draft-mtp` and standalone `ngram-mod` calibration,
  and spawn-config output with two recommendations
  ([usage and qualification limits](integration/tune.md)); live performance validation pending
- [ ] CPU KV spill
- [ ] cache quantization where supported (manual `llamacpp` and `llamaserver` `f8_e4m3`/`q2_0`/`pq2_0m` options are supported; automatic quality-validated policy remains pending)
- [x] tensor placement overrides (MoE experts in RAM; `docs/PLACEMENT.md` step 1)
- [x] llama.cpp KV cache types (K and V separately), flash attention mode and micro-batch size as backend/`serve` options; defaults unchanged, no automatic policy yet (`docs/PLACEMENT.md`, "KV cache types")
- [x] typed per-model [launch profiles](integration/launch-profiles.md): one
  JSON schema drives llama-server argv and the direct llamacpp backend
  (unsupported fields rejected), default sampling per model, `/v1/models`
  profile metadata, and a GGUF-header VRAM fit check (hybrid-aware KV and
  MTP recurrent snapshots; +1.1 % vs the measured Qwen3.8-27B Q3_K_XL 100k
  profile, −1.1 % vs the live 65k MTP profile) that refuses overcommit
- [x] optional typed GGUF-header routed-expert counts, independent of cache
  architecture (`GgufModelInfo::routed_experts`; literal scalar metadata only,
  unknown stays unset; header-byte observation, no artifact revision binding
  or Runtime routing consumption; see `docs/PLACEMENT.md`)
- [x] several launch profiles behind one `serve` (repeatable `--profile` /
  `--all-profiles` with aggregate co-residency VRAM validation, per-profile
  sampling defaults, `llamacpp` multi-model residency, and `llamaserver` attach
  / shared-GGUF spawn aliases)
- [ ] Sonder Runtime tier routing across multiple served profiles
- [x] per-model and multi-model co-residency placement planner from GGUF
  metadata (`docs/PLACEMENT.md` step 2; `plan_model_placement()`,
  `plan_coresident_models()`, repeatable `sonder-infer plan --model`, and
  `POST /v1/sonder/plan` covering RTX, AMD RX XT, Ryzen APU UMA, multi-GPU
  split, MoE `--n-cpu-moe`, ternary/PrismML Hadamard detection, and speculative
  decoding; see `docs/GPU_SUPPORT.md`)
- [ ] optional disk tier
- [ ] model residency manager
- [ ] warm/cold adapter/model policy

## Phase 4 — heterogeneous nodes

- [x] node capability inventory (`sonder-infer node [--json]` and `GET /v1/sonder/node`, schema `sonder.inference.node/1`, reporting host memory, CPU, GPUs, backends, residency, profiles, and feature capabilities; advertised in `/.well-known/sonder-telemetry`)
- [ ] whole-request placement (Runtime-owned: the Runtime places requests across N SI endpoints and SI is a worker; see ADR-021 and `docs/integration/federation-worker-contract.md`)
- [ ] remote workers (deferred; SI-to-SI workers are not planned, `DeviceKind::remote` is reserved for the RPC capacity tier)
- [ ] transfer-cost model
- [ ] remote cache movement where justified
- [ ] background workload offload
- [ ] fault/reconnect semantics

## Phase 5 — advanced execution

Research/benchmark before committing:
- [ ] P/D separation
- [ ] MoE hot/cold expert placement
- [ ] tensor/expert parallel options
- [ ] NPU execution
- [ ] specialized NVIDIA backend
- [ ] compiler-backed backends

## Phase 6 — selective native acceleration

For each candidate operator:
- [ ] profile bottleneck
- [ ] existing-library evaluation
- [ ] prototype
- [ ] correctness differential
- [ ] representative benchmark
- [ ] fallback
- [ ] maintenance-cost review

No "rewrite it ourselves" milestone exists without evidence.

## Ordered architecture follow-ups

After the bounded routed-expert header observation slice, the subsequent
architecture work proceeds through MoE metadata plus placement, real backend
recurrent checkpoints, embedding/reranking, architecture-aware routing,
residency, then Runtime MoT orchestration. These are follow-ups, not additional
capabilities inferred from the count pair. Placement and routing need an exact
model/revision binding; Runtime task-quality evidence is separate from
physical execution traits and backend capability advertisement. MoT across
models is separate from internal MoE routing. Shared-chat hardware, model,
license and provider-quality recommendations require independent evidence.

## Inspected-content foundation and remaining boundaries

- [x] Private opt-in full-artifact inspected-content observation: the same
  caller-owned stream supplies native literal header facts and the matching
  trusted full SHA256/byte length; optional routed-expert counts retain their
  unknown semantics. [Contract and local qualification](integration/gguf-artifact-observation.md).
- [ ] Associate inspected content with bytes actually loaded by each backend.
- [ ] Qualify descriptor/HTTP projection and any placement, routing or cache
  consumer separately; no default policy or public API is added by this slice.

The existing header-only observation milestone remains distinct and unbound.
This foundation does not establish tensor-extent validity, model loadability,
filesystem immutability, digest provenance, backend identity completeness or
model/task quality. The broader placement, recurrent-checkpoint,
embedding/reranking, residency and Runtime orchestration work remains pending.
