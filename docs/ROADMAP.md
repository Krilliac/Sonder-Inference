# Roadmap

## Phase 0 — research + contract

- [x] seed research catalog
- [x] seed architecture
- [x] define scheduler/KV/backend direction
- [x] define Observatory contract
- [x] define C/C++/Rust API boundary (C++20 API + stable C ABI `sonder_inference.h`; Rust binds the C ABI; ADR-011)
- [x] choose implementation language for engine core (C++20 + CMake presets; ADR-011)
- [x] establish benchmark harness (skeleton: `sonder-infer bench`, smoke corpus, results schema `sonder.inference.bench/1`, concurrency 1 only)
- [ ] snapshot Ollama baseline
- [ ] define first supported model/quantization matrix

## Phase 1 — direct native inference path

- [x] Engine / Device / Model / Session abstractions (CPU inventory only; GPU discovery pending)
- [x] llama.cpp/GGML backend (optional build, `SONDER_WITH_LLAMA_CPP=ON`; license approved 2026-09-26)
- [ ] tokenizer + prompt processing
- [x] streaming decode (session API; via mock and Ollama backends; native backend pending)
- [ ] sampling parity (sampling config + validation done; parity checks need a native backend)
- [x] cancellation (cooperative tokens; session cancel/close; Ctrl-C in CLI; cancel latency reported)
- [ ] model load/unload (engine registry + telemetry done; native residency pending)
- [x] telemetry (Observatory envelope v1 JSONL; lifecycle, request, decode, token, device, drop accounting)
- [x] side-by-side Ollama adapter (loopback HTTP, streaming, cancellation)

Gate: correctness + reliability parity for selected models.

## Phase 2 — Sonder-owned cache and scheduler

- [x] block/page logical KV manager (`src/cache`, PR #4; not yet driven by the engine)
- [x] continuous batching (policy: `src/scheduler`, PR #2; engine wiring pending)
- [x] chunked prefill (policy: `src/scheduler`; engine wiring pending)
- [x] prefix reuse (logical, fingerprinted: `src/cache`; backend KV reuse pending)
- [x] session fork/share (logical fork + copy-on-write in `src/cache`; session API wiring pending)
- [x] cache pressure/eviction (watermarks, priority-aware/LRU eviction in `src/cache`)
- [x] workload priorities (7 classes, aging, starvation guard in `src/scheduler`)
- [x] preemption (policy; recompute vs swap decision, swap is a stub)
- [ ] adapter lifecycle

Gate: measurable win on agent fan-out/long-context workloads without interactive-latency regression.

## Phase 3 — speculation + memory hierarchy

- [ ] draft/ngram/backend speculation interfaces
- [ ] adaptive speculation policy
- [ ] CPU KV spill
- [ ] cache quantization where supported
- [ ] optional disk tier
- [ ] model residency manager
- [ ] warm/cold adapter/model policy

## Phase 4 — heterogeneous nodes

- [ ] node capability inventory
- [ ] whole-request placement
- [ ] remote workers
- [ ] transfer-cost model
- [ ] remote cache/session transport
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
