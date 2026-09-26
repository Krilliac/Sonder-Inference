# Roadmap

## Phase 0 — research + contract

- [x] seed research catalog
- [x] seed architecture
- [x] define scheduler/KV/backend direction
- [x] define Observatory contract
- [ ] define C/C++/Rust API boundary
- [ ] choose implementation language for engine core
- [ ] establish benchmark harness
- [ ] snapshot Ollama baseline
- [ ] define first supported model/quantization matrix

## Phase 1 — direct native inference path

- [ ] Engine / Device / Model / Session abstractions
- [ ] llama.cpp/GGML backend
- [ ] tokenizer + prompt processing
- [ ] streaming decode
- [ ] sampling parity
- [ ] cancellation
- [ ] model load/unload
- [ ] telemetry
- [ ] side-by-side Ollama adapter

Gate: correctness + reliability parity for selected models.

## Phase 2 — Sonder-owned cache and scheduler

- [ ] block/page logical KV manager
- [ ] continuous batching
- [ ] chunked prefill
- [ ] prefix reuse
- [ ] session fork/share
- [ ] cache pressure/eviction
- [ ] workload priorities
- [ ] preemption
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
