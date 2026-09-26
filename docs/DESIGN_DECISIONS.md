# Design Decisions

## ADR-001 — Build an inference runtime before a kernel ecosystem

**Decision:** Sonder owns scheduling, sessions, cache, device policy, model lifecycle, and telemetry first. Existing backends provide tensor execution.

**Reason:** this captures most of the Sonder-specific value early while avoiding years of architecture/quantization/kernel reimplementation.

## ADR-002 — Keep Ollama during migration

**Decision:** retain an Ollama adapter until the direct path clears compatibility/reliability gates.

**Reason:** rollout safety and A/B benchmarking.

## ADR-003 — Direct llama.cpp/GGML is the initial native execution candidate

**Decision:** start evaluation with direct embedding/control rather than building CUDA kernels first.

**Reason:** broad consumer hardware and model/quantization support, C/C++ embedding, GGUF ecosystem.

**Not permanent:** the backend interface remains independent.

## ADR-004 — KV/context is a first-class resource

**Decision:** Sonder owns logical KV allocation/reuse/eviction/accounting above backend-specific physical representation.

**Influences:** vLLM PagedAttention, LMCache, Mooncake, SGLang prefix reuse.

## ADR-005 — Scheduler is agent-aware through metadata, not model-name hacks

**Decision:** Runtime passes workload class/priority/deadline/cancellability.

**Reason:** keep Inference generic and prevent hard-coded orchestration model names.

## ADR-006 — Continuous batching + chunked prefill are target capabilities

**Influences:** Orca, vLLM, Sarathi.

**Gate:** implement incrementally and benchmark interactive latency, not just aggregate throughput.

## ADR-007 — Distributed work starts request-level

**Decision:** whole-model/request routing before tensor/model sharding across ordinary LAN links.

**Reason:** simpler correctness and often better cost/benefit on heterogeneous consumer hardware.

## ADR-008 — Native operators are benchmark-driven

**Decision:** replace backend operators only when profiling and reproducible benchmarks justify them.

**Reason:** avoid maintaining redundant kernels without a measured benefit.

## ADR-009 — Observatory is external

**Decision:** telemetry producer only; no 3D renderer in inference process.

**Reason:** isolation and hot-path safety.

## ADR-010 — Research ideas are not automatically code dependencies

**Decision:** concept adoption and code reuse are separate decisions.

Before code reuse:
- license review
- API/ABI stability review
- maintenance status
- security review
- benchmark/correctness validation
