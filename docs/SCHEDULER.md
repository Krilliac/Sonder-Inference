# Scheduler Design

## Goal

Turn Sonder's knowledge of task/agent intent into better inference scheduling without coupling the engine to high-level agent implementation.

The scheduler receives explicit metadata; it does not infer importance from prompt text.

## Work classes

Illustrative default classes:

```text
interactive_user       highest latency sensitivity
owner_orchestrator     high
critic_verification    high/medium
implementation_worker  medium
research_worker        medium/low
background_indexing    low
maintenance            low
```

Users/runtime policy can override these.

## Foundations to borrow conceptually

- Orca: iteration-level scheduling
- vLLM: continuous batching and paged request state
- Sarathi: chunked prefill
- FastServe: preemption/fair scheduling concepts
- SGLang/Preble: prefix-aware scheduling
- DistServe/Splitwise: P/D separation when justified

## Request state machine

```text
created
  -> waiting_admission
  -> admitted
  -> prefill
  -> decode
  -> completed

Any active state
  -> preempted
  -> queued
  -> cancelled/failed
```

## Admission

Admission considers:
- model residency
- expected KV growth
- available cache blocks
- workspace memory
- priority
- deadline/latency class
- maximum active sequences
- backend batch limits
- device health
- remote transfer cost

Do not overcommit VRAM and then rely on OOM recovery as normal control flow.

## Continuous batching

Decode iterations should admit/remove sequences at iteration or bounded scheduling points.

Goals:
- keep accelerator occupied
- avoid head-of-line blocking
- maintain interactive latency
- support cancellation quickly

## Prefill scheduling

Large prefill can starve decode.

Initial policy:
- cap prefill chunk size
- reserve decode capacity
- interleave chunks with decode
- expose TTFT/TBT tradeoff configuration

## Prefix-aware scheduling

If two requests share reusable prefix/KV state:
- prefer placement on a worker/device that already has the prefix when the benefit exceeds queue delay
- track cache affinity as a scheduling input
- never sacrifice a strict interactive deadline for a small cache hit

## Preemption

Possible strategies:
- pause sequence while retaining KV
- evict/move KV and resume later
- recompute prefix as last resort

Record why preemption occurred.

## Agent-aware policy

Sonder Runtime passes:
- workload class
- agent/task ID
- priority
- deadline class
- cancellability
- speculative allowance
- quality mode

Inference does not hard-code "Terra/Luna/Sol" names into the scheduler.

## No-progress protection

Scheduler-level guards:
- bounded requeue count
- detect repeated backend failure with identical configuration
- refuse duplicate identical sequence execution unless explicitly requested
- surface starvation and queue-age metrics
- never spin infinitely on device placement

## Metrics

- queue time
- TTFT
- inter-token latency
- tokens/sec
- batch occupancy
- active/queued sequences
- preemptions
- starvation age
- prefix-cache hit benefit
- prefill/decode time share
- speculative accept rate
