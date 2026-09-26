# KV / Context Cache Design

KV state is a first-class engine resource, not an opaque side effect.

## Goals

- bounded memory
- prefix reuse
- cheap session continuation
- fork/share for agents
- clear ownership/accounting
- multi-tier placement
- observable eviction/movement
- backend-independent logical API

## Logical model

```text
Context lineage
root/system
   |
project prefix
   +----------+
   |          |
agent A     agent B
   |          |
decode      decode
```

Physical KV may be shared for common immutable prefixes.

## Block/page abstraction

Adopt a logical block/page model inspired by PagedAttention.

Do not expose backend-specific raw pointers as the public cache API.

A logical segment contains:
- cache ID
- model/cache-format fingerprint
- token range
- parent/prefix relationship
- physical residency
- byte size
- reference count
- last use
- recompute cost estimate
- privacy/retention class

## Prefix lookup

Cache key should include enough state to prevent incorrect reuse:

- model identity/revision
- adapter set
- tokenizer/config
- relevant positional/rope configuration
- cache quantization/format
- exact prefix token IDs
- backend compatibility fingerprint where necessary

Correctness beats hit rate.

## Tiers

Potential tiers:

```text
GPU KV
  <-> pinned/system RAM
  <-> normal RAM
  <-> local SSD
  <-> remote cache/node
```

Do not implement all tiers at once.

Suggested sequence:
1. GPU block manager
2. CPU spill
3. prefix persistence/reuse
4. optional disk
5. optional remote transfer

## Eviction

Inputs:
- last use
- prefix fan-out/reference count
- recompute cost
- transfer cost
- priority/workload class
- expected near-term reuse
- size

Avoid naive LRU as the only policy once shared prefixes exist.

## Fork/share

Agent forks are a natural Sonder optimization:

```text
system + project + task prefix
          |
      shared KV
     /    |    \
 worker  critic researcher
```

Forking should reference immutable blocks rather than duplicate them.

## Compaction

Context compaction creates a **new semantic lineage**. It is not equivalent to dropping arbitrary KV blocks.

Record:
- old context ID
- compacted context ID
- summarized token range
- new token range
- cache bytes released
- generation that produced the compacted representation

## Quantized KV

Treat KV quantization as a backend/capability choice with quality/performance tests. Cache format becomes part of compatibility fingerprinting.

## Distributed cache

Study LMCache/Mooncake concepts.

Before remote KV movement:
- estimate bytes
- network latency/bandwidth
- recompute cost
- destination pressure
- cache reuse probability

Sometimes recomputing is cheaper than moving.

## Telemetry

Required event families:
- kv.allocated
- kv.reused
- kv.evicted
- kv.moved
- kv.quantized
- kv.pressure

Required gauges:
- bytes by tier
- blocks by state
- hit/miss
- avoided prefill tokens
- eviction reason
- transfer bytes/time
