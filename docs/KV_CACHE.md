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

## Hybrid and recurrent models

Attention KV and recurrent state have different lifetimes. Attention-only
models can reuse a matching token prefix in complete cache blocks. A recurrent
state describes one exact position; it cannot be truncated to an earlier token.
A hybrid combines both forms. For example, the Qwen3.8 layout motivating this
work has 64 layers arranged as 16 groups of three DeltaNet layers and one
attention layer: only the 16 attention layers keep token-indexed KV. Treating
all 64 layers as ordinary KV gives both the wrong memory estimate and the wrong
reuse rule.

`ModelDescriptor::architecture` distinguishes `attention_only`, `hybrid`, and
`recurrent`. The llama.cpp wrapper classifies GGUF `general.architecture` and
SSM, recurrent, and DeltaNet metadata hints at load time, then caches the result.
Attention metadata helps distinguish hybrids; zero attention head counts alone
are not attention evidence. Unknown models with recurrent hints use checkpoint
rules conservatively. Missing metadata retains the existing attention-only
default; a directory listing is not a GGUF inspection. Other backends must
provide architecture information when available.

The engine passes the loaded descriptor through the common generation/chat
request path into each cache sequence. Architecture is part of cache
compatibility even if two callers supply the same fingerprint. Scheduling
reservations remain worst-case block counts; there is no new model probe or
metadata read on the request/decode path.

For hybrid/recurrent sequences, first determine the matching cached token
prefix, then select the longest saved recurrent checkpoint at or before that
position. No checkpoint means zero reusable tokens. A checkpoint at token 6
does **not** authorize restoring token 4, nor may it authorize a request that
matches only 4 tokens. Tokens after the selected checkpoint require prefill.
Checkpoints belong to an exact model/token lineage; a same-length checkpoint
from another branch is not compatible.

The logical API is `save_checkpoint(sequence, position, state_size)`,
`remove_checkpoint(owner, position)`, and the non-mutating `checkpoint_at`
query. The caller owns the physical state; the cache retains its owner,
position, byte count, architecture and block lineage. Owners may outlive their
active sequence. Reusing an owner ID discards its previous checkpoints.
`free_sequence` retains eligible saved state, while block recycling, explicit
removal, truncation beyond a saved position, and `clear_cached` invalidate it.
Fork and nonzero truncate require an exact saved state; they do not manufacture
a new checkpoint or transfer its ownership. Consumers must restore the queried
state when implementing those operations physically.
Saved checkpoints remain evictable even while a descendant is active. Their
removal does not alter the descendant's token blocks or live execution state,
but a later fork/truncate needs another saved checkpoint. Restoring a snapshot
does not pin it indefinitely.

Checkpoint retention is bounded independently of KV blocks: defaults are
64 MiB of backend state and 1,024 records; zero disables registration. LRU
replacement updates on actual reuse, and each removal emits
`CacheEvent::Kind::checkpoint_evicted` with the state record so the backend can
release it. Byte sums are checked against the budget without overflowing.
`checkpoint_bytes` and `checkpoints` are gauges; `checkpoint_evictions` counts
all removals, including replacement and invalidation. Existing KV byte gauges
continue to describe KV blocks only.

`AppendResult::checkpoint` identifies the selected restore state.
`tokens_checkpoint_limited` / `checkpoint_limited_tokens` count matching tokens
that could not be reused in that append, and `checkpoint_reused_tokens` counts
tokens actually reused under checkpoint rules. The existing avoided-prefill
counter records only reusable tokens. Attention-only hashing, behavior, and
existing counters retain their prior meaning.

The prefix index still matches complete KV blocks. An interior checkpoint is
usable when matching full-block coverage reaches that exact position; a partial
tail with no indexed full-block coverage cannot create a hit. Copy-on-write
preserves the block prefix at an interior checkpoint. If retaining that prefix
would exceed the available block pool, append falls back to recompute. Chunked
prefill does not retroactively claim reuse after an earlier chunk missed a
checkpoint; callers seeking a later checkpoint must supply enough matching
tokens in the initial append.

This remains logical bookkeeping. Registering a checkpoint asserts that its
backend state was actually saved at the stated position. It must not be
registered just because the scheduler allocated token blocks. Backends do not
yet expose checkpoint capture/restore through `BackendModel`, so the engine
does not invent checkpoints: hybrid/recurrent engine requests currently report
zero logical prefix reuse. The direct llama.cpp wrapper still clears its
context for each generation. Physical reuse, checkpoint serialization, and
quality/performance measurements are follow-ups.

The checkpoint state size is separate from attention KV bytes. The latter
must count only attention layers/heads and their actual cache dtype; recurrent
state bytes depend on the architecture and must come from the backend, not
from total layer count times context length. Neither a fixture nor the mock
backend establishes actual model memory use or an inference speedup.

### llama.cpp checkpoint reference and pin

Upstream llama-server exposes `--ctx-checkpoints`, `--checkpoint-min-step`,
and `--cache-ram`. Its checkpoint lifecycle fixes are useful references:
[PR #28302](https://github.com/ggml-org/llama.cpp/pull/28302) corrects checkpoint
spacing eviction, and [PR #27530](https://github.com/ggml-org/llama.cpp/pull/27530)
corrects state-restore cleanup. Sonder does not expose those server flags or
inherit the server's checkpoint machinery by linking `llama`.

The optional backend remains pinned to **b11195**. On 2026-09-29, review of
[PR #29385](https://github.com/ggml-org/llama.cpp/pull/29385) found internal
common/server migration toward `llama_batch_ext`/`llama_process`, while Sonder
uses `llama_batch_get_one`/`llama_decode`. That inspection is not a compile
guarantee for b11232 or a later revision. The upgrade is blocked on obtaining
the exact candidate source and compiling/linking the optional backend with its
metadata, batch, sampler, memory, and tensor-placement API calls. Local CMake
stalled during compiler detection and direct source download was refused in
this sandbox. No candidate-tag compile was verified, so tag, archive hash,
license record, and CI pin remain unchanged. An upgrade must verify these
together and exercise the optional backend's tests.

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
