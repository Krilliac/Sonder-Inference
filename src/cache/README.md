# Cache

Logical KV-cache manager (Phase 2 of [KV cache design](../../docs/KV_CACHE.md)).
Pure policy and bookkeeping: it owns *logical* fixed-size blocks and per-sequence
block tables and never touches device memory. Backends map a `BlockId` onto
their physical storage and apply the copy operations the manager reports.

Namespace `sonder::inference::cache`, headers under
`include/sonder/inference/cache/`:

| Header | Contents |
| --- | --- |
| `types.hpp` | `BlockId`, `SequenceId`, `TokenId`, `Priority`, `CacheFingerprint` + `FingerprintBuilder`, `PressureLevel` |
| `eviction_policy.hpp` | `EvictionPolicy` interface, `LruEvictionPolicy`, `PriorityAwareEvictionPolicy`, `make_eviction_policy()` |
| `kv_cache_manager.hpp` | `KvCacheManager`, `KvCacheConfig`, `KvCacheStats`, `CacheEvent`, `BlockCopy` |

## Behaviour

- **Blocks and tables.** `append_tokens()` fills the tail block and allocates new
  blocks as needed. It is all-or-nothing: when the worst-case block need exceeds
  free plus evictable blocks it returns `ErrorCode::unavailable` and changes nothing.
  `can_append()`/`blocks_needed()` support admission checks.
- **Prefix reuse.** Every full block gets a chained hash of the parent hash, the
  `CacheFingerprint` (model, revision, adapters, tokenizer, rope, cache format and
  backend compatibility), and its tokens. Lookups also compare the stored tokens,
  parent hash and fingerprint, so a hash collision cannot cause a wrong reuse. While a
  sequence has been served entirely from cache, each whole block is looked up
  before allocation. `cached_prefix_tokens()` reports how much prefill can be skipped.
  `match_prefix()` is a side-effect-free query for prefix-aware scheduling.
- **Fork / copy-on-write.** `fork_sequence()` shares every block by reference
  count. Writing into a shared partial tail allocates a private copy and queues a
  `BlockCopy{src, dst, num_tokens}` for the backend (`take_pending_copies()`).
  Full blocks are immutable and are never copied.
- **Truncate.** `truncate()` rolls a sequence back, for example after rejected
  speculative tokens. A tail block that is later rewritten is copied if shared,
  or removed from the prefix index if exclusively owned.
- **Eviction.** When a block's reference count reaches zero, a full block stays in
  the prefix index as *cached/evictable*; partial blocks return to the free list.
  Blocks are evicted only when the free list is empty. The default
  `PriorityAwareEvictionPolicy` evicts lowest priority first, then the deepest
  blocks (leaves before roots), then the cheapest recompute, then least recently
  used. `LruEvictionPolicy` is also available, and custom policies plug in
  through the constructor.
- **Pressure and telemetry.** `pressure()` compares pinned utilisation with the
  configured watermarks. `stats()` returns gauges (free, cached, pinned, shared,
  bytes) and counters (hits, misses, avoided prefill tokens, evictions, CoW,
  allocation failures). An optional listener receives `allocated`, `reused`,
  `evicted`, `copy_on_write` and `pressure_changed` events synchronously.

The manager is not thread-safe; the owning scheduler serialises access.

## Tests

`tests/` contains doctest cases registered as `sonder.cache.*`, including a
randomized stress test that checks every invariant (`validate()`) and
reconstructs sequence contents after each operation. To build standalone:

```sh
cmake -S src/cache -B build/cache -G Ninja
cmake --build build/cache -j 4
ctest --test-dir build/cache
```
