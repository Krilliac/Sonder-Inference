# Integration notes: `feat/kv-cache` (src/cache)

This branch changes only `src/cache/**` and this file.

## Build wiring (no root edits needed)

- The root `CMakeLists.txt` already calls `add_subdirectory(src/cache)` when
  `src/cache/CMakeLists.txt` exists. In-tree the module:
  - adds `types.cpp`, `eviction_policy.cpp` and `kv_cache_manager.cpp` to
    `sonder_inference` (`sonder_module_sources`);
  - exports `src/cache/include` (`sonder_module_include_directories`);
  - defines `SONDER_HAS_KV_CACHE=1` (`sonder_module_define`);
  - registers doctest cases as `sonder.cache.<case>` through
    `sonder_add_module_tests(cache ...)`, which builds the `sonder_cache_tests` target.
- The module depends only on the header-only parts of
  `include/sonder/inference/error.hpp` (`Status`, `ErrorCode`) and the standard
  library. It has no dependency on engine, session or backend code.
- A standalone fallback (`cmake -S src/cache`) builds a
  `sonder_inference_cache` static library plus tests from the root `include/`
  and `cmake/` helpers. This lets the module be tested before the core sources
  land. The in-tree path skips it.

## Error mapping (core `ErrorCode`)

| Condition | ErrorCode |
| --- | --- |
| unknown sequence | `not_found` |
| sequence id already exists | `invalid_state` |
| out of free and evictable blocks (no state change) | `unavailable` |
| truncate beyond length | `invalid_argument` |

## Suggested core follow-ups (lead-owned; not done here)

1. **Session/engine:** have each `Session` own a `SequenceId` in a
   per-model `KvCacheManager`. Session fork should call `fork_sequence()`.
   Build the `CacheFingerprint` with `FingerprintBuilder` from the model id and
   revision, adapter set, tokenizer config, rope config, KV quantization/format
   and backend compatibility.
2. **Scheduler (src/scheduler):** use `can_append` / `blocks_needed` /
   `available_blocks` for admission, `match_prefix()` for prefix affinity, and
   `pressure()` for backpressure. Map work classes onto `Priority`, where a
   higher value means more important (e.g. interactive_user=7 … maintenance=0).
   `set_priority()` handles reprioritisation. When `append_tokens` returns
   `unavailable`, preempt or requeue the sequence instead of treating it as a
   failure.
3. **Backends:** once the `kv_copy` capability exists, drain
   `take_pending_copies()` before each step and apply the copies to physical KV.
   `block_table(seq)` is the logical-to-physical page table.
4. **Telemetry:** forward `CacheEvent`s and `stats()` to the Observatory bus:
   - `allocated` → `kv.allocated`
   - `reused` → `kv.reused` (with `avoided_prefill_tokens`)
   - `evicted` → `kv.evicted`
   - `pressure_changed` → `kv.pressure`
   - `copy_on_write` → `kv.allocated`, with reason `cow`, or a new `kv.copied`
   The gauges in `KvCacheStats` cover bytes by tier (GPU only for now), blocks
   by state, and hit/miss. The listener runs synchronously on the caller's thread.
   Keep it cheap, for example by enqueueing to the telemetry bus.
5. **Configuration:** set `KvCacheConfig.num_blocks` and `bytes_per_token` from
   backend-reported capacity, and choose `block_size_tokens` per backend (16 by
   default).

## Not in scope yet (docs/KV_CACHE.md tiers 2–5)

The module does not implement CPU spill, disk or remote tiers, compaction
lineage records, or quantized-KV bookkeeping. `EvictionCandidate` already
carries `recompute_tokens`, so transfer-cost inputs can be added without an API
break.
