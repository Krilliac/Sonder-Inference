# Full-artifact GGUF inspected-content observation

This document records the private inspected-content boundary and its accepted
local native and synthetic qualification. Hosted revision checks and merge
receipts are tracked separately in the pull request.

The private server-module function
`server::detail::observe_gguf_artifact(input, expected_sha256, options, cancel)`
parses the existing GGUF metadata/tensor-table interpretation, then drains and
SHA256-hashes the same caller-owned stream buffer to EOF. Success returns the
verified full digest, byte length, literal architecture and optional native
`RoutedExpertCounts`. It snapshots options and the expected digest before caller
stream-buffer callbacks run, retains a single buffer identity, and refuses buffer
replacement. The caller keeps that buffer alive and supplies the trusted
expectation; the function does not authenticate its provenance.

The existing header-only parse/read entrypoints remain unbound observations.
The new function has no default serving call site. It does not populate
`ModelDescriptor`, nine-key `BackendIdentity`, the C ABI, HTTP models metadata,
Runtime routing, placement, cache reuse or telemetry. Physical expert counts
remain orthogonal to attention/hybrid/recurrent cache classification and task
quality. Missing or ambiguous counts mean unknown. Literal dots and case do
not establish namespace aliases.

The input must be an already-open binary stream positioned at the artifact's
beginning. The function does not seek, reopen a path, close the stream or
change its exception mask. Stream-buffer reads may consume to EOF without
setting the owning stream's EOF flags. Positive short transfers are refilled;
zero progress while a byte remains available is an I/O failure. Non-allocation source exceptions, failed stream state and buffer replacement
yield a sanitized I/O status.

Required options:

- `2 <= max_read_bytes <= UINT64_MAX / 8`. The cap includes a reserved boundary
  byte: admitted byte length is strictly less than the cap. An artifact of
  `cap - 1` bytes can succeed after an EOF probe; a returned cap-th byte causes
  refusal even if it is the final byte. No extra `cap + 1` read is allowed.
- `0 < max_header_bytes < max_read_bytes`. The header/table limit is inclusive,
  checked before reads/skips and declared-string resize. Existing native
  count, per-string, array and block caps remain in force.
- A finite absolute `steady_clock` deadline. Cancellation and deadline checks
  surround bounded reads, parser/postprocessing work and final admission.
- `1 <= read_chunk_bytes <= 65536`. Payload drain uses a fixed 64 KiB scratch
  buffer and retains no artifact-sized tail.

These are logical byte and cooperative time controls. They do not constrain
OS/file-buffer prefetch, interrupt blocked I/O, enforce hard latency or bound
heap/RSS. Estimate-only parsed metadata and vectors are discarded before tail
drain. Caller-owned source behavior and digest provenance remain preconditions.

Every failure returns no observation. Invalid options/digest syntax, malformed
or truncated header, header/read budget exhaustion and digest mismatch return
`invalid_argument`; observed cancellation returns `cancelled`, expiry returns
`timeout`, source failure returns `io_error`, and representable allocation
exhaustion returns `unavailable`. Diagnostics contain fixed categories rather
than paths, raw header/weight bytes, digests or source exception text. Ordinary
header-reader diagnostics retain their established behavior.

Full digest verification certifies the emitted inspected bytes, not tensor
extent validity or a loadable GGUF model. A trusted original digest rejects a
truncated tail; a matching digest of already truncated or header-only content
can still bind its parse-valid header. This does not certify filesystem
immutability, authenticity, actual backend-loaded bytes, tokenizer/template
identity, active parameter counts, routing frequencies or model quality.

Loaded-artifact association and separately qualified descriptor/HTTP/policy
projection remain the next architecture boundaries. This slice does not
complete the broader placement, recurrent-checkpoint, embedding/reranking or
residency roadmap contracts, or replace existing residency controls.


## Local qualification (2026-10-06)

The source candidate starts from `f140081c548f086fb0ebd685477d1271fd3b346c`.
Linux Debug used GCC 14, `-g -std=c++20 -Wall -Wextra -Wpedantic`, no optimization
flag, TLS and llama.cpp disabled, and existing licensed offline test dependencies.
The library's CMake commit macro is `f140081c548f`: a 12-character HEAD stamp
without a dirty marker. Full source hashes identify the modified candidate;
this is not a binary built from clean main or a later hosted commit.

Configure, build, discovery, focused tests, full tests and `git diff --check`
all exited zero. Discovery and full JUnit contain 959 distinct tests; the new
14 cases pass with no failures, errors or skips. They cover full digest/tail
identity, independent SHA padding vectors, truncation, strict read-cap and
inclusive header boundaries, pre-allocation allowance, short/failed reads,
stream exceptions and replacement, option/expectation snapshots, cancellation,
controlled deadline expiry, literal unknown counts and repeated fresh instances.
The first run preserved an actual fixture failure: a renamed architecture
retained a block-count key in its old namespace. The fixture-only correction
qualified subsequently; no implementation, budget or timeout changed.

A separately compiled external `-O0 -g` C++20 driver linked the same qualified
static library. Six Python-hashlib-verified synthetic artifacts total 9,634,023
bytes. They are GGUF-shaped test inputs, not a loadability/model-quality corpus.
The timed operations use a controlled nonseek memory stream; fixture loading,
oracle checks, stream/options setup, result validation and output formatting
are outside the timers. A matched same-stream SHA-only reader uses the same
native incremental SHA implementation. Each cell has three excluded warmups
per version and 20 alternating pairs; all 120 measured rows are retained.

| Artifact | Observation median / p95 (ms) | SHA-only median / p95 (ms) |
| --- | ---: | ---: |
| 64 KiB | 2.072194 / 2.265870 | 2.020218 / 2.117864 |
| 1 MiB | 33.030301 / 38.507039 | 32.836403 / 37.324678 |
| 8 MiB | 260.563357 / 263.663485 | 260.731928 / 281.940295 |

The observation performs parsing, byte/deadline/cancellation checks, hashing
and final admission. These descriptive costs do not establish an optimization
win, statistical significance, physical-disk throughput, provider/model quality
or inference throughput. The 1 MiB observation maximum was 59.758911 ms; it is
retained, not discarded. Nearest-rank p95 uses the 19th of 20 sorted samples.
The median of the 20 within-pair observation/SHA ratios was 1.025720,
1.004756 and 1.000854, respectively. This statistic differs from dividing the
two version medians. The 8 MiB SHA-only maximum was 309.062132 ms; its tail
contributes to the apparent p95 advantage, which establishes no speedup.
Cost-process lifetime peak RSS was 33,320 KiB, including fixture setup and
prior operations and possibly inherited pre-exec interpreter highwater; this
is not per-operation memory or a heap/leak bound.

A separate stress process made 120 serial calls plus two joined workers with
32 fresh calls each: 64 successes, 60 `invalid_argument` refusals and 60
cancellations. Variants include known/unknown counts, identical headers with
changed tails, cancellation before/during short reads and strict cap refusal.
Two extra controls admit header-only bytes with their own trusted full digest
and reject a truncated tail against the original full expectation: 186 total
calls. Fixtures were unchanged after the run. Stress-process lifetime peak RSS
was 33,312 KiB, with the same memory limits on interpretation.

All six native stages and four driver stages record unchanged selected
source/primary/cache/user/HOME preservation and strict owned-process cleanup
through `ECHILD`, with no cleanup signals or owned leftovers. This is the
selected documented preservation scope, not a census of all historic worktrees
or proof against blocked I/O, hard OOM or all possible source behavior.

Retained receipt SHA256s:

- Native qualification: `c6f7809b4bf59de6cf5c4e9579dd047a14bd32282e88a5d1b64e62cc7b464942`.
- Native root tool/stamp receipt: `3ea95f5456d4b2a8a616a2dde9166edbeffa86d7d21ea757471e5b95f2b93d9c`.
- Fixture setup: `dabaee32ab92281abcf5f8d616a93b53931dff5d7c14baf61c46673f2b6894da`.
- Driver compilation: `eff4592611d3d5ad4d516a6011bc5067828228c645bd3cb2e337e22b233f34a9`.
- Cost qualification: `5b59fedb945f5c8435faeb5abfb7d86b3e97ace8f17dec4ff99e20482f583012`.
- Stress qualification: `a09f92bc4f8fff108d076c20510244b190e492893325433eec9a8b3b300fe47f`.
- Completed driver DATA freeze: `70ef1c9270cfb832c0ee76161baa64a4d93f22cc6dccf1dd33de8c1a390f168d`.

Raw receipts, logs, JUnit, process ledgers and fixtures remain in the owned
`sonder-gguf-artifact-qualified-iw6xfvqh` qualification workspace. Generated
artifacts and model weights are not committed. Independent native and driver DATA reviews accepted these local receipts:
`811b7e44dc2811baab42f34a1cb1cae92a1d1f87b96c680c77068fb6b52c014a`
and `40dd68d77e451db5254a2749a015d4894e871fdb504e723cdc077095275700dc`,
respectively. Hosted exact-revision qualification remains separate.
