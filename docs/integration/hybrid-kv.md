# Hybrid/recurrent KV checkpoint validation

2026-09-29. Worktree `feat/hybrid-kv-checkpoints`, baseline
`a2aa72dec3228c902166cdf26aa5ff68f2fd3ccb`. Changes are uncommitted.

## Scope and contracts

- Added the C++ model architecture descriptor and load-time GGUF classification.
  No C ABI, Python projection, HTTP schema, or committed catalog was changed.
- Added logical checkpoint retention, exact-position prefix reuse, separate
  checkpoint accounting, and safe fork/truncate rules in `src/cache`.
- Propagated the descriptor through the common Session/runtime path.
- Updated `KV_CACHE.md`, `PLACEMENT.md`, and `ROADMAP.md`.
- Kept llama.cpp tag, hash, CI pin, and license record at b11195. Candidate-tag
  compilation was not verified; see [the pin decision](../KV_CACHE.md#llamacpp-checkpoint-reference-and-pin).

These are bookkeeping tests, not native inference or quality/performance results.
Physical backend checkpoint capture/restore remains unimplemented. The engine
therefore registers no fabricated checkpoints and reports zero hybrid/recurrent
logical reuse until a backend integration can supply real saved state.

## Measured blast radius

The production source inventory found one `Session::run_request` submission
site, one runtime sequence-creation site, and one call to cache `append_tokens`.
The runtime append helper has three execution call paths: planned prefill,
planned decode, and late-token catch-up. All use the request's stored
architecture. There are zero production callers of the standalone
`match_prefix` query; scheduler reservations remain worst-case block counts.
The loaded llama.cpp descriptor has one `GetModelInfo` consumer. Metadata is
classified once per load, not per request or token.

The new integration test covers 18 combinations (three architectures, two
execution modes, and generate/generic-chat/native-chat routing), two requests
each. All 36 requests complete with stable repeated output and expected reuse
accounting. Hybrid/recurrent requests without checkpoints report zero reuse;
attention-only requests retain full-block accounting.

Attention-cache randomized stress is identical before/after for **both** LRU
and priority-LRU: 2,343 successful operations and 5 capacity rejections across
4,000 attempted operations per policy. Scheduler simulation results and all
1,355 scheduler assertions are unchanged. In the native CLI mock smoke checks,
the before/after generation and chat output strings are identical.

## Build and test evidence

The `ci-windows` baseline configure stalled at `Detecting C compiler ABI info`
after identifying MSVC 19.44.35228. A bounded GCC CMake attempt stalled at the
same stage; both were cancelled. Direct compilation worked with Windows GCC
16.1.0 using C++20, `-Wall -Wextra -Wpedantic -Werror`, `-O0 -g0`, and two
compiler workers. It compiled the default production modules and their full
doctest suites: 83 baseline and 85 changed C++ translation units, plus the CLI
translation unit and C ABI smoke program. Existing local pinned doctest 2.5.3
and cpp-httplib test dependencies were read without modifying their source.
The compiler's own runtime DLL directory was put first in the test process
PATH to avoid loading an incompatible libstdc++ from the inherited PATH.

| Suite | Baseline passing cases | Changed passing cases | Assertions before / after |
| --- | ---: | ---: | ---: |
| Cache | 34 | 62 | 11,665 / 11,908 |
| Scheduler | 36 | 36 | 1,355 / 1,355 |
| Sampling | 96 | 96 | 102,384 / 102,384 |
| Core, including runtime/session | 124 | 129 | 3,487 / 3,687–3,688 |
| Ollama fixture tests | 46 | 46 | 1,241 / 1,241 |
| Server, isolated cases | 57 of 58 | 57 of 58 | One identical crashing case, below |
| Benchmark harness | 14 | 14 | 117 / 117 |

The Ollama executable reports 47 cases, but one live opt-in case returns early
without testing anything because `SONDER_TEST_OLLAMA_MODEL` is unset. That case
is excluded from the table. No model weights or live model service were used.
Core runtime assertion counts varied slightly across runs; the 129 passing
case count was stable.

The same server case crashes on both baseline and changed Windows GCC builds:
`lifecycle: a hung backend probe never blocks health or identity`, exit
`3221226356` (`0xC0000374`, heap corruption). The other 57 cases were run in
separate processes, as CTest does; each reported exactly one executed test and
positive assertions. Commas in doctest filters were escaped to prevent false
zero-test passes. This is an observed baseline/harness failure, not evidence
that MSVC or Linux CI has the same failure.

Additional before/after checks: five CLI smoke invocations each (version,
JSON version, help, mock generate, mock chat), all exit 0; C ABI smoke, exit 0
on both. `git diff --check` passes. Runtime-only `scripts/check_lint_ratchet.py`
and `scripts/check_architecture.py` do not exist in this Inference worktree;
they were not run, and no Runtime repository was modified.

The 33 added doctest cases cover architecture fixtures, checkpoint position and
lineage, interior-block restore/COW, memory/count limits, overflow, eviction
notifications, ID reuse, fork/truncate, chunked appends, attention parity, and
the 18-combination runtime matrix. Checkpoint tests verify actual token contents
and selected state ownership, not just hit counts.

Not verified: either CI preset end-to-end, MSVC project compilation, Linux
execution, optional llama.cpp compilation at either tag, GPU/model-backed
checkpoint restoration, and live model quality/performance. Direct GCC checks
do not substitute for those gates.

Cleanup is blocked: automatic approval review rejected both the path-checked
and literal-path PowerShell deletions with only `blocked by policy` as the
reason. `.hybridkv-validation/` therefore remains untracked and contains the
temporary baseline/build/test evidence. No further deletion mechanism was
attempted after those rejections; remove that directory when permitted.
