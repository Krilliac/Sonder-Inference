# Integration notes: feat/ollama-bench

Owned paths: `src/backends/ollama/**`, `bench/**`, and this file. The branch
extends the adapter and harness skeletons that landed with the foundation, in
place. It changes no root, core, CLI, CI, or other-module files. All public
contracts that core and the CLI use are preserved:

- `sonder::inference::OllamaBackendOptions`, `kOllamaBackendName`, and
  `make_ollama_backend()` (used by `c_api.cpp` and `sonder-infer`)
- `ollama_protocol.hpp`: `build_generate_body`, `parse_stream_line`, and
  `descriptor_from_tag`
- `bench::{Prompt, Corpus, Options, parse_corpus, load_corpus, run, percentile}`,
  corpus schema `sonder.inference.corpus/1`, and result schema
  `sonder.inference.bench/1` (all additions are backward compatible)
- The foundation tests `src/backends/ollama/tests/test_ollama_adapter.cpp`
  and `bench/tests/test_benchmark.cpp`, kept unchanged

## What's added

### Ollama (`src/backends/ollama`)

- **`ollama::OllamaClient`:** `/api/generate` and `/api/chat` NDJSON
  streaming, plus `/api/tags`, `/api/show`, `/api/ps`, and `/api/version`.
  - Cancellation through `CancellationToken`, which the core HTTP client polls
    during I/O. A tripped token returns `ErrorCode::cancelled`.
  - A callback that returns false stops the stream and keeps the partial
    result.
  - Client-side TTFB, TTFT, and wall time.
- **`ollama::StreamDecoder`:** an incremental NDJSON decoder. It handles
  arbitrary byte splits, CRLF, and blank lines, and detects `{"error"}` lines
  (`backend_error`), malformed JSON, and streams truncated before `done`
  (`protocol_error`).
- **Adapter rewritten on the client:**
  - `load_model` uses `/api/show`, which fills in family, parameter size,
    quantization, `context_length`, and size. It never pulls a model.
  - HTTP status mapping:

    | HTTP status | ErrorCode |
    |---|---|
    | 404 | `not_found` |
    | 400 | `invalid_argument` |
    | 503 | `unavailable` |
    | Other non-2xx | `backend_error` |

  - `done_reason` "length" maps to `max_tokens` and "stop" maps to
    `end_of_sequence`.
  - New option `OllamaBackendOptions::emit_thinking_chunks` (default false).
    When false, TTFT measures the first content token.
- **Telemetry:**
  - `ollama::timing_attributes()` returns flat, server-reported fields only.
  - `ollama::emit_timing_events(bus, ctx, timings, model)` emits:
    - `model.load.completed` (only when `load_duration > 0`)
    - `inference.prefill.completed`
    - `inference.decode.completed`
- **Tests (34 `sonder.ollama.*`):** recorded NDJSON fixtures and an in-process
  fake Ollama server, covering chunked streaming, byte splits, chat and
  thinking output, HTTP and mid-stream errors, truncation, cross-thread
  cancellation of a stalled stream, the loopback/TLS policy, and the adapter's
  mapping to `GenerateStats`. No live Ollama is needed. The foundation's opt-in
  live test (`SONDER_TEST_OLLAMA_MODEL`) is still there.

### Bench (`bench/`)

- **Corpus schema:** now accepts optional top-level `fillers`, plus per-prompt
  `context: {filler, repeat}`, `shared_prefix`, and `children`. Each child of
  an agent fan-out prompt runs concurrently in its own session.
- **`bench::run` additions:**
  - `per_prompt` distributions
  - `fanout_batches` (makespan and aggregate tok/s)
  - `client_decode_tokens_per_sec`
  - `p99`
  - `host.hardware`
  - corpus version and fnv1a hash
  - cold first-request and backend load time
  - `wall_seconds`
  - `Options::{hardware, prompt_ids, budget_seconds}` and
    `truncated_by_budget`

  Warmup failures are no longer counted as failures.
- **New functions:** `bench::render_markdown()`, `default_result_stem()`, and
  `write_results()` (JSON plus `.md`).
- **New target `sonder-bench`** (`bench/tools/sonder_bench.cpp`): a standalone
  runner that writes JSON plus markdown to `bench/results/`. It supports
  `--budget-seconds` (default 540), `--prompts`, `--require-idle` (refuses to
  run when a different Ollama model is resident), `--telemetry FILE`,
  `--list-models`, and `--dry-run`.
- **Corpora:**
  - `bench/corpus/baseline.json`: short fact, coding, medium reasoning, a
    ~2.4k-token long-context prompt, and a 4-way fan-out with a shared
    ~600-token prefix.
  - `smoke.json` is unchanged.
- **Tests (12 `sonder.bench.*`):** the 4 foundation tests, 6 extended tests,
  the existing `cli_mock`, and a new `tool_mock` (runs `sonder-bench` on the
  mock backend).

## Requested changes outside owned paths (for the integrator)

1. **CLI (optional).** `sonder-infer bench` could write the markdown summary
   too, for example with a `--markdown FILE` option that writes
   `si::bench::render_markdown(doc)`. It could also expose
   `Options::budget_seconds`, `Options::hardware`, and `Options::prompt_ids`.
   `sonder-bench` already covers all of this.
2. **docs/LICENSE_REVIEW.md:** add a record for the test-only dependency:
   ```
   ### cpp-httplib
   - repository: https://github.com/yhirose/cpp-httplib
   - revision/tag: v0.58.0 (single httplib.h, sha256 aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5)
   - evaluated: 2026-09-26
   - license: MIT
   - intended use: in-process fake Ollama HTTP server for src/backends/ollama tests only
   - linkage/process boundary: header-only, compiled into sonder_ollama_tests only; never linked into sonder_inference
   - notices required: MIT notice (inside the header); not redistributed
   - approved: pending integrator review
   ```
   It is fetched with FetchContent from a pinned URL with URL_HASH, in
   `src/backends/ollama/tests/CMakeLists.txt`. The runtime library has no new
   dependency; transport and JSON are the core `src/net` and
   `sonder/inference/json.hpp`.
3. **docs/BENCHMARK_PLAN.md "Harness (v0.1)":** mention `sonder-bench`, the
   baseline corpus, agent fan-out concurrency, the markdown output, and
   `--require-idle`. The "concurrency 1 only" limitation no longer applies to
   fan-out prompts.
4. **docs/MODULES.md:** update the status rows for `src/backends/ollama/` and
   `bench/`.

## Known gaps

- **Ollama baseline: PENDING** (`bench/results/PENDING_ollama_baseline.md`).
  Nate's workstation was offline while this was built, so no live Ollama was
  reachable. No numbers were invented, and the capture command is in that
  file.
- **No context-window control.** `SamplingConfig` has no `num_ctx`, so the
  long-context case is sized for Ollama's default 4096 window.
- **No TLS.** The core HTTP client is plain HTTP. `https://` returns
  `unsupported`, and a non-loopback host needs `allow_remote`. Reaching
  Node1's TLS worker (`https://10.77.0.2:8443`) needs a TLS transport in core
  `net`, with verification on.
- **Chat not wired into the Backend path.** `/api/chat` is in `OllamaClient`,
  but the Backend path uses `/api/generate` because `GenerateRequest` is
  prompt-only.

## Validation

- On main 0c0b054 plus this branch, GCC 14 Debug with `-Werror` passes
  123/123 CTest tests: 34 ollama, 12 bench, and the rest core and scheduler.
- On the earlier main snapshot (before the scheduler merge), Clang 19 with
  `-Werror`, GCC ASan+UBSan, and GCC TSan each passed 90/90.
- End-to-end runs against a local fake NDJSON Ollama worked through both
  `sonder-bench` (full baseline corpus, including fan-out and a telemetry
  JSONL file) and `sonder-infer bench`. A missing model exits with code 3.
- MSVC was not built locally. The tests avoid signed/unsigned comparisons,
  and the httplib include is wrapped in `#pragma warning(push, 0)`.
