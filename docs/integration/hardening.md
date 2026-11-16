# Integration notes: hardening (sanitizers + fuzzing)

Branch: `feat/hardening`. Base: `main` @ `d40a979`. Adds only new files:
`.github/workflows/hardening.yml`, `fuzz/**`, and this note. No `src/`,
`ci.yml` or root `CMakeLists.txt` changes.

## What was added

### `.github/workflows/hardening.yml` (runs on `pull_request`, pushes to `main`, manual dispatch)

| Job | What it does | Blocking? |
|---|---|---|
| `linux clang ASan+UBSan` | clang-18, whole tree (lib, CLI, every module's tests) with `-fsanitize=address,undefined -fno-sanitize-recover=all`, `-Werror`; `ctest` with leak detection, stack-use-after-return, init-order checks; CLI smoke | yes |
| `linux clang TSan` | same tree with `-fsanitize=thread`; `ctest -j4` | yes |
| `libFuzzer (short run)` | builds `fuzz/` standalone, replays seed corpora (`fuzz-regress`), then fuzzes every target for 60 s (dispatch input `fuzz_seconds` overrides); crash reproducers uploaded as `fuzz-artifacts` | yes |

No job is marked `continue-on-error`. Justification per the "non-blocking only
if flaky" rule:

* ASan/UBSan: 251/251 tests passed locally (clang 19, same flags and
  `ASAN_OPTIONS` as CI); deterministic.
* TSan: 251/251 passed, then `ctest --repeat until-fail:5 -j4` passed with no
  ThreadSanitizer reports. Not flaky locally, so it stays blocking. If it ever
  flakes on hosted runners (TSan is timing-sensitive), the right move is to
  fix the race or quarantine the single test, not to make the job advisory.
* Fuzz: a 60 s run per target is not "flaky" in the retry sense: every failure
  comes with a reproducer file. The trade-off is that a *new* finding can
  surface on an unrelated PR. Locally all five targets ran 60 s clean
  (0.5 to 2.2 M executions each), so today the job is green. If the lead
  prefers PRs never to be blocked by fresh discoveries, set
  `continue-on-error: ${{ github.event_name == 'pull_request' }}` on the
  "Fuzz each target" step only; keep "Replay seed corpora" blocking.

Runner notes: pinned `ubuntu-24.04` + `clang-18` (`libclang-rt-18-dev` for
the sanitizer/libFuzzer runtimes). `vm.mmap_rnd_bits=28` is set because
24.04 kernels' 32-bit mmap ASLR entropy breaks clang-18 sanitizer runtimes
("unexpected memory mapping"). Build parallelism is capped at 4.

### `fuzz/`

Eleven libFuzzer targets (see `fuzz/README.md` for invariants and input
formats), seed corpora under `fuzz/corpus/<target>/`, and
`fuzz/CMakeLists.txt`:

* `sonder_fuzz_ollama_stream`: Ollama NDJSON `StreamDecoder` (split
  invariance) + `parse_stream_line`.
* `sonder_fuzz_bench_corpus`: bench corpus JSON (`parse_corpus`, used by
  `load_corpus`).
* `sonder_fuzz_telemetry_envelope`: the telemetry/recording serializer
  (`TelemetryBus::make_envelope` + dump = JSONL line). There is no separate
  recording/replay format in the tree.
* `sonder_fuzz_json`: the shared JSON parser/serializer under all of the above.
* `sonder_fuzz_bench_report`: `render_markdown` / `default_result_stem`.
* `sonder_fuzz_http_request`: the `sonder-infer serve` request front end
  (request head, query, Host check, correlation headers, chat request).
* `sonder_fuzz_gguf`: the launch-profile GGUF header reader
  (`parse_gguf_model_info`) and `estimate_vram` on what it accepts. Model
  files come from the internet, and the reader's errors reach `/v1/models`.
* `sonder_fuzz_anthropic_request`: the `/v1/messages` front end
  (`parse_messages_request`, `reasoning_budget::parse_body`) and the
  `AnthropicStream` SSE encoder: frame grammar, UTF-8 and JSON re-parse of
  every frame.
* `sonder_fuzz_http_client`: the internal HTTP client's `parse_url`,
  `is_loopback_host` (against a reference oracle), `ChunkedDecoder` and
  `LineSplitter` (split invariance).
* `sonder_fuzz_llamaserver_text`: the text read from a supervised
  llama-server child: Prometheus metrics, `/slots`, the child log
  (`LogDiagnostics`, split invariance), spill policy and finish reasons.
* `sonder_fuzz_backend_config`: residency, warmup and tune-extension config
  parsing plus the llama-server timings and `/props` readers.

`fuzz/CMakeLists.txt` works in two modes:

1. **Standalone** (`cmake -S fuzz ...`): becomes the top-level project, adds
   the repo root as a subdirectory with tests/CLI off, and instruments the
   whole library (`-fsanitize=fuzzer-no-link,address,undefined`). CI uses
   this, so the fuzz job works **before** the lead wires anything.
2. **In-tree**: defines `option(SONDER_BUILD_FUZZERS ... OFF)`; does nothing
   unless it is ON. When ON it adds the fuzzer coverage/sanitizer flags to
   `sonder_inference` (PRIVATE compile, INTERFACE link) so normal builds are
   untouched. With MSVC/GCC it falls back to `SONDER_FUZZ_ENGINE=standalone`
   (replay-only driver, `standalone_main.cpp`).
   On Windows, `fuzz/CMakeLists.txt` also defaults `CMAKE_MSVC_RUNTIME_LIBRARY` to
   `MultiThreaded` for every standalone fuzz engine (not only libFuzzer) unless the
   caller sets it.

Verified locally: standalone clang build + regress + 60 s runs; in-tree build
with the snippet below (`-Werror`, fuzz-regress, and the normal 251 tests
still pass); GCC standalone-engine build + replay.

## What the lead must wire

> **Status on main:** done. The root `CMakeLists.txt` declares
> `SONDER_BUILD_FUZZERS` (OFF) and adds `fuzz/` when it is ON; CI keeps
> building `fuzz/` standalone.

1. Root `CMakeLists.txt`, **after** the `SONDER_MODULE_DIRS` loop (the fuzz
   targets need the bench and ollama module sources in `sonder_inference`):

   ```cmake
   option(SONDER_BUILD_FUZZERS "Build fuzz/ targets (Clang/libFuzzer)" OFF)
   if(SONDER_BUILD_FUZZERS)
       add_subdirectory(fuzz)
   endif()
   ```

   (`fuzz/CMakeLists.txt` also declares the option itself, so the `option()`
   line in the root is optional but documents it.) CI does not depend on this.
2. Optionally mark the three `hardening` jobs as required checks in branch
   protection once they have run green on `main`.
3. When fixing any bug below, add the repro input to the matching
   `fuzz/corpus/<target>/` so `fuzz-regress` pins it.
4. If the Ollama decoder, bench corpus schema, or telemetry envelope change
   shape, update the matching fuzz target's invariants (they compile against
   `ollama.hpp`, `ollama_protocol.hpp`, `benchmark.hpp`, `telemetry.hpp`).

## Bugs found (all fixed in #15)

> **Status on main:** B1-B6 are all **fixed** by #15 (feat/parser-fixes, main
> `23e0865`), together with the `int_field` UB found by
> `sonder_fuzz_ollama_stream`. The first hardening run on the merged tree hit
> that UB again before #15 landed; the repro inputs are in `fuzz/corpus/`.

Sanitizers (ASan, UBSan, LSan, TSan) and 5 x 60 s of fuzzing found **no memory
errors, UB, leaks or data races**. Targeted probing of the same parsers found
these logic/robustness bugs. Repros are exact inputs; results are from
`main` @ `d40a979`.

### B1. Quadratic JSON object parsing (DoS on the Ollama stream), `src/common/json.cpp`

`Parser::parse_object` calls `Object::set`, which linearly `find`s the key
before inserting, so an object with N keys costs O(N^2). Release build (-O2):

| keys | size | parse time |
|---|---|---|
| 50 000 | 0.5 MiB | 2.4 s |
| 100 000 | 1.0 MiB | 15.1 s |
| 200 000 | 2.2 MiB | 50.0 s |

`StreamDecoder` accepts NDJSON lines up to 16 MiB, so one line from a
misbehaving or hostile Ollama endpoint (`OLLAMA_HOST` can be remote) pins a
core for hours; the request timeout does not interrupt parsing.
Repro: `{"k0":0,"k1":0,...,"k199999":0}`.
**Fix:** in `parse_object`, append members without the lookup and resolve
duplicates afterwards (e.g. last-wins via a temporary
`unordered_map<string_view, size_t>` when the object has more than ~32
members), or keep an index alongside `members_`. Also consider a lower
per-line cap for Ollama (1 MiB is plenty for a chunk).

### B2. `max_tokens` silently truncated to int32, `bench/src/benchmark.cpp` (`parse_corpus`)

`static_cast<std::int32_t>(as_int(...))` happens before the range check, so
`"max_tokens": 4294967297` is accepted as `1` (and `4294967360` as `64`).
Repro:
`{"schema":"sonder.inference.corpus/1","prompts":[{"id":"p","prompt":"x","max_tokens":4294967297}]}`
-> ok, `max_tokens == 1`.
**Fix:** range-check the `int64_t` first, then cast.

### B3. Corpus amplification: ~165 000x expansion, `bench/src/benchmark.cpp`

`context.repeat` (up to 10 000) multiplies the filler, and the result is
copied into every fan-out child. A 1.2 KB corpus (1 000-byte filler,
`repeat: 10000`, 20 children) expands to **210 MB** of prompt text; a 64 KB
one reaches tens of GB and OOMs `sonder-bench`/`sonder-infer bench`. Corpora
are local files, so severity is low, but it is an easy footgun.
**Fix:** cap the expanded prefix (e.g. 16 MiB total per prompt, and total
corpus bytes), reject otherwise.

### B4. `\uD800\u0041`: an unpaired high surrogate swallows the next escape, `src/common/json.cpp` (`parse_string`)

When a high surrogate is followed by `\uXXXX` that is not a low surrogate,
both escapes are replaced by a single U+FFFD, so the next character (here
`A`) is lost. Repro: `json::parse(R"("\uD800\u0041")")` -> `"\uFFFD"`;
expected `"\uFFFD" "A"`.
**Fix:** if `lo` is not in `DC00..DFFF`, emit U+FFFD for the high surrogate
and then process `lo` as its own code point (rewind or handle inline).

### B5. Telemetry/JSON output can be invalid UTF-8, `src/common/json.cpp` (`append_escaped`), affects `src/telemetry`

`append_escaped` copies bytes >= 0x80 verbatim without validation. Token
pieces routinely split multi-byte characters (e.g. `"\xE2\x82"` of U+20AC),
so with `capture_text=true` the JSONL recording contains invalid UTF-8,
which RFC 8259 forbids and strict consumers (Observatory, Python
`json.loads` on decoded text) reject. Repro:
`make_envelope("token.generated", ctx, {{"text", "\xE2\x82"}}, 0)` -> raw
`E2 82` bytes in the line. (The fuzz targets assert only
"valid UTF-8 in -> valid UTF-8 out", which holds, so this does not fail CI.)
**Fix:** either buffer incomplete UTF-8 sequences between token events in the
producer, or make `append_escaped` replace invalid sequences with `\ufffd`.

### B6. `parse_stream_line` and `StreamDecoder` disagree, `src/backends/ollama/ollama_protocol.cpp`

Two Ollama line parsers with different rules:

| line | `parse_stream_line` | `StreamDecoder` |
|---|---|---|
| `[1,2]` | ok (empty piece) | protocol_error |
| `"text"` | ok (empty piece) | protocol_error |
| `{"error":null,"response":"a"}` | backend_error `ollama: null` | ok, piece `a` |

**Fix:** make `parse_stream_line` reject non-objects and ignore
`"error": null`, or implement it on top of the decoder's `on_line` logic.

### Minor observations

All three are addressed on main: the first two by #15 (`1e999` is rejected as
"number out of range"; Markdown cells are escaped), the third by the
integration follow-up after #13.

* JSON numbers outside double range (`1e999`) parse to `inf` and serialize
  as `null`, so `parse(dump(x))` changes type. Consider rejecting them in the
  parser.
* `render_markdown` inserts raw strings into Markdown tables; a `|` or
  newline in a model/label field breaks the table.
* `tests/CMakeLists.txt` `sonder.cli.rejects_bad_sampling` is `WILL_FAIL`, so
  a sanitizer abort in that path would also "pass". Checked manually: it exits
  2 with the expected message and no sanitizer report. Prefer
  `PASS_REGULAR_EXPRESSION "temperature must be"` over `WILL_FAIL`.
  **Done on main:** all CLI negative tests now use
  `PASS_REGULAR_EXPRESSION` on the error text instead of `WILL_FAIL`.

## Findings from the second fuzz round (anthropic_request, http_client, llamaserver_text, backend_config)

No crashes, hangs or sanitizer reports in any of the four targets (120 s
each, see the run summary in the PR that added them). Two invariant
deviations and two observations were found. **All four are fixed (this
branch)**; the harnesses assert the strict invariants again.

### F1. `parse_child_metrics` returns partial counters for an invalid payload, `src/backends/llamaserver/child_metrics.cpp`

**Status: fixed (this branch).** `parse_child_metrics` resets the result to a default value on every invalid path.

`valid == false` does not mean "empty": values parsed before the bad line stay
in the result. Input `llamacpp:prompt_tokens_total 10\nllamacpp:requests_processing -1\n`
yields `valid == false` with `prompt_tokens_total == 10`. `parse_child_slots`
does clear its result on error. Consumers are safe today:
`ChildMetricsSampler::sample_at` (same file, around the `metrics_valid`
check) returns before publishing a snapshot when `valid` is false. A future
caller that reads the counters without checking `valid` would see a half
payload. **Fix (done):** `parse_child_metrics` now resets the result to a
default `ChildMetricCounters` on every invalid path, and the harness asserts
"invalid implies default" again.

### F2. "unknown field" config errors echo the whole key, `src/server/src/residency_config.cpp:31`, `src/server/src/warmup_config.cpp:18`, `src/server/src/backend_setup.cpp`, `src/server/src/launch_profile.cpp`

**Status: fixed (this branch).** The echoed key is cut to 64 bytes on a UTF-8 boundary with `...` appended (`src/server/src/config_text.hpp`).

`parse_residency_config` and `parse_warmup_config` put the unknown member name
into the Status message unbounded, so a config with a 1 KiB+ key yields a
message over 1 KiB (a 990-byte key is enough). The GGUF reader already cuts
file-controlled text in its errors for the same reason. Reproducer: `{"` +
990 x `k` + `":1}` as the document (also under `residency` or `warmup`).
**Fix (done):** the echoed text is cut to 64 bytes in both messages, and in
every other message that echoes a caller-supplied key or value:
the top-level, `tls`, `spill_guard` and `stall_guard` unknown-field errors and
the unknown backend name in `backend_setup.cpp`, and in `launch_profile.cpp`
the unknown-field errors (profile and file level), the profile name used as
the error location, the unknown speculative type, the duplicate profile name
and the flag echoed by the `extra_args` refusals. The harness now requires
every error message to be under 1 KiB with no exception
(`fuzz/fuzz_backend_config.cpp`).

### Observations

* **Fixed (this branch).** `parse_warmup_config` mutates `setup` before it finishes validating (e.g. a
  NUL-containing `messages_file` is stored, then rejected), so a rejected
  config leaves a partly applied setup. `parse_residency_config` is atomic.
  Callers appear to discard the setup on error.
* **Fixed (this branch).** `llamaserver::parse_timings(Value, Timings&)` does not reset `out` on
  error (the `BackendTimings` overload does).

### Not covered

* The SSE `EventDecoder` in an unnamed namespace in
  `src/backends/llamaserver/llamaserver_backend.cpp` (reaching it would need a
  production code move).
* Real sockets, TLS and the response head parser of `http_request()`.
* `ChildMetricsSampler` state machine, `LogTail` file reads, `check_kv_cache_pairing`.
* Anthropic request fields beyond what `parse_messages_request` reads (tool
  use, images are rejected by design).

## Local run summary (box, clang 19.1.7, 4 build jobs)

* ASan+UBSan: build clean with `-Werror`, 251/251 tests passed, CLI smoke ok.
* TSan: 251/251 passed; 5 repeated runs passed.
* Fuzz: all 5 targets 60 s each, no crashes (json 2.17 M execs, bench_report
  1.98 M, bench_corpus 1.40 M, ollama_stream 0.69 M, telemetry_envelope
  0.52 M).
* In-tree `SONDER_BUILD_FUZZERS=ON` build (with the root snippet above
  applied to a scratch copy): builds, `fuzz-regress` ok, 251/251 tests pass.
* GCC `SONDER_FUZZ_ENGINE=standalone`: builds, seed replay ok.
