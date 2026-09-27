# fuzz/

libFuzzer targets for Sonder Inference's parsers and serializers. Owner:
`feat/hardening`. Details and findings: `docs/integration/hardening.md`.

| Target | Code under test | Invariants beyond "no crash / no UB" |
|---|---|---|
| `sonder_fuzz_json` | `json::parse` / `Value::dump` (`src/common/json.cpp`) | `dump()` re-parses and is a fixed point; single line; valid UTF-8 in -> valid UTF-8 out |
| `sonder_fuzz_ollama_stream` | `ollama::StreamDecoder` (NDJSON stream) and `ollama::parse_stream_line` | chunk-split invariance (any split == one-shot feed); `finish()` ok implies `done`; non-negative, non-NaN rates |
| `sonder_fuzz_bench_corpus` | `bench::parse_corpus` (what `load_corpus` calls) | accepted corpora have ids, prompts, in-range `max_tokens`, consistent fan-out prefix |
| `sonder_fuzz_bench_report` | `bench::render_markdown`, `bench::default_result_stem` | stem is filesystem-safe |
| `sonder_fuzz_telemetry_envelope` | `TelemetryBus::make_envelope` + dump (the JSONL recording format) | one line per event; round-trips schema/event_type/session_id/attributes; UTF-8 preserved |
| `sonder_fuzz_http_request` | `sonder-infer serve` front end: `parse_request_head`, query decoding, Host check, correlation headers, `parse_chat_request` (`src/server/src/http.cpp`, `openai.cpp`) | complete heads stay within 16 KiB / 64 headers with origin-form paths; errors are 400/431/505; accepted chat requests pass message and sampling validation; accepted correlation values match the pinned pattern |

There is no separate recording/replay serializer in the tree today; the
telemetry JSONL envelope is the recording format, so that is what is fuzzed.

## Build and run

Standalone (no root `CMakeLists.txt` change; this is what CI does):

```sh
CC=clang CXX=clang++ cmake -S fuzz -B build/fuzz -G Ninja
cmake --build build/fuzz
cmake --build build/fuzz --target fuzz-regress          # replay seeds once
cmake --build build/fuzz --target fuzz-run-ollama_stream # 60 s exploration
./build/fuzz/sonder_fuzz_json -max_total_time=300 build/fuzz/work/json fuzz/corpus/json
```

In-tree, once the root has `add_subdirectory(fuzz)` (after the module loop):

```sh
cmake -S . -B build/fuzz -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSONDER_BUILD_FUZZERS=ON
```

Options: `SONDER_BUILD_FUZZERS` (default OFF in-tree, ON standalone),
`SONDER_FUZZ_ENGINE` (`libfuzzer` with Clang, else `standalone`, which links
`standalone_main.cpp` and just replays files: usable with MSVC/GCC as a
regression runner), `SONDER_FUZZ_SANITIZERS` (default `address,undefined`),
`SONDER_FUZZ_SECONDS` (default 60, used by `fuzz-run-*`).

## Corpora

`corpus/<target>/` holds small hand-written seeds (Ollama seeds are derived
from `src/backends/ollama/tests/fixtures`). `corpus/.gitattributes`
disables line-ending normalization so seeds stay byte-exact. Input formats:

* `ollama_stream`: first byte is a control byte (bit 0: 0 = `/api/generate`,
  1 = `/api/chat`; bits 1..7 seed the chunk-split pattern), the rest is the
  raw response body. Seeds use printable control bytes (`0`, `1`, `g`, ...).
* `http_request`: raw client bytes (request head, then the body used as a
  chat completion request).
* `telemetry_envelope`: up to 7 newline-terminated fields (event_type,
  session_id, run_id, request_id, agent_id, task_id, model_instance_id), then
  a JSON document used as the event attributes.

Add a minimized reproducer here (`-minimize_crash=1`) whenever a fuzz bug is
fixed, so `fuzz-regress` keeps covering it.
