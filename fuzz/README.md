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
| `sonder_fuzz_gguf` | launch-profile GGUF header reader `parse_gguf_model_info` and `estimate_vram` on what it accepts (`src/server/src/vram_estimate.cpp`) | errors are `invalid_argument` "GGUF header: ..." under 1 KiB (file text cut short; they reach `/v1/models`); accepted headers have a printable architecture of at most 256 bytes, 1..100000 blocks and per-block vectors of that length; the estimate's total is the sum of its parts |
| `sonder_fuzz_anthropic_request` | `/v1/messages` front end: `parse_messages_request`, `reasoning_budget::parse_body`/`parse_integer`, `AnthropicStream` (`start`/`token`/`finish`), `anthropic_message_response` (`src/server/src/anthropic.cpp`, `reasoning_budget.cpp`) | errors are 4xx with a code and message and a re-parsing error body; accepted jobs pass `validate_chat_messages`/`validate`, have an explicit `max_tokens`, at most 16 stop sequences and a 1..256 byte session key; every SSE frame is valid UTF-8 `event:`/`data:` with a re-parsing payload whose `type` equals the event, follows the block grammar with consecutive indices, and (for valid UTF-8 input) the deltas add up to the input |
| `sonder_fuzz_http_client` | `net::parse_url`, `is_loopback_host`, `ChunkedDecoder`, `LineSplitter` (`src/net/http_client.cpp`) | chunk-split invariance (output, error, `done()`, `stopped()`); decoded body never longer than the input; accepted URLs have scheme http/https, non-empty host, port 1..65535, normalised path; `is_loopback_host` equals a reference oracle (localhost, `::1`, `[::1]`, strict 127/8) |
| `sonder_fuzz_llamaserver_text` | `parse_child_metrics`, `parse_child_slots`, `derive_child_speculation`, `LogDiagnostics`, `parse_spill_policy`, `map_finish_reason` (`src/backends/llamaserver`) | gauges finite and non-negative; `valid == false` leaves the result empty; log diagnostics are chunk-split invariant and bounded; policy and finish-reason tables match their documented values |
| `sonder_fuzz_backend_config` | `parse_residency_config` + `apply_residency_config`, `parse_warmup_config`, `load_llamaserver_extensions`, `llamaserver::parse_timings` (both overloads), `parse_props` (`src/server/src`, `src/backends/llamaserver`) | error messages non-empty and under 1 KiB; a rejected residency or warmup config leaves the setup unchanged; accepted configs satisfy their documented bounds and map exactly to the guard options; timings/props counts are consistent |

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
* `gguf`: the first bytes of a GGUF v3 file (magic, counts, metadata, tensor
  table). Seeds are small synthetic headers: a qwen35 hybrid with ssm.* keys
  and attention tensors, a llama header with per-layer KV head counts and an
  MTP block, and a truncated copy.
* `anthropic_request`: the raw JSON body of `POST /v1/messages`; the same
  bytes also drive the reasoning-budget parser and, cut at UTF-8 boundaries,
  the streaming encoder's text and reasoning chunks.
* `http_client`: byte 0 is a control byte (bits 0..1: stop the line splitter
  at that line, 0 = never; bits 2..7 seed the chunk-split pattern), then one
  URL line (up to the first `\n`), then a raw response body fed to the chunked
  decoder and the line splitter.
* `llamaserver_text`: byte 0 is a control byte (seed for the log chunk-split
  pattern), the rest is given to every parser as Prometheus text, `/slots`
  JSON, child log text, a spill policy name and a finish reason.
* `backend_config`: raw JSON text, given to the residency, warmup, tune
  extension, timings and `/props` readers; the same bytes also build a raw
  residency setup for `apply_residency_config`.

Add a minimized reproducer here (`-minimize_crash=1`) whenever a fuzz bug is
fixed, so `fuzz-regress` keeps covering it.
