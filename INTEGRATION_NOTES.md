# Integration notes: `eco/inf-serve` (module `src/server`)

Branch `eco/inf-serve` implements ecosystem contract v1 sections 2, 5
(Inference side), 6.1 and 7.1: `sonder-infer serve` and the live telemetry
transport. The module itself lives in `src/server/`. This branch acts as the
explicit integrator for the changes below outside the module (the contract
review assigns them to this lane); please review them as core changes.

## Root and core changes made on this branch

1. Root `CMakeLists.txt`: `src/server` added to `SONDER_MODULE_DIRS` (one
   line).
2. `include/sonder/inference/session.hpp`, `src/sessions/session.cpp`:
   additive C++ API only.
   - `Session::chat(messages, on_chunk, sampling_override, RequestOptions)`
     runs chat through the request runtime (scheduler, KV accounting, Sonder
     sampler for non-native chat on `token_logits` backends) with
     `request.queued.kind = "chat"`; native chat backends get the messages,
     others the `format_chat_prompt()` prompt.
   - `Session::generate` gains a defaulted `RequestOptions` parameter.
   - `RequestOptions { request_id, parent_request_id }`;
     `parent_request_id` goes on the five request lifecycle events.
   - `Session::last_scheduler_rejected()`.
   - `#define SONDER_HAS_SESSION_CHAT 1`.
3. `include/sonder/inference/telemetry.hpp`, `src/telemetry/telemetry.cpp`:
   `TelemetryOptions::role` (default `inference`) and `synthetic` (default
   false), emitted as `producer.role` / `producer.synthetic`;
   `TelemetrySink::write_event(sequence, line)` (default forwards to
   `write`), which the bus writer thread now calls.
4. `include/sonder/inference/engine.hpp`, `src/engine/engine.cpp`:
   `EngineOptions::server` (`EngineServerInfo {host, port, api_version}`),
   reported on `engine.started` as `attributes.server`.
5. `fuzz/fuzz_http_request.cpp`, `fuzz/corpus/http_request/*`,
   `fuzz/CMakeLists.txt` (target list), `fuzz/README.md`: new fuzz target for
   the request head parser and chat request mapper. The hardening workflow's
   loop over `sonder_fuzz_*` picks it up.
6. Tests for the core changes live in existing core files:
   `tests/test_chat.cpp` (Session::chat, native chat, parent ids) and
   `tests/test_telemetry.cpp` (role/synthetic, write_event sequences,
   engine.started server). `tests/CMakeLists.txt` is unchanged.

`include/sonder_inference.h` and `src/engine/c_api.cpp` are unchanged; the C
ABI stays at 1. No new dependency.

## Left for other lanes / the integrator

- CLI dispatch (lane `inf-cli-ux`): `tools/sonder-infer/main.cpp` should
  route `sonder-infer serve ...` to
  `sonder::inference::server::serve_main(args_after_serve, std::cout,
  std::cerr)` behind `#if defined(SONDER_HAS_SERVER)` and list `serve` in the
  usage text. `serve_main` handles `--help`, signals and exit codes itself.
  The CLI's own backend construction can switch to
  `sonder/inference/backend_setup.hpp` (`make_backend`,
  `backend_env_defaults`).
- CI (`.github/**`, integrator): add a smoke step to `ci.yml` and the
  hardening ASan job: start `sonder-infer serve --backend mock --port 0
  --ready-file $RUNNER_TEMP/ready.json`, wait for the file, `curl` the `url`
  + `/v1/sonder/health` (expect 200), then SIGINT and expect exit 0.
- The `ci-windows` job must pass before merge: the Winsock paths could not be
  run in the Linux container.
- `docs/CLI.md` and `README.md` (not touched here) should mention `serve` and
  link docs/SERVER.md once the CLI dispatch lands.
