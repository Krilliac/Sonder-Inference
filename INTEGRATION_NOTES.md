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
   `TelemetryOptions::role` (default `inference`) and
   `std::optional<bool> synthetic` (default unset), emitted as
   `producer.role` / `producer.synthetic`; `producer.synthetic` is omitted
   while unset, so hosts that do not set it (CLI, C ABI, bench) never label
   mock output `synthetic: false`;
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
- CI (`.github/**`, integrator), once the CLI dispatch above has landed (it
  cannot run before): add a smoke step to `ci.yml` and the hardening ASan
  job: start `sonder-infer serve --backend mock --port 0 --ready-file
  $RUNNER_TEMP/ready.json`, wait for the file, poll `url` +
  `/v1/sonder/health` until 200 (the file means "listening"; models may
  still be loading), then SIGINT, expect exit 0 and the ready file gone.
- The `ci-windows` job must pass before merge: the Winsock paths could not be
  run in the Linux container.
- `docs/CLI.md` and `README.md` (not touched here) should mention `serve` and
  link docs/SERVER.md once the CLI dispatch lands.

## Open questions (recorded, not decided here)

- ADR-020 is marked proposed: the contract review asks for owner sign-off
  in all three repositories before it is accepted.
- Cancelling a model load: `Backend::load_model()` takes no cancellation
  token, so a shutdown during startup waits for the backend call in progress
  (bounded by the backend's own timeouts; Ollama metadata calls are capped
  at 60 s). A second SIGINT/SIGTERM exits at once. A cancellable load would
  need an additive `Backend`/`Engine::load_model` overload with a
  `CancellationToken`, which is a core API decision.
- Labelling CLI, C ABI and bench recordings synthetic: those hosts now omit
  `producer.synthetic` (unknown). Having the engine derive it (for example
  when only the mock backend is registered) would badge them in Observatory,
  but events emitted before registration would stay unlabelled; left for
  the core owners.
- The engine scheduler's 429 path is unreachable with the current request
  runtime (see docs/SERVER.md, Errors). If the runtime starts setting
  sequence fingerprints or stops clamping `max_new_tokens`, add an HTTP test
  for it.

# Integration notes: `eco/inf-cli-ux` (CLI UX and serve/chat dispatch)

Branch `eco/inf-cli-ux` builds on `eco/inf-serve`. Lane files:
`tools/sonder-infer/main.cpp`, `src/cli/**` (new header-only
`cli_spec.hpp`, `cli_values.hpp`, `cli_env.hpp`,
`sonder_infer_commands.hpp`), `bench/tools/sonder_bench.cpp`,
`tests/test_cli_*.cpp`, `tests/CMakeLists.txt`, the CLI smoke step of
`.github/workflows/ci.yml`, `README.md`, `docs/CLI.md` (new) and
`docs/integration/chat-cli.md`.

## Wiring

- `sonder-infer serve ...` calls
  `sonder::inference::server::serve_main(args_after_serve, std::cout,
  std::cerr)` under `SONDER_HAS_SERVER`; `help serve` calls it with
  `--help`. Without the module, `serve` exits 2 with a clear message.
- `sonder-infer chat` runs every turn through `Session::chat` under
  `SONDER_HAS_SESSION_CHAT` (request telemetry with `kind = "chat"`).
- Backends are built with the shared `make_backend()` and the environment is
  read with `backend_env_defaults()` (`sonder/inference/backend_setup.hpp`)
  when `SONDER_HAS_SERVER` is defined, so `serve` and the other commands
  agree; `src/cli/cli_env.hpp` holds an equivalent fallback for builds
  without the module, and `tests/test_cli_spec.cpp` checks both agree.

## Changes outside the lane's file list

1. `src/server/src/serve_main.cpp` (the one narrow edit of serve's argument
   parser that the contract review allows): an unknown option now uses
   `sonder::cli::unknown_option_message()`, e.g.
   `unknown option --prot (did you mean --port?)`. The table and all other
   behaviour are unchanged.
2. `bench/include/sonder/inference/benchmark.hpp` and
   `bench/src/benchmark.cpp` (module `bench`, additive): `bench::Options`
   gains `run_id`, `agent_id`, `task_id`, `workload` and `priority`, applied
   to every benchmark session, so `sonder-infer bench` can honour the
   correlation flags (lane scope item 6). `run_id` still defaults to
   `label`.
3. `tests/cli_expect.cmake` (new CTest driver for the exact-exit-code CLI
   smoke tests) and `tests/fixtures/chat_repl_input.txt` (REPL input).

## Stale statements in docs owned by `eco/inf-serve` (not edited here)

The lane may not edit these files; the integrator should update them when
merging, because the dispatch now exists:

- `docs/SERVER.md` lines 3-9: drop "The `sonder-infer serve` command line is
  not dispatched yet ... prints `unknown command serve` ...".
- `docs/SERVER.md` "Running it", the paragraph starting "The `serve`
  subcommand is to be dispatched by the CLI": the CLI dispatches it
  (`tools/sonder-infer/main.cpp`).
- `docs/SERVER.md` "Open questions" 1-3: (1) serve keeps its table parser
  but shares the unknown-option wording and suggestions with
  `src/cli/cli_spec.hpp`; (2) the CLI now uses `make_backend()` and
  `backend_env_defaults()`; (3) the `ci.yml` CLI smoke step now starts
  `serve --port 0 --ready-file`, polls health and sends SIGINT (Linux job).
  The hardening workflow's ASan job has no serve smoke yet (`.github/**`
  outside the CLI smoke step is not in this lane).
- `docs/ROADMAP.md` Phase 1: tick "`sonder-infer serve` command-line
  dispatch".
- The "Left for other lanes" list above: the CLI dispatch, the shared
  backend factory, the `ci.yml` smoke and the `docs/CLI.md`/`README.md`
  items are done on this branch.

## Open questions

- `sonder-bench` keeps exit codes 3 (backend unreachable, load failed) and 4
  (`--require-idle` refused) rather than 1; see docs/CLI.md.
- `--ollama-allow-remote` and plain HTTP: the Ollama client
  (`src/backends/ollama/ollama_client.cpp` `make_request()`) checks only the
  host once `allow_remote` is set, so on its own it would send prompts over
  plain `http://` to a remote host. The CLI now refuses that combination with
  exit 2 for every command (`resolve_backend()`, and a pre-check in
  `cmd_serve()` before `serve_main()` runs), using
  `cli::is_plain_http_remote()` (unit-tested against `net::parse_url()` and
  `net::is_loopback_host()`). Direct library users of
  `OllamaBackendOptions::allow_remote` or `BackendSetup` are not covered.
  Owners of the Ollama module and `src/server` should decide whether
  `make_request()` (or `make_backend()`) enforces "https:// only for remote
  hosts" itself; both are outside this lane. `https://` needs
  `SONDER_WITH_TLS=ON`, which root CMake does not include
  (`cmake/SonderTls.cmake` is not wired), so the default build cannot reach a
  remote host at all.
- `serve` keeps its own parser (`src/server/src/serve_main.cpp`; this lane
  used its one allowed edit for the shared unknown-option wording). Its other
  errors have no `serve:` prefix and its numeric errors do not echo the bad
  value (`--port must be an integer from 0 to 65535`). `--help` now wins for
  `serve` as for the other commands because `cmd_serve()` checks for it
  first. Aligning the remaining messages is left to the server module's
  owner; docs/CLI.md "Option syntax" records the difference.
- `run_chat_repl()` keeps its signature but forwards to `run_chat_session()`,
  so it inherits that loop's commands and EOF handling: unknown `/word`
  lines are rejected instead of sent, commands match on the first word
  (`/exit now` exits), and EOF with a pending prompt prints a newline.
- The Windows paths of the CLI (`_isatty`, enabling
  `ENABLE_VIRTUAL_TERMINAL_PROCESSING` for colored REPL labels, CRLF output
  in the CTest driver) are written for MSVC but were not run in the Linux
  container; the `ci-windows` job must pass. The `serve` part of the CI smoke runs on Linux
  only (signal delivery from Git Bash to a native process is not reliable).

# Integration notes: `feat/launch-profiles` (typed launch profiles)

Everything lives in the server module (`src/server`); see
[docs/integration/launch-profiles.md](docs/integration/launch-profiles.md).

- **KV-cache lane interplay** (`feat/kv-cache-types-flash-attn-ubatch`). A
  llamacpp profile's `batch_size`, `ubatch_size`, `cache_type_k`,
  `cache_type_v` and `flash_attn` map onto that lane's `BackendSetup` fields
  (`llamacpp_batch_size`, `llamacpp_ubatch_size`, `llamacpp_kv_cache_type_k`,
  `llamacpp_kv_cache_type_v`, `llamacpp_flash_attention`) with the same value
  names. This branch does not add those fields; a C++20 `requires` check
  (`kLlamaCppContextOptionsAvailable` in `launch_profile.hpp`) enables the
  mapping when they exist and otherwise rejects the fields with "not
  supported by the llamacpp backend; use llamaserver". Either merge order
  compiles without conflict; the direct-backend test follows the flag.
- `threads`/`threads_batch` are rejected for llamacpp profiles although
  `LlamaCppBackendOptions::threads` exists, because `BackendSetup` has no
  field for it and adding one here would collide with the KV-cache lane's
  rewrite of `make_backend`. A follow-up can plumb it.
- `ServerOptions` gains `profile_bindings` and `profile_catalog` (additive;
  empty keeps every code path unchanged). `/v1/models` gains
  `data[].sonder.profile` and `sonder.profiles` only when a profile is served.
- Windows VRAM detection uses DXGI (`dxgi.lib` via `#pragma comment`, MSVC
  only); other platforms need `vram_budget_mib` or `--vram-budget-mib`.
