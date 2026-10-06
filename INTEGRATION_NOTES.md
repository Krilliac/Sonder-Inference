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

- **KV-cache options (#45).** A llamacpp profile's `batch_size`,
  `ubatch_size`, `cache_type_k`, `cache_type_v` and `flash_attn` set #45's
  `BackendSetup` fields (`llamacpp_batch_size`, `llamacpp_ubatch_size`,
  `llamacpp_kv_cache_type_k`, `llamacpp_kv_cache_type_v`,
  `llamacpp_flash_attention`) with the same value names; the compile-time
  switch that waited for those fields is gone now that they exist. With
  `--profile`, `serve` refuses #45's five flags (the profile's typed fields
  own them). Without a profile, the five flags now apply to `--backend
  llamacpp` only: any other backend exits 2 with "option --X applies only to
  --backend llamacpp; for llamaserver set it in the JSON args" instead of
  ignoring them (the GPU lane's finding; default llamacpp behaviour is
  unchanged).
- **Model residency (#43).** The per-model binding (load name, sampling
  defaults, `/v1/models` metadata) is looked up from
  `ServerOptions::profile_bindings` by served id. `load_served_model()` uses
  the load name, so eager, lazy and post-eviction loads all load a llamacpp
  profile's GGUF path; sampling defaults apply after the residency resolves
  `default` to the served id, for `/v1/chat/completions` and `/v1/messages`.
- `threads`/`threads_batch` are rejected for llamacpp profiles although
  `LlamaCppBackendOptions::threads` exists, because `BackendSetup` has no
  field for it. A follow-up can plumb it.
- **Overlap with `sonder-infer tune` (#44/#51).** tune writes a spawn config
  (`tuned.json` with raw `args`) and has its own GGUF header reader
  (`tune_model.cpp`); launch profiles are a typed schema with their own
  reader and a fit check. Key names differ (tune's `no_kv_unified` vs
  `kv_unified`, `cache_ram` vs `cache_ram_mib`), and a tuned config's `args`
  cannot be combined with `--profile` (it refuses config `args`). Unifying
  the two readers or schemas is an owner decision, not done here.
- `ServerOptions` gains `profile_bindings` and `profile_catalog` (additive;
  empty keeps every code path unchanged). `/v1/models` gains
  `data[].sonder.profile` and `sonder.profiles` only when a profile is served.
- Windows VRAM detection uses DXGI (`dxgi.lib` via `#pragma comment`, MSVC
  only); other platforms need `vram_budget_mib` or `--vram-budget-mib`.
- **Runtime status interplay** (#35, VRAM-spill guard). Rebased onto it:
  `data[].sonder` carries `runtime` (from #35, when the backend reports it)
  and `profile` (from this branch) side by side. When the running context
  differs from the profile's (spill guard `auto_fit`), the served profile's
  `context_length` is the running value and `configured_context_length` the
  profile's. `capabilities` lists only what Sonder's endpoint accepts;
  `vision`/`tools` are in `upstream_capabilities` because
  `/v1/chat/completions` rejects tool definitions and image content.

## Closeout follow-up: ignored direct-backend flags

The server module now refuses `--gpu-layers`, `--context-length`,
`--moe-experts` and `--tensor-override` for every backend except `llamacpp`,
alongside the existing batch/KV/flash-attention flag rule. Previously these
four options populated only direct llama.cpp settings and were silently
ignored by mock, Ollama and llamaserver. The refusal is a usage error (exit
2) before model loading or server startup. `--device` remains a valid
telemetry/health label for every backend, and llamacpp profiles retain their
command-line placement options.

Outside the module, `docs/integration/launch-profiles.md` now documents the
complete backend-only flag list. The public `serve_main` regression checks
all nine options against mock, Ollama and llamaserver and confirms the
llamacpp path accepts them.

The qualification pass also found a GCC 14.2 warnings-as-errors build failure
in the existing llamaserver tune report's conditional JSON-value initializer.
`src/backends/llamaserver/tune.cpp` now initializes the optional counters as
null and sets available values explicitly. This preserves JSON field order,
unsigned counts, zero acceptance and unavailable ratios; the existing tune
serialization tests exercise those distinctions. No warning is disabled.

The integrator also adds the stdlib-only POSIX runner `scripts/stress_mock.py`
and its reproduction guide `docs/integration/stability-smoke.md`. It runs
bounded mock-only streaming/nonstreaming fan-out and drains active streams
on SIGINT, with generated receipts kept outside Git.

## SDK request parent integration — 2026-10-05

The parent-only SDK slice crosses module boundaries explicitly: append-only
C header and core ABI wrapper; Python mirrors/loader/API/exports; native C11,
C++ and Python tests; three bounded qualification/fixture scripts; public
documentation and the integration note at
`docs/integration/cabi-request-parent.md`. Native RequestOptions, session/run
ownership, HTTP server, scheduler/KV, telemetry schema and dependencies are
unchanged. Default symbols remain usable, optional metadata symbols require
explicit support, and the broader scheduling/request/callback item remains
open. Root integration owns publication and exact-revision qualification.

## CI follow-up: real mock lifecycle stress

The existing Linux `build-test` context now runs `scripts/stress_mock.py`
against its newly built CLI for three cycles, replacing the redundant idle
health/SIGINT subsection of the CLI smoke. CTest and all CLI/telemetry
assertions remain. The bounded step verifies 204 synthetic requests,
stream usage/termination, twelve active-stream drains, process exit and
ready-marker removal; it publishes JSON receipts and console logs without
masking the harness exit status. No required context is added or renamed.
The same step runs two stdlib CLI-boundary failure controls, including a
real completed cycle followed by a failed restart.

Outside CI, the existing script now preserves completed-cycle evidence in
a failure receipt, requires ready health HTTP 200, and checks health reports
four admitted model requests with all four futures active after first-token
admission, sampled sequentially immediately before sending SIGINT. Those streams use
16 tokens (the mock fixture's maximum); CI explicitly uses a 40 ms mock
delay to give the signal a longer bounded tail inside the existing two-second
grace. This remains mock transport/lifecycle qualification,
not a native model quality or throughput measurement.

## GGUF routed-expert header observations — bounded follow-up

Provisional scope is the `src/server` module under root integration. The
additive C++ header value, parser and synthetic tests stay in:

- `src/server/include/sonder/inference/launch_profile.hpp`
- `src/server/src/vram_estimate.cpp`
- `src/server/tests/test_launch_profile.cpp`

Cross-area documentation changes for the integrator are additive updates to
`docs/PLACEMENT.md`, `docs/ROADMAP.md` and this integration note. No root
CMake, common backend interface, `ModelDescriptor`, C ABI, Python, telemetry,
CLI, dependencies or cache/scheduler/session code changes are needed.

The optional count pair is a header-byte observation of original scalar
integer metadata and exact unique namespace keys. Missing/invalid values
stay unset; model names and array maxima supply no evidence. Existing cache
architecture, cursor/cancellation/rollback behavior, VRAM estimates and
manual expert placement remain unchanged. This does not implement a planner,
backend artifact identity or Runtime task-quality classification. Exact
same-artifact revision binding must precede any later policy consumption or
backend descriptor propagation.

This slice does not replace the separately retained callback-exception
rollback candidate or merged Python close synchronization. Root integration
owns application, exact-revision qualification and publication; no new lane
ownership or live model qualification is claimed here.

## C++ token callback exception lifecycle recovery

Provisional sessions scope: `src/sessions/session.cpp`. Cross-area integrator
changes are the public comment in `include/sonder/inference/session.hpp`,
controls in existing `tests/test_session.cpp`, `tests/test_engine_runtime.cpp`
and `tests/test_c_api.cpp`, and the additive
`docs/integration/request-callback-exceptions.md`. No root CMake, C ABI layout,
backend execution, scheduler/cache policy or Python changes are needed.

This follows the separately retained callback recovery audit after the merged
GGUF header observation slice. It clears internal request latches on ordinary
callback exceptions and attempts acquired-id finalization once, preserving
closed state and the original exception. It does not replay delivered effects,
change Runtime journal/rollback ownership, or claim allocator/backend recovery.
The isolated Linux Debug build and all 945 native tests passed, including the
six new recovery cases. A separate mock driver reproduced two baseline defects
and completed 512 concurrent failure/reuse pairs plus two closed controls, with
once-only terminal/release evidence and zero final logical resources. Normal
cost retains 2,048 raw measurements across both binaries; observed candidate
p95 times increased, so no speedup or production overhead bound is claimed.
See the integration document for exact receipts, timings and supported limits.
Publication still follows required checks on the exact public revision.

## llama-server prefill cancellation observation

Test-only scope: `src/backends/llamaserver/tests/fake_server.hpp` and
`test_backend.cpp`; this integration note is the only cross-module change.
The fake's request-body publication precedes response headers and httplib's
provider admission. Cancelling at that point can bypass its write-based
disconnect observer. A bounded provider-entry barrier now establishes that
headers have been sent, holds token payload until cancelled generation has
returned, and then observes peer closure through the actual sink peer/write
checks. Gate timeout and normal provider completion never count as closure.

Controls cover a live held peer and pre-cancelled generation, plus twenty
serial and sixteen requests across two joined workers with alternating
scheduling yields. Assertions run after joins and check cancelled status,
zero token callbacks and exactly one observed closure per cancelled request.
The existing three-second entry wait and one-second observation window remain
bounded. Production transport/backend code, privacy, batching, cursor,
recovery and rollback contracts are unchanged. This repairs an observer blind
spot; the precise branch taken in the earlier main ASan failure is unknown.
Native and hosted qualification are pending; loopback fake traffic provides
no provider/model quality or throughput evidence.

Subsequent local qualification on 2026-10-06 passed the offline GCC 14 Linux
Debug configure, build, discovery, focused controls, full CTest and diff gates.
All three new controls passed in 3.52 seconds; all 962 native tests passed with
zero failures, errors or native CTest skips in 70.91 seconds. The repetition
case passed 180 assertions in each focused/full invocation, and the retained
case passed five: 73 synthetic cancellation observations total. Supervisor
cleanup reached ECHILD without cleanup signals; deliberate fixture child
statuses remain recorded. Old source/build/raw scopes and cached dependencies
were preserved. Independent source and completed native data reviews passed.

Native qualification SHA-256:
`33c49b611e43e89c0d2316fd9e05faf624ecb0c2a0ec16ed952d6eb24c3c92fc`.
Independent data review SHA-256:
`d78df41e2ac86ed809b74563568a0f8a0d098bd2241dd2d1508bf53068490fcd`.
The tested code is a frozen uncommitted overlay on `d8d02d3ae8d5`; its build
stamp remains that base, with TLS and llama.cpp disabled. This later notes
append does not restamp those executions as a published revision build.
Hosted checks and merging still require the exact final public revision.
