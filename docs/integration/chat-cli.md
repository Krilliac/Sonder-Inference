# Integration notes: `feat/chat-cli`

Scope: chat on the Backend interface, backend chat implementations (mock via
the default, Ollama native, llama.cpp chat template), the `sonder-infer chat`
subcommand, `sonder-infer bench --markdown`, and their tests. No root CMake,
preset, CI, Engine/Session or `SamplingConfig` file was touched.

## Backend interface (`include/sonder/inference/backend.hpp`)

Additions only; existing backends compile unchanged.

- `ChatMessage { role, content }`, `ChatRequest { request_id, messages, sampling }`.
- `is_known_chat_role()`: `system | user | assistant | tool`.
- `validate_chat_messages()`: non-empty, known roles, and the last message is
  `user` or `tool`. Otherwise it returns `invalid_argument`.
- `format_chat_prompt()`: backend-neutral flattening
  (`System: …\n\nUser: …\n\nAssistant: …\n\n…Assistant:`).
- `virtual BackendModel::chat(const ChatRequest&, const CancellationToken&, const TokenCallback&)`.
  It has the same streaming, cancellation and callback contract as `generate()`.
  **Default:** validate, then `format_chat_prompt()`, then `generate()`. The mock
  backend (and any future backend) gets chat this way.
- `virtual bool BackendModel::has_native_chat() const` is `false` by default.
  The CLI reports it.
- Implementations live in `src/backends/backend.cpp`, which is already in the
  core source list.

## Backends

| backend | chat path | has_native_chat |
| --- | --- | --- |
| mock | default (formatted prompt, then `generate`) | no |
| ollama | `POST /api/chat` via the existing `OllamaClient::chat`; server applies the model template | yes |
| llamacpp | `llama_chat_apply_template` with the GGUF `tokenizer.chat_template`; prompt tokenized with `parse_special=true`. Falls back to the default when the model has no template or llama.cpp does not recognise it | yes when the template applies |

- Ollama: `ollama::ChatMessage` is now an alias of the core `ChatMessage` (same
  fields, source compatible). `generate()` and `chat()` share one
  stream-to-`GenerateStats` mapping (`OllamaModel::run`). Messages are
  validated client-side before any request is sent.
- llama.cpp wrapper (`sonder/backends/llamacpp/llamacpp_backend.h`) gains
  `ChatTurn`, `ChatTemplate()`, `ApplyChatTemplate()`, the static
  `FormatChat(template, …)` (works without a model), and a defaulted
  `parse_special` parameter on `Tokenize()`. `llama_chat_apply_template` is
  llama.cpp's built-in template matcher, not a Jinja engine, so exotic
  templates use the generic fallback.

## CLI

- Argument parsing moved to header-only `src/cli/cli_args.hpp`
  (`sonder::cli`). `tools/sonder-infer/main.cpp` includes it by relative path,
  so no new target is needed. The unit tests include it as `cli/cli_args.hpp`.
  Boolean flags: `--capture-text`, `--markdown`.
- `sonder-infer chat --backend B --model M [--system TEXT] [sampling options]`
  - `--messages FILE`: one-shot. FILE is a JSON array of `{role, content}` or
    `{"messages": [...]}`. The reply streams to stdout and a stats line goes to
    stderr (`chat native=… messages=N stop=… prompt_tokens=… completion_tokens=…`).
  - Without `--messages`, chat is interactive on stdin: each line is a user
    turn and history is kept. `/reset` clears it (the system prompt stays),
    and `/exit`, `/quit` or EOF ends the chat. Ctrl-C cancels the current turn.
- `sonder-infer bench … --markdown` renders `bench::render_markdown()` to stdout
  and writes it next to `--out` (`x.json` becomes `x.md`). The JSON output is unchanged.

## Integrator follow-ups (not done here; they touch lead-owned files)

Follow-up 1 is resolved: `Session::chat` landed with eco/inf-serve and the
CLI uses it since eco/inf-cli-ux (see [Update](#update-ecoinf-cli-ux) below).

1. **Session/Engine chat.** The CLI calls `Model::backend_model().chat()`
   directly because `Session` has no chat entry point. Chat turns therefore
   emit no Observatory session/request telemetry, and `--telemetry` only
   records engine-level events for `chat`. Suggested: `Session::chat(messages,
   on_chunk, sampling_override)` mirroring `Session::generate`, then switch
   `cmd_chat` to it.
2. **C ABI.** `sonder_inference.h` has no chat entry yet (append-only when added).
3. README/docs CLI usage could mention `chat` and `--markdown` (only this
   notes file was edited).

## Integration on main

Merged after `feat/sampling-config` (#9), the engine wiring (#11) and the
sampling follow-ups. Conflicts were additive (`backend.hpp`: `TokenStream`,
`tokenize()`, `open_token_stream()` next to the chat API;
`tests/CMakeLists.txt`). Added during integration:

- Ollama `chat()` rejects a non-empty `logit_bias` with `invalid_argument`,
  like `generate()`.
- Sampling flags for the new `SamplingConfig` fields on `generate`, `chat`
  and `bench` (see `sonder-infer --help` and
  [sampling-config.md](sampling-config.md)); bench results record them.
- Follow-up 1 (Session/Engine chat) was still open at that point; it is
  resolved by the update below.

## Tests

- `tests/test_chat.cpp` (`sonder.core.chat.*`, 6 cases): role validation,
  prompt format, mock default path == `generate(format_chat_prompt())`,
  invalid messages, callback stop, cancellation, injected backend error, and
  stop sequences.
- `tests/test_cli_args.cpp` (`sonder.core.cli.*`, 8 cases): chat/bench arg
  parsing, `--markdown` flag, parse errors, message-file parsing, system
  prompt, `run_chat_turn` and the interactive loop (scripted and on the mock).
- `src/backends/ollama/tests/ollama_chat_tests.cpp` (`sonder.ollama.backend chat*`,
  4 cases), all against the existing fake Ollama server: `/api/chat` body
  (messages, options, no prompt), streamed content, stats mapping, thinking
  chunks, callback stop, cancellation (including a stalled stream), client-side
  validation, and server error.
- `src/backends/llamacpp/tests/test_llamacpp_chat.cpp` (`sonder.llamacpp.llamacpp chat*`,
  4 cases, only with `SONDER_WITH_LLAMA_CPP=ON`): chatml formatting, buffer
  growth, invalid inputs, vocab-only GGUF template lookup and `parse_special`.
- CTest CLI checks in `tests/CMakeLists.txt`: `sonder.cli.chat_messages_mock`,
  `sonder.cli.chat_rejects_bad_role`, `sonder.cli.chat_rejects_missing_file`,
  `sonder.cli.bench_markdown_mock`. Fixtures are in `tests/fixtures/`.

Local results (GCC, `ci-linux` preset with `-Werror`, Ninja `-j4`): 273/273
passed, 22 of them new. Clang with `-Wshadow -Wconversion -Wsign-conversion`
is clean and 273/273 pass. With `SONDER_WITH_LLAMA_CPP=ON` (b11195): 294/294
pass, and the integration test was skipped. A manual run of `sonder-infer chat
--backend llamacpp` with `stories15M-q4_0.gguf` (no chat template) exercised
the generic fallback. MSVC was not built locally; CI covers it.

## Update: eco/inf-cli-ux

The CLI reference is now [docs/CLI.md](../CLI.md). Changes to `chat`:

- **Chat through the session.** When the library defines
  `SONDER_HAS_SESSION_CHAT`, every turn (one-shot and REPL) runs through
  `Session::chat` on one session per invocation. Chat therefore goes through
  the engine scheduler and KV accounting, and `--telemetry` records
  `request.queued` (`attributes.kind = "chat"`), `request.started`, the decode
  events and `request.completed`/`cancelled`/`failed`. The correlation flags
  (`--run-id`, `--agent-id`, `--task-id`, `--workload`, `--priority`) set the
  session metadata. Without `SONDER_HAS_SESSION_CHAT` the old direct
  `BackendModel::chat()` path is kept (no request telemetry).
- **REPL.** `run_chat_session()` (`src/cli/cli_args.hpp`) adds `/help`,
  `/stats` (session totals and the last turn's stats), `//text` to send a
  leading `/`, and rejects any other `/word` without sending it. Role labels
  (`you> `, `assistant> `) appear on a terminal, colored only when stdout is
  a terminal that takes ANSI sequences (on Windows the CLI switches on
  virtual-terminal processing and leaves colors off if that fails) and
  `NO_COLOR` is unset. With piped stdin there is no input
  prompt, and the output never ends in a dangling prompt. A cancelled turn
  (Ctrl-C) drops the user message and the partial reply; the REPL continues.
  `run_chat_repl()` keeps its signature, but it now forwards to
  `run_chat_session()`, so its commands and EOF handling follow that loop:
  a `/word` line other than the known commands is rejected instead of sent,
  commands match on the first word (`/exit now` exits), and EOF with a
  pending prompt prints a newline.
- **Stats.** One-shot chat prints
  `[sonder-infer] chat native=no messages=4 outcome=completed stop=... ` by
  default; `--stats json` prints one JSON object instead, and the REPL prints
  per-turn stats only with `--stats text|json`.
- **Exit codes.** An unreadable or invalid `--messages` file exits 2, a
  backend failure 1, a cancelled one-shot turn 130.

Tests: `tests/test_cli_repl.cpp` (`sonder.core.cli repl: *`) covers the REPL
commands, labels and colors, per-turn stats, failed and cancelled turns, and
turns through `Session::chat` emitting `kind = "chat"` request telemetry. The
CTest cases `sonder.cli.chat_telemetry`, `sonder.cli.chat_repl`,
`sonder.cli.chat_repl_stats_json` and `sonder.cli.chat_correlation` run the
binary.
