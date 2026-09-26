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

1. **Session/Engine chat.** The CLI calls `Model::backend_model().chat()`
   directly because `Session` has no chat entry point. Chat turns therefore
   emit no Observatory session/request telemetry, and `--telemetry` only
   records engine-level events for `chat`. Suggested: `Session::chat(messages,
   on_chunk, sampling_override)` mirroring `Session::generate`, then switch
   `cmd_chat` to it.
2. **C ABI.** `sonder_inference.h` has no chat entry yet (append-only when added).
3. README/docs CLI usage could mention `chat` and `--markdown` (only this
   notes file was edited).

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
