# `sonder-infer` and `sonder-bench` command-line reference

Status: implemented (`tools/sonder-infer/main.cpp`, `bench/tools/sonder_bench.cpp`;
shared header-only helpers in `src/cli/`). This page is the reference for the
command line: commands, option syntax, environment defaults, exit codes and
the JSON shapes the commands print. `sonder-infer help <command>` prints the
same option lists from the same tables (`src/cli/sonder_infer_commands.hpp`).

The HTTP server started by `sonder-infer serve` has its own reference:
[SERVER.md](SERVER.md).

## Commands

| Command | Purpose |
| --- | --- |
| `generate` | Generate a completion for one prompt; the text streams to stdout. |
| `chat` | Chat with a model: one-shot from a messages file, or an interactive REPL on stdin. |
| `serve` | Local HTTP API (OpenAI-compatible subset) and live telemetry; see [SERVER.md](SERVER.md). |
| `bench` | Run the benchmark harness over a corpus and write JSON results (plus markdown with `--markdown`). |
| `models` | List the models a backend can serve. |
| `backends` | Probe the backends compiled into this build. |
| `devices` | List host devices (CPU and memory inventory). |
| `version` | Print version, commit, C ABI and HTTP API versions. |
| `help` | `help` prints the overview; `help <command>` prints that command's options and examples. |

`sonder-infer <command> --help` (or `-h`) is the same as `help <command>`
and wins over any other argument on the line (for `serve` too: the CLI checks
for `--help` before handing the arguments to the server's parser). A bare `sonder-infer` prints the
short command list on stderr and exits 2; so does an unknown command, with a
did-you-mean suggestion (`genrate` → `generate`).

## Option syntax

- Options are declared per command. Boolean flags take no value; every other
  option takes exactly one: `--max-tokens 8` or `--max-tokens=8`. A value may
  start with dashes when it is the separate next argument (`--prompt --x`).
- Repeatable options (`--stop`, and for `serve` `--model`, `--model-dir`,
  `--cors-origin`, `--tensor-override`) may be given several times. Any other option given twice is
  a usage error.
- An unknown option is a usage error naming the command, with a suggestion
  when a declared option is at most two edits away (insertions, deletions,
  substitutions or adjacent transpositions):
  `error: generate: unknown option --max-token (did you mean --max-tokens?)`.
- Positional arguments are rejected (`help <command>` is the only one).
- `serve` keeps its own argument parser (`src/server/src/serve_main.cpp`,
  owned by the server module). It shares the unknown-option wording and
  did-you-mean suggestion, but its other errors have no `serve:` prefix and
  its numeric errors state the valid range without echoing the bad value
  (`--port must be an integer from 0 to 65535`).
- Numbers are parsed strictly: the whole value must be a plain decimal
  number, optionally with a leading `-` and an exponent (`0.5`, `-2`, `.5`,
  `1e-3`). There is no leading `+`, no whitespace, no trailing characters, no
  hex (`0x0.8p0`) and no `inf`/`nan`. Errors name the flag and the value:
  `invalid value 'abc' for --top-p (expected a number)`.
  Semantic ranges of sampling values are then checked by `validate()` (for
  example temperature in [0, 10]). Explicit CLI ranges:

  | Option | Range |
  | --- | --- |
  | `--mock-delay-ms` | 0 to 60000 |
  | `--warmup` | 0 to 100 |
  | `--runs` | 1 to 1000 |
  | `--priority` | -16 to 16 |
  | `--max-tokens` | 1 and up (then `validate()`) |
  | `--seed` | 0 to 18446744073709551615 |
  | `--budget-seconds` (sonder-bench) | 0 to 31536000 |

## Environment

Flags always win. Empty variables count as unset.

| Variable | Default for |
| --- | --- |
| `SONDER_INFER_BACKEND` | `--backend` |
| `SONDER_INFER_MODEL` | `--model` |
| `SONDER_OLLAMA_URL` | `--ollama-url` (used verbatim) |
| `OLLAMA_HOST` | `--ollama-url` when `SONDER_OLLAMA_URL` is unset. Accepts Ollama's forms: `host`, `host:port`, or a URL. The default scheme is `http`, the default port 11434 (443 for `https`), and `0.0.0.0`, `[::]` or an empty host map to `127.0.0.1`. |
| `NO_COLOR` | Any non-empty value turns off colored REPL labels. |

`sonder-infer serve` reads the same variables through the same code
(`sonder/inference/backend_setup.hpp`). A build without the server module uses
an equivalent copy in `src/cli/cli_env.hpp`, and the unit tests check that the
two agree.

Model defaults: with `--backend mock` and no model, the model is `mock`.
Every other backend needs `--model` (or `SONDER_INFER_MODEL`); the error
points to `sonder-infer models --backend <name>`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | Runtime or backend failure: model not found, backend unreachable, a failed request, a telemetry or output file that cannot be written, `backends` with no backend available, a bench run with failed requests. |
| 2 | Usage error: unknown command or option, a missing or invalid value, an unknown backend name or one this build does not include, an unreadable or invalid input file (`--messages`, `--corpus`, `--token-file`). |
| 130 | The request was cancelled with Ctrl-C (`generate`, one-shot `chat`). |

In the interactive chat, Ctrl-C cancels the current turn only; the REPL
continues and exits 0, or 1 if any turn failed. `serve` exits 0 after a clean
drain on SIGINT/SIGTERM (a second signal exits at once), 1 on a runtime error
such as a port in use, and 2 on a usage error.

The `sonder.cli.*` CTest cases (`tests/CMakeLists.txt`, driver
`tests/cli_expect.cmake`) check the exact exit code of each command they run:
0, 1 and 2 across the commands, and 130 for `generate` interrupted with
SIGINT (`sonder.cli.generate_sigint_exit_130`, POSIX only). The mapping of a
cancelled request to 130 is also unit-tested (`exit_code_for()`). `backends`
exiting 1 when no backend is available has no CTest case: every build
includes the mock backend, so that state cannot be produced.

## Backends

- `mock`: the deterministic MOCK backend, for tests and harness work only. It
  performs no inference and is never a quality or performance signal. It
  serves the model names `mock` and `mock:<anything>`, and its replies stop
  after about 16 tokens by design (a natural end-of-sequence), unless
  `--max-tokens` is lower. `generate`, `chat` and `bench` print
  `MOCK BACKEND - synthetic output, not a quality or performance signal` on
  stderr unless `--quiet` is given.
- `ollama`: the Ollama compatibility adapter (default
  `http://127.0.0.1:11434`). When Ollama is unreachable the error is followed
  by `hint: Ollama is not reachable; start it with 'ollama serve', or pass
  --ollama-url URL`.
- `llamacpp`: the direct llama.cpp backend, only in builds configured with
  `SONDER_WITH_LLAMA_CPP=ON`; `--model` is a GGUF path.

`--ollama-allow-remote` maps to `OllamaBackendOptions::allow_remote` and
always prints a warning (also with `--quiet`). Prompts and replies never
travel unencrypted to a remote host: the CLI refuses the flag with exit 2
when the Ollama URL (from `--ollama-url`, `SONDER_OLLAMA_URL` or
`OLLAMA_HOST`) is plain `http://` to a non-loopback host (loopback: `localhost`,
`::1`, or an IPv4 literal in 127.0.0.0/8; a DNS name starting with `127.` is remote), for every command
including `serve`:
`error: models: --ollama-allow-remote refuses plain http:// to a non-loopback host ('http://192.0.2.1:11434'); use https:// ...`.
The Ollama client itself only checks the host once `allow_remote` is set, so
this refusal lives in the CLI (`cli::is_plain_http_remote()`, tested against
the client's own URL parser). `https://` needs a `SONDER_WITH_TLS=ON` build,
which the default build does not wire in; in the default build the flag
therefore cannot reach a remote host.

## generate

```sh
sonder-infer generate --backend mock --model mock:tiny --prompt "hello sonder" --max-tokens 8
SONDER_INFER_BACKEND=mock SONDER_INFER_MODEL=mock:tiny \
    sonder-infer generate --prompt hi --quiet --stats json --telemetry events.jsonl
```

The completion streams to stdout, followed by a newline. Option groups:
backend (`--backend`, `--model`, `--ollama-url`, `--ollama-allow-remote`,
`--mock-delay-ms`), input (`--prompt`, required), sampling (see
`help generate` and [sampling-config.md](integration/sampling-config.md)),
telemetry, correlation and output (below).

## chat

One-shot: `--messages FILE` answers the conversation in FILE (a JSON array of
`{"role","content"}` or an object with a `messages` array) and exits.
`--system TEXT` is prepended unless the conversation starts with a system
message.

Interactive (no `--messages`): every non-empty stdin line is a user message.

| Input | Effect |
| --- | --- |
| `/help` | List the commands. |
| `/stats` | Session totals (`turns`, `failed`, `cancelled`, token counts) and the last turn's stats line. |
| `/reset` | Clear the history; the system prompt stays. |
| `/exit`, `/quit`, EOF | End the chat. |
| `//text` | Send `/text` as a message. |
| any other `/word` | Rejected with a hint; nothing is sent. |

On a terminal the REPL shows role labels (`you> ` before input, `assistant> `
before each reply). They are colored only when stdout is a terminal, `NO_COLOR`
is unset and `TERM` is not `dumb`; on Windows the CLI also switches on
virtual-terminal processing for the console and leaves colors off if that
fails. With piped stdin there is no input prompt,
so stdout carries only the replies, one per line; the output always ends with
a newline, never with a dangling prompt.

Every turn runs through `Session::chat` (when the library defines
`SONDER_HAS_SESSION_CHAT`): the engine scheduler and KV accounting apply, and
`--telemetry` records `request.queued` (with `attributes.kind = "chat"`),
`request.started`, the decode events and `request.completed` (or
`request.cancelled`/`request.failed`). Backends with native chat get the
messages; others get the generic prompt format.

Stats: one-shot chat prints a stats line by default; the REPL prints per-turn
stats only with `--stats text` or `--stats json`.

## Telemetry options

`--telemetry PATH` writes Observatory envelope `sonder.observatory.event/1`
JSONL to PATH (`-` for stderr); `--telemetry-level off|metrics|standard|deep`
(default `standard`); `--capture-text` includes generated text in token
events. With `--telemetry -`, telemetry lines and status lines never
interleave mid-line (stderr is written line by line under one lock).

## Correlation and scheduling options

`generate`, `chat` and `bench` accept:

| Option | Maps to |
| --- | --- |
| `--run-id ID` | `SessionOptions::run_id`, the envelope `run_id` (default: the engine id; for `bench`, `--label` when set) |
| `--agent-id ID` | `SessionOptions::agent_id`, the envelope `agent_id` |
| `--task-id ID` | `SessionOptions::task_id`, the envelope `task_id` |
| `--workload CLASS` | `SessionOptions::workload`: `interactive_user`, `owner_orchestrator`, `critic_verification`, `implementation_worker` (default), `research_worker`, `background_indexing`, `maintenance` |
| `--priority N` | `SessionOptions::priority`, -16 to 16. The scheduler uses effective rank = class rank - N (classes rank 0 to 6 in the order listed above; lower runs sooner), so N is not confined to the class: `--workload maintenance --priority 16` gives rank -10 and runs ahead of `interactive_user` work. |

IDs use the same format as the serve API's `X-Sonder-*` headers:
`[A-Za-z0-9._:-]{1,128}`. `request.queued` carries `workload` and `priority`
in its attributes.

## Output options

- `--stats text|json|none` (`generate`, `chat`): per-request stats on stderr.
  The default is `text` for `generate` and one-shot `chat`, `none` for the
  REPL.
- `--quiet` (`generate`, `chat`, `bench`): suppresses the human status lines
  on stderr (the MOCK banner, the chat banner, bench progress) and makes the
  stats default `none`; an explicit `--stats` still applies. Errors and the
  `--ollama-allow-remote` warning are still printed. With `--quiet`,
  `--telemetry -` and optionally `--stats json`, stderr is pure JSON lines.
- `--json` (`version`, `devices`, `backends`, `models`): one JSON document on
  stdout instead of text.

## JSON shapes

All documents are one line. Consumers ignore unknown keys; keys are only
added, never renamed.

`version --json`:

```json
{"name":"sonder-infer","version":"0.1.0","commit":"bab456433cba","abi_version":1,"api_version":1,
 "platform":"linux-x86_64","backends":["mock","ollama"],"modules":{"server":true,"bench":true}}
```

`api_version` is the serve HTTP API major version, or `null` in a build
without the server module.

`devices --json`:

```json
{"platform":"linux-x86_64","devices":[{"id":"cpu:0","kind":"cpu","name":"...","logical_cores":4,
 "total_memory_bytes":16877547520,"available_memory_bytes":15598018560}]}
```

`backends --json` (exit 1 when `available` is 0):

```json
{"backends":[{"name":"mock","available":true,"version":"mock-1","error":null,
  "description":"MOCK deterministic backend for tests only; performs no inference","synthetic":true,
  "capabilities":["tokenization","streaming","deterministic"]},
 {"name":"ollama","available":false,"version":null,"error":"unavailable: connect to 127.0.0.1:11434 refused",
  "description":"Ollama compatibility adapter at http://127.0.0.1:11434","synthetic":false,
  "capabilities":["streaming","remote_process"]}],
 "available":1}
```

`models --backend mock --json`:

```json
{"backend":"mock","synthetic":true,"models":[{"name":"mock:tiny","format":"mock","family":"mock-deterministic",
 "parameter_size":"0","quantization":"none","size_bytes":0,"context_length":4096}]}
```

Stats line (`--stats json`, on stderr, one object per request):

```json
{"command":"generate","backend":"mock","model":"mock:tiny","synthetic":true,"request_id":"req-2359522de2129272",
 "outcome":"completed","stop_reason":"end_of_sequence","prompt_tokens":1,"completion_tokens":16,
 "token_counts_from_backend":true,"ttft_ms":0.89,"total_ms":5.71}
```

`chat` adds `native_chat` and `messages` (the conversation length), and in the
REPL `turn` (1-based). `ttft_ms` is `null` when no chunk arrived. `outcome`
is `completed`, `cancelled` or `failed`; `stop_reason` is a `StopReason` name.

## serve

`sonder-infer serve [options]` calls
`sonder::inference::server::serve_main()` (module `src/server`), which parses
its own options with the same did-you-mean wording
(`unknown option --prot (did you mean --port?)`). `sonder-infer serve --help`
and `sonder-infer help serve` print its option list. In a build without the
server module, `serve` prints `this build does not include the server module
(src/server)` and exits 2. See [SERVER.md](SERVER.md) for the options, the
API and the security notes.

```sh
sonder-infer serve --backend mock --port 18437
curl -fsS http://127.0.0.1:18437/v1/sonder/health
```

## bench and sonder-bench

`sonder-infer bench` runs the harness and writes the results JSON
(`sonder.inference.bench/1`) to `--out`; `--markdown` also prints the markdown
summary and writes it next to `--out`. `--quiet` suppresses progress lines.

`sonder-bench` is the standalone runner (see [bench/README.md](../bench/README.md)):

- With no arguments it prints its usage on stderr and exits 2; `--help`
  prints it on stdout and exits 0.
- Options use the same spec parser (did-you-mean, checked numbers).
- The mock backend prints a `MOCK BACKEND` banner on stderr, and the markdown
  summary is headed `MOCK BACKEND: not a performance claim`.
- Exit codes: 0 ok; 1 a request failed or an output file could not be
  written; 2 usage error (including an unreadable corpus); 3 backend
  unreachable or model load failed; 4 `--require-idle` refused to run.

## Open questions

Recorded, not decided here:

1. `sonder-bench` keeps its historical exit codes 3 and 4 (scripts may rely on
   them) instead of folding them into 1 as `sonder-infer` does.
2. `serve` keeps its own table-driven parser in `src/server` (strict numbers,
   repeatable `--model`); it now shares the unknown-option wording and
   suggestions from `src/cli/cli_spec.hpp`. Moving serve's option table onto
   `sonder::cli::CommandSpec` would be a change inside the server module.
