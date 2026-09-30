# External llama-server backend

`llamaserver` runs inference through an external HTTP server. It is included
in the normal CI builds without `SONDER_WITH_LLAMA_CPP`, and neither links nor
downloads llama.cpp, `common/`, an executable, or model weights. Supply your
own trusted llama-server build and compatible model. The existing `llamacpp`
and `ollama` backends remain available.

## Spawn a managed server

Save this JSON as `llamaserver.json` (adjust the executable and model paths):

```json
{
  "mode": "spawn",
  "executable": "D:/llama.cpp/llama-server.exe",
  "args": [
    "--model", "D:/models/model.gguf",
    "--alias", "local-model",
    "--ctx-size", "8192",
    "-ngl", "99",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3"
  ],
  "startup_timeout_ms": 60000,
  "max_restarts": 3,
  "restart_backoff_ms": 100,
  "max_restart_backoff_ms": 2000
}
```

```powershell
sonder-infer chat --backend llamaserver --model local-model --llamaserver-config llamaserver.json
sonder-infer serve --backend llamaserver --model local-model --llamaserver-config llamaserver.json
```

The MTP example forwards `--spec-type draft-mtp --spec-draft-n-max 3`.
Speculation types and required draft weights depend on your installed
llama-server and model. DFlash and n-gram options use the same `args` array.
Each string is one argument, including paths with spaces; there is no shell
expansion. Model placement, context, chat templates and speculative execution
belong to the child. See the installed version's `llama-server --help` and
the [upstream server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Spawned children are also checked for VRAM spill (Windows PDH counters) and
K/V cache types without a FlashAttention kernel, and their log can be scanned
for performance warnings; see [VRAM spill and KV kernel pairing](vram-spill.md)
for the `spill_guard`, `log_file` and `kv_pairing_check` keys.

The first upstream operation starts the child on `127.0.0.1` at a free port
and waits for `/health`. A background monitor restarts crashed children with
capped exponential backoff and a finite restart budget. In-flight requests
fail normally; generated output is never silently replayed. A restart loses
the child's in-memory slots. Releasing the backend and its loaded model
handles stops the child. Windows uses a kill-on-close Job Object; POSIX uses
a process group, TERM and then KILL after the shutdown grace period.

## Attach to an existing server

```json
{
  "mode": "attach",
  "base_url": "http://127.0.0.1:8080"
}
```

Use the same `--llamaserver-config` option. Attach mode never starts, stops,
or restarts the upstream. The URL is the server root, optionally with a proxy
prefix; do not append `/v1`. Discovery uses `/v1/models`. Native chat sends
the messages to `/v1/chat/completions` so the upstream applies its template.
Raw generation uses llama-server's `/completion` by default.

For a generic OpenAI-compatible server such as vLLM or TabbyAPI, set
`"native_completion": false`; raw generation then uses `/v1/completions`.
Only endpoints and parameters actually implemented by that upstream are
available. Native grammar and slot operations require llama-server support.

## Sampling and observations

Only supplied sampling settings go onto the wire. CLI flags and incoming
HTTP fields retain their presence, including an explicit value equal to a
Sonder default. Unspecified temperature, top-p, top-k and token limits use
the upstream's defaults. For C++ callers, values changed from
`SamplingConfig{}` are recognized automatically; record explicit default
values in `sampling.explicit_fields`, for example:

```cpp
request.sampling.temperature = 0.8f;
request.sampling.explicit_fields.insert("temperature");
```

Parsers set `explicit_fields_only` to honor the presence markers exclusively,
so their own local defaults (including the HTTP server's token budget) are
not forwarded as user choices. Other backends ignore both presence fields.

Configure context size in the child arguments (`--ctx-size`); llama-server
has no per-request context resize. After readiness the backend reads
`GET /props` once per upstream instance (again after a restart or an
`auto_fit` relaunch) and caches the served per-slot context
(`default_generation_settings.n_ctx`) and slot count (`total_slots`); the
model descriptor's `context_length` reports it. A request `num_ctx` up to the
served context is accepted as a no-op (the engine narrows its own
accounting; nothing is sent upstream), so Sonder Runtime bridged turns, which
always carry a positive `num_ctx`, work. A larger `num_ctx` fails before
anything is streamed with `invalid_argument` naming both numbers (`num_ctx
65536 exceeds the served context 32768 ...`; HTTP 400). An upstream without
`/props` (a generic OpenAI server) uses the config's `context_length`; when
neither is known `num_ctx` is accepted unenforced and the upstream keeps its
own window. A non-empty
`grammar` string in the backend config is sent as GBNF to the upstream.
Structured tool calls are not represented by Sonder's text-only backend
interface. A `length` finish reason maps to `StopReason::max_tokens`.

`GenerateStats` preserves backend usage and timing. Its optional
`cached_tokens`, `draft_tokens`, `draft_accepted_tokens` and
`predicted_tokens_per_second` fields remain absent when unreported. A
timings-only prompt count includes both `prompt_n` and `cache_n`; standard
`usage` counts take precedence. Millisecond timings become nanoseconds.

Session telemetry emits `backend_cached_tokens`, `backend_draft_tokens`,
`backend_draft_accepted_tokens`, `backend_draft_acceptance_ratio` (only with
a positive draft count), and `backend_predicted_tokens_per_second` on
request/decode summaries; prefill events also include backend cached tokens.
These observations are distinct from the scheduler's logical
`reused_prompt_tokens`. They do not establish a performance or quality gain.

## Prompt cache and slot affinity

llama-server keeps one KV cache per slot and reuses a prompt prefix only in
the slot that last processed it; on hybrid models (Qwen3.x) reuse reaches
back only to a checkpoint, so landing on the wrong slot means a full re-read
(about 2 minutes for 100k tokens, against 0.13 s for a 121k-token reuse).
In native mode (`native_completion`, the default) every request sends
`"cache_prompt": true`, and with `slot_affinity` (default true; JSON
`"slot_affinity"`) a chat's `ChatRequest::session_key` is pinned to one slot
with `id_slot`:

- The server fills `session_key` from `prompt_cache_key`, else from
  `X-Sonder-Run-Id` / `X-Sonder-Agent-Id` (`run=<id>;agent=<id>`), so each
  agent of a run keeps its own slot. No key: no `id_slot` (llama-server picks
  an idle slot by prompt similarity, as before).
- A known key keeps its slot, even while its previous turn still runs.
- A new key takes the lowest slot no key owns, else the slot of the least
  recently used key whose slot is idle. When every slot is owned and busy the
  request is not pinned. The map holds one key per slot (at most 1024).
- With one slot (`total_slots` 1) every keyed chat uses slot 0; with an
  unknown slot count nothing is pinned.

Generic OpenAI mode (`native_completion: false`) sends neither field, since
strict OpenAI-compatible servers reject unknown parameters. Raw completions
(`generate`) send `cache_prompt` but carry no conversation key.

Thinking controls (`ChatRequest::thinking`) are sent as
`chat_template_kwargs: {"enable_thinking", "reasoning_effort"}` only when set;
see [SERVER.md](../SERVER.md#thinking-control) for pinning them server-wide.

## Slot save and restore

Add `"--slot-save-path", "D:/slot-cache"` to the child arguments and create
that directory yourself. C++ hosts can use the backend-specific API:

```cpp
#include "sonder/inference/backends/llamaserver.hpp"
auto backend = sonder::inference::make_llamaserver_backend(options);
auto saved = backend->save_slot(0, "conversation.bin");
auto restored = backend->restore_slot(0, "conversation.bin");
```

Check each returned `Status`. Only basenames are accepted. Slot identity and
file compatibility belong to the upstream; coordinate these operations with
requests yourself. This is not a portable Sonder KV export/import contract,
nor a new route on Sonder's HTTP server or C ABI.

## Security and configuration limits

- Spawn always appends its own `--host 127.0.0.1` and ephemeral `--port`.
  User host/port overrides and argument-list terminators are rejected.
  `--host 0.0.0.0` is never accepted. The executable is trusted native code,
  not a sandboxed program.
- Attach accepts loopback by default. Remote hosts require
  `"allow_remote": true` **and HTTPS**; plain remote HTTP is refused even
  with that opt-in. HTTPS requires `SONDER_WITH_TLS=ON`.
- The `tls` object uses the existing HTTP client's certificate rules:
  `ca_bundle_path`, `pinned_sha256`, `pinned_cert_path`, `server_name`,
  `handshake_timeout_ms`, and `insecure_skip_verify`. Defaults verify the
  certificate; see [TLS integration](tls.md) for pin-only semantics.
- Keep credentials, model weights, slot snapshots and runtime configuration
  out of Git. The adapter does not currently supply upstream bearer tokens.
- Free-port selection has a bind handoff between supervisor and child.
  A collision can fail startup; the child is never deliberately rebound to a
  public interface. This is not isolation from other processes running as
  the same local user.

Other configuration keys are `connect_timeout_ms`, `request_timeout_ms`,
`poll_interval_ms`, `shutdown_timeout_ms`, `context_length` (served context
when there is no `/props`; 0 = unknown) and `slot_affinity` (boolean). Durations are bounded positive
milliseconds (at most one hour in JSON); `max_restarts: 0` disables restarts.
`max_restarts` is at most 1000 and the initial backoff cannot exceed its cap.
Unknown keys and invalid
types are rejected so misspelled settings do not silently select defaults.

## Calibration output

[`sonder-infer tune`](tune.md) writes a directly loadable spawn configuration.
The additive `env` object supplies child-only string environment overrides;
when absent, environment inheritance is unchanged. The optional `results`
object is inert calibration provenance with schema `sonder.inference.tune/1`;
it is accepted only alongside an explicit spawn configuration. Other unknown
top-level fields remain errors. No existing spawn/attach defaults change.

## Tests

`sonder.llamaserver.*` is registered with CTest. Fake HTTP upstreams exercise
streaming, usage/timing translation, finish reasons, sampling and security;
injectable process launchers exercise startup, readiness, crash recovery and
shutdown. No model, real llama-server installation, or network service is
needed. Run with the usual `ci-linux` or `ci-windows` build/test presets.
