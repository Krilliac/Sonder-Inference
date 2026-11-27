# External llama-server backend

`llamaserver` runs inference through an external HTTP server. It is included
in the normal CI builds without `SONDER_WITH_LLAMA_CPP`, and neither links nor
downloads llama.cpp, `common/`, an executable, or model weights. Supply your
own trusted llama-server build and compatible model. The existing `llamacpp`
and `ollama` backends remain available.

## Spawn a managed server

Save this JSON as `llamaserver.json`. Replace `<llama.cpp-dir>` with the directory holding your llama-server build and `<model-dir>` with the directory holding the GGUF file (the example is a Windows layout; on Linux use `<llama.cpp-dir>/llama-server`):

```json
{
  "mode": "spawn",
  "executable": "<llama.cpp-dir>/llama-server.exe",
  "args": [
    "--model", "<model-dir>/model.gguf",
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

For typed, validated settings (context, KV cache types, offload,
checkpoints, speculative decoding, default sampling) and a VRAM fit check,
use a [launch profile](launch-profiles.md) instead of raw `args`.

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

For a slow upstream that streams at a fraction of a token per second, with
a bearer token, idle and first-content timeouts and a progress watchdog, see
[colibri.md](colibri.md).

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
`"slot_affinity"`) a chat's `ChatRequest::session_key` can reuse an idle slot
with `id_slot` when the upstream has more than one slot:

- The server fills `session_key` from `prompt_cache_key`, else from
  `X-Sonder-Run-Id` / `X-Sonder-Agent-Id` (`run=<id>;agent=<id>`), so each
  agent of a run keeps its own slot. No key: no `id_slot` (llama-server picks
  an idle slot by prompt similarity, as before).
- A known key pins its owned slot only while idle (`hit`). If its previous
  turn still runs, the request omits `id_slot` (`busy_unpinned`), letting
  llama-server select or defer safely. The key keeps ownership for its next
  call; an unpinned request does not release the running request's busy mark.
- A new key prefers a successfully warmed slot no key owns (when warm-up is
  enabled), then takes the lowest slot no key owns, else the slot of the least
  recently used key whose slot is idle. It never takes a busy slot: unowned
  idle slots report `new`, and replacing an idle owner reports `stolen`.
  When every slot is busy the request is not pinned (`busy_unpinned`). The
  map holds one key per slot (at most 1024).
- With one slot (`total_slots` 1), keyed chats omit `id_slot` (`single_slot`):
  pinning buys no affinity and can trigger prompt save/load on a busy slot.
  With an unknown slot count nothing is pinned.

llama.cpp consolidated explicit-id selection and prompt-cache updates in
`get_available_slot` ([PR #24755](https://github.com/ggml-org/llama.cpp/pull/24755)).
Omitting a busy pin avoids forcing that slot through the prompt-cache path.
Busy here means a request holding a lease in this backend; affinity does not
observe other clients talking directly to the upstream. For proxy-side
serialization, use `sonder-infer serve --priority-admission on --backend-capacity 1`
alongside the usual backend options. Disabling affinity with JSON
`"slot_affinity": false` also omits `id_slot`; its option name and default
remain unchanged.

Generic OpenAI mode (`native_completion: false`) sends neither field, since
strict OpenAI-compatible servers reject unknown parameters. Raw completions
(`generate`) send `cache_prompt` but carry no conversation key.

Thinking controls (`ChatRequest::thinking`) are sent as
`chat_template_kwargs: {"enable_thinking", "reasoning_effort"}` only when set;
see [SERVER.md](../SERVER.md#thinking-control) for pinning them server-wide.
With `native_completion=false` (generic OpenAI mode) the controls are instead
sent as top-level `enable_thinking` / `reasoning_effort` and
`chat_template_kwargs` is no longer sent; a vLLM or TabbyAPI attach that relied
on `chat_template_kwargs` is affected.

## Slot prefix warm-up

The llamaserver backend can replay a stable system prompt into selected slots
after `/health` becomes ready. This is disabled by default. Configure it in the
llamaserver JSON (or override only the messages path with
`sonder-infer serve --warmup-file PATH --backend llamaserver`):

```json
{
  "warmup": {
    "messages_file": "<config-dir>/agent-prefix.json",
    "chat_template_kwargs": {"enable_thinking": false},
    "slots": "all",
    "on_restart": true,
    "max_prefix_chars": 262144
  }
}
```

`messages_file` contains a JSON array of chat messages, usually one `system`
message and any stable tools text. Warm-up is available only for native
llama-server chat mode (`native_completion: true`); generic OpenAI mode reports
a warning and sends no warm-up request. Warm-up requests use the native chat route,
`cache_prompt: true`, `max_tokens: 1`, and an explicit `id_slot`; generated
output is discarded. `slots` may be `"all"` or a unique array of slot ids.
Warm-up runs in the background after readiness and, for spawn mode when
`on_restart` is true, after a supervisor restart or accepted `auto_fit`
relaunch. Attach mode performs the startup warm-up only; it cannot observe or
react to an external server restart. It never delays readiness or blocks the
request path. A missing or malformed messages file is a warning and disables
that attempt; it is not a fatal backend error.

**The `chat_template_kwargs` must match the effective kwargs used by real
requests, including thinking and server pin settings.** A mismatch changes the
rendered prefix from its first token and defeats reuse. The file is bounded by
`max_prefix_chars` (UTF-8 bytes including JSON syntax) before parsing.
Its default is 262,144 and its allowed range is 1 through 16,777,216. Relative
paths resolve against the host process's working directory. `slots` defaults
to `"all"`, `on_restart` defaults to true, and `slots: []` requests no replays.
There is one attempt per selected slot per accepted start, with at most 1,024
slots. Each request uses the configured backend timeouts and shutdown cancels
the worker. Requests arriving while warm-up runs continue normally; llama-server
queues requests for a busy slot. Children immediately replaced by the spill
guard are not warmed; the final accepted child is.

Health and model responses add `runtime.warmup` only when warm-up is configured.
It contains a `generation`, a `status` (including `warming`), and per-slot
`id_slot`, `status`, `prompt_tokens`, `cache_n`, and `milliseconds`. Prompt
tokens include cached tokens; `cache_n` describes reuse during the warm-up
itself, not the next real turn. Unreported counters are null. Failed slots
also have a sanitized `error`, and runtime warnings never include prefix text
or upstream response bodies. The engine runtime sampler emits `backend.warmup`
for observed terminal slot results, with the same slot fields plus `backend`
and `generation`; it uses the existing telemetry level and sampling interval.

On hybrid attention/DeltaNet models, llama.cpp currently does not persist the
recurrent checkpoints in slot save files, so slot save/restore cannot provide
this warm state (see [llama.cpp issue #28194](https://github.com/ggml-org/llama.cpp/issues/28194)).
Warm-up deliberately re-runs the prefix in each target slot after every
relevant restart instead.

## Slot save and restore

Add `"--slot-save-path", "<slot-cache-dir>"` to the child arguments and create
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
  `AIP_*` environment overrides are refused and all inherited `AIP_*`
  variables are cleared from the child's environment, because llama-server's
  Vertex AI mode would replace the supervisor-owned port and it parses
  `AIP_HTTP_PORT` unconditionally.
  `--host 0.0.0.0` is never accepted. The executable is trusted native code,
  not a sandboxed program.
- Attach accepts loopback by default. Remote hosts require
  `"allow_remote": true` **and HTTPS**; plain remote HTTP is refused even
  with that opt-in. HTTPS requires `SONDER_WITH_TLS=ON`.
- The `tls` object uses the existing HTTP client's certificate rules (keys in
  the table below). Defaults verify the certificate; see
  [TLS integration](tls.md) for pin-only semantics.
- Keep credentials, model weights, slot snapshots and runtime configuration
  out of Git. To authenticate to an upstream (for example a reverse proxy in
  front of it), set `api_key_env` to the *name* of an environment variable
  that holds the token; the adapter then sends `Authorization: Bearer <value>`.
  The token is never read from the config file, never logged, and runtime
  status shows only the variable name. It is an attach-mode option, and it is
  refused together with `warmup` (see the table below).
- Free-port selection has a bind handoff between supervisor and child.
  A collision can fail startup; the child is never deliberately rebound to a
  public interface. This is not isolation from other processes running as
  the same local user.

## Configuration reference

This is the one list of every key the `--llamaserver-config` JSON file
accepts (checked 2026-11-20 against the loader `load_llamaserver_config` in
`src/server/src/backend_setup.cpp` and the option structs in
`src/backends/llamaserver/include/sonder/inference/backends/llamaserver.hpp`).
The file must be a JSON object of at most 1 MiB. Unknown keys, wrong types
and out-of-range values are rejected with a message that names the key, so a
misspelled setting never silently selects a default. Other pages
([VRAM spill](vram-spill.md), [colibri](colibri.md), [tune](tune.md),
[launch profiles](launch-profiles.md)) explain the behaviour behind
individual keys and link back here for the key list.

Durations are integer milliseconds. Unless a row says otherwise a duration is
in [1, 3600000] (one hour); 0 is rejected.

### Top level

| Key | Type, default | Range / notes |
|---|---|---|
| `mode` | string, `attach` | `attach` or `spawn`. |
| `base_url` | string, `http://127.0.0.1:8080` | Attach mode: the server root (optionally with a proxy prefix; no `/v1`). |
| `executable` | string, empty | Required in spawn mode; the llama-server binary. |
| `args` | string array, `[]` | Spawn mode: one child argument per entry, no shell. `--host`, `--port`, `--` and `--rpc` are refused (cross-host split uses `rpc_worker`). |
| `allow_remote` | bool, `false` | Attach to a non-loopback host; requires HTTPS (see "Security and configuration limits" above). |
| `native_completion` | bool, `true` | `false` selects generic OpenAI mode (`/v1/completions`). |
| `grammar` | string, empty | Upstream GBNF; see [Sampling and observations](#sampling-and-observations). |
| `upstream_grammar` | bool, `true` | `false` declares the upstream cannot honour a grammar; a non-empty `grammar` is then refused at start. |
| `connect_timeout_ms` | `3000` | |
| `request_timeout_ms` | `300000` | Cap on a request (see `unlimited_request_timeout`). |
| `startup_timeout_ms` | `60000` | Spawn: wait for `/health`. |
| `poll_interval_ms` | `50` | |
| `shutdown_timeout_ms` | `2000` | Spawn: grace before the child is killed. |
| `restart_backoff_ms` | `100` | Must not exceed `max_restart_backoff_ms`. |
| `max_restart_backoff_ms` | `2000` | |
| `max_restarts` | `3` | 0 to 1000; 0 disables restarts. |
| `idle_timeout_ms` | `0` (off) | 0 to 86400000. Attach only. Must exceed `keepalive_gap_ms` when non-zero. |
| `first_content_timeout_ms` | `0` (off) | 0 to 86400000. Attach only. |
| `keepalive_gap_ms` | `10000` | The upstream's longest gap between keepalive bytes. |
| `unlimited_request_timeout` | bool, `false` | Removes `request_timeout_ms` as a cap on streaming completions. Attach only; needs both `idle_timeout_ms` and `first_content_timeout_ms` non-zero. |
| `api_key_env` | string, empty | Name (not value) of an environment variable holding a bearer token; sent as `Authorization: Bearer <value>`. Attach only; not allowed with `warmup`. The name is at most 128 characters of letters, digits and `_`, not starting with a digit; the value must be a single visible-ASCII token. The variable must be set and valid when the backend is constructed, or construction fails with `environment variable X is not set`; the value is capped at 4096 bytes. It is read per completion and discovery request in attach mode, so a rotated token is picked up there; the native-attach monitor is built once at construction and keeps the original token. |
| `context_length` | `0` (unknown) | 0 to 4294967296. Served context used when the upstream has no `GET /props`. |
| `slot_affinity` | bool, `true` | See "Prompt cache and slot affinity". |
| `log_file` | string, empty | Spawn: appended as `--log-file` unless `args` already has one; scanned for performance warnings. See [vram-spill.md](vram-spill.md). |
| `kv_pairing_check` | bool, `true` | Warn when FlashAttention may run with mismatched K/V cache types. See [vram-spill.md](vram-spill.md). |
| `env` | object of strings, absent | Child-only environment overrides; at most 128 entries and about 32 KiB. `AIP_*` is refused. |
| `results` | object, absent | Inert tune provenance (schema `sonder.inference.tune/1`); accepted only with an explicit spawn configuration that has `executable` and `args`. |
| `rpc_worker` | object, absent | Spawn mode only: `enabled` (bool, `false`), `servers` (at most 16 `IPV4:PORT` strings) and `allow_public_addresses` (bool, `false`). Makes this child an isolated llama.cpp RPC worker; see [rpc-worker.md](rpc-worker.md). |

### `tls` object

Same certificate rules as the other HTTP clients; see [tls.md](tls.md) for
pin-only semantics. HTTPS needs a build with `SONDER_WITH_TLS=ON`.

| Key | Type, default | Range / notes |
|---|---|---|
| `ca_bundle_path` | string, empty | |
| `pinned_sha256` | string, empty | |
| `pinned_cert_path` | string, empty | |
| `server_name` | string, empty | |
| `insecure_skip_verify` | bool, `false` | |
| `handshake_timeout_ms` | `10000` | 1 to 3600000. |

### `spill_guard` object (spawn mode)

Behaviour: [vram-spill.md](vram-spill.md).

| Key | Default | Range / notes |
|---|---|---|
| `enabled` | `true` | |
| `policy` | `warn` | `warn`, `refuse` or `auto_fit`. |
| `threshold_mib` | `256` | 1 to 1048576. |
| `baseline_mib` | `0` | 0 to 1048576. |
| `baseline_per_1k_ctx_mib` | `0` | 0 to 1024. |
| `sample_interval_ms` | `5000` | 100 to 3600000. |
| `fit_step_factor` | `0.85` | Number strictly between 0 and 1. |
| `fit_step_align` | `1024` | 1 to 1048576. |
| `fit_min_ctx` | `8192` | 1 to 4294967296. |
| `fit_max_attempts` | `4` | 1 to 32. |
| `residency` | object | Sub-keys `enabled` (true), `expect_gpu` (false), `min_dedicated_mib` (512; 1 to 1048576), `consecutive_samples` (3; 1 to 1000), `eviction_fraction` (0.25; 0 to 1), `eviction_mib` (2048; 0 to 1048576), `on_eviction` (`warn` or `restart`; `warn`), `max_eviction_restarts` (1; 0 to 32). `eviction_fraction` and `eviction_mib` cannot both be 0. |

### `stall_guard` object

| Key | Default | Range / notes |
|---|---|---|
| `enabled` | `true` | |
| `stall_seconds` | `90` | 1 to 86400. |
| `policy` | `warn` | `warn`, `restart` or `cancel`. `restart` acts on a spawned child only; `cancel` needs `generic_progress`. |
| `generic_progress` | `false` | Generic OpenAI attach (`native_completion: false`) only; needs `enabled: true` and a policy of `warn` or `cancel`. See [colibri.md](colibri.md). |

### `warmup` object

Behaviour: "Slot prefix warm-up" above.

| Key | Default | Range / notes |
|---|---|---|
| `messages_file` | empty (warm-up off) | JSON array of chat messages. |
| `chat_template_kwargs` | `{}` | Object; must match the kwargs real requests use. |
| `slots` | `"all"` | `"all"` or an array of unique integers in [0, 1023]. |
| `on_restart` | `true` | |
| `max_prefix_chars` | `262144` | 1 to 16777216. |

### `host_memory` object (spawn mode)

Behaviour: "Host-memory admission (MoE offload)" below.

| Key | Default | Range / notes |
|---|---|---|
| `reserve_mib` | `2048` | 0 to 16777216. Kept free on top of the estimate. |
| `override_refusal` | `false` | Start anyway when the pre-spawn check would refuse (a warning is raised). |
| `floor` | object | Sub-keys below. |
| `floor.enabled` | `false` | |
| `floor.policy` | `warn` | `warn`, `stop_admission` or `restart_smaller_ctx`. |
| `floor.floor_mib` | `0` | 0 to 16777216; must be above 0 when `floor.enabled` is true. |
| `floor.resume_mib` | `0` | 0 to 16777216; 0 means `floor_mib` plus a quarter of it. Otherwise at least `floor_mib`. |
| `floor.consecutive_samples` | `2` | 1 to 100. |
| `floor.sample_interval_ms` | `5000` | 10 to 3600000. |
| `floor.restart_cooldown_ms` | `60000` | 0 to 86400000. |
| `floor.max_restarts` | `1` | 0 to 1000. |

### Cross-key rules

- `api_key_env`, `idle_timeout_ms`, `first_content_timeout_ms` and
  `unlimited_request_timeout` are attach-mode options; setting any of them in
  spawn mode fails at backend construction.
- `stall_guard.policy` of `cancel` without `generic_progress`, and
  `generic_progress` with `restart`, are refused.
- `rpc_worker` needs spawn mode, and `rpc_worker.servers` without
  `"enabled": true` is refused rather than ignored.
- A spawned child's own llama-server flags, including model placement and
  context, go in `args` (or come from a [launch profile](launch-profiles.md)).

## Host-memory admission (MoE offload)

For very large mixture-of-experts models the weights can exceed free RAM even
with the GPU in use. In spawn mode the supervisor can check host RAM before
it launches the child, and watch available memory afterwards. This section is
the home for that behaviour; the `host_memory` keys are in the table above.
Facts checked 2026-11-20 against `src/backends/llamaserver/host_memory.cpp`,
`supervisor_host_memory.cpp` and `src/server/src/serve_main.cpp`.

**Pre-spawn check.** It runs only when `sonder-infer serve` has a host-RAM
estimate, which it computes from the GGUF header of the active launch profile
for a `llamaserver` backend (including a CPU-only profile). With no estimate
(a plain `--llamaserver-config` without a launch profile, or an unreadable
GGUF header) no check runs. There is no command-line flag. The check compares
available physical memory (and, where the platform reports it, available
commit against the private estimate) with the estimate plus `reserve_mib`.

- Refused: the start fails with `ErrorCode::unavailable` and a message giving
  the estimate, the reserve and what is available. The refusal is terminal
  on every iteration, including a crash restart, after which the supervisor
  stays failed. Fix it by offloading more layers, lowering `--ctx-size`,
  freeing memory, or setting `override_refusal: true`. (The refusal text
  spells the key `host_memory.admission.override_refusal`; the JSON key is
  `host_memory.override_refusal`.)
- Availability not measurable: the `host_memory_unknown` warning is raised and
  the child starts only when available *physical* memory cannot be read (macOS,
  or an unreadable cgroup v2 limit; cgroup v1 limits are not read). On Linux,
  available physical memory comes from meminfo or the cgroup v2 limit, so the
  check runs. Where available *commit* is unknown (Linux and macOS), only the
  physical comparison runs and no warning is raised for the missing commit
  figure.

**Floor guard** (`host_memory.floor.enabled: true`). After readiness the
supervisor samples available physical memory every `sample_interval_ms`.
Below `floor_mib` for `consecutive_samples` readings it enters pressure; it
leaves pressure after `consecutive_samples` readings at or above `resume_mib`.

| `floor.policy` | In pressure |
|---|---|
| `warn` | Raise `host_memory_low` only. |
| `stop_admission` | Also refuse new requests with `unavailable` until memory recovers; running requests continue. |
| `restart_smaller_ctx` | Also close admission, wait until the child is idle, then relaunch with a smaller `--ctx-size` (needs `--ctx-size` in `args`; uses the `spill_guard` `fit_step_factor`, `fit_step_align` and `fit_min_ctx` steps), at most `max_restarts` times and not more often than `restart_cooldown_ms`. If `args` has no `--ctx-size`, start is not refused (only the policy name is validated), and at runtime the restart is abandoned. The same happens when no smaller context exists. Abandoning a restart does not reopen admission: while in pressure, every policy other than `warn` keeps refusing new requests, and the child only keeps finishing requests already running. |

**Warnings** are part of the backend's runtime status. They appear in
`GET /v1/sonder/health` under `backends[].runtime.warnings` and in
`/v1/models` under `sonder.runtime.warnings`. The raw memory readings are not
exposed in either. The codes are `host_memory_low`,
`host_memory_unknown`, `host_memory_admission_override` and
`host_memory_estimate_lower_bound` (the estimate assumes everything fits the
GPU because llama-server `--fit` decides the offload and no VRAM budget is
known).

**Limits of the estimate.**

- The child inherits the environment (only `AIP_*` is removed). These
  llama-server variables change placement but are not modelled; `serve` prints
  a startup note when one is set: `LLAMA_ARG_FIT`, `LLAMA_ARG_N_GPU_LAYERS`,
  `LLAMA_ARG_REPACK`, `LLAMA_ARG_N_CPU_MOE`, `LLAMA_ARG_CPU_MOE`,
  `LLAMA_ARG_N_CPU_FFN`, `LLAMA_ARG_OVERRIDE_TENSOR`, `LLAMA_ARG_LOAD_MODE`.
- For a `llamaserver` launch profile, place expert tensors on the CPU with
  `extra_args` (for example `-ot` or `--cpu-moe`); see
  [launch-profiles.md](launch-profiles.md).
- `--fit` path. When the profile leaves `n_gpu_layers` unset, `fit` is on (the
  default) and there is no tensor override, llama-server sizes the offload
  itself. `-ot`, `--cpu-moe`, `--n-cpu-moe N>0` and `--n-cpu-ffn N>0` make
  upstream fit abort (`--n-cpu-moe 0` does not). The host estimate then adds
  the overflow beyond the VRAM budget (detected VRAM minus 1.5 GiB, or
  `vram_budget_mib`) plus llama-server's 1 GiB fit margin, experts first. The
  budget is not free VRAM, so if other processes hold VRAM more overflows.
  `--fit-target` is not parsed. With no budget known the estimate is a lower
  bound and `host_memory_estimate_lower_bound` is raised; setting
  `vram_budget_mib` or `n_gpu_layers` avoids it.
- Host tensors of type Q4_0, Q8_0, Q4_K or IQ4_NL are counted as 100% private
  memory (llama.cpp repacks them for CPU matmul) unless `-nr`/`--no-repack` is
  in `extra_args`; `--repack` turns it back on. `LLAMA_ARG_REPACK` is noted,
  not read. Whether a given CPU repacks a given type is unverified.
- Only `-lm`/`--load-mode` is modelled for mmap and mlock.
- A llama-server build with no GPU device puts everything on the host, which
  the estimator cannot see.
- Memory-mapped experts are counted resident at the declared active/total
  ratio, which is optimistic for steady decode; raise `reserve_mib` to cover it.
- On Linux inside a tight container, `memory.current` includes reclaimable
  page cache, so headroom can read near zero and the floor guard may trip early.

## Calibration output

[`sonder-infer tune`](tune.md) writes a directly loadable spawn configuration.
The additive `env` object supplies child-only string environment overrides;
when absent, environment inheritance is unchanged, except that inherited
`AIP_*` variables are always removed and `AIP_*` overrides are refused. The optional `results`
object is inert calibration provenance with schema `sonder.inference.tune/1`;
it is accepted only alongside an explicit spawn configuration. Other unknown
top-level fields remain errors. No existing spawn/attach defaults change.

## Tests

`sonder.llamaserver.*` is registered with CTest. Fake HTTP upstreams exercise
streaming, usage/timing translation, finish reasons, sampling and security;
injectable process launchers exercise startup, readiness, crash recovery and
shutdown. No model, real llama-server installation, or network service is
needed. Run with the usual `ci-linux` or `ci-windows` build/test presets.
