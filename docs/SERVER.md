# `sonder-infer serve`: local HTTP API and live telemetry

Status: implemented in module `src/server` (`SONDER_HAS_SERVER`), ADR-020
(proposed, awaiting owner sign-off). `sonder-infer serve …` runs the server
(see [Running it](#running-it)); a build without the server module answers
`serve` with an error saying so.
This page is the authoritative reference for the Inference HTTP API v1
(`api_version` 1, paths under `/v1`). It implements sections 2, 5 (Inference
side), 6.1 and 7.1 of the Sonder ecosystem integration contract v1
(2026-09-26), with the corrections from that contract's review applied (listed
under [Deviations from the contract text](#deviations-from-the-contract-text)).

The server is written in-house over POSIX sockets and Winsock, in the spirit
of ADR-013: no third-party code, HTTP/1.1 only, `Connection: close` on every
response, one thread per connection, and no TLS. It never emulates the Ollama
API and never presents itself as Ollama.

## Running it

```sh
# Synthetic MOCK backend (tests and harness development only)
sonder-infer serve --backend mock --port 0 --ready-file /tmp/infer.ready.json

# Ollama models behind the Inference policy layer
sonder-infer serve --backend ollama --model llama3.2:3b --model qwen2.5:7b

# llama.cpp (build with SONDER_WITH_LLAMA_CPP=ON)
sonder-infer serve --backend llamacpp --model-dir ~/models --model tiny.gguf
```

The CLI (`tools/sonder-infer`) dispatches the `serve` subcommand to
`sonder::inference::server::serve_main(args, out, err)` from
`<sonder/inference/server.hpp>`, and builds its backends through the same
shared `make_backend()` factory. Embedding hosts can run the same server in process with
`sonder::inference::server::Server` (`ServerOptions`, `start()`, `port()`,
`stop()`); `ServerOptions::backend_instance` accepts a backend the host built
itself.

### Options

| Option | Default | Notes |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | A non-loopback host requires `--token-file` (exit 2 otherwise). Loopback means `localhost`, `::1` / `[::1]` or an IPv4 literal in 127.0.0.0/8; a DNS name that merely starts with `127.` is not loopback. |
| `--port N` | `11437` | `0` picks an ephemeral port; the ready file and banner show it. |
| `--backend mock\|ollama\|llamacpp` | env `SONDER_INFER_BACKEND` | Required. A backend this build does not include is a usage error (exit 2), checked before anything binds. |
| `--model ID` (repeatable) | env `SONDER_INFER_MODEL` | The first model is the default and also answers to `default`. The mock backend serves `mock` when no model is given; other backends require a model (the error points to `sonder-infer models --backend X`). |
| `--ollama-url URL` | env `SONDER_OLLAMA_URL`, then `OLLAMA_HOST` | `OLLAMA_HOST` values such as `0.0.0.0` or `host:port` are normalised (wildcards become `127.0.0.1`, the port defaults to 11434). |
| `--ollama-allow-remote` | off | Allows a non-loopback Ollama host. Remote hosts also need `https://`, which needs a `SONDER_WITH_TLS=ON` build; the default build does not wire TLS, so remote Ollama is unusable in v1. |
| `--model-dir DIR` (repeatable) | none | llama.cpp model directories. |
| `--token-file PATH` | none | Bearer token (one line). Read once, kept in memory, never logged. A warning is printed when the file is readable by group or others. |
| `--cors-origin ORIGIN` (repeatable) | none | Exact-match browser origin, allowed on every route. Must be a serialized origin as browsers send it: lowercase `scheme://host[:port]`, no path, trailing slash, query or wildcard (exit 2 otherwise). |
| `--no-default-cors` | off | Drops the default origin list. |
| `--telemetry-level off\|metrics\|standard\|deep` | `standard` | |
| `--telemetry-buffer N` | `8192` | Retained events for resume (1 to 1048576). |
| `--telemetry PATH` | none | Also write JSONL to `PATH` (`-` for stderr). |
| `--capture-text` | off | Include generated text in token events. Requires `--token-file`. |
| `--max-connections N` | `64` | A warning is printed when it plus 32 exceeds the open-file limit (`ulimit -n`). |
| `--max-body-bytes N` | `4194304` | |
| `--shutdown-grace-ms N` | `5000` | |
| `--ready-file PATH` | none | Written atomically once the socket **listens** (models may still be loading: poll health for 200): `{"url","pid","instance_id","api_version":1}`. Removed when `serve` returns, after a failed start as well as after a clean shutdown. A process killed outright (second signal, SIGKILL) leaves it behind: check that `pid` is alive. On Windows the rename can leave a handle with delete access on the file for a few milliseconds after it appears (longer under CPU load), so a reader that opens it without `FILE_SHARE_DELETE` (the C runtime and `std::ifstream` default) may get a sharing violation on its first attempt: retry the open. |
| `--mock-delay-ms N` | `0` | Mock per-token delay. |
| `--lazy-models` | off | Register the models at startup and load each on its first request (see [Model residency](#model-residency)). |
| `--model-idle-ttl S` | `0` (never) | Unload a model no request has used for `S` seconds (0 to 2592000); the next request loads it again. |
| `--max-resident-models N` | `0` (no cap) | Keep at most `N` models loaded (0 to 4096), evicting the least recently used idle one. |
| `--log-format text\|json` | `text` | Banner, access log and diagnostics on stderr. |
| `--scheduler automatic\|gate\|account\|off` | `automatic` | Engine scheduling, see [Scheduling](#scheduling). `automatic` gates in-process backends per token and only admits and accounts remote-process backends (llamaserver, ollama); `off` disables the scheduler and the logical KV pool. |
| `--kv-pool-tokens N` | `65536` | Logical KV pool size in tokens (16 to 16777216, rounded up to 16-token blocks). |
| `--pin-enable-thinking on\|off` | none | Sent as `chat_template_kwargs.enable_thinking` (llama-server) / `think` (ollama) on every chat request. See [Thinking control](#thinking-control). |
| `--pin-reasoning-effort E` | none | Same for `chat_template_kwargs.reasoning_effort` (1 to 64 of `[A-Za-z0-9._-]`). |
| `--pin-mode override\|default` | `override` | `override` preserves today's behavior and replaces conflicting request thinking fields with a warning; `default` fills only unset `enable_thinking` and `reasoning_effort` fields, so explicit request values win. |
| `--pin-reasoning-budget N` | none | Fills an unset `reasoning_budget_tokens` value. `N` is an integer greater than or equal to `-1`; it never overrides a request or header value. |
| `--pin-reasoning-budget-message TEXT` | none | Fills an unset `reasoning_budget_message` value. The UTF-8 value is at most 512 bytes and never overrides a request value. |
| `--max-concurrent-subagent N` | unlimited (`0`) | Maximum running `subagent` requests; `0` preserves today's unlimited behavior. |
| `--max-concurrent-background N` | unlimited (`0`) | Maximum running `background` requests; `0` preserves today's unlimited behavior. |
| `--max-queue-per-class N` | unlimited (`0`) | Maximum queued requests for each priority class; over-capacity requests receive 429 with `Retry-After: 1`. |
| `--priority-admission auto\|on\|off` | `auto` | Enable class-ordered admission. `auto` enables it only when a nonzero class concurrency or queue cap is configured; `on` always enables it; `off` bypasses it, including configured caps. |
| `--backend-capacity N` | `0` | Explicit concurrent backend-request limit when priority admission is enabled. `0` uses the backend's advertised capacity; an unknown capacity stays ungated. This flag alone does not enable admission. |

`--key=value` is accepted as well as `--key value`. Exit status: 0 after a
clean shutdown, 1 on a runtime error (for example the port is in use; the
message suggests `--port 0`), 2 on a usage error.

As soon as the socket listens, stderr shows `listening on <url>` (for a
wildcard bind: `listening on 0.0.0.0:PORT (all interfaces; local URL
http://127.0.0.1:PORT)`) and `loading models on <backend>: …`. Once the models
are loaded, the banner (`ready on <url>`) lists the auth mode, the CORS
origins, the backend and models, and the discovery, SSE and NDJSON URLs. With
the mock backend it adds `MOCK BACKEND - synthetic output, not a quality or
performance signal`. The access log writes one line per request: time,
method, path without the query, status, milliseconds and the request id
(`X-Sonder-Request-Id`; for chat this is the engine request id). It never
logs bodies or header values. A request whose client disconnected is logged
with status 499.

### Startup and shutdown

`start()` binds the socket, builds the engine (so `engine.started` carries
`server {host, port, api_version}`), starts accepting (health reports
`starting`, other routes answer 503 `not_ready`), writes the ready file,
registers the backend and loads the models, then reports `ready`. A model
that fails to load stops the server with exit 1. No lock is held across
backend I/O, so health keeps answering 503 `starting` while a model loads
(an Ollama `/api/show` that hangs, for example).

### Model residency

By default every `--model` loads during `start()`, before `ready`, and stays
loaded until shutdown, exactly as before these options existed. Three
options change that; none is on unless the operator sets it:

- `--lazy-models` registers the models at startup (id, backend, default flag;
  no backend I/O) and reports `ready` at once. Each model loads on its first
  chat request; concurrent first requests wait for the same load instead of
  starting their own. A model that cannot load fails the requests that
  needed it (404 `model_not_found` when the backend does not know it, 503
  `not_ready` otherwise; nothing was executed) instead of failing the
  start. `/v1/models` and `/v1/sonder/identity` never load a model: identity
  for a model that has never loaded is `null` with a `reason`.
- `--model-idle-ttl S` unloads a model once no request has used it for `S`
  seconds (`model.evicted` with `reason: "idle_ttl"`, then `model.unload`).
  The next request loads it again.
- `--max-resident-models N` keeps at most `N` models loaded: loading another
  evicts the least recently used idle one first (`reason: "max_resident"`).
  Without `--lazy-models`, only the first `N` models load during start.

A request in flight pins its model from before the load until the response
is finished: an idle TTL or the cap never unloads a pinned model. The cap is
therefore soft: pinned models alone can exceed it until they are released.
Health reports the state of each model and the totals (see below); the
residency options add a `residency:` line to the text banner and a
`residency` field to the JSON `ready` log line (both omitted with the
defaults).

A shutdown request while models load abandons the start after the backend
call in progress returns (the backend's own timeout bounds it; for Ollama
metadata calls that is at most 60 s), then drains as below and exits 0.
Health reports `draining` meanwhile. The first SIGINT/SIGTERM restores the
default signal action, so a second one exits immediately at any point.
`Server::stop()` is safe from any thread, also while `start()` runs; `start()`
then returns `cancelled`.

On SIGINT or SIGTERM (or `request_shutdown()`):

1. Health reports 503 `draining`; new chat requests get 503 `not_ready`.
2. The listener closes; idle connections that have not sent a request close.
3. In-flight requests get `--shutdown-grace-ms`, then are cancelled (a
   streaming client then sees `finish_reason: "cancelled"`).
4. The engine stops: `engine.stopped`, plus a final `telemetry.dropped` if the
   bus dropped anything.
5. Telemetry streams deliver what they hold and close. The process exits 0.

## Security

- Loopback by default. A non-loopback bind needs `--token-file`; the server
  warns that the token then travels in cleartext, because there is no TLS
  server in v1. Put a TLS-terminating proxy in front for remote use.
- With a token, every route except the CORS preflight (`OPTIONS`) requires
  `Authorization: Bearer <token>`, compared in constant time. Otherwise 401
  with `WWW-Authenticate: Bearer`.
- DNS-rebinding defence: on a loopback bind the `Host` header must name
  `127.0.0.1`, `localhost` or `[::1]` (any port); anything else, including a
  missing `Host` or `0.0.0.0`, is 403 `forbidden_host`.
- CORS is an exact-match allowlist. The default origins are Observatory's dev
  server, preview and Tauri shells: `http://127.0.0.1:5173`,
  `http://localhost:5173`, `http://127.0.0.1:4173`, `http://localhost:4173`,
  `tauri://localhost`, `http://tauri.localhost`. Because every Vite project
  and every Tauri app shares these origins, **the defaults only reach the
  read-only GET routes**. `POST /v1/chat/completions` from a browser needs an
  explicit `--cors-origin` or a configured token. An `Origin` that is present
  and not allowed gets 403 `forbidden_origin` (without CORS headers).
  Requests without `Origin` are unaffected.
- Telemetry is content-free unless `--capture-text` is set, which requires a
  token and is announced as `text_capture: "on"` in discovery.
- Nothing secret is logged: the access log omits bodies and header values.

## Common behaviour

Every response carries `X-Sonder-Inference-Api: 1`, `Cache-Control: no-store`,
`Connection: close` and `X-Sonder-Request-Id`. For an allowed `Origin`:
`Access-Control-Allow-Origin: <origin>`, `Vary: Origin` and
`Access-Control-Expose-Headers: X-Sonder-Inference-Api, X-Sonder-Request-Id,
Retry-After`. Normal JSON response bodies carry `"sonder": {"api_version":
1, ...}` so clients that cannot read response headers can still check the API
version. Anthropic error envelopes and SSE protocol event payloads follow the
Anthropic wire shape and are exempt from that additive metadata.

The preflight (`OPTIONS`) answers 204 with
`Access-Control-Allow-Methods: GET, POST, OPTIONS`,
`Access-Control-Allow-Headers: Accept, Authorization, Cache-Control,
Content-Type, Last-Event-ID, X-Sonder-Run-Id, X-Sonder-Parent-Request-Id,
X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority,
X-Sonder-Deadline-Ms, X-Sonder-Reasoning-Budget`
and `Access-Control-Max-Age: 600` (plus
`Access-Control-Allow-Private-Network: true` when the browser asks for it).

### Limits

| Condition | Response |
| --- | --- |
| Request line plus headers over 16 KiB, or more than 64 headers | 431 |
| Malformed request line or header, obsolete line folding, conflicting `Content-Length`, `Content-Length` with `Transfer-Encoding`, more than one `Host` header (RFC 9112 section 3.2) | 400 `malformed_request` |
| HTTP version other than 1.x | 505 |
| `POST` without `Content-Length`, or with a chunked body | 411 `length_required` |
| Body over `--max-body-bytes` (checked before reading it) | 413 `payload_too_large` |
| Headers or body not received within 10 s | 408 `request_timeout` |
| At `--max-connections`, a new connection arrives while another has spent 500 ms or more without completing its request head | The oldest such connection gets 408 `request_timeout` and the new one is served (slow clients cannot hold every slot). Connections past their head are never evicted. |
| More than `--max-connections` open connections | 503 `overloaded`, `Retry-After: 1` |
| Priority-class queue is full (`--max-queue-per-class`) | 429 `overloaded`, `Retry-After: 1` |
| A queued request reaches its `deadline_ms` before upstream admission | 504 `deadline_exceeded`; no backend request was made |
| The OS refuses to start a chat request's disconnect-watcher thread (thread or memory limits) | 503 `overloaded`, `Retry-After: 1`; nothing was executed and only that request fails |
| The process is out of file descriptors (`EMFILE`) | 503 `overloaded` ("out of file descriptors") through a reserved descriptor; logged once per episode; the accept loop backs off instead of spinning |
| `Expect` other than `100-continue` | 417 `expectation_failed` (`param` `Expect`) |

`Expect: 100-continue` (curl sends it for bodies of 1 MiB and more) is
answered with `HTTP/1.1 100 Continue` once the request has passed every check
that does not need the body (Host, CORS, auth, route, 411, 413); a request
that fails one of them gets its error instead. HTTP/1.0 requests have
`Expect` ignored.

When the server answers before reading the whole request (401, 403, 404,
405, 411, 413, 417, 400/431 parse errors, 408), it half-closes and keeps
reading and discarding the rest of the request for up to 2 s and 16 MiB
(lingering close) before closing. Clients that write the whole body before
reading the reply (Python `urllib`) therefore receive the error instead of a
connection reset. The connection-cap 503 is sent from the accept thread,
usually before the request arrives, and lingers for up to 200 ms (ending
early after 100 ms without input, or as soon as the client closes). That
covers a client whose connect completes a Windows scheduler tick late, but a
client that sends a large body into a full server may still see a reset
there.

### Errors

Shape: `{"error": {"message", "type", "code", "param"}, "sonder": {"api_version": 1}}`.
`param` names the offending field or header when there is one, else `null`.
Invalid reasoning budget body values use 400 `invalid_json` and identify
`reasoning_budget_tokens`, `thinking_budget_tokens`, or
`reasoning_budget_message` in `param`. An invalid or duplicate budget header
uses 400 `invalid_correlation_header`, with `X-Sonder-Reasoning-Budget` in `param`.

How session failures map (`map_session_failure`): messages are validated
before the session runs (400 `invalid_messages`, `param` `messages`); a prompt
that can never fit the engine's KV pool is also 400 `invalid_messages`; a
sampler with no viable token is 400 `invalid_sampling`; a backend that refuses
a field ("… does not support logit_bias", as the Ollama backend does) is 400
`unsupported_parameter` with that field as `param`; any other refusal is 400
`invalid_request`. The 429 scheduler path is kept for forward compatibility
but is unreachable today: the request runtime clamps `max_new_tokens` to the
KV capacity and sets no sequence fingerprint, so the scheduler can only
reject with `never_fits` or `invalid_request`, both caller errors (400).

| Status | `type` | `code` |
| --- | --- | --- |
| 400 | `invalid_request_error` | `invalid_json`, `invalid_messages`, `unsupported_parameter`, `invalid_sampling`, `invalid_correlation_header`, `malformed_request`, `invalid_request` (bad `format` query, or the engine or backend refused a request parameter without naming it) |
| 401 | `authentication_error` | `unauthorized` |
| 403 | `permission_error` | `forbidden_origin`, `forbidden_host` |
| 404 | `not_found_error` | `model_not_found`, `not_found` |
| 405 | `invalid_request_error` | `method_not_allowed` (with `Allow`) |
| 408 / 411 / 413 / 417 / 431 / 505 | `invalid_request_error` | `request_timeout` / `length_required` / `payload_too_large` / `expectation_failed` / `request_header_fields_too_large` / `http_version_not_supported` |
| 429 | `rate_limit_error` | `overloaded` (the telemetry subscriber cap; also reserved for an engine scheduler rejection before execution, which the current request runtime cannot produce, see below), `Retry-After: 1` |
| 500 | `server_error` | `internal_error` |
| 501 | `not_implemented_error` | `not_implemented` |
| 503 | `service_unavailable` | `not_ready`: returned only before admission, nothing executed, safe to send elsewhere. `backend_unavailable`: the backend failed while executing, not safe to replay. `overloaded`: connection cap |

## Routes

### `GET /v1/sonder/health`

200 when ready, 503 while starting or draining.

```json
{"status":"ready","api_version":1,"version":"0.1.0","commit":"912503a…","abi_version":1,
 "instance_id":"tel-…","node_id":"host","uptime_s":12.5,"synthetic":true,"auth_required":false,
 "backends":[{"name":"mock","available":true,"capabilities":["tokenization","streaming","deterministic"],"version":"mock-1"}],
 "models":[{"id":"mock:tiny","backend":"mock","default":true,"state":"resident","resident":true,"in_flight":0,
            "loads":1,"load_failures":0,"evictions":0,"idle_s":3.2,"model_instance_id":"…"}],
 "queued_by_class":{"interactive":0,"subagent":0,"background":0},
 "residency":{"mode":"eager","idle_ttl_s":null,"max_resident":null,"registered":1,"resident":1,"loading":0,
              "unloading":0,"loads":1,"load_failures":0,"evictions":0},
 "telemetry":{"level":"standard","subscribers":0,"retained":42,"capacity":8192,"emitted":42,"dropped":0,
              "subscriber_dropped_events":0},
 "sonder":{"api_version":1,"features":[],"pins":{"mode":"override"}}}
```

`backends[].available` and `version` come from the backend's `probe()`, run
once before `ready` and then every 2 s by a background refresher: health and
identity only read that cache and never wait on the backend (a hung Ollama
cannot stall them).
A backend that reports runtime observations (a spawned `llamaserver` child)
adds `backends[].runtime` with `gpu_memory`, `context` and `warnings`
([vram-spill](integration/vram-spill.md)); other backends' entries are
unchanged.
`telemetry.emitted` and `telemetry.dropped` are the bus counters;
`subscriber_dropped_events` counts events lost by slow live subscribers.
`sonder.features` is the backend capability list. For a llamaserver backend it
is exactly `["thinking","chat_template_kwargs","enable_thinking",
"reasoning_effort","reasoning_budget","prompt_cache_key","priority_classes"]`;
for the mock backend it is `[]`. `sonder.pins.mode` is `override` or `default`.

Model residency fields (additive; see [Model residency](#model-residency)):
`models[].state` is `unloaded`, `loading`, `resident` or `unloading` (evicted,
its handle still being released: `model.evicted` and `model.unload` have been
emitted once it reads `unloaded`, and a request for it waits for the release
before loading it again); `in_flight` counts the requests pinning the model
(including those waiting for its load);
`loads`, `load_failures` and `evictions` are cumulative; `idle_s` is the time
since the last request finished (null while pinned or not resident);
`model_instance_id` is null unless resident. `residency.mode` is `eager` or
`lazy`, `idle_ttl_s` and `max_resident` are null when off, and the counters
are totals over all models.

### `GET /v1/models`

```json
{"object":"list","data":[{"id":"mock:tiny","object":"model","owned_by":"sonder-inference",
  "sonder":{"backend":"mock","default":true,"synthetic":true}}],
 "sonder":{"api_version":1,"features":[],"pins":{"mode":"override"}}}
```

`sonder.runtime` is added to a model only when its backend reports runtime
status (same object as `backends[].runtime` in health). It describes the
backend process, not the model handle, so it is reported whatever the
model's residency state (lazy, loading or evicted). Top-level `sonder.features` uses the
same exact backend list as health: the seven llama-server features above, or
`[]` for mock.

With `--profile`, the served model's `sonder` object also carries `profile`
(context length, cache types, backend, capabilities, estimated VRAM), and the
top-level `sonder` object lists every profile of the file as `profiles`. Both
are additive and absent without a profile. `context_length` follows a
context the backend reduced at run time (`configured_context_length` then
keeps the profile's), and `capabilities` lists only what this endpoint
accepts (`upstream_capabilities` has the upstream's `vision`/`tools`); see
[launch profiles](integration/launch-profiles.md).

### `GET /v1/sonder/identity[?model=ID]`

```json
{"schema":"sonder.inference.identity/1","model":"mock:tiny","synthetic":true,
 "backend_identity":{"backend":"mock","model":"mock:tiny","model_digest":"<64 hex>","quantization":"none",
   "backend_version":"mock-1","tokenizer_digest":"<64 hex>","template_digest":"<64 hex>",
   "context_tokens":4096,"hardware":"mock backend on cpu:0 …; no inference is performed"},
 "reason":null,"sonder":{"api_version":1}}
```

`backend_identity` has exactly the nine keys of Sonder Runtime's
`BackendIdentity`. Digests are lowercase 64-hex SHA-256 and `context_tokens`
is greater than 0. When any value cannot be measured, `backend_identity` is
`null` and `reason` says why; nothing is fabricated.

- `mock`: digests of fixed descriptor strings (model name and format, the
  mock tokenizer, the generic chat template); `synthetic` is true. Synthetic
  identities must never satisfy identity-bound routing evidence.
- `ollama`: always `null` in v1. The Ollama API does not expose a measurable
  tokenizer digest, and a partial identity would fail `BackendIdentity`
  validation.
- `llamacpp`: `null`, because GGUF file hashing is not implemented.

Unknown model: 404 `model_not_found`.

### `POST /v1/chat/completions`

OpenAI-compatible subset. Every request runs through `Session::chat`, one
session per HTTP request, so it goes through the scheduler and KV accounting
and emits the full request, scheduler and inference telemetry with
`request.queued.kind = "chat"`. Backends with a native chat API or template
(Ollama, llama.cpp GGUF templates) receive the messages; the mock gets the
generic formatted prompt.

Accepted fields: `model` (an id or `default`; absent means `default`),
`messages` (`[{role: system|user|assistant|tool, content: string}]`, the last
from `user` or `tool`), `stream`, `stream_options.include_usage`,
`max_tokens` or `max_completion_tokens` (the latter wins; default 1024),
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop` (a string or at most
4 strings), `presence_penalty`, `frequency_penalty`, `repeat_penalty`,
`logit_bias` (`{"<token id>": bias}`), `n` (1 only), and the Sonder extensions
`num_ctx`, `typical_p`, `repeat_last_n`, `reasoning_budget_tokens`,
`thinking_budget_tokens`, and `reasoning_budget_message`. Other sampling defaults are
`SamplingConfig`'s (temperature 0.8, top_p 0.95, top_k 40, repeat_penalty
1.1).

Scheduling fields are additive: `priority` is `interactive` (default),
`subagent`, or `background`; `deadline_ms` is a positive integer no greater
than `INT32_MAX`, measured from connection arrival. The corresponding headers
are `X-Sonder-Priority` and `X-Sonder-Deadline-Ms`; when both a header and body
field are present, both are validated and the header wins. `X-Sonder-Priority`
also accepts the existing integer range -16 to 16: it keeps its scheduler
priority and integer `request.queued.attributes.priority` telemetry, and selects
the `interactive` admission class even when the body names another class. A
class-name header selects that admission class and leaves the numeric scheduler
priority at its default, zero. Invalid integers or unknown class names return
400 `invalid_correlation_header`. Request hints alone do not enable server-wide
priority admission; deadlines are enforced independently of admission mode.
Numeric headers retain the scheduler rank `workload rank - priority`, including
explicit zero, even with priority admission enabled. Class/FIFO admission still
controls entry to the backend; numeric values do not move a request ahead of an
earlier request in the same admission class.

Request-path extensions (all optional; see the sections below):

- `prompt_cache_key` (OpenAI; a 1-256 byte string): conversation key for
  upstream prompt-cache affinity. Without it the key is derived from
  `X-Sonder-Run-Id` and `X-Sonder-Agent-Id` (`run=<id>;agent=<id>`, either
  part may be absent); with neither there is no affinity.
- `chat_template_kwargs` (object): `enable_thinking` (boolean) and
  `reasoning_effort` (string of 1-64 `[A-Za-z0-9._-]`) are forwarded to
  native-chat backends; other keys are ignored. A non-object, or a key of the
  wrong type, is 400 `invalid_json` naming it.
- `think` (boolean, Ollama's name): same as
  `chat_template_kwargs.enable_thinking`; both set and disagreeing is 400.
- `reasoning_budget_tokens` (signed 64-bit integer >= -1) is the canonical per-request
  reasoning budget. `thinking_budget_tokens` is an alias; when both are
  present, both are validated and the canonical field wins. `reasoning_budget_message`
  is a UTF-8 string of at most 512 bytes. `X-Sonder-Reasoning-Budget` accepts
  the integer budget and wins over either body field. Invalid values return
  400 with the offending field or header named in `param`.
- `messages` assistant entries may include `reasoning_content` (string). The
  field is forwarded to llamaserver history and dropped by other backends.

Reasoning budget pins (`--pin-reasoning-budget` and
`--pin-reasoning-budget-message`) fill only unset request values. For a backend
that does not support the budget, the request still succeeds and
`sonder.warnings` reports that the budget was ignored; this is not a 400.
On this bundled llama.cpp build, budget `0` does not disable thinking (see PR
#28356). To turn thinking off, send `chat_template_kwargs.enable_thinking:
false` (and use `--pin-mode default` if an operator pin should remain a
fallback).

Ignored: `user` and any unknown top-level field.
Rejected with 400 `unsupported_parameter`: `tools`, `tool_choice`,
`functions`, `function_call`, `response_format`, `logprobs: true`,
`top_logprobs`, `n` other than 1, non-string message `content`, and
`tool_calls` on a message. Sampling values outside `validate(SamplingConfig)`
ranges are 400 `invalid_sampling` naming the field.

Non-streaming response:

```json
{"id":"chatcmpl-req-…","object":"chat.completion","created":1790000000,"model":"mock:tiny",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…"},"finish_reason":"length"}],
 "usage":{"prompt_tokens":9,"completion_tokens":8,"total_tokens":17},
 "timings":{"prompt_n":9,"predicted_n":8,"total_ms":0.4,"ttft_ms":0.1,"predicted_ms":0.3,"predicted_per_second":26000},
 "sonder":{"api_version":1,"request_id":"req-…","session_id":"sess-…","backend":"mock","synthetic":true,
           "token_counts_from_backend":true}}
```

- `model` is always the resolved id, never `default`.
- `finish_reason`: `length` (max tokens), `cancelled`, else `stop`.
- `timings` always has `prompt_n`, `predicted_n` and `total_ms`; `ttft_ms`
  appears when a chunk was produced, and `prompt_ms`, `predicted_ms`,
  `prompt_per_second`, `predicted_per_second` only when the backend measured
  them.
- Upstream cache observations are additive and appear only when the backend
  reported them (llamaserver `cache_n` / `prompt_tokens_details`, Ollama
  `prompt_eval_cached_count`); a reported 0 is a real miss, an absent field
  means "not reported": `usage.prompt_tokens_details.cached_tokens` (part of
  `prompt_tokens`), `timings.cache_n` (same count, llama-server's name),
  `timings.draft_n` and `timings.draft_n_accepted` (speculative decoding).
  `timings.queue_ms` is the scheduler's submit-to-admission time and appears
  for scheduled requests only.
- `sonder.warnings` (array of strings) appears when the server changed or could
  not apply a request option, including an `override`-mode thinking pin,
  `empty_content_at_length`, or a reasoning budget ignored by a backend that
  does not support it.

With `"stream": true` the response is `text/event-stream`: `data:` lines
carrying `chat.completion.chunk` objects (the first with
`delta.role = "assistant"`, then `delta.content` pieces), a final chunk with
`finish_reason` plus `timings` and `sonder` (and `usage` when
`stream_options.include_usage` is true), then `data: [DONE]`. Headers are sent
with the first chunk, so a request rejected before generation still gets its
proper status and error body. A backend failure after chunks were sent
arrives as one `data: {"error": …}` event and the stream ends without
`[DONE]`.

A client disconnect (detected while generating, streaming or not) cancels the
session and propagates cancellation to the upstream backend. A running
non-streaming request returns the normal cancellation response; a running
stream ends with its normal partial `cancelled` finish. A queued request that
reaches its deadline returns 504 before upstream execution.

#### Scheduling

Every chat request goes through the engine scheduler unless `--scheduler off`.
How it is paced depends on the backend (`SchedulerMode`,
[engine wiring](integration/engine-wiring.md)):

- `gate` (in-process backends under `automatic`: mock, llamacpp): each
  generated chunk waits for a scheduler grant, and KV is accounted for the
  prompt and the output. When priority admission is enabled, hosted requests
  also wait before the backend call and reserve full logical prompt/output
  capacity up front to avoid preempting a running request. With default server
  options the existing incremental KV and scheduling policy is unchanged.
- `account` (remote-process backends under `automatic`: llamaserver,
  ollama): the request is admitted by the engine scheduler and its prompt is
  accounted (prefix reuse, `queue_ms`), then the backend streams with no
  per-chunk grant. The upstream owns the real KV cache and batching, so a
  prompt larger than the logical pool (`--kv-pool-tokens`, 65,536 tokens by
  default) runs unscheduled (`scheduler.bypassed`) instead of being refused;
  the upstream's own context limit still applies.

`--scheduler gate` forces per-chunk gating everywhere (the old behaviour for
remote backends, including the 400 for a prompt over the pool) and
`--scheduler account` forces admission-only accounting everywhere.

Priority admission is opt-in. `--priority-admission auto` (the default) enables
it only when `--max-concurrent-subagent`, `--max-concurrent-background`, or
`--max-queue-per-class` is nonzero. `on` explicitly enables it; `off` bypasses
the priority queue and its caps. Without caps, default `auto` preserves the
existing concurrency and engine scheduling policy for both known and unknown
backend capacities. It creates no admission tickets, and `queued_by_class`
stays zero. Per-request priority or deadline hints do not switch it on.

When enabled, admission applies in every scheduler mode, including `off`.
A single ticket joins backend-slot/class eligibility with logical KV admission.
A request waiting for KV holds no backend slot, and queue caps and the
`queued_by_class` health gauge include that wait. Eligible requests are selected
by class (`interactive` > `subagent` > `background`) and FIFO within a class,
including when prompt preparation finishes out of order. Running requests are
not preempted. A class at its concurrency cap does not block another eligible
class. All class and queue caps default to zero/unlimited.

With admission enabled, `--backend-capacity N` supplies an explicit capacity.
Its default, zero, uses the backend's advertised `max_concurrent_requests`
(llama-server reports cached `/props.total_slots`). If neither source supplies
a capacity, Sonder adds no backend-capacity limit; the upstream keeps its own
parallelism, and separately configured class/queue caps still apply. No capacity
of one or engine running limit is substituted for unknown capacity. Existing
engine scheduler limits remain independent. For an unknown-capacity backend,
use `--priority-admission on --backend-capacity 2` to opt into a two-request
limit. A capacity flag alone does not activate `auto`.

Deadlines are positive integers up to 2,147,483,647 milliseconds measured on a
monotonic clock from connection arrival (including body read time). `null`,
booleans, negative values and fractional values are invalid. The disconnect
watcher polls every 20 ms; once running, deadline expiry uses the same
cooperative cancellation as disconnect. Upstream HTTP reads also poll their
cancellation token, including a silent prefill, then close the connection.

#### Prompt cache affinity

llamaserver reuses a prompt prefix only inside the llama-server slot that
last processed it. The conversation key above pins each conversation to one
slot (`id_slot`), and every native llama-server request sends
`"cache_prompt": true`. See [llama-server](integration/llama-server.md#prompt-cache-and-slot-affinity).

#### Thinking control

`chat_template_kwargs.enable_thinking` / `reasoning_effort` and `think` are
forwarded only when the request (or a pin) sets them: llama-server gets
`chat_template_kwargs`, Ollama gets `think` (it has no portable
`reasoning_effort`, which is not sent). Cache reuse depends on the model's
chat template: an effort change can invalidate the prefix, while a thinking
toggle can affect only the prompt tail. Measure reuse for the served template.
`--pin-enable-thinking` / `--pin-reasoning-effort` supply server-wide defaults.
Their behavior is controlled by `--pin-mode`:

| Mode | Unset request field | Explicit request field | Warning |
| --- | --- | --- | --- |
| `override` (default) | pin is applied | pin replaces a conflicting request value | `sonder.warnings` records each conflict; agreeing values add no warning |
| `default` | pin is applied | request value wins | no pin warning |

The budget pins always fill only unset budget fields, in either mode. A pin is
an operator decision about the served prompt shape (like llama-server's
`--ctx-size`); `default` is the mode for callers such as JSON-only planner or
reviewer steps that must explicitly disable thinking.

The regular top-level `timings` object and existing statistics remain
unchanged. When llama-server supplies its optional final `timings` object, the
same upstream values are also exposed additively under
`usage.sonder.timings`: `prompt_n`, `cache_n`, `prompt_ms`, `predicted_n`,
`predicted_ms`, `draft_n`, and `draft_n_accepted`. The object is omitted when
the upstream supplies no timings. If `cache_n` is present and
`usage.prompt_tokens_details.cached_tokens` was not already set, the latter is
filled from `cache_n` (including a reported zero).

For example, a llama-server response may add this nested usage object while
retaining the existing top-level `timings` object:

```json
{"usage":{"prompt_tokens":13,"completion_tokens":8,"total_tokens":21,
 "prompt_tokens_details":{"cached_tokens":4},
 "sonder":{"timings":{"prompt_n":9,"cache_n":4,"prompt_ms":2.1,
                         "predicted_n":8,"predicted_ms":31.4,"draft_n":10,
                         "draft_n_accepted":7}}}}
```

When `finish_reason` is `length` and the assistant `content` is empty, the
response carries the additive warning `empty_content_at_length` in
`sonder.warnings`.

Correlation request headers (all optional; values must match
`[A-Za-z0-9._:-]{1,128}`, otherwise 400 `invalid_correlation_header`):

| Header | Effect |
| --- | --- |
| `X-Sonder-Run-Id` | Envelope `run_id` (default: the engine id). |
| `X-Sonder-Parent-Request-Id` | `attributes.parent_request_id` on `request.queued`, `started`, `completed`, `cancelled` and `failed`. |
| `X-Sonder-Agent-Id`, `X-Sonder-Task-Id` | Envelope `agent_id`, `task_id`. |
| `X-Sonder-Workload` | One of the 7 workload classes; default `interactive_user`. |
| `X-Sonder-Priority` | Integer from -16 to 16 (existing scheduler priority and integer telemetry; admission class `interactive`), or `interactive`, `subagent`, or `background` (class selection). Header wins over body `priority`; no hint defaults to numeric `0` and class `interactive`. |
| `X-Sonder-Deadline-Ms` | Positive integer up to `INT32_MAX`, measured from connection arrival; a queued request past it returns 504. |

The envelope `request_id` is always Inference's own id (`req-…`), also
returned as `sonder.request_id`, in the completion id and in
`X-Sonder-Request-Id`.

### `POST /v1/messages`

The server also accepts the Anthropic Messages API shape for text-only local
backends. `max_tokens` is required; `model` defaults to the served `default` alias. `messages`
accepts user and assistant turns whose content is either a string or an array
of `{ "type": "text", "text": "..." }` blocks; `system` accepts a string or
the same text-block array. `temperature`, `top_p`, `top_k`, `stop_sequences`,
`stream`, and `metadata.user_id` are supported. A `metadata.user_id` becomes
the prompt-cache session key; when it is absent, the same
`X-Sonder-Run-Id`/`X-Sonder-Agent-Id` headers used by chat provide affinity.
The current session validator requires the final input turn to be from the
user, so a request ending in an assistant turn is rejected with 400.

Image blocks are rejected with 400 because the current backends are text-only.
`tools` and `tool_choice` are also rejected with 400; tool execution is not
implemented. Unknown fields are ignored. Authentication is the same as for
the other `/v1` routes: a configured token may be sent as `Authorization:
Bearer <token>` or, for Anthropic clients, in `x-api-key`.

Thinking controls map to the backend's existing chat-template controls:
`thinking: {"type":"enabled","budget_tokens":N}` enables thinking and
`{"type":"disabled"}` disables it. The budget is validated as an Anthropic
compatibility field but is not an independent token budget in this release.
The optional `reasoning_effort` or `output_config.effort` values `off`, `low`,
`medium`, and `high` map to disabled, low, medium, and `xhigh` backend effort
respectively. Low, medium and high also enable thinking. The two effort
aliases must agree when both are present. Thinking and effort may be combined
when their enabled/disabled state agrees; contradictory values (for example
enabled thinking with effort `off`) return 400.
Server-wide `--pin-enable-thinking` and `--pin-reasoning-effort` override a
request value; the response carries the same `sonder.warnings` entries as the
chat endpoint when that happens.

Non-streaming responses use Anthropic's `message` shape with `msg_` ids,
`content` text blocks, `stop_reason` (`end_turn`, `max_tokens`, or
`stop_sequence`), and `usage.input_tokens`/`output_tokens`. `input_tokens`
starts from the backend's measured `prompt_tokens`; when cached tokens are
reported, the cache count is subtracted and emitted separately as
`cache_read_input_tokens`. No cache count is guessed or subtracted when it is
unknown, and backend prompt-count conventions may differ from Anthropic's
billing counters. `output_tokens` maps the backend's measured completion token
count. The cache field is omitted when the backend does not report one.
Separately returned reasoning is
represented by a `thinking` block before the text block. `stream: true` emits
Anthropic SSE events in order: `message_start`, content block start/deltas and
stop, `message_delta`, and `message_stop`. The existing chat stream has no
keepalive timer. `message_start` uses zero preliminary usage; final measured
counts and pin warnings are carried in `message_delta`. A midstream failure
emits `event: error` and closes without a success `message_stop`.

`stop_sequence` is the exact matching sequence when known, otherwise `null`;
it is always `null` for other stop reasons. The thinking budget must be a
positive integer at most 1,048,576. `metadata.user_id` is a non-empty string
of at most 256 bytes. Unknown top-level fields are ignored; unsupported
content block types are rejected. Locally produced thinking has no Anthropic
signature and signed-thinking replay is not supported.

Errors on this endpoint use `{ "type": "error", "error": { "type": "...",
"message": "..." } }` and preserve the server's HTTP status and retry headers.
For example invalid requests return 400 `invalid_request_error`, invalid
tokens return 401 `authentication_error`, and oversized bodies return 413
`request_too_large`. Transport failures before a request path is known
(including connection-capacity rejection) use the generic server error format.

```sh
curl http://127.0.0.1:11437/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","max_tokens":128,"messages":[{"role":"user","content":"Say hello"}],"output_config":{"effort":"high"}}'
```

### `POST /v1/embeddings`

501 `not_implemented`.

## Live telemetry

### `GET /.well-known/sonder-telemetry`

```json
{"schema":"sonder.telemetry.producer/1",
 "producer":{"name":"sonder-inference","version":"0.1.0","node_id":"host","instance_id":"tel-…",
             "role":"inference","synthetic":true},
 "event_schema":"sonder.observatory.event/1",
 "streams":[{"transport":"sse","url":"/v1/telemetry/sse"},{"transport":"ndjson","url":"/v1/telemetry/ndjson"}],
 "resume":{"header":"Last-Event-ID","query":"last_event_id","retained_events":42,"oldest_sequence":0,"next_sequence":42},
 "auth":{"required":false,"schemes":["bearer"]},"clock":{"mono_ns":"host-monotonic"},
 "sampling_level":"standard","text_capture":"off",
 "links":{"health":"/v1/sonder/health","identity":"/v1/sonder/identity","models":"/v1/models"},
 "vocabularies":{"sonder.inference.events":1},
 "stats":{"subscribers":0,"max_subscribers":8,"capacity":8192,"subscriber_dropped_events":0},
 "sonder":{"api_version":1}}
```

Stream and link URLs are relative to the discovery URL. `producer.instance_id`
equals the envelopes' `producer.instance_id` and the `event_id` prefix.

### `GET /v1/telemetry/sse`, `/v1/telemetry/ndjson`, `/v1/telemetry`

- Format: the explicit path, then `?format=sse|ndjson`, then `Accept`
  (`application/x-ndjson` selects NDJSON).
- SSE (`text/event-stream; charset=utf-8`): first `retry: 2000`, then per
  event `id: <event_id>` and `data: <envelope on one line>` and a blank line.
  No `event:` field. `: keepalive` every 15 s when idle.
- NDJSON (`application/x-ndjson`): one envelope per line; a blank line every
  15 s when idle.
- Resume: `Last-Event-ID` wins over `?last_event_id=`. Same instance and
  retained: replay after it. Same instance but older than the window:
  `: resume-gap <from>-<to>` (SSE), then the whole window. Unknown instance
  (producer restarted), malformed or no id: the whole window. `?since=now`:
  live only.
- At most 8 concurrent streams; the ninth gets 429 `overloaded` with
  `Retry-After: 1`.
- A slow subscriber loses its oldest undelivered events: the loss shows as a
  sequence gap, SSE subscribers also get `: dropped <n>`, and health and
  discovery count it as `subscriber_dropped_events`. It is not sent as
  `telemetry.dropped` (that event keeps meaning bus-queue drops seen by every
  consumer). The telemetry bus writer never waits on a subscriber: the hub
  is a `TelemetrySink` with a bounded ring and bounded per-subscriber queues,
  and each stream thread does its own socket writes.
- Streams end after the engine stops (they deliver `engine.stopped` first).

## Hooking up Observatory

```sh
sonder-infer serve --backend mock --port 11437
# Observatory dev server (http://127.0.0.1:5173) is in the default allowlist:
#   open http://127.0.0.1:5173/?fixture=0&connect=http://127.0.0.1:11437
# A preview on another origin needs:  --cors-origin http://127.0.0.1:4173
```

Observatory fetches `/.well-known/sonder-telemetry`, picks the SSE stream and
resumes with `Last-Event-ID` on reconnect. With `--token-file`, Observatory
must send the token as a bearer header; it never goes in a URL.

## Curl examples

```sh
curl -s http://127.0.0.1:11437/v1/sonder/health
curl -s http://127.0.0.1:11437/v1/models
curl -s 'http://127.0.0.1:11437/v1/sonder/identity?model=default'
curl -s http://127.0.0.1:11437/v1/chat/completions -H 'Content-Type: application/json' \
  -H 'X-Sonder-Run-Id: turn-1' -H 'X-Sonder-Parent-Request-Id: turn-1' \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
curl -sN http://127.0.0.1:11437/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}'
curl -sN http://127.0.0.1:11437/v1/telemetry/sse
curl -sN -H 'Last-Event-ID: tel-0123456789abcdef-41' http://127.0.0.1:11437/v1/telemetry/sse
curl -sN 'http://127.0.0.1:11437/v1/telemetry/ndjson?since=now'
```

## Tests

`sonder.server.*` (`src/server/tests`) runs in-process servers on ephemeral
loopback ports with the mock backend and a raw-socket client: response
shapes, every error code, auth, the Host check, CORS, request limits, the
connection and subscriber caps, chat (streaming and not), correlation
headers in envelopes, disconnect cancellation, SSE and NDJSON framing,
heartbeats, resume, backpressure (NDJSON counters and the SSE `: dropped <n>`
comment with its matching sequence gap), 503 `not_ready` during a drain,
lingering close for clients that send the whole body first, `Expect:
100-continue`, duplicate `Host` headers, descriptor exhaustion (503 through
the reserve descriptor, no CPU spin), a backend whose `probe()` and
`load_model()` block (health stays responsive; `stop()` during `start()`),
model residency (defaults load everything before ready and never unload;
lazy loading on first use; one load for concurrent first requests; idle-TTL
eviction and reload; a pinned model outlives the TTL; the max-resident cap),
and `serve_main` (help, usage errors, ready file written and removed, port in
use, wildcard-bind banner, graceful shutdown). No network services or weights
are needed. `fuzz/fuzz_http_request.cpp` fuzzes the request head parser and the
chat request mapper.

## Deviations from the contract text

These follow the contract review, which overrides the contract where they
conflict:

- Default CORS origins apply to read-only GET routes only; POST needs an
  explicit `--cors-origin` or a token. `--capture-text` requires a token.
- Per-subscriber losses are reported per subscriber (`: dropped <n>`, sequence
  gaps, `subscriber_dropped_events`), not as `telemetry.dropped` on the shared
  stream.
- Every JSON body carries `sonder.api_version` (Runtime's transport cannot
  read response headers).
- `num_ctx`, `typical_p` and `repeat_last_n` are accepted as extensions, and
  unknown top-level fields are ignored.
- Ollama identity is always `null` with a reason in v1.
- `model` in responses is the resolved id, never `default`.
- Discovery adds `vocabularies` and `stats`; health `telemetry` adds
  `subscriber_dropped_events`.
- Codes for conditions the contract table leaves open (400
  `malformed_request`, 405 `method_not_allowed`, 408, 411, 413, 431, 505) are
  listed in the error table above.

## Open questions

Recorded rather than invented (AGENTS.md):

1. **CLI argument parser.** The review asks `serve` to use
   `sonder::cli::parse_args` (`src/cli/cli_args.hpp`). That parser only knows
   two boolean flags and one repeatable option, and `src/cli` belongs to the
   `inf-cli-ux` lane, so `serve_main` has its own table-driven parser (strict
   numeric checks, `--key=value`, repeatable `--model`, `--model-dir` and
   `--cors-origin`). Folding both into one parser is left to `inf-cli-ux`.
2. **Windows. The Winsock paths mirror `src/net/http_client.cpp` but cannot
   be run in the Linux container; the `ci-windows` job must pass before merge.
3. **Producers on different hosts.** `mono_ns` merging is same-host (Linux)
   only; cross-host clock alignment is a non-goal for v1.
