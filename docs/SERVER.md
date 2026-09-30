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
| `--ready-file PATH` | none | Written atomically once the socket **listens** (models may still be loading: poll health for 200): `{"url","pid","instance_id","api_version":1}`. Removed when `serve` returns, after a failed start as well as after a clean shutdown. A process killed outright (second signal, SIGKILL) leaves it behind: check that `pid` is alive. |
| `--mock-delay-ms N` | `0` | Mock per-token delay. |
| `--log-format text\|json` | `text` | Banner, access log and diagnostics on stderr. |

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
Retry-After`. Every JSON body carries `"sonder": {"api_version": 1, ...}` so
clients that cannot read response headers can still check the API version.

The preflight (`OPTIONS`) answers 204 with
`Access-Control-Allow-Methods: GET, POST, OPTIONS`,
`Access-Control-Allow-Headers: Accept, Authorization, Cache-Control,
Content-Type, Last-Event-ID, X-Sonder-Run-Id, X-Sonder-Parent-Request-Id,
X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority`
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
 "models":[{"id":"mock:tiny","backend":"mock","default":true}],
 "telemetry":{"level":"standard","subscribers":0,"retained":42,"capacity":8192,"emitted":42,"dropped":0,
              "subscriber_dropped_events":0},
 "sonder":{"api_version":1}}
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

### `GET /v1/models`

```json
{"object":"list","data":[{"id":"mock:tiny","object":"model","owned_by":"sonder-inference",
  "sonder":{"backend":"mock","default":true,"synthetic":true}}],"sonder":{"api_version":1}}
```

`sonder.runtime` is added to a model only when its backend reports runtime
status (same object as `backends[].runtime` in health).

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
`num_ctx`, `typical_p`, `repeat_last_n`. Other sampling defaults are
`SamplingConfig`'s (temperature 0.8, top_p 0.95, top_k 40, repeat_penalty
1.1).

Ignored: `user`, `chat_template_kwargs`, and any unknown top-level field.
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
session: the request ends with `request.cancelled`.

Correlation request headers (all optional; values must match
`[A-Za-z0-9._:-]{1,128}`, otherwise 400 `invalid_correlation_header`):

| Header | Effect |
| --- | --- |
| `X-Sonder-Run-Id` | Envelope `run_id` (default: the engine id). |
| `X-Sonder-Parent-Request-Id` | `attributes.parent_request_id` on `request.queued`, `started`, `completed`, `cancelled` and `failed`. |
| `X-Sonder-Agent-Id`, `X-Sonder-Task-Id` | Envelope `agent_id`, `task_id`. |
| `X-Sonder-Workload` | One of the 7 workload classes; default `interactive_user`. |
| `X-Sonder-Priority` | Integer from -16 to 16. |

The envelope `request_id` is always Inference's own id (`req-…`), also
returned as `sonder.request_id`, in the completion id and in
`X-Sonder-Request-Id`.

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
