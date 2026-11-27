# Ollama Cloud provider (`ollama_cloud`)

Module `src/backends/ollama_cloud` (branch `claude/ret-02-ollama-cloud`,
2026-11-26). Sonder Inference serves Ollama's hosted cloud models directly over
HTTPS. There is no local Ollama daemon anywhere on the path: Sonder Inference
is the client of `https://ollama.com` itself.

This is the only Ollama piece that survives the retirement of the local Ollama
daemon and of the `ollama` backend (direction of 2026-11-26). The module is
self-contained: it does not link against `src/backends/ollama`, so deleting that
module does not affect it.

## Upstream API (verified 2026-11-26)

| What | Value | Source |
| --- | --- | --- |
| Base URL | `https://ollama.com`, API under `/api` | https://docs.ollama.com/api/introduction.md ("Base URLs") |
| Auth | `Authorization: Bearer <API key>`; required for direct cloud access | https://docs.ollama.com/api/authentication.md, https://docs.ollama.com/cloud/api.md |
| Chat | `POST /api/chat`, NDJSON stream (`message.content`, `message.thinking`, `message.tool_calls`, final line `done: true` with `done_reason` and the `*_count` / `*_duration` fields) | https://docs.ollama.com/api/chat.md, https://docs.ollama.com/api/streaming.md |
| Model list | `GET /api/tags`; for direct requests use the names it returns (e.g. `gpt-oss:120b`, not the `:cloud` alias a local daemon uses) | https://docs.ollama.com/cloud/models.md, https://docs.ollama.com/api/tags.md |
| Errors | non-2xx with `{"error": "..."}`; mid-stream errors arrive as an `{"error": ...}` NDJSON line with status 200; 429 = rate limited, 502 = cloud model unreachable | https://docs.ollama.com/api/errors.md |
| Structured outputs | "Cloud models currently do not support structured outputs" | https://docs.ollama.com/cloud.md |

Not verified against the cloud specifically (documented only for the Ollama
API in general, and not tried here because tests make no network calls):

- `POST /api/generate` on `ollama.com`. The provider uses it for
  `BackendModel::generate()` (raw prompt completion). Chat, which is what
  `sonder-infer serve` uses for `/v1/chat/completions`, `/v1/responses` and the
  Anthropic route, goes through `/api/chat`.
- Whether `GET /api/tags` on the cloud needs the key. The docs show it without
  one; the provider always sends the key to the configured host anyway.
- `keep_alive`, `/api/show` and `/api/embed` on the cloud: not used.

## Configuration

`sonder-infer serve --backend ollama_cloud --ollama-cloud-config PATH`. Without
`--model`, every model in the allowlist is served (the first is the default).

```json
{
  "base_url": "https://ollama.com",
  "api_key_env": "OLLAMA_API_KEY",
  "models": ["gpt-oss:120b", "qwen3-coder:480b"],
  "connect_timeout_ms": 10000,
  "request_timeout_ms": 900000,
  "idle_timeout_ms": 120000
}
```

| Field | Meaning |
| --- | --- |
| `base_url` | Optional, default `https://ollama.com`. Must be `https://`; an optional port and path prefix are kept (`/api/...` is appended, so do not end it with `/api`). No userinfo or query. Without `allow_custom_endpoint` only the Ollama Cloud origin is accepted (see Endpoint). |
| `allow_custom_endpoint` | Optional, default `false`. `true` accepts a `base_url` on any other host or port. **The API key is sent to that host on every request.** |
| `api_key_env` | NAME of an environment variable holding the key (`OLLAMA_API_KEY` is Ollama's own convention). |
| `api_key_file` | Path of a file holding only the key; surrounding whitespace is trimmed. Restrict its permissions yourself; the provider does not check ACLs. |
| `models` | Required, non-empty: the cloud allowlist (see Policy). |
| `connect_timeout_ms` | 1..600000, default 10000. |
| `request_timeout_ms` | Whole-request cap, 1..86400000, default 900000. |
| `idle_timeout_ms` | Longest silence between received bytes, 0..86400000 (0 = off), default 120000. It also bounds the wait for the response head: a request that gets no body byte within this time fails. |
| `max_response_bytes` | Cap on the raw bytes of one response (status line, headers, chunk framing, trailers and body, counted as received whether or not they carry output), 1..67108864, default 67108864 (64 MiB). It cannot be disabled; exceeding it fails the request with `protocol_error`. |
| `ca_bundle_path` | Optional PEM CA bundle that **replaces** the system trust store; it does not add to it. A bundle that lacks the roots for the configured host (for `https://ollama.com`, its public CA chain) makes every request fail verification. Leave it unset for `ollama.com`. Verification stays on. It never changes where the key goes: the host is always the one `base_url` names, and the endpoint check below applies unchanged. Whoever controls a CA in the bundle can, with control of the network path, impersonate that host, so only list CAs you trust with the key. |

Exactly one of `api_key_env` and `api_key_file` is required. Unknown fields
are rejected. An inline `api_key` (or `key`, `token`) field is refused, so the
key never sits in a config file next to the rest of the settings and never
travels on a command line. There are no TLS options other than
`ca_bundle_path`: `insecure_skip_verify`, pins and `server_name` are unknown
fields here.

C++ hosts build the backend with `make_ollama_cloud_backend(OllamaCloudOptions)`
(`sonder/inference/backends/ollama_cloud.hpp`); the options mirror the JSON and
add a pluggable `transport` for tests.

## Endpoint

The provider sends the API key in the `Authorization` header of every request
to the host that `base_url` names, so that host is trusted with the key.

- By default only the Ollama Cloud origin is accepted: host `ollama.com`
  (ASCII case-insensitive; no subdomain, no trailing dot) on port 443, any
  path. Anything else (another host, an IP literal, `ollama.com:8443`,
  `api.ollama.com`, `ollama.com.example`) fails at config load and again in
  `make_ollama_cloud_backend` with `invalid_argument` and a fixed message that
  names `allow_custom_endpoint` and does not quote the URL. The check runs
  before the key is read and before `serve` logs the URL.
- `"allow_custom_endpoint": true` (`OllamaCloudOptions::allow_custom_endpoint`)
  accepts any other `https://` host. It means "the key is sent to this host";
  use it only for a host you trust with the key (a gateway you run, say).
- `ca_bundle_path` cannot widen this: it only changes which CAs are trusted
  for the configured host.

## Trust model

The configured provider is trusted with its own credential. It receives the
key on every request, so a provider that wanted the key already has it; the
provider is **not** filtered for echoing it back. Content, reasoning and tool
calls from a successful response reach the client and text-capture telemetry
unmodified, and there is no output filter for the key (a provider that echoes
it in a completion would show it to the local client). That is why the
endpoint is restricted (Endpoint above): the key can only be sent to, and so
only be echoed by, the Ollama Cloud origin unless an operator opts in to a
host of their choosing. A third party on the network path is excluded by
mandatory TLS verification (chain and host name; no insecure mode, no pin-only
mode, no SNI override), so it can neither read the key nor inject responses.

What the provider says on error is a different matter: error text is never
needed for the result, so none of it is passed on (Secrets below).

## Policy

- **Opt-in.** A model is a cloud model only when it is listed in `models`.
  `load_model()` refuses anything else with `invalid_argument` ("is not
  configured as a cloud model"), even when the cloud offers it; a listed model
  the cloud does not offer fails with `not_found`. `list_models()` returns the
  allowlisted models present in `GET /api/tags`, with `resident = false`.
- **Local-only requests are refused.** `sonder-infer serve` refuses a request
  carrying `X-Sonder-Local-Only: 1|true` against the `ollama_cloud` backend
  before it resolves or acquires the model: loading a model that is not
  resident (`--lazy-models`, or after an idle eviction) is itself an
  authenticated `GET /api/tags`, so the refusal must come first. That covers
  `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`,
  `/v1/embeddings`, `/v1/rerank`, `/v1/score`, `/v1/tokenize` and
  `/v1/detokenize`: 400 `invalid_request`
  (param `X-Sonder-Local-Only`), "backend 'ollama_cloud' is a remote provider
  and the request is local-only (X-Sonder-Local-Only); nothing was sent to the
  provider". No network call is made. The provider itself refuses
  `GenerateRequest::local_only` / `ChatRequest::local_only` (set from
  `RequestOptions::local_only`) with `invalid_argument` before any network
  I/O as a second layer, for C++ hosts that call it directly. Only
  `ollama_cloud` enforces the flag; the other backends ignore it, including
  the retiring `ollama` (`--ollama-allow-remote`) and external-mode
  `llamaserver` backends, which can still send a prompt off-host.
- **Refused features** (no network call): `response_format` other than `text`
  (cloud models have no structured outputs), GBNF grammars, image inputs,
  `logit_bias`, `typical_p`. Thinking: `ThinkingOptions::enable_thinking` maps
  to `think`; `reasoning_effort` is not sent.
- **Cancellation.** The cancellation token is polled by the transport while it
  waits and on every streamed line; a cancelled request returns `cancelled`
  promptly. A request that is already cancelled sends
  nothing.

## TLS

The default transport is the internal HTTP client (`src/net`) with
`use_tls = true`, `insecure_skip_verify = false`, no pins and no SNI override:
the chain is verified against the system trust store (or `ca_bundle_path`) and
the certificate must match the host name. On Windows that is Schannel, elsewhere
OpenSSL (docs/integration/tls.md). No new dependency.

TLS is compiled only with `-DSONDER_WITH_TLS=ON`. In a build without it the
module still builds (and its tests run with the fake transport), but every real
request fails with `unsupported` ("this build has no TLS; configure with
-DSONDER_WITH_TLS=ON"). The default presets have TLS off, so a deployment that
uses Ollama Cloud needs a TLS-enabled configure.

## Secrets

- The key is read once, when the backend is built. Error messages name the
  variable or the file path, never the value.
- It is sent only as the `Authorization` header to the configured host (see
  Endpoint). The provider's copies (the stored key, the header values handed
  to the transport) are overwritten when they are released, also when the
  transport throws. The HTTP client sizes its serialized request head once
  (no reallocation leaves a copy in freed memory), appends header values in
  place and overwrites the head right after sending it and on every other
  exit path.
- **Remote error text is never passed on.** A non-2xx response body is not
  read at all (the connection is closed at the first body byte), and a
  mid-stream `{"error": ...}` line is not copied. Errors carry fixed local
  text instead: `ollama_cloud: HTTP <status> (<category>)`, with category
  `auth failed; check the configured API key` (401, 403), `rate limited`
  (429), `model not found` (404), `request rejected` (400 and other 4xx),
  `server error` (5xx) or `bad response` (anything else), and
  `ollama_cloud: the provider reported an error mid-stream (server error)`.
  So a key the provider echoes in an error, whole, partial or in any encoding
  (base64, URL or HTML escaping), reaches no client response, log line,
  telemetry event or startup error. Error codes are unchanged (401/403/429/
  502-504 `unavailable`, 404 `not_found`, 400 `invalid_argument`, others
  `backend_error`).
- As a second layer, the remaining local text (transport and TLS errors)
  still goes through the key scrubber: any complete occurrence of the key,
  raw or JSON-escaped, becomes `[redacted]`. Telemetry carries error codes
  only.
- `redacted_config()` and `runtime_status().diagnostics` show the key source
  (`env` + variable name, or `file` + path), never the key.
- Tests cover these paths with an obvious fake key, including the full engine
  telemetry stream (`secrets: ...` cases in
  `src/backends/ollama_cloud/tests/ollama_cloud_tests.cpp`).

## Telemetry

The backend registers as `ollama_cloud` with `remote_process`, `streaming` and
`tool_calling`. Each request that is sent, and each local-only refusal, queues
one `backend.remote.request` event (drained by the engine's runtime sampler
onto the bus, with `backend` added). The other refusals (`response_format`,
grammar, images, `logit_bias`, `typical_p`, invalid messages) return before
any record is written and queue no event:

| Attribute | Value |
| --- | --- |
| `provider` | `ollama_cloud` |
| `locality` | `cloud` |
| `remote` | `true` |
| `host`, `endpoint`, `model` | e.g. `ollama.com`, `/api/chat`, `gpt-oss:120b` |
| `outcome` | `completed`, `failed`, `cancelled` or `refused` (local-only) |
| `http_status`, `error_code` | when known; `error_code` is the `ErrorCode` name only |
| `wall_ms`, `ttfb_ms`, `ttft_ms` | client-measured |
| `prompt_tokens`, `completion_tokens` | when the final line reported them |

The queue holds at most 256 events; overflow is dropped and counted
(`telemetry_dropped` in diagnostics), so the request path never blocks on
telemetry. `runtime_status()` also reports `gpu_memory.status = "unsupported"`
(no local GPU) and a `diagnostics` object with the redacted config, `provider`,
`locality`, `remote`, `tls_compiled_in` and the request, failure, cancellation
and local-only refusal counters.

## Tests

`sonder_ollama_cloud_tests` (CTest `sonder.ollama_cloud.*`) uses a scripted
in-process transport; nothing touches the network. Server-side coverage:
`serve_main: ollama_cloud config ...` and `serve_main: ollama_cloud base_url
off the Ollama Cloud origin ...` (config loader, endpoint opt-in and CLI),
`openai: correlation headers` (header parsing),
`chat: X-Sonder-Local-Only reaches the backend request` (HTTP to backend
plumbing), `chat: a local-only request against ollama_cloud makes no network
call, even with lazy models` and `chat: ollama_cloud error bodies reach
neither the client, the log nor telemetry`. The wire-byte cap is covered in
`http_client: max_response_bytes caps every received byte ...`
(`tests/test_http_client.cpp`).

## Limits and open items

- One serve process runs one backend, so cloud and local models are not mixed
  behind one `sonder-infer serve` yet; that needs multi-backend routing, which
  is a separate design.
- Only `sonder-infer serve` builds this backend. The other `sonder-infer`
  commands (`generate`, `chat`, `models`, ...) refuse `--backend ollama_cloud`
  with a pointer to `serve`, and the C ABI does not expose it.
- No live test against `ollama.com` exists; a TLS-enabled build and a real key
  are needed for that (opt-in, like the other live checks).
- Rate-limit (429) responses map to `unavailable` (HTTP 503 from serve) without
  retry; `Retry-After` from the cloud is not forwarded.
- Bounds on provider data: 4 MiB per NDJSON line, 65,536 JSON values per
  line, 16 MiB and 262,144 JSON values for `GET /api/tags`, the cumulative
  output cap (`max_stream_output_bytes`, 32 MiB) and the raw response cap
  (`max_response_bytes`, 64 MiB). For TLS the raw count is the decrypted HTTP
  byte stream (TLS record overhead is not counted).
- Known limits:
  - Name resolution is a blocking `getaddrinfo` before the connect deadline
    applies (`src/net/http_client.cpp`, `connect_to`); there is no bounded
    or cancellable resolver in `src/net`, so an unresponsive resolver can
    hold a request (or startup) past `connect_timeout_ms` and ignore
    cancellation until the OS resolver gives up.
  - The key lives in ordinary process memory for the backend's lifetime
    (one stored copy, plus short-lived header copies that are overwritten).
    It is not in locked or dump-excluded memory, and TLS libraries keep
    their own buffers of the encrypted request that are not wiped here, so a
    full memory dump of the process can contain it.
  - A response that stays alive with bytes that are not content (for
    example a newline every minute) is bounded by `request_timeout_ms` and
    `max_response_bytes`, not by `idle_timeout_ms`.
  - `probe()`, `list_models()` and `load_model()` take no cancellation token
    (the `Backend` interface has none), so `GET /api/tags` is bounded by its
    timeouts (at most 60 s total, plus `idle_timeout_ms`) rather than
    cancellable.
  - TLS end-of-stream without a close_notify alert is treated as a clean end
    by both TLS backends. Chunked and Content-Length bodies and NDJSON `done`
    lines still detect truncation; a general fix is deferred to its own change.
  - The key environment variable may be inherited by child processes of the
    server, and a key file larger than 4 KiB is truncated.
  - A 2xx response whose first body byte arrives after `idle_timeout_ms`
    (default 120 s, for example a long prefill) now fails, and a stalled
    non-2xx response reports a first-content timeout.
  - The cap of 65,536 JSON values per line could refuse a very large
    tool-call argument.
