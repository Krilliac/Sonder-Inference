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
| `base_url` | Optional, default `https://ollama.com`. Must be `https://`; an optional port and path prefix are kept (`/api/...` is appended, so do not end it with `/api`). No userinfo or query. |
| `api_key_env` | NAME of an environment variable holding the key (`OLLAMA_API_KEY` is Ollama's own convention). |
| `api_key_file` | Path of a file holding only the key; surrounding whitespace is trimmed. Restrict its permissions yourself; the provider does not check ACLs. |
| `models` | Required, non-empty: the cloud allowlist (see Policy). |
| `connect_timeout_ms` | 1..600000, default 10000. |
| `request_timeout_ms` | Whole-request cap, 1..86400000, default 900000. |
| `idle_timeout_ms` | Longest silence between body bytes once the response head arrived, 0..86400000 (0 = off), default 120000. |
| `ca_bundle_path` | Optional extra PEM trust anchors. Verification stays on. |

Exactly one of `api_key_env` and `api_key_file` is required. Unknown fields
are rejected. An inline `api_key` (or `key`, `token`) field is refused, so the
key never sits in a config file next to the rest of the settings and never
travels on a command line. There are no TLS options other than
`ca_bundle_path`: `insecure_skip_verify`, pins and `server_name` are unknown
fields here.

C++ hosts build the backend with `make_ollama_cloud_backend(OllamaCloudOptions)`
(`sonder/inference/backends/ollama_cloud.hpp`); the options mirror the JSON and
add a pluggable `transport` for tests.

## Policy

- **Opt-in.** A model is a cloud model only when it is listed in `models`.
  `load_model()` refuses anything else with `invalid_argument` ("is not
  configured as a cloud model"), even when the cloud offers it; a listed model
  the cloud does not offer fails with `not_found`. `list_models()` returns the
  allowlisted models present in `GET /api/tags`, with `resident = false`.
- **Local-only requests are refused.** `GenerateRequest::local_only` and
  `ChatRequest::local_only` (set from `RequestOptions::local_only`, which
  `sonder-infer serve` sets from the `X-Sonder-Local-Only: 1|true` request
  header) make the provider fail the request with `invalid_argument` before any
  network I/O. Over HTTP that is 400 `invalid_request`: "the backend refused the
  request: ollama_cloud: model 'X' runs on Ollama Cloud (remote) and the request
  is local-only (X-Sonder-Local-Only)". Local backends ignore the flag.
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
- It is sent only as the `Authorization` header to the configured host. The
  provider's copies (the stored key, the header value) are overwritten when
  they are released. Copies inside the HTTP client's internal request buffer are
  not covered by that.
- Every error that leaves the provider (HTTP error bodies, mid-stream
  `{"error": ...}` lines, transport errors) is scrubbed: any occurrence of the
  key becomes `[redacted]`. Telemetry carries error codes only, never upstream
  messages.
- `redacted_config()` and `runtime_status().diagnostics` show the key source
  (`env` + variable name, or `file` + path), never the key.
- Tests cover these paths with an obvious fake key, including the full engine
  telemetry stream (`secrets: ...` cases in
  `src/backends/ollama_cloud/tests/ollama_cloud_tests.cpp`).

## Telemetry

The backend registers as `ollama_cloud` with `remote_process`, `streaming` and
`tool_calling`. Each request queues one `backend.remote.request` event (drained
by the engine's runtime sampler onto the bus, with `backend` added):

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
`serve_main: ollama_cloud config ...` (config loader and CLI),
`openai: correlation headers` (header parsing) and
`chat: X-Sonder-Local-Only reaches the backend request` (HTTP to backend
plumbing).

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
