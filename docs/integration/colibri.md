# Attaching colibri `coli serve`

colibri's `coli serve` is an OpenAI-compatible server that streams very large
MoE models from disk at roughly 0.2-2 tokens per second. This page describes
how to attach it to Sonder-Inference through the `llamaserver` backend in
attach mode. It is a Sonder-side configuration guide only; Sonder does not run,
link or vendor colibri. Upstream behaviour quoted here comes from a read-only
review of colibri's server and has not been re-measured against a live
instance (see "Unverified").

## colibri side

- Bind to loopback: serve on `127.0.0.1` only. Do not expose the port; the
  Sonder attach rules also refuse a non-loopback `http://` upstream.
- Require a key: start `coli serve` with `--api-key`, taking the key from your
  own secret store or environment rather than typing it into a shared shell
  history. Sonder reads the same key from an environment variable (below).
- One slot: colibri runs a single KV slot (`kv_slots=1`). Start `sonder-infer serve` with
  `--backend-capacity 1` so Sonder admits one request at a time to this upstream.
- Queue limits on colibri (defaults): a request waits at most 300 s
  (`--queue-timeout`, `COLI_QUEUE_TIMEOUT`) and at most 8 requests queue
  (`--max-queue`, `COLI_MAX_QUEUE`). At about 1 tok/s a second request that
  arrives behind a long generation can hit the 300 s limit.
- Connection limit: 8 connections per client IP, health probes included.
- There is no `/props`. Set `context_length` explicitly in the Sonder config.

## What colibri sends on the wire

- Before it accepts a request (it may still be queued) it sends no bytes at
  all, not even response headers.
- After acceptance it sends a keepalive chunk every 10 s, then content or
  reasoning deltas as tokens arrive. The keepalive is an empty reasoning
  delta on chat, `data: {"choices":[{"delta":{"reasoning_content":""}}]}`, and
  `{"text": ""}` on `/v1/completions` (colibri `openai_server.py`
  `_keepalive_choice`, used around :9363-9377).
- Do not set `COLI_VISIBLE_KEEPALIVE=1` on the colibri side. It makes each
  keepalive `{"delta":{"reasoning_content":"."}}`, which is non-empty
  reasoning text: Sonder counts it as progress, so `first_content_timeout` and
  the stall watchdog stop working, and `separate_reasoning` callers receive
  "." reasoning chunks.
- A role-only first chunk and a final usage chunk are also sent.

This is why a single wall-clock timeout is the wrong tool: a request can be
healthy for hours while producing a token every few seconds.

## Sonder configuration

```json
{
  "mode": "attach",
  "base_url": "http://127.0.0.1:8080",
  "native_completion": false,
  "context_length": 32768,
  "api_key_env": "COLI_API_KEY",
  "keepalive_gap_ms": 10000,
  "idle_timeout_ms": 45000,
  "first_content_timeout_ms": 600000,
  "unlimited_request_timeout": true,
  "upstream_grammar": false,
  "slot_affinity": false,
  "stall_guard": {"generic_progress": true, "stall_seconds": 900, "policy": "cancel"}
}
```

`policy: "cancel"` is used because with `unlimited_request_timeout` and `warn`,
a stream that sends keepalives forever after its first content is unbounded.

`sonder-infer serve --llamaserver-config` loads these keys. They map onto the
same-named `LlamaServerBackendOptions` fields (`api_key_env`, `idle_timeout`,
`first_content_timeout`, `unlimited_request_timeout`, `keepalive_gap`,
`upstream_grammar`, `stall_guard.generic_progress`); the `_ms` keys are
milliseconds.

### Authentication

`api_key_env` names an environment variable. The token is read from the
process environment on every completion, discovery and /props request (native-attach monitor polling keeps the token read at construction) and sent as
`Authorization: Bearer <token>`. It is never read from the config file, never
stored in the options, never logged, and not part of any error message: errors
name the variable, not the value. `runtime_status()` diagnostics show
`"upstream_auth": {"header": "Authorization", "api_key_env": "COLI_API_KEY",
"value": "[redacted]"}`. The backend refuses to build when the variable is
unset, empty, or holds anything but a single visible-ASCII token (no spaces or
control characters). It is an attach-only option and is refused together with
`warmup` (the warm-up client does not carry the header). Header names and
values containing CR, LF or NUL are refused by the HTTP client before any
connection is made, as are headers the client owns (Host, Content-Length,
Content-Type, Connection, Transfer-Encoding, Expect, Upgrade).

### Timeouts

Four independent bounds apply to a streaming completion:

| Bound | Counts | Default | Fires when |
|---|---|---|---|
| `request_timeout` (total) | wall clock | 300 s | whole request exceeds it; removed by `unlimited_request_timeout` for streaming completions only (discovery and `/props` keep it) |
| `idle_timeout` | any received byte, after the response head | 0 (off) | no bytes at all for that long |
| `first_content_timeout` | content or reasoning text only | 0 (off) | no content/reasoning delta that long after the request was sent |
| `stall_guard.generic_progress` | content or reasoning text only | off | no content/reasoning delta for `stall_seconds`, measured from the last one (or from sending) |

Rules and why:

- `idle_timeout` must be strictly greater than `keepalive_gap` (default 10 s,
  colibri's keepalive interval). A bound at or below the gap would cut off a
  healthy stream whose only traffic for a while is the periodic empty
  keepalive. Use a comfortable multiple (30-60 s). `keepalive_gap` exists so
  tests and other upstreams can declare a different gap.
- `idle_timeout` is only armed once the response head has arrived. colibri
  sends nothing while a request is queued, so before the head only the total
  cap and `first_content_timeout` apply.
- Keepalive deltas reset the idle timeout (they are bytes) but never the
  first-content or stall bounds (they carry no content). Reasoning text counts
  as content even when the caller did not ask for `separate_reasoning`.
- `first_content_timeout` covers queue wait plus prefill. Set it above
  colibri's 300 s queue timeout plus your expected time to first token, or a
  request that is queued legitimately will be cancelled by Sonder first.
- `unlimited_request_timeout` requires `first_content_timeout` and
  `idle_timeout`: the first bounds a request that is never accepted, the
  second a stream that goes silent after it started.
- Config ranges (JSON, milliseconds): `idle_timeout_ms` and
  `first_content_timeout_ms` accept 0..86,400,000 (0 = off);
  `keepalive_gap_ms` accepts 1..3,600,000.
- All defaults are off, so existing configs keep today's single wall-clock
  behaviour.
- The idle, first-content and stall bounds end the request with
  `ErrorCode::protocol_error` and a message naming the bound (`idle timeout`, `first-content timeout`, `stall timeout`). Content
  already delivered to the caller stays delivered; no terminal frame is
  fabricated.

### Stall watchdog

`stall_guard.generic_progress` (attach, `native_completion: false` only) turns
`stall_seconds` into a per-request content-progress bound. `policy: "warn"`
records a `generic_stream_stalled` warning (visible in `runtime_status()`
warnings and a `stall` object) and keeps waiting, re-arming the timer;
`policy: "cancel"` ends the request with a `protocol_error`. `restart` is spawn-only
and is refused here. The watchdog runs inside the request's own read loop; it
adds no thread. Keep `stall_seconds` above colibri's queue wait. The watchdog
relies on colibri's keepalives being empty: with `COLI_VISIBLE_KEEPALIVE=1` on
the colibri side they count as progress and the watchdog never fires.
`generic_progress` together with `stall_guard.enabled: false` is refused as
contradictory.

### Request fields

- Generic mode sends thinking controls as top-level `enable_thinking` and
  `reasoning_effort` fields, and never sends `chat_template_kwargs`. It also
  sends no `cache_prompt` or `id_slot`.
- Declare `upstream_grammar: false`: configuring `grammar` then fails at build
  time instead of sending a field colibri will not honour.
- Send `temperature` explicitly. colibri's default is its own `COLI_TEMP`
  (falling back to 0.7), not what the Sonder caller may assume.
- `seed` is ignored by colibri; do not rely on it for reproducibility.
- A non-empty `logit_bias`, or a non-zero `frequency_penalty` or
  `presence_penalty`, is rejected with HTTP 400 (surfaced as
  `invalid_argument`). Leave those unset.
- Concurrency: with `kv_slots=1` keep Sonder at one active request to this
  upstream (`--backend-capacity 1`), and keep total
  connections, health probes included, under 8 per IP.

## Tests

`src/backends/llamaserver/tests/test_slow_upstream.cpp` drives a scripted
loopback upstream (no colibri, no model, no network service). It covers the
bearer token (401 without it, redaction, malformed variables), keepalive-only
streams versus the idle and first-content bounds, a mid-stream stall, the
progress watchdog in warn and cancel modes, header-injection refusal, and
generic thinking fields.

## Unverified

- colibri's wire behaviour above is from a read-only source review, not a live
  run against `coli serve`.
- Whether a real colibri accepts `enable_thinking` / `reasoning_effort` at top
  level, and which of them it honours, is unverified.
- The `/health`, `/props` and `/v1/models` behaviour of a real colibri while
  its single slot is busy is unverified; `load_model` needs `/v1/models` to
  answer within the connect and request timeouts.
