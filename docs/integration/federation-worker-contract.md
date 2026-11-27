# Federation worker contract

Status: accepted (ADR-021 in [DESIGN_DECISIONS.md](../DESIGN_DECISIONS.md)).

Sonder Runtime places whole requests across N Sonder-Inference (SI)
endpoints. SI is a worker: it does not know about its peers, forward requests,
or retry for the router. This page lists what the router may rely on. Line
numbers are for orientation only; the cited symbols are the contract.

## Fields the router reads

- `GET /v1/sonder/health` ([SERVER.md](../SERVER.md)): `status` (`ready`,
  `starting`, `draining`; HTTP 200 only when ready), `instance_id`, `node_id`,
  `models[]` (`id`, `state`, `resident`, `in_flight`), `queued_by_class`
  (`interactive`, `subagent`, `background`), `residency`, `gpus[]`
  (`effective_free_vram_mib`) and `sonder.features`.
- `GET /v1/sonder/node` (schema `sonder.inference.node/1`): host memory, CPU,
  GPUs, backends, residency, profiles and feature capabilities.
- `GET /v1/sonder/identity[?model=ID]` (schema `sonder.inference.identity/1`):
  the nine-key `backend_identity`. A partial or synthetic identity must never
  satisfy identity-bound routing.

## Staleness bound

The router's bound is the Runtime's `DEFAULT_HEALTH_TTL_SECONDS = 5.0`
(`sonder_inference_gateway.py`, overridable by its health-TTL environment
variable). The 2 s `kGpuInventoryCacheTtl` in `server_endpoints.cpp` is only
the age limit of SI's own GPU inventory cache; it is not the router's bound
and a health body may legitimately carry GPU numbers up to 2 s older than the
poll. Backend `available` flags are refreshed by SI every 2 s. Treat a health
answer older than 5 s as unknown, not as healthy.

## "Nothing executed" status/code pairs

These responses mean no backend work started, so the router may send the same
request to another endpoint. Each pair has a `make_error` site in
`src/server/src/chat_handler.cpp` (the same sites repeat in the other
generation handlers):

| Status | Code | Site | Retry elsewhere |
| --- | --- | --- | --- |
| 503 | `overloaded` | thread-start refusal (`std::system_error` catch, about line 232); message says nothing was executed; `Retry-After: 1` | yes |
| 429 | `overloaded` | priority queue full (about line 265); `Retry-After: 1` | yes |
| 503 | `not_ready` | request cancelled while queued (about line 266) | yes |
| 503 | `not_ready` | draining or engine gone, before admission (`server.cpp`, about lines 497-512) | yes |
| 504 | `deadline_exceeded` | deadline expired while queued (about lines 263 and 476) | no, final |

`504 deadline_exceeded` also says nothing was executed, but the caller's
`X-Sonder-Deadline-Ms` budget is spent. It is final and must never be retried
on another endpoint.

Not safe to replay: `503 backend_unavailable` (the backend failed while
executing), any error after streaming started, and any proxy-hop 5xx other
than the exact `503` + `X-Sonder-Proxy-Upstream: unreachable` pair below
(including a bare 502 and a 504).

## Draining

`State::draining` (`server_policy.hpp`) is entered on SIGINT/SIGTERM or
`request_shutdown()`. Health then answers 503 `draining` and new generation
requests get 503 `not_ready` (checked again under the in-flight lock, so a
request racing the drain is also refused with nothing executed). In-flight
requests continue until `--shutdown-grace-ms`. The router must stop placing
new work on an endpoint as soon as health reports `draining`.

## Priority classes

`X-Sonder-Priority` accepts `interactive`, `subagent`, `background`, or the
legacy integer -16..16 (integer values map to class `interactive` with a
numeric priority). Anything else is 400 `invalid_correlation_header`. The
router should forward the header unchanged and may use `queued_by_class` to
choose an endpoint. `X-Sonder-Deadline-Ms` carries the remaining budget.

## Proxy dial-failure contract

When a reverse proxy (Caddy) fronts an SI endpoint, a failed dial means the
request never reached SI, so it is also "nothing executed". The contract
(this page is the source; POOL-06 implements it in the Caddy config) is
Caddy `handle_errors` on an upstream dial failure returning:

| Status | Header | Code | Retry elsewhere |
| --- | --- | --- | --- |
| 503 | `X-Sonder-Proxy-Upstream: unreachable` | `upstream_unreachable` | yes |

Only that exact pair (503 with the header and JSON code `upstream_unreachable`)
is retryable elsewhere. A bare 502 (it may follow an upstream reset after the
request was sent), a 504, and any other proxy 5xx are final: the request may
already have run, so the router must not replay it. SI never sends 502, so a
proxy 502 is not evidence that nothing executed.

## Out of scope

SI-to-SI remote workers and `DeviceKind::remote` capacity (reserved for the
RPC capacity tier, POOL-21) are deferred.

A cross-host layer split through llama.cpp RPC is not a remote worker: it is
an ordinary SI endpoint whose supervised llama-server child is the only
process that runs the RPC client ([rpc-worker.md](rpc-worker.md)).
