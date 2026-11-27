# Isolated llama.cpp RPC worker

Status: first slice (2026-11-26). Cross-host layer split through llama.cpp's
RPC client runs only in a supervised llama-server child that Sonder-Inference
(SI) owns, never inside the SI process and never in the child that serves the
primary model. This page is the contract for that worker. It follows ADR-021
and the [federation worker contract](federation-worker-contract.md): the
worker is one more SI endpoint that the Runtime places whole requests on.

## Why a separate process

llama.cpp's RPC client trusts its peers. At the pinned revision (b11195,
`ggml/src/ggml-rpc/ggml-rpc.cpp`) a peer that cannot be reached, fails the
hello handshake or reports a different protocol version ends the client
process with `GGML_ABORT`, and every tensor, size and graph result the peer
sends back is parsed inside that process. The protocol has no authentication
or encryption, and the activations of every prompt cross it. A broken or
hostile `rpc-server` must therefore only be able to take down a process whose
loss is already handled: a supervised child. It must not be able to kill or
corrupt the SI server or the process that hosts the main model.

## Layout

```
Runtime router ──HTTP──> SI serve A (primary model; no RPC anywhere)
               └─HTTP──> SI serve B ──loopback HTTP──> llama-server child
                         (rpc_worker)                    --rpc 10.0.0.2:50052
                                                          │ RPC (TCP, clear text)
                                                          └──> rpc-server on the peer host
```

- SI serve B runs the `llamaserver` backend in spawn mode with an
  `rpc_worker` block. It never loads weights itself and never links the RPC
  client.
- The llama-server child is the only process that speaks RPC. It serves only
  the split model; the primary model stays in serve A (or in its own
  llama-server, such as the live Bonsai server), which carries no RPC.
- One `serve` hosts one backend, so "never in the process that hosts the
  primary model" means a separate `serve` instance for the worker. The
  Runtime routes to it as a separate endpoint (ADR-021).

## Configuration

In the `--llamaserver-config` file of the worker instance:

```json
{
  "mode": "spawn",
  "executable": "C:/llama.cpp/build-rpc/bin/llama-server.exe",
  "args": ["-m", "D:/models/split.gguf", "-ngl", "99"],
  "rpc_worker": {
    "enabled": true,
    "servers": ["10.0.0.2:50052"]
  }
}
```

The executable must be a llama-server built with `GGML_RPC=ON` (llama.cpp
registers `--rpc` only when `llama_supports_rpc()` is true; a build without it
exits on the unknown flag, which the supervisor reports as a failed start).
The peer's `rpc-server` must speak the same RPC protocol major version as that
llama-server (7 at b11195).

| Key | Type, default | Rules |
|---|---|---|
| `rpc_worker.enabled` | bool, `false` | Spawn mode only. |
| `rpc_worker.servers` | string array, `[]` | 1 to 16 entries when enabled, no duplicates. Refused when set without `"enabled": true`. |
| `rpc_worker.allow_public_addresses` | bool, `false` | Accept peers outside the local ranges below. |

Each server is `A.B.C.D:PORT`: dotted-decimal IPv4 without leading zeros and a
decimal port in 1..65535. Refused:

- host names and IPv6: llama.cpp splits the endpoint at the first `:`, and a
  name can resolve to a different peer later, so the list is pinned to
  literals;
- `0.0.0.0/8`, multicast (`224/4`), reserved (`240/4`) and broadcast;
- anything with whitespace, a comma, other separators or non-ASCII
  characters, so one entry can never become two in the comma-separated
  `--rpc` value;
- without `allow_public_addresses`: any address outside loopback
  (`127/8`), RFC 1918 (`10/8`, `172.16/12`, `192.168/16`), link-local
  (`169.254/16`) and RFC 6598 shared space (`100.64/10`, used by overlay
  VPNs).

The supervisor puts `--rpc <servers joined by ','>` first on the child's
argv (llama.cpp registers RPC devices while parsing, so a later `--device`
naming one has to follow it), then the operator `args` unchanged, then its own
`--host 127.0.0.1 --port N`.

## Refusals: RPC never runs anywhere else

| Path | Result |
|---|---|
| `--rpc` or `--rpc=...` (any `_` spelling) in spawn `args` or `--llamaserver-arg`, with or without `rpc_worker` | refused at backend construction; use `rpc_worker` |
| `--rpc` in a launch profile's `extra_args` | refused (unchanged, [launch-profiles.md](launch-profiles.md)) |
| `LLAMA_ARG_RPC` as an `env` override | refused (all `LLAMA_*` overrides are); an inherited one is not passed to the child |
| `rpc_worker` on an attach backend | refused: SI must own the process the RPC client runs in |
| `SONDER_WITH_LLAMA_CPP=ON` together with `GGML_RPC=ON` | CMake configure fails before anything is fetched: the in-process backend never contains the RPC client |

An attach backend's `args` only describe an upstream that SI does not run, so
they are not filtered for `--rpc`; an upstream that uses RPC is the operator's
own process and outside this boundary.

## Supervision

The worker child uses the existing llamaserver supervisor
(`src/backends/llamaserver/supervisor.cpp`), unchanged:

- lazy start, readiness through `/health` within `startup_timeout_ms`;
- a child that exits (for example `GGML_ABORT` after a failed handshake) is
  relaunched with exponential backoff from `restart_backoff_ms` up to
  `max_restart_backoff_ms`, at most `max_restarts` consecutive times; a child
  that stayed ready for 60 s resets that budget. When the budget is spent
  the backend reports failure instead of respawning in a loop;
- the stall guard (`stall_guard`) catches a peer that keeps the child alive
  but stops it making progress;
- kill on parent exit: on Windows the child is in a Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; on POSIX it runs in its own process
  group with a parent-liveness pipe watcher. A crashed or killed SI never
  leaves a worker holding the peer connection.

## The IPC boundary

SI talks to the worker only over loopback HTTP on the supervisor-owned
`127.0.0.1` port, through the same request path as any spawned llama-server.
Everything the worker sends back is treated as untrusted: the HTTP client
bounds buffered bodies (64 MiB default), chunk headers and stream framing,
and responses are parsed by the llamaserver protocol code, which refuses
malformed JSON instead of trusting it. Nothing from the peer reaches SI
except through that child. Inputs that cross into the child are the
validated server literals, the operator's `args` (still filtered for
`--host`, `--port`, `--` and `--rpc`) and the environment allowlist in
`env_policy.hpp`.

For the router: a worker that dies while a request runs yields
`503 backend_unavailable` (executed; not replayable). While the worker is
restarting, a new request waits in `Supervisor::start` for the relaunch; a refused admission
follows the "nothing executed" pairs in the federation contract.

The backend's runtime status (`backends[].runtime` in health, see
[SERVER.md](../SERVER.md)) carries
`diagnostics.rpc_worker = {"isolated_child": true, "servers": [...]}` for a
worker backend, so an operator or router can tell a worker endpoint apart.

## Not in this slice

- No pre-spawn reachability probe of the peers: an unreachable peer costs a
  child start and an abort per attempt, bounded by `max_restarts`.
- After the restart budget is spent the backend stays failed until `serve`
  restarts; there is no slow periodic retry for a peer that comes back.
- No runtime check inside the in-process llamacpp backend for an RPC device
  registered by other means; the build-time refusal and the absence of
  dynamic backend loading are the guard.
- `sonder-infer tune` does not filter `--rpc` in its own spawned children
  (they are short-lived and already separate processes).
- `DeviceKind::remote` capacity and placement (POOL-21) are unchanged.

## Tests

`src/backends/llamaserver/tests/test_rpc_worker.cpp` (CTest
`sonder.llamaserver.*`), no network, GPU or llama-server needed: flag
spellings, the address table (local, public, never), injection attempts and
bounded messages, the block rules, argv order, every refusal path through the
backend and the config file, runtime status, and a fake-launcher run in which
the worker dies three times and is relaunched with backoff and the identical
pinned argv until the budget is spent.
