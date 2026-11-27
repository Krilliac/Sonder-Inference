# Isolated llama.cpp RPC worker

Status: first slice (2026-11-26). Cross-host layer split through llama.cpp's
RPC client runs only in a supervised llama-server child that Sonder-Inference
(SI) owns, never inside the SI process and never in the child that serves the
primary model, except through the paths listed as not closed under
"Refusals". The child is a fault boundary, not a privilege boundary (see
"Why a separate process"). This page is the contract for that worker. It follows ADR-021
and the [federation worker contract](federation-worker-contract.md): the
worker is one more SI endpoint that the Runtime places whole requests on.

## Why a separate process

llama.cpp's RPC client trusts its peers. At the pinned revision (b11195,
`ggml/src/ggml-rpc/ggml-rpc.cpp`) a peer that cannot be reached, fails the
hello handshake or reports a different protocol version ends the client
process with `GGML_ABORT`, and every tensor, size and graph result the peer
sends back is parsed inside that process. The protocol has no authentication
or encryption, and the activations of every prompt cross it.

What the separate process delivers in this slice, and what it does not:

- **Delivered: fault isolation.** A peer that makes the client abort or
  crash takes down only the supervised worker child, whose loss is already
  handled (restart with backoff, a bounded budget, `503` for the request in
  flight). The SI server and the process that hosts the primary model keep
  running.
- **Not delivered: a privilege boundary.** The worker child runs under the
  same user token as SI, with no restricted token, lowered integrity level,
  AppContainer or (on POSIX) separate user or sandbox. On Windows the only
  Job Object limit is `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
  (`src/backends/llamaserver/process.cpp`). A peer that gets code execution
  in the worker through the client's response parsing therefore runs as that
  user and can kill or tamper with the SI server, the Bonsai llama-server and
  anything else that user owns. Treat every `rpc-server` peer as trusted at
  the level of the account SI runs under until a sandboxed worker lands (see
  "Not in this slice").
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

## Refusals: where RPC is kept out

| Path | Result |
|---|---|
| `--rpc` or `--rpc=...` (any `_` spelling) in spawn `args` or `--llamaserver-arg`, with or without `rpc_worker` | refused at backend construction; use `rpc_worker` |
| `--models-preset` (any `_` spelling) in spawn `args` or `--llamaserver-arg` | refused at backend construction: a preset INI sets options by key, and an `rpc` key reaches llama-server's `--rpc` handler (`common/preset.cpp` at b11195) |
| `--rpc` or `--models-preset` in a launch profile's `extra_args` | refused (unchanged, [launch-profiles.md](launch-profiles.md)) |
| `rpc_worker` in `--llamaserver-config` together with `--profile` or `--all-profiles` | refused by `serve`: a profile's child serves its model, and in shared spawn mode several profiles share one child |
| llama.cpp user `config.ini` (`%APPDATA%\llama.cpp\config.ini`; `$XDG_CONFIG_HOME` or `~/.config/llama.cpp/config.ini` on POSIX), which llama-server applies before env and argv | closed for every SI-spawned child: `APPDATA` is not inherited on Windows (unset, llama.cpp skips the file) and on POSIX the child gets `XDG_CONFIG_HOME=/dev/null`, under which no file can exist (`env_policy.hpp`, `process_options.cpp`) |
| llama.cpp system `config.ini` on Windows (`%PROGRAMDATA%\llama.cpp\config.ini`) | closed: `PROGRAMDATA` is never inherited and cannot be set as an override |
| `LLAMA_ARG_RPC` / `LLAMA_ARG_MODELS_PRESET` as an `env` override | refused (all `LLAMA_*` overrides are); an inherited one is not passed to the child |
| `rpc_worker` on an attach backend | refused: SI must own the process the RPC client runs in |
| `SONDER_WITH_LLAMA_CPP=ON` together with `GGML_RPC=ON` | CMake configure fails before anything is fetched; `GGML_RPC` is then forced `OFF` in the cache (a fork whose ggml defaults it `ON` cannot turn it back on), and configure also fails if the fetched tree still defines a `ggml-rpc` target |

Paths that are **not** closed in this slice (each can still put the RPC client
into an SI-spawned child, the primary model's included):

- `-hf` / `--hf-repo` naming a Hugging Face repo that has a `preset.ini` at
  its root: llama-server downloads that file and starts in router mode with
  it (`tools/server/server.cpp`, `common/download.cpp` at b11195), and its
  `rpc` key is applied like a local preset's. Operator spawn args accept
  downloads on purpose (launch profiles refuse `-hf`), so refusing it is a
  separate decision.
- POSIX `/etc/llama.cpp/config.ini`: llama.cpp reads it regardless of the
  environment. It is root-owned; whoever can write it controls every
  llama.cpp program on the host.
- `sonder-infer tune` children are not filtered for `--rpc` or
  `--models-preset` (they get the same environment policy, so the
  `config.ini` row above holds for them).

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
`--host`, `--port`, `--`, `--rpc` and `--models-preset`) and the
environment allowlist in `env_policy.hpp`.

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
- No privilege separation for the worker (see "Why a separate process"): a
  restricted token or AppContainer on Windows, or a separate user or sandbox
  on POSIX, is still to do.
- The `-hf` preset-repo path and POSIX `/etc/llama.cpp/config.ini` listed
  under "Refusals" stay open.

## Tests

`src/backends/llamaserver/tests/test_rpc_worker.cpp` (CTest
`sonder.llamaserver.*`), no network, GPU or llama-server needed: flag
spellings, the address table (local, public, never), injection attempts and
bounded messages, the block rules, argv order, every refusal path through the
backend and the config file, runtime status, and a fake-launcher run in which
the worker dies three times and is relaunched with backoff and the identical
pinned argv until the budget is spent.
`test_containment.cpp` plants a llama.cpp `config.ini` where the parent's
`APPDATA` / `XDG_CONFIG_HOME` / `HOME` point and checks that the child block
resolves to no file; `test_serve_main.cpp` covers the `--profile` refusal.
The CMake `GGML_RPC` refusal has no automated test.
