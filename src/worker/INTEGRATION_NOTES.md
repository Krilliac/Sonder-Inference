# ARCH-01 integration

This module is registered before `src/server` in the root module list. It adds
`SONDER_HAS_WORKER`, the `worker` CLI command and the `remote-mock` serve backend.
The launcher resolves the current executable through the OS and passes its
absolute path; the host executable must implement the worker subcommand.

Changes outside this module are the module registration, CLI dispatch/help,
server backend factory/help and positive synthetic provenance for remote mock
work in engine/session telemetry. Session also keeps automatic suspension
hooks out of incapable remote backends; local hooks and explicit caller hooks
retain their existing path. The C ABI and native llamacpp backend are
unchanged. See `docs/design/ARCH-01-multiprocess.md` for the wire contract,
security boundary, operating limits and follow-up slices.

ARCH-01 is **fault isolation, not a security sandbox**. The worker runs with
the parent's identity. Code execution in the worker is not contained: it can
act as the user, kill the parent, and on Linux escape PR_SET_PDEATHSIG via fork.
What is guaranteed is that a crashing, hung or misbehaving worker cannot
crash, hang or exhaust the parent through the IPC channel: the parent bounds
stream output, counters and waits and rejects invalid responses. Trusted
parent callbacks remain the caller's responsibility.

Follow-up: real sandbox (AppContainer/LPAC on Windows; separate uid,
namespaces, no_new_privs, seccomp/Landlock, process-tree containment on Linux).

Integration details for review:

- `src/sessions/session.cpp` no longer attaches automatic `on_safe_point` and
  `suspended_state` hooks to **any** `Capability::remote_process` backend unless
  it advertises execution checkpoints/selective preemption, uses the nonblocking
  gate, or the caller supplies a hook. This affects every remote-process backend,
  not only RemoteBackend. No current remote backend uses these fields for
  suspension; RemoteBackend rejects explicit unsupported hooks. Existing
  local-backend hooks and explicit caller hooks retain their prior path.
- `include/sonder/inference/backends.hpp` adds public inline `is_mock_backend()`
  to positively classify both `mock` and `remote-mock` as synthetic.
- By design, `RestartPolicy::backoff()` never resets `attempts_`: delay reaches
  its configured cap and stays there for the supervisor's lifetime. Only the
  rolling-hour timestamp budget expires; neither healthy uptime nor budget
  expiry resets the backoff exponent.
- On Linux, `PR_SET_PDEATHSIG` follows the **forking thread's** lifetime. The
  persistent supervisor thread must outlive its worker. Moving spawn into a
  transient launch/request thread would kill the worker when that thread exits,
  even while `serve` remains alive.

## Before the native slice

Measure and address these costs before making performance claims:

- One watchdog/cancellation thread per request polling every 5 ms.
- `CreateEventW` per overlapped transfer, including repeated header/payload reads.
- Re-serialization for each canonical JSON check, and parent request validation.
- One ACK and synchronization round trip for every token.
- Idle reader wakeups: worker control reads, supervisor process polling and
  unsolicited-frame checks. There is no event-driven idle dispatcher yet.

Fault-injection tests use a separate `sonder_worker_adversary` executable.
`SONDER_WORKER_TESTING` is a PUBLIC library define when tests and CLI are built,
including the normal presets: it is **not** a test-target-only restriction on
`test_worker_mode`. In those builds the library can forward the fixture flag.
The production `worker` executable still rejects `--misbehave`, and the
serve/CLI backend factories never populate this programmatic option. Builds
without the define reject a nonempty option in RemoteBackend.

On Linux, every worker child holds one slot of a fixed process-wide pool of 64,
reserved before `fork` and released only after `waitpid` returns the PID or
`ECHILD`. When the pool is exhausted, launch fails with `unavailable` before any
child exists. Forced shutdown or restart polls WNOHANG for 250ms; an unreapable
child's slot transfers to one process-lifetime detached reaper without an
allocation or a join in backend teardown, and the reaper sweeps the 64 slots
every 100ms. A supervisor does not start a replacement while its own
predecessor is still unreaped (the backend reports `unavailable` meanwhile).
Bounded shutdown does not imply that a worker stuck in D state has already
released its GPU/OS resources. Windows has no such pool (see ARCH-01).
