# Bounded mock-server stability smoke

After the CMake/CTest checks, run this stdlib-only POSIX harness against the
compiled CLI. It starts only the synthetic mock backend on an ephemeral
loopback port; it needs no model weights or external service.

```sh
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
python scripts/stress_mock.py --binary build/linux-debug/sonder-infer \
    --cycles 3 --mock-delay-ms 40 --out build/qualification/mock-stress.json
```

Each cycle starts a fresh server, checks ready-file discovery and synthetic
health, then sends 64 chat requests with eight clients: 32 streaming and 32
nonstreaming. Every response must contain text and exactly eight completion
tokens; every stream must terminate normally and contain usage. Four more
streaming requests must each produce a first token before SIGINT is sent.
They use 16 completion tokens so their drain tail spans about 600 ms at the
explicit 40 ms mock delay. Immediately before sending SIGINT, health must
report four admitted model requests and all four futures must remain active.
These are sequential samples before the signal, rather than an atomic
guarantee at delivery. The ready health response must have HTTP status 200. The server uses a
two-second drain grace; the clients must finish normally, the server must
exit 0 and its ready file must disappear, with five-second wait caps.
Startup, client reads,
drain waits and process cleanup are bounded. `--cycles` accepts 1–10;
`--mock-delay-ms` accepts 1–50 (default 10), with CI using 40 to leave a
longer active-stream drain window. The mock fixture emits at most 16 tokens.

The receipt records the server build commit and instance identity, request
count, errors, client latency, Linux process
peak RSS when available, shutdown duration and cleanup results. Keep
generated receipts under the ignored `build/` directory. Receipts include
`status`, requested/completed cycle counts, and the verified request count
from completed cycles. On qualification failure they retain those cycles plus
the failure type and a bounded message; `errors` counts qualification
failures, and requests in an incomplete cycle are not claimed as verified.
An atomic `running` checkpoint is written before startup and after each
completed cycle; an interrupted step can retain this explicitly incomplete
receipt. Final `passed`/`failed` receipts replace it atomically.
A failed assertion or request still exits nonzero. Repeat this bounded
command when testing a new build; it is a transport/lifecycle measurement,
never a model quality or inference-throughput claim.

The existing `ubuntu-latest` CI build/test context runs three cycles against
its newly built `ci-linux` binary with a 40 ms mock delay after CTest and CLI
telemetry checks. The step has a three-minute limit, uses `pipefail` while
teeing console output. After an attempted Linux stress step, an `always()`
uploader attempts to retain its JSON receipt and log as
`mock-lifecycle-stress-ci-linux` for 14 days, including ordinary failures
and step timeouts. Runner loss can prevent artifact publication. The former idle health/SIGINT probe is
covered by this gate; context names and Windows
checks are unchanged. No client latency or RSS threshold is a gate.
Two stdlib regression controls also run in this step: early process exit
must leave a failed receipt, and a failed second start must preserve the
first real 68-request cycle. Run them locally with
`SONDER_STRESS_BINARY=build/linux-debug/sonder-infer SONDER_STRESS_DELAY_MS=40 python3 -m unittest scripts.test_stress_mock`.

The initial Linux debug qualification (10 ms delay) completed 204 requests
over three cycles with zero errors. With an intentional 10 ms per mock
token, client p95 was 91.9–93.4 ms; draining four active streams took
163.8–172.1 ms.
Each process exited 0 and removed its ready file. These are observations
from one environment, not performance limits.
