# Bounded mock-server stability smoke

After the CMake/CTest checks, run this stdlib-only POSIX harness against the
compiled CLI. It starts only the synthetic mock backend on an ephemeral
loopback port; it needs no model weights or external service.

```sh
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
python scripts/stress_mock.py --binary build/linux-debug/sonder-infer \
    --cycles 3 --out build/qualification/mock-stress.json
```

Each cycle starts a fresh server, checks ready-file discovery and synthetic
health, then sends 64 chat requests with eight clients: 32 streaming and 32
nonstreaming. Every response must contain text and exactly eight completion
tokens; every stream must terminate normally and contain usage. Four more
streaming requests must each produce a first token before SIGINT is sent.
Those admitted requests must finish within the two-second drain grace, the
server must exit 0 and its ready file must disappear. Startup, client reads,
drain waits and process cleanup are bounded. `--cycles` accepts 1–10.

The receipt records request count, errors, client latency, Linux process
peak RSS when available, shutdown duration and cleanup results. Keep
generated receipts under the ignored `build/` directory. A failed assertion
or request exits nonzero. Repeat this bounded command when testing a new
build; it is a transport/lifecycle measurement, never a model quality or
inference-throughput claim.

The initial Linux debug qualification completed 204 requests over three
cycles with zero errors. With an intentional 10 ms per mock token, client
p95 was 91.9–93.4 ms; draining four active streams took 163.8–172.1 ms.
Each process exited 0 and removed its ready file. These are observations
from one environment, not performance limits.
