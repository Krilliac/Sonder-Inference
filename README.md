# Sonder Inference

[![ci](https://github.com/Krilliac/Sonder-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/Krilliac/Sonder-Inference/actions/workflows/ci.yml)
[![python](https://github.com/Krilliac/Sonder-Inference/actions/workflows/python.yml/badge.svg)](https://github.com/Krilliac/Sonder-Inference/actions/workflows/python.yml)
[![hardening](https://github.com/Krilliac/Sonder-Inference/actions/workflows/hardening.yml/badge.svg)](https://github.com/Krilliac/Sonder-Inference/actions/workflows/hardening.yml)

Sonder Inference is the inference runtime of the Sonder ecosystem
([sondercore.si](https://sondercore.si)). It is a C++20 library with a stable
C ABI that owns inference policy (model and session lifecycle, scheduling, KV
accounting, sampling, cancellation, telemetry) and delegates token execution
to backends. It ships a command-line tool, `sonder-infer`, that includes a
local HTTP server, and ctypes-based Python bindings.

Version 0.1.0 (`CMakeLists.txt`). Design notes and plans are in
[docs/](docs/README.md); the general milestone status is in
[docs/ROADMAP.md](docs/ROADMAP.md), and the plan to consolidate local
inference onto the native in-process `llama.cpp` backend is in
[docs/ROADMAP_SI_ONLY.md](docs/ROADMAP_SI_ONLY.md).

## Capabilities

- **Core API**: `Engine`, `Device` and GPU inventory (CPU plus DXGI, NVML and
  sysfs GPU discovery, with ROCm ordinal mapping and UMA detection), `Model`,
  `Session`, `Backend` with capability advertisement, `SamplingConfig`
  validation and cooperative cancellation. Public C++ headers are in
  [`include/sonder/inference/`](include/sonder/inference/).
- **C ABI**: [`include/sonder_inference.h`](include/sonder_inference.h),
  `SONDER_ABI_VERSION` 1.
- **Backends** ([src/backends/README.md](src/backends/README.md)):
  - `llamacpp` (**target and only long-term local engine**): direct,
    in-process llama.cpp/GGML backend, built with `SONDER_WITH_LLAMA_CPP=ON`
    ([src/backends/llamacpp](src/backends/llamacpp/README.md),
    [docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md),
    [docs/integration/llamacpp.md](docs/integration/llamacpp.md)). Planned to
    move to a single vendored, pinned `ggml`/`llama.cpp` tree with SI's own
    patch series ([docs/ROADMAP_SI_ONLY.md](docs/ROADMAP_SI_ONLY.md)).
  - `ollama_cloud`: Ollama's hosted cloud models straight over HTTPS
    (`https://ollama.com`, bearer API key from an env var or token file), no
    local daemon; opt-in per model, refuses `X-Sonder-Local-Only` requests;
    needs a `SONDER_WITH_TLS=ON` build
    ([docs/integration/ollama-cloud.md](docs/integration/ollama-cloud.md)).
  - `ollama` (**deprecated, scheduled for removal**): HTTP adapter for an
    Ollama server, with streaming and cancellation
    ([src/backends/ollama](src/backends/ollama/README.md)).
  - `llamaserver` (**deprecated, scheduled for removal**): supervised external
    `llama-server`, or an existing compatible HTTP upstream; no llama.cpp
    linkage ([docs/integration/llama-server.md](docs/integration/llama-server.md)).
  - `mock`: deterministic synthetic output for tests; performs no inference.
  - `remote-mock`: the same synthetic engine in a supervised `sonder-infer worker`
    process over private local IPC ([ARCH-01](docs/design/ARCH-01-multiprocess.md));
    the first slice of native engine process isolation, without GPU requirements.

  `ollama` and `llamaserver` are deprecated because Sonder Inference must own
  scheduling, memory, and telemetry end to end; piggy-backed external servers
  obscure memory ownership and hide execution internals from the Sonder
  Observatory.
- **Scheduler** ([src/scheduler](src/scheduler/README.md)): priority queue,
  continuous batching, chunked prefill, admission and preemption policy.
- **KV cache manager** ([src/cache](src/cache/README.md)): logical block
  tables, fingerprinted prefix reuse and eviction policies. It does bookkeeping
  only and does not manage device memory.
- **Sampling** ([src/sampling](src/sampling/README.md)): sampler chain with
  logit bias, repetition/frequency/presence penalties, top-k, typical, top-p,
  min-p, temperature, greedy and seeded sampling, and stop sequences.
- **Telemetry**: JSONL events in the `sonder.observatory.event/1` envelope
  through a bounded, non-blocking queue ([docs/TELEMETRY.md](docs/TELEMETRY.md)).
- **HTTP server** (`sonder-infer serve`): HTTP/1.1 API under `/v1` with health,
  model listing, node/placement planning, backend identity,
  `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`,
  `/v1/messages`, `/v1/embeddings`, `/v1/rerank`, `/v1/score`,
  `/v1/tokenize`/`/v1/detokenize`, and live telemetry over SSE and NDJSON
  ([docs/SERVER.md](docs/SERVER.md)).
- **Benchmark harness**: `sonder-infer bench` and the standalone `sonder-bench`
  ([bench/README.md](bench/README.md)).
- **Optional TLS** for the internal HTTP client (outbound `https://`
  connections, including remote upstreams today and the upcoming Ollama Cloud
  provider), off by default ([docs/integration/tls.md](docs/integration/tls.md)).

## Requirements

- CMake 3.21 or newer and Ninja (the presets use the Ninja generator).
- A C++20 compiler. CI builds with MSVC on `windows-latest` and with the
  default compiler on `ubuntu-latest`; the hardening workflow uses Clang 18.
- Network access on the first configure with tests enabled: doctest (and, for
  the Ollama and llamaserver tests, cpp-httplib; the llamaserver tests reuse the
  copy the Ollama tests fetch) are downloaded at pinned revisions with SHA-256
  checks.
- Optional: OpenSSL 1.1.1 or newer for TLS on non-Windows platforms.
- Python bindings: Python 3.10 or newer.

CI covers Linux and Windows only.

## Build

Configure, build and test presets are defined in
[CMakePresets.json](CMakePresets.json). Output goes to `build/<preset>/`.

| Preset | Platform | Build type |
| --- | --- | --- |
| `msvc-debug`, `msvc-release` | Windows (MSVC) | Debug, RelWithDebInfo |
| `linux-debug`, `linux-release` | non-Windows (GCC/Clang) | Debug, RelWithDebInfo |
| `ci-windows`, `ci-linux` | as above | RelWithDebInfo, warnings as errors |

Opt-in direct `llamacpp` CPU and GPU presets (`windows-cuda-blackwell`,
`windows-cuda-rtx`, `windows-rocm-xt`, `windows-vulkan-xt`, `linux-llamacpp`,
`linux-cuda-blackwell`, `linux-cuda-rtx`, `linux-rocm-xt`, `linux-vulkan-xt`)
are also defined in [CMakePresets.json](CMakePresets.json); see
[docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md).

Linux (as in CI):

```bash
cmake --preset ci-linux
cmake --build --preset ci-linux
ctest --preset ci-linux
```

Windows, from a Visual Studio x64 developer environment (as in CI):

```powershell
cmake --preset ci-windows
cmake --build --preset ci-windows
ctest --preset ci-windows
```

Or let the script set up the developer environment:

```powershell
powershell -NoProfile -File scripts\build.ps1 -Preset msvc-debug -Test
```

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `SONDER_BUILD_TESTS` | ON when top-level | Build the test suites |
| `SONDER_BUILD_CLI` | ON when top-level | Build `sonder-infer` |
| `SONDER_WARNINGS_AS_ERRORS` | OFF | Warnings as errors (ON in `ci-*` presets) |
| `SONDER_WITH_LLAMA_CPP` | OFF | Fetch llama.cpp at pinned tag `v1.0.0` and build the `llamacpp` backend; accelerators through `GGML_*` options |
| `SONDER_WITH_TLS` | OFF | Enable `https://` in the internal HTTP client |
| `SONDER_TLS_BACKEND` | `auto` | `auto` (Schannel on Windows, OpenSSL elsewhere), `openssl` or `schannel` |
| `SONDER_BUILD_FUZZERS` | OFF | Build the libFuzzer targets in `fuzz/` (Clang) |

TLS builds as run in CI:

```bash
# Linux, OpenSSL (requires libssl-dev)
cmake --preset ci-linux -DSONDER_WITH_TLS=ON
```

```powershell
# Windows, Schannel. OpenSSL is used only by the loopback test server.
cmake --preset ci-windows -DSONDER_WITH_TLS=ON "-DOPENSSL_ROOT_DIR=C:/Program Files/OpenSSL"
```

llama.cpp backend on Windows as in CI (the CI job is non-blocking):

```powershell
cmake --preset ci-windows -DSONDER_WITH_LLAMA_CPP=ON -DGGML_NATIVE=OFF
```

Set `FETCHCONTENT_SOURCE_DIR_LLAMACPP` to a local llama.cpp checkout to build
offline. The optional modules and how they are wired are described in
[docs/MODULES.md](docs/MODULES.md).

GPU builds: see
[docs/SETUP.md](docs/SETUP.md#direct-llamacpp-backend-cpu-or-gpu).

Next: [docs/SETUP.md](docs/SETUP.md) covers installing and running the result.

## Usage

### `sonder-infer`

The full command reference (options, environment variables, exit codes) is in
[docs/CLI.md](docs/CLI.md).

```bash
B=build/ci-linux   # or build/<preset>

$B/sonder-infer help
$B/sonder-infer version --json
$B/sonder-infer devices
$B/sonder-infer backends --json

# Mock backend: no model required, output is synthetic
$B/sonder-infer generate --backend mock --model mock:tiny \
    --prompt "hello" --max-tokens 16 --telemetry events.jsonl
$B/sonder-infer chat --backend mock --model mock:tiny \
    --messages tests/fixtures/chat_messages.json

# Native in-process llama.cpp (target local engine; requires SONDER_WITH_LLAMA_CPP=ON)
$B/sonder-infer generate --backend llamacpp --model <path/to/file.gguf> \
    --prompt "Say hi"

# Local Ollama (deprecated, scheduled for removal; default http://127.0.0.1:11434,
# or SONDER_OLLAMA_URL / OLLAMA_HOST)
$B/sonder-infer models --backend ollama
$B/sonder-infer generate --backend ollama --model <pulled-model> --prompt "Say hi"

# Benchmark harness
$B/sonder-infer bench --backend mock --model mock:tiny \
    --corpus bench/corpus/smoke.json --out results.json --warmup 0 --runs 1
```

`SONDER_INFER_BACKEND` and `SONDER_INFER_MODEL` provide defaults for
`--backend` and `--model`. `chat` without `--messages` starts an interactive
session. Exit codes: 0 success, 1 runtime or backend failure, 2 usage error,
130 cancelled with Ctrl-C.

### HTTP server

```bash
$B/sonder-infer serve --backend mock --port 11437
curl -fsS http://127.0.0.1:11437/v1/sonder/health

# Native in-process llama.cpp (requires SONDER_WITH_LLAMA_CPP=ON)
$B/sonder-infer serve --backend llamacpp --model-dir <gguf-dir> --model <file.gguf> --port 11437
```

For local GGUF serving, the target path is `--backend llamacpp` (in-process,
built with `SONDER_WITH_LLAMA_CPP=ON`). The external `llamaserver` backend
(`--llamaserver-url` or `--llamaserver-config`) remains available during
migration but is deprecated and scheduled for removal; setup steps are in
[docs/SETUP.md](docs/SETUP.md).

Routes, options, limits and the Observatory connection are documented in
[docs/SERVER.md](docs/SERVER.md).

### Python

The `sonder-inference` package ([bindings/python](bindings/python/README.md))
wraps the C ABI with ctypes and needs no compiler at install time. Build the
shared library first:

```bash
cmake -S . -B build-shared -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DSONDER_BUILD_TESTS=OFF -DSONDER_BUILD_CLI=OFF
cmake --build build-shared --target sonder_inference
pip install ./bindings/python
export SONDER_INFERENCE_LIB_DIR=$PWD/build-shared
```

On Windows, add `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl` and run from a
developer environment.

```python
import sonder_inference as si

with si.Engine() as engine:
    engine.register_mock_backend()          # or engine.register_ollama_backend()
    with engine.load_model("mock", "mock:tiny") as model:
        with engine.create_session(model, si.SamplingConfig.greedy(max_tokens=32)) as session:
            result = session.generate("hello", on_token=lambda t: print(t, end=""))
            print(result.outcome, result.completion_tokens)
```

`Session.stream()` yields chunks from a worker thread. `Session.chat()` accepts
text conversations and streams through `on_token`; see the [chat contract](docs/integration/cabi-chat.md).
Older ABI v1 libraries without the additive chat export raise `UnsupportedError` for chat.
All four request methods accept keyword-only `RequestMetadata` for caller
parent lineage; see the [request parent contract](docs/integration/cabi-request-parent.md).

## Configuration

- CLI options and environment variables: [docs/CLI.md](docs/CLI.md).
- Server options: `sonder-infer serve --help` and [docs/SERVER.md](docs/SERVER.md).
- Sampling parameters: [docs/integration/sampling-config.md](docs/integration/sampling-config.md).
- Telemetry levels and event schema: [docs/TELEMETRY.md](docs/TELEMETRY.md),
  [docs/OBSERVATORY_CONTRACT.md](docs/OBSERVATORY_CONTRACT.md).

## Security

The following is enforced in code; details are in
[docs/SERVER.md](docs/SERVER.md#security) and
[docs/integration/tls.md](docs/integration/tls.md).

- `serve` listens on `127.0.0.1` by default. Binding a non-loopback address
  requires `--token-file`; requests must then carry
  `Authorization: Bearer <token>`, which is compared in constant time.
- The server does not implement TLS for its own listener. On a non-loopback
  bind it warns that the token and traffic are sent in cleartext; use a
  TLS-terminating proxy.
- On a loopback bind, requests whose `Host` header is not `127.0.0.1`,
  `localhost` or `[::1]` are rejected (DNS-rebinding defence).
- CORS uses an exact-match allowlist. The default origins reach only the
  read-only GET routes.
- Telemetry excludes generated text unless `--capture-text` is set, which
  requires a token.
- The Ollama client connects only to loopback hosts unless remote access is
  allowed (`--ollama-allow-remote`), and the CLI refuses plain `http://` to a
  non-loopback host.
- TLS is built in only with `-DSONDER_WITH_TLS=ON` (default OFF). With it,
  `serve` can reach a remote `https://` Ollama with `--ollama-allow-remote`,
  verified against the system trust store; `serve` has no CA or pin flags for
  the `ollama` backend. The `llamaserver` backend accepts a `tls` object in its
  JSON config (CA bundle, pins, server name, handshake timeout,
  `insecure_skip_verify`). The three
  verification modes (default, pin-only, insecure skip-verify) and the known
  limitations are defined in
  [docs/integration/tls.md](docs/integration/tls.md#behaviour).

## Project layout

| Path | Contents |
| --- | --- |
| `include/` | Public C++ API (`sonder/inference/`) and C ABI (`sonder_inference.h`) |
| `src/` | Library implementation and optional modules ([src/README.md](src/README.md)) |
| `tools/sonder-infer/` | CLI entry point |
| `bindings/python/` | Python ctypes bindings and pytest suite |
| `bench/` | Benchmark harness, corpora and reviewed result snapshots |
| `tests/` | Core doctest suite and CTest CLI checks ([tests/README.md](tests/README.md)) |
| `fuzz/` | libFuzzer targets and seed corpora ([fuzz/README.md](fuzz/README.md)) |
| `cmake/`, `scripts/` | CMake modules and `build.ps1` |
| `docs/` | Architecture, design decisions, contracts and integration notes |

## Testing

C++ suites (doctest, registered with CTest as `sonder.<suite>.*`) run with the
test preset matching the configure preset:

```bash
ctest --preset ci-linux
```

Tests use the mock backend and a fake Ollama server; they do not require model
weights or network services.

Python bindings, against a shared library built as above (as in
`.github/workflows/python.yml`):

```bash
python -m pip install "./bindings/python[test]"
cd bindings/python
SONDER_INFERENCE_LIB_DIR=../../build-shared python -m pytest -v
```

CI workflows:

- [`ci.yml`](.github/workflows/ci.yml): Linux and Windows build and tests, CLI
  smoke checks, TLS lanes (OpenSSL on Linux; Schannel on Windows) and an
  optional non-blocking llama.cpp build on Windows.
- [`python.yml`](.github/workflows/python.yml): shared library on Linux and
  Windows, pytest on Python 3.10 to 3.13.
- [`hardening.yml`](.github/workflows/hardening.yml): ASan+UBSan and TSan
  builds with Clang 18, and short libFuzzer runs
  ([docs/integration/hardening.md](docs/integration/hardening.md)).

## Related projects

- [Sonder Runtime](https://github.com/Krilliac/Sonder-runtime)
- [Sonder Runtime: connect to Sonder Inference](https://github.com/Krilliac/Sonder-runtime/blob/main/docs/runbooks/sonder-inference.md)
- [Sonder Observatory](https://github.com/Krilliac/Sonder-Observatory)
- Website: [sondercore.si](https://sondercore.si)

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Nate
Witkowski. Third-party components (llama.cpp/GGML for the optional backend;
doctest and cpp-httplib for tests only) are fetched at build time at pinned,
hash-checked revisions and keep their own MIT licenses; see [NOTICE](NOTICE)
and [docs/LICENSE_REVIEW.md](docs/LICENSE_REVIEW.md).
