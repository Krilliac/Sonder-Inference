# Set up Sonder Inference

This is the one guide for building `sonder-infer`, serving a model over HTTP,
and putting that server behind a reverse proxy such as Caddy. It was checked
against the code at commit `0fcfafc` on 2026-11-20. Items marked "run" were
executed on 2026-11-20 against a prebuilt Windows `sonder-infer` from an
earlier commit (without TLS or llama.cpp), not a fresh build of `0fcfafc`.
Everything else is traced to the code, and anything not run on a real host
is marked.

Other documents own the details this guide links to instead of repeating:
[README](../README.md#build) (build presets and options),
[CLI](CLI.md) (every command), [SERVER](SERVER.md) (HTTP routes, limits,
errors, shutdown), [llama-server backend](integration/llama-server.md),
[launch profiles](integration/launch-profiles.md) and
[TLS](integration/tls.md).

## Choose a backend

`sonder-infer serve` needs one `--backend` (or the environment variable
`SONDER_INFER_BACKEND`). Without it, `serve` stops with exit code 2 and
`error: --backend is required (mock, ollama, llamacpp or llamaserver; or set
SONDER_INFER_BACKEND)` (`src/server/src/serve_main.cpp:654-657`, run
2026-11-20).

| Backend | Needs | Where inference runs | Pick it when |
| --- | --- | --- | --- |
| `mock` | nothing; in every build | nowhere: synthetic output | Checking the install, a proxy, tokens or a client. Never for quality or speed. |
| `ollama` | an Ollama server; in the normal build | in Ollama, local by default | You already run Ollama on this machine. A remote Ollama needs a TLS build and has hard limits (see [Remote Ollama](#remote-ollama-from-sonder-inference)). |
| `llamacpp` | a build with `-DSONDER_WITH_LLAMA_CPP=ON`; GGUF files | inside the `sonder-infer` process | One process with no second server, CPU or GPU, and you accept building llama.cpp. |
| `llamaserver` | an external `llama-server`, or any OpenAI-style server | in that server (attach) or a child process Sonder starts (spawn) | You want llama.cpp features without compiling it into `sonder-infer`, launch profiles, or an upstream that is remote or not llama.cpp. |

What a given binary contains:

```powershell
sonder-infer backends --json
```

The JSON lists each backend with `"available"` true or false. A backend that
is compiled in but cannot reach its server (Ollama stopped, nothing on port
8080) shows `available: false` with the reason. A backend that was not
compiled in is absent from the list, and choosing it fails with
`this build does not include the llamacpp backend (configure with
SONDER_WITH_LLAMA_CPP=ON)` (`src/server/src/backend_setup.cpp:~431-436`,
`tools/sonder-infer/main.cpp:264-277`). Run on 2026-11-20 against a prebuilt
binary without llama.cpp: it listed `llamaserver`, `mock` and `ollama`.

## Build

Prerequisites (2026-11-20):

- CMake 3.21 or newer (`CMakePresets.json:2`) and Ninja
  (`CMakePresets.json:7`).
- A C++20 compiler (`CMakeLists.txt:~23-25`). On Windows the presets force
  MSVC `cl` (`CMakePresets.json:18-22`), so build from an x64 Visual Studio
  developer environment, or let `scripts/build.ps1` set one up. On Linux,
  GCC or Clang.
- Network access for the first configure: the test framework and, for
  `llamacpp`, llama.cpp are downloaded at pinned versions with SHA-256
  checks (`cmake/SonderDoctest.cmake:6-12`,
  `src/backends/llamacpp/CMakeLists.txt:~26-31`). With tests on (the
  default), the Ollama and llamaserver tests also fetch cpp-httplib, a
  header used only to host their fake servers
  (`src/backends/ollama/tests/CMakeLists.txt:13-22`,
  `src/backends/llamaserver/tests/CMakeLists.txt:7-11`). There is no vcpkg
  and there are no submodules. For an offline llama.cpp build see the
  [README Build section](../README.md#build); that does not cover the
  cpp-httplib download, so an offline build with tests on needs
  `-DSONDER_BUILD_TESTS=OFF`.

The presets (`msvc-debug`, `msvc-release`, `linux-debug`, `linux-release`,
`ci-windows`, `ci-linux`), the build options and the CI recipes are in the
[README Build section](../README.md#build). For a server you actually run,
use a release preset (`msvc-release` or `linux-release`, both
RelWithDebInfo).

Windows, default options:

```powershell
powershell -NoProfile -File scripts\build.ps1 -Preset msvc-release
```

Linux:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

The binary is `build/<preset>/sonder-infer` (`sonder-infer.exe` on Windows).
There is no install rule (no `install()` in the CMake files): copy the one
binary where you want it. Check it:

```bash
build/linux-release/sonder-infer version --json
```

Expected: one JSON object with `name`, `version`, `backends` and
`modules.server: true`. If `modules.server` is false, the build lacks
`src/server` and `serve` is unavailable.

`scripts/build.ps1` takes only `-Preset`, `-Test` and `-Clean`
(`scripts/build.ps1:15-19`); it cannot pass extra `-D` options. To use the
options below on Windows, open an x64 Native Tools / Developer PowerShell
for Visual Studio and run `cmake` yourself, or use the CMake line from the
README (`cmake --preset ci-windows -D...`). Use `scripts/build.ps1 -Clean`
or delete `build/<preset>` when you change an option that the cache keeps.

### Direct llama.cpp backend (CPU or GPU)

```powershell
cmake --preset msvc-release -DSONDER_WITH_LLAMA_CPP=ON -DGGML_NATIVE=OFF
cmake --build --preset msvc-release
```

`SONDER_WITH_LLAMA_CPP` is OFF by default (`CMakeLists.txt:36`). With it
ON, llama.cpp is fetched at tag `v1.0.0` and built as a library only
(`src/backends/llamacpp/CMakeLists.txt:~26-31,62-67`). The CPU backend is always
included. Accelerators are turned on with llama.cpp's own cache variables,
passed on the same command line (`src/backends/llamacpp/CMakeLists.txt:~34-36`):

| Goal | Extra options | Needs installed first |
| --- | --- | --- |
| NVIDIA GPU | `-DGGML_CUDA=ON` | CUDA Toolkit |
| Any Vulkan GPU | `-DGGML_VULKAN=ON` | Vulkan SDK |

`-DGGML_NATIVE=OFF` builds for a generic CPU instead of the build machine's
exact one, so the binary runs on other computers; CI uses it
(`.github/workflows/ci.yml:158`). Leave it ON only for a binary that never
leaves this machine. The SDK installation, supported compilers and
architecture overrides are llama.cpp's to document:
[llama.cpp build guide at the pinned tag](https://github.com/ggml-org/llama.cpp/blob/v1.0.0/docs/build.md)
(CUDA and Vulkan sections). Those GPU recipes were not built for this guide.

How you know the GPU build worked: serve with `--device gpu:0` (below). A
build with no GPU backend refuses with `this llama.cpp build has no GPU
backend` (`src/backends/llamacpp/llamacpp_core_backend.cpp:261`).

### TLS build (needed only for `https://` upstreams)

```bash
cmake --preset linux-release -DSONDER_WITH_TLS=ON
```

`SONDER_WITH_TLS` is OFF by default. It adds `https://` to the HTTP client
that talks to Ollama and `llamaserver` upstreams. It does not add TLS to the
`sonder-infer serve` listener, which is always plain HTTP (see
[Behind Caddy](#behind-caddy)).

`SONDER_TLS_BACKEND` (which TLS library is used), the OpenSSL requirements
and the TLS test dependencies are owned by the
[README Build section](../README.md#build) and `cmake/SonderTls.cmake`. The
protocol limits (Schannel version, no client certificates) are owned by
[TLS](integration/tls.md) "Known limitations".

## Serve quickstarts

`sonder-infer serve --help` prints every flag; the option defaults and the
request limits are owned by [SERVER.md](SERVER.md) (Options, Limits). Below
you only need the defaults host `127.0.0.1` and port `11437`
(`src/server/src/serve_main.cpp:57-138`; `0` picks a free port). Replace
`sonder-infer` with the path to your binary.

Every start prints a banner on standard error. When it says `ready on
http://127.0.0.1:11437` the models are loaded.

### mock (verify the install)

```powershell
sonder-infer serve --backend mock
```

Expected on standard error (run 2026-11-20): `listening on
http://127.0.0.1:11437`, then `ready on ...`, `auth: none (loopback only)`
and `warning: MOCK BACKEND - synthetic output`. In a second terminal:

```powershell
curl.exe http://127.0.0.1:11437/v1/sonder/health
```

Expected: HTTP 200 and JSON with `"status":"ready"` (checked with a token
too, below). Stop the server with Ctrl+C: it prints `shutting down` and
`stopped`, and exits 0.

### ollama (local)

Start Ollama first, then:

```powershell
sonder-infer serve --backend ollama --model <ollama-model-name>
```

`--model` is required for this backend (`error: --model is required for
backend ollama`, run 2026-11-20). It may be repeated; the first model is the
default. The Ollama address is `--ollama-url`, else `SONDER_OLLAMA_URL`
(used as written), else `OLLAMA_HOST` (normalised: scheme `http`, port 11434
or 443 for `https`, and `0.0.0.0`, `[::]` or empty become `127.0.0.1`), else
`http://127.0.0.1:11434`
(`src/server/src/backend_setup.cpp:71-119`,
`src/backends/ollama/include/sonder/inference/backends/ollama.hpp:54`).
`SONDER_INFER_BACKEND` and `SONDER_INFER_MODEL` can stand in for `--backend`
and `--model` (`backend_setup.cpp:60-69`). Flags win over the environment.
List what Ollama has:

```powershell
sonder-infer models --backend ollama
```

### llamacpp (in-process, GGUF files)

Needs the build from [Direct llama.cpp backend](#direct-llamacpp-backend-cpu-or-gpu).
`--model` is the GGUF file name inside a `--model-dir` (or a full path to a
file) (`llamacpp_core_backend.cpp:225-239, 297-303`):

```powershell
sonder-infer serve --backend llamacpp --model-dir <model-dir> `
  --model <file>.gguf --device gpu:0 --gpu-layers -1 --context-length 8192
```

- `--device`, three cases (`llamacpp_core_backend.cpp:250-264`): empty or an
  id starting with `cpu` means CPU (no layers offloaded); an id starting with
  `gpu` (for example `gpu:0`) offloads to the GPU; any other id (for example
  `cuda:0` or `Vulkan0`) is refused with `unsupported device id for
  llamacpp: <id>`. A `gpu` id on a build with no GPU backend is refused with
  `this llama.cpp build has no GPU backend`.
- `--gpu-layers`: layers to offload; `-1` (the default) means all
  (`serve_main.cpp:703-711`, `llamacpp.hpp:27`). Only used with a `gpu:*`
  device.
- `--context-length`: `0` uses the model's training context. If the flag is
  omitted the backend uses 4096 (`llamacpp.hpp:23`, `backend_setup.cpp:40`).
  Maximum 4194304.
- `--model-dir` can repeat. Models load before the server reports ready
  (`--lazy-models` defers each load to its first request).
- `--gpu-layers`, `--context-length`, `--batch-size`, `--ubatch-size`,
  `--cache-type-k`, `--cache-type-v`, `--flash-attn`, `--moe-experts` and
  `--tensor-override` apply to this backend only; any other backend refuses
  them (`serve_main.cpp:744-751`). Their values and ranges are in
  `sonder-infer serve --help` and [llamacpp notes](integration/llamacpp.md).

Failure symptoms: `file_not_found` style errors mean the name does not
match a file in any `--model-dir`; a `unsupported` error about the GPU
backend means the binary was built without CUDA/Vulkan.

### llamaserver (external server)

Attach to a server that is already running. With no options it attaches to
`http://127.0.0.1:8080` (`serve_main.cpp:115`):

```powershell
sonder-infer serve --backend llamaserver --model <alias> `
  --llamaserver-url http://127.0.0.1:8080
```

Spawn: Sonder starts `llama-server` itself and stops it on exit. The
supervisor always adds `--host 127.0.0.1` and a free `--port`; passing your
own `--host` or `--port` in the arguments is refused
(`serve_main.cpp:691-697`). Each `--llamaserver-arg` is one argument,
passed as written:

```powershell
sonder-infer serve --backend llamaserver --model <alias> `
  --llamaserver-mode spawn --llamaserver-executable <path-to>/llama-server `
  --llamaserver-arg --model --llamaserver-arg <model-dir>/<file>.gguf `
  --llamaserver-arg --alias --llamaserver-arg <alias> `
  --llamaserver-arg --ctx-size --llamaserver-arg 8192
```

The same settings as a JSON file, which is easier to keep and the only way
to set timeouts, TLS and the other keys:

```json
{
  "mode": "spawn",
  "executable": "<path-to>/llama-server",
  "args": ["--model", "<model-dir>/<file>.gguf", "--alias", "<alias>", "--ctx-size", "8192"],
  "startup_timeout_ms": 60000
}
```

```powershell
sonder-infer serve --backend llamaserver --model <alias> --llamaserver-config llamaserver.json
```

Spawn mode without an executable fails with `--llamaserver-executable is
required in spawn mode`. An unknown JSON key fails with `llamaserver config:
unknown field '<key>'` (`backend_setup.cpp:205-217`). Every key (timeouts,
restarts, TLS, `api_key_env`, spill guard, host-memory admission for
mixture-of-experts models) is documented once in
[integration/llama-server.md](integration/llama-server.md).
`--warmup-file` requires `--backend llamaserver`.

### Launch profiles (typed settings plus a VRAM check)

A launch profile is one named, validated description of how to serve a
model (backend, GGUF, context, KV cache types, offload, sampling). The file
format and every field are in
[integration/launch-profiles.md](integration/launch-profiles.md). To serve
one:

```powershell
sonder-infer serve --profiles profiles.json --profile <name> `
  --llamaserver-executable <path-to>/llama-server
```

- The profile chooses the backend and the model, served under the
  profile's `name`. `--model` may only repeat that name; a different
  `--backend` is refused. `--profile` without `--profiles FILE` fails with
  `--profile needs --profiles FILE` (`serve_main.cpp:604-631`).
- A `llamaserver` profile uses spawn mode (`serve_main.cpp:672-685`).
- `--vram-budget-mib N` and `--allow-overcommit` apply to profiles only.
  The budget is `--vram-budget-mib` if given, else the profile's
  `vram_budget_mib`, else the detected device VRAM minus 1536 MiB
  (`serve_main.cpp:124-126`). When the estimate does not fit, serve refuses
  to start with `... does not fit; lower ctx_size or n_gpu_layers, use a
  smaller KV cache type, raise the budget, or pass --allow-overcommit`. When
  no VRAM figure can be detected and no budget is set, it refuses with
  `... cannot detect the device's VRAM: set vram_budget_mib in the profile or
  --vram-budget-mib, or pass --allow-overcommit` (`serve_main.cpp:553-562,
  768-772`, `launch_profile.hpp:112`).

## Tokens

Without a token, anyone who can reach the port can use the server, so the
default bind is loopback only. A bind to any other address requires a
token: `--host 0.0.0.0 is not a loopback address; a non-loopback bind
requires --token-file` (`server.cpp:1785-1789`, run 2026-11-20). `--capture-text`
also requires a token (`server.cpp:1790`).

Create a token file with a random value. Pick one line:

```bash
# Linux, macOS, Git Bash (needs openssl; output is 64 hex characters)
(umask 077 && openssl rand -hex 32 > sonder.token)
```

```powershell
# Windows PowerShell 5.1 and PowerShell 7
$b = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
(($b | ForEach-Object { $_.ToString('x2') }) -join '') | Set-Content -NoNewline -Path sonder.token
```

Both were run on 2026-11-20 and produced a 64-character hex token. The file
must hold exactly one non-empty token without spaces or line breaks inside;
leading and trailing whitespace, including a final newline, is trimmed
(`serve_main.cpp:784-794`). Otherwise serve exits 2 with `--token-file must
contain one non-empty token` (or `cannot read --token-file <path>`).

Restrict who can read the file:

- POSIX: `chmod 600 sonder.token`. Serve warns `--token-file ... is readable
  by group or others; restrict it with chmod 600` when it is not
  (`serve_main.cpp:795-801`).
- Windows: serve does not check permissions. Remove inherited access and
  grant only yourself: `icacls sonder.token /inheritance:r /grant:r
  "$($env:USERNAME):(R)"` (run in PowerShell 7 on 2026-11-20; the service
  account must be the one you grant when you run it as a service).

Start with it:

```powershell
sonder-infer serve --backend mock --token-file sonder.token
```

Clients send the token as `Authorization: Bearer <token>`. `POST
/v1/messages` also accepts `x-api-key: <token>`; no other route does
(`server.cpp:1220-1240`; run 2026-11-20: `x-api-key` on `GET /v1/models`
gave 401). With a token set, every route needs it, including
`/v1/sonder/health`, so a readiness probe must send it too (health without
the header returned 401 on 2026-11-20). Only the CORS preflight (`OPTIONS`)
is exempt. A wrong token is `401 unauthorized`
(`WWW-Authenticate: Bearer`).

```powershell
$t = (Get-Content sonder.token -Raw).Trim()
curl.exe -H "Authorization: Bearer $t" http://127.0.0.1:11437/v1/sonder/health
```

To rotate the token, write a new file and restart `serve`; the token is read
once at startup. On Windows the `icacls` step above leaves you read-only
access, so delete the old file first (or grant yourself `(M)` instead of `(R)`).

## Run as a service

`sonder-infer serve` is an ordinary foreground process: it does not
daemonise, write a pid file or log to a file. Logs go to standard error
(`--log-format text|json`). Let a supervisor start it, restart it and
capture standard error.

Behaviour a supervisor depends on (`serve_main.cpp:49-55, 1017-1023`,
[SERVER.md](SERVER.md) "Startup and shutdown"):

- SIGINT or SIGTERM starts a graceful stop: health returns 503, new chat
  requests get 503 `not_ready`, in-flight requests get `--shutdown-grace-ms`
  (default 5000, maximum 600000) and are then cancelled, and the process
  exits 0. A second signal exits at once.
- Exit codes: 0 clean stop, 1 runtime error (for example the port is in use,
  where the message adds `pick another --port, or --port 0 for an ephemeral
  port`), 2 usage or configuration error. Do not auto-restart on 2: the same
  command will fail the same way.
- Set the supervisor's stop timeout above `--shutdown-grace-ms` plus the time
  the backend needs to unload (a loaded model or a spawned `llama-server`
  takes extra seconds), or the supervisor kills the process mid-drain.
- Readiness. `--ready-file PATH` is written atomically once the socket
  listens, as `{"url","pid","instance_id","api_version"}`, and removed when
  `serve` returns. The models may still be loading: poll
  `GET /v1/sonder/health` until it returns 200 (it returns 503 while
  `starting` or `draining`). A process that is killed (second signal,
  SIGKILL, a service wrapper's hard kill) leaves the file behind; check that
  its `pid` is alive before trusting it (`serve_main.cpp:379-399`,
  [SERVER.md](SERVER.md) `--ready-file`). Model loads can take minutes, so
  start-up checks should poll health with a long deadline.

### Linux (systemd)

Not verified on a real host as of 2026-11-20. The directives are standard
systemd; the `ExecStart` flags are the ones verified above. Save as
`/etc/systemd/system/sonder-infer.service`, adjusting the paths and user:

```ini
[Unit]
Description=Sonder Inference
After=network.target

[Service]
Type=simple
User=sonder
RuntimeDirectory=sonder-infer
ExecStart=/opt/sonder/sonder-infer serve --backend llamaserver --model <alias> \
  --llamaserver-config /etc/sonder/llamaserver.json \
  --host 127.0.0.1 --port 11437 \
  --token-file /etc/sonder/sonder.token \
  --shutdown-grace-ms 10000 \
  --ready-file /run/sonder-infer/ready.json
Restart=on-failure
RestartSec=2
RestartPreventExitStatus=2
TimeoutStopSec=60
# The token file must be readable by User= and by nobody else (mode 600).
# Spawn mode: the default KillMode=control-group sends SIGTERM to the
# spawned llama-server child at the same moment as to sonder-infer, which
# defeats the drain. mixed signals only the main process first.
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sonder-infer
journalctl -u sonder-infer -f          # the banner and access log
curl -H "Authorization: Bearer $(sudo cat /etc/sonder/sonder.token)" http://127.0.0.1:11437/v1/sonder/health
```

Roll back with `sudo systemctl disable --now sonder-infer` and deleting the
unit file.

### Windows

Not verified on a real host as of 2026-11-20. On Windows the process
installs handlers through the C runtime's `std::signal` for SIGINT and
SIGTERM only (`serve_main.cpp:49-55, 351-358`, `SignalGuard`). Only Ctrl+C
(`CTRL_C_EVENT`, which the C runtime turns into SIGINT) starts a graceful
drain. Ctrl+Break is delivered as SIGBREAK, a separate signal with no
handler here, so it ends the process at once with no drain, and so do
closing the console window and `TerminateProcess`; each leaves the ready
file behind. How a service wrapper (NSSM, WinSW, `sc.exe` with a wrapper
process) stops it is therefore **unverified**: a wrapper that terminates
the process outright skips the drain. Two options that need no extra
software:

- A Task Scheduler task that runs the command line above at start-up, with
  `--token-file` pointing at a file only that account can read. Ending the
  task terminates the process without a drain, so accept a hard stop (or
  stop the server by sending Ctrl+C to its console).
- A service wrapper of your choice. Test its stop behaviour: start the
  service, send a long streamed request, stop the service, and check whether
  `stopped` appears in the captured standard error.

## Behind Caddy

`sonder-infer serve` speaks plain HTTP/1.1 and closes every connection
(`Connection: close`); it has no TLS server (`docs/SERVER.md`, "Security").
To reach it from another machine over HTTPS, keep it on loopback and let a
TLS-terminating reverse proxy such as [Caddy](https://caddyserver.com) face
the network.

Design (the options not used are in "Alternative" below):

1. Sonder binds `127.0.0.1:11437` and uses `--token-file`.
2. Caddy terminates TLS and forwards to `127.0.0.1:11437`.
3. Clients send the same bearer token through Caddy; Caddy passes
   `Authorization` (and `x-api-key`) through unchanged.

Why each Caddy line exists:

- `header_up Host {upstream_hostport}`: on a loopback bind the `Host` header
  must be `127.0.0.1`, `localhost` or `[::1]`, with any port, or serve
  answers `403 forbidden_host` (`server.cpp:1180-1187`; the 403 was
  reproduced on 2026-11-20 with `-H "Host: example.com"`). Caddy by default
  forwards the client's `Host` (your public name), so it must be rewritten.
  `{upstream_hostport}` is a Caddy placeholder
  ([reverse_proxy docs](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#defaults)).
- `flush_interval -1`: chat streams (`text/event-stream`) and the telemetry
  streams (`text/event-stream` and `application/x-ndjson`) must not be
  buffered. Telemetry streams send a keepalive every 15 seconds
  (`server.hpp:~100`). Caddy's reverse_proxy documents `-1` as disabling
  response buffering.
- `request_body { max_size ... }`: Caddy rejects larger bodies itself; the
  exact status depends on the Caddy version and was not run here. Sonder
  checks `Content-Length` before it reads the body ([SERVER.md](SERVER.md)
  Limits), so keep Caddy's limit at or above Sonder's `--max-body-bytes`
  (default in SERVER.md Options) if you want Sonder's own `413
  payload_too_large` to be the one that applies; set both together if you
  change either.
- Keep `--token-file`. Caddy then is not the only gate, and a process on the
  same machine cannot use Sonder without the token.

Sample Caddyfile. Checked with Caddy 2.9.1 on 2026-11-20: `caddy adapt` and
`caddy validate` both succeeded, with the sample as written and with `tls
internal` added, and the adapted JSON showed `flush_interval` -1, the `Host`
header set to the upstream host and port, and a `max_size` of 8000000. Real
traffic through it was not tested, so run `caddy validate` yourself and then
test a request. The directive names were also checked against the Caddy
documentation
([reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy),
[request_body](https://caddyserver.com/docs/caddyfile/directives/request_body)).

```caddyfile
sonder.example.com {
	request_body {
		max_size 8MB
	}
	reverse_proxy 127.0.0.1:11437 {
		header_up Host {upstream_hostport}
		flush_interval -1
	}
}
```

Steps:

1. Create a DNS name that points at the proxy host (`sonder.example.com` is
   a placeholder), and open port 443 (and 80, which Caddy uses to get a
   certificate) in the host firewall. For a LAN or lab name that no public
   CA can validate, use `tls internal` in the site block, which makes Caddy
   issue from its own local CA; clients must then trust that CA (see
   `caddy trust` in Caddy's docs).
2. Start Sonder on loopback with a token:
   `sonder-infer serve --backend <backend> ... --token-file sonder.token`
   (the default host and port already are `127.0.0.1:11437`).
3. Validate and run Caddy: `caddy validate --config Caddyfile`, then
   `caddy run --config Caddyfile` (or install it as Caddy's own service).
4. Verify from another machine:

   ```bash
   curl -H "Authorization: Bearer $(cat sonder.token)" https://sonder.example.com/v1/sonder/health
   ```

   Expected: 200 and `"status":"ready"`. Without the header: 401. Test a
   stream with a chat request and check that tokens arrive one by one rather
   than all at the end.

Failure symptoms:

| Symptom | Cause and fix |
| --- | --- |
| `403 forbidden_host` through Caddy, works directly | `header_up Host` is missing or wrong. |
| Stream arrives all at once at the end | `flush_interval -1` missing, or another proxy in front is buffering. |
| `411 length_required` on POST | The client sent a chunked body, which Sonder does not accept (`docs/SERVER.md`, Limits). Make the client send `Content-Length`; Caddy's `request_buffers` option on `reverse_proxy` buffers the body so a `Content-Length` is set upstream (per Caddy docs; untested here). |
| A body-size error from Caddy instead of Sonder's `413` | Caddy's `max_size` is lower than `--max-body-bytes`. |
| `401 unauthorized` | The token header is missing or not the file's token; health checks need it too. |
| Browser `403 forbidden_origin` | Add the page's exact origin with `--cors-origin`. CORS is decided by Sonder, not Caddy: a proxy does not change it. |
| `502 Bad Gateway` from Caddy | Sonder is not running, still starting, or on another port. |

### Alternative: bind a non-loopback address instead

Use `--host <address> --token-file sonder.token` (for example `0.0.0.0`). The
loopback `Host` check then does not apply, so no `header_up Host` is needed
(`server.cpp:1180`: only `if (loopback)`). Serve warns `listening on a
non-loopback address without TLS: the bearer token and all traffic travel in
cleartext; put a TLS-terminating proxy in front` (`serve_main.cpp:862-866`).
Use this only when Caddy runs on another machine, bind to that one interface
rather than all interfaces, and firewall the port to the proxy host. Do not
expose it directly; the token would cross the network unencrypted.

## Remote Ollama from Sonder Inference

You may want `sonder-infer serve --backend ollama` on one machine and Ollama
on another, with Caddy providing TLS in front of Ollama. The built-in
`ollama` backend has hard limits (checked in code, 2026-11-20):

- A non-loopback Ollama host needs both `--ollama-allow-remote` and an
  `https://` URL. Plain `http://` to a non-loopback host is refused before
  the server starts: `--ollama-allow-remote refuses plain http:// to a
  non-loopback host ('<url>'); use https:// (needs a SONDER_WITH_TLS=ON
  build)` (`tools/sonder-infer/main.cpp:290-295, 986-1015`; reproduced on
  2026-11-20).
- `https://` needs a [TLS build](#tls-build-needed-only-for-https-upstreams).
  Otherwise: `only http:// URLs are supported (got https://); https://
  needs a build with SONDER_WITH_TLS=ON` (`src/net/http_client.cpp:~58-66`).
- Server certificates are checked against the operating system's trust store
  only. `serve` passes the Ollama backend just the URL and the allow-remote
  flag (`backend_setup.cpp:~427-430`): there is no flag for a private CA
  bundle, a certificate pin or a server name for `--backend ollama`
  (OpenSSL uses the default verify paths, `src/net/tls_openssl.cpp:292`). A
  certificate from a public CA works. A Caddy `tls internal` certificate
  works only if its root CA is installed in the trust store of the machine
  running `sonder-infer`; that was not tested. The certificate must match
  the host name in the URL.
- No `Authorization` header is ever sent to Ollama
  (`src/backends/ollama/`: no header handling). A Caddy `basic_auth` or
  token check in front of Ollama therefore blocks Sonder. Restrict access by
  network instead, for example with Caddy's `remote_ip` matcher or a
  firewall rule that allows only the Sonder host.
- A path prefix in the URL is kept
  (`https://host/ollama` calls `https://host/ollama/api/...`;
  `src/backends/ollama/ollama_client.cpp:84-96`), so you can serve Ollama
  under a prefix with Caddy's `handle_path` or `uri strip_prefix`.
- TLS protocol limits (Schannel version, no client certificates) are in
  [TLS](integration/tls.md) "Known limitations".

Working shape. Checked with Caddy 2.9.1 on 2026-11-20 (`caddy adapt` and
`caddy validate` with `tls internal` added; the `remote_ip` route comes
before the `403`), but not run end to end. `remote_ip` matches the direct
connection's address; if another proxy sits in front of Caddy it sees that
proxy instead:

```caddyfile
ollama.example.com {
	@sonder remote_ip 192.0.2.10
	handle @sonder {
		reverse_proxy 127.0.0.1:11434 {
			header_up Host {upstream_hostport}
			flush_interval -1
		}
	}
	respond 403
}
```

```powershell
sonder-infer serve --backend ollama --model <model> `
  --ollama-url https://ollama.example.com --ollama-allow-remote
```

`--ollama-url https://ollama.example.com` with no port means port 443
(`http_client.cpp:~58-66` sets 443 for https). Also,
`OLLAMA_HOST=https://ollama.example.com` is accepted and normalised to
`https://ollama.example.com:443`.

### Alternative: attach through `llamaserver`

`--backend llamaserver` in attach mode can reach Ollama's OpenAI-compatible
API (`/v1/models`, `/v1/chat/completions`) and has the TLS and key settings
the `ollama` backend lacks. Marked **verify**: the keys below are checked in
code, but the combination with Ollama was not run for this guide.

```json
{
  "mode": "attach",
  "base_url": "https://ollama.example.com",
  "native_completion": false,
  "allow_remote": true,
  "api_key_env": "OLLAMA_PROXY_KEY",
  "tls": {
    "ca_bundle_path": "<path-to>/caddy-root.pem",
    "server_name": "ollama.example.com"
  }
}
```

```powershell
$env:OLLAMA_PROXY_KEY = "<key your proxy expects>"
sonder-infer serve --backend llamaserver --model <ollama-model> --llamaserver-config ollama-openai.json
```

- `base_url` is the server root, without `/v1`
  ([llama-server.md](integration/llama-server.md)). `native_completion:
  false` sends raw generation to `/v1/completions` (the doc's setting for
  generic OpenAI-style servers).
- A non-loopback upstream requires both `allow_remote: true` and an
  `https://` URL: `llamaserver: remote upstream requires allow_remote and
  HTTPS` (`llamaserver_backend.cpp:219-220`), so it also needs a TLS build.
- `api_key_env` names an environment variable holding a key; each request
  then carries `Authorization: Bearer <key>`. Attach mode only, and not
  together with `warmup` (`llamaserver_backend.cpp:180-183, 492-498`). Use
  it with a proxy that checks a bearer token; Ollama itself ignores it.
- The `tls` object accepts `ca_bundle_path`, `pinned_sha256`,
  `pinned_cert_path`, `server_name`, `insecure_skip_verify` (default false;
  do not use it) and `handshake_timeout_ms` (default 10000)
  (`backend_setup.cpp:286-298`, `llamaserver.hpp:21-28`). `ca_bundle_path` is
  how a Caddy internal root can be trusted for this backend.

## Connect the Sonder runtime

The Sonder runtime (the Python application) uses a running `sonder-infer
serve` as a provider. Its settings, environment variables and checks are
owned by the runtime's runbook; follow that rather than this guide:
<https://github.com/Krilliac/Sonder-runtime/blob/main/docs/runbooks/sonder-inference.md>.
From this side you only need the base URL (`http://127.0.0.1:11437` or your
Caddy address), the token file if you set one, and a backend that reports
`"status":"ready"` at `/v1/sonder/health`.

## Troubleshooting

Startup failures print `error: <message> (see 'sonder-infer serve --help')`
on standard error and exit 2 (usage) or 1 (runtime). Messages marked "run"
were reproduced on 2026-11-20; the others are quoted from the code.

| Message or response | Cause | Fix |
| --- | --- | --- |
| `--backend is required (mock, ollama, llamacpp or llamaserver; or set SONDER_INFER_BACKEND)` (run) | No backend given | Pass `--backend` or set the variable. |
| `--model is required for backend ollama` (run) | Backend needs a model | Add `--model` or `SONDER_INFER_MODEL`. |
| `this build does not include the llamacpp backend (configure with SONDER_WITH_LLAMA_CPP=ON)` | Binary lacks llama.cpp | Rebuild with the option, or use `llamaserver`. |
| `--host <addr> is not a loopback address; a non-loopback bind requires --token-file` (run) | Non-loopback bind without a token | Add `--token-file`. |
| `--capture-text exports generated text and requires --token-file` | Text capture without a token | Add a token. |
| `--token-file must contain one non-empty token` / `cannot read --token-file <path>` | Empty file, spaces inside, or wrong path | Recreate the file; one token, no inner whitespace. |
| `warning: --token-file ... is readable by group or others; restrict it with chmod 600` | POSIX file mode | `chmod 600`. |
| `--ollama-allow-remote refuses plain http:// to a non-loopback host` (run) | Remote Ollama over http | Use `https://` with a TLS build. |
| `only http:// URLs are supported (got https://); https:// needs a build with SONDER_WITH_TLS=ON` | `https://` upstream, build without TLS | Rebuild with `-DSONDER_WITH_TLS=ON`. |
| `ollama: refusing non-loopback host '<host>' (set allow_remote to override)` | Remote host without the flag | Add `--ollama-allow-remote` (https only). |
| `llamaserver: remote upstream requires allow_remote and HTTPS` | Remote attach without both | Set `allow_remote: true` and an `https://` `base_url`. |
| `--llamaserver-executable is required in spawn mode` | Spawn without executable | Add `--llamaserver-executable` or `executable` in the JSON. |
| `llamaserver spawn args must not override --host or --port` | `--host`/`--port` in the spawn arguments | Remove them; Sonder supplies its own. |
| `llamaserver config: unknown field '<key>'` | Misspelt JSON key | Check [llama-server.md](integration/llama-server.md). |
| `option --<flag> applies only to --backend llamacpp ...` | A llama.cpp-only flag with another backend | Remove it, or put llama-server's own flag in the JSON `args`. |
| `--warmup-file requires --backend llamaserver` | Wrong backend | Use `llamaserver`. |
| `--profile needs --profiles FILE` / `--profiles needs --profile NAME` | One half of the pair missing | Pass both. |
| `... does not fit; lower ctx_size or n_gpu_layers ...` | Profile VRAM estimate over budget | Shrink the profile, raise `--vram-budget-mib`, or `--allow-overcommit`. |
| `this llama.cpp build has no GPU backend` | `--device gpu:0` on a CPU-only build | Rebuild with `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON`. |
| `unsupported device id for llamacpp: <id>` | `--device` is not empty, `cpu...` or `gpu...` (for example `cuda:0`) | Use `gpu:0`, or `cpu`. |
| `... in use; pick another --port, or --port 0 for an ephemeral port` (exit 1) | Port taken | Another `serve` is running (check the ready file's `pid`), or choose a port. |
| `cannot write ready file <path>.tmp` or `cannot move ready file into place: <path>` (exit 1) | Directory missing or not writable | Create the directory first. |
| HTTP 401 `unauthorized` | Missing or wrong token (health too) | Send `Authorization: Bearer <token>`. |
| HTTP 403 `forbidden_host` | `Host` is not loopback on a loopback bind | Behind a proxy set `header_up Host {upstream_hostport}`. |
| HTTP 403 `forbidden_origin` | Browser origin not allowed | `--cors-origin <exact origin>`. |
| HTTP 411 / 413 / 408 / 431 | Chunked body / body too large / slow client / header over 16 KiB | See the Limits table in [SERVER.md](SERVER.md). |
| HTTP 503 `not_ready` | Models still loading, or draining | Poll `/v1/sonder/health` until 200. |
| HTTP 503 `overloaded`, 429 | Connection or queue limit | `--max-connections`; see SERVER.md. |

For anything not listed, run `sonder-infer serve --help` and
`sonder-infer backends --json`, and read the startup banner and access log
on standard error.
