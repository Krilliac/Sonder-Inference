# Integration notes: `feat/tls`

Optional TLS (`https://`) for the internal HTTP client in `src/net`, which is
the transport the Ollama client uses, so the runtime can reach a remote Ollama
node over TLS (for example Node1 at `https://10.77.0.2:8443`).

- CMake option `SONDER_WITH_TLS`, **OFF by default**, defined in
  `cmake/SonderTls.cmake`. When OFF nothing changes: no extra sources, no
  definitions, no OpenSSL, and `https://` still returns
  `ErrorCode::unsupported`.
- Backends (`SONDER_TLS_BACKEND=auto|openssl|schannel`, default `auto`):
  - **Schannel** on Windows (OS stack: `secur32`, `crypt32`, `bcrypt`; no new
    dependency). TLS 1.2 via `SCHANNEL_CRED`.
  - **OpenSSL** (1.1.1 or newer, TLS 1.2+) everywhere else, found with
    `find_package(OpenSSL)` and linked as a system library. Also selectable on
    Windows with `-DSONDER_TLS_BACKEND=openssl`.
- No code is vendored or fetched. cpp-httplib is not used by the library.

## What the lead needs to wire

1. **Root `CMakeLists.txt`**: add one line after the optional-module
   `foreach` loop and before `add_subdirectory(tests)`:

   ```cmake
   include(SonderTls)  # optional TLS for src/net (feat/tls); OFF by default
   ```

   It has to come after `add_library(sonder_inference)` and after
   `include(SonderDoctest)`, because it adds sources and definitions to
   `sonder_inference` and registers `tests/tls` (CTest `sonder.tls.*`). The
   option and cache variable are declared in the file itself. Nothing else in
   root changes. Until this line is added, the branch builds exactly like
   main does, because the new files are not compiled.

2. **CI**: the existing jobs stay as they are and need no OpenSSL, since TLS
   is OFF. To test TLS ON, add these jobs:

   ```yaml
   tls-linux:
     name: ubuntu-latest (TLS, OpenSSL)
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v4
       - name: Install Ninja + OpenSSL headers
         run: sudo apt-get update && sudo apt-get install -y ninja-build libssl-dev
       - name: Configure
         run: cmake --preset ci-linux -DSONDER_WITH_TLS=ON
       - name: Build
         run: cmake --build --preset ci-linux
       - name: Test
         run: ctest --preset ci-linux

   tls-windows:
     name: windows-latest (TLS, Schannel)
     runs-on: windows-latest
     continue-on-error: true   # make blocking once it has been green on MSVC
     steps:
       - uses: actions/checkout@v4
       - uses: ilammy/msvc-dev-cmd@v1
         with:
           arch: x64
       - name: Ensure Ninja
         shell: pwsh
         run: if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { choco install ninja -y --no-progress }
       - name: OpenSSL for the in-test TLS server only
         shell: pwsh
         run: choco install openssl -y --no-progress
       - name: Configure
         shell: pwsh
         run: cmake --preset ci-windows -DSONDER_WITH_TLS=ON "-DOPENSSL_ROOT_DIR=C:/Program Files/OpenSSL"
       - name: Build
         run: cmake --build --preset ci-windows
       - name: Test
         shell: pwsh
         run: |
           $env:PATH = "C:\Program Files\OpenSSL\bin;$env:PATH"
           ctest --preset ci-windows
   ```

   The Windows client uses Schannel, but the loopback test server and the
   test-time certificate generation in `tests/tls` use OpenSSL. That is why
   the Windows job installs OpenSSL even though the library does not link it.
   The Schannel path was checked locally with a MinGW-w64 cross build run
   under Wine (all tests pass), but it has not yet been built with MSVC.
   That is why the job is `continue-on-error` at first. If you prefer presets,
   add `ci-linux-tls` / `ci-windows-tls` presets that inherit the CI presets
   with `SONDER_WITH_TLS=ON`.

3. **`docs/LICENSE_REVIEW.md`** (AGENTS.md: no upstream library linked
   before its license is recorded). Proposed entry: *OpenSSL 3.x, Apache
   License 2.0 (OpenSSL 1.1.1 uses the OpenSSL/SSLeay dual license). It is
   linked dynamically as the system library, and only when
   `SONDER_WITH_TLS=ON` with the openssl backend (the default on
   Linux/macOS). Nothing is vendored or fetched. Schannel/crypt32/bcrypt are
   Windows OS components.*

4. **Optional follow-ups, not done here**:
   - CLI and bench flags that map to `OllamaBackendOptions::tls` in
     `tools/sonder-infer/main.cpp` and `bench/tools/sonder_bench.cpp`, e.g.
     `--tls-ca <pem>`, `--tls-pin-sha256 <hex>`, `--tls-pinned-cert <pem>`,
     `--tls-server-name <name>`, `--tls-insecure`, plus `--allow-remote`.
   - A `docs/MODULES.md` row, and README/ROADMAP mentions.
   - Flip `src/backends/ollama/README.md`: its line "https:// is refused"
     now holds only for non-TLS builds.

## Files

Owned (new):

- `cmake/SonderTls.cmake`: option, backend selection, sources, `SONDER_HAS_TLS`
  (PUBLIC define), OpenSSL / Schannel link, `tests/tls` registration.
- `src/net/tls.hpp`: `net::TlsOptions` plus `kTlsCompiledIn`. Always compiled,
  so request structs keep the same layout in both configurations.
- `src/net/tls_stream.hpp`: internal `TlsStream` interface, `tls_connect()`,
  and helpers.
- `src/net/tls_common.cpp`: fingerprint parsing, the insecure warning, and
  socket wait (TLS ON only).
- `src/net/tls_openssl.cpp`: OpenSSL backend. It uses its own socket BIO with
  `send(MSG_NOSIGNAL)`, so a peer hang-up is reported as an error and never
  raises SIGPIPE.
- `src/net/tls_schannel.cpp`: Schannel backend.
- `tests/tls/CMakeLists.txt`, `tests/tls/test_tls.cpp`,
  `tests/tls/tls_test_support.hpp`: loopback tests.
- `docs/integration/tls.md` (this file).

Small hooks in existing files (please review):

- `src/net/http_client.hpp`: includes `tls.hpp`; `HttpRequest` gains
  `bool use_tls` and `TlsOptions tls`; comments.
- `src/net/http_client.cpp`:
  - `parse_url` accepts `https` (default port 443) only under
    `#if defined(SONDER_HAS_TLS)`. Otherwise it still returns `unsupported`,
    now with a hint to use SONDER_WITH_TLS.
  - `http_request` sends and receives through a `TlsStream` when `use_tls`
    is set. Without TLS support in the build, `use_tls` returns
    `unsupported`. The plain-socket path is unchanged apart from being
    routed through a `send_bytes` lambda.
- `src/backends/ollama/include/sonder/inference/backends/ollama.hpp`: new
  `OllamaTlsOptions` struct and an `OllamaBackendOptions::tls` field, plus
  header comments. This is additive: existing aggregate initialisers are
  unaffected.
- `src/backends/ollama/ollama_client.cpp`: `make_request` accepts `https` once
  `parse_url` has (so only in TLS builds) and copies `config.tls` into the
  request. Non-loopback hosts still need `allow_remote` for https too, so no
  remote host is ever contacted by default.
- `src/backends/ollama/tests/ollama_client_tests.cpp` and
  `test_ollama_adapter.cpp`: the two assertions "https is unsupported" are
  wrapped in `#if defined(SONDER_HAS_TLS)`. In TLS builds they instead check
  `invalid_argument` (remote refused without `allow_remote`, before any
  connection) and `port == 443`. TLS-OFF behaviour and assertions are
  unchanged.

## Behaviour

`net::TlsOptions` (and the identical `OllamaTlsOptions`):

| Field | Meaning |
| --- | --- |
| `ca_bundle_path` | PEM file of trusted CA certificates. Empty means the system trust store. When set, **only** the bundle is trusted (both backends). |
| `pinned_sha256` | SHA-256 fingerprint of the leaf certificate (DER), as printed by `openssl x509 -noout -fingerprint -sha256`. Hex, case-insensitive, `:` optional. |
| `pinned_cert_path` | PEM file of the exact expected leaf certificate. If both pins are set, both must match. |
| `insecure_skip_verify` | Turns off chain and host-name checks. Every handshake prints a loud `[sonder-tls] WARNING` to stderr. A configured pin is still enforced. |
| `server_name` | Overrides the name used for SNI and verification (default: the URL host). |
| `handshake_timeout` | Bound on the handshake (default 10 s). The request's `connect_timeout` and total `request_timeout` still apply. |

Verification modes:

1. `insecure_skip_verify`: no verification, loud warning.
2. Pin set and no CA bundle ("pin-only", for self-signed nodes): the leaf
   must match the pin. Chain and name are not checked.
3. Otherwise: the chain must verify against the bundle (or the system store),
   and the certificate must match the host name or IP address via
   subjectAltName (or `server_name`). A pin, if set, must also match.

Errors: `invalid_argument` for bad options (unreadable CA or pin file,
malformed fingerprint), `timeout`, `cancelled`, `protocol_error` for handshake
failures, untrusted chains, name mismatches and pin mismatches (the message
says which), and `io_error` for failures after the handshake.

Example for Node1 (the CA must have issued a certificate with
`subjectAltName = IP:10.77.0.2`, or set `server_name` to the name in the
certificate):

```cpp
sonder::inference::OllamaBackendOptions o;
o.base_url = "https://10.77.0.2:8443";
o.allow_remote = true;                          // required for any non-loopback host
o.tls.ca_bundle_path = "C:/sonder/node1-ca.pem"; // or:
// o.tls.pinned_sha256 = "AB:CD:...";           // pin-only for a self-signed node cert
auto backend = sonder::inference::make_ollama_backend(o);
```

Known limitations: no client certificates (mTLS); no revocation checking.
On Schannel: TLS 1.2 only (`SCHANNEL_CRED`; TLS 1.3 would need
`SCH_CREDENTIALS`), and no close_notify is sent on close. The HTTP client
still uses one connection per request (`Connection: close`).

## Tests (`tests/tls`, CTest `sonder.tls.*`, 14 cases)

At test time the tests generate a throwaway PKI with OpenSSL in a temporary
directory: a test CA, an unrelated CA, a leaf certificate for
`IP:127.0.0.1, DNS:localhost`, and a leaf for `DNS:node1.invalid` only. No
keys are checked in. A TLS server runs on `127.0.0.1:<ephemeral>` inside the
test process. No external host is contacted. `10.77.0.2` appears only in
parse-only and refused-before-connect checks.

Covered:

- https URL parsing and the backend name
- fingerprint normalisation
- a trusted CA succeeds (GET, plus a 100 KB POST)
- an untrusted chain fails (system store)
- an unrelated CA bundle fails
- a host-name mismatch fails, and a `server_name` override succeeds
- a pin mismatch fails (with CA, pin-only, insecure, and pinned-cert file)
- a matching pin succeeds (with CA, pin-only, and cert file)
- insecure mode connects
- bad options are rejected before any request is sent
- handshake timeout against a silent peer
- cancellation
- `OllamaClient::version()` over https (trusted / untrusted / wrong pin)
- https to a non-loopback host still needs `allow_remote`

Local results (Linux, GCC, Ninja, `-j4`, RelWithDebInfo, warnings as errors):

| Configuration | Result |
| --- | --- |
| TLS OFF, root not wired (the PR as merged before wiring) | 251/251 pass (same as main) |
| TLS OFF, root wired (`include(SonderTls)`) | 251/251 pass, no OpenSSL linked |
| TLS ON, OpenSSL 3.5 | 265/265 pass (251 + 14 `sonder.tls.*`) |
| TLS ON, OpenSSL, Debug + ASan/UBSan | TLS and Ollama tests pass |
| TLS ON, Schannel, MinGW-w64 cross build (`-Werror`) run under Wine 10 | 265/265 pass |
| TLS OFF, MinGW-w64 under Wine | 251/251 pass |
