# Upstream License and Dependency Review

This repository's research catalog intentionally includes projects under different licenses and governance models.

Do **not** assume that being public/open source means code can be copied into Sonder Inference under whatever license Sonder eventually uses.

Before adding an upstream code dependency or copying an implementation:

1. identify exact repository and revision;
2. read the license at that revision;
3. check submodule/subdependency licenses;
4. distinguish linking, dynamic loading, process boundary, and copied code;
5. preserve notices/attribution where required;
6. check model/weight licenses separately from engine code;
7. record the decision here or in an ADR.

For research papers, implement concepts from the paper/design independently unless a deliberate code-dependency decision is made.

## Project license

Sonder Inference is licensed under the **Apache License, Version 2.0** (root
[`LICENSE`](../LICENSE) with the full standard text, plus [`NOTICE`](../NOTICE):
"Sonder-Inference, Copyright 2026 Nate Witkowski"). Decided 2026-09-26
(ADR-015, revised).

- Decision: Nate chose Apache-2.0 directly on 2026-09-26. This replaces the
  earlier MIT choice (PR #7), which came from a survey of Krilliac's other
  repositories. That survey no longer applies.
- Compatibility: every adopted dependency is MIT (llama.cpp/GGML v1.0.0,
  cpp-httplib v0.58.0, doctest v2.5.3). MIT code may be used, modified,
  statically linked and redistributed inside an Apache-2.0 project as long as
  the MIT copyright and permission notice is kept. `NOTICE` carries those
  notices for llama.cpp/GGML, cpp-httplib and doctest. MIT adds no conditions
  beyond Apache-2.0's, so the combined work can be distributed under
  Apache-2.0 with the MIT notices attached.
- Patent terms: Apache-2.0 section 3 grants a patent license from
  contributors. MIT dependencies neither add nor remove patent terms.

## Dependency record template

```markdown
### <dependency>
- repository:
- revision/tag:
- evaluated:
- license:
- intended use:
- linkage/process boundary:
- notices required:
- security/maintenance notes:
- approved:
```

No license conclusions from the initial broad web sweep are considered authoritative until verified against the exact adopted revision.

## Dependency records

### doctest
- repository: https://github.com/doctest/doctest
- revision/tag: v2.5.3 (release archive SHA-256 `174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331`)
- evaluated: 2026-09-26
- license: MIT (`LICENSE.txt`, Copyright (c) 2016-2023 Viktor Kirilov), read at v2.5.3
- intended use: unit-test framework for `tests/` only
- linkage/process boundary: header-only, compiled into the test executable only; not linked into `sonder_inference` and not shipped
- notices required: MIT notice travels with the downloaded archive; nothing is redistributed by this repository
- security/maintenance notes: fetched by CMake FetchContent with a pinned hash (`cmake/SonderDoctest.cmake`); offline builds can set `FETCHCONTENT_SOURCE_DIR_DOCTEST`
- approved: yes (test-only)

### llama.cpp / GGML
- repository: https://github.com/ggml-org/llama.cpp (ggml vendored in-tree under `ggml/`)
- revision/tag: `v1.0.0` (commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`); fetched by
  CMake FetchContent from the GitHub tag tarball, SHA-256
  `3f5f750e36326aea4d65c7623e7d597a33e395d65f4d76e5e0532171f252532b`
- pin manifest: `cmake/llamacpp-pin.json` is the single source of the tag, commit and SHA-256 above;
  `python -I scripts/llamacpp_pin.py check` fails when this entry disagrees with it
- evaluated: 2026-09-26 (first pin); re-checked 2026-11-26 for the move to this tag
- license: MIT ("Copyright (c) 2023-2026 The ggml authors"), top-level `LICENSE` at that tag
  (byte-identical to the first pin's); ggml has no separate license file in-tree (covered by
  the same MIT license)
- intended use: first native execution backend (`src/backends/llamacpp`), optional,
  `SONDER_WITH_LLAMA_CPP=OFF` by default
- linkage/process boundary: in-process static linking of `llama` and `ggml` (+ ggml-cpu)
  libraries built from source; no llama.cpp code copied into this repository
- subdependencies: only the `llama`, `ggml`, `ggml-base` and `ggml-cpu` archives are built
  and linked (`llama` links `ggml` and the platform thread library). llama.cpp's `vendor/`
  (nlohmann/json MIT,
  miniaudio, stb, sheredom, hash libs; cpp-httplib skipped) is configured but not
  compiled into or linked with those archives, with
  `LLAMA_BUILD_COMMON/TOOLS/SERVER/EXAMPLES/TESTS=OFF` and `LLAMA_OPENSSL=OFF`. ggml uses the toolchain's OpenMP
  runtime when available (libgomp: GPL-3.0 with GCC Runtime Library Exception; MSVC vcomp:
  Visual C++ redistributable terms); set `GGML_OPENMP=OFF` to avoid it. Optional accelerator
  backends (CUDA, Vulkan, Metal, ...) are off unless enabled and bring their SDK terms
- notices required: include the llama.cpp MIT license text and copyright notice in any binary
  distribution that contains the statically linked libraries
- model weights: separate from engine licensing; no weights are committed. Test models are
  supplied at runtime via `SONDER_TEST_GGUF` and carry their own licenses (the tinyllamas
  stories models used for local verification come from karpathy/llama2.c, MIT)
- security/maintenance notes: fast-moving upstream (tagged builds several times a day); pin is
  bumped deliberately by changing tag, commit, URL and SHA-256 together in `cmake/llamacpp-pin.json`
  and verifying the tarball with `python -I scripts/llamacpp_pin.py verify`.
  The ggml RPC server is not built here (`GGML_RPC` off); this tag predates upstream's
  server-side RPC request validation (see docs/BACKENDS.md), so any separately built
  `rpc-server` from it stays on a private link.
  GGUF files are untrusted input parsed by llama.cpp; load only trusted model files.
  GitHub tag tarballs are pinned by hash; if GitHub regenerates an archive the hash check fails
  closed (update hash after verifying the tag commit)
- approved: yes, approved by Nate on 2026-09-26 (MIT, compatible with the project's Apache-2.0
  license; notice kept in `NOTICE`)
- vendored tree and SI patches (in progress, RET-09 in [ROADMAP_SI_ONLY.md](ROADMAP_SI_ONLY.md)):
  vendoring the `ggml`/`llama.cpp` tree and choosing the license of SI's patch
  series need a new review and an owner approval recorded here before RET-09
  lands (`third_party/ggml-patches/` stays empty until that review and owner
  approval are recorded here; `PQ2_0 -> Q2_0` re-blocking in
  `sonder-infer reblock-pq2` is SI's own Apache-2.0 code and copies no fork
  source). Upstream code stays MIT, with its copyright and permission notice
  kept in `NOTICE` and binary distributions. This file does not currently carry
  a verified license record for the PrismML fork; if stock `Q2_0` does not
  reach parity and any `PQ2_0` kernel code is ported from the PrismML fork
  rather than written independently, its exact repository, revision, and
  license must be verified and recorded here before adoption.

### llguidance (optional, `SONDER_WITH_LLGUIDANCE=ON`)
- repository: https://github.com/guidance-ai/llguidance
- revision/tag: v1.0.1, commit `d795912fedc7d393de740177ea9ea761e7905774` (`parser/Cargo.toml` and
  `toktrie/Cargo.toml` both read `version = "1.0.1"`); the same commit llama.cpp `v1.0.0` pins in
  `common/CMakeLists.txt` for `LLAMA_LLGUIDANCE`. Pinned in `src/backends/llamacpp/CMakeLists.txt`
  (`SONDER_LLGUIDANCE_GIT_TAG`); configure warns if the llama.cpp tree in use pins a different commit
- evaluated: 2026-11-26, against that exact commit: `LICENSE` ("MIT License, Copyright (c)
  Microsoft Corporation."), `license = "MIT"` in `parser/Cargo.toml` (crate `llguidance`) and
  `toktrie/Cargo.toml`, and `cargo metadata` of the crates one `cargo build --release --package
  llguidance` compiles on x86_64-pc-windows-msvc
- license: MIT. Its Rust dependency closure is 48 crates (including `llguidance` and `toktrie`
  themselves), every one permissively licensed. By SPDX expression: 32 `MIT OR Apache-2.0`,
  7 `MIT` (`llguidance`, `toktrie`, `derivre`, `referencing`, `fluent-uri`, `strum`,
  `strum_macros`), 2 `Zlib OR Apache-2.0 OR MIT`, 2 `Apache-2.0 OR MIT`, and one each of `MIT-0`
  (borrow-or-share), `Unlicense OR MIT` (memchr),
  `Apache-2.0 OR BSL-1.0` (ryu), `(MIT OR Apache-2.0) AND Unicode-3.0` (unicode-ident) and
  `BSD-2-Clause OR Apache-2.0 OR MIT` (zerocopy). The full list with versions comes from the
  `cargo metadata` command above (check it again whenever the pin moves)
- intended use: constrained decoding for `response_format: {type: json_schema}` in the native
  llama.cpp backend (`llguidance_sampler.cpp` adapts llama.cpp's `common/llguidance.cpp`, MIT, to this
  repository's sampler contract). The default build does not use it
- linkage/process boundary: in-process static link of `llguidance.lib` / `libllguidance.a`, built by
  `cargo build --release --package llguidance` from a `git clone` at the pinned commit, only with
  `SONDER_WITH_LLGUIDANCE=ON` (requires `cargo` and `git`; configure fails with a clear message when
  either is missing and the option is ON). No llguidance code is copied into this repository other
  than the adapted sampler glue
- notices required: include the llguidance MIT license text and Microsoft copyright notice, plus the
  notices of the crates above, in any binary distribution built with the option ON. Not yet added to
  `NOTICE`; do that before shipping such a binary
- security/maintenance notes: the build downloads source from github.com and crates from crates.io at
  build time (the commit hash pins llguidance, but not the crate versions: `cargo build` uses the
  `Cargo.lock` committed at that commit); it is therefore opt-in and unsuitable for offline builds
  without a prepared cargo vendor/cache. Schemas are normalised (`additionalProperties: false`) and
  bounded (`json_schema_normalize.hpp`) before llguidance sees them
- approved: pending: not yet reviewed by Nate; the option stays OFF by default until then

### cpp-httplib
- repository: https://github.com/yhirose/cpp-httplib
- revision/tag: v0.58.0 (single `httplib.h`, SHA-256 `aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5`)
- evaluated: 2026-09-26
- license: MIT
- intended use: in-process fake Ollama HTTP server for `src/backends/ollama` tests only
- linkage/process boundary: header-only, compiled into `sonder_ollama_tests` only; never linked into `sonder_inference`
- notices required: MIT notice (inside the header); not redistributed
- security/maintenance notes: fetched by CMake FetchContent from a pinned URL with `URL_HASH` in `src/backends/ollama/tests/CMakeLists.txt`
- approved: yes (test-only), approved by Nate on 2026-09-26 (MIT, compatible with the project's
  Apache-2.0 license; notice kept in `NOTICE`)

### OpenSSL (optional TLS, system library)
- repository: https://github.com/openssl/openssl
- revision/tag: the system library, 1.1.1 or newer. CI pins what it links: the `ubuntu-24.04` image's libssl-dev (OpenSSL 3.0.x) for tls-linux, and, for the Windows in-test server only, OpenSSL 4.0.3 built in CI from the official release tarball `https://github.com/openssl/openssl/releases/download/openssl-4.0.3/openssl-4.0.3.tar.gz`, SHA-256 `325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9` (matches the `.sha256` published with the GitHub release and on openssl.org). Until 2026-09-29 this was Chocolatey `openssl` 4.0.2, which broke when slproweb.com deleted the 4.0.2 installer the package downloads
- evaluated: 2026-09-26; Windows CI source re-evaluated 2026-09-29 (4.0.3 tarball: `LICENSE.txt` is the Apache License 2.0, `VERSION.dat` 4.0.3)
- license: Apache License 2.0 for OpenSSL 3.x and 4.x; OpenSSL 1.1.1 uses the OpenSSL/SSLeay dual license
- intended use: TLS for the internal HTTP client (`src/net/tls_openssl.cpp`) so the Ollama backend can reach `https://` nodes
- linkage/process boundary: dynamic link to the system library, only when `SONDER_WITH_TLS=ON` with the openssl backend (default on Linux/macOS); the default build (TLS OFF) compiles and links nothing. Nothing is vendored or fetched. On Windows the TLS backend is Schannel (crypt32/secur32/bcrypt, OS components); OpenSSL is used there only by the loopback test server in `tests/tls`
- notices required: none for dynamic linking to the system library; nothing is redistributed
- security/maintenance notes: security updates come from the OS package. The Windows CI build (`tls-windows` in `.github/workflows/ci.yml`) downloads the tarball above, refuses it unless the SHA-256 matches, builds it with MSVC (`VC-WIN64A no-asm no-apps no-tests no-docs`) and caches the install under that hash; to move it, bump `OPENSSL_VERSION` and `OPENSSL_SHA256` together after checking the new release's hash and license. Certificate verification, pinning and the `insecure_skip_verify` warning are documented in `docs/integration/tls.md`
- approved: yes, approved by Nate on 2026-09-26 (dynamic link to the system library, OFF by default; CI revisions pinned as above)

### Ollama Cloud (hosted HTTPS API only)
- repository: none used; the service at https://ollama.com
- revision/tag: the hosted API as documented at https://docs.ollama.com/api/introduction.md (checked 2026-11-26)
- evaluated: 2026-11-26
- license: not applicable; no code is copied, linked or fetched. Use of the service is governed by Ollama's terms and privacy policy (https://ollama.com/privacy), which the operator accepts with their account
- intended use: the `ollama_cloud` provider (`src/backends/ollama_cloud`) sends chat requests to Ollama's hosted models
- linkage/process boundary: remote service over HTTPS through the existing internal HTTP client and the already-reviewed optional TLS layer (Schannel on Windows, the OpenSSL record above elsewhere). No new dependency
- notices required: none
- security/maintenance notes: opt-in per model; certificate verification always on; API key from an env var or token file, never logged or echoed (docs/integration/ollama-cloud.md "Secrets"). Prompts leave the machine, so `X-Sonder-Local-Only` requests are refused
- approved: no approval needed for code (none is used); recorded because prompts now cross a network boundary when an operator opts in

### Ollama (process boundary only)
- repository: https://github.com/ollama/ollama
- revision/tag: whatever server the user runs locally
- evaluated: 2026-09-26
- license: MIT (upstream); not relevant to linking because no code is used
- intended use: compatibility backend over its HTTP API (local Ollama backend
  is deprecated and scheduled for removal in RET-07; the planned opt-in Ollama
  Cloud provider in RET-02 remains strictly a network/HTTP boundary with no
  Ollama code copied or linked)
- linkage/process boundary: separate process over loopback HTTP; no Ollama code is copied or linked
- notices required: none
- security/maintenance notes: loopback-only by default; API behaviour is taken from the public REST API docs
- approved: yes (HTTP API use only)
