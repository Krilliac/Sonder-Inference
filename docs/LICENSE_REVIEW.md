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

Sonder Inference is licensed under the **MIT License** (root [`LICENSE`](../LICENSE),
Copyright (c) 2026 Krilliac), decided 2026-09-26 (ADR-015).

- Rule applied (Nate: "any license will do, whatever matches my other repos"): use
  the license most of Krilliac's other repositories use; MIT if none or a tie.
- Survey (2026-09-26, GitHub search for root license files across the 27 other
  Krilliac repositories): MIT in DuetOS, Lightforge, BrowserGame, SparkTemplates, plus
  ReSymbol (dual MIT/Apache-2.0); Apache-2.0 in Sonder-runtime and
  OmegaStrain-Reimplementation (plus ReSymbol); GPL-3.0 in Blackice-Server;
  custom licenses in SparkEngine and smellslikenapalm; no root license in the
  rest. MIT is the plurality (5 vs 3 counting the dual-licensed repo for both).
- Compatibility: every adopted dependency (llama.cpp/GGML, doctest, cpp-httplib)
  is MIT, which permits use, modification, static linking and redistribution under
  an MIT project provided the upstream copyright and permission notices are
  preserved in distributions that contain their code.

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
- revision/tag: `b11195` (commit `d834d44e643681f7b046a22d357335f6f4ff6107`); fetched by
  CMake FetchContent from the GitHub tag tarball, SHA-256
  `d9818b7786c8a3b063bac7eb7dc8710e275638f1e288e206720c9e22e7ca216a`
- evaluated: 2026-09-26
- license: MIT ("Copyright (c) 2023-2026 The ggml authors"), top-level `LICENSE` at that tag;
  ggml has no separate license file in-tree (covered by the same MIT license)
- intended use: first native execution backend (`src/backends/llamacpp`), optional,
  `SONDER_WITH_LLAMA_CPP=OFF` by default
- linkage/process boundary: in-process static linking of `llama` and `ggml` (+ ggml-cpu)
  libraries built from source; no llama.cpp code copied into this repository
- subdependencies: only the `llama`, `ggml`, `ggml-base` and `ggml-cpu` archives are built
  and linked (`llama` links only `ggml`). llama.cpp's `vendor/` (nlohmann/json MIT,
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
  bumped deliberately by changing tag + SHA-256 together in `src/backends/llamacpp/CMakeLists.txt`.
  GGUF files are untrusted input parsed by llama.cpp; load only trusted model files.
  GitHub tag tarballs are pinned by hash; if GitHub regenerates an archive the hash check fails
  closed (update hash after verifying the tag commit)
- approved: yes, approved by Nate on 2026-09-26 (MIT, compatible with the project's MIT license;
  keep the notice requirement above)

### cpp-httplib
- repository: https://github.com/yhirose/cpp-httplib
- revision/tag: v0.58.0 (single `httplib.h`, SHA-256 `aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5`)
- evaluated: 2026-09-26
- license: MIT
- intended use: in-process fake Ollama HTTP server for `src/backends/ollama` tests only
- linkage/process boundary: header-only, compiled into `sonder_ollama_tests` only; never linked into `sonder_inference`
- notices required: MIT notice (inside the header); not redistributed
- security/maintenance notes: fetched by CMake FetchContent from a pinned URL with `URL_HASH` in `src/backends/ollama/tests/CMakeLists.txt`
- approved: yes (test-only), approved by Nate on 2026-09-26 (MIT, compatible with the project's MIT license)

### Ollama (process boundary only)
- repository: https://github.com/ollama/ollama
- revision/tag: whatever server the user runs locally
- evaluated: 2026-09-26
- license: MIT (upstream); not relevant to linking because no code is used
- intended use: compatibility backend over its HTTP API
- linkage/process boundary: separate process over loopback HTTP; no Ollama code is copied or linked
- notices required: none
- security/maintenance notes: loopback-only by default; API behaviour is taken from the public REST API docs
- approved: yes (HTTP API use only)
