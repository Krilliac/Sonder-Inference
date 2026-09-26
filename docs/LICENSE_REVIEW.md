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
- repository: https://github.com/ggml-org/llama.cpp
- revision/tag: not pinned yet
- evaluated: 2026-09-26 (preliminary)
- license: MIT at `master` (Copyright (c) 2023-2026 The ggml authors); must be re-read at the adopted revision together with vendored subcomponents
- intended use: first direct native backend (ADR-003)
- linkage/process boundary: planned static or shared linking behind the `SONDER_WITH_LLAMA_CPP` CMake option (reserved, OFF, currently a configure error)
- notices required: to be determined at adoption
- security/maintenance notes: fast-moving API; pin a tag and wrap it behind the backend interface
- approved: **no** (deferred; nothing is fetched, vendored, or linked yet)

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
