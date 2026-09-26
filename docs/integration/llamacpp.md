# Integration notes: llama.cpp backend (`feat/llamacpp-backend`)

Owner paths in this branch: `src/backends/llamacpp/**` and this file. No root,
core, docs, or other-module files are modified. Everything below that touches
root/docs is for the lead to apply.

## 1. Root CMake change required (one edit)

The root already lists `src/backends/llamacpp` in `SONDER_MODULE_DIRS`, and the
module gates itself on the existing root option `SONDER_WITH_LLAMA_CPP`
(default OFF). The only blocker is the reservation guard in the root
`CMakeLists.txt`, which must be replaced:

```cmake
# before
option(SONDER_WITH_LLAMA_CPP "Build the direct llama.cpp backend (not implemented yet)" OFF)
if(SONDER_WITH_LLAMA_CPP)
    message(FATAL_ERROR "SONDER_WITH_LLAMA_CPP is reserved; the llama.cpp backend is not implemented yet.")
endif()

# after
# Direct llama.cpp/GGML backend (src/backends/llamacpp). OFF by default: when ON,
# llama.cpp is fetched at a pinned tag and compiled (CPU; accelerators via GGML_*).
option(SONDER_WITH_LLAMA_CPP "Build the direct llama.cpp backend" OFF)
```

No `add_subdirectory` line is needed (module loop already covers it). The
option name is `SONDER_WITH_LLAMA_CPP` (the root's), not `SONDER_WITH_LLAMACPP`.

Optional, recommended: a non-blocking or nightly CI job
`-DSONDER_WITH_LLAMA_CPP=ON -DGGML_NATIVE=OFF` (about 1.5 min at `-j 4` on
Linux). The default CI presets stay unchanged and never fetch llama.cpp.

## 2. What the module provides when ON

- `sonder_backend_llamacpp` (static): policy-free wrapper over the llama.cpp C API,
  `src/backends/llamacpp/include/sonder/backends/llamacpp/llamacpp_backend.h`.
  Load/unload GGUF (full or vocab-only), tokenize/detokenize, chunked prefill
  (`n_batch`), streaming decode with UTF-8-safe text chunks, cancellation polled
  between tokens *and* inside `llama_decode` (llama.cpp abort callback), stop
  reasons, prefill/decode timings, ggml device enumeration (CPU; CUDA/Vulkan/
  Metal devices show up automatically if enabled through `GGML_*`), and
  telemetry hooks (model loaded/unloaded, prefill done, per-token, generation
  done). No llama.cpp header leaks out of the .cpp files.
- Core adapter compiled into `sonder_inference`:
  `sonder::inference::make_llamacpp_backend(LlamaCppBackendOptions)` in
  `sonder/inference/backends/llamacpp.hpp`, backend name `"llamacpp"`.
  Capabilities: tokenization, streaming, batched_prefill, deterministic.
  `load_model` takes a GGUF path (or a file name inside `model_dirs`);
  `device_id` `cpu:*` = no offload, `gpu:*` = offload `gpu_layers` (error
  `unsupported` if the build has no GPU backend). Maps SamplingConfig
  (temperature/top_k/top_p/min_p/repeat_penalty/seed/max_tokens/stop) onto a
  llama.cpp sampler chain; stop sequences are cut in the adapter and never
  emitted. Cancellation returns `ErrorCode::cancelled` like the mock backend;
  callback-false returns `StopReason::callback`. Reports backend token counts
  and prompt/eval ns.
- `SONDER_HAS_LLAMACPP_BACKEND=1` public define on `sonder_inference`, so the
  CLI/engine can register it with `#ifdef` (not done here: CLI/engine are core).

Suggested CLI hook for the lead (core file, not edited here):

```cpp
#ifdef SONDER_HAS_LLAMACPP_BACKEND
#include "sonder/inference/backends/llamacpp.hpp"
engine.register_backend(sonder::inference::make_llamacpp_backend());
#endif
```

## 3. Tests

- `sonder.llamacpp.*` (doctest via `sonder_add_module_tests`), 16 cases, no model
  weights: UTF-8 stream splitting, sampling validation/mapping, stop-sequence
  filter, error paths, pinned version, CPU device present, and real tokenization
  against llama.cpp's bundled vocab-only fixture `models/ggml-vocab-llama-spm.gguf`
  (tokenizer metadata only, shipped inside the llama.cpp source tarball).
- `sonder.llamacpp.integration`: runs only with `SONDER_TEST_GGUF=<file.gguf>`,
  otherwise exits 77 and CTest reports it as **Skipped** (never as passed).
  Optional `SONDER_TEST_THREADS`. Covers wrapper prefill/decode, greedy
  determinism, cancellation, callback stop, context overflow, seeded sampling,
  the core Backend adapter (stop sequences, cancellation) and Engine + Session
  with Observatory telemetry.

Verified on Linux against main `0c0b054` (preset `ci-linux`, GCC, Ninja,
`SONDER_WARNINGS_AS_ERRORS=ON`, `GGML_NATIVE=OFF`, root guard removed
locally): full suite 108 tests, 107 passed + `sonder.llamacpp.integration`
skipped (16 llamacpp unit cases among them). Sources also compile warning-free
with clang `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion`. With
`SONDER_TEST_GGUF` set the integration passed on `stories260K.gguf` and
`stories15M-q4_0.gguf` (ggml-org/models tinyllamas, 1.1 MiB / 17.5 MiB, used
by llama.cpp's own CI; not committed). stories15M Q4_0, 4 threads, AVX2:
decode ~1.6-2.0k tok/s (1 thread: ~0.9-1.0k), prefill ~4-5k tok/s on a
shared, loaded box (tiny test model; not a benchmark claim). Also verified:
the OFF build does not fetch llama.cpp. Not verified on MSVC in this slice.

Known noise: when a request is cancelled mid-`llama_decode`, llama.cpp logs
`llama_decode: failed to decode, ret = 2` at error level. That is the expected
abort path. Set `SONDER_LLAMACPP_VERBOSE=1` to see llama.cpp's info logs.

Offline builds: `-DFETCHCONTENT_SOURCE_DIR_LLAMACPP=/path/to/llama.cpp-b11195`.

## 4. Entry for docs/LICENSE_REVIEW.md

```markdown
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
- approved: pending lead/Nate sign-off
```

## Status on main (applied by the integrator)

- §1 applied: the root `FATAL_ERROR` guard is replaced by a plain
  `option(SONDER_WITH_LLAMA_CPP "Build the direct llama.cpp backend" OFF)`.
- §2 CLI hook applied: `tools/sonder-infer` registers `make_llamacpp_backend()`
  under `#if defined(SONDER_HAS_LLAMACPP_BACKEND)`.
- §4 applied: the dependency record is in `docs/LICENSE_REVIEW.md`
  (still `approved: pending lead/Nate sign-off`).
- CI: optional job `llamacpp-windows` (windows-latest, MSVC,
  `SONDER_WITH_LLAMA_CPP=ON`, `GGML_NATIVE=OFF`, `continue-on-error`), with the
  pinned source tarball cached and passed via `FETCHCONTENT_SOURCE_DIR_LLAMACPP`.
  The default ubuntu/windows jobs keep the option OFF.
