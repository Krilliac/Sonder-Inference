# Integration notes: `feat/python-bindings`

Scope: new `bindings/python/` (ctypes package, pytest suite) and a new
`.github/workflows/python.yml`. No changes to `src/`, `include/`, root CMake
or `ci.yml`.

## Shared library

There is no dedicated shared-library target. The workflow builds the
existing `sonder_inference` target with `-DBUILD_SHARED_LIBS=ON`, tests and
CLI off, and `--target sonder_inference`:

- Root CMake already defines `SONDER_INFERENCE_SHARED` (PUBLIC) and
  `SONDER_INFERENCE_BUILDING` (PRIVATE) in that mode. The C ABI therefore
  gets `__declspec(dllexport)` on Windows and default visibility on Linux.
- Only that one target is built because `sonder-bench` is always added by
  `bench/CMakeLists.txt`, and it, the CLI and the doctest executables link
  C++ symbols. A Windows DLL exports only the C ABI, so a full
  `BUILD_SHARED_LIBS=ON` build is expected to fail to link there. (Not
  tried: the workflow deliberately avoids it.)

Suggested root changes, for the lead to decide:

1. Add an option such as `SONDER_BUILD_SHARED_C_ABI` that creates a separate
   `sonder_inference_shared` SHARED target, or gate `sonder-bench` on
   `SONDER_BUILD_CLI`, so a plain shared build works on Windows.
2. Install rules for the shared library and the header, so wheels can bundle
   it. The package already looks in its own directory and lists
   `*.so`/`*.dll`/`*.dylib` as package data.

## Text chat entry point

The original proposal here was a two-pointer contiguous message array. The
implemented additive ABI v1 entry uses independently versioned records and
an array of pointers, so appending record fields cannot change array stride.
`Session.chat()` now marshals those records and shares the generate trampoline.
Libraries without the additive export still load and report Unsupported for
chat alone. See [cabi-chat.md](cabi-chat.md) for the final signature, bounds,
lifetime, consent and qualification contract. Session metadata and telemetry
callbacks are separate pending ABI work.

## Other ABI gaps seen from the bindings (optional)

- `sonder_model_load` takes no options, so there is no way to pass load-time
  settings such as a context size or device choice.
- The mock backend's `token_delay` is not reachable from the C ABI. Tests
  therefore cancel from inside the callback rather than from another thread.

## Verification (local, Linux, Python 3.13)

Shared library built as above from main `566f859`: 59 pytest cases pass.
