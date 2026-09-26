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

## Chat entry point (not in the C ABI yet)

`Backend::chat` landed in #10, but `include/sonder_inference.h` has no chat
function, so `Session.chat()` is a stub that raises `UnsupportedError`. A
proposed append-only addition that the bindings can map directly:

```c
typedef struct sonder_chat_message {
    const char* role;     /* "system" | "user" | "assistant" | "tool" */
    const char* content;  /* UTF-8, NUL-terminated */
} sonder_chat_message;

/* Same streaming/stats contract as sonder_session_generate. */
SONDER_API sonder_status sonder_session_chat(sonder_session* session,
                                             const sonder_chat_message* messages, size_t message_count,
                                             sonder_token_callback callback, void* user_data,
                                             sonder_generation_stats* out_stats);
```

Once that exists, the binding is about 20 lines: marshal a
`sonder_chat_message` array, then reuse the generate trampoline.

## Other ABI gaps seen from the bindings (optional)

- `sonder_model_load` takes no options, so there is no way to pass load-time
  settings such as a context size or device choice.
- The mock backend's `token_delay` is not reachable from the C ABI. Tests
  therefore cancel from inside the callback rather than from another thread.

## Verification (local, Linux, Python 3.13)

Shared library built as above from main `566f859`: 59 pytest cases pass.
