# sonder-inference (Python)

ctypes bindings for the Sonder Inference C ABI (`include/sonder_inference.h`).
Pure Python: installing needs no compiler. You provide the shared library.

## Build the shared library

```sh
cmake -S . -B build-shared -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DSONDER_BUILD_TESTS=OFF -DSONDER_BUILD_CLI=OFF
cmake --build build-shared --target sonder_inference
```

This produces `libsonder_inference.so` on Linux or `sonder_inference.dll` on Windows.
On Windows, run it from a VS developer prompt, with
`-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`.

## Install and point at the library

```sh
pip install ./bindings/python
export SONDER_INFERENCE_LIB_DIR=$PWD/build-shared      # or SONDER_INFERENCE_LIBRARY=/path/to/file
```

The library is searched for in this order:

1. an explicit `sonder_inference.load_library(path)` (a file or a directory);
2. `SONDER_INFERENCE_LIBRARY` (a file);
3. `SONDER_INFERENCE_LIB_DIR` (a directory);
4. the package directory (for wheels that bundle the library);
5. `ctypes.util.find_library("sonder_inference")`.

The bindings check `sonder_abi_version() == 1` when the library loads.

## Usage

```python
import sonder_inference as si

with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as engine:
    engine.register_mock_backend()            # or engine.register_ollama_backend()
    with engine.load_model("mock", "mock:tiny") as model:
        cfg = si.SamplingConfig.greedy(max_tokens=32, seed=7).replace(
            typical_p=0.95, presence_penalty=0.3, num_ctx=4096, logit_bias={3: 1.5})
        with engine.create_session(model, cfg) as session:
            result = session.generate("hello", on_token=lambda t: print(t, end=""))
            print(result.outcome, result.completion_tokens, result.ttft_ms)

            for chunk in session.stream("streaming"):   # worker thread; close() cancels
                print(chunk, end="")
```

- `on_token(text)`: return `False` to stop early. An exception raised inside it
  stops generation and is re-raised from `generate()`. Chunks are UTF-8 decoded
  incrementally, so a multi-byte character is never split across chunks.
- `Session.cancel()` is thread-safe. A cancelled request returns a result with
  `outcome == Outcome.CANCELLED`; it does not raise.
- `SamplingConfig` fields default to `None`, which means "use the library
  default" (the defaults come from `sonder_sampling_config_init`).
  `SamplingConfig.defaults()` returns them all filled in (the appended fields
  stay `None` on a library that predates them). `validate()` asks the
  library to check the config.
- Errors map onto `SonderError` subclasses, one per `sonder_status`. Each also
  subclasses the natural builtin: `InvalidArgumentError(ValueError)`,
  `NotFoundError(LookupError)`, `SonderTimeoutError(TimeoutError)`,
  `UnsupportedError(NotImplementedError)`, and so on.
- Engines, models and sessions are context managers. `close()` is idempotent,
  and handles are also released at garbage collection. Closing an engine
  closes its sessions and models first. A session keeps its model alive in C,
  so a model may be closed before the sessions created from it.
- `close()` is thread-safe. Closing a session (or its engine) while
  `generate()`/`chat()`/`stream()` runs on another thread cancels that request and
  destroys the C handle only after the call returns; any call on a closed
  handle raises `InvalidStateError`. Calling `close()` from inside the
  session's own token callback raises `InvalidStateError`; return `False` or
  call `cancel()` there instead.
- Strings passed to the C ABI (prompt, chat roles/content, backend, model, paths, URLs) must not
  contain NUL characters; they raise `ValueError` instead of being truncated.

## Struct versioning

`sonder_sampling_config` carries `struct_size`. The appended fields
(`typical_p`, `presence_penalty`, `frequency_penalty`, `repeat_last_n`,
`num_ctx`, `logit_bias`) are only read when `struct_size` covers the whole
current struct.

On first use the bindings check whether the loaded library reads that tail
(an invalid `typical_p` is rejected only by a library that reads it).
Against an older library they send the original `struct_size`. If you set
any appended field in that case, they raise `UnsupportedError` rather than
letting the library silently ignore it.

## Chat

`Session.chat()` accepts 1–1,024 `ChatMessage(role, content)` records or mappings
containing only `role` and `content`. Roles are `system`, `user`, `assistant`
and `tool`; the last message must be `user` or `tool`. Content is text only.
Empty content is allowed; NUL characters, missing fields and unknown roles
are rejected. Reasoning, tool-call objects, images and other mapping fields
raise `UnsupportedError` rather than being silently discarded.

```python
result = session.chat([
    si.ChatMessage("system", "Be concise"),
    {"role": "user", "content": "Hello"},
], on_token=lambda text: print(text, end=""))
```

Native backend chat receives structured messages. Other backends use the
engine's existing role-labelled prompt fallback. Results, incremental UTF-8
decoding, callback stop/exception propagation, cancellation and concurrent
close behavior match `generate()`. There is no `chat_stream()` iterator in this
slice. Iterables consume at most 1,025 records before rejecting an oversized
conversation; message text length is not bounded by this count limit.

The bindings still load older ABI v1 libraries; chat raises `UnsupportedError`
when `sonder_session_chat` is absent. See [the C contract](../../docs/integration/cabi-chat.md).

## Tests

```sh
pip install "./bindings/python[test]"
cd bindings/python && SONDER_INFERENCE_LIB_DIR=../../build-shared python -m pytest
```

The tests use the deterministic mock backend. No network access or model
weights are needed.
