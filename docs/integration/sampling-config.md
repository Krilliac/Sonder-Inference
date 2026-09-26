# Integration notes: `feat/sampling-config`

Scope: this branch reconciles the core `SamplingConfig` with the sampler chain
in `src/sampling/`, passes the new request options through to Ollama, and
exposes them in the C ABI. No root CMake, presets, CI, `engine.cpp` or
`session.cpp` changes.

## Files touched

| file | change |
| --- | --- |
| `include/sonder/inference/sampling.hpp` | new fields + `TokenLogitBias`, new limits |
| `src/engine/sampling_config.cpp` | validation of the new fields (core config impl only) |
| `src/sampling/src/core_bridge.cpp`, `core_bridge.hpp` | `from_core()` maps the new fields |
| `src/backends/ollama/ollama_client.cpp` | `sampling_to_options()` only |
| `include/sonder_inference.h`, `src/engine/c_api.cpp` | appended C ABI fields, `from_c()` / init / struct_size checks only |
| tests | `tests/test_sampling.cpp`, `tests/test_c_api.cpp`, `src/sampling/tests/test_core_bridge.cpp`, `src/backends/ollama/tests/ollama_decoder_tests.cpp` |

## New core fields (all defaults change nothing)

| field | default | valid range | chain (`SamplerConfig`) | Ollama option |
| --- | --- | --- | --- | --- |
| `typical_p` | `1.0` (off) | (0, 1] | `typical_p` | `typical_p` (sent when != 1) |
| `repeat_last_n` | `64` | -1 (whole ctx), 0 (off), [1, 2^24] | `penalty_last_n` | `repeat_last_n` (sent when != 64) |
| `presence_penalty` | `0.0` (off) | [-2, 2] | `presence_penalty` | `presence_penalty` (sent when != 0) |
| `frequency_penalty` | `0.0` (off) | [-2, 2] | `frequency_penalty` | `frequency_penalty` (sent when != 0) |
| `logit_bias` | empty | <= 1024 entries, token >= 0, unique tokens, bias in [-100, 100] or -inf | `logit_bias` | not forwarded (Ollama has no option) |
| `num_ctx` | `0` (backend default) | 0 or [1, 2^24] | none (not a sampler stage) | `num_ctx` (sent when > 0) |

`repeat_penalty` already existed (default 1.1, (0, 10]) and is unchanged.
`repeat_last_n = 64` equals what `from_core()` hard-coded before and the
Ollama/llama.cpp default, so the chain and Ollama behave as before.

With a default config, `sampling_to_options()` emits exactly the same keys as
before (tested by comparing the serialized JSON), and `from_core()` produces
the same `SamplerConfig` as before.

## C ABI (`include/sonder_inference.h`)

- `SONDER_ABI_VERSION` stays `1`: the change is append-only.
- New `sonder_logit_bias { int32_t token; float bias; }`.
- Appended to `sonder_sampling_config` after `max_tokens`: `typical_p`,
  `presence_penalty`, `frequency_penalty`, `repeat_last_n`, `num_ctx`,
  `logit_bias` (borrowed pointer, copied during the call) and
  `logit_bias_count`.
- Versioning: `struct_size` may be anything from the original layout size
  (fields up to `max_tokens`) upward. The appended block is read only when
  `struct_size >= sizeof(sonder_sampling_config)`; older callers get the
  defaults. The check is all-or-nothing on purpose: the original struct has
  tail padding where `typical_p` now sits, so per-field `offsetof` gating
  would read uninitialised padding from old callers. `c_api.cpp` keeps a
  private copy of the original layout with `static_assert`s on the shared
  offsets.
- Previously `sonder_sampling_config_validate` / `sonder_session_create`
  required `struct_size >= sizeof(current struct)`. They now accept the
  original size, which is what keeps old binaries working.
- `logit_bias == NULL` with `logit_bias_count > 0` is rejected with
  `SONDER_ERROR_INVALID_ARGUMENT`.

## Integrator follow-ups (status)

1. **Session chain**: done. Sessions call
   `sampling::make_chain(config, nullptr, vocab_size)`, so the new fields
   apply and `logit_bias` ids are range-checked. `num_ctx` narrows the
   scheduler's context limit (`scheduler.enqueued.context_limit`).
2. **Telemetry**: done. The `sampling` object on `session.created` and
   `request.started` adds `typical_p`, `repeat_last_n`, `presence_penalty`,
   `frequency_penalty`, `logit_bias_count` and `num_ctx`
   (docs/TELEMETRY.md).
3. **llama.cpp adapter**: done. `ToLlamaSampling` maps every new field.
   The wrapper chain is logit bias, penalties (repeat/frequency/presence;
   `repeat_last_n = -1` means the whole context), top-k, typical, top-p,
   min-p, temperature, selector. Bias ids outside the vocab are rejected.
   The context is fixed at load, so a `num_ctx` larger than the loaded
   context is rejected with `invalid_argument`. A smaller one is honoured
   through the engine's scheduling limit.
4. **CLI / bench flags**: done (landed with the `feat/chat-cli` integration).
   `generate`, `chat` and `bench` accept `--typical-p`, `--repeat-last-n`,
   `--presence-penalty`, `--frequency-penalty`, `--num-ctx` and
   `--logit-bias TOKEN:BIAS[,TOKEN:BIAS...]` (BIAS may be `-inf`). Bench
   result files record the new fields under `sampling` (`logit_bias_count`
   instead of the biases).
5. **Ollama + logit_bias**: decided. The Ollama backend **rejects** a
   non-empty `logit_bias` with `ErrorCode::invalid_argument` and sends
   nothing to the server. It no longer drops the field silently. This
   applies to both `generate` and native `chat` (`/api/chat`).
   `sampling_to_options()` still omits it.
6. `docs/integration/sampling.md`: updated.

## Verification (local, Linux)

- `cmake --preset ci-linux` (GCC, Release, `-Werror`): 264/264 CTest cases
  pass (251 on the base commit, +13 new cases).
- Clang Debug with `-Wshadow -Wconversion -Werror` plus ASan/UBSan: 264/264.
- MSVC was not built locally; CI covers it.
