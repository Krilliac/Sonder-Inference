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

## For the integrator (not done here, outside this branch's files)

1. **Engine/session wiring**: when the session builds a sampler chain, use
   `sampling::make_chain(core_config, constraint, vocab_size)`. It already
   carries the new fields, and passing `vocab_size` range-checks
   `logit_bias` token ids. `num_ctx` is not a chain concern: the engine
   should pass it to the backend's context setup (llama.cpp `n_ctx`); Ollama
   gets it through the request options.
2. **Telemetry**: `session.cpp::sampling_json` still records only the old
   fields. Suggested additions: `typical_p`, `repeat_last_n`,
   `presence_penalty`, `frequency_penalty`, `num_ctx`, `logit_bias_count`.
3. **llama.cpp adapter**: `ToLlamaSampling` (`llamacpp_internal.hpp`) does
   not map the new fields yet. `SamplingParams::repeat_last_n` exists and
   `llama_sampler_init_penalties` already takes frequency/presence (currently
   passed `0.0F`); typical_p and logit_bias have llama.cpp samplers too.
4. **CLI / bench**: `tools/sonder-infer` has no flags for the new fields, and
   `bench/src/benchmark.cpp` does not record them in reports.
5. **Ollama + logit_bias**: the adapter silently drops `logit_bias` because
   Ollama's native API has no equivalent. If silent dropping is not wanted,
   the Ollama backend could reject non-empty `logit_bias` with
   `ErrorCode::unsupported`. That is a backend-level decision and is left open.
6. `docs/integration/sampling.md` says the core does not carry typical_p,
   penalties or logit_bias. That is now out of date; this file supersedes that
   paragraph.

## Verification (local, Linux)

- `cmake --preset ci-linux` (GCC, Release, `-Werror`): 264/264 CTest cases
  pass (251 on the base commit, +13 new cases).
- Clang Debug with `-Wshadow -Wconversion -Werror` plus ASan/UBSan: 264/264.
- MSVC was not built locally; CI covers it.
