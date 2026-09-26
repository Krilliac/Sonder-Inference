# Integration notes: `feat/sampling`

Scope: `src/sampling/` only (library sources, headers, doctest tests, README).
No root CMake, preset, CI, or core files were touched.

## Build wiring

- `src/sampling/CMakeLists.txt` uses the lead's module hooks: `sonder_module_sources`,
  `sonder_module_include_directories(include)`, `sonder_module_define(SONDER_HAS_SAMPLER_CHAIN)`
  and `sonder_add_module_tests(sampling …)`. The root file already lists `src/sampling` in
  `SONDER_MODULE_DIRS`, so no root edit is needed.
- Tests use doctest via `sonder_add_module_tests`. Each case is registered as
  `sonder.sampling.<case>`: 90 cases in 7 files.
- Clean under `-Wall -Wextra -Wpedantic -Werror`, also with `-Wconversion -Wsign-conversion
  -Wshadow` and ASan/UBSan (GCC 14, Linux). MSVC was not built locally; CI covers it.

## Mapping the core placeholder config onto the chain

`sonder::inference::SamplingConfig` (include/sonder/inference/sampling.hpp) maps onto
`sonder::inference::sampling::SamplerConfig` with `from_core()` in `core_bridge.hpp`:

| core field | chain field | note |
| --- | --- | --- |
| `temperature` | `temperature`; `greedy = (temperature == 0)` | 0 selects the greedy selector |
| `top_p` | `top_p` | core forbids 0; the chain allows it (keeps `min_keep`) |
| `top_k` | `top_k` | 0 = disabled in both |
| `min_p` | `min_p` | |
| `repeat_penalty` | `repeat_penalty` | window = `penalty_last_n` (64, llama.cpp default) |
| `seed` | `seed` | unset means an entropy seed at build time |
| `stop` | `stop_sequences` | drive `StopSequenceMatcher` from the generation loop |
| `max_tokens` | — | generation-loop concern, not a sampler |

Fields the core does not carry yet keep their disabled defaults: `typical_p`,
`frequency_penalty`, `presence_penalty`, `logit_bias`, `stop_tokens`, `min_keep`
and `stage_order`. Suggested follow-up for the lead: either add these to the core
config and C ABI, or let sessions hold a `SamplerConfig` directly.

`make_chain(core, constraint, vocab)` runs the core `validate()`, then the chain
validation, then builds the chain. It returns `Result<SamplerChain>` carrying
`ErrorCode::invalid_argument`. Suggested engine or session usage for each decode step:

```cpp
auto chain = sampling::make_chain(req.sampling, nullptr, model.vocab_size()).value();
chain.accept_prompt(prompt_tokens);                 // penalties see the prompt
sampling::StopSequenceMatcher stops(req.sampling.stop);
for (;;) {
    auto r = chain.sample(backend_logits);          // span<const float>
    if (!r.ok()) { /* NoViableCandidates -> backend_error / invalid_state */ }
    chain.accept(r.token);
    if (chain.is_stop_token(r.token)) break;
    auto chk = stops.feed(detokenize(r.token));
    emit(chk.emit);
    if (chk.stopped) break;
}
emit(stops.flush());
```

The mock backend can keep its own greedy path. Switching it to `make_chain`
would give seeded, deterministic, parity-tested sampling everywhere.

## Licensing

The semantics follow llama.cpp's public sampler documentation (MIT,
`tools/completion/README.md`). No llama.cpp code was copied or vendored, so
there is no `docs/LICENSE_REVIEW.md` entry to add. The RNG uses the
public-domain xoshiro256** and SplitMix64 algorithms, implemented from their
published descriptions.

## Open questions

- Should the core API expose `SampleStatus::NoViableCandidates` as its own
  error, or fold it into `backend_error`?
- Grammar engine choice (GBNF port vs. an independent implementation) needs a
  license decision before work starts.
