# Sampling

Composable sampler chain that turns one step of logits into a token. It covers
logit bias, a grammar/constraint hook, repetition/frequency/presence penalties,
top-k, locally typical, top-p, min-p, temperature, greedy, and seeded
distribution sampling. Text-level stop sequences and stop tokens are handled
here too. Speculative decoding policy and a real grammar engine are future work.

Namespace: `sonder::inference::sampling`. Headers:
`src/sampling/include/sonder/sampling/*.hpp` (umbrella: `sonder/sampling/sampling.hpp`).
The module builds into `sonder_inference` through the root `SONDER_MODULE_DIRS`
hook and defines `SONDER_HAS_SAMPLER_CHAIN=1`. Its tests run in CTest as
`sonder.sampling.*`.

```cpp
using namespace sonder::inference::sampling;
SamplerConfig cfg;            // llama.cpp defaults
cfg.seed = 42;
SamplerChain chain = build_chain(cfg, /*constraint=*/nullptr, vocab_size);
SampleResult r = chain.sample(logits);   // std::span<const float>, index = token id
if (r.ok()) chain.accept(r.token);       // commit: updates penalties + constraint
```

To go from the core API config, use `make_chain(const sonder::inference::SamplingConfig&)` in
`core_bridge.hpp`. It returns `Result<SamplerChain>`.

## Semantics and ordering

The chain is modelled on llama.cpp's sampler semantics, as documented in
[tools/completion/README.md → Sampling params / Generation Flags](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md)
(MIT). The code here is an independent implementation. No llama.cpp source
was copied.

Pipeline for each step:

```text
logits ─► NaN→-inf ─► [constraint] ─► [logit_bias] ─► stage_order… ─► selector
```

The default `stage_order` is llama.cpp's default `--samplers`
(`penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`)
restricted to the stages implemented here:

`penalties → top_k → typ_p → top_p → min_p → temperature`

Because temperature comes last, the truncation stages operate on untempered
probabilities, as in llama.cpp. `stage_order` can be reordered or subset
(no duplicates allowed). Names parse from llama.cpp spellings via `parse_stage()`.

| Field | Default | Disabled when | Semantics |
| --- | --- | --- | --- |
| `logit_bias` | none | empty | Adds `bias` to each listed token's logit. Repeated entries accumulate. `-inf` bans a token. `+inf`/NaN are rejected. |
| `penalty_last_n` | 64 | 0 | Window of accepted tokens used for penalties. `-1` = unbounded. |
| `repeat_penalty` | 1.0 | 1.0 | Applies when a token is in the window: `logit <= 0 ? logit*r : logit/r`. |
| `frequency_penalty` | 0.0 | 0.0 | `logit -= count * f`. |
| `presence_penalty` | 0.0 | 0.0 | `logit -= p` if count > 0. |
| `top_k` | 40 | 0 | Keeps the k highest logits. `k >= n` keeps everything. |
| `typical_p` | 1.0 | ≥ 1 | Ranks tokens by `|-log p - H|` ascending and keeps the smallest set whose mass is **>** p. This can drop the argmax. |
| `top_p` | 0.95 | ≥ 1 | Keeps the smallest prefix (by probability) whose mass is **≥** p. `p = 0` keeps `min_keep`. |
| `min_p` | 0.05 | ≤ 0 | Keeps tokens with `p ≥ min_p · p_max`. |
| `min_keep` | 0 | — | Minimum number of survivors for top_p/min_p/typical_p. 0 is treated as 1. |
| `temperature` | 0.8 | 1.0 | Divides logits by t. `t = 0` keeps only the argmax of the remaining set (llama.cpp `temp_ext`). Negative values are rejected. |
| `greedy` | false | — | Selector = argmax after bias, constraint and penalties. Truncation and temperature stages are not built. |
| `seed` | random | — | Seeds the portable RNG. `nullopt` draws an entropy seed when the chain is built. |
| `stop_tokens` | none | — | `SamplerChain::is_stop_token()`. |
| `stop_sequences` | none | — | `StopSequenceMatcher` (text level, see below). |

Determinism details:

- **Tie-breaking.** Candidates are compared in canonical order: logit
  descending, then token id ascending. Greedy and `t = 0` therefore pick the
  lowest id among tied maxima, and results do not depend on backend output order.
- **Portable RNG.** The generator is xoshiro256** seeded by SplitMix64, and
  draws are the top 53 bits mapped to `[0,1)`. It does not use
  `std::mt19937` or `std::*_distribution`, whose output varies across standard
  libraries. The same seed produces the same tokens on MSVC, libstdc++ and
  libc++. A golden sequence for seed 42 is pinned in the tests and was
  cross-checked against an independent Python reference.
- **Selector.** The selector draws `u ∈ [0,1)`, walks the canonical order
  accumulating p (in double precision), and returns the first token with
  `u < cum`.

Edge cases:

- NaN logits are treated as `-inf`.
- A `+inf` logit takes all the probability mass. If several tokens are
  `+inf`, the mass is split evenly among them.
- An empty logits span returns `SampleStatus::EmptyLogits`.
- If every candidate is `-inf` (for example, over-constrained or fully
  banned), the result is `SampleStatus::NoViableCandidates` and
  `token = kInvalidToken`. The call never throws and never returns a banned token.

Not implemented yet (llama.cpp has them): DRY, top-nσ, XTC, mirostat v1/v2,
dynamic temperature, adaptive-p. Each one fits as a new `Sampler` plus a
`StageKind`.

## Validation

`validate(config, vocab_size?)` reports **every** problem as a `{field, message}`
pair, and `to_string()` renders them as
`sampling config invalid: top_p: must be a finite value in [0, 1] (got 1.5); ...`.
When `vocab_size` is known, token ids are range-checked. `build_chain()` throws
`std::invalid_argument` with that message. `make_chain()`/`to_status()` return
`ErrorCode::invalid_argument` instead.

## Constraint hook (stub)

`Constraint` exposes `apply` (mask to `-inf`), `accept`, `reset` and
`is_complete`. It runs before every other stage, so disallowed tokens never
get probability mass. `accept_prompt()` feeds penalties only, which mirrors
llama.cpp accepting prompt tokens with the grammar off. The only shipped
implementation is `TokenAllowlistConstraint`. A GBNF/JSON-schema engine is
future work.

## Stop sequences

`StopSequenceMatcher::feed(piece)` works on bytes (UTF-8 safe). It holds back
any suffix that could still begin a stop string, so clients never see a
partial stop. On a match it emits the text before the stop and drops the stop
itself. If several stops match, the earliest one wins; for matches at the same
position, the longest wins. `flush()` releases held text when generation ends.

## Threading and hot path

Use one chain per sequence. Chains are not thread-safe. Copying a chain clones
its stages and state, which is how sessions fork. The constraint is shared by
`shared_ptr`, so install a fresh one for each fork. The candidate buffer is
reused between steps. Only typical-p allocates scratch space on each call.
