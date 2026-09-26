#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace sonder::inference::sampling {

/// Vocabulary index of a token. Negative values are never valid tokens.
using TokenId = std::int32_t;

inline constexpr TokenId kInvalidToken = -1;
inline constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/// One candidate token. `logit` is the working (possibly transformed) logit;
/// `p` is only meaningful after a stage that normalises (see softmax()).
struct TokenData {
    TokenId id = kInvalidToken;
    float logit = 0.0f;
    float p = 0.0f;
};

/// Working set of candidates passed through the sampler chain.
///
/// Invariants maintained by the built-in stages:
///  - `data` never contains NaN logits (they are mapped to -inf on ingest);
///  - when `sorted` is true, `data` is ordered by logit descending, ties by id
///    ascending (the "canonical order"), which makes every stage deterministic
///    regardless of the order the backend produced logits in.
struct Candidates {
    std::vector<TokenData> data;
    bool sorted = false;

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
};

/// Sort `c` into canonical order (logit desc, id asc). No-op if already sorted.
void sort_canonical(Candidates& c);

/// Compute `p` for every candidate from its logit with a numerically stable
/// softmax. -inf logits get p = 0. If any logit is +inf, the +inf candidates
/// share probability mass uniformly. If every logit is -inf, all p are 0 and
/// the function returns false (no viable candidate); otherwise true.
bool softmax(Candidates& c);

/// Number of candidates with a finite-or-+inf logit (i.e. p could be > 0).
[[nodiscard]] std::size_t viable_count(const Candidates& c) noexcept;

}  // namespace sonder::inference::sampling
