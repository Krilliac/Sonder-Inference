#include "sonder/sampling/types.hpp"

#include <algorithm>
#include <cmath>

namespace sonder::inference::sampling {

namespace {
bool canonical_less(const TokenData& a, const TokenData& b) noexcept {
    if (a.logit != b.logit) {
        return a.logit > b.logit;
    }
    return a.id < b.id;
}
}  // namespace

void sort_canonical(Candidates& c) {
    if (c.sorted) {
        return;
    }
    std::sort(c.data.begin(), c.data.end(), canonical_less);
    c.sorted = true;
}

bool softmax(Candidates& c) {
    if (c.data.empty()) {
        return false;
    }
    float max_logit = kNegInf;
    for (const auto& t : c.data) {
        max_logit = std::max(max_logit, t.logit);
    }
    if (max_logit == kNegInf) {
        for (auto& t : c.data) {
            t.p = 0.0f;
        }
        return false;
    }
    if (std::isinf(max_logit)) {  // +inf: uniform over the +inf candidates
        std::size_t n_inf = 0;
        for (const auto& t : c.data) {
            n_inf += (t.logit == max_logit) ? 1U : 0U;
        }
        const float share = 1.0f / static_cast<float>(n_inf);
        for (auto& t : c.data) {
            t.p = (t.logit == max_logit) ? share : 0.0f;
        }
        return true;
    }
    double sum = 0.0;
    for (auto& t : c.data) {
        const double e = std::exp(static_cast<double>(t.logit) - static_cast<double>(max_logit));
        t.p = static_cast<float>(e);
        sum += e;
    }
    for (auto& t : c.data) {
        t.p = static_cast<float>(static_cast<double>(t.p) / sum);
    }
    return true;
}

std::size_t viable_count(const Candidates& c) noexcept {
    std::size_t n = 0;
    for (const auto& t : c.data) {
        n += (t.logit != kNegInf) ? 1U : 0U;
    }
    return n;
}

}  // namespace sonder::inference::sampling
