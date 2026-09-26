#include "sonder/sampling/constraint.hpp"

#include <utility>

namespace sonder::inference::sampling {

TokenAllowlistConstraint::TokenAllowlistConstraint(std::vector<TokenId> allowed)
    : allowed_(allowed.begin(), allowed.end()) {}

void TokenAllowlistConstraint::apply(Candidates& candidates) {
    for (auto& t : candidates.data) {
        if (allowed_.find(t.id) == allowed_.end()) {
            t.logit = kNegInf;
        }
    }
    candidates.sorted = false;
}

}  // namespace sonder::inference::sampling
