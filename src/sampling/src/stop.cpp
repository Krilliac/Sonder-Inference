#include "sonder/sampling/stop.hpp"

#include <algorithm>
#include <utility>

namespace sonder::inference::sampling {

StopSequenceMatcher::StopSequenceMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    // Empty stop strings would match everywhere; config validation rejects
    // them, and the matcher ignores them defensively.
    stops_.erase(std::remove_if(stops_.begin(), stops_.end(), [](const std::string& s) { return s.empty(); }),
                 stops_.end());
}

StopCheck StopSequenceMatcher::feed(std::string_view piece) {
    StopCheck out;
    if (stopped_) {
        out.stopped = true;
        return out;
    }
    pending_.append(piece);

    // 1) Earliest complete match (ties -> longest stop).
    std::size_t best_pos = std::string::npos;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < stops_.size(); ++i) {
        const std::size_t pos = pending_.find(stops_[i]);
        if (pos == std::string::npos) {
            continue;
        }
        if (pos < best_pos || (pos == best_pos && stops_[i].size() > stops_[best_idx].size())) {
            best_pos = pos;
            best_idx = i;
        }
    }
    if (best_pos != std::string::npos) {
        out.stopped = true;
        out.stop_index = best_idx;
        out.emit = pending_.substr(0, best_pos);
        pending_.clear();
        stopped_ = true;
        return out;
    }

    // 2) Hold back the longest suffix that is a proper prefix of some stop.
    std::size_t hold = 0;
    for (const auto& s : stops_) {
        const std::size_t max_len = std::min(s.size() - 1, pending_.size());
        for (std::size_t len = max_len; len > hold; --len) {
            if (pending_.compare(pending_.size() - len, len, s, 0, len) == 0) {
                hold = len;
                break;
            }
        }
    }
    out.emit = pending_.substr(0, pending_.size() - hold);
    pending_.erase(0, pending_.size() - hold);
    return out;
}

std::string StopSequenceMatcher::flush() {
    std::string out = std::move(pending_);
    pending_.clear();
    return out;
}

void StopSequenceMatcher::reset() {
    pending_.clear();
    stopped_ = false;
}

}  // namespace sonder::inference::sampling
