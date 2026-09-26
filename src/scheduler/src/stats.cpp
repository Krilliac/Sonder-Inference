#include "sonder/inference/scheduler/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sonder::inference::scheduler {

namespace {

TimeUs percentile(const std::vector<TimeUs>& sorted, double p) {
    // Nearest-rank percentile on a sorted, non-empty vector.
    const auto n = sorted.size();
    auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(n)));
    rank = std::clamp<std::size_t>(rank, 1, n);
    return sorted[rank - 1];
}

}  // namespace

LatencySummary LatencySummary::from(std::vector<TimeUs> samples) {
    LatencySummary s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.count = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    const long double sum = std::accumulate(samples.begin(), samples.end(), 0.0L,
                                            [](long double a, TimeUs b) { return a + b; });
    s.mean = static_cast<double>(sum / static_cast<long double>(samples.size()));
    s.p50 = percentile(samples, 0.50);
    s.p95 = percentile(samples, 0.95);
    s.p99 = percentile(samples, 0.99);
    return s;
}

}  // namespace sonder::inference::scheduler
