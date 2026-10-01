// Internal helpers of the llama.cpp module adapter (not installed; exposed
// for unit tests only).
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference::llamacpp_detail {

// UTF-8 std::string <-> std::filesystem::path without the C++20-deprecated
// u8path() and without std::u8string leaking into callers.
inline std::filesystem::path PathFromUtf8(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}
inline std::string PathToUtf8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// Maps SamplingConfig onto the wrapper's sampler parameters.
inline sonder::backends::llamacpp::SamplingParams ToLlamaSampling(const SamplingConfig& c) {
    namespace lc = sonder::backends::llamacpp;
    lc::SamplingParams p;
    p.temperature = c.temperature;
    p.top_k = c.top_k;
    p.top_p = c.top_p;
    p.min_p = c.min_p;
    p.repeat_penalty = c.repeat_penalty;
    p.repeat_last_n = c.repeat_last_n;
    p.presence_penalty = c.presence_penalty;
    p.frequency_penalty = c.frequency_penalty;
    p.typical_p = c.typical_p;
    p.logit_bias.reserve(c.logit_bias.size());
    for (const auto& b : c.logit_bias) p.logit_bias.emplace_back(b.token, b.bias);
    if (c.seed.has_value()) {
        // llama.cpp seeds are 32-bit and 0xFFFFFFFF means "random": fold the
        // 64-bit seed and avoid the sentinel so explicit seeds stay reproducible.
        auto folded = static_cast<std::uint32_t>(*c.seed ^ (*c.seed >> 32));
        p.seed = folded == 0xFFFFFFFFU ? 0xFFFFFFFEU : folded;
    } else {
        p.seed = 0xFFFFFFFFU;
    }
    return p;
}

// Holds back text that could be the start of a stop sequence and reports a
// full match. Stop text itself is never emitted (matches Ollama/OpenAI).
class StopSequenceFilter {
public:
    explicit StopSequenceFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    // Appends text; returns the text now safe to emit. Sets matched() on a hit.
    std::string Push(std::string_view text) {
        if (matched_) return {};
        held_.append(text);
        if (stops_.empty()) return std::exchange(held_, {});
        std::size_t cut = std::string::npos;
        const std::string* matched = nullptr;
        for (const auto& s : stops_) {
            const std::size_t at = held_.find(s);
            if (at != std::string::npos &&
                (at < cut || (at == cut && matched && s.size() > matched->size()))) {
                cut = at;
                matched = &s;
            }
        }
        if (cut != std::string::npos) {
            matched_ = true;
            matched_index_ = matched ? std::optional<std::string>(*matched) : std::nullopt;
            std::string out = held_.substr(0, cut);
            held_.clear();
            return out;
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        std::size_t keep = 0;
        for (const auto& s : stops_) {
            const std::size_t limit = std::min(held_.size(), s.size() - 1);
            for (std::size_t k = limit; k > keep; --k) {
                if (std::string_view(held_).substr(held_.size() - k) == std::string_view(s).substr(0, k)) {
                    keep = k;
                    break;
                }
            }
        }
        std::string out = held_.substr(0, held_.size() - keep);
        held_.erase(0, held_.size() - keep);
        return out;
    }
    std::string Flush() { return matched_ ? std::string() : std::exchange(held_, {}); }
    [[nodiscard]] bool matched() const noexcept { return matched_; }
    [[nodiscard]] const std::optional<std::string>& matched_stop() const noexcept { return matched_index_; }

private:
    std::vector<std::string> stops_;
    std::string held_;
    bool matched_ = false;
    std::optional<std::string> matched_index_;
};

}  // namespace sonder::inference::llamacpp_detail
