// Arctic-style bounded Suffix Automaton (SAM) speculative drafter.
//
// Stdlib-only C++20, zero extra VRAM. Maintains:
//   1. An active-request SAM over the current prompt + generated output.
//   2. A bounded cross-request session ring buffer SAM (~64K-256K tokens) with
//      explicit sentinel separators so drafts never bleed across requests.
//
// Supports both standalone `SpeculativeType::kSuffix` drafting and hybrid
// `draft-mtp` + `suffix` escalation (bursting 6..16 tokens on high-confidence
// verbatim matches >= 6 tokens instead of capping at MTP's k=3).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

struct SuffixMatchResult {
    std::vector<Token> draft;
    std::size_t match_len = 0;
    std::uint32_t match_count = 0;
    float confidence = 0.0F;
    bool from_session_ring = false;
};

// Bounded online Suffix Automaton over a contiguous token buffer.
// When the buffer reaches `max_tokens`, compacts to the most recent half and
// rebuilds in O(N) time (amortized O(1) per token, strictly bounded RAM).
class BoundedSuffixAutomaton {
public:
    static constexpr Token kSeparator = -1;
    static constexpr std::size_t kMinCapacity = 64;
    static constexpr std::size_t kMaxCapacity = 262144;
    static constexpr std::size_t kDefaultCapacity = 131072;
    static constexpr std::size_t kMaxQueryLookback = 64;
    static constexpr int kMaxLinkRefresh = 16;

    explicit BoundedSuffixAutomaton(std::size_t max_tokens = kDefaultCapacity)
        : max_tokens_(std::clamp(max_tokens, kMinCapacity, kMaxCapacity)) {
        Reset();
    }

    void Reset() {
        tokens_.clear();
        ResetAutomatonOnly();
    }

    void Append(Token tok) {
        if (tok < 0) {
            AppendSeparator();
            return;
        }
        if (tokens_.size() >= max_tokens_) {
            Compact();
        }
        const auto pos = static_cast<std::int32_t>(tokens_.size());
        tokens_.push_back(tok);
        ExtendSam(tok, pos);
    }

    void AppendSpan(std::span<const Token> toks) {
        for (Token t : toks) {
            Append(t);
        }
    }

    // Breaks suffix continuity between distinct requests in a shared ring.
    void AppendSeparator() {
        if (tokens_.empty() || tokens_.back() == kSeparator) {
            last_state_ = 0;
            return;
        }
        if (tokens_.size() >= max_tokens_) {
            Compact();
        }
        tokens_.push_back(kSeparator);
        last_state_ = 0;
    }

    // Finds the longest suffix of `tail` present in this automaton that has at
    // least one valid non-separator continuation token strictly before
    // `tokens_.size() - exclude_tail_tokens`.
    [[nodiscard]] SuffixMatchResult MatchLongestSuffix(std::span<const Token> tail, std::size_t min_match_len,
                                                       std::size_t max_draft, std::size_t exclude_tail_tokens = 0,
                                                       double draft_factor = 2.0) const {
        SuffixMatchResult res;
        if (tail.empty() || max_draft == 0 || tokens_.size() <= exclude_tail_tokens + 1) {
            return res;
        }
        const std::size_t eff_min = std::max<std::size_t>(1, min_match_len);
        if (tail.size() < eff_min) {
            return res;
        }

        const std::size_t window_len = std::min<std::size_t>(tail.size(), kMaxQueryLookback);
        const std::size_t window_start = tail.size() - window_len;

        std::int32_t st = 0;
        std::size_t matched = 0;
        for (std::size_t i = window_start; i < tail.size(); ++i) {
            const Token tok = tail[i];
            if (tok < 0) {
                st = 0;
                matched = 0;
                continue;
            }
            while (st != 0 && FindEdge(st, tok) < 0) {
                st = states_[static_cast<std::size_t>(st)].link;
                matched = static_cast<std::size_t>(states_[static_cast<std::size_t>(st)].len);
            }
            const std::int32_t next = FindEdge(st, tok);
            if (next >= 0) {
                st = next;
                ++matched;
            } else {
                st = 0;
                matched = 0;
            }
        }

        if (matched < eff_min || st <= 0) {
            return res;
        }

        const std::size_t usable_limit = tokens_.size() - exclude_tail_tokens;
        std::int32_t u = st;
        std::size_t u_match_len = matched;
        std::int32_t chosen_pos = -1;
        std::uint32_t chosen_hits = 0;

        while (u > 0 && u_match_len >= eff_min) {
            const State& s = states_[static_cast<std::size_t>(u)];
            const auto candidate_valid = [&](std::int32_t p) -> bool {
                if (p < 0) return false;
                const auto up = static_cast<std::size_t>(p);
                if (up + 1 >= usable_limit) return false;
                if (tokens_[up + 1] == kSeparator) return false;
                if (up + 1 < u_match_len) return false;
                return true;
            };

            if (candidate_valid(s.last_pos)) {
                chosen_pos = s.last_pos;
                chosen_hits = s.hits;
                break;
            }
            if (candidate_valid(s.prev_pos)) {
                chosen_pos = s.prev_pos;
                chosen_hits = s.hits;
                break;
            }
            u = s.link;
            if (u <= 0) break;
            u_match_len = std::min<std::size_t>(u_match_len, static_cast<std::size_t>(states_[static_cast<std::size_t>(u)].len));
        }

        if (chosen_pos < 0 || u_match_len < eff_min) {
            return res;
        }

        // If the match saturated our 64-token SAM query window, walk backwards
        // directly to measure the true full suffix match length.
        std::size_t full_match = u_match_len;
        if (full_match == window_len && tail.size() > window_len) {
            const auto up = static_cast<std::size_t>(chosen_pos);
            while (full_match < tail.size() && full_match <= up) {
                const Token hist_tok = tokens_[up - full_match];
                if (hist_tok == kSeparator || hist_tok != tail[tail.size() - 1 - full_match]) {
                    break;
                }
                ++full_match;
            }
        }

        const std::size_t start = static_cast<std::size_t>(chosen_pos) + 1;
        std::size_t avail = 0;
        while (start + avail < usable_limit && tokens_[start + avail] != kSeparator) {
            ++avail;
            if (avail >= max_draft) break;
        }
        if (avail == 0) {
            return res;
        }

        const double scaled = std::max(1.0, std::floor(static_cast<double>(full_match) * draft_factor));
        const auto dynamic_cap = static_cast<std::size_t>(
            std::min<double>(scaled, static_cast<double>(max_draft)));
        const std::size_t count = std::min<std::size_t>({avail, max_draft, std::max<std::size_t>(1, dynamic_cap)});

        res.draft.assign(tokens_.begin() + static_cast<std::ptrdiff_t>(start),
                         tokens_.begin() + static_cast<std::ptrdiff_t>(start + count));
        res.match_len = full_match;
        res.match_count = std::max<std::uint32_t>(1u, chosen_hits);
        res.confidence = ComputeConfidence(full_match, res.match_count, false);
        return res;
    }

    [[nodiscard]] static float ComputeConfidence(std::size_t match_len, std::uint32_t hits,
                                                 bool from_ring) noexcept {
        if (match_len == 0) return 0.0F;
        const double len_score = 1.0 - std::exp(-0.28 * static_cast<double>(match_len));
        const double freq_bonus =
            hits > 1u ? std::min(0.15, 0.05 * std::log2(static_cast<double>(hits))) : 0.0;
        const double ring_penalty = from_ring ? 0.04 : 0.0;
        const double score = std::clamp(len_score + freq_bonus - ring_penalty, 0.0, 1.0);
        return static_cast<float>(score);
    }

    [[nodiscard]] std::size_t size() const noexcept { return tokens_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return max_tokens_; }
    [[nodiscard]] std::uint64_t compactions() const noexcept { return compactions_; }

private:
    struct State {
        std::int32_t len = 0;
        std::int32_t link = -1;
        std::int32_t last_pos = -1;
        std::int32_t prev_pos = -1;
        std::int32_t head_edge = -1;
        std::uint32_t hits = 0;
    };

    struct Edge {
        Token key = 0;
        std::int32_t to = -1;
        std::int32_t next = -1;
    };

    void ResetAutomatonOnly() {
        states_.clear();
        edges_.clear();
        root_edges_.clear();
        states_.push_back(State{0, -1, -1, -1, -1, 0});
        last_state_ = 0;
    }

    [[nodiscard]] std::int32_t FindEdge(std::int32_t state, Token key) const noexcept {
        if (state == 0) {
            const auto it = root_edges_.find(key);
            return it == root_edges_.end() ? -1 : it->second;
        }
        std::int32_t e = states_[static_cast<std::size_t>(state)].head_edge;
        while (e >= 0) {
            const Edge& edge = edges_[static_cast<std::size_t>(e)];
            if (edge.key == key) return edge.to;
            e = edge.next;
        }
        return -1;
    }

    void SetEdge(std::int32_t state, Token key, std::int32_t to) {
        if (state == 0) {
            root_edges_[key] = to;
            return;
        }
        std::int32_t e = states_[static_cast<std::size_t>(state)].head_edge;
        while (e >= 0) {
            Edge& edge = edges_[static_cast<std::size_t>(e)];
            if (edge.key == key) {
                edge.to = to;
                return;
            }
            e = edge.next;
        }
        const auto new_idx = static_cast<std::int32_t>(edges_.size());
        edges_.push_back(Edge{key, to, states_[static_cast<std::size_t>(state)].head_edge});
        states_[static_cast<std::size_t>(state)].head_edge = new_idx;
    }

    void CopyEdges(std::int32_t from_state, std::int32_t to_state) {
        std::int32_t e = states_[static_cast<std::size_t>(from_state)].head_edge;
        while (e >= 0) {
            const Edge edge = edges_[static_cast<std::size_t>(e)];
            const auto new_idx = static_cast<std::int32_t>(edges_.size());
            edges_.push_back(Edge{edge.key, edge.to, states_[static_cast<std::size_t>(to_state)].head_edge});
            states_[static_cast<std::size_t>(to_state)].head_edge = new_idx;
            e = edge.next;
        }
    }

    void ExtendSam(Token tok, std::int32_t pos) {
        const auto cur = static_cast<std::int32_t>(states_.size());
        const std::int32_t prev_len = states_[static_cast<std::size_t>(last_state_)].len;
        states_.push_back(State{prev_len + 1, 0, pos, -1, -1, 1});

        std::int32_t p = last_state_;
        while (p != -1 && FindEdge(p, tok) < 0) {
            SetEdge(p, tok, cur);
            p = states_[static_cast<std::size_t>(p)].link;
        }

        if (p == -1) {
            states_[static_cast<std::size_t>(cur)].link = 0;
        } else {
            const std::int32_t q = FindEdge(p, tok);
            if (states_[static_cast<std::size_t>(p)].len + 1 == states_[static_cast<std::size_t>(q)].len) {
                states_[static_cast<std::size_t>(cur)].link = q;
            } else {
                const auto clone = static_cast<std::int32_t>(states_.size());
                const State q_copy = states_[static_cast<std::size_t>(q)];
                states_.push_back(State{
                    states_[static_cast<std::size_t>(p)].len + 1,
                    q_copy.link,
                    q_copy.last_pos,
                    q_copy.prev_pos,
                    -1,
                    q_copy.hits,
                });
                CopyEdges(q, clone);
                while (p != -1 && FindEdge(p, tok) == q) {
                    SetEdge(p, tok, clone);
                    p = states_[static_cast<std::size_t>(p)].link;
                }
                states_[static_cast<std::size_t>(q)].link = clone;
                states_[static_cast<std::size_t>(cur)].link = clone;
            }
        }

        // Refresh recency and occurrence counts along a bounded number of suffix
        // links (strictly O(1) worst-case per appended token).
        std::int32_t u = states_[static_cast<std::size_t>(cur)].link;
        for (int hops = 0; u > 0 && hops < kMaxLinkRefresh; ++hops) {
            State& su = states_[static_cast<std::size_t>(u)];
            if (su.last_pos == pos) break;
            su.prev_pos = su.last_pos;
            su.last_pos = pos;
            ++su.hits;
            u = su.link;
        }

        last_state_ = cur;
    }

    void Compact() {
        const std::size_t keep = std::max<std::size_t>(1, max_tokens_ / 2);
        const std::size_t drop = tokens_.size() > keep ? (tokens_.size() - keep) : 0;
        if (drop > 0) {
            tokens_.erase(tokens_.begin(), tokens_.begin() + static_cast<std::ptrdiff_t>(drop));
        }
        // Drop leading separators after trimming.
        while (!tokens_.empty() && tokens_.front() == kSeparator) {
            tokens_.erase(tokens_.begin());
        }
        ResetAutomatonOnly();
        for (std::size_t i = 0; i < tokens_.size(); ++i) {
            const Token tok = tokens_[i];
            if (tok == kSeparator) {
                last_state_ = 0;
            } else {
                ExtendSam(tok, static_cast<std::int32_t>(i));
            }
        }
        ++compactions_;
    }

    std::size_t max_tokens_ = kDefaultCapacity;
    std::vector<Token> tokens_;
    std::vector<State> states_;
    std::vector<Edge> edges_;
    std::unordered_map<Token, std::int32_t> root_edges_;
    std::int32_t last_state_ = 0;
    std::uint64_t compactions_ = 0;
};

// High-level Arctic-style Suffix Matcher combining per-request prompt+output
// indexing with a persistent cross-request session ring buffer and adaptive
// speculation factor / hybrid MTP burst escalation.
class SuffixMatcher {
public:
    static constexpr std::size_t kDefaultMinMatchLen = 3;
    static constexpr std::size_t kHybridMtpMinMatchLen = 6;
    static constexpr std::size_t kHybridMtpMinBurstDraft = 6;
    static constexpr std::size_t kHybridMtpMaxBurstDraft = 16;
    static constexpr float kHybridMtpMinConfidence = 0.75F;

    explicit SuffixMatcher(std::size_t session_ring_capacity = BoundedSuffixAutomaton::kDefaultCapacity,
                           std::size_t active_capacity = BoundedSuffixAutomaton::kDefaultCapacity)
        : active_(active_capacity), session_ring_(session_ring_capacity) {}

    void ResetAll() {
        active_.Reset();
        session_ring_.Reset();
        current_request_.clear();
        ResetAdaptive();
    }

    void BeginRequest(std::span<const Token> prompt) {
        active_.Reset();
        current_request_.assign(prompt.begin(), prompt.end());
        active_.AppendSpan(prompt);
        ResetAdaptive();
    }

    // Commits a verified token into the active request SAM.
    void CommitToken(Token tok) {
        if (tok < 0) return;
        current_request_.push_back(tok);
        active_.Append(tok);
    }

    // Commits the completed request's prompt + output into the cross-request
    // session ring buffer separated by a sentinel so future requests can match
    // verbatim code/JSON/templates without cross-request boundary bleed.
    void EndRequest() {
        if (!current_request_.empty()) {
            session_ring_.AppendSeparator();
            session_ring_.AppendSpan(current_request_);
            session_ring_.AppendSeparator();
            current_request_.clear();
        }
    }

    // Directly ingests a completed sequence into the cross-request session ring.
    void IngestSessionSequence(std::span<const Token> tokens) {
        if (tokens.empty()) return;
        session_ring_.AppendSeparator();
        session_ring_.AppendSpan(tokens);
        session_ring_.AppendSeparator();
    }

    // Proposes a confidence-scored dynamic draft sequence (`1..max_draft`)
    // matching the longest suffix of `tail` (or `current_request_` + `id_last`
    // via DraftWithNextToken).
    [[nodiscard]] SuffixMatchResult Draft(std::span<const Token> tail, std::size_t max_draft,
                                          std::size_t min_match_len = kDefaultMinMatchLen) const {
        if (max_draft == 0 || tail.empty()) return {};
        const std::size_t capped_max = std::min<std::size_t>(max_draft, static_cast<std::size_t>(kMaxSpecDraftTokens));
        if (cooldown_remaining_ > 0) {
            return {};
        }

        const std::size_t eff_min = std::max(min_match_len, adaptive_min_match_);
        SuffixMatchResult local_match =
            active_.MatchLongestSuffix(tail, eff_min, capped_max, /*exclude_tail_tokens=*/0, draft_factor_);
        SuffixMatchResult ring_match =
            session_ring_.MatchLongestSuffix(tail, eff_min, capped_max, /*exclude_tail_tokens=*/0, draft_factor_);
        if (!ring_match.draft.empty()) {
            ring_match.from_session_ring = true;
            ring_match.confidence =
                BoundedSuffixAutomaton::ComputeConfidence(ring_match.match_len, ring_match.match_count, true);
        }

        if (local_match.draft.empty()) return ring_match;
        if (ring_match.draft.empty()) return local_match;
        if (ring_match.match_len > local_match.match_len ||
            (ring_match.match_len == local_match.match_len && ring_match.confidence > local_match.confidence)) {
            return ring_match;
        }
        return local_match;
    }

    // Convenience overload that queries using `current_request_` followed by
    // `id_last` (without mutating `current_request_` or `active_`).
    [[nodiscard]] SuffixMatchResult DraftForCurrent(Token id_last, std::size_t max_draft,
                                                    std::size_t min_match_len = kDefaultMinMatchLen) {
        query_scratch_.clear();
        const std::size_t keep = std::min<std::size_t>(current_request_.size(), BoundedSuffixAutomaton::kMaxQueryLookback - 1);
        const std::size_t start = current_request_.size() - keep;
        query_scratch_.insert(query_scratch_.end(),
                              current_request_.begin() + static_cast<std::ptrdiff_t>(start), current_request_.end());
        query_scratch_.push_back(id_last);
        return Draft(query_scratch_, max_draft, min_match_len);
    }

    // Hybrid `draft-mtp` + `suffix` escalation:
    // If a high-confidence exact suffix match of length >= 6 is found with at
    // least 6 continuation tokens available (capped at min(max_k, 16)), returns
    // the burst draft (6..16 tokens); otherwise returns an empty draft so the
    // caller falls back cleanly to standard `draft-mtp`.
    [[nodiscard]] SuffixMatchResult TryHybridMtpBurst(std::span<const Token> tail, std::size_t max_k) const {
        if (max_k < kHybridMtpMinBurstDraft || tail.size() < kHybridMtpMinMatchLen) {
            return {};
        }
        const std::size_t burst_cap = std::min<std::size_t>(max_k, kHybridMtpMaxBurstDraft);
        SuffixMatchResult match = Draft(tail, burst_cap, kHybridMtpMinMatchLen);
        if (match.match_len < kHybridMtpMinMatchLen || match.confidence < kHybridMtpMinConfidence ||
            match.draft.size() < kHybridMtpMinBurstDraft) {
            return {};
        }
        if (match.draft.size() > burst_cap) {
            match.draft.resize(burst_cap);
        }
        return match;
    }

    [[nodiscard]] SuffixMatchResult TryHybridMtpBurstForCurrent(Token id_last, std::size_t max_k) {
        if (max_k < kHybridMtpMinBurstDraft || current_request_.size() + 1 < kHybridMtpMinMatchLen) {
            return {};
        }
        query_scratch_.clear();
        const std::size_t keep = std::min<std::size_t>(current_request_.size(), BoundedSuffixAutomaton::kMaxQueryLookback - 1);
        const std::size_t start = current_request_.size() - keep;
        query_scratch_.insert(query_scratch_.end(),
                              current_request_.begin() + static_cast<std::ptrdiff_t>(start), current_request_.end());
        query_scratch_.push_back(id_last);
        return TryHybridMtpBurst(query_scratch_, max_k);
    }

    void NoteSkippedStep() noexcept {
        if (cooldown_remaining_ > 0) {
            --cooldown_remaining_;
        }
    }

    void Observe(std::size_t drafted, std::size_t accepted) noexcept {
        if (drafted == 0) return;
        const std::size_t clamped_acc = std::min(accepted, drafted);
        const double rate = static_cast<double>(clamped_acc) / static_cast<double>(drafted);
        constexpr double kAlpha = 0.25;
        acceptance_ema_ = (steps_observed_ == 0) ? rate : (1.0 - kAlpha) * acceptance_ema_ + kAlpha * rate;
        ++steps_observed_;

        if (clamped_acc == drafted) {
            draft_factor_ = std::min(3.0, draft_factor_ + 0.35);
            adaptive_min_match_ = kDefaultMinMatchLen;
            low_streak_ = 0;
        } else if (acceptance_ema_ >= 0.50) {
            draft_factor_ = std::min(2.75, draft_factor_ + 0.15);
            adaptive_min_match_ = kDefaultMinMatchLen;
            low_streak_ = 0;
        } else if (rate < 0.20) {
            draft_factor_ = std::max(1.0, draft_factor_ * 0.75);
            ++low_streak_;
            if (low_streak_ >= 4 && acceptance_ema_ < 0.20) {
                cooldown_remaining_ = std::min<std::int32_t>(cooldown_period_, 16);
                cooldown_period_ = std::min<std::int32_t>(cooldown_period_ * 2, 16);
                adaptive_min_match_ = 4;
                low_streak_ = 0;
            }
        } else {
            low_streak_ = 0;
        }
    }

    [[nodiscard]] std::size_t active_size() const noexcept { return active_.size(); }
    [[nodiscard]] std::size_t session_ring_size() const noexcept { return session_ring_.size(); }
    [[nodiscard]] std::size_t session_ring_capacity() const noexcept { return session_ring_.capacity(); }
    [[nodiscard]] std::uint64_t session_ring_compactions() const noexcept { return session_ring_.compactions(); }
    [[nodiscard]] double acceptance_ema() const noexcept { return acceptance_ema_; }
    [[nodiscard]] bool paused() const noexcept { return cooldown_remaining_ > 0; }

private:
    void ResetAdaptive() noexcept {
        acceptance_ema_ = 0.5;
        draft_factor_ = 2.0;
        adaptive_min_match_ = kDefaultMinMatchLen;
        steps_observed_ = 0;
        low_streak_ = 0;
        cooldown_remaining_ = 0;
        cooldown_period_ = 4;
    }

    BoundedSuffixAutomaton active_;
    BoundedSuffixAutomaton session_ring_;
    std::vector<Token> current_request_;
    std::vector<Token> query_scratch_;

    double acceptance_ema_ = 0.5;
    double draft_factor_ = 2.0;
    std::size_t adaptive_min_match_ = kDefaultMinMatchLen;
    std::uint32_t steps_observed_ = 0;
    std::int32_t low_streak_ = 0;
    std::int32_t cooldown_remaining_ = 0;
    std::int32_t cooldown_period_ = 4;
};

}  // namespace sonder::backends::llamacpp
