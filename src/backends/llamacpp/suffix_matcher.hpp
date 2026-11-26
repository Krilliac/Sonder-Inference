// Arctic-style bounded Generalized Suffix Automaton (GSAM) speculative drafter.
//
// Stdlib-only C++20, zero extra VRAM. Maintains:
//   1. An active-request GSAM over the current prompt + generated output.
//   2. A bounded cross-request session ring buffer GSAM (~64K-256K tokens) with
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
#include <limits>
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

// Optional measured-cost telemetry comparing suffix verification elapsed
// ms/committed-token against ordinary (non-speculative) decode baseline.
struct SuffixCostStats {
    double suffix_ms_per_committed_token = 0.0;
    double baseline_ms_per_committed_token = 0.0;
    std::uint32_t suffix_cost_samples = 0;
    std::uint32_t baseline_cost_samples = 0;
    std::uint32_t cost_regressions = 0;
    bool cost_paused = false;
};

// Bounded online Generalized Suffix Automaton over a contiguous token buffer.
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
        if (toks.empty()) return;
        // Fast path for huge spans (>= max_tokens_): retain only the trailing
        // `max_tokens_` window and rebuild once in O(max_tokens_) instead of
        // triggering repeated compactions over discarded tokens.
        if (toks.size() >= max_tokens_) {
            const bool had_tokens = !tokens_.empty();
            const std::size_t start = toks.size() - max_tokens_;
            tokens_.clear();
            tokens_.reserve(max_tokens_);
            for (std::size_t i = start; i < toks.size(); ++i) {
                const Token t = toks[i] < 0 ? kSeparator : toks[i];
                if (t == kSeparator && (tokens_.empty() || tokens_.back() == kSeparator)) {
                    continue;
                }
                tokens_.push_back(t);
            }
            ResetAutomatonOnly();
            for (std::size_t i = 0; i < tokens_.size(); ++i) {
                if (tokens_[i] == kSeparator) {
                    last_state_ = 0;
                } else {
                    ExtendSam(tokens_[i], static_cast<std::int32_t>(i));
                }
            }
            if (had_tokens || toks.size() > max_tokens_) {
                ++compactions_;
            }
            return;
        }
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
        if (tokens_.empty() || tokens_.back() == kSeparator) {
            last_state_ = 0;
            return;
        }
        tokens_.push_back(kSeparator);
        last_state_ = 0;
    }

    // Finds the longest suffix of `tail` present in this automaton that has at
    // least `min_draft_len` valid non-separator continuation tokens strictly
    // before `tokens_.size() - exclude_tail_tokens`.
    [[nodiscard]] SuffixMatchResult MatchLongestSuffix(std::span<const Token> tail, std::size_t min_match_len,
                                                       std::size_t max_draft, std::size_t exclude_tail_tokens = 0,
                                                       double draft_factor = 2.0,
                                                       std::size_t min_draft_len = 1) const {
        SuffixMatchResult res;
        if (tail.empty() || max_draft == 0 || exclude_tail_tokens >= tokens_.size()) {
            return res;
        }
        const std::size_t usable_limit = tokens_.size() - exclude_tail_tokens;
        const std::size_t eff_min_draft = std::max<std::size_t>(1, min_draft_len);
        if (usable_limit <= eff_min_draft || max_draft < eff_min_draft) {
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

        std::int32_t u = st;
        std::size_t u_match_len = matched;
        std::int32_t chosen_pos = -1;
        std::uint32_t chosen_hits = 0;

        while (u > 0 && u_match_len >= eff_min) {
            const State& s = states_[static_cast<std::size_t>(u)];
            const auto candidate_valid = [&](std::int32_t p) -> bool {
                if (p < 0) return false;
                const auto up = static_cast<std::size_t>(p);
                if (up + 1 < u_match_len) return false;
                if (up + 1 >= usable_limit) return false;
                if (usable_limit - (up + 1) < eff_min_draft) return false;
                for (std::size_t d = 0; d < eff_min_draft; ++d) {
                    if (tokens_[up + 1 + d] == kSeparator) return false;
                }
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
            u_match_len =
                std::min<std::size_t>(u_match_len, static_cast<std::size_t>(states_[static_cast<std::size_t>(u)].len));
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
        const std::size_t max_avail_cap = std::min(max_draft, usable_limit - start);
        std::size_t avail = 0;
        while (avail < max_avail_cap && tokens_[start + avail] != kSeparator) {
            ++avail;
        }
        if (avail < eff_min_draft) {
            return res;
        }

        const double safe_factor = (!std::isfinite(draft_factor) || draft_factor <= 0.0) ? 1.0 : draft_factor;
        const double cap_d = static_cast<double>(avail);
        const double raw_scaled = static_cast<double>(full_match) * safe_factor;
        const double clamped_d =
            (!std::isfinite(raw_scaled) || raw_scaled >= cap_d) ? cap_d : std::max(1.0, std::floor(raw_scaled));
        const std::size_t dynamic_cap = static_cast<std::size_t>(clamped_d);
        const std::size_t count = std::min<std::size_t>(avail, std::max<std::size_t>(eff_min_draft, dynamic_cap));

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
    [[nodiscard]] std::size_t state_count() const noexcept { return states_.size(); }
    [[nodiscard]] std::uint64_t compactions() const noexcept { return compactions_; }

    // Verifies core GSAM invariants (strictly decreasing suffix link lengths,
    // bounded state count <= 2 * max_tokens_ + 1, valid token positions).
    [[nodiscard]] bool CheckInvariants() const noexcept {
        if (tokens_.size() > max_tokens_) return false;
        if (states_.empty() || states_[0].len != 0 || states_[0].link != -1) return false;
        if (states_.size() > 2u * max_tokens_ + 1u) return false;
        if (last_state_ < 0 || static_cast<std::size_t>(last_state_) >= states_.size()) return false;
        for (std::size_t i = 1; i < states_.size(); ++i) {
            const State& s = states_[i];
            if (s.len <= 0 || s.link < 0 || static_cast<std::size_t>(s.link) >= states_.size()) return false;
            if (states_[static_cast<std::size_t>(s.link)].len >= s.len) return false;
            if (s.last_pos < 0 || static_cast<std::size_t>(s.last_pos) >= tokens_.size()) return false;
            if (s.prev_pos >= 0 && static_cast<std::size_t>(s.prev_pos) >= tokens_.size()) return false;
        }
        return true;
    }

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

    // Refreshes recency and occurrence counts along a bounded number of suffix
    // links (strictly O(1) worst-case per appended token), preserving valid
    // non-separator continuations in `prev_pos`.
    void RefreshLinks(std::int32_t start_state, std::int32_t pos) noexcept {
        std::int32_t u = start_state;
        for (int hops = 0; u > 0 && hops < kMaxLinkRefresh; ++hops) {
            State& su = states_[static_cast<std::size_t>(u)];
            if (su.last_pos == pos) break;
            if (su.last_pos >= 0) {
                const auto prev_up = static_cast<std::size_t>(su.last_pos);
                if (prev_up + 1 < tokens_.size() && tokens_[prev_up + 1] != kSeparator) {
                    su.prev_pos = su.last_pos;
                }
            }
            su.last_pos = pos;
            if (su.hits < std::numeric_limits<std::uint32_t>::max()) {
                ++su.hits;
            }
            u = su.link;
        }
    }

    void ExtendSam(Token tok, std::int32_t pos) {
        // Proper Generalized SAM handling when `last_state_` already has a
        // transition on `tok` (e.g. after `AppendSeparator()` resets `last_state_ = 0`):
        // reuse or clone without creating unreachable equal-length ghost states.
        const std::int32_t existing_q = FindEdge(last_state_, tok);
        if (existing_q >= 0) {
            const std::int32_t next_len = states_[static_cast<std::size_t>(last_state_)].len + 1;
            if (next_len == states_[static_cast<std::size_t>(existing_q)].len) {
                RefreshLinks(existing_q, pos);
                last_state_ = existing_q;
                return;
            }
            const auto clone = static_cast<std::int32_t>(states_.size());
            const State q_copy = states_[static_cast<std::size_t>(existing_q)];
            states_.push_back(State{
                next_len,
                q_copy.link,
                q_copy.last_pos,
                q_copy.prev_pos,
                -1,
                q_copy.hits,
            });
            CopyEdges(existing_q, clone);
            std::int32_t p = last_state_;
            while (p != -1 && FindEdge(p, tok) == existing_q) {
                SetEdge(p, tok, clone);
                p = states_[static_cast<std::size_t>(p)].link;
            }
            states_[static_cast<std::size_t>(existing_q)].link = clone;
            RefreshLinks(clone, pos);
            last_state_ = clone;
            return;
        }

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
            const std::int32_t split_len = states_[static_cast<std::size_t>(p)].len + 1;
            if (split_len == states_[static_cast<std::size_t>(q)].len) {
                states_[static_cast<std::size_t>(cur)].link = q;
            } else {
                const auto clone = static_cast<std::int32_t>(states_.size());
                const State q_copy = states_[static_cast<std::size_t>(q)];
                states_.push_back(State{
                    split_len,
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

        RefreshLinks(states_[static_cast<std::size_t>(cur)].link, pos);
        last_state_ = cur;
    }

    void Compact() {
        const std::size_t keep = std::max<std::size_t>(1, max_tokens_ / 2);
        std::size_t drop = tokens_.size() > keep ? (tokens_.size() - keep) : 0;
        while (drop < tokens_.size() && tokens_[drop] == kSeparator) {
            ++drop;
        }
        if (drop > 0) {
            tokens_.erase(tokens_.begin(), tokens_.begin() + static_cast<std::ptrdiff_t>(drop));
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
    static constexpr std::uint32_t kMinCostSamples = 2u;
    static constexpr double kCostStepDownRatio = 1.05;
    static constexpr double kCostPauseRatio = 1.12;
    static constexpr double kCostImproveRatio = 0.98;

    explicit SuffixMatcher(std::size_t session_ring_capacity = BoundedSuffixAutomaton::kDefaultCapacity,
                           std::size_t active_capacity = BoundedSuffixAutomaton::kDefaultCapacity)
        : active_(active_capacity),
          session_ring_(session_ring_capacity),
          max_request_history_(
              std::max<std::size_t>(session_ring_.capacity(), BoundedSuffixAutomaton::kMaxQueryLookback)) {}

    void ResetAll() {
        active_.Reset();
        session_ring_.Reset();
        current_request_.clear();
        ResetAdaptive();
    }

    // Discards active-request state without ingesting it into the cross-request
    // session ring (used when a single-request fast path transitions to batch mode).
    void ClearActive() {
        active_.Reset();
        current_request_.clear();
        ResetAdaptive();
    }

    void BeginRequest(std::span<const Token> prompt) {
        active_.Reset();
        const std::size_t keep = std::min(prompt.size(), max_request_history_);
        const std::size_t start = prompt.size() - keep;
        current_request_.assign(prompt.begin() + static_cast<std::ptrdiff_t>(start), prompt.end());
        active_.AppendSpan(prompt);
        ResetAdaptive();
    }

    // Commits a verified token into the active request SAM.
    void CommitToken(Token tok) {
        if (tok < 0) return;
        if (current_request_.size() >= max_request_history_ * 2u) {
            const std::size_t drop = current_request_.size() - max_request_history_;
            current_request_.erase(current_request_.begin(),
                                   current_request_.begin() + static_cast<std::ptrdiff_t>(drop));
        }
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

    // Proposes a confidence-scored dynamic draft sequence (`min_draft_len..max_draft`)
    // matching the longest suffix of `tail` (or `current_request_` + `id_last`
    // via DraftForCurrent).
    [[nodiscard]] SuffixMatchResult Draft(std::span<const Token> tail, std::size_t max_draft,
                                          std::size_t min_match_len = kDefaultMinMatchLen,
                                          std::size_t min_draft_len = 1) const {
        if (max_draft == 0 || tail.empty()) return {};
        const std::size_t capped_max = std::min<std::size_t>(max_draft, static_cast<std::size_t>(kMaxSpecDraftTokens));
        const std::size_t eff_min_draft = std::max<std::size_t>(1, min_draft_len);
        if (capped_max < eff_min_draft || cooldown_remaining_ > 0) {
            return {};
        }

        const std::size_t eff_min = std::max(min_match_len, adaptive_min_match_);
        SuffixMatchResult local_match = active_.MatchLongestSuffix(
            tail, eff_min, capped_max, /*exclude_tail_tokens=*/0, draft_factor_, eff_min_draft);
        SuffixMatchResult ring_match = session_ring_.MatchLongestSuffix(
            tail, eff_min, capped_max, /*exclude_tail_tokens=*/0, draft_factor_, eff_min_draft);
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
        const std::size_t keep =
            std::min<std::size_t>(current_request_.size(), BoundedSuffixAutomaton::kMaxQueryLookback - 1);
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
        SuffixMatchResult match = Draft(tail, burst_cap, kHybridMtpMinMatchLen, kHybridMtpMinBurstDraft);
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
        const std::size_t keep =
            std::min<std::size_t>(current_request_.size(), BoundedSuffixAutomaton::kMaxQueryLookback - 1);
        const std::size_t start = current_request_.size() - keep;
        query_scratch_.insert(query_scratch_.end(),
                              current_request_.begin() + static_cast<std::ptrdiff_t>(start), current_request_.end());
        query_scratch_.push_back(id_last);
        return TryHybridMtpBurst(query_scratch_, max_k);
    }

    void NoteSkippedStep() noexcept {
        if (cooldown_remaining_ > 0) {
            --cooldown_remaining_;
            if (cooldown_remaining_ == 0) {
                cost_paused_ = false;
            }
        }
    }

    // Optional measured-cost hook: records an ordinary non-speculative decode
    // step's elapsed ms per committed token as the K=0 baseline.
    void ObserveBaselineStep(double elapsed_ms, std::size_t committed_tokens = 1) noexcept {
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0 || committed_tokens == 0) return;
        const double ms_per_tok = elapsed_ms / static_cast<double>(committed_tokens);
        if (!std::isfinite(ms_per_tok) || ms_per_tok <= 0.0) return;
        if (baseline_cost_samples_ < 4u) {
            const double prev = static_cast<double>(baseline_cost_samples_);
            baseline_ms_per_token_ema_ = (baseline_ms_per_token_ema_ * prev + ms_per_tok) / (prev + 1.0);
        } else {
            constexpr double kAlpha = 0.25;
            baseline_ms_per_token_ema_ = (1.0 - kAlpha) * baseline_ms_per_token_ema_ + kAlpha * ms_per_tok;
        }
        if (baseline_cost_samples_ < std::numeric_limits<std::uint32_t>::max()) {
            ++baseline_cost_samples_;
        }
    }

    void Observe(std::size_t drafted, std::size_t accepted) noexcept {
        if (drafted == 0) return;
        const std::size_t clamped_acc = std::min(accepted, drafted);
        const double rate = static_cast<double>(clamped_acc) / static_cast<double>(drafted);
        constexpr double kAlpha = 0.25;
        acceptance_ema_ = (steps_observed_ == 0) ? rate : (1.0 - kAlpha) * acceptance_ema_ + kAlpha * rate;
        if (steps_observed_ < std::numeric_limits<std::uint32_t>::max()) {
            ++steps_observed_;
        }

        if (clamped_acc == drafted) {
            draft_factor_ = std::min(3.0, draft_factor_ + 0.35);
            adaptive_min_match_ = kDefaultMinMatchLen;
            cooldown_period_ = 4;
            low_streak_ = 0;
        } else if (acceptance_ema_ >= 0.50) {
            if (draft_factor_ < 2.75) {
                draft_factor_ = std::min(2.75, draft_factor_ + 0.15);
            }
            adaptive_min_match_ = kDefaultMinMatchLen;
            cooldown_period_ = 4;
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

    // Optional measured-cost hook: updates acceptance tracking AND compares
    // suffix verify elapsed ms per committed token against the measured
    // ordinary-decode baseline. When `committed_tokens == 0`, defaults to
    // `min(accepted, drafted) + 1` (verified drafts + correction/bonus token).
    // Measured latency regression overrides high acceptance by shrinking
    // `draft_factor_` and triggering cooldown pause.
    void ObserveWithCost(std::size_t drafted, std::size_t accepted, double elapsed_ms,
                         std::size_t committed_tokens = 0) noexcept {
        if (drafted == 0) return;
        const double prev_draft_factor = draft_factor_;
        const std::size_t prev_min_match = adaptive_min_match_;
        const std::int32_t prev_cooldown_period = cooldown_period_;
        Observe(drafted, accepted);

        const std::size_t clamped_acc = std::min(accepted, drafted);
        const std::size_t eff_committed = committed_tokens > 0 ? committed_tokens : (clamped_acc + 1u);
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0 || eff_committed == 0) return;
        const double step_ms_per_tok = elapsed_ms / static_cast<double>(eff_committed);
        if (!std::isfinite(step_ms_per_tok) || step_ms_per_tok <= 0.0) return;

        if (suffix_cost_samples_ < 4u) {
            const double prev = static_cast<double>(suffix_cost_samples_);
            suffix_ms_per_token_ema_ = (suffix_ms_per_token_ema_ * prev + step_ms_per_tok) / (prev + 1.0);
        } else {
            constexpr double kAlpha = 0.25;
            suffix_ms_per_token_ema_ = (1.0 - kAlpha) * suffix_ms_per_token_ema_ + kAlpha * step_ms_per_tok;
        }
        if (suffix_cost_samples_ < std::numeric_limits<std::uint32_t>::max()) {
            ++suffix_cost_samples_;
        }

        if (baseline_cost_samples_ < kMinCostSamples || suffix_cost_samples_ < kMinCostSamples ||
            baseline_ms_per_token_ema_ <= 0.0) {
            return;
        }

        const double ratio = suffix_ms_per_token_ema_ / baseline_ms_per_token_ema_;
        if (ratio >= kCostStepDownRatio) {
            draft_factor_ = std::max(1.0, prev_draft_factor * 0.75);
            cooldown_period_ = prev_cooldown_period;
            ++cost_regress_streak_;
            if (ratio >= kCostPauseRatio || cost_regress_streak_ >= 3) {
                cooldown_remaining_ = std::min<std::int32_t>(cooldown_period_, 16);
                cooldown_period_ = std::min<std::int32_t>(cooldown_period_ * 2, 16);
                adaptive_min_match_ = std::min<std::size_t>(std::max(prev_min_match, adaptive_min_match_) + 1u, 6u);
                cost_paused_ = true;
                cost_regress_streak_ = 0;
                if (cost_regressions_ < std::numeric_limits<std::uint32_t>::max()) {
                    ++cost_regressions_;
                }
            }
        } else if (ratio <= kCostImproveRatio) {
            cost_regress_streak_ = 0;
            cost_paused_ = false;
        } else {
            // Break-even hysteresis deadband: do not inflate draft_factor_ even on
            // high acceptance until suffix verify clearly beats baseline.
            draft_factor_ = std::min(draft_factor_, prev_draft_factor);
            cost_regress_streak_ = 0;
            cost_paused_ = false;
        }
    }

    [[nodiscard]] std::size_t active_size() const noexcept { return active_.size(); }
    [[nodiscard]] std::size_t session_ring_size() const noexcept { return session_ring_.size(); }
    [[nodiscard]] std::size_t session_ring_capacity() const noexcept { return session_ring_.capacity(); }
    [[nodiscard]] std::uint64_t session_ring_compactions() const noexcept { return session_ring_.compactions(); }
    [[nodiscard]] double acceptance_ema() const noexcept { return acceptance_ema_; }
    [[nodiscard]] double draft_factor() const noexcept { return draft_factor_; }
    [[nodiscard]] std::int32_t cooldown_period() const noexcept { return cooldown_period_; }
    [[nodiscard]] bool paused() const noexcept { return cooldown_remaining_ > 0; }
    [[nodiscard]] bool cost_paused() const noexcept { return cost_paused_ && cooldown_remaining_ > 0; }
    [[nodiscard]] double suffix_ms_per_token() const noexcept { return suffix_ms_per_token_ema_; }
    [[nodiscard]] double baseline_ms_per_token() const noexcept { return baseline_ms_per_token_ema_; }
    [[nodiscard]] SuffixCostStats cost_stats() const noexcept {
        return SuffixCostStats{
            suffix_ms_per_token_ema_,
            baseline_ms_per_token_ema_,
            suffix_cost_samples_,
            baseline_cost_samples_,
            cost_regressions_,
            cost_paused(),
        };
    }
    [[nodiscard]] bool CheckInvariants() const noexcept {
        return active_.CheckInvariants() && session_ring_.CheckInvariants() &&
               current_request_.size() <= max_request_history_ * 2u;
    }

private:
    void ResetAdaptive() noexcept {
        acceptance_ema_ = 0.5;
        draft_factor_ = 2.0;
        adaptive_min_match_ = kDefaultMinMatchLen;
        steps_observed_ = 0;
        low_streak_ = 0;
        cooldown_remaining_ = 0;
        cooldown_period_ = 4;
        suffix_ms_per_token_ema_ = 0.0;
        baseline_ms_per_token_ema_ = 0.0;
        suffix_cost_samples_ = 0;
        baseline_cost_samples_ = 0;
        cost_regress_streak_ = 0;
        cost_regressions_ = 0;
        cost_paused_ = false;
    }

    BoundedSuffixAutomaton active_;
    BoundedSuffixAutomaton session_ring_;
    std::size_t max_request_history_ = BoundedSuffixAutomaton::kDefaultCapacity;
    std::vector<Token> current_request_;
    std::vector<Token> query_scratch_;

    double acceptance_ema_ = 0.5;
    double draft_factor_ = 2.0;
    std::size_t adaptive_min_match_ = kDefaultMinMatchLen;
    std::uint32_t steps_observed_ = 0;
    std::int32_t low_streak_ = 0;
    std::int32_t cooldown_remaining_ = 0;
    std::int32_t cooldown_period_ = 4;

    double suffix_ms_per_token_ema_ = 0.0;
    double baseline_ms_per_token_ema_ = 0.0;
    std::uint32_t suffix_cost_samples_ = 0;
    std::uint32_t baseline_cost_samples_ = 0;
    std::int32_t cost_regress_streak_ = 0;
    std::uint32_t cost_regressions_ = 0;
    bool cost_paused_ = false;
};

}  // namespace sonder::backends::llamacpp
