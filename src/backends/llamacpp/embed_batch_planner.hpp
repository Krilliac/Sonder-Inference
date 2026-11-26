// Pure length-aware multi-sequence batch planner for direct embeddings and
// cross-encoder pair scoring. Zero llama.cpp header dependency so it can be
// exhaustively unit-tested without model weights.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::backends::llamacpp {

struct EmbedBatchPlanConfig {
    // Nominal token budget packed into one multi-sequence forward pass.
    std::size_t max_batch_tokens = 2048;
    // Maximum concurrent sequences (n_seq_max) packed into one forward pass.
    std::size_t max_sequences_per_batch = 16;
    // Hard per-input context window cap. 0 means equal to max_batch_tokens.
    std::size_t max_single_input_tokens = 0;
    // When true, inputs with max_batch_tokens < len <= max_single_input_tokens
    // are isolated into single-sequence fallback batches instead of failing.
    bool allow_oversized_single_batch = true;
};

struct EmbedBatchItem {
    std::size_t original_index = 0;
    std::size_t token_count = 0;
    std::int32_t seq_id = 0;
};

struct PlannedEmbedBatch {
    std::vector<EmbedBatchItem> items;
    std::size_t total_tokens = 0;
    bool oversized_fallback = false;
};

struct EmbedBatchPlan {
    bool ok = false;
    bool context_overflow = false;
    std::string error;
    std::size_t failed_index = 0;
    std::vector<PlannedEmbedBatch> batches;
    std::size_t total_inputs = 0;
    std::size_t total_tokens = 0;
    std::size_t max_batch_tokens_used = 0;
    std::size_t max_batch_sequences_used = 0;
};

// Plans length-sorted multi-sequence batches over `lengths`:
//   - Rejects empty inputs or inputs exceeding hard context cap.
//   - Groups inputs with len <= max_batch_tokens sorted by (len ASC, index ASC)
//     so each batch contains similar-length sequences within both token and
//     sequence-count budgets.
//   - Isolates inputs with max_batch_tokens < len <= max_single_input_tokens
//     into dedicated single-sequence fallback batches (oversized_fallback=true)
//     when allow_oversized_single_batch is enabled.
//   - Preserves every item's exact `original_index` so callers restore results
//     in the caller's original order.
inline EmbedBatchPlan PlanEmbedBatches(std::span<const std::size_t> lengths,
                                       const EmbedBatchPlanConfig& config = {}) {
    EmbedBatchPlan plan;
    if (lengths.empty()) {
        plan.error = "embedding inputs are empty";
        return plan;
    }
    if (config.max_batch_tokens == 0) {
        plan.error = "max_batch_tokens must be positive";
        return plan;
    }
    if (config.max_sequences_per_batch == 0) {
        plan.error = "max_sequences_per_batch must be positive";
        return plan;
    }

    const std::size_t hard_cap =
        config.max_single_input_tokens == 0 ? config.max_batch_tokens : config.max_single_input_tokens;

    struct Entry {
        std::size_t index = 0;
        std::size_t len = 0;
    };
    std::vector<Entry> normal;
    std::vector<Entry> oversized;
    normal.reserve(lengths.size());

    for (std::size_t i = 0; i < lengths.size(); ++i) {
        const std::size_t len = lengths[i];
        if (len == 0) {
            plan.failed_index = i;
            plan.error = "embedding input is empty";
            return plan;
        }
        if (len > hard_cap) {
            plan.failed_index = i;
            plan.context_overflow = true;
            plan.error = "embedding input exceeds embedding context window";
            return plan;
        }
        if (len > config.max_batch_tokens) {
            if (!config.allow_oversized_single_batch) {
                plan.failed_index = i;
                plan.context_overflow = true;
                plan.error = "embedding input exceeds embedding batch token budget";
                return plan;
            }
            oversized.push_back(Entry{i, len});
        } else {
            normal.push_back(Entry{i, len});
        }
        plan.total_tokens += len;
    }

    auto by_length_then_index = [](const Entry& a, const Entry& b) noexcept {
        if (a.len != b.len) return a.len < b.len;
        return a.index < b.index;
    };
    std::sort(normal.begin(), normal.end(), by_length_then_index);
    std::sort(oversized.begin(), oversized.end(), by_length_then_index);

    if (!normal.empty()) {
        PlannedEmbedBatch current;
        for (const Entry& e : normal) {
            const bool would_exceed_tokens =
                !current.items.empty() && (current.total_tokens + e.len > config.max_batch_tokens);
            const bool would_exceed_seqs = current.items.size() >= config.max_sequences_per_batch;
            if (would_exceed_tokens || would_exceed_seqs) {
                plan.max_batch_tokens_used = std::max(plan.max_batch_tokens_used, current.total_tokens);
                plan.max_batch_sequences_used = std::max(plan.max_batch_sequences_used, current.items.size());
                plan.batches.push_back(std::move(current));
                current = PlannedEmbedBatch{};
            }
            const auto seq_id = static_cast<std::int32_t>(current.items.size());
            current.items.push_back(EmbedBatchItem{e.index, e.len, seq_id});
            current.total_tokens += e.len;
        }
        if (!current.items.empty()) {
            plan.max_batch_tokens_used = std::max(plan.max_batch_tokens_used, current.total_tokens);
            plan.max_batch_sequences_used = std::max(plan.max_batch_sequences_used, current.items.size());
            plan.batches.push_back(std::move(current));
        }
    }

    for (const Entry& e : oversized) {
        PlannedEmbedBatch single;
        single.oversized_fallback = true;
        single.total_tokens = e.len;
        single.items.push_back(EmbedBatchItem{e.index, e.len, 0});
        plan.max_batch_tokens_used = std::max(plan.max_batch_tokens_used, single.total_tokens);
        plan.max_batch_sequences_used = std::max<std::size_t>(plan.max_batch_sequences_used, 1u);
        plan.batches.push_back(std::move(single));
    }

    plan.total_inputs = lengths.size();
    plan.ok = true;
    return plan;
}

enum class EmbedBatchExecStatus {
    kOk = 0,
    kInvalidPlan,
    kCancelled,
    kBatchFailed,
};

// Executes a planned batch schedule with cooperative cancellation between
// batches and scatters outputs back into exact original input indices.
template <typename T, typename CancelFn, typename RunBatchFn>
EmbedBatchExecStatus ExecuteEmbedBatchPlan(const EmbedBatchPlan& plan, const CancelFn& is_cancelled,
                                           const RunBatchFn& run_batch, std::vector<T>& out_results,
                                           std::size_t* batches_executed = nullptr) {
    out_results.clear();
    if (batches_executed != nullptr) {
        *batches_executed = 0;
    }
    if (!plan.ok || plan.total_inputs == 0) {
        return EmbedBatchExecStatus::kInvalidPlan;
    }
    out_results.resize(plan.total_inputs);
    std::vector<T> batch_out;
    for (const PlannedEmbedBatch& batch : plan.batches) {
        if (is_cancelled()) {
            return EmbedBatchExecStatus::kCancelled;
        }
        batch_out.clear();
        if (!run_batch(batch, batch_out)) {
            return is_cancelled() ? EmbedBatchExecStatus::kCancelled : EmbedBatchExecStatus::kBatchFailed;
        }
        if (batch_out.size() != batch.items.size()) {
            return EmbedBatchExecStatus::kBatchFailed;
        }
        for (std::size_t i = 0; i < batch.items.size(); ++i) {
            const std::size_t dst = batch.items[i].original_index;
            if (dst >= out_results.size()) {
                return EmbedBatchExecStatus::kBatchFailed;
            }
            out_results[dst] = std::move(batch_out[i]);
        }
        if (batches_executed != nullptr) {
            ++(*batches_executed);
        }
    }
    return EmbedBatchExecStatus::kOk;
}

struct RerankTemplateSegment {
    std::string text;
    bool parse_special = false;
};

// Single-pass template splitter for GGUF `rerank` templates with `{query}` and
// `{document}` placeholders. Template scaffolding keeps parse_special=true
// while untrusted query/document text stays parse_special=false (and a query
// containing literal "{document}" is never re-substituted).
inline std::vector<RerankTemplateSegment> SplitRerankTemplate(std::string_view tmpl, std::string_view query,
                                                              std::string_view document) {
    constexpr std::string_view kQuery = "{query}";
    constexpr std::string_view kDoc = "{document}";
    std::vector<RerankTemplateSegment> segments;
    std::size_t cursor = 0;
    while (cursor < tmpl.size()) {
        const std::size_t qpos = tmpl.find(kQuery, cursor);
        const std::size_t dpos = tmpl.find(kDoc, cursor);
        const std::size_t next = std::min(qpos, dpos);
        if (next == std::string_view::npos) {
            segments.push_back(RerankTemplateSegment{std::string(tmpl.substr(cursor)), true});
            break;
        }
        if (next > cursor) {
            segments.push_back(RerankTemplateSegment{std::string(tmpl.substr(cursor, next - cursor)), true});
        }
        if (next == qpos) {
            if (!query.empty()) {
                segments.push_back(RerankTemplateSegment{std::string(query), false});
            }
            cursor = next + kQuery.size();
        } else {
            if (!document.empty()) {
                segments.push_back(RerankTemplateSegment{std::string(document), false});
            }
            cursor = next + kDoc.size();
        }
    }
    return segments;
}

}  // namespace sonder::backends::llamacpp
