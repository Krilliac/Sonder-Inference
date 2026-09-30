#include "sonder/inference/cache/kv_cache_manager.hpp"

#include <numeric>
#include <vector>

#include <doctest/doctest.h>

using namespace sonder::inference;
using namespace sonder::inference::cache;

namespace {

CacheFingerprint fp() { return FingerprintBuilder{}.add("checkpoint-model").add("rev1").build(); }

std::vector<TokenId> tokens(TokenId first, std::size_t count) {
    std::vector<TokenId> result(count);
    std::iota(result.begin(), result.end(), first);
    return result;
}

std::vector<TokenId> contents(const KvCacheManager& manager, SequenceId id) {
    std::vector<TokenId> result;
    const auto length = manager.num_tokens(id);
    for (const auto block : manager.block_table(id)) {
        const auto values = manager.block_tokens(block);
        const auto remaining = length - result.size();
        const auto take = std::min(remaining, values.size());
        result.insert(result.end(), values.begin(), values.begin() + static_cast<std::ptrdiff_t>(take));
        if (result.size() == length) break;
    }
    return result;
}

KvCacheConfig config(std::uint32_t blocks = 16, std::uint32_t block_size = 4) {
    KvCacheConfig result;
    result.num_blocks = blocks;
    result.block_size_tokens = block_size;
    result.bytes_per_token = 16;
    return result;
}

SequenceOptions recurrent() {
    return SequenceOptions{fp(), 0, ModelArchitecture::hybrid};
}

}  // namespace

TEST_CASE("checkpoint: cp8 is sufficient for an eight-token prefix") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 64).ok());
    REQUIRE(manager.free_sequence(1).ok());

    CHECK(manager.match_prefix(fp(), tokens(0, 8), ModelArchitecture::hybrid) == 8);
}

TEST_CASE("checkpoint: partial position is exact and never rounded") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 12)).ok());
    REQUIRE(manager.save_checkpoint(1, 6, 64).ok());
    REQUIRE(manager.free_sequence(1).ok());

    CHECK(manager.match_prefix(fp(), tokens(0, 4), ModelArchitecture::hybrid) == 0);
    CHECK(manager.match_prefix(fp(), tokens(0, 12), ModelArchitecture::hybrid) == 6);
}

TEST_CASE("checkpoint: divergent lineages do not borrow equal-length state") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 64).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    REQUIRE(manager.append_tokens(2, tokens(100, 8)).ok());
    REQUIRE(manager.save_checkpoint(2, 8, 64).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.free_sequence(2).ok());

    CHECK(manager.match_prefix(fp(), tokens(0, 8), ModelArchitecture::hybrid) == 8);
    CHECK(manager.match_prefix(fp(), tokens(100, 8), ModelArchitecture::hybrid) == 8);
    CHECK(manager.match_prefix(fp(), tokens(50, 8), ModelArchitecture::hybrid) == 0);
    REQUIRE(manager.add_sequence(3, recurrent()).ok());
    AppendResult result;
    REQUIRE(manager.append_tokens(3, tokens(100, 8), &result).ok());
    REQUIRE(result.checkpoint.has_value());
    CHECK(result.checkpoint->owner == 2);
    CHECK(result.checkpoint->position == 8);
}

TEST_CASE("checkpoint: replacement resize participates in byte eviction") {
    auto cfg = config();
    cfg.max_checkpoint_bytes = 100;
    KvCacheManager manager(cfg);
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 60).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 90).ok());
    CHECK(manager.stats().checkpoint_bytes == 90);
    REQUIRE(manager.save_checkpoint(1, 8, 40).ok());
    CHECK(manager.stats().checkpoint_bytes <= 100);
    CHECK(manager.stats().checkpoint_evictions >= 1);
}

TEST_CASE("checkpoint: zero capacity disables saving") {
    auto cfg = config();
    cfg.max_checkpoint_bytes = 0;
    KvCacheManager manager(cfg);
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    CHECK(manager.save_checkpoint(1, 4, 1).code() == ErrorCode::invalid_argument);
    CHECK(manager.stats().checkpoints == 0);
    CHECK(manager.stats().checkpoint_bytes == 0);
}

TEST_CASE("checkpoint: failed append leaves sequence and counters unchanged") {
    KvCacheManager manager(config(2));
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    const auto before = manager.stats();
    const auto length = manager.num_tokens(1);
    CHECK(manager.append_tokens(1, tokens(4, 12)).code() == ErrorCode::unavailable);
    CHECK(manager.num_tokens(1) == length);
    CHECK(manager.stats().blocks_allocated == before.blocks_allocated);
    CHECK(manager.stats().allocation_failures == before.allocation_failures + 1);
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: zero-token append does not select a checkpoint") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    AppendResult result;
    REQUIRE(manager.append_tokens(1, {}, &result).ok());
    CHECK(!result.checkpoint.has_value());
    CHECK(result.tokens_checkpoint_limited == 0);
}

TEST_CASE("checkpoint: removal prevents subsequent reuse") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    REQUIRE(manager.remove_checkpoint(1, 4).ok());
    REQUIRE(manager.free_sequence(1).ok());
    CHECK(manager.match_prefix(fp(), tokens(0, 4), ModelArchitecture::hybrid) == 0);
}

TEST_CASE("checkpoint: owner reuse does not inherit a previous owner's state") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(7, recurrent()).ok());
    REQUIRE(manager.append_tokens(7, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(7, 4, 32).ok());
    REQUIRE(manager.free_sequence(7).ok());
    REQUIRE(manager.add_sequence(7, recurrent()).ok());
    CHECK(manager.match_prefix(fp(), tokens(0, 4), ModelArchitecture::hybrid) == 0);
}

TEST_CASE("checkpoint: truncate requires checkpoint boundary or zero") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    CHECK(manager.truncate(1, 2).code() == ErrorCode::invalid_argument);
    CHECK(manager.truncate(1, 4).ok());
    CHECK(manager.truncate(1, 0).ok());
}

TEST_CASE("checkpoint: fork requires a saved current state") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    CHECK(manager.fork_sequence(1, 2).code() == ErrorCode::invalid_state);
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    CHECK(manager.fork_sequence(1, 2).ok());
}

TEST_CASE("checkpoint: block eviction invalidates dependent state") {
    KvCacheManager manager(config(1));
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    REQUIRE(manager.append_tokens(2, tokens(100, 4)).ok());
    CHECK(manager.match_prefix(fp(), tokens(0, 4), ModelArchitecture::hybrid) == 0);
}

TEST_CASE("checkpoint: attention-only default preserves cache gauges and counters") {
    KvCacheManager implicit(config());
    KvCacheManager explicit_arch(config());
    REQUIRE(implicit.add_sequence(1, {fp(), 0}).ok());
    REQUIRE(explicit_arch.add_sequence(1, {fp(), 0, ModelArchitecture::attention_only}).ok());
    REQUIRE(implicit.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(explicit_arch.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(implicit.free_sequence(1).ok());
    REQUIRE(explicit_arch.free_sequence(1).ok());
    const auto a = implicit.stats();
    const auto b = explicit_arch.stats();
    CHECK(a.cached_blocks == b.cached_blocks);
    CHECK(a.cached_bytes == b.cached_bytes);
    CHECK(a.prefix_hit_blocks == b.prefix_hit_blocks);
    CHECK(a.prefix_miss_lookups == b.prefix_miss_lookups);
    CHECK(a.avoided_prefill_tokens == b.avoided_prefill_tokens);
    CHECK(a.evictions == b.evictions);
    CHECK(a.checkpoint_bytes == 0);
    CHECK(b.checkpoint_bytes == 0);
}

TEST_CASE("checkpoint: chunked prefill does not retroactively claim a later checkpoint") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 32).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    AppendResult first;
    REQUIRE(manager.append_tokens(2, tokens(0, 4), &first).ok());
    CHECK(first.tokens_reused == 0);
    AppendResult second;
    REQUIRE(manager.append_tokens(2, tokens(4, 4), &second).ok());
    CHECK(second.tokens_reused == 0);
}

TEST_CASE("checkpoint: chunked prefill can reuse successive saved boundaries") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    REQUIRE(manager.append_tokens(1, tokens(4, 4)).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 32).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    AppendResult first;
    REQUIRE(manager.append_tokens(2, tokens(0, 4), &first).ok());
    CHECK(first.tokens_reused == 4);
    AppendResult second;
    REQUIRE(manager.append_tokens(2, tokens(4, 4), &second).ok());
    CHECK(second.tokens_reused == 4);
}

TEST_CASE("checkpoint: partial append reuses through cp6 and reports its owner") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 12)).ok());
    REQUIRE(manager.save_checkpoint(1, 6, 32).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    AppendResult result;
    REQUIRE(manager.append_tokens(2, tokens(0, 12), &result).ok());
    CHECK(result.tokens_reused == 6);
    REQUIRE(result.checkpoint.has_value());
    CHECK(result.checkpoint->owner == 1);
    CHECK(result.checkpoint->position == 6);
    CHECK(manager.num_tokens(2) == 12);
    CHECK(contents(manager, 2) == tokens(0, 12));
    CHECK(manager.block_table(2).size() == 3);
    CHECK(manager.block_tokens(manager.block_table(2)[0])[0] == 0);
    CHECK(manager.block_tokens(manager.block_table(2)[1])[0] == 4);
    CHECK(manager.block_tokens(manager.block_table(2)[2])[0] == 8);
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: insufficient pool recomputes instead of overallocating") {
    KvCacheManager manager(config(3));
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 12)).ok());
    REQUIRE(manager.save_checkpoint(1, 6, 32).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    AppendResult result;
    REQUIRE(manager.append_tokens(2, tokens(0, 12), &result).ok());
    CHECK(result.tokens_reused == 0);
    CHECK(manager.stats().pinned_blocks <= 3);
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: partial current state remains the matching lineage") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 6)).ok());
    REQUIRE(manager.save_checkpoint(1, 6, 32).ok());
    REQUIRE(manager.append_tokens(1, tokens(6, 2)).ok());
    CHECK(manager.match_prefix(fp(), tokens(0, 8), ModelArchitecture::hybrid) == 6);
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: reuse refreshes the selected checkpoint before LRU eviction") {
    auto cfg = config();
    cfg.max_checkpoint_bytes = 100;
    KvCacheManager manager(cfg);
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 12)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 40).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 40).ok());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    AppendResult reused;
    REQUIRE(manager.append_tokens(2, tokens(0, 4), &reused).ok());
    REQUIRE(manager.save_checkpoint(1, 12, 40).ok());
    CHECK(manager.stats().checkpoint_evictions >= 1);
    CHECK(manager.checkpoint_at(1, 4).has_value());
    CHECK(!manager.checkpoint_at(1, 8).has_value());
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: architecture is part of prefix identity") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, {fp(), 0, ModelArchitecture::attention_only}).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.free_sequence(1).ok());
    CHECK(manager.match_prefix(fp(), tokens(0, 8), ModelArchitecture::hybrid) == 0);
}

TEST_CASE("checkpoint: maximum state size and uint64 budget are safe") {
    auto cfg = config();
    cfg.max_checkpoint_bytes = UINT64_MAX;
    KvCacheManager manager(cfg);
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    CHECK(manager.save_checkpoint(1, 4, 0).code() == ErrorCode::invalid_argument);
    REQUIRE(manager.save_checkpoint(1, 4, UINT64_MAX).ok());
    CHECK(manager.stats().checkpoint_bytes == UINT64_MAX);
    CHECK(manager.save_checkpoint(1, 4, UINT64_MAX).ok());
    CHECK(manager.stats().checkpoint_bytes == UINT64_MAX);
}

TEST_CASE("checkpoint: evicting the parent snapshot preserves a live fork but requires a new snapshot") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    REQUIRE(manager.fork_sequence(1, 2).ok());
    REQUIRE(manager.free_sequence(1).ok());
    REQUIRE(manager.remove_checkpoint(1, 4).ok());
    CHECK(contents(manager, 2) == tokens(0, 4));
    CHECK(manager.validate());
    CHECK(manager.fork_sequence(2, 3).code() == ErrorCode::invalid_state);
    REQUIRE(manager.save_checkpoint(2, 4, 32).ok());
    CHECK(manager.fork_sequence(2, 3).ok());
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: recycled child id is rejected before checkpoint purging") {
    KvCacheManager manager(config());
    REQUIRE(manager.add_sequence(2, recurrent()).ok());
    REQUIRE(manager.append_tokens(2, tokens(0, 4)).ok());
    REQUIRE(manager.save_checkpoint(2, 4, 32).ok());
    REQUIRE(manager.free_sequence(2).ok());
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 4)).ok());
    CHECK(manager.fork_sequence(1, 2).code() == ErrorCode::invalid_state);
    REQUIRE(manager.checkpoint_at(1, 4).has_value());
    CHECK(manager.checkpoint_at(1, 4)->owner == 2);
    CHECK(!manager.has_sequence(2));
    CHECK(manager.validate());
}

TEST_CASE("checkpoint: count cap emits the full evicted snapshot and clear resets gauges") {
    auto cfg = config();
    cfg.max_checkpoints = 1;
    KvCacheManager manager(cfg);
    std::vector<RecurrentStateCheckpoint> evicted;
    manager.set_event_listener([&](const CacheEvent& event) {
        if (event.kind == CacheEvent::Kind::checkpoint_evicted && event.checkpoint) {
            evicted.push_back(*event.checkpoint);
        }
    });
    REQUIRE(manager.add_sequence(1, recurrent()).ok());
    REQUIRE(manager.append_tokens(1, tokens(0, 8)).ok());
    REQUIRE(manager.save_checkpoint(1, 4, 32).ok());
    REQUIRE(manager.save_checkpoint(1, 8, 48).ok());
    REQUIRE(evicted.size() == 1);
    CHECK(evicted[0].owner == 1);
    CHECK(evicted[0].position == 4);
    CHECK(evicted[0].state_size == 32);
    CHECK(manager.stats().checkpoints == 1);
    REQUIRE(manager.free_sequence(1).ok());
    CHECK(manager.clear_cached() >= 1);
    CHECK(manager.stats().checkpoints == 0);
    CHECK(manager.stats().checkpoint_bytes == 0);
    CHECK(manager.stats().free_blocks == manager.stats().total_blocks);
    CHECK(manager.validate());
}
