#include "sonder/inference/cache/kv_cache_manager.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <ostream>  // doctest stringifies std::string_view; MSVC needs the full definition
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

using namespace sonder::inference::cache;
using sonder::inference::ErrorCode;

namespace {

KvCacheConfig cfg(std::uint32_t blocks, std::uint32_t block_size = 4, bool prefix = true) {
    KvCacheConfig c;
    c.block_size_tokens = block_size;
    c.num_blocks = blocks;
    c.enable_prefix_caching = prefix;
    c.bytes_per_token = 100;
    return c;
}

std::vector<TokenId> toks(TokenId start, std::size_t n) {
    std::vector<TokenId> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

std::vector<TokenId> cat(std::vector<TokenId> a, const std::vector<TokenId>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Reconstruct a sequence's logical contents from its block table.
std::vector<TokenId> contents(const KvCacheManager& m, SequenceId id) {
    std::vector<TokenId> out;
    const std::size_t n = m.num_tokens(id);
    for (const BlockId b : m.block_table(id)) {
        for (const TokenId t : m.block_tokens(b)) {
            if (out.size() == n) break;
            out.push_back(t);
        }
    }
    return out;
}

const CacheFingerprint kFp = FingerprintBuilder{}.add("model-a").add("rev1").add(std::uint64_t{16}).build();

}  // namespace

TEST_CASE("config: invalid block size and watermarks throw") {
    bool threw = false;
    try {
        KvCacheManager m(cfg(4, 0));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        auto c = cfg(4);
        c.high_watermark = 0.9;
        c.critical_watermark = 0.5;
        KvCacheManager m(c);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("fingerprint: order and separation sensitive") {
    const auto a = FingerprintBuilder{}.add("ab").add("c").build();
    const auto b = FingerprintBuilder{}.add("a").add("bc").build();
    const auto c = FingerprintBuilder{}.add("ab").add("c").build();
    CHECK(!(a == b));
    CHECK(a == c);
    CHECK(std::string_view(to_string(PressureLevel::critical)) == "critical");
}

TEST_CASE("alloc: block table grows per block and stats account pinned blocks") {
    KvCacheManager m(cfg(8));
    CHECK(m.eviction_policy().name() == "priority_lru");
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(1, toks(0, 10), &r).ok());
    CHECK(r.tokens_appended == 10);
    CHECK(r.blocks_allocated == 3);
    CHECK(r.tokens_reused == 0);
    CHECK(m.num_tokens(1) == 10);
    CHECK(m.block_table(1).size() == 3);
    CHECK(contents(m, 1) == toks(0, 10));
    // Decode one token at a time: fills the partial block, then allocates.
    REQUIRE(m.append_tokens(1, toks(10, 1)).ok());
    REQUIRE(m.append_tokens(1, toks(11, 1)).ok());
    CHECK(m.block_table(1).size() == 3);
    REQUIRE(m.append_tokens(1, toks(12, 1)).ok());
    CHECK(m.block_table(1).size() == 4);
    CHECK(contents(m, 1) == toks(0, 13));
    const auto s = m.stats();
    CHECK(s.total_blocks == 8);
    CHECK(s.pinned_blocks == 4);
    CHECK(s.free_blocks == 4);
    CHECK(s.cached_blocks == 0);
    CHECK(s.sequences == 1);
    CHECK(s.bytes_per_block == 400);
    CHECK(s.pinned_bytes == 1600);
    CHECK(s.blocks_allocated == 4);
    CHECK(s.utilization == 0.5);
    CHECK(m.validate());
}

TEST_CASE("errors: unknown and duplicate sequences") {
    KvCacheManager m(cfg(4));
    CHECK(m.append_tokens(9, toks(0, 1)).code() == ErrorCode::not_found);
    CHECK(m.free_sequence(9).code() == ErrorCode::not_found);
    CHECK(m.truncate(9, 0).code() == ErrorCode::not_found);
    CHECK(m.set_priority(9, 1).code() == ErrorCode::not_found);
    CHECK(m.fork_sequence(9, 10).code() == ErrorCode::not_found);
    REQUIRE(m.add_sequence(1).ok());
    CHECK(m.add_sequence(1).code() == ErrorCode::invalid_state);
    CHECK(m.fork_sequence(1, 1).code() == ErrorCode::invalid_state);
    CHECK(!m.add_sequence(1).message().empty());
    CHECK(m.block_table(9).empty());
    CHECK(m.num_tokens(9) == 0);
    CHECK(!m.has_sequence(9));
    CHECK(m.append_tokens(1, {}).ok());
    CHECK(m.ref_count(12345) == 0);
    CHECK(m.block_tokens(12345).empty());
    CHECK(m.validate());
}

TEST_CASE("free: full blocks stay cached, partial blocks return to free list") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 10)).ok());
    REQUIRE(m.free_sequence(1).ok());
    const auto s = m.stats();
    CHECK(s.cached_blocks == 2);
    CHECK(s.free_blocks == 6);
    CHECK(s.pinned_blocks == 0);
    CHECK(s.cached_bytes == 800);
    CHECK(!m.has_sequence(1));
    CHECK(m.available_blocks() == 8);
    CHECK(m.validate());
}

TEST_CASE("free: prefix caching disabled releases everything") {
    KvCacheManager m(cfg(8, 4, false));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 10)).ok());
    REQUIRE(m.free_sequence(1).ok());
    CHECK(m.stats().free_blocks == 8);
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(0, 10), &r).ok());
    CHECK(r.tokens_reused == 0);
    CHECK(m.match_prefix(kFp, toks(0, 10)) == 0);
    CHECK(m.validate());
}

TEST_CASE("prefix: reuse after the owner finished") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 10)).ok());
    const std::vector<BlockId> a(m.block_table(1).begin(), m.block_table(1).end());
    REQUIRE(m.free_sequence(1).ok());

    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    AppendResult r;
    const auto prompt = cat(toks(0, 8), toks(500, 3));
    REQUIRE(m.append_tokens(2, prompt, &r).ok());
    CHECK(r.tokens_reused == 8);
    CHECK(r.blocks_reused == 2);
    CHECK(r.blocks_allocated == 1);
    CHECK(m.cached_prefix_tokens(2) == 8);
    CHECK(m.block_table(2)[0] == a[0]);
    CHECK(m.block_table(2)[1] == a[1]);
    CHECK(contents(m, 2) == prompt);
    const auto s = m.stats();
    CHECK(s.prefix_hit_blocks == 2);
    CHECK(s.avoided_prefill_tokens == 8);
    CHECK(s.prefix_miss_lookups == 1);  // seq 1's first block; seq 2's 3-token tail is not looked up
    CHECK(s.cached_blocks == 0);
    CHECK(m.validate());
}

TEST_CASE("prefix: concurrent sharing bumps refcount") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(0, 12), &r).ok());
    CHECK(r.blocks_reused == 2);
    CHECK(r.blocks_allocated == 1);
    CHECK(m.ref_count(m.block_table(1)[0]) == 2);
    CHECK(m.stats().shared_blocks == 2);
    CHECK(m.stats().prefix_miss_lookups == 2);  // seq 1 block 0, seq 2 block 2
    REQUIRE(m.free_sequence(1).ok());
    CHECK(m.ref_count(m.block_table(2)[0]) == 1);
    CHECK(m.validate());
}

TEST_CASE("prefix: fingerprint mismatch prevents reuse") {
    KvCacheManager m(cfg(8));
    const auto other = FingerprintBuilder{}.add("model-a").add("rev2").add(std::uint64_t{16}).build();
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.add_sequence(2, {other, 0}).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(0, 8), &r).ok());
    CHECK(r.tokens_reused == 0);
    CHECK(m.stats().shared_blocks == 0);
    CHECK(m.match_prefix(other, toks(0, 8)) == 8);  // its own blocks now registered
    CHECK(m.validate());
}

TEST_CASE("prefix: hash chain requires identical ancestors") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, cat(toks(0, 4), toks(100, 4))).ok());
    // Same second block, different first block -> no hit anywhere.
    CHECK(m.match_prefix(kFp, cat(toks(50, 4), toks(100, 4))) == 0);
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, cat(toks(50, 4), toks(100, 4)), &r).ok());
    CHECK(r.tokens_reused == 0);
    CHECK(m.validate());
}

TEST_CASE("prefix: match_prefix is a pure query") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 10)).ok());
    const auto before = m.stats();
    CHECK(m.match_prefix(kFp, toks(0, 10)) == 8);
    CHECK(m.match_prefix(kFp, toks(0, 7)) == 4);
    CHECK(m.match_prefix(kFp, cat(toks(0, 4), toks(9, 4))) == 4);
    CHECK(m.match_prefix(CacheFingerprint{42}, toks(0, 10)) == 0);
    CHECK(m.match_prefix(kFp, {}) == 0);
    CHECK(m.stats().prefix_hit_blocks == before.prefix_hit_blocks);
    CHECK(m.stats().prefix_miss_lookups == before.prefix_miss_lookups);
}

TEST_CASE("prefix: only the leading run is looked up") {
    KvCacheManager m(cfg(16));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 12)).ok());
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    AppendResult r;
    // hit, miss, then content identical to seq 1 block 2 but different parent anyway
    REQUIRE(m.append_tokens(2, cat(cat(toks(0, 4), toks(77, 4)), toks(8, 4)), &r).ok());
    CHECK(r.blocks_reused == 1);
    CHECK(m.cached_prefix_tokens(2) == 4);
    CHECK(m.stats().prefix_miss_lookups == 2);  // seq 1 block 0, seq 2 block 1 (no further lookups)
    // Chunked prefill across calls keeps the streak.
    REQUIRE(m.add_sequence(3, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(3, toks(0, 4), &r).ok());
    CHECK(r.blocks_reused == 1);
    REQUIRE(m.append_tokens(3, toks(4, 8), &r).ok());
    CHECK(r.blocks_reused == 2);
    CHECK(m.cached_prefix_tokens(3) == 12);
    CHECK(m.validate());
}

TEST_CASE("fork: shares every block without allocating") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 3}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    const auto before = m.stats().blocks_allocated;
    REQUIRE(m.fork_sequence(1, 2).ok());
    REQUIRE(m.fork_sequence(1, 3, Priority{7}).ok());
    CHECK(m.stats().blocks_allocated == before);
    CHECK(m.ref_count(m.block_table(1)[0]) == 3);
    CHECK(m.stats().shared_blocks == 2);
    CHECK(m.num_tokens(3) == 8);
    CHECK(contents(m, 3) == toks(0, 8));
    // Appending at a block boundary allocates a private block, no copy.
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(100, 2), &r).ok());
    CHECK(r.copy_on_write == 0);
    CHECK(r.blocks_allocated == 1);
    CHECK(m.take_pending_copies().empty());
    CHECK(contents(m, 1) == toks(0, 8));
    CHECK(m.validate());
}

TEST_CASE("fork: copy-on-write for shared partial tail") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 6)).ok());
    REQUIRE(m.fork_sequence(1, 2).ok());
    const BlockId shared_tail = m.block_table(1)[1];
    CHECK(m.blocks_needed(2, 1) == 1);  // COW needs a block even though room remains
    CHECK(m.blocks_needed(2, 3) == 2);
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(200, 1), &r).ok());
    CHECK(r.copy_on_write == 1);
    const BlockId child_tail = m.block_table(2)[1];
    CHECK(child_tail != shared_tail);
    CHECK(m.block_table(2)[0] == m.block_table(1)[0]);
    const auto copies = m.take_pending_copies();
    REQUIRE(copies.size() == 1);
    CHECK(copies[0] == (BlockCopy{shared_tail, child_tail, 2}));
    CHECK(m.take_pending_copies().empty());
    CHECK(contents(m, 1) == toks(0, 6));
    CHECK(contents(m, 2) == cat(toks(0, 6), toks(200, 1)));
    // Parent now exclusively owns its tail: writes in place.
    REQUIRE(m.append_tokens(1, toks(6, 1), &r).ok());
    CHECK(r.copy_on_write == 0);
    CHECK(m.block_table(1)[1] == shared_tail);
    CHECK(contents(m, 1) == toks(0, 7));
    CHECK(m.stats().copy_on_write == 1);
    CHECK(m.validate());
}

TEST_CASE("alloc: out_of_blocks is all-or-nothing") {
    KvCacheManager m(cfg(3));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 5)).ok());
    CHECK(!m.can_append(1, 8));
    CHECK(m.can_append(1, 7));
    CHECK(!m.can_append(99, 1));
    const auto st = m.append_tokens(1, toks(5, 8));
    CHECK(st.code() == ErrorCode::unavailable);
    CHECK(m.num_tokens(1) == 5);
    CHECK(m.block_table(1).size() == 2);
    CHECK(m.stats().allocation_failures == 1);
    CHECK(contents(m, 1) == toks(0, 5));
    CHECK(m.append_tokens(1, toks(5, 7)).ok());
    CHECK(m.stats().free_blocks == 0);
    CHECK(m.validate());
}

TEST_CASE("evict: cached blocks are reclaimed on demand, leaves first") {
    KvCacheManager m(cfg(4));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 16)).ok());
    REQUIRE(m.free_sequence(1).ok());
    CHECK(m.stats().cached_blocks == 4);
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(2, toks(1000, 8)).ok());
    const auto s = m.stats();
    CHECK(s.evictions == 2);
    CHECK(s.cached_blocks == 2);
    // Roots survive so the leading prefix is still reusable.
    CHECK(m.match_prefix(kFp, toks(0, 16)) == 8);
    CHECK(m.validate());
}

TEST_CASE("evict: lru policy evicts oldest released chain first") {
    KvCacheManager m(cfg(4), make_eviction_policy(EvictionPolicyKind::lru));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(2, toks(100, 8)).ok());
    REQUIRE(m.free_sequence(1).ok());
    REQUIRE(m.free_sequence(2).ok());
    REQUIRE(m.add_sequence(3, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(3, toks(500, 8)).ok());
    CHECK(m.match_prefix(kFp, toks(0, 8)) == 0);
    CHECK(m.match_prefix(kFp, toks(100, 8)) == 8);
    CHECK(m.validate());
}

TEST_CASE("evict: priority-aware keeps high-priority prefixes") {
    KvCacheManager m(cfg(4));
    REQUIRE(m.add_sequence(1, {kFp, 9}).ok());  // interactive
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.add_sequence(2, {kFp, 1}).ok());  // background
    REQUIRE(m.append_tokens(2, toks(100, 8)).ok());
    REQUIRE(m.free_sequence(2).ok());
    REQUIRE(m.free_sequence(1).ok());  // high-prio released last... and more recent
    // Now make low-prio more recent: reuse and release it again.
    REQUIRE(m.add_sequence(4, {kFp, 1}).ok());
    REQUIRE(m.append_tokens(4, toks(100, 8)).ok());
    REQUIRE(m.free_sequence(4).ok());
    REQUIRE(m.add_sequence(3, {kFp, 5}).ok());
    REQUIRE(m.append_tokens(3, toks(500, 8)).ok());
    CHECK(m.match_prefix(kFp, toks(0, 8)) == 8);
    CHECK(m.match_prefix(kFp, toks(100, 8)) == 0);
    CHECK(m.validate());
}

TEST_CASE("priority: set_priority propagates to held blocks") {
    KvCacheManager m(cfg(4));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.add_sequence(2, {kFp, 3}).ok());
    REQUIRE(m.append_tokens(2, toks(100, 8)).ok());
    REQUIRE(m.set_priority(1, 9).ok());
    REQUIRE(m.free_sequence(1).ok());
    REQUIRE(m.free_sequence(2).ok());
    REQUIRE(m.add_sequence(3, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(3, toks(500, 8)).ok());
    CHECK(m.match_prefix(kFp, toks(0, 8)) == 8);
    CHECK(m.match_prefix(kFp, toks(100, 8)) == 0);
}

TEST_CASE("truncate: releases blocks and rewrites tail safely") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 10)).ok());
    CHECK(m.truncate(1, 11).code() == ErrorCode::invalid_argument);
    REQUIRE(m.truncate(1, 6).ok());
    CHECK(m.num_tokens(1) == 6);
    CHECK(m.block_table(1).size() == 2);
    CHECK(m.stats().free_blocks == 6);
    // Block 1 was full+registered with refcount 1: rewritten in place and
    // no longer served as a prefix block.
    const BlockId b1 = m.block_table(1)[1];
    REQUIRE(m.append_tokens(1, toks(900, 3)).ok());
    CHECK(m.block_table(1)[1] == b1);
    CHECK(contents(m, 1) == cat(toks(0, 6), toks(900, 3)));
    CHECK(m.match_prefix(kFp, toks(0, 8)) == 4);
    CHECK(m.match_prefix(kFp, cat(toks(0, 6), toks(900, 2))) == 8);
    REQUIRE(m.truncate(1, 0).ok());
    CHECK(m.block_table(1).empty());
    CHECK(m.validate());
}

TEST_CASE("truncate: shared full block triggers copy-on-write") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.fork_sequence(1, 2).ok());
    REQUIRE(m.truncate(2, 6).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(700, 1), &r).ok());
    CHECK(r.copy_on_write == 1);
    CHECK(contents(m, 1) == toks(0, 8));
    CHECK(contents(m, 2) == cat(toks(0, 6), toks(700, 1)));
    CHECK(m.match_prefix(kFp, toks(0, 8)) == 8);
    CHECK(m.validate());
}

TEST_CASE("truncate: back to a cached boundary restores prefix lookup") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 8)).ok());
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(2, toks(0, 4)).ok());
    CHECK(m.cached_prefix_tokens(2) == 4);
    REQUIRE(m.append_tokens(2, toks(300, 4)).ok());  // diverges
    REQUIRE(m.truncate(2, 4).ok());
    AppendResult r;
    REQUIRE(m.append_tokens(2, toks(4, 4), &r).ok());
    CHECK(r.blocks_reused == 1);
    CHECK(m.cached_prefix_tokens(2) == 8);
    CHECK(m.validate());
}

TEST_CASE("telemetry: events and pressure transitions") {
    auto c = cfg(10);
    c.high_watermark = 0.5;
    c.critical_watermark = 0.9;
    KvCacheManager m(c);
    std::map<CacheEvent::Kind, int> counts;
    std::vector<PressureLevel> levels;
    m.set_event_listener([&](const CacheEvent& e) {
        ++counts[e.kind];
        if (e.kind == CacheEvent::Kind::pressure_changed) levels.push_back(e.pressure);
    });
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 20)).ok());  // 5 blocks -> high
    CHECK(m.pressure() == PressureLevel::high);
    REQUIRE(m.append_tokens(1, toks(20, 16)).ok());  // 9 blocks -> critical
    CHECK(m.pressure() == PressureLevel::critical);
    REQUIRE(m.free_sequence(1).ok());
    CHECK(m.pressure() == PressureLevel::normal);
    REQUIRE(m.add_sequence(2, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(2, toks(0, 8)).ok());  // 2 reused
    REQUIRE(m.add_sequence(3, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(3, toks(1000, 32)).ok());  // needs 8: 1 free + evict 7 cached
    CHECK(counts[CacheEvent::Kind::allocated] == 9 + 8);
    CHECK(counts[CacheEvent::Kind::reused] == 2);
    CHECK(counts[CacheEvent::Kind::evicted] == 7);
    REQUIRE(levels.size() == 4);
    CHECK(levels[0] == PressureLevel::high);
    CHECK(levels[1] == PressureLevel::critical);
    CHECK(levels[2] == PressureLevel::normal);
    CHECK(levels[3] == PressureLevel::critical);
    CHECK(m.stats().pressure == PressureLevel::critical);
    CHECK(m.validate());
}

TEST_CASE("clear_cached: returns cached blocks to the free list") {
    KvCacheManager m(cfg(8));
    REQUIRE(m.add_sequence(1, {kFp, 0}).ok());
    REQUIRE(m.append_tokens(1, toks(0, 12)).ok());
    REQUIRE(m.free_sequence(1).ok());
    CHECK(m.clear_cached() == 3);
    CHECK(m.stats().free_blocks == 8);
    CHECK(m.match_prefix(kFp, toks(0, 12)) == 0);
    CHECK(m.validate());
}

TEST_CASE("zero-capacity pool is valid and rejects appends") {
    KvCacheManager m(cfg(0));
    REQUIRE(m.add_sequence(1).ok());
    CHECK(m.append_tokens(1, toks(0, 1)).code() == ErrorCode::unavailable);
    CHECK(m.pressure() == PressureLevel::normal);
    CHECK(m.stats().utilization == 0.0);
    CHECK(m.validate());
}

TEST_CASE("stress: randomized ops keep invariants and contents") {
    for (const auto kind : {EvictionPolicyKind::priority_aware, EvictionPolicyKind::lru}) {
        std::mt19937 rng(1234);
        KvCacheManager m(cfg(24, 4), make_eviction_policy(kind));
        std::map<SequenceId, std::vector<TokenId>> shadow;
        SequenceId next = 1;
        int ok_ops = 0;
        int rejected = 0;
        for (int step = 0; step < 4000; ++step) {
            const int op = static_cast<int>(rng() % 100);
            if (shadow.empty() || op < 10) {
                const SequenceId id = next++;
                REQUIRE(m.add_sequence(id, {kFp, static_cast<Priority>(rng() % 4)}).ok());
                shadow[id] = {};
                // Shared system prompt drives prefix reuse.
                std::vector<TokenId> p = toks(0, 4 * (rng() % 4));
                if (m.append_tokens(id, p).ok()) {
                    shadow[id] = p;
                    ++ok_ops;
                } else {
                    ++rejected;
                }
                continue;
            }
            auto it = shadow.begin();
            std::advance(it, static_cast<long>(rng() % shadow.size()));
            const SequenceId id = it->first;
            if (op < 55) {
                const std::size_t n = 1 + rng() % 7;
                std::vector<TokenId> t(n);
                for (auto& x : t) x = static_cast<TokenId>(rng() % 3);  // tiny vocab -> collisions of content
                const auto before = m.num_tokens(id);
                if (m.append_tokens(id, t).ok()) {
                    it->second.insert(it->second.end(), t.begin(), t.end());
                    ++ok_ops;
                } else {
                    ++rejected;
                    CHECK(m.num_tokens(id) == before);
                    CHECK(contents(m, id) == it->second);
                    // Scheduler-style preemption: drop the sequence to relieve pressure.
                    REQUIRE(m.free_sequence(id).ok());
                    shadow.erase(it);
                }
            } else if (op < 62) {
                const SequenceId child = next++;
                REQUIRE(m.fork_sequence(id, child).ok());
                shadow[child] = it->second;
            } else if (op < 74) {
                const std::size_t len = it->second.empty() ? 0 : rng() % (it->second.size() + 1);
                REQUIRE(m.truncate(id, len).ok());
                it->second.resize(len);
            } else {
                REQUIRE(m.free_sequence(id).ok());
                shadow.erase(it);
            }
            if (step % 7 == 0) (void)m.take_pending_copies();
            REQUIRE(m.validate());
        }
        for (const auto& [id, t] : shadow) CHECK(contents(m, id) == t);
        const std::string policy_name{m.eviction_policy().name()};
        MESSAGE("stress[" << policy_name << "]: ok=" << ok_ops << " rejected=" << rejected);
        CHECK(ok_ops > 1000);
        CHECK(rejected > 0);  // pool pressure actually exercised
        const auto s = m.stats();
        CHECK(s.prefix_hit_blocks > 0);
        CHECK(s.copy_on_write > 0);
        CHECK(s.evictions > 0);
        for (const auto& [id, t] : shadow) REQUIRE(m.free_sequence(id).ok());
        CHECK(m.stats().pinned_blocks == 0);
        CHECK(m.validate());
    }
}
