#include "sonder/inference/cache/eviction_policy.hpp"

#include <ostream>  // doctest stringifies std::string_view; MSVC needs the full definition

#include <doctest/doctest.h>

using namespace sonder::inference::cache;

namespace {
EvictionCandidate cand(BlockId b, std::uint64_t last_use, Priority prio = 0, std::uint32_t depth = 0,
                       std::uint64_t recompute = 0) {
    EvictionCandidate c;
    c.block = b;
    c.last_use = last_use;
    c.priority = prio;
    c.depth = depth;
    c.recompute_tokens = recompute;
    return c;
}
}  // namespace

TEST_CASE("lru: evicts least recently used first") {
    LruEvictionPolicy p;
    CHECK(p.empty());
    p.insert(cand(1, 30));
    p.insert(cand(2, 10));
    p.insert(cand(3, 20));
    CHECK(p.size() == 3);
    CHECK(p.pop_victim() == BlockId{2});
    CHECK(p.pop_victim() == BlockId{3});
    CHECK(p.pop_victim() == BlockId{1});
    CHECK(!p.pop_victim().has_value());
}

TEST_CASE("lru: erase and reinsert updates position") {
    LruEvictionPolicy p;
    p.insert(cand(1, 1));
    p.insert(cand(2, 2));
    p.erase(1);
    p.erase(99);  // unknown is a no-op
    CHECK(p.size() == 1);
    p.insert(cand(1, 3));
    p.insert(cand(2, 5));  // reinsert replaces old entry
    CHECK(p.size() == 2);
    CHECK(p.pop_victim() == BlockId{1});
    CHECK(p.pop_victim() == BlockId{2});
}

TEST_CASE("lru: ties broken by insertion order") {
    LruEvictionPolicy p;
    p.insert(cand(7, 5));
    p.insert(cand(3, 5));
    CHECK(p.pop_victim() == BlockId{7});
    CHECK(p.pop_victim() == BlockId{3});
}

TEST_CASE("priority: lowest priority evicted first regardless of recency") {
    PriorityAwareEvictionPolicy p;
    p.insert(cand(1, 1, /*prio*/ 5));
    p.insert(cand(2, 100, /*prio*/ 1));
    p.insert(cand(3, 50, /*prio*/ 9));
    CHECK(p.pop_victim() == BlockId{2});
    CHECK(p.pop_victim() == BlockId{1});
    CHECK(p.pop_victim() == BlockId{3});
}

TEST_CASE("priority: deeper (leaf) blocks before roots at equal priority") {
    PriorityAwareEvictionPolicy p;
    p.insert(cand(10, 1, 0, /*depth*/ 0));
    p.insert(cand(11, 2, 0, /*depth*/ 1));
    p.insert(cand(12, 3, 0, /*depth*/ 2));
    CHECK(p.pop_victim() == BlockId{12});
    CHECK(p.pop_victim() == BlockId{11});
    CHECK(p.pop_victim() == BlockId{10});
}

TEST_CASE("priority: cheaper recompute then LRU as tie breakers") {
    PriorityAwareEvictionPolicy p;
    p.insert(cand(1, 1, 0, 3, /*recompute*/ 64));
    p.insert(cand(2, 9, 0, 3, /*recompute*/ 16));
    p.insert(cand(3, 5, 0, 3, /*recompute*/ 16));
    CHECK(p.pop_victim() == BlockId{3});
    CHECK(p.pop_victim() == BlockId{2});
    CHECK(p.pop_victim() == BlockId{1});
}

TEST_CASE("priority: erase removes candidate") {
    PriorityAwareEvictionPolicy p;
    p.insert(cand(1, 1));
    p.insert(cand(2, 2));
    p.erase(1);
    CHECK(p.size() == 1);
    CHECK(p.pop_victim() == BlockId{2});
    CHECK(p.empty());
}

TEST_CASE("factory: names") {
    CHECK(make_eviction_policy(EvictionPolicyKind::lru)->name() == "lru");
    CHECK(make_eviction_policy(EvictionPolicyKind::priority_aware)->name() == "priority_lru");
}
