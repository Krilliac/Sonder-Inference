#include <doctest/doctest.h>

#include "engine/kv_capacity_adapter.hpp"

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

#include <cstdint>
#include <string>
#include <vector>

#include "sonder/inference/scheduler/scheduler.hpp"

namespace cache = sonder::inference::cache;
namespace sched = sonder::inference::scheduler;
using sonder::inference::KvCacheCapacityAdapter;

namespace {

cache::KvCacheConfig small_config(std::uint32_t blocks) {
    cache::KvCacheConfig c;
    c.block_size_tokens = 4;
    c.num_blocks = blocks;
    return c;
}

std::vector<cache::TokenId> tokens(std::size_t n, cache::TokenId base = 1) {
    std::vector<cache::TokenId> t(n);
    for (std::size_t i = 0; i < n; ++i) t[i] = base + static_cast<cache::TokenId>(i);
    return t;
}

}  // namespace

TEST_CASE("kv_adapter_reports_cache_geometry") {
    cache::KvCacheManager mgr(small_config(8));
    KvCacheCapacityAdapter kv(mgr);
    CHECK(kv.block_size_tokens() == 4);
    CHECK(kv.total_blocks() == 8);
    CHECK(kv.free_blocks() == 8);
}

TEST_CASE("kv_adapter_reservations_are_all_or_nothing") {
    cache::KvCacheManager mgr(small_config(8));
    KvCacheCapacityAdapter kv(mgr);
    CHECK(kv.try_reserve(1, 5));
    CHECK(kv.free_blocks() == 3);
    CHECK_FALSE(kv.try_reserve(2, 4));
    CHECK(kv.blocks_held(2) == 0);
    CHECK(kv.try_reserve(2, 3));
    CHECK(kv.free_blocks() == 0);
    kv.release(1);
    CHECK(kv.blocks_held(1) == 0);
    CHECK(kv.free_blocks() == 5);
}

TEST_CASE("kv_adapter_does_not_double_count_materialised_blocks") {
    cache::KvCacheManager mgr(small_config(8));
    KvCacheCapacityAdapter kv(mgr);
    REQUIRE(kv.try_reserve(7, 3));
    CHECK(kv.free_blocks() == 5);
    REQUIRE(mgr.add_sequence(7).ok());
    const auto t = tokens(8);  // two blocks
    REQUIRE(mgr.append_tokens(7, t).ok());
    CHECK(kv.outstanding() == 1);
    CHECK(kv.free_blocks() == 5);  // 6 available in cache - 1 still pending
}

TEST_CASE("kv_adapter_drives_scheduler_admission") {
    cache::KvCacheManager mgr(small_config(4));  // 16 tokens total
    KvCacheCapacityAdapter kv(mgr);
    sched::SimClock clock;
    sched::SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    sched::Scheduler s(cfg, clock, kv);

    sched::RequestSpec a;
    a.id = 1;
    a.workload = sched::WorkloadClass::InteractiveUser;
    a.task_id = "a";
    a.prompt_tokens = 8;
    a.max_new_tokens = 2;
    CHECK(s.submit(a).accepted);

    sched::RequestSpec huge = a;
    huge.id = 2;
    huge.task_id = "huge";
    huge.prompt_tokens = 64;  // can never fit in 4 blocks of 4 tokens
    CHECK_FALSE(s.submit(huge).accepted);

    const sched::StepPlan plan = s.plan_step();
    CHECK_FALSE(plan.work.empty());
    CHECK(kv.blocks_held(1) >= 2);
    CHECK(kv.free_blocks() + kv.blocks_held(1) <= kv.total_blocks());
}

#else

TEST_CASE("kv_adapter_skipped_without_cache_and_scheduler_modules") {
    CHECK(true);
}

#endif
