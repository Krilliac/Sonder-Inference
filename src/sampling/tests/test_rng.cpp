#include <cstdint>
#include <vector>

#include "sonder/sampling/rng.hpp"
#include "sampling_test_util.hpp"

using sonder::inference::sampling::Rng;

TEST_CASE("same seed yields identical sequence") {
    Rng a(42);
    Rng b(42);
    for (int i = 0; i < 1000; ++i) {
        CHECK(a.next_u64() == b.next_u64());
    }
}

TEST_CASE("different seeds diverge") {
    Rng a(1);
    Rng b(2);
    int equal = 0;
    for (int i = 0; i < 100; ++i) {
        equal += a.next_u64() == b.next_u64() ? 1 : 0;
    }
    CHECK(equal == 0);
}

TEST_CASE("reseed restarts the sequence") {
    Rng a(7);
    std::vector<std::uint64_t> first;
    for (int i = 0; i < 10; ++i) first.push_back(a.next_u64());
    a.reseed(7);
    for (int i = 0; i < 10; ++i) CHECK(a.next_u64() == first[static_cast<std::size_t>(i)]);
}

TEST_CASE("uniform is in the unit interval with sane mean") {
    Rng r(123);
    double sum = 0.0;
    const int n = 100000;
    for (int i = 0; i < n; ++i) {
        const double u = r.uniform();
        CHECK((u >= 0.0 && u < 1.0));
        sum += u;
    }
    CHECK_NEAR(sum / n, 0.5, 0.01);
}

TEST_CASE("known-answer vector pins cross-platform output") {
    // Reference values computed independently (Python) for SplitMix64-seeded
    // xoshiro256** with seed 0. Any change here breaks seed reproducibility.
    Rng r(0);
    CHECK(r.next_u64() == 0x99EC5F36CB75F2B4ULL);
    CHECK(r.next_u64() == 0xBF6E1F784956452AULL);
    CHECK(r.next_u64() == 0x1A5F849D4933E6E0ULL);
}

TEST_CASE("entropy_seed produces varying seeds") {
    const auto a = Rng::entropy_seed();
    const auto b = Rng::entropy_seed();
    const auto c = Rng::entropy_seed();
    CHECK((!(a == b && b == c)));
}
