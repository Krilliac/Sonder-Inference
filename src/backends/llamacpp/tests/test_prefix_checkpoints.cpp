// Prefix-checkpoint policy tests: placement, hybrid detection, host-tier byte
// bound / LRU, and counters. No model weights, no llama.cpp calls.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "prefix_checkpoints.hpp"
#include "sonder/inference/backends/llamacpp.hpp"

using namespace sonder::backends::llamacpp;
using sonder::inference::ModelArchitecture;
using sonder::inference::ModelMetadata;

namespace {

using Positions = std::vector<std::size_t>;

PrefixCheckpointConfig BaseConfig() {
    PrefixCheckpointConfig cfg;
    cfg.spacing_tokens = 2048;
    cfg.dense_spacing_tokens = 1024;
    cfg.min_prompt_tokens = 512;
    cfg.tail_offset_tokens = 16;
    cfg.max_per_request = 8;
    return cfg;
}

std::vector<Token> Seq(std::size_t n, Token base = 1) {
    std::vector<Token> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = base + static_cast<Token>(i);
    return v;
}

std::vector<std::uint8_t> Blob(std::size_t n, std::uint8_t fill = 0xAB) { return std::vector<std::uint8_t>(n, fill); }

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

}  // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: SanitizeConfig clamps every field into range") {
    PrefixCheckpointConfig cfg;
    cfg.spacing_tokens = 0;
    cfg.dense_spacing_tokens = 1;
    cfg.tail_offset_tokens = 0;
    cfg.max_per_request = 0;
    cfg.cache_ram_mb = 1u << 30;
    cfg.host_max_entries = 1u << 30;
    const auto s = SanitizeConfig(cfg);
    CHECK(s.spacing_tokens == PrefixCheckpointConfig{}.spacing_tokens);
    CHECK(s.dense_spacing_tokens == PrefixCheckpointConfig::kMinSpacingTokens);
    CHECK(s.tail_offset_tokens == PrefixCheckpointConfig{}.tail_offset_tokens);
    CHECK(s.max_per_request == PrefixCheckpointConfig{}.max_per_request);
    CHECK(s.cache_ram_mb == PrefixCheckpointConfig::kMaxHostCacheMb);
    CHECK(s.host_max_entries == PrefixCheckpointConfig::kMaxHostEntries);

    PrefixCheckpointConfig big;
    big.spacing_tokens = 0xFFFFFFFFu;
    big.max_per_request = 100000;
    const auto b = SanitizeConfig(big);
    CHECK(b.spacing_tokens == PrefixCheckpointConfig::kMaxSpacingTokens);
    CHECK(b.max_per_request == PrefixCheckpointConfig::kMaxPerRequestLimit);

    // A sane config passes through untouched.
    const auto ok = SanitizeConfig(BaseConfig());
    CHECK(ok.spacing_tokens == 2048u);
    CHECK(ok.dense_spacing_tokens == 1024u);
    CHECK(ok.max_per_request == 8u);
    // Conservative defaults: host tier off (0), 1-2K spacing, denser for hybrid.
    CHECK(PrefixCheckpointConfig{}.cache_ram_mb == 0u);
    CHECK(PrefixCheckpointConfig{}.dense_spacing_tokens < PrefixCheckpointConfig{}.spacing_tokens);

    // Backend option defaults stay aligned with PrefixCheckpointConfig.
    const LoadOptions lo;
    const sonder::inference::LlamaCppBackendOptions bo;
    CHECK(lo.cache_ram_mib == 0u);
    CHECK(bo.cache_ram_mib == 0u);
    CHECK(lo.checkpoint_spacing_tokens == 2048u);
    CHECK(lo.checkpoint_dense_spacing_tokens == 1024u);
    CHECK(lo.checkpoint_max_per_slot == 8u);
    CHECK(bo.checkpoint_spacing_tokens == 2048u);
    CHECK(bo.checkpoint_dense_spacing_tokens == 1024u);
    CHECK(bo.checkpoint_max_per_slot == 8u);
}

// ---------------------------------------------------------------------------
// Hybrid detection / mode
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: stateful architecture names follow llama.cpp v1.0.0") {
    for (const char* name : {"jamba", "falcon-h1", "plamo2", "granitehybrid", "lfm2", "lfm2moe", "lfm3", "nemotron_h",
                             "nemotron_h_moe", "metis16", "qwen3next", "kimi-linear", "bailingmoe3", "kimi-k3",
                             "kimi-k31", "glm5-next", "qwen35", "qwen35moe", "clover_moe", "qwen4exp", "deepseek4",
                             "deepseek41", "minimax-01", "QWEN35MOE"}) {
        CAPTURE(name);
        CHECK(IsHybridArchitectureName(name));
        CHECK_FALSE(IsRecurrentArchitectureName(name));
        CHECK(IsStatefulArchitectureName(name));
    }
    for (const char* name : {"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7", "Mamba2", "ARWKV7"}) {
        CAPTURE(name);
        CHECK(IsRecurrentArchitectureName(name));
        CHECK_FALSE(IsHybridArchitectureName(name));
        CHECK(IsStatefulArchitectureName(name));
    }
    for (const char* name : {"", "llama", "qwen2", "qwen3", "qwen35x", "gemma3", "mamba22", " qwen35", "lfm",
                             "this_architecture_name_is_far_too_long_for_buffer"}) {
        CAPTURE(name);
        CHECK_FALSE(IsHybridArchitectureName(name));
        CHECK_FALSE(IsRecurrentArchitectureName(name));
        CHECK_FALSE(IsStatefulArchitectureName(name));
    }
}

TEST_CASE("prefix checkpoints: hybrid detection from model metadata") {
    // Named only, no recurrent keys at all.
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "qwen35"}}) == ModelArchitecture::hybrid);
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "qwen35moe"}}) == ModelArchitecture::hybrid);
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "nemotron_h"}}) == ModelArchitecture::hybrid);
    // Pure recurrent stays recurrent (including v1.0.0's arwkv7).
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "mamba"}}) == ModelArchitecture::recurrent);
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "arwkv7"}}) == ModelArchitecture::recurrent);
    // Unknown name with SSM + attention keys (Bonsai-2 style) is hybrid.
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "bonsai2"},
                                        {"bonsai2.ssm.state_size", "128"},
                                        {"bonsai2.attention.head_count", "16"}}) == ModelArchitecture::hybrid);
    // Plain attention.
    CHECK(DetectCheckpointArchitecture({{"general.architecture", "llama"}, {"llama.attention.head_count", "32"}}) ==
          ModelArchitecture::attention_only);
    CHECK(DetectCheckpointArchitecture({}) == ModelArchitecture::attention_only);
}

TEST_CASE("prefix checkpoints: mode selection") {
    const auto cfg = BaseConfig();
    const ModelMetadata llama{{"general.architecture", "llama"}};
    const ModelMetadata hybrid{{"general.architecture", "qwen35"}};
    const ModelMetadata ssm{{"general.architecture", "mamba2"}};

    CHECK(ClassifyCheckpointMode(llama, false, std::nullopt, cfg) == CheckpointMode::kOff);
    CHECK(ClassifyCheckpointMode(llama, true, std::nullopt, cfg) == CheckpointMode::kSparse);
    CHECK(ClassifyCheckpointMode(hybrid, false, std::nullopt, cfg) == CheckpointMode::kDense);
    CHECK(ClassifyCheckpointMode(hybrid, true, std::nullopt, cfg) == CheckpointMode::kDense);
    CHECK(ClassifyCheckpointMode(ssm, false, std::nullopt, cfg) == CheckpointMode::kDense);

    // The loaded model's own answer beats the metadata.
    CHECK(ClassifyCheckpointMode(llama, false, true, cfg) == CheckpointMode::kDense);
    CHECK(ClassifyCheckpointMode(hybrid, false, false, cfg) == CheckpointMode::kOff);
    CHECK(ClassifyCheckpointMode(hybrid, true, false, cfg) == CheckpointMode::kSparse);

    // Forcing checkpoints on plain attention models uses the sparse spacing.
    auto forced = cfg;
    forced.force_attention_only = true;
    CHECK(ClassifyCheckpointMode(llama, false, std::nullopt, forced) == CheckpointMode::kSparse);

    // Disabled wins over everything.
    auto off = cfg;
    off.enabled = false;
    CHECK(ClassifyCheckpointMode(hybrid, true, true, off) == CheckpointMode::kOff);

    CHECK(SpacingFor(CheckpointMode::kDense, cfg) == 1024u);
    CHECK(SpacingFor(CheckpointMode::kSparse, cfg) == 2048u);
    CHECK(SpacingFor(CheckpointMode::kOff, cfg) == 0u);
    CHECK(ToString(CheckpointMode::kDense) == "dense");
    CHECK(ToString(CheckpointMode::kSparse) == "sparse");
    CHECK(ToString(CheckpointMode::kOff) == "off");
}

TEST_CASE("prefix checkpoints: SWA model with swa_full gives kOff and keeps full radix budget") {
    const auto cfg = BaseConfig();
    const ModelMetadata gemma_swa{{"general.architecture", "gemma3"}, {"gemma3.attention.sliding_window", "1024"}};
    const ModelMetadata hybrid{{"general.architecture", "qwen35"}};

    // Default SwaKvMode::Full (`swa_full = true`): llama_memory_seq_rm rolls
    // SWA back directly, so checkpoints are kOff and 100% of cache_ram_mb stays
    // in the L2 radix cache.
    const CheckpointMode full_mode = ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, std::nullopt, cfg,
                                                            /*swa_full=*/true);
    CHECK(full_mode == CheckpointMode::kOff);
    CHECK(SpacingFor(full_mode, cfg) == 0u);
    CHECK(PlanCheckpointPositions(0, 8192, full_mode, cfg).empty());
    const HostCacheBudgetSplit full_budget = SplitHostCacheBudget(1024, full_mode);
    CHECK(full_budget.checkpoint_bytes == 0u);
    CHECK(full_budget.radix_bytes == (std::size_t{1024} << 20));

    // Compact SWA (`swa_full = false`): keeps kSparse checkpoints and splits
    // host budget.
    const CheckpointMode compact_mode = ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, std::nullopt, cfg,
                                                               /*swa_full=*/false);
    CHECK(compact_mode == CheckpointMode::kSparse);
    CHECK(SpacingFor(compact_mode, cfg) == 2048u);
    CHECK_FALSE(PlanCheckpointPositions(0, 8192, compact_mode, cfg).empty());
    const HostCacheBudgetSplit compact_budget = SplitHostCacheBudget(1024, compact_mode);
    CHECK(compact_budget.checkpoint_bytes == (std::size_t{128} << 20));
    CHECK(compact_budget.radix_bytes == (std::size_t{896} << 20));

    // Hybrid/recurrent models still get kDense even when swa_full is true.
    CHECK(ClassifyCheckpointMode(hybrid, /*sliding_window=*/true, std::nullopt, cfg, /*swa_full=*/true) ==
          CheckpointMode::kDense);
    CHECK(ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, /*runtime_stateful=*/true, cfg,
                                 /*swa_full=*/true) == CheckpointMode::kDense);

    // Explicit force_attention_only still enables kSparse even when swa_full is true.
    auto forced = cfg;
    forced.force_attention_only = true;
    CHECK(ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, std::nullopt, forced, /*swa_full=*/true) ==
          CheckpointMode::kSparse);
}

TEST_CASE("prefix checkpoints: compact SWA without restore consumer gives kOff") {
    const ModelMetadata gemma_swa{{"general.architecture", "gemma3"}, {"gemma3.attention.sliding_window", "1024"}};

    // Compact SWA (`swa_full=false`, `slot_restore_allowed=false`) with
    // `cache_ram_mb=0` has neither slot nor host restore consumer -> kOff and
    // zero planned positions.
    auto cfg_no_ram = BaseConfig();
    cfg_no_ram.cache_ram_mb = 0;
    const CheckpointMode off_mode = ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, std::nullopt, cfg_no_ram,
                                                           /*swa_full=*/false, /*slot_restore_allowed=*/false);
    CHECK(off_mode == CheckpointMode::kOff);
    CHECK(PlanCheckpointPositions(0, 8192, off_mode, cfg_no_ram).empty());

    // With `cache_ram_mb > 0`, the host tier is active so a restore consumer
    // exists -> stays kSparse and plans checkpoint positions.
    auto cfg_with_ram = BaseConfig();
    cfg_with_ram.cache_ram_mb = 256;
    const CheckpointMode sparse_mode =
        ClassifyCheckpointMode(gemma_swa, /*sliding_window=*/true, std::nullopt, cfg_with_ram,
                               /*swa_full=*/false, /*slot_restore_allowed=*/false);
    CHECK(sparse_mode == CheckpointMode::kSparse);
    CHECK_FALSE(PlanCheckpointPositions(0, 8192, sparse_mode, cfg_with_ram).empty());
    CHECK(SplitHostCacheBudget(cfg_with_ram.cache_ram_mb, sparse_mode).checkpoint_bytes > 0u);
}

TEST_CASE("prefix checkpoints: Clear and Configure reset host tier counters") {
    const std::size_t one = CheckpointHostTier::EntryBytes(8, 1000, 0);
    CheckpointHostTier tier(one * 2, 10);
    REQUIRE(tier.Put(Seq(8, 100), Blob(1000)));
    REQUIRE(tier.Put(Seq(8, 200), Blob(1000)));
    REQUIRE(tier.Put(Seq(8, 300), Blob(1000)));   // triggers eviction
    CHECK_FALSE(tier.Put(Seq(8, 400), Blob(one * 3)));  // triggers rejection

    PrefixCheckpointStats before;
    tier.FillStats(before);
    REQUIRE(before.host_puts == 3u);
    REQUIRE(before.host_evictions == 1u);
    REQUIRE(before.host_rejected == 1u);

    tier.Clear();
    PrefixCheckpointStats after_clear;
    tier.FillStats(after_clear);
    CHECK(after_clear.host_puts == 0u);
    CHECK(after_clear.host_evictions == 0u);
    CHECK(after_clear.host_rejected == 0u);
    CHECK(after_clear.host_bytes == 0u);
    CHECK(after_clear.host_entries == 0u);

    // Populate again and verify Configure() also resets counters (e.g. Unload/Load).
    REQUIRE(tier.Put(Seq(8, 500), Blob(1000)));
    REQUIRE(tier.Put(Seq(8, 600), Blob(1000)));
    REQUIRE(tier.Put(Seq(8, 700), Blob(1000)));
    CHECK_FALSE(tier.Put(Seq(8, 800), Blob(one * 3)));

    tier.Configure(one * 4, 10);
    PrefixCheckpointStats after_cfg;
    tier.FillStats(after_cfg);
    CHECK(after_cfg.host_puts == 0u);
    CHECK(after_cfg.host_evictions == 0u);
    CHECK(after_cfg.host_rejected == 0u);
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: placement for various prompt lengths and spacings") {
    const auto cfg = BaseConfig();

    SUBCASE("short prompts and off/disabled get nothing") {
        CHECK(PlanCheckpointPositions(0, 100, CheckpointMode::kDense, cfg).empty());
        CHECK(PlanCheckpointPositions(0, 511, CheckpointMode::kDense, cfg).empty());
        CHECK(PlanCheckpointPositions(0, 100000, CheckpointMode::kOff, cfg).empty());
        auto off = cfg;
        off.enabled = false;
        CHECK(PlanCheckpointPositions(0, 100000, CheckpointMode::kDense, off).empty());
    }
    SUBCASE("minimum-length prompt gets just the tail") {
        CHECK(PlanCheckpointPositions(0, 512, CheckpointMode::kDense, cfg) == Positions{496});
    }
    SUBCASE("dense spacing: spaced positions plus the tail") {
        CHECK(PlanCheckpointPositions(0, 4096, CheckpointMode::kDense, cfg) == Positions{1024, 2048, 3072, 4080});
    }
    SUBCASE("sparse spacing is coarser") {
        CHECK(PlanCheckpointPositions(0, 10000, CheckpointMode::kSparse, cfg) ==
              Positions{2048, 4096, 6144, 8192, 9984});
    }
    SUBCASE("a spaced position within spacing/2 of the tail is dropped") {
        // tail = 3084; 3072 is 12 tokens before it.
        CHECK(PlanCheckpointPositions(0, 3100, CheckpointMode::kDense, cfg) == Positions{1024, 2048, 3084});
        // tail = 4079 (n=4095): 3072 is 1007 before it (>= 512), kept.
        CHECK(PlanCheckpointPositions(0, 4095, CheckpointMode::kDense, cfg) == Positions{1024, 2048, 3072, 4079});
    }
    SUBCASE("a warm start only plans positions after the cached prefix") {
        CHECK(PlanCheckpointPositions(2048, 4096, CheckpointMode::kDense, cfg) == Positions{3072, 4080});
        // prefill_start exactly on a multiple: that position is not re-planned.
        CHECK(PlanCheckpointPositions(1024, 2200, CheckpointMode::kDense, cfg) == Positions{2184});
        // Nothing left to checkpoint when the tail is not past the start.
        CHECK(PlanCheckpointPositions(590, 600, CheckpointMode::kDense, [&] {
                  auto c = cfg;
                  c.min_prompt_tokens = 0;
                  return c;
              }()).empty());
    }
    SUBCASE("prompt shorter than the tail offset") {
        auto c = cfg;
        c.min_prompt_tokens = 0;
        CHECK(PlanCheckpointPositions(0, 10, CheckpointMode::kDense, c).empty());
        CHECK(PlanCheckpointPositions(0, 16, CheckpointMode::kDense, c).empty());
        CHECK(PlanCheckpointPositions(0, 17, CheckpointMode::kDense, c) == Positions{1});
    }
    SUBCASE("the per-request cap keeps the tail and the newest spaced positions") {
        auto c = cfg;
        c.max_per_request = 3;
        CHECK(PlanCheckpointPositions(0, 10000, CheckpointMode::kSparse, c) == Positions{6144, 8192, 9984});
        c.max_per_request = 1;
        CHECK(PlanCheckpointPositions(0, 10000, CheckpointMode::kSparse, c) == Positions{9984});
    }
    SUBCASE("denser for hybrid than for sparse on the same prompt") {
        const auto dense = PlanCheckpointPositions(0, 20000, CheckpointMode::kDense, cfg);
        const auto sparse = PlanCheckpointPositions(0, 20000, CheckpointMode::kSparse, cfg);
        CHECK(dense.size() >= sparse.size());
        CHECK(dense.size() > 2u);
    }
    SUBCASE("a huge prompt does no work proportional to its length") {
        const auto p = PlanCheckpointPositions(0, kSizeMax / 2, CheckpointMode::kDense, cfg);
        REQUIRE(p.size() == cfg.max_per_request);
        CHECK(p.back() == kSizeMax / 2 - 16);
        const auto q = PlanCheckpointPositions(0, kSizeMax, CheckpointMode::kSparse, cfg);
        REQUIRE(q.size() == cfg.max_per_request);
        CHECK(q.back() == kSizeMax - 16);
        CHECK(PlanCheckpointPositions(kSizeMax, kSizeMax, CheckpointMode::kDense, cfg).empty());
        CHECK(PlanCheckpointPositions(5000, 100, CheckpointMode::kDense, cfg).empty());
    }
}

TEST_CASE("prefix checkpoints: placement invariants over many shapes") {
    std::size_t non_empty = 0;
    std::size_t total = 0;
    for (const CheckpointMode mode : {CheckpointMode::kSparse, CheckpointMode::kDense}) {
        for (const std::uint32_t spacing : {64u, 100u, 1024u, 2048u, 5000u}) {
            for (const std::uint32_t cap : {1u, 2u, 3u, 8u, 64u}) {
                auto cfg = BaseConfig();
                cfg.spacing_tokens = spacing;
                cfg.dense_spacing_tokens = spacing;
                cfg.max_per_request = cap;
                cfg.min_prompt_tokens = 0;
                for (std::size_t n : {0u, 1u, 15u, 16u, 17u, 63u, 64u, 65u, 500u, 1023u, 1024u, 1025u, 4096u, 9999u,
                                      20000u, 123457u}) {
                    for (std::size_t start : {0u, 1u, 16u, 1000u, 4096u, 20000u}) {
                        const auto p = PlanCheckpointPositions(start, n, mode, cfg);
                        ++total;
                        non_empty += p.empty() ? 0 : 1;
                        CAPTURE(spacing);
                        CAPTURE(cap);
                        CAPTURE(n);
                        CAPTURE(start);
                        CHECK(p.size() <= cap);
                        for (std::size_t i = 0; i < p.size(); ++i) {
                            CHECK(p[i] > start);
                            CHECK(p[i] < n);
                            if (i > 0) CHECK(p[i] > p[i - 1]);
                        }
                        if (!p.empty()) {
                            // The tail is always present when anything is planned.
                            CHECK(p.back() == n - 16);
                        }
                        // Planned exactly when there is room past the cached prefix.
                        CHECK(p.empty() == !(n > 16 && n - 16 > start));
                    }
                }
            }
        }
    }
    CHECK(total > 1000u);
    CHECK(non_empty > total / 4);
}

// ---------------------------------------------------------------------------
// Host tier
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: host tier is off by default and rejects when off") {
    CheckpointHostTier tier;
    CHECK_FALSE(tier.enabled());
    CHECK_FALSE(tier.Put(Seq(8), Blob(100)));
    CHECK(tier.size() == 0u);
    CHECK(tier.bytes() == 0u);
    CHECK_FALSE(tier.Lookup(Seq(8), 8).has_value());
    PrefixCheckpointStats st;
    tier.FillStats(st);
    CHECK(st.host_rejected == 1u);
    CHECK(st.host_puts == 0u);

    CheckpointHostTier zero_entries(1 << 20, 0);
    CHECK_FALSE(zero_entries.enabled());
    CHECK_FALSE(zero_entries.Put(Seq(8), Blob(100)));

    tier.ConfigureMiB(0, 64);
    CHECK_FALSE(tier.enabled());
    tier.ConfigureMiB(2, 64);
    CHECK(tier.enabled());
    CHECK(tier.max_bytes() == std::size_t{2} << 20);
    // Huge MiB clamps to kMaxHostCacheMb without overflowing.
    tier.ConfigureMiB(std::numeric_limits<std::uint32_t>::max(), kSizeMax);
    CHECK(tier.max_bytes() == (static_cast<std::size_t>(PrefixCheckpointConfig::kMaxHostCacheMb) << 20));
}

TEST_CASE("prefix checkpoints: host tier byte bound holds after every put") {
    const std::size_t budget = 64 * 1024;
    CheckpointHostTier tier(budget, 1000);
    std::uint32_t seed = 12345;
    std::size_t accepted = 0;
    for (int i = 0; i < 400; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const std::size_t state_bytes = 1 + (seed >> 8) % 30000;
        const std::size_t n_tokens = 1 + (seed >> 4) % 200;
        const bool ok = tier.Put(Seq(n_tokens, static_cast<Token>(i * 1000)), Blob(state_bytes));
        accepted += ok ? 1 : 0;
        REQUIRE(tier.bytes() <= budget);
    }
    CHECK(accepted > 100u);
    PrefixCheckpointStats st;
    tier.FillStats(st);
    CHECK(st.host_bytes == tier.bytes());
    CHECK(st.host_entries == tier.size());
    CHECK(st.host_evictions > 0u);
    CHECK(st.host_puts == accepted);
}

TEST_CASE("prefix checkpoints: host tier evicts least recently used first") {
    const std::size_t entry = CheckpointHostTier::EntryBytes(10, 1000, 0);
    CheckpointHostTier tier(entry * 2 + entry / 2, 100);  // room for two entries, not three
    const auto a = Seq(10, 100), b = Seq(10, 200), c = Seq(10, 300);
    REQUIRE(tier.Put(a, Blob(1000, 1)));
    REQUIRE(tier.Put(b, Blob(1000, 2)));
    REQUIRE(tier.size() == 2u);

    // Touch A so B is now the least recently used.
    auto hit_a = tier.Lookup(a, 10);
    REQUIRE(hit_a.has_value());
    CHECK(hit_a->n_tokens == 10u);

    REQUIRE(tier.Put(c, Blob(1000, 3)));
    CHECK(tier.size() == 2u);
    CHECK(tier.Lookup(a, 10).has_value());
    CHECK_FALSE(tier.Lookup(b, 10).has_value());
    CHECK(tier.Lookup(c, 10).has_value());

    PrefixCheckpointStats st;
    tier.FillStats(st);
    CHECK(st.host_evictions == 1u);
}

TEST_CASE("prefix checkpoints: an oversize entry is rejected without evicting") {
    const std::size_t entry = CheckpointHostTier::EntryBytes(10, 1000, 0);
    CheckpointHostTier tier(entry * 2, 100);
    REQUIRE(tier.Put(Seq(10, 100), Blob(1000)));
    REQUIRE(tier.Put(Seq(10, 200), Blob(1000)));
    const auto before_bytes = tier.bytes();

    CHECK_FALSE(tier.Put(Seq(10, 300), Blob(entry * 2)));  // alone exceeds the budget
    CHECK(tier.size() == 2u);
    CHECK(tier.bytes() == before_bytes);
    PrefixCheckpointStats st;
    tier.FillStats(st);
    CHECK(st.host_rejected == 1u);
    CHECK(st.host_evictions == 0u);

    // Empty prefix or empty state is never stored.
    CHECK_FALSE(tier.Put(std::vector<Token>{}, Blob(10)));
    CHECK_FALSE(tier.Put(Seq(5), std::vector<std::uint8_t>{}));
}

TEST_CASE("prefix checkpoints: pending_h and token bytes are charged") {
    CHECK(CheckpointHostTier::EntryBytes(10, 1000, 0) ==
          1000 + 10 * sizeof(Token) + CheckpointHostTier::kEntryOverheadBytes);
    CHECK(CheckpointHostTier::EntryBytes(10, 1000, 50) ==
          1000 + 10 * sizeof(Token) + 50 * sizeof(float) + CheckpointHostTier::kEntryOverheadBytes);
    CHECK(CheckpointHostTier::EntryBytes(kSizeMax, kSizeMax, kSizeMax) == kSizeMax);
    CHECK(CheckpointHostTier::EntryBytes(kSizeMax / sizeof(Token) + 1, 0, 0) == kSizeMax);

    CheckpointHostTier tier(1 << 20, 10);
    REQUIRE(tier.Put(Seq(10), Blob(1000), std::vector<float>(50, 1.0F)));
    CHECK(tier.bytes() == CheckpointHostTier::EntryBytes(10, 1000, 50));
}

TEST_CASE("prefix checkpoints: replacing an identical prefix does not double count") {
    CheckpointHostTier tier(1 << 20, 10);
    REQUIRE(tier.Put(Seq(10), Blob(1000, 1)));
    const auto one = tier.bytes();
    REQUIRE(tier.Put(Seq(10), Blob(1000, 2)));
    CHECK(tier.size() == 1u);
    CHECK(tier.bytes() == one);
    auto hit = tier.Lookup(Seq(10), 10);
    REQUIRE(hit.has_value());
    REQUIRE(hit->state != nullptr);
    CHECK((*hit->state)[0] == 2);
    // Replacing with a bigger blob re-charges it.
    REQUIRE(tier.Put(Seq(10), Blob(5000, 3)));
    CHECK(tier.size() == 1u);
    CHECK(tier.bytes() == CheckpointHostTier::EntryBytes(10, 5000, 0));
}

TEST_CASE("prefix checkpoints: a failed replacement keeps the old entry") {
    const std::size_t entry = CheckpointHostTier::EntryBytes(10, 1000, 0);
    CheckpointHostTier tier(entry + 10, 10);
    REQUIRE(tier.Put(Seq(10), Blob(1000, 1)));
    CHECK_FALSE(tier.Put(Seq(10), Blob(100000, 2)));  // too big for the budget
    CHECK(tier.size() == 1u);
    auto hit = tier.Lookup(Seq(10), 10);
    REQUIRE(hit.has_value());
    CHECK((*hit->state)[0] == 1);
}

TEST_CASE("prefix checkpoints: the entry-count cap is enforced") {
    CheckpointHostTier tier(1 << 24, 3);
    for (int i = 0; i < 10; ++i) {
        REQUIRE(tier.Put(Seq(8, i * 100), Blob(100)));
        CHECK(tier.size() <= 3u);
    }
    CHECK(tier.size() == 3u);
    // The three newest survive.
    CHECK(tier.Lookup(Seq(8, 900), 8).has_value());
    CHECK(tier.Lookup(Seq(8, 800), 8).has_value());
    CHECK(tier.Lookup(Seq(8, 700), 8).has_value());
    CHECK_FALSE(tier.Lookup(Seq(8, 600), 8).has_value());
}

TEST_CASE("prefix checkpoints: host lookup returns the longest true prefix") {
    CheckpointHostTier tier(1 << 20, 10);
    const auto full = Seq(20);
    REQUIRE(tier.Put(std::span<const Token>(full).first(4), Blob(10, 4)));
    REQUIRE(tier.Put(std::span<const Token>(full).first(8), Blob(10, 8)));

    auto h = tier.Lookup(full, 20);
    REQUIRE(h.has_value());
    CHECK(h->n_tokens == 8u);
    CHECK((*h->state)[0] == 8);

    // max_match_tokens bounds the answer.
    h = tier.Lookup(full, 6);
    REQUIRE(h.has_value());
    CHECK(h->n_tokens == 4u);
    CHECK_FALSE(tier.Lookup(full, 3).has_value());
    CHECK_FALSE(tier.Lookup(full, 0).has_value());

    // A prompt that diverges inside the 8-token checkpoint only gets the 4.
    auto diverged = full;
    diverged[5] = 9999;
    h = tier.Lookup(diverged, 20);
    REQUIRE(h.has_value());
    CHECK(h->n_tokens == 4u);

    // A prompt shorter than a checkpoint cannot use it.
    h = tier.Lookup(std::span<const Token>(full).first(7), 7);
    REQUIRE(h.has_value());
    CHECK(h->n_tokens == 4u);

    // Different first token: nothing.
    auto other = full;
    other[0] = -5;
    CHECK_FALSE(tier.Lookup(other, 20).has_value());
    CHECK_FALSE(tier.Lookup(std::vector<Token>{}, 20).has_value());
}

TEST_CASE("prefix checkpoints: host tier pending_h round-trips") {
    CheckpointHostTier tier(1 << 20, 10);
    REQUIRE(tier.Put(Seq(6), Blob(10), std::vector<float>{1.5F, 2.5F}));
    auto h = tier.Lookup(Seq(6), 6);
    REQUIRE(h.has_value());
    REQUIRE(h->pending_h != nullptr);
    CHECK(h->pending_h->size() == 2u);
    CHECK((*h->pending_h)[1] == doctest::Approx(2.5F));
}

TEST_CASE("prefix checkpoints: EraseNotPrefixOf drops diverged entries") {
    CheckpointHostTier tier(1 << 20, 10);
    const auto base = Seq(20);
    REQUIRE(tier.Put(std::span<const Token>(base).first(4), Blob(10)));
    REQUIRE(tier.Put(std::span<const Token>(base).first(8), Blob(10)));
    REQUIRE(tier.Put(Seq(8, 5000), Blob(10)));

    CHECK(tier.EraseNotPrefixOf(base) == 1u);
    CHECK(tier.size() == 2u);
    auto shorter = std::span<const Token>(base).first(6);
    CHECK(tier.EraseNotPrefixOf(shorter) == 1u);  // the 8-token entry is longer than `keep`
    CHECK(tier.size() == 1u);
    CHECK(tier.bytes() == CheckpointHostTier::EntryBytes(4, 10, 0));
}

TEST_CASE("prefix checkpoints: Configure shrinks then disables then clears") {
    CheckpointHostTier tier(1 << 20, 10);
    for (int i = 0; i < 5; ++i) REQUIRE(tier.Put(Seq(8, i * 100), Blob(1000)));
    const std::size_t one = CheckpointHostTier::EntryBytes(8, 1000, 0);
    tier.Configure(one * 2, 10);
    CHECK(tier.size() == 2u);
    CHECK(tier.bytes() <= one * 2);
    // The most recently used entries survive a shrink.
    CHECK(tier.Lookup(Seq(8, 400), 8).has_value());
    CHECK(tier.Lookup(Seq(8, 300), 8).has_value());

    tier.Configure(0, 10);
    CHECK_FALSE(tier.enabled());
    CHECK(tier.size() == 0u);
    CHECK(tier.bytes() == 0u);

    tier.Configure(1 << 20, 10);
    REQUIRE(tier.Put(Seq(8), Blob(100)));
    tier.Clear();
    CHECK(tier.size() == 0u);
    CHECK(tier.bytes() == 0u);
    CHECK(tier.enabled());
}

// ---------------------------------------------------------------------------
// Counters and telemetry
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: hit/miss counters and hit rate") {
    PrefixCheckpointStats st;
    CHECK(st.Lookups() == 0u);
    CHECK(st.HitRate() == doctest::Approx(0.0));

    st.OnMiss();
    st.OnHit(CheckpointTier::kSlot, 1000);
    st.OnHit(CheckpointTier::kHost, 500);
    st.OnHit(CheckpointTier::kHost, 250);
    CHECK(st.slot_hits == 1u);
    CHECK(st.host_hits == 2u);
    CHECK(st.misses == 1u);
    CHECK(st.restored_tokens == 1750u);
    CHECK(st.Lookups() == 4u);
    CHECK(st.HitRate() == doctest::Approx(0.75));

    st.OnPlanned(3);
    st.OnCaptured(4096);
    st.OnCaptured(1024);
    CHECK(st.checkpoints_planned == 3u);
    CHECK(st.checkpoints_captured == 2u);
    CHECK(st.captured_bytes == 5120u);

    const PrefixCheckpointCounters c = st.ToCounters();
    CHECK(c.slot_hits == 1u);
    CHECK(c.host_hits == 2u);
    CHECK(c.misses == 1u);
    CHECK(c.restored_tokens == 1750u);
    CHECK(c.checkpoints_planned == 3u);
    CHECK(c.checkpoints_captured == 2u);
    CHECK(c.captured_bytes == 5120u);

    // Saturates instead of wrapping.
    st.captured_bytes = std::numeric_limits<std::uint64_t>::max() - 1;
    st.OnCaptured(100);
    CHECK(st.captured_bytes == std::numeric_limits<std::uint64_t>::max());

    // Unloaded backend reports zeroed telemetry counters cleanly.
    LlamaCppBackend backend;
    const PrefixCheckpointCounters empty = backend.GetPrefixCheckpointStats();
    CHECK(empty.slot_hits == 0u);
    CHECK(empty.host_hits == 0u);
    CHECK(empty.misses == 0u);
    CHECK(empty.host_bytes == 0u);
}

TEST_CASE("prefix checkpoints: telemetry fields and detail string are stable and complete") {
    PrefixCheckpointStats st;
    st.slot_hits = 3;
    st.host_hits = 2;
    st.misses = 5;
    st.restored_tokens = 9000;
    st.host_bytes = 4096;
    st.host_entries = 2;
    st.host_evictions = 7;
    const auto fields = StatsToFields(st);

    auto find = [&](const std::string& name) -> std::optional<std::uint64_t> {
        for (const auto& [k, v] : fields) {
            if (k == name) return v;
        }
        return std::nullopt;
    };
    CHECK(find("prefix_checkpoint.slot_hits") == 3u);
    CHECK(find("prefix_checkpoint.host_hits") == 2u);
    CHECK(find("prefix_checkpoint.misses") == 5u);
    CHECK(find("prefix_checkpoint.restored_tokens") == 9000u);
    CHECK(find("prefix_checkpoint.host_bytes") == 4096u);
    CHECK(find("prefix_checkpoint.host_entries") == 2u);
    CHECK(find("prefix_checkpoint.host_evictions") == 7u);
    CHECK(find("prefix_checkpoint.checkpoints_planned").has_value());
    CHECK(find("prefix_checkpoint.checkpoints_captured").has_value());
    CHECK(find("prefix_checkpoint.captured_bytes").has_value());
    CHECK(find("prefix_checkpoint.host_puts").has_value());
    CHECK(find("prefix_checkpoint.host_rejected").has_value());
    // No duplicate names; every name carries the prefix.
    for (std::size_t i = 0; i < fields.size(); ++i) {
        CHECK(fields[i].first.rfind("prefix_checkpoint.", 0) == 0);
        for (std::size_t j = i + 1; j < fields.size(); ++j) CHECK(fields[i].first != fields[j].first);
    }

    const std::string detail = FormatStatsDetail("hit_host", st);
    CHECK(detail.find("hit_host") == 0u);
    CHECK(detail.find("slot_hits=3") != std::string::npos);
    CHECK(detail.find("host_hits=2") != std::string::npos);
    CHECK(detail.find("misses=5") != std::string::npos);
    CHECK(detail.find("host_bytes=4096") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Continuous-batch cursor, shared budget split, and L2+checkpoint restore plan
// ---------------------------------------------------------------------------

TEST_CASE("prefix checkpoints: prefill cursor clamps chunks and never wedges across mode switches") {
    const Positions planned{1024, 2048, 3072, 4080};

    SUBCASE("clamps every chunk in order (single-slot or multi-slot Case B)") {
        std::size_t cursor = 0;
        std::size_t off = 0;
        Positions captured;
        while (off < 4096) {
            const std::size_t raw_end = std::min<std::size_t>(4096, off + 512);
            const std::size_t step_end = ClampPrefillStepToCheckpoint(planned, cursor, off, raw_end);
            REQUIRE(step_end > off);
            off = step_end;
            if (ShouldCaptureAtOffset(planned, cursor, off)) {
                captured.push_back(off);
            }
        }
        CHECK(captured == planned);
        CHECK(cursor == planned.size());
    }

    SUBCASE("uneven batch chunks spanning checkpoints clamp accurately") {
        std::size_t cursor = 0;
        std::size_t off = 0;
        Positions captured;
        const std::size_t chunk_sizes[] = {700, 700, 1500, 1500, 1500, 1500, 1500};
        for (std::size_t chunk : chunk_sizes) {
            if (off >= 4096) break;
            const std::size_t raw_end = std::min<std::size_t>(4096, off + chunk);
            const std::size_t step_end = ClampPrefillStepToCheckpoint(planned, cursor, off, raw_end);
            REQUIRE(step_end > off);
            off = step_end;
            if (ShouldCaptureAtOffset(planned, cursor, off)) captured.push_back(off);
        }
        CHECK(captured == Positions{1024, 2048, 3072, 4080});
    }

    SUBCASE("jumping past an earlier checkpoint advances cursor instead of stalling") {
        std::size_t cursor = 0;
        // Suppose an earlier step jumped straight to 1500 (past 1024).
        CHECK(ClampPrefillStepToCheckpoint(planned, cursor, 1500, 2200) == 2048u);
        CHECK(cursor == 1u);
        CHECK(ShouldCaptureAtOffset(planned, cursor, 2048));
        CHECK(cursor == 2u);
        // Remaining checkpoints (3072, 4080) are still reached.
        CHECK(ClampPrefillStepToCheckpoint(planned, cursor, 2048, 4096) == 3072u);
        CHECK(ShouldCaptureAtOffset(planned, cursor, 3072));
        CHECK(ClampPrefillStepToCheckpoint(planned, cursor, 3072, 4096) == 4080u);
        CHECK(ShouldCaptureAtOffset(planned, cursor, 4080));
        CHECK_FALSE(ShouldCaptureAtOffset(planned, cursor, 4096));
    }

    SUBCASE("empty plan is a no-op") {
        std::size_t cursor = 0;
        CHECK(ClampPrefillStepToCheckpoint({}, cursor, 0, 512) == 512u);
        CHECK_FALSE(ShouldCaptureAtOffset({}, cursor, 512));
    }
}

TEST_CASE("prefix checkpoints: SplitHostCacheBudget never double-allocates cache_ram_mb") {
    for (const CheckpointMode mode : {CheckpointMode::kOff, CheckpointMode::kSparse, CheckpointMode::kDense}) {
        const auto zero = SplitHostCacheBudget(0, mode);
        CHECK(zero.total_bytes == 0u);
        CHECK(zero.radix_bytes == 0u);
        CHECK(zero.checkpoint_bytes == 0u);

        for (const std::uint32_t mib : {1u, 16u, 256u, 4096u, 65536u, std::numeric_limits<std::uint32_t>::max()}) {
            const auto sp = SplitHostCacheBudget(mib, mode);
            const std::uint32_t expected_mb = std::min(mib, PrefixCheckpointConfig::kMaxHostCacheMb);
            const std::size_t expected_total = static_cast<std::size_t>(expected_mb) << 20;
            CHECK(sp.total_bytes == expected_total);
            CHECK(sp.radix_bytes + sp.checkpoint_bytes == sp.total_bytes);
            if (mode == CheckpointMode::kOff) {
                CHECK(sp.checkpoint_bytes == 0u);
                CHECK(sp.radix_bytes == sp.total_bytes);
            } else {
                CHECK(sp.checkpoint_bytes > 0u);
                CHECK(sp.radix_bytes > 0u);
            }
        }
    }
    // Dense (hybrid/recurrent) reserves a larger checkpoint share than sparse (SWA).
    const auto dense = SplitHostCacheBudget(1024, CheckpointMode::kDense);
    const auto sparse = SplitHostCacheBudget(1024, CheckpointMode::kSparse);
    CHECK(dense.checkpoint_bytes > sparse.checkpoint_bytes);
    CHECK(dense.radix_bytes + dense.checkpoint_bytes == (std::size_t{1024} << 20));
}

TEST_CASE("prefix checkpoints: ChooseL2RestorePlan restores evicted hybrid prefixes via host checkpoints") {
    // 1. Attention-only: direct trim without checkpoint overlay.
    {
        const auto p = ChooseL2RestorePlan(/*direct_trim_allowed=*/true, /*min_beat_lcp=*/1, /*min_gain=*/128,
                                           /*trimmed_matched=*/2500, /*trimmed_state=*/4096,
                                           /*host_ckpt=*/0, /*exact_matched=*/0);
        CHECK(p.use_trimmed_l2);
        CHECK_FALSE(p.apply_host_ckpt);
        CHECK(p.target_tokens == 2500u);
    }
    // 2. Hybrid/recurrent: evicted L2 snapshot [0..4096) diverges at 2500, host checkpoint at 2048 -> restore 2048!
    {
        const auto p = ChooseL2RestorePlan(/*direct_trim_allowed=*/false, /*min_beat_lcp=*/1, /*min_gain=*/128,
                                           /*trimmed_matched=*/2500, /*trimmed_state=*/4096,
                                           /*host_ckpt=*/2048, /*exact_matched=*/0);
        CHECK(p.use_trimmed_l2);
        CHECK(p.apply_host_ckpt);
        CHECK(p.target_tokens == 2048u);
    }
    // 3. Hybrid/recurrent: partial L2 match with NO host checkpoint falls back to shorter exact L2 match.
    {
        const auto p = ChooseL2RestorePlan(/*direct_trim_allowed=*/false, /*min_beat_lcp=*/1, /*min_gain=*/128,
                                           /*trimmed_matched=*/2500, /*trimmed_state=*/4096,
                                           /*host_ckpt=*/0, /*exact_matched=*/1024);
        CHECK(p.use_exact_l2);
        CHECK_FALSE(p.use_trimmed_l2);
        CHECK_FALSE(p.apply_host_ckpt);
        CHECK(p.target_tokens == 1024u);
    }
    // 4. Hybrid/recurrent: tie between exact L2 and host checkpoint prefers exact L2 (no trim/overlay).
    {
        const auto p = ChooseL2RestorePlan(/*direct_trim_allowed=*/false, /*min_beat_lcp=*/1, /*min_gain=*/128,
                                           /*trimmed_matched=*/2500, /*trimmed_state=*/4096,
                                           /*host_ckpt=*/2048, /*exact_matched=*/2048);
        CHECK(p.use_exact_l2);
        CHECK_FALSE(p.apply_host_ckpt);
        CHECK(p.target_tokens == 2048u);
    }
    // 5. Hybrid/recurrent: host checkpoint longer than exact L2 wins.
    {
        const auto p = ChooseL2RestorePlan(/*direct_trim_allowed=*/false, /*min_beat_lcp=*/1, /*min_gain=*/128,
                                           /*trimmed_matched=*/2500, /*trimmed_state=*/4096,
                                           /*host_ckpt=*/2048, /*exact_matched=*/1024);
        CHECK(p.use_trimmed_l2);
        CHECK(p.apply_host_ckpt);
        CHECK(p.target_tokens == 2048u);
    }
    // 6. Rejects when host checkpoint exceeds matched tokens or gain < min_gain_tokens.
    {
        const auto bad_pos = ChooseL2RestorePlan(false, 1, 128, 2500, 4096, 3000, 0);
        CHECK(bad_pos.target_tokens == 0u);
        const auto low_gain = ChooseL2RestorePlan(false, 2000, 128, 2500, 4096, 2048, 0);
        CHECK(low_gain.target_tokens == 0u);
    }
    // 7. Hybrid/recurrent exact match inside trimmed lookup (state == matched) needs no checkpoint overlay.
    {
        const auto p = ChooseL2RestorePlan(false, 0, 128, 2048, 2048, 0, 0);
        CHECK(p.use_trimmed_l2);
        CHECK_FALSE(p.apply_host_ckpt);
        CHECK(p.target_tokens == 2048u);
    }
}
