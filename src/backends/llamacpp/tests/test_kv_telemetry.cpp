// Pure unit tests for the native backend's physical KV telemetry: byte maths per
// KV type, the usage rate limiter, event shapes and the bounded queue. No model.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_telemetry.hpp"
#include "llamacpp_internal.hpp"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

namespace {

std::int64_t GetInt(const KvEvent& ev, std::string_view key) {
    const KvAttrValue* v = ev.Find(key);
    REQUIRE_MESSAGE(v != nullptr, "missing attr " << std::string(key) << " in " << ev.type);
    const auto* p = std::get_if<std::int64_t>(v);
    REQUIRE_MESSAGE(p != nullptr, "attr " << std::string(key) << " is not an int");
    return *p;
}
std::string GetStr(const KvEvent& ev, std::string_view key) {
    const KvAttrValue* v = ev.Find(key);
    REQUIRE_MESSAGE(v != nullptr, "missing attr " << std::string(key) << " in " << ev.type);
    const auto* p = std::get_if<std::string>(v);
    REQUIRE_MESSAGE(p != nullptr, "attr " << std::string(key) << " is not a string");
    return *p;
}
double GetNum(const KvEvent& ev, std::string_view key) {
    const KvAttrValue* v = ev.Find(key);
    REQUIRE_MESSAGE(v != nullptr, "missing attr " << std::string(key) << " in " << ev.type);
    const auto* p = std::get_if<double>(v);
    REQUIRE_MESSAGE(p != nullptr, "attr " << std::string(key) << " is not a number");
    return *p;
}
bool GetBool(const KvEvent& ev, std::string_view key) {
    const KvAttrValue* v = ev.Find(key);
    REQUIRE_MESSAGE(v != nullptr, "missing attr " << std::string(key) << " in " << ev.type);
    const auto* p = std::get_if<bool>(v);
    REQUIRE_MESSAGE(p != nullptr, "attr " << std::string(key) << " is not a bool");
    return *p;
}

KvPoolConfig SamplePool() {
    KvPoolConfig c;
    c.n_ctx = 8192;
    c.n_ctx_seq = 2048;
    c.n_seq = 4;
    c.kv_unified = false;
    c.type_k = "q8_0";
    c.type_v = "f16";
    c.bytes_per_token = 1000;
    c.bytes_per_token_source = "estimate";
    c.device = "gpu";
    return c;
}

}  // namespace

TEST_CASE("kv telemetry: row bytes for every supported KV type") {
    struct Row {
        const char* type;
        std::uint64_t n;
        std::uint64_t bytes;
    };
    const Row rows[] = {
        {"f32", 128, 512},   {"f16", 128, 256},   {"bf16", 128, 256}, {"q8_0", 128, 136},
        {"q4_0", 128, 72},   {"q4_1", 128, 80},   {"q5_0", 128, 88},  {"q5_1", 128, 96},
        {"iq4_nl", 128, 72}, {"f16", 1, 2},       {"f32", 0, 0},
    };
    for (const auto& r : rows) {
        const auto got = KvRowBytes(r.type, r.n);
        INFO(r.type << " n=" << r.n);
        REQUIRE(got.has_value());
        CHECK(*got == r.bytes);
    }
    // Quantised types need whole blocks of 32 elements.
    CHECK_FALSE(KvRowBytes("q8_0", 100).has_value());
    CHECK_FALSE(KvRowBytes("q4_0", 33).has_value());
    // Unknown or non-cache types.
    CHECK_FALSE(KvRowBytes("q2_k", 256).has_value());
    CHECK_FALSE(KvRowBytes("", 128).has_value());
    CHECK_FALSE(KvRowBytes("F16", 128).has_value());
    // A row that does not fit in 64 bits is refused, not wrapped.
    CHECK_FALSE(KvRowBytes("f32", std::uint64_t{1} << 62).has_value());
    CHECK_FALSE(KvRowBytes("q8_0", std::numeric_limits<std::uint64_t>::max() - 31).has_value());
}

TEST_CASE("kv telemetry: bytes per token") {
    KvShape s;
    s.n_layer_kv = 32;
    s.n_embd_k_gqa = 1024;
    s.n_embd_v_gqa = 1024;
    CHECK(KvBytesPerToken(s, "f16", "f16") == std::optional<std::uint64_t>(32u * (2048u + 2048u)));
    CHECK(KvBytesPerToken(s, "f32", "f32") == std::optional<std::uint64_t>(32u * (4096u + 4096u)));
    // q8_0 K (32 blocks of 34 B = 1088) + q4_0 V (32 blocks of 18 B = 576).
    CHECK(KvBytesPerToken(s, "q8_0", "q4_0") == std::optional<std::uint64_t>(32u * (1088u + 576u)));
    // K and V widths may differ (MLA-style models).
    s.n_embd_v_gqa = 512;
    CHECK(KvBytesPerToken(s, "f16", "f16") == std::optional<std::uint64_t>(32u * (2048u + 1024u)));
    // No token-indexed layers: a valid, zero answer.
    s.n_layer_kv = 0;
    CHECK(KvBytesPerToken(s, "f16", "f16") == std::optional<std::uint64_t>(0u));
    // Bad inputs.
    s.n_layer_kv = 4;
    CHECK_FALSE(KvBytesPerToken(s, "f16", "bogus").has_value());
    s.n_embd_k_gqa = 100;  // not a multiple of 32
    CHECK_FALSE(KvBytesPerToken(s, "q8_0", "f16").has_value());
    // Overflow across layers is refused.
    s.n_embd_k_gqa = std::uint64_t{1} << 50;
    s.n_embd_v_gqa = std::uint64_t{1} << 50;
    s.n_layer_kv = std::numeric_limits<std::uint32_t>::max();
    CHECK_FALSE(KvBytesPerToken(s, "f32", "f32").has_value());
}

TEST_CASE("kv telemetry: kv.pool.configured shape") {
    const KvEvent ev = MakeKvPoolConfiguredEvent(SamplePool());
    CHECK(ev.type == "kv.pool.configured");
    CHECK(GetStr(ev, "scope") == "physical");
    CHECK(GetInt(ev, "n_ctx") == 8192);
    CHECK(GetInt(ev, "n_ctx_seq") == 2048);
    CHECK(GetInt(ev, "n_seq") == 4);
    CHECK_FALSE(GetBool(ev, "kv_unified"));
    CHECK(GetStr(ev, "type_k") == "q8_0");
    CHECK(GetStr(ev, "type_v") == "f16");
    CHECK(GetInt(ev, "bytes_per_token") == 1000);
    CHECK(GetStr(ev, "bytes_per_token_source") == "estimate");
    CHECK(GetInt(ev, "total_bytes") == 8192 * 1000);
    CHECK(GetStr(ev, "device") == "gpu");
    CHECK(ev.Find("nope") == nullptr);
}

TEST_CASE("kv telemetry: kv.pool.configured with unknown bytes and huge values") {
    KvPoolConfig c = SamplePool();
    c.bytes_per_token = 0;
    c.bytes_per_token_source = "unavailable";
    const KvEvent ev = MakeKvPoolConfiguredEvent(c);
    CHECK(GetInt(ev, "bytes_per_token") == 0);
    CHECK(GetInt(ev, "total_bytes") == 0);
    CHECK(GetStr(ev, "bytes_per_token_source") == "unavailable");

    // n_ctx * bytes_per_token saturates instead of wrapping; values beyond
    // int64 are clamped so JSON never sees a negative size.
    c.n_ctx = std::numeric_limits<std::uint64_t>::max();
    c.bytes_per_token = std::numeric_limits<std::uint64_t>::max();
    const KvEvent big = MakeKvPoolConfiguredEvent(c);
    CHECK(GetInt(big, "n_ctx") == std::numeric_limits<std::int64_t>::max());
    CHECK(GetInt(big, "bytes_per_token") == std::numeric_limits<std::int64_t>::max());
    CHECK(GetInt(big, "total_bytes") == std::numeric_limits<std::int64_t>::max());
}

TEST_CASE("kv telemetry: kv.usage shape and byte maths") {
    const KvEvent ev = MakeKvUsageEvent(KvUsageSample{1000, 4000}, 512);
    CHECK(ev.type == "kv.usage");
    CHECK(GetStr(ev, "scope") == "physical");
    CHECK(GetInt(ev, "cells_used") == 1000);
    CHECK(GetInt(ev, "cells_total") == 4000);
    CHECK(GetInt(ev, "bytes_used") == 1000 * 512);
    CHECK(GetInt(ev, "bytes_total") == 4000 * 512);
    CHECK(GetNum(ev, "occupancy") == doctest::Approx(0.25));

    // Zero total: no division by zero.
    const KvEvent z = MakeKvUsageEvent(KvUsageSample{0, 0}, 512);
    CHECK(GetNum(z, "occupancy") == doctest::Approx(0.0));
    // Over-full is reported raw but occupancy is capped at 1.
    const KvEvent over = MakeKvUsageEvent(KvUsageSample{5000, 4000}, 1);
    CHECK(GetInt(over, "cells_used") == 5000);
    CHECK(GetNum(over, "occupancy") == doctest::Approx(1.0));
    // Unknown bytes per token (0) gives zero byte figures, not garbage.
    const KvEvent nb = MakeKvUsageEvent(KvUsageSample{10, 20}, 0);
    CHECK(GetInt(nb, "bytes_used") == 0);
    CHECK(GetInt(nb, "bytes_total") == 0);
}

TEST_CASE("kv telemetry: kv.evicted and kv.shifted shapes") {
    const KvEvent ev = MakeKvEvictedEvent(3, 1500, 100, "idle_lru");
    CHECK(ev.type == "kv.evicted");
    CHECK(GetStr(ev, "scope") == "physical");
    CHECK(GetInt(ev, "seq_id") == 3);
    CHECK(GetInt(ev, "tokens") == 1500);
    CHECK(GetInt(ev, "bytes") == 150000);
    CHECK(GetStr(ev, "reason") == "idle_lru");

    const KvEvent sh = MakeKvShiftedEvent(2, 4, 1028, -1024, 1024, 100);
    CHECK(sh.type == "kv.shifted");
    CHECK(GetStr(sh, "scope") == "physical");
    CHECK(GetInt(sh, "seq_id") == 2);
    CHECK(GetInt(sh, "p0") == 4);
    CHECK(GetInt(sh, "p1") == 1028);
    CHECK(GetInt(sh, "delta") == -1024);
    CHECK(GetInt(sh, "tokens_discarded") == 1024);
    CHECK(GetInt(sh, "bytes_discarded") == 102400);
}

TEST_CASE("kv telemetry: limiter needs both the token count and the interval") {
    KvUsageLimiter lim(100, 500);
    // The first observation is always allowed.
    CHECK(lim.TryAcquire(10'000));

    // Interval met, tokens not.
    lim.AddTokens(99);
    CHECK_FALSE(lim.WouldAllow(11'000));
    CHECK_FALSE(lim.TryAcquire(11'000));
    // Tokens met, interval not.
    lim.AddTokens(1);
    CHECK_FALSE(lim.WouldAllow(10'499));
    CHECK_FALSE(lim.TryAcquire(10'499));
    // Both met: exactly at the boundary is allowed.
    CHECK(lim.WouldAllow(10'500));
    CHECK(lim.TryAcquire(10'500));
    // The window restarted: tokens were reset.
    CHECK_FALSE(lim.TryAcquire(20'000));
    lim.AddTokens(100);
    CHECK(lim.TryAcquire(20'000));
}

TEST_CASE("kv telemetry: limiter edge cases") {
    {
        KvUsageLimiter lim(100, 500);
        lim.AddTokens(1000);
        CHECK(lim.TryAcquire(0));  // first call, at time zero
        lim.AddTokens(1000);
        // A clock that steps backwards counts as zero elapsed time.
        CHECK_FALSE(lim.TryAcquire(0));
        CHECK(lim.TryAcquire(500));
    }
    {
        KvUsageLimiter lim(0, 0);  // no limits: every observation passes
        CHECK(lim.TryAcquire(5));
        CHECK(lim.TryAcquire(5));
    }
    {
        KvUsageLimiter lim(10, 500);
        CHECK(lim.TryAcquire(1000));
        lim.AddTokens(std::numeric_limits<std::uint64_t>::max());
        lim.AddTokens(std::numeric_limits<std::uint64_t>::max());  // saturates, no wrap to a small value
        CHECK(lim.TryAcquire(1500));
    }
    {
        KvUsageLimiter lim;  // defaults: 64 tokens, 500 ms
        CHECK(lim.TryAcquire(0));
        lim.AddTokens(KvUsageLimiter::kDefaultMinTokens);
        CHECK_FALSE(lim.TryAcquire(KvUsageLimiter::kDefaultMinIntervalMs - 1));
        CHECK(lim.TryAcquire(KvUsageLimiter::kDefaultMinIntervalMs));
        lim.Reset();
        CHECK(lim.TryAcquire(KvUsageLimiter::kDefaultMinIntervalMs));  // first again after a reset
    }
}

TEST_CASE("kv telemetry: Configure queues kv.pool.configured with observed_ms") {
    KvTelemetry kv;
    CHECK(kv.BytesPerToken() == 0);
    kv.Configure(SamplePool(), 1234);
    CHECK(kv.BytesPerToken() == 1000);
    auto evs = kv.Drain();
    REQUIRE(evs.size() == 1);
    CHECK(evs[0].type == "kv.pool.configured");
    CHECK(GetInt(evs[0], "observed_ms") == 1234);
    CHECK(kv.Drain().empty());
}

TEST_CASE("kv telemetry: usage is a coalesced gauge and the last state is never lost") {
    KvTelemetryOptions opt;
    opt.min_tokens = 100;
    opt.min_interval_ms = 500;
    KvTelemetry kv(opt);
    kv.Configure(SamplePool(), 0);
    (void)kv.Drain();

    // First observation passes the limiter.
    CHECK(kv.WantsUsageSample(1000));
    kv.ObserveUsage(KvUsageSample{10, 8192}, 1000);
    // Subsequent ones inside the window are suppressed...
    kv.OnTokens(5);
    CHECK_FALSE(kv.WantsUsageSample(1100));
    kv.ObserveUsage(KvUsageSample{20, 8192}, 1100);
    kv.ObserveUsage(KvUsageSample{30, 8192}, 1200);

    // ...but Drain still reports the newest state, exactly once.
    auto evs = kv.Drain();
    REQUIRE(evs.size() == 1);
    CHECK(evs[0].type == "kv.usage");
    CHECK(GetInt(evs[0], "cells_used") == 30);
    CHECK(GetInt(evs[0], "observed_ms") == 1200);
    CHECK(GetInt(evs[0], "bytes_used") == 30 * 1000);
    CHECK(kv.Drain().empty());

    // Two allowed observations between drains collapse to the newest one.
    kv.OnTokens(100);
    CHECK(kv.WantsUsageSample(2000));
    kv.ObserveUsage(KvUsageSample{100, 8192}, 2000);
    kv.OnTokens(100);
    CHECK(kv.WantsUsageSample(2600));
    kv.ObserveUsage(KvUsageSample{200, 8192}, 2600);
    evs = kv.Drain();
    REQUIRE(evs.size() == 1);
    CHECK(GetInt(evs[0], "cells_used") == 200);
}

TEST_CASE("kv telemetry: discrete events keep order, usage comes last, queue is bounded") {
    KvTelemetryOptions opt;
    opt.max_queued_events = 3;
    KvTelemetry kv(opt);
    kv.Configure(SamplePool(), 1);
    kv.ObserveUsage(KvUsageSample{1, 2}, 2);
    kv.NoteEvicted(0, 10, "idle_lru", 3);
    kv.NoteShifted(1, 0, 8, -8, 8, 4);

    auto evs = kv.Drain();
    REQUIRE(evs.size() == 4);
    CHECK(evs[0].type == "kv.pool.configured");
    CHECK(evs[1].type == "kv.evicted");
    CHECK(GetInt(evs[1], "observed_ms") == 3);
    CHECK(evs[2].type == "kv.shifted");
    CHECK(evs[3].type == "kv.usage");
    CHECK(kv.Dropped() == 0);

    // Over the bound the NEWEST discrete events are dropped (the configured
    // event always survives) and counted.
    kv.Configure(SamplePool(), 10);
    for (int i = 0; i < 5; ++i) kv.NoteEvicted(i, 1, "idle_lru", 11);
    CHECK(kv.Dropped() == 3);
    evs = kv.Drain();
    REQUIRE(evs.size() == 3);
    CHECK(evs[0].type == "kv.pool.configured");
    CHECK(GetInt(evs[1], "seq_id") == 0);
    CHECK(GetInt(evs[2], "seq_id") == 1);
    // The bound frees up after a drain.
    kv.NoteEvicted(9, 1, "idle_lru", 12);
    CHECK(kv.Drain().size() == 1);
}

TEST_CASE("kv telemetry: bytes use the configured bytes_per_token, unknown gives zero") {
    KvTelemetry kv;
    KvPoolConfig c = SamplePool();
    c.bytes_per_token = 0;
    kv.Configure(c, 0);
    (void)kv.Drain();
    kv.NoteEvicted(0, 50, "idle_lru", 1);
    auto evs = kv.Drain();
    REQUIRE(evs.size() == 1);
    CHECK(GetInt(evs[0], "bytes") == 0);
    CHECK(GetInt(evs[0], "tokens") == 50);

    c.bytes_per_token = 64;
    kv.Configure(c, 2);
    (void)kv.Drain();
    kv.NoteEvicted(0, 50, "idle_lru", 3);
    evs = kv.Drain();
    REQUIRE(evs.size() == 1);
    CHECK(GetInt(evs[0], "bytes") == 3200);
}

TEST_CASE("kv telemetry: concurrent producers and a drainer") {
    KvTelemetry kv;
    kv.Configure(SamplePool(), 0);
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> drained_usage{0};
    std::thread drainer([&] {
        while (!stop.load()) {
            for (const auto& e : kv.Drain()) {
                if (e.type == "kv.usage") drained_usage.fetch_add(1);
            }
        }
    });
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&kv, t] {
            for (std::uint64_t i = 0; i < 2000; ++i) {
                kv.OnTokens(1);
                if (kv.WantsUsageSample(i)) kv.ObserveUsage(KvUsageSample{i, 4096}, i);
                if (i % 100 == 0) kv.NoteEvicted(t, i, "idle_lru", i);
            }
        });
    }
    for (auto& p : producers) p.join();
    stop.store(true);
    drainer.join();
    // Whatever interleaving happened, a final drain empties the queue and the
    // newest usage observation was delivered at some point.
    std::uint64_t usage = drained_usage.load();
    for (const auto& e : kv.Drain()) {
        if (e.type == "kv.usage") ++usage;
    }
    CHECK(kv.Drain().empty());
    CHECK(usage > 0);
}

// ---------------------------------------------------------------------------
// Engine wiring: events a backend queues in BackendRuntimeStatus::take_events
// reach the telemetry bus, and `kv.*` ones are standard-level.
// ---------------------------------------------------------------------------

namespace {

using sonder::inference::Backend;
using sonder::inference::BackendCapabilities;
using sonder::inference::BackendModel;
using sonder::inference::BackendRuntimeStatus;
using sonder::inference::Engine;
using sonder::inference::EngineOptions;
using sonder::inference::ErrorCode;
using sonder::inference::MemoryTelemetrySink;
using sonder::inference::ModelDescriptor;
using sonder::inference::ModelLoadOptions;
using sonder::inference::Result;
using sonder::inference::SchedulingOptions;
using sonder::inference::Status;
using sonder::inference::TelemetryLevel;
namespace json = sonder::inference::json;

// Reports one kv.usage and one non-kv event through take_events, once.
class EventBackend final : public Backend {
public:
    std::string name() const override { return "fake-kv"; }
    std::string description() const override { return "fake backend that queues runtime events"; }
    BackendCapabilities capabilities() const override { return {}; }
    Result<std::string> probe() override { return std::string("ok"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
        return Status(ErrorCode::unsupported, "fake backend loads nothing");
    }
    std::optional<BackendRuntimeStatus> runtime_status() const override {
        BackendRuntimeStatus status;
        status.take_events = [sent = sent_] {
            std::vector<std::pair<std::string, json::Object>> out;
            if (!sent->exchange(true)) {
                out.emplace_back("kv.usage", json::Object{{"cells_used", 5}, {"scope", "physical"}});
                out.emplace_back("kv.pool.configured", json::Object{{"n_ctx", 4096}, {"scope", "physical"}});
                out.emplace_back("backend.fake.marker", json::Object{});
            }
            return out;
        };
        return status;
    }

private:
    std::shared_ptr<std::atomic<bool>> sent_ = std::make_shared<std::atomic<bool>>(false);
};

// Event types seen on the bus once the marker (a metrics-level event from the same
// drain) has arrived, so an absent kv.* event proves the level filter, not a missed drain.
std::set<std::string> RunEngineAndCollect(TelemetryLevel level, std::vector<json::Value>* kv_usage_events = nullptr) {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = level;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.sample_gpu_devices = false;
    eo.device_sample_interval = std::chrono::milliseconds{10};
    std::set<std::string> types;
    {
        Engine engine(eo);
        REQUIRE(engine.register_backend(std::make_shared<EventBackend>()).ok());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            engine.telemetry().flush();
            types.clear();
            for (const auto& line : sink->lines()) {
                const auto parsed = json::parse(line);
                if (!parsed.ok()) continue;
                const std::string type = parsed.value().find("event_type")->as_string();
                types.insert(type);
                if (kv_usage_events != nullptr && type == "kv.usage") kv_usage_events->push_back(parsed.value());
            }
            if (types.count("backend.fake.marker") != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
    }
    return types;
}

}  // namespace

TEST_CASE("kv telemetry: engine emits backend kv events at standard level") {
    std::vector<json::Value> usage;
    const auto types = RunEngineAndCollect(TelemetryLevel::standard, &usage);
    REQUIRE(types.count("backend.fake.marker") == 1);
    CHECK(types.count("kv.usage") == 1);
    CHECK(types.count("kv.pool.configured") == 1);
    REQUIRE(usage.size() == 1);
    const auto* attrs = usage[0].find("attributes");
    REQUIRE(attrs != nullptr);
    CHECK(attrs->find("backend")->as_string() == "fake-kv");
    CHECK(attrs->find("scope")->as_string() == "physical");
    CHECK(attrs->find("cells_used")->as_int() == 5);
}

TEST_CASE("kv telemetry: engine drops backend kv events at metrics level but keeps other events") {
    const auto types = RunEngineAndCollect(TelemetryLevel::metrics);
    // The marker arrived (the drain ran) ...
    REQUIRE(types.count("backend.fake.marker") == 1);
    // ... and the kv.* events from that same drain were filtered by level.
    CHECK(types.count("kv.usage") == 0);
    CHECK(types.count("kv.pool.configured") == 0);
}

TEST_CASE("scheduler.configured carries the logical KV capacity in tokens") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.sample_gpu_devices = false;
    eo.device_sample_interval = std::chrono::milliseconds{0};
    eo.scheduling.kv_block_size_tokens = 32;
    eo.scheduling.kv_num_blocks = 100;
    Engine engine(eo);
    REQUIRE(engine.scheduling_active());
    engine.telemetry().flush();
    int seen = 0;
    for (const auto& line : sink->lines()) {
        const auto parsed = json::parse(line);
        if (!parsed.ok() || parsed.value().find("event_type")->as_string() != "scheduler.configured") continue;
        ++seen;
        const auto* attrs = parsed.value().find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("kv_block_size_tokens")->as_int() == 32);
        CHECK(attrs->find("kv_num_blocks")->as_int() == 100);
        CHECK(attrs->find("kv_capacity_tokens")->as_int() == 3200);
        CHECK(attrs->find("mode")->as_string() == "automatic");
    }
    CHECK(seen == 1);
}

TEST_CASE("kv telemetry: core backend exposes a drain that is empty before any model loads") {
    auto be = sonder::inference::llamacpp_detail::MakeLlamaCppBackendForTest(
        {},
        []() -> sonder::inference::llamacpp_detail::ProcessGpuMemoryReading { return {"none", "ok", "", 0, 0}; },
        std::chrono::milliseconds{0}, /*force_uma_mode=*/false, /*force_gpu_offload=*/false);
    const auto status = be->runtime_status();
    REQUIRE(status.has_value());
    REQUIRE(static_cast<bool>(status->take_events));
    CHECK(status->take_events().empty());
}
