#include <doctest/doctest.h>

#include "../src/residency_config.hpp"

using namespace sonder::inference;
namespace json = sonder::inference::json;

namespace {
json::Value parse(const char *text) {
    const auto value = json::parse(text);
    REQUIRE(value.ok());
    return value.value();
}
}  // namespace

TEST_CASE("residency config parser accepts all fields") {
    BackendSetup setup;
    const auto value = parse(R"({"enabled":false,"expect_gpu":true,"min_dedicated_mib":1,
                                "consecutive_samples":1000,"eviction_fraction":0,
                                "eviction_mib":1,"on_eviction":"restart",
                                "max_eviction_restarts":32})");
    REQUIRE(detail::parse_residency_config(value, setup).ok());
    CHECK_FALSE(setup.llamaserver_residency.enabled);
    CHECK(setup.llamaserver_residency.expect_gpu);
    CHECK(setup.llamaserver_residency.min_dedicated_mib == 1);
    CHECK(setup.llamaserver_residency.consecutive_samples == 1000);
    CHECK(setup.llamaserver_residency.eviction_fraction == 0.0);
    CHECK(setup.llamaserver_residency.eviction_mib == 1);
    CHECK(setup.llamaserver_residency.on_eviction == "restart");
    CHECK(setup.llamaserver_residency.max_eviction_restarts == 32);
}

TEST_CASE("residency config parser rejects malformed and unsafe values") {
    const char *const invalid[] = {
        "[]", R"({"unknown":1})", R"({"enabled":1})", R"({"expect_gpu":"yes"})",
        R"({"min_dedicated_mib":0})", R"({"min_dedicated_mib":1048577})",
        R"({"consecutive_samples":0})", R"({"consecutive_samples":1001})",
        R"({"eviction_fraction":-0.1})", R"({"eviction_fraction":1.1})",
        R"({"eviction_mib":1048577})", R"({"on_eviction":"refuse"})",
        R"({"max_eviction_restarts":33})", R"({"eviction_fraction":0,"eviction_mib":0})"};
    for (const auto text : invalid) {
        BackendSetup setup;
        CHECK_MESSAGE(!detail::parse_residency_config(parse(text), setup).ok(), text);
    }
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
TEST_CASE("residency config mapping converts MiB and preserves policy") {
    BackendSetup setup;
    auto &r = setup.llamaserver_residency;
    r.enabled = false;
    r.expect_gpu = true;
    r.min_dedicated_mib = 512;
    r.consecutive_samples = 7;
    r.eviction_fraction = 0.4;
    r.eviction_mib = 2048;
    r.on_eviction = "restart";
    r.max_eviction_restarts = 3;
    LlamaServerResidencyGuardOptions out;
    REQUIRE(detail::apply_residency_config(setup, out).ok());
    CHECK_FALSE(out.enabled);
    CHECK(out.expect_gpu);
    CHECK(out.min_dedicated_bytes == 512ull * 1024 * 1024);
    CHECK(out.consecutive_samples == 7);
    CHECK(out.eviction_fraction == doctest::Approx(0.4));
    CHECK(out.eviction_bytes == 2048ull * 1024 * 1024);
    CHECK(out.on_eviction == LlamaServerEvictionPolicy::restart);
    CHECK(out.max_eviction_restarts == 3);

    r.min_dedicated_mib = 1048577;
    CHECK_FALSE(detail::apply_residency_config(setup, out).ok());
    r.min_dedicated_mib = 512;
    r.eviction_fraction = 0;
    r.eviction_mib = 0;
    CHECK_FALSE(detail::apply_residency_config(setup, out).ok());
}
#endif
