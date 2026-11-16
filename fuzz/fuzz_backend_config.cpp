// libFuzzer target: backend configuration and llama-server response readers
// that take an already-parsed JSON document.
//
// Input: raw JSON text. On json::parse success the document is given to
//   * parse_residency_config() then apply_residency_config()
//     (src/server/src/residency_config.cpp), on the document and on its
//     "residency" / "spill_guard.residency" members;
//   * parse_warmup_config() (warmup_config.cpp), on the document and on its
//     "warmup" member;
//   * load_llamaserver_extensions() (llamaserver_config_extensions.hpp);
//   * llamaserver::parse_timings() (both overloads) and parse_props()
//     (src/backends/llamaserver/llamaserver_protocol.cpp, timings.cpp).
// apply_residency_config() is additionally driven with a BackendSetup whose
// fields come straight from input bytes (no JSON), since callers may build one
// directly.
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary documents.
//   * an error Status has a non-empty message under 1 KiB.
//   * residency: a rejected document leaves the setup unchanged; an accepted
//     one has min_dedicated_mib in [1, 1048576], consecutive_samples in
//     [1, 1000], eviction_fraction finite in [0, 1], eviction_mib in
//     [0, 1048576], max_eviction_restarts in [0, 32], not both eviction
//     thresholds zero, on_eviction warn|restart; apply_residency_config()
//     accepts it, converts MiB to bytes exactly, and the result passes
//     validate_residency_guard().
//   * apply_residency_config() on arbitrary field values either rejects or
//     yields a guard that passes validate_residency_guard() without
//     overflowing the MiB conversion.
//   * warmup: a rejected config leaves the setup unchanged; an accepted config has max_prefix_chars in [1, 16777216], unique
//     slot ids <= 1023 (none when "all"), and no NUL in messages_file.
//   * extensions: an accepted env has at most 128 entries, non-empty keys
//     without '=' or NUL, values without NUL, within 32760 bytes in total.
//   * timings/props: accepted counts are consistent (draft_accepted <= draft;
//     rates finite and non-negative; n_ctx <= 2^32; total_slots <= 2^20);
//     the BackendTimings overload and the Timings overload agree on the
//     counters both report; a rejected BackendTimings / ServerProps is reset.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "backends/llamaserver/gpu_residency.hpp"
#include "backends/llamaserver/llamaserver_protocol.hpp"
#include "backends/llamaserver/timings.hpp"
#include "fuzz_common.hpp"
#include "server/src/llamaserver_config_extensions.hpp"
#include "server/src/residency_config.hpp"
#include "server/src/warmup_config.hpp"
#include "sonder/inference/backend_setup.hpp"

namespace si = sonder::inference;
namespace ls = sonder::inference::llamaserver;
namespace det = sonder::inference::detail;

namespace {

void check_status(const si::Status& st) {
    if (!st.ok()) {
        SONDER_FUZZ_CHECK(!st.message().empty());
        SONDER_FUZZ_CHECK(st.message().size() < 1024);
    }
}

bool same_residency(const si::BackendSetup::LlamaServerResidencyConfig& a,
                    const si::BackendSetup::LlamaServerResidencyConfig& b) {
    return a.enabled == b.enabled && a.expect_gpu == b.expect_gpu && a.min_dedicated_mib == b.min_dedicated_mib &&
           a.consecutive_samples == b.consecutive_samples && a.eviction_fraction == b.eviction_fraction &&
           a.eviction_mib == b.eviction_mib && a.on_eviction == b.on_eviction &&
           a.max_eviction_restarts == b.max_eviction_restarts;
}

void check_guard(const si::LlamaServerResidencyGuardOptions& g, const si::BackendSetup::LlamaServerResidencyConfig& c) {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    SONDER_FUZZ_CHECK(g.min_dedicated_bytes == c.min_dedicated_mib * kMiB);
    SONDER_FUZZ_CHECK(g.eviction_bytes == c.eviction_mib * kMiB);
    SONDER_FUZZ_CHECK(g.consecutive_samples == c.consecutive_samples);
    SONDER_FUZZ_CHECK(g.max_eviction_restarts == c.max_eviction_restarts);
    SONDER_FUZZ_CHECK(g.eviction_fraction == c.eviction_fraction);
    SONDER_FUZZ_CHECK(ls::validate_residency_guard(g).ok());
}

void check_residency(const si::json::Value& doc) {
    si::BackendSetup setup;
    const auto before = setup.llamaserver_residency;
    const si::Status st = det::parse_residency_config(doc, setup);
    check_status(st);
    if (!st.ok()) {
        SONDER_FUZZ_CHECK(same_residency(before, setup.llamaserver_residency));
        return;
    }
    const auto& c = setup.llamaserver_residency;
    SONDER_FUZZ_CHECK(c.min_dedicated_mib >= 1 && c.min_dedicated_mib <= 1048576);
    SONDER_FUZZ_CHECK(c.consecutive_samples >= 1 && c.consecutive_samples <= 1000);
    SONDER_FUZZ_CHECK(std::isfinite(c.eviction_fraction) && c.eviction_fraction >= 0.0 && c.eviction_fraction <= 1.0);
    SONDER_FUZZ_CHECK(c.eviction_mib <= 1048576);
    SONDER_FUZZ_CHECK(c.max_eviction_restarts <= 32);
    SONDER_FUZZ_CHECK(!(c.eviction_fraction == 0.0 && c.eviction_mib == 0));
    SONDER_FUZZ_CHECK(c.on_eviction == "warn" || c.on_eviction == "restart");

    si::LlamaServerResidencyGuardOptions guard;
    const si::Status applied = det::apply_residency_config(setup, guard);
    SONDER_FUZZ_CHECK(applied.ok());
    check_guard(guard, c);
    SONDER_FUZZ_CHECK(guard.enabled == c.enabled && guard.expect_gpu == c.expect_gpu);
}

std::uint64_t take_u64(std::string_view bytes, std::size_t offset) {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        v <<= 8;
        if (offset + i < bytes.size()) v |= static_cast<unsigned char>(bytes[offset + i]);
    }
    return v;
}

void check_apply_direct(std::string_view bytes) {
    si::BackendSetup setup;
    auto& c = setup.llamaserver_residency;
    c.min_dedicated_mib = take_u64(bytes, 0);
    c.consecutive_samples = take_u64(bytes, 8);
    c.eviction_mib = take_u64(bytes, 16);
    c.max_eviction_restarts = take_u64(bytes, 24);
    const std::uint64_t f = take_u64(bytes, 32);
    std::memcpy(&c.eviction_fraction, &f, sizeof(f));
    c.on_eviction = (bytes.size() > 40 && bytes[40] & 1) ? "restart" : (bytes.size() > 40 && bytes[40] & 2) ? "warn" : "x";
    si::LlamaServerResidencyGuardOptions guard;
    const si::Status st = det::apply_residency_config(setup, guard);
    check_status(st);
    if (st.ok()) check_guard(guard, c);
}

void check_warmup(const si::json::Value& doc) {
    si::BackendSetup setup;
    // Start from non-default values so any partial write is visible.
    setup.llamaserver_warmup_messages_file = "keep";
    setup.llamaserver_warmup_on_restart = !setup.llamaserver_warmup_on_restart;
    setup.llamaserver_warmup_all_slots = false;
    setup.llamaserver_warmup_slots = {7, 9};
    const auto before_max = setup.llamaserver_warmup_max_prefix_chars;
    const auto before_restart = setup.llamaserver_warmup_on_restart;
    const si::Status st = det::parse_warmup_config(doc, setup);
    check_status(st);
    if (!st.ok()) {
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_messages_file == "keep");
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_chat_template_kwargs.empty());
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_on_restart == before_restart);
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_max_prefix_chars == before_max);
        SONDER_FUZZ_CHECK(!setup.llamaserver_warmup_all_slots);
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_slots == std::vector<std::uint32_t>({7, 9}));
        return;
    }
    SONDER_FUZZ_CHECK(setup.llamaserver_warmup_messages_file.find('\0') == std::string::npos);
    SONDER_FUZZ_CHECK(setup.llamaserver_warmup_max_prefix_chars >= 1 &&
                      setup.llamaserver_warmup_max_prefix_chars <= 16777216u);
    if (setup.llamaserver_warmup_all_slots) SONDER_FUZZ_CHECK(setup.llamaserver_warmup_slots.empty());
    for (std::size_t i = 0; i < setup.llamaserver_warmup_slots.size(); ++i) {
        SONDER_FUZZ_CHECK(setup.llamaserver_warmup_slots[i] <= 1023u);
        for (std::size_t j = 0; j < i; ++j) {
            SONDER_FUZZ_CHECK(setup.llamaserver_warmup_slots[i] != setup.llamaserver_warmup_slots[j]);
        }
    }
}

void check_extensions(const si::json::Value& doc) {
    if (!doc.is_object()) return;
    std::vector<std::pair<std::string, std::string>> env;
    const si::Status st = si::load_llamaserver_extensions(doc.as_object(), env);
    check_status(st);
    if (!st.ok()) return;
    SONDER_FUZZ_CHECK(env.size() <= 128);
    std::size_t bytes = 0;
    for (const auto& [k, v] : env) {
        SONDER_FUZZ_CHECK(!k.empty());
        SONDER_FUZZ_CHECK(k.find('=') == std::string::npos && k.find('\0') == std::string::npos);
        SONDER_FUZZ_CHECK(v.find('\0') == std::string::npos);
        bytes += k.size() + v.size() + 2;
    }
    SONDER_FUZZ_CHECK(bytes <= 32760);
    if (const auto* results = doc.find("results")) {
        SONDER_FUZZ_CHECK(results->is_object());
        const auto* schema = results->find("schema");
        SONDER_FUZZ_CHECK(schema && schema->as_string() == "sonder.inference.tune/1");
        const auto* mode = doc.find("mode");
        SONDER_FUZZ_CHECK(mode && mode->as_string() == "spawn");
    }
}

void check_timings_and_props(const si::json::Value& doc) {
    si::BackendTimings raw;
    const si::Status raw_st = ls::parse_timings(doc, raw);
    check_status(raw_st);
    if (raw_st.ok()) {
        if (raw.draft_n && raw.draft_n_accepted) SONDER_FUZZ_CHECK(*raw.draft_n_accepted <= *raw.draft_n);
        for (const auto& ms : {raw.prompt_ms, raw.predicted_ms}) {
            SONDER_FUZZ_CHECK(!ms || (std::isfinite(*ms) && *ms >= 0.0));
        }
    } else {
        SONDER_FUZZ_CHECK(!raw.prompt_n && !raw.cache_n && !raw.prompt_ms && !raw.predicted_n &&
                          !raw.predicted_ms && !raw.draft_n && !raw.draft_n_accepted);
    }

    ls::Timings t;
    const si::Status t_st = ls::parse_timings(doc, t);
    check_status(t_st);
    if (!t_st.ok()) {
        SONDER_FUZZ_CHECK(!t.present && !t.backend_timings && t.prompt_tokens == 0 && t.cached_tokens == 0 &&
                          t.completion_tokens == 0 && t.draft_tokens == 0 && t.draft_accepted_tokens == 0);
    }
    if (t_st.ok()) {
        SONDER_FUZZ_CHECK(!t.draft_present || !t.draft_accepted_present || t.draft_accepted_tokens <= t.draft_tokens);
        SONDER_FUZZ_CHECK(std::isfinite(t.predicted_tokens_per_second) && t.predicted_tokens_per_second >= 0.0);
        if (t.backend_timings) {
            const auto& b = *t.backend_timings;
            if (b.prompt_n) SONDER_FUZZ_CHECK(t.prompt_present && t.prompt_tokens == *b.prompt_n);
            if (b.draft_n) SONDER_FUZZ_CHECK(t.draft_present && t.draft_tokens == *b.draft_n);
        }
        if (raw_st.ok() && raw.prompt_n) SONDER_FUZZ_CHECK(t.prompt_present && t.prompt_tokens == *raw.prompt_n);
    }

    ls::ServerProps props;
    const si::Status p_st = ls::parse_props(doc, props);
    check_status(p_st);
    if (p_st.ok()) {
        SONDER_FUZZ_CHECK(props.n_ctx <= (std::uint64_t{1} << 32));
        SONDER_FUZZ_CHECK(props.total_slots <= (1u << 20));
    } else {
        SONDER_FUZZ_CHECK(props.n_ctx == 0 && props.total_slots == 0);
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view input = sonder_fuzz::as_view(data, size);
    check_apply_direct(input);

    const auto parsed = si::json::parse(input);
    if (!parsed.ok()) {
        SONDER_FUZZ_CHECK(!parsed.status().message().empty());
        return 0;
    }
    const si::json::Value& doc = parsed.value();
    check_residency(doc);
    if (const auto* r = doc.find("residency")) check_residency(*r);
    if (const auto* g = doc.find("spill_guard")) {
        if (const auto* r = g->find("residency")) check_residency(*r);
    }
    check_warmup(doc);
    if (const auto* w = doc.find("warmup")) check_warmup(*w);
    check_extensions(doc);
    check_timings_and_props(doc);
    return 0;
}
