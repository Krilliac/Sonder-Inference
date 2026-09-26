// libFuzzer target: telemetry/recording serializer (src/telemetry/telemetry.cpp).
// TelemetryBus::make_envelope() + json dump is exactly what the JSONL sinks
// write for every Observatory event.
//
// Input layout: up to 7 newline-terminated fields (event_type, session_id,
// run_id, request_id, agent_id, task_id, model_instance_id), then a JSON
// document used as the attributes object (non-objects are wrapped as
// {"value": <doc>}; unparsable bytes as {"raw": "<bytes>"}).
//
// Invariants:
//   * the serialized envelope is exactly one line (JSONL framing);
//   * it re-parses as an object carrying the schema, event_type, session_id
//     ("unscoped" when empty) and attributes byte-identical to the input's
//     serialization;
//   * valid UTF-8 in -> valid UTF-8 out.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "fuzz_common.hpp"
#include "sonder/inference/telemetry.hpp"

using namespace sonder::inference;

namespace {

TelemetryBus& bus() {
    // One bus for the whole process: its writer thread is idle (no sinks), and
    // make_envelope() does not touch the queue.
    static TelemetryBus instance([] {
        TelemetryOptions o;
        o.node_id = "fuzz-node";
        return o;
    }());
    return instance;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::string_view rest = sonder_fuzz::as_view(data, size);
    const bool input_utf8 = sonder_fuzz::is_valid_utf8(rest);
    std::vector<std::string> fields;
    while (fields.size() < 7) {
        const auto nl = rest.find('\n');
        if (nl == std::string_view::npos) break;
        fields.emplace_back(rest.substr(0, nl));
        rest.remove_prefix(nl + 1);
    }
    auto field = [&](std::size_t i) -> std::optional<std::string> {
        if (i < fields.size()) return fields[i];
        return std::nullopt;
    };

    const std::string event_type = field(0).value_or("fuzz.event");
    TelemetryContext ctx;
    ctx.session_id = field(1).value_or("");
    ctx.run_id = field(2);
    ctx.request_id = field(3);
    ctx.agent_id = field(4);
    ctx.task_id = field(5);
    ctx.model_instance_id = field(6);

    json::Object attributes;
    if (auto doc = json::parse(rest); doc.ok()) {
        if (doc.value().is_object()) {
            attributes = doc.value().as_object();
        } else {
            attributes.set("value", doc.value());
        }
    } else {
        attributes.set("raw", std::string(rest));
    }
    const std::string attributes_json = json::Value(attributes).dump();

    const std::string line = json::Value(bus().make_envelope(event_type, ctx, attributes, 42)).dump();
    SONDER_FUZZ_CHECK(line.find('\n') == std::string::npos);
    SONDER_FUZZ_CHECK(line.find('\r') == std::string::npos);
    auto back = json::parse(line);
    SONDER_FUZZ_CHECK(back.ok());
    const json::Value& env = back.value();
    SONDER_FUZZ_CHECK(env.is_object());
    SONDER_FUZZ_CHECK(env.find("schema") && env.find("schema")->as_string() == kObservatorySchema);
    SONDER_FUZZ_CHECK(env.find("event_type") && env.find("event_type")->as_string() == event_type);
    SONDER_FUZZ_CHECK(env.find("sequence") && env.find("sequence")->as_int() == 42);
    const std::string expected_session = ctx.session_id.empty() ? std::string("unscoped") : ctx.session_id;
    SONDER_FUZZ_CHECK(env.find("session_id") && env.find("session_id")->as_string() == expected_session);
    SONDER_FUZZ_CHECK(env.find("attributes") && env.find("attributes")->dump() == attributes_json);
    if (input_utf8) {
        SONDER_FUZZ_CHECK(sonder_fuzz::is_valid_utf8(line));
    }
    return 0;
}
