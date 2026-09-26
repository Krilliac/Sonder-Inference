#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sonder/inference.hpp"

namespace sonder_test {

namespace si = sonder::inference;

struct Harness {
    std::shared_ptr<si::MemoryTelemetrySink> sink = std::make_shared<si::MemoryTelemetrySink>();
    std::unique_ptr<si::Engine> engine;
    std::shared_ptr<si::Model> model;

    explicit Harness(si::MockBackendOptions mock = {}, si::TelemetryLevel level = si::TelemetryLevel::standard,
                     bool capture_text = false) {
        si::EngineOptions eo;
        eo.telemetry.level = level;
        eo.telemetry.capture_text = capture_text;
        eo.telemetry_sinks.push_back(sink);
        engine = std::make_unique<si::Engine>(std::move(eo));
        (void)engine->register_backend(si::make_mock_backend(mock));
        si::ModelLoadOptions lo;
        lo.model = "mock:tiny";
        auto loaded = engine->load_model(si::kMockBackendName, lo);
        if (loaded.ok()) {
            model = loaded.value();
        }
    }

    std::shared_ptr<si::Session> session(si::SamplingConfig sampling = si::SamplingConfig::greedy(32)) {
        si::SessionOptions so;
        so.sampling = sampling;
        auto s = engine->create_session(model, so);
        return s.ok() ? s.value() : nullptr;
    }

    // Parsed telemetry events (after flushing the bus).
    std::vector<si::json::Value> events() {
        engine->telemetry().flush();
        std::vector<si::json::Value> out;
        for (const auto& line : sink->lines()) {
            auto v = si::json::parse(line);
            if (v.ok()) {
                out.push_back(v.value());
            }
        }
        return out;
    }

    std::vector<si::json::Value> events_of(const std::string& type) {
        std::vector<si::json::Value> out;
        for (auto& e : events()) {
            if (e.find("event_type")->as_string() == type) {
                out.push_back(e);
            }
        }
        return out;
    }
};

}  // namespace sonder_test
