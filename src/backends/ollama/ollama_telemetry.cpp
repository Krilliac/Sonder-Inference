// Maps Ollama timing fields onto Observatory telemetry attributes/events
// (docs/OBSERVATORY_CONTRACT.md). Only server-reported values are emitted.
#include "sonder/inference/backends/ollama.hpp"

namespace sonder::inference::ollama {

json::Object timing_attributes(const OllamaTimings& t) {
    json::Object a;
    a.set("backend", kOllamaBackendName);
    if (t.has_server_timings) {
        a.set("total_duration_ns", t.total_duration_ns);
        a.set("load_duration_ns", t.load_duration_ns);
        a.set("prompt_eval_count", t.prompt_eval_count);
        a.set("prompt_eval_duration_ns", t.prompt_eval_duration_ns);
        a.set("eval_count", t.eval_count);
        a.set("eval_duration_ns", t.eval_duration_ns);
        a.set("prompt_tokens_per_sec", t.prompt_tokens_per_sec());
        a.set("decode_tokens_per_sec", t.decode_tokens_per_sec());
    }
    if (t.ttfb_ms >= 0) {
        a.set("ttfb_ms", t.ttfb_ms);
    }
    if (t.ttft_ms >= 0) {
        a.set("ttft_ms", t.ttft_ms);
    }
    a.set("wall_ms", t.wall_ms);
    a.set("content_chunks", t.content_chunks);
    return a;
}

int emit_timing_events(TelemetryBus& bus, const TelemetryContext& ctx, const OllamaTimings& t,
                       std::string_view model) {
    if (!t.has_server_timings) {
        return 0;
    }
    int accepted = 0;
    auto base = [&] {
        json::Object a;
        a.set("backend", kOllamaBackendName);
        a.set("model", model);
        return a;
    };
    if (t.load_duration_ns > 0) {
        json::Object a = base();
        a.set("load_duration_ns", t.load_duration_ns);
        accepted += bus.emit("backend.model.load.reported", ctx, std::move(a)) ? 1 : 0;
    }
    if (t.prompt_eval_count > 0 || t.prompt_eval_duration_ns > 0) {
        json::Object a = base();
        a.set("prompt_eval_count", t.prompt_eval_count);
        a.set("prompt_eval_duration_ns", t.prompt_eval_duration_ns);
        a.set("prompt_tokens_per_sec", t.prompt_tokens_per_sec());
        accepted += bus.emit("backend.timing.prefill", ctx, std::move(a)) ? 1 : 0;
    }
    if (t.eval_count > 0 || t.eval_duration_ns > 0) {
        json::Object a = base();
        a.set("eval_count", t.eval_count);
        a.set("eval_duration_ns", t.eval_duration_ns);
        a.set("decode_tokens_per_sec", t.decode_tokens_per_sec());
        if (t.ttft_ms >= 0) {
            a.set("ttft_ms", t.ttft_ms);
        }
        accepted += bus.emit("backend.timing.decode", ctx, std::move(a)) ? 1 : 0;
    }
    return accepted;
}

}  // namespace sonder::inference::ollama
