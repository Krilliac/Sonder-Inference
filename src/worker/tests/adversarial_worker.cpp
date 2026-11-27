// Test-only executable: no fault-injection path is compiled into worker_main.
#include "worker/ipc.hpp"
#include "worker/protocol.hpp"

#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <thread>

namespace si = sonder::inference;
namespace wk = si::worker;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
    std::string endpoint, mode, credit_marker;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--endpoint" && i + 1 < argc)
            endpoint = argv[++i];
        else if (arg.rfind("--misbehave=", 0) == 0)
            mode = arg.substr(12);
    }
    if (const auto separator = mode.find('|'); separator != std::string::npos) {
        credit_marker = mode.substr(separator + 1);
        mode.resize(separator);
    }
    bool credited_cancel_completed = false;
    auto connected = wk::connect(endpoint, 5s);
    if (!connected)
        return 1;
    const auto channel = connected.value();
    const auto send = [&](si::json::Value value) { return channel->write_frame(value.dump(), 2s).ok(); };
    const auto read = [&]() -> si::Result<si::json::Value> {
        auto frame = channel->read_frame(5s);
        if (!frame)
            return frame.status();
        return wk::parse_message(frame.value());
    };
    auto hello = wk::message("hello", 0);
    hello.set("engine", "mock");
    if (!send(std::move(hello)) || !read())
        return 2;
    const auto terminal = [&](std::uint64_t id, std::uint64_t count) {
        si::GenerateStats stats;
        stats.prompt_tokens = 1;
        stats.completion_tokens = stats.chunks = count;
        stats.stop_reason = si::StopReason::end_of_sequence;
        auto values = wk::stats_to_json(stats).as_object();
        if (mode == "bad-stats")
            values.set("completion_tokens", std::numeric_limits<std::uint64_t>::max());
        if (mode == "inconsistent-stats")
            values.set("completion_tokens", count + 1);
        if (mode == "bad-timing")
            values.set("eval_ns", wk::kMaxWorkerTimingNs + 1);
        auto done = wk::message("done", id);
        done.set("stats", std::move(values));
        return send(std::move(done));
    };
    const auto token = [&](std::uint64_t id, std::uint64_t index) {
        std::string text = "x", reasoning;
        if (mode == "oversized")
            text.assign(static_cast<std::size_t>(wk::kMaxTokenPayloadBytes + 1), 'x');
        if (mode == "cumulative")
            text.assign(static_cast<std::size_t>(wk::kMaxTokenPayloadBytes), 'x');
        if (mode == "reasoning")
            reasoning.assign(static_cast<std::size_t>(wk::kMaxTokenPayloadBytes + 1), 'r');
        si::TokenChunk chunk{text, index};
        chunk.reasoning = reasoning;
        if (mode == "unrequested-logprob" || mode == "excess-candidates")
            chunk.logprob = -1.0f;
        if (mode == "unrequested-candidates" || mode == "excess-candidates")
            chunk.top_logprobs.push_back({"x", -1.0f});
        if (mode == "excess-candidates")
            chunk.top_logprobs.push_back({"y", -2.0f});
        if (mode == "candidates")
            for (int i = 0; i < si::SamplingConfig::kMaxTopLogprobs; ++i)
                chunk.top_logprobs.push_back({std::string(4096, 'c'), -1.0f});
        auto value = wk::message("token", id);
        value.set("chunk", wk::chunk_to_json(chunk));
        return send(std::move(value));
    };
    const auto credited_cancel = [&](std::uint64_t id, std::uint64_t next_index) {
        // Published only after receiving the ACK (or the initial request).
        // The test cancels from another thread after seeing this marker, so it
        // cannot accidentally exercise the inline cancel-replaces-ACK path.
        {
            std::ofstream marker(credit_marker, std::ios::binary);
            marker << "credit\n";
            marker.flush();
            if (!marker.good())
                return false;
        }
        const auto cancel = read();
        if (!cancel || cancel->find("type")->as_string() != "cancel" || cancel->find("id")->as_uint() != id)
            return false;
        if (!token(id, next_index))
            return false;
        if (mode == "credited-overrun" && !token(id, next_index + 1))
            return false;
        credited_cancel_completed = true;
        if (mode == "credited-cancel")
            return terminal(id, next_index + 1); // verifies drained tokens count in terminal stats too
        return send(wk::error_message(id, {si::ErrorCode::cancelled, "credited cancellation completed"}));
    };
    for (;;) {
        auto request = read();
        if (!request)
            return 0;
        const auto id = request->find("id")->as_uint();
        const auto type = request->find("type")->as_string();
        if (type == "shutdown")
            return 0;
        if (credited_cancel_completed && type == "ack")
            return 3; // a drained cancellation token must never be acknowledged
        if (type == "probe" || type == "load" || type == "list") {
            auto reply = wk::message("result", id);
            if (type == "probe")
                reply.set("value", "mock-test-adversary");
            else {
                si::ModelDescriptor descriptor;
                descriptor.name = type == "load" ? request->find("model")->as_string() : "mock:tiny";
                descriptor.backend = "mock";
                descriptor.context_length = 4096;
                auto value = wk::descriptor_to_json(descriptor).as_object();
                if (mode == "bad-model")
                    value.set("name", "mock:wrong");
                if (mode == "bad-backend")
                    value.set("backend", "llamacpp");
                if (mode == "bad-descriptor")
                    value.set("context_length", std::numeric_limits<std::uint64_t>::max());
                if (type == "list")
                    reply.set("value", si::json::Array{si::json::Value(std::move(value))});
                else
                    reply.set("value", std::move(value));
            }
            if (!send(std::move(reply)))
                return 0;
            continue;
        }
        if (type != "generate" && type != "chat")
            continue;
        if (mode == "post-terminal") {
            if (!terminal(id, 0) || !token(id, 0))
                return 0;
            continue;
        }
        if (mode == "bad-stats" || mode == "inconsistent-stats" || mode == "bad-timing") {
            if (!terminal(id, 0))
                return 0;
            continue;
        }
        if (mode == "silence") {
            while (read()) {
            } // parent must enforce inactivity/destruction
            return 0;
        }
        const auto *sampling = request->find("request")->find("sampling");
        const auto maximum = sampling->find("max_tokens")->as_uint();
        if (mode == "credited-initial" && !credited_cancel_completed) {
            if (!credited_cancel(id, 0))
                return 0;
            continue;
        }
        const auto limit = mode == "overrun" ? maximum + 1 : maximum;
        bool cancelled_with_credit = false;
        for (std::uint64_t i = 0; i < limit; ++i) {
            if (mode == "trickle")
                std::this_thread::sleep_for(10ms);
            if (!token(id, i))
                return 0;
            auto control = read();
            if (!control)
                return 0;
            const auto control_type = control->find("type")->as_string();
            if ((mode == "credited-cancel" || mode == "credited-overrun" || mode == "credited-cap") &&
                !credited_cancel_completed && control_type == "ack") {
                if (!credited_cancel(id, i + 1))
                    return 0;
                cancelled_with_credit = true;
                break;
            }
            if (control_type == "cancel" || control_type == "stop") {
                if (mode == "post-cancel")
                    (void)token(id, i + 1);
                while (read()) {
                } // intentionally ignore cancellation
                return 0;
            }
            if (mode == "ignore-cancel") {
                while (read()) {
                }
                return 0;
            }
        }
        if (!cancelled_with_credit && !terminal(id, limit))
            return 0;
    }
}
