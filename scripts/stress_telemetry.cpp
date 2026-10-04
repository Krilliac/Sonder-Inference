#include "sonder/inference/telemetry.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace sonder::inference;
using Clock = std::chrono::steady_clock;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

#ifndef SONDER_BASELINE_TELEMETRY
// Ordinary short writes and sync failures with the default exception mask.
class StreamFault final : public std::streambuf {
public:
    explicit StreamFault(bool flush_failure) : flush_failure_(flush_failure) {}
    unsigned writes = 0, flushes = 0;
protected:
    std::streamsize xsputn(const char*, std::streamsize size) override {
        ++writes;
        return flush_failure_ ? size : 0;
    }
    int_type overflow(int_type value) override { return traits_type::not_eof(value); }
    int sync() override {
        ++flushes;
        return flush_failure_ ? -1 : 0;
    }
private:
    bool flush_failure_;
};

class Fault final : public TelemetrySink {
public:
    explicit Fault(bool flush_failure) : flush_failure_(flush_failure) {}
    void write(std::string_view) override {
        ++writes;
        if (!flush_failure_) throw std::ios_base::failure("synthetic I/O failure");
    }
    void flush() override {
        ++flushes;
        if (flush_failure_) throw std::ios_base::failure("synthetic I/O failure");
    }
    std::atomic<unsigned> writes{0}, flushes{0};
private:
    bool flush_failure_;
};
#endif

json::Object cycle(std::string_view mode, unsigned workers, unsigned attempts,
                   unsigned capacity, bool healthy_siblings = true) {
    const std::uint64_t total_attempts = std::uint64_t{workers} * attempts;
    TelemetryOptions options;
    options.synthetic = true;
    options.queue_capacity = capacity;
    options.drop_report_interval = std::chrono::hours(1);
#ifndef SONDER_BASELINE_TELEMETRY
    StreamFault stream_buffer(mode == "stream-flush");
    std::ostream stream(&stream_buffer);
#endif
    std::ostringstream healthy_stream;
    TelemetryBus bus(options);
#ifndef SONDER_BASELINE_TELEMETRY
    std::shared_ptr<Fault> fault;
    const bool stream_failure = mode == "stream-write" || mode == "stream-flush";
    if (stream_failure) {
        auto sink = std::shared_ptr<TelemetrySink>(make_ostream_sink(stream));
        bus.add_sink(sink);
        bus.add_sink(sink);
    } else if (mode != "healthy" && mode != "healthy-stream") {
        fault = std::make_shared<Fault>(mode == "flush");
        bus.add_sink(fault);
        bus.add_sink(fault);
    }
#else
    require(mode == "healthy" || mode == "healthy-stream", "baseline supports healthy-only control");
#endif
    if (mode == "healthy-stream") bus.add_sink(std::shared_ptr<TelemetrySink>(make_ostream_sink(healthy_stream)));
    auto first = std::make_shared<MemoryTelemetrySink>();
    auto second = std::make_shared<MemoryTelemetrySink>();
    if (healthy_siblings) {
        bus.add_sink(first);
        bus.add_sink(second);
    }
    TelemetryContext context;
    context.session_id = "synthetic-stability";
    std::array<std::vector<unsigned>, 6> admitted;
    std::vector<std::thread> threads;
    const auto start = Clock::now();
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            for (unsigned i = 0; i < attempts; ++i) {
                unsigned id = worker * attempts + i;
                if (bus.emit("stress.event", context, json::Object{{"id", id}})) {
                    admitted[worker].push_back(id);
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    const auto drain_start = Clock::now();
    bus.flush();
    bus.shutdown();
    bus.flush();
    bus.shutdown();
    const auto end = Clock::now();
    std::set<unsigned> accepted;
    for (unsigned worker = 0; worker < workers; ++worker) {
        accepted.insert(admitted[worker].begin(), admitted[worker].end());
    }
    require(accepted.size() == bus.emitted_events(), "accepted counter mismatch");
    if (healthy_siblings) {
        require(accepted.size() + bus.dropped_events() == total_attempts,
                "pressure accounting mismatch");
        auto lines = first->lines();
        require(lines == second->lines(), "healthy sibling mismatch");
        if (mode == "healthy-stream") {
            std::string expected;
            for (const auto& line : lines) expected += line + '\n';
            require(healthy_stream.str() == expected, "healthy stream byte mismatch");
            require(healthy_stream.good() && healthy_stream.exceptions() == std::ios::goodbit,
                    "healthy stream state/mask mismatch");
        }
        std::set<unsigned> delivered;
        std::uint64_t seq = 0, last_drops = 0;
        for (const auto& line : lines) {
            auto parsed = json::parse(line);
            require(parsed.ok(), "invalid envelope");
            const auto& envelope = parsed.value();
            require(envelope.find("sequence")->as_uint() == seq, "cursor gap");
            require(envelope.find("event_id")->as_string() == bus.instance_id() + "-" + std::to_string(seq),
                    "event id mismatch");
            const auto* producer = envelope.find("producer");
            require(producer->find("instance_id")->as_string() == bus.instance_id(), "producer mismatch");
            require(producer->find("synthetic")->as_bool(), "missing synthetic label");
            const auto* attributes = envelope.find("attributes");
            if (envelope.find("event_type")->as_string() == "stress.event") {
                auto id = static_cast<unsigned>(attributes->find("id")->as_uint());
                require(delivered.insert(id).second, "duplicate accepted event");
            } else {
                require(envelope.find("event_type")->as_string() == "telemetry.dropped", "unexpected event");
                auto count = attributes->find("dropped_events")->as_uint();
                require(count >= last_drops && count <= bus.dropped_events(), "drop report mismatch");
                last_drops = count;
            }
            ++seq;
        }
        require(delivered == accepted, "accepted delivery mismatch");
        require(last_drops == bus.dropped_events(), "final drops not reported");
        if (capacity >= total_attempts) require(bus.dropped_events() == 0, "unpressured control dropped");
    } else {
        require(!bus.enabled(TelemetryLevel::metrics), "all failed sinks remain enabled");
        require(!accepted.empty(), "all-failed cycle admitted no events");
    }
#ifndef SONDER_BASELINE_TELEMETRY
    require(bus.failed_sinks() == ((fault || stream_failure) ? 1u : 0u), "distinct failure count mismatch");
    if (stream_failure) {
        require(stream.bad() && stream.exceptions() == std::ios::goodbit, "stream state/mask mismatch");
        if (mode == "stream-flush") {
            require(stream_buffer.flushes == 1, "failed stream flush retried");
            require(stream_buffer.writes > 0 && stream_buffer.writes <= capacity * 2,
                    "stream flush write count mismatch");
        } else {
            require(stream_buffer.writes == 1 && stream_buffer.flushes == 0, "failed stream write retried");
        }
    }
    if (fault) {
        if (mode == "flush") {
            require(fault->flushes == 1, "failed flush retried");
            require(fault->writes > 0 && fault->writes <= capacity * 2, "flush fault write count mismatch");
        } else {
            require(fault->writes == 1 && fault->flushes == 0, "failed write retried");
        }
    }
#endif
    return json::Object{{"mode", std::string(mode)}, {"workers", workers}, {"attempts", total_attempts},
                        {"capacity", capacity}, {"healthy_siblings", healthy_siblings},
                        {"accepted", bus.emitted_events()}, {"queue_drops", bus.dropped_events()},
                        {"filtered", total_attempts - bus.emitted_events() - bus.dropped_events()},
                        {"elapsed_ms", std::chrono::duration<double, std::milli>(end - start).count()},
                        {"drain_ms", std::chrono::duration<double, std::milli>(end - drain_start).count()}};
}

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc == 2 ? argv[1] : "healthy";
        require(argc == 1 || (argc == 2 && (mode == "pressure" || mode == "stream-pressure" || mode == "healthy-stream")),
                "usage: telemetry-stress [pressure|stream-pressure|healthy-stream]");
        bool pressure = mode == "pressure" || mode == "stream-pressure";
        json::Array receipts;
        for (unsigned i = 0; i < 8; ++i) {
            if (!pressure) {
                receipts.emplace_back(cycle(mode, 1, 4096, 4096));
            } else {
#ifndef SONDER_BASELINE_TELEMETRY
                receipts.emplace_back(cycle("healthy", 6, 512, 64));
                const bool streams = mode == "stream-pressure";
                receipts.emplace_back(cycle(streams ? "stream-write" : "write", 6, 512, 64));
                receipts.emplace_back(cycle(streams ? "stream-flush" : "flush", 6, 512, 64));
                receipts.emplace_back(cycle(streams ? "stream-write" : "write", 6, 512, 64, false));
#else
                throw std::runtime_error("baseline pressure mode disabled");
#endif
            }
        }
        std::cout << json::Value(json::Object{{"synthetic", true}, {"scope", "Synthetic telemetry transport only; no provider/model calls"},
                                            {"cycles", std::move(receipts)}}).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
