// Internal: bounded queue between the threads that produce access-log and
// diagnostic lines (request threads, the accept thread) and the one logger
// thread that calls ServerOptions::log (docs/SERVER.md "Access log").
//
// The sink may block for as long as it likes (an unbuffered stderr on a
// stalled console or a saturated disk): producers never call it and never
// wait for it. push() takes the queue mutex for a few instructions and
// returns; when `capacity` lines are already waiting the new line is dropped
// and counted. The logger thread swaps the queued lines out under the mutex
// and calls the sink with the mutex released.
//
// Lifetime: the queue state lives in a shared_ptr that the logger thread
// co-owns, so stop() can give up on a logger that is stuck inside the sink
// and detach it: that thread only touches the shared state and its own copies
// of `sink` and `drop_line`, never the LogQueue or its owner.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace sonder::inference::server::detail {

struct LogQueueStats {
    // Lines the sink returned from normally.
    std::uint64_t written = 0;
    // Lines lost: refused because the queue was full or stopped, discarded at
    // the stop() deadline, or the sink threw.
    std::uint64_t dropped = 0;
    // Lines waiting in the queue (not the batch the logger thread already
    // took out). Never above `capacity`.
    std::size_t queued = 0;
    std::size_t capacity = 0;
};

class LogQueue {
public:
    using Sink = std::function<void(const std::string& line)>;
    // Text of a "lines dropped" report: `dropped` is the total so far,
    // `final_report` is true for the one queued by stop().
    using DropLine = std::function<std::string(std::uint64_t dropped, bool final_report)>;

    // Starts the logger thread. `sink` and `drop_line` are copied into the
    // shared state. `sink` runs on the logger thread only, one call at a
    // time, in push order. `drop_line` runs on the logger thread and inside
    // stop(), so it can be called by two threads at the same time: it must
    // be safe to call concurrently with itself. It must not block and must
    // own everything it uses, because a detached logger thread may still
    // call it after stop() returned false.
    // A null `sink` (or a logger thread the OS refuses to start) leaves the
    // queue without a writer: every push() is then counted as dropped.
    // `capacity` 0 is treated as 1. `drop_report_interval` is the least time
    // between two live "lines dropped" reports (the first drop is reported
    // at once); it is a parameter for the tests, the server uses the default.
    LogQueue(Sink sink, std::size_t capacity, DropLine drop_line,
             std::chrono::milliseconds drop_report_interval = std::chrono::seconds(10));
    // stop() with a short patience when the owner has not called it.
    ~LogQueue();
    LogQueue(const LogQueue&) = delete;
    LogQueue& operator=(const LogQueue&) = delete;

    // Queues one line. Never calls the sink and never waits for it. Returns
    // false when the line was dropped (queue full, stopped, or out of
    // memory); the drop is counted. Any thread, also during and after stop().
    bool push(std::string line) noexcept;

    // Refuses further lines, queues a final "lines dropped" report when drops
    // are still unreported (it bypasses `capacity`), and gives the logger
    // thread `patience` to write what is queued.
    //
    // Returns true when no logger thread is left running and stop() itself
    // discarded nothing: the thread handed every accepted line to the sink
    // and was joined, or there never was a thread (null sink, or the OS
    // refused to start one), in which case no line was ever accepted. True
    // does not mean that no line was lost: lines refused by push() and lines
    // the sink threw on are counted in stats().dropped either way.
    //
    // Returns false when the sink did not return within `patience`: the
    // lines not yet handed to it are counted as dropped and the thread is
    // detached. After stop() returns, the sink is never entered again; only
    // a call that was already in progress may still be running.
    //
    // Not safe against a concurrent stop() or the destructor; a repeated
    // call returns the first result.
    bool stop(std::chrono::milliseconds patience);

    // Counters, read from atomics: no lock is taken, so this never waits for
    // a producer, the logger thread or the sink. Any thread.
    [[nodiscard]] LogQueueStats stats() const noexcept;

private:
    struct State;
    static void writer_loop(const std::shared_ptr<State>& st);

    const std::shared_ptr<State> state_;
    // Touched by the constructor, stop() and the destructor only.
    std::thread writer_;
    bool drained_ = true;
};

}  // namespace sonder::inference::server::detail
