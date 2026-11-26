#include "common/log_queue.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <system_error>
#include <utility>

namespace sonder::inference::server::detail {

namespace {

// Patience of a destructor that runs without a stop() before it.
constexpr std::chrono::milliseconds kDestructorPatience{100};

}  // namespace

// Shared between the LogQueue and its logger thread; whichever lets go last
// destroys it (the logger thread, when it was detached while stuck in the
// sink).
struct LogQueue::State {
    struct Entry {
        std::string line;
        // A "lines dropped" report: bypasses `capacity` and is not counted
        // as written or dropped.
        bool report = false;
    };

    State(Sink sink_fn, std::size_t max_lines, DropLine drop_line_fn, std::chrono::milliseconds report_interval)
        : sink(std::move(sink_fn)),
          drop_line(std::move(drop_line_fn)),
          capacity(max_lines == 0 ? 1 : max_lines),
          drop_report_interval(report_interval) {}

    // Immutable after construction.
    const Sink sink;
    const DropLine drop_line;
    const std::size_t capacity;
    // A live "lines dropped" report is queued on the first drop, then at
    // most this often (measured on the steady clock).
    const std::chrono::milliseconds drop_report_interval;

    std::mutex mu;
    std::condition_variable work_cv;  // logger thread: lines queued, or stop
    std::condition_variable exit_cv;  // stop(): the logger thread has exited

    // Guarded by `mu`.
    std::deque<Entry> queue;
    std::size_t queued_lines = 0;   // entries of `queue` that are not reports
    std::size_t batch_pending = 0;  // lines of the logger's batch not yet handed to the sink
    std::uint64_t reported = 0;     // `dropped` when the last report was queued
    std::chrono::steady_clock::time_point last_report{};
    bool closed = false;     // push() refuses new lines
    bool stopping = false;   // the logger thread exits once the queue is empty
    bool abandoned = false;  // the logger thread must not enter the sink again
    bool exited = false;     // the logger thread has left its loop

    // Counters for stats(): readable without `mu`.
    std::atomic<std::uint64_t> written{0};
    std::atomic<std::uint64_t> dropped{0};
    std::atomic<std::size_t> queued{0};  // mirrors queued_lines
};

LogQueue::LogQueue(Sink sink, std::size_t capacity, DropLine drop_line,
                   std::chrono::milliseconds drop_report_interval)
    : state_(std::make_shared<State>(std::move(sink), capacity, std::move(drop_line), drop_report_interval)) {
    if (!state_->sink) {
        state_->closed = true;  // nothing to write to
        return;
    }
    try {
        writer_ = std::thread([st = state_] { writer_loop(st); });
    } catch (const std::system_error&) {
        // No logger thread: lines are dropped and counted (visible through
        // stats()) instead of being written on the producers' threads.
        state_->closed = true;
    }
}

LogQueue::~LogQueue() {
    try {
        (void)stop(kDestructorPatience);
    } catch (...) {
        // A failing mutex or join; nothing left to try.
    }
    if (writer_.joinable()) {
        writer_.detach();  // only when stop() threw: a std::thread must not die joinable
    }
}

bool LogQueue::push(std::string line) noexcept {
    State& st = *state_;
    bool accepted = false;
    try {
        std::lock_guard<std::mutex> lock(st.mu);
        if (!st.closed && st.queued_lines < st.capacity) {
            st.queue.push_back(State::Entry{std::move(line), false});
            ++st.queued_lines;
            st.queued.store(st.queued_lines, std::memory_order_relaxed);
            accepted = true;
        }
    } catch (...) {
        // No memory for the queue node (push_back has no effect when it
        // throws) or a failing mutex. The guard is released by the unwinding
        // before this handler runs: the line is lost, the caller is not.
    }
    if (!accepted) {
        st.dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    st.work_cv.notify_one();
    return true;
}

void LogQueue::writer_loop(const std::shared_ptr<State>& st) {
    // Declared before the lock so that it is destroyed after the lock is
    // released: line storage is never freed under `mu`.
    std::deque<State::Entry> batch;
    std::unique_lock<std::mutex> lock(st->mu);
    for (;;) {
        st->work_cv.wait(lock, [&st] { return st->stopping || st->abandoned || !st->queue.empty(); });
        if (st->abandoned || st->queue.empty()) {
            break;  // given up on, or stopping with nothing left to write
        }
        batch.swap(st->queue);  // `batch` is empty here
        st->batch_pending = st->queued_lines;
        st->queued_lines = 0;
        st->queued.store(0, std::memory_order_relaxed);
        for (const State::Entry& entry : batch) {
            // `mu` is held at this point. stop() sets `abandoned` under it,
            // so once stop() has given up no further sink call starts.
            if (st->abandoned) {
                break;
            }
            if (!entry.report) {
                --st->batch_pending;
            }
            lock.unlock();
            bool ok = true;
            try {
                st->sink(entry.line);
            } catch (...) {
                ok = false;
            }
            if (!entry.report) {
                if (ok) {
                    st->written.fetch_add(1, std::memory_order_relaxed);
                } else {
                    st->dropped.fetch_add(1, std::memory_order_relaxed);
                }
            }
            lock.lock();
        }
        if (st->abandoned) {
            break;  // stop() has already counted the rest of the batch as dropped
        }
        // Live drop report: on the first drop, then rate limited. While
        // stop() runs (`closed`) its final report covers the count instead.
        std::uint64_t dropped_now = 0;
        bool report = false;
        if (!st->closed && st->drop_line) {
            dropped_now = st->dropped.load(std::memory_order_relaxed);
            report = dropped_now > st->reported &&
                     (st->reported == 0 ||
                      std::chrono::steady_clock::now() - st->last_report >= st->drop_report_interval);
        }
        lock.unlock();
        batch.clear();  // frees the written lines with `mu` released
        std::string text;
        if (report) {
            try {
                text = st->drop_line(dropped_now, false);
            } catch (...) {
                report = false;
            }
        }
        lock.lock();
        if (report && !st->closed && !st->abandoned) {
            try {
                st->queue.push_back(State::Entry{std::move(text), true});
                st->reported = dropped_now;
                st->last_report = std::chrono::steady_clock::now();
            } catch (...) {
                // No memory for the report; the next batch tries again.
            }
        }
    }
    st->exited = true;
    st->exit_cv.notify_all();
}

bool LogQueue::stop(std::chrono::milliseconds patience) {
    if (!writer_.joinable()) {
        return drained_;  // no logger thread, or stop() already ran
    }
    State& st = *state_;
    std::unique_lock<std::mutex> lock(st.mu);
    st.closed = true;  // from here on push() refuses and counts
    const std::uint64_t dropped_now = st.dropped.load(std::memory_order_relaxed);
    const bool report = st.drop_line && dropped_now > st.reported;
    lock.unlock();
    // Final accounting line for drops not reported live yet. Built with `mu`
    // released; queued past `capacity` on purpose.
    std::string text;
    bool have_text = false;
    if (report) {
        try {
            text = st.drop_line(dropped_now, true);
            have_text = true;
        } catch (...) {
            // The count stays visible through stats().
        }
    }
    lock.lock();
    if (have_text) {
        try {
            st.queue.push_back(State::Entry{std::move(text), true});
            st.reported = dropped_now;
        } catch (...) {
            // No memory for the report.
        }
    }
    st.stopping = true;
    st.work_cv.notify_all();
    if (st.exit_cv.wait_for(lock, patience, [&st] { return st.exited; })) {
        lock.unlock();
        writer_.join();
        drained_ = true;
        return true;
    }
    // The sink has not returned within `patience`. Whatever was not handed to
    // it yet is lost; the logger thread is left to finish its current call on
    // its own. It only holds the shared state, and it checks `abandoned`
    // under `mu` before every sink call, so it never enters the sink again.
    st.abandoned = true;
    const std::uint64_t lost =
        static_cast<std::uint64_t>(st.queued_lines) + static_cast<std::uint64_t>(st.batch_pending);
    st.queued_lines = 0;
    st.batch_pending = 0;
    st.queued.store(0, std::memory_order_relaxed);
    st.dropped.fetch_add(lost, std::memory_order_relaxed);
    std::deque<State::Entry> discarded;
    discarded.swap(st.queue);
    st.work_cv.notify_all();
    lock.unlock();
    writer_.detach();
    drained_ = false;
    return false;  // `discarded` is freed here, with `mu` released
}

LogQueueStats LogQueue::stats() const noexcept {
    const State& st = *state_;
    LogQueueStats s;
    s.written = st.written.load(std::memory_order_relaxed);
    s.dropped = st.dropped.load(std::memory_order_relaxed);
    s.queued = st.queued.load(std::memory_order_relaxed);
    s.capacity = st.capacity;
    return s;
}

}  // namespace sonder::inference::server::detail
