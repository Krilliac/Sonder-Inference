#include "common/diag_log.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>

namespace sonder::inference::diag {

namespace {

using server::detail::LogQueue;

void stderr_sink(const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

// Runs on the logger thread and inside LogQueue::stop(); owns nothing.
std::string drop_notice(std::uint64_t dropped, bool final_report) {
    return std::string("[sonder-log] ") + std::to_string(dropped) +
           (final_report ? " diagnostic lines dropped in total (log sink too slow)\n"
                         : " diagnostic lines dropped so far (log sink too slow)\n");
}

struct Holder {
    std::mutex mutex;
    std::shared_ptr<LogQueue> queue;  // guarded by `mutex`
};

// Leaked on purpose: the logger thread must outlive static destruction.
Holder& holder() {
    static Holder* const h = new Holder();
    return *h;
}

std::shared_ptr<LogQueue> make_queue(LogQueue::Sink sink, std::size_t capacity) {
    return std::make_shared<LogQueue>(std::move(sink), capacity, &drop_notice);
}

// The queue in use, created on first use. The caller pushes outside the
// mutex, so the lock is held for one shared_ptr copy only.
std::shared_ptr<LogQueue> current() {
    Holder& h = holder();
    std::lock_guard<std::mutex> lock(h.mutex);
    if (!h.queue) {
        h.queue = make_queue(&stderr_sink, kDefaultCapacity);
    }
    return h.queue;
}

}  // namespace

bool log(std::string line) noexcept {
    try {
        return current()->push(std::move(line));
    } catch (...) {
        return false;  // out of memory or a failing mutex: the line is lost, the caller is not
    }
}

void set_sink(LogQueue::Sink sink, std::size_t capacity) {
    if (!sink) {
        sink = &stderr_sink;
    }
    std::shared_ptr<LogQueue> fresh = make_queue(std::move(sink), capacity);
    std::shared_ptr<LogQueue> old;
    {
        Holder& h = holder();
        std::lock_guard<std::mutex> lock(h.mutex);
        old = std::exchange(h.queue, std::move(fresh));
    }
    if (old) {
        // Outside the mutex: a stuck sink costs at most the patience below.
        (void)old->stop(std::chrono::milliseconds(200));
    }
}

server::detail::LogQueueStats stats() { return current()->stats(); }

}  // namespace sonder::inference::diag
