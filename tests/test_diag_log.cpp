// Process-wide diagnostic log (src/common/diag_log.*): a stalled sink must
// never stall the caller.
#include <doctest/doctest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/diag_log.hpp"

namespace {

using namespace std::chrono_literals;
namespace diag = sonder::inference::diag;

// Records lines; while closed, blocks inside the first call like a stalled stderr.
class GatedSink {
public:
    void operator()(const std::string& line) {
        std::unique_lock<std::mutex> lock(state_->mu);
        state_->entered = true;
        state_->cv.notify_all();
        state_->cv.wait(lock, [this] { return state_->open; });
        state_->lines.push_back(line);
    }
    void open() {
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->open = true;
        }
        state_->cv.notify_all();
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(state_->mu);
        return state_->cv.wait_for(lock, 5s, [this] { return state_->entered; });
    }
    bool wait_lines(std::size_t n) {
        std::unique_lock<std::mutex> lock(state_->mu);
        return state_->cv.wait_for(lock, 5s, [this, n] { return state_->lines.size() >= n; });
    }
    std::vector<std::string> lines() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->lines;
    }
    void start_open() { open(); }

private:
    struct State {
        mutable std::mutex mu;
        std::condition_variable cv;
        bool open = false;
        bool entered = false;
        std::vector<std::string> lines;
    };
    // Shared so the std::function copies the queue keeps all see one state.
    std::shared_ptr<State> state_ = std::make_shared<State>();
};

}  // namespace

TEST_CASE("diag log delivers lines to the sink in order") {
    GatedSink sink;
    sink.start_open();
    diag::set_sink(sink);
    CHECK(diag::log("one\n"));
    CHECK(diag::log("two\n"));
    REQUIRE(sink.wait_lines(2));
    const auto lines = sink.lines();
    CHECK(lines[0] == "one\n");
    CHECK(lines[1] == "two\n");
    diag::set_sink(nullptr);
}

TEST_CASE("diag log never blocks the caller while the sink is stuck") {
    GatedSink sink;
    diag::set_sink(sink, 4);
    REQUIRE(diag::log("first\n"));
    REQUIRE(sink.wait_entered());  // the logger thread is now stuck inside the sink

    auto done = std::async(std::launch::async, [] {
        int accepted = 0;
        for (int i = 0; i < 100; ++i) {
            if (diag::log("line\n")) ++accepted;
        }
        return accepted;
    });
    const bool returned = done.wait_for(5s) == std::future_status::ready;
    CHECK(returned);
    if (!returned) sink.open();  // do not hang the suite on a regression
    const int accepted = done.get();
    CHECK(accepted == 4);  // capacity 4: the rest are dropped, not waited for
    CHECK(diag::stats().dropped >= 96);
    sink.open();
    diag::set_sink(nullptr);
}
