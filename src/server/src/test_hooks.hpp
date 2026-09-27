// Internal fault-injection hooks for the server module's tests. Not part of
// the public API; production code only reads them.
#pragma once

#include <atomic>

namespace sonder::inference::server::detail {

// While positive, each chat request's disconnect-watcher thread creation
// fails (and decrements it) as if the OS refused to start a thread.
inline std::atomic<int>& fail_watcher_spawns_for_test() {
    static std::atomic<int> remaining{0};
    return remaining;
}

// True (and consumes one) when the next watcher spawn must fail.
inline bool take_watcher_spawn_failure() {
    auto& remaining = fail_watcher_spawns_for_test();
    int n = remaining.load();
    while (n > 0) {
        if (remaining.compare_exchange_weak(n, n - 1)) {
            return true;
        }
    }
    return false;
}

}  // namespace sonder::inference::server::detail
