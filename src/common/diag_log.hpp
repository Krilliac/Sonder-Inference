// Internal: process-wide, non-blocking sink for diagnostic lines written by
// the library itself (llama.cpp's log callback, the insecure-TLS warning).
//
// Those lines used to go to stderr synchronously on the calling thread, so a
// stalled stderr (a full pipe, a paused console) stalled inference or a
// connect. They now go through the bounded LogQueue added for the server
// access log (common/log_queue.hpp): log() takes a mutex for a few
// instructions and returns, and one logger thread writes to stderr. When the
// queue is full the line is dropped and counted; a "lines dropped" notice is
// written once stderr drains.
//
// Threading: every function is safe from any thread. The queue and its logger
// thread are created on the first log() and live for the rest of the process
// (never destroyed, so a line logged during static destruction is still safe).
#pragma once

#include <cstddef>
#include <string>

#include "common/log_queue.hpp"

namespace sonder::inference::diag {

// Lines the queue holds before dropping new ones.
inline constexpr std::size_t kDefaultCapacity = 1024;

// Queues one line (include the trailing newline) for stderr. Never writes to
// stderr on the calling thread and never waits for it. Returns false when the
// line was dropped. Does not throw.
bool log(std::string line) noexcept;

// Replaces the destination of diagnostic lines. A null `sink` restores the
// default (stderr). The previous queue is stopped with a short patience: a
// sink stuck in a call is abandoned, never waited for. `sink` runs on the
// logger thread only; for tests and for hosts that route diagnostics
// elsewhere.
void set_sink(server::detail::LogQueue::Sink sink, std::size_t capacity = kDefaultCapacity);

// Counters of the current queue (written, dropped, queued, capacity).
[[nodiscard]] server::detail::LogQueueStats stats();

}  // namespace sonder::inference::diag
