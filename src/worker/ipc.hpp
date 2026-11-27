#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::worker {

inline constexpr std::uint32_t kMaxFrameBytes = 1u << 20;

// Owns one duplex connection. Exactly one reader and one writer may run
// concurrently; close() is idempotent and may interrupt either from any thread.
// Frame buffers hold at most kMaxFrameBytes payload bytes plus the 4-byte prefix.
// Header and payload reads each get their own timeout. A timeout before any frame
// bytes were read is retryable; all partial-frame failures close the connection.
class Channel {
  public:
    Channel() = default;
    Channel(const Channel &) = delete;
    Channel &operator=(const Channel &) = delete;
    virtual ~Channel() = default;
    virtual Result<std::string> read_frame(std::chrono::milliseconds timeout, const CancellationToken &cancel = {}) = 0;
    virtual Status write_frame(std::string_view payload, std::chrono::milliseconds timeout,
                               const CancellationToken &cancel = {}) = 0;
    virtual void close() = 0;
};

// Owns a private endpoint and its filesystem/OS resources. Single accept owner;
// destroying the listener must not race accept(). Accepted channels own their
// connections independently. One worker connection is accepted per listener.
class Listener {
  public:
    Listener() = default;
    Listener(const Listener &) = delete;
    Listener &operator=(const Listener &) = delete;
    virtual ~Listener() = default;
    [[nodiscard]] virtual std::string endpoint() const = 0;
    virtual Result<std::shared_ptr<Channel>> accept(std::chrono::milliseconds timeout,
                                                    const CancellationToken &cancel = {}) = 0;
};

Result<std::unique_ptr<Listener>> make_listener();
Result<std::shared_ptr<Channel>> connect(std::string_view endpoint, std::chrono::milliseconds timeout,
                                         const CancellationToken &cancel = {});

} // namespace sonder::inference::worker
