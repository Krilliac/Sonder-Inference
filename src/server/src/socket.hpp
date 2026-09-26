// Internal: portable TCP socket helpers for the server module (POSIX and
// Winsock). Patterns mirror src/net/http_client.cpp: non-blocking sockets,
// poll()/select() waits in short slices so stop flags and deadlines are
// observed, MSG_NOSIGNAL (or SO_NOSIGPIPE) instead of SIGPIPE.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "sonder/inference/error.hpp"

namespace sonder::inference::server::detail {

#if defined(_WIN32)
using native_socket = std::uintptr_t;  // SOCKET
#else
using native_socket = int;
#endif

class Socket {
public:
    Socket() noexcept;
    explicit Socket(native_socket s) noexcept : s_(s) {}
    ~Socket();
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& o) noexcept;
    Socket& operator=(Socket&& o) noexcept;

    [[nodiscard]] native_socket get() const noexcept { return s_; }
    [[nodiscard]] bool valid() const noexcept;
    void close() noexcept;
    // Half-close / full shutdown without releasing the descriptor (wakes a
    // thread blocked on it).
    void shutdown_both() noexcept;
    void shutdown_write() noexcept;

private:
    native_socket s_;
};

// One-time Winsock initialisation (no-op elsewhere).
void ensure_network();

struct Listener {
    Socket socket;
    std::uint16_t port = 0;   // bound port (resolved when 0 was requested)
    bool ipv6 = false;
};

// Binds and listens on host:port. Errors: unavailable ("address in use" is
// reported with in_use = true), invalid_argument (unresolvable host).
Result<Listener> listen_tcp(const std::string& host, std::uint16_t port, bool& in_use);

enum class WaitResult { ready, timeout, stopped, error };

// Waits for readability (or writability) in 50 ms slices until `timeout`
// elapses or `*stop` becomes true (null = never).
WaitResult wait_socket(native_socket s, bool for_write, std::chrono::milliseconds timeout,
                       const std::atomic<bool>* stop);

// Non-blocking accept. Returns an invalid Socket when nothing is pending.
Socket accept_client(const Socket& listener);

// > 0 bytes read, 0 orderly close, -1 error, -2 would block.
long long recv_some(native_socket s, char* buf, std::size_t len);

// Sends everything, waiting while the peer's window is full. Fails with
// timeout when no progress is possible for `stall_timeout`, cancelled when
// `*stop` becomes true, io_error when the peer is gone.
Status send_all(native_socket s, std::string_view data, std::chrono::milliseconds stall_timeout,
                const std::atomic<bool>* stop);

// True when the peer closed or reset the connection. Discards (bounded)
// unexpected bytes the client sent after its request. Never blocks.
bool peer_gone(native_socket s);

// Blocking-with-timeout client connect (used by tests and diagnostics).
Result<Socket> connect_tcp(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout);

}  // namespace sonder::inference::server::detail
