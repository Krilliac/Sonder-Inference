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

enum class AcceptStatus {
    accepted,
    none_pending,        // nothing left in the backlog (EAGAIN) or a transient per-connection failure
    out_of_descriptors,  // EMFILE / ENFILE (WSAEMFILE): the process or system is out of descriptors
    failed,              // any other error (ENOBUFS, ENOMEM, ...): back off before retrying
};

struct Accepted {
    Socket socket;  // valid only when status == accepted
    AcceptStatus status = AcceptStatus::none_pending;
    int native_error = 0;
};

// Non-blocking accept. Distinguishes an empty backlog from resource errors so
// the accept loop can back off instead of spinning on a listener that stays
// readable while the backlog cannot be accepted.
Accepted accept_client(const Socket& listener);

// Descriptor held in reserve (POSIX: /dev/null) so that, when accept() fails
// with EMFILE, one descriptor can be released to accept and answer the
// oldest pending connection instead of leaving the backlog stuck. No-op on
// Windows, where descriptor exhaustion is not per-process. Starts empty;
// call acquire().
class ReserveDescriptor {
public:
    ReserveDescriptor() noexcept;
    ~ReserveDescriptor();
    ReserveDescriptor(const ReserveDescriptor&) = delete;
    ReserveDescriptor& operator=(const ReserveDescriptor&) = delete;
    [[nodiscard]] bool held() const noexcept;
    void release() noexcept;
    // Re-acquires the descriptor; false when none is available yet.
    bool acquire() noexcept;

private:
    int fd_ = -1;
};

// Lingering close: after the response has been sent and the write side shut
// down, reads and discards what the client is still sending (at most
// `max_bytes`, for at most `max_time`, stopping early after `idle` without
// data, on EOF, on error, or when `*stop` becomes true). A client that
// writes its whole request body before reading the response (Python urllib,
// for example) otherwise gets a TCP reset instead of an early 401/413 reply.
void drain_input(native_socket s, std::uint64_t max_bytes, std::chrono::milliseconds max_time,
                 std::chrono::milliseconds idle, const std::atomic<bool>* stop);

// Soft limit on open descriptors (RLIMIT_NOFILE); 0 when unknown or unlimited.
std::uint64_t descriptor_limit() noexcept;

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
