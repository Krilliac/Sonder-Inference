// Portable loopback socket helpers shared by the fault-injection child
// (fault_child.cpp) and the fault tests. Blocking sockets, IPv4 loopback only,
// no TLS. Test code: it never leaves the test tree and never touches a network
// other than 127.0.0.1.
#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <cerrno>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fault_net {

#if defined(_WIN32)
using Sock = SOCKET;
inline constexpr Sock kBad = INVALID_SOCKET;
#else
using Sock = int;
inline constexpr Sock kBad = -1;
#endif

// Idempotent process-wide initialisation (WSAStartup on Windows).
inline void init() {
    static const bool once = [] {
#if defined(_WIN32)
        WSADATA data{};
        (void)WSAStartup(MAKEWORD(2, 2), &data);
#endif
        return true;
    }();
    (void)once;
}

inline void close_sock(Sock s) {
    if (s == kBad)
        return;
#if defined(_WIN32)
    closesocket(s);
#else
    ::close(s);
#endif
}

inline bool set_int_opt(Sock s, int level, int name, int value) {
    return setsockopt(s, level, name, reinterpret_cast<const char *>(&value), sizeof(value)) == 0;
}

inline bool set_send_timeout_ms(Sock s, int ms) {
#if defined(_WIN32)
    const DWORD timeout = static_cast<DWORD>(ms);
    return setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&timeout), sizeof(timeout)) == 0;
#else
    timeval tv{};
    tv.tv_sec = ms / 1000;
    tv.tv_usec = static_cast<decltype(tv.tv_usec)>((ms % 1000) * 1000);
    return setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

// Listens on 127.0.0.1:port. Returns kBad when the port cannot be bound.
inline Sock listen_loopback(std::uint16_t port) {
    init();
    const Sock s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == kBad)
        return kBad;
#if !defined(_WIN32)
    // On Windows SO_REUSEADDR would let a second process hijack the port.
    (void)set_int_opt(s, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::bind(s, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0 || ::listen(s, 16) != 0) {
        close_sock(s);
        return kBad;
    }
    return s;
}

inline Sock accept_conn(Sock listener) { return ::accept(listener, nullptr, nullptr); }

// Blocking connect to 127.0.0.1:port. recv_buffer > 0 sets SO_RCVBUF before
// connecting (used to make a non-reading peer back-pressure quickly).
inline Sock connect_loopback(std::uint16_t port, int recv_buffer = 0) {
    init();
    const Sock s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == kBad)
        return kBad;
    if (recv_buffer > 0)
        (void)set_int_opt(s, SOL_SOCKET, SO_RCVBUF, recv_buffer);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::connect(s, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0) {
        close_sock(s);
        return kBad;
    }
    return s;
}

// Sends all of [data, data+size) or as much as succeeded before an error or
// send timeout. Returns the number of bytes accepted by the kernel.
inline std::size_t send_all(Sock s, const char *data, std::size_t size) {
#if defined(MSG_NOSIGNAL)
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    std::size_t sent = 0;
    while (sent < size) {
        const std::size_t want = std::min<std::size_t>(size - sent, 1u << 20);
        const auto n = ::send(s, data + sent, static_cast<int>(want), flags);
        if (n <= 0)
            break;
        sent += static_cast<std::size_t>(n);
    }
    return sent;
}

// recv(); returns bytes read, 0 on orderly close, negative on error.
inline long recv_some(Sock s, char *buffer, std::size_t size) {
    return static_cast<long>(::recv(s, buffer, static_cast<int>(size), 0));
}

// True when the peer has closed or reset the connection and nothing is left
// to read (a readable socket whose MSG_PEEK returns 0 or an error).
inline bool peer_closed(Sock s) {
    fd_set readable;
    FD_ZERO(&readable);
    FD_SET(s, &readable);
    timeval tv{};
    const int ready = ::select(static_cast<int>(s) + 1, &readable, nullptr, nullptr, &tv);
    if (ready <= 0)
        return false;
    char byte = 0;
    const auto n = ::recv(s, &byte, 1, MSG_PEEK);
    return n <= 0;
}

} // namespace fault_net
