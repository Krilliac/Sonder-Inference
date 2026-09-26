#include "socket.hpp"

#include <algorithm>
#include <mutex>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <cerrno>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace sonder::inference::server::detail {

namespace {

#if defined(_WIN32)
constexpr native_socket kInvalid = static_cast<native_socket>(INVALID_SOCKET);
SOCKET raw(native_socket s) { return static_cast<SOCKET>(s); }
int last_error() { return WSAGetLastError(); }
bool would_block(int err) { return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS || err == WSAEINTR; }
bool addr_in_use(int err) { return err == WSAEADDRINUSE || err == WSAEACCES; }
void set_nonblocking(native_socket s) {
    u_long mode = 1;
    ioctlsocket(raw(s), FIONBIO, &mode);
}
#else
constexpr native_socket kInvalid = -1;
int raw(native_socket s) { return s; }
int last_error() { return errno; }
bool would_block(int err) { return err == EAGAIN || err == EWOULDBLOCK || err == EINPROGRESS || err == EINTR; }
bool addr_in_use(int err) { return err == EADDRINUSE; }
void set_nonblocking(native_socket s) {
    const int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, flags | O_NONBLOCK);
}
#endif

void no_sigpipe(native_socket s) {
#if defined(SO_NOSIGPIPE)
    int one = 1;
    setsockopt(raw(s), SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
    (void)s;
#endif
}

#if !defined(_WIN32)
int send_flags() {
#  if defined(MSG_NOSIGNAL)
    return MSG_NOSIGNAL;
#  else
    return 0;
#  endif
}
#endif

// Single wait slice; returns >0 ready, 0 timeout, <0 error.
int wait_once(native_socket s, bool for_write, int ms) {
#if defined(_WIN32)
    fd_set set;
    fd_set ex;
    FD_ZERO(&set);
    FD_ZERO(&ex);
    FD_SET(raw(s), &set);
    FD_SET(raw(s), &ex);
    timeval tv{};
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    return ::select(0, for_write ? nullptr : &set, for_write ? &set : nullptr, &ex, &tv);
#else
    pollfd pfd{};
    pfd.fd = s;
    pfd.events = for_write ? POLLOUT : POLLIN;
    return ::poll(&pfd, 1, ms);
#endif
}

}  // namespace

Socket::Socket() noexcept : s_(kInvalid) {}
Socket::~Socket() { close(); }
Socket::Socket(Socket&& o) noexcept : s_(o.s_) { o.s_ = kInvalid; }
Socket& Socket::operator=(Socket&& o) noexcept {
    if (this != &o) {
        close();
        s_ = o.s_;
        o.s_ = kInvalid;
    }
    return *this;
}
bool Socket::valid() const noexcept { return s_ != kInvalid; }

void Socket::close() noexcept {
    if (s_ != kInvalid) {
#if defined(_WIN32)
        closesocket(raw(s_));
#else
        ::close(s_);
#endif
        s_ = kInvalid;
    }
}

void Socket::shutdown_both() noexcept {
    if (s_ != kInvalid) {
#if defined(_WIN32)
        ::shutdown(raw(s_), SD_BOTH);
#else
        ::shutdown(s_, SHUT_RDWR);
#endif
    }
}

void Socket::shutdown_write() noexcept {
    if (s_ != kInvalid) {
#if defined(_WIN32)
        ::shutdown(raw(s_), SD_SEND);
#else
        ::shutdown(s_, SHUT_WR);
#endif
    }
}

void ensure_network() {
#if defined(_WIN32)
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    });
#endif
}

Result<Listener> listen_tcp(const std::string& host, std::uint16_t port, bool& in_use) {
    ensure_network();
    in_use = false;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* list = nullptr;
    const std::string port_text = std::to_string(port);
    const std::string bind_host = host.size() > 2 && host.front() == '[' && host.back() == ']'
                                      ? host.substr(1, host.size() - 2)
                                      : host;
    if (getaddrinfo(bind_host.c_str(), port_text.c_str(), &hints, &list) != 0 || list == nullptr) {
        return Status(ErrorCode::invalid_argument, "cannot resolve listen address " + host);
    }
    Status last(ErrorCode::unavailable, "no usable address for " + host);
    for (addrinfo* ai = list; ai != nullptr; ai = ai->ai_next) {
        Socket sock(static_cast<native_socket>(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol)));
        if (!sock.valid()) {
            continue;
        }
        int one = 1;
#if defined(_WIN32)
        setsockopt(raw(sock.get()), SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&one),
                   sizeof(one));
#else
        // Allows an immediate restart while old connections sit in TIME_WAIT;
        // Linux still refuses to bind over an active listener.
        setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
        if (::bind(raw(sock.get()), ai->ai_addr, static_cast<int>(ai->ai_addrlen)) != 0) {
            const int err = last_error();
            if (addr_in_use(err)) {
                in_use = true;
                last = Status(ErrorCode::unavailable, "address " + host + ":" + port_text + " is already in use");
            } else {
                last = Status(ErrorCode::unavailable, "cannot bind " + host + ":" + port_text);
            }
            continue;
        }
        if (::listen(raw(sock.get()), 128) != 0) {
            last = Status(ErrorCode::unavailable, "cannot listen on " + host + ":" + port_text);
            continue;
        }
        set_nonblocking(sock.get());
        sockaddr_storage addr{};
        socklen_t len = sizeof(addr);
        Listener out;
        if (getsockname(raw(sock.get()), reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
            if (addr.ss_family == AF_INET6) {
                out.port = ntohs(reinterpret_cast<const sockaddr_in6*>(&addr)->sin6_port);
                out.ipv6 = true;
            } else {
                out.port = ntohs(reinterpret_cast<const sockaddr_in*>(&addr)->sin_port);
            }
        }
        out.socket = std::move(sock);
        freeaddrinfo(list);
        in_use = false;
        return out;
    }
    freeaddrinfo(list);
    return last;
}

WaitResult wait_socket(native_socket s, bool for_write, std::chrono::milliseconds timeout,
                       const std::atomic<bool>* stop) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        if (stop != nullptr && stop->load(std::memory_order_acquire)) {
            return WaitResult::stopped;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return WaitResult::timeout;
        }
        const auto slice = std::min<std::chrono::steady_clock::duration>(deadline - now, std::chrono::milliseconds(50));
        const int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(slice).count()) + 1;
        const int rc = wait_once(s, for_write, ms);
        if (rc > 0) {
            return WaitResult::ready;
        }
        if (rc < 0 && !would_block(last_error())) {
            return WaitResult::error;
        }
    }
}

Socket accept_client(const Socket& listener) {
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    const auto s = ::accept(raw(listener.get()), reinterpret_cast<sockaddr*>(&addr), &len);
    Socket client(static_cast<native_socket>(s));
    if (!client.valid()) {
        return Socket();
    }
    set_nonblocking(client.get());
    no_sigpipe(client.get());
    int one = 1;
    setsockopt(raw(client.get()), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    return client;
}

long long recv_some(native_socket s, char* buf, std::size_t len) {
    const int cap = static_cast<int>(std::min<std::size_t>(len, 1 << 20));
#if defined(_WIN32)
    const int n = ::recv(raw(s), buf, cap, 0);
#else
    const auto n = ::recv(s, buf, static_cast<std::size_t>(cap), 0);
#endif
    if (n > 0) {
        return static_cast<long long>(n);
    }
    if (n == 0) {
        return 0;
    }
    return would_block(last_error()) ? -2 : -1;
}

Status send_all(native_socket s, std::string_view data, std::chrono::milliseconds stall_timeout,
                const std::atomic<bool>* stop) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const int chunk = static_cast<int>(std::min<std::size_t>(data.size() - sent, 1 << 20));
#if defined(_WIN32)
        const int n = ::send(raw(s), data.data() + sent, chunk, 0);
#else
        const auto n = ::send(s, data.data() + sent, static_cast<std::size_t>(chunk), send_flags());
#endif
        if (n > 0) {
            sent += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && would_block(last_error())) {
            switch (wait_socket(s, true, stall_timeout, stop)) {
                case WaitResult::ready: continue;
                case WaitResult::timeout: return Status(ErrorCode::timeout, "peer stopped reading");
                case WaitResult::stopped: return Status(ErrorCode::cancelled, "server stopping");
                case WaitResult::error: return Status(ErrorCode::io_error, "send failed");
            }
        }
        return Status(ErrorCode::io_error, "peer closed the connection");
    }
    return Status::success();
}

bool peer_gone(native_socket s) {
    char buf[512];
    for (int i = 0; i < 64; ++i) {
        if (wait_once(s, false, 0) <= 0) {
            return false;
        }
        const long long n = recv_some(s, buf, sizeof(buf));
        if (n == 0 || n == -1) {
            return true;
        }
        if (n == -2) {
            return false;
        }
        // Bytes after the request (Connection: close, no pipelining): discard.
    }
    return false;
}

Result<Socket> connect_tcp(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout) {
    ensure_network();
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* list = nullptr;
    const std::string port_text = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_text.c_str(), &hints, &list) != 0 || list == nullptr) {
        return Status(ErrorCode::unavailable, "cannot resolve host " + host);
    }
    Status last(ErrorCode::unavailable, "cannot connect to " + host + ":" + port_text);
    for (addrinfo* ai = list; ai != nullptr; ai = ai->ai_next) {
        Socket sock(static_cast<native_socket>(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol)));
        if (!sock.valid()) {
            continue;
        }
        set_nonblocking(sock.get());
        no_sigpipe(sock.get());
        const int rc = ::connect(raw(sock.get()), ai->ai_addr, static_cast<int>(ai->ai_addrlen));
        if (rc != 0 && !would_block(last_error())) {
            continue;
        }
        if (rc != 0) {
            if (wait_socket(sock.get(), true, timeout, nullptr) != WaitResult::ready) {
                continue;
            }
            int err = 0;
            socklen_t len = sizeof(err);
            getsockopt(raw(sock.get()), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
            if (err != 0) {
                continue;
            }
        }
        freeaddrinfo(list);
        return sock;
    }
    freeaddrinfo(list);
    return last;
}

}  // namespace sonder::inference::server::detail
