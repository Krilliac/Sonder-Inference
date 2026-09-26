#include "http_client.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(SONDER_HAS_TLS)
#  include <memory>

#  include "tls_stream.hpp"
#endif

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
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
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

namespace sonder::inference::net {

// -------------------------------------------------------------- URL utils

Result<Url> parse_url(std::string_view text) {
    Url url;
    const auto scheme_end = text.find("://");
    if (scheme_end == std::string_view::npos) {
        return Status(ErrorCode::invalid_argument, "URL must include a scheme: " + std::string(text));
    }
    url.scheme = std::string(text.substr(0, scheme_end));
    std::transform(url.scheme.begin(), url.scheme.end(), url.scheme.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#if defined(SONDER_HAS_TLS)
    if (url.scheme == "https") {
        url.port = 443;
    } else
#endif
    if (url.scheme != "http") {
        return Status(ErrorCode::unsupported,
                      "only http:// URLs are supported (got " + url.scheme + "://)" +
                          (url.scheme == "https" ? "; https:// needs a build with SONDER_WITH_TLS=ON" : ""));
    }
    std::string_view rest = text.substr(scheme_end + 3);
    const auto path_start = rest.find('/');
    std::string_view authority = rest.substr(0, path_start);
    if (path_start != std::string_view::npos) {
        url.path = std::string(rest.substr(path_start));
        while (url.path.size() > 1 && url.path.back() == '/') {
            url.path.pop_back();
        }
    }
    if (authority.empty()) {
        return Status(ErrorCode::invalid_argument, "URL has no host");
    }
    std::string_view port_text;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) {
            return Status(ErrorCode::invalid_argument, "unterminated IPv6 literal");
        }
        url.host = std::string(authority.substr(1, close - 1));
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') {
                return Status(ErrorCode::invalid_argument, "invalid authority");
            }
            port_text = authority.substr(close + 2);
        }
    } else {
        const auto colon = authority.rfind(':');
        url.host = std::string(authority.substr(0, colon));
        if (colon != std::string_view::npos) {
            port_text = authority.substr(colon + 1);
        }
    }
    if (url.host.empty()) {
        return Status(ErrorCode::invalid_argument, "URL has no host");
    }
    if (!port_text.empty()) {
        unsigned long port = 0;
        for (const char c : port_text) {
            if (c < '0' || c > '9') {
                return Status(ErrorCode::invalid_argument, "invalid port");
            }
            port = port * 10 + static_cast<unsigned long>(c - '0');
            if (port > 65535) {
                return Status(ErrorCode::invalid_argument, "port out of range");
            }
        }
        if (port == 0) {
            return Status(ErrorCode::invalid_argument, "port out of range");
        }
        url.port = static_cast<std::uint16_t>(port);
    }
    return url;
}

bool is_loopback_host(std::string_view host) {
    if (host == "localhost" || host == "::1") {
        return true;
    }
    return host.rfind("127.", 0) == 0;
}

// --------------------------------------------------------- ChunkedDecoder

Status ChunkedDecoder::feed(std::string_view data, const std::function<bool(std::string_view)>& on_data) {
    std::size_t i = 0;
    while (i < data.size() && state_ != State::done && !stopped_) {
        switch (state_) {
            case State::size:
            case State::trailer:
            case State::data_crlf: {
                const char c = data[i++];
                if (c != '\n') {
                    if (line_.size() > 1024) {
                        return Status(ErrorCode::protocol_error, "chunk header line too long");
                    }
                    line_.push_back(c);
                    break;
                }
                if (!line_.empty() && line_.back() == '\r') {
                    line_.pop_back();
                }
                if (state_ == State::data_crlf) {
                    if (!line_.empty()) {
                        return Status(ErrorCode::protocol_error, "missing CRLF after chunk data");
                    }
                    state_ = State::size;
                } else if (state_ == State::trailer) {
                    if (line_.empty()) {
                        state_ = State::done;
                    }
                } else {
                    // Chunk size, optionally followed by ";extensions".
                    const std::string size_text = line_.substr(0, line_.find(';'));
                    if (size_text.empty()) {
                        return Status(ErrorCode::protocol_error, "empty chunk size");
                    }
                    std::uint64_t size = 0;
                    for (const char h : size_text) {
                        int v = -1;
                        if (h >= '0' && h <= '9') v = h - '0';
                        else if (h >= 'a' && h <= 'f') v = h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') v = h - 'A' + 10;
                        else if (h == ' ' || h == '\t') continue;
                        if (v < 0 || size > (1ull << 40)) {
                            return Status(ErrorCode::protocol_error, "invalid chunk size");
                        }
                        size = size * 16 + static_cast<std::uint64_t>(v);
                    }
                    remaining_ = size;
                    state_ = size == 0 ? State::trailer : State::data;
                }
                line_.clear();
                break;
            }
            case State::data: {
                const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(remaining_, data.size() - i));
                if (n > 0 && on_data && !on_data(data.substr(i, n))) {
                    stopped_ = true;
                }
                i += n;
                remaining_ -= n;
                if (remaining_ == 0) {
                    state_ = State::data_crlf;
                }
                break;
            }
            case State::done: break;
        }
    }
    return Status::success();
}

// ----------------------------------------------------------- LineSplitter

bool LineSplitter::feed(std::string_view data, const std::function<bool(std::string_view)>& on_line) {
    std::size_t start = 0;
    while (true) {
        const auto nl = data.find('\n', start);
        if (nl == std::string_view::npos) {
            buffer_.append(data.substr(start));
            return true;
        }
        buffer_.append(data.substr(start, nl - start));
        start = nl + 1;
        std::string line;
        line.swap(buffer_);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!on_line(line)) {
            return false;
        }
    }
}

bool LineSplitter::finish(const std::function<bool(std::string_view)>& on_line) {
    if (buffer_.empty()) {
        return true;
    }
    std::string line;
    line.swap(buffer_);
    return on_line(line);
}

// ---------------------------------------------------------------- sockets

namespace {

#if defined(_WIN32)
void ensure_winsock() {
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    });
}
void close_socket(socket_t s) { closesocket(s); }
int last_socket_error() { return WSAGetLastError(); }
bool would_block(int err) { return err == WSAEWOULDBLOCK || err == WSAETIMEDOUT || err == WSAEINPROGRESS; }
void set_nonblocking(socket_t s, bool on) {
    u_long mode = on ? 1 : 0;
    ioctlsocket(s, FIONBIO, &mode);
}
#else
void ensure_winsock() {}
void close_socket(socket_t s) { ::close(s); }
int last_socket_error() { return errno; }
bool would_block(int err) { return err == EAGAIN || err == EWOULDBLOCK || err == EINPROGRESS || err == EINTR; }
void set_nonblocking(socket_t s, bool on) {
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
}
#endif

class Socket {
public:
    Socket() = default;
    explicit Socket(socket_t s) : s_(s) {}
    ~Socket() {
        if (s_ != kInvalidSocket) {
            close_socket(s_);
        }
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& o) noexcept : s_(o.s_) { o.s_ = kInvalidSocket; }
    socket_t get() const { return s_; }
    bool valid() const { return s_ != kInvalidSocket; }

private:
    socket_t s_ = kInvalidSocket;
};

// Waits until the socket is readable/writable, in slices so cancellation is
// observed. Returns ok, timeout, or cancelled.
Status wait_socket(socket_t s, bool for_write, std::chrono::steady_clock::time_point deadline,
                   const CancellationToken& cancel) {
    while (true) {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled");
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return Status(ErrorCode::timeout, for_write ? "connect timed out" : "read timed out");
        }
        const auto slice = std::min<std::chrono::steady_clock::duration>(deadline - now, std::chrono::milliseconds(100));
        const int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(slice).count()) + 1;
#if defined(_WIN32)
        // select() rather than WSAPoll: older WSAPoll builds do not report
        // failed non-blocking connects. Failed connects land in exceptfds.
        fd_set rw;
        fd_set ex;
        FD_ZERO(&rw);
        FD_ZERO(&ex);
        FD_SET(s, &rw);
        FD_SET(s, &ex);
        timeval tv{};
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        const int rc = ::select(0, for_write ? nullptr : &rw, for_write ? &rw : nullptr, &ex, &tv);
#else
        pollfd pfd{};
        pfd.fd = s;
        pfd.events = for_write ? POLLOUT : POLLIN;
        const int rc = ::poll(&pfd, 1, ms);
#endif
        if (rc > 0) {
            return Status::success();
        }
        if (rc < 0 && !would_block(last_socket_error())) {
            return Status(ErrorCode::io_error, "poll failed");
        }
    }
}

Result<Socket> connect_to(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout,
                          const CancellationToken& cancel) {
    ensure_winsock();
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* list = nullptr;
    const std::string port_text = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_text.c_str(), &hints, &list) != 0 || list == nullptr) {
        return Status(ErrorCode::unavailable, "cannot resolve host " + host);
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    Status last(ErrorCode::unavailable, "no address for " + host);
    for (addrinfo* ai = list; ai != nullptr; ai = ai->ai_next) {
        Socket sock(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
        if (!sock.valid()) {
            continue;
        }
        set_nonblocking(sock.get(), true);
        const int rc = ::connect(sock.get(), ai->ai_addr, static_cast<int>(ai->ai_addrlen));
        if (rc != 0 && !would_block(last_socket_error())) {
            last = Status(ErrorCode::unavailable, "connect to " + host + ":" + port_text + " refused");
            continue;
        }
        if (rc != 0) {
            auto st = wait_socket(sock.get(), true, deadline, cancel);
            if (st.code() == ErrorCode::cancelled) {
                freeaddrinfo(list);
                return st;
            }
            if (!st.ok()) {
                last = Status(ErrorCode::unavailable, "connect to " + host + ":" + port_text + ": " + st.message());
                continue;
            }
            int err = 0;
            socklen_t len = sizeof(err);
            getsockopt(sock.get(), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
            if (err != 0) {
                last = Status(ErrorCode::unavailable, "connect to " + host + ":" + port_text + " refused");
                continue;
            }
        }
        int one = 1;
        setsockopt(sock.get(), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
        freeaddrinfo(list);
        return sock;
    }
    freeaddrinfo(list);
    return last;
}

Status send_all(socket_t s, std::string_view data, std::chrono::steady_clock::time_point deadline,
                const CancellationToken& cancel) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const auto chunk = static_cast<int>(std::min<std::size_t>(data.size() - sent, 1 << 20));
#if defined(_WIN32)
        const int n = ::send(s, data.data() + sent, chunk, 0);
#else
        const auto n = ::send(s, data.data() + sent, static_cast<std::size_t>(chunk), MSG_NOSIGNAL);
#endif
        if (n > 0) {
            sent += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && would_block(last_socket_error())) {
            auto st = wait_socket(s, true, deadline, cancel);
            if (!st.ok()) {
                return st;
            }
            continue;
        }
        return Status(ErrorCode::io_error, "send failed");
    }
    return Status::success();
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

Result<HttpResponseInfo> http_request(const HttpRequest& request, const std::function<bool(std::string_view)>& on_body,
                                      const CancellationToken& cancel) {
    const auto deadline = std::chrono::steady_clock::now() + request.total_timeout;
    auto connected = connect_to(request.host, request.port, request.connect_timeout, cancel);
    if (!connected.ok()) {
        return connected.status();
    }
    Socket sock = std::move(connected).value();
#if defined(SONDER_HAS_TLS)
    std::unique_ptr<TlsStream> tls;  // destroyed before sock (declared after it)
    if (request.use_tls) {
        auto session = tls_connect(static_cast<NativeSocket>(sock.get()), request.host, request.port, request.tls,
                                   deadline, cancel);
        if (!session.ok()) {
            return session.status();
        }
        tls = std::move(session).value();
    }
    auto send_bytes = [&](std::string_view data) {
        return tls ? tls->write_all(data, deadline, cancel) : send_all(sock.get(), data, deadline, cancel);
    };
#else
    if (request.use_tls) {
        return Status(ErrorCode::unsupported, "TLS requested but this build has no SONDER_WITH_TLS support");
    }
    auto send_bytes = [&](std::string_view data) { return send_all(sock.get(), data, deadline, cancel); };
#endif

    std::string host_header = request.host.find(':') != std::string::npos ? "[" + request.host + "]" : request.host;
    std::string head = request.method + " " + request.target + " HTTP/1.1\r\n";
    head += "Host: " + host_header + ":" + std::to_string(request.port) + "\r\n";
    head += "User-Agent: sonder-inference\r\nAccept: */*\r\nConnection: close\r\n";
    if (!request.body.empty() || request.method == "POST" || request.method == "PUT") {
        head += "Content-Type: " + request.content_type + "\r\n";
        head += "Content-Length: " + std::to_string(request.body.size()) + "\r\n";
    }
    head += "\r\n";
    if (auto st = send_bytes(head); !st.ok()) {
        return st;
    }
    if (!request.body.empty()) {
        if (auto st = send_bytes(request.body); !st.ok()) {
            return st;
        }
    }

    HttpResponseInfo info;
    std::string header_buf;
    bool headers_done = false;
    bool chunked = false;
    bool has_length = false;
    std::uint64_t content_length = 0;
    std::uint64_t body_received = 0;
    ChunkedDecoder decoder;
    char buf[16384];

    auto deliver = [&](std::string_view data) -> Status {
        if (data.empty()) {
            return Status::success();
        }
        if (chunked) {
            auto st = decoder.feed(data, [&](std::string_view payload) { return on_body ? on_body(payload) : true; });
            if (decoder.stopped()) {
                info.stopped_by_callback = true;
            }
            return st;
        }
        body_received += data.size();
        if (on_body && !on_body(data)) {
            info.stopped_by_callback = true;
        }
        return Status::success();
    };

    while (true) {
        long long n = 0;
        if (info.stopped_by_callback) {
            return info;
        }
        if (headers_done && ((chunked && decoder.done()) || (has_length && body_received >= content_length))) {
            return info;
        }
#if defined(SONDER_HAS_TLS)
        if (tls) {
            auto got = tls->read_some(buf, sizeof(buf), deadline, cancel);
            if (!got.ok()) {
                return got.status();
            }
            n = static_cast<long long>(got.value());
        } else
#endif
        {
            if (auto st = wait_socket(sock.get(), false, deadline, cancel); !st.ok()) {
                return st;
            }
            n = static_cast<long long>(::recv(sock.get(), buf, static_cast<int>(sizeof(buf)), 0));
        }
        if (n < 0) {
            if (would_block(last_socket_error())) {
                continue;
            }
            return Status(ErrorCode::io_error, "recv failed");
        }
        if (n == 0) {
            if (!headers_done) {
                return Status(ErrorCode::protocol_error, "connection closed before response headers");
            }
            if (chunked && !decoder.done()) {
                return Status(ErrorCode::protocol_error, "connection closed mid chunked body");
            }
            if (has_length && body_received < content_length) {
                return Status(ErrorCode::protocol_error, "connection closed before full body");
            }
            return info;
        }
        std::string_view data(buf, static_cast<std::size_t>(n));
        if (!headers_done) {
            header_buf.append(data);
            const auto end = header_buf.find("\r\n\r\n");
            if (end == std::string::npos) {
                if (header_buf.size() > 64 * 1024) {
                    return Status(ErrorCode::protocol_error, "response headers too large");
                }
                continue;
            }
            headers_done = true;
            const std::string head_text = header_buf.substr(0, end);
            const std::string rest = header_buf.substr(end + 4);
            // Status line: HTTP/1.1 200 OK
            const auto sp = head_text.find(' ');
            if (head_text.rfind("HTTP/", 0) != 0 || sp == std::string::npos) {
                return Status(ErrorCode::protocol_error, "malformed status line");
            }
            info.status = std::atoi(head_text.c_str() + sp + 1);
            std::size_t line_start = head_text.find("\r\n");
            while (line_start != std::string::npos) {
                line_start += 2;
                const auto line_end = head_text.find("\r\n", line_start);
                const std::string line = head_text.substr(line_start, line_end == std::string::npos ? std::string::npos
                                                                                                     : line_end - line_start);
                const auto colon = line.find(':');
                if (colon != std::string::npos) {
                    const std::string key = lower(line.substr(0, colon));
                    std::string value = line.substr(colon + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    if (key == "transfer-encoding" && lower(value).find("chunked") != std::string::npos) {
                        chunked = true;
                    } else if (key == "content-length") {
                        has_length = true;
                        content_length = std::strtoull(value.c_str(), nullptr, 10);
                    }
                }
                line_start = line_end;
            }
            if (chunked) {
                has_length = false;
            }
            if (request.on_status) {
                request.on_status(info.status);
            }
            if (auto st = deliver(rest); !st.ok()) {
                return st;
            }
            continue;
        }
        if (auto st = deliver(data); !st.ok()) {
            return st;
        }
    }
}

Result<HttpResponseInfo> http_request_buffered(const HttpRequest& request, std::string& body_out,
                                               const CancellationToken& cancel, std::size_t max_bytes) {
    body_out.clear();
    bool overflow = false;
    auto res = http_request(
        request,
        [&](std::string_view data) {
            if (body_out.size() + data.size() > max_bytes) {
                overflow = true;
                return false;
            }
            body_out.append(data);
            return true;
        },
        cancel);
    if (overflow) {
        return Status(ErrorCode::protocol_error, "response body exceeds limit");
    }
    return res;
}

}  // namespace sonder::inference::net
