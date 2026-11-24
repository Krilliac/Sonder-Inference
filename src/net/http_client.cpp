#include "http_client.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "../common/loopback.hpp"

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

namespace {

bool is_http_tchar(char ch) noexcept {
    const auto c = static_cast<unsigned char>(ch);
    return c != 0 && c < 0x80 && (std::isalnum(c) != 0 || std::strchr("!#$%&'*+-.^_`|~", ch) != nullptr);
}

bool is_valid_hostname_char(char ch) noexcept {
    const auto c = static_cast<unsigned char>(ch);
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-' ||
           c == '_';
}

bool is_valid_ipv6_literal(std::string_view host) noexcept {
    if (host.empty() || host.size() > 64) {
        return false;
    }
    bool saw_colon = false;
    for (const char ch : host) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == ':') {
            saw_colon = true;
            continue;
        }
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '.';
        if (!ok) {
            return false;
        }
    }
    return saw_colon;
}

bool is_valid_url_path_or_target(std::string_view s) noexcept {
    if (s.empty() || s.size() > 8192 || s.front() != '/') {
        return false;
    }
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        // Reject C0 controls (CR/LF/NUL/TAB), space, DEL, non-ASCII (incl. C1),
        // backslashes and fragments.
        if (c < 0x21 || c >= 0x7f || ch == '\\' || ch == '#') {
            return false;
        }
    }
    return true;
}

}  // namespace

Result<Url> parse_url(std::string_view text) {
    if (text.empty() || text.size() > 8192) {
        return Status(ErrorCode::invalid_argument, "URL must be 1-8192 bytes");
    }
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        // Reject all C0 controls (CR/LF/NUL/TAB), whitespace, DEL, non-ASCII
        // (including UTF-8 C1 controls and confusables), backslashes and fragments.
        if (c < 0x21 || c >= 0x7f) {
            return Status(ErrorCode::invalid_argument, "URL contains control, whitespace, or non-ASCII characters");
        }
        if (ch == '\\' || ch == '#') {
            return Status(ErrorCode::invalid_argument, "URL contains forbidden backslash or fragment");
        }
    }
    Url url;
    const auto scheme_end = text.find("://");
    if (scheme_end == std::string_view::npos || scheme_end == 0) {
        return Status(ErrorCode::invalid_argument, "URL must include a scheme");
    }
    const std::string_view raw_scheme = text.substr(0, scheme_end);
    for (std::size_t i = 0; i < raw_scheme.size(); ++i) {
        const auto c = static_cast<unsigned char>(raw_scheme[i]);
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        const bool extra = c == '+' || c == '-' || c == '.';
        if (i == 0 ? !alpha : !(alpha || digit || extra)) {
            return Status(ErrorCode::invalid_argument, "URL has invalid scheme");
        }
    }
    url.scheme = std::string(raw_scheme);
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
    const std::string_view rest = text.substr(scheme_end + 3);
    const auto path_start = rest.find('/');
    const std::string_view authority = rest.substr(0, path_start);
    if (path_start != std::string_view::npos) {
        const std::string_view raw_path = rest.substr(path_start);
        if (!is_valid_url_path_or_target(raw_path)) {
            return Status(ErrorCode::invalid_argument, "URL path is invalid");
        }
        url.path = std::string(raw_path);
        while (url.path.size() > 1 && url.path.back() == '/') {
            url.path.pop_back();
        }
    }
    if (authority.empty()) {
        return Status(ErrorCode::invalid_argument, "URL has no host");
    }
    if (authority.find('@') != std::string_view::npos) {
        return Status(ErrorCode::invalid_argument, "URL userinfo is not allowed");
    }
    if (authority.find('?') != std::string_view::npos) {
        return Status(ErrorCode::invalid_argument, "URL query before path is invalid");
    }
    std::string_view port_text;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) {
            return Status(ErrorCode::invalid_argument, "unterminated IPv6 literal");
        }
        url.host = std::string(authority.substr(1, close - 1));
        if (!is_valid_ipv6_literal(url.host)) {
            return Status(ErrorCode::invalid_argument, "invalid IPv6 literal");
        }
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
        if (url.host.empty() || url.host.size() > 253) {
            return Status(ErrorCode::invalid_argument, "URL has invalid host");
        }
        // Preserve compatibility for unbracketed loopback "::1:<port>" while
        // rejecting arbitrary colons or characters in unbracketed hosts.
        if (url.host != "::1") {
            for (const char ch : url.host) {
                if (!is_valid_hostname_char(ch)) {
                    return Status(ErrorCode::invalid_argument, "URL host contains an invalid character");
                }
            }
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
    // Only names that cannot resolve off this machine (common/loopback.hpp):
    // a DNS name such as "127.0.0.1.attacker.example" is remote.
    return sonder::inference::detail::is_loopback_literal(host);
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
                    // Chunk size, optionally followed by ";extensions" (with optional
                    // trailing BWS before ';').
                    std::string_view size_text(line_);
                    if (const auto semi = size_text.find(';'); semi != std::string_view::npos) {
                        size_text = size_text.substr(0, semi);
                    }
                    while (!size_text.empty() && (size_text.back() == ' ' || size_text.back() == '\t')) {
                        size_text.remove_suffix(1);
                    }
                    if (size_text.empty()) {
                        return Status(ErrorCode::protocol_error, "empty chunk size");
                    }
                    std::uint64_t size = 0;
                    for (const char h : size_text) {
                        int v = -1;
                        if (h >= '0' && h <= '9') v = h - '0';
                        else if (h >= 'a' && h <= 'f') v = h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') v = h - 'A' + 10;
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
        int sock_type = ai->ai_socktype;
#if !defined(_WIN32) && defined(SOCK_CLOEXEC)
        sock_type |= SOCK_CLOEXEC;
#endif
        Socket sock(::socket(ai->ai_family, sock_type, ai->ai_protocol));
        if (!sock.valid()) {
            continue;
        }
#if !defined(_WIN32)
        {
            const int fd_flags = ::fcntl(sock.get(), F_GETFD, 0);
            if (fd_flags >= 0) {
                ::fcntl(sock.get(), F_SETFD, fd_flags | FD_CLOEXEC);
            }
        }
#endif
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

Status validate_request_headers(const std::vector<std::pair<std::string, std::string>>& headers) {
    static constexpr std::string_view reserved[] = {"host",       "content-length",    "content-type", "connection",
                                                    "expect",     "transfer-encoding", "upgrade"};
    for (const auto& header : headers) {
        const std::string& name = header.first;
        const std::string& value = header.second;
        if (name.empty() || name.size() > 256) {
            return Status(ErrorCode::invalid_argument, "http header name must be 1-256 bytes");
        }
        for (const char ch : name) {
            const auto c = static_cast<unsigned char>(ch);
            const bool token = c != 0 && c < 0x80 && (std::isalnum(c) != 0 || std::strchr("!#$%&'*+-.^_`|~", ch) != nullptr);
            if (!token) {
                return Status(ErrorCode::invalid_argument, "http header name has an invalid character");
            }
        }
        const std::string key = lower(name);
        for (const auto r : reserved) {
            if (key == r) {
                return Status(ErrorCode::invalid_argument, "http header '" + name + "' is managed by the client");
            }
        }
        if (value.size() > 8192) {
            return Status(ErrorCode::invalid_argument, "http header '" + name + "' value is too long");
        }
        for (const char ch : value) {
            const auto c = static_cast<unsigned char>(ch);
            if ((c < 0x20 && c != '\t') || c == 0x7f) {
                return Status(ErrorCode::invalid_argument,
                              "http header '" + name + "' value contains a control character (CR/LF/NUL refused)");
            }
        }
    }
    return Status::success();
}

std::string redacted_header_value(std::string_view name, std::string_view value) {
    const std::string key = lower(std::string(name));
    if (key == "authorization" || key == "proxy-authorization" || key == "cookie" || key == "set-cookie" ||
        key.find("api-key") != std::string::npos || key.find("apikey") != std::string::npos ||
        key.find("token") != std::string::npos || key.find("secret") != std::string::npos) {
        return "[redacted]";
    }
    return std::string(value);
}

Status validate_http_request(const HttpRequest& request) {
    if (request.method.empty() || request.method.size() > 64) {
        return Status(ErrorCode::invalid_argument, "http method must be 1-64 bytes");
    }
    for (const char ch : request.method) {
        if (!is_http_tchar(ch)) {
            return Status(ErrorCode::invalid_argument, "http method contains an invalid character");
        }
    }
    if (!is_valid_url_path_or_target(request.target)) {
        return Status(ErrorCode::invalid_argument, "http request target must be a valid origin-form path starting with '/'");
    }
    if (request.host.empty() || request.host.size() > 255) {
        return Status(ErrorCode::invalid_argument, "http host must be 1-255 bytes");
    }
    if (request.host.find(':') != std::string::npos) {
        if (!is_valid_ipv6_literal(request.host)) {
            return Status(ErrorCode::invalid_argument, "http IPv6 host literal is invalid");
        }
    } else {
        for (const char ch : request.host) {
            if (!is_valid_hostname_char(ch)) {
                return Status(ErrorCode::invalid_argument, "http host contains an invalid character");
            }
        }
    }
    if (request.port == 0) {
        return Status(ErrorCode::invalid_argument, "http port must be 1-65535");
    }
    if (request.content_type.size() > 1024) {
        return Status(ErrorCode::invalid_argument, "http Content-Type is too long");
    }
    for (const char ch : request.content_type) {
        const auto c = static_cast<unsigned char>(ch);
        if ((c < 0x20 && c != '\t') || c >= 0x7f) {
            return Status(ErrorCode::invalid_argument, "http Content-Type contains a control or non-ASCII character");
        }
    }
    return validate_request_headers(request.headers);
}

Result<ParsedResponseHead> parse_response_head(std::string_view head_text) {
    if (head_text.empty()) {
        return Status(ErrorCode::protocol_error, "empty response head");
    }
    const auto reject_bare_ctl = [](std::string_view line) -> bool {
        for (const char ch : line) {
            if (ch == '\r' || ch == '\n' || ch == '\0') {
                return true;
            }
        }
        return false;
    };

    const auto first_crlf = head_text.find("\r\n");
    const std::string_view status_line =
        first_crlf == std::string_view::npos ? head_text : head_text.substr(0, first_crlf);
    if (reject_bare_ctl(status_line)) {
        return Status(ErrorCode::protocol_error, "malformed status line: bare CR/LF/NUL");
    }
    // RFC 9112 status-line: "HTTP/1." ("0" / "1") SP 3DIGIT [ SP reason-phrase ]
    if (status_line.size() < 12 ||
        (status_line.compare(0, 9, "HTTP/1.1 ") != 0 && status_line.compare(0, 9, "HTTP/1.0 ") != 0)) {
        return Status(ErrorCode::protocol_error, "malformed status line");
    }
    int status = 0;
    for (std::size_t i = 9; i < 12; ++i) {
        const char ch = status_line[i];
        if (ch < '0' || ch > '9') {
            return Status(ErrorCode::protocol_error, "malformed status code");
        }
        status = status * 10 + (ch - '0');
    }
    if (status < 100 || status > 599) {
        return Status(ErrorCode::protocol_error, "status code out of range");
    }
    if (status_line.size() > 12) {
        if (status_line[12] != ' ') {
            return Status(ErrorCode::protocol_error, "malformed status line delimiter");
        }
        for (std::size_t i = 13; i < status_line.size(); ++i) {
            const auto c = static_cast<unsigned char>(status_line[i]);
            if ((c < 0x20 && c != '\t') || c == 0x7f) {
                return Status(ErrorCode::protocol_error, "malformed status reason phrase");
            }
        }
    }

    ParsedResponseHead out;
    out.status = status;
    bool has_transfer_encoding = false;
    std::size_t header_count = 0;
    constexpr std::size_t kMaxResponseHeaders = 256;

    std::size_t pos = first_crlf;
    while (pos != std::string_view::npos) {
        pos += 2;
        const auto next_crlf = head_text.find("\r\n", pos);
        const std::string_view line =
            next_crlf == std::string_view::npos ? head_text.substr(pos) : head_text.substr(pos, next_crlf - pos);
        pos = next_crlf;

        if (++header_count > kMaxResponseHeaders) {
            return Status(ErrorCode::protocol_error, "too many response headers");
        }
        if (line.empty()) {
            return Status(ErrorCode::protocol_error, "unexpected empty response header line");
        }
        if (reject_bare_ctl(line)) {
            return Status(ErrorCode::protocol_error, "response header contains bare CR/LF/NUL");
        }
        if (line.front() == ' ' || line.front() == '\t') {
            return Status(ErrorCode::protocol_error, "obsolete response header line folding refused");
        }
        const auto colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0) {
            return Status(ErrorCode::protocol_error, "malformed response header: missing colon or empty name");
        }
        const std::string_view raw_name = line.substr(0, colon);
        for (const char ch : raw_name) {
            if (!is_http_tchar(ch)) {
                return Status(ErrorCode::protocol_error, "response header name contains invalid character");
            }
        }
        std::string_view value = line.substr(colon + 1);
        for (const char ch : value) {
            const auto c = static_cast<unsigned char>(ch);
            if ((c < 0x20 && c != '\t') || c == 0x7f) {
                return Status(ErrorCode::protocol_error, "response header value contains control character");
            }
        }
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.remove_prefix(1);
        }
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
            value.remove_suffix(1);
        }

        const std::string key = lower(std::string(raw_name));
        if (key == "transfer-encoding") {
            if (has_transfer_encoding) {
                return Status(ErrorCode::protocol_error, "duplicate Transfer-Encoding response header");
            }
            has_transfer_encoding = true;
            if (lower(std::string(value)) != "chunked") {
                return Status(ErrorCode::protocol_error, "unsupported Transfer-Encoding response header");
            }
            out.chunked = true;
        } else if (key == "content-length") {
            if (out.has_length) {
                return Status(ErrorCode::protocol_error, "duplicate Content-Length response header");
            }
            if (value.empty()) {
                return Status(ErrorCode::protocol_error, "empty Content-Length response header");
            }
            for (const char ch : value) {
                if (ch < '0' || ch > '9') {
                    return Status(ErrorCode::protocol_error, "non-decimal Content-Length response header");
                }
            }
            std::uint64_t parsed_len = 0;
            const auto fc = std::from_chars(value.data(), value.data() + value.size(), parsed_len, 10);
            if (fc.ec != std::errc{} || fc.ptr != value.data() + value.size()) {
                return Status(ErrorCode::protocol_error, "Content-Length overflow or invalid format");
            }
            out.has_length = true;
            out.content_length = parsed_len;
        }
    }

    if (has_transfer_encoding && out.has_length) {
        return Status(ErrorCode::protocol_error, "conflicting Transfer-Encoding and Content-Length response headers");
    }
    return out;
}

Result<HttpResponseInfo> http_request(const HttpRequest& request, const std::function<bool(std::string_view)>& on_body,
                                      const CancellationToken& cancel) {
    if (auto st = validate_http_request(request); !st.ok()) {
        return st;
    }
    if ((request.first_progress_timeout.count() > 0 || request.progress_stall_timeout.count() > 0) &&
        !request.progress_counter) {
        return Status(ErrorCode::invalid_argument, "progress timeouts need progress_counter");
    }
    // "Unlimited" is a distant finite deadline so the deadline arithmetic in
    // the socket and TLS helpers can never overflow.
    const auto deadline = std::chrono::steady_clock::now() +
                          (request.unlimited_total ? std::chrono::milliseconds(std::chrono::hours(24 * 365))
                                                   : request.total_timeout);
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
    for (const auto& header : request.headers) {
        head += header.first + ": " + header.second + "\r\n";
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

    // Per-read deadline: the earliest of the total cap, the idle bound (after
    // the head), the first-progress bound and the progress-stall bound.
    enum class Bound { total, idle, first_progress, stall };
    const auto sent_at = std::chrono::steady_clock::now();
    auto last_activity = sent_at;
    auto last_progress_at = sent_at;
    std::uint64_t progress_seen = 0;
    const auto read_deadline = [&](Bound& which) {
        auto d = deadline;
        which = Bound::total;
        const auto consider = [&](std::chrono::steady_clock::time_point candidate, Bound kind) {
            if (candidate < d) {
                d = candidate;
                which = kind;
            }
        };
        if (headers_done && request.idle_timeout.count() > 0) {
            consider(last_activity + request.idle_timeout, Bound::idle);
        }
        if (request.first_progress_timeout.count() > 0 && progress_seen == 0) {
            consider(sent_at + request.first_progress_timeout, Bound::first_progress);
        }
        if (request.progress_stall_timeout.count() > 0) {
            consider(last_progress_at + request.progress_stall_timeout, Bound::stall);
        }
        return d;
    };
    const auto timeout_status = [&](Bound which, Status original) {
        switch (which) {
        case Bound::idle:
            return Status(ErrorCode::protocol_error,
                          "idle timeout: no bytes received for " + std::to_string(request.idle_timeout.count()) + " ms");
        case Bound::first_progress:
            return Status(ErrorCode::protocol_error, "first-content timeout: no content within " +
                                                  std::to_string(request.first_progress_timeout.count()) + " ms");
        case Bound::stall:
            return Status(ErrorCode::protocol_error, "stall timeout: no content progress for " +
                                                  std::to_string(request.progress_stall_timeout.count()) + " ms");
        case Bound::total:
            break;
        }
        return original;
    };
    const auto note_progress = [&] {
        if (!request.progress_counter) {
            return;
        }
        const auto value = request.progress_counter();
        if (value != progress_seen) {
            progress_seen = value;
            last_progress_at = std::chrono::steady_clock::now();
        }
    };

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
        if (has_length) {
            if (body_received >= content_length) {
                return Status::success();
            }
            const std::uint64_t remaining = content_length - body_received;
            if (static_cast<std::uint64_t>(data.size()) > remaining) {
                data = data.substr(0, static_cast<std::size_t>(remaining));
            }
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
        Bound bound = Bound::total;
        const auto read_by = read_deadline(bound);
        const bool stall_warns = bound == Bound::stall && static_cast<bool>(request.on_progress_stall);
        if (stall_warns && std::chrono::steady_clock::now() >= read_by) {
            // Warn-only stall policy: report, re-arm, keep reading.
            const auto now = std::chrono::steady_clock::now();
            request.on_progress_stall(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress_at));
            last_progress_at = now;
            continue;
        }
#if defined(SONDER_HAS_TLS)
        if (tls) {
            auto got = tls->read_some(buf, sizeof(buf), read_by, cancel);
            if (!got.ok()) {
                if (got.status().code() == ErrorCode::timeout) {
                    if (stall_warns) {
                        continue;
                    }
                    return timeout_status(bound, got.status());
                }
                return got.status();
            }
            n = static_cast<long long>(got.value());
        } else
#endif
        {
            if (auto st = wait_socket(sock.get(), false, read_by, cancel); !st.ok()) {
                if (st.code() == ErrorCode::timeout) {
                    if (stall_warns) {
                        continue;
                    }
                    return timeout_status(bound, st);
                }
                return st;
            }
            n = static_cast<long long>(::recv(sock.get(), buf, static_cast<int>(sizeof(buf)), 0));
        }
        if (n > 0) {
            last_activity = std::chrono::steady_clock::now();
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
            auto parsed_head = parse_response_head(head_text);
            if (!parsed_head.ok()) {
                return parsed_head.status();
            }
            info.status = parsed_head->status;
            chunked = parsed_head->chunked;
            has_length = parsed_head->has_length;
            content_length = parsed_head->content_length;
            if (request.on_status) {
                request.on_status(info.status);
            }
            if (auto st = deliver(rest); !st.ok()) {
                return st;
            }
            note_progress();
            continue;
        }
        if (auto st = deliver(data); !st.ok()) {
            return st;
        }
        note_progress();
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
