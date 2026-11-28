// Regression tests for outbound HTTP client hardening (M-13 / B2, L-06 / B5, L-09).
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include "net/http_client.hpp"

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
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace {

namespace net = sonder::inference::net;
using sonder::inference::ErrorCode;

// Tiny one-shot raw TCP server on 127.0.0.1:0 that accepts one connection,
// drains the request head, sends raw_response verbatim, and closes.
class RawLoopbackResponder {
public:
    explicit RawLoopbackResponder(std::string raw_response) : response_(std::move(raw_response)) {
#if defined(_WIN32)
        WSADATA wsa{};
        WSAStartup(MAKEWORD(2, 2), &wsa);
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        REQUIRE(listen_fd_ != INVALID_SOCKET);
#else
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(listen_fd_ >= 0);
#endif
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        REQUIRE(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        REQUIRE(::listen(listen_fd_, 4) == 0);
        socklen_t len = sizeof(addr);
        REQUIRE(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
        port_ = ntohs(addr.sin_port);
        worker_ = std::thread([this] { serve(); });
    }

    ~RawLoopbackResponder() {
        if (worker_.joinable()) {
            worker_.join();
        }
#if defined(_WIN32)
        if (listen_fd_ != INVALID_SOCKET) {
            ::closesocket(listen_fd_);
        }
#else
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
        }
#endif
    }

    std::uint16_t port() const noexcept { return port_; }

private:
    void serve() {
        auto client = ::accept(listen_fd_, nullptr, nullptr);
#if defined(_WIN32)
        if (client == INVALID_SOCKET) return;
#else
        if (client < 0) return;
#endif
        char buf[4096];
        std::string req;
        while (req.find("\r\n\r\n") == std::string::npos) {
            const auto n = ::recv(client, buf, static_cast<int>(sizeof(buf)), 0);
            if (n <= 0) break;
            req.append(buf, static_cast<std::size_t>(n));
        }
        std::size_t off = 0;
        while (off < response_.size()) {
            const auto n = ::send(client, response_.data() + off, static_cast<int>(response_.size() - off), 0);
            if (n <= 0) break;
            off += static_cast<std::size_t>(n);
        }
#if defined(_WIN32)
        ::shutdown(client, SD_SEND);
        ::closesocket(client);
#else
        ::shutdown(client, SHUT_WR);
        ::close(client);
#endif
    }

#if defined(_WIN32)
    SOCKET listen_fd_ = INVALID_SOCKET;
#else
    int listen_fd_ = -1;
#endif
    std::uint16_t port_ = 0;
    std::string response_;
    std::thread worker_;
};

}  // namespace

TEST_CASE("http_client: parse_url rejects control characters, whitespace, backslash, fragment and userinfo") {
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/a\r\nX-Injected: 1").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/a\nX-Injected: 1").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/a\rX").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/has space").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/has\ttab").ok());
    CHECK_FALSE(net::parse_url(std::string("http://127.0.0.1:8080/nul\0path", 29)).ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/c1\xc2\x85path").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/back\\slash").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1:8080/path#fragment").ok());
    CHECK_FALSE(net::parse_url("http://user:pass@127.0.0.1:8080/api").ok());
    CHECK_FALSE(net::parse_url("http://127.0.0.1?@evil.example/api").ok());
    CHECK_FALSE(net::parse_url("http://[fe80::1%eth0]:8080/api").ok());

    const auto ok_v4 = net::parse_url("http://127.0.0.1:8080/v1/api/");
    REQUIRE(ok_v4.ok());
    CHECK(ok_v4->host == "127.0.0.1");
    CHECK(ok_v4->port == 8080);
    CHECK(ok_v4->path == "/v1/api");

    const auto ok_v6 = net::parse_url("http://[::1]:11434/");
    REQUIRE(ok_v6.ok());
    CHECK(ok_v6->host == "::1");
    CHECK(ok_v6->port == 11434);
    CHECK(ok_v6->path == "/");
}

TEST_CASE("http_client: validate_http_request rejects injected method, target, host and content_type before connect") {
    net::HttpRequest base;
    base.method = "POST";
    base.host = "127.0.0.1";
    base.port = 1;  // never connected on validation errors
    base.target = "/api/generate";
    base.content_type = "application/json";
    REQUIRE(net::validate_http_request(base).ok());

    {
        auto r = base;
        r.method = "POST /evil HTTP/1.1\r\nX: y";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.target = "http://evil.example/absolute";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.target = "/ok HTTP/1.1\r\nHost: evil.example\r\n\r\nGET /";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.target = "/ok\\bad";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.target = "/ok#frag";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.host = "127.0.0.1\r\nX-Injected: 1";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.host = "[::1]";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.port = 0;
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
    {
        auto r = base;
        r.content_type = "application/json\r\nX-Injected: 1";
        CHECK(net::http_request(r, {}).status().code() == ErrorCode::invalid_argument);
    }
}

TEST_CASE("http_client: parse_response_head strictly rejects malformed and smuggling headers") {
    {
        auto ok = net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: 42\r\nContent-Type: application/json");
        REQUIRE(ok.ok());
        CHECK(ok->status == 200);
        CHECK(ok->has_length);
        CHECK(ok->content_length == 42u);
        CHECK_FALSE(ok->chunked);
    }
    {
        auto ok = net::parse_response_head("HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked");
        REQUIRE(ok.ok());
        CHECK(ok->chunked);
        CHECK_FALSE(ok->has_length);
    }

    // Colon-less header line
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nNotAHeader").status().code() == ErrorCode::protocol_error);
    // Obsolete line folding
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nX-Test: a\r\n folded").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nX-Test: a\r\n\tfolded").status().code() == ErrorCode::protocol_error);
    // Space before colon
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length : 5").status().code() == ErrorCode::protocol_error);
    // Duplicate Content-Length (same or conflicting)
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Length: 5").status().code() ==
          ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Length: 10").status().code() ==
          ErrorCode::protocol_error);
    // Permissive strtoull bypasses: leading sign, hex, trailing junk, overflow
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: +5").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: -1").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: 5abc").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nContent-Length: 999999999999999999999999999").status().code() ==
          ErrorCode::protocol_error);
    // Substring or chained Transfer-Encoding
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nTransfer-Encoding: not-chunked").status().code() ==
          ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked").status().code() ==
          ErrorCode::protocol_error);
    // Conflicting TE + CL
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 5").status().code() ==
          ErrorCode::protocol_error);
    // Bad status line / non-numeric status
    CHECK(net::parse_response_head("HTTP/1.1 abc OK").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/2 200 OK").status().code() == ErrorCode::protocol_error);
    CHECK(net::parse_response_head("HTTP/1.1 200 OK\nContent-Length: 0").status().code() == ErrorCode::protocol_error);
}

TEST_CASE("http_client: never delivers trailing bytes past Content-Length") {
    RawLoopbackResponder srv("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhelloEXTRA_SMUGGLED_BYTES");
    net::HttpRequest req;
    req.method = "GET";
    req.host = "127.0.0.1";
    req.port = srv.port();
    req.target = "/probe";

    std::string body;
    auto res = net::http_request(req, [&](std::string_view chunk) {
        body.append(chunk);
        return true;
    });
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    CHECK(res->status == 200);
    CHECK(body == "hello");
}

TEST_CASE("http_client: rejects TE plus CL response over live loopback socket") {
    RawLoopbackResponder srv("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n0\r\n\r\n");
    net::HttpRequest req;
    req.method = "GET";
    req.host = "127.0.0.1";
    req.port = srv.port();
    req.target = "/probe";

    std::string body;
    auto res = net::http_request(req, [&](std::string_view chunk) {
        body.append(chunk);
        return true;
    });
    CHECK(res.status().code() == ErrorCode::protocol_error);
    CHECK(body.empty());
}

TEST_CASE("http_client: max_response_bytes caps every received byte, framing and trailers included") {
    // One body byte, then ~200 KB of chunked trailer lines: payload-only
    // counting would see 1 byte; the wire cap must stop it.
    std::string raw = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\na\r\n0\r\n";
    for (int i = 0; i < 200; ++i) {
        raw += "X-Trailer-" + std::to_string(i) + ": " + std::string(1000, 't') + "\r\n";
    }
    raw += "\r\n";
    {
        RawLoopbackResponder srv(raw);
        net::HttpRequest req;
        req.host = "127.0.0.1";
        req.port = srv.port();
        req.target = "/probe";
        req.max_response_bytes = 64 * 1024;
        std::string body;
        auto res = net::http_request(req, [&](std::string_view chunk) {
            body.append(chunk);
            return true;
        });
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::protocol_error);
        CHECK(res.status().message().find("response exceeds") != std::string::npos);
        CHECK(body.size() <= 1u);
    }
    {
        // Zero keeps the historical behaviour (no cap): the same response completes.
        RawLoopbackResponder srv(raw);
        net::HttpRequest req;
        req.host = "127.0.0.1";
        req.port = srv.port();
        req.target = "/probe";
        std::string body;
        auto res = net::http_request(req, [&](std::string_view chunk) {
            body.append(chunk);
            return true;
        });
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        CHECK(body == "a");
    }
    {
        // A response within the cap is unaffected; the head counts toward it.
        const std::string small = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        RawLoopbackResponder srv(small);
        net::HttpRequest req;
        req.host = "127.0.0.1";
        req.port = srv.port();
        req.target = "/probe";
        req.max_response_bytes = small.size();
        std::string body;
        auto res = net::http_request(req, [&](std::string_view chunk) {
            body.append(chunk);
            return true;
        });
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        CHECK(body == "hello");
    }
    {
        RawLoopbackResponder srv("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello");
        net::HttpRequest req;
        req.host = "127.0.0.1";
        req.port = srv.port();
        req.target = "/probe";
        req.max_response_bytes = 10;  // smaller than the head alone
        auto res = net::http_request(req, {});
        CHECK(res.status().code() == ErrorCode::protocol_error);
    }
}
