// Minimal blocking HTTP/1.1 client (internal). Plain HTTP, plus https:// when
// built with SONDER_WITH_TLS=ON (see tls.hpp); intended for control-plane
// traffic such as an Ollama server. Supports
// Content-Length, chunked transfer encoding, and read-until-close bodies,
// streaming the decoded body to a callback while polling a cancellation token.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "tls.hpp"

namespace sonder::inference::net {

struct Url {
    std::string scheme;
    std::string host;
    std::uint16_t port = 80;  // 443 for https:// without an explicit port
    std::string path = "/";  // base path, no trailing slash except root
};

// Accepts http:// and, only when SONDER_HAS_TLS is defined, https://; other
// schemes (and https:// in non-TLS builds) return ErrorCode::unsupported.
Result<Url> parse_url(std::string_view text);
bool is_loopback_host(std::string_view host);

// Incremental decoder for Transfer-Encoding: chunked.
class ChunkedDecoder {
public:
    // Feeds raw bytes; decoded payload goes to on_data (return false to stop).
    // Returns protocol_error on malformed input.
    Status feed(std::string_view data, const std::function<bool(std::string_view)>& on_data);
    [[nodiscard]] bool done() const noexcept { return state_ == State::done; }
    [[nodiscard]] bool stopped() const noexcept { return stopped_; }

private:
    enum class State { size, data, data_crlf, trailer, done };
    State state_ = State::size;
    std::string line_;
    std::uint64_t remaining_ = 0;
    bool stopped_ = false;
};

// Splits a byte stream into '\n'-terminated lines ('\r' stripped).
class LineSplitter {
public:
    // on_line returning false stops processing; feed then returns false.
    bool feed(std::string_view data, const std::function<bool(std::string_view)>& on_line);
    // Delivers a trailing unterminated line, if any.
    bool finish(const std::function<bool(std::string_view)>& on_line);

private:
    std::string buffer_;
};

struct HttpRequest {
    std::string method = "GET";
    std::string host;
    std::uint16_t port = 80;
    std::string target = "/";
    std::string body;
    std::string content_type = "application/json";
    std::chrono::milliseconds connect_timeout{3000};
    std::chrono::milliseconds total_timeout{300000};
    // TLS (https). Requires SONDER_HAS_TLS; otherwise the request fails with
    // ErrorCode::unsupported before anything is sent.
    bool use_tls = false;
    TlsOptions tls;
    // Invoked once with the HTTP status code, before any body bytes.
    std::function<void(int)> on_status;
};

struct HttpResponseInfo {
    int status = 0;
    bool stopped_by_callback = false;
};

// Streams the decoded response body to on_body (return false to stop early).
// Errors: unavailable (connect failure), timeout, cancelled, io_error,
// protocol_error.
Result<HttpResponseInfo> http_request(const HttpRequest& request,
                                      const std::function<bool(std::string_view)>& on_body,
                                      const CancellationToken& cancel = {});

// Convenience: buffers the whole body (bounded by max_bytes).
Result<HttpResponseInfo> http_request_buffered(const HttpRequest& request, std::string& body_out,
                                               const CancellationToken& cancel = {},
                                               std::size_t max_bytes = 64u * 1024u * 1024u);

}  // namespace sonder::inference::net
