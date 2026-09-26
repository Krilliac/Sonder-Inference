// Internal: HTTP/1.1 request-head parsing and response formatting for the
// server module. Pure functions (no I/O) so they are unit-tested and fuzzed
// (fuzz/fuzz_http_request.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference::server::detail {

inline constexpr std::size_t kMaxHeadBytes = 16 * 1024;
inline constexpr std::size_t kMaxHeaders = 64;

struct RequestHead {
    std::string method;
    std::string target;   // as sent (origin-form)
    std::string path;     // target without the query
    std::string query;    // after '?', undecoded; empty when absent
    int minor_version = 1;
    // Header names lowercased; values with surrounding whitespace trimmed.
    std::vector<std::pair<std::string, std::string>> headers;
    // Parsed Content-Length (absent when the header is missing).
    std::optional<std::uint64_t> content_length;
    bool has_transfer_encoding = false;

    // First value of a header (name must be lowercase), or null.
    [[nodiscard]] const std::string* header(std::string_view lowercase_name) const;
};

enum class ParseState { incomplete, complete, error };

struct ParseResult {
    ParseState state = ParseState::incomplete;
    // On complete: bytes consumed by the head (including the blank line).
    std::size_t head_bytes = 0;
    // On error: HTTP status to answer with (400, 431, 501, 505) and a reason.
    int status = 0;
    std::string message;
};

// Parses a request head from the start of `buffer`. Lines end with CRLF.
// Enforces kMaxHeadBytes (431) and kMaxHeaders (431); rejects obsolete line
// folding, whitespace before the colon, invalid token characters, non
// origin-form targets, conflicting Content-Length values and a
// Content-Length combined with Transfer-Encoding and more than one Host
// header (400); non-1.x versions
// (505).
ParseResult parse_request_head(std::string_view buffer, RequestHead& out);

// Percent-decoding for query components ('+' becomes a space). Returns
// nullopt on a malformed escape.
std::optional<std::string> url_decode(std::string_view text);

// First value of `key` in a query string (decoded), or nullopt. A present key
// with a malformed value yields nullopt as well.
std::optional<std::string> query_param(std::string_view query, std::string_view key);

// Reason phrase for the status codes the server emits.
const char* reason_phrase(int status) noexcept;

// Serializes a status line plus headers (and the blank line). Header values
// must not contain CR or LF; offending characters are dropped.
std::string format_head(int status, const std::vector<std::pair<std::string, std::string>>& headers);

// Constant-time comparison of equal-length prefixes; false when the lengths
// differ (the length itself is not secret for bearer tokens).
bool constant_time_equals(std::string_view a, std::string_view b) noexcept;

// [A-Za-z0-9._:-]{1,128} (correlation header values).
bool is_correlation_value(std::string_view value) noexcept;

// Host header check for loopback binds: 127.0.0.1, localhost or [::1], with
// an optional numeric port. Case-insensitive for "localhost".
bool is_loopback_host_header(std::string_view host) noexcept;

}  // namespace sonder::inference::server::detail
