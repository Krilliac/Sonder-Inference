#include "http.hpp"

#include <algorithm>
#include <cctype>

namespace sonder::inference::server::detail {

namespace {

bool is_tchar(char c) noexcept {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
        return true;
    }
    switch (c) {
        case '!': case '#': case '$': case '%': case '&': case '\'': case '*': case '+':
        case '-': case '.': case '^': case '_': case '`': case '|': case '~':
            return true;
        default:
            return false;
    }
}

std::string lower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.remove_suffix(1);
    }
    return s;
}

ParseResult fail(int status, std::string message) {
    ParseResult r;
    r.state = ParseState::error;
    r.status = status;
    r.message = std::move(message);
    return r;
}

bool valid_field_value(std::string_view v) noexcept {
    for (const char c : v) {
        const auto u = static_cast<unsigned char>(c);
        if ((u < 0x20 && c != '\t') || u == 0x7F) {
            return false;
        }
    }
    return true;
}

bool valid_target_char(char c) noexcept {
    const auto u = static_cast<unsigned char>(c);
    return u > 0x20 && u < 0x7F;
}

}  // namespace

const std::string* RequestHead::header(std::string_view lowercase_name) const {
    for (const auto& [k, v] : headers) {
        if (k == lowercase_name) {
            return &v;
        }
    }
    return nullptr;
}

ParseResult parse_request_head(std::string_view buffer, RequestHead& out) {
    const std::size_t end = buffer.find("\r\n\r\n");
    if (end == std::string_view::npos) {
        if (buffer.size() > kMaxHeadBytes) {
            return fail(431, "request head exceeds 16 KiB");
        }
        // A request line alone longer than the limit also lands here.
        return {};
    }
    if (end + 4 > kMaxHeadBytes) {
        return fail(431, "request head exceeds 16 KiB");
    }
    out = RequestHead{};
    const std::string_view head = buffer.substr(0, end);

    std::size_t line_end = head.find("\r\n");
    const std::string_view request_line = head.substr(0, line_end);
    // METHOD SP request-target SP HTTP-version
    const std::size_t sp1 = request_line.find(' ');
    const std::size_t sp2 = sp1 == std::string_view::npos ? sp1 : request_line.find(' ', sp1 + 1);
    if (sp1 == std::string_view::npos || sp2 == std::string_view::npos ||
        request_line.find(' ', sp2 + 1) != std::string_view::npos) {
        return fail(400, "malformed request line");
    }
    const std::string_view method = request_line.substr(0, sp1);
    const std::string_view target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
    const std::string_view version = request_line.substr(sp2 + 1);
    if (method.empty() || !std::all_of(method.begin(), method.end(), is_tchar)) {
        return fail(400, "malformed request method");
    }
    if (target.empty() || target.front() != '/' || !std::all_of(target.begin(), target.end(), valid_target_char)) {
        return fail(400, "request target must be an absolute path");
    }
    if (version.size() != 8 || version.substr(0, 5) != "HTTP/" || version[6] != '.' ||
        std::isdigit(static_cast<unsigned char>(version[5])) == 0 ||
        std::isdigit(static_cast<unsigned char>(version[7])) == 0) {
        return fail(400, "malformed HTTP version");
    }
    if (version[5] != '1') {
        return fail(505, "only HTTP/1.x is supported");
    }
    out.method = std::string(method);
    out.target = std::string(target);
    out.minor_version = version[7] - '0';
    const std::size_t q = target.find('?');
    out.path = std::string(target.substr(0, q));
    if (q != std::string_view::npos) {
        out.query = std::string(target.substr(q + 1));
    }
    if (const std::size_t hash = out.path.find('#'); hash != std::string::npos) {
        out.path.resize(hash);
    }

    while (line_end != std::string_view::npos) {
        const std::size_t start = line_end + 2;
        line_end = head.find("\r\n", start);
        const std::string_view line =
            head.substr(start, line_end == std::string_view::npos ? std::string_view::npos : line_end - start);
        if (line.empty()) {
            return fail(400, "empty header line");
        }
        if (line.front() == ' ' || line.front() == '\t') {
            return fail(400, "obsolete header line folding is not accepted");
        }
        if (out.headers.size() >= kMaxHeaders) {
            return fail(431, "more than 64 request headers");
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0) {
            return fail(400, "malformed header line");
        }
        const std::string_view name = line.substr(0, colon);
        if (!std::all_of(name.begin(), name.end(), is_tchar)) {
            return fail(400, "malformed header name");
        }
        const std::string_view value = trim(line.substr(colon + 1));
        if (!valid_field_value(value)) {
            return fail(400, "invalid characters in a header value");
        }
        std::string key = lower(name);
        if (key == "content-length") {
            if (value.empty() || value.size() > 18 ||
                !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; })) {
                return fail(400, "invalid Content-Length");
            }
            std::uint64_t n = 0;
            for (const char c : value) {
                n = n * 10 + static_cast<std::uint64_t>(c - '0');
            }
            if (out.content_length && *out.content_length != n) {
                return fail(400, "conflicting Content-Length headers");
            }
            out.content_length = n;
        } else if (key == "transfer-encoding") {
            out.has_transfer_encoding = true;
        } else if (key == "host" && out.header("host") != nullptr) {
            // RFC 9112 section 3.2: more than one Host line is a 400, never
            // "first one wins" (the Host check would otherwise be bypassable).
            return fail(400, "more than one Host header");
        }
        out.headers.emplace_back(std::move(key), std::string(value));
    }
    if (out.has_transfer_encoding && out.content_length) {
        return fail(400, "Content-Length together with Transfer-Encoding is not accepted");
    }
    ParseResult ok;
    ok.state = ParseState::complete;
    ok.head_bytes = end + 4;
    return ok;
}

std::optional<std::string> url_decode(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '+') {
            out.push_back(' ');
        } else if (c == '%') {
            if (i + 2 >= text.size()) {
                return std::nullopt;
            }
            int v = 0;
            for (int k = 1; k <= 2; ++k) {
                const char h = text[i + static_cast<std::size_t>(k)];
                int d = -1;
                if (h >= '0' && h <= '9') d = h - '0';
                else if (h >= 'a' && h <= 'f') d = h - 'a' + 10;
                else if (h >= 'A' && h <= 'F') d = h - 'A' + 10;
                if (d < 0) {
                    return std::nullopt;
                }
                v = v * 16 + d;
            }
            out.push_back(static_cast<char>(v));
            i += 2;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::optional<std::string> query_param(std::string_view query, std::string_view key) {
    std::size_t pos = 0;
    while (pos <= query.size()) {
        const std::size_t amp = query.find('&', pos);
        const std::string_view item = query.substr(pos, amp == std::string_view::npos ? std::string_view::npos : amp - pos);
        const std::size_t eq = item.find('=');
        const auto name = url_decode(item.substr(0, eq));
        if (name && *name == key) {
            return url_decode(eq == std::string_view::npos ? std::string_view() : item.substr(eq + 1));
        }
        if (amp == std::string_view::npos) {
            break;
        }
        pos = amp + 1;
    }
    return std::nullopt;
}

const char* reason_phrase(int status) noexcept {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Content Too Large";
        case 417: return "Expectation Failed";
        case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
        default: return "Unknown";
    }
}

std::string format_head(int status, const std::vector<std::pair<std::string, std::string>>& headers) {
    std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason_phrase(status) + "\r\n";
    for (const auto& [k, v] : headers) {
        out += k;
        out += ": ";
        for (const char c : v) {
            if (c != '\r' && c != '\n') {
                out.push_back(c);
            }
        }
        out += "\r\n";
    }
    out += "\r\n";
    return out;
}

bool constant_time_equals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = static_cast<unsigned char>(diff | (static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i])));
    }
    return diff == 0;
}

bool is_correlation_value(std::string_view value) noexcept {
    if (value.empty() || value.size() > 128) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '.' || c == '_' || c == ':' || c == '-';
    });
}

bool is_loopback_host_header(std::string_view host) noexcept {
    std::string_view name = host;
    std::string_view port;
    if (!host.empty() && host.front() == '[') {
        const std::size_t close = host.find(']');
        if (close == std::string_view::npos) {
            return false;
        }
        name = host.substr(0, close + 1);
        const std::string_view rest = host.substr(close + 1);
        if (!rest.empty()) {
            if (rest.front() != ':') {
                return false;
            }
            port = rest.substr(1);
            if (port.empty()) {
                return false;
            }
        }
    } else if (const std::size_t colon = host.find(':'); colon != std::string_view::npos) {
        name = host.substr(0, colon);
        port = host.substr(colon + 1);
        if (port.empty()) {
            return false;
        }
    }
    if (port.size() > 5 || !std::all_of(port.begin(), port.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return false;
    }
    return name == "127.0.0.1" || name == "[::1]" || lower(name) == "localhost";
}

}  // namespace sonder::inference::server::detail
