// libFuzzer target: the internal HTTP client's wire helpers
// (src/net/http_client.cpp): parse_url, is_loopback_host, ChunkedDecoder and
// LineSplitter.
//
// Input layout: byte 0 = control (bits 0..1: stop the LineSplitter at that
// line, 0 = never; bits 2..7: seed for the chunk-split pattern); then one URL
// line (up to the first '\n'); the rest is a raw response body fed to the
// decoders. Seeds use printable control bytes ('0', '4', ...).
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary bytes.
//   * parse_url: errors are invalid_argument or unsupported with a non-empty
//     message; an accepted URL has scheme "http" or "https", a non-empty
//     host, a port in 1..65535 and a path that starts with '/' and has no
//     trailing '/' except the root.
//   * is_loopback_host agrees with an independent reference (localhost, ::1,
//     [::1], strict dotted-quad 127/8) on the URL line, the parsed host and
//     the whole input.
//   * Split invariance: feeding the body to ChunkedDecoder / LineSplitter in a
//     fuzz-chosen split gives the same decoded bytes, error state, done() and
//     stopped() as feeding it in one call.
//   * the decoded chunked body is never longer than the input.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fuzz_common.hpp"
#include "net/http_client.hpp"

namespace net = sonder::inference::net;
using sonder::inference::ErrorCode;

namespace {

// Reference oracle for "this name can only reach this machine".
bool ref_ipv4_loopback(std::string_view h) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const auto dot = h.find('.', start);
        if (dot == std::string_view::npos) {
            parts.push_back(h.substr(start));
            break;
        }
        parts.push_back(h.substr(start, dot - start));
        start = dot + 1;
    }
    if (parts.size() != 4) return false;
    int first = -1;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto p = parts[i];
        if (p.empty() || p.size() > 3) return false;
        if (p.size() > 1 && p[0] == '0') return false;
        int v = 0;
        for (const char c : p) {
            if (c < '0' || c > '9') return false;
            v = v * 10 + (c - '0');
        }
        if (v > 255) return false;
        if (i == 0) first = v;
    }
    return first == 127;
}

bool ref_loopback(std::string_view h) {
    if (h == "localhost" || h == "::1") return true;
    if (h.size() >= 3 && h.front() == '[' && h.back() == ']') return h.substr(1, h.size() - 2) == "::1";
    return ref_ipv4_loopback(h);
}

struct ChunkedOutcome {
    std::string out;
    bool errored = false;
    ErrorCode code = ErrorCode::ok;
    bool done = false;
    bool stopped = false;
};

// Splits `body` into 1..17 byte pieces driven by `seed`; seed == 0xFFFFFFFF
// means a single piece.
std::vector<std::string_view> split(std::string_view body, std::uint32_t seed) {
    std::vector<std::string_view> pieces;
    if (seed == 0xFFFFFFFFu) {
        pieces.push_back(body);
        return pieces;
    }
    std::uint32_t state = seed * 2654435761u + 1u;
    while (!body.empty()) {
        state = state * 1103515245u + 12345u;
        const std::size_t n = std::min<std::size_t>(body.size(), 1 + ((state >> 16) % 17u));
        pieces.push_back(body.substr(0, n));
        body.remove_prefix(n);
    }
    return pieces;
}

ChunkedOutcome run_chunked(std::string_view body, std::uint32_t seed, bool stop_on_first_data) {
    ChunkedOutcome o;
    net::ChunkedDecoder d;
    for (const auto piece : split(body, seed)) {
        const auto st = d.feed(piece, [&](std::string_view data) {
            o.out.append(data);
            return !stop_on_first_data;
        });
        if (!st.ok()) {
            o.errored = true;
            o.code = st.code();
            SONDER_FUZZ_CHECK(st.code() == ErrorCode::protocol_error);
            SONDER_FUZZ_CHECK(!st.message().empty());
            break;
        }
    }
    o.done = d.done();
    o.stopped = d.stopped();
    return o;
}

struct LineOutcome {
    std::vector<std::string> lines;
    bool feed_stopped = false;
};

LineOutcome run_lines(std::string_view body, std::uint32_t seed, std::size_t stop_at) {
    LineOutcome o;
    net::LineSplitter s;
    const auto on_line = [&](std::string_view line) {
        o.lines.emplace_back(line);
        return stop_at == 0 || o.lines.size() < stop_at;
    };
    for (const auto piece : split(body, seed)) {
        if (!s.feed(piece, on_line)) {
            o.feed_stopped = true;
            return o;
        }
    }
    if (!s.finish(on_line)) o.feed_stopped = true;
    return o;
}

void check_url(std::string_view line) {
    const auto r = net::parse_url(line);
    if (!r.ok()) {
        const auto code = r.status().code();
        SONDER_FUZZ_CHECK(code == ErrorCode::invalid_argument || code == ErrorCode::unsupported);
        SONDER_FUZZ_CHECK(!r.status().message().empty());
        return;
    }
    const net::Url& u = r.value();
    SONDER_FUZZ_CHECK(u.scheme == "http" || u.scheme == "https");
    SONDER_FUZZ_CHECK(!u.host.empty());
    SONDER_FUZZ_CHECK(u.port >= 1);  // std::uint16_t: 65535 is the upper bound
    SONDER_FUZZ_CHECK(!u.path.empty() && u.path.front() == '/');
    SONDER_FUZZ_CHECK(u.path.size() == 1 || u.path.back() != '/');
    SONDER_FUZZ_CHECK(net::is_loopback_host(u.host) == ref_loopback(u.host));
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) return 0;
    const std::size_t stop_at = data[0] & 3u;
    const std::uint32_t seed = data[0] >> 2;
    const std::string_view rest = sonder_fuzz::as_view(data + 1, size - 1);
    const auto nl = rest.find('\n');
    const std::string_view url_line = rest.substr(0, nl);
    const std::string_view body = nl == std::string_view::npos ? std::string_view{} : rest.substr(nl + 1);

    check_url(url_line);
    SONDER_FUZZ_CHECK(net::is_loopback_host(url_line) == ref_loopback(url_line));
    SONDER_FUZZ_CHECK(net::is_loopback_host(rest) == ref_loopback(rest));

    // Chunked decoding: one-shot (seed 0xFFFFFFFF) vs the fuzz-chosen split.
    for (const bool stop : {false, true}) {
        const ChunkedOutcome whole = run_chunked(body, 0xFFFFFFFFu, stop);
        const ChunkedOutcome parts = run_chunked(body, seed, stop);
        SONDER_FUZZ_CHECK(whole.errored == parts.errored);
        SONDER_FUZZ_CHECK(whole.code == parts.code);
        SONDER_FUZZ_CHECK(whole.done == parts.done);
        SONDER_FUZZ_CHECK(whole.stopped == parts.stopped);
        SONDER_FUZZ_CHECK(whole.out.size() <= body.size());
        SONDER_FUZZ_CHECK(parts.out.size() <= body.size());
        if (!stop) {
            SONDER_FUZZ_CHECK(whole.out == parts.out);
        } else {
            // Stops at the first delivered payload: at most one non-empty
            // piece per feed, and the split run's piece is a prefix of the
            // one-shot run's.
            SONDER_FUZZ_CHECK(whole.stopped == !whole.out.empty());
            SONDER_FUZZ_CHECK(whole.out.compare(0, parts.out.size(), parts.out) == 0);
        }
        if (whole.done) SONDER_FUZZ_CHECK(!whole.errored);
    }

    // Line splitting, on the raw body and on the whole input.
    for (const std::string_view text : {body, rest}) {
        const LineOutcome whole = run_lines(text, 0xFFFFFFFFu, stop_at);
        const LineOutcome parts = run_lines(text, seed, stop_at);
        SONDER_FUZZ_CHECK(whole.lines == parts.lines);
        SONDER_FUZZ_CHECK(whole.feed_stopped == parts.feed_stopped);
        std::size_t total = 0;
        for (const auto& l : whole.lines) {
            SONDER_FUZZ_CHECK(l.find('\n') == std::string::npos);
            total += l.size();
        }
        SONDER_FUZZ_CHECK(total <= text.size());
    }
    return 0;
}
