// Backend-independent TLS helpers (compiled only with SONDER_WITH_TLS=ON).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>

#include "tls_stream.hpp"

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
#  include <poll.h>
#endif

namespace sonder::inference::net {

Result<std::string> normalize_sha256_fingerprint(std::string_view text) {
    std::string out;
    out.reserve(64);
    for (const char c : text) {
        if (c == ':' || c == ' ') {
            continue;
        }
        if (c >= '0' && c <= '9') {
            out.push_back(c);
        } else if (c >= 'a' && c <= 'f') {
            out.push_back(c);
        } else if (c >= 'A' && c <= 'F') {
            out.push_back(static_cast<char>(c - 'A' + 'a'));
        } else {
            return Status(ErrorCode::invalid_argument, "TLS pin: invalid character in SHA-256 fingerprint");
        }
    }
    if (out.size() != 64) {
        return Status(ErrorCode::invalid_argument, "TLS pin: SHA-256 fingerprint must have 32 bytes (64 hex digits)");
    }
    return out;
}

std::string to_hex(const unsigned char* data, std::size_t len) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (std::size_t i = 0; i < len; ++i) {
        out.push_back(kDigits[data[i] >> 4]);
        out.push_back(kDigits[data[i] & 0x0f]);
    }
    return out;
}

bool is_ip_literal(std::string_view host) {
    const std::string h(host);
    unsigned char buf[16];
    return inet_pton(AF_INET, h.c_str(), buf) == 1 || inet_pton(AF_INET6, h.c_str(), buf) == 1;
}

void warn_insecure_tls(const std::string& host, std::uint16_t port) {
    std::fprintf(stderr,
                 "[sonder-tls] WARNING: insecure_skip_verify is set: TLS certificate verification is DISABLED for "
                 "%s:%u. The connection is encrypted but NOT authenticated and can be intercepted. Use a CA bundle "
                 "or a certificate pin instead.\n",
                 host.c_str(), static_cast<unsigned>(port));
    std::fflush(stderr);
}

Status wait_native_socket(NativeSocket socket, bool for_write, Deadline deadline, const CancellationToken& cancel) {
    while (true) {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled");
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return Status(ErrorCode::timeout, for_write ? "TLS write timed out" : "TLS read timed out");
        }
        const auto slice =
            std::min<std::chrono::steady_clock::duration>(deadline - now, std::chrono::milliseconds(100));
        const int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(slice).count()) + 1;
#if defined(_WIN32)
        const auto s = static_cast<SOCKET>(socket);
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
        if (rc < 0) {
            return Status(ErrorCode::io_error, "select failed");
        }
#else
        pollfd pfd{};
        pfd.fd = static_cast<int>(socket);
        pfd.events = for_write ? POLLOUT : POLLIN;
        const int rc = ::poll(&pfd, 1, ms);
        if (rc < 0 && errno != EINTR) {
            return Status(ErrorCode::io_error, "poll failed");
        }
#endif
        if (rc > 0) {
            return Status::success();
        }
    }
}

Result<std::string> read_text_file(const std::string& path, std::string_view what) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status(ErrorCode::invalid_argument, "TLS: cannot read " + std::string(what) + " '" + path + "'");
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace sonder::inference::net
