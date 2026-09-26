// Internal TLS stream interface used by http_client.cpp when SONDER_HAS_TLS is
// defined. One implementation is compiled per build: tls_openssl.cpp or
// tls_schannel.cpp (selected by cmake/SonderTls.cmake). Shared helpers live in
// tls_common.cpp.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "tls.hpp"

namespace sonder::inference::net {

// Platform socket handle (int on POSIX, SOCKET on Windows) as an integer.
using NativeSocket = std::uintptr_t;
using Deadline = std::chrono::steady_clock::time_point;

// A client TLS session layered over a connected, non-blocking socket. The
// socket stays owned by the caller and must outlive the stream.
class TlsStream {
public:
    virtual ~TlsStream() = default;
    // Encrypts and sends all of data. Errors: timeout, cancelled, io_error.
    virtual Status write_all(std::string_view data, Deadline deadline, const CancellationToken& cancel) = 0;
    // Reads up to len decrypted bytes; 0 means the peer closed the stream.
    virtual Result<std::size_t> read_some(char* buf, std::size_t len, Deadline deadline,
                                          const CancellationToken& cancel) = 0;
};

// Performs the client handshake and certificate checks described in tls.hpp.
// Errors: invalid_argument (bad options / unreadable CA or pin file), timeout,
// cancelled, io_error, protocol_error (handshake, verification or pin failure).
Result<std::unique_ptr<TlsStream>> tls_connect(NativeSocket socket, const std::string& host, std::uint16_t port,
                                               const TlsOptions& options, Deadline deadline,
                                               const CancellationToken& cancel);

// Backend identifier: "openssl" or "schannel".
const char* tls_backend_name() noexcept;

// ---- shared helpers (tls_common.cpp) ----

// Normalizes a SHA-256 fingerprint to 64 lowercase hex characters.
Result<std::string> normalize_sha256_fingerprint(std::string_view text);
std::string to_hex(const unsigned char* data, std::size_t len);
bool is_ip_literal(std::string_view host);
// Prints the insecure_skip_verify warning to stderr.
void warn_insecure_tls(const std::string& host, std::uint16_t port);
// Waits for readability/writability in short slices, observing cancellation.
Status wait_native_socket(NativeSocket socket, bool for_write, Deadline deadline, const CancellationToken& cancel);
// Reads a whole (small) file; invalid_argument when it cannot be read.
Result<std::string> read_text_file(const std::string& path, std::string_view what);

}  // namespace sonder::inference::net
