// Optional TLS for the internal HTTP client (src/net/http_client.*).
//
// TLS is compiled in only when the build is configured with
// -DSONDER_WITH_TLS=ON (cmake/SonderTls.cmake), which defines SONDER_HAS_TLS
// for sonder_inference and its consumers. Without it this header still
// compiles (so request structs keep one layout), but https:// URLs are
// rejected with ErrorCode::unsupported. See docs/integration/tls.md.
//
// Verification modes (evaluated per connection):
//   1. insecure_skip_verify = true: no chain or host-name verification. A loud
//      warning is written to stderr on every handshake. A configured pin is
//      still enforced.
//   2. a pin is set and ca_bundle_path is empty: "pin-only" mode for
//      self-signed nodes. The leaf certificate must match the pin; chain and
//      host name are not checked (the pin is the stronger statement).
//   3. otherwise: the chain must verify against ca_bundle_path (or the system
//      trust store when empty) and the certificate must match the host name or
//      IP address (or server_name when set). A pin, if set, must also match.
#pragma once

#include <chrono>
#include <string>

namespace sonder::inference::net {

struct TlsOptions {
    // PEM file with the trusted CA certificate(s). Empty = system trust store.
    std::string ca_bundle_path;
    // SHA-256 fingerprint of the server's leaf certificate (DER), as printed
    // by `openssl x509 -noout -fingerprint -sha256`. Hex, case-insensitive,
    // ':' separators optional.
    std::string pinned_sha256;
    // PEM file holding the exact expected leaf certificate (alternative or
    // addition to pinned_sha256; when both are set both must match).
    std::string pinned_cert_path;
    // DANGEROUS: skip certificate verification. For bring-up only.
    bool insecure_skip_verify = false;
    // SNI and verification name override (default: the URL host).
    std::string server_name;
    // Bound on the TLS handshake (the request's total timeout also applies).
    std::chrono::milliseconds handshake_timeout{10000};

    [[nodiscard]] bool has_pin() const noexcept { return !pinned_sha256.empty() || !pinned_cert_path.empty(); }
};

// True when this build was configured with SONDER_WITH_TLS=ON.
#if defined(SONDER_HAS_TLS)
inline constexpr bool kTlsCompiledIn = true;
#else
inline constexpr bool kTlsCompiledIn = false;
#endif

}  // namespace sonder::inference::net
