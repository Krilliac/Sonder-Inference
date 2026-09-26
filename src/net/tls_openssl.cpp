// OpenSSL TLS backend (SONDER_WITH_TLS=ON, SONDER_TLS_BACKEND=openssl).
// Default on Linux/macOS; usable on Windows when OpenSSL is installed.
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tls_stream.hpp"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#else
#  include <cerrno>
#  include <sys/socket.h>
#  include <sys/types.h>
#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L
#  error "SONDER_WITH_TLS with the OpenSSL backend requires OpenSSL 1.1.1 or newer"
#endif

namespace sonder::inference::net {
namespace {

std::string drain_ssl_errors() {
    std::string out;
    unsigned long e = 0;
    while ((e = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(e, buf, sizeof(buf));
        if (!out.empty()) {
            out += "; ";
        }
        out += buf;
    }
    return out;
}

struct CtxFree {
    void operator()(SSL_CTX* p) const noexcept { SSL_CTX_free(p); }
};
struct SslFree {
    void operator()(SSL* p) const noexcept { SSL_free(p); }
};
struct X509Free {
    void operator()(X509* p) const noexcept { X509_free(p); }
};
using CtxPtr = std::unique_ptr<SSL_CTX, CtxFree>;
using SslPtr = std::unique_ptr<SSL, SslFree>;
using X509Ptr = std::unique_ptr<X509, X509Free>;

// Socket BIO that sends with MSG_NOSIGNAL (a peer hang-up must surface as an
// error, not SIGPIPE; OpenSSL's stock socket BIO uses write()). The BIO data
// holds a heap copy of the native socket handle; the socket is not owned.
bool bio_would_block() {
#if defined(_WIN32)
    const int err = WSAGetLastError();
    return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
#endif
}

NativeSocket bio_socket(BIO* b) { return *static_cast<NativeSocket*>(BIO_get_data(b)); }

int bio_write(BIO* b, const char* data, int len) {
    BIO_clear_retry_flags(b);
#if defined(_WIN32)
    const int n = ::send(static_cast<SOCKET>(bio_socket(b)), data, len, 0);
#else
    const auto n = ::send(static_cast<int>(bio_socket(b)), data, static_cast<std::size_t>(len), MSG_NOSIGNAL);
#endif
    if (n < 0 && bio_would_block()) {
        BIO_set_retry_write(b);
    }
    return static_cast<int>(n);
}

int bio_read(BIO* b, char* data, int len) {
    BIO_clear_retry_flags(b);
#if defined(_WIN32)
    const int n = ::recv(static_cast<SOCKET>(bio_socket(b)), data, len, 0);
#else
    const auto n = ::recv(static_cast<int>(bio_socket(b)), data, static_cast<std::size_t>(len), 0);
#endif
    if (n < 0 && bio_would_block()) {
        BIO_set_retry_read(b);
    }
    return static_cast<int>(n);
}

int bio_puts(BIO* b, const char* str) { return bio_write(b, str, static_cast<int>(std::strlen(str))); }

long bio_ctrl(BIO* /*b*/, int cmd, long /*num*/, void* /*ptr*/) { return cmd == BIO_CTRL_FLUSH ? 1 : 0; }

int bio_create(BIO* b) {
    BIO_set_data(b, nullptr);
    BIO_set_init(b, 0);
    return 1;
}

int bio_destroy(BIO* b) {
    delete static_cast<NativeSocket*>(BIO_get_data(b));
    BIO_set_data(b, nullptr);
    return 1;
}

BIO_METHOD* socket_bio_method() {
    // Created once and intentionally never freed (process lifetime).
    static BIO_METHOD* method = [] {
        BIO_METHOD* m = BIO_meth_new(BIO_get_new_index() | BIO_TYPE_SOURCE_SINK, "sonder-socket");
        if (m != nullptr) {
            BIO_meth_set_write(m, bio_write);
            BIO_meth_set_read(m, bio_read);
            BIO_meth_set_puts(m, bio_puts);
            BIO_meth_set_ctrl(m, bio_ctrl);
            BIO_meth_set_create(m, bio_create);
            BIO_meth_set_destroy(m, bio_destroy);
        }
        return m;
    }();
    return method;
}

BIO* make_socket_bio(NativeSocket s) {
    BIO_METHOD* method = socket_bio_method();
    if (method == nullptr) {
        return nullptr;
    }
    BIO* b = BIO_new(method);
    if (b != nullptr) {
        BIO_set_data(b, new NativeSocket(s));
        BIO_set_init(b, 1);
    }
    return b;
}

Result<std::string> cert_sha256(X509* cert) {
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    if (X509_digest(cert, EVP_sha256(), md, &len) != 1) {
        return Status(ErrorCode::internal, "TLS: cannot hash certificate: " + drain_ssl_errors());
    }
    return to_hex(md, len);
}

Result<std::string> pem_file_sha256(const std::string& path) {
    auto text = read_text_file(path, "pinned certificate");
    if (!text.ok()) {
        return text.status();
    }
    BIO* bio = BIO_new_mem_buf(text.value().data(), static_cast<int>(text.value().size()));
    if (bio == nullptr) {
        return Status(ErrorCode::internal, "TLS: out of memory");
    }
    X509Ptr cert(PEM_read_bio_X509(bio, nullptr, nullptr, nullptr));
    BIO_free(bio);
    if (!cert) {
        drain_ssl_errors();
        return Status(ErrorCode::invalid_argument, "TLS: '" + path + "' is not a PEM certificate");
    }
    return cert_sha256(cert.get());
}

class OpenSslStream final : public TlsStream {
public:
    OpenSslStream(CtxPtr ctx, SslPtr ssl, NativeSocket s) : ctx_(std::move(ctx)), ssl_(std::move(ssl)), socket_(s) {}
    ~OpenSslStream() override {
        if (ssl_) {
            // Best-effort close_notify; never blocks (socket is non-blocking).
            (void)SSL_shutdown(ssl_.get());
            ERR_clear_error();
        }
    }
    OpenSslStream(const OpenSslStream&) = delete;
    OpenSslStream& operator=(const OpenSslStream&) = delete;

    Status write_all(std::string_view data, Deadline deadline, const CancellationToken& cancel) override {
        std::size_t sent = 0;
        while (sent < data.size()) {
            std::size_t n = 0;
            const int rc = SSL_write_ex(ssl_.get(), data.data() + sent, data.size() - sent, &n);
            if (rc == 1) {
                sent += n;
                continue;
            }
            const int err = SSL_get_error(ssl_.get(), rc);
            if (err == SSL_ERROR_WANT_WRITE || err == SSL_ERROR_WANT_READ) {
                if (auto st = wait_native_socket(socket_, err == SSL_ERROR_WANT_WRITE, deadline, cancel); !st.ok()) {
                    return st;
                }
                continue;
            }
            return Status(ErrorCode::io_error, "TLS write failed: " + drain_ssl_errors());
        }
        return Status::success();
    }

    Result<std::size_t> read_some(char* buf, std::size_t len, Deadline deadline,
                                  const CancellationToken& cancel) override {
        while (true) {
            if (cancel.cancelled()) {
                return Status(ErrorCode::cancelled, "cancelled");
            }
            std::size_t n = 0;
            const int rc = SSL_read_ex(ssl_.get(), buf, len, &n);
            if (rc == 1) {
                return n;
            }
            const int err = SSL_get_error(ssl_.get(), rc);
            if (err == SSL_ERROR_ZERO_RETURN) {
                return std::size_t{0};
            }
            if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                if (auto st = wait_native_socket(socket_, err == SSL_ERROR_WANT_WRITE, deadline, cancel); !st.ok()) {
                    return st;
                }
                continue;
            }
            if (err == SSL_ERROR_SYSCALL && ERR_peek_error() == 0) {
                // Peer closed without close_notify (common with
                // "Connection: close" servers). Body framing in the HTTP
                // layer still detects truncation.
                return std::size_t{0};
            }
            return Status(ErrorCode::io_error, "TLS read failed: " + drain_ssl_errors());
        }
    }

private:
    CtxPtr ctx_;
    SslPtr ssl_;
    NativeSocket socket_;
};

}  // namespace

const char* tls_backend_name() noexcept { return "openssl"; }

Result<std::unique_ptr<TlsStream>> tls_connect(NativeSocket socket, const std::string& host, std::uint16_t port,
                                               const TlsOptions& options, Deadline deadline,
                                               const CancellationToken& cancel) {
    ERR_clear_error();
    // Resolve pins first so option errors surface before any network I/O.
    std::vector<std::string> pins;
    if (!options.pinned_sha256.empty()) {
        auto fp = normalize_sha256_fingerprint(options.pinned_sha256);
        if (!fp.ok()) {
            return fp.status();
        }
        pins.push_back(std::move(fp).value());
    }
    if (!options.pinned_cert_path.empty()) {
        auto fp = pem_file_sha256(options.pinned_cert_path);
        if (!fp.ok()) {
            return fp.status();
        }
        pins.push_back(std::move(fp).value());
    }
    const bool pin_only = !pins.empty() && options.ca_bundle_path.empty() && !options.insecure_skip_verify;
    const bool verify_chain = !options.insecure_skip_verify && !pin_only;
    const std::string& name = options.server_name.empty() ? host : options.server_name;

    CtxPtr ctx(SSL_CTX_new(TLS_client_method()));
    if (!ctx) {
        return Status(ErrorCode::internal, "TLS: SSL_CTX_new failed: " + drain_ssl_errors());
    }
    SSL_CTX_set_min_proto_version(ctx.get(), TLS1_2_VERSION);
#if defined(SSL_OP_IGNORE_UNEXPECTED_EOF)
    SSL_CTX_set_options(ctx.get(), SSL_OP_IGNORE_UNEXPECTED_EOF);
#endif
    if (verify_chain) {
        if (!options.ca_bundle_path.empty()) {
            if (SSL_CTX_load_verify_locations(ctx.get(), options.ca_bundle_path.c_str(), nullptr) != 1) {
                return Status(ErrorCode::invalid_argument,
                              "TLS: cannot load CA bundle '" + options.ca_bundle_path + "': " + drain_ssl_errors());
            }
        } else if (SSL_CTX_set_default_verify_paths(ctx.get()) != 1) {
            drain_ssl_errors();  // no system store: every chain will fail, which is the safe outcome
        }
        SSL_CTX_set_verify(ctx.get(), SSL_VERIFY_PEER, nullptr);
    } else {
        SSL_CTX_set_verify(ctx.get(), SSL_VERIFY_NONE, nullptr);
    }
    if (options.insecure_skip_verify) {
        warn_insecure_tls(name, port);
    }

    SslPtr ssl(SSL_new(ctx.get()));
    if (!ssl) {
        return Status(ErrorCode::internal, "TLS: SSL_new failed: " + drain_ssl_errors());
    }
    BIO* bio = make_socket_bio(socket);
    if (bio == nullptr) {
        return Status(ErrorCode::internal, "TLS: cannot create socket BIO: " + drain_ssl_errors());
    }
    SSL_set_bio(ssl.get(), bio, bio);  // ssl owns the BIO from here

    const bool ip = is_ip_literal(name);
    if (!ip) {
        SSL_set_tlsext_host_name(ssl.get(), name.c_str());
    }
    if (verify_chain) {
        if (ip) {
            X509_VERIFY_PARAM* param = SSL_get0_param(ssl.get());
            if (X509_VERIFY_PARAM_set1_ip_asc(param, name.c_str()) != 1) {
                return Status(ErrorCode::invalid_argument, "TLS: invalid IP address '" + name + "'");
            }
        } else if (SSL_set1_host(ssl.get(), name.c_str()) != 1) {
            return Status(ErrorCode::invalid_argument, "TLS: invalid host name '" + name + "'");
        }
    }
    SSL_set_connect_state(ssl.get());

    const auto hs_deadline = std::min(deadline, std::chrono::steady_clock::now() + options.handshake_timeout);
    while (true) {
        const int rc = SSL_connect(ssl.get());
        if (rc == 1) {
            break;
        }
        const int err = SSL_get_error(ssl.get(), rc);
        if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
            auto st = wait_native_socket(socket, err == SSL_ERROR_WANT_WRITE, hs_deadline, cancel);
            if (!st.ok()) {
                if (st.code() == ErrorCode::timeout) {
                    return Status(ErrorCode::timeout, "TLS handshake with " + host + " timed out");
                }
                return st;
            }
            continue;
        }
        const long vr = SSL_get_verify_result(ssl.get());
        if (verify_chain && vr != X509_V_OK) {
            drain_ssl_errors();
            return Status(ErrorCode::protocol_error, "TLS certificate verification failed for " + name + ": " +
                                                         X509_verify_cert_error_string(vr));
        }
        const std::string detail = drain_ssl_errors();
        return Status(ErrorCode::protocol_error,
                      "TLS handshake with " + host + " failed" + (detail.empty() ? std::string() : ": " + detail));
    }

    if (!pins.empty()) {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
        X509Ptr peer(SSL_get1_peer_certificate(ssl.get()));
#else
        X509Ptr peer(SSL_get_peer_certificate(ssl.get()));
#endif
        if (!peer) {
            return Status(ErrorCode::protocol_error, "TLS: server presented no certificate");
        }
        auto got = cert_sha256(peer.get());
        if (!got.ok()) {
            return got.status();
        }
        for (const auto& pin : pins) {
            if (pin != got.value()) {
                return Status(ErrorCode::protocol_error,
                              "TLS certificate pin mismatch for " + name + ": expected sha256 " + pin + ", got " +
                                  got.value());
            }
        }
    }
    return std::unique_ptr<TlsStream>(std::make_unique<OpenSslStream>(std::move(ctx), std::move(ssl), socket));
}

}  // namespace sonder::inference::net
