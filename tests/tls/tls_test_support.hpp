// Test-only PKI generation and a minimal loopback TLS/HTTP server (OpenSSL).
// Never used by the library.
#pragma once

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <csignal>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace tls_test {

#if defined(_WIN32)
using sock_t = SOCKET;
constexpr sock_t kBad = INVALID_SOCKET;
inline void close_sock(sock_t s) { closesocket(s); }
inline void init_sockets() {
    static const bool once = [] {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    (void)once;
}
#else
using sock_t = int;
constexpr sock_t kBad = -1;
inline void close_sock(sock_t s) { ::close(s); }
// The test server may write to a client that already hung up (e.g. after a
// pin mismatch); that must not kill the test process.
inline void init_sockets() { std::signal(SIGPIPE, SIG_IGN); }
#endif

struct KeyFree {
    void operator()(EVP_PKEY* p) const noexcept { EVP_PKEY_free(p); }
};
struct CertFree {
    void operator()(X509* p) const noexcept { X509_free(p); }
};
using KeyPtr = std::unique_ptr<EVP_PKEY, KeyFree>;
using CertPtr = std::unique_ptr<X509, CertFree>;

inline void check(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(std::string("tls test PKI: ") + what);
    }
}

inline KeyPtr make_key() {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
    check(ctx != nullptr, "EVP_PKEY_CTX_new_id");
    EVP_PKEY* key = nullptr;
    const bool ok = EVP_PKEY_keygen_init(ctx) == 1 &&
                    EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1) == 1 &&
                    EVP_PKEY_keygen(ctx, &key) == 1;
    EVP_PKEY_CTX_free(ctx);
    check(ok, "keygen");
    return KeyPtr(key);
}

inline void add_ext(X509* cert, X509* issuer, int nid, const char* value) {
    X509V3_CTX v3;
    X509V3_set_ctx_nodb(&v3);
    X509V3_set_ctx(&v3, issuer, cert, nullptr, nullptr, 0);
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &v3, nid, value);
    check(ext != nullptr, "X509V3_EXT_conf_nid");
    X509_add_ext(cert, ext, -1);
    X509_EXTENSION_free(ext);
}

// Issues a certificate for subject_key. issuer == nullptr => self-signed.
inline CertPtr make_cert(EVP_PKEY* subject_key, const char* cn, X509* issuer, EVP_PKEY* issuer_key, bool is_ca,
                         const char* san) {
    static std::atomic<long> serial{1000};
    CertPtr cert(X509_new());
    check(cert != nullptr, "X509_new");
    X509_set_version(cert.get(), 2);
    ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), serial++);
    X509_gmtime_adj(X509_getm_notBefore(cert.get()), -3600);
    X509_gmtime_adj(X509_getm_notAfter(cert.get()), 24 * 3600);
    X509_set_pubkey(cert.get(), subject_key);
    // Build the subject separately: OpenSSL 4 returns a const X509_NAME* from
    // X509_get_subject_name, so it can no longer be edited in place.
    std::unique_ptr<X509_NAME, void (*)(X509_NAME*)> name(X509_NAME_new(), X509_NAME_free);
    check(name != nullptr, "X509_NAME_new");
    X509_NAME_add_entry_by_txt(name.get(), "O", MBSTRING_ASC,
                               reinterpret_cast<const unsigned char*>("Sonder TLS test"), -1, -1, 0);
    X509_NAME_add_entry_by_txt(name.get(), "CN", MBSTRING_ASC, reinterpret_cast<const unsigned char*>(cn), -1, -1,
                               0);
    check(X509_set_subject_name(cert.get(), name.get()) == 1, "X509_set_subject_name");
    X509* iss = issuer != nullptr ? issuer : cert.get();
    X509_set_issuer_name(cert.get(), X509_get_subject_name(iss));
    if (is_ca) {
        add_ext(cert.get(), iss, NID_basic_constraints, "critical,CA:TRUE");
        add_ext(cert.get(), iss, NID_key_usage, "critical,keyCertSign,cRLSign");
        add_ext(cert.get(), iss, NID_subject_key_identifier, "hash");
    } else {
        add_ext(cert.get(), iss, NID_basic_constraints, "critical,CA:FALSE");
        add_ext(cert.get(), iss, NID_key_usage, "critical,digitalSignature");
        add_ext(cert.get(), iss, NID_ext_key_usage, "serverAuth");
        add_ext(cert.get(), iss, NID_authority_key_identifier, "keyid");
        add_ext(cert.get(), iss, NID_subject_alt_name, san);
    }
    check(X509_sign(cert.get(), issuer_key != nullptr ? issuer_key : subject_key, EVP_sha256()) > 0, "X509_sign");
    return cert;
}

inline void write_cert(const std::filesystem::path& p, X509* cert) {
    BIO* bio = BIO_new_file(p.string().c_str(), "w");
    check(bio != nullptr, "BIO_new_file");
    PEM_write_bio_X509(bio, cert);
    BIO_free(bio);
}

inline void write_key(const std::filesystem::path& p, EVP_PKEY* key) {
    BIO* bio = BIO_new_file(p.string().c_str(), "w");
    check(bio != nullptr, "BIO_new_file");
    PEM_write_bio_PrivateKey(bio, key, nullptr, nullptr, 0, nullptr, nullptr);
    BIO_free(bio);
}

inline std::string fingerprint_colon_upper(X509* cert) {
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    X509_digest(cert, EVP_sha256(), md, &len);
    static constexpr char kDigits[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned int i = 0; i < len; ++i) {
        if (i != 0) {
            out.push_back(':');
        }
        out.push_back(kDigits[md[i] >> 4]);
        out.push_back(kDigits[md[i] & 0x0f]);
    }
    return out;
}

// Files for one test run, in a fresh temporary directory:
//   ca.pem            trusted test CA
//   other_ca.pem      unrelated CA (not the issuer)
//   server.pem/.key   leaf for IP:127.0.0.1 + DNS:localhost, issued by ca
//   wrongname.pem/.key leaf for DNS:node1.invalid only, issued by ca
struct Pki {
    std::filesystem::path dir;
    std::string ca, other_ca, server_cert, server_key, wrong_cert, wrong_key;
    std::string server_fingerprint;  // "AB:CD:..." form
    std::string other_fingerprint;   // fingerprint of a different certificate

    Pki() {
        std::random_device rd;
        dir = std::filesystem::temp_directory_path() /
              ("sonder-tls-test-" + std::to_string(rd()) + "-" + std::to_string(rd()));
        std::filesystem::create_directories(dir);
        auto ca_key = make_key();
        auto ca_cert = make_cert(ca_key.get(), "Sonder Test CA", nullptr, nullptr, true, nullptr);
        auto other_key = make_key();
        auto other_cert = make_cert(other_key.get(), "Unrelated CA", nullptr, nullptr, true, nullptr);
        auto srv_key = make_key();
        auto srv_cert = make_cert(srv_key.get(), "127.0.0.1", ca_cert.get(), ca_key.get(), false,
                                  "IP:127.0.0.1,DNS:localhost");
        auto wrong_k = make_key();
        auto wrong_c =
            make_cert(wrong_k.get(), "node1.invalid", ca_cert.get(), ca_key.get(), false, "DNS:node1.invalid");
        ca = (dir / "ca.pem").string();
        other_ca = (dir / "other_ca.pem").string();
        server_cert = (dir / "server.pem").string();
        server_key = (dir / "server.key").string();
        wrong_cert = (dir / "wrongname.pem").string();
        wrong_key = (dir / "wrongname.key").string();
        write_cert(ca, ca_cert.get());
        write_cert(other_ca, other_cert.get());
        write_cert(server_cert, srv_cert.get());
        write_key(server_key, srv_key.get());
        write_cert(wrong_cert, wrong_c.get());
        write_key(wrong_key, wrong_k.get());
        server_fingerprint = fingerprint_colon_upper(srv_cert.get());
        other_fingerprint = fingerprint_colon_upper(wrong_c.get());
    }
    ~Pki() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    Pki(const Pki&) = delete;
    Pki& operator=(const Pki&) = delete;
};

// Loopback listener on 127.0.0.1:<ephemeral>. In TLS mode every connection is
// TLS-accepted and answered with a fixed HTTP/1.1 JSON response per path; in
// silent mode connections are accepted and never answered (timeouts).
class Server {
public:
    Server(const std::string& cert, const std::string& key) : silent_(false) {
        ctx_ = SSL_CTX_new(TLS_server_method());
        check(ctx_ != nullptr, "SSL_CTX_new");
        check(SSL_CTX_use_certificate_file(ctx_, cert.c_str(), SSL_FILETYPE_PEM) == 1, "server cert");
        check(SSL_CTX_use_PrivateKey_file(ctx_, key.c_str(), SSL_FILETYPE_PEM) == 1, "server key");
        start();
    }
    struct Silent {};
    explicit Server(Silent) : silent_(true) { start(); }

    ~Server() {
        stop_ = true;
        if (thread_.joinable()) {
            thread_.join();
        }
        if (listen_ != kBad) {
            close_sock(listen_);
        }
        if (ctx_ != nullptr) {
            SSL_CTX_free(ctx_);
        }
    }
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    [[nodiscard]] std::uint16_t port() const { return port_; }
    [[nodiscard]] int requests_served() const { return served_.load(); }
    [[nodiscard]] int handshakes_failed() const { return failed_.load(); }

private:
    void start() {
        init_sockets();
        listen_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        check(listen_ != kBad, "socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        check(::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "bind");
        check(::listen(listen_, 16) == 0, "listen");
        socklen_t len = sizeof(addr);
        check(::getsockname(listen_, reinterpret_cast<sockaddr*>(&addr), &len) == 0, "getsockname");
        port_ = ntohs(addr.sin_port);
        thread_ = std::thread([this] { run(); });
    }

    bool readable(sock_t s, int ms) const {
#if defined(_WIN32)
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s, &r);
        timeval tv{};
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        return ::select(0, &r, nullptr, nullptr, &tv) > 0;
#else
        pollfd p{};
        p.fd = s;
        p.events = POLLIN;
        return ::poll(&p, 1, ms) > 0;
#endif
    }

    static void set_timeouts(sock_t s) {
#if defined(_WIN32)
        DWORD ms = 5000;
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
        timeval tv{};
        tv.tv_sec = 5;
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    }

    void run() {
        while (!stop_) {
            if (!readable(listen_, 50)) {
                continue;
            }
            sock_t c = ::accept(listen_, nullptr, nullptr);
            if (c == kBad) {
                continue;
            }
            if (silent_) {
                // Hold the connection open without speaking until told to stop.
                while (!stop_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                close_sock(c);
                continue;
            }
            set_timeouts(c);
            serve(c);
            close_sock(c);
        }
    }

    void serve(sock_t c) {
        SSL* ssl = SSL_new(ctx_);
        SSL_set_fd(ssl, static_cast<int>(c));
        if (SSL_accept(ssl) != 1) {
            ++failed_;
            ERR_clear_error();
            SSL_free(ssl);
            return;
        }
        std::string req;
        char buf[4096];
        while (req.find("\r\n\r\n") == std::string::npos && req.size() < 65536) {
            const int n = SSL_read(ssl, buf, static_cast<int>(sizeof(buf)));
            if (n <= 0) {
                break;
            }
            req.append(buf, static_cast<std::size_t>(n));
        }
        // Drain a Content-Length request body so the client can finish writing.
        const auto head_end = req.find("\r\n\r\n");
        if (head_end == std::string::npos) {
            // Client hung up without sending a request (e.g. rejected the
            // certificate after the handshake): nothing is served.
            SSL_free(ssl);
            ERR_clear_error();
            return;
        }
        std::size_t want = 0;
        const auto cl = req.find("Content-Length: ");
        if (cl != std::string::npos && cl < head_end) {
            want = static_cast<std::size_t>(std::stoull(req.substr(cl + 16)));
        }
        while (req.size() - (head_end + 4) < want) {
            const int n = SSL_read(ssl, buf, static_cast<int>(sizeof(buf)));
            if (n <= 0) {
                break;
            }
            req.append(buf, static_cast<std::size_t>(n));
        }
        std::string body;
        if (req.rfind("GET /api/version ", 0) == 0) {
            body = R"({"version":"0.0.0-tls-test"})";
        } else {
            body = R"({"ok":true,"path":")" + req.substr(0, req.find("\r\n")) + "\"}";
        }
        const std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                                 std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        ++served_;  // before the write, so a client that saw the reply sees the count
        SSL_write(ssl, resp.data(), static_cast<int>(resp.size()));
        SSL_shutdown(ssl);
        SSL_free(ssl);
        ERR_clear_error();
    }

    bool silent_;
    SSL_CTX* ctx_ = nullptr;
    sock_t listen_ = kBad;
    std::uint16_t port_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<int> served_{0};
    std::atomic<int> failed_{0};
    std::thread thread_;
};

}  // namespace tls_test
