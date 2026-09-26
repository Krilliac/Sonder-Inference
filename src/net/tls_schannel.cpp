// Schannel TLS backend (SONDER_WITH_TLS=ON, SONDER_TLS_BACKEND=schannel).
// Default on Windows: uses the OS TLS stack (secur32/crypt32/bcrypt), no
// third-party dependency. TLS 1.2 via SCHANNEL_CRED. Certificate validation
// is done manually (SCH_CRED_MANUAL_CRED_VALIDATION) so the CA-bundle, pin and
// insecure modes behave exactly like the OpenSSL backend (see tls.hpp).
// Revocation is not checked (private-network nodes; see docs/integration/tls.md).
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef SECURITY_WIN32
#  define SECURITY_WIN32
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include <wincrypt.h>
#include <schannel.h>
#include <security.h>
#include <sspi.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tls_stream.hpp"

#ifndef SECURITY_FLAG_IGNORE_UNKNOWN_CA
#  define SECURITY_FLAG_IGNORE_UNKNOWN_CA 0x00000100
#endif
#ifndef SP_PROT_TLS1_2_CLIENT
#  define SP_PROT_TLS1_2_CLIENT 0x00000800
#endif

namespace sonder::inference::net {
namespace {

std::string hex32(long value) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08lx", static_cast<unsigned long>(value));
    return buf;
}

std::wstring widen(const std::string& s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

// ---------------------------------------------------------------- crypto

Result<std::string> sha256_hex(const unsigned char* data, std::size_t len) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) {
        return Status(ErrorCode::internal, "TLS: SHA-256 provider unavailable");
    }
    BCRYPT_HASH_HANDLE hash = nullptr;
    unsigned char digest[32];
    bool ok = BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) == 0;
    ok = ok && BCryptHashData(hash, const_cast<PUCHAR>(data), static_cast<ULONG>(len), 0) == 0;
    ok = ok && BCryptFinishHash(hash, digest, sizeof(digest), 0) == 0;
    if (hash != nullptr) {
        BCryptDestroyHash(hash);
    }
    BCryptCloseAlgorithmProvider(alg, 0);
    if (!ok) {
        return Status(ErrorCode::internal, "TLS: SHA-256 failed");
    }
    return to_hex(digest, sizeof(digest));
}

struct CertFree {
    void operator()(const CERT_CONTEXT* c) const noexcept { CertFreeCertificateContext(c); }
};
using CertPtr = std::unique_ptr<const CERT_CONTEXT, CertFree>;
struct StoreClose {
    void operator()(void* s) const noexcept { CertCloseStore(static_cast<HCERTSTORE>(s), 0); }
};
using StorePtr = std::unique_ptr<void, StoreClose>;
struct ChainFree {
    void operator()(const CERT_CHAIN_CONTEXT* c) const noexcept { CertFreeCertificateChain(c); }
};
using ChainPtr = std::unique_ptr<const CERT_CHAIN_CONTEXT, ChainFree>;

// Decodes every "BEGIN CERTIFICATE" block of a PEM text into DER blobs.
std::vector<std::vector<unsigned char>> pem_certificates(const std::string& pem) {
    static constexpr const char* kBegin = "-----BEGIN CERTIFICATE-----";
    static constexpr const char* kEnd = "-----END CERTIFICATE-----";
    std::vector<std::vector<unsigned char>> out;
    std::size_t pos = 0;
    while ((pos = pem.find(kBegin, pos)) != std::string::npos) {
        const auto end = pem.find(kEnd, pos);
        if (end == std::string::npos) {
            break;
        }
        const std::string block = pem.substr(pos, end + std::strlen(kEnd) - pos);
        DWORD size = 0;
        if (CryptStringToBinaryA(block.c_str(), static_cast<DWORD>(block.size()), CRYPT_STRING_BASE64HEADER, nullptr,
                                 &size, nullptr, nullptr) &&
            size > 0) {
            std::vector<unsigned char> der(size);
            if (CryptStringToBinaryA(block.c_str(), static_cast<DWORD>(block.size()), CRYPT_STRING_BASE64HEADER,
                                     der.data(), &size, nullptr, nullptr)) {
                der.resize(size);
                out.push_back(std::move(der));
            }
        }
        pos = end;
    }
    return out;
}

Result<std::string> pem_file_sha256(const std::string& path) {
    auto text = read_text_file(path, "pinned certificate");
    if (!text.ok()) {
        return text.status();
    }
    const auto ders = pem_certificates(text.value());
    if (ders.empty()) {
        return Status(ErrorCode::invalid_argument, "TLS: '" + path + "' is not a PEM certificate");
    }
    CertPtr cert(CertCreateCertificateContext(X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, ders.front().data(),
                                              static_cast<DWORD>(ders.front().size())));
    if (!cert) {
        return Status(ErrorCode::invalid_argument, "TLS: '" + path + "' is not a valid certificate");
    }
    return sha256_hex(cert->pbCertEncoded, cert->cbCertEncoded);
}

Result<StorePtr> load_ca_bundle(const std::string& path) {
    auto text = read_text_file(path, "CA bundle");
    if (!text.ok()) {
        return text.status();
    }
    StorePtr store(CertOpenStore(CERT_STORE_PROV_MEMORY, 0, 0, CERT_STORE_CREATE_NEW_FLAG, nullptr));
    if (!store) {
        return Status(ErrorCode::internal, "TLS: cannot create certificate store");
    }
    int added = 0;
    for (const auto& der : pem_certificates(text.value())) {
        if (CertAddEncodedCertificateToStore(static_cast<HCERTSTORE>(store.get()), X509_ASN_ENCODING, der.data(),
                                             static_cast<DWORD>(der.size()), CERT_STORE_ADD_USE_EXISTING, nullptr)) {
            ++added;
        }
    }
    if (added == 0) {
        return Status(ErrorCode::invalid_argument, "TLS: cannot load CA bundle '" + path + "': no PEM certificates");
    }
    return store;
}

bool store_contains(HCERTSTORE store, const CERT_CONTEXT* cert) {
    const CERT_CONTEXT* it = nullptr;
    while ((it = CertEnumCertificatesInStore(store, it)) != nullptr) {
        if (it->cbCertEncoded == cert->cbCertEncoded &&
            std::memcmp(it->pbCertEncoded, cert->pbCertEncoded, cert->cbCertEncoded) == 0) {
            CertFreeCertificateContext(it);
            return true;
        }
    }
    return false;
}

// True when the certificate's subjectAltName carries the given IP address.
bool cert_matches_ip(const CERT_CONTEXT* cert, const std::string& ip) {
    unsigned char want[16];
    DWORD want_len = 0;
    if (inet_pton(AF_INET, ip.c_str(), want) == 1) {
        want_len = 4;
    } else if (inet_pton(AF_INET6, ip.c_str(), want) == 1) {
        want_len = 16;
    } else {
        return false;
    }
    const CERT_EXTENSION* ext = CertFindExtension(szOID_SUBJECT_ALT_NAME2, cert->pCertInfo->cExtension,
                                                  cert->pCertInfo->rgExtension);
    if (ext == nullptr) {
        return false;
    }
    CERT_ALT_NAME_INFO* info = nullptr;
    DWORD size = 0;
    if (!CryptDecodeObjectEx(X509_ASN_ENCODING, X509_ALTERNATE_NAME, ext->Value.pbData, ext->Value.cbData,
                             CRYPT_DECODE_ALLOC_FLAG, nullptr, &info, &size)) {
        return false;
    }
    bool match = false;
    for (DWORD i = 0; i < info->cAltEntry && !match; ++i) {
        const CERT_ALT_NAME_ENTRY& e = info->rgAltEntry[i];
        match = e.dwAltNameChoice == CERT_ALT_NAME_IP_ADDRESS && e.IPAddress.cbData == want_len &&
                std::memcmp(e.IPAddress.pbData, want, want_len) == 0;
    }
    LocalFree(info);
    return match;
}

// Chain + name validation for the default (non-pin-only, non-insecure) mode.
Status verify_server_cert(const CERT_CONTEXT* cert, const std::string& name, HCERTSTORE bundle) {
    StorePtr extra;
    if (bundle != nullptr) {
        extra.reset(CertOpenStore(CERT_STORE_PROV_COLLECTION, 0, 0, 0, nullptr));
        if (!extra) {
            return Status(ErrorCode::internal, "TLS: cannot create certificate store");
        }
        CertAddStoreToCollection(static_cast<HCERTSTORE>(extra.get()), bundle, 0, 0);
        if (cert->hCertStore != nullptr) {
            CertAddStoreToCollection(static_cast<HCERTSTORE>(extra.get()), cert->hCertStore, 0, 0);
        }
    }
    LPSTR usages[] = {const_cast<LPSTR>(szOID_PKIX_KP_SERVER_AUTH)};
    CERT_CHAIN_PARA para{};
    para.cbSize = sizeof(para);
    para.RequestedUsage.dwType = USAGE_MATCH_TYPE_AND;
    para.RequestedUsage.Usage.cUsageIdentifier = 1;
    para.RequestedUsage.Usage.rgpszUsageIdentifier = usages;
    const CERT_CHAIN_CONTEXT* raw_chain = nullptr;
    if (!CertGetCertificateChain(nullptr, cert, nullptr, extra ? static_cast<HCERTSTORE>(extra.get()) : cert->hCertStore,
                                 &para, 0, nullptr, &raw_chain)) {
        return Status(ErrorCode::protocol_error,
                      "TLS certificate verification failed for " + name + ": cannot build chain (" +
                          hex32(static_cast<long>(GetLastError())) + ")");
    }
    ChainPtr chain(raw_chain);
    DWORD errors = chain->TrustStatus.dwErrorStatus &
                   ~static_cast<DWORD>(CERT_TRUST_REVOCATION_STATUS_UNKNOWN | CERT_TRUST_IS_OFFLINE_REVOCATION);
    if (bundle != nullptr) {
        // Only the bundle is trusted: the chain must end at one of its certs.
        const CERT_SIMPLE_CHAIN* simple = chain->cChain > 0 ? chain->rgpChain[0] : nullptr;
        const CERT_CONTEXT* root =
            simple != nullptr && simple->cElement > 0 ? simple->rgpElement[simple->cElement - 1]->pCertContext : nullptr;
        if (root == nullptr || !store_contains(bundle, root)) {
            return Status(ErrorCode::protocol_error,
                          "TLS certificate verification failed for " + name + ": not issued by the configured CA bundle");
        }
        errors &= ~static_cast<DWORD>(CERT_TRUST_IS_UNTRUSTED_ROOT);
    }
    if (errors != 0) {
        return Status(ErrorCode::protocol_error, "TLS certificate verification failed for " + name +
                                                     ": chain trust error " + hex32(static_cast<long>(errors)));
    }
    if (is_ip_literal(name)) {
        if (!cert_matches_ip(cert, name)) {
            return Status(ErrorCode::protocol_error,
                          "TLS certificate verification failed for " + name + ": IP address mismatch");
        }
        return Status::success();
    }
    std::wstring wname = widen(name);
    SSL_EXTRA_CERT_CHAIN_POLICY_PARA ssl{};
    ssl.cbSize = sizeof(ssl);
    ssl.dwAuthType = AUTHTYPE_SERVER;
    ssl.fdwChecks = bundle != nullptr ? SECURITY_FLAG_IGNORE_UNKNOWN_CA : 0;
    ssl.pwszServerName = wname.data();
    CERT_CHAIN_POLICY_PARA policy{};
    policy.cbSize = sizeof(policy);
    policy.pvExtraPolicyPara = &ssl;
    CERT_CHAIN_POLICY_STATUS status{};
    status.cbSize = sizeof(status);
    if (!CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, chain.get(), &policy, &status)) {
        return Status(ErrorCode::protocol_error, "TLS certificate verification failed for " + name);
    }
    if (status.dwError != 0) {
        const bool name_error = status.dwError == static_cast<DWORD>(CERT_E_CN_NO_MATCH);
        return Status(ErrorCode::protocol_error,
                      "TLS certificate verification failed for " + name + ": " +
                          (name_error ? std::string("host name mismatch")
                                      : "policy error " + hex32(static_cast<long>(status.dwError))));
    }
    return Status::success();
}

// ------------------------------------------------------------ raw socket

bool would_block() {
    const int err = WSAGetLastError();
    return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS || err == WSAEINTR;
}

Status send_raw(NativeSocket s, const char* data, std::size_t len, Deadline deadline, const CancellationToken& cancel) {
    std::size_t sent = 0;
    while (sent < len) {
        const int chunk = static_cast<int>(std::min<std::size_t>(len - sent, 1 << 20));
        const int n = ::send(static_cast<SOCKET>(s), data + sent, chunk, 0);
        if (n > 0) {
            sent += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && would_block()) {
            if (auto st = wait_native_socket(s, true, deadline, cancel); !st.ok()) {
                return st;
            }
            continue;
        }
        return Status(ErrorCode::io_error, "TLS: send failed");
    }
    return Status::success();
}

// Appends received bytes to buf; returns the count (0 = peer closed).
Result<std::size_t> recv_raw(NativeSocket s, std::string& buf, Deadline deadline, const CancellationToken& cancel) {
    char tmp[16384];
    while (true) {
        if (auto st = wait_native_socket(s, false, deadline, cancel); !st.ok()) {
            return st;
        }
        const int n = ::recv(static_cast<SOCKET>(s), tmp, static_cast<int>(sizeof(tmp)), 0);
        if (n > 0) {
            buf.append(tmp, static_cast<std::size_t>(n));
            return static_cast<std::size_t>(n);
        }
        if (n == 0) {
            return std::size_t{0};
        }
        if (!would_block()) {
            return Status(ErrorCode::io_error, "TLS: recv failed");
        }
    }
}

// --------------------------------------------------------------- session

struct Session {
    CredHandle cred{};
    CtxtHandle ctx{};
    bool has_cred = false;
    bool has_ctx = false;
    Session() = default;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() {
        if (has_ctx) {
            DeleteSecurityContext(&ctx);
        }
        if (has_cred) {
            FreeCredentialsHandle(&cred);
        }
    }
};

class SchannelStream final : public TlsStream {
public:
    SchannelStream(std::unique_ptr<Session> session, NativeSocket s, SecPkgContext_StreamSizes sizes,
                   std::string leftover)
        : session_(std::move(session)), socket_(s), sizes_(sizes), incoming_(std::move(leftover)) {}

    Status write_all(std::string_view data, Deadline deadline, const CancellationToken& cancel) override {
        std::vector<char> msg;
        std::size_t off = 0;
        while (off < data.size()) {
            const std::size_t n = std::min<std::size_t>(data.size() - off, sizes_.cbMaximumMessage);
            msg.assign(sizes_.cbHeader + n + sizes_.cbTrailer, 0);
            std::memcpy(msg.data() + sizes_.cbHeader, data.data() + off, n);
            SecBuffer bufs[4];
            bufs[0] = {sizes_.cbHeader, SECBUFFER_STREAM_HEADER, msg.data()};
            bufs[1] = {static_cast<unsigned long>(n), SECBUFFER_DATA, msg.data() + sizes_.cbHeader};
            bufs[2] = {sizes_.cbTrailer, SECBUFFER_STREAM_TRAILER, msg.data() + sizes_.cbHeader + n};
            bufs[3] = {0, SECBUFFER_EMPTY, nullptr};
            SecBufferDesc desc{SECBUFFER_VERSION, 4, bufs};
            const SECURITY_STATUS ss = EncryptMessage(&session_->ctx, 0, &desc, 0);
            if (ss != SEC_E_OK) {
                return Status(ErrorCode::io_error, "TLS: EncryptMessage failed " + hex32(ss));
            }
            const std::size_t total = bufs[0].cbBuffer + bufs[1].cbBuffer + bufs[2].cbBuffer;
            if (auto st = send_raw(socket_, msg.data(), total, deadline, cancel); !st.ok()) {
                return st;
            }
            off += n;
        }
        return Status::success();
    }

    Result<std::size_t> read_some(char* buf, std::size_t len, Deadline deadline,
                                  const CancellationToken& cancel) override {
        while (true) {
            if (cancel.cancelled()) {
                return Status(ErrorCode::cancelled, "cancelled");
            }
            if (!plain_.empty()) {
                const std::size_t n = std::min(len, plain_.size());
                std::memcpy(buf, plain_.data(), n);
                plain_.erase(0, n);
                return n;
            }
            if (eof_) {
                return std::size_t{0};
            }
            if (!incoming_.empty()) {
                SecBuffer bufs[4];
                bufs[0] = {static_cast<unsigned long>(incoming_.size()), SECBUFFER_DATA, incoming_.data()};
                bufs[1] = {0, SECBUFFER_EMPTY, nullptr};
                bufs[2] = {0, SECBUFFER_EMPTY, nullptr};
                bufs[3] = {0, SECBUFFER_EMPTY, nullptr};
                SecBufferDesc desc{SECBUFFER_VERSION, 4, bufs};
                const SECURITY_STATUS ss = DecryptMessage(&session_->ctx, &desc, 0, nullptr);
                if (ss == SEC_E_OK || ss == SEC_I_CONTEXT_EXPIRED || ss == SEC_I_RENEGOTIATE) {
                    std::string extra;
                    for (auto& b : bufs) {
                        if (b.BufferType == SECBUFFER_DATA && b.cbBuffer > 0) {
                            plain_.append(static_cast<const char*>(b.pvBuffer), b.cbBuffer);
                        } else if (b.BufferType == SECBUFFER_EXTRA && b.cbBuffer > 0) {
                            extra.assign(incoming_.data() + incoming_.size() - b.cbBuffer, b.cbBuffer);
                        }
                    }
                    incoming_ = std::move(extra);
                    if (ss == SEC_I_CONTEXT_EXPIRED) {
                        eof_ = true;  // close_notify
                    } else if (ss == SEC_I_RENEGOTIATE) {
                        // TLS 1.3 post-handshake messages or renegotiation are
                        // not supported (SCHANNEL_CRED negotiates TLS 1.2).
                        return Status(ErrorCode::protocol_error, "TLS: server requested renegotiation");
                    }
                    continue;
                }
                if (ss != SEC_E_INCOMPLETE_MESSAGE) {
                    return Status(ErrorCode::io_error, "TLS: DecryptMessage failed " + hex32(ss));
                }
            }
            auto got = recv_raw(socket_, incoming_, deadline, cancel);
            if (!got.ok()) {
                return got.status();
            }
            if (got.value() == 0) {
                // Peer closed without close_notify; HTTP framing still detects
                // truncated bodies.
                eof_ = true;
                incoming_.clear();
            }
        }
    }

private:
    std::unique_ptr<Session> session_;
    NativeSocket socket_;
    SecPkgContext_StreamSizes sizes_;
    std::string incoming_;  // encrypted bytes not yet decrypted
    std::string plain_;     // decrypted bytes not yet returned
    bool eof_ = false;
};

}  // namespace

const char* tls_backend_name() noexcept { return "schannel"; }

Result<std::unique_ptr<TlsStream>> tls_connect(NativeSocket socket, const std::string& host, std::uint16_t port,
                                               const TlsOptions& options, Deadline deadline,
                                               const CancellationToken& cancel) {
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
    StorePtr bundle;
    if (verify_chain && !options.ca_bundle_path.empty()) {
        auto loaded = load_ca_bundle(options.ca_bundle_path);
        if (!loaded.ok()) {
            return loaded.status();
        }
        bundle = std::move(loaded).value();
    }
    if (options.insecure_skip_verify) {
        warn_insecure_tls(name, port);
    }

    auto session = std::make_unique<Session>();
    SCHANNEL_CRED cred{};
    cred.dwVersion = SCHANNEL_CRED_VERSION;
    cred.grbitEnabledProtocols = SP_PROT_TLS1_2_CLIENT;
    cred.dwFlags = SCH_CRED_MANUAL_CRED_VALIDATION | SCH_CRED_NO_DEFAULT_CREDS | SCH_USE_STRONG_CRYPTO;
    TimeStamp expiry{};
    wchar_t package[] = UNISP_NAME_W;
    SECURITY_STATUS ss = AcquireCredentialsHandleW(nullptr, package, SECPKG_CRED_OUTBOUND, nullptr, &cred, nullptr,
                                                   nullptr, &session->cred, &expiry);
    if (ss != SEC_E_OK) {
        return Status(ErrorCode::internal, "TLS: AcquireCredentialsHandle failed " + hex32(ss));
    }
    session->has_cred = true;

    // SNI only for DNS names (RFC 6066 forbids IP literals).
    std::wstring target = is_ip_literal(name) ? std::wstring() : widen(name);
    const unsigned long req_flags = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT | ISC_REQ_CONFIDENTIALITY |
                                    ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_STREAM | ISC_REQ_EXTENDED_ERROR |
                                    ISC_REQ_MANUAL_CRED_VALIDATION;
    const auto hs_deadline = std::min(deadline, std::chrono::steady_clock::now() + options.handshake_timeout);
    auto hs_error = [&](const Status& st) {
        if (st.code() == ErrorCode::timeout) {
            return Status(ErrorCode::timeout, "TLS handshake with " + host + " timed out");
        }
        return st;
    };

    std::string incoming;
    bool first = true;
    bool need_read = false;
    while (true) {
        if (need_read) {
            auto got = recv_raw(socket, incoming, hs_deadline, cancel);
            if (!got.ok()) {
                return hs_error(got.status());
            }
            if (got.value() == 0) {
                return Status(ErrorCode::protocol_error, "TLS handshake with " + host + " failed: connection closed");
            }
        }
        SecBuffer in_bufs[2];
        in_bufs[0] = {static_cast<unsigned long>(incoming.size()), SECBUFFER_TOKEN,
                      incoming.empty() ? nullptr : incoming.data()};
        in_bufs[1] = {0, SECBUFFER_EMPTY, nullptr};
        SecBufferDesc in_desc{SECBUFFER_VERSION, 2, in_bufs};
        SecBuffer out_bufs[1];
        out_bufs[0] = {0, SECBUFFER_TOKEN, nullptr};
        SecBufferDesc out_desc{SECBUFFER_VERSION, 1, out_bufs};
        unsigned long attrs = 0;
        ss = InitializeSecurityContextW(&session->cred, first ? nullptr : &session->ctx,
                                        target.empty() ? nullptr : target.data(), req_flags, 0, 0,
                                        first ? nullptr : &in_desc, 0, first ? &session->ctx : nullptr, &out_desc,
                                        &attrs, nullptr);
        if (first) {
            session->has_ctx = ss == SEC_E_OK || ss == SEC_I_CONTINUE_NEEDED;
            first = false;
        }
        if (ss == SEC_E_INCOMPLETE_MESSAGE) {
            need_read = true;
            continue;
        }
        if (out_bufs[0].cbBuffer > 0 && out_bufs[0].pvBuffer != nullptr) {
            auto st = send_raw(socket, static_cast<const char*>(out_bufs[0].pvBuffer), out_bufs[0].cbBuffer,
                               hs_deadline, cancel);
            FreeContextBuffer(out_bufs[0].pvBuffer);
            if (!st.ok()) {
                return hs_error(st);
            }
        }
        if (ss == SEC_E_OK || ss == SEC_I_CONTINUE_NEEDED || ss == SEC_I_INCOMPLETE_CREDENTIALS) {
            if (in_bufs[1].BufferType == SECBUFFER_EXTRA && in_bufs[1].cbBuffer > 0) {
                incoming = incoming.substr(incoming.size() - in_bufs[1].cbBuffer);
                need_read = false;
            } else {
                incoming.clear();
                need_read = true;
            }
            if (ss == SEC_E_OK) {
                break;
            }
            if (ss == SEC_I_INCOMPLETE_CREDENTIALS) {
                need_read = false;  // no client certificate: retry with what we have
            }
            continue;
        }
        return Status(ErrorCode::protocol_error, "TLS handshake with " + host + " failed (" + hex32(ss) + ")");
    }

    // Certificate checks (manual validation was requested above).
    const CERT_CONTEXT* raw_cert = nullptr;
    ss = QueryContextAttributesW(&session->ctx, SECPKG_ATTR_REMOTE_CERT_CONTEXT, &raw_cert);
    if (ss != SEC_E_OK || raw_cert == nullptr) {
        return Status(ErrorCode::protocol_error, "TLS: server presented no certificate");
    }
    CertPtr cert(raw_cert);
    if (verify_chain) {
        if (auto st = verify_server_cert(cert.get(), name, static_cast<HCERTSTORE>(bundle.get())); !st.ok()) {
            return st;
        }
    }
    if (!pins.empty()) {
        auto got = sha256_hex(cert->pbCertEncoded, cert->cbCertEncoded);
        if (!got.ok()) {
            return got.status();
        }
        for (const auto& pin : pins) {
            if (pin != got.value()) {
                return Status(ErrorCode::protocol_error, "TLS certificate pin mismatch for " + name +
                                                             ": expected sha256 " + pin + ", got " + got.value());
            }
        }
    }

    SecPkgContext_StreamSizes sizes{};
    ss = QueryContextAttributesW(&session->ctx, SECPKG_ATTR_STREAM_SIZES, &sizes);
    if (ss != SEC_E_OK) {
        return Status(ErrorCode::internal, "TLS: cannot query stream sizes " + hex32(ss));
    }
    return std::unique_ptr<TlsStream>(
        std::make_unique<SchannelStream>(std::move(session), socket, sizes, std::move(incoming)));
}

}  // namespace sonder::inference::net
