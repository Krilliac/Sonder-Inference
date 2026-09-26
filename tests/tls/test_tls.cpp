// TLS transport tests: a self-signed PKI is generated at test time and served
// from 127.0.0.1 inside the test process. No external host is contacted.
#include <doctest/doctest.h>

#include <chrono>
#include <string>

#include "net/http_client.hpp"
#include "net/tls_stream.hpp"
#include "sonder/inference/error.hpp"
#include "tls_test_support.hpp"

#if defined(SONDER_HAS_OLLAMA_BACKEND)
#  include "sonder/inference/backends/ollama.hpp"
#endif

using namespace sonder::inference;

namespace {

net::HttpRequest tls_get(std::uint16_t port, const std::string& target = "/api/version") {
    net::HttpRequest r;
    r.method = "GET";
    r.host = "127.0.0.1";
    r.port = port;
    r.target = target;
    r.use_tls = true;
    r.connect_timeout = std::chrono::milliseconds(2000);
    r.total_timeout = std::chrono::milliseconds(10000);
    r.tls.handshake_timeout = std::chrono::milliseconds(5000);
    return r;
}

Result<net::HttpResponseInfo> fetch(const net::HttpRequest& r, std::string& body) {
    return net::http_request_buffered(r, body);
}

bool contains(const std::string& haystack, const char* needle) { return haystack.find(needle) != std::string::npos; }

}  // namespace

TEST_SUITE("tls") {

TEST_CASE("backend and https URL parsing") {
    CHECK(net::kTlsCompiledIn);
    const std::string backend = net::tls_backend_name();
    CHECK((backend == "openssl" || backend == "schannel"));
    auto u = net::parse_url("https://10.77.0.2:8443");  // parse only; never connected
    REQUIRE(u.ok());
    CHECK(u.value().scheme == "https");
    CHECK(u.value().host == "10.77.0.2");
    CHECK(u.value().port == 8443);
    auto d = net::parse_url("HTTPS://node1.invalid/base/");
    REQUIRE(d.ok());
    CHECK(d.value().port == 443);
    CHECK(d.value().path == "/base");
    CHECK(net::parse_url("ftp://127.0.0.1").status().code() == ErrorCode::unsupported);
}

TEST_CASE("fingerprint normalization") {
    const std::string hex(64, 'a');
    auto a = net::normalize_sha256_fingerprint(hex);
    REQUIRE(a.ok());
    CHECK(a.value() == hex);
    std::string colon;
    for (int i = 0; i < 32; ++i) {
        colon += (i ? ":AB" : "AB");
    }
    auto b = net::normalize_sha256_fingerprint(colon);
    REQUIRE(b.ok());
    CHECK(b.value().size() == 64);
    CHECK(b.value().substr(0, 4) == "abab");
    CHECK(net::normalize_sha256_fingerprint("abcd").status().code() == ErrorCode::invalid_argument);
    CHECK(net::normalize_sha256_fingerprint(std::string(63, 'a') + "z").status().code() ==
          ErrorCode::invalid_argument);
    CHECK(net::is_ip_literal("127.0.0.1"));
    CHECK(net::is_ip_literal("::1"));
    CHECK_FALSE(net::is_ip_literal("localhost"));
}

TEST_CASE("trusted CA succeeds") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    auto req = tls_get(server.port());
    req.tls.ca_bundle_path = pki.ca;
    std::string body;
    auto res = fetch(req, body);
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    CHECK(res.value().status == 200);
    CHECK(body == R"({"version":"0.0.0-tls-test"})");
    CHECK(server.requests_served() == 1);

    // POST with a body round-trips too (exercises the TLS write path).
    auto post = tls_get(server.port(), "/api/generate");
    post.method = "POST";
    post.body = std::string(100000, 'x');
    post.tls.ca_bundle_path = pki.ca;
    std::string body2;
    auto res2 = fetch(post, body2);
    REQUIRE_MESSAGE(res2.ok(), res2.status().to_string());
    CHECK(contains(body2, "POST /api/generate"));
}

TEST_CASE("untrusted server fails against the system trust store") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    std::string body;
    auto res = fetch(tls_get(server.port()), body);
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::protocol_error);
    CHECK(contains(res.status().message(), "TLS"));
    CHECK(server.requests_served() == 0);
}

TEST_CASE("unrelated CA bundle fails") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    auto req = tls_get(server.port());
    req.tls.ca_bundle_path = pki.other_ca;
    std::string body;
    auto res = fetch(req, body);
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::protocol_error);
    CHECK(server.requests_served() == 0);
}

TEST_CASE("host name mismatch fails but a server_name override succeeds") {
    tls_test::Pki pki;
    tls_test::Server server(pki.wrong_cert, pki.wrong_key);  // cert is for node1.invalid only
    auto req = tls_get(server.port());
    req.tls.ca_bundle_path = pki.ca;
    std::string body;
    auto res = fetch(req, body);
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::protocol_error);

    req.tls.server_name = "node1.invalid";  // SNI + verification name; still dials 127.0.0.1
    auto ok = fetch(req, body);
    REQUIRE_MESSAGE(ok.ok(), ok.status().to_string());
    CHECK(ok.value().status == 200);
}

TEST_CASE("pin mismatch fails even with a trusted CA") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    auto req = tls_get(server.port());
    req.tls.ca_bundle_path = pki.ca;
    req.tls.pinned_sha256 = pki.other_fingerprint;
    std::string body;
    auto res = fetch(req, body);
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::protocol_error);
    CHECK(contains(res.status().message(), "pin mismatch"));
    CHECK(server.requests_served() == 0);

    // Pin mismatch also fails in pin-only mode and with insecure_skip_verify.
    req.tls.ca_bundle_path.clear();
    CHECK(fetch(req, body).status().code() == ErrorCode::protocol_error);
    req.tls.insecure_skip_verify = true;
    CHECK(fetch(req, body).status().code() == ErrorCode::protocol_error);
    req.tls.insecure_skip_verify = false;

    // A pinned certificate file that differs from the served one fails.
    req.tls.pinned_sha256.clear();
    req.tls.pinned_cert_path = pki.wrong_cert;
    CHECK(fetch(req, body).status().code() == ErrorCode::protocol_error);
    CHECK(server.requests_served() == 0);
}

TEST_CASE("matching pin succeeds with CA and pin-only and as a pinned cert file") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    std::string body;

    auto with_ca = tls_get(server.port());
    with_ca.tls.ca_bundle_path = pki.ca;
    with_ca.tls.pinned_sha256 = pki.server_fingerprint;
    auto r1 = fetch(with_ca, body);
    REQUIRE_MESSAGE(r1.ok(), r1.status().to_string());

    auto pin_only = tls_get(server.port());  // self-signed style: no CA, pin only
    pin_only.tls.pinned_sha256 = pki.server_fingerprint;
    auto r2 = fetch(pin_only, body);
    REQUIRE_MESSAGE(r2.ok(), r2.status().to_string());

    auto cert_file = tls_get(server.port());
    cert_file.tls.pinned_cert_path = pki.server_cert;
    auto r3 = fetch(cert_file, body);
    REQUIRE_MESSAGE(r3.ok(), r3.status().to_string());
    CHECK(server.requests_served() == 3);
}

TEST_CASE("insecure_skip_verify connects to an untrusted server") {
    tls_test::Pki pki;
    tls_test::Server server(pki.wrong_cert, pki.wrong_key);  // untrusted chain AND wrong name
    auto req = tls_get(server.port());
    req.tls.insecure_skip_verify = true;  // prints a loud warning on stderr
    std::string body;
    auto res = fetch(req, body);
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    CHECK(res.value().status == 200);
}

TEST_CASE("bad TLS options are rejected before any request") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    std::string body;
    auto missing_ca = tls_get(server.port());
    missing_ca.tls.ca_bundle_path = (pki.dir / "does-not-exist.pem").string();
    CHECK(fetch(missing_ca, body).status().code() == ErrorCode::invalid_argument);

    auto bad_pin = tls_get(server.port());
    bad_pin.tls.pinned_sha256 = "not-a-fingerprint";
    CHECK(fetch(bad_pin, body).status().code() == ErrorCode::invalid_argument);

    auto bad_pin_file = tls_get(server.port());
    bad_pin_file.tls.pinned_cert_path = pki.server_key;  // a key, not a certificate
    CHECK(fetch(bad_pin_file, body).status().code() == ErrorCode::invalid_argument);
    CHECK(server.requests_served() == 0);
}

TEST_CASE("handshake timeout against a silent peer") {
    tls_test::Server silent{tls_test::Server::Silent{}};
    auto req = tls_get(silent.port());
    req.tls.insecure_skip_verify = true;
    req.tls.handshake_timeout = std::chrono::milliseconds(300);
    std::string body;
    const auto t0 = std::chrono::steady_clock::now();
    auto res = fetch(req, body);
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::timeout);
    CHECK(elapsed < std::chrono::seconds(5));
}

TEST_CASE("cancellation during handshake") {
    tls_test::Server silent{tls_test::Server::Silent{}};
    auto req = tls_get(silent.port());
    req.tls.insecure_skip_verify = true;
    CancellationSource source;
    source.cancel();
    std::string body;
    auto res = net::http_request_buffered(req, body, source.token());
    REQUIRE_FALSE(res.ok());
    CHECK(res.status().code() == ErrorCode::cancelled);
}

#if defined(SONDER_HAS_OLLAMA_BACKEND)
TEST_CASE("ollama client speaks https to a loopback TLS server") {
    tls_test::Pki pki;
    tls_test::Server server(pki.server_cert, pki.server_key);
    ollama::OllamaConfig cfg;
    cfg.base_url = "https://127.0.0.1:" + std::to_string(server.port());
    cfg.connect_timeout = std::chrono::milliseconds(2000);
    cfg.request_timeout = std::chrono::milliseconds(10000);
    cfg.tls.ca_bundle_path = pki.ca;
    ollama::OllamaClient trusted(cfg);
    auto v = trusted.version();
    REQUIRE_MESSAGE(v.ok(), v.status().to_string());
    CHECK(v.value() == "0.0.0-tls-test");

    cfg.tls.ca_bundle_path.clear();  // untrusted now
    ollama::OllamaClient untrusted(cfg);
    CHECK(untrusted.version().status().code() == ErrorCode::protocol_error);

    cfg.tls.pinned_sha256 = pki.other_fingerprint;
    ollama::OllamaClient wrong_pin(cfg);
    CHECK(wrong_pin.version().status().code() == ErrorCode::protocol_error);
    CHECK(server.requests_served() == 1);
}

TEST_CASE("ollama https to a non-loopback host still needs allow_remote") {
    ollama::OllamaConfig cfg;
    cfg.base_url = "https://10.77.0.2:8443";  // refused before any connection attempt
    cfg.tls.ca_bundle_path = "unused.pem";
    ollama::OllamaClient c(cfg);
    CHECK(c.version().status().code() == ErrorCode::invalid_argument);
}
#endif

}  // TEST_SUITE
