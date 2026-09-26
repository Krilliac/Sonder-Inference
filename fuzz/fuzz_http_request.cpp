// libFuzzer target: the `sonder-infer serve` request front end
// (src/server/src/http.cpp and openai.cpp).
//
// Input: raw bytes as a client would send them. The request head parser runs
// on the whole input; when it completes, the bytes after the head are the
// body, fed to the OpenAI chat request mapper, and the query and headers go
// through the query decoder, the Host check and the correlation parser.
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary bytes.
//   * a complete head consumed at most the input and at most 16 KiB, has at
//     most 64 headers, an origin-form path, and lowercase header names.
//   * errors carry an HTTP status in {400, 431, 505} and a message.
//   * an accepted chat request has valid messages and valid sampling (it
//     passed validate_chat_messages() and validate()); a rejected one is a
//     400 with a non-empty code and message.
//   * correlation values that are accepted match [A-Za-z0-9._:-]{1,128}.
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "fuzz_common.hpp"
#include "server/src/http.hpp"
#include "server/src/openai.hpp"

namespace si = sonder::inference;
namespace det = sonder::inference::server::detail;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view input = sonder_fuzz::as_view(data, size);
    det::RequestHead head;
    const det::ParseResult r = det::parse_request_head(input, head);
    if (r.state == det::ParseState::error) {
        SONDER_FUZZ_CHECK(r.status == 400 || r.status == 431 || r.status == 505);
        SONDER_FUZZ_CHECK(!r.message.empty());
        return 0;
    }
    if (r.state == det::ParseState::incomplete) {
        SONDER_FUZZ_CHECK(input.size() <= det::kMaxHeadBytes);
        return 0;
    }
    SONDER_FUZZ_CHECK(r.head_bytes <= input.size());
    SONDER_FUZZ_CHECK(r.head_bytes <= det::kMaxHeadBytes);
    SONDER_FUZZ_CHECK(head.headers.size() <= det::kMaxHeaders);
    SONDER_FUZZ_CHECK(!head.path.empty() && head.path.front() == '/');
    for (const auto& [name, value] : head.headers) {
        SONDER_FUZZ_CHECK(!name.empty());
        for (const char c : name) {
            SONDER_FUZZ_CHECK(std::tolower(static_cast<unsigned char>(c)) == static_cast<unsigned char>(c));
        }
        (void)value;
    }
    SONDER_FUZZ_CHECK(!(head.content_length && head.has_transfer_encoding));

    (void)det::query_param(head.query, "model");
    (void)det::query_param(head.query, "last_event_id");
    if (const std::string* host = head.header("host")) {
        (void)det::is_loopback_host_header(*host);
    }
    const auto corr = det::parse_correlation(head);
    if (const auto* c = std::get_if<det::Correlation>(&corr)) {
        for (const auto* v : {&c->run_id, &c->parent_request_id, &c->agent_id, &c->task_id}) {
            if (*v) {
                SONDER_FUZZ_CHECK(det::is_correlation_value(**v));
            }
        }
        SONDER_FUZZ_CHECK(c->priority >= -16 && c->priority <= 16);
    } else {
        SONDER_FUZZ_CHECK(std::get<det::ApiError>(corr).code == "invalid_correlation_header");
    }

    const std::string_view body = input.substr(r.head_bytes);
    const auto job = det::parse_chat_request(body);
    if (const auto* ok = std::get_if<det::ChatJob>(&job)) {
        SONDER_FUZZ_CHECK(si::validate_chat_messages(ok->messages).ok());
        SONDER_FUZZ_CHECK(si::validate(ok->sampling).ok());
        SONDER_FUZZ_CHECK(!ok->model.empty());
    } else {
        const auto& e = std::get<det::ApiError>(job);
        SONDER_FUZZ_CHECK(e.status == 400);
        SONDER_FUZZ_CHECK(!e.code.empty());
        SONDER_FUZZ_CHECK(!e.message.empty());
        const std::string doc = si::json::Value(det::error_body(e)).dump();
        SONDER_FUZZ_CHECK(si::json::parse(doc).ok());
    }
    return 0;
}
