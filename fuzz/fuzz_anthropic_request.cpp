// libFuzzer target: the Anthropic Messages front end
// (src/server/src/anthropic.cpp) and the reasoning-budget body parser
// (src/server/src/reasoning_budget.cpp).
//
// Input: the raw request body of POST /v1/messages. The same bytes are
//   * parsed by parse_messages_request();
//   * when they parse as a JSON object, passed to reasoning_budget::parse_body();
//   * cut at UTF-8 boundaries into pieces that drive AnthropicStream
//     (start/token/finish) as text and reasoning chunks, and
//     anthropic_message_response().
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary bytes.
//   * an ApiError is a 4xx with a non-empty code and message, and its
//     Anthropic error body re-parses.
//   * an accepted job passes validate_chat_messages() and validate() (the
//     checks fuzz_http_request asserts for the OpenAI route), has a non-empty
//     model, an explicit max_tokens, at most 16 stop sequences and a session
//     key of 1..256 bytes.
//   * reasoning_budget::parse_body() either sets values within -1..INT64_MAX
//     and a message of at most 512 bytes, or returns a 4xx error.
//   * every stream frame is "event: X\ndata: {json}\n\n", is valid UTF-8, its
//     payload re-parses and carries "type" == X; the frames follow the
//     message_start, (block_start, delta+, block_stop)*, message_delta,
//     message_stop grammar with consecutive block indices; when the input is
//     valid UTF-8 the concatenated deltas equal the input pieces.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "fuzz_common.hpp"
#include "server/src/anthropic.hpp"
#include "server/src/reasoning_budget.hpp"

namespace si = sonder::inference;
namespace det = sonder::inference::server::detail;

namespace {

void check_api_error(const det::ApiError& e) {
    SONDER_FUZZ_CHECK(e.status >= 400 && e.status <= 499);
    SONDER_FUZZ_CHECK(!e.code.empty());
    SONDER_FUZZ_CHECK(!e.message.empty());
    const std::string doc = si::json::Value(det::anthropic_error_body(e)).dump();
    SONDER_FUZZ_CHECK(sonder_fuzz::is_valid_utf8(doc));
    const auto reparsed = si::json::parse(doc);
    SONDER_FUZZ_CHECK(reparsed.ok());
    SONDER_FUZZ_CHECK(reparsed.value().find("type") && reparsed.value().find("type")->as_string() == "error");
}

struct Piece {
    std::string text;
    std::string reasoning;
};

// Cuts `input` into 1..29 byte-length-varied pieces at UTF-8 boundaries so a
// valid input yields valid pieces. The low bits of each piece's first byte
// choose whether it is text, reasoning, both, or empty.
std::vector<Piece> make_pieces(std::string_view input) {
    std::vector<Piece> pieces;
    std::size_t i = 0;
    while (i < input.size() && pieces.size() < 64) {
        const auto lead = static_cast<unsigned char>(input[i]);
        std::size_t n = 1 + (lead % 29u);
        std::size_t end = std::min(input.size(), i + n);
        while (end < input.size() && (static_cast<unsigned char>(input[end]) & 0xC0) == 0x80) ++end;
        const std::string chunk(input.substr(i, end - i));
        Piece p;
        switch (lead & 3u) {
            case 0: p.text = chunk; break;
            case 1: p.reasoning = chunk; break;
            case 2: p.text = chunk; p.reasoning = chunk; break;
            default: break;  // empty chunk: must produce no output
        }
        pieces.push_back(std::move(p));
        i = end;
    }
    return pieces;
}

struct Frame {
    std::string event;
    si::json::Value payload;
};

// Splits `out` into frames; every frame must be well formed.
std::vector<Frame> split_frames(const std::string& out) {
    std::vector<Frame> frames;
    std::size_t pos = 0;
    while (pos < out.size()) {
        const std::size_t end = out.find("\n\n", pos);
        SONDER_FUZZ_CHECK(end != std::string::npos);
        const std::string_view frame(out.data() + pos, end - pos);
        SONDER_FUZZ_CHECK(frame.rfind("event: ", 0) == 0);
        const std::size_t nl = frame.find('\n');
        SONDER_FUZZ_CHECK(nl != std::string_view::npos);
        const std::string_view event = frame.substr(7, nl - 7);
        const std::string_view data_line = frame.substr(nl + 1);
        SONDER_FUZZ_CHECK(data_line.rfind("data: ", 0) == 0);
        const std::string_view payload = data_line.substr(6);
        SONDER_FUZZ_CHECK(payload.find('\n') == std::string_view::npos);
        auto parsed = si::json::parse(payload);
        SONDER_FUZZ_CHECK(parsed.ok());
        SONDER_FUZZ_CHECK(parsed.value().is_object());
        const auto* type = parsed.value().find("type");
        SONDER_FUZZ_CHECK(type && type->is_string() && type->as_string() == event);
        frames.push_back({std::string(event), std::move(parsed.value())});
        pos = end + 2;
    }
    return frames;
}

void check_stream(const std::vector<Piece>& pieces, bool input_valid_utf8) {
    det::AnthropicStream stream("msg_fuzz", "fuzz-model");
    std::string all;

    // Tokens before start() are dropped.
    SONDER_FUZZ_CHECK(stream.token(si::TokenChunk{"early", 0, {}}).empty());
    const std::string started = stream.start();
    SONDER_FUZZ_CHECK(!started.empty());
    SONDER_FUZZ_CHECK(stream.start().empty());  // second start is a no-op
    all += started;
    std::string want_text;
    std::string want_reasoning;
    std::uint64_t idx = 0;
    for (const auto& p : pieces) {
        const std::string frames = stream.token(si::TokenChunk{p.text, idx++, p.reasoning});
        if (p.text.empty() && p.reasoning.empty()) SONDER_FUZZ_CHECK(frames.empty());
        all += frames;
        want_text += p.text;
        want_reasoning += p.reasoning;
    }
    si::GenerationResult result;
    result.text = want_text;
    result.reasoning = want_reasoning;
    result.stats.stop_reason = (pieces.size() % 3 == 0)   ? si::StopReason::max_tokens
                               : (pieces.size() % 3 == 1) ? si::StopReason::stop_sequence
                                                          : si::StopReason::end_of_sequence;
    if (result.stats.stop_reason == si::StopReason::stop_sequence) result.stats.matched_stop = want_text;
    const std::string finished = stream.finish(result, {want_reasoning}, {});
    SONDER_FUZZ_CHECK(!finished.empty());
    all += finished;
    SONDER_FUZZ_CHECK(stream.finish(result).empty());  // second finish is a no-op
    SONDER_FUZZ_CHECK(stream.token(si::TokenChunk{"late", 0, {}}).empty());

    SONDER_FUZZ_CHECK(sonder_fuzz::is_valid_utf8(all));
    const std::vector<Frame> frames = split_frames(all);
    SONDER_FUZZ_CHECK(frames.size() >= 5);  // start, block start/delta?/stop, message_delta, message_stop
    SONDER_FUZZ_CHECK(frames.front().event == "message_start");
    SONDER_FUZZ_CHECK(frames[frames.size() - 2].event == "message_delta");
    SONDER_FUZZ_CHECK(frames.back().event == "message_stop");

    bool open = false;
    std::uint64_t next_index = 0;
    std::string got_text;
    std::string got_reasoning;
    for (std::size_t i = 1; i + 2 < frames.size(); ++i) {
        const auto& f = frames[i];
        const auto* index = f.payload.find("index");
        SONDER_FUZZ_CHECK(index && index->is_integer());
        if (f.event == "content_block_start") {
            SONDER_FUZZ_CHECK(!open);
            SONDER_FUZZ_CHECK(index->as_uint() == next_index);
            open = true;
        } else if (f.event == "content_block_delta") {
            SONDER_FUZZ_CHECK(open);
            SONDER_FUZZ_CHECK(index->as_uint() == next_index);
            const auto* delta = f.payload.find("delta");
            SONDER_FUZZ_CHECK(delta && delta->is_object());
            const auto* dtype = delta->find("type");
            SONDER_FUZZ_CHECK(dtype && dtype->is_string());
            if (dtype->as_string() == "text_delta") {
                const auto* t = delta->find("text");
                SONDER_FUZZ_CHECK(t && t->is_string());
                got_text += t->as_string();
            } else {
                SONDER_FUZZ_CHECK(dtype->as_string() == "thinking_delta");
                const auto* t = delta->find("thinking");
                SONDER_FUZZ_CHECK(t && t->is_string());
                got_reasoning += t->as_string();
            }
        } else {
            SONDER_FUZZ_CHECK(f.event == "content_block_stop");
            SONDER_FUZZ_CHECK(open);
            SONDER_FUZZ_CHECK(index->as_uint() == next_index);
            open = false;
            ++next_index;
        }
    }
    SONDER_FUZZ_CHECK(!open);
    SONDER_FUZZ_CHECK(next_index >= 1);
    if (input_valid_utf8) {
        SONDER_FUZZ_CHECK(got_text == want_text);
        SONDER_FUZZ_CHECK(got_reasoning == want_reasoning);
    }

    // Non-streaming response for the same result.
    const std::string body = si::json::Value(det::anthropic_message_response("msg_fuzz", "fuzz-model", result)).dump();
    SONDER_FUZZ_CHECK(sonder_fuzz::is_valid_utf8(body));
    const auto reparsed = si::json::parse(body);
    SONDER_FUZZ_CHECK(reparsed.ok());
    const auto* content = reparsed.value().find("content");
    SONDER_FUZZ_CHECK(content && content->is_array() && !content->as_array().empty());
    if (input_valid_utf8) {
        const auto& last = content->as_array().back();
        SONDER_FUZZ_CHECK(last.find("text") && last.find("text")->as_string() == want_text);
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string_view input = sonder_fuzz::as_view(data, size);

    const auto job = det::parse_messages_request(input);
    det::ChatJob budget_job;
    if (const auto* ok = std::get_if<det::ChatJob>(&job)) {
        SONDER_FUZZ_CHECK(si::validate_chat_messages(ok->messages).ok());
        SONDER_FUZZ_CHECK(si::validate(ok->sampling).ok());
        SONDER_FUZZ_CHECK(!ok->model.empty());
        SONDER_FUZZ_CHECK(!ok->messages.empty());
        SONDER_FUZZ_CHECK(ok->sampling.max_tokens >= 1 && ok->sampling.max_tokens <= si::SamplingConfig::kMaxTokensLimit);
        SONDER_FUZZ_CHECK((ok->sampling.explicit_fields & si::SamplingConfig::kMaxTokens) != 0);
        SONDER_FUZZ_CHECK(ok->sampling.stop.size() <= si::SamplingConfig::kMaxStopSequences);
        if (ok->session_key) {
            SONDER_FUZZ_CHECK(!ok->session_key->empty() && ok->session_key->size() <= 256);
        }
        for (const auto& m : ok->messages) {
            SONDER_FUZZ_CHECK(m.role == "user" || m.role == "assistant" || m.role == "system");
        }
        budget_job = *ok;
    } else {
        check_api_error(std::get<det::ApiError>(job));
    }

    const auto parsed = si::json::parse(input);
    if (parsed.ok() && parsed.value().is_object()) {
        const auto err = det::reasoning_budget::parse_body(parsed.value().as_object(), budget_job);
        if (err) {
            check_api_error(*err);
        } else {
            if (budget_job.reasoning_budget_tokens) SONDER_FUZZ_CHECK(*budget_job.reasoning_budget_tokens >= -1);
            if (budget_job.reasoning_budget_message) SONDER_FUZZ_CHECK(budget_job.reasoning_budget_message->size() <= 512);
        }
    }

    // Header-style integer parser on the leading line of the input.
    const std::string_view first_line = input.substr(0, input.find('\n'));
    if (const auto n = det::reasoning_budget::parse_integer(first_line)) {
        SONDER_FUZZ_CHECK(*n >= -1);
    }

    check_stream(make_pieces(input), sonder_fuzz::is_valid_utf8(input));
    return 0;
}
