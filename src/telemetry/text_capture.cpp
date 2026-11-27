#include "sonder/inference/text_capture.hpp"

namespace sonder::inference {

namespace {

bool equals_ci(std::string_view text, std::string_view lower_word) noexcept {
    if (text.size() != lower_word.size()) {
        return false;
    }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
        if (c != lower_word[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

const char* to_string(TextCapturePolicy policy) noexcept {
    switch (policy) {
        case TextCapturePolicy::off: return "off";
        case TextCapturePolicy::on: return "on";
        case TextCapturePolicy::full: return "full";
    }
    return "off";
}

std::optional<TextCapturePolicy> parse_text_capture_policy(std::string_view text) noexcept {
    if (equals_ci(text, "off")) return TextCapturePolicy::off;
    if (equals_ci(text, "on")) return TextCapturePolicy::on;
    if (equals_ci(text, "full")) return TextCapturePolicy::full;
    return std::nullopt;
}

std::optional<bool> parse_no_capture_header(std::string_view value) noexcept {
    if (value == "1" || equals_ci(value, "true")) return true;
    if (value == "0" || equals_ci(value, "false")) return false;
    return std::nullopt;
}

std::string text_capture_startup_notice(TextCapturePolicy policy) {
    switch (policy) {
        case TextCapturePolicy::off:
            return {};
        case TextCapturePolicy::on:
            return "text capture is ON: generated output text is exported in telemetry events "
                   "(inference.token.generated); prompts are not. Send X-Sonder-No-Capture: 1 to opt a request out";
        case TextCapturePolicy::full:
            return "text capture is FULL: prompt text (request.started) and generated output text "
                   "(inference.token.generated) are exported in telemetry events. "
                   "Send X-Sonder-No-Capture: 1 to opt a request out";
    }
    return {};
}

std::string_view truncate_utf8(std::string_view text, std::size_t max_bytes) noexcept {
    if (text.size() <= max_bytes) {
        return text;
    }
    std::size_t cut = max_bytes;
    // Back up over continuation bytes (10xxxxxx) so a sequence is never split;
    // at most three, because UTF-8 sequences are at most four bytes. Then, if
    // the byte at `cut` starts a sequence, `cut` is already a boundary.
    std::size_t back = 0;
    while (cut > 0 && back < 3 && (static_cast<unsigned char>(text[cut]) & 0xC0u) == 0x80u) {
        --cut;
        ++back;
    }
    return text.substr(0, cut);
}

}  // namespace sonder::inference
