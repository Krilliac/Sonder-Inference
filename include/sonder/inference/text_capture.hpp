// Sonder Inference: telemetry text-capture policy.
//
// Telemetry is content-free unless the operator opts in. The policy has three
// declared values:
//
//   off   (default) no prompt or generated text in any event;
//   on    generated output text in `inference.token.generated` (and in any
//         future `inference.sampling.candidates` event) `text` fields;
//   full  `on`, plus the prompt text in `request.started`.
//
// The configured policy is declared in `engine.started`, `session.created`
// and in server discovery as `text_capture`. A request can only ever lower
// the policy: the `X-Sonder-No-Capture` header forces `off` for that request
// and nothing a request sends can raise it.
//
// Everything here is pure (no I/O, no clock) so it is unit-testable without
// an engine.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace sonder::inference {

enum class TextCapturePolicy { off = 0, on = 1, full = 2 };

// "off" | "on" | "full".
const char* to_string(TextCapturePolicy policy) noexcept;

// Exact, case-insensitive match of "off", "on" or "full" (no trimming, no
// aliases). Anything else, including the empty string, is nullopt so a typo
// can never silently enable capture or silently disable it.
std::optional<TextCapturePolicy> parse_text_capture_policy(std::string_view text) noexcept;

// Generated output text may be exported (policy `on` or `full`).
[[nodiscard]] constexpr bool captures_output_text(TextCapturePolicy policy) noexcept {
    return policy != TextCapturePolicy::off;
}
// Prompt text may be exported (policy `full` only).
[[nodiscard]] constexpr bool captures_prompt_text(TextCapturePolicy policy) noexcept {
    return policy == TextCapturePolicy::full;
}

// Combines the policy with the legacy boolean option (`capture_text` /
// `--capture-text`, which means `on`): the more permissive of the two wins,
// so a legacy flag never lowers an explicit policy.
[[nodiscard]] constexpr TextCapturePolicy combine_text_capture(TextCapturePolicy policy,
                                                                bool legacy_capture_text) noexcept {
    if (legacy_capture_text && policy == TextCapturePolicy::off) {
        return TextCapturePolicy::on;
    }
    return policy;
}

// The per-request opt-out can only force `off`.
[[nodiscard]] constexpr TextCapturePolicy apply_request_opt_out(TextCapturePolicy policy, bool opt_out) noexcept {
    return opt_out ? TextCapturePolicy::off : policy;
}

// X-Sonder-No-Capture header value: "1" or "true" (case-insensitive) opts
// the request out; "0" or "false" is an explicit no-op; anything else is
// nullopt (the server answers 400 rather than guess about a privacy header).
std::optional<bool> parse_no_capture_header(std::string_view value) noexcept;

// Plain-language startup notice for a non-`off` policy; empty for `off`.
std::string text_capture_startup_notice(TextCapturePolicy policy);

// Upper bound on prompt text copied into one `request.started` event.
inline constexpr std::size_t kMaxCapturedPromptBytes = 64u * 1024u;

// Longest prefix of `text` that is at most `max_bytes` long and does not end
// inside a UTF-8 multi-byte sequence. Never allocates.
[[nodiscard]] std::string_view truncate_utf8(std::string_view text, std::size_t max_bytes) noexcept;

}  // namespace sonder::inference
