// Internal (src/models): the one sanitizer for untrusted text that the model store puts in
// messages, notes and the import-ollama table (model_store.cpp, models_cli.cpp).
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace sonder::inference::models::store_text {

// A bounded, single-line, inert form of untrusted bytes. Escaped as \xNN, byte by byte:
// C0 controls, DEL, every byte that is not part of a well-formed UTF-8 sequence (raw
// 0x80-0xFF continuation or lead bytes, overlong forms, surrogates, > U+10FFFF, a sequence
// cut by `max`), and the bytes of well-formed sequences for C1 controls (U+0080-U+009F),
// Arabic letter mark, zero-width and bidi marks (U+200B-U+200F, U+202A-U+202E,
// U+2060-U+2069), line/paragraph separators (U+2028, U+2029), BOM (U+FEFF) and interlinear
// annotation controls (U+FFF9-U+FFFB). A backslash is doubled (\\), so input text cannot
// forge an escape. Reads at most `max` input bytes and appends "..." when it cut.
// Reentrant; allocates the output only.
std::string printable(std::string_view s, std::size_t max);

}  // namespace sonder::inference::models::store_text
