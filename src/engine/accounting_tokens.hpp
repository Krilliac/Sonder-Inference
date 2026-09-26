// Approximate "accounting" tokenizer used by the engine when a backend does
// not expose its own tokenizer (BackendModel::tokenize). It only feeds KV
// accounting and prefix matching: one token per whitespace-separated word,
// plus a leading BOS, with ids derived from an FNV-1a hash. Real subword
// tokenizers usually produce more tokens than words, so counts are estimates.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"

namespace sonder::inference::detail {

inline constexpr TokenId kAccountingBos = 1;

inline TokenId accounting_token(std::string_view word) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (const char c : word) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
    }
    return static_cast<TokenId>(h & 0x3FFFFFFFull) + 2;
}

inline bool is_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Id for a generated text chunk (whitespace trimmed).
inline TokenId accounting_chunk_token(std::string_view chunk) noexcept {
    while (!chunk.empty() && is_space(chunk.front())) chunk.remove_prefix(1);
    while (!chunk.empty() && is_space(chunk.back())) chunk.remove_suffix(1);
    return accounting_token(chunk);
}

inline std::vector<TokenId> accounting_tokenize(std::string_view text) {
    std::vector<TokenId> out;
    out.push_back(kAccountingBos);
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_space(text[i])) ++i;
        const std::size_t start = i;
        while (i < text.size() && !is_space(text[i])) ++i;
        if (i > start) out.push_back(accounting_token(text.substr(start, i - start)));
    }
    return out;
}

}  // namespace sonder::inference::detail
