// Internal: strict per-request reasoning budget parsing and default-only pins.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "openai.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail::reasoning_budget {

std::optional<std::int64_t> parse_integer(std::string_view text) noexcept;
std::optional<ApiError> parse_body(const json::Object& body, ChatJob& job);
std::optional<ApiError> parse_header(const RequestHead& head, Correlation& correlation);
void apply(const ChatJob& job, const Correlation& correlation, const ServerOptions& pins,
           std::string_view backend, RequestOptions& request, std::vector<std::string>& warnings);

}  // namespace sonder::inference::server::detail::reasoning_budget
