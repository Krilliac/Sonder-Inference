#pragma once

#include <cstddef>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::worker {

inline constexpr int kProtocolVersion = 1;
inline constexpr std::size_t kMaxChatMessages = 1024;
inline constexpr std::uint64_t kMaxTokenPayloadBytes = 64u * 1024u;
inline constexpr std::uint64_t kMaxRequestOutputBytes = 64u * 1024u * 1024u;
inline constexpr std::uint64_t kMaxModelBytes = std::uint64_t{1} << 40;
inline constexpr std::uint64_t kMaxWorkerTimingNs = 86400ull * 1000000000ull;
// All returned values own their memory. Token views must be consumed before
// the parsed message is destroyed. Helpers are stateless and thread-safe.
json::Object message(std::string_view type, std::uint64_t id);
Result<json::Value> parse_message(std::string_view frame);
Status fields(const json::Value &value, std::initializer_list<std::string_view> required);
Result<std::uint64_t> unsigned_field(const json::Value &, std::string_view key);
Result<std::string> string_field(const json::Value &, std::string_view key);
json::Value sampling_to_json(const SamplingConfig &);
Result<SamplingConfig> sampling_from_json(const json::Value &);
json::Value descriptor_to_json(const ModelDescriptor &);
Result<ModelDescriptor> descriptor_from_json(const json::Value &);
json::Value stats_to_json(const GenerateStats &);
Result<GenerateStats> stats_from_json(const json::Value &);
json::Value chunk_to_json(const TokenChunk &);
Result<TokenChunk> chunk_from_json(const json::Value &);
Result<json::Value> generate_to_json(const GenerateRequest &);
Result<GenerateRequest> generate_from_json(const json::Value &);
Result<json::Value> chat_to_json(const ChatRequest &);
Result<ChatRequest> chat_from_json(const json::Value &);
json::Value error_message(std::uint64_t id, const Status &);
Status status_from_message(const json::Value &);

} // namespace sonder::inference::worker
