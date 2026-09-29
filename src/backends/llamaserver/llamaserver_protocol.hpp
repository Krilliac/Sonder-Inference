#pragma once

#include <string>
#include <string_view>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamaserver {

struct Timings {
    bool present = false;
    bool prompt_present = false;
    bool cache_present = false;
    bool completion_present = false;
    bool draft_present = false;
    bool draft_accepted_present = false;
    bool speed_present = false;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t cached_tokens = 0;
    std::uint64_t completion_tokens = 0;
    std::uint64_t draft_tokens = 0;
    std::uint64_t draft_accepted_tokens = 0;
    double predicted_tokens_per_second = 0.0;
    std::uint64_t prompt_ns = 0;
    std::uint64_t eval_ns = 0;
};

json::Value build_completion_body(const std::string &model, const GenerateRequest &request,
                                  bool native_completion, std::string_view grammar = {});
json::Value build_chat_body(const std::string &model, const ChatRequest &request,
                            std::string_view grammar = {});
Status parse_timings(const json::Value &value, Timings &out);
StopReason map_finish_reason(std::string_view reason);
Status validate_sampling(const SamplingConfig &sampling);

} // namespace sonder::inference::llamaserver
