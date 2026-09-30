#pragma once

#include <cstdint>
#include <optional>
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

// llama.cpp-only request fields (never sent to a generic OpenAI upstream).
struct RequestExtras {
    bool cache_prompt = false;             // "cache_prompt": true
    std::optional<std::uint32_t> id_slot;  // "id_slot": pinned slot
};

// GET /props: the per-slot context (default_generation_settings.n_ctx, or a
// top-level n_ctx) and the slot count (total_slots). 0 = not reported.
struct ServerProps {
    std::uint64_t n_ctx = 0;
    std::uint32_t total_slots = 0;
};

json::Value build_completion_body(const std::string &model, const GenerateRequest &request,
                                  bool native_completion, std::string_view grammar = {},
                                  const RequestExtras &extras = {});
// Also forwards request.thinking as chat_template_kwargs when set.
json::Value build_chat_body(const std::string &model, const ChatRequest &request,
                            std::string_view grammar = {}, const RequestExtras &extras = {});
Status parse_timings(const json::Value &value, Timings &out);
Status parse_props(const json::Value &value, ServerProps &out);
StopReason map_finish_reason(std::string_view reason);
// num_ctx is a server property on llama-server: a value up to the served
// per-slot context is a no-op (the engine narrows its own accounting), a
// larger one is refused naming both numbers. served_ctx 0 = unknown, and
// then num_ctx is accepted unenforced (the upstream keeps its own window).
Status validate_sampling(const SamplingConfig &sampling, std::uint64_t served_ctx = 0);

} // namespace sonder::inference::llamaserver
