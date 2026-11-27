// Internal helpers of the llama.cpp module adapter (not installed; exposed
// for unit tests only).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference::llamacpp_detail {

// UTF-8 std::string <-> std::filesystem::path without the C++20-deprecated
// u8path() and without std::u8string leaking into callers.
inline std::filesystem::path PathFromUtf8(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}
inline std::string PathToUtf8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// Maps SamplingConfig onto the wrapper's sampler parameters.
inline sonder::backends::llamacpp::SamplingParams ToLlamaSampling(const SamplingConfig& c) {
    namespace lc = sonder::backends::llamacpp;
    lc::SamplingParams p;
    p.temperature = c.temperature;
    p.top_k = c.top_k;
    p.top_p = c.top_p;
    p.min_p = c.min_p;
    // When explicit_only is set (HTTP OpenAI requests), leave repeat_penalty disabled (1.0)
    // unless the caller explicitly asked for it — matching OpenAI / llama-server defaults
    // and allowing pure GPU backend sampling on greedy/top-p requests.
    p.repeat_penalty = (c.explicit_only && !c.is_explicit(SamplingConfig::kRepeatPenalty)) ? 1.0F : c.repeat_penalty;
    p.repeat_last_n = c.repeat_last_n;
    p.presence_penalty = c.presence_penalty;
    p.frequency_penalty = c.frequency_penalty;
    p.typical_p = c.typical_p;
    p.logit_bias.reserve(c.logit_bias.size());
    for (const auto& b : c.logit_bias) p.logit_bias.emplace_back(b.token, b.bias);
    p.compute_logprobs = c.logprobs;
    p.top_logprobs = c.top_logprobs;
    if (c.seed.has_value()) {
        // llama.cpp seeds are 32-bit and 0xFFFFFFFF means "random": fold the
        // 64-bit seed and avoid the sentinel so explicit seeds stay reproducible.
        auto folded = static_cast<std::uint32_t>(*c.seed ^ (*c.seed >> 32));
        p.seed = folded == 0xFFFFFFFFU ? 0xFFFFFFFEU : folded;
    } else {
        p.seed = 0xFFFFFFFFU;
    }
    return p;
}

// Holds back text that could be the start of a stop sequence and reports a
// full match. Stop text itself is never emitted (matches Ollama/OpenAI).
class StopSequenceFilter {
public:
    explicit StopSequenceFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    // Appends text; returns the text now safe to emit. Sets matched() on a hit.
    std::string Push(std::string_view text) {
        if (matched_) return {};
        held_.append(text);
        if (stops_.empty()) return std::exchange(held_, {});
        std::size_t cut = std::string::npos;
        const std::string* matched = nullptr;
        for (const auto& s : stops_) {
            if (s.empty()) continue;
            const std::size_t at = held_.find(s);
            if (at != std::string::npos &&
                (at < cut || (at == cut && matched && s.size() > matched->size()))) {
                cut = at;
                matched = &s;
            }
        }
        if (cut != std::string::npos) {
            matched_ = true;
            matched_index_ = matched ? std::optional<std::string>(*matched) : std::nullopt;
            std::string out = held_.substr(0, cut);
            held_.clear();
            return out;
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        std::size_t keep = 0;
        for (const auto& s : stops_) {
            if (s.empty()) continue;
            const std::size_t limit = std::min(held_.size(), s.size() - 1);
            for (std::size_t k = limit; k > keep; --k) {
                if (std::string_view(held_).substr(held_.size() - k) == std::string_view(s).substr(0, k)) {
                    keep = k;
                    break;
                }
            }
        }
        std::string out = held_.substr(0, held_.size() - keep);
        held_.erase(0, held_.size() - keep);
        return out;
    }
    std::string Flush() { return matched_ ? std::string() : std::exchange(held_, {}); }
    [[nodiscard]] const std::string& held() const noexcept { return held_; }
    void RestoreState(std::string held, bool matched = false) {
        held_ = std::move(held);
        matched_ = matched;
    }
    [[nodiscard]] bool matched() const noexcept { return matched_; }
    [[nodiscard]] const std::optional<std::string>& matched_stop() const noexcept { return matched_index_; }

private:
    std::vector<std::string> stops_;
    std::string held_;
    bool matched_ = false;
    std::optional<std::string> matched_index_;
};

// Result of one incremental step of ThinkingStreamSplitter.
struct ThinkingSplitResult {
    std::string text;
    std::string reasoning;
};

// Streaming splitter that routes text inside <think>...</think> tags to
// `reasoning` and visible output outside tags to `text`. Safely buffers
// partial tag prefixes across arbitrary token chunk boundaries.
class ThinkingStreamSplitter {
public:
    ThinkingStreamSplitter() = default;
    // `start_in_thinking` begins inside an already-open <think> block (the rendered
    // generation prompt ended with an unclosed <think>).
    explicit ThinkingStreamSplitter(bool start_in_thinking) noexcept : in_thinking_(start_in_thinking) {}

    ThinkingSplitResult Push(std::string_view chunk) {
        ThinkingSplitResult out;
        pending_.append(chunk);
        while (!pending_.empty()) {
            if (!in_thinking_) {
                const std::size_t pos = pending_.find(kOpenTag);
                if (pos != std::string::npos) {
                    out.text.append(pending_, 0, pos);
                    pending_.erase(0, pos + kOpenTag.size());
                    in_thinking_ = true;
                    continue;
                }
                const std::size_t keep = TrailingPrefixLength(pending_, kOpenTag);
                const std::size_t emit = pending_.size() - keep;
                if (emit > 0) {
                    out.text.append(pending_, 0, emit);
                    pending_.erase(0, emit);
                }
                break;
            }
            const std::size_t pos = pending_.find(kCloseTag);
            if (pos != std::string::npos) {
                out.reasoning.append(pending_, 0, pos);
                pending_.erase(0, pos + kCloseTag.size());
                in_thinking_ = false;
                continue;
            }
            const std::size_t keep = TrailingPrefixLength(pending_, kCloseTag);
            const std::size_t emit = pending_.size() - keep;
            if (emit > 0) {
                out.reasoning.append(pending_, 0, emit);
                pending_.erase(0, emit);
            }
            break;
        }
        return out;
    }

    ThinkingSplitResult Flush() {
        ThinkingSplitResult out;
        if (!pending_.empty()) {
            if (in_thinking_) {
                out.reasoning = std::exchange(pending_, {});
            } else {
                out.text = std::exchange(pending_, {});
            }
        }
        return out;
    }

    [[nodiscard]] bool in_thinking() const noexcept { return in_thinking_; }
    [[nodiscard]] std::size_t pending() const noexcept { return pending_.size(); }

private:
    static constexpr std::string_view kOpenTag = "<think>";
    static constexpr std::string_view kCloseTag = "</think>";

    static std::size_t TrailingPrefixLength(std::string_view s, std::string_view tag) noexcept {
        const std::size_t limit = std::min(s.size(), tag.size() - 1);
        for (std::size_t k = limit; k > 0; --k) {
            if (s.substr(s.size() - k) == tag.substr(0, k)) return k;
        }
        return 0;
    }

    std::string pending_;
    bool in_thinking_ = false;
};

// True when the trailing trusted (parse_special) run of the prompt ends inside an
// unclosed <think> block, so the model's first output is reasoning and the
// splitter must start in thinking mode. On the legacy llama_chat_apply_template
// path only a template whose generation prompt ends in an open <think> (the
// built-in bailing-think renderer) triggers this. The ChatML renderer (Qwen3.5)
// ends in `<|im_start|>assistant\n` and the DeepSeek-3 renderer (R1-Distill) ends
// in the Assistant token, neither writes <think>, so a bare leading </think> from
// those models still leaks into content until Jinja/common_chat rendering lands.
inline bool PromptEndsInsideThink(const std::vector<ChatPromptSegment>& segments) {
    std::string trailing;
    std::size_t idx = segments.size();
    while (idx > 0 && segments[idx - 1].parse_special) {
        --idx;
    }
    for (std::size_t i = idx; i < segments.size(); ++i) {
        trailing.append(segments[i].text);
    }
    static constexpr std::string_view kOpen = "<think>";
    static constexpr std::string_view kClose = "</think>";
    const std::size_t open = trailing.rfind(kOpen);
    if (open == std::string::npos) return false;
    return trailing.find(kClose, open) == std::string::npos;
}

// When thinking is explicitly disabled on Qwen3/ChatML-style templates ending
// in `<|im_start|>assistant\n` (and not already containing `<think>`), appends
// `<think>\n\n</think>\n\n` as a parse_special=true scaffolding segment so the
// model skips its reasoning block cleanly (matching Prism's reasoning = off).
inline bool ApplyNoThinkScaffolding(std::vector<ChatPromptSegment>& segments, bool enable_thinking) {
    if (enable_thinking || segments.empty()) return false;
    static constexpr std::string_view kAssistantCue = "<|im_start|>assistant\n";
    std::string trailing;
    std::size_t idx = segments.size();
    while (idx > 0 && segments[idx - 1].parse_special) {
        --idx;
    }
    for (std::size_t i = idx; i < segments.size(); ++i) {
        trailing.append(segments[i].text);
    }
    if (trailing.size() < kAssistantCue.size()) return false;
    if (std::string_view(trailing).substr(trailing.size() - kAssistantCue.size()) != kAssistantCue) {
        return false;
    }
    if (trailing.find("<think>") != std::string::npos) return false;
    segments.push_back(ChatPromptSegment{"<think>\n\n</think>\n\n", /*parse_special=*/true});
    return true;
}

// Serializes an assistant tool call as `<tool_call>{...}</tool_call>`, preserving
// optional call id and safely encoding non-JSON arguments instead of corrupting
// template structure.
inline std::string FormatLlamaCppToolCallMarkup(const ToolCall& tc) {
    json::Object obj;
    if (!tc.id.empty()) {
        obj.set("id", tc.id);
    }
    obj.set("name", tc.name);
    json::Value args_val = json::Object{};
    if (!tc.arguments_json.empty()) {
        auto parsed = json::parse(tc.arguments_json);
        if (parsed.ok() && (parsed.value().is_object() || parsed.value().is_array())) {
            args_val = std::move(parsed).value();
        } else {
            args_val = tc.arguments_json;
        }
    }
    obj.set("arguments", std::move(args_val));
    return "<tool_call>" + json::Value(std::move(obj)).dump() + "</tool_call>";
}

// Formats a `tool` role message so `tool_call_id` and `name` correlation survive
// llama_chat_apply_template() (which only accepts `(role, content)`). Plain tool
// turns without correlation metadata keep raw content unless `wrap_plain_content`
// is set (e.g. when coercing `tool` -> `user` for templates that drop `tool`).
inline std::string FormatLlamaCppToolResultContent(const ChatMessage& m, bool wrap_plain_content = false) {
    const bool has_call_id = m.tool_call_id.has_value() && !m.tool_call_id->empty();
    const bool has_name = m.name.has_value() && !m.name->empty();
    if (!has_call_id && !has_name && !wrap_plain_content) {
        return m.content;
    }
    json::Object obj;
    if (has_call_id) {
        obj.set("tool_call_id", *m.tool_call_id);
    }
    if (has_name) {
        obj.set("name", *m.name);
    }
    obj.set("content", m.content);
    return "<tool_response>" + json::Value(std::move(obj)).dump() + "</tool_response>";
}

// Normalizes assistant `tool_calls` and `tool` result correlation fields into
// G0028-safe `ChatMessage::content` text before template rendering. When
// `coerce_tool_role_to_user` is true, `role == "tool"` messages are mapped to
// `"user"` `<tool_response>` turns for templates whose llama_chat_apply_template()
// branch skips `tool` roles.
inline std::vector<ChatMessage> NormalizeChatMessagesForLlamaCpp(const std::vector<ChatMessage>& messages,
                                                                 bool coerce_tool_role_to_user = false) {
    std::vector<ChatMessage> out = messages;
    for (auto& m : out) {
        if (m.role == "assistant" && !m.tool_calls.empty()) {
            std::string tool_text;
            for (const auto& tc : m.tool_calls) {
                if (!tool_text.empty()) tool_text.push_back('\n');
                tool_text += FormatLlamaCppToolCallMarkup(tc);
            }
            if (!m.content.empty()) {
                m.content += "\n" + tool_text;
            } else {
                m.content = std::move(tool_text);
            }
        } else if (m.role == "tool") {
            m.content = FormatLlamaCppToolResultContent(m, coerce_tool_role_to_user);
            if (coerce_tool_role_to_user) {
                m.role = "user";
            }
        }
    }
    return out;
}

inline std::string LowerAscii(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            out.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

inline GpuVendor InferGgmlDeviceVendor(const sonder::backends::llamacpp::DeviceInfo& d) {
    const std::string name = LowerAscii(d.name);
    const std::string desc = LowerAscii(d.description);
    if (name.rfind("cuda", 0) == 0 || desc.find("nvidia") != std::string::npos ||
        desc.find("geforce") != std::string::npos || desc.find("rtx") != std::string::npos ||
        desc.find("quadro") != std::string::npos || desc.find("tesla") != std::string::npos) {
        return GpuVendor::nvidia;
    }
    if (name.rfind("rocm", 0) == 0 || name.rfind("hip", 0) == 0 || desc.find("amd") != std::string::npos ||
        desc.find("radeon") != std::string::npos) {
        return GpuVendor::amd;
    }
    if (name.rfind("sycl", 0) == 0 || desc.find("intel") != std::string::npos ||
        desc.find("arc") != std::string::npos) {
        return GpuVendor::intel;
    }
    if (name.rfind("metal", 0) == 0 || desc.find("apple") != std::string::npos) {
        return GpuVendor::apple;
    }
    return GpuVendor::unknown;
}

// Pins `gpu:<index>` requests onto the matching discrete llama.cpp GPU device
// when the caller did not pass explicit ggml device names (prevents silent
// fallback to GPU 0, unintended layer splitting across discrete + integrated
// GPUs, duplicate CUDA0/Vulkan0 counting, or cross-vendor ordinal mismatch).
// Integrated GPUs are never counted by `gpu:<index>`; use an explicit `devices`
// allowlist to place a model onto an integrated GPU.
inline Status ResolveGpuDevicePin(std::string_view device_id,
                                  const std::vector<sonder::backends::llamacpp::DeviceInfo>& available_devices,
                                  std::vector<std::string>& out_devices, std::int32_t& out_main_gpu,
                                  const std::vector<float>& tensor_split = {},
                                  const std::vector<GpuDeviceInfo>& os_gpus = {}) {
    namespace lc = sonder::backends::llamacpp;
    if (!out_devices.empty()) return Status::success();
    if (device_id.rfind("gpu:", 0) != 0) {
        if (!tensor_split.empty() && device_id.rfind("gpu", 0) == 0) {
            lc::DeviceSelection auto_sel;
            if (!lc::SelectDevices(available_devices, {}, tensor_split, 0, &auto_sel).ok() || auto_sel.names.empty()) {
                return Status(ErrorCode::invalid_argument,
                              "tensor_split requires at least one discrete GPU when no device allowlist is set");
            }
        }
        return Status::success();
    }
    const std::string_view suffix = device_id.substr(4);
    if (suffix.empty()) {
        return Status(ErrorCode::invalid_argument, "invalid gpu device id: " + std::string(device_id));
    }
    std::size_t idx = 0;
    for (char ch : suffix) {
        if (ch < '0' || ch > '9') {
            return Status(ErrorCode::invalid_argument, "invalid gpu device index in: " + std::string(device_id));
        }
        if (idx > 1000) {
            return Status(ErrorCode::not_found, "gpu device index out of range: " + std::string(device_id));
        }
        idx = idx * 10 + static_cast<std::size_t>(ch - '0');
    }

    // Auto-select de-duplicated discrete GPUs only (never integrated GPUs, and
    // collapsing CUDA0 + Vulkan0 views of the same physical card).
    lc::DeviceSelection auto_sel;
    (void)lc::SelectDevices(available_devices, {}, &auto_sel);
    std::vector<const lc::DeviceInfo*> gpus;
    gpus.reserve(auto_sel.names.size());
    for (const std::string& name : auto_sel.names) {
        const auto it = std::find_if(available_devices.begin(), available_devices.end(),
                                     [&](const lc::DeviceInfo& d) { return d.name == name; });
        if (it != available_devices.end()) gpus.push_back(&*it);
    }

    if (!tensor_split.empty()) {
        if (gpus.empty()) {
            return Status(ErrorCode::invalid_argument,
                          "tensor_split requires at least one discrete GPU when no device allowlist is set");
        }
        return Status::success();
    }

    if (gpus.empty()) {
        return Status(ErrorCode::not_found,
                      "requested device '" + std::string(device_id) +
                          "' has no available discrete llama.cpp GPU (integrated GPUs require an explicit devices "
                          "allowlist)");
    }

    // When OS GPU inventory is supplied, filter to discrete OS GPUs (excluding
    // integrated UMA adapters) and match `gpu:<idx>` against the de-duplicated
    // discrete GGML devices by PCI bus id, adapter name, or intra-vendor ordinal.
    if (!os_gpus.empty()) {
        std::vector<const GpuDeviceInfo*> os_discrete;
        os_discrete.reserve(os_gpus.size());
        for (const auto& g : os_gpus) {
            if (!g.is_integrated_uma) os_discrete.push_back(&g);
        }
        if (idx >= os_discrete.size()) {
            return Status(ErrorCode::not_found, "requested device '" + std::string(device_id) +
                                                    "' exceeds detected discrete OS GPU count (" +
                                                    std::to_string(os_discrete.size()) + ")");
        }
        const GpuDeviceInfo& target = *os_discrete[idx];
        const std::string target_bus = lc::NormalizePciBusId(target.pci_bus_id);
        if (!target_bus.empty()) {
            for (const auto* g : gpus) {
                if (!g->device_id.empty() && lc::NormalizePciBusId(g->device_id) == target_bus) {
                    out_devices = {g->name};
                    out_main_gpu = 0;
                    return Status::success();
                }
            }
        }
        const std::string target_name = LowerAscii(target.name);
        std::size_t vendor_ordinal = 0;
        for (std::size_t i = 0; i < idx; ++i) {
            if (os_discrete[i]->vendor == target.vendor) ++vendor_ordinal;
        }
        std::vector<const lc::DeviceInfo*> same_vendor;
        std::vector<const lc::DeviceInfo*> name_matches;
        for (const auto* g : gpus) {
            const GpuVendor gv = InferGgmlDeviceVendor(*g);
            if (target.vendor != GpuVendor::unknown && gv == target.vendor) {
                same_vendor.push_back(g);
            }
            const std::string gd = LowerAscii(g->description);
            if (!target_name.empty() && !gd.empty() &&
                (gd.find(target_name) != std::string::npos || target_name.find(gd) != std::string::npos)) {
                name_matches.push_back(g);
            }
        }
        if (name_matches.size() == 1) {
            out_devices = {name_matches[0]->name};
            out_main_gpu = 0;
            return Status::success();
        }
        if (!same_vendor.empty() && vendor_ordinal < same_vendor.size()) {
            out_devices = {same_vendor[vendor_ordinal]->name};
            out_main_gpu = 0;
            return Status::success();
        }
        return Status(ErrorCode::not_found,
                      "requested device '" + std::string(device_id) + "' (" + target.name +
                          ") has no matching compiled llama.cpp GPU backend device; specify an explicit "
                          "backend device name such as CUDA0, ROCm0, or Vulkan0");
    }

    if (idx >= gpus.size()) {
        return Status(ErrorCode::not_found, "requested device '" + std::string(device_id) +
                                                "' exceeds available discrete llama.cpp GPU count (" +
                                                std::to_string(gpus.size()) + ")");
    }
    out_devices = {gpus[idx]->name};
    out_main_gpu = 0;
    return Status::success();
}

// One snapshot from the in-process GPU memory probe used by LlamaCppCoreBackend.
struct ProcessGpuMemoryReading {
    std::string probe;   // e.g. "pdh", "amdgpu-fdinfo", "nvml", "ggml", "none"
    std::string status;  // "ok", "unsupported", or "error"
    std::string error;
    std::uint64_t dedicated_bytes = 0;
    std::uint64_t shared_bytes = 0;
};

using ProcessGpuMemoryProbe = std::function<ProcessGpuMemoryReading()>;

// Aggregates GGML GPU device memory usage as a cross-platform fallback when
// OS per-process GPU memory counters are unavailable.
ProcessGpuMemoryReading SummarizeGgmlDeviceMemory(
    const std::vector<sonder::backends::llamacpp::DeviceInfo>& devices,
    const std::vector<std::string>& configured_devices);

// Returns true when any active GGML GPU target is an integrated UMA device
// (shared/GTT allocations are expected rather than a PCIe VRAM spill).
bool HasIntegratedUmaTarget(
    const std::vector<sonder::backends::llamacpp::DeviceInfo>& devices,
    const std::vector<std::string>& configured_devices);

// Populates Sonder GenerateStats from a llama.cpp GenerateResult, keeping
// total prompt tokens (`stats.prompt_tokens = evaluated + cached`), cached
// tokens (`stats.cached_tokens = cached`), and uncached evaluated prompt tokens
// (`stats.backend_timings->prompt_n = evaluated`) strictly separated.
void PopulateGenerateStats(const sonder::backends::llamacpp::GenerateResult& result,
                           std::size_t restore_tokens, std::uint64_t load_ns,
                           GenerateStats& stats) noexcept;

struct DirectChatPromptPlan {
    Status status = Status::success();
    bool template_unsupported = false;
    std::vector<ChatPromptSegment> segments;
    std::vector<std::vector<std::uint8_t>> media_blobs;
    std::vector<sonder::backends::llamacpp::GenerateRequest::MediaPromptPart> media_prompt_parts;
};

// Normalizes chat messages (ordered `parts`, legacy `media`, assistant `tool_calls`,
// and `tool` result correlation metadata), renders template segments with G0028
// `parse_special` boundaries (with automatic fallback coercing `role == "tool"` onto
// `"user"` `<tool_response>` turns if a built-in template skips `tool` messages),
// applies optional `enable_thinking=false` scaffolding, and builds structured multimodal
// prompt parts matching `EvaluateMultimodalPrompt` execution order.
inline DirectChatPromptPlan BuildDirectChatPromptPlan(
    const ChatRequest& request,
    const std::function<Result<std::string>(const std::vector<sonder::backends::llamacpp::ChatTurn>&, bool)>&
        apply_template) {
    namespace lc = sonder::backends::llamacpp;
    DirectChatPromptPlan plan;
    if (Status st = validate_chat_messages(request.messages); !st.ok()) {
        plan.status = st;
        return plan;
    }

    std::vector<std::pair<std::string, std::size_t>> media_sentinels;
    bool has_tool_messages = false;
    bool needs_normalized = false;
    for (const auto& m : request.messages) {
        if (m.role == "tool") has_tool_messages = true;
        if (count_message_images(m) > 0 || !m.parts.empty() || !m.tool_calls.empty() ||
            (m.role == "tool" &&
             ((m.tool_call_id && !m.tool_call_id->empty()) || (m.name && !m.name->empty())))) {
            needs_normalized = true;
        }
    }

    auto prepare_messages = [&](bool coerce_tool_role_to_user,
                                std::vector<ChatMessage>& normalized_out) -> const std::vector<ChatMessage>* {
        plan.media_blobs.clear();
        media_sentinels.clear();
        if (!needs_normalized && !coerce_tool_role_to_user) {
            return &request.messages;
        }
        normalized_out = request.messages;
        std::uint64_t media_nonce = 0x51a7c9e2f03b8d11ULL;
        auto next_media_prefix = [&]() {
            char buf[36];
            std::snprintf(buf, sizeof(buf), "SNDRMM%016llxM", static_cast<unsigned long long>(media_nonce));
            media_nonce = media_nonce * 6364136223846793005ULL + 1442695040888963407ULL;
            return std::string(buf);
        };
        std::string sentinel_prefix = next_media_prefix();
        auto prefix_collides = [&](std::string_view pfx) {
            for (const auto& m : normalized_out) {
                if (m.content.find(pfx) != std::string::npos || m.role.find(pfx) != std::string::npos) return true;
                if (m.tool_call_id && m.tool_call_id->find(pfx) != std::string::npos) return true;
                if (m.name && m.name->find(pfx) != std::string::npos) return true;
                for (const auto& part : m.parts) {
                    if (const auto* tp = std::get_if<TextContentPart>(&part)) {
                        if (tp->text.find(pfx) != std::string::npos) return true;
                    }
                }
                for (const auto& tc : m.tool_calls) {
                    if (tc.id.find(pfx) != std::string::npos || tc.name.find(pfx) != std::string::npos ||
                        tc.arguments_json.find(pfx) != std::string::npos) {
                        return true;
                    }
                }
            }
            return false;
        };
        while (prefix_collides(sentinel_prefix)) {
            sentinel_prefix = next_media_prefix();
        }
        for (auto& m : normalized_out) {
            if (m.role == "tool") {
                m.content = FormatLlamaCppToolResultContent(m, coerce_tool_role_to_user);
                if (coerce_tool_role_to_user) {
                    m.role = "user";
                }
            }
            if (!m.parts.empty()) {
                std::string combined;
                for (const auto& part : m.parts) {
                    if (const auto* tp = std::get_if<TextContentPart>(&part)) {
                        combined += tp->text;
                    } else if (const auto* ip = std::get_if<ImageContentPart>(&part)) {
                        const std::size_t blob_idx = plan.media_blobs.size();
                        plan.media_blobs.push_back(ip->media.bytes);
                        std::string tag = sentinel_prefix + "_" + std::to_string(blob_idx) + "_END";
                        media_sentinels.emplace_back(tag, blob_idx);
                        combined += tag;
                    }
                }
                m.content = std::move(combined);
            } else if (!m.media.empty()) {
                std::string markers;
                for (const auto& blob : m.media) {
                    const std::size_t blob_idx = plan.media_blobs.size();
                    plan.media_blobs.push_back(blob.bytes);
                    std::string tag = sentinel_prefix + "_" + std::to_string(blob_idx) + "_END";
                    media_sentinels.emplace_back(tag, blob_idx);
                    markers += tag;
                }
                m.content = markers + m.content;
            }
            if (m.role == "assistant" && !m.tool_calls.empty()) {
                std::string tool_text;
                for (const auto& tc : m.tool_calls) {
                    if (!tool_text.empty()) tool_text.push_back('\n');
                    tool_text += FormatLlamaCppToolCallMarkup(tc);
                }
                if (!m.content.empty()) {
                    m.content += "\n" + tool_text;
                } else {
                    m.content = std::move(tool_text);
                }
            }
        }
        return &normalized_out;
    };

    std::vector<ChatMessage> normalized_messages;
    const std::vector<ChatMessage>* template_messages =
        prepare_messages(/*coerce_tool_role_to_user=*/false, normalized_messages);
    auto render_cb = [&](const std::vector<ChatMessage>& placeholder_msgs) -> Result<std::string> {
        std::vector<lc::ChatTurn> turns;
        turns.reserve(placeholder_msgs.size());
        for (const auto& m : placeholder_msgs) turns.push_back(lc::ChatTurn{m.role, m.content});
        auto rendered = apply_template ? apply_template(turns, /*add_assistant=*/true)
                                       : Result<std::string>(Status(ErrorCode::invalid_argument,
                                                                    "chat: template callback is empty"));
        if (!rendered.ok()) {
            plan.template_unsupported = true;
            return rendered.status();
        }
        return rendered;
    };
    auto segments = build_chat_template_segments(*template_messages, render_cb);
    if (!segments.ok() && !plan.template_unsupported && has_tool_messages) {
        template_messages = prepare_messages(/*coerce_tool_role_to_user=*/true, normalized_messages);
        segments = build_chat_template_segments(*template_messages, render_cb);
    }
    if (!segments.ok()) {
        plan.status = segments.status();
        return plan;
    }

    plan.segments = std::move(segments.value());
    if (request.thinking.enable_thinking.has_value() && !*request.thinking.enable_thinking) {
        ApplyNoThinkScaffolding(plan.segments, /*enable_thinking=*/false);
    }

    // Build G0028-safe structured multimodal parts preserving per-segment
    // parse_special boundaries: trusted template scaffolding stays
    // parse_special=true, untrusted message text stays parse_special=false,
    // and media sentinels become explicit bitmap parts (zero marker parsing).
    if (!plan.media_blobs.empty()) {
        for (const auto& seg : plan.segments) {
            if (seg.parse_special) {
                if (!seg.text.empty()) {
                    plan.media_prompt_parts.push_back(
                        lc::GenerateRequest::MediaPromptPart{seg.text, /*parse_special=*/true, std::nullopt});
                }
                continue;
            }
            std::size_t cursor = 0;
            while (cursor < seg.text.size()) {
                std::size_t best_pos = std::string::npos;
                std::size_t best_len = 0;
                std::size_t best_blob = 0;
                for (const auto& [tag, bidx] : media_sentinels) {
                    const std::size_t p = seg.text.find(tag, cursor);
                    if (p != std::string::npos && p < best_pos) {
                        best_pos = p;
                        best_len = tag.size();
                        best_blob = bidx;
                    }
                }
                if (best_pos == std::string::npos) {
                    plan.media_prompt_parts.push_back(lc::GenerateRequest::MediaPromptPart{
                        seg.text.substr(cursor), /*parse_special=*/false, std::nullopt});
                    break;
                }
                if (best_pos > cursor) {
                    plan.media_prompt_parts.push_back(lc::GenerateRequest::MediaPromptPart{
                        seg.text.substr(cursor, best_pos - cursor), /*parse_special=*/false, std::nullopt});
                }
                plan.media_prompt_parts.push_back(
                    lc::GenerateRequest::MediaPromptPart{{}, /*parse_special=*/false, best_blob});
                cursor = best_pos + best_len;
            }
        }
    }
    return plan;
}

// Deterministic test factory allowing unit tests to inject a fake per-process
// GPU memory probe and refresh cadence without live GPU counters or weights.
std::shared_ptr<Backend> MakeLlamaCppBackendForTest(
    LlamaCppBackendOptions options, ProcessGpuMemoryProbe probe,
    std::chrono::milliseconds sample_interval = std::chrono::milliseconds{0},
    bool force_uma_mode = false, std::optional<bool> force_gpu_offload = std::nullopt);

}  // namespace sonder::inference::llamacpp_detail
