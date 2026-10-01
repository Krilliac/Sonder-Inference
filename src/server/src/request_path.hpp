// Internal: request-path options of `sonder-infer serve` (docs/SERVER.md
// "Scheduling", "Prompt cache affinity" and "Thinking control"): the
// --scheduler / --kv-pool-tokens / --pin-* flags, their mapping onto the
// engine, the conversation key and the thinking pins. Pure functions.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "openai.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

// Flags parsed by parse_request_path_flags() (all take a value).
inline constexpr const char* kRequestPathFlags[] = {"scheduler", "kv-pool-tokens", "pin-enable-thinking",
                                                    "pin-reasoning-effort", "pin-mode", "pin-reasoning-budget",
                                                    "pin-reasoning-budget-message", "max-concurrent-subagent",
                                                    "max-concurrent-background", "max-queue-per-class",
                                                    "backend-capacity", "priority-admission"};

// Usage lines for the flags above.
inline constexpr const char* kRequestPathUsage =
    "Request path:\n"
    "  --scheduler MODE        automatic (default: per-token gating for in-process\n"
    "                          backends, admission + accounting only for remote ones\n"
    "                          such as llamaserver and ollama), gate, account or off\n"
    "  --kv-pool-tokens N      logical KV pool size in tokens (default 65536)\n"
    "  --pin-enable-thinking on|off  send chat_template_kwargs.enable_thinking (llama-server)\n"
    "                          / think (ollama) on every chat request; a request that\n"
    "                          asks otherwise is overridden and warned (sonder.warnings)\n"
    "  --pin-reasoning-effort E  same for chat_template_kwargs.reasoning_effort\n"
    "  --pin-mode override|default  override request thinking values (default), or\n"
    "                          only fill values the request leaves unset\n"
    "  --pin-reasoning-budget N  default-only llama-server reasoning budget (-1 allowed)\n"
    "  --pin-reasoning-budget-message TEXT  default-only budget guidance (max 512 bytes)\n"
    "  --max-concurrent-subagent N  cap running subagent requests (0 = unlimited)\n"
    "  --max-concurrent-background N  cap running background requests (0 = unlimited)\n"
    "  --max-queue-per-class N  cap queued requests in each class (0 = unlimited)\n"
    "  --backend-capacity N  explicit request capacity when admission is enabled\n"
    "                          (0 = backend-reported; unknown stays ungated)\n"
    "  --priority-admission auto|on|off  class/capacity admission policy (default auto)\n"
    "                          auto enables only with a nonzero class or queue cap\n"
    "                          on enables explicitly; off bypasses all admission caps\n";

using FlagLookup = std::function<std::optional<std::string>(const std::string& name)>;

// Reads the flags above into `o`. invalid_argument naming the flag.
Status parse_request_path_flags(const FlagLookup& get, ServerOptions& o);

// Option checks for validate_options(): kv_pool_tokens range, pin values.
Status validate_request_path_options(const ServerOptions& o);

// --scheduler and --kv-pool-tokens onto the engine's scheduling options.
void apply_scheduling(const ServerOptions& o, SchedulingOptions& scheduling);

// Whether the class admission layer is enabled before backend discovery.
// Automatic mode is deliberately opt-in through explicit class/queue caps.
[[nodiscard]] bool priority_admission_enabled(const ServerOptions& o);

// Conversation key for backend cache affinity: prompt_cache_key when given,
// else "run=<X-Sonder-Run-Id>;agent=<X-Sonder-Agent-Id>" (either part may
// be absent), else empty (no affinity).
std::string chat_session_key(const ChatJob& job, const Correlation& correlation);

// Applies the server-wide thinking pins using ServerOptions::pin_mode.
std::vector<std::string> apply_thinking_pins(const ServerOptions& o, ThinkingOptions& thinking);

}  // namespace sonder::inference::server::detail
