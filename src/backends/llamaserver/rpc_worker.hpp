// Internal: isolated llama.cpp RPC worker policy (docs/integration/rpc-worker.md).
//
// llama.cpp's RPC client aborts its whole process when a peer fails the
// handshake or drops the connection (GGML_ABORT in ggml-rpc.cpp), and every
// tensor the peer returns is parsed in that process. So the client must never
// run in the Sonder process or in a child that serves another model. These
// helpers are the single place that decides whether `--rpc` may reach a
// llama-server argv: operator args may never carry it, and only the validated
// rpc_worker server list is turned into the `--rpc` the supervisor prepends.
//
// All functions are pure and thread-safe (no shared state); they allocate only
// for the returned strings and error messages.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

// True when llama-server would read `arg` as the "--" flag `flag` (given in
// its '-' spelling): the flag itself, "FLAG=VALUE", and any spelling whose '_'
// llama-server turns into '-'.
[[nodiscard]] bool matches_llama_server_flag(std::string_view arg, std::string_view flag) noexcept;

// True when llama-server would read `arg` as its `--rpc` flag.
[[nodiscard]] bool is_rpc_flag(std::string_view arg) noexcept;

// True when llama-server would read `arg` as `--models-preset`. A preset INI
// sets llama-server options by key (an "rpc" key reaches the --rpc handler),
// so it is a second way to put the RPC client into a spawned child.
[[nodiscard]] bool is_models_preset_flag(std::string_view arg) noexcept;

// Refuses every `--rpc` and `--models-preset` spelling in operator spawn
// arguments. The message points at rpc_worker; it never echoes the value.
[[nodiscard]] Status refuse_operator_rpc_arguments(const std::vector<std::string> &arguments);

// One "A.B.C.D:PORT" entry: dotted-decimal IPv4 with no leading zeros, port
// 1..65535 in plain decimal. Refuses names, IPv6, the unspecified, broadcast,
// multicast and reserved ranges, and (unless allow_public) any address outside
// loopback, RFC 1918, link-local and RFC 6598 shared space. Bounded message.
[[nodiscard]] Status validate_rpc_server(std::string_view server, bool allow_public);

// Whole rpc_worker block for the given backend mode. Disabled options are
// valid only when they carry no servers, so a half-written config is refused
// instead of being silently ignored.
[[nodiscard]] Status validate_rpc_worker(const LlamaServerRpcWorkerOptions &options, LlamaServerMode mode);

// The child argv: `--rpc s1,s2,...` first (llama.cpp registers RPC devices
// while parsing, so a later --device naming one must follow it), then
// `arguments` unchanged. Returns `arguments` unchanged when disabled.
// Precondition: validate_rpc_worker(options, spawn) succeeded.
[[nodiscard]] std::vector<std::string> rpc_worker_arguments(const LlamaServerRpcWorkerOptions &options,
                                                            const std::vector<std::string> &arguments);

} // namespace sonder::inference::llamaserver
