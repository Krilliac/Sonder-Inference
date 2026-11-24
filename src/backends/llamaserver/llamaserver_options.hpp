// Internal: LlamaServerBackendOptions validation, supervisor option mapping,
// and bearer API-key loading from environment variables.
#pragma once

#include <string>

#include "sonder/inference/backends/llamaserver.hpp"
#include "supervisor.hpp"

namespace sonder::inference::llamaserver {

SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o);

// False when the policy name is unknown; `out` is then left partly filled.
bool to_host_memory_options(const LlamaServerHostMemoryOptions &o, HostMemoryOptions &out);

// Reads the bearer token from the environment variable `name`. Messages name
// the variable, never the value. Refuses an unset/empty variable, a malformed
// variable name, and a value that is not a single visible-ASCII token.
Result<std::string> read_api_key(const std::string &name);

Status validate_backend_options(const LlamaServerBackendOptions &options);

}  // namespace sonder::inference::llamaserver
