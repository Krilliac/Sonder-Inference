# Module: Ollama compatibility backend (owner: feat/ollama-bench).
# Sources compile into sonder_inference; transport is the core internal HTTP
# client (src/net). No new runtime dependency.
sonder_module_sources(
    ollama_client.cpp
    ollama_protocol.cpp
    ollama_backend.cpp
    ollama_telemetry.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_OLLAMA_BACKEND)

add_subdirectory(tests)
