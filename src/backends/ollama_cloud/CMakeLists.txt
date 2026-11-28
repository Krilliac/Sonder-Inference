# Module: Ollama Cloud provider (owner: claude/ret-02-ollama-cloud).
# Sources compile into sonder_inference. Transport is the core internal HTTP
# client (src/net) over the optional TLS layer (cmake/SonderTls.cmake); no new
# dependency. Self-contained: does not use the local ollama backend module.
sonder_module_sources(
    ollama_cloud_secret.cpp
    ollama_cloud_protocol.cpp
    ollama_cloud_backend.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_OLLAMA_CLOUD_BACKEND)

add_subdirectory(tests)
