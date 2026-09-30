# External process backend: uses only Sonder's HTTP/TLS client at runtime.
# No llama.cpp sources, common library, executable or model downloads.
sonder_module_sources(
    llamaserver_backend.cpp
    llamaserver_protocol.cpp
    process.cpp
    supervisor.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_LLAMASERVER_BACKEND)

add_subdirectory(tests)
