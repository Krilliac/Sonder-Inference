# External process backend: uses only Sonder's HTTP/TLS client at runtime.
# No llama.cpp sources, common library, executable or model downloads.
sonder_module_sources(
    gpu_memory.cpp
    gpu_residency.cpp
    llamaserver_backend.cpp
    llamaserver_protocol.cpp
    log_diagnostics.cpp
    process.cpp
    prefix_warmup.cpp
    slot_affinity.cpp
    supervisor.cpp
    supervisor_residency.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_LLAMASERVER_BACKEND)
if(WIN32)
    # Per-process GPU memory counters (GPU Process Memory) for the spill guard.
    target_link_libraries(sonder_inference PRIVATE pdh)
endif()

add_subdirectory(tests)
