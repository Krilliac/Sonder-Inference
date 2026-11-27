# External process backend: uses only Sonder's HTTP/TLS client at runtime.
# No llama.cpp sources, common library, executable or model downloads.
if(NOT SONDER_BUILD_LLAMASERVER)
    return()
endif()

sonder_module_sources(
    adaptive_speculation.cpp
    child_metrics.cpp
    spill_guard.cpp
    gpu_residency.cpp
    host_memory.cpp
    llamaserver_backend.cpp
    llamaserver_options.cpp
    llamaserver_protocol.cpp
    llamaserver_stream.cpp
    log_diagnostics.cpp
    process.cpp
    process_options.cpp
    prefix_warmup.cpp
    slot_affinity.cpp
    supervisor.cpp
    supervisor_host_memory.cpp
    supervisor_residency.cpp
    timings.cpp
    tune.cpp
    tune_main.cpp
    tune_model.cpp
    tune_runner.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_LLAMASERVER_BACKEND)

add_subdirectory(tests)
