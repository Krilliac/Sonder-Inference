# External process backend: uses only Sonder's HTTP/TLS client at runtime.
# No llama.cpp sources, common library, executable or model downloads.
sonder_module_sources(
    adaptive_speculation.cpp
    child_metrics.cpp
    gpu_memory.cpp
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
    rpc_worker.cpp
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
if(WIN32)
    # Per-process GPU memory counters (GPU Process Memory) for the spill guard.
    target_link_libraries(sonder_inference PRIVATE pdh)
endif()

add_subdirectory(tests)
