# Module: `sonder-infer serve` HTTP server (owner: eco/inf-serve, ADR-020).
#
# In-house HTTP/1.1 server over POSIX sockets / Winsock (no third-party
# code): OpenAI-compatible chat subset, health/models/identity, and the live
# Observatory telemetry streams (SSE and NDJSON). Reference: docs/SERVER.md.
# Sources compile into sonder_inference; consumers test SONDER_HAS_SERVER.
sonder_module_sources(
    src/anthropic.cpp
    src/backend_setup.cpp
    src/chat_handler.cpp
    src/http.cpp
    src/health_features.cpp
    src/identity.cpp
    src/launch_profile.cpp
    src/live_hub.cpp
    src/openai.cpp
    src/priority_admission.cpp
    src/residency.cpp
    src/request_path.cpp
    src/reasoning_budget.cpp
    src/thinking_pins.cpp
    src/residency_config.cpp
    src/serve_main.cpp
    src/server.cpp
    src/sha256.cpp
    src/socket.cpp
    src/vram_estimate.cpp
    src/warmup_config.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_SERVER)

sonder_add_module_tests(server
    tests/test_server_anthropic_units.cpp
    tests/test_server_anthropic_http.cpp
    tests/test_server_units.cpp
    tests/test_server_http.cpp
    tests/test_server_chat.cpp
    tests/test_server_telemetry.cpp
    tests/test_server_lifecycle.cpp
    tests/test_serve_main.cpp
    tests/test_server_faults.cpp
    tests/test_server_residency.cpp
    tests/test_server_runtime.cpp
    tests/test_residency_config.cpp
    tests/test_server_request_path.cpp
    tests/test_priority_parsing.cpp
    tests/test_priority_admission.cpp
    tests/test_server_priority.cpp
    tests/test_launch_profile.cpp
    tests/test_gguf_artifact_observation.cpp)
if(TARGET sonder_server_tests)
    target_include_directories(sonder_server_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # GCC 13 raises a false -Wmaybe-uninitialized (fatal under -Werror) from <string> when a
    # test sets a default-constructed std::optional<std::string> member of ServerOptions;
    # a function-scope pragma does not reach the inlined library frame. GCC only, this file only.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_source_files_properties(tests/test_server_request_path.cpp
            PROPERTIES COMPILE_OPTIONS "-Wno-maybe-uninitialized")
    endif()
endif()
