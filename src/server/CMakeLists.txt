# Module: `sonder-infer serve` HTTP server (owner: eco/inf-serve, ADR-020).
#
# In-house HTTP/1.1 server over POSIX sockets / Winsock (no third-party
# code): OpenAI-compatible chat subset, health/models/identity, and the live
# Observatory telemetry streams (SSE and NDJSON). Reference: docs/SERVER.md.
# Sources compile into sonder_inference; consumers test SONDER_HAS_SERVER.
sonder_module_sources(
    src/backend_setup.cpp
    src/http.cpp
    src/identity.cpp
    src/live_hub.cpp
    src/openai.cpp
    src/request_path.cpp
    src/serve_main.cpp
    src/server.cpp
    src/sha256.cpp
    src/socket.cpp
    src/warmup_config.cpp)
sonder_module_include_directories(include)
sonder_module_define(SONDER_HAS_SERVER)

sonder_add_module_tests(server
    tests/test_server_units.cpp
    tests/test_server_http.cpp
    tests/test_server_chat.cpp
    tests/test_server_telemetry.cpp
    tests/test_server_lifecycle.cpp
    tests/test_serve_main.cpp
    tests/test_server_faults.cpp
    tests/test_server_runtime.cpp
    tests/test_server_request_path.cpp)
if(TARGET sonder_server_tests)
    target_include_directories(sonder_server_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
