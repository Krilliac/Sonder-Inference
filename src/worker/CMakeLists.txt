# ARCH-01: private local worker transport, process containment and proxy backend.
# No dependency on the retiring llama-server backend or new third-party code.
sonder_module_sources(ipc.cpp process.cpp protocol.cpp remote_backend.cpp worker_main.cpp)
sonder_module_define(SONDER_HAS_WORKER)
if(WIN32)
    target_link_libraries(sonder_inference PRIVATE bcrypt)
endif()

if(SONDER_BUILD_TESTS AND SONDER_BUILD_CLI)
    sonder_module_define(SONDER_WORKER_TESTING)
    add_executable(sonder_worker_adversary tests/adversarial_worker.cpp)
    target_link_libraries(sonder_worker_adversary PRIVATE sonder::inference)
    target_include_directories(sonder_worker_adversary PRIVATE "${PROJECT_SOURCE_DIR}/src")
    sonder_set_warnings(sonder_worker_adversary)
    add_executable(sonder_worker_environment_probe tests/process_environment_probe.cpp)
    target_compile_features(sonder_worker_environment_probe PRIVATE cxx_std_20)
    sonder_set_warnings(sonder_worker_environment_probe)
    add_executable(sonder_worker_parent_probe tests/parent_death_probe.cpp)
    target_link_libraries(sonder_worker_parent_probe PRIVATE sonder::inference)
    target_include_directories(sonder_worker_parent_probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    sonder_set_warnings(sonder_worker_parent_probe)
    sonder_add_module_tests(worker tests/test_worker.cpp tests/test_worker_limits.cpp tests/test_worker_reap.cpp)
    target_include_directories(sonder_worker_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/server"
        "${PROJECT_SOURCE_DIR}/src/server/tests")
    target_compile_definitions(sonder_worker_tests PRIVATE
        SONDER_WORKER_EXECUTABLE="$<TARGET_FILE:sonder-infer>"
        SONDER_WORKER_ENV_PROBE="$<TARGET_FILE:sonder_worker_environment_probe>"
        SONDER_WORKER_ADVERSARY="$<TARGET_FILE:sonder_worker_adversary>"
        SONDER_WORKER_PARENT_PROBE="$<TARGET_FILE:sonder_worker_parent_probe>")
    add_dependencies(sonder_worker_tests sonder-infer sonder_worker_environment_probe sonder_worker_parent_probe sonder_worker_adversary)
    add_test(NAME sonder.worker.cli_synthetic_and_default_model
        COMMAND "${CMAKE_COMMAND}" -DWORKER_EXECUTABLE=$<TARGET_FILE:sonder-infer>
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_worker_cli.cmake")
    set_tests_properties(sonder.worker.cli_synthetic_and_default_model PROPERTIES TIMEOUT 90)
endif()
